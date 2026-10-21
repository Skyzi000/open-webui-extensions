from types import SimpleNamespace

import pytest
from pydantic import JsonValue

from functions.pipe import auto_compact as mod


@pytest.fixture
async def real_checkpoint_store(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/ref-exec-checkpoints.db",
        poolclass=NullPool,
    )
    monkeypatch.setattr(mod, "_CHECKPOINT_SCHEMA_READY", False)
    await mod.ensure_checkpoint_table_initialized(async_engine=engine)
    sessionmaker = async_sessionmaker(bind=engine, expire_on_commit=False)
    state = SimpleNamespace(
        sessionmaker=sessionmaker,
        cas_checkpoint_ids=[],
        healed_checkpoint_ids=[],
        force_cas_winner_lookup=False,
    )

    class EngineCheckpointStore(mod.CheckpointStore):
        async def _context(self):
            return sessionmaker()

        async def compare_and_swap_history_ref(
            self,
            checkpoint_id: str,
            *,
            expected_summary_meta: dict[str, JsonValue],
            history_ref: dict[str, str],
        ) -> bool:
            state.cas_checkpoint_ids.append(checkpoint_id)
            updated = await super().compare_and_swap_history_ref(
                checkpoint_id,
                expected_summary_meta=expected_summary_meta,
                history_ref=history_ref,
            )
            if updated and state.force_cas_winner_lookup:
                state.force_cas_winner_lookup = False
                return False
            return updated

        async def lookup_ready_by_id(
            self,
            checkpoint_id: str,
            *,
            namespace: str,
            user_id: str,
            chat_id: str,
            pipe_function_id: str,
            profile_hash: str,
        ) -> dict[str, JsonValue] | None:
            state.healed_checkpoint_ids.append(checkpoint_id)
            return await super().lookup_ready_by_id(
                checkpoint_id,
                namespace=namespace,
                user_id=user_id,
                chat_id=chat_id,
                pipe_function_id=pipe_function_id,
                profile_hash=profile_hash,
            )

    async def checkpoint_schema_ready(**_kwargs) -> None:
        return None

    monkeypatch.setattr(
        mod, "ensure_checkpoint_table_initialized", checkpoint_schema_ready
    )
    monkeypatch.setattr(mod, "CheckpointStore", EngineCheckpointStore)
    try:
        yield state
    finally:
        await engine.dispose()


@pytest.fixture
def create_real_checkpoint():
    async def create(
        request: SimpleNamespace,
        source_messages,
        summary_text: str,
    ) -> dict[str, JsonValue]:
        async def summary_factory(_parent: dict[str, JsonValue] | None) -> str:
            return summary_text

        result = await mod._get_or_create_checkpoint_summary(
            request=request,
            user_id="user-1",
            chat_id="chat-1",
            pipe_function_id="auto_compact",
            source_messages=source_messages,
            summary_meta=mod.build_checkpoint_summary_meta(
                source_messages,
                historical_message_excerpt_bytes=(
                    mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES
                ),
                historical_message_excerpt_count=(
                    mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT
                ),
            ),
            summary_factory=summary_factory,
            use_generation_lease=False,
        )
        assert isinstance(result.checkpoint, dict)
        return result.checkpoint

    return create
