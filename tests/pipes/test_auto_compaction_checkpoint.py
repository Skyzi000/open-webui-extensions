from __future__ import annotations

import asyncio
import copy
import json
import time
from types import SimpleNamespace

import pytest

from functions.pipe import auto_compaction_pipe as mod


def test_source_hash_is_deterministic_for_equivalent_canonical_payloads():
    left = [
        {
            "id": "volatile-a",
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "timestamp": 100,
            "usage": {"total_tokens": 1},
        }
    ]
    right = [
        {
            "id": "volatile-b",
            "timestamp": 200,
            "role": "user",
            "usage": {"total_tokens": 999},
            "content": [{"text": "hello", "type": "text"}],
        }
    ]

    assert mod.compute_source_hash(left) == mod.compute_source_hash(right)


def test_source_hash_ignores_core_file_upload_transients():
    stable = {
        "role": "user",
        "content": [
            {"type": "text", "text": "inspect this file"},
            {
                "type": "file",
                "file": {
                    "id": "file-1",
                    "name": "report.pdf",
                    "signed_url": "https://signed.example/a?token=old",
                    "metadata": {"upload_id": "tmp-a", "sha256": "abc123"},
                },
            },
        ],
        "files": [
            {
                "type": "file",
                "id": "file-1",
                "url": "file-1",
                "name": "report.pdf",
                "collection_name": "knowledge",
                "content_type": "application/pdf",
                "size": 12345,
                "status": "uploaded",
                "error": "",
                "itemId": "ui-item-a",
                "file": {
                    "id": "file-1",
                    "user_id": "user-1",
                    "hash": "abc123",
                    "filename": "report.pdf",
                    "path": "/var/lib/open-webui/uploads/a/report.pdf",
                    "data": {"content": "extracted text", "status": "pending"},
                    "meta": {
                        "name": "report.pdf",
                        "content_type": "application/pdf",
                        "collection_name": "knowledge",
                        "data": {"upload_id": "tmp-a"},
                    },
                    "created_at": 100,
                    "updated_at": 100,
                },
            }
        ],
    }
    changed_transient = copy.deepcopy(stable)
    changed_transient["content"][1]["file"]["signed_url"] = "https://signed.example/a?token=new"
    changed_transient["content"][1]["file"]["metadata"]["upload_id"] = "tmp-b"
    changed_transient["files"][0]["status"] = "processing"
    changed_transient["files"][0]["error"] = "temporary warning"
    changed_transient["files"][0]["itemId"] = "ui-item-b"
    changed_transient["files"][0]["size"] = 99999
    changed_transient["files"][0]["progress"] = {"phase": "extracting", "percent": 50}
    changed_transient["files"][0]["file"]["path"] = "/var/lib/open-webui/uploads/b/report.pdf"
    changed_transient["files"][0]["file"]["data"]["status"] = "done"
    changed_transient["files"][0]["file"]["meta"]["data"]["upload_id"] = "tmp-b"
    changed_transient["files"][0]["file"]["created_at"] = 200
    changed_transient["files"][0]["file"]["updated_at"] = 200
    changed_semantic = copy.deepcopy(stable)
    changed_semantic["files"][0]["file"]["hash"] = "def456"

    assert mod.compute_source_hash([stable]) == mod.compute_source_hash([changed_transient])
    assert mod.compute_source_hash([stable]) != mod.compute_source_hash([changed_semantic])


def test_source_hash_preserves_file_metadata_used_for_source_context():
    first = {
        "role": "user",
        "content": "Use attached page",
        "files": [
            {
                "type": "text",
                "name": "https://example.com/a",
                "url": "https://example.com/a",
                "context": "full",
                "file": {
                    "data": {"content": "same text"},
                    "meta": {"name": "page", "source": "https://example.com/a"},
                },
            }
        ],
    }
    second = copy.deepcopy(first)
    second["files"][0]["file"]["meta"]["source"] = "https://example.com/b"

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_preserves_web_search_attachment_urls_and_queries():
    first = {
        "role": "user",
        "content": "Search the web",
        "files": [
            {
                "collection_name": "web-search-collection",
                "name": "open webui auto compaction",
                "type": "web_search",
                "urls": ["https://example.com/old"],
                "queries": ["open webui auto compaction"],
                "status": "uploaded",
            }
        ],
    }
    changed_url = copy.deepcopy(first)
    changed_url["files"][0]["urls"] = ["https://example.com/new"]
    changed_query = copy.deepcopy(first)
    changed_query["files"][0]["queries"] = ["open webui checkpoint compaction"]

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([changed_url])
    assert mod.compute_source_hash([first]) != mod.compute_source_hash([changed_query])
    assert "status" not in mod.canonicalize_messages_for_source_hash([first])[0]["files"][0]


def test_source_hash_preserves_non_file_content_file_paths():
    first = {
        "role": "user",
        "content": [{"type": "profile", "file": {"path": "/workspace/a.txt", "name": "same"}}],
    }
    second = copy.deepcopy(first)
    second["content"][0]["file"]["path"] = "/workspace/b.txt"

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_preserves_semantic_paths_outside_file_metadata():
    first = {
        "role": "assistant",
        "content": [{"type": "input", "path": "/workspace/a.txt"}],
        "function_call": {"name": "read_file", "arguments": {"path": "/workspace/a.txt"}},
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "/workspace/a.txt"}},
            }
        ],
        "sources": [{"source": {"id": "doc-1", "name": "docs", "path": "docs/a.md"}}],
    }
    second = copy.deepcopy(first)
    second["content"][0]["path"] = "/workspace/b.txt"
    second["function_call"] = {"name": "read_file", "arguments": {"path": "/workspace/b.txt"}}
    second["tool_calls"][0]["function"]["arguments"]["path"] = "/workspace/b.txt"
    second["sources"][0]["source"]["path"] = "docs/b.md"

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_preserves_semantic_urls():
    first = {
        "role": "user",
        "content": "Use this reference URL",
        "sources": [{"title": "docs", "url": "https://example.com/a"}],
    }
    second = {
        "role": "user",
        "content": "Use this reference URL",
        "sources": [{"title": "docs", "url": "https://example.com/b"}],
    }

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_preserves_semantic_download_urls_outside_file_metadata():
    first = {
        "role": "user",
        "content": "Use this reference URL",
        "sources": [{"source": {"name": "docs", "download_url": "https://example.com/a"}}],
    }
    second = {
        "role": "user",
        "content": "Use this reference URL",
        "sources": [{"source": {"name": "docs", "download_url": "https://example.com/b"}}],
    }

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_ignores_retrieval_source_scores():
    stable = {
        "role": "assistant",
        "content": "answer",
        "sources": [
            {
                "source": {"id": "doc-1", "name": "docs", "type": "file"},
                "document": ["same chunk"],
                "metadata": [{"source": "docs/a.md"}],
                "distances": [0.1],
            }
        ],
    }
    changed_score = copy.deepcopy(stable)
    changed_score["sources"][0]["distances"] = [0.9]
    changed_document = copy.deepcopy(stable)
    changed_document["sources"][0]["document"] = ["different chunk"]

    assert mod.compute_source_hash([stable]) == mod.compute_source_hash([changed_score])
    assert mod.compute_source_hash([stable]) != mod.compute_source_hash([changed_document])


def test_profile_hash_only_uses_hard_compatibility_boundaries():
    baseline = mod.compute_profile_hash(
        target_model="gpt-4.1",
        summary_model="gpt-4.1-mini",
        wrapper_suffix="abc",
        summary_prompt="old prompt",
        historical_message_excerpt_bytes=256,
        historical_message_excerpt_count=16,
    )
    ordinary_changes = mod.compute_profile_hash(
        target_model="claude-sonnet",
        summary_model="pipe.summary",
        wrapper_suffix="xyz",
        summary_prompt="new prompt",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=64,
    )
    hard_change = mod.compute_profile_hash(summary_format_family="changed-summary-format-family")

    assert ordinary_changes == baseline
    assert hard_change != baseline


def test_wrapper_ids_round_trip_special_target_model_ids():
    targets = [
        "openai.gpt-4.1",
        "ollama/llama3.2:latest",
        "provider:model/with.dots:and/slashes",
        "pipe.backed.target",
    ]

    for target in targets:
        wrapper_id = mod.build_wrapper_model_id("auto_compact", target)
        assert wrapper_id == f"auto_compact.{target}"
        assert wrapper_id.startswith("auto_compact.")
        decoded = mod.decode_wrapper_model_id(wrapper_id, expected_pipe_function_id="auto_compact")
        assert decoded.pipe_function_id == "auto_compact"
        assert decoded.target_model_id == target


def test_decode_rejects_malformed_or_wrong_pipe_wrapper_ids():
    with pytest.raises(ValueError, match="wrapper"):
        mod.decode_wrapper_model_id("not-a-wrapper", expected_pipe_function_id="auto_compact")

    with pytest.raises(ValueError, match="pipe"):
        mod.decode_wrapper_model_id("other_pipe.gpt", expected_pipe_function_id="auto_compact")


def test_checkpoint_table_contract_names_and_columns():
    table = mod.CHECKPOINT_TABLE

    assert table.name == "skyzi000_owui_ext_autocompact_checkpoint_v1"
    assert table.metadata.schema == mod.OPEN_WEBUI_DATABASE_SCHEMA
    assert {idx.name for idx in table.indexes} == {
        "skyzi000_owui_ext_accp_v1_prefix_idx",
        "skyzi000_owui_ext_accp_v1_recent_idx",
    }
    assert {constraint.name for constraint in table.constraints} >= {
        "skyzi000_owui_ext_accp_v1_lookup_uq"
    }
    assert set(table.c.keys()) == {
        "id",
        "namespace",
        "schema_version",
        "user_id",
        "chat_id",
        "pipe_function_id",
        "profile_hash",
        "source_message_count",
        "source_hash",
        "summary_text",
        "summary_meta",
        "summary_token_count",
        "state",
        "parent_checkpoint_id",
        "claim_token",
        "claim_expires_at",
        "created_at",
        "updated_at",
        "last_used_at",
    }


def test_checkpoint_lookup_indexes_partition_by_pipe_function_id_not_selected_wrapper_id():
    table = mod.CHECKPOINT_TABLE
    lookup = next(
        constraint
        for constraint in table.constraints
        if constraint.name == "skyzi000_owui_ext_accp_v1_lookup_uq"
    )
    prefix = next(idx for idx in table.indexes if idx.name == "skyzi000_owui_ext_accp_v1_prefix_idx")

    assert [column.name for column in lookup.columns] == [
        "namespace",
        "user_id",
        "chat_id",
        "pipe_function_id",
        "profile_hash",
        "source_hash",
    ]
    assert [column.name for column in prefix.columns] == [
        "namespace",
        "user_id",
        "chat_id",
        "pipe_function_id",
        "profile_hash",
        "source_message_count",
    ]


@pytest.mark.asyncio
async def test_checkpoint_table_initialization_is_cached_on_request_state(monkeypatch):
    calls = []

    class DummyConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def run_sync(self, fn):
            calls.append(fn)

    class DummyEngine:
        def begin(self):
            return DummyConnection()

    request = type("Request", (), {"state": type("State", (), {})()})()
    monkeypatch.setattr(mod, "_CHECKPOINT_SCHEMA_READY", False)

    await mod.ensure_checkpoint_table_initialized(request=request, async_engine=DummyEngine())
    await mod.ensure_checkpoint_table_initialized(request=request, async_engine=DummyEngine())

    assert len(calls) == 1
    assert request.state.__dict__[mod.REQUEST_STATE_SCHEMA_READY_KEY] is True


@pytest.mark.asyncio
async def test_checkpoint_table_initialization_uses_process_cache_without_request(monkeypatch):
    calls = []

    class DummyConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def run_sync(self, fn):
            calls.append(fn)

    class DummyEngine:
        def begin(self):
            return DummyConnection()

    monkeypatch.setattr(mod, "_CHECKPOINT_SCHEMA_READY", False)

    await mod.ensure_checkpoint_table_initialized(request=None, async_engine=DummyEngine())
    await mod.ensure_checkpoint_table_initialized(request=None, async_engine=DummyEngine())

    assert len(calls) == 1


def test_generation_lock_cleanup_removes_unused_lock():
    key = ("ns", "user", "chat", "pipe", "profile", "source")
    lock = mod.get_generation_lock(key)

    assert mod._GENERATION_LOCKS[key] is lock

    mod.release_generation_lock(key, lock)

    assert key not in mod._GENERATION_LOCKS


class ClaimStore:
    """In-memory CheckpointStore stand-in implementing the DB claim interface."""

    def __init__(self, rows=None):
        self.rows = [dict(row) for row in (rows or [])]
        self.claimed_rows = []
        self.completed_rows = []
        self.released = []
        self.touched = []

    def _match(self, source_hash):
        for row in self.rows:
            if row.get("source_hash") == source_hash:
                return row
        return None

    async def lookup_any(self, **kwargs):
        row = self._match(kwargs["source_hash"])
        return dict(row) if row else None

    async def lookup_ready(self, **kwargs):
        row = self._match(kwargs["source_hash"])
        if row is not None and row.get("state") == "ready":
            return dict(row)
        return None

    async def find_longest_parent(self, **kwargs):
        return mod.select_longest_matching_parent(self.rows, kwargs["source_messages"])

    async def claim_pending(self, row):
        if self._match(row["source_hash"]) is not None:
            return False
        stored = dict(row)
        self.rows.append(stored)
        self.claimed_rows.append(dict(stored))
        return True

    async def reclaim_pending(self, checkpoint_id, *, claim_token, expires_at, now=None):
        for row in self.rows:
            if row.get("id") != checkpoint_id or row.get("state") != "pending":
                continue
            expires = row.get("claim_expires_at")
            if expires is not None and now is not None and int(expires) > int(now):
                return False
            row["claim_token"] = claim_token
            row["claim_expires_at"] = expires_at
            return True
        return False

    async def extend_claim(self, checkpoint_id, *, claim_token, expires_at):
        for row in self.rows:
            if (
                row.get("id") == checkpoint_id
                and row.get("state") == "pending"
                and row.get("claim_token") == claim_token
            ):
                row["claim_expires_at"] = expires_at
                return True
        return False

    async def release_claim(self, checkpoint_id, *, claim_token):
        for row in list(self.rows):
            if (
                row.get("id") == checkpoint_id
                and row.get("state") == "pending"
                and row.get("claim_token") == claim_token
            ):
                self.rows.remove(row)
                self.released.append(checkpoint_id)
                return True
        return False

    async def complete_pending(
        self,
        checkpoint_id,
        *,
        claim_token,
        summary_text,
        parent_checkpoint_id,
        summary_token_count=None,
        now=None,
    ):
        for row in self.rows:
            if (
                row.get("id") == checkpoint_id
                and row.get("state") == "pending"
                and row.get("claim_token") == claim_token
            ):
                row.update(
                    state="ready",
                    summary_text=summary_text,
                    parent_checkpoint_id=parent_checkpoint_id,
                    summary_token_count=summary_token_count,
                    claim_token=None,
                    claim_expires_at=None,
                )
                self.completed_rows.append(dict(row))
                return dict(row)
        return None

    async def touch(self, checkpoint_id, *, now=None):
        self.touched.append(checkpoint_id)
        return True


def make_checkpoint_row(
    source_messages,
    *,
    state="pending",
    summary_text="",
    claim_token="claim-token-1",
    claim_expires_at=10**12,
    now=100,
):
    return mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(source_messages),
        source_message_count=len(source_messages),
        summary_text=summary_text,
        summary_meta={},
        parent_checkpoint_id=None,
        state=state,
        claim_token=claim_token,
        claim_expires_at=claim_expires_at,
        now=now,
    )


async def noop_initialize(**kwargs):
    return None


async def run_get_or_create(source_messages, summary_factory, *, summary_meta=None, parent_checkpoint=None):
    return await mod._get_or_create_checkpoint_summary(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        source_messages=source_messages,
        summary_meta=summary_meta or {},
        summary_factory=summary_factory,
        parent_checkpoint=parent_checkpoint,
    )


def test_checkpoint_row_uses_stable_identity_and_contract_fields():
    row = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=12,
        summary_text="summary",
        summary_meta={"has_multimodal": False},
        parent_checkpoint_id="parent-1",
        now=123,
    )
    same = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=12,
        summary_text="new summary",
        summary_meta={},
        parent_checkpoint_id=None,
        now=999,
    )
    explicit_count = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=12,
        summary_text="counted summary",
        summary_meta={},
        summary_token_count=42,
        parent_checkpoint_id=None,
        now=123,
    )

    assert same["id"] == row["id"]
    assert explicit_count["id"] == row["id"]
    assert explicit_count["summary_token_count"] == 42
    assert row == {
        "id": row["id"],
        "namespace": mod.CHECKPOINT_NAMESPACE,
        "schema_version": 1,
        "user_id": "user-1",
        "chat_id": "chat-1",
        "pipe_function_id": "auto_compact",
        "profile_hash": "profile",
        "source_message_count": 12,
        "source_hash": "source",
        "summary_text": "summary",
        "summary_meta": {"has_multimodal": False},
        "summary_token_count": None,
        "state": "ready",
        "parent_checkpoint_id": "parent-1",
        "claim_token": None,
        "claim_expires_at": None,
        "created_at": 123,
        "updated_at": 123,
        "last_used_at": 123,
    }


def test_checkpoint_row_supports_pending_claim_state():
    row = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=3,
        summary_text="",
        summary_meta={},
        parent_checkpoint_id=None,
        state="pending",
        claim_token="claim-1",
        claim_expires_at=456,
        now=123,
    )

    assert row["state"] == "pending"
    assert row["summary_text"] == ""
    assert row["claim_token"] == "claim-1"
    assert row["claim_expires_at"] == 456


@pytest.mark.asyncio
async def test_checkpoint_store_queries_filter_by_pipe_function_id():
    from sqlalchemy.dialects import sqlite

    statements = []

    class FakeMappings:
        def first(self):
            return None

        def all(self):
            return []

    class FakeResult:
        def mappings(self):
            return FakeMappings()

    class FakeDb:
        async def execute(self, statement):
            statements.append(str(statement.compile(dialect=sqlite.dialect())))
            return FakeResult()

    store = mod.CheckpointStore(db=FakeDb())

    await store.lookup_ready(
        namespace="auto_compact",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
    )
    await store.lookup_any(
        namespace="auto_compact",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
    )
    await store.find_longest_parent(
        namespace="auto_compact",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_messages=[{"role": "user", "content": "old"}],
    )

    assert statements
    assert all("pipe_function_id =" in statement for statement in statements)


@pytest.mark.asyncio
async def test_checkpoint_parent_query_filters_by_count_without_in_clause():
    from sqlalchemy.dialects import sqlite

    statements = []

    class FakeMappings:
        def all(self):
            return []

    class FakeResult:
        def mappings(self):
            return FakeMappings()

    class FakeDb:
        async def execute(self, statement):
            statements.append(str(statement.compile(dialect=sqlite.dialect())))
            return FakeResult()

    store = mod.CheckpointStore(db=FakeDb())

    await store.find_longest_parent(
        namespace="auto_compact",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_messages=[{"role": "user", "content": f"m{index}"} for index in range(50)],
    )

    assert len(statements) == 1
    assert "source_message_count <=" in statements[0]
    assert " IN " not in statements[0]


def test_parent_checkpoint_selection_requires_exact_source_hash():
    source_messages = [
        {"role": "user", "content": "m1"},
        {"role": "assistant", "content": "m2"},
        {"role": "user", "content": "m3"},
        {"role": "assistant", "content": "m4"},
    ]
    rows = [
        {
            "id": "short",
            "source_message_count": 2,
            "source_hash": mod.compute_source_hash(source_messages[:2]),
            "state": "ready",
        },
        {"id": "mismatch", "source_message_count": 4, "source_hash": "wrong", "state": "ready"},
        {
            "id": "long",
            "source_message_count": 3,
            "source_hash": mod.compute_source_hash(source_messages[:3]),
            "state": "ready",
        },
        {
            "id": "pending",
            "source_message_count": 4,
            "source_hash": mod.compute_source_hash(source_messages),
            "state": "pending",
        },
    ]

    parent = mod.select_longest_matching_parent(rows, source_messages)

    assert parent["id"] == "long"


def test_parent_checkpoint_selection_hashes_only_candidate_counts(monkeypatch):
    source_messages = [{"role": "user", "content": f"m{index}"} for index in range(6)]
    real_compute = mod.compute_source_hash
    hashed_counts = []

    def counting_compute(messages):
        hashed_counts.append(len(messages))
        return real_compute(messages)

    monkeypatch.setattr(mod, "compute_source_hash", counting_compute)
    rows = [
        {"id": "wrong-long", "source_message_count": 4, "source_hash": "wrong", "state": "ready"},
        {
            "id": "match",
            "source_message_count": 2,
            "source_hash": real_compute(source_messages[:2]),
            "state": "ready",
        },
        {"id": "wrong-long-too", "source_message_count": 4, "source_hash": "also-wrong", "state": "ready"},
    ]

    parent = mod.select_longest_matching_parent(rows, source_messages)

    assert parent["id"] == "match"
    assert sorted(set(hashed_counts)) == [2, 4]
    assert len(hashed_counts) == 2


@pytest.mark.asyncio
async def test_checkpoint_exact_hit_uses_ready_summary_without_claiming(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    existing = make_checkpoint_row(
        source_messages,
        state="ready",
        summary_text="existing summary",
        claim_token=None,
        claim_expires_at=None,
    )
    store = ClaimStore([existing])

    async def summary_factory(parent):
        raise AssertionError("summary factory must not run for an exact checkpoint hit")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    result = await run_get_or_create(source_messages, summary_factory)

    assert result == "existing summary"
    assert store.touched == [existing["id"]]
    assert store.claimed_rows == []


@pytest.mark.asyncio
async def test_checkpoint_exact_hit_survives_last_used_update_failure(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    existing = make_checkpoint_row(
        source_messages,
        state="ready",
        summary_text="existing summary",
        claim_token=None,
        claim_expires_at=None,
    )

    class FailingTouchStore(ClaimStore):
        async def touch(self, checkpoint_id, *, now=None):
            raise RuntimeError("touch failed")

    async def summary_factory(parent):
        raise AssertionError("summary factory must not run when touch fails")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: FailingTouchStore([existing]))

    result = await run_get_or_create(source_messages, summary_factory)

    assert result == "existing summary"


@pytest.mark.asyncio
async def test_checkpoint_miss_claims_generates_and_completes_ready_row(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    store = ClaimStore()
    calls = []

    async def summary_factory(parent):
        calls.append(parent)
        return "generated summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    result = await run_get_or_create(
        source_messages,
        summary_factory,
        summary_meta={"has_multimodal": False},
    )

    assert result == "generated summary"
    assert calls == [None]
    assert len(store.claimed_rows) == 1
    assert store.claimed_rows[0]["state"] == "pending"
    assert store.claimed_rows[0]["summary_text"] == ""
    assert store.claimed_rows[0]["claim_token"]
    assert store.claimed_rows[0]["claim_expires_at"] > int(time.time()) - 5
    assert store.claimed_rows[0]["source_hash"] == mod.compute_source_hash(source_messages)
    assert len(store.completed_rows) == 1
    assert store.completed_rows[0]["state"] == "ready"
    assert store.completed_rows[0]["summary_text"] == "generated summary"
    assert store.completed_rows[0]["parent_checkpoint_id"] is None
    assert result.checkpoint["state"] == "ready"


@pytest.mark.asyncio
async def test_pending_checkpoint_waiter_returns_ready_row_without_generating(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    pending = make_checkpoint_row(source_messages, claim_token="other-worker")
    ready = dict(
        pending,
        state="ready",
        summary_text="other worker summary",
        claim_token=None,
        claim_expires_at=None,
    )

    class EventualReadyStore(ClaimStore):
        def __init__(self):
            super().__init__([pending])
            self.lookup_calls = 0

        async def lookup_any(self, **kwargs):
            self.lookup_calls += 1
            if self.lookup_calls >= 3:
                return dict(ready)
            return dict(pending)

    store = EventualReadyStore()

    async def summary_factory(parent):
        raise AssertionError("waiters must not generate a summary while another worker owns the claim")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_POLL_SECONDS", 0.01)

    result = await run_get_or_create(source_messages, summary_factory)

    assert result == "other worker summary"
    assert store.touched == [ready["id"]]
    assert store.claimed_rows == []


@pytest.mark.asyncio
async def test_pending_checkpoint_wait_times_out_with_explicit_error(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    pending = make_checkpoint_row(source_messages, claim_token="other-worker")
    store = ClaimStore([pending])

    async def summary_factory(parent):
        raise AssertionError("waiters must not generate a summary on wait timeout")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_POLL_SECONDS", 0.01)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(RuntimeError, match="[Tt]imed out"):
        await run_get_or_create(source_messages, summary_factory)

    assert store.completed_rows == []


@pytest.mark.asyncio
async def test_stale_pending_claim_is_reclaimed_and_completed(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    stale = make_checkpoint_row(source_messages, claim_token="crashed-worker", claim_expires_at=1, now=1)
    store = ClaimStore([stale])
    calls = []

    async def summary_factory(parent):
        calls.append(parent)
        return "reclaimed summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    result = await run_get_or_create(source_messages, summary_factory)

    assert result == "reclaimed summary"
    assert calls == [None]
    assert len(store.completed_rows) == 1
    assert store.completed_rows[0]["summary_text"] == "reclaimed summary"
    assert store.rows[0]["state"] == "ready"
    assert store.rows[0]["claim_token"] is None


@pytest.mark.asyncio
async def test_lost_claim_falls_back_to_ready_row_from_other_worker(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    other_ready = make_checkpoint_row(
        source_messages,
        state="ready",
        summary_text="other summary",
        claim_token=None,
        claim_expires_at=None,
    )

    class LostClaimStore(ClaimStore):
        async def complete_pending(self, checkpoint_id, **kwargs):
            return None

        async def lookup_ready(self, **kwargs):
            return dict(other_ready)

    store = LostClaimStore()

    async def summary_factory(parent):
        return "my summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    result = await run_get_or_create(source_messages, summary_factory)

    assert result == "other summary"


@pytest.mark.asyncio
async def test_lost_claim_without_ready_row_surfaces_error(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]

    class LostClaimStore(ClaimStore):
        async def complete_pending(self, checkpoint_id, **kwargs):
            return None

    store = LostClaimStore()

    async def summary_factory(parent):
        return "my summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    with pytest.raises(RuntimeError, match="claim was lost"):
        await run_get_or_create(source_messages, summary_factory)


@pytest.mark.asyncio
async def test_checkpoint_summary_surfaces_claim_failure_before_generation(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    calls = []

    class FailingClaimStore(ClaimStore):
        async def claim_pending(self, row):
            raise RuntimeError("db down")

    async def summary_factory(parent):
        calls.append(parent)
        return "generated summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: FailingClaimStore())

    with pytest.raises(RuntimeError, match="db down"):
        await run_get_or_create(source_messages, summary_factory)

    assert calls == []


@pytest.mark.asyncio
async def test_checkpoint_does_not_store_incomplete_summary_and_releases_claim(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    store = ClaimStore()

    async def summary_factory(parent):
        assert parent is None
        return await mod.extract_text_from_completion_response(
            {"choices": [{"message": {"content": "partial summary"}, "finish_reason": "length"}]}
        )

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    with pytest.raises(RuntimeError, match="stopped before completing"):
        await run_get_or_create(source_messages, summary_factory)

    assert store.completed_rows == []
    assert store.released == [store.claimed_rows[0]["id"]]
    assert store.rows == []


@pytest.mark.asyncio
async def test_checkpoint_cancellation_releases_claim(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    store = ClaimStore()
    started = asyncio.Event()

    async def summary_factory(parent):
        assert parent is None
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    task = asyncio.create_task(run_get_or_create(source_messages, summary_factory))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert store.completed_rows == []
    assert store.released == [store.claimed_rows[0]["id"]]
    assert store.rows == []


@pytest.mark.asyncio
async def test_checkpoint_extension_uses_parent_summary_and_stores_lineage(monkeypatch):
    parent_source = [{"role": "user", "content": "old"}]
    parent = make_checkpoint_row(
        parent_source,
        state="ready",
        summary_text="parent summary",
        claim_token=None,
        claim_expires_at=None,
    )
    store = ClaimStore([parent])
    calls = []

    async def summary_factory(candidate_parent):
        calls.append(candidate_parent)
        return "extended summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "new"},
    ]
    result = await run_get_or_create(
        source_messages,
        summary_factory,
        summary_meta={"has_multimodal": False},
    )

    assert result == "extended summary"
    assert len(calls) == 1
    assert calls[0]["id"] == parent["id"]
    assert len(store.completed_rows) == 1
    assert store.completed_rows[0]["source_hash"] == mod.compute_source_hash(source_messages)
    assert store.completed_rows[0]["source_message_count"] == 2
    assert store.completed_rows[0]["parent_checkpoint_id"] == parent["id"]


@pytest.mark.asyncio
async def test_checkpoint_reuse_survives_target_and_summary_model_changes(monkeypatch):
    source_messages = [{"role": "user", "content": "old"}]
    existing = make_checkpoint_row(
        source_messages,
        state="ready",
        summary_text="existing summary",
        claim_token=None,
        claim_expires_at=None,
    )
    lookup_keys = []

    class RecordingClaimStore(ClaimStore):
        async def lookup_any(self, **kwargs):
            lookup_keys.append(kwargs)
            return await super().lookup_any(**kwargs)

    store = RecordingClaimStore([existing])

    async def summary_factory(parent):
        raise AssertionError("checkpoint reuse must not depend on current target or summary model")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    first = await run_get_or_create(source_messages, summary_factory)
    second = await run_get_or_create(
        source_messages,
        summary_factory,
        summary_meta={"summary_model": "changed", "target_model": "changed"},
    )

    assert first == "existing summary"
    assert second == "existing summary"
    assert lookup_keys[0]["profile_hash"] == lookup_keys[1]["profile_hash"]
    assert lookup_keys[0]["source_hash"] == lookup_keys[1]["source_hash"]


@pytest.mark.asyncio
async def test_parent_checkpoint_failure_releases_claim_and_does_not_resubmit_raw_prefix(monkeypatch):
    parent_source = [{"role": "user", "content": "old"}]
    parent = make_checkpoint_row(
        parent_source,
        state="ready",
        summary_text="parent summary",
        claim_token=None,
        claim_expires_at=None,
    )
    store = ClaimStore([parent])
    calls = []

    async def summary_factory(candidate_parent):
        calls.append(candidate_parent)
        raise RuntimeError("summary context overflow")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    with pytest.raises(mod.ParentCheckpointExtensionFailed) as exc_info:
        await run_get_or_create(
            [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "new"},
            ],
            summary_factory,
        )

    assert len(calls) == 1
    assert exc_info.value.parent["id"] == parent["id"]
    assert store.completed_rows == []
    assert store.released == [store.claimed_rows[0]["id"]]
    assert [row["id"] for row in store.rows] == [parent["id"]]


@pytest.mark.asyncio
async def test_checkpoint_store_claim_conflict_returns_false():
    from sqlalchemy.exc import IntegrityError

    class FakeDb:
        def __init__(self):
            self.commit_calls = 0
            self.rollback_calls = 0

        async def execute(self, statement):
            raise IntegrityError("insert", {}, Exception("duplicate"))

        async def commit(self):
            self.commit_calls += 1

        async def rollback(self):
            self.rollback_calls += 1

    db = FakeDb()
    store = mod.CheckpointStore(db=db)
    row = make_checkpoint_row([{"role": "user", "content": "old"}])

    assert await store.claim_pending(row) is False
    assert db.commit_calls == 0
    assert db.rollback_calls == 1


@pytest.mark.asyncio
async def test_compact_body_extends_from_parent_summary_plus_delta(monkeypatch):
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    parent = make_checkpoint_row(
        parent_source,
        state="ready",
        summary_text="parent summary",
        claim_token=None,
        claim_expires_at=None,
    )
    store = ClaimStore([parent])
    captured = {}

    async def generate_summary_text(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "extended summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "model": "target",
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "delta"},
            {"role": "assistant", "content": "delta answer"},
            {"role": "user", "content": "active"},
        ],
    }

    compacted, did_compact = await mod._compact_body(
        request=SimpleNamespace(state=SimpleNamespace()),
        user={"id": "user-1"},
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
        historical_message_excerpt_count=mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
    )

    assert did_compact is True
    assert "<checkpoint_summary><![CDATA[parent summary]]></checkpoint_summary>" in captured["source_messages"][0]["content"]
    assert captured["source_messages"][1:] == [
        {"role": "user", "content": "delta"},
        {"role": "assistant", "content": "delta answer"},
    ]
    assert "extended summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][-1] == {"role": "user", "content": "active"}


@pytest.mark.asyncio
async def test_compact_body_uses_raw_summary_source_but_canonical_checkpoint_hash(monkeypatch):
    store = ClaimStore()
    captured = {}

    async def generate_summary_text(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "model": "target",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": "old file",
                "files": [
                    {
                        "type": "text",
                        "name": "https://example.com/a",
                        "url": "https://example.com/a",
                        "status": "processing",
                        "progress": {"phase": "extracting"},
                        "file": {
                            "id": "file-1",
                            "filename": "page.txt",
                            "path": "/var/lib/open-webui/uploads/a/page.txt",
                            "hash": "abc123",
                            "data": {"content": "page text", "status": "pending"},
                            "meta": {"source": "https://example.com/a", "data": {"upload_id": "tmp-a"}},
                            "created_at": 100,
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    compacted, did_compact = await mod._compact_body(
        request=SimpleNamespace(state=SimpleNamespace()),
        user={"id": "user-1"},
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
        historical_message_excerpt_count=mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
    )

    assert did_compact is True
    summary_file = captured["source_messages"][0]["files"][0]
    assert summary_file == body["messages"][0]["files"][0]
    assert store.claimed_rows[0]["source_hash"] == mod.compute_source_hash(body["messages"][:2])
    assert store.completed_rows[0]["summary_text"] == "summary"
    assert "summary" in compacted["messages"][0]["content"]


@pytest.mark.asyncio
async def test_compact_body_uses_parent_checkpoint_delta_when_extension_summary_fails(monkeypatch):
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    parent = make_checkpoint_row(
        parent_source,
        state="ready",
        summary_text="parent summary",
        claim_token=None,
        claim_expires_at=None,
    )
    store = ClaimStore([parent])

    async def generate_summary_text(**kwargs):
        raise mod.RetryableContextOverflow("summary context overflow")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "model": "target",
        "stream": True,
        "previous_response_id": "resp-old",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "delta"},
            {"role": "assistant", "content": "delta answer"},
            {"role": "user", "content": "active"},
        ],
    }

    compacted, did_compact = await mod._compact_body(
        request=SimpleNamespace(state=SimpleNamespace()),
        user={"id": "user-1"},
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
        historical_message_excerpt_count=mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
    )

    assert did_compact is True
    assert "previous_response_id" not in compacted
    assert compacted["messages"][0] == {"role": "system", "content": "system"}
    assert "parent summary" in compacted["messages"][1]["content"]
    assert {"role": "user", "content": "old"} not in compacted["messages"]
    assert {"role": "assistant", "content": "old answer"} not in compacted["messages"]
    assert compacted["messages"][2:] == [
        {"role": "user", "content": "delta"},
        {"role": "assistant", "content": "delta answer"},
        {"role": "user", "content": "active"},
    ]
    assert store.completed_rows == []
    assert [row["id"] for row in store.rows] == [parent["id"]]


def test_checkpoint_ddl_tolerates_only_duplicate_object_errors():
    from sqlalchemy.exc import OperationalError

    class FakeNested:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConn:
        def __init__(self, error):
            self.error = error

        def begin_nested(self):
            return FakeNested()

        def execute(self, statement):
            raise self.error

    duplicate_table = OperationalError("CREATE TABLE x", {}, Exception("table x already exists"))
    mod._execute_checkpoint_ddl_tolerating_duplicates(FakeConn(duplicate_table), "CREATE TABLE x")

    duplicate_column = OperationalError("ALTER TABLE x", {}, Exception("duplicate column name: claim_token"))
    mod._execute_checkpoint_ddl_tolerating_duplicates(FakeConn(duplicate_column), "ALTER TABLE x")

    broken = OperationalError("CREATE TABLE x", {}, Exception("disk I/O error"))
    with pytest.raises(OperationalError):
        mod._execute_checkpoint_ddl_tolerating_duplicates(FakeConn(broken), "CREATE TABLE x")


def _engine_store_factory(engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    sessionmaker = async_sessionmaker(bind=engine, expire_on_commit=False)

    class EngineCheckpointStore(mod.CheckpointStore):
        async def _context(self):
            return sessionmaker()

    return EngineCheckpointStore


async def _checkpoint_table_columns(engine):
    import sqlalchemy

    async with engine.connect() as conn:
        return await conn.run_sync(
            lambda sync_conn: {
                column["name"]
                for column in sqlalchemy.inspect(sync_conn).get_columns(mod.CHECKPOINT_TABLE_NAME)
            }
        )


@pytest.fixture
async def claim_engine(tmp_path, monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/checkpoints.db",
        poolclass=NullPool,
        connect_args={"timeout": 30},
    )
    monkeypatch.setattr(mod, "_CHECKPOINT_SCHEMA_READY", False)
    yield engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_callers_generate_summary_only_once(claim_engine, monkeypatch):
    await mod.ensure_checkpoint_table_initialized(async_engine=claim_engine)

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", _engine_store_factory(claim_engine))
    monkeypatch.setattr(mod, "get_generation_lock", lambda key: asyncio.Lock())
    monkeypatch.setattr(mod, "release_generation_lock", lambda key, lock: None)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_POLL_SECONDS", 0.02)

    source_messages = [{"role": "user", "content": "old"}]
    factory_calls = []

    def make_summary_factory(label):
        async def summary_factory(parent):
            factory_calls.append(label)
            await asyncio.sleep(0.2)
            return "generated summary"

        return summary_factory

    async def run(label):
        return await mod._get_or_create_checkpoint_summary(
            request=None,
            user_id="user-1",
            chat_id="chat-1",
            pipe_function_id="auto_compact",
            source_messages=source_messages,
            summary_meta={"summary_model": label},
            summary_factory=make_summary_factory(label),
        )

    first, second = await asyncio.gather(run("summary-model-a"), run("summary-model-b"))

    assert first == "generated summary"
    assert second == "generated summary"
    assert len(factory_calls) == 1
    assert first.checkpoint["id"] == second.checkpoint["id"]
    assert first.checkpoint["state"] == "ready"
    assert second.checkpoint["state"] == "ready"


@pytest.mark.asyncio
async def test_stale_owner_cannot_overwrite_reclaimed_checkpoint(claim_engine):
    await mod.ensure_checkpoint_table_initialized(async_engine=claim_engine)
    store = _engine_store_factory(claim_engine)()
    source_messages = [{"role": "user", "content": "old"}]
    pending = make_checkpoint_row(source_messages, claim_token="stale-owner", claim_expires_at=100, now=100)

    assert await store.claim_pending(pending) is True
    assert await store.claim_pending(dict(pending, claim_token="rival")) is False
    assert (
        await store.reclaim_pending(pending["id"], claim_token="too-early", expires_at=10**12, now=50)
        is False
    )
    assert (
        await store.reclaim_pending(pending["id"], claim_token="fresh-owner", expires_at=10**12, now=200)
        is True
    )
    assert (
        await store.complete_pending(
            pending["id"],
            claim_token="stale-owner",
            summary_text="stale summary",
            parent_checkpoint_id=None,
        )
        is None
    )

    completed = await store.complete_pending(
        pending["id"],
        claim_token="fresh-owner",
        summary_text="fresh summary",
        parent_checkpoint_id=None,
    )

    assert completed is not None
    assert completed["state"] == "ready"
    assert completed["summary_text"] == "fresh summary"
    assert completed["claim_token"] is None
    assert (
        await store.reclaim_pending(pending["id"], claim_token="late", expires_at=10**12, now=10**11)
        is False
    )
    ready = await store.lookup_ready(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(source_messages),
    )
    assert ready is not None
    assert ready["summary_text"] == "fresh summary"


@pytest.mark.asyncio
async def test_released_claim_lets_next_caller_claim_again(claim_engine):
    await mod.ensure_checkpoint_table_initialized(async_engine=claim_engine)
    store = _engine_store_factory(claim_engine)()
    source_messages = [{"role": "user", "content": "old"}]
    pending = make_checkpoint_row(source_messages, claim_token="failed-owner")

    assert await store.claim_pending(pending) is True
    assert await store.release_claim(pending["id"], claim_token="other-token") is False
    assert await store.release_claim(pending["id"], claim_token="failed-owner") is True
    assert await store.claim_pending(dict(pending, claim_token="next-owner")) is True


@pytest.mark.asyncio
async def test_schema_init_migrates_existing_table_and_preserves_ready_rows(claim_engine):
    from sqlalchemy import text

    source_messages = [{"role": "user", "content": "old"}]
    legacy = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(source_messages),
        source_message_count=len(source_messages),
        summary_text="legacy summary",
        summary_meta={},
        parent_checkpoint_id=None,
        now=100,
    )
    legacy_create = (
        f'CREATE TABLE "{mod.CHECKPOINT_TABLE_NAME}" ('
        "id TEXT PRIMARY KEY, namespace TEXT NOT NULL, schema_version INTEGER NOT NULL, "
        "user_id TEXT NOT NULL, chat_id TEXT NOT NULL, pipe_function_id TEXT NOT NULL, "
        "profile_hash TEXT NOT NULL, source_message_count INTEGER NOT NULL, source_hash TEXT NOT NULL, "
        "summary_text TEXT NOT NULL, summary_meta JSON NOT NULL, state TEXT NOT NULL, "
        "parent_checkpoint_id TEXT, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, "
        "last_used_at BIGINT NOT NULL)"
    )
    legacy_params = {
        key: value
        for key, value in legacy.items()
        if key not in ("claim_token", "claim_expires_at")
    }
    legacy_params["summary_meta"] = json.dumps(legacy_params["summary_meta"])
    async with claim_engine.begin() as conn:
        await conn.execute(text(legacy_create))
        await conn.execute(
            text(
                f'INSERT INTO "{mod.CHECKPOINT_TABLE_NAME}" '
                "(id, namespace, schema_version, user_id, chat_id, pipe_function_id, profile_hash, "
                "source_message_count, source_hash, summary_text, summary_meta, state, "
                "parent_checkpoint_id, created_at, updated_at, last_used_at) "
                "VALUES (:id, :namespace, :schema_version, :user_id, :chat_id, :pipe_function_id, "
                ":profile_hash, :source_message_count, :source_hash, :summary_text, :summary_meta, "
                ":state, :parent_checkpoint_id, :created_at, :updated_at, :last_used_at)"
            ),
            legacy_params,
        )

    await mod.ensure_checkpoint_table_initialized(async_engine=claim_engine)

    columns = await _checkpoint_table_columns(claim_engine)
    assert {"claim_token", "claim_expires_at"} <= columns

    store = _engine_store_factory(claim_engine)()
    ready = await store.lookup_ready(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(source_messages),
    )
    assert ready is not None
    assert ready["summary_text"] == "legacy summary"
    assert ready["claim_token"] is None


@pytest.mark.asyncio
async def test_checkpoint_schema_init_tolerates_concurrent_and_repeated_runs(claim_engine):
    async def init_once():
        async with claim_engine.begin() as conn:
            await conn.run_sync(mod._initialize_checkpoint_schema)

    await asyncio.gather(init_once(), init_once())
    await init_once()

    columns = await _checkpoint_table_columns(claim_engine)
    assert {"claim_token", "claim_expires_at"} <= columns
