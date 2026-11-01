"""Council must close real MCP/AnyIO scopes in their owning task and order."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp_sdk import patch_mcp_sdk


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "prepare_error", "prepare_cancelled", "cancelled"])
async def test_council_mcp_lifetime(monkeypatch, outcome):
    import multi_model_council as council
    from open_webui.utils.mcp.client import MCPClient

    opened, exit_attempts, closed, loops = [], [], [], []
    started = asyncio.Event()
    blocked = asyncio.Event()
    models = {name: {"id": name} for name in ("first", "second")}

    @asynccontextmanager
    async def connection(label):
        try:
            async with anyio.create_task_group():
                opened.append((label, asyncio.current_task()))
                try:
                    yield None
                finally:
                    exit_attempts.append((label, asyncio.current_task()))
        except asyncio.CancelledError:
            # Core 0.12.5+ closes a session by cancelling the task that holds it.
            closed.append(label)
            raise
        # Core catches RuntimeError, so check successful exits too.
        closed.append(label)

    async def load_tools(**kwargs):
        clients = {}
        for index in range(2):
            client = MCPClient()
            await client.connect(url=f"{kwargs['model']['id']}:{index}")
            clients[str(index)] = client
        await asyncio.sleep(0)
        return {}, clients

    async def register_skill(tools, request, extra):
        if extra["__model__"]["id"] == "first" and outcome == "prepare_error":
            raise RuntimeError("first preparation failed")
        if extra["__model__"]["id"] == "second" and outcome == "prepare_cancelled":
            started.set()
            await blocked.wait()

    async def loop(**kwargs):
        loops.append(kwargs["agent_name"])
        assert kwargs["extra_params"]["__model__"] is kwargs["filter_pipeline"]["model"]
        assert not closed
        if outcome == "cancelled":
            started.set()
            await blocked.wait()
        await asyncio.sleep(0)
        return '{"vote":"A","reasoning":"done"}'

    patch_mcp_sdk(monkeypatch, connection)
    monkeypatch.setattr(council, "build_tools_dict", load_tools)
    monkeypatch.setattr(council, "run_agent_loop", loop)
    monkeypatch.setattr(council, "register_view_skill", register_skill)
    monkeypatch.setattr(council, "extract_skill_manifest", lambda _messages: "<available_skills />")
    monkeypatch.setattr(council, "get_available_models", AsyncMock(return_value=list(models.values())))
    request = SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(state=SimpleNamespace(MODELS=models)),
        body=AsyncMock(return_value=b"{}"),
    )
    tool = council.Tools()
    tool.valves.APPLY_INLET_FILTERS = False
    invocation = asyncio.create_task(tool.council_decide(
        proposition="Choose", option_a="A", option_b="B", models="first,second",
        __request__=request, __model__=models["first"],
        __user__={
            "id": "mcp-user", "email": "mcp@example.com", "name": "MCP User", "role": "user",
            "last_active_at": 0, "updated_at": 0, "created_at": 0,
        },
    ))
    if outcome in {"prepare_cancelled", "cancelled"}:
        await asyncio.wait_for(started.wait(), timeout=5)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert loops == ([] if outcome == "prepare_cancelled" else ["first", "second"])
    else:
        response = json.loads(await invocation)
        assert response["members"]["second"]["vote"] == "A"
        if outcome == "prepare_error":
            assert response["members"]["first"]["vote"] == "abstain"
            assert "first preparation failed" in response["members"]["first"]["reasoning"]
            assert loops == ["second"]
        else:
            assert response["decision"] == "A"
            assert loops == ["first", "second"]

    assert [label for label, _ in opened] == ["first:0", "first:1", "second:0", "second:1"]
    assert dict(exit_attempts) == dict(opened)
    assert [label for label, _ in exit_attempts] == closed == [label for label, _ in reversed(opened)]
