"""MCP acquisition failures must unwind real cancel scopes in their owner task."""

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest

from open_webui.models.config import Config
from open_webui.utils import access_control
from open_webui.utils import tools as core_tools
from open_webui.utils.mcp import client as core_mcp
from owui_ext.shared import mcp_tools, tool_loader


@pytest.fixture
def mcp_lifecycle(monkeypatch):
    clients = []
    closed = []
    state = SimpleNamespace(
        fail_at=None, error=None, clients=clients, closed=closed, emitter=AsyncMock(),
        config=None, event_callers=[], specs=[], instructions=None,
    )

    async def checkpoint(location):
        if state.fail_at == location:
            if state.error is not None:
                raise state.error
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

    class ScopedMCPClient(core_mcp.MCPClient):
        def __init__(self, event_caller=None, **kwargs):
            super().__init__()
            state.event_callers.append(event_caller)

        async def connect(self, url, headers=None, auth=None):
            self.name = url
            self.owner = asyncio.current_task()
            self.scope = anyio.CancelScope()
            self.exit_stack = AsyncExitStack()
            self.exit_stack.enter_context(self.scope)
            self.exit_stack.push_async_callback(checkpoint, f"{url}:disconnect")
            self.session = object()
            self.instructions = state.instructions
            clients.append(self)
            await checkpoint(f"{url}:connect")

        async def list_tool_specs(self):
            await checkpoint(f"{self.name}:list")
            return list(state.specs)

        async def disconnect(self):
            closed.append((self.name, self.owner is asyncio.current_task()))
            # Real Core disconnect clears its handles and can suppress invalid
            # AnyIO exits; checking scope state detects those hidden failures.
            await super().disconnect()

    connections = [
        {"type": "mcp", "info": {"id": name}, "url": name, "config": {"enable": True}}
        for name in ("first", "second")
    ]

    async def headers(**kwargs):
        await checkpoint(f"{kwargs['server_id']}:headers")
        return {}

    async def core_headers(*args, **kwargs):
        return await headers(**kwargs), None

    async def config_get(key, default=None):
        return connections if key == "tool_server.connections" else state.config

    # Core 0.12.0+ connects through its own connect_mcp_server.
    monkeypatch.setattr(core_tools, "MCPClient", ScopedMCPClient, raising=False)
    monkeypatch.setattr(core_tools, "build_tool_server_headers", core_headers, raising=False)
    monkeypatch.setattr(core_tools, "has_connection_access", AsyncMock(return_value=True), raising=False)
    monkeypatch.setattr(Config, "get", config_get)
    monkeypatch.setattr(core_mcp, "MCPClient", ScopedMCPClient)
    monkeypatch.setattr(mcp_tools, "_get_tool_server_connections", AsyncMock(return_value=connections))
    monkeypatch.setattr(mcp_tools, "_build_mcp_headers_with_core", headers)
    monkeypatch.setattr(access_control, "has_connection_access", AsyncMock(return_value=True))

    async def resolve(**kwargs):
        return await mcp_tools.resolve_mcp_tools(
            request=SimpleNamespace(), user=SimpleNamespace(id="user"),
            mcp_tool_ids=["server:mcp:first", "server:mcp:second"],
            extra_params={"__event_emitter__": state.emitter, **kwargs.get("extra_params", {})},
            metadata=kwargs.get("metadata", {}),
        )

    async def cancelled_acquisition(acquire):
        observed = {}

        async def worker():
            try:
                await acquire()
            finally:
                observed["closed"] = list(closed)
                observed["scope_active"] = [client.scope._active for client in clients]
                # A failing regression must not leave scopes in pytest's task.
                # Capture first, then rescue any missing cleanup in this owner.
                for client in reversed(clients):
                    if client.exit_stack is not None:
                        try:
                            await client.disconnect()
                        except BaseException:
                            pass

        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(worker())
        expected = [(client.name, True) for client in reversed(clients)]
        assert observed["closed"] == expected
        assert observed["scope_active"] == [False] * len(clients)

    state.checkpoint = checkpoint
    state.resolve = resolve
    state.cancelled_acquisition = cancelled_acquisition
    return state


@pytest.mark.parametrize("fail_at", [
    "first:connect", "first:list", "second:headers", "second:connect", "second:list",
])
async def test_resolver_cancellation_closes_current_and_previous_clients(mcp_lifecycle, fail_at):
    mcp_lifecycle.fail_at = fail_at
    await mcp_lifecycle.cancelled_acquisition(mcp_lifecycle.resolve)


async def test_resolver_listing_error_closes_only_failed_server(mcp_lifecycle):
    mcp_lifecycle.fail_at = "second:list"
    mcp_lifecycle.error = RuntimeError("Tool listing failed")
    try:
        _, clients = await mcp_lifecycle.resolve()
        first, second = mcp_lifecycle.clients
        assert mcp_lifecycle.closed == [("second", True)]
        assert second.scope._active is False
        assert clients == {"first": first}
        assert first.scope._active is True
        mcp_lifecycle.emitter.assert_awaited_once_with({
            "type": "notification",
            "data": {
                "type": "warning",
                "content": "Could not load MCP tools from 'second': Tool listing failed",
            },
        })
    finally:
        # Also rescue a failed regression in acquisition order's reverse.
        for client in reversed(mcp_lifecycle.clients):
            if client.exit_stack is not None:
                await client.disconnect()


async def test_cleanup_cancellation_closes_remaining_groups_then_reraises(mcp_lifecycle):
    async def acquire_and_cleanup():
        _, clients = await mcp_lifecycle.resolve()
        mcp_lifecycle.fail_at = "second:disconnect"
        await mcp_tools.cleanup_mcp_clients(
            {"first": clients["first"]}, {"second": clients["second"]},
        )

    await mcp_lifecycle.cancelled_acquisition(acquire_and_cleanup)


@pytest.mark.parametrize("fail_at", ["terminal", "builtin"])
async def test_loader_cancellation_closes_acquired_mcp_clients(monkeypatch, mcp_lifecycle, fail_at):
    mcp_lifecycle.fail_at = fail_at
    mcp_lifecycle.config = [{"id": "terminal"}]

    async def terminal(**kwargs):
        await mcp_lifecycle.checkpoint("terminal")
        return {}, ""

    async def builtin(**kwargs):
        await mcp_lifecycle.checkpoint("builtin")
        return {}

    monkeypatch.setattr(core_tools, "get_terminal_tools", terminal)
    monkeypatch.setattr(core_tools, "get_builtin_tools", builtin)
    monkeypatch.setattr(tool_loader, "resolve_mcp_tools", mcp_lifecycle.resolve)

    async def acquire():
        return await tool_loader.build_tools_dict(
            request=SimpleNamespace(), model={"id": "model"}, metadata={},
            user=SimpleNamespace(id="user"), valves=SimpleNamespace(),
            extra_params={}, tool_id_list=["server:mcp:first", "server:mcp:second"],
            excluded_tool_ids=set(), resolved_terminal_id="terminal",
            resolved_direct_tool_servers=[],
        )

    await mcp_lifecycle.cancelled_acquisition(acquire)


needs_core_connect = pytest.mark.skipif(
    not hasattr(core_tools, "connect_mcp_server"),
    reason="Core before 0.12.0 has no connect_mcp_server",
)


@needs_core_connect
async def test_core_connection_receives_event_call_for_elicitation(mcp_lifecycle):
    event_call = AsyncMock()
    _, clients = await mcp_lifecycle.resolve(
        extra_params={"__event_call__": event_call}, metadata={"session_id": "session"},
    )
    await mcp_tools.cleanup_mcp_clients(clients)
    assert mcp_lifecycle.event_callers == [event_call, event_call]


@needs_core_connect
async def test_disabled_tool_servers_open_no_mcp_connection(monkeypatch, mcp_lifecycle):
    monkeypatch.setattr(core_tools, "ENABLE_TOOL_SERVERS", False)
    assert await mcp_lifecycle.resolve() == ({}, {})
    assert mcp_lifecycle.clients == []
    assert mcp_lifecycle.emitter.await_count == 2


async def test_long_mcp_tool_names_fit_the_provider_limit(mcp_lifecycle):
    mcp_lifecycle.specs = [{"name": "x" * 70, "description": "", "parameters": {}}]
    tools, clients = await mcp_lifecycle.resolve()
    await mcp_tools.cleanup_mcp_clients(clients)
    assert len(tools) == 2
    assert {len(name) for name in tools} == {64}
    assert {tool["spec"]["name"] for tool in tools.values()} == set(tools)


async def test_loader_adds_mcp_instructions_to_tool_prompts(monkeypatch, mcp_lifecycle):
    mcp_lifecycle.instructions = "Ask before writing."
    monkeypatch.setattr(core_tools, "get_builtin_tools", AsyncMock(return_value={}))
    extra_params = {"__event_emitter__": mcp_lifecycle.emitter}
    _, clients = await tool_loader.build_tools_dict(
        request=SimpleNamespace(), model={"id": "model"}, metadata={},
        user=SimpleNamespace(id="user"), valves=SimpleNamespace(),
        extra_params=extra_params, tool_id_list=["server:mcp:first"],
        excluded_tool_ids=set(), resolved_terminal_id=None,
        resolved_direct_tool_servers=[],
    )
    await mcp_tools.cleanup_mcp_clients(clients)
    assert extra_params["__direct_tool_server_system_prompts__"] == [
        'MCP server "first" instructions:\nAsk before writing.'
    ]
