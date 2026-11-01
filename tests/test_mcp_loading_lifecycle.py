"""MCP acquisition failures must unwind real cancel scopes in their owner task."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp_sdk import patch_mcp_sdk

from open_webui.models.config import Config
from open_webui.utils import access_control
from open_webui.utils import tools as core_tools
from open_webui.utils.mcp import client as core_mcp
from owui_ext.shared import mcp_tools, tool_loader


@pytest.fixture
def mcp_lifecycle(monkeypatch):
    clients = []
    closed = []
    scopes = {}
    state = SimpleNamespace(
        fail_at=None, error=None, clients=clients, closed=closed, scopes=scopes, emitter=AsyncMock(),
        config=None, event_callers=[], specs=[], instructions=None, owner=None,
    )

    async def checkpoint(location):
        if state.fail_at == location:
            if state.error is not None:
                raise state.error
            (state.owner or asyncio.current_task()).cancel()
            await asyncio.sleep(0)

    @asynccontextmanager
    async def connection(url):
        opener = asyncio.current_task()
        scope = scopes[url] = anyio.CancelScope()
        try:
            with scope:
                await checkpoint(f"{url}:connect")
                try:
                    yield state.instructions
                finally:
                    await checkpoint(f"{url}:disconnect")
        finally:
            # AnyIO leaves a scope active when it is exited from another task.
            closed.append((url, asyncio.current_task() is opener and not scope._active))

    class ScopedMCPClient(core_mcp.MCPClient):
        def __init__(self, event_caller=None, **kwargs):
            super().__init__()
            state.event_callers.append(event_caller)

        async def connect(self, url, *args, **kwargs):
            self.name = url
            clients.append(self)
            await super().connect(url, *args, **kwargs)

        async def list_tool_specs(self):
            await checkpoint(f"{self.name}:list")
            return list(state.specs)

    patch_mcp_sdk(monkeypatch, connection)

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
            state.owner = asyncio.current_task()
            try:
                await acquire()
            finally:
                observed["closed"] = list(closed)
                # A failing regression must not leave scopes in pytest's task.
                # Capture first, then rescue any missing cleanup in this owner.
                for client in reversed(clients):
                    try:
                        await client.disconnect()
                    except BaseException:
                        pass

        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(worker())
        assert observed["closed"] == [(client.name, True) for client in reversed(clients)]

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
        first, _ = mcp_lifecycle.clients
        assert mcp_lifecycle.closed == [("second", True)]
        assert clients == {"first": first}
        assert mcp_lifecycle.scopes["first"]._active is True
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


@pytest.mark.skipif(
    not hasattr(core_tools, "connect_mcp_server"), reason="Core before 0.12.0 does not read MCP server instructions"
)
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
