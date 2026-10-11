"""Recover Core-consumed Direct catalogues without restoring withheld tools."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import sub_agent


CONNECTION = {
    "url": "https://terminal.example.com",
    "key": "connection-key",
    "headers": {"X-Workspace": "workspace-a"},
    "is_terminal": True,
}


def request_with_servers(servers):
    return SimpleNamespace(body=AsyncMock(return_value=json.dumps({"tool_servers": servers}).encode()))


def approved_tool(server, name="run_command"):
    return {"direct": True, "server": server, "spec": {"name": name}}


@pytest.fixture
def core_direct_catalogue():
    """Execute the actual Core Direct-loading and browser-shell gate blocks."""
    path = Path(__file__).parents[2] / "references/open-webui/backend/open_webui/utils/middleware.py"
    module = ast.parse(path.read_text())
    process = next(node for node in module.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "process_chat_payload")
    direct = next(node for node in ast.walk(process) if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "direct_tool_servers")
    resolution = next(node for node in ast.walk(process) if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "payload_tools")
    start = next(index for index, node in enumerate(resolution.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "shell_tools" for target in node.targets))
    end = next(index for index, node in enumerate(resolution.body[start:], start) if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "shell_tools")
    wrapper = ast.parse("""
async def catalogue(metadata, event_caller):
    direct_tool_servers = metadata['tool_servers']
    terminal_id = metadata['terminal_id']
    terminal_capability = True
    form_data = {'messages': []}
    tools_dict = {}
""")
    wrapper.body[0].body.extend([direct, *resolution.body[start:end + 1]])
    wrapper.body[0].body.extend(ast.parse("metadata['tools'] = tools_dict\nreturn tools_dict").body)
    import asyncio

    namespace = {
        "asyncio": asyncio,
        "add_or_update_system_message": lambda content, messages, append: messages + [{"role": "system", "content": content}],
    }
    exec(compile(ast.fix_missing_locations(wrapper), str(path), "exec"), namespace)
    return namespace["catalogue"]


@pytest.mark.parametrize("connected", [True, False])
@pytest.mark.parametrize("is_terminal", [True, False])
async def test_actual_core_consumed_servers_reach_nested_loader(core_direct_catalogue, connected, is_terminal):
    names = ("run_command", "read_user_terminal", "send_user_terminal_input") if is_terminal else ("lookup",)
    server = {
        **CONNECTION,
        "is_terminal": is_terminal,
        "system_prompt": "Use the terminal workspace.",
        "specs": [{"name": name} for name in names],
    }
    request = request_with_servers([server])
    metadata = {
        "tool_servers": [server], "terminal_id": CONNECTION["url"] if is_terminal else "",
        "session_id": "browser", "chat_id": "chat",
    }
    approved = await core_direct_catalogue(metadata, AsyncMock(return_value={"connected": connected}))
    # Core before 0.12.0 consumes these from metadata; later versions keep them.
    consumed = "specs" not in server
    assert ("system_prompt" not in server) is consumed
    snapshot = copy.deepcopy(metadata)

    resolved = await sub_agent.resolve_direct_tool_servers_from_request_and_metadata(
        request=request, metadata=metadata,
    )
    assert metadata == snapshot
    assert request.body.await_count == consumed
    extra = {"__metadata__": metadata}
    loaded, clients = await sub_agent.build_tools_dict(
        request=request, model={}, metadata=metadata,
        user=SimpleNamespace(id="u1", role="user"), valves=SimpleNamespace(),
        extra_params=extra, tool_id_list=[], excluded_tool_ids=None,
        resolved_terminal_id=metadata["terminal_id"], resolved_direct_tool_servers=resolved,
    )
    assert clients == {}
    assert set(loaded) == set(approved)
    assert ("run_command" if is_terminal else "lookup") in loaded
    assert ("read_user_terminal" in loaded) is (is_terminal and connected)
    assert ("send_user_terminal_input" in loaded) is (is_terminal and connected)
    assert extra["__direct_tool_server_system_prompts__"] == ["Use the terminal workspace."]
    # A second wrapper uses the recovered metadata without another raw read.
    again = await sub_agent.resolve_direct_tool_servers_from_request_and_metadata(
        request=request, metadata=metadata,
    )
    assert again == resolved
    assert request.body.await_count == consumed


@pytest.mark.parametrize("servers", [None, []])
async def test_explicit_metadata_denial_never_uses_raw_servers(servers):
    request = request_with_servers([{**CONNECTION, "specs": [{"name": "run_command"}]}])
    result = await sub_agent.resolve_direct_tool_servers_from_request_and_metadata(
        request=request, metadata={"tool_servers": servers, "tools": {"run_command": approved_tool(CONNECTION)}},
    )
    assert result == []
    request.body.assert_not_awaited()


@pytest.mark.parametrize("changed", [{"key": "other-key"}, {"headers": {"X-Workspace": "other"}}, {"is_terminal": False}])
async def test_same_url_with_different_connection_cannot_supply_specs_or_prompt(changed):
    other = {**CONNECTION, **changed}
    request = request_with_servers([{**CONNECTION, "system_prompt": "Raw prompt", "specs": [{"name": "raw_only"}]}])
    metadata = {"tool_servers": [CONNECTION], "tools": {"run_command": approved_tool(other)}}
    result = await sub_agent.resolve_direct_tool_servers_from_request_and_metadata(request=request, metadata=metadata)
    assert result == [CONNECTION]
    request.body.assert_not_awaited()


@pytest.mark.parametrize("specs", [[], None])
async def test_explicit_empty_specs_are_not_replaced(specs):
    server = {**CONNECTION, "specs": specs}
    request = request_with_servers([{**CONNECTION, "system_prompt": "Raw prompt", "specs": [{"name": "raw_only"}]}])
    result = await sub_agent.resolve_direct_tool_servers_from_request_and_metadata(
        request=request, metadata={"tool_servers": [server], "tools": {"run_command": approved_tool(CONNECTION)}},
    )
    assert result == [server]
    request.body.assert_not_awaited()


@pytest.mark.parametrize("prompt", ["", None, "Trusted prompt"])
async def test_explicit_prompt_is_preserved_while_specs_recover(prompt):
    server = {**CONNECTION, "system_prompt": prompt}
    request = request_with_servers([{**CONNECTION, "system_prompt": "Raw prompt"}])
    result = await sub_agent.resolve_direct_tool_servers_from_request_and_metadata(
        request=request, metadata={"tool_servers": [server], "tools": {"run_command": approved_tool(CONNECTION)}},
    )
    assert result == [{**server, "specs": [{"name": "run_command"}]}]
    request.body.assert_not_awaited()


@pytest.mark.parametrize("prompts, expected", [(["one", "two"], None), (["one", None], None), ([123], None), (["same", "same"], "same")])
async def test_only_unambiguous_string_prompt_recovers(prompts, expected):
    raw = [{**CONNECTION, "system_prompt": prompt, "specs": [{"name": "raw_only"}]} for prompt in prompts]
    request = request_with_servers(raw)
    metadata = {"tool_servers": [CONNECTION], "tools": {"run_command": approved_tool(dict(CONNECTION))}}
    snapshot = copy.deepcopy(metadata)
    result = await sub_agent.resolve_direct_tool_servers_from_request_and_metadata(request=request, metadata=metadata)
    assert result[0]["specs"] == [{"name": "run_command"}]
    assert result[0].get("system_prompt") == expected
    assert ("system_prompt" in result[0]) is (expected is not None)
    assert metadata == snapshot
    request.body.assert_awaited_once()


async def test_prompt_requires_matching_raw_connection_and_approved_direct_entry():
    request = request_with_servers([{**CONNECTION, "key": "other-key", "system_prompt": "Wrong connection", "specs": [{"name": "raw_only"}]}])
    metadata = {"tool_servers": [CONNECTION], "tools": {"run_command": approved_tool(CONNECTION)}}
    result = await sub_agent.resolve_direct_tool_servers_from_request_and_metadata(request=request, metadata=metadata)
    assert result == [{**CONNECTION, "specs": [{"name": "run_command"}]}]
    for tools in ({}, {"run_command": {**approved_tool(CONNECTION), "direct": False}}):
        result = await sub_agent.resolve_direct_tool_servers_from_request_and_metadata(request=request, metadata={"tool_servers": [CONNECTION], "tools": tools})
        assert result == [CONNECTION]
    request.body.assert_awaited_once()
