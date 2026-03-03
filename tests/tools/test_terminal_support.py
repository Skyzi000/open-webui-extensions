"""Terminal integration tests for tool wrappers."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add tools directory to path
tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

import magi_decision_support
import multi_model_council
import parallel_tools
import sub_agent


def make_spec(*param_names: str) -> dict:
    return {
        "parameters": {
            "properties": {
                name: {"type": "string"} for name in param_names
            }
        }
    }


@pytest.fixture
def dummy_request():
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(TERMINAL_SERVER_CONNECTIONS=[]),
            )
        )
    )


@pytest.mark.asyncio
async def test_sub_agent_execute_tool_call_terminal_tuple_and_event():
    events = []

    async def event_emitter(event: dict):
        events.append(event)

    async def display_file(path: str):
        return ({"exists": True, "path": path}, {"Content-Type": "application/json"})

    tools_dict = {
        "display_file": {
            "callable": display_file,
            "spec": make_spec("path"),
            "type": "terminal",
            "tool_id": "terminal:test",
        }
    }

    tool_call = {
        "id": "tc-1",
        "function": {
            "name": "display_file",
            "arguments": json.dumps({"path": "/tmp/test.html"}),
        },
    }

    result = await sub_agent.execute_tool_call(
        tool_call=tool_call,
        tools_dict=tools_dict,
        extra_params={},
        event_emitter=event_emitter,
    )

    payload = json.loads(result["content"])
    assert payload["path"] == "/tmp/test.html"
    assert any(
        e.get("type") == "terminal:display_file"
        and e.get("data", {}).get("path") == "/tmp/test.html"
        for e in events
    )


@pytest.mark.asyncio
async def test_sub_agent_load_tools_includes_terminal_tools(monkeypatch, dummy_request):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        return {
            "run_command": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)

    valves = sub_agent.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}, "terminal_id": "term-1"}
    tools_dict = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params={"__metadata__": metadata},
        self_tool_id=None,
    )

    assert "run_command" in tools_dict
    assert tools_dict["run_command"]["type"] == "terminal"


@pytest.mark.asyncio
async def test_sub_agent_load_tools_without_terminal_symbol_keeps_builtin(
    monkeypatch, dummy_request
):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {
            "search_web": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("query"),
                "type": "builtin",
                "tool_id": "builtin:search_web",
            }
        }

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.delattr(ow_tools, "get_terminal_tools", raising=False)

    valves = sub_agent.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}, "terminal_id": "term-compat"}
    tools_dict = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params={"__metadata__": metadata},
        self_tool_id=None,
    )

    assert "search_web" in tools_dict
    assert tools_dict["search_web"]["type"] == "builtin"


@pytest.mark.asyncio
async def test_sub_agent_terminal_error_does_not_skip_builtin(monkeypatch, dummy_request):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {
            "search_web": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("query"),
                "type": "builtin",
                "tool_id": "builtin:search_web",
            }
        }

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        raise RuntimeError("terminal unavailable")

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)

    valves = sub_agent.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}, "terminal_id": "term-fail"}
    tools_dict = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params={"__metadata__": metadata},
        self_tool_id=None,
    )

    assert "search_web" in tools_dict
    assert tools_dict["search_web"]["type"] == "builtin"


@pytest.mark.asyncio
async def test_sub_agent_uses_request_body_terminal_id_when_metadata_missing(
    monkeypatch, dummy_request
):
    import open_webui.utils.tools as ow_tools

    called_terminal_ids = []

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        called_terminal_ids.append(terminal_id)
        return {
            "run_command": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    async def fake_request_body():
        return b'{"terminal_id":"term-from-request"}'

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)
    setattr(dummy_request, "body", fake_request_body)

    valves = sub_agent.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}}
    tools_dict = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params={"__metadata__": metadata},
        self_tool_id=None,
    )

    assert called_terminal_ids == ["term-from-request"]
    assert "run_command" in tools_dict


@pytest.mark.asyncio
async def test_sub_agent_prefers_request_terminal_id_over_metadata(
    monkeypatch, dummy_request
):
    import open_webui.utils.tools as ow_tools

    called_terminal_ids = []

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        called_terminal_ids.append(terminal_id)
        return {
            "run_command": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    async def fake_request_body():
        return b'{"terminal_id":"term-parent"}'

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)
    setattr(dummy_request, "body", fake_request_body)

    valves = sub_agent.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}, "terminal_id": "term-metadata"}
    tools_dict = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params={"__metadata__": metadata},
        self_tool_id=None,
    )

    assert called_terminal_ids == ["term-parent"]
    assert "run_command" in tools_dict


@pytest.mark.asyncio
async def test_sub_agent_uses_request_body_terminal_id_before_json(
    monkeypatch, dummy_request
):
    import open_webui.utils.tools as ow_tools

    called_terminal_ids = []

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        called_terminal_ids.append(terminal_id)
        return {
            "run_command": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    async def fake_request_body():
        return b'{"terminal_id":"term-from-body"}'

    async def fake_request_json():
        raise AssertionError("request.json() should not be called")

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)
    setattr(dummy_request, "body", fake_request_body)
    setattr(dummy_request, "json", fake_request_json)

    valves = sub_agent.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}, "terminal_id": "term-metadata"}
    tools_dict = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params={"__metadata__": metadata},
        self_tool_id=None,
    )

    assert called_terminal_ids == ["term-from-body"]
    assert "run_command" in tools_dict


@pytest.mark.asyncio
async def test_sub_agent_propagates_resolved_terminal_id_to_extra_metadata(
    monkeypatch, dummy_request
):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        return {
            "run_command": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    async def fake_request_body():
        return b'{"terminal_id":"term-propagated"}'

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)
    setattr(dummy_request, "body", fake_request_body)

    valves = sub_agent.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}}
    extra_params = {"__metadata__": metadata}
    await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params=extra_params,
        self_tool_id=None,
    )

    assert metadata["terminal_id"] == "term-propagated"
    assert extra_params["__metadata__"]["terminal_id"] == "term-propagated"


@pytest.mark.asyncio
async def test_multi_model_council_build_tools_dict_includes_terminal(monkeypatch, dummy_request):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        return {
            "run_command": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)

    valves = multi_model_council.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}, "terminal_id": "term-council"}
    tools_dict = await multi_model_council.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        extra_params={"__metadata__": metadata},
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert "run_command" in tools_dict
    assert tools_dict["run_command"]["type"] == "terminal"


@pytest.mark.asyncio
async def test_multi_model_council_resolves_terminal_from_request_body(monkeypatch, dummy_request):
    import open_webui.utils.tools as ow_tools

    called_terminal_ids = []

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        called_terminal_ids.append(terminal_id)
        return {
            "run_command": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    async def fake_request_body():
        return b'{"terminal_id":"term-council-body"}'

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)
    setattr(dummy_request, "body", fake_request_body)

    valves = multi_model_council.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}}
    extra_params = {"__metadata__": metadata}
    tools_dict = await multi_model_council.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        extra_params=extra_params,
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert called_terminal_ids == ["term-council-body"]
    assert extra_params["__metadata__"]["terminal_id"] == "term-council-body"
    assert "run_command" in tools_dict


@pytest.mark.asyncio
async def test_multi_model_council_without_terminal_symbol_keeps_builtin(
    monkeypatch, dummy_request
):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {
            "search_web": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("query"),
                "type": "builtin",
                "tool_id": "builtin:search_web",
            }
        }

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.delattr(ow_tools, "get_terminal_tools", raising=False)

    valves = multi_model_council.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {"tool_ids": [], "features": {}, "terminal_id": "term-council-compat"}
    tools_dict = await multi_model_council.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        extra_params={"__metadata__": metadata},
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert "search_web" in tools_dict
    assert tools_dict["search_web"]["type"] == "builtin"


@pytest.mark.asyncio
async def test_magi_build_tools_dict_includes_terminal(monkeypatch, dummy_request):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        return {
            "run_command": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)

    metadata = {"tool_ids": [], "features": {}, "terminal_id": "term-magi"}
    tools_dict = await magi_decision_support.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        enable_web_tools=True,
        enable_terminal_tools=True,
        extra_params={"__metadata__": metadata},
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert "run_command" in tools_dict
    assert tools_dict["run_command"]["type"] == "terminal"


@pytest.mark.asyncio
async def test_magi_build_tools_dict_resolves_terminal_from_request_body(
    monkeypatch, dummy_request
):
    import open_webui.utils.tools as ow_tools

    called_terminal_ids = []

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        called_terminal_ids.append(terminal_id)
        return {
            "run_command": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    async def fake_request_body():
        return b'{"terminal_id":"term-magi-body"}'

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)
    setattr(dummy_request, "body", fake_request_body)

    metadata = {"tool_ids": [], "features": {}}
    extra_params = {"__metadata__": metadata}
    tools_dict = await magi_decision_support.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        enable_web_tools=True,
        enable_terminal_tools=True,
        extra_params=extra_params,
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert called_terminal_ids == ["term-magi-body"]
    assert extra_params["__metadata__"]["terminal_id"] == "term-magi-body"
    assert "run_command" in tools_dict


@pytest.mark.asyncio
async def test_magi_build_tools_dict_without_terminal_symbol_keeps_builtin(
    monkeypatch, dummy_request
):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {
            "search_web": {
                "callable": lambda **kwargs: None,
                "spec": make_spec("query"),
                "type": "builtin",
                "tool_id": "builtin:search_web",
            }
        }

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.delattr(ow_tools, "get_terminal_tools", raising=False)

    metadata = {"tool_ids": [], "features": {}, "terminal_id": "term-magi-compat"}
    tools_dict = await magi_decision_support.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        enable_web_tools=True,
        enable_terminal_tools=True,
        extra_params={"__metadata__": metadata},
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert "search_web" in tools_dict
    assert tools_dict["search_web"]["type"] == "builtin"


@pytest.mark.asyncio
async def test_parallel_tools_run_tools_parallel_resolves_terminal_tools(
    monkeypatch, dummy_request, mock_user
):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def run_command(command: str):
        return {"stdout": f"executed:{command}"}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        return {
            "run_command": {
                "callable": run_command,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)

    user_payload = {
        **mock_user,
        "last_active_at": 0,
        "updated_at": 0,
        "created_at": 0,
    }

    tool = parallel_tools.Tools()
    result_json = await tool.run_tools_parallel(
        tool_calls=[{"name": "run_command", "args": {"command": "pwd"}}],
        __user__=user_payload,
        __request__=dummy_request,
        __metadata__={"tool_ids": [], "features": {}, "terminal_id": "term-parallel"},
    )

    payload = json.loads(result_json)
    assert "results" in payload
    assert payload["results"][0]["tool_name"] == "run_command"
    assert payload["results"][0]["result"]["stdout"] == "executed:pwd"


@pytest.mark.asyncio
async def test_parallel_tools_resolves_terminal_from_request_body(
    monkeypatch, dummy_request, mock_user
):
    import open_webui.utils.tools as ow_tools

    called_terminal_ids = []

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def run_command(command: str):
        return {"stdout": f"executed:{command}"}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        called_terminal_ids.append(terminal_id)
        return {
            "run_command": {
                "callable": run_command,
                "spec": make_spec("command"),
                "type": "terminal",
                "tool_id": f"terminal:{terminal_id}",
            }
        }

    async def fake_request_body():
        return b'{"terminal_id":"term-parallel-body"}'

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)
    setattr(dummy_request, "body", fake_request_body)

    user_payload = {
        **mock_user,
        "last_active_at": 0,
        "updated_at": 0,
        "created_at": 0,
    }

    tool = parallel_tools.Tools()
    result_json = await tool.run_tools_parallel(
        tool_calls=[{"name": "run_command", "args": {"command": "pwd"}}],
        __user__=user_payload,
        __request__=dummy_request,
        __metadata__={"tool_ids": [], "features": {}},
    )

    payload = json.loads(result_json)
    assert called_terminal_ids == ["term-parallel-body"]
    assert payload["results"][0]["result"]["stdout"] == "executed:pwd"


@pytest.mark.asyncio
async def test_sub_agent_execute_tool_call_not_found_does_not_emit_terminal_event():
    events = []

    async def event_emitter(event: dict):
        events.append(event)

    tool_call = {
        "id": "tc-missing-sub-agent",
        "function": {
            "name": "display_file",
            "arguments": json.dumps({"path": "/tmp/should-not-emit.txt"}),
        },
    }

    result = await sub_agent.execute_tool_call(
        tool_call=tool_call,
        tools_dict={},
        extra_params={},
        event_emitter=event_emitter,
    )

    assert "Tool 'display_file' not found" in result["content"]
    assert not any(event.get("type", "").startswith("terminal:") for event in events)


@pytest.mark.asyncio
async def test_sub_agent_execute_tool_call_exception_does_not_emit_terminal_event():
    events = []

    async def event_emitter(event: dict):
        events.append(event)

    async def failing_display_file(path: str):
        raise RuntimeError(f"failed:{path}")

    tool_call = {
        "id": "tc-error-sub-agent",
        "function": {
            "name": "display_file",
            "arguments": json.dumps({"path": "/tmp/raise-error.txt"}),
        },
    }
    tools_dict = {
        "display_file": {
            "callable": failing_display_file,
            "spec": make_spec("path"),
            "type": "terminal",
            "tool_id": "terminal:test",
        }
    }

    result = await sub_agent.execute_tool_call(
        tool_call=tool_call,
        tools_dict=tools_dict,
        extra_params={},
        event_emitter=event_emitter,
    )

    assert result["content"].startswith("Error:")
    assert not any(event.get("type", "").startswith("terminal:") for event in events)


@pytest.mark.asyncio
async def test_multi_model_council_execute_tool_call_not_found_does_not_emit_terminal_event():
    events = []

    async def event_emitter(event: dict):
        events.append(event)

    tool_call = {
        "id": "tc-missing-council",
        "function": {
            "name": "display_file",
            "arguments": json.dumps({"path": "/tmp/council-missing.txt"}),
        },
    }

    result = await multi_model_council.execute_tool_call(
        tool_call=tool_call,
        tools_dict={},
        extra_params={},
        event_emitter=event_emitter,
    )

    assert "Tool 'display_file' not found" in result["content"]
    assert not any(event.get("type", "").startswith("terminal:") for event in events)


@pytest.mark.asyncio
async def test_magi_execute_tool_call_not_found_does_not_emit_terminal_event():
    events = []

    async def event_emitter(event: dict):
        events.append(event)

    tool_call = {
        "id": "tc-missing-magi",
        "function": {
            "name": "display_file",
            "arguments": json.dumps({"path": "/tmp/magi-missing.txt"}),
        },
    }

    result = await magi_decision_support.execute_tool_call(
        tool_call=tool_call,
        tools_dict={},
        extra_params={},
        event_emitter=event_emitter,
    )

    assert "Tool 'display_file' not found" in result["content"]
    assert not any(event.get("type", "").startswith("terminal:") for event in events)
