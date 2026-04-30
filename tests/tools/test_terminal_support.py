"""Terminal integration tests for tool wrappers."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add tools directory to path
tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

import magi_decision_support  # noqa: E402
import multi_model_council  # noqa: E402
import parallel_tools  # noqa: E402
import sub_agent  # noqa: E402


def make_spec(*param_names: str) -> dict:
    return {
        "parameters": {
            "properties": {
                name: {"type": "string"} for name in param_names
            }
        }
    }


TERMINAL_EVENT_MODULE_CASES = [
    (
        "sub_agent",
        sub_agent,
        "tool_function_name",
        "tool_function_params",
    ),
    (
        "multi_model_council",
        multi_model_council,
        "tool_function_name",
        "tool_function_params",
    ),
    (
        "magi_decision_support",
        magi_decision_support,
        "tool_function_name",
        "tool_function_params",
    ),
    (
        "parallel_tools",
        parallel_tools,
        "tool_function_name",
        "tool_function_params",
    ),
]

RESULT_HELPER_MODULES = [
    ("sub_agent", sub_agent),
    ("multi_model_council", multi_model_council),
    ("magi_decision_support", magi_decision_support),
    ("parallel_tools", parallel_tools),
]

BUILTIN_KNOWLEDGE_MODULES = [
    ("sub_agent", sub_agent),
    ("multi_model_council", multi_model_council),
]


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
    tools_dict, _ = await sub_agent.load_sub_agent_tools(
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
    tools_dict, _ = await sub_agent.load_sub_agent_tools(
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
async def test_sub_agent_load_tools_returns_tools_and_mcp_clients(monkeypatch, dummy_request):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        return {}

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)

    valves = sub_agent.Tools().valves
    metadata = {"tool_ids": [], "features": {}}
    tools_dict, mcp_clients = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params={"__metadata__": metadata},
        self_tool_id=None,
    )

    assert isinstance(tools_dict, dict)
    assert mcp_clients == {}


@pytest.mark.asyncio
async def test_sub_agent_resolve_mcp_tools_supports_oauth_2_1_static(monkeypatch):
    from types import ModuleType

    import open_webui
    import open_webui.env as ow_env
    import open_webui.utils.tools as ow_tools

    oauth_calls = []
    connected_headers = []

    class FakeOAuthClientManager:
        async def get_oauth_token(self, user_id, client_id):
            oauth_calls.append((user_id, client_id))
            return {"access_token": "static-token"}

    class FakeMCPClient:
        async def connect(self, url: str, headers=None):
            connected_headers.append({"url": url, "headers": headers})

        async def list_tool_specs(self):
            return [
                {
                    "name": "lookup_docs",
                    "description": "Lookup docs",
                    "parameters": {"properties": {}},
                }
            ]

        async def disconnect(self):
            return None

    _always_allow = lambda user, connection, user_group_ids=None: True
    monkeypatch.setattr(
        ow_tools,
        "has_tool_server_access",
        _always_allow,
        raising=False,
    )
    try:
        import open_webui.utils.access_control as ow_ac
        monkeypatch.setattr(ow_ac, "has_connection_access", _always_allow, raising=False)
    except ImportError:
        pass
    monkeypatch.setattr(ow_env, "ENABLE_FORWARD_USER_INFO_HEADERS", False, raising=False)

    utils_package = open_webui.utils
    fake_misc_module = ModuleType("open_webui.utils.misc")
    fake_misc_module.is_string_allowed = lambda value, allowlist: True
    monkeypatch.setitem(sys.modules, "open_webui.utils.misc", fake_misc_module)
    monkeypatch.setattr(utils_package, "misc", fake_misc_module, raising=False)

    fake_headers_module = ModuleType("open_webui.utils.headers")
    fake_headers_module.include_user_info_headers = lambda headers, user: headers
    monkeypatch.setitem(sys.modules, "open_webui.utils.headers", fake_headers_module)
    monkeypatch.setattr(utils_package, "headers", fake_headers_module, raising=False)

    fake_mcp_package = ModuleType("open_webui.utils.mcp")
    fake_mcp_client_module = ModuleType("open_webui.utils.mcp.client")
    fake_mcp_client_module.MCPClient = FakeMCPClient
    monkeypatch.setitem(sys.modules, "open_webui.utils.mcp", fake_mcp_package)
    monkeypatch.setitem(sys.modules, "open_webui.utils.mcp.client", fake_mcp_client_module)
    monkeypatch.setattr(utils_package, "mcp", fake_mcp_package, raising=False)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(
                    TOOL_SERVER_CONNECTIONS=[
                        {
                            "type": "mcp",
                            "url": "https://mcp.example.com",
                            "auth_type": "oauth_2.1_static",
                            "headers": {"X-Test": "1"},
                            "config": {"enable": True},
                            "info": {"id": "suite:ctx7"},
                        }
                    ],
                ),
                oauth_client_manager=FakeOAuthClientManager(),
            )
        )
    )

    tools_dict, mcp_clients = await sub_agent.resolve_mcp_tools(
        request=request,
        user=SimpleNamespace(id="u1", role="user"),
        mcp_tool_ids=["server:mcp:suite:ctx7"],
        extra_params={},
        metadata={},
        debug=False,
    )

    # OAuth lookup uses the trailing colon segment to match Open WebUI core,
    # while the mcp_clients cache key below stays on the full id.
    assert oauth_calls == [("u1", "mcp:ctx7")]
    assert connected_headers == [
        {
            "url": "https://mcp.example.com",
            "headers": {
                "Authorization": "Bearer static-token",
                "X-Test": "1",
            },
        }
    ]
    assert "suite_ctx7_lookup_docs" in tools_dict
    assert isinstance(mcp_clients, dict)
    assert "suite:ctx7" in mcp_clients


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
    tools_dict, _ = await sub_agent.load_sub_agent_tools(
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
    tools_dict, _ = await sub_agent.load_sub_agent_tools(
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
    tools_dict, _ = await sub_agent.load_sub_agent_tools(
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
    tools_dict, _ = await sub_agent.load_sub_agent_tools(
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
    _, _ = await sub_agent.load_sub_agent_tools(
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
    tools_dict, _ = await multi_model_council.build_tools_dict(
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
    tools_dict, _ = await multi_model_council.build_tools_dict(
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
async def test_multi_model_council_resolves_terminal_once_before_parallel_members(
    monkeypatch, dummy_request, mock_user
):
    resolve_calls = []
    build_tools_terminal_ids = []

    async def fake_resolve_terminal_id_from_request_and_metadata(
        *, request, metadata, debug=False
    ):
        resolve_calls.append(1)
        return "term-shared"

    async def fake_get_available_models(request, user):
        return [{"id": "model-a"}, {"id": "model-b"}]

    async def fake_build_tools_dict(
        *,
        request,
        model,
        metadata,
        user,
        valves,
        extra_params,
        tool_id_list,
        excluded_tool_ids,
        resolved_terminal_id=None,
        resolved_direct_tool_servers=None,
    ):
        build_tools_terminal_ids.append(resolved_terminal_id)
        return {}, {}

    async def fake_run_agent_loop(**kwargs):
        return json.dumps({"vote": "A", "reasoning": "ok"})

    async def fake_event_emitter(event: dict):
        return None

    monkeypatch.setattr(
        multi_model_council,
        "resolve_terminal_id_from_request_and_metadata",
        fake_resolve_terminal_id_from_request_and_metadata,
    )
    monkeypatch.setattr(
        multi_model_council,
        "get_available_models",
        fake_get_available_models,
    )
    monkeypatch.setattr(
        multi_model_council,
        "build_tools_dict",
        fake_build_tools_dict,
    )
    monkeypatch.setattr(
        multi_model_council,
        "run_agent_loop",
        fake_run_agent_loop,
    )

    setattr(dummy_request.app.state, "MODELS", {"model-a": {}, "model-b": {}})

    user_payload = {
        **mock_user,
        "last_active_at": 0,
        "updated_at": 0,
        "created_at": 0,
    }
    metadata = {"tool_ids": [], "features": {}}

    tool = multi_model_council.Tools()
    result_json = await tool.council_decide(
        proposition="pick one",
        option_a="A",
        option_b="B",
        models="model-a,model-b",
        __user__=user_payload,
        __request__=dummy_request,
        __metadata__=metadata,
        __event_emitter__=fake_event_emitter,
    )

    payload = json.loads(result_json)
    assert payload["decision"] == "A"
    assert len(resolve_calls) == 1
    assert len(build_tools_terminal_ids) == 2
    assert all(tid == "term-shared" for tid in build_tools_terminal_ids)
    assert metadata["terminal_id"] == "term-shared"


def test_multi_model_council_parse_model_ids_deduplicates_values():
    assert multi_model_council.parse_model_ids("model-a, model-a, model-b") == [
        "model-a",
        "model-b",
    ]


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
    tools_dict, _ = await multi_model_council.build_tools_dict(
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
    tools_dict, _ = await magi_decision_support.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=magi_decision_support.Tools().valves,
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
    tools_dict, _ = await magi_decision_support.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=magi_decision_support.Tools().valves,
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
    tools_dict, _ = await magi_decision_support.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=magi_decision_support.Tools().valves,
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


@pytest.mark.asyncio
async def test_sub_agent_load_tools_includes_direct_tool_servers(monkeypatch, dummy_request):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        return {}

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)

    valves = sub_agent.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {
        "tool_ids": [],
        "features": {},
        "tool_servers": [
            {
                "url": "https://direct.example.com",
                "specs": [
                    {
                        "name": "direct_run",
                        "parameters": {"properties": {"command": {"type": "string"}}},
                    }
                ],
            }
        ],
    }
    tools_dict, _ = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params={"__metadata__": metadata},
        self_tool_id=None,
    )

    assert "direct_run" in tools_dict
    assert tools_dict["direct_run"]["direct"] is True
    assert tools_dict["direct_run"]["server"]["url"] == "https://direct.example.com"


@pytest.mark.asyncio
async def test_sub_agent_load_tools_collects_direct_tool_server_system_prompts(
    monkeypatch, dummy_request
):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        return {}

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)

    valves = sub_agent.Tools().valves
    metadata = {
        "tool_ids": [],
        "features": {},
        "tool_servers": [
            {
                "url": "https://direct.example.com",
                "system_prompt": "Direct prompt A",
                "specs": [
                    {
                        "name": "direct_run",
                        "parameters": {"properties": {"command": {"type": "string"}}},
                    }
                ],
            }
        ],
    }
    extra_params = {"__metadata__": metadata}
    _, _ = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params=extra_params,
        self_tool_id=None,
    )

    assert extra_params["__direct_tool_server_system_prompts__"] == ["Direct prompt A"]


@pytest.mark.asyncio
async def test_sub_agent_direct_tool_without_specs_is_not_loaded(monkeypatch, dummy_request):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    async def fake_get_terminal_tools(request, terminal_id, user, extra_params):
        return {}

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)
    monkeypatch.setattr(ow_tools, "get_terminal_tools", fake_get_terminal_tools)

    valves = sub_agent.Tools().valves
    valves.ENABLE_TERMINAL_TOOLS = True

    metadata = {
        "tool_ids": [],
        "features": {},
        "tool_servers": [{"url": "https://direct-hydrate.example.com"}],
    }
    extra_params = {"__metadata__": metadata}
    tools_dict, _ = await sub_agent.load_sub_agent_tools(
        request=dummy_request,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        metadata=metadata,
        model={},
        extra_params=extra_params,
        self_tool_id=None,
    )

    assert "hydrated_run" not in tools_dict
    assert "__direct_tool_server_system_prompts__" not in extra_params


@pytest.mark.asyncio
async def test_sub_agent_execute_direct_tool_uses_event_call_and_session_id():
    events = []
    execute_calls = []

    async def event_emitter(event: dict):
        events.append(event)

    async def event_call(payload: dict):
        execute_calls.append(payload)
        return [{"exists": True, "path": "/tmp/direct.txt"}, {"Content-Type": "application/json"}]

    tool_call = {
        "id": "tc-direct-sub-agent",
        "function": {
            "name": "display_file",
            "arguments": json.dumps({"path": "/tmp/direct.txt"}),
        },
    }
    tools_dict = {
        "display_file": {
            "spec": make_spec("path"),
            "direct": True,
            "server": {"url": "https://direct.example.com"},
            "type": "direct",
        }
    }

    result = await sub_agent.execute_tool_call(
        tool_call=tool_call,
        tools_dict=tools_dict,
        extra_params={
            "__event_call__": event_call,
            "__metadata__": {"session_id": "sess-sub-agent"},
        },
        event_emitter=event_emitter,
    )

    payload = json.loads(result["content"])
    assert payload["path"] == "/tmp/direct.txt"
    assert len(execute_calls) == 1
    assert execute_calls[0]["type"] == "execute:tool"
    assert execute_calls[0]["data"]["session_id"] == "sess-sub-agent"
    assert any(event.get("type") == "terminal:display_file" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("_module_name", "module", "name_key", "params_key"),
    TERMINAL_EVENT_MODULE_CASES,
)
async def test_emit_terminal_tool_event_supports_replace_file_content(
    _module_name, module, name_key, params_key
):
    events = []

    async def event_emitter(event: dict):
        events.append(event)

    kwargs = {
        name_key: "replace_file_content",
        params_key: {"path": "/tmp/replaced.txt"},
        "tool_result": {"ok": True},
        "event_emitter": event_emitter,
    }

    await module.emit_terminal_tool_event(**kwargs)

    assert events == [
        {"type": "terminal:replace_file_content", "data": {"path": "/tmp/replaced.txt"}}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("_module_name", "module", "name_key", "params_key"),
    TERMINAL_EVENT_MODULE_CASES,
)
async def test_emit_terminal_tool_event_supports_run_command(
    _module_name, module, name_key, params_key
):
    events = []

    async def event_emitter(event: dict):
        events.append(event)

    kwargs = {
        name_key: "run_command",
        params_key: {"command": "pwd"},
        "tool_result": {"ok": True},
        "event_emitter": event_emitter,
    }

    await module.emit_terminal_tool_event(**kwargs)

    assert events == [{"type": "terminal:run_command", "data": {}}]


@pytest.mark.asyncio
async def test_sub_agent_execute_direct_tool_without_event_call_fails_without_terminal_event():
    events = []

    async def event_emitter(event: dict):
        events.append(event)

    tool_call = {
        "id": "tc-direct-no-event-call",
        "function": {
            "name": "display_file",
            "arguments": json.dumps({"path": "/tmp/direct-no-call.txt"}),
        },
    }
    tools_dict = {
        "display_file": {
            "spec": make_spec("path"),
            "direct": True,
            "server": {"url": "https://direct.example.com"},
            "type": "direct",
        }
    }

    result = await sub_agent.execute_tool_call(
        tool_call=tool_call,
        tools_dict=tools_dict,
        extra_params={"__metadata__": {"session_id": "sess-sub-agent"}},
        event_emitter=event_emitter,
    )

    assert result["content"].startswith("Error:")
    assert not any(event.get("type", "").startswith("terminal:") for event in events)


@pytest.mark.asyncio
async def test_multi_model_council_build_tools_dict_includes_direct_tools(dummy_request):
    valves = multi_model_council.Tools().valves
    metadata = {
        "tool_ids": [],
        "features": {},
        "tool_servers": [
            {
                "url": "https://direct-council.example.com",
                "specs": [
                    {
                        "name": "direct_lookup",
                        "parameters": {"properties": {"query": {"type": "string"}}},
                    }
                ],
            }
        ],
    }
    extra_params = {"__metadata__": metadata}
    tools_dict, _ = await multi_model_council.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        extra_params=extra_params,
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert "direct_lookup" in tools_dict
    assert tools_dict["direct_lookup"]["direct"] is True


@pytest.mark.asyncio
async def test_multi_model_council_build_tools_dict_collects_direct_tool_server_system_prompts(
    dummy_request,
):
    valves = multi_model_council.Tools().valves
    metadata = {
        "tool_ids": [],
        "features": {},
        "tool_servers": [
            {
                "url": "https://direct-council.example.com",
                "system_prompt": "Direct prompt Council",
                "specs": [
                    {
                        "name": "direct_lookup",
                        "parameters": {"properties": {"query": {"type": "string"}}},
                    }
                ],
            }
        ],
    }
    extra_params = {"__metadata__": metadata}
    await multi_model_council.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        extra_params=extra_params,
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert extra_params["__direct_tool_server_system_prompts__"] == ["Direct prompt Council"]


@pytest.mark.asyncio
async def test_multi_model_council_build_tools_dict_ignores_unloaded_direct_tool_prompts(
    dummy_request,
):
    valves = multi_model_council.Tools().valves
    metadata = {
        "tool_ids": [],
        "features": {},
        "tool_servers": [
            {
                "url": "https://direct-council.example.com",
                "system_prompt": "Ghost prompt Council",
            }
        ],
    }
    extra_params = {"__metadata__": metadata}
    tools_dict, _ = await multi_model_council.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=valves,
        extra_params=extra_params,
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert "direct_lookup" not in tools_dict
    assert "__direct_tool_server_system_prompts__" not in extra_params


@pytest.mark.asyncio
async def test_magi_build_tools_dict_includes_direct_tools(dummy_request):
    metadata = {
        "tool_ids": [],
        "features": {},
        "tool_servers": [
            {
                "url": "https://direct-magi.example.com",
                "specs": [
                    {
                        "name": "direct_search",
                        "parameters": {"properties": {"query": {"type": "string"}}},
                    }
                ],
            }
        ],
    }
    extra_params = {"__metadata__": metadata}
    tools_dict, _ = await magi_decision_support.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=magi_decision_support.Tools().valves,
        extra_params=extra_params,
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert "direct_search" in tools_dict
    assert tools_dict["direct_search"]["direct"] is True


@pytest.mark.asyncio
async def test_magi_build_tools_dict_collects_direct_tool_server_system_prompts(dummy_request):
    metadata = {
        "tool_ids": [],
        "features": {},
        "tool_servers": [
            {
                "url": "https://direct-magi.example.com",
                "system_prompt": "Direct prompt MAGI",
                "specs": [
                    {
                        "name": "direct_search",
                        "parameters": {"properties": {"query": {"type": "string"}}},
                    }
                ],
            }
        ],
    }
    extra_params = {"__metadata__": metadata}
    await magi_decision_support.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=magi_decision_support.Tools().valves,
        extra_params=extra_params,
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert extra_params["__direct_tool_server_system_prompts__"] == ["Direct prompt MAGI"]


@pytest.mark.asyncio
async def test_magi_build_tools_dict_ignores_unloaded_direct_tool_prompts(dummy_request):
    metadata = {
        "tool_ids": [],
        "features": {},
        "tool_servers": [
            {
                "url": "https://direct-magi.example.com",
                "system_prompt": "Ghost prompt MAGI",
            }
        ],
    }
    extra_params = {"__metadata__": metadata}
    tools_dict, _ = await magi_decision_support.build_tools_dict(
        request=dummy_request,
        model={},
        metadata=metadata,
        user=SimpleNamespace(id="u1"),
        valves=magi_decision_support.Tools().valves,
        extra_params=extra_params,
        tool_id_list=[],
        excluded_tool_ids=None,
    )

    assert "direct_search" not in tools_dict
    assert "__direct_tool_server_system_prompts__" not in extra_params


@pytest.mark.asyncio
async def test_parallel_tools_executes_direct_tool_via_event_call(
    monkeypatch, dummy_request, mock_user
):
    import open_webui.utils.tools as ow_tools

    async def fake_get_tools(request, tool_ids, user, extra_params):
        return {}

    def fake_get_builtin_tools(request, extra_params, features=None, model=None):
        return {}

    execute_calls = []

    async def event_call(payload: dict):
        execute_calls.append(payload)
        return [{"ok": True, "command": payload["data"]["params"]["command"]}, {}]

    monkeypatch.setattr(ow_tools, "get_tools", fake_get_tools)
    monkeypatch.setattr(ow_tools, "get_builtin_tools", fake_get_builtin_tools)

    tool = parallel_tools.Tools()
    user_payload = {
        **mock_user,
        "last_active_at": 0,
        "updated_at": 0,
        "created_at": 0,
    }
    result_json = await tool.run_tools_parallel(
        tool_calls=[{"name": "direct_run", "args": {"command": "pwd"}}],
        __user__=user_payload,
        __request__=dummy_request,
        __metadata__={
            "tool_ids": [],
            "features": {},
            "session_id": "sess-parallel",
            "tool_servers": [
                {
                    "url": "https://direct-parallel.example.com",
                    "specs": [
                        {
                            "name": "direct_run",
                            "parameters": {"properties": {"command": {"type": "string"}}},
                        }
                    ],
                }
            ],
        },
        __event_call__=event_call,
    )

    payload = json.loads(result_json)
    assert payload["results"][0]["tool_name"] == "direct_run"
    assert payload["results"][0]["result"]["ok"] is True
    assert execute_calls[0]["type"] == "execute:tool"
    assert execute_calls[0]["data"]["session_id"] == "sess-parallel"


@pytest.mark.parametrize(("module_name", "module"), RESULT_HELPER_MODULES)
def test_normalize_terminal_tools_result_supports_tuple_return(module_name, module):
    extra_params = {}
    terminal_tools = {
        "run_command": {
            "callable": lambda **kwargs: None,
            "spec": make_spec("command"),
            "type": "terminal",
            "tool_id": "terminal:test",
        }
    }

    normalized = module.normalize_terminal_tools_result(
        terminal_tools_result=(terminal_tools, "Use the terminal working directory."),
        extra_params=extra_params,
    )

    assert normalized == terminal_tools, module_name
    assert extra_params["__terminal_system_prompt__"] == "Use the terminal working directory."


@pytest.mark.parametrize(("module_name", "module"), RESULT_HELPER_MODULES)
def test_citation_tools_include_view_file(module_name, module):
    assert "view_file" in module.CITATION_TOOLS, module_name


@pytest.mark.parametrize(("module_name", "module"), BUILTIN_KNOWLEDGE_MODULES)
def test_builtin_knowledge_category_includes_list_knowledge(module_name, module):
    assert "list_knowledge" in module.BUILTIN_TOOL_CATEGORIES["knowledge"], module_name
