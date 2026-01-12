import pytest

import httpx

from functions.filter.token_usage_display import Filter


@pytest.fixture(autouse=True)
def _clear_litellm_env(monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)


@pytest.fixture
def events():
    """Events list fixture for capturing emitted events."""
    return []


@pytest.fixture
def emitter(events):
    """Event emitter fixture."""
    async def _emitter(evt):
        events.append(evt)
    return _emitter


@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Factory fixture for mocking httpx.AsyncClient with custom routes."""
    def _mock_client(get_routes: dict[str, tuple[int, object]], capture: dict):
        def _mock_client_factory(*args, **kwargs):
            return _MockAsyncClient(get_routes=get_routes, capture=capture)
        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory)
    return _mock_client


@pytest.fixture
def request_with_prefix_ids():
    """Factory fixture for creating request objects with prefix IDs."""
    def _make_request(prefix_ids: list[str]):
        return _make_request_with_prefix_ids(prefix_ids)
    return _make_request


class _MockResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = "mock"

    def json(self):
        return self._payload


class _MockAsyncClient:
    def __init__(self, *, get_routes: dict[str, tuple[int, object]], capture: dict):
        self._get_routes = get_routes
        self._capture = capture

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, headers=None):
        self._capture.setdefault("get", []).append({"url": url, "headers": headers or {}})
        status, payload = self._get_routes.get(url, (404, {"error": "not found"}))
        return _MockResponse(status, payload)


class _Dummy:
    pass


def _make_request_with_prefix_ids(prefix_ids: list[str]):
    req = _Dummy()
    req.app = _Dummy()
    req.app.state = _Dummy()
    req.app.state.config = _Dummy()
    # Open WebUI stores prefix_id in these config dicts.
    req.app.state.config.OPENAI_API_CONFIGS = {
        str(i): {"prefix_id": pid} for i, pid in enumerate(prefix_ids)
    }
    req.app.state.config.OLLAMA_API_CONFIGS = {}
    return req


@pytest.mark.asyncio
async def test_emits_status_from_top_level_usage(emitter, events):
    f = Filter()

    body = {
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
    )

    assert any(e.get("type") == "status" for e in events)
    status = next(e for e in events if e.get("type") == "status")
    assert status["data"]["done"] is True
    assert "Tokens: 30 / 100,000" in status["data"]["description"]
    assert "Prompt" not in status["data"]["description"]
    assert "Completion" not in status["data"]["description"]


def test_inlet_forces_stream_include_usage_when_enabled():
    f = Filter()
    body = {"model": "gpt-4o-mini", "stream": True}

    out = f.inlet(body, __user__={"valves": {"enabled": True}})

    assert out["stream"] is True
    assert isinstance(out.get("stream_options"), dict)
    assert out["stream_options"]["include_usage"] is True


def test_inlet_does_not_inject_when_stream_false():
    f = Filter()
    body = {"model": "gpt-4o-mini", "stream": False}

    out = f.inlet(body, __user__={"valves": {"enabled": True}})

    assert "stream_options" not in out


def test_inlet_does_not_inject_when_disabled():
    f = Filter()
    body = {"model": "gpt-4o-mini", "stream": True}

    out = f.inlet(body, __user__={"valves": {"enabled": False}})

    assert out == body


def test_inlet_does_not_inject_when_force_include_usage_false():
    f = Filter()
    f.valves.force_include_usage = False
    body = {"model": "gpt-4o-mini", "stream": True}

    out = f.inlet(body, __user__={"valves": {"enabled": True}})

    assert "stream_options" not in out


def test_inlet_preserves_existing_stream_options():
    f = Filter()
    body = {"model": "gpt-4o-mini", "stream": True, "stream_options": {"other": "value"}}

    out = f.inlet(body, __user__={"valves": {"enabled": True}})

    assert out["stream_options"]["include_usage"] is True
    assert out["stream_options"]["other"] == "value"


def test_inlet_returns_body_when_not_dict():
    f = Filter()
    body = "not a dict"

    out = f.inlet(body, __user__={"valves": {"enabled": True}})

    assert out == body


@pytest.mark.asyncio
async def test_fallback_default_max_input_tokens_used_when_proxy_not_configured(emitter, events):
    f = Filter()
    f.valves.proxy_url = ""

    body = {"model": "gpt-4o-mini", "usage": {"total_tokens": 60000}}

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 50}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 60,000 / 100,000" in desc
    assert "(60.0%)" in desc


@pytest.mark.asyncio
async def test_fallback_model_override_applies(emitter, events):
    f = Filter()
    f.valves.proxy_url = ""
    f.valves.model_max_input_tokens_overrides = "gpt-4o-mini=200000"

    body = {"model": "gpt-4o-mini", "usage": {"total_tokens": 120000}}

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 50}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 120,000 / 200,000" in desc
    assert "(60.0%)" in desc


@pytest.mark.asyncio
async def test_fallback_model_override_applies_to_stripped_name(emitter, events, request_with_prefix_ids):
    f = Filter()
    f.valves.proxy_url = ""
    f.valves.model_max_input_tokens_overrides = "gpt-4o-mini=200000"

    body = {"model": "LiteLLM.gpt-4o-mini", "usage": {"total_tokens": 120000}}

    request = request_with_prefix_ids(["LiteLLM"])
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 50}},
        __request__=request,
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 120,000 / 200,000" in desc
    assert "(60.0%)" in desc


@pytest.mark.asyncio
async def test_emits_status_from_message_usage_by_id(emitter, events):
    f = Filter()

    body = {
        "id": "msg_assistant_1",
        "model": "gpt-4o-mini",
        "messages": [
            {"id": "msg_user_1", "role": "user", "content": "hi"},
            {
                "id": "msg_assistant_1",
                "role": "assistant",
                "content": "hello",
                "usage": {"input_tokens": 7, "output_tokens": 8},
            },
        ],
    }

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "15" in desc
    assert "Tokens (" not in desc
    assert "Prompt" not in desc
    assert "Completion" not in desc


@pytest.mark.asyncio
async def test_no_emit_when_disabled(emitter, events):
    f = Filter()

    body = {
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    await f.outlet(body, __event_emitter__=emitter, __user__={"valves": {"enabled": False}})

    assert events == []


@pytest.mark.asyncio
async def test_includes_max_input_tokens_and_percent_and_strips_prefix(
    mock_httpx_client, emitter, events, request_with_prefix_ids
):
    capture = {}
    base = "http://proxy:4000"

    # LiteLLM's /model/info returns base model ids; Open WebUI may pass a prefixed id.
    get_routes = {
        f"{base}/model/info": (
            200,
            {
                "data": [
                    {"model_name": "gc/gpt-4.1", "model_info": {"max_input_tokens": 10000}}
                ]
            },
        )
    }

    mock_httpx_client(get_routes, capture)

    f = Filter()
    f.valves.proxy_url = base
    f.valves.api_key = ""

    body = {
        "model": "LiteLLM.gc/gpt-4.1",
        "usage": {"total_tokens": 8967},
    }

    request = request_with_prefix_ids(["LiteLLM"])
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True}},
        __request__=request,
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 8,967 / 10,000" in desc
    assert "(89.7%)" in desc

    assert capture["get"][0]["url"].endswith("/model/info")


@pytest.mark.asyncio
async def test_does_not_strip_unprefixed_model_with_dot(
    mock_httpx_client, emitter, events, request_with_prefix_ids
):
    capture = {}
    base = "http://proxy:4000"

    get_routes = {
        f"{base}/model/info": (
            200,
            {
                "data": [
                    {"model_name": "gpt-4.1", "model_info": {"max_input_tokens": 10000}}
                ]
            },
        )
    }

    mock_httpx_client(get_routes, capture)

    f = Filter()
    f.valves.proxy_url = base
    f.valves.api_key = ""

    body = {
        "model": "gpt-4.1",
        "usage": {"total_tokens": 5000},
    }

    request = request_with_prefix_ids(["LiteLLM"])
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True}},
        __request__=request,
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 5,000 / 10,000" in desc
    assert "(50.0%)" in desc


@pytest.mark.asyncio
async def test_does_not_break_claude_model_id_with_dot(
    mock_httpx_client, emitter, events, request_with_prefix_ids
):
    capture = {}
    base = "http://proxy:4000"

    get_routes = {
        f"{base}/model/info": (
            200,
            {
                "data": [
                    {
                        "model_name": "claude-3.5-sonnet",
                        "model_info": {"max_input_tokens": 200000},
                    }
                ]
            },
        )
    }

    mock_httpx_client(get_routes, capture)

    f = Filter()
    f.valves.proxy_url = base

    body = {
        "model": "claude-3.5-sonnet",
        "usage": {"total_tokens": 1000},
    }

    request = request_with_prefix_ids(["LiteLLM"])
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
        __request__=request,
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 1,000 / 200,000" in desc


@pytest.mark.asyncio
async def test_emits_when_percent_exceeds_display_threshold(
    mock_httpx_client, emitter, events, request_with_prefix_ids
):
    capture = {}
    base = "http://proxy:4000"

    get_routes = {
        f"{base}/model/info": (
            200,
            {"data": [{"model_name": "gpt-4o", "model_info": {"max_input_tokens": 100}}]},
        )
    }

    mock_httpx_client(get_routes, capture)

    f = Filter()
    f.valves.proxy_url = base

    body = {
        "model": "gpt-4o",
        "usage": {"total_tokens": 90},
    }

    request = request_with_prefix_ids([])
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 80}},
        __request__=request,
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert desc.startswith("Tokens:")
    assert "(90.0%)" in desc


@pytest.mark.asyncio
async def test_does_not_emit_when_below_display_threshold(
    mock_httpx_client, emitter, events, request_with_prefix_ids
):
    capture = {}
    base = "http://proxy:4000"

    get_routes = {
        f"{base}/model/info": (
            200,
            {"data": [{"model_name": "gpt-4o", "model_info": {"max_input_tokens": 100}}]},
        )
    }

    mock_httpx_client(get_routes, capture)

    f = Filter()
    f.valves.proxy_url = base

    body = {
        "model": "gpt-4o",
        "usage": {"total_tokens": 50},
    }

    request = request_with_prefix_ids([])
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 80}},
        __request__=request,
    )

    assert not any(e.get("type") == "status" for e in events)


@pytest.mark.asyncio
async def test_outlet_returns_body_when_no_event_emitter():
    f = Filter()

    body = {
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    result = await f.outlet(
        body,
        __event_emitter__=None,
        __user__={"valves": {"enabled": True}},
    )

    assert result == body


@pytest.mark.asyncio
async def test_outlet_returns_body_when_no_usage(emitter, events):
    f = Filter()

    body = {"model": "gpt-4o-mini"}

    result = await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True}},
    )

    assert result == body
    assert events == []


@pytest.mark.asyncio
async def test_handles_input_output_token_naming(emitter, events):
    f = Filter()

    body = {
        "model": "claude-3-sonnet",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 150 / 100,000" in desc


@pytest.mark.asyncio
async def test_v1_model_info_fallback(mock_httpx_client, emitter, events, request_with_prefix_ids):
    capture = {}
    base = "http://proxy:4000"

    # /model/info fails, but /v1/model/info succeeds
    get_routes = {
        f"{base}/model/info": (500, {"error": "server error"}),
        f"{base}/v1/model/info": (
            200,
            {
                "data": [
                    {"model_name": "gpt-4o", "model_info": {"max_input_tokens": 50000}}
                ]
            },
        ),
    }

    mock_httpx_client(get_routes, capture)

    f = Filter()
    f.valves.proxy_url = base

    body = {
        "model": "gpt-4o",
        "usage": {"total_tokens": 25000},
    }

    request = request_with_prefix_ids([])
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True}},
        __request__=request,
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 25,000 / 50,000" in desc


@pytest.mark.asyncio
async def test_model_group_info_fallback(mock_httpx_client, emitter, events, request_with_prefix_ids):
    capture = {}
    base = "http://proxy:4000"

    # Both model/info endpoints fail, but model_group/info succeeds
    get_routes = {
        f"{base}/model/info": (404, {"error": "not found"}),
        f"{base}/v1/model/info": (404, {"error": "not found"}),
        f"{base}/model_group/info": (
            200,
            {
                "data": [
                    {"model_group": "gpt-4o", "max_input_tokens": 40000}
                ]
            },
        ),
    }

    mock_httpx_client(get_routes, capture)

    f = Filter()
    f.valves.proxy_url = base

    body = {
        "model": "gpt-4o",
        "usage": {"total_tokens": 20000},
    }

    request = request_with_prefix_ids([])
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True}},
        __request__=request,
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 20,000 / 40,000" in desc


@pytest.mark.asyncio
async def test_cache_ttl_behavior(monkeypatch):
    capture = {}
    base = "http://proxy:4000"

    get_routes = {
        f"{base}/model/info": (
            200,
            {
                "data": [
                    {"model_name": "gpt-4o", "model_info": {"max_input_tokens": 30000}}
                ]
            },
        )
    }

    def _mock_client_factory(*args, **kwargs):
        return _MockAsyncClient(get_routes=get_routes, capture=capture)

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory)

    f = Filter()
    f.valves.proxy_url = base
    f.valves.model_info_ttl_seconds = 300

    events = []

    async def emitter(evt):
        events.append(evt)

    body = {
        "model": "gpt-4o",
        "usage": {"total_tokens": 15000},
    }

    request = _make_request_with_prefix_ids([])
    
    # First call should hit the API
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True}},
        __request__=request,
    )

    first_call_count = len(capture.get("get", []))
    
    # Second call should use cache
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True}},
        __request__=request,
    )

    second_call_count = len(capture.get("get", []))
    
    # Should not make additional HTTP calls due to cache
    assert first_call_count == second_call_count


@pytest.mark.asyncio
async def test_handles_http_error_gracefully(monkeypatch):
    base = "http://proxy:4000"

    class ErrorClient:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc, tb):
            return False
        
        async def get(self, url: str, headers=None):
            raise httpx.ConnectError("Connection failed")

    def _mock_error_client(*args, **kwargs):
        return ErrorClient()

    monkeypatch.setattr(httpx, "AsyncClient", _mock_error_client)

    f = Filter()
    f.valves.proxy_url = base

    events = []

    async def emitter(evt):
        events.append(evt)

    body = {
        "model": "gpt-4o",
        "usage": {"total_tokens": 50000},
    }

    # Should fall back to default without crashing
    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 40}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    # Should use default 100,000 tokens
    assert "Tokens: 50,000 / 100,000" in desc


@pytest.mark.asyncio
async def test_handles_event_emitter_exception(monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)

    f = Filter()
    f.valves.proxy_url = ""

    async def failing_emitter(evt):
        raise RuntimeError("Emitter failed")

    body = {
        "model": "gpt-4o-mini",
        "usage": {"total_tokens": 30},
    }

    # Should not crash when emitter raises exception
    result = await f.outlet(
        body,
        __event_emitter__=failing_emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
    )

    assert result == body


@pytest.mark.asyncio
async def test_handles_nested_usage_structure():
    """Test that nested usage structure (e.g., LiteLLM) is handled correctly."""
    f = Filter()

    events = []

    async def emitter(evt):
        events.append(evt)

    body = {
        "model": "gpt-4o",
        "usage": {
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            }
        },
    }

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 150 / 100,000" in desc


@pytest.mark.asyncio
async def test_messages_usage_fallback_to_last_message():
    """Test that when no id match, uses last message with usage."""
    f = Filter()

    events = []

    async def emitter(evt):
        events.append(evt)

    body = {
        "id": "msg_assistant_2",  # ID doesn't match any message
        "model": "gpt-4o",
        "messages": [
            {"id": "msg_user_1", "role": "user", "content": "hi"},
            {
                "id": "msg_assistant_1",
                "role": "assistant",
                "content": "hello",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
            {"id": "msg_user_2", "role": "user", "content": "bye"},
            {
                "id": "msg_assistant_3",
                "role": "assistant",
                "content": "goodbye",
                "usage": {"input_tokens": 15, "output_tokens": 25},
            },
        ],
    }

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    # Should use the last message with usage (40 tokens)
    assert "Tokens: 40 / 100,000" in desc


@pytest.mark.asyncio
async def test_model_override_newline_separated():
    """Test that model overrides support newline-separated format."""
    f = Filter()
    f.valves.proxy_url = ""
    f.valves.model_max_input_tokens_overrides = "model-a=50000\nmodel-b=75000\n"

    events = []

    async def emitter(evt):
        events.append(evt)

    body = {"model": "model-b", "usage": {"total_tokens": 60000}}

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 60,000 / 75,000" in desc


@pytest.mark.asyncio
async def test_model_override_ignores_invalid_entries():
    """Test that invalid model override entries are silently ignored."""
    f = Filter()
    f.valves.proxy_url = ""
    f.valves.model_max_input_tokens_overrides = "model-a=50000,invalid,model-b=abc,model-c=75000"

    events = []

    async def emitter(evt):
        events.append(evt)

    body = {"model": "model-c", "usage": {"total_tokens": 50000}}

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 50,000 / 75,000" in desc


@pytest.mark.asyncio
async def test_usage_with_only_prompt_and_completion_tokens():
    """Test that total_tokens is calculated when only prompt and completion are provided."""
    f = Filter()

    events = []

    async def emitter(evt):
        events.append(evt)

    body = {
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 100, "completion_tokens": 200},
    }

    await f.outlet(
        body,
        __event_emitter__=emitter,
        __user__={"valves": {"enabled": True, "display_threshold_percent": 0}},
    )

    status = next(e for e in events if e.get("type") == "status")
    desc = status["data"]["description"]
    assert "Tokens: 300 / 100,000" in desc


