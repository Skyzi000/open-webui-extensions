from __future__ import annotations

from types import SimpleNamespace
import asyncio
import copy
import json
import sys
import types

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

from functions.pipe import auto_compaction_pipe as mod


def install_fake_open_webui_user_model(monkeypatch):
    class FakeUserModel:
        def __init__(self, **data):
            self.__dict__.update(data)

    users_module = types.ModuleType("open_webui.models.users")
    users_module.UserModel = FakeUserModel
    monkeypatch.setitem(sys.modules, "open_webui.models.users", users_module)
    return FakeUserModel


def test_target_filter_excludes_own_wrappers_and_arena_but_allows_other_pipes():
    models = {
        "gpt-4.1": {"id": "gpt-4.1", "name": "GPT"},
        mod.build_wrapper_model_id("auto_compact", "gpt-4.1"): {
            "id": mod.build_wrapper_model_id("auto_compact", "gpt-4.1"),
            "name": "wrapped",
            "pipe": {"type": "pipe"},
        },
        "arena-model": {"id": "arena-model", "name": "Arena", "owned_by": "arena", "arena": True},
        "other_pipe.child": {"id": "other_pipe.child", "name": "Other Pipe", "pipe": {"type": "pipe"}},
    }

    targets = mod.filter_target_models(models.values(), mod.Pipe.Valves())

    assert [m["id"] for m in targets] == ["gpt-4.1", "other_pipe.child"]


def test_target_filter_excludes_presets_based_on_own_wrappers():
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "gpt-4.1")
    models = [
        {"id": "gpt-4.1", "name": "GPT"},
        {"id": "compact-preset", "name": "Compact Preset", "info": {"base_model_id": wrapper_id}},
        {"id": "top-level-compact-preset", "name": "Compact Preset 2", "base_model_id": wrapper_id},
        {"id": "normal-preset", "name": "Normal Preset", "info": {"base_model_id": "gpt-4.1"}},
    ]

    targets = mod.filter_target_models(models, mod.Pipe.Valves())

    assert [m["id"] for m in targets] == ["gpt-4.1", "normal-preset"]


@pytest.mark.asyncio
async def test_pipes_uses_runtime_registered_id_for_filtering_and_sync(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["pipe_function_id"] = kwargs["pipe_function_id"]
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={
                "target": {"id": "target", "name": "Target"},
                mod.build_wrapper_model_id("compact_alias", "target"): {
                    "id": mod.build_wrapper_model_id("compact_alias", "target"),
                    "name": "Own Wrapper",
                    "pipe": {"type": "pipe"},
                },
                mod.build_wrapper_model_id("auto_compact", "target"): {
                    "id": mod.build_wrapper_model_id("auto_compact", "target"),
                    "name": "Other Registered Wrapper",
                    "pipe": {"type": "pipe"},
                },
            }
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.Pipe, "__module__", "function_compact_alias")

    pipe = mod.Pipe()

    result = await pipe.pipes()

    assert captured == {
        "pipe_function_id": "compact_alias",
        "target_ids": ["target", mod.build_wrapper_model_id("auto_compact", "target")],
    }
    assert result == [
        {"id": "target", "name": "Compact: Target"},
        {
            "id": mod.build_wrapper_model_id("auto_compact", "target"),
            "name": "Compact: Other Registered Wrapper",
        },
    ]


@pytest.mark.asyncio
async def test_pipes_does_not_direct_fetch_provider_models_when_state_caches_remain_empty(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["pipe_function_id"] = kwargs["pipe_function_id"]
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def fetch_provider_models(*args, **kwargs):
        return [{"id": "direct-target", "name": "Direct Target"}]

    models_module = types.ModuleType("open_webui.utils.models")
    models_module.fetch_openai_models = fetch_provider_models
    models_module.fetch_ollama_models = fetch_provider_models

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={},
            BASE_MODELS=[],
            OPENAI_MODELS={},
            OLLAMA_MODELS={},
            config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", models_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod, "_provider_model_cache_wait_timeout_seconds", lambda: 0)

    result = await mod.Pipe().pipes()

    assert captured == {
        "pipe_function_id": "auto_compact",
        "target_ids": [],
    }
    assert result == []


@pytest.mark.asyncio
async def test_pipes_waits_for_sibling_provider_cache_population(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod, "_provider_model_cache_wait_timeout_seconds", lambda: 0.1)
    monkeypatch.setattr(mod, "PROVIDER_MODEL_CACHE_WAIT_POLL_SECONDS", 0.001)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    async def populate_cache():
        await asyncio.sleep(0)
        state.OPENAI_MODELS = {"sibling-target": {"id": "sibling-target", "name": "Sibling Target"}}

    task = asyncio.create_task(populate_cache())
    try:
        result = await mod.Pipe().pipes()
    finally:
        await task

    assert captured["target_ids"] == ["sibling-target"]
    assert result == [{"id": "sibling-target", "name": "Compact: Sibling Target"}]


@pytest.mark.asyncio
async def test_pipes_waits_for_all_enabled_provider_cache_population(monkeypatch):
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            state.OPENAI_MODELS = {"openai-target": {"id": "openai-target", "name": "OpenAI Target"}}
        elif sleep_calls == 2:
            state.OLLAMA_MODELS = {"ollama-target": {"model": "ollama-target", "name": "Ollama Target"}}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=True),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert sleep_calls == 2
    assert captured["target_ids"] == ["openai-target", "ollama-target"]
    assert result == [
        {"id": "openai-target", "name": "Compact: OpenAI Target"},
        {"id": "ollama-target", "name": "Compact: Ollama Target"},
    ]


@pytest.mark.asyncio
async def test_pipes_waits_when_one_of_multiple_provider_caches_is_still_empty(monkeypatch):
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        state.OLLAMA_MODELS = {"ollama-target": {"model": "ollama-target", "name": "Ollama Target"}}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={"openai-target": {"id": "openai-target", "name": "OpenAI Target"}},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=True),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert sleep_calls == 1
    assert captured["target_ids"] == ["openai-target", "ollama-target"]
    assert result == [
        {"id": "openai-target", "name": "Compact: OpenAI Target"},
        {"id": "ollama-target", "name": "Compact: Ollama Target"},
    ]


@pytest.mark.asyncio
async def test_pipes_does_not_wait_for_disabled_provider_cache(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        raise AssertionError("pipes() should not wait for disabled provider caches")

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={},
            BASE_MODELS=[],
            OPENAI_MODELS={"openai-target": {"id": "openai-target", "name": "OpenAI Target"}},
            OLLAMA_MODELS={},
            config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)

    result = await mod.Pipe().pipes()

    assert captured["target_ids"] == ["openai-target"]
    assert result == [{"id": "openai-target", "name": "Compact: OpenAI Target"}]


@pytest.mark.asyncio
async def test_pipes_does_not_wait_when_empty_provider_cache_already_refreshed(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        raise AssertionError("already-refreshed empty provider cache should not wait")

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={},
            BASE_MODELS=[],
            OPENAI_MODELS={},
            OLLAMA_MODELS={},
            config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)

    result = await mod.Pipe().pipes()

    assert captured["target_ids"] == []
    assert result == []


@pytest.mark.asyncio
async def test_provider_cache_pending_detection_requires_core_request_identity():
    def build_async_function(name: str, filename: str, body: str, **namespace):
        scope = dict(namespace)
        exec(compile(f"async def {name}(request):\n{body}", filename, "exec"), scope)
        return scope[name]

    event = asyncio.Event()
    fetch_openai_models = build_async_function(
        "fetch_openai_models",
        "/site-packages/open_webui/utils/models.py",
        "    await event.wait()\n",
        event=event,
    )
    get_function_models = build_async_function(
        "get_function_models",
        "/site-packages/open_webui/functions.py",
        "    return mod._provider_model_cache_refresh_pending_attrs(['OPENAI_MODELS'])\n",
        mod=mod,
    )
    request = object()
    other_request = object()

    task = asyncio.create_task(fetch_openai_models(other_request))
    await asyncio.sleep(0)
    try:
        assert mod._provider_model_cache_refresh_pending_attrs(["OPENAI_MODELS"]) == set()
        function_task = asyncio.create_task(get_function_models(request))
        assert await function_task == set()
    finally:
        event.set()
        await task

    event = asyncio.Event()
    fetch_openai_models = build_async_function(
        "fetch_openai_models",
        "/site-packages/open_webui/utils/models.py",
        "    await event.wait()\n",
        event=event,
    )
    task = asyncio.create_task(fetch_openai_models(request))
    await asyncio.sleep(0)
    try:
        function_task = asyncio.create_task(get_function_models(request))
        assert await function_task == {"OPENAI_MODELS"}
    finally:
        event.set()
        await task


@pytest.mark.asyncio
async def test_pipes_does_not_wait_for_other_request_provider_refresh(monkeypatch):
    captured = {}

    def build_async_function(name: str, filename: str, body: str, **namespace):
        scope = dict(namespace)
        exec(compile(f"async def {name}(request):\n{body}", filename, "exec"), scope)
        return scope[name]

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        raise AssertionError("pipes() should not wait for another request's provider refresh")

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)

    event = asyncio.Event()
    fetch_openai_models = build_async_function(
        "fetch_openai_models",
        "/site-packages/open_webui/utils/models.py",
        "    await event.wait()\n",
        event=event,
    )
    get_function_models = build_async_function(
        "get_function_models",
        "/site-packages/open_webui/functions.py",
        "    return await pipe.pipes()\n",
        pipe=mod.Pipe(),
    )

    other_task = asyncio.create_task(fetch_openai_models(object()))
    real_sleep = asyncio.sleep
    await real_sleep(0)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    try:
        result = await get_function_models(object())
    finally:
        event.set()
        await other_task

    assert captured["target_ids"] == []
    assert result == []


@pytest.mark.asyncio
async def test_pipes_waits_for_ollama_two_stage_provider_refresh(monkeypatch):
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 25:
            state.OLLAMA_MODELS = {"ollama-target": {"model": "ollama-target", "name": "Ollama Target"}}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=False, ENABLE_OLLAMA_API=True),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod, "_provider_model_cache_wait_timeout_seconds", lambda: 1.0)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert sleep_calls == 25
    assert captured["target_ids"] == ["ollama-target"]
    assert result == [{"id": "ollama-target", "name": "Compact: Ollama Target"}]


@pytest.mark.asyncio
async def test_pipes_normalizes_ollama_provider_cache_models(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_models"] = kwargs["target_models"]

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={},
            BASE_MODELS=[],
            OPENAI_MODELS={},
            OLLAMA_MODELS={"llama3.2:latest": {"model": "llama3.2:latest", "name": "Llama 3.2", "tags": ["local"]}},
            config=SimpleNamespace(ENABLE_OPENAI_API=False, ENABLE_OLLAMA_API=True),
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)

    result = await mod.Pipe().pipes()

    assert captured["target_models"] == [
        {
            "id": "llama3.2:latest",
            "name": "Llama 3.2",
            "object": "model",
            "created": 0,
            "owned_by": "ollama",
            "ollama": {"model": "llama3.2:latest", "name": "Llama 3.2", "tags": ["local"]},
            "loaded": False,
            "connection_type": "local",
            "tags": ["local"],
        }
    ]
    assert result == [{"id": "llama3.2:latest", "name": "Compact: Llama 3.2"}]


@pytest.mark.asyncio
async def test_pipes_waits_until_core_model_list_timeout_for_sibling_provider_cache(monkeypatch):
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 25:
            state.OPENAI_MODELS = {"slow-target": {"id": "slow-target", "name": "Slow Target"}}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod, "_provider_model_cache_wait_timeout_seconds", lambda: 2.0)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert sleep_calls == 25
    assert captured["target_ids"] == ["slow-target"]
    assert result == [{"id": "slow-target", "name": "Compact: Slow Target"}]


@pytest.mark.asyncio
async def test_pipes_stops_waiting_when_sibling_provider_cache_refresh_completes_empty(monkeypatch):
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        state.OPENAI_MODELS = {}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
    )
    initial_cache = state.OPENAI_MODELS
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert state.OPENAI_MODELS is not initial_cache
    assert sleep_calls == 1
    assert captured["target_ids"] == []
    assert result == []


def test_compaction_summary_embed_html_contains_full_escaped_summary():
    summary = "line 1\n" + ("long summary " * 200) + "<script>alert(1)</script>"

    embed_html = mod.render_compaction_summary_embed_html(summary)

    assert "<details" in embed_html
    assert "Compact summary" in embed_html
    assert "Full summary" not in embed_html
    assert "line 1" in embed_html
    assert "long summary " * 20 in embed_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in embed_html
    assert "<script>alert(1)</script>" not in embed_html


def test_compaction_summary_embed_html_renders_sanitized_markdown_summary():
    summary = (
        "# Heading\n\n"
        "- first item\n"
        "- second item\n\n"
        "**strong**\n\n"
        "`code <value>`\n\n"
        "[safe](https://example.com/path?q=1)\n\n"
        'raw <b onclick="alert(1)">bold</b>\n\n'
        'raw <a href="javascript:alert(2)">link</a>\n\n'
        "<script>alert(1)</script>\n\n"
        "[unsafe](javascript:alert(1))"
    )

    embed_html = mod.render_compaction_summary_embed_html(summary)

    assert "<h1>Heading</h1>" in embed_html
    assert "<li>first item</li>" in embed_html
    assert "<strong>strong</strong>" in embed_html
    assert "<code>code &lt;value&gt;</code>" in embed_html
    assert '<a href="https://example.com/path?q=1" target="_blank" rel="noopener noreferrer">safe</a>' in embed_html
    assert "&lt;b onclick=&quot;alert(1)&quot;&gt;bold&lt;/b&gt;" in embed_html
    assert "&lt;a href=&quot;javascript:alert(2)&quot;&gt;link&lt;/a&gt;" in embed_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in embed_html
    assert 'href="javascript:alert' not in embed_html
    assert "<script>alert(1)</script>" not in embed_html


def test_compaction_summary_embed_html_bounds_expanded_summary_height():
    summary = "\n".join(f"summary line {i}" for i in range(200))

    embed_html = mod.render_compaction_summary_embed_html(summary)

    assert "<details" in embed_html
    assert "Compact summary" in embed_html
    assert "max-height:" in embed_html
    assert "overflow:auto" in embed_html

    # Core's FullHeightIframe only resizes from iframe:height postMessage events.
    assert "type:'iframe:height'" in embed_html

    # If IFRAME_CSP blocks inline script, native details[open] must still reveal the body.
    assert "[open]" in embed_html
    assert ".body-wrap{height:auto;opacity:1}" in embed_html
    assert "data-js" in embed_html

    # Regression guards for the two known bad sizing approaches.
    assert "inner.scrollHeight" not in embed_html
    assert "Math.max(document.documentElement.scrollHeight,document.body.scrollHeight,1)" not in embed_html


def test_extract_compaction_summary_text_decodes_cdata_boundaries():
    summary = "before ]]> after </checkpoint_summary> literal"
    rendered = mod.render_summary_message(summary)

    assert mod.extract_compaction_summary_text_from_messages([rendered]) == summary


def test_extract_compaction_summary_text_ignores_non_generated_tags():
    rendered = mod.render_summary_message("generated summary")
    messages = [
        {
            "role": "system",
            "content": "<checkpoint_summary>system prompt fragment</checkpoint_summary>",
        },
        rendered,
        {
            "role": "user",
            "content": "<checkpoint_summary>ordinary user text</checkpoint_summary>",
        },
    ]

    assert mod.extract_compaction_summary_text_from_messages(messages) == "generated summary"


def test_include_exclude_patterns_match_id_and_name_and_deduplicate_targets():
    valves = mod.Pipe.Valves(
        include_model_patterns="*gpt*,Anthropic*",
        exclude_model_patterns="*mini*",
    )
    models = [
        {"id": "gpt-4.1", "name": "GPT"},
        {"id": "gpt-4.1", "name": "duplicate"},
        {"id": "gpt-4.1-mini", "name": "GPT Mini"},
        {"id": "claude-sonnet", "name": "Anthropic Claude"},
        {"id": "llama", "name": "Local"},
    ]

    targets = mod.filter_target_models(models, valves)

    assert [m["id"] for m in targets] == ["gpt-4.1", "claude-sonnet"]


def test_wrapper_model_records_are_built_for_full_wrapper_id_with_function_owner():
    target = {
        "id": "gpt-4.1",
        "name": "GPT",
        "info": {
            "access_grants": [
                {
                    "id": "grant-1",
                    "principal_type": "group",
                    "principal_id": "team-1",
                    "permission": "read",
                }
            ]
        },
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model=target,
        valves=mod.Pipe.Valves(model_name_prefix="Compact: "),
    )

    assert form["id"] == mod.build_wrapper_model_id("auto_compact", "gpt-4.1")
    assert form["user_id"] == "owner-1"
    assert form["base_model_id"] is None
    assert form["name"] == "Compact: GPT"
    assert form["access_grants"] == [
        {
            "id": "grant-1",
            "principal_type": "group",
            "principal_id": "team-1",
            "permission": "read",
        }
    ]


def test_wrapper_model_form_grants_target_owner_read_when_owner_differs_from_function_owner():
    target = {
        "id": "private-target",
        "name": "Private Target",
        "info": {"user_id": "target-owner", "access_grants": []},
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert {
        "principal_type": "user",
        "principal_id": "target-owner",
        "permission": "read",
    } in form["access_grants"]


def test_wrapper_model_form_preserves_public_read_grant_for_normal_user_visibility():
    target = {
        "id": "public-target",
        "name": "Public Target",
        "info": {
            "access_grants": [
                {
                    "principal_type": "user",
                    "principal_id": "*",
                    "permission": "read",
                }
            ]
        },
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert {
        "principal_type": "user",
        "principal_id": "*",
        "permission": "read",
    } in form["access_grants"]


def test_wrapper_model_form_inherits_target_meta_used_by_core_middleware():
    target = {
        "id": "target-with-meta",
        "name": "Target With Meta",
        "info": {
            "meta": {
                "filterIds": ["model-filter"],
                "actionIds": ["model-action"],
                "knowledge": [{"id": "knowledge-1", "name": "Knowledge"}],
                "capabilities": {"file_context": False, "builtin_tools": False},
                "skillIds": ["skill-1"],
                "toolIds": ["tool-1"],
                "defaultFeatureIds": ["web_search"],
                "terminalId": "terminal-1",
                "tags": [{"name": "target-tag"}],
                "description": "Target description",
            }
        },
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert form["meta"]["filterIds"] == ["model-filter"]
    assert form["meta"]["actionIds"] == ["model-action"]
    assert form["meta"]["knowledge"] == [{"id": "knowledge-1", "name": "Knowledge"}]
    assert form["meta"]["capabilities"] == {"file_context": False, "builtin_tools": False}
    assert form["meta"]["skillIds"] == ["skill-1"]
    assert form["meta"]["toolIds"] == ["tool-1"]
    assert form["meta"]["defaultFeatureIds"] == ["web_search"]
    assert form["meta"]["terminalId"] == "terminal-1"
    assert form["meta"]["tags"] == [{"name": "target-tag"}]
    assert form["meta"]["description"] == "Target description"
    assert form["meta"]["auto_compaction"] == {
        "pipe_function_id": "auto_compact",
        "target_model_id": "target-with-meta",
    }


def test_wrapper_model_form_inherits_top_level_display_metadata_without_wrapper_description():
    target = {
        "id": "provider-target",
        "name": "Provider Target",
        "description": "Provider target description",
        "profile_image_url": "https://example.com/provider.png",
        "capabilities": {"vision": True},
        "tags": ["provider", {"name": "custom"}],
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert form["meta"]["description"] == "Provider target description"
    assert form["meta"]["profile_image_url"] == "https://example.com/provider.png"
    assert form["meta"]["capabilities"] == {"vision": True}
    assert form["meta"]["tags"] == [{"name": "provider"}, {"name": "custom"}]
    assert form["meta"]["auto_compaction"] == {
        "pipe_function_id": "auto_compact",
        "target_model_id": "provider-target",
    }


def test_wrapper_model_form_does_not_invent_description_when_target_has_none():
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model={"id": "plain-target", "name": "Plain Target"},
        valves=mod.Pipe.Valves(),
    )

    assert "description" not in form["meta"]
    assert form["meta"]["auto_compaction"] == {
        "pipe_function_id": "auto_compact",
        "target_model_id": "plain-target",
    }


def test_wrapper_model_form_keeps_only_core_metadata_params_and_forces_streaming():
    target_model_info = SimpleNamespace(
        params=SimpleNamespace(
            model_dump=lambda **kwargs: {
                "function_calling": "native",
                "stream_delta_chunk_size": 4,
                "reasoning_tags": ["think"],
                "stream_response": False,
                "system": "Target system prompt",
                "temperature": 0.2,
            }
        ),
        meta=SimpleNamespace(model_dump=lambda **kwargs: {}),
        access_grants=[],
        user_id="target-owner",
    )

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="target-owner",
        target_model={"id": "target-with-params", "name": "Target With Params"},
        target_model_info=target_model_info,
        valves=mod.Pipe.Valves(),
    )

    assert form["params"] == {
        "function_calling": "native",
        "stream_delta_chunk_size": 4,
        "reasoning_tags": ["think"],
        "stream_response": True,
    }


@pytest.mark.asyncio
async def test_wrapper_sync_uses_function_owner_and_scoped_generated_wrapper_ids(monkeypatch):
    calls = {"insert": [], "update": []}

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            assert function_id == "auto_compact"
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelMeta:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id.startswith("auto_compact.")
            if model_id.endswith(mod.encode_target_model_id("existing-target")):
                return {
                    "id": model_id,
                    "base_model_id": None,
                    "name": "Stale wrapper",
                    "params": {},
                    "meta": {},
                    "access_grants": [],
                    "is_active": True,
                }
            return None

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[
            {"id": "existing-target", "name": "Existing", "info": {"access_grants": []}},
            {"id": "new-target", "name": "New", "info": {"access_grants": []}},
        ],
        valves=mod.Pipe.Valves(),
    )

    assert [call[0] for call in calls["update"]] == [mod.build_wrapper_model_id("auto_compact", "existing-target")]
    assert calls["insert"][0][0] == "function-owner"
    assert calls["insert"][0][1].id == mod.build_wrapper_model_id("auto_compact", "new-target")


@pytest.mark.asyncio
async def test_wrapper_sync_uses_target_model_record_params_for_wrapper_record(monkeypatch):
    calls = {"insert": [], "update": []}
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target-with-record")

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

        def model_dump(self, **kwargs):
            return dict(self.payload)

    class FakeModelMeta:
        def __init__(self, **kwargs):
            self.payload = kwargs

        def model_dump(self, **kwargs):
            return dict(self.payload)

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            if model_id == "target-with-record":
                return SimpleNamespace(
                    params=SimpleNamespace(
                        model_dump=lambda **kwargs: {
                            "function_calling": "native",
                            "temperature": 0.2,
                        }
                    ),
                    meta=SimpleNamespace(model_dump=lambda **kwargs: {"capabilities": {"builtin_tools": False}}),
                    access_grants=[],
                    user_id="function-owner",
                )
            if model_id == wrapper_id:
                return None
            raise AssertionError(f"unexpected model id: {model_id}")

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "target-with-record", "name": "Target With Record", "info": {"meta": {}}}],
        valves=mod.Pipe.Valves(),
    )

    inserted = calls["insert"][0][1]
    assert inserted.id == wrapper_id
    assert inserted.params.payload == {"function_calling": "native", "stream_response": True}
    assert inserted.meta.payload["capabilities"] == {"builtin_tools": False}


@pytest.mark.asyncio
async def test_wrapper_sync_ignores_non_dict_top_level_provider_capabilities(monkeypatch):
    calls = {"insert": [], "update": []}
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "provider-target")

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelMeta(BaseModel):
        description: str | None = None
        profile_image_url: str | None = None
        capabilities: dict | None = None

        model_config = ConfigDict(extra="allow")

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            if model_id in {"provider-target", wrapper_id}:
                return None
            raise AssertionError(f"unexpected model id: {model_id}")

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[
            {
                "id": "provider-target",
                "name": "Provider Target",
                "description": "Provider target description",
                "profile_image_url": "https://example.com/provider.png",
                "capabilities": ["vision"],
            }
        ],
        valves=mod.Pipe.Valves(),
    )

    inserted = calls["insert"][0][1]
    meta = inserted.meta.model_dump(exclude_none=True)
    assert inserted.id == wrapper_id
    assert meta["description"] == "Provider target description"
    assert meta["profile_image_url"] == "https://example.com/provider.png"
    assert "capabilities" not in meta
    assert calls["update"] == []


@pytest.mark.asyncio
async def test_wrapper_sync_does_not_make_base_target_public_without_model_info(monkeypatch):
    calls = {"insert": [], "update": []}
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "base-target")

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelMeta:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == wrapper_id
            return {
                "id": wrapper_id,
                "base_model_id": None,
                "name": "Compact: Base Target",
                "params": {},
                "meta": {
                    "auto_compaction": {
                        "pipe_function_id": "auto_compact",
                        "target_model_id": "base-target",
                    },
                },
                "access_grants": [],
                "is_active": True,
            }

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "base-target", "name": "Base Target", "owned_by": "openai"}],
        valves=mod.Pipe.Valves(),
    )

    assert calls["update"][0][0] == wrapper_id
    assert calls["update"][0][1].access_grants == []
    assert calls["insert"] == []


@pytest.mark.asyncio
async def test_wrapper_sync_skips_update_when_existing_record_matches(monkeypatch):
    calls = {"insert": [], "update": []}
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "same-target")

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

        def model_dump(self, **kwargs):
            return dict(self.payload)

    class FakeModelMeta:
        def __init__(self, **kwargs):
            self.payload = kwargs

        def model_dump(self, **kwargs):
            return dict(self.payload)

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    target = {
        "id": "same-target",
        "name": "Same Target",
        "info": {
            "access_grants": [
                {
                    "principal_type": "user",
                    "principal_id": "*",
                    "permission": "read",
                }
            ]
        },
    }

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == wrapper_id
            return {
                "id": wrapper_id,
                "base_model_id": None,
                "name": "Compact: Same Target",
                "params": {"stream_response": True},
                "meta": {
                    "auto_compaction": {
                        "pipe_function_id": "auto_compact",
                        "target_model_id": "same-target",
                    },
                },
                "access_grants": [
                    {
                        "id": "grant-db-id",
                        "principal_type": "user",
                        "principal_id": "*",
                        "permission": "read",
                    }
                ],
                "is_active": True,
            }

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[target],
        valves=mod.Pipe.Valves(),
    )

    assert calls == {"insert": [], "update": []}


@pytest.mark.asyncio
async def test_target_access_uses_core_chat_model_dict_access_check(monkeypatch, pipe_request, pipe_user):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    target_model = {"id": "target", "name": "Target", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {"target": target_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            raise AssertionError("target access should use the app-state model dict, not the DB Model record")

    async def check_model_access(user, model, db=None):
        captured["user_type"] = type(user)
        captured["user_id"] = user.id
        captured["model"] = model

    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    await mod._validate_target_access(target_model_id="target", request=pipe_request, user=pipe_user)

    assert captured == {"user_type": fake_user_model, "user_id": "user-1", "model": target_model}


def test_iter_cache_models_reads_redisdict_like_values():
    class FakeRedisDict:
        def __init__(self, values):
            self._values = values

        def values(self):
            return list(self._values)

    state = SimpleNamespace(
        MODELS=FakeRedisDict([{"id": "redis-target", "name": "Redis Target"}, "not-a-model"]),
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
    )

    assert [model["id"] for model in mod._iter_cache_models_from_state(state)] == ["redis-target"]


@pytest.mark.asyncio
async def test_target_access_bypasses_core_access_check_for_admin_when_admin_bypass_enabled(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
    pipe_request.app.state.MODELS = {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}
    admin_user = {**pipe_user, "role": "admin"}

    async def check_model_access(user, model, db=None):
        raise AssertionError("admin target validation should match core chat path and skip per-model grants")

    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    config_module = types.ModuleType("open_webui.config")
    config_module.BYPASS_ADMIN_ACCESS_CONTROL = True
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.config", config_module)

    await mod._validate_target_access(target_model_id="target", request=pipe_request, user=admin_user)


@pytest.mark.asyncio
async def test_target_access_checks_admin_when_admin_bypass_disabled(monkeypatch, pipe_request, pipe_user):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    target_model = {"id": "target", "name": "Target", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {"target": target_model}
    admin_user = {**pipe_user, "role": "admin"}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "target"
            return SimpleNamespace(id="target")

    async def check_model_access(user, model, db=None):
        captured["user_type"] = type(user)
        captured["user_role"] = user.role
        captured["model"] = model

    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    config_module = types.ModuleType("open_webui.config")
    config_module.BYPASS_ADMIN_ACCESS_CONTROL = False
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.config", config_module)

    await mod._validate_target_access(target_model_id="target", request=pipe_request, user=admin_user)

    assert captured == {"user_type": fake_user_model, "user_role": "admin", "model": target_model}


@pytest.mark.asyncio
async def test_target_access_allows_admin_raw_provider_target_without_model_record(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
    target_model = {"id": "raw-target", "name": "Raw Target", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {"raw-target": target_model}
    admin_user = {**pipe_user, "role": "admin"}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "raw-target"
            return None

    async def check_model_access(user, model, db=None):
        raise AssertionError("admin raw provider target should not require a DB Model access check")

    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    config_module = types.ModuleType("open_webui.config")
    config_module.BYPASS_ADMIN_ACCESS_CONTROL = False
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.config", config_module)

    await mod._validate_target_access(target_model_id="raw-target", request=pipe_request, user=admin_user)


@pytest.mark.asyncio
async def test_target_access_honors_global_model_access_bypass(monkeypatch, pipe_request, pipe_user):
    install_fake_open_webui_user_model(monkeypatch)
    pipe_request.app.state.MODELS = {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}

    async def check_model_access(user, model, db=None):
        raise AssertionError("global BYPASS_MODEL_ACCESS_CONTROL should skip per-model grants")

    env_module = types.ModuleType("open_webui.env")
    env_module.BYPASS_MODEL_ACCESS_CONTROL = True
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    await mod._validate_target_access(target_model_id="target", request=pipe_request, user=pipe_user)


@pytest.mark.asyncio
async def test_target_access_can_resolve_targets_from_base_model_cache(monkeypatch, pipe_request, pipe_user):
    install_fake_open_webui_user_model(monkeypatch)
    target_model = {"id": "LiteLLM.glm-5.1", "name": "GLM", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {
        mod.build_wrapper_model_id("auto_compact", "LiteLLM.glm-5.1"): {
            "id": mod.build_wrapper_model_id("auto_compact", "LiteLLM.glm-5.1"),
            "name": "Compact: GLM",
            "pipe": {"type": "pipe"},
        }
    }
    pipe_request.app.state.BASE_MODELS = [target_model]
    captured = {}

    async def check_model_access(user, model, db=None):
        captured["model"] = model

    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    await mod._validate_target_access(target_model_id="LiteLLM.glm-5.1", request=pipe_request, user=pipe_user)

    assert captured["model"] == target_model


@pytest.mark.asyncio
async def test_target_access_rejects_generated_wrapper_for_runtime_registered_id(pipe_request, pipe_user):
    wrapper_id = mod.build_wrapper_model_id("compact_alias", "target")
    pipe_request.app.state.MODELS = {
        wrapper_id: {"id": wrapper_id, "name": "Alias Wrapper", "owned_by": "openai"}
    }

    with pytest.raises(HTTPException):
        await mod._validate_target_access(
            target_model_id=wrapper_id,
            request=pipe_request,
            user=pipe_user,
            pipe_function_id="compact_alias",
        )


@pytest.mark.asyncio
async def test_forward_target_injects_resolved_base_model_into_request_models(monkeypatch, pipe_request, pipe_user):
    target_model = {"id": "LiteLLM.glm-5.1", "name": "GLM", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {}
    pipe_request.app.state.BASE_MODELS = [target_model]
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["has_target"] = form_data["model"] in request.app.state.MODELS
        captured["target_model"] = request.app.state.MODELS.get(form_data["model"])
        return StreamingResponse(
            iter([b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n']),
            media_type="text/event-stream",
        )

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    response = await mod._forward_streaming_target(
        request=pipe_request,
        user=pipe_user,
        body={"model": "LiteLLM.glm-5.1", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id=mod.build_wrapper_model_id("auto_compact", "LiteLLM.glm-5.1"),
    )
    async for _ in response.body_iterator:
        pass

    assert captured == {"has_target": True, "target_model": target_model}


@pytest.mark.asyncio
async def test_forward_target_preserves_core_bypass_system_prompt_state(monkeypatch, pipe_request, pipe_user):
    pipe_request.app.state.MODELS = {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}
    pipe_request.state.bypass_system_prompt = True
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        request.state.bypass_filter = bypass_filter
        request.state.bypass_system_prompt = bypass_system_prompt
        captured["bypass_filter"] = bypass_filter
        captured["bypass_system_prompt"] = bypass_system_prompt
        captured["state_bypass_system_prompt"] = getattr(request.state, "bypass_system_prompt", None)
        return StreamingResponse(
            iter([b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n']),
            media_type="text/event-stream",
        )

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    response = await mod._forward_streaming_target(
        request=pipe_request,
        user=pipe_user,
        body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id=mod.build_wrapper_model_id("auto_compact", "target"),
    )
    async for _ in response.body_iterator:
        pass

    assert captured == {
        "bypass_filter": True,
        "bypass_system_prompt": True,
        "state_bypass_system_prompt": True,
    }
    assert pipe_request.state.bypass_system_prompt is True
    assert not hasattr(pipe_request.state, "bypass_filter")


def test_summary_model_validation_allows_pipe_backed_and_rejects_arena():
    models = {
        "target": {"id": "target", "name": "Target"},
        "pipe.summary": {"id": "pipe.summary", "pipe": {"type": "pipe"}, "name": "Summary Pipe"},
        "arena": {"id": "arena", "owned_by": "arena", "arena": True},
    }

    assert mod.validate_summary_model_id("", "target", models) == "target"
    assert mod.validate_summary_model_id("pipe.summary", "target", models) == "pipe.summary"

    with pytest.raises(ValueError, match="arena"):
        mod.validate_summary_model_id("arena", "target", models)


def test_summary_model_validation_rejects_missing_configured_model():
    models = {"target": {"id": "target", "name": "Target"}}

    with pytest.raises(ValueError, match="not found"):
        mod.validate_summary_model_id("missing.summary", "target", models)


def test_valve_defaults_are_conservative_for_v1_continuation():
    valves = mod.Pipe.Valves()

    assert valves.trigger_total_tokens == 100000
    assert valves.force_include_usage is True
    assert valves.summary_tool_policy == "fallback_on_tool_call"
    assert valves.historical_message_excerpt_bytes == mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES
    assert valves.historical_message_excerpt_count == mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT
    assert not hasattr(valves, "keep_tail_messages")
    assert not hasattr(valves, "preserve_latest_tool_rounds")


def test_usage_extraction_handles_chat_completions_and_responses_api_shapes():
    chat = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    response = {
        "type": "response.completed",
        "response": {"usage": {"input_tokens": 7, "output_tokens": 3}},
    }

    assert mod.extract_usage_from_stream_payload(chat) == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    assert mod.extract_usage_from_stream_payload(response)["total_tokens"] == 10


def test_request_scoped_usage_precedes_persisted_and_message_usage():
    request = SimpleNamespace(state=SimpleNamespace())
    key = mod.usage_state_key("chat-1", "message-1", "auto_compact.target")
    setattr(request.state, key, {"total_tokens": 999, "input_tokens": 900, "output_tokens": 99})

    usage = mod.choose_usage_signal(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
        persisted_usage={"total_tokens": 100},
        messages=[{"role": "assistant", "usage": {"total_tokens": 200}}],
    )

    assert usage["total_tokens"] == 999


def test_usage_option_injection_preserves_existing_stream_options():
    body = {
        "stream": True,
        "stream_options": {"other": "value"},
    }

    out = mod.inject_stream_usage_options(body)

    assert out is not body
    assert out["stream_options"] == {"other": "value", "include_usage": True}


def test_context_window_classifier_handles_structured_responses_and_http_exception():
    assert mod.is_retryable_context_error(
        JSONResponse(
            {"error": {"code": "context_length_exceeded", "message": "too long"}},
            status_code=400,
        )
    )
    assert mod.is_retryable_context_error(
        PlainTextResponse("maximum context length exceeded", status_code=413)
    )
    assert mod.is_retryable_context_error(
        HTTPException(status_code=400, detail={"error": {"type": "context_window_exceeded"}})
    )
    assert mod.is_retryable_context_error(Exception("maximum context length exceeded"))


def test_context_window_classifier_avoids_rate_limit_quota_and_tpm_false_positives():
    assert not mod.is_retryable_context_error(
        JSONResponse(
            {"error": {"message": "too many tokens per minute for this organization"}},
            status_code=429,
        )
    )
    assert not mod.is_retryable_context_error(
        {"error": {"code": "insufficient_quota", "message": "quota exceeded"}},
    )
    assert not mod.is_retryable_context_error(
        {"error": {"message": "token-per-minute limit exceeded"}},
    )


def test_context_window_message_fallback_uses_only_error_message_fields():
    body_with_unrelated_serialized_text = {
        "metadata": {"raw": "maximum context length exceeded"},
        "error": {"message": "ordinary bad request"},
    }
    body_with_error_message = {"error": {"message": "input is too long for the context window"}}

    assert not mod.is_retryable_context_error(body_with_unrelated_serialized_text, status_code=400)
    assert mod.is_retryable_context_error(body_with_error_message, status_code=400)


def test_sse_error_payload_is_classified_before_any_content():
    chunk = b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'

    assert mod.extract_sse_json_events(chunk) == [
        {"error": {"code": "context_length_exceeded", "message": "too long"}}
    ]
    assert mod.first_chunk_is_retryable_context_error(chunk)


def test_responses_failed_payload_is_classified_before_any_content():
    payload = {
        "type": "response.failed",
        "response": {
            "error": {
                "code": "context_length_exceeded",
                "message": "too long",
            }
        },
    }
    chunk = f"data: {json.dumps(payload)}\n\n".encode()

    assert mod.extract_sse_json_events(chunk) == [payload]
    assert mod.first_chunk_is_retryable_context_error(chunk)


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_split_initial_sse_context_error():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b'data: {"error": {"code": "context_length_'
            yield b'exceeded", "message": "too long"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_initial_responses_failed_context_error():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            payload = {
                "type": "response.failed",
                "response": {
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "too long",
                    }
                },
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_responses_failed_after_pre_output_events():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b'data: {"type": "response.created", "response": {"id": "resp-1"}}\n\n'
            yield b'data: {"type": "response.in_progress", "response": {"id": "resp-1"}}\n\n'
            payload = {
                "type": "response.failed",
                "response": {
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "too long",
                    }
                },
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type",
    [
        "response.refusal.delta",
        "response.reasoning_summary_text.delta",
        "response.mcp_call_arguments.delta",
        "response.custom_tool_call_input.delta",
    ],
)
async def test_streaming_forwarder_treats_responses_output_deltas_as_visible_output(event_type):
    requested_second_chunk = False
    closed = False
    background_count = 0
    first_chunk = f'data: {{"type": "{event_type}", "delta": "visible"}}\n\n'.encode()

    async def chunks():
        nonlocal requested_second_chunk, closed
        try:
            yield first_chunk
            requested_second_chunk = True
            yield b'data: {"type": "response.completed", "response": {"id": "resp-1"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    assert requested_second_chunk is False
    iterator = prepared.body_iterator.__aiter__()
    assert await iterator.__anext__() == first_chunk
    await iterator.aclose()
    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_surfaces_context_error_after_same_chunk_non_error_event():
    chunk = (
        b'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 0}}\n\n'
        b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
    )

    async def chunks():
        yield chunk

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for emitted_chunk in prepared.body_iterator:
        emitted.append(emitted_chunk)

    assert emitted == [chunk]


@pytest.mark.asyncio
async def test_streaming_forwarder_surfaces_context_error_after_first_non_error_event():
    async def chunks():
        yield b'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 0}}\n\n'
        yield b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk)

    assert emitted == [
        b'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 0}}\n\n',
        b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n',
    ]


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_iterator_context_exception_before_visible_output():
    closed = False
    background_ran = False
    restored = False

    async def chunks():
        nonlocal closed
        try:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "context_length_exceeded", "message": "too long"}},
            )
            yield b""
        finally:
            closed = True

    async def background():
        nonlocal background_ran
        background_ran = True

    def restore():
        nonlocal restored
        restored = True

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(
        chunks(),
        media_type="text/event-stream",
        background=BackgroundTask(background),
    )

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
            restore=restore,
        )

    assert restored is True
    assert closed is True
    assert background_ran is True


@pytest.mark.asyncio
async def test_non_context_json_response_is_returned_as_sse_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = JSONResponse({"error": {"code": "provider_error", "message": "unavailable"}}, status_code=500)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error":{"code":"provider_error","message":"unavailable"}}\n\n']


@pytest.mark.asyncio
async def test_non_context_json_detail_response_is_returned_as_sse_error_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = JSONResponse({"detail": "provider unavailable"}, status_code=500)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error": {"code": "provider_error", "message": "provider unavailable"}}\n\n']


@pytest.mark.asyncio
async def test_non_context_json_message_response_is_returned_as_sse_error_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = JSONResponse({"message": "rate limited"}, status_code=429)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error": {"code": "provider_error", "message": "rate limited"}}\n\n']


@pytest.mark.asyncio
async def test_non_context_json_string_error_response_is_returned_as_sse_error_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = JSONResponse({"error": "rate limited"}, status_code=429)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error": {"code": "provider_error", "message": "rate limited"}}\n\n']


@pytest.mark.asyncio
async def test_non_context_plain_text_response_is_returned_as_sse_error_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = PlainTextResponse("provider unavailable", status_code=500)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error": {"code": "provider_error", "message": "provider unavailable"}}\n\n']


@pytest.mark.asyncio
async def test_streaming_forwarder_normalizes_dict_chunks_to_sse_events():
    async def chunks():
        yield {"usage": {"prompt_tokens": 2, "completion_tokens": 0}}
        yield {"choices": [{"delta": {"content": "hello"}}]}

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == [
        'data: {"usage": {"prompt_tokens": 2, "completion_tokens": 0}}\n\n',
        'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n',
    ]


@pytest.mark.asyncio
async def test_streaming_forwarder_surfaces_context_error_after_visible_output():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
        yield b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk)

    assert emitted == [
        b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n',
        b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n',
    ]


@pytest.mark.asyncio
async def test_streaming_forwarder_closes_inner_stream_after_visible_output_early_stop():
    closed = False
    background_ran = False

    async def chunks():
        nonlocal closed
        try:
            yield b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
            await asyncio.sleep(60)
            yield b'data: {"choices": [{"delta": {"content": "late"}}]}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_ran
        background_ran = True

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(
        chunks(),
        media_type="text/event-stream",
        background=BackgroundTask(background),
    )

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    iterator = prepared.body_iterator.__aiter__()
    first = await iterator.__anext__()
    assert first == b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'

    await iterator.aclose()

    assert closed is True
    assert background_ran is True


def test_request_state_proxy_isolates_bypass_and_metadata():
    request = SimpleNamespace(
        state=SimpleNamespace(
            bypass_filter=False,
            bypass_system_prompt=True,
            metadata={"selected_model_id": "arena-a", "chat_id": "chat-1"},
        )
    )

    proxy = mod.RequestStateProxy(
        request,
        bypass_filter=True,
        bypass_system_prompt=False,
        metadata={"task": mod.INTERNAL_SUMMARY_TASK},
    )
    proxy.state.metadata["new"] = "value"
    proxy.state.bypass_system_prompt = False

    assert request.state.bypass_filter is False
    assert request.state.bypass_system_prompt is True
    assert request.state.metadata == {"selected_model_id": "arena-a", "chat_id": "chat-1"}
    assert proxy.state.bypass_filter is True
    assert proxy.state.bypass_system_prompt is False
    assert proxy.state.metadata == {"task": mod.INTERNAL_SUMMARY_TASK, "new": "value"}


def test_request_state_proxy_does_not_add_flags_to_original_request():
    request = SimpleNamespace(state=SimpleNamespace())

    proxy = mod.RequestStateProxy(request, bypass_filter=True, bypass_system_prompt=True)

    assert not hasattr(request.state, "bypass_filter")
    assert not hasattr(request.state, "bypass_system_prompt")
    assert proxy.state.bypass_filter is True
    assert proxy.state.bypass_system_prompt is True


@pytest.mark.asyncio
async def test_request_state_proxy_handles_unpickleable_metadata_without_mutating_original():
    future = asyncio.get_running_loop().create_future()
    request = SimpleNamespace(
        state=SimpleNamespace(
            metadata={
                "chat_id": "chat-1",
                "tools": {"tool": {"future": future}},
            }
        )
    )

    proxy = mod.RequestStateProxy(request, bypass_filter=True)

    proxy.state.metadata["chat_id"] = "summary-chat"
    proxy.state.metadata["tools"]["tool"]["name"] = "Tool"

    assert request.state.metadata["chat_id"] == "chat-1"
    assert "name" not in request.state.metadata["tools"]["tool"]
    assert proxy.state.metadata["tools"]["tool"]["future"] is future


def test_summary_task_metadata_removes_tool_and_arena_state():
    metadata = {
        "chat_id": "chat-1",
        "selected_model_id": "arena-a",
        "tools": {"tool": {}},
        "tool_ids": ["tool"],
    }

    summary_metadata = mod.build_summary_task_metadata(metadata)

    assert summary_metadata["chat_id"] == "chat-1"
    assert summary_metadata["task"] == mod.INTERNAL_SUMMARY_TASK
    assert summary_metadata["tools"] == {}
    assert summary_metadata["tool_ids"] == ["tool"]
    assert "selected_model_id" not in summary_metadata


@pytest.mark.asyncio
async def test_summary_task_metadata_drops_unpickleable_core_values():
    future = asyncio.get_running_loop().create_future()
    metadata = {
        "chat_id": "chat-1",
        "params": {"function_calling": "native"},
        "mcp_clients": {"server": future},
    }

    summary_metadata = mod.build_summary_task_metadata(metadata)

    assert summary_metadata["chat_id"] == "chat-1"
    assert summary_metadata["params"] == {"function_calling": "native"}
    assert "mcp_clients" not in summary_metadata
    assert summary_metadata["task"] == mod.INTERNAL_SUMMARY_TASK


@pytest.mark.asyncio
async def test_summary_generation_overrides_request_state_metadata(monkeypatch, pipe_request, pipe_user):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        request.state.bypass_filter = bypass_filter
        request.state.bypass_system_prompt = bypass_system_prompt
        captured["state_metadata"] = dict(request.state.metadata)
        captured["form_metadata"] = dict(form_data["metadata"])
        captured["bypass_system_prompt"] = bypass_system_prompt
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    pipe_request.state.metadata = {"selected_model_id": "arena-a", "tools": {"bad": {}}, "tool_ids": ["bad"]}
    pipe_request.state.bypass_system_prompt = True

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1", "selected_model_id": "arena-a", "tools": {"bad": {}}, "tool_ids": ["bad"]},
        summary_model_id="target",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
    )

    assert result == "summary"
    assert captured["state_metadata"]["task"] == mod.INTERNAL_SUMMARY_TASK
    assert captured["state_metadata"]["tools"] == {}
    assert captured["state_metadata"]["tool_ids"] == ["bad"]
    assert "selected_model_id" not in captured["state_metadata"]
    assert captured["form_metadata"] == captured["state_metadata"]
    assert captured["bypass_system_prompt"] is False
    assert pipe_request.state.metadata == {
        "selected_model_id": "arena-a",
        "tools": {"bad": {}},
        "tool_ids": ["bad"],
    }
    assert pipe_request.state.bypass_system_prompt is True
    assert not hasattr(pipe_request.state, "bypass_filter")


@pytest.mark.asyncio
async def test_summary_generation_strips_request_response_format_without_overriding_model_params(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}
    model_response_format = {"type": "json_schema", "json_schema": {"name": "Configured", "schema": {"type": "object"}}}
    pipe_request.app.state.MODELS = {
        "summary": {
            "id": "summary",
            "name": "Summary",
            "params": {"response_format": model_response_format},
        }
    }

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["form_body"] = dict(form_data)
        captured["model_params"] = request.app.state.MODELS["summary"]["params"]
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="summary",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={
            "model": "target",
            "stream": True,
            "messages": [{"role": "user", "content": "old"}],
            "response_format": {"type": "json_object"},
        },
    )

    assert result == "summary"
    assert "response_format" not in captured["form_body"]
    assert captured["model_params"]["response_format"] is model_response_format


@pytest.mark.asyncio
async def test_summary_generation_decodes_own_wrapper_summary_model(monkeypatch, pipe_request, pipe_user):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["model"] = form_data["model"]
        captured["stream"] = form_data["stream"]
        captured["metadata"] = dict(form_data["metadata"])
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    wrapper_summary_model = mod.build_wrapper_model_id("auto_compact", "target.summary")

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id=wrapper_summary_model,
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
    )

    assert result == "summary"
    assert captured["model"] == "target.summary"
    assert captured["stream"] is True
    assert captured["metadata"]["task"] == mod.INTERNAL_SUMMARY_TASK


@pytest.mark.asyncio
async def test_summary_generation_decodes_runtime_registered_wrapper_summary_model(monkeypatch, pipe_request, pipe_user):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["model"] = form_data["model"]
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id=mod.build_wrapper_model_id("compact_alias", "target.summary"),
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
        pipe_function_id="compact_alias",
    )

    assert result == "summary"
    assert captured["model"] == "target.summary"


@pytest.mark.asyncio
async def test_summary_generation_preserves_tools_but_disables_forced_tool_choice(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = []

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured.append(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    source_messages = [
        {"role": "system", "content": "target system"},
        {"role": "user", "content": "old"},
    ]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    metadata = {
        "chat_id": "chat-1",
        "selected_model_id": "arena-a",
        "tools": {"lookup": {"spec": tools[0]["function"]}},
        "tool_ids": ["lookup"],
    }
    base_body = {
        "model": "target",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [*source_messages, {"role": "user", "content": "active"}],
        "tools": tools,
        "metadata": {"chat_id": "chat-1"},
        "previous_response_id": "resp_should_not_survive_compaction",
    }

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata=metadata,
        summary_model_id="target.summary",
        source_messages=source_messages,
        base_body={**base_body, "tool_choice": "required"},
    )

    assert result == "summary"
    form_data = captured[-1]
    assert form_data["model"] == "target.summary"
    assert form_data["stream"] is True
    assert form_data["stream_options"] == {"include_usage": True}
    assert form_data["tools"] == tools
    assert form_data["tool_choice"] == "none"
    assert "previous_response_id" not in form_data
    assert form_data["messages"][:-1] == source_messages
    assert form_data["messages"][-1]["role"] == "user"
    assert "AUTO-COMPACTION CHECKPOINT SUMMARY" in form_data["messages"][-1]["content"]
    assert form_data["metadata"]["task"] == mod.INTERNAL_SUMMARY_TASK
    assert form_data["metadata"]["tools"] == {}
    assert form_data["metadata"]["tool_ids"] == ["lookup"]
    assert "selected_model_id" not in form_data["metadata"]

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata=metadata,
        summary_model_id="target.summary",
        source_messages=source_messages,
        base_body={**base_body, "tool_choice": "auto"},
    )

    assert result == "summary"
    assert captured[-1]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_summary_generation_retries_without_tools_after_tool_call_response(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = []

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured.append(copy.deepcopy(form_data))
        if len(captured) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "summary after retry"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    source_messages = [
        {"role": "system", "content": "target system"},
        {"role": "user", "content": "old"},
    ]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    base_body = {
        "model": "target",
        "stream": True,
        "messages": [*source_messages, {"role": "user", "content": "active"}],
        "tools": tools,
        "tool_choice": "auto",
        "functions": [{"name": "legacy_lookup", "parameters": {"type": "object"}}],
        "function_call": "auto",
        "parallel_tool_calls": True,
    }

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1", "tool_ids": ["lookup"]},
        summary_model_id="target.summary",
        source_messages=source_messages,
        base_body=base_body,
    )

    assert result == "summary after retry"
    assert len(captured) == 2
    assert captured[0]["tools"] == tools
    assert captured[0]["tool_choice"] == "auto"
    assert "tools" not in captured[1]
    assert "tool_choice" not in captured[1]
    assert "functions" not in captured[1]
    assert "function_call" not in captured[1]
    assert "parallel_tool_calls" not in captured[1]
    assert captured[1]["messages"] == captured[0]["messages"]
    assert captured[1]["metadata"] == captured[0]["metadata"]


@pytest.mark.asyncio
async def test_summary_generation_strips_tools_before_first_request_when_configured(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = []

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured.append(copy.deepcopy(form_data))
        return {"choices": [{"message": {"content": "summary without tools"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    source_messages = [{"role": "user", "content": "old"}]
    base_body = {
        "model": "target",
        "messages": [*source_messages, {"role": "user", "content": "active"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "functions": [{"name": "legacy_lookup", "parameters": {"type": "object"}}],
        "function_call": "auto",
        "parallel_tool_calls": True,
    }

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1", "tool_ids": ["lookup"]},
        summary_model_id="target.summary",
        source_messages=source_messages,
        base_body=base_body,
        summary_tool_policy="always_strip",
    )

    assert result == "summary without tools"
    assert len(captured) == 1
    assert "tools" not in captured[0]
    assert "tool_choice" not in captured[0]
    assert "functions" not in captured[0]
    assert "function_call" not in captured[0]
    assert "parallel_tool_calls" not in captured[0]


@pytest.mark.asyncio
async def test_summary_generation_errors_on_tool_call_when_fallback_disabled(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = []

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured.append(copy.deepcopy(form_data))
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    source_messages = [{"role": "user", "content": "old"}]
    base_body = {
        "model": "target",
        "messages": [*source_messages, {"role": "user", "content": "active"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
    }

    with pytest.raises(mod.SummaryToolCallError):
        await mod._generate_summary_text(
            request=pipe_request,
            user=pipe_user,
            metadata={"chat_id": "chat-1", "tool_ids": ["lookup"]},
            summary_model_id="target.summary",
            source_messages=source_messages,
            base_body=base_body,
            summary_tool_policy="error_on_tool_call",
        )

    assert len(captured) == 1
    assert captured[0]["tools"] == base_body["tools"]
    assert captured[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_summary_generation_uses_handoff_checkpoint_prompt(monkeypatch, pipe_request, pipe_user):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["messages"] = form_data["messages"]
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="target",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
    )

    assert result == "summary"
    assert captured["messages"][0] == {"role": "user", "content": "old"}
    assert captured["messages"][-1]["role"] == "user"
    prompt = captured["messages"][-1]["content"]
    assert "AUTO-COMPACTION CHECKPOINT SUMMARY" in prompt
    assert "Open WebUI chat" in prompt
    assert "future model call" in prompt
    assert "existing <auto_compaction_context>" in prompt
    assert "Do not invent facts" in prompt
    assert "Do not call tools" in prompt


@pytest.mark.asyncio
async def test_summary_generation_passes_open_webui_user_model_to_inner_completion(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["user_id"] = user.id
        captured["user_role"] = user.role
        captured["user_type"] = type(user)
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="target",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
    )

    assert result == "summary"
    assert captured == {"user_id": "user-1", "user_role": "user", "user_type": fake_user_model}


@pytest.mark.asyncio
async def test_tool_history_compaction_summarizes_before_latest_tool_round_and_preserves_tool_messages(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}

    async def get_or_create_compaction_summary(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "combined summary"

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    original_tool = {"role": "tool", "tool_call_id": "call-1", "content": {"value": 42}}
    compacted, did_compact = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={
            "model": "target",
            "stream": True,
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "old request"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active request"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q":"alpha"}'},
                        }
                    ],
                },
                original_tool,
            ],
        },
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q":"alpha"}'},
                    }
                ],
            },
            original_tool,
        ],
    )

    assert did_compact is True
    assert original_tool == {"role": "tool", "tool_call_id": "call-1", "content": {"value": 42}}
    assert [message["role"] for message in captured["source_messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert compacted[0] == {"role": "system", "content": "system prompt"}
    assert compacted[1]["role"] == "user"
    assert "<auto_compaction_context>" in compacted[1]["content"]
    assert "combined summary" in compacted[1]["content"]
    assert compacted[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"alpha"}'},
            }
        ],
    }
    assert compacted[3] == original_tool


@pytest.mark.asyncio
async def test_tool_history_compaction_always_extends_available_parent_checkpoint(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active request"}
    parent_checkpoint = {
        "id": "history-checkpoint-1",
        "state": "ready",
        "source_message_count": len(history),
        "source_hash": mod.compute_source_hash(history),
        "summary_text": "existing history summary",
        "summary_meta": {},
    }
    rows.append(parent_checkpoint)

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            for row in rows:
                if row["source_hash"] == kwargs["source_hash"]:
                    return row
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent(rows, kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            rows.append(row)
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "tool summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]
    messages = [*history, active, *latest_round]

    compacted, did_compact = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={"model": "target", "stream": True, "messages": messages},
        messages=messages,
    )

    assert did_compact is True
    assert len(summary_inputs) == 1
    assert "existing history summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [active]
    assert len(rows) == 2
    assert rows[1]["parent_checkpoint_id"] == parent_checkpoint["id"]
    assert "tool summary" in compacted[0]["content"]
    assert compacted[1:] == latest_round


@pytest.mark.asyncio
async def test_tool_history_compaction_summarizes_all_but_latest_sequential_tool_round(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}

    async def get_or_create_compaction_summary(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "combined summary"

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    messages = [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "search", "arguments": '{"q":"alpha"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": {"value": 42}},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-2", "type": "function", "function": {"name": "fetch", "arguments": '{"url":"https://example.test"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "fetched body"},
    ]

    compacted, did_compact = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={"model": "target", "stream": True, "messages": messages},
        messages=messages,
    )

    assert did_compact is True
    assert captured["source_messages"] == messages[:3]
    assert "combined summary" in compacted[0]["content"]
    assert compacted[1:] == messages[3:]


@pytest.mark.asyncio
async def test_extract_summary_text_from_streaming_response():
    background_ran = False

    async def chunks():
        yield b'data: {"choices": [{"delta": {"content": "hello "}}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": "world"}}]}\n\n'

    async def background():
        nonlocal background_ran
        background_ran = True

    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    assert await mod.extract_text_from_completion_response(response) == "hello world"
    assert background_ran is True


@pytest.mark.asyncio
async def test_extract_summary_text_from_streamed_chat_completion_message():
    async def chunks():
        payload = {"choices": [{"message": {"role": "assistant", "content": "pipe summary"}}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "pipe summary"


@pytest.mark.asyncio
async def test_extract_summary_text_from_split_sse_event_chunks():
    async def chunks():
        yield b'data: {"choices": [{"delta": '
        yield b'{"content": "hello"}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "hello"


@pytest.mark.asyncio
async def test_extract_summary_text_preserves_split_utf8_sse_chunks():
    payload = {"choices": [{"delta": {"content": "こんにちは"}}]}
    event = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
    split_at = event.index("こ".encode()) + 1

    async def chunks():
        yield event[:split_at]
        yield event[split_at:]

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "こんにちは"


@pytest.mark.asyncio
async def test_extract_summary_text_preserves_streamed_delta_newlines():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"content": "line 1\\n"}}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": "line 2"}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "line 1\nline 2"


@pytest.mark.asyncio
async def test_extract_summary_text_ignores_non_structured_sse_data():
    background_ran = False

    async def chunks():
        yield b"data: plain text\n\n"
        yield b'data: "json string"\n\n'
        yield b"data: 2026\n\n"
        yield b'data: {"q":"alpha"}\n\n'
        yield b"data: [DONE]\n\n"

    async def background():
        nonlocal background_ran
        background_ran = True

    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(RuntimeError, match="did not return text content"):
        await mod.extract_text_from_completion_response(response)
    assert background_ran is True


@pytest.mark.asyncio
async def test_extract_summary_text_propagates_streamed_context_error():
    async def chunks():
        payload = {"error": {"code": "context_length_exceeded", "message": "maximum context length exceeded"}}
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_propagates_responses_api_error_event():
    async def chunks():
        payload = {"type": "error", "code": "context_length_exceeded", "message": "maximum context length exceeded"}
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_propagates_responses_api_failed_event():
    async def chunks():
        payload = {
            "type": "response.failed",
            "response": {"error": {"code": "server_error", "message": "The model failed to generate a response."}},
        }
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(RuntimeError, match="The model failed to generate a response"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_non_sse_streaming_response():
    async def chunks():
        yield b"plain text"

    response = StreamingResponse(chunks(), media_type="text/plain")

    with pytest.raises(RuntimeError, match="structured OpenAI-compatible response"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_from_responses_api_output():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "hello "},
                    {"type": "output_text", "text": "world"},
                ],
            }
        ]
    }

    assert await mod.extract_text_from_completion_response(response) == "hello world"


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_unstructured_non_stream_response():
    with pytest.raises(RuntimeError, match="did not return text content"):
        await mod.extract_text_from_completion_response("plain text")

    with pytest.raises(RuntimeError, match="did not return text content"):
        await mod.extract_text_from_completion_response({"content": "plain text"})


@pytest.mark.asyncio
async def test_extract_summary_text_preserves_regular_short_line_summary():
    summary = "\n".join(f"- id-{i}" for i in range(100))
    response = {"choices": [{"message": {"content": summary}}]}

    assert await mod.extract_text_from_completion_response(response) == summary


@pytest.mark.asyncio
async def test_extract_summary_text_ignores_streamed_reasoning_content():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"reasoning_content": "Let me summarize.\\n"}}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": "Final summary"}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "Final summary"


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_tool_call_response():
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    with pytest.raises(RuntimeError, match="tool call"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_raw_responses_tool_call_output():
    response = {
        "id": "resp-1",
        "output": [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{}",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="tool call"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_streamed_tool_call_response():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"tool_calls": [{"id": "call-1"}]}}]}\n\n'
        yield b'data: {"choices": [{"finish_reason": "tool_calls", "delta": {}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(RuntimeError, match="tool call"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_streamed_message_content_with_tool_call():
    async def chunks():
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "not a final summary",
                        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup"}}],
                    }
                }
            ]
        }
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(RuntimeError, match="tool call"):
        await mod.extract_text_from_completion_response(response)


def test_build_summary_completion_body_strips_response_format():
    source_messages = [{"role": "user", "content": "hello"}]
    base_body = {
        "model": "target",
        "messages": source_messages,
        "stream": True,
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Answer", "schema": {"type": "object"}},
        },
    }

    body = mod.build_summary_completion_body(
        base_body,
        summary_model_id="summary",
        source_messages=source_messages,
        metadata={"chat_id": "chat-1"},
    )

    assert "response_format" not in body
    assert body["temperature"] == 0.2


@pytest.mark.asyncio
async def test_lookup_persisted_usage_reads_current_branch(monkeypatch):
    class FakeChats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id):
            assert chat_id == "chat-1"
            return {
                "root": {"id": "root", "role": "user", "content": "root"},
                "assistant-old": {
                    "id": "assistant-old",
                    "parentId": "root",
                    "role": "assistant",
                    "content": "old",
                    "usage": {"total_tokens": 10},
                },
                "assistant-current": {
                    "id": "assistant-current",
                    "parentId": "assistant-old",
                    "role": "assistant",
                    "content": "current",
                    "info": {"usage": {"prompt_tokens": 100, "completion_tokens": 23}},
                },
                "sibling": {
                    "id": "sibling",
                    "parentId": "root",
                    "role": "assistant",
                    "content": "other branch",
                    "usage": {"total_tokens": 999},
                },
            }

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = FakeChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    usage = await mod.lookup_persisted_usage("chat-1", "assistant-current")

    assert usage["total_tokens"] == 123


@pytest.mark.asyncio
async def test_lookup_persisted_usage_uses_legacy_map_when_message_id_missing(monkeypatch):
    class FakeChats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id):
            return {
                "root": {"id": "root", "role": "user", "content": "root"},
                "assistant-1": {
                    "id": "assistant-1",
                    "parentId": "root",
                    "role": "assistant",
                    "content": "answer",
                    "usage": {"input_tokens": 9, "output_tokens": 4},
                },
            }

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = FakeChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    usage = await mod.lookup_persisted_usage("chat-1", None)

    assert usage["total_tokens"] == 13


@pytest.fixture
def pipe_request():
    return SimpleNamespace(state=SimpleNamespace(), app=SimpleNamespace(state=SimpleNamespace(MODELS={})))


@pytest.fixture
def pipe_user():
    return {
        "id": "user-1",
        "email": "user@example.com",
        "name": "User",
        "role": "user",
        "last_active_at": 0,
        "updated_at": 0,
        "created_at": 0,
    }


@pytest.fixture
def pipe_metadata():
    return {"chat_id": "chat-1", "message_id": "message-1", "session_id": "session-1"}


@pytest.mark.asyncio
async def test_pipe_forwards_below_threshold_to_decoded_target_with_metadata(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}

    async def validate_target_access(**kwargs):
        captured["validated_target"] = kwargs["target_model_id"]

    async def model_dict_from_request(request):
        return {"target.model": {"id": "target.model", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 10, "output_tokens": 0}

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target.model")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    }

    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert captured["validated_target"] == "target.model"
    assert captured["forward_body"]["model"] == "target.model"
    assert captured["forward_body"]["metadata"] == pipe_metadata
    assert captured["forward_body"]["stream_options"]["include_usage"] is True
    assert captured["forward_body"]["messages"] == body["messages"]
    assert events == []


@pytest.mark.asyncio
async def test_pipe_uses_runtime_registered_id_for_decode_and_checkpoint_scope(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}

    async def validate_target_access(**kwargs):
        captured["access_pipe_function_id"] = kwargs["pipe_function_id"]
        captured["validated_target"] = kwargs["target_model_id"]

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 10, "output_tokens": 0}

    async def body_reusable_checkpoint_match(**kwargs):
        captured["checkpoint_pipe_function_id"] = kwargs["pipe_function_id"]
        return None

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod.Pipe, "__module__", "function_compact_alias")

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100

    result = await pipe.pipe(
        {
            "model": mod.build_wrapper_model_id("compact_alias", "target"),
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert captured["access_pipe_function_id"] == "compact_alias"
    assert captured["validated_target"] == "target"
    assert captured["checkpoint_pipe_function_id"] == "compact_alias"
    assert captured["forward_body"]["model"] == "target"


@pytest.mark.asyncio
async def test_pipe_rejects_wrapper_prefix_that_does_not_match_runtime_registered_id(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    async def forward_target(**kwargs):
        raise AssertionError("mismatched wrapper prefix must not be forwarded")

    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod.Pipe, "__module__", "function_compact_alias")

    result = await mod.Pipe().pipe(
        {
            "model": mod.build_wrapper_model_id("auto_compact", "target"),
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "invalid_wrapper_id"
    assert "compact_alias" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_forwards_metadata_with_unpickleable_core_values(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    future = asyncio.get_running_loop().create_future()
    metadata = {
        **pipe_metadata,
        "params": {"function_calling": "native"},
        "tools": {"tool": {"future": future}},
    }
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10}

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=metadata,
    )

    assert result == {"ok": True}
    assert captured["forward_body"]["metadata"]["tools"]["tool"]["future"] is future
    assert metadata["chat_id"] == "chat-1"


@pytest.mark.asyncio
async def test_pipe_forwarding_passes_open_webui_user_model_to_inner_completion(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["model"] = form_data["model"]
        captured["user_id"] = user.id
        captured["user_role"] = user.role
        captured["user_type"] = type(user)
        return StreamingResponse(
            iter([b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n']),
            media_type="text/event-stream",
        )

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    chunks = []
    async for chunk in result.body_iterator:
        chunks.append(chunk)

    assert chunks == [b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n']
    assert captured == {
        "model": "target",
        "user_id": "user-1",
        "user_role": "user",
        "user_type": fake_user_model,
    }


@pytest.mark.asyncio
async def test_pipe_non_streaming_forwards_to_decoded_target_completion(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    captured = {}

    async def validate_target_access(**kwargs):
        captured["validated_target"] = kwargs["target_model_id"]

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["request_state"] = request.state
        captured["form_data"] = form_data
        captured["user"] = user
        captured["bypass_filter"] = bypass_filter
        captured["bypass_system_prompt"] = bypass_system_prompt
        return {
            "id": "chatcmpl-target",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["validated_target"] == "target"
    assert captured["form_data"]["model"] == "target"
    assert captured["form_data"]["stream"] is False
    assert captured["form_data"]["metadata"] == pipe_metadata
    assert "stream_options" not in captured["form_data"]
    assert captured["user"].id == "user-1"
    assert type(captured["user"]) is fake_user_model
    assert captured["request_state"].bypass_filter is True
    assert captured["bypass_filter"] is True


@pytest.mark.asyncio
async def test_pipe_non_streaming_retries_with_compaction_after_context_error(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def get_or_create_checkpoint_summary(**kwargs):
        return "retry summary"

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        calls.append(copy_body(form_data))
        if len(calls) == 1:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "maximum context length exceeded",
                    }
                },
            )
        return {
            "id": "chatcmpl-target",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    def copy_body(body):
        return {
            key: value
            for key, value in body.items()
            if key != "metadata"
        } | {"metadata": body.get("metadata")}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "previous_response_id": "resp-old",
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2
    assert calls[0]["messages"] == body["messages"]
    assert "previous_response_id" not in calls[1]
    assert "retry summary" in calls[1]["messages"][0]["content"]


def test_chat_completion_response_requires_core_template(monkeypatch):
    misc_module = types.ModuleType("open_webui.utils.misc")
    monkeypatch.setitem(sys.modules, "open_webui.utils.misc", misc_module)

    with pytest.raises(ImportError):
        mod._chat_completion_message_response("target", "content")


@pytest.mark.asyncio
async def test_non_streaming_stream_response_error_stays_error_dict():
    response = StreamingResponse(
        iter(
            [
                b'data: {"error": {"code": "rate_limit_exceeded", "message": "rate limited"}}\n\n',
            ]
        ),
        media_type="text/event-stream",
    )

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    assert result == {"error": {"code": "provider_error", "message": "rate limited"}}


@pytest.mark.asyncio
async def test_non_streaming_stream_response_skips_done_chunk():
    response = StreamingResponse(
        iter(
            [
                b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        ),
        media_type="text/event-stream",
    )

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_non_streaming_stream_response_reassembles_split_sse_event():
    response = StreamingResponse(
        iter(
            [
                b'data: {"choices": [{"delta": ',
                b'{"content": "ok"}}]}\n\n',
            ]
        ),
        media_type="text/event-stream",
    )

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_non_streaming_stream_response_retries_split_sse_context_error():
    response = StreamingResponse(
        iter(
            [
                b'data: {"error": {"code": "context_length_',
                b'exceeded", "message": "too long"}}\n\n',
            ]
        ),
        media_type="text/event-stream",
    )

    with pytest.raises(mod.RetryableContextOverflow):
        await mod._coerce_non_streaming_completion_response(response, model_id="target")


@pytest.mark.asyncio
async def test_non_streaming_stream_response_preserves_tool_calls():
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"query":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"open webui"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    response = StreamingResponse(
        iter([f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks]),
        media_type="text/event-stream",
    )

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query":"open webui"}'},
            }
        ],
    }


@pytest.mark.asyncio
async def test_non_streaming_http_error_response_stays_error_dict_without_error_shape():
    response = JSONResponse(status_code=429, content={"message": "rate limited"})

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    assert result == {"error": {"code": "provider_error", "message": "rate limited"}}


@pytest.mark.asyncio
async def test_non_streaming_unknown_response_type_stays_error_dict():
    result = await mod._coerce_non_streaming_completion_response(object(), model_id="target")

    assert result == {
        "error": {
            "code": "unsupported_target_response",
            "message": "Unsupported target response type: object",
        }
    }


@pytest.mark.asyncio
async def test_pipe_rejects_missing_configured_summary_model(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def forward_target(**kwargs):
        raise AssertionError("target must not be called when summary_model is invalid")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.summary_model = "missing.summary"
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "invalid_summary_model"
    assert "missing.summary" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_compacts_above_threshold_and_removes_previous_response_id(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 400, "output_tokens": 100}

    async def get_or_create_checkpoint_summary(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "summary text"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "previous_response_id": "resp-old",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    events = []

    async def event_emitter(event):
        events.append(event)

    await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    forwarded = captured["forward_body"]
    assert "previous_response_id" not in forwarded
    assert forwarded["messages"][0] == {"role": "system", "content": "system"}
    assert forwarded["messages"][1]["role"] == "user"
    assert "summary text" in forwarded["messages"][1]["content"]
    assert forwarded["messages"][-1] == {"role": "user", "content": "active"}
    assert captured["source_messages"] == [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    assert [event["data"]["action"] for event in events if event["type"] == "status"] == [
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]
    embed_events = [event for event in events if event["type"] == "embeds"]
    assert len(embed_events) == 1
    assert embed_events[0]["data"]["replace"] is False
    assert len(embed_events[0]["data"]["embeds"]) == 1
    embed_html = embed_events[0]["data"]["embeds"][0]
    assert "summary text" in embed_html


@pytest.mark.asyncio
async def test_pipe_reuses_existing_checkpoint_even_when_previous_compacted_usage_is_below_threshold(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    exact_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "older follow-up"},
        {"role": "assistant", "content": "older follow-up answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(exact_source),
        "source_hash": mod.compute_source_hash(exact_source),
        "summary_text": "existing summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def noop_initialize(**kwargs):
        return None

    class ExistingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            assert kwargs["pipe_function_id"] == "auto_compact"
            if kwargs["source_hash"] == checkpoint["source_hash"]:
                return checkpoint
            return None

        async def find_longest_parent(self, **kwargs):
            return None

        async def insert_ready(self, row):
            raise AssertionError("exact checkpoint hit must not insert a new row")

        async def touch(self, checkpoint_id):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("exact checkpoint hit must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *exact_source,
            {"role": "user", "content": "active"},
        ],
    }
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert captured["touched"] == "checkpoint-1"
    forwarded = captured["forward_body"]
    assert "existing summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [{"role": "user", "content": "active"}]
    assert events == []


@pytest.mark.asyncio
async def test_pipe_does_not_forward_raw_history_when_checkpoint_lookup_fails(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def forward_target(**kwargs):
        raise AssertionError("raw history must not be forwarded when checkpoint lookup fails")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "checkpoint_unavailable"
    assert "checkpoint db down" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_creates_tool_checkpoint_without_history_parent_when_summary_fits(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            for row in rows:
                if row["source_hash"] == kwargs["source_hash"]:
                    return row
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent(rows, kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            rows.append(row)
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return f"summary {len(summary_inputs)}"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "x" * 200000}
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *history,
            active,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            tool_message,
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    tool_source = [*history, active]
    assert summary_inputs == [tool_source]
    assert len(rows) == 1
    assert rows[0]["source_message_count"] == len(tool_source)
    assert rows[0]["source_hash"] == mod.compute_source_hash(tool_source)
    assert rows[0]["parent_checkpoint_id"] is None
    messages = captured["forward_body"]["messages"]
    assert "summary 1" in messages[0]["content"]
    assert messages[1:] == body["messages"][3:]


@pytest.mark.asyncio
async def test_tool_loop_summary_extends_existing_history_parent_when_summary_fits(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    inserted = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    tool_source = [*history, active]
    history_checkpoint = {
        "id": "history-checkpoint-1",
        "state": "ready",
        "source_message_count": len(history),
        "source_hash": mod.compute_source_hash(history),
        "summary_text": "existing history summary",
        "summary_meta": {},
    }

    async def noop_initialize(**kwargs):
        return None

    class ExistingHistoryCheckpointStore:
        async def lookup_ready(self, **kwargs):
            if kwargs["source_hash"] == history_checkpoint["source_hash"]:
                return history_checkpoint
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent([history_checkpoint], kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            inserted.append(row)
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "direct tool summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingHistoryCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "messages": [
            *tool_source,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
        ],
    }

    compacted, did_compact = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert did_compact is True
    assert len(summary_inputs) == 1
    assert "existing history summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [active]
    assert len(inserted) == 1
    assert inserted[0]["source_hash"] == mod.compute_source_hash(tool_source)
    assert inserted[0]["parent_checkpoint_id"] == history_checkpoint["id"]
    assert "direct tool summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][1:] == body["messages"][3:]


@pytest.mark.asyncio
async def test_pipe_reuses_exact_tool_checkpoint_without_creating_history_checkpoint(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    touched = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    old_tool_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    tool_source = [*history, active, *old_tool_round]
    checkpoint = {
        "id": "tool-checkpoint-1",
        "state": "ready",
        "source_message_count": len(tool_source),
        "source_hash": mod.compute_source_hash(tool_source),
        "summary_text": "existing tool summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    class ExistingToolCheckpointStore:
        async def lookup_ready(self, **kwargs):
            if kwargs["source_hash"] == checkpoint["source_hash"]:
                return checkpoint
            return None

        async def find_longest_parent(self, **kwargs):
            return None

        async def insert_ready(self, row):
            raise AssertionError("exact tool checkpoint hit must not insert")

        async def touch(self, checkpoint_id):
            touched.append(checkpoint_id)
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("exact tool checkpoint hit must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingToolCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [*tool_source, *latest_round],
    }
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert touched == ["tool-checkpoint-1"]
    messages = captured["forward_body"]["messages"]
    assert "existing tool summary" in messages[0]["content"]
    assert messages[1:] == latest_round
    assert events == []


@pytest.mark.asyncio
async def test_pipe_reuses_exact_tool_checkpoint_even_when_usage_is_below_threshold(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    touched = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    old_tool_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    tool_source = [*history, active, *old_tool_round]
    checkpoint = {
        "id": "tool-checkpoint-1",
        "state": "ready",
        "source_message_count": len(tool_source),
        "source_hash": mod.compute_source_hash(tool_source),
        "summary_text": "existing tool summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def noop_initialize(**kwargs):
        return None

    class ExistingToolCheckpointStore:
        async def lookup_ready(self, **kwargs):
            if kwargs["source_hash"] == checkpoint["source_hash"]:
                return checkpoint
            return None

        async def find_longest_parent(self, **kwargs):
            raise AssertionError("exact tool checkpoint reuse must not need a parent lookup")

        async def insert_ready(self, row):
            raise AssertionError("exact tool checkpoint hit must not insert")

        async def touch(self, checkpoint_id):
            touched.append(checkpoint_id)
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("exact tool checkpoint hit must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingToolCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [*tool_source, *latest_round],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert touched == ["tool-checkpoint-1"]
    messages = captured["forward_body"]["messages"]
    assert "existing tool summary" in messages[0]["content"]
    assert messages[1:] == latest_round
    assert events == []


@pytest.mark.asyncio
async def test_pipe_reuses_history_checkpoint_for_tool_loop_when_usage_is_below_threshold(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    history_checkpoint = {
        "id": "history-checkpoint-1",
        "state": "ready",
        "source_message_count": len(history),
        "source_hash": mod.compute_source_hash(history),
        "summary_text": "existing history summary",
        "summary_meta": {},
    }
    rows.append(history_checkpoint)

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            for row in rows:
                if row["source_hash"] == kwargs["source_hash"]:
                    return row
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent(rows, kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            raise AssertionError("below-threshold checkpoint reuse must not create a new checkpoint")

        async def touch(self, checkpoint_id):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("below-threshold checkpoint reuse must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [*history, active, *latest_round],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert summary_inputs == []
    assert len(rows) == 1
    assert captured["touched"] == "history-checkpoint-1"
    messages = captured["forward_body"]["messages"]
    assert "existing history summary" in messages[0]["content"]
    assert messages[1:] == [active, *latest_round]


@pytest.mark.asyncio
async def test_pipe_reuses_longest_history_parent_for_tool_loop_when_usage_is_below_threshold(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    rows = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    short_parent_source = history[:1]
    short_parent = {
        "id": "short-parent-1",
        "state": "ready",
        "source_message_count": len(short_parent_source),
        "source_hash": mod.compute_source_hash(short_parent_source),
        "summary_text": "short parent summary",
        "summary_meta": {},
    }
    rows.append(short_parent)

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            for row in rows:
                if row["source_hash"] == kwargs["source_hash"]:
                    return row
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent(rows, kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            raise AssertionError("below-threshold checkpoint reuse must not create a new checkpoint")

        async def touch(self, checkpoint_id):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("below-threshold checkpoint reuse must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [*history, active, *latest_round],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert len(rows) == 1
    assert captured["touched"] == "short-parent-1"
    messages = captured["forward_body"]["messages"]
    assert "short parent summary" in messages[0]["content"]
    assert messages[1:] == [history[1], active, *latest_round]


@pytest.mark.asyncio
async def test_pipe_falls_back_to_history_parent_when_direct_tool_summary_overflows(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    tool_source = [*history, active]

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            for row in rows:
                if row["source_hash"] == kwargs["source_hash"]:
                    return row
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent(rows, kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            rows.append(row)
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        if kwargs["source_messages"] == tool_source:
            raise mod.RetryableContextOverflow("summary context")
        if kwargs["source_messages"] == history:
            return "history summary"
        return "tool summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *tool_source,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert summary_inputs[0] == tool_source
    assert summary_inputs[1] == history
    assert summary_inputs[2][0]["role"] == "user"
    assert "history summary" in summary_inputs[2][0]["content"]
    assert summary_inputs[2][1] == active
    assert len(rows) == 2
    assert rows[0]["source_hash"] == mod.compute_source_hash(history)
    assert rows[1]["source_hash"] == mod.compute_source_hash(tool_source)
    assert rows[1]["parent_checkpoint_id"] == rows[0]["id"]
    messages = captured["forward_body"]["messages"]
    assert "tool summary" in messages[0]["content"]
    assert messages[1:] == body["messages"][3:]


@pytest.mark.asyncio
async def test_pipe_falls_back_to_history_checkpoint_when_existing_tool_parent_retry_overflows(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    tool_source = [*history, active]
    short_parent_source = history[:1]
    short_parent = {
        "id": "short-parent-1",
        "state": "ready",
        "source_message_count": len(short_parent_source),
        "source_hash": mod.compute_source_hash(short_parent_source),
        "summary_text": "short parent summary",
        "summary_meta": {},
    }
    rows.append(short_parent)

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            for row in rows:
                if row["source_hash"] == kwargs["source_hash"]:
                    return row
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent(rows, kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            rows.append(row)
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        source = kwargs["source_messages"]
        summary_inputs.append(source)
        if source == tool_source:
            raise mod.RetryableContextOverflow("direct tool source overflow")
        if source and "short parent summary" in source[0].get("content", "") and source[1:] == [history[1], active]:
            raise mod.RetryableContextOverflow("short parent delta overflow")
        if source and "short parent summary" in source[0].get("content", "") and source[1:] == [history[1]]:
            return "history summary"
        if source and "history summary" in source[0].get("content", "") and source[1:] == [active]:
            return "tool summary"
        raise AssertionError(f"unexpected summary input: {source!r}")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "messages": [
            *tool_source,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
        ],
    }

    compacted, did_compact = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert did_compact is True
    assert "short parent summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [history[1], active]
    assert "short parent summary" in summary_inputs[1][0]["content"]
    assert summary_inputs[1][1:] == [history[1]]
    assert "history summary" in summary_inputs[2][0]["content"]
    assert summary_inputs[2][1:] == [active]
    assert len(rows) == 3
    assert rows[1]["source_hash"] == mod.compute_source_hash(history)
    assert rows[1]["parent_checkpoint_id"] == short_parent["id"]
    assert rows[2]["source_hash"] == mod.compute_source_hash(tool_source)
    assert rows[2]["parent_checkpoint_id"] == rows[1]["id"]
    assert "tool summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][1:] == body["messages"][3:]


@pytest.mark.asyncio
async def test_history_fallback_does_not_reselect_failed_longer_tool_parent(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    old_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    failed_parent_source = [*history, active]
    failed_parent = {
        "id": "failed-parent-1",
        "state": "ready",
        "source_message_count": len(failed_parent_source),
        "source_hash": mod.compute_source_hash(failed_parent_source),
        "summary_text": "failed parent summary",
        "summary_meta": {},
    }
    rows.append(failed_parent)
    tool_source = [*failed_parent_source, *old_round]

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            for row in rows:
                if row["source_hash"] == kwargs["source_hash"]:
                    return row
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent(rows, kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            rows.append(row)
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        source = kwargs["source_messages"]
        summary_inputs.append(source)
        if source == tool_source:
            raise mod.RetryableContextOverflow("direct tool source overflow")
        if source and "failed parent summary" in source[0].get("content", "") and source[1:] == old_round:
            raise mod.RetryableContextOverflow("failed parent delta overflow")
        if source == history:
            return "history summary"
        if source and "history summary" in source[0].get("content", "") and source[1:] == [active, *old_round]:
            return "tool summary"
        raise AssertionError(f"unexpected summary input: {source!r}")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {"messages": [*tool_source, *latest_round]}

    compacted, did_compact = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert did_compact is True
    assert "failed parent summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == old_round
    assert summary_inputs[1] == history
    assert "history summary" in summary_inputs[2][0]["content"]
    assert summary_inputs[2][1:] == [active, *old_round]
    assert len(rows) == 3
    assert rows[1]["source_hash"] == mod.compute_source_hash(history)
    assert rows[2]["source_hash"] == mod.compute_source_hash(tool_source)
    assert rows[2]["parent_checkpoint_id"] == rows[1]["id"]
    assert "tool summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][1:] == latest_round


@pytest.mark.asyncio
async def test_tool_loop_parent_retry_does_not_swallow_non_context_error(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    tool_source = [*history, active]
    short_parent_source = history[:1]
    rows.append(
        {
            "id": "short-parent-1",
            "state": "ready",
            "source_message_count": len(short_parent_source),
            "source_hash": mod.compute_source_hash(short_parent_source),
            "summary_text": "short parent summary",
            "summary_meta": {},
        }
    )

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            for row in rows:
                if row["source_hash"] == kwargs["source_hash"]:
                    return row
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent(rows, kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            rows.append(row)
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        source = kwargs["source_messages"]
        summary_inputs.append(source)
        if source == tool_source:
            raise mod.RetryableContextOverflow("direct tool source overflow")
        if source and "short parent summary" in source[0].get("content", "") and source[1:] == [history[1], active]:
            raise RuntimeError("parent retry failed")
        raise AssertionError(f"unexpected summary input: {source!r}")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "messages": [
            *tool_source,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
        ],
    }

    with pytest.raises(mod.ParentCheckpointExtensionFailed) as exc_info:
        await mod._compact_body(
            request=pipe_request,
            user=pipe_user,
            metadata={"chat_id": "chat-1"},
            body=body,
            pipe_function_id="auto_compact",
            target_model_id="target",
            summary_model_id="target",
            historical_message_excerpt_bytes=0,
            historical_message_excerpt_count=0,
        )

    assert str(exc_info.value.original) == "parent retry failed"
    assert len(summary_inputs) == 1
    assert "short parent summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [history[1], active]
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_tool_loop_checkpoints_extend_previous_tool_checkpoint_without_history_parent(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    active = {"role": "user", "content": "active"}
    round_1 = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result 1"},
    ]
    round_2 = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "result 2"},
    ]
    round_3 = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-3", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-3", "content": "result 3"},
    ]

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            for row in rows:
                if row["source_hash"] == kwargs["source_hash"]:
                    return row
            return None

        async def find_longest_parent(self, **kwargs):
            return mod.select_longest_matching_parent(rows, kwargs["prefix_hashes"])

        async def insert_ready(self, row):
            rows.append(row)
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        if kwargs["source_messages"] == [active, *round_1, *round_2]:
            raise mod.RetryableContextOverflow("summary context")
        return f"summary {len(summary_inputs)}"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    metadata = {"chat_id": "chat-1"}
    first_body = {
        "messages": [
            active,
            *round_1,
            *round_2,
        ],
    }
    first_compacted, first_did_compact = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata=metadata,
        body=first_body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )
    second_body = {
        "messages": [
            active,
            *round_1,
            *round_2,
            *round_3,
        ],
    }
    second_compacted, second_did_compact = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata=metadata,
        body=second_body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert first_did_compact is True
    assert second_did_compact is True
    assert summary_inputs[0] == [active, *round_1]
    assert summary_inputs[1][0]["role"] == "user"
    assert "summary 1" in summary_inputs[1][0]["content"]
    assert summary_inputs[1][1:] == round_2
    assert len(rows) == 2
    assert rows[0]["source_hash"] == mod.compute_source_hash([active, *round_1])
    assert rows[0]["parent_checkpoint_id"] is None
    assert rows[1]["source_hash"] == mod.compute_source_hash([active, *round_1, *round_2])
    assert rows[1]["parent_checkpoint_id"] == rows[0]["id"]
    assert first_compacted["messages"][1:] == round_2
    assert second_compacted["messages"][1:] == round_3


@pytest.mark.asyncio
async def test_pipe_compacts_history_before_latest_tool_round_when_prior_checkpoint_exists(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    summary_inputs = []
    exact_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(exact_source),
        "source_hash": mod.compute_source_hash(exact_source),
        "summary_text": "existing summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    class ExistingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            if kwargs["source_hash"] == checkpoint["source_hash"]:
                return checkpoint
            return None

        async def find_longest_parent(self, **kwargs):
            return checkpoint

        async def insert_ready(self, row):
            captured["inserted"] = row
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined active summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "x" * 200000}
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *exact_source,
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            tool_message,
        ],
    }
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(summary_inputs) == 1
    assert "existing summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [{"role": "user", "content": "active"}]
    checkpoint_source = [*exact_source, {"role": "user", "content": "active"}]
    assert captured["inserted"]["parent_checkpoint_id"] == checkpoint["id"]
    assert captured["inserted"]["source_message_count"] == len(checkpoint_source)
    assert captured["inserted"]["source_hash"] == mod.compute_source_hash(checkpoint_source)
    messages = captured["forward_body"]["messages"]
    assert "combined active summary" in messages[0]["content"]
    assert messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call-1", "type": "function"}],
    }
    assert messages[2] == tool_message
    assert [event["type"] for event in events] == ["status", "status", "embeds"]


@pytest.mark.asyncio
async def test_pipe_compacts_history_before_latest_tool_round_when_checkpoint_parent_includes_retained_history(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    summary_inputs = []
    exact_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    retained_old = [
        {"role": "user", "content": "retained 1"},
        {"role": "assistant", "content": "retained answer 1"},
        {"role": "user", "content": "retained 2"},
    ]
    checkpoint_source = [*exact_source, *retained_old]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(checkpoint_source),
        "source_hash": mod.compute_source_hash(checkpoint_source),
        "summary_text": "existing summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    class ExistingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            if kwargs["source_hash"] == checkpoint["source_hash"]:
                return checkpoint
            return None

        async def find_longest_parent(self, **kwargs):
            return checkpoint

        async def insert_ready(self, row):
            captured["inserted"] = row
            return row

        async def touch(self, checkpoint_id):
            return True

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined active summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "x" * 200000}
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *exact_source,
            *retained_old,
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            tool_message,
        ],
    }
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(summary_inputs) == 1
    assert "existing summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [{"role": "user", "content": "active"}]
    checkpoint_source_with_active = [*checkpoint_source, {"role": "user", "content": "active"}]
    assert captured["inserted"]["parent_checkpoint_id"] == checkpoint["id"]
    assert captured["inserted"]["source_message_count"] == len(checkpoint_source_with_active)
    assert captured["inserted"]["source_hash"] == mod.compute_source_hash(checkpoint_source_with_active)
    messages = captured["forward_body"]["messages"]
    assert "combined active summary" in messages[0]["content"]
    assert messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call-1", "type": "function"}],
    }
    assert messages[2] == tool_message
    assert [event["type"] for event in events] == ["status", "status", "embeds"]


@pytest.mark.asyncio
async def test_pipe_retries_with_compaction_after_pre_emission_context_error(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def get_or_create_checkpoint_summary(**kwargs):
        return "retry summary"

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        if len(calls) == 1:
            raise mod.RetryableContextOverflow("context")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "previous_response_id": "resp-old",
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert len(calls) == 2
    assert calls[0]["messages"] == body["messages"]
    assert "previous_response_id" not in calls[1]
    assert "retry summary" in calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_stops_context_retries_at_fixed_budget(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def get_or_create_checkpoint_summary(**kwargs):
        return "retry summary"

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        raise mod.RetryableContextOverflow("context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert len(calls) == mod.MAX_CONTEXT_RETRY_ATTEMPTS
    assert result["error"]["code"] == "context_window_exceeded"


@pytest.mark.asyncio
async def test_pipe_does_not_retry_non_context_target_errors(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        raise RuntimeError("rate limit")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    with pytest.raises(RuntimeError, match="rate limit"):
        await pipe.pipe(
            {
                "model": wrapper_id,
                "stream": True,
                "messages": [{"role": "user", "content": "active"}],
            },
            __request__=pipe_request,
            __user__=pipe_user,
            __metadata__=pipe_metadata,
        )

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_pipe_retries_tool_loop_by_summarizing_history_before_latest_tool_round(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    calls = []
    summary_inputs = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def get_or_create_compaction_summary(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined retry summary"

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        if len(calls) == 1:
            raise mod.RetryableContextOverflow("context")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "x" * 10000}
    events = []

    async def event_emitter(event):
        events.append(event)

    body = {
        "model": wrapper_id,
        "stream": True,
        "previous_response_id": "resp-old",
        "messages": [
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            tool_message,
        ],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(summary_inputs) == 1
    assert summary_inputs[0] == [{"role": "user", "content": "active"}]
    retry_messages = calls[1]["messages"]
    assert "combined retry summary" in retry_messages[0]["content"]
    assert retry_messages[1:] == body["messages"][1:]
    assert "previous_response_id" not in calls[1]
    assert [event["data"]["action"] for event in events if event["type"] == "status"] == [
        "auto_compaction_retry",
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]
    status_events = [event for event in events if event["type"] == "status"]
    assert status_events[-1]["data"]["done"] is True
    embed_events = [event for event in events if event["type"] == "embeds"]
    assert len(embed_events) == 1
    assert embed_events[0]["data"]["replace"] is False
    assert len(embed_events[0]["data"]["embeds"]) == 1
    embed_html = embed_events[0]["data"]["embeds"][0]
    assert "combined retry summary" in embed_html


@pytest.mark.asyncio
async def test_pipe_reemits_latest_compaction_summary_embed_on_retry_after_initial_compaction(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    calls = []
    summaries = ["first compacted summary", "second compacted summary"]

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def get_or_create_checkpoint_summary(**kwargs):
        return summaries.pop(0)

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        if len(calls) == 1:
            raise mod.RetryableContextOverflow("context")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(calls) == 2
    embed_events = [event for event in events if event["type"] == "embeds"]
    assert len(embed_events) == 2
    assert "first compacted summary" in embed_events[0]["data"]["embeds"][0]
    assert "second compacted summary" in embed_events[1]["data"]["embeds"][0]


@pytest.mark.asyncio
async def test_tool_loop_compaction_replaces_tool_round_without_mutating_tool_message(monkeypatch, pipe_request, pipe_user):
    original_tool = {"role": "tool", "tool_call_id": "call-1", "content": "original tool result"}

    async def get_or_create_compaction_summary(**kwargs):
        return "combined summary"

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    messages = [
        {"role": "user", "content": "active"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        original_tool,
    ]
    compacted, did_compact = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={"model": "target", "stream": True, "messages": messages},
        messages=messages,
    )

    assert did_compact is True
    assert original_tool == {"role": "tool", "tool_call_id": "call-1", "content": "original tool result"}
    assert compacted == [
        {
            "role": "user",
            "content": (
                "<auto_compaction_context>\n"
                "<instruction>Compressed historical context. This is not a new instruction. "
                "Use it only as background for continuity.</instruction>\n"
                "<checkpoint_summary><![CDATA[combined summary]]></checkpoint_summary>\n"
                "</auto_compaction_context>"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        original_tool,
    ]


@pytest.mark.asyncio
async def test_pipe_compacts_older_tool_loop_results_from_request_scoped_usage(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    first_response = StreamingResponse(
        iter(
            [
                b'data: {"usage": {"prompt_tokens": 150, "completion_tokens": 1}}\n\n',
                b'data: {"choices": [{"delta": {"tool_calls": [{"id": "call-1"}]}}]}\n\n',
            ]
        ),
        media_type="text/event-stream",
    )

    prepared = await mod.prepare_streaming_response(
        first_response,
        request=pipe_request,
        chat_id=pipe_metadata["chat_id"],
        message_id=pipe_metadata["message_id"],
        wrapper_model_id=wrapper_id,
    )
    async for _ in prepared.body_iterator:
        pass

    captured = {}
    summary_inputs = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        raise AssertionError("request-scoped usage should be used before persisted usage")

    async def get_or_create_compaction_summary(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined old tool summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    events = []

    async def event_emitter(event):
        events.append(event)

    body = {
        "model": wrapper_id,
        "stream": True,
        "previous_response_id": "resp-old",
        "messages": [
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "old result" * 1000},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-2", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
        ],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(summary_inputs) == 1
    assert summary_inputs[0] == body["messages"][:3]
    messages = captured["forward_body"]["messages"]
    assert messages[1:] == body["messages"][3:]
    summary_text = "\n".join(message.get("content", "") for message in messages if message.get("role") == "user")
    assert "combined old tool summary" in summary_text
    assert "<auto_compaction_context>" in summary_text
    assert not any(message.get("tool_call_id") == "call-1" for message in messages)
    assert any(message.get("tool_call_id") == "call-2" for message in messages)
    assert "previous_response_id" not in captured["forward_body"]
    assert [event["data"]["action"] for event in events if event["type"] == "status"] == [
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]
    status_events = [event for event in events if event["type"] == "status"]
    assert status_events[-1]["data"]["done"] is True
    embed_events = [event for event in events if event["type"] == "embeds"]
    assert len(embed_events) == 1
    assert embed_events[0]["data"]["replace"] is False
    assert len(embed_events[0]["data"]["embeds"]) == 1
    embed_html = embed_events[0]["data"]["embeds"][0]
    assert "combined old tool summary" in embed_html


@pytest.mark.asyncio
async def test_pipe_closes_compaction_status_when_usage_threshold_has_no_safe_prefix(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        return {"ok": True, "messages": kwargs["body"]["messages"]}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "active"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True, "messages": [{"role": "user", "content": "active"}]}
    status_events = [event for event in events if event["type"] == "status"]
    assert [event["data"]["action"] for event in status_events] == [
        "auto_compaction_compacting",
        "auto_compaction_skipped",
    ]
    assert status_events[-1]["data"]["done"] is True
    assert "Could not compact" in status_events[-1]["data"]["description"]


@pytest.mark.asyncio
async def test_pipe_returns_clear_error_when_latest_user_cannot_fit(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def forward_target(**kwargs):
        raise mod.RetryableContextOverflow("context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [{"role": "user", "content": "x" * 1000000}],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result["error"]["code"] == "active_input_too_large"
    assert "latest user message" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_returns_clear_error_when_latest_tool_result_cannot_be_summarized(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def generate_summary_text(**kwargs):
        raise mod.RetryableContextOverflow("summary context")

    async def forward_target(**kwargs):
        raise mod.RetryableContextOverflow("target context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "x" * 1000000},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result["error"]["code"] == "latest_tool_result_too_large"
    assert "tool result" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_does_not_compact_internal_summary_task(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 999999, "input_tokens": 999999, "output_tokens": 0}

    async def get_or_create_checkpoint_summary(**kwargs):
        raise AssertionError("summary task must not recursively compact")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }
    metadata = {**pipe_metadata, "task": mod.INTERNAL_SUMMARY_TASK}

    await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)

    assert captured["forward_body"]["messages"] == body["messages"]
