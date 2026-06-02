"""Tests for functions/filter/inlet_title_generator.py."""

from __future__ import annotations

import sys
import types
import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from functions.filter import inlet_title_generator as mod
from functions.filter.inlet_title_generator import Filter


class DummyRedis:
    def __init__(self, set_result=True):
        self.set_result = set_result
        self.calls = []

    async def set(self, key, value, ex=None, nx=False):
        self.calls.append({"key": key, "value": value, "ex": ex, "nx": nx})
        return self.set_result


@pytest.fixture
def events():
    return []


@pytest.fixture
def emitter(events):
    async def _emit(event):
        events.append(event)

    return _emit


@pytest.fixture
def dummy_request():
    return SimpleNamespace()


@pytest.fixture
def user():
    return {
        "id": "user-1",
        "email": "user@example.com",
        "name": "Test User",
        "role": "user",
        "last_active_at": 0,
        "updated_at": 0,
        "created_at": 0,
        "valves": {},
    }


@pytest.fixture
def metadata():
    return {
        "chat_id": "chat-1",
        "user_message_id": "u1",
    }


@pytest.fixture
def body():
    return {
        "model": "gpt-test",
        "messages": [{"id": "u1", "role": "user", "content": "Write about clocks"}],
    }


def install_open_webui_stubs(
    monkeypatch,
    *,
    title="New Chat",
    messages_map=None,
    generated_response=None,
    title_before_update=None,
    generate_title_hook=None,
):
    messages_map = messages_map or {}
    generated_response = generated_response or {
        "choices": [{"message": {"content": '{"title": "Generated Title"}'}}]
    }
    state = {
        "generate_calls": [],
        "update_calls": [],
        "title_reads": 0,
    }

    async def get_chat_title_by_id(chat_id):
        state["title_reads"] += 1
        if state["title_reads"] >= 2 and title_before_update is not None:
            return title_before_update
        return title

    async def get_messages_map_by_chat_id(chat_id):
        return messages_map

    async def update_chat_title_by_id(chat_id, new_title):
        state["update_calls"].append((chat_id, new_title))
        return SimpleNamespace(id=chat_id, title=new_title)

    async def generate_title(request, form_data, user):
        state["generate_calls"].append((request, form_data, user))
        if generate_title_hook is not None:
            await generate_title_hook(request, form_data, user)
        return generated_response

    def get_message_list(messages, message_id):
        current = messages.get(message_id)
        result = []
        seen = set()
        while current:
            current_id = current.get("id")
            if current_id in seen:
                break
            seen.add(current_id)
            result.append(current)
            parent_id = current.get("parentId")
            current = messages.get(parent_id) if parent_id else None
        result.reverse()
        return result

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = SimpleNamespace(
        get_chat_title_by_id=get_chat_title_by_id,
        get_messages_map_by_chat_id=get_messages_map_by_chat_id,
        update_chat_title_by_id=update_chat_title_by_id,
    )
    users_module = types.ModuleType("open_webui.models.users")

    class UserModel:
        def __init__(self, **kwargs):
            missing = {
                "last_active_at",
                "updated_at",
                "created_at",
            } - set(kwargs)
            if missing:
                raise ValueError(f"missing user timestamps: {sorted(missing)}")
            self.data = kwargs

        def model_dump(self):
            return self.data

    users_module.UserModel = UserModel
    tasks_module = types.ModuleType("open_webui.routers.tasks")
    tasks_module.generate_title = generate_title
    misc_module = types.ModuleType("open_webui.utils.misc")
    misc_module.get_message_list = get_message_list

    open_webui_module = types.ModuleType("open_webui")
    open_webui_module.__path__ = []
    models_module = types.ModuleType("open_webui.models")
    models_module.__path__ = []
    models_module.chats = chats_module
    models_module.users = users_module

    monkeypatch.setitem(sys.modules, "open_webui", open_webui_module)
    monkeypatch.setitem(sys.modules, "open_webui.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.users", users_module)
    monkeypatch.setitem(sys.modules, "open_webui.routers", types.ModuleType("open_webui.routers"))
    monkeypatch.setitem(sys.modules, "open_webui.routers.tasks", tasks_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils", types.ModuleType("open_webui.utils"))
    monkeypatch.setitem(sys.modules, "open_webui.utils.misc", misc_module)

    return state


def install_user_model_stub(monkeypatch):
    users_module = types.ModuleType("open_webui.models.users")

    class UserModel:
        def __init__(self, **kwargs):
            missing = {
                "last_active_at",
                "updated_at",
                "created_at",
            } - set(kwargs)
            if missing:
                raise ValueError(f"missing user timestamps: {sorted(missing)}")
            self.data = kwargs

        def model_dump(self):
            return self.data

    users_module.UserModel = UserModel
    monkeypatch.setitem(sys.modules, "open_webui.models.users", users_module)
    return UserModel


def load_fresh_filter_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, mod.__file__)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_import_and_default_configuration():
    f = Filter()

    assert mod.PLACEHOLDER_TITLE == "New Chat"
    assert f.valves.priority == 100
    assert f.valves.dedup_time_window == 600
    assert f.valves.dedup_max_entries == 1000
    assert f.valves.avoid_core_initial_title_duplicate is True
    assert f.UserValves().enabled is True
    assert f.UserValves().show_status is False


def test_valves_defaults_read_websocket_redis_environment(monkeypatch):
    monkeypatch.setenv("WEBSOCKET_MANAGER", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://from-env")
    monkeypatch.delenv("WEBSOCKET_REDIS_URL", raising=False)

    fresh_mod = load_fresh_filter_module("inlet_title_generator_env_redis_url")
    f = fresh_mod.Filter()

    assert f.valves.use_redis is True
    assert f.valves.redis_url == "redis://from-env"


def test_valves_defaults_prefer_websocket_redis_url(monkeypatch):
    monkeypatch.setenv("WEBSOCKET_MANAGER", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://general")
    monkeypatch.setenv("WEBSOCKET_REDIS_URL", "redis://websocket")

    fresh_mod = load_fresh_filter_module("inlet_title_generator_env_websocket_redis_url")
    f = fresh_mod.Filter()

    assert f.valves.use_redis is True
    assert f.valves.redis_url == "redis://websocket"


def test_user_model_from_dict_returns_open_webui_user_model_when_available(monkeypatch, user):
    UserModel = install_user_model_stub(monkeypatch)

    result = mod.user_model_from_dict(user)

    assert isinstance(result, UserModel)
    assert result.model_dump()["id"] == "user-1"


@pytest.mark.asyncio
async def test_inlet_skips_invalid_requests_without_launch(dummy_request, user, metadata, body):
    f = Filter()
    launcher = AsyncMock()
    f._launch_background_task = launcher

    assert await f.inlet("bad", __request__=dummy_request, __user__=user, __metadata__=metadata) == "bad"
    assert await f.inlet(body, __request__=dummy_request, __user__={**user, "valves": {"enabled": False}}, __metadata__=metadata) == body
    assert await f.inlet(body, __request__=dummy_request, __user__=user, __metadata__={"chat_id": "local:abc", "user_message_id": "u1"}) == body
    assert await f.inlet(body, __request__=dummy_request, __user__=user, __metadata__={"chat_id": "channel:abc", "user_message_id": "u1"}) == body
    assert await f.inlet(body, __request__=dummy_request, __user__=user, __metadata__={"chat_id": "chat-1"}) == body
    assert await f.inlet(body, __request__=dummy_request, __user__=user, __metadata__=metadata, __task__="title_generation") == body

    launcher.assert_not_called()


@pytest.mark.asyncio
async def test_inlet_returns_body_and_only_launches_background_task(dummy_request, user, metadata, body):
    f = Filter()
    launched = []

    def launch(coro):
        launched.append(coro)
        coro.close()

    f._launch_background_task = launch

    result = await f.inlet(body, __request__=dummy_request, __event_emitter__=AsyncMock(), __user__=user, __metadata__=metadata, __id__="filter-1")

    assert result is body
    assert len(launched) == 1


@pytest.mark.asyncio
async def test_inlet_active_set_suppresses_same_chat_multiple_launches(dummy_request, user, metadata, body):
    f = Filter()
    launched = []

    def launch(coro):
        launched.append(coro)
        coro.close()

    f._launch_background_task = launch

    await f.inlet(body, __request__=dummy_request, __event_emitter__=AsyncMock(), __user__=user, __metadata__=metadata)
    await f.inlet(body, __request__=dummy_request, __event_emitter__=AsyncMock(), __user__=user, __metadata__=metadata)

    assert len(launched) == 1


@pytest.mark.asyncio
async def test_background_exits_when_title_is_not_placeholder(monkeypatch, dummy_request, emitter, user):
    state = install_open_webui_stubs(
        monkeypatch,
        title="Already titled",
        messages_map={"u1": {"id": "u1", "role": "user", "content": "Hi"}},
    )
    f = Filter()
    f.valves.use_redis = False

    await f._run_title_generation(
        request=dummy_request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="gpt-test",
        filter_id="filter-1",
        user_valves=f.UserValves(),
    )

    assert state["generate_calls"] == []
    assert state["update_calls"] == []


@pytest.mark.asyncio
async def test_background_skips_core_initial_root_chat(monkeypatch, dummy_request, emitter, user):
    state = install_open_webui_stubs(
        monkeypatch,
        messages_map={"u1": {"id": "u1", "role": "user", "content": "Hi", "parentId": None}},
    )
    f = Filter()
    f.valves.use_redis = False

    await f._run_title_generation(
        request=dummy_request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="gpt-test",
        filter_id="filter-1",
        user_valves=f.UserValves(),
    )

    assert state["generate_calls"] == []
    assert state["update_calls"] == []


@pytest.mark.asyncio
async def test_background_generates_initial_root_when_duplicate_avoidance_disabled(monkeypatch, dummy_request, emitter, user):
    state = install_open_webui_stubs(
        monkeypatch,
        messages_map={"u1": {"id": "u1", "role": "user", "content": "Hi", "parentId": None}},
    )
    f = Filter()
    f.valves.use_redis = False
    f.valves.avoid_core_initial_title_duplicate = False

    await f._run_title_generation(
        request=dummy_request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="gpt-test",
        filter_id="filter-1",
        user_valves=f.UserValves(),
    )

    assert state["generate_calls"][0][1]["messages"] == [
        {"id": "u1", "role": "user", "content": "Hi", "parentId": None}
    ]
    assert state["update_calls"] == [("chat-1", "Generated Title")]


@pytest.mark.asyncio
async def test_background_generates_root_resend_when_other_user_message_exists(monkeypatch, dummy_request, emitter, user):
    state = install_open_webui_stubs(
        monkeypatch,
        messages_map={
            "u0": {"id": "u0", "role": "user", "content": "Old root", "parentId": None},
            "u1": {"id": "u1", "role": "user", "content": "New root", "parentId": None},
        },
    )
    f = Filter()
    f.valves.use_redis = False

    await f._run_title_generation(
        request=dummy_request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="gpt-test",
        filter_id="filter-1",
        user_valves=f.UserValves(),
    )

    assert state["generate_calls"][0][1]["messages"] == [
        {"id": "u1", "role": "user", "content": "New root", "parentId": None}
    ]
    assert state["update_calls"] == [("chat-1", "Generated Title")]


@pytest.mark.asyncio
async def test_background_generates_non_root_followup(monkeypatch, dummy_request, emitter, user):
    state = install_open_webui_stubs(
        monkeypatch,
        messages_map={
            "u0": {"id": "u0", "role": "user", "content": "First", "parentId": None},
            "a0": {"id": "a0", "role": "assistant", "content": "Answer", "parentId": "u0"},
            "u1": {"id": "u1", "role": "user", "content": "Follow up", "parentId": "a0"},
        },
    )
    f = Filter()
    f.valves.use_redis = False

    await f._run_title_generation(
        request=dummy_request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="gpt-test",
        filter_id="filter-1",
        user_valves=f.UserValves(),
    )

    assert state["generate_calls"][0][1]["messages"] == [
        {"id": "u0", "role": "user", "content": "First", "parentId": None},
        {"id": "a0", "role": "assistant", "content": "Answer", "parentId": "u0"},
        {"id": "u1", "role": "user", "content": "Follow up", "parentId": "a0"},
    ]
    assert state["update_calls"] == [("chat-1", "Generated Title")]


@pytest.mark.asyncio
async def test_background_passes_core_normalized_messages_to_generate_title(
    monkeypatch,
    dummy_request,
    emitter,
    user,
):
    state = install_open_webui_stubs(
        monkeypatch,
        messages_map={
            "u0": {
                "id": "u0",
                "role": "user",
                "content": [{"type": "text", "text": "Root ![image](file.png) visible"}],
                "parentId": None,
            },
            "a0": {
                "id": "a0",
                "content": "Answer <details>hidden</details> visible",
                "parentId": "u0",
            },
            "u1": {
                "id": "u1",
                "role": "user",
                "content": [{"type": "text", "text": "Follow <details>secret</details> up"}],
                "parentId": "a0",
            },
        },
    )
    f = Filter()
    f.valves.use_redis = False

    await f._run_title_generation(
        request=dummy_request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="gpt-test",
        filter_id="filter-1",
        user_valves=f.UserValves(),
    )

    assert state["generate_calls"][0][1]["messages"] == [
        {
            "id": "u0",
            "role": "user",
            "content": "Root  visible",
            "parentId": None,
        },
        {
            "id": "a0",
            "role": "assistant",
            "content": "Answer  visible",
            "parentId": "u0",
        },
        {
            "id": "u1",
            "role": "user",
            "content": "Follow  up",
            "parentId": "a0",
        },
    ]


@pytest.mark.asyncio
async def test_background_title_generation_does_not_mutate_original_request_state_metadata(
    monkeypatch,
    emitter,
    user,
):
    async def mutate_title_request(request, form_data, user_model):
        request.state.metadata.pop("selected_model_id", None)
        request.state.metadata["title_task_only"] = True

    install_open_webui_stubs(
        monkeypatch,
        messages_map={
            "u0": {"id": "u0", "role": "user", "content": "Old", "parentId": None},
            "u1": {"id": "u1", "role": "user", "content": "New", "parentId": None},
        },
        generate_title_hook=mutate_title_request,
    )
    request = SimpleNamespace(
        state=SimpleNamespace(
            metadata={
                "selected_model_id": "arena-child",
                "chat_id": "chat-1",
            }
        )
    )
    f = Filter()
    f.valves.use_redis = False

    await f._run_title_generation(
        request=request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="arena",
        filter_id="filter-1",
        user_valves=f.UserValves(),
    )

    assert request.state.metadata == {
        "selected_model_id": "arena-child",
        "chat_id": "chat-1",
    }


@pytest.mark.asyncio
async def test_background_does_not_update_if_title_changes_before_save(monkeypatch, dummy_request, emitter, user):
    state = install_open_webui_stubs(
        monkeypatch,
        messages_map={
            "u0": {"id": "u0", "role": "user", "content": "Old", "parentId": None},
            "u1": {"id": "u1", "role": "user", "content": "New", "parentId": None},
        },
        title_before_update="Manual Title",
    )
    f = Filter()
    f.valves.use_redis = False

    await f._run_title_generation(
        request=dummy_request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="gpt-test",
        filter_id="filter-1",
        user_valves=f.UserValves(),
    )

    assert state["generate_calls"]
    assert state["update_calls"] == []


@pytest.mark.asyncio
async def test_background_does_not_update_when_core_endpoint_returns_non_dict(monkeypatch, dummy_request, emitter, user):
    state = install_open_webui_stubs(
        monkeypatch,
        messages_map={
            "u0": {"id": "u0", "role": "user", "content": "Old", "parentId": None},
            "u1": {"id": "u1", "role": "user", "content": "New", "parentId": None},
        },
        generated_response=SimpleNamespace(detail="Title generation is disabled"),
    )
    f = Filter()
    f.valves.use_redis = False

    await f._run_title_generation(
        request=dummy_request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="gpt-test",
        filter_id="filter-1",
        user_valves=f.UserValves(),
    )

    assert state["generate_calls"]
    assert state["update_calls"] == []


@pytest.mark.asyncio
async def test_background_emits_title_and_optional_status(monkeypatch, dummy_request, events, emitter, user):
    install_open_webui_stubs(
        monkeypatch,
        messages_map={
            "u0": {"id": "u0", "role": "user", "content": "Old", "parentId": None},
            "u1": {"id": "u1", "role": "user", "content": "New", "parentId": None},
        },
    )
    f = Filter()
    f.valves.use_redis = False

    await f._run_title_generation(
        request=dummy_request,
        event_emitter=emitter,
        user=user,
        chat_id="chat-1",
        user_message_id="u1",
        model_id="gpt-test",
        filter_id="filter-1",
        user_valves=f.UserValves(show_status=True),
    )

    assert {"type": "chat:title", "data": "Generated Title"} in events
    assert any(event.get("type") == "status" for event in events)


def test_core_title_extraction_variants():
    messages = [{"role": "user", "content": "Fallback from first"}]

    assert mod.extract_title_from_response(
        {"choices": [{"message": {"content": '{"title": "From content"}'}}]},
        messages,
        {"content": "User fallback"},
    ) == "From content"
    assert mod.extract_title_from_response(
        {"choices": [{"message": {"reasoning_content": '{"title": "From reasoning"}'}}]},
        messages,
        {"content": "User fallback"},
    ) == "From reasoning"
    assert mod.extract_title_from_response(
        {"choices": [{"message": {"content": "not json"}}]},
        messages,
        {"content": "User fallback"},
    ) == "Fallback from first"
    assert mod.extract_title_from_response(
        {"choices": [{"message": {"content": '{"title": ""}'}}]},
        messages,
        {"content": "User fallback"},
    ) == "Fallback from first"
    assert mod.extract_title_from_response(
        {"choices": [{"message": {"content": '{"title": "One"}'}}, {"message": {"content": '{"title": "Two"}'}}]},
        messages,
        {"content": "User fallback"},
    ) == "Fallback from first"


def test_core_title_extraction_uses_normalized_current_message_fallback():
    messages = mod.normalize_messages_for_title(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": '{"title": "From normalized text"}'}],
            }
        ]
    )

    assert mod.extract_title_from_response(
        {"choices": [{"message": {}}]},
        messages,
        messages[0],
    ) == "From normalized text"


def test_core_title_extraction_uses_normalized_first_message_fallback():
    messages = mod.normalize_messages_for_title(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Fallback from first text part"}],
            }
        ]
    )

    assert mod.extract_title_from_response(
        {"choices": [{"message": {"content": "not json"}}]},
        messages,
        {"role": "user", "content": "User fallback"},
    ) == "Fallback from first text part"


@pytest.mark.asyncio
async def test_redis_dedup_uses_set_nx_ex(monkeypatch):
    f = Filter()
    f.valves.use_redis = True
    f.valves.redis_url = "redis://example"
    f.valves.dedup_time_window = 123
    redis = DummyRedis()
    monkeypatch.setattr(f, "_get_redis_client", AsyncMock(return_value=redis))

    acquired = await f._acquire_dedup("chat-1", "u1", "filter-abc")

    assert acquired is True
    assert redis.calls == [
        {
            "key": "owui:filter-abc:title_dedup:chat-1:u1",
            "value": b"1",
            "ex": 123,
            "nx": True,
        }
    ]


@pytest.mark.asyncio
async def test_memory_dedup_and_cleanup(monkeypatch):
    f = Filter()
    f.valves.use_redis = False
    f.valves.dedup_max_entries = 1

    assert await f._acquire_dedup("chat-1", "u1", "filter-1") is True
    assert await f._acquire_dedup("chat-1", "u1", "filter-1") is False
    assert await f._acquire_dedup("chat-1", "u2", "filter-1") is True

    assert len(f._processed_titles) == 1
