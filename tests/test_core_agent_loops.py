"""Nested agent loops follow real Core for request state, prompts, Tool Search, skills and embeds."""

import asyncio
import collections
import copy
import importlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from open_webui.tools import builtin

import llm_review
import multi_model_council
import parallel_tools
import sub_agent

AGENT_MODULES = ["sub_agent", "llm_review", "magi_decision_support", "multi_model_council"]
LOOKUP_DESCRIPTION = "Look up one record by its key. " * 20

needs_tool_search = pytest.mark.skipif(
    importlib.util.find_spec("open_webui.utils.tool_search") is None,
    reason="Core before 0.12.0 has no Tool Search",
)
needs_core_skill_tools = pytest.mark.skipif(
    not hasattr(builtin, "read_skill_file"),
    reason="Core before 0.12.0 registers no skill tools in get_builtin_tools",
)
HAS_CALL_TOOL = importlib.util.find_spec("open_webui.utils.tool_search") is not None and hasattr(
    importlib.import_module("open_webui.utils.tool_search"), "resolve_tool_call"
)
needs_call_tool = pytest.mark.skipif(not HAS_CALL_TOOL, reason="Core before 0.12.2 has no call_tool")
SKILL_READ_TOOLS = {name for name in ("view_skill", "read_skill_file", "grep_skill_files") if hasattr(builtin, name)}
SKILL_MANIFEST = (
    "<available_skills>\n<skill>\n<id>skill</id>\n<name>Writer</name>\n"
    "<description></description>\n</skill>\n</available_skills>"
)


def make_user(role="user"):
    from open_webui.models.users import UserModel

    return UserModel(
        id="loop-user", email="loop@example.com", name="Loop User", role=role,
        last_active_at=0, updated_at=0, created_at=0,
    )


def make_request(model=None):
    return SimpleNamespace(
        state=SimpleNamespace(internal=False), body=AsyncMock(return_value=b"{}"),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={"model": model} if model else {})),
    )


def loop_tools():
    async def lookup(query: str = ""):
        return "found"

    async def ping():
        return "pong"

    return {
        name: {
            "callable": fn, "type": "tool", "tool_id": "records",
            "spec": {"name": name, "description": description, "parameters": {
                "type": "object", "properties": {"query": {"type": "string"}} if name == "lookup" else {},
            }},
        }
        for name, fn, description in (("lookup", lookup, LOOKUP_DESCRIPTION), ("ping", ping, "Ping."))
    }


def tool_call(name, arguments=None):
    return {"choices": [{"message": {"content": "", "tool_calls": [{
        "id": name, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {})},
    }]}}]}


def system_text(form_data):
    return "\n".join(
        content if isinstance(content, str) else json.dumps(content)
        for content in (message["content"] for message in form_data["messages"] if message["role"] == "system")
    )


def tool_names(form_data):
    return {tool["function"]["name"] for tool in form_data.get("tools") or [] if "function" in tool}


async def run_loop(
    monkeypatch, module_name, responses, *, tools, extra_params=None, messages=None, max_iterations=None, **kwargs,
):
    from open_webui.utils import chat
    from open_webui.utils.misc import add_or_update_system_message

    module = importlib.import_module(module_name)
    sent = []

    async def complete(*, request, form_data, user, bypass_filter):
        sent.append(copy.deepcopy(form_data))
        # Core applies the model's own system prompt to the request in place.
        add_or_update_system_message("Model prompt.", form_data["messages"])
        return responses[len(sent) - 1]

    monkeypatch.setattr(chat, "generate_chat_completion", complete)
    request = make_request()
    user = make_user()
    extra_params = {
        "__request__": request, "__user__": user.model_dump(), "__metadata__": {},
        **(extra_params or {}),
    }
    kwargs = dict(
        request=request, user=user, model_id="model", tools_dict=tools,
        messages=messages or [{"role": "user", "content": "Find the record"}],
        max_iterations=max_iterations or len(responses), apply_inlet_filters=False,
        extra_params=extra_params, **kwargs,
    )
    if module_name == "sub_agent":
        result = await module.run_sub_agent_loop(**kwargs)
    else:
        result = await module.run_agent_loop(**kwargs, agent_name="agent")
    return result, sent, extra_params


@pytest.fixture
def tool_search(monkeypatch):
    from open_webui.models.config import Config

    settings = {
        "chat.tool_search.enable": True, "chat.tool_search.defer_threshold": 400,
        "chat.tool_search.always_loaded": [], "chat.tool_search.defer_builtin_tools": False,
    }
    monkeypatch.setattr(Config, "get", AsyncMock(side_effect=lambda key, default=None: settings.get(key, default)))
    monkeypatch.setattr(Config, "get_many", AsyncMock(side_effect=lambda *keys: {key: settings.get(key) for key in keys}))


async def test_nested_completions_do_not_share_request_state():
    from starlette.requests import Request

    request = Request({"type": "http", "app": object(), "state": {"direct": False}})
    request._body = b"{}"

    metadata = {"model_id": "child"}
    isolated = sub_agent.isolate_request_state(request, metadata)
    isolated.state.direct = True
    isolated.state.model = {"id": "child"}

    assert request.state.direct is False
    assert not hasattr(request.state, "model")
    assert isolated.state.metadata is metadata
    assert not hasattr(request.state, "metadata")
    assert isolated.app is request.app
    assert await isolated.body() == b"{}"


@pytest.mark.parametrize("parallel", [False, True])
async def test_sub_agent_tool_runs_with_the_null_filter_ids_of_a_tool_approval_resume(monkeypatch, parallel):
    from open_webui.utils import chat

    monkeypatch.setattr(chat, "generate_chat_completion", AsyncMock(
        return_value={"choices": [{"message": {"content": "done"}}]}
    ))
    model = {"id": "model"}
    kwargs = {
        "__request__": make_request(model), "__user__": make_user().model_dump(), "__model__": model,
        "__metadata__": {
            "filter_ids": None, "chat_id": "local:resume", "session_id": "browser", "tool_servers": [],
        },
    }
    work = {"description": "Task", "prompt": "Work"}
    tool = sub_agent.Tools()

    if parallel:
        results = json.loads(await tool.run_parallel_sub_agents(tasks=[work], **kwargs))["results"]
        assert [item["result"] for item in results] == ["done"]
    else:
        assert json.loads(await tool.run_sub_agent(**work, **kwargs))["result"] == "done"


@pytest.mark.parametrize("module_name", AGENT_MODULES)
@pytest.mark.parametrize("server_prompts", [[], ["Server prompt."]])
async def test_system_prompts_do_not_accumulate_across_iterations(monkeypatch, module_name, server_prompts):
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Ping twice"}]
    original = copy.deepcopy(messages)

    result, sent, _ = await run_loop(
        monkeypatch, module_name,
        [tool_call("ping"), tool_call("ping"), {"choices": [{"message": {"content": "done"}}]}],
        tools=loop_tools(), messages=messages,
        extra_params={"__direct_tool_server_system_prompts__": server_prompts},
    )

    assert result == "done"
    for form_data in sent:
        text = system_text(form_data)
        assert (text.count("Be brief."), text.count("Server prompt."), text.count("Model prompt.")) == (
            1, len(server_prompts), 0,
        )
    assert messages == original


@needs_tool_search
@pytest.mark.parametrize("module_name", AGENT_MODULES)
async def test_agent_loops_apply_core_tool_search(monkeypatch, tool_search, module_name):
    result, sent, extra_params = await run_loop(
        monkeypatch, module_name,
        [tool_call("search_tools", {"query": "lookup"}), tool_call("lookup", {"query": "key"}),
         {"choices": [{"message": {"content": "done"}}]}],
        tools=loop_tools(),
    )

    assert result == "done"
    for form_data in sent:
        assert {"ping", "search_tools"} <= tool_names(form_data)
        assert "lookup" not in tool_names(form_data)
        assert system_text(form_data).count("<name>lookup</name>") == 1
    tool_results = [message["content"] for message in sent[2]["messages"] if message["role"] == "tool"]
    assert LOOKUP_DESCRIPTION.strip() in tool_results[0]
    assert tool_results[1] == "found"
    assert "__direct_tool_server_system_prompts__" not in extra_params
    if module_name == "sub_agent":
        assert "agent_ref_exec" in tool_names(sent[0])
        assert "<name>agent_ref_exec</name>" not in system_text(sent[0])


@needs_tool_search
async def test_llm_review_submission_tools_stay_loaded_and_final_request_has_no_manifest(monkeypatch, tool_search):
    capture = {"payload": None}
    tools = loop_tools()
    tools["submit_draft"] = llm_review.make_submission_tool(
        spec=llm_review.build_submit_draft_spec(phase="compose", include_sources=True), capture=capture,
    )
    assert len(json.dumps(tools["submit_draft"]["spec"])) > 400

    _, sent, _ = await run_loop(
        monkeypatch, "llm_review",
        [tool_call("ping"), tool_call("submit_draft", {"draft": "Final draft"})],
        tools=tools, max_iterations=1, submission_tool_names={"submit_draft"},
    )

    assert tool_names(sent[0]) == {"ping", "search_tools", "submit_draft", *["call_tool"] * HAS_CALL_TOOL}
    assert "<name>lookup</name>" in system_text(sent[0])
    assert "<name>submit_draft</name>" not in system_text(sent[0])
    assert tool_names(sent[1]) == {"submit_draft"}
    assert "<available_tools>" not in system_text(sent[1])
    assert capture["payload"]["draft"] == "Final draft"


@needs_tool_search
@pytest.mark.parametrize("module_name", ["magi_decision_support", "multi_model_council"])
async def test_final_request_without_tools_has_no_tool_search_manifest(monkeypatch, tool_search, module_name):
    _, sent, _ = await run_loop(
        monkeypatch, module_name,
        [tool_call("ping"), {"choices": [{"message": {"content": "done"}}]}],
        tools=loop_tools(), max_iterations=1,
        extra_params={"__direct_tool_server_system_prompts__": ["Server prompt."]},
    )

    loop_request, final_request = sent
    assert "<available_tools>" in system_text(loop_request)
    assert not final_request.get("tools")
    assert "<available_tools>" not in system_text(final_request)
    assert system_text(final_request).count("Server prompt.") == 1


@needs_tool_search
async def test_sub_agent_final_retry_without_tools_has_no_tool_search_manifest(monkeypatch, tool_search):
    result, sent, _ = await run_loop(
        monkeypatch, "sub_agent",
        [tool_call("ping"), tool_call("ping"), {"choices": [{"message": {"content": "done"}}]}],
        tools=loop_tools(), max_iterations=1,
        extra_params={"__direct_tool_server_system_prompts__": ["Server prompt."]},
    )

    loop_request, final_request, retry = sent
    assert result == "done"
    assert tool_names(final_request) == tool_names(loop_request)
    assert system_text(final_request) == system_text(loop_request)
    assert not retry.get("tools")
    assert "<available_tools>" not in system_text(retry)
    assert system_text(retry).count("Server prompt.") == 1


@needs_tool_search
@pytest.mark.parametrize("as_parts", [False, True])
async def test_sub_agent_summary_retry_without_tools_has_no_tool_search_manifest(as_parts):
    from open_webui.utils.tool_search import build_deferred_tools_manifest

    system = f"Be brief.\n\n{build_deferred_tools_manifest(loop_tools(), ['lookup'])}"
    body = {
        "model": "model",
        "metadata": {"model_id": "model"},
        "tools": [{"type": "function", "function": loop_tools()["ping"]["spec"]}],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system}] if as_parts else system},
            {"role": "user", "content": "Summarize"},
        ],
    }
    original = copy.deepcopy(body)
    sent = []

    async def complete(*, request, form_data, user, bypass_filter):
        sent.append(form_data)
        return [tool_call("ping"), {"choices": [{"message": {"content": "summary"}}]}][len(sent) - 1]

    summary = await sub_agent._request_loop_summary(
        generate_chat_completion=complete, request=make_request(), user_obj=make_user(), summary_body=body,
    )

    assert summary == "summary"
    assert sent[0] == original
    assert "tools" not in sent[1]
    assert json.dumps(sent[1]["messages"]).count("available_tools") == 0
    assert "Be brief." in json.dumps(sent[1]["messages"])
    assert body == original


@needs_tool_search
async def test_parallel_reuses_the_chat_search_tools(monkeypatch, tool_search):
    from open_webui.utils.tool_search import apply_tool_search

    metadata = {}
    chat_tools = loop_tools()
    assert await apply_tool_search({"messages": []}, metadata, chat_tools) == {"lookup"}
    metadata["tools"] = chat_tools

    result = await parallel_tools.Tools().run_tools_parallel(
        [parallel_tools.ToolCallItem(name="search_tools", args={"query": "lookup"})],
        __request__=make_request(), __user__=make_user().model_dump(), __metadata__=metadata,
    )

    assert json.loads(result)["results"][0]["result"]["tools"][0]["name"] == "lookup"


@needs_call_tool
async def test_parallel_runs_call_tool_with_the_chat_dispatcher(monkeypatch, tool_search):
    from open_webui.utils.tool_search import apply_tool_search

    metadata = {}
    chat_tools = loop_tools()
    await apply_tool_search({"messages": []}, metadata, chat_tools)
    metadata["tools"] = chat_tools

    result = await parallel_tools.Tools().run_tools_parallel(
        [
            parallel_tools.ToolCallItem(
                name="call_tool", args={"tool_name": "lookup", "params": json.dumps({"query": "key"})},
            ),
            parallel_tools.ToolCallItem(name="call_tool", args={"tool_name": "missing", "params": "{}"}),
        ],
        __request__=make_request(), __user__=make_user().model_dump(), __metadata__=metadata,
    )

    found, missing = json.loads(result)["results"]
    assert (found["tool_name"], found["result"]) == ("lookup", "found")
    assert missing["result"].startswith("Error: Tool \"missing\" is not available")


async def test_llm_review_final_keeps_embeds_stored_while_it_is_being_sent():
    from open_webui.models.chats import ChatForm, Chats
    from open_webui.models.users import Users
    from open_webui.socket.main import get_event_emitter

    await Users.insert_new_user("embed-user", "Embed User", "embed@example.com", role="admin")
    message = {"id": "message", "role": "assistant", "content": ""}
    await Chats.insert_new_chat("embed-chat", "embed-user", ChatForm(chat={
        "history": {"messages": {"message": message}, "currentId": "message"},
    }))
    emitter = await get_event_emitter({"user_id": "embed-user", "chat_id": "embed-chat", "message_id": "message"})
    open_gate, at_gate = asyncio.Event(), asyncio.Event()
    open_gate.set()

    async def gated(event):
        if not open_gate.is_set():
            at_gate.set()
            await open_gate.wait()
        await emitter(event)

    first, second = [
        llm_review.EventEmitter(
            raw, enabled=True, message_id="message", chat_id="embed-chat", render_markdown=False,
        )
        for raw in (gated, emitter)
    ]
    for run in (first, second):
        await run.init_rich(topic="Topic", total_steps=1, agents=[])

    open_gate.clear()
    finishing = asyncio.create_task(first.finalize(summary={"topic": "FINAL-A", "num_rounds": 1, "agents": []}))
    await at_gate.wait()
    await second.finalize(summary={"topic": "FINAL-B", "num_rounds": 1, "agents": []})
    await emitter({"type": "embeds", "data": {"embeds": ["<div>sibling tool</div>"]}})
    open_gate.set()
    await finishing

    stored = await Chats.get_message_metadata("embed-chat", "message", "embeds")
    assert "<div>sibling tool</div>" in stored
    assert [sum(final in embed for embed in stored) for final in ("FINAL-A", "FINAL-B")] == [1, 1]


@pytest.fixture
def skill_tools(monkeypatch):
    from open_webui.models.config import Config

    monkeypatch.setattr(Config, "get_many", AsyncMock(return_value={}))
    monkeypatch.setattr(Config, "get", AsyncMock(
        side_effect=lambda key, default=None: {} if key == "user.permissions" else default
    ))
    model = {"id": "model", "info": {"meta": {"builtinTools": {
        category: False for category in ("time", "user_input", "knowledge", "chats", "tasks")
    }}}}

    async def load(module, valves, metadata=None, role="admin"):
        metadata = metadata if metadata is not None else {}
        metadata.update({"chat_id": "local:skill-test", "session_id": "browser", "tool_servers": []})
        request = make_request(model)
        user = make_user(role).model_dump()
        tools, _ = await module.build_tools_dict(
            request=request, model=model, metadata=metadata, user=make_user(role), valves=valves,
            extra_params={
                "__request__": request, "__user__": user, "__metadata__": metadata, "__skill_ids__": ["skill"],
            },
            tool_id_list=[], excluded_tool_ids=set(), resolved_terminal_id=None,
            resolved_direct_tool_servers=[],
        )
        return tools, request, user

    return load


@needs_core_skill_tools
@pytest.mark.parametrize("module_name", AGENT_MODULES)
@pytest.mark.parametrize("read, write, expected", [
    (True, None, SKILL_READ_TOOLS),
    (True, True, SKILL_READ_TOOLS | {"update_skill_files", "create_skill"}),
    (False, True, set()),
])
async def test_skill_write_tools_need_their_own_valve(skill_tools, module_name, read, write, expected):
    module = importlib.import_module(module_name)
    valves = module.Tools.Valves(ENABLE_SKILLS_TOOLS=read)
    if write is not None:
        valves.ENABLE_SKILLS_WRITE_TOOLS = write

    tools, _, _ = await skill_tools(module, valves)

    assert (SKILL_READ_TOOLS | {"update_skill_files", "create_skill"}) & set(tools) == expected


@pytest.mark.parametrize("module_name", AGENT_MODULES)
@pytest.mark.parametrize("loadable", [True, False])
async def test_skill_manifest_is_passed_only_with_a_tool_that_loads_it(monkeypatch, module_name, loadable):
    tools = loop_tools()
    if loadable:
        tools["view_skill"] = {**tools["ping"], "type": "builtin", "spec": {**tools["ping"]["spec"], "name": "view_skill"}}
    manifest = "<available_skills>\n<skill>\n<name>Writer</name>\n</skill>\n</available_skills>"
    inline = '<skill name="Editor">\nEdit well.\n</skill>'

    _, sent, _ = await run_loop(
        monkeypatch, module_name, [{"choices": [{"message": {"content": "done"}}]}], tools=tools,
        messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        skills=[inline, manifest],
    )

    system = sent[0]["messages"][0]["content"]
    assert ("<available_skills>" in system) is loadable
    if loadable:
        assert system == f"Be brief.\n{inline}\n{manifest}"


@needs_core_skill_tools
async def test_agents_sharing_loaded_tools_keep_their_own_skill_versions(monkeypatch, skill_tools):
    from open_webui.models.skills import Skills

    monkeypatch.setattr(Skills, "get_skill_by_id", AsyncMock(return_value=SimpleNamespace(
        id="skill", user_id="loop-user", is_active=True, name="Skill", content="Instructions",
        description=None, meta={}, version_id="version-1",
        data={"files": [{"path": "SKILL.md", "content": "Instructions"}]},
    )))
    parent = {}
    tools, request, user = await skill_tools(sub_agent, sub_agent.Tools.Valves(), parent)
    first, second = sub_agent.isolate_chat_context(parent), sub_agent.isolate_chat_context(parent)

    await sub_agent.execute_tool_call(
        {"id": "call", "function": {"name": "view_skill", "arguments": '{"id": "skill"}'}}, tools,
        {"__request__": request, "__user__": user, "__metadata__": first},
    )

    assert first["chat_context"]["skill_versions"] == {"skill": "version-1"}
    assert "skill_versions" not in second["chat_context"]
    assert "skill_versions" not in (parent.get("chat_context") or {})


@pytest.mark.parametrize("tag, skill_id", [
    ('<skill name="Writer notes">\nWrite well.\n</skill>', "terminal:Writer%20notes"),
    ('<skill id="writer" version_id="version-1" name="Writer">\nWrite well.\n</skill>', "writer"),
])
def test_inline_skill_tags_are_inherited_for_both_core_shapes(tag, skill_id):
    messages = [{"role": "system", "content": f"Be brief.\n\n{tag}"}, {"role": "user", "content": "Hi"}]

    assert sub_agent.extract_user_skill_tags(messages) == [tag]
    assert sub_agent.extract_skill_ids(messages) == [skill_id]


async def call_entry(entry, valves=None, **kwargs):
    module = importlib.import_module("sub_agent" if entry == "parallel_sub_agents" else entry)
    tool = module.Tools()
    if module is sub_agent:
        tool.valves.ENABLE_CONTEXT_COMPACTION = False
        tool.valves.LARGE_TOOL_RESULT_MODE = "raw"
    for name, value in (valves or {}).items():
        setattr(tool.valves, name, value)
    if entry == "sub_agent":
        return await tool.run_sub_agent(description="Task", prompt="Task", **kwargs)
    if entry == "parallel_sub_agents":
        tasks = kwargs.pop("tasks", [{"description": "Task", "prompt": "Task"}])
        return await tool.run_parallel_sub_agents(tasks=tasks, **kwargs)
    if entry == "llm_review":
        tool.valves.DEFAULT_MODELS = "parent"
        kwargs["__user__"]["valves"] = {**kwargs["__user__"]["valves"], "ROUNDS": 1, "RICH_PROGRESS": False}
        return await tool.llm_review(topic="Write the draft", **kwargs)
    decision = {"proposition": "Choose", "option_a": "A", "option_b": "B"}
    if entry == "magi_decision_support":
        return await tool.magi_decide(**decision, **kwargs)
    return await tool.council_decide(**decision, models="parent,child", **kwargs)


ENTRIES = ["sub_agent", "parallel_sub_agents", *AGENT_MODULES[1:]]
LOOKUP_CALL = {"tool_name": "lookup", "params": json.dumps({"query": "key"})}
TOOL_SEARCH_SETTINGS = {
    "chat.tool_search.enable": True, "chat.tool_search.defer_threshold": 400,
    "chat.tool_search.always_loaded": [], "chat.tool_search.defer_builtin_tools": False,
}


async def run_entry(
    monkeypatch, entry, respond, *, regular_tools=dict, settings=None, skill_context=SKILL_MANIFEST, child_params=None,
    inlet=None, request_filter=None, observe=None, parent_metadata=None, valves=None, bound_tools=None,
    extra_models=None, event_call=None, direct=False, user_valves=None, event_emitter=None, request_state=None,
    entry_kwargs=None,
):
    from starlette.requests import Request
    from open_webui.models.config import Config
    from open_webui.models.models import ModelForm, Models
    from open_webui.utils import chat
    from open_webui.utils import filter as core_filter
    from open_webui.utils import tools as core_tools

    settings = {"user.permissions": {}, **(settings or {})}
    monkeypatch.setattr(Config, "get", AsyncMock(side_effect=lambda key, default=None: settings.get(key, default)))
    monkeypatch.setattr(Config, "get_many", AsyncMock(side_effect=lambda *keys: {key: settings.get(key) for key in keys}))
    voice_avatar = {"gestures": [{"name": "wave", "description": "Wave."}]}
    child = {"id": "child", "owned_by": "openai", "info": {"meta": {
        "builtinTools": {category: False for category in ("time", "user_input", "knowledge", "chats", "tasks")},
        "voice_avatar": voice_avatar,
    }, "params": {}}}
    parent = {"id": "parent", "owned_by": "arena", "info": {"meta": {"model_ids": ["child"], "voice_avatar": voice_avatar}}}
    models = {"parent": parent, "child": child, **(extra_models or {})}
    request = Request({
        "type": "http", "app": SimpleNamespace(state=SimpleNamespace(MODELS=models)),
        "state": {
            "metadata": parent_metadata if parent_metadata is not None else {"model_id": "parent"},
            **({"direct": True, "model": models["child"]} if direct else {}),
            **(request_state or {}),
        },
    })
    request._body = b"{}"
    sent, loaded, filtered = [], [], []

    async def get_tools(*, extra_params, **_):
        loaded.append((extra_params["__metadata__"]["model_id"], extra_params["__files__"]))
        return await bound_tools(extra_params) if bound_tools else regular_tools()

    async def provider(*, form_data, request=None, **_):
        sent.append({
            "state_metadata": getattr(getattr(request, "state", None), "metadata", None),
            "temperature": form_data.get("temperature"), "tool_choice": form_data.get("tool_choice"),
            "parallel_tool_calls": form_data.get("parallel_tool_calls"),
            "keys": set(form_data),
            "model": form_data["model"], "model_id": form_data["metadata"].get("model_id"),
            "tools": tool_names(form_data), "raw_tools": copy.deepcopy(form_data.get("tools") or []),
            "metadata": dict(form_data.get("metadata") or {}),
            "system": system_text(form_data),
            "messages": copy.deepcopy(form_data["messages"]),
        })
        return respond(sent[-1])

    async def process_filter_functions(*, form_data, extra_params, filter_type=None, filter_context=None, **_):
        filtered.append(extra_params["__metadata__"]["model_id"])
        if observe:
            observe(filter_type, form_data, extra_params, _["request"])
        if inlet and filter_type == "inlet":
            form_data = inlet(form_data)
        if request_filter and filter_type == "request":
            form_data = request_filter(form_data)
        return form_data, {}

    monkeypatch.setattr(core_tools, "get_tools", get_tools)
    monkeypatch.setattr(core_filter, "get_filter_functions", AsyncMock(return_value=[]), raising=False)
    monkeypatch.setattr(core_filter, "get_sorted_filter_ids", AsyncMock(return_value=[]))
    monkeypatch.setattr(core_filter, "process_filter_functions", process_filter_functions)
    monkeypatch.setattr(chat, "generate_openai_chat_completion", provider)
    for module in (llm_review, multi_model_council):
        monkeypatch.setattr(module, "get_available_models", AsyncMock(return_value=list(models.values())))

    if child_params is not None:
        await Models.insert_new_model(ModelForm(id="child", name="Child", meta={}, params=child_params), "loop-user")
    try:
        await call_entry(entry, valves, **{
            "__request__": request, "__model__": parent,
            "__user__": {**make_user("admin").model_dump(), "valves": user_valves or {}},
            "__event_call__": event_call or AsyncMock(),
            **({"__event_emitter__": event_emitter} if event_emitter else {}),
            "__metadata__": parent_metadata if parent_metadata is not None else {
                "model_id": "parent", "files": None, "tool_ids": ["regular"], "features": {"voice": True},
                "chat_id": "local:entry", "session_id": "browser", "tool_servers": [],
            },
            "__messages__": [
                {"role": "system", "content": f"Be brief.\n\n{skill_context}"}, {"role": "user", "content": "Hi"},
            ],
            **(entry_kwargs or {}),
        })
    finally:
        await Models.delete_model_by_id("child")
    return sent, loaded, filtered


def answer(body, draft_call=None):
    if "submit_review" in body["tools"]:
        return tool_call("submit_review", {"key_feedback": "Review feedback"})
    if "submit_draft" in body["tools"]:
        return tool_call(*(draft_call or ("submit_draft", {"draft": "A complete draft"})))
    return {"choices": [{"message": {"content": json.dumps({"vote": "A", "reasoning": "done"})}}]}


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("skill_context", [
    SKILL_MANIFEST,
    '<skill id="skill" name="Writer">\nWrite well.\n</skill>',
    '<skill name="Writer notes">\nWrite well.\n</skill>',
])
async def test_nested_requests_carry_the_child_model_and_the_parent_skills(monkeypatch, entry, skill_context):
    seen = []

    def inlet(form_data):
        seen.append(system_text(form_data))
        return form_data

    sent, loaded, filtered = await run_entry(monkeypatch, entry, answer, skill_context=skill_context, inlet=inlet)

    assert seen and not any(skill_context in text for text in seen)
    assert loaded and all(item == ("child", []) for item in loaded)
    assert filtered and set(filtered) == {"child"}
    assert sent and {(body["model"], body["model_id"]) for body in sent} == {("child", "child")}
    assert not any("play_animation" in body["tools"] for body in sent)
    agent_requests = [body for body in sent if body["tools"]]
    assert any("submit_review" in body["tools"] for body in agent_requests) == (entry == "llm_review")
    for body in agent_requests:
        assert SKILL_READ_TOOLS <= body["tools"]
        assert skill_context in body["system"]


@needs_core_skill_tools
async def test_reviewers_read_skills_before_submitting_without_being_told_to_stop(monkeypatch):
    from open_webui.models.skills import SkillForm, Skills

    reads = [("view_skill", {"id": "writer"}), ("read_skill_file", {"id": "writer", "path": "SKILL.md"})]

    def read_skill_then_answer(body):
        for name, arguments in reads:
            if "submit_review" in body["tools"] and not called(body, name):
                return tool_call(name, arguments)
        return answer(body)

    await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content="Write well."))
    try:
        sent, _, _ = await run_entry(
            monkeypatch, "llm_review", read_skill_then_answer,
            skill_context=SKILL_MANIFEST.replace("<id>skill</id>", "<id>writer</id>"),
        )
    finally:
        await Skills.delete_skill_by_id("writer")

    reviews = [body for body in sent if "submit_review" in body["tools"]]
    submitted = [body for body in reviews if called(body, "read_skill_file")]
    assert submitted and all(
        sum(message["role"] == "tool" and "Write well." in message["content"] for message in body["messages"]) == 2
        for body in submitted
    )
    assert not any(
        note in str(body["messages"]) for body in reviews for note in ("FINAL opportunity", "Maximum tool iterations")
    )
    assert any("submit_draft" in body["tools"] and "Review feedback" in str(body["messages"]) for body in sent)


def tools_with_a_regular_view_skill():
    tools = loop_tools()
    tools["view_skill"] = {**tools["ping"], "spec": {**tools["ping"]["spec"], "name": "view_skill"}}
    return tools


async def test_a_regular_tool_named_like_a_skill_tool_stays_out_of_reviews_when_skills_are_off(monkeypatch):
    sent, _, _ = await run_entry(
        monkeypatch, "llm_review", answer, regular_tools=tools_with_a_regular_view_skill,
        valves={"ENABLE_SKILLS_TOOLS": False},
    )

    reviews = [body for body in sent if "submit_review" in body["tools"]]
    assert reviews and all(body["tools"] == {"submit_review"} for body in reviews)



@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("source", ["model", "default"])
@pytest.mark.parametrize("skill_context", [SKILL_MANIFEST, ""])
async def test_child_custom_tools_param_is_sent_after_its_loaded_tools(monkeypatch, entry, source, skill_context):
    web_search = {"type": "web_search"}
    params = {"custom_params": {"tools": json.dumps([web_search])}}
    sent, _, _ = await run_entry(
        monkeypatch, entry, answer, skill_context=skill_context,
        child_params=params if source == "model" else None,
        settings={"models.default_params": params} if source == "default" else None,
    )

    review_requests = [body for body in sent if "submit_review" in body["tools"]]
    agent_requests = [body for body in sent if body["raw_tools"] and body not in review_requests]
    assert agent_requests
    for body in agent_requests:
        assert body["raw_tools"][-1] == web_search
    assert bool(review_requests) == (entry == "llm_review")
    assert not any(web_search in body["raw_tools"] for body in review_requests)


@pytest.mark.parametrize("entry", ENTRIES)
async def test_inlet_filters_see_only_the_custom_tools_and_theirs_are_sent_last(monkeypatch, entry):
    web_search = {"type": "web_search"}
    added = {"type": "function", "function": {"name": "filter_tool", "parameters": {"type": "object"}}}
    seen = []

    def inlet(form_data):
        seen.append(copy.deepcopy(form_data.get("tools")))
        form_data["tools"] = [*(form_data.get("tools") or []), added]
        return form_data

    sent, _, _ = await run_entry(
        monkeypatch, entry, answer, inlet=inlet, child_params={"custom_params": {"tools": [web_search]}},
    )

    assert [web_search] in seen
    assert all(tools in (None, [web_search]) for tools in seen)
    assert sent and all(body["raw_tools"][-1] == added for body in sent)
    for body in sent:
        if web_search in body["raw_tools"]:
            assert body["raw_tools"][-2:] == [web_search, added]


def lookup_first(body):
    if not any(message["role"] == "tool" for message in body["messages"]):
        if "lookup" in body["tools"]:
            return tool_call("lookup", {"query": "key"})
        if "call_tool" in body["tools"]:
            return tool_call("call_tool", LOOKUP_CALL)
    return answer(body)


@pytest.mark.parametrize("entry", ENTRIES)
async def test_inlet_runs_once_per_agent_loop_and_every_round_keeps_its_result(monkeypatch, entry):
    issued = []

    def inlet(form_data):
        issued.append(f"Inlet note {len(issued) + 1}.")
        form_data["messages"] = [{"role": "system", "content": issued[-1]}, *form_data["messages"]]
        return form_data

    sent, _, _ = await run_entry(monkeypatch, entry, lookup_first, regular_tools=loop_tools, inlet=inlet)

    notes = [[note for note in issued if note in body["system"]] for body in sent]
    assert all(len(found) == 1 for found in notes)
    rounds = collections.Counter(found[0] for found in notes)
    assert set(rounds) == set(issued)
    assert all(body["system"].count(found[0]) == 1 for body, found in zip(sent, notes))
    assert max(rounds.values()) >= 2
    assert any(message["role"] == "tool" for body in sent for message in body["messages"])


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("replace", [False, True], ids=["in-place", "replaced"])
async def test_inlet_metadata_kept_like_core_for_every_request_of_the_loop(monkeypatch, entry, replace):
    def inlet(form_data):
        if replace:
            form_data["metadata"] = {**form_data["metadata"], "inlet_flag": True}
        else:
            form_data["metadata"]["inlet_flag"] = True
        return form_data

    sent, _, _ = await run_entry(monkeypatch, entry, lookup_first, regular_tools=loop_tools, inlet=inlet)

    assert any(message["role"] == "tool" for body in sent for message in body["messages"])
    assert sent and all(body["metadata"].get("inlet_flag") is (None if replace else True) for body in sent)


def lookup_twice(body):
    rounds = sum(message["role"] == "tool" for message in body["messages"])
    if rounds < 2 and "lookup" in body["tools"]:
        return {"choices": [{"message": {"content": "", "tool_calls": [{
            "id": f"lookup-{rounds}", "type": "function",
            "function": {"name": "lookup", "arguments": json.dumps({"query": "key"})},
        }]}}]}
    return answer(body)


@pytest.mark.parametrize("entry", ENTRIES)
async def test_request_filters_see_the_iteration_note(monkeypatch, entry):
    def request_filter(form_data):
        systems = [message for message in form_data["messages"] if message["role"] == "system"]
        if systems:
            form_data["messages"] = [
                {"role": "system", "content": "\n\n".join(str(message["content"]) for message in systems)},
                *(message for message in form_data["messages"] if message["role"] != "system"),
            ]
        return form_data

    note_role = {"ITERATION_NOTE_ROLE": "system"} if entry in ("sub_agent", "parallel_sub_agents", "llm_review") else None
    sent, _, _ = await run_entry(
        monkeypatch, entry, lookup_first, regular_tools=loop_tools, request_filter=request_filter, valves=note_role,
    )

    assert any("[Iteration" in body["system"] for body in sent)
    assert all(message["role"] != "system" for body in sent for message in body["messages"][1:])


@needs_tool_search
@pytest.mark.parametrize("entry", ["llm_review", "magi_decision_support", "multi_model_council"])
async def test_final_requests_drop_the_manifest_a_request_filter_moves_to_developer(monkeypatch, entry):
    def request_filter(form_data):
        form_data["messages"] = [
            {**message, "role": "developer"} if message["role"] == "system" else message
            for message in form_data["messages"]
        ]
        return form_data

    def respond(body):
        return tool_call("ping") if "ping" in body["tools"] else answer(body)

    sent, _, _ = await run_entry(
        monkeypatch, entry, respond, regular_tools=loop_tools, settings=TOOL_SEARCH_SETTINGS,
        request_filter=request_filter, valves={"MAX_ITERATIONS": 1},
    )

    def developer_text(body):
        return json.dumps([message["content"] for message in body["messages"] if message["role"] == "developer"])

    searched = [body for body in sent if "search_tools" in body["tools"]]
    final = [
        body for body in sent
        if body["tools"] in (set(), {"submit_draft"}) and any(message["role"] == "tool" for message in body["messages"])
    ]
    assert searched and final
    assert all("<available_tools>" in developer_text(body) for body in searched)
    assert all("<available_tools>" not in developer_text(body) for body in final)


CORE_FILTER_PARAMS = {
    "__event_emitter__", "__event_call__", "__user__", "__metadata__", "__oauth_token__",
    "__request__", "__model__", "__chat_id__", "__message_id__",
}


@pytest.mark.parametrize("entry", ENTRIES)
async def test_body_metadata_is_the_loops_metadata_like_core(monkeypatch, entry):
    parent_metadata = {
        "model_id": "parent", "files": [{"id": "file", "type": "file"}], "tool_ids": ["regular"],
        "features": {"voice": True}, "chat_id": "local:entry", "session_id": "browser", "tool_servers": [],
    }
    parent_before = copy.deepcopy(parent_metadata)
    calls = []

    def observe(filter_type, form_data, extra_params, request):
        calls.append((filter_type, form_data["metadata"], extra_params["__metadata__"], dict(extra_params), request))
        extra_params["__metadata__"]["written"] = True
        request.state.metadata["through_state"] = True
        form_data["metadata"]["files"][0]["context"] = "full"
        form_data["metadata"]["features"]["touched"] = True

    sent, _, _ = await run_entry(
        monkeypatch, entry, lookup_twice, regular_tools=loop_tools, observe=observe, parent_metadata=parent_metadata,
    )

    assert {filter_type for filter_type, *_ in calls} == {"inlet", "request"}
    for filter_type, body, injected, params, request in calls:
        assert body is injected
        assert params["__request__"] is request and request.state.metadata is body
        assert set(params) <= CORE_FILTER_PARAMS
    assert all(body["chat_id"] == "local:entry" and body["written"] and body["through_state"] for _, body, *_ in calls)
    loop_metadata = [body for filter_type, body, *_ in calls if filter_type == "inlet"]
    assert len({id(body) for body in loop_metadata}) == len(loop_metadata)
    assert all(any(body["state_metadata"] is metadata for metadata in loop_metadata) for body in sent)
    assert parent_metadata == parent_before


@pytest.mark.parametrize("entry", ["sub_agent", "parallel_sub_agents"])
async def test_sub_agent_retry_without_tools_keeps_the_child_metadata(monkeypatch, entry):
    parent_metadata = {
        "model_id": "parent", "files": None, "tool_ids": ["regular"], "features": {"voice": True},
        "chat_id": "local:entry", "session_id": "browser", "tool_servers": [],
        "task": "parent_task", "sub_agent_iteration": 77,
    }

    def keep_calling(body):
        if body["tools"]:
            return {"choices": [{"message": {"content": "", "tool_calls": [{
                "id": f"lookup-{len(body['messages'])}", "type": "function",
                "function": {"name": "lookup", "arguments": json.dumps({"query": "key"})},
            }]}}]}
        return answer(body)

    sent, _, _ = await run_entry(
        monkeypatch, entry, keep_calling, regular_tools=loop_tools, parent_metadata=parent_metadata,
        valves={"MAX_ITERATIONS": 1},
    )

    assert len(sent) == 3 and not sent[-1]["tools"]
    assert [(body["metadata"]["task"], body["metadata"]["sub_agent_iteration"]) for body in sent] == [
        ("sub_agent", 1), ("sub_agent", 2), ("sub_agent", 2),
    ]
    assert parent_metadata["task"] == "parent_task" and parent_metadata["sub_agent_iteration"] == 77


@pytest.mark.parametrize("entry", ENTRIES)
async def test_loaded_tools_run_with_their_request_and_the_calling_loops_metadata(monkeypatch, entry):
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    parent_metadata = {
        "model_id": "parent", "files": None, "tool_ids": ["regular"], "features": {"voice": True},
        "chat_id": "local:entry", "session_id": "browser", "tool_servers": [],
    }
    parent_before = copy.copy(parent_metadata)
    loaded_requests, seen = [], []

    async def bound_tools(extra_params):
        async def lookup(query: str = "", __request__=None, __metadata__=None, __user__=None):
            seen.append((__request__, __metadata__, __user__))
            __metadata__["tool_wrote"] = True
            return "found"

        loaded_requests.append(extra_params["__request__"])
        tools = loop_tools()
        tools["lookup"]["callable"] = await get_async_tool_function_and_apply_extra_params(
            lookup, {**extra_params, "__user__": {"valves": "lookup-valves"}},
        )
        return tools

    def inlet(form_data):
        form_data["metadata"]["from_inlet"] = True
        return form_data

    await run_entry(
        monkeypatch, entry, lookup_first, bound_tools=bound_tools, inlet=inlet, parent_metadata=parent_metadata,
    )

    assert seen
    for request, metadata, user in seen:
        assert any(request.state.metadata is loaded.state.metadata for loaded in loaded_requests)
        assert "task" not in request.state.metadata
        assert metadata["from_inlet"] and metadata["model_id"] == "child"
        assert user == {"valves": "lookup-valves"}
    assert len({id(metadata) for _, metadata, _ in seen}) == len(seen)
    assert parent_metadata == parent_before


@pytest.mark.parametrize("entry", ENTRIES)
async def test_filters_cannot_change_the_parents_filter_ids_or_tools(monkeypatch, entry):
    parent_tools = {"parent_tool": {"spec": {"name": "parent_tool"}}}
    parent_metadata = {
        "model_id": "parent", "files": None, "tool_ids": ["regular"], "features": {}, "filter_ids": ["parent"],
        "chat_id": "local:entry", "session_id": "browser", "tool_servers": [], "tools": parent_tools,
    }

    def inlet(form_data):
        form_data["metadata"]["filter_ids"].append("child")
        form_data["metadata"]["tools"].clear()
        return form_data

    await run_entry(monkeypatch, entry, answer, inlet=inlet, parent_metadata=parent_metadata)

    assert parent_metadata["filter_ids"] == ["parent"]
    assert parent_metadata["tools"] is parent_tools and list(parent_tools) == ["parent_tool"]


@pytest.mark.parametrize("entry", ENTRIES)
async def test_request_filter_model_change_reaches_the_request_metadata(monkeypatch, entry):
    def request_filter(form_data):
        later = any(message["role"] == "tool" for message in form_data["messages"])
        form_data["model"] = "later" if later else "first"
        return form_data

    sent, _, _ = await run_entry(
        monkeypatch, entry, lookup_first, regular_tools=loop_tools, request_filter=request_filter,
        extra_models={name: {"id": name, "owned_by": "openai", "info": {"meta": {}}} for name in ("first", "later")},
    )

    switched = [body for body in sent if body["model"] in ("first", "later")]
    assert {body["model"] for body in switched} == {"first", "later"}
    assert all(body["model_id"] == body["model"] for body in switched)


COMPACTION_VALVES = {
    "ENABLE_CONTEXT_COMPACTION": True, "CONTEXT_COMPACTION_TOKEN_THRESHOLD": 1000, "LARGE_TOOL_RESULT_MODE": "raw",
}
SUB_AGENT_ENTRIES = ["sub_agent", "parallel_sub_agents"]


def big_lookup_tools():
    async def lookup(query: str = ""):
        return "substantial result line with detail\n" * 600

    tools = loop_tools()
    tools["lookup"]["callable"] = lookup
    return tools


def lookup_rounds_then_answer(rounds):
    agent_calls = []

    def respond(body):
        if body["metadata"].get("task") == "sub_agent_summary":
            return {"choices": [{"message": {"content": "SUMMARY"}}]}
        agent_calls.append(body)
        if len(agent_calls) <= rounds:
            return {"choices": [{"message": {"content": "", "tool_calls": [{
                "id": f"round-{len(agent_calls)}", "type": "function",
                "function": {"name": "lookup", "arguments": json.dumps({"query": "key"})},
            }]}}]}
        return answer(body)

    return respond


@pytest.mark.parametrize("entry", SUB_AGENT_ENTRIES)
async def test_sub_agent_compaction_keeps_the_messages_its_inlet_left_before_the_loop(monkeypatch, entry):
    def inlet(form_data):
        messages = form_data["messages"]
        task = max(index for index, message in enumerate(messages) if message["role"] == "user")
        form_data["messages"] = [
            *messages[:task],
            {"role": "user", "content": "EXAMPLE-QUESTION"},
            {"role": "assistant", "content": "EXAMPLE-ANSWER"},
            messages[task],
            {"role": "system", "content": "INJECTED-AFTER-TASK"},
            *messages[task + 1:],
        ]
        return form_data

    sent, _, _ = await run_entry(
        monkeypatch, entry, lookup_rounds_then_answer(4), regular_tools=big_lookup_tools, inlet=inlet,
        valves=COMPACTION_VALVES,
    )

    assert any(body["metadata"].get("task") == "sub_agent_summary" for body in sent)
    compacted = 0
    for body in sent:
        if body["metadata"].get("task") == "sub_agent_summary":
            continue
        texts = [str(message.get("content")) for message in body["messages"]]
        for marker in ("EXAMPLE-QUESTION", "EXAMPLE-ANSWER", "INJECTED-AFTER-TASK"):
            assert sum(marker in text for text in texts) == 1
        task = texts.index("EXAMPLE-ANSWER") + 1
        if "<agent_loop_compaction_context>" in "".join(texts):
            compacted += 1
            assert "<agent_loop_compaction_context>" in texts[task]
    assert compacted


@pytest.mark.parametrize("entry", SUB_AGENT_ENTRIES)
async def test_sub_agent_summary_request_names_the_summary_model(monkeypatch, entry):
    sent, _, _ = await run_entry(
        monkeypatch, entry, lookup_rounds_then_answer(4), regular_tools=big_lookup_tools,
        valves={**COMPACTION_VALVES, "COMPACTION_SUMMARY_MODEL": "summarizer"},
        extra_models={"summarizer": {"id": "summarizer", "owned_by": "openai", "info": {"meta": {}}}},
    )

    summaries = [body for body in sent if body["metadata"].get("task") == "sub_agent_summary"]
    assert summaries
    assert all(body["model"] == "summarizer" and body["model_id"] == "summarizer" for body in summaries)


@pytest.mark.parametrize("entry", SUB_AGENT_ENTRIES)
async def test_sub_agent_compaction_keeps_an_agents_md_its_request_filter_edited(monkeypatch, entry):
    from open_webui.utils import terminals
    from open_webui.utils import tools as core_tools

    if not hasattr(terminals, "add_terminal_agents_md"):
        pytest.skip("Core before 0.12.0 has no terminal AGENTS.md")
    agents_md = "# AGENTS.md\n\nFollow the project conventions."
    if hasattr(terminals, "get_terminal_json"):
        monkeypatch.setattr(terminals, "get_terminal_json", AsyncMock(return_value={
            "home": "/home/test", "content": "Follow the project conventions.",
        }))
    else:
        monkeypatch.setattr(terminals, "get_terminal_agents_md", AsyncMock(return_value=agents_md))
    monkeypatch.setattr(core_tools, "get_terminal_tools", AsyncMock(return_value=({}, "")))

    def request_filter(form_data):
        for message in form_data["messages"]:
            if message["role"] == "user" and message["content"] == agents_md:
                message["content"] += "\nEdited by a request filter."
        return form_data

    sent, _, _ = await run_entry(
        monkeypatch, entry, lookup_rounds_then_answer(4), regular_tools=big_lookup_tools, skill_context="",
        request_filter=request_filter, valves=COMPACTION_VALVES,
        settings={"terminal_server.connections": [{"id": "terminal"}]},
        parent_metadata={
            "model_id": "parent", "tool_ids": ["regular"], "terminal_id": "terminal",
            "chat_id": "local:entry", "tool_servers": [],
        },
    )

    assert any(body["metadata"].get("task") == "sub_agent_summary" for body in sent)
    compacted = 0
    for body in sent:
        if body["metadata"].get("task") == "sub_agent_summary":
            continue
        users = [str(message["content"]) for message in body["messages"] if message["role"] == "user"]
        assert [text for text in users if text.startswith("# AGENTS.md")] == [
            f"{agents_md}\nEdited by a request filter."
        ]
        task = [text for text in users if text.startswith("Task")]
        assert len(task) == 1
        if "<agent_loop_compaction_context>" in "".join(users):
            compacted += 1
            assert "<agent_loop_compaction_context>" in task[0]
    assert compacted


@needs_tool_search
@pytest.mark.parametrize("entry", ["llm_review", "magi_decision_support", "multi_model_council"])
async def test_final_requests_drop_the_tool_search_manifest_from_system_parts(monkeypatch, entry):
    def inlet(form_data):
        for message in form_data["messages"]:
            if message["role"] == "system" and isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        return form_data

    def respond(body):
        return tool_call("ping") if "ping" in body["tools"] else answer(body)

    sent, _, _ = await run_entry(
        monkeypatch, entry, respond, regular_tools=loop_tools, settings=TOOL_SEARCH_SETTINGS, inlet=inlet,
        valves={"MAX_ITERATIONS": 1},
    )

    searched = [body for body in sent if "search_tools" in body["tools"]]
    final = [body for body in sent if body["tools"] in (set(), {"submit_draft"})]
    assert searched and final
    assert all("<available_tools>" in body["system"] for body in searched)
    assert any(
        isinstance(message["content"], list)
        for body in final for message in body["messages"] if message["role"] == "system"
    )
    assert all("<available_tools>" not in body["system"] for body in final)


@pytest.mark.parametrize("entry", ["llm_review", "magi_decision_support", "multi_model_council"])
async def test_final_requests_keep_the_inlets_parallel_tool_calls_only_with_tools(monkeypatch, entry):
    def inlet(form_data):
        form_data["parallel_tool_calls"] = False
        return form_data

    def respond(body):
        return tool_call("ping") if "ping" in body["tools"] else answer(body)

    sent, _, _ = await run_entry(
        monkeypatch, entry, respond, regular_tools=loop_tools, inlet=inlet, valves={"MAX_ITERATIONS": 1},
    )

    final = [
        body for body in sent
        if "ping" not in body["tools"] and any(message["role"] == "tool" for message in body["messages"])
    ]
    assert final
    for body in final:
        assert body["parallel_tool_calls"] is (False if body["tools"] else None)


@pytest.mark.parametrize("entry", ["sub_agent", "parallel_sub_agents"])
async def test_sub_agent_retry_without_tools_runs_the_request_filters(monkeypatch, entry):
    filter_tool = {"type": "function", "function": {"name": "filter_tool", "parameters": {"type": "object"}}}

    def request_filter(form_data):
        form_data["temperature"] = float(len(form_data.get("tools") or []))
        form_data["tools"] = [*(form_data.get("tools") or []), filter_tool]
        return form_data

    def respond(body):
        return tool_call("ping") if body["tools"] else {"choices": [{"message": {"content": "done"}}]}

    sent, _, _ = await run_entry(
        monkeypatch, entry, respond, regular_tools=loop_tools, request_filter=request_filter,
        valves={"MAX_ITERATIONS": 1},
    )

    retries = [body for body in sent if not body["tools"]]
    assert retries and all(body["temperature"] for body in sent if body["tools"])
    assert all(body["temperature"] == 0.0 for body in retries)


@pytest.mark.parametrize("entry", ["llm_review", "magi_decision_support", "multi_model_council"])
async def test_final_request_drops_a_tool_choice_for_a_tool_it_no_longer_sends(monkeypatch, entry):
    def request_filter(form_data):
        if not any(message["role"] == "tool" for message in form_data["messages"]):
            form_data["tool_choice"] = {"type": "function", "function": {"name": "ping"}}
        return form_data

    def respond(body):
        return tool_call("ping") if "ping" in body["tools"] else answer(body)

    sent, _, _ = await run_entry(
        monkeypatch, entry, respond, regular_tools=loop_tools, request_filter=request_filter,
        valves={"MAX_ITERATIONS": 1},
    )

    final = [
        body for body in sent
        if "ping" not in body["tools"] and any(message["role"] == "tool" for message in body["messages"])
    ]
    assert any(body["tool_choice"] for body in sent if "ping" in body["tools"])
    assert final
    for body in final:
        choice = body["tool_choice"]
        assert choice is None or choice["function"]["name"] in body["tools"]


@pytest.mark.parametrize("entry", ENTRIES)
async def test_loaded_tools_get_the_files_the_loops_inlet_left(monkeypatch, entry):
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    parent_metadata = {
        "model_id": "parent", "files": [{"id": "file", "type": "file"}], "tool_ids": ["regular"],
        "features": {}, "chat_id": "local:entry", "session_id": "browser", "tool_servers": [],
    }
    parent_before = copy.deepcopy(parent_metadata)
    seen = []

    async def bound_tools(extra_params):
        async def lookup(query: str = "", __files__=None, __metadata__=None):
            seen.append((__files__, __metadata__))
            return "found"

        tools = loop_tools()
        tools["lookup"]["callable"] = await get_async_tool_function_and_apply_extra_params(lookup, extra_params)
        return tools

    def inlet(form_data):
        for file in form_data["metadata"]["files"]:
            file["context"] = "full"
        return form_data

    sent, _, _ = await run_entry(
        monkeypatch, entry, lookup_first, bound_tools=bound_tools, inlet=inlet, parent_metadata=parent_metadata,
    )

    assert seen
    for files, metadata in seen:
        assert files is metadata["files"]
        assert files == [{"id": "file", "type": "file", "context": "full"}]
    assert parent_metadata == parent_before


SEARCH_TOOLS_FUNCTION = {"type": "function", "function": {"name": "search_tools", "parameters": {"type": "object"}}}


def add_search_tools(form_data):
    form_data["tools"] = [*(form_data.get("tools") or []), SEARCH_TOOLS_FUNCTION]
    return form_data


def drop_search_tools(form_data):
    form_data["tools"] = [tool for tool in form_data.get("tools") or [] if tool != SEARCH_TOOLS_FUNCTION]
    return form_data


@needs_tool_search
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("child_params, inlet, tool_search_on", [
    ({"custom_params": {"tools": [SEARCH_TOOLS_FUNCTION]}}, None, False),
    (None, add_search_tools, False),
    ({"custom_params": {"tools": [SEARCH_TOOLS_FUNCTION]}}, drop_search_tools, True),
], ids=["custom", "inlet", "inlet-drops-custom"])
async def test_search_tools_left_by_inlet_filters_turns_tool_search_off(
    monkeypatch, entry, child_params, inlet, tool_search_on,
):
    sent, _, _ = await run_entry(
        monkeypatch, entry, lookup_first, regular_tools=loop_tools, settings=TOOL_SEARCH_SETTINGS,
        child_params=child_params, inlet=inlet,
    )

    agent_requests = [body for body in sent if "submit_review" not in body["tools"] and body["tools"] - {"search_tools"}]
    assert agent_requests
    for body in agent_requests:
        searched = [tool for tool in body["raw_tools"] if tool.get("function", {}).get("name") == "search_tools"]
        assert ("lookup" not in body["tools"]) == tool_search_on
        assert (searched == [SEARCH_TOOLS_FUNCTION]) != tool_search_on
        assert len(searched) == 1
    if HAS_CALL_TOOL or not tool_search_on:
        assert any(message["role"] == "tool" for body in agent_requests for message in body["messages"])

@needs_call_tool
@pytest.mark.parametrize("entry", ENTRIES)
async def test_call_tool_runs_its_target_and_history_keeps_the_model_call(monkeypatch, entry):
    def respond(body):
        if "call_tool" in body["tools"] and not any(message.get("tool_calls") for message in body["messages"]):
            return tool_call("call_tool", LOOKUP_CALL)
        draft = {"tool_name": "submit_draft", "params": json.dumps({"draft": "A complete draft"})}
        return answer(body, ("call_tool", draft))

    sent, _, _ = await run_entry(monkeypatch, entry, respond, regular_tools=loop_tools, settings=TOOL_SEARCH_SETTINGS)

    followups = [body for body in sent if any(message.get("tool_calls") for message in body["messages"])]
    assert followups
    for body in followups:
        calls = [call for message in body["messages"] for call in message.get("tool_calls") or []]
        assert [call["function"] for call in calls] == [{"name": "call_tool", "arguments": json.dumps(LOOKUP_CALL)}]
        assert not any("provider_function" in call for call in calls)
        assert [message["content"] for message in body["messages"] if message["role"] == "tool"] == ["found"]


def called(body, name):
    return any(
        call["function"]["name"] == name
        for message in body["messages"] if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    )


def calling(*steps):
    """Call each (name, arguments) tool once the loop offers it, then answer."""
    def respond(body):
        for name, arguments in steps:
            if name in body["tools"] and not called(body, name):
                return tool_call(name, arguments)
        return answer(body)

    return respond


PAGE_PARAMETERS = {"type": "object", "anyOf": [
    {"properties": {"id": {"type": "string"}}, "required": ["id"]},
    {"properties": {"path": {"type": "string"}}, "required": ["path"]},
]}


@pytest.fixture
def page_tools(monkeypatch):
    """The same anyOf page tool from an MCP server and as a regular tool."""
    from open_webui.utils import tools as core_tools

    received = []

    async def call_tool(name, function_args):
        received.append(function_args)
        return [{"type": "text", "text": "page"}]

    async def fetch_page(**kwargs):
        received.append(kwargs)
        return "page"

    spec = {"name": "fetch_page", "description": "Fetch a page by id or path.", "parameters": PAGE_PARAMETERS}
    client = SimpleNamespace(call_tool=call_tool, disconnect=AsyncMock(), instructions=None)
    monkeypatch.setattr(core_tools, "connect_mcp_server", AsyncMock(return_value=(client, [spec])), raising=False)
    regular = {"fetch_page": {"type": "tool", "tool_id": "regular", "callable": fetch_page, "spec": spec}}
    kinds = {
        "mcp": ("server:mcp:docs", "docs_fetch_page", {"path": "guide"}),
        "regular": ("regular", "fetch_page", {}),
    }
    return SimpleNamespace(received=received, regular=regular, kinds=kinds)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("kind", ["mcp", "regular"])
async def test_only_mcp_tools_get_arguments_their_schema_does_not_list(monkeypatch, page_tools, entry, kind):
    tool_id, name, expected = page_tools.kinds[kind]
    metadata = {
        "model_id": "parent", "files": None, "tool_ids": [tool_id], "chat_id": "local:entry",
        "session_id": "browser", "tool_servers": [],
    }

    await run_entry(
        monkeypatch, entry, calling((name, {"path": "guide"})), regular_tools=lambda: page_tools.regular,
        parent_metadata=metadata, skill_context="",
    )

    assert page_tools.received and all(arguments == expected for arguments in page_tools.received)


@pytest.mark.parametrize("kind", ["mcp", "regular"])
async def test_parallel_passes_only_mcp_tools_arguments_their_schema_does_not_list(monkeypatch, page_tools, kind):
    from open_webui.utils import middleware
    from open_webui.utils import tools as core_tools

    tool_id, name, expected = page_tools.kinds[kind]
    load_tools = middleware.get_tools

    async def get_tools(request, tool_ids, user, extra_params):
        tools = await load_tools(request, [tool_id for tool_id in tool_ids if tool_id != "regular"], user, extra_params)
        return {**tools, **(page_tools.regular if "regular" in tool_ids else {})}

    monkeypatch.setattr(middleware, "get_tools", get_tools)
    monkeypatch.setattr(middleware, "connect_mcp_server", core_tools.connect_mcp_server)
    model = {"id": "model", "owned_by": "openai", "info": {"meta": {}}}
    metadata = await core_chat_tools(
        monkeypatch, parallel_request(model), make_user("admin"), model, tool_ids=[tool_id],
    )

    await call_parallel(metadata, (name, {"path": "guide"}))

    assert page_tools.received == [expected]


@pytest.fixture
def reconnecting_browser(monkeypatch):
    """Real Core event calls to a browser that reconnects while a confirmation is open."""
    import time
    from open_webui.socket import main as socket

    if not hasattr(socket, "get_user_chat_session_id"):
        pytest.skip("Core before 0.12.4 does not resume input requests on a reconnected session")
    sessions = {"old": {"id": "loop-user"}}
    browser = SimpleNamespace(events=[], replies={}, metadata={
        "model_id": "parent", "files": None, "tool_ids": ["regular"], "tool_servers": [],
        "chat_id": "saved-chat", "message_id": "message", "session_id": "old", "user_id": "loop-user",
    })

    async def call(event_name, payload, *, to, timeout):
        event = payload["data"]
        data = event.get("data") or {}
        browser.events.append((event["type"], to, data))
        if event["type"] == "confirmation":
            if to == "old":
                sessions.clear()
                sessions["new"] = {"id": "loop-user", "chat_ids": ["saved-chat"], "last_seen_at": time.time()}
                await asyncio.sleep(10)
            return True
        if data.get("session_id") != to:
            # The browser never answers a call for another session; failing fast keeps the test from hanging.
            return {"error": f"{to} ignored a call for {data.get('session_id')}"}
        return browser.replies[event["type"]](data)

    monkeypatch.setattr(socket, "SESSION_POOL", sessions)
    monkeypatch.setattr(socket.sio, "call", call)
    monkeypatch.setattr(socket.sio, "emit", AsyncMock())
    browser.sent = lambda kind: [(to, data) for sent_kind, to, data in browser.events if sent_kind == kind]
    return browser


async def confirm_gate(__event_call__=None) -> str:
    """Ask the user whether to continue."""
    return str(await __event_call__(event_data={"type": "confirmation", "data": {"title": "Continue?", "message": "Continue?"}}))


def gated_tools(**targets):
    """Load a confirmation gate and *targets* (name -> (function, parameters)) as Core binds tools."""
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    async def bound(extra_params):
        tools = {}
        for name, (function, properties) in {"confirm_gate": (confirm_gate, {}), **targets}.items():
            tools[name] = {
                "type": "tool", "tool_id": "regular",
                "spec": {"name": name, "description": name, "parameters": {"type": "object", "properties": properties}},
                **({"direct": True, "server": {"url": "http://browser"}} if function is None else {
                    "callable": await get_async_tool_function_and_apply_extra_params(function, extra_params),
                }),
            }
        return tools

    return bound


def assert_followed_the_resumed_session(browser, kind):
    sends = browser.sent(kind)
    assert sends and all(data["session_id"] == to for to, data in sends), sends
    assert "new" in {to for to, _ in sends}
    confirmations = browser.sent("confirmation")
    # Core resumes only confirmations whose data has no session_id.
    assert "new" in {to for to, _ in confirmations}
    assert not any("session_id" in data for _, data in confirmations)
    assert browser.metadata["session_id"] == "new"


@pytest.mark.parametrize("entry", ENTRIES)
async def test_direct_tools_follow_the_session_core_resumes_on(monkeypatch, reconnecting_browser, entry):
    from open_webui.socket.main import get_event_call

    browser = reconnecting_browser
    browser.replies["execute:tool"] = lambda data: "ran"

    await run_entry(
        monkeypatch, entry, calling(("confirm_gate", {}), ("browser_lookup", {})),
        bound_tools=gated_tools(browser_lookup=(None, {})), parent_metadata=browser.metadata,
        event_call=await get_event_call(browser.metadata), skill_context="",
    )

    assert_followed_the_resumed_session(browser, "execute:tool")


async def test_core_code_execution_follows_the_session_core_resumes_on(monkeypatch, reconnecting_browser):
    from open_webui.socket.main import get_event_call

    browser = reconnecting_browser
    browser.replies["execute:python"] = lambda data: {"stdout": "2\n", "stderr": "", "result": ""}

    await run_entry(
        monkeypatch, "sub_agent", calling(("confirm_gate", {}), ("run_python", {"code": "print(1 + 1)"})),
        bound_tools=gated_tools(run_python=(builtin.execute_code, {"code": {"type": "string"}})),
        parent_metadata=browser.metadata, event_call=await get_event_call(browser.metadata), skill_context="",
    )

    assert_followed_the_resumed_session(browser, "execute:python")


async def test_core_terminal_requests_follow_the_session_core_resumes_on(monkeypatch, reconnecting_browser):
    from open_webui.socket.main import get_event_call

    browser = reconnecting_browser
    browser.metadata["terminal_id"] = "user-terminal"
    browser.replies["request:terminal"] = lambda data: {"data": (
        {"name": "Notes", "content": "Terminal notes."} if data["path"].startswith("/skills/read") else None
    )}

    await run_entry(
        monkeypatch, "sub_agent", calling(("confirm_gate", {}), ("read_terminal_skill", {"id": "terminal:Notes"})),
        bound_tools=gated_tools(read_terminal_skill=(builtin.view_skill, {"id": {"type": "string"}})),
        parent_metadata=browser.metadata, event_call=await get_event_call(browser.metadata), skill_context="",
    )

    assert_followed_the_resumed_session(browser, "request:terminal")
    assert any(data["path"].startswith("/skills/read") and to == "new" for to, data in browser.sent("request:terminal"))


@pytest.mark.parametrize("entry", ENTRIES)
async def test_tools_after_a_resumed_request_in_the_same_turn_see_the_session_core_holds(
    monkeypatch, reconnecting_browser, entry
):
    from open_webui.socket.main import get_event_call

    browser = reconnecting_browser
    seen = []

    async def record_session(__metadata__=None) -> str:
        """Report the browser session this tool runs for."""
        seen.append((browser.metadata["session_id"], __metadata__["session_id"]))
        return "recorded"

    def gate_then_record(body):
        if "record_session" in body["tools"] and not called(body, "record_session"):
            calls = [tool_call(name)["choices"][0]["message"]["tool_calls"][0] for name in ("confirm_gate", "record_session")]
            return {"choices": [{"message": {"content": "", "tool_calls": calls}}]}
        return answer(body)

    await run_entry(
        monkeypatch, entry, gate_then_record, bound_tools=gated_tools(record_session=(record_session, {})),
        parent_metadata=browser.metadata, event_call=await get_event_call(browser.metadata), skill_context="",
    )

    assert ("new", "new") in seen and all(core == tool for core, tool in seen), seen


@pytest.mark.parametrize("entry, nested", [
    *((entry, False) for entry in ENTRIES), ("magi_decision_support", True),
], ids=[*ENTRIES, "magi_decision_support called by another agent"])
async def test_direct_model_requests_follow_the_session_core_resumes_on(
    monkeypatch, reconnecting_browser, entry, nested
):
    from open_webui.socket.main import get_event_call

    browser = reconnecting_browser
    respond = calling(("confirm_gate", {}))
    browser.replies["request:chat:completion"] = lambda data: respond({
        "tools": tool_names(data["form_data"]), "messages": data["form_data"]["messages"],
    })

    from open_webui.utils import chat

    async def record_model_caller(info):
        browser.events.append(("model caller", info["session_id"], {}))
        return await get_event_call(info)

    monkeypatch.setattr(chat, "get_event_call", record_model_caller)

    event_call = await get_event_call(browser.metadata)
    await run_entry(
        monkeypatch, entry, answer, bound_tools=gated_tools(), skill_context="", direct=True,
        parent_metadata=copy.deepcopy(browser.metadata) if nested else browser.metadata,
        event_call=sub_agent.follow_core_session(event_call, browser.metadata) if nested else event_call,
    )

    assert_followed_the_resumed_session(browser, "request:chat:completion")
    # Core rejects a caller for a gone session before sending, so check every caller it built.
    resumed = next(index for index, (kind, to, _) in enumerate(browser.events) if (kind, to) == ("confirmation", "new"))
    later_callers = {to for kind, to, _ in browser.events[resumed:] if kind == "model caller"}
    assert later_callers == {"new"}


needs_core_skill_overview = pytest.mark.skipif(
    importlib.util.find_spec("open_webui.utils.skill_files") is None
    or not hasattr(importlib.import_module("open_webui.utils.skill_files"), "skill_overview"),
    reason="Core before 0.12.1 has no skill_overview to inline skills with",
)
NO_SKILL_TOOLS_CHILD = {"id": "child", "owned_by": "openai", "info": {"meta": {"builtinTools": {
    category: False for category in ("time", "user_input", "knowledge", "chats", "tasks", "skills")
}}, "params": {}}}


@needs_core_skill_overview
@pytest.mark.parametrize("child_skill_tools", [True, False])
async def test_a_regular_view_skill_does_not_decide_whether_skills_are_inlined(monkeypatch, child_skill_tools):
    from open_webui.models.skills import SkillForm, Skills

    manifest = SKILL_MANIFEST.replace("<id>skill</id>", "<id>writer</id>")
    await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content="Write well."))
    try:
        sent, _, _ = await run_entry(
            monkeypatch, "llm_review", answer, regular_tools=tools_with_a_regular_view_skill, skill_context=manifest,
            parent_metadata={
                "model_id": "parent", "files": None, "tool_ids": ["regular"], "chat_id": "local:entry",
                "session_id": "browser", "tool_servers": [], "skill_ids": ["writer"],
            },
            extra_models=None if child_skill_tools else {"child": NO_SKILL_TOOLS_CHILD},
        )
    finally:
        await Skills.delete_skill_by_id("writer")

    agent_requests = [body for body in sent if body["tools"]]
    assert any("submit_review" in body["tools"] for body in agent_requests)
    assert any("submit_draft" in body["tools"] and "view_skill" in body["tools"] for body in agent_requests)
    for body in agent_requests:
        assert (manifest in body["system"]) == child_skill_tools
        assert ("Write well." in body["system"]) != child_skill_tools


@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_regular_view_skill_keeps_the_skill_list_where_core_cannot_inline_skills(monkeypatch, entry):
    if importlib.util.find_spec("open_webui.utils.skill_files"):
        monkeypatch.delattr(importlib.import_module("open_webui.utils.skill_files"), "skill_overview", raising=False)

    sent, _, _ = await run_entry(
        monkeypatch, entry, answer, regular_tools=tools_with_a_regular_view_skill,
        extra_models={"child": NO_SKILL_TOOLS_CHILD},
    )

    agent_requests = [body for body in sent if body["tools"]]
    assert any("view_skill" in body["tools"] for body in agent_requests)
    for body in agent_requests:
        assert (SKILL_MANIFEST in body["system"]) == ("view_skill" in body["tools"])


async def run_without_skill_tools(
    monkeypatch, entry, skill_ids, skill_context, metadata=None, respond=answer, inlet=None, **kwargs,
):
    """Run *entry* with a child model that has no skill tools; return its requests and what inlet saw."""
    seen = []

    def record(form_data):
        seen.append(system_text(form_data))
        return inlet(form_data) if inlet else form_data

    metadata = {
        "model_id": "parent", "files": None, "tool_ids": ["regular"], "chat_id": "local:entry",
        "session_id": "browser", "tool_servers": [], "skill_ids": skill_ids, **(metadata or {}),
    }
    sent, _, _ = await run_entry(
        monkeypatch, entry, respond, regular_tools=loop_tools, parent_metadata=metadata, inlet=record,
        skill_context=skill_context, extra_models={"child": NO_SKILL_TOOLS_CHILD}, **kwargs,
    )
    agent_requests = [body for body in sent if body["tools"]]
    assert any("submit_review" in body["tools"] for body in agent_requests) == (entry == "llm_review")
    assert not any(SKILL_READ_TOOLS & body["tools"] for body in agent_requests)
    return agent_requests, seen


@needs_core_skill_overview
@pytest.mark.parametrize("entry", ENTRIES)
async def test_skills_only_listed_for_a_child_without_skill_tools_are_inlined_like_core(monkeypatch, entry):
    from open_webui.models.skills import SkillForm, Skills
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    pins = []

    async def bound_tools(extra_params):
        async def lookup(query: str = "", __metadata__=None):
            pins.append(copy.deepcopy((__metadata__.get("chat_context") or {}).get("skill_versions")))
            return "found"

        tools = loop_tools()
        tools["lookup"]["callable"] = await get_async_tool_function_and_apply_extra_params(lookup, extra_params)
        return tools

    content = "Write well.\n" + "x" * 20_000
    skill = await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content=content))
    try:
        agent_requests, seen = await run_without_skill_tools(
            monkeypatch, entry, ["writer"], SKILL_MANIFEST.replace("<id>skill</id>", "<id>writer</id>"),
            respond=lookup_first, bound_tools=bound_tools,
        )
    finally:
        await Skills.delete_skill_by_id("writer")

    assert seen and not any("Write well." in text or "<available_skills>" in text for text in seen)
    for body in agent_requests:
        assert body["system"].count(f'<skill id="writer" name="Writer">\n{content}\n</skill>') == 1
        assert "<available_skills>" not in body["system"]
    assert pins and all(pin == {"writer": skill.version_id} for pin in pins)


@needs_core_skill_overview
@pytest.mark.parametrize("module_name", AGENT_MODULES)
async def test_skills_a_child_cannot_load_come_before_the_tool_server_prompts_as_in_core(monkeypatch, module_name):
    from open_webui.models.skills import SkillForm, Skills

    manifest = SKILL_MANIFEST.replace("<id>skill</id>", "<id>writer</id>")
    await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content="Write well."))
    try:
        _, sent, _ = await run_loop(
            monkeypatch, module_name, [{"choices": [{"message": {"content": "done"}}]}], tools=loop_tools(),
            messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
            skills=[manifest], extra_params={"__metadata__": {"skill_ids": ["writer"]}, "__direct_tool_server_system_prompts__": ["Server prompt."]},
        )
    finally:
        await Skills.delete_skill_by_id("writer")

    system = system_text(sent[0])
    assert -1 < system.find("Write well.") < system.find("Server prompt.")


@needs_core_skill_overview
@pytest.mark.parametrize("entry", ENTRIES)
async def test_skills_the_parent_inlined_for_skill_tools_are_rebuilt_for_a_child_without_them(monkeypatch, entry):
    from open_webui.models.skills import SkillForm, Skills

    content = "Write well.\n" + "x" * 20_000
    preview = (
        f'<skill id="writer" name="Writer">\n{content[:5_000]}\nSkill instructions truncated.'
        '\nSkill ID: "writer".\nnext_offset: 5000\n</skill>'
    )
    await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content=content))
    try:
        agent_requests, seen = await run_without_skill_tools(monkeypatch, entry, ["writer"], preview)
    finally:
        await Skills.delete_skill_by_id("writer")

    assert seen and not any("Write well." in text for text in seen)
    for body in agent_requests:
        assert body["system"].count(f'<skill id="writer" name="Writer">\n{content}\n</skill>') == 1
        assert "next_offset" not in body["system"]


@needs_core_skill_tools
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_child_without_skill_tools_keeps_only_the_parent_skill_tags_on_core_0_12_0(monkeypatch, entry):
    from open_webui.utils import skill_files

    monkeypatch.delattr(skill_files, "skill_overview", raising=False)
    tag = '<skill id="writer" name="Writer">\nWrite well.\n</skill>'
    manifest = SKILL_MANIFEST.replace("<id>skill</id>", "<id>notes</id>")

    agent_requests, seen = await run_without_skill_tools(
        monkeypatch, entry, ["writer", "notes"], f"{tag}\n\n{manifest}",
    )

    assert seen and not any("<skill" in text for text in seen)
    for body in agent_requests:
        assert body["system"].count(tag) == 1
        assert "<available_skills>" not in body["system"]


@needs_core_skill_overview
@pytest.mark.parametrize("pinned", ["first", "gone"])
async def test_inlined_skills_stay_on_the_version_the_chat_pinned(monkeypatch, pinned):
    from open_webui.models.skills import SkillForm, Skills

    skill = await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content="First."))
    version = skill.version_id if pinned == "first" else "gone"
    try:
        await Skills.update_skill_by_id("writer", {"content": "Second."})
        agent_requests, _ = await run_without_skill_tools(
            monkeypatch, "sub_agent", ["writer"], SKILL_MANIFEST.replace("<id>skill</id>", "<id>writer</id>"),
            metadata={"chat_context": {"skill_versions": {"writer": version}}},
        )
    finally:
        await Skills.delete_skill_by_id("writer")

    for body in agent_requests:
        assert ("First." in body["system"]) is (pinned == "first")
        assert "Second." not in body["system"]


@needs_core_skill_overview
@pytest.mark.parametrize("entry", ENTRIES)
async def test_inlined_skills_follow_the_version_an_inlet_pins_as_in_core(monkeypatch, entry):
    from open_webui.models.skills import SkillForm, Skills

    first = await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content="First."))
    try:
        second = await Skills.update_skill_by_id("writer", {"content": "Second."})

        def inlet(form_data):
            form_data["metadata"]["chat_context"]["skill_versions"]["writer"] = second.version_id
            return form_data

        agent_requests, _ = await run_without_skill_tools(
            monkeypatch, entry, ["writer"], SKILL_MANIFEST.replace("<id>skill</id>", "<id>writer</id>"),
            metadata={"chat_context": {"skill_versions": {"writer": first.version_id}}}, inlet=inlet,
        )
    finally:
        await Skills.delete_skill_by_id("writer")

    for body in agent_requests:
        assert "Second." in body["system"] and "First." not in body["system"]
        assert body["metadata"]["chat_context"]["skill_versions"] == {"writer": second.version_id}


@needs_core_skill_overview
@pytest.mark.parametrize("entry", ["sub_agent", "parallel_sub_agents", "llm_review"])
@pytest.mark.parametrize("skills_on", [True, False])
async def test_skill_tags_in_the_user_system_prompt_stay_and_bring_no_parent_skills(monkeypatch, entry, skills_on):
    from open_webui.models.skills import SkillForm, Skills

    own = '<skill id="own" name="Own">\nOwn rules.\n</skill>'
    await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content="Write well."))
    try:
        agent_requests, _ = await run_without_skill_tools(
            monkeypatch, entry, ["writer"], '<skill id="writer" name="Writer">\nWrite well.\n</skill>',
            valves={"ENABLE_SKILLS_TOOLS": skills_on}, user_valves={"SYSTEM_PROMPT": own},
        )
    finally:
        await Skills.delete_skill_by_id("writer")

    for body in agent_requests:
        assert body["system"].count(own) == 1
        assert ("Write well." in body["system"]) is skills_on


@needs_core_skill_overview
async def test_terminal_skills_a_child_cannot_load_are_inlined_like_core(monkeypatch):
    async def terminal(event_data):
        path = (event_data.get("data") or {}).get("path") or ""
        return {"data": {"name": "Notes", "content": "Terminal notes."} if path.startswith("/skills/read") else None}

    agent_requests, seen = await run_without_skill_tools(
        monkeypatch, "sub_agent", ["terminal:Notes"],
        SKILL_MANIFEST.replace("<id>skill</id>", "<id>terminal:Notes</id>").replace("Writer", "Notes"),
        metadata={"terminal_id": "user-terminal"}, event_call=terminal,
    )

    assert seen and not any("Terminal notes." in text for text in seen)
    for body in agent_requests:
        assert body["system"].count('<skill name="Notes">\nTerminal notes.\n</skill>') == 1
        assert "<available_skills>" not in body["system"]


@needs_core_skill_overview
async def test_skills_inline_with_the_synchronous_skill_overview_of_core_0_12_1(monkeypatch):
    from open_webui.models.skills import SkillForm, Skills
    from open_webui.utils import skill_files

    def synchronous_overview(content, files, field="files", tools_enabled=True):
        assert not tools_enabled
        return {"content": content, "next_offset": None, **skill_files.bounded_skill_manifest(files, field)}

    monkeypatch.setattr(skill_files, "skill_overview", synchronous_overview)
    await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content="Write well."))
    try:
        agent_requests, _ = await run_without_skill_tools(
            monkeypatch, "sub_agent", ["writer"], SKILL_MANIFEST.replace("<id>skill</id>", "<id>writer</id>"),
        )
    finally:
        await Skills.delete_skill_by_id("writer")

    assert all('<skill id="writer" name="Writer">\nWrite well.\n</skill>' in body["system"] for body in agent_requests)


@needs_core_skill_overview
@pytest.mark.parametrize("entry", ENTRIES)
async def test_children_get_no_skills_when_skills_are_turned_off(monkeypatch, entry):
    from open_webui.models.skills import SkillForm, Skills

    await Skills.insert_new_skill("loop-user", SkillForm(id="writer", name="Writer", content="Write well."))
    try:
        agent_requests, _ = await run_without_skill_tools(
            monkeypatch, entry, ["writer"],
            '<skill id="writer" name="Writer">\nWrite well.\n</skill>\n\n'
            + SKILL_MANIFEST.replace("<id>skill</id>", "<id>writer</id>"),
            valves={"ENABLE_SKILLS_TOOLS": False},
        )
    finally:
        await Skills.delete_skill_by_id("writer")

    assert not any("<skill" in body["system"] or "Write well." in body["system"] for body in agent_requests)


ENTRY_METADATA = {
    "model_id": "parent", "files": None, "tool_ids": ["regular"], "chat_id": "local:entry",
    "session_id": "browser", "tool_servers": [],
}
needs_display_file = pytest.mark.skipif(
    not hasattr(builtin, "display_file"), reason="Core before 0.12.5 has no builtin display_file"
)


def child_model(**meta):
    return {"id": "child", "owned_by": "openai", "info": {"meta": meta, "params": {}}}


def calls_in_order(*steps):
    """Make each (name, arguments) call in turn while the loop offers it, then answer."""
    def respond(body):
        done = sum(message["role"] == "tool" for message in body["messages"])
        if done < len(steps) and steps[done][0] in body["tools"]:
            return tool_call(*steps[done])
        return answer(body)
    return respond


@needs_display_file
@pytest.mark.parametrize("files_on, knowledge_on, knowledge, kb_exec, shown", [
    (True, False, [], False, True),
    (False, True, [], False, True),
    (False, True, [{"type": "note", "id": "note"}], False, False),
    (False, True, [{"type": "note", "id": "note"}], True, True),
    (False, False, [], False, False),
])
async def test_display_file_stays_while_a_category_core_offers_it_through_is_on(
    monkeypatch, files_on, knowledge_on, knowledge, kb_exec, shown,
):
    from open_webui import env

    monkeypatch.setattr(env, "ENABLE_KB_EXEC", kb_exec)
    sent, _, _ = await run_entry(
        monkeypatch, "sub_agent", answer,
        extra_models={"child": child_model(capabilities={"file_context": False}, knowledge=knowledge)},
        parent_metadata={**ENTRY_METADATA, "files": [{"type": "file", "id": "file"}]},
        valves={"ENABLE_FILE_TOOLS": files_on, "ENABLE_KNOWLEDGE_TOOLS": knowledge_on},
    )

    assert sent and all(("display_file" in body["tools"]) == shown for body in sent)


@needs_display_file
@pytest.mark.parametrize("builtin_offered", [True, False])
async def test_uploaded_files_display_through_cores_builtin_behind_a_terminal_display_file(
    monkeypatch, builtin_offered,
):
    import uuid
    from open_webui.models.files import FileForm, Files
    from open_webui.utils import tools as core_tools

    terminal_paths, events = [], []

    async def terminal_display_file(path: str, page: int = None, inline: bool = False):
        terminal_paths.append(path)
        return {"exists": True, "path": path}

    terminal_tool = {
        "type": "terminal", "tool_id": "terminal:terminal", "callable": terminal_display_file,
        "spec": {"name": "display_file", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "page": {"type": "integer"}, "inline": {"type": "boolean"},
        }}},
    }
    monkeypatch.setattr(core_tools, "get_terminal_tools", AsyncMock(return_value=({"display_file": terminal_tool}, "")))

    async def emit(event):
        events.append(event)

    file_id = str(uuid.uuid4())
    await Files.insert_new_file("loop-user", FileForm(
        id=file_id, filename="report.pdf", path="", meta={"name": "report.pdf", "content_type": "application/pdf"},
    ))
    try:
        sent, _, _ = await run_entry(
            monkeypatch, "sub_agent",
            calls_in_order(("display_file", {"path": file_id, "page": 2}), ("display_file", {"path": "/home/report.md"})),
            extra_models={"child": child_model()}, valves={"ENABLE_KNOWLEDGE_TOOLS": builtin_offered},
            settings={"terminal_server.connections": [{"id": "terminal"}]},
            parent_metadata={**ENTRY_METADATA, "terminal_id": "terminal"}, event_emitter=emit,
        )
    finally:
        await Files.delete_file_by_id(file_id)

    upload_result = next(message["content"] for message in sent[-1]["messages"] if message["role"] == "tool")
    attached = [file for event in events if event["type"] == "files" for file in event["data"]["files"]]
    assert terminal_paths == ["/home/report.md"]
    assert [event["data"]["path"] for event in events if event["type"] == "terminal:display_file"] == ["/home/report.md"]
    if builtin_offered:
        assert json.loads(upload_result)["source"] == "open_webui" and "page" in json.loads(upload_result)["note"]
        assert attached == [{
            "type": "file", "id": file_id, "name": "report.pdf",
            "url": f"/api/v1/files/{file_id}/content", "content_type": "application/pdf",
        }]
    else:
        assert upload_result == "Error: Uploaded file display is not available."
        assert attached == []


async def test_terminal_display_file_gets_every_path_on_core_without_the_builtin(monkeypatch):
    import uuid
    from open_webui.utils import tools as core_tools

    terminal_paths = []

    async def terminal_display_file(path: str, page: int = None, inline: bool = False):
        terminal_paths.append(path)
        return {"exists": True, "path": path}

    if hasattr(builtin, "display_file"):
        monkeypatch.delattr(builtin, "display_file")
    monkeypatch.setattr(core_tools, "get_terminal_tools", AsyncMock(return_value=({"display_file": {
        "type": "terminal", "tool_id": "terminal:terminal", "callable": terminal_display_file,
        "spec": {"name": "display_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
    }}, "")))
    named_by_id = str(uuid.uuid4())

    await run_entry(
        monkeypatch, "sub_agent", calling(("display_file", {"path": named_by_id})),
        settings={"terminal_server.connections": [{"id": "terminal"}]},
        parent_metadata={**ENTRY_METADATA, "terminal_id": "terminal"},
    )

    assert terminal_paths == [named_by_id]


async def test_a_regular_display_file_result_is_not_taken_for_cores_upload(monkeypatch):
    claimed = {"source": "open_webui", "exists": True, "path": "report.pdf"}
    events = []

    async def display_file(path: str):
        return claimed

    async def emit(event):
        events.append(event)

    tool = {
        "callable": display_file, "type": "tool", "tool_id": "records",
        "spec": {"name": "display_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
    }
    sent, _, _ = await run_entry(
        monkeypatch, "sub_agent", calling(("display_file", {"path": "report.pdf"})),
        regular_tools=lambda: {**loop_tools(), "display_file": tool}, event_emitter=emit,
    )
    parallel = await parallel_tools.Tools().run_tools_parallel(
        [parallel_tools.ToolCallItem(name="display_file", args={"path": "report.pdf"})],
        __request__=SimpleNamespace(), __user__=make_user().model_dump(), __metadata__={"tools": {"display_file": tool}},
        __event_emitter__=emit,
    )

    child_result = next(message["content"] for message in sent[-1]["messages"] if message["role"] == "tool")
    assert json.loads(child_result) == claimed
    assert json.loads(parallel)["results"][0]["result"] == claimed
    assert not [event for event in events if event["type"] == "files"]


@needs_display_file
async def test_a_browser_terminals_display_file_gets_cores_builtin_spec_and_sends_uploads_to_it(monkeypatch):
    import uuid
    from open_webui.models.files import FileForm, Files
    from open_webui.tools.builtin import display_file
    from open_webui.utils.tools import get_builtin_tool_spec

    browser_paths, events = [], []

    async def event_call(event):
        browser_paths.append(event["data"]["params"]["path"])
        return {"exists": True, "path": event["data"]["params"]["path"]}

    async def emit(event):
        events.append(event)

    server = {"url": "http://terminal", "is_terminal": True, "specs": [{
        "name": "display_file", "description": "Show a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }]}
    file_id = str(uuid.uuid4())
    await Files.insert_new_file("loop-user", FileForm(
        id=file_id, filename="report.pdf", path="", meta={"name": "report.pdf", "content_type": "application/pdf"},
    ))
    try:
        sent, _, _ = await run_entry(
            monkeypatch, "sub_agent",
            calls_in_order(("display_file", {"path": file_id}), ("display_file", {"path": "/home/report.md"})),
            extra_models={"child": child_model()}, valves={"ENABLE_KNOWLEDGE_TOOLS": True},
            parent_metadata={**ENTRY_METADATA, "tool_servers": [server]}, event_call=event_call, event_emitter=emit,
        )
    finally:
        await Files.delete_file_by_id(file_id)

    upload_result = next(message["content"] for message in sent[-1]["messages"] if message["role"] == "tool")
    assert browser_paths == ["/home/report.md"]
    assert json.loads(upload_result)["source"] == "open_webui"
    assert [file["id"] for event in events if event["type"] == "files" for file in event["data"]["files"]] == [file_id]
    expected = get_builtin_tool_spec(display_file)
    specs = [
        tool["function"] for body in sent for tool in body["raw_tools"]
        if tool.get("function", {}).get("name") == "display_file"
    ]
    assert specs and all(
        (spec["description"], set(spec["parameters"]["properties"]))
        == (expected["description"], set(expected["parameters"]["properties"]))
        for spec in specs
    )


needs_saved_message_files = pytest.mark.skipif(
    tuple(int(part) for part in re.findall(r"\d+", importlib.import_module("open_webui.env").VERSION)[:2]) < (0, 12),
    reason="Core before 0.12.0 adds message files to the chat history only",
)
FILES_USER = {**make_user("admin").model_dump(), "id": "files-user", "valves": {}}


async def files_chat(chat_id, files=()):
    from open_webui.models.chats import ChatForm, Chats
    from open_webui.models.users import Users

    if not await Users.get_user_by_id("files-user"):
        await Users.insert_new_user("files-user", "Files User", "files@example.com", role="admin")
    chat = await Chats.insert_new_chat(chat_id, "files-user", ChatForm(chat={"history": {"messages": {
        "reply": {"id": "reply", "role": "assistant", "content": "", "files": list(files)},
    }}}))
    return chat, {**ENTRY_METADATA, "chat_id": chat.id, "message_id": "reply", "user_id": "files-user"}


def capture_file_lists(monkeypatch, on_send=None):
    """Record (room, type, files) for each file event Core sends to a socket room."""
    from open_webui.socket import main as socket_main

    sent = []

    async def capture(event, payload, room=None, **_):
        data = payload["data"]
        if data.get("type") in ("files", "chat:message:files"):
            sent.append((room, data["type"], copy.deepcopy(data["data"]["files"])))
            if on_send:
                on_send(room, data["data"]["files"])

    monkeypatch.setattr(socket_main.sio, "emit", capture)
    return sent


def last_files(sent, room):
    return [files for to, _, files in sent if to == room][-1]


@needs_saved_message_files
@pytest.mark.parametrize("entry", ENTRIES)
async def test_file_events_add_new_files_once_and_show_everyone_the_full_list(monkeypatch, entry):
    from open_webui.models.chats import Chats
    from open_webui.socket import main as socket_main
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    sent = capture_file_lists(monkeypatch)
    earlier, attached, second = {"type": "image", "url": "/earlier"}, {"type": "file", "id": "a"}, {"type": "file", "id": "b"}
    chat, metadata = await files_chat(f"files-{entry}", [earlier])

    async def bound_tools(extra_params):
        async def make_files(__event_emitter__=None):
            await __event_emitter__({"type": "files", "data": {"files": [attached, attached]}})
            await __event_emitter__({"type": "files", "data": {"files": [attached, second]}})
            return "made"

        tools = loop_tools()
        tools["make_files"] = {
            **tools["ping"], "spec": {**tools["ping"]["spec"], "name": "make_files"},
            "callable": await get_async_tool_function_and_apply_extra_params(make_files, extra_params),
        }
        return tools

    try:
        await run_entry(
            monkeypatch, entry, calling(("make_files", {})), bound_tools=bound_tools, parent_metadata=metadata,
            event_emitter=await socket_main.get_event_emitter(metadata),
        )
        saved = await Chats.get_message_metadata(chat.id, "reply", "files")
    finally:
        await Chats.delete_chat_by_id(chat.id)

    assert saved == [earlier, attached, second]
    assert last_files(sent, "user:files-user") == last_files(sent, f"chat:{chat.id}") == saved


@needs_saved_message_files
@pytest.mark.parametrize("entry", ["sub_agent", "llm_review", "magi_decision_support", "multi_model_council", "parallel_tools"])
async def test_tools_can_send_file_events_by_cores_keyword(monkeypatch, entry):
    from open_webui.models.chats import Chats
    from open_webui.socket import main as socket_main
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    capture_file_lists(monkeypatch)
    attached = {"type": "file", "id": "attached"}
    chat, metadata = await files_chat(f"keyword-{entry}")

    async def send_file(__event_emitter__=None):
        await __event_emitter__(event_data={"type": "files", "data": {"files": [attached]}})
        return "sent"

    async def bound_tools(extra_params):
        tools = loop_tools()
        tools["send_file"] = {
            **tools["ping"], "spec": {**tools["ping"]["spec"], "name": "send_file"},
            "callable": await get_async_tool_function_and_apply_extra_params(send_file, extra_params),
        }
        return tools

    try:
        if entry == "parallel_tools":
            result = await parallel_tools.Tools().run_tools_parallel(
                [parallel_tools.ToolCallItem(name="send_file")], __request__=SimpleNamespace(), __user__=FILES_USER,
                __metadata__={**metadata, "tools": await bound_tools({})},
                __event_emitter__=await socket_main.get_event_emitter(metadata),
            )
            results = [item["result"] for item in json.loads(result)["results"]]
        else:
            sent, _, _ = await run_entry(
                monkeypatch, entry, calling(("send_file", {})), bound_tools=bound_tools, parent_metadata=metadata,
                event_emitter=await socket_main.get_event_emitter(metadata),
            )
            results = [
                message["content"] for body in sent for message in body["messages"] if message["role"] == "tool"
                and message.get("tool_call_id") == "send_file"
            ]
        saved = await Chats.get_message_metadata(chat.id, "reply", "files")
    finally:
        await Chats.delete_chat_by_id(chat.id)

    assert results and set(results) == {"sent"}
    assert saved == [attached]


@needs_saved_message_files
@pytest.mark.parametrize("failure", ["raises", "reply_missing"])
async def test_a_failed_file_save_still_shows_the_tools_files_and_the_agent_goes_on(monkeypatch, failure):
    from open_webui.models.chats import Chats
    from open_webui.socket import main as socket_main
    from open_webui.utils import files as core_files
    from open_webui.utils import middleware

    shot = {"type": "image", "url": "/api/v1/files/shot/content"}
    monkeypatch.setattr(middleware, "get_file_url_from_base64", AsyncMock(return_value=shot["url"]))
    monkeypatch.setattr(core_files, "get_image_base64_from_url", AsyncMock(return_value="data:image/png;base64,aGVsbG8="))

    async def add(chat_id, message_id, files):
        if failure == "raises":
            raise RuntimeError("database is locked")

    def screenshot_tools():
        async def screenshot():
            return [{"type": "image", "mimeType": "image/png", "data": "aGVsbG8="}]

        tools = loop_tools()
        tools["screenshot"] = {
            **tools["ping"], "type": "mcp", "spec": {**tools["ping"]["spec"], "name": "screenshot"},
            "callable": screenshot,
        }
        return tools

    sent_files = capture_file_lists(monkeypatch)
    monkeypatch.setattr(Chats, "add_message_files_by_id_and_message_id", add)
    earlier = {"type": "file", "id": "earlier"}
    chat, metadata = await files_chat(f"failed-save-{failure}", [earlier])
    try:
        sent, _, _ = await run_entry(
            monkeypatch, "sub_agent", calling(("screenshot", {})), regular_tools=screenshot_tools,
            parent_metadata=metadata, event_emitter=await socket_main.get_event_emitter(metadata),
        )
    finally:
        await Chats.delete_chat_by_id(chat.id)

    assert any(called(body, "screenshot") for body in sent)
    assert last_files(sent_files, "user:files-user") == [earlier, shot]


@pytest.mark.skipif(
    tuple(int(part) for part in re.findall(r"\d+", importlib.import_module("open_webui.env").VERSION)[:2]) < (0, 12),
    reason="Core before 0.12.0 sends chat events only to rooms with a connected socket",
)
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_tools_display_only_file_list_reaches_the_screen_as_sent(monkeypatch, entry):
    from open_webui.models.chats import Chats
    from open_webui.socket import main as socket_main
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    sent = capture_file_lists(monkeypatch)
    previous, preview = {"type": "file", "id": "old", "url": "/old"}, {"type": "file", "id": "preview", "url": "/preview"}
    chat, metadata = await files_chat(f"preview-{entry}", [previous])

    async def bound_tools(extra_params):
        async def preview_file(__event_emitter__=None):
            await __event_emitter__({"type": "chat:message:files", "data": {"files": [preview]}})
            await __event_emitter__({"type": "chat:message:files", "data": {"files": []}})
            return "previewed"

        tools = loop_tools()
        tools["preview_file"] = {
            **tools["ping"], "spec": {**tools["ping"]["spec"], "name": "preview_file"},
            "callable": await get_async_tool_function_and_apply_extra_params(preview_file, extra_params),
        }
        return tools

    try:
        await run_entry(
            monkeypatch, entry, calling(("preview_file", {})), bound_tools=bound_tools, parent_metadata=metadata,
            event_emitter=await socket_main.get_event_emitter(metadata),
        )
        saved = await Chats.get_message_metadata(chat.id, "reply", "files")
    finally:
        await Chats.delete_chat_by_id(chat.id)

    shown = [files for _, _, files in sent]
    assert saved == [previous]
    assert shown and all(files in ([preview], []) for files in shown) and shown.count([preview]) == shown.count([])


@needs_display_file
async def test_parallel_children_keep_every_saved_attachment_and_show_the_latest_list(monkeypatch):
    import uuid
    from open_webui.models.chats import Chats
    from open_webui.models.files import FileForm, Files
    from open_webui.socket import main as socket_main

    upload_writing, image_saved, upload_shown = asyncio.Event(), asyncio.Event(), asyncio.Event()
    file_id, image_id = str(uuid.uuid4()), str(uuid.uuid4())
    generated = {"id": image_id, "name": "generated.png", "url": f"/api/v1/files/{image_id}/content"}

    def has_upload(files):
        return any(isinstance(file, dict) and file.get("id") == file_id for file in files or [])

    sent = capture_file_lists(
        monkeypatch, lambda room, files: room == "user:files-user" and has_upload(files) and upload_shown.set(),
    )
    original_add = Chats.add_message_files_by_id_and_message_id
    original_upsert = Chats.upsert_message_to_chat_by_id_and_message_id

    async def before_upload_write():
        upload_writing.set()
        await asyncio.wait_for(image_saved.wait(), 10)

    async def add(chat_id, message_id, files):
        if has_upload(files):
            await before_upload_write()
            return await original_add(chat_id, message_id, files)
        saved = await original_add(chat_id, message_id, files)
        image_saved.set()
        await asyncio.wait_for(upload_shown.wait(), 10)
        return saved

    async def upsert(chat_id, message_id, message, **kwargs):
        if has_upload(message.get("files")):
            await before_upload_write()
        return await original_upsert(chat_id, message_id, message, **kwargs)

    async def image_provider(**_):
        await asyncio.wait_for(upload_writing.wait(), 10)
        return [generated]

    monkeypatch.setattr(Chats, "add_message_files_by_id_and_message_id", add)
    monkeypatch.setattr(Chats, "upsert_message_to_chat_by_id_and_message_id", upsert)
    monkeypatch.setattr(builtin, "image_generations", image_provider)

    def respond(body):
        task = next(message["content"] for message in body["messages"] if message["role"] == "user")
        name, arguments = (
            ("display_file", {"path": file_id}) if task.startswith("display") else ("generate_image", {"prompt": "A cat"})
        )
        if name in body["tools"] and not called(body, name):
            return tool_call(name, arguments)
        return answer(body)

    chat, metadata = await files_chat(str(uuid.uuid4()))
    metadata = {**metadata, "files": [{"type": "file", "id": file_id}], "features": {"image_generation": True}}
    await Files.insert_new_file("files-user", FileForm(
        id=file_id, filename="report.pdf", path="", meta={"name": "report.pdf", "content_type": "application/pdf"},
    ))
    try:
        await run_entry(
            monkeypatch, "parallel_sub_agents", respond, skill_context="", parent_metadata=metadata,
            event_emitter=await socket_main.get_event_emitter(metadata), settings={"image_generation.enable": True},
            extra_models={"child": child_model(capabilities={"file_context": False})},
            entry_kwargs={
                "tasks": [{"description": task, "prompt": task} for task in ("display", "image")],
                "__user__": FILES_USER, "__chat_id__": chat.id, "__message_id__": "reply",
            },
        )
        saved = await Chats.get_message_metadata(chat.id, "reply", "files")
    finally:
        await Chats.delete_chat_by_id(chat.id)
        await Files.delete_file_by_id(file_id)

    assert upload_writing.is_set() and image_saved.is_set()
    assert sorted(file["id"] for file in saved) == sorted([file_id, image_id])
    assert last_files(sent, "user:files-user") == last_files(sent, f"chat:{chat.id}") == saved


@needs_saved_message_files
async def test_nested_agent_tools_show_file_lists_in_the_order_they_were_saved(monkeypatch):
    import contextlib
    from open_webui.models.chats import Chats
    from open_webui.socket import main as socket_main
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    first, second = {"type": "file", "id": "first"}, {"type": "file", "id": "second"}
    first_added, second_shown = asyncio.Event(), asyncio.Event()
    sent = capture_file_lists(monkeypatch, lambda room, files: second in files and second_shown.set())
    original_add = Chats.add_message_files_by_id_and_message_id

    async def add(chat_id, message_id, files):
        saved = await original_add(chat_id, message_id, files)
        if first in files:
            first_added.set()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(second_shown.wait(), 0.5)
        return saved

    def sending(file, after=None):
        async def send_file(__event_emitter__=None):
            if after:
                await asyncio.wait_for(after.wait(), 10)
            await __event_emitter__({"type": "files", "data": {"files": [file]}})
            return "sent"
        return send_file

    def tool(name, callable_, properties=None):
        return {
            "type": "tool", "tool_id": name, "callable": callable_,
            "spec": {"name": name, "parameters": {"type": "object", "properties": properties or {}}},
        }

    async def bound_tools(extra_params):
        return {
            "run_tools_parallel": tool(
                "run_tools_parallel",
                await get_async_tool_function_and_apply_extra_params(parallel_tools.Tools().run_tools_parallel, extra_params),
                {"tool_calls": {"type": "array"}},
            ),
            "send_first": tool("send_first", await get_async_tool_function_and_apply_extra_params(sending(first), {})),
            "send_second": tool(
                "send_second", await get_async_tool_function_and_apply_extra_params(sending(second, first_added), extra_params),
            ),
        }

    def respond(body):
        task = next(message["content"] for message in body["messages"] if message["role"] == "user")
        name, arguments = (
            ("run_tools_parallel", {"tool_calls": [{"name": "send_first", "args": {}}]})
            if task.startswith("nested") else ("send_second", {})
        )
        if name in body["tools"] and not called(body, name):
            return tool_call(name, arguments)
        return answer(body)

    monkeypatch.setattr(Chats, "add_message_files_by_id_and_message_id", add)
    chat, metadata = await files_chat("files-nested")
    try:
        await run_entry(
            monkeypatch, "parallel_sub_agents", respond, skill_context="", bound_tools=bound_tools,
            parent_metadata=metadata, event_emitter=await socket_main.get_event_emitter(metadata),
            entry_kwargs={
                "tasks": [{"description": task, "prompt": task} for task in ("nested", "direct")],
                "__user__": FILES_USER,
            },
        )
        saved = await Chats.get_message_metadata(chat.id, "reply", "files")
    finally:
        await Chats.delete_chat_by_id(chat.id)

    assert first_added.is_set() and saved == [first, second]
    assert last_files(sent, "user:files-user") == saved


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("vision", [True, False])
async def test_tool_images_reach_a_child_only_with_vision_on(monkeypatch, entry, vision):
    image = "data:image/png;base64,aGVsbG8="

    def screenshot_tools():
        async def screenshot():
            return {"result": {"screenshot": image}}

        tools = loop_tools()
        tools["screenshot"] = {
            **tools["ping"], "spec": {**tools["ping"]["spec"], "name": "screenshot"}, "callable": screenshot,
        }
        return tools

    sent, _, _ = await run_entry(
        monkeypatch, entry, calling(("screenshot", {})), regular_tools=screenshot_tools,
        extra_models={"child": child_model(capabilities={"vision": vision})},
    )

    after = [body for body in sent if called(body, "screenshot")]
    assert after
    for body in after:
        result = next(message["content"] for message in body["messages"] if message["role"] == "tool")
        images = [message for message in body["messages"] if image in json.dumps(message["content"])]
        assert result.endswith("\n[image]") is not vision
        assert bool(images) is vision


@pytest.mark.skipif(
    tuple(int(part) for part in re.findall(r"\d+", importlib.import_module("open_webui.env").VERSION)[:2]) < (0, 12),
    reason="Core before 0.12.0 keeps URL tool images display-only",
)
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_child_without_vision_does_not_fetch_url_tool_images(monkeypatch, entry):
    from open_webui.utils import files as core_files
    from open_webui.utils import middleware

    fetch = AsyncMock(return_value="data:image/png;base64,ZmV0Y2hlZA==")
    monkeypatch.setattr(core_files, "get_image_base64_from_url", fetch)
    monkeypatch.setattr(middleware, "get_file_url_from_base64", AsyncMock(return_value="/api/v1/files/shot/content"))

    def screenshot_tools():
        async def screenshot():
            return [{"type": "image", "mimeType": "image/png", "data": "aGVsbG8="}]

        tools = loop_tools()
        tools["screenshot"] = {
            **tools["ping"], "type": "mcp", "spec": {**tools["ping"]["spec"], "name": "screenshot"},
            "callable": screenshot,
        }
        return tools

    sent, _, _ = await run_entry(
        monkeypatch, entry, calling(("screenshot", {})), regular_tools=screenshot_tools,
        extra_models={"child": child_model(capabilities={"vision": False})},
    )

    after = [body for body in sent if called(body, "screenshot")]
    assert after and fetch.await_count == 0
    assert all(
        next(message["content"] for message in body["messages"] if message["role"] == "tool").endswith("\n[image]")
        for body in after
    )


@pytest.mark.skipif(
    not hasattr(importlib.import_module("open_webui.models.notes"), "NotePageConflict"),
    reason="Core before 0.12.5 lets replace_note_content edit unread notes",
)
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("parent_read", [False, True])
async def test_children_edit_notes_only_after_reading_them_themselves(monkeypatch, entry, parent_read):
    from open_webui.models.notes import NoteForm, Notes
    from open_webui.models.users import Users

    monkeypatch.setattr(Users, "get_user_by_id", AsyncMock(return_value=make_user("admin")))
    note = await Notes.insert_new_note("loop-user", NoteForm(title="Plan", data={"content": {"md": "old"}}))
    steps = [("replace_note_content", {"note_id": note.id, "content": "new"})]
    if not parent_read:
        steps.insert(0, ("view_note", {"note_id": note.id}))
    pages = (note.data or {}).get("pages") or [{}]
    try:
        sent, _, _ = await run_entry(
            monkeypatch, entry, calls_in_order(*steps), extra_models={"child": child_model()},
            settings={"notes.enable": True}, valves={"ENABLE_NOTES_TOOLS": True},
            request_state={"note_page_revisions": {
                ("loop-user", note.id, pages[0].get("id")): pages[0].get("revision", 0),
            }} if parent_read else None,
        )
    finally:
        await Notes.delete_note_by_id(note.id)

    edits = [
        json.loads(message["content"]) for body in sent for message in body["messages"]
        if message["role"] == "tool" and message["tool_call_id"] == "replace_note_content"
    ]
    codes = [edit.get("code") or edit.get("status") for edit in edits]
    if parent_read:
        assert codes and set(codes) == {"read_required"}
    else:
        # Agents that run at once and read before another's edit get Core's conflict.
        assert "success" in codes and set(codes) <= {"success", "page_conflict"}


@pytest.mark.skipif(
    not hasattr(importlib.import_module("open_webui.models.notes"), "NotePageConflict"),
    reason="Core before 0.12.5 lets replace_note_content edit unread notes",
)
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_childs_note_reads_through_parallel_tools_stay_its_own(monkeypatch, entry):
    from open_webui.models.notes import NoteForm, Notes
    from open_webui.models.users import Users

    monkeypatch.setattr(Users, "get_user_by_id", AsyncMock(return_value=make_user("admin")))
    note = await Notes.insert_new_note("loop-user", NoteForm(title="Plan", data={"content": {"md": "old"}}))
    parent_reads = {}
    try:
        sent, _, _ = await run_entry(
            monkeypatch, entry, calls_in_order(
                ("run_tools_parallel", {"tool_calls": [{"name": "view_note", "args": {"note_id": note.id}}]}),
                ("replace_note_content", {"note_id": note.id, "content": "new"}),
            ),
            extra_models={"child": child_model()}, settings={"notes.enable": True},
            valves={"ENABLE_NOTES_TOOLS": True}, bound_tools=parallel_tools_for,
            request_state={"note_page_revisions": parent_reads},
        )
    finally:
        await Notes.delete_note_by_id(note.id)

    edits = [
        json.loads(message["content"]) for body in sent for message in body["messages"]
        if message["role"] == "tool" and message["tool_call_id"] == "replace_note_content"
    ]
    codes = [edit.get("code") or edit.get("status") for edit in edits]
    assert parent_reads == {}
    assert "success" in codes and set(codes) <= {"success", "page_conflict"}


@needs_display_file
async def test_parallel_tools_route_and_attach_displayed_uploads_with_the_replys_full_file_list(monkeypatch):
    import uuid
    from open_webui.models.chats import ChatForm, Chats
    from open_webui.models.files import FileForm, Files
    from open_webui.models.users import Users
    from open_webui.socket import main as socket_main
    from open_webui.tools.builtin import display_file
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    shown, terminal_paths = [], []

    async def capture(event, payload, **_):
        shown.append(copy.deepcopy(payload["data"]))

    async def terminal_display_file(path: str, page: int = None, inline: bool = False):
        terminal_paths.append(path)
        return {"exists": True, "path": path}

    monkeypatch.setattr(socket_main.sio, "emit", capture)
    if not await Users.get_user_by_id("files-user"):
        await Users.insert_new_user("files-user", "Files User", "files@example.com", role="admin")
    user = (await Users.get_user_by_id("files-user")).model_dump()
    earlier = {"type": "image", "url": "/earlier"}
    chat = await Chats.insert_new_chat("files-parallel", "files-user", ChatForm(chat={"history": {"messages": {
        "reply": {"id": "reply", "role": "assistant", "content": "", "files": [earlier]},
    }}}))
    file_id = str(uuid.uuid4())
    await Files.insert_new_file("files-user", FileForm(
        id=file_id, filename="report.pdf", path="", meta={"name": "report.pdf", "content_type": "application/pdf"},
    ))
    metadata = {"chat_id": chat.id, "message_id": "reply", "user_id": "files-user", "terminal_id": "terminal"}
    metadata["tools"] = {"display_file": {
        "type": "terminal", "tool_id": "terminal:terminal", "callable": terminal_display_file,
        "spec": {"name": "display_file", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "page": {"type": "integer"},
        }}},
        "file_tool": {
            "type": "builtin", "tool_id": "builtin:display_file",
            "callable": await get_async_tool_function_and_apply_extra_params(
                display_file, {"__request__": SimpleNamespace(), "__user__": user},
            ),
            "spec": {"name": "display_file", "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "page": {"type": "integer"},
            }}},
        },
    }}
    try:
        result = await parallel_tools.Tools().run_tools_parallel(
            [
                parallel_tools.ToolCallItem(name="display_file", args={"path": file_id, "page": 2}),
                parallel_tools.ToolCallItem(name="display_file", args={"path": "/home/report.md"}),
                parallel_tools.ToolCallItem(name="display_file", args={"path": file_id}),
            ],
            __request__=SimpleNamespace(), __user__=user, __metadata__=metadata,
            __event_emitter__=await socket_main.get_event_emitter(metadata),
        )
        saved = await Chats.get_message_metadata(chat.id, "reply", "files")
    finally:
        await Files.delete_file_by_id(file_id)
        await Chats.delete_chat_by_id(chat.id)

    upload, terminal, _ = json.loads(result)["results"]
    file_lists = [data["data"]["files"] for data in shown if data["type"] in ("files", "chat:message:files")]
    assert upload["result"]["source"] == "open_webui" and "page" in upload["result"]["note"]
    assert terminal_paths == ["/home/report.md"]
    assert [data["data"]["path"] for data in shown if data["type"] == "terminal:display_file"] == ["/home/report.md"]
    assert [file["id"] for file in saved if file.get("id")] == [file_id] and earlier in saved
    assert file_lists and file_lists[-1] == saved


async def test_file_events_pass_through_on_core_before_0_12(monkeypatch):
    from open_webui import env
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    monkeypatch.setattr(env, "VERSION", "0.11.4")
    events, attached = [], {"type": "file", "id": "attached"}

    async def attach(__event_emitter__=None):
        await __event_emitter__({"type": "files", "data": {"files": [attached]}})
        return "attached"

    async def emit(event):
        events.append(event)

    tools = {"attach": {
        "type": "tool", "callable": await get_async_tool_function_and_apply_extra_params(attach, {}),
        "spec": {"name": "attach", "parameters": {"type": "object", "properties": {}}},
    }}
    result = await parallel_tools.Tools().run_tools_parallel(
        [parallel_tools.ToolCallItem(name="attach")],
        __request__=SimpleNamespace(), __user__=make_user().model_dump(),
        __metadata__={"chat_id": "saved-chat", "message_id": "reply", "user_id": "user", "tools": tools},
        __event_emitter__=emit,
    )

    assert json.loads(result)["results"][0]["result"] == "attached"
    assert [event for event in events if "files" in event["type"]] == [{"type": "files", "data": {"files": [attached]}}]


needs_create_tool = pytest.mark.skipif(not hasattr(builtin, "create_tool"), reason="Core before 0.13.0 has no create_tool")
needs_sub_agent_models = pytest.mark.skipif(
    "model_id" not in getattr(getattr(builtin, "delegate_task", None), "__annotations__", {}),
    reason="Core before 0.13.0 delegates only to the current model",
)
ROUTED_CHILD = {"id": "child", "owned_by": "openai", "info": {"meta": {
    "builtinTools": {category: False for category in ("time", "user_input", "knowledge", "chats", "tasks")},
    "subagents": {"model_ids": ["worker"]},
}, "params": {}}}
WORKER = {"id": "worker", "owned_by": "openai", "info": {"meta": {}}}


def capture_sub_agent_chats(monkeypatch):
    from open_webui.models.chats import Chats

    created = []

    async def insert_new_chat(chat_id, user_id, form_data, *args, **kwargs):
        created.append(form_data.chat["models"])

    monkeypatch.setattr(Chats, "insert_new_chat", insert_new_chat)
    return created


@needs_create_tool
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("enabled", [False, True])
async def test_tool_creation_needs_its_own_valve(monkeypatch, entry, enabled):
    sent, _, _ = await run_entry(
        monkeypatch, entry, answer, valves={"ENABLE_TOOL_CREATION_TOOLS": True} if enabled else None,
    )

    assert sent
    assert any("create_tool" in body["tools"] for body in sent) is enabled


@needs_sub_agent_models
@pytest.mark.parametrize("entry", ENTRIES)
async def test_children_delegate_to_the_sub_agent_models_they_are_offered(monkeypatch, entry):
    created = capture_sub_agent_chats(monkeypatch)
    offered = []

    def respond(body):
        delegate = next((tool for tool in body["raw_tools"] if tool["function"]["name"] == "delegate_task"), None)
        if delegate and not any(message["role"] == "tool" for message in body["messages"]):
            offered.append(delegate["function"]["parameters"]["properties"]["model_id"]["enum"])
            return tool_call("delegate_task", {"task": "Check", "model_id": "worker"})
        return answer(body)

    parent_metadata = {
        "model_id": "parent", "model": {"id": "parent", "owned_by": "arena", "info": {"meta": {}}},
        "files": None, "tool_ids": ["regular"], "features": {}, "chat_id": "chat-1",
        "session_id": "browser", "tool_servers": [],
    }
    parent_before = copy.deepcopy(parent_metadata)
    await run_entry(
        monkeypatch, entry, respond, settings={"subagents.enable": True}, parent_metadata=parent_metadata,
        valves={"ENABLE_SUBAGENT_TOOLS": True}, extra_models={"child": ROUTED_CHILD, "worker": WORKER},
        entry_kwargs={"__chat_id__": "chat-1"},
    )

    assert offered and all(models == ["worker"] for models in offered)
    assert created and all(models == ["worker"] for models in created)
    assert parent_metadata == parent_before


def parallel_request(*models):
    from starlette.requests import Request

    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
        MODELS={model["id"]: model for model in models},
    )), "state": {}})
    request._body = b"{}"
    return request


def keep_core_config(monkeypatch, settings):
    """Override *settings* and read every other setting from Core's store."""
    from open_webui.models.config import Config

    get, get_many = Config.get, Config.get_many

    async def patched_get(key, *args, **kwargs):
        return settings[key] if key in settings else await get(key, *args, **kwargs)

    async def patched_get_many(*keys):
        return {**await get_many(*keys), **{key: settings[key] for key in keys if key in settings}}

    monkeypatch.setattr(Config, "get", patched_get)
    monkeypatch.setattr(Config, "get_many", patched_get_many)


PARALLEL_TOOLS_SOURCE = Path(parallel_tools.__file__).read_text()


def workspace_tool(name, prefix):
    """Python source of a workspace tool whose ``name`` method returns ``prefix`` and its command."""
    return f"class Tools:\n    async def {name}(self, command: str = '') -> str:\n        return {prefix!r} + command\n"


async def save_workspace_tools(monkeypatch, user, sources, valves=None):
    """Save *sources* ({tool_id: Python source}) as the user's tools, which Core loads from their source."""
    from open_webui.models.tools import Tools
    from open_webui.utils.plugin import load_tool_module_by_id
    from open_webui.utils.tools import get_tool_specs

    records = {}
    for tool_id, content in sources.items():
        module = (await load_tool_module_by_id(tool_id, content=content))[0]
        records[tool_id] = SimpleNamespace(
            id=tool_id, user_id=user.id, is_active=True, content=content, specs=get_tool_specs(module),
        )
    stores = {
        "get_tools_by_ids": lambda ids: {tool_id: records[tool_id] for tool_id in ids if tool_id in records},
        "get_tool_by_id": records.get,
        "is_tool_active": lambda tool_id: True,
        "get_tool_valves_by_id": lambda tool_id: (valves or {}) if tool_id == "parallel" else {},
        "get_user_valves_by_id_and_user_id": lambda tool_id, user_id: {},
    }
    for name, read in stores.items():
        monkeypatch.setattr(Tools, name, AsyncMock(side_effect=read), raising=False)


async def core_chat_tools(
    monkeypatch, request, user, model, *, sent_to=None, workspace=None, unattached=None, valves=None, tool_ids=(),
    form=None, **metadata,
):
    """Send a chat with Parallel Tools and *workspace* tools attached through Core.

    *unattached* tools are saved but not attached. *sent_to* is the model this response goes to when it is not
    the chat's first model, as for the other models of a multi-model message. Returns the metadata Core gives the
    chat's tools.
    """
    from open_webui.utils import chat as core_chat
    from open_webui.utils.middleware import process_chat_payload

    attached = {"parallel": PARALLEL_TOOLS_SOURCE, **(workspace or {})}
    await save_workspace_tools(monkeypatch, user, {**attached, **(unattached or {})}, valves)
    monkeypatch.setattr(core_chat, "generate_openai_chat_completion", AsyncMock(return_value={}))
    tool_ids = [*attached, *tool_ids]
    metadata = {
        "model": model, "model_id": model["id"], "chat_id": "", "message_id": "reply", "session_id": "browser",
        "user_id": user.id, "tool_ids": tool_ids, "files": None, "features": {},
        "params": {"function_calling": "native"}, "tool_servers": [], **metadata,
    }
    sent_to = sent_to or model
    form_data = {
        "model": sent_to["id"], "messages": [{"role": "user", "content": "Go"}], "tool_ids": tool_ids,
        "metadata": metadata, **(form or {}),
    }
    request.state.metadata = metadata
    form_data, metadata = (await process_chat_payload(request, form_data, user, metadata, sent_to))[:2]
    await core_chat.generate_chat_completion(request, form_data, user)
    return metadata


async def call_parallel(metadata, *calls):
    """Run (name, args) *calls* through the Parallel Tools callable Core gave the chat; return each result."""
    raw = await metadata["tools"]["run_tools_parallel"]["callable"](
        tool_calls=[{"name": name, "args": args} for name, args in calls],
    )
    return [item["result"] for item in json.loads(raw)["results"]]


def parallel_tools_model(metadata):
    return metadata["tools"]["run_tools_parallel"]["callable"].__extra_params__["__model__"]["id"]


@needs_sub_agent_models
async def test_parallel_tools_delegate_to_the_chat_models_sub_agent_models(monkeypatch):
    keep_core_config(monkeypatch, {"subagents.enable": True, "task.model.external": "task"})
    created = capture_sub_agent_chats(monkeypatch)
    chat_model = {"id": "chat", "owned_by": "openai", "info": {"meta": {"subagents": {"model_ids": ["worker"]}}}}
    task_model = {"id": "task", "owned_by": "openai", "info": {"meta": {}}}
    request = parallel_request(chat_model, task_model, WORKER)
    metadata = await core_chat_tools(monkeypatch, request, make_user("admin"), chat_model, chat_id="chat-1")
    assert parallel_tools_model(metadata) == "task"

    await call_parallel(metadata, ("delegate_task", {"task": "Check", "model_id": "worker"}))

    assert created == [["worker"]]


CURRENT_MODEL_CALLS = {
    "sub_agent": ("sub_agent", "run_sub_agent", {"description": "Task", "prompt": "Task"}),
    "parallel_sub_agents": (
        "sub_agent", "run_parallel_sub_agents", {"tasks": [{"description": "Task", "prompt": "Task"}]},
    ),
    "llm_review": ("llm_review", "llm_review", {"topic": "Write the draft"}),
    "magi_decision_support": (
        "magi_decision_support", "magi_decide", {"proposition": "Choose", "option_a": "A", "option_b": "B"},
    ),
}


@pytest.mark.parametrize("entry", list(CURRENT_MODEL_CALLS))
@pytest.mark.parametrize("chat", ["single_model", "arena", "second_of_two_models"])
async def test_agents_without_a_default_model_run_on_the_model_the_response_is_sent_to(monkeypatch, entry, chat):
    import random
    from open_webui.utils import chat as core_chat
    from open_webui.utils import middleware
    from open_webui.utils import models as core_models

    keep_core_config(monkeypatch, {"task.model.external": "task"})
    sent_to = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    other = {"id": "other", "owned_by": "openai", "info": {"meta": {}}}
    arena = {"id": "arena", "owned_by": "arena", "info": {"meta": {"model_ids": ["other", "chat"]}}}
    if chat == "arena":
        # Core picks "chat" for the response; any later pick from the Arena would be "other".
        monkeypatch.setattr(middleware, "random", SimpleNamespace(choice=lambda ids: ids[-1]))
        monkeypatch.setattr(random, "choice", lambda ids: ids[0])
    first = {"single_model": sent_to, "arena": arena, "second_of_two_models": other}[chat]
    module_name, method, arguments = CURRENT_MODEL_CALLS[entry]
    source = Path(importlib.import_module(module_name).__file__).read_text()
    request = parallel_request(sent_to, other, arena, {"id": "task", "owned_by": "openai", "info": {"meta": {}}})
    monkeypatch.setattr(core_models, "get_all_models", AsyncMock(return_value=list(request.app.state.MODELS.values())))
    metadata = await core_chat_tools(
        monkeypatch, request, make_user("admin"), first, sent_to=arena if chat == "arena" else sent_to,
        workspace={module_name: source}, chat_id="chat-1",
    )
    requested = []
    generate = core_chat.generate_chat_completion

    async def generate_chat_completion(request, form_data, *args, **kwargs):
        requested.append(form_data["model"])
        return await generate(request, form_data, *args, **kwargs)

    async def provider(*, form_data, **_):
        return answer({"tools": tool_names(form_data)})

    monkeypatch.setattr(core_chat, "generate_chat_completion", generate_chat_completion)
    monkeypatch.setattr(core_chat, "generate_openai_chat_completion", provider)

    await metadata["tools"][method]["callable"](**arguments)

    assert requested and set(requested) == {"chat"}


@needs_create_tool
async def test_parallel_tools_run_cores_tool_creation_without_a_valve_of_their_own(monkeypatch):
    chat_model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    metadata = await core_chat_tools(monkeypatch, parallel_request(chat_model), make_user("admin"), chat_model)

    [result] = await call_parallel(metadata, ("create_tool", {
        "id": "not an id", "name": "Draft", "description": "Draft.", "content": "class Tools:\n    pass\n",
    }))

    assert "alphanumeric" in result["error"]


@pytest.mark.parametrize("withheld", ["no_session", "legacy", "builtin_tools_off"])
async def test_parallel_tools_never_run_builtins_core_withheld_from_the_chat(monkeypatch, withheld):
    capabilities = {"builtin_tools": withheld != "builtin_tools_off"}
    chat_model = {"id": "chat", "owned_by": "openai", "info": {"meta": {"capabilities": capabilities}}}
    fields = {"no_session": {"session_id": None}, "legacy": {"params": {"function_calling": "legacy"}}}
    metadata = await core_chat_tools(
        monkeypatch, parallel_request(chat_model), make_user("admin"), chat_model, **fields.get(withheld, {}),
    )
    assert "get_current_timestamp" not in metadata["tools"]

    [result] = await call_parallel(metadata, ("get_current_timestamp", {}))

    assert "'get_current_timestamp' not found" in result


@pytest.mark.parametrize("picked_time, task_time", [(True, False), (False, True)])
async def test_parallel_tools_run_the_builtins_core_gave_an_arena_chat(monkeypatch, picked_time, task_time):
    keep_core_config(monkeypatch, {"task.model.external": "task"})
    arena = {"id": "arena", "owned_by": "arena", "info": {"meta": {"model_ids": ["picked"]}}}
    picked = {"id": "picked", "owned_by": "openai", "info": {"meta": {"builtinTools": {"time": picked_time}}}}
    task = {"id": "task", "owned_by": "openai", "info": {"meta": {"builtinTools": {"time": task_time}}}}
    metadata = await core_chat_tools(monkeypatch, parallel_request(arena, picked, task), make_user("admin"), arena)
    # Nothing a tool receives names the model Core picked.
    assert parallel_tools_model(metadata) == "task" and "selected_model_id" not in metadata
    assert ("get_current_timestamp" in metadata["tools"]) is picked_time

    [result] = await call_parallel(metadata, ("get_current_timestamp", {}))

    assert ("'get_current_timestamp' not found" not in str(result)) is picked_time


async def test_parallel_tools_run_the_chats_tool_where_a_withheld_terminal_tool_shares_its_name(monkeypatch):
    from open_webui.utils import middleware, terminals
    from open_webui.utils import tools as core_tools

    keep_core_config(monkeypatch, {"terminal_server.connections": [{"id": "term", "url": "https://terminal.invalid"}]})
    terminal_commands = []

    async def get_terminal_tools(*args, **kwargs):
        async def run_command(command: str = ""):
            terminal_commands.append(command)
            return "terminal:" + command

        return {"run_command": {
            "type": "terminal", "tool_id": "terminal:term", "callable": run_command,
            "spec": {"name": "run_command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
        }}

    monkeypatch.setattr(core_tools, "get_terminal_tools", get_terminal_tools, raising=False)
    monkeypatch.setattr(middleware, "get_terminal_tools", get_terminal_tools, raising=False)
    # Core lists the terminal's skills from the server before loading tools.
    monkeypatch.setattr(terminals, "get_terminal_json", AsyncMock(return_value=None), raising=False)
    monkeypatch.setattr(terminals, "get_terminal_request_info", AsyncMock(return_value=None), raising=False)
    arena = {"id": "arena", "owned_by": "arena", "info": {"meta": {"model_ids": ["picked"]}}}
    picked = {"id": "picked", "owned_by": "openai", "info": {"meta": {"capabilities": {"terminal": False}}}}
    metadata = await core_chat_tools(
        monkeypatch, parallel_request(arena, picked), make_user("admin"), arena,
        workspace={"workspace": workspace_tool("run_command", "workspace:")},
        terminal_id="term", form={"terminal_id": "term"},
    )
    assert metadata["tools"]["run_command"]["tool_id"] == "workspace"

    assert await call_parallel(metadata, ("run_command", {"command": "pwd"})) == ["workspace:pwd"]
    assert terminal_commands == []


async def test_parallel_tools_keep_the_names_core_gave_colliding_tools(monkeypatch):
    chat_model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    metadata = await core_chat_tools(
        monkeypatch, parallel_request(chat_model), make_user("admin"), chat_model, workspace={
            "workspace": workspace_tool("run_command", "workspace:"), "second": workspace_tool("run_command", "second:"),
        }, valves={"EXCLUDED_TOOL_IDS": "workspace"},
    )
    assert metadata["tools"]["second_run_command"]["tool_id"] == "second"

    second, excluded = await call_parallel(
        metadata, ("second_run_command", {"command": "pwd"}), ("run_command", {"command": "pwd"}),
    )

    assert second == "second:pwd"
    assert "'run_command' not found" in excluded


async def test_parallel_tools_available_ids_only_narrow_the_chats_tools(monkeypatch):
    chat_model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    metadata = await core_chat_tools(
        monkeypatch, parallel_request(chat_model), make_user("admin"), chat_model, workspace={
            "workspace": workspace_tool("run_command", "workspace:"), "other": workspace_tool("other_command", "other:"),
        }, unattached={"hidden": workspace_tool("hidden_command", "hidden:")},
        valves={"AVAILABLE_TOOL_IDS": "workspace,hidden"},
    )

    allowed, other, hidden = await call_parallel(
        metadata, ("run_command", {"command": "pwd"}), ("other_command", {}), ("hidden_command", {}),
    )

    assert allowed == "workspace:pwd"
    assert "'other_command' not found" in other
    assert "'hidden_command' not found" in hidden


@pytest.mark.parametrize("server_id", ["docs", "suite:docs"])
async def test_parallel_tools_exclude_an_mcp_server_core_connected(monkeypatch, page_tools, server_id):
    from open_webui.utils import middleware
    from open_webui.utils import tools as core_tools

    monkeypatch.setattr(middleware, "connect_mcp_server", core_tools.connect_mcp_server)
    chat_model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    metadata = await core_chat_tools(
        monkeypatch, parallel_request(chat_model), make_user("admin"), chat_model,
        tool_ids=[f"server:mcp:{server_id}"], valves={"EXCLUDED_TOOL_IDS": f"server:mcp:{server_id}"},
    )
    name = f"{server_id}_fetch_page"
    assert name in metadata["tools"]

    [result] = await call_parallel(metadata, (name, {"path": "guide"}))

    assert f"'{name}' not found" in result
    assert page_tools.received == []


async def test_parallel_tools_run_no_tool_the_chat_was_not_given(monkeypatch):
    chat_model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    user = make_user("admin")
    await save_workspace_tools(monkeypatch, user, {
        "parallel": PARALLEL_TOOLS_SOURCE, "workspace": workspace_tool("run_command", "workspace:"),
    })
    metadata = {
        "model": chat_model, "model_id": "chat", "chat_id": "", "message_id": "reply", "session_id": "browser",
        "tool_ids": ["parallel", "workspace"], "files": None, "features": {}, "params": {}, "tool_servers": [],
    }

    raw = await parallel_tools.Tools().run_tools_parallel(
        [parallel_tools.ToolCallItem(name="run_command", args={"command": "pwd"})],
        __request__=parallel_request(chat_model), __user__=user.model_dump(), __model__=chat_model,
        __metadata__=metadata, __id__="parallel",
    )

    assert "'run_command' not found" in json.loads(raw)["results"][0]["result"]


def parallel_tools_for(extra_params):
    """Parallel Tools as an agent loads it, bound to the agent's parameters."""
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    async def bound():
        return {"run_tools_parallel": {
            "tool_id": "parallel_tools",
            "callable": await get_async_tool_function_and_apply_extra_params(
                parallel_tools.Tools().run_tools_parallel, extra_params,
            ),
            "spec": {"name": "run_tools_parallel", "parameters": {
                "type": "object", "properties": {"tool_calls": {"type": "array"}},
            }},
        }}

    return bound()


def parallel_results(sent):
    return [
        json.loads(message["content"])["results"][0]["result"] for body in sent for message in body["messages"]
        if message["role"] == "tool" and message["tool_call_id"] == "run_tools_parallel"
    ]


@pytest.mark.parametrize("entry", ENTRIES)
async def test_parallel_tools_in_a_child_keep_the_tools_its_valves_turned_off(monkeypatch, entry):
    sent, _, _ = await run_entry(
        monkeypatch, entry, calls_in_order(
            ("run_tools_parallel", {"tool_calls": [{"name": "view_note", "args": {"note_id": "unread"}}]}),
        ),
        extra_models={"child": child_model()}, settings={"notes.enable": True},
        valves={"ENABLE_NOTES_TOOLS": False}, bound_tools=parallel_tools_for,
    )

    results = parallel_results(sent)
    assert results and all("'view_note' not found" in result for result in results)


@pytest.mark.parametrize("submission", ["submit_draft", "submit_review"])
async def test_parallel_tools_cannot_submit_for_a_review_agent(monkeypatch, submission):
    sent, _, _ = await run_entry(
        monkeypatch, "llm_review", calls_in_order(
            ("run_tools_parallel", {"tool_calls": [{"name": submission, "args": {"draft": "Parallel draft"}}]}),
        ),
        bound_tools=parallel_tools_for,
    )

    results = parallel_results(sent)
    assert results and all(f"'{submission}' not found" in result for result in results)


@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_childs_model_changes_stay_in_its_own_loop(monkeypatch, entry):
    from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

    child = copy.deepcopy(ROUTED_CHILD)
    child_before = copy.deepcopy(child)
    parent_metadata = {
        "model_id": "parent", "model": {"id": "parent", "owned_by": "arena", "info": {"meta": {}}},
        "files": None, "tool_ids": ["regular"], "features": {}, "chat_id": "local:entry",
        "session_id": "browser", "tool_servers": [],
    }
    parent_before = copy.deepcopy(parent_metadata)
    seen = []

    async def bound_tools(extra_params):
        async def lookup(query: str = "", __metadata__=None):
            seen.append(copy.deepcopy(__metadata__["model"]))
            __metadata__["model"]["info"]["meta"]["changed"] = True
            return "found"

        tools = loop_tools()
        tools["lookup"]["callable"] = await get_async_tool_function_and_apply_extra_params(lookup, extra_params)
        return tools

    await run_entry(
        monkeypatch, entry, lookup_first, bound_tools=bound_tools, parent_metadata=parent_metadata,
        extra_models={"child": child},
    )

    assert seen and all(model == child_before for model in seen)
    assert child == child_before
    assert parent_metadata == parent_before


@pytest.mark.parametrize("valves", [None, {"STATUS_TRUNCATE_CHARS": 0}])
@pytest.mark.parametrize("entry", ENTRIES)
async def test_step_statuses_are_capped_while_the_model_still_sees_everything(monkeypatch, entry, valves):
    import unicodedata

    long_text = "あ" * 5000
    statuses = []

    async def emit(event):
        if event.get("type") == "status":
            statuses.append(event["data"]["description"])

    def tools():
        async def lookup(query: str = ""):
            return long_text

        tools = loop_tools()
        tools["lookup"]["callable"] = lookup
        return tools

    def respond(body):
        if not any(message["role"] == "tool" for message in body["messages"]) and "lookup" in body["tools"]:
            response = tool_call("lookup", {"query": long_text})
            response["choices"][0]["message"]["content"] = long_text
            return response
        return answer(body)

    sent, _, _ = await run_entry(monkeypatch, entry, respond, regular_tools=tools, event_emitter=emit, valves=valves)

    assert any(
        message["role"] == "tool" and message["content"] == long_text
        for body in sent for message in body["messages"]
    )
    assert any("lookup" in status for status in statuses)
    if valves is None:
        widths = [sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s) for s in statuses]
        assert max(widths) <= 130
    else:
        assert any(long_text in status for status in statuses)
