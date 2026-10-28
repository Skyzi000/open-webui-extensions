"""Nested agent loops follow real Core for request state, prompts, Tool Search, skills and embeds."""

import asyncio
import collections
import copy
import importlib
import importlib.util
import json
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
    monkeypatch.setattr(parallel_tools, "build_tools_dict", AsyncMock(return_value=(loop_tools(), {})))

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
    monkeypatch.setattr(parallel_tools, "build_tools_dict", AsyncMock(return_value=(loop_tools(), {})))

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
        return await tool.run_parallel_sub_agents(tasks=[{"description": "Task", "prompt": "Task"}], **kwargs)
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
    extra_models=None, event_call=None, direct=False, user_valves=None,
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
        await call_entry(
            entry, valves, __request__=request, __model__=parent,
            __user__={**make_user("admin").model_dump(), "valves": user_valves or {}},
            __event_call__=event_call or AsyncMock(),
            __metadata__=parent_metadata if parent_metadata is not None else {
                "model_id": "parent", "files": None, "tool_ids": ["regular"], "features": {"voice": True},
                "chat_id": "local:entry", "session_id": "browser", "tool_servers": [],
            },
            __messages__=[
                {"role": "system", "content": f"Be brief.\n\n{skill_context}"}, {"role": "user", "content": "Hi"},
            ],
        )
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
        assert any(request is loaded for loaded in loaded_requests)
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
    from open_webui.models.config import Config
    from open_webui.utils import tools as core_tools

    tool_id, name, expected = page_tools.kinds[kind]
    monkeypatch.setattr(Config, "get", AsyncMock(side_effect=lambda key, default=None: {} if key == "user.permissions" else default))
    monkeypatch.setattr(Config, "get_many", AsyncMock(return_value={}))
    monkeypatch.setattr(core_tools, "get_tools", AsyncMock(return_value=page_tools.regular))
    model = {"id": "model", "info": {"meta": {"builtinTools": {
        category: False for category in ("time", "user_input", "knowledge", "chats", "tasks", "skills")
    }}}}

    await parallel_tools.Tools().run_tools_parallel(
        [parallel_tools.ToolCallItem(name=name, args={"path": "guide"})],
        __request__=make_request(model), __user__=make_user("admin").model_dump(), __model__=model,
        __metadata__={"model_id": "model", "tool_ids": [tool_id], "chat_id": "local:parallel", "tool_servers": []},
    )

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
