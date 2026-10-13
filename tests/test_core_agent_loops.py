"""Nested agent loops follow real Core for request state, prompts, Tool Search, skills and embeds."""

import asyncio
import copy
import importlib
import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from open_webui.tools import builtin

import llm_review
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
        message["content"] for message in form_data["messages"] if message["role"] == "system"
    )


def tool_names(form_data):
    return {tool["function"]["name"] for tool in form_data.get("tools") or []}


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

    isolated = sub_agent.isolate_request_state(request)
    isolated.state.direct = True
    isolated.state.model = {"id": "child"}

    assert request.state.direct is False
    assert not hasattr(request.state, "model")
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

    assert tool_names(sent[0]) == {"ping", "search_tools", "submit_draft"}
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
            extra_params={"__request__": request, "__user__": user, "__metadata__": metadata},
            tool_id_list=[], excluded_tool_ids=set(), resolved_terminal_id=None,
            resolved_direct_tool_servers=[],
        )
        return tools, request, user

    return load


@needs_core_skill_tools
@pytest.mark.parametrize("module_name", AGENT_MODULES)
@pytest.mark.parametrize("read, write, expected", [
    (True, None, {"view_skill", "read_skill_file"}),
    (True, True, {"view_skill", "read_skill_file", "update_skill_files", "create_skill"}),
    (False, True, set()),
])
async def test_skill_write_tools_need_their_own_valve(skill_tools, module_name, read, write, expected):
    module = importlib.import_module(module_name)
    valves = module.Tools.Valves(ENABLE_SKILLS_TOOLS=read)
    if write is not None:
        valves.ENABLE_SKILLS_WRITE_TOOLS = write

    tools, _, _ = await skill_tools(module, valves)

    skill_names = {"view_skill", "read_skill_file", "update_skill_files", "create_skill"}
    assert skill_names & set(tools) == expected


@pytest.mark.parametrize("module_name", AGENT_MODULES)
@pytest.mark.parametrize("loadable", [True, False])
async def test_skill_manifest_is_passed_only_with_a_tool_that_loads_it(monkeypatch, module_name, loadable):
    tools = loop_tools()
    if loadable:
        tools["view_skill"] = {**tools["ping"], "spec": {**tools["ping"]["spec"], "name": "view_skill"}}
    manifest = "<available_skills>\n<skill>\n<name>Writer</name>\n</skill>\n</available_skills>"
    inline = '<skill name="Editor">\nEdit well.\n</skill>'

    _, sent, _ = await run_loop(
        monkeypatch, module_name, [{"choices": [{"message": {"content": "done"}}]}], tools=tools,
        messages=[
            {"role": "system", "content": f"Be brief.\n\n{inline}\n\n{manifest}"},
            {"role": "user", "content": "Hi"},
        ],
    )

    assert sent[0]["messages"][0]["content"] == f"Be brief.\n\n{inline}" + f"\n\n{manifest}" * loadable


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

    result = await sub_agent.execute_tool_call(
        {"id": "call", "function": {"name": "view_skill", "arguments": '{"id": "skill"}'}}, tools,
        {"__request__": request, "__user__": user, "__metadata__": first},
    )

    assert json.loads(result["content"])["version_id"] == "version-1"
    assert first["chat_context"]["skill_versions"] == {"skill": "version-1"}
    assert "skill_versions" not in second["chat_context"]
    assert "skill_versions" not in (parent.get("chat_context") or {})


@pytest.mark.parametrize("tag", [
    '<skill name="Writer">\nWrite well.\n</skill>',
    '<skill id="writer" version_id="version-1" name="Writer">\nWrite well.\n</skill>',
])
def test_inline_skill_tags_are_inherited_for_both_core_shapes(tag):
    messages = [{"role": "system", "content": f"Be brief.\n\n{tag}"}, {"role": "user", "content": "Hi"}]

    assert sub_agent.extract_user_skill_tags(messages) == [tag]
