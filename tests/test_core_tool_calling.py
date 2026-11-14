"""Agent tools follow Core 0.13.1+ for hidden skills, call_tool batches, call handling and tool approvals."""

import asyncio
import copy
import importlib.util
import json
from unittest.mock import AsyncMock

import pytest
from open_webui.utils import middleware

import test_core_agent_loops as h

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("open_webui.utils.tool_calling") is None,
    reason="Core before 0.13.1 has none of these tool-calling contracts",
)



def core_replays_calls_as_written():
    try:
        from open_webui.utils.misc import get_provider_function
    except ImportError:
        return False
    written = '{"query":"key"}'
    return get_provider_function({"name": "lookup", "arguments": written})["arguments"] == written


def core_gives_hidden_calls_the_reasoning_field():
    try:
        from open_webui.utils.misc import convert_output_to_messages
    except ImportError:
        return False
    if not hasattr(middleware, "hidden_tool_result"):
        return False
    output = middleware.hidden_tool_result("list_available_skills", "skills")
    return convert_output_to_messages(output, raw=True, reasoning_format="reasoning_content")[0].get("reasoning_content") == ""


CORE_REPLAYS_CALLS_AS_WRITTEN = core_replays_calls_as_written()
needs_calls_replayed_as_written = pytest.mark.skipif(
    not CORE_REPLAYS_CALLS_AS_WRITTEN, reason="Core before 0.13.3 replays a call's arguments normalized"
)
needs_reasoning_field_on_hidden_calls = pytest.mark.skipif(
    not core_gives_hidden_calls_the_reasoning_field(),
    reason="Core before 0.13.3 gives hidden tool calls no reasoning_content",
)
APPROVAL_SETTINGS = {
    "chat.tool_permissions.enable": True,
    "chat.tool_permissions.auto.enable": True,
    "chat.tool_permissions.auto.engine": "",
}
NOT_APPROVED = "Error: tool call was not approved."
ASKING_TOOL = """class Tools:
    async def lookup(self, command: str = '', __event_call__=None) -> str:
        await __event_call__({"type": "input", "data": {"title": "Which one?"}})
        return "found:" + command
"""


def approval_metadata(mode):
    return {**h.ENTRY_METADATA, "params": {"function_calling": "native", "tool_approval_mode": mode}}


def recording_tools(executed, result="found"):
    def tools():
        loaded = h.loop_tools()

        async def lookup(query=""):
            executed.append(query)
            return result

        loaded["lookup"]["callable"] = lookup
        return loaded

    return tools


def browser(approve=True):
    """A browser that answers every tool approval question with *approve*; returns its questions too."""
    asked = []

    async def event_call(event):
        if event.get("type") == "confirmation":
            asked.append(event["data"]["tool_call"])
            return approve
        return None

    return asked, event_call


async def bind_browser(metadata, event_call):
    """Give the Parallel Tools callable Core registered *event_call* as its browser, the way Core binds it."""
    from open_webui.utils.tools import get_updated_tool_function

    tool = metadata["tools"]["run_tools_parallel"]
    tool["callable"] = await get_updated_tool_function(tool["callable"], {"__event_call__": event_call})


def tool_results(sent):
    return [message["content"] for body in sent for message in body["messages"] if message["role"] == "tool"]


async def core_parent_with_skill(monkeypatch, skill_id, prompt="Go"):
    """The messages and metadata Core gives the tools of a chat that activated *skill_id* on a model with skill tools."""
    captured = {}
    process = middleware.process_chat_payload

    async def capture(*args, **kwargs):
        result = await process(*args, **kwargs)
        captured["messages"] = copy.deepcopy(result[0]["messages"])
        return result

    monkeypatch.setattr(middleware, "process_chat_payload", capture)
    model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    metadata = await h.core_chat_tools(
        monkeypatch, h.parallel_request(model), h.make_user("admin"), model,
        form={"skill_ids": [skill_id], "messages": [{"role": "user", "content": prompt}]},
    )
    return captured["messages"], metadata


@pytest.mark.parametrize("entry", h.ENTRIES)
@pytest.mark.parametrize("child_skill_tools", [True, False])
async def test_children_get_the_skills_core_gave_the_chat_as_tool_history(monkeypatch, entry, child_skill_tools):
    from open_webui.models.skills import SkillForm, Skills

    h.keep_core_config(monkeypatch, {"skills.lazy_loading.enable": False})
    await Skills.insert_new_skill("loop-user", SkillForm(id="hidden-writer", name="Writer", content="HIDDEN_SKILL_BODY"))
    try:
        messages, parent = await core_parent_with_skill(monkeypatch, "hidden-writer")
        assert not any(message["role"] == "system" for message in messages)
        assert "HIDDEN_SKILL_BODY" in str(messages)
        if child_skill_tools:
            sent, _, _ = await h.run_entry(
                monkeypatch, entry, h.answer,
                parent_metadata={
                    **h.ENTRY_METADATA, "skill_ids": parent["skill_ids"], "chat_context": parent["chat_context"],
                },
                entry_kwargs={"__messages__": messages},
            )
            bodies = [body for body in sent if body["tools"]]
        else:
            bodies, _ = await h.run_without_skill_tools(
                monkeypatch, entry, parent["skill_ids"], "", metadata={"chat_context": parent["chat_context"]},
                entry_kwargs={"__messages__": messages},
            )
    finally:
        await Skills.delete_skill_by_id("hidden-writer")

    assert bodies
    for body in bodies:
        results = [message["content"] for message in body["messages"] if message["role"] == "tool"]
        if child_skill_tools:
            assert h.SKILL_READ_TOOLS <= body["tools"]
            assert any(result.startswith("<available_skills>") and "hidden-writer" in result for result in results)
            assert any("HIDDEN_SKILL_BODY" in result for result in results)
            assert "HIDDEN_SKILL_BODY" not in body["system"]
        else:
            assert "HIDDEN_SKILL_BODY" in body["system"]


@needs_reasoning_field_on_hidden_calls
@pytest.mark.parametrize("entry", h.ENTRIES)
@pytest.mark.parametrize("provider", ["", "llama.cpp"])
async def test_hidden_skill_calls_carry_the_reasoning_field_a_llama_cpp_child_needs(monkeypatch, entry, provider):
    from open_webui.models.skills import SkillForm, Skills

    h.keep_core_config(monkeypatch, {"skills.lazy_loading.enable": False})
    await Skills.insert_new_skill("loop-user", SkillForm(id="hidden-writer", name="Writer", content="HIDDEN_SKILL_BODY"))
    child = {"id": "child", "owned_by": "openai", "provider": provider, "info": {"meta": {}, "params": {}}}
    try:
        messages, parent = await core_parent_with_skill(monkeypatch, "hidden-writer")
        sent, _, _ = await h.run_entry(
            monkeypatch, entry, h.answer,
            parent_metadata={
                **h.ENTRY_METADATA, "skill_ids": parent["skill_ids"], "chat_context": parent["chat_context"],
            },
            entry_kwargs={"__messages__": messages}, extra_models={"child": child},
        )
    finally:
        await Skills.delete_skill_by_id("hidden-writer")

    hidden_calls = [
        message for body in sent if body["tools"] for message in body["messages"] if message.get("tool_calls")
    ]
    assert hidden_calls
    assert all((message.get("reasoning_content") == "") == (provider == "llama.cpp") for message in hidden_calls)


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_children_read_the_version_of_a_skill_core_loaded_for_them(monkeypatch, entry):
    from open_webui.models.skills import SkillForm, Skills

    h.keep_core_config(monkeypatch, {"skills.lazy_loading.enable": False})
    await Skills.insert_new_skill("loop-user", SkillForm(id="versioned-writer", name="Writer", content="OLD_BODY"))
    try:
        messages, parent = await core_parent_with_skill(monkeypatch, "versioned-writer")
        await Skills.update_skill_by_id("versioned-writer", {"content": "NEW_BODY"})
        sent, _, _ = await h.run_entry(
            monkeypatch, entry, h.calling(("read_skill_file", {"id": "versioned-writer", "path": "SKILL.md"})),
            parent_metadata={
                **h.ENTRY_METADATA, "skill_ids": parent["skill_ids"], "chat_context": parent["chat_context"],
            },
            entry_kwargs={"__messages__": messages},
        )
    finally:
        await Skills.delete_skill_by_id("versioned-writer")

    followups = [body for body in sent if h.called(body, "read_skill_file")]
    assert followups
    for body in followups:
        *preloaded, read = [message["content"] for message in body["messages"] if message["role"] == "tool"]
        assert any("NEW_BODY" in result for result in preloaded)
        assert "NEW_BODY" in read and "OLD_BODY" not in read


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_with_lazy_skill_loading_children_load_only_the_skills_the_chat_mentioned(monkeypatch, entry):
    from open_webui.models.skills import SkillForm, Skills

    h.keep_core_config(monkeypatch, {"skills.lazy_loading.enable": True})
    for skill_id, body in (("lazy-writer", "WRITER_BODY"), ("lazy-editor", "EDITOR_BODY")):
        await Skills.insert_new_skill("loop-user", SkillForm(id=skill_id, name=skill_id, content=body))
    try:
        messages, parent = await core_parent_with_skill(monkeypatch, "lazy-writer", "Edit with <$lazy-editor|Editor>")
        assert "EDITOR_BODY" in str(messages) and "WRITER_BODY" not in str(messages)
        sent, _, _ = await h.run_entry(
            monkeypatch, entry, h.answer, settings={"skills.lazy_loading.enable": True},
            parent_metadata={
                **h.ENTRY_METADATA, "skill_ids": parent["skill_ids"], "chat_context": parent["chat_context"],
            },
            entry_kwargs={"__messages__": messages},
        )
    finally:
        for skill_id in ("lazy-writer", "lazy-editor"):
            await Skills.delete_skill_by_id(skill_id)

    bodies = [body for body in sent if body["tools"]]
    assert bodies
    for body in bodies:
        results = [message["content"] for message in body["messages"] if message["role"] == "tool"]
        assert any("lazy-writer" in result and "lazy-editor" in result for result in results)
        assert any("EDITOR_BODY" in result for result in results)
        assert not any("WRITER_BODY" in result for result in results)


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_a_call_tool_batch_runs_each_tool_and_answers_with_cores_combined_result(monkeypatch, entry):
    batch = {"calls": [h.LOOKUP_CALL, {"tool_name": "ping", "params": "{}"}, {"tool_name": "missing", "params": "{}"}]}

    sent, _, _ = await h.run_entry(
        monkeypatch, entry, h.calling(("call_tool", batch)), regular_tools=h.loop_tools, settings=h.TOOL_SEARCH_SETTINGS,
    )

    followups = [body for body in sent if h.called(body, "call_tool")]
    assert followups
    for body in followups:
        calls = [call["function"] for message in body["messages"] for call in message.get("tool_calls") or []]
        assert calls[0] == {"name": "call_tool", "arguments": json.dumps(batch)}
        assert json.loads(next(message["content"] for message in body["messages"] if message["role"] == "tool")) == {
            "results": [
                {"tool_name": "lookup", "status": "completed", "output": [{"type": "input_text", "text": "found"}]},
                {"tool_name": "ping", "status": "completed", "output": [{"type": "input_text", "text": "pong"}]},
                {"tool_name": "missing", "status": "failed", "output": [{"type": "input_text", "text": (
                    'Error: Tool "missing" is not available. Search for an available tool.'
                )}]},
            ],
        }


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_tool_search_by_embeddings_gets_the_request_and_user(monkeypatch, entry):
    from open_webui.utils import tool_search

    seen = []

    async def search(query, candidates, count, request, user):
        seen.append((request, user))
        return ["lookup"]

    monkeypatch.setattr(tool_search, "search_tools_with_embeddings", search)
    sent, _, _ = await h.run_entry(
        monkeypatch, entry, h.calling(("search_tools", {"query": "find a record"})), regular_tools=h.loop_tools,
        settings={**h.TOOL_SEARCH_SETTINGS, "chat.tool_search.engine": "embeddings"},
    )

    assert seen and all(request is not None and user.id == "loop-user" for request, user in seen)
    assert any('"name": "lookup"' in result or '"name":"lookup"' in result for result in tool_results(sent))


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_a_personal_tether_terminal_reaches_the_children(monkeypatch, entry):
    from open_webui.utils import terminals, tether, tools

    terminal_id = "tether:device"
    monkeypatch.setattr(tether, "get_terminal_connections", AsyncMock(return_value=[{"id": terminal_id}]))
    monkeypatch.setattr(terminals, "get_terminal_agents_md", AsyncMock(return_value=None))
    loader = AsyncMock(return_value={})
    monkeypatch.setattr(tools, "get_terminal_tools", loader)

    await h.run_entry(
        monkeypatch, entry, h.answer, settings={"terminal_server.connections": []},
        parent_metadata={**h.ENTRY_METADATA, "terminal_id": terminal_id}, valves={"ENABLE_TERMINAL_TOOLS": True},
    )

    assert loader.await_count and all(call.kwargs["terminal_id"] == terminal_id for call in loader.await_args_list)


@pytest.mark.parametrize("entry", h.ENTRIES)
@pytest.mark.parametrize("specs_missing", [False, True])
async def test_direct_tools_of_children_leave_out_the_arguments_core_added_for_itself(monkeypatch, entry, specs_missing):
    h.keep_core_config(monkeypatch, {"chat.tool_call_labels.enable": True, **APPROVAL_SETTINGS})
    model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    server = {
        "url": "https://direct.invalid", "system_prompt": "Remote tools.",
        "specs": [{"name": "remote_lookup", "description": "Lookup", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
        }}],
    }
    request = h.parallel_request(model)
    request.scope["headers"] = []
    parent = await h.core_chat_tools(
        monkeypatch, request, h.make_user("admin"), model, tool_servers=[server],
        params={"function_calling": "native", "tool_approval_mode": "auto"},
    )
    assert parent["tools"]["remote_lookup"]["display_labels"] and parent["tools"]["remote_lookup"]["approval_review"]
    metadata = {**h.ENTRY_METADATA, "tools": parent["tools"], "tool_servers": copy.deepcopy(parent["tool_servers"])}
    if specs_missing:
        for item in metadata["tool_servers"]:
            item.pop("specs")

    sent, _, _ = await h.run_entry(
        monkeypatch, entry, h.calling(("remote_lookup", {"query": "key"})),
        parent_metadata=metadata, event_call=AsyncMock(return_value="remote-result"),
    )

    specs = [
        tool["function"]["parameters"]
        for body in sent for tool in body["raw_tools"] if tool.get("function", {}).get("name") == "remote_lookup"
    ]
    assert specs and all(spec["properties"] == {"query": {"type": "string"}} for spec in specs)
    assert all(spec["required"] == ["query"] for spec in specs)


@pytest.mark.parametrize("entry", h.ENTRIES)
@pytest.mark.parametrize("kind, reader", [("note", "view_note"), ("collection", "view_file")])
async def test_children_keep_the_reader_for_knowledge_attached_to_a_chat_core_does_not_preload(
    monkeypatch, entry, kind, reader,
):
    from open_webui import env

    monkeypatch.setattr(env, "ENABLE_KB_EXEC", False)

    sent, _, _ = await h.run_entry(
        monkeypatch, entry, h.answer, extra_models={"child": h.child_model(capabilities={})},
        parent_metadata={
            **h.ENTRY_METADATA, "files": [{"type": kind, "id": "attached", "name": "Attached"}],
            "params": {"function_calling": "native"},
        },
        valves={"ENABLE_NOTES_TOOLS": False, "ENABLE_FILE_TOOLS": False, "ENABLE_KNOWLEDGE_TOOLS": True},
    )

    assert sent and reader in sent[0]["tools"]


async def test_parallel_tools_runs_the_callers_tools_without_the_arguments_core_added_for_itself(monkeypatch):
    h.keep_core_config(monkeypatch, {"chat.tool_call_labels.enable": True, **APPROVAL_SETTINGS})
    model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    metadata = await h.core_chat_tools(
        monkeypatch, h.parallel_request(model), h.make_user("admin"), model,
        workspace={"record": h.workspace_tool("lookup", "found:")},
        params={"function_calling": "native", "tool_approval_mode": "auto"},
    )
    assert metadata["tools"]["lookup"]["display_labels"] and metadata["tools"]["lookup"]["approval_review"]

    results = await h.call_parallel(metadata, ("lookup", {
        "command": "key", "__labels__": {"running": "Looking", "completed": "Looked"}, "__approval__": {"required": True},
    }))

    assert results == ["found:key"]


@pytest.mark.parametrize("name", ["call_tool", "functions.call_tool"])
async def test_parallel_tools_runs_a_call_tool_batch_as_one_call_per_tool(monkeypatch, name):
    h.keep_core_config(monkeypatch, {**h.TOOL_SEARCH_SETTINGS, "chat.tool_search.defer_threshold": 1})
    model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    metadata = await h.core_chat_tools(
        monkeypatch, h.parallel_request(model), h.make_user("admin"), model,
        workspace={"record": h.workspace_tool("lookup", "found:")},
    )

    results = await h.call_parallel(
        metadata,
        (name, {"calls": [
            {"tool_name": "lookup", "params": '{"command": "a"}'}, {"tool_name": "lookup", "params": '{"command": "b"}'},
        ]}),
        ("call_tool", {"calls": []}),
    )

    assert results == ["found:a", "found:b", "Error: call_tool requires a non-empty calls array."]


async def test_parallel_tools_reads_arguments_written_as_text_the_way_core_does(monkeypatch):
    h.keep_core_config(monkeypatch, {**h.TOOL_SEARCH_SETTINGS, "chat.tool_search.defer_threshold": 1})
    model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    metadata = await h.core_chat_tools(
        monkeypatch, h.parallel_request(model), h.make_user("admin"), model,
        workspace={"record": h.workspace_tool("lookup", "found:")},
    )
    single_call = json.dumps({"tool_name": "lookup", "params": '{"command": "a"}'})

    results = await h.call_parallel(
        metadata,
        ("lookup", '```json\n{"command": "b"}\n```'),
        ("call_tool", f"```json\n{single_call}\n```"),
        ("lookup", '{{"command": "c"}}'),
    )

    assert results == [
        "found:b", "found:a",
        "Error: Tool arguments must be a complete JSON object. Correct the arguments before retrying.",
    ]


async def test_parallel_tools_runs_what_core_approved_with_its_call_without_asking_again(monkeypatch):
    h.keep_core_config(monkeypatch, APPROVAL_SETTINGS)
    model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    asked, event_call = browser(False)
    metadata = await h.core_chat_tools(
        monkeypatch, h.parallel_request(model), h.make_user("admin"), model,
        workspace={"record": h.workspace_tool("lookup", "ran:")},
        params={"function_calling": "native", "tool_approval_mode": "ask"},
    )
    await bind_browser(metadata, event_call)

    assert await h.call_parallel(metadata, ("lookup", {"command": "key"})) == ["ran:key"]
    assert asked == []


@pytest.mark.parametrize("entry", h.ENTRIES)
@pytest.mark.parametrize("approve", [True, False])
async def test_ask_mode_asks_before_each_call_the_children_make(monkeypatch, entry, approve):
    executed = []
    asked, event_call = browser(approve)

    sent, _, _ = await h.run_entry(
        monkeypatch, entry, h.calling(("lookup", {"query": "key"})), regular_tools=recording_tools(executed),
        settings=APPROVAL_SETTINGS, parent_metadata=approval_metadata("ask"), event_call=event_call,
    )

    assert asked and all(call["name"] == "lookup" and call["arguments"] == {"query": "key"} for call in asked)
    assert executed == (["key"] * len(asked) if approve else [])
    assert (NOT_APPROVED in tool_results(sent)) is not approve


@pytest.mark.parametrize("entry", h.ENTRIES)
@pytest.mark.parametrize("mode", ["ask", "auto"])
async def test_approving_the_agent_tools_call_once_lets_its_children_run_their_calls(monkeypatch, entry, mode):
    executed = []
    asked, event_call = browser(False)

    sent, _, _ = await h.run_entry(
        monkeypatch, entry, h.calling(("lookup", {"query": "key"})), regular_tools=recording_tools(executed),
        settings=APPROVAL_SETTINGS, parent_metadata=approval_metadata(mode), event_call=event_call,
        user_valves={"APPROVE_AGENT_TOOL_CALLS_ONCE": True},
    )

    assert executed and asked == []
    assert not any("__approval__" in json.dumps(body["raw_tools"]) for body in sent)
    assert not any("<tool_approval_policy>" in body["system"] for body in sent)


@pytest.mark.parametrize("entry", h.ENTRIES)
@pytest.mark.parametrize("required", [False, True])
async def test_auto_mode_has_the_child_model_say_which_of_its_calls_need_approval(monkeypatch, entry, required):
    executed = []
    asked, event_call = browser(True)

    sent, _, _ = await h.run_entry(
        monkeypatch, entry, h.calling(("lookup", {"query": "key", "__approval__": {"required": required}})),
        regular_tools=recording_tools(executed), settings=APPROVAL_SETTINGS, parent_metadata=approval_metadata("auto"),
        event_call=event_call,
    )

    agent_requests = [body for body in sent if body["tools"]]
    for body in agent_requests:
        assert "<tool_approval_policy>" in body["system"]
        for tool in body["raw_tools"]:
            function = tool.get("function") or {}
            if function.get("name") in ("lookup", "ping"):
                assert "__approval__" in function["parameters"]["required"]
            elif function.get("name", "").startswith("submit_"):
                assert "__approval__" not in json.dumps(function)
    assert executed and set(executed) == {"key"}
    assert [call["name"] for call in asked] == (["lookup"] * len(executed) if required else [])
    replayed = [
        json.loads(call["function"]["arguments"]) for body in sent for message in body["messages"]
        for call in message.get("tool_calls") or [] if call["function"]["name"] == "lookup"
    ]
    assert replayed and all(("__approval__" in arguments) == CORE_REPLAYS_CALLS_AS_WRITTEN for arguments in replayed)


@pytest.mark.parametrize("entry", ["parallel_sub_agents", "magi_decision_support", "multi_model_council"])
async def test_concurrent_children_ask_one_question_at_a_time(monkeypatch, entry):
    executed, waiting, most = [], [0], [0]

    async def event_call(event):
        if event.get("type") != "confirmation":
            return None
        waiting[0] += 1
        most[0] = max(most[0], waiting[0])
        await asyncio.sleep(0.01)
        waiting[0] -= 1
        return True

    await h.run_entry(
        monkeypatch, entry, h.calling(("lookup", {"query": "key"})), regular_tools=recording_tools(executed),
        settings=APPROVAL_SETTINGS, parent_metadata=approval_metadata("ask"), event_call=event_call,
        entry_kwargs={"tasks": [{"description": "A", "prompt": "A"}, {"description": "B", "prompt": "B"}]}
        if entry == "parallel_sub_agents" else None,
    )

    assert len(executed) >= 2 and most[0] == 1


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_calls_of_a_reply_cut_off_before_its_end_do_not_run_and_tool_use_stops(monkeypatch, entry):
    from open_webui.utils.tool_calling import FINAL_ANSWER_PROMPT, INCOMPLETE_CALL_ERROR

    executed = []

    def respond(body):
        if "lookup" in body["tools"] and not h.called(body, "lookup"):
            response = h.tool_call("lookup", {"query": "key"})
            response["choices"][0]["finish_reason"] = "length"
            return response
        return h.answer(body)

    sent, _, _ = await h.run_entry(monkeypatch, entry, respond, regular_tools=recording_tools(executed))

    assert not executed
    stopped = [body for body in sent if INCOMPLETE_CALL_ERROR in tool_results([body])]
    assert stopped and all(
        FINAL_ANSWER_PROMPT in str(body["messages"]) or "Tool use has stopped for this response." in str(body["messages"])
        for body in stopped
    )


@needs_calls_replayed_as_written
@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_a_call_goes_back_to_the_model_as_the_model_wrote_it(monkeypatch, entry):
    written = '{"query":"k\\u00e9y"}'
    executed = []

    def respond(body):
        if "lookup" in body["tools"] and not h.called(body, "lookup"):
            response = h.tool_call("lookup")
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = written
            return response
        return h.answer(body)

    sent, _, _ = await h.run_entry(monkeypatch, entry, respond, regular_tools=recording_tools(executed))

    replayed = [
        call["function"]["arguments"] for body in sent for message in body["messages"]
        for call in message.get("tool_calls") or [] if call["function"]["name"] == "lookup"
    ]
    assert executed and set(executed) == {"k\u00e9y"}
    assert replayed and set(replayed) == {written}


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_a_call_cut_off_mid_arguments_is_replayed_the_way_core_replays_it(monkeypatch, entry):
    from open_webui.utils.misc import get_provider_function

    cut_off = '{"query": "ke'

    def respond(body):
        if "lookup" in body["tools"] and not h.called(body, "lookup"):
            response = h.tool_call("lookup")
            response["choices"][0]["finish_reason"] = "length"
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = cut_off
            return response
        return h.answer(body)

    sent, _, _ = await h.run_entry(monkeypatch, entry, respond, regular_tools=h.loop_tools)

    replayed = [
        call["function"] for body in sent for message in body["messages"]
        for call in message.get("tool_calls") or [] if call["function"]["name"] == "lookup"
    ]
    assert replayed and all(
        function == get_provider_function({"name": "lookup", "arguments": cut_off}) for function in replayed
    )
    assert all("error" in json.loads(function["arguments"]) for function in replayed)


@pytest.mark.parametrize("entry", h.ENTRIES)
@pytest.mark.parametrize("arguments", ['{{"query": "key"}}', "{[1]: 2}"])
async def test_arguments_core_raises_on_come_back_to_the_model_as_an_error_and_the_loop_goes_on(
    monkeypatch, entry, arguments
):
    """Core 0.13.2 reads arguments as a Python literal without catching the TypeError of an unhashable member."""
    executed = []

    def respond(body):
        if "lookup" not in body["tools"] or h.called(body, "lookup"):
            return h.answer(body)
        response = h.tool_call("lookup")
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
        return response

    sent, _, _ = await h.run_entry(monkeypatch, entry, respond, regular_tools=recording_tools(executed))

    answered = [body for body in sent if h.called(body, "lookup")]
    assert answered and all(
        tool_results([body]) == ["Error: Tool arguments must be a complete JSON object. Correct the arguments before retrying."]
        for body in answered
    )
    assert executed == []


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_a_call_whose_arguments_core_raises_on_can_be_retried_with_other_arguments(monkeypatch, entry):
    executed = []

    def respond(body):
        attempts = len(tool_results([body]))
        if "lookup" not in body["tools"] or attempts > 3:
            return h.answer(body)
        response = h.tool_call("lookup", {"query": "key"})
        if attempts < 3:
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = f'{{{{"query": "{attempts}"}}}}'
        return response

    await h.run_entry(
        monkeypatch, entry, respond, regular_tools=recording_tools(executed), valves={"MAX_ITERATIONS": 6},
    )

    assert set(executed) == {"key"}


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_the_same_failing_call_runs_twice_before_tool_use_stops(monkeypatch, entry):
    from open_webui.utils.tool_calling import tool_repetition_error

    executed = []
    blocked = tool_repetition_error(
        {"name": "lookup", "arguments": "{}"}, [{"name": "lookup", "arguments": "{}", "status": "failed"}] * 2,
    )

    def respond(body):
        if "lookup" in body["tools"] and blocked not in tool_results([body]):
            return h.tool_call("lookup", {"query": "key"})
        return h.answer(body)

    sent, _, _ = await h.run_entry(
        monkeypatch, entry, respond, regular_tools=recording_tools(executed, "Error: the lookup service is down"),
        valves={"MAX_ITERATIONS": 6},
    )

    histories = [tool_results([body]) for body in sent if blocked in tool_results([body])]
    assert histories and all(
        results.count("Error: the lookup service is down") == 2 and results[-1] == blocked for results in histories
    )
    assert set(executed) == {"key"}


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_a_reply_that_is_only_written_tool_calls_runs_them_like_core(monkeypatch, entry):
    executed = []

    def respond(body):
        if "lookup" in body["tools"] and not h.called(body, "lookup"):
            return {"choices": [{"finish_reason": "stop", "message": {
                "content": '<tool_call>{"name": "lookup", "arguments": {"query": "key"}}</tool_call>',
            }}]}
        return h.answer(body)

    sent, _, _ = await h.run_entry(monkeypatch, entry, respond, regular_tools=recording_tools(executed))

    assert executed and set(executed) == {"key"}
    recovered = [
        message for body in sent for message in body["messages"]
        if message["role"] == "assistant" and message.get("tool_calls")
    ]
    assert recovered and all(message["content"] == "" for message in recovered)


async def test_a_review_agent_can_submit_by_writing_the_submission_call(monkeypatch):
    def respond(body):
        if "submit_draft" in body["tools"]:
            return {"choices": [{"finish_reason": "stop", "message": {
                "content": "<function=submit_draft><parameter=draft>A written draft</parameter></function>",
            }}]}
        return h.answer(body)

    sent, _, _ = await h.run_entry(monkeypatch, "llm_review", respond)

    reviews = [
        "\n".join(message["content"] for message in body["messages"] if isinstance(message["content"], str))
        for body in sent if "submit_review" in body["tools"]
    ]
    assert reviews and all("DRAFT:\nA written draft\n" in review for review in reviews)


@pytest.mark.parametrize("kind", ["fence", "double", "scalar", "concatenated"])
async def test_child_tool_arguments_are_normalized_like_core(monkeypatch, kind):
    executed = []

    def tools():
        loaded = recording_tools(executed)()
        loaded["lookup"]["spec"]["parameters"]["properties"]["query"] = {"type": "boolean"}
        return loaded

    def respond(body):
        if h.called(body, "lookup"):
            return h.answer(body)
        raw = '{"query": false}'
        response = h.tool_call("lookup")
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {
            "fence": f"```json\n{raw}\n```", "double": json.dumps(raw), "scalar": '{"query": "false"}',
            "concatenated": raw + raw,
        }[kind]
        return response

    sent, _, _ = await h.run_entry(monkeypatch, "sub_agent", respond, regular_tools=tools)

    assert executed == [False] * (2 if kind == "concatenated" else 1)
    calls = [call["function"]["arguments"] for body in sent[-1:] for message in body["messages"]
             for call in message.get("tool_calls") or []]
    as_written = kind == "scalar" and CORE_REPLAYS_CALLS_AS_WRITTEN
    assert calls == ['{"query": "false"}' if as_written else '{"query": false}'] * len(executed)


@pytest.mark.parametrize("batched", [False, True])
async def test_a_review_agent_submits_through_call_tool_and_nothing_after_it_runs(monkeypatch, batched):
    executed = []
    submission = {"tool_name": "submit_draft", "params": json.dumps({"draft": "Dispatched draft"})}
    call = {"calls": [submission, h.LOOKUP_CALL]} if batched else submission

    sent, _, _ = await h.run_entry(
        monkeypatch, "llm_review", h.calling(("call_tool", call)), regular_tools=recording_tools(executed),
        settings=h.TOOL_SEARCH_SETTINGS,
    )

    assert not executed
    assert not any(h.called(body, "call_tool") for body in sent if "submit_review" not in body["tools"])
    reviews = [
        "\n".join(message["content"] for message in body["messages"] if isinstance(message["content"], str))
        for body in sent if "submit_review" in body["tools"]
    ]
    assert reviews and all("DRAFT:\nDispatched draft\n" in review for review in reviews)


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_a_call_tool_batch_naming_a_tool_by_a_non_string_gets_cores_error_and_the_loop_goes_on(monkeypatch, entry):
    batch = {"calls": [{"tool_name": ["lookup"], "params": "{}"}, h.LOOKUP_CALL]}

    sent, _, _ = await h.run_entry(
        monkeypatch, entry, h.calling(("call_tool", batch)), regular_tools=h.loop_tools, settings=h.TOOL_SEARCH_SETTINGS,
    )

    results = [json.loads(result) for result in tool_results(sent) if result.startswith('{"results"')]
    assert results and all(
        [entry["output"][0]["text"] for entry in result["results"]]
        == ["Error: tool_name must be a non-empty string.", "found"]
        for result in results
    )


@pytest.mark.parametrize("entry", h.ENTRIES)
async def test_a_call_tool_that_fails_to_find_its_tool_twice_stops_tool_use(monkeypatch, entry):
    from open_webui.utils.tool_calling import FINAL_ANSWER_PROMPT

    missing = 'Error: Tool "missing" is not available. Search for an available tool.'

    def respond(body):
        if "call_tool" in body["tools"]:
            return h.tool_call("call_tool", {"tool_name": "missing", "params": "{}"})
        return h.answer(body)

    sent, _, _ = await h.run_entry(
        monkeypatch, entry, respond, regular_tools=h.loop_tools, settings=h.TOOL_SEARCH_SETTINGS,
        valves={"MAX_ITERATIONS": 6},
    )

    histories = [tool_results([body]) for body in sent if missing in tool_results([body])]
    assert histories and max(results.count(missing) for results in histories) == 3
    stopped = [body for body in sent if tool_results([body]).count(missing) == 3]
    assert stopped and all(
        FINAL_ANSWER_PROMPT in str(body["messages"]) or "Tool use has stopped for this response." in str(body["messages"])
        for body in stopped
    )


async def test_agent_tools_run_together_by_parallel_tools_ask_one_question_at_a_time(monkeypatch):
    from pathlib import Path

    from open_webui.utils import chat as core_chat
    from open_webui.utils import models as core_models

    h.keep_core_config(monkeypatch, APPROVAL_SETTINGS)
    model = {"id": "chat", "owned_by": "openai", "info": {"meta": {}}}
    request = h.parallel_request(model)
    monkeypatch.setattr(core_models, "get_all_models", AsyncMock(return_value=[model]))
    agents = {name: Path(importlib.import_module(name).__file__).read_text() for name in ("sub_agent", "llm_review")}
    metadata = await h.core_chat_tools(
        monkeypatch, request, h.make_user("admin"), model,
        workspace={**agents, "record": ASKING_TOOL},
        params={"function_calling": "native", "tool_approval_mode": "ask"},
    )
    asked, waiting, most = [], [0], [0]

    async def event_call(event):
        if event.get("type") not in ("confirmation", "input"):
            return None
        asked.append(event["data"]["tool_call"]["name"] if event["type"] == "confirmation" else "input")
        waiting[0] += 1
        most[0] = max(most[0], waiting[0])
        await asyncio.sleep(0.01)
        waiting[0] -= 1
        return True

    respond = h.calling(("lookup", {"command": "key"}))

    async def provider(*, form_data, **_):
        return respond({"tools": h.tool_names(form_data), "messages": form_data["messages"]})

    monkeypatch.setattr(core_chat, "generate_openai_chat_completion", provider)
    await bind_browser(metadata, event_call)

    results = await h.call_parallel(
        metadata,
        ("run_sub_agent", {"description": "A", "prompt": "A"}),
        ("run_sub_agent", {"description": "B", "prompt": "B"}),
        ("llm_review", {"topic": "Write the draft"}),
    )

    assert all("error" not in result for result in results)
    assert asked.count("lookup") >= 3 and asked.count("input") >= 3 and most[0] == 1
