"""Tests for graphiti/functions/filter/graphiti_memory.py extension."""

import asyncio
import copy
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# Add graphiti filter directory to path
graphiti_filter_dir = Path(__file__).parent.parent.parent / "graphiti" / "functions" / "filter"
sys.path.insert(0, str(graphiti_filter_dir))


class TestGraphitiMemoryFilterInstantiation:
    """Test that the Graphiti Memory filter can be instantiated."""

    def test_import_filter_class(self):
        """Verify Filter class can be imported."""
        from graphiti_memory import Filter

        assert Filter is not None

    def test_instantiate_filter(self):
        """Verify Filter class can be instantiated."""
        from graphiti_memory import Filter

        f = Filter()
        assert f is not None

    def test_valves_initialization(self):
        """Verify Valves are properly initialized."""
        from graphiti_memory import Filter

        f = Filter()
        assert hasattr(f, "valves")
        # Check for key valve attributes
        assert hasattr(f.valves, "priority")
        assert hasattr(f.valves, "api_key")

    def test_has_inlet_method(self):
        """Verify Filter class has inlet method."""
        from graphiti_memory import Filter

        f = Filter()
        assert hasattr(f, "inlet")
        assert callable(f.inlet)

    def test_has_outlet_method(self):
        """Verify Filter class has outlet method."""
        from graphiti_memory import Filter

        f = Filter()
        assert hasattr(f, "outlet")
        assert callable(f.outlet)

    @pytest.mark.parametrize("chat_id", ["local:abc123", "temporary:abc123"])
    def test_temporary_chat_is_not_regular(self, chat_id):
        """Temporary chat IDs aren't backed by Open WebUI's Chats table."""
        from graphiti_memory import _is_regular_chat_id

        assert not _is_regular_chat_id(chat_id)

    @pytest.mark.parametrize("chat_id", ["local:abc123", "temporary:abc123"])
    @pytest.mark.asyncio
    async def test_temporary_chat_skips_inlet_search(self, chat_id):
        """Temporary chats don't retrieve persisted Graphiti memory."""
        from graphiti_memory import Filter

        f = Filter()
        f._ensure_graphiti_initialized = AsyncMock(return_value=True)
        body = {"messages": [{"role": "user", "content": "hello"}]}
        user = {"id": "u1", "email": "u1@example.com", "role": "user", "valves": {}}

        result = await f.inlet(
            body,
            __event_emitter__=AsyncMock(),
            __user__=user,
            __metadata__={"chat_id": chat_id, "message_id": "m1"},
        )

        assert result is body
        f._ensure_graphiti_initialized.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_channel_chat_runs_inlet_search(self):
        """Open WebUI 0.9.5 channel requests run through the full filter pipeline."""
        from graphiti_memory import Filter

        f = Filter()
        f._ensure_graphiti_initialized = AsyncMock(return_value=True)
        f.graphiti = SimpleNamespace(
            search_=AsyncMock(
                return_value=SimpleNamespace(edges=[], nodes=[])
            )
        )

        body = {"messages": [{"role": "user", "content": "hello"}]}
        user = {"id": "u1", "email": "u1@example.com", "role": "user", "valves": {}}

        result = await f.inlet(
            body,
            __event_emitter__=AsyncMock(),
            __user__=user,
            __metadata__={"chat_id": "channel:abc123", "message_id": "m1"},
        )

        assert result is body
        f._ensure_graphiti_initialized.assert_awaited_once()
        f.graphiti.search_.assert_awaited_once()


FACT = "User likes tea"
QUESTION = "What do I like?"


def filter_with_one_fact(memory_message_role: str | None = None):
    from graphiti_memory import Filter

    f = Filter()
    if memory_message_role is not None:
        f.valves.memory_message_role = memory_message_role
    f._ensure_graphiti_initialized = AsyncMock(return_value=True)
    edge = SimpleNamespace(uuid="e1", name="LIKES", fact=FACT, valid_at=None, invalid_at=None)
    f.graphiti = SimpleNamespace(
        search_=AsyncMock(return_value=SimpleNamespace(edges=[edge], nodes=[]))
    )
    return f


def new_request():
    """Stand-in for the Request object Open WebUI passes to both inlet and outlet of one request."""
    return SimpleNamespace(state=SimpleNamespace())


async def run_inlet(
    f, question=QUESTION, chat_id="c1", request=None, message_id="m1", filter_id=None, **metadata
) -> list[dict]:
    body = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": question},
        ]
    }
    user = {"id": "u1", "email": "u1@example.com", "role": "user", "valves": {}}

    result = await f.inlet(
        body,
        __event_emitter__=AsyncMock(),
        __user__=user,
        __metadata__={"chat_id": chat_id, "message_id": message_id, **metadata},
        __id__=filter_id,
        __request__=request,
    )
    return result["messages"]


async def as_open_webui_leaves_them(messages: list[dict], rag_template_text: str) -> tuple[list[dict], str]:
    """What Open WebUI does to the messages after inlet, and the prompt it records on the way."""
    from open_webui.utils.misc import add_or_update_user_message, get_last_user_message
    from open_webui.utils.skills import strip_skill_mentions
    from open_webui.utils.task import rag_template

    strip_skill_mentions(messages, {"tea"})
    prompt = get_last_user_message(messages)
    rag = await rag_template(rag_template_text, "doc", prompt)
    return add_or_update_user_message(rag, messages, append=False), prompt


async def saved_episode_body(f, messages: list[dict], request=None, emitter=None, **metadata) -> str | None:
    """Run outlet the way Open WebUI does for a chat it does not store and return what gets saved, if anything."""
    f.valves.previous_episodes_from_chat = -1
    f.graphiti.add_episode = AsyncMock(return_value=None)
    f._wait_for_chat_title = AsyncMock(return_value="Title")
    user = {
        "id": "u1",
        "email": "u1@example.com",
        "name": "U",
        "role": "user",
        "valves": {"episode_dedup_enabled": False, "show_status": emitter is not None},
    }

    await f.outlet(
        {
            "model": "m",
            "messages": [
                *({"role": m["role"], "content": m["content"]} for m in messages),
                {"role": "assistant", "content": "Tea."},
            ],
        },
        __event_emitter__=emitter or AsyncMock(),
        __user__=user,
        __metadata__={"chat_id": "", "message_id": "m1", **metadata},
        __request__=request,
    )
    await asyncio.gather(*f._background_tasks)
    saved = f.graphiti.add_episode.await_args
    return saved and saved.kwargs["episode_body"]


async def inlet_with_one_fact(memory_message_role: str | None = None) -> list[dict]:
    return await run_inlet(filter_with_one_fact(memory_message_role))


class TestGraphitiMemoryInjectionPlacement:
    """Role and position of the injected memory message."""

    @pytest.mark.parametrize("memory_message_role", [None, "bogus"])
    @pytest.mark.asyncio
    async def test_user_role_memory_is_appended_to_latest_user_message(self, memory_message_role):
        """The default (and any unknown value) adds no message and appends the memory to the user's prompt."""
        messages = await inlet_with_one_fact(memory_message_role)

        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[2]["content"].startswith(f"{QUESTION}\n\n<")
        assert FACT in messages[2]["content"]

    @pytest.mark.asyncio
    async def test_user_role_memory_goes_into_first_text_part_without_mutating_the_original(self):
        """List content gets the memory in its first text part, which is the one Open WebUI reads as the prompt."""
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        question = [{"type": "text", "text": QUESTION}, image]
        original = copy.deepcopy(question)

        messages = await run_inlet(filter_with_one_fact(), question)

        assert messages[2]["content"][0]["text"].startswith(f"{QUESTION}\n\n<")
        assert FACT in messages[2]["content"][0]["text"]
        assert messages[2]["content"][1:] == [image]
        assert question == original

    @pytest.mark.parametrize(
        "question",
        ["/skills:create", "  /Skills:Create summarize\nthis chat  ", "/skills:createx", "please /skills:create"],
    )
    @pytest.mark.asyncio
    async def test_user_role_skips_search_for_exactly_what_core_handles_as_skills_create(self, question):
        """No search or injection for, and only for, text that Open WebUI rewrites as a /skills:create command."""
        from open_webui.utils.skills import apply_skills_create_prompt

        f = filter_with_one_fact()
        messages = await run_inlet(f, question)

        skipped = messages[2] == {"role": "user", "content": question} and not f.graphiti.search_.await_count
        is_command = apply_skills_create_prompt([{"role": "user", "content": question}], allowed=True)
        assert skipped == is_command
        assert len(messages) == 3

    @pytest.mark.parametrize("question", [QUESTION, [{"type": "text", "text": QUESTION}]])
    @pytest.mark.asyncio
    async def test_outlet_does_not_save_memory_appended_by_inlet(self, question):
        """Open WebUI hands outlet the inlet-modified request for chats it does not store, such as API calls."""
        f = filter_with_one_fact()
        request = new_request()
        sent = await run_inlet(f, question, chat_id="", request=request)

        assert FACT in str(sent[2]["content"])
        assert await saved_episode_body(f, sent, request) == f"Assistant: Hello\n\n---\n\nU: {QUESTION}"

    @pytest.mark.asyncio
    async def test_outlet_keeps_user_text_that_looks_like_a_memory_block(self):
        """Only what inlet appended is removed; the user's own text is saved as written."""
        own_text = f"{QUESTION}\n\n<Graphiti Memory>\n# my own notes\n</Graphiti Memory>"
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": own_text},
        ]

        saved = await saved_episode_body(filter_with_one_fact(), messages, new_request())

        assert saved == f"Assistant: Hello\n\n---\n\nU: {own_text}"

    @pytest.mark.asyncio
    async def test_outlet_ignores_memory_appended_in_another_request(self):
        """A block appended in one request never removes the same text from another request's user message."""
        f = filter_with_one_fact()
        other_request = await run_inlet(f, chat_id="", request=new_request())
        own_text = other_request[2]["content"]
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": own_text},
        ]

        saved = await saved_episode_body(f, messages, new_request())

        assert FACT in own_text
        assert saved == f"Assistant: Hello\n\n---\n\nU: {own_text}"

    @pytest.mark.asyncio
    async def test_outlet_keeps_what_else_was_added_to_the_user_message(self):
        """For API calls, retrieved context reaches outlet only inside the user message, where Open WebUI puts it."""
        from open_webui.utils.misc import add_or_update_user_message

        f = filter_with_one_fact()
        request = new_request()
        sent = await run_inlet(f, chat_id="", request=request)
        sent[2] = {**sent[2], "content": f"{sent[2]['content']}\n\n<SYSTEM_CONTEXT>\nnow\n</SYSTEM_CONTEXT>"}
        sent = add_or_update_user_message("<context>\ndoc\n</context>", sent, append=False)

        assert await saved_episode_body(f, sent, request) == (
            f"Assistant: Hello\n\n---\n\nU: <context>\ndoc\n</context>\n{QUESTION}"
            "\n\n<SYSTEM_CONTEXT>\nnow\n</SYSTEM_CONTEXT>"
        )

    @pytest.mark.parametrize(
        "template",
        ["", "<context>{{CONTEXT}}</context>\n<user_query>{{QUERY}}</user_query>", "{{CONTEXT}}\n\n{{QUERY}}"],
        ids=["default template", "query placeholder", "blank line before query"],
    )
    @pytest.mark.parametrize(
        "own_text",
        [
            QUESTION,
            "Why did I get this?\n\n{block}\n\nTell me.",
            "{block}\n\nWhy did I get this?",
            f"{QUESTION} <$tea|Tea skill>",
        ],
        ids=["plain", "pasted block", "pasted block first", "skill mention"],
    )
    @pytest.mark.parametrize("prompt_key", ["base_user_prompt", "user_prompt"])
    @pytest.mark.asyncio
    async def test_outlet_saves_what_open_webui_would_build_from_the_prompt_without_memory(
        self, template, own_text, prompt_key
    ):
        """After inlet, Open WebUI puts skill labels in place of mentions, then a RAG template before the prompt."""
        f = filter_with_one_fact()
        block = (await run_inlet(f))[2]["content"].removeprefix(f"{QUESTION}\n\n")
        own_text = own_text.replace("{block}", block)
        request = new_request()
        sent, prompt = await as_open_webui_leaves_them(
            await run_inlet(f, own_text, chat_id="", request=request), template
        )
        without_memory = (await as_open_webui_leaves_them([{"role": "user", "content": own_text}], template))[0][0][
            "content"
        ]

        assert sent[2]["content"].count(block) > without_memory.count(block)
        assert await saved_episode_body(f, sent, request, **{prompt_key: prompt}) == (
            f"Assistant: Hello\n\n---\n\nU: {without_memory}"
        )

    def test_open_webui_records_the_prompt_between_skill_labels_and_the_rag_template(self):
        """outlet reads the prompt Open WebUI hands to the RAG template: "base_user_prompt", or "user_prompt" before 0.12.0."""
        from open_webui.utils import middleware

        source = inspect.getsource(middleware.process_chat_payload)
        labels = source.index("strip_skill_mentions(")
        key = "base_user_prompt" if "metadata['base_user_prompt']" in source else "user_prompt"
        record = source.index(f"metadata['{key}'] = get_last_user_message(form_data['messages'])")
        rag = source.index("apply_source_context_to_messages(request, form_data['messages'], sources, prompt)")
        assert labels < record < rag

    def test_open_webui_stores_the_resolved_tools_after_inlet_and_before_any_tool_runs(self):
        """inlet reads __metadata__["tools"], in the dict Open WebUI also gives to tools, as a sign that a tool calls it."""
        from open_webui.utils import middleware

        source = inspect.getsource(middleware.process_chat_payload)
        tools_metadata = source.index("'__metadata__': metadata,")
        inlet = source.index("filter_type='inlet'")
        store = source.index("metadata['tools'] = tools_dict")
        first_tool_run = source.index("await chat_completion_tools_handler(")
        assert tools_metadata < inlet < store < first_tool_run

    @pytest.mark.asyncio
    async def test_outlet_removes_memory_after_open_webui_rewrote_the_prompt_in_front_of_it(self):
        """After inlet, Open WebUI replaces mentions of the skills in use with their labels."""
        from open_webui.utils.skills import strip_skill_mentions

        f = filter_with_one_fact()
        request = new_request()
        sent = await run_inlet(f, f"<$tea|Tea skill> {QUESTION}", chat_id="", request=request)
        strip_skill_mentions(sent, {"tea"})

        assert FACT in sent[2]["content"]
        assert await saved_episode_body(f, sent, request) == f"Assistant: Hello\n\n---\n\nU: Tea skill {QUESTION}"

    @pytest.mark.asyncio
    async def test_outlet_removes_its_own_memory_when_several_models_answer_one_request(self):
        """Each answering model gets its own inlet and outlet call, all on the same request."""
        f = filter_with_one_fact()
        other_edge = SimpleNamespace(uuid="e2", name="LIKES", fact="User likes coffee", valid_at=None, invalid_at=None)
        f.graphiti.search_.side_effect = [SimpleNamespace(edges=[other_edge], nodes=[]), f.graphiti.search_.return_value]
        request = new_request()
        await run_inlet(f, chat_id="", request=request, message_id="m2")
        sent = await run_inlet(f, chat_id="", request=request)

        assert FACT in sent[2]["content"]
        assert await saved_episode_body(f, sent, request) == f"Assistant: Hello\n\n---\n\nU: {QUESTION}"

    @pytest.mark.asyncio
    async def test_outlet_ignores_memory_a_tool_appended_by_rerunning_inlet(self):
        """Tools run inlet again, on the same request, for the model calls they make themselves."""
        f = filter_with_one_fact()
        block = (await run_inlet(f))[2]["content"].removeprefix(f"{QUESTION}\n\n")
        own_text = f"Why did I get this?\n\n{block}"
        f.graphiti.search_.side_effect = [SimpleNamespace(edges=[], nodes=[]), f.graphiti.search_.return_value]
        request = new_request()
        sent = await run_inlet(f, own_text, chat_id="", request=request)
        nested = await run_inlet(f, chat_id="", request=request)

        assert sent[2]["content"] == own_text
        assert nested[2]["content"] == f"{QUESTION}\n\n{block}"
        assert await saved_episode_body(f, sent, request) == f"Assistant: Hello\n\n---\n\nU: {own_text}"

    @pytest.mark.parametrize("own_text", [QUESTION, "Why did I get this?\n\n{block}"], ids=["plain", "pasted block"])
    @pytest.mark.asyncio
    async def test_outlet_ignores_memory_another_copy_of_the_filter_appended_for_a_tool(self, own_text):
        """A tool can call a model that has a copy of the filter the answering model lacks, passing its metadata on."""
        answering, called = filter_with_one_fact(), filter_with_one_fact()
        block = (await run_inlet(called, filter_id="called"))[2]["content"].removeprefix(f"{QUESTION}\n\n")
        own_text = own_text.replace("{block}", block)
        request = new_request()
        sent, prompt = await as_open_webui_leaves_them(
            await run_inlet(answering, own_text, chat_id="", request=request, filter_id="answering"), ""
        )
        without_memory = (await as_open_webui_leaves_them([{"role": "user", "content": own_text}], ""))[0][0]["content"]
        nested = await run_inlet(called, chat_id="", request=request, filter_id="called", tools={"sub_agent": {}})

        assert nested[2]["content"] == f"{QUESTION}\n\n{block}"
        assert await saved_episode_body(answering, sent, request, user_prompt=prompt) == (
            f"Assistant: Hello\n\n---\n\nU: {without_memory}"
        )

    @pytest.mark.parametrize("second_fact", ["User likes coffee", FACT], ids=["other memory", "identical block"])
    @pytest.mark.parametrize(
        "own_text, template",
        [(QUESTION, ""), (f"<$tea|Tea skill> {QUESTION}", "{{CONTEXT}}\n\n{{QUERY}}")],
        ids=["plain", "skill mention and query placeholder"],
    )
    @pytest.mark.asyncio
    async def test_outlet_removes_the_memory_of_every_copy_of_the_filter(self, second_fact, own_text, template):
        """Open WebUI runs each installed copy of the filter, under its own id, on the same request and message."""
        first, second = filter_with_one_fact(), filter_with_one_fact()
        first.valves.citation_source_name = second.valves.citation_source_name = "Memory"
        second.graphiti.search_.return_value.edges[0].fact = second_fact
        request = new_request()
        sent = await run_inlet(first, own_text, chat_id="", request=request, filter_id="first")
        sent = await run_inlet(second, sent[2]["content"], chat_id="", request=request, filter_id="second")
        sent, prompt = await as_open_webui_leaves_them(sent, template)
        without_memory = (await as_open_webui_leaves_them([{"role": "user", "content": own_text}], template))[0][0][
            "content"
        ]

        assert prompt.count(FACT) + prompt.count("User likes coffee") == 2
        for f in (first, second):
            assert await saved_episode_body(f, sent, request, user_prompt=prompt) == (
                f"Assistant: Hello\n\n---\n\nU: {without_memory}"
            )

    @pytest.mark.asyncio
    async def test_outlet_saves_nothing_when_the_memory_inlet_appended_is_no_longer_in_the_message(self):
        """Open WebUI puts skill labels in place of mentions inside the appended memory as well."""
        from open_webui.utils.skills import strip_skill_mentions

        f = filter_with_one_fact()
        f.graphiti.search_.return_value.edges[0].fact = "User asked for <$tea|Tea skill>"
        request = new_request()
        sent = await run_inlet(f, chat_id="", request=request)
        strip_skill_mentions(sent, {"tea"})
        emitter = AsyncMock()

        assert "User asked for Tea skill" in sent[2]["content"]
        assert await saved_episode_body(f, sent, request, emitter) is None
        assert emitter.await_args.args[0]["data"] == {
            "description": "⏭️ Skipped saving this turn (could not remove the appended memory from the message)",
            "done": True,
        }

    @pytest.mark.asyncio
    async def test_no_copy_of_the_filter_saves_when_the_memory_of_one_is_no_longer_in_the_message(self):
        """A copy cannot take out another copy's memory either once Open WebUI has rewritten it."""
        from open_webui.utils.skills import strip_skill_mentions

        first, second = filter_with_one_fact(), filter_with_one_fact()
        second.graphiti.search_.return_value.edges[0].fact = "User asked for <$tea|Tea skill>"
        request = new_request()
        sent = await run_inlet(first, chat_id="", request=request, filter_id="first")
        sent = await run_inlet(second, sent[2]["content"], chat_id="", request=request, filter_id="second")
        strip_skill_mentions(sent, {"tea"})

        assert FACT in sent[2]["content"] and "User asked for Tea skill" in sent[2]["content"]
        for f in (first, second):
            assert await saved_episode_body(f, sent, request) is None

    @pytest.mark.asyncio
    async def test_system_role_memory_goes_before_latest_user(self):
        """The system-role memory message keeps the user's prompt as the last message."""
        messages = await inlet_with_one_fact("system")

        assert [m["role"] for m in messages] == ["user", "assistant", "system", "user"]
        assert FACT in messages[2]["content"]
        assert messages[3]["content"] == QUESTION

    @pytest.mark.asyncio
    async def test_system_after_memory_goes_after_latest_user(self):
        """system_after places the system-role memory message right after the latest user message."""
        messages = await inlet_with_one_fact("system_after")

        assert [m["role"] for m in messages] == ["user", "assistant", "user", "system"]
        assert messages[2]["content"] == QUESTION
        assert FACT in messages[3]["content"]
