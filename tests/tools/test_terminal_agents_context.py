"""Terminal AGENTS.md survives nested requests without replacing the task."""

import asyncio
import copy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import llm_review
import magi_decision_support
import multi_model_council
import sub_agent
from open_webui.utils import terminals as core_terminals


AGENTS_MD = "# AGENTS.md\n\nFollow the project conventions."
MESSAGES = [
    {"role": "system", "content": "Agent system prompt"},
    {"role": "user", "content": "The delegated task"},
]


async def test_agents_read_cancellation_does_not_open_mcp_clients(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_webui.utils.terminals", core_terminals)
    monkeypatch.setattr(core_terminals, "get_terminal_agents_md", AsyncMock(side_effect=asyncio.CancelledError))
    mcp = AsyncMock(return_value=({}, {}))
    monkeypatch.setattr(sub_agent, "resolve_mcp_tools", mcp)
    metadata = {"terminal_id": "terminal"}
    with pytest.raises(asyncio.CancelledError):
        await sub_agent.build_tools_dict(
            request=SimpleNamespace(), model={}, metadata=metadata,
            user=SimpleNamespace(id="u1"), valves=SimpleNamespace(),
            extra_params={"__metadata__": metadata},
            tool_id_list=["server:mcp:test"], excluded_tool_ids=None,
            resolved_terminal_id="terminal", resolved_direct_tool_servers=[],
            include_terminal_agents_md=True,
        )
    mcp.assert_not_awaited()


@pytest.mark.parametrize(
    "module", [sub_agent, llm_review, magi_decision_support, multi_model_council]
)
async def test_agents_md_is_once_before_task_in_initial_and_final_request(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "open_webui.utils.terminals", core_terminals)
    calls = []

    async def completion(**kwargs):
        calls.append(copy.deepcopy(kwargs["form_data"]))
        if len(calls) == 1:
            return {"choices": [{"message": {"content": "", "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "noop", "arguments": "{}"},
            }]}}]}
        return {"choices": [{"message": {"content": "done"}}]}

    async def noop():
        return "ok"

    chat = ModuleType("open_webui.utils.chat")
    chat.generate_chat_completion = completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat)
    messages = copy.deepcopy(MESSAGES)
    kwargs = {
        "request": SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(MODELS={}))),
        "user": {"id": "u1", "role": "user"},
        "model_id": "model",
        "messages": messages,
        "tools_dict": {"noop": {"spec": {"name": "noop"}, "callable": noop}},
        "max_iterations": 1,
        "extra_params": {"__terminal_agents_md__": AGENTS_MD},
        "apply_inlet_filters": False,
    }
    if module is sub_agent:
        kwargs["compaction"] = sub_agent.LoopCompactionOptions(enabled=False)
        kwargs["large_results"] = sub_agent.LargeToolResultOptions(mode="raw")
        result = await sub_agent.run_sub_agent_loop(**kwargs)
    else:
        if module is llm_review:
            kwargs["filter_pipeline"] = await module.resolve_model_filter_pipeline(
                False, kwargs["request"], "model", []
            )
            monkeypatch.setattr(
                module, "resolve_model_filter_pipeline",
                AsyncMock(side_effect=AssertionError("The supplied route must be reused")),
            )
        result = await module.run_agent_loop(**kwargs)

    assert result == "done"
    assert len(calls) == 2
    assert messages == MESSAGES
    for body in calls:
        users = [message for message in body["messages"] if message["role"] == "user"]
        assert users[0] == {"role": "user", "content": AGENTS_MD}
        assert users[1]["content"].startswith("The delegated task")
        assert sum(message["content"] == AGENTS_MD for message in users) == 1


@pytest.mark.parametrize("agents_md", [AGENTS_MD, f"{AGENTS_MD}\n\nEdited by a request filter."])
def test_compaction_keeps_agents_md_and_the_task_however_filters_left_them(agents_md):
    messages = [MESSAGES[0], {"role": "user", "content": agents_md}, MESSAGES[1]]
    for index in range(3):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{index}"}]},
            {"role": "tool", "tool_call_id": f"c{index}", "content": "result"},
        ])
    cut = sub_agent.select_loop_compaction_cut(messages, preserved_count=3)
    assert cut.preserved_messages == messages[:3]
    assert sub_agent.loop_task_message_index(messages, 3) == 2


def test_older_core_without_agents_helper_keeps_original_messages(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_webui.utils.terminals", ModuleType("open_webui.utils.terminals"))
    body = sub_agent._append_tool_server_prompts(
        {"messages": copy.deepcopy(MESSAGES)}, {"__terminal_agents_md__": AGENTS_MD}
    )
    assert body["messages"] == MESSAGES


async def test_sub_agent_full_estimate_includes_agents_as_user_instructions(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_webui.utils.terminals", core_terminals)
    run = sub_agent._LoopRunState(
        compaction=sub_agent.LoopCompactionOptions(),
        large_results=sub_agent.LargeToolResultOptions(mode="raw"),
        encoder=object(), encoder_ready=True, filter_identity=[],
    )
    captured = []

    def estimate(body, encoder):
        captured.append(body)
        return 123

    monkeypatch.setattr(sub_agent, "estimate_body_tokens", estimate)
    messages = [MESSAGES[0], {"role": "user", "content": AGENTS_MD}, MESSAGES[1]]
    result = await sub_agent._estimate_loop_tokens(
        run, model_id="model", tools_param=None, current_messages=copy.deepcopy(messages),
        volatile_tokens=0, metadata={}, user_obj=None,
    )
    assert result == 123
    assert captured[0]["messages"] == messages


async def test_review_reuses_agents_md_across_cached_compose_review_and_revise(monkeypatch):
    loader_calls = []
    loop_calls = []

    async def load_tools(**kwargs):
        assert kwargs["include_terminal_agents_md"] is True
        loader_calls.append(kwargs["model"])
        await asyncio.sleep(0)
        kwargs["extra_params"]["__terminal_agents_md__"] = AGENTS_MD
        return {}, {}

    async def run_loop(**kwargs):
        assert kwargs["extra_params"]["__terminal_agents_md__"] == AGENTS_MD
        assert all("Parent private conversation" not in message["content"] for message in kwargs["messages"])
        tool_name = next(iter(kwargs["submission_tool_names"]))
        loop_calls.append(tool_name)
        payload = {"key_feedback": "Review feedback"} if tool_name == "submit_review" else {"draft": "A complete draft"}
        await kwargs["tools_dict"][tool_name]["callable"](**payload)
        return ""

    monkeypatch.setattr(llm_review, "build_tools_dict", load_tools)
    monkeypatch.setattr(llm_review, "run_agent_loop", run_loop)
    monkeypatch.setattr(llm_review, "get_available_models", AsyncMock(return_value=[{"id": "model"}]))
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(MODELS={"model": {"id": "model"}})),
        body=AsyncMock(return_value=b"{}"),
    )
    tool = llm_review.Tools()
    tool.valves.DEFAULT_MODELS = "model"
    await tool.llm_review(
        topic="Write the draft", __request__=request,
        __model__={"id": "model"},
        __user__={"id": "u1", "role": "user", "valves": {"ROUNDS": 1, "RICH_PROGRESS": False}},
        __messages__=[{"role": "user", "content": "Parent private conversation"}],
    )
    assert len(loader_calls) == 1
    assert loop_calls.count("submit_draft") == 6
    assert loop_calls.count("submit_review") == 6


@pytest.mark.parametrize(
    ("route", "terminal_allowed"),
    [("arena", True), ("arena", False), ("direct", True), ("direct", False), ("arena", "alternate")],
)
async def test_review_loads_terminal_context_for_each_effective_route(
    monkeypatch, route, terminal_allowed
):
    from open_webui.models.config import Config
    import open_webui.utils.tools as core_tools

    def model(model_id, allowed):
        return {"id": model_id, "info": {"meta": {"capabilities": {"terminal": allowed}}}}

    allowed_model = model("allowed", True)
    denied_model = model("denied", False)
    nominal = model("route", terminal_allowed is False)
    nominal["owned_by"] = "arena"
    nominal["info"]["meta"]["model_ids"] = ["allowed", "denied"]
    direct_model = model("route", terminal_allowed is True)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(MODELS={
            "route": nominal, "allowed": allowed_model, "denied": denied_model,
        })),
        state=SimpleNamespace(direct=route == "direct", model=direct_model),
        body=AsyncMock(return_value=b"{}"),
    )
    choices = []
    pipelines = []
    terminal_loads = []
    agents_loads = []
    requests = []
    resolve = llm_review.resolve_model_filter_pipeline

    def choose(candidates):
        assert route == "arena"
        selected = (
            "allowed" if len(choices) % 2 == 0 else "denied"
        ) if terminal_allowed == "alternate" else (
            "allowed" if terminal_allowed else "denied"
        )
        assert selected in candidates
        choices.append(selected)
        return selected

    async def resolve_once(*args):
        pipeline = await resolve(*args)
        pipelines.append(pipeline)
        return pipeline

    async def config_get(key, default=None):
        return [{"id": "terminal-1"}] if key == "terminal_server.connections" else default

    async def terminal_tools(**kwargs):
        target = kwargs["extra_params"]["__model__"]
        assert target["info"]["meta"]["capabilities"]["terminal"] is True
        terminal_loads.append(target)
        await asyncio.sleep(0)
        return {"terminal_probe": {"spec": {"name": "terminal_probe"}}}

    async def agents_md(request, user, metadata, extra_params):
        agents_loads.append(extra_params["__model__"])
        return AGENTS_MD

    async def completion(**kwargs):
        body = kwargs["form_data"]
        requests.append(copy.deepcopy(body))
        tool_names = {item["function"]["name"] for item in body["tools"]}
        submission = "submit_review" if "submit_review" in tool_names else "submit_draft"
        args = '{"key_feedback":"Review feedback"}' if submission == "submit_review" else '{"draft":"A complete draft"}'
        return {"choices": [{"message": {"content": "", "tool_calls": [{
            "id": "submit", "type": "function",
            "function": {"name": submission, "arguments": args},
        }]}}]}

    chat = ModuleType("open_webui.utils.chat")
    chat.generate_chat_completion = completion
    terminals = ModuleType("open_webui.utils.terminals")
    terminals.get_terminal_agents_md = agents_md
    terminals.add_terminal_agents_md = core_terminals.add_terminal_agents_md
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat)
    monkeypatch.setitem(sys.modules, "open_webui.utils.terminals", terminals)
    monkeypatch.setattr(Config, "get", staticmethod(config_get))
    monkeypatch.setattr(core_tools, "get_terminal_tools", terminal_tools)
    monkeypatch.setattr(resolve.__globals__["random"], "choice", choose)
    monkeypatch.setattr(llm_review, "resolve_model_filter_pipeline", resolve_once)
    monkeypatch.setattr(llm_review, "get_available_models", AsyncMock(return_value=[{"id": "route"}]))
    tool = llm_review.Tools()
    tool.valves.DEFAULT_MODELS = "route"
    tool.valves.APPLY_INLET_FILTERS = False
    tool.valves.ENABLE_TERMINAL_TOOLS = True
    await tool.llm_review(
        topic="Write the draft", __request__=request, __model__={"id": "route"},
        __user__={"id": "u1", "role": "user", "valves": {"ROUNDS": 1, "RICH_PROGRESS": False}},
        __metadata__={"terminal_id": "terminal-1"},
    )

    assert len(requests) == len(pipelines) == 12  # 3 compose + 6 review + 3 revise
    assert len({id(pipeline) for pipeline in pipelines}) == 12
    assert len(choices) == (12 if route == "arena" else 0)
    expected_loads = [] if terminal_allowed is False else [direct_model if route == "direct" else allowed_model]
    assert terminal_loads == agents_loads == expected_loads
    for body in requests:
        selected = direct_model if route == "direct" else request.app.state.MODELS[body["model"]]
        allowed = selected["info"]["meta"]["capabilities"]["terminal"]
        assert sum(message["content"] == AGENTS_MD for message in body["messages"]) == int(allowed)
        tool_names = {item["function"]["name"] for item in body["tools"]}
        # Reviews intentionally expose only their submission tool.
        if "submit_draft" in tool_names:
            assert ("terminal_probe" in tool_names) is allowed
    if terminal_allowed == "alternate":
        assert {body["model"] for body in requests} == {"allowed", "denied"}
    else:
        expected_id = "route" if route == "direct" else "allowed" if terminal_allowed else "denied"
        assert {body["model"] for body in requests} == {expected_id}
