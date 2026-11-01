"""Exercise review-owned MCP scopes with Core's real connect and disconnect."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp_sdk import patch_mcp_sdk


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancelled", "prepare_error"])
async def test_review_mcp_lifetime_across_effective_models(monkeypatch, outcome):
    import llm_review
    from open_webui.utils.mcp.client import MCPClient

    opened, exit_attempts, closed, loaded, pipelines, loops, summaries, events = (
        [] for _ in range(8)
    )
    review_started = asyncio.Event()
    release_review = asyncio.Event()
    models = {name: {"id": name} for name in ("compose", "review", "revise", "broken")}
    models["arena"] = {
        "id": "arena", "owned_by": "arena",
        "info": {"meta": {"model_ids": list(models)}},
    }
    choices = (["broken", "compose", "compose"] if outcome == "prepare_error" else ["compose"] * 3)
    choices += ["review"] * 5 + ["compose"] + ["revise"] * 3
    selections = []

    def choose(candidates):
        selected = choices[len(selections)]
        assert selected in candidates
        selections.append(selected)
        return selected

    resolve = llm_review.resolve_model_filter_pipeline

    async def resolve_once(*args):
        pipeline = await resolve(*args)
        pipelines.append(pipeline)
        return pipeline

    @asynccontextmanager
    async def connection(label):
        # A real AnyIO task group enforces both task ownership and scope LIFO.
        try:
            async with anyio.create_task_group():
                opened.append((label, asyncio.current_task()))
                try:
                    yield None
                finally:
                    exit_attempts.append((label, asyncio.current_task()))
        except asyncio.CancelledError:
            # Core 0.12.5+ closes a session by cancelling the task that holds it.
            closed.append(label)
            raise
        # Core suppresses RuntimeError on disconnect; only successful scope
        # exits reach this line, so a swallowed ownership error still fails.
        closed.append(label)

    async def load_tools(**kwargs):
        model_id = kwargs["model"]["id"]
        loaded.append(model_id)
        clients = {}
        for index in range(2):
            client = MCPClient()
            await client.connect(url=f"{model_id}:{index}")
            clients[str(index)] = client
        await asyncio.sleep(0)
        return {}, clients

    async def register_skill(tools, request, extra):
        if extra["__model__"]["id"] == "broken":
            raise RuntimeError("preparation failed for broken")

    async def loop(**kwargs):
        pipeline = kwargs["filter_pipeline"]
        assert any(pipeline is resolved for resolved in pipelines)
        assert kwargs["model_id"] == pipeline["model_id"]
        assert kwargs["extra_params"]["__model__"] is pipeline["model"]
        assert not closed
        loops.append(pipeline)
        submission = next(iter(kwargs["submission_tool_names"]))
        if submission == "submit_review" and outcome == "cancelled":
            review_started.set()
            await release_review.wait()
        await asyncio.sleep(0)
        payload = (
            {"key_feedback": "Useful feedback"}
            if submission == "submit_review"
            else {"draft": f"draft-{kwargs['model_id']}"}
        )
        await kwargs["tools_dict"][submission]["callable"](**payload)
        return ""

    finalize = llm_review.EventEmitter.finalize

    async def capture_summary(self, *, summary):
        summaries.append(copy.deepcopy(summary))
        await finalize(self, summary=summary)

    async def emit(event):
        events.append(event)

    monkeypatch.setattr(resolve.__globals__["random"], "choice", choose)
    monkeypatch.setattr(llm_review, "resolve_model_filter_pipeline", resolve_once)
    patch_mcp_sdk(monkeypatch, connection)
    monkeypatch.setattr(llm_review, "build_tools_dict", load_tools)
    monkeypatch.setattr(llm_review, "run_agent_loop", loop)
    monkeypatch.setattr(llm_review, "register_view_skill", register_skill)
    monkeypatch.setattr(llm_review, "extract_skill_manifest", lambda _messages: "<available_skills />")
    monkeypatch.setattr(llm_review, "get_available_models", AsyncMock(return_value=[{"id": "arena"}]))
    monkeypatch.setattr(llm_review.EventEmitter, "finalize", capture_summary)
    request = SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(state=SimpleNamespace(MODELS=models)),
        body=AsyncMock(return_value=b"{}"),
    )
    tool = llm_review.Tools()
    tool.valves.DEFAULT_MODELS = "arena"
    tool.valves.APPLY_INLET_FILTERS = False
    invocation = asyncio.create_task(tool.llm_review(
        topic="Write a draft", __request__=request, __model__={"id": "arena"},
        __user__={
            "id": "mcp-user", "email": "mcp@example.com", "name": "MCP User", "role": "user",
            "last_active_at": 0, "updated_at": 0, "created_at": 0,
            "valves": {"ROUNDS": 1, "RICH_PROGRESS": False},
        },
        __event_emitter__=emit,
    ))
    if outcome == "cancelled":
        await asyncio.wait_for(review_started.wait(), timeout=5)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert summaries[-1]["cancelled"] is True
        assert all(draft["draft"] == "draft-compose" for draft in summaries[-1]["final_drafts"].values())
        assert len(summaries[-1]["final_drafts"]) == 3
        assert events[-1]["data"]["done"] is True
        expected_models = ["compose", "review"]
        expected_routes = 9
    else:
        response = json.loads(await invocation)
        assert len(response["final_drafts"]) == 3
        if outcome == "success":
            assert response["revisions_complete"] is True
            assert all(draft["draft"] == "draft-revise" for draft in response["final_drafts"].values())
            expected_models = ["compose", "review", "revise"]
        else:
            assert response["revisions_complete"] is False
            assert "preparation failed for broken" in response["final_drafts"]["arena#1"]["draft"]
            assert all(response["final_drafts"][aid]["draft"] == "draft-revise" for aid in ("arena#2", "arena#3"))
            expected_models = ["broken", "compose", "review", "revise"]
        expected_routes = 12
        assert len(loops) == expected_routes - int(outcome == "prepare_error")

    assert loaded == expected_models  # Same-target agents and phases share one catalogue.
    assert len(selections) == len(pipelines) == expected_routes
    assert len({id(pipeline) for pipeline in pipelines}) == expected_routes
    expected_closes = [label for label, _ in reversed(opened)]
    assert [label for label, _ in exit_attempts] == closed == expected_closes
    assert dict(exit_attempts) == dict(opened)
