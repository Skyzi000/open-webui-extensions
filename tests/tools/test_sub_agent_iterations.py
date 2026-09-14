"""Iteration semantics for sub_agent: 0=unlimited, default cap, note shape."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

tools_dir = Path(__file__).resolve().parents[2] / "tools"
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

import sub_agent  # noqa: E402


class _AppState:
    MODELS: dict = {}


class _FakeApp:
    state = _AppState()


class _FakeRequest:
    app = _FakeApp()


def _install_fake_completion(monkeypatch, responder):
    calls: list[dict] = []

    async def fake_completion(*, request, form_data, user, bypass_filter):
        calls.append(form_data)
        return responder(len(calls), form_data)

    fake_chat_module = types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = fake_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)
    return calls


async def _run_loop(monkeypatch, responder, *, max_iterations):
    calls = _install_fake_completion(monkeypatch, responder)
    statuses: list[str] = []

    async def emitter(event: dict) -> None:
        if event.get("type") == "status":
            statuses.append(event["data"]["description"])

    result = await sub_agent.run_sub_agent_loop(
        request=_FakeRequest(),
        user={"id": "u1", "role": "user"},
        model_id="m",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ],
        tools_dict={},
        max_iterations=max_iterations,
        event_emitter=emitter,
        apply_inlet_filters=False,
    )
    return result, calls, statuses


def _tool_call_response(call_id: str, name: str = "noop") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }


def _text_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


@pytest.mark.asyncio
async def test_unlimited_iterations_run_without_notes(monkeypatch):
    async def noop_tool(__tool__=None, **kwargs):
        return "ok"

    def responder(round_index: int, form_data: dict) -> dict:
        if round_index == 1:
            return _tool_call_response("call_1")
        return _text_response("all done")

    result, calls, statuses = await _run_loop(monkeypatch, responder, max_iterations=0)

    assert result == "all done"
    assert len(calls) == 2
    for form_data in calls:
        for message in form_data["messages"]:
            content = message.get("content", "")
            if isinstance(content, str):
                assert "[Iteration" not in content
    assert any(s.startswith("Sub-agent iteration 1") for s in statuses)
    assert not any("iteration 1/0" in s for s in statuses)


@pytest.mark.asyncio
async def test_limited_iterations_merge_note_into_task_user(monkeypatch):
    def responder(round_index: int, form_data: dict) -> dict:
        return _text_response("quick finish")

    result, calls, _statuses = await _run_loop(monkeypatch, responder, max_iterations=3)

    assert result == "quick finish"
    first_messages = calls[0]["messages"]
    assert first_messages[-1]["role"] == "user"
    assert "[Iteration 1/3]" in first_messages[-1]["content"]
    assert first_messages[-1]["content"].startswith("task")


@pytest.mark.asyncio
async def test_final_round_note_mentions_last_opportunity(monkeypatch):
    def responder(round_index: int, form_data: dict) -> dict:
        if round_index < 2:
            return _tool_call_response(f"call_{round_index}")
        return _text_response("done")

    result, calls, _statuses = await _run_loop(monkeypatch, responder, max_iterations=2)

    assert result == "done"
    second_messages = calls[1]["messages"]
    assert "[Iteration 2/2]" in second_messages[-1]["content"]
    assert "FINAL tool call opportunity" in second_messages[-1]["content"]


@pytest.mark.asyncio
async def test_exhaustion_path_returns_final_answer(monkeypatch):
    def responder(round_index: int, form_data: dict) -> dict:
        if form_data["messages"][-1].get("role") == "user" and "Maximum tool iterations reached" in str(
            form_data["messages"][-1].get("content", "")
        ):
            return _text_response("final synthesis")
        return _tool_call_response(f"call_{round_index}")

    result, _calls, statuses = await _run_loop(monkeypatch, responder, max_iterations=1)

    assert result == "final synthesis"
    assert any("Max iterations (1) reached" in s for s in statuses)


@pytest.mark.asyncio
async def test_exhaustion_is_unreachable_when_unlimited(monkeypatch):
    # A model that always answers with text exits immediately even with
    # MAX_ITERATIONS=0; the loop guard `max_iterations == 0 or ...` must
    # keep running rounds instead of skipping the loop body entirely.
    def responder(round_index: int, form_data: dict) -> dict:
        return _text_response(f"answer {round_index}")

    result, calls, _statuses = await _run_loop(monkeypatch, responder, max_iterations=0)

    assert result == "answer 1"
    assert len(calls) == 1


def test_valve_default_cap_is_50_and_allows_zero() -> None:
    valves = sub_agent.Tools.Valves()

    assert valves.MAX_ITERATIONS == 50
    field = sub_agent.Tools.Valves.model_fields["MAX_ITERATIONS"]
    assert field.metadata and getattr(field.metadata[0], "ge", None) == 0


def test_system_prompt_rule5_is_note_conditional() -> None:
    default = sub_agent.Tools.UserValves().SYSTEM_PROMPT

    assert (
        "5. If your messages contain [Iteration N/M] notes, your tool call "
        "iterations are limited. Complete the task before reaching the limit."
    ) in default
    assert "You have a limited number of tool call iterations" not in default
    assert "proactively" in default  # rules 3/4 untouched


def test_header_version_and_core_floor() -> None:
    header = (tools_dir / "sub_agent.py").read_text()[:2000]

    assert "version: 0.6.1" in header
    assert "required_open_webui_version: 0.9.6" in header


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("emit_fails", [False, True])
async def test_cancel_reports_status_and_propagates(monkeypatch, parallel, emit_fails):
    started = asyncio.Event()
    statuses = []

    async def run_loop(**kwargs):
        started.set()
        await asyncio.Event().wait()

    async def emit(event):
        statuses.append(event["data"])
        if event["data"]["done"] and emit_fails:
            raise RuntimeError("Notification failed")

    monkeypatch.setattr(sub_agent, "run_sub_agent_loop", run_loop)
    monkeypatch.setattr(sub_agent, "load_sub_agent_tools", AsyncMock(return_value=({}, {})))
    tool = sub_agent.Tools()
    kwargs = {
        "__request__": _FakeRequest(),
        "__user__": {"id": "u1"},
        "__model__": {"id": "m"},
        "__event_emitter__": emit,
    }
    work = {"description": "Task", "prompt": "Work"}
    run = (
        tool.run_parallel_sub_agents(tasks=[work, work], **kwargs)
        if parallel else tool.run_sub_agent(**work, **kwargs)
    )
    task = asyncio.create_task(run)
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert statuses[-1]["done"] is True
        assert "cancelled" in statuses[-1]["description"]
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
