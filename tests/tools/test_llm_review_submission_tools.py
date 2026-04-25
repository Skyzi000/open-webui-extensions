"""Tests for the submit_draft / submit_review pseudo-tool path.

Background: prior to v0.5.0 the compose/review/revise phases extracted the
agent's final payload by parsing a JSON object inside the assistant's
content. That contract broke as soon as the model emitted an unescaped
``"`` inside a string value — ``json.loads`` could not recover the draft.

v0.5.0 replaces the content-JSON contract with submission pseudo-tools
whose arguments arrive already structured via the LLM API's tool-call
channel (no string escaping needed). These tests cover:

- spec shape per phase (compose vs revise, with/without sources)
- capture box is populated when the tool is invoked
- ``extract_submission_payload`` gates on a non-empty primary key
- ``_parse_agent_status`` collapses the Executing/returned status pair
  into a single "submitted" info card and drops the echo line
- ``run_agent_loop`` exits immediately after a submission tool fires
  (no extra LLM round, iteration counter matches expectation)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

tools_dir = Path(__file__).parent.parent.parent / "tools"
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

import llm_review  # noqa: E402


# ---------------------------------------------------------------------------
# spec shape
# ---------------------------------------------------------------------------


def test_submit_draft_spec_compose_with_sources():
    spec = llm_review.build_submit_draft_spec(phase="compose", include_sources=True)
    assert spec["name"] == "submit_draft"
    props = spec["parameters"]["properties"]
    assert set(props.keys()) == {"draft", "approach", "sources"}
    assert spec["parameters"]["required"] == ["draft"]
    assert props["sources"]["type"] == "array"
    assert props["sources"]["items"]["type"] == "object"


def test_submit_draft_spec_compose_without_sources_drops_field():
    spec = llm_review.build_submit_draft_spec(phase="compose", include_sources=False)
    assert "sources" not in spec["parameters"]["properties"]


def test_submit_draft_spec_revise_has_change_fields():
    spec = llm_review.build_submit_draft_spec(phase="revise", include_sources=True)
    props = spec["parameters"]["properties"]
    # Revise gets changes_made / feedback_declined, NOT approach — compose
    # writes approach once and revise preserves it from the compose phase.
    assert set(props.keys()) == {"draft", "changes_made", "feedback_declined", "sources"}
    assert "approach" not in props


def test_submit_review_spec_shape():
    spec = llm_review.build_submit_review_spec()
    assert spec["name"] == "submit_review"
    props = spec["parameters"]["properties"]
    assert set(props.keys()) == {"strengths", "improvements", "key_feedback"}
    assert spec["parameters"]["required"] == ["key_feedback"]


# ---------------------------------------------------------------------------
# capture box + extract_submission_payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submission_tool_captures_kwargs():
    capture: dict = {"payload": None}
    tool = llm_review.make_submission_tool(
        spec=llm_review.build_submit_draft_spec(
            phase="compose", include_sources=True
        ),
        capture=capture,
    )

    # This is the payload that would arrive from the LLM — the key point is
    # that even unescaped ASCII double quotes inside Japanese prose (the
    # original bug trigger) flow through the tool channel unchanged. No
    # json.loads happens on this string anywhere in the submission path.
    tricky_draft = 'そっと忍ばせた"お祝い"のほうが、ずっと心に残ったりするんです。'
    result = await tool["callable"](
        draft=tricky_draft,
        approach="Casual, warm tone.",
        sources=[{"title": "T", "url": "https://example.test/"}],
    )

    assert "submit_draft" in result  # confirmation mentions the tool
    assert capture["payload"] is not None
    assert capture["payload"]["draft"] == tricky_draft
    assert capture["payload"]["approach"] == "Casual, warm tone."
    assert capture["payload"]["sources"] == [
        {"title": "T", "url": "https://example.test/"}
    ]


@pytest.mark.asyncio
async def test_submission_tool_instructs_end_of_turn():
    capture: dict = {"payload": None}
    tool = llm_review.make_submission_tool(
        spec=llm_review.build_submit_review_spec(), capture=capture
    )
    result = await tool["callable"](key_feedback="Great work.")
    # The confirmation string must tell the LLM to stop — otherwise some
    # models will keep calling tools after submission and waste iterations
    # (or worse, overwrite the capture with a bad second call).
    assert "end your turn" in result.lower()


def test_extract_submission_payload_requires_nonempty_primary_key():
    # Missing payload → None
    assert llm_review.extract_submission_payload({"payload": None}, "draft") is None
    # Wrong shape → None (defensive; the tool returns only dicts, but
    # downstream code should not crash if something else sneaks in).
    assert (
        llm_review.extract_submission_payload({"payload": "string"}, "draft") is None
    )
    # Missing primary key → None (prefer fallback parser over a
    # half-populated submission that omits the load-bearing field).
    assert (
        llm_review.extract_submission_payload(
            {"payload": {"approach": "x"}}, "draft"
        )
        is None
    )
    # Empty-string primary key → None
    assert (
        llm_review.extract_submission_payload(
            {"payload": {"draft": "  "}}, "draft"
        )
        is None
    )
    # Populated → returns the dict itself (same identity contract as
    # safe_json_loads so downstream normalize_*_result handling matches).
    payload = {"draft": "actual text", "approach": "x"}
    assert llm_review.extract_submission_payload({"payload": payload}, "draft") is payload


# ---------------------------------------------------------------------------
# status classifier
# ---------------------------------------------------------------------------


def test_parse_agent_status_collapses_submit_draft_initial():
    desc = '[Writer] Executing submit_draft({"draft": "hello", "approach": "brief"})'
    parsed = llm_review._parse_agent_status("Writer", desc)
    assert parsed == {"kind": "info", "text": "Draft submitted"}


def test_parse_agent_status_collapses_submit_draft_revised():
    # Presence of changes_made / feedback_declined signals a revision —
    # rendered with a different label so the user can tell at a glance.
    payload = {
        "draft": "updated text",
        "changes_made": ["Added more imagery."],
    }
    desc = f"[Writer] Executing submit_draft({json.dumps(payload)})"
    parsed = llm_review._parse_agent_status("Writer", desc)
    assert parsed == {"kind": "info", "text": "Revised draft submitted"}


def test_parse_agent_status_collapses_submit_review():
    desc = '[Reviewer] Executing submit_review({"key_feedback": "Strong opening"})'
    parsed = llm_review._parse_agent_status("Reviewer", desc)
    assert parsed == {"kind": "info", "text": "Review submitted"}


def test_parse_agent_status_drops_submit_draft_returned():
    # The matching Executing line was already rendered as an info card —
    # dropping the returned: echo avoids a redundant line right below it.
    desc = "[Writer] submit_draft returned: submit_draft received. End your turn now."
    assert llm_review._parse_agent_status("Writer", desc) is None


def test_parse_agent_status_drops_submit_review_returned():
    desc = "[Reviewer] submit_review returned: submit_review received."
    assert llm_review._parse_agent_status("Reviewer", desc) is None


def test_parse_agent_status_keeps_regular_tool_executing():
    # Regression guard: only the submission pseudo-tools should collapse —
    # a real tool call like web search must still render as an executing
    # card with name/args so the activity feed shows what's happening.
    desc = '[Writer] Executing search_web({"query": "capybara facts"})'
    parsed = llm_review._parse_agent_status("Writer", desc)
    assert parsed is not None
    assert parsed["kind"] == "executing"
    assert parsed["name"] == "search_web"


def test_parse_agent_status_content_json_marked_as_rejected_not_submitted():
    """Content-JSON draft must NOT show as 'Draft submitted' — under the strict
    tool-call contract this run actually failed."""
    raw = '{"draft": "some text", "approach": "x"}'
    desc = f"[Writer] {raw}"
    parsed = llm_review._parse_agent_status("Writer", desc)
    assert parsed is not None
    assert parsed["kind"] == "info"
    assert "Submission rejected" in parsed["text"]
    assert "submit_draft" in parsed["text"]
    assert "submitted" not in parsed["text"].lower().replace("submit_draft", "").replace(
        "submit_review", ""
    )


def test_parse_agent_status_content_json_review_marked_as_rejected():
    raw = '{"key_feedback": "the review text", "missing_info": []}'
    desc = f"[Critic] {raw}"
    parsed = llm_review._parse_agent_status("Critic", desc)
    assert parsed is not None
    assert parsed["kind"] == "info"
    assert "Submission rejected" in parsed["text"]
    assert "submit_review" in parsed["text"]


def test_parse_agent_status_handles_malformed_submit_draft_args():
    # When the arg JSON itself is malformed (shouldn't normally happen
    # because the API-side tool-call layer emits well-formed JSON, but
    # we're parsing the same status string the UI sees), the classifier
    # should still collapse to the initial-draft label rather than
    # bubbling an exception up into the activity feed.
    desc = "[Writer] Executing submit_draft({broken json"
    parsed = llm_review._parse_agent_status("Writer", desc)
    # Regex requires a closing ``)``, so this should fall through the
    # exec_match branch entirely and land as speech — which is the
    # correct safety behaviour: malformed status lines don't forge
    # "submitted" info cards.
    assert parsed is None or parsed.get("kind") == "speech"


def test_build_tool_call_status_description_strips_submit_draft_payload():
    """submit_draft status must NOT carry the full draft body — only a stub
    that ``_parse_agent_status`` can still classify as initial vs revised."""
    huge_draft = "x" * 50_000
    args = json.dumps({"draft": huge_draft, "approach": "y" * 5_000})
    desc = llm_review.build_tool_call_status_description("Writer", "submit_draft", args)
    assert huge_draft not in desc
    assert "y" * 5_000 not in desc
    assert len(desc) < 200
    parsed = llm_review._parse_agent_status("Writer", desc)
    assert parsed == {"kind": "info", "text": "Draft submitted"}


def test_build_tool_call_status_description_strips_submit_draft_revise_payload():
    """Revised submission stub must keep the discriminator so the activity
    card still says 'Revised draft submitted'."""
    huge_draft = "z" * 30_000
    args = json.dumps(
        {
            "draft": huge_draft,
            "changes_made": ["a" * 2_000, "b" * 2_000],
            "feedback_declined": [],
        }
    )
    desc = llm_review.build_tool_call_status_description("Writer", "submit_draft", args)
    assert huge_draft not in desc
    assert "a" * 2_000 not in desc
    assert len(desc) < 200
    parsed = llm_review._parse_agent_status("Writer", desc)
    assert parsed == {"kind": "info", "text": "Revised draft submitted"}


def test_build_tool_call_status_description_strips_submit_review_payload():
    """submit_review status must NOT stream the full feedback body."""
    huge_feedback = "f" * 40_000
    args = json.dumps({"key_feedback": huge_feedback, "missing_info": []})
    desc = llm_review.build_tool_call_status_description("Critic", "submit_review", args)
    assert huge_feedback not in desc
    assert len(desc) < 200
    parsed = llm_review._parse_agent_status("Critic", desc)
    assert parsed == {"kind": "info", "text": "Review submitted"}


def test_build_tool_call_status_description_keeps_regular_tool_args():
    """Non-submission tools keep their args — the truncation policy is
    only for submission pseudo-tools whose body is rendered separately."""
    args = '{"query": "find foo bar"}'
    desc = llm_review.build_tool_call_status_description("Agent", "search", args)
    assert "find foo bar" in desc
    assert desc == '[Agent] Executing search({"query": "find foo bar"})'


def test_build_tool_call_status_description_handles_malformed_args():
    """Malformed JSON for submit_draft must not crash and must default to
    the initial-submission stub."""
    desc = llm_review.build_tool_call_status_description(
        "Writer", "submit_draft", "{not json"
    )
    assert "not json" not in desc
    parsed = llm_review._parse_agent_status("Writer", desc)
    assert parsed == {"kind": "info", "text": "Draft submitted"}


# ---------------------------------------------------------------------------
# run_agent_loop submission-tool early break
# ---------------------------------------------------------------------------


class _AppState:
    MODELS: dict = {}


class _FakeApp:
    state = _AppState()


class _FakeRequest:
    """Minimal stand-in for FastAPI ``Request`` used by run_agent_loop.

    ``generate_chat_completion`` is monkey-patched per test below, so the
    only attribute we need is ``app.state.MODELS`` (looked up to pick the
    model dict passed into filter plumbing). Leaving that empty is fine —
    the loop accepts a missing model entry and skips filter work."""

    app = _FakeApp()


@pytest.mark.asyncio
async def test_run_agent_loop_breaks_after_submission_tool(monkeypatch):
    """When the agent calls a tool in submission_tool_names, the loop MUST
    exit without requesting another LLM round — otherwise we waste an
    iteration (or worse, the agent drifts past submission)."""
    request = _FakeRequest()

    # Capture box shared with the pseudo-tool so we can assert it was
    # populated via the tool channel, not via content parsing.
    capture: dict = {"payload": None}
    tools_dict = {
        "submit_draft": llm_review.make_submission_tool(
            spec=llm_review.build_submit_draft_spec(
                phase="compose", include_sources=False
            ),
            capture=capture,
        )
    }

    # Track how many LLM rounds the loop requests. If the submission-tool
    # early break works, this must equal 1 — a second call would mean the
    # loop ignored submission_tool_names and kept going.
    llm_call_count = {"n": 0}

    async def fake_completion(*, request, form_data, user, bypass_filter):
        llm_call_count["n"] += 1
        if llm_call_count["n"] == 1:
            # First round: agent calls submit_draft with the real payload.
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_draft",
                                        "arguments": json.dumps(
                                            {
                                                "draft": 'includes "inner" quotes',
                                                "approach": "brief",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        # Second round should never happen. Return a sentinel so the
        # assertion below has something to complain about if we ever get
        # here (a deliberate malformed response would itself leak).
        return {
            "choices": [
                {"message": {"content": "UNREACHABLE - submission break failed"}}
            ]
        }

    # run_agent_loop imports generate_chat_completion lazily inside the
    # function body — patch via sys.modules so the import inside picks up
    # our fake. Also stub UserModel so run_agent_loop doesn't try to
    # construct the real OWUI model wrapper.
    import types as _types

    fake_chat_module = _types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = fake_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)

    result = await llm_review.run_agent_loop(
        request=request,
        user={"id": "u1", "name": "tester", "email": "t@example.test", "role": "user"},
        model_id="fake-model",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ],
        tools_dict=tools_dict,
        max_iterations=5,
        apply_inlet_filters=False,
        submission_tool_names={"submit_draft"},
    )

    # Exactly one LLM round — the submission break saved us
    # MAX_ITERATIONS-1 wasted calls.
    assert llm_call_count["n"] == 1
    # Capture box received the parsed tool-call args, including the inner
    # unescaped quote that would have broken the old content-JSON path.
    assert capture["payload"] is not None
    assert capture["payload"]["draft"] == 'includes "inner" quotes'
    # The returned ``content`` is whatever text the LLM emitted alongside
    # its tool call (empty here). The caller reads the capture box, so
    # this string is effectively discarded — just confirm it's a string.
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_agent_loop_max_iterations_fallback_can_still_submit(monkeypatch):
    """Max-iter fallback exposes only submission tools and executes the
    submit_* call so capture is populated even after exploration burns
    the budget."""
    request = _FakeRequest()
    capture: dict = {"payload": None}
    dug_deeper: list[str] = []

    async def dig_deeper_tool(**_kwargs) -> str:
        dug_deeper.append("non_submission_called")
        return "should not be called in max-iterations fallback"

    tools_dict = {
        "submit_draft": llm_review.make_submission_tool(
            spec=llm_review.build_submit_draft_spec(
                phase="compose", include_sources=False
            ),
            capture=capture,
        ),
        "dig_deeper": {
            "callable": dig_deeper_tool,
            "spec": {
                "name": "dig_deeper",
                "description": "Non-submission exploration tool — must be pruned in the max-iter fallback.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    }

    iteration_call_tools: list = []
    llm_call_count = {"n": 0}

    async def fake_completion(*, request, form_data, user, bypass_filter):
        llm_call_count["n"] += 1
        iteration_call_tools.append(form_data.get("tools"))
        if llm_call_count["n"] == 1:
            # Regular iteration — agent uses the non-submission tool,
            # consuming its only iteration without submitting.
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_dig",
                                    "type": "function",
                                    "function": {
                                        "name": "dig_deeper",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        # max_iterations reached → this is the fallback request.
        # Agent now submits properly.
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_submit",
                                "type": "function",
                                "function": {
                                    "name": "submit_draft",
                                    "arguments": json.dumps(
                                        {
                                            "draft": "Final draft delivered via max-iter fallback.",
                                            "approach": "Researched first, submitted last.",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    import types as _types

    fake_chat_module = _types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = fake_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)

    await llm_review.run_agent_loop(
        request=request,
        user={
            "id": "u1",
            "name": "tester",
            "email": "t@example.test",
            "role": "user",
        },
        model_id="fake-model",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ],
        tools_dict=tools_dict,
        max_iterations=1,
        apply_inlet_filters=False,
        submission_tool_names={"submit_draft"},
    )

    assert capture["payload"] is not None
    assert capture["payload"]["draft"] == (
        "Final draft delivered via max-iter fallback."
    )
    assert llm_call_count["n"] == 2
    # Regular iteration: all tools. Max-iter fallback: only submission.
    assert {t["function"]["name"] for t in iteration_call_tools[0]} == {
        "submit_draft",
        "dig_deeper",
    }
    assert {t["function"]["name"] for t in iteration_call_tools[1]} == {
        "submit_draft"
    }
    assert dug_deeper == ["non_submission_called"]


@pytest.mark.asyncio
async def test_run_agent_loop_skips_tools_after_submission_in_same_batch(monkeypatch):
    """[submit_draft, side_tool] in one batch: the loop stops after
    submit_draft so the side tool never runs."""
    request = _FakeRequest()

    capture: dict = {"payload": None}
    # Record executions so we can assert the second tool never ran.
    exec_log: list[str] = []

    async def noisy_side_tool(**_kwargs) -> str:
        exec_log.append("side_tool_ran")
        return "side tool should not have run"

    tools_dict = {
        "submit_draft": llm_review.make_submission_tool(
            spec=llm_review.build_submit_draft_spec(
                phase="compose", include_sources=False
            ),
            capture=capture,
        ),
        "side_tool": {
            "callable": noisy_side_tool,
            "spec": {
                "name": "side_tool",
                "description": "Should never execute after submit_draft.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    }

    async def fake_completion(*, request, form_data, user, bypass_filter):
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_submit",
                                "type": "function",
                                "function": {
                                    "name": "submit_draft",
                                    "arguments": json.dumps(
                                        {"draft": "real draft", "approach": "x"}
                                    ),
                                },
                            },
                            {
                                "id": "call_side",
                                "type": "function",
                                "function": {
                                    "name": "side_tool",
                                    "arguments": "{}",
                                },
                            },
                        ],
                    }
                }
            ]
        }

    import types as _types

    fake_chat_module = _types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = fake_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)

    await llm_review.run_agent_loop(
        request=request,
        user={"id": "u1", "name": "tester", "email": "t@example.test", "role": "user"},
        model_id="fake-model",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ],
        tools_dict=tools_dict,
        max_iterations=3,
        apply_inlet_filters=False,
        submission_tool_names={"submit_draft"},
    )

    assert capture["payload"] == {"draft": "real draft", "approach": "x"}
    # The critical assertion: side_tool, listed AFTER submit_draft in the
    # same batch, must NOT have executed.
    assert exec_log == []


@pytest.mark.asyncio
async def test_run_agent_loop_final_iteration_note_mentions_submission_tool(
    monkeypatch,
):
    """Final-iteration prompt names the submission tool, not legacy JSON
    wording."""
    request = _FakeRequest()
    captured_forms: list[dict] = []

    async def fake_completion(*, request, form_data, user, bypass_filter):
        captured_forms.append(form_data)
        return {"choices": [{"message": {"content": "ok"}}]}

    import types as _types

    fake_chat_module = _types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = fake_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)

    await llm_review.run_agent_loop(
        request=request,
        user={"id": "u1", "name": "tester", "email": "t@example.test", "role": "user"},
        model_id="fake-model",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "initial user"},
        ],
        tools_dict={},
        max_iterations=1,
        apply_inlet_filters=False,
        submission_tool_names={"submit_draft"},
    )

    assert len(captured_forms) >= 1
    joined = json.dumps(captured_forms[0]["messages"], ensure_ascii=False)
    assert "submit_draft" in joined
    assert "final JSON" not in joined.lower()
    assert "json response" not in joined.lower()


@pytest.mark.asyncio
async def test_run_agent_loop_reprompts_on_plain_content_under_strict_contract(
    monkeypatch,
):
    """Content-only response triggers a re-prompt naming the submission
    tool, so the next iteration can still populate capture."""
    request = _FakeRequest()

    capture: dict = {"payload": None}
    tools_dict = {
        "submit_draft": llm_review.make_submission_tool(
            spec=llm_review.build_submit_draft_spec(
                phase="compose", include_sources=False
            ),
            capture=capture,
        )
    }

    captured_forms: list[dict] = []
    llm_call_count = {"n": 0}

    async def fake_completion(*, request, form_data, user, bypass_filter):
        llm_call_count["n"] += 1
        captured_forms.append(form_data)
        if llm_call_count["n"] == 1:
            # Plain content, no tool calls — the exact anti-pattern
            # the re-prompt path is designed to recover from.
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Sure, here is the draft as plain text: "
                                "(the model dumped prose instead of using "
                                "the tool channel)"
                            ),
                        }
                    }
                ]
            }
        # Iter 2: agent took the re-prompt's hint and submitted
        # via the tool channel. Capture should land here.
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_submit",
                                "type": "function",
                                "function": {
                                    "name": "submit_draft",
                                    "arguments": json.dumps(
                                        {
                                            "draft": "Recovered via re-prompt.",
                                            "approach": "Submitted on second try.",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    import types as _types

    fake_chat_module = _types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = fake_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)

    await llm_review.run_agent_loop(
        request=request,
        user={"id": "u1", "name": "tester", "email": "t@example.test", "role": "user"},
        model_id="fake-model",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ],
        tools_dict=tools_dict,
        max_iterations=2,
        apply_inlet_filters=False,
        submission_tool_names={"submit_draft"},
    )

    assert capture["payload"] is not None
    assert capture["payload"]["draft"] == "Recovered via re-prompt."
    assert llm_call_count["n"] == 2
    iter2_joined = json.dumps(captured_forms[1]["messages"], ensure_ascii=False)
    assert "submit_draft" in iter2_joined
    # Iter-1 assistant content survives in iter-2 history.
    assert "the model dumped prose" in iter2_joined


@pytest.mark.asyncio
async def test_run_agent_loop_max_iter_fallback_merges_into_trailing_user_turn(
    monkeypatch,
):
    """After the strict-contract re-prompt path leaves a trailing user turn,
    the max-iter fallback must merge into it instead of appending another
    user message — back-to-back user turns get rejected by some providers."""
    request = _FakeRequest()

    capture: dict = {"payload": None}
    tools_dict = {
        "submit_draft": llm_review.make_submission_tool(
            spec=llm_review.build_submit_draft_spec(
                phase="compose", include_sources=False
            ),
            capture=capture,
        )
    }

    captured_forms: list[dict] = []
    llm_call_count = {"n": 0}

    async def fake_completion(*, request, form_data, user, bypass_filter):
        llm_call_count["n"] += 1
        captured_forms.append(form_data)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            f"Plain text response number {llm_call_count['n']} — "
                            "no tool call, forcing the loop to keep re-prompting "
                            "until max_iterations is exhausted."
                        ),
                    }
                }
            ]
        }

    import types as _types

    fake_chat_module = _types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = fake_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)

    await llm_review.run_agent_loop(
        request=request,
        user={"id": "u1", "name": "tester", "email": "t@example.test", "role": "user"},
        model_id="fake-model",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ],
        tools_dict=tools_dict,
        max_iterations=1,
        apply_inlet_filters=False,
        submission_tool_names={"submit_draft"},
    )

    fallback_messages = captured_forms[-1]["messages"]
    roles = [m.get("role") for m in fallback_messages]
    for prev, nxt in zip(roles, roles[1:]):
        assert not (prev == "user" and nxt == "user"), (
            f"Consecutive user turns in fallback request: {roles}. "
            "Some providers reject this — the fallback must merge "
            "into the trailing user turn instead of appending."
        )
    assert fallback_messages[-1].get("role") == "user"
    final_content = fallback_messages[-1].get("content", "")
    assert "Maximum tool iterations reached" in final_content
    assert "submit_draft" in final_content
    assert "You did not call" in final_content


@pytest.mark.asyncio
async def test_run_agent_loop_status_emit_strips_submit_draft_payload(monkeypatch):
    """The status event for submit_draft must NOT carry the full draft body."""
    request = _FakeRequest()

    capture: dict = {"payload": None}
    tools_dict = {
        "submit_draft": llm_review.make_submission_tool(
            spec=llm_review.build_submit_draft_spec(
                phase="compose", include_sources=False
            ),
            capture=capture,
        )
    }

    huge_draft = "x" * 50_000

    async def fake_completion(*, request, form_data, user, bypass_filter):
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_submit",
                                "type": "function",
                                "function": {
                                    "name": "submit_draft",
                                    "arguments": json.dumps(
                                        {"draft": huge_draft, "approach": "y" * 5_000}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    import types as _types

    fake_chat_module = _types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = fake_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)

    emitted_events: list[dict] = []

    async def emitter(event):
        emitted_events.append(event)

    await llm_review.run_agent_loop(
        request=request,
        user={"id": "u1", "name": "tester", "email": "t@example.test", "role": "user"},
        model_id="fake-model",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ],
        tools_dict=tools_dict,
        max_iterations=1,
        apply_inlet_filters=False,
        submission_tool_names={"submit_draft"},
        event_emitter=emitter,
    )

    assert capture["payload"]["draft"] == huge_draft

    exec_status_events = [
        e for e in emitted_events
        if e.get("type") == "status"
        and "Executing submit_draft" in e.get("data", {}).get("description", "")
    ]
    assert exec_status_events, "Expected an Executing submit_draft status event"
    for event in exec_status_events:
        desc = event["data"]["description"]
        assert huge_draft not in desc, (
            f"Status emit leaks full draft body ({len(desc)} chars). "
            "Submission tool args must be stubbed before emission."
        )
        assert len(desc) < 200


def test_default_system_prompt_does_not_instruct_json_only_output():
    """Default SYSTEM_PROMPT names the submit_* tool channel, not
    legacy JSON-format instructions."""
    tools_cls = llm_review.Tools
    user_valves_field = tools_cls.UserValves.model_fields["SYSTEM_PROMPT"]
    default = user_valves_field.default
    assert "JSON format" not in default
    assert "submit_draft" in default and "submit_review" in default


# ---------------------------------------------------------------------------
# strict tool-call contract (no content-JSON functional fallback)
# ---------------------------------------------------------------------------


def test_render_final_html_shows_warning_for_incomplete_revisions():
    """Regression: even after the rounds_data.revisions entry correctly
    shows a failure, the canonical ``final_drafts`` card still renders
    the last-successful-phase's text without any visible signal that
    the revision pipeline did not complete. Users reading only the
    Final drafts section (the intended "top-line answer") would see
    a clean-looking draft with no hint that later rounds failed.

    The fix surfaces the status two ways:
    (a) a full-width warning banner at the top of the Final drafts
        section naming every agent whose last successful round fell
        short of ``num_rounds``.
    (b) a per-card inline badge describing exactly how far each
        incomplete agent got ("showing round N revision — later
        rounds failed" or the all-revisions-failed variant).

    This test asserts both indicators are present when at least one
    agent has ``last_successful_round < num_rounds``."""
    summary = {
        "topic": "irrelevant",
        "num_rounds": 3,
        "model_ids": ["m1", "m2"],
        "agents": [
            {"id": "m1", "model": "m1", "persona": "Writer A"},
            {"id": "m2", "model": "m2", "persona": "Writer B"},
        ],
        "final_drafts": {
            "m1": {
                "persona": "Writer A",
                "model": "m1",
                "draft": "Full-pipeline draft.",
                "sources": [],
                "last_successful_round": 3,
                "final_round_completed": True,
            },
            "m2": {
                "persona": "Writer B",
                "model": "m2",
                "draft": "Stopped-early draft.",
                "sources": [],
                "last_successful_round": 1,
                "final_round_completed": False,
            },
        },
        "rounds": [],
        "render_markdown": False,
        "force_theme": "auto",
        "elapsed_seconds": 0.0,
    }

    html = llm_review._render_final_html(summary)

    # Aggregate banner names the failing agent explicitly. Check for the
    # banner wrapper element (not the class name, which also appears in
    # the stylesheet) plus the name + round number.
    assert '<div class="revision-warning revision-warning-banner">' in html
    assert "Writer B" in html
    assert "last successful round: 1" in html

    # Per-card badge spells out the partial completion state for the
    # failing agent.
    assert "showing round 1 revision" in html

    # The successful agent must NOT get a per-card warning badge — only
    # one card should contain the "showing round N" phrase.
    assert html.count("showing round 1 revision") == 1


def test_render_final_html_no_warning_when_all_revisions_complete():
    """Companion to the previous test: if every agent completed the
    full pipeline, no warning UI should fire — otherwise the banner
    becomes noise and users stop reading it."""
    summary = {
        "topic": "irrelevant",
        "num_rounds": 2,
        "model_ids": ["m1"],
        "agents": [{"id": "m1", "model": "m1", "persona": "Writer A"}],
        "final_drafts": {
            "m1": {
                "persona": "Writer A",
                "model": "m1",
                "draft": "Full-pipeline draft.",
                "sources": [],
                "last_successful_round": 2,
                "final_round_completed": True,
            }
        },
        "rounds": [],
        "render_markdown": False,
        "force_theme": "auto",
        "elapsed_seconds": 0.0,
    }

    html = llm_review._render_final_html(summary)
    # The banner element should NOT be rendered when every agent
    # completed the full pipeline. (Note: the class name itself does
    # appear in the stylesheet — we check for the opening tag that
    # only appears when the banner is actually emitted.)
    assert '<div class="revision-warning revision-warning-banner">' not in html
    assert "showing round" not in html
    # Extra guard: no warning emoji anywhere in the body.
    # The emoji is unique to the two warning surfaces (banner + badge),
    # so its absence confirms no incomplete-revision signal fired.
    assert "\u26a0" not in html


def test_render_final_html_cancelled_mid_revision_shows_cancellation_badge():
    """Regression: when the user aborts a run mid-pipeline the
    cancellation finalize previously built ``partial_final_drafts``
    WITHOUT the ``last_successful_round`` / ``final_round_completed``
    status fields. The iframe then rendered the partial drafts with
    zero indication that the revision pipeline had been interrupted —
    the user would see a round-1-revision draft as if it were the
    intended final output.

    The fix: cancellation path now mirrors the success path's
    final_drafts shape, and _render_final_html uses the
    ``summary.cancelled`` flag to render cancellation-specific wording
    ("cancelled before later rounds" instead of "later rounds failed")
    so the UI does not misrepresent a user-initiated stop as a tool
    failure."""
    summary = {
        "topic": "irrelevant",
        "num_rounds": 3,
        "model_ids": ["m1"],
        "agents": [{"id": "m1", "model": "m1", "persona": "Writer A"}],
        "final_drafts": {
            "m1": {
                "persona": "Writer A",
                "model": "m1",
                "draft": "Round 1 revision draft (partial).",
                "sources": [],
                "last_successful_round": 1,
                "final_round_completed": False,
            }
        },
        "rounds": [],
        "render_markdown": False,
        "force_theme": "auto",
        "elapsed_seconds": 5.0,
        "cancelled": True,
    }

    html = llm_review._render_final_html(summary)

    # Badge wording must use "cancelled" framing, not "failed".
    assert "cancelled before later rounds" in html
    assert "later rounds failed" not in html

    # Aggregate banner uses the cancellation-specific prefix.
    assert "Run cancelled before the revision pipeline finished" in html
    assert "Revision pipeline did not complete" not in html

    # The warning emoji is still present — the signal itself fires
    # regardless of whether the cause is failure or cancellation.
    assert "\u26a0" in html


def test_render_final_html_cancelled_all_rounds_pending_shows_compose_only_badge():
    """Cancellation fired after compose but before any revision round
    landed. The badge should tell the user that the shown draft is the
    compose-phase output and the revision rounds never got to run."""
    summary = {
        "topic": "irrelevant",
        "num_rounds": 3,
        "model_ids": ["m1"],
        "agents": [{"id": "m1", "model": "m1", "persona": "Writer A"}],
        "final_drafts": {
            "m1": {
                "persona": "Writer A",
                "model": "m1",
                "draft": "Compose-only draft (cancelled before revision).",
                "sources": [],
                "last_successful_round": 0,
                "final_round_completed": False,
            }
        },
        "rounds": [],
        "render_markdown": False,
        "force_theme": "auto",
        "elapsed_seconds": 2.0,
        "cancelled": True,
    }

    html = llm_review._render_final_html(summary)
    assert "cancelled before any of the 3 revision rounds" in html
    # Must NOT use the failure wording that implies the rounds
    # themselves broke.
    assert "all 3 revision rounds failed" not in html


def test_render_final_html_mixed_cancel_and_fail_uses_per_agent_labels():
    """Regression: cancellation and tool-failure can coexist in a single
    run — agent A cleanly FAILED its round 2 revise (no submit_draft
    call), then while the aggregation was still processing, the user
    pressed stop and agent B was interrupted mid-round-2. With a single
    ``is_cancelled`` flag driving badge wording, both agents would get
    "cancelled before later rounds" even though A's round 2 actually
    failed BEFORE the cancel.

    The fix: each ``final_drafts`` entry carries a ``last_phase_outcome``
    ("success" | "failed" | "unknown"). The renderer combines that per-
    agent outcome with ``is_cancelled`` so each agent gets the right
    label. This test codifies the exact mixed scenario Codex flagged.
    """
    summary = {
        "topic": "irrelevant",
        "num_rounds": 3,
        "model_ids": ["m_failed", "m_cancelled"],
        "agents": [
            {"id": "m_failed", "model": "m_failed", "persona": "Writer Fail"},
            {
                "id": "m_cancelled",
                "model": "m_cancelled",
                "persona": "Writer Cancel",
            },
        ],
        "final_drafts": {
            # Agent A: round 1 revise succeeded; round 2 revise ran to
            # completion without a submit_draft call. Outcome = "failed".
            # When cancel fires afterwards the agent's label must stay
            # "failed" — the cancel didn't cause the incompletion here.
            "m_failed": {
                "persona": "Writer Fail",
                "model": "m_failed",
                "draft": "Round 1 revised draft (round 2 failed).",
                "sources": [],
                "last_successful_round": 1,
                "final_round_completed": False,
                "last_phase_outcome": "failed",
            },
            # Agent B: round 1 revise succeeded; still running when the
            # user pressed stop. Outcome stayed "unknown" because the
            # inline setter never ran. During a cancelled run this
            # maps to "cancelled".
            "m_cancelled": {
                "persona": "Writer Cancel",
                "model": "m_cancelled",
                "draft": "Round 1 revised draft (round 2 in flight).",
                "sources": [],
                "last_successful_round": 1,
                "final_round_completed": False,
                "last_phase_outcome": "unknown",
            },
        },
        "rounds": [],
        "render_markdown": False,
        "force_theme": "auto",
        "elapsed_seconds": 10.0,
        "cancelled": True,
    }

    html = llm_review._render_final_html(summary)

    # Agent A's badge must say the round 2 FAILED, and indicate that
    # the cancel then prevented a retry. The old single-flag renderer
    # would have mislabelled this as "cancelled before later rounds".
    assert "round 2 failed; run cancelled before retry" in html
    # Agent B was actively running at cancel — it should stay as
    # "cancelled", not be promoted to a failure.
    assert "cancelled before later rounds" in html

    # Aggregate banner must acknowledge BOTH outcomes in a mixed run,
    # not pick one and hide the other.
    assert "mixed failure and cancellation outcomes" in html
    # Detail rows must annotate each agent with its own sub_kind.
    assert "Writer Fail (failed" in html
    assert "Writer Cancel (cancelled" in html


def test_reset_succeeded_outcomes_preserves_failed_markers():
    """Regression: ``reset_phase_outcomes`` (called at the start of
    each new compose / revise phase) must NOT wipe a ``"failed"``
    marker. Before this fix, the reset indiscriminately set every
    agent back to ``"unknown"`` — meaning a round-1 revise failure
    was invisible to the cancel-time renderer by the time the user
    pressed stop mid round 2. The renderer would then pick the
    "unknown + is_cancelled" branch and mislabel the real failure
    as a cancellation.

    The fix: only ``"success"`` gets reset. ``"failed"`` survives
    across phase boundaries so a later cancellation cannot hide
    a pre-existing failure."""
    outcome_map = {
        "succeeded": "success",
        "failed": "failed",
        "untouched": "unknown",
    }
    llm_review.reset_succeeded_outcomes(
        outcome_map, ["succeeded", "failed", "untouched"]
    )
    # "success" — reset so the new phase starts fresh and will be
    # re-stamped by its inline setter.
    assert outcome_map["succeeded"] == "unknown"
    # "failed" — preserved; the whole point of the fix. If this
    # assertion ever starts failing, the cancel-time renderer will
    # regress to mislabelling pre-cancel failures as cancellations.
    assert outcome_map["failed"] == "failed"
    # "unknown" — left alone; nothing to reset.
    assert outcome_map["untouched"] == "unknown"


def test_reset_succeeded_outcomes_idempotent_on_repeated_calls():
    """Calling the reset twice without a new phase in between must
    be a no-op. Otherwise a spurious double-reset (e.g. from a
    refactor that accidentally invokes it both at cross-review
    start AND revise start) could erase a round's status mid-round.
    """
    outcome_map = {"a": "failed", "b": "success"}
    llm_review.reset_succeeded_outcomes(outcome_map, ["a", "b"])
    first_pass = dict(outcome_map)
    llm_review.reset_succeeded_outcomes(outcome_map, ["a", "b"])
    assert outcome_map == first_pass


def test_reset_succeeded_outcomes_tolerates_missing_agent_ids():
    """An agent id passed to the reset but absent from the outcome
    map must not crash. The Tools-method closure populates the map
    during initialisation but defensive coding here means a future
    caller adding an agent after init won't break the reset path."""
    outcome_map: dict[str, str] = {}
    # Should not raise.
    llm_review.reset_succeeded_outcomes(outcome_map, ["ghost"])
    assert outcome_map == {}


@pytest.mark.asyncio
async def test_per_agent_outcome_pattern_finalizes_failed_on_non_cancel_exception():
    """Regression: a non-cancel exception inside compose_draft /
    revise_draft (e.g. ``ensure_tools`` blowup, transport error,
    unexpected type crash) must bump ``per_agent_last_phase_outcome``
    from its initial ``"unknown"`` to ``"failed"``. Without this, an
    exception that fires shortly before the user presses stop would
    leave the outcome at ``"unknown"`` — and the cancel-time renderer
    would then mislabel the real tool crash as a cancellation.

    compose_draft / revise_draft are closures inside the Tools method,
    so we exercise the exact try/except pattern they now use to lock
    the semantic in: any non-CancelledError exception path finalises
    the outcome; CancelledError re-raises cleanly and leaves it at
    ``"unknown"`` (see the companion test below)."""
    outcome: dict[str, str] = {"agent_x": "unknown"}

    async def crashing_phase():
        raise RuntimeError("ensure_tools crashed before submission")

    with pytest.raises(RuntimeError):
        try:
            await crashing_phase()
            # Normal exit path bumps "unknown" → "failed" too (the
            # submission-missing case). Not reached here because
            # crashing_phase raises.
            if outcome["agent_x"] == "unknown":
                outcome["agent_x"] = "failed"
        except asyncio.CancelledError:
            raise
        except BaseException:
            # This is the branch Codex flagged was missing: ANY non-
            # cancel exception must finalise the outcome before
            # re-raising so the cancel renderer sees the right signal.
            if outcome["agent_x"] == "unknown":
                outcome["agent_x"] = "failed"
            raise

    assert outcome["agent_x"] == "failed", (
        "Non-cancel exception must bump outcome to 'failed' — "
        "otherwise a cancel-time render would mislabel a real crash "
        "as a cancellation."
    )


@pytest.mark.asyncio
async def test_per_agent_outcome_pattern_leaves_unknown_on_cancel():
    """Companion: ``asyncio.CancelledError`` is the ONE exception that
    must NOT touch the outcome. ``"unknown"`` is the renderer's signal
    for "actively running at cancel" — overwriting it to ``"failed"``
    would erase that distinction and regress the mixed-case fix."""
    outcome: dict[str, str] = {"agent_x": "unknown"}

    async def cancelled_phase():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        try:
            await cancelled_phase()
            if outcome["agent_x"] == "unknown":
                outcome["agent_x"] = "failed"
        except asyncio.CancelledError:
            # Critical: this branch MUST raise without touching outcome.
            raise
        except BaseException:
            if outcome["agent_x"] == "unknown":
                outcome["agent_x"] = "failed"
            raise

    assert outcome["agent_x"] == "unknown", (
        "CancelledError must leave outcome at 'unknown' so the "
        "renderer's 'cancelled' branch fires. Overwriting here "
        "would collapse the cancel/fail distinction the renderer "
        "relies on to produce accurate mixed-case labels."
    )


def test_render_final_html_pre_cancel_failure_at_round_1_revise_variant():
    """Companion to the mixed case: when compose succeeded but the very
    first revision round FAILED before a cancellation, the badge must
    say the round failed — not 'cancelled before round 1 could
    complete', because round 1 did run to completion (it just didn't
    submit)."""
    summary = {
        "topic": "irrelevant",
        "num_rounds": 2,
        "model_ids": ["m1"],
        "agents": [{"id": "m1", "model": "m1", "persona": "Writer A"}],
        "final_drafts": {
            "m1": {
                "persona": "Writer A",
                "model": "m1",
                "draft": "Compose-only (round 1 failed).",
                "sources": [],
                "last_successful_round": 0,
                "final_round_completed": False,
                "last_phase_outcome": "failed",
            }
        },
        "rounds": [],
        "render_markdown": False,
        "force_theme": "auto",
        "elapsed_seconds": 3.0,
        "cancelled": True,
    }

    html = llm_review._render_final_html(summary)
    # Must use the "round 1 revision failed; run cancelled before retry"
    # wording — NOT the "cancelled before any revision rounds" variant.
    assert "round 1 revision failed; run cancelled before retry" in html
    assert "cancelled before any of the" not in html


def test_render_final_html_shows_all_revisions_failed_variant():
    """When an agent's compose succeeded but every revision round failed,
    the per-card badge should clearly state "all N revision rounds failed"
    so the user understands the shown text is the COMPOSE draft, not a
    revised version."""
    summary = {
        "topic": "irrelevant",
        "num_rounds": 2,
        "model_ids": ["m1"],
        "agents": [{"id": "m1", "model": "m1", "persona": "Writer A"}],
        "final_drafts": {
            "m1": {
                "persona": "Writer A",
                "model": "m1",
                "draft": "Compose-only draft.",
                "sources": [],
                "last_successful_round": 0,
                "final_round_completed": False,
            }
        },
        "rounds": [],
        "render_markdown": False,
        "force_theme": "auto",
        "elapsed_seconds": 0.0,
    }

    html = llm_review._render_final_html(summary)
    assert "all 2 revision rounds failed" in html


def test_normalize_revise_result_with_error_fallback_does_not_echo_original():
    """Regression: revise failures must not masquerade as minor revisions.

    Pre-fix, the revise-phase aggregation passed ``current_drafts[aid]["draft"]``
    (the previous round's text) as the ``fallback_draft`` to
    ``normalize_revise_result`` whenever the submission was missing. That
    made a failed revision indistinguishable in ``rounds_data`` from a
    successful revision where the agent intentionally kept the draft
    unchanged — same text, empty ``changes_made`` / ``feedback_declined``,
    empty sources. Users would see a "completed" round with no change
    instead of a clearly-flagged failure.

    The fix: pass an explicit error message as the fallback so the
    revisions entry shows the failure verbatim. This test locks that in.
    """
    original_draft = "The previous round's real draft text."
    error_fallback = (
        "Agent did not submit via submit_draft. Response: (empty content)"
    )

    # Failed revision (parsed is None) MUST surface the error, not the
    # original draft. Otherwise the user cannot tell a silent failure
    # from a deliberate no-change revision.
    failed = llm_review.normalize_revise_result(None, error_fallback)
    assert failed["draft"] == error_fallback
    assert original_draft not in failed["draft"], (
        "Revise failure fallback must NOT echo the previous draft — "
        "doing so makes failures look like successful near-zero-change "
        "revisions in rounds_data."
    )
    assert failed["changes_made"] == []
    assert failed["feedback_declined"] == []

    # Successful revision: parsed's own draft wins, fallback is unused
    # (we can pass any sentinel string and it should be ignored).
    parsed = {
        "draft": "The agent's actual revised text.",
        "changes_made": ["Tightened the opening paragraph."],
        "feedback_declined": [],
        "sources": [],
    }
    succeeded = llm_review.normalize_revise_result(parsed, "IGNORED_FALLBACK")
    assert succeeded["draft"] == "The agent's actual revised text."
    assert "IGNORED_FALLBACK" not in succeeded["draft"]
    assert succeeded["changes_made"] == ["Tightened the opening paragraph."]


def test_advance_last_successful_round_if_chained_contract():
    """Gate semantics + bool return for ``advance_last_successful_round_if_chained``."""
    advance = llm_review.advance_last_successful_round_if_chained

    state = {"a": 0}
    assert advance(state, "a", 1) is True
    assert state["a"] == 1

    state = {"a": -1}
    assert advance(state, "a", 1) is False
    assert state["a"] == -1

    state = {"a": 0}
    assert advance(state, "a", 2) is False
    assert state["a"] == 0

    state = {"a": 1}
    assert advance(state, "a", 1) is False
    assert state["a"] == 1

    state_missing: dict[str, int] = {}
    assert advance(state_missing, "missing", 1) is False
    assert "missing" not in state_missing


def _make_final_drafts_summary(
    *,
    draft: str,
    last_successful_round: int,
    last_phase_outcome: str,
    num_rounds: int,
    submitted_rounds: list[int] | None = None,
    cancelled: bool = False,
    final_round_completed: bool = False,
) -> dict:
    entry: dict = {
        "persona": "Writer A",
        "model": "m_a",
        "draft": draft,
        "sources": [],
        "last_successful_round": last_successful_round,
        "final_round_completed": final_round_completed,
        "last_phase_outcome": last_phase_outcome,
    }
    if submitted_rounds is not None:
        entry["submitted_rounds"] = submitted_rounds
    summary: dict = {
        "topic": "test scenario",
        "num_rounds": num_rounds,
        "model_ids": ["m_a"],
        "agents": [{"id": "m_a", "model": "m_a", "persona": "Writer A"}],
        "final_drafts": {"m_a": entry},
        "rounds": [],
        "render_markdown": False,
        "force_theme": "auto",
        "elapsed_seconds": 5.0,
    }
    if cancelled:
        summary["cancelled"] = True
    return summary


def test_render_final_html_compose_fail_then_revise_success_body_matches_label():
    """When compose fails and revise later submits, body must be the compose error and badge must say 'composition failed'."""
    error_marker = "Agent did not submit via submit_draft. Response: (empty content)"
    html = llm_review._render_final_html(
        _make_final_drafts_summary(
            draft=error_marker,
            last_successful_round=-1,
            last_phase_outcome="success",
            num_rounds=1,
        )
    )
    assert "composition failed" in html
    assert error_marker in html
    assert "Revision pipeline did not complete" in html
    assert "Writer A (failed, last successful round: -1)" in html


def test_render_final_html_compose_ok_revise_round1_fail_revise_round2_success_body_matches():
    """Without submitted_rounds, broken-chain at round 1 falls back to legacy 'all N revision rounds failed'."""
    compose_body = "The compose phase draft sentence."
    html = llm_review._render_final_html(
        _make_final_drafts_summary(
            draft=compose_body,
            last_successful_round=0,
            last_phase_outcome="success",
            num_rounds=2,
        )
    )
    assert compose_body in html
    assert "showing compose draft — all 2 revision rounds failed" in html


def test_render_final_html_unchained_revise_success_does_not_claim_all_failed():
    """submitted_rounds with broken chain must surface 'round X submitted but could not extend the chain'."""
    compose_body = "Compose phase draft."
    html = llm_review._render_final_html(
        _make_final_drafts_summary(
            draft=compose_body,
            last_successful_round=0,
            last_phase_outcome="success",
            num_rounds=2,
            submitted_rounds=[0, 2],
        )
    )
    assert "all 2 revision rounds failed" not in html
    assert "round 1 revision failed" in html
    assert "round 2 submitted" in html
    assert "could not extend the chain" in html
    assert compose_body in html


def test_render_final_html_unchained_compose_failure_with_late_submits():
    """Compose failed but both revise rounds submitted: badge must list 'rounds 1, 2 submitted'."""
    error_marker = "Agent did not submit via submit_draft. Response: (no draft)"
    html = llm_review._render_final_html(
        _make_final_drafts_summary(
            draft=error_marker,
            last_successful_round=-1,
            last_phase_outcome="success",
            num_rounds=2,
            submitted_rounds=[1, 2],
        )
    )
    assert "composition failed" in html
    assert "rounds 1, 2 submitted" in html
    assert "could not extend the chain" in html


def test_render_final_html_cancel_with_unchained_submission_surfaces_audit():
    """Cancel + broken-chain submit must surface 'cancelled mid-pipeline' with unchained suffix."""
    html = llm_review._render_final_html(
        _make_final_drafts_summary(
            draft="Compose body pinned by chained_drafts.",
            last_successful_round=0,
            last_phase_outcome="unknown",
            num_rounds=2,
            submitted_rounds=[0, 2],
            cancelled=True,
        )
    )
    assert "cancelled before any of the 2 revision rounds could complete" not in html
    assert "round 1 revision failed" in html
    assert "run cancelled mid-pipeline" in html
    assert "round 2 submitted" in html
    assert "could not extend the chain" in html


def test_render_final_html_failed_pre_cancel_with_unchained_uses_mid_pipeline():
    """failed + cancelled + unchained must say 'mid-pipeline' not 'before retry' to avoid contradicting the suffix."""
    html = llm_review._render_final_html(
        _make_final_drafts_summary(
            draft="Compose body pinned by chained_drafts.",
            last_successful_round=0,
            last_phase_outcome="failed",
            num_rounds=2,
            submitted_rounds=[0, 2],
            cancelled=True,
        )
    )
    assert "cancelled before retry" not in html
    assert "run cancelled mid-pipeline" in html
    assert "round 2 submitted" in html
    assert "could not extend the chain" in html


def test_render_final_html_failed_pre_cancel_without_unchained_keeps_before_retry():
    """Backward-compat: failed + cancelled with no unchained submission keeps legacy 'cancelled before retry'."""
    html = llm_review._render_final_html(
        _make_final_drafts_summary(
            draft="Compose body.",
            last_successful_round=0,
            last_phase_outcome="failed",
            num_rounds=2,
            submitted_rounds=[0],
            cancelled=True,
        )
    )
    assert "round 1 revision failed; run cancelled before retry" in html
    assert "mid-pipeline" not in html
    assert "could not extend the chain" not in html


def test_render_final_html_cancel_no_unchained_keeps_legacy_wording():
    """Backward-compat: cancel with no unchained submission keeps legacy 'cancelled before any rounds could complete'."""
    html = llm_review._render_final_html(
        _make_final_drafts_summary(
            draft="Compose body.",
            last_successful_round=0,
            last_phase_outcome="unknown",
            num_rounds=2,
            submitted_rounds=[0],
            cancelled=True,
        )
    )
    assert "cancelled before any of the 2 revision rounds could complete" in html
    assert "could not extend the chain" not in html


def test_render_final_html_legacy_payload_without_submitted_rounds_keeps_old_wording():
    """Summaries without submitted_rounds fall back to legacy 'all N rounds failed' (contiguous-chain assumption)."""
    html = llm_review._render_final_html(
        _make_final_drafts_summary(
            draft="Compose body",
            last_successful_round=0,
            last_phase_outcome="success",
            num_rounds=2,
        )
    )
    assert "all 2 revision rounds failed" in html
    assert "could not extend the chain" not in html


@pytest.mark.asyncio
async def test_missing_submission_does_not_salvage_content_json(monkeypatch):
    """Regression guard for the strict tool-call contract.

    Before v0.5.0 the compose/review/revise call sites fell back to
    ``safe_json_loads(content, ...)`` whenever the capture box was
    empty. That was a direct contradiction with the default prompt
    (which now says 'never dump JSON into your content') and it also
    re-opened the very unescaped-quote fragility this migration was
    designed to eliminate. The salvage path has been removed: missing
    capture means ``parsed`` stays None and the state ends up 'failed'
    with a clear 'did not submit via submit_draft' label.

    This test simulates an agent that emits a fully-formed JSON in
    content (what a flaky tool-calling model might do) and verifies
    that ``extract_submission_payload`` does NOT latch onto it — the
    draft must come exclusively from the tool channel."""
    request = _FakeRequest()

    capture: dict = {"payload": None}
    tools_dict = {
        "submit_draft": llm_review.make_submission_tool(
            spec=llm_review.build_submit_draft_spec(
                phase="compose", include_sources=False
            ),
            capture=capture,
        )
    }

    # Agent ignores submit_draft and dumps the payload as content JSON —
    # the exact anti-pattern the new contract forbids.
    async def fake_completion(*, request, form_data, user, bypass_filter):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"draft": "I should have called submit_draft", "approach": "x"}
                        ),
                    }
                }
            ]
        }

    import types as _types

    fake_chat_module = _types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = fake_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)

    content = await llm_review.run_agent_loop(
        request=request,
        user={"id": "u1", "name": "tester", "email": "t@example.test", "role": "user"},
        model_id="fake-model",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ],
        tools_dict=tools_dict,
        max_iterations=1,
        apply_inlet_filters=False,
        submission_tool_names={"submit_draft"},
    )

    # The capture box is the ONLY authoritative path. Since the agent
    # didn't call submit_draft, the capture stays empty and
    # extract_submission_payload must return None — even though the
    # content contains a well-formed JSON dict with a valid 'draft'
    # key. This is the strict-contract assertion.
    assert capture["payload"] is None
    assert llm_review.extract_submission_payload(capture, "draft") is None

    # The content is still returned for error-labelling purposes but
    # must not be silently parsed as a draft by downstream code.
    assert "should have called submit_draft" in content
