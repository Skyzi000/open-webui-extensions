"""Integration tests for sub_agent loop compaction and ref externalization."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest

tools_dir = Path(__file__).resolve().parents[2] / "tools"
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

import sub_agent  # noqa: E402

# Imported at module level (before the tools-conftest stubs replace
# ``open_webui.utils`` per test) so the Core boundary test exercises the
# REAL in-place system-prompt mutation, not a lookalike.
from open_webui.utils.payload import apply_system_prompt_to_body  # noqa: E402

CORE_SYSTEM_PROMPT = "MODEL_SYSTEM_PROMPT sentinel"


class _AppState:
    MODELS: dict = {}


class _FakeApp:
    state = _AppState()


class _FakeRequest:
    app = _FakeApp()


def _text_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _tool_call_response(call_id: str, name: str, arguments: dict | None = None) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments or {}),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _with_usage(response: dict, prompt_tokens: int) -> dict:
    return {**response, "usage": {"prompt_tokens": prompt_tokens}}


class _LoopHarness:
    def __init__(self, monkeypatch, *, apply_core_system_prompt: bool = False):
        self.calls: list[dict] = []
        self.script: list[dict] = []
        self.summary_calls: list[dict] = []
        self.statuses: list[str] = []
        self.apply_core_system_prompt = apply_core_system_prompt
        monkeypatch.setattr(
            sub_agent, "_LoopRunState", sub_agent._LoopRunState
        )

    async def completion(self, *, request, form_data, user, bypass_filter):
        if self.apply_core_system_prompt:
            await apply_system_prompt_to_body(
                CORE_SYSTEM_PROMPT,
                form_data,
                form_data.get("metadata"),
                user,
            )
        self.calls.append(form_data)
        task = (form_data.get("metadata") or {}).get("task")
        if task == "sub_agent_summary":
            self.summary_calls.append(form_data)
            script = self.summary_script
        else:
            main_so_far = [
                c
                for c in self.calls
                if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
            ]
            script = self.script
            index = len(main_so_far) - 1
            response = script[min(index, len(script) - 1)]
            if isinstance(response, Exception):
                raise response
            return response
        index = len(self.summary_calls) - 1
        response = script[min(index, len(script) - 1)]
        if isinstance(response, Exception):
            raise response
        return response

    async def emitter(self, event: dict) -> None:
        if event.get("type") == "status":
            self.statuses.append(event["data"]["description"])


def _install(monkeypatch, harness: _LoopHarness):
    fake_chat_module = types.ModuleType("open_webui.utils.chat")
    fake_chat_module.generate_chat_completion = harness.completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", fake_chat_module)


def _big_tool_result(prefix: str, lines: int = 1200) -> str:
    return f"{prefix}\n" + ("substantial result line with detail\n" * lines) + "tail sentinel\n"


def _options(**kwargs):
    compaction = sub_agent.LoopCompactionOptions(
        enabled=kwargs.get("compaction_enabled", True),
        threshold_tokens=kwargs.get("threshold", 80_000),
        summary_model=kwargs.get("summary_model", ""),
    )
    large = sub_agent.LargeToolResultOptions(
        mode=kwargs.get("mode", "ref_exec"),
        threshold_tokens=kwargs.get("large_threshold", 200),
    )
    return compaction, large


async def _run(monkeypatch, harness, *, tools_dict=None, messages=None, **kwargs):
    _install(monkeypatch, harness)
    compaction, large = _options(**kwargs)
    return await sub_agent.run_sub_agent_loop(
        request=_FakeRequest(),
        user={"id": "u1", "role": "user"},
        model_id="m",
        messages=messages
        or [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ],
        tools_dict=tools_dict or {},
        max_iterations=kwargs.get("max_iterations", 10),
        event_emitter=harness.emitter,
        apply_inlet_filters=False,
        compaction=compaction,
        large_results=large,
    )


@pytest.mark.asyncio
async def test_compaction_triggers_and_preserves_prefix(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("ROUND-A")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]
    harness.summary_script = [_text_response("SUMMARY: rounds A and B completed")]

    result = await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=10,
    )

    assert result == "done"
    assert len(harness.summary_calls) >= 1
    first_summary_body = harness.summary_calls[0]
    assert first_summary_body["messages"][-1]["role"] == "user"
    assert "AGENT-LOOP COMPACTION SUMMARY" in first_summary_body["messages"][-1]["content"]

    if len(harness.summary_calls) > 1:
        second_summary_body = harness.summary_calls[1]
        prior_envelope_source = second_summary_body["messages"][1]["content"]
        assert "<agent_loop_compaction_context>" in prior_envelope_source

    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    compacted_send = main_calls[-1]
    payload_messages = compacted_send["messages"]
    assert payload_messages[0]["content"] == "sys"
    task_user = payload_messages[1]
    assert task_user["role"] == "user"
    assert task_user["content"].startswith("task")
    assert task_user["content"].count("<agent_loop_compaction_context>") == 1
    assert "SUMMARY: rounds A and B completed" in task_user["content"]
    tool_messages = [m for m in payload_messages if m.get("role") == "tool"]
    assert 2 <= len(tool_messages) <= 4


@pytest.mark.asyncio
async def test_preview_restores_omitted_middle_via_reader(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("IMPORTANT-HEADER")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "agent_ref_exec", {"command": "ls tool"}),
        _text_response("done"),
    ]

    result = await _run(
        monkeypatch, harness, tools_dict=tools, mode="ref_exec", large_threshold=200
    )

    assert result == "done"
    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    second_send = main_calls[1]
    tool_message = next(m for m in second_send["messages"] if m.get("role") == "tool")
    preview = tool_message["content"]
    assert preview != big
    head, opening, rest = preview.partition("\n<agent_ref_truncated>")
    assert opening
    payload, _closing, tail = rest.partition("</agent_ref_truncated>\n")
    marker = json.loads(payload)
    assert marker["next"].startswith("tail -c +")

    third_send_tool = None
    assert len(main_calls) == 3

    store = sub_agent.RefRunStore()
    import hashlib as _hashlib

    digest = _hashlib.sha256(big.encode()).hexdigest()
    from owui_ext.shared import ref_exec as rx

    store.intern_tool_text(
        big,
        rx.RefCatalogEntry(
            manifest=rx.RefManifest(
                ref=f"tool:{digest}", utf8_bytes=len(big.encode()), sha256=digest
            ),
            source=rx.ZeroCopySourceHandle(text=big),
        ),
    )
    import tiktoken

    reader = rx.build_ref_reader(
        store, threshold_tokens=200, encoder=tiktoken.get_encoding("cl100k_base")
    )
    command = marker["next"]
    middle_parts = []
    for _ in range(100):
        response = await reader(command)
        visible, next_command = _visible_and_next(response)
        middle_parts.append(visible)
        if next_command is None:
            break
        command = next_command
    else:
        pytest.fail("recovery did not terminate")
    recovered = (head + "".join(middle_parts) + tail).encode()
    assert recovered == big.encode()


def _visible_and_next(result: str):
    for tag in ("agent_ref_range", "agent_ref_truncated"):
        opening = f"\n<{tag}>"
        if opening in result:
            visible, encoded_marker = result.split(opening, 1)
            marker = json.loads(encoded_marker.removesuffix(f"</{tag}>"))
            return visible, marker.get("next")
    return result, None


@pytest.mark.asyncio
async def test_projection_is_byte_identical_with_or_without_reader(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("STABLE")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _text_response("done"),
    ]

    await _run(monkeypatch, harness, tools_dict=tools, mode="ref_exec", large_threshold=200)
    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    with_reader = next(
        m for m in main_calls[1]["messages"] if m.get("role") == "tool"
    )["content"]

    harness2 = _LoopHarness(monkeypatch)
    harness2.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _text_response("done"),
    ]

    original_reader_check = sub_agent._reader_available_in_payload
    check_state = {"ref_bearing_sends": 0}

    def stripping_check(form_data):
        messages = form_data.get("messages")
        has_tool_payload = isinstance(messages, list) and any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and isinstance(m.get("content"), str)
            and len(m["content"]) > 1000
            for m in messages
        )
        if has_tool_payload:
            check_state["ref_bearing_sends"] += 1
            if check_state["ref_bearing_sends"] > 1:
                stripped = {**form_data, "tools": []}
                return original_reader_check(stripped)
        return original_reader_check(form_data)

    monkeypatch.setattr(sub_agent, "_reader_available_in_payload", stripping_check)
    await _run(monkeypatch, harness2, tools_dict=tools, mode="ref_exec", large_threshold=200)
    monkeypatch.setattr(sub_agent, "_reader_available_in_payload", original_reader_check)
    main_calls2 = [
        c for c in harness2.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    without_reader_after_first = next(
        m for m in main_calls2[1]["messages"] if m.get("role") == "tool"
    )["content"]

    assert with_reader == without_reader_after_first


@pytest.mark.asyncio
async def test_provider_error_returns_clear_error_without_retry(monkeypatch):
    from starlette.responses import PlainTextResponse

    harness = _LoopHarness(monkeypatch)
    harness.script = [PlainTextResponse("upstream exploded", status_code=500)]

    result = await _run(monkeypatch, harness)

    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    assert len(main_calls) == 1
    assert "500" in result and "upstream exploded" in result


@pytest.mark.asyncio
async def test_parallel_loops_do_not_share_state(monkeypatch):
    import asyncio as _asyncio

    big_a = _big_tool_result("TASK-A")
    big_b = _big_tool_result("TASK-B")

    async def tool_a(**kwargs):
        return big_a

    async def tool_b(**kwargs):
        return big_b

    shared_tools = {
        "tool_a": {"spec": {"name": "tool_a"}, "callable": tool_a},
        "tool_b": {"spec": {"name": "tool_b"}, "callable": tool_b},
    }

    harnesses = []
    for name in ("tool_a", "tool_b"):
        harness = _LoopHarness(monkeypatch)
        harness.script = [_tool_call_response("c1", name), _text_response(f"done-{name}")]
        harnesses.append(harness)

    async def run_one(harness, name):
        _install(monkeypatch, harness)
        compaction, large = _options(mode="ref_exec", large_threshold=200)
        return await sub_agent.run_sub_agent_loop(
            request=_FakeRequest(),
            user={"id": "u1", "role": "user"},
            model_id="m",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"task {name}"},
            ],
            tools_dict=shared_tools,
            max_iterations=5,
            apply_inlet_filters=False,
            compaction=compaction,
            large_results=large,
        )

    results = await _asyncio.gather(run_one(harnesses[0], "tool_a"), run_one(harnesses[1], "tool_b"))
    assert results == ["done-tool_a", "done-tool_b"]

    previews = []
    for harness in harnesses:
        main_calls = [
            c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
        ]
        previews.append(
            next(m for m in main_calls[1]["messages"] if m.get("role") == "tool")["content"]
        )
    assert previews[0] != previews[1]


@pytest.mark.asyncio
async def test_orphan_tool_messages_refuse_compaction(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    harness.script = [_text_response("quick")]

    orphan_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "tool", "tool_call_id": "ghost", "content": "orphan"},
    ]

    await _run(
        monkeypatch,
        harness,
        messages=orphan_messages,
        mode="raw",
        threshold=1,
    )

    assert harness.summary_calls == []


@pytest.mark.asyncio
async def test_summarizer_exhaustion_returns_clear_error(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("ROUND-A")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
    ]
    harness.summary_script = [
        types.SimpleNamespace(status_code=500),
        types.SimpleNamespace(status_code=503),
        types.SimpleNamespace(status_code=502),
    ]

    result = await _run(
        monkeypatch, harness, tools_dict=tools, mode="raw", threshold=600
    )

    assert result.startswith("Sub-agent compaction summary failed")
    assert len(harness.summary_calls) == 3


@pytest.mark.asyncio
async def test_compaction_and_projection_are_independent(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("INDEP")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [_tool_call_response("c1", "noisy"), _text_response("done")]

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="ref_exec",
        compaction_enabled=False,
        large_threshold=200,
    )
    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    preview = next(m for m in main_calls[1]["messages"] if m.get("role") == "tool")["content"]
    assert "agent_ref_truncated" in preview

    harness2 = _LoopHarness(monkeypatch)
    harness2.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]
    harness2.summary_script = [_text_response("summary text")]

    await _run(
        monkeypatch, harness2, tools_dict=tools, mode="raw", threshold=600
    )
    assert len(harness2.summary_calls) >= 1
    main_calls2 = [
        c for c in harness2.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    raw_tool = next(m for m in main_calls2[4]["messages"] if m.get("role") == "tool")
    assert raw_tool["content"] == big


@pytest.mark.asyncio
async def test_reader_unavailable_falls_back_to_raw_not_truncate(monkeypatch, caplog):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("NOREADER")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [_tool_call_response("c1", "noisy"), _text_response("done")]

    original_check = sub_agent._reader_available_in_payload
    monkeypatch.setattr(
        sub_agent,
        "_reader_available_in_payload",
        lambda form_data: (False, "reader schema removed from final payload"),
    )

    with caplog.at_level(logging.ERROR, logger=sub_agent.log.name):
        await _run(monkeypatch, harness, tools_dict=tools, mode="ref_exec", large_threshold=200)

    monkeypatch.setattr(sub_agent, "_reader_available_in_payload", original_check)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and "agent_ref_exec unavailable" in r.message]
    assert len(errors) == 1

    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    tool_message = next(m for m in main_calls[1]["messages"] if m.get("role") == "tool")
    assert tool_message["content"] == big


@pytest.mark.asyncio
async def test_filters_run_once_per_send(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("FILTERS")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [_tool_call_response("c1", "noisy"), _text_response("done")]

    inlet_calls = {"n": 0}
    finalize_calls = {"n": 0}
    original_inlet = sub_agent.apply_inlet_filters_if_enabled
    original_finalize = sub_agent.finalize_model_request

    async def counting_inlet(filter_pipeline, request, form_data, extra_params):
        inlet_calls["n"] += 1
        return await original_inlet(filter_pipeline, request, form_data, extra_params)

    async def counting_finalize(filter_pipeline, request, form_data, extra_params):
        finalize_calls["n"] += 1
        return await original_finalize(filter_pipeline, request, form_data, extra_params)

    monkeypatch.setattr(sub_agent, "apply_inlet_filters_if_enabled", counting_inlet)
    monkeypatch.setattr(sub_agent, "finalize_model_request", counting_finalize)

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="ref_exec",
        large_threshold=200,
        apply_inlet_filters=True,
    )

    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    assert inlet_calls["n"] == len(main_calls)
    assert finalize_calls["n"] == len(main_calls)


@pytest.mark.asyncio
async def test_filters_apply_once_per_new_body_never_for_summary_or_resend(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("FILTERED")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _text_response("final answer"),
    ]
    harness.summary_script = [_text_response("summary")]

    inlet_calls = {"n": 0}
    finalize_calls = {"n": 0}
    original_inlet = sub_agent.apply_inlet_filters_if_enabled
    original_finalize = sub_agent.finalize_model_request

    async def counting_inlet(filter_pipeline, request, form_data, extra_params):
        inlet_calls["n"] += 1
        return await original_inlet(filter_pipeline, request, form_data, extra_params)

    async def counting_finalize(filter_pipeline, request, form_data, extra_params):
        finalize_calls["n"] += 1
        return await original_finalize(filter_pipeline, request, form_data, extra_params)

    monkeypatch.setattr(sub_agent, "apply_inlet_filters_if_enabled", counting_inlet)
    monkeypatch.setattr(sub_agent, "finalize_model_request", counting_finalize)

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=3,
        apply_inlet_filters=True,
    )

    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    assert len(main_calls) == 4
    assert len(harness.summary_calls) >= 1
    assert inlet_calls["n"] == len(main_calls)
    assert finalize_calls["n"] == len(main_calls)


@pytest.mark.asyncio
async def test_anchor_delta_hit_uses_usage_instead_of_full_estimate(monkeypatch):
    harness = _LoopHarness(monkeypatch)

    async def small_tool(**kwargs):
        return "tiny"

    tools = {"small": {"spec": {"name": "small"}, "callable": small_tool}}
    harness.script = [
        _with_usage(_tool_call_response("c1", "small"), 50),
        _with_usage(_tool_call_response("c2", "small"), 55),
        _with_usage(_tool_call_response("c3", "small"), 60),
        _text_response("done"),
    ]

    full_calls = []

    def huge_full_estimate(body, encoder=None):
        full_calls.append(body)
        return 999_999

    monkeypatch.setattr(sub_agent, "estimate_body_tokens", huge_full_estimate)

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=5_000,
        max_iterations=5,
    )

    assert harness.summary_calls == []
    assert len(full_calls) == 1


@pytest.mark.asyncio
async def test_summary_prefix_is_byte_identical_to_last_sent_payload(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("CACHE")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]
    harness.summary_script = [_text_response("cache-stable summary")]

    original_inlet = sub_agent.apply_inlet_filters_if_enabled

    async def marking_inlet(filter_pipeline, request, form_data, extra_params):
        form_data = await original_inlet(filter_pipeline, request, form_data, extra_params)
        messages = form_data.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            head = dict(messages[0])
            content = head.get("content", "")
            if isinstance(content, str):
                head["content"] = f"{content} [F1]"
            messages = [head, *messages[1:]]
            form_data = {**form_data, "messages": messages}
        return form_data

    monkeypatch.setattr(sub_agent, "apply_inlet_filters_if_enabled", marking_inlet)

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=10,
        apply_inlet_filters=True,
    )

    assert harness.summary_calls, "compaction never fired"
    first_summary_index = harness.calls.index(harness.summary_calls[0])
    snapshot = harness.calls[first_summary_index - 1]
    summary_body = harness.summary_calls[0]

    summary_prefix = summary_body["messages"][:-1]
    snapshot_messages = snapshot["messages"]
    assert summary_prefix == snapshot_messages[: len(summary_prefix)]
    assert snapshot_messages[len(summary_prefix)]["role"] == "assistant"
    assert " [F1]" in snapshot_messages[0]["content"]
    assert snapshot_messages[0]["content"].count(" [F1]") == 1
    assert summary_prefix[0]["content"].count(" [F1]") == 1
    assert summary_body["messages"][-1]["role"] == "user"
    assert summary_body.get("tools") is not None
    assert summary_body["model"] == snapshot["model"]
    assert summary_body["stream"] is False


@pytest.mark.asyncio
async def test_reader_name_collision_pins_raw_with_single_error(monkeypatch, caplog):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("COLLIDE")

    async def noisy_tool(**kwargs):
        return big

    async def original_reader(**kwargs):
        return "original"

    tools = {
        "noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool},
        "agent_ref_exec": {
            "spec": {"name": "agent_ref_exec", "description": "ORIGINAL"},
            "callable": original_reader,
        },
    }
    harness.script = [_tool_call_response("c1", "noisy"), _text_response("done")]

    with caplog.at_level(logging.ERROR, logger=sub_agent.log.name):
        await _run(
            monkeypatch, harness, tools_dict=tools, mode="ref_exec", large_threshold=200
        )

    errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "already present" in r.message
    ]
    assert len(errors) == 1

    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    tool_message = next(m for m in main_calls[1]["messages"] if m.get("role") == "tool")
    assert tool_message["content"] == big
    payload_tools = main_calls[1].get("tools") or []
    reader_specs = [
        t for t in payload_tools
        if isinstance(t, dict) and (t.get("function") or {}).get("name") == "agent_ref_exec"
    ]
    assert len(reader_specs) == 1
    assert reader_specs[0]["function"]["description"] == "ORIGINAL"


@pytest.mark.asyncio
async def test_final_send_triggers_compaction_before_final_answer(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("FINALCOMPACT")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
    ]
    harness.summary_script = [_text_response("final compaction summary")]

    async def staged_estimate(
        run,
        *,
        model_id,
        tools_param,
        current_messages,
        volatile_tokens,
        metadata,
        user_obj,
    ):
        if len(current_messages) >= 8:
            return 10**9
        return 0

    monkeypatch.setattr(sub_agent, "_estimate_loop_tokens", staged_estimate)

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=80_000,
        max_iterations=3,
    )

    assert len(harness.summary_calls) == 1
    final_call = harness.calls[-1]
    assert (final_call.get("metadata") or {}).get("task") != "sub_agent_summary"
    task_user = final_call["messages"][1]
    assert "<agent_loop_compaction_context>" in task_user["content"]
    assert "final compaction summary" in task_user["content"]
    assert final_call["messages"][-1]["content"].startswith("Maximum tool iterations reached")


@pytest.mark.asyncio
async def test_final_send_keeps_tools_and_strips_once_on_tool_call(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("FINALSEND")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("fin-1", "noisy"),
        _text_response("FINAL"),
    ]

    original_finalize = sub_agent.finalize_model_request

    async def forcing_finalize(filter_pipeline, request, form_data, extra_params):
        form_data = await original_finalize(filter_pipeline, request, form_data, extra_params)
        messages = form_data.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[-1], dict):
            content = messages[-1].get("content", "")
            if isinstance(content, str) and content.startswith("Maximum tool iterations reached"):
                form_data = {
                    **form_data,
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "noisy"},
                    },
                }
        return form_data

    monkeypatch.setattr(sub_agent, "finalize_model_request", forcing_finalize)

    result = await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="ref_exec",
        large_threshold=200,
        max_iterations=1,
    )

    assert result == "FINAL"
    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    assert len(main_calls) == 3
    final_first, final_retry = main_calls[1], main_calls[2]
    for key in ("tools", "tool_choice", "functions", "function_call", "parallel_tool_calls"):
        assert key not in final_retry
    assert final_retry["messages"] == final_first["messages"]

    payload_tools = final_first.get("tools") or []
    reader_names = [
        (t.get("function") or {}).get("name")
        for t in payload_tools
        if isinstance(t, dict)
    ]
    assert "agent_ref_exec" in reader_names
    assert final_first["tool_choice"] == "none"

    first_tool = next(m for m in final_first["messages"] if m.get("role") == "tool")
    retry_tool = next(m for m in final_retry["messages"] if m.get("role") == "tool")
    assert "agent_ref_truncated" in first_tool["content"]
    assert first_tool["content"] == retry_tool["content"]


@pytest.mark.asyncio
async def test_core_system_prompt_mutation_stays_in_the_sent_copy(monkeypatch):
    harness = _LoopHarness(monkeypatch, apply_core_system_prompt=True)
    big = _big_tool_result("COREMUT")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]
    harness.summary_script = [
        _tool_call_response("s-tool", "noisy"),
        _text_response("core-safe summary"),
    ]

    original_inlet = sub_agent.apply_inlet_filters_if_enabled

    async def mutating_inlet(filter_pipeline, request, form_data, extra_params):
        form_data = await original_inlet(
            filter_pipeline, request, form_data, extra_params
        )
        messages = form_data.get("messages")
        # In-place mutation on purpose: filters may rewrite message dicts
        # they are given, so only a pre-filter deepcopy protects the run's
        # current_messages.
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            head = messages[0]
            if isinstance(head.get("content"), str):
                head["content"] = f"{head['content']} [INLET]"
        return form_data

    monkeypatch.setattr(sub_agent, "apply_inlet_filters_if_enabled", mutating_inlet)

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=10,
        apply_inlet_filters=True,
    )

    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    assert len(main_calls) >= 5
    for payload in main_calls:
        head = payload["messages"][0]
        assert head["role"] == "system"
        assert head["content"] == f"{CORE_SYSTEM_PROMPT}\nsys [INLET]"
        assert head["content"].count(" [INLET]") == 1

    assert len(harness.summary_calls) >= 2
    for payload in harness.summary_calls:
        head = payload["messages"][0]
        assert head["content"] == f"{CORE_SYSTEM_PROMPT}\nsys [INLET]"

    first_summary = harness.summary_calls[0]
    strip_resend = harness.summary_calls[1]
    assert "tools" in first_summary
    assert "tools" not in strip_resend
    assert strip_resend["messages"][:-1] == first_summary["messages"][:-1]


@pytest.mark.asyncio
async def test_summary_model_defaults_to_last_sent_resolved_model(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("ARENA")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]
    harness.summary_script = [_text_response("resolved summary")]

    original_inlet = sub_agent.apply_inlet_filters_if_enabled

    async def resolving_inlet(filter_pipeline, request, form_data, extra_params):
        form_data = await original_inlet(filter_pipeline, request, form_data, extra_params)
        return {**form_data, "model": "arena-child-model"}

    monkeypatch.setattr(sub_agent, "apply_inlet_filters_if_enabled", resolving_inlet)

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=10,
        apply_inlet_filters=True,
    )

    assert harness.summary_calls, "compaction never fired"
    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    assert all(c["model"] == "arena-child-model" for c in main_calls)
    assert harness.summary_calls[0]["model"] == "arena-child-model"


def _message_round_id(message: dict) -> set[str]:
    ids: set[str] = set()
    if message.get("role") == "assistant":
        for tc in message.get("tool_calls") or []:
            if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                ids.add(tc["id"])
    if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
        ids.add(message["tool_call_id"])
    return ids


@pytest.mark.asyncio
async def test_boundary_folded_side_mismatch_aborts_normal_compaction_succeeds(monkeypatch):
    big = _big_tool_result("BOUNDARY")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]

    original_inlet = sub_agent.apply_inlet_filters_if_enabled

    async def round_dropping_inlet(filter_pipeline, request, form_data, extra_params):
        form_data = await original_inlet(filter_pipeline, request, form_data, extra_params)
        messages = form_data.get("messages")
        if isinstance(messages, list):
            kept = [
                m
                for m in messages
                if not (isinstance(m, dict) and _message_round_id(m) == {"c1"})
            ]
            form_data = {**form_data, "messages": kept}
        return form_data

    degraded = _LoopHarness(monkeypatch)
    degraded.script = list(script)
    degraded.summary_script = [_text_response("never used")]
    monkeypatch.setattr(sub_agent, "apply_inlet_filters_if_enabled", round_dropping_inlet)

    result = await _run(
        monkeypatch,
        degraded,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=10,
        apply_inlet_filters=True,
    )

    assert result.startswith("Sub-agent compaction summary failed")
    assert "folded assistant tool-call ids" in result
    assert degraded.summary_calls == []
    main_calls = [
        c for c in degraded.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    # The folded-round mismatch aborts at the first trigger (iteration 4
    # here: three complete rounds exist, the dropped c1 round makes the
    # folded side diverge) before a summary or another send happens.
    assert len(main_calls) == 3

    monkeypatch.setattr(sub_agent, "apply_inlet_filters_if_enabled", original_inlet)
    normal = _LoopHarness(monkeypatch)
    normal.script = list(script)
    normal.summary_script = [_text_response("folded summary")]

    result = await _run(
        monkeypatch,
        normal,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=10,
        apply_inlet_filters=True,
    )

    assert result == "done"
    assert len(normal.summary_calls) >= 1


@pytest.mark.asyncio
async def test_anchor_invalidated_after_compaction(monkeypatch):
    from owui_ext.shared.loop_compaction import estimate_body_tokens as real_ebt

    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("ANCHORRESET")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _with_usage(_tool_call_response("c1", "noisy"), 50),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]
    harness.summary_script = [_text_response("post-anchor summary")]

    full_calls = {"n": 0}

    def counting_ebt(body, encoder=None):
        full_calls["n"] += 1
        return real_ebt(body, encoder=encoder)

    monkeypatch.setattr(sub_agent, "estimate_body_tokens", counting_ebt)

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=10,
    )

    assert len(harness.summary_calls) >= 1
    # Full-estimate calls: iteration 1 (no anchor yet) and every estimate
    # after the compaction invalidated the anchor. A stale anchor would
    # keep absorbing iterations 2+ and stop at one call.
    assert full_calls["n"] >= 2


@pytest.mark.asyncio
async def test_mode_fixed_ref_exec_counts_previews_not_raw(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    huge = "substantial result line with detail\n" * 40_000

    async def noisy_tool(**kwargs):
        return huge

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _text_response("done"),
    ]

    await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="ref_exec",
        large_threshold=200,
        threshold=3_000,
    )

    assert harness.summary_calls == []
    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    tool_message = next(m for m in main_calls[3]["messages"] if m.get("role") == "tool")
    assert "agent_ref_truncated" in tool_message["content"]


@pytest.mark.asyncio
async def test_mode_fixed_raw_without_reader_counts_raw_and_compacts(monkeypatch, caplog):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("RAWFIX")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]
    harness.summary_script = [_text_response("raw-mode summary")]

    monkeypatch.setattr(
        sub_agent,
        "_reader_available_in_payload",
        lambda form_data: (False, "reader removed for test"),
    )
    captured_runs = []
    original_compact = sub_agent._compact_loop_context

    async def run_capturing_compact(run, **kwargs):
        captured_runs.append(run)
        return await original_compact(run, **kwargs)

    monkeypatch.setattr(sub_agent, "_compact_loop_context", run_capturing_compact)

    with caplog.at_level(logging.ERROR, logger=sub_agent.log.name):
        await _run(
            monkeypatch,
            harness,
            tools_dict=tools,
            mode="ref_exec",
            large_threshold=200,
            threshold=600,
        )

    errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "using raw tool results" in r.message
    ]
    assert len(errors) == 1
    assert len(harness.summary_calls) >= 1
    assert captured_runs
    assert all(run.history_ref is None for run in captured_runs)
    main_calls = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    tool_message = next(m for m in main_calls[1]["messages"] if m.get("role") == "tool")
    assert tool_message["content"] == big


@pytest.mark.asyncio
async def test_mode_fixed_keeps_previews_when_reader_later_missing(monkeypatch, caplog):
    big = _big_tool_result("MODEFIX")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]

    original_check = sub_agent._reader_available_in_payload
    check_state = {"calls": 0}

    def flaky_reader(form_data):
        check_state["calls"] += 1
        if check_state["calls"] == 1:
            return original_check(form_data)
        return (False, "reader disappeared after first send")

    monkeypatch.setattr(sub_agent, "_reader_available_in_payload", flaky_reader)

    degraded = _LoopHarness(monkeypatch)
    degraded.script = list(script)
    degraded.summary_script = [_text_response("degraded-mode summary")]

    with caplog.at_level(logging.ERROR, logger=sub_agent.log.name):
        await _run(
            monkeypatch,
            degraded,
            tools_dict=tools,
            mode="ref_exec",
            large_threshold=200,
            threshold=600,
        )

    errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "after mode was fixed" in r.message
    ]
    assert len(errors) == 1
    main_calls = [
        c for c in degraded.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ]
    tool_message = next(m for m in main_calls[1]["messages"] if m.get("role") == "tool")
    assert "agent_ref_truncated" in tool_message["content"]

    monkeypatch.setattr(sub_agent, "_reader_available_in_payload", original_check)
    reference = _LoopHarness(monkeypatch)
    reference.script = list(script)
    reference.summary_script = [_text_response("reference summary")]

    await _run(
        monkeypatch,
        reference,
        tools_dict=tools,
        mode="ref_exec",
        large_threshold=200,
        threshold=600,
    )
    assert len(degraded.summary_calls) == len(reference.summary_calls)


@pytest.mark.asyncio
async def test_summary_transient_http_exception_retries_and_succeeds(monkeypatch):
    from fastapi import HTTPException

    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("HTTP503")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
        _text_response("done"),
    ]
    harness.summary_script = [
        HTTPException(status_code=503),
        _text_response("retried summary"),
    ]

    result = await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=10,
    )

    assert result == "done"
    assert len(harness.summary_calls) >= 2
    assert isinstance(harness.summary_calls[1], dict)
    compacted_send = [
        c for c in harness.calls if (c.get("metadata") or {}).get("task") != "sub_agent_summary"
    ][-1]
    assert "retried summary" in compacted_send["messages"][1]["content"]


@pytest.mark.asyncio
async def test_encoder_resolution_skipped_when_nothing_consumes_tokens(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    calls = {"n": 0}
    original = sub_agent.resolve_tiktoken_encoder

    def counting_resolver(request=None):
        calls["n"] += 1
        return original(request)

    monkeypatch.setattr(sub_agent, "resolve_tiktoken_encoder", counting_resolver)

    async def noop_tool(**kwargs):
        return "ok"

    tools = {"noop": {"spec": {"name": "noop"}, "callable": noop_tool}}
    harness.script = [_tool_call_response("c1", "noop"), _text_response("done")]

    result = await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        compaction_enabled=False,
        max_iterations=3,
    )

    assert result == "done"
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_summarizer_incomplete_finish_reason_fails_without_retry(monkeypatch):
    harness = _LoopHarness(monkeypatch)
    big = _big_tool_result("LENGTHCUT")

    async def noisy_tool(**kwargs):
        return big

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response("c1", "noisy"),
        _tool_call_response("c2", "noisy"),
        _tool_call_response("c3", "noisy"),
        _tool_call_response("c4", "noisy"),
    ]
    harness.summary_script = [
        {"choices": [{"message": {"content": "partial summary"}, "finish_reason": "length"}]}
    ]

    result = await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="raw",
        threshold=600,
        max_iterations=10,
    )

    assert result.startswith("Sub-agent compaction summary failed")
    assert "length" in result
    assert len(harness.summary_calls) == 1


@pytest.mark.asyncio
async def test_full_estimate_counts_model_system_prompt_after_variable_expansion(
    monkeypatch,
):
    harness = _LoopHarness(monkeypatch, apply_core_system_prompt=True)

    class _ArenaAppState:
        MODELS = {
            "arena-parent": {
                "id": "arena-parent",
                "owned_by": "arena",
                "info": {"meta": {"model_ids": ["arena-child-model"]}},
            },
            "arena-child-model": {"id": "arena-child-model"},
        }

    class _ArenaApp:
        state = _ArenaAppState()

    class _ArenaRequest:
        app = _ArenaApp()

    long_bio = "research context " * 800
    monkeypatch.setattr(
        sys.modules[__name__], "CORE_SYSTEM_PROMPT", "Model briefing: {{USER_BIO}}"
    )

    class _FakeUser:
        id = "u1"
        role = "user"

        def model_dump(self):
            return {
                "id": "u1",
                "role": "user",
                "name": "Tester",
                "email": "t@example.com",
                "bio": long_bio,
            }

    requested_ids: list[str] = []

    class _Params:
        def __init__(self, data):
            self._data = data

        def model_dump(self):
            return self._data

    class _ModelRow:
        def __init__(self, data):
            self.params = _Params(data)

    class _ModelsFake:
        @staticmethod
        async def get_model_by_id(model_id):
            requested_ids.append(model_id)
            if model_id == "arena-child-model":
                return _ModelRow({"system": "Model briefing: {{USER_BIO}}"})
            return None

    monkeypatch.setattr(
        sys.modules["open_webui.models.models"], "Models", _ModelsFake
    )

    async def noisy_tool(**kwargs):
        return _big_tool_result("SYS", lines=10)

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": noisy_tool}}
    harness.script = [
        _tool_call_response(f"c{i}", "noisy") for i in range(1, 5)
    ] + [_text_response("done")]
    harness.summary_script = [_text_response("system-prompt-aware summary")]

    _install(monkeypatch, harness)
    compaction, large = _options(mode="raw", threshold=1_500)

    result = await sub_agent.run_sub_agent_loop(
        request=_ArenaRequest(),
        user=_FakeUser(),
        model_id="arena-parent",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ],
        tools_dict=tools,
        max_iterations=10,
        event_emitter=harness.emitter,
        apply_inlet_filters=True,
        compaction=compaction,
        large_results=large,
    )

    assert result == "done"
    assert len(harness.summary_calls) >= 1
    assert requested_ids == ["arena-child-model"]
    assert harness.calls[0]["model"] == "arena-child-model"
    assert "research context" in harness.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_compaction_prunes_caches_of_folded_tool_texts(monkeypatch):
    counter = {"n": 0}
    original_compact = sub_agent._compact_loop_context

    async def observing_compact(run, **kwargs):
        compacted = await original_compact(run, **kwargs)
        counter["n"] += 1
        live = {
            message.get("content")
            for message in compacted
            if isinstance(message, dict)
            and message.get("role") == "tool"
            and isinstance(message.get("content"), str)
        }
        assert set(run.truncate_cache) <= live
        assert set(run.classification_cache) <= live
        return compacted

    monkeypatch.setattr(sub_agent, "_compact_loop_context", observing_compact)

    state = {"n": 0}

    async def varying_tool(**kwargs):
        state["n"] += 1
        return _big_tool_result(f"ROUND-{state['n']}")

    tools = {"noisy": {"spec": {"name": "noisy"}, "callable": varying_tool}}
    harness = _LoopHarness(monkeypatch)
    harness.script = [
        _tool_call_response(f"c{i}", "noisy") for i in range(1, 7)
    ] + [_text_response("done")]
    harness.summary_script = [_text_response("pruning summary")]

    result = await _run(
        monkeypatch,
        harness,
        tools_dict=tools,
        mode="truncate",
        large_threshold=200,
        threshold=600,
        max_iterations=10,
    )

    assert result == "done"
    assert counter["n"] >= 1
