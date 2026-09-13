"""
title: Sub Agent
author: skyzi000
version: 0.6.0
license: MIT
required_open_webui_version: 0.9.6
description: Run autonomous, tool-heavy tasks in a sub-agent and keep the main chat context clean.

Open WebUI v0.7 introduced powerful builtin tools (web search, memory, notes,
knowledge bases, etc.), making complex multi-step tasks possible. However,
heavy tool usage can hit context window limits, causing conversations to fail
silently without returning a response.

This tool solves that problem by delegating tool-heavy tasks to sub-agents
running in isolated contexts. The sub-agent executes tools autonomously,
then returns only the final result - keeping your main conversation clean
and efficient.

Requirements:
- Native Function Calling must be enabled for the model
  (Model settings > Advanced Params> Function Calling: native)

Inspired by VS Code's runSubagent functionality, this tool was developed from scratch specifically for Open WebUI to ensure seamless integration and optimal performance.
"""

import asyncio
import copy
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Type

from fastapi import Request
from pydantic import BaseModel, Field

from owui_ext.shared.async_utils import maybe_await
from owui_ext.shared.builtin_tools import BUILTIN_TOOL_CATEGORIES, VALVE_TO_CATEGORY
from owui_ext.shared.completion_response import format_chat_completion_error
from owui_ext.shared.inlet_filters import (
    apply_inlet_filters_if_enabled,
    finalize_model_request,
    resolve_model_filter_pipeline,
)
from owui_ext.shared.loop_compaction import (
    REQUEST_TOKEN_OVERHEAD,
    LoopUsageAnchor,
    build_loop_input_fingerprint,
    canonical_history_records,
    classify_provider_failure,
    embed_envelope_in_task_message,
    estimate_body_tokens,
    estimate_messages_tokens,
    estimate_with_anchor,
    extract_summary_text,
    render_compaction_envelope,
    resolve_summary_prompt,
    resolve_tiktoken_encoder,
    response_usage,
    select_loop_compaction_cut,
    summary_choice_has_tool_calls,
    summary_response_incomplete_reason,
    usage_input_tokens,
)
from owui_ext.shared.model_features import (
    model_has_note_knowledge,
    model_knowledge_tools_enabled,
)
from owui_ext.shared.notifications import emit_notification
from owui_ext.shared.prompt_utils import (
    _append_tool_server_prompts,
    merge_prompt_sections,
)
from owui_ext.shared.ref_exec import (
    REF_EXEC_TOOL_NAME,
    RefProjectionError,
    RefRunStore,
    apply_ref_projection_plan,
    build_ref_reader,
    classify_ref_text,
    project_native_tool_texts,
    ref_exec_tool_spec_payload,
    render_truncate_preview_sync,
)
from owui_ext.shared.tool_execution import (
    execute_direct_tool_call,
    execute_tool_call,
    normalize_terminal_tools_result,
    process_tool_result,
)
from owui_ext.shared.mcp_tools import cleanup_mcp_clients, resolve_mcp_tools
from owui_ext.shared.tool_loader import build_tools_dict
from owui_ext.shared.skills import (
    extract_skill_manifest,
    extract_user_skill_tags,
    register_view_skill,
)
from owui_ext.shared.tool_servers import (
    build_direct_tools_dict,
    extract_direct_tool_server_prompts,
    normalize_direct_tool_servers,
    resolve_direct_tool_servers_from_request_and_metadata,
    resolve_terminal_id_from_request_and_metadata,
)
from owui_ext.shared.valves import coerce_user_valves

log = logging.getLogger(__name__)


class SubAgentTaskItem(BaseModel):
    """A single sub-agent task specification."""

    description: str = Field(
        description="Brief task summary shown to the user as status text, and it should be written in the user's language."
    )
    prompt: str = Field(
        description="Detailed instructions for the sub-agent; this can be written in any language that best suits the task."
    )


# ============================================================================
# Helper functions (outside class - AI cannot invoke these)
# ============================================================================


def normalize_parallel_sub_agent_tasks(tasks: Any) -> tuple[Optional[list[dict[str, str]]], Optional[str]]:
    """Normalize raw parallel task payloads into validated dicts."""
    if not isinstance(tasks, list):
        return (
            None,
            json.dumps(
                {
                    "error": f"tasks must be a list, got {type(tasks).__name__}",
                    "expected_format": '[{"description": "Task summary", "prompt": "Detailed instructions"}]',
                },
                ensure_ascii=False,
            ),
        )

    validated_tasks: list[dict[str, str]] = []
    for i, task in enumerate(tasks):
        if isinstance(task, SubAgentTaskItem):
            task_item = task
        else:
            if isinstance(task, str):
                try:
                    task = json.loads(task)
                except (json.JSONDecodeError, TypeError):
                    return (
                        None,
                        json.dumps(
                            {"error": f"tasks[{i}] must be an object, got unparseable string"},
                            ensure_ascii=False,
                        ),
                    )

            if not isinstance(task, dict):
                return (
                    None,
                    json.dumps(
                        {"error": f"tasks[{i}] must be an object"},
                        ensure_ascii=False,
                    ),
                )

            try:
                task_item = SubAgentTaskItem.model_validate(task)
            except Exception as exc:
                if hasattr(exc, "errors"):
                    errors = exc.errors()
                    if errors:
                        first_error = errors[0]
                        loc = ".".join(str(part) for part in first_error.get("loc", ()))
                        message = first_error.get("msg", "is invalid")
                        if loc:
                            return (
                                None,
                                json.dumps(
                                    {"error": f"tasks[{i}].{loc} {message}"},
                                    ensure_ascii=False,
                                ),
                            )
                return (
                    None,
                    json.dumps(
                        {"error": f"tasks[{i}] is invalid"},
                        ensure_ascii=False,
                    ),
                )

        description = task_item.description.strip()
        prompt = task_item.prompt.strip()

        if not description:
            return (
                None,
                json.dumps(
                    {"error": f"tasks[{i}].description cannot be empty"},
                    ensure_ascii=False,
                ),
            )
        if not prompt:
            return (
                None,
                json.dumps(
                    {"error": f"tasks[{i}].prompt cannot be empty"},
                    ensure_ascii=False,
                ),
            )

        validated_tasks.append({"description": description, "prompt": prompt})

    return validated_tasks, None


# ============================================================================
# Loop compaction / large-result externalization
# ============================================================================


@dataclass(frozen=True)
class LoopCompactionOptions:
    enabled: bool = True
    threshold_tokens: int = 80_000
    summary_model: str = ""


@dataclass(frozen=True)
class LargeToolResultOptions:
    mode: Literal["ref_exec", "truncate", "raw"] = "ref_exec"
    threshold_tokens: int = 10_000


class _SummaryFailure(RuntimeError):
    pass


class _LoopRunState:
    """Per-run state for compaction and large-result projection."""

    def __init__(
        self,
        *,
        compaction: LoopCompactionOptions,
        large_results: LargeToolResultOptions,
        encoder: Any,
        encoder_ready: bool,
        filter_identity: Any,
        tool_server_prompt_signature: Any,
        tool_server_prompt_texts: str = "",
        model_system_prompt: str | None = None,
    ) -> None:
        self.store = RefRunStore()
        self.encoder = encoder
        self.encoder_ready = encoder_ready
        self.compaction = compaction
        self.large_results = large_results
        self.requested_mode = large_results.mode
        self.effective_mode = large_results.mode
        self.mode_fixed = False
        self.mode_error_logged = False
        self.anchor: LoopUsageAnchor | None = None
        self.summary_text: str | None = None
        self.history_ref: str | None = None
        self.classification_cache: dict[str, Any] = {}
        self.truncate_cache: dict[str, str] = {}
        self.filter_identity = filter_identity
        self.tool_server_prompt_signature = tool_server_prompt_signature
        self.tool_server_prompt_texts = tool_server_prompt_texts
        self.model_system_prompt = model_system_prompt
        self.last_sent_form_data: Optional[dict] = None


_TOOL_REQUEST_STRIP_KEYS = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "parallel_tool_calls",
)


def _is_forced_tool_choice(value: Any) -> bool:
    if isinstance(value, dict):
        choice_type = value.get("type")
        if choice_type in (None, "function", "tool"):
            return True
        return choice_type not in {"auto", "none"}
    if isinstance(value, str):
        return value not in {"auto", "none"}
    return False


def _is_forced_function_call(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("name"))
    if isinstance(value, str):
        return value not in {"auto", "none"}
    return False


def _neutralize_forced_tool_choice(body: dict) -> None:
    """Forced tool_choice/function_call would make the model call a tool on a
    request that must not execute tools; neutralize to "none" while keeping
    the tool definitions attached for cache-shape stability.
    """
    if body.get("tools") and _is_forced_tool_choice(body.get("tool_choice")):
        body["tool_choice"] = "none"
    elif not body.get("tools"):
        body.pop("tool_choice", None)
    if body.get("functions") and _is_forced_function_call(body.get("function_call")):
        body["function_call"] = "none"
    elif not body.get("functions"):
        body.pop("function_call", None)


def _strip_tool_request_keys(body: dict) -> dict:
    return {key: value for key, value in body.items() if key not in _TOOL_REQUEST_STRIP_KEYS}


def _payload_for_send(form_data: dict) -> dict:
    """Deep copy for the Core boundary: Core mutates payloads in place."""
    return copy.deepcopy(form_data)


def _snapshot_boundary_index(
    snapshot_messages: Any,
    boundary_ids: set[str],
) -> int | None:
    """Locate the compaction boundary in the last-sent payload by tool-call ids."""
    if not boundary_ids or not isinstance(snapshot_messages, list):
        return None
    candidates = []
    for index, message in enumerate(snapshot_messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        ids = {
            tc.get("id")
            for tc in tool_calls
            if isinstance(tc, dict) and isinstance(tc.get("id"), str)
        }
        if ids == boundary_ids:
            candidates.append(index)
    if len(candidates) != 1:
        return None
    boundary = candidates[0]
    later_assistant_ids: set[str] = set()
    for message in snapshot_messages[boundary:]:
        if isinstance(message, dict) and message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                        later_assistant_ids.add(tc["id"])
    answered: set[str] = set()
    for message in snapshot_messages[boundary + 1 :]:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if tool_call_id not in later_assistant_ids:
            return None
        answered.add(tool_call_id)
    if not boundary_ids <= answered:
        return None
    return boundary


def _reader_available_in_payload(form_data: dict) -> tuple[bool, str]:
    tools = form_data.get("tools")
    if not isinstance(tools, list) or not tools:
        return False, "tools absent from final payload"
    tool_choice = form_data.get("tool_choice")
    if tool_choice == "none":
        return False, "tool_choice=none in final payload"
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        forced = tool_choice.get("function")
        if isinstance(forced, dict) and forced.get("name") not in (None, REF_EXEC_TOOL_NAME):
            return False, "tool_choice forces another function"
        if choice_type not in (None, "function", "tool", "auto"):
            return False, f"tool_choice type {choice_type!r} excludes tool use"
    names = set()
    for tool in tools:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.add(function["name"])
    if REF_EXEC_TOOL_NAME not in names:
        return False, "reader schema removed from final payload"
    return True, ""


async def _classify_tool_text(run: _LoopRunState, text: str) -> Any:
    cached = run.classification_cache.get(text)
    if cached is not None:
        return cached
    classification = await classify_ref_text(
        text,
        threshold_tokens=run.large_results.threshold_tokens,
        encoder=run.encoder if run.encoder_ready else None,
    )
    run.classification_cache[text] = classification
    return classification


async def _truncate_projected_messages(
    run: _LoopRunState, messages: list[dict]
) -> list[dict]:
    projected = []
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") == "tool"
            and isinstance(message.get("content"), str)
        ):
            content = message["content"]
            classification = await _classify_tool_text(run, content)
            if classification.eligible and classification.utf8_bytes:
                preview = run.truncate_cache.get(content)
                if preview is None:
                    preview = await asyncio.to_thread(
                        render_truncate_preview_sync,
                        content,
                        classification.utf8_bytes,
                        threshold_tokens=run.large_results.threshold_tokens,
                        encoder=run.encoder if run.encoder_ready else None,
                    )
                    if preview is None:
                        raise RefProjectionError(stage="tool truncate rendering")
                    run.truncate_cache[content] = preview
                message = {**message, "content": preview}
        projected.append(message)
    return projected


async def _ref_project_form_data(
    run: _LoopRunState,
    form_data: dict,
    *,
    check_reader: bool = True,
) -> dict:
    messages = form_data.get("messages")
    if not isinstance(messages, list):
        return form_data
    if check_reader:
        available, reason = _reader_available_in_payload(form_data)
        if not available:
            if not run.mode_fixed:
                run.effective_mode = "raw"
                run.mode_fixed = True
                if not run.mode_error_logged:
                    log.error(
                        "agent_ref_exec unavailable; using raw tool results for this loop "
                        "/ requested_mode=ref_exec effective_mode=raw reason=%s",
                        reason,
                    )
                    run.mode_error_logged = True
                return form_data
            if not run.mode_error_logged:
                log.error(
                    "agent_ref_exec unavailable after mode was fixed; keeping previews "
                    "and refs / reason=%s",
                    reason,
                )
                run.mode_error_logged = True
        else:
            run.mode_fixed = True
    plan = await project_native_tool_texts(
        messages,
        threshold_tokens=run.large_results.threshold_tokens,
        encoder=run.encoder if run.encoder_ready else None,
        preview_cache=run.store.preview_cache,
        classification_cache=run.classification_cache,
    )
    if not plan.catalog:
        return form_data
    for entry in plan.catalog:
        run.store.intern_tool_text(entry.source.text, entry)
    projected_messages = await apply_ref_projection_plan(messages, plan)
    return {**form_data, "messages": projected_messages}


async def _project_form_data(
    run: _LoopRunState,
    form_data: dict,
    *,
    check_reader: bool = True,
) -> dict:
    """One projection function for every provider send (loop + final)."""
    if run.effective_mode == "raw":
        return form_data
    if run.effective_mode == "truncate":
        messages = form_data.get("messages")
        if not isinstance(messages, list):
            return form_data
        projected = await _truncate_projected_messages(run, messages)
        return {**form_data, "messages": projected}
    return await _ref_project_form_data(run, form_data, check_reader=check_reader)


async def _estimation_messages(run: _LoopRunState, messages: list[dict]) -> list[dict]:
    """Uses the same ``project_native_tool_texts`` + ``apply_ref_projection_plan``
    path as ``_ref_project_form_data`` (sharing its caches) but never
    interns entries into the store — interning is send-time only.
    """
    if run.effective_mode == "raw":
        return messages
    if run.effective_mode == "truncate":
        return await _truncate_projected_messages(run, messages)
    plan = await project_native_tool_texts(
        messages,
        threshold_tokens=run.large_results.threshold_tokens,
        encoder=run.encoder if run.encoder_ready else None,
        preview_cache=run.store.preview_cache,
        classification_cache=run.classification_cache,
    )
    if not plan.catalog:
        return messages
    return await apply_ref_projection_plan(messages, plan)


def _fingerprint_kwargs(
    run: _LoopRunState,
    *,
    model_id: str,
    tools_param: Any,
) -> dict:
    return {
        "model_id": model_id,
        "tools_param": tools_param or [],
        "filter_identity": run.filter_identity,
        "tool_server_prompt_signature": run.tool_server_prompt_signature,
    }


async def _estimate_loop_tokens(
    run: _LoopRunState,
    *,
    model_id: str,
    tools_param: Any,
    current_messages: list[dict],
    volatile_tokens: int,
    metadata: dict,
    user_obj: Any,
) -> int | None:
    if not run.encoder_ready or run.encoder is None:
        return None
    anchor = run.anchor
    prefix_fingerprint: str | None = None
    if anchor is not None and len(current_messages) >= anchor.stable_message_count:
        prefix_fingerprint = await asyncio.to_thread(
            build_loop_input_fingerprint,
            **_fingerprint_kwargs(run, model_id=model_id, tools_param=tools_param),
            stable_messages=current_messages[: anchor.stable_message_count],
        )
    suffix_estimate: int | None = None
    if (
        prefix_fingerprint is not None
        and prefix_fingerprint == anchor.input_fingerprint
    ):
        if len(current_messages) > anchor.stable_message_count:
            suffix_estimate = await asyncio.to_thread(
                estimate_messages_tokens,
                await _estimation_messages(
                    run, current_messages[anchor.stable_message_count :]
                ),
                encoder=run.encoder,
            )
        else:
            suffix_estimate = 0
        return estimate_with_anchor(
            anchor,
            current_fingerprint=prefix_fingerprint,
            current_messages=current_messages,
            current_volatile_tokens=volatile_tokens,
            suffix_token_estimate=suffix_estimate,
            full_estimate=None,
        )

    estimation_body: dict[str, Any] = {
        "messages": await _estimation_messages(run, current_messages)
    }
    if run.model_system_prompt:
        from open_webui.utils.payload import apply_system_prompt_to_body

        messages = estimation_body["messages"]
        if isinstance(messages, list):
            projected = list(messages)
            if (
                projected
                and isinstance(projected[0], dict)
                and projected[0].get("role") == "system"
            ):
                projected[0] = copy.deepcopy(projected[0])
            body = {"messages": projected, "metadata": metadata}
            await apply_system_prompt_to_body(
                run.model_system_prompt, body, metadata, user_obj
            )
            estimation_body["messages"] = body["messages"]
    if tools_param:
        estimation_body["tools"] = tools_param
    full = await asyncio.to_thread(
        estimate_body_tokens,
        estimation_body,
        encoder=run.encoder,
    )
    if run.tool_server_prompt_texts:
        prompt_tokens = await asyncio.to_thread(
            estimate_messages_tokens,
            [{"role": "system", "content": run.tool_server_prompt_texts}],
            encoder=run.encoder,
        )
        if prompt_tokens is not None:
            full = (full or 0) + prompt_tokens - REQUEST_TOKEN_OVERHEAD
    if run.summary_text:
        envelope = render_compaction_envelope(run.summary_text)
        envelope_tokens = await asyncio.to_thread(
            estimate_messages_tokens,
            [{"role": "user", "content": envelope}],
            encoder=run.encoder,
        )
        if envelope_tokens is not None:
            full = (full or 0) + envelope_tokens
    return full


async def _request_loop_summary(
    *,
    generate_chat_completion: Callable,
    request: Request,
    user_obj: Any,
    summary_body: dict,
) -> str:
    body = summary_body
    tools_attached = bool(body.get("tools") or body.get("functions"))
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            response = await generate_chat_completion(
                request=request,
                form_data=_payload_for_send(body),
                user=user_obj,
                bypass_filter=True,
            )
        except Exception as exc:
            # ``str(exc)`` of an HTTPException is "503: ..." which never
            # matches the "status 503" marker; classify on the real
            # status_code attribute first.
            if (
                classify_provider_failure(
                    status_code=getattr(exc, "status_code", None),
                    error_text=str(exc),
                )
                == "transient"
            ):
                continue
            raise _SummaryFailure(f"Sub-agent compaction summary failed: {exc}") from exc
        if not isinstance(response, dict):
            error_text = format_chat_completion_error(response) or (
                f"Unexpected response type: {type(response).__name__}"
            )
            if (
                classify_provider_failure(
                    status_code=getattr(response, "status_code", None),
                    error_text=error_text,
                )
                == "transient"
            ):
                continue
            raise _SummaryFailure(
                f"Sub-agent compaction summary failed: {error_text}"
            )
        incomplete_reason = summary_response_incomplete_reason(response)
        if incomplete_reason is not None:
            raise _SummaryFailure(
                "Sub-agent compaction summary failed: summarizer stopped before "
                f"completing the summary ({incomplete_reason})"
            )
        if summary_choice_has_tool_calls(response):
            if tools_attached:
                body = _strip_tool_request_keys(body)
                tools_attached = False
                continue
            raise _SummaryFailure(
                "Sub-agent compaction summary failed: summarizer returned tool calls "
                "even without tools"
            )
        text = extract_summary_text(response)
        if text:
            return text
        raise _SummaryFailure(
            "Sub-agent compaction summary failed: summarizer returned no text content"
        )
    raise _SummaryFailure(
        "Sub-agent compaction summary failed: summarizer exhausted 3 attempts"
    )


def _build_summary_body(
    run: _LoopRunState,
    *,
    model_id: str,
    boundary: int,
) -> dict:
    snapshot = run.last_sent_form_data or {}
    snapshot_messages = snapshot.get("messages")
    if not isinstance(snapshot_messages, list):
        raise _SummaryFailure(
            "Sub-agent compaction summary failed: last sent payload has no messages"
        )
    body = dict(snapshot)
    body["messages"] = [
        *list(snapshot_messages[:boundary]),
        {"role": "user", "content": resolve_summary_prompt()},
    ]
    body["model"] = (
        run.compaction.summary_model.strip()
        or snapshot.get("model")
        or model_id
    )
    body["stream"] = False
    metadata = dict(snapshot.get("metadata") or {})
    metadata["task"] = "sub_agent_summary"
    metadata.pop("files", None)
    body["metadata"] = metadata
    _neutralize_forced_tool_choice(body)
    return body


def _assistant_tool_call_id_set(messages: Any) -> set[str]:
    ids: set[str] = set()
    if not isinstance(messages, list):
        return ids
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                    ids.add(tc["id"])
    return ids


async def _compact_loop_context(
    run: _LoopRunState,
    *,
    generate_chat_completion: Callable,
    request: Request,
    user_obj: Any,
    model_id: str,
    current_messages: list[dict],
    event_emitter: Optional[Callable],
) -> list[dict]:
    cut = select_loop_compaction_cut(current_messages)
    if cut is None:
        return current_messages

    snapshot_messages = (run.last_sent_form_data or {}).get("messages")
    first_kept = cut.tail_messages[0] if cut.tail_messages else None
    boundary_ids: set[str] = set()
    if isinstance(first_kept, dict):
        tool_calls = first_kept.get("tool_calls")
        if isinstance(tool_calls, list):
            boundary_ids = {
                tc.get("id")
                for tc in tool_calls
                if isinstance(tc, dict) and isinstance(tc.get("id"), str)
            }
    boundary = _snapshot_boundary_index(snapshot_messages, boundary_ids)
    if boundary is None:
        raise _SummaryFailure(
            "Sub-agent compaction summary failed: compaction boundary could not be "
            "uniquely located in the last sent payload"
        )
    snapshot_folded_ids = _assistant_tool_call_id_set(
        list(snapshot_messages or [])[:boundary]
    )
    if snapshot_folded_ids != _assistant_tool_call_id_set(cut.summarization_prefix):
        raise _SummaryFailure(
            "Sub-agent compaction summary failed: folded assistant tool-call ids "
            "do not match the last sent payload prefix"
        )
    summary_body = _build_summary_body(run, model_id=model_id, boundary=boundary)

    if event_emitter:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "description": "Sub-agent compacting context...",
                    "done": False,
                },
            }
        )

    summary = await _request_loop_summary(
        generate_chat_completion=generate_chat_completion,
        request=request,
        user_obj=user_obj,
        summary_body=summary_body,
    )
    run.summary_text = summary
    run.anchor = None
    if run.effective_mode == "ref_exec":
        records = await asyncio.to_thread(
            canonical_history_records,
            cut.summarization_prefix,
        )
        run.history_ref = await asyncio.to_thread(
            run.store.add_history_records,
            records,
        )

    compacted: list[dict] = []
    if cut.preserved_system_message is not None:
        compacted.append(cut.preserved_system_message)
    compacted.append(cut.task_user_message)
    compacted.extend(cut.tail_messages)
    live_tool_texts = {
        message.get("content")
        for message in compacted
        if isinstance(message, dict)
        and message.get("role") == "tool"
        and isinstance(message.get("content"), str)
    }
    run.truncate_cache = {
        text: preview
        for text, preview in run.truncate_cache.items()
        if text in live_tool_texts
    }
    run.classification_cache = {
        text: classification
        for text, classification in run.classification_cache.items()
        if text in live_tool_texts
    }
    return compacted


def _embed_envelope_for_send(run: _LoopRunState, messages: list[dict]) -> list[dict]:
    if not run.summary_text:
        return messages
    envelope = render_compaction_envelope(
        run.summary_text,
        (
            None
            if run.effective_mode != "ref_exec" or run.history_ref is None
            else _history_manifest_json(run)
        ),
    )
    embedded = []
    task_index = next(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        None,
    )
    for index, message in enumerate(messages):
        if index == task_index:
            embedded.append(embed_envelope_in_task_message(message, envelope))
        else:
            embedded.append(message)
    return embedded


def _history_manifest_json(run: _LoopRunState) -> str | None:
    if run.history_ref is None:
        return None
    payload = run.store.history_manifest_payload(run.history_ref)
    if payload is None:
        return None
    return json.dumps([payload], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def _send_loop_request(
    run: _LoopRunState,
    *,
    generate_chat_completion: Callable,
    request: Request,
    user_obj: Any,
    model_id: str,
    tools_param: Any,
    filter_pipeline: Any,
    extra_params: dict,
    current_messages: list[dict],
    trailing_message: Optional[dict],
    merge_trailing_into_last_user: bool,
    metadata: dict,
    check_reader: bool,
    neutralize_tool_choice: bool,
) -> tuple[Optional[str], Any, dict, Optional[str]]:
    """The one provider-send path shared by the loop and final requests."""
    fingerprint: Optional[str] = None
    if run.compaction.enabled:
        fingerprint = await asyncio.to_thread(
            build_loop_input_fingerprint,
            **_fingerprint_kwargs(run, model_id=model_id, tools_param=tools_param),
            stable_messages=current_messages,
        )

    messages = copy.deepcopy(current_messages)
    if trailing_message is not None:
        last = messages[-1] if messages else None
        last_role = last.get("role") if isinstance(last, dict) else None
        if merge_trailing_into_last_user and last_role == "user":
            # Merging the trailing note into the trailing user message
            # avoids two consecutive user turns, which strict
            # role-alternation validators reject. Later loop iterations
            # end with a tool result, where appending is safe.
            merged = dict(last)
            note_text = trailing_message.get("content", "")
            content = merged.get("content", "")
            if isinstance(content, list):
                merged["content"] = content + [
                    {"type": "text", "text": f"\n\n{note_text}"}
                ]
            else:
                merged["content"] = (
                    f"{content}\n\n{note_text}" if content else note_text
                )
            messages[-1] = merged
        else:
            messages.append(copy.deepcopy(trailing_message))
    messages = _embed_envelope_for_send(run, messages)

    form_data: dict = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "metadata": metadata,
    }
    if tools_param:
        form_data["tools"] = tools_param

    # Match Core's inlet -> tool prompts -> request filter ordering.
    form_data = await apply_inlet_filters_if_enabled(
        filter_pipeline, request, form_data, extra_params
    )
    form_data = _append_tool_server_prompts(form_data, extra_params)
    form_data = await finalize_model_request(
        filter_pipeline, request, form_data, extra_params
    )
    if neutralize_tool_choice:
        _neutralize_forced_tool_choice(form_data)

    try:
        form_data = await _project_form_data(
            run, form_data, check_reader=check_reader
        )
    except RefProjectionError as projection_failure:
        log.error(
            "Sub-agent large-result projection failed: %s", projection_failure
        )
        return (
            f"Error during sub-agent execution: {projection_failure}",
            None,
            {},
            fingerprint,
        )

    try:
        run.last_sent_form_data = form_data
        response = await generate_chat_completion(
            request=request,
            form_data=_payload_for_send(form_data),
            user=user_obj,
            bypass_filter=True,  # We handle filters manually above
        )
    except Exception as e:
        log.exception(f"Error in sub-agent completion: {e}")
        return (
            f"Error during sub-agent execution: {e}",
            None,
            {},
            fingerprint,
        )
    return None, response, form_data, fingerprint


async def run_sub_agent_loop(
    request: Request,
    user: Any,
    model_id: str,
    messages: List[dict],
    tools_dict: dict,
    max_iterations: int,
    event_emitter: Optional[Callable] = None,
    extra_params: Optional[dict] = None,
    apply_inlet_filters: bool = True,
    iteration_note_role: Literal["user", "system"] = "user",
    compaction: Optional[LoopCompactionOptions] = None,
    large_results: Optional[LargeToolResultOptions] = None,
) -> str:
    """Run the sub-agent tool loop until completion.

    Args:
        request: FastAPI request object
        user: User model object
        model_id: Model ID to use for completions
        messages: Initial messages for the sub-agent
        tools_dict: Dict of available tools
        max_iterations: Maximum number of tool call iterations
        event_emitter: Optional event emitter for status updates
        extra_params: Extra parameters for tool execution
        apply_inlet_filters: Whether to apply inlet filters (outlet filters are never applied)
        iteration_note_role: Role for the per-iteration meta note ("user" keeps the
            leading system message intact; "system" appends an extra system message)

    Returns:
        Final text response from the sub-agent
    """
    from open_webui.models.users import UserModel
    from open_webui.utils.chat import generate_chat_completion

    if extra_params is None:
        extra_params = {}

    # Prepare user object
    if isinstance(user, dict):
        user_obj = UserModel(**user)
    else:
        user_obj = user

    filter_pipeline = await resolve_model_filter_pipeline(
        apply_inlet_filters,
        request,
        model_id,
        extra_params.get("__metadata__", {}).get("filter_ids", []),
    )

    compaction_options = compaction or LoopCompactionOptions()
    large_result_options = large_results or LargeToolResultOptions()
    encoder: Any = None
    if compaction_options.enabled or large_result_options.mode != "raw":
        encoder, _encoding_name = await asyncio.to_thread(
            resolve_tiktoken_encoder, request
        )
    model_system_prompt: str | None = None
    if compaction_options.enabled and encoder is not None:
        from open_webui.models.models import Models

        lookup_id = (
            filter_pipeline["model_id"]
            if isinstance(filter_pipeline, dict)
            else model_id
        )
        resolved_model = await Models.get_model_by_id(lookup_id)
        if resolved_model is not None:
            system_prompt = resolved_model.params.model_dump().get("system")
            if isinstance(system_prompt, str) and system_prompt:
                model_system_prompt = system_prompt
    if encoder is None and compaction_options.enabled:
        log.warning(
            "Sub-agent context compaction could not resolve a tiktoken encoder; "
            "compaction will not trigger for this loop"
        )
    filter_identity = list(extra_params.get("__metadata__", {}).get("filter_ids", []))
    tool_server_prompts = []
    terminal_prompt = (extra_params or {}).get("__terminal_system_prompt__")
    if isinstance(terminal_prompt, str) and terminal_prompt.strip():
        tool_server_prompts.append(terminal_prompt)
    direct_prompts = (extra_params or {}).get(
        "__direct_tool_server_system_prompts__", []
    )
    if isinstance(direct_prompts, list):
        tool_server_prompts.extend(
            p for p in direct_prompts if isinstance(p, str) and p.strip()
        )
    run = _LoopRunState(
        compaction=compaction_options,
        large_results=large_result_options,
        encoder=encoder,
        encoder_ready=encoder is not None,
        filter_identity=filter_identity,
        tool_server_prompt_signature={
            "prompts": [
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for prompt in tool_server_prompts
            ]
        },
        tool_server_prompt_texts="\n\n".join(tool_server_prompts),
        model_system_prompt=model_system_prompt,
    )

    loop_tools_dict = dict(tools_dict)
    if large_result_options.mode == "ref_exec":
        if REF_EXEC_TOOL_NAME in tools_dict:
            run.effective_mode = "raw"
            run.mode_error_logged = True
            log.error(
                "agent_ref_exec unavailable; using raw tool results for this loop "
                "/ requested_mode=ref_exec effective_mode=raw reason=tool name "
                "already present in the sub-agent tool set"
            )
        else:
            reader = build_ref_reader(
                run.store,
                threshold_tokens=large_result_options.threshold_tokens,
                encoder=encoder,
            )
            loop_tools_dict[REF_EXEC_TOOL_NAME] = {
                "spec": ref_exec_tool_spec_payload()["function"],
                "callable": reader,
            }

    # Build tools parameter for native function calling
    tools_param = None
    if loop_tools_dict:
        tools_param = [
            {"type": "function", "function": tool.get("spec", {})}
            for tool in loop_tools_dict.values()
        ]

    current_messages = list(messages)
    iteration = 0

    while max_iterations == 0 or iteration < max_iterations:
        iteration += 1

        if event_emitter:
            iteration_label = (
                f"Sub-agent iteration {iteration}"
                if max_iterations == 0
                else f"Sub-agent iteration {iteration}/{max_iterations}"
            )
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": iteration_label,
                        "done": False,
                    },
                }
            )

        # Build iteration context message.
        iteration_info: Optional[str] = None
        volatile_tokens = 0
        if max_iterations != 0:
            iteration_info = f"[Iteration {iteration}/{max_iterations}]"
            if iteration == max_iterations:
                iteration_info += " This is your FINAL tool call opportunity."

        send_metadata = {
            "task": "sub_agent",
            "sub_agent_iteration": iteration,
            "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
        }

        if run.compaction.enabled:
            if iteration_info:
                volatile_tokens = (
                    await asyncio.to_thread(
                        estimate_messages_tokens,
                        [{"role": iteration_note_role, "content": iteration_info}],
                        encoder=run.encoder,
                    )
                    or 0
                )
            try:
                estimate = await _estimate_loop_tokens(
                    run,
                    model_id=model_id,
                    tools_param=tools_param,
                    current_messages=current_messages,
                    volatile_tokens=volatile_tokens,
                    metadata=send_metadata,
                    user_obj=user_obj,
                )
            except RefProjectionError:
                raise
            except Exception:
                log.exception("Sub-agent compaction estimate failed; skipping compaction")
                estimate = None
            if estimate is not None and estimate >= run.compaction.threshold_tokens:
                try:
                    current_messages = await _compact_loop_context(
                        run,
                        generate_chat_completion=generate_chat_completion,
                        request=request,
                        user_obj=user_obj,
                        model_id=model_id,
                        current_messages=current_messages,
                        event_emitter=event_emitter,
                    )
                except _SummaryFailure as failure:
                    log.error("%s", failure)
                    return str(failure)

        error, response, _, fingerprint = (
            await _send_loop_request(
                run,
                generate_chat_completion=generate_chat_completion,
                request=request,
                user_obj=user_obj,
                model_id=model_id,
                tools_param=tools_param,
                filter_pipeline=filter_pipeline,
                extra_params=extra_params,
                current_messages=current_messages,
                trailing_message=(
                    {"role": iteration_note_role, "content": iteration_info}
                    if iteration_info is not None
                    else None
                ),
                merge_trailing_into_last_user=(iteration_note_role == "user"),
                metadata=send_metadata,
                check_reader=True,
                neutralize_tool_choice=False,
            )
        )
        if error is not None:
            return error

        usage = response_usage(response)
        observed_input = usage_input_tokens(usage) if usage is not None else None
        if (
            observed_input is not None
            and run.compaction.enabled
            and fingerprint is not None
        ):
            run.anchor = LoopUsageAnchor(
                input_tokens=observed_input,
                stable_message_count=len(current_messages),
                input_fingerprint=fingerprint,
                volatile_message_tokens=volatile_tokens if iteration_info else 0,
            )

        # Handle response: surface upstream errors (JSONResponse,
        # PlainTextResponse, etc.) verbatim so the parent loop sees the
        # real cause instead of an opaque ``Unexpected response type``.
        error_msg = format_chat_completion_error(response)
        if error_msg is not None:
            return error_msg

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not choices:
                return "No response from model"

            choice = choices[0]
            if not isinstance(choice, Mapping):
                return f"API returned malformed response: choices[0] is {type(choice).__name__}, not a mapping"

            message = choice.get("message", {})
            if not isinstance(message, Mapping):
                return f"API returned malformed response: message is {type(message).__name__}, not a mapping"

            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            # Emit status with LLM response content
            if event_emitter and content:
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[Step {iteration}] Assistant: {content.replace(chr(10), ' ')}",
                            "done": False,
                        },
                    }
                )

            # If no tool calls, we're done
            if not tool_calls:
                return content or ""

            # Normalize: filter out non-mapping entries from tool_calls
            if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
                return (
                    f"API returned malformed response: tool_calls is "
                    f"{type(tool_calls).__name__}, not a sequence. "
                    f"Content so far: {content or '(none)'}"
                )
            raw_count = len(tool_calls)
            tool_calls = [tc for tc in tool_calls if isinstance(tc, Mapping)]
            if not tool_calls:
                if raw_count > 0:
                    return (
                        f"API returned malformed response: {raw_count} tool_calls "
                        f"entries were all non-mapping. "
                        f"Content so far: {content or '(none)'}"
                    )
                return content or ""

            # Emit status with tool calls summary
            if event_emitter:
                tool_names = [
                    tc["function"].get("name", "unknown") if isinstance(tc.get("function"), Mapping) else "malformed"
                    for tc in tool_calls
                ]
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[Step {iteration}] Tool calls: {', '.join(tool_names)}",
                            "done": False,
                        },
                    }
                )

            normalized_tool_calls = []
            for tc in tool_calls:
                tc_func = tc.get("function")
                if not isinstance(tc_func, Mapping):
                    continue
                args = tc_func.get("arguments", "{}")
                if not isinstance(args, str):
                    try:
                        args = json.dumps(args, ensure_ascii=False)
                    except Exception:
                        args = str(args)
                normalized_tool_calls.append({
                    **tc,
                    "function": {**tc_func, "arguments": args},
                })
            if not normalized_tool_calls:
                return (
                    f"API returned malformed response: all tool_calls had invalid "
                    f"'function' fields. Content so far: {content or '(none)'}"
                )

            current_messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": normalized_tool_calls,
                }
            )

            # Execute each tool call
            for tool_call in normalized_tool_calls:
                tc_func = tool_call.get("function")
                tool_args_raw = tc_func.get("arguments", "{}") if isinstance(tc_func, dict) else "{}"
                tool_args_display = str(tool_args_raw).replace(chr(10), " ") if tool_args_raw else "{}"

                if event_emitter:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[Step {iteration}] Args: {tool_args_display}",
                                "done": False,
                            },
                        }
                    )

                result = await execute_tool_call(
                    tool_call,
                    loop_tools_dict,
                    {
                        **extra_params,
                        "__messages__": current_messages,
                    },
                    event_emitter=event_emitter,
                )

                # Emit status with tool result
                if event_emitter:
                    result_content = result["content"].replace(chr(10), ' ') if result["content"] else "(empty)"
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[Step {iteration}] Result: {result_content}",
                                "done": False,
                            },
                        }
                    )

                # Add tool result to conversation
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"],
                    }
                )

    # Max iterations reached
    if event_emitter:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "description": f"Max iterations ({max_iterations}) reached",
                    "done": False,
                },
            }
        )

    final_instruction = (
        "Maximum tool iterations reached. Please provide your final answer "
        "based on the information gathered so far."
    )
    final_metadata = {
        "task": "sub_agent",
        "sub_agent_iteration": max_iterations + 1,
        "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
    }
    if run.compaction.enabled:
        volatile_tokens = (
            await asyncio.to_thread(
                estimate_messages_tokens,
                [{"role": "user", "content": final_instruction}],
                encoder=run.encoder,
            )
            or 0
        )
        try:
            estimate = await _estimate_loop_tokens(
                run,
                model_id=model_id,
                tools_param=tools_param,
                current_messages=current_messages,
                volatile_tokens=volatile_tokens,
                metadata=final_metadata,
                user_obj=user_obj,
            )
        except RefProjectionError:
            raise
        except Exception:
            log.exception("Sub-agent compaction estimate failed; skipping compaction")
            estimate = None
        if estimate is not None and estimate >= run.compaction.threshold_tokens:
            try:
                current_messages = await _compact_loop_context(
                    run,
                    generate_chat_completion=generate_chat_completion,
                    request=request,
                    user_obj=user_obj,
                    model_id=model_id,
                    current_messages=current_messages,
                    event_emitter=event_emitter,
                )
            except _SummaryFailure as failure:
                log.error("%s", failure)
                return str(failure)

    error, response, form_data, _fingerprint = (
        await _send_loop_request(
            run,
            generate_chat_completion=generate_chat_completion,
            request=request,
            user_obj=user_obj,
            model_id=model_id,
            tools_param=tools_param,
            filter_pipeline=filter_pipeline,
            extra_params=extra_params,
            current_messages=current_messages,
            trailing_message={"role": "user", "content": final_instruction},
            merge_trailing_into_last_user=False,
            metadata=final_metadata,
            check_reader=False,
            neutralize_tool_choice=True,
        )
    )
    if error is not None:
        return error

    error_msg = format_chat_completion_error(response)
    if error_msg is not None:
        return error_msg

    def _final_message_text(candidate: Any) -> tuple[Optional[str], bool]:
        if not isinstance(candidate, dict):
            return None, False
        choices = candidate.get("choices", [])
        if not choices or not isinstance(choices[0], Mapping):
            return None, False
        message = choices[0].get("message", {})
        if not isinstance(message, Mapping):
            return None, False
        content = message.get("content", "")
        return (
            content if isinstance(content, str) else None,
            bool(message.get("tool_calls")),
        )

    content, has_tool_calls = _final_message_text(response)
    if has_tool_calls:
        retry_body = _strip_tool_request_keys(form_data)
        try:
            response = await generate_chat_completion(
                request=request,
                form_data=_payload_for_send(retry_body),
                user=user_obj,
                bypass_filter=True,
            )
        except Exception as e:
            log.exception(f"Error getting final response: {e}")
            return f"Error during sub-agent execution: {e}"
        error_msg = format_chat_completion_error(response)
        if error_msg is not None:
            return error_msg
        content, _has_tool_calls = _final_message_text(response)
    if content:
        return content

    return "Sub-agent reached maximum iterations without providing a final response."




async def load_sub_agent_tools(
    request: Request,
    user: Any,
    valves: Any,
    metadata: dict,
    model: dict,
    extra_params: dict,
    self_tool_id: Optional[str],
) -> tuple[dict, dict]:
    """Load regular, MCP, terminal, direct, and builtin tools for sub-agent.

    Thin wrapper around ``shared.tool_loader.build_tools_dict`` that
    parses sub_agent's ``AVAILABLE_TOOL_IDS`` / ``EXCLUDED_TOOL_IDS``
    valves, adds ``self_tool_id`` to the exclusion set to prevent the
    sub-agent from recursing into its own plugin, and pre-resolves the
    terminal binding / direct tool servers so the canonical helper
    doesn't re-read ``request.body()``.
    """
    metadata = metadata or {}
    extra_params = extra_params or {}
    debug = bool(getattr(valves, "DEBUG", False))

    terminal_id = await resolve_terminal_id_from_request_and_metadata(
        request=request,
        metadata=metadata,
        debug=debug,
    )
    direct_tool_servers = await resolve_direct_tool_servers_from_request_and_metadata(
        metadata=metadata,
        request=request,
        debug=debug,
    )

    available_tool_ids: list[str] = []
    if metadata.get("tool_ids"):
        available_tool_ids = list(metadata.get("tool_ids", []))

    if debug:
        log.info(f"[SubAgent] AVAILABLE_TOOL_IDS valve: '{valves.AVAILABLE_TOOL_IDS}'")
        log.info(f"[SubAgent] Available tool_ids from metadata: {available_tool_ids}")
        log.info(f"[SubAgent] self_tool_id: {self_tool_id}")
        log.info(f"[SubAgent] resolved terminal_id: {terminal_id}")
        log.info(
            f"[SubAgent] resolved direct tool servers: {len(direct_tool_servers)}"
        )

    excluded: set = set()
    if valves.EXCLUDED_TOOL_IDS.strip():
        excluded = {
            tid.strip() for tid in valves.EXCLUDED_TOOL_IDS.split(",") if tid.strip()
        }

    # Always exclude this tool itself to prevent infinite recursion
    if not self_tool_id:
        log.warning(
            "[SubAgent] self_tool_id is None, cannot exclude self from tool list. "
            "Recursion prevention may not work."
        )
    else:
        excluded.add(self_tool_id)

    if debug:
        log.info(f"[SubAgent] EXCLUDED_TOOL_IDS valve: '{valves.EXCLUDED_TOOL_IDS}'")
        if excluded:
            log.info(f"[SubAgent] Excluded tool IDs (including self): {sorted(excluded)}")

    if valves.AVAILABLE_TOOL_IDS.strip():
        tool_id_list = [
            tid.strip() for tid in valves.AVAILABLE_TOOL_IDS.split(",") if tid.strip()
        ]
        if debug:
            log.info(f"[SubAgent] Using AVAILABLE_TOOL_IDS valve: {tool_id_list}")
    else:
        tool_id_list = available_tool_ids
        if debug:
            log.info(
                f"[SubAgent] Using all available tool_ids from metadata: {tool_id_list}"
            )

    return await build_tools_dict(
        request=request,
        model=model,
        metadata=metadata,
        user=user,
        valves=valves,
        extra_params=extra_params,
        tool_id_list=tool_id_list,
        excluded_tool_ids=excluded,
        resolved_terminal_id=terminal_id,
        resolved_direct_tool_servers=direct_tool_servers,
    )


# ============================================================================
# Tools class
# ============================================================================


class Tools:
    """Sub-Agent tool for autonomous task completion."""

    class Valves(BaseModel):
        DEFAULT_MODEL: str = Field(
            default="",
            description="Default model ID for sub-agent tasks. Leave empty to use the same model as the main conversation.",
        )
        MAX_ITERATIONS: int = Field(
            default=50,
            ge=0,
            description=(
                "Maximum number of tool-call iterations per sub-agent task. "
                "Set to 0 for no iteration limit."
            ),
        )
        AVAILABLE_TOOL_IDS: str = Field(
            default="",
            description=(
                "[Advanced] Comma-separated list of tool IDs available to sub-agents. "
                "Leave empty (recommended) to use only tools enabled in the chat UI. "
                "When set, ONLY these tools are available (overrides chat UI tool selection). "
                "This controls regular tools only; builtin tools (web search, memory, etc.) "
                "are controlled separately by the ENABLE_*_TOOLS toggles below. "
                "WARNING: Mismatched tool sets between main AI and sub-agent can cause failures - "
                "the main AI may instruct the sub-agent to use tools it doesn't have. "
                "Tool server IDs (e.g., MCPO/OpenAPI) require 'server:' prefix (e.g., 'server:context7'). "
                "To find exact tool IDs, enable DEBUG, enable the desired tools in the chat UI, "
                "invoke the sub-agent, and check server logs for '[SubAgent] Available tool_ids from metadata'."
            ),
        )
        EXCLUDED_TOOL_IDS: str = Field(
            default="",
            description=(
                "Comma-separated list of tool IDs to exclude from sub-agents (e.g., this tool itself to prevent recursion). "
                "This controls regular tools only; to disable builtin tools, use the ENABLE_*_TOOLS toggles. "
                "If unsure about tool IDs or exclusion behavior, enable DEBUG and check server logs."
            ),
        )
        APPLY_INLET_FILTERS: bool = Field(
            default=True,
            description="Apply inlet and request filters (e.g., user_info_injector) to sub-agent model requests. Outlet filters are never applied to sub-agent responses.",
        )

        # Builtin tool category toggles
        ENABLE_TIME_TOOLS: bool = Field(
            default=True,
            description=(
                "Enable time utilities (get_current_timestamp, calculate_timestamp). "
                "NOTE for all ENABLE_*_TOOLS toggles: These can only disable builtin tools; "
                "they cannot enable tools that are disabled by global admin settings, "
                "model capabilities, or chat UI features (e.g., web search)."
            ),
        )
        ENABLE_WEB_TOOLS: bool = Field(
            default=True,
            description="Enable web search tools (search_web, fetch_url).",
        )
        ENABLE_IMAGE_TOOLS: bool = Field(
            default=True,
            description="Enable image generation tools (generate_image, edit_image).",
        )
        ENABLE_FILE_TOOLS: bool = Field(
            default=True,
            description=(
                "Enable Core tools for listing, searching, and reading files attached to the current chat "
                "(list_chat_files, query_chat_files, grep_chat_files, view_file). Files exposed through "
                "attached knowledge remain controlled by ENABLE_KNOWLEDGE_TOOLS."
            ),
        )
        ENABLE_KNOWLEDGE_TOOLS: bool = Field(
            default=True,
            description="Enable knowledge base tools (list/search/query knowledge bases and files).",
        )
        ENABLE_CHAT_TOOLS: bool = Field(
            default=True,
            description="Enable chat history tools (search_chats, view_chat).",
        )
        ENABLE_MEMORY_TOOLS: bool = Field(
            default=True,
            description="Enable memory tools (search_memories, add_memory, replace_memory_content).",
        )
        ENABLE_NOTES_TOOLS: bool = Field(
            default=True,
            description="Enable notes tools (search_notes, view_note, write_note, replace_note_content).",
        )
        ENABLE_CHANNELS_TOOLS: bool = Field(
            default=True,
            description="Enable channels tools (search_channels, search_channel_messages, etc.).",
        )
        ENABLE_TERMINAL_TOOLS: bool = Field(
            default=True,
            description=(
                "Enable Open Terminal tools when terminal_id is available in chat metadata "
                "(e.g., run_command, list_files, read_file, write_file, display_file)."
            ),
        )
        ENABLE_CODE_INTERPRETER_TOOLS: bool = Field(
            default=True,
            description="Enable code interpreter tools (execute_code).",
        )
        ENABLE_SKILLS_TOOLS: bool = Field(
            default=True,
            description="Enable skills tools (view_skill). When enabled and the parent conversation has skills, the sub-agent can view skill contents.",
        )
        ENABLE_SUBAGENT_TOOLS: bool = Field(
            default=False,
            description=(
                "Enable Core subagent tools (delegate_task, timer). Off by default because they use the "
                "parent conversation's model and tools rather than this agent's restrictions, and timer "
                "can schedule future work for the parent chat."
            ),
        )
        ENABLE_TASK_TOOLS: bool = Field(
            default=True,
            description="Enable task management tools (create_tasks, update_task).",
        )
        ENABLE_AUTOMATION_TOOLS: bool = Field(
            default=True,
            description="Enable automation tools (create/update/list/toggle/delete automations).",
        )
        ENABLE_CALENDAR_TOOLS: bool = Field(
            default=True,
            description="Enable calendar tools (search/create/update/delete calendar events).",
        )
        ENABLE_NOTIFICATION_TOOLS: bool = Field(
            default=True,
            description=(
                "Enable Core notification tools (notify), which send messages to the user's configured "
                "notification target."
            ),
        )
        MAX_PARALLEL_AGENTS: int = Field(
            default=5,
            description="Maximum number of sub-agents to run in parallel via run_parallel_sub_agents. To fully disable parallel execution, comment out the run_parallel_sub_agents method.",
        )
        ENABLE_CONTEXT_COMPACTION: bool = Field(
            default=True,
            description=(
                "Compact the sub-agent's internal context by summarizing older loop "
                "rounds once the estimated input tokens reach "
                "CONTEXT_COMPACTION_TOKEN_THRESHOLD. If you disable this, lower "
                "MAX_ITERATIONS accordingly - without compaction, long tool-heavy "
                "loops can exhaust the model context window. Compaction and "
                "LARGE_TOOL_RESULT_MODE work independently."
            ),
        )
        CONTEXT_COMPACTION_TOKEN_THRESHOLD: int = Field(
            default=80000,
            ge=1000,
            description=(
                "Estimated input-token threshold that triggers context compaction "
                "inside a sub-agent loop (same default as Open WebUI Core's "
                "CONTEXT_COMPACTION_TOKEN_THRESHOLD)."
            ),
        )
        COMPACTION_SUMMARY_MODEL: str = Field(
            default="",
            description=(
                "Model ID used for compaction summaries. Leave empty (recommended) to "
                "use the sub-agent's model, which helps preserve prompt caching where possible. "
                "Summary requests include tool definitions and tool-call history; "
                "the selected model must accept both."
            ),
        )
        LARGE_TOOL_RESULT_MODE: Literal["ref_exec", "truncate", "raw"] = Field(
            default="ref_exec",
            description=(
                "How oversized tool results are sent to the model. "
                "ref_exec (default): send a head/tail preview; the sub-agent can read omitted "
                "content during the same task. If the agent_ref_exec reader is unavailable for the "
                "initial request, raw mode is used for the entire task and an error is logged. "
                "truncate: send a head/tail preview without read-back. raw: send the full result. "
                "Independent of ENABLE_CONTEXT_COMPACTION."
            ),
        )
        LARGE_TOOL_RESULT_THRESHOLD_TOKENS: int = Field(
            default=10000,
            ge=1000,
            description=(
                "Tool results at or above this many tokens, or larger than 64 KiB, "
                "are handled by LARGE_TOOL_RESULT_MODE (ref_exec or truncate). raw "
                "ignores this setting."
            ),
        )
        ITERATION_NOTE_ROLE: Literal["user", "system"] = Field(
            default="user",
            description=(
                "Role used for the per-iteration meta note appended to each sub-agent request "
                "(e.g. '[Iteration 2/5]'). Default 'user' keeps the system message at the beginning "
                "of the conversation, preserving prompt caching and avoiding 'System message must be at "
                "the beginning' errors reported by some chat templates or inference APIs. "
                "Set to 'system' to restore the pre-0.5.2 behaviour (the meta note is appended as a "
                "standalone system message at the end of each request) — use this if the new default "
                "causes any regression with your model; note that it may re-trigger the system-position error."
            ),
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging.",
        )
        pass

    class UserValves(BaseModel):
        SYSTEM_PROMPT: str = Field(
            default="""\
You are a sub-agent operating autonomously to complete a delegated task.

CRITICAL RULES:
1. You MUST complete the task fully without asking the user for confirmation or clarification.
2. Continue working autonomously until the task is 100% complete.
3. Use available tools proactively to gather information and perform actions.
4. If you encounter obstacles, try alternative approaches before giving up.
5. If your messages contain [Iteration N/M] notes, your tool call iterations are limited. Complete the task before reaching the limit.

RESPONSE REQUIREMENTS:
- Provide a comprehensive final answer to the main agent.
- Include evidence and reasoning that supports your conclusions.
- If the task cannot be completed, explain what was attempted, why it failed, and provide actionable next steps the main agent should take.""",
            description="System prompt for sub-agent tasks.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def run_sub_agent(
        self,
        description: str,
        prompt: str,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __model__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __id__: Optional[str] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
        __oauth_token__: Optional[dict] = None,
        __messages__: Optional[list] = None,
    ) -> str:
        """
        Delegate a task to a sub-agent for autonomous completion.

        MANDATORY: If a task requires 3+ steps of investigation or complex analysis,
        you MUST NOT perform it yourself. Delegate to this tool immediately.
        Only handle simple 1-2 tool call tasks yourself. When in doubt, delegate.

        The sub-agent runs in a fresh context with NO access to the current
        conversation history — include all necessary context in the prompt.
        It has the same tools and executes them in a loop until completion,
        returning only the final result to keep the main conversation clean.

        :param description: Brief task summary shown to the user as status text, and it should be written in the user's language.
        :param prompt: Detailed instructions for the sub-agent; this can be written in any language that best suits the task.
        :return: Sub-agent's final response after task completion
        """
        if __request__ is None:
            return json.dumps(
                {"error": "Request context not available. Cannot run sub-agent."}
            )

        if __user__ is None:
            return json.dumps(
                {"error": "User context not available. Cannot run sub-agent."}
            )

        # Import here to avoid issues when not running in Open WebUI
        from open_webui.models.users import UserModel

        user = UserModel(**__user__)

        # Get user valves
        raw_user_valves = (__user__ or {}).get("valves", {})
        user_valves = coerce_user_valves(raw_user_valves, self.UserValves)

        # Extract skills from parent conversation messages.
        # Since v0.8.2, user-selected skills are injected as full <skill> tags,
        # while model-attached skills appear in the <available_skills> manifest.
        # __messages__ is injected via get_tools/get_updated_tool_function.
        skill_manifest = extract_skill_manifest(__messages__)
        user_skill_tags = extract_user_skill_tags(__messages__)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Starting sub-agent: {description}",
                        "done": False,
                    },
                }
            )

        # Determine model ID
        # Priority: DEFAULT_MODEL (valve) > chat model (metadata) > task model (__model__)
        model_id = self.valves.DEFAULT_MODEL
        if not model_id and __metadata__:
            model_id = (__metadata__.get("model") or {}).get("id", "")
        if not model_id and __model__:
            model_id = __model__.get("id", "")

        if not model_id:
            return json.dumps(
                {
                    "error": "No model ID available. Set DEFAULT_MODEL in Valves if the issue persists."
                }
            )

        # Resolve the model dict for the actual sub-agent model.
        # When DEFAULT_MODEL differs from the parent model, __model__ carries
        # the parent's capabilities; we need the sub-agent model's dict so
        # get_builtin_tools can correctly check capabilities (web_search, etc.).
        resolved_model = __model__ or {}
        if model_id and model_id != resolved_model.get("id", ""):
            try:
                resolved_model = __request__.app.state.MODELS.get(
                    model_id, resolved_model
                )
            except Exception:
                pass  # Fall back to parent model

        common_extra_params = {
            "__user__": __user__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__request__": __request__,
            "__model__": resolved_model,
            "__metadata__": __metadata__,
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
            "__oauth_token__": __oauth_token__,
            "__files__": __metadata__.get("files", []) if __metadata__ else [],
        }

        tools_dict, mcp_clients = await load_sub_agent_tools(
            request=__request__,
            user=user,
            valves=self.valves,
            metadata=__metadata__ or {},
            model=resolved_model,
            extra_params=common_extra_params,
            self_tool_id=__id__,
        )

        try:
            # Register view_skill if model-attached skills manifest is available
            if skill_manifest and self.valves.ENABLE_SKILLS_TOOLS:
                await register_view_skill(tools_dict, __request__, common_extra_params)

            # Build initial messages with skills context
            prompt_sections: list[str] = [user_valves.SYSTEM_PROMPT]
            if self.valves.ENABLE_SKILLS_TOOLS:
                # User-selected skills: inject full content (v0.8.2+)
                if user_skill_tags:
                    prompt_sections.extend(user_skill_tags)
                # Model-attached skills: inject manifest for lazy loading via view_skill
                if skill_manifest:
                    prompt_sections.append(skill_manifest)
            system_content = merge_prompt_sections(*prompt_sections)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            if __event_emitter__:
                tool_count = len(tools_dict)
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Sub-agent started with {tool_count} tools available",
                            "done": False,
                        },
                    }
                )

            # Run the sub-agent loop
            try:
                result = await run_sub_agent_loop(
                    request=__request__,
                    user=user,
                    model_id=model_id,
                    messages=messages,
                    tools_dict=tools_dict,
                    max_iterations=self.valves.MAX_ITERATIONS,
                    event_emitter=__event_emitter__,
                    extra_params=common_extra_params,
                    apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                    iteration_note_role=self.valves.ITERATION_NOTE_ROLE,
                    compaction=LoopCompactionOptions(
                        enabled=self.valves.ENABLE_CONTEXT_COMPACTION,
                        threshold_tokens=self.valves.CONTEXT_COMPACTION_TOKEN_THRESHOLD,
                        summary_model=self.valves.COMPACTION_SUMMARY_MODEL,
                    ),
                    large_results=LargeToolResultOptions(
                        mode=self.valves.LARGE_TOOL_RESULT_MODE,
                        threshold_tokens=self.valves.LARGE_TOOL_RESULT_THRESHOLD_TOKENS,
                    ),
                )
            except Exception as e:
                log.exception(f"Error in sub-agent execution: {e}")
                result = f"Sub-agent error: {e}"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Sub-agent completed: {description}",
                            "done": True,
                        },
                    }
                )

            return json.dumps(
                {
                    "note": "The user does NOT see this result directly - only you (the main agent) can see it.",
                    "result": result,
                },
                ensure_ascii=False,
            )
        finally:
            await cleanup_mcp_clients(mcp_clients)

    async def run_parallel_sub_agents(
        self,
        tasks: list[SubAgentTaskItem],
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __model__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __id__: Optional[str] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
        __oauth_token__: Optional[dict] = None,
        __messages__: Optional[list] = None,
    ) -> str:
        """
        Run multiple independent sub-agent tasks in parallel (concurrently).

        Use this instead of calling run_sub_agent multiple times when you have
        2 or more tasks that do NOT depend on each other's results.
        All tasks share the same model and tools but each runs in a fresh
        context with NO access to the conversation history, so include all
        necessary context in each prompt. They execute simultaneously and
        finish much faster than sequential calls.
        Craft each prompt as you would for run_sub_agent (role, context,
        specific instructions, expected output format, etc.).

        Example: [
            {"description": "Research topic A", "prompt": "You are a research specialist. ..."},
            {"description": "Analyze data B", "prompt": "You are a data analyst. ..."}
        ]

        :param tasks: List of task objects using the SubAgentTaskItem schema.
        :return: JSON with "results" array in the same order as tasks.
                 Each element has "description" and either "result" or "error".
        """
        if __request__ is None:
            return json.dumps(
                {"error": "Request context not available. Cannot run sub-agents."},
                ensure_ascii=False,
            )

        if __user__ is None:
            return json.dumps(
                {"error": "User context not available. Cannot run sub-agents."},
                ensure_ascii=False,
            )

        if isinstance(tasks, list) and len(tasks) > self.valves.MAX_PARALLEL_AGENTS:
            return json.dumps(
                {
                    "error": f"tasks count ({len(tasks)}) exceeds MAX_PARALLEL_AGENTS ({self.valves.MAX_PARALLEL_AGENTS})",
                    "max_parallel_agents": self.valves.MAX_PARALLEL_AGENTS,
                },
                ensure_ascii=False,
            )

        validated_tasks, tasks_error = normalize_parallel_sub_agent_tasks(tasks)
        if tasks_error is not None:
            return tasks_error

        if not validated_tasks:
            return json.dumps({"error": "tasks array is empty"}, ensure_ascii=False)

        # Import here to avoid issues when not running in Open WebUI
        from open_webui.models.users import UserModel

        user = UserModel(**__user__)

        # Get user valves
        raw_user_valves = (__user__ or {}).get("valves", {})
        user_valves = coerce_user_valves(raw_user_valves, self.UserValves)

        # Extract skills from parent conversation (same as run_sub_agent)
        skill_manifest = extract_skill_manifest(__messages__)
        user_skill_tags = extract_user_skill_tags(__messages__)

        # Determine model ID
        # Priority: DEFAULT_MODEL (valve) > chat model (metadata) > task model (__model__)
        model_id = self.valves.DEFAULT_MODEL
        if not model_id and __metadata__:
            model_id = (__metadata__.get("model") or {}).get("id", "")
        if not model_id and __model__:
            model_id = __model__.get("id", "")

        if not model_id:
            return json.dumps(
                {
                    "error": "No model ID available. Set DEFAULT_MODEL in Valves if the issue persists."
                },
                ensure_ascii=False,
            )

        # Resolve the model dict for the actual sub-agent model (same as run_sub_agent).
        resolved_model = __model__ or {}
        if model_id and model_id != resolved_model.get("id", ""):
            try:
                resolved_model = __request__.app.state.MODELS.get(
                    model_id, resolved_model
                )
            except Exception:
                pass

        # NOTE: __chat_id__ / __message_id__ are intentionally shared across
        # all parallel tasks.  They reference the *parent* conversation message
        # that triggered this tool call; sub-agents build their own internal
        # message history.  Creating fake per-task IDs would be incorrect
        # because no such messages exist in the DB.  Tools that write to the
        # parent message (e.g. generate_image) may interleave, but since all
        # tasks run on the same event loop this is not a data-race.
        common_extra_params = {
            "__user__": __user__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__request__": __request__,
            "__model__": resolved_model,
            "__metadata__": __metadata__,
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
            "__oauth_token__": __oauth_token__,
            "__files__": __metadata__.get("files", []) if __metadata__ else [],
        }

        # Tools are loaded once and shared across all parallel tasks for
        # efficiency.  This is safe because execute_tool_call rebinds
        # __event_emitter__ per invocation via get_updated_tool_function.
        # Caveat: tools that store __event_emitter__ on `self` (non-standard
        # pattern) could see cross-task interference.
        tools_dict, mcp_clients = await load_sub_agent_tools(
            request=__request__,
            user=user,
            valves=self.valves,
            metadata=__metadata__ or {},
            model=resolved_model,
            extra_params=common_extra_params,
            self_tool_id=__id__,
        )

        try:
            # Register view_skill if model-attached skills manifest is available
            if skill_manifest and self.valves.ENABLE_SKILLS_TOOLS:
                await register_view_skill(tools_dict, __request__, common_extra_params)

            # Build system content with skills context
            parallel_prompt_sections: list[str] = [user_valves.SYSTEM_PROMPT]
            if self.valves.ENABLE_SKILLS_TOOLS:
                if user_skill_tags:
                    parallel_prompt_sections.extend(user_skill_tags)
                if skill_manifest:
                    parallel_prompt_sections.append(skill_manifest)
            parallel_system_content = merge_prompt_sections(*parallel_prompt_sections)
            task_mapping = ", ".join(
                f"[{i + 1}] {task['description']}" for i, task in enumerate(validated_tasks)
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Running {len(validated_tasks)} sub-agents: {task_mapping}",
                            "done": False,
                        },
                    }
                )

            async def run_single_task(task_index: int, task: dict) -> dict:
                task_description = task["description"]
                task_prompt = task["prompt"]

                async def indexed_event_emitter(event: dict):
                    if not __event_emitter__:
                        return

                    if (
                        isinstance(event, dict)
                        and event.get("type") == "status"
                        and isinstance(event.get("data"), dict)
                    ):
                        prefixed_data = dict(event["data"])
                        original_description = prefixed_data.get("description", "")
                        if original_description:
                            prefixed_data["description"] = (
                                f"[{task_index}] {original_description}"
                            )
                        await __event_emitter__({"type": "status", "data": prefixed_data})
                        return

                    await __event_emitter__(event)

                try:
                    result = await run_sub_agent_loop(
                        request=__request__,
                        user=user,
                        model_id=model_id,
                        messages=[
                            {"role": "system", "content": parallel_system_content},
                            {"role": "user", "content": task_prompt},
                        ],
                        tools_dict=tools_dict,
                        max_iterations=self.valves.MAX_ITERATIONS,
                        event_emitter=indexed_event_emitter if __event_emitter__ else None,
                        extra_params={
                            **common_extra_params,
                            "__event_emitter__": indexed_event_emitter
                            if __event_emitter__
                            else None,
                        },
                        apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                        iteration_note_role=self.valves.ITERATION_NOTE_ROLE,
                        compaction=LoopCompactionOptions(
                            enabled=self.valves.ENABLE_CONTEXT_COMPACTION,
                            threshold_tokens=self.valves.CONTEXT_COMPACTION_TOKEN_THRESHOLD,
                            summary_model=self.valves.COMPACTION_SUMMARY_MODEL,
                        ),
                        large_results=LargeToolResultOptions(
                            mode=self.valves.LARGE_TOOL_RESULT_MODE,
                            threshold_tokens=self.valves.LARGE_TOOL_RESULT_THRESHOLD_TOKENS,
                        ),
                    )
                    return {"description": task_description, "result": result}
                except Exception as e:
                    log.exception(
                        f"Error in parallel sub-agent [{task_index}] {task_description}: {e}"
                    )
                    error_msg = str(e) or type(e).__name__
                    return {"description": task_description, "error": error_msg}

            task_coroutines = [
                run_single_task(i + 1, task) for i, task in enumerate(validated_tasks)
            ]
            gathered_results = await asyncio.gather(
                *task_coroutines, return_exceptions=True
            )

            processed_results = []
            for i, result in enumerate(gathered_results):
                if isinstance(result, BaseException):
                    processed_results.append(
                        {
                            "description": validated_tasks[i]["description"],
                            "error": str(result) or type(result).__name__,
                        }
                    )
                else:
                    processed_results.append(result)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Sub-agents completed: {task_mapping}",
                            "done": True,
                        },
                    }
                )

            return json.dumps(
                {
                    "note": "The user does NOT see this result directly - only you (the main agent) can see it.",
                    "results": processed_results,
                },
                ensure_ascii=False,
            )
        finally:
            await cleanup_mcp_clients(mcp_clients)
