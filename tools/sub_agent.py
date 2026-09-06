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

# === GENERATED FILE - DO NOT EDIT ===
# Source: src/owui_ext/tools/sub_agent.py
# Regenerate with: uv run python scripts/build_release.py --target sub_agent
# Future imports: (none)
# See release.toml for target definitions.

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

# --- inlined from src/owui_ext/shared/async_utils.py (owui_ext.shared.async_utils) ---
async def maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value

# --- inlined from src/owui_ext/shared/builtin_tools.py (owui_ext.shared.builtin_tools) ---
BUILTIN_TOOL_CATEGORIES: dict[str, set[str]] = {
    "time": {"get_current_timestamp", "calculate_timestamp"},
    "user_input": {"ask_user"},
    "web": {"search_web", "fetch_url"},
    "image": {"generate_image", "edit_image"},
    "files": {
        "list_chat_files",
        "query_chat_files",
        "grep_chat_files",
        "view_file",
    },
    "knowledge": {
        "list_knowledge",
        "list_knowledge_bases",
        "search_knowledge_bases",
        "query_knowledge_bases",
        "search_knowledge_files",
        "query_knowledge_files",
        "grep_knowledge_files",
        "kb_exec",
        "view_file",
        "view_knowledge_file",
    },
    "chat": {"search_chats", "view_chat"},
    "memory": {
        "search_memories",
        "list_memory_paths",
        "read_memory_path",
        "list_memories",
        "update_memory",
        "add_memory",
        "replace_memory_content",
        "delete_memory",
    },
    "notes": {
        "search_notes",
        "view_note",
        "write_note",
        "replace_note_content",
    },
    "channels": {
        "search_channels",
        "search_channel_messages",
        "view_channel_thread",
        "view_channel_message",
    },
    "code_interpreter": {"execute_code"},
    "skills": {"view_skill"},
    "subagents": {"delegate_task", "timer"},
    "tasks": {"create_tasks", "update_task"},
    "automations": {
        "create_automation",
        "update_automation",
        "list_automations",
        "toggle_automation",
        "delete_automation",
    },
    "calendar": {
        "search_calendar_events",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
    },
    "notifications": {"notify"},
}


VALVE_TO_CATEGORY: dict[str, str] = {
    "ENABLE_TIME_TOOLS": "time",
    "ENABLE_WEB_TOOLS": "web",
    "ENABLE_IMAGE_TOOLS": "image",
    "ENABLE_FILE_TOOLS": "files",
    "ENABLE_KNOWLEDGE_TOOLS": "knowledge",
    "ENABLE_CHAT_TOOLS": "chat",
    "ENABLE_MEMORY_TOOLS": "memory",
    "ENABLE_NOTES_TOOLS": "notes",
    "ENABLE_CHANNELS_TOOLS": "channels",
    "ENABLE_CODE_INTERPRETER_TOOLS": "code_interpreter",
    "ENABLE_SKILLS_TOOLS": "skills",
    "ENABLE_SUBAGENT_TOOLS": "subagents",
    "ENABLE_TASK_TOOLS": "tasks",
    "ENABLE_AUTOMATION_TOOLS": "automations",
    "ENABLE_CALENDAR_TOOLS": "calendar",
    "ENABLE_NOTIFICATION_TOOLS": "notifications",
}

# --- inlined from src/owui_ext/shared/completion_response.py (owui_ext.shared.completion_response) ---
import json
from typing import Any, Optional
from starlette.responses import JSONResponse, Response
_RESPONSE_BODY_PREVIEW_CHARS = 1024


def _truncate_preview(text: str) -> str:
    if len(text) > _RESPONSE_BODY_PREVIEW_CHARS:
        return text[:_RESPONSE_BODY_PREVIEW_CHARS] + "...[truncated]"
    return text


def _decode_response_body(response: Response) -> str:
    body = getattr(response, "body", None)
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        try:
            text = bytes(body).decode("utf-8", errors="replace")
        except Exception:
            text = repr(body)
    else:
        text = str(body)
    return _truncate_preview(text.strip())


def _extract_json_response_error(response: JSONResponse) -> str:
    status = getattr(response, "status_code", "unknown")
    try:
        error_data = json.loads(bytes(response.body).decode("utf-8"))
    except Exception:
        return f"API error (status {status}): Failed to parse response"
    if isinstance(error_data, dict):
        error_field = error_data.get("error")
        if isinstance(error_field, dict):
            msg = error_field.get("message")
            if isinstance(msg, str) and msg:
                return f"API error: {_truncate_preview(msg)}"
            return f"API error: {_truncate_preview(str(error_data))}"
        if isinstance(error_field, str) and error_field:
            return f"API error: {_truncate_preview(error_field)}"
        msg = error_data.get("message")
        if isinstance(msg, str) and msg:
            return f"API error: {_truncate_preview(msg)}"
        return f"API error: {_truncate_preview(str(error_data))}"
    return f"API error: {_truncate_preview(str(error_data))}"


def format_chat_completion_error(response: Any) -> Optional[str]:
    """Classify a ``generate_chat_completion`` response.

    Returns:
        ``None`` when ``response`` is a ``dict`` (success path); the
        caller should proceed to read ``response['choices']``.

        ``str`` describing the upstream failure for any other shape.
        ``JSONResponse`` bodies are unwrapped to surface the provider's
        error message; other ``Response`` subclasses (notably
        ``PlainTextResponse`` returned by Open WebUI core when the
        provider replies with non-JSON 400+ content) are reported with
        their status code and a truncated body preview so the caller
        can show the real cause to the parent loop.
    """
    if isinstance(response, dict):
        return None
    if isinstance(response, JSONResponse):
        return _extract_json_response_error(response)
    if isinstance(response, Response):
        status = getattr(response, "status_code", "unknown")
        body_text = _decode_response_body(response)
        type_name = type(response).__name__
        if body_text:
            return f"API error (status {status}, {type_name}): {body_text}"
        return f"API error (status {status}, {type_name}): empty body"
    return f"Unexpected response type: {type(response).__name__}"

# --- inlined from src/owui_ext/shared/inlet_filters.py (owui_ext.shared.inlet_filters) ---
import inspect
import logging
import random
from typing import Any
from fastapi import Request
_inlet_filters_log = logging.getLogger("owui_ext.shared.inlet_filters")


async def _inlet_filters_maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def resolve_model_filter_pipeline(
    apply_filters: bool,
    request: Request,
    model_id: str,
    enabled_filter_ids: list[str] | None,
) -> dict[str, Any] | None:
    """Resolve one nested model route and isolate its cached filter Valves."""
    request_state = getattr(request, "state", None)
    direct_model = getattr(request_state, "model", None)
    server_models = request.app.state.MODELS
    is_direct_target = bool(
        getattr(request_state, "direct", False)
        and isinstance(direct_model, dict)
        and direct_model.get("id") == model_id
    )

    # Core gives the request-scoped Direct model precedence over a server
    # model with the same ID. Otherwise resolve Arena before filters so one
    # logical agent loop keeps the same child for every provider call.
    model = direct_model if is_direct_target else server_models.get(model_id, {})
    if not is_direct_target and model.get("owned_by") == "arena":
        candidate_ids = model.get("info", {}).get("meta", {}).get("model_ids")
        filter_mode = model.get("info", {}).get("meta", {}).get("filter_mode")
        if candidate_ids and filter_mode == "exclude":
            candidate_ids = [
                candidate["id"]
                for candidate in server_models.values()
                if candidate.get("owned_by") != "arena"
                and candidate["id"] not in candidate_ids
            ]

        if not isinstance(candidate_ids, list) or not candidate_ids:
            candidate_ids = [
                candidate["id"]
                for candidate in server_models.values()
                if candidate.get("owned_by") != "arena"
            ]

        selected_model_id = random.choice(candidate_ids)
        selected_model = (
            direct_model
            if getattr(request_state, "direct", False)
            and isinstance(direct_model, dict)
            and direct_model.get("id") == selected_model_id
            else server_models.get(selected_model_id)
        )
        if selected_model:
            model_id = selected_model_id
            model = selected_model

    route = {
        "model": model,
        "model_id": model.get("id", model_id),
        "process": None,
        "functions": [],
        "context": None,
        "supports_context": False,
        "supports_request": False,
    }
    if not apply_filters:
        return route

    request_filters_supported = False
    try:
        from open_webui.utils import filter as filter_utils

        process_filter_functions = filter_utils.process_filter_functions
        supports_context = (
            "filter_context"
            in inspect.signature(process_filter_functions).parameters
        )
        context_factory = getattr(filter_utils, "FilterContext", None)
        # FilterContext exists in v0.11.0, but request filters and
        # get_filter_context were introduced together in v0.11.2.
        request_filters_supported = bool(
            supports_context
            and callable(context_factory)
            and callable(getattr(filter_utils, "get_filter_context", None))
        )

        enabled_filter_ids = list(enabled_filter_ids or [])
        get_filter_functions = getattr(filter_utils, "get_filter_functions", None)
        if callable(get_filter_functions):
            filter_functions = await _inlet_filters_maybe_await(
                get_filter_functions(request, model, enabled_filter_ids)
            )
        else:
            from open_webui.models.functions import Functions

            filter_ids = await _inlet_filters_maybe_await(
                filter_utils.get_sorted_filter_ids(
                    request,
                    model,
                    enabled_filter_ids,
                )
            )
            filter_functions = []
            for filter_id in filter_ids:
                function = await _inlet_filters_maybe_await(
                    Functions.get_function_by_id(filter_id)
                )
                if function:
                    filter_functions.append(function)

        return route | {
            "process": process_filter_functions,
            "functions": filter_functions,
            "context": context_factory() if request_filters_supported else None,
            "supports_context": supports_context,
            "supports_request": request_filters_supported,
        }
    except Exception as exc:
        _inlet_filters_log.warning(f"Error resolving model filters: {exc}")
        if request_filters_supported:
            raise
        return route


async def _apply_filter_pipeline(
    filter_pipeline: dict[str, Any] | None,
    request: Request,
    form_data: dict,
    extra_params: dict,
    filter_type: str,
) -> dict:
    if filter_pipeline is None:
        return form_data

    if filter_type == "inlet":
        form_data["model"] = filter_pipeline["model_id"]

    if filter_pipeline["process"] is None or (
        filter_type == "request" and not filter_pipeline["supports_request"]
    ):
        return form_data

    try:
        process_filter_functions = filter_pipeline["process"]

        # Isolate __user__ so filter UserValves injection doesn't leak out
        # and pollute subsequent tool calls under a different tool id.
        local_extra_params = dict(extra_params or {})
        local_extra_params["__model__"] = filter_pipeline["model"]
        if isinstance(local_extra_params.get("__user__"), dict):
            local_extra_params["__user__"] = dict(local_extra_params["__user__"])

        process_kwargs: dict[str, Any] = {
            "request": request,
            "filter_functions": filter_pipeline["functions"],
            "filter_type": filter_type,
            "form_data": form_data,
            "extra_params": local_extra_params,
        }
        if filter_pipeline["supports_context"]:
            process_kwargs["filter_context"] = filter_pipeline["context"]
        form_data, _ = await process_filter_functions(
            **process_kwargs,
        )
    except Exception as exc:
        _inlet_filters_log.warning(f"Error applying {filter_type} filters: {exc}")
        if filter_type == "request":
            raise
    return form_data


async def apply_inlet_filters_if_enabled(
    filter_pipeline: dict[str, Any] | None,
    request: Request,
    form_data: dict,
    extra_params: dict,
) -> dict:
    return await _apply_filter_pipeline(
        filter_pipeline, request, form_data, extra_params, "inlet"
    )


async def finalize_model_request(
    filter_pipeline: dict[str, Any] | None,
    request: Request,
    form_data: dict,
    extra_params: dict,
) -> dict:
    """Apply request filters immediately before model dispatch."""
    return await _apply_filter_pipeline(
        filter_pipeline, request, form_data, extra_params, "request"
    )

# --- inlined from src/owui_ext/shared/loop_compaction.py (owui_ext.shared.loop_compaction) ---
import hashlib
import json
import math
from contextlib import suppress
from typing import Any, Literal
TOKEN_ESTIMATOR_VERSION = "message-sanitized-media-json-v3"
MESSAGE_TOKEN_OVERHEAD = 4
REQUEST_TOKEN_OVERHEAD = 3
MESSAGE_TOKEN_ESTIMATE_CACHE_MAX_ENTRIES = 8192
MESSAGE_TOKEN_EXACT_ENCODE_MAX_BYTES = 64 * 1024
MESSAGE_TOKEN_SAMPLE_MAX_BYTES = 16 * 1024
MESSAGE_TOKEN_IMAGE_OVERHEAD = 1000
BODY_TOKEN_EXTRA_KEYS = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "parallel_tool_calls",
)
MESSAGE_TOKEN_ESTIMATE_CACHE: dict[tuple[str, str, str], int] = {}


def _json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


_STABLE_FILE_ATTACHMENT_KEYS = {
    "collection_name",
    "collection_names",
    "content",
    "content_type",
    "context",
    "docs",
    "file",
    "id",
    "legacy",
    "name",
    "queries",
    "type",
    "url",
    "urls",
}

_STABLE_EMBEDDED_FILE_KEYS = {
    "collection_name",
    "collection_names",
    "content_type",
    "context",
    "data",
    "file_hash",
    "file_id",
    "filename",
    "hash",
    "id",
    "legacy",
    "meta",
    "metadata",
    "mime_type",
    "name",
    "type",
    "url",
}
_STABLE_FILE_DATA_KEYS = {
    "content",
    "metadata",
}
_FILE_METADATA_TRANSIENT_KEYS = {
    "blob_url",
    "created_at",
    "download_url",
    "error",
    "headers",
    "itemId",
    "item_id",
    "path",
    "preview_url",
    "size",
    "signed_url",
    "status",
    "temp_id",
    "thumbnail_url",
    "tmp_path",
    "updated_at",
    "upload_id",
}

_TRANSIENT_SOURCE_KEYS = {
    "distances",
}

_PROVIDER_PROMPT_CACHE_HINT_KEYS = {
    "cache_control",
    "cacheControl",
}

_FILE_CONTENT_PART_TYPES = {
    "file",
    "input_file",
}

_TOKEN_RAW_MEDIA_BODY_KEYS = {
    "base64",
    "body",
    "bytes",
    "buffer",
    "content",
    "context",
    "data",
    "docs",
    "document",
    "documents",
    "fileData",
    "file_data",
}
_TOKEN_MEDIA_CONTENT_PART_TYPES = {
    "file",
    "image",
    "image_url",
    "input_audio",
    "input_file",
    "input_image",
}
_STABLE_MESSAGE_KEYS = {
    "role",
    "content",
    "name",
    "tool_call_id",
    "tool_calls",
    "function_call",
    "files",
    "sources",
    "reasoning_content",
}
_TOKEN_MESSAGE_KEYS = _STABLE_MESSAGE_KEYS | {"reasoning_details", "thinking"}

def _is_empty_canonical_value(value: Any) -> bool:
    return value in (None, {}, [])


def _canonicalize_general_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            item = _canonicalize_general_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            canonical_item = _canonicalize_general_value(item)
            if not _is_empty_canonical_value(canonical_item):
                out.append(canonical_item)
        return out
    return value

def _canonicalize_file_metadata_map(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key in _FILE_METADATA_TRANSIENT_KEYS:
                continue
            item = _canonicalize_file_metadata_map(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            canonical_item = _canonicalize_file_metadata_map(item)
            if not _is_empty_canonical_value(canonical_item):
                out.append(canonical_item)
        return out
    return value

def _canonicalize_file_data_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key not in _STABLE_FILE_DATA_KEYS:
                continue
            item = _canonicalize_file_metadata_map(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    return _canonicalize_file_metadata_map(value)

def _canonicalize_embedded_file_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key not in _STABLE_EMBEDDED_FILE_KEYS:
                continue
            if key == "data":
                item = _canonicalize_file_data_value(value[key])
            elif key in {"meta", "metadata"}:
                item = _canonicalize_file_metadata_map(value[key])
            else:
                item = _canonicalize_general_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            canonical_item = _canonicalize_embedded_file_value(item)
            if not _is_empty_canonical_value(canonical_item):
                out.append(canonical_item)
        return out
    return _canonicalize_general_value(value)

def _canonicalize_file_attachment_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key not in _STABLE_FILE_ATTACHMENT_KEYS:
                continue
            if key == "file":
                item = _canonicalize_embedded_file_value(value[key])
            else:
                item = _canonicalize_general_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            canonical_item = _canonicalize_file_attachment_value(item)
            if not _is_empty_canonical_value(canonical_item):
                out.append(canonical_item)
        return out
    return _canonicalize_general_value(value)

def _is_file_content_part(value: dict[str, Any]) -> bool:
    part_type = value.get("type")
    return isinstance(part_type, str) and part_type in _FILE_CONTENT_PART_TYPES and "file" in value


def _canonicalize_content_part(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        is_file_part = _is_file_content_part(value)
        for key in sorted(value.keys()):
            if key in _PROVIDER_PROMPT_CACHE_HINT_KEYS:
                continue
            if is_file_part and key == "file":
                item = _canonicalize_embedded_file_value(value[key])
            elif key == "content" and isinstance(value[key], list):
                item = _canonicalize_content_value(value[key])
            else:
                item = _canonicalize_general_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    return _canonicalize_general_value(value)

def _collapse_text_only_content_part(part: Any) -> str | None:
    if not isinstance(part, dict) or part.get("type") != "text":
        return None
    if set(part.keys()) - {"type", "text"}:
        return None
    text = part.get("text", "")
    return text if isinstance(text, str) else None


def _canonicalize_content_value(value: Any) -> Any:
    if isinstance(value, list):
        out = []
        for item in value:
            canonical_item = _canonicalize_content_part(item)
            if not _is_empty_canonical_value(canonical_item):
                out.append(canonical_item)
        # Filters attaching prompt-cache hints must wrap str content in a
        # single text part; hash it as the equivalent plain string so the
        # canonical identity survives the wrap/unwrap across turns.
        if len(out) == 1:
            collapsed = _collapse_text_only_content_part(out[0])
            if collapsed is not None:
                return collapsed
        return out
    return _canonicalize_content_part(value)

def _canonicalize_files_value(value: Any) -> Any:
    return _canonicalize_file_attachment_value(value)


def _canonicalize_source_value(value: Any, *, source_root: bool = False) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if source_root and key in _TRANSIENT_SOURCE_KEYS:
                continue
            item = _canonicalize_general_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    return _canonicalize_general_value(value)


def _canonicalize_sources_value(value: Any) -> Any:
    if isinstance(value, list):
        out = []
        for item in value:
            canonical_item = _canonicalize_source_value(item, source_root=True)
            if not _is_empty_canonical_value(canonical_item):
                out.append(canonical_item)
        return out
    return _canonicalize_source_value(value, source_root=True)


def _canonicalize_tool_definition_for_token_extra(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key in _PROVIDER_PROMPT_CACHE_HINT_KEYS:
                continue
            item = _canonicalize_general_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    return _canonicalize_general_value(value)


def _canonicalize_tools_for_token_extra(value: Any) -> Any:
    if isinstance(value, list):
        out = []
        for item in value:
            canonical_item = _canonicalize_tool_definition_for_token_extra(item)
            if not _is_empty_canonical_value(canonical_item):
                out.append(canonical_item)
        return out
    return _canonicalize_tool_definition_for_token_extra(value)

def _canonicalize_message_value(key: str, value: Any) -> Any:
    if key == "content":
        return _canonicalize_content_value(value)
    if key == "files":
        return _canonicalize_files_value(value)
    if key == "sources":
        return _canonicalize_sources_value(value)
    return _canonicalize_general_value(value)

def canonicalize_message_for_token_estimate(message: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key in sorted(message.keys()):
        if key not in _TOKEN_MESSAGE_KEYS:
            continue
        value = _canonicalize_message_value(key, message[key])
        if _is_empty_canonical_value(value):
            continue
        canonical[key] = value
    canonical.setdefault("role", message.get("role", "assistant"))
    canonical.setdefault("content", "")
    return canonical

def _is_image_file_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") == "image":
        return True
    content_type = item.get("content_type")
    return isinstance(content_type, str) and content_type.startswith("image/")


def _strip_raw_media_payload_fields_for_token_text(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key == "data":
                item = _strip_raw_media_payload_fields_for_token_text(value[key])
                if isinstance(item, dict) and not _is_empty_canonical_value(item):
                    out[key] = item
                continue
            if key in _TOKEN_RAW_MEDIA_BODY_KEYS:
                continue
            item = _strip_raw_media_payload_fields_for_token_text(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            sanitized = _strip_raw_media_payload_fields_for_token_text(item)
            if not _is_empty_canonical_value(sanitized):
                out.append(sanitized)
        return out
    return value

def _sanitize_media_content_part_for_token_text(part: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(part.keys()):
        if key in _PROVIDER_PROMPT_CACHE_HINT_KEYS or key in _TOKEN_RAW_MEDIA_BODY_KEYS:
            continue
        item = _strip_raw_media_payload_fields_for_token_text(part[key])
        if _is_empty_canonical_value(item):
            continue
        out[key] = item
    return out

def _sanitize_media_payloads_for_token_text(canonical: dict[str, Any]) -> int:
    """Remove raw media/file payload bytes from token-text canonical JSON.

    Image content parts and image file attachments are counted via
    ``MESSAGE_TOKEN_IMAGE_OVERHEAD``. Non-image file/audio bodies keep bounded
    metadata but drop raw content before encoder sizing. Only the encoder-facing
    text copy is mutated; source-hash and cache-key canonicalization build their
    own copies and keep the full payload.
    """
    count = 0
    content = canonical.get("content")
    if isinstance(content, list):
        kept_content: list[Any] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type in {"image", "image_url", "input_image"}:
                    count += 1
                    continue
                if part_type in _TOKEN_MEDIA_CONTENT_PART_TYPES:
                    kept_content.append(_sanitize_media_content_part_for_token_text(part))
                    continue
            kept_content.append(part)
        canonical["content"] = kept_content
    files = canonical.get("files")
    if isinstance(files, list):
        kept_files: list[Any] = []
        for item in files:
            if isinstance(item, dict) and _is_image_file_item(item):
                count += 1
                continue
            kept_files.append(_strip_raw_media_payload_fields_for_token_text(item))
        canonical["files"] = kept_files
    return count

def _configured_tiktoken_encoding_names(request: Any = None) -> list[str]:
    names: list[str] = []
    config = getattr(getattr(getattr(request, "app", None), "state", None), "config", None)
    configured = getattr(config, "TIKTOKEN_ENCODING_NAME", None)
    if configured:
        names.append(str(configured))
    try:
        from open_webui import config as open_webui_config

        fallback = getattr(open_webui_config, "TIKTOKEN_ENCODING_NAME", None)
        if fallback:
            names.append(str(fallback))
    except Exception:
        pass
    names.append("cl100k_base")
    return list(dict.fromkeys(names))

def _get_tiktoken_encoder(request: Any = None) -> tuple[Any | None, str | None]:
    try:
        import tiktoken
    except Exception:
        return None, None

    for encoding_name in _configured_tiktoken_encoding_names(request):
        try:
            return tiktoken.get_encoding(encoding_name), encoding_name
        except Exception:
            continue
    return None, None

def _message_token_cache_key(message: dict[str, Any], *, encoding_name: str) -> tuple[str, str, str]:
    canonical = canonicalize_message_for_token_estimate(message)
    return (
        TOKEN_ESTIMATOR_VERSION,
        encoding_name,
        _json_hash({"family": TOKEN_ESTIMATOR_VERSION, "message": canonical}),
    )

def _message_token_image_count_and_text(message: dict[str, Any]) -> tuple[int, str]:
    canonical = canonicalize_message_for_token_estimate(message)
    image_count = _sanitize_media_payloads_for_token_text(canonical)
    text = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return image_count, text


def _remember_message_token_estimate(key: tuple[str, str, str], count: int) -> None:
    if len(MESSAGE_TOKEN_ESTIMATE_CACHE) >= MESSAGE_TOKEN_ESTIMATE_CACHE_MAX_ENTRIES:
        with suppress(Exception):
            MESSAGE_TOKEN_ESTIMATE_CACHE.pop(next(iter(MESSAGE_TOKEN_ESTIMATE_CACHE)))
    MESSAGE_TOKEN_ESTIMATE_CACHE[key] = int(count)

def _estimate_large_text_tokens_sampling(
    text: str,
    *,
    encoder: Any,
    overhead: int = MESSAGE_TOKEN_OVERHEAD,
) -> int | None:
    """Estimate token count for large text via 3-point sampling.

    Samples head, middle, and tail regions (each up to
    ``MESSAGE_TOKEN_SAMPLE_MAX_BYTES`` bytes), encodes them exactly
    with the tiktoken encoder, and extrapolates the aggregate
    bytes/tokens ratio to the full text.

    Returns ``None`` if encoding fails for any sample.
    """
    total_bytes = len(text.encode("utf-8", errors="ignore"))
    if total_bytes == 0:
        return int(overhead)

    total_chars = len(text)
    if total_chars == 0:
        return int(overhead)

    # Approximate chars-per-sample to stay within the byte budget.
    bytes_per_char = total_bytes / total_chars
    sample_chars = max(1, int(MESSAGE_TOKEN_SAMPLE_MAX_BYTES / bytes_per_char))

    # Three sampling regions: head, middle, tail.
    regions = [
        (0, min(sample_chars, total_chars)),
        (
            max(0, total_chars // 2 - sample_chars // 2),
            min(total_chars // 2 + sample_chars // 2, total_chars),
        ),
        (max(0, total_chars - sample_chars), total_chars),
    ]

    total_sample_bytes = 0
    total_sample_tokens = 0

    for start, end in regions:
        if end <= start:
            continue
        sample = text[start:end]
        sample_bytes = len(sample.encode("utf-8", errors="ignore"))
        if sample_bytes == 0:
            continue
        sample_tokens = _encode_text_token_count(encoder, sample)
        if sample_tokens is None:
            return None
        total_sample_bytes += sample_bytes
        total_sample_tokens += sample_tokens

    if total_sample_bytes == 0 or total_sample_tokens == 0:
        return int(overhead)

    bytes_per_token = total_sample_bytes / total_sample_tokens
    return math.ceil(total_bytes / bytes_per_token) + int(overhead)


def _encode_text_token_count(encoder: Any, text: str) -> int | None:
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except TypeError:
        try:
            return len(encoder.encode(text))
        except Exception:
            return None
    except Exception:
        return None

def _estimate_text_tokens_with_encoder(
    text: str,
    *,
    encoder: Any,
    overhead: int = MESSAGE_TOKEN_OVERHEAD,
) -> int | None:
    if len(text.encode("utf-8", errors="ignore")) > MESSAGE_TOKEN_EXACT_ENCODE_MAX_BYTES:
        return _estimate_large_text_tokens_sampling(
            text, encoder=encoder, overhead=overhead
        )
    count = _encode_text_token_count(encoder, text)
    if count is None:
        return None
    return count + int(overhead)

def _estimate_message_tokens_with_encoder(
    message: dict[str, Any],
    *,
    encoder: Any,
    encoding_name: str,
) -> int | None:
    key = _message_token_cache_key(message, encoding_name=encoding_name)
    cached = MESSAGE_TOKEN_ESTIMATE_CACHE.get(key)
    if cached is not None:
        return cached
    image_count, text = _message_token_image_count_and_text(message)
    count = _estimate_text_tokens_with_encoder(
        text,
        encoder=encoder,
        overhead=MESSAGE_TOKEN_OVERHEAD,
    )
    if count is None:
        return None
    if image_count:
        count += image_count * MESSAGE_TOKEN_IMAGE_OVERHEAD
    _remember_message_token_estimate(key, count)
    return count


def estimate_messages_tokens(
    messages: list[dict[str, Any]],
    *,
    request: Any = None,
    encoder: Any = None,
    encoding_name: str | None = None,
) -> int | None:
    if not isinstance(messages, list):
        return None
    resolved_encoder = encoder
    resolved_encoding_name = encoding_name
    if resolved_encoder is None:
        resolved_encoder, resolved_encoding_name = _get_tiktoken_encoder(request)
    if resolved_encoder is None:
        return None
    if not resolved_encoding_name:
        resolved_encoding_name = str(getattr(resolved_encoder, "name", "unknown"))

    total = REQUEST_TOKEN_OVERHEAD
    for message in messages:
        if not isinstance(message, dict):
            continue
        count = _estimate_message_tokens_with_encoder(
            message,
            encoder=resolved_encoder,
            encoding_name=str(resolved_encoding_name),
        )
        if count is None:
            return None
        total += count
    return total

def _body_token_extra_payload(body: dict[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for key in BODY_TOKEN_EXTRA_KEYS:
        if key not in body:
            continue
        if key == "tools":
            value = _canonicalize_tools_for_token_extra(body.get(key))
        else:
            value = _canonicalize_general_value(body.get(key))
        if _is_empty_canonical_value(value):
            continue
        extra[key] = value
    options = body.get("options")
    if isinstance(options, dict) and "think" in options:
        think = _canonicalize_general_value(options.get("think"))
        if not _is_empty_canonical_value(think):
            extra["think"] = think
    return extra


def _body_token_extra_text(body: dict[str, Any]) -> str:
    extra = _body_token_extra_payload(body)
    if not extra:
        return ""
    return json.dumps(
        {"family": TOKEN_ESTIMATOR_VERSION, "body_extra": extra},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

def estimate_body_tokens(
    body: dict[str, Any],
    *,
    request: Any = None,
    encoder: Any = None,
    encoding_name: str | None = None,
) -> int | None:
    if not isinstance(body, dict):
        return None
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    resolved_encoder = encoder
    resolved_encoding_name = encoding_name
    if resolved_encoder is None:
        resolved_encoder, resolved_encoding_name = _get_tiktoken_encoder(request)
    if resolved_encoder is None:
        return None
    if not resolved_encoding_name:
        resolved_encoding_name = str(getattr(resolved_encoder, "name", "unknown"))

    total = estimate_messages_tokens(
        messages,
        request=request,
        encoder=resolved_encoder,
        encoding_name=str(resolved_encoding_name),
    )
    if total is None:
        return None
    extra_text = _body_token_extra_text(body)
    if extra_text:
        extra_tokens = _estimate_text_tokens_with_encoder(
            extra_text,
            encoder=resolved_encoder,
            overhead=MESSAGE_TOKEN_OVERHEAD,
        )
        if extra_tokens is None:
            return None
        total += extra_tokens
    return total


# ---------------------------------------------------------------------------
# Usage anchor raw-key interpretation
# ---------------------------------------------------------------------------


def _strict_usage_token_value(usage: dict[str, Any], key: str) -> int | None:
    if key not in usage:
        return None
    value = usage.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or int(value) != value
    ):
        return None
    value = int(value)
    return value if value >= 0 else None


def usage_input_tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict) or not usage:
        return None

    def positive(value: int | None) -> int | None:
        return value if value is not None and value > 0 else None

    if "prompt_tokens" in usage:
        return positive(_strict_usage_token_value(usage, "prompt_tokens"))

    if "prompt_eval_count" in usage:
        return positive(_strict_usage_token_value(usage, "prompt_eval_count"))

    if "prompt_n" in usage:
        prompt_n = _strict_usage_token_value(usage, "prompt_n")
        cache_n = _strict_usage_token_value(usage, "cache_n")
        if prompt_n is None or cache_n is None:
            return None
        return positive(prompt_n + cache_n)

    if "input_tokens" in usage:
        input_tokens = _strict_usage_token_value(usage, "input_tokens")
        if input_tokens is None:
            return None
        cache_creation = _strict_usage_token_value(usage, "cache_creation_input_tokens")
        cache_read = _strict_usage_token_value(usage, "cache_read_input_tokens")
        if "cache_creation_input_tokens" in usage and cache_creation is None:
            return None
        if "cache_read_input_tokens" in usage and cache_read is None:
            return None
        return positive(input_tokens + (cache_creation or 0) + (cache_read or 0))
    return None


def response_usage(response: Any) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else None


# ---------------------------------------------------------------------------
# Anchor + delta fingerprint
# ---------------------------------------------------------------------------


def build_loop_input_fingerprint(
    *,
    model_id: str,
    tools_param: Any,
    filter_identity: Any,
    tool_server_prompt_signature: Any,
    stable_messages: list[dict[str, Any]],
    body_extras: dict[str, Any] | None = None,
) -> str:
    payload = {
        "family": "agent-loop-input-v1",
        "model": str(model_id or ""),
        "tools": _canonicalize_tools_for_token_extra(tools_param or []),
        "filter_identity": _canonicalize_general_value(filter_identity),
        "tool_server_prompts": _canonicalize_general_value(tool_server_prompt_signature),
        "message_count": len(stable_messages),
        "messages_hash": _json_hash(
            [canonicalize_message_for_token_estimate(message) for message in stable_messages]
        ),
        "body_extras": _canonicalize_general_value(body_extras or {}),
    }
    return _json_hash(payload)


class LoopUsageAnchor:
    """Observed provider input-token anchor for anchor+delta estimation."""

    __slots__ = (
        "input_tokens",
        "stable_message_count",
        "input_fingerprint",
        "volatile_message_tokens",
    )

    def __init__(
        self,
        *,
        input_tokens: int,
        stable_message_count: int,
        input_fingerprint: str,
        volatile_message_tokens: int,
    ) -> None:
        self.input_tokens = int(input_tokens)
        self.stable_message_count = int(stable_message_count)
        self.input_fingerprint = str(input_fingerprint)
        self.volatile_message_tokens = int(volatile_message_tokens)


def estimate_with_anchor(
    anchor: LoopUsageAnchor | None,
    *,
    current_fingerprint: str,
    current_messages: list[dict[str, Any]],
    current_volatile_tokens: int,
    suffix_token_estimate: int | None,
    full_estimate: int | None,
) -> int | None:
    """Anchor+delta estimate; falls back to ``full_estimate`` on any miss."""
    if (
        anchor is None
        or anchor.input_tokens <= 0
        or anchor.input_fingerprint != current_fingerprint
        or len(current_messages) < anchor.stable_message_count
        or suffix_token_estimate is None
    ):
        return full_estimate
    return (
        anchor.input_tokens
        - anchor.volatile_message_tokens
        + current_volatile_tokens
        + suffix_token_estimate
    )


# ---------------------------------------------------------------------------
# Compaction cut (system + task user + last N complete rounds)
# ---------------------------------------------------------------------------

LOOP_COMPACTION_KEEP_ROUNDS = 2


class LoopCompactionCut:
    __slots__ = (
        "preserved_system_message",
        "task_user_message",
        "summarization_prefix",
        "tail_messages",
        "source_message_count",
    )

    def __init__(
        self,
        *,
        preserved_system_message: dict[str, Any] | None,
        task_user_message: dict[str, Any],
        summarization_prefix: list[dict[str, Any]],
        tail_messages: list[dict[str, Any]],
        source_message_count: int,
    ) -> None:
        self.preserved_system_message = preserved_system_message
        self.task_user_message = task_user_message
        self.summarization_prefix = summarization_prefix
        self.tail_messages = tail_messages
        self.source_message_count = source_message_count


def _tool_call_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                ids.add(call["id"])
    return ids


def _assistant_tool_call_ids(messages: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            ids.update(_tool_call_ids(message))
    return ids


def has_orphan_tool_messages(messages: list[dict[str, Any]]) -> bool:
    assistant_ids = _assistant_tool_call_ids(messages)
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") not in assistant_ids:
            return True
    return False


def select_loop_compaction_cut(
    messages: list[dict[str, Any]],
    *,
    completed_rounds_to_keep: int = LOOP_COMPACTION_KEEP_ROUNDS,
) -> LoopCompactionCut | None:
    if not messages or has_orphan_tool_messages(messages):
        return None

    system_index = next(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "system"),
        None,
    )
    preserved_system = messages[system_index] if system_index is not None else None
    working_start = (system_index + 1) if system_index is not None else 0
    working = messages[working_start:]

    task_user_index = next(
        (i for i, m in enumerate(working) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if task_user_index is None:
        return None
    task_user_message = working[task_user_index]
    loop_messages = working[task_user_index + 1 :]
    if not loop_messages:
        return None

    complete_round_starts: list[int] = []
    round_end_by_start: dict[int, int] = {}
    for index, message in enumerate(loop_messages):
        if message.get("role") != "assistant":
            continue
        ids = _tool_call_ids(message)
        if not ids:
            continue
        seen: set[str] = set()
        end = index
        for follower_index in range(index + 1, len(loop_messages)):
            follower = loop_messages[follower_index]
            if follower.get("role") != "tool":
                break
            if follower.get("tool_call_id") in ids:
                seen.add(follower.get("tool_call_id"))
            end = follower_index
        if ids <= seen:
            complete_round_starts.append(index)
            round_end_by_start[index] = end

    if len(complete_round_starts) <= completed_rounds_to_keep:
        return None

    keep_from = complete_round_starts[-completed_rounds_to_keep]
    summarization_prefix = loop_messages[:keep_from]
    tail_messages = loop_messages[keep_from:]
    if not summarization_prefix or not tail_messages:
        return None

    return LoopCompactionCut(
        preserved_system_message=preserved_system,
        task_user_message=task_user_message,
        summarization_prefix=summarization_prefix,
        tail_messages=tail_messages,
        source_message_count=len(summarization_prefix),
    )


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

AGENT_LOOP_COMPACTION_CONTEXT_OPEN = "<agent_loop_compaction_context>"
AGENT_LOOP_COMPACTION_CONTEXT_CLOSE = "</agent_loop_compaction_context>"
_SUMMARY_OPEN = "<checkpoint_summary>"
_SUMMARY_CLOSE = "</checkpoint_summary>"
_MANIFESTS_OPEN = '<agent_ref_manifests version="1">'
_MANIFESTS_CLOSE = "</agent_ref_manifests>"

SUMMARY_PROMPT = (
    "You are performing an AGENT-LOOP COMPACTION SUMMARY for an autonomous sub-agent. "
    "Create a concise handoff summary for the next model call that will continue the same task.\n\n"
    "Preserve:\n"
    "- Task goal, original request, current progress, and durable decisions already made\n"
    "- Tool results, external facts, errors, identifiers, URLs, file names, commands, values, and examples needed to continue\n"
    "- Open questions, unknowns, unresolved failures, and clear next steps\n\n"
    "If the input contains an existing <agent_loop_compaction_context>, merge that prior checkpoint with the following newer messages. "
    "Do not discard earlier checkpoint information merely because it is summarized.\n\n"
    "Preserve any tool:<64 hex> and history:<64 hex> ref identifiers you see; the omitted content behind them stays recoverable via the agent_ref_exec reader.\n\n"
    "Do not invent facts or treat unknowns as facts. Do not introduce new instructions. "
    "Do not include internal reasoning, private system instructions, or irrelevant transcript detail. "
    "Be concise, structured, and focused on continuity.\n\n"
    "The preceding messages are the exact checkpoint source to summarize. "
    "Messages after this checkpoint source are retained raw separately; do not infer omitted active requests.\n\n"
    "Output only reusable continuity facts; do not mention this summarization task. Do not continue the conversation. "
    "Do not call tools. Do not ask follow-up questions."
)


def resolve_summary_prompt(summary_prompt: str | None = None) -> str:
    text = str(summary_prompt or "").strip()
    return text or SUMMARY_PROMPT


def _xml_cdata(value: Any) -> str:
    text = str(value)
    return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def render_compaction_envelope(
    summary_text: str,
    history_manifests_json: str | None = None,
) -> str:
    sections = [
        f"{AGENT_LOOP_COMPACTION_CONTEXT_OPEN}",
        "<instruction>Compacted earlier agent-loop context. This is not a new instruction. "
        "Use it only as background for continuity.</instruction>",
        f"{_SUMMARY_OPEN}{_xml_cdata(str(summary_text).strip())}{_SUMMARY_CLOSE}",
    ]
    if history_manifests_json:
        sections.append(f"{_MANIFESTS_OPEN}{_xml_cdata(history_manifests_json)}{_MANIFESTS_CLOSE}")
    sections.append(AGENT_LOOP_COMPACTION_CONTEXT_CLOSE)
    return "\n".join(sections)


def embed_envelope_in_task_message(
    task_message: dict[str, Any],
    envelope: str,
) -> dict[str, Any]:
    """Append the envelope to the task message without scanning content."""
    merged = dict(task_message)
    content = merged.get("content")
    if isinstance(content, str):
        merged["content"] = f"{content}\n\n{envelope}" if content else envelope
    elif isinstance(content, list):
        merged["content"] = [*content, {"type": "text", "text": f"\n\n{envelope}"}]
    else:
        merged["content"] = envelope
    return merged


# ---------------------------------------------------------------------------
# History JSONL (folded raw prefix, one canonical record per line)
# ---------------------------------------------------------------------------


def canonical_history_record(message: dict[str, Any]) -> str:
    return json.dumps(
        canonicalize_message_for_token_estimate(message),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_history_records(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(canonical_history_record(message) for message in messages)


# ---------------------------------------------------------------------------
# Summarizer outcome classification
# ---------------------------------------------------------------------------


_SUMMARY_INCOMPLETE_FINISH_REASON_PATTERNS = (
    "length",
    "content_filter",
    "max_token",
    "max_output",
    "max output",
    "truncat",
    "incomplete",
)


def _summary_incomplete_finish_reason(reason: Any) -> str | None:
    if not isinstance(reason, str) or not reason:
        return None
    normalized = reason.lower().replace("-", "_")
    if any(
        pattern in normalized
        for pattern in _SUMMARY_INCOMPLETE_FINISH_REASON_PATTERNS
    ):
        return reason
    return None


def summary_response_incomplete_reason(response: Any) -> str | None:
    """Return the finish reason if the summarizer stopped mid-generation."""
    if not isinstance(response, dict):
        return None
    reason = _summary_incomplete_finish_reason(response.get("finish_reason"))
    if reason is not None:
        return reason
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            return _summary_incomplete_finish_reason(choice.get("finish_reason"))
    return None


def summary_choice_has_tool_calls(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    choices = response.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and message.get("tool_calls"):
            return True
    return False


def extract_summary_text(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


SummaryFailureKind = Literal["tool_call", "transient", "fatal"]


def classify_provider_failure(
    *,
    status_code: int | None = None,
    error_text: str = "",
) -> SummaryFailureKind:
    text = str(error_text or "").lower()
    if status_code is not None and 500 <= status_code <= 599:
        return "transient"
    if status_code == 429:
        return "transient"
    if any(
        marker in text
        for marker in (
            "status 500",
            "status 502",
            "status 503",
            "status 504",
            "status 429",
            "rate limit",
            "temporarily unavailable",
            "connection reset",
            "timeout",
        )
    ):
        return "transient"
    return "fatal"


resolve_tiktoken_encoder = _get_tiktoken_encoder

# --- inlined from src/owui_ext/shared/model_features.py (owui_ext.shared.model_features) ---
from typing import Optional
def _attached_knowledge_types(
    model: Optional[dict],
    metadata: Optional[dict] = None,
) -> set[str]:
    if not isinstance(model, dict):
        model = {}
    if not isinstance(metadata, dict):
        metadata = {}

    model_meta = model.get("info", {}).get("meta", {})
    knowledge_items = list(model_meta.get("knowledge") or [])
    knowledge_items.extend(metadata.get("folder_knowledge") or [])
    if not (model_meta.get("capabilities") or {}).get("file_context", True):
        knowledge_items.extend(
            item
            for item in metadata.get("files") or []
            if isinstance(item, dict)
            and item.get("type") in ("collection", "note")
        )

    return {
        item["type"]
        for item in knowledge_items
        if isinstance(item, dict)
        and isinstance(item.get("type"), str)
        and item.get("id")
    }


def model_has_note_knowledge(
    model: Optional[dict],
    metadata: Optional[dict] = None,
) -> bool:
    """Return True if Core can expose view_note for attached knowledge."""
    return "note" in _attached_knowledge_types(model, metadata)


def model_has_file_knowledge(
    model: Optional[dict],
    metadata: Optional[dict] = None,
) -> bool:
    """Return True if Core can expose view_file for attached knowledge."""
    return not _attached_knowledge_types(model, metadata).isdisjoint(
        {"file", "collection"}
    )


def model_knowledge_tools_enabled(model: Optional[dict]) -> bool:
    """Return True if model-level builtin knowledge tools are enabled."""
    if not isinstance(model, dict):
        return True
    builtin_tools = model.get("info", {}).get("meta", {}).get("builtinTools", {})
    if not isinstance(builtin_tools, dict):
        return True
    return bool(builtin_tools.get("knowledge", True))

# --- inlined from src/owui_ext/shared/notifications.py (owui_ext.shared.notifications) ---
import logging
from typing import Callable, Optional
_notifications_log = logging.getLogger("owui_ext.shared.notifications")


async def emit_notification(
    event_emitter: Optional[Callable], *, level: str, content: str
) -> None:
    """Emit a frontend notification toast when the current chat supports it."""
    if not callable(event_emitter):
        return
    if not isinstance(content, str) or not content.strip():
        return
    try:
        await event_emitter(
            {
                "type": "notification",
                "data": {"type": level, "content": content.strip()},
            }
        )
    except Exception as exc:
        _notifications_log.debug(
            f"Error emitting notification ({level}): {exc}"
        )

# --- inlined from src/owui_ext/shared/prompt_utils.py (owui_ext.shared.prompt_utils) ---
from typing import Optional
def merge_prompt_sections(*sections: Optional[str]) -> str:
    """Join non-empty prompt sections with blank lines."""
    merged_sections = []
    for section in sections:
        if not isinstance(section, str):
            continue
        stripped = section.strip()
        if stripped:
            merged_sections.append(stripped)
    return "\n\n".join(merged_sections)


def truncate_text(value: str, limit: int = 200) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _append_tool_server_prompts(form_data: dict, extra_params: dict) -> dict:
    """Append terminal/direct-tool-server system prompts to messages.

    Open WebUI core injects these prompts AFTER inlet filters so they survive
    filters that rewrite the system message.  We replicate the same ordering by
    calling this helper after ``apply_inlet_filters_if_enabled``.
    """
    prompts: list[str] = []
    terminal_prompt = (extra_params or {}).get("__terminal_system_prompt__")
    if isinstance(terminal_prompt, str) and terminal_prompt.strip():
        prompts.append(terminal_prompt)
    direct_prompts = (extra_params or {}).get(
        "__direct_tool_server_system_prompts__", []
    )
    if isinstance(direct_prompts, list):
        prompts.extend(p for p in direct_prompts if isinstance(p, str) and p.strip())
    if not prompts:
        return form_data
    messages = list(form_data.get("messages", []))
    combined = "\n\n".join(prompts)
    if messages and messages[0].get("role") == "system":
        msg = {**messages[0]}
        content = msg.get("content", "")
        if isinstance(content, list):
            msg["content"] = [
                (
                    {**item, "text": f"{item['text']}\n{combined}"}
                    if item.get("type") == "text"
                    else item
                )
                for item in content
            ]
        else:
            msg["content"] = f"{content}\n\n{combined}" if content else combined
        messages[0] = msg
    else:
        messages.insert(0, {"role": "system", "content": combined})
    form_data["messages"] = messages
    return form_data

# --- inlined from src/owui_ext/shared/ref_exec.py (owui_ext.shared.ref_exec) ---
import asyncio
import codecs
import copy
import hashlib
import json
import logging
import re
import shlex
import threading
import time
from array import array
from bisect import bisect_left, bisect_right
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Iterable, Literal, TypeAlias
try:
    import regex as _REGEX
except ImportError:
    _REGEX = None

ref_exec_log = logging.getLogger("owui_ext.shared.ref_exec")

LOG = ref_exec_log

RefKind = Literal["history", "tool"]

REF_TEXT_HASH_CHUNK_CHARS = 16 * 1024
REF_EXEC_TOOL_NAME = "agent_ref_exec"
REF_EXEC_COMMAND_MAX_BYTES = 1_024
REF_EXEC_RESPONSE_MAX_BYTES = 65_536
REF_EXEC_TAIL_MAX_BYTES = 8 * 1024 * 1024
# Retained line objects cost ~200 bytes each regardless of text length,
# so cap the window by line count (~15 MiB) as well.
REF_EXEC_TAIL_MAX_LINES = 65_536
REF_EXEC_USAGE_ERROR = (
    "Error: usage: agent_ref_exec(command). Expected REF: "
    "tool:<64 hex> or history:<64 hex>"
)
REF_EXEC_REGEX_BUDGET_SECONDS = 2.0
REF_EXEC_COMMANDS = ("cat", "grep", "head", "ls", "sed", "stat", "tail", "wc")
REF_EXEC_CLASSIFY_EXACT_ENCODE_MAX_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class ParsedRef:
    kind: RefKind
    value: str


@dataclass(frozen=True, slots=True)
class RefManifest:
    ref: str
    utf8_bytes: int | None
    sha256: str


@dataclass(frozen=True, slots=True)
class ZeroCopySourceHandle:
    text: str


@dataclass(frozen=True, slots=True)
class JsonlHistorySourceHandle:
    """Run-local history ref: one canonical JSON record per line."""

    records: tuple[str, ...]
    utf8_bytes: int
    sha256: str
    line_count: int

    def iter_records(self) -> Iterable[str]:
        return self.records


RefSourceHandle = ZeroCopySourceHandle | JsonlHistorySourceHandle


@dataclass(frozen=True, slots=True)
class RefCatalogEntry:
    manifest: RefManifest
    source: RefSourceHandle
    preview_text: str | None = None


@dataclass(frozen=True, slots=True)
class InvalidRefTextClassificationError(ValueError):
    def __str__(self) -> str:
        return "Eligible ref text classification requires complete measurement"


@dataclass(frozen=True, slots=True)
class RefTextClassification:
    eligible: bool
    utf8_bytes: int | None
    sha256: str | None
    line_count: int | None
    token_count: int | None
    encoder_failed: bool

    def __post_init__(self) -> None:
        if self.eligible and (
            self.utf8_bytes is None
            or self.sha256 is None
            or self.line_count is None
        ):
            raise InvalidRefTextClassificationError


@dataclass(frozen=True, slots=True)
class RefProjectionPlan:
    catalog: tuple[RefCatalogEntry, ...]
    manifests: tuple[RefManifest, ...]
    reader_schema: MappingProxyType | None = None


@dataclass(frozen=True, slots=True)
class RefProjectionError(RuntimeError):
    stage: str

    def __str__(self) -> str:
        return f"Externalized ref {self.stage} failed before provider forward"


@dataclass(frozen=True, slots=True)
class RefExecStage:
    command: str
    ref: str | None = None
    count: int | None = None
    flags: frozenset[str] = frozenset()
    pattern: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    list_kind: RefKind | None = None
    byte_count: int | None = None
    byte_start: int | None = None


@dataclass(frozen=True, slots=True)
class RefExecByteRange:
    requested_start: int
    requested_end: int | None
    actual_start: int
    actual_end: int | None
    marked: bool = False
    actual_empty: bool = False


@dataclass(frozen=True, slots=True)
class RefExecComponent:
    kind: Literal["display_prefix", "text", "synthetic_lf"]
    text: str
    start: int
    end: int
    utf8_bytes: int
    source_byte_start: int | None = None
    source_char_start: int | None = None


@dataclass(frozen=True, slots=True)
class RefExecComponentView:
    components: tuple[RefExecComponent, ...]


@dataclass(frozen=True, slots=True)
class RefExecLine:
    text: str
    number: int
    byte_start: int
    char_start: int
    has_newline: bool
    match_start: int | None = None
    match_end: int | None = None
    display_prefix: str = ""
    byte_range: RefExecByteRange | None = None
    metadata_only: bool = False
    atomic_match: bool = False
    component_view: RefExecComponentView | None = None


@dataclass(frozen=True, slots=True)
class RefExecError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message
def parse_ref(value: str) -> ParsedRef | None:
    tool_match = re.fullmatch(r"tool:([0-9a-f]{64})", value)
    if tool_match is not None:
        return ParsedRef(kind="tool", value=tool_match.group(1))
    history_match = re.fullmatch(r"history:([0-9a-f]{64})", value)
    if history_match is not None:
        return ParsedRef(kind="history", value=history_match.group(1))
    return None


def _split_ref_exec_pipeline(command: str) -> tuple[str, ...]:
    stages: list[str] = []
    buffer: list[str] = []
    single_quoted = False
    double_quoted = False
    escaped = False
    for character in command:
        if escaped:
            buffer.append(character)
            escaped = False
            continue
        if character == "\\" and not single_quoted:
            buffer.append(character)
            escaped = True
            continue
        if character == "'" and not double_quoted:
            single_quoted = not single_quoted
            buffer.append(character)
            continue
        if character == '"' and not single_quoted:
            double_quoted = not double_quoted
            buffer.append(character)
            continue
        if character == "|" and not single_quoted and not double_quoted:
            stage = "".join(buffer).strip()
            if not stage:
                raise RefExecError("Error: empty pipeline stage")
            stages.append(stage)
            buffer = []
            continue
        buffer.append(character)
    if escaped or single_quoted or double_quoted:
        raise RefExecError("Error: malformed quote or escape in command")
    stage = "".join(buffer).strip()
    if not stage:
        raise RefExecError("Error: empty command or pipeline stage")
    stages.append(stage)
    return tuple(stages)


def _parse_ref_exec_count(
    tokens: list[str],
    *,
    command: str,
) -> tuple[int | None, int | None, int | None, list[str]]:
    remaining = list(tokens)
    line_count: int | None = 10
    byte_count: int | None = None
    byte_start: int | None = None
    if remaining and remaining[0] == "-n":
        if len(remaining) < 2 or re.fullmatch(r"[0-9]+", remaining[1]) is None:
            raise RefExecError(
                f"Error: usage: {command} [-n N|-N|-c N{'|-c +N' if command == 'tail' else ''}] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        line_count = int(remaining[1])
        remaining = remaining[2:]
    elif remaining and remaining[0] == "-c":
        valid_fixed = len(remaining) >= 2 and re.fullmatch(
            r"[0-9]+", remaining[1]
        ) is not None
        valid_start = (
            command == "tail"
            and len(remaining) >= 2
            and re.fullmatch(r"\+[1-9][0-9]*", remaining[1]) is not None
        )
        if not valid_fixed and not valid_start:
            raise RefExecError(
                f"Error: usage: {command} [-n N|-N|-c N{'|-c +N' if command == 'tail' else ''}] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        line_count = None
        if valid_start:
            byte_start = int(remaining[1][1:])
        else:
            byte_count = int(remaining[1])
        remaining = remaining[2:]
    elif remaining and re.fullmatch(r"-[0-9]+", remaining[0]) is not None:
        line_count = int(remaining[0][1:])
        remaining = remaining[1:]
    return line_count, byte_count, byte_start, remaining


def _parse_ref_exec_grep(tokens: list[str], *, source: bool) -> RefExecStage:
    flags: set[str] = set()
    remaining = list(tokens)
    while remaining and remaining[0].startswith("-") and remaining[0] != "-":
        token = remaining.pop(0)
        if token == "--":
            break
        combined = token[1:]
        if not combined or any(flag not in "Einco" for flag in combined):
            raise RefExecError(
                "Error: usage: grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        flags.update(combined)
    expected = 2 if source else 1
    if len(remaining) != expected:
        raise RefExecError(
            "Error: usage: grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    ref = remaining[1] if source else None
    if ref is not None and parse_ref(ref) is None:
        raise RefExecError(
            "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    return RefExecStage(
        command="grep",
        ref=ref,
        flags=frozenset(flags),
        pattern=remaining[0],
    )


def _parse_ref_exec_sed(tokens: list[str], *, source: bool) -> RefExecStage:
    expected = 3 if source else 2
    if len(tokens) != expected or tokens[0] != "-n":
        raise RefExecError(
            "Error: usage: sed -n Np|M,Np|M,$p [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    selection = tokens[1]
    match = re.fullmatch(r"([1-9][0-9]*)(?:,([1-9][0-9]*|\$))?p", selection)
    if match is None:
        raise RefExecError(
            "Error: usage: sed -n Np|M,Np|M,$p [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    start = int(match.group(1))
    end_value = match.group(2)
    end = start if end_value is None else (None if end_value == "$" else int(end_value))
    if end is not None and start > end:
        raise RefExecError("Error: sed range start exceeds end")
    ref = tokens[2] if source else None
    if ref is not None and parse_ref(ref) is None:
        raise RefExecError(
            "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    return RefExecStage(command="sed", ref=ref, start_line=start, end_line=end)


def _parse_ref_exec_stage(stage: str, *, source: bool) -> RefExecStage:
    try:
        tokens = shlex.split(stage, posix=True)
    except ValueError as exc:
        raise RefExecError("Error: malformed quote in command") from exc
    if not tokens:
        raise RefExecError("Error: empty pipeline stage")
    command = tokens[0]
    arguments = tokens[1:]
    if command not in REF_EXEC_COMMANDS:
        available = ", ".join(REF_EXEC_COMMANDS)
        raise RefExecError(
            f"Error: unknown command. Available: {available}. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    if not source and command not in {"grep", "head", "sed", "tail", "wc"}:
        raise RefExecError("Error: command is not a valid piped consumer")
    if command == "ls":
        if len(arguments) > 1 or (arguments and arguments[0] not in {"history", "tool"}):
            raise RefExecError(
                "Error: usage: ls [history|tool]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        kind: RefKind | None = arguments[0] if arguments else None
        return RefExecStage(command=command, list_kind=kind)
    if command == "grep":
        return _parse_ref_exec_grep(arguments, source=source)
    if command == "sed":
        return _parse_ref_exec_sed(arguments, source=source)
    if command in {"head", "tail"}:
        count, byte_count, byte_start, remaining = _parse_ref_exec_count(
            arguments, command=command
        )
        expected = 1 if source else 0
        if len(remaining) != expected:
            raise RefExecError(
                f"Error: usage: {command} [-n N|-N|-c N{'|-c +N' if command == 'tail' else ''}] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        ref = remaining[0] if source else None
        if ref is not None and parse_ref(ref) is None:
            raise RefExecError(
                "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        return RefExecStage(
            command=command,
            ref=ref,
            count=count,
            byte_count=byte_count,
            byte_start=byte_start,
        )
    if command == "wc":
        if not arguments or arguments[0] not in {"-l", "-w", "-c"}:
            raise RefExecError(
                "Error: usage: wc [-l|-w|-c] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        expected = 2 if source else 1
        if len(arguments) != expected:
            raise RefExecError(
                "Error: usage: wc [-l|-w|-c] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        ref = arguments[1] if source else None
        if ref is not None and parse_ref(ref) is None:
            raise RefExecError(
                "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        return RefExecStage(command=command, ref=ref, flags=frozenset({arguments[0][1:]}))
    if len(arguments) != 1 or parse_ref(arguments[0]) is None:
        raise RefExecError(
            f"Error: usage: {command} REF. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    return RefExecStage(command=command, ref=arguments[0])


def _parse_ref_exec_command(command: str) -> tuple[RefExecStage, ...]:
    if not command.strip():
        raise RefExecError(REF_EXEC_USAGE_ERROR)
    try:
        encoded_command = command.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RefExecError(REF_EXEC_USAGE_ERROR) from exc
    if len(encoded_command) > REF_EXEC_COMMAND_MAX_BYTES:
        raise RefExecError("Error: command exceeds the 1,024 UTF-8 byte parser limit")
    return tuple(
        _parse_ref_exec_stage(stage, source=index == 0)
        for index, stage in enumerate(_split_ref_exec_pipeline(command))
    )


def _check_ref_exec_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise RefExecError("Error: reader command was cancelled")


def _encode_ref_text_checked(text: str, cancelled: threading.Event) -> bytes:
    _check_ref_exec_cancelled(cancelled)
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RefExecError(
            "Error: externalized ref source is not valid UTF-8"
        ) from exc


def _measure_ref_text_checked(text: str, cancelled: threading.Event) -> tuple[int, str]:
    digest = hashlib.sha256()
    utf8_bytes = 0
    for offset in range(0, len(text), REF_TEXT_HASH_CHUNK_CHARS):
        encoded = _encode_ref_text_checked(
            text[offset : offset + REF_TEXT_HASH_CHUNK_CHARS], cancelled
        )
        utf8_bytes += len(encoded)
        digest.update(encoded)
    return utf8_bytes, digest.hexdigest()


def _measure_ref_text_parts_checked(
    parts: tuple[str, ...],
    cancelled: threading.Event,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    utf8_bytes = 0
    for part in parts:
        for offset in range(0, len(part), REF_TEXT_HASH_CHUNK_CHARS):
            encoded = _encode_ref_text_checked(
                part[offset : offset + REF_TEXT_HASH_CHUNK_CHARS], cancelled
            )
            utf8_bytes += len(encoded)
            digest.update(encoded)
    return utf8_bytes, digest.hexdigest()


def _ref_exec_utf8_range_bytes(
    text: str,
    start: int,
    end: int,
    cancelled: threading.Event,
) -> int:
    utf8_bytes = 0
    for offset in range(start, end, REF_TEXT_HASH_CHUNK_CHARS):
        chunk_end = min(end, offset + REF_TEXT_HASH_CHUNK_CHARS)
        utf8_bytes += len(_encode_ref_text_checked(text[offset:chunk_end], cancelled))
    return utf8_bytes


def _ref_exec_utf8_prefix_index(
    text: str,
    max_bytes: int,
    cancelled: threading.Event,
) -> tuple[int, int]:
    offset = 0
    retained_bytes = 0
    while offset < len(text) and retained_bytes < max_bytes:
        chunk_end = min(len(text), offset + REF_TEXT_HASH_CHUNK_CHARS)
        chunk = text[offset:chunk_end]
        chunk_bytes = len(_encode_ref_text_checked(chunk, cancelled))
        if retained_bytes + chunk_bytes <= max_bytes:
            retained_bytes += chunk_bytes
            offset = chunk_end
            continue
        low = offset
        high = chunk_end
        while low < high:
            midpoint = (low + high + 1) // 2
            candidate_bytes = _ref_exec_utf8_range_bytes(text, offset, midpoint, cancelled)
            if retained_bytes + candidate_bytes <= max_bytes:
                low = midpoint
            else:
                high = midpoint - 1
        retained_bytes += _ref_exec_utf8_range_bytes(text, offset, low, cancelled)
        offset = low
        break
    return offset, retained_bytes


def _ref_exec_utf8_prefix(
    text: str,
    max_bytes: int,
    cancelled: threading.Event,
) -> str:
    offset, _ = _ref_exec_utf8_prefix_index(text, max_bytes, cancelled)
    return text[:offset]


def _iter_ref_text_lines(text: str, cancelled: threading.Event) -> Iterable[RefExecLine]:
    offset = 0
    line_number = 1
    byte_start = 0
    while offset < len(text):
        _check_ref_exec_cancelled(cancelled)
        newline = text.find("\n", offset)
        end = len(text) if newline < 0 else newline
        line = text[offset:end]
        yield RefExecLine(
            text=line,
            number=line_number,
            byte_start=byte_start,
            char_start=offset,
            has_newline=newline >= 0,
        )
        encoded_bytes, _ = _measure_ref_text_checked(line, cancelled)
        byte_start += encoded_bytes + (1 if newline >= 0 else 0)
        offset = end + (1 if newline >= 0 else 0)
        line_number += 1


def _measure_ref_source_checked(
    source: RefSourceHandle,
    cancelled: threading.Event,
) -> tuple[int, str]:
    if isinstance(source, ZeroCopySourceHandle):
        return _measure_ref_text_checked(source.text, cancelled)
    digest = hashlib.sha256()
    utf8_bytes = 0
    emitted = 0
    for record in source.iter_records():
        _check_ref_exec_cancelled(cancelled)
        if emitted > 0:
            digest.update(b"\n")
            utf8_bytes += 1
        for offset in range(0, len(record), REF_TEXT_HASH_CHUNK_CHARS):
            encoded = _encode_ref_text_checked(
                record[offset : offset + REF_TEXT_HASH_CHUNK_CHARS], cancelled
            )
            digest.update(encoded)
            utf8_bytes += len(encoded)
        emitted += 1
    if emitted != source.line_count:
        raise RefExecError("Error: externalized ref integrity verification failed")
    return utf8_bytes, digest.hexdigest()


def _iter_ref_source_lines(
    source: RefSourceHandle,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    if isinstance(source, ZeroCopySourceHandle):
        yield from _iter_ref_text_lines(source.text, cancelled)
        return
    byte_start = 0
    char_start = 0
    for line_number, record in enumerate(source.iter_records(), 1):
        _check_ref_exec_cancelled(cancelled)
        has_newline = line_number < source.line_count
        yield RefExecLine(
            text=record,
            number=line_number,
            byte_start=byte_start,
            char_start=char_start,
            has_newline=has_newline,
        )
        record_bytes, _ = _measure_ref_text_checked(record, cancelled)
        byte_start += record_bytes + int(has_newline)
        char_start += len(record) + int(has_newline)


def _iter_verified_ref_source_lines(
    entry: RefCatalogEntry,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    digest = hashlib.sha256()
    utf8_bytes = 0
    iterator = iter(_iter_ref_source_lines(entry.source, cancelled))

    def update(line: RefExecLine) -> None:
        nonlocal utf8_bytes
        for offset in range(0, len(line.text), REF_TEXT_HASH_CHUNK_CHARS):
            encoded = _encode_ref_text_checked(
                line.text[offset : offset + REF_TEXT_HASH_CHUNK_CHARS], cancelled
            )
            digest.update(encoded)
            utf8_bytes += len(encoded)
        if line.has_newline:
            digest.update(b"\n")
            utf8_bytes += 1

    try:
        current = next(iterator)
    except StopIteration:
        current = None
    if current is not None:
        for following in iterator:
            update(current)
            yield current
            current = following
        update(current)
    if utf8_bytes != entry.manifest.utf8_bytes or digest.hexdigest() != entry.manifest.sha256:
        raise RefExecError("Error: externalized ref integrity verification failed")
    if current is not None:
        yield current


def _ref_exec_head(lines: Iterable[RefExecLine], count: int, cancelled: threading.Event) -> Iterable[RefExecLine]:
    selected = 0
    iterator = iter(lines)
    while selected < count:
        try:
            line = next(iterator)
        except StopIteration:
            return
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            yield line
            continue
        yield line
        selected += 1


def _ref_exec_source_line_count(source: RefSourceHandle) -> int:
    if isinstance(source, ZeroCopySourceHandle):
        text = source.text
        return text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    return source.line_count


def _ref_exec_tail(lines: Iterable[RefExecLine], count: int, cancelled: threading.Event) -> Iterable[RefExecLine]:
    retained: deque[tuple[RefExecLine, int]] = deque()
    retained_bytes = 0
    seen = 0
    metadata: RefExecLine | None = None
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            metadata = line
            continue
        seen += 1
        line_bytes = _ref_exec_presented_line_bytes(line, cancelled)
        retained.append((line, line_bytes))
        retained_bytes += line_bytes
        while (
            len(retained) > count
            or retained_bytes > REF_EXEC_TAIL_MAX_BYTES
            or len(retained) > REF_EXEC_TAIL_MAX_LINES
        ):
            _dropped, dropped_bytes = retained.popleft()
            retained_bytes -= dropped_bytes
    if len(retained) < min(count, seen):
        raise RefExecError(
            "Error: tail line window exceeds 8 MiB or 65,536 lines; reduce -n or filter first"
        )
    for line, _ in retained:
        _check_ref_exec_cancelled(cancelled)
        yield line
    if metadata is not None and count > 0:
        yield metadata


def _ref_exec_component(
    kind: Literal["display_prefix", "text", "synthetic_lf"],
    text: str,
    cancelled: threading.Event,
    *,
    source_start: tuple[int, int] | None = None,
) -> RefExecComponent:
    utf8_bytes, _ = _measure_ref_text_checked(text, cancelled)
    return RefExecComponent(
        kind=kind,
        text=text,
        start=0,
        end=len(text),
        utf8_bytes=utf8_bytes,
        source_byte_start=source_start[0] if source_start is not None else None,
        source_char_start=source_start[1] if source_start is not None else None,
    )


def _ref_exec_line_component_view(
    line: RefExecLine,
    cancelled: threading.Event,
) -> RefExecComponentView:
    if line.component_view is not None:
        return line.component_view
    components: list[RefExecComponent] = []
    if line.display_prefix:
        components.append(
            _ref_exec_component("display_prefix", line.display_prefix, cancelled)
        )
    text_component = _ref_exec_component(
        "text",
        line.text,
        cancelled,
        source_start=(line.byte_start, line.char_start),
    )
    if line.text:
        components.append(text_component)
    if line.has_newline:
        components.append(
            _ref_exec_component(
                "synthetic_lf",
                "\n",
                cancelled,
                source_start=(
                    line.byte_start + text_component.utf8_bytes,
                    line.char_start + len(line.text),
                ),
            )
        )
    return RefExecComponentView(tuple(components))


def _ref_exec_materialize_components(
    components: Iterable[RefExecComponent],
    *,
    include_synthetic_lf: bool = True,
) -> str:
    return "".join(
        component.text[component.start : component.end]
        for component in components
        if include_synthetic_lf or component.kind != "synthetic_lf"
    )


def _ref_exec_materialize_line(
    line: RefExecLine,
    cancelled: threading.Event,
    *,
    include_synthetic_lf: bool = True,
) -> str:
    if line.component_view is None:
        return line.display_prefix + line.text + (
            "\n" if include_synthetic_lf and line.has_newline else ""
        )
    _check_ref_exec_cancelled(cancelled)
    return _ref_exec_materialize_components(
        line.component_view.components,
        include_synthetic_lf=include_synthetic_lf,
    )


def _ref_exec_component_prefix_index(
    component: RefExecComponent,
    max_bytes: int,
    cancelled: threading.Event,
) -> tuple[int, int]:
    offset = component.start
    retained_bytes = 0
    while offset < component.end and retained_bytes < max_bytes:
        chunk_end = min(component.end, offset + REF_TEXT_HASH_CHUNK_CHARS)
        chunk_bytes = _ref_exec_utf8_range_bytes(
            component.text, offset, chunk_end, cancelled
        )
        if retained_bytes + chunk_bytes <= max_bytes:
            retained_bytes += chunk_bytes
            offset = chunk_end
            continue
        low = offset
        high = chunk_end
        while low < high:
            midpoint = (low + high + 1) // 2
            candidate_bytes = _ref_exec_utf8_range_bytes(
                component.text, offset, midpoint, cancelled
            )
            if retained_bytes + candidate_bytes <= max_bytes:
                low = midpoint
            else:
                high = midpoint - 1
        retained_bytes += _ref_exec_utf8_range_bytes(
            component.text, offset, low, cancelled
        )
        offset = low
        break
    return offset, retained_bytes


def _ref_exec_presented_line_bytes(
    line: RefExecLine,
    cancelled: threading.Event,
) -> int:
    if line.component_view is not None:
        return sum(component.utf8_bytes for component in line.component_view.components)
    encoded_bytes, _ = _measure_ref_text_parts_checked(
        (line.display_prefix, line.text), cancelled
    )
    return encoded_bytes + int(line.has_newline)


def _ref_exec_source_byte_bounds(
    line: RefExecLine,
) -> tuple[int, int] | None:
    if line.component_view is None:
        return None
    backed = tuple(
        component
        for component in line.component_view.components
        if component.source_byte_start is not None
    )
    if not backed:
        return None
    return (
        backed[0].source_byte_start + 1,
        backed[-1].source_byte_start + backed[-1].utf8_bytes,
    )


def _ref_exec_slice_presented_line(
    line: RefExecLine,
    *,
    stream_start: int,
    selected_start: int,
    selected_end: int,
    cancelled: threading.Event,
) -> tuple[RefExecLine | None, int, int, bool]:
    actual_start: int | None = None
    actual_end = stream_start
    snapped = False
    component_start = stream_start
    selected_components: list[RefExecComponent] = []
    for component in _ref_exec_line_component_view(line, cancelled).components:
        component_end = component_start + component.utf8_bytes
        overlap_start = max(selected_start, component_start)
        overlap_end = min(selected_end, component_end)
        if overlap_start < overlap_end:
            local_start = overlap_start - component_start
            local_end = overlap_end - component_start
            start_index, start_floor = _ref_exec_component_prefix_index(
                component, local_start, cancelled
            )
            if start_floor < local_start:
                start_index += 1
                snapped = True
            actual_local_start = _ref_exec_utf8_range_bytes(
                component.text, component.start, start_index, cancelled
            )
            end_index, actual_local_end = _ref_exec_component_prefix_index(
                component, local_end, cancelled
            )
            if actual_local_end < local_end:
                snapped = True
            if start_index < end_index:
                part_start = component_start + actual_local_start
                if actual_start is None:
                    actual_start = part_start
                actual_end = component_start + actual_local_end
                selected_components.append(
                    replace(
                        component,
                        start=start_index,
                        end=end_index,
                        utf8_bytes=actual_local_end - actual_local_start,
                        source_byte_start=(
                            component.source_byte_start + actual_local_start
                            if component.source_byte_start is not None
                            else None
                        ),
                        source_char_start=(
                            component.source_char_start
                            + start_index
                            - component.start
                            if component.source_char_start is not None
                            else None
                        ),
                    )
                )
        component_start = component_end
    if actual_start is None:
        if not snapped:
            return None, selected_end, selected_end, False
        return (
            replace(
                line,
                text="",
                byte_start=selected_end,
                has_newline=False,
                match_start=None,
                match_end=None,
                display_prefix="",
                byte_range=None,
                metadata_only=True,
                component_view=RefExecComponentView(()),
            ),
            selected_end,
            selected_end,
            True,
        )
    return (
        replace(
            line,
            text="",
            byte_start=next(
                (
                    component.source_byte_start
                    for component in selected_components
                    if component.kind == "text"
                    and component.source_byte_start is not None
                ),
                line.byte_start,
            ),
            char_start=next(
                (
                    component.source_char_start
                    for component in selected_components
                    if component.kind == "text"
                    and component.source_char_start is not None
                ),
                line.char_start,
            ),
            has_newline=False,
            match_start=None,
            match_end=None,
            display_prefix="",
            byte_range=None,
            component_view=RefExecComponentView(tuple(selected_components)),
        ),
        actual_start,
        actual_end,
        snapped,
    )


def _ref_exec_head_bytes(
    lines: Iterable[RefExecLine],
    count: int,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    if count == 0:
        return
    stream_start = 0
    pending: RefExecLine | None = None
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        line_end = stream_start + _ref_exec_presented_line_bytes(line, cancelled)
        selected, actual_start, actual_end, snapped = _ref_exec_slice_presented_line(
            line,
            stream_start=stream_start,
            selected_start=0,
            selected_end=min(count, line_end),
            cancelled=cancelled,
        )
        if selected is not None:
            source_bounds = (
                _ref_exec_source_byte_bounds(selected)
                if line.component_view is not None
                else None
            )
            range_start = (
                source_bounds[0] if source_bounds is not None else actual_start + 1
            )
            range_end = source_bounds[1] if source_bounds is not None else actual_end
            ranged = replace(
                selected,
                atomic_match=False,
                byte_range=RefExecByteRange(
                    requested_start=1,
                    requested_end=count,
                    actual_start=range_start,
                    actual_end=range_end,
                    marked=snapped or source_bounds is not None,
                    actual_empty=snapped and actual_start == actual_end,
                ),
            )
            if (
                ranged.metadata_only
                and pending is not None
                and pending.byte_range is not None
            ):
                pending = replace(
                    pending,
                    byte_range=replace(
                        pending.byte_range,
                        marked=pending.byte_range.marked or snapped,
                    ),
                )
            else:
                if pending is not None:
                    yield pending
                pending = ranged
        if line_end >= count:
            if pending is not None:
                yield pending
            return
        stream_start = line_end
    if pending is not None:
        yield pending


def _ref_exec_tail_from_bytes(
    lines: Iterable[RefExecLine],
    start: int,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    selected_start = start - 1
    stream_start = 0
    actual_range_start: int | None = None
    snapped_start = False
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            continue
        line_end = stream_start + _ref_exec_presented_line_bytes(line, cancelled)
        if line_end <= selected_start:
            stream_start = line_end
            continue
        selected, actual_start, actual_end, snapped = _ref_exec_slice_presented_line(
            line,
            stream_start=stream_start,
            selected_start=selected_start,
            selected_end=line_end,
            cancelled=cancelled,
        )
        if selected is not None:
            if actual_range_start is None:
                actual_range_start = actual_start + 1
                snapped_start = snapped
            yield replace(
                selected,
                atomic_match=False,
                byte_range=RefExecByteRange(
                    requested_start=start,
                    requested_end=None,
                    actual_start=actual_range_start,
                    actual_end=None,
                    marked=snapped_start,
                    actual_empty=snapped and actual_start == actual_end,
                ),
            )
        stream_start = line_end


@dataclass(frozen=True, slots=True)
class RefExecTailSnapshot:
    retained: bytearray
    write_offset: int
    component_lengths: array
    component_flags: bytearray
    component_source_delta_indexes: array
    component_source_deltas: array
    component_char_delta_indexes: array
    component_char_deltas: array
    record_component_ends: array
    record_flags: bytearray
    record_char_delta_indexes: array
    record_char_deltas: array
    record_number_delta_indexes: array
    record_number_deltas: array
    record_match_starts: array | None
    record_match_ends: array | None
    trailing_plain_default: bool
    trailing_plain_change_indexes: array
    trailing_number_delta_indexes: array
    trailing_number_deltas: array
    trailing_char_delta_indexes: array
    trailing_char_deltas: array
    trailing_match_indexes: array
    trailing_match_starts: array
    trailing_match_ends: array
    compact_missing: int
    retained_bytes: int
    snapped_bytes: int
    stream_start: int
    stream_chars: int
    line_count: int
    trailing_empty_lines: int
    snapped_empty: bool
    snapped_number: int
    snapped_char_start: int
    partial_fallback_char_start: int | None
    count: int


def _iter_ref_exec_tail_snapshot(
    snapshot: RefExecTailSnapshot,
) -> Iterable[RefExecLine]:
    effective_bytes = snapshot.retained_bytes - snapshot.snapped_bytes
    oldest_offset = snapshot.write_offset if snapshot.retained_bytes == snapshot.count else 0
    payload_start = (oldest_offset + snapshot.snapped_bytes) % snapshot.count
    first_bytes = min(effective_bytes, snapshot.count - payload_start)
    decoder = codecs.getincrementaldecoder("utf-8")()
    presented = decoder.decode(
        memoryview(snapshot.retained)[payload_start : payload_start + first_bytes],
        final=first_bytes == effective_bytes,
    )
    if first_bytes < effective_bytes:
        presented += decoder.decode(
            memoryview(snapshot.retained)[: effective_bytes - first_bytes],
            final=True,
        )
    requested_start = max(1, snapshot.stream_start - snapshot.count + 1)
    actual_start = (
        snapshot.stream_start
        - snapshot.retained_bytes
        + snapshot.snapped_bytes
        + 1
    )
    byte_range = RefExecByteRange(
        requested_start=requested_start,
        requested_end=snapshot.stream_start,
        actual_start=actual_start,
        actual_end=snapshot.stream_start,
        marked=actual_start != requested_start,
        actual_empty=actual_start > snapshot.stream_start,
    )
    if not presented and (snapshot.snapped_bytes or snapshot.snapped_empty):
        yield RefExecLine(
            text="",
            number=snapshot.snapped_number,
            byte_start=snapshot.stream_start,
            char_start=snapshot.snapped_char_start,
            has_newline=False,
            byte_range=byte_range,
            metadata_only=True,
            component_view=RefExecComponentView(()),
        )
        if snapshot.trailing_empty_lines == 0:
            return
    output_line_count = (
        len(snapshot.record_component_ends) + snapshot.trailing_empty_lines
    )
    output_number = snapshot.line_count - output_line_count + 1
    presented_char_start = snapshot.stream_chars - len(presented)
    line_byte_start = actual_start - 1
    byte_offset = 0
    char_offset = 0
    component_offset = 0
    source_delta_offset = 0
    char_delta_offset = 0
    number_delta_offset = 0
    record_char_delta_offset = 0
    for record_offset, component_end in enumerate(snapshot.record_component_ends):
        record_flags = snapshot.record_flags[record_offset]
        record_start = bool(record_flags & 0x01)
        record_has_newline = bool(record_flags & 0x02)
        record_plain = bool(record_flags & 0x04)
        record_char_delta = 0
        if (
            record_char_delta_offset < len(snapshot.record_char_delta_indexes)
            and snapshot.record_char_delta_indexes[record_char_delta_offset]
            == record_offset
        ):
            record_char_delta = snapshot.record_char_deltas[
                record_char_delta_offset
            ]
            record_char_delta_offset += 1
        fallback_char_start = presented_char_start + char_offset + record_char_delta
        components: list[RefExecComponent] = []
        while component_offset < component_end:
            component_length = snapshot.component_lengths[component_offset]
            flags = snapshot.component_flags[component_offset]
            kind = flags & 0x03
            component_byte_end = byte_offset + component_length
            component_ring_start = (payload_start + byte_offset) % snapshot.count
            component_first_bytes = min(
                component_length, snapshot.count - component_ring_start
            )
            component_decoder = codecs.getincrementaldecoder("utf-8")()
            component_text = component_decoder.decode(
                memoryview(snapshot.retained)[
                    component_ring_start : component_ring_start
                    + component_first_bytes
                ],
                final=component_first_bytes == component_length,
            )
            if component_first_bytes < component_length:
                component_text += component_decoder.decode(
                    memoryview(snapshot.retained)[
                        : component_length - component_first_bytes
                    ],
                    final=True,
                )
            component_char_end = char_offset + len(component_text)
            source_delta = 0
            if flags & 0x08:
                source_delta = record_char_delta
            elif (
                source_delta_offset < len(snapshot.component_source_delta_indexes)
                and snapshot.component_source_delta_indexes[source_delta_offset]
                == component_offset
            ):
                source_delta = snapshot.component_source_deltas[source_delta_offset]
                source_delta_offset += 1
            source_backed = bool(flags & 0x04)
            source_char_delta = source_delta
            if (
                char_delta_offset < len(snapshot.component_char_delta_indexes)
                and snapshot.component_char_delta_indexes[char_delta_offset]
                == component_offset
            ):
                source_char_delta = snapshot.component_char_deltas[char_delta_offset]
                char_delta_offset += 1
            components.append(
                RefExecComponent(
                    kind=(
                        "display_prefix"
                        if kind == 0
                        else "text" if kind == 1 else "synthetic_lf"
                    ),
                    text=presented,
                    start=char_offset,
                    end=component_char_end,
                    utf8_bytes=component_length,
                    source_byte_start=(
                        actual_start - 1 + byte_offset + source_delta
                        if source_backed
                        else None
                    ),
                    source_char_start=(
                        presented_char_start + char_offset + source_char_delta
                        if source_backed
                        else None
                    ),
                )
            )
            char_offset = component_char_end
            byte_offset = component_byte_end
            component_offset += 1
        number_delta = 0
        if (
            number_delta_offset < len(snapshot.record_number_delta_indexes)
            and snapshot.record_number_delta_indexes[number_delta_offset]
            == record_offset
        ):
            number_delta = snapshot.record_number_deltas[number_delta_offset]
            number_delta_offset += 1
        history_fallback_char_start = fallback_char_start
        if (
            record_offset == 0
            and not record_start
            and snapshot.partial_fallback_char_start is not None
        ):
            history_fallback_char_start = snapshot.partial_fallback_char_start
        reconstructed_char_start = (
            history_fallback_char_start
            if record_start
            else next(
                (
                    component.source_char_start
                    for component in components
                    if component.kind == "text"
                    and component.source_char_start is not None
                ),
                history_fallback_char_start,
            )
        )
        match_start = (
            snapshot.record_match_starts[record_offset]
            if snapshot.record_match_starts is not None
            else snapshot.compact_missing
        )
        match_end = (
            snapshot.record_match_ends[record_offset]
            if snapshot.record_match_ends is not None
            else snapshot.compact_missing
        )
        plain_text = "".join(
            component.text[component.start : component.end]
            for component in components
            if component.kind == "text"
        )
        plain_display_prefix = "".join(
            component.text[component.start : component.end]
            for component in components
            if component.kind == "display_prefix"
        )
        yield RefExecLine(
            text=plain_text if record_plain else "",
            number=output_number + number_delta,
            byte_start=line_byte_start,
            char_start=reconstructed_char_start,
            has_newline=record_start
            and record_has_newline
            and bool(components)
            and components[-1].kind == "synthetic_lf",
            match_start=(
                match_start if match_start != snapshot.compact_missing else None
            ),
            match_end=match_end if match_end != snapshot.compact_missing else None,
            display_prefix=plain_display_prefix if record_plain else "",
            byte_range=byte_range,
            component_view=(
                None if record_plain else RefExecComponentView(tuple(components))
            ),
        )
        output_number += 1
        line_byte_start = actual_start - 1 + byte_offset
    trailing_plain = snapshot.trailing_plain_default
    trailing_plain_change_offset = 0
    trailing_number_delta_offset = 0
    trailing_char_delta_offset = 0
    trailing_match_offset = 0
    for trailing_offset in range(snapshot.trailing_empty_lines):
        if (
            trailing_plain_change_offset
            < len(snapshot.trailing_plain_change_indexes)
            and snapshot.trailing_plain_change_indexes[
                trailing_plain_change_offset
            ]
            == trailing_offset
        ):
            trailing_plain = not trailing_plain
            trailing_plain_change_offset += 1
        number_delta = 0
        if (
            trailing_number_delta_offset
            < len(snapshot.trailing_number_delta_indexes)
            and snapshot.trailing_number_delta_indexes[
                trailing_number_delta_offset
            ]
            == trailing_offset
        ):
            number_delta = snapshot.trailing_number_deltas[
                trailing_number_delta_offset
            ]
            trailing_number_delta_offset += 1
        char_delta = 0
        if (
            trailing_char_delta_offset < len(snapshot.trailing_char_delta_indexes)
            and snapshot.trailing_char_delta_indexes[trailing_char_delta_offset]
            == trailing_offset
        ):
            char_delta = snapshot.trailing_char_deltas[trailing_char_delta_offset]
            trailing_char_delta_offset += 1
        match_start = snapshot.compact_missing
        match_end = snapshot.compact_missing
        if (
            trailing_match_offset < len(snapshot.trailing_match_indexes)
            and snapshot.trailing_match_indexes[trailing_match_offset]
            == trailing_offset
        ):
            match_start = snapshot.trailing_match_starts[trailing_match_offset]
            match_end = snapshot.trailing_match_ends[trailing_match_offset]
            trailing_match_offset += 1
        yield RefExecLine(
            text="",
            number=output_number + number_delta,
            byte_start=snapshot.stream_start,
            char_start=snapshot.stream_chars + char_delta,
            has_newline=False,
            match_start=(
                match_start if match_start != snapshot.compact_missing else None
            ),
            match_end=(
                match_end if match_end != snapshot.compact_missing else None
            ),
            byte_range=byte_range,
            component_view=(None if trailing_plain else RefExecComponentView(())),
        )
        output_number += 1


# allow: SIZE_OK - one-pass circular-buffer state keeps overwrite ordering local.
def _ref_exec_tail_bytes(
    lines: Iterable[RefExecLine],
    count: int,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    if count > REF_EXEC_TAIL_MAX_BYTES:
        raise RefExecError(
            "Error: tail rolling window exceeds 8 MiB; reduce -c or filter first"
    )
    if count == 0:
        return
    retained = bytearray(count)
    retained_bytes = 0
    write_offset = 0
    component_lengths = array("I")
    component_flags = bytearray()
    component_source_delta_indexes = array("I")
    component_source_deltas = array("i")
    component_char_delta_indexes = array("I")
    component_char_deltas = array("i")
    component_base = 0
    record_component_ends = array("I")
    record_flags = bytearray()
    record_char_delta_indexes = array("I")
    record_char_deltas = array("i")
    record_number_delta_indexes = array("I")
    record_number_deltas = array("i")
    record_match_starts: array | None = None
    record_match_ends: array | None = None
    record_base = 0
    trailing_plain_default = False
    trailing_plain_last = False
    trailing_plain_change_indexes = array("Q")
    trailing_number_delta_indexes = array("Q")
    trailing_number_deltas = array("i")
    trailing_char_delta_indexes = array("Q")
    trailing_char_deltas = array("i")
    trailing_match_indexes = array("Q")
    trailing_match_starts = array("i")
    trailing_match_ends = array("i")
    compact_missing = -1
    stream_start = 0
    stream_chars = 0
    retained_char_start = 0
    partial_fallback_char_start: int | None = None
    line_count = 0
    trailing_empty_lines = 0
    snapped_empty = False
    snapped_number = 0
    snapped_char_start = 0

    def append_compact(lane: array, value: int) -> array:
        if lane.typecode == "i" and not -(1 << 31) <= value < 1 << 31:
            lane = array("q", lane)
        lane.append(value)
        return lane

    def drop_metadata(dropped_bytes: int, dropped_offset: int) -> None:
        nonlocal component_base, record_base, partial_fallback_char_start
        previous_component_base = component_base
        remaining = dropped_bytes
        clipped_component = False
        while remaining and component_base < len(component_lengths):
            component_length = component_lengths[component_base]
            if remaining >= component_length:
                remaining -= component_length
                component_base += 1
                continue
            component_lengths[component_base] = component_length - remaining
            remaining = 0
            clipped_component = True
        while (
            record_base < len(record_component_ends)
            and record_component_ends[record_base] <= component_base
        ):
            record_base += 1
        if record_base >= len(record_component_ends):
            return
        record_component_start = (
            record_component_ends[record_base - 1] if record_base else 0
        )
        if record_component_start < component_base or clipped_component:
            if record_flags[record_base] & 0x01:
                record_boundary_bytes = sum(
                    component_lengths[index]
                    for index in range(
                        previous_component_base, record_component_start
                    )
                )
                record_boundary_chars = sum(
                    retained[(dropped_offset + offset) % count] & 0xC0 != 0x80
                    for offset in range(
                        min(record_boundary_bytes, dropped_bytes)
                    )
                )
                char_delta_position = bisect_left(
                    record_char_delta_indexes, record_base
                )
                record_char_delta = (
                    record_char_deltas[char_delta_position]
                    if char_delta_position < len(record_char_delta_indexes)
                    and record_char_delta_indexes[char_delta_position] == record_base
                    else 0
                )
                partial_fallback_char_start = (
                    retained_char_start
                    + record_boundary_chars
                    + record_char_delta
                )
            record_flags[record_base] &= 0x02
            if record_match_starts is not None and record_match_ends is not None:
                record_match_starts[record_base] = compact_missing
                record_match_ends[record_base] = compact_missing

    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            continue
        line_count += 1
        original_view = _ref_exec_line_component_view(line, cancelled)
        line_bytes = sum(component.utf8_bytes for component in original_view.components)
        line_end = stream_start + line_bytes
        line_chars = sum(
            component.end - component.start for component in original_view.components
        )
        selected = line
        selected_start = stream_start
        selected_char_start = stream_chars
        selected_is_snapped_empty = False
        if line_bytes > count:
            selected, selected_start, _, _ = _ref_exec_slice_presented_line(
                line,
                stream_start=stream_start,
                selected_start=line_end - count,
                selected_end=line_end,
                cancelled=cancelled,
            )
            if selected is None:
                stream_start = line_end
                stream_chars += line_chars
                trailing_empty_lines += 1
                continue
            retained_bytes = 0
            write_offset = 0
            component_lengths = array("I")
            component_flags = bytearray()
            component_source_delta_indexes = array("I")
            component_source_deltas = array("i")
            component_char_delta_indexes = array("I")
            component_char_deltas = array("i")
            component_base = 0
            record_component_ends = array("I")
            record_flags = bytearray()
            record_char_delta_indexes = array("I")
            record_char_deltas = array("i")
            record_number_delta_indexes = array("I")
            record_number_deltas = array("i")
            record_match_starts = None
            record_match_ends = None
            record_base = 0
            trailing_empty_lines = 0
            trailing_plain_default = False
            trailing_plain_last = False
            trailing_plain_change_indexes = array("Q")
            trailing_number_delta_indexes = array("Q")
            trailing_number_deltas = array("i")
            trailing_char_delta_indexes = array("Q")
            trailing_char_deltas = array("i")
            trailing_match_indexes = array("Q")
            trailing_match_starts = array("i")
            trailing_match_ends = array("i")
            partial_fallback_char_start = selected.char_start
            snapped_empty = selected.metadata_only
            selected_is_snapped_empty = snapped_empty
            if snapped_empty:
                snapped_number = selected.number
                snapped_char_start = selected.char_start
            selected_chars = sum(
                component.end - component.start
                for component in _ref_exec_line_component_view(
                    selected, cancelled
                ).components
            )
            selected_char_start = stream_chars + line_chars - selected_chars
            retained_char_start = selected_char_start
        selected_view = _ref_exec_line_component_view(selected, cancelled)
        if line_bytes and trailing_empty_lines:
            trailing_empty_lines = 0
            trailing_plain_default = False
            trailing_plain_last = False
            trailing_plain_change_indexes = array("Q")
            trailing_number_delta_indexes = array("Q")
            trailing_number_deltas = array("i")
            trailing_char_delta_indexes = array("Q")
            trailing_char_deltas = array("i")
            trailing_match_indexes = array("Q")
            trailing_match_starts = array("i")
            trailing_match_ends = array("i")
        record_offset = len(record_component_ends)
        record_start = selected_start == stream_start
        record_char_delta = selected.char_start - selected_char_start
        component_stream_start = selected_start
        component_char_start = selected_char_start
        for component in selected_view.components:
            kind = {
                "display_prefix": 0,
                "text": 1,
                "synthetic_lf": 2,
            }[component.kind]
            source_byte_delta = (
                component.source_byte_start - component_stream_start
                if component.source_byte_start is not None
                else 0
            )
            source_char_delta = (
                component.source_char_start - component_char_start
                if component.source_char_start is not None
                else 0
            )
            component_length = 0
            for offset in range(
                component.start, component.end, REF_TEXT_HASH_CHUNK_CHARS
            ):
                chunk_end = min(
                    component.end, offset + REF_TEXT_HASH_CHUNK_CHARS
                )
                encoded = _encode_ref_text_checked(
                    component.text[offset:chunk_end], cancelled
                )
                component_length += len(encoded)
                encoded_offset = 0
                while encoded_offset < len(encoded):
                    available = min(
                        len(encoded) - encoded_offset, count - write_offset
                    )
                    write_end = write_offset + available
                    if retained_bytes == count:
                        drop_metadata(available, write_offset)
                        for retained_offset in range(write_offset, write_end):
                            if retained[retained_offset] & 0xC0 != 0x80:
                                retained_char_start += 1
                    retained[write_offset:write_end] = encoded[
                        encoded_offset : encoded_offset + available
                    ]
                    write_offset = write_end % count
                    encoded_offset += available
                    retained_bytes = min(count, retained_bytes + available)
            if component_length:
                component_lengths.append(component_length)
                flags = kind
                if component.source_byte_start is not None:
                    flags |= 0x04
                    if (
                        source_byte_delta == record_char_delta
                        and source_char_delta == record_char_delta
                    ):
                        flags |= 0x08
                    elif source_byte_delta:
                        component_source_delta_indexes.append(
                            len(component_lengths) - 1
                        )
                        component_source_deltas = append_compact(
                            component_source_deltas, source_byte_delta
                        )
                component_flags.append(flags)
                if source_char_delta != source_byte_delta:
                    component_char_delta_indexes.append(len(component_lengths) - 1)
                    component_char_deltas = append_compact(
                        component_char_deltas, source_char_delta
                    )
            component_stream_start += component.utf8_bytes
            component_char_start += component.end - component.start
        if line_bytes and not selected_is_snapped_empty:
            record_component_ends.append(len(component_lengths))
            flags = int(record_start)
            if record_start and selected.has_newline:
                flags |= 0x02
            if record_start and selected.component_view is None:
                flags |= 0x04
            record_flags.append(flags)
            if record_char_delta:
                record_char_delta_indexes.append(record_offset)
                record_char_deltas = append_compact(
                    record_char_deltas, record_char_delta
                )
            number_delta = selected.number - line_count
            if number_delta:
                record_number_delta_indexes.append(record_offset)
                record_number_deltas = append_compact(
                    record_number_deltas, number_delta
                )
            if selected.match_start is not None or selected.match_end is not None:
                if record_match_starts is None or record_match_ends is None:
                    record_match_starts = array("i", (compact_missing,)) * record_offset
                    record_match_ends = array("i", (compact_missing,)) * record_offset
                record_match_starts = append_compact(
                    record_match_starts,
                    selected.match_start
                    if selected.match_start is not None
                    else compact_missing,
                )
                record_match_ends = append_compact(
                    record_match_ends,
                    selected.match_end
                    if selected.match_end is not None
                    else compact_missing,
                )
            elif record_match_starts is not None and record_match_ends is not None:
                record_match_starts.append(compact_missing)
                record_match_ends.append(compact_missing)
        if retained_bytes:
            physical_oldest_offset = write_offset if retained_bytes == count else 0
            oldest_offset = physical_oldest_offset
            for _ in range(min(4, retained_bytes)):
                if retained[oldest_offset] & 0xC0 != 0x80:
                    break
                oldest_offset = (oldest_offset + 1) % count
            skipped_bytes = (oldest_offset - physical_oldest_offset) % count
            logical_component = component_base
            logical_component_skip = skipped_bytes
            while (
                logical_component < len(component_lengths)
                and logical_component_skip
                >= component_lengths[logical_component]
            ):
                logical_component_skip -= component_lengths[logical_component]
                logical_component += 1
            logical_record = bisect_right(
                record_component_ends,
                logical_component,
                lo=record_base,
            )
            logical_record_start = (
                record_component_ends[logical_record - 1]
                if logical_record
                else 0
            )
            logical_record_intact = (
                logical_record < len(record_flags)
                and bool(record_flags[logical_record] & 0x01)
                and logical_component == logical_record_start
                and logical_component_skip == 0
            )
            if logical_record_intact:
                partial_fallback_char_start = None
            elif (
                logical_record < len(record_component_ends)
                and logical_component < len(component_flags)
            ):
                oldest_flags = component_flags[logical_component]
                if oldest_flags & 0x03 == 1 and oldest_flags & 0x04:
                    char_delta_position = bisect_left(
                        record_char_delta_indexes, logical_record
                    )
                    source_char_delta = 0
                    if (
                        oldest_flags & 0x08
                        and char_delta_position < len(record_char_delta_indexes)
                        and record_char_delta_indexes[char_delta_position]
                        == logical_record
                    ):
                        source_char_delta = record_char_deltas[
                            char_delta_position
                        ]
                    source_delta_position = bisect_left(
                        component_source_delta_indexes, logical_component
                    )
                    if (
                        source_delta_position
                        < len(component_source_delta_indexes)
                        and component_source_delta_indexes[source_delta_position]
                        == logical_component
                    ):
                        source_char_delta = component_source_deltas[
                            source_delta_position
                        ]
                    char_delta_position = bisect_left(
                        component_char_delta_indexes, logical_component
                    )
                    if (
                        char_delta_position < len(component_char_delta_indexes)
                        and component_char_delta_indexes[char_delta_position]
                        == logical_component
                    ):
                        source_char_delta = component_char_deltas[
                            char_delta_position
                        ]
                    partial_fallback_char_start = (
                        retained_char_start + source_char_delta
                    )
        if line_bytes == 0:
            trailing_plain = selected.component_view is None
            if trailing_empty_lines == 0:
                trailing_plain_default = trailing_plain
            elif trailing_plain != trailing_plain_last:
                trailing_plain_change_indexes.append(trailing_empty_lines)
            trailing_plain_last = trailing_plain
            trailing_number_delta = selected.number - line_count
            if trailing_number_delta:
                trailing_number_delta_indexes.append(trailing_empty_lines)
                trailing_number_deltas = append_compact(
                    trailing_number_deltas, trailing_number_delta
                )
            trailing_char_delta = selected.char_start - stream_chars
            if trailing_char_delta:
                trailing_char_delta_indexes.append(trailing_empty_lines)
                trailing_char_deltas = append_compact(
                    trailing_char_deltas, trailing_char_delta
                )
            if selected.match_start is not None or selected.match_end is not None:
                trailing_match_indexes.append(trailing_empty_lines)
                trailing_match_starts = append_compact(
                    trailing_match_starts,
                    selected.match_start
                    if selected.match_start is not None
                    else compact_missing,
                )
                trailing_match_ends = append_compact(
                    trailing_match_ends,
                    selected.match_end
                    if selected.match_end is not None
                    else compact_missing,
                )
            trailing_empty_lines += 1
        stream_start = line_end
        stream_chars += line_chars
        if record_base >= 65_536:
            del record_component_ends[:record_base]
            del record_flags[:record_base]
            char_record_prefix = bisect_left(
                record_char_delta_indexes, record_base
            )
            del record_char_delta_indexes[:char_record_prefix]
            del record_char_deltas[:char_record_prefix]
            for index in range(len(record_char_delta_indexes)):
                record_char_delta_indexes[index] -= record_base
            if record_match_starts is not None and record_match_ends is not None:
                del record_match_starts[:record_base]
                del record_match_ends[:record_base]
            number_prefix = 0
            while (
                number_prefix < len(record_number_delta_indexes)
                and record_number_delta_indexes[number_prefix] < record_base
            ):
                number_prefix += 1
            del record_number_delta_indexes[:number_prefix]
            del record_number_deltas[:number_prefix]
            for index in range(len(record_number_delta_indexes)):
                record_number_delta_indexes[index] -= record_base
            record_base = 0
        if component_base >= 131_072:
            if record_base:
                del record_component_ends[:record_base]
                del record_flags[:record_base]
                char_record_prefix = bisect_left(
                    record_char_delta_indexes, record_base
                )
                del record_char_delta_indexes[:char_record_prefix]
                del record_char_deltas[:char_record_prefix]
                for index in range(len(record_char_delta_indexes)):
                    record_char_delta_indexes[index] -= record_base
                if (
                    record_match_starts is not None
                    and record_match_ends is not None
                ):
                    del record_match_starts[:record_base]
                    del record_match_ends[:record_base]
                number_prefix = bisect_left(
                    record_number_delta_indexes, record_base
                )
                del record_number_delta_indexes[:number_prefix]
                del record_number_deltas[:number_prefix]
                for index in range(len(record_number_delta_indexes)):
                    record_number_delta_indexes[index] -= record_base
                record_base = 0
            del component_lengths[:component_base]
            del component_flags[:component_base]
            source_prefix = 0
            while (
                source_prefix < len(component_source_delta_indexes)
                and component_source_delta_indexes[source_prefix] < component_base
            ):
                source_prefix += 1
            del component_source_delta_indexes[:source_prefix]
            del component_source_deltas[:source_prefix]
            for index in range(len(component_source_delta_indexes)):
                component_source_delta_indexes[index] -= component_base
            char_prefix = 0
            while (
                char_prefix < len(component_char_delta_indexes)
                and component_char_delta_indexes[char_prefix] < component_base
            ):
                char_prefix += 1
            del component_char_delta_indexes[:char_prefix]
            del component_char_deltas[:char_prefix]
            for index in range(len(component_char_delta_indexes)):
                component_char_delta_indexes[index] -= component_base
            for index in range(len(record_component_ends)):
                record_component_ends[index] -= component_base
            component_base = 0
    if retained_bytes == 0 and trailing_empty_lines == 0 and not snapped_empty:
        return
    snapped_bytes = 0
    oldest_offset = write_offset if retained_bytes == count else 0
    while (
        snapped_bytes < retained_bytes
        and retained[(oldest_offset + snapped_bytes) % count] & 0xC0 == 0x80
    ):
        snapped_bytes += 1
    if snapped_bytes:
        drop_metadata(snapped_bytes, oldest_offset)
    if record_base:
        del record_component_ends[:record_base]
        del record_flags[:record_base]
        char_record_prefix = bisect_left(record_char_delta_indexes, record_base)
        del record_char_delta_indexes[:char_record_prefix]
        del record_char_deltas[:char_record_prefix]
        for index in range(len(record_char_delta_indexes)):
            record_char_delta_indexes[index] -= record_base
        if record_match_starts is not None and record_match_ends is not None:
            del record_match_starts[:record_base]
            del record_match_ends[:record_base]
        number_prefix = 0
        while (
            number_prefix < len(record_number_delta_indexes)
            and record_number_delta_indexes[number_prefix] < record_base
        ):
            number_prefix += 1
        del record_number_delta_indexes[:number_prefix]
        del record_number_deltas[:number_prefix]
        for index in range(len(record_number_delta_indexes)):
            record_number_delta_indexes[index] -= record_base
        record_base = 0
    if component_base:
        del component_lengths[:component_base]
        del component_flags[:component_base]
        source_prefix = 0
        while (
            source_prefix < len(component_source_delta_indexes)
            and component_source_delta_indexes[source_prefix] < component_base
        ):
            source_prefix += 1
        del component_source_delta_indexes[:source_prefix]
        del component_source_deltas[:source_prefix]
        for index in range(len(component_source_delta_indexes)):
            component_source_delta_indexes[index] -= component_base
        char_prefix = 0
        while (
            char_prefix < len(component_char_delta_indexes)
            and component_char_delta_indexes[char_prefix] < component_base
        ):
            char_prefix += 1
        del component_char_delta_indexes[:char_prefix]
        del component_char_deltas[:char_prefix]
        for index in range(len(component_char_delta_indexes)):
            component_char_delta_indexes[index] -= component_base
        for index in range(len(record_component_ends)):
            record_component_ends[index] -= component_base
    yield from _iter_ref_exec_tail_snapshot(
        RefExecTailSnapshot(
            retained=retained,
            write_offset=write_offset,
            component_lengths=component_lengths,
            component_flags=component_flags,
            component_source_delta_indexes=component_source_delta_indexes,
            component_source_deltas=component_source_deltas,
            component_char_delta_indexes=component_char_delta_indexes,
            component_char_deltas=component_char_deltas,
            record_component_ends=record_component_ends,
            record_flags=record_flags,
            record_char_delta_indexes=record_char_delta_indexes,
            record_char_deltas=record_char_deltas,
            record_number_delta_indexes=record_number_delta_indexes,
            record_number_deltas=record_number_deltas,
            record_match_starts=record_match_starts,
            record_match_ends=record_match_ends,
            trailing_plain_default=trailing_plain_default,
            trailing_plain_change_indexes=trailing_plain_change_indexes,
            trailing_number_delta_indexes=trailing_number_delta_indexes,
            trailing_number_deltas=trailing_number_deltas,
            trailing_char_delta_indexes=trailing_char_delta_indexes,
            trailing_char_deltas=trailing_char_deltas,
            trailing_match_indexes=trailing_match_indexes,
            trailing_match_starts=trailing_match_starts,
            trailing_match_ends=trailing_match_ends,
            compact_missing=compact_missing,
            retained_bytes=retained_bytes,
            snapped_bytes=snapped_bytes,
            stream_start=stream_start,
            stream_chars=stream_chars,
            line_count=line_count,
            trailing_empty_lines=trailing_empty_lines,
            snapped_empty=snapped_empty,
            snapped_number=snapped_number,
            snapped_char_start=snapped_char_start,
            partial_fallback_char_start=partial_fallback_char_start,
            count=count,
        )
    )


def _ref_exec_sed(lines: Iterable[RefExecLine], start: int, end: int | None, cancelled: threading.Event) -> Iterable[RefExecLine]:
    input_number = 0
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            yield line
            continue
        input_number += 1
        if input_number < start:
            continue
        if end is not None and input_number > end:
            return
        yield line


def _ref_exec_regex_requested(pattern: str, flags: frozenset[str]) -> bool:
    return "E" in flags or any(marker in pattern for marker in ("|", ".*", ".+", ".?", r"\d", r"\w", r"\s")) or bool(re.search(r"\[.+\]", pattern))


def _ref_exec_literal_component_spans(
    line: RefExecLine,
    stage: RefExecStage,
    cancelled: threading.Event,
) -> Iterable[tuple[int, int, RefExecComponent | None, int, int, str | None]]:
    pattern = stage.pattern or ""
    if not pattern:
        component = next(
            (
                candidate
                for candidate in _ref_exec_line_component_view(
                    line, cancelled
                ).components
                if candidate.kind != "synthetic_lf"
            ),
            None,
        )
        if component is None and line.component_view is None:
            component = _ref_exec_component(
                "text",
                line.text,
                cancelled,
                source_start=(line.byte_start, line.char_start),
            )
        position = component.start if component is not None else 0
        yield (0, 0, component, position, position, None)
        return
    components = (
        component
        for component in _ref_exec_line_component_view(line, cancelled).components
        if component.kind != "synthetic_lf"
    )
    compiled_literal = (
        re.compile(re.escape(pattern), re.IGNORECASE)
        if "i" in stage.flags
        else None
    )
    logical_start = 0
    overlap = len(pattern) - 1
    suffix = ""
    next_match_start = 0
    for component in components:
        _check_ref_exec_cancelled(cancelled)
        component_length = component.end - component.start
        if suffix and overlap > 0:
            current_end = min(component.end, component.start + overlap)
            boundary_text = suffix + component.text[component.start:current_end]
            boundary_offset = logical_start - len(suffix)
            boundary = len(suffix)
            search_start = max(0, next_match_start - boundary_offset)
            if compiled_literal is None:
                position = search_start
                while position <= len(boundary_text) - len(pattern):
                    found = boundary_text.find(pattern, position, len(boundary_text))
                    if found < 0:
                        break
                    end = found + len(pattern)
                    if found < boundary < end:
                        global_start = boundary_offset + found
                        global_end = boundary_offset + end
                        yield (
                            global_start,
                            global_end,
                            None,
                            0,
                            0,
                            boundary_text[found:end],
                        )
                        next_match_start = global_end
                        position = max(end, next_match_start - boundary_offset)
                    else:
                        position = found + 1
            else:
                for match in compiled_literal.finditer(
                    boundary_text, search_start, len(boundary_text)
                ):
                    start, end = match.span()
                    if start < boundary < end:
                        global_start = boundary_offset + start
                        global_end = boundary_offset + end
                        yield (
                            global_start,
                            global_end,
                            None,
                            0,
                            0,
                            boundary_text[start:end],
                        )
                        next_match_start = global_end
        if compiled_literal is None:
            position = max(
                component.start,
                component.start + next_match_start - logical_start,
            )
            while position <= component.end - len(pattern):
                found = component.text.find(pattern, position, component.end)
                if found < 0:
                    break
                global_start = logical_start + found - component.start
                global_end = global_start + len(pattern)
                yield (
                    global_start,
                    global_end,
                    component,
                    found,
                    found + len(pattern),
                    None,
                )
                next_match_start = global_end
                position = found + len(pattern)
        else:
            position = max(
                component.start,
                component.start + next_match_start - logical_start,
            )
            for match in compiled_literal.finditer(
                component.text, position, component.end
            ):
                start, end = match.span()
                global_start = logical_start + start - component.start
                global_end = logical_start + end - component.start
                yield (
                    global_start,
                    global_end,
                    component,
                    start,
                    end,
                    None,
                )
                next_match_start = global_end
        if overlap > 0:
            suffix = (
                component.text[component.end - overlap : component.end]
                if component_length >= overlap
                else (
                    suffix + component.text[component.start : component.end]
                )[-overlap:]
            )
        logical_start += component_length


def _ref_exec_source_match(
    line: RefExecLine,
    component: RefExecComponent | None,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    if (
        component is None
        or component.kind != "text"
        or component.source_char_start is None
    ):
        return None
    source_start = component.source_char_start + start - component.start
    return source_start - line.char_start, source_start - line.char_start + end - start


def _ref_exec_logical_source_match(
    line: RefExecLine,
    start: int,
    end: int,
    cancelled: threading.Event,
) -> tuple[RefExecComponent, int, int, tuple[int, int]] | None:
    logical_start = 0
    for component in _ref_exec_line_component_view(line, cancelled).components:
        if component.kind == "synthetic_lf":
            continue
        logical_end = logical_start + component.end - component.start
        if start >= logical_start and end <= logical_end:
            local_start = component.start + start - logical_start
            local_end = component.start + end - logical_start
            source_match = _ref_exec_source_match(
                line, component, local_start, local_end
            )
            if source_match is None:
                return None
            return component, local_start, local_end, source_match
        logical_start = logical_end
    return None


def _ref_exec_line_source_end(
    line: RefExecLine,
    cancelled: threading.Event,
) -> int:
    source_bounds = _ref_exec_source_byte_bounds(line)
    if source_bounds is not None:
        return source_bounds[1]
    line_bytes, _ = _measure_ref_text_checked(line.text, cancelled)
    return line.byte_start + line_bytes + int(line.has_newline)


def _ref_exec_grep(lines: Iterable[RefExecLine], stage: RefExecStage, cancelled: threading.Event) -> Iterable[RefExecLine]:
    pattern = stage.pattern or ""
    use_regex = _ref_exec_regex_requested(pattern, stage.flags)
    only_matching = "o" in stage.flags and "c" not in stage.flags
    compiled = None
    if use_regex:
        if _REGEX is None:
            raise RefExecError("Error: regex support is unavailable")
        normalized = pattern.replace(r"\|", "|")
        try:
            compiled = _REGEX.compile(normalized, _REGEX.IGNORECASE if "i" in stage.flags else 0)
        except (getattr(_REGEX, "error", RuntimeError), RuntimeError) as exc:
            raise RefExecError(f"Error: invalid regex: {exc}") from exc

    def selected() -> Iterable[RefExecLine]:
        count = 0
        budget_start: float | None = None
        last_line = 0
        last_byte = 0
        input_number = 0
        for line in lines:
            _check_ref_exec_cancelled(cancelled)
            if line.metadata_only:
                continue
            input_number += 1
            selected_line = False
            source_match: tuple[int, int] | None = None
            presented_text = None
            if only_matching:
                if compiled is not None:
                    presented_text = _ref_exec_materialize_line(
                        line, cancelled, include_synthetic_lf=False
                    )
                    if budget_start is None:
                        budget_start = time.monotonic()
                    remaining = REF_EXEC_REGEX_BUDGET_SECONDS - (
                        time.monotonic() - budget_start
                    )
                    if remaining <= 0:
                        raise RefExecError(
                            f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                        )
                    regex_matches = iter(
                        compiled.finditer(
                            presented_text,
                            timeout=min(REF_EXEC_REGEX_BUDGET_SECONDS, remaining),
                        )
                    )
                    matches = (
                        (match.span(), match.group(0), None, 0, 0)
                        for match in regex_matches
                    )
                else:
                    matches = (
                        (
                            (start, end),
                            crossing_fragment or pattern,
                            component,
                            local_start,
                            local_end,
                        )
                        for start, end, component, local_start, local_end, crossing_fragment in _ref_exec_literal_component_spans(
                            line,
                            stage,
                            cancelled=cancelled,
                        )
                    )
                measured_text: str | None = None
                measured_origin = 0
                measured_end = 0
                measured_source_byte_start = 0
                measured_bytes = 0
                while True:
                    _check_ref_exec_cancelled(cancelled)
                    if compiled is not None:
                        remaining = REF_EXEC_REGEX_BUDGET_SECONDS - (
                            time.monotonic() - budget_start
                        )
                        if remaining <= 0:
                            raise RefExecError(
                                f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                            )
                    try:
                        match_data = next(matches)
                    except StopIteration:
                        break
                    except TimeoutError as exc:
                        raise RefExecError(
                            f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                        ) from exc
                    (match_start, match_end), fragment, component, local_start, local_end = match_data
                    if match_end == match_start:
                        continue
                    source_match = _ref_exec_source_match(
                        line, component, local_start, local_end
                    )
                    if compiled is not None:
                        logical_source_match = _ref_exec_logical_source_match(
                            line,
                            match_start,
                            match_end,
                            cancelled,
                        )
                        if logical_source_match is not None:
                            component, local_start, local_end, source_match = (
                                logical_source_match
                            )
                    source_backed = source_match is not None
                    source_match_start = source_match[0] if source_match is not None else 0
                    source_prefix_bytes = 0
                    if source_backed:
                        measurement_text = (
                            component.text if component is not None else line.text
                        )
                        measurement_origin = (
                            component.start if component is not None else 0
                        )
                        measurement_start = (
                            local_start
                            if component is not None
                            else source_match_start
                        )
                        measurement_end = (
                            local_end
                            if component is not None
                            else source_match_start + match_end - match_start
                        )
                        measurement_source_byte_start = (
                            component.source_byte_start
                            if component is not None
                            and component.source_byte_start is not None
                            else line.byte_start
                        )
                        if (
                            measured_text is measurement_text
                            and measured_origin == measurement_origin
                            and measured_source_byte_start
                            == measurement_source_byte_start
                            and measured_end <= measurement_start
                        ):
                            source_prefix_bytes = measured_bytes + (
                                _ref_exec_utf8_range_bytes(
                                    measurement_text,
                                    measured_end,
                                    measurement_start,
                                    cancelled,
                                )
                            )
                        else:
                            source_prefix_bytes = _ref_exec_utf8_range_bytes(
                                measurement_text,
                                measurement_origin,
                                measurement_start,
                                cancelled,
                            )
                        measured_bytes = source_prefix_bytes + (
                            _ref_exec_utf8_range_bytes(
                                measurement_text,
                                measurement_start,
                                measurement_end,
                                cancelled,
                            )
                        )
                        measured_text = measurement_text
                        measured_origin = measurement_origin
                        measured_end = measurement_end
                        measured_source_byte_start = measurement_source_byte_start
                    else:
                        measured_text = None
                    yield replace(
                        line,
                        text=(
                            fragment
                            if compiled is not None
                            else (
                                component.text[local_start:local_end]
                                if component is not None
                                else fragment
                            )
                        ),
                        byte_start=(
                            component.source_byte_start + source_prefix_bytes
                            if source_backed
                            and component is not None
                            and component.source_byte_start is not None
                            else line.byte_start + source_prefix_bytes
                        ),
                        char_start=(
                            line.char_start + source_match_start
                            if source_backed
                            else line.char_start
                        ),
                        match_start=None,
                        match_end=None,
                        display_prefix=(
                            f"{input_number}:" if "n" in stage.flags else ""
                        ),
                        has_newline=True,
                        byte_range=None,
                        metadata_only=False,
                        atomic_match=True,
                        component_view=None,
                    )
                last_line = line.number
                last_byte = _ref_exec_line_source_end(line, cancelled)
                continue
            if compiled is not None:
                presented_text = _ref_exec_materialize_line(
                    line, cancelled, include_synthetic_lf=False
                )
                if budget_start is None:
                    budget_start = time.monotonic()
                remaining = REF_EXEC_REGEX_BUDGET_SECONDS - (time.monotonic() - budget_start)
                if remaining <= 0:
                    raise RefExecError(
                        f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                    )
                try:
                    # regex requires one contiguous string for cross-boundary matches.
                    # Keep that unavoidable temporary to one current logical line and
                    # only when a prior stage contributed presentation text.
                    match = compiled.search(
                        presented_text,
                        timeout=min(REF_EXEC_REGEX_BUDGET_SECONDS, remaining),
                    )
                except TimeoutError as exc:
                    raise RefExecError(
                        f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                    ) from exc
                if match is not None:
                    selected_line = True
                    match_start, match_end = match.span()
                    logical_source_match = _ref_exec_logical_source_match(
                        line,
                        match_start,
                        match_end,
                        cancelled,
                    )
                    if logical_source_match is not None:
                        _, _, _, source_match = logical_source_match
            else:
                literal_match = next(
                    iter(
                        _ref_exec_literal_component_spans(
                            line,
                            stage,
                            cancelled=cancelled,
                        )
                    ),
                    None,
                )
                selected_line = literal_match is not None
                if literal_match is not None:
                    _, _, component, local_start, local_end, _ = literal_match
                    source_match = _ref_exec_source_match(
                        line, component, local_start, local_end
                    )
            last_line = line.number
            last_byte = _ref_exec_line_source_end(line, cancelled)
            if not selected_line:
                continue
            count += 1
            if "c" not in stage.flags:
                display_prefix = (
                    f"{input_number}:" if "n" in stage.flags else ""
                )
                component_view = line.component_view
                if component_view is not None and display_prefix:
                    component_view = RefExecComponentView(
                        (
                            _ref_exec_component(
                                "display_prefix", display_prefix, cancelled
                            ),
                            *component_view.components,
                        )
                    )
                    display_prefix = ""
                elif component_view is None:
                    display_prefix += line.display_prefix
                yield replace(
                    line,
                    match_start=source_match[0] if source_match is not None else None,
                    match_end=source_match[1] if source_match is not None else None,
                    display_prefix=display_prefix,
                    component_view=component_view,
                )
        if "c" in stage.flags:
            yield RefExecLine(str(count), 1, 0, 0, False)

    return selected()


def _count_ref_words_checked(
    text: str,
    cancelled: threading.Event,
    *,
    prefix: str = "",
) -> int:
    count = 0
    previous_ended_in_word = False
    for part in (prefix, text):
        for offset in range(0, len(part), REF_TEXT_HASH_CHUNK_CHARS):
            _check_ref_exec_cancelled(cancelled)
            chunk = part[offset : offset + REF_TEXT_HASH_CHUNK_CHARS]
            chunk_count = sum(1 for _ in re.finditer(r"\S+", chunk))
            if previous_ended_in_word and chunk and not chunk[0].isspace():
                chunk_count -= 1
            count += chunk_count
            previous_ended_in_word = bool(chunk) and not chunk[-1].isspace()
    return count


def _ref_exec_wc(lines: Iterable[RefExecLine], flag: str, cancelled: threading.Event) -> Iterable[RefExecLine]:
    line_count = 0
    word_count = 0
    byte_count = 0
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            continue
        line_count += 1
        if line.component_view is None:
            word_count += _count_ref_words_checked(
                line.text, cancelled, prefix=line.display_prefix
            )
            encoded_bytes, _ = _measure_ref_text_parts_checked(
                (line.display_prefix, line.text), cancelled
            )
            byte_count += encoded_bytes + int(line.has_newline)
            continue
        previous_ended_in_word = False
        for component in line.component_view.components:
            byte_count += component.utf8_bytes
            for offset in range(
                component.start, component.end, REF_TEXT_HASH_CHUNK_CHARS
            ):
                _check_ref_exec_cancelled(cancelled)
                chunk_end = min(
                    component.end, offset + REF_TEXT_HASH_CHUNK_CHARS
                )
                chunk = component.text[offset:chunk_end]
                chunk_count = sum(1 for _ in re.finditer(r"\S+", chunk))
                if previous_ended_in_word and chunk and not chunk[0].isspace():
                    chunk_count -= 1
                word_count += chunk_count
                previous_ended_in_word = bool(chunk) and not chunk[-1].isspace()
    value = {"l": line_count, "w": word_count, "c": byte_count}[flag]
    yield RefExecLine(str(value), 1, 0, 0, False)


def _ref_exec_stat(entry: RefCatalogEntry, cancelled: threading.Event) -> Iterable[RefExecLine]:
    line_count = 0
    word_count = 0
    character_count = 0
    for line in _iter_ref_source_lines(entry.source, cancelled):
        line_count += 1
        word_count += _count_ref_words_checked(line.text, cancelled)
        character_count += len(line.text) + int(line.has_newline)
    manifest = entry.manifest
    text = (
        f"ref={manifest.ref} kind={manifest.ref.split(':', 1)[0]} utf8_bytes={manifest.utf8_bytes} "
        f"lines={line_count} words={word_count} chars={character_count} sha256={manifest.sha256}"
    )
    yield RefExecLine(text, 1, 0, 0, False)


def _ref_exec_response_fits(
    text: str,
    *,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> bool:
    encoded_bytes, _ = _measure_ref_text_checked(text, cancelled)
    if encoded_bytes > REF_EXEC_RESPONSE_MAX_BYTES:
        return False
    if encoder is None:
        return encoded_bytes < threshold_tokens
    token_count = _ref_exec_encode_text_token_count(encoder, text)
    if token_count is None:
        return encoded_bytes < threshold_tokens
    return token_count < threshold_tokens


def _ref_exec_truncated_marker(next_command: str) -> str:
    payload = json.dumps(
        {"next": next_command},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "\n<agent_ref_truncated>"
        f"{payload}"
        "</agent_ref_truncated>"
    )


def _ref_exec_truncated_prefix(
    text: str,
    *,
    marker: str,
    continuation_ref: str | None = None,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    marker_bytes, _ = _measure_ref_text_checked(marker, cancelled)
    bounded_text = _ref_exec_utf8_prefix(
        text,
        max(0, REF_EXEC_RESPONSE_MAX_BYTES - marker_bytes),
        cancelled,
    )
    low = 0
    high = len(bounded_text)
    while low < high:
        midpoint = (low + high + 1) // 2
        prefix = bounded_text[:midpoint]
        prefix_bytes, _ = _measure_ref_text_checked(prefix, cancelled)
        candidate_marker = (
            marker
            if continuation_ref is None
            else _ref_exec_truncated_marker(
                f"tail -c +{prefix_bytes + 1} {continuation_ref}"
            )
        )
        candidate = prefix + candidate_marker
        if _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        ):
            low = midpoint
        else:
            high = midpoint - 1
    prefix = bounded_text[:low]
    prefix_bytes, _ = _measure_ref_text_checked(prefix, cancelled)
    return prefix + (
        marker
        if continuation_ref is None
        else _ref_exec_truncated_marker(
            f"tail -c +{prefix_bytes + 1} {continuation_ref}"
        )
    )


def _ref_exec_component_view_prefix(
    view: RefExecComponentView,
    max_bytes: int,
    cancelled: threading.Event,
) -> str:
    parts: list[str] = []
    remaining = max_bytes
    for component in view.components:
        if remaining <= 0:
            break
        if component.utf8_bytes <= remaining:
            parts.append(component.text[component.start : component.end])
            remaining -= component.utf8_bytes
            continue
        end, retained = _ref_exec_component_prefix_index(
            component, remaining, cancelled
        )
        parts.append(component.text[component.start:end])
        remaining -= retained
        break
    return "".join(parts)


def _ref_exec_grep_excerpt(
    line: RefExecLine,
    *,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    match_start = line.match_start or 0
    match_end = line.match_end if line.match_end is not None else match_start
    if line.component_view is None:
        source_text = line.text
        source_start = 0
        source_end = len(source_text)
        display_prefix = line.display_prefix
    else:
        text_component = next(
            (
                component
                for component in line.component_view.components
                if component.kind == "text"
            ),
            None,
        )
        if text_component is None:
            raise RefExecError("Error: grep excerpt has no source-backed text")
        source_text = text_component.text
        source_start = text_component.start
        source_end = text_component.end
        match_start += source_start
        match_end += source_start
        display_prefix = _ref_exec_materialize_components(
            (
                component
                for component in line.component_view.components
                if component.kind == "display_prefix"
            )
        )
    line_bytes = _ref_exec_utf8_range_bytes(
        source_text, source_start, source_end, cancelled
    )
    prefix_bytes = _ref_exec_utf8_range_bytes(
        source_text, source_start, match_start, cancelled
    )
    match_bytes = _ref_exec_utf8_range_bytes(
        source_text, match_start, match_end, cancelled
    )
    display_prefix_bytes, _ = _measure_ref_text_checked(display_prefix, cancelled)
    absolute_byte_start = line.byte_start + prefix_bytes
    absolute_char_start = line.char_start + match_start - source_start
    left = max(source_start, match_start - 8_192)
    right = min(source_end, max(match_end, match_start + 1) + 8_192)
    while True:
        omitted_prefix = _ref_exec_utf8_range_bytes(
            source_text, source_start, left, cancelled
        )
        visible_end_bytes = _ref_exec_utf8_range_bytes(
            source_text, source_start, right, cancelled
        )
        omitted_suffix = line_bytes - visible_end_bytes
        marker_payload = json.dumps(
            {
                "line": line.number,
                "match_byte_start": absolute_byte_start,
                "match_byte_end": absolute_byte_start + match_bytes,
                "match_char_start": absolute_char_start,
                "match_char_end": line.char_start + match_end - source_start,
                "omitted_prefix_bytes": omitted_prefix,
                "omitted_suffix_bytes": omitted_suffix,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        marker = f"\n<agent_ref_excerpt>{marker_payload}</agent_ref_excerpt>"
        visible_bytes = visible_end_bytes - omitted_prefix
        marker_bytes, _ = _measure_ref_text_checked(marker, cancelled)
        if display_prefix_bytes + visible_bytes + marker_bytes > REF_EXEC_RESPONSE_MAX_BYTES:
            if left >= match_start and right <= match_end:
                raise RefExecError("Error: response budget cannot contain the complete grep match")
            left = min(match_start, left + max(1, (match_start - left) // 2))
            right = max(match_end, right - max(1, (right - match_end) // 2))
            continue
        candidate = display_prefix + source_text[left:right] + marker
        if _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        ):
            return candidate
        if left >= match_start and right <= match_end:
            raise RefExecError("Error: response budget cannot contain the complete grep match")
        left = min(match_start, left + max(1, (match_start - left) // 2))
        right = max(match_end, right - max(1, (right - match_end) // 2))


def _ref_exec_format_byte_range(start: int, end: int | None) -> str:
    return f"{start}-{'*' if end is None else end}"


def _ref_exec_byte_range_marker(
    byte_range: RefExecByteRange,
    *,
    continuation: bool = False,
    next_command: str | None = None,
) -> str:
    metadata = {
        "actual": (
            None
            if byte_range.actual_empty
            or byte_range.actual_end is not None
            and byte_range.actual_end < byte_range.actual_start
            else _ref_exec_format_byte_range(
                byte_range.actual_start, byte_range.actual_end
            )
        ),
        "requested": _ref_exec_format_byte_range(
            byte_range.requested_start, byte_range.requested_end
        ),
    }
    if continuation:
        metadata["next"] = next_command or "wc|grep|head|tail|sed"
    payload = json.dumps(
        metadata,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"\n<agent_ref_range>{payload}</agent_ref_range>"


def _ref_exec_truncated_byte_response(
    text: str,
    *,
    byte_range: RefExecByteRange,
    continuation_ref: str | None = None,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    bounded_text = _ref_exec_utf8_prefix(text, REF_EXEC_RESPONSE_MAX_BYTES, cancelled)
    low = 0
    high = len(bounded_text)
    result = ""
    while low <= high:
        midpoint = (low + high) // 2
        prefix = bounded_text[:midpoint]
        prefix_bytes, _ = _measure_ref_text_checked(prefix, cancelled)
        actual_end = byte_range.actual_start + prefix_bytes - 1
        marker_range = replace(
            byte_range,
            actual_end=actual_end,
            marked=True,
            actual_empty=prefix_bytes == 0,
        )
        next_command = None
        if continuation_ref is not None:
            next_command = f"tail -c +{actual_end + 1} {continuation_ref}"
            if byte_range.requested_end is not None:
                remaining = max(0, byte_range.requested_end - actual_end)
                next_command += f" | head -c {remaining}"
        candidate = prefix + _ref_exec_byte_range_marker(
            marker_range,
            continuation=True,
            next_command=next_command,
        )
        if _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        ):
            result = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return result


def _collect_ref_exec_response(
    lines: Iterable[RefExecLine],
    *,
    continuation_ref: str | None,
    final_grep: bool,
    preserve_source_newlines: bool,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    response = ""
    response_bytes = 0
    emitted_output = False
    selected_byte_range: RefExecByteRange | None = None
    atomic_boundaries = [0]
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.byte_range is not None:
            current_range = line.byte_range
            if selected_byte_range is None or (
                selected_byte_range.requested_start,
                selected_byte_range.requested_end,
            ) != (current_range.requested_start, current_range.requested_end):
                selected_byte_range = current_range
            else:
                existing_has_actual = not selected_byte_range.actual_empty
                current_has_actual = not current_range.actual_empty
                selected_byte_range = replace(
                    current_range,
                    actual_start=(
                        selected_byte_range.actual_start
                        if existing_has_actual
                        else current_range.actual_start
                    ),
                    actual_end=(
                        current_range.actual_end
                        if current_has_actual
                        else selected_byte_range.actual_end
                    ),
                    marked=selected_byte_range.marked or current_range.marked,
                    actual_empty=not (existing_has_actual or current_has_actual),
                )
            if line.metadata_only:
                continue
        separator = "\n" if emitted_output and not preserve_source_newlines else ""
        if line.component_view is None:
            line_ending = "\n" if preserve_source_newlines and line.has_newline else ""
            line_bytes, _ = _measure_ref_text_parts_checked(
                (separator, line.display_prefix, line.text, line_ending), cancelled
            )
        else:
            line_ending = ""
            line_bytes = len(separator.encode("utf-8")) + sum(
                component.utf8_bytes
                for component in line.component_view.components
                if preserve_source_newlines or component.kind != "synthetic_lf"
            )
        complete_bytes = response_bytes + line_bytes
        candidate: str | None = None
        if complete_bytes <= REF_EXEC_RESPONSE_MAX_BYTES:
            if line.component_view is None:
                presented_line = line.display_prefix + line.text
            else:
                presented_line = _ref_exec_materialize_line(
                    line,
                    cancelled,
                    include_synthetic_lf=preserve_source_newlines,
                )
            candidate = response + separator + presented_line + line_ending
            if selected_byte_range is not None and not final_grep:
                response = candidate
                response_bytes = complete_bytes
                emitted_output = True
                if line.atomic_match:
                    atomic_boundaries.append(len(response))
                continue
            if _ref_exec_response_fits(
                candidate,
                threshold_tokens=threshold_tokens,
                encoder=encoder,
                cancelled=cancelled,
            ):
                response = candidate
                response_bytes = complete_bytes
                emitted_output = True
                if line.atomic_match:
                    atomic_boundaries.append(len(response))
                continue
        if line.atomic_match:
            if not emitted_output:
                return "Error: complete grep match exceeds the response budget; page source bytes with tail -c +N REF"
            marker = _ref_exec_truncated_marker("wc|grep|head|tail|sed")
            for boundary in reversed(atomic_boundaries):
                atomic_response = response[:boundary] + marker
                if _ref_exec_response_fits(
                    atomic_response,
                    threshold_tokens=threshold_tokens,
                    encoder=encoder,
                    cancelled=cancelled,
                ):
                    return atomic_response
            return marker
        if final_grep and line.match_start is not None and not emitted_output:
            return _ref_exec_grep_excerpt(
                line,
                threshold_tokens=threshold_tokens,
                encoder=encoder,
                cancelled=cancelled,
            )
        marker = _ref_exec_truncated_marker("wc|grep|head|tail|sed")
        if candidate is None:
            remaining_bytes = max(
                0,
                REF_EXEC_RESPONSE_MAX_BYTES - response_bytes - len(separator),
            )
            if line.component_view is None:
                display_prefix = _ref_exec_utf8_prefix(
                    line.display_prefix, remaining_bytes, cancelled
                )
                display_prefix_bytes, _ = _measure_ref_text_checked(
                    display_prefix, cancelled
                )
                source_prefix = (
                    _ref_exec_utf8_prefix(
                        line.text,
                        remaining_bytes - display_prefix_bytes,
                        cancelled,
                    )
                    if display_prefix == line.display_prefix
                    else ""
                )
                line_prefix = display_prefix + source_prefix
            else:
                line_prefix = _ref_exec_component_view_prefix(
                    line.component_view, remaining_bytes, cancelled
                )
            candidate = response + separator + line_prefix
        if selected_byte_range is not None:
            return _ref_exec_truncated_byte_response(
                candidate,
                byte_range=selected_byte_range,
                continuation_ref=continuation_ref,
                threshold_tokens=threshold_tokens,
                encoder=encoder,
                cancelled=cancelled,
            )
        return _ref_exec_truncated_prefix(
            candidate,
            marker=marker,
            continuation_ref=continuation_ref,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )
    if selected_byte_range is not None and selected_byte_range.marked:
        candidate = response + _ref_exec_byte_range_marker(selected_byte_range)
        if _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        ):
            return candidate
        return _ref_exec_truncated_byte_response(
            response,
            byte_range=selected_byte_range,
            continuation_ref=continuation_ref,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )
    if selected_byte_range is not None and not _ref_exec_response_fits(
        response,
        threshold_tokens=threshold_tokens,
        encoder=encoder,
        cancelled=cancelled,
    ):
        return _ref_exec_truncated_byte_response(
            response,
            byte_range=selected_byte_range,
            continuation_ref=continuation_ref,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )
    return response


def _execute_ref_reader_sync(
    stages: tuple[RefExecStage, ...],
    catalog: tuple[RefCatalogEntry, ...],
    *,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    first = stages[0]
    verified_source: Iterable[RefExecLine] | None = None
    if first.command == "ls":
        lines: Iterable[RefExecLine] = (
            RefExecLine(entry.manifest.ref, index, 0, 0, False)
            for index, entry in enumerate(catalog, 1)
            if first.list_kind is None
            or entry.manifest.ref.startswith(f"{first.list_kind}:")
        )
    else:
        requested_hash = parse_ref(first.ref or "") if first.ref else None
        entry = next(
            (
                candidate
                for candidate in catalog
                if requested_hash is not None
                and (candidate_hash := parse_ref(candidate.manifest.ref)) is not None
                and candidate_hash.value == requested_hash.value
            ),
            None,
        )
        if entry is None:
            raise RefExecError(
                "Error: externalized ref is not available in this binding. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        first_is_byte_stage = first.byte_count is not None or first.byte_start is not None
        if first.command in {"grep", "wc"} or first_is_byte_stage:
            verified_source = _iter_verified_ref_source_lines(entry, cancelled)
            lines = verified_source
        else:
            measured_bytes, measured_hash = _measure_ref_source_checked(entry.source, cancelled)
            if measured_bytes != entry.manifest.utf8_bytes or measured_hash != entry.manifest.sha256:
                raise RefExecError("Error: externalized ref integrity verification failed")
            lines = _ref_exec_stat(entry, cancelled) if first.command == "stat" else _iter_ref_source_lines(entry.source, cancelled)
        if first.command == "head":
            lines = (
                _ref_exec_head_bytes(lines, first.byte_count, cancelled)
                if first.byte_count is not None
                else _ref_exec_head(lines, first.count or 0, cancelled)
            )
        elif first.command == "tail":
            if first.byte_start is not None:
                lines = _ref_exec_tail_from_bytes(lines, first.byte_start, cancelled)
            elif first.byte_count is not None:
                lines = _ref_exec_tail_bytes(lines, first.byte_count, cancelled)
            else:
                count = first.count or 0
                if count == 0:
                    lines = iter(())
                else:
                    total = _ref_exec_source_line_count(entry.source)
                    lines = _ref_exec_sed(
                        lines, max(1, total - count + 1), None, cancelled
                    )
        elif first.command == "sed":
            lines = _ref_exec_sed(lines, first.start_line or 1, first.end_line, cancelled)
        elif first.command == "grep":
            lines = _ref_exec_grep(lines, first, cancelled)
        elif first.command == "wc":
            lines = _ref_exec_wc(lines, next(iter(first.flags)), cancelled)
    for stage in stages[1:]:
        if stage.command == "head":
            lines = (
                _ref_exec_head_bytes(lines, stage.byte_count, cancelled)
                if stage.byte_count is not None
                else _ref_exec_head(lines, stage.count or 0, cancelled)
            )
        elif stage.command == "tail":
            if stage.byte_start is not None:
                lines = _ref_exec_tail_from_bytes(lines, stage.byte_start, cancelled)
            elif stage.byte_count is not None:
                lines = _ref_exec_tail_bytes(lines, stage.byte_count, cancelled)
            else:
                lines = _ref_exec_tail(lines, stage.count or 0, cancelled)
        elif stage.command == "sed":
            lines = _ref_exec_sed(lines, stage.start_line or 1, stage.end_line, cancelled)
        elif stage.command == "grep":
            lines = _ref_exec_grep(lines, stage, cancelled)
        elif stage.command == "wc":
            lines = _ref_exec_wc(lines, next(iter(stage.flags)), cancelled)
    final = stages[-1]
    bounded_source_range = (
        len(stages) == 2
        and first.command == "tail"
        and first.byte_start is not None
        and final.command == "head"
        and final.byte_count is not None
    )
    if bounded_source_range:
        # The preview's recovery command is tail -c +N REF | head -c M.
        # head's input-relative range must retain the original source window
        # so every subsequent page stops before the already-visible tail.
        def source_ranges(selected: Iterable[RefExecLine]) -> Iterable[RefExecLine]:
            source_start = None
            for line in selected:
                byte_range = line.byte_range
                if byte_range is not None:
                    if source_start is None and not byte_range.actual_empty:
                        source_start = byte_range.actual_start
                    line = replace(
                        line,
                        byte_range=replace(
                            byte_range,
                            requested_start=first.byte_start,
                            requested_end=(source_start or first.byte_start) + final.byte_count - 1,
                        ),
                    )
                yield line

        lines = source_ranges(lines)
    continuation_ref = (
        first.ref
        if bounded_source_range
        or len(stages) == 1
        and (first.command == "cat" or first.command == "tail" and first.byte_start is not None)
        else None
    )
    response = _collect_ref_exec_response(
        lines,
        continuation_ref=continuation_ref,
        final_grep=final.command == "grep" and "c" not in final.flags,
        preserve_source_newlines=(
            len(stages) == 1 and first.command in {"cat", "head"}
        )
        or final.byte_count is not None
        or final.byte_start is not None,
        threshold_tokens=threshold_tokens,
        encoder=encoder,
        cancelled=cancelled,
    )
    if verified_source is not None:
        for _ in verified_source:
            _check_ref_exec_cancelled(cancelled)
    return response
RefTextMeasurement: TypeAlias = tuple[int, int, str]


def _measure_ref_text(text: str) -> RefTextMeasurement | None:
    digest = hashlib.sha256()
    utf8_bytes = 0
    line_count = 0
    for offset in range(0, len(text), REF_TEXT_HASH_CHUNK_CHARS):
        chunk = text[offset : offset + REF_TEXT_HASH_CHUNK_CHARS]
        try:
            encoded = chunk.encode("utf-8")
        except UnicodeEncodeError:
            return None
        utf8_bytes += len(encoded)
        line_count += chunk.count("\n")
        digest.update(encoded)
    if text and not text.endswith("\n"):
        line_count += 1
    return utf8_bytes, line_count, digest.hexdigest()


def _classify_ref_text_sync(
    text: str,
    *,
    threshold_tokens: int,
    encoder: Any = None,
    request: Any = None,
) -> RefTextClassification:
    utf8_bytes = 0
    line_count = 0
    for offset in range(0, len(text), REF_TEXT_HASH_CHUNK_CHARS):
        chunk = text[offset : offset + REF_TEXT_HASH_CHUNK_CHARS]
        try:
            utf8_bytes += len(chunk.encode("utf-8"))
        except UnicodeEncodeError:
            return RefTextClassification(
                eligible=False,
                utf8_bytes=None,
                sha256=None,
                line_count=None,
                token_count=None,
                encoder_failed=False,
            )
        line_count += chunk.count("\n")
    if text and not text.endswith("\n"):
        line_count += 1
    token_count = None
    encoder_failed = False
    eligible = utf8_bytes > REF_EXEC_CLASSIFY_EXACT_ENCODE_MAX_BYTES
    resolved_encoder = encoder
    if not eligible:
        if resolved_encoder is None:
            resolved_encoder, _ = _ref_exec_get_tiktoken_encoder(request)
        if resolved_encoder is None:
            encoder_failed = True
        else:
            token_count = _ref_exec_encode_text_token_count(resolved_encoder, text)
            encoder_failed = token_count is None
            eligible = token_count is not None and token_count >= threshold_tokens
    if not eligible:
        return RefTextClassification(
            eligible=False,
            utf8_bytes=utf8_bytes,
            sha256=None,
            line_count=line_count,
            token_count=token_count,
            encoder_failed=encoder_failed,
        )
    measurement = _measure_ref_text(text)
    if measurement is None:
        return RefTextClassification(
            eligible=False,
            utf8_bytes=None,
            sha256=None,
            line_count=None,
            token_count=None,
            encoder_failed=False,
        )
    utf8_bytes, line_count, text_hash = measurement
    return RefTextClassification(
        eligible=True,
        utf8_bytes=utf8_bytes,
        sha256=text_hash,
        line_count=line_count,
        token_count=token_count,
        encoder_failed=False,
    )


async def classify_ref_text(
    text: str,
    *,
    threshold_tokens: int,
    encoder: Any = None,
    request: Any = None,
) -> RefTextClassification:
    return await asyncio.to_thread(
        _classify_ref_text_sync,
        text,
        threshold_tokens=threshold_tokens,
        encoder=encoder,
        request=request,
    )


def _native_tool_names_by_call_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    candidate_names: dict[str, list[str | None]] = {}
    for assistant_message in messages:
        if assistant_message.get("role") != "assistant":
            continue
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(call_id, str) and call_id:
                candidate_names.setdefault(call_id, []).append(
                    name if isinstance(name, str) and name else None
                )
    names_by_call_id: dict[str, str] = {}
    for call_id, names in candidate_names.items():
        if REF_EXEC_TOOL_NAME in names:
            names_by_call_id[call_id] = REF_EXEC_TOOL_NAME
        elif len(valid_names := {name for name in names if name is not None}) == 1:
            names_by_call_id[call_id] = next(iter(valid_names))
        else:
            names_by_call_id[call_id] = "unknown"
    return names_by_call_id

def _render_tool_ref_preview_sync(
    text: str,
    ref: str,
    utf8_bytes: int,
    *,
    threshold_tokens: int,
    encoder: Any,
) -> str | None:
    """Keep a near-limit head/tail preview with byte-exact middle recovery.

    Match Core's d40225da3b03 contract: the marker counts against both caps,
    and a preview the token counter cannot measure is judged by the same
    byte upper bound the reader uses. Only bounded edge fragments are
    copied or tokenized, even when the source is a single huge line.
    """
    if utf8_bytes < 2:
        return None
    cancelled = threading.Event()
    marker_overhead = len(
        _ref_exec_truncated_marker(
            f"tail -c +{utf8_bytes + 1} {ref} | head -c {utf8_bytes}"
        )
    ) + 1
    maximum = min(utf8_bytes - 1, REF_EXEC_RESPONSE_MAX_BYTES - marker_overhead)
    if maximum < 0:
        return None
    # UTF-8 uses at least one byte per character. These edge windows suffice
    # for every candidate without encoding or indexing the complete source.
    head_bytes = text[:maximum].encode("utf-8")
    tail_bytes = text[-maximum:].encode("utf-8") if maximum else b""

    def preview_at(prefix_end: int, suffix_start: int) -> str | None:
        if not 0 <= prefix_end <= suffix_start <= len(text):
            return None
        prefix = text[:prefix_end]
        suffix = text[suffix_start:]
        prefix_size = len(prefix.encode("utf-8"))
        omitted = utf8_bytes - prefix_size - len(suffix.encode("utf-8"))
        if omitted <= 0:
            return None
        command = f"tail -c +{prefix_size + 1} {ref} | head -c {omitted}"
        return prefix + _ref_exec_truncated_marker(command) + "\n" + suffix

    def fits(candidate: str | None) -> bool:
        if candidate is None:
            return False
        return _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )

    best = None
    prefix_end, suffix_start = 0, len(text)
    low, high = 0, maximum
    while low <= high:
        retained = (low + high) // 2
        prefix = head_bytes[: retained // 2].decode("utf-8", errors="ignore")
        suffix_size = retained - len(prefix.encode("utf-8"))
        suffix = tail_bytes[-suffix_size:].decode("utf-8", errors="ignore") if suffix_size else ""
        ends = (len(prefix), len(text) - len(suffix))
        candidate = preview_at(*ends)
        if fits(candidate):
            best, (prefix_end, suffix_start) = candidate, ends
            low = retained + 1
        else:
            high = retained - 1
    if best is None:
        return None
    # As in Core, polish each edge after the byte search. The cap bounds
    # work for token counters whose result is not monotone in text length.
    for _ in range(8):
        grew = False
        for ends in ((prefix_end + 1, suffix_start), (prefix_end, suffix_start - 1)):
            candidate = preview_at(*ends)
            if fits(candidate):
                best, (prefix_end, suffix_start) = candidate, ends
                grew = True
        if not grew:
            break
    return best

def _cached_tool_ref_preview_sync(
    text: str,
    ref: str,
    utf8_bytes: int,
    *,
    threshold_tokens: int,
    encoder: Any,
    request: Any,
    cache: dict[tuple[str, int, int], tuple[Any, str]],
) -> str | None:
    if encoder is None:
        encoder, _ = _ref_exec_get_tiktoken_encoder(request)
    key = (ref, threshold_tokens, id(encoder))
    cached = cache.get(key)
    if cached is not None:
        return cached[1]
    preview = _render_tool_ref_preview_sync(
        text, ref, utf8_bytes, threshold_tokens=threshold_tokens, encoder=encoder
    )
    if preview is not None:
        # Retain the encoder to prevent ID reuse within this run. The
        # cache holds bounded renders, not raw sources or reader bindings.
        cache[key] = (encoder, preview)
    return preview


async def project_native_tool_texts(
    messages: list[dict[str, Any]],
    *,
    threshold_tokens: int,
    encoder: Any = None,
    request: Any = None,
    preview_cache: dict[tuple[str, int, int], tuple[Any, str]] | None = None,
    classification_cache: dict[str, RefTextClassification] | None = None,
) -> RefProjectionPlan:
    """Classify oversized native tool texts and build their ref projection plan."""
    catalog_by_hash: dict[str, RefCatalogEntry] = {}
    if preview_cache is None:
        preview_cache = {}
    names_by_call_id = _native_tool_names_by_call_id(messages)

    eligible_results: list[tuple[str, str, RefTextMeasurement]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        tool_call_id = message.get("tool_call_id")
        paired_name = (
            names_by_call_id.get(tool_call_id, "unknown")
            if isinstance(tool_call_id, str)
            else "unknown"
        )
        classification = None
        if classification_cache is not None:
            classification = classification_cache.get(content)
        if classification is None:
            classification = await classify_ref_text(
                content,
                threshold_tokens=threshold_tokens,
                encoder=encoder,
                request=request,
            )
            if classification_cache is not None:
                classification_cache[content] = classification
        if not classification.eligible:
            continue
        utf8_bytes = classification.utf8_bytes
        line_count = classification.line_count
        text_hash = classification.sha256
        if utf8_bytes is None or line_count is None or text_hash is None:
            raise InvalidRefTextClassificationError
        measurement: RefTextMeasurement = (utf8_bytes, line_count, text_hash)
        eligible_results.append((content, paired_name, measurement))

    for content, paired_name, measurement in eligible_results:
        utf8_bytes, line_count, text_hash = measurement
        ref = f"tool:{text_hash}"
        preview = await asyncio.to_thread(
            _cached_tool_ref_preview_sync,
            content,
            ref,
            utf8_bytes,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            request=request,
            cache=preview_cache,
        )
        if preview is None:
            # Fail closed: eligible text must never reach the provider raw.
            raise RefProjectionError(stage="tool preview rendering")
        manifest = RefManifest(
            ref=ref,
            utf8_bytes=utf8_bytes,
            sha256=text_hash,
        )
        source = ZeroCopySourceHandle(text=content)
        catalog_by_hash.setdefault(
            text_hash,
            RefCatalogEntry(
                manifest=manifest,
                source=source,
                preview_text=preview,
            ),
        )
    catalog = tuple(catalog_by_hash.values())
    return RefProjectionPlan(
        catalog=catalog,
        manifests=tuple(entry.manifest for entry in catalog),
        reader_schema=REF_EXEC_TOOL_SPEC if catalog else None,
    )

def _apply_ref_projection_plan_sync(
    messages: list[dict[str, Any]],
    plan: RefProjectionPlan,
) -> list[dict[str, Any]]:
    projected = copy.deepcopy(messages)
    source_previews: dict[str, str] = {}
    for entry in plan.catalog:
        parsed = parse_ref(entry.manifest.ref)
        if (
            parsed is not None
            and parsed.kind == "tool"
            and isinstance(entry.source, ZeroCopySourceHandle)
            and entry.preview_text is not None
        ):
            source_previews.setdefault(entry.source.text, entry.preview_text)
    for message in projected:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        preview = source_previews.get(content)
        if preview is None:
            continue
        message["content"] = preview
    return projected


async def apply_ref_projection_plan(
    messages: list[dict[str, Any]],
    plan: RefProjectionPlan,
) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_apply_ref_projection_plan_sync, messages, plan)


def _ref_exec_encode_text_token_count(encoder: Any, text: str) -> int | None:
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except TypeError:
        try:
            return len(encoder.encode(text))
        except Exception:
            return None
    except Exception:
        return None


def _ref_exec_tiktoken_encoding_names(request: Any = None) -> list[str]:
    names: list[str] = []
    config = getattr(getattr(getattr(request, "app", None), "state", None), "config", None)
    configured = getattr(config, "TIKTOKEN_ENCODING_NAME", None)
    if configured:
        names.append(str(configured))
    try:
        from open_webui import config as open_webui_config

        fallback = getattr(open_webui_config, "TIKTOKEN_ENCODING_NAME", None)
        if fallback:
            names.append(str(fallback))
    except Exception:
        pass
    names.append("cl100k_base")
    return list(dict.fromkeys(names))


def _ref_exec_get_tiktoken_encoder(request: Any = None) -> tuple[Any | None, str | None]:
    try:
        import tiktoken
    except Exception:
        return None, None

    for encoding_name in _ref_exec_tiktoken_encoding_names(request):
        try:
            return tiktoken.get_encoding(encoding_name), encoding_name
        except Exception:
            continue
    return None, None


REF_EXEC_TOOL_SPEC = MappingProxyType(
    {
        "type": "function",
        "function": MappingProxyType(
            {
                "name": REF_EXEC_TOOL_NAME,
                "description": (
                    "Read externalized content in this sub-agent run. Oversized tool results include a bounded head/tail preview and a tool:<64 hex> ref; compacted history uses history:<64 hex>. "
                    "A <agent_ref_truncated> marker embeds a next command to read the omitted span; follow continuation commands across pages when needed. Commands: ls [tool|history]; stat REF; "
                    "wc -l|-w|-c REF; cat REF; head [-n N|-N|-c N] REF; tail [-n N|-N|-c N|-c +N] REF; sed -n 'M,Np' REF; grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN REF "
                    "(patterns match literally unless regex syntax is auto-detected; -E forces regex). REF is the complete token including its tool:/history: prefix, exactly as written. Pipelines are supported; "
                    "only grep/head/tail/sed/wc consume piped input, e.g. grep -n PATTERN tool:<hash> | head -20. Start with stat, then prefer grep/sed/head over cat for large refs."
                ),
                "parameters": MappingProxyType(
                    {
                        "type": "object",
                        "properties": MappingProxyType(
                            {
                                "command": MappingProxyType(
                                    {
                                        "type": "string",
                                        "description": "One command line, max 1,024 UTF-8 bytes, e.g. sed -n '1,120p' tool:<64 hex>",
                                    }
                                )
                            }
                        ),
                        "required": ("command",),
                        "additionalProperties": False,
                    }
                ),
            }
        ),
    }
)


def _mutable_ref_schema_value(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _mutable_ref_schema_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_ref_schema_value(item) for item in value]
    return value


def ref_exec_tool_spec_payload() -> dict[str, Any]:
    payload = _mutable_ref_schema_value(REF_EXEC_TOOL_SPEC)
    if not isinstance(payload, dict):
        raise RefProjectionError(stage="reader schema rendering")
    return payload


def _truncation_only_marker(omitted_bytes: int) -> str:
    payload = json.dumps(
        {"omitted": omitted_bytes},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"\n<{'agent_ref_truncated'}>{payload}</{'agent_ref_truncated'}>"


def render_truncate_preview_sync(
    text: str,
    utf8_bytes: int,
    *,
    threshold_tokens: int,
    encoder: Any,
) -> str | None:
    """Irreversible middle truncation using the ref-exec preview renderer."""
    if utf8_bytes < 2:
        return None
    cancelled = threading.Event()
    marker_overhead = len(_truncation_only_marker(utf8_bytes)) + 1
    maximum = min(utf8_bytes - 1, REF_EXEC_RESPONSE_MAX_BYTES - marker_overhead)
    if maximum < 0:
        return None
    head_bytes = text[:maximum].encode("utf-8")
    tail_bytes = text[-maximum:].encode("utf-8") if maximum else b""

    def preview_at(prefix_end: int, suffix_start: int) -> str | None:
        if not 0 <= prefix_end <= suffix_start <= len(text):
            return None
        prefix = text[:prefix_end]
        suffix = text[suffix_start:]
        prefix_size = len(prefix.encode("utf-8"))
        omitted = utf8_bytes - prefix_size - len(suffix.encode("utf-8"))
        if omitted <= 0:
            return None
        return prefix + _truncation_only_marker(omitted) + "\n" + suffix

    def fits(candidate: str | None) -> bool:
        if candidate is None:
            return False
        return _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )

    best = None
    prefix_end, suffix_start = 0, len(text)
    low, high = 0, maximum
    while low <= high:
        retained = (low + high) // 2
        prefix = head_bytes[: retained // 2].decode("utf-8", errors="ignore")
        suffix_size = retained - len(prefix.encode("utf-8"))
        suffix = tail_bytes[-suffix_size:].decode("utf-8", errors="ignore") if suffix_size else ""
        ends = (len(prefix), len(text) - len(suffix))
        candidate = preview_at(*ends)
        if fits(candidate):
            best, (prefix_end, suffix_start) = candidate, ends
            low = retained + 1
        else:
            high = retained - 1
    if best is None:
        return None
    for _ in range(8):
        grew = False
        for ends in ((prefix_end + 1, suffix_start), (prefix_end, suffix_start - 1)):
            candidate = preview_at(*ends)
            if fits(candidate):
                best, (prefix_end, suffix_start) = candidate, ends
                grew = True
        if not grew:
            break
    return best


class RefRunStore:
    """Run-local catalog of externalized refs (tool texts + folded history)."""

    def __init__(self) -> None:
        self._entries: dict[str, RefCatalogEntry] = {}
        self._history_line_counts: dict[str, int] = {}
        self.preview_cache: dict[tuple[str, int, int], tuple[Any, str]] = {}

    def catalog_tuple(self) -> tuple[RefCatalogEntry, ...]:
        return tuple(self._entries.values())

    def intern_tool_text(self, text: str, entry: RefCatalogEntry) -> str:
        parsed = parse_ref(entry.manifest.ref)
        key = parsed.value if parsed is not None else entry.manifest.ref
        self._entries.setdefault(key, entry)
        return entry.manifest.ref

    def add_history_records(self, records: tuple[str, ...]) -> str:
        digest = hashlib.sha256()
        utf8_bytes = 0
        for index, record in enumerate(records):
            encoded = record.encode("utf-8")
            if index > 0:
                digest.update(b"\n")
                utf8_bytes += 1
            digest.update(encoded)
            utf8_bytes += len(encoded)
        text_hash = digest.hexdigest()
        ref = f"history:{text_hash}"
        self._history_line_counts[text_hash] = len(records)
        if text_hash not in self._entries:
            self._entries[text_hash] = RefCatalogEntry(
                manifest=RefManifest(ref=ref, utf8_bytes=utf8_bytes, sha256=text_hash),
                source=JsonlHistorySourceHandle(
                    records=records,
                    utf8_bytes=utf8_bytes,
                    sha256=text_hash,
                    line_count=len(records),
                ),
            )
        return ref

    def history_manifest_payload(self, ref: str) -> dict[str, Any] | None:
        requested = parse_ref(ref)
        if requested is None:
            return None
        entry = self._entries.get(requested.value)
        if entry is None or entry.manifest.utf8_bytes is None:
            return None
        return {
            "bytes": entry.manifest.utf8_bytes,
            "kind": "history",
            "lines": self._history_line_counts.get(requested.value),
            "ref": ref,
            "version": 1,
        }


def build_ref_reader(
    store: RefRunStore,
    *,
    threshold_tokens: int,
    encoder: Any = None,
) -> Callable[[str], Awaitable[str]]:
    """Bind a reader tool callable to a run-local ref store."""

    async def reader(command: str = "") -> str:
        """Inspect one run-local externalized ref with bounded virtual reader commands.

        Follow the next command in truncation markers to read omitted spans across pages.

        :param command: Use ls, stat, wc, head, tail, sed -n, grep, or cat and optional bounded pipelines.
        """
        try:
            if not isinstance(command, str):
                return REF_EXEC_USAGE_ERROR
            stages = _parse_ref_exec_command(command)
            catalog = store.catalog_tuple()
            cancelled = threading.Event()
            resolved_encoder = encoder
            if resolved_encoder is None and stages[-1].command != "wc":
                # tiktoken.get_encoding may download BPE files with no
                # timeout, so resolution must not run on the event loop;
                # it also must not run inside the sync worker, because a
                # cancelled call would then join-wait the download.
                resolved_encoder, _ = await asyncio.to_thread(
                    _ref_exec_get_tiktoken_encoder
                )
            try:
                worker = asyncio.create_task(
                    asyncio.to_thread(
                        _execute_ref_reader_sync,
                        stages,
                        catalog,
                        threshold_tokens=threshold_tokens,
                        encoder=resolved_encoder,
                        cancelled=cancelled,
                    )
                )
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled.set()
                try:
                    await asyncio.shield(worker)
                except Exception:
                    pass
                raise
        except RefExecError as exc:
            return str(exc)
        except Exception:
            LOG.exception("Unexpected externalized ref reader failure")
            return "Error: externalized ref reader is unavailable"

    reader.__name__ = REF_EXEC_TOOL_NAME
    return reader

# --- inlined from src/owui_ext/shared/tool_execution.py (owui_ext.shared.tool_execution) ---
import ast
import json
import logging
import uuid
from typing import Any, Callable, Optional
from fastapi import Request
_tool_execution_log = logging.getLogger("owui_ext.shared.tool_execution")
_core_process_tool_result = None


CITATION_TOOLS: set[str] = {
    "search_web",
    "view_file",
    "view_knowledge_file",
    "query_chat_files",
    "query_knowledge_files",
    "fetch_url",
}


TERMINAL_EVENT_TOOLS: set[str] = {
    "display_file",
    "write_file",
    "replace_file_content",
    "run_command",
}


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _normalize_user(user: Any) -> Any:
    if user is None or hasattr(user, "id"):
        return user
    if isinstance(user, dict):
        try:
            from open_webui.models.users import UserModel

            return UserModel(**user)
        except Exception:
            from types import SimpleNamespace

            return SimpleNamespace(**user)
    return user


async def process_tool_result(
    *,
    tool_function_name: str = "tool",
    tool_type: str,
    tool_result: Any,
    direct_tool: bool = False,
    request: Optional[Request] = None,
    metadata: Optional[dict] = None,
    user: Any = None,
) -> tuple[Any, list, list]:
    """Process tool result into (payload, files, embeds) using core."""
    global _core_process_tool_result
    if _core_process_tool_result is None:
        try:
            from open_webui.utils.middleware import process_tool_result as fn
        except ImportError as exc:
            raise RuntimeError(
                "Open WebUI process_tool_result helper is required"
            ) from exc
        if not callable(fn):
            raise RuntimeError("Open WebUI process_tool_result helper is not callable")
        _core_process_tool_result = fn

    return await _maybe_await(
        _core_process_tool_result(
            request,
            tool_function_name,
            tool_result,
            tool_type,
            direct_tool=direct_tool,
            metadata=metadata if isinstance(metadata, dict) else {},
            user=_normalize_user(user),
        )
    )


def structure_terminal_file_tool_result(
    tool_function_name: str,
    tool_function_params: dict,
    tool_result: Any,
    tool: dict,
    metadata: Optional[dict],
) -> Any:
    """Apply Core's structured ``display_file`` result when available."""
    from open_webui.utils import middleware as middleware_utils

    builder = getattr(middleware_utils, "build_terminal_file_tool_result", None)
    if not callable(builder):
        return tool_result
    structured_result = builder(
        tool_function_name,
        tool_function_params,
        tool_result,
        tool,
        metadata,
    )
    return tool_result if structured_result is None else structured_result


async def execute_direct_tool_call(
    *,
    tool_function_name: str,
    tool_function_params: dict,
    tool: dict,
    extra_params: dict,
) -> Any:
    """Execute direct tools through ``__event_call__`` like core middleware."""
    event_call = extra_params.get("__event_call__")
    if not callable(event_call):
        raise RuntimeError("Direct tool execution requires __event_call__ context")
    metadata = extra_params.get("__metadata__")
    session_id = metadata.get("session_id") if isinstance(metadata, dict) else None
    return await event_call(
        {
            "type": "execute:tool",
            "data": {
                "id": str(uuid.uuid4()),
                "name": tool_function_name,
                "params": tool_function_params,
                "server": tool.get("server", {}),
                "session_id": session_id,
            },
        }
    )


def normalize_terminal_tools_result(
    *, terminal_tools_result: Any, extra_params: Optional[dict]
) -> dict:
    """Normalize get_terminal_tools() return value across Open WebUI versions."""
    terminal_system_prompt = None
    terminal_tools = terminal_tools_result

    if (
        isinstance(terminal_tools_result, tuple)
        and len(terminal_tools_result) == 2
        and isinstance(terminal_tools_result[0], dict)
    ):
        terminal_tools = terminal_tools_result[0]
        if isinstance(terminal_tools_result[1], str):
            stripped_prompt = terminal_tools_result[1].strip()
            if stripped_prompt:
                terminal_system_prompt = stripped_prompt

    if isinstance(extra_params, dict):
        if terminal_system_prompt:
            extra_params["__terminal_system_prompt__"] = terminal_system_prompt
        else:
            extra_params.pop("__terminal_system_prompt__", None)

    if isinstance(terminal_tools, dict):
        return terminal_tools
    return {}


async def emit_terminal_tool_event(
    *,
    tool_function_name: str,
    tool_function_params: dict,
    tool_result: Any,
    event_emitter: Optional[Callable],
) -> None:
    """Emit ``terminal:*`` UI events for Open Terminal tool results.

    Recognises only the names listed in ``TERMINAL_EVENT_TOOLS``
    (display_file / write_file / replace_file_content / run_command);
    unknown names fall through silently.
    """
    if not event_emitter:
        return
    if tool_function_name == "display_file":
        path = (
            tool_function_params.get("path", "")
            if isinstance(tool_function_params, dict)
            else ""
        )
        if not isinstance(path, str) or not path:
            return
        parsed = tool_result
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                parsed = tool_result
        if isinstance(parsed, dict) and parsed.get("exists") is False:
            return
        page = tool_function_params.get("page")
        # NOTE: nested calls cannot produce Core's top-level structured
        # output pair, so inline requests open the viewer instead.
        event = {
            "type": "terminal:display_file",
            "data": {
                "path": path,
                **({"page": page} if page else {}),
            },
        }
    elif tool_function_name in {"write_file", "replace_file_content"}:
        path = (
            tool_function_params.get("path", "")
            if isinstance(tool_function_params, dict)
            else ""
        )
        if not isinstance(path, str) or not path:
            return
        event = {
            "type": f"terminal:{tool_function_name}",
            "data": {"path": path},
        }
    elif tool_function_name == "run_command":
        event = {"type": "terminal:run_command", "data": {}}
    else:
        return
    try:
        await event_emitter(event)
    except Exception as exc:
        _tool_execution_log.warning(
            f"Error emitting terminal event for {tool_function_name}: {exc}"
        )


async def execute_tool_call(
    tool_call: dict,
    tools_dict: dict,
    extra_params: dict,
    event_emitter: Optional[Callable] = None,
) -> dict:
    """Execute a single tool call and return ``{tool_call_id, content}``."""
    if not isinstance(tool_call, dict):
        return {
            "tool_call_id": str(uuid.uuid4()),
            "content": f"Malformed tool_call: expected dict, got {type(tool_call).__name__}",
        }
    tool_call_id = tool_call.get("id", str(uuid.uuid4()))
    func = tool_call.get("function")
    if not isinstance(func, dict):
        return {
            "tool_call_id": tool_call_id,
            "content": f"Malformed tool_call: 'function' is {type(func).__name__}, not dict",
        }
    tool_function_name = func.get("name", "")
    tool_args_raw = func.get("arguments", "{}")

    tool_function_params: dict = {}
    if isinstance(tool_args_raw, dict):
        tool_function_params = tool_args_raw
    elif isinstance(tool_args_raw, str):
        try:
            tool_function_params = ast.literal_eval(tool_args_raw)
        except Exception:
            try:
                tool_function_params = json.loads(tool_args_raw)
            except Exception as exc:
                _tool_execution_log.error(
                    f"Error parsing tool call arguments: {tool_args_raw} - {exc}"
                )
                return {
                    "tool_call_id": tool_call_id,
                    "content": f"Error parsing arguments: {exc}",
                }
    if not isinstance(tool_function_params, dict):
        tool_function_params = {}

    tool_result: Any = None
    tool_result_files: list[dict] = []
    tool_result_embeds: list[Any] = []
    emit_terminal_event = False
    if tool_function_name in tools_dict:
        tool = tools_dict[tool_function_name]
        spec = tool.get("spec", {})
        direct_tool = bool(tool.get("direct", False))

        try:
            allowed_params = spec.get("parameters", {}).get("properties", {}).keys()
            tool_function_params = {
                k: v for k, v in tool_function_params.items() if k in allowed_params
            }

            if direct_tool:
                tool_result = await execute_direct_tool_call(
                    tool_function_name=tool_function_name,
                    tool_function_params=tool_function_params,
                    tool=tool,
                    extra_params=extra_params,
                )
            else:
                tool_function = tool["callable"]

                # Only override per-call dynamic context — preserve __user__
                # so tool-specific UserValves injected by get_tools() survive.
                from open_webui.utils.tools import get_updated_tool_function

                tool_function = await _maybe_await(get_updated_tool_function(
                    function=tool_function,
                    extra_params={
                        "__messages__": extra_params.get("__messages__", []),
                        "__files__": extra_params.get("__files__", []),
                        "__event_emitter__": extra_params.get("__event_emitter__"),
                        "__event_call__": extra_params.get("__event_call__"),
                    },
                ))

                tool_result = await tool_function(**tool_function_params)

            tool_type = tool.get("type", "")
            tool_result = structure_terminal_file_tool_result(
                tool_function_name,
                tool_function_params,
                tool_result,
                tool,
                extra_params.get("__metadata__"),
            )
            tool_result, tool_result_files, tool_result_embeds = await process_tool_result(
                tool_function_name=tool_function_name,
                tool_type=tool_type,
                tool_result=tool_result,
                direct_tool=direct_tool,
                request=extra_params.get("__request__"),
                metadata=extra_params.get("__metadata__"),
                user=extra_params.get("__user__"),
            )
            emit_terminal_event = True

        except Exception as exc:
            _tool_execution_log.exception(
                f"Error executing tool {tool_function_name}: {exc}"
            )
            tool_result = f"Error: {exc}"
    else:
        tool_result = f"Tool '{tool_function_name}' not found"

    if emit_terminal_event:
        await emit_terminal_tool_event(
            tool_function_name=tool_function_name,
            tool_function_params=tool_function_params,
            tool_result=tool_result,
            event_emitter=event_emitter,
        )
        if event_emitter and tool_result_files:
            await event_emitter({"type": "files", "data": {"files": tool_result_files}})
        if event_emitter and tool_result_embeds:
            await event_emitter({"type": "embeds", "data": {"embeds": tool_result_embeds}})

    if tool_result is None:
        tool_result = ""
    elif not isinstance(tool_result, str):
        try:
            tool_result = json.dumps(tool_result, ensure_ascii=False, default=str)
        except Exception:
            tool_result = str(tool_result)

    if event_emitter and tool_result and tool_function_name in CITATION_TOOLS:
        try:
            from open_webui.utils.middleware import get_citation_source_from_tool_result

            tool_id = tools_dict.get(tool_function_name, {}).get("tool_id", "")
            citation_sources = get_citation_source_from_tool_result(
                tool_name=tool_function_name,
                tool_params=tool_function_params,
                tool_result=tool_result,
                tool_id=tool_id,
            )
            for source in citation_sources:
                await event_emitter({"type": "source", "data": source})
        except Exception as exc:
            _tool_execution_log.warning(
                f"Error extracting citation sources from {tool_function_name}: {exc}"
            )

    return {
        "tool_call_id": tool_call_id,
        "content": tool_result,
    }

# --- inlined from src/owui_ext/shared/mcp_tools.py (owui_ext.shared.mcp_tools) ---
import asyncio
import logging
import re
from typing import Any, Callable, Optional
from fastapi import Request
_mcp_tools_log = logging.getLogger("owui_ext.shared.mcp_tools")


async def _mcp_maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _emit_warning_notification(
    event_emitter: Optional[Callable], content: str
) -> None:
    if not callable(event_emitter):
        return
    if not isinstance(content, str) or not content.strip():
        return
    try:
        await event_emitter(
            {
                "type": "notification",
                "data": {"type": "warning", "content": content.strip()},
            }
        )
    except Exception as exc:
        _mcp_tools_log.debug(f"Error emitting MCP warning notification: {exc}")


async def _build_mcp_headers_with_core(
    *,
    connection: dict,
    request: Request,
    user: Any,
    server_id: str,
    metadata: dict,
    extra_params: dict,
) -> Optional[dict[str, Any]]:
    try:
        from open_webui.utils.tools import build_tool_server_headers
    except ImportError:
        return None

    result = await _mcp_maybe_await(
        build_tool_server_headers(
            connection,
            request,
            user,
            server_id=server_id,
            metadata=metadata,
            extra_params=extra_params,
        )
    )
    headers = result[0] if isinstance(result, tuple) else result
    if not isinstance(headers, dict):
        # Only a missing helper is a legacy Open WebUI compatibility case.
        # Once the core helper exists, exceptions or contract violations point
        # to a v0.9.6+ auth/header bug that should fail visibly instead of
        # being masked by the older hand-built header path.
        raise TypeError(
            "open_webui.utils.tools.build_tool_server_headers() returned "
            f"non-dict headers: {type(headers).__name__}"
        )
    return dict(headers)


async def _build_mcp_headers_legacy(
    *,
    connection: dict,
    request: Request,
    user: Any,
    server_id: str,
    metadata: dict,
    extra_params: dict,
) -> dict[str, Any]:
    from open_webui.utils.headers import include_user_info_headers
    from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS

    try:
        from open_webui.env import (
            FORWARD_SESSION_INFO_HEADER_CHAT_ID,
            FORWARD_SESSION_INFO_HEADER_MESSAGE_ID,
        )
    except ImportError:
        FORWARD_SESSION_INFO_HEADER_CHAT_ID = None
        FORWARD_SESSION_INFO_HEADER_MESSAGE_ID = None

    auth_type = connection.get("auth_type", "")
    headers: dict[str, Any] = {}
    if auth_type == "bearer":
        headers["Authorization"] = f'Bearer {connection.get("key", "")}'
    elif auth_type == "none":
        pass
    elif auth_type == "session":
        token = getattr(getattr(request.state, "token", None), "credentials", "")
        headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "system_oauth":
        oauth_token = extra_params.get("__oauth_token__", None)
        if oauth_token:
            headers["Authorization"] = f'Bearer {oauth_token.get("access_token", "")}'
    elif auth_type in ("oauth_2.1", "oauth_2.1_static"):
        # Open WebUI core (utils/middleware.py) looks up OAuth tokens under
        # the colon-trailing segment of ``server_id`` so that ``host:port``
        # style ids resolve to the key the UI stored. Mirror that for the
        # lookup only -- keep the full ``server_id`` for the ``mcp_clients``
        # cache key, otherwise two servers sharing a trailing segment would
        # collide and the earlier client would be overwritten without cleanup.
        try:
            splits = server_id.split(":")
            oauth_lookup_id = splits[-1] if len(splits) > 1 else server_id

            oauth_token = await request.app.state.oauth_client_manager.get_oauth_token(
                user.id, f"mcp:{oauth_lookup_id}"
            )

            if oauth_token:
                headers["Authorization"] = f'Bearer {oauth_token.get("access_token", "")}'
        except Exception as e:
            _mcp_tools_log.error(
                f"Error getting OAuth token for MCP server {server_id}: {e}"
            )

    connection_headers = connection.get("headers", None)
    if connection_headers and isinstance(connection_headers, dict):
        headers.update(connection_headers)

    if ENABLE_FORWARD_USER_INFO_HEADERS and user:
        headers = include_user_info_headers(headers, user)
        if FORWARD_SESSION_INFO_HEADER_CHAT_ID and metadata.get("chat_id"):
            headers[FORWARD_SESSION_INFO_HEADER_CHAT_ID] = metadata["chat_id"]
        if FORWARD_SESSION_INFO_HEADER_MESSAGE_ID and metadata.get("message_id"):
            headers[FORWARD_SESSION_INFO_HEADER_MESSAGE_ID] = metadata["message_id"]

    return headers


async def _get_tool_server_connections(request: Request) -> list[dict]:
    try:
        from open_webui.models.config import Config
    except ImportError:
        Config = None

    if Config is not None:
        try:
            connections = await _mcp_maybe_await(
                Config.get("tool_server.connections", None)
            )
            if connections is not None:
                return connections if isinstance(connections, list) else []
        except Exception as exc:
            _mcp_tools_log.debug(
                f"Could not read tool_server.connections from Config: {exc}"
            )

    legacy_connections = (
        getattr(getattr(request.app.state, "config", None), "TOOL_SERVER_CONNECTIONS", [])
        or []
    )
    return legacy_connections if isinstance(legacy_connections, list) else []


async def resolve_mcp_tools(
    request: Request,
    user: Any,
    mcp_tool_ids: list[str],
    extra_params: dict,
    metadata: dict,
    debug: bool = False,
) -> tuple[dict, dict]:
    """Resolve MCP ``server:mcp:`` tool IDs into tool callables and live clients.

    Returns ``(mcp_tools_dict, mcp_clients)``. ``mcp_tools_dict`` maps
    prefixed tool names to ``{"spec", "callable", "type": "mcp", "direct"}``
    entries compatible with the shared tool execution path.
    ``mcp_clients`` maps server IDs to the live ``MCPClient`` instances
    so the caller can ``cleanup_mcp_clients`` them when tool execution
    is complete.
    """
    try:
        from open_webui.utils.mcp.client import MCPClient
    except ImportError:
        if debug and mcp_tool_ids:
            _mcp_tools_log.info(
                "MCPClient unavailable; skipping MCP tool resolution"
            )
        return {}, {}

    from open_webui.utils.misc import is_string_allowed

    try:
        from open_webui.utils.access_control import has_connection_access
    except ImportError:
        from open_webui.utils.tools import (
            has_tool_server_access as has_connection_access,
        )

    event_emitter = (extra_params or {}).get("__event_emitter__")

    async def emit_warning(description: str) -> None:
        await _emit_warning_notification(event_emitter, description)

    metadata = metadata or {}
    extra_params = extra_params or {}
    mcp_tools_dict: dict[str, dict] = {}
    mcp_clients: dict[str, Any] = {}
    server_connections = await _get_tool_server_connections(request)

    ordered_server_ids: list[str] = []
    seen_server_ids: set[str] = set()
    for tool_id in mcp_tool_ids:
        if not isinstance(tool_id, str) or not tool_id.startswith("server:mcp:"):
            continue
        server_id = tool_id[len("server:mcp:") :].strip()
        if not server_id:
            continue
        if server_id not in seen_server_ids:
            seen_server_ids.add(server_id)
            ordered_server_ids.append(server_id)

    for server_id in ordered_server_ids:
        client = None
        try:
            mcp_server_connection = next(
                (
                    server_connection
                    for server_connection in server_connections
                    if server_connection.get("type", "") == "mcp"
                    and server_connection.get("info", {}).get("id") == server_id
                ),
                None,
            )

            if not mcp_server_connection:
                _mcp_tools_log.warning(f"MCP server with id {server_id} not found")
                await emit_warning(f"MCP server '{server_id}' was not found")
                continue

            if not mcp_server_connection.get("config", {}).get("enable", True):
                if debug:
                    _mcp_tools_log.info(
                        f"MCP server {server_id} is disabled; skipping"
                    )
                await emit_warning(f"MCP server '{server_id}' is disabled")
                continue

            try:
                has_access = await _mcp_maybe_await(
                    has_connection_access(user, mcp_server_connection)
                )
            except TypeError:
                has_access = await _mcp_maybe_await(
                    has_connection_access(user, mcp_server_connection, None)
                )

            if not has_access:
                _mcp_tools_log.warning(
                    f"Access denied to MCP server {server_id} for user {user.id}"
                )
                await emit_warning(f"Access denied to MCP server '{server_id}'")
                continue

            headers = await _build_mcp_headers_with_core(
                connection=mcp_server_connection,
                request=request,
                user=user,
                server_id=server_id,
                metadata=metadata,
                extra_params=extra_params,
            )
            if headers is None:
                headers = await _build_mcp_headers_legacy(
                    connection=mcp_server_connection,
                    request=request,
                    user=user,
                    server_id=server_id,
                    metadata=metadata,
                    extra_params=extra_params,
                )

            function_name_filter_list = mcp_server_connection.get("config", {}).get(
                "function_name_filter_list", ""
            )
            if isinstance(function_name_filter_list, str):
                function_name_filter_list = [
                    item.strip()
                    for item in function_name_filter_list.split(",")
                    if item.strip()
                ]

            client = MCPClient()
            client_lock = asyncio.Lock()
            setattr(client, "_sub_agent_lock", client_lock)

            await client.connect(
                url=mcp_server_connection.get("url", ""),
                headers=headers if headers else None,
            )

            tool_specs = await client.list_tool_specs() or []

            def make_tool_function(
                mcp_client: Any,
                function_name: str,
                lock: asyncio.Lock,
            ) -> Callable[..., Any]:
                async def tool_function(**kwargs):
                    async with lock:
                        return await mcp_client.call_tool(
                            function_name,
                            function_args=kwargs,
                        )

                return tool_function

            loaded_tool_count = 0
            for tool_spec in tool_specs:
                if not isinstance(tool_spec, dict):
                    continue

                tool_name = tool_spec.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    continue

                if function_name_filter_list and not is_string_allowed(
                    tool_name, function_name_filter_list
                ):
                    continue

                safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", server_id)
                prefixed_name = f"{safe_prefix}_{tool_name}"
                mcp_tools_dict[prefixed_name] = {
                    "spec": {
                        **tool_spec,
                        "name": prefixed_name,
                    },
                    "callable": make_tool_function(client, tool_name, client_lock),
                    "type": "mcp",
                    "direct": False,
                }
                loaded_tool_count += 1

            mcp_clients[server_id] = client

            if debug:
                _mcp_tools_log.info(
                    f"Loaded {loaded_tool_count} MCP tools from server {server_id}"
                )
        except Exception as e:
            _mcp_tools_log.warning(
                f"Failed to load MCP tools from {server_id}: {e}"
            )
            if client is not None:
                try:
                    await client.disconnect()
                except BaseException:
                    pass
            await emit_warning(f"Could not load MCP tools from '{server_id}': {e}")

    return mcp_tools_dict, mcp_clients


async def cleanup_mcp_clients(mcp_clients: dict) -> None:
    """Disconnect all MCP clients, absorbing non-Exception failures.

    Some callers open MCP clients inside child tasks (e.g. coroutines
    fed to ``asyncio.gather``) and close them from an outer ``finally``.
    Catching ``BaseException`` keeps anyio cancel-scope failures (raised
    outside the ``Exception`` hierarchy on cross-task cleanup) and
    ``asyncio.CancelledError`` from escaping the caller and discarding
    the tool's response. Matches upstream Open WebUI main.py chat
    handler MCP cleanup (#24105).
    """
    for client in reversed(list((mcp_clients or {}).values())):
        try:
            await client.disconnect()
        except BaseException as e:
            _mcp_tools_log.debug(f"Error cleaning up MCP client: {e}")

# --- inlined from src/owui_ext/shared/tool_servers.py (owui_ext.shared.tool_servers) ---
import json
import logging
from collections.abc import Mapping
from typing import Any, Optional
from fastapi import Request
_tool_servers_log = logging.getLogger("owui_ext.shared.tool_servers")


def normalize_direct_tool_servers(value: Any) -> list[dict]:
    """Normalize direct tool server payload into a list of dict copies."""
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def extract_direct_tool_server_prompts(direct_tools: Mapping[str, dict]) -> list[str]:
    """Collect unique non-empty system prompts from loaded direct tools only."""
    prompts: list[str] = []
    seen_prompts: set[str] = set()
    for tool in direct_tools.values():
        if not isinstance(tool, dict):
            continue
        server = tool.get("server")
        if not isinstance(server, dict):
            continue
        system_prompt = server.get("system_prompt")
        if isinstance(system_prompt, str):
            stripped_prompt = system_prompt.strip()
            if stripped_prompt and stripped_prompt not in seen_prompts:
                prompts.append(stripped_prompt)
                seen_prompts.add(stripped_prompt)
    return prompts


async def resolve_direct_tool_servers_from_request_and_metadata(
    *,
    request: Optional[Request],
    metadata: Optional[dict],
    debug: bool = False,
) -> list[dict]:
    """Resolve direct tool servers using core-gated metadata as source of truth."""
    metadata_has_tool_servers = isinstance(metadata, dict) and "tool_servers" in metadata
    if metadata_has_tool_servers:
        return normalize_direct_tool_servers(metadata.get("tool_servers"))

    request_servers: list[dict] = []
    if request is not None:
        request_body = getattr(request, "body", None)
        if callable(request_body):
            try:
                raw_body = await request_body()
                if raw_body:
                    body = json.loads(raw_body)
                    if isinstance(body, dict):
                        request_servers = normalize_direct_tool_servers(
                            body.get("tool_servers")
                        )
                        if not request_servers:
                            nested_metadata = body.get("metadata")
                            if isinstance(nested_metadata, dict):
                                request_servers = normalize_direct_tool_servers(
                                    nested_metadata.get("tool_servers")
                                )
            except Exception:
                request_servers = []
    if request_servers:
        return request_servers
    return []


def build_direct_tools_dict(
    *, tool_servers: list[dict], debug: bool = False
) -> dict:
    """Build direct tool entries compatible with Open WebUI middleware."""
    direct_tools: dict = {}
    for server in tool_servers:
        if not isinstance(server, dict):
            continue
        specs = server.get("specs", [])
        if not isinstance(specs, list) or not specs:
            continue
        server_payload = {k: v for k, v in server.items() if k != "specs"}
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not isinstance(name, str) or not name:
                continue
            direct_tools[name] = {
                "spec": spec,
                "direct": True,
                "server": server_payload,
                "type": "direct",
            }
    if debug and tool_servers and not direct_tools:
        _tool_servers_log.info("No direct tools loaded from tool_servers")
    return direct_tools


async def resolve_terminal_id_from_request_and_metadata(
    *,
    request: Optional[Request],
    metadata: Optional[dict],
    debug: bool = False,
) -> str:
    """Resolve ``terminal_id`` preferring the request body over metadata.

    Open WebUI puts the active terminal binding in the request body
    (top-level ``terminal_id`` or nested ``metadata.terminal_id``) and
    in the inlet ``metadata`` dict the plugin is invoked with. The
    request body is the source of truth -- metadata can be stale when
    the user just switched terminals -- so the body wins when both are
    present.
    """

    def _normalize(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    metadata_terminal_id = ""
    if isinstance(metadata, dict):
        metadata_terminal_id = _normalize(metadata.get("terminal_id"))

    request_terminal_id = ""
    if request is not None:
        request_body = getattr(request, "body", None)
        if callable(request_body):
            try:
                raw_body = await request_body()
                if raw_body:
                    body = json.loads(raw_body)
                    if isinstance(body, dict):
                        request_terminal_id = _normalize(body.get("terminal_id"))
                        if not request_terminal_id:
                            nested_metadata = body.get("metadata")
                            if isinstance(nested_metadata, dict):
                                request_terminal_id = _normalize(
                                    nested_metadata.get("terminal_id")
                                )
            except Exception:
                request_terminal_id = ""

    if request_terminal_id:
        if debug and metadata_terminal_id and metadata_terminal_id != request_terminal_id:
            _tool_servers_log.warning(
                "terminal_id mismatch between request body and metadata; "
                "using request body terminal_id"
            )
        return request_terminal_id

    return metadata_terminal_id

# --- inlined from src/owui_ext/shared/tool_loader.py (owui_ext.shared.tool_loader) ---
async def build_tools_dict(
    request,
    model,
    metadata,
    user,
    valves,
    extra_params,
    tool_id_list,
    excluded_tool_ids,
    resolved_terminal_id=None,
    resolved_direct_tool_servers=None,
):
    """Assemble a tools_dict from regular, MCP, terminal, direct, and
    builtin sources.

    Returns ``(tools_dict, mcp_clients)``. Caller must call
    ``shared.mcp_tools.cleanup_mcp_clients(mcp_clients)`` once tool
    execution is done so MCP connections don't leak. ``mcp_clients``
    is empty when no ``server:mcp:`` tool IDs are present.

    ``resolved_terminal_id`` / ``resolved_direct_tool_servers`` are
    optional pre-resolved values: when omitted, the helper resolves
    them from ``request.body()`` / ``metadata`` itself. Pre-resolving
    avoids re-reading ``request.body()`` when the caller already did
    so for its own bookkeeping.
    """
    import inspect
    import logging

    log = logging.getLogger("owui_ext.shared.tool_loader")
    debug = bool(getattr(valves, "DEBUG", False))

    from open_webui.utils.tools import get_builtin_tools, get_tools

    try:
        from open_webui.utils.tools import (
            get_attached_knowledge as core_get_attached_knowledge,
        )
    except ImportError:
        core_get_attached_knowledge = None

    try:
        from open_webui.utils.tools import get_terminal_tools
    except Exception:
        get_terminal_tools = None

    metadata = metadata or {}
    extra_params = extra_params or {}
    model = model or {}
    tools_dict: dict = {}
    extra_metadata = extra_params.get("__metadata__")
    event_emitter = extra_params.get("__event_emitter__")

    if resolved_terminal_id is None:
        terminal_id = await resolve_terminal_id_from_request_and_metadata(
            request=request,
            metadata=metadata,
            debug=debug,
        )
    elif isinstance(resolved_terminal_id, str):
        terminal_id = resolved_terminal_id.strip()
    else:
        terminal_id = ""

    if terminal_id:
        metadata["terminal_id"] = terminal_id
        extra_metadata = extra_params.get("__metadata__")
        if isinstance(extra_metadata, dict):
            extra_metadata["terminal_id"] = terminal_id
        else:
            extra_params["__metadata__"] = metadata
            extra_metadata = metadata

    if resolved_direct_tool_servers is None:
        direct_tool_servers = await resolve_direct_tool_servers_from_request_and_metadata(
            request=request,
            metadata=metadata,
            debug=debug,
        )
    else:
        direct_tool_servers = normalize_direct_tool_servers(resolved_direct_tool_servers)

    if direct_tool_servers:
        metadata["tool_servers"] = direct_tool_servers
        if isinstance(extra_metadata, dict):
            extra_metadata["tool_servers"] = direct_tool_servers
        else:
            extra_params["__metadata__"] = metadata
            extra_metadata = metadata

    # Open WebUI's get_tools() silently skips ``server:mcp:`` entries, so
    # split them out and resolve via resolve_mcp_tools().
    regular_tool_ids = [tid for tid in tool_id_list if not tid.startswith("builtin:")]
    if excluded_tool_ids:
        regular_tool_ids = [tid for tid in regular_tool_ids if tid not in excluded_tool_ids]

    mcp_tool_ids = [tid for tid in regular_tool_ids if tid.startswith("server:mcp:")]
    non_mcp_tool_ids = [
        tid for tid in regular_tool_ids if not tid.startswith("server:mcp:")
    ]

    if debug:
        log.info(f"Regular tool IDs: {regular_tool_ids}")
        if mcp_tool_ids:
            log.info(f"MCP tool IDs: {mcp_tool_ids}")
        if non_mcp_tool_ids != regular_tool_ids:
            log.info(f"Non-MCP regular tool IDs: {non_mcp_tool_ids}")

    mcp_clients: dict = {}

    if non_mcp_tool_ids:
        try:
            tools_dict = await get_tools(
                request=request,
                tool_ids=non_mcp_tool_ids,
                user=user,
                extra_params=extra_params,
            )
            if debug:
                log.info(f"Loaded {len(tools_dict)} regular tools")
        except Exception as e:
            log.exception(f"Error loading tools: {e}")
            await emit_notification(
                event_emitter,
                level="warning",
                content=f"Could not load tools: {e}",
            )

    if mcp_tool_ids:
        try:
            mcp_tools, mcp_clients = await resolve_mcp_tools(
                request=request,
                user=user,
                mcp_tool_ids=mcp_tool_ids,
                extra_params=extra_params,
                metadata=metadata,
                debug=debug,
            )
            if mcp_tools:
                duplicate_names = set(tools_dict.keys()) & set(mcp_tools.keys())
                tools_dict.update(mcp_tools)
                if debug:
                    if duplicate_names:
                        log.warning(
                            "MCP tools overrode existing tool names: "
                            f"{sorted(duplicate_names)}"
                        )
                    log.info(f"Loaded {len(mcp_tools)} MCP tools")
        except Exception as e:
            log.exception(f"Error loading MCP tools: {e}")
            await emit_notification(
                event_emitter,
                level="warning",
                content=f"Could not load MCP tools: {e}",
            )

    if terminal_id and bool(getattr(valves, "ENABLE_TERMINAL_TOOLS", True)):
        if get_terminal_tools is None:
            if debug:
                log.info("get_terminal_tools is unavailable in this Open WebUI version")
        else:
            try:
                terminal_tools_result = await get_terminal_tools(
                    request=request,
                    terminal_id=terminal_id,
                    user=user,
                    extra_params=extra_params,
                )
                terminal_tools = normalize_terminal_tools_result(
                    terminal_tools_result=terminal_tools_result,
                    extra_params=extra_params,
                )
                if terminal_tools:
                    duplicate_names = set(tools_dict.keys()) & set(terminal_tools.keys())
                    tools_dict = {**tools_dict, **terminal_tools}
                    if debug:
                        if duplicate_names:
                            log.warning(
                                "Terminal tools overrode existing tool names: "
                                f"{sorted(duplicate_names)}"
                            )
                        log.info(
                            f"Loaded {len(terminal_tools)} terminal tools for terminal_id={terminal_id}"
                        )
            except Exception as e:
                log.exception(f"Error loading terminal tools: {e}")
                await emit_notification(
                    event_emitter,
                    level="warning",
                    content=f"Could not load terminal tools: {e}",
                )
    elif terminal_id and debug:
        log.info("Terminal tools disabled by ENABLE_TERMINAL_TOOLS valve")

    if direct_tool_servers:
        try:
            direct_tools = build_direct_tools_dict(
                tool_servers=direct_tool_servers,
                debug=debug,
            )
            if direct_tools:
                duplicate_names = set(tools_dict.keys()) & set(direct_tools.keys())
                tools_dict = {**tools_dict, **direct_tools}
                direct_tool_server_prompts = extract_direct_tool_server_prompts(direct_tools)
                if direct_tool_server_prompts:
                    extra_params["__direct_tool_server_system_prompts__"] = direct_tool_server_prompts
                else:
                    extra_params.pop("__direct_tool_server_system_prompts__", None)
                if debug:
                    if duplicate_names:
                        log.warning(
                            "Direct tools overrode existing tool names: "
                            f"{sorted(duplicate_names)}"
                        )
                    log.info(f"Loaded {len(direct_tools)} direct tools")
            else:
                extra_params.pop("__direct_tool_server_system_prompts__", None)
        except Exception as e:
            log.exception(f"Error loading direct tools: {e}")
            extra_params.pop("__direct_tool_server_system_prompts__", None)
            await emit_notification(
                event_emitter,
                level="warning",
                content=f"Could not load direct tools: {e}",
            )
    else:
        extra_params.pop("__direct_tool_server_system_prompts__", None)

    try:
        features = metadata.get("features", {})

        # NOTE: view_skill is NOT registered here; the plugin registers it
        # manually via shared.skills.register_view_skill() when the parent
        # conversation's <available_skills> manifest is detected
        # (model-attached skills).
        builtin_extra_params = {
            "__user__": extra_params.get("__user__"),
            "__event_emitter__": extra_params.get("__event_emitter__"),
            "__event_call__": extra_params.get("__event_call__"),
            "__metadata__": extra_params.get("__metadata__"),
            "__chat_id__": extra_params.get("__chat_id__"),
            "__message_id__": extra_params.get("__message_id__"),
            "__oauth_token__": extra_params.get("__oauth_token__"),
        }

        builtin_kwargs = {
            "request": request,
            "extra_params": builtin_extra_params,
            "features": features,
            "model": model,
        }
        try:
            supports_note_chat = (
                "is_note_chat" in inspect.signature(get_builtin_tools).parameters
            )
        except (TypeError, ValueError):
            supports_note_chat = False
        if supports_note_chat:
            from open_webui.models.chats import Chats
            from open_webui.utils.chat_id import is_saved_chat_id

            chat_id = metadata.get("chat_id")
            chat = (
                await maybe_await(Chats.get_chat_by_id(chat_id))
                if is_saved_chat_id(chat_id)
                else None
            )
            builtin_kwargs["is_note_chat"] = bool(
                chat
                and (chat.meta or {}).get("internal") is True
                and (chat.meta or {}).get("type") == "note"
            )

        all_builtin_tools = await maybe_await(get_builtin_tools(**builtin_kwargs))

        # NOTE: ask_user is excluded from nested loops. The callable itself
        # would work over __event_call__, but the frontend keeps a single
        # event callback, so concurrent request:user_input calls from
        # parallel branches clobber each other and the losing call waits
        # forever (Core overrides sio.call's 60s default timeout with
        # WEBSOCKET_EVENT_CALLER_TIMEOUT, which defaults to None).
        disabled_builtin_tools: set = set(
            BUILTIN_TOOL_CATEGORIES.get("user_input", set())
        )
        for valve_field, category in VALVE_TO_CATEGORY.items():
            if not getattr(valves, valve_field, True):
                disabled_builtin_tools.update(BUILTIN_TOOL_CATEGORIES.get(category, set()))

        knowledge_tools_enabled = bool(getattr(valves, "ENABLE_KNOWLEDGE_TOOLS", True))
        file_tools_enabled = bool(getattr(valves, "ENABLE_FILE_TOOLS", True))
        notes_tools_enabled = bool(getattr(valves, "ENABLE_NOTES_TOOLS", True))
        knowledge_metadata = (
            metadata
            if core_get_attached_knowledge is not None
            else {"folder_knowledge": metadata.get("folder_knowledge")}
        )
        keep_view_note_for_knowledge = (
            (not notes_tools_enabled)
            and knowledge_tools_enabled
            and model_knowledge_tools_enabled(model)
            and model_has_note_knowledge(model, knowledge_metadata)
        )
        keep_view_file = (
            file_tools_enabled and "list_chat_files" in all_builtin_tools
        ) or (
            knowledge_tools_enabled
            and model_knowledge_tools_enabled(model)
            and "kb_exec" not in all_builtin_tools
            and model_has_file_knowledge(model, knowledge_metadata)
        )

        # Regular tools take priority over builtin tools with the same name.
        builtin_count = 0
        for name, tool_dict in all_builtin_tools.items():
            if name in disabled_builtin_tools and not (
                (name == "view_note" and keep_view_note_for_knowledge)
                or (name == "view_file" and keep_view_file)
            ):
                continue
            if name not in tools_dict:
                tools_dict[name] = tool_dict
                builtin_count += 1
            elif debug:
                log.warning(
                    f"Builtin tool '{name}' skipped: "
                    "regular tool with same name takes priority"
                )

        if debug:
            log.info(
                f"Loaded {builtin_count} builtin tools "
                f"(disabled categories: {[c for v, c in VALVE_TO_CATEGORY.items() if not getattr(valves, v, True)]}). "
                f"Total tools: {len(tools_dict)}"
            )
    except Exception as e:
        log.exception(f"Error loading builtin tools: {e}")
        await emit_notification(
            event_emitter,
            level="warning",
            content=f"Could not load builtin tools: {e}",
        )

    return tools_dict, mcp_clients

# --- inlined from src/owui_ext/shared/skills.py (owui_ext.shared.skills) ---
import logging
import re
from typing import Any, Optional
from fastapi import Request
_skills_log = logging.getLogger("owui_ext.shared.skills")

_SKILLS_MANIFEST_START = "<available_skills>"
_SKILLS_MANIFEST_END = "</available_skills>"
_SKILL_TAG_PATTERN = re.compile(
    r"<skill name=.*?>\n.*?\n</skill>", re.DOTALL
)


async def _skills_maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _find_manifest_in_text(text: str) -> str:
    """Return the <available_skills>…</available_skills> substring, or ""."""
    start = text.find(_SKILLS_MANIFEST_START)
    if start == -1:
        return ""
    end = text.find(_SKILLS_MANIFEST_END, start)
    if end == -1:
        return ""
    return text[start : end + len(_SKILLS_MANIFEST_END)]


def _find_skill_tags_in_text(text: str) -> list[str]:
    """Return all ``<skill name="...">…</skill>`` blocks found in *text*."""
    return _SKILL_TAG_PATTERN.findall(text)


def _extract_from_system_messages(
    messages: Optional[list],
    extractor,
):
    """Walk system messages and apply *extractor* to each text chunk.

    ``extractor`` is called with a single ``str`` argument and should return a
    list of results (or a single truthy result).  The function handles both
    plain-string content and list-of-parts content
    (``[{"type": "text", "text": "..."}]``).
    """
    results: list = []
    if not messages:
        return results
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            found = extractor(content)
            if found:
                results.append(found) if isinstance(found, str) else results.extend(
                    found
                )
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    found = extractor(part.get("text") or "")
                    if found:
                        results.append(found) if isinstance(
                            found, str
                        ) else results.extend(found)
    return results


def extract_skill_manifest(messages: Optional[list]) -> str:
    """Extract the ``<available_skills>`` manifest from the parent
    conversation's system messages.

    Since v0.8.2, only **model-attached** skills appear in this manifest.
    User-selected skills are injected as full ``<skill>`` tags instead
    (see :func:`extract_user_skill_tags`).

    Args:
        messages: The parent conversation messages (``__messages__``).

    Returns:
        The manifest XML string, or empty string if not found.
    """
    results = _extract_from_system_messages(messages, _find_manifest_in_text)
    return results[0] if results else ""


def extract_user_skill_tags(messages: Optional[list]) -> list[str]:
    """Extract ``<skill name="...">content</skill>`` tags from the parent
    conversation's system messages.

    Since Open WebUI v0.8.2, user-selected skills are injected as individual
    ``<skill>`` tags with full content (as opposed to the lazy-loading
    manifest used for model-attached skills).

    Args:
        messages: The parent conversation messages (``__messages__``).

    Returns:
        A list of ``<skill …>…</skill>`` strings, possibly empty.
    """
    return _extract_from_system_messages(messages, _find_skill_tags_in_text)


async def register_view_skill(
    tools_dict: dict,
    request: Request,
    extra_params: dict,
) -> None:
    """Manually register the view_skill builtin tool in tools_dict.

    This is needed for **model-attached** skills whose content is not injected
    inline.  The agent loop can call ``view_skill`` to lazily load their
    content from the ``<available_skills>`` manifest.

    Since v0.8.2, user-selected skills are injected as full ``<skill>`` tags
    and do NOT require ``view_skill``; they are passed directly in the system
    message.

    Args:
        tools_dict: The tools dict to add view_skill to (modified in-place).
        request: FastAPI request object.
        extra_params: Extra parameters for tool binding.
    """
    if "view_skill" in tools_dict:
        return

    try:
        from open_webui.tools.builtin import view_skill
        from open_webui.utils.tools import (
            get_async_tool_function_and_apply_extra_params,
            convert_function_to_pydantic_model,
            convert_pydantic_model_to_openai_function_spec,
        )

        callable_fn = await _skills_maybe_await(get_async_tool_function_and_apply_extra_params(
            view_skill,
            {
                "__request__": request,
                "__user__": extra_params.get("__user__", {}),
                "__event_emitter__": extra_params.get("__event_emitter__"),
                "__event_call__": extra_params.get("__event_call__"),
                "__metadata__": extra_params.get("__metadata__"),
                "__chat_id__": extra_params.get("__chat_id__"),
                "__message_id__": extra_params.get("__message_id__"),
            },
        ))

        pydantic_model = convert_function_to_pydantic_model(view_skill)
        spec = convert_pydantic_model_to_openai_function_spec(pydantic_model)

        tools_dict["view_skill"] = {
            "tool_id": "builtin:view_skill",
            "callable": callable_fn,
            "spec": spec,
            "type": "builtin",
        }
    except Exception as e:
        _skills_log.warning(f"Failed to register view_skill: {e}")

# --- inlined from src/owui_ext/shared/valves.py (owui_ext.shared.valves) ---
from typing import Any, Type
from pydantic import BaseModel
def coerce_user_valves(raw_valves: Any, valves_cls: Type[BaseModel]) -> BaseModel:
    """Normalize raw user valves into the target valves class.

    Open WebUI hands ``raw_valves`` over from filter context, where it can
    arrive as the target class itself, a different ``BaseModel`` subclass
    (when the user-valve schema has drifted between plugin versions), a raw
    dict, or anything else. Always return a fresh ``valves_cls`` instance so
    callers can rely on the field set being current.
    """
    if isinstance(raw_valves, valves_cls):
        return raw_valves
    if isinstance(raw_valves, BaseModel):
        try:
            data = raw_valves.model_dump()
        except Exception:
            data = {}
        return valves_cls.model_validate(data)
    if isinstance(raw_valves, dict):
        return valves_cls.model_validate(raw_valves)
    return valves_cls.model_validate({})

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
                "use the sub-agent's model, which helps preserve prompt caching where possible."
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
