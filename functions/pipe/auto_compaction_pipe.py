"""
title: Auto Compaction Pipe
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Manifold Pipe that wraps Open WebUI models, compacts long chats, and persists durable checkpoint summaries.
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import asyncio
import copy
import fnmatch
import hashlib
import html
import json
import time
from contextlib import suppress
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable

from fastapi import HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    Integer,
    JSON,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse


PIPE_FUNCTION_ID = "auto_compact"
CHECKPOINT_NAMESPACE = "skyzi000.open_webui_extensions.auto_compaction_pipe"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_TABLE_NAME = "skyzi000_owui_ext_autocompact_checkpoint_v1"
CHECKPOINT_LOOKUP_UQ = "skyzi000_owui_ext_accp_v1_lookup_uq"
CHECKPOINT_PREFIX_IDX = "skyzi000_owui_ext_accp_v1_prefix_idx"
CHECKPOINT_RECENT_IDX = "skyzi000_owui_ext_accp_v1_recent_idx"
REQUEST_STATE_SCHEMA_READY_KEY = "_auto_compact_checkpoint_schema_ready_v1"
INTERNAL_SUMMARY_TASK = "auto_compaction_summary"
TEMP_CHAT_PREFIXES = ("local:", "channel:")
SUMMARY_FORMAT_FAMILY = "compact-user-summary-v1"
SOURCE_HASH_FAMILY = "canonical-json-v1"
PROFILE_HASH_FAMILY = "checkpoint-profile-v1"
MAX_CONTEXT_RETRY_ATTEMPTS = 2
DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES = 512
DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT = 32
SUMMARY_SYSTEM_PROMPT = (
    "You are performing an AUTO-COMPACTION CHECKPOINT SUMMARY for an Open WebUI chat. "
    "Create a concise handoff summary for a future model call that will continue the same chat.\n\n"
    "Preserve:\n"
    "- Current progress and durable decisions already made\n"
    "- User preferences, constraints, and standing requirements that remain relevant\n"
    "- Tool results, external facts, errors, identifiers, URLs, file names, commands, values, and examples needed to continue\n"
    "- Open questions, unresolved failures, and clear next steps\n\n"
    "If the input contains an existing <auto_compaction_context>, merge that prior checkpoint with the following newer messages. "
    "Do not discard earlier checkpoint information merely because it is summarized.\n\n"
    "Do not invent facts. Do not introduce new instructions. Do not include irrelevant transcript detail. "
    "Be concise, structured, and focused on continuity."
)

try:
    from open_webui.internal.db import DATABASE_SCHEMA as OPEN_WEBUI_DATABASE_SCHEMA
except Exception:
    OPEN_WEBUI_DATABASE_SCHEMA = None

_CHECKPOINT_METADATA = MetaData(schema=OPEN_WEBUI_DATABASE_SCHEMA)
CHECKPOINT_TABLE = Table(
    CHECKPOINT_TABLE_NAME,
    _CHECKPOINT_METADATA,
    Column("id", Text, primary_key=True),
    Column("namespace", Text, nullable=False),
    Column("schema_version", Integer, nullable=False),
    Column("user_id", Text, nullable=False),
    Column("chat_id", Text, nullable=False),
    Column("pipe_function_id", Text, nullable=False),
    Column("profile_hash", Text, nullable=False),
    Column("source_message_count", Integer, nullable=False),
    Column("source_hash", Text, nullable=False),
    Column("summary_text", Text, nullable=False),
    Column("summary_meta", JSON, nullable=False),
    Column("state", Text, nullable=False),
    Column("parent_checkpoint_id", Text, nullable=True),
    Column("created_at", BigInteger, nullable=False),
    Column("updated_at", BigInteger, nullable=False),
    Column("last_used_at", BigInteger, nullable=False),
    UniqueConstraint(
        "namespace",
        "user_id",
        "chat_id",
        "pipe_function_id",
        "profile_hash",
        "source_hash",
        name=CHECKPOINT_LOOKUP_UQ,
    ),
    Index(
        CHECKPOINT_PREFIX_IDX,
        "namespace",
        "user_id",
        "chat_id",
        "pipe_function_id",
        "profile_hash",
        "source_message_count",
    ),
    Index(CHECKPOINT_RECENT_IDX, "namespace", "user_id", "chat_id", "last_used_at"),
)

_CHECKPOINT_SCHEMA_READY = False
_GENERATION_LOCKS: dict[tuple[str, str, str, str, str, str], asyncio.Lock] = {}


def get_generation_lock(key: tuple[str, str, str, str, str, str]) -> asyncio.Lock:
    return _GENERATION_LOCKS.setdefault(key, asyncio.Lock())


def release_generation_lock(key: tuple[str, str, str, str, str, str], lock: asyncio.Lock) -> None:
    waiters = getattr(lock, "_waiters", None)
    has_waiters = bool(waiters)
    if not lock.locked() and not has_waiters and _GENERATION_LOCKS.get(key) is lock:
        _GENERATION_LOCKS.pop(key, None)


@dataclass(frozen=True)
class WrapperIdentity:
    pipe_function_id: str
    target_model_suffix: str
    target_model_id: str


@dataclass(frozen=True)
class ReusableCheckpointMatch:
    kind: str
    source_message_count: int


@dataclass(frozen=True)
class TargetModelContract:
    id: str
    name: str
    meta: dict[str, Any]
    wrapper_params: dict[str, Any]
    access_grants: list[dict[str, Any]]
    owner_user_id: str | None


@dataclass(frozen=True)
class MessageCut:
    preserved_system_message: dict[str, Any] | None
    summarization_prefix: list[dict[str, Any]]
    tail_messages: list[dict[str, Any]]
    source_message_count: int


@dataclass(frozen=True)
class RetryToolResultCut:
    active_user_message: dict[str, Any]
    tool_round_messages: list[dict[str, Any]]
    tool_messages_to_summarize: list[dict[str, Any]]


@dataclass(frozen=True)
class ToolResultCompactionCut:
    active_user_message: dict[str, Any]
    tool_round_messages: list[dict[str, Any]]
    tool_messages_to_summarize: list[dict[str, Any]]


class RetryableContextOverflow(Exception):
    """Raised only before any user-visible target output has been emitted."""


class UnsupportedCompactionInput(Exception):
    """Raised when v1 cannot safely compact without rewriting active input."""

    def __init__(self, message: str, *, code: str = "unsafe_compaction_input"):
        super().__init__(message)
        self.code = code


class ParentCheckpointExtensionFailed(Exception):
    """Raised when a verified parent checkpoint exists but extension summarization fails."""

    def __init__(self, parent: dict[str, Any], original: Exception):
        super().__init__(str(original))
        self.parent = parent
        self.original = original


def _json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode_target_model_id(target_model_id: str) -> str:
    if not isinstance(target_model_id, str) or not target_model_id:
        raise ValueError("target_model_id must be a non-empty string")
    return target_model_id


def build_wrapper_model_id(pipe_function_id: str, target_model_id: str) -> str:
    return f"{pipe_function_id}.{encode_target_model_id(target_model_id)}"


def decode_wrapper_model_id(wrapper_model_id: str, *, expected_pipe_function_id: str = PIPE_FUNCTION_ID) -> WrapperIdentity:
    if not isinstance(wrapper_model_id, str) or "." not in wrapper_model_id:
        raise ValueError("Malformed wrapper id: expected '<pipe_function_id>.<target_model_id>'")
    pipe_function_id, target_model_id = wrapper_model_id.split(".", 1)
    if expected_pipe_function_id and pipe_function_id != expected_pipe_function_id:
        raise ValueError(f"Unexpected wrapper pipe function id: expected {expected_pipe_function_id!r}, got {pipe_function_id!r}")
    if not target_model_id:
        raise ValueError("Malformed wrapper id: missing target model id")
    return WrapperIdentity(
        pipe_function_id=pipe_function_id,
        target_model_suffix=target_model_id,
        target_model_id=target_model_id,
    )


def is_generated_wrapper_model_id(model_id: Any, *, pipe_function_id: str = PIPE_FUNCTION_ID) -> bool:
    return isinstance(model_id, str) and model_id.startswith(f"{pipe_function_id}.")


_TOP_LEVEL_MESSAGE_DROP_KEYS = {
    "id",
    "parentId",
    "childrenIds",
    "timestamp",
    "created_at",
    "updated_at",
    "usage",
    "info",
    "done",
    "status",
    "statusHistory",
    "status_history",
    "error",
}
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
_FILE_CONTENT_PART_TYPES = {
    "file",
    "input_file",
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
}


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
            if is_file_part and key == "file":
                item = _canonicalize_embedded_file_value(value[key])
            else:
                item = _canonicalize_general_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    return _canonicalize_general_value(value)


def _canonicalize_content_value(value: Any) -> Any:
    if isinstance(value, list):
        out = []
        for item in value:
            canonical_item = _canonicalize_content_part(item)
            if not _is_empty_canonical_value(canonical_item):
                out.append(canonical_item)
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


def _canonicalize_message_value(key: str, value: Any) -> Any:
    if key == "content":
        return _canonicalize_content_value(value)
    if key == "files":
        return _canonicalize_files_value(value)
    if key == "sources":
        return _canonicalize_sources_value(value)
    return _canonicalize_general_value(value)


def canonicalize_message_for_source_hash(message: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key in sorted(message.keys()):
        if key in _TOP_LEVEL_MESSAGE_DROP_KEYS:
            continue
        if key not in _STABLE_MESSAGE_KEYS:
            continue
        value = _canonicalize_message_value(key, message[key])
        if _is_empty_canonical_value(value):
            continue
        canonical[key] = value
    canonical.setdefault("role", message.get("role", "assistant"))
    canonical.setdefault("content", "")
    return canonical


def canonicalize_messages_for_source_hash(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [canonicalize_message_for_source_hash(message) for message in messages if isinstance(message, dict)]


def compute_source_hash(messages: list[dict[str, Any]]) -> str:
    return _json_hash(
        {
            "family": SOURCE_HASH_FAMILY,
            "messages": canonicalize_messages_for_source_hash(messages),
        }
    )


def compute_prefix_hashes(messages: list[dict[str, Any]]) -> dict[int, str]:
    return {count: compute_source_hash(messages[:count]) for count in range(1, len(messages) + 1)}


def compute_profile_hash(
    *,
    schema_family: str = CHECKPOINT_TABLE_NAME,
    summary_format_family: str = SUMMARY_FORMAT_FAMILY,
    source_hash_family: str = SOURCE_HASH_FAMILY,
    **_: Any,
) -> str:
    return _json_hash(
        {
            "profile": PROFILE_HASH_FAMILY,
            "schema_family": schema_family,
            "summary_format_family": summary_format_family,
            "source_hash_family": source_hash_family,
        }
    )


def _first_system_index(messages: list[dict[str, Any]]) -> int | None:
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            return index
    return None


def _tool_call_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                ids.add(call["id"])
    return ids


def _assistant_tool_call_ids(messages: Iterable[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            ids.update(_tool_call_ids(message))
    return ids


def _remove_orphan_tool_messages(tail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assistant_ids = _assistant_tool_call_ids(tail)
    cleaned: list[dict[str, Any]] = []
    for message in tail:
        if message.get("role") == "tool" and message.get("tool_call_id") not in assistant_ids:
            continue
        cleaned.append(message)
    while cleaned and cleaned[0].get("role") == "tool":
        cleaned.pop(0)
    return cleaned


def select_safe_message_cut(
    messages: list[dict[str, Any]],
) -> MessageCut:
    if not messages:
        return MessageCut(None, [], [], 0)

    source_messages = [copy.deepcopy(message) for message in messages]
    system_index = _first_system_index(source_messages)
    preserved_system = copy.deepcopy(source_messages[system_index]) if system_index is not None else None
    working = [message for index, message in enumerate(source_messages) if index != system_index]

    if not working:
        return MessageCut(preserved_system, [], [], 0)

    latest_user_index = next(
        (index for index in range(len(working) - 1, -1, -1) if working[index].get("role") == "user"),
        len(working) - 1,
    )
    tail = copy.deepcopy(working[latest_user_index:])
    if not any(message.get("role") == "user" for message in tail):
        tail = [copy.deepcopy(working[latest_user_index])]

    prefix = copy.deepcopy(working[:latest_user_index])
    return MessageCut(
        preserved_system_message=preserved_system,
        summarization_prefix=prefix,
        tail_messages=tail,
        source_message_count=len(prefix),
    )


def select_retry_tool_result_cut(messages: list[dict[str, Any]]) -> RetryToolResultCut | None:
    cut = select_tool_result_compaction_cut(messages)
    if cut is None:
        return None
    return RetryToolResultCut(
        active_user_message=cut.active_user_message,
        tool_round_messages=cut.tool_round_messages,
        tool_messages_to_summarize=cut.tool_messages_to_summarize,
    )


def select_tool_result_compaction_cut(
    messages: list[dict[str, Any]],
) -> ToolResultCompactionCut | None:
    latest_user_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"),
        None,
    )
    if latest_user_index is None or latest_user_index >= len(messages) - 1:
        return None

    tool_round = copy.deepcopy(messages[latest_user_index + 1 :])
    assistant_rounds: list[tuple[set[str], list[dict[str, Any]]]] = []
    all_assistant_ids: set[str] = set()
    for message in tool_round:
        if message.get("role") != "assistant":
            continue
        ids = _tool_call_ids(message)
        if not ids:
            continue
        round_tool_messages = [
            copy.deepcopy(candidate)
            for candidate in tool_round
            if candidate.get("role") == "tool" and candidate.get("tool_call_id") in ids
        ]
        seen = {candidate.get("tool_call_id") for candidate in round_tool_messages}
        if ids <= seen:
            assistant_rounds.append((ids, round_tool_messages))
            all_assistant_ids.update(ids)

    if not assistant_rounds:
        return None

    for message in tool_round:
        if message.get("role") == "tool" and message.get("tool_call_id") not in all_assistant_ids:
            return None

    tool_messages: list[dict[str, Any]] = []
    for _, round_tool_messages in assistant_rounds:
        tool_messages.extend(round_tool_messages)
    if not tool_messages:
        return None

    return ToolResultCompactionCut(
        active_user_message=copy.deepcopy(messages[latest_user_index]),
        tool_round_messages=tool_round,
        tool_messages_to_summarize=tool_messages,
    )


def _message_content_as_excerpt_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def truncate_text_middle_by_utf8_bytes(text: str, max_bytes: int) -> str:
    if not isinstance(text, str):
        text = str(text)
    budget = max(0, int(max_bytes or 0))
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    if budget <= 0:
        return ""

    marker = "...[middle omitted]..."
    marker_bytes = marker.encode("utf-8")
    if budget <= len(marker_bytes):
        return marker_bytes[:budget].decode("utf-8", errors="ignore")

    remaining = budget - len(marker_bytes)
    prefix_budget = remaining // 2
    suffix_budget = remaining - prefix_budget
    prefix = encoded[:prefix_budget].decode("utf-8", errors="ignore")
    suffix = encoded[-suffix_budget:].decode("utf-8", errors="ignore") if suffix_budget > 0 else ""
    result = f"{prefix}{marker}{suffix}"
    while len(result.encode("utf-8")) > budget and suffix:
        suffix = suffix[1:]
        result = f"{prefix}{marker}{suffix}"
    return result


def _xml_attr(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _xml_cdata(value: Any) -> str:
    text = str(value)
    return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def render_historical_user_message_excerpts(
    source_messages: list[dict[str, Any]],
    *,
    excerpt_bytes: int,
    max_messages: int,
) -> str:
    count_limit = int(max_messages or 0)
    if count_limit <= 0:
        return ""

    excerpts: list[str] = []
    for message in source_messages:
        if message.get("role") != "user":
            continue
        text = _message_content_as_excerpt_text(message).strip()
        if not text:
            continue
        excerpts.append(truncate_text_middle_by_utf8_bytes(text, excerpt_bytes))
    excerpts = excerpts[-count_limit:]
    if not excerpts:
        return ""

    lines = [
        (
            f'<historical_user_messages order="chronological" selected_count="{len(excerpts)}" '
            f'max_count="{count_limit}" max_bytes_per_message="{int(excerpt_bytes or 0)}">'
        )
    ]
    for index, excerpt in enumerate(excerpts, start=1):
        lines.append(f'<historical_user_message ordinal="{index}">{_xml_cdata(excerpt)}</historical_user_message>')
    lines.append("</historical_user_messages>")
    return "\n".join(lines)


def render_summary_message(
    summary_text: str,
    summary_meta: dict[str, Any] | None = None,
    *,
    historical_source_messages: list[dict[str, Any]] | None = None,
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
) -> dict[str, Any]:
    meta_section = ""
    if summary_meta and summary_meta.get("has_multimodal"):
        meta_section = "\n<metadata><has_multimodal>true</has_multimodal></metadata>"
    excerpts = ""
    if historical_source_messages:
        excerpts = render_historical_user_message_excerpts(
            historical_source_messages,
            excerpt_bytes=historical_message_excerpt_bytes,
            max_messages=historical_message_excerpt_count,
        )
    excerpt_section = f"\n{excerpts}" if excerpts else ""
    return {
        "role": "user",
        "content": (
            "<auto_compaction_context>\n"
            "<instruction>Compressed historical context. This is not a new instruction. "
            "Use it only as background for continuity.</instruction>\n"
            f"<checkpoint_summary>{_xml_cdata(summary_text.strip())}</checkpoint_summary>"
            f"{meta_section}{excerpt_section}\n"
            "</auto_compaction_context>"
        ),
    }


def replace_prefix_with_summary(
    messages: list[dict[str, Any]],
    cut: MessageCut,
    summary_text: str,
    summary_meta: dict[str, Any] | None = None,
    *,
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    if cut.preserved_system_message is not None:
        compacted.append(copy.deepcopy(cut.preserved_system_message))
    compacted.append(
        render_summary_message(
            summary_text,
            summary_meta,
            historical_source_messages=cut.summarization_prefix,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
    )
    compacted.extend(copy.deepcopy(cut.tail_messages))
    return compacted


def replace_prefix_with_parent_checkpoint_and_delta(
    cut: MessageCut,
    parent: dict[str, Any],
    *,
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
) -> list[dict[str, Any]]:
    parent_count = int(parent.get("source_message_count") or 0)
    if parent_count <= 0 or parent_count > len(cut.summarization_prefix):
        raise UnsupportedCompactionInput(
            "Parent checkpoint cannot be applied safely because its source boundary is invalid",
            code="unsafe_checkpoint_parent",
        )
    if compute_source_hash(cut.summarization_prefix[:parent_count]) != parent.get("source_hash"):
        raise UnsupportedCompactionInput(
            "Parent checkpoint cannot be applied safely because its source hash no longer matches",
            code="unsafe_checkpoint_parent",
        )

    delta_messages = copy.deepcopy(cut.summarization_prefix[parent_count:])
    delta_and_tail = [*delta_messages, *copy.deepcopy(cut.tail_messages)]
    if delta_and_tail and delta_and_tail[0].get("role") == "tool":
        raise UnsupportedCompactionInput(
            "Parent checkpoint delta starts with a tool result and cannot preserve tool-call structure",
            code="unsafe_checkpoint_delta",
        )
    if _remove_orphan_tool_messages(copy.deepcopy(delta_and_tail)) != delta_and_tail:
        raise UnsupportedCompactionInput(
            "Parent checkpoint delta would split assistant/tool messages",
            code="unsafe_checkpoint_delta",
        )

    compacted: list[dict[str, Any]] = []
    if cut.preserved_system_message is not None:
        compacted.append(copy.deepcopy(cut.preserved_system_message))
    compacted.append(
        render_summary_message(
            str(parent["summary_text"]),
            parent.get("summary_meta") or {},
            historical_source_messages=cut.summarization_prefix,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
    )
    compacted.extend(delta_messages)
    compacted.extend(copy.deepcopy(cut.tail_messages))
    return compacted


def build_checkpoint_id(
    *,
    namespace: str,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    profile_hash: str,
    source_hash: str,
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            [namespace, user_id, chat_id, pipe_function_id, profile_hash, source_hash],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"accp_{digest}"


def build_checkpoint_row(
    *,
    namespace: str,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    profile_hash: str,
    source_hash: str,
    source_message_count: int,
    summary_text: str,
    summary_meta: dict[str, Any] | None,
    parent_checkpoint_id: str | None,
    now: int | None = None,
) -> dict[str, Any]:
    timestamp = int(time.time()) if now is None else int(now)
    return {
        "id": build_checkpoint_id(
            namespace=namespace,
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            profile_hash=profile_hash,
            source_hash=source_hash,
        ),
        "namespace": namespace,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "user_id": user_id,
        "chat_id": chat_id,
        "pipe_function_id": pipe_function_id,
        "profile_hash": profile_hash,
        "source_message_count": int(source_message_count),
        "source_hash": source_hash,
        "summary_text": summary_text,
        "summary_meta": summary_meta or {},
        "state": "ready",
        "parent_checkpoint_id": parent_checkpoint_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "last_used_at": timestamp,
    }


def select_ready_checkpoint(rows: Iterable[dict[str, Any]], *, source_hash: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("state") == "ready" and row.get("source_hash") == source_hash:
            return row
    return None


def select_longest_matching_parent(
    rows: Iterable[dict[str, Any]],
    prefix_hashes: dict[int, str],
) -> dict[str, Any] | None:
    candidates = sorted(rows, key=lambda row: int(row.get("source_message_count") or 0), reverse=True)
    for row in candidates:
        if row.get("state") != "ready":
            continue
        count = int(row.get("source_message_count") or 0)
        if prefix_hashes.get(count) == row.get("source_hash"):
            return row
    return None


async def ensure_checkpoint_table_initialized(
    *,
    request: Any = None,
    async_engine: AsyncEngine | Any | None = None,
) -> None:
    global _CHECKPOINT_SCHEMA_READY

    state = getattr(request, "state", None)
    if state is not None and getattr(state, REQUEST_STATE_SCHEMA_READY_KEY, False):
        return
    if _CHECKPOINT_SCHEMA_READY:
        if state is not None:
            setattr(state, REQUEST_STATE_SCHEMA_READY_KEY, True)
        return

    if async_engine is None:
        from open_webui.internal.db import async_engine as open_webui_async_engine

        async_engine = open_webui_async_engine

    async with async_engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: CHECKPOINT_TABLE.create(sync_conn, checkfirst=True))

    _CHECKPOINT_SCHEMA_READY = True
    if state is not None:
        setattr(state, REQUEST_STATE_SCHEMA_READY_KEY, True)


class CheckpointStore:
    def __init__(self, *, db: AsyncSession | None = None):
        self.db = db

    async def _context(self):
        if self.db is not None:
            class ExistingSessionContext:
                async def __aenter__(self_nonlocal):
                    return self.db

                async def __aexit__(self_nonlocal, exc_type, exc, tb):
                    return False

            return ExistingSessionContext()

        from open_webui.internal.db import get_async_db

        return get_async_db()

    async def lookup_ready(
        self,
        *,
        namespace: str,
        user_id: str,
        chat_id: str,
        pipe_function_id: str,
        profile_hash: str,
        source_hash: str,
    ) -> dict[str, Any] | None:
        async with await self._context() as db:
            result = await db.execute(
                select(CHECKPOINT_TABLE).where(
                    CHECKPOINT_TABLE.c.namespace == namespace,
                    CHECKPOINT_TABLE.c.user_id == user_id,
                    CHECKPOINT_TABLE.c.chat_id == chat_id,
                    CHECKPOINT_TABLE.c.pipe_function_id == pipe_function_id,
                    CHECKPOINT_TABLE.c.profile_hash == profile_hash,
                    CHECKPOINT_TABLE.c.source_hash == source_hash,
                    CHECKPOINT_TABLE.c.state == "ready",
                )
            )
            row = result.mappings().first()
            return dict(row) if row else None

    async def find_longest_parent(
        self,
        *,
        namespace: str,
        user_id: str,
        chat_id: str,
        pipe_function_id: str,
        profile_hash: str,
        prefix_hashes: dict[int, str],
    ) -> dict[str, Any] | None:
        if not prefix_hashes:
            return None
        async with await self._context() as db:
            result = await db.execute(
                select(CHECKPOINT_TABLE)
                .where(
                    CHECKPOINT_TABLE.c.namespace == namespace,
                    CHECKPOINT_TABLE.c.user_id == user_id,
                    CHECKPOINT_TABLE.c.chat_id == chat_id,
                    CHECKPOINT_TABLE.c.pipe_function_id == pipe_function_id,
                    CHECKPOINT_TABLE.c.profile_hash == profile_hash,
                    CHECKPOINT_TABLE.c.state == "ready",
                    CHECKPOINT_TABLE.c.source_message_count.in_(list(prefix_hashes.keys())),
                )
                .order_by(CHECKPOINT_TABLE.c.source_message_count.desc())
            )
            rows = [dict(row) for row in result.mappings().all()]
            return select_longest_matching_parent(rows, prefix_hashes)

    async def insert_ready(self, row: dict[str, Any]) -> dict[str, Any] | None:
        async with await self._context() as db:
            try:
                await db.execute(insert(CHECKPOINT_TABLE).values(**row))
                await db.commit()
                return row
            except IntegrityError:
                await db.rollback()
                return await self.lookup_ready(
                    namespace=row["namespace"],
                    user_id=row["user_id"],
                    chat_id=row["chat_id"],
                    pipe_function_id=row["pipe_function_id"],
                    profile_hash=row["profile_hash"],
                    source_hash=row["source_hash"],
                )

    async def touch(self, checkpoint_id: str, *, now: int | None = None) -> bool:
        timestamp = int(time.time()) if now is None else int(now)
        async with await self._context() as db:
            try:
                await db.execute(
                    update(CHECKPOINT_TABLE)
                    .where(CHECKPOINT_TABLE.c.id == checkpoint_id)
                    .values(last_used_at=timestamp, updated_at=timestamp)
                )
                await db.commit()
                return True
            except Exception:
                await db.rollback()
                return False


def _split_patterns(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").replace("\n", ",").split(",") if part.strip()]


def _matches_any_pattern(model: dict[str, Any], patterns: list[str]) -> bool:
    model_id = str(model.get("id") or "")
    name = str(model.get("name") or "")
    return any(fnmatch.fnmatchcase(model_id, pattern) or fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _is_arena_model(model: dict[str, Any]) -> bool:
    return bool(model.get("arena")) or model.get("owned_by") == "arena"


def _model_id(model: dict[str, Any]) -> str | None:
    model_id = model.get("id")
    return model_id if isinstance(model_id, str) and model_id else None


def _model_base_model_id(model: dict[str, Any]) -> str | None:
    base_model_id = model.get("base_model_id")
    if isinstance(base_model_id, str) and base_model_id:
        return base_model_id

    info = model.get("info")
    if isinstance(info, BaseModel):
        info = info.model_dump()
    elif not isinstance(info, dict):
        model_dump = getattr(info, "model_dump", None)
        if callable(model_dump):
            with suppress(Exception):
                info = model_dump()
        elif hasattr(info, "__dict__"):
            info = vars(info)

    if isinstance(info, dict):
        base_model_id = info.get("base_model_id")
        if isinstance(base_model_id, str) and base_model_id:
            return base_model_id
    return None


def _is_based_on_generated_wrapper(model: dict[str, Any], *, pipe_function_id: str = PIPE_FUNCTION_ID) -> bool:
    return is_generated_wrapper_model_id(_model_base_model_id(model), pipe_function_id=pipe_function_id)


def filter_target_models(models: Iterable[dict[str, Any]], valves: Any, *, pipe_function_id: str = PIPE_FUNCTION_ID):
    include_patterns = _split_patterns(getattr(valves, "include_model_patterns", ""))
    exclude_patterns = _split_patterns(getattr(valves, "exclude_model_patterns", ""))
    seen: set[str] = set()
    targets: list[dict[str, Any]] = []

    for raw_model in models:
        if not isinstance(raw_model, dict):
            continue
        model = copy.deepcopy(raw_model)
        model_id = _model_id(model)
        if model_id is None or model_id in seen:
            continue
        if is_generated_wrapper_model_id(model_id, pipe_function_id=pipe_function_id):
            continue
        if _is_based_on_generated_wrapper(model, pipe_function_id=pipe_function_id):
            continue
        if _is_arena_model(model):
            continue
        if include_patterns and not _matches_any_pattern(model, include_patterns):
            continue
        if exclude_patterns and _matches_any_pattern(model, exclude_patterns):
            continue
        seen.add(model_id)
        targets.append(model)
    return targets


def _grant_to_dict(grant: Any) -> dict[str, Any] | None:
    if isinstance(grant, BaseModel):
        grant = grant.model_dump()
    elif not isinstance(grant, dict):
        grant = {
            "id": getattr(grant, "id", None),
            "principal_type": getattr(grant, "principal_type", None),
            "principal_id": getattr(grant, "principal_id", None),
            "permission": getattr(grant, "permission", None),
        }
    principal_type = grant.get("principal_type")
    principal_id = grant.get("principal_id")
    permission = grant.get("permission")
    if principal_type not in ("user", "group") or permission not in ("read", "write"):
        return None
    if not isinstance(principal_id, str) or not principal_id:
        return None
    out = {
        "principal_type": principal_type,
        "principal_id": principal_id,
        "permission": permission,
    }
    if isinstance(grant.get("id"), str) and grant["id"]:
        out["id"] = grant["id"]
    return out


def _dump_model_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, list):
        return copy.deepcopy(value)
    if hasattr(value, "__dict__"):
        return {key: _dump_model_value(item) for key, item in vars(value).items() if not key.startswith("_")}
    return value


def _payload_dict(value: Any) -> dict[str, Any]:
    dumped = _dump_model_value(value)
    return dumped if isinstance(dumped, dict) else {}


def _target_app_info_payload(target_model: dict[str, Any]) -> dict[str, Any]:
    return _payload_dict(target_model.get("info"))


def _target_record_payload(target_model_info: Any | None) -> dict[str, Any]:
    return _payload_dict(target_model_info)


def _payload_meta(payload: dict[str, Any]) -> dict[str, Any]:
    meta = _dump_model_value(payload.get("meta"))
    return copy.deepcopy(meta) if isinstance(meta, dict) else {}


def _payload_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = _dump_model_value(payload.get("params"))
    return copy.deepcopy(params) if isinstance(params, dict) else {}


def build_wrapper_core_params(target_params: dict[str, Any]) -> dict[str, Any]:
    wrapper_params: dict[str, Any] = {"stream_response": True}
    for key in ("function_calling", "stream_delta_chunk_size", "reasoning_tags"):
        if key in target_params and target_params[key] is not None:
            wrapper_params[key] = copy.deepcopy(target_params[key])
    return wrapper_params


def _normalize_access_grants(grants: Any) -> list[dict[str, Any]]:
    if not isinstance(grants, list):
        return []
    out: list[dict[str, Any]] = []
    for grant in grants:
        normalized = _grant_to_dict(grant)
        if normalized is not None:
            out.append(normalized)
    return out


def _payload_access_grants(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    if "access_grants" in payload:
        return _normalize_access_grants(payload.get("access_grants"))
    meta = payload.get("meta")
    if isinstance(meta, dict) and "access_grants" in meta:
        return _normalize_access_grants(meta.get("access_grants"))
    return None


def _payload_owner_user_id(payload: dict[str, Any]) -> str | None:
    user_id = payload.get("user_id")
    return user_id if isinstance(user_id, str) and user_id else None


def build_target_model_contract(target_model: dict[str, Any], target_model_info: Any | None = None) -> TargetModelContract:
    target_id = str(target_model["id"])
    app_info = _target_app_info_payload(target_model)
    record_info = _target_record_payload(target_model_info)

    meta = _payload_meta(app_info) or _payload_meta(record_info)
    top_level_meta = _dump_model_value(target_model.get("meta"))
    if not meta and isinstance(top_level_meta, dict):
        meta = copy.deepcopy(top_level_meta)

    target_params = _payload_params(record_info) or _payload_params(app_info)
    top_level_params = _dump_model_value(target_model.get("params"))
    if not target_params and isinstance(top_level_params, dict):
        target_params = copy.deepcopy(top_level_params)

    grants = None
    if record_info:
        grants = _payload_access_grants(record_info)
    if grants is None:
        grants = _payload_access_grants(app_info)
    if grants is None:
        grants = _normalize_access_grants(target_model.get("access_grants"))

    owner_user_id = _payload_owner_user_id(record_info) or _payload_owner_user_id(app_info)
    user_id = target_model.get("user_id")
    if owner_user_id is None and isinstance(user_id, str) and user_id:
        owner_user_id = user_id

    return TargetModelContract(
        id=target_id,
        name=str(target_model.get("name") or target_id),
        meta=meta,
        wrapper_params=build_wrapper_core_params(target_params),
        access_grants=grants,
        owner_user_id=owner_user_id,
    )


def build_wrapper_model_form(
    *,
    pipe_function_id: str,
    function_owner_user_id: str,
    target_model: dict[str, Any],
    target_model_info: Any | None = None,
    valves: Any,
) -> dict[str, Any]:
    target = build_target_model_contract(target_model, target_model_info)
    target_id = target.id
    wrapper_id = build_wrapper_model_id(pipe_function_id, target_id)
    prefix = str(getattr(valves, "model_name_prefix", "Compact: "))
    access_grants = copy.deepcopy(target.access_grants)
    target_owner = target.owner_user_id
    if target_owner and target_owner != function_owner_user_id:
        owner_read = {
            "principal_type": "user",
            "principal_id": target_owner,
            "permission": "read",
        }
        if not any(
            grant.get("principal_type") == "user"
            and grant.get("principal_id") == target_owner
            and grant.get("permission") == "read"
            for grant in access_grants
        ):
            access_grants.append(owner_read)
    meta = copy.deepcopy(target.meta)
    meta["description"] = f"Auto-compacting wrapper for {target_id}"
    meta["auto_compaction"] = {
        "pipe_function_id": pipe_function_id,
        "target_model_id": target_id,
    }
    return {
        "id": wrapper_id,
        "user_id": function_owner_user_id,
        "base_model_id": None,
        "name": f"{prefix}{target.name}",
        "params": copy.deepcopy(target.wrapper_params),
        "meta": meta,
        "access_grants": access_grants,
        "is_active": True,
    }


def _record_field(record: Any, field: str) -> Any:
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _grant_signature(grant: Any) -> tuple[str, str, str] | None:
    grant_dict = _grant_to_dict(grant)
    if grant_dict is None:
        return None
    return (
        str(grant_dict.get("principal_type") or ""),
        str(grant_dict.get("principal_id") or ""),
        str(grant_dict.get("permission") or ""),
    )


def _grant_signatures(grants: Any) -> list[tuple[str, str, str]]:
    if not isinstance(grants, list):
        return []
    signatures = [_grant_signature(grant) for grant in grants]
    return sorted(signature for signature in signatures if signature is not None)


def _wrapper_model_record_matches_form(existing: Any, model_form: Any) -> bool:
    for field in ("id", "base_model_id", "name", "is_active"):
        if _record_field(existing, field) != getattr(model_form, field, None):
            return False
    if _dump_model_value(_record_field(existing, "params")) != _dump_model_value(getattr(model_form, "params", {})):
        return False
    if _dump_model_value(_record_field(existing, "meta")) != _dump_model_value(getattr(model_form, "meta", {})):
        return False
    return _grant_signatures(_record_field(existing, "access_grants")) == _grant_signatures(
        getattr(model_form, "access_grants", [])
    )


async def sync_wrapper_model_records(
    *,
    pipe_function_id: str,
    target_models: list[dict[str, Any]],
    valves: Any,
) -> None:
    try:
        from open_webui.models.functions import Functions
        from open_webui.models.models import ModelForm, ModelMeta, ModelParams, Models
    except Exception:
        return

    function = await Functions.get_function_by_id(pipe_function_id)
    owner_user_id = getattr(function, "user_id", None)
    if not owner_user_id:
        return

    for target_model in target_models:
        try:
            target_model_info = None
            target_id = _model_id(target_model)
            if target_id:
                with suppress(Exception):
                    target_model_info = await Models.get_model_by_id(target_id)
            form_payload = build_wrapper_model_form(
                pipe_function_id=pipe_function_id,
                function_owner_user_id=owner_user_id,
                target_model=target_model,
                target_model_info=target_model_info,
                valves=valves,
            )
            model_form = ModelForm(
                id=form_payload["id"],
                base_model_id=None,
                name=form_payload["name"],
                params=ModelParams(**form_payload["params"]),
                meta=ModelMeta(**form_payload["meta"]),
                access_grants=form_payload["access_grants"],
                is_active=True,
            )
            existing = await Models.get_model_by_id(form_payload["id"])
            if existing:
                if not _wrapper_model_record_matches_form(existing, model_form):
                    await Models.update_model_by_id(form_payload["id"], model_form)
            else:
                await Models.insert_new_model(model_form, user_id=owner_user_id)
        except Exception:
            continue


def _iter_model_cache_values(value: Any) -> Any:
    if isinstance(value, dict):
        return value.values()
    if isinstance(value, list):
        return value

    values = getattr(value, "values", None)
    if callable(values):
        with suppress(Exception):
            cache_values = values()
            if cache_values is not None:
                return cache_values
    return ()


def _iter_cache_models_from_state(state: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for attr in ("MODELS", "BASE_MODELS", "OPENAI_MODELS", "OLLAMA_MODELS"):
        value = getattr(state, attr, None)
        try:
            iterator = iter(_iter_model_cache_values(value))
        except TypeError:
            continue
        out.extend(item for item in iterator if isinstance(item, dict))
    return out


def validate_summary_model_id(configured_summary_model: str, target_model_id: str, models: dict[str, Any]) -> str:
    configured = (configured_summary_model or "").strip()
    summary_model = configured or target_model_id
    model = models.get(summary_model) if isinstance(models, dict) else None
    if model is None:
        if configured:
            raise ValueError(f"Configured summary_model was not found: {summary_model}")
        raise ValueError(f"Target model was not found for summary generation: {summary_model}")
    if isinstance(model, dict) and _is_arena_model(model):
        raise ValueError("Configured summary_model cannot be an arena model")
    return summary_model


def coerce_open_webui_user(user: Any) -> Any:
    if user is None:
        return None
    if all(hasattr(user, attr) for attr in ("id", "role", "name")):
        return user
    if isinstance(user, dict):
        try:
            from open_webui.models.users import UserModel

            return UserModel(**user)
        except Exception:
            return SimpleNamespace(**user)
    return user


def _normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    try:
        from open_webui.utils.response import normalize_usage

        return normalize_usage(usage)
    except Exception:
        if not usage:
            return {}
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        total_tokens = usage.get("total_tokens") or input_tokens + output_tokens
        result = dict(usage)
        result["input_tokens"] = int(input_tokens)
        result["output_tokens"] = int(output_tokens)
        result["total_tokens"] = int(total_tokens)
        return result


def extract_usage_from_stream_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not usage and isinstance(payload.get("response"), dict):
        usage = payload["response"].get("usage")
    if not usage and isinstance(payload.get("data"), dict):
        usage = payload["data"].get("usage")
    if isinstance(usage, dict) and usage:
        return _normalize_usage(usage)
    return None


def usage_state_key(chat_id: str | None, message_id: str | None, wrapper_model_id: str | None) -> str:
    digest = hashlib.sha256(
        json.dumps([chat_id or "", message_id or "", wrapper_model_id or ""], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"_auto_compact_usage_{digest}"


def store_request_scoped_usage(
    *,
    request: Any,
    chat_id: str | None,
    message_id: str | None,
    wrapper_model_id: str | None,
    usage: dict[str, Any],
) -> None:
    state = getattr(request, "state", None)
    if state is None:
        return
    setattr(state, usage_state_key(chat_id, message_id, wrapper_model_id), _normalize_usage(usage))


def get_request_scoped_usage(
    *,
    request: Any,
    chat_id: str | None,
    message_id: str | None,
    wrapper_model_id: str | None,
) -> dict[str, Any] | None:
    state = getattr(request, "state", None)
    if state is None:
        return None
    usage = getattr(state, usage_state_key(chat_id, message_id, wrapper_model_id), None)
    if isinstance(usage, dict) and usage:
        return _normalize_usage(usage)
    return None


def choose_usage_signal(
    *,
    request: Any,
    chat_id: str | None,
    message_id: str | None,
    wrapper_model_id: str | None,
    persisted_usage: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    usage = get_request_scoped_usage(
        request=request,
        chat_id=chat_id,
        message_id=message_id,
        wrapper_model_id=wrapper_model_id,
    )
    if usage:
        return usage
    if isinstance(persisted_usage, dict) and persisted_usage:
        return _normalize_usage(persisted_usage)
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue
        usage = message.get("usage") or (message.get("info") or {}).get("usage")
        if isinstance(usage, dict) and usage:
            return _normalize_usage(usage)
    return None


async def lookup_persisted_usage(chat_id: str | None, message_id: str | None) -> dict[str, Any] | None:
    if not chat_id:
        return None
    try:
        from open_webui.models.chats import Chats
        from open_webui.utils.misc import get_message_list

        messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
        if not isinstance(messages_map, dict) or not messages_map:
            return None
        branch: list[dict[str, Any]]
        if message_id and message_id in messages_map:
            branch = get_message_list(messages_map, message_id)
        else:
            branch = list(messages_map.values())
        for message in reversed(branch):
            usage = message.get("usage") or (message.get("info") or {}).get("usage")
            if isinstance(usage, dict) and usage:
                return _normalize_usage(usage)
    except Exception:
        return None
    return None


_METADATA_VALUE_DROPPED = object()


def _copy_metadata_value(value: Any, *, drop_uncloneable: bool, memo: dict[int, Any] | None = None) -> Any:
    if memo is None:
        memo = {}

    if isinstance(value, dict):
        value_id = id(value)
        if value_id in memo:
            return memo[value_id]

        copied: dict[Any, Any] = {}
        memo[value_id] = copied
        for key, item in value.items():
            try:
                copied_key = copy.deepcopy(key)
            except Exception:
                if drop_uncloneable:
                    continue
                copied_key = key

            copied_item = _copy_metadata_value(item, drop_uncloneable=drop_uncloneable, memo=memo)
            if copied_item is _METADATA_VALUE_DROPPED:
                continue
            copied[copied_key] = copied_item

        if drop_uncloneable and value and not copied:
            return _METADATA_VALUE_DROPPED
        return copied

    if isinstance(value, list):
        value_id = id(value)
        if value_id in memo:
            return memo[value_id]

        copied_list: list[Any] = []
        memo[value_id] = copied_list
        for item in value:
            copied_item = _copy_metadata_value(item, drop_uncloneable=drop_uncloneable, memo=memo)
            if copied_item is not _METADATA_VALUE_DROPPED:
                copied_list.append(copied_item)

        if drop_uncloneable and value and not copied_list:
            return _METADATA_VALUE_DROPPED
        return copied_list

    if isinstance(value, tuple):
        copied_items = []
        for item in value:
            copied_item = _copy_metadata_value(item, drop_uncloneable=drop_uncloneable, memo=memo)
            if copied_item is not _METADATA_VALUE_DROPPED:
                copied_items.append(copied_item)
        if drop_uncloneable and value and not copied_items:
            return _METADATA_VALUE_DROPPED
        return tuple(copied_items)

    if isinstance(value, (set, frozenset)):
        copied_items = []
        for item in value:
            copied_item = _copy_metadata_value(item, drop_uncloneable=drop_uncloneable, memo=memo)
            if copied_item is not _METADATA_VALUE_DROPPED:
                copied_items.append(copied_item)
        if drop_uncloneable and value and not copied_items:
            return _METADATA_VALUE_DROPPED
        try:
            return type(value)(copied_items)
        except TypeError:
            return _METADATA_VALUE_DROPPED if drop_uncloneable else value

    try:
        return copy.deepcopy(value)
    except Exception:
        return _METADATA_VALUE_DROPPED if drop_uncloneable else value


def _copy_metadata_preserving_references(metadata: Any) -> dict[str, Any]:
    copied = _copy_metadata_value(metadata or {}, drop_uncloneable=False)
    return copied if isinstance(copied, dict) else {}


def _copy_summary_task_metadata(metadata: Any) -> dict[str, Any]:
    copied = _copy_metadata_value(metadata or {}, drop_uncloneable=True)
    return copied if isinstance(copied, dict) else {}


def _copy_body_preserving_metadata(body: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy({key: value for key, value in body.items() if key != "metadata"})
    if "metadata" in body:
        copied["metadata"] = _copy_metadata_preserving_references(body.get("metadata"))
    return copied


def inject_stream_usage_options(body: dict[str, Any], *, force_include_usage: bool = True) -> dict[str, Any]:
    if not force_include_usage or body.get("stream") is not True:
        return body
    stream_options = body.get("stream_options")
    if not isinstance(stream_options, dict):
        stream_options = {}
    if stream_options.get("include_usage") is True:
        return body
    copied = _copy_body_preserving_metadata(body)
    copied["stream_options"] = {**stream_options, "include_usage": True}
    return copied


def _response_body_text(response: Response) -> str:
    body = getattr(response, "body", b"")
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)


def _json_from_response(response: Response) -> Any:
    text = _response_body_text(response)
    try:
        return json.loads(text)
    except Exception:
        return text


def _sse_data_chunk(data: str) -> str:
    return f"data: {data}\n\n"


def _response_to_sse_chunk(response: Response) -> str:
    text = _response_body_text(response)
    if isinstance(response, JSONResponse):
        return _sse_data_chunk(text)
    parsed = _json_from_response(response)
    if isinstance(parsed, (dict, list)):
        return _sse_data_chunk(json.dumps(parsed, ensure_ascii=False))
    return _sse_data_chunk(
        json.dumps(
            {"error": {"code": "provider_error", "message": text}},
            ensure_ascii=False,
        )
    )


def _coerce_stream_chunk(chunk: Any) -> bytes | str:
    if isinstance(chunk, (bytes, str)):
        return chunk
    if isinstance(chunk, (dict, list)):
        return _sse_data_chunk(json.dumps(chunk, ensure_ascii=False))
    return str(chunk)


def _iter_error_structures(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("error"), dict):
            yield value["error"]
        yield value
        detail = value.get("detail")
        if isinstance(detail, dict):
            yield from _iter_error_structures(detail)
    elif isinstance(value, HTTPException):
        yield from _iter_error_structures(value.detail)


def _structured_error_strings(value: Any) -> tuple[list[str], list[str], int | None]:
    status: int | None = None
    codes: list[str] = []
    messages: list[str] = []

    if isinstance(value, HTTPException):
        status = value.status_code
        value = value.detail
    elif isinstance(value, BaseException):
        messages.append(str(value))
        return codes, messages, 400
    elif isinstance(value, Response):
        status = value.status_code
        value = _json_from_response(value)

    if isinstance(value, str):
        messages.append(value)
        return codes, messages, status

    for error in _iter_error_structures(value):
        for key in ("code", "type", "param"):
            item = error.get(key)
            if isinstance(item, str):
                codes.append(item)
        for key in ("message", "detail"):
            item = error.get(key)
            if isinstance(item, str):
                messages.append(item)
            elif isinstance(item, dict):
                nested = item.get("message") or item.get("detail")
                if isinstance(nested, str):
                    messages.append(nested)

    return codes, messages, status


_CONTEXT_CODE_PATTERNS = (
    "context_length_exceeded",
    "context_window_exceeded",
    "max_context_length",
    "maximum_context_length",
    "input_too_long",
    "prompt_too_long",
    "request_too_large",
)
_NEGATIVE_ERROR_PATTERNS = (
    "rate limit",
    "rate-limit",
    "quota",
    "token-per-minute",
    "tokens per minute",
    "tpm",
    "insufficient_quota",
    "authentication",
    "permission",
)
_CONTEXT_MESSAGE_PATTERNS = (
    "maximum context length",
    "max context length",
    "context length exceeded",
    "context window",
    "prompt is too long",
    "prompt too long",
    "input is too long",
    "input too long",
    "too many input tokens",
    "request too large",
)


def is_retryable_context_error(value: Any, *, status_code: int | None = None) -> bool:
    codes, messages, inferred_status = _structured_error_strings(value)
    status = status_code if status_code is not None else inferred_status

    joined_codes = " ".join(codes).lower()
    joined_messages = " ".join(messages).lower()
    if any(pattern in joined_codes for pattern in _NEGATIVE_ERROR_PATTERNS):
        return False
    if any(pattern in joined_codes for pattern in _CONTEXT_CODE_PATTERNS):
        return True
    if any(pattern in joined_messages for pattern in _NEGATIVE_ERROR_PATTERNS):
        return False
    if status not in (400, 413, 422):
        return False
    return any(pattern in joined_messages for pattern in _CONTEXT_MESSAGE_PATTERNS)


def extract_sse_json_events(chunk: bytes | str) -> list[dict[str, Any]]:
    text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def first_chunk_is_retryable_context_error(chunk: bytes | str) -> bool:
    events = extract_sse_json_events(chunk)
    return bool(events and events[0].get("error") and is_retryable_context_error(events[0], status_code=400))


def chunk_has_user_visible_output(chunk: bytes | str) -> bool:
    for payload in extract_sse_json_events(chunk):
        if payload.get("error"):
            continue
        if payload.get("type") in {
            "response.output_text.delta",
            "response.reasoning_text.delta",
            "response.function_call_arguments.delta",
        }:
            return True
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") or {}
            if any(delta.get(key) for key in ("content", "reasoning_content", "tool_calls")):
                return True
    return False


def _request_bypass_system_prompt(request: Any) -> bool:
    state = getattr(request, "state", None)
    return bool(getattr(state, "bypass_system_prompt", False))


def _iter_request_state_items(state: Any) -> Iterable[tuple[str, Any]]:
    if state is None:
        return ()
    raw_state = getattr(state, "_state", None)
    if isinstance(raw_state, dict):
        return tuple(raw_state.items())
    try:
        return tuple(vars(state).items())
    except TypeError:
        return ()


def _copy_state_value(key: str, value: Any) -> Any:
    if key == "metadata":
        return _copy_metadata_preserving_references(value)
    return value


def _build_request_state(base_state: Any, overrides: dict[str, Any]) -> SimpleNamespace:
    state = SimpleNamespace()
    for key, value in _iter_request_state_items(base_state):
        setattr(state, key, _copy_state_value(key, value))
    for key, value in overrides.items():
        setattr(state, key, _copy_state_value(key, value))
    return state


class RequestStateProxy:
    def __init__(self, request: Any, **state_overrides: Any):
        object.__setattr__(self, "_request", request)
        object.__setattr__(self, "state", _build_request_state(getattr(request, "state", None), state_overrides))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request, name)


async def _close_stream_response(response: StreamingResponse) -> None:
    body_iterator = getattr(response, "body_iterator", None)
    aclose = getattr(body_iterator, "aclose", None)
    if callable(aclose):
        with suppress(Exception):
            await aclose()
    background = getattr(response, "background", None)
    if background is not None:
        with suppress(Exception):
            await background()


async def prepare_streaming_response(
    response: Any,
    *,
    request: Any,
    chat_id: str | None,
    message_id: str | None,
    wrapper_model_id: str | None,
    restore: Callable[[], None] | None = None,
) -> StreamingResponse:
    if isinstance(response, dict):
        if response.get("error") and is_retryable_context_error(response, status_code=400):
            if restore:
                restore()
            raise RetryableContextOverflow(str(response.get("error")))
        if restore:
            restore()
        return StreamingResponse(iter([f"data: {json.dumps(response)}\n\n"]), media_type="text/event-stream")

    if isinstance(response, (JSONResponse, PlainTextResponse)):
        if is_retryable_context_error(response):
            if restore:
                restore()
            raise RetryableContextOverflow(_response_body_text(response))
        if restore:
            restore()
        return StreamingResponse(iter([_response_to_sse_chunk(response)]), media_type="text/event-stream")

    if not isinstance(response, StreamingResponse):
        if restore:
            restore()
        return StreamingResponse(iter([f"data: {json.dumps(response)}\n\n"]), media_type="text/event-stream")

    iterator = response.body_iterator.__aiter__()
    buffered_chunks: list[bytes | str] = []
    try:
        while True:
            chunk = _coerce_stream_chunk(await iterator.__anext__())
            events = extract_sse_json_events(chunk)
            for payload in events:
                usage = extract_usage_from_stream_payload(payload)
                if usage:
                    store_request_scoped_usage(
                        request=request,
                        chat_id=chat_id,
                        message_id=message_id,
                        wrapper_model_id=wrapper_model_id,
                        usage=usage,
                    )
            buffered_chunks.append(chunk)
            if events:
                first_payload = events[0]
                if first_payload.get("error") and is_retryable_context_error(first_payload, status_code=400):
                    await _close_stream_response(response)
                    if restore:
                        restore()
                    raise RetryableContextOverflow("Target model reported a context-window error before output")
                break
    except StopAsyncIteration:
        if restore:
            restore()
        await _close_stream_response(response)
        return StreamingResponse(iter(buffered_chunks), media_type="text/event-stream")
    except Exception as exc:
        if restore:
            restore()
        await _close_stream_response(response)
        if is_retryable_context_error(exc):
            raise RetryableContextOverflow(str(exc)) from exc
        raise

    async def stream() -> AsyncIterator[bytes | str]:
        emitted_visible = False
        try:
            for buffered_chunk in buffered_chunks:
                emitted_visible = emitted_visible or chunk_has_user_visible_output(buffered_chunk)
                yield buffered_chunk

            async for raw_chunk in iterator:
                chunk = _coerce_stream_chunk(raw_chunk)
                for payload in extract_sse_json_events(chunk):
                    usage = extract_usage_from_stream_payload(payload)
                    if usage:
                        store_request_scoped_usage(
                            request=request,
                            chat_id=chat_id,
                            message_id=message_id,
                            wrapper_model_id=wrapper_model_id,
                            usage=usage,
                        )
                emitted_visible = emitted_visible or chunk_has_user_visible_output(chunk)
                yield chunk
        finally:
            if restore:
                restore()
            await _close_stream_response(response)

    return StreamingResponse(stream(), media_type="text/event-stream")


def _messages_have_multimodal(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("files"):
            return True
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") not in (None, "text"):
                    return True
    return False


def _chat_id_supported(chat_id: str | None) -> bool:
    return isinstance(chat_id, str) and bool(chat_id) and not chat_id.startswith(TEMP_CHAT_PREFIXES)


def _usage_total(usage: dict[str, Any] | None) -> int | None:
    if not usage:
        return None
    total = usage.get("total_tokens")
    if isinstance(total, bool):
        return None
    if isinstance(total, (int, float)):
        return int(total)
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
        return int(input_tokens + output_tokens)
    return None


def _context_exhaustion_error_response(
    messages: Any,
) -> dict[str, Any]:
    if isinstance(messages, list):
        cut = select_safe_message_cut(messages)
        tool_cut = select_tool_result_compaction_cut(messages)
        latest_user = next((message for message in reversed(messages) if message.get("role") == "user"), None)
        if latest_user is not None and not cut.summarization_prefix and tool_cut is None:
            return _error_response(
                "Target model context window was exceeded after all safe compaction options were exhausted; "
                "the latest user message must remain raw and appears too large to send safely",
                code="active_input_too_large",
            )
    return _error_response(
        "Target model context window was exceeded before output, and no safe retry path remains",
        code="context_window_exceeded",
    )


def _error_response(message: str, *, code: str = "auto_compaction_error") -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _chat_completion_message_response(
    model_id: str,
    message: str,
    *,
    usage: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    from open_webui.utils.misc import openai_chat_completion_message_template

    response = openai_chat_completion_message_template(model_id, message, tool_calls=tool_calls, usage=usage)
    if finish_reason is not None:
        response["choices"][0]["finish_reason"] = finish_reason
    return response


def _completion_error_message(value: Any) -> str:
    codes, messages, _ = _structured_error_strings(value)
    if messages:
        return messages[0]
    if codes:
        return codes[0]
    if isinstance(value, Response):
        return _response_body_text(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _chunk_is_sse_done_only(chunk: bytes | str) -> bool:
    text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
    meaningful_lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith(":")]
    if not meaningful_lines:
        return False
    for line in meaningful_lines:
        if not line.startswith("data:"):
            return False
        if line[len("data:") :].strip() != "[DONE]":
            return False
    return True


def _merge_stream_tool_call_delta(tool_calls: list[dict[str, Any]], delta_tool_call: dict[str, Any]) -> None:
    tool_call_index = delta_tool_call.get("index")
    current_tool_call = None
    if tool_call_index is not None:
        current_tool_call = next(
            (tool_call for tool_call in tool_calls if tool_call.get("index") == tool_call_index),
            None,
        )

    if current_tool_call is None:
        current_tool_call = copy.deepcopy(delta_tool_call)
        function = current_tool_call.get("function")
        if not isinstance(function, dict):
            function = {}
            current_tool_call["function"] = function
        function.setdefault("name", "")
        function.setdefault("arguments", "")
        tool_calls.append(current_tool_call)
        return

    for key in ("id", "type"):
        value = delta_tool_call.get(key)
        if value:
            current_tool_call[key] = value

    function_delta = delta_tool_call.get("function")
    if not isinstance(function_delta, dict):
        return

    function = current_tool_call.setdefault("function", {})
    if not isinstance(function, dict):
        function = {}
        current_tool_call["function"] = function

    name = function_delta.get("name")
    if isinstance(name, str) and name:
        function["name"] = name

    arguments = function_delta.get("arguments")
    if isinstance(arguments, str):
        function["arguments"] = str(function.get("arguments") or "") + arguments


async def emit_compaction_status(
    event_emitter: Callable[[Any], Awaitable[None]] | None,
    *,
    action: str,
    description: str,
    done: bool = False,
    error: bool = False,
    **extra: Any,
) -> None:
    if event_emitter is None:
        return
    data = {
        "action": f"auto_compaction_{action}",
        "description": description,
        "done": done,
    }
    if error:
        data["error"] = True
    data.update({key: value for key, value in extra.items() if value is not None})
    try:
        await event_emitter({"type": "status", "data": data})
    except Exception:
        return


async def extract_text_from_completion_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") or {}
            for key in ("content", "reasoning_content"):
                value = message.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for key in ("content", "text", "summary"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(response, StreamingResponse):
        parts: list[str] = []
        async for chunk in response.body_iterator:
            events = extract_sse_json_events(chunk)
            if events:
                for payload in events:
                    choices = payload.get("choices")
                    if isinstance(choices, list) and choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content") or delta.get("reasoning_content")
                        if isinstance(content, str):
                            parts.append(content)
                    if payload.get("type") == "response.output_text.delta":
                        delta = payload.get("delta")
                        if isinstance(delta, str):
                            parts.append(delta)
                continue
            if isinstance(chunk, bytes):
                parts.append(chunk.decode("utf-8", errors="replace"))
            else:
                parts.append(str(chunk))
        text = "".join(parts).strip()
        if text:
            return text
    raise RuntimeError("Summary model did not return text content")


def build_summary_task_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    copied = _copy_summary_task_metadata(metadata or {})
    copied.pop("selected_model_id", None)
    copied.pop("mcp_clients", None)
    copied["task"] = INTERNAL_SUMMARY_TASK
    copied["tools"] = {}
    copied["tool_ids"] = []
    return copied


def _strip_tool_request_fields(body: dict[str, Any]) -> dict[str, Any]:
    copied = _copy_body_preserving_metadata(body)
    for key in ("tools", "tool_choice", "tool_ids"):
        copied.pop(key, None)
    copied["stream"] = False
    return copied


def _resolve_summary_model_for_call(summary_model_id: str, *, pipe_function_id: str = PIPE_FUNCTION_ID) -> str:
    if is_generated_wrapper_model_id(summary_model_id, pipe_function_id=pipe_function_id):
        return decode_wrapper_model_id(summary_model_id, expected_pipe_function_id=pipe_function_id).target_model_id
    return summary_model_id


async def _generate_summary_text(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    summary_model_id: str,
    source_messages: list[dict[str, Any]],
) -> str:
    summary_metadata = build_summary_task_metadata(metadata)
    inner_request = RequestStateProxy(
        request,
        bypass_filter=True,
        bypass_system_prompt=False,
        metadata=summary_metadata,
    )
    from open_webui.utils.chat import generate_chat_completion

    body = _strip_tool_request_fields(
        {
            "model": _resolve_summary_model_for_call(summary_model_id),
            "messages": [
                {
                    "role": "system",
                    "content": SUMMARY_SYSTEM_PROMPT,
                },
                *copy.deepcopy(source_messages),
            ],
            "metadata": summary_metadata,
        }
    )
    await _ensure_model_in_request_models(inner_request, str(body.get("model") or ""))
    response = await generate_chat_completion(
        inner_request,
        body,
        user=coerce_open_webui_user(user),
        bypass_filter=True,
        bypass_system_prompt=False,
    )
    if is_retryable_context_error(response, status_code=400):
        raise RetryableContextOverflow("Summary model reported a context-window error")
    return await extract_text_from_completion_response(response)


def _messages_without_first_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_index = _first_system_index(messages)
    return [copy.deepcopy(message) for index, message in enumerate(messages) if index != system_index]


async def _compact_retry_tool_results(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    summary_model_id: str,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    cut = select_tool_result_compaction_cut(messages)
    if cut is None:
        return messages, False

    source_messages = _messages_without_first_system(messages)
    try:
        summary = await _generate_summary_text(
            request=request,
            user=user,
            metadata=metadata,
            summary_model_id=summary_model_id,
            source_messages=source_messages,
        )
    except Exception as exc:
        if isinstance(exc, RetryableContextOverflow) or is_retryable_context_error(exc):
            raise UnsupportedCompactionInput(
                "A conversation history with tool results exceeds the summary model context window and cannot be safely compacted",
                code="latest_tool_result_too_large",
            ) from exc
        raise

    compacted: list[dict[str, Any]] = []
    system_index = _first_system_index(messages)
    if system_index is not None:
        compacted.append(copy.deepcopy(messages[system_index]))
    compacted.append(
        render_summary_message(
            summary,
            {"has_multimodal": _messages_have_multimodal(source_messages)},
        )
    )
    compacted.append(copy.deepcopy(cut.active_user_message))
    return compacted, True


async def _get_or_create_checkpoint_summary(
    *,
    request: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    source_messages: list[dict[str, Any]],
    summary_meta: dict[str, Any],
    summary_factory: Callable[[dict[str, Any] | None], Awaitable[str]],
) -> str:
    profile_hash = compute_profile_hash()
    source_hash = compute_source_hash(source_messages)
    lock_key = (CHECKPOINT_NAMESPACE, user_id, chat_id, pipe_function_id, profile_hash, source_hash)
    lock = get_generation_lock(lock_key)

    try:
        async with lock:
            try:
                await ensure_checkpoint_table_initialized(request=request)
                store = CheckpointStore()
            except Exception:
                return await summary_factory(None)

            existing = await store.lookup_ready(
                namespace=CHECKPOINT_NAMESPACE,
                user_id=user_id,
                chat_id=chat_id,
                pipe_function_id=pipe_function_id,
                profile_hash=profile_hash,
                source_hash=source_hash,
            )
            if existing:
                with suppress(Exception):
                    await store.touch(existing["id"])
                return str(existing["summary_text"])

            parent = await store.find_longest_parent(
                namespace=CHECKPOINT_NAMESPACE,
                user_id=user_id,
                chat_id=chat_id,
                pipe_function_id=pipe_function_id,
                profile_hash=profile_hash,
                prefix_hashes=compute_prefix_hashes(source_messages),
            )
            try:
                summary_text = await summary_factory(parent)
            except Exception as exc:
                if parent:
                    raise ParentCheckpointExtensionFailed(parent, exc) from exc
                raise
            row = build_checkpoint_row(
                namespace=CHECKPOINT_NAMESPACE,
                user_id=user_id,
                chat_id=chat_id,
                pipe_function_id=pipe_function_id,
                profile_hash=profile_hash,
                source_hash=source_hash,
                source_message_count=len(source_messages),
                summary_text=summary_text,
                summary_meta=summary_meta,
                parent_checkpoint_id=parent.get("id") if parent else None,
            )
            try:
                inserted = await store.insert_ready(row)
            except Exception:
                return summary_text
            if inserted:
                return str(inserted["summary_text"])
            return summary_text
    finally:
        release_generation_lock(lock_key, lock)


async def _body_reusable_checkpoint_match(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    include_parent: bool,
) -> ReusableCheckpointMatch | None:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")
    if not user_id or not _chat_id_supported(chat_id):
        return None

    profile_hash = compute_profile_hash()
    cut = select_safe_message_cut(messages)
    if not cut.summarization_prefix:
        return None

    try:
        await ensure_checkpoint_table_initialized(request=request)
        store = CheckpointStore()
        source_hash = compute_source_hash(cut.summarization_prefix)
        existing = await store.lookup_ready(
            namespace=CHECKPOINT_NAMESPACE,
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            profile_hash=profile_hash,
            source_hash=source_hash,
        )
        if existing:
            return ReusableCheckpointMatch(kind="exact", source_message_count=cut.source_message_count)
        if include_parent:
            parent = await store.find_longest_parent(
                namespace=CHECKPOINT_NAMESPACE,
                user_id=user_id,
                chat_id=chat_id,
                pipe_function_id=pipe_function_id,
                profile_hash=profile_hash,
                prefix_hashes=compute_prefix_hashes(cut.summarization_prefix),
            )
            if parent is not None:
                return ReusableCheckpointMatch(
                    kind="parent",
                    source_message_count=int(parent.get("source_message_count") or 0),
                )
    except Exception:
        return None
    return None


async def _model_dict_from_request(request: Any) -> dict[str, Any]:
    state = getattr(getattr(request, "app", None), "state", None)
    if state is None:
        return {}
    models: dict[str, Any] = {}
    for model in _iter_cache_models_from_state(state):
        model_id = _model_id(model)
        if model_id is not None and model_id not in models:
            models[model_id] = model
    return models


async def _ensure_model_in_request_models(request: Any, model_id: str) -> dict[str, Any] | None:
    models = await _model_dict_from_request(request)
    model = models.get(model_id)
    if model is None:
        return None

    state = getattr(getattr(request, "app", None), "state", None)
    request_models = getattr(state, "MODELS", None)
    with suppress(Exception):
        if request_models is not None and model_id not in request_models:
            request_models[model_id] = copy.deepcopy(model)
    return model


def _should_bypass_target_access_check(user: Any) -> bool:
    try:
        from open_webui.env import BYPASS_MODEL_ACCESS_CONTROL

        if BYPASS_MODEL_ACCESS_CONTROL:
            return True
    except Exception:
        pass

    if getattr(user, "role", None) != "admin":
        return False

    try:
        from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL

        return bool(BYPASS_ADMIN_ACCESS_CONTROL)
    except Exception:
        return False


async def _target_has_db_model_record(target_model_id: str) -> bool:
    try:
        from open_webui.models.models import Models

        return await Models.get_model_by_id(target_model_id) is not None
    except Exception:
        return True


async def _validate_target_access(
    *,
    target_model_id: str,
    request: Any,
    user: Any,
) -> None:
    models = await _model_dict_from_request(request)
    model = models.get(target_model_id)
    if model is None or _is_arena_model(model) or is_generated_wrapper_model_id(target_model_id):
        raise HTTPException(status_code=403, detail="Model not found")

    user_model = coerce_open_webui_user(user)
    if _should_bypass_target_access_check(user_model):
        return

    if getattr(user_model, "role", None) == "admin" and not await _target_has_db_model_record(target_model_id):
        return

    try:
        from open_webui.utils.models import check_model_access

        await check_model_access(user_model, model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=403, detail="Model not found") from exc


async def _call_target_completion(
    *,
    request: Any,
    user: Any,
    body: dict[str, Any],
) -> Any:
    bypass_system_prompt = _request_bypass_system_prompt(request)
    inner_request = RequestStateProxy(
        request,
        bypass_filter=True,
        bypass_system_prompt=bypass_system_prompt,
    )
    try:
        from open_webui.utils.chat import generate_chat_completion

        await _ensure_model_in_request_models(inner_request, str(body.get("model") or ""))
        response = await generate_chat_completion(
            inner_request,
            body,
            user=coerce_open_webui_user(user),
            bypass_filter=True,
            bypass_system_prompt=bypass_system_prompt,
        )
        return response
    except Exception as exc:
        if is_retryable_context_error(exc):
            raise RetryableContextOverflow(str(exc)) from exc
        raise


async def _forward_streaming_target(
    *,
    request: Any,
    user: Any,
    body: dict[str, Any],
    chat_id: str | None,
    message_id: str | None,
    wrapper_model_id: str,
) -> StreamingResponse:
    response = await _call_target_completion(request=request, user=user, body=body)
    return await prepare_streaming_response(
        response,
        request=request,
        chat_id=chat_id,
        message_id=message_id,
        wrapper_model_id=wrapper_model_id,
    )


async def _coerce_non_streaming_completion_response(response: Any, *, model_id: str) -> dict[str, Any]:
    if isinstance(response, dict):
        if response.get("error") and is_retryable_context_error(response, status_code=400):
            raise RetryableContextOverflow(str(response.get("error")))
        return response

    if isinstance(response, StreamingResponse):
        parts: list[str] = []
        usage: dict[str, Any] | None = None
        tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        provider_error: dict[str, Any] | None = None
        try:
            async for raw_chunk in response.body_iterator:
                chunk = _coerce_stream_chunk(raw_chunk)
                events = extract_sse_json_events(chunk)
                if not events:
                    if _chunk_is_sse_done_only(chunk):
                        continue
                    parts.append(chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk))
                    continue
                for payload in events:
                    if payload.get("error"):
                        if is_retryable_context_error(payload, status_code=400):
                            raise RetryableContextOverflow(str(payload.get("error")))
                        provider_error = _error_response(_completion_error_message(payload), code="provider_error")
                        continue
                    chunk_usage = extract_usage_from_stream_payload(payload)
                    if chunk_usage:
                        usage = chunk_usage
                    choices = payload.get("choices")
                    if isinstance(choices, list) and choices:
                        choice = choices[0]
                        if isinstance(choice.get("finish_reason"), str):
                            finish_reason = choice["finish_reason"]
                        message = choice.get("message") or {}
                        delta = choice.get("delta") or {}
                        for container in (message, delta):
                            content = container.get("content") or container.get("reasoning_content")
                            if isinstance(content, str):
                                parts.append(content)
                            container_tool_calls = container.get("tool_calls")
                            if isinstance(container_tool_calls, list):
                                for tool_call in container_tool_calls:
                                    if isinstance(tool_call, dict):
                                        _merge_stream_tool_call_delta(tool_calls, tool_call)
                    if payload.get("type") == "response.output_text.delta":
                        delta = payload.get("delta")
                        if isinstance(delta, str):
                            parts.append(delta)
        finally:
            await _close_stream_response(response)
        if provider_error is not None:
            return provider_error
        return _chat_completion_message_response(
            model_id,
            "".join(parts).strip(),
            usage=usage,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
        )

    if isinstance(response, (JSONResponse, PlainTextResponse, Response)):
        if is_retryable_context_error(response):
            raise RetryableContextOverflow(_response_body_text(response))
        parsed = _json_from_response(response)
        if response.status_code >= 400:
            return _error_response(_completion_error_message(parsed), code="provider_error")
        if isinstance(parsed, dict) and not parsed.get("error"):
            return parsed
        return _error_response(_completion_error_message(parsed), code="provider_error")

    if isinstance(response, BaseModel):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped

    if isinstance(response, str):
        return _chat_completion_message_response(model_id, response)

    return _error_response(
        f"Unsupported target response type: {type(response).__name__}",
        code="unsupported_target_response",
    )


async def _forward_non_streaming_target(
    *,
    request: Any,
    user: Any,
    body: dict[str, Any],
) -> dict[str, Any]:
    response = await _call_target_completion(request=request, user=user, body=body)
    return await _coerce_non_streaming_completion_response(response, model_id=str(body.get("model") or ""))


async def _compact_body(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    target_model_id: str,
    summary_model_id: str,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
) -> tuple[dict[str, Any], bool]:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return body, False

    cut = select_safe_message_cut(messages)
    if not cut.summarization_prefix:
        compacted_messages, did_compact_tools = await _compact_retry_tool_results(
            request=request,
            user=user,
            metadata=metadata,
            summary_model_id=summary_model_id,
            messages=messages,
        )
        if did_compact_tools:
            compacted = _copy_body_preserving_metadata(body)
            compacted["messages"] = compacted_messages
            compacted.pop("previous_response_id", None)
            return compacted, True
        return body, False

    latest_user = next((m for m in reversed(cut.tail_messages) if m.get("role") == "user"), None)
    if latest_user is None:
        raise UnsupportedCompactionInput("Cannot compact safely without retaining the active latest user message")

    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")
    if not user_id or not _chat_id_supported(chat_id):
        return body, False

    summary_meta = {"has_multimodal": _messages_have_multimodal(cut.summarization_prefix)}
    checkpoint_source_prefix = canonicalize_messages_for_source_hash(cut.summarization_prefix)

    async def summary_factory(parent: dict[str, Any] | None) -> str:
        if parent:
            parent_count = int(parent.get("source_message_count") or 0)
            source = [
                render_summary_message(str(parent["summary_text"]), parent.get("summary_meta") or {}),
                *checkpoint_source_prefix[parent_count:],
            ]
        else:
            source = checkpoint_source_prefix
        return await _generate_summary_text(
            request=request,
            user=user,
            metadata=metadata,
            summary_model_id=summary_model_id,
            source_messages=source,
        )

    compacted = _copy_body_preserving_metadata(body)
    try:
        summary_text = await _get_or_create_checkpoint_summary(
            request=request,
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            source_messages=cut.summarization_prefix,
            summary_meta=summary_meta,
            summary_factory=summary_factory,
        )
        compacted["messages"] = replace_prefix_with_summary(
            messages,
            cut,
            summary_text,
            summary_meta,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
    except ParentCheckpointExtensionFailed as exc:
        compacted["messages"] = replace_prefix_with_parent_checkpoint_and_delta(
            cut,
            exc.parent,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
    compacted["messages"], _ = await _compact_retry_tool_results(
        request=request,
        user=user,
        metadata=metadata,
        summary_model_id=summary_model_id,
        messages=compacted["messages"],
    )
    compacted.pop("previous_response_id", None)
    return compacted, True


class Pipe:
    class Valves(BaseModel):
        model_name_prefix: str = Field(
            default="Compact: ",
            description="Display prefix for generated compact wrapper models.",
        )
        include_model_patterns: str = Field(
            default="",
            description="Comma-separated shell-style patterns matched against target model id and display name.",
        )
        exclude_model_patterns: str = Field(
            default="",
            description="Comma-separated shell-style patterns matched against target model id and display name.",
        )
        summary_model: str = Field(
            default="",
            description="Optional admin-selected Open WebUI model for summarization. Empty means use the decoded target model.",
        )
        trigger_total_tokens: int = Field(
            default=100000,
            ge=1,
            description="Provider-reported total token threshold that starts compaction.",
        )
        force_include_usage: bool = Field(
            default=True,
            description="Set stream_options.include_usage=true on copied streaming target requests when supported.",
        )
        historical_message_excerpt_bytes: int = Field(
            default=DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
            ge=1,
            description="Maximum UTF-8 bytes retained from the middle-truncated text of each historical user message.",
        )
        historical_message_excerpt_count: int = Field(
            default=DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
            ge=0,
            description="Maximum number of recent historical user messages inserted as middle-truncated excerpts. Set 0 to disable excerpts.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def pipes(self) -> list[dict[str, str]]:
        try:
            from open_webui.main import app

            state = app.state
        except Exception:
            return []

        targets = filter_target_models(_iter_cache_models_from_state(state), self.valves)
        await sync_wrapper_model_records(pipe_function_id=PIPE_FUNCTION_ID, target_models=targets, valves=self.valves)

        entries: list[dict[str, str]] = []
        prefix = str(self.valves.model_name_prefix or "")
        for target in targets:
            target_id = str(target["id"])
            target_name = str(target.get("name") or target_id)
            entries.append({"id": encode_target_model_id(target_id), "name": f"{prefix}{target_name}"})
        return entries

    async def pipe(
        self,
        body: dict[str, Any],
        __request__: Any = None,
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
        __event_emitter__: Callable[[Any], Awaitable[None]] | None = None,
        __tools__: dict[str, Any] | None = None,
    ) -> Any:
        if not isinstance(body, dict):
            return _error_response("Request body must be an object", code="invalid_request")
        if __request__ is None:
            return _error_response("Open WebUI request context is required", code="missing_request")

        is_streaming = body.get("stream") is True
        selected_wrapper_id = str(body.get("model") or "")
        try:
            identity = decode_wrapper_model_id(selected_wrapper_id, expected_pipe_function_id=PIPE_FUNCTION_ID)
        except ValueError as exc:
            return _error_response(str(exc), code="invalid_wrapper_id")

        user = __user__ or {}
        metadata = _copy_metadata_preserving_references(__metadata__ or {})
        chat_id = metadata.get("chat_id")
        message_id = metadata.get("message_id") or metadata.get("user_message_id")

        try:
            await _validate_target_access(target_model_id=identity.target_model_id, request=__request__, user=user)
        except Exception:
            return _error_response("Model not found", code="model_access_denied")

        models = await _model_dict_from_request(__request__)
        try:
            summary_model_id = validate_summary_model_id(self.valves.summary_model, identity.target_model_id, models)
        except ValueError as exc:
            return _error_response(str(exc), code="invalid_summary_model")

        inner = _copy_body_preserving_metadata(body)
        inner["model"] = identity.target_model_id
        inner["metadata"] = _copy_metadata_preserving_references(metadata)
        if is_streaming and self.valves.force_include_usage:
            inner = inject_stream_usage_options(inner, force_include_usage=True)

        is_summary_task = metadata.get("task") == INTERNAL_SUMMARY_TASK
        supported_context = _chat_id_supported(chat_id) and not is_summary_task

        request_usage = get_request_scoped_usage(
            request=__request__,
            chat_id=str(chat_id) if chat_id else None,
            message_id=str(message_id) if message_id else None,
            wrapper_model_id=selected_wrapper_id,
        )
        persisted_usage = None
        if request_usage is None:
            persisted_usage = await lookup_persisted_usage(
                str(chat_id) if chat_id else None,
                str(message_id) if message_id else None,
            )
        usage = choose_usage_signal(
            request=__request__,
            chat_id=str(chat_id) if chat_id else None,
            message_id=str(message_id) if message_id else None,
            wrapper_model_id=selected_wrapper_id,
            persisted_usage=persisted_usage or request_usage,
            messages=inner.get("messages") if isinstance(inner.get("messages"), list) else [],
        )
        total_tokens = _usage_total(usage)
        usage_should_compact = (
            supported_context and total_tokens is not None and total_tokens >= self.valves.trigger_total_tokens
        )
        reusable_checkpoint_match = None
        if supported_context:
            reusable_checkpoint_match = await _body_reusable_checkpoint_match(
                request=__request__,
                user=user,
                metadata=metadata,
                body=inner,
                pipe_function_id=identity.pipe_function_id,
                include_parent=usage_should_compact,
            )
        should_compact = usage_should_compact or reusable_checkpoint_match is not None

        attempt = 0
        compacted_once = False
        while attempt < MAX_CONTEXT_RETRY_ATTEMPTS:
            attempt += 1
            candidate = _copy_body_preserving_metadata(inner)
            if should_compact or compacted_once:
                try:
                    emit_progress_status = compacted_once or (
                        usage_should_compact
                        and (reusable_checkpoint_match is None or reusable_checkpoint_match.kind != "exact")
                    )
                    if emit_progress_status:
                        await emit_compaction_status(
                            __event_emitter__,
                            action="compacting",
                            description="Compacting chat history before forwarding to the target model",
                            done=False,
                        )
                    candidate, compacted = await _compact_body(
                        request=__request__,
                        user=user,
                        metadata=metadata,
                        body=candidate,
                        pipe_function_id=identity.pipe_function_id,
                        target_model_id=identity.target_model_id,
                        summary_model_id=summary_model_id,
                        historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                        historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                    )
                    compacted_once = compacted_once or compacted
                    if compacted and emit_progress_status:
                        await emit_compaction_status(
                            __event_emitter__,
                            action="compacted",
                            description="Compacted chat history is ready for the target model",
                            done=True,
                        )
                except UnsupportedCompactionInput as exc:
                    await emit_compaction_status(
                        __event_emitter__,
                        action="failed",
                        description=str(exc),
                        done=True,
                        error=True,
                    )
                    return _error_response(str(exc), code=exc.code)
                except Exception as exc:
                    await emit_compaction_status(
                        __event_emitter__,
                        action="failed",
                        description=f"Failed to compact chat history: {exc}",
                        done=True,
                        error=True,
                    )
                    return _error_response(f"Failed to compact chat history: {exc}", code="summary_failed")

            try:
                if is_streaming:
                    return await _forward_streaming_target(
                        request=__request__,
                        user=user,
                        body=candidate,
                        chat_id=str(chat_id) if chat_id else None,
                        message_id=str(message_id) if message_id else None,
                        wrapper_model_id=selected_wrapper_id,
                    )
                return await _forward_non_streaming_target(
                    request=__request__,
                    user=user,
                    body=candidate,
                )
            except RetryableContextOverflow:
                if not supported_context or attempt >= MAX_CONTEXT_RETRY_ATTEMPTS:
                    error_response = _context_exhaustion_error_response(
                        candidate.get("messages"),
                    )
                    await emit_compaction_status(
                        __event_emitter__,
                        action="failed",
                        description=error_response["error"]["message"],
                        done=True,
                        error=True,
                    )
                    return error_response
                await emit_compaction_status(
                    __event_emitter__,
                    action="retry",
                    description="Target context window was exceeded before output; compacting and retrying",
                    done=False,
                )
                should_compact = True
                compacted_once = True
                continue
