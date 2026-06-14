"""
title: Auto Compact
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Manifold Pipe that wraps Open WebUI models, compacts long chats, and persists durable checkpoint summaries.
version: 0.1.2
license: MIT
required_open_webui_version: 0.9.6
"""

from __future__ import annotations

import asyncio
import codecs
import copy
import fnmatch
import hashlib
import html
import json
import math
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from html.parser import HTMLParser
from types import SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Literal

from fastapi import HTTPException
from open_webui.constants import TASKS
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
    delete,
    insert,
    or_,
    select,
    text as sql_text,
    update,
)
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

try:
    import markdown as _markdown_mod
except Exception:
    _markdown_mod = None


PIPE_FUNCTION_ID = "auto_compact"
CHECKPOINT_NAMESPACE = "skyzi000.open_webui_extensions.auto_compaction_pipe"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_TABLE_NAME = "skyzi000_owui_ext_autocompact_checkpoint_v1"
CHECKPOINT_LOOKUP_UQ = "skyzi000_owui_ext_accp_v1_lookup_uq"
CHECKPOINT_PREFIX_IDX = "skyzi000_owui_ext_accp_v1_prefix_idx"
CHECKPOINT_RECENT_IDX = "skyzi000_owui_ext_accp_v1_recent_idx"
REQUEST_STATE_SCHEMA_READY_KEY = "_auto_compact_checkpoint_schema_ready_v1"
CHECKPOINT_CLAIM_LEASE_SECONDS = 90
CHECKPOINT_CLAIM_HEARTBEAT_SECONDS = 30
CHECKPOINT_PENDING_POLL_SECONDS = 0.25
CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS = 300.0
INTERNAL_SUMMARY_TASK = "auto_compaction_summary"
TEMP_CHAT_PREFIXES = ("local:", "channel:")
SUMMARY_FORMAT_FAMILY = "compact-user-summary-v1"
SOURCE_HASH_FAMILY = "canonical-json-v1"
PROFILE_HASH_FAMILY = "checkpoint-profile-v1"
MAX_CONTEXT_RETRY_ATTEMPTS = 2
DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES = 512
DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT = 32
SUMMARY_META_FORMAT_VERSION_KEY = "summary_meta_format_version"
SUMMARY_META_FORMAT_VERSION = 1
SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY = "historical_user_messages"
SUMMARY_META_HISTORICAL_USER_MESSAGES_FORMAT_VERSION = 1
PROVIDER_MODEL_CACHE_WAIT_DEFAULT_TIMEOUT_SECONDS = 10.0
PROVIDER_MODEL_CACHE_WAIT_POLL_SECONDS = 0.05
OLLAMA_PROVIDER_MODEL_CACHE_REFRESH_REQUESTS = 2
SummaryToolPolicy = Literal["always_strip", "fallback_on_tool_call", "error_on_tool_call"]
CORE_FUNCTION_MODEL_LISTING_COROUTINE = ("get_function_models", "open_webui/functions.py")
PROVIDER_MODEL_CACHE_REFRESH_COROUTINES = {
    "OPENAI_MODELS": (("fetch_openai_models", "open_webui/utils/models.py"),),
    "OLLAMA_MODELS": (("fetch_ollama_models", "open_webui/utils/models.py"),),
}
PROVIDER_MODEL_CACHE_ENABLE_FLAGS = {
    "OPENAI_MODELS": "ENABLE_OPENAI_API",
    "OLLAMA_MODELS": "ENABLE_OLLAMA_API",
}
MISSING_CORE_REQUEST = object()
TARGET_MODEL_RECORD_UNKNOWN = object()
COMPACTION_SUMMARY_EMBED_MARKER = "<!--auto-compaction-summary-embed:v1-->"
SUMMARY_PROMPT = (
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
    "Be concise, structured, and focused on continuity.\n\n"
    "The preceding messages are the exact checkpoint source to summarize. "
    "Messages after this checkpoint source may be retained raw separately; do not infer omitted active requests.\n\n"
    "Output only the checkpoint summary body. Do not continue the conversation. "
    "Do not call tools. Do not ask follow-up questions."
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
    Column("claim_token", Text, nullable=True),
    Column("claim_expires_at", BigInteger, nullable=True),
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
_SCHEMA_INIT_LOCKS: dict[Any, asyncio.Lock] = {}
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
    source_kind: str = "message"


@dataclass(frozen=True)
class TaskPromptSpec:
    config_attr: str
    default_attr: str
    builder_name: str
    strip_configured: bool = False


TASK_PROMPT_SPECS = {
    TASKS.TITLE_GENERATION.value: TaskPromptSpec(
        "TITLE_GENERATION_PROMPT_TEMPLATE",
        "DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE",
        "title_generation_template",
    ),
    TASKS.FOLLOW_UP_GENERATION.value: TaskPromptSpec(
        "FOLLOW_UP_GENERATION_PROMPT_TEMPLATE",
        "DEFAULT_FOLLOW_UP_GENERATION_PROMPT_TEMPLATE",
        "follow_up_generation_template",
    ),
    TASKS.TAGS_GENERATION.value: TaskPromptSpec(
        "TAGS_GENERATION_PROMPT_TEMPLATE",
        "DEFAULT_TAGS_GENERATION_PROMPT_TEMPLATE",
        "tags_generation_template",
    ),
    TASKS.QUERY_GENERATION.value: TaskPromptSpec(
        "QUERY_GENERATION_PROMPT_TEMPLATE",
        "DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE",
        "query_generation_template",
        strip_configured=True,
    ),
    TASKS.IMAGE_PROMPT_GENERATION.value: TaskPromptSpec(
        "IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE",
        "DEFAULT_IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE",
        "image_prompt_generation_template",
    ),
    TASKS.AUTOCOMPLETE_GENERATION.value: TaskPromptSpec(
        "AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE",
        "DEFAULT_AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE",
        "autocomplete_generation_template",
        strip_configured=True,
    ),
}


@dataclass(frozen=True)
class TargetModelContract:
    id: str
    name: str
    meta: dict[str, Any]
    wrapper_params: dict[str, Any]
    access_grants: list[dict[str, Any]]
    owner_user_id: str | None


@dataclass(frozen=True)
class CoreChatModelRoute:
    model_id: str


@dataclass(frozen=True)
class MessageCut:
    preserved_system_message: dict[str, Any] | None
    summarization_prefix: list[dict[str, Any]]
    tail_messages: list[dict[str, Any]]
    source_message_count: int


@dataclass(frozen=True)
class RetryToolResultCut:
    preserved_system_message: dict[str, Any] | None
    summarization_prefix: list[dict[str, Any]]
    tail_messages: list[dict[str, Any]]
    source_message_count: int


@dataclass(frozen=True)
class ToolResultCompactionCut:
    preserved_system_message: dict[str, Any] | None
    summarization_prefix: list[dict[str, Any]]
    tail_messages: list[dict[str, Any]]
    source_message_count: int


class RetryableContextOverflow(Exception):
    """Raised only before any user-visible target output has been emitted."""


class CompactionSummaryResult(str):
    def __new__(cls, summary_text: Any, *, checkpoint: dict[str, Any] | None = None):
        obj = str.__new__(cls, str(summary_text))
        obj.checkpoint = checkpoint
        return obj


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


def pipe_function_id_from_module_name(module_name: Any, *, fallback: str = PIPE_FUNCTION_ID) -> str:
    if isinstance(module_name, str):
        module_basename = module_name.rsplit(".", 1)[-1]
        prefix = "function_"
        if module_basename.startswith(prefix):
            function_id = module_basename[len(prefix) :]
            if function_id:
                return function_id
    return fallback


def runtime_pipe_function_id(pipe: Any, *, fallback: str = PIPE_FUNCTION_ID) -> str:
    return pipe_function_id_from_module_name(getattr(pipe.__class__, "__module__", None), fallback=fallback)


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
        preserved_system_message=copy.deepcopy(cut.preserved_system_message),
        summarization_prefix=copy.deepcopy(cut.summarization_prefix),
        tail_messages=copy.deepcopy(cut.tail_messages),
        source_message_count=cut.source_message_count,
    )


def select_tool_result_compaction_cut(
    messages: list[dict[str, Any]],
) -> ToolResultCompactionCut | None:
    source_messages = [copy.deepcopy(message) for message in messages]
    system_index = _first_system_index(source_messages)
    preserved_system = copy.deepcopy(source_messages[system_index]) if system_index is not None else None
    working = [message for index, message in enumerate(source_messages) if index != system_index]

    latest_user_index = next(
        (index for index in range(len(working) - 1, -1, -1) if working[index].get("role") == "user"),
        None,
    )
    if latest_user_index is None or latest_user_index >= len(working) - 1:
        return None

    tool_loop_start = latest_user_index + 1
    tool_loop_messages = working[tool_loop_start:]
    complete_rounds: list[tuple[int, int, set[str]]] = []
    all_assistant_ids: set[str] = set()
    for relative_index, message in enumerate(tool_loop_messages):
        if message.get("role") != "assistant":
            continue
        ids = _tool_call_ids(message)
        if not ids:
            continue
        round_tool_indices = [
            tool_loop_start + candidate_relative_index
            for candidate_relative_index, candidate in enumerate(tool_loop_messages)
            if candidate.get("role") == "tool" and candidate.get("tool_call_id") in ids
        ]
        seen = {working[index].get("tool_call_id") for index in round_tool_indices}
        if ids <= seen:
            round_start = tool_loop_start + relative_index
            round_end = max(round_tool_indices) + 1
            complete_rounds.append((round_start, round_end, ids))
            all_assistant_ids.update(ids)

    if not complete_rounds:
        return None

    for message in tool_loop_messages:
        if message.get("role") == "tool" and message.get("tool_call_id") not in all_assistant_ids:
            return None

    latest_round_start, _, _ = complete_rounds[-1]
    summarization_prefix = copy.deepcopy(working[:latest_round_start])
    tail_messages = copy.deepcopy(working[latest_round_start:])
    if not summarization_prefix or not tail_messages:
        return None

    tail_assistant_ids = _assistant_tool_call_ids(tail_messages)
    for message in tail_messages:
        if message.get("role") == "tool" and message.get("tool_call_id") not in tail_assistant_ids:
            return None

    if not any(message.get("role") == "tool" for message in tail_messages):
        return None

    return ToolResultCompactionCut(
        preserved_system_message=preserved_system,
        summarization_prefix=summarization_prefix,
        tail_messages=tail_messages,
        source_message_count=len(summarization_prefix),
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


def _decode_xml_cdata(value: str) -> str:
    if value.startswith("<![CDATA[") and value.endswith("]]>"):
        return value[len("<![CDATA[") : -len("]]>")].replace("]]]]><![CDATA[>", "]]>")
    return html.unescape(value)


_MD_ALLOWED_TAGS = frozenset(
    {
        "p",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "strong",
        "b",
        "em",
        "i",
        "u",
        "s",
        "del",
        "ins",
        "blockquote",
        "pre",
        "code",
        "kbd",
        "samp",
        "var",
        "a",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "span",
        "div",
    }
)
_MD_VOID_TAGS = frozenset({"br", "hr"})
_MD_ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "a": frozenset({"href", "title"}),
    "code": frozenset({"class"}),
    "pre": frozenset({"class"}),
    "span": frozenset({"class"}),
    "th": frozenset({"align", "style"}),
    "td": frozenset({"align", "style"}),
}
_MD_CLASS_RE = re.compile(r"^[a-zA-Z][\w\-]{0,40}$")
_MD_STYLE_RE = re.compile(r"^\s*text-align\s*:\s*(left|center|right)\s*;?\s*$", re.IGNORECASE)
_MD_RAW_HTML_TAG_RE = re.compile(
    r"^</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>$|^<!--.*-->$|^<![A-Za-z][^<>]*>$|^<\?[^<>]*\?>$"
)
_MD_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def _html_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return html.escape(text, quote=True).replace("'", "&#x27;")


def _is_safe_http_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    if any(char < " " for char in url):
        return False
    stripped = url.strip()
    lower = stripped.lower()
    return lower.startswith("http://") or lower.startswith("https://")


class _MarkdownSanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.out: list[str] = []
        self._stack: list[str] = []

    def _build_attrs(self, tag: str, attrs: list[tuple[str, str | None]]) -> str:
        allowed = _MD_ALLOWED_ATTRS.get(tag, frozenset())
        safe_pairs: list[tuple[str, str]] = []
        href_present = False
        for name, value in attrs:
            lname = (name or "").lower()
            if lname not in allowed:
                continue
            attr_value = value or ""
            if tag == "a" and lname == "href":
                if not _is_safe_http_url(attr_value):
                    continue
                href_present = True
                safe_pairs.append((lname, attr_value))
            elif lname == "class":
                tokens = [token for token in attr_value.split() if _MD_CLASS_RE.match(token)]
                if tokens:
                    safe_pairs.append((lname, " ".join(tokens)))
            elif lname == "style":
                if _MD_STYLE_RE.match(attr_value):
                    safe_pairs.append((lname, attr_value.strip()))
            elif lname == "align" and attr_value.lower() in {"left", "center", "right"}:
                safe_pairs.append((lname, attr_value.lower()))
            elif lname == "title":
                safe_pairs.append((lname, attr_value))
        attrs_str = "".join(f' {name}="{_html_escape(value)}"' for name, value in safe_pairs)
        if tag == "a" and href_present:
            attrs_str += ' target="_blank" rel="noopener noreferrer"'
        return attrs_str

    def _process_start(self, tag: str, attrs: list[tuple[str, str | None]], self_close: bool) -> None:
        if tag not in _MD_ALLOWED_TAGS:
            self.out.append(_html_escape(self.get_starttag_text() or f"<{tag}>"))
            return
        attrs_str = self._build_attrs(tag, attrs)
        if tag in _MD_VOID_TAGS:
            self.out.append(f"<{tag}{attrs_str}>")
            return
        if self_close:
            self.out.append(f"<{tag}{attrs_str}></{tag}>")
            return
        self.out.append(f"<{tag}{attrs_str}>")
        self._stack.append(tag)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._process_start(tag.lower(), attrs, self_close=False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._process_start(tag.lower(), attrs, self_close=True)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag not in _MD_ALLOWED_TAGS or tag in _MD_VOID_TAGS:
            self.out.append(_html_escape(f"</{tag}>"))
            return
        if tag not in self._stack:
            return
        while self._stack:
            top = self._stack.pop()
            self.out.append(f"</{top}>")
            if top == tag:
                break

    def handle_data(self, data: str) -> None:
        self.out.append(_html_escape(data))

    def handle_entityref(self, name: str) -> None:
        self.out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.out.append(f"&#{name};")

    def close(self) -> None:
        super().close()
        while self._stack:
            self.out.append(f"</{self._stack.pop()}>")


def _escape_markdown_raw_html_line(line: str) -> str:
    output: list[str] = []
    index = 0
    inline_code_ticks = 0
    while index < len(line):
        if line[index] == "`":
            end = index + 1
            while end < len(line) and line[end] == "`":
                end += 1
            tick_count = end - index
            if inline_code_ticks == 0:
                inline_code_ticks = tick_count
            elif inline_code_ticks == tick_count:
                inline_code_ticks = 0
            output.append(line[index:end])
            index = end
            continue
        if inline_code_ticks == 0 and line[index] == "<":
            end = line.find(">", index + 1)
            if end >= 0:
                candidate = line[index : end + 1]
                if _MD_RAW_HTML_TAG_RE.match(candidate):
                    output.append(_html_escape(candidate))
                    index = end + 1
                    continue
        output.append(line[index])
        index += 1
    return "".join(output)


def _escape_markdown_raw_html(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        match = _MD_FENCE_RE.match(body)
        if match:
            marker = match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[0] * len(marker)
                lines.append(line)
                continue
            if marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                in_fence = False
                fence_marker = ""
                lines.append(line)
                continue
        if in_fence or body.startswith(("    ", "\t")):
            lines.append(line)
            continue
        lines.append(_escape_markdown_raw_html_line(body) + newline)
    return "".join(lines)


def _render_summary_markdown(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    if _markdown_mod is None:
        return f'<pre class="md-fallback">{_html_escape(text)}</pre>'
    try:
        raw_html = _markdown_mod.markdown(
            _escape_markdown_raw_html(text),
            extensions=["extra", "sane_lists", "nl2br"],
            output_format="html",
        )
    except Exception:
        return f'<pre class="md-fallback">{_html_escape(text)}</pre>'
    sanitizer = _MarkdownSanitizer()
    try:
        sanitizer.feed(raw_html)
        sanitizer.close()
    except Exception:
        return f'<pre class="md-fallback">{_html_escape(text)}</pre>'
    return "".join(sanitizer.out)


def render_historical_user_message_excerpts(
    source_messages: list[dict[str, Any]],
    *,
    excerpt_bytes: int,
    max_messages: int,
) -> str:
    count_limit = int(max_messages or 0)
    if count_limit <= 0:
        return ""

    excerpts = [item["text"] for item in _build_historical_user_message_excerpt_items(
        source_messages,
        excerpt_bytes=excerpt_bytes,
        max_messages=count_limit,
    )]
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


def _coerce_nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return max(0, int(default))


def _build_historical_user_message_excerpt_items(
    source_messages: list[dict[str, Any]],
    *,
    excerpt_bytes: int,
    max_messages: int,
) -> list[dict[str, Any]]:
    count_limit = _coerce_nonnegative_int(max_messages)
    if count_limit <= 0:
        return []

    excerpts: list[str] = []
    for message in source_messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_content_as_excerpt_text(message).strip()
        if not text:
            continue
        excerpts.append(truncate_text_middle_by_utf8_bytes(text, excerpt_bytes))
    excerpts = excerpts[-count_limit:]
    return [{"ordinal": index, "text": excerpt} for index, excerpt in enumerate(excerpts, start=1)]


def _normalize_stored_historical_user_messages(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_messages = value.get("messages")
    if not isinstance(raw_messages, list):
        return None

    messages: list[dict[str, Any]] = []
    for index, item in enumerate(raw_messages, start=1):
        if not isinstance(item, dict) or "text" not in item:
            continue
        text = item.get("text")
        if text is None:
            continue
        ordinal = _coerce_nonnegative_int(item.get("ordinal"), default=index) or index
        messages.append({"ordinal": ordinal, "text": str(text)})
    if not messages:
        return None

    max_count = _coerce_nonnegative_int(value.get("max_count"), default=len(messages))
    max_bytes = _coerce_nonnegative_int(value.get("max_bytes_per_message"), default=0)
    return {
        "format_version": SUMMARY_META_HISTORICAL_USER_MESSAGES_FORMAT_VERSION,
        "order": "chronological",
        "max_count": max_count,
        "max_bytes_per_message": max_bytes,
        "selected_count": len(messages),
        "messages": messages,
    }


def normalize_summary_meta(summary_meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = copy.deepcopy(summary_meta) if isinstance(summary_meta, dict) else {}
    stored = _normalize_stored_historical_user_messages(meta.get(SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY))
    if stored is None:
        meta.pop(SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY, None)
    else:
        meta[SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY] = stored
    return meta


def enrich_summary_meta_with_historical_excerpts(
    summary_meta: dict[str, Any] | None,
    source_messages: list[dict[str, Any]],
    *,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
) -> dict[str, Any]:
    meta = normalize_summary_meta(summary_meta)
    meta[SUMMARY_META_FORMAT_VERSION_KEY] = SUMMARY_META_FORMAT_VERSION
    meta.pop(SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY, None)

    count_limit = _coerce_nonnegative_int(historical_message_excerpt_count)
    if count_limit <= 0:
        return meta

    excerpt_bytes = _coerce_nonnegative_int(historical_message_excerpt_bytes)
    messages = _build_historical_user_message_excerpt_items(
        source_messages,
        excerpt_bytes=excerpt_bytes,
        max_messages=count_limit,
    )
    if not messages:
        return meta

    meta[SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY] = {
        "format_version": SUMMARY_META_HISTORICAL_USER_MESSAGES_FORMAT_VERSION,
        "order": "chronological",
        "max_count": count_limit,
        "max_bytes_per_message": excerpt_bytes,
        "selected_count": len(messages),
        "messages": messages,
    }
    return meta


def build_checkpoint_summary_meta(
    source_messages: list[dict[str, Any]],
    *,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
) -> dict[str, Any]:
    return enrich_summary_meta_with_historical_excerpts(
        {"has_multimodal": _messages_have_multimodal(source_messages)},
        source_messages,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
    )


def render_stored_historical_user_message_excerpts(summary_meta: dict[str, Any] | None) -> str:
    meta = normalize_summary_meta(summary_meta)
    stored = meta.get(SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY)
    if not isinstance(stored, dict):
        return ""
    messages = stored.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""

    lines = [
        (
            f'<historical_user_messages order="{_xml_attr(stored.get("order") or "chronological")}" '
            f'selected_count="{len(messages)}" '
            f'max_count="{_coerce_nonnegative_int(stored.get("max_count"), default=len(messages))}" '
            f'max_bytes_per_message="{_coerce_nonnegative_int(stored.get("max_bytes_per_message"))}">'
        )
    ]
    for index, item in enumerate(messages, start=1):
        if not isinstance(item, dict):
            continue
        ordinal = _coerce_nonnegative_int(item.get("ordinal"), default=index) or index
        lines.append(f'<historical_user_message ordinal="{ordinal}">{_xml_cdata(item.get("text", ""))}</historical_user_message>')
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
    excerpts = render_stored_historical_user_message_excerpts(summary_meta)
    if not excerpts and historical_source_messages:
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


def render_summary_message_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    historical_source_messages: list[dict[str, Any]] | None = None,
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
) -> dict[str, Any]:
    summary_meta = normalize_summary_meta(checkpoint.get("summary_meta") if isinstance(checkpoint, dict) else {})
    fallback_source_messages = None
    if (
        SUMMARY_META_FORMAT_VERSION_KEY not in summary_meta
        and SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY not in summary_meta
    ):
        fallback_source_messages = historical_source_messages
    return render_summary_message(
        str(checkpoint.get("summary_text") or ""),
        summary_meta,
        historical_source_messages=fallback_source_messages,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
    )


def _checkpoint_from_summary_result(summary_text: Any) -> dict[str, Any] | None:
    checkpoint = getattr(summary_text, "checkpoint", None)
    return checkpoint if isinstance(checkpoint, dict) else None


def _render_summary_message_from_result(
    summary_text: Any,
    summary_meta: dict[str, Any] | None,
    *,
    historical_source_messages: list[dict[str, Any]] | None = None,
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
) -> dict[str, Any]:
    checkpoint = _checkpoint_from_summary_result(summary_text)
    if checkpoint is not None:
        return render_summary_message_from_checkpoint(
            checkpoint,
            historical_source_messages=historical_source_messages,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
    return render_summary_message(
        str(summary_text),
        summary_meta,
        historical_source_messages=historical_source_messages,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
    )


def extract_compaction_summary_text_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    start_tag = "<checkpoint_summary>"
    end_tag = "</checkpoint_summary>"
    for message in messages:
        if not isinstance(message, dict):
            continue
        if not _is_rendered_summary_context_message(message):
            continue
        content = message.get("content")
        start = content.find(start_tag)
        if start < 0:
            continue
        start += len(start_tag)
        end = _find_xml_element_end_outside_cdata(content, start, end_tag)
        if end < 0:
            continue
        summary = _decode_xml_cdata(content[start:end]).strip()
        if summary:
            return summary
    return None


def _is_rendered_summary_context_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    return stripped.startswith("<auto_compaction_context>") and stripped.endswith("</auto_compaction_context>")


def _find_xml_element_end_outside_cdata(text: str, start: int, end_tag: str) -> int:
    pos = start
    while pos < len(text):
        if text.startswith("<![CDATA[", pos):
            cdata_end = text.find("]]>", pos + len("<![CDATA["))
            if cdata_end < 0:
                return -1
            pos = cdata_end + len("]]>")
            continue
        if text.startswith(end_tag, pos):
            return pos
        pos += 1
    return -1


def render_compaction_summary_embed_html(summary_text: str) -> str:
    summary_body_html = _render_summary_markdown(str(summary_text or "").strip())
    if summary_body_html:
        rendered_summary = f'<div class="summary-body md">{summary_body_html}</div>'
    else:
        rendered_summary = f'<div class="summary-body plain">{_html_escape(summary_text)}</div>'
    styles = "".join(
        [
            "*{box-sizing:border-box}",
            (
                ":root{color-scheme:light;--fg:#374151;--fg-strong:#111827;--fg-muted:#6b7280;"
                "--border:rgba(209,213,219,.75);--border-open:rgba(156,163,175,.55);"
                "--surface:rgba(249,250,251,.92);--surface-open:#fff;--body-bg:rgba(255,255,255,.82);"
                "--hover:rgba(243,244,246,.85);--shadow:0 1px 2px rgba(15,23,42,.04)}"
            ),
            (
                ":root[data-theme='dark']{color-scheme:dark;--fg:#d1d5db;--fg-strong:#f9fafb;"
                "--fg-muted:#9ca3af;--border:rgba(75,85,99,.78);--border-open:rgba(107,114,128,.72);"
                "--surface:rgba(17,24,39,.68);--surface-open:rgba(17,24,39,.9);"
                "--body-bg:rgba(3,7,18,.32);--hover:rgba(31,41,55,.72);--shadow:none}"
            ),
            (
                "@media (prefers-color-scheme:dark){:root:not([data-theme='light']){color-scheme:dark;"
                "--fg:#d1d5db;--fg-strong:#f9fafb;--fg-muted:#9ca3af;--border:rgba(75,85,99,.78);"
                "--border-open:rgba(107,114,128,.72);--surface:rgba(17,24,39,.68);"
                "--surface-open:rgba(17,24,39,.9);--body-bg:rgba(3,7,18,.32);"
                "--hover:rgba(31,41,55,.72);--shadow:none}}"
            ),
            (
                "html,body{margin:0;padding:0;background:transparent;color:var(--fg);"
                "font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
                "font-size:13px;line-height:1.45}"
            ),
            "body{padding:2px}",
            (
                ".card{border:1px solid var(--border);border-radius:8px;background:var(--surface);"
                "overflow:hidden;margin:0;box-shadow:var(--shadow);transition:border-color .18s ease,"
                "background-color .18s ease,box-shadow .18s ease}"
            ),
            ".card[data-expanded='true']{border-color:var(--border-open);background:var(--surface-open)}",
            (
                "summary{display:flex;align-items:center;justify-content:space-between;gap:12px;"
                "padding:7px 10px;cursor:pointer;user-select:none;font-weight:500;color:var(--fg-strong);"
                "font-size:12.5px;line-height:1.35;transition:background-color .18s ease,color .18s ease}"
            ),
            "summary:hover{background:var(--hover)}",
            "summary::-webkit-details-marker{display:none}",
            ".title{overflow-wrap:anywhere}",
            (
                ".chevron{position:relative;width:16px;height:16px;flex:0 0 auto;color:var(--fg-muted);"
                "transition:color .18s ease}.chevron:before{content:'';position:absolute;left:4px;top:3px;"
                "width:7px;height:7px;border-right:1.7px solid currentColor;border-bottom:1.7px solid currentColor;"
                "transform:rotate(45deg);transition:transform .28s cubic-bezier(.22,1,.36,1),top .28s cubic-bezier(.22,1,.36,1)}"
            ),
            ".card[data-expanded='true'] .chevron:before{top:6px;transform:rotate(225deg)}",
            ".card[open]:not([data-js='true']) .chevron:before{top:6px;transform:rotate(225deg)}",
            (
                ".body-wrap{height:0;opacity:0;overflow:hidden;transition:height .28s cubic-bezier(.22,1,.36,1),"
                "opacity .18s ease}.card[data-expanded='true'] .body-wrap{opacity:1}"
            ),
            ".card[open]:not([data-js='true']) .body-wrap{height:auto;opacity:1}",
            (
                ".body{border-top:1px solid var(--border);padding:12px;background:var(--body-bg);"
                "max-height:36em;overflow:auto;scrollbar-width:thin}"
            ),
            (
                ".summary-body{margin:0;overflow-wrap:anywhere;word-break:break-word;"
                "font-size:12.5px;line-height:1.55;color:var(--fg-strong)}"
            ),
            ".summary-body.plain,.summary-body .md-fallback{white-space:pre-wrap}",
            (
                ".summary-body.plain,.summary-body .md-fallback,.summary-body.md pre,.summary-body.md code{"
                "font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono',monospace;"
                "font-size:12px}"
            ),
            ".summary-body.md h1,.summary-body.md h2,.summary-body.md h3,.summary-body.md h4{margin:10px 0 5px;font-weight:600;line-height:1.3}",
            ".summary-body.md h1{font-size:16px}.summary-body.md h2{font-size:15px}",
            ".summary-body.md h3{font-size:14px}.summary-body.md h4,.summary-body.md h5,.summary-body.md h6{font-size:13px}",
            ".summary-body.md p{margin:5px 0;overflow-wrap:anywhere;word-break:break-word}",
            ".summary-body.md ul,.summary-body.md ol{margin:5px 0;padding-left:20px}",
            ".summary-body.md li{margin:2px 0;overflow-wrap:anywhere;word-break:break-word}",
            (
                ".summary-body.md code{padding:1px 5px;border-radius:3px;background:rgba(127,127,127,.2);"
                "overflow-wrap:anywhere;word-break:break-all}"
            ),
            (
                ".summary-body.md pre{padding:8px 10px;border-radius:5px;background:rgba(127,127,127,.17);"
                "overflow-x:auto;margin:6px 0;line-height:1.4;white-space:pre-wrap}"
            ),
            ".summary-body.md pre code{background:transparent;padding:0}",
            ".summary-body.md blockquote{margin:6px 0;padding:2px 10px;border-left:3px solid rgba(127,127,127,.4);opacity:.85}",
            ".summary-body.md a{color:#2563eb;text-decoration:underline;text-underline-offset:2px}",
            ":root[data-theme='dark'] .summary-body.md a{color:#60a5fa}",
            ".summary-body.md table{border-collapse:collapse;margin:6px 0;font-size:12px;display:block;overflow-x:auto}",
            ".summary-body.md th,.summary-body.md td{padding:4px 8px;border:1px solid rgba(127,127,127,.3)}",
            ".summary-body.md hr{border:none;border-top:1px solid rgba(127,127,127,.3);margin:8px 0}",
            (
                "@media (prefers-reduced-motion:reduce){.card,summary,.chevron,.chevron:before,.body-wrap{"
                "transition:none!important}}"
            ),
        ]
    )
    initial_script = "".join(
        [
            "(function(){",
            "function applyInitialTheme(){var dark=false;var parentThemeRead=false;try{dark=parent.document.documentElement.classList.contains('dark')||(parent.document.body&&parent.document.body.classList.contains('dark'));parentThemeRead=true;}catch(e){dark=false;}",
            "if(!parentThemeRead&&window.matchMedia){dark=window.matchMedia('(prefers-color-scheme: dark)').matches;}",
            "document.documentElement.dataset.theme=dark?'dark':'light';}",
            "applyInitialTheme();",
            "parent.postMessage({type:'iframe:height',height:0},'*');",
            "})();",
        ]
    )
    script = "".join(
        [
            "(function(){",
            "var root=document.documentElement;",
            "var card=document.querySelector('.card');",
            "var summary=document.querySelector('summary');",
            "var wrap=document.querySelector('.body-wrap');",
            "var inner=document.querySelector('.body');",
            "if(card){card.dataset.js='true';}",
            "function syncTheme(){var dark=false;var parentThemeRead=false;try{dark=parent.document.documentElement.classList.contains('dark')||(parent.document.body&&parent.document.body.classList.contains('dark'));parentThemeRead=true;}catch(e){dark=false;}",
            "if(!parentThemeRead&&window.matchMedia){dark=window.matchMedia('(prefers-color-scheme: dark)').matches;}",
            "root.dataset.theme=dark?'dark':'light';}",
            "function postHeightValue(height){parent.postMessage({type:'iframe:height',height:Math.max(0,Math.ceil(height||0))},'*');}",
            "function documentHeight(){var b=document.body;return b?b.scrollHeight:document.documentElement.scrollHeight;}",
            "function postHeight(){requestAnimationFrame(function(){",
            "var b=document.body;",
            "var h=b?b.scrollHeight:document.documentElement.scrollHeight;",
            "postHeightValue(h);",
            "});}",
            "function reducedMotion(){try{return !!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);}catch(e){return false;}}",
            "function visibleBodyHeight(){if(!inner){return 0;}var height=inner.getBoundingClientRect().height;return Math.ceil(height||inner.offsetHeight||inner.clientHeight||0);}",
            "function currentWrapHeight(){if(!wrap){return 0;}var height=wrap.getBoundingClientRect().height;return Math.ceil(height||wrap.offsetHeight||0);}",
            "function baseDocumentHeight(){return Math.max(0,documentHeight()-currentWrapHeight());}",
            "var iframeAnimationId=0;",
            "function ease(t){return 1-Math.pow(1-t,3);}",
            "function animateIframeHeight(from,to,duration){iframeAnimationId+=1;var id=iframeAnimationId;var start=(window.performance&&window.performance.now)?window.performance.now():Date.now();function step(now){if(id!==iframeAnimationId){return;}var elapsed=Math.max(0,now-start);var progress=duration>0?Math.min(1,elapsed/duration):1;var value=from+(to-from)*ease(progress);postHeightValue(value);if(progress<1){requestAnimationFrame(step);}else{postHeight();}}requestAnimationFrame(step);}",
            "function animate(open){if(!card||!wrap||!inner){return;}",
            "card.dataset.expanded=open?'true':'false';",
            "if(reducedMotion()){if(open){card.open=true;wrap.style.height='auto';wrap.style.opacity='1';}else{wrap.style.height='0px';wrap.style.opacity='0';card.open=false;}postHeight();return;}",
            "var startDocHeight=documentHeight();",
            "if(open){card.open=true;wrap.style.height='0px';wrap.style.opacity='0';postHeightValue(startDocHeight);requestAnimationFrame(function(){var visibleHeight=visibleBodyHeight();var baseHeight=baseDocumentHeight();var targetDocHeight=baseHeight+visibleHeight;wrap.style.height=visibleHeight+'px';wrap.style.opacity='1';animateIframeHeight(startDocHeight,targetDocHeight,280);});}",
            "else{var visibleHeight=visibleBodyHeight();var baseHeight=baseDocumentHeight();var targetDocHeight=baseHeight;wrap.style.height=visibleHeight+'px';wrap.style.opacity='1';postHeightValue(startDocHeight);requestAnimationFrame(function(){wrap.style.height='0px';wrap.style.opacity='0';animateIframeHeight(startDocHeight,targetDocHeight,280);});}}",
            "if(summary){summary.addEventListener('click',function(event){event.preventDefault();animate(!(card.dataset.expanded==='true'));});}",
            "if(wrap){wrap.addEventListener('transitionend',function(event){if(event.propertyName!=='height'){return;}if(card.dataset.expanded==='true'){wrap.style.height='auto';}else{card.open=false;}postHeight();});}",
            "syncTheme();",
            "try{var observer=new MutationObserver(function(){syncTheme();postHeight();});observer.observe(parent.document.documentElement,{attributes:true,attributeFilter:['class']});if(parent.document.body){observer.observe(parent.document.body,{attributes:true,attributeFilter:['class']});}}catch(e){}",
            "postHeight();",
            "window.addEventListener('DOMContentLoaded',postHeight);",
            "window.addEventListener('load',postHeight);",
            "if(window.matchMedia){try{window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',function(){syncTheme();postHeight();});}catch(e){}}",
            "})();",
        ]
    )
    return (
        f"{COMPACTION_SUMMARY_EMBED_MARKER}"
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<script>{initial_script}</script>"
        f"<style>{styles}</style></head><body>"
        "<details class=\"card\" data-expanded=\"false\">"
        "<summary>"
        "<span class=\"title\">Compact summary</span>"
        "<span class=\"chevron\" aria-hidden=\"true\"></span>"
        "</summary>"
        f"<div class=\"body-wrap\"><div class=\"body\">{rendered_summary}</div></div>"
        "</details>"
        f"<script>{script}</script></body></html>"
    )


async def emit_compaction_summary_embed(
    event_emitter: Callable[[Any], Awaitable[None]] | None,
    *,
    summary_text: str | None,
) -> None:
    if event_emitter is None or not summary_text:
        return
    try:
        await event_emitter(
            {
                "type": "embeds",
                "data": {
                    "embeds": [render_compaction_summary_embed_html(summary_text)],
                    "replace": False,
                },
            }
        )
    except Exception:
        return


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
        _render_summary_message_from_result(
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
        render_summary_message_from_checkpoint(
            parent,
            historical_source_messages=cut.summarization_prefix[:parent_count],
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
    state: str = "ready",
    claim_token: str | None = None,
    claim_expires_at: int | None = None,
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
        "summary_meta": normalize_summary_meta(summary_meta),
        "state": state,
        "parent_checkpoint_id": parent_checkpoint_id,
        "claim_token": claim_token,
        "claim_expires_at": claim_expires_at,
        "created_at": timestamp,
        "updated_at": timestamp,
        "last_used_at": timestamp,
    }


def select_longest_matching_parent(
    rows: Iterable[dict[str, Any]],
    source_messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    prefix_hashes: dict[int, str] = {}
    candidates = sorted(rows, key=lambda row: int(row.get("source_message_count") or 0), reverse=True)
    for row in candidates:
        if row.get("state") != "ready":
            continue
        count = int(row.get("source_message_count") or 0)
        if count <= 0 or count > len(source_messages):
            continue
        if count not in prefix_hashes:
            prefix_hashes[count] = compute_source_hash(source_messages[:count])
        if prefix_hashes[count] == row.get("source_hash"):
            return row
    return None


_CHECKPOINT_CLAIM_COLUMN_DDL_TYPES = (("claim_token", "TEXT"), ("claim_expires_at", "BIGINT"))


def _is_duplicate_schema_object_error(exc: Exception) -> bool:
    message = str(getattr(exc, "orig", None) or exc).lower()
    return "already exists" in message or "duplicate" in message


def _execute_checkpoint_ddl_tolerating_duplicates(sync_conn: Any, statement: Any) -> None:
    try:
        with sync_conn.begin_nested():
            sync_conn.execute(statement)
    except (IntegrityError, OperationalError, ProgrammingError) as exc:
        if not _is_duplicate_schema_object_error(exc):
            raise


def _quoted_checkpoint_table_sql() -> str:
    if OPEN_WEBUI_DATABASE_SCHEMA:
        return f'"{OPEN_WEBUI_DATABASE_SCHEMA}"."{CHECKPOINT_TABLE_NAME}"'
    return f'"{CHECKPOINT_TABLE_NAME}"'


def _initialize_checkpoint_schema(sync_conn: Any) -> None:
    _execute_checkpoint_ddl_tolerating_duplicates(sync_conn, CreateTable(CHECKPOINT_TABLE, if_not_exists=True))
    for index in CHECKPOINT_TABLE.indexes:
        _execute_checkpoint_ddl_tolerating_duplicates(sync_conn, CreateIndex(index, if_not_exists=True))
    existing_columns = {
        column["name"]
        for column in sqlalchemy_inspect(sync_conn).get_columns(
            CHECKPOINT_TABLE_NAME, schema=OPEN_WEBUI_DATABASE_SCHEMA
        )
    }
    for column_name, column_ddl_type in _CHECKPOINT_CLAIM_COLUMN_DDL_TYPES:
        if column_name in existing_columns:
            continue
        _execute_checkpoint_ddl_tolerating_duplicates(
            sync_conn,
            sql_text(f"ALTER TABLE {_quoted_checkpoint_table_sql()} ADD COLUMN {column_name} {column_ddl_type}"),
        )


def _schema_init_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _SCHEMA_INIT_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _SCHEMA_INIT_LOCKS[loop] = lock
    return lock


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

    loop = asyncio.get_running_loop()
    lock = _schema_init_lock()
    try:
        async with lock:
            if not _CHECKPOINT_SCHEMA_READY:
                async with async_engine.begin() as conn:
                    await conn.run_sync(_initialize_checkpoint_schema)
                _CHECKPOINT_SCHEMA_READY = True
    finally:
        if not lock.locked() and not getattr(lock, "_waiters", None) and _SCHEMA_INIT_LOCKS.get(loop) is lock:
            _SCHEMA_INIT_LOCKS.pop(loop, None)
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

    def _identity_clauses(
        self,
        *,
        namespace: str,
        user_id: str,
        chat_id: str,
        pipe_function_id: str,
        profile_hash: str,
    ) -> list[Any]:
        return [
            CHECKPOINT_TABLE.c.namespace == namespace,
            CHECKPOINT_TABLE.c.user_id == user_id,
            CHECKPOINT_TABLE.c.chat_id == chat_id,
            CHECKPOINT_TABLE.c.pipe_function_id == pipe_function_id,
            CHECKPOINT_TABLE.c.profile_hash == profile_hash,
        ]

    async def lookup_any(
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
                    *self._identity_clauses(
                        namespace=namespace,
                        user_id=user_id,
                        chat_id=chat_id,
                        pipe_function_id=pipe_function_id,
                        profile_hash=profile_hash,
                    ),
                    CHECKPOINT_TABLE.c.source_hash == source_hash,
                )
            )
            row = result.mappings().first()
            return dict(row) if row else None

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
                    *self._identity_clauses(
                        namespace=namespace,
                        user_id=user_id,
                        chat_id=chat_id,
                        pipe_function_id=pipe_function_id,
                        profile_hash=profile_hash,
                    ),
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
        source_messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not source_messages:
            return None
        async with await self._context() as db:
            result = await db.execute(
                select(CHECKPOINT_TABLE)
                .where(
                    *self._identity_clauses(
                        namespace=namespace,
                        user_id=user_id,
                        chat_id=chat_id,
                        pipe_function_id=pipe_function_id,
                        profile_hash=profile_hash,
                    ),
                    CHECKPOINT_TABLE.c.state == "ready",
                    CHECKPOINT_TABLE.c.source_message_count <= len(source_messages),
                )
                .order_by(CHECKPOINT_TABLE.c.source_message_count.desc())
            )
            rows = [dict(row) for row in result.mappings().all()]
        return select_longest_matching_parent(rows, source_messages)

    async def claim_pending(self, row: dict[str, Any]) -> bool:
        async with await self._context() as db:
            try:
                await db.execute(insert(CHECKPOINT_TABLE).values(**row))
                await db.commit()
                return True
            except IntegrityError:
                await db.rollback()
                return False

    async def reclaim_pending(
        self,
        checkpoint_id: str,
        *,
        claim_token: str,
        expires_at: int,
        now: int | None = None,
    ) -> bool:
        timestamp = int(time.time()) if now is None else int(now)
        async with await self._context() as db:
            try:
                result = await db.execute(
                    update(CHECKPOINT_TABLE)
                    .where(
                        CHECKPOINT_TABLE.c.id == checkpoint_id,
                        CHECKPOINT_TABLE.c.state == "pending",
                        or_(
                            CHECKPOINT_TABLE.c.claim_expires_at.is_(None),
                            CHECKPOINT_TABLE.c.claim_expires_at <= timestamp,
                        ),
                    )
                    .values(claim_token=claim_token, claim_expires_at=int(expires_at), updated_at=timestamp)
                )
                if (result.rowcount or 0) != 1:
                    await db.rollback()
                    return False
                await db.commit()
                return True
            except Exception:
                await db.rollback()
                raise

    async def extend_claim(self, checkpoint_id: str, *, claim_token: str, expires_at: int) -> bool:
        async with await self._context() as db:
            try:
                result = await db.execute(
                    update(CHECKPOINT_TABLE)
                    .where(
                        CHECKPOINT_TABLE.c.id == checkpoint_id,
                        CHECKPOINT_TABLE.c.state == "pending",
                        CHECKPOINT_TABLE.c.claim_token == claim_token,
                    )
                    .values(claim_expires_at=int(expires_at))
                )
                if (result.rowcount or 0) != 1:
                    await db.rollback()
                    return False
                await db.commit()
                return True
            except Exception:
                await db.rollback()
                raise

    async def release_claim(self, checkpoint_id: str, *, claim_token: str) -> bool:
        async with await self._context() as db:
            try:
                result = await db.execute(
                    delete(CHECKPOINT_TABLE).where(
                        CHECKPOINT_TABLE.c.id == checkpoint_id,
                        CHECKPOINT_TABLE.c.state == "pending",
                        CHECKPOINT_TABLE.c.claim_token == claim_token,
                    )
                )
                if (result.rowcount or 0) != 1:
                    await db.rollback()
                    return False
                await db.commit()
                return True
            except Exception:
                await db.rollback()
                raise

    async def complete_pending(
        self,
        checkpoint_id: str,
        *,
        claim_token: str,
        summary_text: str,
        parent_checkpoint_id: str | None,
        now: int | None = None,
    ) -> dict[str, Any] | None:
        timestamp = int(time.time()) if now is None else int(now)
        async with await self._context() as db:
            try:
                result = await db.execute(
                    update(CHECKPOINT_TABLE)
                    .where(
                        CHECKPOINT_TABLE.c.id == checkpoint_id,
                        CHECKPOINT_TABLE.c.state == "pending",
                        CHECKPOINT_TABLE.c.claim_token == claim_token,
                    )
                    .values(
                        state="ready",
                        summary_text=summary_text,
                        parent_checkpoint_id=parent_checkpoint_id,
                        claim_token=None,
                        claim_expires_at=None,
                        updated_at=timestamp,
                        last_used_at=timestamp,
                    )
                )
                if (result.rowcount or 0) != 1:
                    await db.rollback()
                    return None
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            result = await db.execute(select(CHECKPOINT_TABLE).where(CHECKPOINT_TABLE.c.id == checkpoint_id))
            row = result.mappings().first()
            return dict(row) if row else None

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


def _normalize_meta_tags(tags: Any) -> list[dict[str, Any]]:
    if not isinstance(tags, list):
        return []
    normalized = []
    for tag in tags:
        if isinstance(tag, str):
            normalized.append({"name": tag})
        elif isinstance(tag, dict) and isinstance(tag.get("name"), str):
            normalized.append(copy.deepcopy(tag))
    return normalized


def _copy_top_level_display_metadata(meta: dict[str, Any], target_model: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(meta)

    for key in ("description", "profile_image_url"):
        value = target_model.get(key)
        if copied.get(key) is None and isinstance(value, str):
            copied[key] = value

    capabilities = _dump_model_value(target_model.get("capabilities"))
    if copied.get("capabilities") is None and isinstance(capabilities, dict):
        copied["capabilities"] = copy.deepcopy(capabilities)

    if copied.get("tags") is None:
        tags = _normalize_meta_tags(target_model.get("tags"))
        if tags:
            copied["tags"] = tags
    return copied


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
    meta = _copy_top_level_display_metadata(meta, target_model)

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


def _normalize_cache_model(attr: str, model: dict[str, Any]) -> dict[str, Any]:
    if attr != "OLLAMA_MODELS" or _model_id(model) is not None:
        return model

    model_id = model.get("model")
    if not isinstance(model_id, str) or not model_id:
        return model

    return {
        "id": model_id,
        "name": model.get("name") or model_id,
        "object": model.get("object", "model"),
        "created": model.get("created", 0),
        "owned_by": model.get("owned_by", "ollama"),
        "ollama": model.get("ollama", model),
        "loaded": model.get("loaded", "expires_at" in model),
        "connection_type": model.get("connection_type", "local"),
        "tags": model.get("tags", []),
    }


def _provider_model_cache_enabled_state(state: Any, attr: str) -> bool | None:
    flag = PROVIDER_MODEL_CACHE_ENABLE_FLAGS.get(attr)
    if flag is None:
        return None
    config = getattr(state, "config", None)
    if config is None or not hasattr(config, flag):
        return None
    return bool(getattr(config, flag))


def _disabled_provider_model_cache_attrs(state: Any) -> set[str]:
    return {
        attr
        for attr in PROVIDER_MODEL_CACHE_ENABLE_FLAGS
        if _provider_model_cache_enabled_state(state, attr) is False
    }


def _is_provider_cache_origin_model(model: dict[str, Any], attr: str) -> bool:
    if attr == "OPENAI_MODELS":
        return isinstance(model.get("openai"), dict)
    if attr == "OLLAMA_MODELS":
        return isinstance(model.get("ollama"), dict)
    return False


def _is_disabled_provider_cache_origin_model(model: dict[str, Any], disabled_provider_attrs: set[str]) -> bool:
    return any(_is_provider_cache_origin_model(model, attr) for attr in disabled_provider_attrs)


def _iter_cache_models_from_state(state: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    disabled_provider_attrs = _disabled_provider_model_cache_attrs(state)
    for attr in ("MODELS", "BASE_MODELS", "OPENAI_MODELS", "OLLAMA_MODELS"):
        if attr in disabled_provider_attrs:
            continue
        value = getattr(state, attr, None)
        try:
            iterator = iter(_iter_model_cache_values(value))
        except TypeError:
            continue
        for item in iterator:
            if not isinstance(item, dict):
                continue
            model = _normalize_cache_model(attr, item)
            if _is_disabled_provider_cache_origin_model(model, disabled_provider_attrs):
                continue
            out.append(model)
    return out


def _enabled_provider_model_cache_attrs(state: Any) -> list[str]:
    attrs: list[str] = []
    for attr in PROVIDER_MODEL_CACHE_ENABLE_FLAGS:
        if _provider_model_cache_enabled_state(state, attr) is True:
            attrs.append(attr)
    return attrs


def _provider_model_cache_wait_timeout_seconds() -> float:
    with suppress(Exception):
        from open_webui.env import AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST

        if AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST is not None:
            timeout = float(AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST)
            if timeout > 0:
                return timeout
    return PROVIDER_MODEL_CACHE_WAIT_DEFAULT_TIMEOUT_SECONDS


def _provider_cache_has_models(state: Any, attr: str) -> bool:
    value = getattr(state, attr, None)
    try:
        iterator = iter(_iter_model_cache_values(value))
    except TypeError:
        return False
    return any(isinstance(item, dict) for item in iterator)


def _iter_task_coroutine_frames(task: Any) -> Iterable[tuple[str, str, Any]]:
    coro = task.get_coro()
    seen: set[int] = set()
    while coro is not None and id(coro) not in seen:
        seen.add(id(coro))
        code = getattr(coro, "cr_code", None) or getattr(coro, "gi_code", None)
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
        if code is not None:
            yield code.co_name, str(code.co_filename).replace("\\", "/"), frame
        coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)


def _coroutine_code_matches(code_name: str, filename: str, expected: tuple[str, str]) -> bool:
    expected_name, expected_filename = expected
    return code_name == expected_name and filename.endswith(expected_filename)


def _task_coroutine_request(task: Any, expected: tuple[str, str]) -> Any:
    for code_name, filename, frame in _iter_task_coroutine_frames(task):
        if not _coroutine_code_matches(code_name, filename, expected):
            continue
        if frame is None:
            return MISSING_CORE_REQUEST
        return frame.f_locals.get("request", MISSING_CORE_REQUEST)
    return MISSING_CORE_REQUEST


def _provider_model_cache_refresh_pending_attrs(provider_cache_attrs: Iterable[str]) -> set[str]:
    wanted = {
        attr for attr in provider_cache_attrs if attr in PROVIDER_MODEL_CACHE_REFRESH_COROUTINES
    }
    if not wanted:
        return set()

    # Only Core's get_all_base_models() gather gives pipes() sibling provider tasks to wait for.
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        return set()
    if current_task is None:
        return set()
    current_request = _task_coroutine_request(current_task, CORE_FUNCTION_MODEL_LISTING_COROUTINE)
    if current_request is MISSING_CORE_REQUEST:
        return set()

    try:
        tasks = asyncio.all_tasks()
    except RuntimeError:
        return set()

    pending: set[str] = set()
    for task in tasks:
        if task is current_task or task.done():
            continue
        for attr in wanted - pending:
            if any(
                _task_coroutine_request(task, expected) is current_request
                for expected in PROVIDER_MODEL_CACHE_REFRESH_COROUTINES[attr]
            ):
                pending.add(attr)
        if pending == wanted:
            return pending
    return pending


def _provider_model_caches_ready(
    state: Any,
    initial_provider_caches: dict[str, Any],
    pending_provider_cache_attrs: Iterable[str],
) -> bool:
    pending_attrs = set(pending_provider_cache_attrs)
    for attr, initial_cache in initial_provider_caches.items():
        if _provider_cache_has_models(state, attr):
            continue
        if getattr(state, attr, None) is not initial_cache:
            continue
        if attr not in pending_attrs:
            continue
        return False
    return True


async def _wait_for_provider_model_caches(
    state: Any,
    *,
    initial_provider_caches: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    provider_cache_attrs = _enabled_provider_model_cache_attrs(state)
    if not provider_cache_attrs:
        return []

    timeout = max(0.0, float(_provider_model_cache_wait_timeout_seconds()))
    if "OLLAMA_MODELS" in provider_cache_attrs:
        # Core's Ollama refresh waits for /api/tags, then /api/ps before assigning
        # OLLAMA_MODELS. Each request can consume the model-list timeout.
        timeout *= OLLAMA_PROVIDER_MODEL_CACHE_REFRESH_REQUESTS
    poll_seconds = max(0.001, float(PROVIDER_MODEL_CACHE_WAIT_POLL_SECONDS))
    attempts = max(0, math.ceil(timeout / poll_seconds))
    if initial_provider_caches is None:
        initial_provider_caches = {attr: getattr(state, attr, None) for attr in provider_cache_attrs}

    # Core fetches provider models and function models in the same gather call.
    # Empty caches are final if no sibling Core provider fetch is still pending.
    for _ in range(attempts):
        models = _iter_cache_models_from_state(state)
        pending_provider_cache_attrs = _provider_model_cache_refresh_pending_attrs(provider_cache_attrs)
        if _provider_model_caches_ready(state, initial_provider_caches, pending_provider_cache_attrs):
            return models
        await asyncio.sleep(poll_seconds)
        models = _iter_cache_models_from_state(state)
        pending_provider_cache_attrs = _provider_model_cache_refresh_pending_attrs(provider_cache_attrs)
        if _provider_model_caches_ready(state, initial_provider_caches, pending_provider_cache_attrs):
            return models
    return _iter_cache_models_from_state(state)


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


def _normalized_task_name(task: Any) -> str:
    value = getattr(task, "value", None)
    if isinstance(value, str) and value:
        return value
    text = str(task or "")
    if "." in text:
        text = text.rsplit(".", 1)[-1].lower()
    return text


def _task_body_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    task_body = metadata.get("task_body")
    return task_body if isinstance(task_body, dict) else None


def _task_body_history_messages(metadata: dict[str, Any]) -> list[dict[str, Any]] | None:
    task_name = _normalized_task_name(metadata.get("task"))
    if task_name not in TASK_PROMPT_SPECS:
        return None
    task_body = _task_body_from_metadata(metadata)
    if task_body is None:
        return None
    messages = task_body.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        return None
    return messages


def _task_history_source_body_for_compaction(
    body: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    messages = _task_body_history_messages(metadata)
    if not messages:
        return None
    source_body = _copy_body_preserving_metadata(body)
    source_body["messages"] = copy.deepcopy(messages)
    source_body.pop("previous_response_id", None)
    return source_body


def _task_template_from_request(request: Any, spec: TaskPromptSpec) -> str | None:
    config = getattr(getattr(getattr(request, "app", None), "state", None), "config", None)
    configured = getattr(config, spec.config_attr, None) if config is not None else None
    if isinstance(configured, str) and (configured.strip() if spec.strip_configured else configured):
        return configured
    try:
        import open_webui.config as core_config

        default_template = getattr(core_config, spec.default_attr, None)
    except Exception:
        default_template = None
    return default_template if isinstance(default_template, str) and default_template else None


async def _render_task_prompt_from_messages(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str | None:
    task_name = _normalized_task_name(metadata.get("task"))
    spec = TASK_PROMPT_SPECS.get(task_name)
    task_body = _task_body_from_metadata(metadata)
    if spec is None or task_body is None:
        return None
    template = _task_template_from_request(request, spec)
    if template is None:
        return None
    try:
        import open_webui.utils.task as core_task

        builder = getattr(core_task, spec.builder_name)
        if task_name == TASKS.AUTOCOMPLETE_GENERATION.value:
            prompt = task_body.get("prompt")
            if not isinstance(prompt, str):
                return None
            return await builder(template, prompt, messages, task_body.get("type"), user)
        return await builder(template, messages, user)
    except Exception:
        return None


def _replace_task_prompt_message(body: dict[str, Any], content: str) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        return [{"role": "user", "content": content}]

    replaced = [dict(message) for message in messages]
    for index in range(len(replaced) - 1, -1, -1):
        if replaced[index].get("role") == "user":
            replaced[index]["content"] = content
            return replaced
    replaced.append({"role": "user", "content": content})
    return replaced


async def _rebuild_task_body_from_compacted_history(
    *,
    request: Any,
    user: Any,
    base_body: dict[str, Any],
    metadata: dict[str, Any],
    compacted_history_messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    content = await _render_task_prompt_from_messages(
        request=request,
        user=user,
        metadata=metadata,
        messages=compacted_history_messages,
    )
    if content is None:
        return None

    rebuilt = _copy_body_preserving_metadata(base_body)
    rebuilt["messages"] = _replace_task_prompt_message(rebuilt, content)
    rebuilt.pop("previous_response_id", None)

    rebuilt_metadata = _copy_metadata_preserving_references(rebuilt.get("metadata") or metadata)
    task_body = _task_body_from_metadata(rebuilt_metadata)
    if task_body is not None:
        task_body = _copy_metadata_preserving_references(task_body)
        task_body["messages"] = copy.deepcopy(compacted_history_messages)
        rebuilt_metadata["task_body"] = task_body
        rebuilt["metadata"] = rebuilt_metadata
    return rebuilt


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
    parsed = _json_from_response(response)
    if isinstance(response, JSONResponse) and response.status_code >= 400:
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            return _sse_data_chunk(text)
        message_source = parsed.get("error") if isinstance(parsed, dict) and parsed.get("error") else parsed
        return _sse_data_chunk(
            json.dumps(
                _error_response(_completion_error_message(message_source), code="provider_error"),
                ensure_ascii=False,
            )
        )
    if isinstance(response, JSONResponse):
        return _sse_data_chunk(text)
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
    "context_length",
    "context_limit",
    "max_context_length",
    "maximum_context_length",
    "input_length_exceeded",
    "input_too_long",
    "prompt_too_long",
    "request_too_large",
    "token_limit_exceeded",
)
_NEGATIVE_ERROR_PATTERNS = (
    "rate limit",
    "rate-limit",
    "rate_limit",
    "too many requests",
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
    "context limit",
    "context size",
    "exceed context",
    "exceeds context",
    "exceeded context",
    "prompt is too long",
    "prompt too long",
    "input is too long",
    "input too long",
    "input token count exceeds",
    "input tokens are",
    "input tokens exceed",
    "prompt tokens exceed",
    "too many input tokens",
    "request too large",
)
_OUTPUT_TOKEN_PARAMETER_PATTERNS = (
    "max_tokens",
    "max_completion_tokens",
    "max output tokens",
    "max_output_tokens",
)
_OUTPUT_TOKEN_PARAMETER_CONTEXT_ANCHORS = (
    "context",
    "input",
    "prompt",
    "request too large",
)


def is_retryable_context_error(value: Any, *, status_code: int | None = None) -> bool:
    codes, messages, inferred_status = _structured_error_strings(value)
    status = status_code if status_code is not None else inferred_status

    joined_codes = " ".join(codes).lower()
    joined_messages = " ".join(messages).lower()
    if any(pattern in joined_codes for pattern in _NEGATIVE_ERROR_PATTERNS) or any(
        pattern in joined_messages for pattern in _NEGATIVE_ERROR_PATTERNS
    ):
        return False
    if any(pattern in joined_messages for pattern in _OUTPUT_TOKEN_PARAMETER_PATTERNS) and not any(
        pattern in joined_messages for pattern in _OUTPUT_TOKEN_PARAMETER_CONTEXT_ANCHORS
    ):
        return False
    if any(pattern in joined_codes for pattern in _CONTEXT_CODE_PATTERNS):
        return True
    if status not in (400, 413, 422):
        return False
    return any(pattern in joined_messages for pattern in _CONTEXT_MESSAGE_PATTERNS)


_SSE_FIELD_NAMES = ("data", "event", "id", "retry")
_SSE_FIELD_MARKERS = tuple(f"{name}:" for name in _SSE_FIELD_NAMES) + (":",)


def _text_can_start_sse_field(stripped: str) -> bool:
    if not stripped:
        return False
    first_line = stripped.splitlines()[0]
    return (
        first_line in _SSE_FIELD_NAMES
        or first_line.startswith(_SSE_FIELD_MARKERS)
        or any(marker.startswith(first_line) for marker in _SSE_FIELD_MARKERS)
        or ":" in first_line
    )


class _SSEDataParser:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buffer = ""
        self._data_lines: list[str] = []

    def has_pending_event_or_field(self) -> bool:
        return bool(self._data_lines) or _text_can_start_sse_field(self._buffer.lstrip())

    def feed(self, chunk: bytes | str) -> tuple[list[str], bool]:
        text = self._decoder.decode(chunk, final=False) if isinstance(chunk, bytes) else str(chunk)
        self._buffer += text
        lines = self._buffer.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._buffer = lines.pop()
        else:
            self._buffer = ""
        return self._process_lines([line.rstrip("\r\n") for line in lines])

    def flush(self) -> tuple[list[str], bool]:
        lines = []
        remaining_text = self._decoder.decode(b"", final=True)
        if remaining_text:
            self._buffer += remaining_text
        if self._buffer:
            lines.append(self._buffer)
            self._buffer = ""
        values, saw_data_field = self._process_lines(lines)
        final_value = self._consume_event()
        if final_value is not None:
            values.append(final_value)
        return values, saw_data_field

    def _process_lines(self, lines: list[str]) -> tuple[list[str], bool]:
        values: list[str] = []
        saw_data_field = False
        for line in lines:
            if line == "":
                value = self._consume_event()
                if value is not None:
                    values.append(value)
                continue
            field, separator, data = line.partition(":")
            if field != "data":
                continue
            saw_data_field = True
            if not separator:
                data = ""
            if data.startswith(" "):
                data = data[1:]
            self._data_lines.append(data)
        return values, saw_data_field

    def _consume_event(self) -> str | None:
        if not self._data_lines:
            return None
        value = "\n".join(self._data_lines)
        self._data_lines = []
        if value.strip() == "[DONE]":
            return None
        return value


def _parse_sse_data_values(chunk: bytes | str) -> tuple[list[str], bool]:
    parser = _SSEDataParser()
    values, saw_data_field = parser.feed(chunk)
    flushed_values, flushed_saw_data_field = parser.flush()
    values.extend(flushed_values)
    return values, saw_data_field or flushed_saw_data_field


def extract_sse_data_values(chunk: bytes | str) -> list[str]:
    values, _ = _parse_sse_data_values(chunk)
    return values


def _sse_json_event_from_value(data: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(data.strip())
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _sse_json_events_from_values(values: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for data in values:
        payload = _sse_json_event_from_value(data)
        if payload is not None:
            events.append(payload)
    return events


class _SSEJSONEventParser:
    def __init__(self) -> None:
        self._parser = _SSEDataParser()

    def feed(self, chunk: bytes | str) -> list[dict[str, Any]]:
        values, _ = self._parser.feed(chunk)
        return _sse_json_events_from_values(values)

    def feed_events_and_values(self, chunk: bytes | str) -> tuple[list[dict[str, Any]], list[str]]:
        values, _ = self._parser.feed(chunk)
        return _sse_json_events_from_values(values), values

    def has_pending_event_or_field(self) -> bool:
        return self._parser.has_pending_event_or_field()

    def flush(self) -> list[dict[str, Any]]:
        values, _ = self._parser.flush()
        return _sse_json_events_from_values(values)


def extract_sse_json_events(chunk: bytes | str) -> list[dict[str, Any]]:
    return _sse_json_events_from_values(extract_sse_data_values(chunk))


def first_chunk_is_retryable_context_error(chunk: bytes | str) -> bool:
    events = extract_sse_json_events(chunk)
    if not events:
        return False
    error_source = _stream_payload_error_source(events[0])
    return bool(error_source is not None and is_retryable_context_error(error_source, status_code=400))


def _stream_payload_has_user_visible_output(payload: dict[str, Any]) -> bool:
    if payload.get("error"):
        return False
    payload_type = payload.get("type")
    if isinstance(payload_type, str) and payload_type.startswith("response.") and payload_type.endswith(".delta"):
        return True
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        delta = choices[0].get("delta") or {}
        if any(delta.get(key) for key in ("content", "reasoning_content", "tool_calls")):
            return True
    return False


def _stream_payload_is_responses_pre_output_control_event(payload: dict[str, Any]) -> bool:
    if _stream_payload_error_source(payload) is not None:
        return False
    payload_type = payload.get("type")
    if payload_type in {"response.created", "response.in_progress"}:
        return True
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            return False
        if choice.get("finish_reason") not in (None, ""):
            return False
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return False
        if any(
            delta.get(key)
            for key in ("content", "reasoning_content", "tool_calls", "function_call")
        ):
            return False
    return True


def _stream_events_are_responses_pre_output_control_events(events: list[dict[str, Any]]) -> bool:
    return bool(events) and all(_stream_payload_is_responses_pre_output_control_event(payload) for payload in events)


def _chunk_starts_unstructured_stream(chunk: bytes | str) -> bool:
    text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
    stripped = text.lstrip()
    return bool(stripped) and not _text_can_start_sse_field(stripped)


def _stream_chunk_is_empty(chunk: bytes | str) -> bool:
    return chunk == b"" or chunk == ""


def _sse_values_include_unstructured_output(values: Iterable[str]) -> bool:
    return any(value != "" for value in values)


def chunk_has_user_visible_output(chunk: bytes | str) -> bool:
    for payload in extract_sse_json_events(chunk):
        if _stream_payload_has_user_visible_output(payload):
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
    if getattr(response, "_auto_compact_stream_closed", False):
        return
    with suppress(Exception):
        setattr(response, "_auto_compact_stream_closed", True)
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
    sse_parser = _SSEJSONEventParser()
    can_retry_context_error = True
    media_type = getattr(response, "media_type", None) or response.headers.get("content-type", "")
    is_sse_response = "text/event-stream" in str(media_type).lower()
    try:
        while True:
            chunk = _coerce_stream_chunk(await iterator.__anext__())
            if is_sse_response:
                events, sse_values = sse_parser.feed_events_and_values(chunk)
            else:
                events, sse_values = [], []
            for value in sse_values:
                if value == "":
                    continue
                payload = _sse_json_event_from_value(value)
                if payload is None:
                    can_retry_context_error = False
                    continue
                usage = extract_usage_from_stream_payload(payload)
                if usage:
                    store_request_scoped_usage(
                        request=request,
                        chat_id=chat_id,
                        message_id=message_id,
                        wrapper_model_id=wrapper_model_id,
                        usage=usage,
                    )
                if can_retry_context_error:
                    error_source = _stream_payload_error_source(payload)
                    if error_source is not None and is_retryable_context_error(error_source, status_code=400):
                        await _close_stream_response(response)
                        if restore:
                            restore()
                        raise RetryableContextOverflow("Target model reported a context-window error before output")
                if not _stream_payload_is_responses_pre_output_control_event(payload):
                    can_retry_context_error = False
            buffered_chunks.append(chunk)
            if not is_sse_response:
                if _stream_chunk_is_empty(chunk):
                    continue
                break
            if not events and (
                _sse_values_include_unstructured_output(sse_values)
                or (_chunk_starts_unstructured_stream(chunk) and not sse_parser.has_pending_event_or_field())
            ):
                break
            if events and not (
                can_retry_context_error and _stream_events_are_responses_pre_output_control_events(events)
            ):
                break
    except StopAsyncIteration:
        events = sse_parser.flush()
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
            if can_retry_context_error:
                error_source = _stream_payload_error_source(payload)
                if error_source is not None and is_retryable_context_error(error_source, status_code=400):
                    if restore:
                        restore()
                    await _close_stream_response(response)
                    raise RetryableContextOverflow("Target model reported a context-window error before output")
            if not _stream_payload_is_responses_pre_output_control_event(payload):
                can_retry_context_error = False
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
        try:
            for buffered_chunk in buffered_chunks:
                yield buffered_chunk

            async for raw_chunk in iterator:
                chunk = _coerce_stream_chunk(raw_chunk)
                if is_sse_response:
                    events = sse_parser.feed(chunk)
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
                yield chunk
            if is_sse_response:
                for payload in sse_parser.flush():
                    usage = extract_usage_from_stream_payload(payload)
                    if usage:
                        store_request_scoped_usage(
                            request=request,
                            chat_id=chat_id,
                            message_id=message_id,
                            wrapper_model_id=wrapper_model_id,
                            usage=usage,
                        )
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


class SummaryToolCallError(RuntimeError):
    pass


_SUMMARY_INCOMPLETE_FINISH_REASON_PATTERNS = (
    "length",
    "content_filter",
    "max_token",
    "max_output",
    "max output",
    "truncat",
    "incomplete",
)


def _summary_tool_call_error() -> SummaryToolCallError:
    return SummaryToolCallError("Summary model returned a tool call instead of text content")


def _choice_has_tool_call(choice: Any) -> bool:
    if not isinstance(choice, dict):
        return False
    if choice.get("finish_reason") in {"tool_calls", "function_call"}:
        return True
    message = choice.get("message")
    if isinstance(message, dict) and (message.get("tool_calls") or message.get("function_call")):
        return True
    delta = choice.get("delta")
    return isinstance(delta, dict) and bool(delta.get("tool_calls") or delta.get("function_call"))


def _summary_incomplete_finish_reason(reason: Any) -> str | None:
    if not isinstance(reason, str) or not reason:
        return None
    normalized = reason.lower().replace("-", "_")
    if any(pattern in normalized for pattern in _SUMMARY_INCOMPLETE_FINISH_REASON_PATTERNS):
        return reason
    return None


def _summary_response_incomplete_reason(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    status_value = response.get("status")
    if isinstance(status_value, str) and status_value.lower() == "incomplete":
        details = response.get("incomplete_details")
        if isinstance(details, dict):
            reason = details.get("reason")
            if isinstance(reason, str) and reason:
                return reason
        return status_value
    reason = _summary_incomplete_finish_reason(response.get("finish_reason"))
    if reason is not None:
        return reason
    choices = response.get("choices")
    if isinstance(choices, list):
        choice = choices[0] if choices else None
        if isinstance(choice, dict):
            reason = _summary_incomplete_finish_reason(choice.get("finish_reason"))
            if reason is not None:
                return reason
    nested_response = response.get("response")
    if isinstance(nested_response, dict):
        return _summary_response_incomplete_reason(nested_response)
    return None


def _raise_incomplete_summary(reason: str) -> None:
    raise RuntimeError(f"Summary model stopped before completing the checkpoint summary: {reason}")


def _stream_payload_has_tool_call(payload: dict[str, Any]) -> bool:
    payload_type = payload.get("type")
    if isinstance(payload_type, str) and "function_call" in payload_type:
        return True
    item = payload.get("item")
    if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}:
        return True
    choices = payload.get("choices")
    return isinstance(choices, list) and any(_choice_has_tool_call(choice) for choice in choices)


def _responses_output_has_tool_call(response: dict[str, Any]) -> bool:
    output = response.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}:
            return True
    return False


def _responses_output_text(response: dict[str, Any]) -> str | None:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = response.get("output")
    if not isinstance(output, list):
        return None
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict) or content_item.get("type") != "output_text":
                continue
            text = content_item.get("text")
            if isinstance(text, str):
                parts.append(text)
    if not parts:
        return None
    return "".join(parts)


def _choice_message_text(choice: Any) -> str | None:
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    return None


def _stream_payload_text(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            return content
        message_content = _choice_message_text(choice)
        if message_content is not None:
            return message_content
    if payload.get("type") == "response.output_text.delta":
        delta = payload.get("delta")
        if isinstance(delta, str):
            return delta
    return None


def _stream_payload_is_control_event(payload: dict[str, Any]) -> bool:
    if "error" in payload or "usage" in payload:
        return True
    return payload.get("done") is True


def _stream_payload_is_known_transport_event(payload: dict[str, Any]) -> bool:
    payload_type = payload.get("type")
    if isinstance(payload.get("choices"), list):
        return True
    if isinstance(payload_type, str) and payload_type.startswith("response."):
        return True
    return _stream_payload_is_control_event(payload)


def _stream_payload_error_source(payload: dict[str, Any]) -> Any | None:
    if payload.get("error") is not None:
        return payload
    payload_type = payload.get("type")
    if payload_type == "error":
        return payload
    if payload_type == "response.failed":
        response = payload.get("response")
        if isinstance(response, dict) and response.get("error") is not None:
            return {"error": response["error"]}
        return payload
    return None


def _append_summary_sse_value(value: str, parts: list[str]) -> bool:
    try:
        payload = json.loads(value.strip())
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    error_source = _stream_payload_error_source(payload)
    if error_source is not None:
        if is_retryable_context_error(error_source, status_code=400):
            raise RetryableContextOverflow("Summary model reported a context-window error")
        raise RuntimeError(f"Summary model provider error: {_completion_error_message(error_source)}")

    incomplete_reason = _summary_response_incomplete_reason(payload)
    if incomplete_reason is not None:
        _raise_incomplete_summary(incomplete_reason)

    is_known_transport_event = _stream_payload_is_known_transport_event(payload)
    saw_tool_call = is_known_transport_event and _stream_payload_has_tool_call(payload)
    if not is_known_transport_event:
        return False
    text = _stream_payload_text(payload)
    if text is not None:
        parts.append(text)
    return saw_tool_call


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


async def extract_text_from_completion_response(response: Any, *, tools_enabled: bool = False) -> str:
    if isinstance(response, dict):
        if _responses_output_has_tool_call(response):
            raise _summary_tool_call_error()
        incomplete_reason = _summary_response_incomplete_reason(response)
        if incomplete_reason is not None:
            _raise_incomplete_summary(incomplete_reason)
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if _choice_has_tool_call(choice):
                raise _summary_tool_call_error()
            value = _choice_message_text(choice)
            if isinstance(value, str) and value.strip():
                return value.strip()
        responses_text = _responses_output_text(response)
        if responses_text and responses_text.strip():
            return responses_text.strip()
    if isinstance(response, StreamingResponse):
        parts: list[str] = []
        saw_tool_call = False
        sse_parser = _SSEDataParser()
        media_type = getattr(response, "media_type", None) or response.headers.get("content-type", "")
        is_sse_response = "text/event-stream" in media_type
        if not is_sse_response:
            await _close_stream_response(response)
            raise RuntimeError("Summary model did not return a structured OpenAI-compatible response")
        try:
            async for chunk in response.body_iterator:
                sse_values, _ = sse_parser.feed(chunk)
                for value in sse_values:
                    saw_tool_call = saw_tool_call or _append_summary_sse_value(value, parts)
            for value in sse_parser.flush()[0]:
                saw_tool_call = saw_tool_call or _append_summary_sse_value(value, parts)
            text = "".join(parts).strip()
            if saw_tool_call:
                raise _summary_tool_call_error()
            if text:
                return text
        finally:
            await _close_stream_response(response)
    if tools_enabled:
        raise RuntimeError("Summary model returned a tool call or no text content while tools were available")
    raise RuntimeError("Summary model did not return text content")


def build_summary_task_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    copied = _copy_summary_task_metadata(metadata or {})
    copied.pop("selected_model_id", None)
    copied.pop("mcp_clients", None)
    copied["task"] = INTERNAL_SUMMARY_TASK
    copied["tools"] = {}
    return copied


def _resolve_summary_model_for_call(summary_model_id: str, *, pipe_function_id: str = PIPE_FUNCTION_ID) -> str:
    if is_generated_wrapper_model_id(summary_model_id, pipe_function_id=pipe_function_id):
        return decode_wrapper_model_id(summary_model_id, expected_pipe_function_id=pipe_function_id).target_model_id
    return summary_model_id


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


SUMMARY_INHERITED_RESPONSE_CONTROL_KEYS = (
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "stop",
)


def strip_summary_inherited_response_controls(body: dict[str, Any]) -> None:
    for key in SUMMARY_INHERITED_RESPONSE_CONTROL_KEYS:
        body.pop(key, None)


def neutralize_summary_tool_choice(body: dict[str, Any]) -> None:
    if body.get("tools") and _is_forced_tool_choice(body.get("tool_choice")):
        body["tool_choice"] = "none"
    elif not body.get("tools"):
        body.pop("tool_choice", None)
    if body.get("functions") and _is_forced_function_call(body.get("function_call")):
        body["function_call"] = "none"
    elif not body.get("functions"):
        body.pop("function_call", None)


def strip_summary_tools_for_retry(body: dict[str, Any]) -> None:
    for key in ("tools", "tool_choice", "functions", "function_call", "parallel_tool_calls"):
        body.pop(key, None)


def build_summary_request_message() -> dict[str, str]:
    return {"role": "user", "content": SUMMARY_PROMPT}


def build_summary_completion_body(
    base_body: dict[str, Any],
    *,
    summary_model_id: str,
    source_messages: list[dict[str, Any]],
    metadata: dict[str, Any],
    pipe_function_id: str = PIPE_FUNCTION_ID,
    summary_tool_policy: SummaryToolPolicy = "fallback_on_tool_call",
) -> dict[str, Any]:
    body = _copy_body_preserving_metadata(base_body)
    body["model"] = _resolve_summary_model_for_call(summary_model_id, pipe_function_id=pipe_function_id)
    body["messages"] = [*copy.deepcopy(source_messages), build_summary_request_message()]
    body["metadata"] = build_summary_task_metadata(metadata)
    body.pop("previous_response_id", None)
    body.pop("response_format", None)
    strip_summary_inherited_response_controls(body)
    neutralize_summary_tool_choice(body)
    if summary_tool_policy == "always_strip":
        strip_summary_tools_for_retry(body)
    return body


async def _generate_summary_text(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    summary_model_id: str,
    source_messages: list[dict[str, Any]],
    base_body: dict[str, Any],
    pipe_function_id: str = PIPE_FUNCTION_ID,
    summary_tool_policy: SummaryToolPolicy = "fallback_on_tool_call",
) -> str:
    summary_metadata = build_summary_task_metadata(metadata)
    inner_request = RequestStateProxy(
        request,
        bypass_filter=True,
        bypass_system_prompt=False,
        metadata=summary_metadata,
    )
    from open_webui.utils.chat import generate_chat_completion

    body = build_summary_completion_body(
        base_body,
        summary_model_id=summary_model_id,
        source_messages=source_messages,
        metadata=metadata,
        pipe_function_id=pipe_function_id,
        summary_tool_policy=summary_tool_policy,
    )
    route = await _resolve_core_chat_model_route(inner_request, str(body.get("model") or ""))
    body["model"] = route.model_id
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
    try:
        return await extract_text_from_completion_response(response, tools_enabled=bool(body.get("tools")))
    except SummaryToolCallError:
        if summary_tool_policy != "fallback_on_tool_call" or (not body.get("tools") and not body.get("functions")):
            raise
        retry_body = copy.deepcopy(body)
        strip_summary_tools_for_retry(retry_body)
        response = await generate_chat_completion(
            inner_request,
            retry_body,
            user=coerce_open_webui_user(user),
            bypass_filter=True,
            bypass_system_prompt=False,
        )
        if is_retryable_context_error(response, status_code=400):
            raise RetryableContextOverflow("Summary model reported a context-window error")
        return await extract_text_from_completion_response(response, tools_enabled=False)


async def _compact_retry_tool_results(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    pipe_function_id: str,
    summary_model_id: str,
    base_body: dict[str, Any],
    messages: list[dict[str, Any]],
    parent_checkpoint: dict[str, Any] | None = None,
    summary_tool_policy: SummaryToolPolicy = "fallback_on_tool_call",
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
) -> tuple[list[dict[str, Any]], bool]:
    cut = select_tool_result_compaction_cut(messages)
    if cut is None:
        return messages, False

    source_messages = copy.deepcopy(cut.summarization_prefix)
    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")
    if not user_id or not _chat_id_supported(chat_id):
        raise UnsupportedCompactionInput(
            "Cannot compact tool history without a durable checkpoint identity",
            code="checkpoint_identity_missing",
        )
    summary_meta = build_checkpoint_summary_meta(
        source_messages,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
    )
    try:
        summary = await _get_or_create_compaction_summary(
            request=request,
            user=user,
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            metadata=metadata,
            summary_model_id=summary_model_id,
            base_body=base_body,
            source_messages=source_messages,
            summary_meta=summary_meta,
            parent_checkpoint=parent_checkpoint,
            summary_tool_policy=summary_tool_policy,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
    except Exception as exc:
        if isinstance(exc, ParentCheckpointExtensionFailed) and (
            isinstance(exc.original, RetryableContextOverflow) or is_retryable_context_error(exc.original)
        ):
            raise UnsupportedCompactionInput(
                "A conversation history with tool results exceeds the summary model context window and cannot be safely compacted",
                code="latest_tool_result_too_large",
            ) from exc
        if isinstance(exc, RetryableContextOverflow) or is_retryable_context_error(exc):
            raise UnsupportedCompactionInput(
                "A conversation history with tool results exceeds the summary model context window and cannot be safely compacted",
                code="latest_tool_result_too_large",
            ) from exc
        raise

    compacted: list[dict[str, Any]] = []
    if cut.preserved_system_message is not None:
        compacted.append(copy.deepcopy(cut.preserved_system_message))
    compacted.append(
        _render_summary_message_from_result(
            summary,
            summary_meta,
            historical_source_messages=source_messages,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
    )
    compacted.extend(copy.deepcopy(cut.tail_messages))
    return compacted, True


async def _heartbeat_checkpoint_claim(store: Any, checkpoint_id: str, claim_token: str) -> None:
    while True:
        await asyncio.sleep(CHECKPOINT_CLAIM_HEARTBEAT_SECONDS)
        try:
            extended = await store.extend_claim(
                checkpoint_id,
                claim_token=claim_token,
                expires_at=int(time.time()) + CHECKPOINT_CLAIM_LEASE_SECONDS,
            )
        except Exception:
            continue
        if not extended:
            return


async def _claim_or_wait_for_checkpoint(
    store: Any,
    *,
    identity: dict[str, str],
    source_message_count: int,
    summary_meta: dict[str, Any],
    claim_token: str,
) -> str | dict[str, Any]:
    """Return the claimed pending checkpoint id, or a ready row produced by another worker."""
    deadline = time.monotonic() + CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS
    while True:
        row = await store.lookup_any(**identity)
        if row is None:
            now = int(time.time())
            pending_row = build_checkpoint_row(
                **identity,
                source_message_count=source_message_count,
                summary_text="",
                summary_meta=summary_meta,
                parent_checkpoint_id=None,
                state="pending",
                claim_token=claim_token,
                claim_expires_at=now + CHECKPOINT_CLAIM_LEASE_SECONDS,
                now=now,
            )
            if await store.claim_pending(pending_row):
                return pending_row["id"]
            continue
        if row.get("state") == "ready":
            with suppress(Exception):
                await store.touch(row["id"])
            return row
        now = int(time.time())
        expires_at = row.get("claim_expires_at")
        if expires_at is None or int(expires_at) <= now:
            if await store.reclaim_pending(
                row["id"],
                claim_token=claim_token,
                expires_at=now + CHECKPOINT_CLAIM_LEASE_SECONDS,
                now=now,
            ):
                return row["id"]
            continue
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for another worker to finish generating this checkpoint summary"
            )
        await asyncio.sleep(CHECKPOINT_PENDING_POLL_SECONDS)


async def _get_or_create_checkpoint_summary(
    *,
    request: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    source_messages: list[dict[str, Any]],
    summary_meta: dict[str, Any],
    summary_factory: Callable[[dict[str, Any] | None], Awaitable[str]],
    parent_checkpoint: dict[str, Any] | None = None,
) -> str:
    profile_hash = compute_profile_hash()
    source_hash = compute_source_hash(source_messages)
    summary_meta = normalize_summary_meta(summary_meta)
    identity = {
        "namespace": CHECKPOINT_NAMESPACE,
        "user_id": user_id,
        "chat_id": chat_id,
        "pipe_function_id": pipe_function_id,
        "profile_hash": profile_hash,
        "source_hash": source_hash,
    }
    lock_key = (CHECKPOINT_NAMESPACE, user_id, chat_id, pipe_function_id, profile_hash, source_hash)
    lock = get_generation_lock(lock_key)

    try:
        async with lock:
            await ensure_checkpoint_table_initialized(request=request)
            store = CheckpointStore()
            claim_token = uuid.uuid4().hex

            claimed = await _claim_or_wait_for_checkpoint(
                store,
                identity=identity,
                source_message_count=len(source_messages),
                summary_meta=summary_meta,
                claim_token=claim_token,
            )
            if isinstance(claimed, dict):
                return CompactionSummaryResult(claimed["summary_text"], checkpoint=claimed)
            checkpoint_id = claimed

            heartbeat = asyncio.create_task(_heartbeat_checkpoint_claim(store, checkpoint_id, claim_token))
            try:
                if parent_checkpoint is not None:
                    parent_count = int(parent_checkpoint.get("source_message_count") or 0)
                    if (
                        parent_count <= 0
                        or parent_count > len(source_messages)
                        or compute_source_hash(source_messages[:parent_count])
                        != parent_checkpoint.get("source_hash")
                    ):
                        raise UnsupportedCompactionInput(
                            "Parent checkpoint cannot be applied safely because its source boundary is invalid",
                            code="unsafe_checkpoint_parent",
                        )
                    parent = parent_checkpoint
                else:
                    parent = await store.find_longest_parent(
                        namespace=CHECKPOINT_NAMESPACE,
                        user_id=user_id,
                        chat_id=chat_id,
                        pipe_function_id=pipe_function_id,
                        profile_hash=profile_hash,
                        source_messages=source_messages,
                    )
                try:
                    summary_text = await summary_factory(parent)
                except Exception as exc:
                    if parent:
                        raise ParentCheckpointExtensionFailed(parent, exc) from exc
                    raise
                completed = await store.complete_pending(
                    checkpoint_id,
                    claim_token=claim_token,
                    summary_text=summary_text,
                    parent_checkpoint_id=parent.get("id") if parent else None,
                )
                if completed is not None:
                    return CompactionSummaryResult(completed["summary_text"], checkpoint=completed)
                existing = await store.lookup_ready(**identity)
                if existing is not None:
                    with suppress(Exception):
                        await store.touch(existing["id"])
                    return CompactionSummaryResult(existing["summary_text"], checkpoint=existing)
                raise RuntimeError("Checkpoint claim was lost before the generated summary could be stored")
            except (asyncio.CancelledError, Exception):
                # asyncio.CancelledError is outside Exception on Python 3.11+.
                with suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(store.release_claim(checkpoint_id, claim_token=claim_token))
                raise
            finally:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
    finally:
        release_generation_lock(lock_key, lock)


async def _get_or_create_compaction_summary(
    *,
    request: Any,
    user: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    metadata: dict[str, Any],
    summary_model_id: str,
    base_body: dict[str, Any],
    source_messages: list[dict[str, Any]],
    summary_meta: dict[str, Any],
    parent_checkpoint: dict[str, Any] | None = None,
    summary_tool_policy: SummaryToolPolicy = "fallback_on_tool_call",
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
) -> str:
    summary_source_prefix = copy.deepcopy(source_messages)

    async def summary_factory(parent: dict[str, Any] | None) -> str:
        if parent:
            parent_count = int(parent.get("source_message_count") or 0)
            source = [
                render_summary_message_from_checkpoint(
                    parent,
                    historical_source_messages=summary_source_prefix[:parent_count],
                    historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                    historical_message_excerpt_count=historical_message_excerpt_count,
                ),
                *copy.deepcopy(summary_source_prefix[parent_count:]),
            ]
        else:
            source = copy.deepcopy(summary_source_prefix)
        return await _generate_summary_text(
            request=request,
            user=user,
            metadata=metadata,
            summary_model_id=summary_model_id,
            source_messages=source,
            base_body=base_body,
            pipe_function_id=pipe_function_id,
            summary_tool_policy=summary_tool_policy,
        )

    return await _get_or_create_checkpoint_summary(
        request=request,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        source_messages=source_messages,
        summary_meta=summary_meta,
        summary_factory=summary_factory,
        parent_checkpoint=parent_checkpoint,
    )


async def _lookup_ready_checkpoint_for_source(
    *,
    request: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    source_messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    await ensure_checkpoint_table_initialized(request=request)
    return await CheckpointStore().lookup_ready(
        namespace=CHECKPOINT_NAMESPACE,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        profile_hash=compute_profile_hash(),
        source_hash=compute_source_hash(source_messages),
    )


async def _find_reusable_checkpoint_for_source(
    *,
    store: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    profile_hash: str,
    source_messages: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    source_hash = compute_source_hash(source_messages)
    existing = await store.lookup_ready(
        namespace=CHECKPOINT_NAMESPACE,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        profile_hash=profile_hash,
        source_hash=source_hash,
    )
    if existing:
        return "exact", existing

    parent = await store.find_longest_parent(
        namespace=CHECKPOINT_NAMESPACE,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        profile_hash=profile_hash,
        source_messages=source_messages,
    )
    if parent is not None:
        return "parent", parent
    return None


async def _body_reusable_checkpoint_match(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
) -> ReusableCheckpointMatch | None:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")
    if not user_id or not _chat_id_supported(chat_id):
        return None

    tool_cut = select_tool_result_compaction_cut(messages)
    cut = select_safe_message_cut(messages)
    if (tool_cut is None or not tool_cut.summarization_prefix) and not cut.summarization_prefix:
        return None

    profile_hash = compute_profile_hash()
    await ensure_checkpoint_table_initialized(request=request)
    store = CheckpointStore()

    if tool_cut is not None and tool_cut.summarization_prefix:
        tool_match = await _find_reusable_checkpoint_for_source(
            store=store,
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            profile_hash=profile_hash,
            source_messages=tool_cut.summarization_prefix,
        )
        if tool_match is not None:
            kind, checkpoint = tool_match
            return ReusableCheckpointMatch(
                kind=kind,
                source_message_count=int(checkpoint.get("source_message_count") or tool_cut.source_message_count),
                source_kind="tool",
            )

    if not cut.summarization_prefix:
        return None

    message_match = await _find_reusable_checkpoint_for_source(
        store=store,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        profile_hash=profile_hash,
        source_messages=cut.summarization_prefix,
    )
    if message_match is not None:
        kind, checkpoint = message_match
        return ReusableCheckpointMatch(
            kind=kind,
            source_message_count=int(checkpoint.get("source_message_count") or cut.source_message_count),
            source_kind="message",
        )
    return None


async def _compact_body_with_reusable_checkpoint(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    match: ReusableCheckpointMatch,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
) -> tuple[dict[str, Any], bool]:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return body, False

    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")
    if not user_id or not _chat_id_supported(chat_id):
        return body, False

    candidates: list[tuple[str, MessageCut | ToolResultCompactionCut, list[dict[str, Any]]]] = []
    tool_cut = select_tool_result_compaction_cut(messages)
    if tool_cut is not None and tool_cut.summarization_prefix and match.source_kind == "tool":
        candidates.append(("tool", tool_cut, tool_cut.summarization_prefix))

    cut = select_safe_message_cut(messages)
    if cut.summarization_prefix and match.source_kind == "message":
        candidates.append(("message", cut, cut.summarization_prefix))

    await ensure_checkpoint_table_initialized(request=request)
    store = CheckpointStore()
    profile_hash = compute_profile_hash()

    for _kind, candidate_cut, source_messages in candidates:
        checkpoint_match = await _find_reusable_checkpoint_for_source(
            store=store,
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            profile_hash=profile_hash,
            source_messages=source_messages,
        )
        if checkpoint_match is None:
            continue
        _, checkpoint = checkpoint_match

        with suppress(Exception):
            await store.touch(str(checkpoint["id"]))

        compacted = _copy_body_preserving_metadata(body)
        if isinstance(candidate_cut, ToolResultCompactionCut):
            message_cut = MessageCut(
                preserved_system_message=copy.deepcopy(candidate_cut.preserved_system_message),
                summarization_prefix=copy.deepcopy(candidate_cut.summarization_prefix),
                tail_messages=copy.deepcopy(candidate_cut.tail_messages),
                source_message_count=candidate_cut.source_message_count,
            )
        else:
            message_cut = candidate_cut
        compacted["messages"] = replace_prefix_with_parent_checkpoint_and_delta(
            message_cut,
            checkpoint,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
        compacted.pop("previous_response_id", None)
        return compacted, True

    raise RuntimeError("Reusable checkpoint match disappeared before compaction")


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


async def _get_target_db_model_record(target_model_id: str) -> Any:
    try:
        from open_webui.models.models import Models

        return await Models.get_model_by_id(target_model_id)
    except Exception:
        return TARGET_MODEL_RECORD_UNKNOWN


def _target_record_base_model_id(model_info: Any) -> str | None:
    if model_info is None or model_info is TARGET_MODEL_RECORD_UNKNOWN:
        return None
    if isinstance(model_info, dict):
        base_model_id = model_info.get("base_model_id")
    else:
        base_model_id = getattr(model_info, "base_model_id", None)
    return base_model_id if isinstance(base_model_id, str) and base_model_id else None


def _custom_model_fallback_enabled() -> bool:
    try:
        from open_webui.env import ENABLE_CUSTOM_MODEL_FALLBACK

        return bool(ENABLE_CUSTOM_MODEL_FALLBACK)
    except Exception:
        return False


def _custom_model_fallback_model_id(request: Any, models: dict[str, Any]) -> str | None:
    if not _custom_model_fallback_enabled():
        return None
    config = getattr(getattr(request, "app", None), "state", None)
    config = getattr(config, "config", None)
    default_models = str(getattr(config, "DEFAULT_MODELS", None) or "").split(",")
    fallback_model_id = default_models[0].strip() if default_models and default_models[0] else None
    if fallback_model_id and fallback_model_id in models:
        return fallback_model_id
    return None


def _global_model_access_bypass_enabled() -> bool:
    try:
        from open_webui.env import BYPASS_MODEL_ACCESS_CONTROL

        return bool(BYPASS_MODEL_ACCESS_CONTROL)
    except Exception:
        return False


async def _resolve_core_chat_model_route(request: Any, model_id: str) -> CoreChatModelRoute:
    models = await _model_dict_from_request(request)
    if model_id not in models:
        return CoreChatModelRoute(model_id=model_id)
    model_info = await _get_target_db_model_record(model_id)
    base_model_id = _target_record_base_model_id(model_info)
    if base_model_id and base_model_id not in models:
        fallback_model_id = _custom_model_fallback_model_id(request, models)
        if fallback_model_id:
            return CoreChatModelRoute(model_id=fallback_model_id)
    return CoreChatModelRoute(model_id=model_id)


async def _validate_chat_completion_runtime_model_access(
    *,
    request: Any,
    user: Any,
    model_id: str,
) -> None:
    # Mirrors utils.chat.generate_chat_completion's access gate, which this
    # pipe bypasses when forwarding to avoid re-running filters/wrappers.
    user_model = coerce_open_webui_user(user)
    if _global_model_access_bypass_enabled() or getattr(user_model, "role", None) != "user":
        return
    models = await _model_dict_from_request(request)
    model = models.get(model_id)
    if model is None:
        raise HTTPException(status_code=403, detail="Model not found")
    try:
        from open_webui.utils.models import check_model_access

        await check_model_access(user_model, model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=403, detail="Model not found") from exc


async def _validate_target_access(
    *,
    target_model_id: str,
    request: Any,
    user: Any,
    pipe_function_id: str = PIPE_FUNCTION_ID,
) -> None:
    models = await _model_dict_from_request(request)
    model = models.get(target_model_id)
    if model is None or _is_arena_model(model) or is_generated_wrapper_model_id(
        target_model_id,
        pipe_function_id=pipe_function_id,
    ):
        raise HTTPException(status_code=403, detail="Model not found")

    model_info = await _get_target_db_model_record(target_model_id)
    base_model_id = _target_record_base_model_id(model_info)
    if base_model_id and base_model_id not in models and _custom_model_fallback_model_id(request, models) is None:
        raise HTTPException(status_code=403, detail="Model not found")

    user_model = coerce_open_webui_user(user)
    if _should_bypass_target_access_check(user_model):
        return

    if getattr(user_model, "role", None) == "admin" and model_info is None:
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

        def process_payload(payload: dict[str, Any]) -> None:
            nonlocal finish_reason, provider_error, usage
            if payload.get("error"):
                if is_retryable_context_error(payload, status_code=400):
                    raise RetryableContextOverflow(str(payload.get("error")))
                provider_error = _error_response(_completion_error_message(payload), code="provider_error")
                return
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

        media_type = getattr(response, "media_type", None) or response.headers.get("content-type", "")
        is_sse_response = "text/event-stream" in media_type
        sse_parser = _SSEJSONEventParser()
        try:
            async for raw_chunk in response.body_iterator:
                chunk = _coerce_stream_chunk(raw_chunk)
                if is_sse_response:
                    events = sse_parser.feed(chunk)
                    for payload in events:
                        process_payload(payload)
                    continue
                events = extract_sse_json_events(chunk)
                if not events:
                    if _chunk_is_sse_done_only(chunk):
                        continue
                    parts.append(chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk))
                    continue
                for payload in events:
                    process_payload(payload)
            if is_sse_response:
                for payload in sse_parser.flush():
                    process_payload(payload)
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
    summary_tool_policy: SummaryToolPolicy = "fallback_on_tool_call",
) -> tuple[dict[str, Any], bool]:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return body, False

    tool_cut = select_tool_result_compaction_cut(messages)
    cut = select_safe_message_cut(messages)
    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")

    if tool_cut is not None:
        try:
            compacted_messages, did_compact_tools = await _compact_retry_tool_results(
                request=request,
                user=user,
                metadata=metadata,
                pipe_function_id=pipe_function_id,
                summary_model_id=summary_model_id,
                base_body=body,
                messages=messages,
                summary_tool_policy=summary_tool_policy,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
            )
        except UnsupportedCompactionInput as exc:
            if exc.code != "latest_tool_result_too_large":
                raise
            if not cut.summarization_prefix:
                raise
            if not user_id or not _chat_id_supported(chat_id):
                raise UnsupportedCompactionInput(
                    "Cannot compact tool history without a durable checkpoint identity",
                    code="checkpoint_identity_missing",
                ) from exc
            history_summary_meta = build_checkpoint_summary_meta(
                cut.summarization_prefix,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
            )
            await _get_or_create_compaction_summary(
                request=request,
                user=user,
                user_id=user_id,
                chat_id=chat_id,
                pipe_function_id=pipe_function_id,
                metadata=metadata,
                summary_model_id=summary_model_id,
                base_body=body,
                source_messages=cut.summarization_prefix,
                summary_meta=history_summary_meta,
                summary_tool_policy=summary_tool_policy,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
            )
            history_checkpoint = await _lookup_ready_checkpoint_for_source(
                request=request,
                user_id=user_id,
                chat_id=chat_id,
                pipe_function_id=pipe_function_id,
                source_messages=cut.summarization_prefix,
            )
            if history_checkpoint is None:
                raise RuntimeError("History checkpoint was not available after creation")
            compacted_messages, did_compact_tools = await _compact_retry_tool_results(
                request=request,
                user=user,
                metadata=metadata,
                pipe_function_id=pipe_function_id,
                summary_model_id=summary_model_id,
                base_body=body,
                messages=messages,
                parent_checkpoint=history_checkpoint,
                summary_tool_policy=summary_tool_policy,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
            )
        if did_compact_tools:
            compacted = _copy_body_preserving_metadata(body)
            compacted["messages"] = compacted_messages
            compacted.pop("previous_response_id", None)
            return compacted, True

    if not cut.summarization_prefix:
        return body, False

    latest_user = next((m for m in reversed(cut.tail_messages) if m.get("role") == "user"), None)
    if latest_user is None:
        raise UnsupportedCompactionInput("Cannot compact safely without retaining the active latest user message")

    if not user_id or not _chat_id_supported(chat_id):
        return body, False

    summary_meta = build_checkpoint_summary_meta(
        cut.summarization_prefix,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
    )

    compacted = _copy_body_preserving_metadata(body)
    try:
        summary_text = await _get_or_create_compaction_summary(
            request=request,
            user=user,
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            metadata=metadata,
            summary_model_id=summary_model_id,
            base_body=body,
            source_messages=cut.summarization_prefix,
            summary_meta=summary_meta,
            summary_tool_policy=summary_tool_policy,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
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
    compacted.pop("previous_response_id", None)
    return compacted, True


async def _compact_task_body_with_reusable_checkpoint(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    match: ReusableCheckpointMatch,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
) -> tuple[dict[str, Any], bool]:
    source_body = _task_history_source_body_for_compaction(body, metadata)
    if source_body is None:
        return body, False
    compacted_source, compacted = await _compact_body_with_reusable_checkpoint(
        request=request,
        user=user,
        metadata=metadata,
        body=source_body,
        pipe_function_id=pipe_function_id,
        match=match,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
    )
    if not compacted:
        return body, False
    rebuilt = await _rebuild_task_body_from_compacted_history(
        request=request,
        user=user,
        base_body=body,
        metadata=metadata,
        compacted_history_messages=compacted_source["messages"],
    )
    if rebuilt is None:
        raise RuntimeError("Task prompt could not be rebuilt from compacted history")
    return rebuilt, True


async def _compact_task_body(
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
    summary_tool_policy: SummaryToolPolicy,
) -> tuple[dict[str, Any], bool]:
    source_body = _task_history_source_body_for_compaction(body, metadata)
    if source_body is None:
        return body, False
    compacted_source, compacted = await _compact_body(
        request=request,
        user=user,
        metadata=metadata,
        body=source_body,
        pipe_function_id=pipe_function_id,
        target_model_id=target_model_id,
        summary_model_id=summary_model_id,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        summary_tool_policy=summary_tool_policy,
    )
    if not compacted:
        return body, False
    rebuilt = await _rebuild_task_body_from_compacted_history(
        request=request,
        user=user,
        base_body=body,
        metadata=metadata,
        compacted_history_messages=compacted_source["messages"],
    )
    if rebuilt is None:
        raise RuntimeError("Task prompt could not be rebuilt from compacted history")
    return rebuilt, True


class Pipe:
    class Valves(BaseModel):
        model_name_prefix: str = Field(
            default="Compact: ",
            description="Display prefix for generated compact wrapper models.",
        )
        include_model_patterns: str = Field(
            default="",
            description="Comma-separated shell-style patterns for target model id/name. Empty wraps all eligible models; when set, only matches are wrapped.",
        )
        exclude_model_patterns: str = Field(
            default="",
            description="Comma-separated shell-style patterns for target model id/name. Empty excludes nothing; applied after include, matches are not wrapped.",
        )
        summary_model: str = Field(
            default="",
            description="Optional Open WebUI model for summarization. Recommended empty: use the same underlying target model this wrapper calls, preserving quality and prompt-cache reuse for cost efficiency.",
        )
        trigger_total_tokens: int = Field(
            default=100000,
            ge=1,
            description="Usage-token threshold that starts compaction. Requires the provider/Open WebUI response to report accurate token usage.",
        )
        force_include_usage: bool = Field(
            default=True,
            description="Default-on convenience setting: add stream_options.include_usage=true to streaming target requests so supporting providers return usage even if per-model usage was not enabled. Disable to manage usage per model.",
        )
        compact_task_prompts_from_task_body: bool = Field(
            default=False,
            description=(
                "For supported Open WebUI task requests only, rebuild the task prompt from metadata.task_body "
                "so checkpoints can be reused. If inlet/pipeline filters rewrite body.messages without "
                "also updating metadata.task_body, those rewrites are not included in the rebuilt task prompt."
            ),
        )
        summary_tool_policy: SummaryToolPolicy = Field(
            default="fallback_on_tool_call",
            description=(
                "Controls summary requests for tool-enabled chats. Forced tool_choice/function_call is disabled in all modes. "
                "Recommended default fallback_on_tool_call keeps tool definitions on the first request to preserve "
                "provider-visible shape and maximize prompt-cache reuse opportunities, then retries without tools only if the summary model emits a tool call. "
                "always_strip removes tools before the first request. error_on_tool_call keeps tools and fails on tool calls."
            ),
        )
        historical_message_excerpt_bytes: int = Field(
            default=DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
            ge=1,
            description="Maximum UTF-8 bytes per saved historical user-message excerpt. New checkpoints store middle-truncated excerpts; saved excerpts are reused unchanged.",
        )
        historical_message_excerpt_count: int = Field(
            default=DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
            ge=0,
            description="Maximum number of recent historical user-message excerpts saved in new checkpoints and inserted into compacted context. Set 0 to disable excerpts.",
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

        pipe_function_id = runtime_pipe_function_id(self)
        provider_cache_attrs = _enabled_provider_model_cache_attrs(state)
        initial_provider_caches = {attr: getattr(state, attr, None) for attr in provider_cache_attrs}
        model_candidates = _iter_cache_models_from_state(state)
        pending_provider_cache_attrs = _provider_model_cache_refresh_pending_attrs(provider_cache_attrs)
        if pending_provider_cache_attrs and not _provider_model_caches_ready(
            state,
            initial_provider_caches,
            pending_provider_cache_attrs,
        ):
            model_candidates = await _wait_for_provider_model_caches(
                state,
                initial_provider_caches=initial_provider_caches,
            )
        targets = filter_target_models(model_candidates, self.valves, pipe_function_id=pipe_function_id)
        await sync_wrapper_model_records(pipe_function_id=pipe_function_id, target_models=targets, valves=self.valves)

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
        pipe_function_id = runtime_pipe_function_id(self)
        try:
            identity = decode_wrapper_model_id(selected_wrapper_id, expected_pipe_function_id=pipe_function_id)
        except ValueError as exc:
            return _error_response(str(exc), code="invalid_wrapper_id")

        user = __user__ or {}
        metadata = _copy_metadata_preserving_references(__metadata__ or {})
        metadata_task_body = _task_body_from_metadata(metadata)
        chat_id = metadata.get("chat_id")
        if not chat_id and metadata_task_body is not None:
            chat_id = metadata_task_body.get("chat_id")
        if chat_id and not metadata.get("chat_id"):
            metadata["chat_id"] = chat_id
        message_id = metadata.get("message_id") or metadata.get("user_message_id")

        try:
            await _validate_target_access(
                target_model_id=identity.target_model_id,
                request=__request__,
                user=user,
                pipe_function_id=pipe_function_id,
            )
        except Exception:
            return _error_response("Model not found", code="model_access_denied")

        models = await _model_dict_from_request(__request__)
        try:
            summary_model_id = validate_summary_model_id(self.valves.summary_model, identity.target_model_id, models)
        except ValueError as exc:
            return _error_response(str(exc), code="invalid_summary_model")

        inner = _copy_body_preserving_metadata(body)
        target_route = await _resolve_core_chat_model_route(
            __request__,
            identity.target_model_id,
        )
        if target_route.model_id != identity.target_model_id:
            try:
                await _validate_chat_completion_runtime_model_access(
                    request=__request__,
                    user=user,
                    model_id=target_route.model_id,
                )
            except Exception:
                return _error_response("Model not found", code="model_access_denied")
        inner["model"] = target_route.model_id
        inner["metadata"] = _copy_metadata_preserving_references(metadata)
        if is_streaming and self.valves.force_include_usage:
            inner = inject_stream_usage_options(inner, force_include_usage=True)

        is_summary_task = _normalized_task_name(metadata.get("task")) == INTERNAL_SUMMARY_TASK
        supported_context = _chat_id_supported(chat_id) and not is_summary_task
        task_source_body = (
            _task_history_source_body_for_compaction(inner, metadata)
            if supported_context and self.valves.compact_task_prompts_from_task_body
            else None
        )
        checkpoint_lookup_body = task_source_body if task_source_body is not None else inner

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
            try:
                reusable_checkpoint_match = await _body_reusable_checkpoint_match(
                    request=__request__,
                    user=user,
                    metadata=metadata,
                    body=checkpoint_lookup_body,
                    pipe_function_id=identity.pipe_function_id,
                )
            except Exception as exc:
                return _error_response(
                    f"Failed to access auto-compaction checkpoints: {exc}",
                    code="checkpoint_unavailable",
                )
        should_compact = usage_should_compact or reusable_checkpoint_match is not None
        reusable_checkpoint_only = (
            reusable_checkpoint_match
            if reusable_checkpoint_match is not None and not usage_should_compact
            else None
        )

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
                    if reusable_checkpoint_only is not None and not compacted_once:
                        if task_source_body is not None:
                            candidate, compacted = await _compact_task_body_with_reusable_checkpoint(
                                request=__request__,
                                user=user,
                                metadata=metadata,
                                body=candidate,
                                pipe_function_id=identity.pipe_function_id,
                                match=reusable_checkpoint_only,
                                historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                                historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                            )
                        else:
                            candidate, compacted = await _compact_body_with_reusable_checkpoint(
                                request=__request__,
                                user=user,
                                metadata=metadata,
                                body=candidate,
                                pipe_function_id=identity.pipe_function_id,
                                match=reusable_checkpoint_only,
                                historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                                historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                            )
                    else:
                        if task_source_body is not None:
                            candidate, compacted = await _compact_task_body(
                                request=__request__,
                                user=user,
                                metadata=metadata,
                                body=candidate,
                                pipe_function_id=identity.pipe_function_id,
                                target_model_id=identity.target_model_id,
                                summary_model_id=summary_model_id,
                                historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                                historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                                summary_tool_policy=self.valves.summary_tool_policy,
                            )
                        else:
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
                                summary_tool_policy=self.valves.summary_tool_policy,
                            )
                    compacted_once = compacted_once or compacted
                    if emit_progress_status:
                        if compacted:
                            await emit_compaction_status(
                                __event_emitter__,
                                action="compacted",
                                description="Compacted chat history is ready for the target model",
                                done=True,
                            )
                            summary_text = extract_compaction_summary_text_from_messages(candidate.get("messages"))
                            await emit_compaction_summary_embed(
                                __event_emitter__,
                                summary_text=summary_text,
                            )
                        else:
                            await emit_compaction_status(
                                __event_emitter__,
                                action="skipped",
                                description="Could not compact chat history safely; forwarding unchanged",
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
