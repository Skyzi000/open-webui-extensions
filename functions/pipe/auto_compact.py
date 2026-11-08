"""
title: Auto Compact
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Manifold Pipe that wraps Open WebUI models, compacts long chats, and persists durable checkpoint summaries.
version: 0.8.8
license: MIT
required_open_webui_version: 0.9.6
"""

# fmt: off

from __future__ import annotations

import asyncio
import codecs
import copy
import hashlib
import hmac
import html
import inspect
import json
import logging
import math
import random
import re
import secrets
import shlex
import threading
import time
import uuid
import weakref
from array import array
from bisect import bisect_left, bisect_right
from collections import deque
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field as dataclass_field, replace
from enum import StrEnum
from functools import lru_cache
from html.parser import HTMLParser
from itertools import islice
from types import MappingProxyType, SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Literal, TypeAlias, assert_never

from fastapi import HTTPException
from open_webui.constants import TASKS
from open_webui.utils.misc import sanitize_text_for_db
from pydantic import BaseModel, Field, field_validator
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
    exists,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

try:
    from open_webui.utils.chat_variables import render_chat_variables as _render_chat_variables
except ImportError:
    _render_chat_variables = None

try:
    from open_webui.utils.chat_variables import get_chat_variables_schema as _get_chat_variables_schema
except ImportError:
    _get_chat_variables_schema = None

try:
    import markdown as _markdown_mod
except Exception:
    _markdown_mod = None

try:
    import regex as _REGEX
except ImportError:
    _REGEX = None


PIPE_FUNCTION_ID = "auto_compact"
CHECKPOINT_NAMESPACE = "skyzi000.open_webui_extensions.auto_compaction_pipe"
CHECKPOINT_GENERATION_LEASE_NAMESPACE = f"{CHECKPOINT_NAMESPACE}.generation_lease"
USAGE_ANCHOR_NAMESPACE = f"{CHECKPOINT_NAMESPACE}.usage_anchor"
CHECKPOINT_GENERATION_LEASE_SOURCE_HASH = "generation-lease"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_TABLE_NAME = "skyzi000_owui_ext_autocompact_checkpoint_v1"
CHECKPOINT_LOOKUP_UQ = "skyzi000_owui_ext_accp_v1_lookup_uq"
CHECKPOINT_PREFIX_IDX = "skyzi000_owui_ext_accp_v1_prefix_idx"
CHECKPOINT_RECENT_IDX = "skyzi000_owui_ext_accp_v1_recent_idx"
REQUEST_STATE_SCHEMA_READY_KEY = "_auto_compact_checkpoint_schema_ready_v1"
REQUEST_STATE_REF_STORE_KEY = "_skyzi000_auto_compact_ref_exec_v1"
REQUEST_STATE_REF_PREVIEW_CACHE_KEY = "_auto_compact_ref_preview_cache"
CHECKPOINT_CLAIM_LEASE_SECONDS = 90
CHECKPOINT_CLAIM_HEARTBEAT_SECONDS = 30
CHECKPOINT_PENDING_POLL_SECONDS = 0.25
CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS = 300.0
TOKEN_ESTIMATOR_VERSION = "message-sanitized-media-json-v3"
USAGE_ANCHOR_FORMAT_VERSION = 1
USAGE_ANCHOR_PROFILE_FAMILY = "provider-input-anchor-v1"
USAGE_ANCHOR_SOURCE_FAMILY = "assistant-message-anchor-v1"
USAGE_ANCHOR_FINGERPRINT_FAMILY = "provider-input-prefix-v1"
USAGE_ANCHOR_SHAPING_PROFILE_FAMILY = "provider-shaping-profile-v1"
USAGE_ANCHOR_SYSTEM_CLOCK_SENTINELS = (
    ("{{CURRENT_DATE}}", "<auto-compact-current-date>"),
    ("{{CURRENT_TIME}}", "<auto-compact-current-time>"),
    ("{{CURRENT_DATETIME}}", "<auto-compact-current-datetime>"),
    ("{{CURRENT_WEEKDAY}}", "<auto-compact-current-weekday>"),
)
MESSAGE_TOKEN_OVERHEAD = 4
REQUEST_TOKEN_OVERHEAD = 3
MESSAGE_TOKEN_ESTIMATE_CACHE_MAX_ENTRIES = 8192
MESSAGE_TOKEN_EXACT_ENCODE_MAX_BYTES = 64 * 1024
MESSAGE_TOKEN_SAMPLE_MAX_BYTES = 16 * 1024
MESSAGE_TOKEN_IMAGE_OVERHEAD = 1000
REF_TEXT_HASH_CHUNK_CHARS = 16 * 1024
REF_EXEC_TOOL_NAME = "auto_compact_ref_exec"
REF_EXEC_COMMAND_MAX_BYTES = 1_024
REF_EXEC_RESPONSE_MAX_BYTES = 65_536
REF_EXEC_TAIL_MAX_BYTES = 8 * 1024 * 1024
REF_EXEC_USAGE_ERROR = (
    "Error: usage: auto_compact_ref_exec(command). Expected REF: "
    "tool:<64 hex> or history:accp_<64 hex>"
)
REF_EXEC_REGEX_BUDGET_SECONDS = 2.0
REF_EXEC_COMMANDS = ("cat", "grep", "head", "ls", "sed", "stat", "tail", "wc")
_REF_BINDING_LABEL_HMAC_KEY = secrets.token_bytes(32)
_REF_SHARED_REGISTRY_WARNED_REQUESTS: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()
_REF_SHARED_REGISTRY_WARNED_LOCK = threading.Lock()
_KNOWN_SEMANTIC_MESSAGE_KEYS = frozenset(
    {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "function_call",
        "output",
    }
)
BODY_TOKEN_EXTRA_KEYS = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "parallel_tool_calls",
)
INTERNAL_SUMMARY_TASK = "auto_compaction_summary"
# Open WebUI Core context compaction runs before pipes and truncates the message
# chain this wrapper uses for checkpoint identity. Treat its task re-entry as
# unsupported; do not passthrough because that looks like supported coexistence.
OFFICIAL_CONTEXT_COMPACTION_TASK = "context_compaction"
CORE_CONTEXT_COMPACTION_ENABLE_CONFIG_KEY = "chat.context_compaction.enable"
CORE_CONTEXT_COMPACTION_CONFLICT_MESSAGE = (
    "Auto Compact cannot be used while Open WebUI Core context compaction is enabled. "
    "Disable Open WebUI Core context compaction (ENABLE_CONTEXT_COMPACTION / "
    "chat.context_compaction.enable) before using this Pipe, or use the target model directly."
)
CHECKPOINT_STORE_UNAVAILABLE_MESSAGE = ("Auto-compaction could not access its checkpoint store. Please retry.")


TEMP_CHAT_PREFIXES = ("temporary:", "local:", "channel:")
SUMMARY_FORMAT_FAMILY = "compact-user-summary-v1"
SOURCE_HASH_FAMILY = "canonical-json-v1"
PROFILE_HASH_FAMILY = "checkpoint-profile-v1"
MAX_CONTEXT_RETRY_ATTEMPTS = 2
DEFAULT_TRIGGER_INPUT_TOKENS = 180000
DEFAULT_SOFT_TRIGGER_RATIO = 0.5
DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES = 512
DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT = 32
SUMMARY_META_FORMAT_VERSION_KEY = "summary_meta_format_version"
SUMMARY_META_FORMAT_VERSION = 1
SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY = "historical_user_messages"
SUMMARY_META_HISTORICAL_USER_MESSAGES_FORMAT_VERSION = 1
SUMMARY_META_HISTORY_REF_KEY = "history_ref"
HISTORY_REF_FORMAT = "canonical-history-jsonl-v1"
HISTORY_REF_LOGICAL_FORMAT = "canonical-history-jsonl-v2-logical"
PROVIDER_MODEL_CACHE_WAIT_DEFAULT_TIMEOUT_SECONDS = 10.0
PROVIDER_MODEL_CACHE_WAIT_POLL_SECONDS = 0.05
OLLAMA_PROVIDER_MODEL_CACHE_REFRESH_REQUESTS = 2
SummaryToolPolicy = Literal["always_strip", "fallback_on_tool_call", "error_on_tool_call"]
RefKind = Literal["history", "tool"]
CORE_FUNCTION_MODEL_LISTING_COROUTINE = ("get_function_models", "open_webui/functions.py")
PROVIDER_MODEL_CACHE_REFRESH_COROUTINES = {
    "OPENAI_MODELS": (("fetch_openai_models", "open_webui/utils/models.py"),),
    "OLLAMA_MODELS": (("fetch_ollama_models", "open_webui/utils/models.py"),),
}
PROVIDER_MODEL_CACHE_ENABLE_FLAGS = {
    "OPENAI_MODELS": "ENABLE_OPENAI_API",
    "OLLAMA_MODELS": "ENABLE_OLLAMA_API",
}
PROVIDER_MODEL_CACHE_CONFIG_KEYS = {
    "OPENAI_MODELS": "openai.enable",
    "OLLAMA_MODELS": "ollama.enable",
}
ARENA_ENABLE_CONFIG_KEY = "evaluation.arena.enable"
ARENA_MODELS_CONFIG_KEY = "evaluation.arena.models"
DEFAULT_ARENA_MODEL_ID = "arena-model"
MODEL_LISTING_CONFIG_KEYS = (
    *PROVIDER_MODEL_CACHE_CONFIG_KEYS.values(),
    ARENA_ENABLE_CONFIG_KEY,
    ARENA_MODELS_CONFIG_KEY,
)
TIKTOKEN_ENCODING_CONFIG_KEY = "rag.tiktoken_encoding_name"
CONFIG_VALUE_MISSING = object()
MISSING_CORE_REQUEST = object()
TARGET_MODEL_RECORD_UNKNOWN = object()
REF_REGISTRY_ENTRY_MISSING = object()
AUTO_COMPACTION_TARGET_HIDDEN_META_KEY = "auto_compaction_target_hidden_by"
TARGET_MODEL_VISIBILITY_LOCKS: dict[str, asyncio.Lock] = {}
# Context-local guard set while AutoCompact is mid target file-context
# injection. It follows genuine re-entry while keeping concurrent multi-model
# sibling tasks independent even though Core gives them the same request.
AUTO_COMPACT_FILE_CONTEXT_INJECTION_ACTIVE: ContextVar[bool] = ContextVar(
    "auto_compact_file_context_injection_active",
    default=False,
)
AUTO_COMPACT_TIKTOKEN_ENCODING_STATE_KEY = "_auto_compact_tiktoken_encoding_name"
AUTO_COMPACT_TIKTOKEN_ENCODING_LOADED_STATE_KEY = "_auto_compact_tiktoken_encoding_loaded"
PREFIX_FILE_FINGERPRINT_RESOLVER_STATE_KEY = "_auto_compact_prefix_file_fingerprint_resolver_cache"
PREFIX_FILE_FINGERPRINT_FAMILY = "prefix-file-fingerprint-v1"
PREFIX_FILE_FINGERPRINT_RESOLVER_DB_CHAIN_ATTR = "_auto_compact_db_chain"
PREFIX_FILE_FINGERPRINT_RESOLVER_FROZEN_ATTR = "_auto_compact_frozen_fingerprint"
LOG = logging.getLogger(__name__)
REF_EXEC_TOOL_SPEC = MappingProxyType(
    {
        "type": "function",
        "function": MappingProxyType(
            {
                "name": REF_EXEC_TOOL_NAME,
                "description": (
                    "Read externalized content in this chat. Oversized tool results include a bounded head/tail preview and a tool:<64 hex> ref; compacted history uses history:accp_<64 hex>. "
                    "A <auto_compact_ref_truncated> marker embeds a next command to read the omitted span; follow continuation commands across pages when needed. Commands: ls [tool|history]; stat REF; "
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
COMPACTION_SUMMARY_EMBED_MARKER = "<!--auto-compaction-summary-embed:v1-->"
SUMMARY_PROMPT = (
    "You are performing an AUTO-COMPACTION CHECKPOINT SUMMARY for an Open WebUI chat. "
    "Create a concise handoff summary for a future model call that will continue the same chat.\n\n"
    "Preserve:\n"
    "- Session goal, original request, current progress, and durable decisions already made\n"
    "- User preferences, constraints, and standing requirements that remain relevant\n"
    "- Tool results, external facts, errors, identifiers, URLs, file names, commands, values, and examples needed to continue\n"
    "- Open questions, unknowns, unresolved failures, and clear next steps\n\n"
    "If the input contains an existing <auto_compaction_context>, merge that prior checkpoint with the following newer messages. "
    "Do not discard earlier checkpoint information merely because it is summarized.\n\n"
    "If <attached_file_contents> is provided, use it only as supporting context for files attached to messages in this "
    "checkpoint source. Preserve durable file facts, names, identifiers, and relevant excerpts needed to continue, "
    "but do not invent file contents or treat attached files outside the checkpoint source as summarized.\n\n"
    "Do not invent facts or treat unknowns as facts. Do not introduce new instructions. "
    "Do not include internal reasoning, private system instructions, or irrelevant transcript detail. "
    "Be concise, structured, and focused on continuity.\n\n"
    "The preceding messages are the exact checkpoint source to summarize. "
    "Messages after this checkpoint source may be retained raw separately; do not infer omitted active requests.\n\n"
    "Output only reusable continuity facts; do not mention this summarization/checkpointing task. Do not continue the conversation. "
    "Do not call tools. Do not ask follow-up questions."
)


def resolve_summary_prompt(summary_prompt: str | None = None) -> str:
    text = str(summary_prompt or "").strip()
    return text or SUMMARY_PROMPT

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
    Column("summary_token_count", Integer, nullable=True),
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
_SOFT_PREFETCH_TASKS: set[asyncio.Task] = set()
_SOFT_PREFETCH_INFLIGHT_KEYS: set[tuple[str, str, str, str, str, str]] = set()
_SOFT_PREFETCH_INFLIGHT_TASKS: dict[tuple[str, str, str, str, str, str], asyncio.Task] = {}
_MESSAGE_TOKEN_ESTIMATE_CACHE: dict[tuple[str, str, str], int] = {}
# Module-level snapshot of the latest model dict seen during pipe()/pipes()
# processing. Populated opportunistically so the Valves dropdown for
# summary_model can list real models without request access at schema time.
_LATEST_MODELS_CACHE: dict[str, dict[str, Any]] = {}
_LATEST_PROVIDER_MODEL_CACHE_ENABLED_STATES: dict[str, bool | None] = {}
_LATEST_PROVIDER_MODEL_CACHE_STATE_ID: int | None = None


def get_generation_lock(key: tuple[str, str, str, str, str, str]) -> asyncio.Lock:
    return _GENERATION_LOCKS.setdefault(key, asyncio.Lock())


def release_generation_lock(key: tuple[str, str, str, str, str, str], lock: asyncio.Lock) -> None:
    waiters = getattr(lock, "_waiters", None)
    has_waiters = bool(waiters)
    if not lock.locked() and not has_waiters and _GENERATION_LOCKS.get(key) is lock:
        _GENERATION_LOCKS.pop(key, None)


def _release_soft_prefetch_task(
    key: tuple[str, str, str, str, str, str],
    task: asyncio.Task,
) -> None:
    _SOFT_PREFETCH_INFLIGHT_KEYS.discard(key)
    if _SOFT_PREFETCH_INFLIGHT_TASKS.get(key) is task:
        _SOFT_PREFETCH_INFLIGHT_TASKS.pop(key, None)
    _SOFT_PREFETCH_TASKS.discard(task)
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    except Exception:
        LOG.exception(
            "Soft compaction prefetch task failed for user_id=%s chat_id=%s pipe_function_id=%s source_hash=%s",
            key[1],
            key[2],
            key[3],
            key[5],
        )
        return
    if isinstance(exc, BaseException):
        LOG.error(
            "Soft compaction prefetch task failed for user_id=%s chat_id=%s pipe_function_id=%s source_hash=%s",
            key[1],
            key[2],
            key[3],
            key[5],
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _launch_soft_prefetch_task(
    key: tuple[str, str, str, str, str, str],
    coro: Awaitable[Any],
) -> bool:
    if key in _SOFT_PREFETCH_INFLIGHT_KEYS:
        with suppress(Exception):
            close = getattr(coro, "close", None)
            if callable(close):
                close()
        return False
    _SOFT_PREFETCH_INFLIGHT_KEYS.add(key)
    task = asyncio.create_task(coro)
    _SOFT_PREFETCH_INFLIGHT_TASKS[key] = task
    _SOFT_PREFETCH_TASKS.add(task)
    task.add_done_callback(lambda done: _release_soft_prefetch_task(key, done))
    return True


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
    checkpoint: dict[str, Any] | None = None
    logical_snapshot: LogicalHistorySnapshot | None = None


@dataclass(frozen=True)
class UsageAnchor:
    assistant_message_id: str
    input_tokens: int
    stable_message_count: int
    input_fingerprint: str
    volatile_message_tokens: int


@dataclass(frozen=True)
class UsageAnchorInput:
    stable_message_count: int
    input_fingerprint: str
    volatile_message_tokens: int


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
class RefRenderManifest:
    ref: str
    utf8_bytes: int | None
    kind: RefKind
    line_count: int | None
    tool: str
    version: int = 1


@dataclass(frozen=True, slots=True)
class ZeroCopySourceHandle:
    text: str


@dataclass(frozen=True, slots=True)
class CanonicalHistorySourceHandle:
    raw_messages: tuple[dict[str, Any], ...]
    raw_record_limit: int
    transient_message_patterns: TransientMessagePatterns | None
    utf8_bytes: int
    raw_source_hash: str
    line_count: int

    @property
    def sha256(self) -> str:
        return self.raw_source_hash

    def iter_records(self) -> Iterable[str]:
        return _iter_canonical_history_records(
            self.raw_messages,
            self.raw_record_limit,
            self.transient_message_patterns,
        )


@dataclass(frozen=True, slots=True)
class LogicalHistorySnapshot:
    identity: tuple[str, str, str, str, str]
    records: tuple[str, ...]
    source_hash_messages: tuple[str, ...]
    prefix_raw_source_hashes: tuple[str, ...]
    prefix_utf8_bytes: tuple[int, ...]
    # Frozen-table callable from a production resolver; snapshots must not
    # retain the raw DB chain or metadata files, so uncomputed fingerprints
    # are derived lazily from the frozen table instead of being stored.
    prefix_file_fingerprint: Callable[[int], str | None] | None = dataclass_field(
        default=None,
        compare=False,
        repr=False,
    )
    source_hash_by_count: dict[int, str] = dataclass_field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class LogicalHistorySourceHandle:
    snapshot: LogicalHistorySnapshot
    source_message_count: int
    utf8_bytes: int
    raw_source_hash: str
    line_count: int

    @property
    def sha256(self) -> str:
        return self.raw_source_hash

    def iter_records(self) -> Iterable[str]:
        return islice(self.snapshot.records, self.source_message_count)


@dataclass(frozen=True, slots=True)
class HistoryRefSourceHandle:
    checkpoint_id: str
    namespace: str
    user_id: str
    chat_id: str
    pipe_function_id: str
    profile_hash: str
    source_hash: str
    source_message_count: int
    raw_source_hash: str
    user_message_id: str
    transient_message_patterns: TransientMessagePatterns | None
    format: str = HISTORY_REF_FORMAT
    logical_snapshot: LogicalHistorySnapshot | None = None


HistoryRefMetadataState = Literal["absent", "valid-v1", "valid-v2", "invalid"]


@dataclass(frozen=True, slots=True)
class ParsedHistoryRefMetadata:
    state: HistoryRefMetadataState
    value: dict[str, str] | None


RefSourceHandle = (
    ZeroCopySourceHandle
    | CanonicalHistorySourceHandle
    | LogicalHistorySourceHandle
    | HistoryRefSourceHandle
)


@dataclass(frozen=True, slots=True)
class RefCatalogEntry:
    manifest: RefManifest
    source: RefSourceHandle
    preview_text: str | None = dataclass_field(default=None, repr=False)


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
    reader_schema: MappingProxyType | None
    render_manifests: tuple[RefRenderManifest, ...] = ()


class _RenderedSummaryMessage(dict[str, Any]):
    history_ref: str | None = None


@dataclass(frozen=True, slots=True)
class RefResolverResult:
    ref: ParsedRef | None
    content: str | None


class CoreFunctionCallingGeneration(StrEnum):
    NATIVE_DEFAULT = "native_default"
    NATIVE_OPT_IN = "native_opt_in"
    UNKNOWN = "unknown"


class RefModeReason(StrEnum):
    ACTIVE = "active"
    VALVE_OFF = "valve_off"
    NON_NATIVE_CONTEXT = "non_native_context"
    NON_DURABLE_CONTEXT = "non_durable_context"
    PROVIDER_SCHEMA_UNSUPPORTED = "provider_schema_unsupported"
    CORE_REGISTRY_UNAVAILABLE = "core_registry_unavailable"
    READER_COLLISION = "reader_collision"


@dataclass(frozen=True, slots=True)
class EffectiveRefMode:
    active: bool
    reason: RefModeReason


@dataclass(frozen=True, slots=True)
class RefStateDelta:
    added_refs: tuple[str, ...]
    generation: int | None


@dataclass(frozen=True, slots=True, init=False)
class RefBindingKey:
    user_id: str
    chat_id: str
    user_message_id: str
    assistant_message_id: str
    incoming_model_id: str
    base_pipe_id: str
    profile_hash: str
    branch_anchor: str

    def __init__(
        self,
        *,
        user_id: str,
        chat_id: str,
        assistant_message_id: str,
        incoming_model_id: str,
        profile_hash: str,
        user_message_id: str = "",
        base_pipe_id: str = "",
        branch_anchor: str = "",
        pipe_function_id: str = "",
    ) -> None:
        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(self, "chat_id", chat_id)
        object.__setattr__(self, "user_message_id", user_message_id)
        object.__setattr__(self, "assistant_message_id", assistant_message_id)
        object.__setattr__(self, "incoming_model_id", incoming_model_id)
        object.__setattr__(self, "base_pipe_id", base_pipe_id or pipe_function_id)
        object.__setattr__(self, "profile_hash", profile_hash)
        object.__setattr__(self, "branch_anchor", branch_anchor)

    @property
    def pipe_function_id(self) -> str:
        return self.base_pipe_id


@dataclass(frozen=True, slots=True)
class RefBindingState:
    generation: int
    catalog: tuple[RefCatalogEntry, ...]
    registry: dict[str, Any]
    reader: Callable[[str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class RefReservation:
    key: RefBindingKey
    generation: int
    registry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RefAttempt:
    key: RefBindingKey
    generation: int
    plan: RefProjectionPlan
    registry: dict[str, Any]
    reader: Callable[[str], Awaitable[str]]
    previous_binding: RefBindingState | None
    previous_reader_entry: Any

    @property
    def expected_generation(self) -> int:
        return self.generation


@dataclass(frozen=True, slots=True)
class RefDeferredCleanup:
    generation: int
    successor_generation: int
    registry: dict[str, Any]
    reader: Callable[[str], Awaitable[str]]


@dataclass(slots=True)
class RefRequestStore:
    lock: asyncio.Lock = dataclass_field(default_factory=asyncio.Lock)
    next_generation: int = 0
    bindings: dict[RefBindingKey, RefBindingState] = dataclass_field(default_factory=dict)
    registry_owners: dict[int, RefBindingKey] = dataclass_field(default_factory=dict)
    reservations: dict[RefBindingKey, RefReservation] = dataclass_field(default_factory=dict)
    registry_reservations: dict[int, RefReservation] = dataclass_field(default_factory=dict)
    deferred_cleanups: dict[RefBindingKey, RefDeferredCleanup] = dataclass_field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class RefModePreflight:
    valve_enabled: bool
    native_function_calling: bool
    durable_context: bool
    provider_schema_supported: bool
    metadata_tools: Any
    injected_tools: Any
    registry_available: bool = True


@dataclass(frozen=True, slots=True)
class RefProjectionError(RuntimeError):
    stage: str

    def __str__(self) -> str:
        return f"Externalized ref {self.stage} failed before provider forward"


_REF_PROJECTION_DIAGNOSTIC_STAGES = MappingProxyType(
    {
        "reader schema rendering": "reader_schema",
        "reader schema registration": "reader_schema",
        "registration": "registration",
        "generation CAS": "generation_cas",
        "tool preview rendering": "tool_preview_rendering",
    }
)


def _log_ref_projection_failure(exc: RefProjectionError) -> None:
    LOG.warning(
        "Auto Compact ref projection failed: stage=%s reason=operation_failed",
        _REF_PROJECTION_DIAGNOSTIC_STAGES.get(exc.stage, "unknown"),
    )


@dataclass(frozen=True, slots=True)
class CanonicalHistoryError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return f"Canonical history unavailable: {self.reason}"


class HistoryRefStorageUnavailableError(Exception):
    """Raised when history-reference storage access fails without exposing backend details."""


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


@dataclass(frozen=True)
class TaskPromptSpec:
    config_attr: str
    config_key: str
    default_attr: str
    builder_name: str
    strip_configured: bool = False


TASK_PROMPT_SPECS = {
    TASKS.TITLE_GENERATION.value: TaskPromptSpec(
        "TITLE_GENERATION_PROMPT_TEMPLATE",
        "task.title.prompt_template",
        "DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE",
        "title_generation_template",
    ),
    TASKS.FOLLOW_UP_GENERATION.value: TaskPromptSpec(
        "FOLLOW_UP_GENERATION_PROMPT_TEMPLATE",
        "task.follow_up.prompt_template",
        "DEFAULT_FOLLOW_UP_GENERATION_PROMPT_TEMPLATE",
        "follow_up_generation_template",
    ),
    TASKS.TAGS_GENERATION.value: TaskPromptSpec(
        "TAGS_GENERATION_PROMPT_TEMPLATE",
        "task.tags.prompt_template",
        "DEFAULT_TAGS_GENERATION_PROMPT_TEMPLATE",
        "tags_generation_template",
    ),
    TASKS.QUERY_GENERATION.value: TaskPromptSpec(
        "QUERY_GENERATION_PROMPT_TEMPLATE",
        "task.query.prompt_template",
        "DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE",
        "query_generation_template",
        strip_configured=True,
    ),
    TASKS.IMAGE_PROMPT_GENERATION.value: TaskPromptSpec(
        "IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE",
        "task.image.prompt_template",
        "DEFAULT_IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE",
        "image_prompt_generation_template",
    ),
    TASKS.AUTOCOMPLETE_GENERATION.value: TaskPromptSpec(
        "AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE",
        "task.autocomplete.prompt_template",
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
    target_params: dict[str, Any]
    wrapper_params: dict[str, Any]
    access_grants: list[dict[str, Any]]
    owner_user_id: str | None


@dataclass(frozen=True)
class CoreChatModelRoute:
    model_id: str
    fallback_model: dict[str, Any] | None = None
    target_params: dict[str, Any] | None = None
    token_system_prompt: str | None = None
    usage_anchor_shaping_hash: str | None = None
    provider_model_id: str | None = None
    usage_anchor_dropped_message_keys: frozenset[str] = frozenset()


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


class SummaryFileContextUnavailable(UnsupportedCompactionInput):
    """Raised when absorbed prefix files cannot be included in a checkpoint summary."""

    def __init__(
        self,
        message: str = "Attached file context could not be loaded for the compaction summary",
    ):
        super().__init__(message, code="file_context_unavailable")


class ParentCheckpointExtensionFailed(Exception):
    """Raised when a verified parent checkpoint exists but extension summarization fails."""

    def __init__(self, parent: dict[str, Any], original: Exception):
        super().__init__(str(original))
        self.parent = parent
        self.original = original


class _CheckpointGenerationSkipped(Exception):
    """Raised after a pending claim when a later-ready checkpoint makes generation redundant."""


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
_PROVIDER_PROMPT_CACHE_HINT_KEYS = {
    "cache_control",
    "cacheControl",
}
_PREFIX_FILE_ATTACHMENT_IDENTITY_KEYS = {
    "checksum",
    "collection_name",
    "collection_names",
    "content_type",
    "file",
    "file_hash",
    "file_id",
    "filename",
    "hash",
    "id",
    "legacy",
    "mime_type",
    "name",
    "revision",
    "sha256",
    "type",
    "version",
}
_PREFIX_EMBEDDED_FILE_IDENTITY_KEYS = {
    "checksum",
    "collection_name",
    "collection_names",
    "content_type",
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
    "revision",
    "sha256",
    "type",
    "version",
}
_FILE_IDENTITY_STABLE_DISCRIMINATOR_KEYS = {
    "checksum",
    "file_hash",
    "file_id",
    "hash",
    "id",
    "sha256",
}
_PREFIX_FILE_METADATA_TRANSIENT_KEYS = _FILE_METADATA_TRANSIENT_KEYS
_PREFIX_FILE_METADATA_BODY_KEYS = {
    "body",
    "content",
    "context",
    "docs",
    "document",
    "documents",
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


def _canonicalize_content_part_for_source_hash(value: Any) -> Any:
    if not isinstance(value, dict):
        return _canonicalize_content_part(value)
    part_type = value.get("type")
    if not isinstance(part_type, str):
        return _canonicalize_content_part(value)
    # Canonical projection is structure-strict and attribute-tolerant: known nodes
    # drop provider extras from both the source hash and the served history, while
    # hash-only semantic identity (media detail, file identity, files, sources)
    # is kept. The contract is one-way: equal source hashes imply equal served
    # bytes. Source hashing itself stays total: unknown structures fall back to the
    # generic canonicalization here and fail loudly only at direct-history admission.
    if part_type in {"text", "input_text", "output_text"}:
        keys = ("text", "type")
    elif part_type == "input_image":
        keys = ("detail", "image_url", "type")
    elif part_type == "image_url":
        keys = ("image_url", "type")
    elif part_type in _FILE_CONTENT_PART_TYPES:
        keys = ("file", "file_data", "file_id", "filename", "type")
    else:
        return _canonicalize_content_part(value)

    canonical: dict[str, Any] = {}
    for key in keys:
        if key not in value:
            continue
        item_value = value[key]
        if part_type == "image_url" and key == "image_url" and isinstance(item_value, dict):
            image_url: dict[str, Any] = {}
            for image_key in ("detail", "file", "url"):
                if image_key not in item_value:
                    continue
                image_item = _canonicalize_general_value(item_value[image_key])
                if not _is_empty_canonical_value(image_item):
                    image_url[image_key] = image_item
            item = image_url
        elif part_type in _FILE_CONTENT_PART_TYPES and key == "file":
            item = _canonicalize_embedded_file_value(item_value)
        else:
            item = _canonicalize_general_value(item_value)
        if not _is_empty_canonical_value(item):
            canonical[key] = item
    return canonical


def _canonicalize_content_value_for_source_hash(value: Any) -> Any:
    if isinstance(value, list):
        canonical = []
        for item in value:
            canonical_item = _canonicalize_content_part_for_source_hash(item)
            if not _is_empty_canonical_value(canonical_item):
                canonical.append(canonical_item)
        if len(canonical) == 1:
            collapsed = _collapse_text_only_content_part(canonical[0])
            if collapsed is not None:
                return collapsed
        return canonical
    return _canonicalize_content_part_for_source_hash(value)


def _canonicalize_tool_call_for_source_hash(value: Any) -> Any:
    if not isinstance(value, dict) or value.get("type") != "function":
        return _canonicalize_general_value(value)
    canonical: dict[str, Any] = {}
    for key in ("function", "id", "type"):
        if key not in value:
            continue
        item_value = value[key]
        if key == "function" and isinstance(item_value, dict):
            function: dict[str, Any] = {}
            for function_key in ("arguments", "name"):
                if function_key not in item_value:
                    continue
                function_item = _canonicalize_general_value(item_value[function_key])
                if not _is_empty_canonical_value(function_item):
                    function[function_key] = function_item
            item = function
        else:
            item = _canonicalize_general_value(item_value)
        if not _is_empty_canonical_value(item):
            canonical[key] = item
    return canonical


def _canonicalize_tool_calls_for_source_hash(value: Any) -> Any:
    if isinstance(value, list):
        canonical = []
        for item in value:
            canonical_item = _canonicalize_tool_call_for_source_hash(item)
            if not _is_empty_canonical_value(canonical_item):
                canonical.append(canonical_item)
        return canonical
    return _canonicalize_tool_call_for_source_hash(value)


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


def _canonicalize_message_value_for_source_hash(key: str, value: Any) -> Any:
    if key == "content":
        return _canonicalize_content_value_for_source_hash(value)
    if key == "tool_calls":
        return _canonicalize_tool_calls_for_source_hash(value)
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
        value = _canonicalize_message_value_for_source_hash(key, message[key])
        if _is_empty_canonical_value(value):
            continue
        canonical[key] = value
    canonical.setdefault("role", message.get("role", "assistant"))
    canonical.setdefault("content", "")
    return canonical


def canonicalize_message_for_token_estimate(message: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key in sorted(message.keys()):
        if key in _TOP_LEVEL_MESSAGE_DROP_KEYS or key not in _TOKEN_MESSAGE_KEYS:
            continue
        value = _canonicalize_message_value(key, message[key])
        if _is_empty_canonical_value(value):
            continue
        canonical[key] = value
    canonical.setdefault("role", message.get("role", "assistant"))
    canonical.setdefault("content", "")
    return canonical


def _is_system_message(message: dict[str, Any]) -> bool:
    return message.get("role") == "system"


TransientMessagePatterns = tuple[re.Pattern[str], ...]


class _TransientMessageMatcher:
    def __init__(self, patterns: TransientMessagePatterns):
        self.patterns = patterns
        self._masks: dict[int, tuple[list[dict[str, Any]], tuple[bool, ...]]] = {}

    def mask(self, messages: list[dict[str, Any]]) -> tuple[bool, ...] | None:
        if not self.patterns:
            return None
        key = id(messages)
        cached = self._masks.get(key)
        if cached is not None and cached[0] is messages:
            return cached[1]
        mask = tuple(_is_transient_message(message, self.patterns) for message in messages)
        self._masks[key] = (messages, mask)
        return mask


@lru_cache(maxsize=128)
def parse_transient_message_patterns(value: str) -> TransientMessagePatterns:
    patterns: list[re.Pattern[str]] = []
    for line_number, line in enumerate(str(value or "").splitlines(), start=1):
        pattern = line.strip()
        if not pattern:
            continue
        try:
            patterns.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"transient_message_patterns line {line_number}: {exc}") from exc
    return tuple(patterns)


def _first_text_part_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                return part["text"]
    return None


def _first_non_whitespace_index(text: str) -> int:
    for index, char in enumerate(text):
        if not char.isspace():
            return index
    return len(text)


def _is_transient_message(
    message: Any,
    transient_message_patterns: TransientMessagePatterns | _TransientMessageMatcher | None = None,
) -> bool:
    patterns = (
        transient_message_patterns.patterns
        if isinstance(transient_message_patterns, _TransientMessageMatcher)
        else transient_message_patterns
    )
    if not patterns or not isinstance(message, dict) or message.get("role") != "user":
        return False
    text = _first_text_part_text(message.get("content"))
    if text is None:
        return False
    pos = _first_non_whitespace_index(text)
    return any(pattern.match(text, pos) is not None for pattern in patterns)


def _messages_for_transient_aware_rag(
    messages: Any,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> Any:
    if not isinstance(messages, list) or not transient_message_patterns:
        return messages
    filtered = [
        message
        for message in messages
        if not _is_transient_message(message, transient_message_patterns)
    ]
    if len(filtered) == len(messages):
        return messages
    return copy.deepcopy(filtered)


def _merge_rag_messages_preserving_transient_users(
    original_messages: Any,
    applied_messages: Any,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> Any:
    if (
        not isinstance(original_messages, list)
        or not isinstance(applied_messages, list)
        or not transient_message_patterns
    ):
        return applied_messages
    positions = [
        index
        for index, message in enumerate(original_messages)
        if not _is_transient_message(message, transient_message_patterns)
    ]
    if len(positions) == len(original_messages):
        return applied_messages
    applied_prefix: list[Any] = []
    applied_core = applied_messages
    result_offset = 0
    if (
        len(applied_messages) > len(positions)
        and isinstance(applied_messages[0], dict)
        and applied_messages[0].get("role") == "system"
        and (
            not positions
            or not isinstance(original_messages[positions[0]], dict)
            or original_messages[positions[0]].get("role") != "system"
        )
    ):
        applied_prefix = [applied_messages[0]]
        applied_core = applied_messages[1:]
        result_offset = 1
    if len(applied_core) < len(positions):
        return copy.deepcopy(original_messages)
    result = [*copy.deepcopy(applied_prefix), *copy.deepcopy(original_messages)]
    for original_index, applied in zip(positions, applied_core[: len(positions)]):
        result[original_index + result_offset] = copy.deepcopy(applied)
    result.extend(copy.deepcopy(applied_core[len(positions) :]))
    return result


def _transient_message_mask(
    messages: list[dict[str, Any]],
    transient_message_patterns: TransientMessagePatterns | _TransientMessageMatcher | None = None,
) -> tuple[bool, ...] | None:
    if isinstance(transient_message_patterns, _TransientMessageMatcher):
        return transient_message_patterns.mask(messages)
    if not transient_message_patterns:
        return None
    return tuple(_is_transient_message(message, transient_message_patterns) for message in messages)


def _canonical_history_content(content: Any, *, required: bool) -> str | list[dict[str, str]]:
    if content is None and not required:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise CanonicalHistoryError(reason="unknown content shape")
    canonical: list[dict[str, str]] = []
    for part in content:
        if not isinstance(part, dict):
            raise CanonicalHistoryError(reason="unknown content shape")
        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            if not isinstance(part.get("text"), str):
                raise CanonicalHistoryError(reason="unknown content shape")
            canonical.append({"type": "text", "text": part["text"]})
            continue
        if part_type == "input_image":
            if not isinstance(part.get("image_url"), str):
                raise CanonicalHistoryError(reason="unknown content shape")
            canonical.append({"type": "omitted_media", "media": "image"})
            continue
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                if not isinstance(image_url.get("url"), str):
                    raise CanonicalHistoryError(reason="unknown content shape")
            elif not isinstance(image_url, str):
                raise CanonicalHistoryError(reason="unknown content shape")
            canonical.append({"type": "omitted_media", "media": "image"})
            continue
        raise CanonicalHistoryError(reason="unknown content shape")
    return canonical


def _canonical_history_tool_calls(tool_calls: Any) -> list[dict[str, str]]:
    if not isinstance(tool_calls, list):
        raise CanonicalHistoryError(reason="unknown tool call shape")
    canonical: list[dict[str, str]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise CanonicalHistoryError(reason="unknown tool call shape")
        function = call.get("function")
        if (
            call.get("type") != "function"
            or not isinstance(call.get("id"), str)
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not isinstance(function.get("arguments"), str)
        ):
            raise CanonicalHistoryError(reason="unknown tool call shape")
        canonical.append(
            {
                "id": call["id"],
                "name": function["name"],
                "arguments": function["arguments"],
            }
        )
    return canonical


def _canonical_direct_history_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    semantic_keys = set(message) & _KNOWN_SEMANTIC_MESSAGE_KEYS
    if role == "user":
        if semantic_keys != {"role", "content"}:
            raise CanonicalHistoryError(reason="unknown message shape")
        return {
            "role": "user",
            "content": _canonical_history_content(message.get("content"), required=True),
        }
    if role == "assistant":
        if not semantic_keys <= {"role", "content", "tool_calls"} or "role" not in semantic_keys:
            raise CanonicalHistoryError(reason="unknown message shape")
        canonical: dict[str, Any] = {
            "role": "assistant",
            "content": _canonical_history_content(message.get("content"), required=False),
        }
        if "tool_calls" in message:
            calls = _canonical_history_tool_calls(message["tool_calls"])
            if calls:
                canonical["tool_calls"] = calls
        return canonical
    if role == "tool":
        if semantic_keys != {"role", "content", "tool_call_id"}:
            raise CanonicalHistoryError(reason="unknown message shape")
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            raise CanonicalHistoryError(reason="unknown message shape")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": _canonical_history_content(message.get("content"), required=True),
        }
    raise CanonicalHistoryError(reason="unknown message shape")


def _validated_output_call_ids(output: list[Any]) -> tuple[set[str], set[str]]:
    requested: set[str] = set()
    completed: set[str] = set()
    for item in output:
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise CanonicalHistoryError(reason="unknown output shape")
        item_type = item["type"]
        if item_type == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                raise CanonicalHistoryError(reason="unknown output shape")
            requested.add(call_id)
        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                raise CanonicalHistoryError(reason="unknown output shape")
            completed.add(call_id)
    return requested, completed


def _iter_core_output_history_messages(output: list[Any]) -> Iterable[dict[str, Any]]:
    requested, completed = _validated_output_call_ids(output)
    pending_content: list[str] = []
    pending_calls: list[dict[str, str]] = []

    def flush_pending() -> dict[str, Any] | None:
        if not pending_content and not pending_calls:
            return None
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(pending_content) if pending_content else "",
        }
        if pending_calls:
            message["tool_calls"] = list(pending_calls)
        pending_content.clear()
        pending_calls.clear()
        return message

    for item in output:
        item_type = item["type"]
        if item_type == "message":
            parts = item.get("content")
            if not isinstance(parts, list):
                raise CanonicalHistoryError(reason="unknown output shape")
            text = ""
            for part in parts:
                if (
                    not isinstance(part, dict)
                    or part.get("type") != "output_text"
                    or not isinstance(part.get("text"), str)
                ):
                    raise CanonicalHistoryError(reason="unknown content shape")
                text += part["text"]
            if text:
                pending_content.append(text)
            continue
        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, str):
                raise CanonicalHistoryError(reason="unknown output shape")
            if call_id in completed:
                pending_calls.append({"id": call_id, "name": name, "arguments": arguments})
            continue
        if item_type == "function_call_output":
            pending = flush_pending()
            if pending is not None:
                yield pending
            parts = item.get("output")
            if not isinstance(parts, list):
                raise CanonicalHistoryError(reason="unknown output shape")
            text = ""
            images: list[dict[str, str]] = []
            for part in parts:
                if not isinstance(part, dict):
                    raise CanonicalHistoryError(reason="unknown content shape")
                if part.get("type") == "input_text":
                    if not isinstance(part.get("text"), str):
                        raise CanonicalHistoryError(reason="unknown content shape")
                    text += part["text"]
                elif part.get("type") == "input_image":
                    if not isinstance(part.get("image_url"), str):
                        raise CanonicalHistoryError(reason="unknown content shape")
                    images.append({"type": "omitted_media", "media": "image"})
                else:
                    raise CanonicalHistoryError(reason="unknown content shape")
            call_id = item["call_id"]
            if call_id in requested:
                content: str | list[dict[str, str]] = text
                if images:
                    content = [{"type": "text", "text": text}, *images]
                yield {"role": "tool", "tool_call_id": call_id, "content": content}
            continue
        if item_type == "open_webui:code_interpreter":
            code = item.get("code", "")
            code_output = item.get("output", "")
            if not isinstance(code, str):
                raise CanonicalHistoryError(reason="unknown output shape")
            if code:
                pending_content.append(
                    f"<code_interpreter>\n{code}\n</code_interpreter>"
                )
            if isinstance(code_output, dict):
                if not set(code_output) <= {"stdout", "result", "stderr"}:
                    raise CanonicalHistoryError(reason="unknown output shape")
                stdout = code_output.get("stdout", "")
                result = code_output.get("result", "")
                if not isinstance(stdout, str) or not isinstance(result, str):
                    raise CanonicalHistoryError(reason="unknown output shape")
                output_text = stdout or result
            elif isinstance(code_output, str):
                output_text = code_output
            else:
                raise CanonicalHistoryError(reason="unknown output shape")
            if output_text:
                pending_content.append(
                    f"<code_interpreter_output>\n{output_text}\n</code_interpreter_output>"
                )
            continue
        if item_type == "reasoning":
            continue
        # Open WebUI Core's converter skips extension records it does not own;
        # mirror that contract while keeping unknown bare output types strict.
        if item_type.startswith("open_webui:"):
            continue
        raise CanonicalHistoryError(reason="unknown output shape")
    pending = flush_pending()
    if pending is not None:
        yield pending


def _convert_core_output_to_messages(output: list[Any]) -> list[dict[str, Any]]:
    from open_webui.utils.misc import convert_output_to_messages

    converter_kwargs: dict[str, Any] = {
        "raw": True,
        "reasoning_format": None,
    }
    if "flatten_tool_images" in inspect.signature(
        convert_output_to_messages
    ).parameters:
        converter_kwargs["flatten_tool_images"] = True
    return convert_output_to_messages(output, **converter_kwargs)


def _iter_canonical_messages_for_raw(
    message: dict[str, Any],
    transient_message_patterns: TransientMessagePatterns | None,
) -> Iterable[dict[str, Any]]:
    if _is_system_message(message) or _is_transient_message(message, transient_message_patterns):
        return
    output = message.get("output")
    if message.get("role") == "assistant" and output:
        semantic_keys = set(message) & _KNOWN_SEMANTIC_MESSAGE_KEYS
        if not semantic_keys <= {"role", "content", "tool_calls", "output"}:
            raise CanonicalHistoryError(reason="unknown message shape")
        if not isinstance(output, list):
            raise CanonicalHistoryError(reason="unknown output shape")
        emitted = False
        for converted in _iter_core_output_history_messages(output):
            emitted = True
            yield converted
        if emitted or _convert_core_output_to_messages(output):
            return
    if "output" in message:
        message = {key: value for key, value in message.items() if key != "output"}
    yield _canonical_direct_history_message(message)


def _canonical_history_record(message: dict[str, Any]) -> str:
    return json.dumps(
        message,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _iter_canonical_history_records(
    raw_messages: tuple[dict[str, Any], ...],
    raw_record_limit: int,
    transient_message_patterns: TransientMessagePatterns | None,
) -> Iterable[str]:
    for raw_message in raw_messages[:raw_record_limit]:
        yield from (
            _canonical_history_record(message)
            for message in _iter_canonical_messages_for_raw(
                raw_message,
                transient_message_patterns,
            )
        )


def _update_canonical_history_digest(
    digest: Any,
    record: str,
    *,
    separator: bool,
    cancelled: threading.Event | None = None,
) -> int:
    utf8_bytes = 0
    if separator:
        digest.update(b"\n")
        utf8_bytes = 1
    for offset in range(0, len(record), REF_TEXT_HASH_CHUNK_CHARS):
        if cancelled is not None:
            _check_ref_exec_cancelled(cancelled)
        try:
            encoded = record[offset : offset + REF_TEXT_HASH_CHUNK_CHARS].encode(
                "utf-8"
            )
        except UnicodeEncodeError as exc:
            raise CanonicalHistoryError(
                reason="history source is not valid UTF-8"
            ) from exc
        digest.update(encoded)
        utf8_bytes += len(encoded)
    return utf8_bytes


def _build_canonical_history_source_sync(
    raw_messages: tuple[dict[str, Any], ...],
    source_message_count: int,
    transient_message_patterns: TransientMessagePatterns | None,
) -> CanonicalHistorySourceHandle:
    if source_message_count < 0:
        raise CanonicalHistoryError(reason="unsaved source count")
    digest = hashlib.sha256()
    utf8_bytes = 0
    emitted = 0
    source_identity_emitted = 0
    if source_message_count == 0:
        return CanonicalHistorySourceHandle(
            raw_messages=raw_messages,
            raw_record_limit=0,
            transient_message_patterns=transient_message_patterns,
            utf8_bytes=0,
            raw_source_hash=digest.hexdigest(),
            line_count=0,
        )
    for raw_index, raw_message in enumerate(raw_messages):
        for canonical in _iter_canonical_messages_for_raw(
            raw_message,
            transient_message_patterns,
        ):
            record = _canonical_history_record(canonical)
            utf8_bytes += _update_canonical_history_digest(
                digest,
                record,
                separator=emitted > 0,
            )
            emitted += 1
        if _is_source_identity_message(
            raw_message,
            transient_message_patterns=transient_message_patterns,
        ):
            source_identity_emitted += _source_identity_message_span(
                raw_message,
                transient_message_patterns=transient_message_patterns,
            )
        if source_identity_emitted == source_message_count:
            return CanonicalHistorySourceHandle(
                raw_messages=raw_messages,
                raw_record_limit=raw_index + 1,
                transient_message_patterns=transient_message_patterns,
                utf8_bytes=utf8_bytes,
                raw_source_hash=digest.hexdigest(),
                line_count=emitted,
            )
        if source_identity_emitted > source_message_count:
            raise CanonicalHistoryError(reason="checkpoint count is inside a raw record")
    raise CanonicalHistoryError(reason="checkpoint count references an unsaved source")


async def build_canonical_history_source(
    raw_messages: list[dict[str, Any]],
    *,
    source_message_count: int,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> CanonicalHistorySourceHandle:
    return await asyncio.to_thread(
        _build_canonical_history_source_sync,
        tuple(raw_messages),
        source_message_count,
        transient_message_patterns,
    )


def _build_logical_history_snapshot_sync(
    messages: tuple[dict[str, Any], ...],
    file_backed_image_db_chain: tuple[dict[str, Any], ...],
    prefix_file_fingerprint_resolver: Callable[[int], str | None] | None,
    transient_message_patterns: TransientMessagePatterns | None,
    identity: tuple[str, str, str, str, str],
) -> LogicalHistorySnapshot:
    logical_messages = list(messages)
    source_hash_messages = _stable_file_backed_image_source_messages(
        logical_messages,
        list(file_backed_image_db_chain) or None,
        transient_message_patterns=transient_message_patterns,
    )
    mask = _transient_message_mask(logical_messages, transient_message_patterns)
    records: list[str] = []
    canonical_source_messages: list[str] = []
    prefix_raw_source_hashes: list[str] = []
    prefix_utf8_bytes: list[int] = []
    digest = hashlib.sha256()
    utf8_bytes = 0
    for index, message in enumerate(logical_messages):
        if not _is_source_identity_message(
            message,
            transient_message_patterns=transient_message_patterns,
            transient_message_mask=mask,
            index=index,
        ):
            continue
        record = _canonical_history_record(
            canonicalize_message_for_source_hash(
                _canonical_direct_history_message(message)
            )
        )
        utf8_bytes += _update_canonical_history_digest(
            digest,
            record,
            separator=bool(records),
        )
        records.append(record)
        prefix_raw_source_hashes.append(digest.hexdigest())
        prefix_utf8_bytes.append(utf8_bytes)
        canonical_source_messages.append(
            _canonical_history_record(
                canonicalize_message_for_source_hash(source_hash_messages[index])
            )
        )
    return LogicalHistorySnapshot(
        identity=identity,
        records=tuple(records),
        source_hash_messages=tuple(canonical_source_messages),
        prefix_raw_source_hashes=tuple(prefix_raw_source_hashes),
        prefix_utf8_bytes=tuple(prefix_utf8_bytes),
        prefix_file_fingerprint=prefix_file_fingerprint_resolver,
    )


async def build_logical_history_snapshot(
    messages: list[dict[str, Any]],
    *,
    identity: tuple[str, str, str, str, str],
    prefix_file_fingerprint_resolver: Callable[[int], str | None] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> LogicalHistorySnapshot:
    return await asyncio.to_thread(
        _build_logical_history_snapshot_sync,
        tuple(copy.deepcopy(messages)),
        tuple(
            copy.deepcopy(
                _prefix_file_fingerprint_resolver_db_chain(
                    prefix_file_fingerprint_resolver
                )
                or []
            )
        ),
        getattr(
            prefix_file_fingerprint_resolver,
            PREFIX_FILE_FINGERPRINT_RESOLVER_FROZEN_ATTR,
            prefix_file_fingerprint_resolver,
        ),
        transient_message_patterns,
        identity,
    )


REQUEST_STATE_LOGICAL_HISTORY_SNAPSHOT_CACHE_KEY = (
    "_auto_compact_logical_history_snapshot_cache"
)


async def get_or_build_logical_history_snapshot(
    request: Any,
    messages: list[dict[str, Any]],
    *,
    identity: tuple[str, str, str, str, str],
    prefix_file_fingerprint_resolver: Callable[[int], str | None] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    source_hash: str | None = None,
) -> LogicalHistorySnapshot:
    state = getattr(request, "state", None)
    if state is None:
        return await build_logical_history_snapshot(
            messages,
            identity=identity,
            prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
            transient_message_patterns=transient_message_patterns,
        )
    source_message_count = _source_identity_message_count(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    if source_hash is None:
        source_fingerprint = (
            prefix_file_fingerprint_resolver(source_message_count)
            if prefix_file_fingerprint_resolver is not None
            and source_message_count > 0
            else None
        )
        source_hash = compute_summary_source_hash(
            messages,
            source_fingerprint,
            _prefix_file_fingerprint_resolver_db_chain(
                prefix_file_fingerprint_resolver
            ),
            transient_message_patterns=transient_message_patterns,
        )
    cache_key = (
        identity,
        source_hash,
        source_message_count,
    )
    cache = getattr(
        state,
        REQUEST_STATE_LOGICAL_HISTORY_SNAPSHOT_CACHE_KEY,
        None,
    )
    if not isinstance(cache, dict):
        cache = {}
        setattr(
            state,
            REQUEST_STATE_LOGICAL_HISTORY_SNAPSHOT_CACHE_KEY,
            cache,
        )
    cached_slot = cache.get(identity)
    cached = (
        cached_slot[1]
        if cached_slot is not None and cached_slot[0] == cache_key
        else None
    )
    if isinstance(cached, LogicalHistorySnapshot):
        return cached
    if isinstance(cached, asyncio.Task):
        return await asyncio.shield(cached)

    build_task = asyncio.create_task(
        build_logical_history_snapshot(
            messages,
            identity=identity,
            prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
            transient_message_patterns=transient_message_patterns,
        )
    )
    cache[identity] = (cache_key, build_task)

    def finish_snapshot(task: asyncio.Task[LogicalHistorySnapshot]) -> None:
        current_slot = cache.get(identity)
        if (
            current_slot is None
            or current_slot[0] != cache_key
            or current_slot[1] is not task
        ):
            return
        try:
            cache[identity] = (cache_key, task.result())
        except (asyncio.CancelledError, Exception):
            cache.pop(identity, None)

    build_task.add_done_callback(finish_snapshot)
    return await asyncio.shield(build_task)


def _logical_snapshot_source_hash(
    snapshot: LogicalHistorySnapshot,
    source_message_count: int,
) -> str | None:
    if source_message_count <= 0 or source_message_count > len(snapshot.records):
        return None
    if source_message_count in snapshot.source_hash_by_count:
        return snapshot.source_hash_by_count[source_message_count]
    payload: dict[str, Any] = {
        "family": SOURCE_HASH_FAMILY,
        "messages": [
            json.loads(message)
            for message in snapshot.source_hash_messages[:source_message_count]
        ],
    }
    prefix_file_fingerprint = (
        snapshot.prefix_file_fingerprint(source_message_count)
        if snapshot.prefix_file_fingerprint is not None
        else None
    )
    if prefix_file_fingerprint:
        payload["prefix_file_fingerprint"] = prefix_file_fingerprint
    result = _json_hash(payload)
    snapshot.source_hash_by_count[source_message_count] = result
    return result


async def _logical_snapshot_source_hash_async(
    snapshot: LogicalHistorySnapshot,
    source_message_count: int,
) -> str | None:
    if source_message_count <= 0 or source_message_count > len(snapshot.records):
        return None
    if source_message_count in snapshot.source_hash_by_count:
        return snapshot.source_hash_by_count[source_message_count]
    return await asyncio.to_thread(
        _logical_snapshot_source_hash,
        snapshot,
        source_message_count,
    )


def _logical_snapshot_matches_checkpoint(
    snapshot: LogicalHistorySnapshot,
    checkpoint: dict[str, Any],
) -> bool:
    identity = (
        str(checkpoint.get("namespace") or ""),
        str(checkpoint.get("user_id") or ""),
        str(checkpoint.get("chat_id") or ""),
        str(checkpoint.get("pipe_function_id") or ""),
        str(checkpoint.get("profile_hash") or ""),
    )
    try:
        source_message_count = int(checkpoint.get("source_message_count") or 0)
    except (TypeError, ValueError):
        return False
    return (
        snapshot.identity == identity
        and _logical_snapshot_source_hash(snapshot, source_message_count)
        == checkpoint.get("source_hash")
    )


def _logical_history_source_handle(
    snapshot: LogicalHistorySnapshot,
    source_message_count: int,
) -> LogicalHistorySourceHandle:
    if source_message_count <= 0 or source_message_count > len(snapshot.records):
        raise CanonicalHistoryError(reason="checkpoint count references an unsaved source")
    return LogicalHistorySourceHandle(
        snapshot=snapshot,
        source_message_count=source_message_count,
        utf8_bytes=snapshot.prefix_utf8_bytes[source_message_count - 1],
        raw_source_hash=snapshot.prefix_raw_source_hashes[
            source_message_count - 1
        ],
        line_count=source_message_count,
    )


async def load_raw_chat_branch(
    *,
    chat_id: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    from open_webui.models.chats import Chats
    from open_webui.utils.misc import get_message_list

    current_user_message_id = metadata.get("user_message_id")
    if not isinstance(current_user_message_id, str) or not current_user_message_id:
        raise CanonicalHistoryError(reason="metadata user_message_id is required")
    messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
    if not isinstance(messages_map, dict) or current_user_message_id not in messages_map:
        raise CanonicalHistoryError(reason="raw branch is unavailable")
    branch = get_message_list(messages_map, current_user_message_id)
    if not branch or not all(isinstance(message, dict) for message in branch):
        raise CanonicalHistoryError(reason="raw branch is unavailable")
    return branch


def _parse_history_ref_metadata(summary_meta: Any) -> ParsedHistoryRefMetadata:
    if not isinstance(summary_meta, dict):
        return ParsedHistoryRefMetadata(state="invalid", value=None)
    if SUMMARY_META_HISTORY_REF_KEY not in summary_meta:
        return ParsedHistoryRefMetadata(state="absent", value=None)
    value = summary_meta.get(SUMMARY_META_HISTORY_REF_KEY)
    if not isinstance(value, dict) or set(value) != {"format", "raw_source_hash"}:
        return ParsedHistoryRefMetadata(state="invalid", value=None)
    format_value = value.get("format")
    if format_value not in {HISTORY_REF_FORMAT, HISTORY_REF_LOGICAL_FORMAT}:
        return ParsedHistoryRefMetadata(state="invalid", value=None)
    raw_source_hash = value.get("raw_source_hash")
    if not isinstance(raw_source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", raw_source_hash) is None:
        return ParsedHistoryRefMetadata(state="invalid", value=None)
    return ParsedHistoryRefMetadata(
        state="valid-v1" if format_value == HISTORY_REF_FORMAT else "valid-v2",
        value={"format": str(format_value), "raw_source_hash": raw_source_hash},
    )


def _normalized_history_ref_metadata(summary_meta: Any) -> dict[str, str] | None:
    return _parse_history_ref_metadata(summary_meta).value


def _history_ref_source_handle(
    checkpoint: dict[str, Any],
    *,
    user_message_id: str,
    transient_message_patterns: TransientMessagePatterns | None = None,
    logical_snapshot: LogicalHistorySnapshot | None = None,
    require_logical_snapshot_match: bool = True,
) -> HistoryRefSourceHandle | None:
    metadata = _normalized_history_ref_metadata(checkpoint.get("summary_meta"))
    checkpoint_id = checkpoint.get("id")
    if metadata is None or not isinstance(checkpoint_id, str):
        return None
    if re.fullmatch(r"accp_[0-9a-f]{64}", checkpoint_id) is None:
        return None
    try:
        source_message_count = int(checkpoint.get("source_message_count") or 0)
    except (TypeError, ValueError):
        return None
    if source_message_count <= 0 or not user_message_id:
        return None
    logical_source_snapshot = None
    if metadata["format"] == HISTORY_REF_LOGICAL_FORMAT:
        if logical_snapshot is None:
            return None
        logical_source_snapshot = logical_snapshot
        if (
            require_logical_snapshot_match
            and not _logical_snapshot_matches_checkpoint(
                logical_source_snapshot,
                checkpoint,
            )
        ):
            return None
    return HistoryRefSourceHandle(
        checkpoint_id=checkpoint_id,
        namespace=str(checkpoint.get("namespace") or ""),
        user_id=str(checkpoint.get("user_id") or ""),
        chat_id=str(checkpoint.get("chat_id") or ""),
        pipe_function_id=str(checkpoint.get("pipe_function_id") or ""),
        profile_hash=str(checkpoint.get("profile_hash") or ""),
        source_hash=str(checkpoint.get("source_hash") or ""),
        source_message_count=source_message_count,
        raw_source_hash=metadata["raw_source_hash"],
        user_message_id=user_message_id,
        transient_message_patterns=transient_message_patterns,
        format=metadata["format"],
        logical_snapshot=logical_source_snapshot,
    )


async def build_history_ref_catalog(
    *,
    store: Any,
    selected_checkpoint: dict[str, Any] | None,
    user_message_id: str,
    transient_message_patterns: TransientMessagePatterns | None = None,
    logical_snapshot: LogicalHistorySnapshot | None = None,
) -> tuple[RefCatalogEntry, ...]:
    if not isinstance(selected_checkpoint, dict) or selected_checkpoint.get("state") != "ready":
        return ()
    identity = {
        "namespace": str(selected_checkpoint.get("namespace") or ""),
        "user_id": str(selected_checkpoint.get("user_id") or ""),
        "chat_id": str(selected_checkpoint.get("chat_id") or ""),
        "pipe_function_id": str(selected_checkpoint.get("pipe_function_id") or ""),
        "profile_hash": str(selected_checkpoint.get("profile_hash") or ""),
    }
    if not all(identity.values()) or not user_message_id:
        return ()

    entries: list[RefCatalogEntry] = []
    visited: set[str] = set()
    previous_count: int | None = None
    current = selected_checkpoint
    while True:
        is_selected_checkpoint = previous_count is None
        checkpoint_id = current.get("id")
        if (
            not isinstance(checkpoint_id, str)
            or checkpoint_id in visited
        ):
            return () if previous_count is None else tuple(entries)
        if current.get("state") != "ready" or any(current.get(key) != value for key, value in identity.items()):
            return () if previous_count is None else tuple(entries)
        try:
            count = int(current.get("source_message_count") or 0)
        except (TypeError, ValueError):
            return () if previous_count is None else tuple(entries)
        if count <= 0 or (previous_count is not None and count >= previous_count):
            return () if previous_count is None else tuple(entries)
        visited.add(checkpoint_id)
        previous_count = count

        if is_selected_checkpoint and logical_snapshot is not None:
            await _logical_snapshot_source_hash_async(logical_snapshot, count)
        source = _history_ref_source_handle(
            current,
            user_message_id=user_message_id,
            transient_message_patterns=transient_message_patterns,
            logical_snapshot=logical_snapshot,
            require_logical_snapshot_match=is_selected_checkpoint,
        )
        if source is not None:
            entries.append(
                RefCatalogEntry(
                    manifest=RefManifest(
                        ref=f"history:{checkpoint_id}",
                        utf8_bytes=None,
                        sha256=source.raw_source_hash,
                    ),
                    source=source,
                )
            )

        parent_id = current.get("parent_checkpoint_id")
        if parent_id is None:
            return tuple(entries)
        if not isinstance(parent_id, str) or parent_id in visited:
            return tuple(entries)
        lookup = getattr(store, "lookup_ready_descriptor_by_id", None)
        if not callable(lookup):
            return tuple(entries)
        parent = await lookup(parent_id, **identity)
        if not isinstance(parent, dict):
            return tuple(entries)
        current = parent


async def resolve_history_ref_catalog_entry(
    entry: RefCatalogEntry,
    *,
    request: Any,
    metadata: dict[str, Any],
    transient_message_patterns: TransientMessagePatterns | None = None,
    raw_messages: list[dict[str, Any]] | None = None,
    canonical_source: CanonicalHistorySourceHandle | None = None,
) -> RefCatalogEntry:
    source = entry.source
    if not isinstance(source, HistoryRefSourceHandle):
        return entry
    if str(metadata.get("chat_id") or "") != source.chat_id:
        raise CanonicalHistoryError(reason="current branch mismatch")
    current_user_message_id = str(metadata.get("user_message_id") or "")
    if not current_user_message_id or current_user_message_id != source.user_message_id:
        raise CanonicalHistoryError(reason="current branch mismatch")
    if source.format == HISTORY_REF_LOGICAL_FORMAT:
        snapshot = source.logical_snapshot
        if snapshot is None:
            raise CanonicalHistoryError(reason="logical source snapshot is unavailable")
        if snapshot.identity != (
            source.namespace,
            source.user_id,
            source.chat_id,
            source.pipe_function_id,
            source.profile_hash,
        ):
            raise CanonicalHistoryError(reason="checkpoint source identity verification failed")
        if (
            await _logical_snapshot_source_hash_async(
                snapshot,
                source.source_message_count,
            )
            != source.source_hash
        ):
            raise CanonicalHistoryError(reason="checkpoint source integrity verification failed")
        logical_source = _logical_history_source_handle(
            snapshot,
            source.source_message_count,
        )
        if logical_source.raw_source_hash != source.raw_source_hash:
            raise CanonicalHistoryError(
                reason="logical source hash integrity verification failed"
            )
        return RefCatalogEntry(
            manifest=RefManifest(
                ref=entry.manifest.ref,
                utf8_bytes=logical_source.utf8_bytes,
                sha256=logical_source.raw_source_hash,
            ),
            source=logical_source,
        )
    if raw_messages is None:
        raw_messages = await load_raw_chat_branch(
            chat_id=source.chat_id,
            metadata=metadata,
        )
    if canonical_source is None:
        canonical_source = await build_canonical_history_source(
            raw_messages,
            source_message_count=source.source_message_count,
            transient_message_patterns=transient_message_patterns,
        )
    if canonical_source.raw_source_hash != source.raw_source_hash:
        raise CanonicalHistoryError(reason="raw source hash integrity verification failed")

    from open_webui.utils.middleware import process_messages_with_output

    expanded_messages = await asyncio.to_thread(process_messages_with_output, copy.deepcopy(raw_messages))
    raw_limit = _raw_prefix_len_for_source_count(
        expanded_messages,
        source.source_message_count,
        transient_message_patterns=transient_message_patterns,
    )
    if raw_limit is None:
        raise CanonicalHistoryError(reason="checkpoint source boundary verification failed")
    source_messages = expanded_messages[:raw_limit]
    resolver = await _build_prefix_file_fingerprint_resolver(
        request,
        metadata,
        source_messages,
        transient_message_patterns=transient_message_patterns,
    )
    fingerprint = resolver(source.source_message_count) if resolver is not None else None
    if (
        compute_summary_source_hash(
            source_messages,
            fingerprint,
            _prefix_file_fingerprint_resolver_db_chain(resolver),
            transient_message_patterns=transient_message_patterns,
        )
        != source.source_hash
    ):
        raise CanonicalHistoryError(reason="checkpoint source integrity verification failed")
    return RefCatalogEntry(
        manifest=RefManifest(
            ref=entry.manifest.ref,
            utf8_bytes=canonical_source.utf8_bytes,
            sha256=canonical_source.raw_source_hash,
        ),
        source=canonical_source,
    )


def _default_ref_render_manifest(entry: RefCatalogEntry) -> RefRenderManifest:
    kind: RefKind = "history" if entry.manifest.ref.startswith("history:") else "tool"
    source = entry.source
    line_count = (
        source.line_count
        if isinstance(source, (CanonicalHistorySourceHandle, LogicalHistorySourceHandle))
        else None
    )
    return RefRenderManifest(
        ref=entry.manifest.ref,
        utf8_bytes=entry.manifest.utf8_bytes,
        kind=kind,
        line_count=line_count,
        tool=kind,
    )


def build_history_ref_projection_plan(
    catalog: tuple[RefCatalogEntry, ...],
) -> RefProjectionPlan:
    return RefProjectionPlan(
        catalog=catalog,
        manifests=tuple(entry.manifest for entry in catalog),
        reader_schema=REF_EXEC_TOOL_SPEC if catalog else None,
        render_manifests=tuple(_default_ref_render_manifest(entry) for entry in catalog),
    )


def merge_ref_projection_plans(
    left: RefProjectionPlan | None,
    right: RefProjectionPlan | None,
) -> RefProjectionPlan | None:
    if left is None:
        return right
    if right is None:
        return left
    by_ref = {entry.manifest.ref: entry for entry in left.catalog}
    for entry in right.catalog:
        by_ref.setdefault(entry.manifest.ref, entry)
    catalog = tuple(by_ref.values())
    manifests = tuple(entry.manifest for entry in catalog)
    render_by_ref = {manifest.ref: manifest for manifest in left.render_manifests}
    for manifest in right.render_manifests:
        render_by_ref.setdefault(manifest.ref, manifest)
    for entry in catalog:
        render_by_ref.setdefault(entry.manifest.ref, _default_ref_render_manifest(entry))
    return RefProjectionPlan(
        catalog=catalog,
        manifests=manifests,
        reader_schema=REF_EXEC_TOOL_SPEC if catalog else None,
        render_manifests=tuple(render_by_ref.values()),
    )


def build_summary_ref_projection_plan(
    plan: RefProjectionPlan | None,
    parent_checkpoint: dict[str, Any] | None,
    *,
    ref_mode_active: bool,
) -> RefProjectionPlan | None:
    if not ref_mode_active:
        return None
    catalog = tuple(
        entry for entry in (plan.catalog if plan is not None else ())
        if entry.manifest.ref.startswith("tool:")
    )
    manifests = [
        manifest for manifest in (plan.manifests if plan is not None else ())
        if manifest.ref.startswith("tool:")
    ]
    render_manifests = [
        manifest
        for manifest in (plan.render_manifests if plan is not None else ())
        if manifest.ref.startswith("tool:")
    ]
    if isinstance(parent_checkpoint, dict):
        checkpoint_id = parent_checkpoint.get("id")
        parent_ref = (
            f"history:{checkpoint_id}" if isinstance(checkpoint_id, str) else None
        )
        history_manifest = next(
            (
                manifest
                for manifest in (plan.manifests if plan is not None else ())
                if manifest.ref == parent_ref
            ),
            None,
        )
        history_render_manifest = next(
            (
                manifest
                for manifest in (plan.render_manifests if plan is not None else ())
                if manifest.ref == parent_ref
            ),
            None,
        )
        if parent_ref is not None and history_manifest is not None:
            manifests.append(history_manifest)
            render_manifests.append(
                history_render_manifest
                or RefRenderManifest(
                    ref=parent_ref,
                    utf8_bytes=history_manifest.utf8_bytes,
                    kind="history",
                    line_count=None,
                    tool="history",
                )
            )
    if not catalog and not manifests and not render_manifests:
        return None
    return RefProjectionPlan(
        catalog=catalog,
        manifests=tuple(manifests),
        reader_schema=None,
        render_manifests=tuple(render_manifests),
    )


async def extend_ref_projection_plan_with_checkpoint(
    plan: RefProjectionPlan | None,
    checkpoint: dict[str, Any] | None,
    *,
    request: Any,
    metadata: dict[str, Any],
    transient_message_patterns: TransientMessagePatterns | None = None,
    logical_snapshot: LogicalHistorySnapshot | None = None,
) -> RefProjectionPlan | None:
    user_message_id = str(metadata.get("user_message_id") or "")
    if not isinstance(checkpoint, dict) or not user_message_id:
        return plan
    store = CheckpointStore()
    try:
        enriched = await enrich_checkpoint_history_ref(
            store=store,
            checkpoint=checkpoint,
            request=request,
            metadata=metadata,
            transient_message_patterns=transient_message_patterns,
            logical_snapshot=logical_snapshot,
        )
        if enriched is None:
            raise CanonicalHistoryError(
                reason="selected checkpoint history ref proof failed"
            )
        catalog = await build_history_ref_catalog(
            store=store,
            selected_checkpoint=enriched,
            user_message_id=user_message_id,
            transient_message_patterns=transient_message_patterns,
            logical_snapshot=logical_snapshot,
        )
        selected_ref = f"history:{enriched.get('id')}"
        if not any(entry.manifest.ref == selected_ref for entry in catalog):
            raise CanonicalHistoryError(
                reason="selected checkpoint history ref proof failed"
            )
    except SQLAlchemyError as exc:
        LOG.exception(
            "Auto-compaction history-ref enrichment hit a database error "
            "(chat_id=%s)",
            metadata.get("chat_id"),
        )
        raise HistoryRefStorageUnavailableError(CHECKPOINT_STORE_UNAVAILABLE_MESSAGE) from exc
    except HistoryRefStorageUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - unverified refs must not reach the provider
        raise RefProjectionError(stage="history verification") from exc
    history_plan = build_history_ref_projection_plan(catalog)
    if plan is None:
        return history_plan
    retained_catalog = tuple(
        entry
        for entry in plan.catalog
        if not entry.manifest.ref.startswith("history:")
    )
    retained_render_by_ref = {
        manifest.ref: manifest
        for manifest in plan.render_manifests
        if not manifest.ref.startswith("history:")
    }
    for entry in retained_catalog:
        retained_render_by_ref.setdefault(
            entry.manifest.ref,
            _default_ref_render_manifest(entry),
        )
    merged_catalog = (*retained_catalog, *history_plan.catalog)
    return RefProjectionPlan(
        catalog=merged_catalog,
        manifests=(
            *(
                manifest
                for manifest in plan.manifests
                if not manifest.ref.startswith("history:")
            ),
            *history_plan.manifests,
        ),
        reader_schema=REF_EXEC_TOOL_SPEC if merged_catalog else None,
        render_manifests=(
            *retained_render_by_ref.values(),
            *history_plan.render_manifests,
        ),
    )


async def enrich_checkpoint_history_ref(
    *,
    store: Any,
    checkpoint: dict[str, Any],
    request: Any,
    metadata: dict[str, Any],
    transient_message_patterns: TransientMessagePatterns | None = None,
    logical_snapshot: LogicalHistorySnapshot | None = None,
) -> dict[str, Any] | None:
    user_message_id = str(metadata.get("user_message_id") or "")
    if not user_message_id:
        return None
    current_meta = normalize_summary_meta(checkpoint.get("summary_meta"))
    parsed_existing = _parse_history_ref_metadata(current_meta)
    if parsed_existing.state == "invalid":
        return None
    existing = parsed_existing.value
    if logical_snapshot is not None:
        try:
            warm_count = int(checkpoint.get("source_message_count") or 0)
        except (TypeError, ValueError):
            warm_count = 0
        if warm_count > 0:
            await _logical_snapshot_source_hash_async(logical_snapshot, warm_count)
    use_logical_snapshot = (
        logical_snapshot is not None
        and _logical_snapshot_matches_checkpoint(logical_snapshot, checkpoint)
        and (
            existing is None
            or existing.get("format") == HISTORY_REF_LOGICAL_FORMAT
        )
    )
    if existing is not None and existing.get("format") == HISTORY_REF_LOGICAL_FORMAT:
        if not use_logical_snapshot:
            return None
    source_format = (
        HISTORY_REF_LOGICAL_FORMAT if use_logical_snapshot else HISTORY_REF_FORMAT
    )
    source = HistoryRefSourceHandle(
        checkpoint_id=str(checkpoint.get("id") or ""),
        namespace=str(checkpoint.get("namespace") or ""),
        user_id=str(checkpoint.get("user_id") or ""),
        chat_id=str(checkpoint.get("chat_id") or ""),
        pipe_function_id=str(checkpoint.get("pipe_function_id") or ""),
        profile_hash=str(checkpoint.get("profile_hash") or ""),
        source_hash=str(checkpoint.get("source_hash") or ""),
        source_message_count=int(checkpoint.get("source_message_count") or 0),
        raw_source_hash="0" * 64,
        user_message_id=user_message_id,
        transient_message_patterns=transient_message_patterns,
        format=source_format,
        logical_snapshot=logical_snapshot if use_logical_snapshot else None,
    )
    provisional = RefCatalogEntry(
        manifest=RefManifest(
            ref=f"history:{source.checkpoint_id}",
            utf8_bytes=None,
            sha256=source.raw_source_hash,
        ),
        source=source,
    )
    if use_logical_snapshot:
        assert logical_snapshot is not None
        canonical_source = _logical_history_source_handle(
            logical_snapshot,
            source.source_message_count,
        )
        desired = {
            "format": HISTORY_REF_LOGICAL_FORMAT,
            "raw_source_hash": canonical_source.raw_source_hash,
        }
        source = replace(source, raw_source_hash=canonical_source.raw_source_hash)
        await resolve_history_ref_catalog_entry(
            replace(
                provisional,
                source=source,
                manifest=replace(
                    provisional.manifest,
                    sha256=source.raw_source_hash,
                ),
            ),
            request=request,
            metadata=metadata,
            transient_message_patterns=transient_message_patterns,
        )
    else:
        raw_messages = await load_raw_chat_branch(
            chat_id=source.chat_id,
            metadata=metadata,
        )
        canonical_source = await build_canonical_history_source(
            raw_messages,
            source_message_count=source.source_message_count,
            transient_message_patterns=transient_message_patterns,
        )
        desired = {
            "format": HISTORY_REF_FORMAT,
            "raw_source_hash": canonical_source.raw_source_hash,
        }
        source = replace(source, raw_source_hash=canonical_source.raw_source_hash)
        await resolve_history_ref_catalog_entry(
            replace(
                provisional,
                source=source,
                manifest=replace(
                    provisional.manifest,
                    sha256=source.raw_source_hash,
                ),
            ),
            request=request,
            metadata=metadata,
            transient_message_patterns=transient_message_patterns,
            raw_messages=raw_messages,
            canonical_source=canonical_source,
        )

    if existing is not None:
        return copy.deepcopy(checkpoint) if existing == desired else None
    compare_and_swap = getattr(store, "compare_and_swap_history_ref", None)
    if not callable(compare_and_swap):
        return None
    updated = await compare_and_swap(
        source.checkpoint_id,
        expected_summary_meta=current_meta,
        history_ref=desired,
    )
    if updated:
        result = copy.deepcopy(checkpoint)
        result["summary_meta"] = {**current_meta, SUMMARY_META_HISTORY_REF_KEY: desired}
        return result
    lookup = getattr(store, "lookup_ready_by_id", None)
    if not callable(lookup):
        return None
    winner = await lookup(
        source.checkpoint_id,
        namespace=source.namespace,
        user_id=source.user_id,
        chat_id=source.chat_id,
        pipe_function_id=source.pipe_function_id,
        profile_hash=source.profile_hash,
    )
    if not isinstance(winner, dict):
        return None
    return winner if _normalized_history_ref_metadata(winner.get("summary_meta")) == desired else None


def _is_source_identity_message(
    message: Any,
    *,
    transient_message_patterns: TransientMessagePatterns | None = None,
    transient_message_mask: tuple[bool, ...] | None = None,
    index: int | None = None,
) -> bool:
    if not isinstance(message, dict) or _is_system_message(message):
        return False
    if transient_message_mask is not None and index is not None and index < len(transient_message_mask):
        return not transient_message_mask[index]
    return not _is_transient_message(message, transient_message_patterns)


def _source_identity_message_span(
    message: dict[str, Any],
    *,
    transient_message_patterns: TransientMessagePatterns | None,
) -> int:
    output = message.get("output")
    if message.get("role") != "assistant" or not output:
        return 1

    converted = _convert_core_output_to_messages(output)
    if not converted:
        return 1
    return sum(
        _is_source_identity_message(
            converted_message,
            transient_message_patterns=transient_message_patterns,
        )
        for converted_message in converted
    )


def _source_identity_message_count(
    messages: list[dict[str, Any]],
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> int:
    mask = _transient_message_mask(messages, transient_message_patterns)
    return sum(
        _source_identity_message_span(
            message,
            transient_message_patterns=transient_message_patterns,
        )
        for index, message in enumerate(messages)
        if _is_source_identity_message(
            message,
            transient_message_patterns=transient_message_patterns,
            transient_message_mask=mask,
            index=index,
        )
    )


def _raw_prefix_len_for_source_count(
    messages: list[dict[str, Any]],
    source_message_count: int,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> int | None:
    if source_message_count < 0:
        return None
    if source_message_count == 0:
        return 0
    mask = _transient_message_mask(messages, transient_message_patterns)
    seen = 0
    for index, message in enumerate(messages):
        if not _is_source_identity_message(
            message,
            transient_message_patterns=transient_message_patterns,
            transient_message_mask=mask,
            index=index,
        ):
            continue
        seen += _source_identity_message_span(
            message,
            transient_message_patterns=transient_message_patterns,
        )
        if seen == source_message_count:
            boundary = index + 1
            while boundary < len(messages) and not _is_source_identity_message(
                messages[boundary],
                transient_message_patterns=transient_message_patterns,
                transient_message_mask=mask,
                index=boundary,
            ):
                boundary += 1
            return boundary
        if seen > source_message_count:
            return None
    return None


def _raw_chain_boundary(
    db_chain: list[dict[str, Any]],
    count: int,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> int:
    # Imported or API-created histories can store their own system rows in
    # the DB chain, so a non-system source count is not a raw chain index.
    # Mirrors _raw_prefix_len_for_source_count (including trailing-system
    # absorption) but tolerates non-dict chain entries, counting them as
    # pairable positions like the file-backed image cursor does.
    if count <= 0:
        return 0
    mask = _transient_message_mask(db_chain, transient_message_patterns)
    seen = 0
    for index, message in enumerate(db_chain):
        if isinstance(message, dict) and not _is_source_identity_message(
            message,
            transient_message_patterns=transient_message_patterns,
            transient_message_mask=mask,
            index=index,
        ):
            continue
        seen += 1
        if seen == count:
            boundary = index + 1
            while (
                boundary < len(db_chain)
                and isinstance(db_chain[boundary], dict)
                and not _is_source_identity_message(
                    db_chain[boundary],
                    transient_message_patterns=transient_message_patterns,
                    transient_message_mask=mask,
                    index=boundary,
                )
            ):
                boundary += 1
            return boundary
    return len(db_chain)


def canonicalize_messages_for_source_hash(
    messages: list[dict[str, Any]],
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> list[dict[str, Any]]:
    mask = _transient_message_mask(messages, transient_message_patterns)
    return [
        canonicalize_message_for_source_hash(message)
        for index, message in enumerate(messages)
        if _is_source_identity_message(
            message,
            transient_message_patterns=transient_message_patterns,
            transient_message_mask=mask,
            index=index,
        )
    ]


def compute_source_hash(
    messages: list[dict[str, Any]],
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> str:
    return _json_hash(
        {
            "family": SOURCE_HASH_FAMILY,
            "messages": canonicalize_messages_for_source_hash(
                messages,
                transient_message_patterns=transient_message_patterns,
            ),
        }
    )


def compute_summary_source_hash(
    messages: list[dict[str, Any]],
    prefix_file_fingerprint: str | None = None,
    file_backed_image_db_chain: list[dict[str, Any]] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> str:
    source_messages = _stable_file_backed_image_source_messages(
        messages,
        file_backed_image_db_chain,
        transient_message_patterns=transient_message_patterns,
    )
    # When no prefix files were absorbed (or the DB chain could not be loaded)
    # the fingerprint is empty, so the identity intentionally collapses to the
    # canonical message hash. This keeps existing checkpoints reusable and
    # avoids a family bump.
    if not prefix_file_fingerprint:
        return compute_source_hash(
            source_messages,
            transient_message_patterns=transient_message_patterns,
        )
    return _json_hash(
        {
            "family": SOURCE_HASH_FAMILY,
            "messages": canonicalize_messages_for_source_hash(
                source_messages,
                transient_message_patterns=transient_message_patterns,
            ),
            "prefix_file_fingerprint": prefix_file_fingerprint,
        }
    )


def _canonicalize_prefix_file_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key in _PREFIX_FILE_METADATA_TRANSIENT_KEYS or key in _PREFIX_FILE_METADATA_BODY_KEYS:
                continue
            item = _canonicalize_prefix_file_metadata_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    if isinstance(value, list):
        out_list = []
        for item in value:
            canonical_item = _canonicalize_prefix_file_metadata_value(item)
            if not _is_empty_canonical_value(canonical_item):
                out_list.append(canonical_item)
        return out_list
    return value


def _canonicalize_prefix_embedded_file_identity(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key not in _PREFIX_EMBEDDED_FILE_IDENTITY_KEYS:
                continue
            item = _canonicalize_prefix_file_metadata_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    if isinstance(value, list):
        out_list = []
        for item in value:
            canonical_item = _canonicalize_prefix_embedded_file_identity(item)
            if not _is_empty_canonical_value(canonical_item):
                out_list.append(canonical_item)
        return out_list
    return _canonicalize_prefix_file_metadata_value(value)


def _canonicalize_prefix_file_attachment_identity(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key not in _PREFIX_FILE_ATTACHMENT_IDENTITY_KEYS:
                continue
            if key == "file":
                item = _canonicalize_prefix_embedded_file_identity(value[key])
            else:
                item = _canonicalize_prefix_file_metadata_value(value[key])
            if _is_empty_canonical_value(item):
                continue
            out[key] = item
        return out
    if isinstance(value, list):
        out_list = []
        for item in value:
            canonical_item = _canonicalize_prefix_file_attachment_identity(item)
            if not _is_empty_canonical_value(canonical_item):
                out_list.append(canonical_item)
        return out_list
    return _canonicalize_prefix_file_metadata_value(value)


def _stable_file_fingerprint(file_items: list[dict[str, Any]]) -> str:
    if not file_items:
        return ""
    canonical = [
        _canonicalize_prefix_file_attachment_identity(item)
        for item in file_items
        if isinstance(item, dict)
    ]
    canonical = [item for item in canonical if not _is_empty_canonical_value(item)]
    if not canonical:
        return ""
    ordered = sorted(canonical, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    return _json_hash({"family": PREFIX_FILE_FINGERPRINT_FAMILY, "files": ordered})


def _file_identity_has_stable_discriminator(identity: Any) -> bool:
    if isinstance(identity, list):
        return any(_file_identity_has_stable_discriminator(item) for item in identity)
    if not isinstance(identity, dict):
        return False
    for key, value in identity.items():
        if _is_empty_canonical_value(value):
            continue
        if key in {"type", "content_type", "mime_type"}:
            continue
        if key in {"file", "meta", "metadata", "legacy"}:
            if _file_identity_has_stable_discriminator(value):
                return True
            continue
        if key in _FILE_IDENTITY_STABLE_DISCRIMINATOR_KEYS:
            return True
    return False


def _messages_have_user_image_url_parts(messages: list[dict[str, Any]] | None) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False


def _stable_file_backed_image_source_messages(
    source_messages: list[dict[str, Any]],
    db_chain: list[dict[str, Any]] | None,
    *,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> list[dict[str, Any]]:
    if not db_chain:
        return source_messages

    normalized = copy.deepcopy(source_messages)
    # Pair body messages with DB-chain entries by non-system position on
    # BOTH sides: filter-injected system messages exist only in the request
    # body and churn between turns, while imported or API-created histories
    # can store their own system rows in the chain.
    chain_index = 0
    source_mask = _transient_message_mask(normalized, transient_message_patterns)
    chain_mask = _transient_message_mask(db_chain, transient_message_patterns)
    for source_index, message in enumerate(normalized):
        if not _is_source_identity_message(
            message,
            transient_message_patterns=transient_message_patterns,
            transient_message_mask=source_mask,
            index=source_index,
        ):
            continue
        while (
            chain_index < len(db_chain)
            and isinstance(db_chain[chain_index], dict)
            and not _is_source_identity_message(
                db_chain[chain_index],
                transient_message_patterns=transient_message_patterns,
                transient_message_mask=chain_mask,
                index=chain_index,
            )
        ):
            chain_index += 1
        if chain_index >= len(db_chain):
            break
        db_message = db_chain[chain_index]
        chain_index += 1
        if not isinstance(db_message, dict):
            continue
        if message.get("role") != "user":
            continue
        if db_message.get("role") != "user":
            continue
        files = db_message.get("files")
        if not isinstance(files, list):
            continue
        image_files = [item for item in files if _is_image_file_item(item) and item.get("url")]
        if not image_files:
            continue
        if not isinstance(db_message.get("content"), str):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        image_part_indexes = [
            part_index
            for part_index, part in enumerate(content)
            if isinstance(part, dict)
            and part.get("type") == "image_url"
            and isinstance(part.get("image_url"), dict)
        ]
        if len(image_part_indexes) != len(image_files):
            continue

        identities: list[dict[str, Any]] = []
        for image_file in image_files:
            identity = _canonicalize_prefix_file_attachment_identity(image_file)
            if _is_empty_canonical_value(identity) or not _file_identity_has_stable_discriminator(identity):
                identities = []
                break
            identities.append(identity)
        if len(identities) != len(image_part_indexes):
            continue

        for part_index, identity in zip(image_part_indexes, identities):
            part = copy.deepcopy(content[part_index])
            image_url = part.get("image_url")
            stable_image_url = {
                key: value
                for key, value in image_url.items()
                if key != "url"
            }
            stable_image_url["file"] = identity
            part["image_url"] = stable_image_url
            content[part_index] = part

    return normalized


def _prefix_file_fingerprint_resolver_db_chain(
    resolver: Callable[[int], str | None] | None,
) -> list[dict[str, Any]] | None:
    if resolver is None:
        return None
    db_chain = getattr(resolver, PREFIX_FILE_FINGERPRINT_RESOLVER_DB_CHAIN_ATTR, None)
    return db_chain if isinstance(db_chain, list) else None


def _soft_prefetch_inflight_source_hash(
    source_messages: list[dict[str, Any]],
    metadata: dict[str, Any],
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> str:
    files = metadata.get("files")
    if not isinstance(files, list):
        return compute_source_hash(
            source_messages,
            transient_message_patterns=transient_message_patterns,
        )
    fingerprint = _stable_file_fingerprint(
        [item for item in files if isinstance(item, dict) and not _is_image_file_item(item)]
    )
    return compute_summary_source_hash(
        source_messages,
        fingerprint,
        transient_message_patterns=transient_message_patterns,
    )


def _soft_prefetch_inflight_key_for_body(
    *,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    transient_message_patterns: TransientMessagePatterns | None = None,
    source_messages: list[dict[str, Any]] | None = None,
    checkpoint_profile_hash: str | None = None,
) -> tuple[str, str, str, str, str, str] | None:
    if source_messages is None:
        prefetch_source = _soft_prefetch_source_messages(
            body,
            transient_message_patterns=transient_message_patterns,
        )
        if prefetch_source is None:
            return None
        source_messages, _ = prefetch_source
    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user or {}).get("id") or "")
    if not user_id or not _chat_id_supported(chat_id):
        return None
    return (
        CHECKPOINT_NAMESPACE,
        user_id,
        chat_id,
        pipe_function_id,
        checkpoint_profile_hash or ACTIVE_CHECKPOINT_PROFILE_HASH,
        _soft_prefetch_inflight_source_hash(
            source_messages,
            metadata,
            transient_message_patterns=transient_message_patterns,
        ),
    )


def _soft_prefetch_inflight_task_for_body(
    *,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    transient_message_patterns: TransientMessagePatterns | None = None,
    inflight_key: tuple[str, str, str, str, str, str] | None = None,
    checkpoint_profile_hash: str | None = None,
) -> asyncio.Task | None:
    key = inflight_key or _soft_prefetch_inflight_key_for_body(
        user=user,
        metadata=metadata,
        body=body,
        pipe_function_id=pipe_function_id,
        transient_message_patterns=transient_message_patterns,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )
    if key is None:
        return None
    task = _SOFT_PREFETCH_INFLIGHT_TASKS.get(key)
    if task is None or task.done():
        return None
    return task


def _make_prefix_file_fingerprint_resolver(
    db_chain: list[dict[str, Any]] | None,
    metadata_files: Any,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> Callable[[int], str | None]:
    # Fingerprints cover all non-image prefix files up to the DB-chain
    # boundary for `count` non-system entries. Counts are non-system source
    # message counts (the same basis as checkpoint source_message_count):
    # filter-injected system messages exist only in the request body and
    # churn between turns, while imported or API-created histories can store
    # their own system rows in the chain, so neither side is safe to index
    # raw. The frozen table below derives that boundary and the file
    # identities once, at construction; each count then resolves from the
    # table without touching the chain or metadata files again. Using the
    # full prefix range (rather than the delta
    # [parent_count, compaction_prefix_count)) keeps the hash basis identical
    # for a checkpoint and any of its potential children, so parent matching
    # and parent validation use the same fingerprint the stored row was built
    # with. The set is order-independent because the prefix ids form a set
    # and the summary file context only depends on which files are present
    # in the prefix, not their positional order.
    chain_length = len(db_chain) if db_chain else 0
    mask = (
        _transient_message_mask(db_chain, transient_message_patterns)
        if db_chain
        else None
    )
    seen_positions: list[int] = []
    first_index_by_id: dict[str, int] = {}
    if db_chain:
        for index, message in enumerate(db_chain):
            if not isinstance(message, dict):
                seen_positions.append(index)
            elif _is_source_identity_message(
                message,
                transient_message_patterns=transient_message_patterns,
                transient_message_mask=mask,
                index=index,
            ):
                seen_positions.append(index)
            if _is_source_identity_message(
                message,
                transient_message_patterns=transient_message_patterns,
            ):
                for file_id in _extract_non_image_file_ids(message.get("files")):
                    first_index_by_id.setdefault(file_id, index)
    # Attachment identities only: docs/content/file bodies never enter the
    # fingerprint, so the frozen table must not retain them.
    frozen_files = tuple(
        _canonicalize_prefix_file_attachment_identity(item)
        for item in metadata_files
        if isinstance(item, dict) and not _is_image_file_item(item)
    )
    seen_position_index = tuple(seen_positions)
    fingerprint_cache: dict[int, str | None] = {}

    def table_fingerprint(count: int) -> str | None:
        if count in fingerprint_cache:
            return fingerprint_cache[count]
        if count <= 0:
            boundary = 0
        elif count < len(seen_position_index):
            boundary = seen_position_index[count]
        else:
            boundary = chain_length
        prefix_ids = {
            file_id
            for file_id, index in first_index_by_id.items()
            if index < boundary
        }
        result: str | None = None
        if prefix_ids:
            result = (
                _stable_file_fingerprint(
                    [item for item in frozen_files if item.get("id") in prefix_ids]
                )
                or None
            )
        fingerprint_cache[count] = result
        return result

    def resolve(count: int) -> str | None:
        return table_fingerprint(count)

    setattr(resolve, PREFIX_FILE_FINGERPRINT_RESOLVER_DB_CHAIN_ATTR, db_chain)
    setattr(resolve, PREFIX_FILE_FINGERPRINT_RESOLVER_FROZEN_ATTR, table_fingerprint)
    return resolve


async def _build_prefix_file_fingerprint_resolver(
    request: Any,
    metadata: dict[str, Any],
    source_messages: list[dict[str, Any]] | None = None,
    *,
    require_file_context_chain: bool = False,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> Callable[[int], str | None] | None:
    chat_id = str(metadata.get("chat_id") or "")
    current_message_id = str(metadata.get("user_message_id") or metadata.get("message_id") or "")
    cache_key = (chat_id, current_message_id)
    request_state = getattr(request, "state", None)
    cache = (
        getattr(request_state, PREFIX_FILE_FINGERPRINT_RESOLVER_STATE_KEY, None)
        if request_state is not None
        else None
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    metadata_files = metadata.get("files")
    if not isinstance(metadata_files, list) or not metadata_files:
        if not _messages_have_user_image_url_parts(source_messages):
            return None
        metadata_files = []
    required_file_ids = _extract_non_image_file_ids(metadata_files)
    # Cache the resolver on request.state so multiple checkpoint operations in
    # the same request share one DB-chain load. Different (chat_id, message_id)
    # pairs get independent resolvers.
    if request_state is not None:
        if cache is None:
            cache = {}
            with suppress(Exception):
                setattr(request_state, PREFIX_FILE_FINGERPRINT_RESOLVER_STATE_KEY, cache)
    try:
        db_chain = await _load_chat_message_chain(request, chat_id, current_message_id)
    except Exception:
        LOG.exception("Failed to load chat message chain for prefix file fingerprint")
        return None
    if not db_chain:
        if require_file_context_chain and required_file_ids:
            raise SummaryFileContextUnavailable()
        return None
    # Freeze the attachment identities on the event loop, before the worker
    # handoff, so later mutations of metadata.files cannot leak into
    # fingerprints computed in the worker. Canonicalizing first also keeps
    # file bodies (docs/content) out of the deep-copied payload.
    frozen_files = copy.deepcopy(
        [
            _canonicalize_prefix_file_attachment_identity(item)
            for item in metadata_files
            if isinstance(item, dict) and not _is_image_file_item(item)
        ]
    )
    resolver = await asyncio.to_thread(
        _make_prefix_file_fingerprint_resolver,
        db_chain,
        frozen_files,
        transient_message_patterns,
    )
    if cache is not None:
        with suppress(Exception):
            cache[cache_key] = resolver
    return resolver


def _resolve_fingerprint(
    resolver: Callable[[int], str | None] | None,
    count: int,
) -> str | None:
    if resolver is None:
        return None
    try:
        return resolver(count)
    except Exception:
        return None


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


ACTIVE_CHECKPOINT_PROFILE_HASH = compute_profile_hash()
INACTIVE_CHECKPOINT_PROFILE_DISCRIMINATOR = "ref-mode-inactive-v1"


def checkpoint_profile_hash_for_ref_mode(*, ref_mode_active: bool) -> str:
    if ref_mode_active:
        return ACTIVE_CHECKPOINT_PROFILE_HASH
    return _json_hash(
        {
            "active_profile_hash": ACTIVE_CHECKPOINT_PROFILE_HASH,
            "discriminator": INACTIVE_CHECKPOINT_PROFILE_DISCRIMINATOR,
        }
    )


def _request_state_tiktoken_encoding_name(request: Any) -> str | None:
    request_state = getattr(request, "state", None)
    if request_state is None:
        return None
    try:
        configured = getattr(request_state, AUTO_COMPACT_TIKTOKEN_ENCODING_STATE_KEY, None)
    except Exception:
        return None
    if isinstance(configured, str) and configured:
        return configured
    return None


async def _refresh_tiktoken_encoding_config(request: Any = None) -> None:
    if request is None:
        return
    request_state = getattr(request, "state", None)
    if request_state is None:
        return
    if getattr(request_state, AUTO_COMPACT_TIKTOKEN_ENCODING_LOADED_STATE_KEY, False) is True:
        return
    configured = await _open_webui_config_get(TIKTOKEN_ENCODING_CONFIG_KEY)
    with suppress(Exception):
        setattr(request_state, AUTO_COMPACT_TIKTOKEN_ENCODING_LOADED_STATE_KEY, True)
    if configured is CONFIG_VALUE_MISSING or configured is None or not str(configured).strip():
        return
    with suppress(Exception):
        setattr(request_state, AUTO_COMPACT_TIKTOKEN_ENCODING_STATE_KEY, str(configured))


def _configured_tiktoken_encoding_names(request: Any = None) -> list[str]:
    names: list[str] = []
    cached_configured = _request_state_tiktoken_encoding_name(request)
    if cached_configured:
        names.append(cached_configured)
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
    if len(_MESSAGE_TOKEN_ESTIMATE_CACHE) >= MESSAGE_TOKEN_ESTIMATE_CACHE_MAX_ENTRIES:
        with suppress(Exception):
            _MESSAGE_TOKEN_ESTIMATE_CACHE.pop(next(iter(_MESSAGE_TOKEN_ESTIMATE_CACHE)))
    _MESSAGE_TOKEN_ESTIMATE_CACHE[key] = int(count)


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
    eligible = utf8_bytes > MESSAGE_TOKEN_EXACT_ENCODE_MAX_BYTES
    resolved_encoder = encoder
    if not eligible:
        if resolved_encoder is None:
            resolved_encoder, _ = _get_tiktoken_encoder(request)
        if resolved_encoder is None:
            encoder_failed = True
        else:
            token_count = _encode_text_token_count(resolved_encoder, text)
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
    if encoder is None:
        await _refresh_tiktoken_encoding_config(request)
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
        encoder, _ = _get_tiktoken_encoder(request)
    key = (ref, threshold_tokens, id(encoder))
    cached = cache.get(key)
    if cached is not None:
        return cached[1]
    preview = _render_tool_ref_preview_sync(
        text, ref, utf8_bytes, threshold_tokens=threshold_tokens, encoder=encoder
    )
    if preview is not None:
        # Retain the encoder to prevent ID reuse within this request. The
        # cache holds bounded renders, not raw sources or reader bindings.
        cache[key] = (encoder, preview)
    return preview


async def project_native_tool_texts(
    messages: list[dict[str, Any]],
    *,
    threshold_tokens: int,
    encoder: Any = None,
    request: Any = None,
) -> RefProjectionPlan:
    catalog_by_hash: dict[str, RefCatalogEntry] = {}
    render_manifest_by_hash: dict[str, RefRenderManifest] = {}
    names_by_call_id = _native_tool_names_by_call_id(messages)
    state = getattr(request, "state", None)
    preview_cache = getattr(state, REQUEST_STATE_REF_PREVIEW_CACHE_KEY, None)
    if preview_cache is None:
        preview_cache = {}
        if state is not None:
            setattr(state, REQUEST_STATE_REF_PREVIEW_CACHE_KEY, preview_cache)

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
        classification = await classify_ref_text(
            content,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            request=request,
        )
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
        render_manifest_by_hash.setdefault(
            text_hash,
            RefRenderManifest(
                ref=ref,
                utf8_bytes=utf8_bytes,
                kind="tool",
                line_count=line_count,
                tool=paired_name,
            ),
        )
    catalog = tuple(catalog_by_hash.values())
    return RefProjectionPlan(
        catalog=catalog,
        manifests=tuple(entry.manifest for entry in catalog),
        reader_schema=REF_EXEC_TOOL_SPEC if catalog else None,
        render_manifests=tuple(render_manifest_by_hash.values()),
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


def _ref_manifest_payloads(plan: RefProjectionPlan) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for manifest in plan.manifests:
        payload: dict[str, Any] = {
            "ref": manifest.ref,
            "sha256": manifest.sha256,
        }
        if manifest.utf8_bytes is not None:
            payload["utf8_bytes"] = manifest.utf8_bytes
        payloads.append(payload)
    return payloads


def ref_render_manifest_payloads(plan: RefProjectionPlan) -> list[dict[str, Any]]:
    manifests = plan.render_manifests or tuple(
        _default_ref_render_manifest(entry) for entry in plan.catalog
    )
    return [
        {
            "bytes": manifest.utf8_bytes,
            "kind": manifest.kind,
            "lines": manifest.line_count,
            "ref": manifest.ref,
            "tool": manifest.tool,
            "version": manifest.version,
        }
        for manifest in manifests
    ]


def _require_provider_bound_history_ref_manifests(
    body: dict[str, Any],
    plan: RefProjectionPlan | None,
) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    manifest_refs = {manifest.ref for manifest in plan.manifests} if plan is not None else set()
    render_refs = (
        {
            manifest.ref
            for manifest in plan.render_manifests
            if manifest.kind == "history"
        }
        if plan is not None
        else set()
    )
    proven_refs = manifest_refs & render_refs
    for message in messages:
        if not isinstance(message, dict) or not _is_rendered_summary_context_message(
            message
        ):
            continue
        history_ref = getattr(message, "history_ref", None)
        if isinstance(history_ref, str) and history_ref not in proven_refs:
            raise RefProjectionError(stage="history manifest proof")


def _apply_ref_manifests(body: dict[str, Any], plan: RefProjectionPlan) -> None:
    _require_provider_bound_history_ref_manifests(body, plan)
    if not plan.manifests:
        return
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        body["metadata"] = metadata
    metadata["auto_compact_ref_manifests"] = _ref_manifest_payloads(plan)
    render_manifest_payloads = ref_render_manifest_payloads(plan)
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if isinstance(message, dict) and _is_rendered_summary_context_message(message):
            context_prefix, context_close, trailing = message["content"].rpartition(
                "</auto_compaction_context>"
            )
            if not context_close or trailing:
                continue
            context_prefix = re.sub(
                r'<auto_compact_ref_manifests version="1">'
                r'(?:<!\[CDATA\[(?:(?!\]\]>).)*\]\]>)+'
                r'</auto_compact_ref_manifests>\n?\Z',
                "",
                context_prefix,
                flags=re.DOTALL,
            )
            history_ref = getattr(message, "history_ref", None)
            if not isinstance(history_ref, str):
                continue
            own_manifest = next(
                (
                    manifest
                    for manifest in render_manifest_payloads
                    if manifest["ref"] == history_ref and manifest["kind"] == "history"
                ),
                None,
            )
            if own_manifest is None:
                raise RefProjectionError(stage="history manifest proof")
            compact_json = json.dumps(
                [own_manifest],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            manifest_block = (
                '<auto_compact_ref_manifests version="1">'
                f"{_xml_cdata(compact_json)}"
                "</auto_compact_ref_manifests>"
            )
            message["content"] = f"{context_prefix}{manifest_block}\n{context_close}"


def _mutable_ref_schema_value(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _mutable_ref_schema_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_ref_schema_value(item) for item in value]
    return value


def apply_ref_projection_surfaces(
    body: dict[str, Any],
    plan: RefProjectionPlan,
    *,
    include_reader_schema: bool,
) -> None:
    _apply_ref_manifests(body, plan)
    if not include_reader_schema or not plan.catalog or plan.reader_schema is None:
        return
    spec = _mutable_ref_schema_value(plan.reader_schema)
    if not isinstance(spec, dict):
        raise RefProjectionError(stage="reader schema rendering")
    tools = body.get("tools")
    if tools is None:
        body["tools"] = [spec]
    elif isinstance(tools, list):
        if spec not in tools:
            tools.append(spec)
    else:
        raise RefProjectionError(stage="reader schema registration")


def _ref_request_store(request: Any, *, create: bool) -> RefRequestStore | None:
    state = getattr(request, "state", None)
    if state is None:
        return None
    store = getattr(state, REQUEST_STATE_REF_STORE_KEY, None)
    if isinstance(store, RefRequestStore):
        return store
    if not create:
        return None
    store = RefRequestStore()
    setattr(state, REQUEST_STATE_REF_STORE_KEY, store)
    return store


def preflight_attached_ref_registry(
    metadata: dict[str, Any] | None,
    injected_tools: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(metadata, dict) or not isinstance(injected_tools, dict):
        return None
    attached = metadata.get("tools")
    if not isinstance(attached, dict) or attached is not injected_tools:
        return None
    return attached


def _binding_matches_reentry(candidate: RefBindingKey, owned: RefBindingKey) -> bool:
    return (
        candidate.user_id == owned.user_id
        and candidate.chat_id == owned.chat_id
        and candidate.user_message_id == owned.user_message_id
        and candidate.incoming_model_id == owned.incoming_model_id
        and candidate.base_pipe_id == owned.base_pipe_id
        and candidate.profile_hash == owned.profile_hash
        and candidate.branch_anchor == owned.branch_anchor
    )


def _warn_shared_registry_mapping(request: Any, key: RefBindingKey) -> None:
    with _REF_SHARED_REGISTRY_WARNED_LOCK:
        try:
            if request in _REF_SHARED_REGISTRY_WARNED_REQUESTS:
                return
            _REF_SHARED_REGISTRY_WARNED_REQUESTS[request] = True
        except TypeError:
            return
    values = (
        key.user_id,
        key.chat_id,
        key.user_message_id,
        key.assistant_message_id,
        key.incoming_model_id,
        key.base_pipe_id,
        key.profile_hash,
        key.branch_anchor,
    )
    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    label = hmac.new(_REF_BINDING_LABEL_HMAC_KEY, payload, hashlib.sha256).hexdigest()[:16]
    LOG.warning("shared_registry_mapping binding=%s", label)


async def reserve_ref_binding(
    request: Any,
    key: RefBindingKey,
    registry: dict[str, Any],
) -> RefReservation | None:
    existing = registry.get(REF_EXEC_TOOL_NAME)
    store = _ref_request_store(request, create=False)
    if existing is not None and store is None:
        return None
    store = _ref_request_store(request, create=True)
    if store is None:
        return None
    async with store.lock:
        owner_key = store.registry_owners.get(id(registry))
        owner_binding = store.bindings.get(owner_key) if owner_key is not None else None
        if owner_key is not None:
            if (
                owner_binding is None
                or owner_binding.registry is not registry
                or not _binding_matches_reentry(key, owner_key)
            ):
                _warn_shared_registry_mapping(request, key)
                return None
            if (
                not isinstance(existing, dict)
                or existing.get("spec") != ref_exec_tool_spec_payload()["function"]
                or existing.get("callable") is not owner_binding.reader
            ):
                return None
            key = owner_key
        elif existing is not None:
            return None

        binding = store.bindings.get(key)
        if binding is not None and binding.registry is not registry:
            _warn_shared_registry_mapping(request, key)
            return None
        pending = store.registry_reservations.get(id(registry))
        if pending is not None and pending.registry is registry and pending.key != key:
            _warn_shared_registry_mapping(request, key)
            return None

        replaced = store.reservations.get(key)
        if replaced is not None and replaced.registry is not registry:
            _warn_shared_registry_mapping(request, key)
            return None
        store.next_generation += 1
        reservation = RefReservation(
            key=key,
            generation=store.next_generation,
            registry=registry,
        )
        cleanup = store.deferred_cleanups.get(key)
        if (
            cleanup is not None
            and replaced is not None
            and pending is replaced
            and replaced.key == key
            and replaced.registry is registry
            and cleanup.successor_generation == replaced.generation
            and cleanup.registry is registry
        ):
            store.deferred_cleanups[key] = RefDeferredCleanup(
                generation=cleanup.generation,
                successor_generation=reservation.generation,
                registry=cleanup.registry,
                reader=cleanup.reader,
            )
        store.reservations[key] = reservation
        store.registry_reservations[id(registry)] = reservation
        return reservation


async def release_ref_reservation(request: Any, reservation: RefReservation) -> None:
    store = _ref_request_store(request, create=False)
    if store is None:
        return
    async with store.lock:
        _release_ref_reservation_locked(store, reservation)


def _binding_matches_deferred_cleanup(
    binding: RefBindingState,
    cleanup: RefDeferredCleanup,
) -> bool:
    return (
        binding.generation == cleanup.generation
        and binding.registry is cleanup.registry
        and binding.reader is cleanup.reader
    )


def _teardown_ref_binding_locked(
    store: RefRequestStore,
    key: RefBindingKey,
    binding: RefBindingState,
) -> bool:
    if store.bindings.get(key) is not binding:
        return False
    store.bindings.pop(key, None)
    if store.registry_owners.get(id(binding.registry)) == key:
        store.registry_owners.pop(id(binding.registry), None)
    existing = binding.registry.get(REF_EXEC_TOOL_NAME)
    if isinstance(existing, dict) and existing.get("callable") is binding.reader:
        binding.registry.pop(REF_EXEC_TOOL_NAME, None)
    _invalidate_ref_reader(binding.reader)
    return True


def _deferred_cleanup_matches_attempt(
    cleanup: RefDeferredCleanup,
    attempt: RefAttempt,
    binding: RefBindingState,
) -> bool:
    previous = attempt.previous_binding
    return (
        previous is not None
        and cleanup.successor_generation == attempt.generation
        and cleanup.registry is attempt.registry
        and cleanup.reader is attempt.reader
        and _binding_matches_deferred_cleanup(previous, cleanup)
        and binding.generation == attempt.generation
        and binding.registry is attempt.registry
        and binding.reader is attempt.reader
    )


def _deferred_cleanup_targets_binding(
    cleanup: RefDeferredCleanup,
    binding: RefBindingState,
) -> bool:
    return (
        cleanup.successor_generation == binding.generation
        and cleanup.registry is binding.registry
        and cleanup.reader is binding.reader
    )


def _complete_deferred_ref_cleanup_locked(
    store: RefRequestStore,
    reservation: RefReservation,
) -> None:
    cleanup = store.deferred_cleanups.get(reservation.key)
    binding = store.bindings.get(reservation.key)
    if (
        cleanup is None
        or binding is None
        or cleanup.successor_generation != reservation.generation
        or reservation.registry is not cleanup.registry
        or not _binding_matches_deferred_cleanup(binding, cleanup)
    ):
        return
    store.deferred_cleanups.pop(reservation.key, None)
    _teardown_ref_binding_locked(store, reservation.key, binding)


def _release_ref_reservation_locked(
    store: RefRequestStore,
    reservation: RefReservation,
) -> bool:
    if store.reservations.get(reservation.key) is not reservation:
        return False
    store.reservations.pop(reservation.key, None)
    pending = store.registry_reservations.get(id(reservation.registry))
    if pending is reservation:
        store.registry_reservations.pop(id(reservation.registry), None)
    _complete_deferred_ref_cleanup_locked(store, reservation)
    return True


def _ref_registry_available(
    request: Any,
    key: RefBindingKey,
    registry: dict[str, Any],
) -> bool:
    existing = registry.get(REF_EXEC_TOOL_NAME)
    store = _ref_request_store(request, create=False)
    if store is None:
        return existing is None
    owner_key = store.registry_owners.get(id(registry))
    if owner_key is not None:
        binding = store.bindings.get(owner_key)
        if (
            binding is None
            or binding.registry is not registry
            or not _binding_matches_reentry(key, owner_key)
        ):
            _warn_shared_registry_mapping(request, key)
            return False
        return (
            isinstance(existing, dict)
            and existing.get("callable") is binding.reader
            and existing.get("spec") == ref_exec_tool_spec_payload()["function"]
        )
    pending = store.registry_reservations.get(id(registry))
    if pending is not None and pending.registry is registry and pending.key != key:
        _warn_shared_registry_mapping(request, key)
        return False
    return existing is None


def parse_ref(value: str) -> ParsedRef | None:
    tool_match = re.fullmatch(r"tool:([0-9a-f]{64})", value)
    if tool_match is not None:
        return ParsedRef(kind="tool", value=tool_match.group(1))
    history_match = re.fullmatch(r"history:(accp_[0-9a-f]{64})", value)
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
                f"Error: usage: {command} [-n N|-N|-c N{'|-c +N' if command == 'tail' else ''}] [REF]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
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
                f"Error: usage: {command} [-n N|-N|-c N{'|-c +N' if command == 'tail' else ''}] [REF]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
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
                "Error: usage: grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN [REF]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
            )
        flags.update(combined)
    expected = 2 if source else 1
    if len(remaining) != expected:
        raise RefExecError(
            "Error: usage: grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN [REF]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
        )
    ref = remaining[1] if source else None
    if ref is not None and parse_ref(ref) is None:
        raise RefExecError(
            "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
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
            "Error: usage: sed -n Np|M,Np|M,$p [REF]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
        )
    selection = tokens[1]
    match = re.fullmatch(r"([1-9][0-9]*)(?:,([1-9][0-9]*|\$))?p", selection)
    if match is None:
        raise RefExecError(
            "Error: usage: sed -n Np|M,Np|M,$p [REF]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
        )
    start = int(match.group(1))
    end_value = match.group(2)
    end = start if end_value is None else (None if end_value == "$" else int(end_value))
    if end is not None and start > end:
        raise RefExecError("Error: sed range start exceeds end")
    ref = tokens[2] if source else None
    if ref is not None and parse_ref(ref) is None:
        raise RefExecError(
            "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
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
            f"Error: unknown command. Available: {available}. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
        )
    if not source and command not in {"grep", "head", "sed", "tail", "wc"}:
        raise RefExecError("Error: command is not a valid piped consumer")
    if command == "ls":
        if len(arguments) > 1 or (arguments and arguments[0] not in {"history", "tool"}):
            raise RefExecError(
                "Error: usage: ls [history|tool]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
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
                f"Error: usage: {command} [-n N|-N|-c N{'|-c +N' if command == 'tail' else ''}] [REF]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
            )
        ref = remaining[0] if source else None
        if ref is not None and parse_ref(ref) is None:
            raise RefExecError(
                "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
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
                "Error: usage: wc [-l|-w|-c] [REF]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
            )
        expected = 2 if source else 1
        if len(arguments) != expected:
            raise RefExecError(
                "Error: usage: wc [-l|-w|-c] [REF]. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
            )
        ref = arguments[1] if source else None
        if ref is not None and parse_ref(ref) is None:
            raise RefExecError(
                "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
            )
        return RefExecStage(command=command, ref=ref, flags=frozenset({arguments[0][1:]}))
    if len(arguments) != 1 or parse_ref(arguments[0]) is None:
        raise RefExecError(
            f"Error: usage: {command} REF. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
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
    if isinstance(source, HistoryRefSourceHandle):
        raise RefExecError("Error: externalized history ref is unresolved")
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
    if isinstance(source, HistoryRefSourceHandle):
        raise RefExecError("Error: externalized history ref is unresolved")
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


def _ref_exec_tail(lines: Iterable[RefExecLine], count: int, cancelled: threading.Event) -> Iterable[RefExecLine]:
    retained: deque[RefExecLine] = deque()
    metadata: RefExecLine | None = None
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            metadata = line
            continue
        retained.append(line)
        while len(retained) > count:
            retained.popleft()
    for line in retained:
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
    token_count = _encode_text_token_count(encoder, text)
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
        "\n<auto_compact_ref_truncated>"
        f"{payload}"
        "</auto_compact_ref_truncated>"
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
        marker = f"\n<auto_compact_ref_excerpt>{marker_payload}</auto_compact_ref_excerpt>"
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
    return f"\n<auto_compact_ref_range>{payload}</auto_compact_ref_range>"


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
        entry = next(
            (
                candidate
                for candidate in catalog
                if candidate.manifest.ref == (first.ref or "")
            ),
            None,
        )
        if entry is None:
            raise RefExecError(
                "Error: externalized ref is not available in this binding. Expected REF: tool:<64 hex> or history:accp_<64 hex>"
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
                lines = _ref_exec_tail(lines, first.count or 0, cancelled)
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


def _new_ref_reader(
    request: Any,
    key: RefBindingKey,
    *,
    threshold_tokens: int,
    encoder: Any = None,
) -> Callable[[str], Awaitable[str]]:
    request_state: dict[str, Any] = {"request": None}
    resolved_history_refs: dict[str, RefCatalogEntry] = {}
    try:
        request_ref = weakref.ref(request)
    except TypeError:
        request_ref = None
        request_state["request"] = request

    async def reader(command: str = "") -> str:
        """Inspect one binding-local externalized ref with bounded virtual reader commands.

        Follow the next command in truncation markers to read omitted spans across pages.

        :param command: Use ls, stat, wc, head, tail, sed -n, grep, or cat and optional bounded pipelines.
        """
        try:
            if not isinstance(command, str):
                return REF_EXEC_USAGE_ERROR
            active_request = (
                request_ref() if request_ref is not None else request_state["request"]
            )
            if active_request is None:
                return "Error: externalized ref binding is unavailable"
            stages = _parse_ref_exec_command(command)
            first_ref = stages[0].ref
            store = _ref_request_store(active_request, create=False)
            binding = store.bindings.get(key) if store is not None else None
            if binding is None or binding.reader is not reader:
                return "Error: externalized ref binding is unavailable"
            catalog = binding.catalog
            if first_ref is not None:
                requested = next(
                    (entry for entry in catalog if entry.manifest.ref == first_ref),
                    None,
                )
                if requested is not None and isinstance(requested.source, HistoryRefSourceHandle):
                    is_v2_history_ref = (
                        requested.source.format == HISTORY_REF_LOGICAL_FORMAT
                    )
                    resolved = (
                        None
                        if is_v2_history_ref
                        else resolved_history_refs.get(first_ref)
                    )
                    if resolved is None:
                        try:
                            resolved = await resolve_history_ref_catalog_entry(
                                requested,
                                request=active_request,
                                metadata={
                                    "chat_id": requested.source.chat_id,
                                    "user_message_id": requested.source.user_message_id,
                                },
                                transient_message_patterns=requested.source.transient_message_patterns,
                            )
                        except CanonicalHistoryError as exc:
                            return f"Error: externalized history ref unavailable: {exc.reason}"
                        if not is_v2_history_ref:
                            resolved_history_refs[first_ref] = resolved
                    catalog = tuple(
                        resolved if entry.manifest.ref == first_ref else entry
                        for entry in catalog
                    )
            cancelled = threading.Event()
            resolved_encoder = encoder
            if resolved_encoder is None and stages[-1].command != "wc":
                resolved_encoder, _ = _get_tiktoken_encoder(active_request)
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
            LOG.exception(
                "Unexpected externalized ref reader failure (chat_id=%s)",
                key.chat_id,
            )
            return "Error: externalized ref reader is unavailable"

    reader.__name__ = f"ref_reader_{hash(key)}"
    reader._auto_compact_request_state = request_state
    return reader


def _invalidate_ref_reader(reader: Callable[[str], Awaitable[str]]) -> None:
    request_state = getattr(reader, "_auto_compact_request_state", None)
    if isinstance(request_state, dict):
        request_state["request"] = None


def stage_ref_attempt(
    request: Any,
    reservation: RefReservation,
    plan: RefProjectionPlan,
    *,
    threshold_tokens: int = 10_000,
) -> RefAttempt:
    store = _ref_request_store(request, create=False)
    binding = store.bindings.get(reservation.key) if store is not None else None
    reader = (
        binding.reader
        if binding is not None and binding.registry is reservation.registry
        else _new_ref_reader(
            request,
            reservation.key,
            threshold_tokens=threshold_tokens,
        )
    )
    return RefAttempt(
        key=reservation.key,
        generation=reservation.generation,
        plan=plan,
        registry=reservation.registry,
        reader=reader,
        previous_binding=binding,
        previous_reader_entry=reservation.registry.get(
            REF_EXEC_TOOL_NAME,
            REF_REGISTRY_ENTRY_MISSING,
        ),
    )


def register_ref_attempt(attempt: RefAttempt) -> RefStateDelta:
    existing = attempt.registry.get(REF_EXEC_TOOL_NAME)
    if existing is not None and (
        not isinstance(existing, dict) or existing.get("callable") is not attempt.reader
    ):
        raise RefProjectionError(stage="registration")
    return RefStateDelta(
        added_refs=tuple(entry.manifest.ref for entry in attempt.plan.catalog),
        generation=attempt.generation,
    )


async def compare_and_swap_ref_generation(
    request: Any,
    attempt: RefAttempt,
) -> bool:
    store = _ref_request_store(request, create=True)
    if store is None:
        return False
    async with store.lock:
        reservation = store.reservations.get(attempt.key)
        if (
            reservation is None
            or reservation.generation != attempt.generation
            or reservation.registry is not attempt.registry
        ):
            return False
        current = store.bindings.get(attempt.key)
        owner_key = store.registry_owners.get(id(attempt.registry))
        if owner_key is not None and owner_key != attempt.key:
            return False
        existing = attempt.registry.get(REF_EXEC_TOOL_NAME)
        if existing is not None and (
            current is None
            or current.registry is not attempt.registry
            or not isinstance(existing, dict)
            or existing.get("spec") != ref_exec_tool_spec_payload()["function"]
            or existing.get("callable") is not current.reader
        ):
            return False
        catalog_by_ref = {
            entry.manifest.ref: entry
            for entry in (current.catalog if current is not None else ())
        }
        for entry in attempt.plan.catalog:
            existing_entry = catalog_by_ref.get(entry.manifest.ref)
            if existing_entry is None:
                catalog_by_ref[entry.manifest.ref] = entry
                continue
            existing_is_v2_history = (
                isinstance(existing_entry.source, HistoryRefSourceHandle)
                and existing_entry.source.format == HISTORY_REF_LOGICAL_FORMAT
            )
            incoming_is_v2_history = (
                isinstance(entry.source, HistoryRefSourceHandle)
                and entry.source.format == HISTORY_REF_LOGICAL_FORMAT
            )
            if not (existing_is_v2_history and incoming_is_v2_history):
                continue
            if existing_entry.manifest != entry.manifest:
                raise RefProjectionError(
                    stage="v2 history re-advertisement integrity"
                )
            catalog_by_ref[entry.manifest.ref] = entry
        attempt.registry[REF_EXEC_TOOL_NAME] = {
            "spec": ref_exec_tool_spec_payload()["function"],
            "callable": attempt.reader,
        }
        store.bindings[attempt.key] = RefBindingState(
            generation=attempt.generation,
            catalog=tuple(catalog_by_ref.values()),
            registry=attempt.registry,
            reader=attempt.reader,
        )
        store.registry_owners[id(attempt.registry)] = attempt.key
        store.reservations.pop(attempt.key, None)
        pending = store.registry_reservations.get(id(attempt.registry))
        if pending is reservation:
            store.registry_reservations.pop(id(attempt.registry), None)
        return True


async def commit_ref_attempt(request: Any, attempt: RefAttempt) -> RefAttempt:
    if not await compare_and_swap_ref_generation(request, attempt):
        raise RefProjectionError(stage="generation CAS")
    return attempt


async def rollback_ref_attempt(request: Any, attempt: RefAttempt) -> None:
    store = _ref_request_store(request, create=False)
    if store is None:
        return
    async with store.lock:
        reservation = store.reservations.get(attempt.key)
        if reservation is not None:
            if (
                reservation.generation == attempt.generation
                and reservation.registry is attempt.registry
            ):
                _release_ref_reservation_locked(store, reservation)
                return
            if (
                reservation.generation > attempt.generation
                and reservation.registry is attempt.registry
            ):
                return
        binding = store.bindings.get(attempt.key)
        if (
            binding is None
            or binding.generation != attempt.generation
            or binding.registry is not attempt.registry
        ):
            return
        cleanup = store.deferred_cleanups.get(attempt.key)
        if cleanup is not None and _deferred_cleanup_matches_attempt(
            cleanup,
            attempt,
            binding,
        ):
            store.deferred_cleanups.pop(attempt.key, None)
            _teardown_ref_binding_locked(store, attempt.key, binding)
            return
        existing = attempt.registry.get(REF_EXEC_TOOL_NAME)
        owns_registry_entry = (
            isinstance(existing, dict) and existing.get("callable") is attempt.reader
        )
        if not owns_registry_entry:
            store.bindings.pop(attempt.key, None)
            store.registry_owners.pop(id(attempt.registry), None)
            if attempt.previous_binding is None:
                _invalidate_ref_reader(attempt.reader)
            return
        if attempt.previous_binding is None:
            store.bindings.pop(attempt.key, None)
            store.registry_owners.pop(id(attempt.registry), None)
            _invalidate_ref_reader(attempt.reader)
        else:
            store.bindings[attempt.key] = attempt.previous_binding
            store.registry_owners[id(attempt.registry)] = attempt.key
        if attempt.previous_reader_entry is REF_REGISTRY_ENTRY_MISSING:
            attempt.registry.pop(REF_EXEC_TOOL_NAME, None)
        else:
            attempt.registry[REF_EXEC_TOOL_NAME] = attempt.previous_reader_entry


async def cleanup_ref_attempt(request: Any, attempt: RefAttempt) -> None:
    store = _ref_request_store(request, create=False)
    if store is None:
        return
    async with store.lock:
        binding = store.bindings.get(attempt.key)
        if (
            binding is None
            or binding.generation != attempt.generation
            or binding.registry is not attempt.registry
            or binding.reader is not attempt.reader
        ):
            return
        reservation = store.reservations.get(attempt.key)
        if (
            reservation is not None
            and reservation.generation > attempt.generation
            and reservation.registry is attempt.registry
        ):
            store.deferred_cleanups[attempt.key] = RefDeferredCleanup(
                generation=binding.generation,
                successor_generation=reservation.generation,
                registry=binding.registry,
                reader=binding.reader,
            )
            return
        cleanup = store.deferred_cleanups.get(attempt.key)
        if cleanup is not None and _deferred_cleanup_targets_binding(cleanup, binding):
            store.deferred_cleanups.pop(attempt.key, None)
        _teardown_ref_binding_locked(store, attempt.key, binding)


def ref_exec_tool_spec_payload() -> dict[str, Any]:
    payload = _mutable_ref_schema_value(REF_EXEC_TOOL_SPEC)
    if not isinstance(payload, dict):
        raise RefProjectionError(stage="reader schema rendering")
    return payload


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
    cached = _MESSAGE_TOKEN_ESTIMATE_CACHE.get(key)
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


def estimate_message_tokens(
    message: dict[str, Any],
    *,
    request: Any = None,
    encoder: Any = None,
    encoding_name: str | None = None,
) -> int | None:
    if not isinstance(message, dict):
        return None
    resolved_encoder = encoder
    resolved_encoding_name = encoding_name
    if resolved_encoder is None:
        resolved_encoder, resolved_encoding_name = _get_tiktoken_encoder(request)
    if resolved_encoder is None:
        return None
    if not resolved_encoding_name:
        resolved_encoding_name = str(getattr(resolved_encoder, "name", "unknown"))
    return _estimate_message_tokens_with_encoder(
        message,
        encoder=resolved_encoder,
        encoding_name=str(resolved_encoding_name),
    )


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


def estimate_body_extra_tokens(
    body: dict[str, Any],
    *,
    request: Any = None,
    encoder: Any = None,
    encoding_name: str | None = None,
) -> int | None:
    if not isinstance(body, dict):
        return None
    extra_text = _body_token_extra_text(body)
    if not extra_text:
        return 0
    resolved_encoder = encoder
    resolved_encoding_name = encoding_name
    if resolved_encoder is None:
        resolved_encoder, resolved_encoding_name = _get_tiktoken_encoder(request)
    if resolved_encoder is None:
        return None
    if not resolved_encoding_name:
        resolved_encoding_name = str(getattr(resolved_encoder, "name", "unknown"))
    return _estimate_text_tokens_with_encoder(
        extra_text,
        encoder=resolved_encoder,
        overhead=MESSAGE_TOKEN_OVERHEAD,
    )


async def estimate_body_extra_tokens_async(
    body: dict[str, Any],
    *,
    request: Any = None,
    encoder: Any = None,
    encoding_name: str | None = None,
) -> int | None:
    if encoding_name is None and encoder is None:
        await _refresh_tiktoken_encoding_config(request)
    return await asyncio.to_thread(
        estimate_body_extra_tokens,
        body,
        request=request,
        encoder=encoder,
        encoding_name=encoding_name,
    )


async def estimate_body_tokens_async(
    body: dict[str, Any],
    *,
    request: Any = None,
    encoder: Any = None,
    encoding_name: str | None = None,
) -> int | None:
    if encoding_name is None and encoder is None:
        await _refresh_tiktoken_encoding_config(request)
    return await asyncio.to_thread(
        estimate_body_tokens,
        body,
        request=request,
        encoder=encoder,
        encoding_name=encoding_name,
    )


async def estimate_message_tokens_async(
    message: dict[str, Any],
    *,
    request: Any = None,
    encoder: Any = None,
    encoding_name: str | None = None,
) -> int | None:
    if encoding_name is None and encoder is None:
        await _refresh_tiktoken_encoding_config(request)
    return await asyncio.to_thread(
        estimate_message_tokens,
        message,
        request=request,
        encoder=encoder,
        encoding_name=encoding_name,
    )


async def estimate_messages_tokens_async(
    messages: list[dict[str, Any]],
    *,
    request: Any = None,
    encoder: Any = None,
    encoding_name: str | None = None,
) -> int | None:
    if encoding_name is None and encoder is None:
        await _refresh_tiktoken_encoding_config(request)
    return await asyncio.to_thread(
        estimate_messages_tokens,
        messages,
        request=request,
        encoder=encoder,
        encoding_name=encoding_name,
    )


def _partition_usage_anchor_messages(
    messages: Any,
    transient_message_patterns: TransientMessagePatterns | _TransientMessageMatcher | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not isinstance(messages, list):
        return None
    mask = _transient_message_mask(messages, transient_message_patterns)
    stable: list[dict[str, Any]] = []
    volatile: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if _is_source_identity_message(
            message,
            transient_message_patterns=transient_message_patterns,
            transient_message_mask=mask,
            index=index,
        ):
            stable.append(message)
        else:
            volatile.append(message)
    return stable, volatile


def _project_usage_anchor_token_body(
    body: dict[str, Any],
    *,
    dropped_message_keys: frozenset[str],
) -> dict[str, Any]:
    if not dropped_message_keys:
        return body
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(
        isinstance(message, dict) and any(key in message for key in dropped_message_keys) for message in messages
    ):
        return body
    projected = dict(body)
    projected["messages"] = [
        {key: value for key, value in message.items() if key not in dropped_message_keys}
        if isinstance(message, dict) and dropped_message_keys & message.keys()
        else message
        for message in messages
    ]
    return projected


async def _project_system_prompt_for_token_estimate(
    body: dict[str, Any],
    *,
    user: Any,
    system_prompt: str | None,
) -> dict[str, Any]:
    if not system_prompt:
        return body
    messages = body.get("messages")
    projected = dict(body)
    projected_messages = list(messages) if isinstance(messages, list) else []
    if projected_messages and isinstance(projected_messages[0], dict) and _is_system_message(projected_messages[0]):
        projected_messages[0] = copy.deepcopy(projected_messages[0])
    projected["messages"] = projected_messages
    from open_webui.utils.payload import apply_system_prompt_to_body

    metadata = projected.get("metadata")
    return await apply_system_prompt_to_body(
        system_prompt,
        projected,
        metadata if isinstance(metadata, dict) else None,
        coerce_open_webui_user(user),
    )


def provider_visible_ref_estimate_body(body: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if key != "metadata"}


async def _estimate_provider_input_tokens_async(
    body: dict[str, Any],
    *,
    request: Any,
    user: Any,
    system_prompt: str | None,
    dropped_message_keys: frozenset[str] = frozenset(),
) -> int | None:
    token_body = _project_usage_anchor_token_body(
        body,
        dropped_message_keys=dropped_message_keys,
    )
    projected = await _project_system_prompt_for_token_estimate(
        token_body,
        user=user,
        system_prompt=system_prompt,
    )
    return await estimate_body_tokens_async(
        provider_visible_ref_estimate_body(projected),
        request=request,
    )


def _compute_usage_anchor_input_fingerprint(
    body: dict[str, Any],
    stable_messages: list[dict[str, Any]],
    *,
    usage_anchor_shaping_hash: str | None = None,
    encoding_name: str | None = None,
) -> str:
    if encoding_name is None:
        _, encoding_name = _get_tiktoken_encoder()
    return _json_hash(
        {
            "family": USAGE_ANCHOR_FINGERPRINT_FAMILY,
            "provider_shaping": usage_anchor_shaping_hash,
            "token_estimator": {
                "version": TOKEN_ESTIMATOR_VERSION,
                "encoding": str(encoding_name or ""),
            },
            "model": str(body.get("model") or ""),
            "messages": [canonicalize_message_for_token_estimate(message) for message in stable_messages],
            "body_extra": _body_token_extra_payload(body),
        }
    )


async def _estimate_message_token_sum_async(
    messages: list[dict[str, Any]],
    *,
    request: Any,
) -> int | None:
    estimated = await estimate_messages_tokens_async(messages, request=request)
    if estimated is None or estimated < REQUEST_TOKEN_OVERHEAD:
        return None
    return int(estimated) - REQUEST_TOKEN_OVERHEAD


async def _build_usage_anchor_input(
    *,
    request: Any,
    body: dict[str, Any],
    usage_anchor_shaping_hash: str | None = None,
    transient_message_patterns: TransientMessagePatterns | _TransientMessageMatcher | None = None,
) -> UsageAnchorInput | None:
    if not isinstance(body, dict) or body.get("previous_response_id"):
        return None
    await _refresh_tiktoken_encoding_config(request)
    _, encoding_name = await asyncio.to_thread(_get_tiktoken_encoder, request)
    partitioned = _partition_usage_anchor_messages(body.get("messages"), transient_message_patterns)
    if partitioned is None:
        return None
    stable, volatile = partitioned
    volatile_tokens = await _estimate_message_token_sum_async(volatile, request=request)
    if volatile_tokens is None:
        return None
    fingerprint = await asyncio.to_thread(
        _compute_usage_anchor_input_fingerprint,
        body,
        stable,
        usage_anchor_shaping_hash=usage_anchor_shaping_hash,
        encoding_name=encoding_name,
    )
    return UsageAnchorInput(
        stable_message_count=len(stable),
        input_fingerprint=fingerprint,
        volatile_message_tokens=volatile_tokens,
    )


async def _estimate_body_tokens_from_usage_anchor(
    *,
    request: Any,
    body: dict[str, Any],
    anchor: UsageAnchor,
    usage_anchor_shaping_hash: str | None = None,
    transient_message_patterns: TransientMessagePatterns | _TransientMessageMatcher | None = None,
) -> int | None:
    if body.get("previous_response_id"):
        LOG.debug("Auto-compaction usage anchor miss: previous_response_id")
        return None
    await _refresh_tiktoken_encoding_config(request)
    _, encoding_name = await asyncio.to_thread(_get_tiktoken_encoder, request)
    partitioned = _partition_usage_anchor_messages(body.get("messages"), transient_message_patterns)
    if partitioned is None:
        LOG.debug("Auto-compaction usage anchor miss: invalid_messages")
        return None
    stable, volatile = partitioned
    if len(stable) < anchor.stable_message_count:
        LOG.debug("Auto-compaction usage anchor miss: shorter_prefix")
        return None
    prefix = stable[: anchor.stable_message_count]
    fingerprint = await asyncio.to_thread(
        _compute_usage_anchor_input_fingerprint,
        body,
        prefix,
        usage_anchor_shaping_hash=usage_anchor_shaping_hash,
        encoding_name=encoding_name,
    )
    if fingerprint != anchor.input_fingerprint:
        LOG.debug("Auto-compaction usage anchor miss: prefix_mismatch")
        return None
    base_tokens = anchor.input_tokens - anchor.volatile_message_tokens
    if base_tokens < 0:
        LOG.debug("Auto-compaction usage anchor miss: invalid_measurement")
        return None
    current_volatile_tokens = await _estimate_message_token_sum_async(volatile, request=request)
    suffix_tokens = await _estimate_message_token_sum_async(
        stable[anchor.stable_message_count :],
        request=request,
    )
    if current_volatile_tokens is None or suffix_tokens is None:
        LOG.debug("Auto-compaction usage anchor miss: estimate_unavailable")
        return None
    LOG.debug("Auto-compaction usage anchor hit")
    return base_tokens + current_volatile_tokens + suffix_tokens


async def _estimate_rendered_summary_message_tokens(
    *,
    request: Any,
    summary_text: str,
    summary_meta: dict[str, Any] | None,
    historical_source_messages: list[dict[str, Any]] | None = None,
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> int | None:
    return await estimate_message_tokens_async(
        render_summary_message(
            summary_text,
            summary_meta,
            historical_source_messages=historical_source_messages,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            transient_message_patterns=transient_message_patterns,
        ),
        request=request,
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
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> MessageCut:
    if not messages:
        return MessageCut(None, [], [], 0)

    source_messages = [copy.deepcopy(message) for message in messages]
    system_index = _first_system_index(source_messages)
    preserved_system = copy.deepcopy(source_messages[system_index]) if system_index is not None else None
    working = [message for index, message in enumerate(source_messages) if index != system_index]

    if not working:
        return MessageCut(preserved_system, [], [], 0)

    working_transient_mask = _transient_message_mask(working, transient_message_patterns)
    latest_user_index = next(
        (
            index
            for index in range(len(working) - 1, -1, -1)
            if working[index].get("role") == "user"
            and _is_source_identity_message(
                working[index],
                transient_message_patterns=transient_message_patterns,
                transient_message_mask=working_transient_mask,
                index=index,
            )
        ),
        len(working) - 1,
    )
    tail = copy.deepcopy(working[latest_user_index:])

    prefix = copy.deepcopy(working[:latest_user_index])
    return MessageCut(
        preserved_system_message=preserved_system,
        summarization_prefix=prefix,
        tail_messages=tail,
        source_message_count=_source_identity_message_count(
            prefix,
            transient_message_patterns=transient_message_patterns,
        ),
    )


def select_retry_tool_result_cut(
    messages: list[dict[str, Any]],
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> RetryToolResultCut | None:
    cut = select_tool_result_compaction_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
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
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> ToolResultCompactionCut | None:
    source_messages = [copy.deepcopy(message) for message in messages]
    system_index = _first_system_index(source_messages)
    preserved_system = copy.deepcopy(source_messages[system_index]) if system_index is not None else None
    working = [message for index, message in enumerate(source_messages) if index != system_index]

    working_transient_mask = _transient_message_mask(working, transient_message_patterns)
    latest_user_index = next(
        (
            index
            for index in range(len(working) - 1, -1, -1)
            if working[index].get("role") == "user"
            and _is_source_identity_message(
                working[index],
                transient_message_patterns=transient_message_patterns,
                transient_message_mask=working_transient_mask,
                index=index,
            )
        ),
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
        source_message_count=_source_identity_message_count(
            summarization_prefix,
            transient_message_patterns=transient_message_patterns,
        ),
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
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> str:
    count_limit = int(max_messages or 0)
    if count_limit <= 0:
        return ""

    excerpts = [item["text"] for item in _build_historical_user_message_excerpt_items(
        source_messages,
        excerpt_bytes=excerpt_bytes,
        max_messages=count_limit,
        transient_message_patterns=transient_message_patterns,
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
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> list[dict[str, Any]]:
    count_limit = _coerce_nonnegative_int(max_messages)
    if count_limit <= 0:
        return []

    excerpts: list[str] = []
    for message in source_messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if _is_transient_message(message, transient_message_patterns):
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
    transient_message_patterns: TransientMessagePatterns | None = None,
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
        transient_message_patterns=transient_message_patterns,
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
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> dict[str, Any]:
    return enrich_summary_meta_with_historical_excerpts(
        {"has_multimodal": _messages_have_multimodal(source_messages)},
        source_messages,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        transient_message_patterns=transient_message_patterns,
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
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> _RenderedSummaryMessage:
    meta_section = ""
    if summary_meta and summary_meta.get("has_multimodal"):
        meta_section = "\n<metadata><has_multimodal>true</has_multimodal></metadata>"
    excerpts = render_stored_historical_user_message_excerpts(summary_meta)
    if not excerpts and historical_source_messages:
        excerpts = render_historical_user_message_excerpts(
            historical_source_messages,
            excerpt_bytes=historical_message_excerpt_bytes,
            max_messages=historical_message_excerpt_count,
            transient_message_patterns=transient_message_patterns,
        )
    excerpt_section = f"\n{excerpts}" if excerpts else ""
    return _RenderedSummaryMessage(
        {
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
    )


def render_summary_message_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    historical_source_messages: list[dict[str, Any]] | None = None,
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> dict[str, Any]:
    summary_meta = normalize_summary_meta(checkpoint.get("summary_meta") if isinstance(checkpoint, dict) else {})
    fallback_source_messages = None
    if (
        SUMMARY_META_FORMAT_VERSION_KEY not in summary_meta
        and SUMMARY_META_HISTORICAL_USER_MESSAGES_KEY not in summary_meta
    ):
        fallback_source_messages = historical_source_messages
    message = render_summary_message(
        str(checkpoint.get("summary_text") or ""),
        summary_meta,
        historical_source_messages=fallback_source_messages,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        transient_message_patterns=transient_message_patterns,
    )
    checkpoint_id = checkpoint.get("id")
    if isinstance(checkpoint_id, str):
        message.history_ref = f"history:{checkpoint_id}"
    return message


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
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> dict[str, Any]:
    checkpoint = _checkpoint_from_summary_result(summary_text)
    if checkpoint is not None:
        return render_summary_message_from_checkpoint(
            checkpoint,
            historical_source_messages=historical_source_messages,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            transient_message_patterns=transient_message_patterns,
        )
    return render_summary_message(
        str(summary_text),
        summary_meta,
        historical_source_messages=historical_source_messages,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        transient_message_patterns=transient_message_patterns,
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
    transient_message_patterns: TransientMessagePatterns | None = None,
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
            transient_message_patterns=transient_message_patterns,
        )
    )
    compacted.extend(copy.deepcopy(cut.tail_messages))
    return compacted


def replace_prefix_with_parent_checkpoint_and_delta(
    cut: MessageCut,
    parent: dict[str, Any],
    *,
    prefix_file_fingerprint: str | None = None,
    file_backed_image_db_chain: list[dict[str, Any]] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
) -> list[dict[str, Any]]:
    parent_count = int(parent.get("source_message_count") or 0)
    raw_parent_count = _raw_prefix_len_for_source_count(
        cut.summarization_prefix,
        parent_count,
        transient_message_patterns=transient_message_patterns,
    )
    if parent_count <= 0 or raw_parent_count is None:
        raise UnsupportedCompactionInput(
            "Parent checkpoint cannot be applied safely because its source boundary is invalid",
            code="unsafe_checkpoint_parent",
        )
    if (
        compute_summary_source_hash(
            cut.summarization_prefix[:raw_parent_count],
            prefix_file_fingerprint,
            file_backed_image_db_chain,
            transient_message_patterns=transient_message_patterns,
        )
        != parent.get("source_hash")
    ):
        raise UnsupportedCompactionInput(
            "Parent checkpoint cannot be applied safely because its source hash no longer matches",
            code="unsafe_checkpoint_parent",
        )

    delta_messages = copy.deepcopy(cut.summarization_prefix[raw_parent_count:])
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
            historical_source_messages=cut.summarization_prefix[:raw_parent_count],
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            transient_message_patterns=transient_message_patterns,
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
    summary_token_count: int | None = None,
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
        "summary_token_count": summary_token_count,
        "state": state,
        "parent_checkpoint_id": parent_checkpoint_id,
        "claim_token": claim_token,
        "claim_expires_at": claim_expires_at,
        "created_at": timestamp,
        "updated_at": timestamp,
        "last_used_at": timestamp,
    }


def _usage_anchor_profile_hash() -> str:
    return _json_hash({"family": USAGE_ANCHOR_PROFILE_FAMILY, "format_version": USAGE_ANCHOR_FORMAT_VERSION})


def _usage_anchor_source_hash(assistant_message_id: str) -> str:
    return _json_hash({"family": USAGE_ANCHOR_SOURCE_FAMILY, "assistant_message_id": assistant_message_id})


def build_usage_anchor_row(
    *,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    assistant_message_id: str,
    input_tokens: int,
    anchor_input: UsageAnchorInput,
) -> dict[str, Any]:
    return build_checkpoint_row(
        namespace=USAGE_ANCHOR_NAMESPACE,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        profile_hash=_usage_anchor_profile_hash(),
        source_hash=_usage_anchor_source_hash(assistant_message_id),
        source_message_count=anchor_input.stable_message_count,
        summary_text="",
        summary_meta={
            "format_version": USAGE_ANCHOR_FORMAT_VERSION,
            "assistant_message_id": assistant_message_id,
            "input_tokens": input_tokens,
            "input_fingerprint": anchor_input.input_fingerprint,
            "volatile_message_tokens": anchor_input.volatile_message_tokens,
        },
        summary_token_count=None,
        parent_checkpoint_id=None,
    )


def usage_anchor_from_row(row: dict[str, Any] | None) -> UsageAnchor | None:
    if not isinstance(row, dict) or row.get("namespace") != USAGE_ANCHOR_NAMESPACE:
        return None
    meta = row.get("summary_meta")
    if not isinstance(meta, dict) or meta.get("format_version") != USAGE_ANCHOR_FORMAT_VERSION:
        return None
    assistant_message_id = meta.get("assistant_message_id")
    input_fingerprint = meta.get("input_fingerprint")
    input_tokens = meta.get("input_tokens")
    stable_message_count = row.get("source_message_count")
    volatile_message_tokens = meta.get("volatile_message_tokens")
    numeric_values = (input_tokens, stable_message_count, volatile_message_tokens)
    if (
        not isinstance(assistant_message_id, str)
        or not assistant_message_id
        or not isinstance(input_fingerprint, str)
        or not input_fingerprint
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in numeric_values)
    ):
        return None
    if input_tokens <= 0 or input_tokens < volatile_message_tokens:
        return None
    return UsageAnchor(
        assistant_message_id=assistant_message_id,
        input_tokens=input_tokens,
        stable_message_count=stable_message_count,
        input_fingerprint=input_fingerprint,
        volatile_message_tokens=volatile_message_tokens,
    )


def select_longest_matching_checkpoint(
    rows: Iterable[dict[str, Any]],
    source_messages: list[dict[str, Any]],
    *,
    states: set[str] | None = None,
    prefix_file_fingerprint_resolver: Callable[[int], str | None] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> dict[str, Any] | None:
    prefix_hashes: dict[int, str] = {}
    file_backed_image_db_chain = _prefix_file_fingerprint_resolver_db_chain(prefix_file_fingerprint_resolver)
    candidates = sorted(rows, key=lambda row: int(row.get("source_message_count") or 0), reverse=True)
    source_count = _source_identity_message_count(
        source_messages,
        transient_message_patterns=transient_message_patterns,
    )
    for row in candidates:
        if states is not None and row.get("state") not in states:
            continue
        count = int(row.get("source_message_count") or 0)
        if count <= 0 or count > source_count:
            continue
        raw_count = _raw_prefix_len_for_source_count(
            source_messages,
            count,
            transient_message_patterns=transient_message_patterns,
        )
        if raw_count is None:
            continue
        if count not in prefix_hashes:
            fingerprint = (
                prefix_file_fingerprint_resolver(count)
                if prefix_file_fingerprint_resolver is not None
                else None
            )
            prefix_hashes[count] = compute_summary_source_hash(
                source_messages[:raw_count],
                fingerprint,
                file_backed_image_db_chain,
                transient_message_patterns=transient_message_patterns,
            )
        if prefix_hashes[count] == row.get("source_hash"):
            return row
    return None


def select_longest_matching_parent(
    rows: Iterable[dict[str, Any]],
    source_messages: list[dict[str, Any]],
    *,
    prefix_file_fingerprint_resolver: Callable[[int], str | None] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> dict[str, Any] | None:
    return select_longest_matching_checkpoint(
        rows,
        source_messages,
        states={"ready"},
        prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
        transient_message_patterns=transient_message_patterns,
    )


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


def _initialize_checkpoint_schema(sync_conn: Any) -> None:
    _execute_checkpoint_ddl_tolerating_duplicates(sync_conn, CreateTable(CHECKPOINT_TABLE, if_not_exists=True))
    for index in CHECKPOINT_TABLE.indexes:
        _execute_checkpoint_ddl_tolerating_duplicates(sync_conn, CreateIndex(index, if_not_exists=True))


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

    async def lookup_ready_by_id(
        self,
        checkpoint_id: str,
        *,
        namespace: str,
        user_id: str,
        chat_id: str,
        pipe_function_id: str,
        profile_hash: str,
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
                    CHECKPOINT_TABLE.c.id == checkpoint_id,
                    CHECKPOINT_TABLE.c.state == "ready",
                )
            )
            row = result.mappings().first()
            return dict(row) if row else None

    async def lookup_ready_descriptor_by_id(
        self,
        checkpoint_id: str,
        *,
        namespace: str,
        user_id: str,
        chat_id: str,
        pipe_function_id: str,
        profile_hash: str,
    ) -> dict[str, Any] | None:
        async with await self._context() as db:
            result = await db.execute(
                select(
                    CHECKPOINT_TABLE.c.id,
                    CHECKPOINT_TABLE.c.namespace,
                    CHECKPOINT_TABLE.c.user_id,
                    CHECKPOINT_TABLE.c.chat_id,
                    CHECKPOINT_TABLE.c.pipe_function_id,
                    CHECKPOINT_TABLE.c.profile_hash,
                    CHECKPOINT_TABLE.c.source_message_count,
                    CHECKPOINT_TABLE.c.source_hash,
                    CHECKPOINT_TABLE.c.summary_meta,
                    CHECKPOINT_TABLE.c.state,
                    CHECKPOINT_TABLE.c.parent_checkpoint_id,
                ).where(
                    *self._identity_clauses(
                        namespace=namespace,
                        user_id=user_id,
                        chat_id=chat_id,
                        pipe_function_id=pipe_function_id,
                        profile_hash=profile_hash,
                    ),
                    CHECKPOINT_TABLE.c.id == checkpoint_id,
                    CHECKPOINT_TABLE.c.state == "ready",
                )
            )
            row = result.mappings().first()
            return dict(row) if row else None

    async def compare_and_swap_history_ref(
        self,
        checkpoint_id: str,
        *,
        expected_summary_meta: dict[str, Any],
        history_ref: dict[str, str],
    ) -> bool:
        updated_meta = copy.deepcopy(expected_summary_meta)
        updated_meta[SUMMARY_META_HISTORY_REF_KEY] = dict(history_ref)
        async with await self._context() as db:
            try:
                dialect_name = db.get_bind().dialect.name
                if dialect_name == "postgresql":
                    summary_meta_is_object = func.json_typeof(
                        CHECKPOINT_TABLE.c.summary_meta
                    ) == literal("object")
                    history_ref_absent = func.json_typeof(
                        CHECKPOINT_TABLE.c.summary_meta.op("->")(
                            literal(SUMMARY_META_HISTORY_REF_KEY)
                        )
                    ).is_(None)
                else:
                    summary_meta_is_object = func.json_type(
                        CHECKPOINT_TABLE.c.summary_meta
                    ) == literal("object")
                    history_ref_absent = func.json_type(
                        CHECKPOINT_TABLE.c.summary_meta,
                        f'$."{SUMMARY_META_HISTORY_REF_KEY}"',
                    ).is_(None)
                result = await db.execute(
                    update(CHECKPOINT_TABLE)
                    .where(
                        CHECKPOINT_TABLE.c.id == checkpoint_id,
                        CHECKPOINT_TABLE.c.state == "ready",
                        summary_meta_is_object,
                        history_ref_absent,
                    )
                    .values(summary_meta=updated_meta, updated_at=int(time.time()))
                )
                if (result.rowcount or 0) != 1:
                    await db.rollback()
                    return False
                await db.commit()
                return True
            except Exception:
                await db.rollback()
                raise

    async def find_longest_parent(
        self,
        *,
        namespace: str,
        user_id: str,
        chat_id: str,
        pipe_function_id: str,
        profile_hash: str,
        source_messages: list[dict[str, Any]],
        prefix_file_fingerprint_resolver: Callable[[int], str | None] | None = None,
        transient_message_patterns: TransientMessagePatterns | None = None,
    ) -> dict[str, Any] | None:
        if not source_messages:
            return None
        source_count = _source_identity_message_count(
            source_messages,
            transient_message_patterns=transient_message_patterns,
        )
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
                    CHECKPOINT_TABLE.c.source_message_count <= source_count,
                )
                .order_by(CHECKPOINT_TABLE.c.source_message_count.desc())
            )
            rows = [dict(row) for row in result.mappings().all()]
        return select_longest_matching_parent(
            rows,
            source_messages,
            prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
            transient_message_patterns=transient_message_patterns,
        )

    async def find_longest_pending_parent(
        self,
        *,
        namespace: str,
        user_id: str,
        chat_id: str,
        pipe_function_id: str,
        profile_hash: str,
        source_messages: list[dict[str, Any]],
        prefix_file_fingerprint_resolver: Callable[[int], str | None] | None = None,
        transient_message_patterns: TransientMessagePatterns | None = None,
    ) -> dict[str, Any] | None:
        if not source_messages:
            return None
        source_count = _source_identity_message_count(
            source_messages,
            transient_message_patterns=transient_message_patterns,
        )
        now = int(time.time())
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
                    CHECKPOINT_TABLE.c.state == "pending",
                    CHECKPOINT_TABLE.c.claim_expires_at.is_not(None),
                    CHECKPOINT_TABLE.c.claim_expires_at > now,
                    CHECKPOINT_TABLE.c.source_message_count <= source_count,
                )
                .order_by(CHECKPOINT_TABLE.c.source_message_count.desc())
            )
            rows = [dict(row) for row in result.mappings().all()]
        return select_longest_matching_checkpoint(
            rows,
            source_messages,
            states={"pending"},
            prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
            transient_message_patterns=transient_message_patterns,
        )

    async def claim_pending(self, row: dict[str, Any]) -> bool:
        async with await self._context() as db:
            try:
                await db.execute(insert(CHECKPOINT_TABLE).values(**row))
                await db.commit()
                return True
            except IntegrityError:
                await db.rollback()
                return False

    async def upsert_ready(self, row: dict[str, Any]) -> None:
        update_values = {key: value for key, value in row.items() if key not in {"id", "created_at"}}
        async with await self._context() as db:
            try:
                result = await db.execute(
                    update(CHECKPOINT_TABLE)
                    .where(CHECKPOINT_TABLE.c.id == row["id"])
                    .values(**update_values)
                )
                if (result.rowcount or 0) == 1:
                    await db.commit()
                    return
                await db.rollback()
                try:
                    await db.execute(insert(CHECKPOINT_TABLE).values(**row))
                    await db.commit()
                    return
                except IntegrityError:
                    await db.rollback()
                result = await db.execute(
                    update(CHECKPOINT_TABLE)
                    .where(CHECKPOINT_TABLE.c.id == row["id"])
                    .values(**update_values)
                )
                if (result.rowcount or 0) != 1:
                    await db.rollback()
                    raise RuntimeError("Usage anchor upsert lost its conflicting row")
                await db.commit()
            except Exception:
                await db.rollback()
                raise

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
        summary_token_count: int | None = None,
        generation_lease_id: str | None = None,
        generation_lease_claim_token: str | None = None,
        now: int | None = None,
    ) -> dict[str, Any] | None:
        timestamp = int(time.time()) if now is None else int(now)
        async with await self._context() as db:
            try:
                update_conditions = [
                    CHECKPOINT_TABLE.c.id == checkpoint_id,
                    CHECKPOINT_TABLE.c.state == "pending",
                    CHECKPOINT_TABLE.c.claim_token == claim_token,
                ]
                if generation_lease_id is not None and generation_lease_claim_token is not None:
                    lease_table = CHECKPOINT_TABLE.alias("generation_lease")
                    lease_conditions = [
                        lease_table.c.id == generation_lease_id,
                        lease_table.c.namespace == CHECKPOINT_GENERATION_LEASE_NAMESPACE,
                        lease_table.c.state == "pending",
                        lease_table.c.claim_token == generation_lease_claim_token,
                        lease_table.c.claim_expires_at.is_not(None),
                        lease_table.c.claim_expires_at > timestamp,
                    ]
                    lease_result = await db.execute(
                        select(lease_table.c.id).where(*lease_conditions).with_for_update()
                    )
                    if lease_result.scalar_one_or_none() is None:
                        await db.rollback()
                        return None
                    update_conditions.append(
                        exists(select(1).select_from(lease_table).where(*lease_conditions))
                    )
                result = await db.execute(
                    update(CHECKPOINT_TABLE)
                    .where(*update_conditions)
                    .values(
                        state="ready",
                        summary_text=summary_text,
                        parent_checkpoint_id=parent_checkpoint_id,
                        summary_token_count=summary_token_count,
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


async def lookup_usage_anchor(
    *,
    request: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    assistant_message_id: str,
) -> UsageAnchor | None:
    if not user_id or not assistant_message_id or not _chat_id_supported(chat_id):
        return None
    await ensure_checkpoint_table_initialized(request=request)
    row = await CheckpointStore().lookup_ready(
        namespace=USAGE_ANCHOR_NAMESPACE,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        profile_hash=_usage_anchor_profile_hash(),
        source_hash=_usage_anchor_source_hash(assistant_message_id),
    )
    anchor = usage_anchor_from_row(row)
    if anchor is None or anchor.assistant_message_id != assistant_message_id:
        LOG.debug("Auto-compaction usage anchor miss: not_found_or_invalid")
        return None
    return anchor


async def persist_usage_anchor(
    *,
    request: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    assistant_message_id: str,
    anchor_input: UsageAnchorInput | None,
    raw_usage: dict[str, Any] | None,
) -> bool:
    if anchor_input is None or not user_id or not assistant_message_id or not _chat_id_supported(chat_id):
        return False
    input_tokens = _strict_usage_input_tokens(raw_usage)
    if input_tokens is None:
        LOG.debug("Auto-compaction usage anchor not persisted: unsupported_usage_shape")
        return False
    if input_tokens < anchor_input.volatile_message_tokens:
        LOG.debug("Auto-compaction usage anchor not persisted: inconsistent_input_measurement")
        return False
    await ensure_checkpoint_table_initialized(request=request)
    await CheckpointStore().upsert_ready(
        build_usage_anchor_row(
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            assistant_message_id=assistant_message_id,
            input_tokens=input_tokens,
            anchor_input=anchor_input,
        )
    )
    return True


async def _usage_anchor_parent_assistant_message_id(
    *,
    metadata: dict[str, Any],
    chat_id: str,
) -> str | None:
    continued_assistant_id = metadata.get("assistant_message_id")
    if isinstance(continued_assistant_id, str) and continued_assistant_id:
        return continued_assistant_id
    user_message = metadata.get("user_message")
    if isinstance(user_message, dict):
        parent_id = user_message.get("parentId")
        if isinstance(parent_id, str) and parent_id:
            return parent_id
    user_message_id = metadata.get("user_message_id")
    if not _chat_id_supported(chat_id) or not isinstance(user_message_id, str) or not user_message_id:
        return None
    try:
        from open_webui.models.chats import Chats

        stored = await Chats.get_message_by_id_and_message_id(chat_id, user_message_id)
    except Exception:
        return None
    parent_id = stored.get("parentId") if isinstance(stored, dict) else None
    return parent_id if isinstance(parent_id, str) and parent_id else None


def _split_patterns(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").replace("\n", ",").split(",") if part.strip()]


def _wildcard_to_regex(pattern: str) -> str:
    # Only "*" (any run) and "?" (single char) are wildcards; everything else is literal.
    # Model ids contain "[" / "]" verbatim, so unlike fnmatch we never treat them as classes.
    out = []
    for ch in pattern:
        if ch == "*":
            out.append(".*")
        elif ch == "?":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _matches_any_pattern(model: dict[str, Any], patterns: list[str]) -> bool:
    model_id = str(model.get("id") or "")
    name = str(model.get("name") or "")
    for pattern in patterns:
        regex = _wildcard_to_regex(pattern)
        if re.fullmatch(regex, model_id) is not None or re.fullmatch(regex, name) is not None:
            return True
    return False


def parse_per_model_overrides(value: str) -> list[dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        root = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"per_model_overrides_json must be valid JSON: {exc}") from exc
    if not isinstance(root, dict):
        raise ValueError("per_model_overrides_json must be a JSON object")
    unknown_root_keys = set(root) - {"schema_version", "overrides"}
    if unknown_root_keys:
        raise ValueError(f"per_model_overrides_json has unknown keys: {sorted(unknown_root_keys)}")
    schema_version = root.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError("per_model_overrides_json schema_version must be 1")
    raw_overrides = root.get("overrides", [])
    if not isinstance(raw_overrides, list):
        raise ValueError("per_model_overrides_json overrides must be a list")
    overrides: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_overrides):
        if not isinstance(raw, dict):
            raise ValueError(f"overrides[{index}] must be an object")
        unknown_override_keys = set(raw) - {"model_patterns", "trigger_input_tokens", "soft_trigger_ratio"}
        if unknown_override_keys:
            raise ValueError(f"overrides[{index}] has unknown keys: {sorted(unknown_override_keys)}")
        patterns = raw.get("model_patterns")
        if not isinstance(patterns, list) or not patterns or not all(isinstance(p, str) and p for p in patterns):
            raise ValueError(f"overrides[{index}].model_patterns must be a non-empty list of strings")
        override: dict[str, Any] = {"model_patterns": list(patterns)}
        if "trigger_input_tokens" in raw:
            threshold = raw.get("trigger_input_tokens")
            if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
                raise ValueError(f"overrides[{index}].trigger_input_tokens must be an integer >= 1")
            override["trigger_input_tokens"] = threshold
        if "soft_trigger_ratio" in raw:
            ratio = raw.get("soft_trigger_ratio")
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio)
                or ratio < 0
                or ratio >= 1
            ):
                raise ValueError(f"overrides[{index}].soft_trigger_ratio must be a number >= 0 and < 1")
            override["soft_trigger_ratio"] = float(ratio)
        if "trigger_input_tokens" not in override and "soft_trigger_ratio" not in override:
            raise ValueError(f"overrides[{index}] must set trigger_input_tokens or soft_trigger_ratio")
        overrides.append(override)
    return overrides


def resolve_trigger_input_tokens(valves: Any, target_model: dict[str, Any]) -> int:
    overrides = parse_per_model_overrides(getattr(valves, "per_model_overrides_json", ""))
    for override in overrides:
        if "trigger_input_tokens" in override and _matches_any_pattern(target_model, override["model_patterns"]):
            return int(override["trigger_input_tokens"])
    return int(valves.trigger_input_tokens)


def resolve_soft_trigger_ratio(valves: Any, target_model: dict[str, Any]) -> float:
    overrides = parse_per_model_overrides(getattr(valves, "per_model_overrides_json", ""))
    for override in overrides:
        if "soft_trigger_ratio" in override and _matches_any_pattern(target_model, override["model_patterns"]):
            return float(override["soft_trigger_ratio"])
    return float(getattr(valves, "soft_trigger_ratio", DEFAULT_SOFT_TRIGGER_RATIO) or 0)


def resolve_soft_trigger_input_tokens(
    valves: Any,
    target_model: dict[str, Any],
    hard_trigger_input_tokens: int,
) -> int | None:
    soft_trigger_ratio = resolve_soft_trigger_ratio(valves, target_model)
    hard_trigger_input_tokens = int(hard_trigger_input_tokens)
    soft_trigger_input_tokens = int(hard_trigger_input_tokens * soft_trigger_ratio)
    if soft_trigger_input_tokens <= 0 or soft_trigger_input_tokens >= hard_trigger_input_tokens:
        return None
    return soft_trigger_input_tokens


def _is_arena_model(model: dict[str, Any]) -> bool:
    return bool(model.get("arena")) or model.get("owned_by") == "arena"


def _arena_chat_candidate_model_ids(
    models: dict[str, Any],
    arena_model: dict[str, Any],
    *,
    pipe_function_id: str = PIPE_FUNCTION_ID,
) -> list[str]:
    def is_allowed(model_id: str) -> bool:
        candidate = models.get(model_id)
        return not (
            isinstance(candidate, dict)
            and _is_own_wrapper_or_preset(
                model_id,
                candidate,
                pipe_function_id=pipe_function_id,
            )
        )

    info = arena_model.get("info")
    meta = info.get("meta") if isinstance(info, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    model_ids = meta.get("model_ids")
    filter_mode = meta.get("filter_mode")
    if model_ids and filter_mode == "exclude":
        excluded_model_ids = set(model_ids)
        return [
            model_id
            for available_model in list(models.values())
            if isinstance(available_model, dict)
            and available_model.get("owned_by") != "arena"
            and isinstance((model_id := available_model.get("id")), str)
            and model_id not in excluded_model_ids
            and is_allowed(model_id)
        ]
    if isinstance(model_ids, list) and model_ids:
        return [
            model_id
            for model_id in model_ids
            if isinstance(model_id, str) and model_id and is_allowed(model_id)
        ]
    return [
        model_id
        for available_model in list(models.values())
        if isinstance(available_model, dict)
        and available_model.get("owned_by") != "arena"
        and isinstance((model_id := available_model.get("id")), str)
        and is_allowed(model_id)
    ]


def _resolve_arena_chat_model_route(
    models: dict[str, Any],
    route: CoreChatModelRoute,
    *,
    pipe_function_id: str = PIPE_FUNCTION_ID,
) -> tuple[CoreChatModelRoute, str | None]:
    model = models.get(route.model_id)
    if not isinstance(model, dict) or not _is_arena_model(model):
        return route, None
    candidate_model_ids = _arena_chat_candidate_model_ids(
        models,
        model,
        pipe_function_id=pipe_function_id,
    )
    if not candidate_model_ids:
        raise HTTPException(status_code=403, detail="Model not found")
    selected_model_id = random.choice(candidate_model_ids)
    selected_model = models.get(selected_model_id)
    if not isinstance(selected_model, dict) or _is_arena_model(selected_model):
        raise HTTPException(status_code=403, detail="Model not found")
    fallback_model = copy.deepcopy(selected_model) if route.fallback_model is not None else route.fallback_model
    return (
        CoreChatModelRoute(
            model_id=selected_model_id,
            fallback_model=fallback_model,
            target_params=route.target_params,
            usage_anchor_shaping_hash=route.usage_anchor_shaping_hash,
            provider_model_id=selected_model_id,
        ),
        selected_model_id,
    )


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


def _is_own_wrapper_or_preset(
    model_id: str,
    model: dict[str, Any],
    *,
    pipe_function_id: str = PIPE_FUNCTION_ID,
) -> bool:
    return is_generated_wrapper_model_id(
        model_id,
        pipe_function_id=pipe_function_id,
    ) or _is_based_on_generated_wrapper(
        model,
        pipe_function_id=pipe_function_id,
    )


def _is_auto_compaction_wrapper(model: dict[str, Any]) -> bool:
    info = model.get("info")
    if not isinstance(info, dict):
        info = _payload_dict(info)
    for raw_meta in (model.get("meta"), info.get("meta")):
        meta = raw_meta if isinstance(raw_meta, dict) else _payload_dict(raw_meta)
        if isinstance(meta.get("auto_compaction"), dict):
            return True
    return False


def filter_target_models(models: Iterable[dict[str, Any]], valves: Any, *, pipe_function_id: str = PIPE_FUNCTION_ID):
    include_patterns = _split_patterns(getattr(valves, "include_model_patterns", ""))
    exclude_patterns = _split_patterns(getattr(valves, "exclude_model_patterns", ""))
    seen: set[str] = set()
    targets: list[dict[str, Any]] = []
    model_candidates = [model for model in models if isinstance(model, dict)]
    auto_compaction_wrapper_ids = {
        model_id
        for model in model_candidates
        if (model_id := _model_id(model)) is not None and _is_auto_compaction_wrapper(model)
    }

    for raw_model in model_candidates:
        model = copy.deepcopy(raw_model)
        model_id = _model_id(model)
        if model_id is None or model_id in seen:
            continue
        if model_id in auto_compaction_wrapper_ids:
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


def update_latest_models_cache(models: Iterable[dict[str, Any]]) -> None:
    """Snapshot the latest model dict for Valves dropdown population."""
    snapshot: dict[str, dict[str, Any]] = {}
    for raw in models:
        if not isinstance(raw, dict):
            continue
        mid = _model_id(raw)
        if mid and mid not in snapshot:
            snapshot[mid] = raw
    if snapshot:
        _LATEST_MODELS_CACHE.clear()
        _LATEST_MODELS_CACHE.update(snapshot)


def update_latest_provider_model_cache_enabled_states(states: dict[str, bool | None], *, state: Any = None) -> None:
    global _LATEST_PROVIDER_MODEL_CACHE_STATE_ID
    snapshot = {attr: states.get(attr) for attr in PROVIDER_MODEL_CACHE_ENABLE_FLAGS}
    if any(value is not None for value in snapshot.values()):
        _LATEST_PROVIDER_MODEL_CACHE_ENABLED_STATES.clear()
        _LATEST_PROVIDER_MODEL_CACHE_ENABLED_STATES.update(snapshot)
        _LATEST_PROVIDER_MODEL_CACHE_STATE_ID = id(state) if state is not None else None


def _latest_or_legacy_provider_model_cache_enabled_states(state: Any) -> dict[str, bool | None]:
    if (
        _LATEST_PROVIDER_MODEL_CACHE_ENABLED_STATES
        and (_LATEST_PROVIDER_MODEL_CACHE_STATE_ID is None or _LATEST_PROVIDER_MODEL_CACHE_STATE_ID == id(state))
    ):
        return dict(_LATEST_PROVIDER_MODEL_CACHE_ENABLED_STATES)
    return {attr: _provider_model_cache_enabled_state(state, attr) for attr in PROVIDER_MODEL_CACHE_ENABLE_FLAGS}


def refresh_latest_models_cache_from_app_state() -> None:
    """Refresh the dropdown model snapshot from Core's current app.state registries.

    Narrowly scoped: reads exclusively from app.state registries and performs no
    provider fetch, DB lookup, or async wait. If Core state is not importable or
    has no models yet, keep the existing snapshot unchanged.
    """
    try:
        from open_webui.main import app

        provider_states = _latest_or_legacy_provider_model_cache_enabled_states(app.state)
        disabled_provider_attrs = _disabled_provider_model_cache_attrs_from_states(provider_states)
        models = _iter_cache_models_from_state(app.state, disabled_provider_attrs=disabled_provider_attrs)
    except Exception:
        return
    update_latest_models_cache(models)


def build_summary_model_options(
    models: dict[str, dict[str, Any]],
    *,
    pipe_function_id: str = PIPE_FUNCTION_ID,
) -> list[dict[str, str]]:
    """Build dropdown options for summary_model, excluding wrapper and arena models."""
    options: list[dict[str, str]] = []
    for mid, model in sorted(models.items()):
        if not isinstance(model, dict):
            continue
        if is_generated_wrapper_model_id(mid, pipe_function_id=pipe_function_id):
            continue
        if _is_based_on_generated_wrapper(model, pipe_function_id=pipe_function_id):
            continue
        if _is_arena_model(model):
            continue
        name = str(model.get("name") or mid)
        label = f"{name} ({mid})" if name != mid else mid
        options.append({"value": mid, "label": label})
    return options


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
        target_params=target_params,
        wrapper_params=build_wrapper_core_params(target_params),
        access_grants=grants,
        owner_user_id=owner_user_id,
    )


def format_wrapper_model_name(
    target: TargetModelContract,
    *,
    template: Any,
    hide_wrapped_target_models: bool,
) -> str:
    template_text = str(template if template is not None else "auto").strip()
    auto_name = target.name if hide_wrapped_target_models else f"{target.name} (AutoCompact)"

    if not template_text or template_text.lower() == "auto":
        return auto_name

    # Avoid Python's format mini-language: width specs can allocate huge strings.
    unsupported_template = template_text.replace("{target_name}", "").replace("{target_id}", "")
    if "{" in unsupported_template or "}" in unsupported_template:
        return auto_name

    rendered = re.sub(
        r"\{target_name\}|\{target_id\}",
        lambda match: target.name if match.group(0) == "{target_name}" else target.id,
        template_text,
    )
    rendered = rendered.strip()
    return rendered or auto_name


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
    hide_wrapped_target_models = bool(getattr(valves, "hide_wrapped_target_models", False))
    wrapper_name = format_wrapper_model_name(
        target,
        template=getattr(valves, "wrapper_model_name_template", "auto"),
        hide_wrapped_target_models=hide_wrapped_target_models,
    )
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
    meta.pop("hidden", None)
    meta.pop(AUTO_COMPACTION_TARGET_HIDDEN_META_KEY, None)
    if _get_chat_variables_schema is not None:
        chat_variables_schema = _get_chat_variables_schema(target.target_params.get("system"))
        if chat_variables_schema:
            meta["chat_variables_schema"] = chat_variables_schema
    meta["auto_compaction"] = {
        "pipe_function_id": pipe_function_id,
        "target_model_id": target_id,
    }
    wrapper_capabilities = meta.get("capabilities")
    if not isinstance(wrapper_capabilities, dict):
        wrapper_capabilities = {}
        meta["capabilities"] = wrapper_capabilities
    wrapper_capabilities["file_context"] = False
    return {
        "id": wrapper_id,
        "user_id": function_owner_user_id,
        "base_model_id": None,
        "name": wrapper_name,
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


def _record_meta_dict(record: Any) -> dict[str, Any]:
    return _payload_dict(_record_field(record, "meta"))


def _record_params_dict(record: Any) -> dict[str, Any]:
    return _payload_dict(_record_field(record, "params"))


def _record_access_grants(record: Any) -> list[dict[str, Any]]:
    return _normalize_access_grants(_record_field(record, "access_grants"))


def _record_has_meta_key(record: Any, key: str) -> bool:
    return key in _record_meta_dict(record)


def _record_meta_value(record: Any, key: str, default: Any = None) -> Any:
    return _record_meta_dict(record).get(key, default)


def _target_model_visibility_lock(target_model_id: str) -> asyncio.Lock:
    lock = TARGET_MODEL_VISIBILITY_LOCKS.get(target_model_id)
    if lock is None:
        lock = asyncio.Lock()
        TARGET_MODEL_VISIBILITY_LOCKS[target_model_id] = lock
    return lock


def _managed_wrapper_pipe_function_id(record: Any) -> str | None:
    meta = _record_meta_dict(record)
    auto_compaction = meta.get("auto_compaction")
    if not isinstance(auto_compaction, dict):
        return None
    pipe_function_id = auto_compaction.get("pipe_function_id")
    return pipe_function_id if isinstance(pipe_function_id, str) and pipe_function_id else None


def _managed_wrapper_target_model_id(record: Any) -> str | None:
    meta = _record_meta_dict(record)
    auto_compaction = meta.get("auto_compaction")
    if not isinstance(auto_compaction, dict):
        return None
    target_model_id = auto_compaction.get("target_model_id")
    return target_model_id if isinstance(target_model_id, str) and target_model_id else None


def _target_hidden_marker(meta: dict[str, Any]) -> dict[str, Any] | None:
    marker = meta.get(AUTO_COMPACTION_TARGET_HIDDEN_META_KEY)
    return marker if isinstance(marker, dict) else None


def _target_hidden_marker_owner_ids(marker: dict[str, Any]) -> list[str]:
    # pipe_function_ids is a legacy marker shape; new hide claims are single-owner.
    owner_ids: list[str] = []
    pipe_function_ids = marker.get("pipe_function_ids")
    if isinstance(pipe_function_ids, list):
        for owner_id in pipe_function_ids:
            if isinstance(owner_id, str) and owner_id and owner_id not in owner_ids:
                owner_ids.append(owner_id)
    pipe_function_id = marker.get("pipe_function_id")
    if isinstance(pipe_function_id, str) and pipe_function_id and pipe_function_id not in owner_ids:
        owner_ids.append(pipe_function_id)
    return owner_ids


def _target_hidden_marker_restore_state(marker: dict[str, Any]) -> tuple[bool, bool]:
    return (
        bool(marker.get("had_hidden", False)),
        bool(marker.get("previous_hidden", False)),
    )


def _build_target_hidden_marker(
    *,
    pipe_function_ids: list[str],
    had_hidden: bool,
    previous_hidden: bool,
) -> dict[str, Any]:
    if len(pipe_function_ids) == 1:
        marker = {
            "pipe_function_id": pipe_function_ids[0],
            "had_hidden": had_hidden,
            "previous_hidden": previous_hidden,
        }
    else:
        marker = {
            "pipe_function_ids": pipe_function_ids,
            "had_hidden": had_hidden,
            "previous_hidden": previous_hidden,
        }
    return marker


def _mark_target_model_hidden_by_pipe(
    meta: dict[str, Any],
    *,
    pipe_function_id: str,
) -> None:
    marker = _target_hidden_marker(meta)
    if marker is None:
        owner_ids = [pipe_function_id]
        had_hidden = "hidden" in meta
        previous_hidden = bool(meta.get("hidden", False))
    else:
        owner_ids = _target_hidden_marker_owner_ids(marker)
        if owner_ids and pipe_function_id not in owner_ids:
            return
        had_hidden, previous_hidden = _target_hidden_marker_restore_state(marker)
        if not owner_ids:
            had_hidden = "hidden" in meta
            previous_hidden = bool(meta.get("hidden", False))
            owner_ids = [pipe_function_id]
    marker = _build_target_hidden_marker(
        pipe_function_ids=owner_ids,
        had_hidden=had_hidden,
        previous_hidden=previous_hidden,
    )
    meta[AUTO_COMPACTION_TARGET_HIDDEN_META_KEY] = marker
    meta["hidden"] = True


def _restore_target_model_hidden_by_pipe(meta: dict[str, Any], *, pipe_function_id: str) -> bool:
    marker = _target_hidden_marker(meta)
    if marker is None:
        return False
    owner_ids = _target_hidden_marker_owner_ids(marker)
    if pipe_function_id not in owner_ids:
        return False
    had_hidden, previous_hidden = _target_hidden_marker_restore_state(marker)
    remaining_owner_ids = [owner_id for owner_id in owner_ids if owner_id != pipe_function_id]
    if remaining_owner_ids:
        meta[AUTO_COMPACTION_TARGET_HIDDEN_META_KEY] = _build_target_hidden_marker(
            pipe_function_ids=remaining_owner_ids,
            had_hidden=had_hidden,
            previous_hidden=previous_hidden,
        )
        meta["hidden"] = True
    elif had_hidden:
        meta["hidden"] = previous_hidden
        meta.pop(AUTO_COMPACTION_TARGET_HIDDEN_META_KEY, None)
    else:
        meta.pop("hidden", None)
        meta.pop(AUTO_COMPACTION_TARGET_HIDDEN_META_KEY, None)
    return True


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


def _model_form_from_payload(
    *,
    ModelForm: Any,
    ModelMeta: Any,
    ModelParams: Any,
    payload: dict[str, Any],
) -> Any:
    return ModelForm(
        id=payload["id"],
        base_model_id=payload.get("base_model_id"),
        name=payload["name"],
        params=ModelParams(**_payload_dict(payload.get("params"))),
        meta=ModelMeta(**_payload_dict(payload.get("meta"))),
        access_grants=payload.get("access_grants"),
        is_active=bool(payload.get("is_active", True)),
    )


def _model_form_from_record(
    *,
    ModelForm: Any,
    ModelMeta: Any,
    ModelParams: Any,
    record: Any,
    meta: dict[str, Any] | None = None,
    is_active: bool | None = None,
) -> Any:
    return ModelForm(
        id=_record_field(record, "id"),
        base_model_id=_record_field(record, "base_model_id"),
        name=_record_field(record, "name"),
        params=ModelParams(**_record_params_dict(record)),
        meta=ModelMeta(**(_payload_dict(meta) if meta is not None else _record_meta_dict(record))),
        access_grants=_record_access_grants(record),
        is_active=bool(_record_field(record, "is_active") if is_active is None else is_active),
    )


def _target_model_override_form_payload(
    *,
    pipe_function_id: str,
    target_model: dict[str, Any],
    target_model_info: Any | None = None,
) -> dict[str, Any]:
    target = build_target_model_contract(target_model, target_model_info)
    existing_record = (
        target_model_info
        if target_model_info is not None and target_model_info is not TARGET_MODEL_RECORD_UNKNOWN
        else None
    )
    meta = _record_meta_dict(existing_record) if existing_record is not None else copy.deepcopy(target.meta)
    _mark_target_model_hidden_by_pipe(
        meta,
        pipe_function_id=pipe_function_id,
    )
    return {
        "id": target.id,
        "base_model_id": _target_record_base_model_id(target_model_info),
        "name": _record_field(existing_record, "name") or target.name,
        "params": _record_params_dict(existing_record) if existing_record is not None else copy.deepcopy(target.target_params),
        "meta": meta,
        "access_grants": _record_access_grants(existing_record) if existing_record is not None else copy.deepcopy(target.access_grants),
        "is_active": bool(_record_field(target_model_info, "is_active"))
        if target_model_info is not None and target_model_info is not TARGET_MODEL_RECORD_UNKNOWN
        else True,
    }


async def _hide_target_model_record(
    *,
    Models: Any,
    ModelForm: Any,
    ModelMeta: Any,
    ModelParams: Any,
    pipe_function_id: str,
    target_model: dict[str, Any],
    target_model_info: Any | None,
    owner_user_id: str,
) -> None:
    target_id = _model_id(target_model)
    if not target_id:
        return
    async with _target_model_visibility_lock(target_id):
        try:
            existing = await Models.get_model_by_id(target_id)
        except Exception:
            existing = None
        if existing is None and target_model_info is not None:
            return
        payload = _target_model_override_form_payload(
            pipe_function_id=pipe_function_id,
            target_model=target_model,
            target_model_info=existing,
        )
        model_form = _model_form_from_payload(
            ModelForm=ModelForm,
            ModelMeta=ModelMeta,
            ModelParams=ModelParams,
            payload=payload,
        )
        if existing:
            if not _wrapper_model_record_matches_form(existing, model_form):
                await Models.update_model_by_id(target_id, model_form)
        else:
            await Models.insert_new_model(model_form, user_id=owner_user_id)


async def _restore_target_model_hidden_record(
    *,
    Models: Any,
    ModelForm: Any,
    ModelMeta: Any,
    ModelParams: Any,
    pipe_function_id: str,
    target_model_id: str,
    target_model_info: Any | None = None,
) -> bool:
    async with _target_model_visibility_lock(target_model_id):
        try:
            existing = await Models.get_model_by_id(target_model_id)
        except Exception:
            existing = None
        if existing is None:
            return target_model_info is None
        meta = _record_meta_dict(existing)
        restored = _restore_target_model_hidden_by_pipe(meta, pipe_function_id=pipe_function_id)
        if not restored:
            return True
        model_form = _model_form_from_record(
            ModelForm=ModelForm,
            ModelMeta=ModelMeta,
            ModelParams=ModelParams,
            record=existing,
            meta=meta,
        )
        try:
            return await Models.update_model_by_id(target_model_id, model_form) is not None
        except Exception:
            return False


async def _deactivate_stale_wrapper_model_records(
    *,
    Models: Any,
    ModelForm: Any,
    ModelMeta: Any,
    ModelParams: Any,
    pipe_function_id: str,
    desired_wrapper_ids: set[str],
    existing_models_by_id: dict[str, Any],
) -> None:
    for snapshot in existing_models_by_id.values():
        try:
            existing_id = _record_field(snapshot, "id")
            if (
                not isinstance(existing_id, str)
                or existing_id in desired_wrapper_ids
                or _managed_wrapper_pipe_function_id(snapshot) != pipe_function_id
                or _record_field(snapshot, "is_active") is False
            ):
                continue
            existing = await Models.get_model_by_id(existing_id)
            if (
                existing is None
                or existing_id in desired_wrapper_ids
                or _managed_wrapper_pipe_function_id(existing) != pipe_function_id
                or _record_field(existing, "is_active") is False
            ):
                continue
            target_model_id = _managed_wrapper_target_model_id(existing)
            if target_model_id and not await _restore_target_model_hidden_record(
                Models=Models,
                ModelForm=ModelForm,
                ModelMeta=ModelMeta,
                ModelParams=ModelParams,
                pipe_function_id=pipe_function_id,
                target_model_id=target_model_id,
                target_model_info=existing_models_by_id.get(target_model_id),
            ):
                continue
            model_form = _model_form_from_record(
                ModelForm=ModelForm,
                ModelMeta=ModelMeta,
                ModelParams=ModelParams,
                record=existing,
                is_active=False,
            )
            await Models.update_model_by_id(existing_id, model_form)
        except Exception:
            continue


async def sync_wrapper_model_records(
    *,
    pipe_function_id: str,
    target_models: list[dict[str, Any]],
    valves: Any,
    existing_models: list[Any] | None = None,
    target_models_complete: bool = True,
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

    if existing_models is None:
        try:
            existing_models = list(await Models.get_all_models())
        except Exception:
            existing_models = None
    existing_models_by_id = (
        {
            existing_id: existing
            for existing in existing_models
            if isinstance(existing_id := _record_field(existing, "id"), str) and existing_id
        }
        if existing_models is not None
        else {}
    )

    desired_wrapper_ids: set[str] = set()
    hide_wrapped_target_models = bool(getattr(valves, "hide_wrapped_target_models", False))
    for target_model in target_models:
        target_id = _model_id(target_model)
        try:
            target_model_info = None
            if target_id:
                desired_wrapper_ids.add(build_wrapper_model_id(pipe_function_id, target_id))
                if existing_models is not None:
                    target_model_info = existing_models_by_id.get(target_id)
                else:
                    with suppress(Exception):
                        target_model_info = await Models.get_model_by_id(target_id)
            target_marker = _target_hidden_marker(_record_meta_dict(target_model_info))
            should_restore_target = (
                existing_models is None
                or (
                    target_marker is not None
                    and pipe_function_id in _target_hidden_marker_owner_ids(target_marker)
                )
            )
            if not hide_wrapped_target_models and target_id and should_restore_target:
                with suppress(Exception):
                    await _restore_target_model_hidden_record(
                        Models=Models,
                        ModelForm=ModelForm,
                        ModelMeta=ModelMeta,
                        ModelParams=ModelParams,
                        pipe_function_id=pipe_function_id,
                        target_model_id=target_id,
                        target_model_info=target_model_info,
                    )
            form_payload = build_wrapper_model_form(
                pipe_function_id=pipe_function_id,
                function_owner_user_id=owner_user_id,
                target_model=target_model,
                target_model_info=target_model_info,
                valves=valves,
            )
            existing = (
                existing_models_by_id.get(form_payload["id"])
                if existing_models is not None
                else await Models.get_model_by_id(form_payload["id"])
            )
            if existing and _record_has_meta_key(existing, "hidden"):
                form_payload["meta"]["hidden"] = _record_meta_value(existing, "hidden")
            model_form = _model_form_from_payload(
                ModelForm=ModelForm,
                ModelMeta=ModelMeta,
                ModelParams=ModelParams,
                payload=form_payload,
            )
            if not existing or not _wrapper_model_record_matches_form(existing, model_form):
                if existing_models is not None:
                    existing = await Models.get_model_by_id(form_payload["id"])
                    form_payload["meta"].pop("hidden", None)
                    if existing and _record_has_meta_key(existing, "hidden"):
                        form_payload["meta"]["hidden"] = _record_meta_value(existing, "hidden")
                    model_form = _model_form_from_payload(
                        ModelForm=ModelForm,
                        ModelMeta=ModelMeta,
                        ModelParams=ModelParams,
                        payload=form_payload,
                    )
                if existing:
                    if (
                        not _wrapper_model_record_matches_form(existing, model_form)
                        and await Models.update_model_by_id(form_payload["id"], model_form) is None
                    ):
                        raise RuntimeError("wrapper model update failed")
                elif await Models.insert_new_model(model_form, user_id=owner_user_id) is None:
                    raise RuntimeError("wrapper model insert failed")
            if hide_wrapped_target_models:
                with suppress(Exception):
                    await _hide_target_model_record(
                        Models=Models,
                        ModelForm=ModelForm,
                        ModelMeta=ModelMeta,
                        ModelParams=ModelParams,
                        pipe_function_id=pipe_function_id,
                        target_model=target_model,
                        target_model_info=target_model_info,
                        owner_user_id=owner_user_id,
                    )
        except Exception:
            if hide_wrapped_target_models and target_id:
                with suppress(Exception):
                    await _restore_target_model_hidden_record(
                        Models=Models,
                        ModelForm=ModelForm,
                        ModelMeta=ModelMeta,
                        ModelParams=ModelParams,
                        pipe_function_id=pipe_function_id,
                        target_model_id=target_id,
                        target_model_info=target_model_info,
                    )
            continue
    if existing_models is not None and target_models_complete:
        await _deactivate_stale_wrapper_model_records(
            Models=Models,
            ModelForm=ModelForm,
            ModelMeta=ModelMeta,
            ModelParams=ModelParams,
            pipe_function_id=pipe_function_id,
            desired_wrapper_ids=desired_wrapper_ids,
            existing_models_by_id=existing_models_by_id,
        )


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


async def _maybe_await_config_value(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _open_webui_config_get(key: str) -> Any:
    try:
        from open_webui.models.config import Config
    except Exception:
        return CONFIG_VALUE_MISSING

    try:
        return await _maybe_await_config_value(Config.get(key))
    except Exception:
        return CONFIG_VALUE_MISSING


async def _open_webui_config_get_many(*keys: str) -> dict[str, Any] | None:
    try:
        from open_webui.models.config import Config
    except Exception:
        return None

    try:
        values = await _maybe_await_config_value(Config.get_many(*keys))
    except Exception:
        return None
    return values if isinstance(values, dict) else None


async def _effective_arena_model_ids(
    state: Any,
    config_values: Any = CONFIG_VALUE_MISSING,
) -> set[str]:
    if config_values is CONFIG_VALUE_MISSING:
        config_values = await _open_webui_config_get_many(
            ARENA_ENABLE_CONFIG_KEY,
            ARENA_MODELS_CONFIG_KEY,
        )
    legacy_config = getattr(state, "config", None)
    if isinstance(config_values, dict) and ARENA_ENABLE_CONFIG_KEY in config_values:
        enabled = bool(config_values.get(ARENA_ENABLE_CONFIG_KEY))
    else:
        enabled = bool(getattr(legacy_config, "ENABLE_EVALUATION_ARENA_MODELS", False))
    if not enabled:
        return set()

    if isinstance(config_values, dict) and ARENA_MODELS_CONFIG_KEY in config_values:
        configured_models = config_values.get(ARENA_MODELS_CONFIG_KEY)
    else:
        configured_models = getattr(legacy_config, "EVALUATION_ARENA_MODELS", None)
    if isinstance(configured_models, list):
        configured_ids = {
            model_id
            for model in configured_models
            if isinstance(model, dict)
            and isinstance((model_id := model.get("id")), str)
            and model_id
        }
        if configured_ids:
            return configured_ids

    return {DEFAULT_ARENA_MODEL_ID}


def _config_value_is_enabled(value: Any) -> bool:
    if value is CONFIG_VALUE_MISSING:
        return False
    return bool(value)


async def _core_context_compaction_enabled() -> bool:
    return _config_value_is_enabled(await _open_webui_config_get(CORE_CONTEXT_COMPACTION_ENABLE_CONFIG_KEY))


def _core_context_compaction_conflict_response() -> dict[str, Any]:
    return _error_response(
        CORE_CONTEXT_COMPACTION_CONFLICT_MESSAGE,
        code="core_context_compaction_conflict",
    )


SUMMARY_FILE_CONTEXT_RAG_CONFIG_SPECS = {
    "k": ("rag.top_k", "TOP_K", 5),
    "k_reranker": ("rag.top_k_reranker", "TOP_K_RERANKER", 5),
    "r": ("rag.relevance_threshold", "RELEVANCE_THRESHOLD", 0.0),
    "hybrid_bm25_weight": ("rag.hybrid_bm25_weight", "HYBRID_BM25_WEIGHT", 0.5),
    "hybrid_search": ("rag.enable_hybrid_search", "ENABLE_RAG_HYBRID_SEARCH", False),
}


async def _summary_file_context_rag_config(request: Any) -> dict[str, Any]:
    state = getattr(getattr(request, "app", None), "state", None)
    legacy_config = getattr(state, "config", None)
    config_values = await _open_webui_config_get_many(
        *(spec[0] for spec in SUMMARY_FILE_CONTEXT_RAG_CONFIG_SPECS.values())
    )
    resolved: dict[str, Any] = {}
    for arg_name, (config_key, legacy_attr, default) in SUMMARY_FILE_CONTEXT_RAG_CONFIG_SPECS.items():
        value = CONFIG_VALUE_MISSING
        if isinstance(config_values, dict) and config_key in config_values:
            value = config_values.get(config_key)
        if value is CONFIG_VALUE_MISSING or value is None:
            value = getattr(legacy_config, legacy_attr, default)
        resolved[arg_name] = value
    return resolved


def _provider_model_cache_enabled_state(state: Any, attr: str) -> bool | None:
    flag = PROVIDER_MODEL_CACHE_ENABLE_FLAGS.get(attr)
    if flag is None:
        return None
    config = getattr(state, "config", None)
    if config is None or not hasattr(config, flag):
        return None
    return bool(getattr(config, flag))


def _provider_model_cache_enabled_config_value(config_values: dict[str, Any] | None, attr: str) -> bool | None:
    if not isinstance(config_values, dict):
        return None
    config_key = PROVIDER_MODEL_CACHE_CONFIG_KEYS.get(attr)
    if config_key is None or config_key not in config_values:
        return None
    value = config_values.get(config_key)
    if value is None:
        return None
    return bool(value)


def _provider_model_cache_enabled_states_from_config(
    state: Any,
    config_values: dict[str, Any] | None,
) -> dict[str, bool | None]:
    states: dict[str, bool | None] = {}
    for attr in PROVIDER_MODEL_CACHE_ENABLE_FLAGS:
        enabled = _provider_model_cache_enabled_config_value(config_values, attr)
        if enabled is None:
            enabled = _provider_model_cache_enabled_state(state, attr)
        states[attr] = enabled
    return states


async def _provider_model_cache_enabled_states(
    state: Any,
    config_values: Any = CONFIG_VALUE_MISSING,
) -> dict[str, bool | None]:
    if config_values is CONFIG_VALUE_MISSING:
        config_values = await _open_webui_config_get_many(*PROVIDER_MODEL_CACHE_CONFIG_KEYS.values())
    states = _provider_model_cache_enabled_states_from_config(state, config_values)
    update_latest_provider_model_cache_enabled_states(states, state=state)
    return states


def _disabled_provider_model_cache_attrs_from_states(states: dict[str, bool | None]) -> set[str]:
    return {attr for attr, enabled in states.items() if enabled is False}


def _enabled_provider_model_cache_attrs_from_states(states: dict[str, bool | None]) -> list[str]:
    return [attr for attr in PROVIDER_MODEL_CACHE_ENABLE_FLAGS if states.get(attr) is True]


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


def _iter_cache_models_from_state(
    state: Any,
    *,
    disabled_provider_attrs: set[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if disabled_provider_attrs is None:
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


async def _iter_cache_models_from_state_compatible(state: Any) -> list[dict[str, Any]]:
    provider_enabled_states = await _provider_model_cache_enabled_states(state)
    disabled_provider_attrs = _disabled_provider_model_cache_attrs_from_states(provider_enabled_states)
    return _iter_cache_models_from_state(state, disabled_provider_attrs=disabled_provider_attrs)


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
    provider_cache_attrs: Iterable[str] | None = None,
    disabled_provider_attrs: set[str] | None = None,
) -> list[dict[str, Any]]:
    if provider_cache_attrs is None:
        provider_cache_attrs = _enabled_provider_model_cache_attrs(state)
    else:
        provider_cache_attrs = list(provider_cache_attrs)
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
        models = _iter_cache_models_from_state(state, disabled_provider_attrs=disabled_provider_attrs)
        pending_provider_cache_attrs = _provider_model_cache_refresh_pending_attrs(provider_cache_attrs)
        if _provider_model_caches_ready(state, initial_provider_caches, pending_provider_cache_attrs):
            return models
        await asyncio.sleep(poll_seconds)
        models = _iter_cache_models_from_state(state, disabled_provider_attrs=disabled_provider_attrs)
        pending_provider_cache_attrs = _provider_model_cache_refresh_pending_attrs(provider_cache_attrs)
        if _provider_model_caches_ready(state, initial_provider_caches, pending_provider_cache_attrs):
            return models
    return _iter_cache_models_from_state(state, disabled_provider_attrs=disabled_provider_attrs)


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


def _raw_usage_from_stream_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    raw_usage: dict[str, Any] = {}
    containers = [
        payload.get("message"),
        payload.get("response"),
        payload.get("data"),
        payload,
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        usage = container.get("usage")
        if isinstance(usage, dict):
            raw_usage = _merge_usage_fields(raw_usage, usage)
    # llama.cpp emits token counters in a top-level `timings` object. Match
    # Core's raw extraction, but keep the fields unnormalized so prompt_n can
    # never masquerade as a complete input count without cache_n.
    for container in containers:
        if not isinstance(container, dict):
            continue
        timings = container.get("timings")
        if isinstance(timings, dict):
            raw_usage = _merge_usage_fields(raw_usage, timings)
    return raw_usage or None


def extract_usage_from_stream_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    usage = _raw_usage_from_stream_payload(payload)
    return _normalize_usage(usage) if usage else None


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


def _strict_usage_input_tokens(usage: dict[str, Any] | None) -> int | None:
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


def _merge_usage_fields(current: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(current or {})
    if isinstance(incoming, dict):
        for key, value in incoming.items():
            if value is not None:
                merged[key] = copy.deepcopy(value)
    return merged


def usage_state_key(chat_id: str | None, message_id: str | None, wrapper_model_id: str | None) -> str:
    digest = hashlib.sha256(
        json.dumps([chat_id or "", message_id or "", wrapper_model_id or ""], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"_auto_compact_usage_{digest}"


def usage_anchor_input_state_key(
    chat_id: str | None,
    message_id: str | None,
    wrapper_model_id: str | None,
) -> str:
    return f"{usage_state_key(chat_id, message_id, wrapper_model_id)}_anchor_input"


def store_request_scoped_usage(
    *,
    request: Any,
    chat_id: str | None,
    message_id: str | None,
    wrapper_model_id: str | None,
    usage: dict[str, Any],
    anchor_input: UsageAnchorInput | None = None,
) -> None:
    state = getattr(request, "state", None)
    if state is None:
        return
    key = usage_state_key(chat_id, message_id, wrapper_model_id)
    current = getattr(state, key, None)
    setattr(state, key, _merge_usage_fields(current if isinstance(current, dict) else None, usage))
    anchor_key = usage_anchor_input_state_key(chat_id, message_id, wrapper_model_id)
    if anchor_input is not None:
        setattr(state, anchor_key, anchor_input)
    else:
        with suppress(Exception):
            delattr(state, anchor_key)


def clear_request_scoped_usage(
    *,
    request: Any,
    chat_id: str | None,
    message_id: str | None,
    wrapper_model_id: str | None,
) -> None:
    state = getattr(request, "state", None)
    if state is None:
        return
    with suppress(Exception):
        delattr(state, usage_state_key(chat_id, message_id, wrapper_model_id))
    with suppress(Exception):
        delattr(state, usage_anchor_input_state_key(chat_id, message_id, wrapper_model_id))


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
        return _normalize_usage(dict(usage))
    return None


def get_request_scoped_usage_anchor(
    *,
    request: Any,
    chat_id: str | None,
    message_id: str | None,
    wrapper_model_id: str | None,
) -> UsageAnchor | None:
    state = getattr(request, "state", None)
    if state is None:
        return None
    raw_usage = getattr(state, usage_state_key(chat_id, message_id, wrapper_model_id), None)
    anchor_input = getattr(
        state,
        usage_anchor_input_state_key(chat_id, message_id, wrapper_model_id),
        None,
    )
    if not isinstance(raw_usage, dict) or not isinstance(anchor_input, UsageAnchorInput):
        return None
    input_tokens = _strict_usage_input_tokens(raw_usage)
    if input_tokens is None or input_tokens < anchor_input.volatile_message_tokens:
        return None
    return UsageAnchor(
        assistant_message_id="request",
        input_tokens=input_tokens,
        stable_message_count=anchor_input.stable_message_count,
        input_fingerprint=anchor_input.input_fingerprint,
        volatile_message_tokens=anchor_input.volatile_message_tokens,
    )


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
        metadata = body.get("metadata")
        copied_metadata = _copy_metadata_preserving_references(metadata)
        if isinstance(metadata, dict) and isinstance(metadata.get("tools"), dict):
            copied_metadata["tools"] = metadata["tools"]
        copied["metadata"] = copied_metadata
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


def _task_template_is_usable(template: Any, spec: TaskPromptSpec) -> bool:
    if not isinstance(template, str):
        return False
    return bool(template.strip() if spec.strip_configured else template != "")


async def _task_template_from_request(request: Any, spec: TaskPromptSpec) -> str | None:
    configured = await _open_webui_config_get(spec.config_key)
    if _task_template_is_usable(configured, spec):
        return configured

    config = getattr(getattr(getattr(request, "app", None), "state", None), "config", None)
    configured = getattr(config, spec.config_attr, None) if config is not None else None
    if _task_template_is_usable(configured, spec):
        return configured
    try:
        import open_webui.config as core_config

        default_template = getattr(core_config, spec.default_attr, None)
    except Exception:
        default_template = None
    return default_template if _task_template_is_usable(default_template, spec) else None


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
    template = await _task_template_from_request(request, spec)
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
    text = str(data)
    lines = re.split(r"\r\n|\r|\n", text)
    if lines and lines[-1] == "" and text.endswith(("\r", "\n")):
        lines = lines[:-1]
    if not lines:
        lines = [""]
    return "".join(f"data: {line}\n" for line in lines) + "\n"


def _immediate_response_is_error(response: Any) -> bool:
    if isinstance(response, dict):
        return _stream_payload_error_source(response) is not None
    if isinstance(response, (JSONResponse, PlainTextResponse)):
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and status_code >= 400:
            return True
        parsed = _json_from_response(response)
        if isinstance(parsed, dict):
            return _stream_payload_error_source(parsed) is not None
        if isinstance(parsed, list):
            return False
        return isinstance(response, PlainTextResponse)
    return False


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
    first_line = stripped.split("\n", 1)[0].split("\r", 1)[0]
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
        self._previous_chunk_ended_with_cr = False

    def has_pending_event_or_field(self) -> bool:
        return bool(self._data_lines) or _text_can_start_sse_field(self._buffer.lstrip())

    def feed(self, chunk: bytes | str) -> tuple[list[str], bool]:
        text = self._decoder.decode(chunk, final=False) if isinstance(chunk, bytes) else str(chunk)
        return self._process_text(text)

    def _process_text(self, text: str) -> tuple[list[str], bool]:
        if self._previous_chunk_ended_with_cr:
            if not text:
                return [], False
            if text.startswith("\n"):
                text = text[1:]
            self._previous_chunk_ended_with_cr = False
        self._buffer += text
        if not self._buffer:
            return [], False
        self._previous_chunk_ended_with_cr = self._buffer.endswith("\r")
        normalized = self._buffer
        if "\r" in normalized:
            normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        if normalized.endswith("\n"):
            lines.pop()
            self._buffer = ""
        else:
            self._buffer = lines.pop()
        return self._process_lines(lines)

    def flush(self) -> tuple[list[str], bool]:
        lines = []
        remaining_text = self._decoder.decode(b"", final=True)
        if remaining_text:
            values, saw_data_field = self._process_text(remaining_text)
        else:
            values, saw_data_field = [], False
        if self._buffer:
            lines.append(self._buffer)
            self._buffer = ""
        trailing_values, trailing_saw_data_field = self._process_lines(lines)
        values.extend(trailing_values)
        saw_data_field = saw_data_field or trailing_saw_data_field
        self._previous_chunk_ended_with_cr = False
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


def _stream_payload_is_responses_pre_output_control_event(payload: dict[str, Any]) -> bool:
    if _stream_payload_error_source(payload) is not None:
        return False
    payload_type = payload.get("type")
    if payload_type in {
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    }:
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
        streaming_response = StreamingResponse(iter([f"data: {json.dumps(response)}\n\n"]), media_type="text/event-stream")
        if _immediate_response_is_error(response):
            setattr(streaming_response, "_auto_compact_immediate_error", True)
        return streaming_response

    if isinstance(response, (JSONResponse, PlainTextResponse)):
        if is_retryable_context_error(response):
            if restore:
                restore()
            raise RetryableContextOverflow(_response_body_text(response))
        if restore:
            restore()
        streaming_response = StreamingResponse(iter([_response_to_sse_chunk(response)]), media_type="text/event-stream")
        if _immediate_response_is_error(response):
            setattr(streaming_response, "_auto_compact_immediate_error", True)
        return streaming_response

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
    input_tokens = _strict_usage_input_tokens(usage)
    output_tokens = None
    output_present = False
    for key in ("output_tokens", "completion_tokens", "eval_count", "predicted_n"):
        if key not in usage:
            continue
        output_present = True
        output_tokens = _strict_usage_token_value(usage, key)
        if output_tokens is None:
            return None
        break
    if input_tokens is not None and output_present:
        return input_tokens + (output_tokens or 0)
    cache_sensitive_input = any(
        key in usage
        for key in (
            "prompt_n",
            "cache_n",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )
    if cache_sensitive_input:
        if input_tokens is None:
            return None
        return input_tokens + (output_tokens or 0)
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
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> dict[str, Any]:
    if isinstance(messages, list):
        cut = select_safe_message_cut(
            messages,
            transient_message_patterns=transient_message_patterns,
        )
        tool_cut = select_tool_result_compaction_cut(
            messages,
            transient_message_patterns=transient_message_patterns,
        )
        message_mask = _transient_message_mask(messages, transient_message_patterns)
        latest_user = next(
            (
                message
                for index, message in reversed(list(enumerate(messages)))
                if message.get("role") == "user"
                and _is_source_identity_message(
                    message,
                    transient_message_patterns=transient_message_patterns,
                    transient_message_mask=message_mask,
                    index=index,
                )
            ),
            None,
        )
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


def _is_image_file_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") == "image":
        return True
    content_type = item.get("content_type")
    return isinstance(content_type, str) and content_type.startswith("image/")


def _extract_non_image_file_ids(files_value: Any) -> set[str]:
    ids: set[str] = set()
    if not isinstance(files_value, list):
        return ids
    for item in files_value:
        if not isinstance(item, dict) or _is_image_file_item(item):
            continue
        file_id = item.get("id")
        if isinstance(file_id, str) and file_id:
            ids.add(file_id)
    return ids


async def _load_chat_message_chain(
    request: Any,
    chat_id: str | None,
    current_message_id: str | None,
) -> list[dict[str, Any]] | None:
    if not chat_id or not current_message_id:
        return None
    try:
        from open_webui.models.chats import Chats
        from open_webui.utils.misc import get_message_list
        from open_webui.utils.middleware import process_messages_with_output

        messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
        if not isinstance(messages_map, dict) or not messages_map:
            return None
        if current_message_id not in messages_map:
            return None
        chain = get_message_list(messages_map, current_message_id)
        # Expand assistant-with-output messages to align positional indices
        # with the body.messages the pipe receives (Core runs
        # process_messages_with_output before the pipe). Core strips files
        # during that conversion, so retain each raw row's files on its first
        # expanded message for boundary-aware file classification.
        expanded_chain: list[dict[str, Any]] = []
        for message in chain:
            expanded_messages = process_messages_with_output([message])
            files = message.get("files") if isinstance(message, dict) else None
            if expanded_messages and files is not None:
                expanded_messages[0]["files"] = files
            expanded_chain.extend(expanded_messages)
        return expanded_chain
    except Exception:
        return None


def _classify_files_for_target(
    db_chain: list[dict[str, Any]] | None,
    compaction_prefix_count: int,
    metadata_user_message: Any,
    metadata_files: Any,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> list[Any]:
    """Return retained files for target forward injection using only DB-chain positions."""
    if not isinstance(metadata_files, list):
        return metadata_files

    current_file_ids: set[str] = set()
    if isinstance(metadata_user_message, dict):
        current_file_ids |= _extract_non_image_file_ids(metadata_user_message.get("files"))

    if not db_chain:
        # No compaction (prefix_count==0) means the target sees the full
        # conversation — ALL files are relevant.  Only when compaction
        # happened but the DB is unavailable do we conservatively keep
        # just the current-turn files.
        if compaction_prefix_count == 0:
            return list(metadata_files)
        retained_files: list[Any] = []
        for file_item in metadata_files:
            if not isinstance(file_item, dict) or _is_image_file_item(file_item):
                retained_files.append(file_item)
                continue
            file_id = file_item.get("id")
            if not isinstance(file_id, str) or not file_id:
                retained_files.append(file_item)
                continue
            if file_id in current_file_ids:
                retained_files.append(file_item)
        return retained_files

    retained_ids: set[str] = set(current_file_ids)
    # select_safe_message_cut preserves the FIRST system message wherever it
    # sits and replace_prefix_with_summary restores it verbatim into the
    # compacted target body, so files referenced by the chain's first system
    # row must stay in the forward request even when it sits before the
    # compaction boundary. When the body carries an injected model system
    # prompt instead, the chain's first system row is not the preserved one;
    # retaining its files anyway only errs toward keeping context available.
    for message in db_chain:
        if isinstance(message, dict) and _is_system_message(message):
            retained_ids |= _extract_non_image_file_ids(message.get("files"))
            break
    for index in range(
        _raw_chain_boundary(
            db_chain,
            compaction_prefix_count,
            transient_message_patterns=transient_message_patterns,
        ),
        len(db_chain),
    ):
        message = db_chain[index]
        if isinstance(message, dict):
            retained_ids |= _extract_non_image_file_ids(message.get("files"))

    all_db_file_ids: set[str] = set()
    for message in db_chain:
        if isinstance(message, dict):
            all_db_file_ids |= _extract_non_image_file_ids(message.get("files"))

    retained_files: list[Any] = []
    for file_item in metadata_files:
        if not isinstance(file_item, dict) or _is_image_file_item(file_item):
            retained_files.append(file_item)
            continue
        file_id = file_item.get("id")
        if not isinstance(file_id, str) or not file_id:
            retained_files.append(file_item)
            continue
        if file_id in retained_ids or file_id not in all_db_file_ids:
            retained_files.append(file_item)
    return retained_files


def _classify_files_for_summary(
    db_chain: list[dict[str, Any]] | None,
    compaction_prefix_count: int,
    parent_source_message_count: int,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> set[str]:
    """Return prefix file ids for summary context, excluding parent-checkpoint absorbed messages and system rows."""
    if not db_chain:
        return set()
    prefix_ids: set[str] = set()
    for index in range(
        _raw_chain_boundary(
            db_chain,
            parent_source_message_count,
            transient_message_patterns=transient_message_patterns,
        ),
        _raw_chain_boundary(
            db_chain,
            compaction_prefix_count,
            transient_message_patterns=transient_message_patterns,
        ),
    ):
        if index < len(db_chain):
            message = db_chain[index]
            # System rows feed checkpoint fingerprints through these ids, so
            # files stored on them must not affect checkpoint identity.
            if _is_source_identity_message(
                message,
                transient_message_patterns=transient_message_patterns,
            ):
                prefix_ids |= _extract_non_image_file_ids(message.get("files"))
    return prefix_ids


async def _noop_event_emitter(_event: Any) -> None:
    pass


async def _emit_source_events(
    event_emitter: Callable[[Any], Awaitable[None]] | None,
    sources: Any,
) -> None:
    if event_emitter is None or not isinstance(sources, list):
        return
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_info = source.get("source", {})
        if not isinstance(source_info, dict):
            continue
        if not (source_info.get("name", "") or source_info.get("id", "")):
            continue
        try:
            await event_emitter({"type": "source", "data": source})
        except Exception:
            return


def _displayable_sources(sources: Any) -> list[dict[str, Any]]:
    if not isinstance(sources, list):
        return []
    displayable = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_info = source.get("source", {})
        if not isinstance(source_info, dict):
            continue
        if not (source_info.get("name", "") or source_info.get("id", "")):
            continue
        displayable.append(source)
    return displayable


def _merge_source_events_into_response(response: Any, sources: Any) -> Any:
    # Core merges RAG sources into the non-streaming HTTP response body via
    # merge_events_into_response (events carry {"sources": [...]}).  The wrapper
    # disables Core's pre-pipe RAG, so the manual injection sources must be
    # merged here for API / non-event-emitter callers to retain citations.
    # Mirror Core semantics: existing response keys win over merged events.
    if not isinstance(response, dict):
        return response
    displayable = _displayable_sources(sources)
    if not displayable:
        return response
    if "sources" in response:
        return response
    return {"sources": displayable, **response}


def _format_summary_file_context(sources: Any) -> str | None:
    if not isinstance(sources, list) or not sources:
        return None

    lines = ["<attached_file_contents>"]
    emitted = 0
    for source_index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        source_info = source.get("source") if isinstance(source.get("source"), dict) else {}
        documents = source.get("document")
        if not isinstance(documents, list):
            documents = source.get("documents")
        if not isinstance(documents, list):
            documents = []
        metadata_values = source.get("metadata")
        metadatas: list[Any] = metadata_values if isinstance(metadata_values, list) else []
        source_id = source_info.get("id") or source_info.get("source")
        source_name = source_info.get("name") or source_info.get("filename") or source_id or f"source-{source_index}"
        for document_index, document in enumerate(documents, start=1):
            if document is None:
                continue
            metadata: dict[str, Any] = {}
            if document_index - 1 < len(metadatas):
                candidate_metadata = metadatas[document_index - 1]
                if isinstance(candidate_metadata, dict):
                    metadata = candidate_metadata
            file_id = metadata.get("source") or metadata.get("file_id") or source_id or f"source-{source_index}"
            file_name = metadata.get("name") or metadata.get("filename") or source_name
            lines.append(
                f'<file index="{emitted + 1}" source_index="{_xml_attr(source_index)}" '
                f'document_index="{_xml_attr(document_index)}" id="{_xml_attr(file_id)}" '
                f'name="{_xml_attr(file_name)}">{_xml_cdata(document)}</file>'
            )
            emitted += 1
    if emitted == 0:
        return None
    lines.append("</attached_file_contents>")
    return "\n".join(lines)


async def _generate_summary_file_context(
    *,
    request: Any,
    user: Any,
    prefix_files: list[Any],
) -> str | None:
    try:
        from open_webui.retrieval.utils import get_sources_from_items

        state = getattr(getattr(request, "app", None), "state", None)
        embedding_function = getattr(state, "EMBEDDING_FUNCTION", None)
        reranking_function = getattr(state, "RERANKING_FUNCTION", None)
        user_model = coerce_open_webui_user(user)
        rag_config = await _summary_file_context_rag_config(request)

        def embed(query: str, prefix: str) -> Any:
            if embedding_function is None:
                return None
            return embedding_function(query, prefix=prefix, user=user_model)

        rerank = None
        if reranking_function is not None:
            def rerank(query: str, documents: list[Any]) -> Any:
                return reranking_function(query, documents, user=user_model)

        sources = await get_sources_from_items(
            request=request,
            items=prefix_files,
            queries=[""],
            embedding_function=embed,
            k=rag_config["k"],
            reranking_function=rerank,
            k_reranker=rag_config["k_reranker"],
            r=rag_config["r"],
            hybrid_bm25_weight=rag_config["hybrid_bm25_weight"],
            hybrid_search=rag_config["hybrid_search"],
            full_context=True,
            user=user_model,
        )
        return _format_summary_file_context(sources)
    except SummaryFileContextUnavailable:
        raise
    except Exception as exc:
        LOG.exception("Failed to generate summary file context")
        raise SummaryFileContextUnavailable() from exc


async def _prepare_summary_file_context(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    compaction_prefix_count: int,
    parent_source_message_count: int,
    file_context_enabled: bool = True,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> str | None:
    if not file_context_enabled:
        return None
    try:
        metadata_files = metadata.get("files")
        if not isinstance(metadata_files, list) or not metadata_files:
            return None
        if compaction_prefix_count <= parent_source_message_count:
            return None
        if not _extract_non_image_file_ids(metadata_files):
            return None
        required_file_ids = _extract_non_image_file_ids(metadata_files)
        chat_id = str(metadata.get("chat_id") or "")
        current_message_id = str(metadata.get("user_message_id") or metadata.get("message_id") or "")
        db_chain = await _load_chat_message_chain(request, chat_id, current_message_id)
        if db_chain is None:
            if not required_file_ids:
                return None
            raise SummaryFileContextUnavailable()

        prefix_ids = _classify_files_for_summary(
            db_chain=db_chain,
            compaction_prefix_count=compaction_prefix_count,
            parent_source_message_count=parent_source_message_count,
            transient_message_patterns=transient_message_patterns,
        )
        if not prefix_ids:
            return None

        prefix_files = [
            file_item
            for file_item in metadata_files
            if isinstance(file_item, dict) and not _is_image_file_item(file_item) and file_item.get("id") in prefix_ids
        ]
        if not prefix_files:
            return None
        return await _generate_summary_file_context(request=request, user=user, prefix_files=prefix_files)
    except SummaryFileContextUnavailable:
        raise
    except Exception as exc:
        LOG.exception("Failed to prepare summary file context")
        raise SummaryFileContextUnavailable() from exc


def _target_model_supports_file_context(models: dict[str, Any], target_model_id: str) -> bool:
    """Check if the TARGET model (not wrapper) has file_context enabled.
    When the target itself disables file_context, the pipe must not
    inject file content either — the user/admin explicitly opted out."""
    target = models.get(target_model_id)
    if not isinstance(target, dict):
        return True  # unknown model → default to enabled (safe)

    capability_maps: list[Any] = [target.get("capabilities")]
    top_meta = target.get("meta")
    if isinstance(top_meta, dict):
        capability_maps.append(top_meta.get("capabilities"))
    info = target.get("info")
    if isinstance(info, dict):
        info_meta = info.get("meta")
        if isinstance(info_meta, dict):
            capability_maps.append(info_meta.get("capabilities"))

    for capabilities in capability_maps:
        if isinstance(capabilities, dict) and capabilities.get("file_context") is False:
            return False
    return True


async def _inject_target_file_context(
    *,
    request: Any,
    user: Any,
    body: dict[str, Any],
    chat_id: str | None,
    current_message_id: str | None,
    compaction_prefix_count: int,
    metadata_files: Any,
    metadata_user_message: Any,
    event_emitter: Callable[[Any], Awaitable[None]] | None,
    file_context_enabled: bool = True,
    emit_source_events: bool = True,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> dict[str, Any]:
    body_metadata = body.get("metadata")
    if isinstance(body_metadata, dict):
        body_metadata.pop("sources", None)
    if not isinstance(metadata_files, list) or not metadata_files:
        return body
    non_image = [file_item for file_item in metadata_files if isinstance(file_item, dict) and not _is_image_file_item(file_item)]
    if not non_image:
        return body

    db_chain = await _load_chat_message_chain(request, chat_id, current_message_id)
    retained_files = _classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=compaction_prefix_count,
        metadata_user_message=metadata_user_message,
        metadata_files=metadata_files,
        transient_message_patterns=transient_message_patterns,
    )
    # Always update body metadata files to retained-only so downstream pipes/functions
    # don't receive absorbed prefix files even if target RAG is disabled.
    body_metadata = body.get("metadata")
    if isinstance(body_metadata, dict):
        body_metadata["files"] = retained_files
        body_metadata.pop("sources", None)

    if not file_context_enabled:
        return body  # target opted out of file context — skip RAG injection

    if not retained_files:
        return body

    retained_non_image = [
        file_item for file_item in retained_files if isinstance(file_item, dict) and not _is_image_file_item(file_item)
    ]
    if not retained_non_image:
        return body

    if AUTO_COMPACT_FILE_CONTEXT_INJECTION_ACTIVE.get():
        # This execution context re-entered injection (e.g. generate_queries routed the
        # task model back into this wrapper). Skip manual RAG fail-closed so we
        # never recurse into chat_completion_files_handler again; the retained
        # metadata files pruning above still applies.
        return body
    injection_token = AUTO_COMPACT_FILE_CONTEXT_INJECTION_ACTIVE.set(True)
    try:
        try:
            from open_webui.utils.middleware import apply_source_context_to_messages, chat_completion_files_handler
            from open_webui.utils.misc import get_last_user_message

            messages = body.get("messages") if isinstance(body.get("messages"), list) else []
            rag_messages = _messages_for_transient_aware_rag(messages, transient_message_patterns)
            rag_body = {
                **body,
                "messages": rag_messages,
                "metadata": {
                    **(body.get("metadata") if isinstance(body.get("metadata"), dict) else {}),
                    "files": retained_non_image,
                },
            }
            extra_params = {"__event_emitter__": event_emitter or _noop_event_emitter}
            _, flags = await chat_completion_files_handler(request, rag_body, extra_params, coerce_open_webui_user(user))
            sources = flags.get("sources", []) if isinstance(flags, dict) else []
            if sources:
                last_user_msg = get_last_user_message(rag_messages) or ""
                applied_messages = await apply_source_context_to_messages(request, rag_messages, sources, last_user_msg)
                body["messages"] = _merge_rag_messages_preserving_transient_users(
                    messages,
                    applied_messages,
                    transient_message_patterns,
                )
                # Propagate sources to body metadata and event_emitter, matching
                # Core's behaviour so UI citation/source display and downstream
                # consumers work the same as non-AutoCompact file-context.
                body_metadata = body.get("metadata")
                if isinstance(body_metadata, dict):
                    body_metadata["sources"] = sources[:]
                if emit_source_events:
                    await _emit_source_events(event_emitter, sources)
        except Exception:
            LOG.exception("Failed to inject target file context")
    finally:
        AUTO_COMPACT_FILE_CONTEXT_INJECTION_ACTIVE.reset(injection_token)
    return body


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
    if _responses_output_has_tool_call(payload):
        return True
    nested_response = payload.get("response")
    if isinstance(nested_response, dict) and _responses_output_has_tool_call(nested_response):
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
    delta = payload.get("delta")
    if isinstance(delta, dict) and delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
        return delta["text"]
    return None


def _observe_streaming_completion_payload(payload: dict[str, Any], state: dict[str, Any]) -> None:
    if _stream_payload_error_source(payload) is not None:
        state["saw_error"] = True
        return
    raw_usage = _raw_usage_from_stream_payload(payload)
    if raw_usage:
        state["raw_usage"] = _merge_usage_fields(state.get("raw_usage"), raw_usage)
    if _stream_payload_has_tool_call(payload):
        state["saw_tool_call"] = True
    text = _stream_payload_text(payload)
    if text is None and not state.get("parts"):
        text = _responses_output_text(payload)
        nested_response = payload.get("response")
        if text is None and isinstance(nested_response, dict):
            text = _responses_output_text(nested_response)
    if text:
        state.setdefault("parts", []).append(text)


async def _streaming_completion_observer(
    iterator: AsyncIterator[bytes | str],
    *,
    media_type: str,
    request: Any = None,
    chat_id: str | None = None,
    message_id: str | None = None,
    wrapper_model_id: str | None = None,
    anchor_input: UsageAnchorInput | None = None,
    on_complete: Callable[[dict[str, Any]], Any] | None = None,
    on_terminal: Callable[[bool], Any] | None = None,
) -> AsyncIterator[bytes | str]:
    state: dict[str, Any] = {
        "parts": [],
        "raw_usage": None,
        "saw_tool_call": False,
        "saw_error": False,
    }
    is_sse_response = "text/event-stream" in (media_type or "").lower()
    parser = _SSEJSONEventParser() if is_sse_response else None
    try:
        async for raw_chunk in iterator:
            chunk = _coerce_stream_chunk(raw_chunk)
            if is_sse_response and parser is not None:
                events = parser.feed(chunk)
            else:
                events = extract_sse_json_events(chunk)
            for payload in events:
                _observe_streaming_completion_payload(payload, state)
            yield raw_chunk
        if is_sse_response and parser is not None:
            for payload in parser.flush():
                _observe_streaming_completion_payload(payload, state)
        if state.get("saw_error"):
            return
        raw_usage = state.get("raw_usage")
        if isinstance(raw_usage, dict) and raw_usage:
            store_request_scoped_usage(
                request=request,
                chat_id=chat_id,
                message_id=message_id,
                wrapper_model_id=wrapper_model_id,
                usage=raw_usage,
                anchor_input=anchor_input,
            )
        if state.get("saw_tool_call") or on_complete is None:
            return
        content = "".join(state.get("parts") or [])
        if not content and not raw_usage:
            return
        assistant_message = {"role": "assistant", "content": content}
        with suppress(Exception):
            result = on_complete(
                {
                    "assistant_message": assistant_message,
                    "usage": _normalize_usage(raw_usage) if raw_usage else None,
                    "raw_usage": raw_usage,
                }
            )
            if inspect.isawaitable(result):
                await result
    finally:
        aclose = getattr(iterator, "aclose", None)
        if callable(aclose):
            with suppress(Exception):
                await aclose()
        if on_terminal is not None:
            with suppress(Exception):
                result = on_terminal(bool(state.get("saw_tool_call")))
                if inspect.isawaitable(result):
                    await result


def _attach_streaming_completion_observer(
    response: StreamingResponse,
    on_complete: Callable[[dict[str, Any]], Any] | None = None,
    *,
    request: Any = None,
    chat_id: str | None = None,
    message_id: str | None = None,
    wrapper_model_id: str | None = None,
    anchor_input: UsageAnchorInput | None = None,
    on_terminal: Callable[[bool], Any] | None = None,
) -> StreamingResponse:
    media_type = response.headers.get("content-type", getattr(response, "media_type", "") or "")
    response.body_iterator = _streaming_completion_observer(
        response.body_iterator,
        media_type=media_type,
        request=request,
        chat_id=chat_id,
        message_id=message_id,
        wrapper_model_id=wrapper_model_id,
        anchor_input=anchor_input,
        on_complete=on_complete,
        on_terminal=on_terminal,
    )
    return response


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
    if payload.get("error"):
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


@dataclass(frozen=True)
class DisplayTokenContext:
    before: int | None
    usage: int | None
    estimate: int | None
    hard_limit: int
    soft_limit: int | None
    usage_source: str | None
    pct_of_hard: float | None


def _build_display_token_context(
    *,
    estimated_total_tokens: int | None,
    total_tokens: int | None,
    effective_trigger_input_tokens: int,
    effective_soft_trigger_input_tokens: int | None,
    usage_source: str | None,
) -> DisplayTokenContext:
    before = None
    display_usage_source = usage_source
    if estimated_total_tokens is not None:
        before = estimated_total_tokens
        if total_tokens is None or estimated_total_tokens != total_tokens:
            display_usage_source = "estimate"
    elif total_tokens is not None:
        before = total_tokens

    pct_of_hard = None
    if before is not None and effective_trigger_input_tokens > 0:
        pct_of_hard = round(before / effective_trigger_input_tokens * 100, 1)

    return DisplayTokenContext(
        before=before,
        usage=total_tokens,
        estimate=estimated_total_tokens,
        hard_limit=effective_trigger_input_tokens,
        soft_limit=effective_soft_trigger_input_tokens,
        usage_source=display_usage_source,
        pct_of_hard=pct_of_hard,
    )


def _display_context_with_estimate(ctx: DisplayTokenContext, estimate: int | None) -> DisplayTokenContext:
    if estimate is None or ctx.estimate is not None:
        return ctx
    return DisplayTokenContext(
        before=ctx.before,
        usage=ctx.usage,
        estimate=estimate,
        hard_limit=ctx.hard_limit,
        soft_limit=ctx.soft_limit,
        usage_source=ctx.usage_source,
        pct_of_hard=ctx.pct_of_hard,
    )


def _round_half_up_int(value: float) -> int:
    return int(math.floor(value + 0.5))


def _format_token_count(value: int, *, approximate: bool) -> str:
    prefix = "≈" if approximate else ""
    return f"{prefix}{value:,}"


def _format_before_token_count(ctx: DisplayTokenContext) -> str:
    if ctx.before is None:
        return "≈unknown"
    is_provider_usage = ctx.usage_source in {"request", "persisted"} and ctx.usage == ctx.before
    return _format_token_count(ctx.before, approximate=not is_provider_usage)


def _format_token_suffix(
    ctx: DisplayTokenContext,
    *,
    after: int | None,
    show_usage_and_estimate: bool,
    summary: int | None = None,
) -> str:
    hard_limit = f"{ctx.hard_limit:,}"
    before_pct = None if ctx.pct_of_hard is None else _round_half_up_int(ctx.pct_of_hard)

    if summary is not None:
        summary_part = f"summary {_format_token_count(summary, approximate=True)} tokens"
        if show_usage_and_estimate and ctx.usage is not None and ctx.estimate is not None:
            estimate_part = f"candidate ≈{ctx.estimate:,}"
            pct_part = f" · {before_pct}%" if before_pct is not None else ""
            return f"(observed usage {ctx.usage:,} · {estimate_part} / {hard_limit}{pct_part} · {summary_part})"

        before = _format_before_token_count(ctx)
        pct_part = f" · {before_pct}%" if before_pct is not None and ctx.hard_limit > 0 else ""
        return f"({before} / {hard_limit} tokens{pct_part} · {summary_part})"

    if show_usage_and_estimate and ctx.usage is not None and ctx.estimate is not None:
        estimate_part = f"candidate ≈{ctx.estimate:,}"
        pct_part = f" · {before_pct}%" if before_pct is not None else ""
        if after is not None:
            estimate_part += f" → {_format_token_count(after, approximate=True)}"
            if ctx.hard_limit > 0:
                pct_part += f" → {_round_half_up_int(after / ctx.hard_limit * 100)}%"
        return f"(observed usage {ctx.usage:,} · {estimate_part} / {hard_limit}{pct_part})"

    before = _format_before_token_count(ctx)
    if after is not None:
        after_count = _format_token_count(after, approximate=True)
        if before_pct is None or ctx.hard_limit <= 0:
            return f"({before} → {after_count} tokens)"
        after_pct = _round_half_up_int(after / ctx.hard_limit * 100)
        return f"({before} → {after_count} tokens · {before_pct}% → {after_pct}%)"

    if before_pct is None:
        return f"({before} / {hard_limit} tokens)"
    return f"({before} / {hard_limit} tokens · {before_pct}%)"


def _tokens_status_payload(
    ctx: DisplayTokenContext,
    *,
    after: int | None,
    show_usage_and_estimate: bool,
    summary: int | None = None,
) -> dict[str, Any]:
    tokens: dict[str, Any] = {
        "before": ctx.before,
        "hard_limit": ctx.hard_limit,
        "soft_limit": ctx.soft_limit,
        "pct_of_hard": ctx.pct_of_hard,
        "usage_source": ctx.usage_source,
    }
    if summary is not None:
        tokens["summary"] = summary
    elif after is not None:
        tokens["after"] = after
    if show_usage_and_estimate and ctx.usage is not None:
        tokens["usage"] = ctx.usage
    if show_usage_and_estimate and ctx.estimate is not None:
        tokens["estimate"] = ctx.estimate
    return tokens


def _description_with_token_suffix(
    description: str,
    ctx: DisplayTokenContext,
    *,
    after: int | None = None,
    show_usage_and_estimate: bool = False,
    summary: int | None = None,
) -> str:
    return f"{description} {_format_token_suffix(ctx, after=after, show_usage_and_estimate=show_usage_and_estimate, summary=summary)}"


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


def _sanitize_summary_text(value: str) -> str:
    # Core skips surrogate cleanup when no NUL is present.
    return sanitize_text_for_db(value).encode("utf-8", errors="ignore").decode("utf-8")


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
            if isinstance(value, str):
                value = _sanitize_summary_text(value).strip()
                if value:
                    return value
        responses_text = _responses_output_text(response)
        if responses_text:
            responses_text = _sanitize_summary_text(responses_text).strip()
            if responses_text:
                return responses_text
    if isinstance(response, StreamingResponse):
        parts: list[str] = []
        saw_tool_call = False
        sse_parser = _SSEDataParser()
        media_type = getattr(response, "media_type", None) or response.headers.get("content-type", "")
        is_sse_response = "text/event-stream" in str(media_type).lower()
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
            text = _sanitize_summary_text("".join(parts)).strip()
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
    "num_predict",
    "stop",
    "response_format",
    "format",
)


def strip_summary_inherited_response_controls(body: dict[str, Any]) -> None:
    def strip(values: Any) -> None:
        if not isinstance(values, dict):
            return
        for key in SUMMARY_INHERITED_RESPONSE_CONTROL_KEYS:
            values.pop(key, None)
        custom_params = values.get("custom_params")
        if isinstance(custom_params, dict):
            for key in SUMMARY_INHERITED_RESPONSE_CONTROL_KEYS:
                custom_params.pop(key, None)

    strip(body)
    strip(body.get("params"))
    strip(body.get("options"))


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


def finalize_summary_tool_policy(body: dict[str, Any], policy: SummaryToolPolicy) -> None:
    if policy == "always_strip":
        strip_summary_tools_for_retry(body)


def summary_ref_registry() -> MappingProxyType:
    return MappingProxyType({})


def classify_core_function_calling_generation(
    openai_router: Any,
    ollama_router: Any,
) -> CoreFunctionCallingGeneration:
    marker_pair = (
        callable(getattr(openai_router, "get_openai_connection", None)),
        callable(getattr(ollama_router, "get_ollama_runtime_config", None)),
    )
    match marker_pair:
        case (True, True):
            return CoreFunctionCallingGeneration.NATIVE_DEFAULT
        case (False, False):
            return CoreFunctionCallingGeneration.NATIVE_OPT_IN
        case (True, False) | (False, True):
            return CoreFunctionCallingGeneration.UNKNOWN
        case unreachable:
            assert_never(unreachable)


def _core_function_calling_generation() -> CoreFunctionCallingGeneration:
    try:
        from open_webui.routers import ollama as ollama_router
        from open_webui.routers import openai as openai_router
    except Exception:  # noqa: BROAD_EXCEPT_OK - any router import failure makes the generation unknowable.
        return CoreFunctionCallingGeneration.UNKNOWN
    return classify_core_function_calling_generation(openai_router, ollama_router)


def core_function_calling_is_native(
    generation: CoreFunctionCallingGeneration,
    function_calling: Any,
) -> bool:
    match generation:
        case CoreFunctionCallingGeneration.NATIVE_DEFAULT:
            return function_calling != "legacy"
        case CoreFunctionCallingGeneration.NATIVE_OPT_IN:
            return function_calling == "native"
        case CoreFunctionCallingGeneration.UNKNOWN:
            return False
        case unreachable:
            assert_never(unreachable)


def resolve_ref_mode_preflight(
    preflight: RefModePreflight,
) -> EffectiveRefMode:
    checks = (
        (preflight.valve_enabled, RefModeReason.VALVE_OFF),
        (preflight.native_function_calling, RefModeReason.NON_NATIVE_CONTEXT),
        (preflight.durable_context, RefModeReason.NON_DURABLE_CONTEXT),
        (preflight.provider_schema_supported, RefModeReason.PROVIDER_SCHEMA_UNSUPPORTED),
        (
            isinstance(preflight.metadata_tools, dict)
            and preflight.metadata_tools is preflight.injected_tools,
            RefModeReason.CORE_REGISTRY_UNAVAILABLE,
        ),
        (
            preflight.registry_available,
            RefModeReason.READER_COLLISION,
        ),
    )
    for passed, reason in checks:
        if not passed:
            return EffectiveRefMode(active=False, reason=reason)
    return EffectiveRefMode(active=True, reason=RefModeReason.ACTIVE)


def build_summary_request_message(
    prefix_file_context: str | None = None,
    *,
    summary_prompt: str | None = None,
) -> dict[str, str]:
    content = resolve_summary_prompt(summary_prompt)
    if prefix_file_context:
        content = f"{prefix_file_context}\n\n{content}"
    return {"role": "user", "content": content}


def build_summary_completion_body(
    base_body: dict[str, Any],
    *,
    summary_model_id: str,
    source_messages: list[dict[str, Any]],
    preserved_system_message: dict[str, Any] | None = None,
    metadata: dict[str, Any],
    pipe_function_id: str = PIPE_FUNCTION_ID,
    summary_tool_policy: SummaryToolPolicy = "fallback_on_tool_call",
    prefix_file_context: str | None = None,
    summary_prompt: str | None = None,
) -> dict[str, Any]:
    body = _copy_body_preserving_metadata(base_body)
    body["model"] = _resolve_summary_model_for_call(summary_model_id, pipe_function_id=pipe_function_id)
    messages = []
    if preserved_system_message is not None:
        messages.append(copy.deepcopy(preserved_system_message))
    messages.extend(copy.deepcopy(source_messages))
    messages.append(build_summary_request_message(prefix_file_context, summary_prompt=summary_prompt))
    body["messages"] = messages
    body["metadata"] = build_summary_task_metadata(metadata)
    body["metadata"].pop("files", None)
    body.pop("previous_response_id", None)
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
    preserved_system_message: dict[str, Any] | None = None,
    base_body: dict[str, Any],
    pipe_function_id: str = PIPE_FUNCTION_ID,
    on_summary_start: Callable[[], Awaitable[None]] | None = None,
    summary_tool_policy: SummaryToolPolicy = "fallback_on_tool_call",
    compaction_prefix_count: int = 0,
    parent_source_message_count: int = 0,
    file_context_enabled: bool = True,
    summary_prompt: str | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    ref_projection_plan: RefProjectionPlan | None = None,
    ref_mode_active: bool | None = None,
) -> str:
    summary_metadata = build_summary_task_metadata(metadata)
    summary_metadata.pop("files", None)
    inner_request = RequestStateProxy(
        request,
        bypass_filter=True,
        bypass_system_prompt=False,
        metadata=summary_metadata,
    )
    from open_webui.utils.chat import generate_chat_completion

    prefix_file_context = await _prepare_summary_file_context(
        request=request,
        user=user,
        metadata=metadata,
        compaction_prefix_count=compaction_prefix_count,
        parent_source_message_count=parent_source_message_count,
        file_context_enabled=file_context_enabled,
        transient_message_patterns=transient_message_patterns,
    )
    active_ref_mode = (
        ref_projection_plan is not None
        if ref_mode_active is None
        else ref_mode_active
    )
    if active_ref_mode:
        _require_provider_bound_history_ref_manifests(
            {"messages": source_messages},
            ref_projection_plan,
        )
    if ref_projection_plan is not None:
        source_messages = await apply_ref_projection_plan(
            source_messages,
            ref_projection_plan,
        )

    body = build_summary_completion_body(
        base_body,
        summary_model_id=summary_model_id,
        source_messages=source_messages,
        preserved_system_message=preserved_system_message,
        metadata=metadata,
        pipe_function_id=pipe_function_id,
        summary_tool_policy=summary_tool_policy,
        prefix_file_context=prefix_file_context,
        summary_prompt=summary_prompt,
    )
    route = await _resolve_core_chat_model_route(
        inner_request,
        str(body.get("model") or ""),
        pipe_function_id=pipe_function_id,
    )
    models = await _model_dict_from_request(inner_request)
    original_model_id = str(body.get("model") or "")
    route, selected_arena_model_id = await _resolve_arena_chat_model_route_with_access(
        request=inner_request,
        user=user,
        models=models,
        route=route,
        original_model_id=original_model_id,
        pipe_function_id=pipe_function_id,
    )
    body["model"] = route.model_id
    if selected_arena_model_id:
        body["metadata"]["selected_model_id"] = selected_arena_model_id
    body = _apply_resolved_model_route_params(
        body,
        models=models,
        route=route,
    )
    if ref_projection_plan is not None:
        apply_ref_projection_surfaces(
            body,
            ref_projection_plan,
            include_reader_schema=False,
        )
    finalize_summary_tool_policy(body, summary_tool_policy)
    retry_body = (
        _copy_body_preserving_metadata(body)
        if summary_tool_policy == "fallback_on_tool_call" and (body.get("tools") or body.get("functions"))
        else None
    )
    await _ensure_model_in_request_models(inner_request, str(body.get("model") or ""))
    if on_summary_start is not None:
        await on_summary_start()
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
        if retry_body is None:
            raise
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
    file_context_enabled: bool = True,
    summary_prompt: str | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    ref_projection_plan: RefProjectionPlan | None = None,
    ref_mode_active: bool | None = None,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> tuple[list[dict[str, Any]], bool, int]:
    cut = select_tool_result_compaction_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    if cut is None:
        return messages, False, 0

    source_messages = copy.deepcopy(cut.summarization_prefix)
    source_identity_count = _source_identity_message_count(
        source_messages,
        transient_message_patterns=transient_message_patterns,
    )
    if source_identity_count <= 0:
        return messages, False, 0
    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")
    if not user_id or not _chat_id_supported(chat_id):
        raise UnsupportedCompactionInput(
            "Cannot compact tool history without a durable checkpoint identity",
            code="checkpoint_identity_missing",
        )
    prefix_file_fingerprint_resolver = await _build_prefix_file_fingerprint_resolver(
        request,
        metadata,
        source_messages,
        require_file_context_chain=file_context_enabled,
        transient_message_patterns=transient_message_patterns,
    )
    file_backed_image_db_chain = _prefix_file_fingerprint_resolver_db_chain(prefix_file_fingerprint_resolver)
    source_identity_fingerprint = _resolve_fingerprint(
        prefix_file_fingerprint_resolver,
        source_identity_count,
    )
    pending_checkpoint = await _lookup_pending_checkpoint_for_source_prefix(
        request=request,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        source_messages=source_messages,
        prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
        transient_message_patterns=transient_message_patterns,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )
    if pending_checkpoint is not None and not _checkpoint_matches_exact_source(
        pending_checkpoint,
        source_messages,
        prefix_file_fingerprint=source_identity_fingerprint,
        file_backed_image_db_chain=file_backed_image_db_chain,
        transient_message_patterns=transient_message_patterns,
    ):
        ready_checkpoint = await _wait_for_pending_checkpoint_ready(pending_checkpoint)
        if ready_checkpoint is not None:
            ready_count = int(ready_checkpoint.get("source_message_count") or 0)
            ready_raw_count = _raw_prefix_len_for_source_count(
                cut.summarization_prefix,
                ready_count,
                transient_message_patterns=transient_message_patterns,
            )
            if ready_raw_count is not None:
                message_cut = MessageCut(
                    preserved_system_message=copy.deepcopy(cut.preserved_system_message),
                    summarization_prefix=copy.deepcopy(cut.summarization_prefix),
                    tail_messages=copy.deepcopy(cut.tail_messages),
                    source_message_count=cut.source_message_count,
                )
                return (
                    replace_prefix_with_parent_checkpoint_and_delta(
                        message_cut,
                        ready_checkpoint,
                        prefix_file_fingerprint=_resolve_fingerprint(
                            prefix_file_fingerprint_resolver,
                            ready_count,
                        ),
                        file_backed_image_db_chain=file_backed_image_db_chain,
                        transient_message_patterns=transient_message_patterns,
                        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                        historical_message_excerpt_count=historical_message_excerpt_count,
                    ),
                    True,
                    ready_count,
                )
    summary_meta = build_checkpoint_summary_meta(
        source_messages,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        transient_message_patterns=transient_message_patterns,
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
            preserved_system_message=cut.preserved_system_message,
            summary_meta=summary_meta,
            parent_checkpoint=parent_checkpoint,
            summary_tool_policy=summary_tool_policy,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            file_context_enabled=file_context_enabled,
            summary_prompt=summary_prompt,
            transient_message_patterns=transient_message_patterns,
            ref_projection_plan=ref_projection_plan,
            ref_mode_active=ref_mode_active,
            checkpoint_profile_hash=checkpoint_profile_hash,
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
            transient_message_patterns=transient_message_patterns,
        )
    )
    compacted.extend(copy.deepcopy(cut.tail_messages))
    return compacted, True, source_identity_count


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


async def _await_checkpoint_db_before_deadline(
    awaitable: Awaitable[Any], deadline: float
) -> Any:
    remaining = max(0.0, deadline - time.monotonic())
    return await asyncio.wait_for(awaitable, timeout=remaining)


def _checkpoint_wait_timeout_error(subject: str) -> RuntimeError:
    return RuntimeError(f"Timed out waiting for {subject}")


async def _claim_or_wait_for_generation_lease(
    store: Any,
    *,
    identity: dict[str, str],
    claim_token: str,
) -> tuple[str, int]:
    deadline = time.monotonic() + CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS
    timeout_error = _checkpoint_wait_timeout_error(
        "another worker to release the checkpoint generation lease"
    )
    try:
        while True:
            row = await _await_checkpoint_db_before_deadline(
                store.lookup_any(**identity),
                deadline,
            )
            now = int(time.time())
            if row is None:
                expires_at = now + CHECKPOINT_CLAIM_LEASE_SECONDS
                pending_row = build_checkpoint_row(
                    **identity,
                    source_message_count=0,
                    summary_text="",
                    summary_meta={},
                    parent_checkpoint_id=None,
                    state="pending",
                    claim_token=claim_token,
                    claim_expires_at=expires_at,
                    now=now,
                )
                claimed = await _await_checkpoint_db_before_deadline(
                    store.claim_pending(pending_row),
                    deadline,
                )
                if claimed:
                    if time.time() < expires_at:
                        return pending_row["id"], expires_at
                    continue
                continue
            expires_at = row.get("claim_expires_at")
            if expires_at is None or int(expires_at) <= now:
                reclaimed_expires_at = now + CHECKPOINT_CLAIM_LEASE_SECONDS
                reclaimed = await _await_checkpoint_db_before_deadline(
                    store.reclaim_pending(
                        row["id"],
                        claim_token=claim_token,
                        expires_at=reclaimed_expires_at,
                        now=now,
                    ),
                    deadline,
                )
                if reclaimed:
                    if time.time() < reclaimed_expires_at:
                        return row["id"], reclaimed_expires_at
                    continue
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise timeout_error
            await asyncio.sleep(min(CHECKPOINT_PENDING_POLL_SECONDS, remaining))
    except asyncio.TimeoutError as exc:
        raise timeout_error from exc


async def _heartbeat_generation_lease(
    store: Any,
    lease_id: str,
    claim_token: str,
    claim_expires_at: int,
    lease_lost: asyncio.Event,
) -> None:
    lease_deadline = time.monotonic() + max(
        0.0,
        claim_expires_at - time.time(),
    )
    try:
        while True:
            remaining = lease_deadline - time.monotonic()
            if remaining <= 0:
                lease_lost.set()
                return
            heartbeat_delay = (
                CHECKPOINT_CLAIM_HEARTBEAT_SECONDS
                if remaining > CHECKPOINT_CLAIM_HEARTBEAT_SECONDS
                else 0.0
            )
            await asyncio.sleep(min(heartbeat_delay, remaining))

            while True:
                remaining = lease_deadline - time.monotonic()
                if remaining <= 0:
                    lease_lost.set()
                    return
                expires_at = int(time.time()) + CHECKPOINT_CLAIM_LEASE_SECONDS
                try:
                    extended = await asyncio.wait_for(
                        store.extend_claim(
                            lease_id,
                            claim_token=claim_token,
                            expires_at=expires_at,
                        ),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    lease_lost.set()
                    return
                except Exception:
                    remaining = lease_deadline - time.monotonic()
                    if remaining <= 0:
                        lease_lost.set()
                        return
                    await asyncio.sleep(min(CHECKPOINT_PENDING_POLL_SECONDS, remaining))
                    continue
                if not extended:
                    lease_lost.set()
                    return
                lease_deadline = time.monotonic() + max(0.0, expires_at - time.time())
                break
    except asyncio.CancelledError:
        raise
    except Exception:
        if not lease_lost.is_set():
            lease_lost.set()


async def _run_summary_factory_with_generation_lease(
    summary_factory: Callable[[dict[str, Any] | None], Awaitable[str]],
    parent: dict[str, Any] | None,
    lease_lost: asyncio.Event,
) -> str:
    if lease_lost.is_set():
        raise RuntimeError(
            "Checkpoint generation lease was lost before summary generation"
        )
    summary_task = asyncio.create_task(summary_factory(parent))
    lease_loss_task = asyncio.create_task(lease_lost.wait())
    try:
        await asyncio.wait(
            {summary_task, lease_loss_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if lease_lost.is_set():
            summary_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await summary_task
            raise RuntimeError("Checkpoint generation lease was lost during summary generation")
        return await summary_task
    finally:
        for task in (summary_task, lease_loss_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(summary_task, lease_loss_task, return_exceptions=True)


async def _claim_or_wait_for_checkpoint(
    store: Any,
    *,
    identity: dict[str, str],
    source_message_count: int,
    summary_meta: dict[str, Any],
    claim_token: str,
) -> dict[str, Any]:
    """Return the claimed pending row, or a ready row produced by another worker."""
    deadline = time.monotonic() + CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS
    timeout_error = _checkpoint_wait_timeout_error(
        "another worker to finish generating this checkpoint summary"
    )
    try:
        while True:
            row = await _await_checkpoint_db_before_deadline(
                store.lookup_any(**identity),
                deadline,
            )
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
                claimed = await _await_checkpoint_db_before_deadline(
                    store.claim_pending(pending_row),
                    deadline,
                )
                if claimed:
                    return pending_row
                continue
            if row.get("state") == "ready":
                try:
                    await _await_checkpoint_db_before_deadline(
                        store.touch(row["id"]),
                        deadline,
                    )
                except asyncio.TimeoutError:
                    raise
                except Exception:
                    pass
                return row
            now = int(time.time())
            expires_at = row.get("claim_expires_at")
            if expires_at is None or int(expires_at) <= now:
                reclaimed = await _await_checkpoint_db_before_deadline(
                    store.reclaim_pending(
                        row["id"],
                        claim_token=claim_token,
                        expires_at=now + CHECKPOINT_CLAIM_LEASE_SECONDS,
                        now=now,
                    ),
                    deadline,
                )
                if reclaimed:
                    return row
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise timeout_error
            await asyncio.sleep(min(CHECKPOINT_PENDING_POLL_SECONDS, remaining))
    except asyncio.TimeoutError as exc:
        raise timeout_error from exc


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
    parent_checkpoint_guard: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    prefix_file_fingerprint: str | None = None,
    prefix_file_fingerprint_resolver: Callable[[int], str | None] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    use_generation_lease: bool = True,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> str:
    profile_hash = checkpoint_profile_hash
    file_backed_image_db_chain = _prefix_file_fingerprint_resolver_db_chain(prefix_file_fingerprint_resolver)
    source_hash = compute_summary_source_hash(
        source_messages,
        prefix_file_fingerprint,
        file_backed_image_db_chain,
        transient_message_patterns=transient_message_patterns,
    )
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
            existing = await store.lookup_ready(**identity)
            if existing is not None:
                with suppress(Exception):
                    await store.touch(existing["id"])
                return CompactionSummaryResult(existing["summary_text"], checkpoint=existing)

            claim_token = uuid.uuid4().hex
            lease_id: str | None = None
            lease_claim_token: str | None = None
            lease_lost = asyncio.Event()
            generation_heartbeat: asyncio.Task[Any] | None = None
            if use_generation_lease:
                lease_claim_token = uuid.uuid4().hex
                lease_identity = {
                    "namespace": CHECKPOINT_GENERATION_LEASE_NAMESPACE,
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "pipe_function_id": pipe_function_id,
                    "profile_hash": profile_hash,
                    "source_hash": CHECKPOINT_GENERATION_LEASE_SOURCE_HASH,
                }
                lease_id, lease_expires_at = await _claim_or_wait_for_generation_lease(
                    store,
                    identity=lease_identity,
                    claim_token=lease_claim_token,
                )
                generation_heartbeat = asyncio.create_task(
                    _heartbeat_generation_lease(
                        store,
                        lease_id,
                        lease_claim_token,
                        lease_expires_at,
                        lease_lost,
                    )
                )
            try:
                if lease_lost.is_set():
                    raise RuntimeError("Checkpoint generation lease was lost before source claim")
                claimed = await _claim_or_wait_for_checkpoint(
                    store,
                    identity=identity,
                    source_message_count=_source_identity_message_count(
                        source_messages,
                        transient_message_patterns=transient_message_patterns,
                    ),
                    summary_meta=summary_meta,
                    claim_token=claim_token,
                )
                if claimed.get("state") == "ready":
                    return CompactionSummaryResult(claimed["summary_text"], checkpoint=claimed)
                checkpoint_id = str(claimed["id"])
                checkpoint_summary_meta = normalize_summary_meta(claimed.get("summary_meta"))

                heartbeat = asyncio.create_task(_heartbeat_checkpoint_claim(store, checkpoint_id, claim_token))
                release_source_claim = False
                try:
                    if parent_checkpoint is not None:
                        parent_count = int(parent_checkpoint.get("source_message_count") or 0)
                        raw_parent_count = _raw_prefix_len_for_source_count(
                            source_messages,
                            parent_count,
                            transient_message_patterns=transient_message_patterns,
                        )
                        parent_fingerprint = (
                            prefix_file_fingerprint_resolver(parent_count)
                            if prefix_file_fingerprint_resolver is not None
                            else None
                        )
                        if (
                            parent_count <= 0
                            or raw_parent_count is None
                            or compute_summary_source_hash(
                                source_messages[:raw_parent_count],
                                parent_fingerprint,
                                file_backed_image_db_chain,
                                transient_message_patterns=transient_message_patterns,
                            )
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
                            prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
                            transient_message_patterns=transient_message_patterns,
                        )
                    if lease_lost.is_set():
                        raise RuntimeError("Checkpoint generation lease was lost before summary generation")
                    if parent is not None and parent_checkpoint_guard is not None:
                        await parent_checkpoint_guard(parent)
                    if lease_lost.is_set():
                        raise RuntimeError("Checkpoint generation lease was lost before summary generation")
                    try:
                        summary_text = (
                            await _run_summary_factory_with_generation_lease(
                                summary_factory,
                                parent,
                                lease_lost,
                            )
                            if use_generation_lease
                            else await summary_factory(parent)
                        )
                    except (
                        SummaryFileContextUnavailable,
                        RefProjectionError,
                        HistoryRefStorageUnavailableError,
                    ):
                        raise
                    except Exception as exc:
                        if parent:
                            raise ParentCheckpointExtensionFailed(parent, exc) from exc
                        raise
                    if lease_lost.is_set():
                        raise RuntimeError("Checkpoint generation lease was lost before summary storage")
                    summary_token_count = await _estimate_rendered_summary_message_tokens(
                        request=request,
                        summary_text=summary_text,
                        summary_meta=checkpoint_summary_meta,
                        transient_message_patterns=transient_message_patterns,
                    )
                    if lease_lost.is_set():
                        raise RuntimeError("Checkpoint generation lease was lost before summary storage")
                    completed = await store.complete_pending(
                        checkpoint_id,
                        claim_token=claim_token,
                        summary_text=summary_text,
                        parent_checkpoint_id=parent.get("id") if parent else None,
                        summary_token_count=summary_token_count,
                        generation_lease_id=lease_id,
                        generation_lease_claim_token=lease_claim_token,
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
                    release_source_claim = True
                    raise
                finally:
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat
                    if release_source_claim:
                        with suppress(asyncio.CancelledError, Exception):
                            await asyncio.shield(
                                store.release_claim(
                                    checkpoint_id, claim_token=claim_token
                                )
                            )
            finally:
                if generation_heartbeat is not None:
                    generation_heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await generation_heartbeat
                if lease_id is not None and lease_claim_token is not None:
                    with suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(
                            store.release_claim(lease_id, claim_token=lease_claim_token)
                        )
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
    preserved_system_message: dict[str, Any] | None = None,
    summary_meta: dict[str, Any],
    parent_checkpoint: dict[str, Any] | None = None,
    parent_checkpoint_guard: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    on_summary_start: Callable[[], Awaitable[None]] | None = None,
    summary_tool_policy: SummaryToolPolicy = "fallback_on_tool_call",
    historical_message_excerpt_bytes: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
    historical_message_excerpt_count: int = DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
    file_context_enabled: bool = True,
    summary_prompt: str | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    use_generation_lease: bool = False,
    ref_projection_plan: RefProjectionPlan | None = None,
    ref_mode_active: bool | None = None,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> str:
    summary_source_prefix = copy.deepcopy(source_messages)
    active_ref_mode = (
        ref_projection_plan is not None
        if ref_mode_active is None
        else ref_mode_active
    )
    prefix_file_fingerprint_resolver = await _build_prefix_file_fingerprint_resolver(
        request,
        metadata,
        source_messages,
        require_file_context_chain=file_context_enabled,
        transient_message_patterns=transient_message_patterns,
    )

    async def summary_factory(parent: dict[str, Any] | None) -> str:
        parent_count = 0
        raw_parent_count = 0
        if parent:
            parent_count = int(parent.get("source_message_count") or 0)
            raw_count = _raw_prefix_len_for_source_count(
                summary_source_prefix,
                parent_count,
                transient_message_patterns=transient_message_patterns,
            )
            if raw_count is None:
                raise UnsupportedCompactionInput(
                    "Parent checkpoint cannot be applied safely because its source boundary is invalid",
                    code="unsafe_checkpoint_parent",
                )
            raw_parent_count = raw_count
            source = [
                render_summary_message_from_checkpoint(
                    parent,
                    historical_source_messages=summary_source_prefix[:raw_parent_count],
                    historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                    historical_message_excerpt_count=historical_message_excerpt_count,
                    transient_message_patterns=transient_message_patterns,
                ),
                *copy.deepcopy(summary_source_prefix[raw_parent_count:]),
            ]
        else:
            source = copy.deepcopy(summary_source_prefix)
        verified_projection_plan = build_summary_ref_projection_plan(
            ref_projection_plan,
            None,
            ref_mode_active=active_ref_mode,
        )
        if active_ref_mode and parent is not None:
            logical_snapshot = await get_or_build_logical_history_snapshot(
                request,
                summary_source_prefix,
                identity=(
                    CHECKPOINT_NAMESPACE,
                    user_id,
                    chat_id,
                    pipe_function_id,
                    checkpoint_profile_hash,
                ),
                prefix_file_fingerprint_resolver=(
                    prefix_file_fingerprint_resolver
                ),
                transient_message_patterns=transient_message_patterns,
            )
            verified_projection_plan = await extend_ref_projection_plan_with_checkpoint(
                verified_projection_plan,
                parent,
                request=request,
                metadata=metadata,
                transient_message_patterns=transient_message_patterns,
                logical_snapshot=logical_snapshot,
            )
        summary_ref_projection_plan = build_summary_ref_projection_plan(
            verified_projection_plan,
            parent,
            ref_mode_active=active_ref_mode,
        )
        return await _generate_summary_text(
            request=request,
            user=user,
            metadata=metadata,
            summary_model_id=summary_model_id,
            source_messages=source,
            preserved_system_message=preserved_system_message,
            base_body=base_body,
            pipe_function_id=pipe_function_id,
            on_summary_start=on_summary_start,
            summary_tool_policy=summary_tool_policy,
            compaction_prefix_count=_source_identity_message_count(
                summary_source_prefix,
                transient_message_patterns=transient_message_patterns,
            ),
            parent_source_message_count=parent_count,
            file_context_enabled=file_context_enabled,
            summary_prompt=summary_prompt,
            transient_message_patterns=transient_message_patterns,
            ref_projection_plan=summary_ref_projection_plan,
            ref_mode_active=active_ref_mode,
        )

    identity_fingerprint = (
        prefix_file_fingerprint_resolver(
            _source_identity_message_count(
                source_messages,
                transient_message_patterns=transient_message_patterns,
            )
        )
        if prefix_file_fingerprint_resolver is not None
        else None
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
        parent_checkpoint_guard=parent_checkpoint_guard,
        prefix_file_fingerprint=identity_fingerprint,
        prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
        transient_message_patterns=transient_message_patterns,
        use_generation_lease=use_generation_lease,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )


async def _lookup_ready_checkpoint_for_source(
    *,
    request: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    source_messages: list[dict[str, Any]],
    prefix_file_fingerprint: str | None = None,
    file_backed_image_db_chain: list[dict[str, Any]] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> dict[str, Any] | None:
    await ensure_checkpoint_table_initialized(request=request)
    return await CheckpointStore().lookup_ready(
        namespace=CHECKPOINT_NAMESPACE,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        profile_hash=checkpoint_profile_hash,
        source_hash=compute_summary_source_hash(
            source_messages,
            prefix_file_fingerprint,
            file_backed_image_db_chain,
            transient_message_patterns=transient_message_patterns,
        ),
    )


async def _lookup_pending_checkpoint_for_source_prefix(
    *,
    request: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    source_messages: list[dict[str, Any]],
    prefix_file_fingerprint_resolver: Callable[[int], str | None] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> dict[str, Any] | None:
    if not source_messages:
        return None
    await ensure_checkpoint_table_initialized(request=request)
    store = CheckpointStore()
    find_pending = getattr(store, "find_longest_pending_parent", None)
    if not callable(find_pending):
        return None
    return await find_pending(
        namespace=CHECKPOINT_NAMESPACE,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        profile_hash=checkpoint_profile_hash,
        source_messages=source_messages,
        prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
        transient_message_patterns=transient_message_patterns,
    )


def _checkpoint_identity_from_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "namespace": str(row.get("namespace") or CHECKPOINT_NAMESPACE),
        "user_id": str(row.get("user_id") or ""),
        "chat_id": str(row.get("chat_id") or ""),
        "pipe_function_id": str(row.get("pipe_function_id") or ""),
        "profile_hash": str(row.get("profile_hash") or ""),
        "source_hash": str(row.get("source_hash") or ""),
    }


def _checkpoint_matches_exact_source(
    row: dict[str, Any],
    source_messages: list[dict[str, Any]],
    *,
    prefix_file_fingerprint: str | None = None,
    file_backed_image_db_chain: list[dict[str, Any]] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> bool:
    try:
        source_message_count = int(row.get("source_message_count") or 0)
    except Exception:
        return False
    return (
        source_message_count
        == _source_identity_message_count(
            source_messages,
            transient_message_patterns=transient_message_patterns,
        )
        and row.get("source_hash")
        == compute_summary_source_hash(
            source_messages,
            prefix_file_fingerprint,
            file_backed_image_db_chain,
            transient_message_patterns=transient_message_patterns,
        )
    )


async def _wait_for_pending_checkpoint_ready(row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    if row.get("state") == "ready":
        return row
    identity = _checkpoint_identity_from_row(row)
    if not all(identity.values()):
        return None
    store = CheckpointStore()
    deadline = time.monotonic() + CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS
    while True:
        try:
            current = await _await_checkpoint_db_before_deadline(
                store.lookup_any(**identity),
                deadline,
            )
        except asyncio.TimeoutError:
            return None
        if current is None:
            return None
        if current.get("state") == "ready":
            try:
                await _await_checkpoint_db_before_deadline(
                    store.touch(str(current["id"])),
                    deadline,
                )
            except asyncio.TimeoutError:
                return None
            except Exception:
                pass
            return current
        if current.get("state") != "pending":
            return None
        expires_at = current.get("claim_expires_at")
        now = int(time.time())
        if expires_at is None or int(expires_at) <= now:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(CHECKPOINT_PENDING_POLL_SECONDS, remaining))


def _soft_prefetch_source_messages(
    body: dict[str, Any],
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None] | None:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    tool_cut = select_tool_result_compaction_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    if tool_cut is not None and tool_cut.summarization_prefix:
        return copy.deepcopy(tool_cut.summarization_prefix), copy.deepcopy(tool_cut.preserved_system_message)
    cut = select_safe_message_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    if cut.summarization_prefix:
        return copy.deepcopy(cut.summarization_prefix), copy.deepcopy(cut.preserved_system_message)
    return None


async def _prefetch_compaction_checkpoint(
    *,
    request: Any,
    user: Any,
    user_id: str,
    chat_id: str,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    summary_model_id: str,
    source_messages: list[dict[str, Any]],
    preserved_system_message: dict[str, Any] | None = None,
    summary_tool_policy: SummaryToolPolicy,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
    effective_trigger_input_tokens: int = 100000,
    effective_soft_trigger_input_tokens: int | None = None,
    trigger_observed_tokens: int | None = None,
    trigger_estimated_tokens: int | None = None,
    trigger_usage_source: str | None = None,
    token_status_detail: Literal["before", "before_after"] = "before",
    token_status_show_usage_and_estimate: bool = False,
    event_emitter: Callable[[Any], Awaitable[None]] | None = None,
    file_context_enabled: bool = True,
    task_estimate_body: dict[str, Any] | None = None,
    summary_prompt: str | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    token_system_prompt: str | None = None,
    dropped_message_keys: frozenset[str] = frozenset(),
    ref_mode_active: bool = False,
    ref_substitution_threshold_tokens: int = 10_000,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> bool:
    if not source_messages:
        return False
    source_identity_count = _source_identity_message_count(
        source_messages,
        transient_message_patterns=transient_message_patterns,
    )
    if source_identity_count <= 0:
        return False
    if not user_id or not _chat_id_supported(chat_id):
        return False
    prefix_file_fingerprint_resolver = await _build_prefix_file_fingerprint_resolver(
        request,
        metadata,
        source_messages,
        transient_message_patterns=transient_message_patterns,
    )
    pending_checkpoint = await _lookup_pending_checkpoint_for_source_prefix(
        request=request,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        source_messages=source_messages,
        prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
        transient_message_patterns=transient_message_patterns,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )
    if pending_checkpoint is not None:
        ready_checkpoint = await _wait_for_pending_checkpoint_ready(pending_checkpoint)
        if ready_checkpoint is None:
            return False
    summary_meta = build_checkpoint_summary_meta(
        source_messages,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        transient_message_patterns=transient_message_patterns,
    )
    summary_projection_plan = (
        await project_native_tool_texts(
            copy.deepcopy(source_messages),
            threshold_tokens=ref_substitution_threshold_tokens,
            request=request,
        )
        if ref_mode_active
        else None
    )

    source_kind = "message"
    messages = body.get("messages")
    if isinstance(messages, list):
        tool_cut = select_tool_result_compaction_cut(
            messages,
            transient_message_patterns=transient_message_patterns,
        )
        if tool_cut is not None and tool_cut.summarization_prefix == source_messages:
            source_kind = "tool"

    def reusable_match_covers_prefetch_source(match: ReusableCheckpointMatch) -> bool:
        file_backed_image_db_chain = _prefix_file_fingerprint_resolver_db_chain(prefix_file_fingerprint_resolver)
        if match.checkpoint is None or match.source_kind != source_kind:
            return False
        checkpoint_count = int(match.checkpoint.get("source_message_count") or match.source_message_count or 0)
        source_count = source_identity_count
        raw_checkpoint_count = _raw_prefix_len_for_source_count(
            source_messages,
            checkpoint_count,
            transient_message_patterns=transient_message_patterns,
        )
        if checkpoint_count <= 0 or checkpoint_count > source_count or raw_checkpoint_count is None:
            return False
        if match.kind == "exact" and checkpoint_count != source_count:
            return False
        checkpoint_fingerprint = (
            prefix_file_fingerprint_resolver(checkpoint_count)
            if prefix_file_fingerprint_resolver is not None
            else None
        )
        return (
            compute_summary_source_hash(
                source_messages[:raw_checkpoint_count],
                checkpoint_fingerprint,
                file_backed_image_db_chain,
                transient_message_patterns=transient_message_patterns,
            )
            == match.checkpoint.get("source_hash")
        )

    task_metadata_body = _task_body_from_metadata(metadata)

    async def estimate_prefetch_checkpoint_applied_tokens(match: ReusableCheckpointMatch) -> int | None:
        if task_estimate_body is not None:
            return await _estimate_task_checkpoint_applied_body_tokens(
                request=request,
                user=user,
                metadata=metadata,
                body=task_estimate_body,
                pipe_function_id=pipe_function_id,
                match=match,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
                file_context_enabled=file_context_enabled,
                transient_message_patterns=transient_message_patterns,
                token_system_prompt=token_system_prompt,
                dropped_message_keys=dropped_message_keys,
                checkpoint_profile_hash=checkpoint_profile_hash,
            )
        if task_metadata_body is not None:
            return None
        return await _estimate_checkpoint_applied_body_tokens(
            request=request,
            user=user,
            metadata=metadata,
            body=body,
            pipe_function_id=pipe_function_id,
            match=match,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            file_context_enabled=file_context_enabled,
            transient_message_patterns=transient_message_patterns,
            token_system_prompt=token_system_prompt,
            dropped_message_keys=dropped_message_keys,
            checkpoint_profile_hash=checkpoint_profile_hash,
        )

    reusable_checkpoint_match = await _body_reusable_checkpoint_match(
        request=request,
        user=user,
        metadata=metadata,
        body=body,
        pipe_function_id=pipe_function_id,
        transient_message_patterns=transient_message_patterns,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )
    if reusable_checkpoint_match is not None and reusable_match_covers_prefetch_source(reusable_checkpoint_match):
        if reusable_checkpoint_match.kind == "exact":
            return False
        checkpoint_applied_estimate = await estimate_prefetch_checkpoint_applied_tokens(reusable_checkpoint_match)
        if checkpoint_applied_estimate is None:
            LOG.error(
                "auto-compaction prefetch: checkpoint-applied token estimate was unavailable; "
                "skipping background checkpoint generation (user_id=%s chat_id=%s source_kind=%s match_kind=%s)",
                user_id,
                chat_id,
                source_kind,
                reusable_checkpoint_match.kind,
            )
            return False
        if (
            effective_soft_trigger_input_tokens is not None
            and checkpoint_applied_estimate < effective_soft_trigger_input_tokens
        ):
            return False

    summary_started = False
    prefetch_display_context: DisplayTokenContext | None = None
    prefetch_display_context_ready = False

    async def skip_child_generation_if_parent_below_soft(parent: dict[str, Any]) -> None:
        if effective_soft_trigger_input_tokens is None:
            return
        parent_count = int(parent.get("source_message_count") or 0)
        checkpoint_applied_estimate = await estimate_prefetch_checkpoint_applied_tokens(
            ReusableCheckpointMatch(
                kind="parent",
                source_message_count=parent_count,
                source_kind=source_kind,
                checkpoint=parent,
            )
        )
        if checkpoint_applied_estimate is None:
            LOG.error(
                "auto-compaction prefetch: parent checkpoint-applied token estimate was unavailable; "
                "skipping background checkpoint generation (user_id=%s chat_id=%s source_kind=%s parent_source_message_count=%s)",
                user_id,
                chat_id,
                source_kind,
                parent_count,
            )
            raise _CheckpointGenerationSkipped()
        if checkpoint_applied_estimate < effective_soft_trigger_input_tokens:
            raise _CheckpointGenerationSkipped()

    async def display_token_context() -> DisplayTokenContext:
        nonlocal prefetch_display_context, prefetch_display_context_ready
        if not prefetch_display_context_ready:
            estimated = trigger_estimated_tokens
            total = trigger_observed_tokens
            usage_src = trigger_usage_source
            if estimated is None and total is None:
                with suppress(Exception):
                    LOG.debug(
                        "auto-compaction prefetch: no trigger token value available, "
                        "falling back to fresh body estimate (user_id=%s chat_id=%s)",
                        user_id,
                        chat_id,
                    )
                    estimated = await _estimate_provider_input_tokens_async(
                        body,
                        request=request,
                        user=user,
                        system_prompt=token_system_prompt,
                        dropped_message_keys=dropped_message_keys,
                    )
                usage_src = "estimate" if estimated is not None else None
            prefetch_display_context = _build_display_token_context(
                estimated_total_tokens=estimated,
                total_tokens=total,
                effective_trigger_input_tokens=effective_trigger_input_tokens,
                effective_soft_trigger_input_tokens=effective_soft_trigger_input_tokens,
                usage_source=usage_src,
            )
            prefetch_display_context_ready = True
        assert prefetch_display_context is not None
        return prefetch_display_context

    async def emit_summary_start() -> None:
        nonlocal summary_started
        summary_started = True
        if event_emitter is None:
            return
        token_context = await display_token_context()
        await emit_compaction_status(
            event_emitter,
            action="prefetching",
            description=_description_with_token_suffix(
                "Creating an auto-compaction checkpoint in the background",
                token_context,
            ),
            done=False,
            tokens=_tokens_status_payload(token_context, after=None, show_usage_and_estimate=False),
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
            base_body=body,
            source_messages=source_messages,
            preserved_system_message=preserved_system_message,
            summary_meta=summary_meta,
            parent_checkpoint_guard=skip_child_generation_if_parent_below_soft,
            on_summary_start=emit_summary_start,
            summary_tool_policy=summary_tool_policy,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            file_context_enabled=file_context_enabled,
            summary_prompt=summary_prompt,
            transient_message_patterns=transient_message_patterns,
            use_generation_lease=True,
            ref_projection_plan=summary_projection_plan,
            ref_mode_active=ref_mode_active,
            checkpoint_profile_hash=checkpoint_profile_hash,
        )
    except _CheckpointGenerationSkipped:
        return False
    except asyncio.CancelledError:
        if summary_started and event_emitter is not None:
            token_context = await display_token_context()
            await emit_compaction_status(
                event_emitter,
                action="failed",
                description=_description_with_token_suffix(
                    "Background auto-compaction checkpoint creation was cancelled",
                    token_context,
                ),
                done=True,
                error=True,
                tokens=_tokens_status_payload(token_context, after=None, show_usage_and_estimate=False),
            )
        raise
    except Exception as exc:
        if summary_started and event_emitter is not None:
            token_context = await display_token_context()
            await emit_compaction_status(
                event_emitter,
                action="failed",
                description=_description_with_token_suffix(
                    f"Failed to create background auto-compaction checkpoint: {exc}",
                    token_context,
                ),
                done=True,
                error=True,
                tokens=_tokens_status_payload(token_context, after=None, show_usage_and_estimate=False),
            )
        raise
    if summary_started and event_emitter is not None:
        token_context = await display_token_context()
        summary_tokens = None
        if token_status_detail == "before_after":
            checkpoint = getattr(summary, "checkpoint", None)
            persisted = checkpoint.get("summary_token_count") if isinstance(checkpoint, dict) else None
            if isinstance(persisted, int):
                summary_tokens = persisted
            if summary_tokens is None:
                with suppress(Exception):
                    summary_tokens = await _estimate_rendered_summary_message_tokens(
                        request=request,
                        summary_text=str(summary),
                        summary_meta=summary_meta,
                        historical_source_messages=source_messages,
                        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                        historical_message_excerpt_count=historical_message_excerpt_count,
                        transient_message_patterns=transient_message_patterns,
                    )
        await emit_compaction_status(
            event_emitter,
            action="prefetched",
            description=_description_with_token_suffix(
                "Auto-compaction checkpoint is ready",
                token_context,
                summary=summary_tokens,
                show_usage_and_estimate=token_status_show_usage_and_estimate,
            ),
            done=True,
            tokens=_tokens_status_payload(
                token_context,
                after=None,
                show_usage_and_estimate=token_status_show_usage_and_estimate,
                summary=summary_tokens,
            ),
        )
        await emit_compaction_summary_embed(
            event_emitter,
            summary_text=str(summary),
        )
    return True


@dataclass(frozen=True, slots=True)
class _PreparedSoftPrefetch:
    key: tuple[str, str, str, str, str, str]
    run: Callable[[asyncio.Task[Any] | None], Awaitable[bool]]


def _prepare_soft_compaction_prefetch(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    summary_model_id: str,
    summary_tool_policy: SummaryToolPolicy,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
    effective_trigger_input_tokens: int = 100000,
    effective_soft_trigger_input_tokens: int | None = None,
    trigger_observed_tokens: int | None = None,
    trigger_estimated_tokens: int | None = None,
    trigger_usage_source: str | None = None,
    token_status_detail: Literal["before", "before_after"] = "before",
    token_status_show_usage_and_estimate: bool = False,
    event_emitter: Callable[[Any], Awaitable[None]] | None = None,
    file_context_enabled: bool = True,
    task_estimate_body: dict[str, Any] | None = None,
    summary_prompt: str | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    token_system_prompt: str | None = None,
    dropped_message_keys: frozenset[str] = frozenset(),
    ref_mode_active: bool = False,
    ref_substitution_threshold_tokens: int = 10_000,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> _PreparedSoftPrefetch | None:
    prefetch_source = _soft_prefetch_source_messages(
        body,
        transient_message_patterns=transient_message_patterns,
    )
    if prefetch_source is None:
        return None
    source_messages, preserved_system_message = prefetch_source
    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user or {}).get("id") or "")
    if not user_id or not _chat_id_supported(chat_id):
        return None
    key = _soft_prefetch_inflight_key_for_body(
        user=user,
        metadata=metadata,
        body=body,
        pipe_function_id=pipe_function_id,
        transient_message_patterns=transient_message_patterns,
        source_messages=source_messages,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )
    if key is None:
        return None
    prefetch_user = copy.deepcopy(user)
    prefetch_metadata = _copy_metadata_preserving_references(metadata)
    prefetch_metadata.pop("tools", None)
    prefetch_request = RequestStateProxy(request, metadata=prefetch_metadata)
    if hasattr(prefetch_request.state, REQUEST_STATE_REF_STORE_KEY):
        delattr(prefetch_request.state, REQUEST_STATE_REF_STORE_KEY)
    prefetch_body = _copy_body_preserving_metadata(body)
    prefetch_body_metadata = prefetch_body.get("metadata")
    if isinstance(prefetch_body_metadata, dict):
        prefetch_body_metadata.pop("tools", None)
    prefetch_task_estimate_body = (
        _copy_body_preserving_metadata(task_estimate_body) if task_estimate_body is not None else None
    )
    if prefetch_task_estimate_body is not None:
        prefetch_task_metadata = prefetch_task_estimate_body.get("metadata")
        if isinstance(prefetch_task_metadata, dict):
            prefetch_task_metadata.pop("tools", None)

    async def run_prefetch(parent_prefetch_task: asyncio.Task[Any] | None) -> bool:
        if parent_prefetch_task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(parent_prefetch_task),
                    CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                return False
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        try:
            return await _prefetch_compaction_checkpoint(
                request=prefetch_request,
                user=prefetch_user,
                user_id=user_id,
                chat_id=chat_id,
                metadata=prefetch_metadata,
                body=prefetch_body,
                pipe_function_id=pipe_function_id,
                summary_model_id=summary_model_id,
                source_messages=source_messages,
                preserved_system_message=preserved_system_message,
                summary_tool_policy=summary_tool_policy,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
                effective_trigger_input_tokens=effective_trigger_input_tokens,
                effective_soft_trigger_input_tokens=effective_soft_trigger_input_tokens,
                trigger_observed_tokens=trigger_observed_tokens,
                trigger_estimated_tokens=trigger_estimated_tokens,
                trigger_usage_source=trigger_usage_source,
                token_status_detail=token_status_detail,
                token_status_show_usage_and_estimate=(
                    token_status_show_usage_and_estimate
                ),
                event_emitter=event_emitter,
                file_context_enabled=file_context_enabled,
                task_estimate_body=prefetch_task_estimate_body,
                summary_prompt=summary_prompt,
                transient_message_patterns=transient_message_patterns,
                token_system_prompt=token_system_prompt,
                dropped_message_keys=dropped_message_keys,
                ref_mode_active=ref_mode_active,
                ref_substitution_threshold_tokens=(
                    ref_substitution_threshold_tokens
                ),
                checkpoint_profile_hash=checkpoint_profile_hash,
            )
        except RefProjectionError as exc:
            _log_ref_projection_failure(exc)
            return False

    return _PreparedSoftPrefetch(key=key, run=run_prefetch)


def _start_soft_compaction_prefetch(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    summary_model_id: str,
    summary_tool_policy: SummaryToolPolicy,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
    effective_trigger_input_tokens: int = 100000,
    effective_soft_trigger_input_tokens: int | None = None,
    trigger_observed_tokens: int | None = None,
    trigger_estimated_tokens: int | None = None,
    trigger_usage_source: str | None = None,
    token_status_detail: Literal["before", "before_after"] = "before",
    token_status_show_usage_and_estimate: bool = False,
    event_emitter: Callable[[Any], Awaitable[None]] | None = None,
    file_context_enabled: bool = True,
    task_estimate_body: dict[str, Any] | None = None,
    summary_prompt: str | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    token_system_prompt: str | None = None,
    dropped_message_keys: frozenset[str] = frozenset(),
    ref_mode_active: bool = False,
    ref_substitution_threshold_tokens: int = 10_000,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
    parent_prefetch_task: asyncio.Task[Any] | None = None,
    _prepared: _PreparedSoftPrefetch | None = None,
) -> bool:
    prepared = _prepared or _prepare_soft_compaction_prefetch(
        request=request,
        user=user,
        metadata=metadata,
        body=body,
        pipe_function_id=pipe_function_id,
        summary_model_id=summary_model_id,
        summary_tool_policy=summary_tool_policy,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        effective_trigger_input_tokens=effective_trigger_input_tokens,
        effective_soft_trigger_input_tokens=effective_soft_trigger_input_tokens,
        trigger_observed_tokens=trigger_observed_tokens,
        trigger_estimated_tokens=trigger_estimated_tokens,
        trigger_usage_source=trigger_usage_source,
        token_status_detail=token_status_detail,
        token_status_show_usage_and_estimate=token_status_show_usage_and_estimate,
        event_emitter=event_emitter,
        file_context_enabled=file_context_enabled,
        task_estimate_body=task_estimate_body,
        summary_prompt=summary_prompt,
        transient_message_patterns=transient_message_patterns,
        token_system_prompt=token_system_prompt,
        dropped_message_keys=dropped_message_keys,
        ref_mode_active=ref_mode_active,
        ref_substitution_threshold_tokens=ref_substitution_threshold_tokens,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )
    if prepared is None:
        return False
    return _launch_soft_prefetch_task(prepared.key, prepared.run(parent_prefetch_task))


def _choice_assistant_message_for_prefetch(choice: Any) -> dict[str, Any] | None:
    if _choice_has_tool_call(choice):
        return None
    content = _choice_message_text(choice)
    if content is None:
        return None
    return {"role": "assistant", "content": content}


def _assistant_message_for_completed_prefetch(response: Any) -> dict[str, Any] | None:
    if not isinstance(response, dict) or response.get("error"):
        return None
    if _responses_output_has_tool_call(response):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        return _choice_assistant_message_for_prefetch(choices[0])
    responses_text = _responses_output_text(response)
    if isinstance(responses_text, str):
        return {"role": "assistant", "content": responses_text}
    return None


def _completed_turn_prefetch_body(body: dict[str, Any], assistant_message: dict[str, Any]) -> dict[str, Any] | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    completed = _copy_body_preserving_metadata(body)
    completed["messages"] = [
        *copy.deepcopy(messages),
        copy.deepcopy(assistant_message),
        {"role": "user", "content": ""},
    ]
    return completed


async def _find_reusable_checkpoint_for_source(
    *,
    store: Any,
    user_id: str,
    chat_id: str,
    pipe_function_id: str,
    profile_hash: str,
    source_messages: list[dict[str, Any]],
    prefix_file_fingerprint: str | None = None,
    prefix_file_fingerprint_resolver: Callable[[int], str | None] | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    source_hash: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    if source_hash is None:
        source_hash = compute_summary_source_hash(
            source_messages,
            prefix_file_fingerprint,
            _prefix_file_fingerprint_resolver_db_chain(
                prefix_file_fingerprint_resolver
            ),
            transient_message_patterns=transient_message_patterns,
        )
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
        prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
        transient_message_patterns=transient_message_patterns,
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
    transient_message_patterns: TransientMessagePatterns | None = None,
    capture_logical_snapshot: bool = False,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> ReusableCheckpointMatch | None:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")
    if not user_id or not _chat_id_supported(chat_id):
        return None

    tool_cut = select_tool_result_compaction_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    cut = select_safe_message_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    if (tool_cut is None or not tool_cut.summarization_prefix) and not cut.summarization_prefix:
        return None

    profile_hash = checkpoint_profile_hash
    await ensure_checkpoint_table_initialized(request=request)
    store = CheckpointStore()
    resolver_source_messages: list[dict[str, Any]] = []
    if tool_cut is not None and tool_cut.summarization_prefix:
        resolver_source_messages.extend(tool_cut.summarization_prefix)
    if cut.summarization_prefix:
        resolver_source_messages.extend(cut.summarization_prefix)
    prefix_file_fingerprint_resolver = await _build_prefix_file_fingerprint_resolver(
        request,
        metadata,
        resolver_source_messages,
        transient_message_patterns=transient_message_patterns,
    )
    file_backed_image_db_chain = _prefix_file_fingerprint_resolver_db_chain(
        prefix_file_fingerprint_resolver
    )

    if tool_cut is not None and tool_cut.summarization_prefix:
        tool_identity_fingerprint = (
            prefix_file_fingerprint_resolver(
                _source_identity_message_count(
                    tool_cut.summarization_prefix,
                    transient_message_patterns=transient_message_patterns,
                )
            )
            if prefix_file_fingerprint_resolver is not None
            else None
        )
        tool_source_hash = compute_summary_source_hash(
            tool_cut.summarization_prefix,
            tool_identity_fingerprint,
            file_backed_image_db_chain,
            transient_message_patterns=transient_message_patterns,
        )
        tool_match = await _find_reusable_checkpoint_for_source(
            store=store,
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            profile_hash=profile_hash,
            source_messages=tool_cut.summarization_prefix,
            prefix_file_fingerprint=tool_identity_fingerprint,
            prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
            transient_message_patterns=transient_message_patterns,
            source_hash=tool_source_hash,
        )
        if tool_match is not None:
            kind, checkpoint = tool_match
            logical_snapshot = (
                await get_or_build_logical_history_snapshot(
                    request,
                    tool_cut.summarization_prefix,
                    identity=(
                        CHECKPOINT_NAMESPACE,
                        user_id,
                        chat_id,
                        pipe_function_id,
                        profile_hash,
                    ),
                    prefix_file_fingerprint_resolver=(
                        prefix_file_fingerprint_resolver
                    ),
                    transient_message_patterns=transient_message_patterns,
                    source_hash=tool_source_hash,
                )
                if capture_logical_snapshot
                else None
            )
            return ReusableCheckpointMatch(
                kind=kind,
                source_message_count=int(checkpoint.get("source_message_count") or tool_cut.source_message_count),
                source_kind="tool",
                checkpoint=checkpoint,
                logical_snapshot=logical_snapshot,
            )

    if not cut.summarization_prefix:
        return None

    message_identity_fingerprint = (
        prefix_file_fingerprint_resolver(
            _source_identity_message_count(
                cut.summarization_prefix,
                transient_message_patterns=transient_message_patterns,
            )
        )
        if prefix_file_fingerprint_resolver is not None
        else None
    )
    message_source_hash = compute_summary_source_hash(
        cut.summarization_prefix,
        message_identity_fingerprint,
        file_backed_image_db_chain,
        transient_message_patterns=transient_message_patterns,
    )
    message_match = await _find_reusable_checkpoint_for_source(
        store=store,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        profile_hash=profile_hash,
        source_messages=cut.summarization_prefix,
        prefix_file_fingerprint=message_identity_fingerprint,
        prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
        transient_message_patterns=transient_message_patterns,
        source_hash=message_source_hash,
    )
    if message_match is not None:
        kind, checkpoint = message_match
        logical_snapshot = (
            await get_or_build_logical_history_snapshot(
                request,
                cut.summarization_prefix,
                identity=(
                    CHECKPOINT_NAMESPACE,
                    user_id,
                    chat_id,
                    pipe_function_id,
                    profile_hash,
                ),
                prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
                transient_message_patterns=transient_message_patterns,
                source_hash=message_source_hash,
            )
            if capture_logical_snapshot
            else None
        )
        return ReusableCheckpointMatch(
            kind=kind,
            source_message_count=int(checkpoint.get("source_message_count") or cut.source_message_count),
            source_kind="message",
            checkpoint=checkpoint,
            logical_snapshot=logical_snapshot,
        )
    return None


async def _summary_token_count_from_checkpoint(
    *,
    request: Any,
    checkpoint: dict[str, Any],
    historical_source_messages: list[dict[str, Any]] | None,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
    transient_message_patterns: TransientMessagePatterns | None = None,
) -> int | None:
    stored = checkpoint.get("summary_token_count")
    if isinstance(stored, int) and stored >= 0:
        return stored
    if isinstance(stored, str) and stored.isdigit():
        return int(stored)
    return await estimate_message_tokens_async(
        render_summary_message_from_checkpoint(
            checkpoint,
            historical_source_messages=historical_source_messages,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            transient_message_patterns=transient_message_patterns,
        ),
        request=request,
    )


async def _estimate_checkpoint_applied_body_tokens(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    match: ReusableCheckpointMatch,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
    file_context_enabled: bool = True,
    transient_message_patterns: TransientMessagePatterns | None = None,
    token_system_prompt: str | None = None,
    dropped_message_keys: frozenset[str] = frozenset(),
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> int | None:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    candidates: list[tuple[MessageCut | ToolResultCompactionCut, list[dict[str, Any]]]] = []
    tool_cut = select_tool_result_compaction_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    if tool_cut is not None and tool_cut.summarization_prefix and match.source_kind == "tool":
        candidates.append((tool_cut, tool_cut.summarization_prefix))

    cut = select_safe_message_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    if cut.summarization_prefix and match.source_kind == "message":
        candidates.append((cut, cut.summarization_prefix))

    if not candidates:
        return None

    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")
    profile_hash = checkpoint_profile_hash
    store = CheckpointStore()
    resolver_source_messages: list[dict[str, Any]] = []
    for _candidate_cut, source_messages in candidates:
        resolver_source_messages.extend(source_messages)
    prefix_file_fingerprint_resolver = await _build_prefix_file_fingerprint_resolver(
        request,
        metadata,
        resolver_source_messages,
        transient_message_patterns=transient_message_patterns,
    )
    file_backed_image_db_chain = _prefix_file_fingerprint_resolver_db_chain(prefix_file_fingerprint_resolver)

    for candidate_cut, source_messages in candidates:
        checkpoint = match.checkpoint
        if checkpoint is None:
            identity_fingerprint = (
                prefix_file_fingerprint_resolver(
                    _source_identity_message_count(
                        source_messages,
                        transient_message_patterns=transient_message_patterns,
                    )
                )
                if prefix_file_fingerprint_resolver is not None
                else None
            )
            checkpoint_match = await _find_reusable_checkpoint_for_source(
                store=store,
                user_id=user_id,
                chat_id=chat_id,
                pipe_function_id=pipe_function_id,
                profile_hash=profile_hash,
                source_messages=source_messages,
                prefix_file_fingerprint=identity_fingerprint,
                prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
                transient_message_patterns=transient_message_patterns,
            )
            checkpoint = checkpoint_match[1] if checkpoint_match is not None else None
        if checkpoint is None:
            continue

        parent_count = int(checkpoint.get("source_message_count") or 0)
        raw_parent_count = _raw_prefix_len_for_source_count(
            source_messages,
            parent_count,
            transient_message_patterns=transient_message_patterns,
        )
        if parent_count <= 0 or raw_parent_count is None:
            continue
        parent_fingerprint = (
            prefix_file_fingerprint_resolver(parent_count)
            if prefix_file_fingerprint_resolver is not None
            else None
        )
        if (
            compute_summary_source_hash(
                source_messages[:raw_parent_count],
                parent_fingerprint,
                file_backed_image_db_chain,
                transient_message_patterns=transient_message_patterns,
            )
            != checkpoint.get("source_hash")
        ):
            continue

        if isinstance(candidate_cut, ToolResultCompactionCut):
            message_cut = MessageCut(
                preserved_system_message=copy.deepcopy(candidate_cut.preserved_system_message),
                summarization_prefix=copy.deepcopy(candidate_cut.summarization_prefix),
                tail_messages=copy.deepcopy(candidate_cut.tail_messages),
                source_message_count=candidate_cut.source_message_count,
            )
        else:
            message_cut = candidate_cut

        summary_tokens = await _summary_token_count_from_checkpoint(
            request=request,
            checkpoint=checkpoint,
            historical_source_messages=message_cut.summarization_prefix[:raw_parent_count],
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            transient_message_patterns=transient_message_patterns,
        )
        if summary_tokens is None:
            return None

        estimate_source = []
        if message_cut.preserved_system_message is not None:
            estimate_source.append(message_cut.preserved_system_message)
        estimate_source.extend(message_cut.summarization_prefix[raw_parent_count:])
        estimate_source.extend(message_cut.tail_messages)

        remaining_body = _copy_body_preserving_metadata(body)
        remaining_body["messages"] = estimate_source
        remaining_body.pop("previous_response_id", None)
        # Reflect retained file context in the estimate so the threshold decision
        # accounts for what the target will actually receive after compaction.
        if file_context_enabled:
            metadata_files = (body.get("metadata") or {}).get("files") if isinstance(body.get("metadata"), dict) else None
            remaining_body = await _inject_target_file_context(
                request=request,
                user=user,
                body=remaining_body,
                chat_id=chat_id or None,
                current_message_id=str(metadata.get("user_message_id") or metadata.get("message_id") or "") or None,
                compaction_prefix_count=parent_count,
                metadata_files=metadata_files,
                metadata_user_message=metadata.get("user_message"),
                event_emitter=None,
                file_context_enabled=True,
                emit_source_events=False,
                transient_message_patterns=transient_message_patterns,
            )
        remaining_tokens = await _estimate_provider_input_tokens_async(
            remaining_body,
            request=request,
            user=user,
            system_prompt=token_system_prompt,
            dropped_message_keys=dropped_message_keys,
        )
        if remaining_tokens is None:
            return None
        return summary_tokens + remaining_tokens
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
    file_context_enabled: bool = True,
    transient_message_patterns: TransientMessagePatterns | None = None,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> tuple[dict[str, Any], bool, int]:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return body, False, 0

    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")
    if not user_id or not _chat_id_supported(chat_id):
        return body, False, 0

    candidates: list[tuple[str, MessageCut | ToolResultCompactionCut, list[dict[str, Any]]]] = []
    tool_cut = select_tool_result_compaction_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    if tool_cut is not None and tool_cut.summarization_prefix and match.source_kind == "tool":
        candidates.append(("tool", tool_cut, tool_cut.summarization_prefix))

    cut = select_safe_message_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    if cut.summarization_prefix and match.source_kind == "message":
        candidates.append(("message", cut, cut.summarization_prefix))

    await ensure_checkpoint_table_initialized(request=request)
    store = CheckpointStore()
    profile_hash = checkpoint_profile_hash
    resolver_source_messages: list[dict[str, Any]] = []
    for _kind, _candidate_cut, source_messages in candidates:
        resolver_source_messages.extend(source_messages)
    prefix_file_fingerprint_resolver = await _build_prefix_file_fingerprint_resolver(
        request,
        metadata,
        resolver_source_messages,
        require_file_context_chain=file_context_enabled,
        transient_message_patterns=transient_message_patterns,
    )
    file_backed_image_db_chain = _prefix_file_fingerprint_resolver_db_chain(prefix_file_fingerprint_resolver)

    for _kind, candidate_cut, source_messages in candidates:
        identity_fingerprint = (
            prefix_file_fingerprint_resolver(
                _source_identity_message_count(
                    source_messages,
                    transient_message_patterns=transient_message_patterns,
                )
            )
            if prefix_file_fingerprint_resolver is not None
            else None
        )
        checkpoint_match = await _find_reusable_checkpoint_for_source(
            store=store,
            user_id=user_id,
            chat_id=chat_id,
            pipe_function_id=pipe_function_id,
            profile_hash=profile_hash,
            source_messages=source_messages,
            prefix_file_fingerprint=identity_fingerprint,
            prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
            transient_message_patterns=transient_message_patterns,
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
        checkpoint_count = int(checkpoint.get("source_message_count") or 0)
        checkpoint_raw_count = _raw_prefix_len_for_source_count(
            message_cut.summarization_prefix,
            checkpoint_count,
            transient_message_patterns=transient_message_patterns,
        )
        if checkpoint_raw_count is None:
            continue
        compacted["messages"] = replace_prefix_with_parent_checkpoint_and_delta(
            message_cut,
            checkpoint,
            prefix_file_fingerprint=_resolve_fingerprint(
                prefix_file_fingerprint_resolver,
                checkpoint_count,
            ),
            file_backed_image_db_chain=file_backed_image_db_chain,
            transient_message_patterns=transient_message_patterns,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
        compacted.pop("previous_response_id", None)
        return compacted, True, checkpoint_count

    raise RuntimeError("Reusable checkpoint match disappeared before compaction")


async def _estimate_task_checkpoint_applied_body_tokens(
    *,
    request: Any,
    user: Any,
    metadata: dict[str, Any],
    body: dict[str, Any],
    pipe_function_id: str,
    match: ReusableCheckpointMatch,
    historical_message_excerpt_bytes: int,
    historical_message_excerpt_count: int,
    file_context_enabled: bool = True,
    transient_message_patterns: TransientMessagePatterns | None = None,
    token_system_prompt: str | None = None,
    dropped_message_keys: frozenset[str] = frozenset(),
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> int | None:
    task_metadata = metadata
    body_metadata = body.get("metadata")
    if isinstance(body_metadata, dict) and _task_body_from_metadata(body_metadata) is not None:
        task_metadata = body_metadata
    source_body = _task_history_source_body_for_compaction(body, task_metadata) or body
    try:
        compacted_source, compacted, compaction_prefix_count = await _compact_body_with_reusable_checkpoint(
            request=request,
            user=user,
            metadata=task_metadata,
            body=source_body,
            pipe_function_id=pipe_function_id,
            match=match,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            file_context_enabled=file_context_enabled,
            transient_message_patterns=transient_message_patterns,
            checkpoint_profile_hash=checkpoint_profile_hash,
        )
    except SummaryFileContextUnavailable:
        return None
    if not compacted:
        return None
    rebuilt = await _rebuild_task_body_from_compacted_history(
        request=request,
        user=user,
        base_body=body,
        metadata=task_metadata,
        compacted_history_messages=compacted_source["messages"],
    )
    if rebuilt is None:
        return None
    if file_context_enabled:
        rebuilt = await _inject_target_file_context(
            request=request,
            user=user,
            body=rebuilt,
            chat_id=str(task_metadata.get("chat_id") or "") or None,
            current_message_id=str(task_metadata.get("user_message_id") or task_metadata.get("message_id") or "") or None,
            compaction_prefix_count=compaction_prefix_count,
            metadata_files=(rebuilt.get("metadata") or {}).get("files") if isinstance(rebuilt.get("metadata"), dict) else None,
            metadata_user_message=task_metadata.get("user_message"),
            event_emitter=None,
            file_context_enabled=True,
            emit_source_events=False,
            transient_message_patterns=transient_message_patterns,
        )
    return await _estimate_provider_input_tokens_async(
        rebuilt,
        request=request,
        user=user,
        system_prompt=token_system_prompt,
        dropped_message_keys=dropped_message_keys,
    )


async def _model_dict_from_request(request: Any) -> dict[str, Any]:
    state = getattr(getattr(request, "app", None), "state", None)
    if state is None:
        return {}
    models: dict[str, Any] = {}
    for model in await _iter_cache_models_from_state_compatible(state):
        model_id = _model_id(model)
        if model_id is not None and model_id not in models:
            models[model_id] = model
    return models


def _provider_cache_model_from_request(
    request: Any,
    cache_attr: str,
    model_id: str,
) -> dict[str, Any] | None:
    state = getattr(getattr(request, "app", None), "state", None)
    if state is None:
        return None
    for model in _iter_model_cache_values(getattr(state, cache_attr, None)):
        if not isinstance(model, dict):
            continue
        cached_model_id = model.get("model") if cache_attr == "OLLAMA_MODELS" else _model_id(model)
        if cached_model_id == model_id:
            return model
    return None


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


def _legacy_default_models_config_value(request: Any) -> Any:
    config = getattr(getattr(request, "app", None), "state", None)
    config = getattr(config, "config", None)
    return getattr(config, "DEFAULT_MODELS", None)


def _available_custom_model_fallback_id(
    default_models_value: Any,
    models: dict[str, Any],
    *,
    pipe_function_id: str = PIPE_FUNCTION_ID,
) -> str | None:
    default_models = str(default_models_value or "").split(",")
    fallback_model_id = default_models[0].strip() if default_models and default_models[0] else None
    fallback_model = models.get(fallback_model_id) if fallback_model_id else None
    if not fallback_model_id or not isinstance(fallback_model, dict):
        return None
    if _is_own_wrapper_or_preset(
        fallback_model_id,
        fallback_model,
        pipe_function_id=pipe_function_id,
    ):
        return None
    return fallback_model_id


async def _custom_model_fallback_model_id_compatible(
    request: Any,
    models: dict[str, Any],
    *,
    pipe_function_id: str = PIPE_FUNCTION_ID,
) -> str | None:
    if not _custom_model_fallback_enabled():
        return None
    default_models_value = await _open_webui_config_get("ui.default_models")
    if default_models_value is CONFIG_VALUE_MISSING or default_models_value is None:
        default_models_value = _legacy_default_models_config_value(request)
    return _available_custom_model_fallback_id(
        default_models_value,
        models,
        pipe_function_id=pipe_function_id,
    )


def _target_record_params(model_info: Any) -> dict[str, Any]:
    if model_info is None or model_info is TARGET_MODEL_RECORD_UNKNOWN:
        return {}
    if isinstance(model_info, dict):
        return _payload_params(model_info)
    params = getattr(model_info, "params", None)
    dumped = _dump_model_value(params)
    return copy.deepcopy(dumped) if isinstance(dumped, dict) else {}


def _target_record_system_prompt(model_info: Any) -> str | None:
    system = _target_record_params(model_info).get("system")
    return system if isinstance(system, str) and system else None


async def _usage_anchor_resolved_system_identity(
    system: str,
    *,
    metadata: dict[str, Any] | None,
    user: Any,
) -> str:
    from open_webui.utils.task import prompt_template, prompt_variables_template

    if _render_chat_variables is not None and metadata:
        system = _render_chat_variables(
            system,
            metadata.get("chat_variables", {}),
            required=False,
        )

    variables = metadata.get("variables", {}) if isinstance(metadata, dict) else {}
    if variables:
        if not isinstance(variables, dict):
            raise TypeError("metadata variables must be an object")
        normalized_variables = dict(variables)
        for placeholder, sentinel in USAGE_ANCHOR_SYSTEM_CLOCK_SENTINELS:
            if placeholder in normalized_variables:
                normalized_variables[placeholder] = sentinel
        system = prompt_variables_template(system, normalized_variables)

    # Mirror Core's prompt expansion while keeping its per-request clock values
    # stable. A future clock placeholder omitted here only causes safe misses.
    for placeholder, sentinel in USAGE_ANCHOR_SYSTEM_CLOCK_SENTINELS:
        system = system.replace(placeholder, sentinel)
    return await prompt_template(system, coerce_open_webui_user(user))


async def _usage_anchor_model_shaping_profile(
    request: Any,
    model_id: str,
    model_info: Any,
    *,
    metadata: dict[str, Any] | None = None,
    user: Any = None,
) -> dict[str, Any] | None:
    if model_info is TARGET_MODEL_RECORD_UNKNOWN:
        return None
    configured_base_model_id = _target_record_base_model_id(model_info)
    request_base_model_id = getattr(request, "base_model_id", None)
    effective_base_model_id = (
        request_base_model_id
        if configured_base_model_id and isinstance(request_base_model_id, str) and request_base_model_id
        else configured_base_model_id
    )
    params = _target_record_params(model_info)
    profile = {
        "model_id": model_id,
        "base_model_id": effective_base_model_id,
        "params": _canonicalize_general_value(params),
    }
    system = params.get("system")
    if system is None or system == "" or (metadata is None and user is None):
        return profile
    if not isinstance(system, str):
        return None
    try:
        profile["resolved_system"] = await _usage_anchor_resolved_system_identity(
            system,
            metadata=metadata,
            user=user,
        )
    except Exception:
        LOG.debug("Auto-compaction usage anchor disabled: system prompt expansion failed", exc_info=True)
        return None
    return profile


def _usage_anchor_shaping_hash(
    profiles: list[dict[str, Any]],
    *,
    transport_profile: dict[str, Any],
) -> str:
    return _json_hash(
        {
            "family": USAGE_ANCHOR_SHAPING_PROFILE_FAMILY,
            "models": profiles,
            "transport": transport_profile,
        }
    )


FALLBACK_OPEN_WEBUI_PARAM_KEYS = frozenset(
    {
        "stream_response",
        "stream_delta_chunk_size",
        "function_calling",
        "reasoning_tags",
        "compact_token_threshold",
        "system",
    }
)
FALLBACK_OPENAI_PROVIDER_PARAM_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "min_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_effort",
        "seed",
        "stop",
        "logit_bias",
        "response_format",
    }
)
FALLBACK_OLLAMA_OPTION_PARAM_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "num_ctx",
        "num_batch",
        "num_keep",
        "num_predict",
        "repeat_last_n",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "num_gpu",
        "use_mmap",
        "use_mlock",
        "num_thread",
    }
)
FALLBACK_OLLAMA_ROOT_PARAM_KEYS = frozenset({"format", "keep_alive", "think"})
FALLBACK_PROVIDER_TOP_LEVEL_PARAM_KEYS = (
    FALLBACK_OPENAI_PROVIDER_PARAM_KEYS | FALLBACK_OLLAMA_OPTION_PARAM_KEYS | FALLBACK_OLLAMA_ROOT_PARAM_KEYS
)
FALLBACK_PROVIDER_PARAM_ALIASES = {"max_tokens": "num_predict"}
FALLBACK_OLLAMA_ROOT_PASSTHROUGH_PARAM_KEYS = frozenset({"response_format"})
FALLBACK_BODY_NON_PARAM_KEYS = frozenset(
    {
        "model",
        "messages",
        "metadata",
        "stream",
        "files",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "parallel_tool_calls",
    }
)


def _fallback_provider_top_level_param_keys(target_params: dict[str, Any] | None) -> set[str]:
    keys = set(FALLBACK_PROVIDER_TOP_LEVEL_PARAM_KEYS)
    if isinstance(target_params, dict):
        for key, value in target_params.items():
            if key == "custom_params" and isinstance(value, dict):
                keys.update(param_key for param_key in value if param_key not in FALLBACK_BODY_NON_PARAM_KEYS)
                continue
            if key not in FALLBACK_OPEN_WEBUI_PARAM_KEYS and key not in FALLBACK_BODY_NON_PARAM_KEYS:
                keys.add(key)
            alias = FALLBACK_PROVIDER_PARAM_ALIASES.get(key)
            if alias:
                keys.add(alias)
    return keys


def _apply_custom_model_fallback_params(
    body: dict[str, Any],
    *,
    fallback_model: dict[str, Any] | None,
    target_params: dict[str, Any] | None,
) -> dict[str, Any]:
    if not fallback_model:
        return body
    is_ollama_fallback = fallback_model.get("owned_by") == "ollama"
    existing_params = body.get("params")
    merged_params = copy.deepcopy(target_params or {})
    top_level_param_keys = _fallback_provider_top_level_param_keys(merged_params)
    top_level_request_params = {
        key: copy.deepcopy(body[key])
        for key in top_level_param_keys
        if key in body and body[key] is not None
    }
    existing_options = body.get("options")
    option_request_params = (
        {key: copy.deepcopy(value) for key, value in existing_options.items() if value is not None}
        if isinstance(existing_options, dict)
        else {}
    )
    request_params = (
        {key: copy.deepcopy(value) for key, value in existing_params.items() if value is not None}
        if isinstance(existing_params, dict)
        else {}
    )
    root_passthrough_request_params = {}
    if is_ollama_fallback:
        root_passthrough_request_params = {
            key: top_level_request_params.pop(key)
            for key in list(top_level_request_params)
            if key in FALLBACK_OLLAMA_ROOT_PASSTHROUGH_PARAM_KEYS
        }
    if not merged_params and not isinstance(existing_params, dict) and not top_level_request_params and not option_request_params:
        return body
    try:
        from open_webui.utils.middleware import apply_params_to_form_data
    except ImportError:
        return body

    patched = _copy_body_preserving_metadata(body)
    if request_params:
        merged_params.update(request_params)
    merged_params.update(top_level_request_params)
    request_visible_params = {**request_params, **top_level_request_params}
    custom_params = merged_params.get("custom_params")
    if isinstance(custom_params, dict):
        custom_params = copy.deepcopy(custom_params)
        for key, value in request_visible_params.items():
            if key in custom_params:
                custom_params[key] = copy.deepcopy(value)
        if is_ollama_fallback:
            for source_key, alias_key in FALLBACK_PROVIDER_PARAM_ALIASES.items():
                if alias_key in request_visible_params or alias_key in option_request_params:
                    custom_params.pop(source_key, None)
            for key in root_passthrough_request_params:
                custom_params.pop(key, None)
        merged_params["custom_params"] = custom_params
    if is_ollama_fallback:
        for source_key, alias_key in FALLBACK_PROVIDER_PARAM_ALIASES.items():
            if alias_key in request_visible_params or alias_key in option_request_params:
                merged_params.pop(source_key, None)
        for key in root_passthrough_request_params:
            merged_params.pop(key, None)
    patched["params"] = merged_params
    patched = apply_params_to_form_data(patched, fallback_model)
    if option_request_params and isinstance(patched.get("options"), dict):
        patched["options"].update(option_request_params)
    return patched


def _apply_resolved_model_route_params(
    body: dict[str, Any],
    *,
    models: dict[str, Any],
    route: CoreChatModelRoute,
) -> dict[str, Any]:
    params_model = route.fallback_model
    if params_model is None:
        resolved_model = models.get(route.model_id)
        if isinstance(resolved_model, dict) and resolved_model.get("owned_by") == "ollama":
            params_model = resolved_model
    return _apply_custom_model_fallback_params(
        body,
        fallback_model=params_model,
        target_params=route.target_params,
    )


def _legacy_provider_transport_config(
    request: Any,
    *,
    base_urls_attr: str,
    api_configs_attr: str,
) -> tuple[list[Any], dict[Any, Any]] | None:
    state = getattr(getattr(request, "app", None), "state", None)
    config = getattr(state, "config", None)
    base_urls = getattr(config, base_urls_attr, None)
    api_configs = getattr(config, api_configs_attr, None)
    if not isinstance(base_urls, list) or not isinstance(api_configs, dict):
        return None
    return base_urls, api_configs


async def _usage_anchor_transport_profile(
    request: Any,
    models: dict[str, Any],
    provider_model_id: str,
) -> tuple[dict[str, Any] | None, frozenset[str]]:
    provider_model = models.get(provider_model_id)
    if not isinstance(provider_model, dict):
        return None, frozenset()
    owned_by = str(provider_model.get("owned_by") or "")
    if provider_model.get("owned_by") == "ollama":
        dropped_message_keys = frozenset({"reasoning_content", "reasoning_details"})
        provider_cache_model = _provider_cache_model_from_request(request, "OLLAMA_MODELS", provider_model_id)
        digest = provider_cache_model.get("digest") if provider_cache_model is not None else None
        if not isinstance(digest, str) or not digest:
            return None, dropped_message_keys
        urls = provider_cache_model.get("urls") if provider_cache_model is not None else None
        if not isinstance(urls, list) or not urls or any(
            not isinstance(url_idx, int) or isinstance(url_idx, bool) for url_idx in urls
        ):
            return None, dropped_message_keys
        url_indices = sorted(set(urls))
        try:
            from open_webui.routers import ollama as ollama_router
        except Exception:
            LOG.debug("Could not resolve Ollama transport for usage-anchor identity", exc_info=True)
            return None, dropped_message_keys
        get_runtime_config = getattr(ollama_router, "get_ollama_runtime_config", None)
        if callable(get_runtime_config):
            try:
                _, base_urls, api_configs = await get_runtime_config()
            except Exception:
                LOG.debug("Could not resolve Ollama transport for usage-anchor identity", exc_info=True)
                return None, dropped_message_keys
        else:
            legacy_config = _legacy_provider_transport_config(
                request,
                base_urls_attr="OLLAMA_BASE_URLS",
                api_configs_attr="OLLAMA_API_CONFIGS",
            )
            if legacy_config is None:
                return None, dropped_message_keys
            base_urls, api_configs = legacy_config
        if not isinstance(base_urls, list) or not isinstance(api_configs, dict):
            return None, dropped_message_keys
        backends = []
        for url_idx in url_indices:
            if url_idx < 0 or url_idx >= len(base_urls):
                return None, dropped_message_keys
            url = str(base_urls[url_idx])
            api_config = api_configs.get(str(url_idx), api_configs.get(url, {}))
            if not isinstance(api_config, dict):
                return None, dropped_message_keys
            backends.append(
                {
                    "url_idx": url_idx,
                    "url": str(url),
                    "api_config": _canonicalize_general_value(api_config),
                }
            )
        return {
            "owned_by": "ollama",
            "digest": digest,
            # Core treats identical model IDs across these backends as one
            # load-balanced logical model. Track the complete eligible set so
            # config changes invalidate anchors without disabling HA replicas.
            "backends": backends,
        }, dropped_message_keys
    if provider_model.get("pipe") or owned_by == "arena":
        return None, frozenset()
    provider_cache_model = _provider_cache_model_from_request(request, "OPENAI_MODELS", provider_model_id)
    url_idx = provider_cache_model.get("urlIdx") if provider_cache_model is not None else None
    if not isinstance(url_idx, int) or isinstance(url_idx, bool):
        return None, frozenset()
    try:
        from open_webui.routers import openai as openai_router
    except Exception:
        LOG.debug("Could not resolve OpenAI transport for usage-anchor identity", exc_info=True)
        return None, frozenset()
    get_connection = getattr(openai_router, "get_openai_connection", None)
    if callable(get_connection):
        try:
            url, _, api_config = await get_connection(url_idx)
        except Exception:
            LOG.debug("Could not resolve OpenAI transport for usage-anchor identity", exc_info=True)
            return None, frozenset()
    else:
        legacy_config = _legacy_provider_transport_config(
            request,
            base_urls_attr="OPENAI_API_BASE_URLS",
            api_configs_attr="OPENAI_API_CONFIGS",
        )
        if legacy_config is None:
            return None, frozenset()
        base_urls, api_configs = legacy_config
        if url_idx < 0 or url_idx >= len(base_urls):
            return None, frozenset()
        url = str(base_urls[url_idx])
        api_config = api_configs.get(str(url_idx), api_configs.get(url, {}))
    api_type = str(api_config.get("api_type") or "chat_completions") if isinstance(api_config, dict) else "chat_completions"
    transport = {
        "owned_by": owned_by or "openai",
        "url_idx": url_idx,
        "url": str(url),
        "api_type": api_type,
        "api_config": _canonicalize_general_value(api_config) if isinstance(api_config, dict) else {},
    }
    dropped_message_keys = (
        frozenset({"reasoning_content", "reasoning_details", "thinking"})
        if api_type == "responses"
        else frozenset()
    )
    return transport, dropped_message_keys


def _global_model_access_bypass_enabled() -> bool:
    try:
        from open_webui.env import BYPASS_MODEL_ACCESS_CONTROL

        return bool(BYPASS_MODEL_ACCESS_CONTROL)
    except Exception:
        return False


async def _resolve_core_chat_model_route(
    request: Any,
    model_id: str,
    *,
    pipe_function_id: str = PIPE_FUNCTION_ID,
    metadata: dict[str, Any] | None = None,
    user: Any = None,
) -> CoreChatModelRoute:
    models = await _model_dict_from_request(request)
    if model_id not in models:
        return CoreChatModelRoute(model_id=model_id)
    model_info = await _get_target_db_model_record(model_id)
    target_shaping_profile = await _usage_anchor_model_shaping_profile(
        request,
        model_id,
        model_info,
        metadata=metadata,
        user=user,
    )
    base_model_id = _target_record_base_model_id(model_info)
    if base_model_id and base_model_id not in models:
        fallback_model_id = await _custom_model_fallback_model_id_compatible(
            request,
            models,
            pipe_function_id=pipe_function_id,
        )
        fallback_model = models.get(fallback_model_id) if fallback_model_id else None
        if fallback_model_id and isinstance(fallback_model, dict):
            fallback_model_info = await _get_target_db_model_record(fallback_model_id)
            fallback_shaping_profile = await _usage_anchor_model_shaping_profile(
                request,
                fallback_model_id,
                fallback_model_info,
                metadata=metadata,
                user=user,
            )
            shaping_profiles = (
                [target_shaping_profile, fallback_shaping_profile]
                if target_shaping_profile is not None and fallback_shaping_profile is not None
                else None
            )
            provider_model_id = (
                str(fallback_shaping_profile.get("base_model_id") or fallback_model_id)
                if fallback_shaping_profile is not None
                else fallback_model_id
            )
            transport_profile, dropped_message_keys = await _usage_anchor_transport_profile(
                request,
                models,
                provider_model_id,
            )
            # Intentional late fallback: wrapper preprocessing already used the selected model's mirrored metadata.
            # Preserve that logical configuration; do not re-run preprocessing for the fallback route.
            return CoreChatModelRoute(
                model_id=fallback_model_id,
                fallback_model=copy.deepcopy(fallback_model),
                target_params=_target_record_params(model_info),
                token_system_prompt=_target_record_system_prompt(fallback_model_info),
                usage_anchor_shaping_hash=(
                    _usage_anchor_shaping_hash(
                        shaping_profiles,
                        transport_profile=transport_profile,
                    )
                    if shaping_profiles is not None and transport_profile is not None
                    else None
                ),
                provider_model_id=provider_model_id,
                usage_anchor_dropped_message_keys=dropped_message_keys,
            )
    provider_model_id = (
        str(target_shaping_profile.get("base_model_id") or model_id)
        if target_shaping_profile is not None
        else model_id
    )
    transport_profile, dropped_message_keys = await _usage_anchor_transport_profile(
        request,
        models,
        provider_model_id,
    )
    return CoreChatModelRoute(
        model_id=model_id,
        token_system_prompt=_target_record_system_prompt(model_info),
        usage_anchor_shaping_hash=(
            _usage_anchor_shaping_hash(
                [target_shaping_profile],
                transport_profile=transport_profile,
            )
            if target_shaping_profile is not None and transport_profile is not None
            else None
        ),
        provider_model_id=provider_model_id,
        usage_anchor_dropped_message_keys=dropped_message_keys,
    )


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


async def _resolve_arena_chat_model_route_with_access(
    *,
    request: Any,
    user: Any,
    models: dict[str, Any],
    route: CoreChatModelRoute,
    original_model_id: str,
    pipe_function_id: str = PIPE_FUNCTION_ID,
) -> tuple[CoreChatModelRoute, str | None]:
    route, selected_arena_model_id = _resolve_arena_chat_model_route(
        models,
        route,
        pipe_function_id=pipe_function_id,
    )
    if route.model_id != original_model_id:
        await _validate_chat_completion_runtime_model_access(
            request=request,
            user=user,
            model_id=route.model_id,
        )
    if selected_arena_model_id is not None:
        selected_model_info = await _get_target_db_model_record(route.model_id)
        route = replace(
            route,
            token_system_prompt=_target_record_system_prompt(selected_model_info),
        )
    return route, selected_arena_model_id


async def _validate_target_access(
    *,
    target_model_id: str,
    request: Any,
    user: Any,
    pipe_function_id: str = PIPE_FUNCTION_ID,
) -> None:
    models = await _model_dict_from_request(request)
    model = models.get(target_model_id)
    state = getattr(getattr(request, "app", None), "state", None)
    if model is not None and state is not None:
        provider_enabled_states = await _provider_model_cache_enabled_states(state)
        disabled_provider_attrs = _disabled_provider_model_cache_attrs_from_states(provider_enabled_states)
        if _is_disabled_provider_cache_origin_model(model, disabled_provider_attrs):
            raise HTTPException(status_code=403, detail="Model not found")
    if model is None or _is_arena_model(model) or is_generated_wrapper_model_id(
        target_model_id,
        pipe_function_id=pipe_function_id,
    ):
        raise HTTPException(status_code=403, detail="Model not found")

    model_info = await _get_target_db_model_record(target_model_id)
    base_model_id = _target_record_base_model_id(model_info)
    if base_model_id and base_model_id not in models:
        fallback_model_id = await _custom_model_fallback_model_id_compatible(
            request,
            models,
            pipe_function_id=pipe_function_id,
        )
        if fallback_model_id is None:
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
    state_overrides: dict[str, Any] = {
        "bypass_filter": True,
        "bypass_system_prompt": False,
    }
    body_metadata = body.get("metadata")
    if isinstance(body_metadata, dict):
        state_overrides["metadata"] = body_metadata
    inner_request = RequestStateProxy(request, **state_overrides)
    try:
        from open_webui.utils.chat import generate_chat_completion

        await _ensure_model_in_request_models(inner_request, str(body.get("model") or ""))
        response = await generate_chat_completion(
            inner_request,
            body,
            user=coerce_open_webui_user(user),
            bypass_filter=True,
            bypass_system_prompt=False,
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
    anchor_input: UsageAnchorInput | None = None,
    on_complete: Callable[[dict[str, Any]], Any] | None = None,
    on_terminal: Callable[[bool], Any] | None = None,
    track_request_usage: bool = True,
) -> StreamingResponse:
    if track_request_usage:
        clear_request_scoped_usage(
            request=request,
            chat_id=chat_id,
            message_id=message_id,
            wrapper_model_id=wrapper_model_id,
        )
    response = await _call_target_completion(request=request, user=user, body=body)
    prepared = await prepare_streaming_response(
        response,
        request=request,
        chat_id=chat_id,
        message_id=message_id,
        wrapper_model_id=wrapper_model_id,
    )
    return _attach_streaming_completion_observer(
        prepared,
        request=request if track_request_usage else None,
        chat_id=chat_id,
        message_id=message_id,
        wrapper_model_id=wrapper_model_id,
        anchor_input=anchor_input,
        on_complete=on_complete,
        on_terminal=on_terminal,
    )


async def _coerce_non_streaming_completion_response(
    response: Any,
    *,
    model_id: str,
    raw_usage_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def merge_raw_usage(payload: dict[str, Any]) -> None:
        if raw_usage_out is None:
            return
        raw_usage = _raw_usage_from_stream_payload(payload)
        if not raw_usage:
            return
        merged = _merge_usage_fields(raw_usage_out, raw_usage)
        raw_usage_out.clear()
        raw_usage_out.update(merged)

    if isinstance(response, dict):
        merge_raw_usage(response)
        if response.get("error") and is_retryable_context_error(response, status_code=400):
            raise RetryableContextOverflow(str(response.get("error")))
        return response

    if isinstance(response, StreamingResponse):
        parts: list[str] = []
        raw_usage: dict[str, Any] = {}
        tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        provider_error: dict[str, Any] | None = None

        def process_payload(payload: dict[str, Any]) -> None:
            nonlocal finish_reason, provider_error, raw_usage
            error_source = _stream_payload_error_source(payload)
            if error_source is not None:
                if is_retryable_context_error(error_source, status_code=400):
                    raise RetryableContextOverflow(_completion_error_message(error_source))
                provider_error = _error_response(_completion_error_message(error_source), code="provider_error")
                return
            chunk_usage = _raw_usage_from_stream_payload(payload)
            if chunk_usage:
                raw_usage = _merge_usage_fields(raw_usage, chunk_usage)
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
        is_sse_response = "text/event-stream" in str(media_type).lower()
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
        if raw_usage_out is not None:
            raw_usage_out.update(raw_usage)
        return _chat_completion_message_response(
            model_id,
            "".join(parts).strip(),
            usage=_normalize_usage(raw_usage) if raw_usage else None,
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
            merge_raw_usage(parsed)
            return parsed
        return _error_response(_completion_error_message(parsed), code="provider_error")

    if isinstance(response, BaseModel):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            merge_raw_usage(dumped)
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
    chat_id: str | None = None,
    message_id: str | None = None,
    wrapper_model_id: str | None = None,
    anchor_input: UsageAnchorInput | None = None,
    on_complete: Callable[[dict[str, Any]], Any] | None = None,
    track_request_usage: bool = True,
) -> dict[str, Any]:
    if track_request_usage:
        clear_request_scoped_usage(
            request=request,
            chat_id=chat_id,
            message_id=message_id,
            wrapper_model_id=wrapper_model_id,
        )
    response = await _call_target_completion(request=request, user=user, body=body)
    raw_usage: dict[str, Any] = {}
    coerced = await _coerce_non_streaming_completion_response(
        response,
        model_id=str(body.get("model") or ""),
        raw_usage_out=raw_usage,
    )
    error_source = _stream_payload_error_source(coerced)
    if error_source is not None:
        if is_retryable_context_error(error_source, status_code=400):
            raise RetryableContextOverflow(_completion_error_message(error_source))
        if coerced.get("error"):
            return coerced
        return _error_response(_completion_error_message(error_source), code="provider_error")
    if coerced.get("error"):
        return coerced
    if raw_usage and track_request_usage:
        store_request_scoped_usage(
            request=request,
            chat_id=chat_id,
            message_id=message_id,
            wrapper_model_id=wrapper_model_id,
            usage=raw_usage,
            anchor_input=anchor_input,
        )
    has_tool_call = _responses_output_has_tool_call(coerced)
    choices = coerced.get("choices")
    if isinstance(choices, list):
        has_tool_call = has_tool_call or any(_choice_has_tool_call(choice) for choice in choices)
    if has_tool_call or on_complete is None:
        return coerced
    completion = dict(coerced)
    completion["raw_usage"] = raw_usage or None
    if raw_usage:
        completion["usage"] = _normalize_usage(raw_usage)
    with suppress(Exception):
        result = on_complete(completion)
        if inspect.isawaitable(result):
            await result
    return coerced


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
    file_context_enabled: bool = True,
    summary_prompt: str | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    ref_projection_plan: RefProjectionPlan | None = None,
    ref_mode_active: bool | None = None,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> tuple[dict[str, Any], bool, int]:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return body, False, 0

    tool_cut = select_tool_result_compaction_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    cut = select_safe_message_cut(
        messages,
        transient_message_patterns=transient_message_patterns,
    )
    chat_id = str(metadata.get("chat_id") or "")
    user_id = str((user.get("id") if isinstance(user, dict) else getattr(user, "id", "")) or "")

    if tool_cut is not None:
        tool_compaction_prefix_count = 0
        try:
            compacted_messages, did_compact_tools, tool_compaction_prefix_count = await _compact_retry_tool_results(
                request=request,
                user=user,
                metadata=metadata,
                pipe_function_id=pipe_function_id,
                summary_model_id=summary_model_id,
                base_body=body,
                messages=messages,
                summary_tool_policy=summary_tool_policy,
                file_context_enabled=file_context_enabled,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
                summary_prompt=summary_prompt,
                transient_message_patterns=transient_message_patterns,
                ref_projection_plan=ref_projection_plan,
                ref_mode_active=ref_mode_active,
                checkpoint_profile_hash=checkpoint_profile_hash,
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
            history_source_identity_count = _source_identity_message_count(
                cut.summarization_prefix,
                transient_message_patterns=transient_message_patterns,
            )
            if history_source_identity_count <= 0:
                raise
            history_summary_meta = build_checkpoint_summary_meta(
                cut.summarization_prefix,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
                transient_message_patterns=transient_message_patterns,
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
                preserved_system_message=cut.preserved_system_message,
                summary_meta=history_summary_meta,
                summary_tool_policy=summary_tool_policy,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
                file_context_enabled=file_context_enabled,
                summary_prompt=summary_prompt,
                transient_message_patterns=transient_message_patterns,
                ref_projection_plan=ref_projection_plan,
                ref_mode_active=ref_mode_active,
                checkpoint_profile_hash=checkpoint_profile_hash,
            )
            history_prefix_file_fingerprint_resolver = await _build_prefix_file_fingerprint_resolver(
                request,
                metadata,
                cut.summarization_prefix,
                transient_message_patterns=transient_message_patterns,
            )
            history_checkpoint = await _lookup_ready_checkpoint_for_source(
                request=request,
                user_id=user_id,
                chat_id=chat_id,
                pipe_function_id=pipe_function_id,
                source_messages=cut.summarization_prefix,
                prefix_file_fingerprint=_resolve_fingerprint(
                    history_prefix_file_fingerprint_resolver,
                    history_source_identity_count,
                ),
                file_backed_image_db_chain=_prefix_file_fingerprint_resolver_db_chain(
                    history_prefix_file_fingerprint_resolver
                ),
                transient_message_patterns=transient_message_patterns,
                checkpoint_profile_hash=checkpoint_profile_hash,
            )
            if history_checkpoint is None:
                raise RuntimeError("History checkpoint was not available after creation")
            compacted_messages, did_compact_tools, tool_compaction_prefix_count = await _compact_retry_tool_results(
                request=request,
                user=user,
                metadata=metadata,
                pipe_function_id=pipe_function_id,
                summary_model_id=summary_model_id,
                base_body=body,
                messages=messages,
                parent_checkpoint=history_checkpoint,
                file_context_enabled=file_context_enabled,
                summary_tool_policy=summary_tool_policy,
                historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                historical_message_excerpt_count=historical_message_excerpt_count,
                summary_prompt=summary_prompt,
                transient_message_patterns=transient_message_patterns,
                ref_projection_plan=ref_projection_plan,
                ref_mode_active=ref_mode_active,
                checkpoint_profile_hash=checkpoint_profile_hash,
            )
        if did_compact_tools:
            compacted = _copy_body_preserving_metadata(body)
            compacted["messages"] = compacted_messages
            compacted.pop("previous_response_id", None)
            return compacted, True, tool_compaction_prefix_count

    if not cut.summarization_prefix:
        return body, False, 0
    source_identity_count = _source_identity_message_count(
        cut.summarization_prefix,
        transient_message_patterns=transient_message_patterns,
    )
    if source_identity_count <= 0:
        return body, False, 0

    tail_transient_mask = _transient_message_mask(cut.tail_messages, transient_message_patterns)
    latest_user = next(
        (
            message
            for index, message in reversed(list(enumerate(cut.tail_messages)))
            if message.get("role") == "user"
            and _is_source_identity_message(
                message,
                transient_message_patterns=transient_message_patterns,
                transient_message_mask=tail_transient_mask,
                index=index,
            )
        ),
        None,
    )
    if latest_user is None:
        raise UnsupportedCompactionInput("Cannot compact safely without retaining the active latest user message")

    if not user_id or not _chat_id_supported(chat_id):
        return body, False, 0

    prefix_file_fingerprint_resolver = await _build_prefix_file_fingerprint_resolver(
        request,
        metadata,
        cut.summarization_prefix,
        require_file_context_chain=file_context_enabled,
        transient_message_patterns=transient_message_patterns,
    )
    file_backed_image_db_chain = _prefix_file_fingerprint_resolver_db_chain(prefix_file_fingerprint_resolver)
    prefix_identity_fingerprint = _resolve_fingerprint(
        prefix_file_fingerprint_resolver,
        source_identity_count,
    )
    pending_checkpoint = await _lookup_pending_checkpoint_for_source_prefix(
        request=request,
        user_id=user_id,
        chat_id=chat_id,
        pipe_function_id=pipe_function_id,
        source_messages=cut.summarization_prefix,
        prefix_file_fingerprint_resolver=prefix_file_fingerprint_resolver,
        transient_message_patterns=transient_message_patterns,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )
    if pending_checkpoint is not None and not _checkpoint_matches_exact_source(
        pending_checkpoint,
        cut.summarization_prefix,
        prefix_file_fingerprint=prefix_identity_fingerprint,
        file_backed_image_db_chain=file_backed_image_db_chain,
        transient_message_patterns=transient_message_patterns,
    ):
        ready_checkpoint = await _wait_for_pending_checkpoint_ready(pending_checkpoint)
        if ready_checkpoint is not None:
            ready_count = int(ready_checkpoint.get("source_message_count") or 0)
            ready_raw_count = _raw_prefix_len_for_source_count(
                cut.summarization_prefix,
                ready_count,
                transient_message_patterns=transient_message_patterns,
            )
            if ready_raw_count is not None:
                compacted = _copy_body_preserving_metadata(body)
                compacted["messages"] = replace_prefix_with_parent_checkpoint_and_delta(
                    cut,
                    ready_checkpoint,
                    prefix_file_fingerprint=_resolve_fingerprint(
                        prefix_file_fingerprint_resolver,
                        ready_count,
                    ),
                    file_backed_image_db_chain=file_backed_image_db_chain,
                    transient_message_patterns=transient_message_patterns,
                    historical_message_excerpt_bytes=historical_message_excerpt_bytes,
                    historical_message_excerpt_count=historical_message_excerpt_count,
                )
                compacted.pop("previous_response_id", None)
                return compacted, True, ready_count

    summary_meta = build_checkpoint_summary_meta(
        cut.summarization_prefix,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        transient_message_patterns=transient_message_patterns,
    )

    compacted = _copy_body_preserving_metadata(body)
    compaction_prefix_count_for_return: int
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
            preserved_system_message=cut.preserved_system_message,
            summary_meta=summary_meta,
            summary_tool_policy=summary_tool_policy,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            file_context_enabled=file_context_enabled,
            summary_prompt=summary_prompt,
            transient_message_patterns=transient_message_patterns,
            ref_projection_plan=ref_projection_plan,
            ref_mode_active=ref_mode_active,
            checkpoint_profile_hash=checkpoint_profile_hash,
        )
        compacted["messages"] = replace_prefix_with_summary(
            messages,
            cut,
            summary_text,
            summary_meta,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
            transient_message_patterns=transient_message_patterns,
        )
        compaction_prefix_count_for_return = source_identity_count
    except ParentCheckpointExtensionFailed as exc:
        parent_count = int(exc.parent.get("source_message_count") or 0)
        parent_raw_count = _raw_prefix_len_for_source_count(
            cut.summarization_prefix,
            parent_count,
            transient_message_patterns=transient_message_patterns,
        )
        if parent_raw_count is None:
            raise UnsupportedCompactionInput(
                "Parent checkpoint cannot be applied safely because its source boundary is invalid",
                code="unsafe_checkpoint_parent",
            ) from exc
        compacted["messages"] = replace_prefix_with_parent_checkpoint_and_delta(
            cut,
            exc.parent,
            prefix_file_fingerprint=_resolve_fingerprint(
                prefix_file_fingerprint_resolver,
                parent_count,
            ),
            file_backed_image_db_chain=file_backed_image_db_chain,
            transient_message_patterns=transient_message_patterns,
            historical_message_excerpt_bytes=historical_message_excerpt_bytes,
            historical_message_excerpt_count=historical_message_excerpt_count,
        )
        compaction_prefix_count_for_return = parent_count
    compacted.pop("previous_response_id", None)
    return compacted, True, compaction_prefix_count_for_return


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
    file_context_enabled: bool = True,
    transient_message_patterns: TransientMessagePatterns | None = None,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> tuple[dict[str, Any], bool, int]:
    source_body = _task_history_source_body_for_compaction(body, metadata)
    if source_body is None:
        return body, False, 0
    compacted_source, compacted, compaction_prefix_count = await _compact_body_with_reusable_checkpoint(
        request=request,
        user=user,
        metadata=metadata,
        body=source_body,
        pipe_function_id=pipe_function_id,
        match=match,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        file_context_enabled=file_context_enabled,
        transient_message_patterns=transient_message_patterns,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )
    if not compacted:
        return body, False, 0
    rebuilt = await _rebuild_task_body_from_compacted_history(
        request=request,
        user=user,
        base_body=body,
        metadata=metadata,
        compacted_history_messages=compacted_source["messages"],
    )
    if rebuilt is None:
        raise RuntimeError("Task prompt could not be rebuilt from compacted history")
    return rebuilt, True, compaction_prefix_count


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
    file_context_enabled: bool = True,
    summary_prompt: str | None = None,
    transient_message_patterns: TransientMessagePatterns | None = None,
    ref_projection_plan: RefProjectionPlan | None = None,
    ref_mode_active: bool | None = None,
    checkpoint_profile_hash: str = ACTIVE_CHECKPOINT_PROFILE_HASH,
) -> tuple[dict[str, Any], bool, int]:
    source_body = _task_history_source_body_for_compaction(body, metadata)
    if source_body is None:
        return body, False, 0
    compacted_source, compacted, compaction_prefix_count = await _compact_body(
        request=request,
        user=user,
        metadata=metadata,
        body=source_body,
        pipe_function_id=pipe_function_id,
        target_model_id=target_model_id,
        summary_model_id=summary_model_id,
        historical_message_excerpt_bytes=historical_message_excerpt_bytes,
        historical_message_excerpt_count=historical_message_excerpt_count,
        file_context_enabled=file_context_enabled,
        summary_tool_policy=summary_tool_policy,
        summary_prompt=summary_prompt,
        transient_message_patterns=transient_message_patterns,
        ref_projection_plan=ref_projection_plan,
        ref_mode_active=ref_mode_active,
        checkpoint_profile_hash=checkpoint_profile_hash,
    )
    if not compacted:
        return body, False, 0
    rebuilt = await _rebuild_task_body_from_compacted_history(
        request=request,
        user=user,
        base_body=body,
        metadata=metadata,
        compacted_history_messages=compacted_source["messages"],
    )
    if rebuilt is None:
        raise RuntimeError("Task prompt could not be rebuilt from compacted history")
    return rebuilt, True, compaction_prefix_count


class Pipe:
    class Valves(BaseModel):
        wrapper_model_name_template: str = Field(
            default="auto",
            description=(
                "Wrapper display name template. Use 'auto' to show the raw target name when hide_wrapped_target_models is on, "
                "otherwise '{target_name} (AutoCompact)'. Available placeholders: {target_name}, {target_id}. "
                "Unsupported placeholders or Python format syntax fall back to the same name as 'auto'. "
                "Blank or whitespace-only values are also treated as 'auto'. "
                "With 'auto' and hidden targets, normal chat model pickers stay clean, but Workspace base-model dropdowns may still "
                "show hidden raw targets with the same name; use an explicit template such as '{target_name} (AutoCompact)' if you "
                "need those dropdowns disambiguated."
            ),
        )
        include_model_patterns: str = Field(
            default="",
            description=(
                "Comma/newline-separated patterns matched against the full target model id or name. "
                "Only * and ? are wildcards; all other characters are literal. Empty wraps every eligible "
                "target model except AutoCompact wrappers from any Pipe, models/presets based on this Pipe's "
                "own wrappers, and arena models; other pipe-backed models can still be wrapped."
            ),
        )
        exclude_model_patterns: str = Field(
            default="",
            description=(
                "Comma/newline-separated patterns matched against the full target model id or name. "
                "Only * and ? are wildcards; all other characters are literal. Empty excludes nothing; "
                "applied after include_model_patterns, so matches are not wrapped."
            ),
        )
        hide_wrapped_target_models: bool = Field(
            default=False,
            description=(
                "Automatically set meta.hidden=true on raw target models wrapped by this pipe so chat model pickers show compact wrappers instead. "
                "Hidden target models remain available for admin/workspace base-model configuration. Use one AutoCompact Pipe per environment for "
                "this hiding feature; other AutoCompact copies with different function IDs are not coordinated for target restoration. "
                "Before deleting or disabling this Pipe, "
                "turn this off and let the model list refresh once so targets hidden by this Pipe are restored to their previous visibility; "
                "targets that were already hidden or owned by another Pipe's hide marker intentionally stay hidden. Otherwise raw models may remain hidden."
            ),
        )
        summary_model: str = Field(
            default="",
            description=(
                "Open WebUI model used for internal summarization. Keep this on Default to reuse the same "
                "underlying target model this wrapper calls; that keeps summaries aligned with the model in "
                "use and tries to preserve prompt-cache reuse where possible. Switching to Custom and "
                "selecting a specific model does NOT check that model's per-user access, and internal "
                "summary requests bypass filters; ensure the selected model is acceptable for all users "
                "who can reach any wrapped target."
            ),
            json_schema_extra={
                "input": {
                    "type": "select",
                    "options": "get_summary_model_options",
                }
            },
        )
        ref_exec_enabled: bool = Field(
            default=False,
            description="Enable externalized refs for eligible native-tool text in supported durable chats.",
        )
        ref_substitution_threshold_tokens: int = Field(
            default=10_000,
            ge=1_000,
            description="Externalize native-tool text at this exact token threshold or above 65,536 UTF-8 bytes. Head/tail previews, including their recovery marker, stay below this token threshold and within 65,536 UTF-8 bytes.",
        )
        trigger_input_tokens: int = Field(
            default=DEFAULT_TRIGGER_INPUT_TOKENS,
            ge=1,
            description=(
                "Input-token threshold applied to the current candidate estimate to decide whether "
                f"foreground compaction is required. The default {DEFAULT_TRIGGER_INPUT_TOKENS:,} leaves "
                "output-token headroom in a 256k context window. Very low values are outside this Pipe's "
                "intended operating range: reaching the threshold triggers a foreground compaction attempt, "
                "but the resulting request is not guaranteed to fall below it, and required active input may "
                "still exceed the model's context limit and prevent continuation. Override per model with "
                "per_model_overrides_json."
            ),
        )
        soft_trigger_ratio: float = Field(
            default=DEFAULT_SOFT_TRIGGER_RATIO,
            ge=0,
            lt=1,
            description=(
                "Ratio of the effective trigger_input_tokens that starts asynchronous background "
                "summary generation without pausing the current request. Before forwarding, this uses the "
                "candidate estimate; after a target response completes, that response's own input-plus-output "
                "usage can also start it as a proxy for the next request's input when it is >= soft and < hard. "
                "The summary is saved as a checkpoint and reused by "
                "a later request if ready. Set 0, or a value that rounds to 0, to disable. Override per "
                "model with soft_trigger_ratio in per_model_overrides_json."
            ),
        )
        per_model_overrides_json: str = Field(
            default="",
            description=(
                "Optional JSON object containing sparse per-target-model setting overrides. Currently supports "
                "trigger_input_tokens and soft_trigger_ratio. Shape: "
                '{"overrides": [{"model_patterns": ["claude-*", "Claude *"], "trigger_input_tokens": 160000, "soft_trigger_ratio": 0.75}]}. '
                "model_patterns match against target model id/name (same as include/exclude); only * (any run) "
                'and ? (single char) are wildcards and everything else is literal, so "[" needs no escaping: '
                '"claude-opus-4-8[1m]" matches that exact id and "*[1m]" matches every id ending in "[1m]". '
                "Each override requires non-empty model_patterns and at least one of trigger_input_tokens (integer >= 1) "
                "or soft_trigger_ratio (finite number >= 0 and < 1); optional schema_version must be 1. For each setting, "
                "overrides are evaluated top-to-bottom and the first matching override containing that setting wins. Empty disables overrides. "
                "Invalid JSON, non-object roots, unknown keys, or invalid shapes are rejected on save."
            ),
        )
        transient_message_patterns: str = Field(
            default="",
            description=(
                "Newline-separated Python regex patterns, validated on save. A "
                "user-role message is treated exactly like an injected system message "
                "(kept in requests and summarization input, excluded from checkpoint "
                "identity) when any pattern matches from the start of its first text "
                "part (leading whitespace skipped; re.match semantics, so anchor the "
                "end with \\Z to require the whole first text part to be the injected block). "
                "Example: (?s)<SYSTEM_CONTEXT>.*</SYSTEM_CONTEXT>\\s*\\Z  "
                "Keep patterns simple and linear; they run against the entire first text part. "
                "Empty disables this."
            ),
        )
        force_include_usage: bool = Field(
            default=True,
            description="Default-on convenience setting: add stream_options.include_usage=true to streaming target requests so supporting providers return usage even if per-model usage was not enabled. Disable to manage usage per model.",
        )
        compact_task_prompts_from_task_body: bool = Field(
            default=False,
            description=(
                "For supported Open WebUI task requests only, rebuild the task prompt from metadata.task_body "
                "so checkpoints can be reused. If upstream filters or pipelines rewrite body.messages without "
                "also updating metadata.task_body, those rewrites are not included in the rebuilt task prompt."
            ),
        )
        summary_tool_policy: SummaryToolPolicy = Field(
            default="fallback_on_tool_call",
            description=(
                "Controls summary requests for tool-enabled chats. Forced tool_choice/function_call is disabled in all modes. "
                "Recommended default fallback_on_tool_call keeps tools/functions on the first request to keep the request shape "
                "stable and avoid disrupting prompt-cache reuse where possible, then retries without tools/functions only if the summary model emits a tool call. "
                "always_strip removes tools/functions before the first request. error_on_tool_call keeps tools/functions and fails on tool calls."
            ),
        )
        summary_prompt: str = Field(
            default=SUMMARY_PROMPT,
            description=(
                "Admin-editable auto-compaction checkpoint summary prompt sent to the summary model. "
                "Leave empty or whitespace-only to fall back to the built-in prompt. This value is applied ONLY when "
                "generating NEW checkpoint summaries; existing durable checkpoints are always reused as-is regardless of "
                "prompt changes, so editing it never invalidates or re-identifies ready checkpoints. When file context "
                "is prepended for prefix files, it is prepended to the effective prompt."
            ),
        )
        historical_message_excerpt_bytes: int = Field(
            default=DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES,
            ge=1,
            description=(
                "Maximum UTF-8 bytes per saved excerpt of each recent historical user message from "
                "the summarized prefix. New checkpoints store middle-truncated excerpts; saved "
                "excerpts are reused unchanged."
            ),
        )
        historical_message_excerpt_count: int = Field(
            default=DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT,
            ge=0,
            description=(
                "Maximum number of recent historical user-message excerpts from the summarized "
                "prefix saved in new checkpoints and inserted into compacted context. Set 0 to stop "
                "saving excerpts in new checkpoints; excerpts already stored in existing checkpoints "
                "are still reused."
            ),
        )
        token_status_visibility: Literal["compaction_only", "always"] = Field(
            default="compaction_only",
            description=(
                "When to emit token-count status. 'compaction_only' (default) emits token "
                "info only on existing compaction lifecycle events (compacting/compacted/"
                "skipped/retry/prefetching/prefetched/failed), keeping normal requests silent. "
                "'always' additionally emits one token-pressure status per non-summary request "
                "with a known count (action auto_compaction_status), including requests that later "
                "compact. Has no effect on compaction decisions, only display."
            ),
        )
        token_status_detail: Literal["before", "before_after"] = Field(
            default="before",
            description=(
                "Token detail shown on compaction lifecycle status events. 'before' (default) shows the "
                "pre-compaction count only, using values already computed for thresholding "
                "(zero added cost). 'before_after' attempts to show the full post-compaction count on "
                "'compacted' events by estimating the compacted body once, and the rendered summary-message "
                "count on completed 'prefetched' events, when those estimates are available."
            ),
        )
        token_status_show_usage_and_estimate: bool = Field(
            default=False,
            description=(
                "When true, include the observed input-plus-output token total from the preceding target "
                "response and the current candidate input estimate in token status payloads when available. "
                "In a tool loop the values are staggered: a status shows usage for the preceding target "
                "request and an estimate for the current one. With token_status_visibility='always', compare "
                "the estimate for a candidate forwarded without further compaction or retry with the usage in "
                "the following target request's status, not with the usage shown beside it. The observed total "
                "includes reported output tokens, while the estimate may "
                "combine an input count from an earlier target response, held in request-local state or in "
                "this Pipe's durable usage anchor, with a locally estimated delta; therefore the gap is not "
                "a pure full-body tiktoken accuracy figure. The first target request and some retry or "
                "prefetch paths may show only one value. If no decision estimate is available but observed "
                "usage exists, this Valve attempts a local display-only estimate; compaction decisions are "
                "unchanged."
            ),
        )

        @field_validator("per_model_overrides_json")
        @classmethod
        def _validate_per_model_overrides_json(cls, value: str) -> str:
            parse_per_model_overrides(value)
            return value

        @field_validator("transient_message_patterns")
        @classmethod
        def _validate_transient_message_patterns(cls, value: str) -> str:
            parse_transient_message_patterns(value)
            return value

        @classmethod
        def get_summary_model_options(cls) -> list[dict[str, str]]:
            pipe_function_id = pipe_function_id_from_module_name(getattr(cls, "__module__", None))
            refresh_latest_models_cache_from_app_state()
            return build_summary_model_options(
                _LATEST_MODELS_CACHE, pipe_function_id=pipe_function_id
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
        config_values = await _open_webui_config_get_many(*MODEL_LISTING_CONFIG_KEYS)
        provider_enabled_states = await _provider_model_cache_enabled_states(state, config_values)
        provider_cache_attrs = _enabled_provider_model_cache_attrs_from_states(provider_enabled_states)
        disabled_provider_attrs = _disabled_provider_model_cache_attrs_from_states(provider_enabled_states)
        initial_provider_caches = {attr: getattr(state, attr, None) for attr in provider_cache_attrs}
        model_candidates = _iter_cache_models_from_state(state, disabled_provider_attrs=disabled_provider_attrs)
        pending_provider_cache_attrs = _provider_model_cache_refresh_pending_attrs(provider_cache_attrs)
        if pending_provider_cache_attrs and not _provider_model_caches_ready(
            state,
            initial_provider_caches,
            pending_provider_cache_attrs,
        ):
            try:
                model_candidates = await _wait_for_provider_model_caches(
                    state,
                    initial_provider_caches=initial_provider_caches,
                    provider_cache_attrs=provider_cache_attrs,
                    disabled_provider_attrs=disabled_provider_attrs,
                )
            except Exception:
                LOG.exception("Failed to wait for provider model caches during AutoCompact pipe listing")
        persisted_models: list[Any] | None = None
        try:
            from open_webui.models.models import Models

            persisted_models = list(await Models.get_all_models())
        except Exception:
            LOG.exception("Failed to load persisted Workspace models during AutoCompact pipe listing")
        else:
            arena_model_ids = await _effective_arena_model_ids(state, config_values)
            seen_model_ids = {
                model_id
                for model in model_candidates
                if (model_id := _model_id(model)) is not None
            }
            for persisted_model in persisted_models:
                persisted_id = _record_field(persisted_model, "id")
                if (
                    not isinstance(persisted_id, str)
                    or not persisted_id
                    or persisted_id in seen_model_ids
                    or _record_field(persisted_model, "base_model_id") is None
                    or _record_field(persisted_model, "is_active") is False
                ):
                    continue
                candidate = _payload_dict(persisted_model)
                if _record_field(persisted_model, "base_model_id") in arena_model_ids:
                    candidate["owned_by"] = "arena"
                model_candidates.append(candidate)
                seen_model_ids.add(persisted_id)
        targets = filter_target_models(model_candidates, self.valves, pipe_function_id=pipe_function_id)
        update_latest_models_cache(model_candidates)
        try:
            await sync_wrapper_model_records(
                pipe_function_id=pipe_function_id,
                target_models=targets,
                valves=self.valves,
                existing_models=persisted_models,
                target_models_complete=persisted_models is not None,
            )
        except Exception:
            LOG.exception("Failed to sync AutoCompact wrapper model records")

        entries: list[dict[str, str]] = []
        hide_wrapped_target_models = bool(getattr(self.valves, "hide_wrapped_target_models", False))
        for target in targets:
            target_contract = build_target_model_contract(target, None)
            entries.append(
                {
                    "id": encode_target_model_id(target_contract.id),
                    "name": format_wrapper_model_name(
                        target_contract,
                        template=getattr(self.valves, "wrapper_model_name_template", "auto"),
                        hide_wrapped_target_models=hide_wrapped_target_models,
                    ),
                }
            )
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
        await _refresh_tiktoken_encoding_config(__request__)

        is_streaming = body.get("stream") is True
        incoming_model_id = str(body.get("model") or "")
        selected_wrapper_id = incoming_model_id
        injected_assistant_message_id = (
            str(__metadata__.get("message_id") or "")
            if isinstance(__metadata__, dict)
            else ""
        )
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
        task_name = _normalized_task_name(metadata.get("task"))
        is_task_request = bool(task_name) or metadata_task_body is not None
        if task_name == OFFICIAL_CONTEXT_COMPACTION_TASK:
            return _core_context_compaction_conflict_response()
        is_summary_task = task_name == INTERNAL_SUMMARY_TASK
        if not is_summary_task and await _core_context_compaction_enabled():
            return _core_context_compaction_conflict_response()

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
        update_latest_models_cache(models.values())
        summary_model_id = identity.target_model_id
        if not is_summary_task:
            try:
                summary_model_id = validate_summary_model_id(
                    self.valves.summary_model,
                    identity.target_model_id,
                    models,
                )
            except ValueError as exc:
                return _error_response(str(exc), code="invalid_summary_model")

        inner = _copy_body_preserving_metadata(body)
        target_route = await _resolve_core_chat_model_route(
            __request__,
            identity.target_model_id,
            pipe_function_id=pipe_function_id,
            metadata=metadata,
            user=user,
        )
        try:
            target_route, selected_arena_model_id = await _resolve_arena_chat_model_route_with_access(
                request=__request__,
                user=user,
                models=models,
                route=target_route,
                original_model_id=identity.target_model_id,
                pipe_function_id=pipe_function_id,
            )
        except Exception:
            return _error_response("Model not found", code="model_access_denied")
        inner["model"] = target_route.model_id
        inner["metadata"] = _copy_metadata_preserving_references(metadata)
        attached_registry = preflight_attached_ref_registry(__metadata__, __tools__)
        if attached_registry is not None:
            inner["metadata"]["tools"] = attached_registry
        if selected_arena_model_id:
            inner["metadata"]["selected_model_id"] = selected_arena_model_id
        if is_streaming and self.valves.force_include_usage:
            inner = inject_stream_usage_options(inner, force_include_usage=True)
        usage_anchor_dropped_message_keys = target_route.usage_anchor_dropped_message_keys

        # query_generation runs Core generate_queries, which may route the task
        # model back into this wrapper (TASK_MODEL pointing here). Injecting
        # target file context then would recurse via chat_completion_files_handler.
        is_query_generation_task = task_name == TASKS.QUERY_GENERATION.value
        # Stateful Responses continuations expose only the new tool-loop items;
        # sizing, compacting, or reinjecting RAG against that partial history is unsafe.
        is_stateful_responses_continuation = bool(inner.get("previous_response_id"))
        if is_stateful_responses_continuation:
            LOG.debug("Auto-compaction skipped: previous_response_id")
        supported_context = (
            _chat_id_supported(chat_id)
            and not is_summary_task
            and not is_stateful_responses_continuation
            and (is_task_request or bool(metadata.get("message_id")))
        )
        original_metadata_tools = attached_registry
        provider_schema_supported = (
            ("tools" not in body or isinstance(body.get("tools"), list))
            and ("functions" not in body or isinstance(body.get("functions"), list))
        )
        metadata_params = metadata.get("params")
        native_function_calling = (
            not is_task_request
            and not is_summary_task
            and isinstance(metadata_params, dict)
            and core_function_calling_is_native(
                _core_function_calling_generation(),
                metadata_params.get("function_calling"),
            )
        )
        user_id = str((user or {}).get("id") or "")
        ref_binding_key = RefBindingKey(
            user_id=user_id,
            chat_id=str(chat_id or ""),
            user_message_id=str(metadata.get("user_message_id") or ""),
            assistant_message_id=injected_assistant_message_id,
            incoming_model_id=incoming_model_id,
            base_pipe_id=identity.pipe_function_id,
            profile_hash=compute_profile_hash(),
            branch_anchor=str(metadata.get("user_message_id") or ""),
        )
        registry_available = (
            original_metadata_tools is not None
            and _ref_registry_available(
                __request__,
                ref_binding_key,
                original_metadata_tools,
            )
        )
        effective_ref_mode = resolve_ref_mode_preflight(
            RefModePreflight(
                valve_enabled=self.valves.ref_exec_enabled,
                native_function_calling=native_function_calling,
                durable_context=bool(supported_context),
                provider_schema_supported=provider_schema_supported,
                metadata_tools=original_metadata_tools,
                injected_tools=__tools__,
                registry_available=registry_available,
            )
        )
        checkpoint_profile_hash = checkpoint_profile_hash_for_ref_mode(
            ref_mode_active=effective_ref_mode.active
        )
        if self.valves.ref_exec_enabled and not effective_ref_mode.active:
            LOG.info(
                "Auto Compact ref mode inactive: %s",
                effective_ref_mode.reason,
            )
        ref_reservation: RefReservation | None = None
        task_source_body = (
            _task_history_source_body_for_compaction(inner, metadata)
            if supported_context and self.valves.compact_task_prompts_from_task_body
            else None
        )
        checkpoint_lookup_body = task_source_body if task_source_body is not None else _copy_body_preserving_metadata(inner)
        estimate_lookup_body = checkpoint_lookup_body
        try:
            compiled_transient_message_patterns = parse_transient_message_patterns(
                getattr(self.valves, "transient_message_patterns", "")
            )
            transient_message_patterns = (
                _TransientMessageMatcher(compiled_transient_message_patterns)
                if compiled_transient_message_patterns
                else None
            )
        except ValueError as exc:
            return _error_response(str(exc), code="invalid_transient_message_patterns")
        ref_projection_plan: RefProjectionPlan | None = None
        if effective_ref_mode.active:
            raw_messages = checkpoint_lookup_body.get("messages")
            if isinstance(raw_messages, list):
                try:
                    ref_projection_plan = await project_native_tool_texts(
                        raw_messages,
                        threshold_tokens=self.valves.ref_substitution_threshold_tokens,
                        request=__request__,
                    )
                except RefProjectionError as exc:
                    if ref_reservation is not None:
                        await release_ref_reservation(__request__, ref_reservation)
                    _log_ref_projection_failure(exc)
                    return _error_response(str(exc), code="ref_projection_failed")
        # Token decisions must reflect the body actually forwarded to the target.
        # For task-prompt compaction that is the rebuilt provider prompt (inner),
        # NOT the raw task history — even when the target opted out of file
        # context.  Checkpoint lookup / source hash keep using the clean
        # checkpoint_lookup_body (raw task history).
        if task_source_body is not None:
            estimate_lookup_body = inner

        # Inject file context into inner BEFORE token estimation so the
        # estimate reflects the actual forwarded payload (wrapper disables
        # Core's pre-pipe RAG).  Checkpoint lookup / source hashes keep using
        # checkpoint_lookup_body (clean, pre-RAG); estimate_lookup_body carries
        # the injected payload.  inner is used directly when no compaction
        # occurs.  When compaction runs, clean messages are restored for
        # the compaction cut, then re-injected with the correct prefix count.
        target_file_context_enabled = _target_model_supports_file_context(models, target_route.model_id)
        reusable_checkpoint_match = None
        # checkpoint_lookup_unavailable: set when the initial checkpoint lookup
        # failed because the DB was unavailable. We do NOT fail closed here:
        # for a request that turns out to be within limits we forward unchanged
        # (no checkpoint, no prefetch); we fail closed later (R3) only when
        # compaction is actually required to stay under the model limit. The
        # fail-closed branch returns a fixed message without database details.
        checkpoint_lookup_unavailable = False
        if supported_context:
            try:
                reusable_checkpoint_match = await _body_reusable_checkpoint_match(
                    request=__request__,
                    user=user,
                    metadata=metadata,
                    body=checkpoint_lookup_body,
                    pipe_function_id=identity.pipe_function_id,
                    transient_message_patterns=transient_message_patterns,
                    capture_logical_snapshot=effective_ref_mode.active,
                    checkpoint_profile_hash=checkpoint_profile_hash,
                )
            except Exception:
                checkpoint_lookup_unavailable = True
                LOG.warning(
                    "Auto-compaction checkpoint lookup failed (chat_id=%s); "
                    "request will be forwarded unchanged if within limits, "
                    "fail closed if compaction is required",
                    chat_id,
                    exc_info=True,
                )
        if effective_ref_mode.active and reusable_checkpoint_match is not None:
            ref_projection_plan = await extend_ref_projection_plan_with_checkpoint(
                ref_projection_plan,
                reusable_checkpoint_match.checkpoint,
                request=__request__,
                metadata=metadata,
                transient_message_patterns=transient_message_patterns,
                logical_snapshot=reusable_checkpoint_match.logical_snapshot,
            )
        pre_rag_messages: list[dict[str, Any]] | None = None
        pre_injected_file_context_sources = None
        if (
            not is_summary_task
            and not is_query_generation_task
            and not is_stateful_responses_continuation
            and target_file_context_enabled
            and reusable_checkpoint_match is None
        ):
            target_user_message_id_for_inject = str(metadata.get("user_message_id") or message_id or "") or None
            pre_rag_messages = copy.deepcopy(inner.get("messages") if isinstance(inner.get("messages"), list) else [])
            inner = await _inject_target_file_context(
                request=__request__,
                user=user,
                body=inner,
                chat_id=str(chat_id) if chat_id else None,
                current_message_id=target_user_message_id_for_inject,
                compaction_prefix_count=0,
                metadata_files=metadata.get("files"),
                metadata_user_message=metadata.get("user_message"),
                event_emitter=None,
                file_context_enabled=target_file_context_enabled,
                emit_source_events=False,
                transient_message_patterns=transient_message_patterns,
            )
            inner_metadata = inner.get("metadata")
            if isinstance(inner_metadata, dict):
                pre_injected_file_context_sources = inner_metadata.get("sources")
            # Token decisions use the injected provider body so file context is
            # included.  For task-prompt compaction the target still receives the
            # rebuilt provider prompt, so inner (not the raw task history) is the
            # correct estimate basis.
            estimate_lookup_body = inner

        request_usage = get_request_scoped_usage(
            request=__request__,
            chat_id=str(chat_id) if chat_id else None,
            message_id=str(message_id) if message_id else None,
            wrapper_model_id=selected_wrapper_id,
        )
        request_usage_anchor = get_request_scoped_usage_anchor(
            request=__request__,
            chat_id=str(chat_id) if chat_id else None,
            message_id=str(message_id) if message_id else None,
            wrapper_model_id=selected_wrapper_id,
        )
        total_tokens = _usage_total(request_usage)
        usage_source: str | None = "request" if total_tokens is not None else None

        durable_usage_anchor = None
        anchor_user_id = str((user or {}).get("id") or "")
        if (
            supported_context
            and not is_task_request
            and not checkpoint_lookup_unavailable
            and target_route.usage_anchor_shaping_hash is not None
            and anchor_user_id
            and chat_id
        ):
            parent_assistant_message_id = await _usage_anchor_parent_assistant_message_id(
                metadata=metadata,
                chat_id=str(chat_id),
            )
            if parent_assistant_message_id:
                try:
                    durable_usage_anchor = await lookup_usage_anchor(
                        request=__request__,
                        user_id=anchor_user_id,
                        chat_id=str(chat_id),
                        pipe_function_id=identity.pipe_function_id,
                        assistant_message_id=parent_assistant_message_id,
                    )
                except Exception:
                    LOG.warning(
                        "Auto-compaction usage anchor lookup failed (chat_id=%s, assistant_message_id=%s)",
                        chat_id,
                        parent_assistant_message_id,
                        exc_info=True,
                    )
            else:
                LOG.debug("Auto-compaction usage anchor miss: parent_assistant_unknown")
        target_model_for_limits = models.get(identity.target_model_id) or {
            "id": identity.target_model_id,
            "name": identity.target_model_id,
        }
        try:
            effective_trigger_input_tokens = resolve_trigger_input_tokens(self.valves, target_model_for_limits)
        except ValueError as exc:
            return _error_response(str(exc), code="invalid_per_model_overrides")
        effective_soft_trigger_input_tokens = resolve_soft_trigger_input_tokens(
            self.valves,
            target_model_for_limits,
            effective_trigger_input_tokens,
        )
        # --- candidate-payload estimate: the decision basis ---
        # total_tokens is an OBSERVATION of a past payload, not the size of the
        # candidate we are about to send. Hard/soft decisions are driven by a
        # candidate estimate (decision_total), never by raw observed usage.
        checkpoint_applied_estimate = None
        estimated_total_tokens = None
        prepared_reusable_key = None
        prepared_reusable_candidate = None
        prepared_reusable_forward_candidate = None
        prepared_reusable_prefix_count = 0
        prepared_reusable_source_events = None
        prepared_uncompacted_candidate = None
        prepared_uncompacted_forward_candidate = None

        async def apply_target_ref_projection(candidate: dict[str, Any]) -> dict[str, Any]:
            if not effective_ref_mode.active:
                return candidate
            _require_provider_bound_history_ref_manifests(candidate, ref_projection_plan)
            if ref_projection_plan is None:
                return candidate
            candidate_messages = candidate.get("messages")
            if not isinstance(candidate_messages, list):
                return candidate
            candidate["messages"] = await apply_ref_projection_plan(
                candidate_messages,
                ref_projection_plan,
            )
            apply_ref_projection_surfaces(
                candidate,
                ref_projection_plan,
                include_reader_schema=True,
            )
            return candidate

        def reusable_match_key(match: ReusableCheckpointMatch) -> tuple[Any, ...]:
            checkpoint = match.checkpoint or {}
            return (
                match.kind,
                match.source_kind,
                match.source_message_count,
                checkpoint.get("id"),
            )

        async def prepare_reusable_checkpoint_match(
            match: ReusableCheckpointMatch,
        ) -> tuple[dict[str, Any], dict[str, Any], int, Any] | None:
            nonlocal prepared_reusable_key
            nonlocal prepared_reusable_candidate
            nonlocal prepared_reusable_forward_candidate
            nonlocal prepared_reusable_prefix_count
            nonlocal prepared_reusable_source_events

            match_key = reusable_match_key(match)
            if prepared_reusable_key == match_key and prepared_reusable_candidate is not None:
                return (
                    prepared_reusable_candidate,
                    prepared_reusable_forward_candidate,
                    prepared_reusable_prefix_count,
                    prepared_reusable_source_events,
                )

            candidate = _copy_body_preserving_metadata(inner)
            if pre_rag_messages is not None:
                candidate["messages"] = copy.deepcopy(pre_rag_messages)
            try:
                if task_source_body is not None:
                    candidate, compacted, prefix_count = await _compact_task_body_with_reusable_checkpoint(
                        request=__request__,
                        user=user,
                        metadata=metadata,
                        body=candidate,
                        pipe_function_id=identity.pipe_function_id,
                        match=match,
                        historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                        historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                        file_context_enabled=target_file_context_enabled,
                        transient_message_patterns=transient_message_patterns,
                        checkpoint_profile_hash=checkpoint_profile_hash,
                    )
                else:
                    candidate, compacted, prefix_count = await _compact_body_with_reusable_checkpoint(
                        request=__request__,
                        user=user,
                        metadata=metadata,
                        body=candidate,
                        pipe_function_id=identity.pipe_function_id,
                        match=match,
                        historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                        historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                        file_context_enabled=target_file_context_enabled,
                        transient_message_patterns=transient_message_patterns,
                        checkpoint_profile_hash=checkpoint_profile_hash,
                    )
            except (SummaryFileContextUnavailable, RuntimeError):
                return None
            if not compacted:
                return None

            source_events = None
            if not is_summary_task and not is_query_generation_task:
                candidate = await _inject_target_file_context(
                    request=__request__,
                    user=user,
                    body=candidate,
                    chat_id=str(chat_id) if chat_id else None,
                    current_message_id=str(metadata.get("user_message_id") or message_id or "") or None,
                    compaction_prefix_count=prefix_count,
                    metadata_files=metadata.get("files"),
                    metadata_user_message=metadata.get("user_message"),
                    event_emitter=None,
                    file_context_enabled=target_file_context_enabled,
                    emit_source_events=False,
                    transient_message_patterns=transient_message_patterns,
                )
                candidate_metadata = candidate.get("metadata")
                if isinstance(candidate_metadata, dict):
                    source_events = candidate_metadata.get("sources")
            candidate = await apply_target_ref_projection(candidate)
            forward_candidate = _apply_resolved_model_route_params(
                candidate,
                models=models,
                route=target_route,
            )
            prepared_reusable_key = match_key
            prepared_reusable_candidate = candidate
            prepared_reusable_forward_candidate = forward_candidate
            prepared_reusable_prefix_count = prefix_count
            prepared_reusable_source_events = source_events
            return candidate, forward_candidate, prefix_count, source_events

        async def estimate_candidate(candidate: dict[str, Any]) -> int | None:
            if effective_ref_mode.active:
                _require_provider_bound_history_ref_manifests(
                    candidate,
                    ref_projection_plan,
                )
            token_candidate = _project_usage_anchor_token_body(
                candidate,
                dropped_message_keys=usage_anchor_dropped_message_keys,
            )
            if target_route.usage_anchor_shaping_hash is not None and request_usage_anchor is not None:
                request_anchor_estimate = await _estimate_body_tokens_from_usage_anchor(
                    request=__request__,
                    body=token_candidate,
                    anchor=request_usage_anchor,
                    usage_anchor_shaping_hash=target_route.usage_anchor_shaping_hash,
                    transient_message_patterns=transient_message_patterns,
                )
                if request_anchor_estimate is not None:
                    return request_anchor_estimate
            if target_route.usage_anchor_shaping_hash is not None and durable_usage_anchor is not None:
                durable_estimate = await _estimate_body_tokens_from_usage_anchor(
                    request=__request__,
                    body=token_candidate,
                    anchor=durable_usage_anchor,
                    usage_anchor_shaping_hash=target_route.usage_anchor_shaping_hash,
                    transient_message_patterns=transient_message_patterns,
                )
                if durable_estimate is not None:
                    return durable_estimate
            return await _estimate_provider_input_tokens_async(
                candidate,
                request=__request__,
                user=user,
                system_prompt=target_route.token_system_prompt,
                dropped_message_keys=usage_anchor_dropped_message_keys,
            )

        async def estimate_reusable_checkpoint_match(match: ReusableCheckpointMatch) -> int | None:
            prepared = await prepare_reusable_checkpoint_match(match)
            if prepared is None:
                return None
            return await estimate_candidate(prepared[1])

        try:
            if supported_context:
                if reusable_checkpoint_match is not None:
                    checkpoint_applied_estimate = await estimate_reusable_checkpoint_match(
                        reusable_checkpoint_match
                    )
                else:
                    if effective_ref_mode.active:
                        prepared_uncompacted_candidate = await apply_target_ref_projection(
                            _copy_body_preserving_metadata(estimate_lookup_body)
                        )
                        prepared_uncompacted_forward_candidate = (
                            _apply_resolved_model_route_params(
                                prepared_uncompacted_candidate,
                                models=models,
                                route=target_route,
                            )
                        )
                        estimate_candidate_body = prepared_uncompacted_forward_candidate
                    else:
                        estimate_candidate_body = _apply_resolved_model_route_params(
                            estimate_lookup_body,
                            models=models,
                            route=target_route,
                        )
                    estimated_total_tokens = await estimate_candidate(
                        estimate_candidate_body
                    )
        except RefProjectionError as exc:
            _log_ref_projection_failure(exc)
            return _error_response(str(exc), code="ref_projection_failed")
        # decision_total: checkpoint-applied estimate (the compacted body we
        # would actually forward) takes priority over the usage-anchor / full-body
        # estimate. It NEVER falls back to raw observed total_tokens.
        if checkpoint_applied_estimate is not None:
            decision_total = checkpoint_applied_estimate
        elif estimated_total_tokens is not None:
            decision_total = estimated_total_tokens
        else:
            decision_total = None

        def compute_threshold_decisions() -> tuple[bool, bool, bool]:
            # A reusable TASK checkpoint whose applied estimate cannot be computed
            # cannot be confirmed safe, so foreground compaction is still required.
            checkpoint_applied_unknown_needs_foreground = (
                reusable_checkpoint_match is not None
                and task_source_body is not None
                and checkpoint_applied_estimate is None
            )
            # hard/soft decisions look ONLY at decision_total (the candidate estimate).
            hard = (
                supported_context
                and (
                    (decision_total is not None and decision_total >= effective_trigger_input_tokens)
                    or checkpoint_applied_unknown_needs_foreground
                )
            )
            soft = (
                supported_context
                and not is_summary_task
                and effective_soft_trigger_input_tokens is not None
                and decision_total is not None
                and decision_total >= effective_soft_trigger_input_tokens
                and decision_total < effective_trigger_input_tokens
            )
            should = hard or reusable_checkpoint_match is not None
            return hard, soft, should

        hard_should_compact, soft_should_prefetch, should_compact = compute_threshold_decisions()
        if checkpoint_lookup_unavailable:
            # Fail closed unless we can positively confirm the candidate is
            # safely under the hard limit. hard_should_compact already encodes
            # "compaction required", but decision_total can be None (encoder
            # unavailable) — in that case hard is False yet we still must not
            # forward, because we cannot confirm the request is under limit.
            if (
                hard_should_compact
                or decision_total is None
                or decision_total >= effective_trigger_input_tokens
            ):
                return _error_response(
                    CHECKPOINT_STORE_UNAVAILABLE_MESSAGE,
                    code="checkpoint_unavailable",
                )
            # DB down + confirmed below the hard limit: forward unchanged.
            # Disable prefetch; R4 guards skip the late rechecks (which would
            # just hit the DB again). No checkpoint is created (R5): the
            # compaction block only runs when should_compact is True.
            soft_should_prefetch = False
        if (
            supported_context
            and not checkpoint_lookup_unavailable
            and reusable_checkpoint_match is None
            and (hard_should_compact or soft_should_prefetch)
        ):
            try:
                late_checkpoint_match = await _body_reusable_checkpoint_match(
                    request=__request__,
                    user=user,
                    metadata=metadata,
                    body=checkpoint_lookup_body,
                    pipe_function_id=identity.pipe_function_id,
                    transient_message_patterns=transient_message_patterns,
                    capture_logical_snapshot=effective_ref_mode.active,
                    checkpoint_profile_hash=checkpoint_profile_hash,
                )
            except Exception as exc:
                if not hard_should_compact and soft_should_prefetch:
                    LOG.warning(
                        "Auto-compaction late checkpoint lookup failed during soft prefetch check "
                        "(chat_id=%s); disabling soft prefetch: %s",
                        chat_id,
                        exc,
                        exc_info=True,
                    )
                    late_checkpoint_match = None
                    soft_should_prefetch = False
                else:
                    LOG.warning(
                        "Auto-compaction late checkpoint lookup failed during hard compaction "
                        "(chat_id=%s); failing closed",
                        chat_id,
                        exc_info=True,
                    )
                    return _error_response(
                        CHECKPOINT_STORE_UNAVAILABLE_MESSAGE,
                        code="checkpoint_unavailable",
                    )
            if late_checkpoint_match is not None:
                reusable_checkpoint_match = late_checkpoint_match
                if effective_ref_mode.active:
                    ref_projection_plan = await extend_ref_projection_plan_with_checkpoint(
                        ref_projection_plan,
                        late_checkpoint_match.checkpoint,
                        request=__request__,
                        metadata=metadata,
                        transient_message_patterns=transient_message_patterns,
                        logical_snapshot=late_checkpoint_match.logical_snapshot,
                    )
                prepared_reusable_key = None
                prepared_reusable_candidate = None
                prepared_reusable_forward_candidate = None
                prepared_reusable_prefix_count = 0
                prepared_reusable_source_events = None
                prepared_uncompacted_candidate = None
                prepared_uncompacted_forward_candidate = None
                try:
                    checkpoint_applied_estimate = (
                        await estimate_reusable_checkpoint_match(late_checkpoint_match)
                    )
                except RefProjectionError as exc:
                    _log_ref_projection_failure(exc)
                    return _error_response(str(exc), code="ref_projection_failed")
                # Once a checkpoint is reusable, the checkpoint-applied payload is
                # the only candidate that matters. If that estimate is unavailable,
                # do not fall back to the raw estimate that caused this late check.
                decision_total = checkpoint_applied_estimate
                hard_should_compact, soft_should_prefetch, should_compact = compute_threshold_decisions()

        status_estimate_tokens = None
        if (
            supported_context
            and self.valves.token_status_show_usage_and_estimate
            and decision_total is None
            and total_tokens is not None
        ):
            with suppress(Exception):
                status_estimate_tokens = await _estimate_provider_input_tokens_async(
                    estimate_lookup_body,
                    request=__request__,
                    user=user,
                    system_prompt=target_route.token_system_prompt,
                    dropped_message_keys=usage_anchor_dropped_message_keys,
                )
        display_token_context = _build_display_token_context(
            estimated_total_tokens=decision_total,
            total_tokens=total_tokens,
            effective_trigger_input_tokens=effective_trigger_input_tokens,
            effective_soft_trigger_input_tokens=effective_soft_trigger_input_tokens,
            usage_source=usage_source,
        )
        display_token_context = _display_context_with_estimate(display_token_context, status_estimate_tokens)
        show_usage_and_estimate = bool(self.valves.token_status_show_usage_and_estimate)
        should_compact = hard_should_compact or reusable_checkpoint_match is not None
        reusable_checkpoint_only = None
        if reusable_checkpoint_match is not None and not hard_should_compact:
            reusable_checkpoint_only = reusable_checkpoint_match
        if (
            self.valves.token_status_visibility == "always"
            and not is_summary_task
            and display_token_context.before is not None
        ):
            await emit_compaction_status(
                __event_emitter__,
                action="status",
                description=_description_with_token_suffix(
                    "Context pressure",
                    display_token_context,
                    show_usage_and_estimate=show_usage_and_estimate,
                ),
                done=True,
                tokens=_tokens_status_payload(
                    display_token_context,
                    after=None,
                    show_usage_and_estimate=show_usage_and_estimate,
                ),
            )
        if not hard_should_compact and soft_should_prefetch and not checkpoint_lookup_unavailable:
            if reusable_checkpoint_match is None or reusable_checkpoint_match.kind == "parent":
                try:
                    late_checkpoint_match = await _body_reusable_checkpoint_match(
                        request=__request__,
                        user=user,
                        metadata=metadata,
                        body=checkpoint_lookup_body,
                        pipe_function_id=identity.pipe_function_id,
                        transient_message_patterns=transient_message_patterns,
                        capture_logical_snapshot=effective_ref_mode.active,
                        checkpoint_profile_hash=checkpoint_profile_hash,
                    )
                except Exception as exc:
                    LOG.warning(
                        "Auto-compaction late checkpoint lookup failed during soft prefetch check "
                        "(chat_id=%s); disabling soft prefetch: %s",
                        chat_id,
                        exc,
                        exc_info=True,
                    )
                    late_checkpoint_match = None
                    soft_should_prefetch = False
                if late_checkpoint_match is not None:
                    late_checkpoint_is_better = (
                        reusable_checkpoint_match is None
                        or late_checkpoint_match.kind == "exact"
                        or late_checkpoint_match.source_message_count > reusable_checkpoint_match.source_message_count
                    )
                    if late_checkpoint_is_better:
                        reusable_checkpoint_match = late_checkpoint_match
                        if effective_ref_mode.active:
                            ref_projection_plan = await extend_ref_projection_plan_with_checkpoint(
                                ref_projection_plan,
                                late_checkpoint_match.checkpoint,
                                request=__request__,
                                metadata=metadata,
                                transient_message_patterns=transient_message_patterns,
                                logical_snapshot=late_checkpoint_match.logical_snapshot,
                            )
                        prepared_reusable_key = None
                        prepared_reusable_candidate = None
                        prepared_reusable_forward_candidate = None
                        prepared_reusable_prefix_count = 0
                        prepared_reusable_source_events = None
                        prepared_uncompacted_candidate = None
                        prepared_uncompacted_forward_candidate = None
                        try:
                            checkpoint_applied_estimate = (
                                await estimate_reusable_checkpoint_match(
                                    late_checkpoint_match
                                )
                            )
                        except RefProjectionError as exc:
                            _log_ref_projection_failure(exc)
                            return _error_response(
                                str(exc), code="ref_projection_failed"
                            )
                        decision_total = checkpoint_applied_estimate
                        hard_should_compact, soft_should_prefetch, should_compact = compute_threshold_decisions()
                        should_compact = hard_should_compact or reusable_checkpoint_match is not None
                        reusable_checkpoint_only = None
                        if reusable_checkpoint_match is not None and not hard_should_compact:
                            reusable_checkpoint_only = reusable_checkpoint_match
            if not hard_should_compact and soft_should_prefetch:
                _start_soft_compaction_prefetch(
                    request=__request__,
                    user=user,
                    metadata=metadata,
                    body=checkpoint_lookup_body,
                    pipe_function_id=identity.pipe_function_id,
                    summary_model_id=summary_model_id,
                    summary_tool_policy=self.valves.summary_tool_policy,
                    summary_prompt=self.valves.summary_prompt,
                    historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                    historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                    effective_trigger_input_tokens=effective_trigger_input_tokens,
                    effective_soft_trigger_input_tokens=effective_soft_trigger_input_tokens,
                    trigger_estimated_tokens=decision_total,
                    token_status_detail=self.valves.token_status_detail,
                    token_status_show_usage_and_estimate=self.valves.token_status_show_usage_and_estimate,
                    event_emitter=__event_emitter__,
                    file_context_enabled=target_file_context_enabled,
                    task_estimate_body=inner if task_source_body is not None else None,
                    transient_message_patterns=transient_message_patterns,
                    token_system_prompt=target_route.token_system_prompt,
                    dropped_message_keys=usage_anchor_dropped_message_keys,
                    ref_mode_active=effective_ref_mode.active,
                    ref_substitution_threshold_tokens=(
                        self.valves.ref_substitution_threshold_tokens
                    ),
                    checkpoint_profile_hash=checkpoint_profile_hash,
                )

        def schedule_completed_turn_soft_prefetch(completion: dict[str, Any]) -> None:
            try:
                if effective_soft_trigger_input_tokens is None:
                    return
                # Completed-turn prefetch is grounded ONLY in the just-completed
                # response's own usage. No usage => no prefetch (never fall back to a
                # previous turn's observed usage). Its input-plus-output total is a
                # proxy for the next request input after appending the assistant reply.
                completion_total_tokens = _usage_total(completion.get("usage"))
                if completion_total_tokens is None:
                    return
                if (
                    completion_total_tokens < effective_soft_trigger_input_tokens
                    or completion_total_tokens >= effective_trigger_input_tokens
                ):
                    return
                completed_user_id = str((user or {}).get("id") or "")
                completed_chat_id = str(metadata.get("chat_id") or "")
                completed_message_id = str(metadata.get("message_id") or "")
                if not completed_user_id or not _chat_id_supported(completed_chat_id) or not completed_message_id:
                    return
                coordinator_key = (
                    f"{CHECKPOINT_NAMESPACE}.completed_turn",
                    completed_user_id,
                    completed_chat_id,
                    identity.pipe_function_id,
                    checkpoint_profile_hash,
                    completed_message_id,
                )

                def prepare() -> tuple[
                    dict[str, Any],
                    tuple[str, str, str, str, str, str] | None,
                    _PreparedSoftPrefetch,
                ] | None:
                    assistant_message = completion.get("assistant_message")
                    if not isinstance(assistant_message, dict):
                        assistant_message = _assistant_message_for_completed_prefetch(completion)
                    if assistant_message is None:
                        return None
                    completed_body = _completed_turn_prefetch_body(checkpoint_lookup_body, assistant_message)
                    if completed_body is None:
                        return None
                    parent_key = _soft_prefetch_inflight_key_for_body(
                        user=user,
                        metadata=metadata,
                        body=checkpoint_lookup_body,
                        pipe_function_id=identity.pipe_function_id,
                        transient_message_patterns=transient_message_patterns,
                        checkpoint_profile_hash=checkpoint_profile_hash,
                    )
                    prepared = _prepare_soft_compaction_prefetch(
                        request=__request__,
                        user=user,
                        metadata=metadata,
                        body=completed_body,
                        pipe_function_id=identity.pipe_function_id,
                        summary_model_id=summary_model_id,
                        summary_tool_policy=self.valves.summary_tool_policy,
                        summary_prompt=self.valves.summary_prompt,
                        historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                        historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                        effective_trigger_input_tokens=effective_trigger_input_tokens,
                        effective_soft_trigger_input_tokens=effective_soft_trigger_input_tokens,
                        trigger_observed_tokens=completion_total_tokens,
                        trigger_usage_source="request",
                        token_status_detail=self.valves.token_status_detail,
                        token_status_show_usage_and_estimate=self.valves.token_status_show_usage_and_estimate,
                        event_emitter=__event_emitter__,
                        file_context_enabled=target_file_context_enabled,
                        transient_message_patterns=transient_message_patterns,
                        token_system_prompt=target_route.token_system_prompt,
                        dropped_message_keys=usage_anchor_dropped_message_keys,
                        ref_mode_active=effective_ref_mode.active,
                        ref_substitution_threshold_tokens=(
                            self.valves.ref_substitution_threshold_tokens
                        ),
                        checkpoint_profile_hash=checkpoint_profile_hash,
                    )
                    if prepared is None:
                        return None
                    return completed_body, parent_key, prepared

                async def prepare_and_start() -> None:
                    try:
                        prepared_result = await asyncio.to_thread(prepare)
                        if prepared_result is None:
                            return
                        completed_body, parent_key, prepared = prepared_result
                        parent_prefetch_task = (
                            _soft_prefetch_inflight_task_for_body(
                                user=user,
                                metadata=metadata,
                                body=checkpoint_lookup_body,
                                pipe_function_id=identity.pipe_function_id,
                                transient_message_patterns=transient_message_patterns,
                                inflight_key=parent_key,
                                checkpoint_profile_hash=checkpoint_profile_hash,
                            )
                            if parent_key is not None
                            else None
                        )
                        started = _start_soft_compaction_prefetch(
                            request=__request__,
                            user=user,
                            metadata=metadata,
                            body=completed_body,
                            pipe_function_id=identity.pipe_function_id,
                            summary_model_id=summary_model_id,
                            summary_tool_policy=self.valves.summary_tool_policy,
                            summary_prompt=self.valves.summary_prompt,
                            historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                            historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                            effective_trigger_input_tokens=effective_trigger_input_tokens,
                            effective_soft_trigger_input_tokens=effective_soft_trigger_input_tokens,
                            trigger_observed_tokens=completion_total_tokens,
                            trigger_usage_source="request",
                            token_status_detail=self.valves.token_status_detail,
                            token_status_show_usage_and_estimate=self.valves.token_status_show_usage_and_estimate,
                            event_emitter=__event_emitter__,
                            file_context_enabled=target_file_context_enabled,
                            transient_message_patterns=transient_message_patterns,
                            token_system_prompt=target_route.token_system_prompt,
                            dropped_message_keys=usage_anchor_dropped_message_keys,
                            parent_prefetch_task=parent_prefetch_task,
                            checkpoint_profile_hash=checkpoint_profile_hash,
                            _prepared=prepared,
                        )
                        if started:
                            child_task = _SOFT_PREFETCH_INFLIGHT_TASKS.get(prepared.key)
                            if child_task is not None:
                                await child_task
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        LOG.exception(
                            "Completed-turn soft compaction prefetch failed",
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )

                _launch_soft_prefetch_task(coordinator_key, prepare_and_start())
            except Exception as exc:
                LOG.exception(
                    "Completed-turn soft compaction prefetch failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        current_assistant_message_id = str(metadata.get("message_id") or "")
        continued_assistant_message_id = str(metadata.get("assistant_message_id") or "")
        # Continue Response extends this assistant in place, so its completed-turn
        # anchor/prefetch body cannot represent the next persisted history.
        is_continue_response = bool(
            continued_assistant_message_id
            and current_assistant_message_id == continued_assistant_message_id
        )

        async def handle_completed_turn(
            completion: dict[str, Any],
            anchor_input: UsageAnchorInput | None,
        ) -> None:
            if (
                anchor_input is not None
                and anchor_user_id
                and current_assistant_message_id
                and _chat_id_supported(str(chat_id or ""))
            ):
                try:
                    await persist_usage_anchor(
                        request=__request__,
                        user_id=anchor_user_id,
                        chat_id=str(chat_id),
                        pipe_function_id=identity.pipe_function_id,
                        assistant_message_id=current_assistant_message_id,
                        anchor_input=anchor_input,
                        raw_usage=completion.get("raw_usage"),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOG.warning(
                        "Auto-compaction usage anchor persistence failed "
                        "(chat_id=%s, assistant_message_id=%s)",
                        chat_id,
                        current_assistant_message_id,
                        exc_info=True,
                    )
            if not is_task_request and not checkpoint_lookup_unavailable:
                schedule_completed_turn_soft_prefetch(completion)

        attempt = 0
        compacted_once = False
        compaction_prefix_count = 0
        while attempt < MAX_CONTEXT_RETRY_ATTEMPTS:
            attempt += 1
            use_prepared_uncompacted_candidate = (
                attempt == 1
                and not should_compact
                and not compacted_once
                and prepared_uncompacted_candidate is not None
                and prepared_uncompacted_forward_candidate is not None
            )
            candidate = (
                prepared_uncompacted_candidate
                if use_prepared_uncompacted_candidate
                else _copy_body_preserving_metadata(inner)
            )
            candidate_is_projected = use_prepared_uncompacted_candidate
            selected_prepared_forward_candidate = (
                prepared_uncompacted_forward_candidate
                if use_prepared_uncompacted_candidate
                else None
            )
            # Restore pre-RAG clean messages for compaction so the cut is
            # computed on original content, not RAG-inflated text.
            if pre_rag_messages is not None and (should_compact or compacted_once):
                candidate["messages"] = copy.deepcopy(pre_rag_messages)
            compaction_prefix_count = 0
            candidate_source_events = None
            used_prepared_reusable_candidate = False
            if should_compact or compacted_once:
                try:
                    checkpoint_only_attempt = reusable_checkpoint_only is not None and not compacted_once
                    emit_progress_status = compacted_once or (
                        hard_should_compact
                        and not checkpoint_only_attempt
                        and (reusable_checkpoint_match is None or reusable_checkpoint_match.kind != "exact")
                    )
                    if emit_progress_status:
                        await emit_compaction_status(
                            __event_emitter__,
                            action="compacting",
                            description=_description_with_token_suffix(
                                "Compacting chat history before forwarding to the target model",
                                display_token_context,
                                show_usage_and_estimate=show_usage_and_estimate,
                            ),
                            done=False,
                            tokens=_tokens_status_payload(
                                display_token_context,
                                after=None,
                                show_usage_and_estimate=show_usage_and_estimate,
                            ),
                        )
                    if reusable_checkpoint_only is not None and not compacted_once:
                        prepared = await prepare_reusable_checkpoint_match(reusable_checkpoint_only)
                        if prepared is not None:
                            candidate = prepared[0]
                            compacted = True
                            compaction_prefix_count = prepared[2]
                            candidate_source_events = prepared[3]
                            used_prepared_reusable_candidate = True
                            candidate_is_projected = True
                            selected_prepared_forward_candidate = prepared[1]
                        elif task_source_body is not None:
                            candidate, compacted, compaction_prefix_count = await _compact_task_body_with_reusable_checkpoint(
                                request=__request__,
                                user=user,
                                metadata=metadata,
                                body=candidate,
                                pipe_function_id=identity.pipe_function_id,
                                match=reusable_checkpoint_only,
                                historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                                historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                                file_context_enabled=target_file_context_enabled,
                                transient_message_patterns=transient_message_patterns,
                                checkpoint_profile_hash=checkpoint_profile_hash,
                            )
                        else:
                            candidate, compacted, compaction_prefix_count = await _compact_body_with_reusable_checkpoint(
                                request=__request__,
                                user=user,
                                metadata=metadata,
                                body=candidate,
                                pipe_function_id=identity.pipe_function_id,
                                match=reusable_checkpoint_only,
                                historical_message_excerpt_bytes=self.valves.historical_message_excerpt_bytes,
                                historical_message_excerpt_count=self.valves.historical_message_excerpt_count,
                                file_context_enabled=target_file_context_enabled,
                                transient_message_patterns=transient_message_patterns,
                                checkpoint_profile_hash=checkpoint_profile_hash,
                            )
                    else:
                        if task_source_body is not None:
                            candidate, compacted, compaction_prefix_count = await _compact_task_body(
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
                                summary_prompt=self.valves.summary_prompt,
                                file_context_enabled=target_file_context_enabled,
                                transient_message_patterns=transient_message_patterns,
                                ref_projection_plan=ref_projection_plan,
                                ref_mode_active=effective_ref_mode.active,
                                checkpoint_profile_hash=checkpoint_profile_hash,
                            )
                        else:
                            candidate, compacted, compaction_prefix_count = await _compact_body(
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
                                summary_prompt=self.valves.summary_prompt,
                                file_context_enabled=target_file_context_enabled,
                                transient_message_patterns=transient_message_patterns,
                                ref_projection_plan=ref_projection_plan,
                                ref_mode_active=effective_ref_mode.active,
                                checkpoint_profile_hash=checkpoint_profile_hash,
                            )
                    compacted_once = compacted_once or compacted
                    if (
                        not is_summary_task
                        and not is_query_generation_task
                        and (should_compact or compacted_once)
                        and not used_prepared_reusable_candidate
                    ):
                        target_user_message_id = str(metadata.get("user_message_id") or message_id or "") or None
                        candidate = await _inject_target_file_context(
                            request=__request__,
                            user=user,
                            body=candidate,
                            chat_id=str(chat_id) if chat_id else None,
                            current_message_id=target_user_message_id,
                            compaction_prefix_count=compaction_prefix_count,
                            metadata_files=metadata.get("files"),
                            metadata_user_message=metadata.get("user_message"),
                            event_emitter=None,
                            file_context_enabled=target_file_context_enabled,
                            emit_source_events=False,
                            transient_message_patterns=transient_message_patterns,
                        )
                        candidate_metadata = candidate.get("metadata")
                        if isinstance(candidate_metadata, dict):
                            candidate_source_events = candidate_metadata.get("sources")
                    if emit_progress_status:
                        if compacted:
                            after_tokens = None
                            if self.valves.token_status_detail == "before_after":
                                with suppress(Exception):
                                    after_tokens = await _estimate_provider_input_tokens_async(
                                        candidate,
                                        request=__request__,
                                        user=user,
                                        system_prompt=target_route.token_system_prompt,
                                        dropped_message_keys=usage_anchor_dropped_message_keys,
                                    )
                            await emit_compaction_status(
                                __event_emitter__,
                                action="compacted",
                                description=_description_with_token_suffix(
                                    "Compacted chat history is ready for the target model",
                                    display_token_context,
                                    after=after_tokens,
                                    show_usage_and_estimate=show_usage_and_estimate,
                                ),
                                done=True,
                                tokens=_tokens_status_payload(
                                    display_token_context,
                                    after=after_tokens,
                                    show_usage_and_estimate=show_usage_and_estimate,
                                ),
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
                                description=_description_with_token_suffix(
                                    "Could not compact chat history safely; forwarding unchanged",
                                    display_token_context,
                                    show_usage_and_estimate=show_usage_and_estimate,
                                ),
                                done=True,
                                tokens=_tokens_status_payload(
                                    display_token_context,
                                    after=None,
                                    show_usage_and_estimate=show_usage_and_estimate,
                                ),
                            )
                except UnsupportedCompactionInput as exc:
                    await emit_compaction_status(
                        __event_emitter__,
                        action="failed",
                        description=_description_with_token_suffix(
                            str(exc),
                            display_token_context,
                            show_usage_and_estimate=show_usage_and_estimate,
                        ),
                        done=True,
                        error=True,
                        tokens=_tokens_status_payload(
                            display_token_context,
                            after=None,
                            show_usage_and_estimate=show_usage_and_estimate,
                        ),
                    )
                    return _error_response(str(exc), code=exc.code)
                except Exception as exc:
                    await emit_compaction_status(
                        __event_emitter__,
                        action="failed",
                        description=_description_with_token_suffix(
                            f"Failed to compact chat history: {exc}",
                            display_token_context,
                            show_usage_and_estimate=show_usage_and_estimate,
                        ),
                        done=True,
                        error=True,
                        tokens=_tokens_status_payload(
                            display_token_context,
                            after=None,
                            show_usage_and_estimate=show_usage_and_estimate,
                        ),
                    )
                    return _error_response(f"Failed to compact chat history: {exc}", code="summary_failed")

            # When no compaction happened, candidate inherited inner's pre-injected
            # file context.  Defer source events until the target forward succeeds,
            # and emit only the sources our own injection produced (never inherited
            # metadata sources from upstream processing).
            if not is_summary_task and not (should_compact or compacted_once):
                candidate_source_events = pre_injected_file_context_sources

            ref_attempt: RefAttempt | None = None
            try:
                final_projection_changed = False
                forward_candidate = None
                if effective_ref_mode.active:
                    selected_history_match = reusable_checkpoint_match
                    if compacted_once:
                        try:
                            selected_history_match = await _body_reusable_checkpoint_match(
                                request=__request__,
                                user=user,
                                metadata=metadata,
                                body=checkpoint_lookup_body,
                                pipe_function_id=identity.pipe_function_id,
                                transient_message_patterns=transient_message_patterns,
                                capture_logical_snapshot=True,
                                checkpoint_profile_hash=checkpoint_profile_hash,
                            )
                        except Exception as exc:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - failed relookup must block provider forwarding
                            raise RefProjectionError(stage="checkpoint relookup") from exc
                    selected_history_checkpoint = (
                        selected_history_match.checkpoint
                        if selected_history_match is not None
                        else None
                    )
                    previous_projection_surface = (
                        ref_projection_plan.manifests,
                        ref_projection_plan.render_manifests,
                    ) if ref_projection_plan is not None else ((), ())
                    ref_projection_plan = await extend_ref_projection_plan_with_checkpoint(
                        ref_projection_plan,
                        selected_history_checkpoint,
                        request=__request__,
                        metadata=metadata,
                        transient_message_patterns=transient_message_patterns,
                        logical_snapshot=(
                            selected_history_match.logical_snapshot
                            if selected_history_match is not None
                            else None
                        ),
                    )
                    current_projection_surface = (
                        ref_projection_plan.manifests,
                        ref_projection_plan.render_manifests,
                    ) if ref_projection_plan is not None else ((), ())
                    final_projection_changed = current_projection_surface != previous_projection_surface
                    if final_projection_changed:
                        candidate_is_projected = False
                        selected_prepared_forward_candidate = None
                if effective_ref_mode.active:
                    _require_provider_bound_history_ref_manifests(
                        candidate,
                        ref_projection_plan,
                    )
                if effective_ref_mode.active and ref_projection_plan is not None:
                    if not candidate_is_projected:
                        candidate = await apply_target_ref_projection(candidate)
                    if final_projection_changed:
                        forward_candidate = _apply_resolved_model_route_params(
                            candidate,
                            models=models,
                            route=target_route,
                        )
                        await estimate_candidate(forward_candidate)
                    if ref_projection_plan.catalog:
                        if ref_reservation is None:
                            ref_reservation = await reserve_ref_binding(
                                __request__,
                                ref_binding_key,
                                original_metadata_tools,
                            )
                        if ref_reservation is None:
                            raise RefProjectionError(stage="registration")
                        ref_attempt = stage_ref_attempt(
                            __request__,
                            ref_reservation,
                            ref_projection_plan,
                            threshold_tokens=self.valves.ref_substitution_threshold_tokens,
                        )
                    elif ref_reservation is not None:
                        await release_ref_reservation(__request__, ref_reservation)
                        ref_reservation = None
                if forward_candidate is None:
                    forward_candidate = (
                        selected_prepared_forward_candidate
                        if selected_prepared_forward_candidate is not None
                        else _apply_resolved_model_route_params(
                            candidate,
                            models=models,
                            route=target_route,
                        )
                    )
                    if effective_ref_mode.active:
                        await estimate_candidate(forward_candidate)
                forward_anchor_input = None
                if (
                    supported_context
                    and not is_task_request
                    and target_route.usage_anchor_shaping_hash is not None
                    and anchor_user_id
                    and metadata.get("message_id")
                    and _chat_id_supported(str(chat_id or ""))
                ):
                    forward_anchor_input = await _build_usage_anchor_input(
                        request=__request__,
                        body=_project_usage_anchor_token_body(
                            forward_candidate,
                            dropped_message_keys=usage_anchor_dropped_message_keys,
                        ),
                        usage_anchor_shaping_hash=target_route.usage_anchor_shaping_hash,
                        transient_message_patterns=transient_message_patterns,
                    )

                async def on_forward_complete(
                    completion: dict[str, Any],
                    anchor_input: UsageAnchorInput | None = forward_anchor_input,
                ) -> None:
                    await handle_completed_turn(completion, anchor_input)

                completion_callback = (
                    on_forward_complete
                    if (
                        not is_task_request
                        and supported_context
                        and not checkpoint_lookup_unavailable
                        and not is_continue_response
                    )
                    else None
                )
                if ref_attempt is not None:
                    register_ref_attempt(ref_attempt)
                    await commit_ref_attempt(__request__, ref_attempt)
                if is_streaming:
                    async def on_stream_terminal(saw_tool_call: bool) -> None:
                        if ref_attempt is not None and not saw_tool_call:
                            await cleanup_ref_attempt(__request__, ref_attempt)

                    streaming_kwargs = {
                        "request": __request__,
                        "user": user,
                        "body": forward_candidate,
                        "chat_id": str(chat_id) if chat_id else None,
                        "message_id": str(message_id) if message_id else None,
                        "wrapper_model_id": selected_wrapper_id,
                        "anchor_input": forward_anchor_input,
                        "on_complete": completion_callback,
                        "on_terminal": on_stream_terminal,
                        "track_request_usage": not is_task_request,
                    }
                    streaming_response = await _forward_streaming_target(**streaming_kwargs)
                    if getattr(streaming_response, "_auto_compact_immediate_error", False):
                        if ref_attempt is not None:
                            await rollback_ref_attempt(__request__, ref_attempt)
                    else:
                        await _emit_source_events(__event_emitter__, candidate_source_events)
                    return streaming_response
                response = await _forward_non_streaming_target(
                    request=__request__,
                    user=user,
                    body=forward_candidate,
                    chat_id=str(chat_id) if chat_id else None,
                    message_id=str(message_id) if message_id else None,
                    wrapper_model_id=selected_wrapper_id,
                    anchor_input=forward_anchor_input,
                    on_complete=completion_callback,
                    track_request_usage=not is_task_request,
                )
                if isinstance(response, dict) and response.get("error"):
                    if ref_attempt is not None:
                        await rollback_ref_attempt(__request__, ref_attempt)
                    return response
                has_tool_call = _responses_output_has_tool_call(response)
                choices = response.get("choices")
                if isinstance(choices, list):
                    has_tool_call = has_tool_call or any(
                        _choice_has_tool_call(choice) for choice in choices
                    )
                is_terminal_completion = isinstance(choices, list) or isinstance(
                    response.get("output"), list
                )
                if ref_attempt is not None and is_terminal_completion and not has_tool_call:
                    await cleanup_ref_attempt(__request__, ref_attempt)
                await _emit_source_events(__event_emitter__, candidate_source_events)
                response = _merge_source_events_into_response(response, candidate_source_events)
                return response
            except asyncio.CancelledError:
                if ref_attempt is not None:
                    await rollback_ref_attempt(__request__, ref_attempt)
                elif ref_reservation is not None:
                    await release_ref_reservation(__request__, ref_reservation)
                raise
            except RefProjectionError as exc:
                if ref_attempt is not None:
                    await rollback_ref_attempt(__request__, ref_attempt)
                elif ref_reservation is not None:
                    await release_ref_reservation(__request__, ref_reservation)
                _log_ref_projection_failure(exc)
                return _error_response(str(exc), code="ref_projection_failed")
            except RetryableContextOverflow:
                if ref_attempt is not None:
                    await rollback_ref_attempt(__request__, ref_attempt)
                    ref_reservation = None
                elif ref_reservation is not None:
                    await release_ref_reservation(__request__, ref_reservation)
                    ref_reservation = None
                if checkpoint_lookup_unavailable:
                    # The request was forwarded under the limit because the
                    # initial checkpoint lookup failed, but the target overflow
                    # proves compaction is actually required. The checkpoint DB
                    # was already known unavailable, so re-entering the
                    # compaction block would either fail with summary_failed or
                    # silently create a checkpoint if the DB recovered
                    # mid-request. Fail closed with the fixed store message.
                    return _error_response(
                        CHECKPOINT_STORE_UNAVAILABLE_MESSAGE,
                        code="checkpoint_unavailable",
                    )
                if not supported_context or attempt >= MAX_CONTEXT_RETRY_ATTEMPTS:
                    error_response = _context_exhaustion_error_response(
                        candidate.get("messages"),
                        transient_message_patterns=transient_message_patterns,
                    )
                    await emit_compaction_status(
                        __event_emitter__,
                        action="failed",
                        description=_description_with_token_suffix(
                            error_response["error"]["message"],
                            display_token_context,
                            show_usage_and_estimate=show_usage_and_estimate,
                        ),
                        done=True,
                        error=True,
                        tokens=_tokens_status_payload(
                            display_token_context,
                            after=None,
                            show_usage_and_estimate=show_usage_and_estimate,
                        ),
                    )
                    return error_response
                await emit_compaction_status(
                    __event_emitter__,
                    action="retry",
                    description=_description_with_token_suffix(
                        "Target context window was exceeded before output; compacting and retrying",
                        display_token_context,
                        show_usage_and_estimate=show_usage_and_estimate,
                    ),
                    done=False,
                    tokens=_tokens_status_payload(
                        display_token_context,
                        after=None,
                        show_usage_and_estimate=show_usage_and_estimate,
                    ),
                )
                should_compact = True
                compacted_once = True
                continue
            except Exception:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - cleanup re-raises unchanged
                if ref_attempt is not None:
                    await rollback_ref_attempt(__request__, ref_attempt)
                elif ref_reservation is not None:
                    await release_ref_reservation(__request__, ref_reservation)
                raise
