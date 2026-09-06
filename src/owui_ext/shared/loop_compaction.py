"""Agent-loop compaction helpers for sub_agent.

Token estimation, message canonicalization, and the anchor+delta accounting
are ported from the production Auto Compact pipe (commit
54bb3153659368a607df51c53216167de7f41021, auto_compact.py blob
b3617497411c5518c840b9fa3b3959b00d4e0d8d).
"""

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
