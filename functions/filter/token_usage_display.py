"""
title: Token Usage Display
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Displays token usage from LLM response payload with automatic stream_options.include_usage injection. Shows usage percentage against model context window (fetched from LiteLLM or fallback). Display threshold configurable per user (default: show when ≥50%).
version: 0.5.3
license: MIT
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Awaitable, Callable

import httpx
from pydantic import BaseModel, Field


def _normalize_base_url(url: str) -> str:
    normalized = (url or "").rstrip("/")
    for suffix in ("/openai/v1", "/v1"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized.rstrip("/")


def _build_auth_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "x-api-key": api_key}


def _short_error_text(text: str, limit: int = 160) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def _collect_known_prefix_ids(request: Any) -> set[str]:
    prefix_ids: set[str] = set()
    if request is None:
        return prefix_ids

    try:
        config = request.app.state.config
    except Exception:
        return prefix_ids

    for attr in ("OPENAI_API_CONFIGS", "OLLAMA_API_CONFIGS"):
        api_configs = getattr(config, attr, None)
        if not isinstance(api_configs, dict):
            continue
        for value in api_configs.values():
            if not isinstance(value, dict):
                continue
            prefix_id = value.get("prefix_id")
            if isinstance(prefix_id, str) and prefix_id:
                prefix_ids.add(prefix_id)

    return prefix_ids


def _strip_open_webui_prefix_id(*, model: str, request: Any) -> str:
    """Strip Open WebUI connection Prefix ID exactly.

    Open WebUI applies Prefix ID as: '{prefix_id}.{model_id}'.
    This function only strips when it matches a configured prefix_id.
    """

    if not isinstance(model, str) or not model:
        return ""

    for prefix_id in _collect_known_prefix_ids(request):
        prefix = f"{prefix_id}."
        if model.startswith(prefix):
            return model[len(prefix) :]

    return model


def _extract_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except Exception:
            return None
    return None


def _format_int(n: int) -> str:
    return f"{n:,}"


def _find_usage_from_completed_payload(body: dict[str, Any]) -> dict[str, Any] | None:
    usage = body.get("usage")
    if isinstance(usage, dict) and usage:
        return usage

    response_message_id = body.get("id")

    messages = body.get("messages")
    if isinstance(messages, list):
        # Prefer usage on the completed assistant message.
        if response_message_id is not None:
            for m in messages:
                if not isinstance(m, dict):
                    continue
                if m.get("id") != response_message_id:
                    continue
                u = m.get("usage")
                if isinstance(u, dict) and u:
                    return u

        # Fallback: last message that has usage.
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            u = m.get("usage")
            if isinstance(u, dict) and u:
                return u

    return None


def _extract_usage_counts(usage: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    # OpenAI-compatible
    prompt = _extract_int(usage.get("prompt_tokens"))
    completion = _extract_int(usage.get("completion_tokens"))
    total = _extract_int(usage.get("total_tokens"))

    # Some providers use input/output naming
    if prompt is None:
        prompt = _extract_int(usage.get("input_tokens"))
    if completion is None:
        completion = _extract_int(usage.get("output_tokens"))

    # LiteLLM sometimes nests usage
    if prompt is None and completion is None and total is None:
        nested = usage.get("usage")
        if isinstance(nested, dict):
            return _extract_usage_counts(nested)

    if total is None:
        if prompt is not None and completion is not None:
            total = prompt + completion

    return prompt, completion, total


def _extract_model_info_mapping(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}

    data = payload.get("data")
    if not isinstance(data, list):
        return {}

    mapping: dict[str, int] = {}
    for item in data:
        if not isinstance(item, dict):
            continue

        model_name = item.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            model_name = (
                (item.get("model_info") or {}).get("key")
                or (item.get("litellm_params") or {}).get("model")
            )

        if not isinstance(model_name, str) or not model_name:
            continue

        model_info = item.get("model_info")
        if not isinstance(model_info, dict):
            continue

        max_input = _extract_int(model_info.get("max_input_tokens"))
        if max_input is None:
            max_input = _extract_int(model_info.get("max_tokens"))

        if max_input is None:
            continue

        mapping[model_name] = max_input

    return mapping


def _extract_model_group_info_mapping(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}

    data = payload.get("data")
    if not isinstance(data, list):
        return {}

    mapping: dict[str, int] = {}
    for item in data:
        if not isinstance(item, dict):
            continue

        model_group = item.get("model_group")
        if not isinstance(model_group, str) or not model_group:
            continue

        max_input = _extract_int(item.get("max_input_tokens"))
        if max_input is None:
            continue

        mapping[model_group] = max_input

    return mapping


def _parse_model_max_input_tokens_overrides(text: str) -> dict[str, int]:
    """Parse model max_input_tokens overrides.

    Format: 'model_id=100000,other=200000' (commas/newlines supported).
    Invalid entries are ignored.
    """

    if not isinstance(text, str) or not text.strip():
        return {}

    mapping: dict[str, int] = {}

    # Support comma-separated or newline-separated.
    parts: list[str] = []
    for chunk in text.replace("\n", ",").split(","):
        c = chunk.strip()
        if c:
            parts.append(c)

    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        model = key.strip()
        n = _extract_int(value.strip())
        if not model or n is None or n <= 0:
            continue
        mapping[model] = n

    return mapping


def _fallback_max_input_tokens_for_model(*, valves: Any, model: str, request: Any) -> int:
    """Return fallback max_input_tokens for a model.

    Priority:
    1) Exact match in overrides
    2) Exact match for stripped Open WebUI prefix id
    3) Global default
    """

    default_max = _extract_int(getattr(valves, "default_max_input_tokens", None)) or 100000

    overrides_text = getattr(valves, "model_max_input_tokens_overrides", "")
    overrides = _parse_model_max_input_tokens_overrides(overrides_text)
    if not overrides:
        return default_max

    if isinstance(model, str) and model and model in overrides:
        return overrides[model]

    stripped = ""
    if isinstance(model, str) and model:
        stripped = _strip_open_webui_prefix_id(model=model, request=request)
    if stripped and stripped in overrides:
        return overrides[stripped]

    return default_max


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=90,
            description="Priority level for the filter operations. Higher values run later.",
        )
        proxy_url: str = Field(
            default=os.environ.get("LITELLM_BASE_URL") or "",
            description="LiteLLM base URL (e.g., http://localhost:4000). Default reads env LITELLM_BASE_URL.",
        )
        api_key: str = Field(
            default=os.environ.get("LITELLM_API_KEY") or "",
            description="LiteLLM API key (Bearer). Default reads env LITELLM_API_KEY.",
        )
        request_timeout_seconds: float = Field(
            default=4.0,
            description="HTTP timeout for LiteLLM Proxy requests.",
        )
        model_info_ttl_seconds: int = Field(
            default=300,
            description="Cache TTL for /model/info results.",
        )
        force_include_usage: bool = Field(
            default=True,
            description=(
                "Force stream usage payload collection by setting stream_options.include_usage=true in inlet. "
                "Disable this only if your provider rejects stream_options."
            ),
        )
        default_max_input_tokens: int = Field(
            default=100000,
            ge=1,
            description=(
                "Fallback max_input_tokens when provider usage includes total_tokens but the model context window cannot be fetched. "
                "Used for percentage calculation and display gating."
            ),
        )
        model_max_input_tokens_overrides: str = Field(
            default="",
            description=(
                "Optional per-model max_input_tokens overrides. "
                "Format: 'model_id=100000,other-model=200000' (commas or newlines)."
            ),
        )
        pass

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable token usage status display.",
        )
        display_threshold_percent: int = Field(
            default=50,
            ge=0,
            le=100,
            description=(
                "Only display token usage status when context usage percentage is greater than or equal to this value. "
                "Set to 0 to always display."
            ),
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self._model_max_input_tokens_cache: dict[str, int] = {}
        self._model_info_last_fetch: float = 0.0

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        user_valves = self.UserValves.model_validate((__user__ or {}).get("valves", {}))
        if not user_valves.enabled:
            return body

        if not self.valves.force_include_usage:
            return body

        if not isinstance(body, dict):
            return body

        # OpenAI-compatible streaming usage requires stream_options.include_usage=true.
        # Open WebUI may expose this as a per-model setting; we force it here to avoid
        # toggling it model-by-model.
        if body.get("stream") is True:
            stream_options = body.get("stream_options")
            if not isinstance(stream_options, dict):
                stream_options = {}
            if stream_options.get("include_usage") is not True:
                stream_options = {**stream_options, "include_usage": True}
                body = {**body, "stream_options": stream_options}
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]] | None = None,
        __user__: dict | None = None,
        __metadata__: dict | None = None,
        __request__: Any = None,
        __model__: Any = None,
        __id__: str | None = None,
    ) -> dict:
        user_valves = self.UserValves.model_validate((__user__ or {}).get("valves", {}))
        if not user_valves.enabled:
            return body

        if not __event_emitter__:
            return body

        usage = _find_usage_from_completed_payload(body)
        if not usage:
            return body

        prompt_tokens, completion_tokens, total_tokens = _extract_usage_counts(usage)
        if total_tokens is None:
            return body

        model = body.get("model")
        model_text = model if isinstance(model, str) and model else ""

        # NOTE: Open WebUI overwrites `function_module.valves` from DB at runtime.
        # Apply env-based defaults at call time.
        base_url = _normalize_base_url(
            self.valves.proxy_url or os.environ.get("LITELLM_BASE_URL") or ""
        )
        api_key = self.valves.api_key or os.environ.get("LITELLM_API_KEY") or ""
        headers = _build_auth_headers(api_key)

        max_input_tokens: int | None = None
        if base_url:
            try:
                timeout = httpx.Timeout(self.valves.request_timeout_seconds)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    max_input_tokens = await self._get_model_max_input_tokens(
                        client=client,
                        base_url=base_url,
                        headers=headers,
                        model=model_text,
                        request=__request__,
                    )
            except httpx.ConnectError:
                # 接続エラーはフォールバックが妥当
                max_input_tokens = None
            except httpx.TimeoutException as e:
                # タイムアウトは設定ミスの可能性があるのでログ出力
                print(f"[TokenUsageDisplay] LiteLLM timeout: {e}", file=sys.stderr)
                max_input_tokens = None
            except Exception as e:
                # 予期しないエラーは必ずログ出力
                print(f"[TokenUsageDisplay] Unexpected error fetching model info: {e}", file=sys.stderr)
                max_input_tokens = None

        threshold = int(user_valves.display_threshold_percent)

        if max_input_tokens is None or max_input_tokens <= 0:
            max_input_tokens = _fallback_max_input_tokens_for_model(
                valves=self.valves, model=model_text, request=__request__
            )

        percent = (total_tokens / max_input_tokens) * 100
        if percent < threshold:
            return body
        description = (
            f"Tokens: {_format_int(total_tokens)} / {_format_int(max_input_tokens)} "
            f"({percent:.1f}%)"
        )

        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": True,
                    },
                }
            )
        except Exception:
            return body

        return body

    async def _get_model_max_input_tokens(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
        model: str,
        request: Any,
    ) -> int | None:
        now = time.time()
        ttl = max(0, int(self.valves.model_info_ttl_seconds))

        model_candidates: list[str] = []
        if isinstance(model, str) and model:
            model_candidates.append(model)
            stripped = _strip_open_webui_prefix_id(model=model, request=request)
            if stripped and stripped != model:
                model_candidates.append(stripped)

        if ttl > 0 and (now - self._model_info_last_fetch) < ttl:
            for candidate in model_candidates:
                cached = self._model_max_input_tokens_cache.get(candidate)
                if cached is not None:
                    return cached

        mapping: dict[str, int] = {}

        for path in ("/model/info", "/v1/model/info"):
            url = f"{base_url}{path}"
            response = await client.get(url, headers=headers)
            if response.status_code >= 400:
                continue
            mapping = _extract_model_info_mapping(response.json())
            if mapping:
                break

        if not mapping:
            for path in ("/model_group/info",):
                url = f"{base_url}{path}"
                response = await client.get(url, headers=headers)
                if response.status_code >= 400:
                    continue
                mapping = _extract_model_group_info_mapping(response.json())
                if mapping:
                    break

        if mapping:
            self._model_max_input_tokens_cache = mapping
            self._model_info_last_fetch = now

        for candidate in model_candidates:
            if candidate in mapping:
                return mapping[candidate]

        # Safe fallback (when request config isn't available): try stripping the
        # first segment only if it matches an actual mapping key.
        if isinstance(model, str) and model and "." in model:
            _, rest = model.split(".", 1)
            if rest in mapping:
                return mapping[rest]

        return None
