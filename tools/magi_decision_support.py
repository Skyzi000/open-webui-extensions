"""
title: MAGI decision support
author: https://github.com/skyzi000
version: 0.2.18
license: MIT
required_open_webui_version: 0.7.0

MAGI decision support tool for Open WebUI.
This tool orchestrates three independent perspectives for multi-agent decision support:
- MELCHIOR: Scientific/technical analysis (logic, data, feasibility)
- BALTHASAR: Legal/ethical analysis (laws, ethics, social responsibility)
- CASPER: Emotional/trend analysis (human experience, trends, empathy)

Inspired by the MAGI supercomputer system from Neon Genesis Evangelion.
Not affiliated with the official Neon Genesis Evangelion, of course.

Agent characteristics referenced from https://github.com/yostos/claude-code-plugins (MIT License).
"""

# === GENERATED FILE - DO NOT EDIT ===
# Source: src/owui_ext/tools/magi_decision_support.py
# Regenerate with: uv run python scripts/build_release.py --target magi_decision_support
# Future imports: (none)
# See release.toml for target definitions.

import copy
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from fastapi import Request
from pydantic import BaseModel, Field

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

# --- inlined from src/owui_ext/shared/event_emitter.py (owui_ext.shared.event_emitter) ---
from typing import Any, Callable, Optional
class EventEmitter:
    def __init__(self, event_emitter: Optional[Callable[[dict], Any]] = None):
        self.event_emitter = event_emitter

    async def emit(
        self,
        description: str = "Unknown state",
        status: str = "in_progress",
        done: bool = False,
    ) -> None:
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )

# --- inlined from src/owui_ext/shared/inlet_filters.py (owui_ext.shared.inlet_filters) ---
import copy
import inspect
import logging
import random
from typing import Any
from fastapi import Request
from starlette.datastructures import State
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


def isolate_request_state(request: Request, model_id: str) -> Request:
    """Copy *request* with its own ``state`` for a completion by *model_id*.

    Core's ``generate_chat_completion`` writes per-call flags there, so
    concurrent nested completions sharing one request would race on them.
    It also merges ``state.metadata`` over the payload metadata, so the
    caller's ``model_id`` would otherwise reach the child's headers.
    """
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return request
    state = dict(scope.get("state") or {})
    state["metadata"] = {**(state.get("metadata") or {}), "model_id": model_id}
    isolated = copy.copy(request)
    isolated.scope = {**scope, "state": state}
    isolated._state = State(state)
    return isolated

# --- inlined from src/owui_ext/shared/mcp_tools.py (owui_ext.shared.mcp_tools) ---
import asyncio
import hashlib
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

    try:
        from open_webui.utils.tools import connect_mcp_server
    except ImportError:
        connect_mcp_server = None

    event_emitter = (extra_params or {}).get("__event_emitter__")

    async def emit_warning(description: str) -> None:
        await _emit_warning_notification(event_emitter, description)

    metadata = metadata or {}
    extra_params = extra_params or {}
    mcp_tools_dict: dict[str, dict] = {}
    mcp_clients: dict[str, Any] = {}
    server_connections = (
        [] if connect_mcp_server else await _get_tool_server_connections(request)
    )

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

    try:
        for server_id in ordered_server_ids:
            client = None
            try:
                function_name_filter_list = None
                if connect_mcp_server is not None:
                    connected = await connect_mcp_server(
                        request,
                        server_id,
                        user,
                        {**extra_params, "__metadata__": metadata},
                    )
                    if connected is None:
                        await emit_warning(f"MCP server '{server_id}' is not available")
                        continue
                    client, tool_specs = connected
                    mcp_clients[server_id] = client
                else:
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
                    # Own the client before connect/listing can be cancelled.
                    mcp_clients[server_id] = client

                    await client.connect(
                        url=mcp_server_connection.get("url", ""),
                        headers=headers if headers else None,
                    )

                    tool_specs = await client.list_tool_specs() or []

                client_lock = asyncio.Lock()

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
                    # Providers reject tool names over 64 characters.
                    if len(prefixed_name) > 64:
                        prefixed_name = (
                            f"{prefixed_name[:55]}_"
                            f"{hashlib.sha256(prefixed_name.encode()).hexdigest()[:8]}"
                        )
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

                if debug:
                    _mcp_tools_log.info(
                        f"Loaded {loaded_tool_count} MCP tools from server {server_id}"
                    )
            except Exception as e:
                _mcp_tools_log.warning(
                    f"Failed to load MCP tools from {server_id}: {e}"
                )
                if client is not None:
                    mcp_clients.pop(server_id, None)
                    await cleanup_mcp_clients({server_id: client})
                await emit_warning(f"Could not load MCP tools from '{server_id}': {e}")
    except BaseException:
        await cleanup_mcp_clients(mcp_clients)
        raise

    return mcp_tools_dict, mcp_clients


async def cleanup_mcp_clients(mcp_clients: dict | None = None, *more_clients: dict) -> None:
    """Close client groups in reverse acquisition order in their opening task.

    Error suppression cannot repair cross-task or out-of-order teardown.
    Preserve the response on internal cleanup errors, but re-raise genuine
    task cancellation after attempting the remaining clients.
    """
    cancelled = None
    for clients in reversed((mcp_clients, *more_clients)):
        for client in reversed(list((clients or {}).values())):
            try:
                await client.disconnect()
            except asyncio.CancelledError as exc:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    cancelled = exc
                else:
                    _mcp_tools_log.debug(f"Internal MCP cleanup cancellation: {exc}")
            except BaseException as exc:
                _mcp_tools_log.debug(f"Error cleaning up MCP client: {exc}")
    if cancelled is not None:
        raise cancelled

# --- inlined from src/owui_ext/shared/parsing.py (owui_ext.shared.parsing) ---
import json
from typing import Optional
def safe_json_loads(text: str) -> Optional[dict]:
    """Parse JSON, falling back to the largest ``{...}`` substring on failure.

    LLM responses often surround the JSON object with prose ("Here's the
    answer: {..}"). The fallback grabs the slice between the first ``{`` and
    the last ``}`` and retries; if that also fails we give up.
    """
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip()

# --- inlined from src/owui_ext/shared/prompt_utils.py (owui_ext.shared.prompt_utils) ---
from typing import Iterable, Optional
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
    """Add terminal instructions and tool-server prompts to a request snapshot.

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
    agents_md = (extra_params or {}).get("__terminal_agents_md__")
    if not isinstance(agents_md, str):
        agents_md = ""
    if not prompts and not agents_md:
        return form_data
    messages = list(form_data.get("messages", []))
    combined = "\n\n".join(prompts)
    if combined and messages and messages[0].get("role") == "system":
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
    elif combined:
        messages.insert(0, {"role": "system", "content": combined})
    if agents_md and not any(
        message.get("role") == "user" and message.get("content") == agents_md
        for message in messages
    ):
        try:
            from open_webui.utils.terminals import add_terminal_agents_md
        except ImportError:
            pass  # Older Core versions do not support terminal AGENTS.md.
        else:
            messages = add_terminal_agents_md(messages, agents_md)
    form_data["messages"] = messages
    return form_data


async def apply_agent_tool_search(
    tools_dict: dict, extra_params: dict, always_loaded: Iterable[str] = ()
) -> tuple[dict, dict, set]:
    """Apply Core's Tool Search to one agent loop.

    Returns the loop's tools (with ``search_tools``), extra_params carrying the
    manifest, and the names to leave out of the request's ``tools``. The search
    state lives in ``extra_params["__metadata__"]``, which must belong to this agent.
    """
    try:
        from open_webui.utils.tool_search import apply_tool_search
    except ImportError:
        return tools_dict, extra_params, set()

    kept = {name: tools_dict[name] for name in always_loaded if name in tools_dict}
    loop_tools = {name: tool for name, tool in tools_dict.items() if name not in kept}
    metadata = extra_params.get("__metadata__")
    if not isinstance(metadata, dict):
        metadata = {}
    form_data: dict = {"messages": []}
    deferred = await apply_tool_search(form_data, metadata, loop_tools)
    if not deferred:
        return tools_dict, extra_params, set()
    metadata["tools"] = loop_tools
    prompts = extra_params.get("__direct_tool_server_system_prompts__") or []
    return (
        {**loop_tools, **kept},
        {
            **extra_params,
            "__metadata__": metadata,
            "__direct_tool_server_system_prompts__": [
                *prompts,
                form_data["messages"][0]["content"],
            ],
        },
        deferred,
    )

# --- inlined from src/owui_ext/shared/skills.py (owui_ext.shared.skills) ---
import copy
import logging
import re
from typing import Any, Optional
from urllib.parse import quote
from fastapi import Request
_skills_log = logging.getLogger("owui_ext.shared.skills")

_SKILLS_MANIFEST_START = "<available_skills>"
_SKILLS_MANIFEST_END = "</available_skills>"
_SKILL_TAG_PATTERN = re.compile(
    r"<skill [^\n]*?name=.*?>\n.*?\n</skill>", re.DOTALL
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
    """Return all ``<skill ... name="...">…</skill>`` blocks found in *text*."""
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

    Core can list model-attached, accessible workspace, and terminal skills.
    Inline skills may also appear as full ``<skill>`` tags
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

    Mentioned skills, or selected skills on older Core / without builtin
    tools, can appear as ``<skill>`` tags alongside the available manifest.

    Args:
        messages: The parent conversation messages (``__messages__``).

    Returns:
        A list of ``<skill …>…</skill>`` strings, possibly empty.
    """
    return _extract_from_system_messages(messages, _find_skill_tags_in_text)


def extract_skill_ids(messages: Optional[list]) -> list[str]:
    """Return the IDs of the skills the parent lists or inlines, as Core's ``__skill_ids__``."""
    ids = set(re.findall(r"<id>(.*?)</id>", extract_skill_manifest(messages)))
    for tag in extract_user_skill_tags(messages):
        match = re.match(r'<skill (?:id="([^"]+)"|name="([^"]*)")', tag)
        if match:
            # Core inlines terminal skills without an id attribute.
            ids.add(match.group(1) or f'terminal:{quote(match.group(2), safe="")}')
    return sorted(ids)


def drop_unloadable_skill_manifest(messages: list, tools_dict: Optional[dict]) -> list:
    """Remove the ``<available_skills>`` manifest when no ``view_skill`` tool can load it."""
    if "view_skill" in (tools_dict or {}):
        return messages

    def without_manifest(message: dict) -> dict:
        content = message.get("content")
        if message.get("role") != "system" or not isinstance(content, str):
            return message
        manifest = _find_manifest_in_text(content)
        if not manifest:
            return message
        parts = (part.strip() for part in content.split(manifest, 1))
        return {**message, "content": "\n\n".join(part for part in parts if part)}

    return [without_manifest(message) for message in messages]


def isolate_chat_context(metadata: Optional[dict]) -> dict:
    """Copy *metadata* with its own ``chat_context``, as Core does for sub-agent runs."""
    metadata = metadata or {}
    return {
        **metadata,
        "chat_context": copy.deepcopy(metadata.get("chat_context") or {}),
    }


async def register_view_skill(
    tools_dict: dict,
    request: Request,
    extra_params: dict,
) -> None:
    """Manually register the view_skill builtin tool in tools_dict.

    This is needed for available skills whose content is not injected
    inline. The agent loop can call ``view_skill`` to lazily load their
    content from the ``<available_skills>`` manifest.

    Skills already supplied as full ``<skill>`` tags do not require
    ``view_skill``; they are passed directly in the system message.

    Args:
        tools_dict: The tools dict to add view_skill to (modified in-place).
        request: FastAPI request object.
        extra_params: Extra parameters for tool binding.
    """
    if "view_skill" in tools_dict:
        return

    try:
        from open_webui.tools import builtin

        # Core 0.12.0+ registers skill tools in get_builtin_tools per model.
        if hasattr(builtin, "read_skill_file"):
            return
        view_skill = builtin.view_skill
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

# --- inlined from src/owui_ext/shared/tool_execution.py (owui_ext.shared.tool_execution) ---
import ast
import json
import logging
import re
import uuid
from typing import Any, Callable, Optional
from fastapi import Request
_tool_execution_log = logging.getLogger("owui_ext.shared.tool_execution")
_core_process_tool_result = None


CITATION_TOOLS: set[str] = {
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


async def split_tool_result_files(files: list, user: Any = None) -> tuple[list[str], list]:
    """Match native Core: data images go to the model, URL images to both, other files to display."""
    images = []
    display_files = []
    for file in files:
        url = file.get("url") if isinstance(file, dict) and file.get("type") == "image" else None
        if isinstance(url, str) and url.startswith("data:"):
            images.append(url)
            continue
        display_files.append(file)
        if isinstance(url, str) and url:
            from open_webui.env import VERSION

            # Core before 0.12.0 keeps URL images display-only.
            if tuple(int(part) for part in re.findall(r"\d+", VERSION)[:2]) >= (0, 12):
                from open_webui.utils.files import get_image_base64_from_url

                image = await get_image_base64_from_url(url, user=_normalize_user(user))
                if image:
                    images.append(image)
    return images, display_files


def append_tool_result_images(messages: list[dict], urls: list[str]) -> None:
    """Keep tool replies contiguous, then attach images as a user message."""
    if urls:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here are the images from the tool results above. Please analyze them.",
                    },
                    *[{"type": "image_url", "image_url": {"url": url}} for url in urls],
                ],
            }
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
    if not event_emitter or tool_function_name not in TERMINAL_EVENT_TOOLS:
        return
    parsed = tool_result
    if tool_function_name != "run_command" and isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (ValueError, TypeError):
            parsed = None
    resolved_path = parsed.get("path") if isinstance(parsed, dict) else None
    if tool_function_name == "display_file":
        path = resolved_path or (
            tool_function_params.get("path", "")
            if isinstance(tool_function_params, dict)
            else ""
        )
        if not isinstance(path, str) or not path:
            return
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
        path = resolved_path or (
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


def resolve_call_tool(tool_call: dict, tools_dict: dict) -> tuple[dict, Optional[str]]:
    """Return a copy of *tool_call* targeting the tool a Tool Search ``call_tool`` names, or an error."""
    try:
        from open_webui.utils.tool_search import resolve_tool_call
    except ImportError:
        return tool_call, None
    resolved = dict(tool_call)
    return resolved, resolve_tool_call(resolved, tools_dict)


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
    if not isinstance(tool_call.get("function"), dict):
        return {
            "tool_call_id": tool_call_id,
            "content": f"Malformed tool_call: 'function' is {type(tool_call.get('function')).__name__}, not dict",
        }
    tool_call, error = resolve_call_tool(tool_call, tools_dict)
    if error:
        return {"tool_call_id": tool_call_id, "content": error}
    func = tool_call["function"]
    tool_function_name = func.get("name", "")
    tool_args_raw = func.get("arguments", "{}")

    tool_function_params = tool_args_raw
    if isinstance(tool_args_raw, str):
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
        return {
            "tool_call_id": tool_call_id,
            "content": "Error: Tool call arguments must be a JSON object.",
        }

    tool_result: Any = None
    tool_result_files: list[dict] = []
    tool_result_embeds: list[Any] = []
    tool_result_images: list[str] = []
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

                call_params = {
                    "__messages__": extra_params.get("__messages__", []),
                    "__files__": extra_params.get("__files__", []),
                    "__event_emitter__": extra_params.get("__event_emitter__"),
                    "__event_call__": extra_params.get("__event_call__"),
                }
                # Core builtins keep per-agent state (skill revisions, Tool
                # Search) in metadata, while loaded tools are shared by agents.
                if tool.get("type") == "builtin" and isinstance(
                    extra_params.get("__metadata__"), dict
                ):
                    call_params["__metadata__"] = extra_params["__metadata__"]
                tool_function = await _maybe_await(get_updated_tool_function(
                    function=tool_function,
                    extra_params=call_params,
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
            tool_result_images, tool_result_files = await split_tool_result_files(
                tool_result_files, extra_params.get("__user__")
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
        **({"images": tool_result_images} if tool_result_images else {}),
    }

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
    "skills": {"view_skill", "read_skill_file", "grep_skill_files"},
    "skills_write": {"update_skill_files", "create_skill"},
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
    "ENABLE_SKILLS_WRITE_TOOLS": "skills_write",
    "ENABLE_SUBAGENT_TOOLS": "subagents",
    "ENABLE_TASK_TOOLS": "tasks",
    "ENABLE_AUTOMATION_TOOLS": "automations",
    "ENABLE_CALENDAR_TOOLS": "calendar",
    "ENABLE_NOTIFICATION_TOOLS": "notifications",
}

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
    servers: list[dict] = []
    missing_prompts: list[tuple[dict, dict]] = []

    def connection_settings(server: dict) -> dict:
        return {
            key: value for key, value in server.items()
            if key not in {"specs", "system_prompt"}
        }

    if metadata_has_tool_servers:
        servers = normalize_direct_tool_servers(metadata.get("tool_servers"))
        parent_tools = metadata.get("tools")
        parent_tools = parent_tools if isinstance(parent_tools, dict) else {}
        for server in servers:
            if "specs" in server and "system_prompt" in server:
                continue
            settings = connection_settings(server)
            approved_specs = [
                dict(tool["spec"])
                for tool in parent_tools.values()
                if isinstance(tool, dict)
                and tool.get("direct") is True
                and isinstance(tool.get("server"), dict)
                and connection_settings(tool["server"]) == settings
                and isinstance(tool.get("spec"), dict)
                and isinstance(tool["spec"].get("name"), str)
                and tool["spec"]["name"]
            ]
            # Core pops specs from metadata.tool_servers, but retains only
            # the permitted entries in metadata.tools. Never use raw specs
            # to recreate a tool that Core removed.
            if "specs" not in server and approved_specs:
                server["specs"] = approved_specs
            if approved_specs and server.get("specs") and "system_prompt" not in server:
                missing_prompts.append((server, settings))
        if not missing_prompts:
            return servers

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
    if not metadata_has_tool_servers:
        return request_servers

    for server, settings in missing_prompts:
        prompts = [
            candidate.get("system_prompt")
            for candidate in request_servers
            if connection_settings(candidate) == settings
        ]
        # Prompt-only recovery is safe once Core approved this exact
        # connection. Ambiguous duplicate connections stay without a prompt.
        if prompts and all(isinstance(prompt, str) and prompt == prompts[0] for prompt in prompts):
            server["system_prompt"] = prompts[0]
    return servers


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
    include_terminal_agents_md=False,
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

    ``include_terminal_agents_md`` marks callers that construct an internal
    model conversation: it loads Core's terminal instructions once and
    omits ``play_animation``, which Core withholds from internal requests.
    """
    import inspect
    import json
    import logging
    import re

    log = logging.getLogger("owui_ext.shared.tool_loader")
    debug = bool(getattr(valves, "DEBUG", False))

    from open_webui.env import VERSION
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
    terminal_context_enabled = (
        bool(getattr(valves, "ENABLE_TERMINAL_TOOLS", True))
        and (model.get("info", {}).get("meta", {}).get("capabilities") or {}).get("terminal", True)
    )
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

    is_admin_terminal = True
    if terminal_id and terminal_context_enabled:
        try:
            from open_webui.models.config import Config
        except ImportError:
            pass
        else:
            terminal_connections = await maybe_await(
                Config.get("terminal_server.connections", None)
            )
            if terminal_connections is not None:
                is_admin_terminal = terminal_id in {
                    connection.get("id") for connection in terminal_connections
                }

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

    # Fetch file instructions before opening MCP clients so cancellation during
    # this network request cannot strand clients not yet returned to the caller.
    extra_params.pop("__terminal_agents_md__", None)
    if include_terminal_agents_md and terminal_id and terminal_context_enabled:
        try:
            from open_webui.utils.terminals import get_terminal_agents_md
        except ImportError:
            get_terminal_agents_md = None
        if get_terminal_agents_md is not None:
            agents_md = await get_terminal_agents_md(request, user, metadata, extra_params)
            if agents_md:
                extra_params["__terminal_agents_md__"] = agents_md

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

    try:
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

        if terminal_id and terminal_context_enabled and is_admin_terminal:
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
        elif terminal_id and not terminal_context_enabled and debug:
            log.info("Terminal tools disabled by the plugin valve or model capability")

        if direct_tool_servers:
            try:
                direct_tools = build_direct_tools_dict(
                    tool_servers=[
                        server for server in direct_tool_servers
                        if server.get("is_terminal") is not True or terminal_context_enabled
                    ],
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

        mcp_instructions = [
            f"MCP server {json.dumps(server_id, ensure_ascii=False)} instructions:\n{client.instructions}"
            for server_id, client in mcp_clients.items()
            if getattr(client, "instructions", None)
        ]
        if mcp_instructions:
            extra_params["__direct_tool_server_system_prompts__"] = [
                *extra_params.get("__direct_tool_server_system_prompts__", []),
                *mcp_instructions,
            ]

        try:
            features = metadata.get("features", {})

            # Core before 0.12.0 omits view_skill here; callers add it with
            # shared.skills.register_view_skill().
            builtin_extra_params = {
                "__user__": extra_params.get("__user__"),
                "__event_emitter__": extra_params.get("__event_emitter__"),
                "__event_call__": extra_params.get("__event_call__"),
                "__metadata__": extra_params.get("__metadata__"),
                "__chat_id__": extra_params.get("__chat_id__"),
                "__message_id__": extra_params.get("__message_id__"),
                "__oauth_token__": extra_params.get("__oauth_token__"),
                "__skill_ids__": extra_params.get("__skill_ids__"),
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

            disabled_builtin_tools: set = set()
            # Core before 0.12.0 keeps a single browser event callback and no
            # call timeout, so concurrent ask_user prompts can wait forever.
            if tuple(int(part) for part in re.findall(r"\d+", VERSION)[:2]) < (0, 12):
                disabled_builtin_tools.update(BUILTIN_TOOL_CATEGORIES["user_input"])
            for valve_field, category in VALVE_TO_CATEGORY.items():
                if not getattr(valves, valve_field, True):
                    disabled_builtin_tools.update(BUILTIN_TOOL_CATEGORIES.get(category, set()))
            if not getattr(valves, "ENABLE_SKILLS_TOOLS", True):
                disabled_builtin_tools.update(BUILTIN_TOOL_CATEGORIES["skills_write"])
            if include_terminal_agents_md:
                disabled_builtin_tools.add("play_animation")

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

        # Core already checked the originating browser's shell connection. Do not
        # restore shell tools it withheld when rebuilding a nested tool catalogue.
        parent_tools = metadata.get("tools") or {}
        inherit_user_shell = (
            terminal_context_enabled
            and metadata.get("session_id")
            and metadata.get("chat_id")
            and not metadata.get("automation_id")
            and not metadata.get("internal")
        )

        def is_selected_terminal_tool(tool):
            return isinstance(tool, dict) and bool(terminal_id) and (
                (tool.get("type") == "terminal" and tool.get("tool_id") == f"terminal:{terminal_id}")
                or (
                    tool.get("direct")
                    and tool.get("server", {}).get("is_terminal") is True
                    and tool.get("server", {}).get("url") == terminal_id
                )
            )

        for name in ("read_user_terminal", "send_user_terminal_input"):
            tool = tools_dict.get(name)
            if tool and (
                tool.get("type") == "terminal"
                or tool.get("server", {}).get("is_terminal") is True
            ):
                if not (
                    inherit_user_shell
                    and is_selected_terminal_tool(tool)
                    and is_selected_terminal_tool(parent_tools.get(name))
                ):
                    tools_dict.pop(name)
    except BaseException:
        await cleanup_mcp_clients(mcp_clients)
        raise

    return tools_dict, mcp_clients

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

# --- inlined from src/owui_ext/shared/voting.py (owui_ext.shared.voting) ---
from typing import List
def compute_vote_tally(votes: List[str]) -> dict:
    tally = {"A": 0, "B": 0, "abstain": 0}
    for vote in votes:
        if vote in tally:
            tally[vote] += 1
    return tally


def decide_majority(tally: dict) -> str:
    if tally.get("A", 0) > tally.get("B", 0):
        return "A"
    if tally.get("B", 0) > tally.get("A", 0):
        return "B"
    if tally.get("A", 0) == tally.get("B", 0) and tally.get("A", 0) > 0:
        return "tie"
    return "no_decision"

log = logging.getLogger(__name__)


# Agent-specific characteristics based on MAGI system design
# Adapted for decision support: scientific, ethical, and emotional perspectives
# Reference: https://github.com/yostos/claude-code-plugins (MIT License)
AGENT_CHARACTERISTICS = {
    "MELCHIOR": {
        "role": "Scientific Analysis",
        "traits": [
            "Deep knowledge of science and technology",
            "Makes judgments based on logic and rationality",
            "Emphasizes data, evidence, and causal relationships",
            "Uses efficiency, feasibility, and technical validity as evaluation criteria",
        ],
        "behavioral_guidelines": [
            "Evaluate the technical feasibility of each option",
            "Make judgments based on scientific evidence and data",
            "Analyze quantitative aspects such as cost, efficiency, and scalability",
            "Search for and reference the latest technological trends as needed",
            "Clarify logical causal relationships",
            "Document all external references (search queries, URLs, data sources)",
        ],
        "judgment_criteria": [
            "Technical feasibility",
            "Efficiency and performance",
            "Scalability and sustainability",
            "Cost-performance ratio",
            "Quality and quantity of evidence",
        ],
    },
    "BALTHASAR": {
        "role": "Guardian of Law and Ethics",
        "traits": [
            "Has a 'maternal' nature, emphasizing protection and norms",
            "High legal awareness, judges based on laws and precedents",
            "High ethical awareness, prioritizes ethics over rationality",
            "Considers social responsibility and public interest",
        ],
        "behavioral_guidelines": [
            "Actively search for relevant laws, regulations, and guidelines",
            "Reference precedents and prior cases to assess legal risks",
            "Examine ethical issues from multiple perspectives",
            "Analyze from the perspective of protecting the vulnerable and social equity",
            "Emphasize compliance and social responsibility",
            "Document all external references (laws, regulations, court cases, guidelines)",
        ],
        "judgment_criteria": [
            "Legal compliance",
            "Ethical validity",
            "Social responsibility and public interest",
            "Risk management and safety",
            "Impact on stakeholders",
        ],
    },
    "CASPER": {
        "role": "Advocate of Emotion and Trends",
        "traits": [
            "Has a 'feminine' nature, makes judgments based on emotions",
            "Emphasizes people's emotions and satisfaction (sympathy, joy, anxiety, etc.)",
            "Sensitive to current trends",
            "Actively investigates new cases and progressive initiatives",
            "Emphasizes user experience and human aspects",
        ],
        "behavioral_guidelines": [
            "Analyze the emotional impact each option has on stakeholders",
            "Search for and reference the latest trends",
            "Evaluate from the perspective of user experience and customer satisfaction",
            "Assess innovation and progressiveness",
            "Maintain a perspective of empathy and consideration",
            "Document all external references (trend reports, case studies, survey data)",
        ],
        "judgment_criteria": [
            "Emotional impact and satisfaction",
            "Quality of user experience",
            "Alignment with trends",
            "Innovation and progressiveness",
            "Human consideration and empathy",
        ],
    },
}


def build_agent_prompts(
    role_name: str,
    perspective: str,
    proposition: str,
    prerequisites: str,
    option_a: str,
    option_b: str,
    include_sources: bool,
) -> Tuple[str, str]:
    characteristics = AGENT_CHARACTERISTICS.get(role_name, {})
    role = characteristics.get("role", perspective)
    traits = characteristics.get("traits", [])
    guidelines = characteristics.get("behavioral_guidelines", [])
    criteria = characteristics.get("judgment_criteria", [])

    traits_text = "\n".join(f"- {t}" for t in traits) if traits else ""
    guidelines_text = "\n".join(f"- {g}" for g in guidelines) if guidelines else ""
    criteria_text = "\n".join(f"- {c}" for c in criteria) if criteria else ""

    system_prompt = (
        f"You are {role_name}, the {role} agent in the MAGI decision support system.\n\n"
        "CORE CHARACTERISTICS:\n"
        f"{traits_text}\n\n"
        "BEHAVIORAL GUIDELINES:\n"
        f"{guidelines_text}\n\n"
        "JUDGMENT CRITERIA:\n"
        f"{criteria_text}\n\n"
        "CRITICAL RULES:\n"
        "1. You operate independently and must not assume other agents' opinions.\n"
        "2. Use available tools proactively to gather information relevant to YOUR perspective.\n"
        "3. After gathering information, provide your final answer as JSON.\n"
        "4. You have limited tool call iterations. Use them wisely before the limit.\n\n"
        "FINAL RESPONSE FORMAT (JSON only):\n"
        "- vote: 'A', 'B', or 'abstain'\n"
        "- reasoning: detailed explanation from your unique perspective\n"
        "- benefits: array of short bullet phrases for chosen option\n"
        "- risks: array of short bullet phrases for chosen option\n"
        "- sources: array of {title, url} objects for data you gathered"
    )

    sources_note = "Include all sources from your web research." if include_sources else ""

    # Perspective-specific search guidance
    search_guidance = {
        "MELCHIOR": "Search for technical data, benchmarks, efficiency metrics, feasibility studies, and scientific evidence.",
        "BALTHASAR": "Search for relevant laws, regulations, legal precedents, ethical guidelines, and compliance requirements.",
        "CASPER": "Search for user experience reports, trend analyses, emotional impact studies, and innovative case studies.",
    }
    perspective_search = search_guidance.get(role_name, "Search for relevant information.")

    user_prompt = (
        "Decision Context:\n"
        f"- Proposition: {proposition}\n"
        f"- Prerequisites: {prerequisites or 'None'}\n"
        f"- Option A: {option_a}\n"
        f"- Option B: {option_b}\n\n"
        "Instructions:\n"
        f"1. {perspective_search}\n"
        "2. Analyze findings through your unique lens and judgment criteria.\n"
        "3. Provide your final JSON response with vote and detailed reasoning.\n"
        f"{sources_note}"
    )

    return system_prompt, user_prompt


async def generate_single_completion(
    request: Request,
    user: Any,
    model_id: str,
    messages: List[dict],
    extra_params: dict,
    apply_inlet_filters: bool,
) -> str:
    from open_webui.utils.chat import generate_chat_completion

    form_data = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "metadata": {
            "task": "magi_arbitrator",
            "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
        },
    }
    filter_pipeline = await resolve_model_filter_pipeline(
        apply_inlet_filters,
        request,
        model_id,
        extra_params.get("__metadata__", {}).get("filter_ids", []),
    )
    extra_params = {
        **extra_params,
        "__metadata__": {
            **extra_params.get("__metadata__", {}),
            "model_id": filter_pipeline["model_id"],
        },
    }
    form_data = await apply_inlet_filters_if_enabled(
        filter_pipeline, request, form_data, extra_params
    )
    form_data = await finalize_model_request(
        filter_pipeline, request, form_data, extra_params
    )

    response = await generate_chat_completion(
        request=isolate_request_state(request, form_data["model"]),
        form_data=form_data,
        user=user,
        bypass_filter=True,
    )

    error_msg = format_chat_completion_error(response)
    if error_msg is not None:
        return error_msg

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    return ""


async def run_agent_loop(
    request: Request,
    user: Any,
    model_id: str,
    messages: List[dict],
    tools_dict: dict,
    max_iterations: int,
    extra_params: dict,
    apply_inlet_filters: bool,
    agent_name: str = "Agent",
    event_emitter: Optional[Callable] = None,
    filter_pipeline: Optional[dict] = None,
) -> str:
    from open_webui.models.users import UserModel
    from open_webui.utils.chat import generate_chat_completion

    if extra_params is None:
        extra_params = {}

    # Prepare user object
    if isinstance(user, dict):
        user_obj = UserModel(**user)
    else:
        user_obj = user

    if filter_pipeline is None:
        filter_pipeline = await resolve_model_filter_pipeline(
            apply_inlet_filters,
            request,
            model_id,
            extra_params.get("__metadata__", {}).get("filter_ids", []),
        )

    final_extra_params = extra_params
    tools_dict, extra_params, deferred_tools = await apply_agent_tool_search(
        tools_dict or {}, extra_params
    )
    tools_param = None
    if tools_dict:
        tools_param = [
            {"type": "function", "function": tool.get("spec", {})}
            for name, tool in tools_dict.items()
            if name not in deferred_tools
        ]

    current_messages = drop_unloadable_skill_manifest(list(messages), tools_dict)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": f"[{agent_name}] Step {iteration}/{max_iterations}: Thinking...",
                        "done": False,
                    },
                }
            )

        iteration_info = f"[Iteration {iteration}/{max_iterations}]"
        if iteration == max_iterations:
            iteration_info += " This is your FINAL tool call opportunity. Provide your final JSON response now."

        messages_with_context = current_messages + [
            {"role": "system", "content": iteration_info}
        ]

        form_data = {
            "model": model_id,
            "messages": copy.deepcopy(messages_with_context),
            "stream": False,
            "metadata": {
                "task": "magi_agent",
                "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
            },
        }

        if tools_param:
            form_data["tools"] = tools_param

        form_data = await apply_inlet_filters_if_enabled(
            filter_pipeline, request, form_data, extra_params
        )
        form_data = _append_tool_server_prompts(form_data, extra_params)
        form_data = await finalize_model_request(
            filter_pipeline, request, form_data, extra_params
        )

        try:
            response = await generate_chat_completion(
                request=isolate_request_state(request, form_data["model"]),
                form_data=form_data,
                user=user_obj,
                bypass_filter=True,
            )
        except Exception as exc:
            log.exception(f"Error in MAGI agent completion: {exc}")
            return f"Error during MAGI agent execution: {exc}"

        error_msg = format_chat_completion_error(response)
        if error_msg is not None:
            return error_msg

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not choices:
                return "No response from model"

            message = choices[0].get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            # Emit status with LLM response content
            if event_emitter and content:
                truncated = content.replace(chr(10), ' ')[:100]
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[{agent_name}] Step {iteration}: {truncated}...",
                            "done": False,
                        },
                    }
                )

            if not tool_calls:
                if event_emitter:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[{agent_name}] Analysis complete.",
                                "done": False,
                            },
                        }
                    )
                return content or ""

            # Emit status with tool calls summary
            if event_emitter:
                tool_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[{agent_name}] Step {iteration}: Calling {', '.join(tool_names)}",
                            "done": False,
                        },
                    }
                )

            current_messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
            )

            tool_images = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args_raw = tool_call.get("function", {}).get("arguments", "{}")

                if event_emitter:
                    tool_args_display = (
                        str(tool_args_raw).replace(chr(10), " ")
                        if tool_args_raw
                        else "{}"
                    )
                    args_preview = tool_args_display[:80]
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[{agent_name}] Executing {tool_name}({args_preview}...)",
                                "done": False,
                            },
                        }
                    )

                result = await execute_tool_call(
                    tool_call,
                    tools_dict,
                    {
                        **extra_params,
                        "__messages__": current_messages,
                    },
                    event_emitter=event_emitter,
                )

                # Emit status with tool result preview
                tool_images.extend(result.get("images", []))
                if event_emitter:
                    result_preview = (result["content"] or "(empty)").replace(chr(10), ' ')[:80]
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[{agent_name}] {tool_name} returned: {result_preview}...",
                                "done": False,
                            },
                        }
                    )

                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"],
                    }
                )

            append_tool_result_images(current_messages, tool_images)

    # Max iterations reached
    if event_emitter:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "description": f"[{agent_name}] Max iterations ({max_iterations}) reached, finalizing...",
                    "done": False,
                },
            }
        )

    form_data = {
        "model": model_id,
        "messages": copy.deepcopy(current_messages)
        + [
            {
                "role": "user",
                "content": "Maximum tool iterations reached. Provide your final answer based on gathered information.",
            }
        ],
        "stream": False,
        "metadata": {
            "task": "magi_agent",
            "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
        },
    }

    form_data = await apply_inlet_filters_if_enabled(
        filter_pipeline, request, form_data, final_extra_params
    )
    form_data = _append_tool_server_prompts(form_data, final_extra_params)
    form_data = await finalize_model_request(
        filter_pipeline, request, form_data, final_extra_params
    )

    try:
        response = await generate_chat_completion(
            request=isolate_request_state(request, form_data["model"]),
            form_data=form_data,
            user=user_obj,
            bypass_filter=True,
        )
        error_msg = format_chat_completion_error(response)
        if error_msg is not None:
            return error_msg
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
    except Exception as exc:
        log.exception(f"Error getting final MAGI agent response: {exc}")

    return "MAGI agent reached maximum iterations without providing a final response."


class Tools:
    """MAGI decision support tool."""

    class Valves(BaseModel):
        DEFAULT_MODEL: str = Field(
            default="",
            description="Default model ID for MAGI agents. Leave empty to use the current model.",
        )
        MAX_ITERATIONS: int = Field(
            default=6,
            description="Maximum tool-call iterations per agent.",
        )
        APPLY_INLET_FILTERS: bool = Field(
            default=True,
            description="Apply inlet and request filters (e.g., user_info_injector) to MAGI model requests.",
        )

        # Builtin tool category toggles
        ENABLE_TIME_TOOLS: bool = Field(
            default=True,
            description="Enable time utilities (get_current_timestamp, calculate_timestamp).",
        )
        ENABLE_WEB_TOOLS: bool = Field(
            default=True,
            description="Enable web search tools (search_web, fetch_url) for MAGI agents.",
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
            description=(
                "Enable skill context for MAGI agents. When the parent "
                "conversation has a <available_skills> manifest, the manifest "
                "is appended to each agent's system prompt and the skill reading "
                "tools (view_skill, read_skill_file) are registered so agents can "
                "lazy-load skill contents. "
                "User-selected <skill> tags are also inlined when present."
            ),
        )
        ENABLE_SKILLS_WRITE_TOOLS: bool = Field(
            default=False,
            description=(
                "Enable skill writing tools (update_skill_files, create_skill; Open WebUI 0.12.0+). "
                "Off by default because they change saved skills. Requires ENABLE_SKILLS_TOOLS."
            ),
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
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging.",
        )
        pass

    class UserValves(BaseModel):
        INCLUDE_SOURCES: bool = Field(
            default=True,
            description="Include sources in agent outputs when web research is used.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def magi_decide(
        self,
        proposition: str,
        option_a: str,
        option_b: str,
        prerequisites: Optional[str] = None,
        __user__: dict = None,  # type: ignore
        __request__: Request = None,
        __model__: dict = None,
        __metadata__: dict = None,
        __messages__: Optional[list] = None,
        __id__: str = None,
        __event_emitter__: Callable[[dict], Any] = None,  # type: ignore
        __event_call__: Callable[[dict], Any] = None,  # type: ignore
        __chat_id__: str = None,
        __message_id__: str = None,
        __oauth_token__: Optional[dict] = None,
    ) -> str:
        """
        Run MAGI decision support with three independent perspectives and majority vote.

        :param proposition: The decision question to analyze.
        :param option_a: First option to evaluate.
        :param option_b: Second option to evaluate.
        :param prerequisites: Optional constraints and assumptions for the decision.

        :return: JSON string with:
        - decision: Majority vote result ("A", "B", "tie", or "no_decision")
        - vote_tally: Vote counts for each option
        - summary: Brief summary of the analysis
        - agents: MELCHIOR, BALTHASAR, CASPER results with votes and reasoning
        """
        emitter = EventEmitter(__event_emitter__)

        if __request__ is None:
            return json.dumps(
                {
                    "error": "Request context not available.",
                    "action": "Ensure __request__ is provided by Open WebUI.",
                },
                ensure_ascii=False,
            )

        if __user__ is None:
            return json.dumps(
                {
                    "error": "User context not available.",
                    "action": "Ensure __user__ is provided by Open WebUI.",
                },
                ensure_ascii=False,
            )

        from open_webui.models.users import UserModel

        user = UserModel(**__user__)
        raw_user_valves = (__user__ or {}).get("valves", {})
        user_valves = coerce_user_valves(raw_user_valves, self.UserValves)

        include_sources = bool(user_valves.INCLUDE_SOURCES)

        # Normalize inputs
        prop = normalize_text(proposition)
        prereq = normalize_text(prerequisites)
        opt_a = normalize_text(option_a)
        opt_b = normalize_text(option_b)

        # Validate required fields
        missing = []
        if not prop:
            missing.append("proposition")
        if not opt_a:
            missing.append("option_a")
        if not opt_b:
            missing.append("option_b")
        if missing:
            return json.dumps(
                {
                    "error": "Missing required fields for MAGI decision.",
                    "missing": missing,
                    "action": "Provide proposition, option_a, and option_b parameters.",
                },
                ensure_ascii=False,
            )

        model_id = self.valves.DEFAULT_MODEL or (__model__ or {}).get("id", "")
        if not model_id:
            return json.dumps(
                {
                    "error": "No model ID available.",
                    "action": "Set DEFAULT_MODEL in Valves or provide __model__.",
                },
                ensure_ascii=False,
            )

        extra_params = {
            "__user__": __user__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__request__": __request__,
            "__metadata__": __metadata__ or {},
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
            "__oauth_token__": __oauth_token__,
            "__files__": (__metadata__ or {}).get("files") or [],
            "__skill_ids__": extract_skill_ids(__messages__),
        }

        tool_id_list = []
        if __metadata__ and __metadata__.get("tool_ids"):
            tool_id_list = list(__metadata__.get("tool_ids", []))

        excluded_tool_ids = set()
        if __id__:
            excluded_tool_ids.add(__id__)

        skills_enabled = bool(getattr(self.valves, "ENABLE_SKILLS_TOOLS", True))
        skill_manifest = extract_skill_manifest(__messages__) if skills_enabled else ""
        user_skill_tags = extract_user_skill_tags(__messages__) if skills_enabled else []

        roles = [
            ("MELCHIOR", "scientific/technical"),
            ("BALTHASAR", "legal/ethical"),
            ("CASPER", "emotional/trend"),
        ]

        agent_results: Dict[str, dict] = {}
        raw_outputs: Dict[str, str] = {}

        import asyncio

        # Share tools and loaded context per effective model while each role
        # keeps its own Arena selection and filter pipeline.
        tools_cache: dict[str, tuple[dict, dict]] = {}
        all_mcp_clients: list[dict] = []

        async def ensure_tools(filter_pipeline: dict):
            effective_model_id = filter_pipeline["model_id"]
            cached = tools_cache.get(effective_model_id)
            if cached is not None:
                return cached
            loader_extra_params = {
                **extra_params,
                "__model__": filter_pipeline["model"],
                "__metadata__": {**extra_params["__metadata__"], "model_id": effective_model_id},
            }
            tools_dict, mcp_clients = await build_tools_dict(
                request=__request__,
                model=filter_pipeline["model"],
                metadata=loader_extra_params["__metadata__"],
                user=user,
                valves=self.valves,
                extra_params=loader_extra_params,
                tool_id_list=tool_id_list,
                excluded_tool_ids=excluded_tool_ids,
                include_terminal_agents_md=True,
            )
            # Record live clients before skill setup can fail or be cancelled.
            if mcp_clients:
                all_mcp_clients.append(mcp_clients)
            if skills_enabled and skill_manifest:
                await register_view_skill(tools_dict, __request__, loader_extra_params)
            cached = (tools_dict, loader_extra_params)
            tools_cache[effective_model_id] = cached
            return cached

        async def run_single_agent(
            role_name: str, perspective: str, filter_pipeline: dict, loaded_tools: tuple
        ) -> Tuple[str, str, dict]:
            """Run a single MAGI agent and return (role_name, raw_output, parsed_result)."""
            tools_dict, loaded_extra_params = loaded_tools
            agent_extra_params = {
                **loaded_extra_params,
                "__model__": filter_pipeline["model"],
                "__metadata__": isolate_chat_context(loaded_extra_params["__metadata__"]),
            }

            base_system_prompt, user_prompt = build_agent_prompts(
                role_name=role_name,
                perspective=perspective,
                proposition=prop,
                prerequisites=prereq,
                option_a=opt_a,
                option_b=opt_b,
                include_sources=include_sources,
            )
            if skills_enabled and (user_skill_tags or skill_manifest):
                system_prompt = merge_prompt_sections(
                    base_system_prompt,
                    *user_skill_tags,
                    skill_manifest,
                )
            else:
                system_prompt = base_system_prompt
            content = await run_agent_loop(
                request=__request__,
                user=user,
                model_id=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools_dict=tools_dict,
                max_iterations=self.valves.MAX_ITERATIONS,
                extra_params=agent_extra_params,
                apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                filter_pipeline=filter_pipeline,
                agent_name=role_name,
                event_emitter=__event_emitter__,
            )

            parsed = safe_json_loads(content) or {
                "vote": "abstain",
                "reasoning": content,
                "benefits": [],
                "risks": [],
                "sources": [],
            }
            return role_name, content, parsed

        try:
            # MCP sessions must be opened and closed by this same parent task.
            prepared_agents = []
            for role_name, perspective in roles:
                filter_pipeline = await resolve_model_filter_pipeline(
                    self.valves.APPLY_INLET_FILTERS,
                    __request__,
                    model_id,
                    extra_params["__metadata__"].get("filter_ids", []),
                )
                loaded_tools = await ensure_tools(filter_pipeline)
                prepared_agents.append((role_name, perspective, filter_pipeline, loaded_tools))
            await emitter.emit(
                description=f"MAGI Starting: {prop}",
                status="agents_starting",
                done=False,
            )
            results = await asyncio.gather(
                *(run_single_agent(*prepared) for prepared in prepared_agents),
                return_exceptions=True,
            )
        finally:
            await cleanup_mcp_clients(*all_mcp_clients)

        for result in results:
            if isinstance(result, Exception):
                log.exception(f"Agent execution failed: {result}")
                continue
            role_name, content, parsed = result
            raw_outputs[role_name] = content
            agent_results[role_name] = parsed

        votes = []
        for role_name, _ in roles:
            vote = normalize_text(agent_results.get(role_name, {}).get("vote", "")).upper()
            if vote not in {"A", "B", "ABSTAIN"}:
                vote = "ABSTAIN"
            votes.append("abstain" if vote == "ABSTAIN" else vote)

        tally = compute_vote_tally(votes)
        decision = decide_majority(tally)

        summary_prompt = (
            "Summarize the MAGI decision succinctly based on the three agents' analysis results. "
            "Highlight the key points from each perspective (technical, legal/ethical, emotional/trend). "
            "Provide 3-6 sentences."
        )
        summary_input = {
            "proposition": prop,
            "option_a": opt_a,
            "option_b": opt_b,
            "vote_tally": tally,
            "agents": agent_results,
        }

        summary = ""
        try:
            summary = await generate_single_completion(
                request=__request__,
                user=user,
                model_id=model_id,
                messages=[
                    {"role": "system", "content": summary_prompt},
                    {"role": "user", "content": json.dumps(summary_input, ensure_ascii=False)},
                ],
                extra_params=extra_params,
                apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
            )
        except Exception as exc:
            log.exception(f"Error generating summary: {exc}")
            summary = ""

        decision_text = opt_a if decision == "A" else opt_b if decision == "B" else decision
        await emitter.emit(
            description=f"MAGI Complete: {decision_text}",
            status="complete",
            done=True,
        )

        return json.dumps(
            {
                "decision": decision,
                "vote_tally": tally,
                "summary": summary,
                "agents": agent_results,
                "raw_outputs": raw_outputs if self.valves.DEBUG else None,
            },
            ensure_ascii=False,
        )
