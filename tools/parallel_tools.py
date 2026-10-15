"""
title: Parallel Tools
author: skyzi000
version: 0.2.8
license: MIT
required_open_webui_version: 0.7.0
description: Execute multiple independent tool calls in parallel for faster results.

Open WebUI executes tool calls sequentially, which can be slow when multiple
independent tools need to run. This tool allows you to batch independent tool
calls and execute them concurrently using asyncio.gather.

Limitations:
- Requires a high-capability model (e.g., GPT-5.2, Claude Opus 4.6) to correctly
  invoke this tool. Smaller/mid-tier models may fail to pass the tool_calls
  parameter in the expected format.
"""

# === GENERATED FILE - DO NOT EDIT ===
# Source: src/owui_ext/tools/parallel_tools.py
# Regenerate with: uv run python scripts/build_release.py --target parallel_tools
# Future imports: (none)
# See release.toml for target definitions.

import asyncio
import ast
import json
import logging
from typing import Any, Callable, List, Optional
from fastapi import Request
from pydantic import BaseModel, Field

# --- inlined from src/owui_ext/shared/async_utils.py (owui_ext.shared.async_utils) ---
async def maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value

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

log = logging.getLogger(__name__)


class ToolCallItem(BaseModel):
    """A single tool call specification."""

    name: str = Field(description="Tool function name to call")
    args: dict = Field(default_factory=dict, description="Arguments to pass to the tool")


# ============================================================================
# Helper functions (outside class - AI cannot invoke these)
# ============================================================================


async def execute_single_tool(
    tool_name: str,
    tool_args: dict,
    tools_dict: dict,
    extra_params: dict,
    event_emitter: Optional[Callable] = None,
) -> dict:
    """Execute a single tool and return the result.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments to pass to the tool
        tools_dict: Dict of available tools {name: {callable, spec, ...}}
        extra_params: Extra parameters to pass to tool functions
        event_emitter: Optional event emitter for status updates

    Returns:
        Dict with tool_name and result (any JSON-serializable value)
    """
    # Strip "functions." prefix if present (OpenAI models sometimes add this)
    if tool_name.startswith("functions."):
        tool_name = tool_name[len("functions.") :]

    resolved, error = resolve_call_tool(
        {"function": {"name": tool_name, "arguments": json.dumps(tool_args)}}, tools_dict
    )
    if error:
        return {"tool_name": tool_name, "result": error}
    if "provider_function" in resolved:
        tool_name = resolved["function"]["name"]
        tool_args = json.loads(resolved["function"]["arguments"])

    if tool_name not in tools_dict:
        return {
            "tool_name": tool_name,
            "result": f"Error: Tool '{tool_name}' not found. Check the tool name.",
        }

    tool = tools_dict[tool_name]
    spec = tool.get("spec", {})

    try:
        # Filter to allowed parameters
        allowed_params = spec.get("parameters", {}).get("properties", {}).keys()
        filtered_args = {k: v for k, v in tool_args.items() if k in allowed_params}
        direct_tool = bool(tool.get("direct", False))
        tool_result_files: list[dict] = []
        tool_result_embeds: list[Any] = []

        if direct_tool:
            result = await execute_direct_tool_call(
                tool_function_name=tool_name,
                tool_function_params=filtered_args,
                tool=tool,
                extra_params=extra_params,
            )
        else:
            tool_function = tool["callable"]

            # Update function with current messages/files context
            from open_webui.utils.tools import get_updated_tool_function

            tool_function = await maybe_await(get_updated_tool_function(
                function=tool_function,
                extra_params={
                    "__messages__": extra_params.get("__messages__", []),
                    "__files__": extra_params.get("__files__", []),
                    "__event_emitter__": extra_params.get("__event_emitter__"),
                    "__event_call__": extra_params.get("__event_call__"),
                },
            ))

            result = await tool_function(**filtered_args)

        # Handle OpenAPI/external/direct tool results that return (data, headers)
        tool_type = tool.get("type", "")
        result = structure_terminal_file_tool_result(
            tool_name,
            filtered_args,
            result,
            tool,
            extra_params.get("__metadata__"),
        )
        result, tool_result_files, tool_result_embeds = await process_tool_result(
            tool_function_name=tool_name,
            tool_type=tool_type,
            tool_result=result,
            direct_tool=direct_tool,
            request=extra_params.get("__request__"),
            metadata=extra_params.get("__metadata__"),
            user=extra_params.get("__user__"),
        )
        tool_result_images = []
        if tool_result_files:
            from open_webui.utils import middleware

            # Only Parallel needs Core to extract images from its return container.
            if callable(getattr(middleware, "extract_base64_images", None)):
                tool_result_images, tool_result_files = await split_tool_result_files(
                    tool_result_files, extra_params.get("__user__")
                )

        # Emit terminal:* events for display/refresh behavior in UI
        await emit_terminal_tool_event(
            tool_function_name=tool_name,
            tool_function_params=filtered_args,
            tool_result=result,
            event_emitter=event_emitter,
        )
        defer_artifact_events = bool(extra_params.get("__defer_artifact_events__", False))
        if event_emitter and tool_result_files and not defer_artifact_events:
            await event_emitter({"type": "files", "data": {"files": tool_result_files}})
        if event_emitter and tool_result_embeds and not defer_artifact_events:
            await event_emitter({"type": "embeds", "data": {"embeds": tool_result_embeds}})

        # Try to parse JSON string results to avoid double-encoding
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass  # Keep as string if not valid JSON

        # Extract and emit citation sources for tools that generate them
        if event_emitter and result and tool_name in CITATION_TOOLS:
            # For citation extraction, we need a string representation
            result_str = (
                result
                if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False, default=str)
            )
            try:
                from open_webui.utils.middleware import (
                    get_citation_source_from_tool_result,
                )

                tool_id = tools_dict.get(tool_name, {}).get("tool_id", "")
                citation_sources = get_citation_source_from_tool_result(
                    tool_name=tool_name,
                    tool_params=filtered_args,
                    tool_result=result_str,
                    tool_id=tool_id,
                )
                for source in citation_sources:
                    await event_emitter({"type": "source", "data": source})
            except Exception as e:
                log.warning(f"Error extracting citation sources from {tool_name}: {e}")

        return {
            "tool_name": tool_name,
            "result": result,
            **({"images": tool_result_images} if tool_result_images else {}),
            **({"files": tool_result_files} if tool_result_files else {}),
            **(
                {
                    "embeds": tool_result_embeds
                }
                if tool_result_embeds and (not event_emitter or defer_artifact_events)
                else {}
            ),
        }

    except Exception as e:
        log.exception(f"Error executing tool {tool_name}: {e}")
        return {
            "tool_name": tool_name,
            "result": f"Error: {e}",
        }


# ============================================================================
# Tools class
# ============================================================================


class Tools:
    """Parallel tool execution for independent operations."""

    class Valves(BaseModel):
        AVAILABLE_TOOL_IDS: str = Field(
            default="",
            description=(
                "[Advanced] Comma-separated list of regular or MCP tool IDs available for parallel execution. "
                "Leave empty (recommended) to use regular and MCP tools enabled in the chat UI. "
                "Tool server IDs require their full prefix (e.g., 'server:mcp:context7' or 'server:context7'). "
                "Terminal tools are controlled by ENABLE_TERMINAL_TOOLS, and builtin/direct tools follow the current chat metadata."
            ),
        )
        EXCLUDED_TOOL_IDS: str = Field(
            default="",
            description=(
                "Comma-separated list of regular or MCP tool IDs to exclude from parallel execution. "
                "This is not a security boundary for tools the main AI can call directly."
            ),
        )
        ENABLE_SUBAGENT_TOOLS: bool = Field(
            default=True,
            description=(
                "Enable Core subagent tools (delegate_task, timer) when available in the current chat. "
                "Disable this only to prevent Parallel Tools from executing delegation and timer calls."
            ),
        )
        ENABLE_TERMINAL_TOOLS: bool = Field(
            default=True,
            description=(
                "Enable Open Terminal tools when terminal_id is available in chat metadata "
                "(e.g., run_command, list_files, read_file, write_file, display_file)."
            ),
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def run_tools_parallel(
        self,
        tool_calls: list[ToolCallItem],
        __user__: dict = None,
        __request__: Request = None,
        __model__: dict = None,
        __metadata__: dict = None,
        __id__: str = None,
        __event_emitter__: Callable[[dict], Any] = None,
        __event_call__: Callable[[dict], Any] = None,
        __chat_id__: str = None,
        __message_id__: str = None,
        __oauth_token__: Optional[dict] = None,
        __messages__: Optional[List[dict]] = None,
    ) -> str | dict:
        """
        Execute multiple independent tool calls in parallel.

        IMPORTANT - Read carefully:
        Open WebUI executes tool calls sequentially by default. When you call multiple
        tools separately, each one waits for the previous to complete, causing significant
        delays. Even if you use multi_tool_use.parallel or return multiple tool calls at
        once, Open WebUI will process them one by one - NOT in parallel.

        The user has enabled this tool specifically to solve this problem. Use this tool
        when you need to call 2+ independent tools. Batch them together in a single call
        to run concurrently. This dramatically reduces wait time.

        Example scenario:
        - BAD: Call search_web("Python"), wait, call search_web("FastAPI"), wait → slow
        - BAD: Use multi_tool_use.parallel → still sequential in Open WebUI
        - GOOD: Call run_tools_parallel with both searches → true parallel execution

        :param tool_calls: List of tool calls. Each item has "name" (tool name) and "args" (arguments dict). Example: [{"name": "search_web", "args": {"query": "Python"}}]
        :return: JSON object with results for each tool call
        """
        if __request__ is None:
            return json.dumps({"error": "Request context not available."})

        if __user__ is None:
            return json.dumps({"error": "User context not available."})

        # Convert ToolCallItem instances to dicts for uniform processing
        if isinstance(tool_calls, list):
            calls = [
                c.model_dump() if isinstance(c, ToolCallItem) else c
                for c in tool_calls
            ]
        elif isinstance(tool_calls, str):
            try:
                calls = ast.literal_eval(tool_calls)
            except Exception:
                try:
                    calls = json.loads(tool_calls)
                except Exception as e:
                    return json.dumps(
                        {
                            "error": f"Failed to parse tool_calls: {e}",
                            "expected_format": '[{"name": "tool_name", "arguments": {"arg1": "value1"}}]',
                        }
                    )
        else:
            return json.dumps(
                {
                    "error": f"tool_calls must be a list or JSON string, got {type(tool_calls).__name__}",
                }
            )

        if not isinstance(calls, list):
            return json.dumps(
                {
                    "error": "tool_calls must be a JSON array",
                    "expected_format": '[{"name": "tool_name", "arguments": {"arg1": "value1"}}]',
                }
            )

        if not calls:
            return json.dumps({"error": "tool_calls array is empty"})

        # Validate each call
        for i, call in enumerate(calls):
            # Fallback: parse JSON strings (some LLMs pass stringified objects
            # when the schema advertises items as strings).
            if isinstance(call, str):
                try:
                    call = json.loads(call)
                    calls[i] = call
                except (json.JSONDecodeError, TypeError):
                    return json.dumps(
                        {"error": f"tool_calls[{i}] must be an object, got unparseable string"},
                        ensure_ascii=False,
                    )
            if not isinstance(call, dict):
                return json.dumps({"error": f"tool_calls[{i}] must be an object"})
            if "name" not in call:
                return json.dumps({"error": f"tool_calls[{i}] missing 'name' field"})
            # Accept "args", "arguments", and "parameters" keys
            if "args" in call:
                calls[i]["arguments"] = call["args"]
            elif "arguments" not in call:
                if "parameters" in call:
                    calls[i]["arguments"] = call["parameters"]
                else:
                    calls[i]["arguments"] = {}

            # Parse arguments if it's a JSON string
            if isinstance(calls[i].get("arguments"), str):
                try:
                    calls[i]["arguments"] = json.loads(calls[i]["arguments"])
                except json.JSONDecodeError:
                    pass  # Keep as-is if not valid JSON

            # Ensure arguments is a dict (handle null/None and other invalid types)
            if not isinstance(calls[i].get("arguments"), dict):
                calls[i]["arguments"] = {}

        # Import here to avoid issues when not running in Open WebUI
        from open_webui.models.users import UserModel

        user = UserModel(**__user__)
        __metadata__ = __metadata__ or {}

        extra_params = {
            "__user__": __user__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__request__": __request__,
            "__model__": __model__,
            "__metadata__": __metadata__,
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
            "__oauth_token__": __oauth_token__,
            "__files__": __metadata__.get("files") or [],
        }

        # Determine which tools to use
        available_tool_ids: list[str] = []
        if __metadata__ and __metadata__.get("tool_ids"):
            available_tool_ids = list(__metadata__.get("tool_ids", []))

        if self.valves.AVAILABLE_TOOL_IDS.strip():
            tool_id_list = [
                tid.strip()
                for tid in self.valves.AVAILABLE_TOOL_IDS.split(",")
                if tid.strip()
            ]
        else:
            tool_id_list = available_tool_ids

        # Apply exclusions
        excluded = set()
        if self.valves.EXCLUDED_TOOL_IDS.strip():
            excluded = {
                tid.strip()
                for tid in self.valves.EXCLUDED_TOOL_IDS.split(",")
                if tid.strip()
            }

        # Exclude this tool itself
        if __id__:
            excluded.add(__id__)

        if self.valves.DEBUG:
            log.info(f"[ParallelTools] Tool IDs: {tool_id_list}")
            if excluded:
                log.info(f"[ParallelTools] Excluded tool IDs: {sorted(excluded)}")

        try:
            tools_dict, mcp_clients = await build_tools_dict(
                request=__request__,
                model=__model__ or {},
                metadata=__metadata__,
                user=user,
                valves=self.valves,
                extra_params=extra_params,
                tool_id_list=tool_id_list,
                excluded_tool_ids=excluded,
            )
        except Exception as e:
            log.exception(f"Error loading tools: {e}")
            return json.dumps({"error": f"Failed to load tools: {e}"})

        try:
            # Core binds these builtins to chat state the loader cannot rebuild.
            parent_tools = __metadata__.get("tools") or {}
            for name in (
                "view_skill",
                "read_skill_file",
                "grep_skill_files",
                "update_skill_files",
                "search_tools",
                "call_tool",
            ):
                parent_tool = parent_tools.get(name)
                if (
                    isinstance(parent_tool, dict)
                    and parent_tool.get("type") == "builtin"
                    and parent_tool.get("tool_id") == f"builtin:{name}"
                ):
                    tools_dict.setdefault(name, parent_tool)

            if self.valves.DEBUG:
                log.info(f"[ParallelTools] Total tools available: {len(tools_dict)}")

            # Prepare extra params for execution. Reuse the loader params so
            # metadata mutations such as resolved terminal/direct bindings stay
            # visible to direct, terminal, and MCP result processing.
            exec_extra_params = {
                **extra_params,
                "__messages__": __messages__ or [],
                "__defer_artifact_events__": True,
            }

            # Execute all tools in parallel
            tasks = [
                execute_single_tool(
                    tool_name=call.get("name", ""),
                    tool_args=call.get("arguments", {}),
                    tools_dict=tools_dict,
                    extra_params=exec_extra_params,
                    event_emitter=__event_emitter__,
                )
                for call in calls
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_results = []
            aggregated_files = []
            aggregated_embeds = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(
                        {
                            "tool_name": calls[i].get("name", "unknown"),
                            "result": f"Error: {result}",
                        }
                    )
                else:
                    if isinstance(result, dict):
                        files = result.get("files", [])
                        embeds = result.get("embeds", [])
                        if isinstance(files, list):
                            aggregated_files.extend(files)
                            if __event_emitter__:
                                result = {k: v for k, v in result.items() if k != "files"}
                        if isinstance(embeds, list):
                            aggregated_embeds.extend(embeds)
                            if __event_emitter__:
                                result = {k: v for k, v in result.items() if k != "embeds"}
                    processed_results.append(result)

            if __event_emitter__ and aggregated_files:
                await __event_emitter__(
                    {"type": "files", "data": {"files": aggregated_files}}
                )
            if __event_emitter__ and aggregated_embeds:
                await __event_emitter__(
                    {"type": "embeds", "data": {"embeds": aggregated_embeds}}
                )

            payload = {"results": processed_results}
            # Core extracts data-image values from containers, not JSON strings.
            if any(result.get("images") for result in processed_results if isinstance(result, dict)):
                from open_webui.utils.middleware import extract_base64_images

                for result in processed_results:
                    if not isinstance(result, dict):
                        continue
                    embedded_images = []
                    extract_base64_images(result.get("result"), embedded_images)
                    if embedded_images:
                        # JSON parsing must not promote another tool's text
                        # into extra images when Core processes this batch.
                        result["result"] = json.dumps(result["result"], ensure_ascii=False)
                return payload
            return json.dumps(payload, ensure_ascii=False)
        finally:
            await cleanup_mcp_clients(mcp_clients)
