"""MCP tool resolution shared by ``llm_review`` and ``sub_agent``.

Open WebUI's tool catalogue can include ``server:mcp:<id>`` entries that
``open_webui.utils.tools.get_tools()`` silently skips, so each agent-loop
plugin that wants MCP support has to resolve these IDs itself: look up
the server connection, run access control, build auth headers, connect
an ``MCPClient``, list its tool specs, and wrap them in callables that
the tool-call middleware can invoke. ``cleanup_mcp_clients`` is the
matching teardown that disconnects each client when the agent loop
ends.

This module is its own shared dep (rather than living next to the
direct-tool-server helpers in ``shared.tool_servers``) on purpose:
``mc`` / ``magi`` / ``parallel_tools`` import from ``shared.tool_servers``
but never call the MCP helpers, and the inliner has no tree-shaking. If
the MCP code lived in ``tool_servers``, every plugin that imports any
direct-tool helper would include the ~270-line MCP block in its bundle
as unused code.

The module also can't import from sibling shared modules
(``shared.notifications`` / ``shared.async_utils``) because the inliner
forbids a shared dep from mixing external imports (``asyncio``,
``open_webui.*``, etc.) with cross-shared imports. The notification
emission and ``maybe_await`` helpers it needs are inlined privately
below.
"""

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
    entries compatible with Open WebUI's tool-call middleware.
    ``mcp_clients`` maps server IDs to the live ``MCPClient`` instances
    so the caller can ``cleanup_mcp_clients`` them when the agent loop
    ends.
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
    from open_webui.utils.headers import include_user_info_headers
    from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS

    try:
        from open_webui.utils.access_control import has_connection_access
    except ImportError:
        from open_webui.utils.tools import (
            has_tool_server_access as has_connection_access,
        )

    try:
        from open_webui.env import (
            FORWARD_SESSION_INFO_HEADER_CHAT_ID,
            FORWARD_SESSION_INFO_HEADER_MESSAGE_ID,
        )
    except ImportError:
        FORWARD_SESSION_INFO_HEADER_CHAT_ID = None
        FORWARD_SESSION_INFO_HEADER_MESSAGE_ID = None

    event_emitter = (extra_params or {}).get("__event_emitter__")

    async def emit_warning(description: str) -> None:
        await _emit_warning_notification(event_emitter, description)

    metadata = metadata or {}
    extra_params = extra_params or {}
    mcp_tools_dict: dict[str, dict] = {}
    mcp_clients: dict[str, Any] = {}
    server_connections = (
        getattr(getattr(request.app.state, "config", None), "TOOL_SERVER_CONNECTIONS", [])
        or []
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

            auth_type = mcp_server_connection.get("auth_type", "")
            headers: dict[str, Any] = {}
            if auth_type == "bearer":
                headers["Authorization"] = f'Bearer {mcp_server_connection.get("key", "")}'
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
                # Open WebUI core (utils/middleware.py) looks up OAuth
                # tokens under the colon-trailing segment of ``server_id``
                # so that ``host:port`` style ids resolve to the key the
                # UI stored. Mirror that for the lookup only -- keep the
                # full ``server_id`` for the ``mcp_clients`` cache key,
                # otherwise two servers sharing a trailing segment (e.g.
                # ``alpha:443`` and ``beta:443``) would collide on
                # ``mcp_clients[443]`` and the earlier client would be
                # overwritten without being added to the cleanup set.
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

            connection_headers = mcp_server_connection.get("headers", None)
            if connection_headers and isinstance(connection_headers, dict):
                headers.update(connection_headers)

            if ENABLE_FORWARD_USER_INFO_HEADERS and user:
                headers = include_user_info_headers(headers, user)
                if FORWARD_SESSION_INFO_HEADER_CHAT_ID and metadata.get("chat_id"):
                    headers[FORWARD_SESSION_INFO_HEADER_CHAT_ID] = metadata["chat_id"]
                if FORWARD_SESSION_INFO_HEADER_MESSAGE_ID and metadata.get("message_id"):
                    headers[FORWARD_SESSION_INFO_HEADER_MESSAGE_ID] = metadata["message_id"]

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
