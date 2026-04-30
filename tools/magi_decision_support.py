"""
title: MAGI decision support
author: https://github.com/skyzi000
version: 0.2.8
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

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from fastapi import Request
from pydantic import BaseModel, Field

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
import logging
from typing import Any
from fastapi import Request
_inlet_filters_log = logging.getLogger("owui_ext.shared.inlet_filters")


async def _inlet_filters_maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def apply_inlet_filters_if_enabled(
    apply_inlet_filters: bool,
    request: Request,
    model: dict,
    form_data: dict,
    extra_params: dict,
) -> dict:
    if not apply_inlet_filters:
        return form_data
    try:
        from open_webui.models.functions import Functions
        from open_webui.utils.filter import (
            get_sorted_filter_ids,
            process_filter_functions,
        )

        # Isolate __user__ so filter UserValves injection doesn't leak out
        # and pollute subsequent tool calls under a different tool id.
        local_extra_params = dict(extra_params or {})
        if isinstance(local_extra_params.get("__user__"), dict):
            local_extra_params["__user__"] = dict(local_extra_params["__user__"])

        filter_ids = await _inlet_filters_maybe_await(
            get_sorted_filter_ids(
                request,
                model,
                form_data.get("metadata", {}).get("filter_ids", []),
            )
        )
        filter_functions = []
        for filter_id in filter_ids:
            function = await _inlet_filters_maybe_await(Functions.get_function_by_id(filter_id))
            if function:
                filter_functions.append(function)
        form_data, _ = await process_filter_functions(
            request=request,
            filter_functions=filter_functions,
            filter_type="inlet",
            form_data=form_data,
            extra_params=local_extra_params,
        )
    except Exception as exc:
        _inlet_filters_log.warning(f"Error applying inlet filters: {exc}")
    return form_data

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
                except Exception:
                    pass
            await emit_warning(f"Could not load MCP tools from '{server_id}': {e}")

    return mcp_tools_dict, mcp_clients


async def cleanup_mcp_clients(mcp_clients: dict) -> None:
    """Disconnect all MCP clients, mirroring Open WebUI task cleanup."""
    for client in reversed(list((mcp_clients or {}).values())):
        try:
            await client.disconnect()
        except Exception as e:
            _mcp_tools_log.debug(f"Error cleaning up MCP client: {e}")

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
    """Process tool result into (payload, files, embeds) using core when available."""
    global _core_process_tool_result
    if _core_process_tool_result is None:
        try:
            from open_webui.utils.middleware import process_tool_result as fn

            if fn is not None:
                _core_process_tool_result = fn
        except ImportError:
            pass

    if _core_process_tool_result is not None:
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

    if isinstance(tool_result, tuple):
        tool_result = tool_result[0] if tool_result else ""
    elif direct_tool and isinstance(tool_result, list) and len(tool_result) == 2:
        tool_result = tool_result[0]
    if isinstance(tool_result, (dict, list)):
        tool_result = json.dumps(tool_result, indent=2, ensure_ascii=False)
    elif tool_result is not None and not isinstance(tool_result, str):
        tool_result = str(tool_result)
    return tool_result, [], []


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
        event = {"type": "terminal:display_file", "data": {"path": path}}
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

# --- inlined from src/owui_ext/shared/async_utils.py (owui_ext.shared.async_utils) ---
async def maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value

# --- inlined from src/owui_ext/shared/builtin_tools.py (owui_ext.shared.builtin_tools) ---
BUILTIN_TOOL_CATEGORIES: dict[str, set[str]] = {
    "time": {"get_current_timestamp", "calculate_timestamp"},
    "web": {"search_web", "fetch_url"},
    "image": {"generate_image", "edit_image"},
    "knowledge": {
        "list_knowledge",
        "list_knowledge_bases",
        "search_knowledge_bases",
        "query_knowledge_bases",
        "search_knowledge_files",
        "query_knowledge_files",
        "view_file",
        "view_knowledge_file",
    },
    "chat": {"search_chats", "view_chat"},
    "memory": {"search_memories", "add_memory", "replace_memory_content", "delete_memory", "list_memories"},
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
}


VALVE_TO_CATEGORY: dict[str, str] = {
    "ENABLE_TIME_TOOLS": "time",
    "ENABLE_WEB_TOOLS": "web",
    "ENABLE_IMAGE_TOOLS": "image",
    "ENABLE_KNOWLEDGE_TOOLS": "knowledge",
    "ENABLE_CHAT_TOOLS": "chat",
    "ENABLE_MEMORY_TOOLS": "memory",
    "ENABLE_NOTES_TOOLS": "notes",
    "ENABLE_CHANNELS_TOOLS": "channels",
    "ENABLE_CODE_INTERPRETER_TOOLS": "code_interpreter",
    "ENABLE_SKILLS_TOOLS": "skills",
    "ENABLE_TASK_TOOLS": "tasks",
    "ENABLE_AUTOMATION_TOOLS": "automations",
    "ENABLE_CALENDAR_TOOLS": "calendar",
}

# --- inlined from src/owui_ext/shared/model_features.py (owui_ext.shared.model_features) ---
from typing import Optional
def model_has_note_knowledge(model: Optional[dict]) -> bool:
    """Return True if the current model has note-type attached knowledge."""
    if not isinstance(model, dict):
        return False
    knowledge_items = model.get("info", {}).get("meta", {}).get("knowledge") or []
    if not isinstance(knowledge_items, list):
        return False
    return any(
        item.get("type") == "note"
        for item in knowledge_items
        if isinstance(item, dict)
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
    """Resolve direct tool servers from request.body() first, then metadata."""
    metadata_servers = normalize_direct_tool_servers(
        (metadata or {}).get("tool_servers") if isinstance(metadata, dict) else None
    )
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
        if debug and metadata_servers and len(metadata_servers) != len(request_servers):
            _tool_servers_log.warning(
                "tool_servers mismatch between request body and metadata; "
                "using request body tool_servers"
            )
        return request_servers
    return metadata_servers


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
    """Assemble the agent-loop tools_dict from regular, MCP, terminal,
    direct, and builtin sources.

    Returns ``(tools_dict, mcp_clients)``. Caller must call
    ``shared.mcp_tools.cleanup_mcp_clients(mcp_clients)`` once the
    agent loop is done so MCP connections don't leak. ``mcp_clients``
    is empty when no ``server:mcp:`` tool IDs are present.

    ``resolved_terminal_id`` / ``resolved_direct_tool_servers`` are
    optional pre-resolved values: when omitted, the helper resolves
    them from ``request.body()`` / ``metadata`` itself. Pre-resolving
    avoids re-reading ``request.body()`` when the caller already did
    so for its own bookkeeping.
    """
    import logging

    log = logging.getLogger("owui_ext.shared.tool_loader")
    debug = bool(getattr(valves, "DEBUG", False))

    from open_webui.utils.tools import get_builtin_tools, get_tools

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

        all_builtin_tools = await maybe_await(get_builtin_tools(
            request=request,
            extra_params=builtin_extra_params,
            features=features,
            model=model,
        ))

        disabled_builtin_tools: set = set()
        for valve_field, category in VALVE_TO_CATEGORY.items():
            if not getattr(valves, valve_field, True):
                disabled_builtin_tools.update(BUILTIN_TOOL_CATEGORIES.get(category, set()))

        knowledge_tools_enabled = bool(getattr(valves, "ENABLE_KNOWLEDGE_TOOLS", True))
        notes_tools_enabled = bool(getattr(valves, "ENABLE_NOTES_TOOLS", True))
        keep_view_note_for_knowledge = (
            (not notes_tools_enabled)
            and knowledge_tools_enabled
            and model_knowledge_tools_enabled(model)
            and model_has_note_knowledge(model)
        )

        # Regular tools take priority over builtin tools with the same name.
        builtin_count = 0
        for name, tool_dict in all_builtin_tools.items():
            if name in disabled_builtin_tools and not (
                name == "view_note" and keep_view_note_for_knowledge
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
) -> str:
    from open_webui.utils.chat import generate_chat_completion

    form_data = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "metadata": {"task": "magi_arbitrator"},
    }

    response = await generate_chat_completion(
        request=request,
        form_data=form_data,
        user=user,
        bypass_filter=True,
    )

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

    models = request.app.state.MODELS
    model = models.get(model_id, {})

    tools_param = None
    if tools_dict:
        tools_param = [
            {"type": "function", "function": tool.get("spec", {})}
            for tool in tools_dict.values()
        ]

    current_messages = list(messages)
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
            "messages": messages_with_context,
            "stream": False,
            "metadata": {
                "task": "magi_agent",
                "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
            },
        }

        if tools_param:
            form_data["tools"] = tools_param

        form_data = await apply_inlet_filters_if_enabled(
            apply_inlet_filters, request, model, form_data, extra_params
        )
        form_data = _append_tool_server_prompts(form_data, extra_params)

        try:
            response = await generate_chat_completion(
                request=request,
                form_data=form_data,
                user=user_obj,
                bypass_filter=True,
            )
        except Exception as exc:
            log.exception(f"Error in MAGI agent completion: {exc}")
            return f"Error during MAGI agent execution: {exc}"

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

            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", "{}")

                if event_emitter:
                    args_preview = tool_args.replace(chr(10), ' ')[:80]
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
        else:
            return f"Unexpected response type: {type(response)}"

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
        "messages": current_messages
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
        apply_inlet_filters, request, model, form_data, extra_params
    )
    form_data = _append_tool_server_prompts(form_data, extra_params)

    try:
        response = await generate_chat_completion(
            request=request,
            form_data=form_data,
            user=user_obj,
            bypass_filter=True,
        )
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
            description="Apply inlet filters (e.g., user_info_injector) to MAGI agent requests.",
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
                "is appended to each agent's system prompt and view_skill is "
                "registered as a tool so agents can lazy-load skill contents. "
                "User-selected <skill> tags are also inlined when present."
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

        Returns a JSON string with:
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
        user_valves = self.UserValves.model_validate((__user__ or {}).get("valves", {}))

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
            "__model__": __model__,
            "__metadata__": __metadata__ or {},
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
            "__oauth_token__": __oauth_token__,
            "__files__": (__metadata__ or {}).get("files", []),
        }

        tool_id_list = []
        if __metadata__ and __metadata__.get("tool_ids"):
            tool_id_list = list(__metadata__.get("tool_ids", []))

        excluded_tool_ids = set()
        if __id__:
            excluded_tool_ids.add(__id__)

        # Skill context from the parent conversation. The agent loop bypasses
        # the core middleware that would normally bind view_skill, so it has
        # to be registered manually after build_tools_dict returns.
        skills_enabled = bool(getattr(self.valves, "ENABLE_SKILLS_TOOLS", True))
        skill_manifest = extract_skill_manifest(__messages__) if skills_enabled else ""
        user_skill_tags = extract_user_skill_tags(__messages__) if skills_enabled else []

        tools_dict, mcp_clients = await build_tools_dict(
            request=__request__,
            model=__model__ or {},
            metadata=__metadata__ or {},
            user=user,
            valves=self.valves,
            extra_params=extra_params,
            tool_id_list=tool_id_list,
            excluded_tool_ids=excluded_tool_ids,
        )

        # MCP clients are live as soon as build_tools_dict returns; the
        # try/finally has to start *here* so an exception or cancellation in
        # the status emit / setup steps below still triggers cleanup.
        try:
            if skills_enabled and skill_manifest:
                await register_view_skill(tools_dict, __request__, extra_params)

            roles = [
                ("MELCHIOR", "scientific/technical"),
                ("BALTHASAR", "legal/ethical"),
                ("CASPER", "emotional/trend"),
            ]

            agent_results: Dict[str, dict] = {}
            raw_outputs: Dict[str, str] = {}

            await emitter.emit(
                description=f"MAGI Starting: {prop}",
                status="agents_starting",
                done=False,
            )

            async def run_single_agent(role_name: str, perspective: str) -> Tuple[str, str, dict]:
                """Run a single MAGI agent and return (role_name, raw_output, parsed_result)."""
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
                    extra_params=extra_params,
                    apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
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

            import asyncio

            agent_tasks = [run_single_agent(role_name, perspective) for role_name, perspective in roles]
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        finally:
            if mcp_clients:
                await cleanup_mcp_clients(mcp_clients)

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
