"""
title: Sub Agent
author: skyzi000
version: 0.5.3
license: MIT
required_open_webui_version: 0.7.0
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
import ast
import json
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Callable, List, Literal, Optional, Type
from fastapi import Request
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field

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

# --- inlined from src/owui_ext/shared/tool_event_metadata.py (owui_ext.shared.tool_event_metadata) ---
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

log = logging.getLogger(__name__)

_core_process_tool_result = None


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


def coerce_user_valves(raw_valves: Any, valves_cls: Type[BaseModel]) -> BaseModel:
    """Normalize raw user valves into the target valves class."""
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


def model_has_note_knowledge(model: Optional[dict]) -> bool:
    """Return True if the current model has note-type attached knowledge."""
    if not isinstance(model, dict):
        return False
    knowledge_items = (model.get("info", {}).get("meta", {}).get("knowledge") or [])
    if not isinstance(knowledge_items, list):
        return False
    return any(item.get("type") == "note" for item in knowledge_items if isinstance(item, dict))


def model_knowledge_tools_enabled(model: Optional[dict]) -> bool:
    """Return True if model-level builtin knowledge tools are enabled."""
    if not isinstance(model, dict):
        return True
    builtin_tools = model.get("info", {}).get("meta", {}).get("builtinTools", {})
    if not isinstance(builtin_tools, dict):
        return True
    return bool(builtin_tools.get("knowledge", True))


async def resolve_terminal_id_for_sub_agent(
    *,
    metadata: Optional[dict],
    request: Optional[Request],
    debug: bool,
) -> str:
    """Resolve terminal_id using request.body() first, then metadata."""

    def normalize_terminal_id(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    metadata_terminal_id = ""
    if isinstance(metadata, dict):
        metadata_terminal_id = normalize_terminal_id(metadata.get("terminal_id"))

    request_terminal_id = ""
    if request is not None:
        body = None
        request_body = getattr(request, "body", None)
        if callable(request_body):
            try:
                raw_body = await request_body()
                if raw_body:
                    body = json.loads(raw_body)
            except Exception:
                body = None

        if isinstance(body, dict):
            request_terminal_id = normalize_terminal_id(body.get("terminal_id"))
            if not request_terminal_id:
                nested_metadata = body.get("metadata")
                if isinstance(nested_metadata, dict):
                    request_terminal_id = normalize_terminal_id(
                        nested_metadata.get("terminal_id")
                    )

    if request_terminal_id:
        if debug and metadata_terminal_id and metadata_terminal_id != request_terminal_id:
            log.warning(
                "[SubAgent] terminal_id mismatch between request body and metadata; "
                "using request body terminal_id to match parent agent behavior"
            )
        return request_terminal_id

    if metadata_terminal_id:
        return metadata_terminal_id

    return ""


def normalize_direct_tool_servers(value: Any) -> List[dict]:
    """Normalize direct tool server payload into a list of dict copies."""
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def extract_direct_tool_server_prompts(direct_tools: Mapping[str, dict]) -> List[str]:
    """Collect unique non-empty system prompts from loaded direct tools only."""
    prompts: List[str] = []
    seen_prompts = set()
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


async def resolve_direct_tool_servers_for_sub_agent(
    *,
    metadata: Optional[dict],
    request: Optional[Request],
    debug: bool,
) -> List[dict]:
    """Resolve direct tool servers from request.body() first, then metadata."""
    metadata_servers = normalize_direct_tool_servers(
        (metadata or {}).get("tool_servers") if isinstance(metadata, dict) else None
    )

    request_servers: List[dict] = []
    if request is not None:
        request_body = getattr(request, "body", None)
        if callable(request_body):
            try:
                raw_body = await request_body()
                if raw_body:
                    body = json.loads(raw_body)
                    if isinstance(body, dict):
                        request_servers = normalize_direct_tool_servers(body.get("tool_servers"))
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
            log.warning(
                "[SubAgent] tool_servers mismatch between request body and metadata; "
                "using request body tool_servers"
            )
        return request_servers

    return metadata_servers


def build_direct_tools_dict(*, tool_servers: List[dict], debug: bool) -> dict:
    """Build direct tool entries compatible with Open WebUI middleware."""
    direct_tools = {}
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
        log.info("[SubAgent] No direct tools loaded from tool_servers")

    return direct_tools


async def execute_direct_tool_call(
    *,
    tool_function_name: str,
    tool_function_params: dict,
    tool: dict,
    extra_params: dict,
) -> Any:
    """Execute direct tools through __event_call__ like core middleware."""
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


def normalize_terminal_tools_result(*, terminal_tools_result: Any, extra_params: Optional[dict]) -> dict:
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


def _normalize_user(user: Any) -> Any:
    """Convert raw __user__ dict payloads into UserModel when needed."""
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


async def emit_notification(
    event_emitter: Optional[Callable],
    *,
    level: str,
    content: str,
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
                "data": {
                    "type": level,
                    "content": content.strip(),
                },
            }
        )
    except Exception as e:
        log.debug(f"[SubAgent] Error emitting notification ({level}): {e}")


async def resolve_mcp_tools(
    request: Request,
    user: Any,
    mcp_tool_ids: list[str],
    extra_params: dict,
    metadata: dict,
    debug: bool = False,
) -> tuple[dict, dict]:
    """Resolve MCP ``server:mcp:`` tool IDs into tool callables and live clients."""
    try:
        from open_webui.utils.mcp.client import MCPClient
    except ImportError:
        if debug and mcp_tool_ids:
            log.info("[SubAgent] MCPClient unavailable; skipping MCP tool resolution")
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

    async def emit_warning(description: str) -> None:
        await emit_notification(
            extra_params.get("__event_emitter__"),
            level="warning",
            content=description,
        )

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
                log.warning(f"[SubAgent] MCP server with id {server_id} not found")
                await emit_warning(f"MCP server '{server_id}' was not found")
                continue

            if not mcp_server_connection.get("config", {}).get("enable", True):
                if debug:
                    log.info(f"[SubAgent] MCP server {server_id} is disabled; skipping")
                await emit_warning(f"MCP server '{server_id}' is disabled")
                continue

            try:
                has_access = await maybe_await(has_connection_access(user, mcp_server_connection))
            except TypeError:
                has_access = await maybe_await(has_connection_access(user, mcp_server_connection, None))

            if not has_access:
                log.warning(f"[SubAgent] Access denied to MCP server {server_id} for user {user.id}")
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
                try:
                    oauth_token = await request.app.state.oauth_client_manager.get_oauth_token(
                        user.id, f"mcp:{server_id}"
                    )

                    if oauth_token:
                        headers["Authorization"] = f'Bearer {oauth_token.get("access_token", "")}'
                except Exception as e:
                    log.error(
                        f"[SubAgent] Error getting OAuth token for MCP server {server_id}: {e}"
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
                log.info(
                    f"[SubAgent] Loaded {loaded_tool_count} MCP tools from server {server_id}"
                )
        except Exception as e:
            log.warning(f"[SubAgent] Failed to load MCP tools from {server_id}: {e}")
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
            log.debug(f"[SubAgent] Error cleaning up MCP client: {e}")


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
        return await maybe_await(_core_process_tool_result(
            request,
            tool_function_name,
            tool_result,
            tool_type,
            direct_tool=direct_tool,
            metadata=metadata if isinstance(metadata, dict) else {},
            user=_normalize_user(user),
        ))
    # Fallback for Open WebUI < 0.8.x
    if isinstance(tool_result, tuple):
        tool_result = tool_result[0] if tool_result else ""
    elif direct_tool and isinstance(tool_result, list) and len(tool_result) == 2:
        tool_result = tool_result[0]
    if isinstance(tool_result, (dict, list)):
        tool_result = json.dumps(tool_result, indent=2, ensure_ascii=False)
    elif tool_result is not None and not isinstance(tool_result, str):
        tool_result = str(tool_result)
    return tool_result, [], []


async def emit_terminal_tool_event(
    *,
    tool_function_name: str,
    tool_function_params: dict,
    tool_result: Any,
    event_emitter: Optional[Callable],
) -> None:
    """Emit terminal:* UI events for Open Terminal tool results."""
    if not event_emitter or tool_function_name not in TERMINAL_EVENT_TOOLS:
        return

    if tool_function_name == "display_file":
        path = tool_function_params.get("path", "") if isinstance(tool_function_params, dict) else ""
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
        path = tool_function_params.get("path", "") if isinstance(tool_function_params, dict) else ""
        if not isinstance(path, str) or not path:
            return
        event = {"type": f"terminal:{tool_function_name}", "data": {"path": path}}
    elif tool_function_name == "run_command":
        event = {"type": "terminal:run_command", "data": {}}
    else:
        return

    try:
        await event_emitter(event)
    except Exception as e:
        log.warning(f"Error emitting terminal event for {tool_function_name}: {e}")


async def execute_tool_call(
    tool_call: dict,
    tools_dict: dict,
    extra_params: dict,
    event_emitter: Optional[Callable] = None,
) -> dict:
    """Execute a single tool call and return the result.

    Args:
        tool_call: The tool call dict with id, function.name, function.arguments
        tools_dict: Dict of available tools {name: {callable, spec, ...}}
        extra_params: Extra parameters to pass to tool functions
        event_emitter: Optional event emitter for citation/source events

    Returns:
        Dict with tool_call_id and content (result)
    """
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

    # Parse arguments - handle both JSON string and pre-parsed dict/list
    tool_function_params: dict = {}
    if isinstance(tool_args_raw, dict):
        tool_function_params = tool_args_raw
    elif isinstance(tool_args_raw, str):
        try:
            tool_function_params = ast.literal_eval(tool_args_raw)
        except Exception:
            try:
                tool_function_params = json.loads(tool_args_raw)
            except Exception as e:
                log.error(f"Error parsing tool call arguments: {tool_args_raw} - {e}")
                return {
                    "tool_call_id": tool_call_id,
                    "content": f"Error parsing arguments: {e}",
                }
    if not isinstance(tool_function_params, dict):
        tool_function_params = {}

    # Execute the tool
    tool_result = None
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

                # Update function with current context
                # __event_emitter__ is included so parallel sub-agents can
                # override it with a task-specific indexed emitter.
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

                tool_result = await tool_function(**tool_function_params)

            # Handle OpenAPI/external tool results that return (data, headers) tuple
            # Headers (CIMultiDictProxy) are not JSON-serializable
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

        except Exception as e:
            log.exception(f"Error executing tool {tool_function_name}: {e}")
            tool_result = f"Error: {e}"
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

    # Process result to string
    if tool_result is None:
        tool_result = ""
    elif not isinstance(tool_result, str):
        try:
            tool_result = json.dumps(tool_result, ensure_ascii=False, default=str)
        except Exception:
            tool_result = str(tool_result)

    # Extract and emit citation sources for tools that generate them
    # This enables proper source attribution in the UI (e.g., web search results, KB documents)
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
        except Exception as e:
            log.warning(f"Error extracting citation sources from {tool_function_name}: {e}")

    return {
        "tool_call_id": tool_call_id,
        "content": tool_result,
    }


async def apply_inlet_filters_if_enabled(
    apply_inlet_filters: bool,
    request: Request,
    model: dict,
    form_data: dict,
    extra_params: dict,
) -> dict:
    """Apply inlet filters to form_data if enabled.

    Args:
        apply_inlet_filters: Whether to apply inlet filters
        request: FastAPI request object
        model: Model info dict
        form_data: Form data dict to process
        extra_params: Extra parameters for filter processing

    Returns:
        Processed form_data (may be modified by filters)
    """
    if not apply_inlet_filters:
        return form_data

    try:
        from open_webui.models.functions import Functions
        from open_webui.utils.filter import get_sorted_filter_ids, process_filter_functions

        # Isolate __user__ so filter UserValves injection doesn't leak out.
        local_extra_params = dict(extra_params or {})
        if isinstance(local_extra_params.get("__user__"), dict):
            local_extra_params["__user__"] = dict(local_extra_params["__user__"])

        filter_ids = await maybe_await(get_sorted_filter_ids(
            request, model, form_data.get("metadata", {}).get("filter_ids", [])
        ))
        filter_functions = []
        for filter_id in filter_ids:
            function = await maybe_await(Functions.get_function_by_id(filter_id))
            if function:
                filter_functions.append(function)
        form_data, _ = await process_filter_functions(
            request=request,
            filter_functions=filter_functions,
            filter_type="inlet",
            form_data=form_data,
            extra_params=local_extra_params,
        )
    except Exception as e:
        log.warning(f"Error applying inlet filters: {e}")

    return form_data


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
        # Copy the dict so we don't mutate current_messages across iterations
        msg = {**messages[0]}
        content = msg.get("content", "")
        # Handle structured content (list of {"type": "text", "text": ...})
        # the same way core's update_message_content does.
        if isinstance(content, list):
            msg["content"] = [
                {**item, "text": f'{item["text"]}\n{combined}'}
                if item.get("type") == "text"
                else item
                for item in content
            ]
        else:
            msg["content"] = f"{content}\n\n{combined}" if content else combined
        messages[0] = msg
    else:
        messages.insert(0, {"role": "system", "content": combined})
    form_data["messages"] = messages
    return form_data


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

    # Get model info for filter processing
    models = request.app.state.MODELS
    model = models.get(model_id, {})

    # Build tools parameter for native function calling
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
                        "description": f"Sub-agent iteration {iteration}/{max_iterations}",
                        "done": False,
                    },
                }
            )

        # Build iteration context message
        iteration_info = f"[Iteration {iteration}/{max_iterations}]"
        if iteration == max_iterations:
            iteration_info += " This is your FINAL tool call opportunity."

        # Append iteration info as a meta note. Default role is "user" so the
        # leading system message stays at the beginning of the conversation —
        # some chat templates and inference APIs reject requests where a
        # system message appears after the first turn.
        #
        # When the last message is already ``user`` (the initial request on
        # iteration 1), merging the note into that user message avoids two
        # consecutive user turns, which strict role-alternation validators
        # also reject. Subsequent iterations always end with a ``tool``
        # result (an assistant turn without tool calls exits the loop), so
        # appending a fresh user message there is safe.
        messages_with_context = list(current_messages)
        last = messages_with_context[-1] if messages_with_context else None
        last_role = last.get("role") if isinstance(last, dict) else None
        if iteration_note_role == "user" and last_role == "user":
            merged = dict(last)
            content = merged.get("content", "")
            if isinstance(content, list):
                merged["content"] = content + [
                    {"type": "text", "text": f"\n\n{iteration_info}"}
                ]
            else:
                merged["content"] = (
                    f"{content}\n\n{iteration_info}" if content else iteration_info
                )
            messages_with_context[-1] = merged
        else:
            messages_with_context.append(
                {"role": iteration_note_role, "content": iteration_info}
            )

        # Prepare request
        form_data = {
            "model": model_id,
            "messages": messages_with_context,
            "stream": False,
            "metadata": {
                "task": "sub_agent",
                "sub_agent_iteration": iteration,
                "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
            },
        }

        if tools_param:
            form_data["tools"] = tools_param

        # Apply inlet filters if enabled, then append tool-server prompts
        # (core injects terminal/direct prompts AFTER inlet filters)
        form_data = await apply_inlet_filters_if_enabled(
            apply_inlet_filters, request, model, form_data, extra_params
        )
        form_data = _append_tool_server_prompts(form_data, extra_params)

        try:
            response = await generate_chat_completion(
                request=request,
                form_data=form_data,
                user=user_obj,
                bypass_filter=True,  # We handle filters manually above
            )
        except Exception as e:
            log.exception(f"Error in sub-agent completion: {e}")
            return f"Error during sub-agent execution: {e}"

        # Handle response
        # Handle JSONResponse (returned on HTTP errors like 400+)
        if isinstance(response, JSONResponse):
            try:
                error_data = json.loads(bytes(response.body).decode("utf-8"))
                error_field = error_data.get("error") if isinstance(error_data, dict) else None
                if isinstance(error_field, dict):
                    error_msg = error_field.get("message", str(error_data))
                elif isinstance(error_field, str):
                    error_msg = error_field
                else:
                    error_msg = error_data.get("message", str(error_data)) if isinstance(error_data, dict) else str(error_data)
                return f"API error: {error_msg}"
            except Exception:
                return f"API error (status {response.status_code}): Failed to parse response"

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
                    tools_dict,
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

        else:
            # Unexpected response type
            return f"Unexpected response type: {type(response)}"

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

    # Try to get final response without tools
    form_data = {
        "model": model_id,
        "messages": current_messages
        + [
            {
                "role": "user",
                "content": "Maximum tool iterations reached. Please provide your final answer based on the information gathered so far.",
            }
        ],
        "stream": False,
        "metadata": {
            "task": "sub_agent",
            "sub_agent_iteration": max_iterations + 1,
            "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
        },
    }

    # Apply inlet filters if enabled, then append tool-server prompts
    form_data = await apply_inlet_filters_if_enabled(
        apply_inlet_filters, request, model, form_data, extra_params
    )
    form_data = _append_tool_server_prompts(form_data, extra_params)

    try:
        response = await generate_chat_completion(
            request=request,
            form_data=form_data,
            user=user_obj,
            bypass_filter=True,  # We handle filters manually above
        )

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                choice = choices[0]
                if isinstance(choice, Mapping):
                    message = choice.get("message", {})
                    if isinstance(message, Mapping):
                        return message.get("content", "")
    except Exception as e:
        log.exception(f"Error getting final response: {e}")

    return "Sub-agent reached maximum iterations without providing a final response."


_SKILLS_MANIFEST_START = "<available_skills>"
_SKILLS_MANIFEST_END = "</available_skills>"

_SKILL_TAG_PATTERN = re.compile(
    r"<skill name=.*?>\n.*?\n</skill>", re.DOTALL
)


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
    inline.  The sub-agent can call ``view_skill`` to lazily load their content
    from the ``<available_skills>`` manifest.

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

        callable_fn = await maybe_await(get_async_tool_function_and_apply_extra_params(
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
        log.warning(f"Failed to register view_skill: {e}")


async def load_sub_agent_tools(
    request: Request,
    user: Any,
    valves: Any,
    metadata: dict,
    model: dict,
    extra_params: dict,
    self_tool_id: Optional[str],
) -> tuple[dict, dict]:
    """Load regular, MCP, terminal, direct, and builtin tools for sub-agent."""
    from open_webui.utils.tools import get_builtin_tools, get_tools

    try:
        from open_webui.utils.tools import get_terminal_tools
    except Exception:
        get_terminal_tools = None

    metadata = metadata or {}
    model = model or {}
    extra_params = extra_params or {}
    event_emitter = extra_params.get("__event_emitter__")
    extra_metadata = extra_params.get("__metadata__")
    terminal_id = await resolve_terminal_id_for_sub_agent(
        metadata=metadata,
        request=request,
        debug=bool(getattr(valves, "DEBUG", False)),
    )
    direct_tool_servers = await resolve_direct_tool_servers_for_sub_agent(
        metadata=metadata,
        request=request,
        debug=bool(getattr(valves, "DEBUG", False)),
    )

    if terminal_id:
        metadata["terminal_id"] = terminal_id
        if isinstance(extra_metadata, dict):
            extra_metadata["terminal_id"] = terminal_id
        else:
            extra_params["__metadata__"] = metadata
            extra_metadata = metadata

    if direct_tool_servers:
        metadata["tool_servers"] = direct_tool_servers
        if isinstance(extra_metadata, dict):
            extra_metadata["tool_servers"] = direct_tool_servers
        else:
            extra_params["__metadata__"] = metadata
            extra_metadata = metadata

    # Determine tool IDs
    # Use metadata's tool_ids (main conversation's available tools)
    available_tool_ids = []
    if metadata.get("tool_ids"):
        available_tool_ids = list(metadata.get("tool_ids", []))

    if valves.DEBUG:
        log.info(f"[SubAgent] AVAILABLE_TOOL_IDS valve: '{valves.AVAILABLE_TOOL_IDS}'")
        log.info(f"[SubAgent] Available tool_ids from metadata: {available_tool_ids}")
        log.info(f"[SubAgent] self_tool_id: {self_tool_id}")
        log.info(f"[SubAgent] resolved terminal_id: {terminal_id}")
        log.info(
            f"[SubAgent] resolved direct tool servers: {len(direct_tool_servers)}"
        )

    # Apply exclusions first
    excluded = set()
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

    if valves.DEBUG:
        log.info(f"[SubAgent] EXCLUDED_TOOL_IDS valve: '{valves.EXCLUDED_TOOL_IDS}'")
        if excluded:
            log.info(f"[SubAgent] Excluded tool IDs (including self): {sorted(excluded)}")

    # Determine which tools to use
    if valves.AVAILABLE_TOOL_IDS.strip():
        # Use Valves setting (admin-configured list)
        tool_id_list = [
            tid.strip() for tid in valves.AVAILABLE_TOOL_IDS.split(",") if tid.strip()
        ]
        if valves.DEBUG:
            log.info(f"[SubAgent] Using AVAILABLE_TOOL_IDS valve: {tool_id_list}")
    else:
        # Use all available tools from metadata
        tool_id_list = available_tool_ids
        if valves.DEBUG:
            log.info(
                f"[SubAgent] Using all available tool_ids from metadata: {tool_id_list}"
            )

    tool_id_list = [tid for tid in tool_id_list if tid not in excluded]

    # Note: Builtin tools are NOT included in tool_ids.
    # They are loaded separately via get_builtin_tools() and controlled by Valves.
    # Regular tools only (filter out any builtin: prefixed IDs just in case)
    regular_tool_ids = [tid for tid in tool_id_list if not tid.startswith("builtin:")]
    mcp_tool_ids = [tid for tid in regular_tool_ids if tid.startswith("server:mcp:")]
    non_mcp_tool_ids = [
        tid for tid in regular_tool_ids if not tid.startswith("server:mcp:")
    ]

    if valves.DEBUG:
        log.info(f"[SubAgent] Regular tool IDs: {regular_tool_ids}")
        if mcp_tool_ids:
            log.info(f"[SubAgent] MCP tool IDs: {mcp_tool_ids}")
        if non_mcp_tool_ids != regular_tool_ids:
            log.info(f"[SubAgent] Non-MCP regular tool IDs: {non_mcp_tool_ids}")

    # Load regular tools
    tools_dict = {}
    mcp_clients: dict[str, Any] = {}
    if non_mcp_tool_ids:
        try:
            tools_dict = await get_tools(
                request=request,
                tool_ids=non_mcp_tool_ids,
                user=user,
                extra_params=extra_params,
            )

            if valves.DEBUG:
                log.info(f"[SubAgent] Loaded {len(tools_dict)} regular tools")

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
                debug=bool(getattr(valves, "DEBUG", False)),
            )

            if mcp_tools:
                duplicate_names = set(tools_dict.keys()) & set(mcp_tools.keys())
                tools_dict.update(mcp_tools)
                if valves.DEBUG:
                    if duplicate_names:
                        log.warning(
                            "[SubAgent] MCP tools overrode existing tool names: "
                            f"{sorted(duplicate_names)}"
                        )
                    log.info(f"[SubAgent] Loaded {len(mcp_tools)} MCP tools")
        except Exception as e:
            log.exception(f"Error loading MCP tools: {e}")
            await emit_notification(
                event_emitter,
                level="warning",
                content=f"Could not load MCP tools: {e}",
            )

    # Resolve terminal tools (if available in chat metadata)
    if terminal_id and bool(getattr(valves, "ENABLE_TERMINAL_TOOLS", True)):
        if get_terminal_tools is None:
            if valves.DEBUG:
                log.info("[SubAgent] get_terminal_tools is unavailable in this Open WebUI version")
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
                    if valves.DEBUG:
                        if duplicate_names:
                            log.warning(
                                "[SubAgent] Terminal tools overrode existing tool names: "
                                f"{sorted(duplicate_names)}"
                            )
                        log.info(
                            f"[SubAgent] Loaded {len(terminal_tools)} terminal tools for terminal_id={terminal_id}"
                        )
            except Exception as e:
                log.exception(f"Error loading terminal tools: {e}")
                await emit_notification(
                    event_emitter,
                    level="warning",
                    content=f"Could not load terminal tools: {e}",
                )
    elif terminal_id and valves.DEBUG:
        log.info("[SubAgent] Terminal tools disabled by ENABLE_TERMINAL_TOOLS valve")

    # Resolve direct tool servers
    if direct_tool_servers:
        try:
            direct_tools = build_direct_tools_dict(
                tool_servers=direct_tool_servers,
                debug=bool(getattr(valves, "DEBUG", False)),
            )
            if direct_tools:
                duplicate_names = set(tools_dict.keys()) & set(direct_tools.keys())
                tools_dict = {**tools_dict, **direct_tools}
                direct_tool_server_prompts = extract_direct_tool_server_prompts(direct_tools)
                if direct_tool_server_prompts:
                    extra_params["__direct_tool_server_system_prompts__"] = direct_tool_server_prompts
                else:
                    extra_params.pop("__direct_tool_server_system_prompts__", None)
                if valves.DEBUG:
                    if duplicate_names:
                        log.warning(
                            "[SubAgent] Direct tools overrode existing tool names: "
                            f"{sorted(duplicate_names)}"
                        )
                    log.info(f"[SubAgent] Loaded {len(direct_tools)} direct tools")
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

    # Load builtin tools
    try:
        # Get features from metadata (for memory, etc.)
        features = metadata.get("features", {})

        # NOTE: view_skill is NOT registered here; it's registered manually
        # via register_view_skill() when the parent conversation's
        # <available_skills> manifest is detected (model-attached skills).
        builtin_extra_params = {
            "__user__": extra_params.get("__user__"),
            "__event_emitter__": event_emitter,
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

        # Build set of disabled tools based on Valves
        disabled_builtin_tools = set()
        for valve_field, category in VALVE_TO_CATEGORY.items():
            if not getattr(valves, valve_field):
                disabled_builtin_tools.update(BUILTIN_TOOL_CATEGORIES[category])

        knowledge_tools_enabled = bool(getattr(valves, "ENABLE_KNOWLEDGE_TOOLS", True))
        notes_tools_enabled = bool(getattr(valves, "ENABLE_NOTES_TOOLS", True))
        keep_view_note_for_knowledge = (
            (not notes_tools_enabled)
            and knowledge_tools_enabled
            and model_knowledge_tools_enabled(model)
            and model_has_note_knowledge(model)
        )

        # Filter builtin tools based on Valves settings
        # NOTE: Regular tools take priority over builtin tools with the same name.
        # This allows users to override builtin behavior with custom tools.
        builtin_count = 0
        for name, tool_dict in all_builtin_tools.items():
            if name in disabled_builtin_tools and not (
                name == "view_note" and keep_view_note_for_knowledge
            ):
                continue

            if name not in tools_dict:
                tools_dict[name] = tool_dict
                builtin_count += 1
            elif valves.DEBUG:
                log.warning(
                    f"[SubAgent] Builtin tool '{name}' skipped: "
                    "regular tool with same name takes priority"
                )

        if valves.DEBUG:
            log.info(
                f"[SubAgent] Loaded {builtin_count} builtin tools "
                f"(disabled categories: {[c for v, c in VALVE_TO_CATEGORY.items() if not getattr(valves, v)]}). "
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
            default=10,
            description="Maximum number of tool call iterations for sub-agent.",
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
            description="Apply inlet filters (e.g., user_info_injector) to sub-agent requests. Outlet filters are never applied to sub-agent responses.",
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
        MAX_PARALLEL_AGENTS: int = Field(
            default=5,
            description="Maximum number of sub-agents to run in parallel via run_parallel_sub_agents. To fully disable parallel execution, comment out the run_parallel_sub_agents method.",
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
5. You have a limited number of tool call iterations. Complete the task before reaching the limit.

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
