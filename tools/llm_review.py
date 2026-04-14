"""
title: LLM Review
description: Run an LLM Review collaborative writing process where multiple agents with distinct personas independently draft, cross-review, and revise their work over multiple rounds.
author: https://github.com/skyzi000
version: 0.1.0
license: MIT
required_open_webui_version: 0.7.0
"""

import asyncio
import ast
import json
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Type

from fastapi import Request
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_core_process_tool_result = None


# ============================================================================
# Builtin tool categories (copied from tools/sub_agent.py)
# ============================================================================

# Builtin tool categories
# NOTE: This mapping must be updated when Open WebUI adds/removes builtin tools.
# Check open_webui/utils/tools.py:get_builtin_tools() for the current list.
BUILTIN_TOOL_CATEGORIES = {
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
}

# Mapping from Valves field names to category names
VALVE_TO_CATEGORY = {
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
}

# Tools that generate citation sources
# NOTE: Update this set when new citation-capable tools are added to Open WebUI.
# See open_webui/utils/middleware.py:get_citation_source_from_tool_result() for supported tools.
CITATION_TOOLS = {
    "search_web",
    "view_file",
    "view_knowledge_file",
    "query_knowledge_files",
    "fetch_url",
}


# Terminal tool names that should emit UI refresh/display events.
TERMINAL_EVENT_TOOLS = {
    "display_file",
    "write_file",
    "replace_file_content",
    "run_command",
}


# ============================================================================
# Default personas for LLM Review
# ============================================================================

DEFAULT_PERSONAS = [
    {
        "name": "Analytical Thinker",
        "description": (
            "You approach topics with rigorous logical analysis and systematic thinking. "
            "You value clarity, precision, and evidence-based reasoning. "
            "You excel at identifying assumptions, evaluating arguments, and structuring complex ideas coherently."
        ),
    },
    {
        "name": "Creative Visionary",
        "description": (
            "You bring imaginative and innovative perspectives to any topic. "
            "You value originality, narrative engagement, and thought-provoking ideas. "
            "You excel at finding unexpected connections and presenting ideas in compelling ways."
        ),
    },
    {
        "name": "Practical Strategist",
        "description": (
            "You focus on real-world applicability and actionable insights. "
            "You value practicality, feasibility, and concrete outcomes. "
            "You excel at considering implementation challenges and stakeholder perspectives."
        ),
    },
]


# ============================================================================
# Persona item schema (for Pydantic-based validation in list[dict] args)
# ============================================================================


class PersonaItem(BaseModel):
    name: str = Field(..., description="Short persona label, e.g., 'Analytical Thinker'.")
    description: str = Field(
        ...,
        description="Full persona description driving the agent's voice and priorities.",
    )


# ============================================================================
# Event emitter
# ============================================================================


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):  # type: ignore
        self.event_emitter = event_emitter

    async def emit(
        self, description: str = "Unknown state", done: bool = False
    ) -> None:
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                    },
                }
            )


# ============================================================================
# Helper functions (outside class - AI cannot invoke these)
# ============================================================================


def safe_json_loads(text: str) -> Optional[dict]:
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


async def resolve_terminal_id_from_request_and_metadata(
    *,
    request: Optional[Request],
    metadata: Optional[dict],
    debug: bool = False,
) -> str:
    """Resolve terminal_id from request.body() first, then metadata."""

    def normalize_terminal_id(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    metadata_terminal_id = ""
    if isinstance(metadata, dict):
        metadata_terminal_id = normalize_terminal_id(metadata.get("terminal_id"))

    request_terminal_id = ""
    if request is not None:
        request_body = getattr(request, "body", None)
        if callable(request_body):
            try:
                raw_body = await request_body()
                if raw_body:
                    body = json.loads(raw_body)
                    if isinstance(body, dict):
                        request_terminal_id = normalize_terminal_id(body.get("terminal_id"))
                        if not request_terminal_id:
                            nested_metadata = body.get("metadata")
                            if isinstance(nested_metadata, dict):
                                request_terminal_id = normalize_terminal_id(
                                    nested_metadata.get("terminal_id")
                                )
            except Exception:
                request_terminal_id = ""

    if request_terminal_id:
        if debug and metadata_terminal_id and metadata_terminal_id != request_terminal_id:
            log.warning(
                "[LLMReview] terminal_id mismatch between request body and metadata; "
                "using request body terminal_id to match parent agent behavior"
            )
        return request_terminal_id

    return metadata_terminal_id


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
                "[LLMReview] tool_servers mismatch between request body and metadata; "
                "using request body tool_servers"
            )
        return request_servers

    return metadata_servers


def build_direct_tools_dict(*, tool_servers: list[dict], debug: bool) -> dict:
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
        log.info("[LLMReview] No direct tools loaded from tool_servers")

    return direct_tools


def truncate_text(value: str, limit: int = 200) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


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
        log.debug(f"[LLMReview] Error emitting notification ({level}): {e}")


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
            log.info("[LLMReview] MCPClient unavailable; skipping MCP tool resolution")
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
                log.warning(f"[LLMReview] MCP server with id {server_id} not found")
                await emit_warning(f"MCP server '{server_id}' was not found")
                continue

            if not mcp_server_connection.get("config", {}).get("enable", True):
                if debug:
                    log.info(f"[LLMReview] MCP server {server_id} is disabled; skipping")
                await emit_warning(f"MCP server '{server_id}' is disabled")
                continue

            try:
                has_access = has_connection_access(user, mcp_server_connection)
            except TypeError:
                has_access = has_connection_access(user, mcp_server_connection, None)

            if not has_access:
                log.warning(f"[LLMReview] Access denied to MCP server {server_id} for user {user.id}")
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
                        f"[LLMReview] Error getting OAuth token for MCP server {server_id}: {e}"
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
                    f"[LLMReview] Loaded {loaded_tool_count} MCP tools from server {server_id}"
                )
        except Exception as e:
            log.warning(f"[LLMReview] Failed to load MCP tools from {server_id}: {e}")
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
            log.debug(f"[LLMReview] Error cleaning up MCP client: {e}")


def process_tool_result(
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
        return _core_process_tool_result(
            request,
            tool_function_name,
            tool_result,
            tool_type,
            direct_tool=direct_tool,
            metadata=metadata if isinstance(metadata, dict) else {},
            user=_normalize_user(user),
        )
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


def parse_model_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]

    seen = set()
    result: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        text = normalize_text(item)
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


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


# ============================================================================
# LLM Review prompt builders
# ============================================================================


def build_compose_system_prompt(
    base_prompt: str,
    persona: dict,
    include_sources: bool,
) -> str:
    sources_note = (
        "Include all sources from your research in the sources array."
        if include_sources
        else "If you did not use external sources, leave sources as an empty array."
    )

    persona_name = persona.get("name", "Writer")
    persona_desc = persona.get("description", "")

    compose_prompt = (
        f"You are '{persona_name}', an independent writing agent.\n"
        f"PERSONA: {persona_desc}\n\n"
        "CRITICAL RULES:\n"
        "1. Write independently according to your unique perspective and persona.\n"
        "2. Use available tools proactively if they help your research and writing.\n"
        "3. Maintain your distinct voice throughout all revisions.\n"
        "4. Return ONLY a JSON object as your final response.\n"
        "5. You have limited tool call iterations. Use them wisely.\n\n"
        "FINAL RESPONSE FORMAT (JSON only):\n"
        "- draft: your complete written response to the topic\n"
        "- approach: brief explanation of your writing approach (2-3 sentences)\n"
        "- sources: array of {title, url} objects\n\n"
        f"{sources_note}"
    )

    base_prompt = normalize_text(base_prompt)
    if base_prompt:
        return f"{base_prompt}\n\n{compose_prompt}"
    return compose_prompt


def build_compose_user_prompt(
    topic: str,
    requirements: str,
) -> str:
    return (
        "Writing Task:\n"
        f"- Topic: {topic}\n"
        f"- Requirements: {requirements or 'None specified'}\n\n"
        "Instructions:\n"
        "1. Research the topic using available tools if needed.\n"
        "2. Write a comprehensive draft that reflects your unique perspective.\n"
        "3. Provide your final JSON response with draft, approach, and sources."
    )


def build_review_system_prompt(
    base_prompt: str,
    persona: dict,
) -> str:
    persona_name = persona.get("name", "Reviewer")
    persona_desc = persona.get("description", "")

    review_prompt = (
        f"You are '{persona_name}', providing feedback on a peer's draft.\n"
        f"PERSONA: {persona_desc}\n\n"
        "CRITICAL RULES:\n"
        "1. Provide constructive, specific feedback from your unique perspective.\n"
        "2. Focus on helping the author improve while respecting their voice.\n"
        "3. Be direct but supportive in your critique.\n"
        "4. Return ONLY a JSON object as your final response.\n\n"
        "FINAL RESPONSE FORMAT (JSON only):\n"
        "- strengths: array of specific strengths in the draft\n"
        "- improvements: array of specific, actionable suggestions\n"
        "- key_feedback: one paragraph summarizing your most important feedback"
    )

    base_prompt = normalize_text(base_prompt)
    if base_prompt:
        return f"{base_prompt}\n\n{review_prompt}"
    return review_prompt


def build_review_user_prompt(
    topic: str,
    author_persona_name: str,
    draft: str,
) -> str:
    return (
        f"Review the following draft written by '{author_persona_name}':\n\n"
        f"TOPIC: {topic}\n\n"
        f"DRAFT:\n{draft}\n\n"
        "Provide your feedback in the required JSON format."
    )


def build_revise_system_prompt(
    base_prompt: str,
    persona: dict,
    include_sources: bool,
) -> str:
    sources_note = (
        "Update sources if you conducted additional research."
        if include_sources
        else "If you did not use external sources, leave sources as an empty array."
    )

    persona_name = persona.get("name", "Writer")
    persona_desc = persona.get("description", "")

    revise_prompt = (
        f"You are '{persona_name}', revising your draft based on peer feedback.\n"
        f"PERSONA: {persona_desc}\n\n"
        "CRITICAL RULES:\n"
        "1. Consider the feedback carefully but maintain your unique voice.\n"
        "2. You may use tools to gather additional information if needed.\n"
        "3. Accept suggestions that improve your work; respectfully decline others.\n"
        "4. Return ONLY a JSON object as your final response.\n\n"
        "FINAL RESPONSE FORMAT (JSON only):\n"
        "- draft: your revised draft\n"
        "- changes_made: array of specific changes you made based on feedback\n"
        "- feedback_declined: array of suggestions you chose not to incorporate (with reasons)\n"
        "- sources: array of {title, url} objects\n\n"
        f"{sources_note}"
    )

    base_prompt = normalize_text(base_prompt)
    if base_prompt:
        return f"{base_prompt}\n\n{revise_prompt}"
    return revise_prompt


def build_revise_user_prompt(
    topic: str,
    original_draft: str,
    feedbacks: list[dict],
) -> str:
    feedback_text = ""
    for fb in feedbacks:
        reviewer_name = fb.get("reviewer", "Anonymous")
        key_feedback = fb.get("key_feedback", "No summary provided.")
        strengths = fb.get("strengths", [])
        improvements = fb.get("improvements", [])

        feedback_text += f"\n--- Feedback from '{reviewer_name}' ---\n"
        feedback_text += f"Key Feedback: {key_feedback}\n"
        if strengths:
            feedback_text += "Strengths:\n"
            for s in strengths:
                feedback_text += f"  - {s}\n"
        if improvements:
            feedback_text += "Suggested Improvements:\n"
            for imp in improvements:
                feedback_text += f"  - {imp}\n"

    return (
        f"TOPIC: {topic}\n\n"
        f"YOUR PREVIOUS DRAFT:\n{original_draft}\n\n"
        f"PEER FEEDBACK:{feedback_text}\n\n"
        "Revise your draft based on this feedback. "
        "Maintain your unique perspective while incorporating valuable suggestions. "
        "Provide your response in the required JSON format."
    )


def _normalize_sources(sources: Any) -> list[dict]:
    normalized = []
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict):
                title = normalize_text(source.get("title"))
                url = normalize_text(source.get("url"))
                if title or url:
                    normalized.append({"title": title, "url": url})
    return normalized


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalize_text(item) for item in value if normalize_text(item)]


def normalize_draft_result(parsed: Optional[dict], fallback_text: str) -> dict:
    if not isinstance(parsed, dict):
        return {
            "draft": fallback_text,
            "approach": "",
            "sources": [],
        }

    draft = normalize_text(parsed.get("draft")) or fallback_text
    approach = normalize_text(parsed.get("approach")) or ""

    return {
        "draft": draft,
        "approach": approach,
        "sources": _normalize_sources(parsed.get("sources")),
    }


def normalize_review_result(parsed: Optional[dict], fallback_feedback: str) -> dict:
    if not isinstance(parsed, dict):
        return {
            "strengths": [],
            "improvements": [],
            "key_feedback": fallback_feedback,
        }

    return {
        "strengths": _normalize_str_list(parsed.get("strengths")),
        "improvements": _normalize_str_list(parsed.get("improvements")),
        "key_feedback": normalize_text(parsed.get("key_feedback")) or fallback_feedback,
    }


def normalize_revise_result(parsed: Optional[dict], fallback_draft: str) -> dict:
    if not isinstance(parsed, dict):
        return {
            "draft": fallback_draft,
            "changes_made": [],
            "feedback_declined": [],
            "sources": [],
        }

    return {
        "draft": normalize_text(parsed.get("draft")) or fallback_draft,
        "changes_made": _normalize_str_list(parsed.get("changes_made")),
        "feedback_declined": _normalize_str_list(parsed.get("feedback_declined")),
        "sources": _normalize_sources(parsed.get("sources")),
    }


# ============================================================================
# Agent runtime helpers
# ============================================================================


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

                # Update function with current context.
                # Only override dynamic context — do NOT override __user__ /
                # __request__ / __model__ / __metadata__ / __chat_id__ /
                # __message_id__: get_tools() injected tool-specific UserValves
                # into __user__["valves"] and we must preserve that.
                from open_webui.utils.tools import get_updated_tool_function

                tool_function = get_updated_tool_function(
                    function=tool_function,
                    extra_params={
                        "__messages__": extra_params.get("__messages__", []),
                        "__files__": extra_params.get("__files__", []),
                        "__event_emitter__": extra_params.get("__event_emitter__"),
                        "__event_call__": extra_params.get("__event_call__"),
                    },
                )

                tool_result = await tool_function(**tool_function_params)

            # Handle OpenAPI/external tool results that return (data, headers) tuple
            # Headers (CIMultiDictProxy) are not JSON-serializable
            tool_type = tool.get("type", "")
            tool_result, tool_result_files, tool_result_embeds = process_tool_result(
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
    if not apply_inlet_filters:
        return form_data

    try:
        from open_webui.models.functions import Functions
        from open_webui.utils.filter import get_sorted_filter_ids, process_filter_functions

        # Isolate __user__ so filter UserValves injection doesn't leak out
        # and pollute subsequent tool calls under a different tool id.
        local_extra_params = dict(extra_params or {})
        if isinstance(local_extra_params.get("__user__"), dict):
            local_extra_params["__user__"] = dict(local_extra_params["__user__"])

        filter_ids = get_sorted_filter_ids(
            request, model, form_data.get("metadata", {}).get("filter_ids", [])
        )
        filter_functions = [
            Functions.get_function_by_id(filter_id) for filter_id in filter_ids
        ]
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
        msg = {**messages[0]}
        content = msg.get("content", "")
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


async def run_agent_loop(
    request: Request,
    user: Any,
    model_id: str,
    messages: list[dict],
    tools_dict: dict,
    max_iterations: int,
    event_emitter: Optional[Callable] = None,
    extra_params: Optional[dict] = None,
    apply_inlet_filters: bool = True,
    agent_name: str = "Agent",
) -> str:
    """Run the LLM Review agent tool loop until completion.

    Adapted from sub_agent.run_sub_agent_loop with branded status messages.
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
                "task": "llm_review",
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
            log.exception(f"Error in review agent completion: {e}")
            return f"Error during review agent execution: {e}"

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
                            "description": f"[{agent_name}] Step {iteration}: {content.replace(chr(10), ' ')}",
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
                            "description": f"[{agent_name}] Step {iteration}: Calling {', '.join(tool_names)}",
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
                tool_name = tc_func.get("name", "unknown") if isinstance(tc_func, dict) else "unknown"
                tool_args_raw = tc_func.get("arguments", "{}") if isinstance(tc_func, dict) else "{}"
                tool_args_display = str(tool_args_raw).replace(chr(10), " ") if tool_args_raw else "{}"

                if event_emitter:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[{agent_name}] Executing {tool_name}({tool_args_display})",
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
                    result_content = result["content"].replace(chr(10), ' ') if result["content"] else "(empty)"
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[{agent_name}] {tool_name} returned: {result_content}",
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
                    "description": f"[{agent_name}] Max iterations ({max_iterations}) reached, finalizing...",
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
            "task": "llm_review",
            "sub_agent_iteration": max_iterations + 1,
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
        log.exception(f"Error getting final review agent response: {e}")

    return "Review agent reached maximum iterations without providing a final response."


async def build_tools_dict(
    request: Request,
    model: dict,
    metadata: Optional[dict],
    user: Any,
    valves: Any,
    extra_params: dict,
    tool_id_list: list[str],
    excluded_tool_ids: Optional[set],
    resolved_terminal_id: Optional[str] = None,
    resolved_direct_tool_servers: Optional[list[dict]] = None,
) -> tuple[dict, dict]:
    from open_webui.utils.tools import get_builtin_tools, get_tools

    try:
        from open_webui.utils.tools import get_terminal_tools
    except Exception:
        get_terminal_tools = None

    metadata = metadata or {}
    extra_params = extra_params or {}
    model = model or {}
    tools_dict = {}
    extra_metadata = extra_params.get("__metadata__")
    if resolved_terminal_id is None:
        terminal_id = await resolve_terminal_id_from_request_and_metadata(
            request=request,
            metadata=metadata,
            debug=bool(getattr(valves, "DEBUG", False)),
        )
    else:
        terminal_id = normalize_text(resolved_terminal_id)
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
            debug=bool(getattr(valves, "DEBUG", False)),
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

    # Load regular tools available to the main agent. Open WebUI's get_tools()
    # silently skips ``server:mcp:`` entries, so split them out and resolve via
    # resolve_mcp_tools() like the sub-agent does.
    regular_tool_ids = [tid for tid in tool_id_list if not tid.startswith("builtin:")]
    if excluded_tool_ids:
        regular_tool_ids = [tid for tid in regular_tool_ids if tid not in excluded_tool_ids]

    mcp_tool_ids = [tid for tid in regular_tool_ids if tid.startswith("server:mcp:")]
    non_mcp_tool_ids = [
        tid for tid in regular_tool_ids if not tid.startswith("server:mcp:")
    ]

    if getattr(valves, "DEBUG", False):
        log.info(f"[LLMReview] Regular tool IDs: {regular_tool_ids}")
        if mcp_tool_ids:
            log.info(f"[LLMReview] MCP tool IDs: {mcp_tool_ids}")
        if non_mcp_tool_ids != regular_tool_ids:
            log.info(f"[LLMReview] Non-MCP regular tool IDs: {non_mcp_tool_ids}")

    event_emitter = extra_params.get("__event_emitter__")
    mcp_clients: dict[str, Any] = {}

    if non_mcp_tool_ids:
        try:
            tools_dict = await get_tools(
                request=request,
                tool_ids=non_mcp_tool_ids,
                user=user,
                extra_params=extra_params,
            )
            if getattr(valves, "DEBUG", False):
                log.info(f"[LLMReview] Loaded {len(tools_dict)} regular tools")
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
                if getattr(valves, "DEBUG", False):
                    if duplicate_names:
                        log.warning(
                            "[LLMReview] MCP tools overrode existing tool names: "
                            f"{sorted(duplicate_names)}"
                        )
                    log.info(f"[LLMReview] Loaded {len(mcp_tools)} MCP tools")
        except Exception as e:
            log.exception(f"Error loading MCP tools: {e}")
            await emit_notification(
                event_emitter,
                level="warning",
                content=f"Could not load MCP tools: {e}",
            )

    # Load terminal tools (if available in chat metadata)
    if terminal_id and bool(getattr(valves, "ENABLE_TERMINAL_TOOLS", True)):
        if get_terminal_tools is None:
            if getattr(valves, "DEBUG", False):
                log.info("[LLMReview] get_terminal_tools is unavailable in this Open WebUI version")
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
                    if getattr(valves, "DEBUG", False):
                        if duplicate_names:
                            log.warning(
                                "[LLMReview] Terminal tools overrode existing tool names: "
                                f"{sorted(duplicate_names)}"
                            )
                        log.info(
                            f"[LLMReview] Loaded {len(terminal_tools)} terminal tools for terminal_id={terminal_id}"
                        )
            except Exception as e:
                log.exception(f"Error loading terminal tools: {e}")
                await emit_notification(
                    event_emitter,
                    level="warning",
                    content=f"Could not load terminal tools: {e}",
                )
    elif terminal_id and getattr(valves, "DEBUG", False):
        log.info("[LLMReview] Terminal tools disabled by ENABLE_TERMINAL_TOOLS valve")

    # Load direct tools from metadata.tool_servers
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
                if getattr(valves, "DEBUG", False):
                    if duplicate_names:
                        log.warning(
                            "[LLMReview] Direct tools overrode existing tool names: "
                            f"{sorted(duplicate_names)}"
                        )
                    log.info(f"[LLMReview] Loaded {len(direct_tools)} direct tools")
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
        features = metadata.get("features", {})

        # NOTE: view_skill is NOT registered here; it's registered manually
        # via register_view_skill() when the parent conversation's
        # <available_skills> manifest is detected (model-attached skills).
        builtin_extra_params = {
            "__user__": extra_params.get("__user__"),
            "__event_emitter__": extra_params.get("__event_emitter__"),
            "__event_call__": extra_params.get("__event_call__"),
            "__metadata__": extra_params.get("__metadata__"),
            "__chat_id__": extra_params.get("__chat_id__"),
            "__message_id__": extra_params.get("__message_id__"),
            "__oauth_token__": extra_params.get("__oauth_token__"),
        }

        all_builtin_tools = get_builtin_tools(
            request=request,
            extra_params=builtin_extra_params,
            features=features,
            model=model,
        )

        # Build set of disabled tools based on Valves categories
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

        # Filter builtin tools based on Valves settings.
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
            elif getattr(valves, "DEBUG", False):
                log.warning(
                    f"[LLMReview] Builtin tool '{name}' skipped: "
                    "regular tool with same name takes priority"
                )

        if getattr(valves, "DEBUG", False):
            log.info(
                f"[LLMReview] Loaded {builtin_count} builtin tools "
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
# Skill manifest helpers (mirrors sub_agent.py)
# ============================================================================


_SKILLS_MANIFEST_START = "<available_skills>"
_SKILLS_MANIFEST_END = "</available_skills>"
_SKILL_TAG_PATTERN = re.compile(
    r"<skill name=.*?>\n.*?\n</skill>", re.DOTALL
)


def _find_manifest_in_text(text: str) -> str:
    """Return the <available_skills>...</available_skills> substring, or ""."""
    start = text.find(_SKILLS_MANIFEST_START)
    if start == -1:
        return ""
    end = text.find(_SKILLS_MANIFEST_END, start)
    if end == -1:
        return ""
    return text[start : end + len(_SKILLS_MANIFEST_END)]


def _find_skill_tags_in_text(text: str) -> list[str]:
    """Return all ``<skill name="...">...</skill>`` blocks found in *text*."""
    return _SKILL_TAG_PATTERN.findall(text)


def _extract_from_system_messages(
    messages: Optional[list],
    extractor,
):
    """Walk system messages and apply *extractor* to each text chunk."""
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
                results.append(found) if isinstance(found, str) else results.extend(found)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    found = extractor(part.get("text") or "")
                    if found:
                        results.append(found) if isinstance(found, str) else results.extend(found)
    return results


def extract_skill_manifest(messages: Optional[list]) -> str:
    """Extract the ``<available_skills>`` manifest from the parent
    conversation's system messages.

    Since v0.8.2, only **model-attached** skills appear in this manifest.
    User-selected skills are injected as full ``<skill>`` tags instead.
    """
    results = _extract_from_system_messages(messages, _find_manifest_in_text)
    return results[0] if results else ""


def extract_user_skill_tags(messages: Optional[list]) -> list[str]:
    """Extract ``<skill name="...">content</skill>`` tags from the parent
    conversation's system messages."""
    return _extract_from_system_messages(messages, _find_skill_tags_in_text)


def register_view_skill(
    tools_dict: dict,
    request: Request,
    extra_params: dict,
) -> None:
    """Manually register the view_skill builtin tool in tools_dict.

    Needed for **model-attached** skills whose content is not injected
    inline. The agent can call ``view_skill`` to lazily load their content
    from the ``<available_skills>`` manifest.
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

        callable_fn = get_async_tool_function_and_apply_extra_params(
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
        )

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


async def get_available_models(
    request: Request,
    user: Any,
) -> list[dict]:
    from open_webui.utils.models import get_all_models, get_filtered_models

    all_models = await get_all_models(request, refresh=False, user=user)

    filtered = []
    for model in all_models:
        if "pipeline" in model and model["pipeline"].get("type") == "filter":
            continue
        filtered.append(model)

    return get_filtered_models(filtered, user)


def extract_model_ids(models: list[dict]) -> list[str]:
    ids = []
    seen = set()
    for model in models:
        model_id = normalize_text(model.get("id"))
        if model_id and model_id not in seen:
            seen.add(model_id)
            ids.append(model_id)
    return ids


# ============================================================================
# Tools class
# ============================================================================


class Tools:
    """LLM Review collaborative writing tool."""

    class Valves(BaseModel):
        DEFAULT_MODELS: str = Field(
            default="",
            description="Comma-separated default model IDs (3 expected for LLM Review). If empty, models must be provided in the call.",
        )
        MAX_ITERATIONS: int = Field(
            default=10,
            description="Maximum tool-call iterations per agent during compose/revise phases.",
        )
        REVIEW_MAX_ITERATIONS: int = Field(
            default=2,
            description="Maximum iterations for the review (no-tool) phase.",
        )
        AVAILABLE_TOOL_IDS: str = Field(
            default="",
            description=(
                "[Advanced] Comma-separated list of tool IDs available to review agents. "
                "Leave empty (recommended) to use only tools enabled in the chat UI. "
                "When set, ONLY these tools are available (overrides chat UI tool selection). "
                "This controls regular tools only; builtin tools (web search, memory, etc.) "
                "are controlled separately by the ENABLE_*_TOOLS toggles below. "
                "WARNING: Mismatched tool sets between main AI and review agents can cause failures - "
                "the main AI may instruct the review agents to use tools they don't have. "
                "Tool server IDs (e.g., MCPO/OpenAPI) require 'server:' prefix (e.g., 'server:context7'). "
                "To find exact tool IDs, enable DEBUG, enable the desired tools in the chat UI, "
                "invoke the review, and check server logs for '[LLMReview] Regular tool IDs'."
            ),
        )
        EXCLUDED_TOOL_IDS: str = Field(
            default="",
            description=(
                "Comma-separated list of tool IDs to exclude from review agents (e.g., this tool itself to prevent recursion). "
                "This controls regular tools only; to disable builtin tools, use the ENABLE_*_TOOLS toggles. "
                "If unsure about tool IDs or exclusion behavior, enable DEBUG and check server logs."
            ),
        )
        APPLY_INLET_FILTERS: bool = Field(
            default=True,
            description="Apply inlet filters (e.g., user_info_injector) to review agent requests. Outlet filters are never applied to review agent responses.",
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
            description="Enable skills tools (view_skill). When enabled and the parent conversation has skills, review agents can view skill contents.",
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging.",
        )
        pass

    class UserValves(BaseModel):
        SYSTEM_PROMPT: str = Field(
            default="""\
You are a collaborative writing agent participating in a multi-agent review process.

CRITICAL RULES:
1. Maintain your unique perspective and writing voice throughout.
2. Use tools when they help research and improve your writing.
3. Respond ONLY with the required JSON format.
4. Be open to feedback but don't lose your distinctive style.""",
            description="Base system prompt prepended to every LLM Review agent prompt.",
        )
        INCLUDE_SOURCES: bool = Field(
            default=True,
            description="Include sources in agent outputs when web research is used.",
        )
        ROUNDS: int = Field(
            default=3,
            ge=1,
            le=5,
            description="Number of review-revise rounds (1-5).",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def list_models(
        self,
        query: str = "",
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
    ) -> str:
        """
        List model IDs available to the current user.

        :param query: Optional search string to filter models by ID or name; examples include "gpt", "claude", "gemini", and "ollama", and an empty string returns all available models.

        Returns a JSON string:
        {"models": ["id1", "id2", ...]}
        """
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

        try:
            models = await get_available_models(__request__, user)
            model_ids = extract_model_ids(models)

            query_text = normalize_text(query).lower()
            if query_text:
                filtered_ids = []
                for model in models:
                    model_id = normalize_text(model.get("id"))
                    model_name = normalize_text(model.get("name"))
                    if query_text in model_id.lower() or query_text in model_name.lower():
                        if model_id:
                            filtered_ids.append(model_id)
                model_ids = filtered_ids

            return json.dumps({"models": model_ids}, ensure_ascii=False)
        except Exception as e:
            log.exception(f"Error listing models: {e}")
            return json.dumps(
                {
                    "error": "Failed to list models.",
                    "detail": str(e),
                    "action": "Retry or check server logs for details.",
                },
                ensure_ascii=False,
            )

    async def llm_review(
        self,
        topic: str,
        requirements: Optional[str] = None,
        models: Optional[str] = None,
        personas: Optional[list[PersonaItem]] = None,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __model__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __messages__: Optional[list[dict]] = None,
        __id__: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,  # type: ignore
        __event_call__: Callable[[dict], Any] = None,  # type: ignore
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
        __oauth_token__: Optional[dict] = None,
    ) -> str:
        """
        Run an LLM Review collaborative writing workflow with cross-feedback.

        Exactly 3 agents with distinct personas independently draft the topic,
        exchange peer feedback, and revise over UserValves.ROUNDS rounds (1-5).
        Each agent's voice is preserved across rounds. If `personas` is omitted
        or fewer than 3 entries are provided, built-in default personas are used
        (Analytical Thinker, Creative Visionary, Practical Strategist).

        :param topic: The writing topic or prompt (required).
        :param requirements: Optional specific requirements, constraints, length targets, or style notes.
        :param models: Optional comma-separated model ID list; exactly 3 IDs are required (use list_models() to discover). If omitted or empty, DEFAULT_MODELS (Valves) is used; example "gpt-5.2, claude-4-5-sonnet, gemini-2.5-pro".
        :param personas: Optional list of exactly 3 persona objects, each with `name` and `description` fields; if omitted or fewer than 3 valid entries are provided, default personas (Analytical Thinker, Creative Visionary, Practical Strategist) are used.

        Returns a JSON string with:
        - topic, requirements, num_rounds
        - agents: {model_id: persona_name}
        - final_drafts: {model_id: {persona, draft, sources}}
        - rounds: array including initial composition and each review/revision round
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
        request = __request__
        metadata = __metadata__ or {}
        raw_user_valves = (__user__ or {}).get("valves", {})
        user_valves = coerce_user_valves(raw_user_valves, self.UserValves)

        include_sources = bool(user_valves.INCLUDE_SOURCES)
        num_rounds = int(user_valves.ROUNDS)

        # Extract skills from parent conversation messages.
        # Since v0.8.2, user-selected skills are injected as full <skill> tags,
        # while model-attached skills appear in the <available_skills> manifest.
        # Matches sub_agent's injection order: SYSTEM_PROMPT, user tags, manifest.
        skill_manifest = extract_skill_manifest(__messages__)
        user_skill_tags = extract_user_skill_tags(__messages__)

        prompt_sections: list[str] = [str(user_valves.SYSTEM_PROMPT)]
        if bool(getattr(self.valves, "ENABLE_SKILLS_TOOLS", True)):
            if user_skill_tags:
                prompt_sections.extend(user_skill_tags)
            if skill_manifest:
                prompt_sections.append(skill_manifest)
        base_system_prompt = merge_prompt_sections(*prompt_sections)

        topic_text = normalize_text(topic)
        requirements_text = normalize_text(requirements)

        if not topic_text:
            return json.dumps(
                {
                    "error": "Missing required field: topic.",
                    "action": "Provide the topic parameter for LLM Review.",
                },
                ensure_ascii=False,
            )

        # Parse personas
        agent_personas: list[dict] = []
        if personas:
            for p in personas:
                if isinstance(p, PersonaItem):
                    name = normalize_text(p.name)
                    description = normalize_text(p.description)
                elif isinstance(p, dict):
                    name = normalize_text(p.get("name"))
                    description = normalize_text(p.get("description"))
                else:
                    continue
                if name:
                    agent_personas.append(
                        {"name": name, "description": description}
                    )

        if len(agent_personas) < 3:
            agent_personas = [dict(p) for p in DEFAULT_PERSONAS]

        # Resolve model IDs
        provided_models = parse_model_ids(models)
        default_models = parse_model_ids(self.valves.DEFAULT_MODELS)

        model_ids = provided_models or default_models
        if len(model_ids) < 3:
            try:
                available_models = await get_available_models(__request__, user)
                available_ids = extract_model_ids(available_models)
            except Exception:
                available_ids = []

            return json.dumps(
                {
                    "error": "Exactly 3 model IDs are required for LLM Review.",
                    "provided_models": model_ids,
                    "available_models": available_ids,
                    "action": "Provide exactly 3 model IDs in the models parameter (or DEFAULT_MODELS valve).",
                },
                ensure_ascii=False,
            )

        # Use exactly 3 models
        model_ids = model_ids[:3]

        # Validate model IDs
        try:
            available_models = await get_available_models(__request__, user)
            available_ids = set(extract_model_ids(available_models))
        except Exception as e:
            log.exception(f"Error checking available models: {e}")
            return json.dumps(
                {
                    "error": "Failed to validate model IDs.",
                    "detail": str(e),
                    "action": "Retry or check server logs for details.",
                },
                ensure_ascii=False,
            )

        unknown_ids = [mid for mid in model_ids if mid not in available_ids]
        if unknown_ids:
            return json.dumps(
                {
                    "error": "Unknown model IDs.",
                    "unknown_models": unknown_ids,
                    "available_models": sorted(list(available_ids)),
                    "action": "Pick model IDs from available_models or call list_models().",
                },
                ensure_ascii=False,
            )

        # Resolve terminal_id regardless of ENABLE_TERMINAL_TOOLS so metadata
        # downstream consumers (filters etc.) see the same context as
        # sub_agent. The valve only gates terminal *tool loading*.
        resolved_terminal_id = await resolve_terminal_id_from_request_and_metadata(
            request=request,
            metadata=metadata,
            debug=bool(getattr(self.valves, "DEBUG", False)),
        )
        if resolved_terminal_id:
            metadata["terminal_id"] = resolved_terminal_id

        resolved_direct_tool_servers = await resolve_direct_tool_servers_from_request_and_metadata(
            request=request,
            metadata=metadata,
            debug=bool(getattr(self.valves, "DEBUG", False)),
        )
        if resolved_direct_tool_servers:
            metadata["tool_servers"] = resolved_direct_tool_servers

        extra_params = {
            "__user__": __user__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__request__": request,
            "__model__": __model__,
            "__metadata__": metadata,
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
            "__oauth_token__": __oauth_token__,
            "__files__": metadata.get("files", []),
        }

        # Build tool IDs
        available_tool_ids = []
        if metadata.get("tool_ids"):
            available_tool_ids = list(metadata.get("tool_ids", []))

        if self.valves.DEBUG:
            log.info(f"[LLMReview] AVAILABLE_TOOL_IDS valve: '{self.valves.AVAILABLE_TOOL_IDS}'")
            log.info(f"[LLMReview] Available tool_ids from metadata: {available_tool_ids}")
            log.info(f"[LLMReview] self_tool_id (__id__): {__id__}")
            log.info(f"[LLMReview] resolved terminal_id: {resolved_terminal_id}")
            log.info(
                f"[LLMReview] resolved direct tool servers: {len(resolved_direct_tool_servers)}"
            )

        if self.valves.AVAILABLE_TOOL_IDS.strip():
            tool_id_list = [
                tid.strip()
                for tid in self.valves.AVAILABLE_TOOL_IDS.split(",")
                if tid.strip()
            ]
            if self.valves.DEBUG:
                log.info(f"[LLMReview] Using AVAILABLE_TOOL_IDS valve: {tool_id_list}")
        else:
            tool_id_list = available_tool_ids
            if self.valves.DEBUG:
                log.info(
                    f"[LLMReview] Using all available tool_ids from metadata: {tool_id_list}"
                )

        excluded_tool_ids = set()
        if self.valves.EXCLUDED_TOOL_IDS.strip():
            excluded_tool_ids = {
                tid.strip()
                for tid in self.valves.EXCLUDED_TOOL_IDS.split(",")
                if tid.strip()
            }

        if not __id__:
            log.warning(
                "[LLMReview] __id__ is None, cannot exclude self from tool list. "
                "Recursion prevention may not work."
            )
        else:
            excluded_tool_ids.add(__id__)

        if self.valves.DEBUG:
            log.info(f"[LLMReview] EXCLUDED_TOOL_IDS valve: '{self.valves.EXCLUDED_TOOL_IDS}'")
            if excluded_tool_ids:
                log.info(
                    f"[LLMReview] Excluded tool IDs (including self): {sorted(excluded_tool_ids)}"
                )

        # Assign personas to models (first 3 personas)
        agents = {model_ids[i]: agent_personas[i] for i in range(3)}

        await emitter.emit(
            description=f"LLM Review Starting: {topic_text}",
            done=False,
        )

        # Cache (tools_dict, terminal_prompt, direct_prompts) per model so the
        # tool-server system prompts surfaced by build_tools_dict via mutation
        # of extra_params survive into every run_agent_loop invocation.
        tools_cache: dict[str, tuple[dict, Optional[str], list[str]]] = {}
        all_mcp_clients: list[dict[str, Any]] = []
        rounds_data: list[dict] = []
        current_drafts: dict[str, dict] = {}

        async def ensure_tools(
            model_id: str,
        ) -> tuple[dict, Optional[str], list[str]]:
            cached = tools_cache.get(model_id)
            if cached is not None:
                return cached
            member_model = request.app.state.MODELS.get(model_id, {})
            # build_tools_dict mutates this dict to set __terminal_system_prompt__
            # and __direct_tool_server_system_prompts__ — keep it captive.
            scratch_extra_params = {
                **extra_params,
                "__model__": member_model,
            }
            tools_dict, mcp_clients = await build_tools_dict(
                request=request,
                model=member_model,
                metadata=metadata,
                user=user,
                valves=self.valves,
                extra_params=scratch_extra_params,
                tool_id_list=tool_id_list,
                excluded_tool_ids=excluded_tool_ids,
                resolved_terminal_id=resolved_terminal_id,
                resolved_direct_tool_servers=resolved_direct_tool_servers,
            )
            # Manually register view_skill if the parent conversation has a
            # skills manifest and skills tools are enabled (model-attached
            # skills only — user-selected skills are inlined elsewhere).
            if (
                skill_manifest
                and bool(getattr(self.valves, "ENABLE_SKILLS_TOOLS", True))
            ):
                register_view_skill(tools_dict, request, scratch_extra_params)

            terminal_prompt = scratch_extra_params.get("__terminal_system_prompt__")
            direct_prompts = scratch_extra_params.get(
                "__direct_tool_server_system_prompts__", []
            ) or []
            cached = (tools_dict, terminal_prompt, list(direct_prompts))
            tools_cache[model_id] = cached
            if mcp_clients:
                all_mcp_clients.append(mcp_clients)
            return cached

        def make_member_extra_params(
            model_id: str,
            terminal_prompt: Optional[str],
            direct_prompts: list[str],
        ) -> dict:
            member_model = request.app.state.MODELS.get(model_id, {})
            params = {
                **extra_params,
                "__model__": member_model,
            }
            if terminal_prompt:
                params["__terminal_system_prompt__"] = terminal_prompt
            if direct_prompts:
                params["__direct_tool_server_system_prompts__"] = list(direct_prompts)
            return params

        try:
            # ---------- Phase 1: Initial Composition ----------
            await emitter.emit(
                description="Phase 1: Initial Composition",
                done=False,
            )

            async def compose_draft(model_id: str) -> tuple[str, str, dict]:
                persona = agents[model_id]
                tools_dict, terminal_prompt, direct_prompts = await ensure_tools(
                    model_id
                )
                member_extra_params = make_member_extra_params(
                    model_id, terminal_prompt, direct_prompts
                )

                system_prompt = build_compose_system_prompt(
                    base_system_prompt, persona, include_sources
                )
                user_prompt = build_compose_user_prompt(topic_text, requirements_text)

                content = await run_agent_loop(
                    request=request,
                    user=user,
                    model_id=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tools_dict=tools_dict,
                    max_iterations=self.valves.MAX_ITERATIONS,
                    extra_params=member_extra_params,
                    apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                    agent_name=f"{persona['name']}",
                    event_emitter=__event_emitter__,
                )

                parsed = safe_json_loads(content)
                return model_id, content, parsed or {}

            compose_results = await asyncio.gather(
                *[compose_draft(mid) for mid in model_ids], return_exceptions=True
            )

            for idx, result in enumerate(compose_results):
                model_id = model_ids[idx]
                if isinstance(result, BaseException):
                    log.exception(f"Compose failed ({model_id}): {result}")
                    current_drafts[model_id] = normalize_draft_result(
                        None, f"Composition failed: {truncate_text(str(result), 180)}"
                    )
                    continue

                mid, content, parsed = result
                current_drafts[mid] = normalize_draft_result(
                    parsed, f"Non-JSON response: {truncate_text(content, 180)}"
                )

            rounds_data.append(
                {
                    "round": 0,
                    "phase": "initial_composition",
                    "drafts": {
                        mid: {
                            "persona": agents[mid]["name"],
                            "draft": current_drafts[mid]["draft"],
                            "approach": current_drafts[mid].get("approach", ""),
                            "sources": current_drafts[mid].get("sources", []),
                        }
                        for mid in model_ids
                    },
                }
            )

            # ---------- Review / Revision Rounds ----------
            for round_num in range(1, num_rounds + 1):
                await emitter.emit(
                    description=f"Round {round_num}/{num_rounds}: Cross-Review",
                    done=False,
                )

                all_reviews: dict[str, dict[str, dict]] = {mid: {} for mid in model_ids}

                async def review_draft(
                    reviewer_id: str, author_id: str
                ) -> tuple[str, str, str, dict]:
                    reviewer_persona = agents[reviewer_id]
                    author_persona = agents[author_id]
                    author_draft = current_drafts[author_id]["draft"]

                    # Reviews don't use tools, but reuse the cached prompts so
                    # any inlet filter referencing them sees consistent context.
                    _, terminal_prompt, direct_prompts = await ensure_tools(
                        reviewer_id
                    )
                    member_extra_params = make_member_extra_params(
                        reviewer_id, terminal_prompt, direct_prompts
                    )

                    system_prompt = build_review_system_prompt(
                        base_system_prompt, reviewer_persona
                    )
                    user_prompt = build_review_user_prompt(
                        topic_text, author_persona["name"], author_draft
                    )

                    content = await run_agent_loop(
                        request=request,
                        user=user,
                        model_id=reviewer_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        tools_dict={},  # No tools for review
                        max_iterations=int(self.valves.REVIEW_MAX_ITERATIONS),
                        extra_params=member_extra_params,
                        apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                        agent_name=f"{reviewer_persona['name']}\u2192{author_persona['name']}",
                        event_emitter=__event_emitter__,
                    )

                    parsed = safe_json_loads(content)
                    return reviewer_id, author_id, content, parsed or {}

                review_tasks = [
                    review_draft(reviewer_id, author_id)
                    for reviewer_id in model_ids
                    for author_id in model_ids
                    if reviewer_id != author_id
                ]
                review_results = await asyncio.gather(*review_tasks, return_exceptions=True)

                for result in review_results:
                    if isinstance(result, BaseException):
                        log.exception(f"Review failed: {result}")
                        continue

                    reviewer_id, author_id, content, parsed = result
                    normalized = normalize_review_result(
                        parsed, f"Review parsing failed: {truncate_text(content, 100)}"
                    )
                    normalized["reviewer"] = agents[reviewer_id]["name"]
                    all_reviews[author_id][reviewer_id] = normalized

                await emitter.emit(
                    description=f"Round {round_num}/{num_rounds}: Revision",
                    done=False,
                )

                async def revise_draft(model_id: str) -> tuple[str, str, dict]:
                    persona = agents[model_id]
                    original_draft = current_drafts[model_id]["draft"]
                    feedbacks = list(all_reviews[model_id].values())

                    tools_dict, terminal_prompt, direct_prompts = await ensure_tools(
                        model_id
                    )
                    member_extra_params = make_member_extra_params(
                        model_id, terminal_prompt, direct_prompts
                    )

                    system_prompt = build_revise_system_prompt(
                        base_system_prompt, persona, include_sources
                    )
                    user_prompt = build_revise_user_prompt(
                        topic_text, original_draft, feedbacks
                    )

                    content = await run_agent_loop(
                        request=request,
                        user=user,
                        model_id=model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        tools_dict=tools_dict,
                        max_iterations=self.valves.MAX_ITERATIONS,
                        extra_params=member_extra_params,
                        apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                        agent_name=f"{persona['name']}",
                        event_emitter=__event_emitter__,
                    )

                    parsed = safe_json_loads(content)
                    return model_id, content, parsed or {}

                revise_results = await asyncio.gather(
                    *[revise_draft(mid) for mid in model_ids], return_exceptions=True
                )

                revised_drafts: dict[str, dict] = {}
                for idx, result in enumerate(revise_results):
                    model_id = model_ids[idx]
                    if isinstance(result, BaseException):
                        log.exception(f"Revision failed ({model_id}): {result}")
                        revised_drafts[model_id] = normalize_revise_result(
                            None, current_drafts[model_id]["draft"]
                        )
                        continue

                    mid, content, parsed = result
                    revised_drafts[mid] = normalize_revise_result(
                        parsed, current_drafts[mid]["draft"]
                    )

                # Update current drafts with revisions. Merge prior + new sources
                # by (title, url) to preserve citations across rounds even when
                # the revision agent only reports newly gathered sources or omits
                # the array entirely. Order: prior sources first, then new ones.
                for mid in model_ids:
                    prior_sources = current_drafts[mid].get("sources", []) or []
                    new_sources = revised_drafts[mid].get("sources") or []
                    seen: set = set()
                    merged_sources: list[dict] = []
                    for src in list(prior_sources) + list(new_sources):
                        if not isinstance(src, dict):
                            continue
                        title = normalize_text(src.get("title"))
                        url = normalize_text(src.get("url"))
                        if not (title or url):
                            continue
                        key = (title, url)
                        if key in seen:
                            continue
                        seen.add(key)
                        merged_sources.append({"title": title, "url": url})
                    current_drafts[mid] = {
                        "draft": revised_drafts[mid]["draft"],
                        "approach": current_drafts[mid].get("approach", ""),
                        "sources": merged_sources,
                    }

                rounds_data.append(
                    {
                        "round": round_num,
                        "reviews": {
                            mid: {
                                reviewer_id: {
                                    "reviewer": review["reviewer"],
                                    "key_feedback": review["key_feedback"],
                                    "strengths": review["strengths"],
                                    "improvements": review["improvements"],
                                }
                                for reviewer_id, review in all_reviews[mid].items()
                            }
                            for mid in model_ids
                        },
                        "revisions": {
                            mid: {
                                "persona": agents[mid]["name"],
                                "draft": revised_drafts[mid]["draft"],
                                "changes_made": revised_drafts[mid].get("changes_made", []),
                                "feedback_declined": revised_drafts[mid].get(
                                    "feedback_declined", []
                                ),
                                # Use merged sources (prior + new) so the round
                                # history shows the full citation accumulation.
                                "sources": current_drafts[mid].get("sources", []),
                            }
                            for mid in model_ids
                        },
                    }
                )

            await emitter.emit(
                description="LLM Review Complete",
                done=True,
            )

            final_drafts = {
                mid: {
                    "persona": agents[mid]["name"],
                    "draft": current_drafts[mid]["draft"],
                    "sources": current_drafts[mid].get("sources", []),
                }
                for mid in model_ids
            }

            response_payload = {
                "topic": topic_text,
                "requirements": requirements_text,
                "num_rounds": num_rounds,
                "agents": {mid: agents[mid]["name"] for mid in model_ids},
                "final_drafts": final_drafts,
                "rounds": rounds_data,
            }

            return json.dumps(response_payload, ensure_ascii=False)
        finally:
            for mcp_clients in all_mcp_clients:
                await cleanup_mcp_clients(mcp_clients)
