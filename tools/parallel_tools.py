"""
title: Parallel Tools
author: skyzi000
version: 0.2.12
license: MIT
required_open_webui_version: 0.11.0
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
import asyncio
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


_UPLOADED_FILE_ID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def is_terminal_tool(tool: dict) -> bool:
    return tool.get("type") == "terminal" or (tool.get("server") or {}).get("is_terminal") is True


def resolve_display_file_tool(tool: dict, params: dict) -> tuple[Optional[dict], Optional[str]]:
    """Send an uploaded file's ``display_file`` call to Core's builtin, as Core does for terminal tools."""
    path = str(params.get("path", ""))
    if not (path.startswith("/api/v1/files/") or _UPLOADED_FILE_ID.fullmatch(path)):
        return tool, None
    if tool.get("file_tool"):
        return tool["file_tool"], None
    try:
        from open_webui.tools.builtin import display_file  # noqa: F401
    except ImportError:
        # Core before 0.12.5 sends every display_file path to the terminal.
        return tool, None
    if is_terminal_tool(tool):
        return None, "Error: Uploaded file display is not available."
    return tool, None


def attach_displayed_upload(tool_result: Any, files: list) -> Any:
    """Attach the uploaded file ``display_file`` showed to the reply, since a nested call cannot preview it inline."""
    try:
        result = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
    except ValueError:
        return tool_result
    if not (isinstance(result, dict) and result.get("source") == "open_webui" and result.get("exists") is True):
        return tool_result
    files.append(
        {
            "type": "file",
            "id": result["path"].split("/")[4],
            "name": result.get("name"),
            "url": result["path"],
            "content_type": result.get("content_type"),
        }
    )
    if not result.get("page"):
        return tool_result
    return json.dumps(
        {**result, "note": "The file is attached to the reply instead of previewed, so the page was not opened."},
        ensure_ascii=False,
    )


def core_version() -> tuple[int, ...]:
    from open_webui.env import VERSION

    return tuple(int(part) for part in re.findall(r"\d+", VERSION)[:2])


def _saved_reply(metadata: Optional[dict]) -> bool:
    metadata = metadata or {}
    # Core before 0.12.0 adds message files to the chat history only, not to the message table it reads them from.
    if core_version() < (0, 12) or not (metadata.get("user_id") and metadata.get("message_id")):
        return False
    from open_webui.utils.chat_id import is_saved_chat_id

    return is_saved_chat_id(metadata.get("chat_id"))


def _reply_file_lock(request: Any, chat_id: str, message_id: str) -> asyncio.Lock:
    """Share one lock per reply with nested agent tools, which get copies of this request's state."""
    state = getattr(request, "state", None)
    locks = getattr(state, "reply_file_locks", None)
    if locks is None:
        locks = {}
        if state is not None:
            state.reply_file_locks = locks
    return locks.setdefault((chat_id, message_id), asyncio.Lock())


def order_file_events(event_emitter: Optional[Callable], metadata: Optional[dict], request: Any) -> Optional[Callable]:
    """Add each file event's new files to a saved chat's reply in one write, then show everyone viewing it the full list.

    Core saves a file event by reading the reply's files and writing them back, which drops files another
    tool adds in between, and its chat screen replaces the reply's files with each event's list.
    """
    if not callable(event_emitter) or not _saved_reply(metadata):
        return event_emitter
    from open_webui.models.chats import Chats
    from open_webui.socket.main import get_event_emitter

    chat_id, message_id = metadata["chat_id"], metadata["message_id"]
    lock = _reply_file_lock(request, chat_id, message_id)

    async def emit(event_data):
        event_type = event_data.get("type") if isinstance(event_data, dict) else None
        if not (event_type == "files" or (event_type == "chat:message:files" and event_data.get(_ALREADY_SAVED))):
            return await event_emitter(event_data)
        data = event_data.get("data") or {}
        files = list(data.get("files") or [])
        async with lock:
            saved, added = [], files
            try:
                stored = await Chats.get_message_metadata(chat_id, message_id, "files")
                saved = stored if isinstance(stored, list) else []
                added = [file for index, file in enumerate(files) if file not in saved and file not in files[:index]]
                if added:
                    saved = await Chats.add_message_files_by_id_and_message_id(chat_id, message_id, added) or saved
            except Exception:
                _tool_execution_log.exception("Could not save the reply's files")
            show = await get_event_emitter(metadata, update_db=False)
            # Core's emitter shows each event before saving it and goes on when the save fails.
            await show({"type": "files", "data": {**data, "files": saved + [file for file in added if file not in saved]}})

    return emit


_ALREADY_SAVED = "already_saved"


def mark_saved_image_files(event_emitter: Optional[Callable], tool: dict, tool_name: str) -> Optional[Callable]:
    """Mark the file list Core's image tools send after saving, so ``order_file_events`` shows the reply's current list.

    Core ignores the extra key, so an emitter without ``order_file_events`` handles the event as sent.
    """
    if not (callable(event_emitter) and tool.get("type") == "builtin" and tool_name in ("generate_image", "edit_image")):
        return event_emitter

    async def emit(event_data):
        if isinstance(event_data, dict) and event_data.get("type") == "chat:message:files":
            event_data = {**event_data, _ALREADY_SAVED: True}
        return await event_emitter(event_data)

    return emit


def _model_has_vision(model: Optional[dict]) -> bool:
    meta = ((model or {}).get("info") or {}).get("meta") or {}
    return bool((meta.get("capabilities") or {}).get("vision", True))


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


async def split_tool_result_files(
    files: list, user: Any = None, fetch_url_images: bool = True
) -> tuple[list[str], list]:
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
            # Core before 0.12.0 keeps URL images display-only.
            if core_version() >= (0, 12):
                if not fetch_url_images:
                    images.append(url)
                    continue
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
        if isinstance(parsed, dict) and (parsed.get("exists") is False or parsed.get("source") == "open_webui"):
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


def filter_tool_arguments(tool: dict, arguments: dict) -> dict:
    """Keep the *arguments* *tool*'s spec lists, as Core does; MCP servers validate their own."""
    if tool.get("type") == "mcp":
        return arguments
    allowed = tool.get("spec", {}).get("parameters", {}).get("properties", {}).keys()
    return {k: v for k, v in arguments.items() if k in allowed}


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
    vision = _model_has_vision(extra_params.get("__model__"))
    tool_result_images: list[str] = []
    emit_terminal_event = False
    if tool_function_name in tools_dict:
        tool = tools_dict[tool_function_name]
        if tool_function_name == "display_file":
            tool, error = resolve_display_file_tool(tool, tool_function_params)
            if error:
                return {"tool_call_id": tool_call_id, "content": error}
        direct_tool = bool(tool.get("direct", False))
        metadata = extra_params.get("__metadata__")
        core_metadata = getattr(extra_params.get("__event_call__"), "core_metadata", None)
        if isinstance(metadata, dict) and isinstance(core_metadata, dict) and "session_id" in metadata:
            metadata["session_id"] = core_metadata.get("session_id")

        try:
            tool_function_params = filter_tool_arguments(tool, tool_function_params)

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
                    "__files__": (
                        metadata.get("files")
                        if isinstance(metadata, dict)
                        else extra_params.get("__files__")
                    )
                    or [],
                    "__event_emitter__": mark_saved_image_files(
                        extra_params.get("__event_emitter__"), tool, tool_function_name
                    ),
                    "__event_call__": extra_params.get("__event_call__"),
                }
                # Loaded tools are shared by agents, while Core binds each
                # chat's own metadata to its tools.
                if isinstance(metadata, dict):
                    call_params["__metadata__"] = metadata
                if extra_params.get("__request__") is not None:
                    call_params["__request__"] = extra_params["__request__"]
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
                tool_result_files, extra_params.get("__user__"), fetch_url_images=vision
            )
            if tool_function_name == "display_file" and tool.get("type") == "builtin":
                tool_result = attach_displayed_upload(tool_result, tool_result_files)
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

    if tool_result_images and not vision:
        tool_result += "\n[image]" * len(tool_result_images)
        tool_result_images = []

    return {
        "tool_call_id": tool_call_id,
        "content": tool_result,
        **({"images": tool_result_images} if tool_result_images else {}),
    }

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
        "display_file",
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
        "display_file",
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
    "tool_creation": {"create_tool"},
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
    "ENABLE_TOOL_CREATION_TOOLS": "tool_creation",
    "ENABLE_SUBAGENT_TOOLS": "subagents",
    "ENABLE_TASK_TOOLS": "tasks",
    "ENABLE_AUTOMATION_TOOLS": "automations",
    "ENABLE_CALENDAR_TOOLS": "calendar",
    "ENABLE_NOTIFICATION_TOOLS": "notifications",
}

log = logging.getLogger(__name__)


class ToolCallItem(BaseModel):
    """A single tool call specification."""

    name: str = Field(description="Tool function name to call")
    args: dict = Field(default_factory=dict, description="Arguments to pass to the tool")


# ============================================================================
# Helper functions (outside class - AI cannot invoke these)
# ============================================================================


def select_caller_tools(
    caller_tools: dict, mcp_clients: dict, valves: Any, excluded_ids: set, available_ids: set
) -> dict:
    """Return the caller's tools that Parallel Tools may run, under the names the caller offered them."""
    # Core's MCP entries carry their client instead of a tool ID.
    mcp_tool_ids = {
        id(client): f"server:mcp:{server_id}" for server_id, client in (mcp_clients or {}).items()
    }
    disabled_builtins: set = set()
    # Core before 0.12.0 keeps a single browser event callback and no call
    # timeout, so concurrent ask_user prompts can wait forever.
    if core_version() < (0, 12):
        disabled_builtins.update(BUILTIN_TOOL_CATEGORIES["user_input"])
    for valve_field, category in VALVE_TO_CATEGORY.items():
        if not getattr(valves, valve_field, True):
            disabled_builtins.update(BUILTIN_TOOL_CATEGORIES.get(category, set()))

    selected = {}
    for name, tool in caller_tools.items():
        if not isinstance(tool, dict):
            continue
        if is_terminal_tool(tool) and not valves.ENABLE_TERMINAL_TOOLS:
            tool = tool.get("file_tool") if name == "display_file" else None
            if not isinstance(tool, dict):
                continue
        tool_id = tool.get("tool_id") or mcp_tool_ids.get(id(tool.get("client")), "")
        if tool_id in excluded_ids:
            continue
        if tool.get("type") == "builtin":
            if name in disabled_builtins:
                continue
        elif (
            available_ids
            and not tool.get("direct")
            and not is_terminal_tool(tool)
            and tool_id not in available_ids
        ):
            continue
        selected[name] = tool
    return selected


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
    if tool_name == "display_file":
        tool, error = resolve_display_file_tool(tool, tool_args)
        if error:
            return {"tool_name": tool_name, "result": error}

    try:
        filtered_args = filter_tool_arguments(tool, tool_args)
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
                    "__event_emitter__": mark_saved_image_files(extra_params.get("__event_emitter__"), tool, tool_name),
                    "__event_call__": extra_params.get("__event_call__"),
                    # The caller's tools are bound to the request and metadata of the agent
                    # that loaded them; a sub-agent's note reads live on its own request.
                    "__metadata__": extra_params.get("__metadata__"),
                    "__request__": extra_params.get("__request__"),
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
        if tool_name == "display_file" and tool.get("type") == "builtin":
            result = attach_displayed_upload(result, tool_result_files)

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
                "[Advanced] Comma-separated list of regular or MCP tool IDs to allow for parallel execution, "
                "among the tools the current chat offers; it never adds others. "
                "Leave empty (recommended) to allow every regular and MCP tool the chat offers. "
                "Tool server IDs require their full prefix (e.g., 'server:mcp:context7' or 'server:context7'). "
                "Terminal tools are controlled by ENABLE_TERMINAL_TOOLS, and builtin/direct tools follow the current chat."
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
                "Enable the Open Terminal tools the current chat offers "
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

        __metadata__ = __metadata__ or {}
        __event_emitter__ = order_file_events(__event_emitter__, __metadata__, __request__)

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

        available = {
            tid.strip() for tid in self.valves.AVAILABLE_TOOL_IDS.split(",") if tid.strip()
        }
        excluded = {
            tid.strip() for tid in self.valves.EXCLUDED_TOOL_IDS.split(",") if tid.strip()
        }
        # Exclude this tool itself
        if __id__:
            excluded.add(__id__)

        if self.valves.DEBUG:
            if available:
                log.info(f"[ParallelTools] Allowed tool IDs: {sorted(available)}")
            if excluded:
                log.info(f"[ParallelTools] Excluded tool IDs: {sorted(excluded)}")

        # Core resolved the caller's tools under its own names and gates, for the
        # model that runs the chat (an Arena's pick, which tools cannot identify).
        caller_tools = __metadata__.get("tools")
        tools_dict = select_caller_tools(
            caller_tools if isinstance(caller_tools, dict) else {},
            __metadata__.get("mcp_clients") or {},
            self.valves,
            excluded,
            available,
        )

        if self.valves.DEBUG:
            log.info(f"[ParallelTools] Total tools available: {len(tools_dict)}")

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
