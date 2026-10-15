"""Tool-call orchestration shared across owui_ext tool plugins.

The module owns the canonical agent-shape ``execute_tool_call`` together
with its sub-helpers:

- ``process_tool_result`` delegates to Open WebUI's core helper of the
  same name so MCP resources, files, embeds, and direct-tool results keep
  the exact core behavior.
- ``execute_direct_tool_call`` runs direct tools through ``__event_call__``
  the way core middleware does.
- ``normalize_terminal_tools_result`` flattens the tuple-vs-dict return
  shape that ``get_terminal_tools()`` exposes across versions and lifts
  the terminal system prompt into ``extra_params`` for downstream use.
- ``emit_terminal_tool_event`` translates a successful Open Terminal
  builtin tool result into the matching ``terminal:*`` UI event.
- ``execute_tool_call`` orchestrates the above and only injects the
  per-call dynamic context (``__messages__`` / ``__files__`` /
  ``__event_emitter__`` / ``__event_call__``, plus the calling agent's
  ``__metadata__`` for builtin tools) — never ``__user__`` /
  ``__request__`` / ``__model__`` / ``__chat_id__`` /
  ``__message_id__``, because ``get_tools()`` already attached tool-specific
  ``UserValves`` to ``__user__["valves"]`` and overriding it would
  silently overwrite per-tool valves.

The constants ``CITATION_TOOLS`` and ``TERMINAL_EVENT_TOOLS`` describe
which tool names trigger citation sources / terminal UI events. They
live here (not in a separate metadata module) so ``execute_tool_call``
can default to them without forcing callers to inject them — the inliner
forbids a shared dep module from mixing external imports with imports
from other ``owui_ext.shared.*`` modules, so co-locating these helpers
keeps the orchestrator self-contained.

The module also carries its own private inlines of ``maybe_await`` and
the user-payload normalizer for the same reason, so the orchestrator
does not depend on another shared module while being inlined.
"""

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
