"""Open Terminal UI event emission shared by tool wrappers.

Several plugins (parallel_tools, sub_agent, multi_model_council, magi,
llm_review) translate the result of an Open Terminal builtin tool into a
``terminal:*`` UI event so the frontend can refresh the right viewer pane.
The translation is identical across plugins; this module owns the single
copy. Plugins decide *whether* to emit (e.g. via an ``ENABLE_TERMINAL_TOOLS``
Valve) before calling.
"""

import json
import logging
from typing import Any, Callable, Optional

_terminal_events_log = logging.getLogger("owui_ext.shared.terminal_events")


async def emit_terminal_tool_event(
    *,
    tool_function_name: str,
    tool_function_params: dict,
    tool_result: Any,
    event_emitter: Optional[Callable],
) -> None:
    """Emit ``terminal:*`` UI events for Open Terminal tool results.

    The function recognises only the names listed in
    ``TERMINAL_EVENT_TOOLS`` (display_file / write_file /
    replace_file_content / run_command) -- unknown names hit the
    final ``else`` and silently return.
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
        _terminal_events_log.warning(
            f"Error emitting terminal event for {tool_function_name}: {exc}"
        )
