"""
title: LLM Review
description: Run a collaborative writing process where multiple persona agents each produce a distinct, original draft — drafting independently, reviewing peers, and revising their own draft across multiple rounds. Returns one divergent draft per persona rather than a merged output. Independent implementation inspired by arXiv:2601.08003 "LLM Review".
author: https://github.com/skyzi000
version: 0.5.1
license: MIT
required_open_webui_version: 0.7.0
"""

# === GENERATED FILE - DO NOT EDIT ===
# Source: src/owui_ext/tools/llm_review.py
# Regenerate with: uv run python scripts/build_release.py --target llm_review
# Future imports: (none)
# See release.toml for target definitions.

import asyncio
import ast
import json
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from typing import Any, Callable, Literal, Optional, Type
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

# --- inlined from src/owui_ext/shared/terminal_events.py (owui_ext.shared.terminal_events) ---
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

# --- inlined from src/owui_ext/shared/tool_execution.py (owui_ext.shared.tool_execution) ---
import json
import uuid
from typing import Any, Optional
from fastapi import Request
_core_process_tool_result = None


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

try:
    import markdown as _markdown_mod
except ImportError:
    _markdown_mod = None

log = logging.getLogger(__name__)


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


# Inlined inside every iframe IIFE to reflect Open WebUI's active theme
# on the iframe body via ``.dark`` so our class-based CSS picks it up.
#
# BIG CAVEAT: OWUI renders our iframe with sandbox=allow-scripts WITHOUT
# allow-same-origin by default (see FullHeightIframe.svelte:16 /
# ResponseMessage.svelte:707 — gated behind a user setting). Without
# same-origin, ``parent.document`` access throws SecurityError and we
# fall back to ``prefers-color-scheme``, which follows the OS — WRONG
# whenever the user's OS theme and their OWUI theme disagree.
#
# Users hitting that mismatch can set UserValves.RICH_PROGRESS_THEME to
# 'light' or 'dark' to force the correct palette.


def _theme_detect_js(force_theme: str) -> str:
    """Return the theme-setup JS block to embed in each iframe.

    ``force_theme`` in {"auto", "light", "dark"}:
      - "light"/"dark": lock the iframe to the named theme (no detection).
      - "auto": try ``parent.document.documentElement.classList`` and
        observe it; fall back to ``prefers-color-scheme`` if the sandbox
        blocks parent access.
    """
    ft = (force_theme or "auto").strip().lower()
    if ft == "light":
        return "(function(){document.body.classList.remove('dark');})();"
    if ft == "dark":
        return "(function(){document.body.classList.add('dark');})();"
    # auto
    return (
        "(function(){"
        "const apply=(d)=>{document.body.classList.toggle('dark',!!d);};"
        "let observed=false;"
        "try{"
        "const ph=parent.document.documentElement;"
        "apply(ph.classList.contains('dark'));"
        "new MutationObserver(()=>apply(ph.classList.contains('dark')))"
        ".observe(ph,{attributes:true,attributeFilter:['class']});"
        "observed=true;"
        "}catch(_){}"
        "if(!observed){"
        "const mq=matchMedia('(prefers-color-scheme: dark)');"
        "apply(mq.matches);"
        "mq.addEventListener('change',e=>apply(e.matches));"
        "}"
        "})();"
    )


def _safe_script_json(obj: Any) -> str:
    """JSON-serialize for embedding inside a <script> block or HTML attribute.

    Escapes every HTML-significant character (<, >, &, U+2028, U+2029) to its
    \\uXXXX form so the payload cannot break out of the surrounding <script>
    element, start an HTML-style comment (<!-- / -->), or terminate a JS
    string literal via a U+2028/U+2029 line terminator.
    """
    return (
        json.dumps(obj, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _is_rich_shell_embed(embed: Any, marker: str) -> bool:
    """True when ``embed`` is the rich progress shell identified by ``marker``.

    The shell is always a string containing the run-scoped HTML comment
    marker; anything else (other tool-produced embeds, dict-shaped
    payloads, etc.) is treated as non-shell and eligible for preservation.
    """
    return isinstance(embed, str) and marker in embed


def _embed_dedupe_key(embed: Any) -> str:
    """Stable string key for deduping ``embed`` across wrapper observations.

    Strings key off their content. Dict/list-shaped embeds go through
    ``json.dumps(sort_keys=True)``; if that fails we fall back to
    ``repr`` so dedup never blows up on an exotic payload.
    """
    if isinstance(embed, str):
        return "s:" + embed
    try:
        return "j:" + json.dumps(embed, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return "r:" + repr(embed)


def _render_shell_html(
    tag: str,
    topic: str,
    agents: list[dict],
    force_theme: str = "auto",
) -> str:
    """Initial hidden iframe shell that receives postMessage state updates."""
    initial_agents = [
        {
            **a,
            "activities": a.get("activities") or [],
            "reviews": a.get("reviews") or [],
        }
        for a in agents
    ]
    initial_state = {
        "phase": "Starting",
        "value": 0,
        "detail": None,
        "agents": initial_agents,
        "topic": topic,
    }
    shell_marker = f"<!--llm-review-shell:{tag}-->"
    # Design trade-off (see backend socket/main.py: embeds append to DB):
    # Open WebUI cannot REPLACE or CLEAR a message's embeds from a Tool; the
    # socket handler always prepends new embeds to the stored list. After
    # finalize, DB therefore ends up as [final_html, shell_html]. To avoid
    # visual clutter on reload, this shell stays height:0 unless a postMessage
    # with the matching tag arrives — which only happens during the live run.
    # On reload the tag is stale, no message arrives, and the shell stays
    # fully hidden. The first script below posts height:0 immediately so the
    # invisible state latches before the iframe default height can flash.
    return (
        f"{shell_marker}"
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<script>parent.postMessage({type:'iframe:height',height:0},'*');</script>"
        "<style>"
        "*{box-sizing:border-box}"
        "html,body{margin:0;padding:0}"
        "body{display:none;font-family:system-ui,-apple-system,sans-serif;color:#1f2328;background:transparent;font-size:13px;line-height:1.4}"
        "body.ready{display:block}"
        "body.dark{color:#e6edf3}"
        ".card{padding:14px 16px;border:1px solid rgba(127,127,127,0.25);border-radius:10px;background:rgba(127,127,127,0.04)}"
        ".topic{font-size:12px;opacity:0.7;margin-bottom:4px;overflow-wrap:anywhere;word-break:break-word}"
        ".phase{font-weight:600;font-size:14px;margin-bottom:8px;display:flex;justify-content:space-between;gap:8px}"
        ".phase .pct{font-variant-numeric:tabular-nums;opacity:0.75}"
        ".bar{position:relative;height:8px;background:rgba(127,127,127,0.2);border-radius:4px;overflow:hidden;margin-bottom:10px}"
        # Filled bar: width transitions smoothly between sub-step ticks,
        # and the gradient itself flows leftward via background-position
        # animation so the bar feels alive even between updates. Wider
        # background-size (300%) keeps the flow visible at small widths
        # where the bar is still narrow.
        ".bar>i{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#8b5cf6,#6366f1,#3b82f6);background-size:300% 100%;border-radius:4px;transition:width 0.3s ease;animation:barFlow 3s linear infinite}"
        "@keyframes barFlow{0%{background-position:300% 0}100%{background-position:0% 0}}"
        # Match Open WebUI's multi-model response layout (see core
        # MultiResponseMessages.svelte): flex row with horizontal snap scroll
        # and a per-card min-width that switches to full-width on narrow
        # viewports. No stacking fallback — on narrow screens each card
        # becomes 100% width and the user swipes horizontally, exactly like
        # the standard multi-model chat.
        ".agents{display:flex;overflow-x:auto;scroll-snap-type:x mandatory;gap:10px;margin-bottom:6px;align-items:stretch;scrollbar-width:thin;padding-bottom:2px}"
        ".agents::-webkit-scrollbar{height:6px}"
        ".agents::-webkit-scrollbar-thumb{background:rgba(127,127,127,0.35);border-radius:3px}"
        ".ag{flex:1 1 320px;min-width:320px;scroll-snap-align:center;padding:9px 11px;border-radius:8px;background:rgba(127,127,127,0.06);border:1px solid rgba(127,127,127,0.22);display:flex;flex-direction:column}"
        "@media (max-width:720px){.ag{flex:0 0 100%;min-width:100%}}"
        ".ag.active{border-color:#3b82f6;background:rgba(59,130,246,0.07)}"
        ".ag.done{border-color:#10b981;background:rgba(16,185,129,0.07)}"
        ".ag.error{border-color:#ef4444;background:rgba(239,68,68,0.07)}"
        ".ag .hdr{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:4px}"
        ".ag .name{font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".ag .state{font-size:11px;padding:2px 7px;border-radius:999px;background:rgba(127,127,127,0.15);white-space:nowrap}"
        ".ag .model{font-size:10.5px;opacity:0.6;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,monospace}"
        ".ag.active .state{background:rgba(59,130,246,0.2);color:#3b82f6}"
        ".ag.done .state{background:rgba(16,185,129,0.2);color:#10b981}"
        ".ag.error .state{background:rgba(239,68,68,0.2);color:#ef4444}"
        ".log{display:flex;flex-direction:column;gap:3px;font-size:11.5px;margin-top:4px}"
        ".log .li{display:flex;gap:6px;opacity:0.88;overflow-wrap:anywhere;word-break:break-word}"
        ".log .li .k{flex:0 0 auto;opacity:0.6;font-variant-numeric:tabular-nums;min-width:1em}"
        ".log .li .t{flex:1 1 auto;white-space:pre-wrap;overflow-wrap:anywhere}"
        ".log .li.thinking{opacity:0.55;font-style:italic}"
        ".log .li.speech .t{color:inherit}"
        ".log .li.info .t{opacity:0.7;font-size:11px}"
        # ---- Tool call card (OWUI ToolCallDisplay-alike) ----
        ".log .toolcall{margin:1px 0;font-size:11.5px}"
        ".log .toolcall>summary{cursor:pointer;list-style:none;padding:4px 8px;"
        "border-radius:6px;background:rgba(127,127,127,0.09);"
        "display:flex;align-items:center;gap:7px;user-select:none}"
        ".log .toolcall>summary::-webkit-details-marker{display:none}"
        ".log .toolcall>summary:hover{background:rgba(127,127,127,0.16)}"
        ".log .toolcall .tc-icon{flex:0 0 auto;display:inline-block;width:14px;"
        "text-align:center;font-size:12px}"
        ".log .toolcall .tc-label{flex:1 1 auto;overflow:hidden;"
        "text-overflow:ellipsis;white-space:nowrap}"
        ".log .toolcall .tc-label b{font-weight:600}"
        # Suppress the generic .ag details>summary::before chevron that
        # would otherwise inherit from the parent agent card rules and
        # render a second icon on the left of the summary.
        ".log .toolcall>summary::before{content:none}"
        # Single chevron, right-aligned (OWUI ToolCallDisplay layout).
        ".log .toolcall>summary::after{content:'\u25b8';margin-left:auto;"
        "flex:0 0 auto;opacity:0.5;transition:transform 0.15s;display:inline-block}"
        ".log .toolcall[open]>summary::after{transform:rotate(90deg)}"
        ".log .toolcall[data-status=\"running\"] .tc-icon{color:#6366f1}"
        "body.dark .log .toolcall[data-status=\"running\"] .tc-icon{color:#a5b4fc}"
        ".log .toolcall[data-status=\"done\"] .tc-icon{color:#10b981}"
        "body.dark .log .toolcall[data-status=\"done\"] .tc-icon{color:#6ee7b7}"
        ".log .toolcall .tc-spin{animation:tcspin 1.2s linear infinite;display:inline-block}"
        "@keyframes tcspin{to{transform:rotate(360deg)}}"
        ".log .toolcall[data-status=\"running\"]>summary .tc-label{color:#6366f1}"
        "body.dark .log .toolcall[data-status=\"running\"]>summary .tc-label{color:#a5b4fc}"
        ".log .toolcall[data-status=\"running\"]>summary{background:linear-gradient(90deg,rgba(99,102,241,0.08),rgba(99,102,241,0.04))}"
        ".log .toolcall .tc-body{padding:6px 10px 4px 8px}"
        ".log .toolcall .tc-section{margin-top:5px}"
        ".log .toolcall .tc-section:first-child{margin-top:3px}"
        ".log .toolcall .tc-sec-h{font-size:10px;font-weight:600;opacity:0.65;"
        "text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px}"
        ".log .toolcall .tc-json{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
        "font-size:10.5px;padding:5px 7px;background:rgba(127,127,127,0.12);"
        "border-radius:4px;max-height:14em;overflow:auto;"
        "white-space:pre-wrap;overflow-wrap:anywhere;margin:0;line-height:1.45}"
        # Plain string result — match OWUI's bare <pre> output (no bg,
        # slightly lighter text, natural break).
        ".log .toolcall .tc-pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
        "font-size:10.5px;padding:2px 2px;max-height:14em;overflow:auto;"
        "white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;"
        "margin:0;line-height:1.5;opacity:0.85}"
        # Input key-value list (OWUI ToolCallDisplay's parsedArgs layout).
        ".log .toolcall .tc-kvlist{padding:0 4px;display:flex;"
        "flex-direction:column;gap:2px}"
        ".log .toolcall .tc-kv{display:flex;gap:8px;font-size:11px;"
        "padding:1px 0;align-items:baseline}"
        ".log .toolcall .tc-k{font-weight:600;opacity:0.65;flex:0 0 auto;"
        "max-width:40%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".log .toolcall .tc-v{flex:1 1 auto;opacity:0.92;"
        "overflow-wrap:anywhere;word-break:break-word;font-family:ui-monospace,"
        "SFMono-Regular,Menlo,monospace}"
        ".ag details{margin-top:6px}"
        ".ag details>summary{cursor:pointer;list-style:none;padding:4px 6px;border-radius:6px;font-size:11.5px;font-weight:500;user-select:none;display:flex;align-items:center;gap:6px;background:rgba(127,127,127,0.07)}"
        ".ag details>summary:hover{background:rgba(127,127,127,0.14)}"
        ".ag details>summary::-webkit-details-marker{display:none}"
        ".ag details>summary::before{content:'\u25b8';transition:transform 0.15s;display:inline-block;opacity:0.6}"
        ".ag details[open]>summary::before{transform:rotate(90deg)}"
        ".ag details>summary .count{margin-left:auto;opacity:0.6;font-weight:400}"
        ".ag .body{padding:8px 10px 4px;font-size:12.5px;line-height:1.5}"
        ".ag .draft .body{overflow-wrap:anywhere;word-break:break-word;max-height:24em;overflow-y:auto;background:rgba(255,255,255,0.25);border-radius:6px;margin-top:4px}"
        ".ag .draft .body.plain{white-space:pre-wrap}"
        ".ag .draft .body.md h1,.ag .draft .body.md h2,.ag .draft .body.md h3,.ag .draft .body.md h4{margin:8px 0 4px;font-weight:600}"
        ".ag .draft .body.md h1{font-size:15px}"
        ".ag .draft .body.md h2{font-size:14px}"
        ".ag .draft .body.md h3,.ag .draft .body.md h4{font-size:13px}"
        ".ag .draft .body.md p{margin:4px 0;overflow-wrap:anywhere;word-break:break-word}"
        ".ag .draft .body.md ul,.ag .draft .body.md ol{margin:4px 0;padding-left:18px}"
        ".ag .draft .body.md li{margin:1px 0;overflow-wrap:anywhere;word-break:break-word}"
        ".ag .draft .body.md code{padding:1px 4px;border-radius:3px;background:rgba(127,127,127,0.18);font-size:11.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere;word-break:break-all}"
        ".ag .draft .body.md pre{padding:6px 8px;border-radius:5px;background:rgba(127,127,127,0.15);overflow-x:auto;font-size:11.5px;margin:4px 0}"
        ".ag .draft .body.md pre code{background:transparent;padding:0}"
        ".ag .draft .body.md blockquote{margin:4px 0;padding:2px 8px;border-left:3px solid rgba(127,127,127,0.4);opacity:0.85}"
        ".ag .draft .body.md a{color:#2563eb;text-decoration:underline;text-underline-offset:2px}"
        "body.dark .ag .draft .body.md a{color:#60a5fa}"
        ".ag .draft .body.md table{border-collapse:collapse;margin:4px 0;font-size:11.5px}"
        ".ag .draft .body.md th,.ag .draft .body.md td{padding:3px 6px;border:1px solid rgba(127,127,127,0.3)}"
        ".ag .draft .body.md .md-fallback,.ag .draft .body.plain .md-fallback{white-space:pre-wrap;font-family:inherit;margin:0}"
        "body.dark .ag .draft .body{background:rgba(255,255,255,0.03)}"
        ".ag .approach{font-style:italic;font-size:11.5px;opacity:0.72;margin-bottom:6px;overflow-wrap:anywhere;word-break:break-word}"
        ".ag .meta-list{font-size:11.5px;margin-top:6px;padding:0;list-style:none}"
        ".ag .meta-list li{padding:1px 0 1px 14px;position:relative;overflow-wrap:anywhere;word-break:break-word}"
        ".ag .meta-list li::before{content:'\u2022';position:absolute;left:2px;opacity:0.5}"
        ".ag .meta-list.changes li::before{content:'\u270e';color:#10b981}"
        ".ag .meta-list.declined li::before{content:'\u2715';color:#ef4444}"
        ".ag .review{padding:8px 10px;margin-top:6px;border-radius:6px;background:rgba(99,102,241,0.07);border-left:3px solid #6366f1}"
        ".ag .review .rv-head{font-weight:600;font-size:12px;margin-bottom:4px}"
        ".ag .review .rv-feedback{font-size:12.5px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}"
        ".ag .review .rv-sect{margin-top:5px;font-size:11.5px}"
        ".ag .review .rv-sect .label{font-weight:600;opacity:0.8;margin-right:4px}"
        ".ag .review .rv-sect.strengths .label{color:#10b981}"
        ".ag .review .rv-sect.improvements .label{color:#f59e0b}"
        ".ag .review .rv-sect ul{margin:2px 0 0;padding-left:16px}"
        ".ag .review .rv-sect li{margin:1px 0;overflow-wrap:anywhere;word-break:break-word}"
        ".detail{font-size:11px;opacity:0.6;margin-top:6px;min-height:1em}"
        "</style></head><body>"
        # Apply the theme BEFORE rendering — either honour the locked
        # force_theme ('light'/'dark'), or try to detect OWUI's theme in
        # 'auto'. Runs before the body becomes visible (body stays
        # display:none until .ready) so there's no flash of wrong theme.
        f"<script>{_theme_detect_js(force_theme)}</script>"
        "<div id=\"root\"></div>"
        "<script>(function(){"
        f"const TAG={_safe_script_json(tag)};"
        f"let state={_safe_script_json(initial_state)};"
        # Track the bar's last rendered width so each rerender can mount
        # the <i> at the previous width and transition to the new one.
        # Without this, innerHTML rebuilds the element with width:${v}%
        # baked in and CSS transitions never fire (transitions need an
        # existing property to CHANGE, not a fresh element).
        "let lastBarV=0;"
        "const root=document.getElementById('root');"
        # Full HTML escape including " and ' so the same function is safe
        # in both text content AND attribute-value contexts. Model IDs,
        # tool names, etc. can flow into data-oid/data-scroll-id attrs
        # that are bounded by double quotes — a stray " would otherwise
        # let a hostile value escape the attribute and inject new ones.
        "const esc=s=>String(s==null?'':s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));"
        "const KIND_GLYPH={thinking:'\u00b7',speech:'\u201d',info:'\u2139'};"
        # Shared: try to parse JSON, return [parsed, ok]. Anything that's
        # not strict JSON (e.g. Python dict repr) falls through to raw text.
        "const tryJson=(s)=>{try{return [JSON.parse(s),true];}catch(_){return [null,false];}};"
        # Render args — prefer OWUI-native key-value layout when the args
        # are a plain JSON object; otherwise fall back to a JSON code
        # block. Matches ToolCallDisplay's Input section.
        "const renderArgs=(args)=>{"
        "if(!args)return '';"
        "const [obj,ok]=tryJson(args);"
        "if(ok&&obj&&typeof obj==='object'&&!Array.isArray(obj)){"
        "const rows=Object.entries(obj).map(([k,v])=>{"
        "const vs=(v!==null&&typeof v==='object')?JSON.stringify(v):String(v);"
        "return `<div class=\"tc-kv\"><span class=\"tc-k\">${esc(k)}</span>"
        "<span class=\"tc-v\">${esc(vs)}</span></div>`;"
        "}).join('');"
        "return `<div class=\"tc-section\"><div class=\"tc-sec-h\">Input</div>"
        "<div class=\"tc-kvlist\">${rows}</div></div>`;"
        "}"
        "return `<div class=\"tc-section\"><div class=\"tc-sec-h\">Input</div>"
        "<pre class=\"tc-json\">${esc(args)}</pre></div>`;"
        "};"
        # Render result — pretty-print structured JSON, otherwise show as
        # plain monospace text (no background, matching OWUI core).
        "const renderResult=(result)=>{"
        "if(!result)return '';"
        "const [obj,ok]=tryJson(result);"
        "if(ok&&obj!==null&&typeof obj==='object'){"
        "const pretty=JSON.stringify(obj,null,2);"
        "return `<div class=\"tc-section\"><div class=\"tc-sec-h\">Output</div>"
        "<pre class=\"tc-json\">${esc(pretty)}</pre></div>`;"
        "}"
        "return `<div class=\"tc-section\"><div class=\"tc-sec-h\">Output</div>"
        "<pre class=\"tc-pre\">${esc(String(result))}</pre></div>`;"
        "};"
        # renderToolCall: Open WebUI-style collapsible tool invocation.
        # Mirrors core ToolCallDisplay.svelte — running shows a spinning
        # indicator + \"Executing <name>...\", done shows a checkmark +
        # \"View Result from <name>\". Args/Result appear only when
        # populated. data-oid keyed by tool-call id so the user's expand
        # state survives rerenders (openSet/seen sets below).
        "const renderToolCall=(x)=>{"
        "const st=x.status||'running';"
        "const icon=st==='running'"
        "?'<span class=\"tc-icon tc-spin\">\u25d0</span>'"
        ":'<span class=\"tc-icon tc-check\">\u2713</span>';"
        "const label=st==='running'"
        "?`Executing <b>${esc(x.name||'')}</b>\u2026`"
        ":`View Result from <b>${esc(x.name||'')}</b>`;"
        "const argsBlk=renderArgs(x.args);"
        "const resultBlk=(st==='done')?renderResult(x.result):'';"
        "const oid=`tc-${esc(x.id||x.name||'x')}`;"
        # No explicit chevron element in HTML — the CSS ::after rule for
        # .log .toolcall > summary draws the single right-aligned chevron.
        # We also suppress the generic .ag details > summary::before rule
        # for toolcalls so only one indicator is ever rendered.
        "return `<details class=\"toolcall\" data-oid=\"${oid}\" data-status=\"${st}\">"
        "<summary>${icon}<span class=\"tc-label\">${label}</span></summary>"
        "<div class=\"tc-body\">${argsBlk}${resultBlk}</div></details>`;"
        "};"
        # seen tracks details ids that appeared at least once so we only
        # auto-open them on first appearance. openSet remembers user choices.
        "const seen=new Set();const openSet=new Set();"
        "const renderDraft=(a)=>{"
        "if(!a.draft||!a.draft.text)return '';"
        "const d=a.draft;"
        "const approach=d.approach?`<div class=\"approach\">${esc(d.approach)}</div>`:'';"
        "const changes=(d.changes_made&&d.changes_made.length)"
        "?`<ul class=\"meta-list changes\">${d.changes_made.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''; "
        "const declined=(d.feedback_declined&&d.feedback_declined.length)"
        "?`<ul class=\"meta-list declined\">${d.feedback_declined.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''; "
        "const did=`draft-${esc(a.id)}`;"
        # d.html is already sanitized server-side via _MarkdownSanitizer, so
        # inserting it as innerHTML is safe. Fall back to escaped raw text
        # (pre-wrap class) when markdown rendering was disabled or empty.
        "const bodyInner=d.html?d.html:`<pre class=\"md-fallback\">${esc(d.text)}</pre>`;"
        "const bodyCls=d.html?'body md':'body plain';"
        # data-scroll-id lets saveScroll/restoreScroll keep the user's
        # draft-body vertical scroll position across rerenders.
        "return `<details class=\"draft\" data-oid=\"${did}\" data-default-open=\"1\">"
        "<summary>Draft <span class=\"count\">${esc(d.label||'')}</span></summary>"
        "<div class=\"${bodyCls}\" data-scroll-id=\"draft-${esc(a.id)}\">${approach}${bodyInner}</div>"
        "${changes?`<div class=\"body\"><div class=\"approach\">Changes made</div>${changes}</div>`:''}"
        "${declined?`<div class=\"body\"><div class=\"approach\">Feedback declined</div>${declined}</div>`:''}"
        "</details>`;"
        "};"
        "const renderReviews=(a)=>{"
        "const rs=a.reviews||[];"
        "if(!rs.length)return '';"
        "const rid=`reviews-${esc(a.id)}`;"
        "const items=rs.map(r=>{"
        "const str=(r.strengths&&r.strengths.length)"
        "?`<div class=\"rv-sect strengths\"><span class=\"label\">Strengths</span><ul>${r.strengths.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div>`:''; "
        "const imp=(r.improvements&&r.improvements.length)"
        "?`<div class=\"rv-sect improvements\"><span class=\"label\">Improvements</span><ul>${r.improvements.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div>`:''; "
        "return `<div class=\"review\">"
        "<div class=\"rv-head\">\u2190 ${esc(r.reviewer||'reviewer')}</div>"
        "<div class=\"rv-feedback\">${esc(r.key_feedback||'')}</div>"
        "${str}${imp}</div>`;"
        "}).join('');"
        "return `<details class=\"reviews\" data-oid=\"${rid}\">"
        "<summary>Reviews received <span class=\"count\">${rs.length}</span></summary>"
        "<div class=\"body\">${items}</div></details>`;"
        "};"
        # Measure the *content* height (body.scrollHeight) rather than the
        # document/html scrollHeight — the latter never shrinks below the
        # current iframe viewport height once the parent has stretched it,
        # which is why collapsed <details> used to leave a giant gap. rAF
        # ensures we read the measurement AFTER the browser has re-laid
        # out, not mid-toggle.
        "const postHeight=()=>{requestAnimationFrame(()=>{"
        "const b=document.body;"
        "const h=b?b.scrollHeight:document.documentElement.scrollHeight;"
        "parent.postMessage({type:'iframe:height',height:h},'*');"
        "});};"
        # Save scroll positions of every scrollable region before we
        # rebuild innerHTML, then reapply them afterwards. Without this,
        # every push() reset the user's horizontal swipe position in the
        # .agents row and the vertical position inside each draft-body.
        "const saveScroll=()=>{"
        "const saved={};"
        "const ag=root.querySelector('.agents');"
        "if(ag)saved.__ag=ag.scrollLeft;"
        "root.querySelectorAll('[data-scroll-id]').forEach(el=>{"
        "saved[el.dataset.scrollId]={t:el.scrollTop,l:el.scrollLeft};"
        "});"
        "return saved;"
        "};"
        "const restoreScroll=(saved)=>{"
        "const ag=root.querySelector('.agents');"
        "if(ag&&saved.__ag!=null)ag.scrollLeft=saved.__ag;"
        "root.querySelectorAll('[data-scroll-id]').forEach(el=>{"
        "const p=saved[el.dataset.scrollId];"
        "if(p){el.scrollTop=p.t;el.scrollLeft=p.l;}"
        "});"
        "};"
        "const render=()=>{"
        # Save currently-open details so the user's toggles survive a rerender.
        "root.querySelectorAll('details[data-oid]').forEach(d=>{"
        "if(d.open)openSet.add(d.dataset.oid);else openSet.delete(d.dataset.oid);"
        "});"
        "const savedScroll=saveScroll();"
        "const v=Math.max(0,Math.min(100,Number(state.value)||0));"
        # Exact-match state classifier (a previous partial-match regex
        # collapsed "reviewed 1/2" into done-green and misclassified
        # "waiting" as active-blue). Done = fully finished states and
        # "reviewed N/N" with N==M; active = currently-running verbs and
        # partial "reviewed N/M" (N<M); everything else (idle, waiting,
        # blank) stays neutral grey.
        "const statusClass=(raw)=>{"
        "const s=String(raw||'').trim().toLowerCase();"
        # Check 'failed' first so review states like "reviewed 1/2 (1 failed)"
        # classify as error even though they contain the 'reviewed' token.
        "if(s.indexOf('failed')>=0)return 'error';"
        "if(s==='done'||s==='ready'||s==='drafted'||s==='revised')return 'done';"
        "const rm=/^reviewed\\s+(\\d+)\\s*\\/\\s*(\\d+)$/.exec(s);"
        "if(rm){return rm[1]===rm[2]?'done':'active';}"
        "if(s==='composing'||s==='revising'||s==='thinking')return 'active';"
        "if(/^reviewing\\b/.test(s))return 'active';"
        "return '';"
        "};"
        "const cards=(state.agents||[]).map(a=>{"
        "const s=String(a.state||'idle');"
        "const cls=statusClass(s);"
        "const acts=(a.activities||[]);"
        "const renderAct=(x)=>{"
        "if(x.kind==='tool_call')return renderToolCall(x);"
        "const glyph=KIND_GLYPH[x.kind]||'\u2022';"
        "const txt=x.kind==='thinking'?'thinking\u2026':(x.text||'');"
        "return `<div class=\"li ${esc(x.kind||'')}\">"
        "<span class=\"k\">${esc(glyph)}</span>"
        "<span class=\"t\">${esc(txt)}</span></div>`;"
        "};"
        "const logHtml=acts.length"
        "?acts.map(renderAct).join('')"
        ":'<div class=\"li thinking\"><span class=\"k\">\u00b7</span><span class=\"t\">waiting\u2026</span></div>';"
        "return `<div class=\"ag ${cls}\">"
        "<div class=\"hdr\"><span class=\"name\">${esc(a.persona||a.id)}</span>"
        "<span class=\"state\">${esc(s)}</span></div>"
        # Show the underlying model id as a small subtitle so the user can
        # tell which model sits behind each persona — essential when the
        # same model is used in multiple slots (e.g. gpt-5 ×3).
        "${a.model?`<div class=\"model\">${esc(a.model)}</div>`:''}"
        "<div class=\"log\">${logHtml}</div>"
        "${renderDraft(a)}${renderReviews(a)}</div>`;"
        "}).join('');"
        "root.innerHTML=`<div class=\"card\">`+"
        "(state.topic?`<div class=\"topic\">${esc(state.topic)}</div>`:'')+"
        "`<div class=\"phase\"><span>${esc(state.phase||'')}</span><span class=\"pct\">${v.toFixed(0)}%</span></div>"
        "<div class=\"bar\"><i style=\"width:${lastBarV}%\"></i></div>"
        "<div class=\"agents\">${cards}</div>"
        "<div class=\"detail\">${esc(state.detail||'')}</div></div>`;"
        # Restore open state: if user had it open, keep open; on first appearance
        # apply data-default-open="1"; otherwise closed.
        "root.querySelectorAll('details[data-oid]').forEach(d=>{"
        "const oid=d.dataset.oid;"
        "if(openSet.has(oid)){d.open=true;}"
        "else if(!seen.has(oid)){seen.add(oid);if(d.dataset.defaultOpen==='1')d.open=true;}"
        "});"
        "restoreScroll(savedScroll);"
        # Trigger the bar's width transition by mounting at lastBarV (set
        # in the innerHTML above) and then updating to v on the next frame
        # — the browser sees a real property change and runs the ease.
        "const barFill=root.querySelector('.bar>i');"
        "if(barFill){requestAnimationFrame(()=>{barFill.style.width=v+'%';});}"
        "lastBarV=v;"
        "document.body.classList.add('ready');"
        "postHeight();"
        "};"
        # Re-post height when user toggles a <details> so the iframe
        # grows AND shrinks. Using body.scrollHeight via postHeight() so a
        # collapsed details really shortens the iframe instead of leaving
        # a giant gap.
        "root.addEventListener('toggle',e=>{"
        "const d=e.target;if(!d||!d.dataset||!d.dataset.oid)return;"
        "if(d.open)openSet.add(d.dataset.oid);else openSet.delete(d.dataset.oid);"
        "postHeight();"
        "},true);"
        # Post height:0 at multiple lifecycle points so a stale shell (no
        # tagged message ever arrives) stays collapsed on reload without a
        # default-height flash.
        "const hide=()=>parent.postMessage({type:'iframe:height',height:0},'*');"
        "hide();"
        "window.addEventListener('DOMContentLoaded',hide);"
        "window.addEventListener('load',hide);"
        "window.addEventListener('message',e=>{"
        "if(!e.data||e.data.tag!==TAG)return;"
        "if(e.data.state)state=Object.assign(state,e.data.state);"
        "render();"
        "});"
        "})();</script></body></html>"
    )


def _html_escape(value: Any) -> str:
    """Escape HTML-significant characters for safe interpolation as text or
    inside double-quoted attributes."""
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _is_safe_http_url(url: str) -> bool:
    """Accept only absolute http(s) URLs. LLM-supplied sources may contain
    ``javascript:``, ``data:``, ``vbscript:`` or other dangerous schemes that
    HTML-attribute escaping can't neutralise, so we whitelist the scheme and
    reject everything else (including protocol-relative ``//foo``, since the
    iframe itself is served from a data URL without meaningful origin)."""
    if not isinstance(url, str):
        return False
    # Reject control characters that browsers may ignore when URL-parsing.
    if any(c < " " for c in url):
        return False
    stripped = url.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    return lower.startswith("http://") or lower.startswith("https://")


# ============================================================================
# Markdown rendering for drafts (server-side render + whitelist sanitizer)
# ============================================================================

_MD_ALLOWED_TAGS = frozenset({
    "p", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li",
    "strong", "b", "em", "i", "u", "s", "del", "ins",
    "blockquote", "pre", "code", "kbd", "samp", "var",
    "a",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td",
    "span", "div",
})
_MD_VOID_TAGS = frozenset({"br", "hr"})
_MD_ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "a": frozenset({"href", "title"}),
    "code": frozenset({"class"}),
    "pre": frozenset({"class"}),
    "span": frozenset({"class"}),
    "th": frozenset({"align", "style"}),
    "td": frozenset({"align", "style"}),
}
# Only permit class values that come from markdown-extra/codehilite, e.g.
# "language-python", "highlight", "codehilite". Reject anything else so
# LLM input can't smuggle weird class names that parent CSS might match.
_MD_CLASS_RE = re.compile(r"^[a-zA-Z][\w\-]{0,40}$")
_MD_STYLE_RE = re.compile(
    r"^\s*text-align\s*:\s*(left|center|right)\s*;?\s*$", re.IGNORECASE
)


_MD_DROP_CONTENT_TAGS = frozenset(
    {"script", "style", "svg", "iframe", "object", "embed", "noscript", "template"}
)


class _MarkdownSanitizer(HTMLParser):
    """Whitelist HTML sanitizer for markdown-rendered draft content.

    Maintains an explicit stack of open whitelisted tags so closing-tag
    boundaries are strictly enforced: self-closing non-void tags get an
    immediate companion close tag, orphan close tags are ignored, and any
    tags still open at end-of-input are auto-closed in reverse order. Drops
    ``<script>``/``<style>``/``<svg>`` etc. plus their entire inner payload
    (text, entities, charrefs) via a separate depth counter. Validates
    ``<a href>`` through :func:`_is_safe_http_url` and rewrites every safe
    ``<a>`` with ``target="_blank" rel="noopener noreferrer"``.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.out: list[str] = []
        # Stack of currently-open whitelisted tags. Ensures every emitted
        # ``<tag>`` has a matching ``</tag>`` emitted in the correct order.
        self._stack: list[str] = []
        # Depth counter for nested drop-content tags. While > 0, every data,
        # entity, and charref handler is a no-op so the tag's inner payload
        # (e.g. <style> CSS or <script> JS) never leaks into the output as
        # readable text that might later be misinterpreted.
        self._skip_depth = 0

    def _build_attrs(
        self, tag: str, attrs: list[tuple[str, Optional[str]]]
    ) -> tuple[str, bool]:
        allowed = _MD_ALLOWED_ATTRS.get(tag, frozenset())
        safe_pairs: list[tuple[str, str]] = []
        href_present = False
        for name, value in attrs:
            lname = (name or "").lower()
            if lname not in allowed:
                continue
            v = value or ""
            if tag == "a" and lname == "href":
                if not _is_safe_http_url(v):
                    continue
                href_present = True
                safe_pairs.append((lname, v))
            elif lname == "class":
                tokens = [t for t in v.split() if _MD_CLASS_RE.match(t)]
                if tokens:
                    safe_pairs.append((lname, " ".join(tokens)))
            elif lname == "style":
                if _MD_STYLE_RE.match(v):
                    safe_pairs.append((lname, v.strip()))
            elif lname == "align" and v.lower() in {"left", "center", "right"}:
                safe_pairs.append((lname, v.lower()))
            elif lname == "title":
                safe_pairs.append((lname, v))
        attrs_str = "".join(f' {n}="{_html_escape(v)}"' for n, v in safe_pairs)
        if tag == "a" and href_present:
            attrs_str += ' target="_blank" rel="noopener noreferrer"'
        return attrs_str, href_present

    def _process_start(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
        self_close: bool,
    ) -> None:
        # Drop-content tags suppress their children entirely. Self-closing
        # variants don't bump the skip depth since they have no children.
        if tag in _MD_DROP_CONTENT_TAGS:
            if not self_close:
                self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if tag not in _MD_ALLOWED_TAGS:
            return
        attrs_str, _ = self._build_attrs(tag, attrs)
        if tag in _MD_VOID_TAGS:
            # Void tags like <br>/<hr> never carry children or a close tag.
            self.out.append(f"<{tag}{attrs_str}>")
            return
        if self_close:
            # Non-void self-closing (<a/>, <div/> etc.) — emit a balanced
            # empty element so the open tag can never swallow arbitrary
            # following content as its children.
            self.out.append(f"<{tag}{attrs_str}></{tag}>")
            return
        self.out.append(f"<{tag}{attrs_str}>")
        self._stack.append(tag)

    def handle_starttag(self, tag, attrs):
        self._process_start(tag.lower(), attrs, self_close=False)

    def handle_startendtag(self, tag, attrs):
        self._process_start(tag.lower(), attrs, self_close=True)

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in _MD_DROP_CONTENT_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if t not in _MD_ALLOWED_TAGS or t in _MD_VOID_TAGS:
            return
        # Only close if the tag is actually open. If there are intermediate
        # unclosed tags above it on the stack, close them first to keep the
        # output balanced (mimics the browser's implicit-close behaviour
        # without leaving the stack ever skipping a level).
        if t not in self._stack:
            return
        while self._stack:
            top = self._stack.pop()
            self.out.append(f"</{top}>")
            if top == t:
                break

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        # Markdown with convert_charrefs=False delivers raw text that
        # markdown itself has NOT HTML-escaped (e.g. legitimate "<" or "&"
        # that survived because they appeared inside code). Escape on the
        # way out so nothing can reintroduce markup.
        self.out.append(_html_escape(data))

    def handle_entityref(self, name):
        if self._skip_depth > 0:
            return
        self.out.append(f"&{name};")

    def handle_charref(self, name):
        if self._skip_depth > 0:
            return
        self.out.append(f"&#{name};")

    def close(self) -> None:
        super().close()
        # Auto-close any tags left open at EOF so the emitted HTML is
        # always well-formed regardless of markdown output quirks.
        while self._stack:
            self.out.append(f"</{self._stack.pop()}>")


def _render_markdown(text: Any) -> str:
    """Convert Markdown ``text`` to sanitized HTML.

    Returns ``""`` for blank input. If the ``markdown`` library isn't
    importable or raises mid-render, falls back to a ``<pre>`` of the
    HTML-escaped original so the draft is always at least readable.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    if _markdown_mod is None:
        return f'<pre class="md-fallback">{_html_escape(text)}</pre>'
    try:
        raw_html = _markdown_mod.markdown(
            text,
            extensions=["extra", "sane_lists", "nl2br"],
            output_format="html",
        )
    except Exception as e:
        log.warning(f"[LLMReview] markdown render failed: {e}")
        return f'<pre class="md-fallback">{_html_escape(text)}</pre>'
    sanitizer = _MarkdownSanitizer()
    try:
        sanitizer.feed(raw_html)
        sanitizer.close()
    except Exception as e:
        log.warning(f"[LLMReview] markdown sanitize failed: {e}")
        return f'<pre class="md-fallback">{_html_escape(text)}</pre>'
    return "".join(sanitizer.out)


def _render_sources_block(sources: Any) -> str:
    if not isinstance(sources, list):
        return ""
    items: list[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title") or "").strip()
        raw_url = str(s.get("url") or "").strip()
        safe_url = raw_url if _is_safe_http_url(raw_url) else ""
        if not (title or safe_url or raw_url):
            continue
        if safe_url:
            href = _html_escape(safe_url)
            label = _html_escape(title or safe_url)
            items.append(
                f'<li><a href="{href}" target="_blank" rel="noopener noreferrer">{label}</a></li>'
            )
        elif title:
            # Title only, or URL with unsafe scheme (still show the title so
            # the reader knows a source exists, just don't make it clickable).
            items.append(f"<li>{_html_escape(title)}</li>")
        elif raw_url:
            # URL-only entry with an unsafe scheme: render as inert text so
            # the user can see what the LLM cited without letting the browser
            # follow a javascript:/data: link.
            items.append(
                f'<li><span class="unsafe-url" title="URL rejected: unsafe scheme">{_html_escape(raw_url)}</span></li>'
            )
    if not items:
        return ""
    return f'<div class="srcs"><div class="srcs-h">Sources</div><ol>{"".join(items)}</ol></div>'


def _render_str_list(items: Any, cls: str, label: str) -> str:
    if not isinstance(items, list):
        return ""
    cleaned = [str(x) for x in items if isinstance(x, str) and x.strip()]
    if not cleaned:
        return ""
    lis = "".join(f"<li>{_html_escape(x)}</li>" for x in cleaned)
    return f'<div class="meta-list-wrap"><div class="meta-list-h">{_html_escape(label)}</div><ul class="meta-list {cls}">{lis}</ul></div>'


def _render_agent_card(agent_id: str, persona: str, body_html: str) -> str:
    return (
        f'<div class="ag">'
        f'<div class="hdr"><span class="name">{_html_escape(persona or agent_id)}</span>'
        f'<span class="mid">{_html_escape(agent_id)}</span></div>'
        f"{body_html}"
        f"</div>"
    )


def _render_final_html(summary: dict) -> str:
    """Persistent final embed summarising a completed review.

    Contains EVERYTHING needed after reload: per-agent final drafts (full
    body + sources) AND the full round history with cross-reviews and
    revisions. Sections are collapsible <details> elements so the page stays
    manageable while preserving full access to every piece of data.
    """
    topic = str(summary.get("topic") or "")
    num_rounds = int(summary.get("num_rounds") or 0)
    elapsed = summary.get("elapsed_seconds")
    agents_meta = summary.get("agents") or []
    # ``model_ids`` in the summary actually carries per-slot **agent_ids**
    # (may include ``#N`` suffixes when the same underlying model is used
    # multiple times). Every dict in ``final_drafts``/``rounds`` is keyed by
    # agent_id; the real underlying model id for display sits alongside as
    # the ``model`` field on each agent entry.
    agent_ids = summary.get("model_ids") or [
        (a.get("id") if isinstance(a, dict) else "") for a in agents_meta
    ]
    final_drafts = summary.get("final_drafts") or {}
    rounds = summary.get("rounds") or []
    render_md = bool(summary.get("render_markdown", True))
    force_theme = str(summary.get("force_theme") or "auto")
    is_cancelled = bool(summary.get("cancelled"))
    header_mark = "\u23F9" if is_cancelled else "\u2713"
    header_text = "LLM Review cancelled" if is_cancelled else "LLM Review complete"
    cancel_cls = " cancelled" if is_cancelled else ""

    def _draft_body_html(draft_text: str) -> str:
        """Render draft text as either markdown-sanitized HTML (md class) or
        pre-formatted plain text (plain class). Both paths are XSS-safe:
        markdown output goes through :class:`_MarkdownSanitizer`, plain path
        goes through :func:`_html_escape`."""
        if render_md:
            rendered = _render_markdown(draft_text)
            if rendered:
                return f'<div class="draft-body md">{rendered}</div>'
        return f'<div class="draft-body plain">{_html_escape(draft_text)}</div>'

    persona_by_agent: dict[str, str] = {}
    model_by_agent: dict[str, str] = {}
    for a in agents_meta:
        if isinstance(a, dict):
            aid_s = str(a.get("id") or a.get("model") or "")
            persona_by_agent[aid_s] = str(a.get("persona") or "")
            model_by_agent[aid_s] = str(a.get("model") or aid_s)
    for aid in agent_ids:
        aid_s = str(aid or "")
        fd = final_drafts.get(aid_s) if isinstance(final_drafts, dict) else None
        if isinstance(fd, dict):
            persona_by_agent.setdefault(aid_s, str(fd.get("persona") or ""))
            model_by_agent.setdefault(aid_s, str(fd.get("model") or aid_s))

    # ---- Header (topic + meta + agent pills) ----
    elapsed_str = (
        f'<span>Elapsed: {float(elapsed):.1f}s</span>'
        if isinstance(elapsed, (int, float))
        else ""
    )
    pills_html = "".join(
        f'<div class="pill"><div class="n">{_html_escape(persona_by_agent.get(str(aid), ""))}</div>'
        f'<div class="m">{_html_escape(model_by_agent.get(str(aid), str(aid)))}</div></div>'
        for aid in agent_ids
    )

    # ---- Final drafts section (one card per agent, side-by-side) ----
    num_rounds_raw = summary.get("num_rounds")
    total_rounds = num_rounds_raw if isinstance(num_rounds_raw, int) else 0
    # (persona, last_successful_round, sub_kind) per incomplete agent.
    # sub_kind distinguishes pre-cancel failures from in-flight cancels
    # within the same run.
    incomplete_entries: list[tuple[str, int, str]] = []
    final_cards = []
    for aid in agent_ids:
        aid_s = str(aid or "")
        fd = final_drafts.get(aid_s) if isinstance(final_drafts, dict) else None
        if not isinstance(fd, dict):
            continue
        draft_text = str(fd.get("draft") or "")
        sources_html = _render_sources_block(fd.get("sources"))
        lsr_raw = fd.get("last_successful_round")
        lsr = lsr_raw if isinstance(lsr_raw, int) else None
        outcome_raw = fd.get("last_phase_outcome")
        outcome = (
            outcome_raw
            if outcome_raw in ("success", "failed", "unknown")
            else "unknown"
        )
        # Legacy payloads (no submitted_rounds key) fall back to a
        # contiguous-chain assumption so existing callers see the
        # original wording instead of garbage.
        submitted_raw = fd.get("submitted_rounds")
        if isinstance(submitted_raw, list) and all(
            isinstance(x, int) for x in submitted_raw
        ):
            submitted_rounds_set = set(submitted_raw)
        elif lsr is not None and lsr >= 0:
            submitted_rounds_set = set(range(0, lsr + 1))
        else:
            submitted_rounds_set = set()
        chain_end_for_audit = lsr if (lsr is not None and lsr >= 0) else -1
        unchained_submitted = sorted(
            r for r in submitted_rounds_set if r > chain_end_for_audit
        )
        has_unchained = bool(unchained_submitted)
        incomplete_badge = ""
        if (
            total_rounds > 0
            and lsr is not None
            and lsr < total_rounds
        ):
            # During cancel, "failed" outcome wins over the run-level
            # flag so a pre-cancel bug stays labelled as a failure.
            if not is_cancelled or outcome == "failed":
                sub_kind = "failed"
            else:
                sub_kind = "cancelled"
            incomplete_entries.append(
                (persona_by_agent.get(aid_s, aid_s), lsr, sub_kind)
            )
            if has_unchained:
                rounds_str = ", ".join(str(r) for r in unchained_submitted)
                noun = "round" if len(unchained_submitted) == 1 else "rounds"
                unchained_suffix = (
                    f" (revise {noun} {rounds_str} submitted but could "
                    "not extend the chain)"
                )
            else:
                unchained_suffix = ""
            if sub_kind == "cancelled":
                if lsr < 0:
                    base_label = "composition cancelled before completion"
                elif lsr == 0:
                    if has_unchained:
                        base_label = (
                            "showing compose draft — round 1 revision "
                            "failed; run cancelled mid-pipeline"
                        )
                    else:
                        base_label = (
                            f"showing compose draft — cancelled before any of the "
                            f"{total_rounds} revision rounds could complete"
                        )
                else:
                    base_label = (
                        f"showing round {lsr} revision — cancelled before "
                        "later rounds"
                    )
            else:
                if lsr < 0:
                    base_label = "composition failed"
                elif lsr == 0:
                    if is_cancelled and has_unchained:
                        base_label = (
                            "showing compose draft — round 1 revision "
                            "failed; run cancelled mid-pipeline"
                        )
                    elif is_cancelled:
                        base_label = (
                            "showing compose draft — round 1 revision "
                            "failed; run cancelled before retry"
                        )
                    elif has_unchained:
                        base_label = (
                            "showing compose draft — round 1 revision failed"
                        )
                    else:
                        base_label = (
                            f"showing compose draft — all {total_rounds} "
                            "revision rounds failed"
                        )
                else:
                    if is_cancelled and has_unchained:
                        base_label = (
                            f"showing round {lsr} revision — round "
                            f"{lsr + 1} failed; run cancelled mid-pipeline"
                        )
                    elif is_cancelled:
                        base_label = (
                            f"showing round {lsr} revision — round "
                            f"{lsr + 1} failed; run cancelled before retry"
                        )
                    elif has_unchained:
                        base_label = (
                            f"showing round {lsr} revision — round "
                            f"{lsr + 1} failed"
                        )
                    else:
                        base_label = (
                            f"showing round {lsr} revision — later rounds failed"
                        )
            badge_label = base_label + unchained_suffix
            incomplete_badge = (
                f'<div class="revision-warning">\u26a0\ufe0f {_html_escape(badge_label)}</div>'
            )
        body = f"{incomplete_badge}{_draft_body_html(draft_text)}{sources_html}"
        final_cards.append(
            _render_agent_card(
                model_by_agent.get(aid_s, aid_s),
                persona_by_agent.get(aid_s, ""),
                body,
            )
        )
    section_banner = ""
    if incomplete_entries:
        has_failed = any(kind == "failed" for _, _, kind in incomplete_entries)
        has_cancelled = any(
            kind == "cancelled" for _, _, kind in incomplete_entries
        )
        details = ", ".join(
            f"{_html_escape(persona)} ({kind}, last successful round: {rn})"
            for persona, rn, kind in incomplete_entries
        )
        if has_failed and has_cancelled:
            banner_prefix = (
                "Revision pipeline incomplete (mixed failure and "
                "cancellation outcomes)"
            )
        elif has_cancelled:
            banner_prefix = (
                "Run cancelled before the revision pipeline finished for "
                "all agents"
            )
        else:
            banner_prefix = (
                "Revision pipeline did not complete for all agents"
            )
        section_banner = (
            f'<div class="revision-warning revision-warning-banner">'
            f"\u26a0\ufe0f {banner_prefix}: {details}."
            f"</div>"
        )
    final_section = (
        f'<section class="final-section">'
        f'<h3>Final drafts</h3>'
        f"{section_banner}"
        f'<div class="agents">{"".join(final_cards)}</div>'
        f"</section>"
        if final_cards
        else ""
    )

    # ---- Round history section ----
    history_blocks: list[str] = []
    for r in rounds:
        if not isinstance(r, dict):
            continue
        rn = r.get("round", 0)
        if r.get("phase") == "initial_composition":
            # Round 0: full drafts payload from finalize summary (always
            # kept for the Rich embed regardless of INCLUDE_FULL_HISTORY).
            drafts_map = r.get("drafts") or {}
            cards: list[str] = []
            for aid in agent_ids:
                aid_s = str(aid or "")
                d = drafts_map.get(aid_s) if isinstance(drafts_map, dict) else None
                if not isinstance(d, dict):
                    continue
                approach = str(d.get("approach") or "").strip()
                draft_text = str(d.get("draft") or "")
                sources_html = _render_sources_block(d.get("sources"))
                approach_html = (
                    f'<div class="approach">{_html_escape(approach)}</div>'
                    if approach
                    else ""
                )
                body = f"{approach_html}{_draft_body_html(draft_text)}{sources_html}"
                cards.append(
                    _render_agent_card(
                        model_by_agent.get(aid_s, aid_s),
                        persona_by_agent.get(aid_s, ""),
                        body,
                    )
                )
            if cards:
                history_blocks.append(
                    f'<details class="round" data-round="0">'
                    f"<summary>Round 0 \u00b7 Initial composition</summary>"
                    f'<div class="agents">{"".join(cards)}</div>'
                    f"</details>"
                )
            continue

        # review/revise round
        reviews = r.get("reviews") or {}
        revisions = r.get("revisions") or {}
        cards = []
        for aid in agent_ids:
            aid_s = str(aid or "")
            persona = persona_by_agent.get(aid_s, "")

            # Per-author reviews (from each peer reviewer)
            review_items = []
            author_reviews = reviews.get(aid_s) if isinstance(reviews, dict) else None
            if isinstance(author_reviews, dict):
                for reviewer_id, rv in author_reviews.items():
                    if not isinstance(rv, dict):
                        continue
                    reviewer_name = str(rv.get("reviewer") or reviewer_id)
                    kf = str(rv.get("key_feedback") or "")
                    strengths_html = _render_str_list(
                        rv.get("strengths"), "strengths", "Strengths"
                    )
                    improvements_html = _render_str_list(
                        rv.get("improvements"), "improvements", "Improvements"
                    )
                    review_items.append(
                        f'<div class="review">'
                        f'<div class="rv-head">\u2190 {_html_escape(reviewer_name)}</div>'
                        f'<div class="rv-feedback">{_html_escape(kf)}</div>'
                        f"{strengths_html}{improvements_html}"
                        f"</div>"
                    )
            reviews_html = (
                f'<div class="reviews-block"><div class="block-h">Reviews received</div>{"".join(review_items)}</div>'
                if review_items
                else ""
            )

            # Revision: changes_made/feedback_declined + full revised
            # draft body (always present in finalize summary, which
            # carries the full-history rounds_data regardless of the
            # INCLUDE_FULL_HISTORY valve that only trims the return JSON).
            rev = revisions.get(aid_s) if isinstance(revisions, dict) else None
            revision_html = ""
            if isinstance(rev, dict):
                changes_html = _render_str_list(
                    rev.get("changes_made"), "changes", "Changes made"
                )
                declined_html = _render_str_list(
                    rev.get("feedback_declined"), "declined", "Feedback declined"
                )
                rev_draft_text = str(rev.get("draft") or "").strip()
                sources_html = _render_sources_block(rev.get("sources"))
                draft_block = ""
                if rev_draft_text:
                    draft_block = (
                        f'<details class="rev-draft" data-round="{int(rn)}-{_html_escape(aid_s)}">'
                        f"<summary>Revised draft</summary>"
                        f"{_draft_body_html(rev_draft_text)}"
                        f"{sources_html}"
                        f"</details>"
                    )
                revision_html = (
                    f'<div class="revision-block"><div class="block-h">Revision</div>'
                    f"{changes_html}{declined_html}{draft_block}"
                    f"</div>"
                )
            cards.append(
                _render_agent_card(
                    model_by_agent.get(aid_s, aid_s),
                    persona,
                    f"{reviews_html}{revision_html}",
                )
            )
        if cards:
            history_blocks.append(
                f'<details class="round" data-round="{int(rn)}">'
                f"<summary>Round {int(rn)} \u00b7 Cross-review &amp; revision</summary>"
                f'<div class="agents">{"".join(cards)}</div>'
                f"</details>"
            )
    history_section = (
        f'<section class="history-section"><h3>Round history</h3>{"".join(history_blocks)}</section>'
        if history_blocks
        else ""
    )

    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
        "*{box-sizing:border-box}"
        "html,body{margin:0;padding:0}"
        "body{font-family:system-ui,-apple-system,sans-serif;color:#1f2328;background:transparent;font-size:13px;line-height:1.45}"
        "body.dark{color:#e6edf3}"
        "a{color:inherit;text-decoration:underline;text-underline-offset:2px}"
        ".wrap{padding:14px 16px;border:1px solid rgba(16,185,129,0.4);border-radius:10px;background:rgba(16,185,129,0.06)}"
        ".wrap.cancelled{border-color:rgba(148,163,184,0.4);background:rgba(148,163,184,0.06)}"
        ".head{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-weight:600;font-size:14px}"
        ".head .mark{color:#10b981;font-size:16px}"
        ".head.cancelled .mark{color:#94a3b8}"
        ".topic{font-size:13px;margin-bottom:8px;overflow-wrap:anywhere;word-break:break-word}"
        ".meta{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;opacity:0.75;margin-bottom:10px}"
        ".pills{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}"
        ".pill{padding:6px 9px;border-radius:8px;background:rgba(127,127,127,0.08);border:1px solid rgba(127,127,127,0.2);font-size:12px;min-width:0}"
        ".pill .n{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".pill .m{opacity:0.7;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        "h3{font-size:13px;margin:12px 0 6px;font-weight:600;opacity:0.9}"
        ".agents{display:flex;overflow-x:auto;scroll-snap-type:x mandatory;gap:10px;align-items:stretch;scrollbar-width:thin;padding-bottom:2px}"
        ".agents::-webkit-scrollbar{height:6px}"
        ".agents::-webkit-scrollbar-thumb{background:rgba(127,127,127,0.35);border-radius:3px}"
        ".ag{flex:1 1 320px;min-width:320px;scroll-snap-align:center;padding:10px 12px;border-radius:8px;background:rgba(127,127,127,0.05);border:1px solid rgba(127,127,127,0.22);display:flex;flex-direction:column;gap:6px}"
        "@media (max-width:720px){.ag{flex:0 0 100%;min-width:100%}}"
        ".ag .hdr{display:flex;justify-content:space-between;align-items:baseline;gap:8px}"
        ".ag .name{font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".ag .mid{font-size:10.5px;opacity:0.6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,monospace}"
        ".approach{font-style:italic;font-size:11.5px;opacity:0.72;overflow-wrap:anywhere;word-break:break-word}"
        ".draft-body{overflow-wrap:anywhere;word-break:break-word;font-size:12.5px;line-height:1.55;background:rgba(255,255,255,0.25);border-radius:6px;padding:8px 10px;max-height:32em;overflow-y:auto}"
        ".draft-body.plain{white-space:pre-wrap}"
        ".draft-body.md h1,.draft-body.md h2,.draft-body.md h3,.draft-body.md h4{margin:10px 0 5px;font-weight:600;line-height:1.3}"
        ".draft-body.md h1{font-size:16px}"
        ".draft-body.md h2{font-size:15px}"
        ".draft-body.md h3{font-size:14px}"
        ".draft-body.md h4,.draft-body.md h5,.draft-body.md h6{font-size:13px}"
        ".draft-body.md p{margin:5px 0;overflow-wrap:anywhere;word-break:break-word}"
        ".draft-body.md ul,.draft-body.md ol{margin:5px 0;padding-left:20px}"
        ".draft-body.md li{margin:2px 0;overflow-wrap:anywhere;word-break:break-word}"
        ".draft-body.md code{padding:1px 5px;border-radius:3px;background:rgba(127,127,127,0.2);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;overflow-wrap:anywhere;word-break:break-all}"
        ".draft-body.md pre{padding:8px 10px;border-radius:5px;background:rgba(127,127,127,0.17);overflow-x:auto;font-size:11.5px;margin:6px 0;line-height:1.4}"
        ".draft-body.md pre code{background:transparent;padding:0}"
        ".draft-body.md blockquote{margin:6px 0;padding:2px 10px;border-left:3px solid rgba(127,127,127,0.4);opacity:0.85}"
        ".draft-body.md a{color:#2563eb;text-decoration:underline;text-underline-offset:2px}"
        "body.dark .draft-body.md a{color:#60a5fa}"
        ".draft-body.md table{border-collapse:collapse;margin:6px 0;font-size:12px}"
        ".draft-body.md th,.draft-body.md td{padding:4px 8px;border:1px solid rgba(127,127,127,0.3)}"
        ".draft-body.md hr{border:none;border-top:1px solid rgba(127,127,127,0.3);margin:8px 0}"
        ".draft-body .md-fallback{white-space:pre-wrap;font-family:inherit;margin:0}"
        "body.dark .draft-body{background:rgba(255,255,255,0.03)}"
        ".srcs{font-size:11.5px;margin-top:4px}"
        ".srcs-h{font-weight:600;opacity:0.7;margin-bottom:2px}"
        ".srcs ol{margin:0;padding-left:18px}"
        ".srcs li{margin:1px 0;word-break:break-all}"
        ".unsafe-url{opacity:0.55;text-decoration:line-through;font-style:italic}"
        ".reviews-block,.revision-block{font-size:12px;display:flex;flex-direction:column;gap:6px}"
        ".block-h{font-weight:600;opacity:0.75;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em}"
        ".review{padding:8px 10px;border-radius:6px;background:rgba(99,102,241,0.07);border-left:3px solid #6366f1}"
        ".review .rv-head{font-weight:600;font-size:12px;margin-bottom:4px}"
        ".review .rv-feedback{font-size:12.5px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}"
        ".meta-list-wrap{font-size:11.5px;margin-top:5px}"
        ".meta-list-h{font-weight:600;opacity:0.8;margin-bottom:2px}"
        ".meta-list-h{color:inherit}"
        ".meta-list{margin:0;padding:0 0 0 16px;list-style:none}"
        ".meta-list li{padding:1px 0 1px 14px;position:relative;overflow-wrap:anywhere;word-break:break-word}"
        ".meta-list li::before{content:'\u2022';position:absolute;left:-2px;opacity:0.5}"
        ".meta-list.changes li::before{content:'\u270e';color:#10b981}"
        ".meta-list.declined li::before{content:'\u2715';color:#ef4444}"
        ".meta-list.strengths li::before{content:'\u2713';color:#10b981}"
        ".meta-list.improvements li::before{content:'\u25c6';color:#f59e0b}"
        ".rev-draft{margin-top:4px}"
        ".rev-draft>summary{cursor:pointer;list-style:none;padding:4px 6px;border-radius:6px;font-size:11.5px;font-weight:500;background:rgba(127,127,127,0.07);display:flex;align-items:center;gap:6px}"
        ".rev-draft>summary::-webkit-details-marker{display:none}"
        ".rev-draft>summary::before{content:'\u25b8';transition:transform 0.15s;opacity:0.6}"
        ".rev-draft[open]>summary::before{transform:rotate(90deg)}"
        ".rev-draft>summary:hover{background:rgba(127,127,127,0.14)}"
        ".round{margin-top:6px;border:1px solid rgba(127,127,127,0.2);border-radius:8px;overflow:hidden}"
        ".round>summary{cursor:pointer;list-style:none;padding:8px 12px;font-weight:600;font-size:12.5px;background:rgba(127,127,127,0.06);display:flex;align-items:center;gap:6px;user-select:none}"
        ".round>summary::-webkit-details-marker{display:none}"
        ".round>summary::before{content:'\u25b8';transition:transform 0.15s;opacity:0.6}"
        ".round[open]>summary::before{transform:rotate(90deg)}"
        ".round>summary:hover{background:rgba(127,127,127,0.12)}"
        ".round>.agents{padding:10px 10px 12px}"
        ".final-section h3,.history-section h3{margin-bottom:6px}"
        # Incomplete-revision indicator — used both as a per-card badge
        # inside a final draft card (small inline warning) and as a
        # full-width banner at the top of the Final drafts section
        # (aggregate overview). Amber palette matches the conventional
        # "warning but not fatal" signal.
        ".revision-warning{background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.35);color:#92400e;padding:5px 8px;border-radius:5px;font-size:11.5px;font-weight:500;margin:4px 0;display:flex;align-items:center;gap:6px;overflow-wrap:anywhere}"
        "body.dark .revision-warning{background:rgba(245,158,11,0.18);color:#fbbf24;border-color:rgba(245,158,11,0.45)}"
        ".revision-warning-banner{font-size:12px;margin:0 0 8px}"
        "</style></head><body>"
        # Apply the theme BEFORE paint — honours the user's
        # RICH_PROGRESS_THEME valve ('light'/'dark' force), or tries to
        # detect OWUI's theme in 'auto'. See _theme_detect_js().
        f"<script>{_theme_detect_js(force_theme)}</script>"
        f'<div class="wrap{cancel_cls}">'
        f'<div class="head{cancel_cls}"><span class="mark">{header_mark}</span>'
        f'<span>{header_text}</span></div>'
        f'<div class="topic">{_html_escape(topic)}</div>'
        f'<div class="meta"><span>Rounds: {num_rounds}</span>{elapsed_str}</div>'
        f'<div class="pills">{pills_html}</div>'
        f"{final_section}"
        f"{history_section}"
        "</div>"
        "<script>(function(){"
        # Measure body.scrollHeight via rAF so the iframe truly shrinks
        # back when the user collapses a <details> (documentElement's
        # scrollHeight is capped by the parent-assigned viewport height,
        # which is why the chat used to keep a giant gap after a collapse).
        "const post=()=>{requestAnimationFrame(()=>{"
        "const b=document.body;"
        "const h=b?b.scrollHeight:document.documentElement.scrollHeight;"
        "parent.postMessage({type:'iframe:height',height:h},'*');"
        "});};"
        "post();"
        "window.addEventListener('load',post);"
        # Re-measure whenever the user expands OR collapses a <details>.
        "document.addEventListener('toggle',post,true);"
        "})();</script></body></html>"
    )


_ACTIVITY_LOG_LIMIT = 12


def _truncate(s: Any, n: int = 160) -> str:
    text = "" if s is None else str(s)
    text = text.replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "\u2026"


def _parse_agent_status(agent_name: str, description: str) -> Optional[dict]:
    """Pull a run_agent_loop status line apart into a structured activity.

    run_agent_loop emits lines like "[{agent_name}] Step 2: Calling foo" —
    see the emit sites in run_agent_loop. Returns structured entries so the
    iframe can pair ``executing`` with its matching ``result`` into a single
    Open WebUI-style collapsible tool-call card:
      - ``thinking``: {kind: "thinking"}
      - ``executing``: {kind: "executing", name, args}
      - ``result``:    {kind: "result", name, result}
      - ``info``:      {kind: "info", text}
      - ``speech``:    {kind: "speech", text}
      - ``calling``:   None (drop; redundant with executing)
    """
    prefix = f"[{agent_name}]"
    if not description.startswith(prefix):
        return None
    rest = description[len(prefix):].lstrip(" :")

    # Strip a leading "Step N/M:" or "Step N:" prefix so the activity text
    # focuses on the actual event.
    step_match = re.match(r"Step\s+\d+(?:/\d+)?\s*:?\s*", rest)
    if step_match:
        rest = rest[step_match.end():].lstrip()

    if not rest:
        return None

    low = rest.lower()
    if low.startswith("thinking"):
        return {"kind": "thinking"}
    if low.startswith("calling "):
        # Redundant with the per-tool executing entries that follow, and
        # would double up in the UI; skip entirely.
        return None
    exec_match = re.match(r"Executing\s+([^(]+)\((.*)\)\s*$", rest, re.DOTALL)
    if exec_match:
        name = exec_match.group(1).strip()
        raw_args = exec_match.group(2)
        # Submission pseudo-tools carry the full draft / review payload in
        # their arguments, which the Draft / Review side-panels already
        # render. Dumping the same bytes into the activity feed here would
        # both spam the feed and race with live truncation — collapse them
        # to a one-line info card instead. Peek at args to distinguish the
        # initial vs revised label without waiting for the ``returned:``
        # status (which is filtered out below).
        if name == "submit_draft":
            parsed_args: Any = {}
            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                parsed_args = {}
            if isinstance(parsed_args, dict) and (
                parsed_args.get("changes_made")
                or parsed_args.get("feedback_declined")
            ):
                return {"kind": "info", "text": "Revised draft submitted"}
            return {"kind": "info", "text": "Draft submitted"}
        if name == "submit_review":
            return {"kind": "info", "text": "Review submitted"}
        # Keep args readable but cap so huge payloads don't blow up
        # postMessage traffic on every push.
        args = _truncate(raw_args, 1500)
        return {"kind": "executing", "name": name, "args": args}
    returned_match = re.match(r"([^\s]+)\s+returned:\s*(.*)$", rest, re.DOTALL)
    if returned_match:
        name = returned_match.group(1).strip()
        # Drop the submission-tool acknowledgement — the matching
        # ``Executing`` line was already rendered as a "submitted" info
        # card above, so the "tool_name received..." echo would just be
        # redundant noise right under it.
        if name in ("submit_draft", "submit_review"):
            return None
        result = _truncate(returned_match.group(2), 2000)
        return {"kind": "result", "name": name, "result": result}
    if "Max iterations" in rest and "reached" in rest:
        return {"kind": "info", "text": rest}
    parsed_final = safe_json_loads(rest, expect_keys=("draft", "key_feedback"))
    if isinstance(parsed_final, dict) and (
        isinstance(parsed_final.get("draft"), str)
        or isinstance(parsed_final.get("key_feedback"), str)
    ):
        return {
            "kind": "info",
            "text": (
                "Submission rejected: payload emitted as content JSON instead of "
                "calling submit_draft / submit_review"
            ),
        }
    return {"kind": "speech", "text": _truncate(rest, 400)}


class EventEmitter:
    def __init__(
        self,
        event_emitter: Callable[[dict], Any] = None,  # type: ignore
        *,
        enabled: bool = False,
        message_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        render_markdown: bool = True,
        force_theme: str = "auto",
    ) -> None:
        # _raw_emitter is the pristine event_emitter given by Open WebUI.
        # self.event_emitter is what everything else (status emits, execute
        # pushes, downstream agent loops) uses — when rich mode is on it's a
        # wrapper that re-inserts our shell into any `embeds` events emitted
        # by downstream tools, so the live message.embeds replacement doesn't
        # kick our shell out from index 0. See _wrap_emitter below.
        self._raw_emitter = event_emitter
        # Rich progress is only active when explicitly enabled AND we have a
        # message_id to scope postMessage updates to our own iframe.
        self._rich_enabled = bool(enabled) and bool(message_id) and bool(event_emitter)
        self._render_markdown = bool(render_markdown)
        # Normalise theme override to {'auto', 'light', 'dark'}; everything
        # else falls back to 'auto'.
        ft = str(force_theme or "auto").strip().lower()
        self._force_theme = ft if ft in ("auto", "light", "dark") else "auto"
        self._message_id = message_id
        self._chat_id = chat_id
        self.tag = f"owui-{uuid.uuid4().hex[:8]}"
        self._shell_marker = f"<!--llm-review-shell:{self.tag}-->"
        self._shell_html: Optional[str] = None
        self._rich_initialized = False
        self._finalized = False
        # Downstream tools (web search previews, citations, etc.) may emit
        # their own `embeds` events while we're running. The frontend replaces
        # message.embeds on every embed event, so after we emit our final
        # summary those peer embeds would vanish from the live view even
        # though they're safe in DB. Track the non-shell ones here so
        # finalize() can re-append them via a live-only correction event.
        self._preserved_embeds: list = []
        self._preserved_embed_keys: set = set()
        self._total_steps = 1
        # Sub-step tracking lets the bar advance per agent task within a
        # phase instead of jumping only at phase boundaries. _current_step
        # mirrors the integer step set by push(); _sub_total / _sub_done
        # count atomic tasks (compose, review, revise) inside that step.
        self._current_step: int = 0
        self._sub_total: int = 0
        self._sub_done: int = 0
        self._state: dict = {
            "phase": "",
            "value": 0,
            "detail": None,
            "agents": [],
            "topic": "",
        }
        self._agents_by_id: dict[str, dict] = {}
        self._activities_by_id: dict[str, list[dict]] = {}
        # Latest draft text per agent and peer reviews addressed TO each
        # author, so the iframe can render full bodies and per-reviewer
        # feedback alongside the live activity feed.
        self._drafts_by_id: dict[str, dict] = {}
        self._reviews_by_author: dict[str, dict[str, dict]] = {}
        self._lock: Optional[asyncio.Lock] = None
        self.event_emitter: Optional[Callable[[dict], Any]] = (
            self._wrap_emitter(event_emitter) if self._rich_enabled else event_emitter
        )

    def _wrap_emitter(
        self, raw: Callable[[dict], Any]
    ) -> Callable[[dict], Any]:
        async def wrapped(event: dict) -> None:
            event_type = event.get("type") if isinstance(event, dict) else None
            if (
                event_type in ("embeds", "chat:message:embeds")
                and self._rich_initialized
                and not self._finalized
                and self._shell_html is not None
            ):
                data = event.get("data") or {}
                embeds = list(data.get("embeds") or [])
                # Preservation: only track embeds carried by persisted
                # ``type: "embeds"`` events. The tracker exists solely
                # to compensate for a side-effect of the DB persist
                # path — every ``type: "embeds"`` event both writes to
                # DB (via backend ``extend``) and replaces ``message
                # .embeds`` in the live view; our finalize live-fix
                # re-issues those so the live view matches DB again.
                # ``chat:message:embeds`` doesn't touch DB, so we have
                # no DB/live mismatch to correct for there. Whatever a
                # downstream tool chose to render via that channel is
                # its own live-only lifecycle decision — re-issuing
                # such embeds in our finalize payload would override
                # that decision and could show content that the tool
                # never intended to persist past its own next update.
                if event_type == "embeds":
                    for e in embeds:
                        if _is_rich_shell_embed(e, self._shell_marker):
                            continue
                        key = _embed_dedupe_key(e)
                        if key in self._preserved_embed_keys:
                            continue
                        self._preserved_embed_keys.add(key)
                        self._preserved_embeds.append(e)
                # Prepend shell to any embed-carrying event (persisted or
                # live-only) that doesn't already include it — otherwise
                # the frontend's ``message.embeds = data.embeds`` replace
                # would kick our progress shell out of the live view
                # whenever a downstream tool refreshes its own embeds.
                # Our own init_rich/finalize emissions bypass this
                # wrapper (via _raw_emitter) so they aren't affected.
                if not any(
                    _is_rich_shell_embed(e, self._shell_marker) for e in embeds
                ):
                    event = {
                        "type": event_type,
                        "data": {"embeds": [self._shell_html] + embeds},
                    }
            await raw(event)

        return wrapped

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

    async def init_rich(
        self, topic: str, total_steps: int, agents: list[dict]
    ) -> None:
        if not self._rich_enabled or self._rich_initialized:
            return
        self._total_steps = max(1, int(total_steps))
        self._current_step = 0
        self._sub_total = 0
        self._sub_done = 0
        self._agents_by_id = {a["id"]: dict(a) for a in agents}
        self._activities_by_id = {a["id"]: [] for a in agents}
        self._state.update(
            {
                "topic": topic,
                "phase": "Starting",
                "value": 0,
                "detail": None,
                "agents": self._snapshot_agents(),
            }
        )
        self._lock = asyncio.Lock()
        self._shell_html = _render_shell_html(
            self.tag,
            topic,
            self._snapshot_agents(),
            force_theme=self._force_theme,
        )
        assert self._raw_emitter is not None
        # Before we fire the shell, snapshot any embeds that earlier
        # tools in this same assistant turn already persisted. The
        # incoming `type:"embeds"` shell event will replace
        # message.embeds in the live view (backend still keeps them via
        # extend, so reload is fine), and since those embeds were
        # emitted before our wrapper existed we wouldn't otherwise know
        # they're there. Seeding them into _preserved_embeds lets
        # finalize() rebuild a correct live view, and the follow-up
        # live-fix below keeps them visible during the whole run.
        pre_existing: list = []
        if self._chat_id and self._message_id:
            try:
                from open_webui.models.chats import Chats

                msg = await maybe_await(Chats.get_message_by_id_and_message_id(
                    self._chat_id, self._message_id
                ))
                existing = list((msg or {}).get("embeds") or [])
                for e in existing:
                    # Skip BOTH the current run's shell (defensive; shouldn't
                    # appear yet) and any stale shells from prior runs on the
                    # same message (those are height:0 invisible anyway and
                    # re-emitting them would just bloat events).
                    if isinstance(e, str) and "<!--llm-review-shell:" in e:
                        continue
                    key = _embed_dedupe_key(e)
                    if key in self._preserved_embed_keys:
                        continue
                    self._preserved_embed_keys.add(key)
                    self._preserved_embeds.append(e)
                    pre_existing.append(e)
            except Exception as e:
                log.debug(
                    f"[LLMReview] Pre-existing embeds read failed: {e}"
                )
        try:
            # Emit via _raw_emitter so the wrapper doesn't try to re-prepend
            # the shell to this first shell-only embed event.
            await self._raw_emitter(
                {"type": "embeds", "data": {"embeds": [self._shell_html]}}
            )
            if pre_existing:
                # Live-only correction so the user keeps seeing the
                # embeds that were on-screen before we started. The
                # backend doesn't persist chat:message:embeds, so DB is
                # not duplicated — those entries are already in DB from
                # their original producers' persist events.
                try:
                    await self._raw_emitter(
                        {
                            "type": "chat:message:embeds",
                            "data": {
                                "embeds": [self._shell_html] + pre_existing
                            },
                        }
                    )
                except Exception as e:
                    log.debug(
                        f"[LLMReview] Pre-existing live-fix failed: {e}"
                    )
            # Give the iframe srcdoc a moment to load before the first execute.
            await asyncio.sleep(0.05)
            self._rich_initialized = True
        except Exception as e:
            log.warning(f"[LLMReview] Rich progress init failed: {e}")
            self._rich_enabled = False

    def _snapshot_agents(self) -> list[dict]:
        """Build the agent list the iframe renders from, with activities,
        current draft, and peer reviews merged."""
        out: list[dict] = []
        for aid, info in self._agents_by_id.items():
            entry = dict(info)
            entry["activities"] = list(self._activities_by_id.get(aid, []))
            draft_entry = self._drafts_by_id.get(aid)
            if draft_entry:
                entry["draft"] = draft_entry
            review_map = self._reviews_by_author.get(aid) or {}
            entry["reviews"] = list(review_map.values())
            out.append(entry)
        return out

    def _record_activity(self, model_id: str, activity: dict) -> None:
        """Append an activity entry for ``model_id``.

        Tool-call lifecycle (executing → result) is paired into a single
        ``tool_call`` entry so the iframe renders one collapsible card per
        tool invocation rather than two separate rows. Each new tool_call
        gets a stable monotonic ``id`` so the user's expanded-state
        survives rerenders (see renderToolCall / openSet in the shell JS)."""
        if model_id not in self._activities_by_id:
            return
        log_list = self._activities_by_id[model_id]
        kind = activity.get("kind")
        if kind == "thinking":
            # Collapse consecutive "thinking" entries so the log doesn't flood.
            if log_list and log_list[-1].get("kind") == "thinking":
                return
            log_list.append({"kind": "thinking"})
        elif kind == "executing":
            self._tc_counter = getattr(self, "_tc_counter", 0) + 1
            log_list.append(
                {
                    "kind": "tool_call",
                    "id": f"tc{self._tc_counter}",
                    "name": str(activity.get("name") or ""),
                    "args": str(activity.get("args") or ""),
                    "result": "",
                    "status": "running",
                }
            )
        elif kind == "result":
            name = str(activity.get("name") or "")
            result_text = str(activity.get("result") or "")
            matched = False
            # Pair with the most recent still-running tool_call of the
            # same name. We assume FIFO completion within one agent's
            # step, which matches run_agent_loop's serial tool loop.
            for entry in reversed(log_list):
                if (
                    entry.get("kind") == "tool_call"
                    and entry.get("name") == name
                    and entry.get("status") == "running"
                ):
                    entry["result"] = result_text
                    entry["status"] = "done"
                    matched = True
                    break
            if not matched:
                # Orphan result (missed the start somehow) — still surface
                # it as a completed card so the user sees the output.
                self._tc_counter = getattr(self, "_tc_counter", 0) + 1
                log_list.append(
                    {
                        "kind": "tool_call",
                        "id": f"tc{self._tc_counter}",
                        "name": name,
                        "args": "",
                        "result": result_text,
                        "status": "done",
                    }
                )
        elif kind in ("speech", "info"):
            log_list.append({"kind": kind, "text": str(activity.get("text") or "")})
        else:
            return
        if len(log_list) > _ACTIVITY_LOG_LIMIT:
            del log_list[: len(log_list) - _ACTIVITY_LOG_LIMIT]

    async def _flush(self) -> None:
        """Push the current state snapshot to the iframe (lock must be held)."""
        self._state["agents"] = self._snapshot_agents()
        payload = {"tag": self.tag, "state": dict(self._state)}
        selector_id = f"{self._message_id}-embeds-0"
        js = (
            "(function(){var c=document.getElementById("
            f"{_safe_script_json(selector_id)}"
            ");var f=c&&c.querySelector('iframe');"
            "if(!f||!f.contentWindow)return;"
            f"f.contentWindow.postMessage({_safe_script_json(payload)},'*');"
            "})();"
        )
        assert self._raw_emitter is not None
        try:
            # execute events don't need the shell-injection wrapper; go raw.
            await self._raw_emitter(
                {"type": "execute", "data": {"code": js}}
            )
        except Exception as e:
            log.warning(f"[LLMReview] Rich progress push failed: {e}")

    async def push(
        self,
        *,
        step: Optional[int] = None,
        phase: Optional[str] = None,
        detail: Optional[str] = None,
        agent_state: Optional[dict[str, str]] = None,
        total_substeps: Optional[int] = None,
    ) -> None:
        if not self._rich_enabled or not self._rich_initialized:
            return
        assert self._lock is not None
        async with self._lock:
            if step is not None:
                # Switching phase: snap value to the new step boundary and
                # reset sub-step tracking so a stale fraction can't carry
                # into the next phase.
                self._current_step = int(step)
                self._sub_total = 0
                self._sub_done = 0
                self._state["value"] = max(
                    0.0,
                    min(100.0, (float(step) / float(self._total_steps)) * 100.0),
                )
            if total_substeps is not None:
                self._sub_total = max(0, int(total_substeps))
                self._sub_done = 0
            if phase is not None:
                self._state["phase"] = phase
            if detail is not None:
                self._state["detail"] = detail
            if agent_state:
                for aid, s in agent_state.items():
                    if aid in self._agents_by_id:
                        self._agents_by_id[aid]["state"] = s
            await self._flush()

    async def tick_substep(
        self,
        *,
        detail: Optional[str] = None,
        agent_state: Optional[dict[str, str]] = None,
    ) -> None:
        """Advance the bar by ``1 / _sub_total`` within the current phase.

        Used at compose/review/revise task completion so the percentage
        moves smoothly inside a phase. No-op when sub_total is 0 (i.e.
        the caller never declared a sub-step budget for this phase) or
        when rich progress is disabled."""
        if not self._rich_enabled or not self._rich_initialized:
            return
        assert self._lock is not None
        async with self._lock:
            if self._sub_total > 0:
                self._sub_done += 1
                fraction = min(1.0, self._sub_done / self._sub_total)
                self._state["value"] = max(
                    0.0,
                    min(
                        100.0,
                        ((float(self._current_step) + fraction)
                         / float(self._total_steps)) * 100.0,
                    ),
                )
            if detail is not None:
                self._state["detail"] = detail
            if agent_state:
                for aid, s in agent_state.items():
                    if aid in self._agents_by_id:
                        self._agents_by_id[aid]["state"] = s
            await self._flush()

    async def push_activity(
        self, model_id: str, activity: dict
    ) -> None:
        """Append one structured activity for an agent and flush to the
        iframe. ``activity`` shape is the dict returned by
        :func:`_parse_agent_status`."""
        if not self._rich_enabled or not self._rich_initialized:
            return
        assert self._lock is not None
        async with self._lock:
            self._record_activity(model_id, activity)
            await self._flush()

    async def set_draft(
        self,
        model_id: str,
        *,
        draft: Any,
        phase_label: str,
        approach: Optional[str] = None,
        changes_made: Optional[list] = None,
        feedback_declined: Optional[list] = None,
    ) -> None:
        """Record the latest full draft body for ``model_id`` and flush.

        Only accepts a non-empty ``str`` draft — a list/dict/int stringified
        into this slot would be a malformed response masquerading as a real
        draft, so those are silently dropped (main processing's
        ``normalize_draft_result`` still recovers the content separately)."""
        if not self._rich_enabled or not self._rich_initialized:
            return
        if not isinstance(draft, str) or not draft.strip():
            return
        assert self._lock is not None
        async with self._lock:
            entry: dict[str, Any] = {
                "label": phase_label,
                "text": draft,
            }
            if self._render_markdown:
                rendered = _render_markdown(draft)
                if rendered:
                    entry["html"] = rendered
            if isinstance(approach, str) and approach.strip():
                entry["approach"] = approach
            if isinstance(changes_made, list):
                cm = [x for x in changes_made if isinstance(x, str) and x.strip()]
                if cm:
                    entry["changes_made"] = cm
            if isinstance(feedback_declined, list):
                fd = [x for x in feedback_declined if isinstance(x, str) and x.strip()]
                if fd:
                    entry["feedback_declined"] = fd
            self._drafts_by_id[model_id] = entry
            await self._flush()

    async def set_review(
        self,
        *,
        author_id: str,
        reviewer_id: str,
        reviewer_name: str,
        review: Any,
    ) -> None:
        """Record one peer review of ``author_id`` by ``reviewer_id`` and flush.

        ``review`` is typed as Any because it originates from
        ``safe_json_loads`` which may return non-dict shapes for malformed
        LLM output. The rich UI rejects any review whose ``key_feedback``
        is not a non-empty string, so stringified garbage is never shown
        as real peer feedback."""
        if not self._rich_enabled or not self._rich_initialized:
            return
        if not isinstance(review, dict):
            return
        kf = review.get("key_feedback")
        if not isinstance(kf, str) or not kf.strip():
            return
        raw_strengths = review.get("strengths")
        raw_improvements = review.get("improvements")
        strengths = (
            [s for s in raw_strengths if isinstance(s, str) and s.strip()]
            if isinstance(raw_strengths, list)
            else []
        )
        improvements = (
            [s for s in raw_improvements if isinstance(s, str) and s.strip()]
            if isinstance(raw_improvements, list)
            else []
        )
        assert self._lock is not None
        async with self._lock:
            bucket = self._reviews_by_author.setdefault(author_id, {})
            bucket[reviewer_id] = {
                "reviewer": reviewer_name,
                "key_feedback": kf,
                "strengths": strengths,
                "improvements": improvements,
            }
            await self._flush()

    async def clear_reviews(self) -> None:
        """Drop the review bucket so the next round starts with empty feedback.

        Called at the top of every Cross-Review phase so the iframe doesn't
        keep showing stale reviews from the previous round."""
        if not self._rich_enabled or not self._rich_initialized:
            return
        assert self._lock is not None
        async with self._lock:
            self._reviews_by_author = {}
            await self._flush()

    def wrap_agent(
        self, model_id: str, agent_name: str
    ) -> Callable[[dict], Any]:
        """Return an event_emitter-compatible callable that taps events for
        this agent before forwarding them to the original emitter.

        The wrapped callable is passed to run_agent_loop so every
        ``[{agent_name}] …`` status line it emits also lands in the iframe's
        per-agent activity log as a structured entry (thinking / calling /
        executing / result / speech)."""
        original = self.event_emitter
        rich_enabled = self._rich_enabled

        async def wrapped(event: dict) -> None:
            # Always forward so the regular chat status line keeps working.
            if original is not None:
                try:
                    await original(event)
                except Exception as e:
                    log.warning(f"[LLMReview] underlying emitter failed: {e}")
            if not rich_enabled or not self._rich_initialized:
                return
            if not isinstance(event, dict) or event.get("type") != "status":
                return
            data = event.get("data") or {}
            description = data.get("description")
            if not isinstance(description, str):
                return
            parsed = _parse_agent_status(agent_name, description)
            if parsed is None:
                return
            await self.push_activity(model_id, parsed)

        return wrapped

    async def finalize(self, *, summary: dict) -> None:
        if not self._rich_enabled or not self._rich_initialized:
            return
        assert self._raw_emitter is not None
        # Flip the flag BEFORE emitting so any late downstream embeds event
        # races don't re-prepend the shell to the final payload.
        self._finalized = True
        final_html = _render_final_html(summary)
        try:
            # 1) Persisted final: a single [final_html] via type "embeds".
            #    The backend extends this with existing DB embeds (prior
            #    shell snapshots + any downstream tool embeds we wrapped),
            #    so DB ends up with final + everything already stored.
            #    Don't include preserved embeds here — they're already in
            #    DB and would be duplicated.
            await self._raw_emitter(
                {"type": "embeds", "data": {"embeds": [final_html]}}
            )
        except Exception as e:
            log.warning(f"[LLMReview] Rich progress finalize failed: {e}")
            return
        if not self._preserved_embeds:
            return
        try:
            # 2) Live-only correction: the frontend replaces message.embeds
            #    on every embed event, so step (1) just wiped the live view
            #    of any tool-generated embeds. Emit a "chat:message:embeds"
            #    event that the backend doesn't persist (only "embeds"
            #    triggers the DB upsert) to restore them in the live UI.
            #    Shell is intentionally excluded — final view shouldn't
            #    contain the progress skeleton anymore.
            await self._raw_emitter(
                {
                    "type": "chat:message:embeds",
                    "data": {
                        "embeds": [final_html] + list(self._preserved_embeds)
                    },
                }
            )
        except Exception as e:
            log.warning(
                f"[LLMReview] Rich progress live correction failed: {e}"
            )


# ============================================================================
# Helper functions (outside class - AI cannot invoke these)
# ============================================================================


def _iter_balanced_json_spans(text: str):
    """Yield every balanced ``{...}`` span in *text* in source order.

    Unlike the old ``text.find('{') .. rfind('}')`` approach this survives
    prose preambles that also contain curly braces (e.g. ``"About
    {topic1, topic2}. Final: {\"draft\": ...}"``) — each top-level
    balanced pair is surfaced as its own candidate so the caller can try
    parsing them in order and keep the one that's actually valid JSON
    with the schema it's looking for."""
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        closed_at = -1
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        closed_at = j
                        break
            j += 1
        if closed_at >= 0:
            yield text[i : closed_at + 1]
            i = closed_at + 1
        else:
            # unmatched — give up on this start, continue past it
            i += 1


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _iter_json_candidates(text: str):
    """Yield every parseable JSON value that can be extracted from *text*
    in rough priority order: whole-text, each markdown fence, then every
    balanced ``{...}`` span in source order."""
    # Strategy 1 — the whole thing (cheapest for clean output).
    try:
        yield json.loads(text)
    except Exception:
        pass
    # Strategy 2 — every ``` ```json … ``` ``` fenced block.
    for fence in _JSON_FENCE_RE.finditer(text):
        inner = fence.group(1).strip()
        try:
            yield json.loads(inner)
        except Exception:
            for span in _iter_balanced_json_spans(inner):
                try:
                    yield json.loads(span)
                except Exception:
                    continue
    # Strategy 3 — every balanced brace span from anywhere in the text.
    for span in _iter_balanced_json_spans(text):
        try:
            yield json.loads(span)
        except Exception:
            continue


def safe_json_loads(
    text: str, *, expect_keys: Optional[Sequence[str]] = None
) -> Optional[dict]:
    """Best-effort JSON extraction from a noisy LLM response.

    Scans the text for JSON candidates (whole text → markdown fences →
    balanced ``{...}`` spans) and returns the first parseable dict.

    When ``expect_keys`` is provided the match is strict: only a dict
    whose value at **at least one** of those keys is a **non-empty
    string** is returned. This is stricter than "contains the key" —
    LLMs sometimes emit a scratchpad like ``{"approach": "thinking..."}``
    or ``{"changes_made": [...]}`` before the real draft; those share
    *some* schema keys with the target payload but lack the load-bearing
    primary content (``draft`` / ``key_feedback``). Using non-empty
    string validation on the primary key lets us skip past them and
    pick up the real response that follows.

    Without ``expect_keys``, behaviour matches the original "first
    parseable dict wins" contract.

    Returns ``None`` when nothing valid matches."""
    if not text:
        return None
    text = text.strip()

    expected: Optional[set[str]] = None
    if expect_keys is not None:
        expected = {k for k in expect_keys if isinstance(k, str) and k}

    first_dict: Optional[dict] = None
    for cand in _iter_json_candidates(text):
        if not isinstance(cand, dict):
            continue
        if expected is None:
            return cand
        for k in expected:
            v = cand.get(k)
            if isinstance(v, str) and v.strip():
                return cand
        if first_dict is None:
            first_dict = cand

    # expect_keys set but no candidate carried a populated primary key —
    # return None so the caller doesn't silently latch onto a scratchpad
    # dict that happens to share a schema slot.
    if expected is not None:
        return None
    return first_dict


# ============================================================================
# Submission pseudo-tools
# ----------------------------------------------------------------------------
# The compose / review / revise phases historically required the LLM to emit a
# JSON object as its final content (``{"draft": "...", ...}``). That contract
# breaks whenever the model forgets to escape an inner ``"`` — a single
# unescaped quote in prose dumps the whole run into the "failed" branch and
# ``safe_json_loads`` cannot recover the draft.
#
# Native function-calling sidesteps the escaping problem entirely: tool-call
# arguments arrive already structured on the API side. We expose per-phase
# ``submit_draft`` / ``submit_review`` pseudo-tools whose sole purpose is to
# hand the final payload back to the orchestrator through the tool channel
# instead of the content channel. ``run_agent_loop`` treats them as
# loop terminators (executes the call, then exits without another LLM
# round) so the agent cannot drift past submission, and the content-JSON
# fallback is preserved downstream for models that ignore the tool and
# emit JSON anyway.
# ============================================================================


# Pseudo-tool names that should terminate the agent loop after they're called.
# Centralised so run_agent_loop's early-break, _parse_agent_status's activity
# classifier, and the per-phase factories all agree on the same set.
SUBMISSION_TOOL_NAMES = frozenset({"submit_draft", "submit_review"})


def _sources_schema() -> dict:
    """OpenAI function-spec fragment for the ``sources`` array."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the source."},
                "url": {"type": "string", "description": "URL of the source."},
            },
            "required": ["title", "url"],
        },
        "description": (
            "Sources you used during research. Leave empty if none. "
            "Each entry is an object with title and url."
        ),
    }


def build_submit_draft_spec(
    *, phase: Literal["compose", "revise"], include_sources: bool
) -> dict:
    """Build the OpenAI function spec for ``submit_draft``.

    Compose vs revise get different schemas — compose expects ``approach``
    while revise expects ``changes_made`` / ``feedback_declined`` — but
    the tool *name* stays stable so downstream status parsing and
    loop-termination handling only need to recognise one label."""
    properties: dict[str, dict] = {
        "draft": {
            "type": "string",
            "description": (
                "Your complete draft as a single string. Write naturally — "
                "newlines, quotes, and any other characters are handled for "
                "you by the tool channel, so do not attempt manual escaping."
            ),
        }
    }
    required = ["draft"]

    if phase == "compose":
        properties["approach"] = {
            "type": "string",
            "description": (
                "Brief explanation (2-3 sentences) of your writing approach."
            ),
        }
        description = (
            "Submit your final initial draft. Call this EXACTLY ONCE as your "
            "very last action — do not emit any further text or tool calls "
            "after invoking it."
        )
    else:  # revise
        properties["changes_made"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific changes you made based on peer feedback, as short "
                "bullet strings."
            ),
        }
        properties["feedback_declined"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Feedback items you chose not to incorporate (include reasons), "
                "as short bullet strings."
            ),
        }
        description = (
            "Submit your revised draft. Call this EXACTLY ONCE as your very "
            "last action — do not emit any further text or tool calls after "
            "invoking it."
        )
    if include_sources:
        properties["sources"] = _sources_schema()

    return {
        "name": "submit_draft",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def build_submit_review_spec() -> dict:
    """Build the OpenAI function spec for ``submit_review``."""
    return {
        "name": "submit_review",
        "description": (
            "Submit your peer review. Call this EXACTLY ONCE as your very "
            "last action — do not emit any further text or tool calls after "
            "invoking it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific strengths of the draft, as short bullet strings."
                    ),
                },
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific, actionable improvements, as short bullet strings."
                    ),
                },
                "key_feedback": {
                    "type": "string",
                    "description": (
                        "One paragraph summarising your most important feedback."
                    ),
                },
            },
            "required": ["key_feedback"],
        },
    }


def make_submission_tool(*, spec: dict, capture: dict) -> dict:
    """Build a ``tools_dict`` entry that captures the agent's final payload.

    When the agent invokes the tool, the arguments (already parsed and
    escaped by the LLM API's tool-call channel) are stored in the caller-
    owned ``capture`` dict under the ``payload`` key. The tool returns a
    short confirmation string so the agent receives a clear signal that
    submission succeeded and it should end its turn.

    ``execute_tool_call`` filters kwargs against the spec's declared
    properties before invoking the callable, so ``**kwargs`` is safe —
    only schema-declared fields reach the closure."""

    tool_name = spec.get("name", "submit")

    async def _submit(**kwargs: Any) -> str:
        capture["payload"] = dict(kwargs)
        return (
            f"{tool_name} received. End your turn now — do not emit any further "
            "text and do not call any more tools."
        )

    return {
        "callable": _submit,
        "spec": spec,
    }


def advance_last_successful_round_if_chained(
    last_successful_round: dict[str, int],
    aid: str,
    round_num: int,
) -> bool:
    """Advance ``last_successful_round[aid]`` to ``round_num`` iff the
    prior phase succeeded. Returns whether the chain advanced."""
    if last_successful_round.get(aid) == round_num - 1:
        last_successful_round[aid] = round_num
        return True
    return False


def build_tool_call_status_description(
    agent_name: str, tool_name: str, tool_args_raw: Any
) -> str:
    """Format the ``Executing tool(args)`` status line for the activity feed.

    Submission pseudo-tools (``submit_draft`` / ``submit_review``) carry the
    full draft / review text in their arguments. Streaming megabyte-scale
    payloads through every status emit chokes the WebSocket on long outputs
    while adding nothing to the UI — the side panel already renders the
    body. Strip the payload here, preserving only the discriminator fields
    ``_parse_agent_status`` consults to label the activity card.
    """
    if tool_name == "submit_draft":
        try:
            parsed = (
                json.loads(tool_args_raw)
                if isinstance(tool_args_raw, str)
                else (tool_args_raw if isinstance(tool_args_raw, dict) else {})
            )
        except Exception:
            parsed = {}
        is_revise = isinstance(parsed, dict) and bool(
            parsed.get("changes_made") or parsed.get("feedback_declined")
        )
        stub_args = '{"changes_made": [1]}' if is_revise else "{}"
        return f"[{agent_name}] Executing {tool_name}({stub_args})"
    if tool_name == "submit_review":
        return f"[{agent_name}] Executing {tool_name}({{}})"
    args_display = (
        str(tool_args_raw).replace(chr(10), " ") if tool_args_raw else "{}"
    )
    return f"[{agent_name}] Executing {tool_name}({args_display})"


def reset_succeeded_outcomes(
    outcome_map: dict[str, str], agent_ids: Sequence[str]
) -> None:
    """Reset per-agent phase outcomes at a phase boundary.

    Applied at the start of each new compose / revise phase so the
    OLD phase's status doesn't leak into the NEW phase's cancel-time
    labelling. The reset is asymmetric:

    - ``"success"`` → ``"unknown"``: the new phase might legitimately
      succeed, fail, or be cancelled; carrying the previous success
      forward would be a lie about the new phase.
    - ``"failed"`` → retained: a previously-failed agent keeps the
      ``"failed"`` marker so a subsequent cancellation cannot
      mislabel the real pre-existing failure as a cancellation. If
      the new phase happens to succeed, the inline setter in
      compose_draft / revise_draft will overwrite ``"failed"`` with
      ``"success"`` anyway.
    - ``"unknown"`` → retained: nothing to reset.

    Extracted as a module-level helper so the invariants can be
    regression-tested directly; the Tools-method closure calls into
    this function with its local outcome map and agent id list."""
    for aid in agent_ids:
        if outcome_map.get(aid) == "success":
            outcome_map[aid] = "unknown"


def extract_submission_payload(
    capture: dict, primary_key: str
) -> Optional[dict]:
    """Return the captured submission payload iff it carries a usable value.

    ``primary_key`` is the load-bearing field (``draft`` for writers,
    ``key_feedback`` for reviewers). A payload missing a non-empty string
    at that key is treated as not-captured, so the caller falls through
    to the content-JSON fallback rather than latching onto a tool call
    that arrived with all-empty args."""
    payload = capture.get("payload") if isinstance(capture, dict) else None
    if not isinstance(payload, dict):
        return None
    primary = payload.get(primary_key)
    if not isinstance(primary, str) or not primary.strip():
        return None
    return payload


def normalize_text(value: Any) -> str:
    # Only accept real strings. str(non_string) would coerce lists/dicts/ints
    # into plausible-looking prose (e.g. "[1, 2, 3]") and let a malformed LLM
    # response masquerade as a real draft / review / source field in the
    # final JSON output — reject those outright so callers fall back to
    # their explicit fallback text.
    if not isinstance(value, str):
        return ""
    return value.strip()


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
                has_access = await maybe_await(has_connection_access(user, mcp_server_connection))
            except TypeError:
                has_access = await maybe_await(has_connection_access(user, mcp_server_connection, None))

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


def parse_model_ids(value: Any) -> list[str]:
    """Parse a comma-separated string or list of model IDs.

    Duplicate model IDs are intentionally preserved — a caller who passes
    ``["gpt-5", "gpt-5", "gpt-5"]`` wants three agents running the same
    underlying model (with three different personas). Uniqueness among
    agent slots is handled downstream by :func:`llm_review` via synthetic
    agent IDs like ``gpt-5#1`` / ``gpt-5#2`` so internal dicts don't
    collide."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]

    result: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        text = normalize_text(item)
        if not text:
            continue
        result.append(text)
    return result


# ============================================================================
# LLM Review prompt builders
# ============================================================================


def build_compose_system_prompt(
    base_prompt: str,
    persona: dict,
    include_sources: bool,
) -> str:
    sources_note = (
        "If you used any external sources, pass them as the ``sources`` array."
        if include_sources
        else "Leave ``sources`` as an empty array if the tool schema includes it."
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
        "4. You have limited tool-call iterations. Use them wisely.\n\n"
        "FINAL SUBMISSION (tool call):\n"
        "When your draft is ready, call the ``submit_draft`` tool EXACTLY ONCE "
        "as your last action. Put the full draft text in the ``draft`` argument "
        "— newlines, quotation marks, and symbols are passed verbatim through "
        "the tool channel, so write naturally and do not hand-escape anything. "
        "After calling ``submit_draft`` end your turn immediately; do not emit "
        "any further text or tool calls.\n\n"
        "submit_draft arguments:\n"
        "- draft: your complete written response to the topic (required)\n"
        "- approach: brief explanation of your writing approach (2-3 sentences)\n"
        + ("- sources: array of {title, url} objects\n\n" if include_sources else "\n")
        + sources_note
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
        "3. Finish by calling ``submit_draft`` with your full draft, approach, "
        "and any sources."
    )


def parse_review_criteria(value: Any) -> list[str]:
    """Split a user-supplied list of review criteria into trimmed items.

    Accepts either a newline-separated list (preferred) or comma-separated
    — both formats are common and users shouldn't have to guess which one
    the tool wants. Empty/whitespace-only entries are dropped."""
    if not isinstance(value, str):
        return []
    items: list[str] = []
    for raw in value.replace(",", "\n").split("\n"):
        s = raw.strip()
        if s:
            items.append(s)
    return items


def build_review_system_prompt(
    base_prompt: str,
    persona: dict,
    criteria: Optional[list[str]] = None,
) -> str:
    persona_name = persona.get("name", "Reviewer")
    persona_desc = persona.get("description", "")

    # Inject a bulleted evaluation-axes block so reviewers explicitly cover
    # each axis the admin/user configured (e.g. originality, world-building,
    # imagery). When no criteria are set, the prompt keeps its previous
    # persona-only shape.
    criteria_block = ""
    clean_criteria = [c for c in (criteria or []) if isinstance(c, str) and c.strip()]
    if clean_criteria:
        axes_lines = "\n".join(f"- {c.strip()}" for c in clean_criteria)
        criteria_block = (
            "\nEVALUATION AXES (address each explicitly in strengths AND improvements):\n"
            f"{axes_lines}\n"
        )

    review_prompt = (
        f"You are '{persona_name}', providing feedback on a peer's draft.\n"
        f"PERSONA: {persona_desc}\n"
        f"{criteria_block}\n"
        "CRITICAL RULES:\n"
        "1. Provide constructive, specific feedback from your unique perspective.\n"
        "2. Focus on helping the author improve while respecting their voice.\n"
        "3. Be direct but supportive in your critique.\n\n"
        "FINAL SUBMISSION (tool call):\n"
        "When your review is ready, call the ``submit_review`` tool EXACTLY ONCE "
        "as your last action. The tool channel passes strings verbatim, so write "
        "naturally — do not hand-escape quotes or newlines. After calling "
        "``submit_review`` end your turn immediately; do not emit any further "
        "text or tool calls.\n\n"
        "submit_review arguments:\n"
        "- strengths: array of specific strengths in the draft\n"
        "- improvements: array of specific, actionable suggestions\n"
        "- key_feedback: one paragraph summarising your most important feedback (required)"
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
        "Finish by calling ``submit_review`` with your strengths, improvements, "
        "and key feedback."
    )


def build_revise_system_prompt(
    base_prompt: str,
    persona: dict,
    include_sources: bool,
) -> str:
    sources_note = (
        "Update the ``sources`` array if you conducted additional research."
        if include_sources
        else "Leave ``sources`` as an empty array if the tool schema includes it."
    )

    persona_name = persona.get("name", "Writer")
    persona_desc = persona.get("description", "")

    revise_prompt = (
        f"You are '{persona_name}', revising your draft based on peer feedback.\n"
        f"PERSONA: {persona_desc}\n\n"
        "CRITICAL RULES:\n"
        "1. Consider the feedback carefully but maintain your unique voice.\n"
        "2. You may use tools to gather additional information if needed.\n"
        "3. Accept suggestions that improve your work; respectfully decline others.\n\n"
        "FINAL SUBMISSION (tool call):\n"
        "When your revision is ready, call the ``submit_draft`` tool EXACTLY ONCE "
        "as your last action. Put the full revised text in the ``draft`` argument "
        "— newlines, quotation marks, and symbols are passed verbatim through "
        "the tool channel, so write naturally and do not hand-escape anything. "
        "After calling ``submit_draft`` end your turn immediately; do not emit "
        "any further text or tool calls.\n\n"
        "submit_draft arguments:\n"
        "- draft: your revised draft (required)\n"
        "- changes_made: array of specific changes you made based on feedback\n"
        "- feedback_declined: array of suggestions you chose not to incorporate (with reasons)\n"
        + ("- sources: array of {title, url} objects\n\n" if include_sources else "\n")
        + sources_note
    )

    base_prompt = normalize_text(base_prompt)
    if base_prompt:
        return f"{base_prompt}\n\n{revise_prompt}"
    return revise_prompt


def build_revise_user_prompt(
    topic: str,
    original_draft: str,
    feedbacks: list[dict],
    anonymize: bool = False,
) -> str:
    """Build the user prompt for a revision turn.

    When ``anonymize`` is True every reviewer is relabelled as ``Reviewer A``,
    ``Reviewer B`` (based on feedback order) so the revising agent judges
    suggestions on content rather than author identity. The iframe UI and
    the return JSON still show the real persona names — anonymisation is
    only applied to the LLM-facing prompt."""
    feedback_text = ""
    for idx, fb in enumerate(feedbacks):
        if anonymize:
            # Use capital letters A/B/C/... so two reviewers feel like two
            # independent voices without hinting at whose feedback it is.
            label = f"Reviewer {chr(ord('A') + idx)}"
        else:
            label = str(fb.get("reviewer") or "Anonymous")
        key_feedback = fb.get("key_feedback", "No summary provided.")
        strengths = fb.get("strengths", [])
        improvements = fb.get("improvements", [])

        feedback_text += f"\n--- Feedback from '{label}' ---\n"
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
        "Finish by calling ``submit_draft`` with your revised draft, changes_made, "
        "feedback_declined, and any sources."
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
    iteration_note_role: Literal["user", "system"] = "user",
    submission_tool_names: Optional[set[str]] = None,
) -> str:
    """Run the LLM Review agent tool loop until completion.

    Adapted from sub_agent.run_sub_agent_loop with branded status messages.

    ``submission_tool_names`` is an optional set of tool names that signal
    task completion when called. Once the agent invokes any tool in this
    set during an iteration, the loop executes the call(s) for the side
    effect (e.g. capturing the submission payload into a caller-owned
    closure) and then exits without running another LLM round. This both
    saves an iteration and prevents the agent from drifting past the
    submission with additional text or tool calls.
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
            # The old wording told the agent to "provide your final JSON
            # response now" — that directive conflicts with the submission
            # pseudo-tool contract (the new compose/review/revise prompts
            # require a ``submit_draft`` / ``submit_review`` tool call and
            # explicitly forbid content-JSON). When a submission tool is
            # registered, name it so the agent has a clear, unambiguous
            # final action; otherwise fall back to a tool-agnostic nudge.
            if submission_tool_names:
                names = ", ".join(sorted(f"``{n}``" for n in submission_tool_names))
                iteration_info += (
                    f" This is your FINAL opportunity. Call {names} now with "
                    "your complete payload and end your turn — do not emit "
                    "any further text."
                )
            else:
                iteration_info += (
                    " This is your FINAL tool call opportunity. Complete "
                    "your task now and end your turn."
                )

        # Default role is "user" so the leading system message stays at the
        # beginning of the conversation — some chat templates and inference
        # APIs reject requests where a system message appears after the
        # first turn.
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

            # If no tool calls, we're done — except under the strict
            # tool-call contract, where the caller is relying on a
            # submission tool to deposit the final payload via a
            # capture closure. Returning a content-only response there
            # would short-circuit the submission contract entirely:
            # the caller checks the capture box, sees nothing, and
            # marks the phase as failed even when the agent produced
            # perfectly fine prose. Models that prefer free-form
            # answers would fail every single time.
            #
            # Instead, push the assistant turn into history, inject
            # a re-prompt explaining that submission requires the
            # submission tool, and continue. If the model keeps
            # refusing through ``max_iterations``, the loop exits
            # naturally and the max-iterations fallback below
            # (which exposes ONLY the submission tools and tells the
            # agent to submit) gets one more shot at the capture.
            if not tool_calls:
                if submission_tool_names:
                    current_messages.append(
                        {
                            "role": "assistant",
                            "content": content or "",
                        }
                    )
                    names = ", ".join(
                        sorted(f"``{n}``" for n in submission_tool_names)
                    )
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"You did not call {names}. Your turn is "
                                "not complete until you submit your final "
                                f"payload via {names}. Call the tool now "
                                "with the complete payload — do not emit "
                                "the answer as plain text."
                            ),
                        }
                    )
                    if event_emitter:
                        await event_emitter(
                            {
                                "type": "status",
                                "data": {
                                    "description": (
                                        f"[{agent_name}] Step {iteration}: "
                                        "no submission tool called — re-prompting"
                                    ),
                                    "done": False,
                                },
                            }
                        )
                    continue
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

            # Execute each tool call in source order. If any of them lands
            # in ``submission_tool_names`` we stop immediately AFTER that
            # call — both to skip the remaining (now-pointless) tools in
            # the same iteration and to exit the outer while-loop without
            # requesting another LLM round. Running tools after submission
            # would waste compute and, if the agent erroneously called
            # submit_* twice in the same batch, would overwrite the
            # capture payload with whatever came second.
            submission_tool_fired = False
            for tool_call in normalized_tool_calls:
                tc_func = tool_call.get("function")
                tool_name = tc_func.get("name", "unknown") if isinstance(tc_func, dict) else "unknown"
                tool_args_raw = tc_func.get("arguments", "{}") if isinstance(tc_func, dict) else "{}"

                if event_emitter:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": build_tool_call_status_description(
                                    agent_name, tool_name, tool_args_raw
                                ),
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

                if submission_tool_names and tool_name in submission_tool_names:
                    submission_tool_fired = True
                    break

            if submission_tool_fired:
                # Submission payload is already in the caller-owned capture
                # closure; any further LLM round or tool execution here
                # would be drift past the agent's own completion signal.
                return content or ""

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

    # Build the final-chance follow-up. Two contract differences from a
    # normal iteration:
    #
    # 1. Non-submission tools are pruned. The agent has already burned
    #    its exploration budget, so we don't want it digging further
    #    into web-search / Open-Terminal / memory tools. Only the
    #    submission tools (``submit_draft`` / ``submit_review``) remain
    #    exposed.
    # 2. Submission tools MUST stay reachable. Previously this block
    #    ran with no ``tools`` at all and relied on the caller to
    #    parse a JSON object out of ``content``. With the strict
    #    tool-call contract that fallback is gone, so if we don't
    #    expose the submission tools here the agent physically CANNOT
    #    submit — every tool-heavy run that spent its iterations on
    #    research would end in a guaranteed failure even when the
    #    agent actually produced a usable final answer.
    final_tools_param = None
    final_tools_dict = {}
    if submission_tool_names and tools_dict:
        final_tools_dict = {
            name: tool
            for name, tool in tools_dict.items()
            if name in submission_tool_names
        }
        if final_tools_dict:
            final_tools_param = [
                {"type": "function", "function": tool.get("spec", {})}
                for tool in final_tools_dict.values()
            ]

    # Follow-up instruction. When submission tools are exposed, be
    # explicit about what the agent must do next — otherwise models
    # that ignored the submission tool on earlier iterations may
    # default to free-form content again.
    if final_tools_param:
        names = ", ".join(sorted(f"``{n}``" for n in submission_tool_names or ()))
        final_user_content = (
            "Maximum tool iterations reached. Call "
            f"{names} NOW with your complete payload to submit your final "
            "response. No other tools are available — submission is your "
            "only remaining action. Do not emit any additional text."
        )
    else:
        final_user_content = (
            "Maximum tool iterations reached. Please provide your final "
            "answer based on the information gathered so far."
        )

    final_messages = list(current_messages)
    if final_messages and final_messages[-1].get("role") == "user":
        existing = final_messages[-1].get("content", "") or ""
        merged = (
            f"{existing}\n\n{final_user_content}" if existing else final_user_content
        )
        final_messages[-1] = {**final_messages[-1], "content": merged}
    else:
        final_messages.append({"role": "user", "content": final_user_content})

    form_data = {
        "model": model_id,
        "messages": final_messages,
        "stream": False,
        "metadata": {
            "task": "llm_review",
            "sub_agent_iteration": max_iterations + 1,
            "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
        },
    }
    if final_tools_param:
        form_data["tools"] = final_tools_param

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
                        final_content = message.get("content", "") or ""
                        final_tool_calls = message.get("tool_calls", []) or []
                        # Process a submission if the agent took the hint
                        # and called submit_*. This is the whole point of
                        # exposing the submission tools here — without
                        # executing the call the capture box stays empty
                        # and the strict-contract caller would still mark
                        # the phase as failed.
                        if (
                            final_tools_dict
                            and isinstance(final_tool_calls, Sequence)
                            and not isinstance(final_tool_calls, (str, bytes))
                        ):
                            for tc in final_tool_calls:
                                if not isinstance(tc, Mapping):
                                    continue
                                tc_func = tc.get("function")
                                if not isinstance(tc_func, Mapping):
                                    continue
                                tool_name = tc_func.get("name", "")
                                if tool_name not in submission_tool_names:
                                    # Non-submission tools were pruned
                                    # from tools_param, so the API
                                    # shouldn't return them — but
                                    # defensive skip keeps us robust
                                    # against misbehaving providers.
                                    continue
                                # Normalize args to a JSON string so
                                # execute_tool_call's parser path
                                # matches the regular iteration loop.
                                args = tc_func.get("arguments", "{}")
                                if not isinstance(args, str):
                                    try:
                                        args = json.dumps(
                                            args, ensure_ascii=False
                                        )
                                    except Exception:
                                        args = str(args)
                                normalized_tc = {
                                    **tc,
                                    "function": {**tc_func, "arguments": args},
                                }
                                if event_emitter:
                                    await event_emitter(
                                        {
                                            "type": "status",
                                            "data": {
                                                "description": build_tool_call_status_description(
                                                    agent_name, tool_name, args
                                                ),
                                                "done": False,
                                            },
                                        }
                                    )
                                try:
                                    await execute_tool_call(
                                        normalized_tc,
                                        final_tools_dict,
                                        {
                                            **extra_params,
                                            "__messages__": current_messages,
                                        },
                                        event_emitter=event_emitter,
                                    )
                                except Exception as exec_exc:
                                    log.warning(
                                        f"[LLMReview] final-iteration submission "
                                        f"tool exec failed: {exec_exc}"
                                    )
                                # One submission is enough — the capture
                                # box is now populated (or the call
                                # genuinely failed). Stop here so a
                                # duplicate submission cannot overwrite
                                # it.
                                break
                        return final_content
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

        all_builtin_tools = await maybe_await(get_builtin_tools(
            request=request,
            extra_params=builtin_extra_params,
            features=features,
            model=model,
        ))

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


async def register_view_skill(
    tools_dict: dict,
    request: Request,
    extra_params: dict,
) -> None:
    """Manually register the view_skill builtin tool in tools_dict.

    This is needed for **model-attached** skills whose content is not injected
    inline.  The review agent can call ``view_skill`` to lazily load their content
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

    return await maybe_await(get_filtered_models(filtered, user))


def extract_model_ids(models: list[dict]) -> list[str]:
    ids = []
    seen = set()
    for model in models:
        model_id = normalize_text(model.get("id"))
        if model_id and model_id not in seen:
            seen.add(model_id)
            ids.append(model_id)
    return ids


def resolve_chat_model_id(
    metadata: Optional[dict], model_dict: Optional[dict]
) -> str:
    # Open WebUI injects the lightweight task model into __model__, NOT the
    # user's chat model. The real chat model lives in __metadata__["model"]["id"].
    # See sub_agent.py for the same pattern.
    chat_model_id = ""
    if metadata:
        chat_model_id = (metadata.get("model") or {}).get("id", "") or ""
    if not chat_model_id and model_dict:
        chat_model_id = model_dict.get("id", "") or ""
    return chat_model_id


def resolve_review_model_ids(
    valves: Any,
    models_arg: Optional[str],
    chat_model_id: str,
) -> list[str]:
    default_models = parse_model_ids(getattr(valves, "DEFAULT_MODELS", ""))

    if getattr(valves, "ALLOW_AI_MODEL_SELECTION", False):
        provided_models = parse_model_ids(models_arg)
    else:
        provided_models = []

    if provided_models:
        model_ids = list(provided_models)
    elif default_models:
        model_ids = list(default_models)
    elif chat_model_id:
        model_ids = [chat_model_id, chat_model_id, chat_model_id]
    else:
        model_ids = []

    if 0 < len(model_ids) < 3 and chat_model_id:
        while len(model_ids) < 3:
            model_ids.append(chat_model_id)

    return model_ids[:3]


# ============================================================================
# Tools class
# ============================================================================


class Tools:
    """LLM Review collaborative writing tool."""

    class Valves(BaseModel):
        ALLOW_AI_MODEL_SELECTION: bool = Field(
            default=False,
            description=(
                "If True, the AI may dynamically choose models via the `models` "
                "parameter of llm_review. If False (default, safer), the AI's "
                "`models` argument is silently ignored and the lineup falls back "
                "to DEFAULT_MODELS or the current chat model."
            ),
        )
        DEFAULT_MODELS: str = Field(
            default="",
            description=(
                "Comma-separated default model IDs (1-3 entries). If fewer than 3 "
                "are provided, the remaining slots are padded with the current "
                "chat model. If empty, the current chat model is used for all 3 "
                "slots. Always honored regardless of ALLOW_AI_MODEL_SELECTION; "
                "only consulted when the AI did not (or was not allowed to) "
                "specify models."
            ),
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
            default=False,
            description=(
                "Enable memory tools (search_memories, add_memory, "
                "replace_memory_content). Off by default for LLM Review: review "
                "agents occasionally dump their drafts into long-term memory "
                "instead of submitting them via submit_draft/submit_review, "
                "polluting it. Turn on only if review agents genuinely need to "
                "consult user memories."
            ),
        )
        ENABLE_NOTES_TOOLS: bool = Field(
            default=False,
            description=(
                "Enable notes tools (search_notes, view_note, write_note, "
                "replace_note_content). Off by default for LLM Review: review "
                "agents will otherwise persist drafts as notes instead of "
                "submitting them via submit_draft/submit_review. Turn on only "
                "if review agents should read/modify the user's notes."
            ),
        )
        ENABLE_CHANNELS_TOOLS: bool = Field(
            default=True,
            description="Enable channels tools (search_channels, search_channel_messages, etc.).",
        )
        ENABLE_TERMINAL_TOOLS: bool = Field(
            default=False,
            description=(
                "Enable Open Terminal tools when terminal_id is available in chat "
                "metadata (e.g., run_command, list_files, read_file, write_file, "
                "display_file). Off by default for LLM Review: review agents "
                "occasionally write drafts directly to disk via write_file "
                "instead of submitting them via submit_draft/submit_review, "
                "which is an unwanted side effect for a drafting workflow. "
                "Turn on only if review agents genuinely need terminal access."
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
        ENABLE_TASK_TOOLS: bool = Field(
            default=False,
            description="Enable task management tools (create_tasks, update_task).",
        )
        ENABLE_AUTOMATION_TOOLS: bool = Field(
            default=False,
            description="Enable automation tools (create/update/list/toggle/delete automations). Off by default for LLM Review to avoid scheduling side effects.",
        )
        ENABLE_CALENDAR_TOOLS: bool = Field(
            default=False,
            description="Enable calendar tools (search/create/update/delete calendar events).",
        )
        ITERATION_NOTE_ROLE: Literal["user", "system"] = Field(
            default="user",
            description=(
                "Role used for the per-iteration meta note appended to each review-agent "
                "request (e.g. '[Iteration 2/5]'). Default 'user' keeps the system message "
                "at the beginning of the conversation, preserving prompt caching and avoiding "
                "'System message must be at the beginning' errors reported by some chat "
                "templates or inference APIs. Set to 'system' to restore the pre-0.4.2 "
                "behaviour (the meta note is appended as a standalone system message at the "
                "end of each request) — use this if the new default causes any regression "
                "with your model; note that it may re-trigger the system-position error."
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
You are a collaborative writing agent participating in a multi-agent review process.

CRITICAL RULES:
1. Maintain your unique perspective and writing voice throughout.
2. Use tools when they help research and improve your writing.
3. Submit your final output ONLY via the ``submit_draft`` / ``submit_review`` tool call specified in the phase-specific instructions below — never dump JSON into your content.
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
        RICH_PROGRESS: bool = Field(
            default=True,
            description=(
                "Show a rich iframe progress bar in the chat while LLM Review "
                "runs: per-agent dashboard with live tool-call cards, draft "
                "preview, and peer review panels, plus a persistent completion "
                "summary. Disable to fall back to plain status-only progress "
                "if the rich UI isn't needed, or if it conflicts with other "
                "embeds emitted during the same message."
            ),
        )
        RENDER_DRAFTS_AS_MARKDOWN: bool = Field(
            default=True,
            description=(
                "Render draft bodies as Markdown (headings, lists, bold/italic, code, "
                "links) inside the rich progress UI. Disable to show drafts as plain "
                "monospace-friendly pre-formatted text. Only affects the iframe view "
                "— the JSON returned to the AI is unchanged."
            ),
        )
        RICH_PROGRESS_THEME: Literal["auto", "light", "dark"] = Field(
            default="auto",
            description=(
                "Force the rich UI color scheme. 'auto' (default) tries to detect "
                "Open WebUI's theme; if the rich UI looks unreadable (e.g. light "
                "text on a light chat), pick 'light' or 'dark' to match your OWUI "
                "theme.\n\n"
                "Why 'auto' can fail: OWUI renders our iframe with "
                "sandbox=allow-scripts and, by default, WITHOUT allow-same-origin, "
                "which blocks us from probing the parent's html.dark class. "
                "'auto' then falls back to the OS preference "
                "(prefers-color-scheme), which is wrong whenever the OS theme and "
                "the OWUI theme differ.\n\n"
                "Two ways to make 'auto' work correctly:\n"
                "  (1) In OWUI, open Settings > Interface and turn ON "
                "'iframe Sandbox Allow Same Origin' — our iframe can then read "
                "OWUI's theme directly and even follow live theme switches.\n"
                "  (2) Or simply set this valve to 'light' / 'dark' to match "
                "your current OWUI theme without changing any global setting."
            ),
        )
        INCLUDE_FULL_HISTORY: bool = Field(
            default=False,
            description=(
                "Include every intermediate draft body in the return JSON's "
                "rounds array (round 0 initial-composition full drafts, plus "
                "each per-round revision's draft + sources). Default leaves "
                "them out to keep the tool-call context small — final drafts "
                "are always present, and the Rich iframe keeps the full live "
                "history regardless of this setting. Turn on when the AI "
                "needs programmatic access to per-round draft progression."
            ),
        )
        ANONYMIZE_REVIEWERS: bool = Field(
            default=False,
            description=(
                "When revising, hide peer reviewer persona names and relabel them "
                "as 'Reviewer A'/'Reviewer B' in the prompt so the agent judges "
                "feedback by its content rather than by who wrote it. The iframe "
                "UI and return JSON still carry the real reviewer names."
            ),
        )
        REVIEW_CRITERIA: str = Field(
            default="",
            description=(
                "Optional newline- or comma-separated list of review axes the peer "
                "reviewers should explicitly cover (e.g. 'originality\\nworld-"
                "building\\nimagery\\ncharacter depth'). When blank, reviewers use "
                "their persona's own priorities only. Each axis is surfaced to the "
                "reviewer as a bullet inside the review system prompt."
            ),
        )
        # Default personas — used when the AI doesn't pass its own `personas`
        # argument or passes fewer than 3 entries. Each persona must have a
        # name AND description; if either is blank, the built-in fallback for
        # that slot is used instead.
        DEFAULT_PERSONA_1_NAME: str = Field(
            default="Analytical Thinker",
            description="Persona 1 name (used when the AI omits the personas argument).",
        )
        DEFAULT_PERSONA_1_DESCRIPTION: str = Field(
            default=(
                "You approach topics with rigorous logical analysis and systematic thinking. "
                "You value clarity, precision, and evidence-based reasoning. "
                "You excel at identifying assumptions, evaluating arguments, and structuring complex ideas coherently."
            ),
            description="Persona 1 description (drives the agent's voice and priorities).",
        )
        DEFAULT_PERSONA_2_NAME: str = Field(
            default="Creative Visionary",
            description="Persona 2 name (used when the AI omits the personas argument).",
        )
        DEFAULT_PERSONA_2_DESCRIPTION: str = Field(
            default=(
                "You bring imaginative and innovative perspectives to any topic. "
                "You value originality, narrative engagement, and thought-provoking ideas. "
                "You excel at finding unexpected connections and presenting ideas in compelling ways."
            ),
            description="Persona 2 description (drives the agent's voice and priorities).",
        )
        DEFAULT_PERSONA_3_NAME: str = Field(
            default="Practical Strategist",
            description="Persona 3 name (used when the AI omits the personas argument).",
        )
        DEFAULT_PERSONA_3_DESCRIPTION: str = Field(
            default=(
                "You focus on real-world applicability and actionable insights. "
                "You value practicality, feasibility, and concrete outcomes. "
                "You excel at considering implementation challenges and stakeholder perspectives."
            ),
            description="Persona 3 description (drives the agent's voice and priorities).",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def list_models(
        self,
        query: str = "",
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> str:
        """
        List model IDs available to the current user. Call this only when you need to know which model IDs can be specified.

        :param query: Optional case-insensitive substring to filter model IDs/names; pass an empty string for all.

        Returns a JSON string. The shape varies with operator configuration:
        - {"models": ["id1", "id2", ...]} when free model selection is enabled
        - {"message": "...", "models_that_will_be_used": [...]} when the operator has locked the model lineup
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

        if not getattr(self.valves, "ALLOW_AI_MODEL_SELECTION", False):
            chat_model_id = resolve_chat_model_id(__metadata__, __model__)
            locked_ids = resolve_review_model_ids(
                self.valves, models_arg=None, chat_model_id=chat_model_id
            )
            payload: dict[str, Any] = {
                "message": (
                    "Model selection is locked by the operator. llm_review will "
                    "run with the models below regardless of any `models` "
                    "argument."
                ),
                "models_that_will_be_used": locked_ids,
            }
            if not locked_ids:
                payload["message"] = (
                    "Model selection is locked by the operator and no fallback "
                    "model is available (DEFAULT_MODELS is empty and the chat "
                    "model could not be detected). llm_review will fail until "
                    "the operator sets DEFAULT_MODELS."
                )
            return json.dumps(payload, ensure_ascii=False)

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

        Three persona agents independently draft the topic, exchange peer
        feedback, and revise over UserValves.ROUNDS rounds. Each agent's voice
        is preserved across rounds. Returns one draft per persona — never a
        single merged output — so use this for tasks where divergent angles
        matter (essays, design analyses, brainstorming, creative writing), not
        for queries needing one authoritative answer.

        Independence: drafts are composed in parallel (no agent sees others');
        reviewers see only the single assigned draft; revising agents see only
        their own prior draft plus feedback summaries on it — never other
        agents' raw drafts. UserValves.ANONYMIZE_REVIEWERS additionally hides
        reviewer identities for fully blind review.

        :param topic: The writing topic or prompt (required).
        :param requirements: Optional specific requirements, constraints, length targets, or style notes.
        :param models: Optional comma-separated 3 model IDs; honored only when the operator enabled AI model selection in Valves, otherwise silently ignored — DEFAULT_MODELS or the current chat model is used instead.
        :param personas: Optional list of exactly 3 persona objects with `name` and `description`; choose markedly contrasting profiles to maximize divergence — similar personas defeat the tool's purpose. If fewer than 3 are provided, defaults are used (Analytical Thinker, Creative Visionary, Practical Strategist).
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
        request = __request__
        metadata = __metadata__ or {}
        raw_user_valves = (__user__ or {}).get("valves", {})
        user_valves = coerce_user_valves(raw_user_valves, self.UserValves)

        include_sources = bool(user_valves.INCLUDE_SOURCES)
        num_rounds = int(user_valves.ROUNDS)
        anonymize_reviewers = bool(
            getattr(user_valves, "ANONYMIZE_REVIEWERS", False)
        )
        review_criteria = parse_review_criteria(
            getattr(user_valves, "REVIEW_CRITERIA", "") or ""
        )
        include_full_history = bool(
            getattr(user_valves, "INCLUDE_FULL_HISTORY", False)
        )

        emitter = EventEmitter(
            __event_emitter__,
            enabled=bool(user_valves.RICH_PROGRESS),
            message_id=__message_id__,
            chat_id=__chat_id__,
            render_markdown=bool(getattr(user_valves, "RENDER_DRAFTS_AS_MARKDOWN", True)),
            force_theme=str(getattr(user_valves, "RICH_PROGRESS_THEME", "auto") or "auto"),
        )

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
            # Pull defaults from UserValves (admin/user-editable). Each slot
            # falls back to the hard-coded persona if either the name or the
            # description is left blank, so a partially-cleared form never
            # leaves an agent without a usable voice.
            valve_defaults = []
            for idx, fallback in enumerate(DEFAULT_PERSONAS, start=1):
                vname = normalize_text(getattr(user_valves, f"DEFAULT_PERSONA_{idx}_NAME", ""))
                vdesc = normalize_text(
                    getattr(user_valves, f"DEFAULT_PERSONA_{idx}_DESCRIPTION", "")
                )
                if vname and vdesc:
                    valve_defaults.append({"name": vname, "description": vdesc})
                else:
                    valve_defaults.append(dict(fallback))
            agent_personas = valve_defaults

        # Resolve model IDs.
        # Priority: AI-provided `models` (only if ALLOW_AI_MODEL_SELECTION) >
        # DEFAULT_MODELS valve > current chat model × 3. Sub-3 lists are padded
        # with the chat model. See resolve_review_model_ids() for the helper.
        chat_model_id = resolve_chat_model_id(__metadata__, __model__)
        model_ids = resolve_review_model_ids(
            self.valves, models_arg=models, chat_model_id=chat_model_id
        )

        if len(model_ids) < 3:
            try:
                available_models = await get_available_models(__request__, user)
                available_ids = extract_model_ids(available_models)
            except Exception:
                available_ids = []

            return json.dumps(
                {
                    "error": "Cannot determine 3 models for LLM Review.",
                    "provided_models": model_ids,
                    "available_models": available_ids,
                    "action": (
                        "Set DEFAULT_MODELS in Valves (1-3 IDs; <3 will be "
                        "padded with the current chat model), or enable "
                        "ALLOW_AI_MODEL_SELECTION and pass `models` explicitly."
                    ),
                },
                ensure_ascii=False,
            )

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
            if getattr(self.valves, "ALLOW_AI_MODEL_SELECTION", False):
                action = "Pick model IDs from available_models or call list_models()."
            else:
                action = (
                    "Operator-configured DEFAULT_MODELS contains unknown IDs (or "
                    "the current chat model is unavailable). Ask the user to fix "
                    "DEFAULT_MODELS in the Valves UI."
                )
            return json.dumps(
                {
                    "error": "Unknown model IDs.",
                    "unknown_models": unknown_ids,
                    "available_models": sorted(list(available_ids)),
                    "action": action,
                },
                ensure_ascii=False,
            )

        # Build per-slot unique agent IDs so the same underlying model can
        # occupy multiple slots (e.g. run "gpt-5" three times with three
        # personas) without dict-key collisions. When a model only appears
        # once it keeps its raw ID; duplicates get a ``#N`` suffix per
        # occurrence. A final while-loop guarantees absolute uniqueness
        # even in adversarial edge cases — for example a user passing
        # ``["gpt-5", "gpt-5#2", "gpt-5"]`` where the raw suffix form
        # clashes with a later synthesised id. The ``agent_model`` lookup
        # always maps back to the real model ID used at the LLM call sites.
        dup_counts: dict[str, int] = {}
        for _mid in model_ids:
            dup_counts[_mid] = dup_counts.get(_mid, 0) + 1
        agent_ids: list[str] = []
        agent_model: dict[str, str] = {}
        used_agent_ids: set[str] = set()
        occ: dict[str, int] = {}
        for _mid in model_ids:
            if dup_counts[_mid] > 1:
                occ[_mid] = occ.get(_mid, 0) + 1
                aid = f"{_mid}#{occ[_mid]}"
            else:
                aid = _mid
            # Escalate with ``~N`` if the candidate already collides with
            # an earlier slot's id (e.g. the raw model id literally
            # contained a ``#N`` that matches what we just synthesised).
            if aid in used_agent_ids:
                bump = 2
                base = aid
                while f"{base}~{bump}" in used_agent_ids:
                    bump += 1
                aid = f"{base}~{bump}"
            used_agent_ids.add(aid)
            agent_ids.append(aid)
            agent_model[aid] = _mid

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
            # Use emitter.event_emitter (the shell-injecting wrapper when
            # RICH_PROGRESS is on, raw otherwise) so any downstream tool that
            # emits `embeds` also keeps our progress shell at index 0.
            "__event_emitter__": emitter.event_emitter,
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

        # Assign personas to agent slots (first 3 personas). Keyed by the
        # synthetic agent_id so duplicate underlying models don't collapse.
        agents = {agent_ids[i]: agent_personas[i] for i in range(3)}

        # One compose phase + (review + revise) per round.
        total_steps = 1 + 2 * num_rounds
        await emitter.init_rich(
            topic=topic_text,
            total_steps=total_steps,
            agents=[
                {
                    "id": aid,
                    "persona": agents[aid]["name"],
                    "model": agent_model[aid],
                    "state": "idle",
                }
                for aid in agent_ids
            ],
        )

        await emitter.emit(
            description=f"LLM Review Starting: {topic_text}",
            done=False,
        )
        started_at = asyncio.get_event_loop().time()

        # Cache (tools_dict, terminal_prompt, direct_prompts) per REAL
        # model_id — not agent_id — so three agents running the same model
        # share one tool resolution pass instead of triple-building it.
        tools_cache: dict[str, tuple[dict, Optional[str], list[str]]] = {}
        all_mcp_clients: list[dict[str, Any]] = []
        rounds_data: list[dict] = []
        # current_drafts: latest body for each agent, updated on every
        # successful submission so cross-review during the run sees real
        # prose. chained_drafts: only mirrored when the success chain
        # actually advances; consumed by final_drafts so the displayed
        # body matches last_successful_round.
        current_drafts: dict[str, dict] = {}
        chained_drafts: dict[str, dict] = {}
        # -1 = compose did not submit; 0 = compose only; N = revise N.
        last_successful_round: dict[str, int] = {aid: -1 for aid in agent_ids}
        # Rounds where the agent called submit_draft regardless of chain
        # state. Lets the renderer differentiate "submitted on top of
        # broken chain" from "did not submit at all".
        submitted_rounds: dict[str, set[int]] = {aid: set() for aid in agent_ids}
        # Per-agent outcome of the most recently ATTEMPTED phase. Reset
        # to "unknown" at the start of each phase and set to "success"
        # or "failed" inline inside compose_draft / revise_draft just
        # before the task returns — so if cancellation interrupts a
        # phase mid-flight the entry stays at "unknown". This lets the
        # cancel-time renderer tell a mixed run apart: an agent whose
        # latest attempted phase cleanly failed before cancellation
        # shows "round N failed", while an agent that was actively
        # running when the stop button was pressed shows "cancelled
        # before round N could complete". Without this distinction,
        # a single ``is_cancelled`` flag mislabels all incomplete
        # agents as cancellation victims.
        per_agent_last_phase_outcome: dict[str, str] = {
            aid: "unknown" for aid in agent_ids
        }

        def reset_phase_outcomes() -> None:
            """Thin wrapper over the module-level ``reset_succeeded_outcomes``
            so this closure can be called at phase boundaries without
            re-stating the semantic every time. See that helper's
            docstring for why "failed" is preserved across resets."""
            reset_succeeded_outcomes(per_agent_last_phase_outcome, agent_ids)

        async def ensure_tools(
            agent_id: str,
        ) -> tuple[dict, Optional[str], list[str]]:
            real_model_id = agent_model[agent_id]
            cached = tools_cache.get(real_model_id)
            if cached is not None:
                return cached
            member_model = request.app.state.MODELS.get(real_model_id, {})
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
                await register_view_skill(tools_dict, request, scratch_extra_params)

            terminal_prompt = scratch_extra_params.get("__terminal_system_prompt__")
            direct_prompts = scratch_extra_params.get(
                "__direct_tool_server_system_prompts__", []
            ) or []
            cached = (tools_dict, terminal_prompt, list(direct_prompts))
            tools_cache[real_model_id] = cached
            if mcp_clients:
                all_mcp_clients.append(mcp_clients)
            return cached

        def make_member_extra_params(
            agent_id: str,
            terminal_prompt: Optional[str],
            direct_prompts: list[str],
        ) -> dict:
            member_model = request.app.state.MODELS.get(agent_model[agent_id], {})
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
            await emitter.push(
                step=0,
                phase="Phase 1: Initial Composition",
                agent_state={aid: "composing" for aid in agent_ids},
                total_substeps=len(agent_ids),
            )
            reset_phase_outcomes()

            async def compose_draft(agent_id: str) -> tuple[str, str, dict]:
                persona = agents[agent_id]
                # ``state`` stays "failed" unless we actually produce a usable
                # draft. run_agent_loop returns API/auth/parse errors as
                # string content (e.g. "API error: ..."), so a successful
                # ``await`` is NOT enough — we must see a non-empty string
                # ``draft`` field in parsed JSON to call the task a success.
                # The try/finally also guarantees tick_substep fires on
                # exception paths so the progress bar isn't frozen at the
                # failed agent's sub-step.
                state = "failed"
                content: str = ""
                parsed: Any = {}
                try:
                    tools_dict, terminal_prompt, direct_prompts = await ensure_tools(
                        agent_id
                    )
                    member_extra_params = make_member_extra_params(
                        agent_id, terminal_prompt, direct_prompts
                    )

                    system_prompt = build_compose_system_prompt(
                        base_system_prompt, persona, include_sources
                    )
                    user_prompt = build_compose_user_prompt(topic_text, requirements_text)

                    # Capture box + submission pseudo-tool. The agent calls
                    # ``submit_draft`` with its final payload and the tool
                    # channel deposits already-escaped args here — no more
                    # content-JSON parsing hazard. A per-agent capture box
                    # keeps compose_draft tasks that gather in parallel
                    # from overwriting each other's payloads.
                    submit_capture: dict = {"payload": None}
                    submit_spec = build_submit_draft_spec(
                        phase="compose", include_sources=include_sources
                    )
                    # Build a fresh dict so the cached ``tools_dict`` from
                    # ``ensure_tools`` is not mutated (multiple agents may
                    # share the same cache entry).
                    combined_tools = {
                        **tools_dict,
                        "submit_draft": make_submission_tool(
                            spec=submit_spec, capture=submit_capture
                        ),
                    }

                    content = await run_agent_loop(
                        request=request,
                        user=user,
                        model_id=agent_model[agent_id],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        tools_dict=combined_tools,
                        max_iterations=self.valves.MAX_ITERATIONS,
                        extra_params=member_extra_params,
                        apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                        agent_name=f"{persona['name']}",
                        event_emitter=emitter.wrap_agent(agent_id, persona["name"]),
                        iteration_note_role=self.valves.ITERATION_NOTE_ROLE,
                        submission_tool_names={"submit_draft"},
                    )

                    # Strict tool-call contract: the only authoritative
                    # source of the draft payload is the submit_draft
                    # capture box. The prompt explicitly forbids content
                    # JSON, so we do NOT silently resurrect a content-JSON
                    # fallback here — that would both contradict the
                    # prompt and re-open the unescaped-quote parse bug
                    # this whole migration set out to fix. Missing
                    # capture → state stays "failed" with a clear label.
                    parsed = extract_submission_payload(submit_capture, "draft")
                    if isinstance(parsed, dict):
                        draft_val = parsed.get("draft")
                        if isinstance(draft_val, str) and draft_val.strip():
                            state = "drafted"
                            approach_val = parsed.get("approach")
                            approach_clean = (
                                approach_val.strip()
                                if isinstance(approach_val, str) and approach_val.strip()
                                else None
                            )
                            try:
                                await emitter.set_draft(
                                    agent_id,
                                    draft=draft_val,
                                    phase_label="Initial composition",
                                    approach=approach_clean,
                                )
                            except Exception as e:
                                log.warning(f"[LLMReview] set_draft failed: {e}")
                            # Commit now so a mid-gather cancel preserves
                            # what the iframe just displayed. The outer
                            # aggregation loop below also assigns this
                            # key, which is idempotent because it uses
                            # the same normalize_draft_result(parsed,...)
                            # output.
                            current_drafts[agent_id] = normalize_draft_result(
                                parsed,
                                f"Agent did not submit via submit_draft. Response: {truncate_text(content, 180)}",
                            )
                            chained_drafts[agent_id] = current_drafts[agent_id]
                            # Inline so a mid-gather cancel still sees
                            # this agent as having finished compose;
                            # outer aggregation re-assigns the same
                            # value, idempotent.
                            last_successful_round[agent_id] = 0
                            submitted_rounds[agent_id].add(0)
                            per_agent_last_phase_outcome[agent_id] = "success"
                    # If we reach here with parsed still None or missing
                    # a usable draft, the compose phase completed normally
                    # but produced no submission — that's a clean failure,
                    # not a cancellation. Tag it so the cancel-time
                    # renderer can distinguish this from agents that were
                    # interrupted mid-phase (outcome stays "unknown" for
                    # those because CancelledError bypasses this block).
                    if per_agent_last_phase_outcome[agent_id] == "unknown":
                        per_agent_last_phase_outcome[agent_id] = "failed"
                    return agent_id, content, parsed or {}
                except asyncio.CancelledError:
                    # Genuine in-flight cancellation: leave outcome at
                    # "unknown" so the renderer picks the "cancelled"
                    # label branch. Re-raise so gather sees the cancel
                    # and the outer try/except:CancelledError path in
                    # the main body can run the cancel finalize.
                    raise
                except BaseException:
                    # Non-cancel exception (e.g. ensure_tools crash, OOM,
                    # transport blowup, etc.) before the success/failure
                    # inline setters fired. Without this branch the
                    # outcome stays "unknown" which the cancel-time
                    # renderer would mislabel as cancellation. Treating
                    # it as a clean failure keeps "failed" badges on
                    # real tool failures even when the user happens to
                    # press stop shortly after.
                    if per_agent_last_phase_outcome[agent_id] == "unknown":
                        per_agent_last_phase_outcome[agent_id] = "failed"
                    raise
                finally:
                    await emitter.tick_substep(agent_state={agent_id: state})

            compose_results = await asyncio.gather(
                *[compose_draft(aid) for aid in agent_ids], return_exceptions=True
            )

            for idx, result in enumerate(compose_results):
                agent_id = agent_ids[idx]
                if isinstance(result, BaseException):
                    log.exception(f"Compose failed ({agent_id}): {result}")
                    current_drafts[agent_id] = normalize_draft_result(
                        None, f"Composition failed: {truncate_text(str(result), 180)}"
                    )
                    chained_drafts[agent_id] = current_drafts[agent_id]
                    continue

                aid, content, parsed = result
                current_drafts[aid] = normalize_draft_result(
                    parsed,
                    f"Agent did not submit via submit_draft. Response: {truncate_text(content, 180)}",
                )
                chained_drafts[aid] = current_drafts[aid]
                if (
                    isinstance(parsed, dict)
                    and isinstance(parsed.get("draft"), str)
                    and parsed["draft"].strip()
                ):
                    last_successful_round[aid] = 0
                    submitted_rounds[aid].add(0)

            # rounds_data always carries the FULL per-round data (drafts,
            # reviews, revisions with bodies) so the Rich iframe's final
            # embed can render every intermediate draft on reload. The
            # return payload (response_payload) strips bodies when
            # INCLUDE_FULL_HISTORY is off — see _strip_round_bodies below.
            rounds_data.append(
                {
                    "round": 0,
                    "phase": "initial_composition",
                    "drafts": {
                        aid: {
                            "persona": agents[aid]["name"],
                            "model": agent_model[aid],
                            "draft": current_drafts[aid]["draft"],
                            "approach": current_drafts[aid].get("approach", ""),
                            "sources": current_drafts[aid].get("sources", []),
                        }
                        for aid in agent_ids
                    },
                }
            )

            # ---------- Review / Revision Rounds ----------
            for round_num in range(1, num_rounds + 1):
                await emitter.emit(
                    description=f"Round {round_num}/{num_rounds}: Cross-Review",
                    done=False,
                )
                await emitter.push(
                    step=1 + 2 * (round_num - 1),
                    phase=f"Round {round_num}/{num_rounds}: Cross-Review",
                    agent_state={aid: "reviewing 0/2" for aid in agent_ids},
                    # N reviewers each review (N-1) peers — total reviews
                    # this round drives the per-completion sub-step grain.
                    total_substeps=len(agent_ids) * (len(agent_ids) - 1),
                )
                # Drop stale reviews from the previous round so the iframe
                # shows only the current round's feedback while it lands.
                await emitter.clear_reviews()

                all_reviews: dict[str, dict[str, dict]] = {aid: {} for aid in agent_ids}
                review_counts: dict[str, int] = {aid: 0 for aid in agent_ids}
                review_fails: dict[str, int] = {aid: 0 for aid in agent_ids}
                review_counts_lock = asyncio.Lock()

                async def review_draft(
                    reviewer_id: str, author_id: str
                ) -> tuple[str, str, str, dict]:
                    reviewer_persona = agents[reviewer_id]
                    author_persona = agents[author_id]
                    author_draft = current_drafts[author_id]["draft"]
                    # ``review_ok`` flips only after we see a usable
                    # ``key_feedback`` string — run_agent_loop returning an
                    # "API error: ..." content, malformed JSON, or a parsed
                    # dict missing ``key_feedback`` all stay as failure so
                    # the reviewer card shows red instead of done-green.
                    review_ok = False
                    content: str = ""
                    parsed: Any = {}
                    try:
                        _, terminal_prompt, direct_prompts = await ensure_tools(
                            reviewer_id
                        )
                        member_extra_params = make_member_extra_params(
                            reviewer_id, terminal_prompt, direct_prompts
                        )

                        system_prompt = build_review_system_prompt(
                            base_system_prompt,
                            reviewer_persona,
                            criteria=review_criteria,
                        )
                        user_prompt = build_review_user_prompt(
                            topic_text, author_persona["name"], author_draft
                        )

                        # Reviewers use only the submit_review pseudo-tool —
                        # they do not need research tools, and locking them
                        # to the submission channel matches compose/revise's
                        # escape-free tool-call contract.
                        review_capture: dict = {"payload": None}
                        review_tools = {
                            "submit_review": make_submission_tool(
                                spec=build_submit_review_spec(),
                                capture=review_capture,
                            )
                        }

                        content = await run_agent_loop(
                            request=request,
                            user=user,
                            model_id=agent_model[reviewer_id],
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            tools_dict=review_tools,
                            max_iterations=int(self.valves.REVIEW_MAX_ITERATIONS),
                            extra_params=member_extra_params,
                            apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                            agent_name=f"{reviewer_persona['name']}\u2192{author_persona['name']}",
                            event_emitter=emitter.wrap_agent(
                                reviewer_id,
                                f"{reviewer_persona['name']}\u2192{author_persona['name']}",
                            ),
                            iteration_note_role=self.valves.ITERATION_NOTE_ROLE,
                            submission_tool_names={"submit_review"},
                        )

                        # Strict tool-call contract: only the capture from
                        # submit_review is authoritative. The prompt
                        # forbids content-JSON, so we intentionally do
                        # NOT fall back to parsing it here — doing so
                        # would contradict the prompt and re-introduce
                        # the unescaped-quote fragility this migration
                        # eliminated.
                        parsed = extract_submission_payload(
                            review_capture, "key_feedback"
                        )
                        if isinstance(parsed, dict):
                            kf = parsed.get("key_feedback")
                            if isinstance(kf, str) and kf.strip():
                                review_ok = True
                                try:
                                    await emitter.set_review(
                                        author_id=author_id,
                                        reviewer_id=reviewer_id,
                                        reviewer_name=reviewer_persona["name"],
                                        review=parsed,
                                    )
                                except Exception as e:
                                    log.warning(f"[LLMReview] set_review failed: {e}")
                        return reviewer_id, author_id, content, parsed or {}
                    finally:
                        async with review_counts_lock:
                            review_counts[reviewer_id] += 1
                            if not review_ok:
                                review_fails[reviewer_id] += 1
                            attempted = review_counts[reviewer_id]
                            failed = review_fails[reviewer_id]
                            succeeded = attempted - failed
                        if failed:
                            # N/2 tracks successes; the paren tail lets the
                            # JS statusClass pick this up as an error card.
                            state_text = f"reviewed {succeeded}/2 ({failed} failed)"
                        else:
                            state_text = f"reviewed {succeeded}/2"
                        await emitter.tick_substep(
                            agent_state={reviewer_id: state_text}
                        )

                review_tasks = [
                    review_draft(reviewer_id, author_id)
                    for reviewer_id in agent_ids
                    for author_id in agent_ids
                    if reviewer_id != author_id
                ]
                review_results = await asyncio.gather(*review_tasks, return_exceptions=True)

                for result in review_results:
                    if isinstance(result, BaseException):
                        log.exception(f"Review failed: {result}")
                        continue

                    reviewer_id, author_id, content, parsed = result
                    normalized = normalize_review_result(
                        parsed,
                        f"Reviewer did not submit via submit_review. Response: {truncate_text(content, 100)}",
                    )
                    normalized["reviewer"] = agents[reviewer_id]["name"]
                    all_reviews[author_id][reviewer_id] = normalized

                await emitter.emit(
                    description=f"Round {round_num}/{num_rounds}: Revision",
                    done=False,
                )
                await emitter.push(
                    step=2 + 2 * (round_num - 1),
                    phase=f"Round {round_num}/{num_rounds}: Revision",
                    agent_state={aid: "revising" for aid in agent_ids},
                    total_substeps=len(agent_ids),
                )
                # Reset outcome markers so a previous round's success
                # doesn't leak into THIS round's cancellation labelling.
                reset_phase_outcomes()

                async def revise_draft(agent_id: str) -> tuple[str, str, dict]:
                    persona = agents[agent_id]
                    original_draft = current_drafts[agent_id]["draft"]
                    feedbacks = list(all_reviews[agent_id].values())
                    # Same contract as compose_draft: only flip ``state``
                    # to "revised" after we observe a real non-empty
                    # ``draft`` string in the parsed payload. API-error
                    # strings and malformed JSON leave it as "failed".
                    state = "failed"
                    content: str = ""
                    parsed: Any = {}
                    try:
                        tools_dict, terminal_prompt, direct_prompts = await ensure_tools(
                            agent_id
                        )
                        member_extra_params = make_member_extra_params(
                            agent_id, terminal_prompt, direct_prompts
                        )

                        system_prompt = build_revise_system_prompt(
                            base_system_prompt, persona, include_sources
                        )
                        user_prompt = build_revise_user_prompt(
                            topic_text,
                            original_draft,
                            feedbacks,
                            anonymize=anonymize_reviewers,
                        )

                        # Same escape-free submission pattern as compose —
                        # but the spec exposes changes_made / feedback_declined
                        # instead of approach, matching the revise schema.
                        submit_capture: dict = {"payload": None}
                        submit_spec = build_submit_draft_spec(
                            phase="revise", include_sources=include_sources
                        )
                        combined_tools = {
                            **tools_dict,
                            "submit_draft": make_submission_tool(
                                spec=submit_spec, capture=submit_capture
                            ),
                        }

                        content = await run_agent_loop(
                            request=request,
                            user=user,
                            model_id=agent_model[agent_id],
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            tools_dict=combined_tools,
                            max_iterations=self.valves.MAX_ITERATIONS,
                            extra_params=member_extra_params,
                            apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                            agent_name=f"{persona['name']}",
                            event_emitter=emitter.wrap_agent(agent_id, persona["name"]),
                            iteration_note_role=self.valves.ITERATION_NOTE_ROLE,
                            submission_tool_names={"submit_draft"},
                        )

                        # Strict tool-call contract — same rationale as
                        # compose_draft above: the submission capture is
                        # the single source of truth, no content-JSON
                        # fallback. See the compose_draft comment for
                        # the full reasoning.
                        parsed = extract_submission_payload(submit_capture, "draft")
                        if isinstance(parsed, dict):
                            draft_val = parsed.get("draft")
                            if isinstance(draft_val, str) and draft_val.strip():
                                state = "revised"
                                raw_changes = parsed.get("changes_made")
                                raw_declined = parsed.get("feedback_declined")
                                try:
                                    await emitter.set_draft(
                                        agent_id,
                                        draft=draft_val,
                                        phase_label=f"Round {round_num}/{num_rounds} revision",
                                        changes_made=[
                                            s for s in raw_changes if isinstance(s, str) and s.strip()
                                        ]
                                        if isinstance(raw_changes, list)
                                        else None,
                                        feedback_declined=[
                                            s for s in raw_declined if isinstance(s, str) and s.strip()
                                        ]
                                        if isinstance(raw_declined, list)
                                        else None,
                                    )
                                except Exception as e:
                                    log.warning(f"[LLMReview] set_draft failed: {e}")
                                # Commit now so a mid-gather cancel keeps
                                # the revision the iframe just displayed.
                                # Mirrors the post-gather merge below:
                                # prior sources + new sources, deduped by
                                # (title, url). Approach is preserved from
                                # the compose phase.
                                revise_out = normalize_revise_result(
                                    parsed, original_draft
                                )
                                prior = current_drafts.get(agent_id) or {}
                                prior_sources = list(prior.get("sources") or [])
                                new_sources = list(revise_out.get("sources") or [])
                                seen: set = set()
                                merged_sources: list[dict] = []
                                for src in prior_sources + new_sources:
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
                                current_drafts[agent_id] = {
                                    "draft": revise_out["draft"],
                                    "approach": prior.get("approach", ""),
                                    "sources": merged_sources,
                                }
                                # Inline so a mid-gather cancel still
                                # sees this agent as having finished
                                # round N; outer aggregation re-assigns,
                                # idempotent.
                                submitted_rounds[agent_id].add(round_num)
                                if advance_last_successful_round_if_chained(
                                    last_successful_round, agent_id, round_num
                                ):
                                    chained_drafts[agent_id] = current_drafts[
                                        agent_id
                                    ]
                                per_agent_last_phase_outcome[agent_id] = "success"
                        # Reached here without a valid submission (and
                        # without being cancelled — CancelledError would
                        # skip this block). Mark a clean failure so the
                        # cancel-time renderer can distinguish "this
                        # agent's round actually failed" from "this agent
                        # was cut off mid-round".
                        if per_agent_last_phase_outcome[agent_id] == "unknown":
                            per_agent_last_phase_outcome[agent_id] = "failed"
                        return agent_id, content, parsed or {}
                    except asyncio.CancelledError:
                        # Genuine in-flight cancellation: leave outcome
                        # at "unknown" so the renderer uses the
                        # "cancelled" label branch. Re-raise so the
                        # outer cancel finalize path runs.
                        raise
                    except BaseException:
                        # Non-cancel exception (e.g. ensure_tools crash,
                        # transport blowup) before the success/failure
                        # inline setters fired. Mark the outcome as
                        # "failed" so the cancel-time renderer doesn't
                        # mislabel a real tool failure as cancellation
                        # just because the user pressed stop shortly
                        # after the exception propagated.
                        if per_agent_last_phase_outcome[agent_id] == "unknown":
                            per_agent_last_phase_outcome[agent_id] = "failed"
                        raise
                    finally:
                        await emitter.tick_substep(agent_state={agent_id: state})

                revise_results = await asyncio.gather(
                    *[revise_draft(aid) for aid in agent_ids], return_exceptions=True
                )

                revised_drafts: dict[str, dict] = {}
                revision_succeeded: dict[str, bool] = {}
                for idx, result in enumerate(revise_results):
                    agent_id = agent_ids[idx]
                    if isinstance(result, BaseException):
                        log.exception(f"Revision failed ({agent_id}): {result}")
                        revised_drafts[agent_id] = normalize_revise_result(
                            None,
                            f"Revision failed ({agents[agent_id]['name']}): "
                            f"{truncate_text(str(result), 120)}",
                        )
                        revision_succeeded[agent_id] = False
                        continue

                    aid, content, parsed = result
                    if (
                        isinstance(parsed, dict)
                        and isinstance(parsed.get("draft"), str)
                        and parsed["draft"].strip()
                    ):
                        revised_drafts[aid] = normalize_revise_result(parsed, "")
                        revision_succeeded[aid] = True
                    else:
                        revised_drafts[aid] = normalize_revise_result(
                            None,
                            f"Agent did not submit via submit_draft. "
                            f"Response: {truncate_text(content, 180)}",
                        )
                        revision_succeeded[aid] = False

                # Failed revisions keep the previous current_drafts so the
                # next round's cross-review has real prose to evaluate
                # instead of cascading the error string. Sources are
                # merged across rounds by (title, url).
                for aid in agent_ids:
                    if not revision_succeeded.get(aid, False):
                        continue
                    submitted_rounds[aid].add(round_num)
                    chain_advanced = advance_last_successful_round_if_chained(
                        last_successful_round, aid, round_num
                    )
                    prior_sources = current_drafts[aid].get("sources", []) or []
                    new_sources = revised_drafts[aid].get("sources") or []
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
                    current_drafts[aid] = {
                        "draft": revised_drafts[aid]["draft"],
                        "approach": current_drafts[aid].get("approach", ""),
                        "sources": merged_sources,
                    }
                    if chain_advanced:
                        chained_drafts[aid] = current_drafts[aid]

                rounds_data.append(
                    {
                        "round": round_num,
                        "reviews": {
                            aid: {
                                reviewer_id: {
                                    "reviewer": review["reviewer"],
                                    "key_feedback": review["key_feedback"],
                                    "strengths": review["strengths"],
                                    "improvements": review["improvements"],
                                }
                                for reviewer_id, review in all_reviews[aid].items()
                            }
                            for aid in agent_ids
                        },
                        "revisions": {
                            aid: {
                                "persona": agents[aid]["name"],
                                "model": agent_model[aid],
                                "draft": revised_drafts[aid]["draft"],
                                "changes_made": revised_drafts[aid].get(
                                    "changes_made", []
                                ),
                                "feedback_declined": revised_drafts[aid].get(
                                    "feedback_declined", []
                                ),
                                "sources": current_drafts[aid].get(
                                    "sources", []
                                ),
                            }
                            for aid in agent_ids
                        },
                    }
                )

                # Round complete: park all agents in a waiting state until the
                # next round (or finalize) flips them. Last round flows into
                # finalize without re-parking.
                if round_num < num_rounds:
                    await emitter.push(
                        agent_state={aid: "waiting" for aid in agent_ids}
                    )

            await emitter.push(
                step=total_steps,
                phase="Complete",
                detail=None,
                agent_state={aid: "done" for aid in agent_ids},
            )
            await emitter.emit(
                description="LLM Review Complete",
                done=True,
            )
            # Body comes from chained_drafts so it matches
            # last_successful_round (current_drafts may carry a
            # post-break "success" body that contradicts the badge).
            final_drafts = {
                aid: {
                    "persona": agents[aid]["name"],
                    "model": agent_model[aid],
                    "draft": chained_drafts.get(aid, current_drafts[aid])[
                        "draft"
                    ],
                    "sources": chained_drafts.get(
                        aid, current_drafts[aid]
                    ).get("sources", []),
                    "last_successful_round": last_successful_round[aid],
                    "final_round_completed": (
                        last_successful_round[aid] == num_rounds
                    ),
                    "submitted_rounds": sorted(submitted_rounds[aid]),
                    "last_phase_outcome": per_agent_last_phase_outcome[aid],
                }
                for aid in agent_ids
            }

            # Ship final drafts + full round history into the persistent embed
            # so a user reloading the page can still see every draft, review,
            # and revision from the completed run — not just "Complete".
            # ``agents``/``model_ids`` here are keyed by agent_id for iframe
            # iteration, while each entry also carries the real underlying
            # ``model`` id for display (same model used twice shows up twice).
            await emitter.finalize(
                summary={
                    "topic": topic_text,
                    "num_rounds": num_rounds,
                    "agents": [
                        {
                            "id": aid,
                            "model": agent_model[aid],
                            "persona": agents[aid]["name"],
                        }
                        for aid in agent_ids
                    ],
                    "model_ids": list(agent_ids),
                    "final_drafts": final_drafts,
                    "rounds": rounds_data,
                    "render_markdown": bool(
                        getattr(user_valves, "RENDER_DRAFTS_AS_MARKDOWN", True)
                    ),
                    "force_theme": getattr(
                        user_valves, "RICH_PROGRESS_THEME", "auto"
                    ) or "auto",
                    "elapsed_seconds": (
                        asyncio.get_event_loop().time() - started_at
                    ),
                }
            )

            # Strip heavy per-round draft bodies for the return payload
            # (unless the user opted into INCLUDE_FULL_HISTORY). The
            # full-body rounds_data has already been shipped to
            # finalize() for the Rich iframe, which always keeps
            # everything regardless of this valve.
            if include_full_history:
                return_rounds = rounds_data
            else:
                return_rounds = []
                for r in rounds_data:
                    stripped = dict(r)
                    if stripped.get("phase") == "initial_composition":
                        # Keep only approaches, drop full drafts.
                        stripped.pop("drafts", None)
                        stripped["approaches"] = {
                            aid: (
                                stripped.get("drafts", {})
                                .get(aid, {})
                                .get("approach", "")
                            )
                            or r.get("drafts", {}).get(aid, {}).get("approach", "")
                            for aid in agent_ids
                            if r.get("drafts", {}).get(aid, {}).get("approach")
                        }
                    elif "revisions" in stripped:
                        stripped["revisions"] = {
                            aid: {
                                "changes_made": rev.get("changes_made", []),
                                "feedback_declined": rev.get(
                                    "feedback_declined", []
                                ),
                            }
                            for aid, rev in stripped["revisions"].items()
                        }
                    return_rounds.append(stripped)

            # Return-shape notes (kept out of the docstring to avoid bloating
            # the LLM tool spec — the calling LLM can read the JSON directly):
            # - final_drafts: canonical "latest per-persona output". Read this
            #   for the answer; in strip mode it is the ONLY place the final
            #   draft body lives (the last round's revisions drop ``draft``).
            #   Each entry carries ``last_successful_round`` (0 = compose,
            #   N = round N revision) and ``final_round_completed`` so the
            #   caller can tell whether the draft actually made it through
            #   the full revision pipeline.
            # - rounds: heterogeneous audit trail. Round 0 carries composition
            #   data — full ``drafts`` in full-history mode, only
            #   ``approaches`` when stripped. Rounds 1+ carry ``reviews`` plus
            #   ``revisions`` (which lose the ``draft`` body when
            #   INCLUDE_FULL_HISTORY is off — see strip block above).
            # - revisions_complete: top-level boolean, ``True`` iff every
            #   agent's last-successful-round equals ``num_rounds``. A
            #   single-field short-circuit for LLM callers that only need
            #   "did everything land?" without iterating final_drafts.
            response_payload = {
                "topic": topic_text,
                "num_rounds": num_rounds,
                "final_drafts": final_drafts,
                "rounds": return_rounds,
                "revisions_complete": all(
                    last_successful_round[aid] == num_rounds
                    for aid in agent_ids
                ),
            }

            return json.dumps(response_payload, ensure_ascii=False)
        except asyncio.CancelledError:
            # User pressed the chat's stop button (or the request was
            # otherwise aborted). Finalize with whatever partial state
            # we have so the iframe flips to a "Cancelled" view instead
            # of leaving a frozen progress bar behind; also close out
            # the status line so the global spinner stops. Wrapped in a
            # broad try/except because any failure here must not prevent
            # the CancelledError from propagating — MCP cleanup still
            # has to run in the outer finally.
            try:
                # Mirror the successful-path final_drafts shape down to the
                # ``last_successful_round`` / ``final_round_completed``
                # status fields. Without these the cancellation finalize
                # would render the partial drafts with NO indication that
                # the revision pipeline was interrupted mid-flight — the
                # user presses stop partway through round 2, and the
                # iframe happily shows the round-1 draft as if it were
                # the intended final output. ``last_successful_round``
                # naturally captures "how far we got before cancellation"
                # because it is only advanced on phase success.
                # Mid-cancel both maps may be partial; chained_drafts
                # is preferred but current_drafts catches in-flight
                # bodies the cancel cut short of mirroring.
                partial_final_drafts = {
                    aid: {
                        "persona": agents[aid]["name"],
                        "model": agent_model[aid],
                        "draft": chained_drafts.get(
                            aid, current_drafts.get(aid, {})
                        ).get("draft", ""),
                        "sources": chained_drafts.get(
                            aid, current_drafts.get(aid, {})
                        ).get("sources", []),
                        "last_successful_round": last_successful_round[aid],
                        "final_round_completed": (
                            last_successful_round[aid] == num_rounds
                        ),
                        "submitted_rounds": sorted(submitted_rounds[aid]),
                        "last_phase_outcome": per_agent_last_phase_outcome[aid],
                    }
                    for aid in agent_ids
                    if (
                        isinstance(chained_drafts.get(aid), dict)
                        and chained_drafts[aid].get("draft")
                    )
                    or (
                        isinstance(current_drafts.get(aid), dict)
                        and current_drafts[aid].get("draft")
                    )
                }
                cancel_summary = {
                    "topic": topic_text,
                    "num_rounds": num_rounds,
                    "agents": [
                        {
                            "id": aid,
                            "model": agent_model[aid],
                            "persona": agents[aid]["name"],
                        }
                        for aid in agent_ids
                    ],
                    "model_ids": list(agent_ids),
                    "final_drafts": partial_final_drafts,
                    "rounds": rounds_data,
                    "render_markdown": bool(
                        getattr(user_valves, "RENDER_DRAFTS_AS_MARKDOWN", True)
                    ),
                    "force_theme": getattr(
                        user_valves, "RICH_PROGRESS_THEME", "auto"
                    ) or "auto",
                    "elapsed_seconds": (
                        asyncio.get_event_loop().time() - started_at
                    ),
                    "cancelled": True,
                }
                await emitter.finalize(summary=cancel_summary)
                await emitter.emit(
                    description="LLM Review cancelled",
                    done=True,
                )
            except BaseException as cleanup_exc:
                log.warning(
                    f"[LLMReview] Cancel finalize failed: {cleanup_exc}"
                )
            raise
        finally:
            for mcp_clients in all_mcp_clients:
                await cleanup_mcp_clients(mcp_clients)
