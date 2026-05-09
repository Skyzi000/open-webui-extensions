"""Canonical agent-tools loader shared by ``llm_review`` / ``sub_agent`` /
``multi_model_council``.

All three plugins need to assemble the ``tools_dict`` that their
spawned agent loop will hand to ``get_updated_tool_function`` — and
they all need the same five tool sources merged in the same priority
order:

1. **Regular tools** loaded by ``open_webui.utils.tools.get_tools()``
   (filtering out ``builtin:`` and ``server:mcp:`` IDs first because
   ``get_tools`` silently skips them).
2. **MCP tools** resolved via ``shared.mcp_tools.resolve_mcp_tools``
   for any ``server:mcp:`` IDs the user has selected.
3. **Terminal tools** from ``get_terminal_tools()`` when the user has
   bound a terminal in chat metadata and ``ENABLE_TERMINAL_TOOLS`` is
   on.
4. **Direct tools** from ``metadata.tool_servers`` (OpenAPI servers).
5. **Builtin tools** from ``get_builtin_tools()``, filtered by the
   plugin's per-category Valves (``ENABLE_*_TOOLS``) plus the
   "keep ``view_note`` for knowledge" carve-out.

Each merge step warns on duplicate names (debug-only) and surfaces
load failures to the user via ``shared.notifications.emit_notification``.

Caller responsibilities (kept outside this helper):

- Parse ``AVAILABLE_TOOL_IDS`` / ``EXCLUDED_TOOL_IDS`` valves into the
  ``tool_id_list`` / ``excluded_tool_ids`` arguments.
- Add the plugin's own tool ID to ``excluded_tool_ids`` to prevent the
  agent loop from recursing into the plugin that spawned it.
- After the agent loop ends, call
  ``shared.mcp_tools.cleanup_mcp_clients`` on the returned
  ``mcp_clients`` mapping so MCP connections don't leak.

Top-level imports are cross-shared only; ``logging`` and the
``open_webui.*`` helpers are imported lazily inside the function so
this module satisfies the inliner's "no mixing external + cross-shared
imports at top level" rule for shared deps.
"""

from owui_ext.shared.async_utils import maybe_await
from owui_ext.shared.builtin_tools import BUILTIN_TOOL_CATEGORIES, VALVE_TO_CATEGORY
from owui_ext.shared.mcp_tools import resolve_mcp_tools
from owui_ext.shared.model_features import (
    model_has_note_knowledge,
    model_knowledge_tools_enabled,
)
from owui_ext.shared.notifications import emit_notification
from owui_ext.shared.tool_execution import normalize_terminal_tools_result
from owui_ext.shared.tool_servers import (
    build_direct_tools_dict,
    extract_direct_tool_server_prompts,
    normalize_direct_tool_servers,
    resolve_direct_tool_servers_from_request_and_metadata,
    resolve_terminal_id_from_request_and_metadata,
)


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
