"""
title: Parallel Tools
author: skyzi000
version: 0.1.13
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

import asyncio
import ast
import json
import logging
import uuid
from typing import Any, Callable, List, Optional

from fastapi import Request
from pydantic import BaseModel, Field

from owui_ext.shared.async_utils import maybe_await
from owui_ext.shared.tool_event_metadata import CITATION_TOOLS, TERMINAL_EVENT_TOOLS

log = logging.getLogger(__name__)


class ToolCallItem(BaseModel):
    """A single tool call specification."""

    name: str = Field(description="Tool function name to call")
    args: dict = Field(default_factory=dict, description="Arguments to pass to the tool")


_core_process_tool_result = None


# ============================================================================
# Helper functions (outside class - AI cannot invoke these)
# ============================================================================


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
                "[ParallelTools] terminal_id mismatch between request body and metadata; "
                "using request body terminal_id"
            )
        return request_terminal_id

    return metadata_terminal_id


def normalize_direct_tool_servers(value: Any) -> List[dict]:
    """Normalize direct tool server payload into a list of dict copies."""
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


async def resolve_direct_tool_servers_from_request_and_metadata(
    *,
    request: Optional[Request],
    metadata: Optional[dict],
    debug: bool = False,
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
                "[ParallelTools] tool_servers mismatch between request body and metadata; "
                "using request body tool_servers"
            )
        return request_servers

    return metadata_servers


def build_direct_tools_dict(*, tool_servers: List[dict]) -> dict:
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
    return direct_tools


async def execute_direct_tool_call(
    *,
    tool_name: str,
    tool_args: dict,
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
                "name": tool_name,
                "params": tool_args,
                "server": tool.get("server", {}),
                "session_id": session_id,
            },
        }
    )


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
                tool_name=tool_name,
                tool_args=filtered_args,
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
                },
            ))

            result = await tool_function(**filtered_args)

        # Handle OpenAPI/external/direct tool results that return (data, headers)
        tool_type = tool.get("type", "")
        result, tool_result_files, tool_result_embeds = await process_tool_result(
            tool_function_name=tool_name,
            tool_type=tool_type,
            tool_result=result,
            direct_tool=direct_tool,
            request=extra_params.get("__request__"),
            metadata=extra_params.get("__metadata__"),
            user=extra_params.get("__user__"),
        )

        # Emit terminal:* events for display/refresh behavior in UI
        await emit_terminal_tool_event(
            tool_name=tool_name,
            tool_args=filtered_args,
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


async def emit_terminal_tool_event(
    *,
    tool_name: str,
    tool_args: dict,
    tool_result: Any,
    event_emitter: Optional[Callable],
) -> None:
    """Emit terminal:* UI events for Open Terminal tool results."""
    if not event_emitter or tool_name not in TERMINAL_EVENT_TOOLS:
        return

    if tool_name == "display_file":
        path = tool_args.get("path", "") if isinstance(tool_args, dict) else ""
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
    elif tool_name in {"write_file", "replace_file_content"}:
        path = tool_args.get("path", "") if isinstance(tool_args, dict) else ""
        if not isinstance(path, str) or not path:
            return
        event = {"type": f"terminal:{tool_name}", "data": {"path": path}}
    elif tool_name == "run_command":
        event = {"type": "terminal:run_command", "data": {}}
    else:
        return

    try:
        await event_emitter(event)
    except Exception as e:
        log.warning(f"Error emitting terminal event for {tool_name}: {e}")


# ============================================================================
# Tools class
# ============================================================================


class Tools:
    """Parallel tool execution for independent operations."""

    class Valves(BaseModel):
        AVAILABLE_TOOL_IDS: str = Field(
            default="",
            description="Comma-separated list of tool IDs available for parallel execution. Leave empty to use all tools.",
        )
        EXCLUDED_TOOL_IDS: str = Field(
            default="",
            description="Comma-separated list of tool IDs to exclude from parallel execution.",
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
    ) -> str:
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
        from open_webui.utils.tools import get_tools

        user = UserModel(**__user__)
        __metadata__ = __metadata__ or {}

        terminal_id = await resolve_terminal_id_from_request_and_metadata(
            request=__request__,
            metadata=__metadata__,
            debug=self.valves.DEBUG,
        )
        if terminal_id:
            __metadata__["terminal_id"] = terminal_id

        direct_tool_servers = await resolve_direct_tool_servers_from_request_and_metadata(
            request=__request__,
            metadata=__metadata__,
            debug=self.valves.DEBUG,
        )
        if direct_tool_servers:
            __metadata__["tool_servers"] = direct_tool_servers

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
            "__files__": __metadata__.get("files", []) if __metadata__ else [],
        }

        # Determine which tools to use
        available_tool_ids = []
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

        tool_id_list = [tid for tid in tool_id_list if tid not in excluded]

        # Filter out builtin: prefixed IDs
        regular_tool_ids = [
            tid for tid in tool_id_list if not tid.startswith("builtin:")
        ]

        if self.valves.DEBUG:
            log.info(f"[ParallelTools] Tool IDs: {regular_tool_ids}")

        # Load tools
        tools_dict = {}
        if regular_tool_ids:
            try:
                tools_dict = await get_tools(
                    request=__request__,
                    tool_ids=regular_tool_ids,
                    user=user,
                    extra_params=extra_params,
                )

                if self.valves.DEBUG:
                    log.info(f"[ParallelTools] Loaded {len(tools_dict)} tools")

            except Exception as e:
                log.exception(f"Error loading tools: {e}")
                return json.dumps({"error": f"Failed to load tools: {e}"})

        # Load terminal tools
        if terminal_id and self.valves.ENABLE_TERMINAL_TOOLS:
            try:
                from open_webui.utils.tools import get_terminal_tools

                terminal_tools_result = await get_terminal_tools(
                    request=__request__,
                    terminal_id=terminal_id,
                    user=user,
                    extra_params=extra_params,
                )
                terminal_tools = normalize_terminal_tools_result(
                    terminal_tools_result=terminal_tools_result,
                    extra_params=extra_params,
                )
                if terminal_tools:
                    tools_dict = {**tools_dict, **terminal_tools}
                    if self.valves.DEBUG:
                        log.info(
                            f"[ParallelTools] Loaded {len(terminal_tools)} terminal tools for terminal_id={terminal_id}"
                        )
            except Exception as e:
                log.exception(f"Error loading terminal tools: {e}")

        # Load direct tools from metadata.tool_servers
        if direct_tool_servers:
            try:
                direct_tools = build_direct_tools_dict(tool_servers=direct_tool_servers)
                if direct_tools:
                    tools_dict = {**tools_dict, **direct_tools}
                    if self.valves.DEBUG:
                        log.info(f"[ParallelTools] Loaded {len(direct_tools)} direct tools")
            except Exception as e:
                log.exception(f"Error loading direct tools: {e}")

        # Load builtin tools
        try:
            from open_webui.utils.tools import get_builtin_tools

            features = __metadata__.get("features", {}) if __metadata__ else {}
            model = __model__ or {}

            builtin_extra_params = {
                "__user__": __user__,
                "__event_emitter__": __event_emitter__,
                "__event_call__": __event_call__,
                "__metadata__": __metadata__,
                "__chat_id__": __chat_id__,
                "__message_id__": __message_id__,
                "__oauth_token__": __oauth_token__,
            }

            all_builtin_tools = await maybe_await(get_builtin_tools(
                request=__request__,
                extra_params=builtin_extra_params,
                features=features,
                model=model,
            ))

            # Add builtin tools (regular tools take priority)
            for name, tool_dict in all_builtin_tools.items():
                if name not in tools_dict:
                    tools_dict[name] = tool_dict

            if self.valves.DEBUG:
                log.info(f"[ParallelTools] Total tools available: {len(tools_dict)}")

        except Exception as e:
            log.exception(f"Error loading builtin tools: {e}")

        # Prepare extra params for execution
        exec_extra_params = {
            "__user__": __user__,
            "__request__": __request__,
            "__model__": __model__,
            "__metadata__": __metadata__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
            "__oauth_token__": __oauth_token__,
            "__messages__": __messages__ or [],
            "__files__": __metadata__.get("files", []) if __metadata__ else [],
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
            await __event_emitter__({"type": "files", "data": {"files": aggregated_files}})
        if __event_emitter__ and aggregated_embeds:
            await __event_emitter__({"type": "embeds", "data": {"embeds": aggregated_embeds}})

        return json.dumps(
            {"results": processed_results},
            ensure_ascii=False,
        )
