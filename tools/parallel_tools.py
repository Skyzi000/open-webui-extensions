"""
title: Parallel Tools
author: skyzi000
version: 0.1.1
license: MIT
required_open_webui_version: 0.7.0
description: Execute multiple independent tool calls in parallel for faster results.

Open WebUI executes tool calls sequentially, which can be slow when multiple
independent tools need to run. This tool allows you to batch independent tool
calls and execute them concurrently using asyncio.gather.

Tip: Combine with the Sub Agent tool to run multiple sub-agents in parallel.
https://github.com/Skyzi000/open-webui-extensions/blob/main/tools/sub_agent.py
For example, you can spawn several research sub-agents simultaneously,
each investigating a different topic, and get all results at once.

Watch out for rate limits on web search APIs, LLM providers, etc.!
"""

import asyncio
import ast
import json
import logging
import uuid
from typing import Any, Callable, List, Optional

from fastapi import Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Tools that generate citation sources
# NOTE: Update this set when new citation-capable tools are added to Open WebUI.
CITATION_TOOLS = {"search_web", "view_knowledge_file", "query_knowledge_files"}


# ============================================================================
# Helper functions (outside class - AI cannot invoke these)
# ============================================================================


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

        tool_function = tool["callable"]

        # Update function with current messages/files context
        from open_webui.utils.tools import get_updated_tool_function

        tool_function = get_updated_tool_function(
            function=tool_function,
            extra_params={
                "__messages__": extra_params.get("__messages__", []),
                "__files__": extra_params.get("__files__", []),
            },
        )

        result = await tool_function(**filtered_args)

        # Handle OpenAPI/external tool results that return (data, headers) tuple
        # Headers (CIMultiDictProxy) are not JSON-serializable, so extract just the data
        tool_type = tool.get("type", "")
        if tool_type == "external" and isinstance(result, tuple) and len(result) == 2:
            result = result[0]  # Extract data, discard headers

        # Keep result as-is for proper JSON serialization (avoid double-encoding)
        # If result is already a string, it stays a string
        # If result is a dict/list, it will be serialized once at the end

        # Extract and emit citation sources for tools that generate them
        if event_emitter and result and tool_name in CITATION_TOOLS:
            # For citation extraction, we need a string representation
            result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            try:
                from open_webui.utils.middleware import get_citation_source_from_tool_result

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
            description="Comma-separated list of tool IDs available for parallel execution. Leave empty to use all tools.",
        )
        EXCLUDED_TOOL_IDS: str = Field(
            default="",
            description="Comma-separated list of tool IDs to exclude from parallel execution.",
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
        tool_calls: str,
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
        Execute multiple tool calls in parallel to avoid slow sequential execution.

        CRITICAL: Open WebUI executes tool calls sequentially by default. When you call
        multiple tools separately, each one waits for the previous to complete, causing
        significant delays. The user has enabled this tool specifically to solve this problem.

        ALWAYS USE THIS TOOL when you need to call 2+ independent tools. Batch them together
        in a single call to run concurrently. This dramatically reduces wait time.

        Example scenario:
        - BAD: Call search_web("Python"), wait, call search_web("FastAPI"), wait → slow
        - GOOD: Call run_tools_parallel with both searches → both run at the same time

        :param tool_calls: JSON array of tool calls. Each item must have "name" (tool name) and "arguments" (dict of args).
                          Example: [{"name": "search_web", "arguments": {"query": "Python"}}, {"name": "search_web", "arguments": {"query": "FastAPI"}}]
        :return: JSON object with results for each tool call

        Note: Only batch tools that don't depend on each other's results.
        """
        if __request__ is None:
            return json.dumps({"error": "Request context not available."})

        if __user__ is None:
            return json.dumps({"error": "User context not available."})

        # Parse tool_calls (may already be parsed by Open WebUI)
        if isinstance(tool_calls, list):
            calls = tool_calls
        elif isinstance(tool_calls, str):
            try:
                calls = ast.literal_eval(tool_calls)
            except Exception:
                try:
                    calls = json.loads(tool_calls)
                except Exception as e:
                    return json.dumps({
                        "error": f"Failed to parse tool_calls: {e}",
                        "expected_format": '[{"name": "tool_name", "arguments": {"arg1": "value1"}}]',
                    })
        else:
            return json.dumps({
                "error": f"tool_calls must be a list or JSON string, got {type(tool_calls).__name__}",
            })

        if not isinstance(calls, list):
            return json.dumps({
                "error": "tool_calls must be a JSON array",
                "expected_format": '[{"name": "tool_name", "arguments": {"arg1": "value1"}}]',
            })

        if not calls:
            return json.dumps({"error": "tool_calls array is empty"})

        # Validate each call
        for i, call in enumerate(calls):
            if not isinstance(call, dict):
                return json.dumps({"error": f"tool_calls[{i}] must be an object"})
            if "name" not in call:
                return json.dumps({"error": f"tool_calls[{i}] missing 'name' field"})
            if "arguments" not in call:
                calls[i]["arguments"] = {}

        # Import here to avoid issues when not running in Open WebUI
        from open_webui.models.users import UserModel
        from open_webui.utils.tools import get_tools

        user = UserModel(**__user__)

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

        # Load builtin tools
        try:
            from open_webui.utils.tools import get_builtin_tools

            features = __metadata__.get("features", {}) if __metadata__ else {}
            model = __model__ or {}

            builtin_extra_params = {
                "__user__": __user__,
                "__event_emitter__": __event_emitter__,
                "__chat_id__": __chat_id__,
                "__message_id__": __message_id__,
                "__oauth_token__": __oauth_token__,
            }

            all_builtin_tools = get_builtin_tools(
                request=__request__,
                extra_params=builtin_extra_params,
                features=features,
                model=model,
            )

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
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "tool_name": calls[i].get("name", "unknown"),
                    "result": f"Error: {result}",
                })
            else:
                processed_results.append(result)

        return json.dumps(
            {"results": processed_results},
            ensure_ascii=False,
        )
