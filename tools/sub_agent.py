"""
title: Sub Agent
author: skyzi000
version: 0.3.3
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
  (Model settings > Parameters > Function Calling: native)

Inspired by VS Code's runSubagent functionality, this tool was developed from scratch specifically for Open WebUI to ensure seamless integration and optimal performance.

💡 Since v0.3, this tool natively supports parallel sub-agent execution via
   run_parallel_sub_agents — no need for "Parallel Tools" anymore!
   If parallel execution causes issues (e.g., search API rate limits),
   reduce MAX_PARALLEL_AGENTS in Valves, or comment out the
   run_parallel_sub_agents method to disable it entirely.
"""

import asyncio
import ast
import json
import logging
import uuid
from typing import Any, Callable, List, Optional, Type

from fastapi import Request
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Builtin tool categories
# NOTE: This mapping must be updated when Open WebUI adds/removes builtin tools.
# Check open_webui/utils/tools.py:get_builtin_tools() for the current list.
BUILTIN_TOOL_CATEGORIES = {
    "time": {"get_current_timestamp", "calculate_timestamp"},
    "web": {"search_web", "fetch_url"},
    "image": {"generate_image", "edit_image"},
    "knowledge": {
        "list_knowledge_bases",
        "search_knowledge_bases",
        "query_knowledge_bases",
        "search_knowledge_files",
        "query_knowledge_files",
        "view_knowledge_file",
    },
    "chat": {"search_chats", "view_chat"},
    "memory": {"search_memories", "add_memory", "replace_memory_content"},
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
}

# Tools that generate citation sources
# NOTE: Update this set when new citation-capable tools are added to Open WebUI.
# See open_webui/utils/middleware.py:get_citation_source_from_tool_result() for supported tools.
CITATION_TOOLS = {"search_web", "view_knowledge_file", "query_knowledge_files"}


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
    tool_call_id = tool_call.get("id", str(uuid.uuid4()))
    tool_function_name = tool_call.get("function", {}).get("name", "")
    tool_args_str = tool_call.get("function", {}).get("arguments", "{}")

    # Parse arguments
    tool_function_params = {}
    try:
        tool_function_params = ast.literal_eval(tool_args_str)
    except Exception:
        try:
            tool_function_params = json.loads(tool_args_str)
        except Exception as e:
            log.error(f"Error parsing tool call arguments: {tool_args_str} - {e}")
            return {
                "tool_call_id": tool_call_id,
                "content": f"Error parsing arguments: {e}",
            }

    # Execute the tool
    tool_result = None
    if tool_function_name in tools_dict:
        tool = tools_dict[tool_function_name]
        spec = tool.get("spec", {})

        try:
            allowed_params = spec.get("parameters", {}).get("properties", {}).keys()
            tool_function_params = {
                k: v for k, v in tool_function_params.items() if k in allowed_params
            }

            tool_function = tool["callable"]

            # Update function with current context
            # __event_emitter__ is included so parallel sub-agents can
            # override it with a task-specific indexed emitter.
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
            if tool_type == "external" and isinstance(tool_result, tuple) and len(tool_result) == 2:
                tool_result = tool_result[0]  # Extract data, discard headers

        except Exception as e:
            log.exception(f"Error executing tool {tool_function_name}: {e}")
            tool_result = f"Error: {e}"
    else:
        tool_result = f"Tool '{tool_function_name}' not found"

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

        # Add iteration info as a system message for context
        messages_with_context = current_messages + [
            {"role": "system", "content": iteration_info}
        ]

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

        # Apply inlet filters if enabled
        form_data = await apply_inlet_filters_if_enabled(
            apply_inlet_filters, request, model, form_data, extra_params
        )

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
                error_msg = error_data.get("error", {}).get("message", str(error_data))
                return f"API error: {error_msg}"
            except Exception:
                return f"API error (status {response.status_code}): Failed to parse response"

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not choices:
                return "No response from model"

            message = choices[0].get("message", {})
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

            # Emit status with tool calls summary
            if event_emitter:
                tool_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[Step {iteration}] Tool calls: {', '.join(tool_names)}",
                            "done": False,
                        },
                    }
                )

            # Add assistant message with tool calls to conversation
            current_messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
            )

            # Execute each tool call
            for tool_call in tool_calls:
                tool_args = tool_call.get("function", {}).get("arguments", "{}")

                if event_emitter:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[Step {iteration}] Args: {tool_args.replace(chr(10), ' ')}",
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

    # Apply inlet filters if enabled
    form_data = await apply_inlet_filters_if_enabled(
        apply_inlet_filters, request, model, form_data, extra_params
    )

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
                return choices[0].get("message", {}).get("content", "")
    except Exception as e:
        log.exception(f"Error getting final response: {e}")

    return "Sub-agent reached maximum iterations without providing a final response."


async def load_sub_agent_tools(
    request: Request,
    user: Any,
    valves: Any,
    metadata: dict,
    model: dict,
    extra_params: dict,
    self_tool_id: Optional[str],
) -> dict:
    """Load regular + builtin tools for sub-agent, returns tools_dict."""
    from open_webui.utils.tools import get_builtin_tools, get_tools

    metadata = metadata or {}
    model = model or {}
    extra_params = extra_params or {}
    event_emitter = extra_params.get("__event_emitter__")

    # Determine tool IDs
    # Use metadata's tool_ids (main conversation's available tools)
    available_tool_ids = []
    if metadata.get("tool_ids"):
        available_tool_ids = list(metadata.get("tool_ids", []))

    if valves.DEBUG:
        log.info(f"[SubAgent] AVAILABLE_TOOL_IDS valve: '{valves.AVAILABLE_TOOL_IDS}'")
        log.info(f"[SubAgent] Available tool_ids from metadata: {available_tool_ids}")
        log.info(f"[SubAgent] self_tool_id: {self_tool_id}")

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

    # Apply exclusions
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

    tool_id_list = [tid for tid in tool_id_list if tid not in excluded]

    # Note: Builtin tools are NOT included in tool_ids.
    # They are loaded separately via get_builtin_tools() and controlled by Valves.
    # Regular tools only (filter out any builtin: prefixed IDs just in case)
    regular_tool_ids = [tid for tid in tool_id_list if not tid.startswith("builtin:")]

    if valves.DEBUG:
        log.info(f"[SubAgent] Regular tool IDs: {regular_tool_ids}")

    # Load regular tools
    tools_dict = {}
    if regular_tool_ids:
        try:
            tools_dict = await get_tools(
                request=request,
                tool_ids=regular_tool_ids,
                user=user,
                extra_params=extra_params,
            )

            if valves.DEBUG:
                log.info(f"[SubAgent] Loaded {len(tools_dict)} regular tools")

        except Exception as e:
            log.exception(f"Error loading tools: {e}")
            if event_emitter:
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Warning: Could not load tools: {e}",
                            "done": False,
                        },
                    }
                )

    # Load builtin tools
    try:
        # Get features from metadata (for memory, etc.)
        features = metadata.get("features", {})

        builtin_extra_params = {
            "__user__": extra_params.get("__user__"),
            "__event_emitter__": event_emitter,
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

        # Build set of disabled tools based on Valves
        disabled_builtin_tools = set()
        for valve_field, category in VALVE_TO_CATEGORY.items():
            if not getattr(valves, valve_field):
                disabled_builtin_tools.update(BUILTIN_TOOL_CATEGORIES[category])

        # Filter builtin tools based on Valves settings
        # NOTE: Regular tools take priority over builtin tools with the same name.
        # This allows users to override builtin behavior with custom tools.
        builtin_count = 0
        for name, tool_dict in all_builtin_tools.items():
            if name not in disabled_builtin_tools:
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
        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": f"Warning: Could not load builtin tools: {e}",
                        "done": False,
                    },
                }
            )

    return tools_dict


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
            description="Comma-separated list of tool IDs available to sub-agents. Leave empty to use all tools.",
        )
        EXCLUDED_TOOL_IDS: str = Field(
            default="",
            description="Comma-separated list of tool IDs to exclude from sub-agents (e.g., this tool itself to prevent recursion).",
        )
        APPLY_INLET_FILTERS: bool = Field(
            default=True,
            description="Apply inlet filters (e.g., user_info_injector) to sub-agent requests. Outlet filters are never applied to sub-agent responses.",
        )

        # Builtin tool category toggles
        ENABLE_TIME_TOOLS: bool = Field(
            default=True,
            description="Enable time utilities (get_current_timestamp, calculate_timestamp).",
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
        ENABLE_CODE_INTERPRETER_TOOLS: bool = Field(
            default=True,
            description="Enable code interpreter tools (execute_code).",
        )
        MAX_PARALLEL_AGENTS: int = Field(
            default=5,
            description="Maximum number of sub-agents to run in parallel via run_parallel_sub_agents. To fully disable parallel execution, comment out the run_parallel_sub_agents method.",
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
    ) -> str:
        """
        Delegate a task to a sub-agent for autonomous completion.

        MANDATORY: If a task requires 3+ steps of investigation or complex analysis,
        you MUST NOT perform it yourself. Delegate to this tool immediately.
        Only handle simple 1-2 tool call tasks yourself. When in doubt, delegate.

        The sub-agent runs in an isolated context with access to the same tools.
        It executes tools in a loop until completion, returning only the final result
        to keep the main conversation context clean.

        :param description: Brief task summary (shown to user as status)
        :param prompt: Detailed instructions for the sub-agent
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

        tools_dict = await load_sub_agent_tools(
            request=__request__,
            user=user,
            valves=self.valves,
            metadata=__metadata__ or {},
            model=resolved_model,
            extra_params=common_extra_params,
            self_tool_id=__id__,
        )

        # Build initial messages
        messages = [
            {"role": "system", "content": user_valves.SYSTEM_PROMPT},
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

    async def run_parallel_sub_agents(
        self,
        tasks: list,
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
    ) -> str:
        """
        Run multiple independent sub-agent tasks in parallel (concurrently).

        Use this instead of calling run_sub_agent multiple times when you have
        2 or more tasks that do NOT depend on each other's results.
        All tasks share the same model and tools but run in isolated contexts,
        so they execute simultaneously and finish much faster than sequential calls.

        :param tasks: List of task objects. Each must have "description" and "prompt".
                      Craft each prompt as you would for run_sub_agent (role, context,
                      specific instructions, expected output format, etc.).
                      Example: [
                          {"description": "Research topic A", "prompt": "You are a research specialist. ..."},
                          {"description": "Analyze data B", "prompt": "You are a data analyst. ..."}
                      ]
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

        if not isinstance(tasks, list):
            return json.dumps(
                {
                    "error": f"tasks must be a list, got {type(tasks).__name__}",
                    "expected_format": '[{"description": "Task summary", "prompt": "Detailed instructions"}]',
                },
                ensure_ascii=False,
            )

        if not tasks:
            return json.dumps({"error": "tasks array is empty"}, ensure_ascii=False)

        if len(tasks) > self.valves.MAX_PARALLEL_AGENTS:
            return json.dumps(
                {
                    "error": f"tasks count ({len(tasks)}) exceeds MAX_PARALLEL_AGENTS ({self.valves.MAX_PARALLEL_AGENTS})",
                    "max_parallel_agents": self.valves.MAX_PARALLEL_AGENTS,
                },
                ensure_ascii=False,
            )

        validated_tasks = []
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                return json.dumps(
                    {"error": f"tasks[{i}] must be an object"},
                    ensure_ascii=False,
                )
            if "description" not in task:
                return json.dumps(
                    {"error": f"tasks[{i}] missing 'description' field"},
                    ensure_ascii=False,
                )
            if "prompt" not in task:
                return json.dumps(
                    {"error": f"tasks[{i}] missing 'prompt' field"},
                    ensure_ascii=False,
                )
            if not isinstance(task.get("description"), str):
                return json.dumps(
                    {"error": f"tasks[{i}].description must be a string"},
                    ensure_ascii=False,
                )
            if not isinstance(task.get("prompt"), str):
                return json.dumps(
                    {"error": f"tasks[{i}].prompt must be a string"},
                    ensure_ascii=False,
                )

            description = task.get("description", "").strip()
            prompt = task.get("prompt", "").strip()

            if not description:
                return json.dumps(
                    {"error": f"tasks[{i}].description cannot be empty"},
                    ensure_ascii=False,
                )
            if not prompt:
                return json.dumps(
                    {"error": f"tasks[{i}].prompt cannot be empty"},
                    ensure_ascii=False,
                )

            validated_tasks.append({"description": description, "prompt": prompt})

        # Import here to avoid issues when not running in Open WebUI
        from open_webui.models.users import UserModel

        user = UserModel(**__user__)

        # Get user valves
        raw_user_valves = (__user__ or {}).get("valves", {})
        user_valves = coerce_user_valves(raw_user_valves, self.UserValves)

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
        tools_dict = await load_sub_agent_tools(
            request=__request__,
            user=user,
            valves=self.valves,
            metadata=__metadata__ or {},
            model=resolved_model,
            extra_params=common_extra_params,
            self_tool_id=__id__,
        )

        if __event_emitter__:
            task_mapping = ", ".join(
                f"[{i + 1}] {task['description']}" for i, task in enumerate(validated_tasks)
            )
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
                        {"role": "system", "content": user_valves.SYSTEM_PROMPT},
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
        gathered_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

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
