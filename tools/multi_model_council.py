"""
title: Multi Model Council
description: Run a multi-model council decision with majority vote. Each council member operates independently, can use tools (web search, knowledge bases, etc.) for analysis, and returns their vote with reasoning.
author: https://github.com/skyzi000
version: 0.1.4
license: MIT
required_open_webui_version: 0.7.0
"""

import ast
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from fastapi import Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


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
}

# Tools that generate citation sources
# NOTE: Update this set when new citation-capable tools are added to Open WebUI.
# See open_webui/utils/middleware.py:get_citation_source_from_tool_result() for supported tools.
CITATION_TOOLS = {"search_web", "view_knowledge_file", "query_knowledge_files"}


# ============================================================================
# Event emitter
# ============================================================================


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):  # type: ignore
        self.event_emitter = event_emitter

    async def emit(
        self, description: str = "Unknown state", status: str = "in_progress", done: bool = False
    ) -> None:
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
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


def truncate_text(value: str, limit: int = 200) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def coerce_user_valves(raw_valves: Any, valves_cls: Type[BaseModel]) -> BaseModel:
    if isinstance(raw_valves, valves_cls):
        return raw_valves
    if hasattr(raw_valves, "model_dump"):
        try:
            data = raw_valves.model_dump()
        except Exception:
            data = {}
    elif isinstance(raw_valves, dict):
        data = raw_valves
    else:
        data = {}
    return valves_cls.model_validate(data)


def parse_model_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]

    seen = set()
    result: List[str] = []
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


def build_council_system_prompt(base_prompt: str, include_sources: bool) -> str:
    sources_note = (
        "Include all sources from your research in the sources array."
        if include_sources
        else "If you did not use external sources, leave sources as an empty array."
    )

    council_prompt = (
        "You are an independent council member evaluating a decision proposal.\n\n"
        "CRITICAL RULES:\n"
        "1. Operate independently and do NOT assume other members' opinions.\n"
        "2. Use available tools proactively if they help your analysis.\n"
        "3. Return ONLY a JSON object as your final response.\n"
        "4. You have limited tool call iterations. Use them wisely.\n\n"
        "FINAL RESPONSE FORMAT (JSON only):\n"
        "- vote: 'A', 'B', or 'abstain'\n"
        "- reasoning: detailed explanation for your vote\n"
        "- benefits: array of short bullet phrases for the chosen option\n"
        "- risks: array of short bullet phrases for the chosen option\n"
        "- sources: array of {title, url} objects\n\n"
        f"{sources_note}"
    )

    base_prompt = normalize_text(base_prompt)
    if base_prompt:
        return f"{base_prompt}\n\n{council_prompt}"
    return council_prompt


def build_council_user_prompt(
    proposition: str,
    prerequisites: str,
    option_a: str,
    option_b: str,
) -> str:
    return (
        "Decision Context:\n"
        f"- Proposition: {proposition}\n"
        f"- Prerequisites: {prerequisites or 'None'}\n"
        f"- Option A: {option_a}\n"
        f"- Option B: {option_b}\n\n"
        "Instructions:\n"
        "1. Analyze the proposition and compare Option A vs Option B.\n"
        "2. If the decision is unclear or information is insufficient, abstain.\n"
        "3. Provide your final JSON response."
    )


def normalize_member_result(parsed: Optional[dict], fallback_reason: str) -> dict:
    if not isinstance(parsed, dict):
        return {
            "vote": "abstain",
            "reasoning": fallback_reason,
            "benefits": [],
            "risks": [],
            "sources": [],
        }

    vote_raw = normalize_text(parsed.get("vote", "")).upper()
    if vote_raw not in {"A", "B", "ABSTAIN"}:
        vote_raw = "ABSTAIN"

    reasoning = normalize_text(parsed.get("reasoning")) or fallback_reason

    benefits = parsed.get("benefits", [])
    if not isinstance(benefits, list):
        benefits = []
    else:
        benefits = [normalize_text(item) for item in benefits if normalize_text(item)]

    risks = parsed.get("risks", [])
    if not isinstance(risks, list):
        risks = []
    else:
        risks = [normalize_text(item) for item in risks if normalize_text(item)]

    sources = parsed.get("sources", [])
    normalized_sources = []
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict):
                title = normalize_text(source.get("title"))
                url = normalize_text(source.get("url"))
                if title or url:
                    normalized_sources.append({"title": title, "url": url})

    return {
        "vote": "abstain" if vote_raw == "ABSTAIN" else vote_raw,
        "reasoning": reasoning,
        "benefits": benefits,
        "risks": risks,
        "sources": normalized_sources,
    }


def compute_vote_tally(votes: List[str]) -> dict:
    tally = {"A": 0, "B": 0, "abstain": 0}
    for vote in votes:
        if vote in tally:
            tally[vote] += 1
    return tally


def decide_majority(tally: dict) -> str:
    if tally.get("A", 0) > tally.get("B", 0):
        return "A"
    if tally.get("B", 0) > tally.get("A", 0):
        return "B"
    if tally.get("A", 0) == tally.get("B", 0) and tally.get("A", 0) > 0:
        return "tie"
    return "no_decision"


async def execute_tool_call(
    tool_call: dict,
    tools_dict: dict,
    extra_params: dict,
    event_emitter: Optional[Callable] = None,
) -> dict:
    tool_call_id = tool_call.get("id", "")
    tool_function_name = tool_call.get("function", {}).get("name", "")
    tool_args_str = tool_call.get("function", {}).get("arguments", "{}")

    tool_function_params = {}
    try:
        tool_function_params = ast.literal_eval(tool_args_str)
    except Exception:
        try:
            tool_function_params = json.loads(tool_args_str)
        except Exception as exc:
            return {
                "tool_call_id": tool_call_id,
                "content": f"Error parsing arguments: {exc}",
            }

    if tool_function_name in tools_dict:
        tool = tools_dict[tool_function_name]
        spec = tool.get("spec", {})
        try:
            allowed_params = spec.get("parameters", {}).get("properties", {}).keys()
            tool_function_params = {
                k: v for k, v in tool_function_params.items() if k in allowed_params
            }

            tool_function = tool["callable"]

            from open_webui.utils.tools import get_updated_tool_function

            tool_function = get_updated_tool_function(
                function=tool_function,
                extra_params={
                    "__messages__": extra_params.get("__messages__", []),
                    "__files__": extra_params.get("__files__", []),
                    "__user__": extra_params.get("__user__"),
                    "__request__": extra_params.get("__request__"),
                    "__model__": extra_params.get("__model__"),
                    "__metadata__": extra_params.get("__metadata__"),
                    "__event_emitter__": extra_params.get("__event_emitter__"),
                    "__event_call__": extra_params.get("__event_call__"),
                    "__chat_id__": extra_params.get("__chat_id__"),
                    "__message_id__": extra_params.get("__message_id__"),
                },
            )

            tool_result = await tool_function(**tool_function_params)
        except Exception as exc:
            log.exception(f"Error executing tool {tool_function_name}: {exc}")
            tool_result = f"Error: {exc}"
    else:
        tool_result = f"Tool '{tool_function_name}' not found"

    if tool_result is None:
        tool_result = ""
    elif not isinstance(tool_result, str):
        try:
            tool_result = json.dumps(tool_result, ensure_ascii=False, default=str)
        except Exception:
            tool_result = str(tool_result)

    # Extract and emit citation sources for tools that generate them
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
            log.warning(
                f"Error extracting citation sources from {tool_function_name}: {exc}"
            )

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
            extra_params=extra_params,
        )
    except Exception as exc:
        log.warning(f"Error applying inlet filters: {exc}")

    return form_data


async def run_agent_loop(
    request: Request,
    user: Any,
    model_id: str,
    messages: List[dict],
    tools_dict: dict,
    max_iterations: int,
    extra_params: dict,
    apply_inlet_filters: bool,
    agent_name: str = "Agent",
    event_emitter: Optional[Callable] = None,
) -> str:
    from open_webui.models.users import UserModel
    from open_webui.utils.chat import generate_chat_completion

    if extra_params is None:
        extra_params = {}

    # Prepare user object
    if isinstance(user, dict):
        user_obj = UserModel(**user)
    else:
        user_obj = user

    models = request.app.state.MODELS
    model = models.get(model_id, {})

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
                "task": "multi_model_council",
                "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
            },
        }

        if tools_param:
            form_data["tools"] = tools_param

        form_data = await apply_inlet_filters_if_enabled(
            apply_inlet_filters, request, model, form_data, extra_params
        )

        try:
            response = await generate_chat_completion(
                request=request,
                form_data=form_data,
                user=user_obj,
                bypass_filter=True,
            )
        except Exception as exc:
            log.exception(f"Error in council agent completion: {exc}")
            return f"Error during council agent execution: {exc}"

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not isinstance(choices, list) or not choices:
                return "No response from model"

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                return "Unexpected response format"

            message = first_choice.get("message", {})
            if not isinstance(message, dict):
                message = {}

            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                tool_calls = []

            # Emit status with LLM response content
            if event_emitter and content:
                truncated = content.replace(chr(10), " ")[:100]
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[{agent_name}] Step {iteration}: {truncated}...",
                            "done": False,
                        },
                    }
                )

            if not tool_calls:
                if event_emitter:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[{agent_name}] Analysis complete.",
                                "done": False,
                            },
                        }
                    )
                return content or ""

            # Emit status with tool calls summary
            if event_emitter:
                tool_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[{agent_name}] Step {iteration}: Calling {', '.join(tool_names)}",
                            "done": False,
                        },
                    }
                )

            current_messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", "{}")

                if event_emitter:
                    args_preview = tool_args.replace(chr(10), " ")[:80]
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[{agent_name}] Executing {tool_name}({args_preview}...)",
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
                    result_preview = (result["content"] or "(empty)").replace(chr(10), " ")[:80]
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[{agent_name}] {tool_name} returned: {result_preview}...",
                                "done": False,
                            },
                        }
                    )

                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"],
                    }
                )
        else:
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

    form_data = {
        "model": model_id,
        "messages": current_messages
        + [
            {
                "role": "user",
                "content": "Maximum tool iterations reached. Provide your final answer based on gathered information.",
            }
        ],
        "stream": False,
        "metadata": {
            "task": "multi_model_council",
            "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
        },
    }

    form_data = await apply_inlet_filters_if_enabled(
        apply_inlet_filters, request, model, form_data, extra_params
    )

    try:
        response = await generate_chat_completion(
            request=request,
            form_data=form_data,
            user=user_obj,
            bypass_filter=True,
        )
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message", {})
                    if isinstance(message, dict):
                        return message.get("content", "")
    except Exception as exc:
        log.exception(f"Error getting final council agent response: {exc}")

    return "Council agent reached maximum iterations without providing a final response."


async def build_tools_dict(
    request: Request,
    model: dict,
    metadata: Optional[dict],
    user: Any,
    valves: Any,
    extra_params: dict,
    tool_id_list: List[str],
    excluded_tool_ids: Optional[set],
) -> dict:
    from open_webui.utils.tools import get_builtin_tools, get_tools

    tools_dict: Dict[str, dict] = {}

    # Load regular tools available to the main agent
    regular_tool_ids = [tid for tid in tool_id_list if not tid.startswith("builtin:")]
    if excluded_tool_ids:
        regular_tool_ids = [tid for tid in regular_tool_ids if tid not in excluded_tool_ids]

    if regular_tool_ids:
        try:
            tools_dict = await get_tools(
                request=request,
                tool_ids=regular_tool_ids,
                user=user,
                extra_params=extra_params,
            )
        except Exception as exc:
            log.exception(f"Error loading regular tools: {exc}")

    # Load builtin tools
    features = metadata.get("features", {}) if metadata else {}
    all_builtin_tools = get_builtin_tools(
        request=request,
        extra_params={
            "__user__": extra_params.get("__user__"),
            "__event_emitter__": extra_params.get("__event_emitter__"),
            "__chat_id__": extra_params.get("__chat_id__"),
            "__message_id__": extra_params.get("__message_id__"),
        },
        features=features,
        model=model,
    )

    # Build set of disabled tools based on Valves categories
    disabled_builtin_tools = set()
    for valve_field, category in VALVE_TO_CATEGORY.items():
        if not getattr(valves, valve_field, True):
            disabled_builtin_tools.update(BUILTIN_TOOL_CATEGORIES.get(category, set()))

    for name, tool_dict in all_builtin_tools.items():
        if name in disabled_builtin_tools:
            continue
        if name not in tools_dict:
            tools_dict[name] = tool_dict

    return tools_dict


async def get_available_models(
    request: Request,
    user: Any,
) -> List[dict]:
    from open_webui.utils.models import get_all_models, get_filtered_models

    all_models = await get_all_models(request, refresh=False, user=user)

    filtered = []
    for model in all_models:
        if "pipeline" in model and model["pipeline"].get("type") == "filter":
            continue
        filtered.append(model)

    return get_filtered_models(filtered, user)


def extract_model_ids(models: List[dict]) -> List[str]:
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
    """Multi-model council tool."""

    class Valves(BaseModel):
        DEFAULT_MODELS: str = Field(
            default="",
            description="Comma-separated default model IDs. If empty, models must be provided in the call.",
        )
        MAX_ITERATIONS: int = Field(
            default=6,
            description="Maximum tool-call iterations per council member.",
        )
        AVAILABLE_TOOL_IDS: str = Field(
            default="",
            description="Comma-separated list of tool IDs available to council members. Leave empty to use all tools.",
        )
        EXCLUDED_TOOL_IDS: str = Field(
            default="",
            description="Comma-separated list of tool IDs to exclude from council members.",
        )
        APPLY_INLET_FILTERS: bool = Field(
            default=True,
            description="Apply inlet filters (e.g., user_info_injector) to council member requests.",
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
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging and raw outputs.",
        )
        pass

    class UserValves(BaseModel):
        SYSTEM_PROMPT: str = Field(
            default="""\
You are a council member tasked with making an independent decision.

CRITICAL RULES:
1. You MUST decide independently without relying on other members.
2. Use tools when they can improve decision quality.
3. Respond ONLY with the required JSON format.
4. Keep your reasoning grounded and concise.""",
            description="Base system prompt for council members.",
        )
        INCLUDE_SOURCES: bool = Field(
            default=True,
            description="Include sources in member outputs when web research is used.",
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

        Args:
            query: Optional search string to filter models by ID or name.
                - Example: "gpt", "claude", "gemini", "ollama"
                - Empty string returns all available models.

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
        except Exception as exc:
            log.exception(f"Error listing models: {exc}")
            return json.dumps(
                {
                    "error": "Failed to list models.",
                    "detail": str(exc),
                    "action": "Retry or check server logs for details.",
                },
                ensure_ascii=False,
            )

    async def council_decide(
        self,
        proposition: str,
        option_a: str,
        option_b: str,
        prerequisites: Optional[str] = None,
        models: Optional[str] = None,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __model__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __id__: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,  # type: ignore
        __event_call__: Callable[[dict], Any] = None,  # type: ignore
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
    ) -> str:
        """
        Run a multi-model council decision with majority vote.

        Args:
            proposition: The decision question to analyze (required).
            option_a: First option to evaluate (required).
            option_b: Second option to evaluate (required).
            prerequisites: Optional constraints and assumptions.
            models: Optional model ID list as a comma-separated string.
                - Example: "gpt-5.2, claude-4-5-sonnet, gemini-2.5-pro"
                - If omitted or empty, DEFAULT_MODELS (Valves) is used.

        Returns a JSON string with:
        - decision: "A" | "B" | "tie" | "no_decision"
        - vote_tally: {"A": n, "B": n, "abstain": n}
        - members: per-model outputs
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

        # Normalize inputs
        prop = normalize_text(proposition)
        prereq = normalize_text(prerequisites)
        opt_a = normalize_text(option_a)
        opt_b = normalize_text(option_b)

        # Validate required fields
        missing = []
        if not prop:
            missing.append("proposition")
        if not opt_a:
            missing.append("option_a")
        if not opt_b:
            missing.append("option_b")
        if missing:
            return json.dumps(
                {
                    "error": "Missing required fields for council decision.",
                    "missing": missing,
                    "action": "Provide proposition, option_a, and option_b parameters.",
                },
                ensure_ascii=False,
            )

        # Resolve model IDs
        provided_models = parse_model_ids(models)
        default_models = parse_model_ids(self.valves.DEFAULT_MODELS)

        model_ids = provided_models or default_models
        if not model_ids:
            try:
                available_models = await get_available_models(__request__, user)
                available_ids = extract_model_ids(available_models)
            except Exception:
                available_ids = []

            return json.dumps(
                {
                    "error": "No model IDs provided.",
                    "action": "Provide two or more model IDs in models.",
                    "available_models": available_ids,
                },
                ensure_ascii=False,
            )

        if len(model_ids) < 2:
            try:
                available_models = await get_available_models(__request__, user)
                available_ids = extract_model_ids(available_models)
            except Exception:
                available_ids = []

            return json.dumps(
                {
                    "error": "At least two model IDs are required.",
                    "provided_models": model_ids,
                    "available_models": available_ids,
                    "action": "Provide two or more model IDs in models.",
                },
                ensure_ascii=False,
            )

        # Validate model IDs before execution
        try:
            available_models = await get_available_models(__request__, user)
            available_ids = set(extract_model_ids(available_models))
        except Exception as exc:
            log.exception(f"Error checking available models: {exc}")
            return json.dumps(
                {
                    "error": "Failed to validate model IDs.",
                    "detail": str(exc),
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

        extra_params = {
            "__user__": __user__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__request__": request,
            "__model__": __model__,
            "__metadata__": metadata,
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
            "__files__": metadata.get("files", []),
        }

        # Determine tool IDs (same flow as sub_agent, but MAGI-style execution)
        available_tool_ids = []
        if metadata.get("tool_ids"):
            available_tool_ids = list(metadata.get("tool_ids", []))

        if self.valves.AVAILABLE_TOOL_IDS.strip():
            tool_id_list = [
                tid.strip()
                for tid in self.valves.AVAILABLE_TOOL_IDS.split(",")
                if tid.strip()
            ]
        else:
            tool_id_list = available_tool_ids

        excluded_tool_ids = set()
        if self.valves.EXCLUDED_TOOL_IDS.strip():
            excluded_tool_ids = {
                tid.strip()
                for tid in self.valves.EXCLUDED_TOOL_IDS.split(",")
                if tid.strip()
            }

        if __id__:
            excluded_tool_ids.add(__id__)

        base_system_prompt = build_council_system_prompt(
            base_prompt=user_valves.SYSTEM_PROMPT,
            include_sources=include_sources,
        )
        user_prompt = build_council_user_prompt(prop, prereq, opt_a, opt_b)

        await emitter.emit(
            description=f"Council Starting: {prop}",
            status="agents_starting",
            done=False,
        )

        tools_cache: Dict[str, dict] = {}

        async def run_single_member(member_model_id: str) -> Tuple[str, str, dict]:
            """Run one council member and return (model_id, raw_output, parsed_result)."""
            system_prompt = base_system_prompt
            member_model = {}
            member_model = request.app.state.MODELS.get(member_model_id, {})

            member_extra_params = {
                **extra_params,
                "__model__": member_model,
            }

            tools_dict = tools_cache.get(member_model_id)
            if tools_dict is None:
                tools_dict = await build_tools_dict(
                    request=request,
                    model=member_model,
                    metadata=metadata,
                    user=user,
                    valves=self.valves,
                    extra_params=member_extra_params,
                    tool_id_list=tool_id_list,
                    excluded_tool_ids=excluded_tool_ids,
                )
                tools_cache[member_model_id] = tools_dict

            content = await run_agent_loop(
                request=__request__,
                user=user,
                model_id=member_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools_dict=tools_dict,
                max_iterations=self.valves.MAX_ITERATIONS,
                extra_params=member_extra_params,
                apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                agent_name=member_model_id,
                event_emitter=__event_emitter__,
            )

            parsed = safe_json_loads(content)
            return member_model_id, content, parsed or {}

        import asyncio

        member_tasks = [run_single_member(mid) for mid in model_ids]
        results = await asyncio.gather(*member_tasks, return_exceptions=True)

        members: Dict[str, dict] = {}
        raw_outputs: Dict[str, str] = {}

        for idx, result in enumerate(results):
            model_id = model_ids[idx] if idx < len(model_ids) else "unknown"
            if isinstance(result, BaseException):
                log.exception(f"Council member failed ({model_id}): {result}")
                members[model_id] = normalize_member_result(
                    None,
                    f"Execution failed: {truncate_text(str(result), 180)}",
                )
                if self.valves.DEBUG:
                    raw_outputs[model_id] = f"Error: {result}"
                continue

            member_id, content, parsed = result
            raw_outputs[member_id] = content

            if not parsed:
                failure_reason = "Invalid or non-JSON response from model."
                content_text = normalize_text(content)
                if content_text:
                    failure_reason = f"Non-JSON response: {truncate_text(content_text, 180)}"
                members[member_id] = normalize_member_result(
                    None,
                    failure_reason,
                )
                continue

            members[member_id] = normalize_member_result(
                parsed,
                "Invalid response structure from model.",
            )

        votes = []
        for model_id in model_ids:
            vote = normalize_text(members.get(model_id, {}).get("vote", "")).upper()
            if vote not in {"A", "B", "ABSTAIN"}:
                vote = "ABSTAIN"
            votes.append("abstain" if vote == "ABSTAIN" else vote)

        tally = compute_vote_tally(votes)
        decision = decide_majority(tally)

        decision_text = opt_a if decision == "A" else opt_b if decision == "B" else decision
        await emitter.emit(
            description=f"Council Complete: {decision_text}",
            status="complete",
            done=True,
        )

        response_payload = {
            "decision": decision,
            "vote_tally": tally,
            "members": members,
        }

        if self.valves.DEBUG:
            response_payload["raw_outputs"] = raw_outputs

        return json.dumps(response_payload, ensure_ascii=False)
