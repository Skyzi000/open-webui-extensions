"""
title: Multi Model Council
description: Run a multi-model council decision with majority vote. Each council member operates independently, can use tools (web search, knowledge bases, etc.) for analysis, and returns their vote with reasoning. Also supports LLM Review collaborative writing with cross-feedback.
author: https://github.com/skyzi000
version: 0.2.0
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


# ============================================================================
# LLM Review helper functions
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
    feedbacks: List[dict],
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


def normalize_draft_result(parsed: Optional[dict], fallback_text: str) -> dict:
    if not isinstance(parsed, dict):
        return {
            "draft": fallback_text,
            "approach": "",
            "sources": [],
        }

    draft = normalize_text(parsed.get("draft")) or fallback_text
    approach = normalize_text(parsed.get("approach")) or ""

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
        "draft": draft,
        "approach": approach,
        "sources": normalized_sources,
    }


def normalize_review_result(parsed: Optional[dict], fallback_feedback: str) -> dict:
    if not isinstance(parsed, dict):
        return {
            "strengths": [],
            "improvements": [],
            "key_feedback": fallback_feedback,
        }

    strengths = parsed.get("strengths", [])
    if not isinstance(strengths, list):
        strengths = []
    else:
        strengths = [normalize_text(s) for s in strengths if normalize_text(s)]

    improvements = parsed.get("improvements", [])
    if not isinstance(improvements, list):
        improvements = []
    else:
        improvements = [normalize_text(i) for i in improvements if normalize_text(i)]

    key_feedback = normalize_text(parsed.get("key_feedback")) or fallback_feedback

    return {
        "strengths": strengths,
        "improvements": improvements,
        "key_feedback": key_feedback,
    }


def normalize_revise_result(parsed: Optional[dict], fallback_draft: str) -> dict:
    if not isinstance(parsed, dict):
        return {
            "draft": fallback_draft,
            "changes_made": [],
            "feedback_declined": [],
            "sources": [],
        }

    draft = normalize_text(parsed.get("draft")) or fallback_draft

    changes_made = parsed.get("changes_made", [])
    if not isinstance(changes_made, list):
        changes_made = []
    else:
        changes_made = [normalize_text(c) for c in changes_made if normalize_text(c)]

    feedback_declined = parsed.get("feedback_declined", [])
    if not isinstance(feedback_declined, list):
        feedback_declined = []
    else:
        feedback_declined = [
            normalize_text(f) for f in feedback_declined if normalize_text(f)
        ]

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
        "draft": draft,
        "changes_made": changes_made,
        "feedback_declined": feedback_declined,
        "sources": normalized_sources,
    }


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
        LLM_REVIEW_SYSTEM_PROMPT: str = Field(
            default="""\
You are a collaborative writing agent participating in a multi-agent review process.

CRITICAL RULES:
1. Maintain your unique perspective and writing voice throughout.
2. Use tools when they help research and improve your writing.
3. Respond ONLY with the required JSON format.
4. Be open to feedback but don't lose your distinctive style.""",
            description="Base system prompt for LLM Review agents.",
        )
        LLM_REVIEW_ROUNDS: int = Field(
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

    async def llm_review(
        self,
        topic: str,
        requirements: Optional[str] = None,
        models: Optional[str] = None,
        personas: Optional[str] = None,
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
        Run an LLM Review collaborative writing process with cross-feedback.

        This implements the LLM Review methodology where multiple agents with
        distinct personas independently write drafts, exchange feedback, and
        revise their work over multiple rounds.

        Args:
            topic: The writing topic or prompt (required).
            requirements: Optional specific requirements or constraints.
            models: Optional model ID list as a comma-separated string.
                - Example: "gpt-5.2, claude-4-5-sonnet, gemini-2.5-pro"
                - Exactly 3 models are required for the review process.
                - If omitted, DEFAULT_MODELS (Valves) is used.
            personas: Optional JSON string with custom personas.
                - Format: [{"name": "...", "description": "..."}, ...]
                - If omitted, default personas (Analytical, Creative, Practical) are used.

        Returns a JSON string with:
        - rounds: array of round results with drafts and feedback
        - final_drafts: the final draft from each agent
        - agents: mapping of model_id to persona name
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
        num_rounds = int(user_valves.LLM_REVIEW_ROUNDS)
        base_system_prompt = str(user_valves.LLM_REVIEW_SYSTEM_PROMPT)

        # Normalize inputs
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
        agent_personas: List[dict] = []
        if personas:
            personas_text = normalize_text(personas)
            if personas_text:
                parsed_personas = safe_json_loads(personas_text)
                if isinstance(parsed_personas, list):
                    for p in parsed_personas:
                        if isinstance(p, dict) and p.get("name"):
                            agent_personas.append(p)

        if len(agent_personas) < 3:
            agent_personas = list(DEFAULT_PERSONAS)

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
                    "action": "Provide exactly 3 model IDs in models parameter.",
                },
                ensure_ascii=False,
            )

        # Use exactly 3 models
        model_ids = model_ids[:3]

        # Validate model IDs
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

        # Build tools dict
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

        # Assign personas to models
        agents = {model_ids[i]: agent_personas[i] for i in range(3)}

        await emitter.emit(
            description=f"LLM Review Starting: {truncate_text(topic_text, 50)}",
            status="review_starting",
            done=False,
        )

        import asyncio

        tools_cache: Dict[str, dict] = {}
        rounds_data: List[dict] = []

        # Current drafts for each agent
        current_drafts: Dict[str, dict] = {}

        # Phase 1: Initial Composition
        await emitter.emit(
            description="Phase 1: Initial Composition",
            status="composing",
            done=False,
        )

        async def compose_draft(model_id: str) -> Tuple[str, str, dict]:
            """Have an agent compose their initial draft."""
            persona = agents[model_id]
            member_model = request.app.state.MODELS.get(model_id, {})

            member_extra_params = {
                **extra_params,
                "__model__": member_model,
            }

            tools_dict = tools_cache.get(model_id)
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
                tools_cache[model_id] = tools_dict

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

        compose_tasks = [compose_draft(mid) for mid in model_ids]
        compose_results = await asyncio.gather(*compose_tasks, return_exceptions=True)

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

        # Store initial drafts
        rounds_data.append(
            {
                "round": 0,
                "phase": "initial_composition",
                "drafts": {
                    mid: {
                        "persona": agents[mid]["name"],
                        "draft": current_drafts[mid]["draft"],
                        "approach": current_drafts[mid].get("approach", ""),
                    }
                    for mid in model_ids
                },
            }
        )

        # Rounds of Review and Revision
        for round_num in range(1, num_rounds + 1):
            await emitter.emit(
                description=f"Round {round_num}/{num_rounds}: Cross-Review",
                status="reviewing",
                done=False,
            )

            # Phase 2: Cross-Review
            # Each agent reviews the other two drafts
            all_reviews: Dict[str, Dict[str, dict]] = {mid: {} for mid in model_ids}

            async def review_draft(
                reviewer_id: str, author_id: str
            ) -> Tuple[str, str, str, dict]:
                """Have reviewer provide feedback on author's draft."""
                reviewer_persona = agents[reviewer_id]
                author_persona = agents[author_id]
                author_draft = current_drafts[author_id]["draft"]

                member_model = request.app.state.MODELS.get(reviewer_id, {})
                member_extra_params = {
                    **extra_params,
                    "__model__": member_model,
                }

                system_prompt = build_review_system_prompt(
                    base_system_prompt, reviewer_persona
                )
                user_prompt = build_review_user_prompt(
                    topic_text, author_persona["name"], author_draft
                )

                # Reviews don't need tools - just analysis
                content = await run_agent_loop(
                    request=request,
                    user=user,
                    model_id=reviewer_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tools_dict={},  # No tools for review
                    max_iterations=2,  # Quick review, minimal iterations
                    extra_params=member_extra_params,
                    apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                    agent_name=f"{reviewer_persona['name']}→{author_persona['name']}",
                    event_emitter=__event_emitter__,
                )

                parsed = safe_json_loads(content)
                return reviewer_id, author_id, content, parsed or {}

            # Generate all review tasks (each agent reviews the other two)
            review_tasks = []
            for reviewer_id in model_ids:
                for author_id in model_ids:
                    if reviewer_id != author_id:
                        review_tasks.append(review_draft(reviewer_id, author_id))

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

            # Phase 3: Revision
            await emitter.emit(
                description=f"Round {round_num}/{num_rounds}: Revision",
                status="revising",
                done=False,
            )

            async def revise_draft(model_id: str) -> Tuple[str, str, dict]:
                """Have an agent revise their draft based on feedback."""
                persona = agents[model_id]
                original_draft = current_drafts[model_id]["draft"]

                # Collect feedback from the other two agents
                feedbacks = []
                for reviewer_id, review in all_reviews[model_id].items():
                    feedbacks.append(review)

                member_model = request.app.state.MODELS.get(model_id, {})
                member_extra_params = {
                    **extra_params,
                    "__model__": member_model,
                }

                tools_dict = tools_cache.get(model_id, {})

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

            revise_tasks = [revise_draft(mid) for mid in model_ids]
            revise_results = await asyncio.gather(*revise_tasks, return_exceptions=True)

            revised_drafts: Dict[str, dict] = {}
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

            # Update current drafts with revisions
            for mid in model_ids:
                current_drafts[mid] = {
                    "draft": revised_drafts[mid]["draft"],
                    "approach": current_drafts[mid].get("approach", ""),
                    "sources": revised_drafts[mid].get("sources", []),
                }

            # Store round data
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
                        }
                        for mid in model_ids
                    },
                }
            )

        await emitter.emit(
            description="LLM Review Complete",
            status="complete",
            done=True,
        )

        # Compile final response
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
