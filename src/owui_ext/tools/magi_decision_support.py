"""
title: MAGI decision support
author: https://github.com/skyzi000
version: 0.2.9
license: MIT
required_open_webui_version: 0.7.0

MAGI decision support tool for Open WebUI.
This tool orchestrates three independent perspectives for multi-agent decision support:
- MELCHIOR: Scientific/technical analysis (logic, data, feasibility)
- BALTHASAR: Legal/ethical analysis (laws, ethics, social responsibility)
- CASPER: Emotional/trend analysis (human experience, trends, empathy)

Inspired by the MAGI supercomputer system from Neon Genesis Evangelion.
Not affiliated with the official Neon Genesis Evangelion, of course.

Agent characteristics referenced from https://github.com/yostos/claude-code-plugins (MIT License).
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import Request
from pydantic import BaseModel, Field

from owui_ext.shared.completion_response import format_chat_completion_error
from owui_ext.shared.event_emitter import EventEmitter
from owui_ext.shared.inlet_filters import apply_inlet_filters_if_enabled
from owui_ext.shared.mcp_tools import cleanup_mcp_clients
from owui_ext.shared.parsing import normalize_text, safe_json_loads
from owui_ext.shared.prompt_utils import (
    _append_tool_server_prompts,
    merge_prompt_sections,
)
from owui_ext.shared.skills import (
    extract_skill_manifest,
    extract_user_skill_tags,
    register_view_skill,
)
from owui_ext.shared.tool_execution import (
    execute_direct_tool_call,
    execute_tool_call,
    process_tool_result,
)
from owui_ext.shared.tool_loader import build_tools_dict
from owui_ext.shared.valves import coerce_user_valves
from owui_ext.shared.voting import compute_vote_tally, decide_majority

log = logging.getLogger(__name__)


# Agent-specific characteristics based on MAGI system design
# Adapted for decision support: scientific, ethical, and emotional perspectives
# Reference: https://github.com/yostos/claude-code-plugins (MIT License)
AGENT_CHARACTERISTICS = {
    "MELCHIOR": {
        "role": "Scientific Analysis",
        "traits": [
            "Deep knowledge of science and technology",
            "Makes judgments based on logic and rationality",
            "Emphasizes data, evidence, and causal relationships",
            "Uses efficiency, feasibility, and technical validity as evaluation criteria",
        ],
        "behavioral_guidelines": [
            "Evaluate the technical feasibility of each option",
            "Make judgments based on scientific evidence and data",
            "Analyze quantitative aspects such as cost, efficiency, and scalability",
            "Search for and reference the latest technological trends as needed",
            "Clarify logical causal relationships",
            "Document all external references (search queries, URLs, data sources)",
        ],
        "judgment_criteria": [
            "Technical feasibility",
            "Efficiency and performance",
            "Scalability and sustainability",
            "Cost-performance ratio",
            "Quality and quantity of evidence",
        ],
    },
    "BALTHASAR": {
        "role": "Guardian of Law and Ethics",
        "traits": [
            "Has a 'maternal' nature, emphasizing protection and norms",
            "High legal awareness, judges based on laws and precedents",
            "High ethical awareness, prioritizes ethics over rationality",
            "Considers social responsibility and public interest",
        ],
        "behavioral_guidelines": [
            "Actively search for relevant laws, regulations, and guidelines",
            "Reference precedents and prior cases to assess legal risks",
            "Examine ethical issues from multiple perspectives",
            "Analyze from the perspective of protecting the vulnerable and social equity",
            "Emphasize compliance and social responsibility",
            "Document all external references (laws, regulations, court cases, guidelines)",
        ],
        "judgment_criteria": [
            "Legal compliance",
            "Ethical validity",
            "Social responsibility and public interest",
            "Risk management and safety",
            "Impact on stakeholders",
        ],
    },
    "CASPER": {
        "role": "Advocate of Emotion and Trends",
        "traits": [
            "Has a 'feminine' nature, makes judgments based on emotions",
            "Emphasizes people's emotions and satisfaction (sympathy, joy, anxiety, etc.)",
            "Sensitive to current trends",
            "Actively investigates new cases and progressive initiatives",
            "Emphasizes user experience and human aspects",
        ],
        "behavioral_guidelines": [
            "Analyze the emotional impact each option has on stakeholders",
            "Search for and reference the latest trends",
            "Evaluate from the perspective of user experience and customer satisfaction",
            "Assess innovation and progressiveness",
            "Maintain a perspective of empathy and consideration",
            "Document all external references (trend reports, case studies, survey data)",
        ],
        "judgment_criteria": [
            "Emotional impact and satisfaction",
            "Quality of user experience",
            "Alignment with trends",
            "Innovation and progressiveness",
            "Human consideration and empathy",
        ],
    },
}


def build_agent_prompts(
    role_name: str,
    perspective: str,
    proposition: str,
    prerequisites: str,
    option_a: str,
    option_b: str,
    include_sources: bool,
) -> Tuple[str, str]:
    characteristics = AGENT_CHARACTERISTICS.get(role_name, {})
    role = characteristics.get("role", perspective)
    traits = characteristics.get("traits", [])
    guidelines = characteristics.get("behavioral_guidelines", [])
    criteria = characteristics.get("judgment_criteria", [])

    traits_text = "\n".join(f"- {t}" for t in traits) if traits else ""
    guidelines_text = "\n".join(f"- {g}" for g in guidelines) if guidelines else ""
    criteria_text = "\n".join(f"- {c}" for c in criteria) if criteria else ""

    system_prompt = (
        f"You are {role_name}, the {role} agent in the MAGI decision support system.\n\n"
        "CORE CHARACTERISTICS:\n"
        f"{traits_text}\n\n"
        "BEHAVIORAL GUIDELINES:\n"
        f"{guidelines_text}\n\n"
        "JUDGMENT CRITERIA:\n"
        f"{criteria_text}\n\n"
        "CRITICAL RULES:\n"
        "1. You operate independently and must not assume other agents' opinions.\n"
        "2. Use available tools proactively to gather information relevant to YOUR perspective.\n"
        "3. After gathering information, provide your final answer as JSON.\n"
        "4. You have limited tool call iterations. Use them wisely before the limit.\n\n"
        "FINAL RESPONSE FORMAT (JSON only):\n"
        "- vote: 'A', 'B', or 'abstain'\n"
        "- reasoning: detailed explanation from your unique perspective\n"
        "- benefits: array of short bullet phrases for chosen option\n"
        "- risks: array of short bullet phrases for chosen option\n"
        "- sources: array of {title, url} objects for data you gathered"
    )

    sources_note = "Include all sources from your web research." if include_sources else ""

    # Perspective-specific search guidance
    search_guidance = {
        "MELCHIOR": "Search for technical data, benchmarks, efficiency metrics, feasibility studies, and scientific evidence.",
        "BALTHASAR": "Search for relevant laws, regulations, legal precedents, ethical guidelines, and compliance requirements.",
        "CASPER": "Search for user experience reports, trend analyses, emotional impact studies, and innovative case studies.",
    }
    perspective_search = search_guidance.get(role_name, "Search for relevant information.")

    user_prompt = (
        "Decision Context:\n"
        f"- Proposition: {proposition}\n"
        f"- Prerequisites: {prerequisites or 'None'}\n"
        f"- Option A: {option_a}\n"
        f"- Option B: {option_b}\n\n"
        "Instructions:\n"
        f"1. {perspective_search}\n"
        "2. Analyze findings through your unique lens and judgment criteria.\n"
        "3. Provide your final JSON response with vote and detailed reasoning.\n"
        f"{sources_note}"
    )

    return system_prompt, user_prompt


async def generate_single_completion(
    request: Request,
    user: Any,
    model_id: str,
    messages: List[dict],
) -> str:
    from open_webui.utils.chat import generate_chat_completion

    form_data = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "metadata": {"task": "magi_arbitrator"},
    }

    response = await generate_chat_completion(
        request=request,
        form_data=form_data,
        user=user,
        bypass_filter=True,
    )

    error_msg = format_chat_completion_error(response)
    if error_msg is not None:
        return error_msg

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    return ""


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
                "task": "magi_agent",
                "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
            },
        }

        if tools_param:
            form_data["tools"] = tools_param

        form_data = await apply_inlet_filters_if_enabled(
            apply_inlet_filters, request, model, form_data, extra_params
        )
        form_data = _append_tool_server_prompts(form_data, extra_params)

        try:
            response = await generate_chat_completion(
                request=request,
                form_data=form_data,
                user=user_obj,
                bypass_filter=True,
            )
        except Exception as exc:
            log.exception(f"Error in MAGI agent completion: {exc}")
            return f"Error during MAGI agent execution: {exc}"

        error_msg = format_chat_completion_error(response)
        if error_msg is not None:
            return error_msg

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not choices:
                return "No response from model"

            message = choices[0].get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            # Emit status with LLM response content
            if event_emitter and content:
                truncated = content.replace(chr(10), ' ')[:100]
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
                tool_args_raw = tool_call.get("function", {}).get("arguments", "{}")

                if event_emitter:
                    tool_args_display = (
                        str(tool_args_raw).replace(chr(10), " ")
                        if tool_args_raw
                        else "{}"
                    )
                    args_preview = tool_args_display[:80]
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
                    result_preview = (result["content"] or "(empty)").replace(chr(10), ' ')[:80]
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
            "task": "magi_agent",
            "filter_ids": extra_params.get("__metadata__", {}).get("filter_ids", []),
        },
    }

    form_data = await apply_inlet_filters_if_enabled(
        apply_inlet_filters, request, model, form_data, extra_params
    )
    form_data = _append_tool_server_prompts(form_data, extra_params)

    try:
        response = await generate_chat_completion(
            request=request,
            form_data=form_data,
            user=user_obj,
            bypass_filter=True,
        )
        error_msg = format_chat_completion_error(response)
        if error_msg is not None:
            return error_msg
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
    except Exception as exc:
        log.exception(f"Error getting final MAGI agent response: {exc}")

    return "MAGI agent reached maximum iterations without providing a final response."


class Tools:
    """MAGI decision support tool."""

    class Valves(BaseModel):
        DEFAULT_MODEL: str = Field(
            default="",
            description="Default model ID for MAGI agents. Leave empty to use the current model.",
        )
        MAX_ITERATIONS: int = Field(
            default=6,
            description="Maximum tool-call iterations per agent.",
        )
        APPLY_INLET_FILTERS: bool = Field(
            default=True,
            description="Apply inlet filters (e.g., user_info_injector) to MAGI agent requests.",
        )

        # Builtin tool category toggles
        ENABLE_TIME_TOOLS: bool = Field(
            default=True,
            description="Enable time utilities (get_current_timestamp, calculate_timestamp).",
        )
        ENABLE_WEB_TOOLS: bool = Field(
            default=True,
            description="Enable web search tools (search_web, fetch_url) for MAGI agents.",
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
        ENABLE_TERMINAL_TOOLS: bool = Field(
            default=True,
            description=(
                "Enable Open Terminal tools when terminal_id is available in chat metadata "
                "(e.g., run_command, list_files, read_file, write_file, display_file)."
            ),
        )
        ENABLE_CODE_INTERPRETER_TOOLS: bool = Field(
            default=True,
            description="Enable code interpreter tools (execute_code).",
        )
        ENABLE_SKILLS_TOOLS: bool = Field(
            default=True,
            description=(
                "Enable skill context for MAGI agents. When the parent "
                "conversation has a <available_skills> manifest, the manifest "
                "is appended to each agent's system prompt and view_skill is "
                "registered as a tool so agents can lazy-load skill contents. "
                "User-selected <skill> tags are also inlined when present."
            ),
        )
        ENABLE_TASK_TOOLS: bool = Field(
            default=True,
            description="Enable task management tools (create_tasks, update_task).",
        )
        ENABLE_AUTOMATION_TOOLS: bool = Field(
            default=True,
            description="Enable automation tools (create/update/list/toggle/delete automations).",
        )
        ENABLE_CALENDAR_TOOLS: bool = Field(
            default=True,
            description="Enable calendar tools (search/create/update/delete calendar events).",
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging.",
        )
        pass

    class UserValves(BaseModel):
        INCLUDE_SOURCES: bool = Field(
            default=True,
            description="Include sources in agent outputs when web research is used.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def magi_decide(
        self,
        proposition: str,
        option_a: str,
        option_b: str,
        prerequisites: Optional[str] = None,
        __user__: dict = None,  # type: ignore
        __request__: Request = None,
        __model__: dict = None,
        __metadata__: dict = None,
        __messages__: Optional[list] = None,
        __id__: str = None,
        __event_emitter__: Callable[[dict], Any] = None,  # type: ignore
        __event_call__: Callable[[dict], Any] = None,  # type: ignore
        __chat_id__: str = None,
        __message_id__: str = None,
        __oauth_token__: Optional[dict] = None,
    ) -> str:
        """
        Run MAGI decision support with three independent perspectives and majority vote.

        :param proposition: The decision question to analyze.
        :param option_a: First option to evaluate.
        :param option_b: Second option to evaluate.
        :param prerequisites: Optional constraints and assumptions for the decision.

        Returns a JSON string with:
        - decision: Majority vote result ("A", "B", "tie", or "no_decision")
        - vote_tally: Vote counts for each option
        - summary: Brief summary of the analysis
        - agents: MELCHIOR, BALTHASAR, CASPER results with votes and reasoning
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
                    "error": "Missing required fields for MAGI decision.",
                    "missing": missing,
                    "action": "Provide proposition, option_a, and option_b parameters.",
                },
                ensure_ascii=False,
            )

        model_id = self.valves.DEFAULT_MODEL or (__model__ or {}).get("id", "")
        if not model_id:
            return json.dumps(
                {
                    "error": "No model ID available.",
                    "action": "Set DEFAULT_MODEL in Valves or provide __model__.",
                },
                ensure_ascii=False,
            )

        # Keep tool loading and injected __model__ aligned with the actual
        # model used for MAGI completions when DEFAULT_MODEL overrides the
        # parent chat model.
        resolved_model = __model__ or {}
        if model_id and model_id != resolved_model.get("id", ""):
            try:
                resolved_model = __request__.app.state.MODELS.get(
                    model_id, resolved_model
                )
            except Exception:
                pass  # Fall back to parent model metadata.

        extra_params = {
            "__user__": __user__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__request__": __request__,
            "__model__": resolved_model,
            "__metadata__": __metadata__ or {},
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
            "__oauth_token__": __oauth_token__,
            "__files__": (__metadata__ or {}).get("files", []),
        }

        tool_id_list = []
        if __metadata__ and __metadata__.get("tool_ids"):
            tool_id_list = list(__metadata__.get("tool_ids", []))

        excluded_tool_ids = set()
        if __id__:
            excluded_tool_ids.add(__id__)

        # Skill context from the parent conversation. The agent loop bypasses
        # the core middleware that would normally bind view_skill, so it has
        # to be registered manually after build_tools_dict returns.
        skills_enabled = bool(getattr(self.valves, "ENABLE_SKILLS_TOOLS", True))
        skill_manifest = extract_skill_manifest(__messages__) if skills_enabled else ""
        user_skill_tags = extract_user_skill_tags(__messages__) if skills_enabled else []

        tools_dict, mcp_clients = await build_tools_dict(
            request=__request__,
            model=resolved_model,
            metadata=__metadata__ or {},
            user=user,
            valves=self.valves,
            extra_params=extra_params,
            tool_id_list=tool_id_list,
            excluded_tool_ids=excluded_tool_ids,
        )

        # MCP clients are live as soon as build_tools_dict returns; the
        # try/finally has to start *here* so an exception or cancellation in
        # the status emit / setup steps below still triggers cleanup.
        try:
            if skills_enabled and skill_manifest:
                await register_view_skill(tools_dict, __request__, extra_params)

            roles = [
                ("MELCHIOR", "scientific/technical"),
                ("BALTHASAR", "legal/ethical"),
                ("CASPER", "emotional/trend"),
            ]

            agent_results: Dict[str, dict] = {}
            raw_outputs: Dict[str, str] = {}

            await emitter.emit(
                description=f"MAGI Starting: {prop}",
                status="agents_starting",
                done=False,
            )

            async def run_single_agent(role_name: str, perspective: str) -> Tuple[str, str, dict]:
                """Run a single MAGI agent and return (role_name, raw_output, parsed_result)."""
                base_system_prompt, user_prompt = build_agent_prompts(
                    role_name=role_name,
                    perspective=perspective,
                    proposition=prop,
                    prerequisites=prereq,
                    option_a=opt_a,
                    option_b=opt_b,
                    include_sources=include_sources,
                )
                if skills_enabled and (user_skill_tags or skill_manifest):
                    system_prompt = merge_prompt_sections(
                        base_system_prompt,
                        *user_skill_tags,
                        skill_manifest,
                    )
                else:
                    system_prompt = base_system_prompt
                content = await run_agent_loop(
                    request=__request__,
                    user=user,
                    model_id=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tools_dict=tools_dict,
                    max_iterations=self.valves.MAX_ITERATIONS,
                    extra_params=extra_params,
                    apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                    agent_name=role_name,
                    event_emitter=__event_emitter__,
                )

                parsed = safe_json_loads(content) or {
                    "vote": "abstain",
                    "reasoning": content,
                    "benefits": [],
                    "risks": [],
                    "sources": [],
                }
                return role_name, content, parsed

            import asyncio

            agent_tasks = [run_single_agent(role_name, perspective) for role_name, perspective in roles]
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        finally:
            if mcp_clients:
                await cleanup_mcp_clients(mcp_clients)

        for result in results:
            if isinstance(result, Exception):
                log.exception(f"Agent execution failed: {result}")
                continue
            role_name, content, parsed = result
            raw_outputs[role_name] = content
            agent_results[role_name] = parsed

        votes = []
        for role_name, _ in roles:
            vote = normalize_text(agent_results.get(role_name, {}).get("vote", "")).upper()
            if vote not in {"A", "B", "ABSTAIN"}:
                vote = "ABSTAIN"
            votes.append("abstain" if vote == "ABSTAIN" else vote)

        tally = compute_vote_tally(votes)
        decision = decide_majority(tally)

        summary_prompt = (
            "Summarize the MAGI decision succinctly based on the three agents' analysis results. "
            "Highlight the key points from each perspective (technical, legal/ethical, emotional/trend). "
            "Provide 3-6 sentences."
        )
        summary_input = {
            "proposition": prop,
            "option_a": opt_a,
            "option_b": opt_b,
            "vote_tally": tally,
            "agents": agent_results,
        }

        summary = ""
        try:
            summary = await generate_single_completion(
                request=__request__,
                user=user,
                model_id=model_id,
                messages=[
                    {"role": "system", "content": summary_prompt},
                    {"role": "user", "content": json.dumps(summary_input, ensure_ascii=False)},
                ],
            )
        except Exception as exc:
            log.exception(f"Error generating summary: {exc}")
            summary = ""

        decision_text = opt_a if decision == "A" else opt_b if decision == "B" else decision
        await emitter.emit(
            description=f"MAGI Complete: {decision_text}",
            status="complete",
            done=True,
        )

        return json.dumps(
            {
                "decision": decision,
                "vote_tally": tally,
                "summary": summary,
                "agents": agent_results,
                "raw_outputs": raw_outputs if self.valves.DEBUG else None,
            },
            ensure_ascii=False,
        )
