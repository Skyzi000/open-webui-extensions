"""Prompt-assembly helpers shared across owui_ext tool plugins."""

import copy
import inspect
from typing import Iterable, Optional


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
    """Add terminal instructions and tool-server prompts to a request snapshot.

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
    agents_md = (extra_params or {}).get("__terminal_agents_md__")
    if not isinstance(agents_md, str):
        agents_md = ""
    if not prompts and not agents_md:
        return form_data
    messages = list(form_data.get("messages", []))
    combined = "\n\n".join(prompts)
    if combined and messages and messages[0].get("role") == "system":
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
    elif combined:
        messages.insert(0, {"role": "system", "content": combined})
    if agents_md and not any(
        message.get("role") == "user" and message.get("content") == agents_md
        for message in messages
    ):
        try:
            from open_webui.utils.terminals import add_terminal_agents_md
        except ImportError:
            pass  # Older Core versions do not support terminal AGENTS.md.
        else:
            messages = add_terminal_agents_md(messages, agents_md)
    form_data["messages"] = messages
    return form_data


async def model_custom_tools(filter_pipeline: dict) -> list:
    """Return the model's custom ``tools`` param, which Core 0.12.3+ sends after the loaded tools."""
    try:
        from open_webui.models.config import Config
        from open_webui.models.models import Models
        from open_webui.utils.misc import merge_model_params
        from open_webui.utils.payload import apply_params_to_form_data
    except ImportError:
        return []
    model = filter_pipeline["model"]
    model_info = (
        None
        if model.get("direct")
        else await Models.get_model_by_id(filter_pipeline["model_id"])
    )
    params = merge_model_params(
        copy.deepcopy(await Config.get("models.default_params", {}) or {}),
        model_info.params.model_dump() if model_info and model_info.params else {},
    )
    tools = apply_params_to_form_data({}, model, params).get("tools")
    return tools if isinstance(tools, list) else []


def build_agent_tools_param(
    tools_dict: dict, deferred_tools: set, custom_tools: list
) -> Optional[list]:
    tools_param = [
        {"type": "function", "function": tool.get("spec", {})}
        for name, tool in tools_dict.items()
        if name not in deferred_tools
    ]
    return tools_param + custom_tools or None


async def apply_agent_tool_search(
    tools_dict: dict,
    extra_params: dict,
    always_loaded: Iterable[str] = (),
    extra_tools: Iterable[dict] = (),
) -> tuple[dict, dict, set]:
    """Apply Core's Tool Search to one agent loop.

    Returns the loop's tools (with ``search_tools``), extra_params carrying the
    manifest, and the names to leave out of the request's ``tools``. The search
    state lives in ``extra_params["__metadata__"]``, which must belong to this agent.
    """
    try:
        from open_webui.utils.tool_search import SEARCH_TOOL_NAME, apply_tool_search
    except ImportError:
        return tools_dict, extra_params, set()

    kept = {name: tools_dict[name] for name in always_loaded if name in tools_dict}
    loop_tools = {name: tool for name, tool in tools_dict.items() if name not in kept}
    metadata = extra_params.get("__metadata__")
    if not isinstance(metadata, dict):
        metadata = {}
    form_data: dict = {"messages": []}
    if "extra_tools" in inspect.signature(apply_tool_search).parameters:
        deferred = await apply_tool_search(
            form_data, metadata, loop_tools, list(extra_tools)
        )
    elif any(
        (tool.get("function") or tool).get("name") == SEARCH_TOOL_NAME
        for tool in extra_tools
    ):
        return tools_dict, extra_params, set()
    else:
        deferred = await apply_tool_search(form_data, metadata, loop_tools)
    if not deferred:
        return tools_dict, extra_params, set()
    metadata["tools"] = loop_tools
    prompts = extra_params.get("__direct_tool_server_system_prompts__") or []
    return (
        {**loop_tools, **kept},
        {
            **extra_params,
            "__metadata__": metadata,
            "__direct_tool_server_system_prompts__": [
                *prompts,
                form_data["messages"][0]["content"],
            ],
        },
        deferred,
    )
