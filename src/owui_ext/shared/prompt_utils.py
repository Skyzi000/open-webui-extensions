"""Prompt-assembly helpers shared across owui_ext tool plugins."""

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
