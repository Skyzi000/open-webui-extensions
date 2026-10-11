"""Skill manifest / tag extraction shared by nested agent tools.

Open WebUI's skill subsystem injects two different shapes into the
parent conversation's system messages depending on how a skill was
selected:

- **Available skills** are listed in a single ``<available_skills>``
  manifest block; their content is loaded lazily by ``view_skill``.
  Core v0.11.4+ can include accessible workspace and terminal skills,
  in addition to model-attached skills.
- **Inline skills** use full ``<skill ... name="...">...</skill>`` blocks
  (Core v0.12.0 adds ``id`` and ``version_id`` attributes);
  ``view_skill`` is unnecessary for their already-expanded content.
  Older Core versions inline user-selected skills. Core v0.11.4+ inlines
  mentioned skills, or selected skills when builtin tools are unavailable.

The plugins that spawn agent loops
need to read both shapes out of ``__messages__`` so the spawned agent
inherits the parent's skill context. Before Core v0.12.0 they also need
to register ``view_skill`` manually in their own ``tools_dict`` because
``get_builtin_tools`` only returns it to the core middleware.

Lives in its own shared dep (rather than ``shared.tool_servers`` or a
larger generic module) so plugins that don't use skills don't include
this code in their bundle; the inliner has no tree-shaking. It follows
the same shared-dep "no mixing external + cross-shared imports" rule
that already shaped ``shared.tool_execution`` / ``shared.mcp_tools``:
the tiny ``maybe_await`` helper is inlined privately rather than
imported from ``shared.async_utils``.
"""

import copy
import logging
import re
from typing import Any, Optional

from fastapi import Request

_skills_log = logging.getLogger("owui_ext.shared.skills")

_SKILLS_MANIFEST_START = "<available_skills>"
_SKILLS_MANIFEST_END = "</available_skills>"
_SKILL_TAG_PATTERN = re.compile(
    r"<skill [^\n]*?name=.*?>\n.*?\n</skill>", re.DOTALL
)


async def _skills_maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _find_manifest_in_text(text: str) -> str:
    """Return the <available_skills>…</available_skills> substring, or ""."""
    start = text.find(_SKILLS_MANIFEST_START)
    if start == -1:
        return ""
    end = text.find(_SKILLS_MANIFEST_END, start)
    if end == -1:
        return ""
    return text[start : end + len(_SKILLS_MANIFEST_END)]


def _find_skill_tags_in_text(text: str) -> list[str]:
    """Return all ``<skill ... name="...">…</skill>`` blocks found in *text*."""
    return _SKILL_TAG_PATTERN.findall(text)


def _extract_from_system_messages(
    messages: Optional[list],
    extractor,
):
    """Walk system messages and apply *extractor* to each text chunk.

    ``extractor`` is called with a single ``str`` argument and should return a
    list of results (or a single truthy result).  The function handles both
    plain-string content and list-of-parts content
    (``[{"type": "text", "text": "..."}]``).
    """
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
                results.append(found) if isinstance(found, str) else results.extend(
                    found
                )
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    found = extractor(part.get("text") or "")
                    if found:
                        results.append(found) if isinstance(
                            found, str
                        ) else results.extend(found)
    return results


def extract_skill_manifest(messages: Optional[list]) -> str:
    """Extract the ``<available_skills>`` manifest from the parent
    conversation's system messages.

    Core can list model-attached, accessible workspace, and terminal skills.
    Inline skills may also appear as full ``<skill>`` tags
    (see :func:`extract_user_skill_tags`).

    Args:
        messages: The parent conversation messages (``__messages__``).

    Returns:
        The manifest XML string, or empty string if not found.
    """
    results = _extract_from_system_messages(messages, _find_manifest_in_text)
    return results[0] if results else ""


def extract_user_skill_tags(messages: Optional[list]) -> list[str]:
    """Extract ``<skill name="...">content</skill>`` tags from the parent
    conversation's system messages.

    Mentioned skills, or selected skills on older Core / without builtin
    tools, can appear as ``<skill>`` tags alongside the available manifest.

    Args:
        messages: The parent conversation messages (``__messages__``).

    Returns:
        A list of ``<skill …>…</skill>`` strings, possibly empty.
    """
    return _extract_from_system_messages(messages, _find_skill_tags_in_text)


def drop_unloadable_skill_manifest(messages: list, tools_dict: Optional[dict]) -> list:
    """Remove the ``<available_skills>`` manifest when no ``view_skill`` tool can load it."""
    if "view_skill" in (tools_dict or {}):
        return messages

    def without_manifest(message: dict) -> dict:
        content = message.get("content")
        if message.get("role") != "system" or not isinstance(content, str):
            return message
        manifest = _find_manifest_in_text(content)
        if not manifest:
            return message
        parts = (part.strip() for part in content.split(manifest, 1))
        return {**message, "content": "\n\n".join(part for part in parts if part)}

    return [without_manifest(message) for message in messages]


def isolate_chat_context(metadata: Optional[dict]) -> dict:
    """Copy *metadata* with its own ``chat_context``, as Core does for sub-agent runs."""
    metadata = metadata or {}
    return {
        **metadata,
        "chat_context": copy.deepcopy(metadata.get("chat_context") or {}),
    }


async def register_view_skill(
    tools_dict: dict,
    request: Request,
    extra_params: dict,
) -> None:
    """Manually register the view_skill builtin tool in tools_dict.

    This is needed for available skills whose content is not injected
    inline. The agent loop can call ``view_skill`` to lazily load their
    content from the ``<available_skills>`` manifest.

    Skills already supplied as full ``<skill>`` tags do not require
    ``view_skill``; they are passed directly in the system message.

    Args:
        tools_dict: The tools dict to add view_skill to (modified in-place).
        request: FastAPI request object.
        extra_params: Extra parameters for tool binding.
    """
    if "view_skill" in tools_dict:
        return

    try:
        from open_webui.tools import builtin

        # Core 0.12.0+ registers skill tools in get_builtin_tools per model.
        if hasattr(builtin, "read_skill_file"):
            return
        view_skill = builtin.view_skill
        from open_webui.utils.tools import (
            get_async_tool_function_and_apply_extra_params,
            convert_function_to_pydantic_model,
            convert_pydantic_model_to_openai_function_spec,
        )

        callable_fn = await _skills_maybe_await(get_async_tool_function_and_apply_extra_params(
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
        _skills_log.warning(f"Failed to register view_skill: {e}")
