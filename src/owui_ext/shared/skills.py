"""Skill manifest / tag extraction shared by nested agent tools.

Open WebUI's skill subsystem injects two different shapes into the
parent conversation's system messages:

- **Available skills** are listed in a single ``<available_skills>``
  manifest block; their content is loaded lazily by ``view_skill``.
  Core v0.11.4+ can include accessible workspace and terminal skills,
  in addition to model-attached skills.
- **Inline skills** use ``<skill ... name="...">...</skill>`` blocks.

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
from urllib.parse import quote

from fastapi import Request

_skills_log = logging.getLogger("owui_ext.shared.skills")

_SKILL_READ_TOOLS = ("view_skill", "read_skill_file", "grep_skill_files")
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
    Inline skills may also appear as ``<skill>`` tags
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

    Args:
        messages: The parent conversation messages (``__messages__``).

    Returns:
        A list of ``<skill …>…</skill>`` strings, possibly empty.
    """
    return _extract_from_system_messages(messages, _find_skill_tags_in_text)


def extract_skill_ids(messages: Optional[list]) -> list[str]:
    """Return the IDs of the skills the parent lists or inlines, as Core's ``__skill_ids__``."""
    ids = set(re.findall(r"<id>(.*?)</id>", extract_skill_manifest(messages)))
    for tag in extract_user_skill_tags(messages):
        match = re.match(r'<skill (?:id="([^"]+)"|name="([^"]*)")', tag)
        if match:
            # Core inlines terminal skills without an id attribute.
            ids.add(match.group(1) or f'terminal:{quote(match.group(2), safe="")}')
    return sorted(ids)


async def _core_inline_skill_tags(
    skill_ids: list[str], request: Request, extra_params: dict
) -> Optional[list[str]]:
    """Render *skill_ids* as Core inlines them for a model without skill tools; None if Core cannot."""
    try:
        from fastapi import HTTPException
        from open_webui.models.skills import Skills, get_skill_snapshot
        from open_webui.utils.skill_files import format_skill_content, skill_overview
        from open_webui.utils.terminals import (
            format_terminal_skill_context,
            get_terminal_skill,
        )
    except ImportError:
        return None
    from urllib.parse import unquote

    user = extra_params.get("__user__") or {}
    metadata = extra_params.get("__metadata__") or {}
    context = metadata.get("chat_context") or {}
    metadata["chat_context"] = context
    skill_versions = context.setdefault("skill_versions", {})
    db_ids = [sid for sid in skill_ids if not sid.startswith("terminal:")]
    skills = (
        {skill.id: skill for skill in await Skills.get_skills(user_id=user.get("id"), ids=db_ids)}
        if db_ids
        else {}
    )
    tags = []
    for sid in db_ids:
        skill = skills.get(sid)
        if not skill or not skill.is_active:
            continue
        version_id = skill_versions.get(sid) or skill.version_id
        try:
            snapshot = await get_skill_snapshot(skill, version_id)
        except HTTPException as error:
            if error.status_code != 404:
                raise
            continue
        skill_versions[sid] = version_id
        # Core 0.12.1 and 0.12.2 define skill_overview synchronously.
        overview = await _skills_maybe_await(
            skill_overview(
                snapshot["content"],
                [f["path"] for f in snapshot["data"]["files"] if f["path"] != "SKILL.md"],
                tools_enabled=False,
            )
        )
        hint = (
            "\nSupporting files require a model with skill tools enabled."
            if overview.get("files")
            else ""
        )
        if overview.get("notice"):
            hint += "\n" + overview["notice"]
        tags.append(
            f'<skill id="{skill.id}" name="{skill.name}">\n'
            f"{format_skill_content(overview, skill.id, False)}{hint}\n</skill>"
        )
    if metadata.get("terminal_id"):
        for sid in skill_ids:
            if not sid.startswith("terminal:"):
                continue
            loaded = await get_terminal_skill(
                request,
                user,
                metadata,
                unquote(sid.removeprefix("terminal:")),
                extra_params,
                overview=True,
                tools_enabled=False,
            )
            if loaded:
                tags.append(format_terminal_skill_context(loaded, sid, False))
    return tags


async def add_skills(
    messages: list,
    skills: list[str],
    tools_dict: Optional[dict],
    request: Request,
    extra_params: dict,
) -> list:
    """Append the parent's *skills* to the system message, as Core adds them after the inlet filters.

    A child without Core's skill reading tools gets the chat's skills (``metadata["skill_ids"]``) as Core inlines
    them for a model without skill tools. When Core cannot render them, it keeps the parent's skills and drops
    the ``<available_skills>`` list unless a ``view_skill`` tool can load it.
    """
    if not skills:
        return messages
    tools_dict = tools_dict or {}
    if not any(tools_dict.get(name, {}).get("type") == "builtin" for name in _SKILL_READ_TOOLS):
        skill_ids = (extra_params.get("__metadata__") or {}).get("skill_ids") or []
        rendered = await _core_inline_skill_tags(skill_ids, request, extra_params)
        if rendered is not None:
            skills = rendered
        elif "view_skill" not in tools_dict:
            skills = [skill for skill in skills if not skill.startswith(_SKILLS_MANIFEST_START)]
    from open_webui.utils.misc import add_or_update_system_message

    for skill in skills:
        messages = add_or_update_system_message(skill, messages, append=True)
    return messages


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
