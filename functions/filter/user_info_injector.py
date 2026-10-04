"""
title: User Info Injector
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Appends user name and group information to the user's latest message instead of using the built-in system prompt variables. The aim is to avoid breaking prompt caching. A system role is available in Valves.
version: 1.1.0
license: MIT
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from open_webui.models.groups import Groups


async def maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


# Same match as Open WebUI's /skills:create command, which replaces the whole user message
SKILLS_CREATE_RE = re.compile(r"/skills:create(?:\s|$)", re.IGNORECASE)


def is_skills_create_command(message: dict) -> bool:
    content = message.get("content")
    if isinstance(content, list):
        content = next(
            (part.get("text") for part in content if isinstance(part, dict) and part.get("type") == "text"),
            None,
        )
    return isinstance(content, str) and bool(SKILLS_CREATE_RE.match(content.strip()))


def with_appended_text(message: dict, addition: str) -> dict:
    """Copy of the message with addition appended to its first text part, the one Open WebUI reads as the prompt."""
    content = message.get("content")
    if not isinstance(content, list):
        return {**message, "content": f"{content}\n\n{addition}" if content else addition}
    index = next(
        (i for i, part in enumerate(content) if isinstance(part, dict) and part.get("type") == "text"),
        None,
    )
    if index is None:
        return {**message, "content": [*content, {"type": "text", "text": addition}]}
    text = content[index].get("text")
    part = {**content[index], "text": f"{text}\n\n{addition}" if text else addition}
    return {**message, "content": [*content[:index], part, *content[index + 1 :]]}


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=9,
            description="Priority level for the filter operations. Higher values run later.",
        )
        include_role: bool = Field(
            default=True,
            description="Include user role in the message.",
        )
        include_groups: bool = Field(
            default=True,
            description="Include user groups in the message.",
        )
        message_role: Literal["user", "system", "system_after"] = Field(
            default="user",
            description="Role and position of the injected message. 'user' appends it to the text of the latest user message, wrapped in <SYSTEM_CONTEXT> tags; nothing is injected when that message is a /skills:create command. 'system' inserts it as a system message just before the latest user message. 'system_after' inserts that system message right after the latest user message instead; standard Open WebUI then adds RAG context as a separate user message instead of adding it to the user's message, unless RAG_SYSTEM_CONTEXT is enabled. Open WebUI 0.8.11+ merges all system messages into one at the start of the conversation.",
        )
        pass

    class UserValves(BaseModel):
        excluded_groups: str = Field(
            default="",
            description="Comma-separated list of group names to exclude from display (e.g., 'internal,test,archive').",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        """Inject user name and group information into or next to the user's latest message, depending on message_role."""
        messages = body.get("messages", [])
        if not messages:
            return body

        # If no user information available, return unchanged
        if not __user__:
            return body

        # Find the index of the last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return body

        # Get user information
        user_name = __user__.get("name") or "Unknown User"
        user_id = __user__.get("id")
        user_role = __user__.get("role")

        # Build content parts
        content_parts = [f"# Current User", f"Name: {user_name}"]

        # Add role if enabled
        if self.valves.include_role and user_role:
            content_parts.append(f"Role: {user_role}")

        # Add groups if enabled
        if self.valves.include_groups and user_id:
            try:
                # Get user valves
                user_valves = self.UserValves.model_validate(
                    (__user__ or {}).get("valves", {})
                )

                # Get excluded groups as set
                excluded_groups_set = set()
                if user_valves.excluded_groups:
                    excluded_groups_set = {
                        g.strip() for g in user_valves.excluded_groups.split(",") if g.strip()
                    }

                user_groups = await maybe_await(Groups.get_groups_by_member_id(user_id))
                if user_groups:
                    # Filter out excluded groups
                    group_names = [
                        group.name for group in user_groups
                        if group.name not in excluded_groups_set
                    ]
                    if group_names:
                        content_parts.append(f"Groups: {', '.join(group_names)}")
                    else:
                        content_parts.append("Groups: None")
                else:
                    content_parts.append("Groups: None")
            except Exception:
                # If group retrieval fails, show N/A
                content_parts.append("Groups: N/A")

        content = "\n".join(content_parts)

        if self.valves.message_role == "system":
            messages.insert(last_user_idx, {"role": "system", "content": content})
        elif self.valves.message_role == "system_after":
            messages.insert(last_user_idx + 1, {"role": "system", "content": content})
        elif not is_skills_create_command(messages[last_user_idx]):
            messages[last_user_idx] = with_appended_text(
                messages[last_user_idx], f"<SYSTEM_CONTEXT>\n{content}\n</SYSTEM_CONTEXT>"
            )
        body["messages"] = messages

        return body
