"""
title: Current DateTime Injector
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Appends the current date/time to the user's latest message instead of using a built-in system prompt variable such as {{CURRENT_DATETIME}}, which changes the system prompt on every request. The aim is to avoid breaking prompt caching. A system role is available in Valves.
version: 1.1.0
license: MIT
"""

import re
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


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
            default=50,
            description="Priority level for the filter operations. Higher values run later.",
        )
        message_role: Literal["user", "system", "system_after"] = Field(
            default="user",
            description="Role and position of the injected message. 'user' appends it to the text of the latest user message, wrapped in <SYSTEM_CONTEXT> tags; nothing is injected when that message is a /skills:create command. 'system' inserts it as a system message just before the latest user message. 'system_after' inserts that system message right after the latest user message instead; standard Open WebUI then adds RAG context as a separate user message instead of adding it to the user's message, unless RAG_SYSTEM_CONTEXT is enabled. Open WebUI 0.8.11+ merges all system messages into one at the start of the conversation.",
        )
        pass

    class UserValves(BaseModel):
        timezone: str = Field(
            default="",
            description="Timezone for displaying current date/time (e.g., Asia/Tokyo, UTC, America/New_York). Leave empty to use server's local timezone.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        """Inject current date/time into or next to the user's latest message, depending on message_role."""
        messages = body.get("messages", [])
        if not messages:
            return body

        # Find the index of the last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return body

        # Get user valves
        user_valves = self.UserValves.model_validate(
            (__user__ or {}).get("valves", {})
        )

        # Get current date/time with timezone
        if user_valves.timezone:
            try:
                tz = ZoneInfo(user_valves.timezone)
                now = datetime.now(tz)
                tz_name = user_valves.timezone
            except Exception:
                now = datetime.now().astimezone()
                tz_name = str(now.tzinfo)
        else:
            now = datetime.now().astimezone()
            tz_name = str(now.tzinfo)

        weekday = now.strftime("%A")

        content = (
            f"# Current DateTime\n"
            f"{now.strftime('%Y-%m-%d')} ({weekday}) {now.strftime('%H:%M:%S')} ({tz_name})"
        )

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
