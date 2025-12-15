"""
title: Current DateTime Injector
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Injects current date/time as a system message just before the user's latest prompt. This preserves prompt caching for the main system prompt while still providing dynamic time information.
version: 1.0.1
license: MIT
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=50,
            description="Priority level for the filter operations. Higher values run later.",
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
        """Inject current date/time as a system message before the user's latest prompt."""
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

        datetime_message = {
            "role": "system",
            "content": (
                f"# Current DateTime\n"
                f"{now.strftime('%Y-%m-%d')} ({weekday}) {now.strftime('%H:%M:%S')} ({tz_name})"
            ),
        }

        # Insert system message just before the last user message
        messages.insert(last_user_idx, datetime_message)
        body["messages"] = messages

        return body
