"""
title: User Info Injector
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Injects user name and group information as a system message just before the user's latest prompt.
version: 1.0.0
license: MIT
"""

from pydantic import BaseModel, Field

from open_webui.models.groups import Groups


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
        pass

    class UserValves(BaseModel):
        excluded_groups: str = Field(
            default="",
            description="Comma-separated list of group names to exclude from display (e.g., 'internal,test,archive').",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        """Inject user name and group information as a system message before the user's latest prompt."""
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

                user_groups = Groups.get_groups_by_member_id(user_id)
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

        user_info_message = {
            "role": "system",
            "content": "\n".join(content_parts),
        }

        # Insert system message just before the last user message
        messages.insert(last_user_idx, user_info_message)
        body["messages"] = messages

        return body
