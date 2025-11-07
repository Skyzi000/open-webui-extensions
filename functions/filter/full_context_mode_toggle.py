"""
title: Full Context Mode
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Toggle full context mode for all attached files at once. Sets the "context": "full" flag to leverage Open WebUI's native full context processing (RAG template with citations, etc.). Note: If the global RAG_FULL_CONTEXT setting is already enabled, this filter has no additional effect.
version: 1.0.0
license: MIT
"""

from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        """Set the “context”: “full” flag for all files."""
        files = body.get("metadata", {}).get("files", [])
        if not files:
            return body

        for file in files:
            file["context"] = "full"

        return body
