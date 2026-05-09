"""Open WebUI status-event emitter shared by tool plugins."""

from typing import Any, Callable, Optional


class EventEmitter:
    def __init__(self, event_emitter: Optional[Callable[[dict], Any]] = None):
        self.event_emitter = event_emitter

    async def emit(
        self,
        description: str = "Unknown state",
        status: str = "in_progress",
        done: bool = False,
    ) -> None:
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )
