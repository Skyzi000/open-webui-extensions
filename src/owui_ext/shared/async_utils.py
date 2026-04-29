"""Tiny async helpers shared across plugins.

Open WebUI's core occasionally returns either a coroutine *or* a plain value
from the same call (e.g. ``Chats.get_message_by_id_and_message_id`` is sync
in tests but async in production). ``maybe_await`` accepts either and lets
callers stay agnostic.
"""


async def maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value
