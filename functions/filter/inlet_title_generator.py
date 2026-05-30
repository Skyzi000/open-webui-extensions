"""
title: Inlet Title Generator
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Starts Open WebUI-compatible chat title generation from inlet using the current user-message history when the chat title is still New Chat.
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


PLACEHOLDER_TITLE = "New Chat"
TITLE_GENERATION_TASK = "title_generation"
TEMP_CHAT_PREFIXES = ("local:", "channel:")


def is_regular_chat_id(chat_id: Any) -> bool:
    return isinstance(chat_id, str) and bool(chat_id) and not chat_id.startswith(TEMP_CHAT_PREFIXES)


def is_title_generation_task(task: Any) -> bool:
    return str(task) == TITLE_GENERATION_TASK


def user_model_from_dict(user: dict[str, Any]) -> Any:
    from open_webui.models.users import UserModel

    if isinstance(user, UserModel):
        return user
    return UserModel(**user)


def clone_request_for_title_generation(request: Any) -> Any:
    state = getattr(request, "state", None)
    metadata = getattr(state, "metadata", None) if state is not None else None

    if hasattr(request, "scope"):
        scope = dict(request.scope)
        scope_state = dict(scope.get("state") or {})
        if isinstance(metadata, dict):
            scope_state["metadata"] = dict(metadata)
        scope["state"] = scope_state

        from starlette.requests import Request

        receive = getattr(request, "_receive", None)
        if receive is not None:
            return Request(scope, receive)
        return Request(scope)

    cloned_request = copy.copy(request)
    if state is not None:
        cloned_state = copy.copy(state)
        if isinstance(metadata, dict):
            setattr(cloned_state, "metadata", dict(metadata))
        setattr(cloned_request, "state", cloned_state)
    return cloned_request


def get_content_from_message(message: dict[str, Any]) -> str | None:
    if isinstance(message.get("content"), list):
        for item in message["content"]:
            if item["type"] == "text":
                return item["text"]
    else:
        return message.get("content")
    return None


def get_last_user_message(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message["role"] == "user":
            return get_content_from_message(message)
    return None


def normalize_messages_for_title(message_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for message in message_list:
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    content = item["text"]
                    break

        if isinstance(content, str):
            content = re.sub(
                r"<details\b[^>]*>.*?<\/details>|!\[.*?\]\(.*?\)",
                "",
                content,
                flags=re.S | re.I,
            ).strip()

        messages.append(
            {
                **message,
                "role": message.get("role", "assistant"),
                "content": content,
            }
        )
    return messages


def extract_title_from_response(
    res: Any,
    messages: list[dict[str, Any]],
    message: dict[str, Any],
) -> Any | None:
    if not res or not isinstance(res, dict):
        return None

    user_message = get_last_user_message(messages)
    if user_message and len(user_message) > 100:
        user_message = user_message[:100] + "..."

    if len(res.get("choices", [])) == 1:
        response_message = res.get("choices", [])[0].get("message", {})

        title_string = (
            response_message.get("content")
            or response_message.get("reasoning_content")
            or message.get("content", user_message)
        )
    else:
        title_string = ""

    title_string = title_string[title_string.find("{") : title_string.rfind("}") + 1]

    try:
        title = json.loads(title_string).get("title", user_message)
    except Exception:
        title = ""

    if not title:
        title = messages[0].get("content", user_message)

    return title


def has_other_user_message(messages_map: dict[str, Any], user_message_id: str) -> bool:
    for message_id, message in messages_map.items():
        if message_id == user_message_id or not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return True
    return False


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=100,
            description="Priority level for the filter operations. Higher values run later.",
        )
        dedup_time_window: int = Field(
            default=600,
            description="Time window in seconds for distributed and in-memory title generation deduplication.",
        )
        dedup_max_entries: int = Field(
            default=1000,
            description="Maximum number of in-memory deduplication entries to keep.",
        )
        avoid_core_initial_title_duplicate: bool = Field(
            default=True,
            description="Skip root new-chat requests that Open WebUI core is expected to title automatically. Disable this when core auto title is disabled or duplicate generation is acceptable.",
        )
        use_redis: bool = Field(
            default=(os.environ.get("WEBSOCKET_MANAGER", "").lower() == "redis"),
            description="Enable Redis for distributed title generation deduplication. Defaults to True when WEBSOCKET_MANAGER=redis.",
        )
        redis_url: str = Field(
            default=os.environ.get("WEBSOCKET_REDIS_URL", os.environ.get("REDIS_URL", "")),
            description="Redis connection URL for distributed title generation deduplication. Defaults to Open WebUI's WEBSOCKET_REDIS_URL, then REDIS_URL.",
        )
        redis_key_prefix: str = Field(
            default="",
            description="Base prefix for Redis keys. Empty auto-generates owui:{filter_id}.",
        )
        debug_print: bool = Field(
            default=False,
            description="Print debug logs to the server console.",
        )
        pass

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable inlet title generation for this user.",
        )
        show_status: bool = Field(
            default=False,
            description="Show status events when background title generation starts, completes, or fails.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self._active_chats: set[str] = set()
        self._active_lock = asyncio.Lock()
        self._processed_titles: dict[tuple[str, str], float] = {}
        self._processed_lock = asyncio.Lock()
        self._redis_client = None
        self._redis_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task] = set()

    def _launch_background_task(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _cleanup_processed_titles(self) -> None:
        current_time = time.time()
        cutoff = current_time - self.valves.dedup_time_window

        self._processed_titles = {
            key: timestamp
            for key, timestamp in self._processed_titles.items()
            if timestamp > cutoff
        }

        if len(self._processed_titles) > self.valves.dedup_max_entries:
            sorted_items = sorted(self._processed_titles.items(), key=lambda item: item[1])
            self._processed_titles = dict(sorted_items[-self.valves.dedup_max_entries :])

    async def _get_redis_client(self):
        if not self.valves.redis_url:
            return None

        if self._redis_client is not None:
            return self._redis_client

        async with self._redis_lock:
            if self._redis_client is not None:
                return self._redis_client

            try:
                from redis import asyncio as aioredis

                client = aioredis.from_url(self.valves.redis_url, decode_responses=False)
                await client.ping()
                self._redis_client = client
                return self._redis_client
            except Exception as e:
                if self.valves.debug_print:
                    print(f"Inlet Title Generator Redis unavailable: {e}")
                self._redis_client = None
                return None

    async def _acquire_dedup(self, chat_id: str, user_message_id: str, filter_id: str | None) -> bool:
        if self.valves.use_redis and self.valves.redis_url:
            base_prefix = self.valves.redis_key_prefix or f"owui:{filter_id or 'inlet_title_generator'}"
            redis_key = f"{base_prefix}:title_dedup:{chat_id}:{user_message_id}"
            redis_client = await self._get_redis_client()
            if redis_client:
                try:
                    return bool(
                        await redis_client.set(
                            redis_key,
                            b"1",
                            ex=self.valves.dedup_time_window,
                            nx=True,
                        )
                    )
                except Exception as e:
                    if self.valves.debug_print:
                        print(f"Inlet Title Generator Redis dedup failed, falling back to memory: {e}")

        key = (chat_id, user_message_id)
        async with self._processed_lock:
            self._cleanup_processed_titles()
            if key in self._processed_titles:
                return False
            self._processed_titles[key] = time.time()
            self._cleanup_processed_titles()
            return True

    async def _emit_status(
        self,
        event_emitter: Callable[[Any], Awaitable[None]] | None,
        user_valves: "Filter.UserValves",
        description: str,
        done: bool,
    ) -> None:
        if not user_valves.show_status or event_emitter is None:
            return
        await event_emitter({"type": "status", "data": {"description": description, "done": done}})

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]] | None = None,
        __user__: dict | None = None,
        __metadata__: dict | None = None,
        __id__: str | None = None,
        __task__: Any = None,
        __request__: Any = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body

        user_valves = self.UserValves.model_validate((__user__ or {}).get("valves", {}))
        if not user_valves.enabled:
            return body

        metadata = __metadata__ if isinstance(__metadata__, dict) else body.get("metadata", {})
        if not isinstance(metadata, dict):
            return body

        if is_title_generation_task(__task__) or is_title_generation_task(metadata.get("task")):
            return body

        chat_id = metadata.get("chat_id")
        user_message_id = metadata.get("user_message_id")
        model_id = body.get("model")
        messages = body.get("messages")

        if (
            __request__ is None
            or __event_emitter__ is None
            or not __user__
            or not is_regular_chat_id(chat_id)
            or not isinstance(user_message_id, str)
            or not user_message_id
            or not isinstance(model_id, str)
            or not model_id
            or not isinstance(messages, list)
        ):
            return body

        async with self._active_lock:
            if chat_id in self._active_chats:
                return body
            self._active_chats.add(chat_id)

        try:
            self._launch_background_task(
                self._run_title_generation(
                    request=__request__,
                    event_emitter=__event_emitter__,
                    user=__user__,
                    chat_id=chat_id,
                    user_message_id=user_message_id,
                    model_id=model_id,
                    filter_id=__id__,
                    user_valves=user_valves,
                )
            )
        except Exception:
            async with self._active_lock:
                self._active_chats.discard(chat_id)
            raise

        return body

    async def _run_title_generation(
        self,
        *,
        request: Any,
        event_emitter: Callable[[Any], Awaitable[None]] | None,
        user: dict[str, Any],
        chat_id: str,
        user_message_id: str,
        model_id: str,
        filter_id: str | None,
        user_valves: "Filter.UserValves",
    ) -> None:
        try:
            await self._emit_status(event_emitter, user_valves, "Generating chat title...", False)

            if not await self._acquire_dedup(chat_id, user_message_id, filter_id):
                await self._emit_status(event_emitter, user_valves, "Skipped duplicate title generation.", True)
                return

            from open_webui.models.chats import Chats
            from open_webui.routers.tasks import generate_title
            from open_webui.utils.misc import get_message_list

            title = await Chats.get_chat_title_by_id(chat_id)
            if title != PLACEHOLDER_TITLE:
                await self._emit_status(event_emitter, user_valves, "Skipped title generation; title already exists.", True)
                return

            messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
            if not isinstance(messages_map, dict) or not messages_map:
                await self._emit_status(event_emitter, user_valves, "Skipped title generation; message history is unavailable.", True)
                return

            current_message = messages_map.get(user_message_id)
            if not isinstance(current_message, dict) or current_message.get("role") != "user":
                await self._emit_status(event_emitter, user_valves, "Skipped title generation; current user message is unavailable.", True)
                return

            is_root_user_message = not current_message.get("parentId")
            if (
                self.valves.avoid_core_initial_title_duplicate
                and is_root_user_message
                and not has_other_user_message(messages_map, user_message_id)
            ):
                await self._emit_status(event_emitter, user_valves, "Skipped core initial title generation duplicate.", True)
                return

            message_list = get_message_list(messages_map, user_message_id)
            if not message_list:
                await self._emit_status(event_emitter, user_valves, "Skipped title generation; message chain is unavailable.", True)
                return
            messages = normalize_messages_for_title(message_list)
            title_message = messages[-1]

            user_model = user_model_from_dict(user)
            title_request = clone_request_for_title_generation(request)
            res = await generate_title(
                title_request,
                {
                    "model": model_id,
                    "messages": messages,
                    "chat_id": chat_id,
                },
                user_model,
            )
            generated_title = extract_title_from_response(res, messages, title_message)
            if generated_title is None:
                await self._emit_status(event_emitter, user_valves, "Skipped title generation; no title was generated.", True)
                return

            latest_title = await Chats.get_chat_title_by_id(chat_id)
            if latest_title != PLACEHOLDER_TITLE:
                await self._emit_status(event_emitter, user_valves, "Skipped title update; title changed before save.", True)
                return

            await Chats.update_chat_title_by_id(chat_id, generated_title)
            if event_emitter is not None:
                await event_emitter({"type": "chat:title", "data": generated_title})
            await self._emit_status(event_emitter, user_valves, "Chat title generated.", True)
        except Exception as e:
            if self.valves.debug_print:
                print(f"Inlet Title Generator failed: {e}")
            await self._emit_status(event_emitter, user_valves, "Title generation failed.", True)
        finally:
            async with self._active_lock:
                self._active_chats.discard(chat_id)
