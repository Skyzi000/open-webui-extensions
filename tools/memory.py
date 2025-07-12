"""
title: Memory
author: https://github.com/skyzi000
version: 0.0.3
license: MIT

This tool supports a complete experience when using OpenAI API
(and any API fully compatible with OpenAI API format) or Gemini models
in native Function Calling mode.

If the API format is not supported, you can still use the default
Function Calling mode, but the experience will be significantly reduced.

This tool is an improved version of https://openwebui.com/t/cooksleep/memory,
fully utilizing Open WebUI's native memory functionality.

You don't need to enable the memory switch,
as this tool only requires access to its database.

Based on https://openwebui.com/t/cooksleep/memory
  (MIT License)
  
Special thanks to cooksleep for the original implementation.
Original concept by mhio: https://openwebui.com/t/mhio/met

# Changelog
0.0.3:Added confirmation dialog when updating memory
0.0.2:Added confirmation dialog when deleting memory
"""

import json
from typing import Callable, Any, List

from open_webui.models.memories import Memories # type: ignore
from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None): # type: ignore
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown state", status="in_progress", done=False):
        """
        Send a status event to the event emitter.

        :param description: Event description
        :param status: Event status
        :param done: Whether the event is complete
        """
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


# Pydantic model for memory update operations
class MemoryUpdate(BaseModel):
    index: int = Field(..., description="Index of the memory entry (1-based)")
    content: str = Field(..., description="Updated content for the memory")


class Tools:
    """
    Memory

    Use this tool to autonomously save/modify/query memories across conversations.
    
    IMPORTANT: Users rarely explicitly tell you what to remember!
    You must actively observe and identify important information that should be stored.
    
    Key features:
    1. Proactive memory creation: Identify user preferences, project context, and recurring patterns
    2. Intelligent memory usage: Reference stored information without requiring users to repeat themselves
    3. Best practices: Store valuable information, maintain relevance, provide memories at appropriate times
    4. Language matching: Always create memories in the user's preferred language and writing style
    """

    class Valves(BaseModel):
        USE_MEMORY: bool = Field(
            default=True, description="Enable or disable memory usage."
        )
        CONFIRMATION_TIMEOUT: int = Field(
            default=60,
            description="Timeout in seconds for confirmation dialogs (default: 60).",
        )
        DEBUG: bool = Field(default=True, description="Enable or disable debug mode.")

    def __init__(self):
        """Initialize the memory management tool."""
        self.valves = self.Valves()

    async def recall_memories(
        self, __user__: dict = None, __event_emitter__: Callable[[dict], Any] = None # type: ignore
    ) -> str:
        """
        Retrieves all stored memories from the user's memory vault.

        IMPORTANT: Proactively check memories to enhance your responses!
        Don't wait for users to ask what you remember.

        Returns memories in chronological order with index numbers.
        Use when you need to check stored information, reference previous
        preferences, or build context for responses.

        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter for tracking status
        :return: JSON string with indexed memories list
        """
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        await emitter.emit(
            description="Retrieving stored memories.",
            status="recall_in_progress",
            done=False,
        )

        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "No memory stored."
            await emitter.emit(description=message, status="recall_complete", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        content_list = [
            f"{index}. {memory.content}"
            for index, memory in enumerate(
                sorted(user_memories, key=lambda m: m.created_at), start=1
            )
        ]

        await emitter.emit(
            description=f"{len(user_memories)} memories loaded",
            status="recall_complete",
            done=True,
        )

        return f"Memories from the users memory vault: {content_list}"

    async def add_memory(
        self,
        input_text: List[
            str
        ],  # Modified to only accept list, JSON Schema items.type is string
        __user__: dict = None, # type: ignore
        __event_emitter__: Callable[[dict], Any] = None, # type: ignore
    ) -> str:
        """
        Adds one or more memories to the user's memory vault.

        IMPORTANT: Users rarely explicitly tell you what to remember!
        You must actively observe and identify important information that should be stored.

        Good candidates for memories:
        - Personal preferences (favorite topics, entertainment, colors)
        - Professional information (field of expertise, current projects)
        - Important relationships (family, pets, close friends)
        - Recurring needs or requests (common questions, regular workflows)
        - Learning goals and interests (topics they're studying, skills they want to develop)

        Always use the user's preferred language and writing style.

        Memories should start with "User", for example (English):
        - "User likes blue"
        - "User is a software engineer"
        - "User has a golden retriever named Max"

        :param input_text: Single memory string or list of memory strings to store
        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter for tracking status
        :return: JSON string with result message
        """
        emitter = EventEmitter(__event_emitter__)
        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Handle single string input if needed
        if isinstance(input_text, str):
            input_text = [input_text]

        await emitter.emit(
            description="Adding entries to the memory vault.",
            status="add_in_progress",
            done=False,
        )

        # Process each memory item
        added_items = []
        failed_items = []

        for item in input_text:
            new_memory = Memories.insert_new_memory(user_id, item)
            if new_memory:
                added_items.append(item)
            else:
                failed_items.append(item)

        if not added_items:
            message = "Failed to add any memories."
            await emitter.emit(description=message, status="add_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Prepare result message
        added_count = len(added_items)
        failed_count = len(failed_items)

        if failed_count > 0:
            message = (
                f"Added {added_count} memories, failed to add {failed_count} memories."
            )
        else:
            message = f"Successfully added {added_count} memories."

        await emitter.emit(
            description=message,
            status="add_complete",
            done=True,
        )
        return json.dumps({"message": message}, ensure_ascii=False)

    # async def delete_memory(
    #     self,
    #     indices: List[int],  # Modified to only accept list, items.type is integer
    #     __user__: dict = None,
    #     __event_emitter__: Callable[[dict], Any] = None,
    # ) -> str:
    #     """
    #     Delete one or more memory entries from the user's memory vault.

    #     Use to remove outdated or incorrect memories.

    #     For single deletion: provide an integer index
    #     For multiple deletions: provide a list of integer indices

    #     Indices refer to the position in the sorted list (1-based).

    #     :param indices: Single index (int) or list of indices to delete
    #     :param __user__: User dictionary containing the user ID
    #     :param __event_emitter__: Optional event emitter
    #     :return: JSON string with result message
    #     """
    #     emitter = EventEmitter(__event_emitter__)

    #     if not __user__:
    #         message = "User ID not provided."
    #         await emitter.emit(description=message, status="missing_user_id", done=True)
    #         return json.dumps({"message": message}, ensure_ascii=False)

    #     user_id = __user__.get("id")
    #     if not user_id:
    #         message = "User ID not provided."
    #         await emitter.emit(description=message, status="missing_user_id", done=True)
    #         return json.dumps({"message": message}, ensure_ascii=False)

    #     # Handle single integer input if needed
    #     if isinstance(indices, int):
    #         indices = [indices]

    #     await emitter.emit(
    #         description=f"Deleting {len(indices)} memory entries.",
    #         status="delete_in_progress",
    #         done=False,
    #     )

    #     # Get all memories for this user
    #     user_memories = Memories.get_memories_by_user_id(user_id)
    #     if not user_memories:
    #         message = "No memories found to delete."
    #         await emitter.emit(description=message, status="delete_failed", done=True)
    #         return json.dumps({"message": message}, ensure_ascii=False)

    #     sorted_memories = sorted(user_memories, key=lambda m: m.created_at)
    #     responses = []

    #     for index in indices:
    #         if index < 1 or index > len(sorted_memories):
    #             message = f"Memory index {index} does not exist."
    #             responses.append(message)
    #             await emitter.emit(
    #                 description=message, status="delete_failed", done=False
    #             )
    #             continue

    #         # Get the memory by index (1-based index)
    #         memory_to_delete = sorted_memories[index - 1]

    #         # Delete the memory
    #         result = Memories.delete_memory_by_id_and_user_id(
    #             memory_to_delete.id, user_id
    #         )
    #         if not result:
    #             message = f"Failed to delete memory at index {index}."
    #             responses.append(message)
    #             await emitter.emit(
    #                 description=message, status="delete_failed", done=False
    #             )
    #         else:
    #             message = f"Memory at index {index} deleted successfully."
    #             responses.append(message)
    #             await emitter.emit(
    #                 description=message, status="delete_success", done=False
    #             )

    #     await emitter.emit(
    #         description="All requested memory deletions have been processed.",
    #         status="delete_complete",
    #         done=True,
    #     )
    #     return json.dumps({"message": "\n".join(responses)}, ensure_ascii=False)

    async def delete_memory(
        self,
        indices: List[int],  # Modified to only accept list, items.type is integer
        __user__: dict = None, # type: ignore
        __event_emitter__: Callable[[dict], Any] = None, # type: ignore
        __event_call__: Callable[[dict], Any] = None,  # Added for confirmation dialog # type: ignore
    ) -> str:
        """
        Delete one or more memory entries from the user's memory vault.

        Use to remove outdated or incorrect memories.

        For single deletion: provide an integer index
        For multiple deletions: provide a list of integer indices

        Indices refer to the position in the sorted list (1-based).
        
        IMPORTANT NOTE ON CLEARING MEMORIES:
        If a user asks to clear all memories, DO NOT attempt to implement this via code.
        Instead, inform them that clearing all memories is a high-risk operation that
        should be performed through their personal account settings panel using the
        Clear All Memories button. This prevents accidental data loss.

        :param indices: Single index (int) or list of indices to delete
        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter
        :param __event_call__: Optional event caller for confirmation dialogs
        :return: JSON string with result message and deleted content details
        """
        import asyncio

        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Handle single integer input if needed
        if isinstance(indices, int):
            indices = [indices]

        # Get all memories for this user
        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "削除するメモリが見つかりません。"
            await emitter.emit(description=message, status="delete_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        sorted_memories = sorted(user_memories, key=lambda m: m.created_at)

        # Validate indices before showing confirmation
        invalid_indices = []
        valid_indices = []
        for index in indices:
            if index < 1 or index > len(sorted_memories):
                invalid_indices.append(index)
            else:
                valid_indices.append(index)

        if invalid_indices:
            message = f"指定された無効なインデックス: {invalid_indices}. 有効な範囲は 1-{len(sorted_memories)} です。"
            await emitter.emit(description=message, status="delete_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Show confirmation dialog if __event_call__ is available
        if __event_call__:
            # Debug: Log how many memories are being processed
            if self.valves.DEBUG:
                await emitter.emit(
                    description=f"削除対象: {len(valid_indices)}個のメモリ（インデックス: {valid_indices}）",
                    status="debug",
                    done=False,
                )

            # Prepare preview of memories to be deleted
            preview_items = []
            for i, index in enumerate(valid_indices, 1):
                memory_content = sorted_memories[index - 1].content
                # Clean up memory content: remove newlines and extra whitespace
                memory_content = (
                    memory_content.replace("\n", " ").replace("\r", " ").strip()
                )
                # Remove multiple spaces
                memory_content = " ".join(memory_content.split())

                # Limit preview length to avoid overwhelming UI
                if len(memory_content) > 80:
                    memory_content = memory_content[:80] + "..."

                preview_items.append(f"[{i}] (インデックス:{index}):  \n{memory_content}")

            # Join with double newlines for better separation
            preview_text = "  \n".join(preview_items)

            confirmation_message = f"""以下のメモリを削除しますか？  
  
{preview_text}  
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
⚠️ この操作は取り消すことができません。  
⏰ {self.valves.CONFIRMATION_TIMEOUT}秒以内に選択しないと自動的にキャンセルされます。"""

            try:
                # Add timeout to prevent hanging indefinitely

                confirmation_task = __event_call__(
                    {
                        "type": "confirmation",
                        "data": {
                            "title": "メモリの削除確認",
                            "message": confirmation_message,
                        },
                    }
                )

                # Wait for confirmation with timeout
                try:
                    result = await asyncio.wait_for(
                        confirmation_task, timeout=self.valves.CONFIRMATION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    # Handle timeout case
                    timeout_message = f"ユーザーが{self.valves.CONFIRMATION_TIMEOUT}秒以内に操作しなかったため、タイムアウトしました。削除をキャンセルします。"
                    await emitter.emit(
                        description=timeout_message,
                        status="delete_timeout",
                        done=True,
                    )
                    return json.dumps({"message": timeout_message}, ensure_ascii=False)

                # Check result (False = cancel, None/undefined also treated as cancel)
                if not result:
                    # User cancelled the operation or closed dialog
                    cancel_message = "ユーザーがメモリの削除をキャンセルしました。"
                    await emitter.emit(
                        description=cancel_message, status="delete_cancelled", done=True
                    )
                    return json.dumps({"message": cancel_message}, ensure_ascii=False)

            except Exception as e:
                # If confirmation fails, ask user what to do
                error_message = f"確認ダイアログでエラーが発生しました（{e}）。安全のため削除をキャンセルします。"
                await emitter.emit(
                    description=error_message,
                    status="confirmation_error",
                    done=True,
                )
                return json.dumps({"message": error_message}, ensure_ascii=False)

        await emitter.emit(
            description=f"{len(valid_indices)}個のメモリエントリを削除中...",
            status="delete_in_progress",
            done=False,
        )

        responses = []
        deleted_memories = []  # Store deleted memory contents for the return value
        successful_deletions = 0
        failed_deletions = 0

        for index in valid_indices:
            # Get the memory by index (1-based index)
            memory_to_delete = sorted_memories[index - 1]
            memory_content = memory_to_delete.content

            # Delete the memory
            result = Memories.delete_memory_by_id_and_user_id(
                memory_to_delete.id, user_id
            )
            if not result:
                message = f"インデックス {index} のメモリ削除に失敗しました。"
                responses.append(message)
                failed_deletions += 1
                await emitter.emit(
                    description=message, status="delete_failed", done=False
                )
            else:
                message = f"インデックス {index} のメモリが正常に削除されました。"
                responses.append(message)
                successful_deletions += 1
                # Record the deleted memory content (full content, no truncation)
                deleted_memories.append({"index": index, "content": memory_content})
                await emitter.emit(
                    description=message, status="delete_success", done=False
                )

        await emitter.emit(
            description="すべてのメモリ削除処理が完了しました。",
            status="delete_complete",
            done=True,
        )

        # Prepare detailed result including deleted memory contents
        result_data = {
            "message": "\n".join(responses),
            "summary": f"指定された{len(valid_indices)}個のメモリのうち、{successful_deletions}個を正常に削除、{failed_deletions}個の削除に失敗",
            "deleted_memories": deleted_memories,
            "total_deleted": successful_deletions,
            "total_failed": failed_deletions,
        }

        return json.dumps(result_data, ensure_ascii=False)

    async def update_memory(
        self,
        updates: List[
            MemoryUpdate
        ],  # Modified to accept list of MemoryUpdate objects, items.type is object
        __user__: dict = None,  # type: ignore
        __event_emitter__: Callable[[dict], Any] = None,  # type: ignore
        __event_call__: Callable[[dict], Any] = None,  # Added for confirmation dialog # type: ignore
    ) -> str:
        """
        Update one or more memory entries in the user's memory vault.

        Use to modify existing memories when information changes.

        For single update: provide a dict with 'index' and 'content' keys
        For multiple updates: provide a list of dicts with 'index' and 'content' keys

        The 'index' refers to the position in the sorted list (1-based).

        Common scenarios: Correcting information, adding details,
        updating preferences, or refining wording.

        :param updates: Dict with 'index' and 'content' keys OR a list of such dicts
        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter
        :param __event_call__: Optional event caller for confirmation dialogs
        :return: JSON string with result message and update details
        """
        import asyncio

        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Get all memories for this user
        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "更新するメモリが見つかりません。"
            await emitter.emit(description=message, status="update_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        sorted_memories = sorted(user_memories, key=lambda m: m.created_at)

        # Validate updates and prepare for confirmation
        valid_updates = []
        invalid_indices = []

        for update_item in updates:
            # Convert dict to MemoryUpdate object if needed
            if isinstance(update_item, dict):
                try:
                    update_item = MemoryUpdate.parse_obj(update_item)
                except Exception as e:
                    invalid_indices.append(f"Invalid update format: {update_item}")
                    continue

            index = update_item.index
            if index < 1 or index > len(sorted_memories):
                invalid_indices.append(index)
                continue

            valid_updates.append(update_item)

        if invalid_indices:
            message = f"指定された無効なインデックス: {invalid_indices}. 有効な範囲は 1-{len(sorted_memories)} です。"
            await emitter.emit(description=message, status="update_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Show confirmation dialog if __event_call__ is available
        if __event_call__:
            # Debug: Log how many memories are being processed
            if self.valves.DEBUG:
                update_indices = [update.index for update in valid_updates]
                await emitter.emit(
                    description=f"更新対象: {len(valid_updates)}個のメモリ（インデックス: {update_indices}）",
                    status="debug",
                    done=False,
                )

            # Prepare preview of memory updates
            preview_items = []
            for i, update_item in enumerate(valid_updates, 1):
                index = update_item.index
                new_content = update_item.content
                old_content = sorted_memories[index - 1].content

                # Clean up content: remove newlines and extra whitespace
                old_content_clean = (
                    old_content.replace("\n", " ").replace("\r", " ").strip()
                )
                new_content_clean = (
                    new_content.replace("\n", " ").replace("\r", " ").strip()
                )
                old_content_clean = " ".join(old_content_clean.split())
                new_content_clean = " ".join(new_content_clean.split())

                # Limit preview length
                if len(old_content_clean) > 60:
                    old_content_clean = old_content_clean[:60] + "..."
                if len(new_content_clean) > 60:
                    new_content_clean = new_content_clean[:60] + "..."

                # Use more explicit formatting with separators
                preview_items.append(f"[{i}] インデックス{index}:")
                preview_items.append(f"    変更前: {old_content_clean}")
                preview_items.append(f"    変更後: {new_content_clean}")
                preview_items.append("\n")  # Empty line for separation

            # Join with single newlines since we're adding explicit empty lines
            preview_text = "  \n".join(preview_items).rstrip()  # Remove final empty line

            confirmation_message = f"""以下のメモリを更新しますか？  
  
{preview_text}  
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
⚠️ この操作は元の内容を上書きします。  
⏰ {self.valves.CONFIRMATION_TIMEOUT}秒以内に選択しないと自動的にキャンセルされます。"""

            try:
                # Add timeout to prevent hanging indefinitely
                confirmation_task = __event_call__(
                    {
                        "type": "confirmation",
                        "data": {
                            "title": "メモリの更新確認",
                            "message": confirmation_message,
                        },
                    }
                )

                # Wait for confirmation with timeout
                try:
                    result = await asyncio.wait_for(
                        confirmation_task, timeout=self.valves.CONFIRMATION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    # Handle timeout case
                    timeout_message = f"ユーザーが{self.valves.CONFIRMATION_TIMEOUT}秒以内に操作しなかったため、タイムアウトしました。更新をキャンセルします。"
                    await emitter.emit(
                        description=timeout_message,
                        status="update_timeout",
                        done=True,
                    )
                    return json.dumps({"message": timeout_message}, ensure_ascii=False)

                # Check result (False = cancel, None/undefined also treated as cancel)
                if not result:
                    # User cancelled the operation or closed dialog
                    cancel_message = "ユーザーがメモリの更新をキャンセルしました。"
                    await emitter.emit(
                        description=cancel_message, status="update_cancelled", done=True
                    )
                    return json.dumps({"message": cancel_message}, ensure_ascii=False)

            except Exception as e:
                # If confirmation fails, cancel for safety
                error_message = f"確認ダイアログでエラーが発生しました（{e}）。安全のため更新をキャンセルします。"
                await emitter.emit(
                    description=error_message,
                    status="confirmation_error",
                    done=True,
                )
                return json.dumps({"message": error_message}, ensure_ascii=False)

        await emitter.emit(
            description=f"{len(valid_updates)}個のメモリエントリを更新中...",
            status="update_in_progress",
            done=False,
        )

        responses = []
        updated_memories = []  # Store updated memory details for the return value
        successful_updates = 0
        failed_updates = 0

        for update_item in valid_updates:
            index = update_item.index
            content = update_item.content

            # Get the memory by index (1-based index)
            memory_to_update = sorted_memories[index - 1]
            old_content = memory_to_update.content

            # Update the memory
            updated_memory = Memories.update_memory_by_id_and_user_id(
                memory_to_update.id, user_id, content
            )
            if not updated_memory:
                message = f"インデックス {index} のメモリ更新に失敗しました。"
                responses.append(message)
                failed_updates += 1
                await emitter.emit(
                    description=message, status="update_failed", done=False
                )
            else:
                message = f"インデックス {index} のメモリが正常に更新されました。"
                responses.append(message)
                successful_updates += 1
                # Record the updated memory details (full content, no truncation)
                updated_memories.append(
                    {"index": index, "old_content": old_content, "new_content": content}
                )
                await emitter.emit(
                    description=message, status="update_success", done=False
                )

        await emitter.emit(
            description="すべてのメモリ更新処理が完了しました。",
            status="update_complete",
            done=True,
        )

        # Prepare detailed result including updated memory contents
        result_data = {
            "message": "\n".join(responses),
            "summary": f"指定された{len(valid_updates)}個のメモリのうち、{successful_updates}個を正常に更新、{failed_updates}個の更新に失敗",
            "updated_memories": updated_memories,
            "total_updated": successful_updates,
            "total_failed": failed_updates,
        }

        return json.dumps(result_data, ensure_ascii=False)
