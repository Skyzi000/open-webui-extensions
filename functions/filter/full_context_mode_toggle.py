"""
title: Full Context Mode
author: Skyzi000
author_url: https://github.com/Skyzi000/open-webui-extensions
description: Toggle full context mode for user-attached files at once. Sets the "context": "full" flag to leverage Open WebUI's native full context processing (RAG template with citations, etc.). Auto-injected model/folder knowledge entries are left untouched so large knowledge bases keep their normal chunk-based retrieval. Note: if the global RAG_FULL_CONTEXT setting is already enabled, this filter has no additional effect.
version: 1.1.0
license: MIT
"""

from pydantic import BaseModel, Field


def file_id(file: dict) -> str | None:
    if not isinstance(file, dict):
        return None
    value = file.get("id")
    return str(value) if value else None


def file_ids(files: list | None) -> set[str]:
    if not isinstance(files, list):
        return set()
    return {id_ for file in files if (id_ := file_id(file))}


def model_knowledge_file_ids(metadata: dict) -> set[str]:
    model = metadata.get("model") or {}
    knowledge = model.get("info", {}).get("meta", {}).get("knowledge") or []

    ids = set()
    for item in knowledge:
        if not isinstance(item, dict):
            continue
        if item.get("collection_name") or item.get("collection_names"):
            continue
        if item.get("type", "file") == "file" and (id_ := file_id(item)):
            ids.add(id_)

    return ids


def folder_knowledge_file_ids(metadata: dict, __user__: dict | None) -> set[str]:
    user_id = (__user__ or {}).get("id")
    if not user_id:
        return set()

    folder_id = metadata.get("folder_id")
    chat_id = metadata.get("chat_id")

    try:
        if not folder_id and chat_id and not chat_id.startswith("local:"):
            from open_webui.models.chats import Chats

            folder_id = Chats.get_chat_folder_id(chat_id, user_id)

        if folder_id:
            from open_webui.models.folders import Folders

            folder = Folders.get_folder_by_id_and_user_id(folder_id, user_id)
            return file_ids((folder.data or {}).get("files", []) if folder else [])
    except Exception:
        return set()

    return set()


def auto_injected_file_ids(body: dict, __user__: dict | None) -> set[str]:
    metadata = body.get("metadata", {})
    if not isinstance(metadata, dict):
        return set()

    return model_knowledge_file_ids(metadata) | folder_knowledge_file_ids(
        metadata, __user__
    )


def current_message_file_ids(body: dict) -> set[str] | None:
    metadata = body.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    parent_message = metadata.get("parent_message") or {}
    if not isinstance(parent_message, dict):
        return None

    files = parent_message.get("files")
    if not isinstance(files, list):
        return None

    return file_ids(files)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        """Set the "context": "full" flag for user-attached files in both body["files"] and body["metadata"]["files"] payload shapes. Auto-injected model/folder knowledge entries are skipped to avoid forcing full-context retrieval on large knowledge bases."""
        if not isinstance(body, dict):
            return body

        top_level_files = body.get("files")
        metadata_files = body.get("metadata", {}).get("files")
        auto_file_ids = auto_injected_file_ids(body, __user__)
        explicit_file_ids = current_message_file_ids(body)

        for files in (top_level_files, metadata_files):
            if not isinstance(files, list):
                continue
            for file in files:
                if not isinstance(file, dict):
                    continue
                id_ = file_id(file)
                if explicit_file_ids is not None and id_ not in explicit_file_ids:
                    # Skip only entries known to be auto-injected. Explicitly
                    # attached knowledge items may also be legacy/collection.
                    if (
                        file.get("legacy") is True
                        or file.get("type") == "collection"
                        or id_ in auto_file_ids
                    ):
                        continue
                file["context"] = "full"

        return body
