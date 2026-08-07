"""Model-metadata feature helpers shared across owui_ext tool plugins."""

from typing import Optional


def _attached_knowledge_types(
    model: Optional[dict],
    metadata: Optional[dict] = None,
) -> set[str]:
    if not isinstance(model, dict):
        model = {}
    if not isinstance(metadata, dict):
        metadata = {}

    model_meta = model.get("info", {}).get("meta", {})
    knowledge_items = list(model_meta.get("knowledge") or [])
    knowledge_items.extend(metadata.get("folder_knowledge") or [])
    if not (model_meta.get("capabilities") or {}).get("file_context", True):
        knowledge_items.extend(
            item
            for item in metadata.get("files") or []
            if isinstance(item, dict)
            and item.get("type") in ("collection", "note")
        )

    return {
        item["type"]
        for item in knowledge_items
        if isinstance(item, dict)
        and isinstance(item.get("type"), str)
        and item.get("id")
    }


def model_has_note_knowledge(
    model: Optional[dict],
    metadata: Optional[dict] = None,
) -> bool:
    """Return True if Core can expose view_note for attached knowledge."""
    return "note" in _attached_knowledge_types(model, metadata)


def model_has_file_knowledge(
    model: Optional[dict],
    metadata: Optional[dict] = None,
) -> bool:
    """Return True if Core can expose view_file for attached knowledge."""
    return not _attached_knowledge_types(model, metadata).isdisjoint(
        {"file", "collection"}
    )


def model_knowledge_tools_enabled(model: Optional[dict]) -> bool:
    """Return True if model-level builtin knowledge tools are enabled."""
    if not isinstance(model, dict):
        return True
    builtin_tools = model.get("info", {}).get("meta", {}).get("builtinTools", {})
    if not isinstance(builtin_tools, dict):
        return True
    return bool(builtin_tools.get("knowledge", True))
