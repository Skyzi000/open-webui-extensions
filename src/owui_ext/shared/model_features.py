"""Model-metadata feature helpers shared across owui_ext tool plugins."""

from typing import Optional


def model_has_note_knowledge(model: Optional[dict]) -> bool:
    """Return True if the current model has note-type attached knowledge."""
    if not isinstance(model, dict):
        return False
    knowledge_items = model.get("info", {}).get("meta", {}).get("knowledge") or []
    if not isinstance(knowledge_items, list):
        return False
    return any(
        item.get("type") == "note"
        for item in knowledge_items
        if isinstance(item, dict)
    )


def model_knowledge_tools_enabled(model: Optional[dict]) -> bool:
    """Return True if model-level builtin knowledge tools are enabled."""
    if not isinstance(model, dict):
        return True
    builtin_tools = model.get("info", {}).get("meta", {}).get("builtinTools", {})
    if not isinstance(builtin_tools, dict):
        return True
    return bool(builtin_tools.get("knowledge", True))
