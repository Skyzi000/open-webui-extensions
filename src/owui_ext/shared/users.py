"""User-payload normalization shared across owui_ext tool plugins."""

from typing import Any


def _normalize_user(user: Any) -> Any:
    """Convert raw __user__ dict payloads into UserModel when needed."""
    if user is None or hasattr(user, "id"):
        return user
    if isinstance(user, dict):
        try:
            from open_webui.models.users import UserModel

            return UserModel(**user)
        except Exception:
            from types import SimpleNamespace

            return SimpleNamespace(**user)
    return user
