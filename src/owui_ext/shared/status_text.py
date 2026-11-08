"""Length cap for Open WebUI status descriptions."""

import unicodedata


def truncate_status(description: str, limit: int) -> str:
    """Cut ``description`` to ``limit`` display columns (full-width characters count as 2); 0 disables."""
    if limit <= 0:
        return description
    width = 0
    for index, char in enumerate(description):
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if width > limit:
            return description[:index] + "..."
    return description
