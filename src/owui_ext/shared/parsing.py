"""Tolerant text/JSON parsing helpers shared by tool plugins."""

import json
from typing import Optional


def safe_json_loads(text: str) -> Optional[dict]:
    """Parse JSON, falling back to the largest ``{...}`` substring on failure.

    LLM responses often surround the JSON object with prose ("Here's the
    answer: {..}"). The fallback grabs the slice between the first ``{`` and
    the last ``}`` and retries; if that also fails we give up.
    """
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip()
