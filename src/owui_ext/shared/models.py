"""Model listing helpers shared by ``llm_review`` and ``multi_model_council``.

Both plugins need to enumerate the models available to the current
user (filtering out filter pipelines and applying Open WebUI's own
``get_filtered_models`` access rules) and extract their IDs into a
deduplicated list. The two implementations were byte-identical except
for ``list`` vs ``typing.List`` annotations.

Lives in its own shared dep so non-using plugins (sub_agent / magi /
parallel_tools) don't include this code in their bundle; the inliner
has no tree-shaking. It follows the same shared-dep "no mixing external
+ cross-shared imports" rule that already shaped
``shared.tool_execution`` / ``shared.mcp_tools`` / ``shared.skills``:
the tiny ``maybe_await`` helper is inlined privately rather than
imported from ``shared.async_utils``.

Note: ``parse_model_ids`` is **not** here. The two plugins'
``parse_model_ids`` copies look almost identical but diverge on a
behaviorally important detail: ``llm_review`` preserves duplicates intentionally
(same model + different personas → multiple agent slots), while
``multi_model_council`` deduplicates (no point in two votes from the
same model). Each plugin keeps its own version.
"""

import logging
from typing import Any

from fastapi import Request

_models_log = logging.getLogger("owui_ext.shared.models")


async def _models_maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def get_available_models(
    request: Request,
    user: Any,
) -> list[dict]:
    """Return models visible to *user* with filter-type pipelines removed."""
    from open_webui.utils.models import get_all_models, get_filtered_models

    all_models = await get_all_models(request, refresh=False, user=user)

    filtered = []
    for model in all_models:
        if "pipeline" in model and model["pipeline"].get("type") == "filter":
            continue
        filtered.append(model)

    return await _models_maybe_await(get_filtered_models(filtered, user))


def extract_model_ids(models: list[dict]) -> list[str]:
    """Extract unique non-empty string IDs from *models* in source order."""
    ids: list[str] = []
    seen: set[str] = set()
    for model in models:
        model_id = model.get("id")
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if model_id and model_id not in seen:
            seen.add(model_id)
            ids.append(model_id)
    return ids
