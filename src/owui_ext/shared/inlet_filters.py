"""Inlet-filter application helper shared by tool plugins.

Open WebUI normally runs inlet filter functions before forwarding a
request to the model. Plugins that issue their own ``generate_chat_completion``
calls miss that pipeline, so they re-run the inlet filter chain manually
when their ``APPLY_INLET_FILTERS`` Valve is on.

The helper inlines a private ``_inlet_filters_maybe_await`` instead of importing
``maybe_await`` from ``shared.async_utils`` because the inliner forbids
a shared dep module from mixing external imports (``fastapi.Request``)
with imports from other ``owui_ext.shared.*`` modules.
"""

import logging
from typing import Any

from fastapi import Request

_inlet_filters_log = logging.getLogger("owui_ext.shared.inlet_filters")


async def _inlet_filters_maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def apply_inlet_filters_if_enabled(
    apply_inlet_filters: bool,
    request: Request,
    model: dict,
    form_data: dict,
    extra_params: dict,
) -> dict:
    if not apply_inlet_filters:
        return form_data
    try:
        from open_webui.models.functions import Functions
        from open_webui.utils.filter import (
            get_sorted_filter_ids,
            process_filter_functions,
        )

        filter_ids = await _inlet_filters_maybe_await(
            get_sorted_filter_ids(
                request,
                model,
                form_data.get("metadata", {}).get("filter_ids", []),
            )
        )
        filter_functions = []
        for filter_id in filter_ids:
            function = await _inlet_filters_maybe_await(Functions.get_function_by_id(filter_id))
            if function:
                filter_functions.append(function)
        form_data, _ = await process_filter_functions(
            request=request,
            filter_functions=filter_functions,
            filter_type="inlet",
            form_data=form_data,
            extra_params=extra_params,
        )
    except Exception as exc:
        _inlet_filters_log.warning(f"Error applying inlet filters: {exc}")
    return form_data
