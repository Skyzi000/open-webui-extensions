"""Inlet-filter application helper shared by tool plugins.

Open WebUI normally runs inlet filter functions before forwarding a
request to the model. Plugins that issue their own ``generate_chat_completion``
calls miss that pipeline, so they re-run the inlet filter chain manually
when their ``APPLY_INLET_FILTERS`` Valve is on.

``process_filter_functions`` re-injects the matched filter's
``UserValves`` into the ``__user__`` dict it receives, which means a
shared ``__user__`` object would leak filter-specific valves into the
caller's subsequent tool calls under a different tool id. The helper
guards against that by shallow-copying ``extra_params`` and the inner
``__user__`` dict before handing them to core.

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

        # Isolate __user__ so filter UserValves injection doesn't leak out
        # and pollute subsequent tool calls under a different tool id.
        local_extra_params = dict(extra_params or {})
        if isinstance(local_extra_params.get("__user__"), dict):
            local_extra_params["__user__"] = dict(local_extra_params["__user__"])

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
            extra_params=local_extra_params,
        )
    except Exception as exc:
        _inlet_filters_log.warning(f"Error applying inlet filters: {exc}")
    return form_data
