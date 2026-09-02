"""Model-request filter helpers shared by tool plugins.

Open WebUI normally runs inlet filter functions before forwarding a
request to the model. Since v0.11.2 it also runs request filters immediately
before every model call. Plugins that issue their own
``generate_chat_completion`` calls miss that pipeline, so they mirror both
stages when their ``APPLY_INLET_FILTERS`` Valve is on.

``process_filter_functions`` re-injects the matched filter's
``UserValves`` into the ``__user__`` dict it receives, which means a
shared ``__user__`` object would leak filter-specific valves into the
caller's subsequent tool calls under a different tool id. The helper
guards against that by shallow-copying ``extra_params`` and the inner
``__user__`` dict before handing them to core.

Each nested model pipeline resolves its effective route and filter functions
once and owns a fresh ``FilterContext``. Core's request-scoped context belongs
to the outer model's fixed filter set and cannot safely serve different nested
models. Arena routes stay pinned to one child for the lifetime of the nested
loop.

The helper inlines a private ``_inlet_filters_maybe_await`` instead of importing
``maybe_await`` from ``shared.async_utils`` because the inliner forbids
a shared dep module from mixing external imports (``fastapi.Request``)
with imports from other ``owui_ext.shared.*`` modules.
"""

import inspect
import logging
import random
from typing import Any

from fastapi import Request

_inlet_filters_log = logging.getLogger("owui_ext.shared.inlet_filters")


async def _inlet_filters_maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def resolve_model_filter_pipeline(
    apply_filters: bool,
    request: Request,
    model_id: str,
    enabled_filter_ids: list[str] | None,
) -> dict[str, Any] | None:
    """Resolve one nested model route and isolate its cached filter Valves."""
    request_state = getattr(request, "state", None)
    direct_model = getattr(request_state, "model", None)
    server_models = request.app.state.MODELS
    is_direct_target = bool(
        getattr(request_state, "direct", False)
        and isinstance(direct_model, dict)
        and direct_model.get("id") == model_id
    )

    # Core gives the request-scoped Direct model precedence over a server
    # model with the same ID. Otherwise resolve Arena before filters so one
    # logical agent loop keeps the same child for every provider call.
    model = direct_model if is_direct_target else server_models.get(model_id, {})
    if not is_direct_target and model.get("owned_by") == "arena":
        candidate_ids = model.get("info", {}).get("meta", {}).get("model_ids")
        filter_mode = model.get("info", {}).get("meta", {}).get("filter_mode")
        if candidate_ids and filter_mode == "exclude":
            candidate_ids = [
                candidate["id"]
                for candidate in server_models.values()
                if candidate.get("owned_by") != "arena"
                and candidate["id"] not in candidate_ids
            ]

        if not isinstance(candidate_ids, list) or not candidate_ids:
            candidate_ids = [
                candidate["id"]
                for candidate in server_models.values()
                if candidate.get("owned_by") != "arena"
            ]

        selected_model_id = random.choice(candidate_ids)
        selected_model = (
            direct_model
            if getattr(request_state, "direct", False)
            and isinstance(direct_model, dict)
            and direct_model.get("id") == selected_model_id
            else server_models.get(selected_model_id)
        )
        if selected_model:
            model_id = selected_model_id
            model = selected_model

    route = {
        "model": model,
        "model_id": model.get("id", model_id),
        "process": None,
        "functions": [],
        "context": None,
        "supports_context": False,
        "supports_request": False,
    }
    if not apply_filters:
        return route

    request_filters_supported = False
    try:
        from open_webui.utils import filter as filter_utils

        process_filter_functions = filter_utils.process_filter_functions
        supports_context = (
            "filter_context"
            in inspect.signature(process_filter_functions).parameters
        )
        context_factory = getattr(filter_utils, "FilterContext", None)
        # FilterContext exists in v0.11.0, but request filters and
        # get_filter_context were introduced together in v0.11.2.
        request_filters_supported = bool(
            supports_context
            and callable(context_factory)
            and callable(getattr(filter_utils, "get_filter_context", None))
        )

        enabled_filter_ids = list(enabled_filter_ids or [])
        get_filter_functions = getattr(filter_utils, "get_filter_functions", None)
        if callable(get_filter_functions):
            filter_functions = await _inlet_filters_maybe_await(
                get_filter_functions(request, model, enabled_filter_ids)
            )
        else:
            from open_webui.models.functions import Functions

            filter_ids = await _inlet_filters_maybe_await(
                filter_utils.get_sorted_filter_ids(
                    request,
                    model,
                    enabled_filter_ids,
                )
            )
            filter_functions = []
            for filter_id in filter_ids:
                function = await _inlet_filters_maybe_await(
                    Functions.get_function_by_id(filter_id)
                )
                if function:
                    filter_functions.append(function)

        return route | {
            "process": process_filter_functions,
            "functions": filter_functions,
            "context": context_factory() if request_filters_supported else None,
            "supports_context": supports_context,
            "supports_request": request_filters_supported,
        }
    except Exception as exc:
        _inlet_filters_log.warning(f"Error resolving model filters: {exc}")
        if request_filters_supported:
            raise
        return route


async def _apply_filter_pipeline(
    filter_pipeline: dict[str, Any] | None,
    request: Request,
    form_data: dict,
    extra_params: dict,
    filter_type: str,
) -> dict:
    if filter_pipeline is None:
        return form_data

    if filter_type == "inlet":
        form_data["model"] = filter_pipeline["model_id"]

    if filter_pipeline["process"] is None or (
        filter_type == "request" and not filter_pipeline["supports_request"]
    ):
        return form_data

    try:
        process_filter_functions = filter_pipeline["process"]

        # Isolate __user__ so filter UserValves injection doesn't leak out
        # and pollute subsequent tool calls under a different tool id.
        local_extra_params = dict(extra_params or {})
        local_extra_params["__model__"] = filter_pipeline["model"]
        if isinstance(local_extra_params.get("__user__"), dict):
            local_extra_params["__user__"] = dict(local_extra_params["__user__"])

        process_kwargs: dict[str, Any] = {
            "request": request,
            "filter_functions": filter_pipeline["functions"],
            "filter_type": filter_type,
            "form_data": form_data,
            "extra_params": local_extra_params,
        }
        if filter_pipeline["supports_context"]:
            process_kwargs["filter_context"] = filter_pipeline["context"]
        form_data, _ = await process_filter_functions(
            **process_kwargs,
        )
    except Exception as exc:
        _inlet_filters_log.warning(f"Error applying {filter_type} filters: {exc}")
        if filter_type == "request":
            raise
    return form_data


async def apply_inlet_filters_if_enabled(
    filter_pipeline: dict[str, Any] | None,
    request: Request,
    form_data: dict,
    extra_params: dict,
) -> dict:
    return await _apply_filter_pipeline(
        filter_pipeline, request, form_data, extra_params, "inlet"
    )


async def finalize_model_request(
    filter_pipeline: dict[str, Any] | None,
    request: Request,
    form_data: dict,
    extra_params: dict,
) -> dict:
    """Apply request filters immediately before model dispatch."""
    return await _apply_filter_pipeline(
        filter_pipeline, request, form_data, extra_params, "request"
    )
