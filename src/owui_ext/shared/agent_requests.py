"""Request assembly for nested agent loops in Open WebUI Core's order."""

from owui_ext.shared.inlet_filters import (
    apply_inlet_filters_if_enabled,
    finalize_model_request,
    isolate_request_state,
)
from owui_ext.shared.prompt_utils import (
    _append_tool_server_prompts,
    apply_agent_tool_search,
    build_agent_tools_param,
)


def strip_tool_search_manifest(messages: list) -> list:
    """Drop Core's Tool Search manifest from system messages, for requests sent without its tools."""
    try:
        from open_webui.utils.tool_search import strip_deferred_tools_manifest
    except ImportError:
        return messages

    def without_manifest(message):
        if not isinstance(message, dict) or message.get("role") != "system":
            return message
        content = message.get("content")
        if isinstance(content, str):
            return {**message, "content": strip_deferred_tools_manifest(content)}
        if isinstance(content, list):
            return {
                **message,
                "content": [
                    {**part, "text": strip_deferred_tools_manifest(part["text"])}
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                    else part
                    for part in content
                ],
            }
        return message

    return [without_manifest(message) for message in messages]


TOOL_REQUEST_KEYS = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "parallel_tool_calls",
)


def strip_tool_request_keys(body: dict) -> dict:
    """Copy *body* without its tools, tool selection or Tool Search manifest."""
    body = {key: value for key, value in body.items() if key not in TOOL_REQUEST_KEYS}
    if isinstance(body.get("messages"), list):
        body["messages"] = strip_tool_search_manifest(body["messages"])
    return body


def own_metadata(metadata: dict) -> dict:
    """Copy inherited metadata so a nested request can change it without touching its parent."""
    import copy

    owned = {}
    for key, value in metadata.items():
        # Core keeps live objects (tool callables, MCP clients) in metadata.
        if key in ("tools", "mcp_clients"):
            owned[key] = dict(value) if isinstance(value, dict) else value
            continue
        try:
            owned[key] = copy.deepcopy(value)
        except Exception:
            owned[key] = value
    return owned


class AgentRequests:
    """One agent loop's requests, assembled the way Core assembles a chat request.

    ``prepare`` runs the inlet filters once with only the model's custom tools in the
    request, then Tool Search with the tools they left and the tool prompts. Every
    request of the loop starts from that form and runs the request filters. Body,
    ``__metadata__`` and request state share one dict, and ``chat_request`` carries
    that state through the filters and every send.
    """

    def __init__(
        self,
        tools_dict: dict,
        extra_params: dict,
        custom_tools: list = (),
        always_loaded=(),
    ):
        metadata = extra_params.get("__metadata__")
        metadata = own_metadata(metadata) if isinstance(metadata, dict) else {}
        extra_params = {**extra_params, "__metadata__": metadata}
        self.tools_dict = tools_dict
        self.extra_params = extra_params
        self.metadata = metadata
        self.custom_tools = list(custom_tools)
        self.always_loaded = tuple(always_loaded)
        self.base: dict = {}
        self.chat_request = None

    @property
    def tools_param(self) -> list | None:
        return self.base.get("tools")

    async def prepare(self, filter_pipeline, request, form_data: dict) -> list:
        """Run the loop's inlet filters once and return the messages the loop starts from."""
        import copy

        self.chat_request = isolate_request_state(request, self.metadata)
        request = self.chat_request
        filter_params = self._filter_params()
        self.metadata.update(own_metadata(form_data.get("metadata") or {}))
        form_data["metadata"] = self.metadata
        if self.custom_tools:
            form_data["tools"] = self.custom_tools
        form_data = await apply_inlet_filters_if_enabled(
            filter_pipeline, request, form_data, filter_params
        )
        form_data["metadata"] = self.metadata
        self._sync_model_id(form_data)
        filter_tools = form_data.get("tools") or []
        (
            self.tools_dict,
            self.extra_params,
            deferred,
        ) = await apply_agent_tool_search(
            self.tools_dict, self.extra_params, self.always_loaded, filter_tools
        )
        if not deferred:
            if self.tools_dict:
                self.metadata["tools"] = self.tools_dict
            else:
                self.metadata.pop("tools", None)
        tools = build_agent_tools_param(self.tools_dict, deferred, filter_tools)
        if tools:
            form_data["tools"] = tools
        else:
            form_data.pop("tools", None)
        form_data = _append_tool_server_prompts(form_data, self.extra_params)
        self.base = {
            key: value
            for key, value in form_data.items()
            if key not in ("messages", "metadata")
        }
        return copy.deepcopy(form_data["messages"])

    async def next_form(
        self,
        filter_pipeline,
        messages: list,
        tools=...,
    ) -> dict:
        """Return the loop's next request; ``tools`` replaces the loop's tools when given."""
        import copy

        form_data = {
            **copy.deepcopy(self.base),
            "messages": messages,
            "stream": False,
            "metadata": self.metadata,
        }
        if tools is not ...:
            choice = form_data.get("tool_choice")
            parallel = form_data.get("parallel_tool_calls")
            form_data = strip_tool_request_keys(form_data)
            if tools:
                form_data["tools"] = tools
                if parallel is not None:
                    form_data["parallel_tool_calls"] = parallel
                names = {(tool.get("function") or {}).get("name") for tool in tools}
                if isinstance(choice, str) or (
                    isinstance(choice, dict)
                    and ((choice.get("function") or {}).get("name") or choice.get("name"))
                    in names
                ):
                    form_data["tool_choice"] = choice
        form_data = await finalize_model_request(
            filter_pipeline,
            self.chat_request,
            form_data,
            self._filter_params(),
        )
        self._sync_model_id(form_data)
        return form_data

    def _filter_params(self) -> dict:
        return {
            **self.extra_params,
            "__metadata__": self.metadata,
        }

    def _sync_model_id(self, form_data: dict) -> None:
        if form_data.get("model"):
            self.metadata["model_id"] = form_data["model"]
