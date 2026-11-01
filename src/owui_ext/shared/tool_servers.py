"""Direct-tool-server payload helpers shared across owui_ext tool plugins.

Open WebUI sends ``tool_servers`` (the user's MCP / direct-tool-server
config) through ``metadata.tool_servers`` after core-level feature and
permission gates, and older paths may still expose it in the request
body. The plugins in this repo all need the same handful of operations:

- coerce raw payloads into a list of dict copies,
- trust ``metadata.tool_servers`` when core supplied it, recovering consumed
  specs from Core's resolved tools and prompts from the same request connection,
- expand ``specs`` into the per-name dict that core middleware expects,
- collect non-empty system prompts from the loaded servers.

``resolve_terminal_id_from_request_and_metadata`` still prefers the
request body over metadata for the ``terminal_id`` field that selects
the user's terminal binding, so it lives in this module too.

The implementations were byte-identical across plugins (modulo a log
prefix and ``list`` vs ``typing.List`` annotations); this module owns
the single copy.
"""

import json
import logging
from collections.abc import Mapping
from typing import Any, Optional

from fastapi import Request

_tool_servers_log = logging.getLogger("owui_ext.shared.tool_servers")


def normalize_direct_tool_servers(value: Any) -> list[dict]:
    """Normalize direct tool server payload into a list of dict copies."""
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def extract_direct_tool_server_prompts(direct_tools: Mapping[str, dict]) -> list[str]:
    """Collect unique non-empty system prompts from loaded direct tools only."""
    prompts: list[str] = []
    seen_prompts: set[str] = set()
    for tool in direct_tools.values():
        if not isinstance(tool, dict):
            continue
        server = tool.get("server")
        if not isinstance(server, dict):
            continue
        system_prompt = server.get("system_prompt")
        if isinstance(system_prompt, str):
            stripped_prompt = system_prompt.strip()
            if stripped_prompt and stripped_prompt not in seen_prompts:
                prompts.append(stripped_prompt)
                seen_prompts.add(stripped_prompt)
    return prompts


async def resolve_direct_tool_servers_from_request_and_metadata(
    *,
    request: Optional[Request],
    metadata: Optional[dict],
    debug: bool = False,
) -> list[dict]:
    """Resolve direct tool servers using core-gated metadata as source of truth."""
    metadata_has_tool_servers = isinstance(metadata, dict) and "tool_servers" in metadata
    servers: list[dict] = []
    missing_prompts: list[tuple[dict, dict]] = []

    def connection_settings(server: dict) -> dict:
        return {
            key: value for key, value in server.items()
            if key not in {"specs", "system_prompt"}
        }

    if metadata_has_tool_servers:
        servers = normalize_direct_tool_servers(metadata.get("tool_servers"))
        parent_tools = metadata.get("tools")
        parent_tools = parent_tools if isinstance(parent_tools, dict) else {}
        for server in servers:
            if "specs" in server and "system_prompt" in server:
                continue
            settings = connection_settings(server)
            approved_specs = [
                dict(tool["spec"])
                for tool in parent_tools.values()
                if isinstance(tool, dict)
                and tool.get("direct") is True
                and isinstance(tool.get("server"), dict)
                and connection_settings(tool["server"]) == settings
                and isinstance(tool.get("spec"), dict)
                and isinstance(tool["spec"].get("name"), str)
                and tool["spec"]["name"]
            ]
            # Core pops specs from metadata.tool_servers, but retains only
            # the permitted entries in metadata.tools. Never use raw specs
            # to recreate a tool that Core removed.
            if "specs" not in server and approved_specs:
                server["specs"] = approved_specs
            if approved_specs and server.get("specs") and "system_prompt" not in server:
                missing_prompts.append((server, settings))
        if not missing_prompts:
            return servers

    request_servers: list[dict] = []
    if request is not None:
        request_body = getattr(request, "body", None)
        if callable(request_body):
            try:
                raw_body = await request_body()
                if raw_body:
                    body = json.loads(raw_body)
                    if isinstance(body, dict):
                        request_servers = normalize_direct_tool_servers(
                            body.get("tool_servers")
                        )
                        if not request_servers:
                            nested_metadata = body.get("metadata")
                            if isinstance(nested_metadata, dict):
                                request_servers = normalize_direct_tool_servers(
                                    nested_metadata.get("tool_servers")
                                )
            except Exception:
                request_servers = []
    if not metadata_has_tool_servers:
        return request_servers

    for server, settings in missing_prompts:
        prompts = [
            candidate.get("system_prompt")
            for candidate in request_servers
            if connection_settings(candidate) == settings
        ]
        # Prompt-only recovery is safe once Core approved this exact
        # connection. Ambiguous duplicate connections stay without a prompt.
        if prompts and all(isinstance(prompt, str) and prompt == prompts[0] for prompt in prompts):
            server["system_prompt"] = prompts[0]
    return servers


def _direct_tool_spec(server: dict, spec: dict) -> dict:
    """Give a browser terminal's ``display_file`` Core's builtin spec, as Core 0.12.5+ does."""
    if not (server.get("is_terminal") and spec.get("name") == "display_file"):
        return spec
    try:
        from open_webui.tools.builtin import display_file
        from open_webui.utils.tools import get_builtin_tool_spec
    except ImportError:
        return spec
    return get_builtin_tool_spec(display_file)


def build_direct_tools_dict(
    *, tool_servers: list[dict], debug: bool = False
) -> dict:
    """Build direct tool entries compatible with Open WebUI middleware."""
    direct_tools: dict = {}
    for server in tool_servers:
        if not isinstance(server, dict):
            continue
        specs = server.get("specs", [])
        if not isinstance(specs, list) or not specs:
            continue
        server_payload = {k: v for k, v in server.items() if k != "specs"}
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not isinstance(name, str) or not name:
                continue
            direct_tools[name] = {
                "spec": _direct_tool_spec(server_payload, spec),
                "direct": True,
                "server": server_payload,
                "type": "direct",
            }
    if debug and tool_servers and not direct_tools:
        _tool_servers_log.info("No direct tools loaded from tool_servers")
    return direct_tools


async def resolve_terminal_id_from_request_and_metadata(
    *,
    request: Optional[Request],
    metadata: Optional[dict],
    debug: bool = False,
) -> str:
    """Resolve ``terminal_id`` preferring the request body over metadata.

    Open WebUI puts the active terminal binding in the request body
    (top-level ``terminal_id`` or nested ``metadata.terminal_id``) and
    in the inlet ``metadata`` dict the plugin is invoked with. The
    request body is the source of truth -- metadata can be stale when
    the user just switched terminals -- so the body wins when both are
    present.
    """

    def _normalize(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    metadata_terminal_id = ""
    if isinstance(metadata, dict):
        metadata_terminal_id = _normalize(metadata.get("terminal_id"))

    request_terminal_id = ""
    if request is not None:
        request_body = getattr(request, "body", None)
        if callable(request_body):
            try:
                raw_body = await request_body()
                if raw_body:
                    body = json.loads(raw_body)
                    if isinstance(body, dict):
                        request_terminal_id = _normalize(body.get("terminal_id"))
                        if not request_terminal_id:
                            nested_metadata = body.get("metadata")
                            if isinstance(nested_metadata, dict):
                                request_terminal_id = _normalize(
                                    nested_metadata.get("terminal_id")
                                )
            except Exception:
                request_terminal_id = ""

    if request_terminal_id:
        if debug and metadata_terminal_id and metadata_terminal_id != request_terminal_id:
            _tool_servers_log.warning(
                "terminal_id mismatch between request body and metadata; "
                "using request body terminal_id"
            )
        return request_terminal_id

    return metadata_terminal_id
