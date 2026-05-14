"""Open WebUI module stubs scoped to tests/tools/ only.

Importing ``open_webui.utils.tools`` or ``open_webui.utils.middleware``
triggers a deep import chain (builtin -> retrieval -> torch/scipy) that
segfaults in the test environment.  This fixture installs thin stubs via
monkeypatch so they are automatically cleaned up after each test, and the
rest of the suite can still import the real modules.
"""

import importlib
import json
import sys
import types

import pytest


def _build_module(name: str, **attrs):
    """Create a minimal module object populated with known attributes."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _build_package(name: str):
    """Create a minimal package-like module for parent package imports."""
    package = types.ModuleType(name)
    package.__path__ = []
    return package


@pytest.fixture(autouse=True)
def install_open_webui_tool_stubs(monkeypatch):
    """Install child-module stubs needed by tool tests only."""
    open_webui = importlib.import_module("open_webui")
    utils_package = _build_package("open_webui.utils")
    models_package = _build_package("open_webui.models")
    monkeypatch.setitem(sys.modules, "open_webui.utils", utils_package)
    monkeypatch.setitem(sys.modules, "open_webui.models", models_package)
    monkeypatch.setattr(open_webui, "utils", utils_package, raising=False)
    monkeypatch.setattr(open_webui, "models", models_package, raising=False)

    async def _noop_get_tools(request, tool_ids, user, extra_params):
        return {}

    tools_module = _build_module(
        "open_webui.utils.tools",
        get_updated_tool_function=lambda function, extra_params: function,
        get_tools=_noop_get_tools,
        get_builtin_tools=lambda request, extra_params, **kw: {},
        get_terminal_tools=None,
    )
    monkeypatch.setitem(sys.modules, "open_webui.utils.tools", tools_module)
    monkeypatch.setattr(utils_package, "tools", tools_module, raising=False)

    async def _process_tool_result(
        request,
        tool_function_name,
        tool_result,
        tool_type,
        direct_tool=False,
        metadata=None,
        user=None,
    ):
        if isinstance(tool_result, tuple):
            tool_result = tool_result[0] if tool_result else ""
        elif direct_tool and isinstance(tool_result, list) and len(tool_result) == 2:
            tool_result = tool_result[0]

        if isinstance(tool_result, (dict, list)):
            tool_result = json.dumps(tool_result, indent=2, ensure_ascii=False)
        elif tool_result is not None and not isinstance(tool_result, str):
            tool_result = str(tool_result)

        return tool_result, [], []

    middleware_module = _build_module(
        "open_webui.utils.middleware",
        process_tool_result=_process_tool_result,
        get_citation_source_from_tool_result=lambda *a, **kw: [],
        get_file_url_from_base64=None,
    )
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setattr(utils_package, "middleware", middleware_module, raising=False)

    files_module = _build_module(
        "open_webui.utils.files",
        get_file_url_from_base64=None,
    )
    monkeypatch.setitem(sys.modules, "open_webui.utils.files", files_module)
    monkeypatch.setattr(utils_package, "files", files_module, raising=False)

    class _UserModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    users_module = _build_module(
        "open_webui.models.users",
        UserModel=_UserModel,
    )
    monkeypatch.setitem(sys.modules, "open_webui.models.users", users_module)
    monkeypatch.setattr(models_package, "users", users_module, raising=False)

    functions_module = _build_module("open_webui.models.functions")
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setattr(models_package, "functions", functions_module, raising=False)

    filter_module = _build_module("open_webui.utils.filter")
    monkeypatch.setitem(sys.modules, "open_webui.utils.filter", filter_module)
    monkeypatch.setattr(utils_package, "filter", filter_module, raising=False)
