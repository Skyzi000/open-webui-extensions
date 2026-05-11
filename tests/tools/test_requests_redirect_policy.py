"""Regression tests for Open WebUI 0.9.5-style redirect handling."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUEST_METHODS = {"get", "post", "put", "request"}
TARGETS = [
    ROOT / "tools" / "universal_file_generator_pandoc.py",
]


def _uses_redirect_policy_helper(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg != "allow_redirects":
            continue
        value = keyword.value
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_http_allow_redirects"
        )
    return False


def test_requests_calls_use_core_redirect_policy() -> None:
    missing: list[str] = []

    for path in TARGETS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr in REQUEST_METHODS
                and isinstance(func.value, ast.Name)
                and func.value.id == "requests"
            ):
                continue
            if not _uses_redirect_policy_helper(node):
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert missing == []


def test_font_download_urls_do_not_require_redirects() -> None:
    redirect_prone_urls: list[str] = []

    for path in TARGETS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value.startswith("https://github.com/") and "/raw/" in node.value:
                redirect_prone_urls.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.value}")

    assert redirect_prone_urls == []
