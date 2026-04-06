#!/usr/bin/env python3
"""
Lint script for Tools class safety and Open WebUI schema compatibility.

Checks performed:
- Helper methods must not live inside the ``Tools`` class
- Public tool methods must not use Google-style ``Args:`` blocks
- Public tool methods must not use multiline ``:param`` descriptions
- Public tool methods must not expose raw ``dict`` / ``list[dict]`` schemas
  through their parameters or nested Pydantic models, except for explicit
  lint-side passthrough allowlist entries
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


PARAM_PATTERN = re.compile(r":param\s+(\w+)\s*:(.*)$")
SPHINX_DIRECTIVE_PATTERN = re.compile(r"^:[A-Za-z_][\w-]*\b")
DEFAULT_PATHS = (Path("tools"), Path("graphiti/tools"))
RAW_OBJECT_PASSTHROUGH_ALLOWLIST: dict[str, dict[str, set[str]]] = {
    "parallel_tools.py": {
        "ToolCallItem": {"args"},
    }
}


def _iter_python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py" and path.exists():
            files.append(path)
    return files


def _iter_public_tools_methods(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Tools":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                    methods.append(item)
    return methods


def _class_inherits_base_model(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if _annotation_name(base) == "BaseModel":
            return True
    return False


def _collect_model_classes(tree: ast.AST) -> dict[str, ast.ClassDef]:
    models: dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _class_inherits_base_model(node):
            models[node.name] = node
    return models


def _annotation_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _iter_union_members(node: ast.AST | None) -> list[ast.AST]:
    if node is None:
        return []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _iter_union_members(node.left) + _iter_union_members(node.right)
    if isinstance(node, ast.Subscript):
        container_name = _annotation_name(node.value)
        if container_name == "Optional":
            return _iter_union_members(node.slice)
        if container_name == "Union":
            if isinstance(node.slice, ast.Tuple):
                members: list[ast.AST] = []
                for element in node.slice.elts:
                    members.extend(_iter_union_members(element))
                return members
            return _iter_union_members(node.slice)
    return [node]


def _is_raw_dict_annotation(node: ast.AST | None) -> bool:
    if node is None:
        return False

    name = _annotation_name(node)
    if name in {"dict", "Dict"}:
        return True

    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value) in {"dict", "Dict"}

    return False


def _is_list_of_raw_dict_annotation(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Subscript):
        return False

    if _annotation_name(node.value) not in {"list", "List"}:
        return False

    item_node = node.slice
    if isinstance(item_node, ast.Tuple) and item_node.elts:
        item_node = item_node.elts[0]

    for member in _iter_union_members(item_node):
        if _is_raw_dict_annotation(member):
            return True

    return False


def _public_param_uses_raw_schema(arg: ast.arg) -> bool:
    if arg.arg == "self" or arg.arg.startswith("__"):
        return False

    for member in _iter_union_members(arg.annotation):
        if _is_raw_dict_annotation(member) or _is_list_of_raw_dict_annotation(member):
            return True
    return False


def _get_raw_object_passthrough_fields(filepath: Path, model_name: str) -> set[str]:
    file_allowlist = RAW_OBJECT_PASSTHROUGH_ALLOWLIST.get(filepath.name, {})
    return set(file_allowlist.get(model_name, set()))


def _iter_model_fields(model_node: ast.ClassDef) -> list[ast.AnnAssign]:
    fields: list[ast.AnnAssign] = []
    for item in model_node.body:
        if not isinstance(item, ast.AnnAssign):
            continue
        if not isinstance(item.target, ast.Name):
            continue
        field_name = item.target.id
        if field_name.startswith("__"):
            continue
        fields.append(item)
    return fields


def _iter_referenced_model_names(annotation: ast.AST | None, models: dict[str, ast.ClassDef]) -> set[str]:
    if annotation is None:
        return set()

    names: set[str] = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in models:
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in models:
            names.add(node.attr)
    return names


def _find_nested_raw_schema_issues(
    *,
    filepath: Path,
    annotation: ast.AST | None,
    models: dict[str, ast.ClassDef],
    visited: set[str] | None = None,
    path: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    if annotation is None:
        return issues

    visited = visited or set()

    for model_name in sorted(_iter_referenced_model_names(annotation, models)):
        if model_name in visited:
            continue
        model_node = models[model_name]
        passthrough_fields = _get_raw_object_passthrough_fields(filepath, model_name)
        next_visited = visited | {model_name}
        for field in _iter_model_fields(model_node):
            field_name = field.target.id
            field_path = path + (model_name, field_name)
            field_annotation = field.annotation

            raw_schema = False
            for member in _iter_union_members(field_annotation):
                if _is_raw_dict_annotation(member) or _is_list_of_raw_dict_annotation(member):
                    raw_schema = True
                    break

            if raw_schema:
                if field_name not in passthrough_fields:
                    annotation_text = (
                        ast.unparse(field_annotation) if field_annotation is not None else "unknown"
                    )
                    issues.append((".".join(field_path), annotation_text))
                continue

            issues.extend(
                _find_nested_raw_schema_issues(
                    filepath=filepath,
                    annotation=field_annotation,
                    models=models,
                    visited=next_visited,
                    path=path + (model_name, field_name),
                )
            )

    return issues


def _iter_public_params(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.arg]:
    params: list[ast.arg] = []
    params.extend(method.args.posonlyargs)
    params.extend(method.args.args)
    params.extend(method.args.kwonlyargs)
    if method.args.vararg is not None:
        params.append(method.args.vararg)
    if method.args.kwarg is not None:
        params.append(method.args.kwarg)
    return [param for param in params if param.arg != "self" and not param.arg.startswith("__")]


def _starts_new_docstring_block(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if SPHINX_DIRECTIVE_PATTERN.match(stripped):
        return True
    if stripped.startswith(("Args:", "Returns", "Raises", "Example", "Examples")):
        return True
    return False


def _find_multiline_param_names(docstring: str) -> set[str]:
    multiline_names: set[str] = set()
    lines = docstring.splitlines()
    for index, line in enumerate(lines):
        match = PARAM_PATTERN.match(line.strip())
        if not match:
            continue

        param_name = match.group(1)
        if match.group(2).strip():
            inline_description = True
        else:
            inline_description = False
        cursor = index + 1
        while cursor < len(lines):
            next_line = lines[cursor]
            stripped = next_line.strip()
            if not stripped:
                break
            if _starts_new_docstring_block(next_line):
                break
            if inline_description or next_line[:1].isspace():
                multiline_names.add(param_name)
            break
    return multiline_names


def check_tools_class(filepath: Path) -> list[str]:
    """Detect Tools class API and schema compatibility issues."""
    errors: list[str] = []
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{filepath}: Parse error - {exc}"]

    models = _collect_model_classes(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Tools":
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name.startswith("_") and not item.name.startswith("__"):
                    errors.append(
                        f"{filepath}:{item.lineno}: "
                        f"Helper method '{item.name}' found in Tools class. "
                        f"Move it outside the class."
                    )

    for method in _iter_public_tools_methods(tree):
        docstring = ast.get_docstring(method) or ""
        multiline_params = _find_multiline_param_names(docstring)

        if any(line.strip() == "Args:" for line in docstring.splitlines()):
            errors.append(
                f"{filepath}:{method.lineno}: {method.name} uses Google-style Args block; "
                "use single-line :param entries for Open WebUI compatibility."
            )

        for arg in _iter_public_params(method):
            if arg.arg in multiline_params:
                errors.append(
                    f"{filepath}:{method.lineno}: {method.name} has multiline :param for "
                    f"'{arg.arg}'; only the first line is parsed by Open WebUI."
                )
            if _public_param_uses_raw_schema(arg):
                annotation = ast.unparse(arg.annotation) if arg.annotation is not None else "unknown"
                errors.append(
                    f"{filepath}:{method.lineno}: {method.name} uses raw dict/list[dict] "
                    f"annotation for public parameter '{arg.arg}' ({annotation}); use a named Pydantic model instead."
                )
            for nested_path, annotation in _find_nested_raw_schema_issues(
                filepath=filepath,
                annotation=arg.annotation,
                models=models,
            ):
                errors.append(
                    f"{filepath}:{method.lineno}: {method.name} uses raw dict/list[dict] "
                    f"annotation inside public model field '{nested_path}' ({annotation}); "
                    "use a named Pydantic model instead or add a lint-side allowlist "
                    "entry for intentional passthrough."
                )

    return errors


def check_paths(paths: list[Path]) -> list[str]:
    """Run the lint over files and directories."""
    errors: list[str] = []
    for filepath in _iter_python_files(paths):
        errors.extend(check_tools_class(filepath))
    return errors


def main() -> int:
    """Main entry point."""
    raw_paths = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else list(DEFAULT_PATHS)
    errors = check_paths(raw_paths)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
