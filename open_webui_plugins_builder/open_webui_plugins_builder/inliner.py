"""Module-based AST inliner.

Given a single target source file plus a set of declared local-import roots
(e.g., ``owui_ext.shared``), this module:

1. Finds top-level ``from <root>.<sub> import <names>`` statements in the
   target.
2. Loads each referenced shared module and recursively follows its own local
   shared imports.
3. Verifies that every imported name exists as a top-level binding in the
   referenced module.
4. Topologically sorts the dependency set, dedupes by module path, and inlines
   each dependency's body (with its own top-level imports stripped).
5. Emits the target's top-level external imports in a shared section
   above all dep blocks; each dep's external imports stay attached to
   its own block (next to the dep body) so the per-dep
   ``externals -> body`` order is preserved across the dependency chain.
   ``from __future__`` imports are unioned across target and deps into a
   single line at the top.
6. Emits a generated file with a do-not-edit marker.

Anything that breaks these invariants -- aliases on local shared imports,
``import owui_ext.shared.foo`` style, star imports, function-level local
imports, package-relative imports, top-level imports interleaved with
non-import code, an external import that follows a local-shared import,
a shared dep module that mixes external and local-shared imports,
circular imports, or imported names that don't exist -- raises
``BuildError``.

The original source-text of non-import statements is preserved by slicing the
file by AST line numbers rather than re-emitting via ``ast.unparse``. This
keeps comments and formatting intact.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import BuildError


GENERATED_MARKER = "# === GENERATED FILE - DO NOT EDIT ==="


@dataclass(frozen=True)
class _ParsedModule:
    path: Path
    source: str
    tree: ast.Module


@dataclass(frozen=True)
class _ImportRef:
    """A resolved local-shared import edge."""

    importer_path: Path  # file that contained the import
    target_module: str  # dotted module name, e.g. "owui_ext.shared.foo"
    target_path: Path  # resolved path to that module's .py file
    names: tuple[str, ...]  # top-level names being imported
    lineno: int
    end_lineno: int


@dataclass
class _DepModule:
    dotted_name: str
    path: Path
    source: str
    tree: ast.Module
    content_hash: str
    body_text: str
    external_imports: list[ast.stmt]
    future_imports: list[ast.ImportFrom]
    exported_names: set[str]
    local_imports: list[_ImportRef]


def build_target(
    *,
    source_text: str,
    source_path: Path,
    repo_root: Path,
    local_import_roots: tuple[str, ...],
    source_root: str = "src",
    target_name: str | None = None,
    read_source: "ReadSource | None" = None,
) -> str:
    """Build the inlined output for a single target.

    ``read_source`` allows callers to substitute a custom source provider
    (e.g., reading from a Git index for ``--staged`` mode). When ``None`` it
    defaults to filesystem reads.

    ``target_name`` is recorded in the regenerate-command line of the marker
    block so the comment is directly copy-pasteable. When omitted we fall back
    to ``--all``.
    """

    reader = read_source or _filesystem_reader

    target_module = _parse_module(source_path, source_text)

    target_doc_range = _module_docstring_range(target_module.tree)
    target_future = _collect_future_imports(target_module.tree)
    _verify_no_relative_imports(target_module)
    target_external, target_local = _classify_top_level_imports(
        target_module, local_import_roots
    )
    _verify_no_nested_local_imports(target_module, local_import_roots)
    _verify_externals_precede_locals(target_module, local_import_roots)

    target_local_refs = [
        _resolve_local_import(
            target_module.path,
            node,
            repo_root,
            local_import_roots,
            source_root,
            reader,
        )
        for node in target_local
    ]

    deps_ordered = _resolve_dependency_graph(
        seeds=target_local_refs,
        repo_root=repo_root,
        local_import_roots=local_import_roots,
        source_root=source_root,
        reader=reader,
    )

    _verify_dep_exports(deps_ordered, target_local_refs)
    _verify_no_name_collisions(
        target_module=target_module,
        target_local_imports=target_local,
        target_external_imports=target_external,
        deps_ordered=deps_ordered,
    )

    target_drop_ranges: list[tuple[int, int]] = []
    if target_doc_range is not None:
        target_drop_ranges.append(target_doc_range)
    for node in target_future + target_external + target_local:
        target_drop_ranges.append((node.lineno, node.end_lineno))
    target_body_text = _slice_excluding(target_module.source, target_drop_ranges)

    # Dep externals stay attached to each dep block (see _assemble_output).
    # Hoisting them into the shared merged section reorders side effects
    # across the dep chain: dep B's `import madeup_plugin` would run before
    # dep A's body, even though the source loaded A first. By keeping each
    # dep's externals next to its body, we preserve the per-dep
    # ``externals -> body`` order exactly as the source executed it.
    merged_external_text = _format_merged_imports(target_external, [])

    merged_future_text = _format_future_imports(
        target_future + [imp for dep in deps_ordered for imp in dep.future_imports]
    )

    merged_future_names = _format_future_names(
        target_future + [imp for dep in deps_ordered for imp in dep.future_imports]
    )

    return _assemble_output(
        target_module=target_module,
        target_doc_range=target_doc_range,
        merged_future_text=merged_future_text,
        merged_future_names=merged_future_names,
        merged_external_text=merged_external_text,
        deps_ordered=deps_ordered,
        target_body_text=target_body_text,
        repo_root=repo_root,
        target_name=target_name,
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


ReadSource = "callable[[Path], str]"


def _filesystem_reader(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_module(path: Path, source: str) -> _ParsedModule:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise BuildError(f"Syntax error parsing {path}: {exc}") from exc
    return _ParsedModule(path=path, source=source, tree=tree)


def _module_docstring_range(tree: ast.Module) -> tuple[int, int] | None:
    if not tree.body:
        return None
    first = tree.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return (first.lineno, first.end_lineno or first.lineno)
    return None


def _collect_future_imports(tree: ast.Module) -> list[ast.ImportFrom]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]


def _is_local_module(module: str | None, local_roots: Iterable[str]) -> bool:
    if module is None:
        return False
    for root in local_roots:
        if module == root or module.startswith(root + "."):
            return True
    return False


def _classify_top_level_imports(
    parsed: _ParsedModule,
    local_roots: tuple[str, ...],
) -> tuple[list[ast.stmt], list[ast.ImportFrom]]:
    """Split top-level imports into (external, local-shared).

    Imports detected here are eligible for hoisting into the merged-import
    section, which means they must:

    1. Be absolute (``level == 0``). Package-relative imports like
       ``from ..util import helper`` would emit ``from ..util import helper``
       into the single-file output, which fails at runtime because Open WebUI
       loads the file with no enclosing package.
    2. Come before any non-import top-level statement. Hoisting an import
       that originally executed *after* setup code (e.g., environment
       variables or monkey-patches) would silently change runtime ordering.

    Both invariants are enforced here so the user gets a build error rather
    than a subtly broken release artifact.
    """

    _verify_imports_are_leading(parsed)

    external: list[ast.stmt] = []
    local: list[ast.ImportFrom] = []

    for node in parsed.tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            if node.level != 0:
                raise BuildError(
                    f"{parsed.path}:{node.lineno}: package-relative imports "
                    f"({'.' * node.level}{node.module or ''}) are not allowed "
                    f"at module top level. The single-file output has no "
                    f"enclosing package; rewrite as an absolute import."
                )
            if _is_local_module(node.module, local_roots):
                _validate_local_import(parsed.path, node, local_roots)
                local.append(node)
            else:
                external.append(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == r or alias.name.startswith(r + ".")
                    for r in local_roots
                ):
                    raise BuildError(
                        f"{parsed.path}:{node.lineno}: "
                        f"`import {alias.name}` is not allowed; use "
                        f"`from {alias.name} import <names>` instead."
                    )
            external.append(node)
    return external, local


def _verify_imports_are_leading(parsed: _ParsedModule) -> None:
    """Refuse modules whose top-level imports are interleaved with code.

    Python's normal style places imports at the top, but the language permits
    interleaving. Hoisting late imports into the merged section would silently
    change execution order -- e.g., a top-level ``os.environ['X'] = '1'``
    followed by ``from somelib import Y`` (where ``somelib`` reads ``X`` at
    import time) would no longer see the env var. We refuse such modules
    rather than emit a release that diverges from the source.

    A leading module docstring and ``from __future__ import`` lines are
    allowed before regular imports. Top-level conditional blocks (``try``,
    ``if``, ``with``, ``for``, ``while``) count as code: imports nested in
    those blocks stay in place, but no further top-level imports may appear
    after them.
    """

    body = parsed.tree.body
    saw_code = False
    for index, node in enumerate(body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            # Module docstring -- always allowed first, doesn't count as code.
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if saw_code:
                raise BuildError(
                    f"{parsed.path}:{node.lineno}: top-level imports must "
                    f"appear before any other top-level statements. Move "
                    f"this import to the top of the file."
                )
            continue
        saw_code = True


def _validate_local_import(
    path: Path, node: ast.ImportFrom, local_roots: tuple[str, ...]
) -> None:
    if node.level != 0:
        raise BuildError(
            f"{path}:{node.lineno}: relative imports of local shared modules are not supported."
        )
    module = node.module or ""
    if module in local_roots:
        raise BuildError(
            f"{path}:{node.lineno}: local shared imports must reference a leaf "
            f"module (e.g., `from {module}.<submodule> import ...`)."
        )
    for alias in node.names:
        if alias.name == "*":
            raise BuildError(
                f"{path}:{node.lineno}: `from {module} import *` is not allowed for local shared modules."
            )
        if alias.asname is not None:
            raise BuildError(
                f"{path}:{node.lineno}: aliases on local shared imports are not allowed "
                f"(`from {module} import {alias.name} as {alias.asname}`)."
            )


def _verify_no_nested_local_imports(
    parsed: _ParsedModule, local_roots: tuple[str, ...]
) -> None:
    top_level_ids = {
        id(node)
        for node in parsed.tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.ImportFrom):
            if (
                _is_local_module(node.module, local_roots)
                and id(node) not in top_level_ids
            ):
                raise BuildError(
                    f"{parsed.path}:{node.lineno}: local shared imports must be at "
                    f"module top level (found inside a function or conditional block)."
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == r or alias.name.startswith(r + ".")
                    for r in local_roots
                ):
                    if id(node) not in top_level_ids:
                        raise BuildError(
                            f"{parsed.path}:{node.lineno}: local shared imports must be at "
                            f"module top level (found `import {alias.name}` inside a "
                            f"function or conditional block)."
                        )
                    break


def _verify_no_relative_imports(parsed: _ParsedModule) -> None:
    """Refuse package-relative imports anywhere in the module.

    Relative imports (``from . import X``, ``from ..util import Y``) survive
    in the inlined output verbatim -- whether they live at the top level,
    inside a function, or inside a ``try``/``if`` block. Open WebUI loads
    the single-file artifact with no enclosing package, so any such import
    raises ``ImportError`` at runtime. Rather than silently emitting a
    broken release, refuse the source.
    """

    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.ImportFrom) and node.level != 0:
            raise BuildError(
                f"{parsed.path}:{node.lineno}: package-relative imports "
                f"({'.' * node.level}{node.module or ''}) are not allowed. "
                f"The single-file output has no enclosing package; rewrite "
                f"as an absolute import."
            )


def _verify_externals_precede_locals(
    parsed: _ParsedModule, local_roots: tuple[str, ...]
) -> None:
    """Refuse modules whose external imports follow a local-shared import.

    The assembler hoists every external import above every dependency body.
    If the source had ``from owui_ext.shared.patch import helper`` first and
    ``import madeup_plugin`` second, the source ran ``patch.py`` top-level
    code (which might register ``madeup_plugin``) before importing
    ``madeup_plugin`` -- but the generated file imports ``madeup_plugin``
    first, before any dep body runs, breaking modules that rely on this
    side-effect order. Refuse the source rather than emit a release whose
    runtime behavior diverges from the original.
    """

    saw_local = False
    first_local: ast.ImportFrom | None = None
    for node in parsed.tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            if node.level == 0 and _is_local_module(node.module, local_roots):
                if not saw_local:
                    first_local = node
                saw_local = True
                continue
            if saw_local:
                module = node.module or ""
                raise BuildError(
                    f"{parsed.path}:{node.lineno}: external import "
                    f"`from {module} import ...` follows a local-shared "
                    f"import on line {first_local.lineno if first_local else '?'} "
                    f"(`from {first_local.module if first_local else ''} import ...`). "
                    f"Move all external imports above any "
                    f"`from {local_roots[0]}...` lines so the inliner can "
                    f"hoist them without changing execution order."
                )
        elif isinstance(node, ast.Import):
            if saw_local:
                names = ", ".join(
                    f"{a.name} as {a.asname}" if a.asname else a.name
                    for a in node.names
                )
                raise BuildError(
                    f"{parsed.path}:{node.lineno}: external import "
                    f"`import {names}` follows a local-shared import on line "
                    f"{first_local.lineno if first_local else '?'} "
                    f"(`from {first_local.module if first_local else ''} import ...`). "
                    f"Move all external imports above any "
                    f"`from {local_roots[0]}...` lines so the inliner can "
                    f"hoist them without changing execution order."
                )


def _resolve_local_import(
    importer_path: Path,
    node: ast.ImportFrom,
    repo_root: Path,
    local_roots: tuple[str, ...],
    source_root: str,
    reader,
) -> _ImportRef:
    module = node.module or ""
    target_path = _resolve_module_path(
        module, repo_root, source_root, importer_path, node.lineno, reader
    )
    return _ImportRef(
        importer_path=importer_path,
        target_module=module,
        target_path=target_path,
        names=tuple(alias.name for alias in node.names),
        lineno=node.lineno,
        end_lineno=node.end_lineno or node.lineno,
    )


def _resolve_module_path(
    module: str,
    repo_root: Path,
    source_root: str,
    importer_path: Path,
    lineno: int,
    reader,
) -> Path:
    """Locate the shared-module file via the active reader.

    The reader is the single source of truth for whether a path "exists" in
    this build. With the default filesystem reader that is the worktree; with
    the index reader (``--staged``) it is the staged blob set, which can
    diverge from the worktree. Using ``Path.is_file()`` here would always
    consult the worktree and silently disagree with the reader in staged
    mode, so we delegate to the reader.
    """

    parts = module.split(".")
    candidate = repo_root / source_root / Path(*parts).with_suffix(".py")
    try:
        reader(candidate)
    except (FileNotFoundError, BuildError):
        raise BuildError(
            f"{importer_path}:{lineno}: cannot locate local shared module "
            f"{module!r} (expected at {candidate})."
        ) from None
    return candidate.resolve()


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------


def _resolve_dependency_graph(
    *,
    seeds: list[_ImportRef],
    repo_root: Path,
    local_import_roots: tuple[str, ...],
    source_root: str,
    reader,
) -> list[_DepModule]:
    """Resolve all transitive deps from ``seeds`` and return them in topo order.

    Topological order: a module appears AFTER all of its own dependencies, so
    that downstream code can reference upstream symbols. Equivalently, deps
    are emitted deepest-first.
    """

    cache: dict[Path, _DepModule] = {}
    visiting: set[Path] = set()
    order: list[_DepModule] = []

    def visit(ref: _ImportRef) -> _DepModule:
        path = ref.target_path
        if path in cache:
            return cache[path]
        if path in visiting:
            raise BuildError(
                f"Circular local-shared import detected involving {path}."
            )
        visiting.add(path)
        try:
            source = reader(path)
        except FileNotFoundError as exc:
            raise BuildError(
                f"Cannot read local shared module {path}: {exc}"
            ) from exc
        parsed = _parse_module(path, source)
        _verify_no_relative_imports(parsed)
        _verify_no_nested_local_imports(parsed, local_import_roots)
        _verify_externals_precede_locals(parsed, local_import_roots)
        externals, locals_ = _classify_top_level_imports(parsed, local_import_roots)
        future_imports = _collect_future_imports(parsed.tree)

        # Shared dep modules may not mix external and local-shared imports.
        # Topological sort emits a dep AFTER its own dependencies, which means
        # a dep's externals end up running after its sub-dep's body in the
        # generated file -- but in source they ran BEFORE the sub-dep loaded.
        # The target module is exempt because target externals lead the merged
        # section and run before any dep body, matching source semantics.
        if externals and locals_:
            raise BuildError(
                f"{path}: shared dep modules cannot mix external imports with "
                f"`from {local_import_roots[0]}...` imports. The single-file "
                f"layout would run the external imports after this module's "
                f"sub-dependency bodies, reordering side effects relative to "
                f"the source. Move the external imports to the target, or "
                f"split this module so externals and shared imports live in "
                f"different files."
            )

        local_refs = [
            _resolve_local_import(
                path, node, repo_root, local_import_roots, source_root, reader
            )
            for node in locals_
        ]

        for sub in local_refs:
            visit(sub)

        doc_range = _module_docstring_range(parsed.tree)
        drop_ranges: list[tuple[int, int]] = []
        if doc_range is not None:
            drop_ranges.append(doc_range)
        for node in externals + future_imports + locals_:
            drop_ranges.append((node.lineno, node.end_lineno or node.lineno))
        body_text = _slice_excluding(parsed.source, drop_ranges)

        exported = _collect_exported_names(parsed.tree)

        dep = _DepModule(
            dotted_name=ref.target_module,
            path=path,
            source=source,
            tree=parsed.tree,
            content_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            body_text=body_text,
            external_imports=externals,
            future_imports=future_imports,
            exported_names=exported,
            local_imports=local_refs,
        )
        cache[path] = dep
        order.append(dep)
        visiting.discard(path)
        return dep

    for seed in seeds:
        visit(seed)

    return order


def _collect_exported_names(tree: ast.Module) -> set[str]:
    """Names a shared module *exports* (i.e., that ``from M import X`` can target).

    A shared module's exports are every name bound at module scope, regardless
    of whether the binding sits in the module body directly or inside a
    top-level conditional block (``try/except``, ``if/else``, ``with``,
    ``for``, ``while``). Re-exports via direct top-level ``Import`` /
    ``ImportFrom`` count too -- a target's ``from owui_ext.shared.util import
    VALUE`` should resolve as long as ``util.py`` makes ``VALUE`` reachable at
    module scope.
    """

    return _collect_inlined_module_bindings(tree, include_top_level_imports=True)


def _collect_inlined_module_bindings(
    tree: ast.Module, *, include_top_level_imports: bool = False
) -> set[str]:
    """Names bound at module scope.

    Module scope means anything not inside a function, async function, class,
    or lambda body. *Conditional* blocks at top level -- ``try/except``,
    ``if/else``, ``with``, ``for``, ``while`` -- bind names at module scope
    even when they sit inside a compound statement, so we recurse into them.

    When ``include_top_level_imports`` is False (the default, used by the
    cross-source collision check), direct top-level ``Import`` /
    ``ImportFrom`` statements are skipped because the build pipeline strips
    them and tracks their bindings separately via
    ``_bound_names_with_source``. When True (used by export verification),
    they are included so that re-exports via top-level import survive.
    """

    collector = _ModuleScopeBindingCollector()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if include_top_level_imports:
                collector.visit(node)
            continue
        collector.visit(node)
    return collector.names


class _ModuleScopeBindingCollector(ast.NodeVisitor):
    """AST visitor that records every name bound at module global scope.

    Function / async-function / class / lambda bodies are *not* recursed into
    because their bindings live in their own local scope. Everything else is
    recursed into so that imports and assignments nested inside ``try``,
    ``if``, ``with``, ``for``, and ``while`` blocks are still captured.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        # Lambdas have parameters but those bind in lambda-local scope.
        return

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            _collect_assign_names(target, self.names)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        _collect_assign_names(node.target, self.names)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        _collect_assign_names(node.target, self.names)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name == "*":
                continue
            self.names.add(alias.asname or alias.name)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        _collect_assign_names(node.target, self.names)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
        _collect_assign_names(node.target, self.names)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        for item in node.items:
            if item.optional_vars is not None:
                _collect_assign_names(item.optional_vars, self.names)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:  # noqa: N802
        for item in node.items:
            if item.optional_vars is not None:
                _collect_assign_names(item.optional_vars, self.names)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        # Note: ``except E as exc:`` does NOT permanently bind ``exc`` -- per
        # the Python language reference, the as-target is implicitly deleted
        # at the end of the except clause to break the traceback's reference
        # cycle. We therefore must not add ``handler.name`` to the global
        # binding set, only recurse into the handler body to catch any real
        # bindings made there (e.g., ``helper = something`` inside an except).
        self.generic_visit(node)


def _verify_no_name_collisions(
    *,
    target_module: _ParsedModule,
    target_local_imports: list[ast.ImportFrom],
    target_external_imports: list[ast.stmt],
    deps_ordered: list[_DepModule],
) -> None:
    """Refuse builds where two inlined sources bind the same top-level name.

    After inlining, every dep body, the target body, and the merged top-level
    imports share one module namespace. If two of them bind the same name --
    whether via ``def helper`` in two different shared modules, or via
    ``from x import helper`` and ``def helper`` in another shared module --
    Python silently keeps the last definition and routes earlier callers'
    global lookups to the wrong target. The builder catches this up front so
    the error appears at build time rather than as a runtime mystery.

    External top-level imports themselves are deduped and conflict-checked
    elsewhere (in ``_format_merged_imports``); this pass cross-checks names
    bound by imports against names bound by code.
    """

    seen: dict[str, str] = {}

    def _record(name: str, source_label: str) -> None:
        existing = seen.get(name)
        if existing is not None and existing != source_label:
            raise BuildError(
                f"Top-level name {name!r} is bound by both {existing} and "
                f"{source_label}. Inlining would collapse them into one module "
                f"and silently pick the last binding. Rename one before "
                f"rebuilding."
            )
        seen[name] = source_label

    # Names bound by code in shared deps (functions / classes / assignments).
    for dep in deps_ordered:
        for name in _collect_inlined_module_bindings(dep.tree):
            _record(name, f"{dep.path} (definition)")

    # Names bound by code in the target.
    target_locally_bound = {
        alias.name
        for node in target_local_imports
        for alias in node.names
        if alias.name != "*"
    }
    for name in _collect_inlined_module_bindings(target_module.tree):
        if name in target_locally_bound:
            raise BuildError(
                f"{target_module.path}: top-level name {name!r} is both "
                f"imported from a local shared module and redefined in the "
                f"target body."
            )
        _record(name, f"{target_module.path} (definition)")

    # Names that the merged-import section will bind. We need to flag the
    # case where, say, a dep defines ``helper`` and the target imports
    # ``from foo import helper`` -- both end up at module scope after
    # inlining and would collide silently.
    #
    # The label is the import's stringified form so two identical imports
    # from different modules dedupe rather than colliding (this matches the
    # later ``_format_merged_imports`` dedupe). Distinct imports binding the
    # same name -- e.g., ``from json import loads`` vs ``from yaml import
    # loads`` -- will have different labels and properly collide.
    merged_imports = list(target_external_imports) + [
        imp for dep in deps_ordered for imp in dep.external_imports
    ]
    for node in merged_imports:
        for bound, source_repr in _bound_names_with_source(node):
            _record(bound, source_repr)


def _collect_assign_names(target: ast.expr, names: set[str]) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_assign_names(elt, names)


def _verify_dep_exports(
    deps_ordered: list[_DepModule], target_seeds: list[_ImportRef]
) -> None:
    by_path = {dep.path: dep for dep in deps_ordered}
    # Verify both the target's direct seeds and each dep's transitive seeds.
    all_refs: list[_ImportRef] = list(target_seeds)
    for dep in deps_ordered:
        all_refs.extend(dep.local_imports)
    for ref in all_refs:
        dep = by_path.get(ref.target_path)
        if dep is None:
            raise BuildError(
                f"{ref.importer_path}:{ref.lineno}: dependency on "
                f"{ref.target_module!r} was not resolved (internal error)."
            )
        for name in ref.names:
            if name not in dep.exported_names:
                raise BuildError(
                    f"{ref.importer_path}:{ref.lineno}: name {name!r} is not "
                    f"defined at the top level of {ref.target_module} "
                    f"({dep.path})."
                )


# ---------------------------------------------------------------------------
# Source slicing & formatting
# ---------------------------------------------------------------------------


def _slice_excluding(source: str, line_ranges: list[tuple[int, int]]) -> str:
    if not line_ranges:
        return _normalize_blank_lines(source)
    lines = source.splitlines(keepends=True)
    keep = [True] * len(lines)
    for start, end in line_ranges:
        for i in range(start - 1, end):
            if 0 <= i < len(keep):
                keep[i] = False
    sliced = "".join(lines[i] for i in range(len(lines)) if keep[i])
    return _normalize_blank_lines(sliced)


def _normalize_blank_lines(text: str) -> str:
    out: list[str] = []
    blank_run = 0
    for line in text.splitlines(keepends=True):
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                out.append(line if line.endswith("\n") else line + "\n")
        else:
            blank_run = 0
            out.append(line)
    result = "".join(out)
    return result.lstrip("\n").rstrip() + ("\n" if result.endswith("\n") else "")


def _format_future_imports(nodes: list[ast.ImportFrom]) -> str:
    names = _ordered_future_names(nodes)
    if not names:
        return ""
    return f"from __future__ import {', '.join(names)}\n"


def _format_future_names(nodes: list[ast.ImportFrom]) -> str:
    names = _ordered_future_names(nodes)
    return ", ".join(names)


def _ordered_future_names(nodes: list[ast.ImportFrom]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        for alias in node.names:
            key = (
                alias.name
                if alias.asname is None
                else f"{alias.name} as {alias.asname}"
            )
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def _format_merged_imports(
    target_imports: list[ast.stmt],
    dep_imports: list[ast.stmt],
) -> str:
    """Merge target + dep top-level external imports.

    Bound names are tracked across all sources; if the same bound name comes
    from different sources, raise BuildError. Identical statements dedupe.
    """

    bound_to_source: dict[str, str] = {}
    seen_lines: set[str] = set()
    out_lines: list[str] = []

    for node in target_imports + dep_imports:
        line = ast.unparse(node)
        if line in seen_lines:
            continue
        for bound, source_repr in _bound_names_with_source(node):
            existing = bound_to_source.get(bound)
            if existing is not None and existing != source_repr:
                raise BuildError(
                    f"Conflicting top-level import bindings for name {bound!r}: "
                    f"{existing} vs {source_repr}."
                )
            bound_to_source[bound] = source_repr
        seen_lines.add(line)
        out_lines.append(line)

    return "\n".join(out_lines) + ("\n" if out_lines else "")


def _bound_names_with_source(node: ast.stmt) -> Iterable[tuple[str, str]]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.asname is None:
                # ``import a.b.c`` binds the *leftmost* package name (``a``)
                # at module scope, and that ``a`` is the same package object
                # regardless of which submodule the import targets. So
                # ``import email.mime.text`` and ``import email.mime.multipart``
                # both bind ``email`` to the same package and are mergeable.
                # The canonical source for collision purposes is therefore
                # the leftmost package, not the full submodule path.
                bound = alias.name.split(".")[0]
                source_repr = f"import {bound}"
            else:
                # ``import a.b as alias`` binds ``alias`` directly to the
                # submodule object. Two such imports with the same alias but
                # different targets (e.g., ``import a.x as foo`` vs
                # ``import a.y as foo``) bind different objects -- those *do*
                # conflict, so the source repr keeps the full target.
                bound = alias.asname
                source_repr = f"import {alias.name} as {alias.asname}"
            yield bound, source_repr
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            source_repr = f"from {node.module or ''} import {alias.name}"
            yield bound, source_repr


def _assemble_output(
    *,
    target_module: _ParsedModule,
    target_doc_range: tuple[int, int] | None,
    merged_future_text: str,
    merged_future_names: str,
    merged_external_text: str,
    deps_ordered: list[_DepModule],
    target_body_text: str,
    repo_root: Path,
    target_name: str | None,
) -> str:
    sections: list[str] = []

    if target_doc_range is not None:
        lines = target_module.source.splitlines(keepends=True)
        doc_text = "".join(lines[target_doc_range[0] - 1 : target_doc_range[1]])
        sections.append(doc_text.rstrip("\n"))

    if merged_future_text:
        sections.append(merged_future_text.rstrip("\n"))

    rel_source = _relative_to(target_module.path, repo_root)
    regen_arg = f"--target {target_name}" if target_name else "--all"
    marker_lines = [
        GENERATED_MARKER,
        f"# Source: {rel_source}",
        f"# Regenerate with: uv run python scripts/build_release.py {regen_arg}",
        f"# Future imports: {merged_future_names or '(none)'}",
        "# See release.toml for target definitions.",
    ]
    sections.append("\n".join(marker_lines))

    if merged_external_text:
        sections.append(merged_external_text.rstrip("\n"))

    for dep in deps_ordered:
        rel_dep = _relative_to(dep.path, repo_root)
        dep_block = f"# --- inlined from {rel_dep} ({dep.dotted_name}) ---"
        dep_externals_text = _format_merged_imports(
            list(dep.external_imports), []
        ).rstrip("\n")
        if dep_externals_text:
            dep_block = f"{dep_block}\n{dep_externals_text}"
        body = dep.body_text.strip("\n")
        if body:
            dep_block = f"{dep_block}\n{body}"
        sections.append(dep_block)

    body = target_body_text.strip("\n")
    if body:
        sections.append(body)

    return "\n\n".join(sections).rstrip() + "\n"


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
