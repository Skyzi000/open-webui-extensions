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
   The target's ``from __future__`` flags are emitted once at the top;
   every dep must declare the exact same ``__future__`` set (verified up
   front) so the merged compilation unit's parse semantics match every
   source file.
6. Emits a generated file with a do-not-edit marker.

Anything that breaks these invariants -- aliases on local shared imports,
``import owui_ext.shared.foo`` style, star imports, function-level local
imports, package-relative imports, top-level imports interleaved with
non-import code, an external import that follows a local-shared import,
a shared dep module that mixes external and local-shared imports,
``from __future__`` flags that diverge between target and any dep,
circular imports, or imported names that don't exist -- raises
``BuildError``.

The original source-text of non-import statements is preserved by slicing the
file by AST line numbers rather than re-emitting via ``ast.unparse``. This
keeps comments and formatting intact.
"""

from __future__ import annotations

import ast
import hashlib
import symtable
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

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
    _verify_no_future_annotations(target_module, target_future)
    _verify_no_inline_semicolons(target_module)
    _verify_no_relative_imports(target_module)
    _verify_no_star_imports(target_module)
    _verify_no_unbounded_globals_access(target_module)
    _verify_no_unknown_globals_mutation(target_module)
    _verify_no_unknown_globals_reads(target_module)
    _verify_no_annotations_dict_access(target_module, is_target=True)
    _verify_no_alternative_globals_access(target_module)
    _verify_no_builtins_rebinding(target_module)
    _verify_no_module_scope_walrus_or_match(target_module)
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
    _verify_future_imports_match(
        target_path=target_module.path,
        target_future=target_future,
        deps_ordered=deps_ordered,
    )
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


def _future_names(nodes: list[ast.ImportFrom]) -> set[str]:
    return {alias.name for node in nodes for alias in node.names}


def _verify_no_future_annotations(
    parsed: _ParsedModule, future_imports: list[ast.ImportFrom]
) -> None:
    """Refuse ``from __future__ import annotations`` in any inlined
    source -- target included.

    PEP 563 is per-compilation-unit, so it cannot be opted into
    half-and-half across the merged file:

    * In a DEP, the flag turns annotations into strings that
      ``typing.get_type_hints``, ``dataclasses``, ``pydantic`` etc.
      later evaluate against ``fn.__globals__`` -- which after
      inlining is the merged target dict. Names that were not visible
      in the dep originally now resolve through other deps.
    * In the TARGET (with deps that lack the flag), the merged file
      still has the flag at the top, so PEP 563 applies to the dep
      bodies too. Annotations that were live (def-time evaluated) in
      the dep's original source become strings in the inlined output,
      changing runtime annotation semantics for the dep.

    Either direction is a silent semantic change vs. the source. The
    canonical fix is to drop the future import: in modern Python
    (3.9+) every type-hinting form needed for normal code (``list[int]``,
    ``X | Y``, etc.) works without PEP 563, so deps and targets that
    used the flag for that reason can simply omit it.
    """

    for node in future_imports:
        for alias in node.names:
            if alias.name == "annotations":
                raise BuildError(
                    f"{parsed.path}:{node.lineno}: "
                    f"`from __future__ import annotations` is not "
                    f"allowed in inlined sources (target or deps). "
                    f"PEP 563 is per-compilation-unit, so opting in "
                    f"on either side changes the merged module's "
                    f"annotation semantics for every other source. "
                    f"Drop the future import; modern Python (3.9+) "
                    f"supports `list[int]`, `X | Y`, etc. without it."
                )


def _verify_no_inline_semicolons(parsed: _ParsedModule) -> None:
    """Refuse sources where multiple top-level statements share one line.

    The slicer that drops inlined imports / the leading docstring
    works on whole *physical lines* of ``parsed.source``. When two
    top-level statements share a line via a semicolon
    (e.g. ``from owui_ext.shared.x import y; Z = 42``), removing the
    import range would also remove the co-located ``Z = 42`` and the
    generated single file would raise ``NameError`` at runtime.

    Walk every line each top-level node spans (``lineno``..``end_lineno``)
    and refuse the build if a later body node touches a line already
    claimed. This catches both the simple case (one-line import with
    a trailing ``; Z = 42``) and the multi-line case where the
    semicolon attaches to the closing paren's line of a parenthesised
    ``from X import (a, b,); Z = 42`` -- ``ast`` records the import's
    ``end_lineno`` on that final line, and the assign starts there
    too. Refusing the pattern keeps the slicer's "line in, line out"
    invariant simple; nobody writes ``import x; y = 1`` in real code.
    """

    occupied: dict[int, ast.stmt] = {}
    for node in parsed.tree.body:
        start = node.lineno
        end = getattr(node, "end_lineno", None) or start
        for line in range(start, end + 1):
            if line in occupied:
                other = occupied[line]
                raise BuildError(
                    f"{parsed.path}:{line}: multiple top-level "
                    f"statements share line {line} "
                    f"({type(other).__name__} and {type(node).__name__}). "
                    f"Split each statement onto its own line; the inliner "
                    f"strips imports by physical line, so a co-located "
                    f"statement would be silently dropped from the build."
                )
            occupied[line] = node


def _verify_future_imports_match(
    *,
    target_path: Path,
    target_future: list[ast.ImportFrom],
    deps_ordered: list["_DepModule"],
) -> None:
    """Refuse builds where any dep's ``from __future__`` set differs from the target.

    ``__future__`` flags change parse-level semantics for the whole
    compilation unit. After inlining, every dep body lives in the
    *target's* compilation unit, so the dep's original future flags
    no longer apply -- only the target's do. If the target opts out
    of a flag the dep relied on, the dep's code may break silently;
    if a dep adds a flag the target lacks, the target's code now
    runs under different semantics. Either direction is a silent
    semantic change, so we refuse builds where the sets differ.
    (``annotations`` is refused outright by
    ``_verify_no_future_annotations`` -- this check covers the
    remaining flags, which are all no-ops in modern Python.)
    """

    target_set = _future_names(target_future)
    for dep in deps_ordered:
        dep_set = _future_names(list(dep.future_imports))
        if dep_set == target_set:
            continue
        missing_in_dep = sorted(target_set - dep_set)
        extra_in_dep = sorted(dep_set - target_set)
        bits: list[str] = []
        if extra_in_dep:
            bits.append(
                f"dep adds {extra_in_dep!r} that the target does not declare"
            )
        if missing_in_dep:
            bits.append(
                f"dep is missing {missing_in_dep!r} that the target declares"
            )
        raise BuildError(
            f"{dep.path}: `from __future__` imports diverge from "
            f"{target_path} ({'; '.join(bits)}). Inlining shares one "
            f"compilation unit, so the dep's flags must match the "
            f"target's exactly. Align the `from __future__ import ...` "
            f"line in both files."
        )


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


def _verify_no_module_scope_walrus_or_match(parsed: _ParsedModule) -> None:
    """Refuse walrus and ``match`` evaluated at module scope.

    Module scope means the place where the AST node is executed when the
    module is imported. The walked surfaces are:

    - Module-body statements themselves (top-level statements).
    - Decorators, default argument values, base classes, class keywords,
      and parameter / return annotations of any ``def``/``async def``/
      ``class``/``lambda`` whose **defining statement** lives at module
      scope: those expressions execute in the enclosing scope when the
      def/class statement runs.

    Function, method, lambda, and class bodies are intentionally **not**
    walked. A walrus inside a method default like
    ``class Demo: def method(self, x=(helper := 1)): ...`` evaluates in
    the class's own scope (and binds in the class namespace), not module
    globals; same for ``def outer(): def inner(x=(y := 1)): ...`` whose
    inner default runs in ``outer``'s local scope.

    The orthogonal danger -- ``def f(): global helper; helper = ...`` --
    is handled by ``_collect_global_writes`` feeding the
    name-collision detector, not by this verifier.
    """

    visitor = _WalrusMatchChecker(parsed.path)
    visitor.visit(parsed.tree)


class _WalrusMatchChecker(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        raise BuildError(
            f"{self.path}:{node.lineno}: walrus expressions (``:=``) at "
            f"module scope are not allowed. The single-file output's "
            f"collision detector cannot track the bound name; rewrite as "
            f"a regular assignment."
        )

    def visit_Match(self, node: ast.Match) -> None:  # noqa: N802
        raise BuildError(
            f"{self.path}:{node.lineno}: ``match`` statements at module "
            f"scope are not allowed. The single-file output's collision "
            f"detector cannot track names bound by capture patterns; move "
            f"the ``match`` inside a function."
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_def_outer(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_def_outer(node)

    def _visit_def_outer(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        self._visit_arguments(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        # Body is intentionally not walked: bindings inside it are
        # function-local. ``global``-mediated module-global writes are
        # tracked by ``_collect_global_writes`` and the collision detector.

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._visit_arguments(node.args)
        # Lambda body intentionally not walked.

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw)
        # Class body intentionally not walked.

    def _visit_arguments(self, args: ast.arguments) -> None:
        for default in args.defaults:
            self.visit(default)
        for default in args.kw_defaults:
            if default is not None:
                self.visit(default)
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if args.vararg is not None and args.vararg.annotation is not None:
            self.visit(args.vararg.annotation)
        if args.kwarg is not None and args.kwarg.annotation is not None:
            self.visit(args.kwarg.annotation)


def _ast_has_type_alias() -> bool:
    """True on Python 3.12+, where ``ast.TypeAlias`` exists.

    Wrapping every ``ast.TypeAlias`` reference in this guard keeps the
    builder importable on older Pythons while still accepting PEP 695
    sources on 3.12+. ``hasattr`` resolves at call time, but the
    attribute is module-level on ``ast`` so the lookup is cheap.
    """

    return hasattr(ast, "TypeAlias")


def _collect_global_writes(tree: ast.AST) -> set[str]:
    """Names that some function/class scope in this module both declares
    ``global`` AND actually rebinds.

    Background. The name-collision detector currently only walks
    module-scope bindings (``def``/``class``/``Assign``/imports/etc.). It
    does not look inside function or class bodies. Without help, code
    like::

        # shared_a.py
        def clobber():
            global helper
            helper = lambda: "a"

        # shared_b.py
        def helper():
            return "b"

    would inline cleanly -- the collision detector sees only ``clobber``
    from shared_a and ``helper`` from shared_b, no overlap. But after
    inlining, ``shared_a.clobber()`` writes ``helper`` into the merged
    module's globals (the same namespace shared_b's ``def helper``
    populated), and the call from the target now invokes the wrong
    implementation.

    Why "actually rebinds". A bare ``global X`` declaration *without* any
    rebinding -- e.g., ``def run(): global helper; return helper()`` --
    does not write to module globals; it is just a lookup-scope hint. If
    we counted those names, a target that imports ``helper`` from a
    shared module and then references it via ``global helper`` for
    explicitness would falsely fail the
    "imported-and-redefined" check. So this collector only returns names
    that are both (a) listed in some ``Global`` statement and (b) actually
    targeted by a binding form (``Assign`` / ``AnnAssign`` / ``AugAssign``,
    ``def`` / ``class``, ``Import`` / ``ImportFrom``, ``for`` / ``with``
    targets, ``NamedExpr``, or a ``match`` capture pattern) within the
    same scope.

    Each function / async function / class body is analyzed as its own
    scope. Lambdas are not entered: their body is an expression, cannot
    contain ``global``, and walrus inside a lambda body binds in the
    lambda's local scope per PEP 572.
    """

    names: set[str] = set()

    def analyze_scope(body: list[ast.stmt]) -> None:
        scope_globals: set[str] = set()
        scope_writes: set[str] = set()
        nested_scopes: list[list[ast.stmt]] = []

        def walk_args_in_current_scope(args: ast.arguments) -> None:
            for default in args.defaults:
                walk(default)
            for default in args.kw_defaults:
                if default is not None:
                    walk(default)
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if arg.annotation is not None:
                    walk(arg.annotation)
            if args.vararg is not None and args.vararg.annotation is not None:
                walk(args.vararg.annotation)
            if args.kwarg is not None and args.kwarg.annotation is not None:
                walk(args.kwarg.annotation)

        def walk(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # The ``def`` statement itself binds ``node.name`` in the
                # current scope -- relevant if ``global node.name`` is in
                # effect here.
                scope_writes.add(node.name)
                # Decorators / argument defaults / parameter and return
                # annotations evaluate in the *current* scope (they run
                # when the ``def`` statement executes, not when the
                # function is later called). A walrus inside any of them
                # writes to this scope, so the current scope's ``global``
                # set still applies.
                for dec in node.decorator_list:
                    walk(dec)
                walk_args_in_current_scope(node.args)
                if node.returns is not None:
                    walk(node.returns)
                # Body is the function's own scope; analyzed separately.
                nested_scopes.append(node.body)
                return
            if isinstance(node, ast.ClassDef):
                scope_writes.add(node.name)
                # Decorators, bases, and keywords evaluate in the
                # current scope, just like for ``def``.
                for dec in node.decorator_list:
                    walk(dec)
                for base in node.bases:
                    walk(base)
                for kw in node.keywords:
                    walk(kw)
                nested_scopes.append(node.body)
                return
            if isinstance(node, ast.Lambda):
                # Lambda body is an expression and is lambda-local. But
                # the lambda's defaults evaluate in the current scope
                # when the ``lambda`` expression itself is evaluated.
                walk_args_in_current_scope(node.args)
                return
            if isinstance(node, ast.ExceptHandler):
                # ``except E as X`` *temporarily* binds ``X`` and then
                # implicitly deletes it at the end of the clause. With
                # ``global X`` in effect, that bind+delete acts on the
                # module global, which corrupts shared state in the
                # merged file. Treat the as-name as a write so the
                # collision detector picks it up.
                if node.name is not None:
                    scope_writes.add(node.name)
                # Continue walking the handler body for further writes.

            if isinstance(node, ast.Global):
                scope_globals.update(node.names)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    _collect_assign_names(target, scope_writes)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                _collect_assign_names(node.target, scope_writes)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                _collect_assign_names(node.target, scope_writes)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        _collect_assign_names(item.optional_vars, scope_writes)
            elif isinstance(node, ast.NamedExpr):
                if isinstance(node.target, ast.Name):
                    scope_writes.add(node.target.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    scope_writes.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        scope_writes.add(alias.asname or alias.name)
            elif isinstance(node, ast.Match):
                _collect_match_capture_names(node, scope_writes)
            elif _ast_has_type_alias() and isinstance(node, ast.TypeAlias):
                # PEP 695 / Python 3.12+: ``type X = ...`` binds ``X``
                # in the enclosing scope. Under ``global X`` in
                # effect here, the binding lands in module globals.
                if isinstance(node.name, ast.Name):
                    scope_writes.add(node.name.id)
            elif isinstance(node, ast.Delete):
                # ``del name`` removes the binding. Under ``global X``,
                # that removal hits the module global and erases another
                # dep's top-level name from the merged namespace. Treat
                # delete targets as writes for collision-detection
                # purposes so the build refuses up front.
                for target in node.targets:
                    _collect_assign_names(target, scope_writes)

            for child in ast.iter_child_nodes(node):
                walk(child)

        for stmt in body:
            walk(stmt)

        names.update(scope_globals & scope_writes)

        for nested in nested_scopes:
            analyze_scope(nested)

    if isinstance(tree, ast.Module):
        analyze_scope(tree.body)
    return names


def _collect_external_module_refs(
    source: str, path: Path, tree: ast.Module
) -> set[str]:
    """Names this source reads from module globals but does not bind itself.

    A name without a local or enclosing binding falls back to module
    globals at runtime, regardless of whether the surrounding function
    used an explicit ``global`` declaration -- so both ``def read():
    global helper; return helper()`` and ``def read(): return helper()``
    count. The same is true at **module scope itself**: a name appearing
    in a top-level ``def f(x=helper)`` default, decorator, base class,
    or other expression evaluated at import time looks ``helper`` up in
    module globals when the def/class statement runs.

    In the original separated modules, an unset name raises
    ``NameError``. After inlining, every dep shares one globals dict
    with the target, so the read accidentally resolves to a binding
    that happened to live in another source -- silently changing
    runtime behavior.

    Built-ins (``print``, ``len``, ``list``, ...) are intentionally NOT
    filtered out here. Python's name resolution checks module globals
    *before* the builtins module, so if any source binds (e.g.) ``list``
    at top level, every other source's ``list(...)`` reference resolves
    to that binding instead of the builtin. We rely on
    ``_verify_no_name_collisions`` Phase 2 to ignore builtin references
    that no source happens to shadow -- the cost is one ``definitions``
    lookup per builtin per source, which is negligible.

    Names returned here are checked by ``_verify_no_name_collisions``
    against actual top-level definitions in *other* sources; matching
    pairs trip a ``BuildError``. If no source binds the name, the
    behavior in source and merge is identical (both ``NameError`` for
    free identifiers, or both resolved-to-builtin for builtin names)
    and the build is allowed.
    """

    try:
        table = symtable.symtable(source, str(path), "exec")
    except (SyntaxError, ValueError):
        return set()

    bound_in_module = (
        _collect_inlined_module_bindings(tree, include_top_level_imports=True)
        | _collect_global_writes(tree)
    )

    refs: set[str] = set()

    def visit(scope: symtable.SymbolTable) -> None:
        is_module = scope.get_type() == "module"
        for sym in scope.get_symbols():
            if not sym.is_referenced():
                continue
            if is_module:
                # At module scope, "free" means referenced but not
                # bound here -- assignments / imports / def / class are
                # all bindings.
                if (
                    sym.is_assigned()
                    or sym.is_imported()
                    or sym.is_namespace()
                ):
                    continue
            else:
                # Function/class scope: only globals leak to module
                # namespace; locals / parameters / cells stay put.
                if not sym.is_global():
                    continue
            name = sym.get_name()
            if name in bound_in_module:
                continue
            refs.add(name)
        for child in scope.get_children():
            visit(child)

    visit(table)
    return refs


def _collect_class_body_unbound_refs(tree: ast.AST) -> set[str]:
    """Names referenced at class-body-evaluation level before they are
    bound in the same class scope.

    Class bodies use ``LOAD_NAME`` (not ``LOAD_GLOBAL``), which falls
    back to module globals when the name has no class-local binding at
    the point of the load. ``symtable`` cannot distinguish "use before
    bind" from "bind before use" because it merges both flags into the
    same symbol; only an ordered AST walk of the class body decides
    correctly.

    Concretely, given::

        # shared_a.py
        helper = "external"

        # shared_b.py
        class Demo:
            picked = helper        # LOAD_NAME -> module globals
            helper = "local"       # binds class-scope helper

    in the original separated modules, ``shared_b``'s ``picked = helper``
    raises ``NameError`` because ``helper`` has not been bound in class
    scope yet and ``shared_b`` has no module-global ``helper``. After
    inlining, ``shared_a``'s ``helper`` lives in the merged module's
    globals, so the ``LOAD_NAME`` falls through to it and ``Demo.picked``
    silently becomes ``"external"``.

    Walks every ``ClassDef`` reachable from ``tree`` without entering
    function / async-function / lambda bodies. For each class body, walks
    statements in source order; for each statement, yields class-eval-time
    ``Load`` references that are not yet in the accumulated bound set,
    then folds the statement's bindings into the bound set. ``return``
    annotations and parameter annotations are ignored when the module
    declares ``from __future__ import annotations``.

    Must-bind only. A name is added to the post-statement bound set
    only when every execution path that exits the statement assigns it.
    Direct sequential assignments and bindings inside a ``with`` body
    qualify; bindings inside ``for`` / ``while`` (loop body may not run)
    do not. ``if`` cascades only when both branches bind the name (i.e.,
    has an explicit ``else``); otherwise the binding is may-bind and a
    later reference is treated as unbound. ``try`` cascades only the
    intersection of try-body bindings and every ``except`` handler's
    bindings; without exhaustive exception coverage we cannot guarantee
    any single handler ran. ``match`` cascades only when at least one
    case is a wildcard / bare-name pattern (covering all values) and
    every case binds the same names.

    Names returned by this function are checked by
    ``_verify_no_name_collisions`` Phase 2 against actual top-level
    definitions in *other* sources; matching pairs trip ``BuildError``.
    """

    refs: set[str] = set()
    classes: list[tuple[ast.ClassDef, frozenset[str]]] = []
    _gather_class_definitions(tree, classes, frozenset())

    future_annotations = isinstance(tree, ast.Module) and _has_future_annotations(
        tree
    )

    # Names bound at this source's own module scope (including top-level
    # imports). A class-body load that resolves through these falls back
    # to the same source's own binding -- no cross-source global lookup,
    # no silent inline divergence. Filtering them out here mirrors how
    # ``_collect_external_module_refs`` filters its symtable results
    # against the same set.
    if isinstance(tree, ast.Module):
        bound_in_module = _collect_inlined_module_bindings(
            tree, include_top_level_imports=True
        ) | _collect_global_writes(tree)
    else:
        bound_in_module = set()

    for class_node, enclosing_function_locals in classes:
        # Class bodies use ``LOAD_NAME``, which falls through to the
        # nearest enclosing function scope (skipping enclosing class
        # scopes -- a Python quirk) before module globals. For a class
        # nested inside a function, we therefore must seed ``bound``
        # with that function's locals (and every enclosing function's
        # locals further up); otherwise a class-body load that
        # legitimately resolves to an enclosing local would be flagged.
        _walk_class_body_block(
            class_node.body,
            set(bound_in_module) | set(enclosing_function_locals),
            refs,
            future_annotations,
        )

    return refs


def _walk_class_body_block(
    stmts: Iterable[ast.stmt],
    bound: set[str],
    refs: set[str],
    future_annotations: bool,
) -> tuple[set[str], set[str]]:
    """Walk a sequence of class-body statements in execution order.

    For each statement, records class-eval-time ``Load`` references
    (names not currently in ``bound``) into ``refs``, then folds the
    statement's ``(must_add, may_remove)`` into the running ``bound``
    so that later statements in the block see the updated set.

    Returns the block's aggregated ``(must_add, may_remove)`` so the
    caller can compose it into the enclosing scope's classification.
    """

    must_add: set[str] = set()
    may_remove: set[str] = set()
    for stmt in stmts:
        s_add, s_remove = _walk_class_body_stmt(
            stmt, bound, refs, future_annotations
        )
        # Update the running bound set for the next sibling statement,
        # then fold this statement's effect into the block aggregate.
        bound = (bound | s_add) - s_remove
        must_add = (must_add | s_add) - s_remove
        may_remove = (may_remove - s_add) | s_remove
    return must_add, may_remove


def _walk_class_body_stmt(
    stmt: ast.stmt,
    bound: set[str],
    refs: set[str],
    future_annotations: bool,
) -> tuple[set[str], set[str]]:
    """Walk a single class-body statement, recording class-eval-time
    ``Load`` references against the appropriate ``bound`` snapshot for
    each sub-block.

    Compound statements introduce header bindings (``for X in iter:``,
    ``with cm() as X:``, ``except E as X:``, match capture patterns)
    that are visible inside their bodies. The simple statement-by-
    statement walker would miss these and falsely flag references like
    ``with nullcontext() as helper: picked = helper`` as unbound. This
    walker threads ``bound`` correctly through each sub-scope.

    Returns ``(must_add, may_remove)`` for the statement's effect on
    the post-statement bound set, matching the contract of
    ``_classify_class_body_stmt``.
    """

    if isinstance(stmt, ast.If):
        for name in _iter_class_eval_loads(stmt.test, future_annotations):
            if name not in bound:
                refs.add(name)
        body_add, body_remove = _walk_class_body_block(
            stmt.body, set(bound), refs, future_annotations
        )
        if not stmt.orelse:
            return (set(), body_remove)
        else_add, else_remove = _walk_class_body_block(
            stmt.orelse, set(bound), refs, future_annotations
        )
        return (body_add & else_add, body_remove | else_remove)

    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        for name in _iter_class_eval_loads(stmt.iter, future_annotations):
            if name not in bound:
                refs.add(name)
        target_names: set[str] = set()
        _collect_assign_names(stmt.target, target_names)
        body_add, body_remove = _walk_class_body_block(
            stmt.body, bound | target_names, refs, future_annotations
        )
        else_add, else_remove = _walk_class_body_block(
            stmt.orelse, set(bound), refs, future_annotations
        )
        # Loop body may not run (empty iterable); nothing is must-added
        # by the loop. The body's may_remove still applies because if
        # the loop runs at least once, those deletions could happen.
        # ``orelse`` runs only when the loop completes without ``break``
        # so it cannot be relied on for must_add either.
        del body_add, else_add  # intentionally unused
        return (set(), body_remove | else_remove)

    if isinstance(stmt, ast.While):
        for name in _iter_class_eval_loads(stmt.test, future_annotations):
            if name not in bound:
                refs.add(name)
        body_remove = _walk_class_body_block(
            stmt.body, set(bound), refs, future_annotations
        )[1]
        else_remove = _walk_class_body_block(
            stmt.orelse, set(bound), refs, future_annotations
        )[1]
        return (set(), body_remove | else_remove)

    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        body_bound = set(bound)
        as_targets: set[str] = set()
        for item in stmt.items:
            for name in _iter_class_eval_loads(
                item.context_expr, future_annotations
            ):
                if name not in body_bound:
                    refs.add(name)
            if item.optional_vars is not None:
                _collect_assign_names(item.optional_vars, as_targets)
                body_bound = body_bound | as_targets
        body_add, body_remove = _walk_class_body_block(
            stmt.body, body_bound, refs, future_annotations
        )
        return ((as_targets | body_add) - body_remove, body_remove)

    if isinstance(stmt, (ast.Try, ast.TryStar)):
        # ``ast.Try`` and ``ast.TryStar`` (Python 3.11 ``except*``) share
        # the same field layout (``body``/``handlers``/``orelse``/
        # ``finalbody``) and the same class-body name-resolution
        # semantics: each handler's ``except[*] E as name`` is implicitly
        # deleted at handler end, control may exit via the try body, the
        # else clause, or any handler's body.
        try_add, try_remove = _walk_class_body_block(
            stmt.body, set(bound), refs, future_annotations
        )
        if stmt.orelse:
            else_bound = (bound | try_add) - try_remove
            else_add, else_remove = _walk_class_body_block(
                stmt.orelse, else_bound, refs, future_annotations
            )
            try_path_add = (try_add | else_add) - else_remove
            try_path_remove = (try_remove - else_add) | else_remove
        else:
            try_path_add = try_add
            try_path_remove = try_remove
        if stmt.handlers:
            handler_intersect_add: set[str] | None = None
            handler_union_remove: set[str] = set()
            for handler in stmt.handlers:
                if handler.type is not None:
                    for name in _iter_class_eval_loads(
                        handler.type, future_annotations
                    ):
                        if name not in bound:
                            refs.add(name)
                handler_bound = set(bound)
                if handler.name is not None:
                    handler_bound.add(handler.name)
                h_add, h_remove = _walk_class_body_block(
                    handler.body, handler_bound, refs, future_annotations
                )
                # ``except E as name`` is implicitly deleted at handler
                # end -- record may_remove regardless of body deletes.
                if handler.name is not None:
                    h_remove = h_remove | {handler.name}
                    h_add = h_add - {handler.name}
                if handler_intersect_add is None:
                    handler_intersect_add = set(h_add)
                else:
                    handler_intersect_add &= h_add
                handler_union_remove |= h_remove
            assert handler_intersect_add is not None
            cluster_add = try_path_add & handler_intersect_add
            cluster_remove = try_path_remove | handler_union_remove
        else:
            cluster_add = try_path_add
            cluster_remove = try_path_remove
        # ``finally`` runs whether the cluster completed normally or
        # raised (and might re-raise). The bound state at finally start
        # is the conservative "guaranteed bound across every path that
        # leads here" state -- i.e., names that survived both the
        # try-path and every reachable handler. That is exactly the
        # cluster_add/cluster_remove projection we just computed.
        finally_bound = (bound | cluster_add) - cluster_remove
        finally_add, finally_remove = _walk_class_body_block(
            stmt.finalbody, set(finally_bound), refs, future_annotations
        )
        cluster_add = (cluster_add | finally_add) - finally_remove
        cluster_remove = (cluster_remove - finally_add) | finally_remove
        return (cluster_add, cluster_remove)

    if isinstance(stmt, ast.Match):
        for name in _iter_class_eval_loads(stmt.subject, future_annotations):
            if name not in bound:
                refs.add(name)
        has_default = False
        case_adds: list[set[str]] = []
        case_removes: set[str] = set()
        for case_ in stmt.cases:
            # ``case Helper():`` (MatchClass) loads ``Helper`` at match
            # time; ``case Mod.SENTINEL:`` (MatchValue) loads ``Mod``;
            # ``case {key: pattern}`` (MatchMapping) evaluates each
            # ``key``. None of these loads see same-statement class
            # captures yet -- the pattern is checked against the same
            # ``bound`` that held when we entered the ``match``. Drop
            # them into ``refs`` if they aren't already bound.
            for expr in _iter_pattern_runtime_load_exprs(case_.pattern):
                for name in _iter_class_eval_loads(expr, future_annotations):
                    if name not in bound:
                        refs.add(name)
            captures: set[str] = set()
            _collect_pattern_capture_names(case_.pattern, captures)
            case_bound = bound | captures
            if case_.guard is not None:
                for name in _iter_class_eval_loads(
                    case_.guard, future_annotations
                ):
                    if name not in case_bound:
                        refs.add(name)
            body_add, body_remove = _walk_class_body_block(
                case_.body, case_bound, refs, future_annotations
            )
            case_adds.append((captures | body_add) - body_remove)
            case_removes |= body_remove
            if case_.guard is None and _is_match_default_pattern(
                case_.pattern
            ):
                has_default = True
        if has_default and case_adds:
            must_add = set(case_adds[0])
            for ca in case_adds[1:]:
                must_add &= ca
        else:
            must_add = set()
        return (must_add, case_removes)

    # Simple statement (no nested control flow that introduces new
    # bindings visible to its own body): collect all class-eval Loads
    # and classify against the current bound. ``FunctionDef`` /
    # ``AsyncFunctionDef`` / ``ClassDef`` reach this branch too --
    # their decorators / defaults / annotations / bases / keywords
    # evaluate in current scope, and ``_iter_class_eval_loads`` already
    # stops at the inner body so only the header is yielded.
    for name in _iter_class_eval_loads(stmt, future_annotations):
        if name not in bound:
            refs.add(name)
    return _classify_class_body_stmt(stmt)


def _has_future_annotations(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            for alias in stmt.names:
                if alias.name == "annotations":
                    return True
    return False


def _gather_class_definitions(
    node: ast.AST,
    out: list[tuple[ast.ClassDef, frozenset[str]]],
    enclosing_function_locals: frozenset[str],
) -> None:
    """Collect every ``ClassDef`` in ``node`` together with the union of
    its enclosing function scopes' locals.

    Includes classes defined at module scope, classes nested inside
    other class bodies, AND classes defined inside function bodies.
    Function-internal class bodies execute when the function is called,
    and a class-body ``LOAD_NAME`` for a name not bound in the class
    scope falls through to the nearest enclosing *function* scope
    (skipping any enclosing class scope -- Python's class-namespaces-
    don't-leak quirk), then to module globals, then builtins. So a
    class inside ``def make(): ...`` with a ``LOAD_NAME`` for ``helper``
    that has no class-local or function-local binding would silently
    resolve to another inlined source's ``helper`` after merging.
    Knowing which function scope's locals are visible lets us avoid
    flagging the legitimate ``def make(): helper = 1; class Demo:
    picked = helper`` case while still catching the surprising ``def
    make(): class Demo: picked = helper; helper = "local"`` one.

    Lambdas can't contain class definitions (their body is an
    expression) so we don't recurse into them.
    """

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Compute this function's own locals; chain it onto the
        # accumulator so deeper nested classes see every enclosing
        # function's locals up the spine.
        new_chain = enclosing_function_locals | _compute_function_locals(node)
        for child in node.body:
            _gather_class_definitions(child, out, new_chain)
        return
    if isinstance(node, ast.Lambda):
        return
    if isinstance(node, ast.ClassDef):
        out.append((node, enclosing_function_locals))
        # Recurse into the class body for nested classes. Crucially we
        # do NOT add this class's locals to the chain, because Python
        # classes don't expose their bindings to nested classes /
        # functions / lambdas as enclosing-scope free vars.
        for child in node.body:
            _gather_class_definitions(child, out, enclosing_function_locals)
        return
    for child in ast.iter_child_nodes(node):
        _gather_class_definitions(child, out, enclosing_function_locals)


def _compute_function_locals(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Return the set of names bound in this function's local scope.

    Includes parameters, names assigned at the function's own level
    (recursing through control flow but NOT entering nested function /
    class / lambda bodies), and the names of nested ``def`` / ``async
    def`` / ``class`` statements (those bind a local in the enclosing
    function). Excludes any name declared ``global`` or ``nonlocal``
    at this scope -- those don't live in the function's locals and so
    don't shadow module globals for nested class-body lookups.
    """

    locals_set: set[str] = set()
    declared_global: set[str] = set()
    declared_nonlocal: set[str] = set()

    for arg in (
        *fn_node.args.posonlyargs,
        *fn_node.args.args,
        *fn_node.args.kwonlyargs,
    ):
        locals_set.add(arg.arg)
    if fn_node.args.vararg is not None:
        locals_set.add(fn_node.args.vararg.arg)
    if fn_node.args.kwarg is not None:
        locals_set.add(fn_node.args.kwarg.arg)

    def walk(n: ast.AST) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # The def/class statement itself binds n.name in current scope.
            locals_set.add(n.name)
            return  # Body is its own scope; do not descend.
        if isinstance(n, ast.Lambda):
            return
        if isinstance(n, ast.Global):
            declared_global.update(n.names)
            return
        if isinstance(n, ast.Nonlocal):
            declared_nonlocal.update(n.names)
            return
        if isinstance(n, ast.Assign):
            for target in n.targets:
                _collect_assign_names(target, locals_set)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            _collect_assign_names(n.target, locals_set)
        elif isinstance(n, ast.NamedExpr):
            if isinstance(n.target, ast.Name):
                locals_set.add(n.target.id)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            _collect_assign_names(n.target, locals_set)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars is not None:
                    _collect_assign_names(item.optional_vars, locals_set)
        elif isinstance(n, ast.ExceptHandler):
            if n.name is not None:
                locals_set.add(n.name)
        elif isinstance(n, ast.Import):
            for alias in n.names:
                locals_set.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for alias in n.names:
                if alias.name != "*":
                    locals_set.add(alias.asname or alias.name)
        elif isinstance(n, ast.Match):
            captures: set[str] = set()
            _collect_match_capture_names(n, captures)
            locals_set.update(captures)
        for child in ast.iter_child_nodes(n):
            walk(child)

    for stmt in fn_node.body:
        walk(stmt)

    return frozenset(locals_set - declared_global - declared_nonlocal)


def _iter_class_eval_loads(
    node: ast.AST, future_annotations: bool
) -> Iterator[str]:
    """Yield ``Load`` names evaluated at class-body evaluation level.

    Stops at ``FunctionDef`` / ``AsyncFunctionDef`` / ``Lambda`` *bodies*
    (those are separate scopes) but DOES walk their decorators, default
    values, parameter annotations, and return annotations because those
    expressions evaluate when the ``def``/``lambda`` statement runs --
    i.e., during the enclosing class body. Stops at nested ``ClassDef``
    *bodies* (each nested class body is analyzed by its own pass via
    ``_gather_classes_outside_functions``) but DOES walk their
    decorators, base classes, and class keywords.

    Comprehensions are their own implicit function scope: only the first
    generator's ``iter`` is evaluated in the enclosing class scope.
    """

    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load):
            yield node.id
        return

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for dec in node.decorator_list:
            yield from _iter_class_eval_loads(dec, future_annotations)
        yield from _iter_args_class_eval_loads(node.args, future_annotations)
        if node.returns is not None and not future_annotations:
            yield from _iter_class_eval_loads(node.returns, future_annotations)
        return

    if isinstance(node, ast.ClassDef):
        for dec in node.decorator_list:
            yield from _iter_class_eval_loads(dec, future_annotations)
        for base in node.bases:
            yield from _iter_class_eval_loads(base, future_annotations)
        for kw in node.keywords:
            yield from _iter_class_eval_loads(kw.value, future_annotations)
        return

    if isinstance(node, ast.Lambda):
        yield from _iter_args_class_eval_loads(node.args, future_annotations)
        return

    if isinstance(node, ast.AnnAssign):
        if node.annotation is not None and not future_annotations:
            yield from _iter_class_eval_loads(node.annotation, future_annotations)
        if node.value is not None:
            yield from _iter_class_eval_loads(node.value, future_annotations)
        # The target's outer name is in Store context (no Load), but if
        # it's an attribute / subscript its inner expression is loaded.
        if not isinstance(node.target, ast.Name):
            yield from _iter_class_eval_loads(node.target, future_annotations)
        return

    if isinstance(node, ast.AugAssign):
        # ``x += value`` reads the current value of ``x`` before storing.
        if isinstance(node.target, ast.Name):
            yield node.target.id
        else:
            yield from _iter_class_eval_loads(node.target, future_annotations)
        yield from _iter_class_eval_loads(node.value, future_annotations)
        return

    if isinstance(
        node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    ):
        if node.generators:
            yield from _iter_class_eval_loads(
                node.generators[0].iter, future_annotations
            )
        return

    for child in ast.iter_child_nodes(node):
        yield from _iter_class_eval_loads(child, future_annotations)


def _iter_args_class_eval_loads(
    args: ast.arguments, future_annotations: bool
) -> Iterator[str]:
    for default in args.defaults:
        yield from _iter_class_eval_loads(default, future_annotations)
    for default in args.kw_defaults:
        if default is not None:
            yield from _iter_class_eval_loads(default, future_annotations)
    if not future_annotations:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if arg.annotation is not None:
                yield from _iter_class_eval_loads(
                    arg.annotation, future_annotations
                )
        if args.vararg is not None and args.vararg.annotation is not None:
            yield from _iter_class_eval_loads(
                args.vararg.annotation, future_annotations
            )
        if args.kwarg is not None and args.kwarg.annotation is not None:
            yield from _iter_class_eval_loads(
                args.kwarg.annotation, future_annotations
            )


def _classify_class_body_stmt(
    node: ast.AST,
) -> tuple[set[str], set[str]]:
    """Return ``(must_add, may_remove)`` for a class-body statement.

    ``must_add`` is the set of names guaranteed to be bound after this
    statement on every execution path that exits it normally.
    ``may_remove`` is the set of names that some execution path *might*
    have unbound (via ``del`` or via the implicit cleanup at the end of
    an ``except E as name`` clause) -- conservatively dropped from the
    bound set so a subsequent load is treated as unbound.

    Bodies of nested ``def`` / ``async def`` / ``class`` / ``lambda``
    are NOT walked because their bindings are local to themselves and
    do not affect the enclosing class scope's ``bound`` set.
    """

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ({node.name}, set())
    if isinstance(node, ast.Lambda):
        return (set(), set())
    if isinstance(node, ast.Assign):
        names: set[str] = set()
        for target in node.targets:
            _collect_assign_names(target, names)
        return (names, set())
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        names = set()
        _collect_assign_names(node.target, names)
        return (names, set())
    if isinstance(node, ast.NamedExpr):
        if isinstance(node.target, ast.Name):
            return ({node.target.id}, set())
        return (set(), set())
    if isinstance(node, ast.Import):
        names = set()
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
        return (names, set())
    if isinstance(node, ast.ImportFrom):
        names = set()
        for alias in node.names:
            if alias.name != "*":
                names.add(alias.asname or alias.name)
        return (names, set())
    if isinstance(node, ast.Delete):
        removed: set[str] = set()
        for target in node.targets:
            if isinstance(target, ast.Name):
                removed.add(target.id)
        return (set(), removed)
    if isinstance(node, (ast.For, ast.AsyncFor)):
        # Loop body may not execute (empty iterable); the for-target is
        # bound only when the iterable yields. ``orelse`` runs only when
        # the loop completes without ``break``. Be conservative: nothing
        # is must-added by the loop. ``del``s inside the body still
        # contribute to may_remove because they might happen.
        body_remove = _classify_block(node.body)[1]
        else_remove = _classify_block(node.orelse)[1]
        return (set(), body_remove | else_remove)
    if isinstance(node, ast.While):
        body_remove = _classify_block(node.body)[1]
        else_remove = _classify_block(node.orelse)[1]
        return (set(), body_remove | else_remove)
    if isinstance(node, ast.If):
        body_add, body_remove = _classify_block(node.body)
        if not node.orelse:
            # No else: if branch may not run, so must_add is empty;
            # body deletions still count as may_remove.
            return (set(), body_remove)
        else_add, else_remove = _classify_block(node.orelse)
        return (body_add & else_add, body_remove | else_remove)
    if isinstance(node, (ast.With, ast.AsyncWith)):
        # ``with cm() as X: body`` always runs ``body`` if ``__enter__``
        # succeeds; reaching the statement after the ``with`` implies
        # the as-target and every must-add from the body are bound.
        as_targets: set[str] = set()
        for item in node.items:
            if item.optional_vars is not None:
                _collect_assign_names(item.optional_vars, as_targets)
        body_add, body_remove = _classify_block(node.body)
        return (as_targets | body_add, body_remove)
    if isinstance(node, (ast.Try, ast.TryStar)):
        # ``ast.Try`` and ``ast.TryStar`` (Python 3.11 ``except*``)
        # share field layout and class-body name-resolution semantics.
        # Two ways to exit the try cluster: (a) try body completes ->
        # ``orelse`` runs after it; (b) try body raises -> a matching
        # handler runs. In case (a) we have try_path_add bindings; in
        # case (b) we have whichever handler ran. Without exhaustive
        # exception coverage we take the intersection across all
        # alternatives so ``must_add`` reflects every reachable exit.
        # ``except[*] E as name`` is implicitly deleted at handler end
        # so we add ``handler.name`` to ``may_remove``. ``finally`` runs
        # unconditionally and is composed sequentially after the try
        # cluster.
        try_path_add, try_path_remove = _classify_block(node.body)
        else_add, else_remove = _classify_block(node.orelse)
        try_path_add = (try_path_add | else_add) - else_remove
        try_path_remove = (try_path_remove - else_add) | else_remove
        if node.handlers:
            handler_intersect_add: set[str] | None = None
            handler_union_remove: set[str] = set()
            for handler in node.handlers:
                h_add, h_remove = _classify_block(handler.body)
                if handler.name is not None:
                    h_remove = h_remove | {handler.name}
                if handler_intersect_add is None:
                    handler_intersect_add = set(h_add)
                else:
                    handler_intersect_add &= h_add
                handler_union_remove |= h_remove
            assert handler_intersect_add is not None
            cluster_add = try_path_add & handler_intersect_add
            cluster_remove = try_path_remove | handler_union_remove
        else:
            cluster_add = try_path_add
            cluster_remove = try_path_remove
        finally_add, finally_remove = _classify_block(node.finalbody)
        # Apply finally as a sequential update after the cluster.
        cluster_add = (cluster_add | finally_add) - finally_remove
        cluster_remove = (cluster_remove - finally_add) | finally_remove
        return (cluster_add, cluster_remove)
    if isinstance(node, ast.Match):
        # Must-bind after ``match`` requires that the cases are
        # exhaustive (some case is a wildcard / bare-name pattern with
        # no guard) AND every case binds the same names. Otherwise
        # execution may fall off the match without binding.
        has_default = False
        case_adds: list[set[str]] = []
        case_removes: set[str] = set()
        for case_ in node.cases:
            captures: set[str] = set()
            _collect_pattern_capture_names(case_.pattern, captures)
            body_add, body_remove = _classify_block(case_.body)
            case_adds.append((captures | body_add) - body_remove)
            case_removes |= body_remove
            if case_.guard is None and _is_match_default_pattern(
                case_.pattern
            ):
                has_default = True
        if has_default and case_adds:
            must_add = set(case_adds[0])
            for ca in case_adds[1:]:
                must_add &= ca
        else:
            must_add = set()
        return (must_add, case_removes)
    return (set(), set())


def _classify_block(
    stmts: Iterable[ast.stmt],
) -> tuple[set[str], set[str]]:
    """Sequentially fold ``_classify_class_body_stmt`` over a block.

    Aggregating two statements ``s1`` then ``s2`` with ``s2 = (a2,
    r2)`` and accumulator ``(A, R)``::

        A = (A | a2) - r2
        R = (R - a2) | r2

    The same identity used by the caller of this helper, so that every
    cumulative ``must_add`` survives a later may-remove only when *not*
    overwritten by another must-add."""

    must_add: set[str] = set()
    may_remove: set[str] = set()
    for stmt in stmts:
        s_add, s_remove = _classify_class_body_stmt(stmt)
        must_add = (must_add | s_add) - s_remove
        may_remove = (may_remove - s_add) | s_remove
    return must_add, may_remove


def _iter_pattern_runtime_load_exprs(
    pattern: ast.pattern,
) -> Iterator[ast.expr]:
    """Yield every expression evaluated at match-pattern check time.

    ``MatchClass(cls=...)`` looks ``cls`` up at runtime via the normal
    name-resolution rules (LOAD_NAME at class scope, LOAD_GLOBAL at
    function scope). ``MatchValue(value=...)`` evaluates ``value`` --
    typically a dotted name like ``Mod.SENTINEL``. ``MatchMapping``
    evaluates each key. ``MatchSequence`` / ``MatchOr`` / ``MatchAs``
    are containers that don't add their own loads but recurse into
    nested patterns. ``MatchSingleton`` / ``MatchStar`` only match
    constants or capture, with no expression evaluation.
    """

    if isinstance(pattern, ast.MatchValue):
        yield pattern.value
    elif isinstance(pattern, ast.MatchClass):
        yield pattern.cls
        for sub in pattern.patterns:
            yield from _iter_pattern_runtime_load_exprs(sub)
        for sub in pattern.kwd_patterns:
            yield from _iter_pattern_runtime_load_exprs(sub)
    elif isinstance(pattern, ast.MatchMapping):
        for key in pattern.keys:
            yield key
        for sub in pattern.patterns:
            yield from _iter_pattern_runtime_load_exprs(sub)
    elif isinstance(pattern, ast.MatchSequence):
        for sub in pattern.patterns:
            yield from _iter_pattern_runtime_load_exprs(sub)
    elif isinstance(pattern, ast.MatchOr):
        for sub in pattern.patterns:
            yield from _iter_pattern_runtime_load_exprs(sub)
    elif isinstance(pattern, ast.MatchAs):
        if pattern.pattern is not None:
            yield from _iter_pattern_runtime_load_exprs(pattern.pattern)


def _is_match_default_pattern(pattern: ast.pattern) -> bool:
    """Return True if ``pattern`` matches every possible subject. Wildcard
    ``_`` (``MatchAs(pattern=None, name=None)``) and bare-name capture
    ``case x:`` (``MatchAs(pattern=None, name='x')``) both match every
    value; nested ``MatchOr`` is exhaustive iff at least one alternative
    is itself exhaustive."""

    if isinstance(pattern, ast.MatchAs):
        return pattern.pattern is None
    if isinstance(pattern, ast.MatchOr):
        return any(_is_match_default_pattern(p) for p in pattern.patterns)
    return False


def _collect_module_scope_destructive_writes(tree: ast.Module) -> set[str]:
    """Names this module destructively writes at module scope via
    ``del X`` or ``except E as X``.

    ``del X`` removes ``X`` from module globals immediately. ``except E
    as X`` temporarily binds ``X`` for the duration of the handler and
    then implicitly deletes it (per the Python language reference, to
    break the traceback's reference cycle). After inlining, every source
    shares one module namespace, so either operation in one source can
    silently erase a binding established at module scope by another
    inlined source -- a quiet semantic divergence vs. the original
    separated modules where the destructive write would either no-op or
    raise ``NameError``.

    Function / async function / class / lambda bodies are NOT recursed
    into. Nested-scope ``del X`` / ``except E as X`` paired with
    ``global X`` is already tracked by ``_collect_global_writes``, which
    feeds the same collision detector.
    """

    names: set[str] = set()
    collector = _ModuleScopeDestructiveCollector(names)
    for stmt in tree.body:
        collector.visit(stmt)
    return names


class _ModuleScopeDestructiveCollector(ast.NodeVisitor):
    """Walks module-scope statements (with control-flow recursion) and
    records ``del`` targets and ``except as`` names. Function / class /
    lambda bodies are skipped because their destructive writes only hit
    module globals when explicitly declared ``global X``, which is
    already covered by ``_collect_global_writes``."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        for target in node.targets:
            _collect_assign_names(target, self.names)

    def visit_ExceptHandler(  # noqa: N802
        self, node: ast.ExceptHandler
    ) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)


def _collect_match_capture_names(
    match: ast.Match, names: set[str]
) -> None:
    for case_ in match.cases:
        _collect_pattern_capture_names(case_.pattern, names)
        if case_.guard is not None:
            for sub in ast.walk(case_.guard):
                if isinstance(sub, ast.NamedExpr) and isinstance(
                    sub.target, ast.Name
                ):
                    names.add(sub.target.id)


def _collect_pattern_capture_names(
    pattern: ast.pattern, names: set[str]
) -> None:
    if isinstance(pattern, ast.MatchAs):
        if pattern.name is not None:
            names.add(pattern.name)
        if pattern.pattern is not None:
            _collect_pattern_capture_names(pattern.pattern, names)
    elif isinstance(pattern, ast.MatchOr):
        for sub in pattern.patterns:
            _collect_pattern_capture_names(sub, names)
    elif isinstance(pattern, ast.MatchSequence):
        for sub in pattern.patterns:
            _collect_pattern_capture_names(sub, names)
    elif isinstance(pattern, ast.MatchMapping):
        for sub in pattern.patterns:
            _collect_pattern_capture_names(sub, names)
        if pattern.rest is not None:
            names.add(pattern.rest)
    elif isinstance(pattern, ast.MatchClass):
        for sub in pattern.patterns:
            _collect_pattern_capture_names(sub, names)
        for sub in pattern.kwd_patterns:
            _collect_pattern_capture_names(sub, names)
    elif isinstance(pattern, ast.MatchStar):
        if pattern.name is not None:
            names.add(pattern.name)
    # MatchValue / MatchSingleton bind no names.


def _verify_no_star_imports(parsed: _ParsedModule) -> None:
    """Refuse ``from X import *`` anywhere in the module.

    Star imports bind names that the inliner cannot statically enumerate, so
    its name-collision detection silently skips them. In the merged single-file
    output, names brought in by a star import can be shadowed by a later
    inlined dep definition (or vice versa), changing which implementation the
    target body actually calls without any build-time warning. Refuse rather
    than ship a release whose behavior diverges from the source.
    """

    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "*" for alias in node.names
        ):
            module = node.module or ""
            prefix = "." * node.level
            raise BuildError(
                f"{parsed.path}:{node.lineno}: `from {prefix}{module} import *` "
                f"is not allowed. The single-file output cannot statically "
                f"track the bound names, so they may silently collide with "
                f"inlined dep definitions. List the names explicitly."
            )


def _verify_no_builtins_rebinding(parsed: _ParsedModule) -> None:
    """Refuse modules that rebind ``__builtins__`` for the current module.

    ``__builtins__`` is the module-level reference Python uses to resolve
    every unqualified builtin name (``len``, ``print``, ...). After
    inlining, every source shares one module namespace, so any rebinding
    that lands on the merged module's ``__builtins__`` silently overrides
    builtin resolution for every other inlined source -- a quiet
    semantic divergence vs. the original separated modules where each
    module has its own ``__builtins__`` reference.

    The general name-collision detector (``_verify_no_name_collisions``)
    does not flag a lone ``__builtins__`` write because in source form
    no other module binds ``__builtins__`` at all, so it has nothing to
    collide with. ``__builtins__`` is therefore policed as a reserved
    name independent of the collision detector.

    Two rebinding shapes are both refused:

    1. **Static binding** -- ``__builtins__ = X``, ``__builtins__: T = X``,
       ``import X as __builtins__``, ``from M import X as __builtins__``,
       ``def __builtins__: ...``, ``class __builtins__: ...``,
       ``global __builtins__; __builtins__ = X``, etc. Enumerated via
       ``_collect_inlined_module_bindings`` and ``_collect_global_writes``.

    2. **Dynamic ``globals()`` / ``vars()`` / ``locals()`` writes** --
       the same set that the general collision detector now ingests,
       so writes through ``globals()['__builtins__'] = X``,
       ``globals().__setitem__('__builtins__', X)``,
       ``globals().update({'__builtins__': X})``,
       ``dict.__setitem__(globals(), '__builtins__', X)``, and
       ``dict.update(globals(), {'__builtins__': X})`` are all refused.
       Both the bound-method form (``globals().METHOD(...)``) and the
       dict-descriptor form (``dict.METHOD(globals(), ...)``) are
       normalized in ``_GlobalsMutationVisitor`` and feed the same
       refusal logic. ``vars()`` and ``locals()`` are policed only at
       module scope (where they alias ``globals()``); inside a function
       / class / lambda / comprehension body they return the local
       frame and so cannot mutate merged module globals.
       ``_verify_no_unknown_globals_mutation`` separately blocks
       ``update`` calls whose argument shape we cannot statically
       extract, so this collector only sees forms it can fully resolve.

    Documented gaps -- the inliner is meant to catch accidental silent
    divergence, not to be a malicious-code sandbox:

    * Indirect aliases (``g = globals(); g['__builtins__'] = X``) -- the
      collector matches the literal call ``globals()`` / ``vars()``
      only. Tracking dataflow into intermediate variables would require
      a real escape analysis pass.
    * Attribute writes via ``sys.modules[__name__]``, ``setattr`` on
      the current module, frame-introspection
      (``inspect.currentframe().f_globals``), and ``exec`` / ``compile``
      string contents.
    * Mutation via ``__builtins__['len'] = X`` -- this writes to the
      builtins *module* (or its dict), which is already process-wide in
      source form, so the inliner introduces no new behavior.
    """

    # ``include_top_level_imports=True`` catches ``import X as __builtins__``
    # and ``from X import Y as __builtins__`` at the top level, which the
    # collision-detection caller of this collector deliberately skips
    # (imports there are tracked elsewhere via ``_bound_names_with_source``).
    # Here we want every form of binding, so we opt in.
    static_bindings = _collect_inlined_module_bindings(
        parsed.tree, include_top_level_imports=True
    ) | _collect_global_writes(parsed.tree)
    if "__builtins__" in static_bindings:
        line = _first_builtins_store_line(parsed.tree) or 0
        raise BuildError(
            f"{parsed.path}:{line}: top-level binding of `__builtins__` "
            f"is not allowed. After inlining, this rebinding would "
            f"override builtin resolution (`len`, `print`, ...) for "
            f"every other inlined source -- a silent semantic change "
            f"vs. the original separated modules. Move the binding "
            f"inside a function (where its scope is local) or rename it "
            f"before rebuilding."
        )

    dynamic_writes = _collect_dynamic_globals_writes(parsed.tree)
    if "__builtins__" in dynamic_writes:
        raise BuildError(
            f"{parsed.path}: writing to `__builtins__` via "
            f"`globals()` / `vars()` / `locals()` (subscript, "
            f"`__setitem__`, or `update` with a literal "
            f"`'__builtins__'` key) is not allowed. After inlining, "
            f"this would override builtin resolution (`len`, `print`, "
            f"...) for every other inlined source -- a silent semantic "
            f"change vs. the original separated modules."
        )

    # Destructive removal of `__builtins__` from the merged module's
    # globals is just as disruptive as rebinding: the next builtin
    # lookup falls back to the interpreter's default ``builtins``
    # module, but if a downstream inlined source then reads
    # ``globals()['__builtins__']`` it will see ``KeyError`` instead of
    # whatever it had access to in source form. Refuse all destructive
    # forms: ``del __builtins__``, ``except E as __builtins__`` (which
    # implicitly deletes after the handler), ``del globals()['__builtins__']``,
    # and ``globals().__delitem__('__builtins__')``.
    static_destructive = _collect_module_scope_destructive_writes(parsed.tree)
    if "__builtins__" in static_destructive:
        raise BuildError(
            f"{parsed.path}: destructively removing `__builtins__` "
            f"from module scope (via `del __builtins__` or "
            f"`except E as __builtins__`, which implicitly clears the "
            f"binding after the handler) is not allowed. After "
            f"inlining, every source shares one module namespace, so "
            f"this would silently break builtin lookups in every "
            f"inlined source that runs after the deletion -- a quiet "
            f"semantic change vs. the original separated modules."
        )

    dynamic_deletes = _collect_dynamic_globals_deletes(parsed.tree)
    if "__builtins__" in dynamic_deletes:
        raise BuildError(
            f"{parsed.path}: deleting `__builtins__` via "
            f"`globals()` / `vars()` / `locals()` (`del globals()[\"__builtins__\"]` "
            f"or `globals().__delitem__(\"__builtins__\")`) is not "
            f"allowed. After inlining, this would silently break "
            f"builtin lookups in every other inlined source -- a quiet "
            f"semantic change vs. the original separated modules."
        )


# Module-level dunders that Python creates implicitly per-module and
# that every inlined source would observe AS IF its own. After
# inlining, all inlined sources share the merged module's metadata
# (``__name__`` / ``__file__`` / ...) and the merged module's
# auto-populated ``__annotations__`` dict, so any source that
# references one of these names sees a value that depends on the
# merged set rather than its own original module.
#
# We handle them under one verifier so the per-name policy stays
# consistent: refuse explicit Name references anywhere, refuse
# explicit binding / rebinding via def / class / import as / type
# alias / global write / dynamic globals call, refuse explicit
# deletion. ``__annotations__`` was the original motivation; the
# rest are module metadata dunders Codex flagged as similar leak
# vectors.
_RESERVED_MODULE_DUNDERS = (
    "__annotations__",
    "__name__",
    "__file__",
    "__package__",
    "__spec__",
    "__loader__",
    "__cached__",
    "__doc__",
    "__path__",
    "__builtins__",  # also covered separately, but tracked here for
                     # defensive duplication
)


def _verify_no_annotations_dict_access(
    parsed: _ParsedModule, *, is_target: bool = False
) -> None:
    """Refuse explicit references to module dunders that diverge
    after inlining.

    The ``__annotations__`` dunder is refused in BOTH the target and
    its deps: every module-scope ``AnnAssign`` (``x: int``, ``y: T = v``,
    ...) implicitly populates ``module.__annotations__``, and after
    inlining every source's AnnAssigns land in one shared dict. So
    even reading ``__annotations__`` from the target sees a merged
    view that did not exist in the original target alone.

    Identity dunders are refused in DEPS for the obvious reason: a
    dep reading ``__name__`` etc. saw the dep's own identity before
    inlining and the merged target's identity after. For the TARGET
    we relax exactly one name -- ``__name__`` -- because the canonical
    Python idiom ``log = logging.getLogger(__name__)`` is widely used
    and the target's ``__name__`` post-inlining still identifies the
    same module the developer intended. Every other identity dunder
    (``__file__``, ``__package__``, ``__spec__``, ``__loader__``,
    ``__cached__``, ``__doc__``, ``__path__``) stays refused in the
    target as well: ``__file__`` in particular changes between the
    dev-time source path and the inlined output path, so reads like
    ``Path(__file__).parent / "data.json"`` resolve to different
    files before and after inlining; the others have no compelling
    use case to justify the divergence risk.

    Why all depths for the unbound-Name form: Python resolves an
    unbound dunder reference inside a function / class / comprehension
    body the same way as any other free variable -- it falls back to
    module globals. So a dep with ``def get_name(): return __name__``
    reads the merged module's name at call time, even though the
    reference is syntactically inside a function body.

    Class bodies for ``__annotations__`` behave slightly differently
    (Python may implicitly create a class-local ``__annotations__``),
    but distinguishing reliably requires control-flow analysis we
    don't do; refusing across the board is the safer default.

    Annotations themselves (``x: int = 1`` etc.) are still fine to
    write -- only the explicit name references and rebindings of
    these dunders are refused.
    """

    if is_target:
        # Target relaxation: ``__name__`` only. Every other identity
        # dunder stays refused because it either (a) changes between
        # source and inlined output paths (``__file__``,
        # ``__package__``) or (b) has no realistic legitimate use
        # case (``__spec__``, ``__loader__``, ``__cached__``,
        # ``__doc__``, ``__path__``).
        forbidden_names = tuple(
            name
            for name in _RESERVED_MODULE_DUNDERS
            if name not in ("__builtins__", "__name__")
        )
    else:
        forbidden_names = tuple(
            name
            for name in _RESERVED_MODULE_DUNDERS
            if name != "__builtins__"
        )
    forbidden_set = frozenset(forbidden_names)

    visitor = _AnnotationsDictRefVisitor()
    visitor.visit(parsed.tree)
    for node in visitor.module_scope_refs:
        if node.id not in forbidden_set:
            continue
        raise BuildError(
            f"{parsed.path}:{node.lineno}: explicit reference to "
            f"`{node.id}` is not allowed. After inlining, every "
            f"source shares one module's `{node.id}` -- so a "
            f"reference here sees the merged target's value, not the "
            f"original dep's. This is a silent semantic divergence "
            f"vs. the original separated modules. "
            + (
                f"Use the annotated names directly instead of reading "
                f"the dict."
                if node.id == "__annotations__"
                else f"If you need a per-source identifier, hard-code it "
                f"as a string constant before inlining."
            )
        )

    # Reserved-name binding forms that ``visit_Name`` cannot see:
    # ``def NAME()``, ``class NAME:``, ``import X as NAME``,
    # ``from X import Y as NAME``, ``type NAME = ...`` (PEP 695).
    # These appear as module-scope bindings in
    # ``_collect_inlined_module_bindings`` / ``_collect_global_writes``.
    # Bindings/destructive writes/dynamic mutations are ALWAYS refused
    # for every reserved dunder, target included: rebinding the
    # target's own ``__name__`` etc. would still corrupt the merged
    # module's metadata. Only PASSIVE READS get the target relaxation.
    static_bindings = _collect_inlined_module_bindings(
        parsed.tree, include_top_level_imports=True
    ) | _collect_global_writes(parsed.tree)
    for name in _RESERVED_MODULE_DUNDERS:
        if name == "__builtins__":
            # Already handled with a richer error message in
            # ``_verify_no_builtins_rebinding``; skip here so the
            # message is not duplicated.
            continue
        if name in static_bindings:
            raise BuildError(
                f"{parsed.path}: top-level binding of `{name}` "
                f"(via `def`, `class`, `import ... as {name}`, "
                f"`type {name} = ...`, or a `global` write) is not "
                f"allowed. After inlining, this would replace the "
                f"merged module's per-module `{name}` and corrupt "
                f"every other inlined source's view of it."
            )

    static_destructive = _collect_module_scope_destructive_writes(parsed.tree)
    for name in _RESERVED_MODULE_DUNDERS:
        if name == "__builtins__":
            continue
        if name in static_destructive:
            raise BuildError(
                f"{parsed.path}: destructively removing `{name}` "
                f"from module scope (via `del {name}` or `except E "
                f"as {name}`) is not allowed. After inlining, this "
                f"would silently drop the merged module's `{name}` "
                f"out from under every other inlined source."
            )

    dyn_writes = _collect_dynamic_globals_writes(parsed.tree)
    dyn_deletes = _collect_dynamic_globals_deletes(parsed.tree)
    dyn_reads = _collect_dynamic_globals_reads(parsed.tree)
    for name in _RESERVED_MODULE_DUNDERS:
        if name == "__builtins__":
            continue
        if name in dyn_writes:
            raise BuildError(
                f"{parsed.path}: writing to `{name}` via `globals()` "
                f"/ `vars()` / `locals()` (subscript, `__setitem__`, "
                f"or `update` with a literal `'{name}'` key) is not "
                f"allowed -- same reason as the static binding form."
            )
        if name in dyn_deletes:
            raise BuildError(
                f"{parsed.path}: deleting `{name}` via `globals()` "
                f"/ `vars()` / `locals()` is not allowed -- same "
                f"reason as `del {name}`."
            )
        if name in dyn_reads and name in forbidden_set:
            raise BuildError(
                f"{parsed.path}: reading `{name}` via "
                f"`globals()['{name}']`, `globals().get('{name}')`, "
                f"or `'{name}' in globals()` is not allowed -- the "
                f"value depends on the merged set of inlined sources, "
                f"a silent semantic divergence from the original "
                f"module."
            )


class _AnnotationsDictRefVisitor(ast.NodeVisitor):
    """Scope-aware walker that records every reference to a reserved
    module dunder (``__annotations__``, ``__name__``, ``__file__``,
    ``__package__``, ``__spec__``, ``__loader__``, ``__cached__``,
    ``__doc__``, ``__path__``) at any depth -- regardless of context
    (Load / Store / Del). Mirrors ``_GlobalsMutationVisitor``'s
    header / body scope split so the depth field is consistent with
    other visitors, although references at every depth are flagged.
    """

    def __init__(self) -> None:
        self.depth = 0
        self.module_scope_refs: list[ast.Name] = []

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function_like_header(node)
        self.depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= 1

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._visit_function_like_header(node)
        self.depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= 1

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._visit_arguments_header(node.args)
        self.depth += 1
        self.visit(node.body)
        self.depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        for d in node.decorator_list:
            self.visit(d)
        for b in node.bases:
            self.visit(b)
        for kw in node.keywords:
            self.visit(kw)
        for tp in getattr(node, "type_params", []) or []:
            self.visit(tp)
        self.depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._visit_comprehension(node, ("elt",))

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        self._visit_comprehension(node, ("elt",))

    def visit_GeneratorExp(  # noqa: N802
        self, node: ast.GeneratorExp
    ) -> None:
        self._visit_comprehension(node, ("elt",))

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        self._visit_comprehension(node, ("key", "value"))

    def _visit_function_like_header(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for d in node.decorator_list:
            self.visit(d)
        self._visit_arguments_header(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for tp in getattr(node, "type_params", []) or []:
            self.visit(tp)

    def _visit_arguments_header(self, args: ast.arguments) -> None:
        for d in args.defaults:
            self.visit(d)
        for d in args.kw_defaults:
            if d is not None:
                self.visit(d)
        for arg_list in (
            getattr(args, "posonlyargs", []) or [],
            args.args,
            args.kwonlyargs,
        ):
            for a in arg_list:
                if a.annotation is not None:
                    self.visit(a.annotation)
        if args.vararg is not None and args.vararg.annotation is not None:
            self.visit(args.vararg.annotation)
        if args.kwarg is not None and args.kwarg.annotation is not None:
            self.visit(args.kwarg.annotation)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        body_attrs: tuple[str, ...],
    ) -> None:
        if not node.generators:
            for attr in body_attrs:
                self.visit(getattr(node, attr))
            return
        first = node.generators[0]
        self.visit(first.iter)
        self.depth += 1
        self.visit(first.target)
        for cond in first.ifs:
            self.visit(cond)
        for gen in node.generators[1:]:
            self.visit(gen)
        for attr in body_attrs:
            self.visit(getattr(node, attr))
        self.depth -= 1

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        # Unbound reserved dunders (`__annotations__`, `__name__`,
        # `__file__`, `__package__`, `__spec__`, `__loader__`,
        # `__cached__`, `__doc__`, `__path__`) resolve to module
        # globals at any scope -- they are free variables when not
        # locally bound -- so a reference inside a function / class /
        # comprehension body is just as dangerous as one at module
        # scope. Refuse all explicit references rather than try to
        # decide at AST time whether the name is locally bound.
        # `__builtins__` is excluded here because rebinding /
        # observing it has its own dedicated verifier with a richer
        # error message.
        if node.id in _RESERVED_MODULE_DUNDERS and node.id != "__builtins__":
            self.module_scope_refs.append(node)


_BUILTINS_GLOBALS_NAMES = ("globals", "vars", "locals", "dir")
# Names that may not be aliased / attribute-accessed — extends the
# globals-family to cover the getattr family and ``__dict__`` so the
# bare-name and attribute-access bans cover every literal-attribute
# / accessor route.
_FORBIDDEN_BUILTIN_NAMES = frozenset(
    ("globals", "vars", "locals", "dir", "getattr", "setattr", "delattr")
)
_FORBIDDEN_ATTRS = frozenset(
    (
        "globals",
        "vars",
        "locals",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "__globals__",
        "__dict__",
    )
)
_BLOCKED_ATTR_LITERALS = frozenset(
    (
        "__globals__",
        "__dict__",
        "globals",
        "vars",
        "locals",
        "dir",
        "getattr",
        "setattr",
        "delattr",
    )
)
_GETATTR_FAMILY = frozenset(("getattr", "setattr", "delattr"))
_DUNDER_GETATTR_FAMILY = frozenset(
    ("__getattribute__", "__setattr__", "__delattr__")
)


def _verify_no_alternative_globals_access(parsed: _ParsedModule) -> None:
    """Refuse alternative routes to the merged module's globals dict
    or namespace that bypass ``_is_module_globals_call``:

    1. ``builtins.globals()`` / ``builtins.vars()`` / ``builtins.locals()``
       / ``builtins.dir()`` -- both via ``import builtins`` and
       ``__import__("builtins")``. The bare-name detector only sees
       ``globals()``; an attribute call routes around it.
    2. ``from builtins import globals as g; g()`` -- the rebound name
       is invisible to the bare-name detector.
    3. ``fn.__globals__`` -- every function carries a reference to its
       defining module's globals dict. After inlining, that's the
       merged dict; a dep can mutate any reserved dunder through
       ``(lambda: None).__globals__["..."]`` without touching
       ``globals()`` at all.
    4. Bare ``dir()`` -- returns the merged module's full namespace
       at module scope, exposing every other dep's identifiers.
    5. Aliasing the builtin name itself (``G = globals; G()``,
       ``snapshot = dir``). Any non-call use of ``globals`` /
       ``vars`` / ``locals`` / ``dir`` as a ``Name(Load)`` is refused.
    6. ``getattr(fn, "__globals__")`` / ``fn.__getattribute__(
       "__globals__")`` / ``object.__getattribute__(fn, "__globals__")``
       and the ``setattr`` / ``delattr`` mirrors -- literal-string
       attribute names that match the blocked set are refused.
    7. ``builtins.__dict__["globals"]()`` / ``vars(builtins)["globals"]
       ()`` -- ``__dict__[literal]`` and ``vars(obj)[literal]`` lookups
       for blocked names are refused.

    Refusing alias forms is conservative; a dep that genuinely needs
    ``builtins.X`` can use the bare builtin name instead.
    """

    visitor = _AlternativeGlobalsVisitor()
    visitor.visit(parsed.tree)

    for node in visitor.module_scope_dir_calls:
        raise BuildError(
            f"{parsed.path}:{node.lineno}: bare ``dir()`` at module "
            f"scope is not allowed. After inlining, ``dir()`` returns "
            f"the merged module's namespace -- every dep's "
            f"identifiers, not just this source's. Pass an explicit "
            f"argument (``dir(some_object)``) if you need it."
        )

    for node, attr in visitor.attribute_refs:
        if attr == "__globals__":
            raise BuildError(
                f"{parsed.path}:{node.lineno}: reading "
                f"``.__globals__`` is not allowed. After inlining, "
                f"every function's ``__globals__`` is the merged "
                f"target module's dict, so this would expose / "
                f"mutate state shared with every other dep."
            )
        if attr == "__dict__":
            raise BuildError(
                f"{parsed.path}:{node.lineno}: reading ``.__dict__`` "
                f"is not allowed. The dict-form of attribute access "
                f"(``X.__dict__[\"globals\"]``) routes around the "
                f"direct attribute-access ban; refusing every "
                f"``.__dict__`` reference closes that route."
            )
        raise BuildError(
            f"{parsed.path}:{node.lineno}: attribute access "
            f"``.{attr}`` is not allowed (matches a refused "
            f"builtin name). After inlining, qualified forms like "
            f"``builtins.{attr}(...)`` route around the static "
            f"globals/vars/locals/dir/getattr/setattr/delattr check. "
            f"Use the bare builtin name instead."
        )

    for node, alias_name in visitor.builtin_imports:
        raise BuildError(
            f"{parsed.path}:{node.lineno}: "
            f"``from builtins import {alias_name}`` is not allowed. "
            f"Use the bare builtin ``{alias_name}()`` instead so "
            f"static analysis can see it."
        )

    for node in visitor.bare_name_aliases:
        raise BuildError(
            f"{parsed.path}:{node.lineno}: using ``{node.id}`` as a "
            f"value (anything other than an immediate call) is not "
            f"allowed. Aliasing the builtin (``G = {node.id}``) "
            f"would let later code call it through a name the static "
            f"checker cannot see. Call ``{node.id}()`` directly at "
            f"each use site instead."
        )

    for record in visitor.getattr_literals:
        if record.literal == "<unpacked>":
            raise BuildError(
                f"{parsed.path}:{record.node.lineno}: "
                f"``{record.func_name}`` with starred / `**` arg "
                f"unpacking is not allowed. The literal-name check "
                f"cannot see through unpacked iterables; use direct "
                f"positional arguments."
            )
        raise BuildError(
            f"{parsed.path}:{record.node.lineno}: "
            f"``{record.func_name}`` with a literal attribute name "
            f"``{record.literal!r}`` is not allowed. This routes "
            f"around the direct attribute-access ban for the same "
            f"set of names."
        )

    for record in visitor.dunder_getattr_literals:
        if record.literal == "<unpacked>":
            raise BuildError(
                f"{parsed.path}:{record.node.lineno}: "
                f"``{record.func_name}`` with starred / `**` arg "
                f"unpacking is not allowed."
            )
        raise BuildError(
            f"{parsed.path}:{record.node.lineno}: "
            f"``{record.func_name}`` with a literal attribute name "
            f"``{record.literal!r}`` is not allowed."
        )

    for node, literal in visitor.dunder_dict_subscripts:
        raise BuildError(
            f"{parsed.path}:{node.lineno}: ``.__dict__[{literal!r}]`` "
            f"and ``vars(obj)[{literal!r}]`` lookups are not allowed; "
            f"they reach the same builtins / function-globals dict "
            f"that the direct attribute ban already covers."
        )


class _GetattrCall(NamedTuple):
    """A call site for ``getattr`` / ``setattr`` / ``delattr`` whose
    second argument is a string literal in the blocked set.
    """

    node: ast.Call
    literal: str
    func_name: str


class _AlternativeGlobalsVisitor(ast.NodeVisitor):
    """Walks a module's AST and records every form that could obtain
    or alias the merged module's globals dict / namespace beyond the
    bare-name ``globals()`` / ``vars()`` / ``locals()`` / ``dir()``
    call shape covered by ``_is_module_globals_call``.

    Keeps a parent stack so call-position vs. value-position uses of
    the builtin names can be distinguished. Header / body scope split
    mirrors ``_GlobalsMutationVisitor``: defaults, decorators,
    annotations, and class bases are evaluated at the enclosing
    scope; only function bodies bump the depth.
    """

    def __init__(self) -> None:
        self._stack: list[ast.AST] = []
        self.depth = 0
        self.module_scope_dir_calls: list[ast.Call] = []
        self.attribute_refs: list[tuple[ast.Attribute, str]] = []
        self.builtin_imports: list[tuple[ast.ImportFrom, str]] = []
        self.bare_name_aliases: list[ast.Name] = []
        self.getattr_literals: list[_GetattrCall] = []
        self.dunder_getattr_literals: list[_GetattrCall] = []
        self.dunder_dict_subscripts: list[tuple[ast.Subscript, str]] = []

    def visit(self, node: ast.AST) -> None:
        self._stack.append(node)
        try:
            super().visit(node)
        finally:
            self._stack.pop()

    def _parent(self) -> ast.AST | None:
        return self._stack[-2] if len(self._stack) >= 2 else None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function_like_header(node)
        self.depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= 1

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._visit_function_like_header(node)
        self.depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= 1

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._visit_arguments_header(node.args)
        self.depth += 1
        self.visit(node.body)
        self.depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        for d in node.decorator_list:
            self.visit(d)
        for b in node.bases:
            self.visit(b)
        for kw in node.keywords:
            self.visit(kw)
        for tp in getattr(node, "type_params", []) or []:
            self.visit(tp)
        # Class body is a real scope for ``dir()`` (it returns the
        # class namespace), but for the *other* checks (attribute
        # access, getattr literal, builtin-name aliasing) class body
        # is no different from module scope: those forms are refused
        # everywhere, so depth doesn't matter. Bump depth so
        # ``module_scope_dir_calls`` only collects depth==0 sites.
        self.depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._visit_comprehension(node, ("elt",))

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        self._visit_comprehension(node, ("elt",))

    def visit_GeneratorExp(  # noqa: N802
        self, node: ast.GeneratorExp
    ) -> None:
        self._visit_comprehension(node, ("elt",))

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        self._visit_comprehension(node, ("key", "value"))

    def _visit_function_like_header(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for d in node.decorator_list:
            self.visit(d)
        self._visit_arguments_header(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for tp in getattr(node, "type_params", []) or []:
            self.visit(tp)

    def _visit_arguments_header(self, args: ast.arguments) -> None:
        for d in args.defaults:
            self.visit(d)
        for d in args.kw_defaults:
            if d is not None:
                self.visit(d)
        for arg_list in (
            getattr(args, "posonlyargs", []) or [],
            args.args,
            args.kwonlyargs,
        ):
            for a in arg_list:
                if a.annotation is not None:
                    self.visit(a.annotation)
        if args.vararg is not None and args.vararg.annotation is not None:
            self.visit(args.vararg.annotation)
        if args.kwarg is not None and args.kwarg.annotation is not None:
            self.visit(args.kwarg.annotation)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        body_attrs: tuple[str, ...],
    ) -> None:
        # Outermost iterable evaluates in the enclosing scope; the
        # rest evaluates inside an implicit function frame.
        if not node.generators:
            for attr in body_attrs:
                self.visit(getattr(node, attr))
            return
        first = node.generators[0]
        self.visit(first.iter)
        self.depth += 1
        self.visit(first.target)
        for cond in first.ifs:
            self.visit(cond)
        for gen in node.generators[1:]:
            self.visit(gen)
        for attr in body_attrs:
            self.visit(getattr(node, attr))
        self.depth -= 1

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module == "builtins":
            for alias in node.names:
                if alias.name in _FORBIDDEN_BUILTIN_NAMES:
                    self.builtin_imports.append((node, alias.name))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # Refused attribute names cover the globals family
        # (``globals``/``vars``/``locals``/``dir``), the getattr
        # family (``getattr``/``setattr``/``delattr``), and the two
        # dict-carrier dunders (``__globals__`` and ``__dict__``).
        if node.attr in _FORBIDDEN_ATTRS:
            self.attribute_refs.append((node, node.attr))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        # Builtin-name aliasing: ``globals``/``vars``/``locals``/
        # ``dir``/``getattr``/``setattr``/``delattr`` referenced as a
        # value. The only legitimate shape is the immediate Call
        # ``Name(id="globals")()`` with no args (or for the getattr
        # family, with no Starred / kwargs); any other position lets
        # the dep capture the builtin and call it through a name the
        # static checker cannot follow.
        if (
            node.id in _FORBIDDEN_BUILTIN_NAMES
            and isinstance(node.ctx, ast.Load)
            and not self._is_legitimate_call_func(node)
        ):
            self.bare_name_aliases.append(node)
        self.generic_visit(node)

    def _is_legitimate_call_func(self, name_node: ast.Name) -> bool:
        """A bare-name reference is legitimate only when it is the
        ``func`` of a ``Call`` whose argument list does NOT use
        starred / **kw unpacking. Splat unpacking would let the
        argument shape escape the per-call literal/empty-args check
        (``globals(*())`` is syntactically a call with one Starred
        arg, but evaluates to ``globals()``; the literal-arg analyzer
        wouldn't see the empty effect either way).
        """
        parent = self._parent()
        if not isinstance(parent, ast.Call) or parent.func is not name_node:
            return False
        for arg in parent.args:
            if isinstance(arg, ast.Starred):
                return False
        for kw in parent.keywords:
            if kw.arg is None:
                return False
        return True

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Bare ``dir()`` at module scope.
        if (
            self.depth == 0
            and isinstance(node.func, ast.Name)
            and node.func.id == "dir"
            and not node.args
            and not node.keywords
        ):
            self.module_scope_dir_calls.append(node)

        # ``getattr(obj, "__globals__")`` / ``setattr`` / ``delattr``.
        # Refuse outright if any argument is starred / **kw unpacked
        # (the literal could hide inside the unpacked iterable, and
        # we choose not to constant-fold tuples), then check
        # positional literals.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _GETATTR_FAMILY
        ):
            if any(isinstance(a, ast.Starred) for a in node.args) or any(
                kw.arg is None for kw in node.keywords
            ):
                self.getattr_literals.append(
                    _GetattrCall(node, "<unpacked>", node.func.id)
                )
            elif len(node.args) >= 2:
                literal = _string_constant(node.args[1])
                if literal is not None and literal in _BLOCKED_ATTR_LITERALS:
                    self.getattr_literals.append(
                        _GetattrCall(node, literal, node.func.id)
                    )

        # ``obj.__getattribute__("__globals__")`` / ``__setattr__`` /
        # ``__delattr__`` -- callee is an Attribute whose ``attr`` is
        # one of the dunders. Bound method form puts the literal at
        # args[0]; unbound form ``object.__getattribute__(obj, "X")``
        # puts the literal at args[1]. Refuse splatted forms outright.
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _DUNDER_GETATTR_FAMILY
        ):
            if any(isinstance(a, ast.Starred) for a in node.args) or any(
                kw.arg is None for kw in node.keywords
            ):
                self.dunder_getattr_literals.append(
                    _GetattrCall(node, "<unpacked>", node.func.attr)
                )
            elif node.args:
                for candidate in node.args[:2]:
                    literal = _string_constant(candidate)
                    if literal is not None and literal in _BLOCKED_ATTR_LITERALS:
                        self.dunder_getattr_literals.append(
                            _GetattrCall(node, literal, node.func.attr)
                        )
                        break

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        # ``obj.__dict__[literal]`` and ``vars(obj)[literal]`` lookups
        # for blocked names. Both let the dep reach builtins or a
        # function's globals dict by name. The aliased form
        # ``D = builtins.__dict__; D[literal]`` is caught by the
        # ``__dict__`` Attribute ban above (``D = ...`` already
        # tripped on the right-hand side), so this only needs to see
        # direct shapes.
        literal = _string_constant(node.slice)
        if literal is not None and literal in _BLOCKED_ATTR_LITERALS:
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "__dict__":
                self.dunder_dict_subscripts.append((node, literal))
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "vars"
            ):
                self.dunder_dict_subscripts.append((node, literal))
        self.generic_visit(node)


def _verify_no_unbounded_globals_access(parsed: _ParsedModule) -> None:
    """Refuse modules that reference ``globals()`` / ``vars()`` /
    ``locals()`` in any syntactic context whose effect on the merged
    module's globals dict cannot be statically reasoned about.

    The per-shape collectors and verifiers (``subscript_writes``,
    ``subscript_reads``, ``method_calls`` -> ``_verify_no_unknown_globals_*``)
    only catch direct uses: ``globals()[K]``, ``globals().METHOD(...)``,
    ``dict.METHOD(globals(), ...)``, ``K in globals()``. A dep that
    instead aliases the dict (``g = globals(); g['__builtins__'] = X``)
    or saves a bound method (``setitem = globals().__setitem__;
    setitem('__builtins__', X)``) routes mutations through a name we
    cannot follow, bypassing every later check. Likewise,
    ``dict(**globals())``, ``list(globals())``, ``for k in globals():``,
    ``globals() | other``, ``globals() == other`` observe the merged
    dict in ways whose result silently changes when other inlined
    sources add or remove keys.

    Rather than try to follow aliases (which would require a real
    escape analysis pass), we refuse outright every use of
    ``globals()`` that does not appear in one of these forms:

    * ``globals()[K]`` (Subscript)
    * ``globals().METHOD(...)`` (immediately invoked, not stored)
    * ``dict.METHOD(globals(), ...)`` (dict-descriptor form)
    * ``X in globals()`` / ``X not in globals()`` (Compare with In /
      NotIn)

    Scope-aware: ``globals()`` matches at any scope, ``vars()`` /
    ``locals()`` only at module scope. Inside a function / class /
    lambda / comprehension body those latter two return the local
    frame and their misuse cannot corrupt merged module globals, so
    they are silently ignored.
    """

    visitor = _GlobalsMutationVisitor()
    visitor.visit(parsed.tree)
    for call in visitor.unbounded_calls:
        # ``call.func`` is a Name node here (we filtered on
        # ``_is_module_globals_call`` which requires a bare Name call).
        name = call.func.id  # type: ignore[attr-defined]
        raise BuildError(
            f"{parsed.path}:{call.lineno}: `{name}()` is used here in "
            f"a context the inliner cannot statically reason about. "
            f"After inlining, every source shares one module namespace, "
            f"so any reference to `{name}()` that does not directly "
            f"index it (`{name}()[K]`), call a known method on it "
            f"(`{name}().METHOD(...)`), pass it to a `dict.METHOD` "
            f"descriptor, or test membership "
            f"(`K in {name}()`) could silently mutate or observe the "
            f"merged globals dict in a way the inliner cannot detect. "
            f"Rewrite to use those forms directly with literal keys, or "
            f"avoid `{name}()` entirely (e.g. read the name itself, "
            f"`name`, instead of `{name}()[\"name\"]`)."
        )


# Allowlist of dict methods we know how to statically reason about on
# the merged module's globals dict. Anything else (``__repr__``,
# ``__or__``, ``fromkeys``, ``__ior__``, ``pop``, ``clear``,
# ``setdefault``, ``__init__``, ``copy``, ``keys``, ``items``,
# ``values``, ``__iter__``, ``__len__``, ...) either replaces, observes
# whole-dict shape, or otherwise lacks a literal-key form -- all silent
# divergence risks after inlining. We deny by default.
_PERMITTED_GLOBALS_METHODS = frozenset(
    {
        # literal-key reads (ingested as cross-module references via
        # ``_collect_dynamic_globals_reads``)
        "__getitem__",
        "get",
        "__contains__",
        # literal-key writes / deletes (ingested as definitions /
        # destructive writes)
        "__setitem__",
        "__delitem__",
        # literal-keyed bulk write -- ``update({...})``; runtime-keyed
        # forms are refused below.
        "update",
    }
)
# Subset of permitted methods whose first positional argument must be a
# string-constant key for static reasoning to apply.
_LITERAL_FIRST_ARG_METHODS = frozenset(
    {"__getitem__", "get", "__contains__", "__setitem__", "__delitem__"}
)


def _verify_no_unknown_globals_mutation(parsed: _ParsedModule) -> None:
    """Refuse modules that read or mutate the merged module's globals
    through forms whose effect cannot be reduced to a known set of
    literal string names.

    Two things are checked here:

    1. **Subscript writes / deletes with non-literal keys** --
       ``globals()[K] = X`` and ``del globals()[K]`` (in any
       assignment-target shape: Assign, AugAssign, AnnAssign with
       value, ``for`` / ``with`` / comprehension targets, nested
       Tuple/List/Starred unpacking) are refused unless ``K`` is a
       string constant. Literal-key forms feed
       ``_collect_dynamic_globals_writes`` /
       ``_collect_dynamic_globals_deletes`` and are checked against
       cross-source collisions. Subscript reads are handled in
       ``_verify_no_unknown_globals_reads``.

    2. **Method calls outside the allowlist** --
       ``globals().METHOD(...)`` (and the equivalent
       ``dict.METHOD(globals(), ...)`` descriptor form, normalized in
       ``_GlobalsMutationVisitor``) is refused unless ``METHOD`` is in
       ``_PERMITTED_GLOBALS_METHODS``. For methods in the allowlist,
       per-method validation enforces literal-key constraints
       (``__setitem__`` / ``__delitem__`` / ``__getitem__`` / ``get`` /
       ``__contains__`` need a literal first argument, ``update`` needs
       a literal-keyed dict literal).

       Allowlist-by-default catches ``globals().__repr__()``,
       ``globals().__or__(...)``, ``dict.fromkeys(globals())``,
       ``globals().pop(...)``, ``.clear()``, ``.setdefault(...)``,
       ``.__init__(...)``, ``.__ior__(...)``, ``.keys()``, ``.items()``,
       ``.values()``, ``.copy()``, ``.__iter__()``, ``.__len__()`` --
       each silently changes behavior post-inline when other inlined
       sources add or remove keys.

    Scope-aware: ``globals()`` always returns the *module's* globals
    regardless of call site, so it is policed everywhere. ``vars()``
    and ``locals()`` only return the module's globals when called at
    module scope -- inside a function / async function / class /
    lambda / comprehension they return the local frame's mapping (or
    the class namespace), so mutations there cannot corrupt merged
    module globals. We use a scope-aware visitor so that legitimate
    function-local ``vars()`` / ``locals()`` writes are not falsely
    refused.

    Out of scope (documented in ``_verify_no_builtins_rebinding``):
    indirect aliases (``g = globals(); g[k] = X`` -- already refused
    by ``_verify_no_unbounded_globals_access``), ``sys.modules`` writes,
    ``setattr`` on the current module, ``exec`` / ``compile`` string
    contents, frame-introspection.
    """

    visitor = _GlobalsMutationVisitor()
    visitor.visit(parsed.tree)

    for node in visitor.subscript_writes:
        if _string_constant(node.slice) is None:
            raise BuildError(
                f"{parsed.path}:{node.lineno}: writing to "
                f"`globals()[<non-literal>]` (or `vars()[...]`) is "
                f"not allowed. The inliner can only check for "
                f"collisions with other inlined sources when the key "
                f"is a string constant. Use a literal key, or rewrite "
                f"as an explicit `name = ...` statement."
            )

    for node in visitor.subscript_deletes:
        if _string_constant(node.slice) is None:
            raise BuildError(
                f"{parsed.path}:{node.lineno}: deleting "
                f"`globals()[<non-literal>]` (or `vars()[...]`) is "
                f"not allowed. The inliner can only check for "
                f"collisions with other inlined sources when the key "
                f"is a string constant. Use a literal key, or rewrite "
                f"as an explicit `del name` statement."
            )

    for entry in visitor.method_calls:
        attr = entry.attr
        call = entry.call
        if attr not in _PERMITTED_GLOBALS_METHODS:
            raise BuildError(
                f"{parsed.path}:{call.lineno}: `globals().{attr}(...)` "
                f"(or the equivalent `vars()`/`locals()`/"
                f"`dict.{attr}(globals(), ...)` form) is not allowed. "
                f"After inlining, every source shares one module's "
                f"globals dict, and `{attr}` either observes the whole "
                f"key set, mutates / replaces unknown keys, or lacks a "
                f"static literal-key form -- each silently changes "
                f"behavior when other inlined sources add or remove "
                f"bindings. Only these methods are permitted: "
                f"{sorted(_PERMITTED_GLOBALS_METHODS)}. Rewrite using "
                f"one of those, or read / write the names directly."
            )
        if attr == "update":
            if not _update_normalized_has_extractable_literal_keys(entry):
                raise BuildError(
                    f"{parsed.path}:{call.lineno}: "
                    f"`globals().update(...)` (or the equivalent "
                    f"`vars()`/`locals()`/`dict.update(globals(), ...)` "
                    f"form) with a runtime-determined key set is not "
                    f"allowed. The inliner cannot statically determine "
                    f"which module globals would be written, so it "
                    f"cannot check for collisions with other inlined "
                    f"sources. Use a dict literal whose keys are all "
                    f"string constants, or rewrite as explicit "
                    f"assignments."
                )
        if attr in _LITERAL_FIRST_ARG_METHODS:
            if not entry.args or _string_constant(entry.args[0]) is None:
                raise BuildError(
                    f"{parsed.path}:{call.lineno}: "
                    f"`globals().{attr}(<non-literal>, ...)` (or the "
                    f"equivalent `dict.{attr}(globals(), <non-literal>, "
                    f"...)`) is not allowed. The inliner can only check "
                    f"for collisions when the key is a string constant. "
                    f"Pass a literal key, or rewrite as an direct "
                    f"name reference / assignment."
                )


def _is_module_globals_call(node: ast.expr) -> bool:
    """True if ``node`` is the no-argument call ``globals()``,
    ``vars()``, or ``locals()`` -- the three CPython builtins that can
    return the merged module's globals dict.

    * ``globals()`` -- always returns the module's globals at any scope.
    * ``vars()`` -- returns the module's globals when called at module
      scope; inside a function/class/lambda/comprehension it returns
      the local frame's mapping (or the class namespace), which is NOT
      the merged module's namespace.
    * ``locals()`` -- same scope rule as ``vars()``: at module scope
      Python documents ``locals() is globals()``; inside a function /
      lambda / class / comprehension it returns the frame's local
      mapping.

    Scope filtering for ``vars()`` and ``locals()`` happens in
    ``_GlobalsMutationVisitor._refers_to_module_globals`` based on the
    visitor's current ``depth``. This helper just identifies the call
    shape; it does not look at scope.
    """

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"globals", "vars", "locals"}
        and not node.args
        and not node.keywords
    )


def _string_constant(node: ast.expr) -> str | None:
    """Return ``node``'s string value if it is a syntactic string
    constant, else ``None``. Folds constant-only f-strings
    (``ast.JoinedStr`` whose every value is a string ``Constant``)
    so that ``f"globals"`` and ``f"__globals__"`` are recognized as
    the same literal as their plain-string equivalents.
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    return None


def _update_normalized_has_extractable_literal_keys(
    entry: "_ModuleGlobalsMethodCall",
) -> bool:
    """True if an ``update(...)`` call (in either bound-method or
    dict-descriptor form) has keys that can all be statically
    enumerated as string literals.

    The normalized ``args`` here is what comes *after* the receiver, so
    for ``globals().update({"k": v})`` it is ``[Dict({"k": v})]`` and
    for ``dict.update(globals(), {"k": v})`` it is also
    ``[Dict({"k": v})]``. Accepts exactly one positional ``Dict``
    argument with no ``**kwargs`` and no other args, where every key
    is a string constant (no computed keys, no ``**inner`` unpacking).
    Otherwise returns False -- the caller refuses such calls outright.
    """

    if entry.keywords:
        # `update(**kw)` -- key set is runtime-determined.
        return False
    if len(entry.args) != 1:
        return False
    arg = entry.args[0]
    if not isinstance(arg, ast.Dict):
        return False
    for key in arg.keys:
        # `**inner` unpacking yields a None key in ast.Dict.
        if key is None:
            return False
        if _string_constant(key) is None:
            return False
    return True


class _ModuleGlobalsMethodCall(NamedTuple):
    """Normalized representation of a method-style access on the merged
    module's globals dict.

    ``call`` is the original ``ast.Call`` node (kept for ``lineno`` /
    error messages and ``ctx`` checks).
    ``attr`` is the method name -- e.g. ``"__setitem__"``, ``"update"``,
    ``"get"``, ``"keys"``.
    ``args`` is the positional argument list *after* the receiver: for a
    bound-method form ``globals().METHOD(a, b)`` it is ``[a, b]``; for a
    descriptor-form ``dict.METHOD(globals(), a, b)`` it is also
    ``[a, b]``. This makes the rest of the pipeline shape-agnostic.
    ``keywords`` is forwarded for completeness (e.g. ``update(**kw)``).
    """

    call: ast.AST
    attr: str
    args: list[ast.expr]
    keywords: list[ast.keyword]


class _GlobalsMutationVisitor(ast.NodeVisitor):
    """Walks the AST scope-aware, collecting every ``globals()`` /
    ``vars()`` / ``locals()`` reference whose use mutates *or reads*
    the merged module's globals dict.

    Buckets:

    * ``subscript_writes`` -- Subscript nodes with ``ctx == Store`` whose
      ``value`` is a module-globals call. This covers every assignment
      target form: plain ``Assign`` (``globals()[K] = X``), AugAssign
      (``+=`` on a globals subscript), AnnAssign with a value
      (``globals()[K]: T = X``), iteration targets (``for
      globals()[K] in items``), context-manager targets (``with cm()
      as globals()[K]``), comprehension targets, and tuple / list
      unpacking targets (``globals()[K], other = (1, 2)``). Python
      gives the Subscript ``ctx == Store`` for all of these, so
      visiting ``ast.Subscript`` once catches them all.

    * ``subscript_deletes`` -- same, but ``ctx == Del`` (``del
      globals()[K]`` and the equivalent inside compound delete targets
      like ``del (globals()[K],)``).

    * ``subscript_reads`` -- same, but ``ctx == Load``
      (``globals()[K]``, ``return globals()[K]``, ``f(globals()[K])``,
      etc.). After inlining, a literal-key read silently observes
      bindings introduced by other inlined sources, so the collision
      detector ingests these as cross-module references.

    * ``method_calls`` -- a list of ``_ModuleGlobalsMethodCall`` records.
      Both bound-method form (``globals().METHOD(...)``) and the
      equivalent dict-descriptor form (``dict.METHOD(globals(), ...)``)
      land here normalized to the same shape, so downstream consumers
      (``_verify_no_unknown_globals_*`` and ``_collect_dynamic_globals_*``)
      do not need to know which surface form was used. ``Compare``
      nodes with ``In`` / ``NotIn`` against a module-globals call
      (``"foo" in globals()``) are also normalized into a synthetic
      ``__contains__`` entry so the same downstream logic catches them.

    Annotation-only AnnAssign (``globals()[K]: T`` without a value)
    parses with ``ctx == Store`` on the Subscript but does not actually
    execute a store at runtime. We record the target Subscript's
    object id so the visitor skips it when reached.

    Scope handling:

    ``globals()`` always returns the calling module's globals dict, so
    it matches at every scope. ``vars()`` and ``locals()`` only return
    the module's globals when called at module scope; inside a function
    / async function / class / lambda / comprehension they return the
    local frame's mapping (or the class namespace), which is NOT the
    merged module's namespace. We track ``depth`` and skip ``vars()`` /
    ``locals()`` matches when ``depth > 0``.

    Each scope-creating node is split into a *header* (evaluated when
    the surrounding statement runs, in the enclosing scope) and a
    *body* (evaluated later, in the new local scope). For a
    ``FunctionDef``, the header is the decorators, default values,
    argument annotations, return annotation, and PEP 695 type
    parameters; the body is the statement list. For comprehensions,
    the header is the *outermost* iterable and the body is everything
    else (target, ``if`` clauses, later generators, element
    expression). This matches Python's actual evaluation order, so
    ``def f(x=vars().__setitem__(...)): ...`` is treated as a
    module-scope ``vars()`` call.
    """

    def __init__(self) -> None:
        self.depth = 0
        self.subscript_writes: list[ast.Subscript] = []
        self.subscript_deletes: list[ast.Subscript] = []
        self.subscript_reads: list[ast.Subscript] = []
        self.method_calls: list[_ModuleGlobalsMethodCall] = []
        # Module-globals calls (``globals()`` / ``vars()`` / ``locals()``
        # in module-globals scope) that appear in a syntactic context we
        # cannot statically reason about: aliased to a name (``g =
        # globals()``), passed as a generic call argument
        # (``dict(**globals())``, ``list(globals())``), iterated over
        # (``for k in globals(): ...``), compared with anything other
        # than ``in`` / ``not in``, used in a binop (``globals() | {}``),
        # or stored as a bound method (``setitem = globals().__setitem__``).
        # ``_verify_no_unbounded_globals_access`` refuses each of these.
        self.unbounded_calls: list[ast.Call] = []
        self._skip_subscript_ids: set[int] = set()
        self._parent_stack: list[ast.AST] = []

    def visit(self, node: ast.AST) -> None:
        # Override the dispatch to maintain a parent stack so individual
        # visit_* methods can introspect the syntactic parent (and
        # grandparent) of the node currently being visited. Used by
        # visit_Call to decide whether a ``globals()`` call is in a
        # context we know how to reason about.
        self._parent_stack.append(node)
        try:
            super().visit(node)
        finally:
            self._parent_stack.pop()

    def _parent(self) -> ast.AST | None:
        return self._parent_stack[-2] if len(self._parent_stack) >= 2 else None

    def _grandparent(self) -> ast.AST | None:
        return self._parent_stack[-3] if len(self._parent_stack) >= 3 else None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function_like_header(node)
        self.depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= 1

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._visit_function_like_header(node)
        self.depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= 1

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._visit_arguments_header(node.args)
        self.depth += 1
        self.visit(node.body)
        self.depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        for d in node.decorator_list:
            self.visit(d)
        for b in node.bases:
            self.visit(b)
        for kw in node.keywords:
            self.visit(kw)
        # PEP 695 type parameters (3.12+). Their default expressions
        # (PEP 696, 3.13+) evaluate in the enclosing scope when the
        # class statement runs, so visit them at the current depth.
        for tp in getattr(node, "type_params", []) or []:
            self.visit(tp)
        self.depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._visit_comprehension(node, ("elt",))

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        self._visit_comprehension(node, ("elt",))

    def visit_GeneratorExp(  # noqa: N802
        self, node: ast.GeneratorExp
    ) -> None:
        self._visit_comprehension(node, ("elt",))

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        self._visit_comprehension(node, ("key", "value"))

    def _visit_function_like_header(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for d in node.decorator_list:
            self.visit(d)
        self._visit_arguments_header(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for tp in getattr(node, "type_params", []) or []:
            self.visit(tp)

    def _visit_arguments_header(self, args: ast.arguments) -> None:
        # Default values and the kw-default sentinels are evaluated
        # when the def/lambda statement runs, in the enclosing scope.
        for d in args.defaults:
            self.visit(d)
        for d in args.kw_defaults:
            if d is not None:
                self.visit(d)
        # Argument annotations are likewise evaluated at def time
        # unless ``from __future__ import annotations`` is in effect.
        # The inliner already enforces that all merged sources share
        # the same ``__future__`` set, but we still bias toward
        # treating annotations as enclosing-scope expressions: a
        # mutating ``vars()`` call in an annotation is exotic enough
        # that flagging it (under non-PEP-563 semantics) is safer than
        # missing it.
        for arg_list in (
            getattr(args, "posonlyargs", []) or [],
            args.args,
            args.kwonlyargs,
        ):
            for a in arg_list:
                if a.annotation is not None:
                    self.visit(a.annotation)
        if args.vararg is not None and args.vararg.annotation is not None:
            self.visit(args.vararg.annotation)
        if args.kwarg is not None and args.kwarg.annotation is not None:
            self.visit(args.kwarg.annotation)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        body_attrs: tuple[str, ...],
    ) -> None:
        # The *outermost* generator's iterable is evaluated in the
        # enclosing scope; everything else (its target / ``if`` clauses,
        # later generators, the element expression(s)) lives in the
        # comprehension's own implicit-function scope.
        if not node.generators:
            for attr in body_attrs:
                self.visit(getattr(node, attr))
            return
        first = node.generators[0]
        self.visit(first.iter)
        self.depth += 1
        self.visit(first.target)
        for cond in first.ifs:
            self.visit(cond)
        for gen in node.generators[1:]:
            self.visit(gen)
        for attr in body_attrs:
            self.visit(getattr(node, attr))
        self.depth -= 1

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is None and isinstance(node.target, ast.Subscript):
            # `globals()[K]: T` parses with the target Subscript's ctx as
            # Store but executes only the annotation; no runtime store
            # happens. Mark the target so visit_Subscript ignores it.
            self._skip_subscript_ids.add(id(node.target))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        if id(node) in self._skip_subscript_ids:
            self.generic_visit(node)
            return
        if self._refers_to_module_globals(node.value):
            if isinstance(node.ctx, ast.Store):
                self.subscript_writes.append(node)
            elif isinstance(node.ctx, ast.Del):
                self.subscript_deletes.append(node)
            elif isinstance(node.ctx, ast.Load):
                self.subscript_reads.append(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Bound-method form: ``globals().METHOD(*args, **kw)``.
        if isinstance(node.func, ast.Attribute) and self._refers_to_module_globals(
            node.func.value
        ):
            self.method_calls.append(
                _ModuleGlobalsMethodCall(
                    call=node,
                    attr=node.func.attr,
                    args=list(node.args),
                    keywords=list(node.keywords),
                )
            )
        # Descriptor form: ``dict.METHOD(globals(), *args, **kw)``. The
        # first positional argument is the receiver; everything after is
        # what would have been passed to a bound-method call. Other
        # ``some_obj.METHOD(globals(), ...)`` forms are not normalized
        # here because we cannot statically tell whether ``some_obj`` is
        # ``dict`` (a name lookup is required); the dict-descriptor form
        # uses the literal ``dict`` Name, which is unambiguous.
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "dict"
            and node.args
            and self._refers_to_module_globals(node.args[0])
        ):
            self.method_calls.append(
                _ModuleGlobalsMethodCall(
                    call=node,
                    attr=node.func.attr,
                    args=list(node.args[1:]),
                    keywords=list(node.keywords),
                )
            )
        # When the call ITSELF is a module-globals reference, check
        # whether its syntactic context is one we can statically reason
        # about. Anything else (alias assignment, generic call argument,
        # iteration target, BinOp, Compare with non-In op, etc.) is
        # recorded as an "unbounded" use and refused by
        # ``_verify_no_unbounded_globals_access``. Without this check, a
        # dep can save the bound method (``setitem =
        # globals().__setitem__``) or the dict itself (``g =
        # globals()``) into a name and mutate the merged globals through
        # the alias, which the per-shape mutation/read collectors cannot
        # follow.
        if self._refers_to_module_globals(node) and not self._is_in_permitted_context(
            node
        ):
            self.unbounded_calls.append(node)
        self.generic_visit(node)

    def _is_in_permitted_context(self, call: ast.Call) -> bool:
        """True if this ``globals()/vars()/locals()`` call appears in a
        syntactic context whose effect on merged module globals can be
        resolved by the existing per-shape collectors and verifiers.

        Permitted contexts:

        * ``Subscript.value`` -- ``globals()[K]`` (Store / Del / Load).
          The collision detector and read-side ingestion handle these
          via ``subscript_writes`` / ``subscript_deletes`` /
          ``subscript_reads``.

        * ``Attribute.value`` *and* the resulting Attribute is the
          ``func`` of a Call -- ``globals().METHOD(...)``. The
          ``_verify_no_unknown_globals_*`` functions filter on
          ``METHOD``. Storing the bound method (``setitem =
          globals().__setitem__``) without immediately calling it does
          NOT count: the saved attribute can later be invoked behind
          our back, so it is rejected here.

        * First positional argument of ``dict.METHOD(globals(), ...)``
          -- the dict-descriptor form, normalized into ``method_calls``.

        * Comparator of an ``ast.Compare`` paired with ``In`` /
          ``NotIn`` -- ``X in globals()`` / ``X not in globals()``,
          synthesized into a ``__contains__`` ``method_call`` entry.
        """

        parent = self._parent()
        if parent is None:
            return False
        # globals()[K]
        if isinstance(parent, ast.Subscript) and parent.value is call:
            return True
        # globals().METHOD(...) -- the Attribute must be the func of a
        # Call grandparent, so the bound method is invoked immediately
        # rather than escaping into a name.
        if isinstance(parent, ast.Attribute) and parent.value is call:
            grandparent = self._grandparent()
            if (
                isinstance(grandparent, ast.Call)
                and grandparent.func is parent
            ):
                return True
            return False
        # dict.METHOD(globals(), ...)
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Attribute)
            and isinstance(parent.func.value, ast.Name)
            and parent.func.value.id == "dict"
            and parent.args
            and parent.args[0] is call
        ):
            return True
        # X in globals() / X not in globals(). Only permitted when the
        # Compare is a *single* binary comparison, i.e. one op and one
        # comparator, AND the call is that lone comparator. A chained
        # compare like ``"x" in globals() in Probe()`` would syntactically
        # match In/NotIn for the first operand pairing, but Python then
        # threads ``globals()`` itself into the second comparison's
        # ``Probe.__contains__``, leaking the merged dict to a
        # caller-defined operator that observes the whole key set. Same
        # for ``X in globals() == something``. Refusing the chained form
        # is conservative; if a single membership check is what was
        # wanted, the user can write it without chaining.
        if isinstance(parent, ast.Compare):
            if len(parent.ops) == 1 and len(parent.comparators) == 1:
                op = parent.ops[0]
                comp = parent.comparators[0]
                if comp is call and isinstance(op, (ast.In, ast.NotIn)):
                    return True
            return False
        return False

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
        # ``X in globals()`` / ``X not in globals()`` is a
        # ``__contains__`` observation on the merged module's globals
        # dict. We synthesize a ``_ModuleGlobalsMethodCall`` so the
        # rest of the pipeline treats it identically to
        # ``globals().__contains__(X)``.
        #
        # Only the SIMPLE binary form (one op, one comparator) is
        # synthesized here. Chained Compares (``"x" in globals() in
        # Probe()`` etc.) are refused outright by
        # ``_is_in_permitted_context`` -- the inner ``globals()`` call
        # in a chain leaks the dict to the next operand's comparator,
        # so we never want to treat them as benign membership checks.
        if (
            len(node.ops) == 1
            and len(node.comparators) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and self._refers_to_module_globals(node.comparators[0])
        ):
            self.method_calls.append(
                _ModuleGlobalsMethodCall(
                    call=node,
                    attr="__contains__",
                    args=[node.left],
                    keywords=[],
                )
            )
        self.generic_visit(node)

    def _refers_to_module_globals(self, node: ast.expr) -> bool:
        if not _is_module_globals_call(node):
            return False
        # ``vars()`` and ``locals()`` inside a function / class / lambda /
        # comprehension body return the local namespace, not module
        # globals -- skip them so legitimate function-local uses aren't
        # refused. ``globals()`` always returns module globals, so it is
        # matched at every scope.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"vars", "locals"}
            and self.depth > 0
        ):
            return False
        return True


def _collect_dynamic_globals_writes(tree: ast.Module) -> set[str]:
    """Names written to module globals via literal-key
    ``globals()`` / ``vars()`` mutations, anywhere in the AST.

    Captured forms (every assignment-target shape in Python where the
    Subscript's ``ctx`` is ``Store`` -- Assign, AugAssign, AnnAssign
    (with value), iteration targets, context-manager targets,
    comprehension targets, nested Tuple/List/Starred unpacking):

    * ``globals()[literal] = X``, ``... += X``, ``...: T = X``,
      ``for globals()[literal] in items``, ``with cm() as
      globals()[literal]``, ``[... for globals()[literal] in ...]``,
      ``globals()[literal], other = (a, b)``, etc.
    * ``globals().__setitem__(literal, X)``
    * ``globals().update({literal: X, ...})`` (literal-keyed dict;
      ``_verify_no_unknown_globals_mutation`` refuses runtime-keyed
      ``update`` forms).

    Scope: ``globals()`` matches at any scope (it always points at the
    module's globals). ``vars()`` matches only at module scope (inside
    function / class / lambda / comprehension it returns local
    namespaces, not module globals).

    Non-literal-key forms have already been refused by
    ``_verify_no_unknown_globals_mutation``; the defensive
    ``_string_constant(...)`` checks here remain so reordering passes
    can never silently regress to a pre-fix state.

    Does NOT capture: aliased ``g = globals(); g[k] = X``, attribute
    writes via ``sys.modules[__name__]``, ``setattr`` on the current
    module, ``exec`` / ``compile`` string contents.
    """

    visitor = _GlobalsMutationVisitor()
    visitor.visit(tree)

    names: set[str] = set()
    for sub in visitor.subscript_writes:
        key = _string_constant(sub.slice)
        if key is not None:
            names.add(key)
    for entry in visitor.method_calls:
        if entry.attr == "__setitem__" and entry.args:
            key = _string_constant(entry.args[0])
            if key is not None:
                names.add(key)
        elif entry.attr == "update":
            if (
                len(entry.args) == 1
                and not entry.keywords
                and isinstance(entry.args[0], ast.Dict)
            ):
                for k in entry.args[0].keys:
                    if k is None:
                        continue
                    s = _string_constant(k)
                    if s is not None:
                        names.add(s)
    return names


def _collect_dynamic_globals_deletes(tree: ast.Module) -> set[str]:
    """Names removed from module globals via literal-key
    ``globals()`` / ``vars()`` deletions, anywhere in the AST.

    Captured forms:

    * ``del globals()[literal]`` (and the equivalent nested in
      compound delete targets, ``del (globals()[literal],)``).
    * ``globals().__delitem__(literal)``.

    ``globals().pop(...)``, ``.popitem()``, ``.clear()``, etc. are
    refused outright by ``_verify_no_unknown_globals_mutation`` and so
    do not appear here.

    Scope handling and non-literal-key handling match
    ``_collect_dynamic_globals_writes``.
    """

    visitor = _GlobalsMutationVisitor()
    visitor.visit(tree)

    names: set[str] = set()
    for sub in visitor.subscript_deletes:
        key = _string_constant(sub.slice)
        if key is not None:
            names.add(key)
    for entry in visitor.method_calls:
        if entry.attr == "__delitem__" and entry.args:
            key = _string_constant(entry.args[0])
            if key is not None:
                names.add(key)
    return names


# Literal-key read methods on dict that the collision detector ingests
# as cross-module references via ``_collect_dynamic_globals_reads``.
_LITERAL_KEY_READ_METHODS = frozenset({"__getitem__", "get", "__contains__"})


def _verify_no_unknown_globals_reads(parsed: _ParsedModule) -> None:
    """Refuse modules that read the merged module's globals through
    forms whose effect cannot be reduced to a known set of literal
    string names.

    Method-call forms (``globals().__getitem__(...)``,
    ``globals().get(...)``, ``globals().__contains__(...)``,
    ``globals().keys()``, ``.items()``, ``.values()``, ``.copy()``,
    ``.__iter__()``, etc.) are policed by the allowlist in
    ``_verify_no_unknown_globals_mutation`` -- only the literal-key
    read methods (``__getitem__``, ``get``, ``__contains__``) are
    permitted, and only with a string-constant first argument.

    This function handles the *Subscript* read form: ``globals()[K]``
    in a Load context. Non-literal ``K`` is refused because the
    inliner cannot tell which name is being read, so it cannot tell
    whether another inlined source silently satisfies the read.
    Literal-key forms feed ``_collect_dynamic_globals_reads`` and are
    ingested into ``pending_refs`` by the collision detector.

    Scope-aware handling mirrors ``_verify_no_unknown_globals_mutation``:
    ``globals()`` policed everywhere, ``vars()`` / ``locals()`` only at
    module scope.

    Out of scope (documented in ``_verify_no_builtins_rebinding``):
    indirect aliases -- ``_verify_no_unbounded_globals_access`` already
    refuses ``g = globals()``-style aliases.
    """

    visitor = _GlobalsMutationVisitor()
    visitor.visit(parsed.tree)

    for node in visitor.subscript_reads:
        if _string_constant(node.slice) is None:
            raise BuildError(
                f"{parsed.path}:{node.lineno}: reading "
                f"`globals()[<non-literal>]` (or `vars()[...]`/"
                f"`locals()[...]`) is not allowed. The inliner can only "
                f"check for cross-module reference collisions when the "
                f"key is a string constant. Use a literal key, or read "
                f"the name directly (e.g. `name` instead of "
                f"`globals()[\"name\"]`)."
            )


def _collect_dynamic_globals_reads(tree: ast.Module) -> set[str]:
    """Names read from module globals via literal-key
    ``globals()`` / ``vars()`` / ``locals()`` accesses, anywhere in the
    AST.

    Captured forms:

    * ``globals()[literal]`` (and the equivalent inside any expression
      context: ``return globals()[literal]``, ``f(globals()[literal])``,
      etc. -- Python parses them all with ``ctx == Load``).
    * ``globals().__getitem__(literal)``.
    * ``globals().get(literal)`` / ``.get(literal, default)``.
    * ``globals().__contains__(literal)`` (and ``literal in globals()``,
      which the AST lowers to the same call).
    * The ``dict.METHOD(globals(), literal, ...)`` descriptor form of
      each of the above.

    Whole-dict observations (``.keys()``, ``.items()``, etc.) and
    non-literal-key reads have already been refused by
    ``_verify_no_unknown_globals_reads`` and so do not appear here.

    Scope handling and non-literal-key handling match
    ``_collect_dynamic_globals_writes``.
    """

    visitor = _GlobalsMutationVisitor()
    visitor.visit(tree)

    names: set[str] = set()
    for sub in visitor.subscript_reads:
        key = _string_constant(sub.slice)
        if key is not None:
            names.add(key)
    for entry in visitor.method_calls:
        if entry.attr in _LITERAL_KEY_READ_METHODS and entry.args:
            key = _string_constant(entry.args[0])
            if key is not None:
                names.add(key)
    return names


def _first_builtins_store_line(tree: ast.Module) -> int | None:
    """Best-effort lineno of the first AST node that writes ``__builtins__``.

    Used only to enrich the ``_verify_no_builtins_rebinding`` error
    message -- the membership check there is authoritative. Returns
    ``None`` when no write site is found in the AST (e.g., binding
    happened in a way the walk cannot represent with a lineno, such as
    bare ``ast.alias`` nodes on Python versions where they lack one).
    """

    earliest: int | None = None

    def update(lineno: int | None) -> None:
        nonlocal earliest
        if lineno is None:
            return
        if earliest is None or lineno < earliest:
            earliest = lineno

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                update(node.lineno)
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if node.name == "__builtins__":
                update(node.lineno)
        elif isinstance(node, ast.ExceptHandler):
            if node.name == "__builtins__":
                update(node.lineno)
        elif isinstance(node, ast.alias):
            if (node.asname or node.name) == "__builtins__":
                update(getattr(node, "lineno", None))
        elif isinstance(node, ast.Global):
            if "__builtins__" in node.names:
                update(node.lineno)
        elif _ast_has_type_alias() and isinstance(node, ast.TypeAlias):
            if (
                isinstance(node.name, ast.Name)
                and node.name.id == "__builtins__"
            ):
                update(node.lineno)
    return earliest


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
        _verify_no_inline_semicolons(parsed)
        _verify_no_relative_imports(parsed)
        _verify_no_star_imports(parsed)
        _verify_no_unbounded_globals_access(parsed)
        _verify_no_unknown_globals_mutation(parsed)
        _verify_no_unknown_globals_reads(parsed)
        _verify_no_annotations_dict_access(parsed)
        _verify_no_alternative_globals_access(parsed)
        _verify_no_builtins_rebinding(parsed)
        _verify_no_module_scope_walrus_or_match(parsed)
        _verify_no_nested_local_imports(parsed, local_import_roots)
        _verify_externals_precede_locals(parsed, local_import_roots)
        externals, locals_ = _classify_top_level_imports(parsed, local_import_roots)
        future_imports = _collect_future_imports(parsed.tree)
        _verify_no_future_annotations(parsed, future_imports)

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

    def visit_TypeAlias(self, node) -> None:  # noqa: N802
        # PEP 695 / Python 3.12+: ``type X = ...`` binds ``X`` at the
        # enclosing scope (module-scope here). The ``TypeAliasType``
        # body executes lazily but the name is bound when the
        # statement runs, exactly like ``X = ...``. Without this hook
        # ``type helper = int`` would not collide with another source's
        # ``def helper(): ...`` and ``type __builtins__ = int`` would
        # not be flagged as a reserved-name rebinding.
        if isinstance(node.name, ast.Name):
            self.names.add(node.name.id)

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

    definitions: dict[str, str] = {}
    definition_paths: dict[str, set[Path]] = defaultdict(set)
    pending_refs: list[tuple[str, str]] = []
    destructive_writes: list[tuple[str, Path, str]] = []

    def _record_def(name: str, source_path: Path, source_label: str) -> None:
        existing = definitions.get(name)
        if existing is not None and existing != source_label:
            raise BuildError(
                f"Top-level name {name!r} is bound by both {existing} and "
                f"{source_label}. Inlining would collapse them into one module "
                f"and silently pick the last binding. Rename one before "
                f"rebuilding."
            )
        definitions[name] = source_label
        definition_paths[name].add(source_path)

    # Phase 1a: names bound by code in shared deps (functions / classes /
    # assignments / ``global`` writes from inside function or class bodies /
    # dynamic ``globals()[literal] = X`` style writes anywhere in the AST).
    for dep in deps_ordered:
        dep_static_redefs = _collect_inlined_module_bindings(
            dep.tree
        ) | _collect_global_writes(dep.tree)
        dep_dynamic_writes = _collect_dynamic_globals_writes(dep.tree)
        dep_redefs = dep_static_redefs | dep_dynamic_writes
        for name in dep_static_redefs:
            _record_def(name, dep.path, f"{dep.path} (definition)")
        for name in dep_dynamic_writes - dep_static_redefs:
            _record_def(
                name,
                dep.path,
                f"{dep.path} (dynamic globals write)",
            )
        # Cross-module references this dep relies on (``return helper()``
        # with no local binding, ``global X; ...`` with no write,
        # ``return globals()["helper"]`` with no local helper, etc.) are
        # deferred to phase 2.
        dep_class_refs = _collect_class_body_unbound_refs(dep.tree)
        dep_dynamic_reads = _collect_dynamic_globals_reads(dep.tree)
        for name in _collect_external_module_refs(
            dep.source, dep.path, dep.tree
        ) | dep_class_refs:
            if name in dep_redefs:
                continue
            pending_refs.append((name, f"{dep.path} (unbound reference)"))
        # Literal-key globals/vars/locals reads ingested as references
        # too: ``globals()["helper"]`` in a dep that does not itself bind
        # ``helper`` would silently resolve to another inlined source's
        # ``helper`` after merging, while in source form it raises
        # ``KeyError``. Names bound by this dep's own top-level externals
        # are excluded -- ``import math; ...; globals()["math"]`` resolves
        # to the same ``math`` before and after inlining, so it is not a
        # cross-module reference.
        dep_external_bound = {
            bound
            for node in dep.external_imports
            for bound, _ in _bound_names_with_source(node)
        }
        for name in dep_dynamic_reads - dep_redefs - dep_external_bound:
            pending_refs.append(
                (name, f"{dep.path} (dynamic globals read)")
            )
        for name in _collect_module_scope_destructive_writes(dep.tree):
            destructive_writes.append(
                (
                    name,
                    dep.path,
                    f"{dep.path} (destructive write at module scope)",
                )
            )
        for name in _collect_dynamic_globals_deletes(dep.tree):
            destructive_writes.append(
                (
                    name,
                    dep.path,
                    f"{dep.path} (dynamic globals delete)",
                )
            )

    # Phase 1b: names bound by code in the target.
    target_locally_bound = {
        alias.name
        for node in target_local_imports
        for alias in node.names
        if alias.name != "*"
    }
    target_static_redefs = _collect_inlined_module_bindings(
        target_module.tree
    ) | _collect_global_writes(target_module.tree)
    target_dynamic_writes = _collect_dynamic_globals_writes(target_module.tree)
    target_redefs = target_static_redefs | target_dynamic_writes
    for name in target_redefs:
        if name in target_locally_bound:
            raise BuildError(
                f"{target_module.path}: top-level name {name!r} is both "
                f"imported from a local shared module and redefined in the "
                f"target body."
            )
    for name in target_static_redefs:
        _record_def(
            name, target_module.path, f"{target_module.path} (definition)"
        )
    for name in target_dynamic_writes - target_static_redefs:
        _record_def(
            name,
            target_module.path,
            f"{target_module.path} (dynamic globals write)",
        )
    target_class_refs = _collect_class_body_unbound_refs(target_module.tree)
    target_dynamic_reads = _collect_dynamic_globals_reads(target_module.tree)
    for name in _collect_external_module_refs(
        target_module.source, target_module.path, target_module.tree
    ) | target_class_refs:
        if name in target_locally_bound or name in target_redefs:
            continue
        pending_refs.append((name, f"{target_module.path} (unbound reference)"))
    # Same dynamic-read ingestion as deps, applied to the target.
    target_external_bound = {
        bound
        for node in target_external_imports
        for bound, _ in _bound_names_with_source(node)
    }
    for name in (
        target_dynamic_reads
        - target_locally_bound
        - target_redefs
        - target_external_bound
    ):
        pending_refs.append(
            (name, f"{target_module.path} (dynamic globals read)")
        )
    for name in _collect_module_scope_destructive_writes(target_module.tree):
        destructive_writes.append(
            (
                name,
                target_module.path,
                f"{target_module.path} (destructive write at module scope)",
            )
        )
    for name in _collect_dynamic_globals_deletes(target_module.tree):
        destructive_writes.append(
            (
                name,
                target_module.path,
                f"{target_module.path} (dynamic globals delete)",
            )
        )

    # Phase 1c: names that the merged-import section will bind. We need
    # to flag the case where, say, a dep defines ``helper`` and the
    # target imports ``from foo import helper`` -- both end up at module
    # scope after inlining and would collide silently.
    #
    # The label is the import's stringified form so two identical imports
    # from different modules dedupe rather than colliding (this matches the
    # later ``_format_merged_imports`` dedupe). Distinct imports binding the
    # same name -- e.g., ``from json import loads`` vs ``from yaml import
    # loads`` -- will have different labels and properly collide.
    #
    # We attribute each bound name to the *originating* source file (the
    # target or the dep that wrote the import), not a synthetic path, so
    # Phase 3's destructive-write check can tell "same source imports and
    # cleans up its own external" (legal) apart from "one source erases
    # another source's import" (silent corruption of merged globals).
    for node in target_external_imports:
        for bound, source_repr in _bound_names_with_source(node):
            _record_def(bound, target_module.path, source_repr)
    for dep in deps_ordered:
        for node in dep.external_imports:
            for bound, source_repr in _bound_names_with_source(node):
                _record_def(bound, dep.path, source_repr)

    # Phase 2: cross-module unbound references vs definitions. A read-only
    # reference is silently satisfied by another source's definition
    # post-inlining; flag that. Two unbound references with no definition
    # are NOT a collision -- both raise ``NameError`` in source and merge.
    for name, ref_label in pending_refs:
        defined_label = definitions.get(name)
        if defined_label is None or defined_label == ref_label:
            continue
        raise BuildError(
            f"Top-level name {name!r} is referenced as a module global by "
            f"{ref_label} but bound by {defined_label}. After inlining, the "
            f"reference would silently resolve to {defined_label} -- in the "
            f"original separated modules it would raise NameError. Define "
            f"{name!r} in the same source or rename one side before rebuilding."
        )

    # Phase 3: destructive module-scope writes vs definitions in *other*
    # sources. ``del X`` and ``except E as X`` at module scope of one
    # inlined source would silently erase ``X`` if another source bound
    # it -- in the original separated modules they don't share a
    # namespace, so the erase either no-ops or NameErrors instead of
    # corrupting another module's binding.
    #
    # A destructive write of ``X`` collides only when *some other* source
    # also binds ``X``. A dep that imports a module and immediately deletes
    # the binding (``import math; ...; del math``) without any other dep
    # using ``math`` is legal: in the merged file the delete only undoes
    # this dep's own contribution, mirroring source semantics. We compare
    # ``definition_paths[name]`` (the set of every source that binds
    # ``name``) against the destructive writer's path and only flag when
    # the remaining set is non-empty.
    for name, dest_path, dest_label in destructive_writes:
        binding_paths = definition_paths.get(name)
        if not binding_paths:
            continue
        other_sources = binding_paths - {dest_path}
        if not other_sources:
            continue
        defined_label = definitions[name]
        raise BuildError(
            f"Top-level name {name!r} is bound by {defined_label} but "
            f"destructively erased at module scope by {dest_label}. After "
            f"inlining, the erase would silently destroy the binding from "
            f"{defined_label} -- in the original separated modules those "
            f"sources don't share a namespace, so {dest_label} would never "
            f"see {defined_label}'s binding. Refactor so the destructive "
            f"write is scoped to its own module-local name or move it inside "
            f"a function that owns the binding."
        )


def _collect_assign_names(target: ast.expr, names: set[str]) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_assign_names(elt, names)
    elif isinstance(target, ast.Starred):
        # ``*helper, = [...]`` and ``a, *helper, b = [...]`` bind ``helper``
        # at the wrapped target. ``ast.Starred.value`` may itself be a
        # tuple/list (rare nested unpacking), so recurse.
        _collect_assign_names(target.value, names)


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
