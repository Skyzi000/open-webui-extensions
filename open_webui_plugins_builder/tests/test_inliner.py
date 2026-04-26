"""Inliner behavior tests.

These tests construct fake source trees on disk under ``tmp_path`` and exercise
``build_target`` end-to-end. The contract under test is:

* shared modules referenced by ``from owui_ext.shared.X import ...`` are
  inlined whole;
* missing names raise ``BuildError`` rather than silently producing broken
  output;
* the target's external imports lead the file; each dep's externals stay
  attached to its block so per-dep ``externals -> body`` order is preserved;
* conflicting top-level import bindings (e.g., two sources binding the same
  name to different modules) raise ``BuildError``;
* every dep's ``from __future__`` set must match the target's exactly
  (otherwise the merged compilation unit would silently change parse
  semantics for one side or the other);
* top-level conditional ``try/except ImportError`` blocks remain in place;
* alias / star / module-style local shared imports raise ``BuildError``;
* nested local shared imports (inside functions) raise ``BuildError``;
* package-relative imports anywhere in the source raise ``BuildError``;
* top-level imports interleaved with code, or external imports that follow
  a local-shared import, raise ``BuildError``;
* a shared dep that mixes external and local-shared imports raises
  ``BuildError`` (would reorder side effects across the dep chain);
* circular shared dependencies raise ``BuildError``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from open_webui_plugins_builder.errors import BuildError
from open_webui_plugins_builder.inliner import GENERATED_MARKER, build_target


LOCAL_ROOTS = ("owui_ext.shared",)


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _build(
    repo: Path,
    target_rel: str,
    *,
    target_name: str | None = None,
    source_root: str = "src",
) -> str:
    target_path = repo / target_rel
    return build_target(
        source_text=target_path.read_text(encoding="utf-8"),
        source_path=target_path,
        repo_root=repo,
        local_import_roots=LOCAL_ROOTS,
        source_root=source_root,
        target_name=target_name,
    )


def test_inlines_shared_module_body(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        """Utility helpers."""

        from __future__ import annotations

        import json


        def jload(text: str) -> dict:
            return json.loads(text)
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """
        title: Demo
        version: 0.1.0
        """

        from __future__ import annotations

        from owui_ext.shared.util import jload


        def run(text: str) -> dict:
            return jload(text)
        ''',
    )

    output = _build(tmp_path, "src/owui_ext/tools/demo.py")

    assert GENERATED_MARKER in output
    assert "def jload" in output
    assert "from owui_ext.shared.util" not in output
    # The dep's external imports stay attached to the dep block (after the
    # dep marker), not hoisted above all deps.
    assert "import json" in output


def test_emits_matching_future_imports_once(tmp_path: Path) -> None:
    """When target and deps declare the same ``__future__`` set, emit one line."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        from __future__ import annotations


        def helper() -> int:
            return 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from __future__ import annotations

        from owui_ext.shared.util import helper

        result = helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")

    future_lines = [
        line for line in output.splitlines() if line.startswith("from __future__ import")
    ]
    assert len(future_lines) == 1
    assert "annotations" in future_lines[0]


def test_rejects_dep_with_extra_future_import(tmp_path: Path) -> None:
    """A dep cannot introduce a ``__future__`` flag the target lacks.

    Inlining shares one compilation unit, so the union would silently
    apply the dep's flag to the target's body. Refuse rather than emit
    a release whose semantics differ from the source.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        from __future__ import annotations


        def helper() -> int:
            return 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    with pytest.raises(BuildError, match="from __future__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_target_with_extra_future_import(tmp_path: Path) -> None:
    """The target cannot declare a ``__future__`` flag a dep lacks.

    The dep's body would run under the target's flag in the merged
    file, which silently changes the dep's semantics relative to the
    source. Refuse the build instead.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper() -> int:\n    return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from __future__ import annotations

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    with pytest.raises(BuildError, match="from __future__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_target_and_dep_externals_are_emitted_in_order(tmp_path: Path) -> None:
    """Target externals lead; each dep's externals stay attached to its block.

    The dep-chain ordering invariant is preserved by emitting each dep's
    externals next to its body rather than merging them into the shared top
    section. Identical imports may therefore appear more than once -- Python
    caches modules, so repeated ``import json`` is a no-op.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        import json


        def helper(x: str) -> dict:
            return json.loads(x)
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import json
        from owui_ext.shared.util import helper

        helper("{}")
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    target_import_idx = output.index("import json")
    dep_marker_idx = output.index("# --- inlined from")
    second_import_idx = output.index("import json", dep_marker_idx)
    assert target_import_idx < dep_marker_idx < second_import_idx


def test_rejects_conflicting_external_imports(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        from json import loads


        def helper(x: str):
            return loads(x)
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from yaml import loads
        from owui_ext.shared.util import helper

        helper("{}")
        ''',
    )
    with pytest.raises(BuildError, match="bound by both"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_missing_imported_name_errors(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        def alpha():
            return 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import beta

        beta()
        ''',
    )
    with pytest.raises(BuildError, match="not defined at the top level"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_alias_on_local_import_errors(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper as h

        h()
        ''',
    )
    with pytest.raises(BuildError, match="aliases on local shared imports"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_module_style_local_import_errors(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import owui_ext.shared.util

        owui_ext.shared.util.helper()
        ''',
    )
    with pytest.raises(BuildError, match="not allowed; use"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_star_local_import_errors(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import *

        helper()
        ''',
    )
    with pytest.raises(BuildError, match="not allowed for local shared modules"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_function_level_local_import_errors(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""


        def run():
            from owui_ext.shared.util import helper
            return helper()
        ''',
    )
    with pytest.raises(BuildError, match="must be at module top level"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_nested_module_style_local_import_errors(tmp_path: Path) -> None:
    """Regression: nested ``import owui_ext.shared.util`` must not slip through."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""


        def run():
            import owui_ext.shared.util
            return owui_ext.shared.util.helper()
        ''',
    )
    with pytest.raises(BuildError, match="must be at module top level"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_conditional_top_level_module_style_local_import_errors(tmp_path: Path) -> None:
    """A top-level ``try: import owui_ext.shared.X`` is still nested w.r.t. body."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        try:
            import owui_ext.shared.util
        except ImportError:
            pass
        ''',
    )
    with pytest.raises(BuildError, match="must be at module top level"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_conditional_top_level_local_from_import_errors(tmp_path: Path) -> None:
    """A top-level ``try: from owui_ext.shared.X import ...`` is also rejected."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        try:
            from owui_ext.shared.util import helper
        except ImportError:
            helper = None
        ''',
    )
    with pytest.raises(BuildError, match="must be at module top level"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_circular_local_import_errors(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/a.py",
        '''
        from owui_ext.shared.b import b_helper


        def a_helper():
            return b_helper()
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/shared/b.py",
        '''
        from owui_ext.shared.a import a_helper


        def b_helper():
            return a_helper()
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.a import a_helper

        a_helper()
        ''',
    )
    with pytest.raises(BuildError, match="Circular"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_top_level_try_import_block_preserved(tmp_path: Path) -> None:
    """Top-level conditional imports stay where they are; we don't merge them."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        try:
            import cjson as json_lib
        except ImportError:
            import json as json_lib

        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "try:" in output
    assert "import cjson as json_lib" in output
    assert "except ImportError" in output


def test_marker_block_appears_before_imports(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    marker_pos = output.index(GENERATED_MARKER)
    inline_pos = output.index("--- inlined from")
    docstring_pos = output.index('version: 0.1')
    assert docstring_pos < marker_pos < inline_pos


def test_marker_includes_target_name_when_provided(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py", target_name="demo")
    assert "--target demo" in output
    assert "<name>" not in output


def test_marker_falls_back_to_all_when_no_target_name(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py", target_name=None)
    assert "--all" in output


def test_marker_lists_future_imports(tmp_path: Path) -> None:
    """The marker must declare the future imports actually emitted into the file."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        from __future__ import annotations, division


        def helper() -> int:
            return 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from __future__ import annotations, division

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py", target_name="demo")
    marker_lines = [
        line
        for line in output.splitlines()
        if line.startswith("# Future imports:")
    ]
    assert len(marker_lines) == 1
    line = marker_lines[0]
    assert "annotations" in line
    assert "division" in line


def test_marker_records_no_future_imports_when_absent(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py", target_name="demo")
    assert "# Future imports: (none)" in output


def test_rejects_collision_between_two_shared_modules(tmp_path: Path) -> None:
    """Two shared modules defining the same top-level name silently overwrite.

    The builder must refuse this -- after inlining both bodies live in one
    module namespace and Python's last-definition-wins rule turns earlier
    callers' lookups into wrong-target calls.
    """

    _write(
        tmp_path / "src/owui_ext/shared/a.py",
        '''
        def _helper() -> int:
            return 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/shared/b.py",
        '''
        def _helper() -> int:
            return 2
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.a import _helper as _a_helper_unused  # noqa
        from owui_ext.shared.b import _helper as _b_helper_unused  # noqa
        ''',
    )
    # Plan rejects aliases; rewrite without aliases. The collision is the
    # point being tested.
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.a import _helper

        result = _helper()
        ''',
    )
    # Pull dep b transitively by chaining: have `a` import from `b`.
    _write(
        tmp_path / "src/owui_ext/shared/a.py",
        '''
        from owui_ext.shared.b import _helper as _b_helper  # noqa
        ''',
    )
    with pytest.raises(BuildError):
        # Aliased local import is itself a BuildError, so use a different
        # path to set up the actual collision: two deps both define _helper.
        _build(tmp_path, "src/owui_ext/tools/demo.py")

    # Set up the genuine collision: a chains to b, both define _helper.
    _write(
        tmp_path / "src/owui_ext/shared/a.py",
        '''
        from owui_ext.shared.b import other


        def _helper() -> int:
            return other() + 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/shared/b.py",
        '''
        def other() -> int:
            return 2


        def _helper() -> int:
            return 99
        ''',
    )
    with pytest.raises(BuildError, match="bound by both"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_collision_between_shared_and_target(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        def helper() -> int:
            return 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper


        def helper() -> int:
            return 99
        ''',
    )
    with pytest.raises(BuildError, match="imported from a local shared module"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_collision_between_external_import_and_shared_def(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        def loads() -> str:
            return "fake"
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from json import loads
        from owui_ext.shared.util import loads as _unused
        ''',
    )
    # The above has aliasing on local import -- rewrite to avoid it.
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from json import loads

        from owui_ext.shared.util import loads
        ''',
    )
    # Both import `loads` from json AND import-and-redefine via the local
    # shared module's `def loads`. The collision check must fire.
    with pytest.raises(BuildError):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_conditional_import_collision_between_deps(tmp_path: Path) -> None:
    """``try/except ImportError`` blocks inline verbatim, so two deps both
    binding the same name through conditional imports must be flagged."""

    _write(
        tmp_path / "src/owui_ext/shared/a.py",
        '''
        try:
            import cjson as json_lib
        except ImportError:
            import json as json_lib


        def from_a() -> int:
            return 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/shared/b.py",
        '''
        try:
            import cjson as json_lib
        except ImportError:
            import json as json_lib


        def from_b() -> int:
            return 2
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.a import from_a
        from owui_ext.shared.b import from_b

        from_a()
        from_b()
        ''',
    )
    with pytest.raises(BuildError, match="bound by both"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_conditional_assignment_collision(tmp_path: Path) -> None:
    """Top-level ``if/else`` assignments bind to module scope, so a dep using
    one and a target using another to bind the same name must collide."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        try:
            HELPER = 1
        except Exception:
            HELPER = 0


        def helper_value() -> int:
            return HELPER
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper_value

        if True:
            HELPER = 99
        else:
            HELPER = 100

        helper_value()
        ''',
    )
    with pytest.raises(BuildError, match="bound by both"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_except_handler_names_do_not_collide(tmp_path: Path) -> None:
    """``except E as exc`` does *not* persistently bind ``exc``.

    Python implicitly deletes the as-target at the end of the except clause
    (to break the traceback reference cycle). The collision check must not
    treat the same handler name in two modules as a persistent collision.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        try:
            BASE = 1
        except Exception as exc:
            BASE = 0


        def helper() -> int:
            return BASE
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        try:
            X = 1
        except Exception as exc:
            X = 0

        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "def helper" in output


def test_bindings_inside_except_body_still_collide(tmp_path: Path) -> None:
    """Real bindings inside an except clause persist (the as-target doesn't,
    but ``helper = something`` inside the except body does). The collision
    check must still fire for those."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        def base() -> int:
            return 1


        try:
            BASE = 1
        except Exception:
            from_except_helper = base
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import base

        try:
            X = 1
        except Exception:
            from_except_helper = base

        from_except_helper()
        ''',
    )
    with pytest.raises(BuildError, match="bound by both"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_conditional_top_level_binding_is_exportable(tmp_path: Path) -> None:
    """Shared modules can bind names inside top-level conditionals; targets
    must be allowed to import those names."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        try:
            VALUE = "real"
        except Exception:
            VALUE = "fallback"


        if True:
            ALT = 1
        else:
            ALT = 2
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import VALUE, ALT

        print(VALUE, ALT)
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "VALUE" in output
    assert "ALT" in output


def test_reexport_via_top_level_import_is_exportable(tmp_path: Path) -> None:
    """Shared modules can re-export third-party names via top-level imports;
    targets must be allowed to import the re-exported name."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        from json import loads as parse_json
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import parse_json

        parse_json("{}")
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "parse_json" in output


def test_same_package_submodule_imports_merge(tmp_path: Path) -> None:
    """``import email.mime.text`` and ``import email.mime.multipart`` both
    bind the same ``email`` package object; the builder must not flag the
    pair as a name collision."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        import email.mime.multipart


        def helper() -> str:
            return "ok"
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import email.mime.text

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "import email.mime.text" in output
    assert "import email.mime.multipart" in output


def test_same_alias_to_different_submodules_collides(tmp_path: Path) -> None:
    """``import a.b as foo`` and ``import a.c as foo`` bind different objects
    to ``foo`` and *should* fail."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        import email.mime.text as mime


        def helper() -> str:
            return "ok"
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import email.mime.multipart as mime

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    with pytest.raises(BuildError, match="bound by both"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_function_local_assignments_do_not_collide(tmp_path: Path) -> None:
    """Bindings inside function or class bodies are local scope; sharing
    the name across deps must not be a build-time collision."""

    _write(
        tmp_path / "src/owui_ext/shared/a.py",
        '''
        def from_a() -> int:
            local_helper = 1
            return local_helper
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/shared/b.py",
        '''
        def from_b() -> int:
            local_helper = 2  # same name, but local to from_b's frame
            return local_helper
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.a import from_a
        from owui_ext.shared.b import from_b

        from_a()
        from_b()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "def from_a" in output
    assert "def from_b" in output


def test_identical_imports_from_target_and_dep_do_not_collide(
    tmp_path: Path,
) -> None:
    """Regression: ``import json`` in both target and dep must build cleanly.

    Identical imports across modules used to be deduped into a single line.
    The dep-chain ordering fix moves each dep's externals next to its body,
    so the same ``import json`` may appear in both the target section and
    the dep block. The build must still succeed (no name collision raised).
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        import json


        def parse(text: str) -> dict:
            return json.loads(text)
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import json

        from owui_ext.shared.util import parse

        parse("{}")
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py", target_name="demo")
    assert "import json" in output
    assert "def parse" in output


def test_custom_source_root_resolves_shared_modules(tmp_path: Path) -> None:
    """``settings.source_root`` lets repos with non-``src`` layouts opt in."""

    _write(
        tmp_path / "lib/owui_ext/shared/util.py",
        "def helper(): return 7\n",
    )
    _write(
        tmp_path / "lib/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    output = _build(
        tmp_path,
        "lib/owui_ext/tools/demo.py",
        source_root="lib",
    )
    assert "def helper" in output
    assert "from owui_ext.shared" not in output


def test_rejects_relative_import_at_top_level(tmp_path: Path) -> None:
    """Package-relative imports cannot survive in the single-file output.

    Open WebUI loads the artifact with no enclosing package, so
    ``from ..shared.util import helper`` would raise ``ImportError`` at
    runtime. The builder must refuse it instead of emitting a broken file.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from ..shared.util import helper

        helper()
        ''',
    )
    with pytest.raises(BuildError, match="package-relative imports"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_relative_import_in_shared_dep(tmp_path: Path) -> None:
    """Relative imports inside a shared dependency are also refused."""

    _write(
        tmp_path / "src/owui_ext/shared/base.py",
        "BASE = 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            from __future__ import annotations

            from .base import BASE


            def helper() -> int:
                return BASE
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    with pytest.raises(BuildError, match="package-relative imports"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_top_level_import_after_code(tmp_path: Path) -> None:
    """Hoisting a late import would silently change execution order.

    A top-level statement that runs *before* an import (e.g., setting an
    env var the imported module reads) must not be reordered. The builder
    refuses such modules rather than emit a release that diverges from the
    source.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import os

        os.environ["FEATURE_FLAG"] = "1"

        import json

        json.dumps({})
        ''',
    )
    with pytest.raises(BuildError, match="must appear before"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_docstring_and_future_before_imports(tmp_path: Path) -> None:
    """Module docstring and ``from __future__`` lines may precede regular imports."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "from __future__ import annotations\n\n\ndef helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from __future__ import annotations

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "def helper" in output


def test_rejects_relative_import_inside_function_body(tmp_path: Path) -> None:
    """Function-internal relative imports also leak into the artifact verbatim."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def run() -> int:
            from .helper import compute
            return compute()
        ''',
    )
    with pytest.raises(BuildError, match="package-relative imports"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_relative_import_inside_conditional_block(tmp_path: Path) -> None:
    """``try: from .util import X`` at module top level still emits as-is."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        try:
            from .util import helper
        except ImportError:
            helper = None
        ''',
    )
    with pytest.raises(BuildError, match="package-relative imports"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_relative_import_inside_shared_dep_function(tmp_path: Path) -> None:
    """A shared dep with a function-level relative import must also be refused."""

    _write(
        tmp_path / "src/owui_ext/shared/base.py",
        "BASE = 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            def helper() -> int:
                from .base import BASE
                return BASE
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    with pytest.raises(BuildError, match="package-relative imports"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_external_from_import_after_local_shared(tmp_path: Path) -> None:
    """External ``from X import Y`` must not appear after a local-shared import.

    The inliner hoists every external import above every dep body, so a
    source where the external import depended on a local-shared import's
    side effects would break in the generated file.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper
        from json import dumps

        helper()
        dumps({})
        ''',
    )
    with pytest.raises(BuildError, match="follows a local-shared import"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_external_import_after_local_shared(tmp_path: Path) -> None:
    """An ``import madeup_plugin`` after a local-shared import is also refused."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper
        import json

        helper()
        json.dumps({})
        ''',
    )
    with pytest.raises(BuildError, match="follows a local-shared import"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_external_after_local_shared_in_dep(tmp_path: Path) -> None:
    """The same ordering constraint applies inside shared dependency modules."""

    _write(
        tmp_path / "src/owui_ext/shared/base.py",
        "BASE = 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            from owui_ext.shared.base import BASE
            import json


            def helper() -> str:
                return json.dumps({"base": BASE})
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper

        helper()
        ''',
    )
    with pytest.raises(BuildError, match="follows a local-shared import"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_externals_before_local_shared(tmp_path: Path) -> None:
    """The canonical ordering (externals first, then locals) is accepted."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper(x): return x\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import json

        from owui_ext.shared.util import helper

        helper(json.dumps({}))
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "def helper" in output
    assert "import json" in output


def test_rejects_dep_with_externals_and_local_imports(tmp_path: Path) -> None:
    """A shared dep module cannot mix external imports with local-shared imports.

    Topological sort places ``B`` before ``A`` when ``A`` depends on ``B``,
    which means ``A``'s external imports get emitted *after* ``B``'s body --
    but in source they ran *before* ``B`` was loaded. Refuse the build so
    the user re-arranges the modules rather than shipping silently
    reordered side effects.
    """

    _write(
        tmp_path / "src/owui_ext/shared/base.py",
        "BASE_MEAN = 2.0\n",
    )
    _write(
        tmp_path / "src/owui_ext/shared/derived.py",
        textwrap.dedent(
            """
            import math

            from owui_ext.shared.base import BASE_MEAN


            DERIVED = math.sqrt(BASE_MEAN)
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.derived import DERIVED

        DERIVED
        ''',
    )
    with pytest.raises(BuildError, match="cannot mix external imports"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_dep_chain_with_pure_local_intermediate_preserves_order(
    tmp_path: Path,
) -> None:
    """``A -> B -> C`` where only the leaf C has externals: order is preserved.

    The intermediate dep ``B`` has only local-shared imports, the leaf
    ``C`` has only externals, so the new "no mixing" rule allows it. The
    flat layout still mirrors source side-effect order:
    ``c1 externals -> C body -> B body -> A body -> target body``.
    """

    _write(
        tmp_path / "src/owui_ext/shared/leaf.py",
        textwrap.dedent(
            """
            import math


            LEAF = math.pi
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/middle.py",
        "from owui_ext.shared.leaf import LEAF\n\nMIDDLE = LEAF\n",
    )
    _write(
        tmp_path / "src/owui_ext/shared/top.py",
        "from owui_ext.shared.middle import MIDDLE\n\nTOP = MIDDLE\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.top import TOP

        TOP
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")

    leaf_idx = output.index("# --- inlined from src/owui_ext/shared/leaf.py")
    middle_idx = output.index("# --- inlined from src/owui_ext/shared/middle.py")
    top_idx = output.index("# --- inlined from src/owui_ext/shared/top.py")
    math_idx = output.index("import math")

    # Topological order: leaf before middle before top.
    assert leaf_idx < middle_idx < top_idx
    # The leaf's external import sits inside the leaf block.
    assert leaf_idx < math_idx < middle_idx


def test_rejects_top_level_import_after_conditional_block(tmp_path: Path) -> None:
    """A top-level ``try`` counts as code: imports after it cannot be hoisted."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import os

        try:
            os.environ["X"] = "1"
        except Exception:
            pass

        import json

        json.dumps({})
        ''',
    )
    with pytest.raises(BuildError, match="must appear before"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")
