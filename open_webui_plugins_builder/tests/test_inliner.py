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
    """When target and deps declare the same ``__future__`` set, emit one line.

    ``annotations`` is refused (F5), but other no-op flags like
    ``division`` may still appear and exercise the merging logic.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        from __future__ import division


        def helper() -> int:
            return 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from __future__ import division

        from owui_ext.shared.util import helper

        result = helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")

    future_lines = [
        line for line in output.splitlines() if line.startswith("from __future__ import")
    ]
    assert len(future_lines) == 1
    assert "division" in future_lines[0]


def test_rejects_dep_with_extra_future_import(tmp_path: Path) -> None:
    """A dep cannot introduce a ``__future__`` flag the target lacks.

    Inlining shares one compilation unit, so the union would silently
    apply the dep's flag to the target's body. Refuse rather than emit
    a release whose semantics differ from the source. (Uses
    ``division`` here since ``annotations`` is refused outright by F5.)
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        from __future__ import division


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


def test_rejects_target_with_future_annotations(tmp_path: Path) -> None:
    """``from __future__ import annotations`` is refused in the target
    too: PEP 563 is per-compilation-unit, so a target-only opt-in
    still applies to the dep bodies in the merged file, changing
    annotation semantics for deps that were live in the source.
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
    with pytest.raises(BuildError, match="from __future__ import annotations"):
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
    with pytest.raises(BuildError, match="import \\*"):
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
    """The marker must declare the future imports actually emitted into the file.

    ``annotations`` is refused everywhere by F5; this test exercises
    the marker emission with ``division``, which is a no-op flag in
    modern Python but still parses as a valid future import.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        from __future__ import division


        def helper() -> int:
            return 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from __future__ import division

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


def test_rejects_external_star_import_at_top_level(tmp_path: Path) -> None:
    """`from X import *` at the top of the target is refused.

    Star imports bind names that the inliner cannot enumerate, so the
    name-collision detector silently skips them. A later inlined dep
    definition could shadow a star-bound name (or be shadowed by it),
    silently changing which implementation the target body calls. Refuse
    rather than emit a release that diverges from source semantics.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from math import *

        sin(0)
        ''',
    )
    with pytest.raises(BuildError, match="import \\*"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_star_import_in_shared_dep(tmp_path: Path) -> None:
    """Star imports inside a shared dependency are also refused."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            from math import *


            def helper() -> float:
                return sin(0)
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
    with pytest.raises(BuildError, match="import \\*"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_star_import_inside_top_level_conditional(tmp_path: Path) -> None:
    """Star imports inside ``try``/``if`` blocks at module level are refused too.

    Python only permits ``from X import *`` at module-level scope, but that
    includes top-level conditional blocks. Names brought in there still leak
    into module globals and can collide with inlined deps.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        try:
            from math import *
        except ImportError:
            pass

        sin(0)
        ''',
    )
    with pytest.raises(BuildError, match="import \\*"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_scope_walrus(tmp_path: Path) -> None:
    """Walrus assignments at module scope are refused.

    The collision detector enumerates module-level bindings via ``Assign``,
    imports, ``def``, and a few statement-form targets. It does not track
    ``NamedExpr`` (walrus). A walrus-bound name in one dep could silently
    collide with a ``def`` of the same name in another dep without the
    detector noticing. Refuse rather than ship a release that may call a
    different implementation than the source.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        if (helper := 1) > 0:
            print(helper)
        ''',
    )
    with pytest.raises(BuildError, match="walrus"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_scope_walrus_in_shared_dep(tmp_path: Path) -> None:
    """Walrus at module scope inside a shared dep is also refused."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            if (cached := 0) >= 0:
                _GLOBAL = cached


            def helper() -> int:
                return _GLOBAL
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
    with pytest.raises(BuildError, match="walrus"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_walrus_in_function_default_value(tmp_path: Path) -> None:
    """Walrus inside a function's default arg evaluates at module scope.

    The ``def`` statement runs at module scope; default values are
    evaluated then, so ``def f(x=(y := 1)):`` binds ``y`` as a module
    global. The body itself is local, but the default is not.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def main(x: int = (helper := 1)) -> int:
            return x + helper


        main()
        ''',
    )
    with pytest.raises(BuildError, match="walrus"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_walrus_in_decorator_expression(tmp_path: Path) -> None:
    """Walrus inside a decorator expression also runs at module scope."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""


        def make_decorator(value):
            def _wrap(fn):
                return fn

            return _wrap


        @make_decorator(captured := 1)
        def main() -> int:
            return captured


        main()
        ''',
    )
    with pytest.raises(BuildError, match="walrus"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_walrus_in_class_base(tmp_path: Path) -> None:
    """Walrus inside a class's base list runs at module scope."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""


        class _Mix:
            pass


        class Demo((helper := _Mix)):
            pass


        Demo()
        ''',
    )
    with pytest.raises(BuildError, match="walrus"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_read_only_global_declaration_with_local_import(
    tmp_path: Path,
) -> None:
    """A ``global X`` declaration that does NOT rebind X is just a lookup
    hint, not a write. The build must allow it even when X is also
    imported from a local shared module."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "def helper() -> str:\n    return \"shared\"\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper


        def run() -> str:
            global helper
            return helper()


        run()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "global helper" in output


def test_global_write_in_dep_collides_with_other_dep_definition(
    tmp_path: Path,
) -> None:
    """``global X; X = ...`` inside a function body must reach the
    name-collision detector via ``_collect_global_writes``.

    Without that hook, the inliner sees only ``clobber`` from shared_a and
    ``helper`` from shared_b -- no overlap. But after inlining, the
    merged module's ``shared_a.clobber()`` writes ``helper`` into the
    same namespace ``shared_b`` populated, silently overriding the
    function the target meant to call.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def clobber() -> None:
                global helper
                helper = lambda: "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import clobber
        from owui_ext.shared.shared_b import helper

        clobber()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_global_walrus_in_dep_collides_with_other_dep_definition(
    tmp_path: Path,
) -> None:
    """The walrus form of the same module-global write also collides."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def clobber() -> None:
                global helper
                (helper := lambda: "a")
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import clobber
        from owui_ext.shared.shared_b import helper

        clobber()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_global_match_in_dep_collides_with_other_dep_definition(
    tmp_path: Path,
) -> None:
    """``match`` capture writing through ``global`` also collides."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def clobber(value: int) -> None:
                global helper
                match value:
                    case 1:
                        helper = lambda: "one"
                    case _:
                        helper = lambda: "other"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import clobber
        from owui_ext.shared.shared_b import helper

        clobber(1)
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_global_except_as_in_dep_collides(tmp_path: Path) -> None:
    """``except E as helper`` under ``global helper`` temporarily binds
    and then deletes the module global. After inlining, that delete
    erases shared_b's ``helper`` from the merged module, so target
    callers see ``NameError`` -- the build must refuse it up front.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def clobber() -> None:
                global helper
                try:
                    raise RuntimeError("boom")
                except RuntimeError as helper:
                    pass
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import clobber
        from owui_ext.shared.shared_b import helper

        clobber()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_global_walrus_in_nested_def_default_collides(tmp_path: Path) -> None:
    """A walrus in a nested function's default evaluates in the outer
    function's scope. With ``global helper`` there, the walrus writes
    module global helper -- which collides with another dep's ``def
    helper``."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def outer() -> None:
                global helper

                def inner(x: object = (helper := lambda: "a")) -> object:
                    return x

                inner()
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import outer
        from owui_ext.shared.shared_b import helper

        outer()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_read_only_global_in_dep_collides_with_other_dep_definition(
    tmp_path: Path,
) -> None:
    """A dep's ``global X; return X()`` (no write) silently resolves to
    another dep's top-level ``X`` after inlining -- the original
    separated modules would NameError. Refuse this divergence."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def read() -> object:
                global helper
                return helper()
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import read
        from owui_ext.shared.shared_b import helper

        read()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_implicit_global_reference_in_dep_collides_with_other_dep_definition(
    tmp_path: Path,
) -> None:
    """A function that references ``helper`` without any ``global``
    declaration still resolves to module globals at runtime. After
    inlining, that lookup silently picks up another dep's ``def helper``.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def read() -> object:
                return helper()
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import read
        from owui_ext.shared.shared_b import helper

        read()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_two_unbound_global_references_with_no_definition(
    tmp_path: Path,
) -> None:
    """Two deps that both *only* read a name via ``global`` (with no
    definition anywhere) are NOT a collision: in source AND in the
    merged file, the reference raises ``NameError``. The build must
    allow this rather than flag a phantom binding clash."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def read_a() -> object:
                global helper
                return helper()
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def read_b() -> object:
                global helper
                return helper()
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import read_a
        from owui_ext.shared.shared_b import read_b

        read_a
        read_b
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "global helper" in output


def test_builtin_shadow_in_dep_collides(tmp_path: Path) -> None:
    """A dep that calls a builtin (``list(...)``) must not silently
    resolve to a same-named top-level definition in another dep after
    inlining. Module-globals lookup beats the builtins module."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def use_list(xs: object) -> object:
                return list(xs)
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def list(items: object) -> object:
                return [items]
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import use_list
        from owui_ext.shared.shared_b import list

        use_list((1, 2))
        list(3)
        ''',
    )
    with pytest.raises(BuildError, match="(?i)list"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_module_scope_default_reference_in_dep_collides(
    tmp_path: Path,
) -> None:
    """A function default at module top is evaluated when the ``def``
    statement runs. ``def read(x=helper):`` reads ``helper`` from module
    globals at import. After inlining, that lookup picks up another
    dep's ``def helper`` even though shared_a never imported it."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def read(x: object = helper) -> object:
                return x
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import read
        from owui_ext.shared.shared_b import helper

        read()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_module_scope_decorator_reference_in_dep_collides(
    tmp_path: Path,
) -> None:
    """Decorators on a module-top ``def`` evaluate at module scope."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            @helper
            def read() -> int:
                return 1
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper(fn):
                return fn
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import read
        from owui_ext.shared.shared_b import helper

        read()
        helper
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_future_annotations_in_dep_with_cross_dep_type(
    tmp_path: Path,
) -> None:
    """F5: a dep using ``from __future__ import annotations`` plus a
    type name that only resolves through another dep is exactly the
    silent-divergence case the dep refusal exists to catch. Original
    semantics: ``typing.get_type_hints(use)`` would raise NameError
    in dep_a alone (Helper is in dep_b). Inlined semantics: it
    resolves through the merged globals. Refused at the dep's
    future-annotations import.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            from __future__ import annotations


            def use(x: Helper) -> int:
                return 1
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Helper:
                pass
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import use
        from owui_ext.shared.shared_b import Helper

        use(Helper())
        ''',
    )
    with pytest.raises(BuildError, match="from __future__ import annotations"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_builtin_reference_in_dep(tmp_path: Path) -> None:
    """References to Python built-ins (``print``, ``len``, ...) inside a
    function are *not* unbound globals -- they live in the builtins
    module. The collision check must not flag them."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            def helper() -> int:
                return len([1, 2, 3])
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import helper


        def main() -> int:
            return print(helper()) or 0


        main()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "len([1, 2, 3])" in output


def test_read_only_global_in_target_collides_with_dep_definition(
    tmp_path: Path,
) -> None:
    """The same divergence on the target side: target's ``global X``
    accidentally finds a dep's ``def X`` after inlining."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def trigger() -> None:
                pass
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import trigger
        from owui_ext.shared.shared_b import helper as _shared_helper


        def run() -> object:
            global helper
            return helper()


        trigger()
        run()
        _shared_helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_global_del_in_dep_collides(tmp_path: Path) -> None:
    """``global helper; del helper`` removes the module-global binding.
    After inlining, that delete erases another dep's top-level
    ``helper``, so the target sees ``NameError`` on the next call. The
    build must refuse this up front."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            helper = "placeholder"


            def clobber() -> None:
                global helper
                del helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import clobber
        from owui_ext.shared.shared_b import helper

        clobber()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_module_scope_del_in_dep_destroys_other_dep_definition(
    tmp_path: Path,
) -> None:
    """``del helper`` at module scope of one dep removes the merged
    module's ``helper`` global -- silently destroying another dep's
    top-level binding. In the original separated modules, that ``del``
    raises ``NameError`` because ``helper`` was never bound there."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            marker = "a"
            del helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import marker
        from owui_ext.shared.shared_b import helper

        marker
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_module_scope_except_as_in_dep_destroys_other_dep_definition(
    tmp_path: Path,
) -> None:
    """``except E as helper`` at module scope temporarily binds ``helper``
    and then implicitly deletes it after the handler. After inlining,
    that erases another dep's top-level ``helper``."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            marker = "a"
            try:
                raise ValueError("trigger")
            except ValueError as helper:
                _ = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import marker
        from owui_ext.shared.shared_b import helper

        marker
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_module_scope_del_inside_top_level_if_collides(
    tmp_path: Path,
) -> None:
    """A destructive ``del`` nested inside top-level control flow still
    runs at module load time and is detected."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            FLAG = True
            marker = "a"
            if FLAG:
                del helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import marker
        from owui_ext.shared.shared_b import helper

        marker
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_module_scope_del_with_no_other_dep_definition(
    tmp_path: Path,
) -> None:
    """A self-contained ``del`` of a name that no other source binds is
    allowed -- the destruction stays inside the originating module's
    own state and does not silently mutate cross-source bindings."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            _scratch = 1
            del _scratch


            def value() -> int:
                return 1
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import value

        value()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "del _scratch" in output


def test_allows_del_inside_function_body_without_global(
    tmp_path: Path,
) -> None:
    """A ``del`` inside a function body without ``global X`` is local
    and does not erase module globals -- the build must allow it even
    when another dep binds the same name."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def runner() -> int:
                helper = 5
                del helper
                return 0
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import runner
        from owui_ext.shared.shared_b import helper

        runner()
        helper()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "def runner" in output
    assert "def helper" in output


def test_class_body_use_before_bind_collides(tmp_path: Path) -> None:
    """A class body that loads ``helper`` before binding it falls back to
    module globals via ``LOAD_NAME``. After inlining, that lookup
    silently resolves to another dep's top-level ``helper``."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                picked = helper
                helper = "local"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_body_never_bound_reference_collides(tmp_path: Path) -> None:
    """A class body that references ``helper`` and never binds it falls
    back to module globals at every use. Another dep's ``def helper``
    silently fills the gap after inlining."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_method_default_value_reference_collides(
    tmp_path: Path,
) -> None:
    """Method defaults inside a class body evaluate at class-body time
    and look up free names in module globals (LOAD_NAME)."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            DEFAULT = 99
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                def method(self, x: int = DEFAULT) -> int:
                    return x
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import DEFAULT
        from owui_ext.shared.shared_b import Demo

        DEFAULT
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)default"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_method_decorator_reference_collides(tmp_path: Path) -> None:
    """A decorator on a method inside a class body evaluates at
    class-body time and resolves free names through module globals."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper(fn):
                return fn
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                @helper
                def method(self) -> int:
                    return 1
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_inner_class_base_reference_collides(tmp_path: Path) -> None:
    """A nested class's base list inside an outer class body is
    evaluated in the outer class's namespace, which uses LOAD_NAME and
    falls back to module globals when the name is not a class attribute."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            class BaseHelper:
                pass
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                class Inner(BaseHelper):
                    pass
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import BaseHelper
        from owui_ext.shared.shared_b import Demo

        BaseHelper
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)basehelper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_param_annotation_without_future_collides(
    tmp_path: Path,
) -> None:
    """Without ``from __future__ import annotations``, parameter
    annotations on methods evaluate at class-body time and look up free
    names in module globals."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            class Helper:
                pass
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                def method(self, x: Helper) -> int:
                    return 1
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import Helper
        from owui_ext.shared.shared_b import Demo

        Helper
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dep_with_class_method_using_future_annotations(
    tmp_path: Path,
) -> None:
    """F5: a dep that uses ``from __future__ import annotations`` to
    declare a class method whose param annotation references a name
    bound by another dep is exactly the cross-dep evaluation hazard
    the dep refusal targets. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            class Helper:
                pass
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            from __future__ import annotations


            class Demo:
                def method(self, x: Helper) -> int:
                    return 1
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import Helper
        from owui_ext.shared.shared_b import Demo

        Helper
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="from __future__ import annotations"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_body_except_star_implicit_delete_collides(
    tmp_path: Path,
) -> None:
    """``except* E as helper`` in a class body inside an ``ast.TryStar``
    (Python 3.11 ``try/except*``) implicitly deletes ``helper`` at
    handler end. Without analyzing ``TryStar``, the build would let a
    later ``picked = helper`` read pass even though after inlining it
    silently resolves to another dep's top-level binding."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                helper = "local"
                try:
                    raise ExceptionGroup("g", [ValueError("x")])
                except* ValueError as helper:
                    pass
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_function_internal_class_use_before_bind_collides(
    tmp_path: Path,
) -> None:
    """A class defined inside a function still uses ``LOAD_NAME`` for
    free names, falling through to the enclosing function's locals
    and then to module globals. ``def make(): class Demo: picked =
    helper; helper = "local"`` would silently resolve ``helper`` to
    another dep's top-level binding after inlining."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def make() -> object:
                class Demo:
                    picked = helper
                    helper = "local"

                return Demo
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import make

        helper()
        make()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_function_internal_class_using_function_local(
    tmp_path: Path,
) -> None:
    """When the enclosing function binds ``helper`` as a local, a
    nested class body that loads ``helper`` finds it via the enclosing
    function scope -- no fallback to module globals, no inline
    divergence. The build must allow this even when another source
    binds the same module-global name."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def make() -> object:
                helper = 1

                class Demo:
                    picked = helper

                return Demo
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import make

        helper()
        make()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_class_body_finally_reads_after_try_del_collides(
    tmp_path: Path,
) -> None:
    """When the ``try`` body deletes a class-local binding, the
    ``finally`` block runs with that name unbound. A subsequent load
    inside ``finally`` falls through to module globals after inlining
    -- silently picking up another dep's same-named binding instead of
    raising ``NameError`` like the original separated module would."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                helper = "local"
                try:
                    del helper
                finally:
                    picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_body_match_class_pattern_load_collides(
    tmp_path: Path,
) -> None:
    """``case Helper():`` (MatchClass) loads ``Helper`` at match time
    via ``LOAD_NAME``. When the class body later binds ``Helper``,
    symtable cannot tell the load from the bind ordering and would
    let the build through -- but at runtime the merged file would
    silently pick up another dep's ``Helper`` definition."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            class Helper:
                pass
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            VALUE: object = object()


            class Demo:
                match VALUE:
                    case Helper():
                        picked = "matched"
                    case _:
                        picked = "fallback"
                Helper = "local"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import Helper
        from owui_ext.shared.shared_b import Demo

        Helper
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_class_body_with_as_target_used_in_body(
    tmp_path: Path,
) -> None:
    """``with cm() as helper: picked = helper`` reads ``helper`` inside
    the with body where it is bound by the as-target. Must not flag a
    cross-source collision when another dep happens to bind the same
    module-global name."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            from contextlib import nullcontext


            class Demo:
                with nullcontext("inside") as helper:
                    picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo.picked
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_allows_class_body_for_target_used_in_body(
    tmp_path: Path,
) -> None:
    """``for helper in iter: picked = helper`` reads ``helper`` inside
    the loop body where it is bound by the for-target. Must not flag
    a cross-source collision."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                for helper in (1, 2):
                    picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_allows_class_body_except_as_used_in_handler(
    tmp_path: Path,
) -> None:
    """``except E as helper: picked = helper`` reads ``helper`` inside
    the except handler where it is bound by the as-name. Must not flag
    a cross-source collision."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                try:
                    raise ValueError("trigger")
                except ValueError as helper:
                    picked = repr(helper)
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_allows_class_body_match_capture_used_in_case(
    tmp_path: Path,
) -> None:
    """``case helper: picked = helper`` captures ``helper`` for the
    duration of the case body. Must not flag a cross-source collision
    when another dep binds the same module-global name."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            VALUE = 1


            class Demo:
                match VALUE:
                    case helper:
                        picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_class_body_finally_del_unbinds_for_later_load(
    tmp_path: Path,
) -> None:
    """A class body that binds ``helper`` then unbinds it in a
    ``finally`` block must not leave a later load satisfied. The
    finally always runs, so the post-statement ``helper`` is gone --
    the merged file would silently substitute another dep's
    top-level ``helper``."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                helper = "local"
                try:
                    pass
                finally:
                    del helper
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_body_if_branch_del_drops_must_bind(
    tmp_path: Path,
) -> None:
    """Even when ``helper`` is bound earlier in the class body, a
    branch that ``del``s it via ``if/else`` makes the post-statement
    ``helper`` may-remove, so a subsequent load is treated as
    unbound."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            FLAG = True


            class Demo:
                helper = "local"
                if FLAG:
                    del helper
                else:
                    pass
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_body_except_as_name_drops_must_bind(
    tmp_path: Path,
) -> None:
    """``except E as helper`` implicitly deletes ``helper`` after the
    handler. A class body that uses ``helper`` only to bind it through
    that mechanism cannot rely on it being present at a later load."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                try:
                    raise ValueError("trigger")
                except ValueError as helper:
                    pass
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_global_starred_unpack_in_dep_collides_with_other_dep(
    tmp_path: Path,
) -> None:
    """``global helper; *helper, = [...]`` writes the captured slice into
    the module global ``helper``. After inlining, that overwrites
    another dep's top-level ``def helper`` -- a silent corruption that
    the build must refuse."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def clobber() -> None:
                global helper
                *helper, = [lambda: "a"]
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import clobber
        from owui_ext.shared.shared_b import helper

        clobber()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_module_scope_starred_for_target_collides_with_dep_definition(
    tmp_path: Path,
) -> None:
    """A top-level ``for *helper, x in ...`` binds ``helper`` at module
    scope (sliced list of leading items). When another dep defines
    ``helper`` as a function, the target ends up calling the rebound
    list."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            for *helper, x in [(1, 2)]:
                pass
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import x
        from owui_ext.shared.shared_b import helper

        x
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_body_conditional_bind_without_else_collides(
    tmp_path: Path,
) -> None:
    """A bare ``if`` (no ``else``) in a class body is may-bind: when the
    branch does not run, the post-statement load falls back to module
    globals. After inlining, that load silently picks up another dep's
    same-named binding -- so the build must refuse this construction
    even when the static condition looks like it should run."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            FLAG = False


            class Demo:
                if FLAG:
                    helper = "local"
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_body_for_loop_bind_collides(tmp_path: Path) -> None:
    """A ``for`` loop body in a class body is may-bind: if the iterable
    is empty, the loop body never runs, and the loop target stays
    unbound. Refuse class bodies that lean on a for-loop binding for a
    later same-line reference when another source defines that name."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                for helper in ():
                    pass
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_class_body_match_without_default_collides(tmp_path: Path) -> None:
    """A ``match`` without an exhaustive case (no wildcard or bare-name
    pattern) is may-bind: if no case matches, no capture happens and
    the post-match load falls back to module globals."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            VALUE = 1


            class Demo:
                match VALUE:
                    case 1:
                        helper = "one"
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_class_body_if_else_both_branches_bind(
    tmp_path: Path,
) -> None:
    """``if cond: helper = ...; else: helper = ...`` inside a class body
    is must-bind: both branches assign ``helper``, so a later reference
    finds the class-local binding regardless of the branch taken."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            FLAG = True


            class Demo:
                if FLAG:
                    helper = "yes"
                else:
                    helper = "no"
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo.picked
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_allows_class_body_try_except_both_bind(tmp_path: Path) -> None:
    """``try/except`` where both the try body and every handler bind
    ``helper`` is must-bind: reaching after the try means one of those
    branches ran, so the class-local ``helper`` is set."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                try:
                    helper = "imported"
                except ImportError:
                    helper = None
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo.picked
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_allows_class_body_with_block_bind(tmp_path: Path) -> None:
    """A ``with`` block inside a class body always runs its body when
    ``__enter__`` succeeds. Bindings made inside the body and the
    as-target are must-bind for code after the ``with``."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            from contextlib import nullcontext


            class Demo:
                with nullcontext("inside") as helper:
                    pass
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo.picked
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_allows_same_dep_external_import_and_del(tmp_path: Path) -> None:
    """A dep that imports and immediately deletes its own external name
    must build cleanly when no other source binds that name. The merged
    file has the import hoisted, the dep body deletes it -- behavior
    matches the original separated module's import-and-cleanup."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            import math


            VALUE = math.pi
            del math
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import VALUE

        VALUE
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "del math" in output


def test_external_import_destroyed_by_other_dep_collides(
    tmp_path: Path,
) -> None:
    """When two deps both import ``math`` (so the merged file has the
    hoisted import shared between them), one dep's ``del math`` would
    silently break the other dep's reference. The build must refuse
    this cross-source destruction."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            import math


            VALUE = math.pi
            del math
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            import math


            def diameter(r: float) -> float:
                return 2 * math.pi * r
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import VALUE
        from owui_ext.shared.shared_b import diameter

        VALUE
        diameter(1.0)
        ''',
    )
    with pytest.raises(BuildError, match="(?i)math"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_class_body_use_after_bind(tmp_path: Path) -> None:
    """``class Demo: helper = 1; y = helper`` binds ``helper`` in class
    scope before the load, so the LOAD_NAME finds the class-local
    binding and never falls back to module globals -- the build must
    allow this even when another source binds the same module-global
    name."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                helper = 1
                picked = helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo.picked
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_allows_class_body_self_reference_inside_method(
    tmp_path: Path,
) -> None:
    """Methods access class attributes through ``self`` (or ``cls``),
    not via free names in the class body. A method body that reads
    ``self.attr`` must not trigger the class-body unbound-ref check
    even when another dep happens to bind the same module-global
    name."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            class Demo:
                helper = "local"

                def method(self) -> str:
                    return self.helper
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import helper
        from owui_ext.shared.shared_b import Demo

        helper()
        Demo()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Demo:" in output


def test_global_walrus_in_class_base_collides(tmp_path: Path) -> None:
    """A walrus in a class's base list evaluates in the enclosing
    function's scope. With ``global`` there, it writes a module global."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            class _Mix:
                pass


            def outer() -> None:
                global helper

                class Demo((helper := _Mix)):
                    pass

                Demo
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import outer
        from owui_ext.shared.shared_b import helper

        outer()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_global_write_in_nested_function_collides(tmp_path: Path) -> None:
    """Nested ``def`` declaring ``global`` is also tracked."""

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            def outer() -> None:
                def clobber() -> None:
                    global helper
                    helper = lambda: "a"

                clobber()
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import outer
        from owui_ext.shared.shared_b import helper

        outer()
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="(?i)helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_method_default_walrus_inside_class_body(tmp_path: Path) -> None:
    """A walrus inside a method default at class body scope binds in the
    class namespace, not module globals -- so the build is allowed."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""


        class Demo:
            def method(self, x: int = (cached := 1)) -> int:
                return x + cached


        Demo()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "(cached := 1)" in output


def test_allows_nested_function_default_walrus(tmp_path: Path) -> None:
    """A walrus in a nested function's default binds in the outer
    function's local scope, not module globals."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""


        def outer() -> int:
            def inner(x: int = (cached := 1)) -> int:
                return x + cached

            return inner()


        outer()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "(cached := 1)" in output


def test_allows_walrus_in_function_body_without_global(tmp_path: Path) -> None:
    """Walrus that does not interact with ``global`` stays local and is allowed."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""


        def main() -> int:
            if (helper := 1) > 0:
                return helper
            return 0


        main()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "(helper := 1)" in output


def test_allows_walrus_inside_function_body(tmp_path: Path) -> None:
    """Walrus inside a function body stays local and is allowed."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def main() -> int:
            if (helper := 1) > 0:
                return helper
            return 0


        main()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "(helper := 1)" in output


def test_rejects_module_scope_match_statement(tmp_path: Path) -> None:
    """``match`` statements at module scope are refused.

    Capture patterns (``case x:``, ``case [a, b]:``, etc.) bind names into
    module globals, but the collision detector does not enumerate them.
    Refuse rather than risk a silent shadow of a dep ``def``.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        value = 1
        match value:
            case 1:
                result = "one"
            case _:
                result = "other"

        print(result)
        ''',
    )
    with pytest.raises(BuildError, match="``match``"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_match_inside_function_body(tmp_path: Path) -> None:
    """``match`` inside a function body stays local and is allowed."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def classify(value: int) -> str:
            match value:
                case 1:
                    return "one"
                case _:
                    return "other"


        classify(1)
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "match value:" in output


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
    """Module docstring and ``from __future__`` lines may precede regular imports.

    ``annotations`` is refused (F5), so this test uses ``division``
    -- a no-op flag in modern Python -- to exercise the
    docstring-then-future-then-imports parsing.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        "from __future__ import division\n\n\ndef helper(): return 1\n",
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from __future__ import division

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


def test_rejects_builtins_rebinding_in_dep_that_would_leak_to_other_dep(
    tmp_path: Path,
) -> None:
    """A dep that rebinds ``__builtins__`` at module scope must be
    refused: in source, ``shared_a.__builtins__`` is local to that
    module and cannot affect ``shared_b``'s ``len()`` call. After
    inlining, both bodies share one module namespace, so the
    ``__builtins__`` override silently changes ``len(...)`` to return
    99 for ``shared_b`` too -- a runtime semantic divergence the
    general name-collision detector cannot catch (it tracks named
    definitions, not the implicit builtin lookup chain).

    This is the exact attack scenario flagged by the adversarial
    review: rebinding ``__builtins__`` is a backdoor around the
    collision detector.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            VALUE = 1
            __builtins__ = {"len": lambda x: 99}
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def size(x: object) -> int:
                return len(x)
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import VALUE
        from owui_ext.shared.shared_b import size

        size([VALUE])
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_builtins_rebinding_in_target(tmp_path: Path) -> None:
    """The target itself may not rebind ``__builtins__`` at module scope.

    The override would apply during dep body execution too, since deps
    run inside the target's compilation unit after inlining.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        __builtins__ = {"len": lambda x: 0}

        print(len([1, 2, 3]))
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_builtins_rebinding_via_annotated_assign(
    tmp_path: Path,
) -> None:
    """``__builtins__: dict = {...}`` at module scope is also refused.

    AnnAssign rebinding has the same module-namespace effect as a plain
    Assign and must be flagged by the same pass.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            __builtins__: dict = {"len": lambda x: 0}


            def helper() -> int:
                return 1
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
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_builtins_rebinding_via_global_in_function(
    tmp_path: Path,
) -> None:
    """``global __builtins__; __builtins__ = ...`` from inside a
    function still rebinds at module scope and must be refused.

    The rebind only fires when ``configure()`` is called, but in the
    inlined output any caller of ``configure`` would corrupt builtin
    resolution for every other inlined source. Refuse statically rather
    than rely on the function never being invoked.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            def configure() -> None:
                global __builtins__
                __builtins__ = {"len": lambda x: 0}


            def helper() -> int:
                return 1
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
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_builtins_rebinding_via_import_alias(
    tmp_path: Path,
) -> None:
    """``import foo as __builtins__`` aliases the binding into
    ``__builtins__`` at module scope and must be refused.

    The check covers any module-scope statement that lands on the name,
    not just literal ``__builtins__ = ...`` syntax.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            import os as __builtins__


            def helper() -> int:
                return 1
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
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_function_local_builtins_assignment_without_global(
    tmp_path: Path,
) -> None:
    """A function-local ``__builtins__ = ...`` without ``global`` is
    allowed: it binds a local-only variable that never reaches module
    globals, so the inlined output still sees the real builtins.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            def helper() -> int:
                __builtins__ = {"len": lambda x: 0}
                return len(__builtins__)
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
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "def helper()" in output


def test_rejects_dynamic_globals_subscript_builtins_rebinding_in_dep(
    tmp_path: Path,
) -> None:
    """``globals()["__builtins__"] = X`` at module scope of a dep is
    refused. This is the dynamic equivalent of ``__builtins__ = X``:
    once the dep body executes during inlined import, the merged
    module's ``__builtins__`` is replaced and every other dep's builtin
    lookup picks up the fake.

    The static-binding pass cannot catch this -- the AST shows a
    ``Subscript`` write on a ``Call`` value, not a ``Name`` target.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            VALUE = 1
            globals()["__builtins__"] = {"len": lambda x: 99}
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def size(x: object) -> int:
                return len(x)
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import VALUE
        from owui_ext.shared.shared_b import size

        size([VALUE])
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dynamic_globals_setitem_builtins_rebinding_in_dep(
    tmp_path: Path,
) -> None:
    """``globals().__setitem__("__builtins__", X)`` is the explicit-
    method equivalent of subscript-store and is refused for the same
    reason: it produces the same merged-module corruption when run.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            globals().__setitem__("__builtins__", {"len": lambda x: 0})


            def helper() -> int:
                return 1
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
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dynamic_globals_subscript_builtins_rebinding_in_target(
    tmp_path: Path,
) -> None:
    """The target is held to the same standard as deps: writes to
    ``globals()["__builtins__"]`` in target body are refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        globals()["__builtins__"] = {"len": lambda x: 0}

        print(len([1, 2, 3]))
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dynamic_globals_builtins_rebinding_inside_function(
    tmp_path: Path,
) -> None:
    """``globals()`` returns the module's globals regardless of call
    site, so a write inside a function body still corrupts merged
    builtins once that function runs. Refuse statically rather than
    rely on the function never being invoked.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            def configure() -> None:
                globals()["__builtins__"] = {"len": lambda x: 0}


            def helper() -> int:
                return 1
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
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_dynamic_globals_write_to_unrelated_key(
    tmp_path: Path,
) -> None:
    """The check matches only the literal ``"__builtins__"`` key.
    Writing other keys via ``globals()[...]`` is a normal Python
    pattern (e.g., dynamic registration) and must remain allowed.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            globals()["registered"] = True


            def helper() -> bool:
                return registered
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
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "globals()[\"registered\"]" in output


def test_allows_builtins_attribute_mutation(tmp_path: Path) -> None:
    """``__builtins__["len"] = ...`` mutates the (process-wide)
    builtins dict; that is identical in source and inlined form, so
    the inliner does not refuse it.

    Only *rebinding* ``__builtins__`` is the inlining-specific hazard
    -- mutation already affects every module in the process pre-inline,
    so flagging it would be over-reach without a corresponding
    semantic-change argument.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            __builtins__["len_alias"] = len


            def helper() -> int:
                return 1
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
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "__builtins__[\"len_alias\"]" in output


def test_rejects_globals_update_builtins_rebinding(tmp_path: Path) -> None:
    """``globals().update({"__builtins__": ...})`` with a literal-keyed
    dict is reachable by the static analyzer (we can extract every
    key), but the resulting write to ``__builtins__`` is refused by
    the same reserved-name check that catches ``__builtins__ = X``.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            globals().update({"__builtins__": {"len": lambda x: 0}})


            def helper() -> int:
                return 1
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
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_update_with_runtime_key_set(
    tmp_path: Path,
) -> None:
    """``globals().update(some_var)`` cannot be reduced to a known set
    of bindings statically, so it is refused outright. Any inlined
    source it would mutate could collide silently with another source's
    bindings without the collision detector seeing it.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            CONFIG = {"__builtins__": {"len": lambda x: 0}}
            globals().update(CONFIG)


            def helper() -> int:
                return 1
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
    with pytest.raises(BuildError, match=r"globals\(\)\.update"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_update_with_kwargs(tmp_path: Path) -> None:
    """``globals().update(**kw)`` likewise hides its key set, so it is
    refused outright by the same pass.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        OVERRIDES = {"X": 1}
        globals().update(**OVERRIDES)
        ''',
    )
    with pytest.raises(BuildError, match=r"globals\(\)\.update"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_update_with_non_literal_dict_key(
    tmp_path: Path,
) -> None:
    """Mixed literal/non-literal dict keys also defeat static analysis
    of which names get written, so the call is refused outright.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def _key():
            return "X"

        globals().update({"OK": 1, _key(): 2})
        ''',
    )
    with pytest.raises(BuildError, match=r"globals\(\)\.update"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_pop_call(tmp_path: Path) -> None:
    """``globals().pop(...)`` is refused outright; whether the literal
    key collides with another inlined source or not, the call's
    side effects on the merged namespace are not worth the analysis
    burden of a single-call form.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        X = 1
        globals().pop("X", None)
        ''',
    )
    with pytest.raises(BuildError, match=r"globals\(\)\.pop"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_clear_call(tmp_path: Path) -> None:
    """``globals().clear()`` would erase every other inlined source's
    bindings; refuse outright.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        globals().clear()
        ''',
    )
    with pytest.raises(BuildError, match=r"globals\(\)\.clear"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_setdefault_call(tmp_path: Path) -> None:
    """``globals().setdefault(...)`` is also refused; even with a
    literal key, the conditional-write semantics are unusual enough
    to refuse rather than special-case in the collision detector.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        globals().setdefault("X", 1)
        ''',
    )
    with pytest.raises(BuildError, match=r"globals\(\)\.setdefault"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dynamic_globals_write_collision_between_deps(
    tmp_path: Path,
) -> None:
    """A dep that writes ``globals()['helper'] = X`` and another dep
    that defines ``def helper()`` would silently collide post-inline,
    even though the original separated modules each owned their own
    ``helper``. The general collision detector now ingests dynamic
    literal-key globals writes and flags this case at build time.

    This is the broader bypass that ``_verify_no_builtins_rebinding``
    alone could not catch: any name, not just ``__builtins__``, is
    routed through one shared namespace post-inline.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            SHARED_A_MARKER = True


            def _bootstrap() -> None:
                globals()["helper"] = lambda: "from_a"


            _bootstrap()
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "from_b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import SHARED_A_MARKER
        from owui_ext.shared.shared_b import helper

        SHARED_A_MARKER
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dynamic_globals_setitem_collision_between_deps(
    tmp_path: Path,
) -> None:
    """The same collision check fires for the explicit-method form
    ``globals().__setitem__(literal, X)``.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            SHARED_A_MARKER = True
            globals().__setitem__("helper", lambda: "from_a")
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "from_b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import SHARED_A_MARKER
        from owui_ext.shared.shared_b import helper

        SHARED_A_MARKER
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dynamic_globals_update_collision_between_deps(
    tmp_path: Path,
) -> None:
    """``globals().update({"helper": ...})`` with a literal key is also
    ingested as a binding by the collision detector.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            SHARED_A_MARKER = True
            globals().update({"helper": lambda: "from_a"})
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "from_b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import SHARED_A_MARKER
        from owui_ext.shared.shared_b import helper

        SHARED_A_MARKER
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dynamic_globals_delete_destroys_other_dep_definition(
    tmp_path: Path,
) -> None:
    """``del globals()[literal]`` (and ``globals().__delitem__(literal)``)
    is treated as a destructive module-scope write: erasing a name that
    another inlined source bound is a silent semantic change vs. the
    source form, where the dep's delete only affected its own globals.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            SHARED_A_MARKER = True


            def _cleanup() -> None:
                del globals()["helper"]


            _cleanup()
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> int:
                return 1
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import SHARED_A_MARKER
        from owui_ext.shared.shared_b import helper

        SHARED_A_MARKER
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_subscript_with_non_literal_key(
    tmp_path: Path,
) -> None:
    """``KEY = "helper"; globals()[KEY] = X`` writes to module globals
    without exposing the key to static analysis. Refuse outright; the
    collision detector cannot match it against another inlined source's
    bindings, and ``__builtins__`` rebinding hides behind it equally
    well.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        KEY = "helper"
        globals()[KEY] = 1
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_subscript_delete_with_non_literal_key(
    tmp_path: Path,
) -> None:
    """``del globals()[KEY]`` with a non-literal key likewise hides
    its target from the destructive-write detector.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        helper = 1
        KEY = "helper"
        del globals()[KEY]
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_setitem_with_non_literal_first_arg(
    tmp_path: Path,
) -> None:
    """``globals().__setitem__(KEY, X)`` with a non-literal first arg
    is the explicit-method equivalent of subscript-store with a
    runtime key. Refuse outright.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        KEY = "helper"
        globals().__setitem__(KEY, 1)
        ''',
    )
    with pytest.raises(BuildError, match=r"globals\(\)\.__setitem__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_delitem_with_non_literal_first_arg(
    tmp_path: Path,
) -> None:
    """``globals().__delitem__(KEY)`` with a non-literal first arg
    is similarly refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        helper = 1
        KEY = "helper"
        globals().__delitem__(KEY)
        ''',
    )
    with pytest.raises(BuildError, match=r"globals\(\)\.__delitem__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_init_call(tmp_path: Path) -> None:
    """``globals().__init__(...)`` re-initializes the merged-module's
    globals dict in place; refuse outright.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        globals().__init__({"X": 1})
        ''',
    )
    with pytest.raises(BuildError, match=r"globals\(\)\.__init__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_ior_call(tmp_path: Path) -> None:
    """``globals().__ior__({...})`` is the in-place ``|=`` merge from
    PEP 584. The mapping argument can have arbitrary runtime-determined
    keys, so refuse outright.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        globals().__ior__({"X": 1})
        ''',
    )
    with pytest.raises(BuildError, match=r"globals\(\)\.__ior__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_annassign_globals_subscript_builtins_rebinding(
    tmp_path: Path,
) -> None:
    """``globals()["__builtins__"]: object = ...`` is an AnnAssign with
    a value, which executes the same subscript store as
    ``globals()["__builtins__"] = ...``. The reserved-name check fires.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        globals()["__builtins__"]: object = {"len": lambda x: 0}
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_annassign_globals_subscript_collision_between_deps(
    tmp_path: Path,
) -> None:
    """An AnnAssign-form literal-key write (``globals()["helper"]:
    object = X``) collides with another dep's ``def helper`` just
    like the plain Assign form does.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            SHARED_A_MARKER = True
            globals()["helper"]: object = lambda: "from_a"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "from_b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import SHARED_A_MARKER
        from owui_ext.shared.shared_b import helper

        SHARED_A_MARKER
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_annassign_globals_subscript_with_non_literal_key(
    tmp_path: Path,
) -> None:
    """An AnnAssign-form non-literal-key write is also refused
    outright.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        KEY = "helper"
        globals()[KEY]: object = 1
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_annassign_globals_subscript_without_value(
    tmp_path: Path,
) -> None:
    """``globals()["x"]: int`` (annotation-only, no value) does not
    execute a subscript store, so it isn't a write. Allow it.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            globals()["unique_anno_only"]: int


            def helper() -> int:
                return 1
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
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "globals()[\"unique_anno_only\"]" in output


def test_rejects_globals_subscript_in_tuple_unpacking_target(
    tmp_path: Path,
) -> None:
    """``globals()[KEY], other = (1, 2)`` is a legal Python assignment
    target whose Subscript still carries ``ctx == Store``. Previous
    versions of the verifier only inspected top-level Assign targets
    and missed nested Tuple unpacking, leaving non-literal-key writes
    unchecked. The scope-aware visitor visits every Subscript node
    once, regardless of how deeply it is nested in the target.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        KEY = "helper"
        globals()[KEY], other = (1, 2)
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_subscript_in_for_target(tmp_path: Path) -> None:
    """``for globals()[KEY] in items: ...`` likewise stores into
    module globals via a target with a non-literal key.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        KEY = "helper"
        for globals()[KEY] in [1, 2]:
            pass
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_subscript_in_with_target(tmp_path: Path) -> None:
    """``with cm() as globals()[KEY]:`` stores the context manager's
    bound value into module globals at a non-literal key. Refuse.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from contextlib import nullcontext

        KEY = "helper"
        with nullcontext(1) as globals()[KEY]:
            pass
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_subscript_in_compound_delete_target(
    tmp_path: Path,
) -> None:
    """``del (globals()[KEY],)`` deletes module globals via a Subscript
    nested in a Tuple delete target. The visitor's per-Subscript walk
    catches this just like the simpler ``del globals()[KEY]`` form.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        helper = 1
        KEY = "helper"
        del (globals()[KEY],)
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_subscript_collision_via_tuple_unpacking(
    tmp_path: Path,
) -> None:
    """A literal-key write nested in a Tuple unpacking target also
    feeds the collision detector: ``globals()["helper"], other = (a,
    b)`` registers ``helper`` as a binding and collides with another
    dep's ``def helper``.
    """

    _write(
        tmp_path / "src/owui_ext/shared/shared_a.py",
        textwrap.dedent(
            """
            SHARED_A_MARKER = True


            def _bootstrap() -> None:
                globals()["helper"], _other = (lambda: "from_a", None)


            _bootstrap()
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/shared_b.py",
        textwrap.dedent(
            """
            def helper() -> str:
                return "from_b"
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.shared_a import SHARED_A_MARKER
        from owui_ext.shared.shared_b import helper

        SHARED_A_MARKER
        helper()
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_function_local_vars_subscript_write(
    tmp_path: Path,
) -> None:
    """``vars()`` inside a function body returns the *local* frame's
    mapping, not module globals -- writes to it cannot corrupt the
    merged module's namespace, so they must remain allowed.

    (CPython does not let you usefully mutate a function's locals
    through ``vars()`` anyway, but the inliner's correctness argument
    only requires that we don't *falsely* refuse the call.)
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            def helper() -> None:
                key = "x"
                vars()[key] = 1
                vars().update({"y": 2})


            VALUE = 0
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import VALUE, helper

        helper()
        VALUE
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "vars()" in output


def test_allows_class_body_vars_call(tmp_path: Path) -> None:
    """``vars()`` inside a class body returns the class namespace
    being constructed, not module globals. Allow.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            class Container:
                vars().update({"member": 1})


            VALUE = 0
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import VALUE, Container

        Container
        VALUE
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "class Container" in output


def test_allows_function_local_vars_pop_call(tmp_path: Path) -> None:
    """``vars().pop(...)`` inside a function body affects the local
    frame, not module globals; do not refuse.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            def helper() -> None:
                tmp = 1
                vars().pop("tmp", None)


            VALUE = 0
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import VALUE, helper

        helper()
        VALUE
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "vars().pop" in output


def test_rejects_module_scope_vars_write(tmp_path: Path) -> None:
    """At module scope, ``vars()`` IS the module's globals dict
    (Python documents the equivalence), so mutations there must still
    be refused. The scope-aware visitor only relaxes ``vars()``
    inside function / class / lambda / comprehension bodies.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        vars()["__builtins__"] = {"len": lambda x: 0}
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_dynamic_globals_write_with_no_collision(
    tmp_path: Path,
) -> None:
    """A dep that writes a unique literal key via ``globals()[...]``
    and is the sole binder of that name is allowed -- the collision
    detector adds the name to its definition map but finds no other
    source binding it.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            globals()["unique_registry"] = {}


            def helper() -> dict:
                return unique_registry
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
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "globals()[\"unique_registry\"]" in output


# ---------------------------------------------------------------------------
# Scope-boundary bypass regression tests.
#
# Python evaluates a number of expressions at the *enclosing* scope when a
# def/class/lambda/comprehension statement runs: decorators, default values,
# argument annotations, return annotations, class bases/keywords, and the
# *outermost* comprehension iterable. A scope-unaware visitor that bumps
# ``depth`` for the entire FunctionDef/ClassDef/Lambda/Comprehension would
# see ``vars()`` calls in those header positions as in-scope locals and
# wrongly let them through. The fixed visitor visits each header expression
# at the enclosing depth, so a header-position ``vars().__setitem__(
# "__builtins__", ...)`` is still flagged as a module-globals mutation.
# ---------------------------------------------------------------------------


def test_rejects_vars_setitem_in_function_default(tmp_path: Path) -> None:
    """``def f(x=vars().__setitem__("__builtins__", ...))`` runs the
    default expression in the enclosing (module) scope when the def
    statement executes, so ``vars()`` resolves to module globals.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(x=vars().__setitem__("__builtins__", {"len": lambda x: 0})):
            return x
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_async_function_default(
    tmp_path: Path,
) -> None:
    """Same enclosing-scope evaluation rule applies to async def."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        async def f(x=vars().__setitem__("__builtins__", {})):
            return x
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_function_kw_default(
    tmp_path: Path,
) -> None:
    """Keyword-only argument defaults also evaluate in the enclosing
    scope.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(*, x=vars().__setitem__("__builtins__", {})):
            return x
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_function_annotation(
    tmp_path: Path,
) -> None:
    """Argument annotations evaluate at def-time (without
    ``from __future__ import annotations``), in the enclosing scope.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(x: vars().__setitem__("__builtins__", {})) -> None:
            return None
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_function_return_annotation(
    tmp_path: Path,
) -> None:
    """The return annotation also evaluates at def-time."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f() -> vars().__setitem__("__builtins__", {}):
            return None
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_decorator(tmp_path: Path) -> None:
    """Decorator expressions evaluate when the def runs, in the
    enclosing scope.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        @vars().__setitem__("__builtins__", {})
        def f():
            return None
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_class_base(tmp_path: Path) -> None:
    """Base classes evaluate when the class statement runs, in the
    enclosing scope.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        class C(vars().__setitem__("__builtins__", {}) or object):
            pass
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_class_decorator(tmp_path: Path) -> None:
    """Class decorators evaluate in the enclosing scope."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        @vars().__setitem__("__builtins__", {})
        class C:
            pass
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_class_keyword(tmp_path: Path) -> None:
    """Class keyword arguments (``metaclass=`` etc.) evaluate in the
    enclosing scope.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        class C(metaclass=type, x=vars().__setitem__("__builtins__", {})):
            pass
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_lambda_default(tmp_path: Path) -> None:
    """Lambda defaults evaluate when the lambda expression is
    constructed, in the enclosing scope.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        f = lambda x=vars().__setitem__("__builtins__", {}): x
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_outermost_comp_iter(
    tmp_path: Path,
) -> None:
    """The outermost generator's iterable in a comprehension is
    evaluated in the enclosing scope; only the rest of the comprehension
    runs in its own implicit-function scope.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        result = [
            x
            for x in [vars().__setitem__("__builtins__", {})]
        ]
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_outermost_genexp_iter(
    tmp_path: Path,
) -> None:
    """Same rule for generator expressions."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        gen = (
            x
            for x in [vars().__setitem__("__builtins__", {})]
        )
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_setitem_in_outermost_dictcomp_iter(
    tmp_path: Path,
) -> None:
    """Same rule for dict comprehensions."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        d = {
            x: x
            for x in [vars().__setitem__("__builtins__", {})]
        }
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_vars_setitem_in_inner_comp_iter(tmp_path: Path) -> None:
    """The *inner* generator's iterable runs inside the comprehension
    scope, where ``vars()`` returns the implicit-function locals (not
    module globals). A literal-key write there does not mutate module
    globals and must NOT be refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        # Outer iter is a plain list, inner iter contains the vars()
        # call. Inside the comprehension scope, ``vars()`` is the
        # implicit function's locals, so writing to its dict has no
        # effect on the merged module's globals.
        result = [
            (a, b)
            for a in [1, 2]
            for b in [vars().__setitem__("name", "irrelevant")]
        ]
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "vars()" in output


def test_rejects_globals_setitem_in_function_default(
    tmp_path: Path,
) -> None:
    """``globals()`` always returns module globals regardless of scope,
    so the same bypass via a function default with ``globals()`` is
    refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(x=globals().__setitem__("__builtins__", {})):
            return x
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


# ---------------------------------------------------------------------------
# locals() bypass regression tests.
#
# At module scope, Python documents ``locals() is globals()`` -- so any
# mutation of ``locals()`` at the top level lands in the merged module's
# globals dict. The visitor treats ``locals()`` with the same scope-aware
# rule as ``vars()``: matched at depth==0, ignored inside function /
# class / lambda / comprehension bodies (where it returns the local
# frame, not module globals).
# ---------------------------------------------------------------------------


def test_rejects_module_scope_locals_subscript_builtins(
    tmp_path: Path,
) -> None:
    """``locals()['__builtins__'] = X`` at module scope rebinds the
    merged module's ``__builtins__`` reference and must be refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        locals()["__builtins__"] = {"len": lambda x: 0}
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_scope_locals_setitem_builtins(
    tmp_path: Path,
) -> None:
    """``locals().__setitem__('__builtins__', X)`` is the explicit-
    method equivalent of the subscript form and is refused for the
    same reason.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        locals().__setitem__("__builtins__", {"len": lambda x: 0})
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_scope_locals_update_builtins(
    tmp_path: Path,
) -> None:
    """``locals().update({...})`` with a literal-keyed dict at module
    scope is the same mutation form as ``globals().update(...)`` and
    must be refused when ``__builtins__`` is among the keys.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        locals().update({"__builtins__": {"len": lambda x: 0}})
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_locals_collision_between_deps(tmp_path: Path) -> None:
    """``locals()['name'] = X`` at module scope of one dep collides
    with another dep's binding of the same name. Codex round 6
    finding 1 PoC: shared_a uses ``locals()`` at module scope to
    smuggle a value, shared_b independently binds the same name.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            locals()["shared_value"] = 123
            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            shared_value = 456
            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER

        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="shared_value"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_function_local_locals_call(tmp_path: Path) -> None:
    """Inside a function body ``locals()`` returns the local frame's
    mapping, NOT the merged module's globals. Writing to it has no
    effect on module globals, so legitimate function-local uses must
    not be refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(x):
            # Mutating locals() inside a function in CPython does
            # nothing observable, but it is a legitimate diagnostic /
            # debugging pattern that we must not refuse.
            locals()["debug_value"] = x
            return x

        RESULT = f(1)
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "locals()" in output


def test_allows_class_body_locals_call(tmp_path: Path) -> None:
    """Inside a class body ``locals()`` returns the class namespace
    (the dict that becomes the class's ``__dict__``), not module
    globals. Writing to it must not be refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        class C:
            locals()["dynamic_attr"] = 42

        VALUE = C.dynamic_attr
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "C.dynamic_attr" in output


def test_rejects_locals_setitem_in_function_default(
    tmp_path: Path,
) -> None:
    """Function defaults evaluate in the enclosing scope. At module
    scope that means ``locals()`` is module globals.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(x=locals().__setitem__("__builtins__", {})):
            return x
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_locals_unknown_mutator(tmp_path: Path) -> None:
    """``locals().pop(...)`` at module scope falls under the same
    refusal as ``globals().pop(...)`` and ``vars().pop(...)``.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        locals().pop("anything", None)
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


# ---------------------------------------------------------------------------
# dict.METHOD(globals(), ...) descriptor-form regression tests.
#
# A bound-method call ``globals().METHOD(*args)`` and the equivalent
# descriptor call ``dict.METHOD(globals(), *args)`` produce the same
# mutation on the merged module's globals dict. The visitor normalizes
# both forms so all downstream refusals fire identically.
# ---------------------------------------------------------------------------


def test_rejects_dict_setitem_descriptor_builtins(
    tmp_path: Path,
) -> None:
    """``dict.__setitem__(globals(), '__builtins__', X)`` is the dict-
    descriptor form of ``globals()['__builtins__'] = X`` and must be
    refused for the same reason. Codex round 6 finding 2 PoC.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        dict.__setitem__(globals(), "__builtins__", {"len": lambda x: 0})
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_update_descriptor_builtins(tmp_path: Path) -> None:
    """``dict.update(globals(), {'__builtins__': X})`` is the dict-
    descriptor form of ``globals().update({'__builtins__': X})``.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        dict.update(globals(), {"__builtins__": {"len": lambda x: 0}})
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_delitem_descriptor(tmp_path: Path) -> None:
    """``dict.__delitem__(globals(), 'name')`` removes ``name`` from
    module globals and is treated like ``del globals()['name']`` for
    destructive-write detection.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            shared_thing = "value"
            SHARED_A_MARKER = True


            def helper():
                return shared_thing
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_A_MARKER, helper

        dict.__delitem__(globals(), "shared_thing")
        RESULT = helper()
        ''',
    )
    with pytest.raises(BuildError, match="shared_thing"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_pop_descriptor(tmp_path: Path) -> None:
    """``dict.pop(globals(), ...)`` is refused for the same reason as
    ``globals().pop(...)``.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        dict.pop(globals(), "anything", None)
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_clear_descriptor(tmp_path: Path) -> None:
    """``dict.clear(globals())`` empties the merged globals dict."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        dict.clear(globals())
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_setitem_descriptor_with_non_literal_key(
    tmp_path: Path,
) -> None:
    """Non-literal first-positional argument means we cannot determine
    which name is being written, so refuse outright.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        KEY = "some_name"
        dict.__setitem__(globals(), KEY, 1)
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_setitem_descriptor_collision_between_deps(
    tmp_path: Path,
) -> None:
    """Cross-dep collision detection includes descriptor-form writes."""

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            dict.__setitem__(globals(), "shared_value", 123)
            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            shared_value = 456
            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER

        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="shared_value"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_update_descriptor_collision_between_deps(
    tmp_path: Path,
) -> None:
    """``dict.update(globals(), {literal: X})`` collides with another
    dep's binding of the same literal key.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            dict.update(globals(), {"shared_value": 123})
            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            shared_value = 456
            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER

        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="shared_value"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_setitem_descriptor_with_vars_at_module_scope(
    tmp_path: Path,
) -> None:
    """The descriptor form pairs with any module-globals call, not
    just ``globals()``. ``dict.__setitem__(vars(), ...)`` at module
    scope (where ``vars() is globals()``) must be refused too.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        dict.__setitem__(vars(), "__builtins__", {})
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_setitem_descriptor_with_locals_at_module_scope(
    tmp_path: Path,
) -> None:
    """And with ``locals()`` at module scope."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        dict.__setitem__(locals(), "__builtins__", {})
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


# ---------------------------------------------------------------------------
# globals()[K] read-side collision regression tests.
#
# The collision detector ingests literal-key reads (subscript Load,
# ``__getitem__``, ``get``, ``__contains__``) as cross-module references,
# so a dep that reads a name without binding it locally is flagged when
# another inlined source happens to define that name. Whole-dict
# observations (``keys`` / ``items`` / ``values`` / ``copy`` /
# ``__iter__`` / ``__len__``) are refused outright.
# ---------------------------------------------------------------------------


def test_rejects_globals_subscript_read_resolved_by_other_dep(
    tmp_path: Path,
) -> None:
    """Codex round 6 finding 3 PoC: shared_b reads ``globals()["helper"]``
    without binding ``helper`` itself; shared_a defines ``helper``.
    Pre-merge ``shared_b.run()`` would raise ``KeyError`` because
    shared_b's globals dict has no ``helper`` key. Post-merge it
    silently picks up shared_a's ``helper``. Refuse the build.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            def helper():
                return "from_a"


            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            def run():
                return globals()["helper"]()


            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER, run

        RESULT = run()
        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_getitem_call_resolved_by_other_dep(
    tmp_path: Path,
) -> None:
    """Same finding via the explicit ``__getitem__`` method form."""

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            def helper():
                return "from_a"


            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            def run():
                return globals().__getitem__("helper")()


            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER, run

        RESULT = run()
        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_get_call_resolved_by_other_dep(
    tmp_path: Path,
) -> None:
    """Same finding via the ``get`` method form."""

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            def helper():
                return "from_a"


            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            def run():
                return globals().get("helper")()


            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER, run

        RESULT = run()
        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_contains_resolved_by_other_dep(
    tmp_path: Path,
) -> None:
    """``"helper" in globals()`` lowers to ``globals().__contains__("helper")``;
    the post-merge result depends on whether another dep binds
    ``helper``, so it is treated as a cross-module reference.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            helper = lambda: "from_a"


            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            def has_helper():
                return "helper" in globals()


            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER, has_helper

        RESULT = has_helper()
        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_getitem_descriptor_resolved_by_other_dep(
    tmp_path: Path,
) -> None:
    """``dict.__getitem__(globals(), "helper")`` is the descriptor
    equivalent of the bound-method form, ingested as a reference too.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            def helper():
                return "from_a"


            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            def run():
                return dict.__getitem__(globals(), "helper")()


            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER, run

        RESULT = run()
        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_subscript_read_with_non_literal_key(
    tmp_path: Path,
) -> None:
    """A non-literal-key globals read cannot be checked against any
    binding set, so refuse outright.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        KEY = "name"
        VALUE = globals()[KEY]
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_keys_call(tmp_path: Path) -> None:
    """``globals().keys()`` observes the entire merged-globals key set,
    which silently grows when other inlined sources add bindings.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        ALL_NAMES = list(globals().keys())
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_items_call(tmp_path: Path) -> None:
    """Same for ``items``."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        ALL_PAIRS = list(globals().items())
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_values_call(tmp_path: Path) -> None:
    """And ``values``."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        ALL_VALUES = list(globals().values())
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_copy_call(tmp_path: Path) -> None:
    """And ``copy``."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        SNAPSHOT = globals().copy()
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_function_local_globals_subscript_read(
    tmp_path: Path,
) -> None:
    """A literal-key globals read whose name *is* bound by the same
    source (e.g. via a regular ``def`` in the same dep) does not need
    to resolve to another source post-merge, so it must NOT be refused.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        textwrap.dedent(
            """
            def helper():
                return "self_bound"


            def run():
                return globals()["helper"]()


            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_A_MARKER, run

        RESULT = run()
        FLAGS = (SHARED_A_MARKER,)
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "globals()[\"helper\"]" in output


def test_allows_function_local_vars_subscript_read(
    tmp_path: Path,
) -> None:
    """``vars()`` inside a function returns the local frame, which is
    not the merged module's globals -- so a literal-key read there
    does not need cross-source collision checking.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f():
            x = 1
            return vars()["x"]

        RESULT = f()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "vars()[\"x\"]" in output


def test_rejects_globals_get_with_non_literal_key(tmp_path: Path) -> None:
    """``globals().get(<non-literal>)`` is refused too."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        KEY = "name"
        VALUE = globals().get(KEY)
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_contains_with_non_literal_key(
    tmp_path: Path,
) -> None:
    """``globals().__contains__(<non-literal>)`` is refused too."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        KEY = "name"
        FLAG = globals().__contains__(KEY)
        ''',
    )
    with pytest.raises(BuildError, match="non-literal"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


# ---------------------------------------------------------------------------
# Unbounded-context regression tests.
#
# A ``globals()/vars()/locals()`` call (in module-globals scope) outside
# of the four permitted contexts -- ``Subscript.value``,
# ``Attribute.value`` of an immediately-invoked method,
# ``dict.METHOD(globals(), ...)`` first arg, and ``Compare`` comparator
# with In/NotIn -- is refused outright by
# ``_verify_no_unbounded_globals_access`` because static reasoning about
# the call's effect is not possible there.
# ---------------------------------------------------------------------------


def test_rejects_globals_alias_assignment(tmp_path: Path) -> None:
    """Codex round 7 finding 1: ``g = globals()`` saves the merged
    globals dict into an alias, after which ``g[K] = V`` mutates module
    globals without ever syntactically touching ``globals()`` again --
    bypassing every per-shape mutation collector. Refuse the alias
    assignment outright.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        G = globals()
        G["__builtins__"] = {"len": lambda x: 0}
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_bound_method_alias(tmp_path: Path) -> None:
    """``setitem = globals().__setitem__`` saves the bound method into
    a name; later ``setitem(...)`` mutates module globals through the
    alias. Refuse storing the bound method by requiring that
    ``globals().METHOD`` is *immediately* invoked.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        setitem = globals().__setitem__
        setitem("__builtins__", {"len": lambda x: 0})
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_passed_to_dict_unpack(tmp_path: Path) -> None:
    """Codex round 7 finding 2 PoC: ``dict(**globals())`` reads every
    key of the merged globals dict, so adding a binding in another
    inlined source silently changes the result.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        SNAPSHOT = dict(**globals())
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_in_dict_literal_unpack(tmp_path: Path) -> None:
    """``{**globals()}`` is the same whole-dict observation in dict-
    display form.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        SNAPSHOT = {**globals()}
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_passed_to_list(tmp_path: Path) -> None:
    """``list(globals())`` materializes every key of the merged dict."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        ALL_NAMES = list(globals())
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_as_for_iter(tmp_path: Path) -> None:
    """``for k in globals():`` iterates every key, again whole-dict
    observation.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        NAMES = []
        for k in globals():
            NAMES.append(k)
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_in_compare_eq(tmp_path: Path) -> None:
    """``globals() == something`` is not the In/NotIn shape we permit;
    it observes the whole dict's contents.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        FLAG = globals() == {}
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_in_binop(tmp_path: Path) -> None:
    """``globals() | other`` invokes ``__or__`` on the merged dict, a
    whole-dict observation.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        SNAPSHOT = globals() | {}
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_passed_to_print(tmp_path: Path) -> None:
    """``print(globals())`` observes the whole dict (its repr depends
    on the merged key set).
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        print(globals())
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_locals_alias_assignment(tmp_path: Path) -> None:
    """At module scope ``locals() is globals()``, so the alias
    bypass applies to ``locals()`` too.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        L = locals()
        L["__builtins__"] = {}
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_alias_assignment(tmp_path: Path) -> None:
    """Same for ``vars()`` at module scope."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        V = vars()
        V["__builtins__"] = {}
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_alias_in_function_default(tmp_path: Path) -> None:
    """Function defaults evaluate in the enclosing scope, so an alias
    assignment there counts as a module-scope alias too. Specifically,
    a default expression that *evaluates* ``globals()`` outside of the
    permitted contexts should be refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(g=globals()):
            return g
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_function_local_globals_alias(tmp_path: Path) -> None:
    """Inside a function body ``globals()`` STILL returns the module's
    globals (not local frame), so an alias inside a function is just
    as dangerous as one at module scope. The visitor must refuse it,
    even though the alias assignment statement is inside a def.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f():
            g = globals()
            g["__builtins__"] = {}
        ''',
    )
    with pytest.raises(BuildError, match="unbounded|reason about|context"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_function_local_vars_alias(tmp_path: Path) -> None:
    """``vars()`` inside a function returns the local frame's mapping,
    not the merged module's globals. Saving it to an alias inside a
    function is harmless and must NOT be refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(x):
            v = vars()
            return v.get("x")

        RESULT = f(1)
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "vars()" in output


def test_allows_function_local_locals_alias(tmp_path: Path) -> None:
    """Same for ``locals()`` inside a function."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(x):
            ls = locals()
            return ls.get("x")

        RESULT = f(1)
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "locals()" in output


# ---------------------------------------------------------------------------
# __builtins__ destructive write regression tests.
#
# Codex round 7 finding 3: ``del __builtins__`` and the dynamic-delete
# equivalents are just as disruptive as rebinding because subsequent
# inlined sources that read ``globals()['__builtins__']`` would see a
# KeyError instead of whatever was there in source form.
# ---------------------------------------------------------------------------


def test_rejects_static_del_builtins(tmp_path: Path) -> None:
    """``del __builtins__`` at module scope removes the merged
    module's ``__builtins__`` reference and is refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        del __builtins__
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_except_as_builtins(tmp_path: Path) -> None:
    """``except E as __builtins__`` binds ``__builtins__`` for the
    handler then implicitly deletes it -- both effects refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        try:
            raise ValueError()
        except ValueError as __builtins__:
            pass
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_del_globals_subscript_builtins(tmp_path: Path) -> None:
    """``del globals()["__builtins__"]`` removes the merged module's
    ``__builtins__`` reference via the dynamic-delete path.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        del globals()["__builtins__"]
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_delitem_builtins(tmp_path: Path) -> None:
    """The explicit-method form ``globals().__delitem__("__builtins__")``."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        globals().__delitem__("__builtins__")
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_delitem_descriptor_builtins(tmp_path: Path) -> None:
    """The descriptor form ``dict.__delitem__(globals(), "__builtins__")``."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        dict.__delitem__(globals(), "__builtins__")
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_locals_delitem_builtins(tmp_path: Path) -> None:
    """``locals().__delitem__("__builtins__")`` at module scope is the
    same as ``globals().__delitem__("__builtins__")``.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        locals().__delitem__("__builtins__")
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


# ---------------------------------------------------------------------------
# Allowlist refusal regression tests (Codex round 8 finding 1).
#
# ``_verify_no_unknown_globals_mutation`` deny-by-default refuses any
# method on globals() that isn't one of ``__getitem__``, ``get``,
# ``__contains__``, ``__setitem__``, ``__delitem__``, ``update``. This
# closes the bypass where blacklist-only logic let through arbitrary
# whole-dict observation methods like ``__repr__``, ``__or__``,
# ``fromkeys``, ``__reduce__``, etc.
# ---------------------------------------------------------------------------


def test_rejects_globals_repr_call(tmp_path: Path) -> None:
    """``globals().__repr__()`` returns a string containing every
    binding name -- silently changes when other inlined sources add
    keys.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        SNAPSHOT = globals().__repr__()
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_or_call(tmp_path: Path) -> None:
    """``globals().__or__({...})`` builds a new dict from the merged
    globals plus the right operand -- whole-dict observation.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        MERGED = globals().__or__({})
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_fromkeys_globals(tmp_path: Path) -> None:
    """``dict.fromkeys(globals())`` reads every key of the merged
    globals dict.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        SNAPSHOT = dict.fromkeys(globals())
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_reduce_call(tmp_path: Path) -> None:
    """``globals().__reduce__()`` returns the dict and its full
    contents (for pickling).
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        SERIALIZED = globals().__reduce__()
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dict_keys_descriptor_globals(tmp_path: Path) -> None:
    """The descriptor form of ``keys`` is also refused."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        ALL_NAMES = list(dict.keys(globals()))
        ''',
    )
    with pytest.raises(BuildError, match="not allowed"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_str_via_method_call(tmp_path: Path) -> None:
    """``globals().__class__`` access through a method-style hop is
    also refused; any uninvoked attribute access on ``globals()`` is
    refused by the unbounded-context check (since the Attribute is
    not the func of a Call).
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        CLS = globals().__class__
        ''',
    )
    # Either the unbounded-context check (uninvoked Attribute) or the
    # allowlist check (if Attribute is invoked) must reject this.
    with pytest.raises(BuildError):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


# ---------------------------------------------------------------------------
# PEP 695 type alias regression tests (Codex round 8 finding 2).
#
# ``type X = ...`` (Python 3.12+) binds ``X`` at the enclosing scope. We
# integrate it into ``_collect_inlined_module_bindings``,
# ``_collect_global_writes``, and ``_first_builtins_store_line`` so it
# participates in name-collision detection and reserved-name
# enforcement just like ``X = ...``.
# ---------------------------------------------------------------------------


def test_rejects_type_alias_collision_between_deps(tmp_path: Path) -> None:
    """``type helper = int`` in one dep collides with ``def helper()``
    in another. PoC for Codex round 8 finding 2.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            type helper = int

            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            def helper():
                return "from_b"

            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER

        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_type_alias_builtins(tmp_path: Path) -> None:
    """``type __builtins__ = int`` is a top-level binding of
    ``__builtins__`` and is refused for the same reason as
    ``__builtins__ = X``.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        type __builtins__ = int
        ''',
    )
    with pytest.raises(BuildError, match="__builtins__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_type_alias_via_global_in_function(tmp_path: Path) -> None:
    """A nested ``global X; type X = ...`` writes to module globals
    and is recorded by ``_collect_global_writes`` so the collision
    detector picks it up.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            def clobber():
                global helper
                type helper = int


            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            def helper():
                return "from_b"


            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER

        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="helper"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


# ---------------------------------------------------------------------------
# __annotations__ regression tests (Codex round 8 finding 3).
#
# Module-scope ``AnnAssign`` implicitly populates ``__annotations__``,
# so every inlined source's annotations land in one shared dict after
# inlining. Any explicit reference to ``__annotations__`` at module
# scope is refused so cross-source observation cannot happen silently.
# ---------------------------------------------------------------------------


def test_rejects_annotations_dict_module_scope_read(tmp_path: Path) -> None:
    """``"x" in __annotations__`` at module scope reads the merged
    annotations dict, observing keys from every inlined source.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        x: int = 1
        FLAG = "x" in __annotations__
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_annotations_dict_module_scope_subscript(
    tmp_path: Path,
) -> None:
    """``__annotations__["x"]`` at module scope is the same hazard."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        x: int = 1
        T = __annotations__["x"]
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_annotations_dict_module_scope_assign(tmp_path: Path) -> None:
    """``__annotations__ = {...}`` replaces the auto-populated dict
    and corrupts every other source's view of its own annotations.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        __annotations__ = {}
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_del_annotations_dict(tmp_path: Path) -> None:
    """``del __annotations__`` removes the merged annotations dict."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        x: int = 1
        del __annotations__
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_subscript_annotations(tmp_path: Path) -> None:
    """``globals()["__annotations__"]`` is the dynamic-globals form
    of the same hazard.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        x: int = 1
        T = globals()["__annotations__"]
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_annotations_dep_collision(tmp_path: Path) -> None:
    """A dep that reads ``__annotations__`` while another binds it
    must be refused. PoC for Codex round 8 finding 3.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util_a.py",
        textwrap.dedent(
            """
            SECRET: int = 0


            SHARED_A_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/shared/util_b.py",
        textwrap.dedent(
            """
            FLAG = "SECRET" in __annotations__


            SHARED_B_MARKER = True
            """
        ).lstrip(),
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util_a import SHARED_A_MARKER
        from owui_ext.shared.util_b import SHARED_B_MARKER

        FLAGS = (SHARED_A_MARKER, SHARED_B_MARKER)
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_function_local_annotations_read(tmp_path: Path) -> None:
    """Codex round 9 finding 1: inside a function body, an unbound
    ``__annotations__`` reference resolves to module globals (Python
    treats it as any other free variable). After inlining, that means
    the function reads the *merged module's* annotations dict, not
    the function's own. So we must refuse explicit ``__annotations__``
    references inside function bodies too.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f():
            x: int = 1
            return "x" in __annotations__
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_module_scope_annassign(tmp_path: Path) -> None:
    """Plain module-scope ``AnnAssign`` (``x: int = 1``) does NOT
    explicitly reference ``__annotations__`` -- it just implicitly
    populates the dict. It must be allowed.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        x: int = 1
        y: str = "hello"
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "x: int = 1" in output


def test_rejects_def_annotations_at_module_scope(tmp_path: Path) -> None:
    """``def __annotations__()`` at module scope rebinds the implicit
    annotations dict to a function -- refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def __annotations__():
            return None
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_class_body_annotations_read(tmp_path: Path) -> None:
    """Codex round 9 finding 1 (continued): inside a class body, an
    unbound ``__annotations__`` reference resolves to the *module*
    globals after inlining. Class bodies have their own implicit
    ``__annotations__`` dict, but a naked Name reads through to module
    scope. Refused at all depths.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        class C:
            x: int = 1
            CAPTURED = __annotations__
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_comprehension_annotations_read(tmp_path: Path) -> None:
    """``__annotations__`` referenced inside a comprehension -- still
    resolves to module globals. Refused at all depths.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f():
            return [k for k in __annotations__]
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_chained_compare_globals_contains(tmp_path: Path) -> None:
    """Codex round 9 finding 2: chained comparison
    ``"x" in globals() in Probe()`` -- the ``__contains__`` synthesis
    in ``visit_Compare`` must not normalize chained compares (only
    simple binary form), since chained semantics differ. Refuse to
    keep the analysis safe.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        class Probe:
            def __contains__(self, item):
                return True

        def check():
            return "x" in globals() in Probe()
        ''',
    )
    with pytest.raises(BuildError):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_compare_globals_with_non_string_key(tmp_path: Path) -> None:
    """``X in globals()`` where ``X`` is not a static string literal
    cannot be analyzed. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def check(name):
            return name in globals()
        ''',
    )
    with pytest.raises(BuildError):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_name_read(tmp_path: Path) -> None:
    """Codex round 9 finding 3: ``__name__`` inside a dependency
    silently shifts from the dep's own module identity to the merged
    target module's name. Refused as reserved dunder.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        SHARED_MARKER = __name__
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_MARKER
        ''',
    )
    with pytest.raises(BuildError, match="__name__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_file_read(tmp_path: Path) -> None:
    """``__file__`` references after inlining point at the merged
    target file, not the dep's source. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        SHARED_MARKER = __file__
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_MARKER
        ''',
    )
    with pytest.raises(BuildError, match="__file__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_package_read(tmp_path: Path) -> None:
    """``__package__`` is module-identity metadata. Refused in deps
    (after inlining the dep's own package identity is lost). The
    target itself can still read its own ``__package__``.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        SHARED_MARKER = __package__
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_MARKER
        ''',
    )
    with pytest.raises(BuildError, match="__package__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_spec_read(tmp_path: Path) -> None:
    """``__spec__`` references the loader spec, also module-identity
    metadata that diverges after inlining. Refused in deps.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        SHARED_MARKER = __spec__
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_MARKER
        ''',
    )
    with pytest.raises(BuildError, match="__spec__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_loader_read(tmp_path: Path) -> None:
    """``__loader__`` is module-identity metadata. Refused in deps."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        SHARED_MARKER = __loader__
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_MARKER
        ''',
    )
    with pytest.raises(BuildError, match="__loader__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_path_read(tmp_path: Path) -> None:
    """``__path__`` is package-identity metadata. Refused in deps."""

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        SHARED_MARKER = __path__
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_MARKER
        ''',
    )
    with pytest.raises(BuildError, match="__path__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_name_via_globals_get(tmp_path: Path) -> None:
    """Dynamic read of ``__name__`` via ``globals().get('__name__')``
    is refused in deps. The target's own ``globals().get('__name__')``
    still resolves to the correct target name post-inlining, so it
    is allowed.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        SHARED_MARKER = globals().get("__name__")
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_MARKER
        ''',
    )
    with pytest.raises(BuildError, match="__name__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_name_via_globals_setitem(tmp_path: Path) -> None:
    """Dynamic write of ``__name__`` via
    ``globals()['__name__'] = ...`` is refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        globals()["__name__"] = "fake"
        ''',
    )
    with pytest.raises(BuildError, match="__name__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_file_static_assignment(tmp_path: Path) -> None:
    """Static ``__file__ = "fake"`` at module scope is refused."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        __file__ = "fake.py"
        ''',
    )
    with pytest.raises(BuildError, match="__file__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_name_del(tmp_path: Path) -> None:
    """``del __name__`` at module scope is refused."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        del __name__
        ''',
    )
    with pytest.raises(BuildError, match="__name__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_module_dunder_doc_rebinding(tmp_path: Path) -> None:
    """``__doc__`` rebinding is refused (would silently overwrite
    the merged module's docstring identity).
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        __doc__ = "fake doc"
        ''',
    )
    with pytest.raises(BuildError, match="__doc__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_target_to_read_own_name(tmp_path: Path) -> None:
    """The target's ``__name__`` is the target itself after inlining,
    so ``log = logging.getLogger(__name__)`` -- the canonical Python
    idiom -- must keep working in the target. Identity dunders are
    only refused in deps.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import logging

        log = logging.getLogger(__name__)
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "logging.getLogger(__name__)" in output


def test_target_still_rejects_file_dunder_read(tmp_path: Path) -> None:
    """The target relaxation only covers ``__name__``. ``__file__``
    stays refused even in the target because the dev-time source
    path and the inlined output path differ -- e.g.
    ``Path(__file__).parent / "data.json"`` resolves to a different
    file before and after inlining.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        SOURCE_PATH = __file__
        ''',
    )
    with pytest.raises(BuildError, match="__file__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_future_annotations_in_dep(tmp_path: Path) -> None:
    """Codex round 10 finding 5 (F5): under PEP 563 a dep's
    annotations become strings; callers like ``typing.get_type_hints``
    later evaluate them against the merged target's globals, so a
    name that was not visible in the original dep can silently
    resolve after inlining. Refuse the future import in deps.
    """

    _write(
        tmp_path / "src/owui_ext/shared/util.py",
        '''
        from __future__ import annotations

        SHARED_MARKER = 1
        ''',
    )
    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from owui_ext.shared.util import SHARED_MARKER
        ''',
    )
    with pytest.raises(BuildError, match="from __future__ import annotations"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_future_annotations_in_target_no_deps(tmp_path: Path) -> None:
    """``from __future__ import annotations`` is refused even in a
    deps-less target. PEP 563 is per-compilation-unit, so the same
    rule applies regardless of whether the build merges deps; refusing
    uniformly keeps the policy simple and catches the case where the
    target later grows a dep.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from __future__ import annotations

        x: int = 1
        ''',
    )
    with pytest.raises(BuildError, match="from __future__ import annotations"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_target_still_rejects_doc_dunder_read(tmp_path: Path) -> None:
    """``__doc__`` stays refused even in the target -- the relaxation
    is intentionally limited to ``__name__`` (the only widespread
    legitimate idiom: ``logging.getLogger(__name__)``).
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        DOC = __doc__
        ''',
    )
    with pytest.raises(BuildError, match="__doc__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_target_still_rejects_annotations_dict_read(tmp_path: Path) -> None:
    """``__annotations__`` must still be refused even at the target,
    because deps' module-scope AnnAssigns merge into the target's
    annotation dict, leaking dep-private annotations into anything
    that reads the target's ``__annotations__``.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        x: int = 1
        KEYS = list(__annotations__)
        ''',
    )
    with pytest.raises(BuildError, match="__annotations__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_builtins_globals_attribute_call(tmp_path: Path) -> None:
    """Codex round 10 finding 1: ``builtins.globals()`` bypasses the
    bare-name detector because ``_is_module_globals_call`` only sees
    ``Call(Name("globals"))``. Refused via attribute-call ban.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import builtins

        G = builtins.globals()
        ''',
    )
    with pytest.raises(BuildError, match="globals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dunder_import_globals(tmp_path: Path) -> None:
    """``__import__("builtins").globals()`` bypasses the bare-name
    detector. Refused via attribute-call ban.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        G = __import__("builtins").globals()
        ''',
    )
    with pytest.raises(BuildError, match="globals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_from_builtins_import_globals(tmp_path: Path) -> None:
    """``from builtins import globals as g; g()`` rebinds the name so
    the bare-name detector cannot see the call. Refused at the import
    line.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from builtins import globals as g
        ''',
    )
    with pytest.raises(BuildError, match="globals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_from_builtins_import_vars(tmp_path: Path) -> None:
    """``from builtins import vars`` is refused for the same reason as
    ``globals``.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from builtins import vars
        ''',
    )
    with pytest.raises(BuildError, match="vars"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_from_builtins_import_locals(tmp_path: Path) -> None:
    """``from builtins import locals`` is refused."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from builtins import locals
        ''',
    )
    with pytest.raises(BuildError, match="locals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_from_builtins_import_dir(tmp_path: Path) -> None:
    """``from builtins import dir`` is refused."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from builtins import dir
        ''',
    )
    with pytest.raises(BuildError, match="dir"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_function_dunder_globals_access(tmp_path: Path) -> None:
    """Codex round 10 finding 2: every function's ``.__globals__``
    attribute is the merged target module's dict. A dep can mutate
    ``__builtins__`` via ``(lambda: None).__globals__["__builtins__"]
    = ...`` without ever calling ``globals()``. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        G = (lambda: None).__globals__
        ''',
    )
    with pytest.raises(BuildError, match="__globals__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_function_dunder_globals_subscript_write(tmp_path: Path) -> None:
    """Mutating ``__builtins__`` through a function's ``__globals__``
    attribute is refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f():
            pass

        f.__globals__["x"] = 1
        ''',
    )
    with pytest.raises(BuildError, match="__globals__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_bare_dir_at_module_scope(tmp_path: Path) -> None:
    """Codex round 10 finding 3: bare ``dir()`` at module scope
    returns the merged module's full namespace -- every other inlined
    dep's identifiers leak in. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        NAMES = dir()
        ''',
    )
    with pytest.raises(BuildError, match="dir"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_builtins_dir_attribute_call(tmp_path: Path) -> None:
    """``builtins.dir()`` bypasses the bare-name detector for
    ``dir``. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import builtins

        NAMES = builtins.dir()
        ''',
    )
    with pytest.raises(BuildError, match="dir"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_allows_dir_with_explicit_argument(tmp_path: Path) -> None:
    """``dir(obj)`` with an explicit argument is fine -- it inspects
    the argument, not the merged namespace.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        class C:
            x = 1

        NAMES = dir(C)
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "dir(C)" in output


def test_allows_dir_inside_function(tmp_path: Path) -> None:
    """Bare ``dir()`` inside a function body returns the function's
    locals, not the merged module's namespace -- safe to allow.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def list_locals():
            x = 1
            return dir()
        ''',
    )
    output = _build(tmp_path, "src/owui_ext/tools/demo.py")
    assert "return dir()" in output


def test_rejects_globals_name_aliasing(tmp_path: Path) -> None:
    """Codex round 11 finding 1: ``G = globals; G()`` aliases the
    builtin to a local name. The bare-name detector only sees
    ``globals()`` Calls -- the rebound ``G`` is invisible. Refuse any
    Name(Load) reference to the builtins outside an immediate call.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        G = globals
        ''',
    )
    with pytest.raises(BuildError, match="globals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dir_name_aliasing(tmp_path: Path) -> None:
    """``D = dir; D()`` -- same alias bypass for ``dir``."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        D = dir
        ''',
    )
    with pytest.raises(BuildError, match="dir"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_passed_as_argument(tmp_path: Path) -> None:
    """``register(globals)`` passes the builtin as a value -- refused
    because the callee can call it back later.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def register(fn):
            return fn

        REG = register(globals)
        ''',
    )
    with pytest.raises(BuildError, match="globals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_getattr_dunder_globals_literal(tmp_path: Path) -> None:
    """Codex round 11 finding 2: ``getattr(fn, "__globals__")`` with a
    literal attribute name routes around the direct attribute ban.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        G = getattr(lambda: None, "__globals__")
        ''',
    )
    with pytest.raises(BuildError, match="__globals__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_setattr_dunder_globals_literal(tmp_path: Path) -> None:
    """``setattr(fn, "__globals__", ...)`` -- same literal-attr
    bypass. CPython rejects this at runtime, but we refuse at AST
    time for consistency with the direct attribute ban.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(): pass

        setattr(f, "__globals__", {})
        ''',
    )
    with pytest.raises(BuildError, match="__globals__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_object_getattribute_dunder_globals(tmp_path: Path) -> None:
    """``object.__getattribute__(fn, "__globals__")`` -- another
    literal-attr bypass. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(): pass

        G = object.__getattribute__(f, "__globals__")
        ''',
    )
    with pytest.raises(BuildError, match="__globals__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dunder_getattribute_method_call(tmp_path: Path) -> None:
    """``fn.__getattribute__("__globals__")`` -- bound dunder method
    form. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(): pass

        G = f.__getattribute__("__globals__")
        ''',
    )
    with pytest.raises(BuildError, match="__globals__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_getattr_globals_string_literal(tmp_path: Path) -> None:
    """``getattr(builtins, "globals")()`` -- literal-attr bypass for
    the builtin name itself. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import builtins

        F = getattr(builtins, "globals")
        ''',
    )
    with pytest.raises(BuildError, match="globals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dunder_dict_globals_subscript(tmp_path: Path) -> None:
    """Codex round 11 finding 3: ``builtins.__dict__["globals"]()``
    accesses the builtin via Subscript on ``__dict__``. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import builtins

        F = builtins.__dict__["globals"]
        ''',
    )
    with pytest.raises(BuildError, match="globals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_vars_obj_globals_subscript(tmp_path: Path) -> None:
    """``vars(builtins)["globals"]`` -- equivalent to
    ``builtins.__dict__["globals"]``. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import builtins

        F = vars(builtins)["globals"]
        ''',
    )
    with pytest.raises(BuildError, match="globals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dunder_dict_globals_dunder_subscript(tmp_path: Path) -> None:
    """``fn.__dict__["__globals__"]`` -- function objects don't
    actually keep ``__globals__`` in ``__dict__``, but the form is
    refused symmetrically with the others. The ``.__dict__``
    attribute itself is now banned (round 12 finding 4), which fires
    before the subscript check.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(): pass

        G = f.__dict__["__globals__"]
        ''',
    )
    with pytest.raises(BuildError, match="__dict__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dir_in_function_default_argument(tmp_path: Path) -> None:
    """Codex round 11 finding 4: ``def f(snapshot=dir())`` evaluates
    ``dir()`` at module scope (default arguments are evaluated when
    the function is defined). The visitor must descend into headers
    at the enclosing depth, not the function body's depth.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(snapshot=dir()):
            return snapshot
        ''',
    )
    with pytest.raises(BuildError, match="dir"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dir_in_lambda_default_argument(tmp_path: Path) -> None:
    """``lambda snapshot=dir(): snapshot`` -- same scope issue, lambda
    form.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        g = lambda snapshot=dir(): snapshot
        ''',
    )
    with pytest.raises(BuildError, match="dir"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dir_in_decorator(tmp_path: Path) -> None:
    """``@some_decorator_factory(dir())`` evaluates ``dir()`` at the
    decorator-evaluation site, which is module scope.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def memo(names):
            def deco(fn):
                return fn
            return deco

        @memo(dir())
        def f():
            pass
        ''',
    )
    with pytest.raises(BuildError, match="dir"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_globals_starred_call(tmp_path: Path) -> None:
    """Codex round 12 finding 1: ``globals(*())`` is syntactically a
    call with one Starred arg, evaluating to ``globals()``. The
    bare-name detector requires no args, but the splat hides the
    empty args. Refused via the strict legitimate-call shape.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        G = globals(*())
        ''',
    )
    with pytest.raises(BuildError, match="globals"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dir_kwargs_unpack(tmp_path: Path) -> None:
    """``dir(**{})`` -- empty kwargs unpack form. Refused."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        N = dir(**{})
        ''',
    )
    with pytest.raises(BuildError, match="dir"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_getattr_starred_args(tmp_path: Path) -> None:
    """Codex round 12 finding 2: ``getattr(*(fn, "__globals__"))``
    routes through Starred, hiding the literal from positional check.
    Refused outright when starred / `**` unpacking is present.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(): pass

        G = getattr(*(f, "__globals__"))
        ''',
    )
    with pytest.raises(BuildError, match="starred"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dunder_getattribute_starred_args(tmp_path: Path) -> None:
    """``object.__getattribute__(*(f, "__globals__"))`` -- same
    starred bypass for the dunder accessor. Refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(): pass

        G = object.__getattribute__(*(f, "__globals__"))
        ''',
    )
    with pytest.raises(BuildError, match="starred"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_getattr_aliasing(tmp_path: Path) -> None:
    """Codex round 12 finding 3: ``READ = object.__getattribute__;
    READ(fn, "__globals__")`` aliases the dunder accessor. The
    aliasing trips on ``object.__getattribute__`` Attribute access
    (``.__getattribute__`` is not directly banned, but aliasing the
    builtin ``getattr`` is).
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        READ = getattr
        ''',
    )
    with pytest.raises(BuildError, match="getattr"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_setattr_aliasing(tmp_path: Path) -> None:
    """``W = setattr; W(...)`` -- alias bypass for ``setattr``."""

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        W = setattr
        ''',
    )
    with pytest.raises(BuildError, match="setattr"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_builtins_getattr_attr_access(tmp_path: Path) -> None:
    """``builtins.getattr(fn, "__globals__")`` -- ``.getattr`` is
    refused regardless of receiver.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import builtins

        F = builtins.getattr
        ''',
    )
    with pytest.raises(BuildError, match="getattr"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_from_builtins_import_getattr(tmp_path: Path) -> None:
    """``from builtins import getattr`` -- refused for the same
    reason as ``from builtins import globals``.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        from builtins import getattr
        ''',
    )
    with pytest.raises(BuildError, match="getattr"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_dunder_dict_get_method(tmp_path: Path) -> None:
    """Codex round 12 finding 4: ``builtins.__dict__.get("globals")``
    -- the ``.get`` method form of dict subscript. The ``.__dict__``
    attribute access ban catches this on the receiver.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import builtins

        F = builtins.__dict__.get("globals")
        ''',
    )
    with pytest.raises(BuildError, match="__dict__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_aliased_dict_subscript(tmp_path: Path) -> None:
    """``D = builtins.__dict__; D["globals"]()`` -- aliasing the
    dict carrier is caught at the assignment line because
    ``.__dict__`` Attribute access is refused.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import builtins

        D = builtins.__dict__
        ''',
    )
    with pytest.raises(BuildError, match="__dict__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_constant_fstring_attr_literal(tmp_path: Path) -> None:
    """Codex round 12 finding 5: ``getattr(fn, f"__globals__")`` --
    a constant-only f-string is semantically the same as the plain
    string. ``_string_constant`` now folds these.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        def f(): pass

        G = getattr(f, f"__globals__")
        ''',
    )
    with pytest.raises(BuildError, match="__globals__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")


def test_rejects_constant_fstring_subscript(tmp_path: Path) -> None:
    """``builtins.__dict__[f"globals"]`` -- the f-string subscript
    is also folded.
    """

    _write(
        tmp_path / "src/owui_ext/tools/demo.py",
        '''
        """version: 0.1"""

        import builtins

        F = builtins.__dict__[f"globals"]
        ''',
    )
    with pytest.raises(BuildError, match="__dict__"):
        _build(tmp_path, "src/owui_ext/tools/demo.py")
