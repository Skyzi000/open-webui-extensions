"""Version-bump gate."""

from __future__ import annotations

import textwrap
from pathlib import Path

from open_webui_plugins_builder.config import Target
from open_webui_plugins_builder.version_check import (
    check_version_bump,
    extract_version,
)


def _target(tmp_path: Path) -> Target:
    return Target(
        name="demo",
        source=(tmp_path / "src/demo.py").resolve(),
        output=(tmp_path / "tools/demo.py").resolve(),
    )


HEAD_OUTPUT = textwrap.dedent(
    '''
    """
    title: Demo
    version: 0.1.0
    """
    print("hi")
    '''
).lstrip()


def test_extract_version_handles_leading_docstring() -> None:
    assert extract_version(HEAD_OUTPUT) == "0.1.0"


def test_no_change_passes(tmp_path: Path) -> None:
    target = _target(tmp_path)
    assert check_version_bump(
        target=target, rebuilt_output=HEAD_OUTPUT, head_output=HEAD_OUTPUT
    ) is None


def test_no_head_passes(tmp_path: Path) -> None:
    target = _target(tmp_path)
    assert check_version_bump(
        target=target, rebuilt_output=HEAD_OUTPUT, head_output=None
    ) is None


def test_change_without_bump_fails(tmp_path: Path) -> None:
    target = _target(tmp_path)
    rebuilt = HEAD_OUTPUT.replace('print("hi")', 'print("hi there")')
    error = check_version_bump(
        target=target, rebuilt_output=rebuilt, head_output=HEAD_OUTPUT
    )
    assert error is not None
    assert "version" in error.lower()
    assert "0.1.0" in error


def test_change_with_bump_passes(tmp_path: Path) -> None:
    target = _target(tmp_path)
    rebuilt = HEAD_OUTPUT.replace("0.1.0", "0.1.1").replace(
        'print("hi")', 'print("hi there")'
    )
    assert check_version_bump(
        target=target, rebuilt_output=rebuilt, head_output=HEAD_OUTPUT
    ) is None


def test_missing_version_field_fails(tmp_path: Path) -> None:
    target = _target(tmp_path)
    rebuilt = '"""no version here"""\nprint("hi there")\n'
    error = check_version_bump(
        target=target, rebuilt_output=rebuilt, head_output=HEAD_OUTPUT
    )
    assert error is not None
    assert "version:" in error


def test_extract_version_ignores_comment_lines() -> None:
    """A ``# version: 0.2.0`` comment must not satisfy the gate.

    Without an AST-based extractor a regex that scans the first 4 KiB
    would happily grab the comment's value, letting a release ship
    without a real docstring version line.
    """

    source = textwrap.dedent(
        """
        # version: 0.2.0
        \"\"\"no version field in this docstring\"\"\"
        print("hi")
        """
    ).lstrip()
    assert extract_version(source) is None


def test_extract_version_ignores_non_leading_string() -> None:
    """A ``version: ...`` line in a later string literal is not the leader."""

    source = textwrap.dedent(
        """
        \"\"\"no version field here\"\"\"

        DOCS = \"\"\"version: 0.2.0\"\"\"
        """
    ).lstrip()
    assert extract_version(source) is None


def test_extract_version_ignores_module_with_no_docstring() -> None:
    """Source whose first statement is code (not a docstring) yields None."""

    source = textwrap.dedent(
        """
        # version: 0.2.0
        VALUE = "version: 0.2.0"
        """
    ).lstrip()
    assert extract_version(source) is None


def test_extract_version_ignores_invalid_syntax() -> None:
    """Unparseable source returns None rather than crashing the gate."""

    assert extract_version("def broken(:\n") is None


def test_change_without_leading_docstring_version_fails(tmp_path: Path) -> None:
    """A rebuilt file whose leading docstring lacks ``version:`` must fail.

    Even if a comment elsewhere reads ``# version: 0.2.0``, the gate must
    still report a missing field -- the leading docstring is the only
    source of truth for the public version.
    """

    target = _target(tmp_path)
    rebuilt = textwrap.dedent(
        """
        # version: 0.2.0
        \"\"\"title: Demo\"\"\"
        print("hi there")
        """
    ).lstrip()
    error = check_version_bump(
        target=target, rebuilt_output=rebuilt, head_output=HEAD_OUTPUT
    )
    assert error is not None
    assert "version:" in error
