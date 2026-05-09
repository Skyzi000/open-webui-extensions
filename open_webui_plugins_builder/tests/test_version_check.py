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


def test_missing_head_version_field_fails_when_output_changes(
    tmp_path: Path,
) -> None:
    """If HEAD has no leading-docstring version, the gate cannot compare safely."""

    target = _target(tmp_path)
    head = '"""title: Demo"""\nprint("hi")\n'
    rebuilt = HEAD_OUTPUT.replace('print("hi")', 'print("hi there")')
    error = check_version_bump(
        target=target,
        rebuilt_output=rebuilt,
        head_output=head,
    )
    assert error is not None
    assert "HEAD" in error
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


def test_branch_internal_iteration_passes_when_baseline_already_bumped(
    tmp_path: Path,
) -> None:
    """A feature branch that already bumped vs main may iterate without re-bumping.

    Without this carve-out the gate would force a fresh version bump for
    every commit on the branch -- so a same-cycle source correction (e.g.
    fixing a misplaced future import in 0.1.13 before merging 0.1.13 to
    main) would have to ship as 0.1.14 even though 0.1.13 has not landed
    on main yet.
    """

    target = _target(tmp_path)
    baseline = HEAD_OUTPUT  # main still has 0.1.0
    head = HEAD_OUTPUT.replace("0.1.0", "0.1.1")  # branch already bumped to 0.1.1
    rebuilt = head.replace('print("hi")', 'print("hi there")')  # iterate again at 0.1.1
    error = check_version_bump(
        target=target,
        rebuilt_output=rebuilt,
        head_output=head,
        baseline_output=baseline,
        baseline_is_ancestor_of_head=True,
    )
    assert error is None


def test_baseline_with_same_version_still_requires_bump(tmp_path: Path) -> None:
    """If branch hasn't bumped vs main, the gate still demands a bump."""

    target = _target(tmp_path)
    baseline = HEAD_OUTPUT  # main: 0.1.0
    head = HEAD_OUTPUT  # branch unchanged: 0.1.0
    rebuilt = head.replace('print("hi")', 'print("hi there")')  # change without bump
    error = check_version_bump(
        target=target,
        rebuilt_output=rebuilt,
        head_output=head,
        baseline_output=baseline,
        baseline_is_ancestor_of_head=True,
    )
    assert error is not None
    assert "0.1.0" in error


def test_missing_baseline_falls_back_to_head_check(tmp_path: Path) -> None:
    """When main is unreachable (e.g., shallow clone), gate uses HEAD only."""

    target = _target(tmp_path)
    rebuilt = HEAD_OUTPUT.replace('print("hi")', 'print("hi there")')
    error = check_version_bump(
        target=target,
        rebuilt_output=rebuilt,
        head_output=HEAD_OUTPUT,
        baseline_output=None,
    )
    assert error is not None  # same as the no-baseline-arg path


def test_stale_branch_demands_rebase_with_unbumped_version(tmp_path: Path) -> None:
    """A branch that does not descend from baseline must be rebased.

    Scenario: ``origin/main`` already shipped 0.1.1, but the branch was
    cut before that bump and HEAD is still 0.1.0. Without the rebase
    requirement, ``new(0.1.0) != baseline(0.1.1)`` would satisfy the
    carve-out and the gate would silently accept a rolled-back release.
    """

    target = _target(tmp_path)
    baseline = HEAD_OUTPUT.replace("0.1.0", "0.1.1")  # main: 0.1.1
    head = HEAD_OUTPUT  # stale branch HEAD: 0.1.0
    rebuilt = head.replace('print("hi")', 'print("hi there")')
    error = check_version_bump(
        target=target,
        rebuilt_output=rebuilt,
        head_output=head,
        baseline_output=baseline,
        baseline_is_ancestor_of_head=False,
    )
    assert error is not None
    assert "Rebase" in error or "rebase" in error


def test_stale_branch_demands_rebase_even_with_locally_bumped_version(
    tmp_path: Path,
) -> None:
    """A stale branch that locally bumps to a numerically behind value
    must still be rejected. The HEAD-only check would otherwise see
    "version changed" and pass, but 0.1.6 < 0.2.0 is a rollback against
    the actual baseline.
    """

    target = _target(tmp_path)
    baseline = HEAD_OUTPUT.replace("0.1.0", "0.2.0")  # main: 0.2.0
    head = HEAD_OUTPUT.replace("0.1.0", "0.1.5")  # stale branch: 0.1.5
    rebuilt = head.replace("0.1.5", "0.1.6").replace(
        'print("hi")', 'print("hi there")'
    )
    error = check_version_bump(
        target=target,
        rebuilt_output=rebuilt,
        head_output=head,
        baseline_output=baseline,
        baseline_is_ancestor_of_head=False,
    )
    assert error is not None
    assert "rebase" in error.lower()


def test_baseline_present_without_ancestor_flag_demands_rebase(
    tmp_path: Path,
) -> None:
    """The default ``baseline_is_ancestor_of_head=False`` is the safe
    fallback: a baseline blob exists but ancestry could not be verified
    (or the branch is genuinely diverged), so the gate refuses the
    change.
    """

    target = _target(tmp_path)
    baseline = HEAD_OUTPUT.replace("0.1.0", "0.1.5")
    head = HEAD_OUTPUT
    rebuilt = head.replace('print("hi")', 'print("hi there")')
    error = check_version_bump(
        target=target,
        rebuilt_output=rebuilt,
        head_output=head,
        baseline_output=baseline,
        # Default: False
    )
    assert error is not None
    assert "rebase" in error.lower()


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
