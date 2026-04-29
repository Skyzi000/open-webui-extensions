"""Version-bump gate.

Open WebUI ships these files as raw ``.py`` modules; users update by replacing
the file. Without a visible version bump in the leading docstring, downstream
consumers have no signal that the file changed. This module enforces:

    "If the rebuilt output differs from HEAD's output, the docstring's
    ``version:`` field must also differ from HEAD's -- unless the rebuilt
    output's version already differs from the shipping baseline (``main``)
    *and* HEAD descends from the baseline, in which case a same-cycle
    in-branch iteration is allowed."

The HEAD-descends-from-baseline guard exists so a branch forked from an
older ``main`` cannot satisfy ``new_version != baseline_version`` by
carrying a *behind* version: without it, a stale fork would silently
slip a same-or-older version through the gate.

The gate is intentionally lenient about value direction (we don't enforce
SemVer ordering) -- only that the value is not byte-identical to the
relevant comparison point.
"""

from __future__ import annotations

import ast
import re

from .config import Target


# Matches a single ``version: <value>`` line within a docstring's text.
_VERSION_LINE_RE = re.compile(r"^\s*version\s*:\s*(\S.*?)\s*$", re.MULTILINE)


def extract_version(source: str) -> str | None:
    """Return the ``version:`` value from the module's leading docstring.

    Open WebUI plugins declare metadata in the *leading* module docstring.
    Earlier versions of this gate scanned the first 4 KiB with a regex,
    which would also pick up ``version:`` mentions in comments or other
    string literals -- letting a release without a leading-docstring
    version line slip through. We now parse the AST and only inspect the
    first statement when it's a string literal.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    first = tree.body[0]
    if not (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return None
    match = _VERSION_LINE_RE.search(first.value.value)
    if match is None:
        return None
    return match.group(1)


def check_version_bump(
    *,
    target: Target,
    rebuilt_output: str,
    head_output: str | None,
    baseline_output: str | None = None,
    baseline_is_ancestor_of_head: bool = False,
) -> str | None:
    """Return an error message if the version-bump gate is violated.

    Returns ``None`` when the gate passes (or doesn't apply because there is
    nothing to compare).

    ``baseline_output`` is the file's content on the shipping baseline
    (``main``); when provided *and* HEAD descends from that baseline,
    a rebuilt output whose version already differs from the baseline
    passes the gate even if HEAD has the same version. This lets a feature
    branch iterate on the same shipping cycle without requiring a fresh
    bump per commit. The ancestry guard prevents a stale fork (HEAD does
    not descend from baseline) from satisfying ``new_version !=
    baseline_version`` by carrying a behind/rolled-back version.
    """

    if head_output is None:
        return None  # First commit of this output -- nothing to compare.
    if rebuilt_output == head_output:
        return None

    new_version = extract_version(rebuilt_output)
    if new_version is None:
        return (
            f"{target.name}: rebuilt output {target.output} is missing a "
            f"`version:` field in its leading docstring."
        )
    if baseline_output is not None and baseline_is_ancestor_of_head:
        baseline_version = extract_version(baseline_output)
        if baseline_version is not None and new_version != baseline_version:
            return None
    old_version = extract_version(head_output)
    if old_version is not None and new_version == old_version:
        return (
            f"{target.name}: output {target.output} differs from HEAD but "
            f"`version:` was not bumped (still {new_version!r}). "
            f"Bump version in the source's leading docstring before regenerating."
        )
    return None
