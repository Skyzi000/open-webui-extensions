"""Version-bump gate.

Open WebUI ships these files as raw ``.py`` modules; users update by replacing
the file. Without a visible version bump in the leading docstring, downstream
consumers have no signal that the file changed. This module enforces:

    "If the rebuilt output differs from HEAD's output, the docstring's
    ``version:`` field must also differ from HEAD's."

The gate is intentionally lenient about value direction (we don't enforce
SemVer ordering) -- only that the value is not byte-identical to HEAD.
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
) -> str | None:
    """Return an error message if the version-bump gate is violated.

    Returns ``None`` when the gate passes (or doesn't apply because there is
    nothing to compare).
    """

    if head_output is None:
        return None  # First commit of this output -- nothing to compare.
    if rebuilt_output == head_output:
        return None

    new_version = extract_version(rebuilt_output)
    old_version = extract_version(head_output)
    if new_version is None:
        return (
            f"{target.name}: rebuilt output {target.output} is missing a "
            f"`version:` field in its leading docstring."
        )
    if old_version is not None and new_version == old_version:
        return (
            f"{target.name}: output {target.output} differs from HEAD but "
            f"`version:` was not bumped (still {new_version!r}). "
            f"Bump version in the source's leading docstring before regenerating."
        )
    return None
