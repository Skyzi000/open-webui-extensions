"""Git index / worktree helpers used by the builder.

Kept tiny on purpose: tests substitute their own callables so unit tests don't
need a real Git repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .errors import BuildError


def run_git(args: list[str], cwd: Path) -> str:
    """Run ``git`` and return stdout. Raises BuildError on non-zero exit."""

    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BuildError("`git` executable not found on PATH.") from exc
    if proc.returncode != 0:
        raise BuildError(
            f"git {' '.join(args)} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()}"
        )
    return proc.stdout


def read_index_blob(repo_root: Path, rel_path: str) -> str | None:
    """Return the contents of ``rel_path`` from the Git index, or None if absent."""

    try:
        out = subprocess.run(
            ["git", "show", f":{rel_path}"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BuildError("`git` executable not found on PATH.") from exc
    if out.returncode != 0:
        return None
    return out.stdout


def read_head_blob(repo_root: Path, rel_path: str) -> str | None:
    """Return the contents of ``rel_path`` from HEAD, or None if absent."""

    try:
        out = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BuildError("`git` executable not found on PATH.") from exc
    if out.returncode != 0:
        return None
    return out.stdout


def read_baseline_blob(repo_root: Path, rel_path: str) -> str | None:
    """Return ``rel_path`` from the shipping baseline (``origin/main`` / ``main``).

    The version-bump gate compares against the *last shipped* version, not
    against HEAD: when a feature branch has already bumped past main and the
    author iterates on the same shipping cycle, the gate would otherwise
    require a fresh bump for every commit on the branch. Returns None when
    no baseline ref is reachable, so callers can fall back to a HEAD
    comparison in that case.

    ``origin/main`` is preferred over the local ``main`` because a stale
    local ``main`` (the dev forgot to ``git pull``) would otherwise let
    the gate read a baseline whose version is older than what is actually
    shipped, allowing a same-version change to slip through.
    """

    for ref in ("origin/main", "main"):
        try:
            out = subprocess.run(
                ["git", "show", f"{ref}:{rel_path}"],
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise BuildError("`git` executable not found on PATH.") from exc
        if out.returncode == 0:
            return out.stdout
    return None


def list_changed_files(repo_root: Path, *, staged: bool) -> set[str]:
    """List files differing from HEAD.

    With ``staged=True``, return paths that differ between HEAD and the index.
    Otherwise, return paths that differ between HEAD and the worktree
    (including untracked files but excluding ignored files).
    """

    if staged:
        out = run_git(
            ["diff", "--name-only", "--cached", "HEAD"], cwd=repo_root
        )
        return {line.strip() for line in out.splitlines() if line.strip()}

    tracked = run_git(["diff", "--name-only", "HEAD"], cwd=repo_root)
    untracked = run_git(
        ["ls-files", "--others", "--exclude-standard"], cwd=repo_root
    )
    return {
        line.strip()
        for source in (tracked, untracked)
        for line in source.splitlines()
        if line.strip()
    }


def is_inside_repo(path: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"
