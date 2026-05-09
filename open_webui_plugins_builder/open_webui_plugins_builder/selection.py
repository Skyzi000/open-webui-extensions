"""Target selection logic for ``--changed`` mode.

If ``release.toml``, the builder package, or the ``scripts/build_release.py``
wrapper has changed, all targets are selected (broad invalidation). Otherwise
we pick targets whose source, declared shared dependency directory, or
generated output appears in the changed set.

Shared dependency tracking is done at directory granularity in v1: any change
under ``src/owui_ext/shared/`` invalidates every target that imports something
from ``owui_ext.shared``. A more precise per-module dependency walk is left to
a future schema bump because v1 doesn't track per-target dependency graphs in
the cache.
"""

from __future__ import annotations

from pathlib import Path

from .config import ReleaseConfig, Target


GLOBAL_INVALIDATORS = (
    "release.toml",
    "open_webui_plugins_builder/",
    "scripts/build_release.py",
)


def select_changed_targets(
    config: ReleaseConfig, *, changed_files: set[str]
) -> list[Target]:
    if any(_matches_global(path) for path in changed_files):
        return list(config.targets)

    shared_dirs = _shared_directories(
        config.repo_root, config.local_import_roots, config.source_root
    )
    shared_changed = any(
        _path_within(path, shared_dir, config.repo_root)
        for path in changed_files
        for shared_dir in shared_dirs
    )

    selected: list[Target] = []
    for target in config.targets:
        rel_source = _relative(target.source, config.repo_root)
        rel_output = _relative(target.output, config.repo_root)
        hit = rel_source in changed_files or rel_output in changed_files
        if not hit and shared_changed:
            hit = True
        if hit:
            selected.append(target)
    return selected


def _matches_global(path: str) -> bool:
    for prefix in GLOBAL_INVALIDATORS:
        if prefix.endswith("/"):
            if path.startswith(prefix):
                return True
        elif path == prefix:
            return True
    return False


def _shared_directories(
    repo_root: Path, local_roots: tuple[str, ...], source_root: str
) -> list[Path]:
    return [
        repo_root / source_root / Path(*root.split("."))
        for root in local_roots
    ]


def _path_within(rel_path: str, dir_path: Path, repo_root: Path) -> bool:
    candidate = (repo_root / rel_path).resolve()
    try:
        candidate.relative_to(dir_path.resolve())
    except ValueError:
        return False
    return True


def _relative(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
