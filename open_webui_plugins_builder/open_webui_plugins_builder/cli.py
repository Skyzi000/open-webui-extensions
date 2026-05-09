"""Command-line entry point for the single-file release builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import ReleaseConfig, Target, load_config, parse_config
from .errors import BuildError
from .git_io import (
    baseline_is_ancestor_of_head,
    is_inside_repo,
    list_changed_files,
    read_baseline_blob,
    read_head_blob,
    read_index_blob,
)
from .inliner import build_target
from .selection import select_changed_targets
from .smoke import smoke_test_outputs
from .version_check import check_version_bump


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="open-webui-plugins-builder",
        description="Inline-build single-file Open WebUI plugin releases.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--release-toml",
        type=Path,
        default=None,
        help="Path to release.toml (default: search upward from CWD).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--target",
        action="append",
        metavar="NAME",
        help="Build (or check) a single target by name. May be passed multiple times.",
    )
    mode.add_argument("--all", action="store_true", help="Build (or check) every target.")
    mode.add_argument(
        "--changed",
        action="store_true",
        help="Build (or check) only targets whose source / output has changed.",
    )
    mode.add_argument(
        "--list-outputs",
        action="store_true",
        help="Print each target's output path, one per line.",
    )
    mode.add_argument(
        "--smoke-test-all",
        action="store_true",
        help="Import every managed output via importlib and report failures.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write outputs. Exit non-zero if any target's rebuilt "
            "content differs from its current output, or if the version-bump "
            "gate is violated."
        ),
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help=(
            "Read sources, dependencies, release.toml, and outputs from the "
            "Git index instead of the worktree."
        ),
    )

    args = parser.parse_args(argv)

    try:
        config_path = _locate_release_toml(args.release_toml)
        config = _load_config(config_path, staged=args.staged)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list_outputs:
        for target in config.targets:
            print(_safe_relative(target.output, config.repo_root))
        return 0

    if args.smoke_test_all:
        return _run_smoke(config)

    try:
        selected = _select_targets(config, args)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not selected:
        # Nothing to do is not an error -- pre-commit will call us with no
        # changed targets all the time.
        return 0

    return _run_build_or_check(config, selected, args)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _locate_release_toml(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "release.toml"
        if candidate.is_file():
            return candidate
    raise BuildError(
        "release.toml not found in current directory or any parent. "
        "Pass --release-toml explicitly."
    )


def _load_config(config_path: Path, *, staged: bool) -> ReleaseConfig:
    if not staged:
        return load_config(config_path)

    import io
    import tomllib

    repo_root = config_path.parent.resolve()
    rel = str(config_path.resolve().relative_to(repo_root)).replace("\\", "/")
    index_text = read_index_blob(repo_root, rel)
    if index_text is None:
        raise BuildError(
            f"release.toml is not staged in the index at {rel}; "
            f"stage it before running with --staged."
        )
    try:
        raw = tomllib.load(io.BytesIO(index_text.encode("utf-8")))
    except tomllib.TOMLDecodeError as exc:
        raise BuildError(f"Failed to parse staged {rel}: {exc}") from exc
    return parse_config(raw, repo_root=repo_root)


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def _select_targets(config: ReleaseConfig, args: argparse.Namespace) -> list[Target]:
    if args.target:
        seen: set[str] = set()
        ordered: list[Target] = []
        for name in args.target:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(config.target_by_name(name))
        return ordered

    if args.all:
        return list(config.targets)

    if args.changed:
        if not is_inside_repo(config.repo_root):
            raise BuildError(
                "--changed requires a Git repository at the release.toml root."
            )
        changed = list_changed_files(config.repo_root, staged=args.staged)
        return select_changed_targets(config, changed_files=changed)

    return []


# ---------------------------------------------------------------------------
# Build / check pipeline
# ---------------------------------------------------------------------------


def _run_build_or_check(
    config: ReleaseConfig,
    selected: list[Target],
    args: argparse.Namespace,
) -> int:
    reader = _make_reader(config, staged=args.staged)
    has_repo = is_inside_repo(config.repo_root)
    failed = False

    for target in selected:
        try:
            source_text = reader(target.source)
        except FileNotFoundError as exc:
            print(f"error: {target.name}: {exc}", file=sys.stderr)
            failed = True
            continue
        except BuildError as exc:
            print(f"error: {target.name}: {exc}", file=sys.stderr)
            failed = True
            continue

        try:
            rebuilt = build_target(
                source_text=source_text,
                source_path=target.source,
                repo_root=config.repo_root,
                local_import_roots=config.local_import_roots,
                source_root=config.source_root,
                target_name=target.name,
                read_source=reader,
            )
        except BuildError as exc:
            print(f"error: {target.name}: {exc}", file=sys.stderr)
            failed = True
            continue

        if args.check:
            on_disk = _read_current_output(config, target, staged=args.staged)
            if on_disk is None:
                print(
                    f"OUT-OF-DATE: {target.name}: output {target.output} does not exist; "
                    f"regenerate with `--target {target.name}` and stage the result.",
                    file=sys.stderr,
                )
                failed = True
                continue
            if on_disk != rebuilt:
                print(
                    f"OUT-OF-DATE: {target.name}: rebuilt content differs from "
                    f"{target.output}. Run `uv run python scripts/build_release.py "
                    f"--target {target.name}` and stage the result.",
                    file=sys.stderr,
                )
                failed = True
                continue
            if has_repo:
                head_output = _read_head_output(config, target)
                baseline_output = _read_baseline_output(config, target)
                baseline_ancestor = (
                    baseline_is_ancestor_of_head(config.repo_root)
                    if baseline_output is not None
                    else False
                )
                error = check_version_bump(
                    target=target,
                    rebuilt_output=rebuilt,
                    head_output=head_output,
                    baseline_output=baseline_output,
                    baseline_is_ancestor_of_head=baseline_ancestor,
                )
                if error:
                    print(f"VERSION-BUMP REQUIRED: {error}", file=sys.stderr)
                    failed = True
        else:
            target.output.parent.mkdir(parents=True, exist_ok=True)
            target.output.write_text(rebuilt, encoding="utf-8")
            print(f"wrote {target.output}")

    return 1 if failed else 0


def _make_reader(config: ReleaseConfig, *, staged: bool):
    if not staged:
        def fs_reader(path: Path) -> str:
            return path.read_text(encoding="utf-8")
        return fs_reader

    repo_root = config.repo_root

    def index_reader(path: Path) -> str:
        rel = _safe_relative(path, repo_root)
        text = read_index_blob(repo_root, rel)
        if text is None:
            raise BuildError(
                f"file {rel} is not present in the Git index; stage it before "
                f"running with --staged."
            )
        return text

    return index_reader


def _read_current_output(
    config: ReleaseConfig, target: Target, *, staged: bool
) -> str | None:
    if staged:
        rel = _safe_relative(target.output, config.repo_root)
        return read_index_blob(config.repo_root, rel)
    if not target.output.is_file():
        return None
    return target.output.read_text(encoding="utf-8")


def _read_head_output(config: ReleaseConfig, target: Target) -> str | None:
    rel = _safe_relative(target.output, config.repo_root)
    return read_head_blob(config.repo_root, rel)


def _read_baseline_output(config: ReleaseConfig, target: Target) -> str | None:
    rel = _safe_relative(target.output, config.repo_root)
    return read_baseline_blob(config.repo_root, rel)


def _safe_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError as exc:
        raise BuildError(
            f"path {path} is not inside the repo root {repo_root}; "
            f"release.toml entries must be repo-relative."
        ) from exc


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _run_smoke(config: ReleaseConfig) -> int:
    failures = smoke_test_outputs(config)
    if not failures:
        for target in config.targets:
            print(f"ok   {target.output}")
        return 0
    for path, error_text in failures:
        print(f"FAIL {path}\n{error_text}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
