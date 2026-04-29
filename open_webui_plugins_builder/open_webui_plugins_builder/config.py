"""Loader for ``release.toml``.

The schema is intentionally minimal in v1:

    [meta]
    schema_version = 1

    [settings]
    local_import_roots = ["owui_ext.shared"]
    source_root = "src"  # optional, defaults to "src"

    [[targets]]
    name = "parallel_tools"
    source = "src/owui_ext/tools/parallel_tools.py"
    output = "tools/parallel_tools.py"
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import BuildError


SUPPORTED_SCHEMA_VERSION = 1
DEFAULT_SOURCE_ROOT = "src"


@dataclass(frozen=True)
class Target:
    name: str
    source: Path
    output: Path


@dataclass(frozen=True)
class ReleaseConfig:
    repo_root: Path
    local_import_roots: tuple[str, ...]
    source_root: str
    targets: tuple[Target, ...]

    def target_by_name(self, name: str) -> Target:
        for target in self.targets:
            if target.name == name:
                return target
        raise BuildError(f"Unknown target: {name!r}")


def load_config(path: Path) -> ReleaseConfig:
    """Load and validate ``release.toml`` from disk."""

    if not path.is_file():
        raise BuildError(f"release.toml not found at {path}")

    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise BuildError(f"Failed to parse {path}: {exc}") from exc

    return parse_config(raw, repo_root=path.parent.resolve())


def parse_config(raw: dict, *, repo_root: Path) -> ReleaseConfig:
    """Validate already-parsed TOML and resolve paths against ``repo_root``."""

    meta = raw.get("meta") or {}
    if not isinstance(meta, dict):
        raise BuildError(
            f"[meta] must be a table in release.toml (got {type(meta).__name__})."
        )
    schema_version = meta.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise BuildError(
            f"Unsupported release.toml schema_version {schema_version!r}; "
            f"this builder only supports schema_version={SUPPORTED_SCHEMA_VERSION}."
        )

    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        raise BuildError(
            f"[settings] must be a table in release.toml (got "
            f"{type(settings).__name__})."
        )
    raw_roots = settings.get("local_import_roots", [])
    if not isinstance(raw_roots, list) or not all(isinstance(r, str) for r in raw_roots):
        raise BuildError(
            "settings.local_import_roots must be a list of strings in release.toml."
        )
    if not raw_roots:
        raise BuildError(
            "settings.local_import_roots must declare at least one shared root."
        )
    local_import_roots = tuple(raw_roots)

    raw_source_root = settings.get("source_root", DEFAULT_SOURCE_ROOT)
    if not isinstance(raw_source_root, str) or not raw_source_root:
        raise BuildError(
            "settings.source_root must be a non-empty string in release.toml."
        )
    # Only strip trailing slashes -- leading slashes signal an absolute path
    # that the validator must reject rather than silently coerce away.
    source_root = raw_source_root.rstrip("/")
    if not source_root:
        raise BuildError(
            "settings.source_root must be a non-empty string in release.toml."
        )
    _validate_repo_relative(
        source_root, repo_root, label="settings.source_root"
    )

    raw_targets = raw.get("targets") or []
    if not isinstance(raw_targets, list):
        raise BuildError("[[targets]] must be a list in release.toml.")

    seen_names: set[str] = set()
    seen_outputs: set[Path] = set()
    targets: list[Target] = []
    for entry in raw_targets:
        if not isinstance(entry, dict):
            raise BuildError("Each [[targets]] entry must be a table in release.toml.")
        name = entry.get("name")
        source = entry.get("source")
        output = entry.get("output")
        if not isinstance(name, str) or not name:
            raise BuildError("Each [[targets]] entry must have a non-empty 'name'.")
        if not isinstance(source, str) or not source:
            raise BuildError(
                f"Target {name!r} must declare a non-empty 'source' path."
            )
        if not isinstance(output, str) or not output:
            raise BuildError(
                f"Target {name!r} must declare a non-empty 'output' path."
            )
        if name in seen_names:
            raise BuildError(f"Duplicate target name in release.toml: {name!r}")
        seen_names.add(name)

        source_path = _resolve_inside_repo(
            source, repo_root, name=name, field="source"
        )
        output_path = _resolve_inside_repo(
            output, repo_root, name=name, field="output"
        )
        if output_path in seen_outputs:
            raise BuildError(
                f"Duplicate output path in release.toml: {output_path}"
            )
        seen_outputs.add(output_path)

        targets.append(
            Target(name=name, source=source_path, output=output_path)
        )

    if not targets:
        raise BuildError("release.toml must declare at least one [[targets]] entry.")

    return ReleaseConfig(
        repo_root=repo_root.resolve(),
        local_import_roots=local_import_roots,
        source_root=source_root,
        targets=tuple(targets),
    )


def _resolve_inside_repo(
    raw_path: str, repo_root: Path, *, name: str, field: str
) -> Path:
    """Resolve a target source/output path and refuse anything outside the repo."""

    return _validate_repo_relative(
        raw_path, repo_root, label=f"Target {name!r}: {field}"
    )


def _validate_repo_relative(
    raw_path: str, repo_root: Path, *, label: str
) -> Path:
    """Refuse paths that escape the repo root.

    Absolute paths and ``..`` traversal are rejected even if they happen to
    land back inside ``repo_root`` after normalization, because they signal a
    misconfiguration we don't want to silently honor. The check protects both
    accidental misconfigurations (a target accidentally pointing at a sibling
    project, or ``settings.source_root = "../"``) and malicious release.toml
    injecting an arbitrary read/write target outside the workspace.
    """

    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise BuildError(
            f"{label} {raw_path!r} must be a relative path inside the repository."
        )
    if any(part == ".." for part in candidate.parts):
        raise BuildError(
            f"{label} {raw_path!r} must not contain '..' path segments."
        )
    resolved = (repo_root / candidate).resolve()
    repo_resolved = repo_root.resolve()
    try:
        resolved.relative_to(repo_resolved)
    except ValueError as exc:
        raise BuildError(
            f"{label} {raw_path!r} resolves outside the repository "
            f"({resolved} is not under {repo_resolved})."
        ) from exc
    return resolved
