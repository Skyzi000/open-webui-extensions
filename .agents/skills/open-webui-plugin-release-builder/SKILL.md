---
name: open-webui-plugin-release-builder
description: Use when editing src/owui_ext/, release.toml, scripts/build_release.py, generated plugin outputs, or debugging release-builder output sync, version-bump gates, smoke tests, and pre-commit release hooks. Applies to any plugin type (Tools, Filters, Pipes, Actions).
---

# Open WebUI Plugin Release Builder

## Overview

Open WebUI installs extensions (Tools, Filters, Pipes, Actions) as single `.py` files. This repo keeps larger managed extensions as maintainable source under `src/owui_ext/` and builds installable outputs under `tools/` and `functions/`.

The builder is plugin-type agnostic. Each `[[targets]]` entry in `release.toml` declares a `name`, a `source` path, and an `output` path, regardless of whether the plugin is a Tool, Filter, Pipe, or Action.

Pipeline:

```text
src/owui_ext/... + release.toml
  -> open_webui_plugins_builder
  -> single-file Open WebUI releases (tools/*.py, functions/.../*.py, ...)
```

## Identifying Managed Targets

`release.toml` is the source of truth. Always check the current managed outputs before assuming a file is generated or standalone:

```bash
uv run python scripts/build_release.py --list-outputs
```

Never rely on memory or filename guesses; targets may have been added to or removed from `release.toml` since your last visit.

## Generated vs Standalone Files

Never edit generated outputs directly. Detect generated files by their header, not by filename:

```python
# === GENERATED FILE - DO NOT EDIT ===
# Source: src/owui_ext/...
# Regenerate with: uv run python scripts/build_release.py --target ...
```

For generated targets:

1. Edit the source file under `src/owui_ext/` (or shared code under `src/owui_ext/shared/`).
2. Bump the plugin header version in the source when behavior or installable plugin content changes.
3. Rebuild the corresponding output.
4. Keep source and generated output changes together.

Standalone files (anything in `tools/`, `functions/`, etc. without a generated header, plus any file not listed by `--list-outputs`) are edited directly.

## Builder Commands

Use the project wrapper:

```bash
uv run python scripts/build_release.py --all
uv run python scripts/build_release.py --target <name>
uv run python scripts/build_release.py --changed --check
uv run python scripts/build_release.py --changed --check --staged
uv run python scripts/build_release.py --smoke-test-all
```

Modes:

- `--all` builds/checks every target.
- `--target NAME` builds/checks a named target; may be repeated.
- `--changed` selects targets affected by Git worktree or index changes.
- `--check` exits non-zero if rebuilt content differs or version-bump gates fail.
- `--staged` reads files from the Git index; used by pre-commit.
- `--smoke-test-all` imports every managed output via importlib.

## release.toml Rules

Each target declares a `name`, a `source`, and an `output` path; any plugin type follows the same shape:

```toml
[[targets]]
name = "<name>"
source = "src/owui_ext/<area>/<name>.py"
output = "<install_area>/<name>.py"
```

Add or update a target only when the output should be managed by the builder. Keep `name`, `source`, and `output` repo-relative and unique.

## Inlining Rules

- `local_import_roots` (configured in `release.toml`) controls which local imports may be inlined. Read that key before assuming an import will be inlined.
- Keep reusable generated-target helpers in `src/owui_ext/shared/`.
- Source files and `release.toml` are canonical; generated output internals are not source of truth.
- Shared modules that need Open WebUI or other external packages often import them lazily inside functions to satisfy inliner constraints. Preserve that pattern unless the builder is verified to accept the change.
- Do not mix unsupported external and cross-shared top-level imports in shared deps; run builder checks after touching shared code.

## Pre-commit Hook

`.pre-commit-config.yaml` has `release-builder-check`:

```bash
uv run --no-sync python scripts/build_release.py --changed --check --staged
```

`--no-sync` is intentional. It avoids re-resolving the editable Open WebUI dev dependency and frontend-related setup on every commit. The hook expects the builder to already be installed in the active uv environment. Run once:

```bash
uv sync --extra test --group dev
# or
uv pip install -e ./open_webui_plugins_builder
```

## Verification Checklist

When generated targets may be affected:

1. Run the narrowest target rebuild first, e.g. `uv run python scripts/build_release.py --target <name>` for the specific target you changed.
2. Run `uv run python scripts/build_release.py --changed --check`.
3. Run `uv run python scripts/build_release.py --smoke-test-all` for generated output import safety.
4. Run relevant tests; use full `uv run pytest -v --tb=short` when shared code, Open WebUI contracts, or builder behavior changed.
5. Inspect `git diff` and `git status --short` for accidental generated-output, submodule, cache, lockfile, registry URL, or secret changes.

If `uv run` rewrites `uv.lock` only by changing registry URLs, treat it as an accidental local-environment artifact and revert that lockfile change unless the user explicitly asked to update dependencies.
