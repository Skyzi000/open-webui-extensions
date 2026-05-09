"""``--smoke-test-all`` runner.

Loads each managed output via importlib in an isolated namespace. This catches
syntax errors, import-time failures, and module-level NameError caused by
broken inlining. We do not run any test functions -- we only require that
``importlib.util.spec_from_file_location(...).loader.exec_module(...)``
completes without raising.

To avoid name collisions when multiple outputs are smoke-tested in one
process, each output is loaded under a unique synthetic module name.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
import uuid
from pathlib import Path

from .config import ReleaseConfig


def smoke_test_outputs(config: ReleaseConfig) -> list[tuple[Path, str]]:
    """Return a list of ``(path, error_text)`` tuples for failures."""

    failures: list[tuple[Path, str]] = []
    for target in config.targets:
        path = target.output
        if not path.is_file():
            failures.append((path, f"output not found: {path}"))
            continue
        synthetic_name = f"_owui_smoke_{path.stem}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(synthetic_name, path)
        if spec is None or spec.loader is None:
            failures.append((path, "could not create import spec"))
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[synthetic_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:  # noqa: BLE001 -- want to capture SystemExit too
            failures.append((path, traceback.format_exc()))
        finally:
            sys.modules.pop(synthetic_name, None)
    return failures
