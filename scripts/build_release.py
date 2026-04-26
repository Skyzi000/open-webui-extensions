#!/usr/bin/env python3
"""Thin wrapper that delegates to ``open_webui_plugins_builder.cli.main``.

Run with::

    uv run python scripts/build_release.py [args...]

All real logic lives in the ``open-webui-plugins-builder`` package so the same
entry point can be reused from CI, hooks, and unit tests.
"""

from __future__ import annotations

import sys

from open_webui_plugins_builder.cli import main


if __name__ == "__main__":
    sys.exit(main())
