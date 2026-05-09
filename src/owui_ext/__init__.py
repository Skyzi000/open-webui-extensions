"""Source tree for Open WebUI plugin extensions.

Files under this package are inlined into single-file release artifacts by
``open-webui-plugins-builder`` and emitted to the destinations declared in
``release.toml``. Do NOT import from ``owui_ext`` at runtime in the released
files -- the builder rewrites those imports away.
"""
