# open-webui-plugins-builder

Single-file release builder for Open WebUI plugin extensions.

Open WebUI distributes Tools, Filters, Pipes, and Actions as single `.py`
files. This package builds those files from a multi-file source tree under
`src/owui_ext/` by inlining local shared modules into each release target.
The builder is plugin-type agnostic -- each `[[targets]]` entry in
`release.toml` declares a `name`, a `source` path, and an `output` path.

Run via the project wrapper:

```bash
uv run python scripts/build_release.py --all
```

See `release.toml` at the repository root for target configuration.
