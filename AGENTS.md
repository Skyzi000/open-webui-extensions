# AGENTS.md

## Repository Map

- `src/owui_ext/` - source for generated plugin releases (Tools, Filters, Pipes, Actions); `shared/` holds inlined helpers.
- `tools/`, `functions/` - installable Open WebUI extensions; some are generated outputs.
- `open_webui_plugins_builder/` - local release-builder package.
- `tests/` - extension tests plus builder tests.
- `references/open-webui/`, `graphiti/` - upstream core and Graphiti submodules.

## Generated Release Outputs

- Never edit generated outputs directly. Edit source under `src/owui_ext/` and rebuild the corresponding output.
- Bump plugin header versions when installable extension behavior changes, including standalone files.
- For builder commands, target lists, inliner constraints, version-bump gates, smoke tests, and pre-commit details, read `.agents/skills/open-webui-plugin-release-builder/SKILL.md`.

## Setup, Tests, and Core Compatibility

- Use `uv`; do not assume system `pip` exists.
- Install dev/test deps with `uv sync --extra test --group dev`.
- Run the full suite with `uv run pytest -v --tb=short`; do not pass only `tests/` unless intentionally skipping builder tests.
- Before adding a Tool, Filter, Pipe, or Action, read at least one existing same-type implementation in this repo first.
- Inspect `references/open-webui/` before implementing behavior that depends on Open WebUI internals; never guess undocumented behavior.
- If current upstream behavior matters, update or explicitly verify the submodule commit first, then add/update tests for the contract.
- Do not invent constraints the Open WebUI core does not impose; track upstream changes in `references/open-webui/` and follow them when necessary.

## Valves and UserValves

### CRITICAL: Never Rename Released Valve Fields

- Once a `Valves` or `UserValves` field name has appeared on `main`, renaming it is immediately breaking: Open WebUI persists saved settings by field name and provides no migration path. Add new fields with defaults instead.

### Schema Rules

- Declare nested `Valves` and `UserValves` subclasses inheriting `BaseModel`; set `self.valves = self.Valves()` in `__init__` for admin defaults.
- `Valves` are instance/admin knobs; `UserValves` are per-user preferences.
- Fetch user valves with `self.UserValves.model_validate((__user__ or {}).get("valves", {}))`.
- Treat `__user__["valves"]` as a Pydantic model; dict indexing returns defaults and silently ignores user-set values.
- Use `Literal[...]` for dropdown choices and `bool` for switches; Open WebUI generates UI controls from the Pydantic schema.
- Document each `Field` and end each nested Valve class with `pass`. For multi-select, use documented comma-separated strings since Open WebUI has no native multi-select widget.

## Tool Implementation Rules

- Keep AI-facing docs (AGENTS.md, SKILL.md), Tool docstrings, and Tool return values in English for token efficiency.
- Put Tool-call guidance in method docstrings, not class docstrings; Open WebUI does not surface class docstrings to the model.
- Every public method on `class Tools` is AI-callable. Expose only methods the AI should invoke; keep helpers at module scope because underscore names are still exposed.
- Public Tool docstrings must guide an autonomous AI caller, not a human.
- Do not document injected params such as `__user__`, `__request__`, `__event_emitter__`, or `__metadata__`.
- Write each `:param name:` description on one line; continuation lines are dropped by Open WebUI's parser. Do not use Google-style `Args:` blocks.
- Avoid raw `dict`, `Dict`, and `list[dict]` public parameter schemas; use Pydantic `BaseModel` types for structured objects/lists.
- Validate AI-provided arguments defensively and return concise actionable errors instead of uncaught exceptions.
- Never add silent fallbacks that hide unrecoverable failures or speculate about unhandled paths; only fall back when degradation is clearly justified, otherwise return a clear actionable error.
- Keep Tool return payloads small: include only data needed for the next AI action.
- Run `python scripts/lint_tools_class.py` when changing Tool public APIs.

## Filter Implementation Rules

- Usually do not set `self.toggle`; set `self.toggle = True` only for filters that need chat-UI temporary activation.
- When `toggle = True`, do not re-check it inside `inlet`/`outlet`; if either runs, the toggle is already enabled.
- inlet/outlet must be deterministic, safe on missing metadata, and must not mutate caller-owned body/metadata unless tested.
- Keep injected prompt/metadata additions minimal and stable across calls for prompt caching and debugging.

## Scalability and Distributed Deployment

### CRITICAL: No Instance Variables for Cross-Worker State

- Instance variables (`self._cache`, `self._lock`, etc.) are per-worker only and NOT shared across workers or instances; never use them for state that must be consistent in multi-worker deployments.
- For cross-worker coordination (dedup, locking, caching), use Redis via Open WebUI's `REDIS_URL` / `WEBSOCKET_REDIS_URL` with `redis.asyncio`; follow the established pattern in `functions/filter/inlet_title_generator.py` and `graphiti/functions/filter/graphiti_memory.py` (SETNX with TTL, in-memory fallback for single-instance).
- Never block the event loop with synchronous I/O or CPU-heavy work; at thousands-of-users scale one blocking call degrades everyone. Use async APIs for I/O, executors for CPU-bound work, and background tasks for long-running operations.
- `asyncio.Lock()` only synchronizes within a single worker process; use Redis SETNX with TTL for distributed dedup/locking.

## Test Placement

- Add/update tests near the affected area: `tests/tools/`, `tests/filters/`, `tests/graphiti/`, or `open_webui_plugins_builder/tests/`.
- For generated extensions, test installable generated outputs but edit source under `src/owui_ext/`.
- For Open WebUI compatibility, prefer tests importing real core modules from `references/open-webui/backend` over mock-only tests.
- Do not skip or xfail failing tests unless the reason is documented and unrelated to your change.

## Commit Messages

- Use English Conventional Commits in this repo and submodules.
- Write why the change was made, not what changed; the diff already shows the what.
- For large or mixed commits, use `feat` when the primary strategic change adds capability even if follow-up fixes are included.
- Before committing in a submodule, inspect recent history with `git log -5 --format=medium` and follow that submodule's style.

## Final Checklist

1. Confirm whether edited files are generated outputs or source files.
2. If generated outputs may be affected, follow the release-builder skill.
3. Run relevant tests or explain exactly why they could not be run.
4. Check `git status --short` and review the diff for accidental submodule, generated-file, cache, lockfile, registry URL, or secret changes.
5. Report changed paths and real verification results in the user's language.
