# AGENTS.md

## Global Communication Rules

- Always answer the user in Japanese.
- Instructions intended for the AI (including this document and the return
  values of `Tools` methods) must remain in English, even though user-facing
  replies are in Japanese.

## Working With Open WebUI Core

1. The upstream Open WebUI core lives in the `references` submodule. Before
   consulting or copying code, update the submodule to the latest commit so you
   rely on current behavior.
2. Implement features only after inspecting the actual core and related
   libraries; never guess about undocumented behavior.
3. When adapting snippets, keep them consistent with the upstream style and
   ensure any dependencies are mirrored locally.
4. Before implementing a new Filter/Tool/Action, inspect at least one existing
   implementation of the same type in this repository (and in
   `references/open-webui` if needed) so you follow established patterns and
   edge-case handling.

## Tool Implementation Notes

### General Guidelines

- Assume every public method you add will be called by the AI. Provide precise
  docstrings that describe the purpose, parameters, expected argument formats,
  possible errors, and example correction strategies.
- Tools are never invoked directly by humans; the AI alone decides when to
  call them and assembles the arguments, so design signatures and validation
  with that autonomous caller in mind.
- Expose only callable methods; keep helpers outside of the `Tools` class so the
  AI cannot invoke unsupported operations (method names starting with `_` are
  not an exception).

### Valves vs UserValves

- Follow the official pattern: declare both classes as nested subclasses of the
  Filter/Tools/Pipe you are implementing, inherit from `BaseModel`, and always
  call `self.valves = self.Valves()` inside `__init__` for admin-controlled
  defaults.
- `Valves` describe instance-wide knobs (API keys, DB backends, semaphore limits,
  etc.) and may include fields like `priority` for filter ordering. `UserValves`
  expose only per-user preferences (feature toggles, language choices, etc.).
- Fetch user valves via
  `self.UserValves.model_validate((__user__ or {}).get("valves", {}))` so that
  raw dict payloads from Open WebUI are validated, missing fields fall
  back to defaults, and you avoid sharing mutable state.
- Treat the object stored under `__user__["valves"]` as a Pydantic model, not a
  dict; access attributes (`__user__["valves"].test_user_valve`) or cast with
  `dict(__user__["valves"])` if you truly need a mapping. Indexing into it as if
  it were a dict returns defaults and ignores user-set values.
- Keep the schemas disjoint and document each `Field` carefully so the UI can
  generate appropriate controls (e.g., `Literal[...]` for dropdowns, `bool` for
  switches, `int` for sliders). Always include `pass` at the end of each class as
  recommended in the core documentation.
- The current UI does not expose true multi-select widgets, so when you need
  multiple choices, accept comma-separated strings (e.g., `"choiceA,choiceB"`)
  and document that input format for users.

### Input Validation

- Validate inputs defensively. On invalid arguments, never raise exceptions;
  instead, return structured guidance that tells the AI exactly which field to
  fix and how.

### Return Payload Budget

- Limit every return payload (success or error) to just the data the AI needs
  for its next action so Tool calls do not waste context window space.

### Error Handling

- Anticipate malformed payloads, missing context, or external failures and
  handle them without crashing or raising exceptions.
- When rejecting a request, respond with actionable remediation steps (e.g.,
  "Provide `conversation_id`" or "Retry after refreshing credentials").
- Log or surface enough context in the return payload so the AI can decide
  whether to retry, adjust arguments, or abort.
- Keep return payloads concise; include only the information the AI needs to
  correct its inputs so the context window is not wasted.

## Filter Implementation Notes

### Toggle Mechanics

- Setting `self.toggle = True` displays a toggle button in the Open WebUI UI.
- `enabled_filter_ids` (defined in `core/utils/filter.py`) tracks which filters
  are active. When a user switches the toggle ON, the filter ID is added and
  both `inlet` and `outlet` run; when OFF, the ID is removed and neither
  function is invoked.
- The value of `self.toggle` (True/False) does **not** affect whether the filter
  is enabled by default. It only determines whether a toggle button is shown.
  Users must explicitly enable the filter via the UI toggle.
- Do not re-check `self.toggle` inside `inlet`/`outlet`; if either function
  runs, the toggle is already ON.

### Minimal Skeleton

```python
def __init__(self):
    self.valves = self.Valves()
    self.toggle = True  # Expose toggle in the UI

def inlet(self, body: dict, __user__: dict | None = None) -> dict:
    # Called only when the toggle is enabled
    return body
```
