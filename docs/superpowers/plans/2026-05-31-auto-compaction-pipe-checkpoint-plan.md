# Auto Compaction Pipe Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Open WebUI manifold Pipe that keeps arbitrarily long Open WebUI chats continuing without provider context-window overflow by compacting each target-model call before it becomes too large. The Pipe exposes compacting wrapper models for many real models, forwards to the selected real model, and persists reusable compaction checkpoints as durable continuation state in an extension-owned DB table.

**Architecture:** The Pipe stays thin: Open WebUI core continues to own model dispatch, native tool execution, events, files, citations, and chat persistence. The Pipe is a manifold Pipe: it dynamically lists compact wrapper models for eligible real models, keeps DB-backed wrapper Model records/access grants synchronized through existing Open WebUI model APIs, decodes the selected wrapper back to its target model, rewrites only a copied `messages` payload, and stores/reuses prefix summaries in a private checkpoint table.

**Tech Stack:** Open WebUI Function Pipe API, Python, Pydantic, SQLAlchemy async engine/session from `open_webui.internal.db`, pytest, `uv`.

---

## Plan Style Constraint

This plan intentionally avoids concrete implementation code and concrete test code. It specifies required behavior, contracts, files, and verification points. The implementation agent is responsible for designing the actual functions, types, and code structure that satisfy these requirements.

Concrete identifiers that are part of external contracts, table names, Valve names, task names, file paths, commands, or Open WebUI APIs may appear because they are required integration points.

## Primary Continuation Requirement

This Pipe exists to prevent long-running chats from stopping because the accumulated Open WebUI history no longer fits into the target model context window. Every design decision in v1 must preserve that purpose.

- Checkpoints are durable continuation state, not disposable optimization cache.
- Existing ready checkpoints must remain readable and extendable across ordinary implementation, prompt, summary model, target model, and cut-policy changes whenever the stored summary format can still be consumed.
- If any ready checkpoint matches a prefix in the current chat-history chain, the Pipe must use it regardless of the provider-usage threshold. Exact checkpoint reuse, rolling checkpoint extension, or parent-checkpoint compaction is mandatory once a valid checkpoint exists; raw direct forwarding is allowed only before any applicable checkpoint exists and no compaction is otherwise required.
- Provider context-window errors are compaction signals, not terminal failures. When the first attempt overflows, the Pipe must compact with the same latest-user boundary and retry instead of surfacing the error whenever no target content has already been emitted.
- Automatic checkpoint expiry, pruning, or prompt-version invalidation violates the v1 objective because it can make an otherwise continuable long chat impossible to continue.
- External provider failures and provider errors that cannot be classified as context overflow remain explicit v1 risks, not reasons to weaken the continuation requirement.

## Confirmed Core Constraints

- Open WebUI 0.9.5 registers Pipe functions as model entries and dispatches selected Pipe models through `generate_function_chat_completion()`.
- A Pipe that defines `pipes` is treated as a manifold. Open WebUI calls it without `request` or `user`, and exposes returned submodels as `{pipe_function_id}.{sub_pipe_id}`.
- `generate_function_chat_completion()` resolves the function id by splitting `body["model"]` at the first dot. The full selected wrapper model id remains in `body["model"]`, so the Pipe can decode the sub-pipe suffix to recover the target model.
- Pipe signatures may receive `__request__`, `__user__`, `__metadata__`, `__event_emitter__`, and `__tools__`.
- `generate_function_chat_completion()` removes `metadata` from `form_data` before calling the Pipe, so the Pipe must rebuild forwarded metadata from `__metadata__`; it must not assume `body["metadata"]` exists.
- Open WebUI checks access before normal model dispatch. Inner target calls may use `bypass_filter=True` only after the Pipe has explicitly validated that the current user may access the decoded target model, because generated manifold wrapper visibility is not the same thing as target-model access. Summary calls are an admin-controlled service dependency and must never take model ids from user payloads.
- Open WebUI propagates `bypass_filter` through shared `request.state`. Every inner target or summary call that sets bypass state must snapshot and restore the prior request-state value in a scoped cleanup path, including failure and streaming-close paths.
- Pure generated manifold submodels without DB-backed Model info are visible only to admins in normal model filtering. v1 must therefore include wrapper Model record/access grant synchronization, or it would be effectively admin-only.
- Native tool calling is handled outside the Pipe by `streaming_chat_response_handler()`. After tool results are appended, Open WebUI recursively calls `generate_chat_completion()` using the original selected model id, so a selected compact Pipe is re-entered on every tool-loop LLM call.
- Open WebUI normalizes provider usage into `input_tokens`, `output_tokens`, and `total_tokens` via `normalize_usage()` and persists usage on completed assistant messages when the provider reports it.
- Open WebUI's LLM payload loading does not reliably preserve persisted `usage` in the forwarded `messages` list. A Pipe must not assume message payloads contain prior usage; it must use request-scoped stream capture first and a read-only lookup of Open WebUI persisted chat/message state when cross-turn usage is needed.
- During an in-flight native tool loop, the recursive message list may not include the current assistant tool-call response usage. The Pipe must bridge this by wrapping the inner target streaming response, yielding chunks unchanged, extracting provider usage chunks, and storing the latest normalized usage in `request.state` for the same HTTP request.
- Open WebUI runtime DB access uses async SQLAlchemy. Sync sessions are intended for startup/config paths.
- Open WebUI extensions do not ship Alembic migrations, so the Pipe must create its own table idempotently at runtime.
- Open WebUI normalized chat-message loading keeps a narrow LLM payload shape, so extension-only message fields are not a reliable checkpoint store.
- During native tool loops, the same assistant message id can receive additional output items over time. Message ids are not stable validity keys, and v1 must not store best-effort message anchors that are not used for lookup.

## Non-Goals

- Do not modify Open WebUI core table schemas or migrations. DB-backed wrapper Model records may be created or updated only through existing Open WebUI model/access APIs so normal users can see and select authorized wrapper models.
- Do not store raw historical message prefixes in the checkpoint table.
- Do not store checkpoint state in `chat.meta`, `chat.summary`, assistant `output`, or custom message fields.
- Do not replace Open WebUI persisted chat history with compacted history.
- Do not reimplement tool execution, citation handling, file handling, MCP handling, or UI event persistence inside the Pipe.
- Do not support global monkey-patching as the primary path.
- Do not accept `target_model` from request payloads, custom params, or `UserValves`; it must be decoded only from the selected manifold wrapper id.
- Do not accept `summary_model` from request payloads, custom params, or `UserValves`; the optional summary override is admin-controlled `Valves` only.
- Do not use serialized character counts as a compaction trigger.
- Do not use `tiktoken` or another model-specific tokenizer as the default trigger source.
- Do not add arbitrary generated-model count caps. The manifold should cover all eligible models unless include/exclude filters intentionally narrow the set.
- Do not automatically expire or prune checkpoints in v1. Once a chat becomes too large for the summary model to re-summarize from raw history, a valid checkpoint may be required to continue that chat.
- Do not support non-streaming calls in v1. This Pipe is a streaming-only wrapper for Open WebUI chat/tool-loop usage.

## Model Exposure Strategy

Use a manifold Pipe, not one manually configured Pipe per real model.

- Wrapper id format: `auto_compact.<target model id>`.
- The sub-pipe id must keep target model ids human-readable and must round-trip ids containing dots, slashes, colons, and provider prefixes.
- The wrapper list must be generated from currently known Open WebUI model caches and filtered by admin Valves.
- Wrapper Model records must be synchronized for generated wrappers so normal users can see wrappers through Open WebUI's normal model filtering.
- Because Open WebUI calls `pipes()` without `request` or `user`, v1 must treat the Function id as a required install contract: the Pipe must be imported with the base Function id `auto_compact`. The Pipe may then use that id to read its Function owner from Open WebUI's Functions table when synchronizing wrapper Model records.
- `pipes()` must not call request-bound model refresh helpers. It may read process-local app state caches from `open_webui.main.app.state`, specifically currently loaded `MODELS`, `BASE_MODELS`, and provider caches such as `OPENAI_MODELS` and `OLLAMA_MODELS`, and must tolerate missing or cold caches by returning an empty wrapper list.
- Wrapper Model synchronization must be best-effort and narrowly scoped to this Pipe's generated wrapper ids. It must use the Function owner's `user_id` as the DB Model owner, must create/update wrapper records through existing Model and access-grant APIs, and must not call destructive global model sync routines that could remove unrelated Model records.
- Wrapper visibility records created by this Pipe are access-control overrides for generated wrapper ids, not Workspace/preset models. They must use the full wrapper model id as the record id and must not create a separate Workspace model based on the decoded target. User-created Workspace models may independently use a compact wrapper as their base model; Open WebUI resolves those to the wrapper id before Pipe dispatch, and v1 must not manage them.
- Wrapper access grants must be equivalent to or stricter than the decoded target model's grants; the Pipe must still re-check target access at runtime before any bypassed inner call.
- If Open WebUI model caches are cold, the manifold may temporarily return no wrappers until model refresh. This is acceptable for v1.
- Exclude this Pipe's own generated wrapper models and arena models from target candidates. Do not exclude unrelated Pipe models solely because they are Pipe-backed; they may be wrapped when they pass include/exclude filters and access validation.

## Admin Valves

Define the complete v1 Valve surface in the Pipe. Do not add additional Valves during implementation unless a task explicitly requires the new behavior.

- `model_name_prefix`: display prefix for generated wrapper models.
- `include_model_patterns`: comma-separated shell-style patterns matched against model id and display name.
- `exclude_model_patterns`: comma-separated shell-style patterns matched against model id and display name.
- `summary_model`: optional admin-selected Open WebUI model for summarization; empty means use the selected target model. Pipe-backed summary models are allowed. Arena models are not valid summary models.
- `trigger_total_tokens`: global provider-usage threshold for starting compaction.
- `force_include_usage`: when enabled, the copied inner streaming target request must ask OpenAI-compatible providers to include usage chunks. Disable only for providers that reject this request option.
- `historical_message_excerpt_bytes`: maximum UTF-8 bytes retained from the middle-truncated text of each historical user message inserted alongside the AI-generated summary.
- `historical_message_excerpt_count`: maximum number of recent historical user messages inserted as middle-truncated excerpts. The Pipe selects the most recent historical user messages before the latest user message, then renders those selected excerpts in chronological order. `0` disables mechanical historical excerpts.

## Compaction Trigger Strategy

Use provider-reported usage as the proactive trigger, not character counts. Use provider context-window errors as the reactive trigger when usage was missing or underestimated the next request.

- Admin Valve: `trigger_total_tokens`.
- Compare provider-reported `total_tokens` with `trigger_total_tokens`.
- If `total_tokens` is absent but both `input_tokens` and `output_tokens` are available, use their sum.
- Do not try to discover every model context window in v1.
- If no provider usage is available before a target call, the first attempt may forward unchanged. If that attempt returns a context-window error, the Pipe must treat the error as a retryable compaction trigger.

Usage lookup order:

1. Latest in-flight provider usage captured by the Pipe pass-through stream wrapper in `request.state`.
2. Latest provider usage read from Open WebUI persisted chat/message state for the current chat and branch, without modifying core tables.
3. Provider usage already present on the forwarded message payload, if a future or provider-specific path includes it.

Tool-loop usage bridge requirements:

- The Pipe must apply the same runtime request-shaping principle used by `functions/filter/token_usage_display.py`: on copied streaming target requests, preserve existing stream options and request usage chunks when the provider supports that option.
- The Pipe must wrap only streaming responses from the inner target call.
- The wrapper must yield the exact original stream bytes/chunks unchanged.
- The wrapper must detect common usage payloads from Chat Completions streams and Responses API streams.
- The wrapper must store normalized usage in request-scoped state using a key scoped to the current chat/message/wrapper model.
- The next recursive Pipe invocation in the same native tool loop must consult this request-scoped usage before looking at persisted message usage.
- The wrapper must clean up the inner response iterator and background cleanup path even when the client disconnects or stream consumption stops early.

Context-overflow retry requirements:

- The Pipe must detect retryable context-window failures from provider exceptions, pre-stream error responses, and the first observed streaming error event when no user-visible target output has been emitted.
- Classifier inputs must include Open WebUI response shapes that can hide the original provider exception class: `JSONResponse` status/body, `PlainTextResponse` status/body, `HTTPException.status_code` and `detail`, and SSE `{"error": ...}` payloads observed by the stream wrapper.
- Context-error classification must be conservative and ordered: inspect structured fields first, then apply limited message fallback only to error message fields.
- Structured inspection must prefer fields such as HTTP status, provider exception class, `error.code`, `error.type`, `error.param`, `error.message`, top-level `detail`, and top-level `message`.
- Explicit context-window codes such as `context_length_exceeded` are the strongest signal.
- Message fallback must inspect only error message/detail strings, not arbitrary serialized response bodies. It should cover clear context-window wording such as maximum context length, context window, prompt/input too long, too many input tokens, and request too large.
- Broad token wording must be treated carefully. For example, token-per-minute, quota, and rate-limit messages must not be classified as context-window errors.
- Message fallback should normally be gated by a compatible client-error status such as 400 or 413, unless a structured provider code already proves context overflow.
- Retry must be bounded by an internal fixed attempt limit, not a new Valve.
- Retry must not be attempted for authentication, permission, rate-limit, unavailable-model, malformed-request, or other non-context errors.
- For streaming target calls, v1 must not fully buffer the stream to enable retry. This would break streaming UX, delay Open WebUI tool-call handling, and make the Pipe too stateful.
- The stream wrapper may inspect the first SSE event before yielding it. If the first event is a retryable context error and no user-visible chunks have been yielded, retry is allowed.
- After the first non-error event, or after any target content, reasoning delta, or tool-call delta has been yielded, streaming error chunks must be captured for status/diagnostics but must not trigger an internal retry in v1.
- If any target content, reasoning delta, or tool-call delta has already been emitted to the client, the Pipe must not silently restart that call; it should surface the error and status because the response is no longer cleanly retryable.
- A retry must remove provider state such as `previous_response_id` when compaction changes the message payload.
- Compaction always uses the same boundary: the first system message is preserved separately, every message before the latest user message is summarized, and the latest user message remains raw. Tool-call/result rounds after that latest user message are never retained as `assistant`/`tool` protocol messages after compaction; they are replaced by user-role tool-result summary messages.
- Retry compaction must not silently summarize or replace the active latest user message. If the latest user message itself is too large to send raw after all safe compaction options are exhausted, the Pipe must fail clearly instead of rewriting the user's request.
- Summary creation must always resolve an exact or longest matching parent checkpoint before calling the summary model. If a parent checkpoint exists, the summary model receives the parent summary plus only the delta messages after that parent, including tool-loop summaries. Dynamic adaptive summary splitting is out of v1 scope. If ordinary message summary extension fails after a verified parent checkpoint is found, the Pipe may continue the current target call with a checkpoint-derived payload containing the parent summary plus the verified raw delta and tail messages. This is not raw direct forwarding: the already summarized prefix must stay replaced by the checkpoint summary, and the Pipe must not insert a full-prefix checkpoint that was not actually generated. Tool-result compaction, unsafe deltas, and active inputs that cannot be sent safely must return a clear error instead of inventing a lossy split strategy or bypassing the checkpoint.

## Persistence Strategy

Use one extension-owned table:

- `skyzi000_owui_ext_autocompact_checkpoint_v1`

Use extension-owned indexes:

- `skyzi000_owui_ext_accp_v1_lookup_uq`
- `skyzi000_owui_ext_accp_v1_prefix_idx`
- `skyzi000_owui_ext_accp_v1_recent_idx`

The table stores derived durable compaction state, not raw chat source of truth. Open WebUI chat history remains canonical, but valid checkpoints may become operationally necessary for continuing very long chats after the original prefix exceeds the summary model context window. Checkpoints must support both exact reuse and rolling extension: whenever a current summarization prefix has an exact checkpoint or a matching parent checkpoint, the Pipe must summarize or compact from that checkpoint plus only the new delta messages after it, instead of resubmitting the already summarized raw prefix. Checkpoint storage is not an optional cache: if checkpoint lookup, initialization, validation, or required persistence fails for a durable chat context, the Pipe must fail clearly instead of silently forwarding a request that may destroy the continuation path.

Use SQLAlchemy Core table metadata local to the Pipe module. Do not register a Declarative ORM class from a reloadable Pipe. Do not add foreign keys in v1.

Required checkpoint columns:

- `id`
- `namespace`
- `schema_version`
- `user_id`
- `chat_id`
- `pipe_function_id`
- `profile_hash`
- `source_message_count`
- `source_hash`
- `summary_text`
- `summary_meta`
- `state`
- `parent_checkpoint_id`
- `created_at`
- `updated_at`
- `last_used_at`

Required unique lookup key:

- `namespace`
- `user_id`
- `chat_id`
- `pipe_function_id`
- `profile_hash`
- `source_hash`

`profile_hash` is a checkpoint compatibility partition, not a prompt-version invalidation key. In v1 it must remain stable across normal implementation updates so existing checkpoints remain usable for long-chat continuation. It may include only hard compatibility boundaries that the implementation cannot read safely, such as:

- checkpoint table/schema compatibility family
- summary text format compatibility family
- source-hash canonicalization compatibility family, only when legacy source-hash lookup cannot be supported

`profile_hash` must not include selected target model, summary model, wrapper model suffix, cut-selection settings, summary prompt wording, or ordinary summary-policy changes. Checkpoint applicability is determined by the exact summarized prefix and the Pipe's ability to consume the stored `summary_text`, so a user switching target models mid-chat should still be able to reuse an existing checkpoint when `source_hash` and `profile_hash` match.

`summary_model` affects only generation on a checkpoint miss. Once a checkpoint is ready, changing `summary_model` or changing the prompt used for future summaries must not invalidate that `summary_text`. If a future implementation introduces a truly incompatible summary format, it should add a legacy reader or a new table/schema version rather than casually changing the v1 `profile_hash` and breaking continuation of existing long chats.

`pipe_function_id` in checkpoint keys means the base Pipe function id before the manifold suffix, not the full selected wrapper id. The target model suffix must not indirectly scope checkpoint reuse.

`source_hash` must be computed from a canonical JSON serialization of the exact stable prefix being summarized. Canonicalization must avoid transient values such as signed URLs, volatile file metadata, generated temporary ids, and provider-specific decoration that is not semantically part of the conversation. If an active multimodal/file reference cannot be canonicalized safely, keep it with the latest user turn or skip compaction.

Parent checkpoint selection must be exact, not approximate. To extend a checkpoint, the Pipe must choose the longest ready checkpoint in the same namespace, user, chat, Pipe Function id, and profile whose `source_hash` matches a canonical prefix of the current summarization prefix. `source_message_count` may be used to order candidates, but it must not be trusted without `source_hash` validation. `parent_checkpoint_id` is nullable for base checkpoints and is set when a newly inserted checkpoint is derived from an existing checkpoint.

Do not store message ids or output indexes in v1. They are not needed for the planned lookup key or rolling checkpoint extension.

## Data Flow

1. Pipe receives `body`, `__metadata__`, `__request__`, and `__user__`.
2. Pipe deep-copies `body` into an inner request payload.
3. Pipe resolves `pipe_function_id` and `target_model` by stripping the selected manifold wrapper prefix from original `body["model"]`.
4. Pipe rejects malformed wrapper ids, unknown target models, this Pipe's own generated wrapper models, arena models, and target models the current user may not access.
5. Pipe restores forwarded metadata from `__metadata__`.
6. Pipe sets only the copied inner payload model to `target_model`.
7. Pipe rejects non-streaming requests in v1 and forwards streaming requests as streaming target calls.
8. Pipe requests provider usage on copied streaming target requests when `force_include_usage` is enabled.
9. Pipe skips compaction for unsupported contexts: no `chat_id`, temporary `local:` or `channel:` chats, and known internal tasks including its own summary task.
10. Pipe reads provider usage from request-scoped in-flight usage first, then from Open WebUI persisted chat/message state, then from message payload usage if present.
11. Pipe chooses the safe message cut points needed to determine whether an exact checkpoint or parent checkpoint already matches the current chain, even when usage is absent or below `trigger_total_tokens`.
12. Pipe excludes the preserved first system message from any summarization prefix.
13. Pipe computes `source_hash`, prefix hashes, and `profile_hash`.
14. Pipe initializes the checkpoint table idempotently; failure to access required checkpoint state in a durable chat context is a clear error, not a raw-forward fallback.
15. Pipe first checks for exact checkpoint reuse and then longest matching parent checkpoints for tool-loop and ordinary message cuts.
16. If usage is absent or below threshold and no exact or parent checkpoint exists anywhere in the current cut chain, Pipe may make a first target attempt with unchanged copied messages, scoped request-state restoration, and streaming response wrapping for usage and context-error capture.
17. If usage is absent or below threshold but an exact or parent checkpoint exists anywhere in the current cut chain, Pipe applies that checkpoint and forwards the checkpoint-derived compacted payload without creating a new summary.
18. If usage is at or above `trigger_total_tokens`, Pipe enters a per-process generation lock for the namespace/user/chat/pipe/profile/source hash.
19. Inside the lock, Pipe looks up an existing ready checkpoint by source hash and profile hash, independent of selected target model and summary model.
20. On exact hit, Pipe updates `last_used_at` and reuses `summary_text`.
21. On miss, Pipe searches for the longest exact parent checkpoint that represents a canonical prefix of the desired summarization prefix.
22. If a parent checkpoint exists, Pipe calls the summary model with the parent summary as compressed historical context plus only the delta messages after the parent prefix.
23. Summary calls must be marked with summary-task metadata, must not request tools, must preserve and restore request-scoped bypass state, and may target a Pipe-backed summary model as long as the summary task cannot recursively compact itself.
24. If no parent checkpoint exists, Pipe calls the summary model with the raw summarization prefix and lets the compaction failure behavior handle summary-context overflow.
25. Pipe inserts the checkpoint for the full desired prefix before relying on that checkpoint as durable continuation state. If it was derived from a parent checkpoint, the row records that parent checkpoint id. If a concurrent insert wins elsewhere, Pipe fetches and uses the existing ready checkpoint.
26. Pipe replaces the old prefix with the preserved system message, a compact user-role summary message containing both AI summary text and chronological middle-truncated historical user excerpts, and the raw latest user message.
27. Pipe removes `previous_response_id` only when compaction was applied.
28. Pipe forwards the compacted copied payload to the target model with scoped request-state restoration and wraps any streaming response for usage and context-error capture.
29. If a wrapped target call fails with a retryable context-window error before any target output is emitted, Pipe compacts with the same latest-user boundary and retries within the fixed retry budget.
30. If retry compaction succeeds, Pipe returns the retried streaming response as the user-visible response and stores any reusable checkpoint state created along the way.

## Message Compaction Rules

- Preserve the first system message unchanged.
- Insert the summary as a `role: "user"` message immediately after the first system message, or at index 0 when there is no system message.
- The summary message must be XML-tag structured under `<auto_compaction_context>`, with the AI-generated checkpoint summary inside `<checkpoint_summary>` and mechanical historical user excerpts inside `<historical_user_messages>`.
- Arbitrary summary, excerpt, and tool-result-summary text must be wrapped in CDATA sections, escaping `]]>` by splitting CDATA sections.
- The summary message must state inside an `<instruction>` tag that it is compressed historical context and not a new instruction.
- Do not include the preserved first system message in the summarization prefix.
- The latest user message must remain raw.
- Historical user messages in the summarized prefix must also appear as chronological middle-truncated excerpts inside the summary message, bounded by `historical_message_excerpt_bytes` per message and `historical_message_excerpt_count` total messages.
- Do not retain tool result messages as `role: "tool"` after compaction. Complete assistant/tool rounds after the latest user message must be replaced by user-role `<auto_compacted_tool_results>` summary messages.
- If the latest user message itself is the reason the request cannot fit after all safe compaction options are exhausted, the Pipe must return a clear error rather than silently rewriting it.
- Treat multimodal content conservatively. If summarized history contains images or file-derived content, record that fact in summary metadata and keep recent active image references in the tail.

## Files To Create

- Create: `functions/pipe/auto_compaction_pipe.py`
  - Single-file Open WebUI Pipe function for initial delivery.
  - Contains the Pipe, Valves, manifold model enumeration, usage trigger handling, stream usage capture, context-error retry handling, DB helpers, checkpoint CRUD, safe-cut logic, summary generation, and forwarding behavior.
- Create: `tests/pipes/test_auto_compaction_checkpoint.py`
  - Unit tests for checkpoint identity, checkpoint profile/source hashing, row payload contracts, and table metadata.
- Create: `tests/pipes/test_auto_compaction_manifold.py`
  - Unit tests for target model id encoding, target resolution, model filtering, and usage extraction/lookup behavior.
- Create: `tests/pipes/test_auto_compaction_messages.py`
  - Unit tests for safe-cut behavior, summary insertion, preserved system messages, and tool-pair preservation.

Optional later builder integration:

- Create: `src/owui_ext/pipes/auto_compaction_pipe.py`
- Modify: `release.toml`
- Generate: `functions/pipe/auto_compaction_pipe.py`

Do not start with builder integration unless the single-file proof works.

---

## Task 1: Establish Checkpoint And Manifold Identity Contracts

**Files:**

- Create: `functions/pipe/auto_compaction_pipe.py`
- Test: `tests/pipes/test_auto_compaction_checkpoint.py`

- [ ] Write tests that prove source hashes are deterministic for equivalent canonical message payloads.
- [ ] Write tests that prove source hash canonicalization ignores transient file and multimodal metadata while preserving stable semantic content.
- [ ] Write tests that prove profile hashes change only when hard checkpoint compatibility boundaries change.
- [ ] Write tests that prove profile hashes do not change for summary prompt wording, target model, summary model, wrapper suffix, or cut-policy changes.
- [ ] Write tests that prove target model ids with special characters round-trip through manifold wrapper ids.
- [ ] Write tests that prove selected wrapper ids split into the Pipe Function id and decoded target model id.
- [ ] Run the checkpoint test file and confirm it fails before implementation.
- [ ] Implement only the primitives needed to satisfy these tests.
- [ ] Run the checkpoint test file and confirm it passes.

## Task 2: Add Idempotent Checkpoint Table Initialization

**Files:**

- Modify: `functions/pipe/auto_compaction_pipe.py`
- Modify: `tests/pipes/test_auto_compaction_checkpoint.py`

- [ ] Add tests that verify the checkpoint table uses the required `skyzi000` table name and index names.
- [ ] Add tests that verify the table columns match the v1 persistence contract.
- [ ] Add tests that verify schema initialization is request-state cached when possible and process cached otherwise.
- [ ] Implement SQLAlchemy Core table metadata and async idempotent table creation.
- [ ] Verify table initialization uses Open WebUI async DB primitives.
- [ ] Run checkpoint tests and compile the Pipe file.

## Task 3: Add Conservative Safe-Cut Message Logic

**Files:**

- Modify: `functions/pipe/auto_compaction_pipe.py`
- Create: `tests/pipes/test_auto_compaction_messages.py`

- [ ] Add tests that verify compaction keeps the latest user message raw and does not retain tool-role messages.
- [ ] Add tests that verify assistant tool calls and corresponding tool results are not split.
- [ ] Add tests that verify orphan tool result messages are not retained.
- [ ] Add tests that verify historical user messages are inserted as chronological middle-truncated excerpts.
- [ ] Add tests that verify complete assistant/tool rounds after the latest user message are replaced by user-role summary messages.
- [ ] Implement the safe-cut selector.
- [ ] Run message tests.

## Task 4: Add Summary Rendering And Message Replacement

**Files:**

- Modify: `functions/pipe/auto_compaction_pipe.py`
- Modify: `tests/pipes/test_auto_compaction_messages.py`

- [ ] Add tests that verify the first system message is preserved unchanged.
- [ ] Add tests that verify the summary is inserted only as a user-role historical context message.
- [ ] Add tests that verify no system message is created or modified for summary insertion.
- [ ] Add tests that verify the latest user message is appended unchanged after the summary message.
- [ ] Implement summary rendering and prefix replacement.
- [ ] Run message tests.

## Task 5: Add Checkpoint Row Construction And CRUD

**Files:**

- Modify: `functions/pipe/auto_compaction_pipe.py`
- Modify: `tests/pipes/test_auto_compaction_checkpoint.py`

- [ ] Add tests for stable checkpoint ids and row payload fields.
- [ ] Add tests for ready-state lookup behavior.
- [ ] Add tests for longest exact parent checkpoint lookup when extending a longer prefix.
- [ ] Add tests that a parent candidate with matching message count but mismatched `source_hash` is not used.
- [ ] Add tests for concurrent insert conflict behavior.
- [ ] Implement checkpoint row construction, exact lookup, parent lookup, insert, and last-used update behavior.
- [ ] Run checkpoint tests and compile the Pipe file.

## Task 6: Add Manifold Model Listing, Usage Trigger, And Forwarding Skeleton

**Files:**

- Modify: `functions/pipe/auto_compaction_pipe.py`
- Test: `tests/pipes/test_auto_compaction_manifold.py`

- [ ] Add tests for excluding this Pipe's own generated wrappers and arena models from manifold targets while allowing unrelated Pipe-backed models.
- [ ] Add tests for include/exclude pattern behavior.
- [ ] Add tests for duplicate target model handling.
- [ ] Add tests that prove generated wrapper models have DB-backed Model info required for normal-user visibility.
- [ ] Add tests that prove wrapper visibility synchronization creates access-control overrides for full wrapper ids, not Workspace/preset models based on decoded targets.
- [ ] Add tests that prove wrapper synchronization uses the `auto_compact` Function owner's `user_id` as the wrapper Model owner.
- [ ] Add tests that prove wrapper synchronization is scoped to generated wrapper ids and does not call destructive global model sync behavior.
- [ ] Add tests that prove wrapper access grants are not broader than the decoded target model's access grants.
- [ ] Add tests that prove a user without target access cannot reach that target through a wrapper when the inner call uses `bypass_filter=True`.
- [ ] Add tests that verify configured `summary_model` may be Pipe-backed, must be admin-controlled, and must not be an arena model.
- [ ] Add tests for provider usage extraction from common stream payload shapes.
- [ ] Add tests for context-window error classification from common provider exception and pre-stream error response shapes.
- [ ] Add tests for classifier inputs from `JSONResponse`, `PlainTextResponse`, `HTTPException`, and SSE `{"error": ...}` payloads.
- [ ] Add tests that structured context-error fields take precedence over message fallback.
- [ ] Add tests that message fallback is limited to error message/detail fields and compatible client-error statuses.
- [ ] Add tests that non-context provider errors are not classified as retryable compaction triggers.
- [ ] Add tests that rate-limit, quota, and token-per-minute messages are not misclassified as context-window errors.
- [ ] Add tests for request-scoped usage taking precedence over persisted message usage.
- [ ] Add tests that persisted usage lookup can read the current branch from Open WebUI's normalized `chat_message` path.
- [ ] Add tests that persisted usage lookup handles the legacy embedded chat-history fallback when normalized rows are absent or incomplete.
- [ ] Add tests for forwarding unchanged when usage is absent or below threshold only after verifying no exact or parent checkpoint matches the current chat-history chain.
- [ ] Implement the complete v1 Admin Valve surface listed in this plan, without adding extra knobs.
- [ ] Implement manifold listing from the allowed Open WebUI process-local app-state model caches, tolerating cold caches.
- [ ] Implement wrapper Model record and access grant synchronization through existing Open WebUI APIs as generated-wrapper access-control overrides, not as Workspace/preset models.
- [ ] Implement target model resolution from selected wrapper ids.
- [ ] Implement target model access validation before bypassed inner target forwarding.
- [ ] Implement summary model validation before any summary call.
- [ ] Implement usage option injection on copied streaming target requests.
- [ ] Implement request-scoped usage capture, conservative context-error classification, and persisted usage lookup.
- [ ] Implement scoped request-state restoration around every bypassed target and summary call.
- [ ] Implement the streaming-only forwarding skeleton with `bypass_filter=True`, usage capture, request-state restoration, and pre-emission retryable context-error detection.
- [ ] Run manifold tests and compile the Pipe file.

## Task 7: Add Summary Generation And Checkpoint Reuse

**Files:**

- Modify: `functions/pipe/auto_compaction_pipe.py`
- Modify: `tests/pipes/test_auto_compaction_checkpoint.py`
- Modify: `tests/pipes/test_auto_compaction_messages.py`

- [ ] Add tests that verify the preserved first system message is excluded from the summarization prefix.
- [ ] Add tests that verify summary calls are marked as internal summary tasks.
- [ ] Add tests that verify summary calls do not request tool execution.
- [ ] Add tests that verify Pipe-backed summary models do not recursively compact summary-task calls.
- [ ] Add tests that verify request-state bypass flags are restored after summary calls, target calls, exceptions, and streaming response cleanup.
- [ ] Add tests that verify a checkpoint hit avoids a summary model call.
- [ ] Add tests that verify a checkpoint miss generates a summary and stores a ready checkpoint.
- [ ] Add tests that verify a longer-prefix checkpoint miss uses the longest matching parent checkpoint summary plus delta messages, not the raw already-summarized prefix.
- [ ] Add tests that verify extended checkpoints store the full new prefix hash and parent checkpoint lineage.
- [ ] Add tests that verify ordinary message summary-extension failures use the verified parent checkpoint plus safe raw delta fallback without inserting a full-prefix checkpoint.
- [ ] Add tests that verify dynamic adaptive summary splitting is not used in v1.
- [ ] Add tests that verify a latest tool result that alone exceeds the summary model capacity returns a clear error.
- [ ] Add tests that verify oversized retained tool results can be compacted during context-error retry while preserving valid tool-call/result structure.
- [ ] Add tests that verify the active latest user message is never silently summarized or replaced during retry compaction.
- [ ] Add tests that verify an oversized active latest user message returns a clear error when it cannot be sent raw.
- [ ] Add tests that verify checkpoint reuse survives target model and summary model changes when source hash and profile hash match.
- [ ] Add tests that verify below-threshold requests still reuse an exact or parent checkpoint when one exists in the current chat-history chain.
- [ ] Add tests that verify tool-loop summary creation always extends an available parent checkpoint and never exposes an opt-out path for parent checkpoint use.
- [ ] Add tests that verify compacted requests remove `previous_response_id`.
- [ ] Implement summary generation, checkpoint compatibility handling, exact checkpoint reuse, rolling checkpoint extension, and compacted forwarding.
- [ ] Run all Pipe tests and compile the Pipe file.

## Task 8: Add Continuation-Preserving Failure Behavior

**Files:**

- Modify: `functions/pipe/auto_compaction_pipe.py`
- Modify: `tests/pipes/test_auto_compaction_checkpoint.py`

- [ ] Add tests that verify required checkpoint storage failures fail clearly instead of forwarding without durable continuation state.
- [ ] Add tests that verify exact checkpoint hits are still used when non-critical checkpoint update operations fail.
- [ ] Add tests that verify summary-extension failures never raw-forward the full history: ordinary safe deltas may use parent-checkpoint plus raw delta fallback, while tool-result compaction and unsafe deltas return a clear error.
- [ ] Add tests that verify missing usage permits only the first unchanged attempt and that a resulting context-window error triggers compaction and retry.
- [ ] Add tests that verify pre-stream context-window errors and first-event SSE context errors are retried before any user-visible chunks are emitted.
- [ ] Add tests that verify streaming error chunks observed after stream body iteration starts are surfaced and not internally retried in v1.
- [ ] Add tests that verify retry stops at a fixed internal retry budget and does not retry non-context errors.
- [ ] Add tests that verify raw direct forwarding is limited to cases where compaction is not required and no exact or parent checkpoint exists in the current chat-history chain.
- [ ] Add tests that verify the allowed first unchanged target attempt still captures stream usage.
- [ ] Implement conservative context-window error detection, pre-emission compact-and-retry behavior, failure handling, and status emission without treating checkpoints as optional cache.
- [ ] Run all Pipe tests and compile the Pipe file.

## Task 9: Manual Open WebUI Verification

**Files:**

- Use: `functions/pipe/auto_compaction_pipe.py`

- [ ] Import the Pipe in Open WebUI as a Pipe Function with the Function id `auto_compact`.
- [ ] Configure the v1 Valves listed in this plan.
- [ ] Enable native function calling for the wrapper model or request params.
- [ ] Verify a basic below-threshold chat with no applicable checkpoint forwards to the decoded target model and emits no compaction status.
- [ ] Verify a below-threshold chat with any applicable exact or parent checkpoint uses that checkpoint before forwarding.
- [ ] Verify manifold wrapper generation covers the expected real models without manual per-model Pipe configuration.
- [ ] Verify a non-admin user can see and use authorized wrapper models through normal Open WebUI model filtering.
- [ ] Verify a non-admin user cannot use a wrapper to reach a target model they cannot directly access.
- [ ] Verify wrapper Model records are created with the `auto_compact` Function owner and do not alter unrelated Model records.
- [ ] Verify non-streaming API requests fail with a clear unsupported-mode error.
- [ ] Verify above-threshold compaction preserves Open WebUI persisted chat history unchanged.
- [ ] Verify the forwarded provider input preserves the original system message and inserts summary as user-role historical context.
- [ ] Verify the checkpoint table receives a ready checkpoint row.
- [ ] Verify repeating the same long prefix reuses the existing checkpoint and updates `last_used_at`.
- [ ] Verify extending the same chat beyond an existing checkpoint creates a new checkpoint from the parent summary plus delta messages.
- [ ] Verify switching target models mid-chat can reuse an existing checkpoint when the summarized prefix and summary contract are unchanged.
- [ ] Verify native tool loops re-enter the Pipe.
- [ ] Verify a huge tool result that causes a pre-emission target-model context-window error is compacted and retried without ending the chat.
- [ ] Verify a first-event SSE context error is retried before any client-visible chunk is emitted.
- [ ] Verify a streaming provider error that arrives after output has begun is surfaced rather than silently retried.
- [ ] Verify summary creation always uses an available parent checkpoint; if ordinary message extension fails, safe raw delta fallback still forwards a checkpoint-derived payload, while tool-result or unsafe-delta failures return a clear error.
- [ ] Verify a Pipe-backed `summary_model` can be used without recursively compacting the summary task.
- [ ] Verify `request.state.bypass_filter` does not remain changed after target or summary forwarding.
- [ ] Verify a latest tool result that exceeds the summary model capacity returns a clear error.
- [ ] Verify non-context provider errors are not retried as compaction failures.
- [ ] Verify usage captured from one target stream is available to a later recursive Pipe call in the same tool loop.
- [ ] Verify tool result messages are not split from their assistant tool calls.

## Verification Commands

- `uv run --no-sync pytest tests/pipes/test_auto_compaction_checkpoint.py tests/pipes/test_auto_compaction_manifold.py tests/pipes/test_auto_compaction_messages.py -q`
- `uv run --no-sync python -m py_compile functions/pipe/auto_compaction_pipe.py`

## Risk Register

- **DB DDL race:** Multiple workers may initialize the table at the same time. Mitigation: use idempotent table creation, per-process locking, and tolerate duplicate-create races if observed.
- **Provider state collision:** If compaction is applied while `previous_response_id` is present, client-side summary and provider-side state can diverge. Mitigation: remove `previous_response_id` only when compaction is applied.
- **Access control:** `bypass_filter=True` skips direct access checks for decoded `target_model`. Mitigation: explicitly validate current-user access to the decoded target model before the bypassed inner target call, keep summary model ids admin-controlled, never accept them from user payloads, and restore request-scoped bypass state after every inner call.
- **Wrapper visibility:** Pure generated manifold wrappers are admin-only under normal model filtering when no DB Model info exists. Mitigation: v1 synchronizes DB-backed wrapper Model records/access grants through existing Open WebUI APIs using the `auto_compact` Function owner and still validates target access at runtime.
- **Manifold cache cold start:** Open WebUI calls manifold listing without `request`, so dynamic enumeration reads process-local app-state caches. Mitigation: use only allowed app-state cache reads, tolerate an empty list before model cache warm-up, and verify behavior after model refresh.
- **Wrapper sync side effects:** Synchronizing wrapper records from manifold listing could accidentally alter unrelated Model records if implemented with broad sync APIs. Mitigation: create/update only generated wrapper ids through narrow Model/access-grant operations and never call destructive global model sync routines from this Pipe.
- **Message id staleness:** A native tool loop can grow the same assistant message over multiple LLM calls. Mitigation: do not store or trust message ids in v1 checkpoint rows; require `source_hash` for validity.
- **Usage unavailable:** Some providers or stream modes may not report usage. Mitigation: request usage chunks on copied streaming target calls when enabled, capture provider usage when present, store it in request-scoped state for native tool-loop recursion, read persisted usage for later turns, and use context-window errors as the reactive compaction trigger when usage is unavailable.
- **Context-error classification gaps:** Providers may report context overflow with different status codes, error codes, or stream payloads. Mitigation: prefer structured error fields, gate message fallback by compatible client-error status, avoid retrying non-context errors, and include manual/provider-specific verification for the configured deployments.
- **Retry after partial output:** Once target output has been emitted to the client, an internal retry would duplicate or corrupt the stream. Mitigation: retry only before user-visible output begins, do not fully buffer normal streams, and surface streaming errors that arrive after body iteration has started.
- **Oversized active input:** The latest user message or latest tool result can itself exceed available context or summary-model capacity. Mitigation: never rewrite the active user request silently; compact oversized tool results only when the summary model can safely process them, and return a clear error otherwise.
- **Summary authority:** Putting user/tool history into a system message can elevate historical text. Mitigation: never modify system messages; always inject summaries as user-role historical context.
- **Rolling summary drift:** Repeatedly extending summaries can accumulate omissions or distortions. Mitigation: use the longest exact parent checkpoint, include only verified delta messages after that parent, include bounded chronological middle-truncated historical user excerpts, and keep the summary output contract backward compatible.
- **Parent-delta fallback recurrence:** If ordinary message summary extension fails but the target can accept parent summary plus raw delta, the current turn may continue without creating a new full-prefix checkpoint. Mitigation: apply this fallback only to verified safe ordinary deltas, keep the parent checkpoint as the durable boundary, remove provider state from compacted payloads, and surface target context overflow if the fallback payload cannot be sent safely.
- **Summary quality:** Bad summaries can degrade answer quality. Mitigation: include bounded historical user excerpts alongside the AI summary, keep summary compatibility stable for existing checkpoints, reuse existing ready checkpoints across target/summary model changes, and make compaction threshold conservative by default.
- **Table lifecycle:** Extension-owned tables will not be managed by Open WebUI migrations. Mitigation: version table names and never destructively migrate in place.
- **Privacy:** Summaries may contain sensitive data. Mitigation: do not store raw prefixes and document that summaries are persisted in the same DB as chat data.
- **DB growth:** Checkpoints accumulate outside Open WebUI migrations. Mitigation: store only summaries and compact metadata in v1, keep the table versioned, and defer explicit admin cleanup/export policy until there is real usage data. Do not prune automatically in v1 because doing so can break continuation of long chats.

## Open Decisions Before Implementation

1. Whether the first release should remain a direct single-file Pipe under `functions/pipe/` or immediately integrate with the release builder after proof.

## Recommended v1 Scope

Ship a single-file Pipe with:

- `skyzi000_owui_ext_autocompact_checkpoint_v1`
- manifold wrapper model generation for many real models
- DB-backed wrapper Model/access grant synchronization for normal-user visibility
- persistent checkpoint reuse
- rolling checkpoint extension from existing ready checkpoints plus delta messages
- source-hash validation without unused message anchors
- provider-usage `total_tokens` compaction trigger with one admin-controlled threshold
- runtime provider usage request injection on copied streaming target calls
- request-scoped stream usage capture for native tool-loop recursion
- read-only persisted usage lookup for later turns
- conservative context-window error classification and pre-emission compact-and-retry behavior
- tool-result summary replacement for complete assistant/tool rounds after the latest user message
- conservative safe-cut logic
- user-role summary injection only
- backward-compatible summary consumption
- native tool-loop compatibility
- no builder integration

Defer:

- admin UI beyond Valves
- table migration beyond `_v1`
- checkpoint export/import
- checkpoint cleanup/pruning policy
- dynamic adaptive summary splitting
- background compaction
- replacing Open WebUI canonical chat history
