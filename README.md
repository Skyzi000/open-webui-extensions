# Open WebUI Extensions

[日本語版 / Japanese](README.ja.md)

A collection of tools and filters for Open WebUI.

## 🌟 Highlights

**[Sub Agent Tool](tools/sub_agent.py)** ([openwebui.com](https://openwebui.com/posts/sub_agent_7bfeb0b7)) - **#1 most upvoted** on openwebui.com with **20,000+ downloads**! Featured in Open WebUI's official [Community Newsletter, January 28th 2026](https://openwebui.com/blog/newsletter-january-28-2026) as one of "This Week's Most Useful".

Delegate tool-heavy tasks to sub-agents running in isolated contexts, keeping your main conversation clean and efficient. Fully leverages Open WebUI v0.7+ built-in tools (web search, memory, knowledge bases, etc.).

### What's New

- **v0.6** — Automatic context compaction and large-result previews with selective read-back for sub-agents. Requires Open WebUI 0.9.6+ (previously 0.7.0+). [Details](#sub-agent-context-compaction)
- **v0.5** — MCP servers configured in Open WebUI are directly available to sub-agents — no mcpo proxy needed ([#6](https://github.com/Skyzi000/open-webui-extensions/issues/6))
- **v0.4.5** — Open Terminal tools (Open WebUI v0.8.6+) are automatically forwarded to sub-agents
- **v0.4** — Skills introduced in Open WebUI v0.8 are automatically propagated to sub-agents (experimental)
- **v0.3** — Native parallel sub-agent execution via `run_parallel_sub_agents`

Full changelog: [commits on sub_agent.py](https://github.com/Skyzi000/open-webui-extensions/commits/main/tools/sub_agent.py)

> [!TIP]
> If parallel execution causes issues (e.g., search API rate limits), reduce `MAX_PARALLEL_AGENTS` in Valves, or comment out the `run_parallel_sub_agents` method to disable it entirely.

### Sub-agent context compaction

Tool-heavy tasks can fill the sub-agent's own context with history and results. Starting in v0.6, two mechanisms reduce the context sent to the model to help long-running tasks keep going. Both are enabled by default and can be configured independently in Valves.

- **Automatic history compaction** — When estimated input tokens reach the threshold (default: 80,000), older rounds are summarized while the task and the most recent rounds are kept. By default, summaries use the same model and reuse the previously sent prefix and tool definitions, with care taken to preserve prompt caching where possible.
- **Large tool result previews** — Large results (by default, 10,000 tokens or more, or over 64 KiB) are sent as a short head/tail preview instead of the full text. The original text is retained during the same task, and the sub-agent can read back the lines or ranges it needs through `agent_ref_exec`, using commands such as `head`, `tail`, `sed -n`, and `grep`.

Open WebUI's built-in chat-history compaction does not apply to this tool's internal sub-agent loop. This feature covers that internal history.

| Valve | Purpose |
| ----- | ------- |
| `ENABLE_CONTEXT_COMPACTION` | Enable or disable automatic history compaction (default: on) |
| `CONTEXT_COMPACTION_TOKEN_THRESHOLD` | Estimated input tokens at which compaction starts (default: 80,000) |
| `COMPACTION_SUMMARY_MODEL` | Model used for summaries; leave empty to use the sub-agent's model (recommended) |
| `LARGE_TOOL_RESULT_MODE` | `ref_exec` (preview + read-back, default) / `truncate` (omit the middle, no read-back) / `raw` (send as is) |
| `LARGE_TOOL_RESULT_THRESHOLD_TOKENS` | Token threshold for large results (default: 10,000) |
| `MAX_ITERATIONS` | Iteration cap (default: 50, previously 10; 0 means unlimited) |

**[Parallel Tools](tools/parallel_tools.py)** ([openwebui.com](https://openwebui.com/posts/parallel_tools_1d44cfce)) - Featured in Open WebUI's official [Community Newsletter, March 17th 2026](https://openwebui.com/blog/community-newsletter-march-17th-2026) as one of the "Editor's Picks".

Execute multiple independent tool calls in parallel for faster execution.

## Tools

| Tool | Description |
| ---- | ----------- |
| [**Sub Agent**](tools/sub_agent.py) | Delegate tasks to autonomous sub-agents to keep context consumption low |
| [**Parallel Tools**](tools/parallel_tools.py) | Execute multiple independent tool calls in parallel for faster results (note: often requires a strong flagship model to invoke correctly) |
| [**Multi Model Council**](tools/multi_model_council.py) | Run a multi-model council decision with majority vote |
| [**LLM Review**](tools/llm_review.py) | Multi-persona creative writing that preserves divergent voices — each persona produces a distinctive draft through independent revision and peer feedback, rather than merging into one (inspired by arXiv:2601.08003) |
| [**User Location**](tools/user_location.py) | Get user's current location via the browser's Geolocation API |
| [**Universal File Generator (Pandoc)**](tools/universal_file_generator_pandoc.py) | Generate files in various formats using Pandoc. *Not intended for use by others at this time* |

## Filters

| Filter | Description |
| ------ | ----------- |
| [**Current DateTime Injector**](functions/filter/current_datetime_injector.py) | Inject current datetime into system prompt (implemented as a filter to leverage OpenAI prompt caching) |
| [**User Info Injector**](functions/filter/user_info_injector.py) | Inject user info into system prompt (same reason as above) |
| [**Full Context Mode Toggle**](functions/filter/full_context_mode_toggle.py) | Batch toggle full context mode per chat (the built-in feature only supports per-file toggling) |

## Graphiti Memory (Submodule)

A knowledge-graph-based memory extension powered by [Graphiti](https://github.com/getzep/graphiti).

Managed in a separate repository: [open-webui-graphiti-memory](https://github.com/Skyzi000/open-webui-graphiti-memory). Referenced here via the `graphiti/` submodule.

- Filter: [graphiti_memory.py](https://github.com/Skyzi000/open-webui-graphiti-memory/blob/main/functions/filter/graphiti_memory.py)
- Tool: [graphiti_memory_manage.py](https://github.com/Skyzi000/open-webui-graphiti-memory/blob/main/tools/graphiti_memory_manage.py)
- Action: [add_graphiti_memory_action.py](https://github.com/Skyzi000/open-webui-graphiti-memory/blob/main/functions/action/add_graphiti_memory_action.py)

## Setup

### Initialize Submodules

After cloning this repository, initialize the submodules:

```bash
git submodule init
git submodule update
```

Or clone with submodules in one step:

```bash
git clone --recurse-submodules https://github.com/Skyzi000/open-webui-extensions.git
```

## License

MIT License
