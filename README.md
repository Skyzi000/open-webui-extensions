# Open WebUI Extensions

[日本語版 / Japanese](README.ja.md)

A collection of tools and filters for Open WebUI.

## 🌟 Highlights

**[Sub Agent Tool](tools/sub_agent.py)** ([openwebui.com](https://openwebui.com/posts/sub_agent_7bfeb0b7)) - **#1 most upvoted** on openwebui.com with **1,400+ downloads**! (as of 2026/02/08)

Delegate tool-heavy tasks to sub-agents running in isolated contexts, keeping your main conversation clean and efficient. Fully leverages Open WebUI v0.7+ built-in tools (web search, memory, knowledge bases, etc.).

> [!TIP]
> Since v0.3, this tool natively supports parallel sub-agent execution via `run_parallel_sub_agents` — no need to use "Parallel Tools" just for running sub-agents in parallel.
> If parallel execution causes issues (e.g., search API rate limits), reduce `MAX_PARALLEL_AGENTS` in Valves, or comment out the `run_parallel_sub_agents` method to disable it entirely.

<!-- markdownlint-disable-next-line MD028 -->

> [!NOTE]
> Direct MCP server connections are not supported ([#6](https://github.com/Skyzi000/open-webui-extensions/issues/6)), but MCP tools work fine via [mcpo](https://github.com/open-webui/mcpo) (converts MCP to OpenAPI).

## Tools

| Tool | Description |
| ---- | ----------- |
| [**Sub Agent**](tools/sub_agent.py) | Delegate tasks to autonomous sub-agents to keep context consumption low |
| [**Parallel Tools**](tools/parallel_tools.py) | Execute multiple independent tool calls in parallel for faster results (note: often requires a strong flagship model to invoke correctly) |
| [**Multi Model Council**](tools/multi_model_council.py) | Run a multi-model council decision with majority vote |
| [**User Location**](tools/user_location.py) | Get user's current location via the browser's Geolocation API |
| [**Universal File Generator (Pandoc)**](tools/universal_file_generator_pandoc.py) | Generate files in various formats using Pandoc. *Not intended for use by others at this time* |

## Filters

| Filter | Description |
| ------ | ----------- |
| [**Token Usage Display**](functions/filter/token_usage_display.py) | Display token usage as a percentage of the context window (requires LiteLLM Proxy) |
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

*Applies only to code authored by Skyzi000.*

MIT License
