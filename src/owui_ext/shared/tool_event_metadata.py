"""Citation and terminal-event tool name catalogues.

These sets describe how Open WebUI treats specific tools at the message-event
level:

* ``CITATION_TOOLS`` produce citation sources that the UI renders inline.
* ``TERMINAL_EVENT_TOOLS`` are the "final" actions that trigger UI
  refresh/display events when they complete.

Update these whenever Open WebUI ships a new citation- or terminal-emitting
tool. They live in shared so that any plugin that needs to mirror Open WebUI's
event semantics (parallel runner, council voters, decision support, etc.) can
import them without rolling its own copy.
"""

from __future__ import annotations


CITATION_TOOLS: set[str] = {
    "search_web",
    "view_file",
    "view_knowledge_file",
    "query_knowledge_files",
    "fetch_url",
}


TERMINAL_EVENT_TOOLS: set[str] = {
    "display_file",
    "write_file",
    "replace_file_content",
    "run_command",
}
