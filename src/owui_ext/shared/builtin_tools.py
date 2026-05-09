"""Open WebUI builtin tool catalogues.

Two related maps used by plugins that expose category-level on/off Valves:

* ``BUILTIN_TOOL_CATEGORIES`` -- category name -> set of tool function names
  shipped under that category by Open WebUI. Source of truth: ``open_webui/
  utils/tools.py:get_builtin_tools()``. Update whenever the core ships a new
  builtin tool.

* ``VALVE_TO_CATEGORY`` -- ``ENABLE_<X>_TOOLS`` Valves field name -> category
  name. Plugins iterate this to decide which categories to disable based on
  their Valves; ``getattr(valves, name, True)`` makes the lookup safe even
  when a plugin's Valves doesn't declare every entry (a missing Valve is
  treated as "enabled").
"""

BUILTIN_TOOL_CATEGORIES: dict[str, set[str]] = {
    "time": {"get_current_timestamp", "calculate_timestamp"},
    "web": {"search_web", "fetch_url"},
    "image": {"generate_image", "edit_image"},
    "knowledge": {
        "list_knowledge",
        "list_knowledge_bases",
        "search_knowledge_bases",
        "query_knowledge_bases",
        "search_knowledge_files",
        "query_knowledge_files",
        "view_file",
        "view_knowledge_file",
    },
    "chat": {"search_chats", "view_chat"},
    "memory": {"search_memories", "add_memory", "replace_memory_content", "delete_memory", "list_memories"},
    "notes": {
        "search_notes",
        "view_note",
        "write_note",
        "replace_note_content",
    },
    "channels": {
        "search_channels",
        "search_channel_messages",
        "view_channel_thread",
        "view_channel_message",
    },
    "code_interpreter": {"execute_code"},
    "skills": {"view_skill"},
    "tasks": {"create_tasks", "update_task"},
    "automations": {
        "create_automation",
        "update_automation",
        "list_automations",
        "toggle_automation",
        "delete_automation",
    },
    "calendar": {
        "search_calendar_events",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
    },
}


VALVE_TO_CATEGORY: dict[str, str] = {
    "ENABLE_TIME_TOOLS": "time",
    "ENABLE_WEB_TOOLS": "web",
    "ENABLE_IMAGE_TOOLS": "image",
    "ENABLE_KNOWLEDGE_TOOLS": "knowledge",
    "ENABLE_CHAT_TOOLS": "chat",
    "ENABLE_MEMORY_TOOLS": "memory",
    "ENABLE_NOTES_TOOLS": "notes",
    "ENABLE_CHANNELS_TOOLS": "channels",
    "ENABLE_CODE_INTERPRETER_TOOLS": "code_interpreter",
    "ENABLE_SKILLS_TOOLS": "skills",
    "ENABLE_TASK_TOOLS": "tasks",
    "ENABLE_AUTOMATION_TOOLS": "automations",
    "ENABLE_CALENDAR_TOOLS": "calendar",
}
