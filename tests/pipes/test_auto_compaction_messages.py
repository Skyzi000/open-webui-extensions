from __future__ import annotations

from copy import deepcopy

from functions.pipe import auto_compaction_pipe as mod


def test_safe_cut_tail_never_starts_with_tool_message():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function"}], "content": ""},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
        {"role": "user", "content": "continue"},
    ]

    cut = mod.select_safe_message_cut(messages, keep_tail_messages=2, preserve_latest_tool_rounds=1)

    assert cut.tail_messages[0]["role"] != "tool"
    assert cut.tail_messages[0]["role"] == "assistant"


def test_safe_cut_does_not_split_assistant_tool_call_from_results():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call-a", "type": "function"},
            {"id": "call-b", "type": "function"},
        ],
    }
    messages = [
        {"role": "user", "content": "start"},
        assistant,
        {"role": "tool", "tool_call_id": "call-a", "content": "a"},
        {"role": "tool", "tool_call_id": "call-b", "content": "b"},
        {"role": "user", "content": "next"},
    ]

    cut = mod.select_safe_message_cut(messages, keep_tail_messages=2, preserve_latest_tool_rounds=1)

    assert assistant in cut.tail_messages
    assert {"role": "tool", "tool_call_id": "call-a", "content": "a"} in cut.tail_messages
    assert {"role": "tool", "tool_call_id": "call-b", "content": "b"} in cut.tail_messages


def test_safe_cut_drops_orphan_tool_result_messages_from_tail():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "tool", "tool_call_id": "missing", "content": "orphan"},
        {"role": "user", "content": "latest"},
    ]

    cut = mod.select_safe_message_cut(messages, keep_tail_messages=3, preserve_latest_tool_rounds=1)

    assert {"role": "tool", "tool_call_id": "missing", "content": "orphan"} not in cut.tail_messages
    assert cut.tail_messages[-1] == {"role": "user", "content": "latest"}


def test_safe_cut_keeps_latest_user_message_raw():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active request"},
    ]

    cut = mod.select_safe_message_cut(messages, keep_tail_messages=1, preserve_latest_tool_rounds=0)

    assert cut.tail_messages[-1] == {"role": "user", "content": "active request"}


def test_summary_insertion_preserves_system_and_appends_tail_unchanged():
    messages = [
        {"role": "system", "content": "system stays"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]
    original = deepcopy(messages)
    cut = mod.select_safe_message_cut(messages, keep_tail_messages=1, preserve_latest_tool_rounds=0)

    compacted = mod.replace_prefix_with_summary(messages, cut, "Old discussion summary.")

    assert messages == original
    assert compacted[0] == {"role": "system", "content": "system stays"}
    assert compacted[1]["role"] == "user"
    assert "compressed historical context" in compacted[1]["content"].lower()
    assert "not a new instruction" in compacted[1]["content"].lower()
    assert compacted[2:] == cut.tail_messages
    assert all(m["role"] != "system" for m in compacted[1:])


def test_summary_insertion_without_system_places_summary_first():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]
    cut = mod.select_safe_message_cut(messages, keep_tail_messages=1, preserve_latest_tool_rounds=0)

    compacted = mod.replace_prefix_with_summary(messages, cut, "Summary")

    assert compacted[0]["role"] == "user"
    assert compacted[0]["content"].endswith("Summary")
    assert compacted[1:] == cut.tail_messages


def test_preserved_system_message_is_excluded_from_summarization_prefix():
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]

    cut = mod.select_safe_message_cut(messages, keep_tail_messages=1, preserve_latest_tool_rounds=0)

    assert cut.preserved_system_message == {"role": "system", "content": "policy"}
    assert {"role": "system", "content": "policy"} not in cut.summarization_prefix


def test_retry_cut_can_compact_oversized_historical_tool_result_but_not_latest_user():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function"}], "content": ""},
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 10_000},
        {"role": "user", "content": "active request"},
    ]

    cut = mod.select_safe_message_cut(
        messages,
        keep_tail_messages=1,
        preserve_latest_tool_rounds=0,
        aggressive=True,
    )

    assert cut.tail_messages == [{"role": "user", "content": "active request"}]
    assert messages[-1] in cut.tail_messages


def test_retry_tool_loop_cut_keeps_active_user_and_tool_structure():
    messages = [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 10000},
    ]

    cut = mod.select_retry_tool_result_cut(messages)

    assert cut is not None
    assert cut.active_user_message == {"role": "user", "content": "active request"}
    assert cut.tool_round_messages[0]["role"] == "assistant"
    assert cut.tool_round_messages[1]["role"] == "tool"
    assert cut.tool_messages_to_summarize == [messages[2]]


def test_tool_loop_compaction_preserves_latest_round_and_summarizes_older_results():
    messages = [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "fetch"}}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]

    cut = mod.select_tool_result_compaction_cut(
        messages,
        preserve_latest_tool_rounds=1,
        aggressive=False,
    )

    assert cut is not None
    assert cut.active_user_message == {"role": "user", "content": "active request"}
    assert [message["tool_call_id"] for message in cut.tool_messages_to_summarize] == ["call-1"]


def test_safe_cut_preserves_requested_number_of_latest_tool_rounds():
    messages = [
        {"role": "user", "content": "initial request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "first result"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "fetch"}}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "second result"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-3", "type": "function", "function": {"name": "read"}}],
        },
        {"role": "tool", "tool_call_id": "call-3", "content": "third result"},
        {"role": "user", "content": "latest request"},
    ]

    cut = mod.select_safe_message_cut(
        messages,
        keep_tail_messages=1,
        preserve_latest_tool_rounds=2,
    )

    assert [message.get("tool_call_id") for message in cut.tail_messages if message.get("role") == "tool"] == [
        "call-2",
        "call-3",
    ]
    assert cut.tail_messages[0]["role"] == "assistant"
    assert cut.tail_messages[0]["tool_calls"][0]["id"] == "call-2"
    assert cut.tail_messages[-1] == {"role": "user", "content": "latest request"}


def test_tool_loop_compaction_returns_none_when_only_latest_round_must_remain_raw():
    messages = [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]

    assert (
        mod.select_tool_result_compaction_cut(
            messages,
            preserve_latest_tool_rounds=1,
            aggressive=False,
        )
        is None
    )


def test_aggressive_tool_loop_compaction_can_summarize_latest_round():
    messages = [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]

    cut = mod.select_tool_result_compaction_cut(
        messages,
        preserve_latest_tool_rounds=1,
        aggressive=True,
    )

    assert cut is not None
    assert [message["tool_call_id"] for message in cut.tool_messages_to_summarize] == ["call-1"]
