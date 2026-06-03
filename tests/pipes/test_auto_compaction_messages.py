from __future__ import annotations

from copy import deepcopy

from functions.pipe import auto_compaction_pipe as mod


def test_safe_cut_keeps_only_latest_user_and_following_active_turn_raw():
    messages = [
        {"role": "system", "content": "system stays"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
    ]

    cut = mod.select_safe_message_cut(messages)

    assert cut.preserved_system_message == {"role": "system", "content": "system stays"}
    assert cut.summarization_prefix == [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    assert cut.tail_messages == [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
    ]


def test_safe_cut_keeps_latest_user_raw_when_no_following_tool_round():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active request"},
    ]

    cut = mod.select_safe_message_cut(messages)

    assert cut.summarization_prefix == [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    assert cut.tail_messages == [{"role": "user", "content": "active request"}]


def test_summary_insertion_preserves_system_and_adds_chronological_user_excerpts():
    long_user = "first " + ("x" * 200) + " last"
    messages = [
        {"role": "system", "content": "system stays"},
        {"role": "user", "content": long_user},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "short user"},
        {"role": "assistant", "content": "short answer"},
        {"role": "user", "content": "active"},
    ]
    original = deepcopy(messages)
    cut = mod.select_safe_message_cut(messages)

    compacted = mod.replace_prefix_with_summary(
        messages,
        cut,
        "Old discussion summary.",
        historical_message_excerpt_bytes=40,
        historical_message_excerpt_count=16,
    )

    assert messages == original
    assert compacted[0] == {"role": "system", "content": "system stays"}
    summary_message = compacted[1]
    assert summary_message["role"] == "user"
    assert "<auto_compaction_context>" in summary_message["content"]
    assert "<checkpoint_summary><![CDATA[Old discussion summary.]]></checkpoint_summary>" in summary_message["content"]
    assert "<historical_user_messages" in summary_message["content"]
    assert '<historical_user_message ordinal="1"><![CDATA[first ' in summary_message["content"]
    assert "middle omitted" in summary_message["content"]
    assert " last" in summary_message["content"]
    assert '<historical_user_message ordinal="2"><![CDATA[short user]]></historical_user_message>' in summary_message["content"]
    assert compacted[2:] == [{"role": "user", "content": "active"}]
    assert all(m["role"] != "system" for m in compacted[1:])


def test_summary_insertion_limits_historical_user_excerpts_to_recent_count():
    messages = [
        {"role": "user", "content": "oldest"},
        {"role": "assistant", "content": "oldest answer"},
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
        {"role": "user", "content": "newest historical"},
        {"role": "assistant", "content": "newest answer"},
        {"role": "user", "content": "active"},
    ]
    cut = mod.select_safe_message_cut(messages)

    compacted = mod.replace_prefix_with_summary(
        messages,
        cut,
        "Summary",
        historical_message_excerpt_bytes=128,
        historical_message_excerpt_count=2,
    )

    content = compacted[0]["content"]
    assert "oldest" not in content
    assert '<historical_user_message ordinal="1"><![CDATA[middle]]></historical_user_message>' in content
    assert (
        '<historical_user_message ordinal="2"><![CDATA[newest historical]]></historical_user_message>'
        in content
    )
    assert content.index('ordinal="1"') < content.index('ordinal="2"')


def test_summary_insertion_without_historical_user_messages_omits_excerpt_section():
    messages = [
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]
    cut = mod.select_safe_message_cut(messages)

    compacted = mod.replace_prefix_with_summary(messages, cut, "Summary")

    assert compacted[0]["role"] == "user"
    assert "Summary" in compacted[0]["content"]
    assert "<historical_user_messages" not in compacted[0]["content"]
    assert compacted[1:] == [{"role": "user", "content": "active"}]


def test_tool_loop_compaction_summarizes_every_complete_round():
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

    cut = mod.select_tool_result_compaction_cut(messages)

    assert cut is not None
    assert cut.active_user_message == {"role": "user", "content": "active request"}
    assert [message["tool_call_id"] for message in cut.tool_messages_to_summarize] == ["call-1", "call-2"]


def test_tool_loop_compaction_rejects_orphan_tool_messages():
    messages = [
        {"role": "user", "content": "active request"},
        {"role": "tool", "tool_call_id": "missing", "content": "orphan"},
    ]

    assert mod.select_tool_result_compaction_cut(messages) is None


def test_utf8_middle_truncation_respects_byte_budget():
    text = "先頭" + ("中" * 100) + "末尾"

    truncated = mod.truncate_text_middle_by_utf8_bytes(text, 48)

    assert len(truncated.encode("utf-8")) <= 48
    assert truncated.startswith("先頭")
    assert truncated.endswith("末尾")
    assert "middle omitted" in truncated


def test_xml_cdata_sections_escape_cdata_terminators():
    message = mod.render_summary_message(
        "summary ]]> tail",
        historical_source_messages=[{"role": "user", "content": "user ]]> tail"}],
    )

    assert "<checkpoint_summary><![CDATA[summary ]]]]><![CDATA[> tail]]></checkpoint_summary>" in message["content"]
    assert "<![CDATA[user ]]]]><![CDATA[> tail]]>" in message["content"]


def test_historical_user_excerpt_count_zero_disables_excerpt_section():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]
    cut = mod.select_safe_message_cut(messages)

    compacted = mod.replace_prefix_with_summary(
        messages,
        cut,
        "Summary",
        historical_message_excerpt_count=0,
    )

    assert "<historical_user_messages" not in compacted[0]["content"]
