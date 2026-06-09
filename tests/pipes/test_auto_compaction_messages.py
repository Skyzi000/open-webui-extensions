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


def test_tool_loop_compaction_retains_only_latest_complete_round_raw():
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
    assert cut.summarization_prefix == [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    assert cut.tail_messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "fetch"}}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]


def test_tool_loop_compaction_keeps_latest_parallel_tool_call_round_intact():
    latest_assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call-2", "type": "function", "function": {"name": "fetch"}},
            {"id": "call-3", "type": "function", "function": {"name": "read"}},
        ],
    }
    latest_tools = [
        {"role": "tool", "tool_call_id": "call-2", "content": "fetch result"},
        {"role": "tool", "tool_call_id": "call-3", "content": "read result"},
    ]
    messages = [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
        latest_assistant,
        *latest_tools,
    ]

    cut = mod.select_tool_result_compaction_cut(messages)

    assert cut is not None
    assert cut.summarization_prefix == messages[:3]
    assert cut.tail_messages == [latest_assistant, *latest_tools]


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


def test_checkpoint_summary_meta_stores_excerpts_without_mutating_summary_text():
    source_messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "newer request"},
    ]

    summary_meta = mod.enrich_summary_meta_with_historical_excerpts(
        {"has_multimodal": False},
        source_messages,
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=2,
    )
    row = mod.build_checkpoint_row(
        namespace="ns",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=len(source_messages),
        summary_text="AI generated summary only.",
        summary_meta=summary_meta,
        parent_checkpoint_id=None,
        now=123,
    )

    assert row["summary_text"] == "AI generated summary only."
    stored = row["summary_meta"]["historical_user_messages"]
    assert stored == {
        "format_version": 1,
        "order": "chronological",
        "max_count": 2,
        "max_bytes_per_message": 64,
        "selected_count": 2,
        "messages": [
            {"ordinal": 1, "text": "old request"},
            {"ordinal": 2, "text": "newer request"},
        ],
    }
    rendered = mod.render_summary_message_from_checkpoint(row)
    assert "AI generated summary only." in rendered["content"]
    assert '<historical_user_message ordinal="1"><![CDATA[old request]]></historical_user_message>' in rendered["content"]
    assert '<historical_user_message ordinal="2"><![CDATA[newer request]]></historical_user_message>' in rendered["content"]


def test_checkpoint_summary_meta_count_zero_does_not_store_or_render_excerpts():
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]

    summary_meta = mod.enrich_summary_meta_with_historical_excerpts(
        {"has_multimodal": False},
        source_messages,
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=0,
    )
    row = mod.build_checkpoint_row(
        namespace="ns",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=len(source_messages),
        summary_text="Summary",
        summary_meta=summary_meta,
        parent_checkpoint_id=None,
        now=123,
    )

    assert "historical_user_messages" not in row["summary_meta"]
    assert "<historical_user_messages" not in mod.render_summary_message_from_checkpoint(row)["content"]
    assert (
        "<historical_user_messages"
        not in mod.render_summary_message_from_checkpoint(row, historical_source_messages=source_messages)["content"]
    )


def test_checkpoint_render_uses_stored_excerpts_not_current_excerpt_limits():
    source_messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "newer request"},
    ]
    summary_meta = mod.enrich_summary_meta_with_historical_excerpts(
        {},
        source_messages,
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=2,
    )
    row = mod.build_checkpoint_row(
        namespace="ns",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=len(source_messages),
        summary_text="Summary",
        summary_meta=summary_meta,
        parent_checkpoint_id=None,
        now=123,
    )

    expected = mod.render_summary_message_from_checkpoint(row)
    changed_valve_render = mod.render_summary_message(
        row["summary_text"],
        row["summary_meta"],
        historical_source_messages=[{"role": "user", "content": "current valve must not matter"}],
        historical_message_excerpt_bytes=1,
        historical_message_excerpt_count=0,
    )

    assert changed_valve_render == expected


def test_legacy_checkpoint_without_excerpts_can_render_from_source_prefix_when_available():
    source_messages = [
        {"role": "user", "content": "legacy old"},
        {"role": "assistant", "content": "legacy answer"},
    ]
    legacy_row = mod.build_checkpoint_row(
        namespace="ns",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=len(source_messages),
        summary_text="Legacy summary",
        summary_meta={},
        parent_checkpoint_id=None,
        now=123,
    )

    without_source = mod.render_summary_message_from_checkpoint(legacy_row)
    with_source = mod.render_summary_message_from_checkpoint(legacy_row, historical_source_messages=source_messages)

    assert "Legacy summary" in without_source["content"]
    assert "<historical_user_messages" not in without_source["content"]
    assert '<historical_user_message ordinal="1"><![CDATA[legacy old]]></historical_user_message>' in with_source["content"]


def test_legacy_parent_checkpoint_respects_disabled_excerpt_fallback():
    messages = [
        {"role": "user", "content": "legacy old"},
        {"role": "assistant", "content": "legacy answer"},
        {"role": "user", "content": "active"},
    ]
    cut = mod.select_safe_message_cut(messages)
    parent = mod.build_checkpoint_row(
        namespace="ns",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash=mod.compute_source_hash(cut.summarization_prefix),
        source_message_count=len(cut.summarization_prefix),
        summary_text="Legacy parent summary",
        summary_meta={},
        parent_checkpoint_id=None,
        now=123,
    )

    compacted = mod.replace_prefix_with_parent_checkpoint_and_delta(
        cut,
        parent,
        historical_message_excerpt_count=0,
    )

    assert "Legacy parent summary" in compacted[0]["content"]
    assert "<historical_user_messages" not in compacted[0]["content"]
