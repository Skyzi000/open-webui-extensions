"""Loop compaction helper tests (cut, envelope, anchor, truncate, usage)."""

from __future__ import annotations

import pytest

from owui_ext.shared import loop_compaction as m


def _round(call_id: str, result: str = "ok") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _messages(rounds: int) -> list[dict]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        *[msg for i in range(rounds) for msg in _round(f"c{i}")],
    ]


# ---------------------------------------------------------------------------
# Cut selection
# ---------------------------------------------------------------------------


def test_cut_keeps_system_task_user_and_last_two_rounds() -> None:
    cut = m.select_loop_compaction_cut(_messages(5))

    assert cut is not None
    assert cut.preserved_system_message == {"role": "system", "content": "sys"}
    assert cut.task_user_message == {"role": "user", "content": "task"}
    assert len(cut.summarization_prefix) == 2 * 3
    assert len(cut.tail_messages) == 2 * 2
    assert cut.tail_messages[0]["tool_calls"][0]["id"] == "c3"
    assert cut.summarization_prefix[-1]["role"] == "tool"


def test_cut_refuses_when_only_keepable_rounds_exist() -> None:
    assert m.select_loop_compaction_cut(_messages(2)) is None
    assert m.select_loop_compaction_cut(_messages(1)) is None


def test_cut_refuses_orphan_tool_messages() -> None:
    messages = _messages(4)
    messages.append({"role": "tool", "tool_call_id": "missing", "content": "orphan"})

    assert m.select_loop_compaction_cut(messages) is None


def test_cut_refuses_incomplete_round_tail() -> None:
    messages = _messages(3)
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "cX",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
        ]
    )

    cut = m.select_loop_compaction_cut(messages)

    assert cut is not None
    assert cut.tail_messages[-1]["role"] == "assistant"


def test_cut_refuses_without_task_user() -> None:
    assert m.select_loop_compaction_cut([{"role": "system", "content": "s"}]) is None


def test_cut_defaults_to_two_kept_rounds_constant() -> None:
    assert m.LOOP_COMPACTION_KEEP_ROUNDS == 2


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def test_envelope_appends_to_task_content() -> None:
    task = {"role": "user", "content": "do the thing"}
    first = m.embed_envelope_in_task_message(
        task, m.render_compaction_envelope("first summary")
    )

    assert first["content"].startswith("do the thing")
    assert "first summary" in first["content"]
    assert first["content"].count(m.AGENT_LOOP_COMPACTION_CONTEXT_OPEN) == 1
    assert first["content"].endswith(m.AGENT_LOOP_COMPACTION_CONTEXT_CLOSE)


def test_envelope_supports_multipart_content() -> None:
    task = {
        "role": "user",
        "content": [
            {"type": "text", "text": "part one"},
            {"type": "image_url", "image_url": {"url": "data:,"}},
        ],
    }
    embedded = m.embed_envelope_in_task_message(
        task, m.render_compaction_envelope("multipart summary")
    )

    parts = embedded["content"]
    assert len(parts) == 3
    assert parts[0] == {"type": "text", "text": "part one"}
    assert parts[1] == {"type": "image_url", "image_url": {"url": "data:,"}}
    assert parts[2]["type"] == "text"
    assert "multipart summary" in parts[2]["text"]
    assert parts[2]["text"].startswith("\n\n")
    assert parts[2]["text"].rstrip().endswith(m.AGENT_LOOP_COMPACTION_CONTEXT_CLOSE)


def test_envelope_append_never_truncates_task_text_mentioning_the_tag() -> None:
    task = {
        "role": "user",
        "content": (
            "investigate how <agent_loop_compaction_context> is parsed; "
            "keep constraints below\nconstraint A"
        ),
    }

    embedded = m.embed_envelope_in_task_message(
        task, m.render_compaction_envelope("tag-mentioning summary")
    )

    assert embedded["content"].startswith(task["content"])
    assert embedded["content"].count(m.AGENT_LOOP_COMPACTION_CONTEXT_OPEN) == 2
    assert embedded["content"].endswith(m.AGENT_LOOP_COMPACTION_CONTEXT_CLOSE)
    assert embedded["content"].endswith(
        "constraint A\n\n"
        + m.render_compaction_envelope("tag-mentioning summary")
    )


def test_envelope_contains_history_manifest_cdata() -> None:
    envelope = m.render_compaction_envelope(
        "summary", '[{"bytes": 10, "kind": "history", "ref": "history:ab"}]'
    )

    assert '<agent_ref_manifests version="1"><![CDATA[[{"bytes": 10' in envelope
    assert envelope.endswith(m.AGENT_LOOP_COMPACTION_CONTEXT_CLOSE)


def test_summary_prompt_is_adapted_not_verbatim_pipe() -> None:
    assert "attached_file_contents" not in m.SUMMARY_PROMPT
    assert "auto_compaction_context" not in m.SUMMARY_PROMPT
    assert "agent_loop_compaction_context" in m.SUMMARY_PROMPT
    assert "Do not call tools." in m.SUMMARY_PROMPT


# ---------------------------------------------------------------------------
# Anchor + delta
# ---------------------------------------------------------------------------


def _fingerprint(messages: list[dict], *, model: str = "m1") -> str:
    return m.build_loop_input_fingerprint(
        model_id=model,
        tools_param=[{"type": "function", "function": {"name": "t"}}],
        filter_identity=["f1"],
        tool_server_prompt_signature={"servers": 1},
        stable_messages=messages,
    )


def test_anchor_estimate_hits_on_matching_fingerprint() -> None:
    stable = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    anchor = m.LoopUsageAnchor(
        input_tokens=10_000,
        stable_message_count=2,
        input_fingerprint=_fingerprint(stable),
        volatile_message_tokens=20,
    )
    estimate = m.estimate_with_anchor(
        anchor,
        current_fingerprint=_fingerprint(stable),
        current_messages=stable,
        current_volatile_tokens=25,
        suffix_token_estimate=100,
        full_estimate=9_999,
    )

    assert estimate == 10_000 - 20 + 25 + 100


def test_anchor_estimate_hits_when_messages_grew_beyond_anchor_prefix() -> None:
    anchored_prefix = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    grown = [
        *anchored_prefix,
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    anchor = m.LoopUsageAnchor(
        input_tokens=10_000,
        stable_message_count=2,
        input_fingerprint=_fingerprint(anchored_prefix),
        volatile_message_tokens=20,
    )
    estimate = m.estimate_with_anchor(
        anchor,
        current_fingerprint=_fingerprint(anchored_prefix),
        current_messages=grown,
        current_volatile_tokens=25,
        suffix_token_estimate=300,
        full_estimate=5_000,
    )

    assert estimate == 10_000 - 20 + 25 + 300


def test_anchor_estimate_misses_when_anchor_prefix_changes() -> None:
    anchored_prefix = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    rewritten_prefix = [{"role": "system", "content": "s"}, {"role": "user", "content": "CHANGED"}]
    anchor = m.LoopUsageAnchor(
        input_tokens=10_000,
        stable_message_count=2,
        input_fingerprint=_fingerprint(anchored_prefix),
        volatile_message_tokens=20,
    )
    estimate = m.estimate_with_anchor(
        anchor,
        current_fingerprint=_fingerprint(rewritten_prefix),
        current_messages=rewritten_prefix,
        current_volatile_tokens=25,
        suffix_token_estimate=100,
        full_estimate=5_000,
    )

    assert estimate == 5_000


def test_fingerprint_changes_with_tools_filter_or_prompts() -> None:
    stable = [{"role": "system", "content": "s"}]
    base = dict(
        model_id="m1",
        tools_param=[],
        filter_identity=[],
        tool_server_prompt_signature=None,
        stable_messages=stable,
    )
    fp = m.build_loop_input_fingerprint(**base)

    assert fp != m.build_loop_input_fingerprint(**{**base, "tools_param": [{"function": {"name": "x"}}]})
    assert fp != m.build_loop_input_fingerprint(**{**base, "filter_identity": ["f"]})
    assert fp != m.build_loop_input_fingerprint(**{**base, "tool_server_prompt_signature": {"s": 1}})
    assert fp != m.build_loop_input_fingerprint(**{**base, "stable_messages": stable + [{"role": "user", "content": "u"}]})


# ---------------------------------------------------------------------------
# History records
# ---------------------------------------------------------------------------


def test_history_records_are_canonical_jsonl() -> None:
    records = m.canonical_history_records(
        [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
        ]
    )

    assert records == (
        '{"content":"hi","role":"user"}',
        '{"content":"out","role":"tool","tool_call_id":"c1"}',
    )


# ---------------------------------------------------------------------------
# Usage raw keys
# ---------------------------------------------------------------------------


def test_usage_input_tokens_prefers_prompt_tokens_and_never_sums() -> None:
    assert m.usage_input_tokens({"prompt_tokens": 5, "input_tokens": 100}) == 5
    assert m.usage_input_tokens({"prompt_eval_count": 7, "input_tokens": 100}) == 7
    assert m.usage_input_tokens({"input_tokens": 10, "cache_read_input_tokens": 5}) == 15
    assert m.usage_input_tokens({"prompt_tokens": 0, "input_tokens": 4}) is None
    assert m.usage_input_tokens({"prompt_tokens": "bad"}) is None
    assert m.usage_input_tokens({}) is None


def test_response_usage_extracts_dict_only() -> None:
    assert m.response_usage({"usage": {"prompt_tokens": 3}}) == {"prompt_tokens": 3}
    assert m.response_usage({"usage": None}) is None
    assert m.response_usage("nope") is None


# ---------------------------------------------------------------------------
# Summarizer classification
# ---------------------------------------------------------------------------


def test_summary_tool_call_detection() -> None:
    assert (
        m.summary_choice_has_tool_calls(
            {"choices": [{"message": {"tool_calls": [{"id": "c"}]}}]}
        )
        is True
    )
    assert m.summary_choice_has_tool_calls({"choices": [{"message": {"content": "x"}}]}) is False
    assert m.summary_choice_has_tool_calls(None) is False


def test_extract_summary_text() -> None:
    assert m.extract_summary_text({"choices": [{"message": {"content": " s "}}]}) == "s"
    assert m.extract_summary_text({"choices": []}) is None
    assert m.extract_summary_text({"choices": [{"message": {"content": ""}}]}) is None


def test_classify_provider_failure() -> None:
    assert m.classify_provider_failure(status_code=503) == "transient"
    assert m.classify_provider_failure(status_code=429) == "transient"
    assert m.classify_provider_failure(error_text="API error (status 500, X): boom") == "transient"
    assert m.classify_provider_failure(status_code=401) == "fatal"
    assert m.classify_provider_failure(error_text="invalid api key") == "fatal"


# ---------------------------------------------------------------------------
# Truncate preview
# ---------------------------------------------------------------------------


def test_truncate_preview_fits_threshold_below_and_above_64k() -> None:
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    from owui_ext.shared import ref_exec as rx

    for text in ("word " * 3_000, "data " * 20_000):
        estimated = m.estimate_messages_tokens(
            [{"role": "tool", "tool_call_id": "c", "content": text}], encoder=encoder
        )
        assert estimated is not None and estimated >= 1_000

        preview = rx.render_truncate_preview_sync(
            text, len(text.encode()), threshold_tokens=1_000, encoder=encoder
        )
        assert preview is not None
        assert len(encoder.encode(preview)) < 1_000
        assert "tail -c" not in preview


def test_image_file_items_add_overhead_without_name_error() -> None:
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    plain = m.estimate_messages_tokens(
        [{"role": "user", "content": "hi"}], encoder=encoder
    )
    with_image = m.estimate_messages_tokens(
        [
            {
                "role": "user",
                "content": "hi",
                "files": [{"type": "image", "url": "data:image/png;base64,AAAA"}],
            }
        ],
        encoder=encoder,
    )

    assert plain is not None and with_image is not None
    assert with_image - plain >= m.MESSAGE_TOKEN_IMAGE_OVERHEAD
