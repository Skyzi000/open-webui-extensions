"""Ref-exec engine tests for the sub_agent port.

Selected and adapted from the pinned Auto Compact pipe tests at commit
54bb3153659368a607df51c53216167de7f41021
(tests/pipes/test_auto_compaction_ref_exec.py) to the run-local store
architecture: preview tests, parser/command surface, caps, byte paging,
regex fallback, and fail-closed projection.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import inspect
import json
import re
import threading
from typing import Any, Callable, Final

import pytest

from owui_ext.shared import ref_exec as mod


class CountingEncoder:
    def __init__(self, *, count: int = 1, fail: bool = False) -> None:
        self.count = count
        self.fail = fail
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def encode(self, text: str, **kwargs: object) -> list[int]:
        self.calls.append((text, kwargs, threading.get_ident()))
        if self.fail:
            raise RuntimeError("intentional encoder failure")
        return list(range(self.count))


class PreviewByteEncoder:
    def encode(self, text: str, **_kwargs: object) -> range:
        return range((len(text.encode("utf-8")) + 3) // 4)


class Item7ByteEncoder:
    def encode(self, text: str, **_kwargs: object) -> list[int]:
        return list(text.encode())


def _preview_parts(content: str) -> tuple[str, str, str]:
    head, opening, rest = content.partition("\n<agent_ref_truncated>")
    assert opening, content
    encoded, closing, tail = rest.partition("</agent_ref_truncated>\n")
    assert closing, content
    return head, json.loads(encoded)["next"], tail


def _preview_ref(content: str) -> str:
    _head, command, _tail = _preview_parts(content)
    match = re.fullmatch(r"tail -c \+\d+ (tool:[0-9a-f]{64}) \| head -c \d+", command)
    assert match is not None, command
    return match[1]


async def _reader_fixture(
    texts: tuple[str, ...] = ("alpha\nbeta target\ngamma",),
    *,
    threshold_tokens: int = 10_000,
    encoder: object | None = None,
) -> tuple[Callable[..., Any], tuple[str, ...], mod.RefRunStore]:
    store = mod.RefRunStore()
    refs = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).hexdigest()
        ref = f"tool:{digest}"
        refs.append(ref)
        store.intern_tool_text(
            text,
            mod.RefCatalogEntry(
                manifest=mod.RefManifest(
                    ref=ref, utf8_bytes=len(text.encode()), sha256=digest
                ),
                source=mod.ZeroCopySourceHandle(text=text),
            ),
        )
    reader = mod.build_ref_reader(
        store, threshold_tokens=threshold_tokens, encoder=encoder
    )
    return reader, tuple(refs), store


async def _read(reader: Callable[..., Any], command: str) -> str:
    result = reader(command)
    assert inspect.isawaitable(result)
    return await result


_REF_EXEC_USAGE_ERROR: Final = (
    "Error: usage: agent_ref_exec(command). Expected REF: "
    "tool:<64 hex> or history:<64 hex>"
)


# ---------------------------------------------------------------------------
# Threshold classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ref_threshold_skips_tokenizer_above_65536_bytes() -> None:
    encoder = CountingEncoder(count=1)

    result = await mod.classify_ref_text(
        "x" * 65_537, threshold_tokens=10_000, encoder=encoder
    )

    assert result.eligible is True
    assert result.utf8_bytes == 65_537
    assert encoder.calls == []


@pytest.mark.asyncio
async def test_ref_threshold_exact_counts_65536_bytes_off_thread() -> None:
    encoder = CountingEncoder(count=9_999)
    event_loop_thread = threading.get_ident()

    result = await mod.classify_ref_text(
        "x" * 65_536, threshold_tokens=10_000, encoder=encoder
    )

    assert result.eligible is False
    assert result.token_count == 9_999
    assert encoder.calls[0][2] != event_loop_thread


@pytest.mark.asyncio
async def test_ref_threshold_externalizes_at_equal_token_limit() -> None:
    result = await mod.classify_ref_text(
        "boundary",
        threshold_tokens=1_000,
        encoder=CountingEncoder(count=1_000),
    )

    assert result.eligible is True
    assert result.token_count == 1_000


@pytest.mark.asyncio
async def test_ref_threshold_encoder_failure_keeps_only_that_text_raw() -> None:
    bounded = await mod.classify_ref_text(
        "x" * 65_536,
        threshold_tokens=1_000,
        encoder=CountingEncoder(fail=True),
    )
    giant = await mod.classify_ref_text(
        "x" * 65_537,
        threshold_tokens=1_000,
        encoder=CountingEncoder(fail=True),
    )

    assert bounded.eligible is False
    assert bounded.encoder_failed is True
    assert giant.eligible is True
    assert giant.encoder_failed is False


@pytest.mark.asyncio
async def test_ref_threshold_uses_utf8_bytes_not_characters() -> None:
    result = await mod.classify_ref_text(
        "🙂" * 16_385,
        threshold_tokens=1_000,
        encoder=CountingEncoder(count=1),
    )

    assert result.eligible is True
    assert result.utf8_bytes == 65_540


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size, externalized", [(3_996, False), (4_000, True), (4_004, True)]
)
async def test_tool_preview_threshold_includes_marker_budget(
    size: int, externalized: bool
) -> None:
    encoder = PreviewByteEncoder()
    source = "x" * size
    messages = [{"role": "tool", "tool_call_id": "call", "content": source}]
    plan = await mod.project_native_tool_texts(
        messages, threshold_tokens=1_000, encoder=encoder
    )
    projected = await mod.apply_ref_projection_plan(messages, plan)

    assert messages[0]["content"] == source
    assert bool(plan.catalog) is externalized
    if not externalized:
        assert projected == messages
        return
    preview = projected[0]["content"]
    ref = _preview_ref(preview)
    assert ref == f"tool:{hashlib.sha256(source.encode()).hexdigest()}"
    assert len(encoder.encode(preview)) < 1_000
    assert len(preview.encode()) <= mod.REF_EXEC_RESPONSE_MAX_BYTES
    assert plan.catalog[0].source.text is source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "head of the result\n" + "middle line\n" * 900 + "tail of the result\n",
        "日本語の長い結果。\n" * 1_200,
        "🙂🚀漢字αβγ" * 3_000,
        "single giant line " + "z" * 100_000,
    ],
    ids=("lines", "japanese", "emoji", "giant-line"),
)
async def test_tool_preview_preserves_utf8_head_tail_and_full_source_identity(
    source: str,
) -> None:
    encoder = PreviewByteEncoder()
    plan = await mod.project_native_tool_texts(
        [{"role": "tool", "tool_call_id": "call", "content": source}],
        threshold_tokens=1_000,
        encoder=encoder,
    )
    entry = plan.catalog[0]
    head, command, tail = _preview_parts(entry.preview_text or "")
    retained_head = len(head.encode())
    omitted = len(source.encode()) - retained_head - len(tail.encode())

    assert head and tail
    assert source.startswith(head) and source.endswith(tail)
    assert abs(retained_head - len(tail.encode())) <= 4
    assert (
        command
        == f"tail -c +{retained_head + 1} {entry.manifest.ref} | head -c {omitted}"
    )
    assert entry.manifest.sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert entry.source.text is source
    assert len(encoder.encode(entry.preview_text or "")) < 1_000
    assert len((entry.preview_text or "").encode()) <= mod.REF_EXEC_RESPONSE_MAX_BYTES


@pytest.mark.asyncio
async def test_tool_preview_fills_token_and_byte_budgets_without_exceeding_them() -> (
    None
):
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    source = "Realistic english sentences with several words each.\n" * 700
    plan = await mod.project_native_tool_texts(
        [{"role": "tool", "tool_call_id": "call", "content": source}],
        threshold_tokens=1_000,
        encoder=encoder,
    )
    assert 950 <= len(encoder.encode(plan.catalog[0].preview_text or "")) < 1_000

    byte_plan = await mod.project_native_tool_texts(
        [{"role": "tool", "tool_call_id": "call", "content": "a" * 200_000}],
        threshold_tokens=1_000_000,
        encoder=CountingEncoder(count=1),
    )
    assert (
        len((byte_plan.catalog[0].preview_text or "").encode())
        == mod.REF_EXEC_RESPONSE_MAX_BYTES
    )


def _item7_visible_and_next(result: str) -> tuple[str, str | None]:
    for tag in ("agent_ref_range", "agent_ref_truncated"):
        opening = f"\n<{tag}>"
        if opening in result:
            visible, encoded_marker = result.split(opening, 1)
            marker = json.loads(encoded_marker.removesuffix(f"</{tag}>"))
            return visible, marker.get("next")
    return result, None


@pytest.mark.asyncio
async def test_tool_preview_marker_recovers_omitted_middle_across_reader_pages() -> (
    None
):
    source = "format=record key=visible_head\n" + ("α界" + "x" * 70 + "\n") * 1_200
    encoder = PreviewByteEncoder()
    plan = await mod.project_native_tool_texts(
        [{"role": "tool", "tool_call_id": "call", "content": source}],
        threshold_tokens=1_000,
        encoder=encoder,
    )
    head, command, tail = _preview_parts(plan.catalog[0].preview_text or "")
    reader, refs, _store = await _reader_fixture(
        (source,), threshold_tokens=1_000, encoder=encoder
    )
    assert head.startswith("format=record key=visible_head\n")
    assert (
        await _read(reader, f"grep -c -- {json.dumps(head.splitlines()[0])} {refs[0]}")
        == "1"
    )
    middle = []
    for _ in range(100):
        response = await _read(reader, command)
        visible, next_command = _item7_visible_and_next(response)
        middle.append(visible)
        if next_command is None:
            break
        assert next_command.startswith("tail -c +"), response
        assert refs[0] in next_command
        command = next_command
    else:
        pytest.fail("preview recovery did not finish")

    assert len(middle) > 1
    assert (head + "".join(middle) + tail).encode() == source.encode()


@pytest.mark.asyncio
async def test_tool_preview_counter_failure_falls_back_to_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecoveringEncoder(PreviewByteEncoder):
        fail = True

        def encode(self, text: str, **kwargs: object) -> range:
            if self.fail:
                raise RuntimeError("counter unavailable")
            return super().encode(text, **kwargs)

    source = "oversized result\n" * 10_000
    messages = [{"role": "tool", "tool_call_id": "call", "content": source}]
    monkeypatch.setattr(
        mod, "_ref_exec_get_tiktoken_encoder", lambda _request=None: (None, None)
    )

    recovering_encoder = RecoveringEncoder()
    for encoder in (None, recovering_encoder):
        plan = await mod.project_native_tool_texts(
            messages, threshold_tokens=1_000, encoder=encoder
        )
        assert len(plan.catalog) == 1
        projected = await mod.apply_ref_projection_plan(messages, plan)
        preview = projected[0]["content"]
        assert _preview_ref(preview) == plan.catalog[0].manifest.ref
        assert len(preview.encode("utf-8")) < 1_000


@pytest.mark.asyncio
async def test_tool_preview_rendering_is_worker_bound_and_run_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "preview cache result\n" * 1_000
    messages = [{"role": "tool", "tool_call_id": "call", "content": source}]
    encoder = PreviewByteEncoder()
    render = mod._render_tool_ref_preview_sync
    threads = []

    def tracked(*args: object, **kwargs: object) -> str | None:
        threads.append(threading.get_ident())
        return render(*args, **kwargs)

    monkeypatch.setattr(mod, "_render_tool_ref_preview_sync", tracked)
    cache: dict[tuple[str, int, int], tuple[Any, str]] = {}
    first = await mod.project_native_tool_texts(
        messages, threshold_tokens=1_000, encoder=encoder, preview_cache=cache
    )
    second = await mod.project_native_tool_texts(
        messages, threshold_tokens=1_000, encoder=encoder, preview_cache=cache
    )
    projected = await mod.apply_ref_projection_plan(messages, first)
    assert await mod.apply_ref_projection_plan(messages, second) == projected
    assert await mod.apply_ref_projection_plan(projected, second) == projected
    assert len(threads) == 1
    assert threads[0] != threading.get_ident()

    smaller = await mod.project_native_tool_texts(
        messages, threshold_tokens=500, encoder=encoder, preview_cache=cache
    )
    assert len(threads) == 2
    assert len(encoder.encode(smaller.catalog[0].preview_text or "")) < 500
    assert smaller.catalog[0].preview_text != first.catalog[0].preview_text
    await mod.project_native_tool_texts(
        messages, threshold_tokens=1_000, encoder=encoder, preview_cache={}
    )
    assert len(threads) == 3
    await mod.project_native_tool_texts(
        messages, threshold_tokens=1_000, encoder=PreviewByteEncoder(), preview_cache=cache
    )
    assert len(threads) == 4


@pytest.mark.asyncio
async def test_tool_preview_rendering_failure_never_forwards_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = "eligible" * 10_000
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "preview-call",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "preview-call", "content": raw},
    ]

    monkeypatch.setattr(
        mod, "_cached_tool_ref_preview_sync", lambda *_args, **_kwargs: None
    )

    with pytest.raises(mod.RefProjectionError) as failure:
        await mod.project_native_tool_texts(
            messages, threshold_tokens=1_000, encoder=PreviewByteEncoder()
        )

    assert "before provider forward" in str(failure.value)


# ---------------------------------------------------------------------------
# Parser surface
# ---------------------------------------------------------------------------


def test_ref_exec_parser_preserves_valid_command_behavior() -> None:
    ref = f"tool:{'a' * 64}"

    stages = mod._parse_ref_exec_command(f"cat {ref}")

    assert stages == (mod.RefExecStage(command="cat", ref=ref),)


def test_ref_exec_parser_accepts_history_refs_without_pipe_prefix() -> None:
    ref = f"history:{'b' * 64}"

    stages = mod._parse_ref_exec_command(f"cat {ref}")

    assert stages == (mod.RefExecStage(command="cat", ref=ref),)
    assert mod.parse_ref(ref) == mod.ParsedRef(kind="history", value="b" * 64)
    assert mod.parse_ref(f"history:accp_{'b' * 59}") is None


@pytest.mark.parametrize(
    ("command", "expected_fields"),
    (
        ("head -c 0", {"command": "head", "byte_count": 0}),
        ("head -c 12", {"command": "head", "byte_count": 12}),
        ("tail -c 0", {"command": "tail", "byte_count": 0}),
        ("tail -c 12", {"command": "tail", "byte_count": 12}),
        ("tail -c +1", {"command": "tail", "byte_start": 1}),
        ("tail -c +12", {"command": "tail", "byte_start": 12}),
        ("head -12", {"command": "head", "count": 12}),
        ("tail -12", {"command": "tail", "count": 12}),
    ),
)
def test_ref_exec_parser_distinguishes_supported_byte_and_line_counts(
    command: str,
    expected_fields: dict[str, str | int],
) -> None:
    stage = mod._parse_ref_exec_stage(command, source=False)

    assert dataclasses.asdict(stage) == {
        "ref": None,
        "count": None,
        "flags": frozenset(),
        "pattern": None,
        "start_line": None,
        "end_line": None,
        "list_kind": None,
        "byte_count": None,
        "byte_start": None,
        **expected_fields,
    }


@pytest.mark.parametrize(
    "command",
    (
        "head -c",
        "tail -c",
        "head -c +1",
        "head -c -1",
        "tail -c -1",
        "tail -c +0",
        "tail -c ++1",
        "tail -c 1K",
    ),
)
def test_ref_exec_parser_rejects_unsupported_byte_count_forms(command: str) -> None:
    with pytest.raises(mod.RefExecError) as failure:
        mod._parse_ref_exec_command(command)

    assert "usage:" in str(failure.value)
    assert "-c" in str(failure.value)


def test_ref_exec_parser_preserves_utf8_byte_limit_error() -> None:
    command = "x" * (mod.REF_EXEC_COMMAND_MAX_BYTES + 1)

    with pytest.raises(mod.RefExecError) as failure:
        mod._parse_ref_exec_command(command)

    assert str(failure.value) == "Error: command exceeds the 1,024 UTF-8 byte parser limit"


def test_ref_exec_parser_maps_unpaired_surrogate_to_ref_exec_error() -> None:
    command = "cat \ud800"

    with pytest.raises(mod.RefExecError) as failure:
        mod._parse_ref_exec_command(command)

    assert "Expected REF: tool:<64 hex> or history:<64 hex>" in str(failure.value)


def test_ref_exec_pipeline_rejects_malformed_quotes() -> None:
    with pytest.raises(mod.RefExecError, match="malformed quote"):
        mod._split_ref_exec_pipeline("grep 'unclosed | pipe")


def test_ref_exec_pipeline_splits_only_outside_quotes() -> None:
    stages = mod._split_ref_exec_pipeline("grep -n 'a|b' tool:x | head -2")

    assert stages == ("grep -n 'a|b' tool:x", "head -2")


# ---------------------------------------------------------------------------
# Reader surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ref_exec_reader_maps_unpaired_surrogate_to_reader_error() -> None:
    reader, _, _ = await _reader_fixture()

    result = await _read(reader, "cat \ud800")

    assert "Expected REF: tool:<64 hex> or history:<64 hex>" in result


@pytest.mark.asyncio
async def test_ref_exec_reader_omitted_command_returns_usage_error() -> None:
    reader, _, _ = await _reader_fixture()

    result = await reader()

    assert result == _REF_EXEC_USAGE_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize(("command"), (None, []), ids=("none", "list"))
async def test_ref_exec_reader_rejects_non_string_before_parser(
    monkeypatch: pytest.MonkeyPatch,
    command: Any,
) -> None:
    reader, _, _ = await _reader_fixture()
    parser_calls = 0
    original_parser = mod._parse_ref_exec_command

    def observed_parser(value: str) -> tuple[mod.RefExecStage, ...]:
        nonlocal parser_calls
        parser_calls += 1
        return original_parser(value)

    monkeypatch.setattr(mod, "_parse_ref_exec_command", observed_parser)

    result = await reader(command=command)

    assert result == _REF_EXEC_USAGE_ERROR
    assert parser_calls == 0


@pytest.mark.asyncio
async def test_ref_exec_reader_signature_preserves_required_provider_schema() -> None:
    reader, _, _ = await _reader_fixture()

    command = inspect.signature(reader, eval_str=True).parameters["command"]
    parameters = mod.ref_exec_tool_spec_payload()["function"]["parameters"]
    command_schema = parameters["properties"]["command"]

    assert command.default == ""
    assert command.annotation is str
    assert command_schema["type"] == "string"
    assert parameters["required"] == ["command"]
    assert "default" not in command_schema


@pytest.mark.asyncio
async def test_ref_exec_supports_bounded_commands() -> None:
    reader, refs, _ = await _reader_fixture()
    ref = refs[0]

    assert await _read(reader, f"cat {ref}") == "alpha\nbeta target\ngamma"
    assert "target" in await _read(reader, f"grep target {ref}")
    assert ref in await _read(reader, "ls tool")
    assert "utf8_bytes=" in await _read(reader, f"stat {ref}")
    spec = mod.ref_exec_tool_spec_payload()["function"]
    assert spec["name"] == "agent_ref_exec"
    assert "preview" in spec["description"]
    assert "<agent_ref_truncated>" in spec["description"]
    assert "next command" in spec["description"]
    assert spec["parameters"] == {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "One command line, max 1,024 UTF-8 bytes, e.g. sed -n '1,120p' tool:<64 hex>",
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    }


def test_ref_exec_tool_spec_advertises_all_commands_and_ref_schemes() -> None:
    description = mod.ref_exec_tool_spec_payload()["function"]["description"]

    for command in ("ls", "stat", "wc", "cat", "head", "tail", "sed", "grep"):
        assert f"{command} " in description
    assert "tool:<64 hex>" in description
    assert "history:<64 hex>" in description


@pytest.mark.asyncio
async def test_ref_exec_supports_head_tail_sed_and_wc_ux() -> None:
    reader, refs, _ = await _reader_fixture()
    ref = refs[0]

    assert await _read(reader, f"head -n 1 {ref}") == "alpha\n"
    assert await _read(reader, f"tail -1 {ref}") == "gamma"
    assert await _read(reader, f"sed -n '2,3p' {ref}") == "beta target\ngamma"
    assert await _read(reader, f"wc -l {ref}") == "3"
    assert await _read(reader, f"wc -w {ref}") == "4"
    assert await _read(reader, f"wc -c {ref}") == str(
        len("alpha\nbeta target\ngamma".encode())
    )


@pytest.mark.parametrize(
    "command",
    ("sed -n '1,2p' {ref}", "tail -n 2 {ref}"),
    ids=("sed", "tail"),
)
@pytest.mark.asyncio
async def test_ref_exec_transformed_stage_preserves_leading_empty_output(
    command: str,
) -> None:
    reader, refs, _ = await _reader_fixture(("\nalpha",))

    result = await _read(reader, command.format(ref=refs[0]))

    assert result == "\nalpha"


# ---------------------------------------------------------------------------
# UTF-8 safety
# ---------------------------------------------------------------------------


def test_ref_exec_checked_measurement_maps_unpaired_surrogate_and_prioritizes_cancellation() -> (
    None
):
    cancelled = threading.Event()

    with pytest.raises(mod.RefExecError) as failure:
        mod._measure_ref_text_checked("bad-\ud800", cancelled)

    assert str(failure.value) == "Error: externalized ref source is not valid UTF-8"
    assert isinstance(failure.value.__cause__, UnicodeEncodeError)

    cancelled.set()
    with pytest.raises(mod.RefExecError, match="cancelled") as cancelled_failure:
        mod._measure_ref_text_checked("bad-\ud800", cancelled)
    assert cancelled_failure.value.__cause__ is None


def test_ref_exec_utf8_range_maps_surrogate_inside_selected_range() -> None:
    text = "outside-k\ud800t-outside"

    with pytest.raises(mod.RefExecError) as failure:
        mod._ref_exec_utf8_range_bytes(text, 8, 11, threading.Event())

    assert str(failure.value) == "Error: externalized ref source is not valid UTF-8"
    assert isinstance(failure.value.__cause__, UnicodeEncodeError)

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(mod.RefExecError, match="cancelled") as cancelled_failure:
        mod._ref_exec_utf8_range_bytes(text, 8, 11, cancelled)
    assert cancelled_failure.value.__cause__ is None


def test_ref_exec_zero_copy_maps_surrogate_in_later_chunk() -> None:
    cancelled = threading.Event()
    source = mod.ZeroCopySourceHandle(
        text="first-part" + "x" * mod.REF_TEXT_HASH_CHUNK_CHARS + "\ud800"
    )
    entry = mod.RefCatalogEntry(
        manifest=mod.RefManifest(
            ref=f"tool:{'a' * 64}",
            utf8_bytes=0,
            sha256="0" * 64,
        ),
        source=source,
    )

    with pytest.raises(mod.RefExecError) as measurement_failure:
        mod._measure_ref_source_checked(source, cancelled)
    with pytest.raises(mod.RefExecError) as reader_failure:
        mod._execute_ref_reader_sync(
            (mod.RefExecStage(command="wc", ref=entry.manifest.ref, flags=frozenset({"c"})),),
            (entry,),
            threshold_tokens=10_000,
            encoder=None,
            cancelled=threading.Event(),
        )

    assert isinstance(measurement_failure.value.__cause__, UnicodeEncodeError)
    assert isinstance(reader_failure.value.__cause__, UnicodeEncodeError)


def test_ref_exec_verified_iterator_maps_unpaired_surrogate_during_traversal() -> None:
    entry = mod.RefCatalogEntry(
        manifest=mod.RefManifest(
            ref=f"tool:{'b' * 64}",
            utf8_bytes=0,
            sha256="0" * 64,
        ),
        source=mod.ZeroCopySourceHandle(text="first\nsecond-\ud800"),
    )

    with pytest.raises(mod.RefExecError) as failure:
        list(mod._iter_verified_ref_source_lines(entry, threading.Event()))

    assert str(failure.value) == "Error: externalized ref source is not valid UTF-8"
    assert isinstance(failure.value.__cause__, UnicodeEncodeError)


# ---------------------------------------------------------------------------
# Run-local store + history refs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_store_history_ref_is_reader_visible_and_byte_exact() -> None:
    store = mod.RefRunStore()
    records = (
        json.dumps({"content": "task", "role": "user"}, sort_keys=True),
        json.dumps(
            {"content": "", "role": "assistant", "tool_calls": [{"id": "c1"}]},
            sort_keys=True,
        ),
        json.dumps({"content": "result", "role": "tool", "tool_call_id": "c1"}, sort_keys=True),
    )
    ref = store.add_history_records(records)
    reader = mod.build_ref_reader(store, threshold_tokens=10_000)

    assert ref.startswith("history:")
    assert ref in await _read(reader, "ls history")
    assert await _read(reader, f"wc -l {ref}") == "3"
    assert await _read(reader, f"cat {ref}") == "\n".join(records)
    payload = store.history_manifest_payload(ref)
    assert payload is not None
    assert payload["kind"] == "history"
    assert payload["lines"] == 3
    assert payload["bytes"] == len(("\n".join(records)).encode())


@pytest.mark.asyncio
async def test_run_store_dedupes_identical_tool_texts() -> None:
    store = mod.RefRunStore()
    text = "duplicate payload\n" * 50
    digest = hashlib.sha256(text.encode()).hexdigest()
    entry = mod.RefCatalogEntry(
        manifest=mod.RefManifest(
            ref=f"tool:{digest}", utf8_bytes=len(text.encode()), sha256=digest
        ),
        source=mod.ZeroCopySourceHandle(text=text),
    )

    first = store.intern_tool_text(text, entry)
    second = store.intern_tool_text(text, dataclasses.replace(entry))

    assert first == second
    assert len(store.catalog_tuple()) == 1


@pytest.mark.asyncio
async def test_reader_response_caps_enforced() -> None:
    store = mod.RefRunStore()
    text = "line\n" * 100_000
    digest = hashlib.sha256(text.encode()).hexdigest()
    store.intern_tool_text(
        text,
        mod.RefCatalogEntry(
            manifest=mod.RefManifest(
                ref=f"tool:{digest}", utf8_bytes=len(text.encode()), sha256=digest
            ),
            source=mod.ZeroCopySourceHandle(text=text),
        ),
    )
    reader = mod.build_ref_reader(store, threshold_tokens=1_000)

    response = await _read(reader, f"cat tool:{digest}")

    assert len(response.encode()) <= mod.REF_EXEC_RESPONSE_MAX_BYTES
    assert "agent_ref_truncated" in response or "agent_ref_range" in response


@pytest.mark.asyncio
async def test_truncate_preview_has_no_recovery_command() -> None:
    encoder = PreviewByteEncoder()
    source = "truncate me\n" * 20_000
    preview = mod.render_truncate_preview_sync(
        source, len(source.encode()), threshold_tokens=1_000, encoder=encoder
    )

    assert preview is not None
    assert "tail -c" not in preview
    assert "<agent_ref_truncated>" in preview
    marker = json.loads(
        preview.split("<agent_ref_truncated>")[1].split("</agent_ref_truncated>")[0]
    )
    assert marker["omitted"] > 0
    head, _opening, tail = preview.partition("\n<agent_ref_truncated>")
    tail = tail.split("</agent_ref_truncated>\n", 1)[1]
    assert source.startswith(head)
    assert source.endswith(tail)
    assert len(encoder.encode(preview)) < 1_000


@pytest.mark.asyncio
async def test_history_ref_shares_content_address_with_tool_ref() -> None:
    store = mod.RefRunStore()
    text = "shared line one\nshared line two"
    digest = hashlib.sha256(text.encode()).hexdigest()
    store.intern_tool_text(
        text,
        mod.RefCatalogEntry(
            manifest=mod.RefManifest(
                ref=f"tool:{digest}", utf8_bytes=len(text.encode()), sha256=digest
            ),
            source=mod.ZeroCopySourceHandle(text=text),
        ),
    )

    ref = store.add_history_records(("shared line one", "shared line two"))

    assert ref == f"history:{digest}"
    reader = mod.build_ref_reader(store, threshold_tokens=10_000)
    assert await _read(reader, f"cat {ref}") == text
    assert await _read(reader, f"cat tool:{digest}") == text
    payload = store.history_manifest_payload(ref)
    assert payload is not None
    assert payload["bytes"] == len(text.encode())
    assert payload["lines"] == 2
    assert payload["ref"] == ref
    assert len(store.catalog_tuple()) == 1


@pytest.mark.asyncio
async def test_reader_resolves_encoder_off_the_event_loop(monkeypatch) -> None:
    import contextlib

    loop = asyncio.get_running_loop()
    loop_thread = threading.current_thread()
    started = asyncio.Event()
    release = threading.Event()
    seen: list[threading.Thread] = []

    def stub_resolver(_request: Any = None) -> tuple[None, None]:
        seen.append(threading.current_thread())
        loop.call_soon_threadsafe(started.set)
        if threading.current_thread() is loop_thread:
            return (None, None)
        release.wait(timeout=5.0)
        return (None, None)

    monkeypatch.setattr(mod, "_ref_exec_get_tiktoken_encoder", stub_resolver)

    reader, refs, _store = await _reader_fixture(
        ("alpha\nbeta",), threshold_tokens=10_000
    )
    task = asyncio.create_task(reader(f"cat {refs[0]}"))
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        done, _pending = await asyncio.wait({task}, timeout=1.0)
        assert task in done
        assert task.cancelled()
    finally:
        release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert seen
    assert seen[0] is not loop_thread


@pytest.mark.asyncio
async def test_direct_ref_tail_stream_large_line_counts() -> None:
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    reader, refs, _store = await _reader_fixture(
        ("detail\n" * 70_000, ("x" * (3 * 1024 * 1024) + "\n") * 3),
        encoder=encoder,
    )

    assert await _read(reader, f"tail -n 70000 {refs[0]} | wc -l") == "70000"
    assert await _read(reader, f"tail -n 3 {refs[1]} | wc -l") == "3"


@pytest.mark.asyncio
async def test_direct_ref_tail_output_pins_prebounded_implementation() -> None:
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    reader, refs, _store = await _reader_fixture(
        ("a\nb\n", "alpha\nbeta target\ngamma"),
        encoder=encoder,
    )

    assert await _read(reader, f"tail -n 0 {refs[0]}") == ""
    assert await _read(reader, f"tail -n 1 {refs[0]}") == "b"
    assert await _read(reader, f"tail -n 5 {refs[1]}") == "alpha\nbeta target\ngamma"


@pytest.mark.asyncio
async def test_pipeline_tail_rejects_when_window_cannot_hold_needed_lines() -> None:
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    reader, refs, _store = await _reader_fixture(
        (
            ("x" * (3 * 1024 * 1024) + "\n") * 3,
            ("😀" * 1_000_000 + "\n") * 3,
            "x" * (9 * 1024 * 1024) + "\n" + "short1\nshort2\nshort3",
            "detail\n" * 70_000,
        ),
        encoder=encoder,
    )
    ref, ref_e, ref_m, ref_s = refs

    result = await _read(reader, f"head -n 1000000 {ref} | tail -n 3 | wc -l")
    assert "reduce -n or filter first" in result
    assert await _read(reader, f"head -n 1000000 {ref} | tail -n 2 | wc -l") == "2"

    result = await _read(reader, f"head -n 1000000 {ref_e} | tail -n 3 | wc -l")
    assert "reduce -n or filter first" in result
    assert await _read(reader, f"head -n 1000000 {ref_e} | tail -n 2 | wc -l") == "2"

    result = await _read(reader, f"head -c 9437186 {ref} | tail -n 3 | wc -l")
    assert "reduce -n or filter first" in result
    assert await _read(reader, f"head -c 9437186 {ref} | tail -n 2 | wc -l") == "2"

    assert (
        await _read(reader, f"head -n 1000000 {ref_m} | tail -n 3 | wc -l") == "3"
    )
    result = await _read(reader, f"head -n 1000000 {ref_m} | tail -n 4 | wc -l")
    assert "reduce -n or filter first" in result

    result = await _read(reader, f"head -n 1000000 {ref_s} | tail -n 70000 | wc -l")
    assert "reduce -n or filter first" in result
    assert await _read(reader, f"head -n 1000000 {ref_s} | tail -n 65536 | wc -l") == "65536"
