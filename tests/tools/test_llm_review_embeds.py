"""EventEmitter embed-preservation tests for tools/llm_review.py.

Focused on the interaction with Open WebUI's current embed handling:

- Backend (``references/open-webui`` ``socket/main.py``) only persists events
  whose ``type == "embeds"``, and it extends the new payload with whatever is
  already in the message's ``embeds`` column.
- Frontend (``Chat.svelte``) replaces ``message.embeds`` with ``data.embeds``
  for *both* ``embeds`` and ``chat:message:embeds`` events.

Therefore the rich-progress ``finalize`` has to:

1. Emit ``type: embeds`` with ``[final_html]`` only (persisted, no dup).
2. If any non-shell tool embed was observed mid-run, follow up with
   ``type: chat:message:embeds`` containing ``[final_html, *preserved]`` to
   restore the live view without writing those preserved embeds to DB again.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Optional

import pytest

tools_dir = Path(__file__).parent.parent.parent / "tools"
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

import llm_review  # noqa: E402


@pytest.fixture
def fake_chats(monkeypatch):
    """Inject a fake ``open_webui.models.chats`` into ``sys.modules``.

    Needed because pytest's import ordering for this repo (some earlier
    test or llm_review's own lazy import inside a try/except) can leave
    ``open_webui.models`` cached as a namespace package with an empty
    ``__path__``, making a plain ``import open_webui.models.chats`` in
    the test body fail with ``ModuleNotFoundError``.

    The fake stays active only for the duration of the test because
    ``monkeypatch.setitem`` restores the original entry on teardown.
    ``init_rich`` does ``from open_webui.models.chats import Chats``,
    which resolves against ``sys.modules`` — so replacing the module
    there is enough to intercept the call without touching real DB.
    """

    class FakeChats:
        get_message_by_id_and_message_id = staticmethod(lambda c, m: None)

    fake_module = types.ModuleType("open_webui.models.chats")
    fake_module.Chats = FakeChats  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", fake_module)
    return FakeChats


class RecordingEmitter:
    """Minimal async event_emitter substitute that captures every call."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def __call__(self, event: dict) -> None:
        self.events.append(event)


async def _make_initialized_emitter(
    *, chat_id: str | None = None
) -> tuple[llm_review.EventEmitter, RecordingEmitter]:
    raw = RecordingEmitter()
    ee = llm_review.EventEmitter(
        raw,
        enabled=True,
        message_id="msg-1",
        chat_id=chat_id,
        render_markdown=False,
        force_theme="auto",
    )
    await ee.init_rich(
        topic="Test topic",
        total_steps=3,
        agents=[
            {"id": "m1", "persona": "Alpha"},
            {"id": "m2", "persona": "Beta"},
            {"id": "m3", "persona": "Gamma"},
        ],
    )
    return ee, raw


@pytest.mark.asyncio
async def test_init_rich_emits_persisted_shell() -> None:
    ee, raw = await _make_initialized_emitter()

    assert ee._rich_initialized is True
    assert raw.events, "init_rich should have emitted at least one event"

    shell_event = raw.events[0]
    assert shell_event["type"] == "embeds", "shell must be persisted"
    embeds = shell_event["data"]["embeds"]
    assert len(embeds) == 1
    assert ee._shell_marker in embeds[0]


@pytest.mark.asyncio
async def test_wrap_emitter_records_non_shell_embeds() -> None:
    ee, raw = await _make_initialized_emitter()
    assert ee.event_emitter is not None

    await ee.event_emitter(
        {"type": "embeds", "data": {"embeds": ["<div>tool-a</div>"]}}
    )
    await ee.event_emitter(
        {"type": "embeds", "data": {"embeds": ["<div>tool-b</div>"]}}
    )

    assert ee._preserved_embeds == ["<div>tool-a</div>", "<div>tool-b</div>"]


@pytest.mark.asyncio
async def test_wrap_emitter_dedupes_preserved() -> None:
    ee, raw = await _make_initialized_emitter()
    assert ee.event_emitter is not None

    for _ in range(3):
        await ee.event_emitter(
            {"type": "embeds", "data": {"embeds": ["<div>tool-a</div>"]}}
        )
    await ee.event_emitter(
        {
            "type": "embeds",
            "data": {"embeds": [{"kind": "rich", "body": "b"}]},
        }
    )
    await ee.event_emitter(
        {
            "type": "embeds",
            "data": {"embeds": [{"body": "b", "kind": "rich"}]},
        }
    )

    assert ee._preserved_embeds == [
        "<div>tool-a</div>",
        {"kind": "rich", "body": "b"},
    ]


@pytest.mark.asyncio
async def test_wrap_emitter_skips_shell_in_preservation() -> None:
    ee, raw = await _make_initialized_emitter()
    assert ee.event_emitter is not None
    assert ee._shell_html is not None

    await ee.event_emitter(
        {"type": "embeds", "data": {"embeds": [ee._shell_html]}}
    )

    assert ee._preserved_embeds == []


@pytest.mark.asyncio
async def test_wrap_emitter_prepends_shell_to_live_correction() -> None:
    """A downstream tool's own `chat:message:embeds` live-fix must not kick
    our progress shell out of the live view. This caught a real regression
    where Embed Probe's cumulative live correction wiped LLM Review's
    shell after every probe call."""
    ee, raw = await _make_initialized_emitter()
    assert ee.event_emitter is not None
    assert ee._shell_html is not None

    await ee.event_emitter(
        {
            "type": "chat:message:embeds",
            "data": {"embeds": ["<div>running-1</div>", "<div>running-2</div>"]},
        }
    )

    wrapped = raw.events[-1]
    assert wrapped["type"] == "chat:message:embeds", "original type preserved"
    payload = wrapped["data"]["embeds"]
    assert payload[0] is ee._shell_html, "shell must be re-prepended"
    assert payload[1:] == ["<div>running-1</div>", "<div>running-2</div>"]


@pytest.mark.asyncio
async def test_wrap_emitter_does_not_preserve_from_live_correction() -> None:
    """`chat:message:embeds` payloads are live-only duplicates of already-
    persisted embeds. Preserving from them would make finalize re-issue an
    embed that was never actually written to DB, which would vanish on
    reload."""
    ee, raw = await _make_initialized_emitter()
    assert ee.event_emitter is not None

    await ee.event_emitter(
        {
            "type": "chat:message:embeds",
            "data": {"embeds": ["<div>never-persisted</div>"]},
        }
    )

    assert ee._preserved_embeds == []


@pytest.mark.asyncio
async def test_finalize_persists_final_only_then_live_correction() -> None:
    ee, raw = await _make_initialized_emitter()
    assert ee.event_emitter is not None

    await ee.event_emitter(
        {"type": "embeds", "data": {"embeds": ["<div>tool-a</div>"]}}
    )
    await ee.event_emitter(
        {"type": "embeds", "data": {"embeds": ["<div>tool-b</div>"]}}
    )

    pre_count = len(raw.events)
    await ee.finalize(summary={"topic": "Test topic", "num_rounds": 1, "agents": []})

    new_events = raw.events[pre_count:]
    assert len(new_events) == 2, (
        "finalize should emit exactly a persisted embeds event and a live correction"
    )

    persisted, live = new_events

    assert persisted["type"] == "embeds"
    persisted_embeds = persisted["data"]["embeds"]
    assert len(persisted_embeds) == 1
    assert "LLM Review" in persisted_embeds[0] or "<" in persisted_embeds[0]
    # The persisted payload must not carry preserved tool embeds — those
    # are already in DB via their own persist events and would duplicate.
    assert all(
        "tool-a" not in e and "tool-b" not in e
        for e in persisted_embeds
        if isinstance(e, str)
    )

    assert live["type"] == "chat:message:embeds"
    live_embeds = live["data"]["embeds"]
    assert live_embeds[0] is persisted_embeds[0]
    assert live_embeds[1:] == ["<div>tool-a</div>", "<div>tool-b</div>"]
    # Shell must NOT leak into the final live view.
    assert not any(
        isinstance(e, str) and ee._shell_marker in e for e in live_embeds
    )


@pytest.mark.asyncio
async def test_finalize_without_preserved_skips_live_correction() -> None:
    ee, raw = await _make_initialized_emitter()

    pre_count = len(raw.events)
    await ee.finalize(summary={"topic": "Test topic", "num_rounds": 1, "agents": []})

    new_events = raw.events[pre_count:]
    assert len(new_events) == 1
    assert new_events[0]["type"] == "embeds"


@pytest.mark.asyncio
async def test_finalize_is_noop_when_not_initialized() -> None:
    raw = RecordingEmitter()
    ee = llm_review.EventEmitter(
        raw, enabled=False, message_id=None, render_markdown=False
    )

    await ee.finalize(summary={"topic": "x", "num_rounds": 0, "agents": []})

    assert raw.events == []


def test_embed_dedupe_key_handles_unhashable_and_equivalent_dicts() -> None:
    k1 = llm_review._embed_dedupe_key({"a": 1, "b": 2})
    k2 = llm_review._embed_dedupe_key({"b": 2, "a": 1})
    assert k1 == k2, "dict key should be order-insensitive"

    class Weird:
        def __repr__(self) -> str:
            return "weird"

    assert llm_review._embed_dedupe_key(Weird()).startswith(("j:", "r:"))


@pytest.mark.asyncio
async def test_init_rich_seeds_preserved_from_db_embeds(fake_chats) -> None:
    """A tool that ran before LLM Review in the same turn already
    persisted an embed into the message's DB row. The backend would
    still have it on reload, but our upcoming shell emit replaces
    message.embeds in the live view — without an explicit restore the
    user would see the embed vanish for the whole run. init_rich reads
    DB, seeds preserved so finalize's live-fix includes it, and fires
    a second live-only event so it stays on screen right away."""

    def fake_get(chat_id, message_id):
        assert chat_id == "chat-1"
        assert message_id == "msg-1"
        return {"embeds": ["<div>pre-X</div>", "<div>pre-Y</div>"]}

    fake_chats.get_message_by_id_and_message_id = staticmethod(fake_get)

    ee, raw = await _make_initialized_emitter(chat_id="chat-1")

    assert ee._preserved_embeds == ["<div>pre-X</div>", "<div>pre-Y</div>"]

    assert raw.events[0]["type"] == "embeds"
    assert raw.events[0]["data"]["embeds"] == [ee._shell_html]

    assert raw.events[1]["type"] == "chat:message:embeds"
    assert raw.events[1]["data"]["embeds"] == [
        ee._shell_html,
        "<div>pre-X</div>",
        "<div>pre-Y</div>",
    ]

    pre = len(raw.events)
    await ee.finalize(summary={"topic": "t", "num_rounds": 1, "agents": []})
    new = raw.events[pre:]
    assert len(new) == 2
    live = new[1]
    assert live["type"] == "chat:message:embeds"
    assert live["data"]["embeds"][1:] == [
        "<div>pre-X</div>",
        "<div>pre-Y</div>",
    ]


@pytest.mark.asyncio
async def test_init_rich_skips_pre_existing_shells(fake_chats) -> None:
    """A leftover shell from a previous completed run would still be in
    DB (height:0, invisible). Don't seed it as a tool embed — it would
    re-issue our own old shell and also fail dedup against the new one."""
    stale_shell = "<div><!--llm-review-shell:oldtag--></div>"
    fake_chats.get_message_by_id_and_message_id = staticmethod(
        lambda c, m: {"embeds": [stale_shell, "<div>pre-real</div>"]}
    )

    ee, raw = await _make_initialized_emitter(chat_id="chat-1")

    assert ee._preserved_embeds == ["<div>pre-real</div>"]
    assert stale_shell not in ee._preserved_embeds


@pytest.mark.asyncio
async def test_init_rich_without_chat_id_skips_db_read(fake_chats) -> None:
    """If chat_id isn't available, skip the DB read entirely rather
    than guessing — avoids spurious errors and undefined behavior."""
    called = {"count": 0}

    def fail_get(chat_id, message_id):
        called["count"] += 1
        raise AssertionError("should not be called without chat_id")

    fake_chats.get_message_by_id_and_message_id = staticmethod(fail_get)

    ee, raw = await _make_initialized_emitter(chat_id=None)

    assert called["count"] == 0
    assert ee._preserved_embeds == []
    live_events = [e for e in raw.events if e["type"] == "chat:message:embeds"]
    assert live_events == [], "no live-fix when nothing to restore"


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", ["local:abc123", "channel:abc123"])
async def test_init_rich_skips_non_regular_chat_db_read(fake_chats, chat_id) -> None:
    called = {"count": 0}

    def fail_get(chat_id, message_id):
        called["count"] += 1
        raise AssertionError("should not read Chats for local/channel chat IDs")

    fake_chats.get_message_by_id_and_message_id = staticmethod(fail_get)

    ee, raw = await _make_initialized_emitter(chat_id=chat_id)

    assert called["count"] == 0
    assert ee._preserved_embeds == []
    live_events = [e for e in raw.events if e["type"] == "chat:message:embeds"]
    assert live_events == []


@pytest.mark.asyncio
async def test_init_rich_db_read_failure_does_not_break_init(fake_chats) -> None:
    """If the DB call raises (unusual setup, permissions, etc.), init
    must keep going — pre-existing embed restoration is best-effort."""

    def boom(chat_id, message_id):
        raise RuntimeError("simulated DB failure")

    fake_chats.get_message_by_id_and_message_id = staticmethod(boom)

    ee, raw = await _make_initialized_emitter(chat_id="chat-1")

    assert ee._rich_initialized is True
    assert ee._preserved_embeds == []


def test_is_rich_shell_embed_identifies_marker() -> None:
    marker = "<!--llm-review-shell:deadbeef-->"
    assert llm_review._is_rich_shell_embed(f"<div>x</div>{marker}", marker)
    assert not llm_review._is_rich_shell_embed("<div>x</div>", marker)
    assert not llm_review._is_rich_shell_embed({"html": marker}, marker)
    assert not llm_review._is_rich_shell_embed(None, marker)
