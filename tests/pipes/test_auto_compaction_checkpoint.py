from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from functions.pipe import auto_compaction_pipe as mod


def test_source_hash_is_deterministic_for_equivalent_canonical_payloads():
    left = [
        {
            "id": "volatile-a",
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "timestamp": 100,
            "usage": {"total_tokens": 1},
        }
    ]
    right = [
        {
            "id": "volatile-b",
            "timestamp": 200,
            "role": "user",
            "usage": {"total_tokens": 999},
            "content": [{"text": "hello", "type": "text"}],
        }
    ]

    assert mod.compute_source_hash(left) == mod.compute_source_hash(right)


def test_source_hash_ignores_core_file_upload_transients():
    stable = {
        "role": "user",
        "content": [
            {"type": "text", "text": "inspect this file"},
            {
                "type": "file",
                "file": {
                    "id": "file-1",
                    "name": "report.pdf",
                    "signed_url": "https://signed.example/a?token=old",
                    "metadata": {"upload_id": "tmp-a", "sha256": "abc123"},
                },
            },
        ],
        "files": [
            {
                "type": "file",
                "id": "file-1",
                "url": "file-1",
                "name": "report.pdf",
                "collection_name": "knowledge",
                "content_type": "application/pdf",
                "size": 12345,
                "status": "uploaded",
                "error": "",
                "itemId": "ui-item-a",
                "file": {
                    "id": "file-1",
                    "user_id": "user-1",
                    "hash": "abc123",
                    "filename": "report.pdf",
                    "path": "/var/lib/open-webui/uploads/a/report.pdf",
                    "data": {"content": "extracted text", "status": "pending"},
                    "meta": {
                        "name": "report.pdf",
                        "content_type": "application/pdf",
                        "collection_name": "knowledge",
                        "data": {"upload_id": "tmp-a"},
                    },
                    "created_at": 100,
                    "updated_at": 100,
                },
            }
        ],
    }
    changed_transient = copy.deepcopy(stable)
    changed_transient["content"][1]["file"]["signed_url"] = "https://signed.example/a?token=new"
    changed_transient["content"][1]["file"]["metadata"]["upload_id"] = "tmp-b"
    changed_transient["files"][0]["status"] = "processing"
    changed_transient["files"][0]["error"] = "temporary warning"
    changed_transient["files"][0]["itemId"] = "ui-item-b"
    changed_transient["files"][0]["size"] = 99999
    changed_transient["files"][0]["progress"] = {"phase": "extracting", "percent": 50}
    changed_transient["files"][0]["file"]["path"] = "/var/lib/open-webui/uploads/b/report.pdf"
    changed_transient["files"][0]["file"]["data"]["status"] = "done"
    changed_transient["files"][0]["file"]["meta"]["data"]["upload_id"] = "tmp-b"
    changed_transient["files"][0]["file"]["created_at"] = 200
    changed_transient["files"][0]["file"]["updated_at"] = 200
    changed_semantic = copy.deepcopy(stable)
    changed_semantic["files"][0]["file"]["hash"] = "def456"

    assert mod.compute_source_hash([stable]) == mod.compute_source_hash([changed_transient])
    assert mod.compute_source_hash([stable]) != mod.compute_source_hash([changed_semantic])


def test_source_hash_preserves_file_metadata_used_for_source_context():
    first = {
        "role": "user",
        "content": "Use attached page",
        "files": [
            {
                "type": "text",
                "name": "https://example.com/a",
                "url": "https://example.com/a",
                "context": "full",
                "file": {
                    "data": {"content": "same text"},
                    "meta": {"name": "page", "source": "https://example.com/a"},
                },
            }
        ],
    }
    second = copy.deepcopy(first)
    second["files"][0]["file"]["meta"]["source"] = "https://example.com/b"

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_preserves_web_search_attachment_urls_and_queries():
    first = {
        "role": "user",
        "content": "Search the web",
        "files": [
            {
                "collection_name": "web-search-collection",
                "name": "open webui auto compaction",
                "type": "web_search",
                "urls": ["https://example.com/old"],
                "queries": ["open webui auto compaction"],
                "status": "uploaded",
            }
        ],
    }
    changed_url = copy.deepcopy(first)
    changed_url["files"][0]["urls"] = ["https://example.com/new"]
    changed_query = copy.deepcopy(first)
    changed_query["files"][0]["queries"] = ["open webui checkpoint compaction"]

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([changed_url])
    assert mod.compute_source_hash([first]) != mod.compute_source_hash([changed_query])
    assert "status" not in mod.canonicalize_messages_for_source_hash([first])[0]["files"][0]


def test_source_hash_preserves_non_file_content_file_paths():
    first = {
        "role": "user",
        "content": [{"type": "profile", "file": {"path": "/workspace/a.txt", "name": "same"}}],
    }
    second = copy.deepcopy(first)
    second["content"][0]["file"]["path"] = "/workspace/b.txt"

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_preserves_semantic_paths_outside_file_metadata():
    first = {
        "role": "assistant",
        "content": [{"type": "input", "path": "/workspace/a.txt"}],
        "function_call": {"name": "read_file", "arguments": {"path": "/workspace/a.txt"}},
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "/workspace/a.txt"}},
            }
        ],
        "sources": [{"source": {"id": "doc-1", "name": "docs", "path": "docs/a.md"}}],
    }
    second = copy.deepcopy(first)
    second["content"][0]["path"] = "/workspace/b.txt"
    second["function_call"] = {"name": "read_file", "arguments": {"path": "/workspace/b.txt"}}
    second["tool_calls"][0]["function"]["arguments"]["path"] = "/workspace/b.txt"
    second["sources"][0]["source"]["path"] = "docs/b.md"

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_preserves_semantic_urls():
    first = {
        "role": "user",
        "content": "Use this reference URL",
        "sources": [{"title": "docs", "url": "https://example.com/a"}],
    }
    second = {
        "role": "user",
        "content": "Use this reference URL",
        "sources": [{"title": "docs", "url": "https://example.com/b"}],
    }

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_preserves_semantic_download_urls_outside_file_metadata():
    first = {
        "role": "user",
        "content": "Use this reference URL",
        "sources": [{"source": {"name": "docs", "download_url": "https://example.com/a"}}],
    }
    second = {
        "role": "user",
        "content": "Use this reference URL",
        "sources": [{"source": {"name": "docs", "download_url": "https://example.com/b"}}],
    }

    assert mod.compute_source_hash([first]) != mod.compute_source_hash([second])


def test_source_hash_ignores_retrieval_source_scores():
    stable = {
        "role": "assistant",
        "content": "answer",
        "sources": [
            {
                "source": {"id": "doc-1", "name": "docs", "type": "file"},
                "document": ["same chunk"],
                "metadata": [{"source": "docs/a.md"}],
                "distances": [0.1],
            }
        ],
    }
    changed_score = copy.deepcopy(stable)
    changed_score["sources"][0]["distances"] = [0.9]
    changed_document = copy.deepcopy(stable)
    changed_document["sources"][0]["document"] = ["different chunk"]

    assert mod.compute_source_hash([stable]) == mod.compute_source_hash([changed_score])
    assert mod.compute_source_hash([stable]) != mod.compute_source_hash([changed_document])


def test_profile_hash_only_uses_hard_compatibility_boundaries():
    baseline = mod.compute_profile_hash(
        target_model="gpt-4.1",
        summary_model="gpt-4.1-mini",
        wrapper_suffix="abc",
        summary_prompt="old prompt",
        keep_tail_messages=8,
    )
    ordinary_changes = mod.compute_profile_hash(
        target_model="claude-sonnet",
        summary_model="pipe.summary",
        wrapper_suffix="xyz",
        summary_prompt="new prompt",
        keep_tail_messages=32,
    )
    hard_change = mod.compute_profile_hash(summary_format_family="compact-user-summary-v2")

    assert ordinary_changes == baseline
    assert hard_change != baseline


def test_wrapper_ids_round_trip_special_target_model_ids():
    targets = [
        "openai.gpt-4.1",
        "ollama/llama3.2:latest",
        "provider:model/with.dots:and/slashes",
        "pipe.backed.target",
    ]

    for target in targets:
        wrapper_id = mod.build_wrapper_model_id("auto_compact", target)
        assert wrapper_id == f"auto_compact.{target}"
        assert wrapper_id.startswith("auto_compact.")
        decoded = mod.decode_wrapper_model_id(wrapper_id, expected_pipe_id="auto_compact")
        assert decoded.pipe_model_id == "auto_compact"
        assert decoded.target_model_id == target


def test_decode_rejects_malformed_or_wrong_pipe_wrapper_ids():
    with pytest.raises(ValueError, match="wrapper"):
        mod.decode_wrapper_model_id("not-a-wrapper", expected_pipe_id="auto_compact")

    with pytest.raises(ValueError, match="pipe"):
        mod.decode_wrapper_model_id("other_pipe.gpt", expected_pipe_id="auto_compact")


def test_checkpoint_table_contract_names_and_columns():
    table = mod.CHECKPOINT_TABLE

    assert table.name == "skyzi000_owui_ext_autocompact_checkpoint_v1"
    assert table.metadata.schema == mod.OPEN_WEBUI_DATABASE_SCHEMA
    assert {idx.name for idx in table.indexes} == {
        "skyzi000_owui_ext_accp_v1_prefix_idx",
        "skyzi000_owui_ext_accp_v1_recent_idx",
    }
    assert {constraint.name for constraint in table.constraints} >= {
        "skyzi000_owui_ext_accp_v1_lookup_uq"
    }
    assert set(table.c.keys()) == {
        "id",
        "namespace",
        "schema_version",
        "user_id",
        "chat_id",
        "pipe_model_id",
        "profile_hash",
        "source_message_count",
        "source_hash",
        "summary_text",
        "summary_meta",
        "state",
        "parent_checkpoint_id",
        "created_at",
        "updated_at",
        "last_used_at",
    }


@pytest.mark.asyncio
async def test_checkpoint_table_initialization_is_cached_on_request_state(monkeypatch):
    calls = []

    class DummyConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def run_sync(self, fn):
            calls.append(fn)

    class DummyEngine:
        def begin(self):
            return DummyConnection()

    request = type("Request", (), {"state": type("State", (), {})()})()
    monkeypatch.setattr(mod, "_CHECKPOINT_SCHEMA_READY", False)

    await mod.ensure_checkpoint_table_initialized(request=request, async_engine=DummyEngine())
    await mod.ensure_checkpoint_table_initialized(request=request, async_engine=DummyEngine())

    assert len(calls) == 1
    assert request.state.__dict__[mod.REQUEST_STATE_SCHEMA_READY_KEY] is True


@pytest.mark.asyncio
async def test_checkpoint_table_initialization_uses_process_cache_without_request(monkeypatch):
    calls = []

    class DummyConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def run_sync(self, fn):
            calls.append(fn)

    class DummyEngine:
        def begin(self):
            return DummyConnection()

    monkeypatch.setattr(mod, "_CHECKPOINT_SCHEMA_READY", False)

    await mod.ensure_checkpoint_table_initialized(request=None, async_engine=DummyEngine())
    await mod.ensure_checkpoint_table_initialized(request=None, async_engine=DummyEngine())

    assert len(calls) == 1


def test_generation_lock_cleanup_removes_unused_lock():
    key = ("ns", "user", "chat", "pipe", "profile", "source")
    lock = mod.get_generation_lock(key)

    assert mod._GENERATION_LOCKS[key] is lock

    mod.release_generation_lock(key, lock)

    assert key not in mod._GENERATION_LOCKS


def test_checkpoint_row_uses_stable_identity_and_contract_fields():
    row = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_model_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=12,
        summary_text="summary",
        summary_meta={"has_multimodal": False},
        parent_checkpoint_id="parent-1",
        now=123,
    )
    same = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_model_id="auto_compact",
        profile_hash="profile",
        source_hash="source",
        source_message_count=12,
        summary_text="new summary",
        summary_meta={},
        parent_checkpoint_id=None,
        now=999,
    )

    assert same["id"] == row["id"]
    assert row == {
        "id": row["id"],
        "namespace": mod.CHECKPOINT_NAMESPACE,
        "schema_version": 1,
        "user_id": "user-1",
        "chat_id": "chat-1",
        "pipe_model_id": "auto_compact",
        "profile_hash": "profile",
        "source_message_count": 12,
        "source_hash": "source",
        "summary_text": "summary",
        "summary_meta": {"has_multimodal": False},
        "state": "ready",
        "parent_checkpoint_id": "parent-1",
        "created_at": 123,
        "updated_at": 123,
        "last_used_at": 123,
    }


def test_parent_checkpoint_selection_requires_exact_source_hash():
    rows = [
        {"id": "short", "source_message_count": 2, "source_hash": "h2", "state": "ready"},
        {"id": "mismatch", "source_message_count": 4, "source_hash": "wrong", "state": "ready"},
        {"id": "long", "source_message_count": 3, "source_hash": "h3", "state": "ready"},
    ]

    parent = mod.select_longest_matching_parent(rows, {2: "h2", 3: "h3", 4: "h4"})

    assert parent["id"] == "long"


def test_ready_checkpoint_lookup_ignores_non_ready_rows():
    rows = [
        {"id": "pending", "state": "pending", "source_hash": "h1"},
        {"id": "ready", "state": "ready", "source_hash": "h1"},
    ]

    assert mod.select_ready_checkpoint(rows, source_hash="h1")["id"] == "ready"


@pytest.mark.asyncio
async def test_checkpoint_exact_hit_uses_ready_summary_without_calling_summary_factory(monkeypatch):
    existing = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_hash": mod.compute_source_hash([{"role": "user", "content": "old"}]),
        "summary_text": "existing summary",
    }

    async def noop_initialize(**kwargs):
        return None

    class HitStore:
        def __init__(self):
            self.touched = []

        async def lookup_ready(self, **kwargs):
            return existing

        async def touch(self, checkpoint_id):
            self.touched.append(checkpoint_id)
            return True

    store = HitStore()

    async def summary_factory(parent):
        raise AssertionError("summary factory must not run for an exact checkpoint hit")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)

    result = await mod._get_or_create_checkpoint_summary(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_model_id="auto_compact",
        source_messages=[{"role": "user", "content": "old"}],
        summary_meta={},
        summary_factory=summary_factory,
    )

    assert result == "existing summary"
    assert store.touched == ["checkpoint-1"]


@pytest.mark.asyncio
async def test_checkpoint_exact_hit_survives_last_used_update_failure(monkeypatch):
    existing = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_hash": mod.compute_source_hash([{"role": "user", "content": "old"}]),
        "summary_text": "existing summary",
    }

    async def noop_initialize(**kwargs):
        return None

    class HitStore:
        async def lookup_ready(self, **kwargs):
            return existing

        async def touch(self, checkpoint_id):
            raise RuntimeError("touch failed")

    async def summary_factory(parent):
        raise AssertionError("summary factory must not run when touch fails")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: HitStore())

    result = await mod._get_or_create_checkpoint_summary(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_model_id="auto_compact",
        source_messages=[{"role": "user", "content": "old"}],
        summary_meta={},
        summary_factory=summary_factory,
    )

    assert result == "existing summary"


@pytest.mark.asyncio
async def test_checkpoint_summary_returns_generated_summary_when_insert_fails(monkeypatch):
    calls = []

    async def noop_initialize(**kwargs):
        return None

    class FailingInsertStore:
        async def lookup_ready(self, **kwargs):
            return None

        async def find_longest_parent(self, **kwargs):
            return None

        async def insert_ready(self, row):
            raise RuntimeError("db down")

    async def summary_factory(parent):
        calls.append(parent)
        return "generated summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: FailingInsertStore())

    result = await mod._get_or_create_checkpoint_summary(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_model_id="auto_compact",
        source_messages=[{"role": "user", "content": "old"}],
        summary_meta={},
        summary_factory=summary_factory,
    )

    assert result == "generated summary"
    assert calls == [None]


@pytest.mark.asyncio
async def test_checkpoint_store_concurrent_insert_conflict_returns_existing_ready_row():
    from sqlalchemy.exc import IntegrityError

    existing = {
        "id": "checkpoint-1",
        "namespace": "auto_compact",
        "user_id": "user-1",
        "chat_id": "chat-1",
        "pipe_model_id": "auto_compact",
        "profile_hash": "profile",
        "source_hash": "source",
        "state": "ready",
        "summary_text": "existing summary",
    }

    class FakeMappings:
        def first(self):
            return existing

    class FakeResult:
        def mappings(self):
            return FakeMappings()

    class FakeDb:
        def __init__(self):
            self.execute_calls = 0
            self.commit_calls = 0
            self.rollback_calls = 0

        async def execute(self, statement):
            self.execute_calls += 1
            if self.execute_calls == 1:
                raise IntegrityError("insert", {}, Exception("duplicate"))
            return FakeResult()

        async def commit(self):
            self.commit_calls += 1

        async def rollback(self):
            self.rollback_calls += 1

    db = FakeDb()
    store = mod.CheckpointStore(db=db)
    row = {
        "id": "checkpoint-1",
        "namespace": "auto_compact",
        "user_id": "user-1",
        "chat_id": "chat-1",
        "pipe_model_id": "auto_compact",
        "profile_hash": "profile",
        "source_hash": "source",
        "state": "ready",
        "source_message_count": 1,
        "summary_text": "generated summary",
        "summary_meta": {},
        "schema_version": 1,
        "parent_checkpoint_id": None,
        "created_at": 1,
        "updated_at": 1,
        "last_used_at": 1,
    }

    result = await store.insert_ready(row)

    assert result == existing
    assert db.execute_calls == 2
    assert db.commit_calls == 0
    assert db.rollback_calls == 1


@pytest.mark.asyncio
async def test_checkpoint_miss_generates_summary_and_stores_ready_row(monkeypatch):
    inserted_rows = []

    async def noop_initialize(**kwargs):
        return None

    class MissStore:
        async def lookup_ready(self, **kwargs):
            return None

        async def find_longest_parent(self, **kwargs):
            return None

        async def insert_ready(self, row):
            inserted_rows.append(row)
            return row

    async def summary_factory(parent):
        assert parent is None
        return "generated summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: MissStore())

    source_messages = [{"role": "user", "content": "old"}]
    result = await mod._get_or_create_checkpoint_summary(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_model_id="auto_compact",
        source_messages=source_messages,
        summary_meta={"has_multimodal": False},
        summary_factory=summary_factory,
    )

    assert result == "generated summary"
    assert len(inserted_rows) == 1
    assert inserted_rows[0]["state"] == "ready"
    assert inserted_rows[0]["source_hash"] == mod.compute_source_hash(source_messages)
    assert inserted_rows[0]["summary_text"] == "generated summary"
    assert inserted_rows[0]["parent_checkpoint_id"] is None


@pytest.mark.asyncio
async def test_checkpoint_extension_uses_parent_summary_and_stores_lineage(monkeypatch):
    parent = {
        "id": "parent-1",
        "state": "ready",
        "source_message_count": 1,
        "source_hash": mod.compute_source_hash([{"role": "user", "content": "old"}]),
        "summary_text": "parent summary",
        "summary_meta": {},
    }
    calls = []
    inserted_rows = []

    async def noop_initialize(**kwargs):
        return None

    class ParentStore:
        async def lookup_ready(self, **kwargs):
            return None

        async def find_longest_parent(self, **kwargs):
            calls.append(("parent_lookup", kwargs["prefix_hashes"]))
            return parent

        async def insert_ready(self, row):
            inserted_rows.append(row)
            return row

    async def summary_factory(candidate_parent):
        calls.append(("summary_parent", candidate_parent))
        return "extended summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentStore())

    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "new"},
    ]
    result = await mod._get_or_create_checkpoint_summary(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_model_id="auto_compact",
        source_messages=source_messages,
        summary_meta={"has_multimodal": False},
        summary_factory=summary_factory,
    )

    assert result == "extended summary"
    assert ("summary_parent", parent) in calls
    assert len(inserted_rows) == 1
    assert inserted_rows[0]["source_hash"] == mod.compute_source_hash(source_messages)
    assert inserted_rows[0]["source_message_count"] == 2
    assert inserted_rows[0]["parent_checkpoint_id"] == "parent-1"


@pytest.mark.asyncio
async def test_compact_body_extends_from_parent_summary_plus_delta(monkeypatch):
    parent = {
        "id": "parent-1",
        "state": "ready",
        "source_message_count": 2,
        "source_hash": mod.compute_source_hash(
            [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
            ]
        ),
        "summary_text": "parent summary",
        "summary_meta": {},
    }
    captured = {}

    async def noop_initialize(**kwargs):
        return None

    class ParentStore:
        async def lookup_ready(self, **kwargs):
            return None

        async def find_longest_parent(self, **kwargs):
            return parent

        async def insert_ready(self, row):
            return row

    async def generate_summary_text(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "extended summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "model": "target",
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "delta"},
            {"role": "assistant", "content": "delta answer"},
            {"role": "user", "content": "active"},
        ],
    }

    compacted, did_compact = await mod._compact_body(
        request=SimpleNamespace(state=SimpleNamespace()),
        user={"id": "user-1"},
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_model_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        keep_tail_messages=1,
        preserve_latest_tool_rounds=0,
        aggressive=False,
    )

    assert did_compact is True
    assert captured["source_messages"][0]["content"].endswith("parent summary")
    assert captured["source_messages"][1:] == [
        {"role": "user", "content": "delta"},
        {"role": "assistant", "content": "delta answer"},
    ]
    assert "extended summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][-1] == {"role": "user", "content": "active"}


@pytest.mark.asyncio
async def test_compact_body_uses_canonical_checkpoint_source_for_summary(monkeypatch):
    captured = {}

    async def noop_initialize(**kwargs):
        return None

    class EmptyStore:
        async def lookup_ready(self, **kwargs):
            return None

        async def find_longest_parent(self, **kwargs):
            return None

        async def insert_ready(self, row):
            return row

    async def generate_summary_text(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: EmptyStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "model": "target",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": "old file",
                "files": [
                    {
                        "type": "text",
                        "name": "https://example.com/a",
                        "url": "https://example.com/a",
                        "status": "processing",
                        "progress": {"phase": "extracting"},
                        "file": {
                            "id": "file-1",
                            "filename": "page.txt",
                            "path": "/var/lib/open-webui/uploads/a/page.txt",
                            "hash": "abc123",
                            "data": {"content": "page text", "status": "pending"},
                            "meta": {"source": "https://example.com/a", "data": {"upload_id": "tmp-a"}},
                            "created_at": 100,
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    compacted, did_compact = await mod._compact_body(
        request=SimpleNamespace(state=SimpleNamespace()),
        user={"id": "user-1"},
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_model_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        keep_tail_messages=1,
        preserve_latest_tool_rounds=0,
        aggressive=False,
    )

    assert did_compact is True
    summary_file = captured["source_messages"][0]["files"][0]
    assert summary_file == {
        "file": {
            "data": {"content": "page text"},
            "filename": "page.txt",
            "hash": "abc123",
            "id": "file-1",
            "meta": {"source": "https://example.com/a"},
        },
        "name": "https://example.com/a",
        "type": "text",
        "url": "https://example.com/a",
    }
    assert "summary" in compacted["messages"][0]["content"]


@pytest.mark.asyncio
async def test_checkpoint_reuse_survives_target_and_summary_model_changes(monkeypatch):
    existing = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_hash": mod.compute_source_hash([{"role": "user", "content": "old"}]),
        "summary_text": "existing summary",
    }
    lookup_keys = []

    async def noop_initialize(**kwargs):
        return None

    class HitStore:
        async def lookup_ready(self, **kwargs):
            lookup_keys.append(kwargs)
            return existing

        async def touch(self, checkpoint_id):
            return True

    async def summary_factory(parent):
        raise AssertionError("checkpoint reuse must not depend on current target or summary model")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: HitStore())

    source_messages = [{"role": "user", "content": "old"}]
    first = await mod._get_or_create_checkpoint_summary(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_model_id="auto_compact",
        source_messages=source_messages,
        summary_meta={},
        summary_factory=summary_factory,
    )
    second = await mod._get_or_create_checkpoint_summary(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_model_id="auto_compact",
        source_messages=source_messages,
        summary_meta={"summary_model": "changed", "target_model": "changed"},
        summary_factory=summary_factory,
    )

    assert first == "existing summary"
    assert second == "existing summary"
    assert lookup_keys[0]["profile_hash"] == lookup_keys[1]["profile_hash"]
    assert lookup_keys[0]["source_hash"] == lookup_keys[1]["source_hash"]


@pytest.mark.asyncio
async def test_parent_checkpoint_failure_does_not_resubmit_raw_prefix(monkeypatch):
    parent = {
        "id": "parent-1",
        "state": "ready",
        "source_message_count": 1,
        "source_hash": mod.compute_source_hash([{"role": "user", "content": "old"}]),
        "summary_text": "parent summary",
        "summary_meta": {},
    }
    calls = []

    async def noop_initialize(**kwargs):
        return None

    class ParentStore:
        async def lookup_ready(self, **kwargs):
            return None

        async def find_longest_parent(self, **kwargs):
            return parent

        async def insert_ready(self, row):
            raise AssertionError("insert should not run when summary fails")

    async def summary_factory(candidate_parent):
        calls.append(candidate_parent)
        raise RuntimeError("summary context overflow")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentStore())

    with pytest.raises(mod.ParentCheckpointExtensionFailed) as exc_info:
        await mod._get_or_create_checkpoint_summary(
            request=None,
            user_id="user-1",
            chat_id="chat-1",
            pipe_model_id="auto_compact",
            source_messages=[
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "new"},
            ],
            summary_meta={},
            summary_factory=summary_factory,
        )

    assert calls == [parent]
    assert exc_info.value.parent == parent


@pytest.mark.asyncio
async def test_compact_body_uses_parent_checkpoint_delta_when_extension_summary_fails(monkeypatch):
    parent = {
        "id": "parent-1",
        "state": "ready",
        "source_message_count": 2,
        "source_hash": mod.compute_source_hash(
            [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
            ]
        ),
        "summary_text": "parent summary",
        "summary_meta": {},
    }

    async def noop_initialize(**kwargs):
        return None

    class ParentStore:
        async def lookup_ready(self, **kwargs):
            return None

        async def find_longest_parent(self, **kwargs):
            return parent

        async def insert_ready(self, row):
            raise AssertionError("failed extension summaries must not insert a full-prefix checkpoint")

    async def generate_summary_text(**kwargs):
        raise mod.RetryableContextOverflow("summary context overflow")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "model": "target",
        "stream": True,
        "previous_response_id": "resp-old",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "delta"},
            {"role": "assistant", "content": "delta answer"},
            {"role": "user", "content": "active"},
        ],
    }

    compacted, did_compact = await mod._compact_body(
        request=SimpleNamespace(state=SimpleNamespace()),
        user={"id": "user-1"},
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_model_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        keep_tail_messages=1,
        preserve_latest_tool_rounds=0,
        aggressive=False,
    )

    assert did_compact is True
    assert "previous_response_id" not in compacted
    assert compacted["messages"][0] == {"role": "system", "content": "system"}
    assert "parent summary" in compacted["messages"][1]["content"]
    assert {"role": "user", "content": "old"} not in compacted["messages"]
    assert {"role": "assistant", "content": "old answer"} not in compacted["messages"]
    assert compacted["messages"][2:] == [
        {"role": "user", "content": "delta"},
        {"role": "assistant", "content": "delta answer"},
        {"role": "user", "content": "active"},
    ]
