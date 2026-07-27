from __future__ import annotations

from types import SimpleNamespace
import asyncio
import copy
import inspect
import json
import math
import sys
import threading
import types

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

from functions.pipe import auto_compact as mod


TRANSIENT_MARKER = r"(?s)<SYSTEM_CONTEXT>.*</SYSTEM_CONTEXT>\s*\Z"


class ClaimCheckpointStore:
    """CheckpointStore stand-in implementing the DB claim interface over shared row dicts."""

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.claimed_rows = []
        self.completed_rows = []
        self.released = []
        self.touched = []

    def _match(self, source_hash):
        for row in self.rows:
            if row.get("source_hash") == source_hash:
                return row
        return None

    async def lookup_any(self, **kwargs):
        row = self._match(kwargs["source_hash"])
        return dict(row) if row else None

    async def lookup_ready(self, **kwargs):
        row = self._match(kwargs["source_hash"])
        if row is not None and row.get("state") == "ready":
            return dict(row)
        return None

    async def find_longest_parent(self, **kwargs):
        return mod.select_longest_matching_parent(self.rows, kwargs["source_messages"])

    async def find_longest_pending_parent(self, **kwargs):
        return mod.select_longest_matching_checkpoint(
            self.rows,
            kwargs["source_messages"],
            states={"pending"},
        )

    async def claim_pending(self, row):
        if self._match(row["source_hash"]) is not None:
            return False
        stored = dict(row)
        self.rows.append(stored)
        self.claimed_rows.append(dict(stored))
        return True

    async def reclaim_pending(self, checkpoint_id, *, claim_token, expires_at, now=None):
        return False

    async def extend_claim(self, checkpoint_id, *, claim_token, expires_at):
        return True

    async def release_claim(self, checkpoint_id, *, claim_token):
        for row in list(self.rows):
            if (
                row.get("id") == checkpoint_id
                and row.get("state") == "pending"
                and row.get("claim_token") == claim_token
            ):
                self.rows.remove(row)
                self.released.append(checkpoint_id)
                return True
        return False

    async def complete_pending(
        self,
        checkpoint_id,
        *,
        claim_token,
        summary_text,
        parent_checkpoint_id,
        summary_token_count=None,
        generation_lease_id=None,
        generation_lease_claim_token=None,
        now=None,
    ):
        timestamp = int(mod.time.time()) if now is None else int(now)
        if generation_lease_id is not None and generation_lease_claim_token is not None:
            lease = next(
                (
                    row
                    for row in self.rows
                    if row.get("id") == generation_lease_id
                    and row.get("namespace") == mod.CHECKPOINT_GENERATION_LEASE_NAMESPACE
                    and row.get("state") == "pending"
                    and row.get("claim_token") == generation_lease_claim_token
                    and int(row.get("claim_expires_at") or 0) > timestamp
                ),
                None,
            )
            if lease is None:
                return None
        for row in self.rows:
            if (
                row.get("id") == checkpoint_id
                and row.get("state") == "pending"
                and row.get("claim_token") == claim_token
            ):
                row.update(
                    state="ready",
                    summary_text=summary_text,
                    parent_checkpoint_id=parent_checkpoint_id,
                    summary_token_count=summary_token_count,
                    claim_token=None,
                    claim_expires_at=None,
                )
                self.completed_rows.append(dict(row))
                return dict(row)
        return None

    async def touch(self, checkpoint_id, *, now=None):
        self.touched.append(checkpoint_id)
        return True


class PendingTransitionCheckpointStore(ClaimCheckpointStore):
    def __init__(self, rows, *, ready_after_lookup):
        super().__init__(rows)
        self.ready_after_lookup = ready_after_lookup
        self.lookup_any_count = 0

    async def lookup_any(self, **kwargs):
        self.lookup_any_count += 1
        row = self._match(kwargs["source_hash"])
        if (
            row is not None
            and row.get("state") == "pending"
            and self.ready_after_lookup is not None
            and self.lookup_any_count >= self.ready_after_lookup
        ):
            row.update(
                state="ready",
                summary_text="ready parent summary",
                claim_token=None,
                claim_expires_at=None,
            )
        return dict(row) if row else None


def install_fake_open_webui_user_model(monkeypatch):
    class FakeUserModel:
        def __init__(self, **data):
            self.__dict__.update(data)

    users_module = types.ModuleType("open_webui.models.users")
    users_module.UserModel = FakeUserModel
    monkeypatch.setitem(sys.modules, "open_webui.models.users", users_module)
    return FakeUserModel


def install_fake_open_webui_config(monkeypatch, config_cls):
    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = config_cls
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    try:
        import open_webui.models as core_models

        monkeypatch.setattr(core_models, "config", config_module, raising=False)
    except Exception:
        pass
    return config_module


def install_unavailable_open_webui_config(monkeypatch):
    class FakeConfig:
        @staticmethod
        async def get(key):
            raise RuntimeError("config unavailable")

        @staticmethod
        async def get_many(*keys):
            raise RuntimeError("config unavailable")

    return install_fake_open_webui_config(monkeypatch, FakeConfig)


def _file(file_id, *, file_type="file"):
    return {"id": file_id, "type": file_type, "name": f"{file_id}.txt"}


class FakeModelMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    profile_image_url: str | None = None
    description: str | None = None
    capabilities: dict | None = None


class FakeModelParams(BaseModel):
    model_config = ConfigDict(extra="allow")


class FakeModelForm(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    base_model_id: str | None = None
    name: str
    meta: FakeModelMeta
    params: FakeModelParams
    access_grants: list[dict | None] | None = None
    is_active: bool = True


def install_fake_open_webui_model_modules(monkeypatch, records=None):
    records = records or {}
    calls = {
        "inserted": [],
        "updated": [],
        "deleted": [],
        "get_by_id": [],
        "get_all": 0,
    }

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return SimpleNamespace(id=function_id, user_id="function-owner")

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            calls["get_by_id"].append(model_id)
            return records.get(model_id)

        @staticmethod
        async def get_all_models():
            calls["get_all"] += 1
            return list(records.values())

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["inserted"].append((model_form, user_id))
            records[model_form.id] = model_form
            return model_form

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["updated"].append((model_id, model_form))
            records[model_id] = model_form
            return model_form

        @staticmethod
        async def delete_model_by_id(model_id):
            calls["deleted"].append(model_id)
            records.pop(model_id, None)
            return True

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    return records, calls


def test_target_filter_excludes_own_wrappers_and_arena_but_allows_other_pipes():
    models = {
        "gpt-4.1": {"id": "gpt-4.1", "name": "GPT"},
        mod.build_wrapper_model_id("auto_compact", "gpt-4.1"): {
            "id": mod.build_wrapper_model_id("auto_compact", "gpt-4.1"),
            "name": "wrapped",
            "pipe": {"type": "pipe"},
        },
        "arena-model": {"id": "arena-model", "name": "Arena", "owned_by": "arena", "arena": True},
        "other_pipe.child": {"id": "other_pipe.child", "name": "Other Pipe", "pipe": {"type": "pipe"}},
    }

    targets = mod.filter_target_models(models.values(), mod.Pipe.Valves())

    assert [m["id"] for m in targets] == ["gpt-4.1", "other_pipe.child"]


def test_target_filter_excludes_other_auto_compaction_wrappers_across_cache_representations():
    wrapper_id = mod.build_wrapper_model_id("compact_b", "gpt-4.1")
    state = SimpleNamespace(
        MODELS={
            wrapper_id: {
                "id": wrapper_id,
                "name": "Compact B",
                "info": {
                    "meta": {
                        "auto_compaction": {
                            "pipe_function_id": "compact_b",
                            "target_model_id": "gpt-4.1",
                        }
                    }
                },
            },
            "gpt-4.1": {"id": "gpt-4.1", "name": "GPT"},
            "other_pipe.child": {
                "id": "other_pipe.child",
                "name": "Other Pipe",
                "pipe": {"type": "pipe"},
            },
        },
        BASE_MODELS=[{"id": wrapper_id, "name": "Compact B"}],
        OPENAI_MODELS=[],
        OLLAMA_MODELS=[],
    )

    candidates = mod._iter_cache_models_from_state(state, disabled_provider_attrs=set())
    assert [model["id"] for model in candidates].count(wrapper_id) == 2

    targets = mod.filter_target_models(candidates, mod.Pipe.Valves())

    assert [model["id"] for model in targets] == ["gpt-4.1", "other_pipe.child"]


def test_target_filter_excludes_presets_based_on_own_wrappers():
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "gpt-4.1")
    models = [
        {"id": "gpt-4.1", "name": "GPT"},
        {"id": "compact-preset", "name": "Compact Preset", "info": {"base_model_id": wrapper_id}},
        {"id": "top-level-compact-preset", "name": "Compact Preset 2", "base_model_id": wrapper_id},
        {"id": "normal-preset", "name": "Normal Preset", "info": {"base_model_id": "gpt-4.1"}},
    ]

    targets = mod.filter_target_models(models, mod.Pipe.Valves())

    assert [m["id"] for m in targets] == ["gpt-4.1", "normal-preset"]


def test_wrapper_model_form_does_not_copy_target_hidden():
    form_payload = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model={
            "id": "target",
            "name": "Target",
            "info": {
                "meta": {
                    "hidden": True,
                    "description": "target description",
                },
                "params": {"temperature": 0.2},
            },
        },
        valves=mod.Pipe.Valves(),
    )

    assert form_payload["meta"]["description"] == "target description"
    assert "hidden" not in form_payload["meta"]


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_preserves_existing_wrapper_hidden_when_updating(monkeypatch):
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    existing_wrapper = FakeModelForm(
        id=wrapper_id,
        base_model_id=None,
        name="Old Wrapper Name",
        params=FakeModelParams(temperature=0.1),
        meta=FakeModelMeta(
            hidden=True,
            auto_compaction={
                "pipe_function_id": "auto_compact",
                "target_model_id": "target",
            },
        ),
        access_grants=[],
        is_active=True,
    )
    _, calls = install_fake_open_webui_model_modules(monkeypatch, {wrapper_id: existing_wrapper})

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[
            {
                "id": "target",
                "name": "Target",
                "info": {"params": {"temperature": 0.2}},
            }
        ],
        valves=mod.Pipe.Valves(),
    )

    assert len(calls["updated"]) == 1
    updated_id, updated_form = calls["updated"][0]
    assert updated_id == wrapper_id
    assert updated_form.name == "Target (AutoCompact)"
    assert updated_form.meta.hidden is True


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_reuses_bulk_snapshot_on_steady_state(monkeypatch):
    records, calls = install_fake_open_webui_model_modules(monkeypatch)
    targets = [
        {"id": "target-a", "name": "Target A"},
        {"id": "target-b", "name": "Target B"},
    ]

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=targets,
        valves=mod.Pipe.Valves(),
    )

    for key in ("inserted", "updated", "deleted", "get_by_id"):
        calls[key].clear()
    calls["get_all"] = 0

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=targets,
        valves=mod.Pipe.Valves(),
    )

    assert calls["get_all"] == 1
    assert calls["get_by_id"] == []
    assert calls["inserted"] == []
    assert calls["updated"] == []
    assert calls["deleted"] == []
    assert set(records) == {
        mod.build_wrapper_model_id("auto_compact", "target-a"),
        mod.build_wrapper_model_id("auto_compact", "target-b"),
    }


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_keeps_fresh_target_reads_when_hiding(monkeypatch):
    _, calls = install_fake_open_webui_model_modules(monkeypatch)
    targets = [
        {"id": "target-a", "name": "Target A"},
        {"id": "target-b", "name": "Target B"},
    ]
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=targets,
        valves=valves,
    )

    for key in ("inserted", "updated", "deleted", "get_by_id"):
        calls[key].clear()
    calls["get_all"] = 0

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=targets,
        valves=valves,
    )

    assert calls["get_all"] == 1
    assert calls["get_by_id"] == ["target-a", "target-b"]
    assert calls["inserted"] == []
    assert calls["updated"] == []
    assert calls["deleted"] == []


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_does_not_hide_target_from_stale_snapshot_when_fresh_read_fails(
    monkeypatch,
):
    records = {
        "provider-target": FakeModelForm(
            id="provider-target",
            base_model_id=None,
            name="Admin Target",
            params=FakeModelParams(temperature=0.7),
            meta=FakeModelMeta(description="admin description"),
            access_grants=[],
            is_active=True,
        )
    }
    records, calls = install_fake_open_webui_model_modules(monkeypatch, records)
    Models = sys.modules["open_webui.models.models"].Models
    original_get_model_by_id = Models.get_model_by_id

    async def fail_target_read(model_id):
        if model_id == "provider-target":
            calls["get_by_id"].append(model_id)
            return None
        return await original_get_model_by_id(model_id)

    monkeypatch.setattr(Models, "get_model_by_id", staticmethod(fail_target_read))
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "provider-target", "name": "Provider Target"}],
        valves=valves,
    )

    target = records["provider-target"]
    target_meta = target.meta.model_dump(exclude_unset=True)
    assert target.name == "Admin Target"
    assert target.params.temperature == 0.7
    assert target.meta.description == "admin description"
    assert "hidden" not in target_meta
    assert "auto_compaction_target_hidden_by" not in target_meta
    assert calls["updated"] == []


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_skips_stale_cleanup_when_bulk_read_fails(monkeypatch):
    stale_wrapper_id = mod.build_wrapper_model_id("auto_compact", "stale-target")
    records = {
        stale_wrapper_id: FakeModelForm(
            id=stale_wrapper_id,
            base_model_id=None,
            name="Stale Target (AutoCompact)",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                auto_compaction={
                    "pipe_function_id": "auto_compact",
                    "target_model_id": "stale-target",
                }
            ),
            access_grants=[],
            is_active=True,
        )
    }
    records, _ = install_fake_open_webui_model_modules(monkeypatch, records)
    Models = sys.modules["open_webui.models.models"].Models

    async def fail_bulk_read():
        raise RuntimeError("bulk read failed")

    monkeypatch.setattr(Models, "get_all_models", staticmethod(fail_bulk_read))

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "active-target", "name": "Active Target"}],
        valves=mod.Pipe.Valves(),
    )

    assert records[stale_wrapper_id].is_active is True
    assert mod.build_wrapper_model_id("auto_compact", "active-target") in records


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_hides_target_models_when_enabled(monkeypatch):
    existing_target = FakeModelForm(
        id="target",
        base_model_id="provider-target",
        name="Custom Target Name",
        params=FakeModelParams(),
        meta=FakeModelMeta(description="custom target description"),
        access_grants=[{"principal_type": "group", "principal_id": "team", "permission": "read"}],
        is_active=True,
    )
    _, calls = install_fake_open_webui_model_modules(monkeypatch, {"target": existing_target})
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[
            {
                "id": "target",
                "name": "Target",
                "info": {
                    "meta": {"description": "target description"},
                    "params": {"temperature": 0.2},
                },
            }
        ],
        valves=valves,
    )

    updated_by_id = {model_id: model_form for model_id, model_form in calls["updated"]}
    inserted_by_id = {model_form.id: model_form for model_form, _ in calls["inserted"]}
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    assert updated_by_id["target"].name == "Custom Target Name"
    assert updated_by_id["target"].base_model_id == "provider-target"
    assert updated_by_id["target"].params.model_dump(exclude_unset=True) == {}
    assert updated_by_id["target"].meta.description == "custom target description"
    assert updated_by_id["target"].meta.hidden is True
    assert updated_by_id["target"].meta.auto_compaction_target_hidden_by == {
        "pipe_function_id": "auto_compact",
        "had_hidden": False,
        "previous_hidden": False,
    }
    assert updated_by_id["target"].access_grants == [
        {"principal_type": "group", "principal_id": "team", "permission": "read"}
    ]
    wrapper_meta = inserted_by_id[wrapper_id].meta.model_dump(exclude_unset=True)
    assert "hidden" not in wrapper_meta
    assert "auto_compaction_target_hidden_by" not in wrapper_meta


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_restores_target_hidden_when_hide_valve_disabled(monkeypatch):
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    existing_target = FakeModelForm(
        id="target",
        base_model_id="provider-target",
        name="Custom Target Name",
        params=FakeModelParams(),
        meta=FakeModelMeta(
            hidden=True,
            auto_compaction_target_hidden_by={
                "pipe_function_id": "auto_compact",
                "had_hidden": False,
                "previous_hidden": False,
            },
            description="custom target description",
        ),
        access_grants=[],
        is_active=True,
    )
    existing_wrapper = FakeModelForm(
        id=wrapper_id,
        base_model_id=None,
        name="Target (AutoCompact)",
        params=FakeModelParams(),
        meta=FakeModelMeta(
            auto_compaction={
                "pipe_function_id": "auto_compact",
                "target_model_id": "target",
            },
        ),
        access_grants=[],
        is_active=True,
    )
    _, calls = install_fake_open_webui_model_modules(
        monkeypatch,
        {"target": existing_target, wrapper_id: existing_wrapper},
    )

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "target", "name": "Target"}],
        valves=mod.Pipe.Valves(),
    )

    updated_by_id = {model_id: model_form for model_id, model_form in calls["updated"]}
    target_meta = updated_by_id["target"].meta.model_dump(exclude_unset=True)
    assert "hidden" not in target_meta
    assert "auto_compaction_target_hidden_by" not in target_meta
    assert target_meta["description"] == "custom target description"


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_keeps_current_wrapper_desired_when_target_hide_fails(monkeypatch):
    active_wrapper_id = mod.build_wrapper_model_id("auto_compact", "active-target")
    records = {
        active_wrapper_id: FakeModelForm(
            id=active_wrapper_id,
            base_model_id=None,
            name="Active (AutoCompact)",
            params=FakeModelParams(temperature=0.1),
            meta=FakeModelMeta(
                auto_compaction={
                    "pipe_function_id": "auto_compact",
                    "target_model_id": "active-target",
                },
            ),
            access_grants=[],
            is_active=True,
        )
    }
    _, calls = install_fake_open_webui_model_modules(monkeypatch, records)
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    async def fail_hiding_target(**kwargs):
        raise RuntimeError("target update failed")

    monkeypatch.setattr(mod, "_hide_target_model_record", fail_hiding_target)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "active-target", "name": "Active Target", "params": {"temperature": 0.3}}],
        valves=valves,
    )

    updated_by_id = {model_id: model_form for model_id, model_form in calls["updated"]}
    assert updated_by_id[active_wrapper_id].name == "Active Target"
    assert updated_by_id[active_wrapper_id].params.stream_response is True
    assert updated_by_id[active_wrapper_id].is_active is True


@pytest.mark.asyncio
@pytest.mark.parametrize("insert_failure", ["exception", "none"])
async def test_sync_wrapper_model_records_does_not_hide_existing_target_when_wrapper_insert_fails(
    monkeypatch, insert_failure
):
    existing_target = FakeModelForm(
        id="target",
        base_model_id="provider-target",
        name="Target",
        params=FakeModelParams(),
        meta=FakeModelMeta(description="target description"),
        access_grants=[],
        is_active=True,
    )
    records, calls = install_fake_open_webui_model_modules(monkeypatch, {"target": existing_target})
    Models = sys.modules["open_webui.models.models"].Models
    original_insert_new_model = Models.insert_new_model

    async def fail_wrapper_insert(model_form, user_id):
        if model_form.id == mod.build_wrapper_model_id("auto_compact", "target"):
            if insert_failure == "none":
                return None
            raise RuntimeError("wrapper insert failed")
        return await original_insert_new_model(model_form, user_id)

    monkeypatch.setattr(Models, "insert_new_model", staticmethod(fail_wrapper_insert))
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "target", "name": "Target"}],
        valves=valves,
    )

    target_meta = records["target"].meta.model_dump(exclude_unset=True)
    assert target_meta == {"description": "target description"}
    assert calls["updated"] == []
    assert mod.build_wrapper_model_id("auto_compact", "target") not in records


@pytest.mark.asyncio
@pytest.mark.parametrize("insert_failure", ["exception", "none"])
async def test_sync_wrapper_model_records_does_not_create_target_override_when_wrapper_insert_fails(
    monkeypatch, insert_failure
):
    records, calls = install_fake_open_webui_model_modules(monkeypatch)
    Models = sys.modules["open_webui.models.models"].Models
    original_insert_new_model = Models.insert_new_model

    async def fail_wrapper_insert(model_form, user_id):
        if model_form.id == mod.build_wrapper_model_id("auto_compact", "provider-target"):
            if insert_failure == "none":
                return None
            raise RuntimeError("wrapper insert failed")
        return await original_insert_new_model(model_form, user_id)

    monkeypatch.setattr(Models, "insert_new_model", staticmethod(fail_wrapper_insert))
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "provider-target", "name": "Provider Target"}],
        valves=valves,
    )

    assert "provider-target" not in records
    assert mod.build_wrapper_model_id("auto_compact", "provider-target") not in records
    assert calls["inserted"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("update_failure", ["exception", "none"])
async def test_sync_wrapper_model_records_restores_target_when_wrapper_update_fails(monkeypatch, update_failure):
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    records = {
        "target": FakeModelForm(
            id="target",
            base_model_id="provider-target",
            name="Target",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                hidden=True,
                auto_compaction_target_hidden_by={
                    "pipe_function_id": "auto_compact",
                    "had_hidden": False,
                    "previous_hidden": False,
                },
            ),
            access_grants=[],
            is_active=True,
        ),
        wrapper_id: FakeModelForm(
            id=wrapper_id,
            base_model_id=None,
        name="Target (AutoCompact)",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                auto_compaction={
                    "pipe_function_id": "auto_compact",
                    "target_model_id": "target",
                },
            ),
            access_grants=[],
            is_active=False,
        ),
    }
    records, calls = install_fake_open_webui_model_modules(monkeypatch, records)
    Models = sys.modules["open_webui.models.models"].Models
    original_update_model_by_id = Models.update_model_by_id

    async def fail_wrapper_update(model_id, model_form):
        if model_id == wrapper_id:
            if update_failure == "none":
                return None
            raise RuntimeError("wrapper update failed")
        return await original_update_model_by_id(model_id, model_form)

    monkeypatch.setattr(Models, "update_model_by_id", staticmethod(fail_wrapper_update))
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "target", "name": "Target"}],
        valves=valves,
    )

    target_meta = records["target"].meta.model_dump(exclude_unset=True)
    assert "hidden" not in target_meta
    assert "auto_compaction_target_hidden_by" not in target_meta
    assert records[wrapper_id].is_active is False
    assert [model_id for model_id, _ in calls["updated"]] == ["target"]


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_deactivates_stale_managed_wrappers(monkeypatch):
    active_wrapper_id = mod.build_wrapper_model_id("auto_compact", "active-target")
    stale_wrapper_id = mod.build_wrapper_model_id("auto_compact", "stale-target")
    other_pipe_wrapper_id = mod.build_wrapper_model_id("other_pipe", "stale-target")
    records = {
        stale_wrapper_id: FakeModelForm(
            id=stale_wrapper_id,
            base_model_id=None,
        name="Stale Target (AutoCompact)",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                hidden=True,
                auto_compaction={
                    "pipe_function_id": "auto_compact",
                    "target_model_id": "stale-target",
                },
            ),
            access_grants=[],
            is_active=True,
        ),
        other_pipe_wrapper_id: FakeModelForm(
            id=other_pipe_wrapper_id,
            base_model_id=None,
            name="Other Pipe",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                auto_compaction={
                    "pipe_function_id": "other_pipe",
                    "target_model_id": "stale-target",
                },
            ),
            access_grants=[],
            is_active=True,
        ),
    }
    _, calls = install_fake_open_webui_model_modules(monkeypatch, records)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "active-target", "name": "Active Target"}],
        valves=mod.Pipe.Valves(),
    )

    updated_by_id = {model_id: model_form for model_id, model_form in calls["updated"]}
    assert active_wrapper_id not in updated_by_id
    assert stale_wrapper_id in updated_by_id
    assert updated_by_id[stale_wrapper_id].is_active is False
    assert updated_by_id[stale_wrapper_id].meta.hidden is True
    assert other_pipe_wrapper_id not in updated_by_id


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_restores_hidden_target_before_deactivating_stale_wrapper(monkeypatch):
    stale_wrapper_id = mod.build_wrapper_model_id("auto_compact", "stale-target")
    records = {
        "stale-target": FakeModelForm(
            id="stale-target",
            base_model_id="provider-target",
            name="Stale Target",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                hidden=True,
                auto_compaction_target_hidden_by={
                    "pipe_function_id": "auto_compact",
                    "had_hidden": False,
                    "previous_hidden": False,
                },
            ),
            access_grants=[],
            is_active=True,
        ),
        stale_wrapper_id: FakeModelForm(
            id=stale_wrapper_id,
            base_model_id=None,
        name="Stale Target (AutoCompact)",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                auto_compaction={
                    "pipe_function_id": "auto_compact",
                    "target_model_id": "stale-target",
                },
            ),
            access_grants=[],
            is_active=True,
        ),
    }
    _, calls = install_fake_open_webui_model_modules(monkeypatch, records)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[],
        valves=mod.Pipe.Valves(),
    )

    assert [model_id for model_id, _ in calls["updated"]] == ["stale-target", stale_wrapper_id]
    target_meta = calls["updated"][0][1].meta.model_dump(exclude_unset=True)
    assert "hidden" not in target_meta
    assert "auto_compaction_target_hidden_by" not in target_meta
    assert calls["updated"][1][1].is_active is False


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_retains_pipe_created_target_override_when_stale(monkeypatch):
    records, calls = install_fake_open_webui_model_modules(monkeypatch)
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[
            {
                "id": "provider-target",
                "name": "Provider Target",
                "access_grants": [
                    {
                        "principal_type": "group",
                        "principal_id": "team",
                        "permission": "read",
                    }
                ],
            }
        ],
        valves=valves,
    )

    target_override = records["provider-target"]
    assert target_override.meta.hidden is True
    assert target_override.meta.auto_compaction_target_hidden_by == {
        "pipe_function_id": "auto_compact",
        "had_hidden": False,
        "previous_hidden": False,
    }

    calls["inserted"].clear()
    calls["updated"].clear()
    calls["deleted"].clear()

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[],
        valves=mod.Pipe.Valves(),
    )

    wrapper_id = mod.build_wrapper_model_id("auto_compact", "provider-target")
    restored_meta = records["provider-target"].meta.model_dump(exclude_unset=True)
    assert calls["deleted"] == []
    assert "hidden" not in restored_meta
    assert "auto_compaction_target_hidden_by" not in restored_meta
    assert records[wrapper_id].is_active is False


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_preserves_admin_edited_target_override(monkeypatch):
    records, calls = install_fake_open_webui_model_modules(monkeypatch)
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True
    target = {"id": "provider-target", "name": "Provider Target"}

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[target],
        valves=valves,
    )

    original_marker = copy.deepcopy(
        records["provider-target"].meta.auto_compaction_target_hidden_by
    )
    records["provider-target"] = FakeModelForm(
        id="provider-target",
        base_model_id=None,
        name="Admin Edited Target",
        params=FakeModelParams(temperature=0.7),
        meta=FakeModelMeta(
            hidden=True,
            description="admin description",
            auto_compaction_target_hidden_by=original_marker,
        ),
        access_grants=[
            {
                "id": "admin-grant-id",
                "principal_type": "user",
                "principal_id": "admin-user",
                "permission": "write",
            }
        ],
        is_active=False,
    )

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[target],
        valves=valves,
    )

    assert records["provider-target"].meta.auto_compaction_target_hidden_by == original_marker
    for key in ("inserted", "updated", "deleted", "get_by_id"):
        calls[key].clear()
    calls["get_all"] = 0

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[],
        valves=mod.Pipe.Valves(),
    )

    restored = records["provider-target"]
    restored_meta = restored.meta.model_dump(exclude_unset=True)
    assert calls["deleted"] == []
    assert restored.name == "Admin Edited Target"
    assert restored.params.temperature == 0.7
    assert restored_meta["description"] == "admin description"
    assert "hidden" not in restored_meta
    assert "auto_compaction_target_hidden_by" not in restored_meta
    assert {
        (grant["principal_type"], grant["principal_id"], grant["permission"])
        for grant in restored.access_grants
    } == {("user", "admin-user", "write")}
    assert restored.is_active is False
    assert records[mod.build_wrapper_model_id("auto_compact", "provider-target")].is_active is False


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_restores_previous_created_marker_without_deleting_override(monkeypatch):
    records, calls = install_fake_open_webui_model_modules(monkeypatch)
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "provider-target", "name": "Provider Target"}],
        valves=valves,
    )

    records["provider-target"].meta.auto_compaction_target_hidden_by.update(
        {
            "created_model_record": True,
            "created_model_record_fingerprint": "sha256:previous-version",
        }
    )
    for key in ("inserted", "updated", "deleted", "get_by_id"):
        calls[key].clear()
    calls["get_all"] = 0

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[],
        valves=mod.Pipe.Valves(),
    )

    restored_meta = records["provider-target"].meta.model_dump(exclude_unset=True)
    assert calls["deleted"] == []
    assert "hidden" not in restored_meta
    assert "auto_compaction_target_hidden_by" not in restored_meta
    assert records[mod.build_wrapper_model_id("auto_compact", "provider-target")].is_active is False


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_retains_pipe_created_target_override_when_hide_valve_disabled(
    monkeypatch,
):
    records, calls = install_fake_open_webui_model_modules(monkeypatch)
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "provider-target", "name": "Provider Target"}],
        valves=valves,
    )

    calls["inserted"].clear()
    calls["updated"].clear()
    calls["deleted"].clear()

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "provider-target", "name": "Provider Target"}],
        valves=mod.Pipe.Valves(),
    )

    wrapper_id = mod.build_wrapper_model_id("auto_compact", "provider-target")
    restored_meta = records["provider-target"].meta.model_dump(exclude_unset=True)
    assert calls["deleted"] == []
    assert "hidden" not in restored_meta
    assert "auto_compaction_target_hidden_by" not in restored_meta
    assert records[wrapper_id].is_active is True


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_keeps_preexisting_hidden_target_when_restoring(monkeypatch):
    stale_wrapper_id = mod.build_wrapper_model_id("auto_compact", "stale-target")
    records = {
        "stale-target": FakeModelForm(
            id="stale-target",
            base_model_id="provider-target",
            name="Stale Target",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                hidden=True,
                auto_compaction_target_hidden_by={
                    "pipe_function_id": "auto_compact",
                    "had_hidden": True,
                    "previous_hidden": True,
                },
            ),
            access_grants=[],
            is_active=True,
        ),
        stale_wrapper_id: FakeModelForm(
            id=stale_wrapper_id,
            base_model_id=None,
        name="Stale Target (AutoCompact)",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                auto_compaction={
                    "pipe_function_id": "auto_compact",
                    "target_model_id": "stale-target",
                },
            ),
            access_grants=[],
            is_active=True,
        ),
    }
    _, calls = install_fake_open_webui_model_modules(monkeypatch, records)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[],
        valves=mod.Pipe.Valves(),
    )

    target_meta = calls["updated"][0][1].meta.model_dump(exclude_unset=True)
    assert target_meta["hidden"] is True
    assert "auto_compaction_target_hidden_by" not in target_meta


@pytest.mark.asyncio
@pytest.mark.parametrize("restore_failure", ["exception", "none", "read_none"])
async def test_sync_wrapper_model_records_keeps_stale_wrapper_active_when_target_restore_fails(
    monkeypatch, restore_failure
):
    stale_wrapper_id = mod.build_wrapper_model_id("auto_compact", "stale-target")
    records = {
        "stale-target": FakeModelForm(
            id="stale-target",
            base_model_id="provider-target",
            name="Stale Target",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                hidden=True,
                auto_compaction_target_hidden_by={
                    "pipe_function_id": "auto_compact",
                    "had_hidden": False,
                    "previous_hidden": False,
                },
            ),
            access_grants=[],
            is_active=True,
        ),
        stale_wrapper_id: FakeModelForm(
            id=stale_wrapper_id,
            base_model_id=None,
            name="Stale Target (AutoCompact)",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                auto_compaction={
                    "pipe_function_id": "auto_compact",
                    "target_model_id": "stale-target",
                },
            ),
            access_grants=[],
            is_active=True,
        ),
    }
    records, calls = install_fake_open_webui_model_modules(monkeypatch, records)
    Models = sys.modules["open_webui.models.models"].Models
    original_get_model_by_id = Models.get_model_by_id
    original_update_model_by_id = Models.update_model_by_id

    async def fail_target_restore_read(model_id):
        if restore_failure == "read_none" and model_id == "stale-target":
            calls["get_by_id"].append(model_id)
            return None
        return await original_get_model_by_id(model_id)

    async def fail_target_restore_update(model_id, model_form):
        if restore_failure != "read_none" and model_id == "stale-target":
            if restore_failure == "none":
                return None
            raise RuntimeError("restore failed")
        return await original_update_model_by_id(model_id, model_form)

    monkeypatch.setattr(Models, "get_model_by_id", staticmethod(fail_target_restore_read))
    monkeypatch.setattr(Models, "update_model_by_id", staticmethod(fail_target_restore_update))

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[],
        valves=mod.Pipe.Valves(),
    )

    assert calls["updated"] == []
    assert records["stale-target"].meta.hidden is True
    assert records[stale_wrapper_id].is_active is True


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_does_not_claim_target_hidden_by_other_pipe(monkeypatch):
    records = {
        "target": FakeModelForm(
            id="target",
            base_model_id="provider-target",
            name="Target",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                hidden=True,
                auto_compaction_target_hidden_by={
                    "pipe_function_id": "compact_a",
                    "had_hidden": False,
                    "previous_hidden": False,
                },
            ),
            access_grants=[],
            is_active=True,
        )
    }
    records, _ = install_fake_open_webui_model_modules(monkeypatch, records)
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="compact_b",
        target_models=[{"id": "target", "name": "Target"}],
        valves=valves,
    )

    target_meta = records["target"].meta.model_dump(exclude_unset=True)
    assert target_meta["hidden"] is True
    assert target_meta["auto_compaction_target_hidden_by"] == {
        "pipe_function_id": "compact_a",
        "had_hidden": False,
        "previous_hidden": False,
    }
    assert records[mod.build_wrapper_model_id("compact_b", "target")].is_active is True

    await mod.sync_wrapper_model_records(
        pipe_function_id="compact_b",
        target_models=[],
        valves=mod.Pipe.Valves(),
    )

    target_meta = records["target"].meta.model_dump(exclude_unset=True)
    assert target_meta["hidden"] is True
    assert target_meta["auto_compaction_target_hidden_by"] == {
        "pipe_function_id": "compact_a",
        "had_hidden": False,
        "previous_hidden": False,
    }
    assert records[mod.build_wrapper_model_id("compact_b", "target")].is_active is False


@pytest.mark.asyncio
async def test_hide_target_model_record_preserves_foreign_marker_with_stale_snapshot(monkeypatch):
    records = {
        "target": FakeModelForm(
            id="target",
            base_model_id="provider-target",
            name="Target",
            params=FakeModelParams(),
            meta=FakeModelMeta(
                hidden=True,
                auto_compaction_target_hidden_by={
                    "pipe_function_id": "compact_a",
                    "had_hidden": False,
                    "previous_hidden": False,
                },
            ),
            access_grants=[],
            is_active=True,
        )
    }
    install_fake_open_webui_model_modules(monkeypatch, records)
    Models = sys.modules["open_webui.models.models"].Models

    stale_snapshot_b = FakeModelForm(
        id="target",
        base_model_id="provider-target",
        name="Target",
        params=FakeModelParams(),
        meta=FakeModelMeta(),
        access_grants=[],
        is_active=True,
    )

    await mod._hide_target_model_record(
        Models=Models,
        ModelForm=FakeModelForm,
        ModelMeta=FakeModelMeta,
        ModelParams=FakeModelParams,
        pipe_function_id="compact_b",
        target_model={"id": "target", "name": "Target"},
        target_model_info=stale_snapshot_b,
        owner_user_id="function-owner",
    )

    target_meta = records["target"].meta.model_dump(exclude_unset=True)
    assert target_meta["hidden"] is True
    assert target_meta["auto_compaction_target_hidden_by"] == {
        "pipe_function_id": "compact_a",
        "had_hidden": False,
        "previous_hidden": False,
    }


@pytest.mark.asyncio
async def test_pipes_uses_runtime_registered_id_for_filtering_and_sync(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["pipe_function_id"] = kwargs["pipe_function_id"]
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={
                "target": {"id": "target", "name": "Target"},
                mod.build_wrapper_model_id("compact_alias", "target"): {
                    "id": mod.build_wrapper_model_id("compact_alias", "target"),
                    "name": "Own Wrapper",
                    "pipe": {"type": "pipe"},
                },
                mod.build_wrapper_model_id("auto_compact", "target"): {
                    "id": mod.build_wrapper_model_id("auto_compact", "target"),
                    "name": "Other Registered Wrapper",
                    "pipe": {"type": "pipe"},
                },
            }
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.Pipe, "__module__", "function_compact_alias")

    pipe = mod.Pipe()

    result = await pipe.pipes()

    assert captured == {
        "pipe_function_id": "compact_alias",
        "target_ids": ["target", mod.build_wrapper_model_id("auto_compact", "target")],
    }
    assert result == [
        {"id": "target", "name": "Target (AutoCompact)"},
        {
            "id": mod.build_wrapper_model_id("auto_compact", "target"),
            "name": "Other Registered Wrapper (AutoCompact)",
        },
    ]


@pytest.mark.asyncio
async def test_pipes_returns_entries_when_wrapper_record_sync_fails(monkeypatch):
    async def sync_wrapper_model_records(**kwargs):
        raise RuntimeError("db unavailable")

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={"target": {"id": "target", "name": "Target"}},
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)

    result = await mod.Pipe().pipes()

    assert result == [{"id": "target", "name": "Target (AutoCompact)"}]


@pytest.mark.asyncio
async def test_pipes_does_not_direct_fetch_provider_models_when_state_caches_remain_empty(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["pipe_function_id"] = kwargs["pipe_function_id"]
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def fetch_provider_models(*args, **kwargs):
        return [{"id": "direct-target", "name": "Direct Target"}]

    models_module = types.ModuleType("open_webui.utils.models")
    models_module.fetch_openai_models = fetch_provider_models
    models_module.fetch_ollama_models = fetch_provider_models

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={},
            BASE_MODELS=[],
            OPENAI_MODELS={},
            OLLAMA_MODELS={},
            config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", models_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod, "_provider_model_cache_wait_timeout_seconds", lambda: 0)

    result = await mod.Pipe().pipes()

    assert captured == {
        "pipe_function_id": "auto_compact",
        "target_ids": [],
    }
    assert result == []


@pytest.mark.asyncio
async def test_pipes_waits_for_sibling_provider_cache_population(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod, "_provider_model_cache_wait_timeout_seconds", lambda: 0.1)
    monkeypatch.setattr(mod, "PROVIDER_MODEL_CACHE_WAIT_POLL_SECONDS", 0.001)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    async def populate_cache():
        await asyncio.sleep(0)
        state.OPENAI_MODELS = {"sibling-target": {"id": "sibling-target", "name": "Sibling Target"}}

    task = asyncio.create_task(populate_cache())
    try:
        result = await mod.Pipe().pipes()
    finally:
        await task

    assert captured["target_ids"] == ["sibling-target"]
    assert result == [{"id": "sibling-target", "name": "Sibling Target (AutoCompact)"}]


@pytest.mark.asyncio
async def test_pipes_waits_for_all_enabled_provider_cache_population(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            state.OPENAI_MODELS = {"openai-target": {"id": "openai-target", "name": "OpenAI Target"}}
        elif sleep_calls == 2:
            state.OLLAMA_MODELS = {"ollama-target": {"model": "ollama-target", "name": "Ollama Target"}}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=True),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert sleep_calls == 2
    assert captured["target_ids"] == ["openai-target", "ollama-target"]
    assert result == [
        {"id": "openai-target", "name": "OpenAI Target (AutoCompact)"},
        {"id": "ollama-target", "name": "Ollama Target (AutoCompact)"},
    ]


@pytest.mark.asyncio
async def test_pipes_waits_when_one_of_multiple_provider_caches_is_still_empty(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        state.OLLAMA_MODELS = {"ollama-target": {"model": "ollama-target", "name": "Ollama Target"}}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={"openai-target": {"id": "openai-target", "name": "OpenAI Target"}},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=True),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert sleep_calls == 1
    assert captured["target_ids"] == ["openai-target", "ollama-target"]
    assert result == [
        {"id": "openai-target", "name": "OpenAI Target (AutoCompact)"},
        {"id": "ollama-target", "name": "Ollama Target (AutoCompact)"},
    ]


@pytest.mark.asyncio
async def test_pipes_does_not_wait_for_disabled_provider_cache(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        raise AssertionError("pipes() should not wait for disabled provider caches")

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={},
            BASE_MODELS=[],
            OPENAI_MODELS={"openai-target": {"id": "openai-target", "name": "OpenAI Target"}},
            OLLAMA_MODELS={},
            config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)

    result = await mod.Pipe().pipes()

    assert captured["target_ids"] == ["openai-target"]
    assert result == [{"id": "openai-target", "name": "OpenAI Target (AutoCompact)"}]


@pytest.mark.asyncio
async def test_pipes_excludes_disabled_provider_stale_cache_models(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    state = SimpleNamespace(
        MODELS={
            "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "owned_by": "openai", "openai": {}},
            "other-pipe.child": {
                "id": "other-pipe.child",
                "name": "Other Pipe",
                "owned_by": "openai",
                "pipe": {"type": "pipe"},
            },
            "workspace-preset": {
                "id": "workspace-preset",
                "name": "Workspace Preset",
                "owned_by": "openai",
                "preset": True,
                "info": {"base_model_id": "stale-openai"},
            },
        },
        BASE_MODELS=[
            {"id": "stale-ollama", "name": "Stale Ollama", "owned_by": "ollama", "ollama": {}},
        ],
        OPENAI_MODELS={"direct-openai": {"id": "direct-openai", "name": "Direct OpenAI", "openai": {}}},
        OLLAMA_MODELS={"direct-ollama": {"model": "direct-ollama", "name": "Direct Ollama"}},
        config=SimpleNamespace(ENABLE_OPENAI_API=False, ENABLE_OLLAMA_API=False),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)

    result = await mod.Pipe().pipes()

    assert captured["target_ids"] == ["other-pipe.child", "workspace-preset"]
    assert result == [
        {"id": "other-pipe.child", "name": "Other Pipe (AutoCompact)"},
        {"id": "workspace-preset", "name": "Workspace Preset (AutoCompact)"},
    ]


@pytest.mark.asyncio
async def test_pipes_uses_config_provider_enable_flags_when_legacy_attrs_are_missing(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            captured["config_keys"] = keys
            return {"openai.enable": False, "ollama.enable": True}

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    state = SimpleNamespace(
        MODELS={
            "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "owned_by": "openai", "openai": {}},
            "stale-ollama": {"id": "stale-ollama", "name": "Stale Ollama", "owned_by": "ollama", "ollama": {}},
        },
        BASE_MODELS=[],
        OPENAI_MODELS={"direct-openai": {"id": "direct-openai", "name": "Direct OpenAI", "openai": {}}},
        OLLAMA_MODELS={"direct-ollama": {"model": "direct-ollama", "name": "Direct Ollama"}},
        config=SimpleNamespace(),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)

    result = await mod.Pipe().pipes()

    assert captured["config_keys"] == ("openai.enable", "ollama.enable")
    assert captured["target_ids"] == ["stale-ollama", "direct-ollama"]
    assert result == [
        {"id": "stale-ollama", "name": "Stale Ollama (AutoCompact)"},
        {"id": "direct-ollama", "name": "Direct Ollama (AutoCompact)"},
    ]


@pytest.mark.asyncio
async def test_pipes_falls_back_to_legacy_provider_enable_flags_when_config_errors(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            captured["config_keys"] = keys
            raise RuntimeError("config unavailable")

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    state = SimpleNamespace(
        MODELS={
            "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "owned_by": "openai", "openai": {}},
            "stale-ollama": {"id": "stale-ollama", "name": "Stale Ollama", "owned_by": "ollama", "ollama": {}},
        },
        BASE_MODELS=[],
        OPENAI_MODELS={"direct-openai": {"id": "direct-openai", "name": "Direct OpenAI", "openai": {}}},
        OLLAMA_MODELS={"direct-ollama": {"model": "direct-ollama", "name": "Direct Ollama"}},
        config=SimpleNamespace(ENABLE_OPENAI_API=False, ENABLE_OLLAMA_API=True),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)

    result = await mod.Pipe().pipes()

    assert captured["config_keys"] == ("openai.enable", "ollama.enable")
    assert captured["target_ids"] == ["stale-ollama", "direct-ollama"]
    assert result == [
        {"id": "stale-ollama", "name": "Stale Ollama (AutoCompact)"},
        {"id": "direct-ollama", "name": "Direct Ollama (AutoCompact)"},
    ]


@pytest.mark.asyncio
async def test_model_dict_from_request_excludes_config_disabled_stale_provider_caches(
    monkeypatch, pipe_request
):
    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            return {"openai.enable": False, "ollama.enable": True}

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    pipe_request.app.state.config = SimpleNamespace()
    pipe_request.app.state.MODELS = {
        "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "owned_by": "openai", "openai": {}},
        "stale-ollama": {"id": "stale-ollama", "name": "Stale Ollama", "owned_by": "ollama", "ollama": {}},
        "direct": {"id": "direct", "name": "Direct"},
    }
    pipe_request.app.state.BASE_MODELS = []
    pipe_request.app.state.OPENAI_MODELS = {
        "direct-openai": {"id": "direct-openai", "name": "Direct OpenAI", "openai": {}}
    }
    pipe_request.app.state.OLLAMA_MODELS = {
        "direct-ollama": {"model": "direct-ollama", "name": "Direct Ollama"}
    }

    models = await mod._model_dict_from_request(pipe_request)

    assert set(models) == {"direct", "stale-ollama", "direct-ollama"}


@pytest.mark.asyncio
async def test_ensure_model_in_request_models_does_not_inject_config_disabled_provider_model(
    monkeypatch, pipe_request
):
    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            return {"openai.enable": False, "ollama.enable": True}

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    pipe_request.app.state.config = SimpleNamespace()
    pipe_request.app.state.MODELS = {}
    pipe_request.app.state.BASE_MODELS = []
    pipe_request.app.state.OPENAI_MODELS = {
        "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "openai": {}}
    }
    pipe_request.app.state.OLLAMA_MODELS = {}

    model = await mod._ensure_model_in_request_models(pipe_request, "stale-openai")

    assert model is None
    assert pipe_request.app.state.MODELS == {}


@pytest.mark.asyncio
async def test_pipes_does_not_wait_when_empty_provider_cache_already_refreshed(monkeypatch):
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        raise AssertionError("already-refreshed empty provider cache should not wait")

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={},
            BASE_MODELS=[],
            OPENAI_MODELS={},
            OLLAMA_MODELS={},
            config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)

    result = await mod.Pipe().pipes()

    assert captured["target_ids"] == []
    assert result == []


@pytest.mark.asyncio
async def test_provider_cache_pending_detection_requires_core_request_identity():
    def build_async_function(name: str, filename: str, body: str, **namespace):
        scope = dict(namespace)
        exec(compile(f"async def {name}(request):\n{body}", filename, "exec"), scope)
        return scope[name]

    event = asyncio.Event()
    fetch_openai_models = build_async_function(
        "fetch_openai_models",
        "/site-packages/open_webui/utils/models.py",
        "    await event.wait()\n",
        event=event,
    )
    get_function_models = build_async_function(
        "get_function_models",
        "/site-packages/open_webui/functions.py",
        "    return mod._provider_model_cache_refresh_pending_attrs(['OPENAI_MODELS'])\n",
        mod=mod,
    )
    request = object()
    other_request = object()

    task = asyncio.create_task(fetch_openai_models(other_request))
    await asyncio.sleep(0)
    try:
        assert mod._provider_model_cache_refresh_pending_attrs(["OPENAI_MODELS"]) == set()
        function_task = asyncio.create_task(get_function_models(request))
        assert await function_task == set()
    finally:
        event.set()
        await task

    event = asyncio.Event()
    fetch_openai_models = build_async_function(
        "fetch_openai_models",
        "/site-packages/open_webui/utils/models.py",
        "    await event.wait()\n",
        event=event,
    )
    task = asyncio.create_task(fetch_openai_models(request))
    await asyncio.sleep(0)
    try:
        function_task = asyncio.create_task(get_function_models(request))
        assert await function_task == {"OPENAI_MODELS"}
    finally:
        event.set()
        await task


@pytest.mark.asyncio
async def test_pipes_does_not_wait_for_other_request_provider_refresh(monkeypatch):
    captured = {}

    def build_async_function(name: str, filename: str, body: str, **namespace):
        scope = dict(namespace)
        exec(compile(f"async def {name}(request):\n{body}", filename, "exec"), scope)
        return scope[name]

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        raise AssertionError("pipes() should not wait for another request's provider refresh")

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)

    event = asyncio.Event()
    fetch_openai_models = build_async_function(
        "fetch_openai_models",
        "/site-packages/open_webui/utils/models.py",
        "    await event.wait()\n",
        event=event,
    )
    get_function_models = build_async_function(
        "get_function_models",
        "/site-packages/open_webui/functions.py",
        "    return await pipe.pipes()\n",
        pipe=mod.Pipe(),
    )

    other_task = asyncio.create_task(fetch_openai_models(object()))
    real_sleep = asyncio.sleep
    await real_sleep(0)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    try:
        result = await get_function_models(object())
    finally:
        event.set()
        await other_task

    assert captured["target_ids"] == []
    assert result == []


@pytest.mark.asyncio
async def test_pipes_waits_for_ollama_two_stage_provider_refresh(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 25:
            state.OLLAMA_MODELS = {"ollama-target": {"model": "ollama-target", "name": "Ollama Target"}}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=False, ENABLE_OLLAMA_API=True),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod, "_provider_model_cache_wait_timeout_seconds", lambda: 1.0)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert sleep_calls == 25
    assert captured["target_ids"] == ["ollama-target"]
    assert result == [{"id": "ollama-target", "name": "Ollama Target (AutoCompact)"}]


@pytest.mark.asyncio
async def test_pipes_normalizes_ollama_provider_cache_models(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_models"] = kwargs["target_models"]

    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(
        state=SimpleNamespace(
            MODELS={},
            BASE_MODELS=[],
            OPENAI_MODELS={},
            OLLAMA_MODELS={"llama3.2:latest": {"model": "llama3.2:latest", "name": "Llama 3.2", "tags": ["local"]}},
            config=SimpleNamespace(ENABLE_OPENAI_API=False, ENABLE_OLLAMA_API=True),
        )
    )
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)

    result = await mod.Pipe().pipes()

    assert captured["target_models"] == [
        {
            "id": "llama3.2:latest",
            "name": "Llama 3.2",
            "object": "model",
            "created": 0,
            "owned_by": "ollama",
            "ollama": {"model": "llama3.2:latest", "name": "Llama 3.2", "tags": ["local"]},
            "loaded": False,
            "connection_type": "local",
            "tags": ["local"],
        }
    ]
    assert result == [{"id": "llama3.2:latest", "name": "Llama 3.2 (AutoCompact)"}]


@pytest.mark.asyncio
async def test_pipes_waits_until_core_model_list_timeout_for_sibling_provider_cache(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 25:
            state.OPENAI_MODELS = {"slow-target": {"id": "slow-target", "name": "Slow Target"}}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod, "_provider_model_cache_wait_timeout_seconds", lambda: 2.0)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert sleep_calls == 25
    assert captured["target_ids"] == ["slow-target"]
    assert result == [{"id": "slow-target", "name": "Slow Target (AutoCompact)"}]


@pytest.mark.asyncio
async def test_pipes_keeps_current_cache_when_provider_cache_wait_fails(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def wait_for_provider_model_caches(*args, **kwargs):
        raise RuntimeError("provider wait failed")

    state = SimpleNamespace(
        MODELS={"cached-target": {"id": "cached-target", "name": "Cached Target"}},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
    )
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))
    monkeypatch.setattr(mod, "_wait_for_provider_model_caches", wait_for_provider_model_caches)

    result = await mod.Pipe().pipes()

    assert captured["target_ids"] == ["cached-target"]
    assert result == [{"id": "cached-target", "name": "Cached Target (AutoCompact)"}]


@pytest.mark.asyncio
async def test_pipes_stops_waiting_when_sibling_provider_cache_refresh_completes_empty(monkeypatch):
    install_unavailable_open_webui_config(monkeypatch)
    captured = {}
    sleep_calls = 0

    async def sync_wrapper_model_records(**kwargs):
        captured["target_ids"] = [model["id"] for model in kwargs["target_models"]]

    async def sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        state.OPENAI_MODELS = {}

    state = SimpleNamespace(
        MODELS={},
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
        config=SimpleNamespace(ENABLE_OPENAI_API=True, ENABLE_OLLAMA_API=False),
    )
    initial_cache = state.OPENAI_MODELS
    main_module = types.ModuleType("open_webui.main")
    main_module.app = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "open_webui.main", main_module)
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)
    monkeypatch.setattr(mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(mod, "_provider_model_cache_refresh_pending_attrs", lambda attrs: set(attrs))

    result = await mod.Pipe().pipes()

    assert state.OPENAI_MODELS is not initial_cache
    assert sleep_calls == 1
    assert captured["target_ids"] == []
    assert result == []


def test_compaction_summary_embed_html_contains_full_escaped_summary():
    summary = "line 1\n" + ("long summary " * 200) + "<script>alert(1)</script>"

    embed_html = mod.render_compaction_summary_embed_html(summary)

    assert "<details" in embed_html
    assert "Compact summary" in embed_html
    assert "Full summary" not in embed_html
    assert "line 1" in embed_html
    assert "long summary " * 20 in embed_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in embed_html
    assert "<script>alert(1)</script>" not in embed_html


def test_compaction_summary_embed_html_renders_sanitized_markdown_summary():
    summary = (
        "# Heading\n\n"
        "- first item\n"
        "- second item\n\n"
        "**strong**\n\n"
        "`code <value>`\n\n"
        "[safe](https://example.com/path?q=1)\n\n"
        'raw <b onclick="alert(1)">bold</b>\n\n'
        'raw <a href="javascript:alert(2)">link</a>\n\n'
        "<script>alert(1)</script>\n\n"
        "[unsafe](javascript:alert(1))"
    )

    embed_html = mod.render_compaction_summary_embed_html(summary)

    assert "<h1>Heading</h1>" in embed_html
    assert "<li>first item</li>" in embed_html
    assert "<strong>strong</strong>" in embed_html
    assert "<code>code &lt;value&gt;</code>" in embed_html
    assert '<a href="https://example.com/path?q=1" target="_blank" rel="noopener noreferrer">safe</a>' in embed_html
    assert "&lt;b onclick=&quot;alert(1)&quot;&gt;bold&lt;/b&gt;" in embed_html
    assert "&lt;a href=&quot;javascript:alert(2)&quot;&gt;link&lt;/a&gt;" in embed_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in embed_html
    assert 'href="javascript:alert' not in embed_html
    assert "<script>alert(1)</script>" not in embed_html


def test_compaction_summary_embed_html_bounds_expanded_summary_height():
    summary = "\n".join(f"summary line {i}" for i in range(200))

    embed_html = mod.render_compaction_summary_embed_html(summary)

    assert "<details" in embed_html
    assert "Compact summary" in embed_html
    assert "max-height:" in embed_html
    assert "overflow:auto" in embed_html

    # Core's FullHeightIframe only resizes from iframe:height postMessage events.
    assert "type:'iframe:height'" in embed_html

    # If IFRAME_CSP blocks inline script, native details[open] must still reveal the body.
    assert "[open]" in embed_html
    assert ".body-wrap{height:auto;opacity:1}" in embed_html
    assert "data-js" in embed_html

    # Regression guards for the two known bad sizing approaches.
    assert "inner.scrollHeight" not in embed_html
    assert "Math.max(document.documentElement.scrollHeight,document.body.scrollHeight,1)" not in embed_html


def test_extract_compaction_summary_text_decodes_cdata_boundaries():
    summary = "before ]]> after </checkpoint_summary> literal"
    rendered = mod.render_summary_message(summary)

    assert mod.extract_compaction_summary_text_from_messages([rendered]) == summary


def test_extract_compaction_summary_text_ignores_non_generated_tags():
    rendered = mod.render_summary_message("generated summary")
    messages = [
        {
            "role": "system",
            "content": "<checkpoint_summary>system prompt fragment</checkpoint_summary>",
        },
        rendered,
        {
            "role": "user",
            "content": "<checkpoint_summary>ordinary user text</checkpoint_summary>",
        },
    ]

    assert mod.extract_compaction_summary_text_from_messages(messages) == "generated summary"


def test_message_token_estimates_are_cached_by_canonical_hash():
    cache = getattr(mod, "_MESSAGE_TOKEN_ESTIMATE_CACHE", None)
    if cache is not None:
        cache.clear()

    class CountingEncoder:
        def __init__(self):
            self.calls = []

        def encode(self, text):
            self.calls.append(text)
            return [1, 2, 3]

    encoder = CountingEncoder()
    message = {"role": "user", "content": "unchanged", "transient": "ignored"}
    messages = [message, copy.deepcopy(message)]

    first = mod.estimate_messages_tokens(messages, encoder=encoder, encoding_name="unit-test")
    second = mod.estimate_messages_tokens(messages, encoder=encoder, encoding_name="unit-test")

    assert first == second
    assert len(encoder.calls) == 1


def test_message_token_estimate_strips_image_url_data_url_payload():
    """image_url data URLs must not be tokenized as prose text."""

    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    huge_base64 = "A" * 100_000
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{huge_base64}"},
            },
        ],
    }

    count = mod.estimate_message_tokens(
        message, encoder=LengthEncoder(), encoding_name="unit-test"
    )

    assert isinstance(count, int)
    # Without stripping the 100k-char base64 would dominate the estimate; the
    # bounded estimate must stay far below the payload size.
    assert count < len(huge_base64) // 10
    # A single fixed image overhead is added on top of the surviving text.
    assert count >= len("describe this") + mod.MESSAGE_TOKEN_IMAGE_OVERHEAD


def test_message_token_estimate_strips_image_file_attachment_data_url():
    """Message-level image file attachments with data URLs must not be tokenized."""

    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    huge_base64 = "B" * 100_000
    message = {
        "role": "user",
        "content": "look at this image",
        "files": [
            {
                "type": "image",
                "id": "img-1",
                "name": "photo.png",
                "url": f"data:image/png;base64,{huge_base64}",
                "file": {"id": "img-1", "hash": "abc", "data": {"content": huge_base64}},
            }
        ],
    }

    count = mod.estimate_message_tokens(
        message, encoder=LengthEncoder(), encoding_name="unit-test"
    )

    assert isinstance(count, int)
    assert count < len(huge_base64) // 10
    assert count >= len("look at this image") + mod.MESSAGE_TOKEN_IMAGE_OVERHEAD


def test_message_token_estimate_strips_image_content_part_payload():
    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    huge_base64 = "I" * 100_000
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {
                "type": "image",
                "source": {"media_type": "image/png", "data": huge_base64},
            },
        ],
    }

    count = mod.estimate_message_tokens(
        message, encoder=LengthEncoder(), encoding_name="unit-test"
    )

    assert isinstance(count, int)
    assert count < len(huge_base64) // 10
    assert count >= len("describe this") + mod.MESSAGE_TOKEN_IMAGE_OVERHEAD


def test_message_token_estimate_counts_input_image_content_part_as_image():
    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    huge_base64 = "R" * 100_000
    message = {
        "role": "tool",
        "tool_call_id": "call-image",
        "content": [
            {"type": "input_text", "text": "generated image"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{huge_base64}"},
        ],
    }

    count = mod.estimate_message_tokens(
        message, encoder=LengthEncoder(), encoding_name="unit-test"
    )

    assert isinstance(count, int)
    assert count < len(huge_base64) // 10
    assert count >= len("generated image") + mod.MESSAGE_TOKEN_IMAGE_OVERHEAD


def test_message_token_estimate_strips_file_content_part_raw_payloads():
    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    huge_base64 = "F" * 100_000
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "summarize attached files"},
            {
                "type": "input_file",
                "filename": "report.pdf",
                "file_data": huge_base64,
                "file": {"id": "file-1", "data": {"content": huge_base64}},
            },
            {
                "type": "file",
                "name": "notes.txt",
                "content": huge_base64,
                "file": {"id": "file-2", "data": {"content": huge_base64}},
            },
        ],
    }

    count = mod.estimate_message_tokens(
        message, encoder=LengthEncoder(), encoding_name="unit-test"
    )

    assert isinstance(count, int)
    assert count < len(huge_base64) // 10


def test_message_token_estimate_strips_input_audio_raw_payloads():
    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    huge_base64 = "A" * 100_000
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "transcribe this"},
            {
                "type": "input_audio",
                "input_audio": {"format": "mp3", "data": huge_base64},
                "content": huge_base64,
            },
        ],
    }

    count = mod.estimate_message_tokens(
        message, encoder=LengthEncoder(), encoding_name="unit-test"
    )

    assert isinstance(count, int)
    assert count < len(huge_base64) // 10


def test_message_token_estimate_strips_non_image_file_attachment_raw_bodies():
    class CaptureEncoder:
        def __init__(self):
            self.text = ""

        def encode(self, text, **kwargs):
            self.text = text
            return [0] * len(text)

    huge_body = "D" * 100_000
    metadata_sha = "metadata-only-sha"
    message = {
        "role": "user",
        "content": "use the attached document",
        "files": [
            {
                "type": "file",
                "id": "doc-1",
                "name": "brief.txt",
                "content_type": "text/plain",
                "hash": "hash-1",
                "content": huge_body,
                "context": huge_body,
                "docs": [huge_body],
                "document": huge_body,
                "documents": [huge_body],
                "file": {
                    "id": "doc-1",
                    "hash": "hash-1",
                    "data": {"content": huge_body, "metadata": {"sha256": metadata_sha}},
                },
            }
        ],
    }

    encoder = CaptureEncoder()
    count = mod.estimate_message_tokens(message, encoder=encoder, encoding_name="unit-test")

    assert isinstance(count, int)
    assert count < len(huge_body) // 10
    assert huge_body not in encoder.text
    assert "hash-1" in encoder.text
    assert metadata_sha in encoder.text


def test_message_token_estimate_keeps_text_part_text_payload():
    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    huge_text = "T" * 100_000
    message = {"role": "user", "content": [{"type": "text", "text": huge_text}]}

    count = mod.estimate_message_tokens(
        message, encoder=LengthEncoder(), encoding_name="unit-test"
    )

    assert isinstance(count, int)
    assert count > len(huge_text) // 2


def test_message_token_cache_key_and_source_hash_reflect_raw_file_part_payload():
    base = {
        "role": "user",
        "content": [
            {"type": "text", "text": "summarize"},
            {
                "type": "input_file",
                "filename": "report.pdf",
                "content": "payload-a",
                "file": {"id": "file-1", "data": {"content": "payload-a"}},
            },
        ],
    }
    different = copy.deepcopy(base)
    different["content"][1]["content"] = "payload-b"
    different["content"][1]["file"]["data"]["content"] = "payload-b"

    assert mod._message_token_cache_key(base, encoding_name="enc") != mod._message_token_cache_key(
        different, encoding_name="enc"
    )
    assert mod.compute_source_hash([base]) != mod.compute_source_hash([different])


def test_message_token_estimate_adds_fixed_overhead_per_image():
    """Each additional image part adds exactly MESSAGE_TOKEN_IMAGE_OVERHEAD."""

    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    def build(image_count):
        parts = [{"type": "text", "text": "caption"}]
        parts.extend(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,XYZ"},
            }
            for _ in range(image_count)
        )
        return {"role": "user", "content": parts}

    single = mod.estimate_message_tokens(
        build(1), encoder=LengthEncoder(), encoding_name="unit-test"
    )
    triple = mod.estimate_message_tokens(
        build(3), encoder=LengthEncoder(), encoding_name="unit-test"
    )

    # The surviving text is identical once image parts are dropped, so the only
    # difference between one and three images is the fixed per-image overhead.
    assert triple - single == 2 * mod.MESSAGE_TOKEN_IMAGE_OVERHEAD


def test_message_token_image_overhead_matches_core_context_compaction():
    from open_webui.utils.context_compaction import _estimate_messages_tokens

    text_only = [{"role": "user", "content": []}]
    with_image = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,XYZ"},
                }
            ],
        }
    ]

    core_image_overhead = _estimate_messages_tokens(with_image) - _estimate_messages_tokens(
        text_only
    )

    assert mod.MESSAGE_TOKEN_IMAGE_OVERHEAD == core_image_overhead


def test_message_token_cache_key_reflects_full_image_payload():
    """Cache identity must stay based on the full canonical message (payload included)."""
    base = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aaa"}},
        ],
    }
    different = copy.deepcopy(base)
    different["content"][1]["image_url"]["url"] = "data:image/png;base64,bbb"

    assert mod._message_token_cache_key(base, encoding_name="enc") != mod._message_token_cache_key(
        different, encoding_name="enc"
    )


def test_large_message_token_estimate_uses_sampling_for_large_text():
    """Large messages (>64 KB) use 3-point sampling instead of full encode."""

    call_log = []

    class SamplingEncoder:
        def encode(self, text, **kwargs):
            call_log.append(len(text))
            # 1 char = 1 token
            return list(range(len(text)))

    large_content = "a" * (mod.MESSAGE_TOKEN_EXACT_ENCODE_MAX_BYTES + 10000)
    message = {"role": "user", "content": large_content}

    count = mod.estimate_message_tokens(
        message, encoder=SamplingEncoder(), encoding_name="unit-test"
    )

    assert isinstance(count, int)
    # Encoder called exactly 3 times (head, middle, tail)
    assert len(call_log) == 3
    # Each sample within byte budget (allow small rounding slack)
    for sample_size in call_log:
        assert sample_size <= mod.MESSAGE_TOKEN_SAMPLE_MAX_BYTES + 100
    # Result within +/-30 % of char count (1 char = 1 token for this encoder)
    expected = len(large_content) + mod.MESSAGE_TOKEN_OVERHEAD
    assert abs(count - expected) < expected * 0.3


def test_large_message_token_estimate_sampling_handles_none_on_encode_failure():
    """If the encoder fails on a sample, the estimate returns None."""

    class FailingEncoder:
        def encode(self, text, **kwargs):
            raise TypeError("boom")

    large_content = "x" * (mod.MESSAGE_TOKEN_EXACT_ENCODE_MAX_BYTES + 100)
    message = {"role": "user", "content": large_content}

    count = mod.estimate_message_tokens(
        message, encoder=FailingEncoder(), encoding_name="unit-test"
    )

    # _encode_text_token_count catches TypeError and returns None,
    # which propagates through _estimate_large_text_tokens_sampling.
    assert count is None


def test_message_token_estimate_treats_special_token_strings_as_plain_text():
    class SpecialAwareEncoder:
        def encode(self, text, **kwargs):
            if kwargs.get("disallowed_special") == ():
                return [1, 2, 3]
            raise ValueError("special token disallowed")

    count = mod.estimate_message_tokens(
        {"role": "user", "content": "literal <|endoftext|> in logs"},
        encoder=SpecialAwareEncoder(),
        encoding_name="unit-test",
    )

    assert count == 3 + mod.MESSAGE_TOKEN_OVERHEAD


def test_body_token_estimate_includes_provider_visible_tool_payload():
    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    body = {
        "messages": [{"role": "user", "content": "short"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "huge_tool",
                    "parameters": {"description": "tool schema payload " * 20},
                },
            }
        ],
    }

    message_only = mod.estimate_messages_tokens(body["messages"], encoder=LengthEncoder(), encoding_name="unit-test")
    body_total = mod.estimate_body_tokens(body, encoder=LengthEncoder(), encoding_name="unit-test")

    assert body_total > message_only


def test_body_token_estimate_ignores_provider_prompt_cache_hints_on_tools():
    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    body = {
        "messages": [{"role": "user", "content": "short"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    }
    with_cache_hint = copy.deepcopy(body)
    with_cache_hint["tools"][0]["cache_control"] = {"type": "ephemeral"}

    assert mod.estimate_body_tokens(body, encoder=LengthEncoder(), encoding_name="unit-test") == mod.estimate_body_tokens(
        with_cache_hint,
        encoder=LengthEncoder(),
        encoding_name="unit-test",
    )


@pytest.mark.asyncio
async def test_token_estimator_uses_db_config_tiktoken_encoding_before_legacy(monkeypatch, pipe_request):
    captured = []
    config_gets = []

    class FakeConfig:
        @staticmethod
        async def get(key):
            config_gets.append(key)
            assert key == mod.TIKTOKEN_ENCODING_CONFIG_KEY
            return "db_encoding"

    class DbOnlyEncoder:
        name = "db_encoding"

        def encode(self, text, **kwargs):
            return [0] * len(text)

    def get_encoding(name):
        captured.append(name)
        if name == "db_encoding":
            return DbOnlyEncoder()
        raise ValueError(f"unexpected encoding: {name}")

    config_module = types.ModuleType("open_webui.models.config")
    setattr(config_module, "Config", FakeConfig)
    tiktoken_module = types.ModuleType("tiktoken")
    setattr(tiktoken_module, "get_encoding", get_encoding)
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken_module)
    pipe_request.app.state.config = SimpleNamespace(TIKTOKEN_ENCODING_NAME="legacy_encoding")

    count = await mod.estimate_message_tokens_async({"role": "user", "content": "hello"}, request=pipe_request)
    second_count = await mod.estimate_message_tokens_async({"role": "user", "content": "again"}, request=pipe_request)

    assert count is not None
    assert second_count is not None
    assert captured[0] == "db_encoding"
    assert config_gets == [mod.TIKTOKEN_ENCODING_CONFIG_KEY]
    assert getattr(pipe_request.state, mod.AUTO_COMPACT_TIKTOKEN_ENCODING_STATE_KEY) == "db_encoding"


@pytest.mark.asyncio
async def test_token_estimator_falls_back_to_legacy_tiktoken_encoding_when_config_errors(
    monkeypatch, pipe_request
):
    captured = []

    class FakeConfig:
        @staticmethod
        async def get(key):
            assert key == mod.TIKTOKEN_ENCODING_CONFIG_KEY
            raise RuntimeError("config unavailable")

    class LegacyEncoder:
        name = "legacy_encoding"

        def encode(self, text, **kwargs):
            return [0] * len(text)

    def get_encoding(name):
        captured.append(name)
        if name == "legacy_encoding":
            return LegacyEncoder()
        raise ValueError(f"unexpected encoding: {name}")

    config_module = types.ModuleType("open_webui.models.config")
    setattr(config_module, "Config", FakeConfig)
    tiktoken_module = types.ModuleType("tiktoken")
    setattr(tiktoken_module, "get_encoding", get_encoding)
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken_module)
    pipe_request.app.state.config = SimpleNamespace(TIKTOKEN_ENCODING_NAME="legacy_encoding")

    count = await mod.estimate_message_tokens_async({"role": "user", "content": "hello"}, request=pipe_request)

    assert count is not None
    assert captured[0] == "legacy_encoding"
    assert not hasattr(pipe_request.state, mod.AUTO_COMPACT_TIKTOKEN_ENCODING_STATE_KEY)


@pytest.mark.asyncio
async def test_checkpoint_completion_stores_rendered_summary_token_count(monkeypatch, pipe_request, pipe_user):
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    store = ClaimCheckpointStore([])

    async def noop_initialize(**kwargs):
        return None

    async def estimate_rendered_summary_message_tokens(**kwargs):
        assert kwargs["summary_text"] == "stored summary"
        return 123

    async def summary_factory(parent):
        return "stored summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_estimate_rendered_summary_message_tokens", estimate_rendered_summary_message_tokens, raising=False)

    summary = await mod._get_or_create_checkpoint_summary(
        request=pipe_request,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        source_messages=source_messages,
        summary_meta={},
        summary_factory=summary_factory,
    )

    assert str(summary) == "stored summary"
    assert store.completed_rows[0]["summary_token_count"] == 123


@pytest.mark.asyncio
async def test_summary_token_count_from_checkpoint_uses_checkpoint_rendering(monkeypatch, pipe_request):
    source_messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    checkpoint = mod.build_checkpoint_row(
        namespace="ns",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash="profile",
        source_hash=mod.compute_source_hash(source_messages),
        source_message_count=len(source_messages),
        summary_text="Stored summary",
        summary_meta={mod.SUMMARY_META_FORMAT_VERSION_KEY: mod.SUMMARY_META_FORMAT_VERSION},
        summary_token_count=None,
        parent_checkpoint_id=None,
        now=123,
    )
    estimated_messages = []

    def estimate_message_tokens(message, **kwargs):
        estimated_messages.append(copy.deepcopy(message))
        return 17

    monkeypatch.setattr(mod, "estimate_message_tokens", estimate_message_tokens)

    count = await mod._summary_token_count_from_checkpoint(
        request=pipe_request,
        checkpoint=checkpoint,
        historical_source_messages=source_messages,
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
    )

    assert count == 17
    assert estimated_messages == [
        mod.render_summary_message_from_checkpoint(
            checkpoint,
            historical_source_messages=source_messages,
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        )
    ]
    assert "<historical_user_messages" not in estimated_messages[0]["content"]


@pytest.mark.asyncio
async def test_checkpoint_applied_estimate_batches_delta_tail_token_estimation(monkeypatch, pipe_request, pipe_user):
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    delta_messages = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    tail_messages = [{"role": "user", "content": "active"}]
    checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(parent_source),
        source_message_count=len(parent_source),
        summary_text="existing parent summary",
        summary_meta={},
        summary_token_count=20,
        parent_checkpoint_id=None,
        now=123,
    )
    body_calls = []

    async def estimate_body_tokens_async(body, **kwargs):
        body_calls.append(copy.deepcopy(body))
        return 33

    async def estimate_message_tokens_async(message, **kwargs):
        raise AssertionError("checkpoint-applied estimate must batch delta/tail token estimation")

    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "estimate_message_tokens_async", estimate_message_tokens_async)

    count = await mod._estimate_checkpoint_applied_body_tokens(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body={"messages": [*parent_source, *delta_messages, *tail_messages]},
        pipe_function_id="auto_compact",
        match=mod.ReusableCheckpointMatch(
            kind="parent",
            source_message_count=len(parent_source),
            source_kind="message",
            checkpoint=checkpoint,
        ),
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
        token_system_prompt="target system",
    )

    assert count == 53
    assert body_calls == [
        {
            "messages": [
                {"role": "system", "content": "target system"},
                *delta_messages,
                *tail_messages,
            ],
        }
    ]


@pytest.mark.asyncio
async def test_checkpoint_applied_estimate_includes_retained_file_context(monkeypatch, pipe_request, pipe_user):
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    delta_messages = [{"role": "user", "content": "middle"}]
    tail_messages = [{"role": "user", "content": "active"}]
    checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(parent_source),
        source_message_count=len(parent_source),
        summary_text="existing parent summary",
        summary_meta={},
        summary_token_count=20,
        parent_checkpoint_id=None,
        now=123,
    )
    captured = {}

    async def inject_target_file_context(**kwargs):
        captured["prefix_count"] = kwargs["compaction_prefix_count"]
        captured["metadata_files"] = copy.deepcopy(kwargs["metadata_files"])
        body = copy.deepcopy(kwargs["body"])
        body["messages"][-1]["content"] += "\nFILE_CONTEXT"
        return body

    async def estimate_body_tokens_async(body, **kwargs):
        captured["estimate_body"] = copy.deepcopy(body)
        return 33

    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)

    count = await mod._estimate_checkpoint_applied_body_tokens(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1", "files": [_file("retained")], "user_message": {"files": [_file("retained")]}},
        body={"metadata": {"files": [_file("retained")]}, "messages": [*parent_source, *delta_messages, *tail_messages]},
        pipe_function_id="auto_compact",
        match=mod.ReusableCheckpointMatch(
            kind="parent",
            source_message_count=len(parent_source),
            source_kind="message",
            checkpoint=checkpoint,
        ),
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
        file_context_enabled=True,
    )

    assert count == 53
    assert captured["prefix_count"] == len(parent_source)
    assert captured["metadata_files"] == [_file("retained")]
    assert captured["estimate_body"]["messages"] == [*delta_messages, {"role": "user", "content": "active\nFILE_CONTEXT"}]


def test_include_exclude_patterns_match_id_and_name_and_deduplicate_targets():
    valves = mod.Pipe.Valves(
        include_model_patterns="*gpt*,Anthropic*",
        exclude_model_patterns="*mini*",
    )
    models = [
        {"id": "gpt-4.1", "name": "GPT"},
        {"id": "gpt-4.1", "name": "duplicate"},
        {"id": "gpt-4.1-mini", "name": "GPT Mini"},
        {"id": "claude-sonnet", "name": "Anthropic Claude"},
        {"id": "llama", "name": "Local"},
    ]

    targets = mod.filter_target_models(models, valves)

    assert [m["id"] for m in targets] == ["gpt-4.1", "claude-sonnet"]


def test_wrapper_model_name_template_auto_uses_target_name_when_target_hidden():
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "gpt-4.1", "name": "GPT"},
        valves=valves,
    )

    assert form["name"] == "GPT"


def test_wrapper_model_name_template_supports_postfix_template():
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "gpt-4.1", "name": "GPT"},
        valves=mod.Pipe.Valves(wrapper_model_name_template="{target_name} (AutoCompact)"),
    )

    assert form["name"] == "GPT (AutoCompact)"


@pytest.mark.parametrize("template", ["{target_name.foo}", "{target_id[bad]}"])
def test_wrapper_model_name_template_falls_back_for_unsupported_format_syntax(template):
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "gpt-4.1", "name": "GPT"},
        valves=mod.Pipe.Valves(wrapper_model_name_template=template),
    )

    assert form["name"] == "GPT (AutoCompact)"


def test_wrapper_model_name_template_rejects_format_width_specs():
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "gpt-4.1", "name": "GPT"},
        valves=mod.Pipe.Valves(wrapper_model_name_template="{target_name:>100000}"),
    )

    assert len(form["name"]) < 1000
    assert form["name"] == "GPT (AutoCompact)"


@pytest.mark.parametrize("target_name", ["Model {target_id}", "Model {v2}"])
def test_wrapper_model_name_template_does_not_reinterpret_target_name_braces(target_name):
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "gpt-4.1", "name": target_name},
        valves=mod.Pipe.Valves(wrapper_model_name_template="Wrapped: {target_name}"),
    )

    assert form["name"] == f"Wrapped: {target_name}"


def test_wrapper_model_name_template_does_not_reinterpret_target_id_braces():
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "provider/{target_name}", "name": "GPT"},
        valves=mod.Pipe.Valves(wrapper_model_name_template="Wrapped: {target_id}"),
    )

    assert form["name"] == "Wrapped: provider/{target_name}"


def test_wrapper_model_name_prefix_is_not_migrated_to_template():
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "gpt-4.1", "name": "GPT"},
        valves=mod.Pipe.Valves(model_name_prefix="Legacy: "),
    )

    assert form["name"] == "GPT (AutoCompact)"


def test_wrapper_model_records_are_built_for_full_wrapper_id_with_function_owner():
    target = {
        "id": "gpt-4.1",
        "name": "GPT",
        "info": {
            "access_grants": [
                {
                    "id": "grant-1",
                    "principal_type": "group",
                    "principal_id": "team-1",
                    "permission": "read",
                }
            ]
        },
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert form["id"] == mod.build_wrapper_model_id("auto_compact", "gpt-4.1")
    assert form["user_id"] == "owner-1"
    assert form["base_model_id"] is None
    assert form["name"] == "GPT (AutoCompact)"
    assert form["access_grants"] == [
        {
            "id": "grant-1",
            "principal_type": "group",
            "principal_id": "team-1",
            "permission": "read",
        }
    ]


def test_wrapper_model_form_grants_target_owner_read_when_owner_differs_from_function_owner():
    target = {
        "id": "private-target",
        "name": "Private Target",
        "info": {"user_id": "target-owner", "access_grants": []},
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert {
        "principal_type": "user",
        "principal_id": "target-owner",
        "permission": "read",
    } in form["access_grants"]


def test_wrapper_model_form_preserves_public_read_grant_for_normal_user_visibility():
    target = {
        "id": "public-target",
        "name": "Public Target",
        "info": {
            "access_grants": [
                {
                    "principal_type": "user",
                    "principal_id": "*",
                    "permission": "read",
                }
            ]
        },
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert {
        "principal_type": "user",
        "principal_id": "*",
        "permission": "read",
    } in form["access_grants"]


def test_wrapper_model_form_inherits_target_meta_used_by_core_middleware():
    target = {
        "id": "target-with-meta",
        "name": "Target With Meta",
        "info": {
            "meta": {
                "filterIds": ["model-filter"],
                "actionIds": ["model-action"],
                "knowledge": [{"id": "knowledge-1", "name": "Knowledge"}],
                "capabilities": {"file_context": False, "builtin_tools": False},
                "skillIds": ["skill-1"],
                "toolIds": ["tool-1"],
                "defaultFeatureIds": ["web_search"],
                "terminalId": "terminal-1",
                "tags": [{"name": "target-tag"}],
                "description": "Target description",
            }
        },
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert form["meta"]["filterIds"] == ["model-filter"]
    assert form["meta"]["actionIds"] == ["model-action"]
    assert form["meta"]["knowledge"] == [{"id": "knowledge-1", "name": "Knowledge"}]
    assert form["meta"]["capabilities"] == {"file_context": False, "builtin_tools": False}
    assert form["meta"]["skillIds"] == ["skill-1"]
    assert form["meta"]["toolIds"] == ["tool-1"]
    assert form["meta"]["defaultFeatureIds"] == ["web_search"]
    assert form["meta"]["terminalId"] == "terminal-1"
    assert form["meta"]["tags"] == [{"name": "target-tag"}]
    assert form["meta"]["description"] == "Target description"
    assert form["meta"]["auto_compaction"] == {
        "pipe_function_id": "auto_compact",
        "target_model_id": "target-with-meta",
    }


def test_target_model_supports_file_context_respects_top_level_opt_out():
    models = {
        "target": {
            "id": "target",
            "capabilities": {"file_context": False},
            "info": {"meta": {"capabilities": {"file_context": True}}},
        }
    }

    assert mod._target_model_supports_file_context(models, "target") is False


def test_wrapper_model_form_disables_core_file_context_for_wrapper():
    target = {
        "id": "target-with-file-context",
        "name": "Target With File Context",
        "info": {
            "meta": {
                "capabilities": {
                    "file_context": True,
                    "vision": True,
                },
            },
        },
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert form["meta"]["capabilities"]["file_context"] is False
    assert form["meta"]["capabilities"]["vision"] is True


def test_classify_target_no_compaction_all_retained():
    metadata_files = [_file("old"), _file("current")]
    db_chain = [
        {"id": "m1", "role": "user", "files": [_file("old")]},
        {"id": "m2", "role": "user", "files": [_file("current")]},
    ]

    retained = mod._classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=0,
        metadata_user_message={"files": [_file("current")]},
        metadata_files=metadata_files,
    )

    assert retained == metadata_files


def test_classify_target_foreground_compaction():
    metadata_files = [_file("absorbed"), _file("also-absorbed"), _file("current")]
    db_chain = [
        {"id": "m1", "role": "user", "files": [_file("absorbed")]},
        {"id": "m2", "role": "assistant", "files": [_file("also-absorbed")]},
        {"id": "m3", "role": "user", "files": [_file("current")]},
    ]

    retained = mod._classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=2,
        metadata_user_message={"files": [_file("current")]},
        metadata_files=metadata_files,
    )

    assert retained == [_file("current")]


def test_classify_target_checkpoint_reuse():
    metadata_files = [_file("parent"), _file("delta"), _file("tail")]
    db_chain = [
        {"id": "m1", "role": "user", "files": [_file("parent")]},
        {"id": "m2", "role": "assistant"},
        {"id": "m3", "role": "user", "files": [_file("delta")]},
        {"id": "m4", "role": "assistant", "files": [_file("tail")]},
    ]

    retained = mod._classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=2,
        metadata_user_message={"files": [_file("delta")]},
        metadata_files=metadata_files,
    )

    assert retained == [_file("delta"), _file("tail")]


def test_classify_target_tool_result_compaction_no_whole_messages_absorbed():
    metadata_files = [_file("history"), _file("current")]
    db_chain = [
        {"id": "m1", "role": "user", "files": [_file("history")]},
        {"id": "m2", "role": "user", "files": [_file("current")]},
    ]

    retained = mod._classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=0,
        metadata_user_message={"files": [_file("current")]},
        metadata_files=metadata_files,
    )

    assert retained == metadata_files


def test_classify_target_db_unavailable_keeps_current_only():
    metadata_files = [_file("absorbed"), _file("current"), _file("knowledge")]

    retained = mod._classify_files_for_target(
        db_chain=None,
        compaction_prefix_count=2,
        metadata_user_message={"files": [_file("current")]},
        metadata_files=metadata_files,
    )

    assert retained == [_file("current")]


def test_classify_target_db_unavailable_retains_unidentifiable():
    unidentified = {"type": "file", "name": "unknown.txt"}
    retained = mod._classify_files_for_target(
        db_chain=None,
        compaction_prefix_count=2,
        metadata_user_message={"files": [_file("current")]},
        metadata_files=[_file("absorbed"), unidentified, _file("current")],
    )

    assert retained == [unidentified, _file("current")]


def test_classify_target_images_always_retained():
    image = _file("absorbed-image", file_type="image")
    retained = mod._classify_files_for_target(
        db_chain=[{"id": "m1", "role": "user", "files": [_file("absorbed"), image]}],
        compaction_prefix_count=1,
        metadata_user_message={},
        metadata_files=[_file("absorbed"), image],
    )

    assert retained == [image]


def test_classify_target_unidentifiable_retained():
    unidentified = {"type": "file", "name": "unknown.txt"}

    retained = mod._classify_files_for_target(
        db_chain=[{"id": "m1", "role": "user", "files": [_file("absorbed")]}],
        compaction_prefix_count=1,
        metadata_user_message={},
        metadata_files=[_file("absorbed"), unidentified],
    )

    assert retained == [unidentified]


def test_classify_target_chat_level_knowledge_retained():
    knowledge = _file("knowledge")

    retained = mod._classify_files_for_target(
        db_chain=[{"id": "m1", "role": "user", "files": [_file("absorbed")]}],
        compaction_prefix_count=1,
        metadata_user_message={},
        metadata_files=[_file("absorbed"), knowledge],
    )

    assert retained == [knowledge]


def test_classify_target_with_tool_call_expansion_in_prefix():
    """DB chain assistant-with-output expands via process_messages_with_output.
    compaction_prefix_count counts expanded messages, so the DB chain must be
    expanded too for positional alignment. A delta user message between the
    expanded assistant block and the current message is the critical test —
    without expansion its file would be silently dropped."""
    from open_webui.utils.middleware import process_messages_with_output

    assistant_with_output = {
        "id": "a1",
        "role": "assistant",
        "content": "",
        "output": [
            {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "result"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
        ],
    }
    # Unexpanded DB chain: 4 messages
    db_chain_raw = [
        {"id": "u1", "role": "user", "files": [_file("prefix-file")]},
        assistant_with_output,
        {"id": "u2", "role": "user", "files": [_file("delta-file")]},
        {"id": "u3", "role": "user", "files": [_file("current-file")]},
    ]
    # Expanded: process_messages_with_output produces 6 messages
    # [u1, assistant(tool_calls), tool(result), assistant(final), u2(delta), u3(current)]
    expanded = process_messages_with_output(db_chain_raw)
    assert len(expanded) == 6, f"Expected 6 expanded messages, got {len(expanded)}"

    # compaction_prefix_count=4 (absorb u1 + expanded assistant block)
    # Retained: range(4, 6) = [u2(delta), u3(current)]
    retained = mod._classify_files_for_target(
        db_chain=expanded,
        compaction_prefix_count=4,
        metadata_user_message={"files": [_file("current-file")]},
        metadata_files=[_file("prefix-file"), _file("delta-file"), _file("current-file")],
    )

    # delta-file AND current-file retained; prefix-file dropped
    assert retained == [_file("delta-file"), _file("current-file")]


@pytest.mark.asyncio
async def test_load_chat_message_chain_expands_assistant_with_output(monkeypatch):
    """Guards the process_messages_with_output call inside _load_chat_message_chain.
    Without expansion, positional indices would misalign when assistant messages
    have output (tool calls)."""
    assistant_with_output = {
        "id": "a1",
        "role": "assistant",
        "content": "",
        "output": [
            {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "result"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
        ],
    }
    messages_map = {
        "u1": {"id": "u1", "parentId": None, "role": "user", "content": "hi", "files": []},
        "a1": {"id": "a1", "parentId": "u1", **assistant_with_output},
        "u2": {"id": "u2", "parentId": "a1", "role": "user", "content": "bye", "files": []},
    }

    class FakeChats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id):
            return messages_map

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = FakeChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    chain = await mod._load_chat_message_chain(
        request=None,
        chat_id="chat-1",
        current_message_id="u2",
    )

    # Unexpanded would be 3 messages; expanded is 5.
    assert chain is not None
    assert len(chain) == 5, f"Expected 5 expanded messages, got {len(chain)}"


def test_classify_summary_skips_parent_absorbed():
    db_chain = [
        {"id": "m1", "role": "user", "files": [_file("parent-1")]},
        {"id": "m2", "role": "assistant", "files": [_file("parent-2")]},
        {"id": "m3", "role": "user", "files": [_file("delta-1")]},
        {"id": "m4", "role": "assistant", "files": [_file("delta-2")]},
        {"id": "m5", "role": "user", "files": [_file("tail")]},
    ]

    prefix_ids = mod._classify_files_for_summary(
        db_chain=db_chain,
        compaction_prefix_count=4,
        parent_source_message_count=2,
    )

    assert prefix_ids == {"delta-1", "delta-2"}


def test_classify_files_treats_transient_user_messages_like_system_boundaries():
    patterns = mod.parse_transient_message_patterns(TRANSIENT_MARKER)
    metadata_files = [_file("parent"), _file("volatile"), _file("delta"), _file("tail")]
    db_chain = [
        {"id": "m1", "role": "user", "files": [_file("parent")]},
        {
            "id": "m2",
            "role": "user",
            "content": "<SYSTEM_CONTEXT>now: 10:00</SYSTEM_CONTEXT>",
            "files": [_file("volatile")],
        },
        {"id": "m3", "role": "assistant", "files": [_file("delta")]},
        {"id": "m4", "role": "user", "files": [_file("tail")]},
    ]

    prefix_ids = mod._classify_files_for_summary(
        db_chain=db_chain,
        compaction_prefix_count=2,
        parent_source_message_count=1,
        transient_message_patterns=patterns,
    )
    retained = mod._classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=2,
        metadata_user_message={"files": []},
        metadata_files=metadata_files,
        transient_message_patterns=patterns,
    )

    assert prefix_ids == {"delta"}
    assert retained == [_file("tail")]


def test_classify_target_skips_db_chain_system_rows():
    metadata_files = [_file("absorbed"), _file("kept")]
    db_chain = [
        {"id": "m0", "role": "system", "content": "stored system"},
        {"id": "m1", "role": "user", "files": [_file("absorbed")]},
        {"id": "m2", "role": "user", "files": [_file("kept")]},
    ]

    retained = mod._classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=1,
        metadata_user_message={"files": []},
        metadata_files=metadata_files,
    )

    assert retained == [_file("kept")]


def test_classify_target_retains_leading_system_files():
    metadata_files = [_file("sys-doc"), _file("absorbed"), _file("kept")]
    db_chain = [
        {"id": "m0", "role": "system", "content": "preserved system", "files": [_file("sys-doc")]},
        {"id": "m1", "role": "user", "files": [_file("absorbed")]},
        {"id": "m2", "role": "user", "files": [_file("kept")]},
    ]

    retained = mod._classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=1,
        metadata_user_message={"files": []},
        metadata_files=metadata_files,
    )

    assert retained == [_file("sys-doc"), _file("kept")]


def test_classify_target_retains_mid_chain_first_system_files():
    metadata_files = [_file("absorbed"), _file("sys-doc"), _file("kept")]
    db_chain = [
        {"id": "m0", "role": "user", "files": [_file("absorbed")]},
        {"id": "m1", "role": "system", "content": "preserved system", "files": [_file("sys-doc")]},
        {"id": "m2", "role": "user", "files": [_file("kept")]},
    ]

    retained = mod._classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=1,
        metadata_user_message={"files": []},
        metadata_files=metadata_files,
    )

    assert retained == [_file("sys-doc"), _file("kept")]


def test_classify_target_prunes_second_system_row_files():
    metadata_files = [_file("preserved"), _file("absorbed-sys"), _file("kept")]
    db_chain = [
        {"id": "m0", "role": "system", "content": "preserved system", "files": [_file("preserved")]},
        {"id": "m1", "role": "system", "content": "absorbed system", "files": [_file("absorbed-sys")]},
        {"id": "m2", "role": "user", "content": "old"},
        {"id": "m3", "role": "user", "files": [_file("kept")]},
    ]

    retained = mod._classify_files_for_target(
        db_chain=db_chain,
        compaction_prefix_count=1,
        metadata_user_message={"files": []},
        metadata_files=metadata_files,
    )

    assert retained == [_file("preserved"), _file("kept")]


def test_classify_summary_skips_db_chain_system_rows():
    db_chain = [
        {"id": "m1", "role": "user", "files": [_file("first")]},
        {"id": "m2", "role": "system", "content": "stored system", "files": [_file("system-owned")]},
        {"id": "m3", "role": "user", "files": [_file("second")]},
    ]

    prefix_ids = mod._classify_files_for_summary(
        db_chain=db_chain,
        compaction_prefix_count=2,
        parent_source_message_count=0,
    )

    assert prefix_ids == {"first", "second"}


def test_wrapper_model_form_inherits_top_level_display_metadata_without_wrapper_description():
    target = {
        "id": "provider-target",
        "name": "Provider Target",
        "description": "Provider target description",
        "profile_image_url": "https://example.com/provider.png",
        "capabilities": {"vision": True},
        "tags": ["provider", {"name": "custom"}],
    }

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model=target,
        valves=mod.Pipe.Valves(),
    )

    assert form["meta"]["description"] == "Provider target description"
    assert form["meta"]["profile_image_url"] == "https://example.com/provider.png"
    assert form["meta"]["capabilities"] == {"vision": True, "file_context": False}
    assert form["meta"]["tags"] == [{"name": "provider"}, {"name": "custom"}]
    assert form["meta"]["auto_compaction"] == {
        "pipe_function_id": "auto_compact",
        "target_model_id": "provider-target",
    }


def test_wrapper_model_form_does_not_invent_description_when_target_has_none():
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="function-owner",
        target_model={"id": "plain-target", "name": "Plain Target"},
        valves=mod.Pipe.Valves(),
    )

    assert "description" not in form["meta"]
    assert form["meta"]["auto_compaction"] == {
        "pipe_function_id": "auto_compact",
        "target_model_id": "plain-target",
    }


def test_wrapper_model_form_keeps_only_core_metadata_params_and_forces_streaming():
    target_model_info = SimpleNamespace(
        params=SimpleNamespace(
            model_dump=lambda **kwargs: {
                "function_calling": "native",
                "stream_delta_chunk_size": 4,
                "reasoning_tags": ["think"],
                "stream_response": False,
                "system": "Target system prompt",
                "temperature": 0.2,
            }
        ),
        meta=SimpleNamespace(model_dump=lambda **kwargs: {}),
        access_grants=[],
        user_id="target-owner",
    )

    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="target-owner",
        target_model={"id": "target-with-params", "name": "Target With Params"},
        target_model_info=target_model_info,
        valves=mod.Pipe.Valves(),
    )

    assert form["params"] == {
        "function_calling": "native",
        "stream_delta_chunk_size": 4,
        "reasoning_tags": ["think"],
        "stream_response": True,
    }


def test_open_webui_0101_default_native_tool_loop_redispatch_preserves_wrapper_id():
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "provider-target")
    form_data = {
        "model": wrapper_id,
        "stream": False,
        "messages": [{"role": "user", "content": "call a tool"}],
        "params": {},
        "metadata": {"params": {}},
    }
    model_id = form_data["model"]

    assert form_data["metadata"]["params"].get("function_calling") != "legacy"

    new_form_data = {
        **form_data,
        "model": model_id,
        "stream": True,
        "metadata": form_data["metadata"],
        "messages": [
            *form_data["messages"],
            {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
        ],
    }

    assert new_form_data["model"] == wrapper_id
    assert mod.decode_wrapper_model_id(new_form_data["model"]).target_model_id == "provider-target"


@pytest.mark.asyncio
async def test_wrapper_sync_uses_function_owner_and_scoped_generated_wrapper_ids(monkeypatch):
    calls = {"insert": [], "update": []}

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            assert function_id == "auto_compact"
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelMeta:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id.startswith("auto_compact.")
            if model_id.endswith(mod.encode_target_model_id("existing-target")):
                return {
                    "id": model_id,
                    "base_model_id": None,
                    "name": "Stale wrapper",
                    "params": {},
                    "meta": {},
                    "access_grants": [],
                    "is_active": True,
                }
            return None

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[
            {"id": "existing-target", "name": "Existing", "info": {"access_grants": []}},
            {"id": "new-target", "name": "New", "info": {"access_grants": []}},
        ],
        valves=mod.Pipe.Valves(),
    )

    assert [call[0] for call in calls["update"]] == [mod.build_wrapper_model_id("auto_compact", "existing-target")]
    assert calls["insert"][0][0] == "function-owner"
    assert calls["insert"][0][1].id == mod.build_wrapper_model_id("auto_compact", "new-target")


@pytest.mark.asyncio
async def test_wrapper_sync_uses_target_model_record_params_for_wrapper_record(monkeypatch):
    calls = {"insert": [], "update": []}
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target-with-record")

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

        def model_dump(self, **kwargs):
            return dict(self.payload)

    class FakeModelMeta:
        def __init__(self, **kwargs):
            self.payload = kwargs

        def model_dump(self, **kwargs):
            return dict(self.payload)

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            if model_id == "target-with-record":
                return SimpleNamespace(
                    params=SimpleNamespace(
                        model_dump=lambda **kwargs: {
                            "function_calling": "native",
                            "temperature": 0.2,
                        }
                    ),
                    meta=SimpleNamespace(model_dump=lambda **kwargs: {"capabilities": {"builtin_tools": False}}),
                    access_grants=[],
                    user_id="function-owner",
                )
            if model_id == wrapper_id:
                return None
            raise AssertionError(f"unexpected model id: {model_id}")

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "target-with-record", "name": "Target With Record", "info": {"meta": {}}}],
        valves=mod.Pipe.Valves(),
    )

    inserted = calls["insert"][0][1]
    assert inserted.id == wrapper_id
    assert inserted.params.payload == {"function_calling": "native", "stream_response": True}
    assert inserted.meta.payload["capabilities"] == {"builtin_tools": False, "file_context": False}


@pytest.mark.asyncio
async def test_wrapper_sync_ignores_non_dict_top_level_provider_capabilities(monkeypatch):
    calls = {"insert": [], "update": []}
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "provider-target")

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelMeta(BaseModel):
        description: str | None = None
        profile_image_url: str | None = None
        capabilities: dict | None = None

        model_config = ConfigDict(extra="allow")

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            if model_id in {"provider-target", wrapper_id}:
                return None
            raise AssertionError(f"unexpected model id: {model_id}")

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[
            {
                "id": "provider-target",
                "name": "Provider Target",
                "description": "Provider target description",
                "profile_image_url": "https://example.com/provider.png",
                "capabilities": ["vision"],
            }
        ],
        valves=mod.Pipe.Valves(),
    )

    inserted = calls["insert"][0][1]
    meta = inserted.meta.model_dump(exclude_none=True)
    assert inserted.id == wrapper_id
    assert meta["description"] == "Provider target description"
    assert meta["profile_image_url"] == "https://example.com/provider.png"
    assert meta["capabilities"] == {"file_context": False}
    assert calls["update"] == []


@pytest.mark.asyncio
async def test_wrapper_sync_does_not_make_base_target_public_without_model_info(monkeypatch):
    calls = {"insert": [], "update": []}
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "base-target")

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelMeta:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == wrapper_id
            return {
                "id": wrapper_id,
                "base_model_id": None,
                "name": "Base Target (AutoCompact)",
                "params": {},
                "meta": {
                    "auto_compaction": {
                        "pipe_function_id": "auto_compact",
                        "target_model_id": "base-target",
                    },
                },
                "access_grants": [],
                "is_active": True,
            }

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "base-target", "name": "Base Target", "owned_by": "openai"}],
        valves=mod.Pipe.Valves(),
    )

    assert calls["update"][0][0] == wrapper_id
    assert calls["update"][0][1].access_grants == []
    assert calls["insert"] == []


@pytest.mark.asyncio
async def test_wrapper_sync_skips_update_when_existing_record_matches(monkeypatch):
    calls = {"insert": [], "update": []}
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "same-target")

    class FakeFunction:
        user_id = "function-owner"

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return FakeFunction()

    class FakeModelParams:
        def __init__(self, **kwargs):
            self.payload = kwargs

        def model_dump(self, **kwargs):
            return dict(self.payload)

    class FakeModelMeta:
        def __init__(self, **kwargs):
            self.payload = kwargs

        def model_dump(self, **kwargs):
            return dict(self.payload)

    class FakeModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    target = {
        "id": "same-target",
        "name": "Same Target",
        "info": {
            "access_grants": [
                {
                    "principal_type": "user",
                    "principal_id": "*",
                    "permission": "read",
                }
            ]
        },
    }

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == wrapper_id
            return {
                "id": wrapper_id,
                "base_model_id": None,
            "name": "Same Target (AutoCompact)",
                "params": {"stream_response": True},
                "meta": {
                    "auto_compaction": {
                        "pipe_function_id": "auto_compact",
                        "target_model_id": "same-target",
                    },
                    "capabilities": {"file_context": False},
                },
                "access_grants": [
                    {
                        "id": "grant-db-id",
                        "principal_type": "user",
                        "principal_id": "*",
                        "permission": "read",
                    }
                ],
                "is_active": True,
            }

        @staticmethod
        async def update_model_by_id(model_id, model_form):
            calls["update"].append((model_id, model_form))

        @staticmethod
        async def insert_new_model(model_form, user_id):
            calls["insert"].append((user_id, model_form))

    functions_module = types.ModuleType("open_webui.models.functions")
    functions_module.Functions = FakeFunctions
    models_module = types.ModuleType("open_webui.models.models")
    models_module.ModelForm = FakeModelForm
    models_module.ModelMeta = FakeModelMeta
    models_module.ModelParams = FakeModelParams
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.functions", functions_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[target],
        valves=mod.Pipe.Valves(),
    )

    assert calls == {"insert": [], "update": []}


@pytest.mark.asyncio
async def test_target_access_uses_core_chat_model_dict_when_db_record_lookup_is_unknown(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    target_model = {"id": "target", "name": "Target", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {"target": target_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "target"
            raise RuntimeError("DB record lookup unavailable")

    async def check_model_access(user, model, db=None):
        captured["user_type"] = type(user)
        captured["user_id"] = user.id
        captured["model"] = model

    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    await mod._validate_target_access(target_model_id="target", request=pipe_request, user=pipe_user)

    assert captured == {"user_type": fake_user_model, "user_id": "user-1", "model": target_model}


def test_iter_cache_models_reads_redisdict_like_values():
    class FakeRedisDict:
        def __init__(self, values):
            self._values = values

        def values(self):
            return list(self._values)

    state = SimpleNamespace(
        MODELS=FakeRedisDict([{"id": "redis-target", "name": "Redis Target"}, "not-a-model"]),
        BASE_MODELS=[],
        OPENAI_MODELS={},
        OLLAMA_MODELS={},
    )

    assert [model["id"] for model in mod._iter_cache_models_from_state(state)] == ["redis-target"]


@pytest.mark.asyncio
async def test_target_access_bypasses_core_access_check_for_admin_when_admin_bypass_enabled(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
    pipe_request.app.state.MODELS = {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}
    admin_user = {**pipe_user, "role": "admin"}

    async def check_model_access(user, model, db=None):
        raise AssertionError("admin target validation should match core chat path and skip per-model grants")

    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    config_module = types.ModuleType("open_webui.config")
    config_module.BYPASS_ADMIN_ACCESS_CONTROL = True
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.config", config_module)

    await mod._validate_target_access(target_model_id="target", request=pipe_request, user=admin_user)


@pytest.mark.asyncio
async def test_target_access_checks_admin_when_admin_bypass_disabled(monkeypatch, pipe_request, pipe_user):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    target_model = {"id": "target", "name": "Target", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {"target": target_model}
    admin_user = {**pipe_user, "role": "admin"}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "target"
            return SimpleNamespace(id="target")

    async def check_model_access(user, model, db=None):
        captured["user_type"] = type(user)
        captured["user_role"] = user.role
        captured["model"] = model

    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    config_module = types.ModuleType("open_webui.config")
    config_module.BYPASS_ADMIN_ACCESS_CONTROL = False
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.config", config_module)

    await mod._validate_target_access(target_model_id="target", request=pipe_request, user=admin_user)

    assert captured == {"user_type": fake_user_model, "user_role": "admin", "model": target_model}


@pytest.mark.asyncio
async def test_target_access_allows_admin_raw_provider_target_without_model_record(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
    target_model = {"id": "raw-target", "name": "Raw Target", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {"raw-target": target_model}
    admin_user = {**pipe_user, "role": "admin"}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "raw-target"
            return None

    async def check_model_access(user, model, db=None):
        raise AssertionError("admin raw provider target should not require a DB Model access check")

    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    config_module = types.ModuleType("open_webui.config")
    config_module.BYPASS_ADMIN_ACCESS_CONTROL = False
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.config", config_module)

    await mod._validate_target_access(target_model_id="raw-target", request=pipe_request, user=admin_user)


@pytest.mark.asyncio
async def test_target_access_rejects_disabled_provider_stale_cache_target(monkeypatch, pipe_request, pipe_user):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    pipe_request.app.state.config = SimpleNamespace(ENABLE_OPENAI_API=False, ENABLE_OLLAMA_API=False)
    pipe_request.app.state.MODELS = {
        "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "owned_by": "openai", "openai": {}}
    }
    pipe_request.app.state.BASE_MODELS = [
        {"id": "stale-ollama", "name": "Stale Ollama", "owned_by": "ollama", "ollama": {}}
    ]
    pipe_request.app.state.OPENAI_MODELS = {
        "direct-openai": {"id": "direct-openai", "name": "Direct OpenAI", "openai": {}}
    }
    pipe_request.app.state.OLLAMA_MODELS = {
        "direct-ollama": {"model": "direct-ollama", "name": "Direct Ollama"}
    }

    async def check_model_access(user, model, db=None):
        return None

    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    for target_model_id in ("stale-openai", "stale-ollama", "direct-openai", "direct-ollama"):
        with pytest.raises(HTTPException):
            await mod._validate_target_access(
                target_model_id=target_model_id,
                request=pipe_request,
                user=pipe_user,
            )


@pytest.mark.asyncio
async def test_target_access_rejects_config_disabled_provider_stale_cache_target(monkeypatch, pipe_request, pipe_user):
    install_fake_open_webui_user_model(monkeypatch)
    captured = {}

    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            captured["config_keys"] = keys
            return {"openai.enable": False, "ollama.enable": True}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "stale-openai"
            return None

    async def check_model_access(user, model, db=None):
        return None

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    pipe_request.app.state.config = SimpleNamespace()
    pipe_request.app.state.MODELS = {
        "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "owned_by": "openai", "openai": {}},
        "stale-ollama": {"id": "stale-ollama", "name": "Stale Ollama", "owned_by": "ollama", "ollama": {}},
    }
    pipe_request.app.state.BASE_MODELS = []
    pipe_request.app.state.OPENAI_MODELS = {}
    pipe_request.app.state.OLLAMA_MODELS = {}

    with pytest.raises(HTTPException):
        await mod._validate_target_access(
            target_model_id="stale-openai",
            request=pipe_request,
            user=pipe_user,
        )

    assert captured["config_keys"] == ("openai.enable", "ollama.enable")


def test_custom_model_fallback_excludes_only_runtime_own_wrapper_and_presets():
    own_wrapper_id = mod.build_wrapper_model_id("compact_alias", "target")
    other_wrapper_id = mod.build_wrapper_model_id("other_pipe", "target")
    models = {
        own_wrapper_id: {"id": own_wrapper_id, "owned_by": "openai"},
        "own-preset": {
            "id": "own-preset",
            "owned_by": "openai",
            "info": {"base_model_id": own_wrapper_id},
        },
        other_wrapper_id: {"id": other_wrapper_id, "owned_by": "openai"},
        "other-preset": {
            "id": "other-preset",
            "owned_by": "openai",
            "info": {"base_model_id": other_wrapper_id},
        },
    }

    assert (
        mod._available_custom_model_fallback_id(
            own_wrapper_id,
            models,
            pipe_function_id="compact_alias",
        )
        is None
    )
    assert (
        mod._available_custom_model_fallback_id(
            "own-preset",
            models,
            pipe_function_id="compact_alias",
        )
        is None
    )
    assert (
        mod._available_custom_model_fallback_id(
            other_wrapper_id,
            models,
            pipe_function_id="compact_alias",
        )
        == other_wrapper_id
    )
    assert (
        mod._available_custom_model_fallback_id(
            "other-preset",
            models,
            pipe_function_id="compact_alias",
        )
        == "other-preset"
    )


@pytest.mark.parametrize(
    "arena_meta",
    [
        {"model_ids": ["own-wrapper", "own-preset", "other-wrapper", "other-preset"]},
        {"model_ids": ["excluded"], "filter_mode": "exclude"},
        {},
    ],
)
def test_arena_candidates_exclude_only_runtime_own_wrapper_and_presets(arena_meta):
    own_wrapper_id = mod.build_wrapper_model_id("compact_alias", "target")
    other_wrapper_id = mod.build_wrapper_model_id("other_pipe", "target")
    aliases = {
        "own-wrapper": own_wrapper_id,
        "other-wrapper": other_wrapper_id,
    }
    normalized_meta = copy.deepcopy(arena_meta)
    if "model_ids" in normalized_meta:
        normalized_meta["model_ids"] = [aliases.get(model_id, model_id) for model_id in normalized_meta["model_ids"]]
    arena_model = {
        "id": "arena",
        "owned_by": "arena",
        "arena": True,
        "info": {"meta": normalized_meta},
    }
    models = {
        "arena": arena_model,
        own_wrapper_id: {"id": own_wrapper_id, "owned_by": "openai"},
        "own-preset": {
            "id": "own-preset",
            "owned_by": "openai",
            "base_model_id": own_wrapper_id,
        },
        other_wrapper_id: {"id": other_wrapper_id, "owned_by": "openai"},
        "other-preset": {
            "id": "other-preset",
            "owned_by": "openai",
            "base_model_id": other_wrapper_id,
        },
        "excluded": {"id": "excluded", "owned_by": "openai"},
    }

    candidates = mod._arena_chat_candidate_model_ids(
        models,
        arena_model,
        pipe_function_id="compact_alias",
    )

    assert own_wrapper_id not in candidates
    assert "own-preset" not in candidates
    assert other_wrapper_id in candidates
    assert "other-preset" in candidates


@pytest.mark.asyncio
async def test_target_access_rejects_custom_model_when_base_model_is_unavailable(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    pipe_request.app.state.config = SimpleNamespace(ENABLE_OPENAI_API=False, ENABLE_OLLAMA_API=False)
    pipe_request.app.state.MODELS = {
        "workspace-preset": {
            "id": "workspace-preset",
            "name": "Workspace Preset",
            "owned_by": "openai",
            "preset": True,
            "info": {"base_model_id": "stale-openai"},
        }
    }
    pipe_request.app.state.BASE_MODELS = []
    pipe_request.app.state.OPENAI_MODELS = {
        "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "openai": {}}
    }
    pipe_request.app.state.OLLAMA_MODELS = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            if model_id == "workspace-preset":
                return SimpleNamespace(id="workspace-preset", base_model_id="stale-openai")
            assert model_id == "fallback-legacy"
            return None

    async def check_model_access(user, model, db=None):
        return None

    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    with pytest.raises(HTTPException):
        await mod._validate_target_access(
            target_model_id="workspace-preset",
            request=pipe_request,
            user=pipe_user,
        )


@pytest.mark.asyncio
async def test_target_access_allows_custom_model_missing_base_when_core_fallback_is_available(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
    class FakeConfig:
        @staticmethod
        async def get(key):
            raise RuntimeError("config unavailable")

        @staticmethod
        async def get_many(*keys):
            raise RuntimeError("config unavailable")

    install_fake_open_webui_config(monkeypatch, FakeConfig)
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-model", "name": "Fallback", "owned_by": "openai", "openai": {}}
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-model,other")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-model": fallback_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(id="workspace-preset", base_model_id="stale-openai")

    async def check_model_access(user, model, db=None):
        captured["model"] = model

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    await mod._validate_target_access(
        target_model_id="workspace-preset",
        request=pipe_request,
        user=pipe_user,
    )

    assert captured["model"] == target_model


@pytest.mark.asyncio
async def test_target_access_allows_custom_model_when_base_model_is_available(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            raise RuntimeError("config unavailable")

    install_fake_open_webui_config(monkeypatch, FakeConfig)
    base_model = {"id": "available-base", "name": "Available Base", "owned_by": "openai", "openai": {}}
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "available-base"},
    }
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "available-base": base_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(id="workspace-preset", base_model_id="available-base")

    async def check_model_access(user, model, db=None):
        captured["model"] = model

    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    await mod._validate_target_access(
        target_model_id="workspace-preset",
        request=pipe_request,
        user=pipe_user,
    )

    assert captured["model"] == target_model


@pytest.mark.asyncio
async def test_target_access_honors_global_model_access_bypass(monkeypatch, pipe_request, pipe_user):
    install_fake_open_webui_user_model(monkeypatch)
    pipe_request.app.state.MODELS = {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}

    async def check_model_access(user, model, db=None):
        raise AssertionError("global BYPASS_MODEL_ACCESS_CONTROL should skip per-model grants")

    env_module = types.ModuleType("open_webui.env")
    env_module.BYPASS_MODEL_ACCESS_CONTROL = True
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    await mod._validate_target_access(target_model_id="target", request=pipe_request, user=pipe_user)


@pytest.mark.asyncio
async def test_target_access_can_resolve_targets_from_base_model_cache(monkeypatch, pipe_request, pipe_user):
    install_fake_open_webui_user_model(monkeypatch)
    target_model = {"id": "LiteLLM.glm-5.1", "name": "GLM", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {
        mod.build_wrapper_model_id("auto_compact", "LiteLLM.glm-5.1"): {
            "id": mod.build_wrapper_model_id("auto_compact", "LiteLLM.glm-5.1"),
            "name": "GLM (AutoCompact)",
            "pipe": {"type": "pipe"},
        }
    }
    pipe_request.app.state.BASE_MODELS = [target_model]
    captured = {}

    async def check_model_access(user, model, db=None):
        captured["model"] = model

    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)

    await mod._validate_target_access(target_model_id="LiteLLM.glm-5.1", request=pipe_request, user=pipe_user)

    assert captured["model"] == target_model


@pytest.mark.asyncio
async def test_target_access_rejects_generated_wrapper_for_runtime_registered_id(pipe_request, pipe_user):
    wrapper_id = mod.build_wrapper_model_id("compact_alias", "target")
    pipe_request.app.state.MODELS = {
        wrapper_id: {"id": wrapper_id, "name": "Alias Wrapper", "owned_by": "openai"}
    }

    with pytest.raises(HTTPException):
        await mod._validate_target_access(
            target_model_id=wrapper_id,
            request=pipe_request,
            user=pipe_user,
            pipe_function_id="compact_alias",
        )


@pytest.mark.asyncio
async def test_forward_target_injects_resolved_base_model_into_request_models(monkeypatch, pipe_request, pipe_user):
    target_model = {"id": "LiteLLM.glm-5.1", "name": "GLM", "owned_by": "openai"}
    pipe_request.app.state.MODELS = {}
    pipe_request.app.state.BASE_MODELS = [target_model]
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["has_target"] = form_data["model"] in request.app.state.MODELS
        captured["target_model"] = request.app.state.MODELS.get(form_data["model"])
        return StreamingResponse(
            iter([b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n']),
            media_type="text/event-stream",
        )

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    response = await mod._forward_streaming_target(
        request=pipe_request,
        user=pipe_user,
        body={"model": "LiteLLM.glm-5.1", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id=mod.build_wrapper_model_id("auto_compact", "LiteLLM.glm-5.1"),
    )
    async for _ in response.body_iterator:
        pass

    assert captured == {"has_target": True, "target_model": target_model}


@pytest.mark.asyncio
async def test_forward_target_reapplies_target_system_prompt_on_fresh_body(monkeypatch, pipe_request, pipe_user):
    pipe_request.app.state.MODELS = {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}
    pipe_request.state.bypass_system_prompt = True
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        request.state.bypass_filter = bypass_filter
        request.state.bypass_system_prompt = bypass_system_prompt
        captured["bypass_filter"] = bypass_filter
        captured["bypass_system_prompt"] = bypass_system_prompt
        captured["state_bypass_system_prompt"] = getattr(request.state, "bypass_system_prompt", None)
        return StreamingResponse(
            iter([b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n']),
            media_type="text/event-stream",
        )

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    response = await mod._forward_streaming_target(
        request=pipe_request,
        user=pipe_user,
        body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id=mod.build_wrapper_model_id("auto_compact", "target"),
    )
    async for _ in response.body_iterator:
        pass

    assert captured == {
        "bypass_filter": True,
        "bypass_system_prompt": False,
        "state_bypass_system_prompt": False,
    }
    assert pipe_request.state.bypass_system_prompt is True
    assert not hasattr(pipe_request.state, "bypass_filter")


def _install_real_core_provider_capture(
    monkeypatch,
    request,
    *,
    provider,
    responses=None,
):
    import open_webui.utils.chat as core_chat

    captured = []
    queued_responses = list(responses or [])

    class FakeParams:
        def model_dump(self):
            return {"system": "TARGET {{CURRENT_TIME}}"}

    class FakeModelInfo:
        base_model_id = None
        params = FakeParams()

    async def get_model_by_id(model_id):
        assert model_id == "target"
        return FakeModelInfo()

    monkeypatch.setattr(core_chat.Models, "get_model_by_id", staticmethod(get_model_by_id))
    request.app.state.MODELS = {
        "target": {"id": "target", "name": "Target", "owned_by": provider},
    }

    if provider == "openai":
        import open_webui.routers.openai as provider_router

        request.app.state.OPENAI_MODELS = {"target": {"urlIdx": 0}}

        async def config_get(key, default=None):
            if key == "openai.enable":
                return True
            return default

        class FakeResponse:
            status = 200
            headers = {"Content-Type": "application/json"}

            async def json(self, **kwargs):
                if queued_responses:
                    return queued_responses.pop(0)
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

            async def text(self):
                return ""

        class FakeSession:
            async def request(self, **kwargs):
                captured.append(json.loads(kwargs["data"]))
                return FakeResponse()

        async def get_openai_connection(index):
            assert index == 0
            return "http://provider", "key", {}

        async def get_headers_and_cookies(*args, **kwargs):
            return {}, {}

        async def get_session():
            return FakeSession()

        async def cleanup_response(response):
            return None

        async def check_model_access(*args, **kwargs):
            return None

        monkeypatch.setattr(provider_router, "Config", SimpleNamespace(get=config_get), raising=False)
        monkeypatch.setattr(provider_router, "get_openai_connection", get_openai_connection)
        monkeypatch.setattr(provider_router, "get_headers_and_cookies", get_headers_and_cookies)
        monkeypatch.setattr(provider_router, "get_session", get_session)
        monkeypatch.setattr(provider_router, "cleanup_response", cleanup_response)
        monkeypatch.setattr(provider_router, "check_model_access", check_model_access)
    else:
        import open_webui.routers.ollama as provider_router

        async def config_get(key, default=None):
            if key == "ollama.enable":
                return True
            return default

        async def get_ollama_url(*args, **kwargs):
            return "http://provider", 0

        async def send_request(*args, **kwargs):
            captured.append(json.loads(kwargs["payload"]))
            if queued_responses:
                return queued_responses.pop(0)
            return {"model": "target", "message": {"role": "assistant", "content": "ok"}, "done": True}

        async def check_model_access(*args, **kwargs):
            return None

        monkeypatch.setattr(provider_router, "Config", SimpleNamespace(get=config_get), raising=False)
        monkeypatch.setattr(provider_router, "get_ollama_url", get_ollama_url)
        monkeypatch.setattr(provider_router, "resolve_api_config", lambda *args, **kwargs: {})
        monkeypatch.setattr(provider_router, "get_api_key", lambda *args, **kwargs: None)
        monkeypatch.setattr(provider_router, "send_request", send_request)
        monkeypatch.setattr(provider_router, "check_model_access", check_model_access)

    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "ollama"])
async def test_real_core_provider_system_prefix_is_stable_across_tool_rounds(
    monkeypatch,
    pipe_request,
    pipe_user,
    provider,
):
    captured = _install_real_core_provider_capture(
        monkeypatch,
        pipe_request,
        provider=provider,
    )
    common_messages = [
        {"role": "system", "content": "CHAT SYSTEM"},
        {"role": "user", "content": "call a tool"},
    ]
    tool_suffix = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
    metadata = {"variables": {"{{CURRENT_TIME}}": "12:34:56 PM"}}

    for outer_bypass, messages in ((False, common_messages), (True, [*common_messages, *tool_suffix])):
        pipe_request.state.bypass_system_prompt = outer_bypass
        await mod._call_target_completion(
            request=pipe_request,
            user=pipe_user,
            body={
                "model": "target",
                "stream": False,
                "messages": copy.deepcopy(messages),
                "metadata": copy.deepcopy(metadata),
            },
        )

    assert len(captured) == 2
    assert captured[0]["messages"][0] == {
        "role": "system",
        "content": "TARGET 12:34:56 PM\nCHAT SYSTEM",
    }
    assert captured[1]["messages"][: len(captured[0]["messages"])] == captured[0]["messages"]
    assert pipe_request.state.bypass_system_prompt is True
    assert not hasattr(pipe_request.state, "bypass_filter")


@pytest.mark.asyncio
async def test_real_core_summary_retry_keeps_a_single_stable_system_prefix(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = _install_real_core_provider_capture(
        monkeypatch,
        pipe_request,
        provider="openai",
        responses=[
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {"choices": [{"message": {"role": "assistant", "content": "summary"}}]},
        ],
    )
    pipe_request.state.bypass_system_prompt = True
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={
            "chat_id": "chat-1",
            "variables": {"{{CURRENT_TIME}}": "12:34:56 PM"},
        },
        summary_model_id="target",
        source_messages=[{"role": "user", "content": "old"}],
        preserved_system_message={"role": "system", "content": "CHAT SYSTEM"},
        base_body={
            "model": "target",
            "stream": False,
            "messages": [{"role": "user", "content": "old"}],
            "tools": tools,
            "tool_choice": "auto",
        },
        file_context_enabled=False,
    )

    assert result == "summary"
    assert len(captured) == 2
    assert [payload["messages"][0] for payload in captured] == [
        {"role": "system", "content": "TARGET 12:34:56 PM\nCHAT SYSTEM"},
        {"role": "system", "content": "TARGET 12:34:56 PM\nCHAT SYSTEM"},
    ]
    assert "tools" in captured[0]
    assert "tools" not in captured[1]


def test_summary_model_validation_allows_pipe_backed_and_rejects_arena():
    models = {
        "target": {"id": "target", "name": "Target"},
        "pipe.summary": {"id": "pipe.summary", "pipe": {"type": "pipe"}, "name": "Summary Pipe"},
        "arena": {"id": "arena", "owned_by": "arena", "arena": True},
    }

    assert mod.validate_summary_model_id("", "target", models) == "target"
    assert mod.validate_summary_model_id("pipe.summary", "target", models) == "pipe.summary"

    with pytest.raises(ValueError, match="arena"):
        mod.validate_summary_model_id("arena", "target", models)


def test_summary_model_validation_rejects_missing_configured_model():
    models = {"target": {"id": "target", "name": "Target"}}

    with pytest.raises(ValueError, match="not found"):
        mod.validate_summary_model_id("missing.summary", "target", models)


@pytest.mark.asyncio
async def test_summary_model_validation_rejects_config_disabled_provider_cache_model(
    monkeypatch, pipe_request
):
    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            return {"openai.enable": False, "ollama.enable": True}

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    pipe_request.app.state.config = SimpleNamespace()
    pipe_request.app.state.MODELS = {
        "target": {"id": "target", "name": "Target"},
        "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "openai": {}},
    }
    pipe_request.app.state.BASE_MODELS = []
    pipe_request.app.state.OPENAI_MODELS = {}
    pipe_request.app.state.OLLAMA_MODELS = {}

    models = await mod._model_dict_from_request(pipe_request)

    with pytest.raises(ValueError, match="not found"):
        mod.validate_summary_model_id("stale-openai", "target", models)


def test_valve_defaults_are_conservative_for_v1_continuation():
    valves = mod.Pipe.Valves()

    assert valves.trigger_input_tokens == mod.DEFAULT_TRIGGER_INPUT_TOKENS
    assert valves.force_include_usage is True
    assert valves.compact_task_prompts_from_task_body is False
    assert valves.summary_tool_policy == "fallback_on_tool_call"
    assert valves.historical_message_excerpt_bytes == mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES
    assert valves.historical_message_excerpt_count == mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT
    assert not hasattr(valves, "keep_tail_messages")
    assert not hasattr(valves, "preserve_latest_tool_rounds")


def test_valve_schema_uses_release_field_names():
    fields = set(mod.Pipe.Valves.model_fields)

    assert {
        "wrapper_model_name_template",
        "trigger_input_tokens",
        "per_model_overrides_json",
        "transient_message_patterns",
        "token_status_show_usage_and_estimate",
    } <= fields
    assert fields.isdisjoint(
        {
            "model_name_template",
            "trigger_total_tokens",
            "trigger_total_tokens_overrides_json",
            "transient_message_markers",
            "token_status_compare_estimate",
        }
    )


def test_per_model_overrides_default_is_empty_string():
    valves = mod.Pipe.Valves()

    assert valves.per_model_overrides_json == ""


def test_soft_trigger_ratio_default_resolves_against_hard_trigger():
    valves = mod.Pipe.Valves()

    assert valves.soft_trigger_ratio == mod.DEFAULT_SOFT_TRIGGER_RATIO
    assert (
        mod.resolve_soft_trigger_input_tokens(
            valves,
            {"id": "target", "name": "Target"},
            valves.trigger_input_tokens,
        )
        == int(mod.DEFAULT_TRIGGER_INPUT_TOKENS * mod.DEFAULT_SOFT_TRIGGER_RATIO)
    )


def test_soft_trigger_ratio_missing_field_falls_back_to_default_constant():
    valves = SimpleNamespace(per_model_overrides_json="")

    resolved = mod.resolve_soft_trigger_input_tokens(valves, {"id": "target", "name": "Target"}, 1000)

    assert resolved == int(1000 * mod.DEFAULT_SOFT_TRIGGER_RATIO)


def test_per_model_overrides_empty_string_is_valid():
    valves = mod.Pipe.Valves(per_model_overrides_json="")

    assert valves.per_model_overrides_json == ""


def test_per_model_overrides_accepts_valid_ordered_json():
    payload = json.dumps(
        {
            "schema_version": 1,
            "overrides": [
                {"model_patterns": ["claude-fable-5[1m]"], "trigger_input_tokens": 950000},
                {"model_patterns": ["claude-*", "Claude *"], "trigger_input_tokens": 160000},
            ],
        }
    )

    valves = mod.Pipe.Valves(per_model_overrides_json=payload)

    assert valves.per_model_overrides_json == payload


def test_per_model_overrides_rejects_non_json_string():
    with pytest.raises(ValidationError):
        mod.Pipe.Valves(per_model_overrides_json="not json")


def test_per_model_overrides_rejects_non_list_overrides():
    payload = json.dumps({"overrides": {"model_patterns": ["claude-*"], "trigger_input_tokens": 1}})

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(per_model_overrides_json=payload)


def test_per_model_overrides_rejects_empty_model_patterns():
    payload = json.dumps({"overrides": [{"model_patterns": [], "trigger_input_tokens": 160000}]})

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(per_model_overrides_json=payload)


def test_per_model_overrides_rejects_zero_trigger_input_tokens():
    payload = json.dumps({"overrides": [{"model_patterns": ["claude-*"], "trigger_input_tokens": 0}]})

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(per_model_overrides_json=payload)


def test_per_model_overrides_rejects_unknown_root_key():
    payload = json.dumps(
        {
            "overrides": [{"model_patterns": ["claude-*"], "trigger_input_tokens": 100}],
            "soft_trigger_input_tokens": 50,
        }
    )

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(per_model_overrides_json=payload)


def test_per_model_overrides_rejects_legacy_trigger_total_tokens():
    payload = json.dumps(
        {
            "overrides": [
                {"model_patterns": ["*"], "trigger_total_tokens": 100}
            ]
        }
    )

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(per_model_overrides_json=payload)


def test_per_model_overrides_accepts_soft_trigger_ratio_without_hard_threshold():
    payload = json.dumps({"overrides": [{"model_patterns": ["claude-*"], "soft_trigger_ratio": 0.5}]})

    valves = mod.Pipe.Valves(per_model_overrides_json=payload)

    assert (
        mod.resolve_trigger_input_tokens(valves, {"id": "claude-sonnet", "name": "Sonnet"})
        == valves.trigger_input_tokens
    )
    assert (
        mod.resolve_soft_trigger_input_tokens(
            valves,
            {"id": "claude-sonnet", "name": "Sonnet"},
            100000,
        )
        == 50000
    )


def test_resolve_trigger_input_tokens_matches_target_model_id_and_name():
    valves = mod.Pipe.Valves(
        trigger_input_tokens=100000,
        per_model_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-*"], "trigger_input_tokens": 160000}]}
        ),
    )

    by_id = mod.resolve_trigger_input_tokens(valves, {"id": "claude-sonnet", "name": "Anthropic"})
    by_name = mod.resolve_trigger_input_tokens(valves, {"id": "anthropic-1", "name": "claude-opus"})

    assert by_id == 160000
    assert by_name == 160000


def test_resolve_trigger_input_tokens_first_match_wins():
    valves = mod.Pipe.Valves(
        trigger_input_tokens=100000,
        per_model_overrides_json=json.dumps(
            {
                "overrides": [
                    {"model_patterns": ["claude-opus"], "trigger_input_tokens": 950000},
                    {"model_patterns": ["claude-*"], "trigger_input_tokens": 160000},
                ]
            }
        ),
    )

    resolved = mod.resolve_trigger_input_tokens(valves, {"id": "claude-opus", "name": "Opus"})

    assert resolved == 950000


def test_resolve_trigger_input_tokens_matches_literal_bracket_id_without_escaping():
    # "[" and "]" are literal (not glob classes), so a bracketed id matches as-is and a
    # different id does not accidentally match via character-class expansion.
    valves = mod.Pipe.Valves(
        trigger_input_tokens=100000,
        per_model_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-opus-4-8[1m]"], "trigger_input_tokens": 950000}]}
        ),
    )

    assert mod.resolve_trigger_input_tokens(valves, {"id": "claude-opus-4-8[1m]", "name": "Opus 1M"}) == 950000
    assert mod.resolve_trigger_input_tokens(valves, {"id": "claude-opus-4-81", "name": "Opus"}) == 100000


def test_resolve_trigger_input_tokens_bulk_matches_bracket_suffix_with_wildcard():
    # Motivating bulk case: one override for every "[1m]" model id via "*[1m]".
    valves = mod.Pipe.Valves(
        trigger_input_tokens=100000,
        per_model_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["*[1m]"], "trigger_input_tokens": 950000}]}
        ),
    )

    assert mod.resolve_trigger_input_tokens(valves, {"id": "claude-opus-4-8[1m]", "name": "Opus"}) == 950000
    assert mod.resolve_trigger_input_tokens(valves, {"id": "claude-fable-5[1m]", "name": "Fable"}) == 950000
    assert mod.resolve_trigger_input_tokens(valves, {"id": "claude-opus-4-8", "name": "Opus"}) == 100000


def test_resolve_soft_trigger_input_tokens_uses_effective_hard_threshold():
    valves = mod.Pipe.Valves(soft_trigger_ratio=0.75)

    assert mod.resolve_soft_trigger_input_tokens(valves, {"id": "target", "name": "Target"}, 1000) == 750
    assert mod.resolve_soft_trigger_input_tokens(valves, {"id": "target", "name": "Target"}, 1001) == 750


def test_resolve_soft_trigger_input_tokens_uses_model_ratio_override():
    valves = mod.Pipe.Valves(
        soft_trigger_ratio=0.8,
        per_model_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-*"], "soft_trigger_ratio": 0.5}]}
        ),
    )

    assert mod.resolve_soft_trigger_input_tokens(valves, {"id": "claude-sonnet", "name": "Sonnet"}, 1000) == 500
    assert mod.resolve_soft_trigger_input_tokens(valves, {"id": "other", "name": "Other"}, 1000) == 800


def test_resolve_soft_trigger_input_tokens_zero_ratio_disables_prefetch():
    global_disabled = mod.Pipe.Valves(soft_trigger_ratio=0)
    override_disabled = mod.Pipe.Valves(
        soft_trigger_ratio=0.8,
        per_model_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-*"], "soft_trigger_ratio": 0}]}
        ),
    )

    assert mod.resolve_soft_trigger_input_tokens(global_disabled, {"id": "target", "name": "Target"}, 1000) is None
    assert (
        mod.resolve_soft_trigger_input_tokens(
            override_disabled,
            {"id": "claude-sonnet", "name": "Sonnet"},
            1000,
        )
        is None
    )


def test_soft_trigger_ratio_rejects_one_or_greater():
    with pytest.raises(ValidationError):
        mod.Pipe.Valves(soft_trigger_ratio=1)

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(soft_trigger_ratio=math.nan)


def test_per_model_overrides_rejects_non_finite_soft_trigger_ratio():
    payload = json.dumps({"overrides": [{"model_patterns": ["claude-*"], "soft_trigger_ratio": math.nan}]})

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(per_model_overrides_json=payload)


def test_transient_message_patterns_reject_invalid_regex():
    with pytest.raises(ValidationError) as exc_info:
        mod.Pipe.Valves(transient_message_patterns="[")

    assert "transient_message_patterns line 1" in str(exc_info.value)


def test_matches_any_pattern_uses_star_question_wildcards_with_literal_brackets():
    model = {"id": "claude-opus-4-8[1m]", "name": "Opus 1M"}

    assert mod._matches_any_pattern(model, ["claude-opus-4-8[1m]"])  # exact id, brackets literal
    assert mod._matches_any_pattern(model, ["Opus 1M"])  # exact name
    assert mod._matches_any_pattern(model, ["*[1m]"])  # bulk suffix, brackets literal
    assert mod._matches_any_pattern(model, ["claude-*"])  # "*" wildcard
    assert mod._matches_any_pattern(model, ["claude-?pus-4-8[1m]"])  # "?" single char
    assert not mod._matches_any_pattern(model, ["gpt-*"])  # non-match
    # character classes are not special: "[1m]" never expands to one-of "1"/"m"
    assert not mod._matches_any_pattern({"id": "claude-opus-4-81", "name": "Opus"}, ["claude-opus-4-8[1m]"])


def test_resolve_trigger_input_tokens_falls_back_to_global_default_on_no_match():
    valves = mod.Pipe.Valves(
        trigger_input_tokens=100000,
        per_model_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-*"], "trigger_input_tokens": 160000}]}
        ),
    )

    resolved = mod.resolve_trigger_input_tokens(valves, {"id": "gpt-4.1", "name": "GPT"})

    assert resolved == 100000


def test_resolve_trigger_input_tokens_falls_back_when_overrides_empty():
    valves = mod.Pipe.Valves(trigger_input_tokens=123456)

    resolved = mod.resolve_trigger_input_tokens(valves, {"id": "claude-sonnet", "name": "Anthropic"})

    assert resolved == 123456


def test_usage_extraction_handles_chat_completions_and_responses_api_shapes():
    chat = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    response = {
        "type": "response.completed",
        "response": {"usage": {"input_tokens": 7, "output_tokens": 3}},
    }

    assert mod.extract_usage_from_stream_payload(chat) == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    assert mod.extract_usage_from_stream_payload(response)["total_tokens"] == 10

    llama = {"timings": {"prompt_n": 3, "cache_n": 7, "predicted_n": 2}}
    raw_llama = mod._raw_usage_from_stream_payload(llama)
    assert raw_llama == llama["timings"]
    assert mod._strict_usage_input_tokens(raw_llama) == 10


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens": 100}, 100),
        ({"prompt_eval_count": 100}, 100),
        ({"prompt_n": 25, "cache_n": 75}, 100),
        (
            {
                "input_tokens": 60,
                "cache_creation_input_tokens": 15,
                "cache_read_input_tokens": 25,
            },
            100,
        ),
    ],
)
def test_strict_usage_input_tokens_accepts_complete_provider_shapes(usage, expected):
    assert mod._strict_usage_input_tokens(usage) == expected


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_n": 100},
        {"prompt_n": 100, "input_tokens": 100},
        {"total_tokens": 100},
        {"prompt_tokens": 0},
        {"prompt_tokens": 0, "input_tokens": 100},
        {"input_tokens": 0},
        {"input_tokens": 100, "cache_read_input_tokens": "10"},
    ],
)
def test_strict_usage_input_tokens_rejects_ambiguous_or_empty_measurements(usage):
    assert mod._strict_usage_input_tokens(usage) is None


def test_usage_total_recomputes_split_fields_and_rejects_incomplete_llama_cache_usage():
    assert mod._usage_total({"total_tokens": 100, "input_tokens": 100, "output_tokens": 20}) == 120
    assert mod._usage_total({"total_tokens": 100, "prompt_n": 100, "output_tokens": 0}) is None


@pytest.mark.parametrize("chat_id", ["local:socket", "channel:thread", "temporary:session"])
def test_chat_id_supported_rejects_core_non_saved_prefixes(chat_id):
    assert not mod._chat_id_supported(chat_id)


@pytest.mark.asyncio
async def test_usage_anchor_estimate_rebases_volatile_and_suffix_tokens(monkeypatch):
    prefix = {"role": "user", "content": "old"}
    current_system = {"role": "system", "content": "current system"}
    suffix = {"role": "assistant", "content": "answer"}
    body = {
        "model": "target",
        "messages": [current_system, prefix, suffix],
    }
    anchor = mod.UsageAnchor(
        assistant_message_id="assistant-1",
        input_tokens=100,
        stable_message_count=1,
        input_fingerprint=mod._compute_usage_anchor_input_fingerprint(body, [prefix]),
        volatile_message_tokens=12,
    )

    async def estimate_message_sum(messages, *, request):
        assert request is None
        if messages == [current_system]:
            return 15
        if messages == [suffix]:
            return 20
        raise AssertionError(f"unexpected token-estimate input: {messages!r}")

    monkeypatch.setattr(mod, "_estimate_message_token_sum_async", estimate_message_sum)

    assert await mod._estimate_body_tokens_from_usage_anchor(
        request=None,
        body=body,
        anchor=anchor,
    ) == 123


@pytest.mark.asyncio
@pytest.mark.parametrize("edited_index", [0, 1])
async def test_usage_anchor_estimate_rejects_edited_stable_prefix(monkeypatch, edited_index):
    stable = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    original_body = {"model": "target", "messages": stable}
    anchor = mod.UsageAnchor(
        assistant_message_id="assistant-1",
        input_tokens=100,
        stable_message_count=2,
        input_fingerprint=mod._compute_usage_anchor_input_fingerprint(original_body, stable),
        volatile_message_tokens=0,
    )
    edited = copy.deepcopy(stable)
    edited[edited_index]["content"] += " edited"

    async def unexpected_estimate(*args, **kwargs):
        raise AssertionError("fingerprint mismatch must be rejected before token estimation")

    monkeypatch.setattr(mod, "_estimate_message_token_sum_async", unexpected_estimate)

    assert await mod._estimate_body_tokens_from_usage_anchor(
        request=None,
        body={"model": "target", "messages": edited},
        anchor=anchor,
    ) is None


@pytest.mark.asyncio
async def test_usage_anchor_estimate_rejects_negative_measured_base(monkeypatch):
    prefix = {"role": "user", "content": "old"}
    body = {"model": "target", "messages": [prefix]}
    anchor = mod.UsageAnchor(
        assistant_message_id="assistant-1",
        input_tokens=11,
        stable_message_count=1,
        input_fingerprint=mod._compute_usage_anchor_input_fingerprint(body, [prefix]),
        volatile_message_tokens=12,
    )

    async def unexpected_estimate(*args, **kwargs):
        raise AssertionError("negative measured base must be rejected before token estimation")

    monkeypatch.setattr(mod, "_estimate_message_token_sum_async", unexpected_estimate)

    assert await mod._estimate_body_tokens_from_usage_anchor(
        request=None,
        body=body,
        anchor=anchor,
    ) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_change", ["shaping", "encoding", "estimator_version"])
async def test_usage_anchor_rejects_changed_provider_or_estimator_identity(monkeypatch, identity_change):
    message = {"role": "user", "content": "old"}
    body = {"model": "target", "messages": [message]}
    anchor = mod.UsageAnchor(
        assistant_message_id="assistant-1",
        input_tokens=100,
        stable_message_count=1,
        input_fingerprint=mod._compute_usage_anchor_input_fingerprint(
            body,
            [message],
            usage_anchor_shaping_hash="shape-a",
            encoding_name="encoding-a",
        ),
        volatile_message_tokens=0,
    )

    current_shaping_hash = "shape-b" if identity_change == "shaping" else "shape-a"
    current_encoding_name = "encoding-b" if identity_change == "encoding" else "encoding-a"
    monkeypatch.setattr(mod, "_get_tiktoken_encoder", lambda request=None: (object(), current_encoding_name))
    if identity_change == "estimator_version":
        monkeypatch.setattr(mod, "TOKEN_ESTIMATOR_VERSION", "future-estimator-version")

    async def unexpected_estimate(*args, **kwargs):
        raise AssertionError("identity mismatch must be rejected before token estimation")

    monkeypatch.setattr(mod, "_estimate_message_token_sum_async", unexpected_estimate)

    assert await mod._estimate_body_tokens_from_usage_anchor(
        request=None,
        body=body,
        anchor=anchor,
        usage_anchor_shaping_hash=current_shaping_hash,
    ) is None


@pytest.mark.asyncio
async def test_usage_anchor_resolves_encoder_off_the_event_loop(monkeypatch):
    event_loop_thread = threading.get_ident()
    encoder_threads = []

    def get_tiktoken_encoder(request=None):
        encoder_threads.append(threading.get_ident())
        return object(), "test-encoding"

    async def estimate_message_sum(messages, *, request):
        return 0

    monkeypatch.setattr(mod, "_get_tiktoken_encoder", get_tiktoken_encoder)
    monkeypatch.setattr(mod, "_estimate_message_token_sum_async", estimate_message_sum)

    body = {"model": "target", "messages": [{"role": "user", "content": "old"}]}
    anchor_input = await mod._build_usage_anchor_input(
        request=None,
        body=body,
        usage_anchor_shaping_hash="shape",
    )
    assert anchor_input is not None
    estimate = await mod._estimate_body_tokens_from_usage_anchor(
        request=None,
        body=body,
        anchor=mod.UsageAnchor(
            assistant_message_id="assistant-1",
            input_tokens=100,
            stable_message_count=anchor_input.stable_message_count,
            input_fingerprint=anchor_input.input_fingerprint,
            volatile_message_tokens=anchor_input.volatile_message_tokens,
        ),
        usage_anchor_shaping_hash="shape",
    )

    assert estimate == 100
    assert len(encoder_threads) == 2
    assert all(thread_id != event_loop_thread for thread_id in encoder_threads)


@pytest.mark.asyncio
async def test_usage_anchor_reuses_same_checkpoint_and_rejects_a_different_one(monkeypatch):
    system = {"role": "system", "content": "system"}
    checkpoint_a = {"role": "assistant", "content": "checkpoint A"}
    active = {"role": "user", "content": "active"}
    original = {"model": "target", "messages": [system, checkpoint_a, active]}
    anchor = mod.UsageAnchor(
        assistant_message_id="assistant-1",
        input_tokens=100,
        stable_message_count=2,
        input_fingerprint=mod._compute_usage_anchor_input_fingerprint(original, [checkpoint_a, active]),
        volatile_message_tokens=12,
    )
    suffix = [
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next"},
    ]

    async def estimate_message_sum(messages, *, request):
        if messages == [system]:
            return 15
        if messages == suffix:
            return 20
        raise AssertionError(f"unexpected token-estimate input: {messages!r}")

    monkeypatch.setattr(mod, "_estimate_message_token_sum_async", estimate_message_sum)

    same_checkpoint = {"model": "target", "messages": [system, checkpoint_a, active, *suffix]}
    assert await mod._estimate_body_tokens_from_usage_anchor(
        request=None,
        body=same_checkpoint,
        anchor=anchor,
    ) == 123

    different_checkpoint = copy.deepcopy(same_checkpoint)
    different_checkpoint["messages"][1]["content"] = "checkpoint B"
    assert await mod._estimate_body_tokens_from_usage_anchor(
        request=None,
        body=different_checkpoint,
        anchor=anchor,
    ) is None


def test_usage_anchor_fingerprint_round_trips_core_expanded_assistant_output():
    from open_webui.utils.middleware import process_messages_with_output

    raw_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "output": [
                {"type": "function_call", "call_id": "call-1", "name": "search", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": [{"type": "input_text", "text": "result"}],
                },
                {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
            ],
        },
    ]
    first = process_messages_with_output(copy.deepcopy(raw_messages))
    round_trip = process_messages_with_output(copy.deepcopy(raw_messages))
    body = {"model": "target", "messages": first}

    assert mod._compute_usage_anchor_input_fingerprint(body, first) == mod._compute_usage_anchor_input_fingerprint(
        {"model": "target", "messages": round_trip},
        round_trip,
    )

    edited = copy.deepcopy(raw_messages)
    edited[1]["output"][-1]["content"][0]["text"] = "edited answer"
    expanded_edit = process_messages_with_output(edited)
    assert mod._compute_usage_anchor_input_fingerprint(
        {"model": "target", "messages": expanded_edit},
        expanded_edit,
    ) != mod._compute_usage_anchor_input_fingerprint(body, first)


def test_usage_anchor_fingerprint_tracks_ollama_think_but_not_sampling_options():
    from open_webui.utils.payload import convert_payload_openai_to_ollama

    messages = [{"role": "user", "content": "question"}]
    enabled = {
        "model": "target",
        "messages": messages,
        "options": {"think": True, "temperature": 0.1},
    }
    disabled = copy.deepcopy(enabled)
    disabled["options"]["think"] = False
    different_temperature = copy.deepcopy(enabled)
    different_temperature["options"]["temperature"] = 0.9

    assert convert_payload_openai_to_ollama(copy.deepcopy(enabled))["think"] is True
    assert mod._body_token_extra_payload(enabled) == {"think": True}
    assert mod._body_token_extra_payload(disabled) == {"think": False}
    enabled_fingerprint = mod._compute_usage_anchor_input_fingerprint(enabled, messages)
    assert enabled_fingerprint != mod._compute_usage_anchor_input_fingerprint(disabled, messages)
    assert enabled_fingerprint == mod._compute_usage_anchor_input_fingerprint(
        different_temperature,
        messages,
    )


def test_reasoning_content_is_semantic_but_provider_reasoning_fields_are_token_only_identity():
    from open_webui.utils.payload import convert_messages_openai_to_ollama

    class LengthEncoder:
        def encode(self, text, **kwargs):
            return [0] * len(text)

    base = {"role": "assistant", "content": "answer"}
    with_reasoning = {**base, "reasoning_content": "reasoning"}
    with_details = {**base, "reasoning_details": [{"type": "signature", "data": "abc"}]}
    with_thinking = {**base, "thinking": "ollama reasoning"}

    assert mod.compute_source_hash([base]) != mod.compute_source_hash([with_reasoning])
    assert mod.compute_source_hash([base]) == mod.compute_source_hash([with_details])
    assert mod.compute_source_hash([base]) == mod.compute_source_hash([with_thinking])
    assert convert_messages_openai_to_ollama([with_thinking])[0]["thinking"] == "ollama reasoning"
    assert mod.estimate_message_tokens(
        with_thinking,
        encoder=LengthEncoder(),
        encoding_name="unit-test",
    ) > mod.estimate_message_tokens(base, encoder=LengthEncoder(), encoding_name="unit-test")
    base_fingerprint = mod._compute_usage_anchor_input_fingerprint(
        {"model": "target", "messages": [base]}, [base]
    )
    for message in (with_details, with_thinking):
        assert base_fingerprint != mod._compute_usage_anchor_input_fingerprint(
            {"model": "target", "messages": [message]},
            [message],
        )


def test_reasoning_token_projection_copies_only_changed_messages():
    affected = {
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
        "reasoning_details": [{"type": "encrypted", "data": "secret"}],
    }
    unchanged = {"role": "user", "content": [{"type": "text", "text": "question"}]}
    tools = [{"type": "function", "function": {"name": "search"}}]
    body = {"model": "target", "messages": [affected, unchanged], "tools": tools}

    projected = mod._project_usage_anchor_token_body(
        body,
        dropped_message_keys=frozenset({"reasoning_details"}),
    )

    assert projected is not body
    assert projected["messages"] is not body["messages"]
    assert projected["messages"][0] is not affected
    assert projected["messages"][0]["content"] is affected["content"]
    assert "reasoning_details" not in projected["messages"][0]
    assert "reasoning_details" in affected
    assert projected["messages"][1] is unchanged
    assert projected["tools"] is tools


@pytest.mark.asyncio
async def test_system_prompt_token_projection_matches_core_without_mutating_body(pipe_user):
    chat_system = {"role": "system", "content": "chat system"}
    user_message = {"role": "user", "content": "hello"}
    body = {
        "model": "target",
        "messages": [chat_system, user_message],
        "metadata": {"variables": {"{{CUSTOM}}": "expanded"}},
    }

    projected = await mod._project_system_prompt_for_token_estimate(
        body,
        user=pipe_user,
        system_prompt="target {{CUSTOM}}",
    )

    assert projected["messages"] == [
        {"role": "system", "content": "target expanded\nchat system"},
        user_message,
    ]
    assert projected["messages"][1] is user_message
    assert body["messages"] == [chat_system, user_message]


@pytest.mark.asyncio
async def test_reasoning_token_projection_matches_core_transports(monkeypatch, pipe_request):
    import open_webui.routers.ollama as ollama_router
    import open_webui.routers.openai as openai_router
    from open_webui.utils.payload import convert_payload_openai_to_ollama

    message = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "reasoning",
        "reasoning_details": [{"type": "encrypted", "data": "secret"}],
        "thinking": "thinking",
    }
    body = {"model": "target", "messages": [message]}

    async def responses_connection(index):
        assert index == 0
        return "http://provider", "key", {"api_type": "responses"}

    monkeypatch.setattr(openai_router, "get_openai_connection", responses_connection)
    pipe_request.app.state.OPENAI_MODELS = {"target": {"id": "target", "urlIdx": 0}}
    responses_transport, responses_dropped_keys = await mod._usage_anchor_transport_profile(
        pipe_request,
        {"target": {"id": "target", "owned_by": "openai", "urlIdx": 0}},
        "target",
    )
    projected = mod._project_usage_anchor_token_body(
        body,
        dropped_message_keys=responses_dropped_keys,
    )
    responses_payload = openai_router.convert_to_responses_payload(copy.deepcopy(body))

    assert responses_transport == {
        "owned_by": "openai",
        "url_idx": 0,
        "url": "http://provider",
        "api_type": "responses",
        "api_config": {"api_type": "responses"},
    }
    assert responses_dropped_keys == {"reasoning_content", "reasoning_details", "thinking"}
    assert not ({"reasoning_content", "reasoning_details", "thinking"} & projected["messages"][0].keys())
    assert all(key not in json.dumps(responses_payload) for key in responses_dropped_keys)
    assert mod._compute_usage_anchor_input_fingerprint(projected, projected["messages"]) == (
        mod._compute_usage_anchor_input_fingerprint(
            {"model": "target", "messages": [{"role": "assistant", "content": "answer"}]},
            [{"role": "assistant", "content": "answer"}],
        )
    )

    async def ollama_runtime_config():
        return (
            True,
            ["http://ollama-a", "http://ollama-b"],
            {
                "0": {"prefix_id": "a"},
                "1": {"prefix_id": "b"},
            },
        )

    monkeypatch.setattr(ollama_router, "get_ollama_runtime_config", ollama_runtime_config)
    pipe_request.app.state.OLLAMA_MODELS = {
        "target": {"model": "target", "digest": "sha256:digest-a", "urls": [1]}
    }
    ollama_transport, ollama_dropped_keys = await mod._usage_anchor_transport_profile(
        pipe_request,
        {"target": {"id": "target", "owned_by": "ollama", "ollama": {"urls": [1]}}},
        "target",
    )
    ollama_projected = mod._project_usage_anchor_token_body(body, dropped_message_keys=ollama_dropped_keys)
    ollama_payload = convert_payload_openai_to_ollama(copy.deepcopy(body))
    assert ollama_transport == {
        "owned_by": "ollama",
        "digest": "sha256:digest-a",
        "backends": [
            {
                "url_idx": 1,
                "url": "http://ollama-b",
                "api_config": {"prefix_id": "b"},
            }
        ],
    }
    assert ollama_dropped_keys == {"reasoning_content", "reasoning_details"}
    assert "reasoning_details" not in ollama_payload["messages"][0]
    assert "reasoning_content" not in ollama_payload["messages"][0]
    assert ollama_projected["messages"][0]["thinking"] == ollama_payload["messages"][0]["thinking"]

    pipe_request.app.state.OLLAMA_MODELS = {
        "target": {"model": "target", "digest": "sha256:digest-a", "urls": [0, 1]}
    }
    multi_transport, multi_dropped_keys = await mod._usage_anchor_transport_profile(
        pipe_request,
        {"target": {"id": "target", "owned_by": "ollama", "ollama": {"urls": [0, 1]}}},
        "target",
    )
    assert multi_transport == {
        "owned_by": "ollama",
        "digest": "sha256:digest-a",
        "backends": [
            {
                "url_idx": 0,
                "url": "http://ollama-a",
                "api_config": {"prefix_id": "a"},
            },
            {
                "url_idx": 1,
                "url": "http://ollama-b",
                "api_config": {"prefix_id": "b"},
            },
        ],
    }
    assert multi_dropped_keys == ollama_dropped_keys

    pipe_request.app.state.OLLAMA_MODELS["target"]["digest"] = "sha256:digest-b"
    changed_transport, _ = await mod._usage_anchor_transport_profile(
        pipe_request,
        {"target": {"id": "target", "owned_by": "ollama", "ollama": {"urls": [0, 1]}}},
        "target",
    )
    assert changed_transport is not None
    assert changed_transport["digest"] == "sha256:digest-b"
    assert changed_transport != multi_transport

    pipe_request.app.state.OLLAMA_MODELS = {"target": {"model": "target", "urls": [0]}}
    missing_digest_transport, missing_digest_dropped_keys = await mod._usage_anchor_transport_profile(
        pipe_request,
        {"target": {"id": "target", "owned_by": "ollama", "ollama": {"urls": [0]}}},
        "target",
    )
    assert missing_digest_transport is None
    assert missing_digest_dropped_keys == ollama_dropped_keys

    pipe_request.app.state.OLLAMA_MODELS = {
        "target": {"model": "target", "digest": "sha256:digest-b", "urls": [2]}
    }
    unresolved_transport, unresolved_dropped_keys = await mod._usage_anchor_transport_profile(
        pipe_request,
        {"target": {"id": "target", "owned_by": "ollama", "ollama": {"urls": [2]}}},
        "target",
    )
    assert unresolved_transport is None
    assert unresolved_dropped_keys == ollama_dropped_keys

    async def chat_completions_connection(index):
        return "http://provider", "key", {}

    monkeypatch.setattr(openai_router, "get_openai_connection", chat_completions_connection)
    chat_transport, chat_dropped_keys = await mod._usage_anchor_transport_profile(
        pipe_request,
        {"target": {"id": "target", "owned_by": "openai", "urlIdx": 0}},
        "target",
    )
    assert chat_transport == {
        "owned_by": "openai",
        "url_idx": 0,
        "url": "http://provider",
        "api_type": "chat_completions",
        "api_config": {},
    }
    assert chat_dropped_keys == frozenset()


@pytest.mark.asyncio
async def test_usage_anchor_transport_profile_supports_v096_runtime_config(monkeypatch, pipe_request):
    import open_webui.routers.ollama as ollama_router
    import open_webui.routers.openai as openai_router

    monkeypatch.delattr(openai_router, "get_openai_connection")
    monkeypatch.delattr(ollama_router, "get_ollama_runtime_config")
    pipe_request.app.state.config = SimpleNamespace(
        OPENAI_API_BASE_URLS=["http://openai-a", "http://openai-b"],
        OPENAI_API_CONFIGS={"1": {"api_type": "responses"}},
        OLLAMA_BASE_URLS=["http://ollama-a"],
        OLLAMA_API_CONFIGS={"0": {"prefix_id": "local"}},
    )
    pipe_request.app.state.OPENAI_MODELS = {"openai-target": {"id": "openai-target", "urlIdx": 1}}
    pipe_request.app.state.OLLAMA_MODELS = {
        "ollama-target": {
            "model": "ollama-target",
            "digest": "sha256:legacy",
            "urls": [0],
        }
    }

    openai_transport, openai_dropped_keys = await mod._usage_anchor_transport_profile(
        pipe_request,
        {"openai-target": {"id": "openai-target", "owned_by": "openai", "urlIdx": 1}},
        "openai-target",
    )
    ollama_transport, ollama_dropped_keys = await mod._usage_anchor_transport_profile(
        pipe_request,
        {"ollama-target": {"id": "ollama-target", "owned_by": "ollama"}},
        "ollama-target",
    )

    assert openai_transport == {
        "owned_by": "openai",
        "url_idx": 1,
        "url": "http://openai-b",
        "api_type": "responses",
        "api_config": {"api_type": "responses"},
    }
    assert openai_dropped_keys == {"reasoning_content", "reasoning_details", "thinking"}
    assert ollama_transport == {
        "owned_by": "ollama",
        "digest": "sha256:legacy",
        "backends": [
            {
                "url_idx": 0,
                "url": "http://ollama-a",
                "api_config": {"prefix_id": "local"},
            }
        ],
    }
    assert ollama_dropped_keys == {"reasoning_content", "reasoning_details"}


def test_function_wrapper_core_round_trip_drops_target_reasoning_text():
    from open_webui.utils.middleware import get_reasoning_format, process_messages_with_output

    wrapper_model = {
        "id": "auto_compact.target",
        "owned_by": "openai",
        "pipe": {"type": "pipe"},
    }
    persisted = [
        {
            "role": "assistant",
            "content": "answer",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "output_text", "text": "private reasoning"}],
                    "reasoning_details": [{"type": "signature", "data": "abc"}],
                },
                {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
            ],
        }
    ]

    reasoning_format = get_reasoning_format(wrapper_model)
    restored = process_messages_with_output(copy.deepcopy(persisted), reasoning_format=reasoning_format)

    assert reasoning_format is None
    assert restored[0].get("reasoning_content") is None
    assert mod.compute_source_hash(restored) == mod.compute_source_hash(
        [{"role": "assistant", "content": "answer"}]
    )


@pytest.mark.asyncio
async def test_persist_usage_anchor_saves_only_strict_final_input_measurement(monkeypatch):
    rows = []

    class Store:
        async def upsert_ready(self, row):
            rows.append(row)

    async def initialize(**kwargs):
        return None

    monkeypatch.setattr(mod, "CheckpointStore", Store)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize)
    anchor_input = mod.UsageAnchorInput(
        stable_message_count=2,
        input_fingerprint="fingerprint",
        volatile_message_tokens=10,
    )

    assert await mod.persist_usage_anchor(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        assistant_message_id="assistant-1",
        anchor_input=anchor_input,
        raw_usage={
            "input_tokens": 60,
            "cache_creation_input_tokens": 15,
            "cache_read_input_tokens": 25,
        },
    )
    assert mod.usage_anchor_from_row(rows[0]).input_tokens == 100

    assert not await mod.persist_usage_anchor(
        request=None,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        assistant_message_id="assistant-1",
        anchor_input=anchor_input,
        raw_usage={"prompt_n": 100},
    )
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_usage_anchor_upsert_recovers_from_insert_race():
    class FakeDb:
        def __init__(self):
            self.execute_count = 0
            self.commits = 0
            self.rollbacks = 0

        async def execute(self, statement):
            self.execute_count += 1
            if self.execute_count == 1:
                return SimpleNamespace(rowcount=0)
            if self.execute_count == 2:
                raise mod.IntegrityError(None, None, RuntimeError("duplicate"))
            return SimpleNamespace(rowcount=1)

        async def commit(self):
            self.commits += 1

        async def rollback(self):
            self.rollbacks += 1

    db = FakeDb()
    row = mod.build_usage_anchor_row(
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        assistant_message_id="assistant-1",
        input_tokens=100,
        anchor_input=mod.UsageAnchorInput(
            stable_message_count=2,
            input_fingerprint="fingerprint",
            volatile_message_tokens=10,
        ),
    )

    await mod.CheckpointStore(db=db).upsert_ready(row)

    assert (db.execute_count, db.rollbacks, db.commits) == (3, 2, 1)


def test_request_scoped_usage_returns_current_tool_loop_measurement():
    request = SimpleNamespace(state=SimpleNamespace())
    anchor_input = mod.UsageAnchorInput(
        stable_message_count=2,
        input_fingerprint="fingerprint",
        volatile_message_tokens=10,
    )
    mod.store_request_scoped_usage(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
        usage={"total_tokens": 999, "input_tokens": 900, "output_tokens": 99},
        anchor_input=anchor_input,
    )

    usage = mod.get_request_scoped_usage(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    assert usage["total_tokens"] == 999
    assert mod.get_request_scoped_usage_anchor(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    ) == mod.UsageAnchor(
        assistant_message_id="request",
        input_tokens=900,
        stable_message_count=2,
        input_fingerprint="fingerprint",
        volatile_message_tokens=10,
    )

    mod.store_request_scoped_usage(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
        usage={"input_tokens": 901},
    )
    assert mod.get_request_scoped_usage_anchor(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    ) is None


def test_usage_option_injection_preserves_existing_stream_options():
    body = {
        "stream": True,
        "stream_options": {"other": "value"},
    }

    out = mod.inject_stream_usage_options(body)

    assert out is not body
    assert out["stream_options"] == {"other": "value", "include_usage": True}


def test_context_window_classifier_handles_structured_responses_and_http_exception():
    assert mod.is_retryable_context_error(
        JSONResponse(
            {"error": {"code": "context_length_exceeded", "message": "too long"}},
            status_code=400,
        )
    )
    assert mod.is_retryable_context_error(
        PlainTextResponse("maximum context length exceeded", status_code=413)
    )
    assert mod.is_retryable_context_error(
        HTTPException(status_code=400, detail={"error": {"type": "context_window_exceeded"}})
    )
    assert mod.is_retryable_context_error(Exception("maximum context length exceeded"))


def test_context_window_classifier_avoids_rate_limit_quota_and_tpm_false_positives():
    assert not mod.is_retryable_context_error(
        JSONResponse(
            {"error": {"message": "too many tokens per minute for this organization"}},
            status_code=429,
        )
    )
    assert not mod.is_retryable_context_error(
        {"error": {"code": "insufficient_quota", "message": "quota exceeded"}},
    )
    assert not mod.is_retryable_context_error(
        {"error": {"message": "token-per-minute limit exceeded"}},
    )
    assert not mod.is_retryable_context_error(
        {"error": {"code": "token_limit_exceeded", "message": "quota exceeded"}},
        status_code=400,
    )
    assert not mod.is_retryable_context_error(
        {"error": {"code": "context_length", "message": "rate limit exceeded"}},
        status_code=400,
    )


def test_context_window_message_fallback_uses_only_error_message_fields():
    body_with_unrelated_serialized_text = {
        "metadata": {"raw": "maximum context length exceeded"},
        "error": {"message": "ordinary bad request"},
    }
    body_with_error_message = {"error": {"message": "input is too long for the context window"}}

    assert not mod.is_retryable_context_error(body_with_unrelated_serialized_text, status_code=400)
    assert mod.is_retryable_context_error(body_with_error_message, status_code=400)


def test_context_window_classifier_handles_proxy_and_local_model_variants():
    assert mod.is_retryable_context_error(
        {
            "error": {
                "type": "invalid_request_error",
                "message": "input length and max_tokens exceed context limit: 200000 + 4096 > 200000",
            }
        },
        status_code=400,
    )
    assert mod.is_retryable_context_error(
        {"error": {"message": "This model supports at most 8192 tokens, but input tokens are 9000"}},
        status_code=400,
    )
    assert mod.is_retryable_context_error(
        {"error": {"message": "The prompt tokens exceed the configured token limit"}},
        status_code=422,
    )
    assert not mod.is_retryable_context_error(
        {"error": {"message": "rate_limit_exceeded: too many requests"}},
        status_code=400,
    )


def test_context_window_classifier_rejects_input_length_validation_errors():
    assert not mod.is_retryable_context_error(
        {"error": {"message": "input length must be at least 1"}},
        status_code=400,
    )
    assert not mod.is_retryable_context_error(
        {"error": {"message": "invalid input length"}},
        status_code=422,
    )
    assert not mod.is_retryable_context_error(
        {"error": {"message": "input length must be less than max_batch_size"}},
        status_code=400,
    )
    assert not mod.is_retryable_context_error(
        {"error": {"message": "max_tokens must be less than the configured token limit"}},
        status_code=400,
    )
    assert not mod.is_retryable_context_error(
        {
            "error": {
                "code": "token_limit_exceeded",
                "message": "max_tokens must be less than the configured token limit",
            }
        },
        status_code=400,
    )
    assert not mod.is_retryable_context_error(
        {"error": {"message": "max_completion_tokens exceeds the maximum number of tokens"}},
        status_code=422,
    )


def test_sse_json_parser_preserves_unicode_line_separator():
    payload = {"choices": [{"delta": {"content": "before\u2028after"}}]}
    chunk = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    assert mod.extract_sse_json_events(chunk) == [payload]


def test_sse_parser_swallows_split_crlf_across_empty_chunk():
    parser = mod._SSEDataParser()

    assert parser.feed("data: first\r")[0] == []
    assert parser.feed("")[0] == []
    assert parser.feed("\ndata: second\r\n\r\n")[0] == ["first\nsecond"]


def test_sse_parser_accepts_bare_cr_immediately():
    parser = mod._SSEDataParser()

    assert parser.feed("data: first\rdata: second\r\r")[0] == ["first\nsecond"]


@pytest.mark.parametrize("tail", ["data: final", "data: final\r"])
def test_sse_parser_flushes_incomplete_final_event_once(tail):
    parser = mod._SSEDataParser()

    assert parser.feed(tail)[0] == []
    assert parser.flush()[0] == ["final"]
    assert parser.flush()[0] == []


def test_sse_error_payload_is_classified_before_any_content():
    chunk = b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'

    assert mod.extract_sse_json_events(chunk) == [
        {"error": {"code": "context_length_exceeded", "message": "too long"}}
    ]
    assert mod.first_chunk_is_retryable_context_error(chunk)


def test_responses_failed_payload_is_classified_before_any_content():
    payload = {
        "type": "response.failed",
        "response": {
            "error": {
                "code": "context_length_exceeded",
                "message": "too long",
            }
        },
    }
    chunk = f"data: {json.dumps(payload)}\n\n".encode()

    assert mod.extract_sse_json_events(chunk) == [payload]
    assert mod.first_chunk_is_retryable_context_error(chunk)


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_split_initial_sse_context_error():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b'data: {"error": {"code": "context_length_'
            yield b'exceeded", "message": "too long"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_waits_for_split_sse_field_name_before_retrying_context_error():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b"da"
            yield b'ta: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_waits_for_split_sse_payload_before_retrying_context_error():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b"data: "
            yield b'{"error": {"code": "context_length_'
            yield b'exceeded", "message": "too long"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_ignores_empty_sse_data_before_retrying_context_error():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b"data:\n\n"
            yield b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_ignores_bare_empty_sse_data_before_retrying_context_error():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b"data\n\n"
            yield b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ignored_field", [b"ping: 1\n\n", b"x-trace: proxy\n\n"])
async def test_streaming_forwarder_ignores_unknown_sse_fields_before_retrying_context_error(ignored_field):
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield ignored_field
            yield b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_treats_sse_content_type_case_insensitively():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="Text/Event-Stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_initial_responses_failed_context_error():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            payload = {
                "type": "response.failed",
                "response": {
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "too long",
                    }
                },
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_responses_failed_after_pre_output_events():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b'data: {"type": "response.created", "response": {"id": "resp-1"}}\n\n'
            yield b'data: {"type": "response.in_progress", "response": {"id": "resp-1"}}\n\n'
            payload = {
                "type": "response.failed",
                "response": {
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "too long",
                    }
                },
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_context_error_after_chat_role_delta():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
            yield b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_context_error_after_same_chunk_chat_role_delta():
    closed = False
    background_count = 0

    async def chunks():
        nonlocal closed
        try:
            yield (
                b'data: {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}\n\n'
                b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
            )
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )

    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type",
    [
        "response.refusal.delta",
        "response.reasoning_summary_text.delta",
        "response.mcp_call_arguments.delta",
        "response.custom_tool_call_input.delta",
    ],
)
async def test_streaming_forwarder_treats_responses_output_deltas_as_visible_output(event_type):
    requested_second_chunk = False
    closed = False
    background_count = 0
    first_chunk = f'data: {{"type": "{event_type}", "delta": "visible"}}\n\n'.encode()

    async def chunks():
        nonlocal requested_second_chunk, closed
        try:
            yield first_chunk
            requested_second_chunk = True
            yield b'data: {"type": "response.completed", "response": {"id": "resp-1"}}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    assert requested_second_chunk is False
    iterator = prepared.body_iterator.__aiter__()
    assert await iterator.__anext__() == first_chunk
    await iterator.aclose()
    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_chunk", "media_type"),
    [
        (b"plain text token", "text/plain"),
        (b'data: "plain text token"\n\n', "text/event-stream"),
    ],
)
async def test_streaming_forwarder_starts_unparsed_stream_after_first_chunk(first_chunk, media_type):
    requested_second_chunk = False
    closed = False
    background_count = 0

    async def chunks():
        nonlocal requested_second_chunk, closed
        try:
            yield first_chunk
            requested_second_chunk = True
            await asyncio.sleep(60)
            yield b"late token"
        finally:
            closed = True

    async def background():
        nonlocal background_count
        background_count += 1

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type=media_type, background=BackgroundTask(background))

    prepared = await asyncio.wait_for(
        mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        ),
        timeout=0.2,
    )

    assert requested_second_chunk is False
    iterator = prepared.body_iterator.__aiter__()
    assert await iterator.__anext__() == first_chunk
    await iterator.aclose()
    assert closed is True
    assert background_count == 1


@pytest.mark.asyncio
async def test_streaming_forwarder_surfaces_context_error_after_same_chunk_non_error_event():
    chunk = (
        b'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 0}}\n\n'
        b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
    )

    async def chunks():
        yield chunk

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for emitted_chunk in prepared.body_iterator:
        emitted.append(emitted_chunk)

    assert emitted == [chunk]


@pytest.mark.asyncio
async def test_streaming_forwarder_surfaces_context_error_after_same_chunk_scalar_sse_output():
    chunk = (
        b'data: "hello"\n\n'
        b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
    )

    async def chunks():
        yield chunk

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for emitted_chunk in prepared.body_iterator:
        emitted.append(emitted_chunk)

    assert emitted == [chunk]


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_context_error_before_same_chunk_scalar_sse_output():
    chunk = (
        b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'
        b'data: "hello"\n\n'
    )

    async def chunks():
        yield chunk

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
        )


@pytest.mark.asyncio
async def test_streaming_forwarder_surfaces_context_error_after_first_non_error_event():
    async def chunks():
        yield b'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 0}}\n\n'
        yield b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk)

    assert emitted == [
        b'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 0}}\n\n',
        b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n',
    ]


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_iterator_context_exception_before_visible_output():
    closed = False
    background_ran = False
    restored = False

    async def chunks():
        nonlocal closed
        try:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "context_length_exceeded", "message": "too long"}},
            )
            yield b""
        finally:
            closed = True

    async def background():
        nonlocal background_ran
        background_ran = True

    def restore():
        nonlocal restored
        restored = True

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(
        chunks(),
        media_type="text/event-stream",
        background=BackgroundTask(background),
    )

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
            restore=restore,
        )

    assert restored is True
    assert closed is True
    assert background_ran is True


@pytest.mark.asyncio
async def test_streaming_forwarder_retries_non_sse_context_exception_after_empty_chunk():
    closed = False
    background_ran = False
    restored = False

    async def chunks():
        nonlocal closed
        try:
            yield b""
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "context_length_exceeded", "message": "too long"}},
            )
            yield b""
        finally:
            closed = True

    async def background():
        nonlocal background_ran
        background_ran = True

    def restore():
        nonlocal restored
        restored = True

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(
        chunks(),
        media_type="text/plain",
        background=BackgroundTask(background),
    )

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.prepare_streaming_response(
            response,
            request=request,
            chat_id="chat-1",
            message_id="message-1",
            wrapper_model_id="auto_compact.target",
            restore=restore,
        )

    assert restored is True
    assert closed is True
    assert background_ran is True


@pytest.mark.asyncio
async def test_non_context_json_response_is_returned_as_sse_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = JSONResponse({"error": {"code": "provider_error", "message": "unavailable"}}, status_code=500)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error":{"code":"provider_error","message":"unavailable"}}\n\n']


def test_sse_data_chunk_preserves_single_event_multiline_payloads():
    assert mod._sse_data_chunk("line1") == "data: line1\n\n"
    assert mod._sse_data_chunk("line1\nline2") == "data: line1\ndata: line2\n\n"
    assert mod._sse_data_chunk("line1\rline2\r\nline3") == "data: line1\ndata: line2\ndata: line3\n\n"
    assert mod._sse_data_chunk("line1\n") == "data: line1\n\n"
    assert mod._sse_data_chunk("line1\u2028line2\u2029line3\x85line4") == (
        "data: line1\u2028line2\u2029line3\x85line4\n\n"
    )


@pytest.mark.asyncio
async def test_non_context_json_detail_response_is_returned_as_sse_error_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = JSONResponse({"detail": "provider unavailable"}, status_code=500)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error": {"code": "provider_error", "message": "provider unavailable"}}\n\n']


@pytest.mark.asyncio
async def test_non_context_json_message_response_is_returned_as_sse_error_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = JSONResponse({"message": "rate limited"}, status_code=429)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error": {"code": "provider_error", "message": "rate limited"}}\n\n']


@pytest.mark.asyncio
async def test_non_context_json_string_error_response_is_returned_as_sse_error_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = JSONResponse({"error": "rate limited"}, status_code=429)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error": {"code": "provider_error", "message": "rate limited"}}\n\n']


@pytest.mark.asyncio
async def test_non_context_plain_text_response_is_returned_as_sse_error_event():
    request = SimpleNamespace(state=SimpleNamespace())
    response = PlainTextResponse("provider unavailable", status_code=500)

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == ['data: {"error": {"code": "provider_error", "message": "provider unavailable"}}\n\n']


@pytest.mark.asyncio
async def test_streaming_forwarder_normalizes_dict_chunks_to_sse_events():
    async def chunks():
        yield {"usage": {"prompt_tokens": 2, "completion_tokens": 0}}
        yield {"choices": [{"delta": {"content": "hello"}}]}

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    assert emitted == [
        'data: {"usage": {"prompt_tokens": 2, "completion_tokens": 0}}\n\n',
        'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n',
    ]


@pytest.mark.asyncio
async def test_streaming_forwarder_surfaces_context_error_after_visible_output():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
        yield b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n'

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    emitted = []
    async for chunk in prepared.body_iterator:
        emitted.append(chunk)

    assert emitted == [
        b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n',
        b'data: {"error": {"code": "context_length_exceeded", "message": "too long"}}\n\n',
    ]


@pytest.mark.asyncio
async def test_streaming_forwarder_closes_inner_stream_after_visible_output_early_stop():
    closed = False
    background_ran = False

    async def chunks():
        nonlocal closed
        try:
            yield b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
            await asyncio.sleep(60)
            yield b'data: {"choices": [{"delta": {"content": "late"}}]}\n\n'
        finally:
            closed = True

    async def background():
        nonlocal background_ran
        background_ran = True

    request = SimpleNamespace(state=SimpleNamespace())
    response = StreamingResponse(
        chunks(),
        media_type="text/event-stream",
        background=BackgroundTask(background),
    )

    prepared = await mod.prepare_streaming_response(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )

    iterator = prepared.body_iterator.__aiter__()
    first = await iterator.__anext__()
    assert first == b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'

    await iterator.aclose()

    assert closed is True
    assert background_ran is True


def test_request_state_proxy_isolates_bypass_and_metadata():
    request = SimpleNamespace(
        state=SimpleNamespace(
            bypass_filter=False,
            bypass_system_prompt=True,
            metadata={"selected_model_id": "arena-a", "chat_id": "chat-1"},
        )
    )

    proxy = mod.RequestStateProxy(
        request,
        bypass_filter=True,
        bypass_system_prompt=False,
        metadata={"task": mod.INTERNAL_SUMMARY_TASK},
    )
    proxy.state.metadata["new"] = "value"
    proxy.state.bypass_system_prompt = False

    assert request.state.bypass_filter is False
    assert request.state.bypass_system_prompt is True
    assert request.state.metadata == {"selected_model_id": "arena-a", "chat_id": "chat-1"}
    assert proxy.state.bypass_filter is True
    assert proxy.state.bypass_system_prompt is False
    assert proxy.state.metadata == {"task": mod.INTERNAL_SUMMARY_TASK, "new": "value"}


def test_request_state_proxy_does_not_add_flags_to_original_request():
    request = SimpleNamespace(state=SimpleNamespace())

    proxy = mod.RequestStateProxy(request, bypass_filter=True, bypass_system_prompt=True)

    assert not hasattr(request.state, "bypass_filter")
    assert not hasattr(request.state, "bypass_system_prompt")
    assert proxy.state.bypass_filter is True
    assert proxy.state.bypass_system_prompt is True


@pytest.mark.asyncio
async def test_request_state_proxy_handles_unpickleable_metadata_without_mutating_original():
    future = asyncio.get_running_loop().create_future()
    request = SimpleNamespace(
        state=SimpleNamespace(
            metadata={
                "chat_id": "chat-1",
                "tools": {"tool": {"future": future}},
            }
        )
    )

    proxy = mod.RequestStateProxy(request, bypass_filter=True)

    proxy.state.metadata["chat_id"] = "summary-chat"
    proxy.state.metadata["tools"]["tool"]["name"] = "Tool"

    assert request.state.metadata["chat_id"] == "chat-1"
    assert "name" not in request.state.metadata["tools"]["tool"]
    assert proxy.state.metadata["tools"]["tool"]["future"] is future


def test_summary_task_metadata_removes_tool_and_arena_state():
    metadata = {
        "chat_id": "chat-1",
        "selected_model_id": "arena-a",
        "tools": {"tool": {}},
        "tool_ids": ["tool"],
    }

    summary_metadata = mod.build_summary_task_metadata(metadata)

    assert summary_metadata["chat_id"] == "chat-1"
    assert summary_metadata["task"] == mod.INTERNAL_SUMMARY_TASK
    assert summary_metadata["tools"] == {}
    assert summary_metadata["tool_ids"] == ["tool"]
    assert "selected_model_id" not in summary_metadata


def test_build_summary_completion_body_appends_prefix_file_context_to_user_message_and_drops_files():
    body = mod.build_summary_completion_body(
        {
            "model": "target",
            "messages": [{"role": "user", "content": "old"}],
            "metadata": {"files": [_file("source")]},
            "previous_response_id": "response-1",
            "response_format": {"type": "json_object"},
        },
        summary_model_id="summary",
        source_messages=[{"role": "user", "content": "source"}],
        metadata={"chat_id": "chat-1", "files": [_file("source")]},
        prefix_file_context="<attached_file_contents>source context</attached_file_contents>",
    )

    assert body["messages"][:-1] == [{"role": "user", "content": "source"}]
    assert body["messages"][-1]["role"] == "user"
    assert body["messages"][-1]["content"].startswith(
        "<attached_file_contents>source context</attached_file_contents>\n\n"
    )
    assert "AUTO-COMPACTION CHECKPOINT SUMMARY" in body["messages"][-1]["content"]
    assert "files" not in body["metadata"]
    assert "previous_response_id" not in body
    assert "response_format" not in body


@pytest.mark.asyncio
async def test_prepare_summary_file_context_fails_closed_when_prefix_file_context_is_unavailable(
    monkeypatch, pipe_request, pipe_user
):
    async def load_chat_message_chain(request, chat_id, current_message_id):
        assert chat_id == "chat-1"
        assert current_message_id == "message-1"
        return [
            {"role": "user", "content": "old", "files": [_file("absorbed-file")]},
            {"role": "user", "content": "current"},
        ]

    async def generate_summary_file_context(*, request, user, prefix_files):
        assert prefix_files == [_file("absorbed-file")]
        raise mod.SummaryFileContextUnavailable("summary file context unavailable")

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "_generate_summary_file_context", generate_summary_file_context)

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._prepare_summary_file_context(
            request=pipe_request,
            user=pipe_user,
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("absorbed-file")],
            },
            compaction_prefix_count=1,
            parent_source_message_count=0,
            file_context_enabled=True,
        )


@pytest.mark.asyncio
async def test_prepare_summary_file_context_fails_closed_when_db_chain_is_unavailable(
    monkeypatch, pipe_request, pipe_user
):
    async def load_chat_message_chain(request, chat_id, current_message_id):
        assert chat_id == "chat-1"
        assert current_message_id == "message-1"
        return None

    async def generate_summary_file_context(*, request, user, prefix_files):
        raise AssertionError("summary file context must not be generated without DB chain")

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "_generate_summary_file_context", generate_summary_file_context)

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._prepare_summary_file_context(
            request=pipe_request,
            user=pipe_user,
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("maybe-absorbed-file")],
            },
            compaction_prefix_count=1,
            parent_source_message_count=0,
            file_context_enabled=True,
        )


@pytest.mark.asyncio
async def test_prepare_summary_file_context_fails_closed_for_current_file_when_db_chain_is_unavailable(
    monkeypatch, pipe_request, pipe_user
):
    async def load_chat_message_chain(request, chat_id, current_message_id):
        return None

    async def generate_summary_file_context(*, request, user, prefix_files):
        raise AssertionError("current-only files must not be summarized")

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "_generate_summary_file_context", generate_summary_file_context)

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._prepare_summary_file_context(
            request=pipe_request,
            user=pipe_user,
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("current-file")],
                "user_message": {"files": [_file("current-file")]},
            },
            compaction_prefix_count=1,
            parent_source_message_count=0,
            file_context_enabled=True,
        )


@pytest.mark.asyncio
async def test_prepare_summary_file_context_fails_when_current_file_may_be_stripped_from_prefix_without_db_chain(
    monkeypatch, pipe_request, pipe_user
):
    async def load_chat_message_chain(request, chat_id, current_message_id):
        return None

    async def generate_summary_file_context(*, request, user, prefix_files):
        raise AssertionError("summary file context must not be generated without DB chain")

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "_generate_summary_file_context", generate_summary_file_context)

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._prepare_summary_file_context(
            request=pipe_request,
            user=pipe_user,
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("shared-file")],
                "user_message": {"files": [_file("shared-file")]},
            },
            compaction_prefix_count=1,
            parent_source_message_count=0,
            file_context_enabled=True,
        )


@pytest.mark.asyncio
async def test_prepare_summary_file_context_includes_current_file_when_it_is_in_prefix(
    monkeypatch, pipe_request, pipe_user
):
    async def load_chat_message_chain(request, chat_id, current_message_id):
        assert chat_id == "chat-1"
        assert current_message_id == "message-1"
        return [
            {"role": "user", "content": "active with file", "files": [_file("current-file")]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-2", "type": "function"}]},
            {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
        ]

    async def generate_summary_file_context(*, request, user, prefix_files):
        assert prefix_files == [_file("current-file")]
        return "current file context"

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "_generate_summary_file_context", generate_summary_file_context)

    context = await mod._prepare_summary_file_context(
        request=pipe_request,
        user=pipe_user,
        metadata={
            "chat_id": "chat-1",
            "user_message_id": "message-1",
            "files": [_file("current-file")],
            "user_message": {"files": [_file("current-file")]},
        },
        compaction_prefix_count=3,
        parent_source_message_count=0,
        file_context_enabled=True,
    )

    assert context == "current file context"


@pytest.mark.asyncio
async def test_generate_summary_file_context_accepts_core_best_effort_partial_sources(
    monkeypatch, pipe_request, pipe_user
):
    install_fake_open_webui_user_model(monkeypatch)

    async def get_sources_from_items(**kwargs):
        assert [file["id"] for file in kwargs["items"]] == ["file-a", "file-b"]
        return [
            {
                "source": {"id": "file-a", "name": "file-a.txt"},
                "document": ["context for file-a"],
                "metadata": [{"source": "file-a"}],
            }
        ]

    retrieval_module = types.ModuleType("open_webui.retrieval.utils")
    setattr(retrieval_module, "get_sources_from_items", get_sources_from_items)
    monkeypatch.setitem(sys.modules, "open_webui.retrieval.utils", retrieval_module)

    context = await mod._generate_summary_file_context(
        request=pipe_request,
        user=pipe_user,
        prefix_files=[_file("file-a"), _file("file-b")],
    )

    assert context is not None
    assert "context for file-a" in context
    assert "file-b" not in context


@pytest.mark.asyncio
async def test_generate_summary_file_context_uses_db_config_rag_values(
    monkeypatch, pipe_request, pipe_user
):
    install_fake_open_webui_user_model(monkeypatch)
    captured = {}

    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            captured["config_keys"] = keys
            return {
                "rag.top_k": 7,
                "rag.top_k_reranker": 8,
                "rag.relevance_threshold": 0.42,
                "rag.hybrid_bm25_weight": 0.73,
                "rag.enable_hybrid_search": True,
            }

    async def get_sources_from_items(**kwargs):
        captured["rag_kwargs"] = {
            "k": kwargs["k"],
            "k_reranker": kwargs["k_reranker"],
            "r": kwargs["r"],
            "hybrid_bm25_weight": kwargs["hybrid_bm25_weight"],
            "hybrid_search": kwargs["hybrid_search"],
            "full_context": kwargs["full_context"],
        }
        return [
            {
                "source": {"id": "file-a", "name": "file-a.txt"},
                "document": ["db rag context"],
                "metadata": [{"source": "file-a"}],
            }
        ]

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    retrieval_module = types.ModuleType("open_webui.retrieval.utils")
    retrieval_module.get_sources_from_items = get_sources_from_items
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "open_webui.retrieval.utils", retrieval_module)
    pipe_request.app.state.config = SimpleNamespace(
        TOP_K=1,
        TOP_K_RERANKER=2,
        RELEVANCE_THRESHOLD=0.1,
        HYBRID_BM25_WEIGHT=0.2,
        ENABLE_RAG_HYBRID_SEARCH=False,
    )

    context = await mod._generate_summary_file_context(
        request=pipe_request,
        user=pipe_user,
        prefix_files=[_file("file-a")],
    )

    assert captured["config_keys"] == (
        "rag.top_k",
        "rag.top_k_reranker",
        "rag.relevance_threshold",
        "rag.hybrid_bm25_weight",
        "rag.enable_hybrid_search",
    )
    assert captured["rag_kwargs"] == {
        "k": 7,
        "k_reranker": 8,
        "r": 0.42,
        "hybrid_bm25_weight": 0.73,
        "hybrid_search": True,
        "full_context": True,
    }
    assert context is not None
    assert "db rag context" in context


@pytest.mark.asyncio
async def test_generate_summary_file_context_falls_back_to_legacy_rag_config(
    monkeypatch, pipe_request, pipe_user
):
    install_fake_open_webui_user_model(monkeypatch)
    captured = {}

    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            raise RuntimeError("config unavailable")

    async def get_sources_from_items(**kwargs):
        captured["rag_kwargs"] = {
            "k": kwargs["k"],
            "k_reranker": kwargs["k_reranker"],
            "r": kwargs["r"],
            "hybrid_bm25_weight": kwargs["hybrid_bm25_weight"],
            "hybrid_search": kwargs["hybrid_search"],
        }
        return [
            {
                "source": {"id": "file-a", "name": "file-a.txt"},
                "document": ["legacy rag context"],
                "metadata": [{"source": "file-a"}],
            }
        ]

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    retrieval_module = types.ModuleType("open_webui.retrieval.utils")
    retrieval_module.get_sources_from_items = get_sources_from_items
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "open_webui.retrieval.utils", retrieval_module)
    pipe_request.app.state.config = SimpleNamespace(
        TOP_K=4,
        TOP_K_RERANKER=5,
        RELEVANCE_THRESHOLD=0.33,
        HYBRID_BM25_WEIGHT=0.66,
        ENABLE_RAG_HYBRID_SEARCH=True,
    )

    context = await mod._generate_summary_file_context(
        request=pipe_request,
        user=pipe_user,
        prefix_files=[_file("file-a")],
    )

    assert captured["rag_kwargs"] == {
        "k": 4,
        "k_reranker": 5,
        "r": 0.33,
        "hybrid_bm25_weight": 0.66,
        "hybrid_search": True,
    }
    assert context is not None
    assert "legacy rag context" in context


@pytest.mark.asyncio
async def test_generate_summary_file_context_fails_closed_when_core_helper_raises(
    monkeypatch, pipe_request, pipe_user
):
    install_fake_open_webui_user_model(monkeypatch)

    async def get_sources_from_items(**kwargs):
        raise RuntimeError("retrieval unavailable")

    retrieval_module = types.ModuleType("open_webui.retrieval.utils")
    setattr(retrieval_module, "get_sources_from_items", get_sources_from_items)
    monkeypatch.setitem(sys.modules, "open_webui.retrieval.utils", retrieval_module)

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._generate_summary_file_context(
            request=pipe_request,
            user=pipe_user,
            prefix_files=[_file("file-a")],
        )


@pytest.mark.asyncio
async def test_summary_task_metadata_drops_unpickleable_core_values():
    future = asyncio.get_running_loop().create_future()
    metadata = {
        "chat_id": "chat-1",
        "params": {"function_calling": "native"},
        "mcp_clients": {"server": future},
    }

    summary_metadata = mod.build_summary_task_metadata(metadata)

    assert summary_metadata["chat_id"] == "chat-1"
    assert summary_metadata["params"] == {"function_calling": "native"}
    assert "mcp_clients" not in summary_metadata
    assert summary_metadata["task"] == mod.INTERNAL_SUMMARY_TASK


@pytest.mark.asyncio
async def test_target_completion_uses_forward_body_metadata_for_request_state(monkeypatch, pipe_request, pipe_user):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["state_metadata"] = copy.deepcopy(request.state.metadata)
        captured["form_metadata"] = copy.deepcopy(form_data["metadata"])
        captured["bypass_filter"] = bypass_filter
        return {"choices": [{"message": {"content": "ok"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    original_files = [{"id": "absorbed", "type": "file", "name": "old.pdf"}]
    retained_files = [{"id": "retained", "type": "file", "name": "current.pdf"}]
    pipe_request.state.metadata = {
        "chat_id": "chat-1",
        "message_id": "message-1",
        "files": original_files,
        "sources": [{"source": {"id": "stale"}}],
    }
    pipe_request.app.state.MODELS = {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}
    body = {
        "model": "target",
        "metadata": {
            "chat_id": "chat-1",
            "message_id": "message-1",
            "files": retained_files,
        },
        "messages": [{"role": "user", "content": "hello"}],
    }

    response = await mod._call_target_completion(request=pipe_request, user=pipe_user, body=body)

    assert response == {"choices": [{"message": {"content": "ok"}}]}
    assert captured["state_metadata"] == captured["form_metadata"]
    assert captured["state_metadata"]["files"] == retained_files
    assert "sources" not in captured["state_metadata"]
    assert captured["bypass_filter"] is True
    assert pipe_request.state.metadata["files"] == original_files
    assert pipe_request.state.metadata["sources"] == [{"source": {"id": "stale"}}]


@pytest.mark.asyncio
async def test_summary_generation_overrides_metadata_and_reapplies_system_prompt(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        request.state.bypass_filter = bypass_filter
        request.state.bypass_system_prompt = bypass_system_prompt
        captured["state_metadata"] = dict(request.state.metadata)
        captured["form_metadata"] = dict(form_data["metadata"])
        captured["bypass_system_prompt"] = bypass_system_prompt
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    pipe_request.state.metadata = {"selected_model_id": "arena-a", "tools": {"bad": {}}, "tool_ids": ["bad"]}
    pipe_request.state.bypass_system_prompt = True

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1", "selected_model_id": "arena-a", "tools": {"bad": {}}, "tool_ids": ["bad"]},
        summary_model_id="target",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
    )

    assert result == "summary"
    assert captured["state_metadata"]["task"] == mod.INTERNAL_SUMMARY_TASK
    assert captured["state_metadata"]["tools"] == {}
    assert captured["state_metadata"]["tool_ids"] == ["bad"]
    assert "selected_model_id" not in captured["state_metadata"]
    assert captured["form_metadata"] == captured["state_metadata"]
    assert captured["bypass_system_prompt"] is False
    assert pipe_request.state.metadata == {
        "selected_model_id": "arena-a",
        "tools": {"bad": {}},
        "tool_ids": ["bad"],
    }
    assert pipe_request.state.bypass_system_prompt is True
    assert not hasattr(pipe_request.state, "bypass_filter")


@pytest.mark.asyncio
async def test_summary_generation_does_not_emit_summary_start_before_route_resolution(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    events = []

    async def resolve_core_chat_model_route(request, model_id, **kwargs):
        raise RuntimeError("route failed")

    async def on_summary_start():
        events.append("started")

    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_core_chat_model_route)

    with pytest.raises(RuntimeError, match="route failed"):
        await mod._generate_summary_text(
            request=pipe_request,
            user=pipe_user,
            metadata={"chat_id": "chat-1"},
            summary_model_id="target",
            source_messages=[{"role": "user", "content": "old"}],
            base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
            on_summary_start=on_summary_start,
        )

    assert events == []


@pytest.mark.asyncio
async def test_summary_generation_strips_request_response_format_without_overriding_model_params(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}
    model_response_format = {"type": "json_schema", "json_schema": {"name": "Configured", "schema": {"type": "object"}}}
    model_max_tokens = 64000
    pipe_request.app.state.MODELS = {
        "summary": {
            "id": "summary",
            "name": "Summary",
            "params": {
                "max_tokens": model_max_tokens,
                "response_format": model_response_format,
            },
        }
    }

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["form_body"] = dict(form_data)
        captured["model_params"] = request.app.state.MODELS["summary"]["params"]
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="summary",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={
            "model": "target",
            "stream": True,
            "messages": [{"role": "user", "content": "old"}],
            "max_tokens": 1,
            "response_format": {"type": "json_object"},
        },
    )

    assert result == "summary"
    assert "max_tokens" not in captured["form_body"]
    assert "response_format" not in captured["form_body"]
    assert captured["model_params"]["max_tokens"] == model_max_tokens
    assert captured["model_params"]["response_format"] is model_response_format


@pytest.mark.asyncio
async def test_summary_generation_applies_fallback_params_when_route_falls_back(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}
    checked_model_ids = []
    fallback_model = {"id": "fallback-openai", "name": "Fallback", "owned_by": "openai", "openai": {}}
    pipe_request.app.state.MODELS = {"fallback-openai": fallback_model}

    async def resolve_core_chat_model_route(request, model_id, **kwargs):
        assert model_id == "summary-preset"
        return mod.CoreChatModelRoute(
            model_id="fallback-openai",
            fallback_model=fallback_model,
            target_params={
                "temperature": 0.2,
                "max_tokens": 128,
                "system": "drop this like Core fallback param handling",
                "custom_params": {"provider_flag": "true"},
            },
        )

    def apply_params_to_form_data(form_data, model):
        params = copy.deepcopy(form_data.pop("params", {}) or {})
        custom_params = params.pop("custom_params", {}) or {}
        for key in (
            "stream_response",
            "stream_delta_chunk_size",
            "function_calling",
            "reasoning_tags",
            "compact_token_threshold",
            "system",
        ):
            params.pop(key, None)
        params.update(custom_params)
        form_data.update({key: value for key, value in params.items() if value is not None})
        return form_data

    async def model_dict_from_request(request):
        return dict(pipe_request.app.state.MODELS)

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["form_body"] = copy.deepcopy(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.apply_params_to_form_data = apply_params_to_form_data
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_core_chat_model_route)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="summary-preset",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={
            "model": "target",
            "stream": True,
            "messages": [{"role": "user", "content": "old"}],
            "params": {"temperature": 0.4, "top_p": 0.7},
        },
    )

    assert result == "summary"
    assert checked_model_ids == ["fallback-openai"]
    assert captured["form_body"]["model"] == "fallback-openai"
    assert captured["form_body"]["temperature"] == 0.4
    assert captured["form_body"]["top_p"] == 0.7
    assert captured["form_body"]["max_tokens"] == 128
    assert captured["form_body"]["provider_flag"] == "true"
    assert "params" not in captured["form_body"]
    assert "system" not in captured["form_body"]


@pytest.mark.asyncio
async def test_summary_generation_checks_access_for_non_arena_fallback_model(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    fallback_model = {"id": "fallback-openai", "name": "Fallback", "owned_by": "openai", "openai": {}}
    pipe_request.app.state.MODELS = {"fallback-openai": fallback_model}
    checked_model_ids = []

    async def resolve_core_chat_model_route(request, model_id, **kwargs):
        assert model_id == "summary-preset"
        return mod.CoreChatModelRoute(
            model_id="fallback-openai",
            fallback_model=fallback_model,
            target_params=None,
        )

    async def model_dict_from_request(request):
        return dict(pipe_request.app.state.MODELS)

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])
        if model["id"] == "fallback-openai":
            raise HTTPException(status_code=403, detail="Model not found")

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        raise AssertionError("fallback model access denial must stop before summary generation")

    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_core_chat_model_route)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)

    with pytest.raises(HTTPException):
        await mod._generate_summary_text(
            request=pipe_request,
            user=pipe_user,
            metadata={"chat_id": "chat-1"},
            summary_model_id="summary-preset",
            source_messages=[{"role": "user", "content": "old"}],
            base_body={
                "model": "target",
                "stream": True,
                "messages": [{"role": "user", "content": "old"}],
            },
        )

    assert checked_model_ids == ["fallback-openai"]


@pytest.mark.asyncio
async def test_summary_generation_resolves_arena_fallback_before_params(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}
    fallback_arena = {
        "id": "fallback-arena",
        "name": "Fallback Arena",
        "owned_by": "arena",
        "arena": True,
        "info": {"meta": {"model_ids": ["arena-selected-ollama"]}},
    }
    selected_model = {
        "id": "arena-selected-ollama",
        "name": "Arena Selected Ollama",
        "owned_by": "ollama",
        "ollama": {},
    }
    pipe_request.app.state.MODELS = {
        "fallback-arena": fallback_arena,
        "arena-selected-ollama": selected_model,
    }

    async def resolve_core_chat_model_route(request, model_id, **kwargs):
        assert model_id == "summary-preset"
        return mod.CoreChatModelRoute(
            model_id="fallback-arena",
            fallback_model=fallback_arena,
            target_params={
                "temperature": 0.2,
                "max_tokens": 128,
                "custom_params": {"provider_flag": "true"},
            },
        )

    async def model_dict_from_request(request):
        return dict(pipe_request.app.state.MODELS)

    def apply_params_to_form_data(form_data, model):
        params = copy.deepcopy(form_data.pop("params", {}) or {})
        custom_params = params.pop("custom_params", {}) or {}
        params.update(custom_params)
        if params.get("max_tokens") is not None:
            params["num_predict"] = params.pop("max_tokens")
        if model.get("owned_by") == "ollama":
            form_data["options"] = params
        else:
            form_data.update({key: value for key, value in params.items() if value is not None})
        return form_data

    async def check_model_access(user, model, db=None):
        captured.setdefault("checked_models", []).append(model["id"])

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["form_body"] = copy.deepcopy(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.apply_params_to_form_data = apply_params_to_form_data
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_core_chat_model_route)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="summary-preset",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={
            "model": "target",
            "stream": True,
            "messages": [{"role": "user", "content": "old"}],
            "params": {"temperature": 0.4, "num_predict": 99},
        },
    )

    assert result == "summary"
    assert captured["checked_models"] == ["arena-selected-ollama"]
    assert captured["form_body"]["model"] == "arena-selected-ollama"
    assert captured["form_body"]["metadata"]["selected_model_id"] == "arena-selected-ollama"
    assert captured["form_body"]["options"]["temperature"] == 0.4
    assert captured["form_body"]["options"]["num_predict"] == 128
    assert captured["form_body"]["options"]["provider_flag"] == "true"
    assert "params" not in captured["form_body"]
    assert "max_tokens" not in captured["form_body"]["options"]


@pytest.mark.asyncio
async def test_summary_generation_checks_access_for_selected_arena_fallback_model(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    fallback_arena = {
        "id": "fallback-arena",
        "name": "Fallback Arena",
        "owned_by": "arena",
        "arena": True,
        "info": {"meta": {"model_ids": ["arena-selected"]}},
    }
    selected_model = {
        "id": "arena-selected",
        "name": "Arena Selected",
        "owned_by": "openai",
        "openai": {},
    }
    pipe_request.app.state.MODELS = {
        "fallback-arena": fallback_arena,
        "arena-selected": selected_model,
    }
    checked_model_ids = []

    async def resolve_core_chat_model_route(request, model_id, **kwargs):
        assert model_id == "summary-preset"
        return mod.CoreChatModelRoute(
            model_id="fallback-arena",
            fallback_model=fallback_arena,
            target_params=None,
        )

    async def model_dict_from_request(request):
        return dict(pipe_request.app.state.MODELS)

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])
        if model["id"] == "fallback-arena":
            raise AssertionError("arena wrapper access must not be checked after fallback selection")
        if model["id"] == "arena-selected":
            raise HTTPException(status_code=403, detail="Model not found")

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        raise AssertionError("selected arena model access denial must stop before summary generation")

    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_core_chat_model_route)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)

    with pytest.raises(HTTPException):
        await mod._generate_summary_text(
            request=pipe_request,
            user=pipe_user,
            metadata={"chat_id": "chat-1"},
            summary_model_id="summary-preset",
            source_messages=[{"role": "user", "content": "old"}],
            base_body={
                "model": "target",
                "stream": True,
                "messages": [{"role": "user", "content": "old"}],
            },
        )

    assert checked_model_ids == ["arena-selected"]


@pytest.mark.asyncio
async def test_summary_generation_rejects_stale_arena_fallback_candidate_before_forwarding(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    fallback_arena = {
        "id": "fallback-arena",
        "name": "Fallback Arena",
        "owned_by": "arena",
        "arena": True,
        "info": {"meta": {"model_ids": ["stale-id", "arena-selected"]}},
    }
    selected_model = {
        "id": "arena-selected",
        "name": "Arena Selected",
        "owned_by": "openai",
        "openai": {},
    }
    pipe_request.app.state.MODELS = {
        "fallback-arena": fallback_arena,
        "arena-selected": selected_model,
    }

    async def resolve_core_chat_model_route(request, model_id, **kwargs):
        assert model_id == "summary-preset"
        return mod.CoreChatModelRoute(
            model_id="fallback-arena",
            fallback_model=fallback_arena,
            target_params=None,
        )

    async def model_dict_from_request(request):
        return dict(pipe_request.app.state.MODELS)

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        raise AssertionError("stale arena fallback candidate must stop before summary generation")

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_core_chat_model_route)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod.random, "choice", lambda items: "stale-id")

    with pytest.raises(HTTPException):
        await mod._generate_summary_text(
            request=pipe_request,
            user=pipe_user,
            metadata={"chat_id": "chat-1"},
            summary_model_id="summary-preset",
            source_messages=[{"role": "user", "content": "old"}],
            base_body={
                "model": "target",
                "stream": True,
                "messages": [{"role": "user", "content": "old"}],
            },
        )


@pytest.mark.asyncio
async def test_summary_generation_strips_inherited_response_limits_and_stop(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}

    async def generate_chat_completion(
        request, form_data, user, bypass_filter=False, bypass_system_prompt=False
    ):
        captured["form_body"] = dict(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={},
        summary_model_id="summary",
        source_messages=[{"role": "user", "content": "source"}],
        base_body={
            "model": "target",
            "stream": True,
            "messages": [{"role": "user", "content": "old"}],
            "max_tokens": 32,
            "max_completion_tokens": 64,
            "max_output_tokens": 128,
            "stop": ["END"],
            "temperature": 0.2,
            "top_p": 0.9,
        },
    )

    assert result == "summary"
    assert "max_tokens" not in captured["form_body"]
    assert "max_completion_tokens" not in captured["form_body"]
    assert "max_output_tokens" not in captured["form_body"]
    assert "stop" not in captured["form_body"]
    assert captured["form_body"]["temperature"] == 0.2
    assert captured["form_body"]["top_p"] == 0.9


@pytest.mark.asyncio
async def test_summary_generation_reshapes_sampling_params_without_inheriting_ollama_response_controls(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}
    pipe_request.app.state.MODELS = {
        "summary-ollama": {
            "id": "summary-ollama",
            "name": "Summary Ollama",
            "owned_by": "ollama",
            "ollama": {},
        }
    }

    async def model_dict_from_request(request):
        return dict(pipe_request.app.state.MODELS)

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["form_body"] = copy.deepcopy(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="summary-ollama",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={
            "model": "target",
            "stream": True,
            "messages": [{"role": "user", "content": "old"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 1,
            "format": "json",
            "options": {
                "max_tokens": 2,
                "num_predict": 3,
                "stop": ["OPTION_STOP"],
                "response_format": {"type": "json_object"},
                "format": "json",
            },
            "params": {
                "max_completion_tokens": 4,
                "custom_params": {
                    "max_output_tokens": 5,
                    "num_predict": 6,
                    "stop": ["CUSTOM_STOP"],
                    "response_format": {"type": "json_object"},
                    "format": "json",
                },
            },
        },
    )

    assert result == "summary"
    assert captured["form_body"]["model"] == "summary-ollama"
    assert captured["form_body"]["options"] == {
        "temperature": 0.2,
        "top_p": 0.9,
    }
    for key in mod.SUMMARY_INHERITED_RESPONSE_CONTROL_KEYS:
        assert key not in captured["form_body"]
        assert key not in captured["form_body"]["options"]
    assert "params" not in captured["form_body"]


@pytest.mark.asyncio
async def test_summary_generation_decodes_own_wrapper_summary_model(monkeypatch, pipe_request, pipe_user):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["model"] = form_data["model"]
        captured["stream"] = form_data["stream"]
        captured["metadata"] = dict(form_data["metadata"])
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    wrapper_summary_model = mod.build_wrapper_model_id("auto_compact", "target.summary")

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id=wrapper_summary_model,
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
    )

    assert result == "summary"
    assert captured["model"] == "target.summary"
    assert captured["stream"] is True
    assert captured["metadata"]["task"] == mod.INTERNAL_SUMMARY_TASK


@pytest.mark.asyncio
async def test_summary_generation_decodes_runtime_registered_wrapper_summary_model(monkeypatch, pipe_request, pipe_user):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["model"] = form_data["model"]
        captured["base_model_id"] = getattr(request, "base_model_id", None)
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id=mod.build_wrapper_model_id("compact_alias", "target.summary"),
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
        pipe_function_id="compact_alias",
    )

    assert result == "summary"
    assert captured["model"] == "target.summary"


@pytest.mark.asyncio
async def test_summary_generation_uses_core_fallback_default_for_custom_model_missing_base(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}
    checked_model_ids = []
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-summary")
    pipe_request.app.state.MODELS = {
        "summary-preset": {
            "id": "summary-preset",
            "name": "Summary Preset",
            "owned_by": "openai",
            "info": {"base_model_id": "stale-summary-base"},
        },
        "fallback-summary": {"id": "fallback-summary", "name": "Fallback Summary", "owned_by": "openai"},
    }

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "summary-preset"
            return SimpleNamespace(id="summary-preset", base_model_id="stale-summary-base")

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["model"] = form_data["model"]
        captured["base_model_id"] = getattr(request, "base_model_id", None)
        return {"choices": [{"message": {"content": "summary"}}]}

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="summary-preset",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
    )

    assert result == "summary"
    assert checked_model_ids == ["fallback-summary"]
    assert captured == {"model": "fallback-summary", "base_model_id": None}


@pytest.mark.asyncio
async def test_summary_generation_preserves_tools_but_disables_forced_tool_choice(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = []

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured.append(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    source_messages = [
        {"role": "system", "content": "target system"},
        {"role": "user", "content": "old"},
    ]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    metadata = {
        "chat_id": "chat-1",
        "selected_model_id": "arena-a",
        "tools": {"lookup": {"spec": tools[0]["function"]}},
        "tool_ids": ["lookup"],
    }
    base_body = {
        "model": "target",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [*source_messages, {"role": "user", "content": "active"}],
        "tools": tools,
        "metadata": {"chat_id": "chat-1"},
        "previous_response_id": "resp_should_not_survive_compaction",
    }

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata=metadata,
        summary_model_id="target.summary",
        source_messages=source_messages,
        base_body={**base_body, "tool_choice": "required"},
    )

    assert result == "summary"
    form_data = captured[-1]
    assert form_data["model"] == "target.summary"
    assert form_data["stream"] is True
    assert form_data["stream_options"] == {"include_usage": True}
    assert form_data["tools"] == tools
    assert form_data["tool_choice"] == "none"
    assert "previous_response_id" not in form_data
    assert form_data["messages"][:-1] == source_messages
    assert form_data["messages"][-1]["role"] == "user"
    assert "AUTO-COMPACTION CHECKPOINT SUMMARY" in form_data["messages"][-1]["content"]
    assert form_data["metadata"]["task"] == mod.INTERNAL_SUMMARY_TASK
    assert form_data["metadata"]["tools"] == {}
    assert form_data["metadata"]["tool_ids"] == ["lookup"]
    assert "selected_model_id" not in form_data["metadata"]

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata=metadata,
        summary_model_id="target.summary",
        source_messages=source_messages,
        base_body={**base_body, "tool_choice": "auto"},
    )

    assert result == "summary"
    assert captured[-1]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_summary_generation_retries_without_tools_after_tool_call_response(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = []
    bypass_values = []

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured.append(copy.deepcopy(form_data))
        bypass_values.append(bypass_system_prompt)
        if len(captured) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "summary after retry"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    pipe_request.state.bypass_system_prompt = True

    source_messages = [
        {"role": "system", "content": "target system"},
        {"role": "user", "content": "old"},
    ]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    base_body = {
        "model": "target",
        "stream": True,
        "messages": [*source_messages, {"role": "user", "content": "active"}],
        "tools": tools,
        "tool_choice": "auto",
        "functions": [{"name": "legacy_lookup", "parameters": {"type": "object"}}],
        "function_call": "auto",
        "parallel_tool_calls": True,
    }

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1", "tool_ids": ["lookup"]},
        summary_model_id="target.summary",
        source_messages=source_messages,
        base_body=base_body,
    )

    assert result == "summary after retry"
    assert len(captured) == 2
    assert bypass_values == [False, False]
    assert captured[0]["tools"] == tools
    assert captured[0]["tool_choice"] == "auto"
    assert "tools" not in captured[1]
    assert "tool_choice" not in captured[1]
    assert "functions" not in captured[1]
    assert "function_call" not in captured[1]
    assert "parallel_tool_calls" not in captured[1]
    assert captured[1]["messages"] == captured[0]["messages"]
    assert captured[1]["metadata"] == captured[0]["metadata"]


@pytest.mark.asyncio
async def test_summary_generation_strips_tools_before_first_request_when_configured(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = []

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured.append(copy.deepcopy(form_data))
        return {"choices": [{"message": {"content": "summary without tools"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    source_messages = [{"role": "user", "content": "old"}]
    base_body = {
        "model": "target",
        "messages": [*source_messages, {"role": "user", "content": "active"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "functions": [{"name": "legacy_lookup", "parameters": {"type": "object"}}],
        "function_call": "auto",
        "parallel_tool_calls": True,
    }

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1", "tool_ids": ["lookup"]},
        summary_model_id="target.summary",
        source_messages=source_messages,
        base_body=base_body,
        summary_tool_policy="always_strip",
    )

    assert result == "summary without tools"
    assert len(captured) == 1
    assert "tools" not in captured[0]
    assert "tool_choice" not in captured[0]
    assert "functions" not in captured[0]
    assert "function_call" not in captured[0]
    assert "parallel_tool_calls" not in captured[0]


@pytest.mark.asyncio
async def test_summary_generation_errors_on_tool_call_when_fallback_disabled(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = []

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured.append(copy.deepcopy(form_data))
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    source_messages = [{"role": "user", "content": "old"}]
    base_body = {
        "model": "target",
        "messages": [*source_messages, {"role": "user", "content": "active"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
    }

    with pytest.raises(mod.SummaryToolCallError):
        await mod._generate_summary_text(
            request=pipe_request,
            user=pipe_user,
            metadata={"chat_id": "chat-1", "tool_ids": ["lookup"]},
            summary_model_id="target.summary",
            source_messages=source_messages,
            base_body=base_body,
            summary_tool_policy="error_on_tool_call",
        )

    assert len(captured) == 1
    assert captured[0]["tools"] == base_body["tools"]
    assert captured[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_summary_generation_uses_handoff_checkpoint_prompt(monkeypatch, pipe_request, pipe_user):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["messages"] = form_data["messages"]
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="target",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
    )

    assert result == "summary"
    assert captured["messages"][0] == {"role": "user", "content": "old"}
    assert captured["messages"][-1]["role"] == "user"
    prompt = captured["messages"][-1]["content"]
    assert "AUTO-COMPACTION CHECKPOINT SUMMARY" in prompt
    assert "Open WebUI chat" in prompt
    assert "future model call" in prompt
    assert "existing <auto_compaction_context>" in prompt
    assert "Do not invent facts" in prompt
    assert "Do not call tools" in prompt


def test_resolve_summary_prompt_returns_builtin_for_empty_or_whitespace():
    assert mod.resolve_summary_prompt(None) is mod.SUMMARY_PROMPT
    assert mod.resolve_summary_prompt("") is mod.SUMMARY_PROMPT
    assert mod.resolve_summary_prompt("   \n\t ") is mod.SUMMARY_PROMPT


def test_resolve_summary_prompt_returns_stripped_custom_prompt():
    custom = "CUSTOM HANDOFF PROMPT: be terse."
    assert mod.resolve_summary_prompt(custom) == custom
    assert mod.resolve_summary_prompt("  \n" + custom + "\n  ") == custom


@pytest.mark.asyncio
async def test_summary_generation_uses_custom_summary_prompt(monkeypatch, pipe_request, pipe_user):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["messages"] = form_data["messages"]
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    custom_prompt = "DISTINCTIVE CUSTOM MARKER 7c4f-9a21: rewrite the handoff in haiku form."

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="target",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
        summary_prompt=custom_prompt,
    )

    assert result == "summary"
    assert captured["messages"][0] == {"role": "user", "content": "old"}
    assert captured["messages"][-1] == {"role": "user", "content": custom_prompt}


@pytest.mark.asyncio
async def test_summary_generation_falls_back_to_builtin_prompt_when_summary_prompt_empty(
    monkeypatch, pipe_request, pipe_user
):
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["messages"] = form_data["messages"]
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    base_body = {"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]}

    for blank in ("", "   \n\t "):
        captured.clear()
        await mod._generate_summary_text(
            request=pipe_request,
            user=pipe_user,
            metadata={"chat_id": "chat-1"},
            summary_model_id="target",
            source_messages=[{"role": "user", "content": "old"}],
            base_body=base_body,
            summary_prompt=blank,
        )
        assert captured["messages"][-1]["content"] is mod.SUMMARY_PROMPT


def test_build_summary_request_message_prepends_file_context_to_custom_summary_prompt():
    prefix = "<attached_file_contents>ctx</attached_file_contents>"
    custom = "CUSTOM PROMPT: keep it short."

    message = mod.build_summary_request_message(prefix, summary_prompt=custom)

    assert message == {"role": "user", "content": f"{prefix}\n\n{custom}"}


def test_build_summary_completion_body_uses_custom_summary_prompt_in_final_message():
    custom = "DISTINCTIVE SUMMARY OVERRIDE 1a2b-3c4d."
    body = mod.build_summary_completion_body(
        {"model": "target", "messages": [{"role": "user", "content": "old"}], "metadata": {}},
        summary_model_id="summary",
        source_messages=[{"role": "user", "content": "source"}],
        metadata={"chat_id": "chat-1"},
        summary_prompt=custom,
    )

    assert body["messages"][:-1] == [{"role": "user", "content": "source"}]
    assert body["messages"][-1] == {"role": "user", "content": custom}


def test_build_summary_completion_body_prepends_preserved_system_message():
    system = {"role": "system", "content": "system prompt"}
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]

    body = mod.build_summary_completion_body(
        {"model": "target", "messages": [{"role": "user", "content": "old"}], "metadata": {}},
        summary_model_id="summary",
        source_messages=source_messages,
        preserved_system_message=system,
        metadata={"chat_id": "chat-1"},
    )

    assert body["messages"][:-1] == [system, *source_messages]
    assert body["messages"][-1]["role"] == "user"
    assert "AUTO-COMPACTION CHECKPOINT SUMMARY" in body["messages"][-1]["content"]


def test_summary_prompt_does_not_affect_checkpoint_profile_hash_or_identity():
    # compute_profile_hash intentionally ignores summary_prompt entirely (**_),
    # so changing it can never re-identify or invalidate a ready checkpoint.
    baseline = mod.compute_profile_hash()
    assert mod.compute_profile_hash(summary_prompt="") == baseline
    assert mod.compute_profile_hash(summary_prompt="anything-at-all") == baseline
    assert mod.compute_profile_hash(summary_prompt=" " * 16) == baseline

    # The Valve itself must not be a hash input either: two Valve sets that
    # differ ONLY in summary_prompt must yield identical profile hashes.
    valves_default = mod.Pipe.Valves()
    valves_custom = mod.Pipe.Valves()
    valves_custom.summary_prompt = "totally different prompt"
    assert valves_default.summary_prompt == mod.SUMMARY_PROMPT
    assert valves_custom.summary_prompt == "totally different prompt"
    assert baseline == mod.compute_profile_hash(
        schema_family=mod.CHECKPOINT_TABLE_NAME,
        summary_format_family=mod.SUMMARY_FORMAT_FAMILY,
        source_hash_family=mod.SOURCE_HASH_FAMILY,
    )


@pytest.mark.asyncio
async def test_compact_body_summary_request_preserves_system_but_checkpoint_identity_excludes_it(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    captured = {}
    system = {"role": "system", "content": "system prompt"}
    volatile_system = {"role": "system", "content": "volatile middle system"}
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    request_prefix = [source_messages[0], volatile_system, source_messages[1]]
    messages = [*request_prefix, {"role": "user", "content": "active"}]

    async def noop_initialize(**kwargs):
        return None

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["messages"] = copy.deepcopy(form_data["messages"])
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))

    compacted, did_compact, prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body={"model": "target", "messages": [system, *messages]},
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
    )

    assert did_compact is True
    assert prefix_count == len(source_messages)
    assert captured["messages"][:-1] == [system, *request_prefix]
    assert captured["messages"][-1]["role"] == "user"
    assert rows[0]["source_message_count"] == len(source_messages)
    assert rows[0]["source_hash"] == mod.compute_source_hash(source_messages)
    assert rows[0]["source_hash"] == mod.compute_source_hash([system, *request_prefix])
    assert rows[0]["summary_meta"]["historical_user_messages"]["messages"] == [
        {"ordinal": 1, "text": "old"}
    ]
    assert compacted["messages"][0] == system
    assert "summary" in compacted["messages"][1]["content"]
    assert compacted["messages"][2:] == [{"role": "user", "content": "active"}]
    assert volatile_system not in compacted["messages"]


@pytest.mark.asyncio
async def test_compact_body_reuses_checkpoint_when_only_system_content_changes(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(source_messages),
        source_message_count=len(source_messages),
        summary_text="reused summary",
        summary_meta=mod.build_checkpoint_summary_meta(
            source_messages,
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        ),
        parent_checkpoint_id=None,
        now=123,
    )
    store = ClaimCheckpointStore([checkpoint])

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        raise AssertionError("system-only changes must not invalidate a reusable checkpoint")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    compacted, did_compact, prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body={
            "model": "target",
            "messages": [
                {"role": "system", "content": "changed system prompt"},
                *source_messages,
                {"role": "user", "content": "active"},
            ],
        },
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
    )

    assert did_compact is True
    assert prefix_count == len(source_messages)
    assert store.touched == [checkpoint["id"]]
    assert compacted["messages"][0] == {"role": "system", "content": "changed system prompt"}
    assert "reused summary" in compacted["messages"][1]["content"]


@pytest.mark.asyncio
async def test_compact_body_reuses_checkpoint_when_transient_user_content_changes(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    patterns = mod.parse_transient_message_patterns(TRANSIENT_MARKER)
    stable_source_messages = [
        {"role": "user", "content": "old"},
        {"role": "user", "content": "<SYSTEM_CONTEXT>now: 10:00</SYSTEM_CONTEXT>"},
        {"role": "assistant", "content": "old answer"},
    ]
    checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(stable_source_messages, transient_message_patterns=patterns),
        source_message_count=mod._source_identity_message_count(
            stable_source_messages,
            transient_message_patterns=patterns,
        ),
        summary_text="reused summary",
        summary_meta=mod.build_checkpoint_summary_meta(
            stable_source_messages,
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
            transient_message_patterns=patterns,
        ),
        parent_checkpoint_id=None,
        now=123,
    )
    store = ClaimCheckpointStore([checkpoint])

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        raise AssertionError("transient user message changes must not invalidate a reusable checkpoint")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    compacted, did_compact, prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body={
            "model": "target",
            "messages": [
                {"role": "system", "content": "first system prompt"},
                {"role": "user", "content": "old"},
                {"role": "user", "content": "  <SYSTEM_CONTEXT>now: 10:01</SYSTEM_CONTEXT>\n"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
        transient_message_patterns=patterns,
    )

    assert did_compact is True
    assert prefix_count == 2
    assert store.touched == [checkpoint["id"]]
    assert compacted["messages"][0] == {"role": "system", "content": "first system prompt"}
    assert "reused summary" in compacted["messages"][1]["content"]
    assert "now: 10:00" not in compacted["messages"][1]["content"]
    assert "now: 10:01" not in compacted["messages"][1]["content"]
    assert compacted["messages"][2:] == [{"role": "user", "content": "active"}]


@pytest.mark.asyncio
async def test_compact_body_skips_checkpoint_when_prefix_has_only_transient_user(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    patterns = mod.parse_transient_message_patterns(TRANSIENT_MARKER)
    rows = []

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        raise AssertionError("transient-only prefixes must not create checkpoints")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "model": "target",
        "messages": [
            {"role": "user", "content": "<SYSTEM_CONTEXT>now: 10:00</SYSTEM_CONTEXT>"},
            {"role": "user", "content": "active"},
        ],
    }

    compacted, did_compact, prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
        transient_message_patterns=patterns,
    )

    assert compacted is body
    assert did_compact is False
    assert prefix_count == 0
    assert rows == []


@pytest.mark.asyncio
async def test_compact_body_reuses_checkpoint_when_middle_system_presence_changes(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    stable_source_messages = [
        {"role": "user", "content": "old"},
        {"role": "system", "content": "original volatile middle system"},
        {"role": "assistant", "content": "old answer"},
    ]
    checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(stable_source_messages),
        source_message_count=len([message for message in stable_source_messages if message["role"] != "system"]),
        summary_text="reused summary",
        summary_meta=mod.build_checkpoint_summary_meta(
            stable_source_messages,
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        ),
        parent_checkpoint_id=None,
        now=123,
    )
    store = ClaimCheckpointStore([checkpoint])

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        raise AssertionError("middle system presence changes must not invalidate a reusable checkpoint")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    compacted, did_compact, prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body={
            "model": "target",
            "messages": [
                {"role": "system", "content": "first system prompt"},
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
    )

    assert did_compact is True
    assert prefix_count == 2
    assert store.touched == [checkpoint["id"]]
    assert compacted["messages"][0] == {"role": "system", "content": "first system prompt"}
    assert "reused summary" in compacted["messages"][1]["content"]
    assert compacted["messages"][2:] == [{"role": "user", "content": "active"}]


@pytest.mark.asyncio
async def test_compact_body_parent_extension_failure_returns_chain_parent_boundary_after_middle_system(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    parent_raw_source = [
        {"role": "user", "content": "old"},
        {"role": "system", "content": "volatile middle system"},
    ]
    parent = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(parent_raw_source),
        source_message_count=1,
        summary_text="parent summary",
        summary_meta=mod.build_checkpoint_summary_meta(
            parent_raw_source,
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        ),
        parent_checkpoint_id=None,
        now=123,
    )
    store = ClaimCheckpointStore([parent])
    summary_inputs = []

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        raise RuntimeError("parent extension failed")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    compacted, did_compact, prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body={
            "model": "target",
            "messages": [
                {"role": "system", "content": "first system prompt"},
                {"role": "user", "content": "old"},
                {"role": "system", "content": "volatile middle system"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
    )

    assert did_compact is True
    assert prefix_count == 1
    assert store.touched == []
    assert summary_inputs[0][0]["role"] == "user"
    assert "parent summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [{"role": "assistant", "content": "old answer"}]
    assert compacted["messages"][0] == {"role": "system", "content": "first system prompt"}
    assert "parent summary" in compacted["messages"][1]["content"]
    assert compacted["messages"][2:] == [
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]


@pytest.mark.asyncio
async def test_tool_result_compaction_summary_request_preserves_system_but_identity_excludes_it(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    captured = {}
    system = {"role": "system", "content": "system prompt"}
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]
    latest_round = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]
    messages = [system, *source_messages, *latest_round]

    async def noop_initialize(**kwargs):
        return None

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["messages"] = copy.deepcopy(form_data["messages"])
        return {"choices": [{"message": {"content": "tool summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))

    compacted, did_compact, prefix_count = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={"model": "target", "messages": messages},
        messages=messages,
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
    )

    assert did_compact is True
    assert prefix_count == len(source_messages)
    assert captured["messages"][:-1] == [system, *source_messages]
    assert captured["messages"][-1]["role"] == "user"
    assert rows[0]["source_message_count"] == len(source_messages)
    assert rows[0]["source_hash"] == mod.compute_source_hash(source_messages)
    assert compacted[0] == system


@pytest.mark.asyncio
async def test_summary_generation_passes_open_webui_user_model_to_inner_completion(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    captured = {}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["user_id"] = user.id
        captured["user_role"] = user.role
        captured["user_type"] = type(user)
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)

    result = await mod._generate_summary_text(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        summary_model_id="target",
        source_messages=[{"role": "user", "content": "old"}],
        base_body={"model": "target", "stream": True, "messages": [{"role": "user", "content": "old"}]},
    )

    assert result == "summary"
    assert captured == {"user_id": "user-1", "user_role": "user", "user_type": fake_user_model}


@pytest.mark.asyncio
async def test_tool_history_compaction_summarizes_before_latest_tool_round_and_preserves_tool_messages(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}

    async def get_or_create_compaction_summary(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "combined summary"

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    original_tool = {"role": "tool", "tool_call_id": "call-1", "content": {"value": 42}}
    compacted, did_compact, _tool_prefix_count = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={
            "model": "target",
            "stream": True,
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "old request"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active request"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q":"alpha"}'},
                        }
                    ],
                },
                original_tool,
            ],
        },
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q":"alpha"}'},
                    }
                ],
            },
            original_tool,
        ],
    )

    assert did_compact is True
    assert original_tool == {"role": "tool", "tool_call_id": "call-1", "content": {"value": 42}}
    assert [message["role"] for message in captured["source_messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert compacted[0] == {"role": "system", "content": "system prompt"}
    assert compacted[1]["role"] == "user"
    assert "<auto_compaction_context>" in compacted[1]["content"]
    assert "combined summary" in compacted[1]["content"]
    assert compacted[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"alpha"}'},
            }
        ],
    }
    assert compacted[3] == original_tool


@pytest.mark.asyncio
async def test_tool_history_checkpoint_render_skips_transient_user_excerpts(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    patterns = mod.parse_transient_message_patterns(TRANSIENT_MARKER)

    async def get_or_create_compaction_summary(**kwargs):
        return SimpleNamespace(
            checkpoint={
                "summary_text": "tool summary",
                "summary_meta": {},
            }
        )

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    latest_round = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]
    historical = [
        {"role": "user", "content": "old request"},
        {"role": "user", "content": "<SYSTEM_CONTEXT>now: 10:00</SYSTEM_CONTEXT>"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active request"},
    ]

    compacted, did_compact, _tool_prefix_count = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={"model": "target", "messages": [*historical, *latest_round]},
        messages=[*historical, *latest_round],
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=4,
        transient_message_patterns=patterns,
    )

    assert did_compact is True
    assert "tool summary" in compacted[0]["content"]
    assert "now: 10:00" not in compacted[0]["content"]
    assert "old request" in compacted[0]["content"]
    assert "active request" in compacted[0]["content"]


@pytest.mark.asyncio
async def test_tool_history_compaction_always_extends_available_parent_checkpoint(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active request"}
    parent_checkpoint = {
        "id": "history-checkpoint-1",
        "state": "ready",
        "source_message_count": len(history),
        "source_hash": mod.compute_source_hash(history),
        "summary_text": "existing history summary",
        "summary_meta": {},
    }
    rows.append(parent_checkpoint)

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "tool summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]
    messages = [*history, active, *latest_round]

    compacted, did_compact, _tool_prefix_count = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={"model": "target", "stream": True, "messages": messages},
        messages=messages,
    )

    assert did_compact is True
    assert len(summary_inputs) == 1
    assert "existing history summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [active]
    assert len(rows) == 2
    assert rows[1]["parent_checkpoint_id"] == parent_checkpoint["id"]
    assert "tool summary" in compacted[0]["content"]
    assert compacted[1:] == latest_round


@pytest.mark.asyncio
async def test_tool_history_parent_handoff_uses_parent_checkpoint_saved_excerpts(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    summary_inputs = []
    history = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active request"}
    parent_checkpoint = {
        "id": "history-checkpoint-1",
        "state": "ready",
        "source_message_count": len(history),
        "source_hash": mod.compute_source_hash(history),
        "summary_text": "existing history summary",
        "summary_meta": mod.enrich_summary_meta_with_historical_excerpts(
            {},
            history,
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        ),
    }
    rows = [parent_checkpoint]

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "tool summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]

    compacted, did_compact, _tool_prefix_count = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={"model": "target", "stream": True, "messages": [*history, active, *latest_round]},
        messages=[*history, active, *latest_round],
    )

    assert did_compact is True
    parent_context = summary_inputs[0][0]["content"]
    assert "existing history summary" in parent_context
    assert "<historical_user_messages" in parent_context
    assert '<historical_user_message ordinal="1"><![CDATA[old request]]></historical_user_message>' in parent_context
    assert "tool summary" in compacted[0]["content"]


@pytest.mark.asyncio
async def test_tool_history_compaction_summarizes_all_but_latest_sequential_tool_round(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    captured = {}

    async def get_or_create_compaction_summary(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "combined summary"

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    messages = [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "search", "arguments": '{"q":"alpha"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": {"value": 42}},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-2", "type": "function", "function": {"name": "fetch", "arguments": '{"url":"https://example.test"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "fetched body"},
    ]

    compacted, did_compact, _tool_prefix_count = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={"model": "target", "stream": True, "messages": messages},
        messages=messages,
    )

    assert did_compact is True
    assert captured["source_messages"] == messages[:3]
    assert "combined summary" in compacted[0]["content"]
    assert compacted[1:] == messages[3:]


@pytest.mark.asyncio
async def test_extract_summary_text_from_streaming_response():
    background_ran = False

    async def chunks():
        yield b'data: {"choices": [{"delta": {"content": "hello "}}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": "world"}}]}\n\n'

    async def background():
        nonlocal background_ran
        background_ran = True

    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    assert await mod.extract_text_from_completion_response(response) == "hello world"
    assert background_ran is True


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["\u0000", "\ud800", "\udc00"])
async def test_extract_summary_text_rejects_all_invalid_db_characters_streaming_response(
    content,
):
    async def chunks():
        payload = {"choices": [{"delta": {"content": content}}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(RuntimeError, match="did not return text content"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_sanitizes_invalid_db_characters_from_streaming_response():
    async def chunks():
        payload = {"choices": [{"delta": {"content": "hel\ud800😀\udc00lo"}}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "hel😀lo"


@pytest.mark.asyncio
async def test_extract_summary_text_accepts_mixed_case_sse_media_type():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="Text/Event-Stream")

    assert await mod.extract_text_from_completion_response(response) == "hello"


@pytest.mark.asyncio
async def test_extract_summary_text_from_streamed_chat_completion_message():
    async def chunks():
        payload = {"choices": [{"message": {"role": "assistant", "content": "pipe summary"}}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "pipe summary"


@pytest.mark.asyncio
async def test_extract_summary_text_from_split_sse_event_chunks():
    async def chunks():
        yield b'data: {"choices": [{"delta": '
        yield b'{"content": "hello"}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "hello"


@pytest.mark.asyncio
async def test_extract_summary_text_preserves_split_utf8_sse_chunks():
    payload = {"choices": [{"delta": {"content": "こんにちは"}}]}
    event = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
    split_at = event.index("こ".encode()) + 1

    async def chunks():
        yield event[:split_at]
        yield event[split_at:]

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "こんにちは"


@pytest.mark.asyncio
async def test_extract_summary_text_preserves_streamed_delta_newlines():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"content": "line 1\\n"}}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": "line 2"}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "line 1\nline 2"


@pytest.mark.asyncio
async def test_extract_summary_text_ignores_non_structured_sse_data():
    background_ran = False

    async def chunks():
        yield b"data: plain text\n\n"
        yield b'data: "json string"\n\n'
        yield b"data: 2026\n\n"
        yield b'data: {"q":"alpha"}\n\n'
        yield b"data: [DONE]\n\n"

    async def background():
        nonlocal background_ran
        background_ran = True

    response = StreamingResponse(chunks(), media_type="text/event-stream", background=BackgroundTask(background))

    with pytest.raises(RuntimeError, match="did not return text content"):
        await mod.extract_text_from_completion_response(response)
    assert background_ran is True


@pytest.mark.asyncio
async def test_extract_summary_text_propagates_streamed_context_error():
    async def chunks():
        payload = {"error": {"code": "context_length_exceeded", "message": "maximum context length exceeded"}}
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_propagates_responses_api_error_event():
    async def chunks():
        payload = {"type": "error", "code": "context_length_exceeded", "message": "maximum context length exceeded"}
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(mod.RetryableContextOverflow):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_propagates_responses_api_failed_event():
    async def chunks():
        payload = {
            "type": "response.failed",
            "response": {"error": {"code": "server_error", "message": "The model failed to generate a response."}},
        }
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(RuntimeError, match="The model failed to generate a response"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_non_sse_streaming_response():
    async def chunks():
        yield b"plain text"

    response = StreamingResponse(chunks(), media_type="text/plain")

    with pytest.raises(RuntimeError, match="structured OpenAI-compatible response"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_from_responses_api_output():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "hello "},
                    {"type": "output_text", "text": "world"},
                ],
            }
        ]
    }

    assert await mod.extract_text_from_completion_response(response) == "hello world"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"choices": [{"message": {"content": "hel\ud800😀\udc00lo"}}]},
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hel\ud800😀\udc00lo"}],
                }
            ]
        },
    ],
)
async def test_extract_summary_text_sanitizes_invalid_db_characters_from_dict_response(
    response,
):
    assert await mod.extract_text_from_completion_response(response) == "hel😀lo"


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_unstructured_non_stream_response():
    with pytest.raises(RuntimeError, match="did not return text content"):
        await mod.extract_text_from_completion_response("plain text")

    with pytest.raises(RuntimeError, match="did not return text content"):
        await mod.extract_text_from_completion_response({"content": "plain text"})


@pytest.mark.asyncio
async def test_extract_summary_text_preserves_regular_short_line_summary():
    summary = "\n".join(f"- id-{i}" for i in range(100))
    response = {"choices": [{"message": {"content": summary}}]}

    assert await mod.extract_text_from_completion_response(response) == summary


@pytest.mark.asyncio
async def test_extract_summary_text_ignores_unused_incomplete_choices():
    response = {
        "choices": [
            {"message": {"content": "complete summary"}, "finish_reason": "stop"},
            {"message": {"content": "unused partial summary"}, "finish_reason": "length"},
        ]
    }

    assert await mod.extract_text_from_completion_response(response) == "complete summary"


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_length_finished_chat_completion():
    response = {"choices": [{"message": {"content": "partial summary"}, "finish_reason": "length"}]}

    with pytest.raises(RuntimeError, match="stopped before completing"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_max_output_finished_chat_completion():
    response = {"choices": [{"message": {"content": "partial summary"}, "finish_reason": "max_output_tokens"}]}

    with pytest.raises(RuntimeError, match="stopped before completing"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_length_finished_streaming_completion():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"content": "partial summary"}}]}\n\n'
        yield b'data: {"choices": [{"finish_reason": "length", "delta": {}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(RuntimeError, match="stopped before completing"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_max_output_finished_streaming_completion():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"content": "partial summary"}}]}\n\n'
        yield b'data: {"choices": [{"finish_reason": "max_output_tokens", "delta": {}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(RuntimeError, match="stopped before completing"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_incomplete_responses_api_output():
    response = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output_text": "partial summary",
    }

    with pytest.raises(RuntimeError, match="stopped before completing"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_ignores_streamed_reasoning_content():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"reasoning_content": "Let me summarize.\\n"}}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": "Final summary"}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    assert await mod.extract_text_from_completion_response(response) == "Final summary"


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_tool_call_response():
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    with pytest.raises(RuntimeError, match="tool call"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_raw_responses_tool_call_output():
    response = {
        "id": "resp-1",
        "output": [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{}",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="tool call"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_streamed_tool_call_response():
    async def chunks():
        yield b'data: {"choices": [{"delta": {"tool_calls": [{"id": "call-1"}]}}]}\n\n'
        yield b'data: {"choices": [{"finish_reason": "tool_calls", "delta": {}}]}\n\n'

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(RuntimeError, match="tool call"):
        await mod.extract_text_from_completion_response(response)


@pytest.mark.asyncio
async def test_extract_summary_text_rejects_streamed_message_content_with_tool_call():
    async def chunks():
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "not a final summary",
                        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup"}}],
                    }
                }
            ]
        }
        yield f"data: {json.dumps(payload)}\n\n".encode()

    response = StreamingResponse(chunks(), media_type="text/event-stream")

    with pytest.raises(RuntimeError, match="tool call"):
        await mod.extract_text_from_completion_response(response)


def test_build_summary_completion_body_strips_response_format():
    source_messages = [{"role": "user", "content": "hello"}]
    base_body = {
        "model": "target",
        "messages": source_messages,
        "stream": True,
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Answer", "schema": {"type": "object"}},
        },
    }

    body = mod.build_summary_completion_body(
        base_body,
        summary_model_id="summary",
        source_messages=source_messages,
        metadata={"chat_id": "chat-1"},
    )

    assert "response_format" not in body
    assert body["temperature"] == 0.2


@pytest.fixture
def pipe_request():
    return SimpleNamespace(state=SimpleNamespace(), app=SimpleNamespace(state=SimpleNamespace(MODELS={})))


@pytest.fixture
def pipe_user():
    return {
        "id": "user-1",
        "email": "user@example.com",
        "name": "User",
        "role": "user",
        "last_active_at": 0,
        "updated_at": 0,
        "created_at": 0,
    }


@pytest.fixture
def pipe_metadata():
    return {"chat_id": "chat-1", "message_id": "message-1", "session_id": "session-1"}


def _install_candidate_token_estimate(monkeypatch, value, *, captured=None):
    async def estimate_body_tokens_async(body, **kwargs):
        if captured is not None:
            captured.append(copy.deepcopy(body))
        return value

    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)


def _install_known_openai_usage_anchor_transport(monkeypatch):
    async def transport_profile(request, models, provider_model_id):
        return {
            "owned_by": "openai",
            "url_idx": 0,
            "url": "http://provider",
            "api_type": "chat_completions",
            "api_config": {},
        }, frozenset()

    monkeypatch.setattr(mod, "_usage_anchor_transport_profile", transport_profile)


def _install_durable_usage_anchor_estimate(monkeypatch, value, *, captured=None):
    async def parent_assistant_message_id(**kwargs):
        return "assistant-parent"

    async def lookup_usage_anchor(**kwargs):
        return mod.UsageAnchor(
            assistant_message_id="assistant-parent",
            input_tokens=1,
            stable_message_count=0,
            input_fingerprint="test-anchor",
            volatile_message_tokens=0,
        )

    async def estimate_body_tokens_from_usage_anchor(**kwargs):
        if captured is not None:
            captured.append(copy.deepcopy(kwargs["body"]))
        return value

    monkeypatch.setattr(mod, "_usage_anchor_parent_assistant_message_id", parent_assistant_message_id)
    monkeypatch.setattr(mod, "lookup_usage_anchor", lookup_usage_anchor)
    monkeypatch.setattr(mod, "_estimate_body_tokens_from_usage_anchor", estimate_body_tokens_from_usage_anchor)
    _install_known_openai_usage_anchor_transport(monkeypatch)


def _disable_usage_anchor_persistence(monkeypatch):
    async def persist_usage_anchor(**kwargs):
        return True

    monkeypatch.setattr(mod, "persist_usage_anchor", persist_usage_anchor)


@pytest.mark.asyncio
@pytest.mark.parametrize("continue_response", [False, True])
async def test_pipe_persists_final_usage_for_the_actual_forward_candidate(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
    continue_response,
):
    captured = {}
    metadata = dict(pipe_metadata)
    if continue_response:
        metadata["assistant_message_id"] = metadata["message_id"]
    anchor_input = mod.UsageAnchorInput(
        stable_message_count=1,
        input_fingerprint="fingerprint",
        volatile_message_tokens=0,
    )

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_target_db_model_record(model_id):
        assert model_id == "target"
        return None

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def lookup_anchor(**kwargs):
        return None

    async def build_anchor_input(**kwargs):
        captured["candidate"] = copy.deepcopy(kwargs["body"])
        return anchor_input

    async def persist_anchor(**kwargs):
        captured["persist"] = kwargs
        return True

    async def call_target(**kwargs):
        return {
            "usage": {"prompt_tokens": 42, "completion_tokens": 3},
            "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        }

    forward_non_streaming_target = mod._forward_non_streaming_target

    async def forward_target(**kwargs):
        captured["on_complete"] = kwargs.get("on_complete")
        return await forward_non_streaming_target(**kwargs)

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_target_db_model_record", get_target_db_model_record)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "lookup_usage_anchor", lookup_anchor)
    monkeypatch.setattr(mod, "_build_usage_anchor_input", build_anchor_input)
    monkeypatch.setattr(mod, "persist_usage_anchor", persist_anchor)
    monkeypatch.setattr(mod, "_call_target_completion", call_target)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    _install_candidate_token_estimate(monkeypatch, 10)
    _install_known_openai_usage_anchor_transport(monkeypatch)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0
    result = await pipe.pipe(
        {
            "model": mod.build_wrapper_model_id("auto_compact", "target"),
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=metadata,
    )

    assert result["choices"][0]["message"]["content"] == "answer"
    assert captured["candidate"]["model"] == "target"
    if continue_response:
        assert captured["on_complete"] is None
        assert "persist" not in captured
    else:
        assert callable(captured["on_complete"])
        assert captured["persist"]["assistant_message_id"] == metadata["message_id"]
        assert captured["persist"]["anchor_input"] is anchor_input
        assert captured["persist"]["raw_usage"] == {
            "prompt_tokens": 42,
            "completion_tokens": 3,
        }
    assert mod.get_request_scoped_usage_anchor(
        request=pipe_request,
        chat_id=metadata["chat_id"],
        message_id=metadata["message_id"],
        wrapper_model_id=mod.build_wrapper_model_id("auto_compact", "target"),
    ) == mod.UsageAnchor(
        assistant_message_id="request",
        input_tokens=42,
        stable_message_count=1,
        input_fingerprint="fingerprint",
        volatile_message_tokens=0,
    )


@pytest.mark.asyncio
async def test_task_forward_preserves_outer_request_usage_anchor(monkeypatch, pipe_request):
    chat_id = "chat-1"
    message_id = "message-1"
    wrapper_model_id = mod.build_wrapper_model_id("auto_compact", "target")
    anchor_input = mod.UsageAnchorInput(
        stable_message_count=1,
        input_fingerprint="outer-forward",
        volatile_message_tokens=2,
    )
    mod.store_request_scoped_usage(
        request=pipe_request,
        chat_id=chat_id,
        message_id=message_id,
        wrapper_model_id=wrapper_model_id,
        usage={"prompt_tokens": 40, "completion_tokens": 3},
        anchor_input=anchor_input,
    )

    async def call_target(**kwargs):
        return {
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            "choices": [{"message": {"role": "assistant", "content": "query"}}],
        }

    monkeypatch.setattr(mod, "_call_target_completion", call_target)

    response = await mod._forward_non_streaming_target(
        request=pipe_request,
        user={},
        body={"model": "target", "messages": [{"role": "user", "content": "query"}]},
        chat_id=chat_id,
        message_id=message_id,
        wrapper_model_id=wrapper_model_id,
        track_request_usage=False,
    )

    assert response["choices"][0]["message"]["content"] == "query"
    assert mod.get_request_scoped_usage_anchor(
        request=pipe_request,
        chat_id=chat_id,
        message_id=message_id,
        wrapper_model_id=wrapper_model_id,
    ) == mod.UsageAnchor(
        assistant_message_id="request",
        input_tokens=40,
        stable_message_count=1,
        input_fingerprint="outer-forward",
        volatile_message_tokens=2,
    )


@pytest.mark.asyncio
async def test_pipe_rejects_when_core_context_compaction_is_enabled(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    class FakeConfig:
        @staticmethod
        async def get(key, default=None):
            if key == mod.CORE_CONTEXT_COMPACTION_ENABLE_CONFIG_KEY:
                return True
            return default

    async def validate_target_access(**kwargs):
        raise AssertionError("target access must not run when Core context compaction is enabled")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    install_fake_open_webui_config(monkeypatch, FakeConfig)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "core_context_compaction_conflict"
    assert "Disable Open WebUI Core context compaction" in result["error"]["message"]


def test_core_context_compaction_guard_matches_core_bool_semantics():
    assert mod._config_value_is_enabled(mod.CONFIG_VALUE_MISSING) is False
    assert mod._config_value_is_enabled(None) is False
    assert mod._config_value_is_enabled("false") is True


@pytest.mark.asyncio
async def test_pipe_forwards_below_threshold_to_decoded_target_with_metadata(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}

    async def validate_target_access(**kwargs):
        captured["validated_target"] = kwargs["target_model_id"]

    async def model_dict_from_request(request):
        return {"target.model": {"id": "target.model", "name": "Target"}}

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target.model")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    }

    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert captured["validated_target"] == "target.model"
    assert captured["forward_body"]["model"] == "target.model"
    assert captured["forward_body"]["metadata"] == pipe_metadata
    assert captured["forward_body"]["stream_options"]["include_usage"] is True
    assert captured["forward_body"]["messages"] == body["messages"]
    assert events == []


@pytest.mark.asyncio
async def test_pipe_reshapes_request_params_for_normal_ollama_target(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {
            "target-ollama": {
                "id": "target-ollama",
                "name": "Target Ollama",
                "owned_by": "ollama",
                "ollama": {},
            }
        }

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    result = await pipe.pipe(
        {
            "model": mod.build_wrapper_model_id("auto_compact", "target-ollama"),
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.3,
            "top_p": 0.8,
            "options": {"num_predict": 99},
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert captured["forward_body"]["model"] == "target-ollama"
    assert captured["forward_body"]["options"] == {
        "temperature": 0.3,
        "top_p": 0.8,
        "num_predict": 99,
    }
    assert "params" not in captured["forward_body"]


@pytest.mark.asyncio
async def test_pipe_injects_file_context_for_persisted_chat(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    install_fake_open_webui_user_model(monkeypatch)
    captured = {"target_file_calls": [], "source_events": [], "forward_attempts": 0}
    rows = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def estimate_body_tokens_async(body, **kwargs):
        captured["estimate_body"] = copy.deepcopy(body)
        return 10

    async def noop_initialize(**kwargs):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        captured["checkpoint_lookup_body"] = copy.deepcopy(kwargs["body"])
        return None

    async def generate_summary_file_context(*, request, user, prefix_files):
        captured["summary_prefix_files"] = copy.deepcopy(prefix_files)
        ids = ",".join(file["id"] for file in prefix_files)
        return f"<attached_file_contents>SUMMARY_CONTEXT:{ids}</attached_file_contents>"

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["summary_body"] = copy.deepcopy(form_data)
        return {"choices": [{"message": {"content": "summary"}}]}

    async def forward_target(**kwargs):
        captured["forward_attempts"] += 1
        captured["target_body"] = copy.deepcopy(kwargs["body"])
        if captured["forward_attempts"] == 1:
            raise mod.RetryableContextOverflow("too large before output")
        return {"ok": True}

    async def chat_completion_files_handler(request, rag_body, extra_params, user):
        captured["target_files"] = copy.deepcopy(rag_body["metadata"]["files"])
        captured["target_file_calls"].append(copy.deepcopy(rag_body["metadata"]["files"]))
        return rag_body, {
            "sources": [
                {
                    "source": {"id": file["id"], "name": file["name"]},
                    "document": [f"context for {file['id']}"],
                    "metadata": [{"source": file["id"]}],
                }
                for file in rag_body["metadata"]["files"]
            ]
        }

    async def apply_source_context_to_messages(request, messages, sources, last_user_msg):
        captured["target_last_user"] = last_user_msg
        ids = ",".join(source["source"]["id"] for source in sources)
        updated = copy.deepcopy(messages)
        updated[-1]["content"] = f"{updated[-1]['content']}\nTARGET_CONTEXT:{ids}"
        return updated

    def get_message_list(messages_map, message_id):
        result = []
        current = messages_map.get(message_id)
        while current is not None:
            result.append(current)
            parent_id = current.get("parentId")
            current = messages_map.get(parent_id) if parent_id else None
        result.reverse()
        return result

    def get_last_user_message(messages):
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content") or ""
        return ""

    db_messages = {
        "db-1": {
            "id": "db-1",
            "parentId": None,
            "role": "user",
            "content": "same text",
            "files": [_file("absorbed-file")],
        },
        "db-2": {
            "id": "db-2",
            "parentId": "db-1",
            "role": "assistant",
            "content": "old answer",
        },
        "message-1": {
            "id": "message-1",
            "parentId": "db-2",
            "role": "user",
            "content": "same text",
            "files": [_file("current-file")],
        },
    }

    class FakeChats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id):
            assert chat_id == pipe_metadata["chat_id"]
            return db_messages

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.chat_completion_files_handler = chat_completion_files_handler
    middleware_module.apply_source_context_to_messages = apply_source_context_to_messages
    # _load_chat_message_chain calls process_messages_with_output to align DB chain
    # with expanded body.messages. Provide an identity passthrough for the test.
    middleware_module.process_messages_with_output = lambda messages, reasoning_format=None: messages
    misc_module = types.ModuleType("open_webui.utils.misc")
    misc_module.get_message_list = get_message_list
    misc_module.get_last_user_message = get_last_user_message
    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = FakeChats

    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.misc", misc_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_generate_summary_file_context", generate_summary_file_context, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    metadata_files = [_file("absorbed-file"), _file("current-file"), _file("knowledge-file")]
    metadata = {
        **pipe_metadata,
        "files": metadata_files,
        "user_message": {"files": [_file("current-file")]},
    }
    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "same text"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "same text"},
        ],
    }

    async def event_emitter(event):
        if isinstance(event, dict) and event.get("type") == "source":
            captured["source_events"].append(copy.deepcopy(event))

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert captured["forward_attempts"] == 2
    assert captured["checkpoint_lookup_body"]["messages"][-1]["content"] == "same text"
    assert "TARGET_CONTEXT:absorbed-file,current-file,knowledge-file" in captured["estimate_body"]["messages"][-1]["content"]
    assert captured["summary_prefix_files"] == [_file("absorbed-file")]
    assert "SUMMARY_CONTEXT:absorbed-file" in captured["summary_body"]["messages"][-1]["content"]
    assert "files" not in captured["summary_body"]["metadata"]
    assert captured["target_file_calls"] == [
        [_file("absorbed-file"), _file("current-file"), _file("knowledge-file")],
        [_file("current-file"), _file("knowledge-file")],
    ]
    assert captured["target_files"] == [_file("current-file"), _file("knowledge-file")]
    assert [event["data"]["source"]["id"] for event in captured["source_events"]] == [
        "current-file",
        "knowledge-file",
    ]
    assert "TARGET_CONTEXT:current-file,knowledge-file" in captured["target_body"]["messages"][-1]["content"]
    assert "absorbed-file" not in captured["target_body"]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_pipe_non_streaming_merges_manual_rag_sources_into_response(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    install_fake_open_webui_user_model(monkeypatch)
    sources = [{"source": {"id": "file-1", "name": "manual.pdf"}, "document": ["manual context"]}]

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(
        *,
        body,
        metadata_files,
        request,
        user,
        event_emitter,
        file_context_enabled=True,
        emit_source_events=True,
        **kwargs,
    ):
        injected = copy.deepcopy(body)
        injected.setdefault("metadata", {})["sources"] = copy.deepcopy(sources)
        return injected

    async def forward_target(**kwargs):
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "target",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__={**pipe_metadata, "files": [{"id": "file-1", "type": "file", "name": "manual.pdf"}]},
        __event_emitter__=None,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert result["sources"] == sources


@pytest.mark.asyncio
async def test_pipe_non_streaming_source_event_failure_still_returns_response(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    install_fake_open_webui_user_model(monkeypatch)
    sources = [{"source": {"id": "file-1", "name": "manual.pdf"}, "document": ["manual context"]}]

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(
        *,
        body,
        metadata_files,
        request,
        user,
        event_emitter,
        file_context_enabled=True,
        emit_source_events=True,
        **kwargs,
    ):
        injected = copy.deepcopy(body)
        injected.setdefault("metadata", {})["sources"] = copy.deepcopy(sources)
        return injected

    async def forward_target(**kwargs):
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "target",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    async def event_emitter(event):
        raise RuntimeError("event channel closed")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__={**pipe_metadata, "files": [{"id": "file-1", "type": "file", "name": "manual.pdf"}]},
        __event_emitter__=event_emitter,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert result["sources"] == sources


@pytest.mark.asyncio
async def test_pipe_non_streaming_error_does_not_merge_or_emit_manual_rag_sources(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    install_fake_open_webui_user_model(monkeypatch)
    sources = [{"source": {"id": "file-1", "name": "manual.pdf"}, "document": ["manual context"]}]
    emitted = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(
        *,
        body,
        metadata_files,
        request,
        user,
        event_emitter,
        file_context_enabled=True,
        emit_source_events=True,
        **kwargs,
    ):
        injected = copy.deepcopy(body)
        injected.setdefault("metadata", {})["sources"] = copy.deepcopy(sources)
        return injected

    async def forward_target(**kwargs):
        return {"error": {"message": "provider failed", "code": "provider_error"}}

    async def event_emitter(event):
        emitted.append(copy.deepcopy(event))

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__={**pipe_metadata, "files": [{"id": "file-1", "type": "file", "name": "manual.pdf"}]},
        __event_emitter__=event_emitter,
    )

    assert result == {"error": {"message": "provider failed", "code": "provider_error"}}
    assert "sources" not in result
    assert emitted == []


@pytest.mark.asyncio
async def test_pipe_streaming_immediate_error_does_not_emit_manual_file_sources(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    install_fake_open_webui_user_model(monkeypatch)
    sources = [{"source": {"id": "file-1", "name": "manual.pdf"}, "document": ["manual context"]}]
    emitted = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(
        *,
        body,
        metadata_files,
        request,
        user,
        event_emitter,
        file_context_enabled=True,
        emit_source_events=True,
        **kwargs,
    ):
        injected = copy.deepcopy(body)
        injected.setdefault("metadata", {})["sources"] = copy.deepcopy(sources)
        return injected

    async def call_target_completion(**kwargs):
        return {"error": {"message": "provider failed", "code": "provider_error"}}

    async def event_emitter(event):
        emitted.append(copy.deepcopy(event))

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_call_target_completion", call_target_completion)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__={**pipe_metadata, "files": [{"id": "file-1", "type": "file", "name": "manual.pdf"}]},
        __event_emitter__=event_emitter,
    )

    assert isinstance(result, StreamingResponse)
    _ = [chunk async for chunk in result.body_iterator]
    assert emitted == []


@pytest.mark.asyncio
async def test_pipe_streaming_plaintext_non_json_error_does_not_emit_manual_file_sources(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    install_fake_open_webui_user_model(monkeypatch)
    sources = [{"source": {"id": "file-1", "name": "manual.pdf"}, "document": ["manual context"]}]
    emitted = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(
        *,
        body,
        metadata_files,
        request,
        user,
        event_emitter,
        file_context_enabled=True,
        emit_source_events=True,
        **kwargs,
    ):
        injected = copy.deepcopy(body)
        injected.setdefault("metadata", {})["sources"] = copy.deepcopy(sources)
        return injected

    async def call_target_completion(**kwargs):
        return PlainTextResponse("upstream connect error", status_code=200)

    async def event_emitter(event):
        emitted.append(copy.deepcopy(event))

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_call_target_completion", call_target_completion)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__={**pipe_metadata, "files": [{"id": "file-1", "type": "file", "name": "manual.pdf"}]},
        __event_emitter__=event_emitter,
    )

    assert isinstance(result, StreamingResponse)
    _ = [chunk async for chunk in result.body_iterator]
    assert emitted == []


@pytest.mark.asyncio
async def test_pipe_streaming_immediate_success_dict_emits_manual_file_sources(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    install_fake_open_webui_user_model(monkeypatch)
    sources = [{"source": {"id": "file-1", "name": "manual.pdf"}, "document": ["manual context"]}]
    emitted = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target", "owned_by": "openai"}}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(
        *,
        body,
        metadata_files,
        request,
        user,
        event_emitter,
        file_context_enabled=True,
        emit_source_events=True,
        **kwargs,
    ):
        injected = copy.deepcopy(body)
        injected.setdefault("metadata", {})["sources"] = copy.deepcopy(sources)
        return injected

    async def call_target_completion(**kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    async def event_emitter(event):
        emitted.append(copy.deepcopy(event))

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_call_target_completion", call_target_completion)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__={**pipe_metadata, "files": [{"id": "file-1", "type": "file", "name": "manual.pdf"}]},
        __event_emitter__=event_emitter,
    )

    assert isinstance(result, StreamingResponse)
    assert [event["data"]["source"]["id"] for event in emitted] == ["file-1"]


@pytest.mark.asyncio
async def test_pipe_skips_file_context_injection_for_query_generation_task(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    # TASK_MODEL may point at the AutoCompact wrapper; Core generate_queries
    # then re-enters Pipe.pipe with task=QUERY_GENERATION. The wrapper must NOT
    # inject target file context (which would recurse via
    # chat_completion_files_handler), and must forward normally.
    install_fake_open_webui_user_model(monkeypatch)
    captured = {"handler_calls": 0, "forward_body": None, "track_request_usage": None}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        captured["track_request_usage"] = kwargs["track_request_usage"]
        return {"ok": True}

    async def chat_completion_files_handler(request, rag_body, extra_params, user):
        captured["handler_calls"] += 1
        return rag_body, {"sources": []}

    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.chat_completion_files_handler = chat_completion_files_handler
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    metadata = {
        **pipe_metadata,
        "task": mod.TASKS.QUERY_GENERATION.value,
        "files": [_file("prefix-file"), _file("current-file")],
        "user_message": {"files": [_file("current-file")]},
    }
    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [{"role": "user", "content": "generate queries"}],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=metadata,
        __event_emitter__=None,
    )

    assert result == {"ok": True}
    assert captured["handler_calls"] == 0
    assert captured["forward_body"]["messages"] == body["messages"]
    assert captured["track_request_usage"] is False


@pytest.mark.asyncio
async def test_pipe_rejects_official_context_compaction_task_as_dict_error(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    class FakeConfig:
        @staticmethod
        async def get(key, default=None):
            return default

    async def validate_target_access(**kwargs):
        raise AssertionError("target access must not run for Core context compaction tasks")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    install_fake_open_webui_config(monkeypatch, FakeConfig)

    metadata = {
        **pipe_metadata,
        "task": mod.OFFICIAL_CONTEXT_COMPACTION_TASK,
        "files": [_file("prefix-file"), _file("current-file")],
        "user_message": {"files": [_file("current-file")]},
    }
    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1000
    pipe.valves.summary_model = "missing-summary"
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [{"role": "user", "content": "summarize the conversation"}],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=metadata,
        __event_emitter__=None,
    )

    assert result["error"]["code"] == "core_context_compaction_conflict"
    assert "Disable Open WebUI Core context compaction" in result["error"]["message"]


@pytest.mark.asyncio
async def test_inject_target_file_context_skips_manual_rag_when_reentry_guard_active(
    monkeypatch, pipe_request
):
    # When the same request re-enters _inject_target_file_context while the
    # injection guard is already active (generate_queries reentry), manual RAG
    # via chat_completion_files_handler must be skipped fail-closed. Metadata
    # files pruning still applies.
    captured = {"handler_calls": 0}

    async def chat_completion_files_handler(request, rag_body, extra_params, user):
        captured["handler_calls"] += 1
        return rag_body, {"sources": []}

    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.chat_completion_files_handler = chat_completion_files_handler
    middleware_module.apply_source_context_to_messages = lambda *args, **kwargs: args[1]
    misc_module = types.ModuleType("open_webui.utils.misc")
    misc_module.get_last_user_message = lambda messages: ""
    chats_module = types.ModuleType("open_webui.models.chats")

    class FakeChats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id):
            return {
                "msg-1": {
                    "id": "msg-1",
                    "parentId": None,
                    "role": "user",
                    "content": "x",
                    "files": [_file("absorbed-file")],
                },
                "msg-2": {"id": "msg-2", "parentId": "msg-1", "role": "assistant", "content": "y"},
                "message-1": {
                    "id": "message-1",
                    "parentId": "msg-2",
                    "role": "user",
                    "content": "active",
                },
            }

    chats_module.Chats = FakeChats

    def get_message_list(messages_map, message_id):
        result = []
        current = messages_map.get(message_id)
        while current is not None:
            result.append(current)
            parent_id = current.get("parentId")
            current = messages_map.get(parent_id) if parent_id else None
        result.reverse()
        return result

    misc_module.get_message_list = get_message_list
    middleware_module.process_messages_with_output = lambda messages, reasoning_format=None: messages
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.misc", misc_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    metadata_files = [_file("absorbed-file"), _file("current-file")]
    body = {
        "model": "target",
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
            {"role": "user", "content": "active"},
        ],
        "metadata": {"files": metadata_files},
    }

    token = mod.AUTO_COMPACT_FILE_CONTEXT_INJECTION_ACTIVE.set(True)
    try:
        result = await mod._inject_target_file_context(
            request=pipe_request,
            user={"id": "user-1"},
            body=body,
            chat_id="chat-1",
            current_message_id="message-1",
            compaction_prefix_count=2,
            metadata_files=metadata_files,
            metadata_user_message={"files": [_file("current-file")]},
            event_emitter=None,
            file_context_enabled=True,
            emit_source_events=False,
        )
        assert mod.AUTO_COMPACT_FILE_CONTEXT_INJECTION_ACTIVE.get() is True
    finally:
        mod.AUTO_COMPACT_FILE_CONTEXT_INJECTION_ACTIVE.reset(token)

    assert captured["handler_calls"] == 0
    # Pruning of absorbed prefix files still applied even when skipping RAG.
    retained = result["metadata"]["files"]
    assert [file["id"] for file in retained] == ["current-file"]


@pytest.mark.asyncio
async def test_inject_target_file_context_keeps_concurrent_sibling_tasks_independent(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    calls = []

    async def load_chat_message_chain(request, chat_id, current_message_id):
        return [{"id": current_message_id, "role": "user", "content": "active"}]

    def classify_files_for_target(**kwargs):
        return kwargs["metadata_files"]

    async def chat_completion_files_handler(request, rag_body, extra_params, user):
        calls.append(rag_body["model"])
        if rag_body["model"] == "first":
            first_started.set()
            await release_first.wait()
        else:
            second_started.set()
        return rag_body, {"sources": []}

    async def apply_source_context_to_messages(request, messages, sources, last_user_msg):
        return messages

    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.chat_completion_files_handler = chat_completion_files_handler
    middleware_module.apply_source_context_to_messages = apply_source_context_to_messages
    misc_module = types.ModuleType("open_webui.utils.misc")
    misc_module.get_last_user_message = lambda messages: ""
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.misc", misc_module)
    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "_classify_files_for_target", classify_files_for_target)

    metadata_files = [_file("current-file")]

    async def inject(model_id):
        return await mod._inject_target_file_context(
            request=pipe_request,
            user=pipe_user,
            body={
                "model": model_id,
                "messages": [{"role": "user", "content": "active"}],
                "metadata": {"files": metadata_files},
            },
            chat_id="chat-1",
            current_message_id=f"{model_id}-message",
            compaction_prefix_count=0,
            metadata_files=metadata_files,
            metadata_user_message={"files": metadata_files},
            event_emitter=None,
            file_context_enabled=True,
            emit_source_events=False,
        )

    first_task = asyncio.create_task(inject("first"))
    second_task = None
    task_results = []
    try:
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second_task = asyncio.create_task(inject("second"))
        await asyncio.wait_for(second_started.wait(), timeout=1)
    finally:
        release_first.set()
        task_results = await asyncio.gather(
            *(task for task in (first_task, second_task) if task is not None),
            return_exceptions=True,
        )

    assert calls == ["first", "second"]
    assert not [result for result in task_results if isinstance(result, BaseException)]


@pytest.mark.asyncio
async def test_inject_target_file_context_uses_non_transient_user_for_manual_rag(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
    patterns = mod.parse_transient_message_patterns(TRANSIENT_MARKER)
    captured = {}

    async def chat_completion_files_handler(request, rag_body, extra_params, user):
        captured["rag_messages"] = copy.deepcopy(rag_body["messages"])
        return rag_body, {
            "sources": [
                {
                    "source": {"id": "current-file", "name": "current.pdf"},
                    "document": ["current file context"],
                    "metadata": [{"source": "current-file"}],
                }
            ]
        }

    async def apply_source_context_to_messages(request, messages, sources, last_user_msg):
        captured["apply_messages"] = copy.deepcopy(messages)
        captured["last_user_msg"] = last_user_msg
        updated = copy.deepcopy(messages)
        updated[-1]["content"] = f"{updated[-1]['content']}\nTARGET_CONTEXT:current-file"
        return updated

    def get_last_user_message(messages):
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content") or ""
        return ""

    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.chat_completion_files_handler = chat_completion_files_handler
    middleware_module.apply_source_context_to_messages = apply_source_context_to_messages
    misc_module = types.ModuleType("open_webui.utils.misc")
    misc_module.get_last_user_message = get_last_user_message
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.misc", misc_module)

    transient_context = "<SYSTEM_CONTEXT>now: 10:00</SYSTEM_CONTEXT>"
    metadata_files = [_file("current-file")]
    body = {
        "model": "target",
        "messages": [
            {"role": "user", "content": "real question"},
            {"role": "user", "content": transient_context},
        ],
        "metadata": {"files": metadata_files},
    }

    result = await mod._inject_target_file_context(
        request=pipe_request,
        user=pipe_user,
        body=body,
        chat_id=None,
        current_message_id=None,
        compaction_prefix_count=0,
        metadata_files=metadata_files,
        metadata_user_message={"files": metadata_files},
        event_emitter=None,
        file_context_enabled=True,
        emit_source_events=False,
        transient_message_patterns=patterns,
    )

    assert captured["rag_messages"] == [{"role": "user", "content": "real question"}]
    assert captured["apply_messages"] == [{"role": "user", "content": "real question"}]
    assert captured["last_user_msg"] == "real question"
    assert result["messages"] == [
        {"role": "user", "content": "real question\nTARGET_CONTEXT:current-file"},
        {"role": "user", "content": transient_context},
    ]


def test_merge_rag_messages_preserves_appended_user_context_from_core_default_rag():
    patterns = mod.parse_transient_message_patterns(TRANSIENT_MARKER)
    transient_context = "<SYSTEM_CONTEXT>now: 10:00</SYSTEM_CONTEXT>"
    original = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "tool loop result"},
        {"role": "user", "content": transient_context},
    ]
    applied = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "tool loop result"},
        {"role": "user", "content": "TARGET_CONTEXT:current-file"},
    ]

    result = mod._merge_rag_messages_preserving_transient_users(original, applied, patterns)

    assert result == [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "tool loop result"},
        {"role": "user", "content": transient_context},
        {"role": "user", "content": "TARGET_CONTEXT:current-file"},
    ]


@pytest.mark.asyncio
async def test_inject_target_file_context_treats_content_type_images_as_images(
    monkeypatch, pipe_request
):
    captured = {"handler_calls": 0}

    async def chat_completion_files_handler(request, rag_body, extra_params, user):
        captured["handler_calls"] += 1
        return rag_body, {"sources": []}

    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.chat_completion_files_handler = chat_completion_files_handler
    middleware_module.apply_source_context_to_messages = lambda *args, **kwargs: args[1]
    misc_module = types.ModuleType("open_webui.utils.misc")
    misc_module.get_last_user_message = lambda messages: ""
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.misc", misc_module)

    image_file = {
        "id": "image-file",
        "content_type": "image/png",
        "name": "image.png",
        "url": "https://example.test/image.png",
    }
    body = {
        "model": "target",
        "messages": [{"role": "user", "content": "describe image"}],
        "metadata": {"files": [image_file]},
    }

    result = await mod._inject_target_file_context(
        request=pipe_request,
        user={"id": "user-1"},
        body=body,
        chat_id="chat-1",
        current_message_id="message-1",
        compaction_prefix_count=0,
        metadata_files=[image_file],
        metadata_user_message={"files": [image_file]},
        event_emitter=None,
        file_context_enabled=True,
        emit_source_events=False,
    )

    assert captured["handler_calls"] == 0
    assert result["metadata"]["files"] == [image_file]


@pytest.mark.asyncio
async def test_pipe_forwards_custom_model_missing_base_to_core_fallback_default(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    class FakeConfig:
        @staticmethod
        async def get(key):
            raise RuntimeError("config unavailable")

        @staticmethod
        async def get_many(*keys):
            raise RuntimeError("config unavailable")

    install_fake_open_webui_config(monkeypatch, FakeConfig)
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-ollama", "name": "Fallback", "owned_by": "ollama", "ollama": {}}
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-ollama,other")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-ollama": fallback_model}
    captured = {}
    checked_model_ids = []

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(id="workspace-preset", base_model_id="stale-openai")

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["forward_model"] = form_data["model"]
        captured["base_model_id"] = getattr(request, "base_model_id", None)
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert checked_model_ids == ["workspace-preset", "fallback-ollama"]
    assert captured["forward_model"] == "fallback-ollama"
    assert captured["base_model_id"] is None
    assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.parametrize("fallback_is_preset", [False, True])
@pytest.mark.asyncio
async def test_pipe_rejects_runtime_own_wrapper_as_missing_base_fallback(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
    fallback_is_preset,
):
    install_unavailable_open_webui_config(monkeypatch)
    own_wrapper_id = mod.build_wrapper_model_id("compact_alias", "fallback-target")
    fallback_model_id = "own-wrapper-preset" if fallback_is_preset else own_wrapper_id
    fallback_model = (
        {
            "id": fallback_model_id,
            "name": "Own Wrapper Preset",
            "owned_by": "openai",
            "info": {"base_model_id": own_wrapper_id},
        }
        if fallback_is_preset
        else {"id": own_wrapper_id, "name": "Own Wrapper", "owned_by": "openai"}
    )
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS=fallback_model_id)
    pipe_request.app.state.MODELS = {
        "workspace-preset": {
            "id": "workspace-preset",
            "name": "Workspace Preset",
            "owned_by": "openai",
            "preset": True,
            "info": {"base_model_id": "stale-openai"},
        },
        fallback_model_id: fallback_model,
    }

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(id=model_id, base_model_id="stale-openai")

    async def forward_target(**kwargs):
        raise AssertionError("own wrapper fallback must be rejected before forwarding")

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod.Pipe, "__module__", "function_compact_alias")

    result = await mod.Pipe().pipe(
        {
            "model": mod.build_wrapper_model_id("compact_alias", "workspace-preset"),
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "model_access_denied"


@pytest.mark.asyncio
async def test_pipe_applies_target_params_when_missing_base_uses_custom_model_fallback(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    target_params = {
        "temperature": 0.25,
        "top_p": 0.8,
        "max_tokens": 321,
        "system": "target system prompt is removed by Core fallback param handling",
        "function_calling": "native",
        "custom_params": {
            "vendor_flag": "enabled",
            "format": '{"type":"json"}',
        },
    }
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-ollama", "name": "Fallback", "owned_by": "ollama", "ollama": {}}
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-ollama")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-ollama": fallback_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=SimpleNamespace(**target_params),
            )

    async def check_model_access(user, model, db=None):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    def apply_params_to_form_data(form_data, model):
        params = copy.deepcopy(form_data.pop("params", {}) or {})
        custom_params = params.pop("custom_params", {}) or {}
        for key in (
            "stream_response",
            "stream_delta_chunk_size",
            "function_calling",
            "reasoning_tags",
            "compact_token_threshold",
            "system",
        ):
            params.pop(key, None)
        for key, value in list(custom_params.items()):
            if isinstance(value, str):
                try:
                    custom_params[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        params.update(custom_params)
        if model.get("owned_by") == "ollama":
            form_data["options"] = params
        else:
            form_data.update({key: value for key, value in params.items() if value is not None})
        return form_data

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["forward_body"] = copy.deepcopy(form_data)
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.apply_params_to_form_data = apply_params_to_form_data
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "params": {"top_p": 0.7, "presence_penalty": 0.1},
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["forward_body"]["model"] == "fallback-ollama"
    assert captured["forward_body"]["options"] == {
        "temperature": 0.25,
        "top_p": 0.7,
        "max_tokens": 321,
        "presence_penalty": 0.1,
        "vendor_flag": "enabled",
        "format": {"type": "json"},
    }
    assert "params" not in captured["forward_body"]
    assert "system" not in captured["forward_body"]
    assert captured["forward_body"]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_pipe_applies_request_params_when_missing_base_fallback_has_no_target_params(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-ollama", "name": "Fallback", "owned_by": "ollama", "ollama": {}}
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-ollama")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-ollama": fallback_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=None,
            )

    async def check_model_access(user, model, db=None):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    def apply_params_to_form_data(form_data, model):
        params = copy.deepcopy(form_data.pop("params", {}) or {})
        custom_params = params.pop("custom_params", {}) or {}
        params.update(custom_params)
        if model.get("owned_by") == "ollama":
            form_data["options"] = params
        else:
            form_data.update({key: value for key, value in params.items() if value is not None})
        return form_data

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["forward_body"] = copy.deepcopy(form_data)
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.apply_params_to_form_data = apply_params_to_form_data
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "params": {"temperature": 0.33, "top_p": None, "presence_penalty": 0.1},
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["forward_body"]["model"] == "fallback-ollama"
    assert captured["forward_body"]["options"] == {
        "temperature": 0.33,
        "presence_penalty": 0.1,
    }
    assert "params" not in captured["forward_body"]


@pytest.mark.asyncio
async def test_pipe_preserves_top_level_request_params_when_missing_base_uses_fallback_params(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    target_params = {
        "temperature": 0.25,
        "top_p": 0.8,
        "max_tokens": 321,
        "presence_penalty": 0.2,
        "response_format": {"type": "text"},
        "custom_params": {"vendor_flag": "target"},
    }
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-openai", "name": "Fallback", "owned_by": "openai", "openai": {}}
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-openai")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-openai": fallback_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=SimpleNamespace(**target_params),
            )

    async def check_model_access(user, model, db=None):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    def apply_params_to_form_data(form_data, model):
        params = copy.deepcopy(form_data.pop("params", {}) or {})
        custom_params = params.pop("custom_params", {}) or {}
        params.update(custom_params)
        form_data.update({key: value for key, value in params.items() if value is not None})
        return form_data

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["forward_body"] = copy.deepcopy(form_data)
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.apply_params_to_form_data = apply_params_to_form_data
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.9,
            "top_p": 0.6,
            "presence_penalty": 0.4,
            "response_format": {"type": "json_object"},
            "vendor_flag": "request",
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["forward_body"]["model"] == "fallback-openai"
    assert captured["forward_body"]["temperature"] == 0.9
    assert captured["forward_body"]["top_p"] == 0.6
    assert captured["forward_body"]["max_tokens"] == 321
    assert captured["forward_body"]["presence_penalty"] == 0.4
    assert captured["forward_body"]["response_format"] == {"type": "json_object"}
    assert captured["forward_body"]["vendor_flag"] == "request"
    assert "params" not in captured["forward_body"]


@pytest.mark.asyncio
async def test_pipe_preserves_top_level_ollama_provider_params_when_missing_base_uses_fallback_params(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    target_params = {
        "temperature": 0.25,
        "max_tokens": 321,
    }
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-ollama", "name": "Fallback", "owned_by": "ollama", "ollama": {}}
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-ollama")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-ollama": fallback_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=SimpleNamespace(**target_params),
            )

    async def check_model_access(user, model, db=None):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    def apply_params_to_form_data(form_data, model):
        params = copy.deepcopy(form_data.pop("params", {}) or {})
        if params.get("max_tokens") is not None:
            params["num_predict"] = params.pop("max_tokens")
        if model.get("owned_by") == "ollama":
            form_data["options"] = params
        else:
            form_data.update({key: value for key, value in params.items() if value is not None})
        return form_data

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["forward_body"] = copy.deepcopy(form_data)
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.apply_params_to_form_data = apply_params_to_form_data
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.9,
            "num_predict": 99,
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["forward_body"]["model"] == "fallback-ollama"
    assert captured["forward_body"]["options"]["temperature"] == 0.9
    assert captured["forward_body"]["options"]["num_predict"] == 99
    assert "params" not in captured["forward_body"]


def test_custom_model_fallback_params_drop_custom_max_tokens_when_ollama_num_predict_requested():
    from open_webui.utils.payload import convert_payload_openai_to_ollama

    patched = mod._apply_custom_model_fallback_params(
        {
            "model": "fallback-ollama",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "num_predict": 99,
        },
        fallback_model={"id": "fallback-ollama", "owned_by": "ollama"},
        target_params={
            "temperature": 0.25,
            "custom_params": {
                "max_tokens": 321,
            },
        },
    )

    assert patched["options"]["num_predict"] == 99
    assert "max_tokens" not in patched["options"]

    converted = convert_payload_openai_to_ollama(patched)
    assert converted["options"]["num_predict"] == 99
    assert "max_tokens" not in converted["options"]


def test_custom_model_fallback_params_drop_custom_max_tokens_when_ollama_params_num_predict_requested():
    from open_webui.utils.payload import convert_payload_openai_to_ollama

    patched = mod._apply_custom_model_fallback_params(
        {
            "model": "fallback-ollama",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "params": {"num_predict": 99},
        },
        fallback_model={"id": "fallback-ollama", "owned_by": "ollama"},
        target_params={
            "custom_params": {
                "max_tokens": 321,
            },
        },
    )

    assert patched["options"]["num_predict"] == 99
    assert "max_tokens" not in patched["options"]

    converted = convert_payload_openai_to_ollama(patched)
    assert converted["options"]["num_predict"] == 99
    assert "max_tokens" not in converted["options"]


def test_custom_model_fallback_params_keep_ollama_response_format_at_root_for_convert():
    from open_webui.utils.payload import convert_payload_openai_to_ollama

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        },
    }
    patched = mod._apply_custom_model_fallback_params(
        {
            "model": "fallback-ollama",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.9,
            "response_format": response_format,
        },
        fallback_model={"id": "fallback-ollama", "owned_by": "ollama"},
        target_params={
            "temperature": 0.25,
            "response_format": {"type": "text"},
            "custom_params": {
                "response_format": {"type": "json_object"},
            },
        },
    )

    assert patched["response_format"] == response_format
    assert patched["options"]["temperature"] == 0.9
    assert "response_format" not in patched["options"]

    converted = convert_payload_openai_to_ollama(patched)
    assert converted["format"] == response_format["json_schema"]["schema"]
    assert "response_format" not in converted.get("options", {})


@pytest.mark.asyncio
async def test_pipe_preserves_top_level_ollama_provider_params_when_fallback_has_no_target_params(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-ollama", "name": "Fallback", "owned_by": "ollama", "ollama": {}}
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-ollama")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-ollama": fallback_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=None,
            )

    async def check_model_access(user, model, db=None):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    def apply_params_to_form_data(form_data, model):
        params = copy.deepcopy(form_data.pop("params", {}) or {})
        if model.get("owned_by") == "ollama":
            form_data["options"] = params
        else:
            form_data.update({key: value for key, value in params.items() if value is not None})
        return form_data

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["forward_body"] = copy.deepcopy(form_data)
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.apply_params_to_form_data = apply_params_to_form_data
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.9,
            "top_p": 0.7,
            "presence_penalty": 0.2,
            "num_predict": 99,
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["forward_body"]["model"] == "fallback-ollama"
    assert captured["forward_body"]["options"] == {
        "temperature": 0.9,
        "top_p": 0.7,
        "presence_penalty": 0.2,
        "num_predict": 99,
    }
    assert "params" not in captured["forward_body"]


@pytest.mark.asyncio
async def test_pipe_preserves_existing_ollama_options_when_missing_base_uses_fallback_params(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    target_params = {
        "temperature": 0.25,
        "max_tokens": 321,
    }
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-ollama", "name": "Fallback", "owned_by": "ollama", "ollama": {}}
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-ollama")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-ollama": fallback_model}
    captured = {}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=SimpleNamespace(**target_params),
            )

    async def check_model_access(user, model, db=None):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    def apply_params_to_form_data(form_data, model):
        params = copy.deepcopy(form_data.pop("params", {}) or {})
        if params.get("max_tokens") is not None:
            params["num_predict"] = params.pop("max_tokens")
        if model.get("owned_by") == "ollama":
            form_data["options"] = params
        else:
            form_data.update({key: value for key, value in params.items() if value is not None})
        return form_data

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["forward_body"] = copy.deepcopy(form_data)
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.apply_params_to_form_data = apply_params_to_form_data
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "options": {"temperature": 0.9, "num_predict": 99},
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["forward_body"]["model"] == "fallback-ollama"
    assert captured["forward_body"]["options"] == {"temperature": 0.9, "num_predict": 99}
    assert "params" not in captured["forward_body"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_file_context", "fallback_file_context", "expected_injections"),
    [
        (False, True, 1),
        (True, False, 0),
    ],
)
async def test_pipe_uses_fallback_model_file_context_capability_after_missing_base_fallback(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
    target_file_context,
    fallback_file_context,
    expected_injections,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    file_item = {"id": "file-1", "type": "file", "name": "file-1.txt"}
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {
            "base_model_id": "stale-openai",
            "meta": {"capabilities": {"file_context": target_file_context}},
        },
    }
    fallback_model = {
        "id": "fallback-openai",
        "name": "Fallback",
        "owned_by": "openai",
        "openai": {},
        "info": {"meta": {"capabilities": {"file_context": fallback_file_context}}},
    }
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-openai")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-openai": fallback_model}
    captured = {"injections": 0}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=None,
            )

    async def check_model_access(user, model, db=None):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(**kwargs):
        captured["injections"] += 1
        assert kwargs["file_context_enabled"] is True
        return kwargs["body"]

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": kwargs["body"]["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__={
            **pipe_metadata,
            "files": [file_item],
            "user_message": {"files": [file_item]},
        },
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["forward_body"]["model"] == "fallback-openai"
    assert captured["injections"] == expected_injections


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected_file_context", "expected_injections"),
    [
        (False, 0),
        (True, 1),
    ],
)
async def test_pipe_resolves_arena_fallback_before_file_context_capability_check(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
    selected_file_context,
    expected_injections,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    file_item = {"id": "file-1", "type": "file", "name": "file-1.txt"}
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {
            "base_model_id": "stale-openai",
            "meta": {"capabilities": {"file_context": True}},
        },
    }
    fallback_arena = {
        "id": "fallback-arena",
        "name": "Fallback Arena",
        "owned_by": "arena",
        "arena": True,
        "info": {"meta": {"model_ids": ["arena-selected"]}},
    }
    selected_model = {
        "id": "arena-selected",
        "name": "Arena Selected",
        "owned_by": "openai",
        "openai": {},
        "info": {"meta": {"capabilities": {"file_context": selected_file_context}}},
    }
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-arena")
    pipe_request.app.state.MODELS = {
        "workspace-preset": target_model,
        "fallback-arena": fallback_arena,
        "arena-selected": selected_model,
    }
    captured = {"injections": 0}

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=None,
            )

    async def check_model_access(user, model, db=None):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(**kwargs):
        captured["injections"] += 1
        assert kwargs["file_context_enabled"] is True
        return kwargs["body"]

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": kwargs["body"]["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__={
            **pipe_metadata,
            "files": [file_item],
            "user_message": {"files": [file_item]},
        },
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["forward_body"]["model"] == "arena-selected"
    assert captured["forward_body"]["metadata"]["selected_model_id"] == "arena-selected"
    assert captured["injections"] == expected_injections


@pytest.mark.asyncio
async def test_pipe_checks_access_for_selected_arena_fallback_model(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_arena = {
        "id": "fallback-arena",
        "name": "Fallback Arena",
        "owned_by": "arena",
        "arena": True,
        "info": {"meta": {"model_ids": ["arena-selected"]}},
    }
    selected_model = {
        "id": "arena-selected",
        "name": "Arena Selected",
        "owned_by": "openai",
        "openai": {},
    }
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-arena")
    pipe_request.app.state.MODELS = {
        "workspace-preset": target_model,
        "fallback-arena": fallback_arena,
        "arena-selected": selected_model,
    }
    checked_model_ids = []

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=None,
            )

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])
        if model["id"] == "arena-selected":
            raise HTTPException(status_code=403, detail="Model not found")

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(**kwargs):
        raise AssertionError("selected arena model access denial must stop before file-context injection")

    async def forward_target(**kwargs):
        raise AssertionError("selected arena model access denial must stop before forwarding")

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "model_access_denied"
    assert checked_model_ids == ["workspace-preset", "arena-selected"]


@pytest.mark.asyncio
async def test_pipe_rejects_stale_arena_fallback_candidate_before_forwarding(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_arena = {
        "id": "fallback-arena",
        "name": "Fallback Arena",
        "owned_by": "arena",
        "arena": True,
        "info": {"meta": {"model_ids": ["stale-id", "arena-selected"]}},
    }
    selected_model = {
        "id": "arena-selected",
        "name": "Arena Selected",
        "owned_by": "openai",
        "openai": {},
    }
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-arena")
    pipe_request.app.state.MODELS = {
        "workspace-preset": target_model,
        "fallback-arena": fallback_arena,
        "arena-selected": selected_model,
    }
    checked_model_ids = []

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=None,
            )

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(**kwargs):
        raise AssertionError("stale arena fallback candidate must stop before file-context injection")

    async def forward_target(**kwargs):
        raise AssertionError("stale arena fallback candidate must stop before forwarding")

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod.random, "choice", lambda items: "stale-id")

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "model_access_denied"
    assert checked_model_ids == ["workspace-preset"]


@pytest.mark.asyncio
async def test_pipe_rejects_nested_arena_fallback_candidate_before_forwarding(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    install_unavailable_open_webui_config(monkeypatch)
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_arena = {
        "id": "fallback-arena",
        "name": "Fallback Arena",
        "owned_by": "arena",
        "arena": True,
        "info": {"meta": {"model_ids": ["nested-arena"]}},
    }
    nested_arena = {
        "id": "nested-arena",
        "name": "Nested Arena",
        "owned_by": "arena",
        "arena": True,
        "info": {"meta": {"model_ids": ["private-model"]}},
    }
    private_model = {
        "id": "private-model",
        "name": "Private Model",
        "owned_by": "openai",
        "openai": {},
    }
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-arena")
    pipe_request.app.state.MODELS = {
        "workspace-preset": target_model,
        "fallback-arena": fallback_arena,
        "nested-arena": nested_arena,
        "private-model": private_model,
    }
    checked_model_ids = []

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(
                id="workspace-preset",
                base_model_id="stale-openai",
                params=None,
            )

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def inject_target_file_context(**kwargs):
        raise AssertionError("nested arena fallback candidate must stop before file-context injection")

    async def forward_target(**kwargs):
        raise AssertionError("nested arena fallback candidate must stop before forwarding")

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "model_access_denied"
    assert checked_model_ids == ["workspace-preset"]


@pytest.mark.asyncio
async def test_pipe_forwards_custom_model_missing_base_to_config_default_model(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-config", "name": "Fallback", "owned_by": "openai", "openai": {}}
    pipe_request.app.state.config = SimpleNamespace()
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-config": fallback_model}
    captured = {"config_gets": []}
    checked_model_ids = []

    class FakeConfig:
        @staticmethod
        async def get(key):
            captured["config_gets"].append(key)
            if key == mod.TIKTOKEN_ENCODING_CONFIG_KEY:
                return None
            assert key == "ui.default_models"
            return "fallback-config,other"

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(id="workspace-preset", base_model_id="stale-openai")

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["forward_model"] = form_data["model"]
        captured["base_model_id"] = getattr(request, "base_model_id", None)
        return {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert captured["config_gets"] == [
        mod.TIKTOKEN_ENCODING_CONFIG_KEY,
        mod.CORE_CONTEXT_COMPACTION_ENABLE_CONFIG_KEY,
        "ui.default_models",
        "ui.default_models",
    ]
    assert checked_model_ids == ["workspace-preset", "fallback-config"]
    assert captured["forward_model"] == "fallback-config"
    assert captured["base_model_id"] is None
    assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_resolved_route_usage_anchor_hash_tracks_downstream_model_shaping(monkeypatch, pipe_request):
    import open_webui.routers.openai as openai_router

    pipe_request.app.state.MODELS = {
        "target": {"id": "target", "owned_by": "openai"},
        "base-a": {"id": "base-a", "owned_by": "openai", "urlIdx": 0},
        "base-b": {"id": "base-b", "owned_by": "openai", "urlIdx": 0},
    }
    pipe_request.app.state.OPENAI_MODELS = {
        "base-a": {"id": "base-a", "urlIdx": 0},
        "base-b": {"id": "base-b", "urlIdx": 0},
    }
    api_type = {"value": "chat_completions"}
    current = {
        "base_model_id": "base-a",
        "params": {"system": "system-a", "response_format": {"type": "text"}},
    }

    async def get_target_db_model_record(model_id):
        assert model_id == "target"
        return SimpleNamespace(
            base_model_id=current["base_model_id"],
            params=SimpleNamespace(model_dump=lambda: copy.deepcopy(current["params"])),
        )

    async def get_openai_connection(index):
        assert index in {0, 1}
        return f"http://provider-{index}", "key", {"api_type": api_type["value"]}

    monkeypatch.setattr(mod, "_get_target_db_model_record", get_target_db_model_record)
    monkeypatch.setattr(openai_router, "get_openai_connection", get_openai_connection)

    original = (await mod._resolve_core_chat_model_route(pipe_request, "target")).usage_anchor_shaping_hash
    assert original is not None

    pipe_request.app.state.OPENAI_MODELS["base-a"]["urlIdx"] = 1
    provider_cache_changed = (
        await mod._resolve_core_chat_model_route(pipe_request, "target")
    ).usage_anchor_shaping_hash
    assert provider_cache_changed is not None
    assert provider_cache_changed != original
    pipe_request.app.state.OPENAI_MODELS["base-a"]["urlIdx"] = 0

    api_type["value"] = "responses"
    transport_changed = (await mod._resolve_core_chat_model_route(pipe_request, "target")).usage_anchor_shaping_hash
    assert transport_changed is not None
    assert transport_changed != original
    api_type["value"] = "chat_completions"

    for replacement in (
        {"base_model_id": "base-b", "params": current["params"]},
        {"base_model_id": "base-a", "params": {**current["params"], "system": "system-b"}},
        {
            "base_model_id": "base-a",
            "params": {**current["params"], "response_format": {"type": "json_object"}},
        },
    ):
        current.clear()
        current.update(copy.deepcopy(replacement))
        changed = (await mod._resolve_core_chat_model_route(pipe_request, "target")).usage_anchor_shaping_hash
        assert changed is not None
        assert changed != original

    async def unknown_target_db_model_record(model_id):
        assert model_id == "target"
        return mod.TARGET_MODEL_RECORD_UNKNOWN

    monkeypatch.setattr(mod, "_get_target_db_model_record", unknown_target_db_model_record)
    assert (await mod._resolve_core_chat_model_route(pipe_request, "target")).usage_anchor_shaping_hash is None


@pytest.mark.asyncio
async def test_usage_anchor_system_identity_keeps_chat_variables_without_core_support(monkeypatch):
    monkeypatch.setattr(mod, "_render_chat_variables", None)

    assert (
        await mod._usage_anchor_resolved_system_identity(
            "Project={{ chat.variables.project }}",
            metadata={"chat_variables": {"project": "alpha"}},
            user=None,
        )
        == "Project={{ chat.variables.project }}"
    )


@pytest.mark.asyncio
async def test_resolved_route_usage_anchor_hash_normalizes_clock_but_tracks_expanded_system(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    import open_webui.models.groups as groups_module
    import open_webui.utils.task as task_module

    pipe_request.app.state.MODELS = {"target": {"id": "target", "owned_by": "openai"}}
    params = {
        "system": (
            "{{CURRENT_DATETIME}} {{USER_BIO}} {{USER_GROUPS}} {{CUSTOM}} "
            "{{ chat.variables.project }}"
        ),
    }
    group_names = ["group-a"]

    async def get_target_db_model_record(model_id):
        assert model_id == "target"
        return SimpleNamespace(
            base_model_id=None,
            params=SimpleNamespace(model_dump=lambda: copy.deepcopy(params)),
        )

    async def transport_profile(request, models, provider_model_id):
        assert provider_model_id == "target"
        return {"owned_by": "openai"}, frozenset()

    async def get_groups_by_member_id(user_id):
        assert user_id == pipe_user["id"]
        return [SimpleNamespace(name=name) for name in group_names]

    def render_chat_variables(system, variables, *, required=True):
        assert required is False
        return system.replace("{{ chat.variables.project }}", variables.get("project", ""))

    monkeypatch.setattr(mod, "_get_target_db_model_record", get_target_db_model_record)
    monkeypatch.setattr(mod, "_usage_anchor_transport_profile", transport_profile)
    if mod._render_chat_variables is None:
        monkeypatch.setattr(mod, "_render_chat_variables", render_chat_variables)
    monkeypatch.setattr(groups_module.Groups, "get_groups_by_member_id", get_groups_by_member_id)

    user = {**pipe_user, "bio": "bio-a"}
    metadata = {
        "chat_variables": {"project": "project-a"},
        "variables": {
            "{{CURRENT_DATETIME}}": "clock-a",
            "{{CUSTOM}}": "custom-a",
        }
    }

    async def shaping_hash(*, resolved_user=user, resolved_metadata=metadata):
        return (
            await mod._resolve_core_chat_model_route(
                pipe_request,
                "target",
                metadata=resolved_metadata,
                user=resolved_user,
            )
        ).usage_anchor_shaping_hash

    original = await shaping_hash()
    assert original is not None

    metadata["variables"]["{{CURRENT_DATETIME}}"] = "clock-b"
    assert await shaping_hash() == original
    assert await shaping_hash(resolved_user={**user, "bio": "bio-b"}) != original

    changed_metadata = copy.deepcopy(metadata)
    changed_metadata["variables"]["{{CUSTOM}}"] = "custom-b"
    assert await shaping_hash(resolved_metadata=changed_metadata) != original

    changed_metadata = copy.deepcopy(metadata)
    changed_metadata["chat_variables"]["project"] = "project-b"
    assert await shaping_hash(resolved_metadata=changed_metadata) != original

    # No user-variable counterpart: UserModel.variables is exclude=True, so the
    # Pipe-injected user never carries them and Core renders them to '' for our
    # forwards. Add identity coverage here once Core exposes them to Pipes.

    group_names[:] = ["group-b"]
    assert await shaping_hash() != original

    async def fail_prompt_template(template, user):
        raise RuntimeError("prompt expansion unavailable")

    monkeypatch.setattr(task_module, "prompt_template", fail_prompt_template)
    assert await shaping_hash() is None


@pytest.mark.asyncio
async def test_raw_fallback_usage_anchor_hash_tracks_its_db_override(monkeypatch, pipe_request):
    import open_webui.routers.openai as openai_router

    models = {
        "missing-preset": {"id": "missing-preset", "owned_by": "openai", "preset": True},
        "raw-fallback": {"id": "raw-fallback", "owned_by": "openai", "openai": {}, "urlIdx": 0},
    }
    pipe_request.app.state.MODELS = models
    pipe_request.app.state.OPENAI_MODELS = {
        "raw-fallback": {"id": "raw-fallback", "urlIdx": 0},
    }
    fallback_params = {"system": "fallback system a"}
    lookups = []

    async def get_target_db_model_record(model_id):
        lookups.append(model_id)
        if model_id == "missing-preset":
            return SimpleNamespace(base_model_id="missing-base", params=SimpleNamespace(model_dump=lambda: {}))
        assert model_id == "raw-fallback"
        return SimpleNamespace(
            base_model_id=None,
            params=SimpleNamespace(model_dump=lambda: copy.deepcopy(fallback_params)),
        )

    async def fallback_model_id(*args, **kwargs):
        return "raw-fallback"

    async def model_dict_from_request(request):
        return models

    async def get_openai_connection(index):
        assert index == 0
        return "http://provider", "key", {}

    monkeypatch.setattr(mod, "_get_target_db_model_record", get_target_db_model_record)
    monkeypatch.setattr(mod, "_custom_model_fallback_model_id_compatible", fallback_model_id)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(openai_router, "get_openai_connection", get_openai_connection)

    first = await mod._resolve_core_chat_model_route(pipe_request, "missing-preset")

    async def validate_runtime_access(**kwargs):
        assert kwargs["model_id"] == "raw-fallback"

    monkeypatch.setattr(mod, "_validate_chat_completion_runtime_model_access", validate_runtime_access)
    forwarded, selected_arena_model_id = await mod._resolve_arena_chat_model_route_with_access(
        request=pipe_request,
        user=None,
        models=models,
        route=first,
        original_model_id="missing-preset",
    )

    assert selected_arena_model_id is None
    assert forwarded.token_system_prompt == "fallback system a"
    assert lookups == ["missing-preset", "raw-fallback"]

    fallback_params["system"] = "fallback system b"
    second = await mod._resolve_core_chat_model_route(pipe_request, "missing-preset")

    assert first.model_id == second.model_id == "raw-fallback"
    assert first.token_system_prompt == "fallback system a"
    assert second.token_system_prompt == "fallback system b"
    assert first.usage_anchor_shaping_hash != second.usage_anchor_shaping_hash
    assert lookups == ["missing-preset", "raw-fallback", "missing-preset", "raw-fallback"]


@pytest.mark.asyncio
async def test_resolve_core_chat_model_route_falls_back_to_legacy_default_models_when_config_errors(
    monkeypatch,
    pipe_request,
):
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-legacy,other")
    pipe_request.app.state.MODELS = {
        "workspace-preset": {
            "id": "workspace-preset",
            "name": "Workspace Preset",
            "owned_by": "openai",
            "preset": True,
            "info": {"base_model_id": "stale-openai"},
        },
        "fallback-legacy": {"id": "fallback-legacy", "name": "Fallback", "owned_by": "openai", "openai": {}},
    }

    class FakeConfig:
        @staticmethod
        async def get(key):
            assert key == "ui.default_models"
            raise RuntimeError("config unavailable")

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(id="workspace-preset", base_model_id="stale-openai")

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)

    route = await mod._resolve_core_chat_model_route(pipe_request, "workspace-preset")

    assert route.model_id == "fallback-legacy"


@pytest.mark.asyncio
async def test_pipe_rejects_custom_model_fallback_default_without_user_access(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    class FakeConfig:
        @staticmethod
        async def get(key):
            raise RuntimeError("config unavailable")

        @staticmethod
        async def get_many(*keys):
            raise RuntimeError("config unavailable")

    install_fake_open_webui_config(monkeypatch, FakeConfig)
    target_model = {
        "id": "workspace-preset",
        "name": "Workspace Preset",
        "owned_by": "openai",
        "preset": True,
        "info": {"base_model_id": "stale-openai"},
    }
    fallback_model = {"id": "fallback-model", "name": "Fallback", "owned_by": "openai", "openai": {}}
    pipe_request.app.state.config = SimpleNamespace(DEFAULT_MODELS="fallback-model")
    pipe_request.app.state.MODELS = {"workspace-preset": target_model, "fallback-model": fallback_model}
    checked_model_ids = []

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            assert model_id == "workspace-preset"
            return SimpleNamespace(id="workspace-preset", base_model_id="stale-openai")

    async def check_model_access(user, model, db=None):
        checked_model_ids.append(model["id"])
        if model["id"] == "fallback-model":
            raise HTTPException(status_code=403, detail="denied fallback")

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        raise AssertionError("fallback target must not be called when fallback model access is denied")

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    utils_models_module = types.ModuleType("open_webui.utils.models")
    utils_models_module.check_model_access = check_model_access
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.models", utils_models_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "workspace-preset")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "model_access_denied"
    assert checked_model_ids == ["workspace-preset", "fallback-model"]


@pytest.mark.asyncio
async def test_pipe_uses_runtime_registered_id_for_decode_and_checkpoint_scope(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}

    async def validate_target_access(**kwargs):
        captured["access_pipe_function_id"] = kwargs["pipe_function_id"]
        captured["validated_target"] = kwargs["target_model_id"]

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def resolve_core_chat_model_route(request, model_id, *, pipe_function_id, **kwargs):
        captured["route_pipe_function_id"] = pipe_function_id
        return mod.CoreChatModelRoute(model_id=model_id)

    async def resolve_arena_chat_model_route_with_access(**kwargs):
        captured["arena_pipe_function_id"] = kwargs["pipe_function_id"]
        return kwargs["route"], None

    async def body_reusable_checkpoint_match(**kwargs):
        captured["checkpoint_pipe_function_id"] = kwargs["pipe_function_id"]
        return None

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_core_chat_model_route)
    monkeypatch.setattr(
        mod,
        "_resolve_arena_chat_model_route_with_access",
        resolve_arena_chat_model_route_with_access,
    )
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod.Pipe, "__module__", "function_compact_alias")

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100

    result = await pipe.pipe(
        {
            "model": mod.build_wrapper_model_id("compact_alias", "target"),
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert captured["access_pipe_function_id"] == "compact_alias"
    assert captured["validated_target"] == "target"
    assert captured["route_pipe_function_id"] == "compact_alias"
    assert captured["arena_pipe_function_id"] == "compact_alias"
    assert captured["checkpoint_pipe_function_id"] == "compact_alias"
    assert captured["forward_body"]["model"] == "target"


@pytest.mark.asyncio
async def test_pipe_rejects_wrapper_prefix_that_does_not_match_runtime_registered_id(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    async def forward_target(**kwargs):
        raise AssertionError("mismatched wrapper prefix must not be forwarded")

    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod.Pipe, "__module__", "function_compact_alias")

    result = await mod.Pipe().pipe(
        {
            "model": mod.build_wrapper_model_id("auto_compact", "target"),
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "invalid_wrapper_id"
    assert "compact_alias" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_forwards_metadata_with_unpickleable_core_values(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    future = asyncio.get_running_loop().create_future()
    metadata = {
        **pipe_metadata,
        "params": {"function_calling": "native"},
        "tools": {"tool": {"future": future}},
    }
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=metadata,
    )

    assert result == {"ok": True}
    assert captured["forward_body"]["metadata"]["tools"]["tool"]["future"] is future
    assert metadata["chat_id"] == "chat-1"


@pytest.mark.asyncio
async def test_pipe_forwarding_passes_open_webui_user_model_to_inner_completion(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["model"] = form_data["model"]
        captured["user_id"] = user.id
        captured["user_role"] = user.role
        captured["user_type"] = type(user)
        return StreamingResponse(
            iter([b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n']),
            media_type="text/event-stream",
        )

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    chunks = []
    async for chunk in result.body_iterator:
        chunks.append(chunk)

    assert chunks == [b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n']
    assert captured == {
        "model": "target",
        "user_id": "user-1",
        "user_role": "user",
        "user_type": fake_user_model,
    }


@pytest.mark.asyncio
async def test_pipe_non_streaming_forwards_to_decoded_target_completion(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    fake_user_model = install_fake_open_webui_user_model(monkeypatch)
    captured = {}

    async def validate_target_access(**kwargs):
        captured["validated_target"] = kwargs["target_model_id"]

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["request_state"] = request.state
        captured["form_data"] = form_data
        captured["user"] = user
        captured["bypass_filter"] = bypass_filter
        captured["bypass_system_prompt"] = bypass_system_prompt
        return {
            "id": "chatcmpl-target",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["validated_target"] == "target"
    assert captured["form_data"]["model"] == "target"
    assert captured["form_data"]["stream"] is False
    assert captured["form_data"]["metadata"] == pipe_metadata
    assert "stream_options" not in captured["form_data"]
    assert captured["user"].id == "user-1"
    assert type(captured["user"]) is fake_user_model
    assert captured["request_state"].bypass_filter is True
    assert captured["bypass_filter"] is True


@pytest.mark.asyncio
async def test_pipe_non_streaming_retries_with_compaction_after_context_error(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    install_fake_open_webui_user_model(monkeypatch)
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_or_create_checkpoint_summary(**kwargs):
        return "retry summary"

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        calls.append(copy_body(form_data))
        if len(calls) == 1:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "maximum context length exceeded",
                    }
                },
            )
        return {
            "id": "chatcmpl-target",
            "object": "chat.completion",
            "model": form_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    def copy_body(body):
        return {
            key: value
            for key, value in body.items()
            if key != "metadata"
        } | {"metadata": body.get("metadata")}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2
    assert calls[0]["messages"] == body["messages"]
    assert "retry summary" in calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_message",
    [
        "input length must be at least 1",
        "max_tokens must be less than the configured token limit",
    ],
)
async def test_pipe_non_streaming_does_not_retry_provider_validation_error(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
    provider_message,
):
    install_fake_open_webui_user_model(monkeypatch)
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_or_create_checkpoint_summary(**kwargs):
        raise AssertionError("validation errors must not trigger compaction retry")

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        calls.append(form_data)
        return JSONResponse(
            status_code=400,
            content={"error": {"message": provider_message}},
        )

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [{"role": "user", "content": "active"}],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"error": {"code": "provider_error", "message": provider_message}}
    assert len(calls) == 1


def test_chat_completion_response_requires_core_template(monkeypatch):
    misc_module = types.ModuleType("open_webui.utils.misc")
    monkeypatch.setitem(sys.modules, "open_webui.utils.misc", misc_module)

    with pytest.raises(ImportError):
        mod._chat_completion_message_response("target", "content")


@pytest.mark.asyncio
@pytest.mark.parametrize("error_marker", [False, {}])
async def test_non_streaming_forward_preserves_success_with_falsey_error_marker(monkeypatch, error_marker):
    response = {
        "error": error_marker,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }

    async def call_target_completion(**kwargs):
        return response

    monkeypatch.setattr(mod, "_call_target_completion", call_target_completion)

    result = await mod._forward_non_streaming_target(
        request=SimpleNamespace(state=SimpleNamespace()),
        user={},
        body={"model": "target"},
        track_request_usage=False,
    )

    assert result == response


@pytest.mark.asyncio
async def test_non_streaming_stream_response_error_stays_error_dict():
    response = StreamingResponse(
        iter(
            [
                b'data: {"error": {"code": "rate_limit_exceeded", "message": "rate limited"}}\n\n',
            ]
        ),
        media_type="text/event-stream",
    )

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    assert result == {"error": {"code": "provider_error", "message": "rate limited"}}


@pytest.mark.asyncio
async def test_non_streaming_stream_response_skips_done_chunk():
    response = StreamingResponse(
        iter(
            [
                b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        ),
        media_type="text/event-stream",
    )

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_non_streaming_stream_response_reassembles_split_sse_event():
    response = StreamingResponse(
        iter(
            [
                b'data: {"choices": [{"delta": ',
                b'{"content": "ok"}}]}\n\n',
            ]
        ),
        media_type="text/event-stream",
    )

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_non_streaming_stream_response_retries_split_sse_context_error():
    response = StreamingResponse(
        iter(
            [
                b'data: {"error": {"code": "context_length_',
                b'exceeded", "message": "too long"}}\n\n',
            ]
        ),
        media_type="text/event-stream",
    )

    with pytest.raises(mod.RetryableContextOverflow):
        await mod._coerce_non_streaming_completion_response(response, model_id="target")


@pytest.mark.asyncio
async def test_non_streaming_stream_response_preserves_tool_calls():
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"query":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"open webui"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    response = StreamingResponse(
        iter([f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks]),
        media_type="text/event-stream",
    )

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query":"open webui"}'},
            }
        ],
    }


@pytest.mark.asyncio
async def test_non_streaming_http_error_response_stays_error_dict_without_error_shape():
    response = JSONResponse(status_code=429, content={"message": "rate limited"})

    result = await mod._coerce_non_streaming_completion_response(response, model_id="target")

    assert result == {"error": {"code": "provider_error", "message": "rate limited"}}


@pytest.mark.asyncio
async def test_non_streaming_unknown_response_type_stays_error_dict():
    result = await mod._coerce_non_streaming_completion_response(object(), model_id="target")

    assert result == {
        "error": {
            "code": "unsupported_target_response",
            "message": "Unsupported target response type: object",
        }
    }


@pytest.mark.asyncio
async def test_pipe_rejects_missing_configured_summary_model(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def forward_target(**kwargs):
        raise AssertionError("target must not be called when summary_model is invalid")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.summary_model = "missing.summary"
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "invalid_summary_model"
    assert "missing.summary" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_skips_auto_compaction_for_stateful_responses_continuation(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    forwarded = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def estimate_body_tokens_async(body, **kwargs):
        raise AssertionError("stateful continuation must not be estimated")

    async def reusable_checkpoint_match(**kwargs):
        raise AssertionError("stateful continuation must not look up checkpoints")

    async def inject_target_file_context(**kwargs):
        raise AssertionError("stateful continuation must not reinject file context")

    async def compact_body(**kwargs):
        raise AssertionError("stateful continuation must not be compacted")

    async def forward_target(**kwargs):
        forwarded.append(copy.deepcopy(kwargs["body"]))
        assert kwargs["anchor_input"] is None
        assert kwargs["on_complete"] is None
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_compact_body", compact_body)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "previous_response_id": "resp-old",
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "x" * 10_000},
        ],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert len(forwarded) == 1
    assert forwarded[0]["previous_response_id"] == "resp-old"
    assert forwarded[0]["messages"] == body["messages"]


@pytest.mark.asyncio
async def test_pipe_does_not_retry_stateful_responses_after_context_error(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    forwarded = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def compact_body(**kwargs):
        raise AssertionError("stateful continuation must not be compacted after overflow")

    async def forward_target(**kwargs):
        forwarded.append(copy.deepcopy(kwargs["body"]))
        raise mod.RetryableContextOverflow("context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_compact_body", compact_body)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "previous_response_id": "resp-old",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "context_window_exceeded"
    assert len(forwarded) == 1
    assert forwarded[0]["previous_response_id"] == "resp-old"
    assert forwarded[0]["messages"] == body["messages"]


@pytest.mark.asyncio
async def test_pipe_reuses_existing_checkpoint_even_when_previous_compacted_usage_is_below_threshold(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    exact_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "older follow-up"},
        {"role": "assistant", "content": "older follow-up answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(exact_source),
        "source_hash": mod.compute_source_hash(exact_source),
        "summary_text": "existing summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    class ExistingCheckpointStore(ClaimCheckpointStore):
        async def lookup_any(self, **kwargs):
            assert kwargs["pipe_function_id"] == "auto_compact"
            return await super().lookup_any(**kwargs)

        async def lookup_ready(self, **kwargs):
            assert kwargs["pipe_function_id"] == "auto_compact"
            return await super().lookup_ready(**kwargs)

        async def claim_pending(self, row):
            raise AssertionError("exact checkpoint hit must not insert a new row")

        async def touch(self, checkpoint_id, *, now=None):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("exact checkpoint hit must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        raise AssertionError("checkpoint hit must not start a seed prefetch")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *exact_source,
            {"role": "user", "content": "active"},
        ],
    }
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert captured["touched"] == "checkpoint-1"
    forwarded = captured["forward_body"]
    assert "existing summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [{"role": "user", "content": "active"}]
    assert events == []


@pytest.mark.asyncio
async def test_pipe_newly_ready_checkpoint_does_not_reuse_previous_raw_candidate_usage(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    estimate_calls = []
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    delta_messages = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(parent_source),
        "source_hash": mod.compute_source_hash(parent_source),
        "summary_text": "existing parent summary",
        "summary_meta": {},
        "summary_token_count": 20,
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        estimate_calls.append(copy.deepcopy(body))
        return 40

    class ParentCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            raise AssertionError("safe checkpoint application must not create a foreground checkpoint")

        async def touch(self, checkpoint_id, *, now=None):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("safe checkpoint application must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    raw_messages = [
        *parent_source,
        *delta_messages,
        {"role": "user", "content": "active"},
    ]
    assistant_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ],
    }
    tool_result = {"role": "tool", "tool_call_id": "call-1", "content": "result"}
    mod.store_request_scoped_usage(
        request=pipe_request,
        chat_id=pipe_metadata["chat_id"],
        message_id=pipe_metadata["message_id"],
        wrapper_model_id=wrapper_id,
        usage={"total_tokens": 500, "input_tokens": 450, "output_tokens": 50},
        anchor_input=mod.UsageAnchorInput(
            stable_message_count=len(raw_messages),
            input_fingerprint=mod._compute_usage_anchor_input_fingerprint(
                {"model": "target", "messages": raw_messages},
                raw_messages,
            ),
            volatile_message_tokens=0,
        ),
    )
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                *raw_messages,
                assistant_tool_call,
                tool_result,
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(estimate_calls) == 1
    assert captured["touched"] == "checkpoint-1"
    forwarded = captured["forward_body"]
    assert "existing parent summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [
        *delta_messages,
        {"role": "user", "content": "active"},
        assistant_tool_call,
        tool_result,
    ]
    assert parent_source[0] not in forwarded["messages"]
    assert [event["data"]["action"] for event in events if event["type"] == "status"] == []


@pytest.mark.asyncio
async def test_pipe_same_checkpoint_tool_loop_reuses_request_usage_anchor(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(parent_source),
        "source_hash": mod.compute_source_hash(parent_source),
        "summary_text": "existing parent summary",
        "summary_meta": {},
        "summary_token_count": 20,
    }
    active = {"role": "user", "content": "active"}
    assistant_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ],
    }
    tool_result = {"role": "tool", "tool_call_id": "call-1", "content": "result"}
    previous_forward_messages = [
        mod.render_summary_message_from_checkpoint(
            checkpoint,
            historical_source_messages=parent_source,
        ),
        active,
    ]

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_target_db_model_record(model_id):
        assert model_id == "target"
        return None

    async def noop_initialize(**kwargs):
        return None

    async def estimate_message_sum(messages, *, request):
        if not messages:
            return 0
        if messages == [assistant_tool_call, tool_result]:
            return 20
        raise AssertionError(f"unexpected request-anchor suffix: {messages!r}")

    async def estimate_body_tokens_async(*args, **kwargs):
        raise AssertionError("same-checkpoint request anchor must avoid a full-body estimate")

    class ParentCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            raise AssertionError("safe checkpoint reuse must not create a foreground checkpoint")

        async def touch(self, checkpoint_id, *, now=None):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("same checkpoint reuse must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_target_db_model_record", get_target_db_model_record)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_estimate_message_token_sum_async", estimate_message_sum)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_known_openai_usage_anchor_transport(monkeypatch)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    mod.store_request_scoped_usage(
        request=pipe_request,
        chat_id=pipe_metadata["chat_id"],
        message_id=pipe_metadata["message_id"],
        wrapper_model_id=wrapper_id,
        usage={"input_tokens": 60, "output_tokens": 10},
        anchor_input=mod.UsageAnchorInput(
            stable_message_count=len(previous_forward_messages),
            input_fingerprint=mod._compute_usage_anchor_input_fingerprint(
                {"model": "target", "messages": previous_forward_messages},
                previous_forward_messages,
                usage_anchor_shaping_hash=_known_empty_model_shaping_hash(),
            ),
            volatile_message_tokens=0,
        ),
    )

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [*parent_source, active, assistant_tool_call, tool_result],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert captured["touched"] == "checkpoint-1"
    assert captured["forward_body"]["messages"] == [
        *previous_forward_messages,
        assistant_tool_call,
        tool_result,
    ]


@pytest.mark.asyncio
async def test_pipe_applies_parent_checkpoint_when_hard_observed_total_but_checkpoint_estimate_safe(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    estimate_calls = []
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    delta_messages = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(parent_source),
        "source_hash": mod.compute_source_hash(parent_source),
        "summary_text": "existing parent summary",
        "summary_meta": {},
        "summary_token_count": 20,
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        estimate_calls.append(copy.deepcopy(body))
        return 40

    class ParentCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            raise AssertionError("safe checkpoint application must not create a foreground checkpoint")

        async def touch(self, checkpoint_id, *, now=None):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("safe checkpoint application must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                *parent_source,
                *delta_messages,
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(estimate_calls) == 1
    assert captured["touched"] == "checkpoint-1"
    forwarded = captured["forward_body"]
    assert "existing parent summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [*delta_messages, {"role": "user", "content": "active"}]
    assert parent_source[0] not in forwarded["messages"]
    assert events == []


@pytest.mark.asyncio
async def test_pipe_applies_parent_checkpoint_when_estimate_just_below_hard_no_foreground(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    estimate_calls = []
    summary_inputs = []
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    delta_messages = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(parent_source),
        "source_hash": mod.compute_source_hash(parent_source),
        "summary_text": "existing parent summary",
        "summary_meta": {},
        "summary_token_count": 80,
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        estimate_calls.append(copy.deepcopy(body))
        return 95

    class ParentCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            raise AssertionError("safe checkpoint application must not create a foreground checkpoint")

        async def touch(self, checkpoint_id, *, now=None):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("safe checkpoint application must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.soft_trigger_ratio = 0
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                *parent_source,
                *delta_messages,
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(estimate_calls) == 1
    assert captured["touched"] == "checkpoint-1"
    forwarded = captured["forward_body"]
    assert "existing parent summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [*delta_messages, {"role": "user", "content": "active"}]
    assert len(summary_inputs) == 0
    assert events == []


async def _run_parent_checkpoint_decision_case(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
    *,
    persisted_total_tokens,
    checkpoint_applied_estimate,
    trigger_input_tokens,
    soft_trigger_ratio,
    foreground_summary_text=None,
):
    captured = {}
    estimate_calls = []
    summary_inputs = []
    prefetch_calls = []
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    delta_messages = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(parent_source),
        "source_hash": mod.compute_source_hash(parent_source),
        "summary_text": "existing parent summary",
        "summary_meta": {},
        "summary_token_count": 20,
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        estimate_calls.append(copy.deepcopy(body))
        return checkpoint_applied_estimate

    class ParentCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            if foreground_summary_text is None:
                raise AssertionError("safe checkpoint application must not create a foreground checkpoint")
            return await super().claim_pending(row)

        async def touch(self, checkpoint_id, *, now=None):
            captured["touched"] = checkpoint_id
            return True

    existing_store = ParentCheckpointStore([checkpoint])

    async def generate_summary_text(**kwargs):
        if foreground_summary_text is None:
            raise AssertionError("safe checkpoint application must not call the summary model")
        summary_inputs.append(kwargs["source_messages"])
        return foreground_summary_text

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        stored = {
            key: copy.deepcopy(value)
            for key, value in kwargs.items()
            if key not in {"event_emitter", "request", "user"}
        }
        stored["event_emitter"] = kwargs.get("event_emitter")
        prefetch_calls.append(stored)
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: existing_store)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = trigger_input_tokens
    pipe.valves.soft_trigger_ratio = soft_trigger_ratio
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                *parent_source,
                *delta_messages,
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    return {
        "result": result,
        "captured": captured,
        "estimate_calls": estimate_calls,
        "summary_inputs": summary_inputs,
        "prefetch_calls": prefetch_calls,
        "parent_source": parent_source,
        "delta_messages": delta_messages,
        "store": existing_store,
        "events": events,
    }


@pytest.mark.asyncio
async def test_pipe_skips_compaction_and_prefetch_when_checkpoint_estimate_below_soft(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    case = await _run_parent_checkpoint_decision_case(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        persisted_total_tokens=500,
        checkpoint_applied_estimate=5,
        trigger_input_tokens=100,
        soft_trigger_ratio=0.5,
    )

    assert case["result"] == {"ok": True}
    forwarded = case["captured"]["forward_body"]
    assert "existing parent summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [*case["delta_messages"], {"role": "user", "content": "active"}]
    assert case["parent_source"][0] not in forwarded["messages"]
    assert case["events"] == []
    assert case["prefetch_calls"] == []


@pytest.mark.asyncio
async def test_pipe_skips_prefetch_when_checkpoint_estimate_below_soft_despite_high_usage(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    case = await _run_parent_checkpoint_decision_case(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        persisted_total_tokens=500,
        checkpoint_applied_estimate=40,
        trigger_input_tokens=1000,
        soft_trigger_ratio=0.1,
    )

    assert case["result"] == {"ok": True}
    forwarded = case["captured"]["forward_body"]
    assert "existing parent summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [*case["delta_messages"], {"role": "user", "content": "active"}]
    assert case["prefetch_calls"] == []
    assert case["events"] == []


@pytest.mark.asyncio
async def test_pipe_prefetches_when_checkpoint_estimate_between_soft_and_hard(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    case = await _run_parent_checkpoint_decision_case(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        persisted_total_tokens=50,
        checkpoint_applied_estimate=500,
        trigger_input_tokens=1000,
        soft_trigger_ratio=0.1,
    )

    assert case["result"] == {"ok": True}
    assert case["summary_inputs"] == []
    forwarded = case["captured"]["forward_body"]
    assert "existing parent summary" in forwarded["messages"][0]["content"]
    assert len(case["prefetch_calls"]) == 1
    assert case["prefetch_calls"][0]["trigger_estimated_tokens"] == 500
    assert "trigger_observed_tokens" not in case["prefetch_calls"][0]


@pytest.mark.asyncio
async def test_pipe_rechecks_ready_checkpoint_before_soft_prefetch(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    injected_messages = []
    injection_prefix_counts = []
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    delta_messages = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(parent_source),
        "source_hash": mod.compute_source_hash(parent_source),
        "summary_text": "just-finished parent summary",
        "summary_meta": {},
        "summary_token_count": 20,
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        text = json.dumps(body.get("messages", []))
        return 40 if "just-finished parent summary" in text else 500

    class LateReadyCheckpointStore(ClaimCheckpointStore):
        def __init__(self):
            super().__init__([])
            self.parent_lookup_count = 0

        async def find_longest_parent(self, **kwargs):
            self.parent_lookup_count += 1
            if self.parent_lookup_count == 1:
                self.rows.append(dict(checkpoint))
                return None
            return await super().find_longest_parent(**kwargs)

        async def claim_pending(self, row):
            raise AssertionError("late-ready parent checkpoint must make the new soft prefetch redundant")

    store = LateReadyCheckpointStore()

    async def generate_summary_text(**kwargs):
        raise AssertionError("late-ready parent checkpoint must be applied instead of generating a fresh summary")

    async def inject_target_file_context(**kwargs):
        body = copy.deepcopy(kwargs["body"])
        messages = body["messages"]
        injected_messages.append(copy.deepcopy(messages))
        injection_prefix_counts.append(kwargs["compaction_prefix_count"])
        for message in reversed(messages):
            if message.get("role") == "user":
                message["content"] += "\nFILE_CONTEXT"
                break
        return body

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        raise AssertionError("soft prefetch must be skipped after late checkpoint revalidation")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1000
    pipe.valves.soft_trigger_ratio = 0.1
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                *parent_source,
                *delta_messages,
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    forwarded = captured["forward_body"]
    assert "just-finished parent summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [
        *delta_messages,
        {"role": "user", "content": "active\nFILE_CONTEXT"},
    ]
    assert injection_prefix_counts == [0, len(parent_source)]
    assert "FILE_CONTEXT" not in json.dumps(injected_messages[1])
    assert json.dumps(forwarded["messages"]).count("FILE_CONTEXT") == 1
    assert store.claimed_rows == []
    assert events == []


@pytest.mark.asyncio
async def test_pipe_keeps_forwarding_when_soft_only_late_checkpoint_recheck_fails(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    calls = {"lookup": 0, "prefetch": 0}
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def body_reusable_checkpoint_match(**kwargs):
        calls["lookup"] += 1
        if calls["lookup"] == 1:
            return None
        raise RuntimeError("checkpoint db flaked during soft recheck")

    async def estimate_body_tokens_async(body, **kwargs):
        return 500

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        calls["prefetch"] += 1
        raise AssertionError("soft prefetch must be skipped when its late recheck fails")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1000
    pipe.valves.soft_trigger_ratio = 0.1
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == body["messages"]
    assert calls == {"lookup": 2, "prefetch": 0}


@pytest.mark.asyncio
async def test_pipe_keeps_forwarding_when_prefetch_launch_recheck_fails(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    calls = {"lookup": 0, "prefetch": 0}
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def body_reusable_checkpoint_match(**kwargs):
        calls["lookup"] += 1
        if calls["lookup"] <= 2:
            return None
        raise RuntimeError("checkpoint db flaked before prefetch launch")

    async def estimate_body_tokens_async(body, **kwargs):
        return 500

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        calls["prefetch"] += 1
        raise AssertionError("soft prefetch must be skipped when launch recheck fails")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1000
    pipe.valves.soft_trigger_ratio = 0.1
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == body["messages"]
    assert calls == {"lookup": 3, "prefetch": 0}


@pytest.mark.asyncio
async def test_pipe_rechecks_ready_checkpoint_after_token_status_before_soft_prefetch(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    delta_messages = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    checkpoint = {
        "id": "checkpoint-after-status",
        "state": "ready",
        "source_message_count": len(parent_source),
        "source_hash": mod.compute_source_hash(parent_source),
        "summary_text": "status-gap parent summary",
        "summary_meta": {},
        "summary_token_count": 20,
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        text = json.dumps(body.get("messages", []))
        return 40 if "status-gap parent summary" in text else 500

    class StatusReadyCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            raise AssertionError("status-gap ready checkpoint must make soft prefetch redundant")

    store = StatusReadyCheckpointStore([])

    async def generate_summary_text(**kwargs):
        raise AssertionError("status-gap ready checkpoint must be applied instead of generating a fresh summary")

    async def inject_target_file_context(**kwargs):
        return kwargs["body"]

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        raise AssertionError("soft prefetch must be skipped when checkpoint becomes ready during status emit")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1000
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.token_status_visibility = "always"
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)
        if not store.rows:
            store.rows.append(dict(checkpoint))

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                *parent_source,
                *delta_messages,
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    forwarded = captured["forward_body"]
    assert "status-gap parent summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [*delta_messages, {"role": "user", "content": "active"}]
    assert store.claimed_rows == []
    assert events


@pytest.mark.asyncio
async def test_pipe_rechecks_better_checkpoint_after_token_status_when_parent_was_already_reusable(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    delta_messages = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    exact_source = [*parent_source, *delta_messages]
    parent_checkpoint = {
        "id": "initial-parent-checkpoint",
        "state": "ready",
        "source_message_count": len(parent_source),
        "source_hash": mod.compute_source_hash(parent_source),
        "summary_text": "initial parent summary",
        "summary_meta": {},
        "summary_token_count": 20,
    }
    exact_checkpoint = {
        "id": "status-gap-exact-checkpoint",
        "state": "ready",
        "source_message_count": len(exact_source),
        "source_hash": mod.compute_source_hash(exact_source),
        "summary_text": "status-gap exact summary",
        "summary_meta": {},
        "summary_token_count": 20,
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        text = json.dumps(body.get("messages", []))
        return 40 if "status-gap exact summary" in text else 500

    class StatusBetterCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            raise AssertionError("late exact checkpoint must make soft prefetch redundant")

    store = StatusBetterCheckpointStore([dict(parent_checkpoint)])

    async def generate_summary_text(**kwargs):
        raise AssertionError("late exact checkpoint must be applied instead of generating a fresh summary")

    async def inject_target_file_context(**kwargs):
        return kwargs["body"]

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        raise AssertionError("soft prefetch must be skipped when a better checkpoint becomes ready during status emit")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1000
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.token_status_visibility = "always"
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)
        if not any(row.get("id") == exact_checkpoint["id"] for row in store.rows):
            store.rows.append(dict(exact_checkpoint))

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                *exact_source,
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    forwarded = captured["forward_body"]
    assert "status-gap exact summary" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1:] == [{"role": "user", "content": "active"}]
    assert store.claimed_rows == []
    assert events


@pytest.mark.asyncio
async def test_pipe_compacts_foreground_when_checkpoint_estimate_at_or_above_hard(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    case = await _run_parent_checkpoint_decision_case(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        persisted_total_tokens=50,
        checkpoint_applied_estimate=150,
        trigger_input_tokens=100,
        soft_trigger_ratio=0.1,
        foreground_summary_text="fresh checkpoint summary",
    )

    assert case["result"] == {"ok": True}
    assert len(case["summary_inputs"]) == 1
    forwarded = case["captured"]["forward_body"]
    assert "fresh checkpoint summary" in forwarded["messages"][0]["content"]
    completed = case["store"].completed_rows[0]
    assert completed["parent_checkpoint_id"] == "checkpoint-1"
    assert [event["data"]["action"] for event in case["events"] if event["type"] == "status"] == [
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]


@pytest.mark.asyncio
async def test_pipe_raw_usage_alone_does_not_trigger_compaction_without_candidate_estimate(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    prefetch_calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_body_tokens_async(*args, **kwargs):
        return 5

    async def generate_summary_text(**kwargs):
        raise AssertionError("raw observed usage alone must not trigger foreground compaction")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        prefetch_calls.append(kwargs)
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.soft_trigger_ratio = 0.1
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == body["messages"]
    assert prefetch_calls == []
    assert events == []


def _install_threshold_decision_stubs(monkeypatch, captured, *, total_tokens):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_body_tokens_async(*args, **kwargs):
        return total_tokens

    async def get_or_create_checkpoint_summary(**kwargs):
        return "summary text"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)


def _compactable_body():
    return {
        "model": mod.build_wrapper_model_id("auto_compact", "target"),
        "stream": True,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }


@pytest.mark.asyncio
async def test_pipe_override_lowers_threshold_and_triggers_compaction(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}
    _install_threshold_decision_stubs(monkeypatch, captured, total_tokens=500)

    pipe = mod.Pipe()
    # Global default would NOT compact 500 tokens, but the per-model override (100) does.
    pipe.valves.trigger_input_tokens = 100000
    pipe.valves.per_model_overrides_json = json.dumps(
        {"overrides": [{"model_patterns": ["target"], "trigger_input_tokens": 100}]}
    )

    events = []

    async def event_emitter(event):
        events.append(event)

    await pipe.pipe(
        _compactable_body(),
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert [event["data"]["action"] for event in events if event["type"] == "status"] == [
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]
    assert "summary text" in captured["forward_body"]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_pipe_override_raises_threshold_and_skips_compaction(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}
    _install_threshold_decision_stubs(monkeypatch, captured, total_tokens=500)

    pipe = mod.Pipe()
    # Global default would compact 500 tokens, but the per-model override (1000000) keeps it forwarding.
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.per_model_overrides_json = json.dumps(
        {"overrides": [{"model_patterns": ["target"], "trigger_input_tokens": 1000000}]}
    )

    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        _compactable_body(),
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert events == []
    assert captured["forward_body"]["messages"] == _compactable_body()["messages"]


@pytest.mark.asyncio
async def test_pipe_override_non_match_uses_global_trigger_input_tokens(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}
    _install_threshold_decision_stubs(monkeypatch, captured, total_tokens=500)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    # Override targets a different model, so the global default (100) is used and 500 tokens compacts.
    pipe.valves.per_model_overrides_json = json.dumps(
        {"overrides": [{"model_patterns": ["other-*"], "trigger_input_tokens": 1000000}]}
    )

    events = []

    async def event_emitter(event):
        events.append(event)

    await pipe.pipe(
        _compactable_body(),
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert [event["data"]["action"] for event in events if event["type"] == "status"] == [
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]


@pytest.mark.asyncio
async def test_pipe_empty_overrides_json_preserves_global_threshold_behavior(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}
    _install_threshold_decision_stubs(monkeypatch, captured, total_tokens=500)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    assert pipe.valves.per_model_overrides_json == ""

    events = []

    async def event_emitter(event):
        events.append(event)

    await pipe.pipe(
        _compactable_body(),
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert [event["data"]["action"] for event in events if event["type"] == "status"] == [
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]


@pytest.mark.asyncio
async def test_pipe_returns_error_when_overrides_json_invalid_at_runtime(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def forward_target(**kwargs):
        captured["forwarded"] = True
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    # Bypass save-time validation (validate_assignment is off) to simulate a tampered stored value.
    pipe.valves.per_model_overrides_json = "not json"

    result = await pipe.pipe(
        _compactable_body(),
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=None,
    )

    assert result["error"]["code"] == "invalid_per_model_overrides"
    assert "forwarded" not in captured


@pytest.mark.asyncio
async def test_task_template_selection_matches_core_whitespace_rules(monkeypatch, pipe_request):
    class FakeConfig:
        @staticmethod
        async def get(key):
            raise RuntimeError("config unavailable")

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    core_config_module = types.ModuleType("open_webui.config")
    core_config_module.DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE = "Default query {{MESSAGES}}"
    core_config_module.DEFAULT_AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE = "Default autocomplete {{PROMPT}}"
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "open_webui.config", core_config_module)
    import open_webui as core_package

    monkeypatch.setattr(core_package, "config", core_config_module, raising=False)
    pipe_request.app.state.config = SimpleNamespace(
        TITLE_GENERATION_PROMPT_TEMPLATE="   ",
        QUERY_GENERATION_PROMPT_TEMPLATE="   ",
        AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE="   ",
    )

    title_template = await mod._task_template_from_request(
        pipe_request,
        mod.TASK_PROMPT_SPECS[mod.TASKS.TITLE_GENERATION.value],
    )
    query_template = await mod._task_template_from_request(
        pipe_request,
        mod.TASK_PROMPT_SPECS[mod.TASKS.QUERY_GENERATION.value],
    )
    autocomplete_template = await mod._task_template_from_request(
        pipe_request,
        mod.TASK_PROMPT_SPECS[mod.TASKS.AUTOCOMPLETE_GENERATION.value],
    )

    assert title_template == "   "
    assert query_template != "   "
    assert "{{MESSAGES" in query_template
    assert autocomplete_template != "   "
    assert "{{PROMPT}}" in autocomplete_template


@pytest.mark.asyncio
async def test_render_task_prompt_uses_db_config_templates_for_all_supported_tasks(
    monkeypatch, pipe_request, pipe_user
):
    expected_config_keys = {
        mod.TASKS.TITLE_GENERATION.value: "task.title.prompt_template",
        mod.TASKS.FOLLOW_UP_GENERATION.value: "task.follow_up.prompt_template",
        mod.TASKS.TAGS_GENERATION.value: "task.tags.prompt_template",
        mod.TASKS.QUERY_GENERATION.value: "task.query.prompt_template",
        mod.TASKS.IMAGE_PROMPT_GENERATION.value: "task.image.prompt_template",
        mod.TASKS.AUTOCOMPLETE_GENERATION.value: "task.autocomplete.prompt_template",
    }
    captured_templates = {}
    requested_keys = []

    class FakeConfig:
        @staticmethod
        async def get(key):
            requested_keys.append(key)
            return f"db::{key}"

    task_module = types.ModuleType("open_webui.utils.task")

    for task_name, spec in mod.TASK_PROMPT_SPECS.items():
        if task_name == mod.TASKS.AUTOCOMPLETE_GENERATION.value:

            async def autocomplete_builder(template, prompt, messages, type, user, *, _task_name=task_name):
                captured_templates[_task_name] = template
                return f"rendered::{_task_name}::{template}"

            setattr(task_module, spec.builder_name, autocomplete_builder)
        else:

            async def builder(template, messages, user, *, _task_name=task_name):
                captured_templates[_task_name] = template
                return f"rendered::{_task_name}::{template}"

            setattr(task_module, spec.builder_name, builder)

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.task", task_module)
    import open_webui.utils as core_utils

    monkeypatch.setattr(core_utils, "task", task_module, raising=False)
    pipe_request.app.state.config = SimpleNamespace(
        **{spec.config_attr: f"legacy::{task_name}" for task_name, spec in mod.TASK_PROMPT_SPECS.items()}
    )

    for task_name in mod.TASK_PROMPT_SPECS:
        task_body = {
            "messages": [{"role": "user", "content": f"history for {task_name}"}],
            "prompt": f"prompt for {task_name}",
            "type": "sentence",
        }
        rendered = await mod._render_task_prompt_from_messages(
            request=pipe_request,
            user=pipe_user,
            metadata={"task": task_name, "task_body": task_body},
            messages=task_body["messages"],
        )
        expected_template = f"db::{expected_config_keys[task_name]}"
        assert rendered == f"rendered::{task_name}::{expected_template}"
        assert captured_templates[task_name] == expected_template

    assert requested_keys == [expected_config_keys[task_name] for task_name in mod.TASK_PROMPT_SPECS]


@pytest.mark.asyncio
async def test_render_task_prompt_falls_back_to_legacy_templates_for_all_supported_tasks(
    monkeypatch, pipe_request, pipe_user
):
    captured_templates = {}

    class FakeConfig:
        @staticmethod
        async def get(key):
            raise RuntimeError("config unavailable")

    task_module = types.ModuleType("open_webui.utils.task")

    for task_name, spec in mod.TASK_PROMPT_SPECS.items():
        if task_name == mod.TASKS.AUTOCOMPLETE_GENERATION.value:

            async def autocomplete_builder(template, prompt, messages, type, user, *, _task_name=task_name):
                captured_templates[_task_name] = template
                return f"rendered::{_task_name}::{template}"

            setattr(task_module, spec.builder_name, autocomplete_builder)
        else:

            async def builder(template, messages, user, *, _task_name=task_name):
                captured_templates[_task_name] = template
                return f"rendered::{_task_name}::{template}"

            setattr(task_module, spec.builder_name, builder)

    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "open_webui.utils.task", task_module)
    import open_webui.utils as core_utils

    monkeypatch.setattr(core_utils, "task", task_module, raising=False)
    pipe_request.app.state.config = SimpleNamespace(
        **{spec.config_attr: f"legacy::{task_name}" for task_name, spec in mod.TASK_PROMPT_SPECS.items()}
    )

    for task_name in mod.TASK_PROMPT_SPECS:
        task_body = {
            "messages": [{"role": "user", "content": f"history for {task_name}"}],
            "prompt": f"prompt for {task_name}",
            "type": "sentence",
        }
        rendered = await mod._render_task_prompt_from_messages(
            request=pipe_request,
            user=pipe_user,
            metadata={"task": task_name, "task_body": task_body},
            messages=task_body["messages"],
        )
        expected_template = f"legacy::{task_name}"
        assert rendered == f"rendered::{task_name}::{expected_template}"
        assert captured_templates[task_name] == expected_template


@pytest.mark.asyncio
async def test_pipe_does_not_rebuild_task_prompt_from_task_body_by_default(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    exact_source = [
        {"role": "user", "content": "old raw request"},
        {"role": "assistant", "content": "old raw answer"},
        {"role": "user", "content": "older raw follow-up"},
        {"role": "assistant", "content": "older raw follow-up answer"},
    ]
    checkpoint = {
        "id": "checkpoint-task",
        "state": "ready",
        "source_message_count": len(exact_source),
        "source_hash": mod.compute_source_hash(exact_source),
        "summary_text": "checkpointed task history",
        "summary_meta": {mod.SUMMARY_META_FORMAT_VERSION_KEY: mod.SUMMARY_META_FORMAT_VERSION},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    class ExistingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            captured.setdefault("lookup_source_hashes", []).append(kwargs["source_hash"])
            if kwargs["source_hash"] == checkpoint["source_hash"]:
                return checkpoint
            return None

        async def find_longest_parent(self, **kwargs):
            return None

        async def insert_ready(self, row):
            raise AssertionError("default task handling must not create a checkpoint from task_body")

        async def touch(self, checkpoint_id):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("default task handling must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe_request.app.state.config = SimpleNamespace(TAGS_GENERATION_PROMPT_TEMPLATE="Task:\n{{MESSAGES}}")
    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    task_history = [*exact_source, {"role": "user", "content": "active task input"}]
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "system", "content": "pipeline system instruction"},
            {"role": "user", "content": "FILTERED TASK PROMPT"},
        ],
    }
    metadata = {
        **pipe_metadata,
        "task": mod.TASKS.TAGS_GENERATION.value,
        "task_body": {
            "model": wrapper_id,
            "chat_id": pipe_metadata["chat_id"],
            "messages": task_history,
        },
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)

    assert result == {"ok": True}
    assert "touched" not in captured
    forwarded = captured["forward_body"]
    assert forwarded["messages"] == body["messages"]
    assert "checkpointed task history" not in forwarded["messages"][1]["content"]
    assert "old raw request" not in forwarded["messages"][1]["content"]


@pytest.mark.asyncio
async def test_task_prompt_estimate_uses_provider_body_when_file_context_disabled(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target", "capabilities": {"file_context": False}}}

    async def body_reusable_checkpoint_match(**kwargs):
        captured["checkpoint_lookup_body"] = copy.deepcopy(kwargs["body"])
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        captured["estimate_body"] = copy.deepcopy(body)
        return 10

    async def inject_target_file_context(**kwargs):
        raise AssertionError("target file_context=false must not inject file context")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.compact_task_prompts_from_task_body = True
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [{"role": "user", "content": "FILTERED TASK PROMPT"}],
    }
    metadata = {
        **pipe_metadata,
        "task": mod.TASKS.TAGS_GENERATION.value,
        "task_body": {
            "model": wrapper_id,
            "chat_id": pipe_metadata["chat_id"],
            "messages": [
                {"role": "user", "content": "raw task input"},
                {"role": "assistant", "content": "raw task answer"},
                {"role": "user", "content": "active task input"},
            ],
        },
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)

    assert result == {"ok": True}
    assert captured["checkpoint_lookup_body"]["messages"] == metadata["task_body"]["messages"]
    assert captured["estimate_body"]["messages"] == body["messages"]
    assert captured["forward_body"]["messages"] == body["messages"]


@pytest.mark.asyncio
async def test_pipe_rebuilds_open_webui_task_prompt_with_reusable_checkpoint(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    exact_source = [
        {"role": "user", "content": "old raw request"},
        {"role": "assistant", "content": "old raw answer"},
        {"role": "user", "content": "older raw follow-up"},
        {"role": "assistant", "content": "older raw follow-up answer"},
    ]
    checkpoint = {
        "id": "checkpoint-task",
        "state": "ready",
        "source_message_count": len(exact_source),
        "source_hash": mod.compute_source_hash(exact_source),
        "summary_text": "checkpointed task history",
        "summary_meta": {mod.SUMMARY_META_FORMAT_VERSION_KEY: mod.SUMMARY_META_FORMAT_VERSION},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    class ExistingCheckpointStore:
        async def lookup_ready(self, **kwargs):
            if kwargs["source_hash"] == checkpoint["source_hash"]:
                return checkpoint
            return None

        async def find_longest_parent(self, **kwargs):
            return None

        async def insert_ready(self, row):
            raise AssertionError("task checkpoint reuse must not create a new checkpoint")

        async def touch(self, checkpoint_id):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("task checkpoint reuse must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe_request.app.state.config = SimpleNamespace(TAGS_GENERATION_PROMPT_TEMPLATE="Task:\n{{MESSAGES}}")
    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    pipe.valves.compact_task_prompts_from_task_body = True
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    task_history = [*exact_source, {"role": "user", "content": "active task input"}]
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "system", "content": "pipeline system instruction"},
            {"role": "user", "content": "Task:\nold raw request\nold raw answer\nactive task input"},
        ],
    }
    metadata = {
        **pipe_metadata,
        "task": mod.TASKS.TAGS_GENERATION.value,
        "task_body": {
            "model": wrapper_id,
            "chat_id": pipe_metadata["chat_id"],
            "messages": task_history,
        },
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)

    assert result == {"ok": True}
    assert captured["touched"] == "checkpoint-task"
    forwarded = captured["forward_body"]
    assert forwarded["messages"][0] == {"role": "system", "content": "pipeline system instruction"}
    forwarded_prompt = forwarded["messages"][1]["content"]
    assert "checkpointed task history" in forwarded_prompt
    assert "active task input" in forwarded_prompt
    assert "old raw request" not in forwarded_prompt
    assert "old raw answer" not in forwarded_prompt
    task_body_messages = forwarded["metadata"]["task_body"]["messages"]
    assert "checkpointed task history" in task_body_messages[0]["content"]
    assert task_body_messages[1:] == [{"role": "user", "content": "active task input"}]


@pytest.mark.asyncio
async def test_task_checkpoint_applied_estimate_uses_rebuilt_provider_body(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}
    compacted_source = {
        "messages": [
            {"role": "user", "content": "checkpointed task history"},
            {"role": "user", "content": "active task input"},
        ]
    }
    rebuilt_body = {
        "model": "target",
        "messages": [{"role": "user", "content": "Task:\ncheckpointed task history\nactive task input"}],
    }

    async def compact_body_with_reusable_checkpoint(**kwargs):
        captured["source_body"] = copy.deepcopy(kwargs["body"])
        return copy.deepcopy(compacted_source), True, 1

    async def rebuild_task_body_from_compacted_history(**kwargs):
        captured["rebuilt_history"] = copy.deepcopy(kwargs["compacted_history_messages"])
        return copy.deepcopy(rebuilt_body)

    async def estimate_body_tokens_async(body, **kwargs):
        captured["estimated_body"] = copy.deepcopy(body)
        return 123

    monkeypatch.setattr(mod, "_compact_body_with_reusable_checkpoint", compact_body_with_reusable_checkpoint)
    monkeypatch.setattr(mod, "_rebuild_task_body_from_compacted_history", rebuild_task_body_from_compacted_history)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)

    count = await mod._estimate_task_checkpoint_applied_body_tokens(
        request=pipe_request,
        user=pipe_user,
        metadata=pipe_metadata,
        body={"messages": [{"role": "user", "content": "source task history"}]},
        pipe_function_id="auto_compact",
        match=mod.ReusableCheckpointMatch(kind="exact", source_message_count=1, checkpoint={"id": "checkpoint-task"}),
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert count == 123
    assert captured["rebuilt_history"] == compacted_source["messages"]
    assert captured["estimated_body"] == rebuilt_body


@pytest.mark.asyncio
async def test_task_checkpoint_applied_estimate_returns_none_when_file_context_unavailable(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    async def compact_body_with_reusable_checkpoint(**kwargs):
        raise mod.SummaryFileContextUnavailable("attached file context unavailable")

    monkeypatch.setattr(mod, "_compact_body_with_reusable_checkpoint", compact_body_with_reusable_checkpoint)

    count = await mod._estimate_task_checkpoint_applied_body_tokens(
        request=pipe_request,
        user=pipe_user,
        metadata=pipe_metadata,
        body={"messages": [{"role": "user", "content": "source task history"}]},
        pipe_function_id="auto_compact",
        match=mod.ReusableCheckpointMatch(kind="exact", source_message_count=1, checkpoint={"id": "checkpoint-task"}),
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert count is None


@pytest.mark.asyncio
async def test_task_reusable_checkpoint_passes_file_context_disabled_to_history_compaction(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}

    async def compact_body_with_reusable_checkpoint(**kwargs):
        captured["file_context_enabled"] = kwargs.get("file_context_enabled")
        return {"messages": [{"role": "user", "content": "checkpointed task history"}]}, True, 1

    async def rebuild_task_body_from_compacted_history(**kwargs):
        return {"messages": [{"role": "user", "content": "rebuilt task prompt"}]}

    monkeypatch.setattr(mod, "_compact_body_with_reusable_checkpoint", compact_body_with_reusable_checkpoint)
    monkeypatch.setattr(mod, "_rebuild_task_body_from_compacted_history", rebuild_task_body_from_compacted_history)

    rebuilt, compacted, prefix_count = await mod._compact_task_body_with_reusable_checkpoint(
        request=pipe_request,
        user=pipe_user,
        metadata={
            **pipe_metadata,
            "task": mod.TASKS.TAGS_GENERATION.value,
            "task_body": {
                "model": "target",
                "messages": [
                    {"role": "user", "content": "old task input"},
                    {"role": "assistant", "content": "old task answer"},
                    {"role": "user", "content": "active task input"},
                ],
            },
        },
        body={"messages": [{"role": "user", "content": "provider task prompt"}]},
        pipe_function_id="auto_compact",
        match=mod.ReusableCheckpointMatch(kind="exact", source_message_count=2, source_kind="message"),
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
        file_context_enabled=False,
    )

    assert compacted is True
    assert prefix_count == 1
    assert rebuilt["messages"] == [{"role": "user", "content": "rebuilt task prompt"}]
    assert captured["file_context_enabled"] is False


@pytest.mark.asyncio
async def test_pipe_uses_task_checkpoint_applied_estimate_for_reusable_checkpoint_guard(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return mod.ReusableCheckpointMatch(
            kind="exact",
            source_message_count=2,
            source_kind="message",
            checkpoint={"id": "checkpoint-task"},
        )

    async def estimate_task_checkpoint_applied_body_tokens(**kwargs):
        captured["task_estimate_body"] = copy.deepcopy(kwargs["body"])
        return 150

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        raise AssertionError("task reusable checkpoint guard must estimate the rebuilt task body")

    async def compact_task_body(**kwargs):
        captured["foreground_compaction"] = True
        compacted = copy.deepcopy(kwargs["body"])
        compacted["messages"] = [{"role": "user", "content": "foreground task summary"}]
        return compacted, True, 2

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_estimate_task_checkpoint_applied_body_tokens", estimate_task_checkpoint_applied_body_tokens, raising=False)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_compact_task_body", compact_task_body)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe_request.app.state.config = SimpleNamespace(TAGS_GENERATION_PROMPT_TEMPLATE="Task:\n{{MESSAGES}}")
    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.compact_task_prompts_from_task_body = True
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    metadata = {
        **pipe_metadata,
        "task": mod.TASKS.TAGS_GENERATION.value,
        "task_body": {
            "model": wrapper_id,
            "chat_id": pipe_metadata["chat_id"],
            "messages": [
                {"role": "user", "content": "old task input"},
                {"role": "assistant", "content": "old task answer"},
                {"role": "user", "content": "active task input"},
            ],
        },
    }
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [{"role": "user", "content": "Task:\nold task input\nactive task input"}],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)

    assert result == {"ok": True}
    assert captured["foreground_compaction"] is True
    assert captured["forward_body"]["messages"] == [{"role": "user", "content": "foreground task summary"}]


@pytest.mark.asyncio
async def test_task_soft_prefetch_passes_rebuilt_prompt_for_checkpoint_estimates(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def body_reusable_checkpoint_match(**kwargs):
        captured.setdefault("checkpoint_lookup_body", copy.deepcopy(kwargs["body"]))
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        captured["estimate_body"] = copy.deepcopy(body)
        return 150

    def start_soft_prefetch(**kwargs):
        captured["prefetch_kwargs"] = {
            key: copy.deepcopy(value)
            for key, value in kwargs.items()
            if key not in {"request", "user", "event_emitter"}
        }
        return True

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe_request.app.state.config = SimpleNamespace(TAGS_GENERATION_PROMPT_TEMPLATE="Task:\n{{MESSAGES}}")
    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 200
    pipe.valves.soft_trigger_ratio = 0.5
    pipe.valves.compact_task_prompts_from_task_body = True
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    task_history = [
        {"role": "user", "content": "old task input"},
        {"role": "assistant", "content": "old task answer"},
        {"role": "user", "content": "active task input"},
    ]
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [{"role": "user", "content": "Task:\nold task input\nactive task input"}],
    }
    metadata = {
        **pipe_metadata,
        "task": mod.TASKS.TAGS_GENERATION.value,
        "task_body": {
            "model": wrapper_id,
            "chat_id": pipe_metadata["chat_id"],
            "messages": task_history,
        },
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)

    assert result == {"ok": True}
    assert captured["checkpoint_lookup_body"]["messages"] == task_history
    assert captured["estimate_body"]["messages"] == body["messages"]
    assert captured["prefetch_kwargs"]["body"]["messages"] == task_history
    assert captured["prefetch_kwargs"]["task_estimate_body"]["messages"] == body["messages"]
    assert captured["forward_body"]["messages"] == body["messages"]


@pytest.mark.asyncio
async def test_pipe_forwards_unchanged_when_checkpoint_lookup_fails_and_request_under_limit(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    # T1a: DB down on initial lookup + below hard limit (and below soft window)
    # forwards unchanged. No prefetch or compaction.
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_body_tokens_async(body, **kwargs):
        return 10

    async def generate_summary_text(**kwargs):
        raise AssertionError("must not compact when DB is down and request is within limits")

    def start_soft_prefetch(**kwargs):
        raise AssertionError("must not prefetch when checkpoint lookup is unavailable")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        captured["on_complete"] = kwargs.get("on_complete")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.soft_trigger_ratio = 0.5
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]

    result = await pipe.pipe(
        {"model": wrapper_id, "stream": True, "messages": messages},
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == messages
    assert captured["on_complete"] is None


@pytest.mark.asyncio
async def test_pipe_fails_closed_when_checkpoint_lookup_fails_and_hard_compaction_required(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    # T1b/T5: DB down + hard compaction required (decision_total >= trigger)
    # fails closed with checkpoint_unavailable; target is never forwarded.
    forward_calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_body_tokens_async(body, **kwargs):
        return 250000

    async def forward_target(**kwargs):
        forward_calls.append(copy.deepcopy(kwargs["body"]))
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 200
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "checkpoint_unavailable"
    assert "checkpoint db down" in result["error"]["message"]
    assert forward_calls == []


@pytest.mark.asyncio
async def test_pipe_forwards_unchanged_when_checkpoint_lookup_fails_and_request_in_soft_window(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    # T2: DB down + soft window (soft <= decision_total < hard). soft prefetch
    # is disabled (R3/R4); the request is forwarded unchanged with no prefetch.
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_body_tokens_async(body, **kwargs):
        # 50 (soft) <= 75 < 100 (hard): soft window.
        return 75

    def start_soft_prefetch(**kwargs):
        raise AssertionError("must not prefetch when checkpoint lookup is unavailable")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        captured["on_complete"] = kwargs.get("on_complete")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.soft_trigger_ratio = 0.5
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]

    result = await pipe.pipe(
        {"model": wrapper_id, "stream": True, "messages": messages},
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == messages
    assert captured["on_complete"] is None


@pytest.mark.asyncio
async def test_pipe_db_down_lookup_creates_no_checkpoint_and_skips_completed_turn_prefetch_non_streaming(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    # T3 (non-streaming): DB down + below limit. No checkpoint is created (R5),
    # no completed-turn soft prefetch is launched (R7 non-streaming). The
    # response carries an assistant message with usage in the soft window so
    # that, absent the R7 skip, the completed-turn prefetch would fire.
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def generate_summary_text(**kwargs):
        raise AssertionError("must not compact when DB is down (no checkpoint created)")

    def start_soft_prefetch(**kwargs):
        raise AssertionError(
            "must not start soft prefetch (pre-forward or completed-turn) when DB is down"
        )

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        captured["on_complete"] = kwargs.get("on_complete")
        # Usage in the soft window so completed-turn prefetch would otherwise fire.
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 75, "input_tokens": 70, "output_tokens": 5},
        }

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.soft_trigger_ratio = 0.5
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]

    result = await pipe.pipe(
        {"model": wrapper_id, "stream": False, "messages": messages},
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert isinstance(result, dict)
    assert result.get("choices") is not None
    assert captured["forward_body"]["messages"] == messages
    assert captured["on_complete"] is None


@pytest.mark.asyncio
async def test_pipe_fails_closed_when_checkpoint_lookup_fails_and_decision_total_is_none(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    # T6: DB down + estimator cannot produce a token count (decision_total is
    # None). We cannot confirm the request is under limit, so fail closed.
    forward_calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_body_tokens_async(body, **kwargs):
        return None

    async def forward_target(**kwargs):
        forward_calls.append(copy.deepcopy(kwargs["body"]))
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result["error"]["code"] == "checkpoint_unavailable"
    assert "checkpoint db down" in result["error"]["message"]
    assert forward_calls == []


@pytest.mark.asyncio
async def test_pipe_fails_closed_on_overflow_retry_when_checkpoint_lookup_was_unavailable(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    # DB-down on the initial lookup, but the request is confirmed under the hard
    # limit, so it is forwarded unchanged. The target then reports a context
    # overflow (RetryableContextOverflow), which proves compaction is actually
    # required. Since the checkpoint DB was already known unavailable, the retry
    # must fail closed with checkpoint_unavailable using the ORIGINAL lookup
    # error — NOT attempt compaction (which would surface summary_failed or
    # create a checkpoint if the DB recovered mid-request).
    forward_attempts = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def generate_summary_text(**kwargs):
        raise AssertionError("must not attempt compaction when checkpoint lookup was unavailable")

    def start_soft_prefetch(**kwargs):
        raise AssertionError("must not prefetch when checkpoint lookup is unavailable")

    async def forward_target(**kwargs):
        forward_attempts.append(copy.deepcopy(kwargs["body"]))
        raise mod.RetryableContextOverflow("target context window exceeded before output")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.soft_trigger_ratio = 0.5
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]

    result = await pipe.pipe(
        {"model": wrapper_id, "stream": True, "messages": messages},
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    # The request was forwarded once (proving the under-limit path ran), but the
    # overflow retry must NOT re-enter compaction.
    assert len(forward_attempts) == 1
    assert result["error"]["code"] == "checkpoint_unavailable"
    assert "checkpoint db down" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_creates_tool_checkpoint_without_history_parent_when_summary_fits(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return f"summary {len(summary_inputs)}"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_candidate_token_estimate(monkeypatch, 250000)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "x" * 200000}
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *history,
            active,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            tool_message,
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    tool_source = [*history, active]
    assert summary_inputs == [tool_source]
    assert len(rows) == 1
    assert rows[0]["source_message_count"] == len(tool_source)
    assert rows[0]["source_hash"] == mod.compute_source_hash(tool_source)
    assert rows[0]["parent_checkpoint_id"] is None
    messages = captured["forward_body"]["messages"]
    assert "summary 1" in messages[0]["content"]
    assert messages[1:] == body["messages"][3:]


@pytest.mark.asyncio
async def test_direct_tool_compaction_and_reusable_checkpoint_render_same_saved_excerpts(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active request"}
    old_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    tool_source = [*history, active, *old_round]
    body = {"messages": [*tool_source, *latest_round]}

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "tool summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    direct_body, direct_did_compact, direct_prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
    )
    reusable_body, reusable_did_compact, reusable_prefix_count = await mod._compact_body_with_reusable_checkpoint(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        match=mod.ReusableCheckpointMatch(
            kind="exact",
            source_message_count=len(tool_source),
            source_kind="tool",
        ),
        historical_message_excerpt_bytes=1,
        historical_message_excerpt_count=99,
    )

    assert direct_did_compact is True
    assert reusable_did_compact is True
    assert direct_prefix_count == reusable_prefix_count == len(tool_source)
    assert len(summary_inputs) == 1
    assert len(rows) == 1
    stored = rows[0]["summary_meta"]["historical_user_messages"]
    assert stored["max_count"] == 1
    assert stored["max_bytes_per_message"] == 64
    assert stored["messages"] == [{"ordinal": 1, "text": "active request"}]
    assert direct_body["messages"][0]["content"] == reusable_body["messages"][0]["content"]
    assert '<historical_user_message ordinal="1"><![CDATA[active request]]></historical_user_message>' in direct_body["messages"][0]["content"]
    assert direct_body["messages"][1:] == latest_round
    assert reusable_body["messages"][1:] == latest_round


@pytest.mark.asyncio
async def test_reusable_checkpoint_fails_closed_when_db_chain_is_unavailable_for_attached_files(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    messages = [
        {"role": "user", "content": "old with file"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]
    cut = mod.select_safe_message_cut(messages)
    assert cut.summarization_prefix
    checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(cut.summarization_prefix),
        source_message_count=len(cut.summarization_prefix),
        summary_text="summary without attached file context",
        summary_meta={},
        parent_checkpoint_id=None,
        now=123,
    )

    async def load_chat_message_chain(request, chat_id, current_message_id):
        assert chat_id == "chat-1"
        assert current_message_id == "message-1"
        return None

    async def noop_initialize(**kwargs):
        return None

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore([checkpoint]))

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._compact_body_with_reusable_checkpoint(
            request=pipe_request,
            user=pipe_user,
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("absorbed-file")],
            },
            body={"messages": messages},
            pipe_function_id="auto_compact",
            match=mod.ReusableCheckpointMatch(
                kind="exact",
                source_message_count=len(cut.summarization_prefix),
                source_kind="message",
            ),
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        )


@pytest.mark.asyncio
async def test_tool_reusable_checkpoint_fails_closed_when_current_file_is_in_prefix_and_db_chain_unavailable(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    messages = [
        {"role": "user", "content": "active with file"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-2", "type": "function"}]},
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    tool_cut = mod.select_tool_result_compaction_cut(messages)
    assert tool_cut is not None
    assert tool_cut.summarization_prefix[0]["content"] == "active with file"
    checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(tool_cut.summarization_prefix),
        source_message_count=len(tool_cut.summarization_prefix),
        summary_text="summary without current file context",
        summary_meta={},
        parent_checkpoint_id=None,
        now=123,
    )

    async def load_chat_message_chain(request, chat_id, current_message_id):
        return None

    async def noop_initialize(**kwargs):
        return None

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore([checkpoint]))

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._compact_body_with_reusable_checkpoint(
            request=pipe_request,
            user=pipe_user,
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("current-file")],
                "user_message": {"files": [_file("current-file")]},
            },
            body={"messages": messages},
            pipe_function_id="auto_compact",
            match=mod.ReusableCheckpointMatch(
                kind="exact",
                source_message_count=len(tool_cut.summarization_prefix),
                source_kind="tool",
                checkpoint=checkpoint,
            ),
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        )


@pytest.mark.asyncio
async def test_tool_compaction_fails_closed_when_current_file_is_in_prefix_and_db_chain_unavailable(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    messages = [
        {"role": "user", "content": "active with file"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-2", "type": "function"}]},
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    tool_cut = mod.select_tool_result_compaction_cut(messages)
    assert tool_cut is not None
    assert tool_cut.summarization_prefix[0]["content"] == "active with file"

    async def load_chat_message_chain(request, chat_id, current_message_id):
        return None

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._compact_retry_tool_results(
            request=pipe_request,
            user=pipe_user,
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("current-file")],
                "user_message": {"files": [_file("current-file")]},
            },
            pipe_function_id="auto_compact",
            summary_model_id="target",
            base_body={"messages": messages},
            messages=messages,
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        )


@pytest.mark.asyncio
async def test_completed_turn_prefetch_fails_closed_when_current_file_is_in_prefix_and_db_chain_unavailable(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    import open_webui.utils.chat as chat_module

    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active with file"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": ""},
    ]
    prefetch_source = mod._soft_prefetch_source_messages({"model": "target", "messages": messages})
    assert prefetch_source == (messages[:-1], None)
    source_messages, preserved_system_message = prefetch_source

    async def load_chat_message_chain(request, chat_id, current_message_id):
        return None

    async def generate_chat_completion(*args, **kwargs):
        raise AssertionError("summary generation must not run without absorbed file context")

    async def noop_initialize(**kwargs):
        return None

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore([]))
    chat_module.generate_chat_completion = generate_chat_completion

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._prefetch_compaction_checkpoint(
            request=pipe_request,
            user=pipe_user,
            user_id=pipe_user["id"],
            chat_id="chat-1",
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("current-file")],
                "user_message": {"content": "active with file", "files": [_file("current-file")]},
            },
            body={"model": "target", "messages": messages},
            pipe_function_id="auto_compact",
            summary_model_id="target",
            source_messages=source_messages,
            preserved_system_message=preserved_system_message,
            summary_tool_policy="fallback_on_tool_call",
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        )


@pytest.mark.asyncio
async def test_message_reusable_checkpoint_fails_closed_for_current_file_when_db_chain_is_unavailable(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    messages = [
        {"role": "user", "content": "old text"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active with file"},
    ]
    cut = mod.select_safe_message_cut(messages)
    assert cut.summarization_prefix
    checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(cut.summarization_prefix),
        source_message_count=len(cut.summarization_prefix),
        summary_text="old text summary",
        summary_meta={},
        parent_checkpoint_id=None,
        now=123,
    )

    async def load_chat_message_chain(request, chat_id, current_message_id):
        return None

    async def noop_initialize(**kwargs):
        return None

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore([checkpoint]))

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._compact_body_with_reusable_checkpoint(
            request=pipe_request,
            user=pipe_user,
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("current-file")],
                "user_message": {"files": [_file("current-file")]},
            },
            body={"messages": messages},
            pipe_function_id="auto_compact",
            match=mod.ReusableCheckpointMatch(
                kind="exact",
                source_message_count=len(cut.summarization_prefix),
                source_kind="message",
                checkpoint=checkpoint,
            ),
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        )


@pytest.mark.asyncio
async def test_message_reusable_checkpoint_fails_closed_when_current_file_id_is_also_in_prefix(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    messages = [
        {"role": "user", "content": "old with shared file"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active with same file"},
    ]
    cut = mod.select_safe_message_cut(messages)
    assert cut.summarization_prefix
    checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(cut.summarization_prefix),
        source_message_count=len(cut.summarization_prefix),
        summary_text="summary without shared file context",
        summary_meta={},
        parent_checkpoint_id=None,
        now=123,
    )

    async def load_chat_message_chain(request, chat_id, current_message_id):
        return None

    async def noop_initialize(**kwargs):
        return None

    monkeypatch.setattr(mod, "_load_chat_message_chain", load_chat_message_chain)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore([checkpoint]))

    with pytest.raises(mod.SummaryFileContextUnavailable):
        await mod._compact_body_with_reusable_checkpoint(
            request=pipe_request,
            user=pipe_user,
            metadata={
                "chat_id": "chat-1",
                "user_message_id": "message-1",
                "files": [_file("shared-file")],
                "user_message": {"files": [_file("shared-file")]},
            },
            body={"messages": messages},
            pipe_function_id="auto_compact",
            match=mod.ReusableCheckpointMatch(
                kind="exact",
                source_message_count=len(cut.summarization_prefix),
                source_kind="message",
                checkpoint=checkpoint,
            ),
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        )


@pytest.mark.asyncio
async def test_hard_compaction_delegates_exact_pending_to_checkpoint_claim_wait(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    pending_row = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(source_messages),
        source_message_count=len(source_messages),
        summary_text="",
        summary_meta={},
        parent_checkpoint_id=None,
        state="pending",
        claim_token="claim-1",
        claim_expires_at=9999999999,
    )
    captured = {}

    async def noop_initialize(**kwargs):
        return None

    async def wait_for_pending_checkpoint_ready(row):
        raise AssertionError("exact pending should use the existing checkpoint claim/wait path only")

    async def get_or_create_compaction_summary(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "exact pending summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore([pending_row]))
    monkeypatch.setattr(mod, "_wait_for_pending_checkpoint_ready", wait_for_pending_checkpoint_ready)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    compacted, did_compact, _compaction_prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body={
            "messages": [
                *source_messages,
                {"role": "user", "content": "active"},
            ]
        },
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
    )

    assert did_compact is True
    assert captured["source_messages"] == source_messages
    assert "exact pending summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][1:] == [{"role": "user", "content": "active"}]


@pytest.mark.asyncio
async def test_wait_for_pending_checkpoint_ready_bounds_stalled_lookup(monkeypatch):
    pending_row = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash="source",
        source_message_count=1,
        summary_text="",
        summary_meta={},
        parent_checkpoint_id=None,
        state="pending",
        claim_token="claim-1",
        claim_expires_at=9999999999,
    )
    stalled_lookup = asyncio.get_running_loop().create_future()
    lookup_started = asyncio.Event()

    class StalledStore:
        async def lookup_any(self, **kwargs):
            lookup_started.set()
            return await stalled_lookup

    monkeypatch.setattr(mod, "CheckpointStore", StalledStore)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS", 0)

    try:
        result = await asyncio.wait_for(
            mod._wait_for_pending_checkpoint_ready(pending_row),
            timeout=0.1,
        )
    finally:
        if not stalled_lookup.done():
            stalled_lookup.cancel()

    assert result is None
    assert not lookup_started.is_set()


@pytest.mark.asyncio
async def test_tool_compaction_waits_for_pending_chain_prefix_before_foreground_summary(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    old_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    pending_source = [*history, active]
    pending_row = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(pending_source),
        source_message_count=len(pending_source),
        summary_text="",
        summary_meta={},
        parent_checkpoint_id=None,
        state="pending",
        claim_token="claim-1",
        claim_expires_at=9999999999,
    )

    class CompletingPendingStore(ClaimCheckpointStore):
        async def lookup_any(self, **kwargs):
            row = self._match(kwargs["source_hash"])
            if row is not None and row.get("state") == "pending":
                row.update(
                    state="ready",
                    summary_text="soft pending summary",
                    summary_meta={},
                    claim_token=None,
                    claim_expires_at=None,
                )
            return dict(row) if row else None

    store = CompletingPendingStore([pending_row])

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        raise AssertionError("hard compaction should wait for and reuse the pending soft summary first")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {"messages": [*history, active, *old_round, *latest_round]}
    compacted, did_compact, _compaction_prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=64,
        historical_message_excerpt_count=1,
    )

    assert did_compact is True
    assert "soft pending summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][1:] == [*old_round, *latest_round]


@pytest.mark.asyncio
async def test_tool_loop_summary_extends_existing_history_parent_when_summary_fits(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    tool_source = [*history, active]
    history_checkpoint = {
        "id": "history-checkpoint-1",
        "state": "ready",
        "source_message_count": len(history),
        "source_hash": mod.compute_source_hash(history),
        "summary_text": "existing history summary",
        "summary_meta": {},
    }

    async def noop_initialize(**kwargs):
        return None

    history_store = ClaimCheckpointStore([history_checkpoint])

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "direct tool summary"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: history_store)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "messages": [
            *tool_source,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
        ],
    }

    compacted, did_compact, _compaction_prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert did_compact is True
    assert len(summary_inputs) == 1
    assert "existing history summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [active]
    assert len(history_store.completed_rows) == 1
    assert history_store.completed_rows[0]["source_hash"] == mod.compute_source_hash(tool_source)
    assert history_store.completed_rows[0]["parent_checkpoint_id"] == history_checkpoint["id"]
    assert "direct tool summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][1:] == body["messages"][3:]


@pytest.mark.asyncio
async def test_parent_checkpoint_extension_summary_request_preserves_system(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    captured = {}
    system = {"role": "system", "content": "system prompt"}
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    source_messages = [*parent_source, active]
    parent_checkpoint = {
        "id": "parent-checkpoint-1",
        "state": "ready",
        "source_message_count": len(parent_source),
        "source_hash": mod.compute_source_hash(parent_source),
        "summary_text": "existing parent summary",
        "summary_meta": {},
    }

    async def noop_initialize(**kwargs):
        return None

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["messages"] = copy.deepcopy(form_data["messages"])
        return {"choices": [{"message": {"content": "extended summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))

    summary = await mod._get_or_create_compaction_summary(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        metadata={"chat_id": "chat-1"},
        summary_model_id="target",
        base_body={"model": "target", "messages": [system, *source_messages]},
        source_messages=source_messages,
        preserved_system_message=system,
        summary_meta=mod.build_checkpoint_summary_meta(
            source_messages,
            historical_message_excerpt_bytes=64,
            historical_message_excerpt_count=1,
        ),
        parent_checkpoint=parent_checkpoint,
    )

    assert summary == "extended summary"
    assert captured["messages"][0] == system
    assert "existing parent summary" in captured["messages"][1]["content"]
    assert captured["messages"][2:-1] == [active]
    assert captured["messages"][-1]["role"] == "user"
    assert rows[0]["source_message_count"] == len(source_messages)
    assert rows[0]["source_hash"] == mod.compute_source_hash(source_messages)


@pytest.mark.asyncio
async def test_pipe_reuses_exact_tool_checkpoint_without_creating_history_checkpoint(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    touched = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    old_tool_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    tool_source = [*history, active, *old_tool_round]
    checkpoint = {
        "id": "tool-checkpoint-1",
        "state": "ready",
        "source_message_count": len(tool_source),
        "source_hash": mod.compute_source_hash(tool_source),
        "summary_text": "existing tool summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    class ExistingToolCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            raise AssertionError("exact tool checkpoint hit must not insert")

        async def touch(self, checkpoint_id, *, now=None):
            touched.append(checkpoint_id)
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("exact tool checkpoint hit must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingToolCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [*tool_source, *latest_round],
    }
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert touched == ["tool-checkpoint-1"]
    messages = captured["forward_body"]["messages"]
    assert "existing tool summary" in messages[0]["content"]
    assert messages[1:] == latest_round
    assert events == []


@pytest.mark.asyncio
async def test_pipe_reuses_exact_tool_checkpoint_even_when_usage_is_below_threshold(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    touched = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    old_tool_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    tool_source = [*history, active, *old_tool_round]
    checkpoint = {
        "id": "tool-checkpoint-1",
        "state": "ready",
        "source_message_count": len(tool_source),
        "source_hash": mod.compute_source_hash(tool_source),
        "summary_text": "existing tool summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    class ExistingToolCheckpointStore(ClaimCheckpointStore):
        async def find_longest_parent(self, **kwargs):
            raise AssertionError("exact tool checkpoint reuse must not need a parent lookup")

        async def claim_pending(self, row):
            raise AssertionError("exact tool checkpoint hit must not insert")

        async def touch(self, checkpoint_id, *, now=None):
            touched.append(checkpoint_id)
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("exact tool checkpoint hit must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingToolCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [*tool_source, *latest_round],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert touched == ["tool-checkpoint-1"]
    messages = captured["forward_body"]["messages"]
    assert "existing tool summary" in messages[0]["content"]
    assert messages[1:] == latest_round
    assert events == []


@pytest.mark.asyncio
async def test_pipe_reuses_history_checkpoint_for_tool_loop_when_usage_is_below_threshold(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    history_checkpoint = {
        "id": "history-checkpoint-1",
        "state": "ready",
        "source_message_count": len(history),
        "source_hash": mod.compute_source_hash(history),
        "summary_text": "existing history summary",
        "summary_meta": {},
    }
    rows.append(history_checkpoint)

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            raise AssertionError("below-threshold checkpoint reuse must not create a new checkpoint")

        async def touch(self, checkpoint_id, *, now=None):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("below-threshold checkpoint reuse must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [*history, active, *latest_round],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert summary_inputs == []
    assert len(rows) == 1
    assert captured["touched"] == "history-checkpoint-1"
    messages = captured["forward_body"]["messages"]
    assert "existing history summary" in messages[0]["content"]
    assert messages[1:] == [active, *latest_round]


@pytest.mark.asyncio
async def test_pipe_reuses_longest_history_parent_for_tool_loop_when_usage_is_below_threshold(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    rows = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    short_parent_source = history[:1]
    short_parent = {
        "id": "short-parent-1",
        "state": "ready",
        "source_message_count": len(short_parent_source),
        "source_hash": mod.compute_source_hash(short_parent_source),
        "summary_text": "short parent summary",
        "summary_meta": {},
    }
    rows.append(short_parent)

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    class RecordingCheckpointStore(ClaimCheckpointStore):
        async def claim_pending(self, row):
            raise AssertionError("below-threshold checkpoint reuse must not create a new checkpoint")

        async def touch(self, checkpoint_id, *, now=None):
            captured["touched"] = checkpoint_id
            return True

    async def generate_summary_text(**kwargs):
        raise AssertionError("below-threshold checkpoint reuse must not call the summary model")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
    ]

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [*history, active, *latest_round],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert result == {"ok": True}
    assert len(rows) == 1
    assert captured["touched"] == "short-parent-1"
    messages = captured["forward_body"]["messages"]
    assert "short parent summary" in messages[0]["content"]
    assert messages[1:] == [history[1], active, *latest_round]


@pytest.mark.asyncio
async def test_pipe_falls_back_to_history_parent_when_direct_tool_summary_overflows(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    tool_source = [*history, active]

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        if kwargs["source_messages"] == tool_source:
            raise mod.RetryableContextOverflow("summary context")
        if kwargs["source_messages"] == history:
            return "history summary"
        return "tool summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_candidate_token_estimate(monkeypatch, 250000)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *tool_source,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert summary_inputs[0] == tool_source
    assert summary_inputs[1] == history
    assert summary_inputs[2][0]["role"] == "user"
    assert "history summary" in summary_inputs[2][0]["content"]
    assert summary_inputs[2][1] == active
    assert len(rows) == 2
    assert rows[0]["source_hash"] == mod.compute_source_hash(history)
    assert rows[1]["source_hash"] == mod.compute_source_hash(tool_source)
    assert rows[1]["parent_checkpoint_id"] == rows[0]["id"]
    messages = captured["forward_body"]["messages"]
    assert "tool summary" in messages[0]["content"]
    assert messages[1:] == body["messages"][3:]


@pytest.mark.asyncio
async def test_pipe_falls_back_to_history_checkpoint_when_existing_tool_parent_retry_overflows(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    tool_source = [*history, active]
    short_parent_source = history[:1]
    short_parent = {
        "id": "short-parent-1",
        "state": "ready",
        "source_message_count": len(short_parent_source),
        "source_hash": mod.compute_source_hash(short_parent_source),
        "summary_text": "short parent summary",
        "summary_meta": {},
    }
    rows.append(short_parent)

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        source = kwargs["source_messages"]
        summary_inputs.append(source)
        if source == tool_source:
            raise mod.RetryableContextOverflow("direct tool source overflow")
        if source and "short parent summary" in source[0].get("content", "") and source[1:] == [history[1], active]:
            raise mod.RetryableContextOverflow("short parent delta overflow")
        if source and "short parent summary" in source[0].get("content", "") and source[1:] == [history[1]]:
            return "history summary"
        if source and "history summary" in source[0].get("content", "") and source[1:] == [active]:
            return "tool summary"
        raise AssertionError(f"unexpected summary input: {source!r}")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "messages": [
            *tool_source,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
        ],
    }

    compacted, did_compact, _compaction_prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert did_compact is True
    assert "short parent summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [history[1], active]
    assert "short parent summary" in summary_inputs[1][0]["content"]
    assert summary_inputs[1][1:] == [history[1]]
    assert "history summary" in summary_inputs[2][0]["content"]
    assert summary_inputs[2][1:] == [active]
    assert len(rows) == 3
    assert rows[1]["source_hash"] == mod.compute_source_hash(history)
    assert rows[1]["parent_checkpoint_id"] == short_parent["id"]
    assert rows[2]["source_hash"] == mod.compute_source_hash(tool_source)
    assert rows[2]["parent_checkpoint_id"] == rows[1]["id"]
    assert "tool summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][1:] == body["messages"][3:]


@pytest.mark.asyncio
async def test_history_fallback_does_not_reselect_failed_longer_tool_parent(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    old_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    latest_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    failed_parent_source = [*history, active]
    failed_parent = {
        "id": "failed-parent-1",
        "state": "ready",
        "source_message_count": len(failed_parent_source),
        "source_hash": mod.compute_source_hash(failed_parent_source),
        "summary_text": "failed parent summary",
        "summary_meta": {},
    }
    rows.append(failed_parent)
    tool_source = [*failed_parent_source, *old_round]

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        source = kwargs["source_messages"]
        summary_inputs.append(source)
        if source == tool_source:
            raise mod.RetryableContextOverflow("direct tool source overflow")
        if source and "failed parent summary" in source[0].get("content", "") and source[1:] == old_round:
            raise mod.RetryableContextOverflow("failed parent delta overflow")
        if source == history:
            return "history summary"
        if source and "history summary" in source[0].get("content", "") and source[1:] == [active, *old_round]:
            return "tool summary"
        raise AssertionError(f"unexpected summary input: {source!r}")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {"messages": [*tool_source, *latest_round]}

    compacted, did_compact, _compaction_prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        body=body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert did_compact is True
    assert "failed parent summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == old_round
    assert summary_inputs[1] == history
    assert "history summary" in summary_inputs[2][0]["content"]
    assert summary_inputs[2][1:] == [active, *old_round]
    assert len(rows) == 3
    assert rows[1]["source_hash"] == mod.compute_source_hash(history)
    assert rows[2]["source_hash"] == mod.compute_source_hash(tool_source)
    assert rows[2]["parent_checkpoint_id"] == rows[1]["id"]
    assert "tool summary" in compacted["messages"][0]["content"]
    assert compacted["messages"][1:] == latest_round


@pytest.mark.asyncio
async def test_tool_loop_parent_retry_does_not_swallow_non_context_error(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    active = {"role": "user", "content": "active"}
    tool_source = [*history, active]
    short_parent_source = history[:1]
    rows.append(
        {
            "id": "short-parent-1",
            "state": "ready",
            "source_message_count": len(short_parent_source),
            "source_hash": mod.compute_source_hash(short_parent_source),
            "summary_text": "short parent summary",
            "summary_meta": {},
        }
    )

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        source = kwargs["source_messages"]
        summary_inputs.append(source)
        if source == tool_source:
            raise mod.RetryableContextOverflow("direct tool source overflow")
        if source and "short parent summary" in source[0].get("content", "") and source[1:] == [history[1], active]:
            raise RuntimeError("parent retry failed")
        raise AssertionError(f"unexpected summary input: {source!r}")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    body = {
        "messages": [
            *tool_source,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "latest result"},
        ],
    }

    with pytest.raises(mod.ParentCheckpointExtensionFailed) as exc_info:
        await mod._compact_body(
            request=pipe_request,
            user=pipe_user,
            metadata={"chat_id": "chat-1"},
            body=body,
            pipe_function_id="auto_compact",
            target_model_id="target",
            summary_model_id="target",
            historical_message_excerpt_bytes=0,
            historical_message_excerpt_count=0,
        )

    assert str(exc_info.value.original) == "parent retry failed"
    assert len(summary_inputs) == 1
    assert "short parent summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [history[1], active]
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_tool_loop_checkpoints_extend_previous_tool_checkpoint_without_history_parent(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    rows = []
    summary_inputs = []
    active = {"role": "user", "content": "active"}
    round_1 = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result 1"},
    ]
    round_2 = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "result 2"},
    ]
    round_3 = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-3", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-3", "content": "result 3"},
    ]

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        if kwargs["source_messages"] == [active, *round_1, *round_2]:
            raise mod.RetryableContextOverflow("summary context")
        return f"summary {len(summary_inputs)}"

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    metadata = {"chat_id": "chat-1"}
    first_body = {
        "messages": [
            active,
            *round_1,
            *round_2,
        ],
    }
    first_compacted, first_did_compact, _first_prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata=metadata,
        body=first_body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )
    second_body = {
        "messages": [
            active,
            *round_1,
            *round_2,
            *round_3,
        ],
    }
    second_compacted, second_did_compact, _second_prefix_count = await mod._compact_body(
        request=pipe_request,
        user=pipe_user,
        metadata=metadata,
        body=second_body,
        pipe_function_id="auto_compact",
        target_model_id="target",
        summary_model_id="target",
        historical_message_excerpt_bytes=0,
        historical_message_excerpt_count=0,
    )

    assert first_did_compact is True
    assert second_did_compact is True
    assert summary_inputs[0] == [active, *round_1]
    assert summary_inputs[1][0]["role"] == "user"
    assert "summary 1" in summary_inputs[1][0]["content"]
    assert summary_inputs[1][1:] == round_2
    assert len(rows) == 2
    assert rows[0]["source_hash"] == mod.compute_source_hash([active, *round_1])
    assert rows[0]["parent_checkpoint_id"] is None
    assert rows[1]["source_hash"] == mod.compute_source_hash([active, *round_1, *round_2])
    assert rows[1]["parent_checkpoint_id"] == rows[0]["id"]
    assert first_compacted["messages"][1:] == round_2
    assert second_compacted["messages"][1:] == round_3


@pytest.mark.asyncio
async def test_pipe_compacts_history_before_latest_tool_round_when_prior_checkpoint_exists(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    summary_inputs = []
    exact_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(exact_source),
        "source_hash": mod.compute_source_hash(exact_source),
        "summary_text": "existing summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    existing_store = ClaimCheckpointStore([checkpoint])

    async def estimate_body_tokens_async(body, **kwargs):
        return 250000

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined active summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: existing_store)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "x" * 200000}
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *exact_source,
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            tool_message,
        ],
    }
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(summary_inputs) == 1
    assert "existing summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [{"role": "user", "content": "active"}]
    checkpoint_source = [*exact_source, {"role": "user", "content": "active"}]
    assert existing_store.completed_rows[0]["parent_checkpoint_id"] == checkpoint["id"]
    assert existing_store.completed_rows[0]["source_message_count"] == len(checkpoint_source)
    assert existing_store.completed_rows[0]["source_hash"] == mod.compute_source_hash(checkpoint_source)
    messages = captured["forward_body"]["messages"]
    assert "combined active summary" in messages[0]["content"]
    assert messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call-1", "type": "function"}],
    }
    assert messages[2] == tool_message
    assert [event["type"] for event in events] == ["status", "status", "embeds"]


@pytest.mark.asyncio
async def test_pipe_compacts_history_before_latest_tool_round_when_checkpoint_parent_includes_retained_history(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    captured = {}
    summary_inputs = []
    exact_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    retained_old = [
        {"role": "user", "content": "retained 1"},
        {"role": "assistant", "content": "retained answer 1"},
        {"role": "user", "content": "retained 2"},
    ]
    checkpoint_source = [*exact_source, *retained_old]
    checkpoint = {
        "id": "checkpoint-1",
        "state": "ready",
        "source_message_count": len(checkpoint_source),
        "source_hash": mod.compute_source_hash(checkpoint_source),
        "summary_text": "existing summary",
        "summary_meta": {},
    }

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def noop_initialize(**kwargs):
        return None

    existing_store = ClaimCheckpointStore([checkpoint])

    async def estimate_body_tokens_async(body, **kwargs):
        return 250000

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined active summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: existing_store)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "x" * 200000}
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            *exact_source,
            *retained_old,
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            tool_message,
        ],
    }
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(summary_inputs) == 1
    assert "existing summary" in summary_inputs[0][0]["content"]
    assert summary_inputs[0][1:] == [{"role": "user", "content": "active"}]
    checkpoint_source_with_active = [*checkpoint_source, {"role": "user", "content": "active"}]
    assert existing_store.completed_rows[0]["parent_checkpoint_id"] == checkpoint["id"]
    assert existing_store.completed_rows[0]["source_message_count"] == len(checkpoint_source_with_active)
    assert existing_store.completed_rows[0]["source_hash"] == mod.compute_source_hash(checkpoint_source_with_active)
    messages = captured["forward_body"]["messages"]
    assert "combined active summary" in messages[0]["content"]
    assert messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call-1", "type": "function"}],
    }
    assert messages[2] == tool_message
    assert [event["type"] for event in events] == ["status", "status", "embeds"]


@pytest.mark.asyncio
async def test_pipe_retries_with_compaction_after_pre_emission_context_error(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_or_create_checkpoint_summary(**kwargs):
        return "retry summary"

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        if len(calls) == 1:
            raise mod.RetryableContextOverflow("context")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert len(calls) == 2
    assert calls[0]["messages"] == body["messages"]
    assert "retry summary" in calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_stops_context_retries_at_fixed_budget(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_or_create_checkpoint_summary(**kwargs):
        return "retry summary"

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        raise mod.RetryableContextOverflow("context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
    )

    assert len(calls) == mod.MAX_CONTEXT_RETRY_ATTEMPTS
    assert result["error"]["code"] == "context_window_exceeded"


@pytest.mark.asyncio
async def test_pipe_does_not_retry_non_context_target_errors(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        raise RuntimeError("rate limit")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")

    with pytest.raises(RuntimeError, match="rate limit"):
        await pipe.pipe(
            {
                "model": wrapper_id,
                "stream": True,
                "messages": [{"role": "user", "content": "active"}],
            },
            __request__=pipe_request,
            __user__=pipe_user,
            __metadata__=pipe_metadata,
        )

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_pipe_retries_tool_loop_by_summarizing_history_before_latest_tool_round(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    calls = []
    summary_inputs = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_or_create_compaction_summary(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined retry summary"

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        if len(calls) == 1:
            raise mod.RetryableContextOverflow("context")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "x" * 10000}
    events = []

    async def event_emitter(event):
        events.append(event)

    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            tool_message,
        ],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(summary_inputs) == 1
    assert summary_inputs[0] == [{"role": "user", "content": "active"}]
    retry_messages = calls[1]["messages"]
    assert "combined retry summary" in retry_messages[0]["content"]
    assert retry_messages[1:] == body["messages"][1:]
    assert [event["data"]["action"] for event in events if event["type"] == "status"] == [
        "auto_compaction_retry",
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]
    status_events = [event for event in events if event["type"] == "status"]
    assert status_events[-1]["data"]["done"] is True
    embed_events = [event for event in events if event["type"] == "embeds"]
    assert len(embed_events) == 1
    assert embed_events[0]["data"]["replace"] is False
    assert len(embed_events[0]["data"]["embeds"]) == 1
    embed_html = embed_events[0]["data"]["embeds"][0]
    assert "combined retry summary" in embed_html


@pytest.mark.asyncio
async def test_pipe_reemits_latest_compaction_summary_embed_on_retry_after_initial_compaction(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    calls = []
    summaries = ["first compacted summary", "second compacted summary"]

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_or_create_checkpoint_summary(**kwargs):
        return summaries.pop(0)

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        if len(calls) == 1:
            raise mod.RetryableContextOverflow("context")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_candidate_token_estimate(monkeypatch, 500)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(calls) == 2
    embed_events = [event for event in events if event["type"] == "embeds"]
    assert len(embed_events) == 2
    assert "first compacted summary" in embed_events[0]["data"]["embeds"][0]
    assert "second compacted summary" in embed_events[1]["data"]["embeds"][0]


@pytest.mark.asyncio
async def test_tool_loop_compaction_replaces_tool_round_without_mutating_tool_message(monkeypatch, pipe_request, pipe_user):
    original_tool = {"role": "tool", "tool_call_id": "call-1", "content": "original tool result"}

    async def get_or_create_compaction_summary(**kwargs):
        return "combined summary"

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    messages = [
        {"role": "user", "content": "active"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        original_tool,
    ]
    compacted, did_compact, _tool_prefix_count = await mod._compact_retry_tool_results(
        request=pipe_request,
        user=pipe_user,
        metadata={"chat_id": "chat-1"},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        base_body={"model": "target", "stream": True, "messages": messages},
        messages=messages,
    )

    assert did_compact is True
    assert original_tool == {"role": "tool", "tool_call_id": "call-1", "content": "original tool result"}
    assert compacted[0]["role"] == "user"
    assert "<checkpoint_summary><![CDATA[combined summary]]></checkpoint_summary>" in compacted[0]["content"]
    assert '<historical_user_message ordinal="1"><![CDATA[active]]></historical_user_message>' in compacted[0]["content"]
    assert compacted[1:] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        original_tool,
    ]


def _status_events(events):
    return [event for event in events if event["type"] == "status"]


def _known_empty_model_shaping_hash(model_id="target"):
    return mod._usage_anchor_shaping_hash(
        [{"model_id": model_id, "base_model_id": None, "params": {}}],
        transport_profile={
            "owned_by": "openai",
            "url_idx": 0,
            "url": "http://provider",
            "api_type": "chat_completions",
            "api_config": {},
        },
    )


async def _run_token_status_pipe(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
    *,
    usage=None,
    estimate_tokens=None,
    estimate_sequence=None,
    pipe=None,
    body=None,
    trigger_input_tokens=1000,
    forward_raises_once=False,
):
    captured = {"forward_bodies": []}
    estimate_calls = []
    events = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_target_db_model_record(model_id):
        assert model_id == "target"
        return None

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        estimate_calls.append(copy.deepcopy(body))
        if estimate_sequence is not None:
            index = min(len(estimate_calls) - 1, len(estimate_sequence) - 1)
            return estimate_sequence[index]
        return estimate_tokens

    async def get_or_create_compaction_summary(**kwargs):
        return "status summary"

    async def forward_target(**kwargs):
        captured["forward_bodies"].append(copy.deepcopy(kwargs["body"]))
        if forward_raises_once and len(captured["forward_bodies"]) == 1:
            raise mod.RetryableContextOverflow("context")
        return {"ok": True}

    async def event_emitter(event):
        events.append(event)

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_target_db_model_record", get_target_db_model_record)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_lookup_pending_checkpoint_for_source_prefix",
        lookup_pending_checkpoint_for_source_prefix,
    )
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_known_openai_usage_anchor_transport(monkeypatch)

    pipe = pipe or mod.Pipe()
    pipe.valves.trigger_input_tokens = trigger_input_tokens
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = body or {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }
    if usage is not None:
        status_uses_local_estimate = (
            pipe.valves.token_status_show_usage_and_estimate
            and (estimate_tokens is not None or estimate_sequence is not None)
        )
        anchor_input = None
        if not status_uses_local_estimate:
            target_body = copy.deepcopy(body)
            target_body["model"] = "target"
            anchor_input = await mod._build_usage_anchor_input(
                request=pipe_request,
                body=target_body,
                usage_anchor_shaping_hash=_known_empty_model_shaping_hash(),
            )
        mod.store_request_scoped_usage(
            request=pipe_request,
            chat_id=pipe_metadata["chat_id"],
            message_id=pipe_metadata["message_id"],
            wrapper_model_id=wrapper_id,
            usage=usage,
            anchor_input=anchor_input,
        )

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    return result, events, captured, estimate_calls


@pytest.mark.asyncio
async def test_status_shows_before_tokens_by_default(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    result, events, _captured, _estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        usage={"total_tokens": 1500, "input_tokens": 1500, "output_tokens": 0},
        trigger_input_tokens=1000,
    )

    assert result == {"ok": True}
    status_events = _status_events(events)
    assert [event["data"]["action"] for event in status_events] == [
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]
    compacting = status_events[0]["data"]
    assert "1,500" in compacting["description"]
    assert "· 150%" in compacting["description"]
    assert compacting["tokens"]["before"] == 1500
    assert compacting["tokens"]["hard_limit"] == 1000
    assert compacting["tokens"]["pct_of_hard"] >= 100


@pytest.mark.asyncio
async def test_status_before_after_mode(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    pipe = mod.Pipe()
    pipe.valves.token_status_detail = "before_after"

    result, events, _captured, _estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        pipe=pipe,
        usage={"total_tokens": 1500, "input_tokens": 1500, "output_tokens": 0},
        estimate_sequence=[120],
        trigger_input_tokens=1000,
    )

    assert result == {"ok": True}
    compacted = _status_events(events)[1]["data"]
    assert compacted["action"] == "auto_compaction_compacted"
    assert "->" not in compacted["description"]
    assert "→" in compacted["description"]
    assert "120" in compacted["description"]
    assert compacted["tokens"]["after"] == 120
    assert compacted["tokens"]["after"] < compacted["tokens"]["before"]


@pytest.mark.asyncio
async def test_status_show_usage_and_estimate_mode(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    pipe = mod.Pipe()
    pipe.valves.token_status_show_usage_and_estimate = True

    result, events, _captured, estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        pipe=pipe,
        usage={"total_tokens": 1250, "input_tokens": 1250, "output_tokens": 0},
        estimate_tokens=1280,
        trigger_input_tokens=1000,
    )

    assert result == {"ok": True}
    assert len(estimate_calls) == 1
    compacting = _status_events(events)[0]["data"]
    assert "observed usage" in compacting["description"]
    assert "candidate" in compacting["description"]
    assert compacting["tokens"]["usage"] == 1250
    assert compacting["tokens"]["estimate"] == 1280
    assert compacting["tokens"]["pct_of_hard"] == 128.0
    assert "128%" in compacting["description"]


@pytest.mark.asyncio
async def test_status_show_usage_and_estimate_and_before_after_combined(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    pipe = mod.Pipe()
    pipe.valves.token_status_show_usage_and_estimate = True
    pipe.valves.token_status_detail = "before_after"

    result, events, _captured, estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        pipe=pipe,
        usage={"total_tokens": 1250, "input_tokens": 1250, "output_tokens": 0},
        estimate_sequence=[1280, 500],
        trigger_input_tokens=1000,
    )

    assert result == {"ok": True}
    compacted = _status_events(events)[-1]["data"]
    assert compacted["action"] == "auto_compaction_compacted"
    assert "observed usage" in compacted["description"]
    assert "candidate" in compacted["description"]
    assert "→" in compacted["description"]
    assert "500" in compacted["description"]
    assert compacted["tokens"]["after"] == 500
    assert compacted["tokens"]["usage"] == 1250
    assert compacted["tokens"]["estimate"] == 1280


def test_format_token_suffix_summary_mode_distinguishes_summary_from_after():
    ctx = mod.DisplayTokenContext(
        before=850,
        usage=None,
        estimate=None,
        hard_limit=1000,
        soft_limit=800,
        usage_source="estimate",
        pct_of_hard=85.0,
    )

    assert (
        mod._format_token_suffix(ctx, after=None, show_usage_and_estimate=False, summary=50)
        == "(≈850 / 1,000 tokens · 85% · summary ≈50 tokens)"
    )


def test_format_token_suffix_summary_mode_with_usage_and_estimate():
    ctx = mod.DisplayTokenContext(
        before=850,
        usage=850,
        estimate=860,
        hard_limit=1000,
        soft_limit=800,
        usage_source="request",
        pct_of_hard=85.0,
    )

    assert (
        mod._format_token_suffix(ctx, after=50, show_usage_and_estimate=True, summary=50)
        == "(observed usage 850 · candidate ≈860 / 1,000 · 85% · summary ≈50 tokens)"
    )


@pytest.mark.asyncio
async def test_status_always_mode_emits_pressure(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    pipe = mod.Pipe()
    pipe.valves.token_status_visibility = "always"

    result, events, captured, _estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        pipe=pipe,
        usage=None,
        estimate_tokens=450,
        trigger_input_tokens=1000,
    )

    assert result == {"ok": True}
    assert captured["forward_bodies"][0]["messages"][0]["content"] == "old"
    status_events = _status_events(events)
    assert [event["data"]["action"] for event in status_events] == ["auto_compaction_status"]
    assert status_events[0]["data"]["done"] is True
    assert "450" in status_events[0]["data"]["description"]
    assert status_events[0]["data"]["tokens"]["before"] == 450


@pytest.mark.asyncio
async def test_status_compaction_only_silent_below_threshold(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    result, events, _captured, _estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        usage=None,
        estimate_tokens=450,
        trigger_input_tokens=1000,
    )

    assert result == {"ok": True}
    assert _status_events(events) == []


@pytest.mark.asyncio
async def test_status_unknown_tokens_render(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    result, events, captured, _estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        usage=None,
        estimate_tokens=None,
        trigger_input_tokens=1000,
        forward_raises_once=True,
    )

    assert result == {"ok": True}
    assert len(captured["forward_bodies"]) == 2
    status_events = _status_events(events)
    assert [event["data"]["action"] for event in status_events] == [
        "auto_compaction_retry",
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]
    assert "≈unknown" in status_events[0]["data"]["description"]
    assert status_events[0]["data"]["tokens"]["before"] is None


@pytest.mark.asyncio
async def test_status_prefetch_shows_trigger_tokens(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    events = []
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]

    async def event_emitter(event):
        events.append(event)

    async def get_or_create_compaction_summary(*, on_summary_start=None, **kwargs):
        assert on_summary_start is not None
        await on_summary_start()
        return mod.CompactionSummaryResult(
            "prefetched summary",
            checkpoint={"summary_text": "prefetched summary", "summary_meta": {}},
        )

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    assert (
        await mod._prefetch_compaction_checkpoint(
            request=pipe_request,
            user=pipe_user,
            user_id=pipe_user["id"],
            chat_id=pipe_metadata["chat_id"],
            metadata=pipe_metadata,
            body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
            pipe_function_id="auto_compact",
            summary_model_id="target",
            source_messages=source_messages,
            summary_tool_policy="fallback_on_tool_call",
            historical_message_excerpt_bytes=1024,
            historical_message_excerpt_count=3,
            effective_trigger_input_tokens=1000,
            effective_soft_trigger_input_tokens=800,
            trigger_observed_tokens=850,
            trigger_usage_source="request",
            event_emitter=event_emitter,
        )
        is True
    )

    status_events = [event for event in events if event["type"] == "status"]
    assert [event["data"]["action"] for event in status_events] == [
        "auto_compaction_prefetching",
        "auto_compaction_prefetched",
    ]
    prefetching = status_events[0]["data"]
    assert "850" in prefetching["description"]
    assert prefetching["tokens"]["before"] == 850
    assert prefetching["tokens"]["hard_limit"] == 1000
    assert prefetching["tokens"]["pct_of_hard"] == 85.0
    prefetched = status_events[1]["data"]
    assert prefetched["tokens"]["before"] == 850
    assert "summary" not in prefetched["tokens"]
    assert "summary" not in prefetched["description"].lower()


@pytest.mark.asyncio
async def test_status_prefetched_shows_summary_in_before_after_mode(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    events = []
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]

    async def event_emitter(event):
        events.append(event)

    async def get_or_create_compaction_summary(*, on_summary_start=None, **kwargs):
        assert on_summary_start is not None
        await on_summary_start()
        return mod.CompactionSummaryResult(
            "prefetched summary",
            checkpoint={
                "summary_text": "prefetched summary",
                "summary_token_count": 50,
                "summary_meta": {},
            },
        )

    async def estimate_rendered_summary_message_tokens(**kwargs):
        raise AssertionError("persisted summary_token_count should avoid fallback estimation")

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(
        mod,
        "_estimate_rendered_summary_message_tokens",
        estimate_rendered_summary_message_tokens,
    )

    assert (
        await mod._prefetch_compaction_checkpoint(
            request=pipe_request,
            user=pipe_user,
            user_id=pipe_user["id"],
            chat_id=pipe_metadata["chat_id"],
            metadata=pipe_metadata,
            body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
            pipe_function_id="auto_compact",
            summary_model_id="target",
            source_messages=source_messages,
            summary_tool_policy="fallback_on_tool_call",
            historical_message_excerpt_bytes=1024,
            historical_message_excerpt_count=3,
            effective_trigger_input_tokens=1000,
            effective_soft_trigger_input_tokens=800,
            trigger_observed_tokens=850,
            trigger_usage_source="request",
            token_status_detail="before_after",
            event_emitter=event_emitter,
        )
        is True
    )

    status_events = _status_events(events)
    assert [event["data"]["action"] for event in status_events] == [
        "auto_compaction_prefetching",
        "auto_compaction_prefetched",
    ]
    prefetching = status_events[0]["data"]
    assert prefetching["tokens"]["before"] == 850
    assert "after" not in prefetching["tokens"]
    assert "→" not in prefetching["description"]
    prefetched = status_events[1]["data"]
    assert prefetched["tokens"]["before"] == 850
    assert prefetched["tokens"]["summary"] == 50
    assert isinstance(prefetched["tokens"]["summary"], int)
    assert "after" not in prefetched["tokens"]
    assert "→" not in prefetched["description"]
    assert "summary" in prefetched["description"].lower()
    assert "summary ≈50 tokens" in prefetched["description"]


@pytest.mark.asyncio
async def test_status_prefetched_estimates_summary_when_checkpoint_count_is_missing(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    events = []
    estimate_calls = []
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]

    async def event_emitter(event):
        events.append(event)

    async def get_or_create_compaction_summary(*, on_summary_start=None, **kwargs):
        assert on_summary_start is not None
        await on_summary_start()
        return mod.CompactionSummaryResult(
            "fallback summary",
            checkpoint={
                "summary_text": "fallback summary",
                "summary_token_count": None,
                "summary_meta": {},
            },
        )

    async def estimate_rendered_summary_message_tokens(**kwargs):
        estimate_calls.append(kwargs)
        return 37

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(
        mod,
        "_estimate_rendered_summary_message_tokens",
        estimate_rendered_summary_message_tokens,
    )

    assert (
        await mod._prefetch_compaction_checkpoint(
            request=pipe_request,
            user=pipe_user,
            user_id=pipe_user["id"],
            chat_id=pipe_metadata["chat_id"],
            metadata=pipe_metadata,
            body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
            pipe_function_id="auto_compact",
            summary_model_id="target",
            source_messages=source_messages,
            summary_tool_policy="fallback_on_tool_call",
            historical_message_excerpt_bytes=1024,
            historical_message_excerpt_count=3,
            effective_trigger_input_tokens=1000,
            effective_soft_trigger_input_tokens=800,
            trigger_observed_tokens=850,
            trigger_usage_source="request",
            token_status_detail="before_after",
            event_emitter=event_emitter,
        )
        is True
    )

    assert len(estimate_calls) == 1
    assert estimate_calls[0]["summary_text"] == "fallback summary"
    assert estimate_calls[0]["historical_source_messages"] == source_messages
    prefetched = _status_events(events)[1]["data"]
    assert prefetched["tokens"]["summary"] == 37
    assert "after" not in prefetched["tokens"]
    assert "→" not in prefetched["description"]
    assert "summary" in prefetched["description"].lower()
    assert "summary ≈37 tokens" in prefetched["description"]


@pytest.mark.asyncio
async def test_status_skipped_action_carries_tokens(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1000

    captured = {"forward_bodies": []}
    events = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def compact_body(**kwargs):
        return kwargs["body"], False, 0

    async def forward_target(**kwargs):
        captured["forward_bodies"].append(copy.deepcopy(kwargs["body"]))
        return {"ok": True}

    async def event_emitter(event):
        events.append(event)

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_lookup_pending_checkpoint_for_source_prefix",
        lookup_pending_checkpoint_for_source_prefix,
    )
    monkeypatch.setattr(mod, "_compact_body", compact_body)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_candidate_token_estimate(monkeypatch, 1500)

    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }
    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata, __event_emitter__=event_emitter)

    assert result == {"ok": True}
    status_events = _status_events(events)
    assert status_events[-1]["data"]["action"] == "auto_compaction_skipped"
    assert "Could not compact" in status_events[-1]["data"]["description"]
    assert status_events[-1]["data"]["tokens"]["before"] == 1500
    assert status_events[-1]["data"]["tokens"]["hard_limit"] == 1000


@pytest.mark.asyncio
async def test_summary_file_context_unavailable_stops_before_target_forward(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1000
    captured = {"forward_called": False}
    events = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def compact_body(**kwargs):
        raise mod.SummaryFileContextUnavailable("attached file context unavailable")

    async def forward_target(**kwargs):
        captured["forward_called"] = True
        return {"ok": True}

    async def event_emitter(event):
        events.append(event)

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_lookup_pending_checkpoint_for_source_prefix", lookup_pending_checkpoint_for_source_prefix)
    monkeypatch.setattr(mod, "_compact_body", compact_body)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_candidate_token_estimate(monkeypatch, 1500)

    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "active"},
            ],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result["error"]["code"] == "file_context_unavailable"
    assert "attached file context unavailable" in result["error"]["message"]
    assert captured["forward_called"] is False
    status_events = _status_events(events)
    assert status_events[-1]["data"]["action"] == "auto_compaction_failed"


@pytest.mark.asyncio
async def test_status_before_after_renders_when_after_estimate_fails(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    pipe = mod.Pipe()
    pipe.valves.token_status_detail = "before_after"

    result, events, _captured, _estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        pipe=pipe,
        usage={"total_tokens": 1500, "input_tokens": 1500, "output_tokens": 0},
        estimate_sequence=[None],
        trigger_input_tokens=1000,
    )

    assert result == {"ok": True}
    compacted = _status_events(events)[-1]["data"]
    assert compacted["action"] == "auto_compaction_compacted"
    assert "→" not in compacted["description"]
    assert "after" not in compacted["tokens"]


@pytest.mark.asyncio
async def test_status_always_mode_plus_compaction_emits_both(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    pipe = mod.Pipe()
    pipe.valves.token_status_visibility = "always"
    pipe.valves.trigger_input_tokens = 1000

    result, events, captured, _estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        pipe=pipe,
        usage={"total_tokens": 1500, "input_tokens": 1500, "output_tokens": 0},
        trigger_input_tokens=1000,
    )

    assert result == {"ok": True}
    actions = [event["data"]["action"] for event in _status_events(events)]
    assert actions[0] == "auto_compaction_status"
    assert "auto_compaction_compacting" in actions
    assert "auto_compaction_compacted" in actions
    assert _status_events(events)[0]["data"]["tokens"]["before"] == 1500


@pytest.mark.asyncio
async def test_pipe_compacts_older_tool_loop_results_from_request_scoped_usage(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    first_response = StreamingResponse(
        iter(
            [
                b'data: {"usage": {"prompt_tokens": 150, "completion_tokens": 1}}\n\n',
                b'data: {"choices": [{"delta": {"tool_calls": [{"id": "call-1"}]}}]}\n\n',
            ]
        ),
        media_type="text/event-stream",
    )

    prepared = await mod.prepare_streaming_response(
        first_response,
        request=pipe_request,
        chat_id=pipe_metadata["chat_id"],
        message_id=pipe_metadata["message_id"],
        wrapper_model_id=wrapper_id,
    )
    observed = mod._attach_streaming_completion_observer(
        prepared,
        request=pipe_request,
        chat_id=pipe_metadata["chat_id"],
        message_id=pipe_metadata["message_id"],
        wrapper_model_id=wrapper_id,
    )
    async for _ in observed.body_iterator:
        pass

    captured = {}
    summary_inputs = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_or_create_compaction_summary(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined old tool summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    events = []

    async def event_emitter(event):
        events.append(event)

    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "old result" * 1000},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-2", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
        ],
    }

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert len(summary_inputs) == 1
    assert summary_inputs[0] == body["messages"][:3]
    messages = captured["forward_body"]["messages"]
    assert messages[1:] == body["messages"][3:]
    summary_text = "\n".join(message.get("content", "") for message in messages if message.get("role") == "user")
    assert "combined old tool summary" in summary_text
    assert "<auto_compaction_context>" in summary_text
    assert not any(message.get("tool_call_id") == "call-1" for message in messages)
    assert any(message.get("tool_call_id") == "call-2" for message in messages)
    assert [event["data"]["action"] for event in events if event["type"] == "status"] == [
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]
    status_events = [event for event in events if event["type"] == "status"]
    assert status_events[-1]["data"]["done"] is True
    embed_events = [event for event in events if event["type"] == "embeds"]
    assert len(embed_events) == 1
    assert embed_events[0]["data"]["replace"] is False
    assert len(embed_events[0]["data"]["embeds"]) == 1
    embed_html = embed_events[0]["data"]["embeds"][0]
    assert "combined old tool summary" in embed_html


@pytest.mark.asyncio
async def test_pipe_closes_compaction_status_when_usage_threshold_has_no_safe_prefix(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        return {"ok": True, "messages": kwargs["body"]["messages"]}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_candidate_token_estimate(monkeypatch, 500)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        {
            "model": wrapper_id,
            "stream": True,
            "messages": [{"role": "user", "content": "active"}],
        },
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True, "messages": [{"role": "user", "content": "active"}]}
    status_events = [event for event in events if event["type"] == "status"]
    assert [event["data"]["action"] for event in status_events] == [
        "auto_compaction_compacting",
        "auto_compaction_skipped",
    ]
    assert status_events[-1]["data"]["done"] is True
    assert "Could not compact" in status_events[-1]["data"]["description"]


@pytest.mark.asyncio
async def test_pipe_returns_clear_error_when_latest_user_cannot_fit(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def forward_target(**kwargs):
        raise mod.RetryableContextOverflow("context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [{"role": "user", "content": "x" * 1000000}],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result["error"]["code"] == "active_input_too_large"
    assert "latest user message" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_returns_clear_error_when_latest_tool_result_cannot_be_summarized(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def generate_summary_text(**kwargs):
        raise mod.RetryableContextOverflow("summary context")

    async def forward_target(**kwargs):
        raise mod.RetryableContextOverflow("target context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "x" * 1000000},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result["error"]["code"] == "latest_tool_result_too_large"
    assert "tool result" in result["error"]["message"]


@pytest.mark.asyncio
async def test_pipe_does_not_create_transient_only_history_checkpoint_for_large_tool_result(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    transient_context = "<SYSTEM_CONTEXT>now: 10:00</SYSTEM_CONTEXT>"
    summary_sources = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def generate_summary_text(**kwargs):
        summary_sources.append(copy.deepcopy(kwargs["source_messages"]))
        raise mod.RetryableContextOverflow("summary context")

    async def forward_target(**kwargs):
        raise mod.RetryableContextOverflow("target context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.transient_message_patterns = TRANSIENT_MARKER
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": transient_context},
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "x" * 1000000},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result["error"]["code"] == "latest_tool_result_too_large"
    assert summary_sources == [[body["messages"][0], body["messages"][1]]]


@pytest.mark.asyncio
async def test_pipe_does_not_compact_internal_summary_task(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_or_create_checkpoint_summary(**kwargs):
        raise AssertionError("summary task must not recursively compact")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }
    metadata = {**pipe_metadata, "task": mod.INTERNAL_SUMMARY_TASK}

    await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)

    assert captured["forward_body"]["messages"] == body["messages"]


@pytest.mark.asyncio
async def test_soft_prefetch_emits_status_and_embed_when_checkpoint_is_created(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    events = []
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]

    async def event_emitter(event):
        events.append(event)

    async def get_or_create_compaction_summary(*, on_summary_start=None, **kwargs):
        assert on_summary_start is not None
        await on_summary_start()
        return mod.CompactionSummaryResult(
            "prefetched summary",
            checkpoint={"summary_text": "prefetched summary", "summary_meta": {}},
        )

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    assert (
        await mod._prefetch_compaction_checkpoint(
            request=pipe_request,
            user=pipe_user,
            user_id=pipe_user["id"],
            chat_id=pipe_metadata["chat_id"],
            metadata=pipe_metadata,
            body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
            pipe_function_id="auto_compact",
            summary_model_id="target",
            source_messages=source_messages,
            summary_tool_policy="fallback_on_tool_call",
            historical_message_excerpt_bytes=1024,
            historical_message_excerpt_count=3,
            event_emitter=event_emitter,
        )
        is True
    )

    status_events = [event for event in events if event["type"] == "status"]
    assert [event["data"]["action"] for event in status_events] == [
        "auto_compaction_prefetching",
        "auto_compaction_prefetched",
    ]
    assert status_events[0]["data"]["done"] is False
    assert status_events[1]["data"]["done"] is True
    embed_events = [event for event in events if event["type"] == "embeds"]
    assert len(embed_events) == 1
    assert "prefetched summary" in embed_events[0]["data"]["embeds"][0]


@pytest.mark.asyncio
async def test_soft_prefetch_waits_for_pending_parent_then_claims_child_at_soft_threshold(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    source_messages = [*parent_source, {"role": "user", "content": "middle"}]
    pending = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(parent_source),
        source_message_count=len(parent_source),
        summary_text="",
        summary_meta={},
        parent_checkpoint_id=None,
        state="pending",
        claim_token="claim-1",
        claim_expires_at=9999999999,
    )
    store = PendingTransitionCheckpointStore([pending], ready_after_lookup=2)

    async def noop_initialize(**kwargs):
        return None

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        return 100

    async def generate_summary_text(**kwargs):
        return "child summary"

    async def estimate_rendered_summary_message_tokens(**kwargs):
        return 5

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_POLL_SECONDS", 0)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_estimate_rendered_summary_message_tokens", estimate_rendered_summary_message_tokens)

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
        effective_soft_trigger_input_tokens=100,
    )

    assert prefetched is True
    assert store.lookup_any_count >= 2
    assert len(
        [row for row in store.claimed_rows if row["namespace"] == mod.CHECKPOINT_NAMESPACE]
    ) == 1
    assert store.completed_rows[0]["parent_checkpoint_id"] == pending["id"]
    assert store.completed_rows[0]["summary_text"] == "child summary"


@pytest.mark.asyncio
async def test_soft_prefetch_waits_for_pending_parent_then_skips_below_soft(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    parent_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    source_messages = [*parent_source, {"role": "user", "content": "middle"}]
    pending = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(parent_source),
        source_message_count=len(parent_source),
        summary_text="",
        summary_meta={},
        parent_checkpoint_id=None,
        state="pending",
        claim_token="claim-1",
        claim_expires_at=9999999999,
    )
    store = PendingTransitionCheckpointStore([pending], ready_after_lookup=2)

    async def noop_initialize(**kwargs):
        return None

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        return 99

    async def generate_summary_text(**kwargs):
        raise AssertionError("a below-soft ready parent must skip child generation")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_POLL_SECONDS", 0)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
        effective_soft_trigger_input_tokens=100,
    )

    assert prefetched is False
    assert store.lookup_any_count >= 2
    assert store.claimed_rows == []


@pytest.mark.asyncio
async def test_soft_prefetch_waits_for_pending_exact_checkpoint_then_skips(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    pending = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(source_messages),
        source_message_count=len(source_messages),
        summary_text="",
        summary_meta={},
        parent_checkpoint_id=None,
        state="pending",
        claim_token="claim-1",
        claim_expires_at=9999999999,
    )
    store = PendingTransitionCheckpointStore([pending], ready_after_lookup=2)

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        raise AssertionError("an exact checkpoint that became ready must skip generation")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_POLL_SECONDS", 0)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
        effective_soft_trigger_input_tokens=100,
    )

    assert prefetched is False
    assert store.lookup_any_count >= 2
    assert store.claimed_rows == []


@pytest.mark.asyncio
async def test_soft_prefetch_pending_parent_timeout_is_bounded_and_starts_no_child(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "middle"},
    ]
    pending = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_source_hash(source_messages[:2]),
        source_message_count=2,
        summary_text="",
        summary_meta={},
        parent_checkpoint_id=None,
        state="pending",
        claim_token="claim-1",
        claim_expires_at=9999999999,
    )
    store = PendingTransitionCheckpointStore([pending], ready_after_lookup=None)

    async def noop_initialize(**kwargs):
        return None

    async def generate_summary_text(**kwargs):
        raise AssertionError("a timed-out pending parent must not start a child")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
        effective_soft_trigger_input_tokens=100,
    )

    assert prefetched is False
    assert store.lookup_any_count <= 1
    assert store.claimed_rows == []


@pytest.mark.asyncio
async def test_soft_prefetch_emits_done_status_when_checkpoint_creation_is_cancelled(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    events = []
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]

    async def event_emitter(event):
        events.append(event)

    async def get_or_create_compaction_summary(*, on_summary_start=None, **kwargs):
        assert on_summary_start is not None
        await on_summary_start()
        raise asyncio.CancelledError()

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    with pytest.raises(asyncio.CancelledError):
        await mod._prefetch_compaction_checkpoint(
            request=pipe_request,
            user=pipe_user,
            user_id=pipe_user["id"],
            chat_id=pipe_metadata["chat_id"],
            metadata=pipe_metadata,
            body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
            pipe_function_id="auto_compact",
            summary_model_id="target",
            source_messages=source_messages,
            summary_tool_policy="fallback_on_tool_call",
            historical_message_excerpt_bytes=1024,
            historical_message_excerpt_count=3,
            event_emitter=event_emitter,
        )

    status_events = [event for event in events if event["type"] == "status"]
    assert [event["data"]["action"] for event in status_events] == [
        "auto_compaction_prefetching",
        "auto_compaction_failed",
    ]
    assert status_events[1]["data"]["done"] is True
    assert status_events[1]["data"]["error"] is True
    assert [event for event in events if event["type"] == "embeds"] == []


@pytest.mark.asyncio
async def test_pipe_does_not_launch_completed_turn_soft_prefetch_for_internal_summary_task(
    monkeypatch, pipe_request, pipe_user
):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "summary answer"},
                    "finish_reason": "stop",
                }
            ]
        }

    def start_soft_prefetch(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    metadata = {"chat_id": "chat-1", "message_id": "msg-1", "task": mod.INTERNAL_SUMMARY_TASK}
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)

    assert calls == []


@pytest.mark.asyncio
async def test_pipe_launches_soft_prefetch_below_hard_without_foreground_compaction(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        return 150

    async def generate_summary_text(**kwargs):
        raise AssertionError("soft prefetch must not synchronously generate summaries")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        captured["on_complete"] = kwargs.get("on_complete")
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        assert "target_model_id" not in kwargs
        captured["prefetch"] = copy.deepcopy({key: value for key, value in kwargs.items() if key != "event_emitter"})
        captured["prefetch_event_emitter"] = kwargs.get("event_emitter")
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == body["messages"]
    assert captured["prefetch"]["body"]["messages"] == body["messages"]
    assert captured["prefetch_event_emitter"] is event_emitter
    assert callable(captured["on_complete"])


@pytest.mark.asyncio
async def test_pipe_logs_late_parent_checkpoint_lookup_failure_for_soft_prefetch(
    monkeypatch, caplog, pipe_request, pipe_user, pipe_metadata
):
    calls = {"lookup": 0, "prefetch": 0}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        calls["lookup"] += 1
        if calls["lookup"] == 1:
            return mod.ReusableCheckpointMatch(kind="parent", source_message_count=2)
        raise RuntimeError("late checkpoint lookup failed")

    async def estimate_body_tokens_async(body, **kwargs):
        return 150

    async def compact_body_with_reusable_checkpoint(**kwargs):
        return kwargs["body"], True, 2

    async def forward_target(**kwargs):
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        calls["prefetch"] += 1
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async)
    monkeypatch.setattr(mod, "_compact_body_with_reusable_checkpoint", compact_body_with_reusable_checkpoint)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    with caplog.at_level("WARNING", logger=mod.LOG.name):
        result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert calls == {"lookup": 2, "prefetch": 0}
    assert "late checkpoint lookup failed during soft prefetch check" in caplog.text
    assert "disabling soft prefetch" in caplog.text


@pytest.mark.asyncio
async def test_pipe_uses_full_body_estimate_for_request_usage_without_tool_suffix(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    body_estimates = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_target_db_model_record(model_id):
        assert model_id == "target"
        return SimpleNamespace(
            base_model_id=None,
            params=SimpleNamespace(model_dump=lambda: {"system": "target system"}),
        )

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def parent_assistant_message_id(**kwargs):
        raise AssertionError("route without a shaping hash must not query durable usage anchors")

    async def estimate_messages_tokens_async(messages, **kwargs):
        if not messages:
            return mod.REQUEST_TOKEN_OVERHEAD
        raise AssertionError("request-scoped usage must only anchor a safe tool-result suffix")

    async def estimate_body_tokens_async(body, **kwargs):
        body_estimates.append(copy.deepcopy(body))
        return 120 if body["messages"][0] == {"role": "system", "content": "target system"} else 10

    async def get_or_create_compaction_summary(**kwargs):
        return "full estimate from request usage without tool suffix"

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_target_db_model_record", get_target_db_model_record)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_usage_anchor_parent_assistant_message_id", parent_assistant_message_id)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    mod.store_request_scoped_usage(
        request=pipe_request,
        chat_id=pipe_metadata["chat_id"],
        message_id=pipe_metadata["message_id"],
        wrapper_model_id=wrapper_id,
        usage={"total_tokens": 90, "input_tokens": 70, "output_tokens": 20},
    )
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert len(body_estimates) == 1
    assert body_estimates[0]["messages"][0] == {"role": "system", "content": "target system"}
    assert all(message.get("content") != "target system" for message in captured["forward_body"]["messages"])
    assert "full estimate from request usage without tool suffix" in captured["forward_body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_ignores_body_message_usage_without_request_or_persisted_usage(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    body_estimates = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
        if not messages:
            return mod.REQUEST_TOKEN_OVERHEAD
        raise AssertionError("body.messages usage should not be treated as a normal Open WebUI usage anchor")

    async def estimate_body_tokens_async(body, **kwargs):
        body_estimates.append(copy.deepcopy(body))
        return 120

    async def get_or_create_compaction_summary(**kwargs):
        return "body estimate because message usage is ignored"

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer", "usage": {"total_tokens": 90, "input_tokens": 70, "output_tokens": 20}},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert len(body_estimates) == 1
    assert "body estimate because message usage is ignored" in captured["forward_body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_does_not_prefetch_from_stable_body_extras_already_counted_by_usage_anchor(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    estimated_messages = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
        estimated_messages.extend(copy.deepcopy(messages))
        return 50

    async def estimate_body_extra_tokens_async(*args, **kwargs):
        raise AssertionError("stable body extras are already counted by the observed usage anchor")

    async def estimate_body_tokens_async(*args, **kwargs):
        raise AssertionError("safe usage anchor should avoid full-body token estimation on stable raw shape")

    async def get_or_create_compaction_summary(**kwargs):
        raise AssertionError("stable tool schemas must not force foreground compaction from usage anchoring")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        captured["prefetch"] = copy.deepcopy(kwargs)
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_extra_tokens_async", estimate_body_extra_tokens_async, raising=False)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    _install_durable_usage_anchor_estimate(monkeypatch, 750)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 1000
    pipe.valves.soft_trigger_ratio = 0.8
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "large_tool",
                    "parameters": {"description": "stable huge tool schema payload" * 1000},
                },
            }
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == body["messages"]
    assert "prefetch" not in captured


@pytest.mark.asyncio
async def test_pipe_compacts_from_usage_anchor_plus_latest_user_delta_without_full_body_estimate(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    estimated_messages = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
        estimated_messages.extend(copy.deepcopy(messages))
        return 15

    async def estimate_body_tokens_async(*args, **kwargs):
        raise AssertionError("usage anchor should avoid full-body token estimation on stable raw shape")

    async def get_or_create_compaction_summary(**kwargs):
        return "anchored hard summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_durable_usage_anchor_estimate(monkeypatch, 105)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert "anchored hard summary" in captured["forward_body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_usage_anchor_ignores_trailing_transient_user_delta(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    estimated_messages = []
    transient_context = "<SYSTEM_CONTEXT>now: 10:00</SYSTEM_CONTEXT>"

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
        estimated_messages.extend(copy.deepcopy(messages))
        if messages == [{"role": "user", "content": "active"}]:
            return 15
        if messages == [{"role": "user", "content": transient_context}]:
            return 1
        raise AssertionError(f"unexpected usage-anchor delta: {messages!r}")

    async def estimate_body_tokens_async(*args, **kwargs):
        raise AssertionError("transient-aware usage anchor should avoid full-body token estimation")

    async def get_or_create_compaction_summary(**kwargs):
        return "anchored hard summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_durable_usage_anchor_estimate(monkeypatch, 105)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    pipe.valves.transient_message_patterns = TRANSIENT_MARKER
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
            {"role": "user", "content": transient_context},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert "anchored hard summary" in captured["forward_body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_launches_soft_prefetch_from_usage_anchor_plus_latest_user_delta(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
        return 40

    async def estimate_body_tokens_async(*args, **kwargs):
        raise AssertionError("usage anchor should avoid full-body token estimation on stable raw shape")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        captured["prefetch"] = copy.deepcopy({key: value for key, value in kwargs.items() if key != "event_emitter"})
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    _install_durable_usage_anchor_estimate(monkeypatch, 110)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == body["messages"]
    assert captured["prefetch"]["body"]["messages"] == body["messages"]


@pytest.mark.asyncio
async def test_pipe_compacts_from_request_anchor_plus_tool_loop_suffix_without_full_body_estimate(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    estimated_batches = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def get_target_db_model_record(model_id):
        assert model_id == "target"
        return None

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
        estimated_batches.append(copy.deepcopy(messages))
        return mod.REQUEST_TOKEN_OVERHEAD if not messages else 50

    async def estimate_body_tokens_async(*args, **kwargs):
        raise AssertionError("tool-loop request usage anchor should avoid full-body token estimation for safe trailing tool results")

    async def get_or_create_compaction_summary(**kwargs):
        return "tool loop hard summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_get_target_db_model_record", get_target_db_model_record)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    _install_known_openai_usage_anchor_transport(monkeypatch)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    previous_messages = [{"role": "user", "content": "old"}]
    mod.store_request_scoped_usage(
        request=pipe_request,
        chat_id=pipe_metadata["chat_id"],
        message_id=pipe_metadata["message_id"],
        wrapper_model_id=wrapper_id,
        usage={"total_tokens": 70, "input_tokens": 60, "output_tokens": 10},
        anchor_input=mod.UsageAnchorInput(
            stable_message_count=len(previous_messages),
                input_fingerprint=mod._compute_usage_anchor_input_fingerprint(
                    {"model": "target", "messages": previous_messages},
                    previous_messages,
                    usage_anchor_shaping_hash=_known_empty_model_shaping_hash(),
                ),
            volatile_message_tokens=0,
        ),
    )
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "huge tool result"}
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "big", "arguments": "{}"}}],
            },
            tool_message,
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "big", "arguments": "{}"}}
            ],
        },
        tool_message,
    ] in estimated_batches
    assert "tool loop hard summary" in captured["forward_body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_uses_full_body_estimate_when_visible_usage_suffix_is_not_anchor_safe(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    body_estimates = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
        if not messages:
            return mod.REQUEST_TOKEN_OVERHEAD
        raise AssertionError("unsafe visible usage suffix must not use usage anchor fallback")

    async def estimate_body_tokens_async(body, **kwargs):
        body_estimates.append(copy.deepcopy(body))
        return 150

    async def get_or_create_compaction_summary(**kwargs):
        return "full estimate hard summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer", "usage": {"total_tokens": 80, "input_tokens": 60, "output_tokens": 20}},
            {"role": "user", "content": "middle"},
            {"role": "assistant", "content": "large unmeasured assistant output"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert len(body_estimates) == 1
    assert "full estimate hard summary" in captured["forward_body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_launches_soft_prefetch_from_estimate_when_usage_is_missing(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        return 150

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        captured["prefetch"] = copy.deepcopy({key: value for key, value in kwargs.items() if key != "event_emitter"})
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == body["messages"]
    assert captured["prefetch"]["body"]["messages"] == body["messages"]


@pytest.mark.asyncio
async def test_pipe_does_not_launch_soft_prefetch_when_hard_compaction_will_run(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    def estimate_messages_tokens(messages, **kwargs):
        return 50

    async def get_or_create_compaction_summary(**kwargs):
        return "hard summary"

    async def estimate_body_tokens_async(*args, **kwargs):
        return 200

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        raise AssertionError("hard compaction must not also start soft prefetch")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens", estimate_messages_tokens)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.2
    pipe.valves.trigger_input_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert result == {"ok": True}
    assert "hard summary" in captured["forward_body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_does_not_launch_seed_prefetch_below_soft_when_no_reusable_checkpoint(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    def estimate_messages_tokens(messages, **kwargs):
        return 50

    async def generate_summary_text(**kwargs):
        raise AssertionError("seed prefetch must not synchronously generate summaries")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        assert "target_model_id" not in kwargs
        calls.append(
            {
                **copy.deepcopy({key: value for key, value in kwargs.items() if key != "event_emitter"}),
                "event_emitter": kwargs.get("event_emitter"),
            }
        )
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens", estimate_messages_tokens)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.5
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == body["messages"]
    assert calls == []
    assert events == []


@pytest.mark.asyncio
async def test_pipe_does_not_launch_seed_prefetch_when_soft_prefetch_disabled(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    def estimate_messages_tokens(messages, **kwargs):
        return 50

    async def generate_summary_text(**kwargs):
        raise AssertionError("disabled soft prefetch must not synchronously generate summaries")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens", estimate_messages_tokens)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    events = []

    async def event_emitter(event):
        events.append(event)

    result = await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )

    assert result == {"ok": True}
    assert captured["forward_body"]["messages"] == body["messages"]
    assert calls == []
    assert events == []


def test_soft_prefetch_internal_signatures_do_not_carry_target_model_id():
    start_signature = inspect.signature(mod._start_soft_compaction_prefetch)
    prefetch_signature = inspect.signature(mod._prefetch_compaction_checkpoint)

    assert "target_model_id" not in start_signature.parameters
    assert "target_model_id" not in prefetch_signature.parameters
    assert "source_messages" in prefetch_signature.parameters


def test_soft_prefetch_task_release_logs_background_exception(monkeypatch):
    captured = {}

    class DoneTask:
        def exception(self):
            raise RuntimeError("prefetch failed")

    def log_exception(message, *args):
        captured["message"] = message
        captured["args"] = args

    key = ("namespace", "user", "chat", "pipe", "profile", "source")
    mod._SOFT_PREFETCH_INFLIGHT_KEYS.add(key)
    monkeypatch.setattr(mod.LOG, "exception", log_exception)

    mod._release_soft_prefetch_task(key, DoneTask())

    assert key not in mod._SOFT_PREFETCH_INFLIGHT_KEYS
    assert captured == {
        "message": (
            "Soft compaction prefetch task failed for user_id=%s chat_id=%s "
            "pipe_function_id=%s source_hash=%s"
        ),
        "args": ("user", "chat", "pipe", "source"),
    }


def test_soft_prefetch_task_release_logs_returned_task_exception(monkeypatch):
    captured = {}
    exc = RuntimeError("prefetch failed")

    class DoneTask:
        def exception(self):
            return exc

    def log_error(message, *args, exc_info=None):
        captured["message"] = message
        captured["args"] = args
        captured["exc_info"] = exc_info

    key = ("namespace", "user", "chat", "pipe", "profile", "source")
    mod._SOFT_PREFETCH_INFLIGHT_KEYS.add(key)
    monkeypatch.setattr(mod.LOG, "error", log_error)

    mod._release_soft_prefetch_task(key, DoneTask())

    assert key not in mod._SOFT_PREFETCH_INFLIGHT_KEYS
    assert captured == {
        "message": (
            "Soft compaction prefetch task failed for user_id=%s chat_id=%s "
            "pipe_function_id=%s source_hash=%s"
        ),
        "args": ("user", "chat", "pipe", "source"),
        "exc_info": (RuntimeError, exc, exc.__traceback__),
    }


@pytest.mark.asyncio
async def test_start_soft_prefetch_passes_selected_source_messages_to_task(monkeypatch, pipe_user):
    selected_source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    calls = {"source_selection": 0}
    launched = {}

    def soft_prefetch_source_messages(body, **kwargs):
        calls["source_selection"] += 1
        return copy.deepcopy(selected_source), None

    async def get_or_create_compaction_summary(
        *,
        source_messages,
        preserved_system_message,
        **kwargs,
    ):
        launched["source_messages"] = copy.deepcopy(source_messages)
        launched["preserved_system_message"] = preserved_system_message
        return "summary"

    def launch_soft_prefetch_task(key, coro):
        launched["key"] = key
        launched["coro"] = coro
        return True

    monkeypatch.setattr(mod, "_soft_prefetch_source_messages", soft_prefetch_source_messages)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_launch_soft_prefetch_task", launch_soft_prefetch_task)

    assert (
        mod._start_soft_compaction_prefetch(
            request=object(),
            user=pipe_user,
            metadata={"chat_id": "chat-1"},
            body={
                "model": "target",
                "messages": [
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": "active"},
                ],
            },
            pipe_function_id="auto_compact",
            summary_model_id="target",
            summary_tool_policy="fallback_on_tool_call",
            historical_message_excerpt_bytes=1024,
            historical_message_excerpt_count=3,
        )
        is True
    )

    await launched["coro"]

    assert calls["source_selection"] == 1
    assert launched["source_messages"] == selected_source
    assert launched["preserved_system_message"] is None


@pytest.mark.asyncio
async def test_soft_prefetch_uses_tool_aware_source_prefix(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}

    async def get_or_create_compaction_summary(
        *,
        source_messages,
        **kwargs,
    ):
        captured["source_messages"] = copy.deepcopy(source_messages)
        return "summary"

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    messages = [
        {"role": "user", "content": "active request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "latest result"},
    ]
    prefetch_source = mod._soft_prefetch_source_messages({"model": "target", "messages": messages})
    assert prefetch_source == (messages[:3], None)
    source_messages, preserved_system_message = prefetch_source

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": messages},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        preserved_system_message=preserved_system_message,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
    )

    assert prefetched is True
    assert captured["source_messages"] == messages[:3]


@pytest.mark.asyncio
async def test_soft_prefetch_summary_request_preserves_system_but_identity_excludes_it(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    rows = []
    captured = {}
    system = {"role": "system", "content": "system prompt"}
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    messages = [system, *source_messages, {"role": "user", "content": "active"}]

    async def noop_initialize(**kwargs):
        return None

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured["messages"] = copy.deepcopy(form_data["messages"])
        return {"choices": [{"message": {"content": "prefetch summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))

    prefetch_source = mod._soft_prefetch_source_messages({"model": "target", "messages": messages})
    assert prefetch_source == (source_messages, system)
    selected_source_messages, preserved_system_message = prefetch_source

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": messages},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=selected_source_messages,
        preserved_system_message=preserved_system_message,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
    )

    assert prefetched is True
    assert captured["messages"][:-1] == [system, *source_messages]
    assert captured["messages"][-1]["role"] == "user"
    assert rows[0]["source_message_count"] == len(source_messages)
    assert rows[0]["source_hash"] == mod.compute_source_hash(source_messages)


@pytest.mark.asyncio
async def test_soft_prefetch_skips_checkpoint_when_prefix_has_only_transient_user(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    patterns = mod.parse_transient_message_patterns(TRANSIENT_MARKER)
    transient_context = "<SYSTEM_CONTEXT>now: 10:00</SYSTEM_CONTEXT>"
    messages = [
        {"role": "user", "content": transient_context},
        {"role": "user", "content": "active"},
    ]
    prefetch_source = mod._soft_prefetch_source_messages(
        {"model": "target", "messages": messages},
        transient_message_patterns=patterns,
    )
    assert prefetch_source == ([messages[0]], None)
    source_messages, preserved_system_message = prefetch_source

    async def get_or_create_compaction_summary(**kwargs):
        raise AssertionError("transient-only prefetch prefixes must not create checkpoints")

    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": messages},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        preserved_system_message=preserved_system_message,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
        transient_message_patterns=patterns,
    )

    assert prefetched is False


def test_start_soft_prefetch_preserves_uncopyable_metadata_references(monkeypatch, pipe_user):
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy")

    launched = {}

    def launch_soft_prefetch_task(key, coro):
        launched["key"] = key
        coro.close()
        return True

    monkeypatch.setattr(mod, "_launch_soft_prefetch_task", launch_soft_prefetch_task)

    metadata = {"chat_id": "chat-1", "uncopyable": Uncopyable()}
    body = {
        "model": "target",
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    assert (
        mod._start_soft_compaction_prefetch(
            request=object(),
            user=pipe_user,
            metadata=metadata,
            body=body,
            pipe_function_id="auto_compact",
            summary_model_id="target",
            summary_tool_policy="fallback_on_tool_call",
            historical_message_excerpt_bytes=1024,
            historical_message_excerpt_count=3,
        )
        is True
    )
    assert launched["key"][2] == "chat-1"


@pytest.mark.asyncio
async def test_pipe_launches_completed_turn_soft_prefetch_when_no_parent_prefetch_is_in_flight(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    _disable_usage_anchor_persistence(monkeypatch)
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        response = {
            "usage": {"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ]
        }
        return response

    def start_soft_prefetch(**kwargs):
        assert "target_model_id" not in kwargs
        assert mod._soft_prefetch_source_messages(kwargs["body"])
        calls.append(
            {
                **copy.deepcopy({key: value for key, value in kwargs.items() if key != "event_emitter"}),
                "event_emitter": kwargs.get("event_emitter"),
            }
        )
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_call_target_completion", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    _install_candidate_token_estimate(monkeypatch, 150)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    events = []

    async def event_emitter(event):
        events.append(event)

    await pipe.pipe(
        body,
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=event_emitter,
    )
    await _drain_completed_turn_prefetch_tasks()

    assert len(calls) == 2
    assert all(call.get("event_emitter") is event_emitter for call in calls)
    assert calls[0]["body"]["messages"] == [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
    ]
    assert calls[1]["body"]["messages"] == [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": ""},
    ]


@pytest.mark.asyncio
async def test_pipe_completed_turn_prefetch_waits_for_in_flight_parent_prefetch(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    _disable_usage_anchor_persistence(monkeypatch)
    calls = []
    release_parent_prefetch = asyncio.Event()

    async def parent_prefetch():
        await release_parent_prefetch.wait()

    parent_prefetch_task = asyncio.create_task(parent_prefetch())

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        response = {
            "usage": {"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
        }
        return response

    def in_flight_parent_prefetch(**kwargs):
        return parent_prefetch_task

    def start_soft_prefetch(**kwargs):
        call = copy.deepcopy(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"event_emitter", "parent_prefetch_task"}
            }
        )
        call["parent_prefetch_task"] = kwargs.get("parent_prefetch_task")
        calls.append(call)
        return True

    monkeypatch.setattr(mod, "_SOFT_PREFETCH_TASKS", set())
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_call_target_completion", forward_target)
    monkeypatch.setattr(mod, "_soft_prefetch_inflight_task_for_body", in_flight_parent_prefetch)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    _install_candidate_token_estimate(monkeypatch, 150)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    try:
        result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

        assert result["choices"][0]["message"]["content"] == "answer"
        await _drain_completed_turn_prefetch_tasks()
        assert len(calls) == 2
        assert calls[0]["parent_prefetch_task"] is None
        assert calls[1]["parent_prefetch_task"] is parent_prefetch_task
        assert calls[1]["body"]["messages"][-2:] == [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": ""},
        ]
    finally:
        release_parent_prefetch.set()
        await asyncio.gather(parent_prefetch_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_task_completed_turn_prefetch_skips_summary_generation(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        return {
            "usage": {"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "task answer"},
                    "finish_reason": "stop",
                }
            ],
        }

    def start_soft_prefetch(**kwargs):
        calls.append({key: copy.deepcopy(value) for key, value in kwargs.items() if key != "event_emitter"})
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    pipe.valves.compact_task_prompts_from_task_body = False
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    task_history = [
        {"role": "user", "content": "old task input"},
        {"role": "assistant", "content": "old task answer"},
        {"role": "user", "content": "active task input"},
    ]
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [{"role": "user", "content": "Task:\nold task input\nactive task input"}],
    }
    metadata = {
        **pipe_metadata,
        "task": mod.TASKS.TAGS_GENERATION.value,
        "task_body": {
            "model": wrapper_id,
            "chat_id": pipe_metadata["chat_id"],
            "messages": task_history,
        },
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)
    await _drain_completed_turn_prefetch_tasks()

    assert result["choices"][0]["message"]["content"] == "task answer"
    assert calls == []


@pytest.mark.asyncio
async def test_streaming_task_completed_turn_prefetch_does_not_register_on_complete(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    calls = []
    rebuild_calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def rebuild_task_body_from_compacted_history(**kwargs):
        rebuild_calls.append(kwargs)
        return {"messages": [{"role": "user", "content": "rebuilt completed task prompt"}]}

    async def forward_target(**kwargs):
        captured["on_complete"] = kwargs.get("on_complete")
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        calls.append({key: copy.deepcopy(value) for key, value in kwargs.items() if key != "event_emitter"})
        return True

    monkeypatch.setattr(mod, "_SOFT_PREFETCH_TASKS", set())
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_rebuild_task_body_from_compacted_history", rebuild_task_body_from_compacted_history)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    pipe.valves.compact_task_prompts_from_task_body = False
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    task_history = [
        {"role": "user", "content": "old task input"},
        {"role": "assistant", "content": "old task answer"},
        {"role": "user", "content": "active task input"},
    ]
    body = {
        "model": wrapper_id,
        "stream": True,
        "messages": [{"role": "user", "content": "Task:\nold task input\nactive task input"}],
    }
    metadata = {
        **pipe_metadata,
        "task": mod.TASKS.TAGS_GENERATION.value,
        "task_body": {
            "model": wrapper_id,
            "chat_id": pipe_metadata["chat_id"],
            "messages": task_history,
        },
    }

    await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)

    assert captured["on_complete"] is None
    assert mod._SOFT_PREFETCH_TASKS == set()
    assert calls == []
    assert rebuild_calls == []


@pytest.mark.asyncio
async def test_pipe_skips_completed_turn_soft_prefetch_for_tool_call_response(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    _disable_usage_anchor_persistence(monkeypatch)
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call-1", "type": "function"}],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        return response

    def start_soft_prefetch(**kwargs):
        assert "target_model_id" not in kwargs
        assert mod._soft_prefetch_source_messages(kwargs["body"])
        calls.append(copy.deepcopy(kwargs))
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_call_target_completion", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    _install_candidate_token_estimate(monkeypatch, 150)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    assert len(calls) == 1
    assert calls[0]["body"]["messages"] == body["messages"]


async def _run_completed_turn_prefetch_usage_case(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
    *,
    response_usage=None,
    body_reusable_checkpoint_match=None,
    messages=None,
):
    _disable_usage_anchor_persistence(monkeypatch)
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ]
        }
        if response_usage is not None:
            response["usage"] = response_usage
        return response

    def start_soft_prefetch(**kwargs):
        calls.append(copy.deepcopy({key: value for key, value in kwargs.items() if key != "event_emitter"}))
        return True

    body_reusable_checkpoint_match = body_reusable_checkpoint_match or reusable_checkpoint_match

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_call_target_completion", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": messages
        or [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)

    return calls


async def _drain_completed_turn_prefetch_tasks() -> None:
    retained_tasks = list(mod._SOFT_PREFETCH_TASKS)
    if retained_tasks:
        await asyncio.wait_for(asyncio.gather(*retained_tasks), timeout=1)
    await asyncio.sleep(0)


async def _wait_for_completed_turn_child_task(completed_key):
    async with asyncio.timeout(1):
        while True:
            task = mod._SOFT_PREFETCH_INFLIGHT_TASKS.get(completed_key)
            if task is not None:
                return task
            await asyncio.sleep(0)


def _completed_turn_registry_case(monkeypatch):
    _disable_usage_anchor_persistence(monkeypatch)
    child_calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        response = {
            "usage": {"total_tokens": 500, "prompt_tokens": 400, "completion_tokens": 100},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
        }
        return response

    async def prefetch_compaction_checkpoint(**kwargs):
        child_calls.append(
            {
                "body": copy.deepcopy(kwargs["body"]),
                "source_messages": copy.deepcopy(kwargs["source_messages"]),
                "trigger_observed_tokens": kwargs["trigger_observed_tokens"],
            }
        )
        return True

    monkeypatch.setattr(mod, "_SOFT_PREFETCH_TASKS", set())
    monkeypatch.setattr(mod, "_SOFT_PREFETCH_INFLIGHT_KEYS", set())
    monkeypatch.setattr(mod, "_SOFT_PREFETCH_INFLIGHT_TASKS", {})
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_call_target_completion", forward_target)
    monkeypatch.setattr(mod, "_prefetch_compaction_checkpoint", prefetch_compaction_checkpoint)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    body = {
        "model": mod.build_wrapper_model_id("auto_compact", "target"),
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }
    return pipe, body, child_calls


def _install_registry_parent_prefetch(pipe_identity, body, release_parent):
    pipe_user, pipe_metadata = pipe_identity

    async def parent_prefetch():
        await release_parent.wait()

    parent_key = mod._soft_prefetch_inflight_key_for_body(
        user=pipe_user,
        metadata=pipe_metadata,
        body=body,
        pipe_function_id="auto_compact",
    )
    assert parent_key is not None
    assert mod._launch_soft_prefetch_task(parent_key, parent_prefetch()) is True
    return mod._SOFT_PREFETCH_INFLIGHT_TASKS[parent_key]


def _completed_turn_registry_key(pipe_user, pipe_metadata, body):
    completed_body = mod._completed_turn_prefetch_body(
        body,
        {"role": "assistant", "content": "answer"},
    )
    assert completed_body is not None
    completed_key = mod._soft_prefetch_inflight_key_for_body(
        user=pipe_user,
        metadata=pipe_metadata,
        body=completed_body,
        pipe_function_id="auto_compact",
    )
    assert completed_key is not None
    return completed_key


@pytest.mark.asyncio
async def test_completed_turn_coordinator_owns_completed_body_key_before_parent_wait(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    pipe, body, child_calls = _completed_turn_registry_case(monkeypatch)
    release_parent = asyncio.Event()
    parent_task = _install_registry_parent_prefetch((pipe_user, pipe_metadata), body, release_parent)

    try:
        result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)
        completed_key = _completed_turn_registry_key(pipe_user, pipe_metadata, body)
        coordinator = await _wait_for_completed_turn_child_task(completed_key)

        assert result["choices"][0]["message"]["content"] == "answer"
        assert coordinator is not None
        assert coordinator in mod._SOFT_PREFETCH_TASKS
        assert completed_key in mod._SOFT_PREFETCH_INFLIGHT_KEYS
        assert child_calls == []
    finally:
        release_parent.set()
        await asyncio.gather(parent_task, return_exceptions=True)
        await _drain_completed_turn_prefetch_tasks()


@pytest.mark.asyncio
async def test_completed_turn_coordinator_rejects_duplicate_completed_body_attempts(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    pipe, body, child_calls = _completed_turn_registry_case(monkeypatch)
    release_parent = asyncio.Event()
    _install_registry_parent_prefetch((pipe_user, pipe_metadata), body, release_parent)

    try:
        await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)
        await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)
        await asyncio.sleep(0)
        completed_key = _completed_turn_registry_key(pipe_user, pipe_metadata, body)

        assert completed_key in mod._SOFT_PREFETCH_INFLIGHT_KEYS
        assert len([task for key, task in mod._SOFT_PREFETCH_INFLIGHT_TASKS.items() if key == completed_key]) == 1
        assert child_calls == []

        release_parent.set()
        await _drain_completed_turn_prefetch_tasks()

        assert len(child_calls) == 1
    finally:
        release_parent.set()
        for task in list(mod._SOFT_PREFETCH_TASKS):
            if not task.done():
                task.cancel()
        await asyncio.gather(*list(mod._SOFT_PREFETCH_TASKS), return_exceptions=True)


@pytest.mark.asyncio
async def test_completed_turn_coordinator_parent_wait_timeout_starts_no_child(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    pipe, body, child_calls = _completed_turn_registry_case(monkeypatch)
    release_parent = asyncio.Event()
    parent_task = _install_registry_parent_prefetch((pipe_user, pipe_metadata), body, release_parent)
    monkeypatch.setattr(mod, "CHECKPOINT_PENDING_WAIT_TIMEOUT_SECONDS", 0)

    try:
        await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)
        await asyncio.sleep(0)
        coordinators = [task for task in mod._SOFT_PREFETCH_TASKS if task is not parent_task]
        assert len(coordinators) == 1

        await asyncio.wait_for(asyncio.shield(coordinators[0]), timeout=0.1)

        assert child_calls == []
    finally:
        release_parent.set()
        for task in list(mod._SOFT_PREFETCH_TASKS):
            if not task.done():
                task.cancel()
        await asyncio.gather(parent_task, *list(mod._SOFT_PREFETCH_TASKS), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_completed_turn_coordinator_while_waiting_starts_no_child(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    pipe, body, child_calls = _completed_turn_registry_case(monkeypatch)
    release_parent = asyncio.Event()
    parent_task = _install_registry_parent_prefetch((pipe_user, pipe_metadata), body, release_parent)

    try:
        await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)
        completed_key = _completed_turn_registry_key(pipe_user, pipe_metadata, body)
        coordinator = await _wait_for_completed_turn_child_task(completed_key)
        assert coordinator is not None

        coordinator.cancel()
        await asyncio.gather(coordinator, return_exceptions=True)
        await asyncio.sleep(0)

        assert child_calls == []
        assert completed_key not in mod._SOFT_PREFETCH_INFLIGHT_KEYS
        assert completed_key not in mod._SOFT_PREFETCH_INFLIGHT_TASKS
    finally:
        release_parent.set()
        await asyncio.gather(parent_task, return_exceptions=True)
        await _drain_completed_turn_prefetch_tasks()


@pytest.mark.asyncio
async def test_cancelled_parent_prefetch_starts_no_completed_turn_child(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    pipe, body, child_calls = _completed_turn_registry_case(monkeypatch)
    release_parent = asyncio.Event()
    parent_task = _install_registry_parent_prefetch((pipe_user, pipe_metadata), body, release_parent)

    try:
        await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)
        completed_key = _completed_turn_registry_key(pipe_user, pipe_metadata, body)
        coordinator = await _wait_for_completed_turn_child_task(completed_key)
        assert coordinator is not None

        parent_task.cancel()
        await asyncio.gather(parent_task, return_exceptions=True)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(coordinator), timeout=0.1)

        assert coordinator.cancelled()
        assert child_calls == []
    finally:
        release_parent.set()
        for task in list(mod._SOFT_PREFETCH_TASKS):
            if not task.done():
                task.cancel()
        await asyncio.gather(*list(mod._SOFT_PREFETCH_TASKS), return_exceptions=True)


@pytest.mark.asyncio
async def test_pipe_completed_turn_prefetch_requires_response_usage(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    calls = await _run_completed_turn_prefetch_usage_case(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
    )

    assert calls == []


@pytest.mark.asyncio
async def test_pipe_completed_turn_prefetch_fires_from_response_usage(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    calls = await _run_completed_turn_prefetch_usage_case(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        response_usage={"total_tokens": 500, "prompt_tokens": 400, "completion_tokens": 100},
    )

    await _drain_completed_turn_prefetch_tasks()

    assert len(calls) == 1
    assert calls[0]["trigger_observed_tokens"] == 500
    assert calls[0]["body"]["messages"][-2:] == [
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": ""},
    ]


@pytest.mark.asyncio
async def test_pipe_completed_turn_prefetch_requires_core_persisted_message_id(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    metadata = {**pipe_metadata, "user_message_id": pipe_metadata["message_id"]}
    metadata.pop("message_id")

    calls = await _run_completed_turn_prefetch_usage_case(
        monkeypatch,
        pipe_request,
        pipe_user,
        metadata,
        response_usage={"total_tokens": 500, "prompt_tokens": 400, "completion_tokens": 100},
    )

    await _drain_completed_turn_prefetch_tasks()

    assert calls == []


@pytest.mark.asyncio
async def test_pipe_skips_checkpoints_without_core_persisted_message_id(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    checkpoint_lookups = []
    prefetch_calls = []
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        checkpoint_lookups.append(kwargs)
        return None

    async def forward_target(**kwargs):
        captured["body"] = copy.deepcopy(kwargs["body"])
        return {
            "usage": {"total_tokens": 500, "prompt_tokens": 400, "completion_tokens": 100},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
        }

    def start_soft_prefetch(**kwargs):
        prefetch_calls.append(kwargs)
        return True

    monkeypatch.setattr(mod, "_SOFT_PREFETCH_TASKS", set())
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }
    metadata = {"chat_id": pipe_metadata["chat_id"]}

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=metadata)
    await _drain_completed_turn_prefetch_tasks()

    assert result["choices"][0]["message"]["content"] == "answer"
    assert checkpoint_lookups == []
    assert prefetch_calls == []
    assert captured["body"] == {**body, "model": "target", "metadata": metadata}


@pytest.mark.asyncio
async def test_pipe_completed_turn_prefetch_skips_parent_lookup_without_parent_key(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    parent_lookups = []

    def lookup_parent_task(**kwargs):
        parent_lookups.append(kwargs)
        return None

    monkeypatch.setattr(mod, "_soft_prefetch_inflight_task_for_body", lookup_parent_task)

    calls = await _run_completed_turn_prefetch_usage_case(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        response_usage={"total_tokens": 500, "prompt_tokens": 400, "completion_tokens": 100},
        messages=[{"role": "user", "content": "large first turn"}],
    )

    await _drain_completed_turn_prefetch_tasks()

    assert parent_lookups == []
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_pipe_completed_turn_prefetch_background_error_does_not_block_response(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    _disable_usage_anchor_persistence(monkeypatch)
    exception_logs = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        return {
            "usage": {"total_tokens": 500, "prompt_tokens": 400, "completion_tokens": 100},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
        }

    def start_soft_prefetch(**kwargs):
        raise RuntimeError("prefetch failed")

    def log_exception(message, *args, **kwargs):
        exception_logs.append((message, kwargs.get("exc_info")))

    monkeypatch.setattr(mod, "_SOFT_PREFETCH_TASKS", set())
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_call_target_completion", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod.LOG, "exception", log_exception)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    result = await pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata)
    retained_tasks = list(mod._SOFT_PREFETCH_TASKS)
    if retained_tasks:
        await asyncio.gather(*retained_tasks, return_exceptions=True)
    await asyncio.sleep(0)

    assert result["choices"][0]["message"]["content"] == "answer"
    assert mod._SOFT_PREFETCH_TASKS == set()
    assert len(exception_logs) == 1
    assert exception_logs[0][0] == "Completed-turn soft compaction prefetch failed"
    exc_info = exception_logs[0][1]
    assert exc_info is not None
    assert exc_info[0] is RuntimeError


@pytest.mark.asyncio
async def test_pipe_completed_turn_prefetch_does_not_lookup_or_estimate_before_return(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    _disable_usage_anchor_persistence(monkeypatch)
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()
    preparation_started = asyncio.Event()
    preparation_thread_ids = []
    event_loop_thread_id = threading.get_ident()
    calls = []
    completed_turn_prefetch_body = mod._completed_turn_prefetch_body

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def resolve_core_chat_model_route(request, model_id, **kwargs):
        return mod.CoreChatModelRoute(model_id=model_id)

    async def body_reusable_checkpoint_match(**kwargs):
        body = kwargs["body"]
        if len(body["messages"]) < 5:
            return None
        lookup_started.set()
        await release_lookup.wait()
        return None

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        raise AssertionError("completed-turn prefetch must not estimate before the response returns")

    async def estimate_task_checkpoint_applied_body_tokens(**kwargs):
        raise AssertionError("completed-turn task prefetch must not estimate before the response returns")

    async def forward_target(**kwargs):
        return {
            "usage": {"total_tokens": 500, "prompt_tokens": 400, "completion_tokens": 100},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
        }

    def start_soft_prefetch(**kwargs):
        calls.append(copy.deepcopy({key: value for key, value in kwargs.items() if key != "event_emitter"}))
        return True

    def prepare_completed_turn_prefetch_body(body, assistant_message):
        preparation_thread_ids.append(threading.get_ident())
        preparation_started.set()
        return completed_turn_prefetch_body(body, assistant_message)

    monkeypatch.setattr(mod, "_SOFT_PREFETCH_TASKS", set())
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_core_chat_model_route)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_estimate_task_checkpoint_applied_body_tokens", estimate_task_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_call_target_completion", forward_target)
    monkeypatch.setattr(mod, "_completed_turn_prefetch_body", prepare_completed_turn_prefetch_body)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_input_tokens = 1000
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
    }

    pipe_task = asyncio.create_task(pipe.pipe(body, __request__=pipe_request, __user__=pipe_user, __metadata__=pipe_metadata))
    try:
        result = await asyncio.wait_for(pipe_task, timeout=0.1)
        assert not lookup_started.is_set()
        async with asyncio.timeout(1):
            while not preparation_started.is_set():
                await asyncio.sleep(0)
    finally:
        release_lookup.set()
        if not pipe_task.done():
            pipe_task.cancel()
            await asyncio.gather(pipe_task, return_exceptions=True)
        await _drain_completed_turn_prefetch_tasks()

    assert result["choices"][0]["message"]["content"] == "answer"
    assert len(calls) == 1
    assert preparation_thread_ids and preparation_thread_ids[0] != event_loop_thread_id


@pytest.mark.asyncio
async def test_soft_prefetch_worker_skips_exact_reusable_checkpoint(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]

    async def build_prefix_file_fingerprint_resolver(*args, **kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return mod.ReusableCheckpointMatch(
            kind="exact",
            source_message_count=len(source_messages),
            source_kind="message",
            checkpoint={
                "source_message_count": len(source_messages),
                "source_hash": mod.compute_summary_source_hash(source_messages),
            },
        )

    async def get_or_create_compaction_summary(**kwargs):
        raise AssertionError("exact reusable checkpoint must skip background summary generation")

    monkeypatch.setattr(mod, "_build_prefix_file_fingerprint_resolver", build_prefix_file_fingerprint_resolver)
    monkeypatch.setattr(mod, "_lookup_pending_checkpoint_for_source_prefix", lookup_pending_checkpoint_for_source_prefix)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
        effective_soft_trigger_input_tokens=100,
    )

    assert prefetched is False


@pytest.mark.asyncio
async def test_soft_prefetch_worker_logs_and_skips_when_parent_applied_estimate_is_unavailable(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    error_logs = []
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "middle"},
    ]

    async def build_prefix_file_fingerprint_resolver(*args, **kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return mod.ReusableCheckpointMatch(
            kind="parent",
            source_message_count=2,
            source_kind="message",
            checkpoint={
                "source_message_count": 2,
                "source_hash": mod.compute_summary_source_hash(source_messages[:2]),
            },
        )

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        return None

    async def get_or_create_compaction_summary(**kwargs):
        raise AssertionError("soft prefetch must not generate without a checkpoint-applied threshold estimate")

    def log_error(message, *args, **kwargs):
        error_logs.append(message % args)

    monkeypatch.setattr(mod, "_build_prefix_file_fingerprint_resolver", build_prefix_file_fingerprint_resolver)
    monkeypatch.setattr(mod, "_lookup_pending_checkpoint_for_source_prefix", lookup_pending_checkpoint_for_source_prefix)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod.LOG, "error", log_error)

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
        effective_soft_trigger_input_tokens=100,
    )

    assert prefetched is False
    assert error_logs == [
        "auto-compaction prefetch: checkpoint-applied token estimate was unavailable; "
        "skipping background checkpoint generation "
        f"(user_id={pipe_user['id']} chat_id={pipe_metadata['chat_id']} source_kind=message match_kind=parent)"
    ]


@pytest.mark.asyncio
async def test_soft_prefetch_worker_skips_parent_below_soft(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "middle"},
    ]

    async def build_prefix_file_fingerprint_resolver(*args, **kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return mod.ReusableCheckpointMatch(
            kind="parent",
            source_message_count=2,
            source_kind="message",
            checkpoint={
                "source_message_count": 2,
                "source_hash": mod.compute_summary_source_hash(source_messages[:2]),
            },
        )

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        assert kwargs["token_system_prompt"] == "target system"
        assert kwargs["dropped_message_keys"] == {"reasoning_details"}
        return 40

    async def get_or_create_compaction_summary(**kwargs):
        raise AssertionError("below-soft parent-applied estimate must skip background summary generation")

    monkeypatch.setattr(mod, "_build_prefix_file_fingerprint_resolver", build_prefix_file_fingerprint_resolver)
    monkeypatch.setattr(mod, "_lookup_pending_checkpoint_for_source_prefix", lookup_pending_checkpoint_for_source_prefix)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
        effective_soft_trigger_input_tokens=100,
        token_system_prompt="target system",
        dropped_message_keys=frozenset({"reasoning_details"}),
    )

    assert prefetched is False


@pytest.mark.asyncio
async def test_soft_prefetch_worker_proceeds_when_parent_applied_estimate_above_soft(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    source_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "middle"},
    ]

    async def build_prefix_file_fingerprint_resolver(*args, **kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return mod.ReusableCheckpointMatch(
            kind="parent",
            source_message_count=2,
            source_kind="message",
            checkpoint={
                "source_message_count": 2,
                "source_hash": mod.compute_summary_source_hash(source_messages[:2]),
            },
        )

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        return 150

    async def get_or_create_compaction_summary(**kwargs):
        await kwargs["parent_checkpoint_guard"](
            {
                "source_message_count": 2,
                "source_hash": mod.compute_summary_source_hash(source_messages[:2]),
            }
        )
        captured["source_messages"] = copy.deepcopy(kwargs["source_messages"])
        return "summary"

    monkeypatch.setattr(mod, "_build_prefix_file_fingerprint_resolver", build_prefix_file_fingerprint_resolver)
    monkeypatch.setattr(mod, "_lookup_pending_checkpoint_for_source_prefix", lookup_pending_checkpoint_for_source_prefix)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    prefetched = await mod._prefetch_compaction_checkpoint(
        request=pipe_request,
        user=pipe_user,
        user_id=pipe_user["id"],
        chat_id=pipe_metadata["chat_id"],
        metadata=pipe_metadata,
        body={"model": "target", "messages": [*source_messages, {"role": "user", "content": "active"}]},
        pipe_function_id="auto_compact",
        summary_model_id="target",
        source_messages=source_messages,
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
        effective_soft_trigger_input_tokens=100,
    )

    assert prefetched is True
    assert captured["source_messages"] == source_messages


@pytest.mark.asyncio
async def test_pipe_completed_turn_prefetch_still_requires_usage_even_when_checkpoint_ready(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    async def body_reusable_checkpoint_match(**kwargs):
        body = kwargs["body"]
        if len(body["messages"]) < 5:
            return None
        return mod.ReusableCheckpointMatch(
            kind="exact",
            source_message_count=len(body["messages"]),
            source_kind="message",
            checkpoint={"source_message_count": len(body["messages"]), "source_hash": "completed-body"},
        )

    calls = await _run_completed_turn_prefetch_usage_case(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        body_reusable_checkpoint_match=body_reusable_checkpoint_match,
    )

    assert calls == []


@pytest.mark.asyncio
async def test_streaming_completion_observer_merges_raw_usage_and_awaits_callback():
    observed = []
    callback_finished = False
    request = SimpleNamespace(state=SimpleNamespace())
    chunks = [
        b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n',
        b'data: {"usage": {"input_tokens": 60, "cache_read_input_tokens": 40}}\n\n',
        b'data: {"usage": {"output_tokens": 5}}\n\n',
    ]
    response = StreamingResponse(iter(chunks), media_type="text/event-stream")
    anchor_input = mod.UsageAnchorInput(
        stable_message_count=2,
        input_fingerprint="fingerprint",
        volatile_message_tokens=10,
    )

    async def on_complete(completion):
        nonlocal callback_finished
        await asyncio.sleep(0)
        observed.append(completion)
        callback_finished = True

    wrapped = mod._attach_streaming_completion_observer(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
        anchor_input=anchor_input,
        on_complete=on_complete,
    )
    emitted = [chunk async for chunk in wrapped.body_iterator]

    assert emitted == chunks
    assert callback_finished is True
    assert len(observed) == 1
    assert observed[0]["assistant_message"] == {"role": "assistant", "content": "hello"}
    assert observed[0]["raw_usage"] == {
        "input_tokens": 60,
        "cache_read_input_tokens": 40,
        "output_tokens": 5,
    }
    assert mod.get_request_scoped_usage_anchor(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    ) == mod.UsageAnchor(
        assistant_message_id="request",
        input_tokens=100,
        stable_message_count=2,
        input_fingerprint="fingerprint",
        volatile_message_tokens=10,
    )


@pytest.mark.asyncio
async def test_streaming_completion_observer_reads_immediate_responses_output():
    payload = {
        "type": "response.completed",
        "response": {
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "answer"}],
                }
            ],
        },
    }
    response = StreamingResponse(
        iter([f"data: {json.dumps(payload)}\n\n".encode()]),
        media_type="text/event-stream",
    )
    observed = []
    wrapped = mod._attach_streaming_completion_observer(response, observed.append)

    async for _ in wrapped.body_iterator:
        pass

    assert observed[0]["assistant_message"]["content"] == "answer"
    assert observed[0]["raw_usage"] == {"input_tokens": 10, "output_tokens": 2}


@pytest.mark.asyncio
async def test_streaming_completion_observer_skips_tool_call_completion():
    observed = []
    request = SimpleNamespace(state=SimpleNamespace())
    chunks = [
        b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1"}]}}]}\n\n',
        b'data: {"usage": {"prompt_tokens": 150}}\n\n',
    ]
    response = StreamingResponse(iter(chunks), media_type="text/event-stream")
    anchor_input = mod.UsageAnchorInput(
        stable_message_count=1,
        input_fingerprint="fingerprint",
        volatile_message_tokens=0,
    )

    async def on_complete(completion):
        observed.append(completion)

    wrapped = mod._attach_streaming_completion_observer(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
        anchor_input=anchor_input,
        on_complete=on_complete,
    )
    emitted = [chunk async for chunk in wrapped.body_iterator]

    assert emitted == chunks
    assert observed == []
    assert mod.get_request_scoped_usage(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    )["total_tokens"] == 150
    assert mod.get_request_scoped_usage_anchor(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    ) == mod.UsageAnchor(
        assistant_message_id="request",
        input_tokens=150,
        stable_message_count=1,
        input_fingerprint="fingerprint",
        volatile_message_tokens=0,
    )


@pytest.mark.asyncio
async def test_streaming_completion_observer_closes_inner_iterator_on_early_close():
    closed = False
    chunk = b'data: {"choices": [{"delta": {"content": "partial"}}], "usage": {"input_tokens": 50}}\n\n'
    request = SimpleNamespace(state=SimpleNamespace())

    async def chunks():
        nonlocal closed
        try:
            yield chunk
            await asyncio.Event().wait()
        finally:
            closed = True

    observed = []
    response = StreamingResponse(chunks(), media_type="text/event-stream")

    async def on_complete(completion):
        observed.append(completion)

    wrapped = mod._attach_streaming_completion_observer(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
        on_complete=on_complete,
    )

    assert await wrapped.body_iterator.__anext__() == chunk
    await wrapped.body_iterator.aclose()

    assert closed is True
    assert observed == []
    assert mod.get_request_scoped_usage(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    ) is None


@pytest.mark.asyncio
async def test_streaming_completion_observer_close_failure_does_not_mask_stream_failure():
    class StreamFailure(Exception):
        pass

    class CleanupFailure(Exception):
        pass

    class FailingIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StreamFailure

        async def aclose(self):
            raise CleanupFailure

    observed = mod._streaming_completion_observer(
        FailingIterator(),
        media_type="text/event-stream",
        request=None,
        chat_id=None,
        message_id=None,
        wrapper_model_id=None,
        on_complete=lambda _completion: None,
    )

    with pytest.raises(StreamFailure):
        await observed.__anext__()


# ---------------------------------------------------------------------------
# summary_model dropdown options
# ---------------------------------------------------------------------------


def _teardown_summary_model_cache():
    mod._LATEST_MODELS_CACHE.clear()
    latest_provider_states = getattr(mod, "_LATEST_PROVIDER_MODEL_CACHE_ENABLED_STATES", None)
    if latest_provider_states is not None:
        latest_provider_states.clear()
    if hasattr(mod, "_LATEST_PROVIDER_MODEL_CACHE_STATE_ID"):
        mod._LATEST_PROVIDER_MODEL_CACHE_STATE_ID = None


def test_build_summary_model_options_has_no_empty_default_option():
    assert mod.build_summary_model_options({}) == []


def test_build_summary_model_options_excludes_wrapper_and_arena():
    models = {
        "gpt-4o": {"id": "gpt-4o", "name": "GPT-4o"},
        "auto_compact.gpt-4o": {
            "id": "auto_compact.gpt-4o",
            "name": "Auto Compact GPT-4o",
        },
        "arena-model": {"id": "arena-model", "name": "Arena", "arena": True},
        "claude-sonnet": {"id": "claude-sonnet", "name": "Claude Sonnet"},
    }
    options = mod.build_summary_model_options(models, pipe_function_id="auto_compact")
    values = [opt["value"] for opt in options]
    assert "gpt-4o" in values
    assert "claude-sonnet" in values
    assert "auto_compact.gpt-4o" not in values
    assert "arena-model" not in values


def test_build_summary_model_options_excludes_wrapper_based_models():
    models = {
        "custom-wrapper": {
            "id": "custom-wrapper",
            "name": "Custom Wrapper",
            "info": {"base_model_id": "auto_compact.gpt-4o"},
        },
        "real-model": {"id": "real-model", "name": "Real Model"},
    }
    options = mod.build_summary_model_options(models, pipe_function_id="auto_compact")
    values = [opt["value"] for opt in options]
    assert "real-model" in values
    assert "custom-wrapper" not in values


def test_build_summary_model_options_sorted_by_model_id():
    models = {
        "z-model": {"id": "z-model", "name": "Z Model"},
        "a-model": {"id": "a-model", "name": "A Model"},
    }
    options = mod.build_summary_model_options(models, pipe_function_id="auto_compact")
    assert options[0]["value"] == "a-model"
    assert options[1]["value"] == "z-model"


def test_build_summary_model_options_label_format():
    models = {
        "gpt-4o": {"id": "gpt-4o", "name": "GPT-4o"},
        "no-name": {"id": "no-name"},
    }
    options = mod.build_summary_model_options(models, pipe_function_id="auto_compact")
    by_value = {opt["value"]: opt["label"] for opt in options}
    assert by_value["gpt-4o"] == "GPT-4o (gpt-4o)"
    assert by_value["no-name"] == "no-name"


@pytest.mark.asyncio
async def test_get_summary_model_options_keeps_config_disabled_provider_cache_hidden_after_async_refresh(
    monkeypatch,
):
    _teardown_summary_model_cache()
    captured = {}

    async def sync_wrapper_model_records(**kwargs):
        captured["synced_targets"] = [model["id"] for model in kwargs["target_models"]]

    class FakeConfig:
        @staticmethod
        async def get_many(*keys):
            return {"openai.enable": False, "ollama.enable": True}

    state = SimpleNamespace(
        MODELS={
            "stale-openai": {"id": "stale-openai", "name": "Stale OpenAI", "openai": {}},
            "stale-ollama": {"id": "stale-ollama", "name": "Stale Ollama", "ollama": {}},
        },
        BASE_MODELS=[],
        OPENAI_MODELS={"direct-openai": {"id": "direct-openai", "name": "Direct OpenAI", "openai": {}}},
        OLLAMA_MODELS={"direct-ollama": {"model": "direct-ollama", "name": "Direct Ollama"}},
        config=SimpleNamespace(),
    )
    config_module = types.ModuleType("open_webui.models.config")
    config_module.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "open_webui.models.config", config_module)
    monkeypatch.setitem(sys.modules, "open_webui.main", types.SimpleNamespace(app=types.SimpleNamespace(state=state)))
    monkeypatch.setattr(mod, "sync_wrapper_model_records", sync_wrapper_model_records)

    await mod.Pipe().pipes()
    options = mod.Pipe.Valves.get_summary_model_options()
    values = [opt["value"] for opt in options]

    assert captured["synced_targets"] == ["stale-ollama", "direct-ollama"]
    assert values == ["direct-ollama", "stale-ollama"]
    _teardown_summary_model_cache()


def test_update_latest_models_cache_snapshots_models_first_duplicate_wins():
    _teardown_summary_model_cache()
    models = [
        {"id": "gpt-4o", "name": "GPT-4o Override"},
        {"id": "claude", "name": "Claude"},
        {"id": "gpt-4o", "name": "GPT-4o Base"},
        {"not-a-dict"},
    ]
    mod.update_latest_models_cache(models)
    assert set(mod._LATEST_MODELS_CACHE.keys()) == {"gpt-4o", "claude"}
    assert mod._LATEST_MODELS_CACHE["gpt-4o"]["name"] == "GPT-4o Override"
    _teardown_summary_model_cache()


def test_update_latest_models_cache_clears_on_update():
    _teardown_summary_model_cache()
    mod.update_latest_models_cache([{"id": "a", "name": "A"}])
    assert "a" in mod._LATEST_MODELS_CACHE
    mod.update_latest_models_cache([{"id": "b", "name": "B"}])
    assert "a" not in mod._LATEST_MODELS_CACHE
    assert "b" in mod._LATEST_MODELS_CACHE
    _teardown_summary_model_cache()


def test_update_latest_models_cache_ignores_empty():
    _teardown_summary_model_cache()
    mod.update_latest_models_cache([{"id": "a", "name": "A"}])
    assert "a" in mod._LATEST_MODELS_CACHE
    mod.update_latest_models_cache([])
    assert "a" in mod._LATEST_MODELS_CACHE  # empty input does not wipe
    _teardown_summary_model_cache()


def test_get_summary_model_options_excludes_wrappers_for_non_default_function_id(monkeypatch):
    """When loaded as function_<custom_id>, wrapper models for that id must be excluded."""
    _teardown_summary_model_cache()
    models = {
        "gpt-4o": {"id": "gpt-4o", "name": "GPT-4o"},
        "custom_pipe.gpt-4o": {
            "id": "custom_pipe.gpt-4o",
            "name": "Custom Pipe GPT-4o",
        },
        "auto_compact.claude": {
            "id": "auto_compact.claude",
            "name": "Auto Compact Claude",
        },
    }
    mod.update_latest_models_cache(list(models.values()))
    monkeypatch.setattr(mod, "_iter_cache_models_from_state", lambda state: list(models.values()))
    monkeypatch.setitem(sys.modules, "open_webui.main", types.SimpleNamespace(app=types.SimpleNamespace(state=object())))

    monkeypatch.setattr(mod.Pipe.Valves, "__module__", "function_custom_pipe")
    options = mod.Pipe.Valves.get_summary_model_options()
    values = [opt["value"] for opt in options]
    assert "gpt-4o" in values
    assert "custom_pipe.gpt-4o" not in values
    # default-id wrappers are NOT this instance's own, so they remain visible
    assert "auto_compact.claude" in values
    _teardown_summary_model_cache()


def test_refresh_latest_models_cache_from_app_state_replaces_stale_populated_cache(monkeypatch):
    """Schema-time refresh should repair a stale non-empty pipes() snapshot."""
    _teardown_summary_model_cache()
    mod.update_latest_models_cache([{"id": "base-only", "name": "Base Only"}])

    fake_models = [
        {"id": "custom-preset", "name": "Custom Preset"},
        {"id": "base-only", "name": "Base Override"},
        {"id": "base-only", "name": "Base Provider"},
    ]

    def _fake_iter(state, **kwargs):
        return fake_models

    monkeypatch.setattr(mod, "_iter_cache_models_from_state", _fake_iter)
    monkeypatch.setitem(sys.modules, "open_webui.main", types.SimpleNamespace(app=types.SimpleNamespace(state=object())))

    mod.refresh_latest_models_cache_from_app_state()
    assert set(mod._LATEST_MODELS_CACHE.keys()) == {"custom-preset", "base-only"}
    assert mod._LATEST_MODELS_CACHE["base-only"]["name"] == "Base Override"
    _teardown_summary_model_cache()


def test_refresh_latest_models_cache_from_app_state_keeps_existing_on_empty_state(monkeypatch):
    _teardown_summary_model_cache()
    mod.update_latest_models_cache([{"id": "existing", "name": "Existing"}])

    monkeypatch.setattr(mod, "_iter_cache_models_from_state", lambda state: [])
    monkeypatch.setitem(sys.modules, "open_webui.main", types.SimpleNamespace(app=types.SimpleNamespace(state=object())))

    mod.refresh_latest_models_cache_from_app_state()
    assert set(mod._LATEST_MODELS_CACHE.keys()) == {"existing"}
    _teardown_summary_model_cache()


def test_refresh_latest_models_cache_from_app_state_silent_on_failure(monkeypatch):
    _teardown_summary_model_cache()

    def _raise_import_error(*args, **kwargs):
        raise ImportError("no app")

    monkeypatch.setattr(mod, "_iter_cache_models_from_state", _raise_import_error)
    monkeypatch.setitem(sys.modules, "open_webui.main", types.SimpleNamespace(app=types.SimpleNamespace(state=object())))

    # Should not raise
    mod.refresh_latest_models_cache_from_app_state()
    assert len(mod._LATEST_MODELS_CACHE) == 0
    _teardown_summary_model_cache()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_chunk",
    [
        b'data: {"error": {"code": "provider_error", "message": "boom"}}\n\n',
        b'data: {"type": "response.failed", "response": {"error": {"code": "server_error", "message": "boom"}}}\n\n',
    ],
)
async def test_streaming_completion_observer_skips_error_terminated_completion(error_chunk):
    observed = []
    request = SimpleNamespace(state=SimpleNamespace())
    chunks = [
        b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
        b'data: {"usage": {"input_tokens": 50}}\n\n',
        error_chunk,
    ]
    response = StreamingResponse(iter(chunks), media_type="text/event-stream")

    async def on_complete(completion):
        observed.append(completion)

    wrapped = mod._attach_streaming_completion_observer(
        response,
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
        on_complete=on_complete,
    )
    emitted = [chunk async for chunk in wrapped.body_iterator]

    assert emitted == chunks
    assert observed == []
    assert mod.get_request_scoped_usage(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
    ) is None
