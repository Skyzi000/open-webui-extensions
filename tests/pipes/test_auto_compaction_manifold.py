from __future__ import annotations

from types import SimpleNamespace
import asyncio
import copy
import inspect
import json
import math
import sys
import types

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

from functions.pipe import auto_compaction_pipe as mod


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
        now=None,
    ):
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


def install_fake_open_webui_user_model(monkeypatch):
    class FakeUserModel:
        def __init__(self, **data):
            self.__dict__.update(data)

    users_module = types.ModuleType("open_webui.models.users")
    users_module.UserModel = FakeUserModel
    monkeypatch.setitem(sys.modules, "open_webui.models.users", users_module)
    return FakeUserModel


def _file(file_id, *, file_type="file"):
    return {"id": file_id, "type": file_type, "name": f"{file_id}.txt"}


class FakeModelMeta(BaseModel):
    model_config = ConfigDict(extra="allow")


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
    }

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id):
            return SimpleNamespace(id=function_id, user_id="function-owner")

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            return records.get(model_id)

        @staticmethod
        async def get_all_models():
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
async def test_sync_wrapper_model_records_does_not_hide_existing_target_when_wrapper_insert_fails(
    monkeypatch,
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
async def test_sync_wrapper_model_records_does_not_create_target_override_when_wrapper_insert_fails(
    monkeypatch,
):
    records, calls = install_fake_open_webui_model_modules(monkeypatch)
    Models = sys.modules["open_webui.models.models"].Models
    original_insert_new_model = Models.insert_new_model

    async def fail_wrapper_insert(model_form, user_id):
        if model_form.id == mod.build_wrapper_model_id("auto_compact", "provider-target"):
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
async def test_sync_wrapper_model_records_restores_target_when_wrapper_update_fails(monkeypatch):
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
async def test_sync_wrapper_model_records_deletes_pipe_created_target_override_when_stale(monkeypatch):
    records, calls = install_fake_open_webui_model_modules(monkeypatch)
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="auto_compact",
        target_models=[{"id": "provider-target", "name": "Provider Target"}],
        valves=valves,
    )

    target_override = records["provider-target"]
    assert target_override.meta.hidden is True
    assert target_override.meta.auto_compaction_target_hidden_by == {
        "pipe_function_id": "auto_compact",
        "had_hidden": False,
        "previous_hidden": False,
        "created_model_record": True,
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
    assert calls["deleted"] == ["provider-target"]
    assert "provider-target" not in records
    assert records[wrapper_id].is_active is False


@pytest.mark.asyncio
async def test_sync_wrapper_model_records_deletes_pipe_created_target_override_when_hide_valve_disabled(
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
    assert calls["deleted"] == ["provider-target"]
    assert "provider-target" not in records
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
async def test_sync_wrapper_model_records_keeps_stale_wrapper_active_when_target_restore_fails(monkeypatch):
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
    original_update_model_by_id = Models.update_model_by_id

    async def fail_target_restore_update(model_id, model_form):
        if model_id == "stale-target":
            raise RuntimeError("restore failed")
        return await original_update_model_by_id(model_id, model_form)

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
async def test_sync_wrapper_model_records_tracks_multiple_target_hide_owners(monkeypatch):
    records = {
        "target": FakeModelForm(
            id="target",
            base_model_id="provider-target",
            name="Target",
            params=FakeModelParams(),
            meta=FakeModelMeta(),
            access_grants=[],
            is_active=True,
        )
    }
    records, _ = install_fake_open_webui_model_modules(monkeypatch, records)
    valves = mod.Pipe.Valves()
    valves.hide_wrapped_target_models = True

    await mod.sync_wrapper_model_records(
        pipe_function_id="compact_a",
        target_models=[{"id": "target", "name": "Target"}],
        valves=valves,
    )
    await mod.sync_wrapper_model_records(
        pipe_function_id="compact_b",
        target_models=[{"id": "target", "name": "Target"}],
        valves=valves,
    )

    target_meta = records["target"].meta.model_dump(exclude_unset=True)
    assert target_meta["hidden"] is True
    assert target_meta["auto_compaction_target_hidden_by"] == {
        "pipe_function_ids": ["compact_a", "compact_b"],
        "had_hidden": False,
        "previous_hidden": False,
    }

    await mod.sync_wrapper_model_records(
        pipe_function_id="compact_a",
        target_models=[],
        valves=mod.Pipe.Valves(),
    )

    target_meta = records["target"].meta.model_dump(exclude_unset=True)
    assert target_meta["hidden"] is True
    assert target_meta["auto_compaction_target_hidden_by"] == {
        "pipe_function_id": "compact_b",
        "had_hidden": False,
        "previous_hidden": False,
    }
    assert records[mod.build_wrapper_model_id("compact_a", "target")].is_active is False
    assert records[mod.build_wrapper_model_id("compact_b", "target")].is_active is True

    await mod.sync_wrapper_model_records(
        pipe_function_id="compact_b",
        target_models=[],
        valves=mod.Pipe.Valves(),
    )

    target_meta = records["target"].meta.model_dump(exclude_unset=True)
    assert "hidden" not in target_meta
    assert "auto_compaction_target_hidden_by" not in target_meta
    assert records[mod.build_wrapper_model_id("compact_b", "target")].is_active is False


@pytest.mark.asyncio
async def test_hide_target_model_record_merges_owners_from_current_record_when_given_stale_snapshot(monkeypatch):
    records = {
        "target": FakeModelForm(
            id="target",
            base_model_id="provider-target",
            name="Target",
            params=FakeModelParams(),
            meta=FakeModelMeta(),
            access_grants=[],
            is_active=True,
        )
    }
    install_fake_open_webui_model_modules(monkeypatch, records)
    Models = sys.modules["open_webui.models.models"].Models

    stale_snapshot_a = FakeModelForm(
        id="target",
        base_model_id="provider-target",
        name="Target",
        params=FakeModelParams(),
        meta=FakeModelMeta(),
        access_grants=[],
        is_active=True,
    )
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
        pipe_function_id="compact_a",
        target_model={"id": "target", "name": "Target"},
        target_model_info=stale_snapshot_a,
        owner_user_id="function-owner",
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
        "pipe_function_ids": ["compact_a", "compact_b"],
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
async def test_pipes_does_not_direct_fetch_provider_models_when_state_caches_remain_empty(monkeypatch):
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
async def test_pipes_stops_waiting_when_sibling_provider_cache_refresh_completes_empty(monkeypatch):
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


def test_checkpoint_schema_init_adds_summary_token_count_to_existing_tables(monkeypatch):
    class NestedTransaction:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSyncConn:
        def __init__(self):
            self.statements = []

        def begin_nested(self):
            return NestedTransaction()

        def execute(self, statement):
            self.statements.append(str(statement))

    class FakeInspector:
        def get_columns(self, table_name, schema=None):
            return [
                {"name": column.name}
                for column in mod.CHECKPOINT_TABLE.columns
                if column.name != "summary_token_count"
            ]

    sync_conn = FakeSyncConn()
    monkeypatch.setattr(mod, "sqlalchemy_inspect", lambda conn: FakeInspector())

    mod._initialize_checkpoint_schema(sync_conn)

    assert any(
        "ALTER TABLE" in statement and "summary_token_count" in statement
        for statement in sync_conn.statements
    )


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
    )

    assert count == 53
    assert body_calls == [
        {
            "messages": [*delta_messages, *tail_messages],
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
        valves=mod.Pipe.Valves(model_name_template="{target_name} (AutoCompact)"),
    )

    assert form["name"] == "GPT (AutoCompact)"


@pytest.mark.parametrize("template", ["{target_name.foo}", "{target_id[bad]}"])
def test_wrapper_model_name_template_falls_back_for_unsupported_format_syntax(template):
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "gpt-4.1", "name": "GPT"},
        valves=mod.Pipe.Valves(model_name_template=template),
    )

    assert form["name"] == "GPT (AutoCompact)"


def test_wrapper_model_name_template_rejects_format_width_specs():
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "gpt-4.1", "name": "GPT"},
        valves=mod.Pipe.Valves(model_name_template="{target_name:>100000}"),
    )

    assert len(form["name"]) < 1000
    assert form["name"] == "GPT (AutoCompact)"


@pytest.mark.parametrize("target_name", ["Model {target_id}", "Model {v2}"])
def test_wrapper_model_name_template_does_not_reinterpret_target_name_braces(target_name):
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "gpt-4.1", "name": target_name},
        valves=mod.Pipe.Valves(model_name_template="Wrapped: {target_name}"),
    )

    assert form["name"] == f"Wrapped: {target_name}"


def test_wrapper_model_name_template_does_not_reinterpret_target_id_braces():
    form = mod.build_wrapper_model_form(
        pipe_function_id="auto_compact",
        function_owner_user_id="owner-1",
        target_model={"id": "provider/{target_name}", "name": "GPT"},
        valves=mod.Pipe.Valves(model_name_template="Wrapped: {target_id}"),
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
async def test_target_access_rejects_custom_model_when_base_model_is_unavailable(
    monkeypatch,
    pipe_request,
    pipe_user,
):
    install_fake_open_webui_user_model(monkeypatch)
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
            assert model_id == "workspace-preset"
            return SimpleNamespace(id="workspace-preset", base_model_id="stale-openai")

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
async def test_forward_target_preserves_core_bypass_system_prompt_state(monkeypatch, pipe_request, pipe_user):
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
        "bypass_system_prompt": True,
        "state_bypass_system_prompt": True,
    }
    assert pipe_request.state.bypass_system_prompt is True
    assert not hasattr(pipe_request.state, "bypass_filter")


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


def test_valve_defaults_are_conservative_for_v1_continuation():
    valves = mod.Pipe.Valves()

    assert valves.trigger_total_tokens == mod.DEFAULT_TRIGGER_TOTAL_TOKENS
    assert valves.force_include_usage is True
    assert valves.compact_task_prompts_from_task_body is False
    assert valves.summary_tool_policy == "fallback_on_tool_call"
    assert valves.historical_message_excerpt_bytes == mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_BYTES
    assert valves.historical_message_excerpt_count == mod.DEFAULT_HISTORICAL_MESSAGE_EXCERPT_COUNT
    assert not hasattr(valves, "keep_tail_messages")
    assert not hasattr(valves, "preserve_latest_tool_rounds")


def test_trigger_total_tokens_overrides_default_is_empty_string():
    valves = mod.Pipe.Valves()

    assert valves.trigger_total_tokens_overrides_json == ""


def test_soft_trigger_ratio_default_resolves_against_hard_trigger():
    valves = mod.Pipe.Valves()

    assert valves.soft_trigger_ratio == mod.DEFAULT_SOFT_TRIGGER_RATIO
    assert (
        mod.resolve_soft_trigger_total_tokens(
            valves,
            {"id": "target", "name": "Target"},
            valves.trigger_total_tokens,
        )
        == int(mod.DEFAULT_TRIGGER_TOTAL_TOKENS * mod.DEFAULT_SOFT_TRIGGER_RATIO)
    )


def test_trigger_total_tokens_overrides_empty_string_is_valid():
    valves = mod.Pipe.Valves(trigger_total_tokens_overrides_json="")

    assert valves.trigger_total_tokens_overrides_json == ""


def test_trigger_total_tokens_overrides_accepts_valid_ordered_json():
    payload = json.dumps(
        {
            "schema_version": 1,
            "overrides": [
                {"model_patterns": ["claude-fable-5[1m]"], "trigger_total_tokens": 950000},
                {"model_patterns": ["claude-*", "Claude *"], "trigger_total_tokens": 160000},
            ],
        }
    )

    valves = mod.Pipe.Valves(trigger_total_tokens_overrides_json=payload)

    assert valves.trigger_total_tokens_overrides_json == payload


def test_trigger_total_tokens_overrides_rejects_non_json_string():
    with pytest.raises(ValidationError):
        mod.Pipe.Valves(trigger_total_tokens_overrides_json="not json")


def test_trigger_total_tokens_overrides_rejects_non_list_overrides():
    payload = json.dumps({"overrides": {"model_patterns": ["claude-*"], "trigger_total_tokens": 1}})

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(trigger_total_tokens_overrides_json=payload)


def test_trigger_total_tokens_overrides_rejects_empty_model_patterns():
    payload = json.dumps({"overrides": [{"model_patterns": [], "trigger_total_tokens": 160000}]})

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(trigger_total_tokens_overrides_json=payload)


def test_trigger_total_tokens_overrides_rejects_zero_trigger_total_tokens():
    payload = json.dumps({"overrides": [{"model_patterns": ["claude-*"], "trigger_total_tokens": 0}]})

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(trigger_total_tokens_overrides_json=payload)


def test_trigger_total_tokens_overrides_rejects_unknown_root_key():
    payload = json.dumps(
        {
            "overrides": [{"model_patterns": ["claude-*"], "trigger_total_tokens": 100}],
            "soft_trigger_total_tokens": 50,
        }
    )

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(trigger_total_tokens_overrides_json=payload)


def test_trigger_total_tokens_overrides_rejects_unknown_override_key():
    payload = json.dumps(
        {
            "overrides": [
                {"model_patterns": ["*"], "trigger_total_tokens": 100, "soft_trigger_total_tokens": 50}
            ]
        }
    )

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(trigger_total_tokens_overrides_json=payload)


def test_trigger_total_tokens_overrides_accepts_soft_trigger_ratio_without_hard_threshold():
    payload = json.dumps({"overrides": [{"model_patterns": ["claude-*"], "soft_trigger_ratio": 0.5}]})

    valves = mod.Pipe.Valves(trigger_total_tokens_overrides_json=payload)

    assert (
        mod.resolve_trigger_total_tokens(valves, {"id": "claude-sonnet", "name": "Sonnet"})
        == valves.trigger_total_tokens
    )
    assert (
        mod.resolve_soft_trigger_total_tokens(
            valves,
            {"id": "claude-sonnet", "name": "Sonnet"},
            100000,
        )
        == 50000
    )


def test_resolve_trigger_total_tokens_matches_target_model_id_and_name():
    valves = mod.Pipe.Valves(
        trigger_total_tokens=100000,
        trigger_total_tokens_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-*"], "trigger_total_tokens": 160000}]}
        ),
    )

    by_id = mod.resolve_trigger_total_tokens(valves, {"id": "claude-sonnet", "name": "Anthropic"})
    by_name = mod.resolve_trigger_total_tokens(valves, {"id": "anthropic-1", "name": "claude-opus"})

    assert by_id == 160000
    assert by_name == 160000


def test_resolve_trigger_total_tokens_first_match_wins():
    valves = mod.Pipe.Valves(
        trigger_total_tokens=100000,
        trigger_total_tokens_overrides_json=json.dumps(
            {
                "overrides": [
                    {"model_patterns": ["claude-opus"], "trigger_total_tokens": 950000},
                    {"model_patterns": ["claude-*"], "trigger_total_tokens": 160000},
                ]
            }
        ),
    )

    resolved = mod.resolve_trigger_total_tokens(valves, {"id": "claude-opus", "name": "Opus"})

    assert resolved == 950000


def test_resolve_trigger_total_tokens_matches_literal_bracket_id_without_escaping():
    # "[" and "]" are literal (not glob classes), so a bracketed id matches as-is and a
    # different id does not accidentally match via character-class expansion.
    valves = mod.Pipe.Valves(
        trigger_total_tokens=100000,
        trigger_total_tokens_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-opus-4-8[1m]"], "trigger_total_tokens": 950000}]}
        ),
    )

    assert mod.resolve_trigger_total_tokens(valves, {"id": "claude-opus-4-8[1m]", "name": "Opus 1M"}) == 950000
    assert mod.resolve_trigger_total_tokens(valves, {"id": "claude-opus-4-81", "name": "Opus"}) == 100000


def test_resolve_trigger_total_tokens_bulk_matches_bracket_suffix_with_wildcard():
    # Motivating bulk case: one override for every "[1m]" model id via "*[1m]".
    valves = mod.Pipe.Valves(
        trigger_total_tokens=100000,
        trigger_total_tokens_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["*[1m]"], "trigger_total_tokens": 950000}]}
        ),
    )

    assert mod.resolve_trigger_total_tokens(valves, {"id": "claude-opus-4-8[1m]", "name": "Opus"}) == 950000
    assert mod.resolve_trigger_total_tokens(valves, {"id": "claude-fable-5[1m]", "name": "Fable"}) == 950000
    assert mod.resolve_trigger_total_tokens(valves, {"id": "claude-opus-4-8", "name": "Opus"}) == 100000


def test_resolve_soft_trigger_total_tokens_uses_effective_hard_threshold():
    valves = mod.Pipe.Valves(soft_trigger_ratio=0.75)

    assert mod.resolve_soft_trigger_total_tokens(valves, {"id": "target", "name": "Target"}, 1000) == 750
    assert mod.resolve_soft_trigger_total_tokens(valves, {"id": "target", "name": "Target"}, 1001) == 750


def test_resolve_soft_trigger_total_tokens_uses_model_ratio_override():
    valves = mod.Pipe.Valves(
        soft_trigger_ratio=0.8,
        trigger_total_tokens_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-*"], "soft_trigger_ratio": 0.5}]}
        ),
    )

    assert mod.resolve_soft_trigger_total_tokens(valves, {"id": "claude-sonnet", "name": "Sonnet"}, 1000) == 500
    assert mod.resolve_soft_trigger_total_tokens(valves, {"id": "other", "name": "Other"}, 1000) == 800


def test_resolve_soft_trigger_total_tokens_zero_ratio_disables_prefetch():
    global_disabled = mod.Pipe.Valves(soft_trigger_ratio=0)
    override_disabled = mod.Pipe.Valves(
        soft_trigger_ratio=0.8,
        trigger_total_tokens_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-*"], "soft_trigger_ratio": 0}]}
        ),
    )

    assert mod.resolve_soft_trigger_total_tokens(global_disabled, {"id": "target", "name": "Target"}, 1000) is None
    assert (
        mod.resolve_soft_trigger_total_tokens(
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


def test_trigger_total_tokens_overrides_rejects_non_finite_soft_trigger_ratio():
    payload = json.dumps({"overrides": [{"model_patterns": ["claude-*"], "soft_trigger_ratio": math.nan}]})

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(trigger_total_tokens_overrides_json=payload)


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


def test_resolve_trigger_total_tokens_falls_back_to_global_default_on_no_match():
    valves = mod.Pipe.Valves(
        trigger_total_tokens=100000,
        trigger_total_tokens_overrides_json=json.dumps(
            {"overrides": [{"model_patterns": ["claude-*"], "trigger_total_tokens": 160000}]}
        ),
    )

    resolved = mod.resolve_trigger_total_tokens(valves, {"id": "gpt-4.1", "name": "GPT"})

    assert resolved == 100000


def test_resolve_trigger_total_tokens_falls_back_when_overrides_empty():
    valves = mod.Pipe.Valves(trigger_total_tokens=123456)

    resolved = mod.resolve_trigger_total_tokens(valves, {"id": "claude-sonnet", "name": "Anthropic"})

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


def test_request_scoped_usage_precedes_persisted_and_message_usage():
    request = SimpleNamespace(state=SimpleNamespace())
    key = mod.usage_state_key("chat-1", "message-1", "auto_compact.target")
    setattr(request.state, key, {"total_tokens": 999, "input_tokens": 900, "output_tokens": 99})

    usage = mod.choose_usage_signal(
        request=request,
        chat_id="chat-1",
        message_id="message-1",
        wrapper_model_id="auto_compact.target",
        persisted_usage={"total_tokens": 100},
        messages=[{"role": "assistant", "usage": {"total_tokens": 200}}],
    )

    assert usage["total_tokens"] == 999


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
async def test_summary_generation_overrides_request_state_metadata(monkeypatch, pipe_request, pipe_user):
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

    async def resolve_core_chat_model_route(request, model_id):
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
    pipe_request.app.state.MODELS = {
        "summary": {
            "id": "summary",
            "name": "Summary",
            "params": {"response_format": model_response_format},
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
            "response_format": {"type": "json_object"},
        },
    )

    assert result == "summary"
    assert "response_format" not in captured["form_body"]
    assert captured["model_params"]["response_format"] is model_response_format


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

    env_module = types.ModuleType("open_webui.env")
    env_module.ENABLE_CUSTOM_MODEL_FALLBACK = True
    models_module = types.ModuleType("open_webui.models.models")
    models_module.Models = FakeModels
    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.env", env_module)
    monkeypatch.setitem(sys.modules, "open_webui.models.models", models_module)
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

    async def generate_chat_completion(request, form_data, user, bypass_filter=False, bypass_system_prompt=False):
        captured.append(copy.deepcopy(form_data))
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


@pytest.mark.asyncio
async def test_lookup_persisted_usage_reads_current_branch(monkeypatch):
    class FakeChats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id):
            assert chat_id == "chat-1"
            return {
                "root": {"id": "root", "role": "user", "content": "root"},
                "assistant-old": {
                    "id": "assistant-old",
                    "parentId": "root",
                    "role": "assistant",
                    "content": "old",
                    "usage": {"total_tokens": 10},
                },
                "assistant-current": {
                    "id": "assistant-current",
                    "parentId": "assistant-old",
                    "role": "assistant",
                    "content": "current",
                    "info": {"usage": {"prompt_tokens": 100, "completion_tokens": 23}},
                },
                "sibling": {
                    "id": "sibling",
                    "parentId": "root",
                    "role": "assistant",
                    "content": "other branch",
                    "usage": {"total_tokens": 999},
                },
            }

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = FakeChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    usage = await mod.lookup_persisted_usage("chat-1", "assistant-current")

    assert usage["total_tokens"] == 123


@pytest.mark.asyncio
async def test_lookup_persisted_usage_uses_legacy_map_when_message_id_missing(monkeypatch):
    class FakeChats:
        @staticmethod
        async def get_messages_map_by_chat_id(chat_id):
            return {
                "root": {"id": "root", "role": "user", "content": "root"},
                "assistant-1": {
                    "id": "assistant-1",
                    "parentId": "root",
                    "role": "assistant",
                    "content": "answer",
                    "usage": {"input_tokens": 9, "output_tokens": 4},
                },
            }

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = FakeChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    usage = await mod.lookup_persisted_usage("chat-1", None)

    assert usage["total_tokens"] == 13


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


@pytest.mark.asyncio
async def test_pipe_forwards_below_threshold_to_decoded_target_with_metadata(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}

    async def validate_target_access(**kwargs):
        captured["validated_target"] = kwargs["target_model_id"]

    async def model_dict_from_request(request):
        return {"target.model": {"id": "target.model", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 10, "output_tokens": 0}

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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
async def test_pipe_injects_file_context_for_persisted_chat(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    install_fake_open_webui_user_model(monkeypatch)
    captured = {"target_file_calls": [], "source_events": [], "forward_attempts": 0}
    rows = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 10, "output_tokens": 0}

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        captured["estimate_body"] = copy.deepcopy(kwargs["body"])
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
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
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_call_target_completion", call_target_completion)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_call_target_completion", call_target_completion)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_call_target_completion", call_target_completion)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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
    captured = {"handler_calls": 0, "forward_body": None}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 10, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    async def chat_completion_files_handler(request, rag_body, extra_params, user):
        captured["handler_calls"] += 1
        return rag_body, {"sources": []}

    middleware_module = types.ModuleType("open_webui.utils.middleware")
    middleware_module.chat_completion_files_handler = chat_completion_files_handler
    monkeypatch.setitem(sys.modules, "open_webui.utils.middleware", middleware_module)
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
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
    pipe.valves.trigger_total_tokens = 100000
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

    mod._set_request_file_context_injection_active(pipe_request.state, True)
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

    assert captured["handler_calls"] == 0
    # Guard flag must be left untouched (caller owns it in this scenario).
    assert mod._request_file_context_injection_active(pipe_request.state) is True
    # Pruning of absorbed prefix files still applied even when skipping RAG.
    retained = result["metadata"]["files"]
    assert [file["id"] for file in retained] == ["current-file"]


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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
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


@pytest.mark.asyncio
async def test_pipe_rejects_custom_model_fallback_default_without_user_access(
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 10, "output_tokens": 0}

    async def body_reusable_checkpoint_match(**kwargs):
        captured["checkpoint_pipe_function_id"] = kwargs["pipe_function_id"]
        return None

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod.Pipe, "__module__", "function_compact_alias")

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100

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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10}

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": False,
        "previous_response_id": "resp-old",
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
    assert "previous_response_id" not in calls[1]
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
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
async def test_pipe_compacts_above_threshold_and_removes_previous_response_id(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 400, "output_tokens": 100}

    async def get_or_create_checkpoint_summary(**kwargs):
        captured["source_messages"] = kwargs["source_messages"]
        return "summary text"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "previous_response_id": "resp-old",
        "messages": [
            {"role": "system", "content": "system"},
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

    forwarded = captured["forward_body"]
    assert "previous_response_id" not in forwarded
    assert forwarded["messages"][0] == {"role": "system", "content": "system"}
    assert forwarded["messages"][1]["role"] == "user"
    assert "summary text" in forwarded["messages"][1]["content"]
    assert forwarded["messages"][-1] == {"role": "user", "content": "active"}
    assert captured["source_messages"] == [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    assert [event["data"]["action"] for event in events if event["type"] == "status"] == [
        "auto_compaction_compacting",
        "auto_compaction_compacted",
    ]
    embed_events = [event for event in events if event["type"] == "embeds"]
    assert len(embed_events) == 1
    assert embed_events[0]["data"]["replace"] is False
    assert len(embed_events[0]["data"]["embeds"]) == 1
    embed_html = embed_events[0]["data"]["embeds"][0]
    assert "summary text" in embed_html


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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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
async def test_pipe_applies_parent_checkpoint_below_hard_usage_without_foreground_when_estimate_is_safe(
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 50, "input_tokens": 50, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        estimate_calls.append(kwargs)
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens, raising=False)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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
    assert [event["data"]["action"] for event in events if event["type"] == "status"] == []


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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        estimate_calls.append(kwargs)
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens, raising=False)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 50, "input_tokens": 50, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        estimate_calls.append(kwargs)
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ParentCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens, raising=False)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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
    trigger_total_tokens,
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {
            "total_tokens": persisted_total_tokens,
            "input_tokens": persisted_total_tokens,
            "output_tokens": 0,
        }

    async def noop_initialize(**kwargs):
        return None

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        estimate_calls.append(kwargs)
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: existing_store)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens, raising=False)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = trigger_total_tokens
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
        trigger_total_tokens=100,
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
        trigger_total_tokens=1000,
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
        trigger_total_tokens=1000,
        soft_trigger_ratio=0.1,
    )

    assert case["result"] == {"ok": True}
    assert case["summary_inputs"] == []
    forwarded = case["captured"]["forward_body"]
    assert "existing parent summary" in forwarded["messages"][0]["content"]
    assert len(case["prefetch_calls"]) == 1
    assert case["prefetch_calls"][0]["trigger_estimated_tokens"] == 500
    assert "trigger_total_tokens" not in case["prefetch_calls"][0]


@pytest.mark.asyncio
async def test_pipe_rechecks_ready_checkpoint_before_soft_prefetch(
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 500

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        return 40

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
        return kwargs["body"]

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        raise AssertionError("soft prefetch must be skipped after late checkpoint revalidation")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 1000
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
    assert forwarded["messages"][1:] == [*delta_messages, {"role": "user", "content": "active"}]
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def body_reusable_checkpoint_match(**kwargs):
        calls["lookup"] += 1
        if calls["lookup"] == 1:
            return None
        raise RuntimeError("checkpoint db flaked during soft recheck")

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 500

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        calls["prefetch"] += 1
        raise AssertionError("soft prefetch must be skipped when its late recheck fails")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 1000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def body_reusable_checkpoint_match(**kwargs):
        calls["lookup"] += 1
        if calls["lookup"] <= 2:
            return None
        raise RuntimeError("checkpoint db flaked before prefetch launch")

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 500

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        calls["prefetch"] += 1
        raise AssertionError("soft prefetch must be skipped when launch recheck fails")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 1000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 500

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        return 40

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 1000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        checkpoint = (kwargs["match"].checkpoint or {})
        if checkpoint.get("id") == exact_checkpoint["id"]:
            return 40
        return 500

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: store)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 1000
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
        trigger_total_tokens=100,
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": total_tokens, "input_tokens": total_tokens, "output_tokens": 0}

    async def body_reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return total_tokens

    async def estimate_body_tokens_async(*args, **kwargs):
        return total_tokens

    async def get_or_create_checkpoint_summary(**kwargs):
        return "summary text"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
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
    pipe.valves.trigger_total_tokens = 100000
    pipe.valves.trigger_total_tokens_overrides_json = json.dumps(
        {"overrides": [{"model_patterns": ["target"], "trigger_total_tokens": 100}]}
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
    pipe.valves.trigger_total_tokens = 100
    pipe.valves.trigger_total_tokens_overrides_json = json.dumps(
        {"overrides": [{"model_patterns": ["target"], "trigger_total_tokens": 1000000}]}
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
async def test_pipe_override_non_match_uses_global_trigger_total_tokens(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}
    _install_threshold_decision_stubs(monkeypatch, captured, total_tokens=500)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    # Override targets a different model, so the global default (100) is used and 500 tokens compacts.
    pipe.valves.trigger_total_tokens_overrides_json = json.dumps(
        {"overrides": [{"model_patterns": ["other-*"], "trigger_total_tokens": 1000000}]}
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
    pipe.valves.trigger_total_tokens = 100
    assert pipe.valves.trigger_total_tokens_overrides_json == ""

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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def forward_target(**kwargs):
        captured["forwarded"] = True
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    # Bypass save-time validation (validate_assignment is off) to simulate a tampered stored value.
    pipe.valves.trigger_total_tokens_overrides_json = "not json"

    result = await pipe.pipe(
        _compactable_body(),
        __request__=pipe_request,
        __user__=pipe_user,
        __metadata__=pipe_metadata,
        __event_emitter__=None,
    )

    assert result["error"]["code"] == "invalid_trigger_total_tokens_overrides"
    assert "forwarded" not in captured


def test_task_template_selection_matches_core_whitespace_rules(pipe_request):
    pipe_request.app.state.config = SimpleNamespace(
        TITLE_GENERATION_PROMPT_TEMPLATE="   ",
        QUERY_GENERATION_PROMPT_TEMPLATE="   ",
        AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE="   ",
    )

    title_template = mod._task_template_from_request(
        pipe_request,
        mod.TASK_PROMPT_SPECS[mod.TASKS.TITLE_GENERATION.value],
    )
    query_template = mod._task_template_from_request(
        pipe_request,
        mod.TASK_PROMPT_SPECS[mod.TASKS.QUERY_GENERATION.value],
    )
    autocomplete_template = mod._task_template_from_request(
        pipe_request,
        mod.TASK_PROMPT_SPECS[mod.TASKS.AUTOCOMPLETE_GENERATION.value],
    )

    assert title_template == "   "
    assert query_template != "   "
    assert "{{MESSAGES" in query_template
    assert autocomplete_template != "   "
    assert "{{PROMPT}}" in autocomplete_template


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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe_request.app.state.config = SimpleNamespace(TAGS_GENERATION_PROMPT_TEMPLATE="Task:\n{{MESSAGES}}")
    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def body_reusable_checkpoint_match(**kwargs):
        captured["checkpoint_lookup_body"] = copy.deepcopy(kwargs["body"])
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        captured["estimate_body"] = copy.deepcopy(kwargs["body"])
        return 10

    async def inject_target_file_context(**kwargs):
        raise AssertionError("target file_context=false must not inject file context")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_estimate_next_input_tokens_from_usage_anchor", estimate_next_input_tokens_from_usage_anchor, raising=False)
    monkeypatch.setattr(mod, "_inject_target_file_context", inject_target_file_context)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingCheckpointStore())
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe_request.app.state.config = SimpleNamespace(TAGS_GENERATION_PROMPT_TEMPLATE="Task:\n{{MESSAGES}}")
    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_estimate_task_checkpoint_applied_body_tokens", estimate_task_checkpoint_applied_body_tokens, raising=False)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens)
    monkeypatch.setattr(mod, "_compact_task_body", compact_task_body)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe_request.app.state.config = SimpleNamespace(TAGS_GENERATION_PROMPT_TEMPLATE="Task:\n{{MESSAGES}}")
    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def body_reusable_checkpoint_match(**kwargs):
        captured.setdefault("checkpoint_lookup_body", copy.deepcopy(kwargs["body"]))
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        captured["estimate_body"] = copy.deepcopy(kwargs["body"])
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", body_reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe_request.app.state.config = SimpleNamespace(TAGS_GENERATION_PROMPT_TEMPLATE="Task:\n{{MESSAGES}}")
    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 200
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
    # forwards unchanged. No prefetch, no compaction, streaming on_complete unset.
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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
    # R7 (streaming): completed-turn soft prefetch must not be scheduled.
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 250000

    async def forward_target(**kwargs):
        forward_calls.append(copy.deepcopy(kwargs["body"]))
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 200
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 75, "input_tokens": 70, "output_tokens": 5}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 10

    async def generate_summary_text(**kwargs):
        raise AssertionError("must not compact when DB is down (no checkpoint created)")

    def start_soft_prefetch(**kwargs):
        raise AssertionError(
            "must not start soft prefetch (pre-forward or completed-turn) when DB is down"
        )

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return None

    async def estimate_body_tokens_async(body, **kwargs):
        return None

    async def forward_target(**kwargs):
        forward_calls.append(copy.deepcopy(kwargs["body"]))
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def initialize_fails(**kwargs):
        raise RuntimeError("checkpoint db down")

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        # Below the hard limit so the request is forwarded unchanged.
        return 10

    async def generate_summary_text(**kwargs):
        raise AssertionError("must not attempt compaction when checkpoint lookup was unavailable")

    def start_soft_prefetch(**kwargs):
        raise AssertionError("must not prefetch when checkpoint lookup is unavailable")

    async def forward_target(**kwargs):
        forward_attempts.append(copy.deepcopy(kwargs["body"]))
        raise mod.RetryableContextOverflow("target context window exceeded before output")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", initialize_fails)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingToolCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ExistingToolCheckpointStore([checkpoint]))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: RecordingCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    existing_store = ClaimCheckpointStore([checkpoint])

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        return 250000

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined active summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: existing_store)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens, raising=False)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 250000, "input_tokens": 250000, "output_tokens": 0}

    async def noop_initialize(**kwargs):
        return None

    existing_store = ClaimCheckpointStore([checkpoint])

    async def estimate_checkpoint_applied_body_tokens(**kwargs):
        return 250000

    async def generate_summary_text(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined active summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: existing_store)
    monkeypatch.setattr(mod, "_estimate_checkpoint_applied_body_tokens", estimate_checkpoint_applied_body_tokens, raising=False)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def get_or_create_checkpoint_summary(**kwargs):
        return "retry summary"

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        if len(calls) == 1:
            raise mod.RetryableContextOverflow("context")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    body = {
        "model": wrapper_id,
        "stream": True,
        "previous_response_id": "resp-old",
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
    assert "previous_response_id" not in calls[1]
    assert "retry summary" in calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_pipe_stops_context_retries_at_fixed_budget(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def get_or_create_checkpoint_summary(**kwargs):
        return "retry summary"

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        raise mod.RetryableContextOverflow("context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        raise RuntimeError("rate limit")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
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
        "previous_response_id": "resp-old",
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
    assert "previous_response_id" not in calls[1]
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def get_or_create_checkpoint_summary(**kwargs):
        return summaries.pop(0)

    async def forward_target(**kwargs):
        calls.append(kwargs["body"])
        if len(calls) == 1:
            raise mod.RetryableContextOverflow("context")
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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
    trigger_total_tokens=1000,
    forward_raises_once=False,
):
    captured = {"forward_bodies": []}
    estimate_calls = []
    events = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return usage

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        if (
            pipe is not None
            and pipe.valves.token_status_compare_estimate
            and (estimate_tokens is not None or estimate_sequence is not None)
        ):
            return None
        return mod._usage_total(usage)

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_lookup_pending_checkpoint_for_source_prefix",
        lookup_pending_checkpoint_for_source_prefix,
    )
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = pipe or mod.Pipe()
    pipe.valves.trigger_total_tokens = trigger_total_tokens
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
        trigger_total_tokens=1000,
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
        trigger_total_tokens=1000,
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
async def test_status_compare_estimate_mode(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    pipe = mod.Pipe()
    pipe.valves.token_status_compare_estimate = True

    result, events, _captured, estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        pipe=pipe,
        usage={"total_tokens": 1250, "input_tokens": 1250, "output_tokens": 0},
        estimate_tokens=1280,
        trigger_total_tokens=1000,
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
async def test_status_compare_estimate_and_before_after_combined(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    pipe = mod.Pipe()
    pipe.valves.token_status_compare_estimate = True
    pipe.valves.token_status_detail = "before_after"

    result, events, _captured, estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        pipe=pipe,
        usage={"total_tokens": 1250, "input_tokens": 1250, "output_tokens": 0},
        estimate_sequence=[1280, 500],
        trigger_total_tokens=1000,
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
        mod._format_token_suffix(ctx, after=None, compare_estimate=False, summary=50)
        == "(≈850 / 1,000 tokens · 85% · summary ≈50 tokens)"
    )


def test_format_token_suffix_summary_mode_with_compare_estimate():
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
        mod._format_token_suffix(ctx, after=50, compare_estimate=True, summary=50)
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
        trigger_total_tokens=1000,
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
        trigger_total_tokens=1000,
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
        trigger_total_tokens=1000,
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
            effective_trigger_total_tokens=1000,
            effective_soft_trigger_total_tokens=800,
            trigger_total_tokens=850,
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
            effective_trigger_total_tokens=1000,
            effective_soft_trigger_total_tokens=800,
            trigger_total_tokens=850,
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
            effective_trigger_total_tokens=1000,
            effective_soft_trigger_total_tokens=800,
            trigger_total_tokens=850,
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
    pipe.valves.trigger_total_tokens = 1000

    captured = {"forward_bodies": []}
    events = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 1500, "input_tokens": 1500, "output_tokens": 0}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 1500

    async def compact_body(**kwargs):
        return kwargs["body"], False, 0

    async def forward_target(**kwargs):
        captured["forward_bodies"].append(copy.deepcopy(kwargs["body"]))
        return {"ok": True}

    async def event_emitter(event):
        events.append(event)

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_lookup_pending_checkpoint_for_source_prefix",
        lookup_pending_checkpoint_for_source_prefix,
    )
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_compact_body", compact_body)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

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
    pipe.valves.trigger_total_tokens = 1000
    captured = {"forward_called": False}
    events = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 1500, "input_tokens": 1500, "output_tokens": 0}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def lookup_pending_checkpoint_for_source_prefix(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 1500

    async def compact_body(**kwargs):
        raise mod.SummaryFileContextUnavailable("attached file context unavailable")

    async def forward_target(**kwargs):
        captured["forward_called"] = True
        return {"ok": True}

    async def event_emitter(event):
        events.append(event)

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_lookup_pending_checkpoint_for_source_prefix", lookup_pending_checkpoint_for_source_prefix)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_compact_body", compact_body)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

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
        trigger_total_tokens=1000,
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
    pipe.valves.trigger_total_tokens = 1000

    result, events, captured, _estimate_calls = await _run_token_status_pipe(
        monkeypatch,
        pipe_request,
        pipe_user,
        pipe_metadata,
        pipe=pipe,
        usage={"total_tokens": 1500, "input_tokens": 1500, "output_tokens": 0},
        trigger_total_tokens=1000,
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
    async for _ in prepared.body_iterator:
        pass

    captured = {}
    summary_inputs = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        raise AssertionError("request-scoped usage should be used before persisted usage")

    async def get_or_create_compaction_summary(**kwargs):
        summary_inputs.append(kwargs["source_messages"])
        return "combined old tool summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    events = []

    async def event_emitter(event):
        events.append(event)

    body = {
        "model": wrapper_id,
        "stream": True,
        "previous_response_id": "resp-old",
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
    assert "previous_response_id" not in captured["forward_body"]
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 500, "input_tokens": 500, "output_tokens": 0}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def forward_target(**kwargs):
        return {"ok": True, "messages": kwargs["body"]["messages"]}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def forward_target(**kwargs):
        raise mod.RetryableContextOverflow("context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def generate_summary_text(**kwargs):
        raise mod.RetryableContextOverflow("summary context")

    async def forward_target(**kwargs):
        raise mod.RetryableContextOverflow("target context")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
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
async def test_pipe_does_not_compact_internal_summary_task(monkeypatch, pipe_request, pipe_user, pipe_metadata):
    captured = {}

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 999999, "input_tokens": 999999, "output_tokens": 0}

    async def get_or_create_checkpoint_summary(**kwargs):
        raise AssertionError("summary task must not recursively compact")

    async def forward_target(**kwargs):
        captured["forward_body"] = kwargs["body"]
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_get_or_create_checkpoint_summary", get_or_create_checkpoint_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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
async def test_soft_prefetch_skips_summary_generation_when_chain_prefix_is_pending(
    monkeypatch,
    pipe_request,
    pipe_user,
    pipe_metadata,
):
    pending_source = [{"role": "user", "content": "active request"}]
    later_source = [
        *pending_source,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
    ]
    rows = [
        mod.build_checkpoint_row(
            namespace=mod.CHECKPOINT_NAMESPACE,
            user_id=pipe_user["id"],
            chat_id=pipe_metadata["chat_id"],
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
    ]
    events = []

    async def event_emitter(event):
        events.append(event)

    async def noop_initialize(**kwargs):
        return None

    async def get_or_create_compaction_summary(**kwargs):
        raise AssertionError("a newer soft prefetch must not start while a chain prefix summary is pending")

    monkeypatch.setattr(mod, "ensure_checkpoint_table_initialized", noop_initialize)
    monkeypatch.setattr(mod, "CheckpointStore", lambda: ClaimCheckpointStore(rows))
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)

    assert (
        await mod._prefetch_compaction_checkpoint(
            request=pipe_request,
            user=pipe_user,
            user_id=pipe_user["id"],
            chat_id=pipe_metadata["chat_id"],
            metadata=pipe_metadata,
            body={"model": "target", "messages": later_source},
            pipe_function_id="auto_compact",
            summary_model_id="target",
            source_messages=later_source,
            summary_tool_policy="fallback_on_tool_call",
            historical_message_excerpt_bytes=1024,
            historical_message_excerpt_count=3,
            event_emitter=event_emitter,
        )
        is False
    )
    assert events == []


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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 150, "input_tokens": 150, "output_tokens": 0}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_total_tokens = 1000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 150, "input_tokens": 150, "output_tokens": 0}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_total_tokens = 1000
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
async def test_pipe_uses_full_body_estimate_for_request_usage_without_tool_suffix(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    body_estimates = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
        raise AssertionError("request-scoped usage must only anchor a safe tool-result suffix")

    async def estimate_body_tokens_async(body, **kwargs):
        body_estimates.append(copy.deepcopy(body))
        return 120

    async def get_or_create_compaction_summary(**kwargs):
        return "full estimate from request usage without tool suffix"

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 700, "input_tokens": 550, "output_tokens": 150}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_extra_tokens_async", estimate_body_extra_tokens_async, raising=False)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 1000
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
    assert estimated_messages == [{"role": "user", "content": "active"}]
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 90, "input_tokens": 70, "output_tokens": 20}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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
    assert estimated_messages == [{"role": "user", "content": "active"}]
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 70, "input_tokens": 50, "output_tokens": 20}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_total_tokens = 1000
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
async def test_pipe_compacts_from_usage_anchor_plus_tool_result_suffix_without_full_body_estimate(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    captured = {}
    estimated_batches = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
        estimated_batches.append(copy.deepcopy(messages))
        return 35

    async def estimate_body_tokens_async(*args, **kwargs):
        raise AssertionError("tool-loop request usage anchor should avoid full-body token estimation for safe trailing tool results")

    async def get_or_create_compaction_summary(**kwargs):
        return "tool loop hard summary"

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
    wrapper_id = mod.build_wrapper_model_id("auto_compact", "target")
    mod.store_request_scoped_usage(
        request=pipe_request,
        chat_id=pipe_metadata["chat_id"],
        message_id=pipe_metadata["message_id"],
        wrapper_model_id=wrapper_id,
        usage={"total_tokens": 70, "input_tokens": 60, "output_tokens": 10},
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
    assert estimated_batches == [[tool_message]]
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_messages_tokens_async(messages, **kwargs):
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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens_async", estimate_messages_tokens_async)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return None

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_total_tokens = 1000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 200, "input_tokens": 200, "output_tokens": 0}

    async def reusable_checkpoint_match(**kwargs):
        return None

    def estimate_messages_tokens(messages, **kwargs):
        return 50

    async def get_or_create_compaction_summary(**kwargs):
        return "hard summary"

    async def estimate_body_tokens_async(*args, **kwargs):
        raise AssertionError("hard usage should short-circuit token estimation")

    async def forward_target(**kwargs):
        captured["forward_body"] = copy.deepcopy(kwargs["body"])
        return {"ok": True}

    def start_soft_prefetch(**kwargs):
        raise AssertionError("hard compaction must not also start soft prefetch")

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens", estimate_messages_tokens)
    monkeypatch.setattr(mod, "estimate_body_tokens_async", estimate_body_tokens_async, raising=False)
    monkeypatch.setattr(mod, "_get_or_create_compaction_summary", get_or_create_compaction_summary)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.2
    pipe.valves.trigger_total_tokens = 100
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 50, "input_tokens": 50, "output_tokens": 0}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens", estimate_messages_tokens)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.5
    pipe.valves.trigger_total_tokens = 1000
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

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 50, "input_tokens": 50, "output_tokens": 0}

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(mod, "estimate_messages_tokens", estimate_messages_tokens)
    monkeypatch.setattr(mod, "_generate_summary_text", generate_summary_text)
    monkeypatch.setattr(mod, "_forward_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0
    pipe.valves.trigger_total_tokens = 1000
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

    def soft_prefetch_source_messages(body):
        calls["source_selection"] += 1
        return copy.deepcopy(selected_source)

    async def get_or_create_compaction_summary(
        *,
        source_messages,
        **kwargs,
    ):
        launched["source_messages"] = copy.deepcopy(source_messages)
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
    source_messages = mod._soft_prefetch_source_messages({"model": "target", "messages": messages})
    assert source_messages == messages[:3]

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
        summary_tool_policy="fallback_on_tool_call",
        historical_message_excerpt_bytes=1024,
        historical_message_excerpt_count=3,
    )

    assert prefetched is True
    assert captured["source_messages"] == messages[:3]


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
async def test_pipe_launches_completed_turn_soft_prefetch_for_non_tool_response(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 150, "input_tokens": 150, "output_tokens": 0}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 150

    async def forward_target(**kwargs):
        return {
            "usage": {"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ]
        }

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_total_tokens = 1000
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

    assert len(calls) == 2
    assert all(call.get("event_emitter") is event_emitter for call in calls)
    completed_messages = calls[-1]["body"]["messages"]
    assert completed_messages == [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": ""},
    ]


@pytest.mark.asyncio
async def test_task_completed_turn_prefetch_passes_rebuilt_prompt_for_checkpoint_estimates(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 10

    async def render_task_prompt_from_messages(**kwargs):
        return "rebuilt completed task prompt"

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
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_render_task_prompt_from_messages", render_task_prompt_from_messages)
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_total_tokens = 1000
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

    assert result["choices"][0]["message"]["content"] == "task answer"
    assert len(calls) == 1
    completed_messages = [
        *task_history,
        {"role": "assistant", "content": "task answer"},
        {"role": "user", "content": ""},
    ]
    assert calls[0]["body"]["messages"] == completed_messages
    task_estimate_body = calls[0]["task_estimate_body"]
    assert task_estimate_body["messages"] == [{"role": "user", "content": "rebuilt completed task prompt"}]
    assert task_estimate_body["metadata"]["task_body"]["messages"] == completed_messages


@pytest.mark.asyncio
async def test_pipe_skips_completed_turn_soft_prefetch_for_tool_call_response(
    monkeypatch, pipe_request, pipe_user, pipe_metadata
):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 150, "input_tokens": 150, "output_tokens": 0}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 150

    async def forward_target(**kwargs):
        return {
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

    def start_soft_prefetch(**kwargs):
        assert "target_model_id" not in kwargs
        assert mod._soft_prefetch_source_messages(kwargs["body"])
        calls.append(copy.deepcopy(kwargs))
        return True

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_total_tokens = 1000
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
):
    calls = []

    async def validate_target_access(**kwargs):
        return None

    async def model_dict_from_request(request):
        return {"target": {"id": "target", "name": "Target"}}

    async def lookup_persisted_usage(chat_id, message_id):
        return {"total_tokens": 10, "input_tokens": 10, "output_tokens": 0}

    async def reusable_checkpoint_match(**kwargs):
        return None

    async def estimate_next_input_tokens_from_usage_anchor(**kwargs):
        return 10

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

    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "lookup_persisted_usage", lookup_persisted_usage)
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint_match)
    monkeypatch.setattr(
        mod,
        "_estimate_next_input_tokens_from_usage_anchor",
        estimate_next_input_tokens_from_usage_anchor,
        raising=False,
    )
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)
    monkeypatch.setattr(mod, "_start_soft_compaction_prefetch", start_soft_prefetch, raising=False)

    pipe = mod.Pipe()
    pipe.valves.soft_trigger_ratio = 0.1
    pipe.valves.trigger_total_tokens = 1000
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

    return calls


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

    assert len(calls) == 1
    assert calls[0]["trigger_total_tokens"] == 500
    assert calls[0]["body"]["messages"][-2:] == [
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": ""},
    ]


@pytest.mark.asyncio
async def test_streaming_completion_observer_reports_text_completion():
    observed = []
    chunks = [
        b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n',
        b'data: {"usage": {"total_tokens": 150}}\n\n',
    ]
    response = StreamingResponse(iter(chunks), media_type="text/event-stream")

    wrapped = mod._attach_streaming_completion_observer(response, observed.append)
    emitted = [chunk async for chunk in wrapped.body_iterator]

    assert emitted == chunks
    assert observed == [
        {
            "assistant_message": {"role": "assistant", "content": "hello"},
            "usage": {"total_tokens": 150, "input_tokens": 0, "output_tokens": 0},
        }
    ]


@pytest.mark.asyncio
async def test_streaming_completion_observer_skips_tool_call_completion():
    observed = []
    chunks = [
        b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1"}]}}]}\n\n',
        b'data: {"usage": {"total_tokens": 150}}\n\n',
    ]
    response = StreamingResponse(iter(chunks), media_type="text/event-stream")

    wrapped = mod._attach_streaming_completion_observer(response, observed.append)
    emitted = [chunk async for chunk in wrapped.body_iterator]

    assert emitted == chunks
    assert observed == []


# ---------------------------------------------------------------------------
# summary_model dropdown options
# ---------------------------------------------------------------------------


def _teardown_summary_model_cache():
    mod._LATEST_MODELS_CACHE.clear()


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

    def _fake_iter(state):
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
    chunks = [
        b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n',
        error_chunk,
    ]
    response = StreamingResponse(iter(chunks), media_type="text/event-stream")

    wrapped = mod._attach_streaming_completion_observer(response, observed.append)
    emitted = [chunk async for chunk in wrapped.body_iterator]

    assert emitted == chunks
    assert observed == []
