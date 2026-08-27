from __future__ import annotations

# noqa: SIZE_OK - one isolated child process owns the complete pinned-Core runtime contract.

import hashlib
import importlib.util
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import pkgutil
import re
import sys
from types import SimpleNamespace
from typing import Final

import anyio
from fastapi import HTTPException
import pytest


PINNED_SHA: Final = "1a97751e376e00a1897bc3679215ae1c7bd8fd42"
PINNED_LOCK_SHA256: Final = (
    "f8484dfea258a70f1401b18fbab8eb1e7b783b8315ca962e32b263645740b07d"
)
SCENARIOS: Final = {
    "archive",
    "conversion_boundary",
    "function_calling_gate",
    "registry_dispatch",
    "outer_context",
    "two_reader",
    "multimodel",
}


class PinnedMessageIds(dict[str, str]):
    """Expose v0.9.6's dict contract and Task 8's list fixture contract."""

    def __init__(self, values: dict[str, str]):
        super().__init__(values)
        self.items_calls = 0

    def __iter__(self):
        return iter(
            {"model_id": model_id, "message_id": message_id}
            for model_id, message_id in dict.items(self)
        )

    def __getitem__(self, key):
        if key == 0:
            model_id, message_id = next(iter(self.items()))
            return {"model_id": model_id, "message_id": message_id}
        return super().__getitem__(key)

    def items(self):
        self.items_calls += 1
        return super().items()


def _runtime_versions() -> dict[str, str]:
    distributions = ("fastapi", "pydantic", "starlette", "tiktoken")
    return {
        distribution: metadata.version(distribution) for distribution in distributions
    }


async def _run_route(harness, **kwargs):
    import open_webui.main as core_main

    if not hasattr(core_main, "Config"):
        core_main.Config = SimpleNamespace(get=None)
    if not hasattr(core_main, "publish_event"):
        core_main.publish_event = None
    request = kwargs.get("request") or harness._Task7Request()
    request.app.state.config = core_main.app.state.config
    kwargs["request"] = request
    inject_hidden_args = kwargs.pop("inject_hidden_args", False)
    raw_target_metadata = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        original_forward = harness.mod._forward_streaming_target

        async def capture_target_metadata(**forward_kwargs):
            raw_target_metadata.append(forward_kwargs["body"]["metadata"])
            return await original_forward(**forward_kwargs)

        monkeypatch.setattr(
            harness.mod, "_forward_streaming_target", capture_target_metadata
        )
        if inject_hidden_args:
            original_dumps = json.dumps

            def dumps_with_hidden_args(value, *args, **dump_kwargs):
                if isinstance(value, dict) and set(value) == {"command"}:
                    value = {
                        **value,
                        "__request__": "must-not-pass",
                        "__metadata__": "must-not-pass",
                        "__messages__": "must-not-pass",
                    }
                return original_dumps(value, *args, **dump_kwargs)

            monkeypatch.setattr(json, "dumps", dumps_with_hidden_args)
        observed = await harness._task8_run_current_core_route(monkeypatch, **kwargs)
    observed["raw_target_metadata"] = raw_target_metadata
    return observed


def _registry(harness, label: str = "unrelated") -> dict:  # noqa: DICT_OK - Open WebUI's registry is dynamically shaped.
    return {
        label: {
            "spec": {"name": label, "parameters": {"type": "object", "properties": {}}},
            "callable": lambda: None,
        }
    }


async def _single_route_assertions(harness, observations: dict) -> dict[str, bool]:
    registry = _registry(harness)
    observed = await _run_route(
        harness,
        registry=registry,
        inject_hidden_args=True,
        text="task3 raw recursive tool output",
    )
    assert observed["active_readers"]
    reader_entry = observed["active_readers"][0]["entry"]
    reader_spec = reader_entry["spec"]
    store = getattr(observed["request"].state, harness.mod.REQUEST_STATE_REF_STORE_KEY)
    provider_tool_specs = [
        tool["function"]
        for call in observed["provider"]
        for tool in call.get("tools", [])
        if tool["function"]["name"] == harness.mod.REF_EXEC_TOOL_NAME
    ]
    assert len(observed["provider"]) == 3, {
        "provider_count": len(observed["provider"]),
        "provider_messages": [call["messages"] for call in observed["provider"]],
        "injected_count": len(observed["injected"]),
        "emitted": observed["emitted"],
        "result": observed["result"],
    }
    assert len(observed["injected"]) == 3
    assert all(
        call["__request__"] is observed["request"] for call in observed["injected"]
    )
    assert all(call["__tools__"] is registry for call in observed["injected"])
    assert observed["outer"] is not observed["injected"][0]["__metadata__"]
    assert observed["outer"]["tools"] is registry
    assert observed["injected"][0]["__metadata__"]["tools"] is registry
    assert observed["raw_target_metadata"]
    assert all(
        metadata["tools"] is registry for metadata in observed["raw_target_metadata"]
    ), [
        (id(metadata), id(metadata["tools"]), id(registry))
        for metadata in observed["raw_target_metadata"]
    ]
    assert set(reader_spec["parameters"]["properties"]) == {"command"}
    assert all(
        set(spec["parameters"]["properties"]) == {"command"}
        for spec in provider_tool_specs
    )
    assert list(inspect.signature(reader_entry["callable"]).parameters) == ["command"]
    assert harness.mod.REF_EXEC_TOOL_NAME not in registry
    assert (
        not store.bindings
        and not store.reservations
        and not store.registry_reservations
        and not store.registry_owners
    )
    emitted = json.dumps(observed["emitted"])
    assert "call-wc" in emitted and "call-head" in emitted
    assert emitted.count("function_call_output") >= 2
    assert registry["unrelated"]["callable"]() is None
    entries = []
    owned_reader = observed["active_readers"][0]["entry"]["callable"]
    for index, call in enumerate(observed["injected"]):
        entries.append(
            {
                "index": index,
                "messages": call["body"]["messages"],
                "body_has_metadata": "metadata" in call["body"],
                "tools_registry_id": id(call["__tools__"]),
                "metadata_registry_id": id(call["__metadata__"]["tools"]),
                "expected_registry_id": id(registry),
                "reader_callable_id": (
                    id(call["entry_reader_callable"])
                    if call["entry_reader_callable"] is not None
                    else None
                ),
                "owned_reader_callable_id": (id(owned_reader) if index > 0 else None),
            }
        )
    observations["single_route_entries"] = entries
    recursive_entries = entries[1:]
    raw_recursive_messages = all(
        any(
            message.get("role") == "tool"
            and "task3 raw recursive tool output" in str(message.get("content"))
            for message in entry["messages"]
        )
        and re.search(r"tool:[0-9a-f]{64}", json.dumps(entry["messages"])) is None
        and re.search(r"history:accp_[0-9a-f]{64}", json.dumps(entry["messages"]))
        is None
        and "<auto_compaction_context" not in json.dumps(entry["messages"])
        and "<auto_compact_ref_manifests" not in json.dumps(entry["messages"])
        and not entry["body_has_metadata"]
        for entry in recursive_entries
    )
    recursive_registry_identity = all(
        entry["tools_registry_id"]
        == entry["metadata_registry_id"]
        == entry["expected_registry_id"]
        for entry in recursive_entries
    )
    recursive_reader_identity = all(
        entry["reader_callable_id"] == entry["owned_reader_callable_id"]
        for entry in recursive_entries
    )
    assert len(recursive_entries) == 2
    assert raw_recursive_messages
    assert recursive_registry_identity
    assert recursive_reader_identity

    return {
        "same_request_reentry": len(recursive_entries) == 2,
        "outer_registry_dispatch": observed["outer"]["tools"] is registry,
        "raw_recursive_messages": raw_recursive_messages,
        "raw_target_registry_identity": all(
            metadata["tools"] is registry
            for metadata in observed["raw_target_metadata"]
        ),
        "reader_key_advanced_in_place": store.next_generation == 3,
        "recursive_reader_identity": recursive_reader_identity,
        "recursive_registry_identity": recursive_registry_identity,
        "schema_allowlists_command_only": all(
            set(spec["parameters"]["properties"]) == {"command"}
            for spec in provider_tool_specs
        ),
        "tool_output_conversion": emitted.count("function_call_output") >= 2,
    }


async def _exact_wc_assertions(harness) -> dict[str, bool]:
    text = "λ" * 40_000
    encoded = text.encode("utf-8")
    assert len(encoded) > 65_536
    assert len(encoded) == 80_000

    owner_registry = _registry(harness, "owner-exact-wc-tool")
    owner_observed = await _run_route(
        harness,
        registry=owner_registry,
        owner=True,
        role="user",
        text=text,
        ref_substitution_threshold_tokens=1_000,
        dispatch_exact_ref_command=True,
    )
    admin_registry = _registry(harness, "admin-exact-wc-tool")
    admin_observed = await _run_route(
        harness,
        registry=admin_registry,
        owner=False,
        role="admin",
        text=text,
        ref_substitution_threshold_tokens=1_000,
        dispatch_exact_ref_command=True,
    )
    assert owner_registry is not admin_registry

    for observed, registry in (
        (owner_observed, owner_registry),
        (admin_observed, admin_registry),
    ):
        assert len(observed["provider"]) == 2
        assert len(observed["injected"]) == 2
        assert len(observed["active_readers"]) == 1
        active_reader = observed["active_readers"][0]
        assert active_reader["registry"] is registry
        assert active_reader["catalog"]
        reader_parameters = active_reader["entry"]["spec"]["parameters"]
        assert set(reader_parameters["properties"]) == {"command"}
        provider_reader_specs = [
            tool["function"]
            for call in observed["provider"]
            for tool in call.get("tools", [])
            if tool["function"]["name"] == harness.mod.REF_EXEC_TOOL_NAME
        ]
        assert provider_reader_specs
        assert all(
            set(spec["parameters"]["properties"]) == {"command"}
            for spec in provider_reader_specs
        )
        projected_refs = set(
            re.findall(
                r"tool:[0-9a-f]{64}",
                json.dumps(observed["provider"][0]["messages"]),
            )
        )
        assert len(projected_refs) == 1
        projected_ref = next(iter(projected_refs))
        assert projected_ref in {
            entry.manifest.ref for entry in active_reader["catalog"]
        }
        second_provider_messages = observed["provider"][1]["messages"]
        reader_tool_calls = [
            tool_call
            for message in second_provider_messages
            for tool_call in message.get("tool_calls", [])
            if tool_call.get("function", {}).get("name")
            == harness.mod.REF_EXEC_TOOL_NAME
        ]
        assert len(reader_tool_calls) == 1
        reader_tool_call = reader_tool_calls[0]
        assert json.loads(reader_tool_call["function"]["arguments"]) == {
            "command": f"wc -c {projected_ref}"
        }
        reader_tool_messages = [
            message
            for message in second_provider_messages
            if message.get("role") == "tool"
            and message.get("tool_call_id") == reader_tool_call["id"]
        ]
        assert len(reader_tool_messages) == 1
        assert reader_tool_messages[0]["content"] == "80000"
        function_call_outputs = [
            item
            for event in observed["emitted"]
            if event.get("type") == "chat:completion"
            for item in event.get("data", {}).get("output", [])
            if item.get("type") == "function_call_output"
            and item.get("call_id") == reader_tool_call["id"]
        ]
        assert function_call_outputs
        assert all(
            item.get("output") == [{"type": "input_text", "text": "80000"}]
            for item in function_call_outputs
        )
        reader_output_surface = json.dumps(
            {
                "emitted": observed["emitted"],
                "second_provider_messages": second_provider_messages,
            }
        )
        assert (
            f'Error: Tool "{harness.mod.REF_EXEC_TOOL_NAME}" not found.'
            not in reader_output_surface
        )
        assert "Error: usage:" not in reader_output_surface
        assert "not available in this binding" not in reader_output_surface
        assert harness.mod.REF_EXEC_TOOL_NAME not in registry
        store = getattr(
            observed["request"].state, harness.mod.REQUEST_STATE_REF_STORE_KEY
        )
        assert not store.bindings
        assert not store.reservations
        assert not store.registry_reservations
        assert not store.registry_owners

    return {
        "owner_exact_wc_output": True,
        "admin_exact_wc_output": True,
        "admin_non_owner_reader_dispatch_after_core_admission": True,
    }


async def _core_admission_assertions(harness) -> dict[str, bool]:
    denied = False
    try:
        await _run_route(
            harness,
            registry=_registry(harness, "denied-tool"),
            owner=False,
            role="user",
            dispatch_reader=False,
        )
    except HTTPException as exc:
        denied = exc.status_code == 404
    assert denied
    return {"ordinary_non_owner_denied_before_pipe": True}


async def _detached_parity_assertions(harness) -> dict[str, bool]:
    for size in (8, 50 * 1024, 30 * 1024 * 1024, 65_537):
        enabled = await _run_route(harness, text="x" * size, dispatch_reader=False)
        disabled = await _run_route(
            harness,
            text="x" * size,
            dispatch_reader=False,
            valve_enabled=False,
        )
        assert enabled["provider"] == disabled["provider"]
        assert enabled["result"] == disabled["result"]
        assert "tools" not in enabled["outer"]
        assert enabled["injected"][0]["__tools__"] == {}
        assert not hasattr(
            enabled["request"].state, harness.mod.REQUEST_STATE_REF_STORE_KEY
        )
    hard_kwargs = {
        "text": "hard threshold payload " * 512,
        "dispatch_reader": False,
        "trigger_input_tokens": 1,
        "observe_hard_compaction": True,
    }
    hard_enabled = await _run_route(harness, **hard_kwargs)
    hard_disabled = await _run_route(harness, valve_enabled=False, **hard_kwargs)
    assert hard_enabled["summary_requests"] == hard_disabled["summary_requests"]
    assert hard_enabled["checkpoint_rows"] == hard_disabled["checkpoint_rows"]
    assert hard_enabled["provider"] == hard_disabled["provider"]
    assert hard_enabled["result"] == hard_disabled["result"]
    assert hard_enabled["checkpoint_rows"]
    return {"detached_parity_all_sizes": True, "detached_hard_compaction_parity": True}


async def _function_calling_gate_assertions(harness) -> dict[str, bool]:
    native_registry = _registry(harness, "native-tool")
    native = await _run_route(
        harness,
        registry=native_registry,
        params={"function_calling": "native"},
        dispatch_reader=False,
    )
    assert native["active_readers"]

    for params in ({}, {"function_calling": "legacy"}):
        registry = _registry(harness, "inactive-tool")
        enabled = await _run_route(
            harness,
            registry=registry,
            params=params,
            dispatch_reader=False,
        )
        disabled = await _run_route(
            harness,
            registry=registry,
            params=params,
            dispatch_reader=False,
            valve_enabled=False,
        )
        assert enabled["provider"] == disabled["provider"]
        assert enabled["result"] == disabled["result"]
        assert not enabled["active_readers"]
        assert not hasattr(
            enabled["request"].state, harness.mod.REQUEST_STATE_REF_STORE_KEY
        )
        assert registry.keys() == {"inactive-tool"}

    return {
        "legacy_default_inactive": True,
        "legacy_legacy_inactive": True,
        "legacy_native_active": True,
        "legacy_inactive_valve_off_parity": True,
        "legacy_inactive_no_ref_state": True,
    }


async def _multimodel_assertions(harness) -> dict[str, bool]:
    targets = ("target-one", "target-two")
    wrappers = tuple(
        harness.mod.build_wrapper_model_id("auto_compact", target) for target in targets
    )
    expected_ids = {
        wrapper: f"assistant-{index}" for index, wrapper in enumerate(wrappers, 1)
    }
    message_ids = PinnedMessageIds(expected_ids)
    registries = {
        message_id: _registry(harness, f"tool-{index}")
        for index, message_id in enumerate(message_ids.values(), 1)
    }
    observed = await _run_route(
        harness,
        message_ids=message_ids,
        registries_by_message=registries,
    )
    first, second = registries.values()
    assert first is not second
    captured_by_registry = {
        id(captured["registry"]): captured["entry"]
        for captured in observed["active_readers"]
    }
    assert id(first) in captured_by_registry and id(second) in captured_by_registry
    assert (
        captured_by_registry[id(first)]["callable"]
        is not captured_by_registry[id(second)]["callable"]
    )
    assert harness.mod.REF_EXEC_TOOL_NAME not in first
    assert harness.mod.REF_EXEC_TOOL_NAME not in second
    assert {id(metadata["tools"]) for metadata in observed["outers"]} == {
        id(first),
        id(second),
    }
    assert len(observed["provider"]) == 4
    assert json.dumps(observed["emitted"]).count("function_call_output") >= 2
    assert message_ids.items_calls >= 1
    assert dict(dict.items(message_ids)) == expected_ids
    assert tuple(message_ids.keys()) == tuple(expected_ids)
    assert tuple(message_ids.values()) == tuple(expected_ids.values())
    store = getattr(observed["request"].state, harness.mod.REQUEST_STATE_REF_STORE_KEY)
    assert (
        not store.bindings
        and not store.reservations
        and not store.registry_reservations
    )
    for call in observed["injected"]:
        message_id = call["__metadata__"]["message_id"]
        assert call["__tools__"] is registries[message_id]
        assert call["__metadata__"]["tools"] is registries[message_id]
    for target, message_id in zip(targets, expected_ids.values(), strict=True):
        model_calls = [call for call in observed["provider"] if call["model"] == target]
        assert len(model_calls) == 2
        reader_specs = [
            tool["function"]
            for call in model_calls
            for tool in call.get("tools", [])
            if tool["function"]["name"] == harness.mod.REF_EXEC_TOOL_NAME
        ]
        assert reader_specs and all(
            set(spec["parameters"]["properties"]) == {"command"}
            for spec in reader_specs
        )
        own_ref_match = re.search(
            r"tool:[0-9a-f]{64}", json.dumps(model_calls[1]["messages"])
        )
        assert own_ref_match is not None
        own_ref = own_ref_match.group(0)
        assert own_ref in json.dumps(model_calls[1]["messages"])
        assert any(
            message.get("role") == "tool" and own_ref in str(message.get("content"))
            for message in model_calls[1]["messages"]
        )

    isolated_request = harness._Task7Request()
    isolated_registries = {
        message_id: _registry(harness, f"cancel-tool-{index}")
        for index, message_id in enumerate(message_ids.values(), 1)
    }
    cancelled = await _run_route(
        harness,
        message_ids=message_ids,
        registries_by_message=isolated_registries,
        cancel_model_id=targets[0],
        request=isolated_request,
        text="first sibling payload\n" * 3_000,
    )
    first_nonce = getattr(
        cancelled["request"].state, harness.mod.REQUEST_STATE_REF_STORE_KEY
    ).next_generation
    retried = await _run_route(
        harness,
        message_ids=message_ids,
        registries_by_message=isolated_registries,
        cancel_model_id=targets[1],
        request=isolated_request,
        text="retried sibling payload\n" * 3_000,
    )
    isolated_store = getattr(
        retried["request"].state, harness.mod.REQUEST_STATE_REF_STORE_KEY
    )
    assert isolated_store.next_generation > first_nonce
    assert not isolated_store.bindings
    assert not isolated_store.reservations and not isolated_store.registry_reservations
    emitted_refs = tuple(
        dict.fromkeys(
            re.findall(
                r"tool:[0-9a-f]{64}",
                json.dumps(
                    [
                        cancelled["emitted"],
                        retried["emitted"],
                        cancelled["provider"],
                        retried["provider"],
                    ],
                    default=str,
                ),
            )
        )
    )
    assert len(emitted_refs) >= 2
    first_ref, second_ref = emitted_refs[:2]
    assert first_ref != second_ref
    captured_readers = [
        captured["entry"]["callable"]
        for route in (cancelled, retried)
        for captured in route["active_readers"]
    ]
    assert captured_readers
    invalidated_results = [
        await reader(f"cat {second_ref}") for reader in captured_readers
    ]
    assert all(result.startswith("Error:") for result in invalidated_results)
    assert not any(
        ref in result for result in invalidated_results for ref in emitted_refs
    )
    cancelled_output = json.dumps(cancelled["emitted"])
    retried_output = json.dumps(retried["emitted"])
    assert "function_call_output" in cancelled_output
    assert "function_call_output" in retried_output
    return {
        "private_registry_isolation": True,
        "binding_model_message_identity": True,
        "command_only_schema_per_sibling": True,
        "message_ids_dict_contract": True,
        "no_cross_resolution": True,
        "own_reader_dispatch_per_sibling": True,
        "registry_identity_on_recursive_entries": True,
        "retry_and_cancellation_isolation": True,
        "request_generation_nonce_monotonic": True,
    }


async def _conversion_boundary_assertions(
    harness, observations: dict
) -> dict[str, bool]:
    from open_webui.utils.middleware import process_messages_with_output

    raw = [
        {"id": "prior-user", "role": "user", "content": "prior turn"},
        {
            "id": "prior-assistant",
            "role": "assistant",
            "content": "stale UI content",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": [{"type": "input_text", "text": "tool result"}],
                },
            ],
        },
        {"id": "current-user", "role": "user", "content": "current turn"},
    ]

    expanded = process_messages_with_output(raw, reasoning_format=None)
    source = await harness.mod.build_canonical_history_source(
        raw,
        source_message_count=len(expanded) - 1,
    )
    records = tuple(source.iter_records())
    observations["conversion_boundary"] = {
        "raw_count": len(raw),
        "expanded_count": len(expanded),
        "source_line_count": source.line_count,
        "raw_record_limit": source.raw_record_limit,
        "records": records,
    }
    return {
        "raw_three_expand_to_four": len(raw) == 3 and len(expanded) == 4,
        "current_turn_excluded_from_history_ref": (
            source.raw_record_limit == 2
            and source.line_count == 3
            and all("current turn" not in record for record in records)
        ),
    }


async def _execute(scenario: str, harness) -> dict:
    assertions = {"import_isolated": True, "isolation_guard_rejects_current_core": True}
    observations = {}
    if scenario == "conversion_boundary":
        assertions.update(await _conversion_boundary_assertions(harness, observations))
    elif scenario == "registry_dispatch":
        assertions.update(await _single_route_assertions(harness, observations))
        assertions.update(await _exact_wc_assertions(harness))
        assertions.update(await _core_admission_assertions(harness))
    elif scenario == "two_reader":
        assertions.update(await _single_route_assertions(harness, observations))
        assertions.update(await _core_admission_assertions(harness))
    elif scenario == "outer_context":
        assertions.update(await _single_route_assertions(harness, observations))
        assertions.update(await _detached_parity_assertions(harness))
    elif scenario == "multimodel":
        assertions.update(await _multimodel_assertions(harness))
    elif scenario == "function_calling_gate":
        assertions.update(await _function_calling_gate_assertions(harness))
    return {"assertions": assertions, "observations": observations}


def main() -> int:
    scenario = sys.argv[1] if len(sys.argv) == 2 else ""
    assert scenario in SCENARIOS
    os.environ["FROM_INIT_PY"] = "False"
    archive_root = Path(os.environ["TASK9_ARCHIVE_ROOT"]).resolve()
    task8_bound = "task8_pinned_harness" in sys.modules
    auto_compact_bound = "functions.pipe.auto_compact" in sys.modules
    import open_webui

    open_webui_file = Path(open_webui.__file__).resolve()
    assert open_webui_file.is_relative_to(archive_root), (
        f"isolation_guard task8_bound={task8_bound} auto_compact_bound={auto_compact_bound}"
    )
    package_version = json.loads((archive_root / "package.json").read_text())["version"]
    lock_sha256 = hashlib.sha256((archive_root / "uv.lock").read_bytes()).hexdigest()
    assert package_version == "0.9.6"
    assert lock_sha256 == PINNED_LOCK_SHA256
    assert os.environ["TASK9_PINNED_SHA"] == PINNED_SHA
    assert not task8_bound and not auto_compact_bound
    original_get_data = pkgutil.get_data
    pkgutil.get_data = lambda package, resource: (
        b""
        if package == "open_webui" and resource == "CHANGELOG.md"
        else original_get_data(package, resource)
    )
    from open_webui import env as open_webui_env

    assert open_webui_env.VERSION == "0.9.6"

    harness_path = Path.cwd() / "tests" / "pipes" / "test_auto_compaction_manifold.py"
    harness_spec = importlib.util.spec_from_file_location(
        "task8_pinned_harness", harness_path
    )
    assert harness_spec is not None and harness_spec.loader is not None
    harness = importlib.util.module_from_spec(harness_spec)
    sys.modules[harness_spec.name] = harness
    harness_spec.loader.exec_module(harness)

    execution = anyio.run(_execute, scenario, harness)
    report = {
        "scenario": scenario,
        "pinned_sha": PINNED_SHA,
        "package_version": package_version,
        "env_version": open_webui_env.VERSION,
        "lock_sha256": lock_sha256,
        "open_webui_file": str(open_webui_file),
        "archive_root": str(archive_root),
        "parent_sha256": os.environ["TASK9_PARENT_SHA256"],
        "child_sha256": os.environ["TASK9_CHILD_SHA256"],
        "runtime_versions": _runtime_versions(),
        "assertions": execution["assertions"],
        "observations": execution["observations"],
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
