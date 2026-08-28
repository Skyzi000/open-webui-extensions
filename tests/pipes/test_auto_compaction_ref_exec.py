from __future__ import annotations

import asyncio
import ast
import builtins
import copy
import dataclasses
import gc
import hashlib
import importlib
import inspect
import json
import re
import sys
import textwrap
import threading
import tracemalloc
import types
from types import MappingProxyType, SimpleNamespace
from typing import Awaitable, Callable, Final, Protocol

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from functions.pipe import auto_compact as mod

# Core migrations import Chat while loading middleware. Load that real module
# before the request-local Chats fake can enter sys.modules.
importlib.import_module("open_webui.utils.middleware")


def _task1_surface() -> tuple[object, object, object, object]:
    classifier = getattr(mod, "classify_ref_text", None)
    projector = getattr(mod, "project_native_tool_texts", None)
    resolver = getattr(mod, "resolve_ref_mode_preflight", None)
    preflight_type = getattr(mod, "RefModePreflight", None)
    assert callable(classifier), "Task 1 ref classifier is not implemented"
    assert callable(projector), "Task 1 ref projector is not implemented"
    assert callable(resolver), "Task 1 ref-mode preflight resolver is not implemented"
    assert callable(preflight_type), "Task 1 preflight contract is not implemented"
    return classifier, projector, resolver, preflight_type


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


async def _run_pipe_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    messages: list[dict[str, object]],
    user_id: str | None = "user-1",
    function_calling_capability: bool | str | None = None,
    valve_enabled: bool = True,
    registry_has_tool: bool = True,
    registry_ref_collision: bool = False,
    estimated_tokens: int = 1,
    body_overrides: dict[str, object] | None = None,
    metadata_overrides: dict[str, object] | None = None,
    raw_message_map: dict[str, dict[str, object]] | None = None,
    configure_pipe: Callable[[mod.Pipe], None] | None = None,
    forward_response: dict[str, object] | None = None,
    reusable_checkpoint_matches: tuple[mod.ReusableCheckpointMatch | None, ...]
    | None = None,
    estimated_token_values: tuple[int, ...] | None = None,
    forwarded_capture: list[dict[str, object]] | None = None,
) -> tuple[
    object,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    SimpleNamespace,
]:
    registry: dict[str, object] = (
        {"existing": {"spec": {"name": "existing"}}} if registry_has_tool else {}
    )
    if registry_ref_collision:
        registry[mod.REF_EXEC_TOOL_NAME] = {"spec": {"name": "collision"}}
    metadata: dict[str, object] = {
        "chat_id": "chat-1",
        "message_id": "message-1",
        "session_id": "session-1",
        "params": {"function_calling": "native"},
        "tools": registry,
    }
    metadata.update(metadata_overrides or {})
    request = SimpleNamespace(
        state=SimpleNamespace(), app=SimpleNamespace(state=SimpleNamespace(MODELS={}))
    )
    request.state.raw_branch_loads = 0
    user = {"role": "user"}
    if user_id is not None:
        user["id"] = user_id
    forwarded: list[dict[str, object]] = (
        forwarded_capture if forwarded_capture is not None else []
    )
    checkpoint_bodies: list[dict[str, object]] = []
    checkpoint_match_index = 0
    estimate_index = 0

    class FakeChats:
        @staticmethod
        async def get_messages_map_by_chat_id(
            chat_id: str,
        ) -> dict[str, dict[str, object]] | None:
            assert chat_id == "chat-1"
            request.state.raw_branch_loads += 1
            return raw_message_map

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = FakeChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    async def no_refresh(_request: object) -> None:
        return None

    async def core_compaction_enabled() -> bool:
        return False

    async def validate_target_access(**_kwargs: object) -> None:
        return None

    async def model_dict_from_request(_request: object) -> dict[str, object]:
        info: dict[str, object] = {}
        if isinstance(function_calling_capability, bool):
            info = {
                "meta": {
                    "capabilities": {
                        "function_calling": function_calling_capability,
                    }
                }
            }
        elif function_calling_capability == "malformed":
            info = {"meta": {"capabilities": []}}
        return {
            "target": {
                "id": "target",
                "name": "Target",
                "info": info,
            }
        }

    async def resolve_route(
        *_args: object, **_kwargs: object
    ) -> mod.CoreChatModelRoute:
        return mod.CoreChatModelRoute(model_id="target")

    async def resolve_arena_route(
        **kwargs: object,
    ) -> tuple[mod.CoreChatModelRoute, None]:
        return kwargs["route"], None

    async def reusable_checkpoint(
        **kwargs: object,
    ) -> mod.ReusableCheckpointMatch | None:
        nonlocal checkpoint_match_index
        checkpoint_bodies.append(copy.deepcopy(kwargs["body"]))
        if reusable_checkpoint_matches is None:
            return None
        match = reusable_checkpoint_matches[
            min(checkpoint_match_index, len(reusable_checkpoint_matches) - 1)
        ]
        checkpoint_match_index += 1
        return match

    async def estimate_tokens(*_args: object, **_kwargs: object) -> int:
        nonlocal estimate_index
        if estimated_token_values is not None:
            value = estimated_token_values[
                min(estimate_index, len(estimated_token_values) - 1)
            ]
            estimate_index += 1
            return value
        return estimated_tokens

    async def forward_target(**kwargs: object) -> dict[str, object]:
        forwarded.append(copy.deepcopy(kwargs["body"]))
        return (
            copy.deepcopy(forward_response)
            if forward_response is not None
            else {"ok": True}
        )

    monkeypatch.setattr(mod, "_refresh_tiktoken_encoding_config", no_refresh)
    monkeypatch.setattr(
        mod, "_core_context_compaction_enabled", core_compaction_enabled
    )
    monkeypatch.setattr(mod, "_validate_target_access", validate_target_access)
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_route)
    monkeypatch.setattr(
        mod, "_resolve_arena_chat_model_route_with_access", resolve_arena_route
    )
    monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint)
    monkeypatch.setattr(mod, "_estimate_provider_input_tokens_async", estimate_tokens)
    monkeypatch.setattr(
        mod, "_target_model_supports_file_context", lambda *_args: False
    )
    monkeypatch.setattr(mod, "_forward_non_streaming_target", forward_target)

    pipe = mod.Pipe()
    pipe.valves.ref_exec_enabled = valve_enabled
    if configure_pipe is not None:
        configure_pipe(pipe)
    request_body: dict[str, object] = {
        "model": mod.build_wrapper_model_id("auto_compact", "target"),
        "stream": False,
        "messages": messages,
    }
    request_body.update(body_overrides or {})
    result = await pipe.pipe(
        request_body,
        __request__=request,
        __user__=user,
        __metadata__=metadata,
        __tools__=registry,
    )
    return result, forwarded, checkpoint_bodies, registry, request


@dataclasses.dataclass(frozen=True, slots=True)
class RefModeBoundaryCase:
    expected_reason: mod.RefModeReason | None
    generation: mod.CoreFunctionCallingGeneration = (
        mod.CoreFunctionCallingGeneration.NATIVE_DEFAULT
    )
    valve_enabled: bool = True
    metadata_overrides: dict[str, object] | None = None
    body_overrides: dict[str, object] | None = None
    registry_ref_collision: bool = False


def _linked_raw_message_map(
    records: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    messages: dict[str, dict[str, object]] = {}
    parent_id: str | None = None
    for index, record in enumerate(records):
        message_id = str(record.get("id") or f"raw-{index}")
        linked = {**record, "id": message_id, "parentId": parent_id}
        messages[message_id] = linked
        parent_id = message_id
    return messages


def _raw_native_round(
    *,
    assistant_id: str,
    call_id: str,
    name: str,
    output_parts: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "id": assistant_id,
        "role": "assistant",
        "content": "stale UI content",
        "output": [
            {
                "type": "function_call",
                "id": f"fc-{assistant_id}",
                "call_id": call_id,
                "name": name,
                "arguments": "{}",
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "id": f"fco-{assistant_id}",
                "call_id": call_id,
                "output": output_parts,
                "status": "completed",
            },
        ],
    }


def _expanded_core_messages(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    from open_webui.utils.middleware import process_messages_with_output

    return process_messages_with_output(copy.deepcopy(records))


_PROVIDER_VISIBLE_CORE_FIELDS: Final = frozenset(
    {"role", "content", "output", "files", "contextSummary"}
)
CoreFixtureValue = (
    str
    | int
    | float
    | bool
    | None
    | list["CoreFixtureValue"]
    | dict[str, "CoreFixtureValue"]
)


class ReaderFixture(Protocol):
    def __call__(self, command: CoreFixtureValue = "") -> Awaitable[str]: ...


def _provider_visible_core_messages(
    records: list[dict[str, CoreFixtureValue]],
    *,
    leading_systems: tuple[str, ...] = (),
) -> list[dict[str, CoreFixtureValue]]:
    loaded = [
        {
            key: value
            for key, value in record.items()
            if key in _PROVIDER_VISIBLE_CORE_FIELDS
        }
        for record in records
    ]
    for message in loaded:
        files = message.get("files", [])
        assert isinstance(files, list)
        image_files: list[dict[str, CoreFixtureValue]] = []
        for file in files:
            assert isinstance(file, dict)
            content_type = file.get("content_type")
            if file.get("type") == "image" or (
                isinstance(content_type, str) and content_type.startswith("image/")
            ):
                image_files.append(file)
        content = message.get("content")
        if message.get("role") == "user" and image_files and isinstance(content, str):
            message["content"] = [
                {"type": "text", "text": content},
                *(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                    for file in image_files
                    if (url := file.get("url"))
                ),
            ]
        message.pop("files", None)
        message.pop("contextSummary", None)
        message.pop("context_summary", None)
    return [
        *({"role": "system", "content": content} for content in leading_systems),
        *_expanded_core_messages(loaded),
    ]


def _committed_ref_binding(request: SimpleNamespace) -> mod.RefBindingState:
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    assert len(store.bindings) == 1
    return next(iter(store.bindings.values()))


@pytest.mark.parametrize(
    ("openai_marker", "ollama_marker", "expected"),
    (
        (lambda: None, lambda: None, "native_default"),
        (None, None, "native_opt_in"),
        (lambda: None, None, "unknown"),
        (None, lambda: None, "unknown"),
    ),
)
def test_core_function_calling_generation_uses_both_router_capability_markers(
    openai_marker,
    ollama_marker,
    expected,
) -> None:
    openai_router = SimpleNamespace(get_openai_connection=openai_marker)
    ollama_router = SimpleNamespace(get_ollama_runtime_config=ollama_marker)

    generation = mod.classify_core_function_calling_generation(
        openai_router,
        ollama_router,
    )

    assert generation.value == expected


def test_core_function_calling_generation_import_failure_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fail_router_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "open_webui.routers" and "ollama" in fromlist:
            raise ImportError("isolated router import failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_router_import)

    assert (
        mod._core_function_calling_generation()
        is mod.CoreFunctionCallingGeneration.UNKNOWN
    )


@pytest.mark.parametrize(
    ("generation", "function_calling", "expected"),
    (
        ("native_default", None, True),
        ("native_default", "default", True),
        ("native_default", "native", True),
        ("native_default", "legacy", False),
        ("native_opt_in", None, False),
        ("native_opt_in", "default", False),
        ("native_opt_in", "native", True),
        ("native_opt_in", "legacy", False),
        ("unknown", None, False),
        ("unknown", "native", False),
    ),
)
def test_function_calling_native_eligibility_matches_core_generation(
    generation,
    function_calling,
    expected,
) -> None:
    assert (
        mod.core_function_calling_is_native(
            mod.CoreFunctionCallingGeneration(generation),
            function_calling,
        )
        is expected
    )


@pytest.mark.asyncio
async def test_enabled_inactive_ref_mode_logs_only_reason_once_at_info(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_values = (
        "user-1",
        "private-chat-id",
        "target",
        "private-message-id",
        "private-tool-content",
        "private-prompt-content",
    )
    messages = [
        {"role": "user", "content": sensitive_values[-1]},
        {"role": "tool", "content": sensitive_values[-2]},
    ]
    caplog.set_level("DEBUG", logger=mod.__name__)
    monkeypatch.setattr(
        mod,
        "_core_function_calling_generation",
        lambda: mod.CoreFunctionCallingGeneration.UNKNOWN,
    )

    _, _, _, _, request = await _run_pipe_boundary(
        monkeypatch,
        messages=messages,
        metadata_overrides={
            "params": {},
            "chat_id": sensitive_values[1],
            "message_id": sensitive_values[3],
        },
    )

    inactive_records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("Auto Compact ref mode inactive:")
    ]
    assert len(inactive_records) == 1
    assert inactive_records[0].levelname == "INFO"
    assert (
        inactive_records[0].getMessage()
        == "Auto Compact ref mode inactive: non_native_context"
    )
    assert all(
        sensitive not in inactive_records[0].getMessage()
        for sensitive in sensitive_values
    )
    assert not hasattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)


@pytest.mark.parametrize(
    "case",
    (
        RefModeBoundaryCase(
            expected_reason=None,
            valve_enabled=False,
        ),
        RefModeBoundaryCase(
            expected_reason=mod.RefModeReason.NON_NATIVE_CONTEXT,
            generation=mod.CoreFunctionCallingGeneration.UNKNOWN,
        ),
        RefModeBoundaryCase(
            expected_reason=mod.RefModeReason.NON_DURABLE_CONTEXT,
            metadata_overrides={"message_id": None},
        ),
        RefModeBoundaryCase(
            expected_reason=mod.RefModeReason.PROVIDER_SCHEMA_UNSUPPORTED,
            body_overrides={"tools": {}},
        ),
        RefModeBoundaryCase(
            expected_reason=mod.RefModeReason.CORE_REGISTRY_UNAVAILABLE,
            metadata_overrides={"tools": {"detached": {}}},
        ),
        RefModeBoundaryCase(
            expected_reason=mod.RefModeReason.READER_COLLISION,
            registry_ref_collision=True,
        ),
    ),
    ids=(
        "valve-off-no-log",
        "non-native-context",
        "non-durable-context",
        "provider-schema-unsupported",
        "core-registry-unavailable",
        "reader-collision",
    ),
)
@pytest.mark.asyncio
async def test_ref_mode_boundary_reports_truthful_reason_after_required_checks(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: RefModeBoundaryCase,
) -> None:
    caplog.set_level("INFO", logger=mod.__name__)
    monkeypatch.setattr(
        mod,
        "_core_function_calling_generation",
        lambda: case.generation,
    )

    _, _, _, _, _ = await _run_pipe_boundary(
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
        valve_enabled=case.valve_enabled,
        metadata_overrides=case.metadata_overrides,
        body_overrides=case.body_overrides,
        registry_ref_collision=case.registry_ref_collision,
    )

    inactive_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Auto Compact ref mode inactive:")
    ]
    expected_messages = (
        []
        if case.expected_reason is None
        else [f"Auto Compact ref mode inactive: {case.expected_reason}"]
    )
    assert inactive_messages == expected_messages


def test_ref_mode_active_path_returns_active_without_owner_reauthorization() -> None:
    # Given
    tools = {}
    preflight = mod.RefModePreflight(
        valve_enabled=True,
        native_function_calling=True,
        durable_context=True,
        provider_schema_supported=True,
        metadata_tools=tools,
        injected_tools=tools,
        registry_available=True,
    )

    # When
    result = mod.resolve_ref_mode_preflight(preflight)

    # Then
    assert result == mod.EffectiveRefMode(active=True, reason=mod.RefModeReason.ACTIVE)


@pytest.mark.asyncio
async def test_ref_mode_valve_off_emits_no_ref_mode_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("DEBUG", logger=mod.__name__)

    monkeypatch.setattr(
        mod,
        "_core_function_calling_generation",
        lambda: mod.CoreFunctionCallingGeneration.NATIVE_DEFAULT,
    )
    await _run_pipe_boundary(
        monkeypatch,
        messages=[{"role": "user", "content": "private-prompt-content"}],
        valve_enabled=False,
    )

    assert not any(
        record.getMessage().startswith("Auto Compact ref mode inactive:")
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_ref_threshold_skips_tokenizer_above_65536_bytes() -> None:
    classifier, _, _, _ = _task1_surface()
    encoder = CountingEncoder(count=1)

    result = await classifier("x" * 65_537, threshold_tokens=10_000, encoder=encoder)

    assert result.eligible is True
    assert result.utf8_bytes == 65_537
    assert encoder.calls == []


@pytest.mark.asyncio
async def test_ref_threshold_exact_counts_65536_bytes_off_thread() -> None:
    classifier, _, _, _ = _task1_surface()
    encoder = CountingEncoder(count=9_999)
    event_loop_thread = threading.get_ident()

    result = await classifier("x" * 65_536, threshold_tokens=10_000, encoder=encoder)

    assert result.eligible is False
    assert result.token_count == 9_999
    assert encoder.calls[0][2] != event_loop_thread


@pytest.mark.asyncio
async def test_ref_threshold_externalizes_at_equal_token_limit() -> None:
    classifier, _, _, _ = _task1_surface()

    result = await classifier(
        "boundary",
        threshold_tokens=1_000,
        encoder=CountingEncoder(count=1_000),
    )

    assert result.eligible is True
    assert result.token_count == 1_000


@pytest.mark.asyncio
async def test_ref_threshold_encoder_failure_keeps_only_that_text_raw() -> None:
    classifier, _, _, _ = _task1_surface()

    bounded = await classifier(
        "x" * 65_536,
        threshold_tokens=1_000,
        encoder=CountingEncoder(fail=True),
    )
    giant = await classifier(
        "x" * 65_537,
        threshold_tokens=1_000,
        encoder=CountingEncoder(fail=True),
    )

    assert bounded.eligible is False
    assert bounded.encoder_failed is True
    assert giant.eligible is True
    assert giant.encoder_failed is False


@pytest.mark.asyncio
async def test_invalid_eligible_ref_classification_is_rejected_before_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_message = "Eligible ref text classification requires complete measurement"

    with pytest.raises(ValueError, match=error_message):
        mod.RefTextClassification(
            eligible=True,
            utf8_bytes=None,
            sha256=None,
            line_count=None,
            token_count=None,
            encoder_failed=False,
        )

    malformed = object.__new__(mod.RefTextClassification)
    object.__setattr__(malformed, "eligible", True)
    object.__setattr__(malformed, "utf8_bytes", None)
    object.__setattr__(malformed, "sha256", None)
    object.__setattr__(malformed, "line_count", None)
    object.__setattr__(malformed, "token_count", None)
    object.__setattr__(malformed, "encoder_failed", False)

    async def classify_malformed(*_args: object, **_kwargs: object) -> object:
        return malformed

    monkeypatch.setattr(mod, "classify_ref_text", classify_malformed)
    with pytest.raises(ValueError, match=error_message):
        await mod.project_native_tool_texts(
            [{"role": "tool", "tool_call_id": "call-1", "content": "valid"}],
            threshold_tokens=1_000,
        )


@pytest.mark.asyncio
async def test_valid_utf8_measurement_classification_and_projection_remain_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "valid-界\ntext"
    below_threshold = "small"
    encoded = text.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    encoder = CountingEncoder(count=1_000)
    messages = [
        {"role": "tool", "tool_call_id": "call-1", "content": text},
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": below_threshold,
        },
    ]

    measurement = mod._measure_ref_text(text)
    classification = await mod.classify_ref_text(
        text,
        threshold_tokens=1_000,
        encoder=encoder,
    )
    measured_texts = []
    measure_ref_text = mod._measure_ref_text

    def measure(candidate: str):
        measured_texts.append(candidate)
        return measure_ref_text(candidate)

    class EligibilityEncoder:
        def encode(self, candidate: str, **_kwargs: object) -> list[int]:
            return list(range(1_000 if candidate == text else 1))

    monkeypatch.setattr(mod, "_measure_ref_text", measure)
    plan = await mod.project_native_tool_texts(
        messages,
        threshold_tokens=1_000,
        encoder=EligibilityEncoder(),
    )
    projected = await mod.apply_ref_projection_plan(messages, plan)

    expected_ref = f"tool:{digest}"
    assert measurement == (len(encoded), 2, digest)
    assert classification == mod.RefTextClassification(
        eligible=True,
        utf8_bytes=len(encoded),
        sha256=digest,
        line_count=2,
        token_count=1_000,
        encoder_failed=False,
    )
    assert projected == [
        {"role": "tool", "tool_call_id": "call-1", "content": expected_ref},
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": below_threshold,
        },
    ]
    assert measured_texts == [text]
    assert plan.manifests == (
        mod.RefManifest(ref=expected_ref, utf8_bytes=len(encoded), sha256=digest),
    )


@pytest.mark.asyncio
async def test_projection_binds_classified_text_when_message_mutates_across_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classified_text = "classified-A"
    mutated_text = "mutated-B"
    digest = hashlib.sha256(classified_text.encode()).hexdigest()
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": classified_text,
        }
    ]
    classify_ref_text = mod.classify_ref_text

    async def classify_and_mutate(text: str, **kwargs: object):
        classification = await classify_ref_text(text, **kwargs)
        messages[0]["content"] = mutated_text
        return classification

    monkeypatch.setattr(mod, "classify_ref_text", classify_and_mutate)
    plan = await mod.project_native_tool_texts(
        messages,
        threshold_tokens=1_000,
        encoder=CountingEncoder(count=1_000),
    )
    projected = await mod.apply_ref_projection_plan(messages, plan)

    assert len(plan.catalog) == 1
    entry = plan.catalog[0]
    assert isinstance(entry.source, mod.ZeroCopySourceHandle)
    assert entry.source.text == classified_text
    assert entry.manifest.sha256 == digest
    assert projected[0]["content"] == mutated_text


@pytest.mark.asyncio
async def test_unencodable_ref_text_is_ineligible_without_encoder_call() -> None:
    text = "bad-\ud800-text"
    encoder = CountingEncoder(count=1_000)

    measurement = mod._measure_ref_text(text)
    classification = await mod.classify_ref_text(
        text,
        threshold_tokens=1_000,
        encoder=encoder,
    )

    assert measurement is None
    assert classification == mod.RefTextClassification(
        eligible=False,
        utf8_bytes=None,
        sha256=None,
        line_count=None,
        token_count=None,
        encoder_failed=False,
    )
    assert encoder.calls == []


@pytest.mark.asyncio
async def test_pipe_keeps_only_unencodable_tool_text_raw_while_valid_text_externalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = "bad-\ud800-text"
    valid = "valid-line\n" * 7_000
    valid_digest = hashlib.sha256(valid.encode("utf-8")).hexdigest()
    valid_ref = f"tool:{valid_digest}"
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "bad-call",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                },
                {
                    "id": "valid-call",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "bad-call", "content": bad},
        {"role": "tool", "tool_call_id": "valid-call", "content": valid},
        {"role": "user", "content": "continue"},
    ]

    result, forwarded, _, registry, request = await _run_pipe_boundary(
        monkeypatch,
        messages=messages,
    )

    forwarded_messages = forwarded[0]["messages"]
    assert result == {"ok": True}
    assert forwarded_messages[1]["content"] == bad
    assert forwarded_messages[2]["content"] == valid_ref
    assert valid not in repr(forwarded[0])
    assert forwarded[0]["metadata"]["auto_compact_ref_manifests"] == [
        {
            "ref": valid_ref,
            "utf8_bytes": len(valid.encode("utf-8")),
            "sha256": valid_digest,
        }
    ]
    binding = _committed_ref_binding(request)
    assert binding.catalog == (
        mod.RefCatalogEntry(
            manifest=mod.RefManifest(
                ref=valid_ref,
                utf8_bytes=len(valid.encode("utf-8")),
                sha256=valid_digest,
            ),
            source=mod.ZeroCopySourceHandle(text=valid),
        ),
    )
    assert registry[mod.REF_EXEC_TOOL_NAME]["callable"] is binding.reader
    assert await _read(binding.reader, f"head -1 {valid_ref}") == "valid-line\n"


@pytest.mark.asyncio
async def test_ref_threshold_uses_utf8_bytes_not_characters() -> None:
    classifier, _, _, _ = _task1_surface()
    encoder = CountingEncoder(count=1)

    result = await classifier("界" * 21_846, threshold_tokens=10_000, encoder=encoder)

    assert len("界" * 21_846) < 65_536
    assert result.utf8_bytes == 65_538
    assert result.eligible is True
    assert encoder.calls == []


@pytest.mark.asyncio
async def test_ref_threshold_treats_special_tokens_as_plain_text() -> None:
    classifier, _, _, _ = _task1_surface()
    encoder = CountingEncoder(count=1_000)

    result = await classifier(
        "literal <|endoftext|> tool output",
        threshold_tokens=1_000,
        encoder=encoder,
    )

    assert result.eligible is True
    assert encoder.calls[0][1] == {"disallowed_special": ()}


def test_ref_threshold_valve_rejects_values_below_1000() -> None:
    _task1_surface()

    with pytest.raises(ValidationError):
        mod.Pipe.Valves(ref_substitution_threshold_tokens=999)

    valves = mod.Pipe.Valves()
    assert valves.ref_exec_enabled is False
    assert valves.ref_substitution_threshold_tokens == 10_000


@pytest.mark.asyncio
async def test_oversized_tool_text_externalizes_without_tokenizer_or_raw_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _task1_surface()
    raw = "z" * (30 * 1024 * 1024)
    projector_calls = 0
    original_projector = mod.project_native_tool_texts

    async def count_projector(*args: object, **kwargs: object) -> mod.RefProjectionPlan:
        nonlocal projector_calls
        projector_calls += 1
        return await original_projector(*args, **kwargs)

    monkeypatch.setattr(mod, "project_native_tool_texts", count_projector)
    result, forwarded, _, registry, request = await _run_pipe_boundary(
        monkeypatch,
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "private-call",
                        "type": "function",
                        "function": {"name": "existing", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "private-call", "content": raw},
            {"role": "user", "content": "continue"},
        ],
    )

    expected_ref = f"tool:{hashlib.sha256(raw.encode()).hexdigest()}"
    assert result == {"ok": True}
    assert forwarded[0]["messages"][1]["content"] == expected_ref
    assert raw not in repr(forwarded[0])
    assert projector_calls == 1
    assert forwarded[0]["metadata"]["auto_compact_ref_manifests"] == [
        {
            "ref": expected_ref,
            "utf8_bytes": len(raw),
            "sha256": expected_ref.removeprefix("tool:"),
        }
    ]
    assert mod.REF_EXEC_TOOL_NAME in registry
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY, None)
    assert store is not None
    assert len(store.bindings) == 1
    binding = next(iter(store.bindings.values()))
    assert binding.catalog[0].source.text is raw
    assert registry[mod.REF_EXEC_TOOL_NAME]["callable"] is binding.reader


@pytest.mark.asyncio
async def test_pipe_boundary_fake_chats_does_not_poison_core_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_pipe_boundary(
        monkeypatch,
        messages=[{"role": "user", "content": "continue"}],
    )

    from open_webui.models.functions import Functions

    assert await Functions.get_active_filter_ids() == []


@pytest.mark.asyncio
async def test_ref_projection_failure_never_forwards_eligible_text_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _task1_surface()
    raw = "eligible" * 10_000

    stage_operations = (
        "register_ref_attempt",
        "commit_ref_attempt",
        "compare_and_swap_ref_generation",
    )
    for operation_name in stage_operations:
        assert callable(getattr(mod, operation_name, None)), (
            f"Task 1 {operation_name} operation is not implemented"
        )

    valid_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": raw},
        {"role": "user", "content": "continue"},
    ]
    async_operations = {
        "commit_ref_attempt",
        "compare_and_swap_ref_generation",
    }
    for operation_name in stage_operations:
        with monkeypatch.context() as operation_patch:
            if operation_name in async_operations:

                async def fail_async(
                    *_args: object, stage: str = operation_name
                ) -> object:
                    raise mod.RefProjectionError(stage=stage)

                operation_patch.setattr(mod, operation_name, fail_async)
            else:

                def fail_sync(*_args: object, stage: str = operation_name) -> object:
                    raise mod.RefProjectionError(stage=stage)

                operation_patch.setattr(mod, operation_name, fail_sync)
            failed, failed_forwards, _, failed_registry, _ = await _run_pipe_boundary(
                operation_patch,
                messages=valid_messages,
            )
            assert failed["error"]["code"] == "ref_projection_failed"
            assert "before provider forward" in failed["error"]["message"]
            assert failed_forwards == []
            assert mod.REF_EXEC_TOOL_NAME not in failed_registry


@pytest.mark.asyncio
async def test_unavailable_ref_registry_matches_valve_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _task1_surface()
    eligible = "x" * 70_000
    eligible_orphan = [{"role": "tool", "tool_call_id": "orphan", "content": eligible}]
    controls = {
        "tools": [{"type": "function", "function": {"name": "existing"}}],
        "tool_choice": "auto",
        "functions": [{"name": "legacy"}],
        "function_call": "auto",
        "parallel_tool_calls": True,
    }
    valve_off = await _run_pipe_boundary(
        monkeypatch,
        messages=eligible_orphan,
        valve_enabled=False,
        body_overrides=controls,
        registry_has_tool=False,
        metadata_overrides={"tools": {"detached": {}}},
    )
    inactive = await _run_pipe_boundary(
        monkeypatch,
        messages=eligible_orphan,
        body_overrides=controls,
        registry_has_tool=False,
        metadata_overrides={"tools": {"detached": {}}},
    )

    assert inactive[0] == valve_off[0] == {"ok": True}
    assert inactive[1:] == valve_off[1:]


@pytest.mark.asyncio
async def test_always_strip_summary_keeps_refs_without_reader_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _task1_surface()
    raw = "summary tool result" * 5_000
    source_messages = [
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
        {"role": "tool", "tool_call_id": "call-1", "content": raw},
    ]
    base_body = {
        "model": "target",
        "messages": source_messages,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto",
        "functions": [object()],
        "function_call": "auto",
        "parallel_tool_calls": True,
    }
    request = SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={})),
    )
    projection_plan = await mod.project_native_tool_texts(
        source_messages,
        threshold_tokens=1_000,
        encoder=CountingEncoder(count=1_000),
    )
    captured: dict[str, list[dict[str, object]]] = {
        "always": [],
        "fallback": [],
        "pipe": [],
    }
    active_policy = "always"

    async def prepare_file_context(**_kwargs: object) -> None:
        return None

    async def resolve_route(
        *_args: object, **_kwargs: object
    ) -> mod.CoreChatModelRoute:
        return mod.CoreChatModelRoute(model_id="target")

    async def resolve_arena_route(
        **kwargs: object,
    ) -> tuple[mod.CoreChatModelRoute, None]:
        return kwargs["route"], None

    async def model_dict_from_request(_request: object) -> dict[str, object]:
        return {"target": {"id": "target", "name": "Target"}}

    async def ensure_model(*_args: object, **_kwargs: object) -> None:
        return None

    async def generate_chat_completion(
        _request: object,
        form_data: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        captured[active_policy].append(copy.deepcopy(form_data))
        if active_policy == "fallback" and len(captured[active_policy]) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "reader-call",
                                    "type": "function",
                                    "function": {
                                        "name": mod.REF_EXEC_TOOL_NAME,
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "summary"}}]}

    chat_module = types.ModuleType("open_webui.utils.chat")
    chat_module.generate_chat_completion = generate_chat_completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_module)
    monkeypatch.setattr(mod, "_prepare_summary_file_context", prepare_file_context)
    monkeypatch.setattr(mod, "_resolve_core_chat_model_route", resolve_route)
    monkeypatch.setattr(
        mod, "_resolve_arena_chat_model_route_with_access", resolve_arena_route
    )
    monkeypatch.setattr(mod, "_model_dict_from_request", model_dict_from_request)
    monkeypatch.setattr(mod, "_ensure_model_in_request_models", ensure_model)
    monkeypatch.setattr(mod, "coerce_open_webui_user", lambda user: user)

    await mod._generate_summary_text(
        request=request,
        user={"id": "user-1"},
        metadata={"chat_id": "chat-1"},
        summary_model_id="target",
        source_messages=source_messages,
        base_body=base_body,
        summary_tool_policy="always_strip",
        ref_projection_plan=projection_plan,
    )
    active_policy = "fallback"
    await mod._generate_summary_text(
        request=request,
        user={"id": "user-1"},
        metadata={"chat_id": "chat-1"},
        summary_model_id="target",
        source_messages=source_messages,
        base_body=base_body,
        summary_tool_policy="fallback_on_tool_call",
        ref_projection_plan=projection_plan,
    )

    expected_ref = f"tool:{hashlib.sha256(raw.encode()).hexdigest()}"
    always_body = captured["always"][0]
    fallback_first, fallback_retry = captured["fallback"]
    assert always_body["messages"][1]["content"] == expected_ref
    assert fallback_first["messages"] == always_body["messages"]
    assert any(
        tool["function"]["name"] == mod.REF_EXEC_TOOL_NAME
        for tool in fallback_first["tools"]
    )
    assert fallback_retry == always_body
    assert (
        not {
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "parallel_tool_calls",
        }
        & always_body.keys()
    )
    assert mod.summary_ref_registry() == MappingProxyType({})

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/ref-checkpoints.db",
        poolclass=NullPool,
    )
    monkeypatch.setattr(mod, "_CHECKPOINT_SCHEMA_READY", False)
    await mod.ensure_checkpoint_table_initialized(async_engine=engine)
    sessionmaker = async_sessionmaker(bind=engine, expire_on_commit=False)

    class EngineCheckpointStore(mod.CheckpointStore):
        async def _context(self):
            return sessionmaker()

    async def checkpoint_schema_ready(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        mod, "ensure_checkpoint_table_initialized", checkpoint_schema_ready
    )
    monkeypatch.setattr(mod, "CheckpointStore", EngineCheckpointStore)
    active_policy = "pipe"

    def configure_pipe(pipe: mod.Pipe) -> None:
        pipe.valves.summary_tool_policy = "always_strip"
        pipe.valves.trigger_input_tokens = 2

    try:
        (
            pipe_result,
            pipe_forwards,
            pipe_checkpoint_bodies,
            pipe_registry,
            _,
        ) = await _run_pipe_boundary(
            monkeypatch,
            messages=[*source_messages, {"role": "user", "content": "continue"}],
            estimated_tokens=10,
            configure_pipe=configure_pipe,
        )

        second_messages = [
            *source_messages,
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "intermediate answer"},
            {"role": "user", "content": "next"},
        ]
        (
            second_result,
            second_forwards,
            _,
            second_registry,
            _,
        ) = await _run_pipe_boundary(
            monkeypatch,
            messages=second_messages,
            estimated_tokens=10,
            configure_pipe=configure_pipe,
        )
        summary_calls_after_child = len(captured["pipe"])
        (
            reused_result,
            reused_forwards,
            _,
            reused_registry,
            _,
        ) = await _run_pipe_boundary(
            monkeypatch,
            messages=second_messages,
            estimated_tokens=10,
            configure_pipe=configure_pipe,
        )

        async with sessionmaker() as session:
            checkpoint_rows = list(
                (await session.execute(mod.CHECKPOINT_TABLE.select())).mappings()
            )
    finally:
        await engine.dispose()

    assert pipe_result == {"ok": True}
    assert pipe_checkpoint_bodies[0]["messages"] == [
        *source_messages,
        {"role": "user", "content": "continue"},
    ]
    assert captured["pipe"][0]["messages"][1]["content"] == expected_ref
    assert (
        not {
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "parallel_tool_calls",
        }
        & captured["pipe"][0].keys()
    )
    assert (
        pipe_forwards[0]["metadata"]["auto_compact_ref_manifests"]
        == always_body["metadata"]["auto_compact_ref_manifests"]
    )
    assert any(
        tool["function"]["name"] == mod.REF_EXEC_TOOL_NAME
        for tool in pipe_forwards[0]["tools"]
    )
    assert mod.REF_EXEC_TOOL_NAME in pipe_registry
    assert second_result == reused_result == {"ok": True}
    assert second_forwards[0]["metadata"]["tools"] == second_registry
    assert reused_forwards[0]["metadata"]["tools"] == reused_registry
    assert (
        second_registry[mod.REF_EXEC_TOOL_NAME]["callable"]
        is not reused_registry[mod.REF_EXEC_TOOL_NAME]["callable"]
    )
    second_without_registry = copy.deepcopy(second_forwards)
    reused_without_registry = copy.deepcopy(reused_forwards)
    second_without_registry[0]["metadata"].pop("tools")
    reused_without_registry[0]["metadata"].pop("tools")
    assert second_without_registry == reused_without_registry
    assert len(captured["pipe"]) == summary_calls_after_child
    assert len(checkpoint_rows) == 2
    parent, child = sorted(checkpoint_rows, key=lambda row: row["source_message_count"])
    assert parent["state"] == child["state"] == "ready"
    assert parent["profile_hash"] == child["profile_hash"] == mod.compute_profile_hash()
    assert parent["source_hash"] == mod.compute_summary_source_hash(source_messages)
    child_source_messages = second_messages[: child["source_message_count"]]
    assert child["source_hash"] == mod.compute_summary_source_hash(
        child_source_messages
    )
    assert child["source_hash"] != parent["source_hash"]
    assert child["source_message_count"] > parent["source_message_count"]
    assert parent["parent_checkpoint_id"] is None
    assert child["parent_checkpoint_id"] == parent["id"]


@pytest.mark.asyncio
async def test_ref_profile_hash_and_lineage_are_identical_across_mode_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _task1_surface()
    baseline = mod.compute_profile_hash()

    variants = (
        mod.compute_profile_hash(ref_exec_enabled=True),
        mod.compute_profile_hash(model="other"),
        mod.compute_profile_hash(ref_substitution_threshold_tokens=1_000),
        mod.compute_profile_hash(encoding="other"),
        mod.compute_profile_hash(reader_contract="v99"),
    )

    assert all(value == baseline for value in variants)
    raw = "lineage" * 10_000
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": raw},
        {"role": "user", "content": "continue"},
    ]
    active = await _run_pipe_boundary(monkeypatch, messages=messages)
    inactive = await _run_pipe_boundary(
        monkeypatch, messages=messages, valve_enabled=False
    )

    assert active[0] == inactive[0] == {"ok": True}
    assert active[2] == inactive[2]
    assert active[2][0]["messages"] == messages
    parent_source = messages[:1]
    parent = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=baseline,
        source_hash=mod.compute_source_hash(parent_source),
        source_message_count=1,
        summary_text="parent",
        summary_meta=None,
        parent_checkpoint_id=None,
        now=1,
    )
    child = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=baseline,
        source_hash=mod.compute_source_hash(messages),
        source_message_count=len(messages),
        summary_text="child",
        summary_meta=None,
        parent_checkpoint_id=parent["id"],
        now=2,
    )
    assert child["parent_checkpoint_id"] == parent["id"]
    assert child["profile_hash"] == parent["profile_hash"] == baseline
    assert child["source_hash"] == mod.compute_source_hash(messages)
    assert parent["source_hash"] == mod.compute_source_hash(parent_source)
    for contract_name in (
        "ParsedRef",
        "RefManifest",
        "ZeroCopySourceHandle",
        "RefCatalogEntry",
        "RefProjectionPlan",
        "RefResolverResult",
        "EffectiveRefMode",
        "RefStateDelta",
        "RefBindingKey",
        "RefBindingState",
        "RefAttempt",
    ):
        contract = getattr(mod, contract_name, None)
        assert dataclasses.is_dataclass(contract)
        assert contract.__dataclass_params__.frozen is True
        assert "__slots__" in contract.__dict__


async def _reader_fixture(
    monkeypatch: pytest.MonkeyPatch,
    texts: tuple[str, ...] = ("alpha\nbeta target\ngamma",),
    *,
    threshold_tokens: int = 10_000,
    encoder: object | None = None,
    suffix: str = "main",
) -> tuple[ReaderFixture, tuple[str, ...], SimpleNamespace, mod.RefBindingKey]:
    request = SimpleNamespace(
        state=SimpleNamespace(), app=SimpleNamespace(state=SimpleNamespace())
    )
    key = mod.RefBindingKey(
        incoming_model_id="target",
        assistant_message_id=f"private-assistant-{suffix}",
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
    )
    entries = []
    refs = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).hexdigest()
        ref = f"tool:{digest}"
        refs.append(ref)
        entries.append(
            mod.RefCatalogEntry(
                manifest=mod.RefManifest(
                    ref=ref, utf8_bytes=len(text.encode()), sha256=digest
                ),
                source=mod.ZeroCopySourceHandle(text=text),
            )
        )
    registry: dict[str, object] = {"unrelated": {"spec": {"name": "unrelated"}}}

    reader = mod._new_ref_reader(
        request,
        key,
        threshold_tokens=threshold_tokens,
        encoder=encoder,
    )
    store = mod.RefRequestStore()
    store.bindings[key] = mod.RefBindingState(
        generation=1,
        catalog=tuple(entries),
        registry=registry,
        reader=reader,
    )
    store.registry_owners[id(registry)] = key
    setattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY, store)
    registry[mod.REF_EXEC_TOOL_NAME] = {
        "spec": mod.ref_exec_tool_spec_payload(),
        "callable": reader,
    }
    return reader, tuple(refs), request, key


async def _read(reader: Callable[[str], object], command: str) -> str:
    result = reader(command)
    assert inspect.isawaitable(result)
    return await result


class Item7ByteEncoder:
    def encode(self, text: str, **_kwargs: object) -> list[int]:
        return list(text.encode())


def _item7_minified_json_source() -> str:
    source = json.dumps(
        {
            "records": [
                {
                    "id": index,
                    "match": f"hit-{index:04d}",
                    "payload": "x" * 72,
                }
                for index in range(1_400)
            ]
        },
        separators=(",", ":"),
    )
    assert 150 * 1_024 <= len(source.encode()) < 160 * 1_024
    assert "\n" not in source
    return source


def _item7_visible_and_next(result: str) -> tuple[str, str | None]:
    for tag in ("auto_compact_ref_range", "auto_compact_ref_truncated"):
        opening = f"\n<{tag}>"
        if opening in result:
            visible, encoded_marker = result.split(opening, 1)
            marker = json.loads(encoded_marker.removesuffix(f"</{tag}>"))
            return visible, marker["next"]
    return result, None


_REF_EXEC_USAGE_ERROR: Final = (
    "Error: usage: auto_compact_ref_exec(command). Expected REF: "
    "tool:<64 hex> or history:accp_<64 hex>"
)


def test_ref_exec_parser_preserves_valid_command_behavior() -> None:
    ref = f"tool:{'a' * 64}"

    stages = mod._parse_ref_exec_command(f"cat {ref}")

    assert stages == (mod.RefExecStage(command="cat", ref=ref),)


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

    assert "Expected REF: tool:<64 hex> or history:accp_<64 hex>" in str(
        failure.value
    )


@pytest.mark.asyncio
async def test_ref_exec_reader_maps_unpaired_surrogate_to_reader_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, _, _, _ = await _reader_fixture(monkeypatch)

    result = await _read(reader, "cat \ud800")

    assert "Expected REF: tool:<64 hex> or history:accp_<64 hex>" in result


@pytest.mark.asyncio
async def test_ref_exec_reader_omitted_command_returns_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, _, _, _ = await _reader_fixture(monkeypatch)

    result = await reader()

    assert result == _REF_EXEC_USAGE_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize(("command"), (None, []), ids=("none", "list"))
async def test_ref_exec_reader_rejects_non_string_before_parser(
    monkeypatch: pytest.MonkeyPatch,
    command: CoreFixtureValue,
) -> None:
    reader, _, _, _ = await _reader_fixture(monkeypatch)
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
async def test_ref_exec_reader_signature_preserves_required_provider_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, _, _, _ = await _reader_fixture(monkeypatch)

    command = inspect.signature(reader, eval_str=True).parameters["command"]
    parameters = mod.ref_exec_tool_spec_payload()["function"]["parameters"]
    command_schema = parameters["properties"]["command"]

    assert command.default == ""
    assert command.annotation is str
    assert command_schema["type"] == "string"
    assert parameters["required"] == ["command"]
    assert "default" not in command_schema


@pytest.mark.asyncio
async def test_ref_exec_supports_bounded_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
    ref = refs[0]

    assert await _read(reader, f"cat {ref}") == "alpha\nbeta target\ngamma"
    assert "target" in await _read(reader, f"grep target {ref}")
    assert ref in await _read(reader, "ls tool")
    assert "utf8_bytes=" in await _read(reader, f"stat {ref}")
    spec = mod.ref_exec_tool_spec_payload()["function"]
    assert spec["name"] == "auto_compact_ref_exec"
    assert spec["description"] == (
        "Read externalized content in this chat. Oversized tool results and compacted history are replaced by ref tokens: tool:<64 hex> or history:accp_<64 hex>. "
        "When a tool message's content is such a token, the original text is retrievable only through this tool. Commands: ls [tool|history]; stat REF; "
        "wc -l|-w|-c REF; cat REF; head [-n N|-N|-c N] REF; tail [-n N|-N|-c N|-c +N] REF; sed -n 'M,Np' REF; grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN REF "
        "(patterns match literally unless regex syntax is auto-detected; -E forces regex). REF is the complete token including its tool:/history: prefix, exactly as written. Pipelines are supported; "
        "only grep/head/tail/sed/wc consume piped input, e.g. grep -n PATTERN tool:<hash> | head -20. Start with stat, then prefer grep/sed/head over cat for large refs."
    )
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
    assert "history:accp_<64 hex>" in description


@pytest.mark.asyncio
async def test_ref_exec_supports_head_tail_sed_and_wc_ux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
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
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("\nalpha",))

    result = await _read(reader, command.format(ref=refs[0]))

    assert result == "\nalpha"


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


def test_ref_exec_utf8_prefix_index_maps_surrogate_reached_by_scan() -> None:
    text = "prefix-\ud800-suffix"

    with pytest.raises(mod.RefExecError) as failure:
        mod._ref_exec_utf8_prefix_index(text, 64, threading.Event())

    assert str(failure.value) == "Error: externalized ref source is not valid UTF-8"
    assert isinstance(failure.value.__cause__, UnicodeEncodeError)

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(mod.RefExecError, match="cancelled") as cancelled_failure:
        mod._ref_exec_utf8_prefix_index(text, 64, cancelled)
    assert cancelled_failure.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "cat {ref}",
        "wc -l {ref}",
        "wc -w {ref}",
        "wc -c {ref}",
        "stat {ref}",
        "head -c 4 {ref}",
        "tail -c +2 {ref}",
    ),
)
async def test_ref_exec_zero_copy_reader_commands_map_unpaired_surrogate(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    reader, refs, request, key = await _reader_fixture(monkeypatch, ("valid",))
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    entry = store.bindings[key].catalog[0]
    store.bindings[key] = dataclasses.replace(
        store.bindings[key],
        catalog=(
            dataclasses.replace(
                entry,
                source=mod.ZeroCopySourceHandle(text="valid\ud800"),
            ),
        ),
    )

    result = await _read(reader, command.format(ref=refs[0]))

    assert result == "Error: externalized ref source is not valid UTF-8"


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


@pytest.mark.asyncio
async def test_ref_exec_checked_utf8_boundaries_preserve_valid_bytes_ranges_and_hash() -> (
    None
):
    text = "A界\n🙂é"
    encoded = text.encode("utf-8")
    expected_hash = hashlib.sha256(encoded).hexdigest()
    monkeypatch = pytest.MonkeyPatch()
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (text,))
    ref = refs[0]

    try:
        assert await _read(reader, f"cat {ref}") == text
        assert await _read(reader, f"wc -c {ref}") == str(len(encoded))
        assert await _read(reader, f"head -c 4 {ref}") == "A界"
        assert await _read(reader, f"tail -c +5 {ref}") == "\n🙂é"
        stat = await _read(reader, f"stat {ref}")
        assert f"utf8_bytes={len(encoded)}" in stat
        assert f"sha256={expected_hash}" in stat
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_ref_exec_head_byte_ranges_use_presented_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("alpha\nbeta",))
    ref = refs[0]

    assert await _read(reader, f"head -c 0 {ref}") == ""
    assert await _read(reader, f"head -c 7 {ref}") == "alpha\nb"
    assert await _read(reader, f"head -c 999 {ref}") == "alpha\nbeta"
    assert "auto_compact_ref_range" not in await _read(reader, f"head -c 999 {ref}")


@pytest.mark.asyncio
async def test_ref_exec_tail_final_byte_ranges_use_presented_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("alpha\nbeta\n",))
    ref = refs[0]

    assert await _read(reader, f"tail -c 0 {ref}") == ""
    assert await _read(reader, f"tail -c 5 {ref}") == "beta\n"
    assert await _read(reader, f"tail -c 999 {ref}") == "alpha\nbeta\n"
    assert "auto_compact_ref_range" not in await _read(reader, f"tail -c 999 {ref}")


@pytest.mark.asyncio
async def test_ref_exec_tail_from_offset_is_one_based_and_clips_at_source_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("alpha\nbeta",))
    ref = refs[0]

    assert await _read(reader, f"tail -c +1 {ref}") == "alpha\nbeta"
    assert await _read(reader, f"tail -c +7 {ref}") == "beta"
    assert await _read(reader, f"tail -c +999 {ref}") == ""
    assert "auto_compact_ref_range" not in await _read(reader, f"tail -c +999 {ref}")


@pytest.mark.parametrize(
    (
        "operation",
        "amount",
        "expected_text",
        "expected_byte_starts",
        "expected_char_starts",
        "expected_newlines",
        "expected_matches",
        "expected_ranges",
        "expected_components",
    ),
    (
        pytest.param(
            "head",
            5,
            ("abc\n", "d"),
            (100, 104),
            (200, 204),
            (False, False),
            ((None, None), (None, None)),
            ((1, 5, 1, 4, False, False), (1, 5, 5, 5, False, False)),
            (
                (
                    ("text", "abc", 100, 200),
                    ("synthetic_lf", "\n", 103, 203),
                ),
                (("text", "d", 104, 204),),
            ),
            id="head-retained-and-terminal-partial",
        ),
        pytest.param(
            "tail",
            7,
            ("c\n", "de\n", "fg"),
            (2, 4, 7),
            (202, 204, 207),
            (False, True, False),
            ((None, None), (0, 2), (0, 2)),
            ((3, 9, 3, 9, False, False),) * 3,
            (
                (
                    ("text", "c", 102, 202),
                    ("synthetic_lf", "\n", 103, 203),
                ),
                (
                    ("text", "de", 4, 204),
                    ("synthetic_lf", "\n", 6, 206),
                ),
                (("text", "fg", 7, 207),),
            ),
            id="tail-partial-and-retained",
        ),
        pytest.param(
            "tail-from",
            3,
            ("c\n", "de\n", "fg"),
            (102, 104, 107),
            (202, 204, 207),
            (False, False, False),
            ((None, None), (None, None), (None, None)),
            ((3, None, 3, None, False, False),) * 3,
            (
                (
                    ("text", "c", 102, 202),
                    ("synthetic_lf", "\n", 103, 203),
                ),
                (
                    ("text", "de", 104, 204),
                    ("synthetic_lf", "\n", 106, 206),
                ),
                (("text", "fg", 107, 207),),
            ),
            id="tail-from-partial-and-passthrough",
        ),
    ),
)
def test_ref_exec_byte_stages_clear_atomic_match_on_every_output_branch(
    operation: str,
    amount: int,
    expected_text: tuple[str, ...],
    expected_byte_starts: tuple[int, ...],
    expected_char_starts: tuple[int, ...],
    expected_newlines: tuple[bool, ...],
    expected_matches: tuple[tuple[int | None, int | None], ...],
    expected_ranges: tuple[tuple[int | None, ...], ...],
    expected_components: tuple[
        tuple[tuple[str, str, int | None, int | None], ...], ...
    ],
) -> None:
    cancelled = threading.Event()
    lines = (
        mod.RefExecLine("abc", 10, 100, 200, True, 0, 3, atomic_match=True),
        mod.RefExecLine("de", 20, 104, 204, True, 0, 2, atomic_match=True),
        mod.RefExecLine("fg", 30, 107, 207, False, 0, 2, atomic_match=True),
    )
    stage = {
        "head": mod._ref_exec_head_bytes,
        "tail": mod._ref_exec_tail_bytes,
        "tail-from": mod._ref_exec_tail_from_bytes,
    }[operation]

    selected = tuple(stage(iter(lines), amount, cancelled))

    assert tuple(
        mod._ref_exec_materialize_line(line, cancelled) for line in selected
    ) == expected_text
    assert tuple(line.number for line in selected) == (10, 20, 30)[: len(selected)]
    assert tuple(line.byte_start for line in selected) == expected_byte_starts
    assert tuple(line.char_start for line in selected) == expected_char_starts
    assert tuple(line.has_newline for line in selected) == expected_newlines
    assert tuple((line.match_start, line.match_end) for line in selected) == (
        expected_matches
    )
    assert tuple(
        (
            line.byte_range.requested_start,
            line.byte_range.requested_end,
            line.byte_range.actual_start,
            line.byte_range.actual_end,
            line.byte_range.marked,
            line.byte_range.actual_empty,
        )
        for line in selected
        if line.byte_range is not None
    ) == expected_ranges
    assert tuple(
        tuple(
            (
                component.kind,
                component.text[component.start : component.end],
                component.source_byte_start,
                component.source_char_start,
            )
            for component in mod._ref_exec_line_component_view(
                line, cancelled
            ).components
        )
        for line in selected
    ) == expected_components
    assert [line.atomic_match for line in selected] == [False] * len(selected)


def test_ref_exec_grep_only_reestablishes_atomic_match_after_byte_stage() -> None:
    cancelled = threading.Event()
    source = mod.RefExecLine(
        "MATCH",
        9,
        500,
        600,
        False,
        0,
        5,
        atomic_match=True,
    )

    cleared = tuple(mod._ref_exec_head_bytes(iter((source,)), 5, cancelled))
    matches = tuple(
        mod._ref_exec_grep(
            iter(cleared),
            mod.RefExecStage(
                command="grep", pattern="MATCH", flags=frozenset({"o"})
            ),
            cancelled,
        )
    )

    assert len(cleared) == 1
    assert len(matches) == 1
    assert (matches[0].text, matches[0].byte_start, matches[0].char_start) == (
        "MATCH",
        500,
        600,
    )
    assert matches[0].atomic_match is True
    assert cleared[0].atomic_match is False


@pytest.mark.asyncio
async def test_ref_exec_byte_ranges_snap_utf8_boundaries_inward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("A界B",))
    ref = refs[0]

    head = await _read(reader, f"head -c 3 {ref}")
    tail = await _read(reader, f"tail -c 3 {ref}")
    from_offset = await _read(reader, f"tail -c +3 {ref}")

    assert head.startswith("A")
    assert tail.startswith("B")
    assert from_offset.startswith("B")
    for result, requested, actual in (
        (head, "1-3", "1-1"),
        (tail, "3-5", "5-5"),
        (from_offset, "3-*", "5-*"),
    ):
        payload = result.split("<auto_compact_ref_range>", 1)[1].split(
            "</auto_compact_ref_range>", 1
        )[0]
        assert json.loads(payload) == {"actual": actual, "requested": requested}
        assert "�" not in result


@pytest.mark.parametrize(
    ("command", "requested"),
    (
        ("head -c 1", "1-1"),
        ("head -c 2", "1-2"),
        ("tail -c 1", "3-3"),
        ("tail -c 2", "2-3"),
        ("tail -c +2", "2-*"),
        ("tail -c +3", "3-*"),
    ),
)
@pytest.mark.asyncio
async def test_ref_exec_empty_inward_byte_snaps_keep_range_metadata(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    requested: str,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("界",))

    result = await _read(reader, f"{command} {refs[0]}")
    payload = result.split("<auto_compact_ref_range>", 1)[1].split(
        "</auto_compact_ref_range>", 1
    )[0]

    assert json.loads(payload) == {"actual": None, "requested": requested}
    assert "�" not in result


@pytest.mark.asyncio
async def test_ref_exec_natural_empty_byte_ranges_remain_unmarked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("界",))

    for command in ("head -c 0", "tail -c 0", "tail -c +4"):
        assert await _read(reader, f"{command} {refs[0]}") == ""


def test_ref_exec_zero_byte_budget_prefix_marks_an_empty_actual_range() -> None:
    result = mod._ref_exec_truncated_byte_response(
        "",
        byte_range=mod.RefExecByteRange(
            requested_start=1,
            requested_end=500,
            actual_start=1,
            actual_end=500,
        ),
        threshold_tokens=10_000,
        encoder=None,
        cancelled=threading.Event(),
    )
    payload = result.split("<auto_compact_ref_range>", 1)[1].split(
        "</auto_compact_ref_range>", 1
    )[0]

    assert json.loads(payload) == {
        "actual": None,
        "next": "wc|grep|head|tail|sed",
        "requested": "1-500",
    }


@pytest.mark.asyncio
async def test_ref_exec_multiline_head_aggregates_actual_byte_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("A\n界B",))

    result = await _read(reader, f"head -c 3 {refs[0]}")
    visible, _, marker = result.partition("\n<auto_compact_ref_range>")
    payload = marker.split("</auto_compact_ref_range>", 1)[0]

    assert visible == "A\n"
    assert json.loads(payload) == {"actual": "1-2", "requested": "1-3"}


@pytest.mark.asyncio
async def test_ref_exec_multiline_budget_range_starts_at_first_returned_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ByteEncoder:
        def encode(self, text: str, **_kwargs: object) -> list[int]:
            return list(text.encode())

    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        ("a" * 100 + "\n" + "b" * 400,),
        threshold_tokens=180,
        encoder=ByteEncoder(),
    )

    result = await _read(reader, f"head -c 500 {refs[0]}")
    visible, _, marker = result.partition("\n<auto_compact_ref_range>")
    payload = marker.split("</auto_compact_ref_range>", 1)[0]

    assert visible.startswith("a")
    assert json.loads(payload) == {
        "actual": f"1-{len(visible.encode())}",
        "next": "wc|grep|head|tail|sed",
        "requested": "1-500",
    }


@pytest.mark.asyncio
async def test_ref_exec_rolling_tail_discards_stale_snap_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("α\n界\nZ",))

    result = await _read(reader, f"tail -c 2 {refs[0]}")

    assert result == "\nZ"
    assert "auto_compact_ref_range" not in result


def test_ref_exec_fixed_tail_bounds_newline_only_memory() -> None:
    source_bytes = 1024 * 1024
    cancelled = threading.Event()

    def source():
        for index in range(source_bytes + 1):
            yield mod.RefExecLine(
                "",
                index + 1,
                index,
                index,
                index < source_bytes,
            )

    emitted_bytes = 0
    tracemalloc.start()
    try:
        for line in mod._ref_exec_tail_bytes(source(), source_bytes, cancelled):
            emitted_bytes += mod._ref_exec_presented_line_bytes(line, cancelled)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert emitted_bytes == source_bytes
    assert peak_bytes < 16 * 1024 * 1024


def test_ref_exec_fixed_tail_enumerates_source_once() -> None:
    iterations = 0

    class OneShotLines:
        def __iter__(self):
            nonlocal iterations
            iterations += 1
            assert iterations == 1
            yield mod.RefExecLine("alpha", 1, 0, 0, True)
            yield mod.RefExecLine("beta", 2, 6, 6, False)

    cancelled = threading.Event()
    selected = mod._ref_exec_tail_bytes(OneShotLines(), 5, cancelled)
    result = "".join(
        mod._ref_exec_materialize_line(line, cancelled) for line in selected
    )

    assert result == "\nbeta"
    assert iterations == 1


def test_ref_exec_fixed_tail_prioritizes_tampered_suffix_integrity() -> None:
    expected = "prefix\ntrusted"
    ref = f"tool:{hashlib.sha256(expected.encode()).hexdigest()}"
    entry = mod.RefCatalogEntry(
        manifest=mod.RefManifest(
            ref=ref,
            utf8_bytes=len(expected.encode()),
            sha256=hashlib.sha256(expected.encode()).hexdigest(),
        ),
        source=mod.ZeroCopySourceHandle(text="prefix\ntampered"),
    )

    with pytest.raises(mod.RefExecError, match="integrity verification failed"):
        mod._execute_ref_reader_sync(
            (mod.RefExecStage(command="tail", ref=ref, byte_count=1),),
            (entry,),
            threshold_tokens=10_000,
            encoder=None,
            cancelled=threading.Event(),
        )


def test_ref_exec_fixed_tail_prioritizes_cancellation_over_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = threading.Event()
    expected = "trusted"
    ref = f"tool:{hashlib.sha256(expected.encode()).hexdigest()}"
    entry = mod.RefCatalogEntry(
        manifest=mod.RefManifest(
            ref=ref,
            utf8_bytes=len(expected.encode()),
            sha256=hashlib.sha256(expected.encode()).hexdigest(),
        ),
        source=mod.ZeroCopySourceHandle(text="tampered"),
    )

    def tampered_source(_source, _cancelled):
        yield mod.RefExecLine("tampered", 1, 0, 0, False)
        cancelled.set()

    monkeypatch.setattr(mod, "_iter_ref_source_lines", tampered_source)

    with pytest.raises(mod.RefExecError, match="cancelled") as failure:
        mod._execute_ref_reader_sync(
            (mod.RefExecStage(command="tail", ref=ref, byte_count=1),),
            (entry,),
            threshold_tokens=10_000,
            encoder=None,
            cancelled=cancelled,
        )

    assert "integrity" not in str(failure.value)


def test_ref_exec_fixed_tail_preserves_fully_retained_metadata() -> None:
    lines = (
        mod.RefExecLine("a", 10, 100, 200, True, 0, 1, atomic_match=True),
        mod.RefExecLine("b", 20, 102, 202, True, 0, 1, atomic_match=True),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(iter(lines), 4, threading.Event())
    )

    assert [
        (
            line.number,
            line.byte_start,
            line.char_start,
            line.match_start,
            line.match_end,
            line.atomic_match,
        )
        for line in selected
    ] == [(10, 0, 200, 0, 1, False), (20, 2, 202, 0, 1, False)]


def test_ref_exec_fixed_tail_preserves_partial_first_line_metadata() -> None:
    line = mod.RefExecLine(
        "abc",
        30,
        300,
        400,
        True,
        0,
        3,
        atomic_match=True,
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(iter((line,)), 3, threading.Event())
    )

    assert [
        (
            result.number,
            result.char_start,
            result.match_start,
            result.match_end,
            result.atomic_match,
        )
        for result in selected
    ] == [(30, 401, None, None, False)]


def test_ref_exec_fixed_tail_preserves_trailing_empty_metadata() -> None:
    lines = (
        mod.RefExecLine("x", 4, 0, 8, False, 0, 1, atomic_match=True),
        mod.RefExecLine("", 9, 1, 9, False, 0, 0, atomic_match=True),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(iter(lines), 1, threading.Event())
    )

    assert [
        (
            line.number,
            line.char_start,
            line.match_start,
            line.match_end,
            line.atomic_match,
        )
        for line in selected
    ] == [(4, 8, 0, 1, False), (9, 9, 0, 0, False)]


def test_ref_exec_fixed_tail_preserves_snapped_empty_metadata() -> None:
    line = mod.RefExecLine(
        "é",
        44,
        90,
        700,
        False,
        0,
        1,
        atomic_match=True,
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(iter((line,)), 1, threading.Event())
    )

    assert [
        (
            result.number,
            result.byte_start,
            result.char_start,
            result.match_start,
            result.match_end,
            result.atomic_match,
            result.metadata_only,
        )
        for result in selected
    ] == [(44, 2, 700, None, None, False, True)]


def test_ref_exec_fixed_tail_preserves_synthetic_only_partial_metadata() -> None:
    lines = (
        mod.RefExecLine("a", 10, 0, 40, True),
        mod.RefExecLine("b", 20, 2, 42, True),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(iter(lines), 3, threading.Event())
    )

    assert [
        (
            mod._ref_exec_materialize_line(line, threading.Event()),
            line.number,
            line.byte_start,
            line.char_start,
            line.has_newline,
        )
        for line in selected
    ] == [("\n", 10, 1, 40, False), ("b\n", 20, 2, 42, True)]


def test_ref_exec_fixed_tail_marks_presliced_synthetic_record_partial() -> None:
    lines = (
        mod.RefExecLine("", 10, 0, 0, True),
        mod.RefExecLine("a", 20, 1, 1, True),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(iter(lines), 1, threading.Event())
    )

    assert [
        (
            mod._ref_exec_materialize_line(line, threading.Event()),
            line.number,
            line.byte_start,
            line.char_start,
            line.has_newline,
        )
        for line in selected
    ] == [("\n", 20, 2, 1, False)]


def test_ref_exec_fixed_tail_discards_stale_bytes_after_empty_utf8_snap() -> None:
    lines = (
        mod.RefExecLine("", 10, 0, 0, True, 0, 0, atomic_match=True),
        mod.RefExecLine("界", 20, 1, 1, False, 0, 1, atomic_match=True),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(iter(lines), 1, threading.Event())
    )

    assert [
        (
            mod._ref_exec_materialize_line(line, threading.Event()),
            line.number,
            line.byte_start,
            line.char_start,
            line.has_newline,
            line.match_start,
            line.match_end,
            line.metadata_only,
            line.atomic_match,
        )
        for line in selected
    ] == [("", 20, 4, 1, False, None, None, True, False)]


def test_ref_exec_fixed_tail_discards_stale_bytes_before_partial_rewrite() -> None:
    lines = (
        mod.RefExecLine("", 10, 0, 0, True),
        mod.RefExecLine("界", 20, 1, 1, True),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(iter(lines), 2, threading.Event())
    )

    assert [
        (
            mod._ref_exec_materialize_line(line, threading.Event()),
            line.number,
            line.byte_start,
            line.char_start,
            line.has_newline,
            line.match_start,
            line.match_end,
            line.metadata_only,
        )
        for line in selected
    ] == [("\n", 20, 4, 1, False, None, None, False)]


def test_ref_exec_fixed_tail_preserves_transformed_false_newline_metadata() -> None:
    source = mod.RefExecLine("a", 10, 0, 40, True)
    transformed = mod._ref_exec_head_bytes(
        iter((source,)),
        2,
        threading.Event(),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(transformed, 2, threading.Event())
    )

    assert [
        (
            mod._ref_exec_materialize_line(line, threading.Event()),
            line.number,
            line.byte_start,
            line.char_start,
            line.has_newline,
        )
        for line in selected
    ] == [("a\n", 10, 0, 40, False)]


def test_ref_exec_fixed_tail_preserves_fully_retained_plain_representation() -> None:
    cancelled = threading.Event()
    source = mod.RefExecLine("abc", 7, 0, 0, False)

    selected = tuple(mod._ref_exec_tail_bytes(iter((source,)), 3, cancelled))
    nested = tuple(mod._ref_exec_head_bytes(iter(selected), 3, cancelled))

    assert selected[0].component_view is None
    assert (
        selected[0].text,
        selected[0].number,
        selected[0].byte_start,
        selected[0].char_start,
        selected[0].has_newline,
        selected[0].display_prefix,
    ) == ("abc", 7, 0, 0, False, "")
    assert nested[0].byte_range is not None
    assert (
        nested[0].byte_range.requested_start,
        nested[0].byte_range.requested_end,
        nested[0].byte_range.actual_start,
        nested[0].byte_range.actual_end,
        nested[0].byte_range.marked,
        nested[0].byte_range.actual_empty,
    ) == (1, 3, 1, 3, False, False)


def test_ref_exec_fixed_tail_preserves_fully_retained_component_view() -> None:
    cancelled = threading.Event()
    source = mod.RefExecLine("abc", 7, 0, 0, False)
    viewed = dataclasses.replace(
        source,
        component_view=mod._ref_exec_line_component_view(source, cancelled),
    )

    selected = tuple(mod._ref_exec_tail_bytes(iter((viewed,)), 3, cancelled))
    nested = tuple(mod._ref_exec_head_bytes(iter(selected), 3, cancelled))

    assert selected[0].component_view is not None
    assert nested[0].byte_range is not None
    assert (
        nested[0].byte_range.requested_start,
        nested[0].byte_range.requested_end,
        nested[0].byte_range.actual_start,
        nested[0].byte_range.actual_end,
        nested[0].byte_range.marked,
        nested[0].byte_range.actual_empty,
    ) == (1, 3, 1, 3, True, False)


def test_ref_exec_fixed_tail_trailing_plain_empty_preserves_grep_coordinates() -> None:
    cancelled = threading.Event()
    lines = (
        mod.RefExecLine("bar", 1, 0, 0, True),
        mod.RefExecLine("", 2, 4, 4, False),
    )

    selected = tuple(mod._ref_exec_tail_bytes(iter(lines), 1, cancelled))
    matched = tuple(
        mod._ref_exec_grep(
            iter(selected),
            mod.RefExecStage(command="grep", pattern="", flags=frozenset()),
            cancelled,
        )
    )

    trailing = selected[-1]
    assert trailing.byte_range is not None
    assert (
        trailing.text,
        trailing.number,
        trailing.byte_start,
        trailing.char_start,
        trailing.has_newline,
        trailing.match_start,
        trailing.match_end,
        trailing.display_prefix,
        trailing.metadata_only,
        trailing.atomic_match,
        trailing.component_view,
        (
            trailing.byte_range.requested_start,
            trailing.byte_range.requested_end,
            trailing.byte_range.actual_start,
            trailing.byte_range.actual_end,
            trailing.byte_range.marked,
            trailing.byte_range.actual_empty,
        ),
    ) == ("", 2, 4, 4, False, None, None, "", False, False, None, (4, 4, 4, 4, False, False))
    assert (matched[-1].number, matched[-1].match_start, matched[-1].match_end) == (
        2,
        0,
        0,
    )


def test_ref_exec_fixed_tail_trailing_viewed_empty_stays_viewed() -> None:
    cancelled = threading.Event()
    viewed_empty = mod.RefExecLine(
        "",
        2,
        4,
        4,
        False,
        component_view=mod.RefExecComponentView(()),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(
            iter((mod.RefExecLine("bar", 1, 0, 0, True), viewed_empty)),
            1,
            cancelled,
        )
    )
    matched = tuple(
        mod._ref_exec_grep(
            iter(selected),
            mod.RefExecStage(command="grep", pattern="", flags=frozenset()),
            cancelled,
        )
    )

    trailing = selected[-1]
    assert trailing.component_view == mod.RefExecComponentView(())
    assert (trailing.number, trailing.byte_start, trailing.char_start) == (2, 4, 4)
    assert (matched[-1].number, matched[-1].match_start, matched[-1].match_end) == (
        2,
        None,
        None,
    )


def test_ref_exec_fixed_tail_keeps_snapped_empty_before_mixed_trailing_records() -> None:
    cancelled = threading.Event()
    viewed_empty = mod.RefExecLine(
        "",
        51,
        92,
        701,
        False,
        component_view=mod.RefExecComponentView(()),
    )
    lines = (
        mod.RefExecLine(
            "é",
            44,
            90,
            700,
            False,
            0,
            1,
            atomic_match=True,
        ),
        mod.RefExecLine("", 50, 92, 701, False),
        viewed_empty,
    )

    selected = tuple(mod._ref_exec_tail_bytes(iter(lines), 1, cancelled))
    matched = tuple(
        mod._ref_exec_grep(
            iter(selected),
            mod.RefExecStage(command="grep", pattern="", flags=frozenset()),
            cancelled,
        )
    )

    assert [
        (
            line.number,
            line.byte_start,
            line.char_start,
            line.metadata_only,
            line.atomic_match,
            line.component_view is None,
        )
        for line in selected
    ] == [
        (44, 2, 700, True, False, False),
        (50, 2, 701, False, False, True),
        (51, 2, 701, False, False, False),
    ]
    assert [
        (line.number, line.match_start, line.match_end) for line in matched
    ] == [(50, 0, 0), (51, None, None)]


def test_ref_exec_fixed_tail_repeated_multibyte_trim_preserves_fallback_char_start() -> None:
    cancelled = threading.Event()
    source = "界\na界\né\né\nb\n"

    selected = tuple(
        mod._ref_exec_tail_bytes(
            mod._iter_ref_text_lines(source, cancelled),
            10,
            cancelled,
        )
    )

    first = selected[0]
    assert first.byte_range is not None
    assert first.component_view is not None
    assert (
        mod._ref_exec_materialize_line(first, cancelled),
        first.number,
        first.byte_start,
        first.char_start,
        first.has_newline,
        first.match_start,
        first.match_end,
        first.metadata_only,
        first.atomic_match,
        (
            first.byte_range.requested_start,
            first.byte_range.requested_end,
            first.byte_range.actual_start,
            first.byte_range.actual_end,
            first.byte_range.marked,
            first.byte_range.actual_empty,
        ),
        tuple(
            (
                component.kind,
                component.text[component.start : component.end],
                component.utf8_bytes,
                component.source_byte_start,
                component.source_char_start,
            )
            for component in first.component_view.components
        ),
    ) == (
        "\n",
        2,
        8,
        3,
        False,
        None,
        None,
        False,
        False,
        (8, 17, 9, 17, True, False),
        (("synthetic_lf", "\n", 1, 8, 4),),
    )


@pytest.mark.parametrize(
    ("source", "count", "expected_byte_start", "expected_char_start"),
    (
        pytest.param("x\nax\nq\nb\n", 5, 4, 2, id="ascii-control"),
        pytest.param("é\naé\nq\nb\n", 5, 6, 3, id="two-byte-history"),
        pytest.param("界\na界\nq\nb\n", 6, 8, 3, id="three-byte-history"),
        pytest.param("😀\na😀\nq\nb\n", 7, 10, 3, id="four-byte-history"),
    ),
)
def test_ref_exec_fixed_tail_repeated_trim_history_tracks_utf8_boundaries(
    source: str,
    count: int,
    expected_byte_start: int,
    expected_char_start: int,
) -> None:
    cancelled = threading.Event()

    selected = tuple(
        mod._ref_exec_tail_bytes(
            mod._iter_ref_text_lines(source, cancelled),
            count,
            cancelled,
        )
    )

    first = selected[0]
    assert (
        mod._ref_exec_materialize_line(first, cancelled),
        first.number,
        first.byte_start,
        first.char_start,
        first.has_newline,
    ) == ("\n", 2, expected_byte_start, expected_char_start, False)


def test_ref_exec_fixed_tail_later_overwrite_uses_retained_source_text_char_start() -> None:
    cancelled = threading.Event()
    source = mod.RefExecLine("abc", 7, 0, 0, True, display_prefix="1:")
    viewed = dataclasses.replace(
        source,
        component_view=mod._ref_exec_line_component_view(source, cancelled),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(
            iter((viewed, mod.RefExecLine("z", 8, 6, 4, False))),
            6,
            cancelled,
        )
    )

    first = selected[0]
    assert first.byte_range is not None
    assert first.component_view is not None
    assert (
        mod._ref_exec_materialize_line(first, cancelled),
        first.number,
        first.byte_start,
        first.char_start,
        first.has_newline,
        first.match_start,
        first.match_end,
        first.display_prefix,
        first.metadata_only,
        first.atomic_match,
        first.component_view is None,
        (
            first.byte_range.requested_start,
            first.byte_range.requested_end,
            first.byte_range.actual_start,
            first.byte_range.actual_end,
            first.byte_range.marked,
            first.byte_range.actual_empty,
        ),
        tuple(
            (
                component.kind,
                component.text[component.start : component.end],
                component.utf8_bytes,
                component.source_byte_start,
                component.source_char_start,
            )
            for component in first.component_view.components
        ),
    ) == (
        ":abc\n",
        7,
        1,
        0,
        False,
        None,
        None,
        "",
        False,
        False,
        False,
        (2, 7, 2, 7, False, False),
        (
            ("display_prefix", ":", 1, None, None),
            ("text", "abc", 3, 0, 0),
            ("synthetic_lf", "\n", 1, 3, 3),
        ),
    )


def test_ref_exec_fixed_tail_later_overwrite_uses_unbacked_text_fallback() -> None:
    cancelled = threading.Event()
    viewed = mod.RefExecLine(
        "",
        9,
        0,
        40,
        True,
        component_view=mod.RefExecComponentView(
            (
                mod.RefExecComponent("text", "éx", 0, 2, 3),
                mod.RefExecComponent("synthetic_lf", "\n", 0, 1, 1),
            )
        ),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(
            iter((viewed, mod.RefExecLine("yz", 10, 4, 43, False))),
            4,
            cancelled,
        )
    )

    first = selected[0]
    assert first.component_view is not None
    assert (
        mod._ref_exec_materialize_line(first, cancelled),
        first.byte_start,
        first.char_start,
        tuple(
            (
                component.kind,
                component.text[component.start : component.end],
                component.source_byte_start,
                component.source_char_start,
            )
            for component in first.component_view.components
        ),
    ) == (
        "x\n",
        2,
        40,
        (("text", "x", None, None), ("synthetic_lf", "\n", None, None)),
    )


def test_ref_exec_fixed_tail_later_overwrite_snaps_multibyte_prefix() -> None:
    cancelled = threading.Event()
    source = mod.RefExecLine("abc", 7, 0, 0, True, display_prefix="界:")
    viewed = dataclasses.replace(
        source,
        component_view=mod._ref_exec_line_component_view(source, cancelled),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(
            iter((viewed, mod.RefExecLine("yz", 8, 8, 5, False))),
            8,
            cancelled,
        )
    )

    first = selected[0]
    assert first.byte_range is not None
    assert first.component_view is not None
    assert (
        mod._ref_exec_materialize_line(first, cancelled),
        first.byte_start,
        first.char_start,
        first.byte_range.requested_start,
        first.byte_range.actual_start,
        first.byte_range.marked,
        tuple(
            component.text[component.start : component.end]
            for component in first.component_view.components
        ),
    ) == (":abc\n", 3, 0, 3, 4, True, (":", "abc", "\n"))


def test_ref_exec_fixed_tail_later_overwrite_keeps_source_text_offset() -> None:
    cancelled = threading.Event()
    source = mod.RefExecLine("éx", 7, 10, 20, True)
    viewed = dataclasses.replace(
        source,
        component_view=mod._ref_exec_line_component_view(source, cancelled),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(
            iter((viewed, mod.RefExecLine("yz", 8, 14, 23, False))),
            4,
            cancelled,
        )
    )

    first = selected[0]
    assert first.component_view is not None
    assert (
        mod._ref_exec_materialize_line(first, cancelled),
        first.byte_start,
        first.char_start,
        first.component_view.components[0].source_byte_start,
        first.component_view.components[0].source_char_start,
    ) == ("x\n", 2, 21, 12, 21)


def test_ref_exec_fixed_tail_later_overwrite_keeps_synthetic_only_fallback() -> None:
    cancelled = threading.Event()
    source = mod.RefExecLine("abc", 7, 0, 40, True)
    viewed = dataclasses.replace(
        source,
        component_view=mod._ref_exec_line_component_view(source, cancelled),
    )

    selected = tuple(
        mod._ref_exec_tail_bytes(
            iter((viewed, mod.RefExecLine("xyz", 8, 4, 44, False))),
            4,
            cancelled,
        )
    )

    first = selected[0]
    assert first.component_view is not None
    assert (
        mod._ref_exec_materialize_line(first, cancelled),
        first.byte_start,
        first.char_start,
        first.component_view.components[0].kind,
        first.component_view.components[0].source_char_start,
    ) == ("\n", 3, 40, "synthetic_lf", 43)


def test_ref_exec_fixed_tail_single_oversized_view_keeps_slice_char_start() -> None:
    cancelled = threading.Event()
    source = mod.RefExecLine("abc", 7, 0, 0, True, display_prefix="1:")
    viewed = dataclasses.replace(
        source,
        component_view=mod._ref_exec_line_component_view(source, cancelled),
    )

    selected = tuple(mod._ref_exec_tail_bytes(iter((viewed,)), 5, cancelled))

    assert selected[0].component_view is not None
    assert (
        mod._ref_exec_materialize_line(selected[0], cancelled),
        selected[0].byte_start,
        selected[0].char_start,
    ) == (":abc\n", 1, 0)


@pytest.mark.asyncio
async def test_ref_exec_transformed_byte_budget_uses_stage_relative_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ByteEncoder:
        def encode(self, text: str, **_kwargs: object) -> list[int]:
            return list(text.encode())

    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        ("miss\nhit" + "x" * 300,),
        threshold_tokens=180,
        encoder=ByteEncoder(),
    )

    result = await _read(reader, f"grep -n hit {refs[0]} | tail -c 999")
    visible, _, marker = result.partition("\n<auto_compact_ref_range>")
    payload = marker.split("</auto_compact_ref_range>", 1)[0]

    assert visible == "2:hit" + "x" * 56
    assert json.loads(payload) == {
        "actual": "1-61",
        "next": "wc|grep|head|tail|sed",
        "requested": "1-305",
    }


@pytest.mark.asyncio
async def test_ref_exec_snapped_empty_metadata_is_not_a_pipeline_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("A\n界",))
    ref = refs[0]

    tail_result = await _read(reader, f"head -c 3 {ref} | tail -1")
    visible, _, marker = tail_result.partition("\n<auto_compact_ref_range>")
    payload = marker.split("</auto_compact_ref_range>", 1)[0]

    assert visible == "A"
    assert json.loads(payload) == {"actual": "1-2", "requested": "1-3"}
    assert await _read(reader, f"head -c 3 {ref} | wc -l") == "1"


@pytest.mark.asyncio
async def test_ref_exec_response_budget_marks_requested_and_actual_byte_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ByteEncoder:
        def encode(self, text: str, **_kwargs: object) -> list[int]:
            return list(text.encode())

    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        ("0123456789" * 100,),
        threshold_tokens=180,
        encoder=ByteEncoder(),
    )

    result = await _read(reader, f"head -c 500 {refs[0]}")
    payload = result.split("<auto_compact_ref_range>", 1)[1].split(
        "</auto_compact_ref_range>", 1
    )[0]
    ranges = json.loads(payload)

    assert ranges["requested"] == "1-500"
    assert ranges["actual"].startswith("1-")
    assert ranges["actual"] != ranges["requested"]
    assert ranges["next"] == "wc|grep|head|tail|sed"
    assert len(result.encode()) <= mod.REF_EXEC_RESPONSE_MAX_BYTES
    assert len(ByteEncoder().encode(result)) < 180


@pytest.mark.asyncio
async def test_ref_exec_direct_source_continuations_reconstruct_utf8_bytes_losslessly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ByteEncoder:
        def encode(self, text: str, **_kwargs: object) -> list[int]:
            return list(text.encode())

    # Given a UTF-8 source large enough to require several reader pages.
    source = ("α界\nbeta\n" * 60) + "終"
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (source,),
        threshold_tokens=350,
        encoder=ByteEncoder(),
    )
    ref = refs[0]
    command = f"cat {ref}"
    reconstructed = bytearray()
    commands = [command]

    # When every paste-ready continuation is executed through the reader.
    while True:
        result = await _read(reader, command)
        if "\n<auto_compact_ref_range>" in result:
            visible, _, encoded_marker = result.partition(
                "\n<auto_compact_ref_range>"
            )
            marker = json.loads(
                encoded_marker.removesuffix("</auto_compact_ref_range>")
            )
        elif "\n<auto_compact_ref_truncated>" in result:
            visible, _, encoded_marker = result.partition(
                "\n<auto_compact_ref_truncated>"
            )
            marker = json.loads(
                encoded_marker.removesuffix("</auto_compact_ref_truncated>")
            )
        else:
            reconstructed.extend(result.encode())
            break
        visible_bytes = visible.encode()
        reconstructed.extend(visible_bytes)
        command = marker["next"]
        assert command == f"tail -c +{len(reconstructed) + 1} {ref}"
        commands.append(command)

    # Then cat and direct tail-from pages cover every source byte exactly once.
    assert len(commands) > 2
    assert bytes(reconstructed) == source.encode()


@pytest.mark.asyncio
async def test_ref_exec_item7_minified_json_continuations_reconstruct_every_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a realistic single-line minified JSON source that requires several pages.
    source = _item7_minified_json_source()
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (source,),
        threshold_tokens=65_536,
        encoder=Item7ByteEncoder(),
        suffix="item7-continuation",
    )
    ref = refs[0]
    command = f"cat {ref}"
    reconstructed = bytearray()
    offsets: list[int] = []

    # When every returned continuation is executed through the registered reader.
    while True:
        visible, next_command = _item7_visible_and_next(await _read(reader, command))
        reconstructed.extend(visible.encode())
        if next_command is None:
            break
        match = re.fullmatch(rf"tail -c \+(\d+) {re.escape(ref)}", next_command)
        assert match is not None
        offset = int(match.group(1))
        assert offset == len(reconstructed) + 1
        assert not offsets or offset > offsets[-1]
        offsets.append(offset)
        command = next_command

    # Then advancing offsets reconstruct the source without gaps or duplicates.
    assert len(offsets) >= 2
    assert bytes(reconstructed) == source.encode()


@pytest.mark.asyncio
async def test_ref_exec_item7_many_match_grep_preserves_atomic_records_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given one realistic source line containing many distinct grep matches.
    source = _item7_minified_json_source()
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (source,),
        threshold_tokens=65_536,
        encoder=Item7ByteEncoder(),
        suffix="item7-grep",
    )
    ref = refs[0]
    pattern = '"match":"hit-[0-9]{4}"'
    expected = [f'"match":"hit-{index:04d}"' for index in range(1_400)]

    # When normal, numbered, and count forms execute over the complete source.
    normal = await _read(reader, f"grep -Eo '{pattern}' {ref}")
    numbered = await _read(reader, f"grep -Eno '{pattern}' {ref}")
    count = await _read(reader, f"grep -Ec '{pattern}' {ref}")

    # Then matches remain complete atomic records and flags keep grep semantics.
    assert normal.splitlines() == expected
    assert numbered.splitlines() == [f"1:{match}" for match in expected]
    assert count == "1"


@pytest.mark.asyncio
async def test_ref_exec_item7_many_match_grep_budget_stops_between_atomic_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given the realistic many-match source under a constrained response budget.
    source = _item7_minified_json_source()
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (source,),
        threshold_tokens=500,
        encoder=Item7ByteEncoder(),
        suffix="item7-grep-budget",
    )
    pattern = '"match":"hit-[0-9]{4}"'

    # When only-matching grep reaches the response budget.
    visible, next_command = _item7_visible_and_next(
        await _read(reader, f"grep -Eo '{pattern}' {refs[0]}")
    )

    # Then output stops between complete matches and retains generic guidance.
    assert visible
    assert all(re.fullmatch(pattern, match) for match in visible.splitlines())
    assert len(visible.splitlines()) < 1_400
    assert next_command == "wc|grep|head|tail|sed"


@pytest.mark.parametrize(
    "command_template",
    (
        "head -c -1 {ref}",
        "head -c +1 {ref}",
        "tail -c -1 {ref}",
        "tail -c +0 {ref}",
    ),
)
@pytest.mark.asyncio
async def test_ref_exec_item7_invalid_byte_operands_reject_with_valid_ref(
    monkeypatch: pytest.MonkeyPatch,
    command_template: str,
) -> None:
    # Given a valid registered ref paired with an unsupported byte operand.
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (_item7_minified_json_source(),),
        suffix="item7-invalid-byte",
    )

    # When the malformed command crosses the real reader boundary.
    result = await _read(reader, command_template.format(ref=refs[0]))

    # Then parsing rejects the operand rather than misclassifying the ref.
    assert result.startswith("Error: usage:")
    assert "-c" in result


@pytest.mark.asyncio
async def test_ref_exec_item7_transformed_byte_head_matches_large_source_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a single-line source larger than the 64 KiB response ceiling.
    source = _item7_minified_json_source()
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (source,),
        threshold_tokens=65_536,
        encoder=Item7ByteEncoder(),
        suffix="item7-byte-head",
    )
    count = 60_000

    # When a positive byte head is applied as a transformed stream stage.
    result = await _read(reader, f"cat {refs[0]} | head -c {count}")

    # Then it is byte-equivalent to the requested source prefix.
    assert result.encode() == source.encode()[:count]


@pytest.mark.asyncio
async def test_ref_exec_item7_transformed_truncation_retains_generic_reader_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a transformed byte stream larger than the configured page budget.
    source = _item7_minified_json_source()
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (source,),
        threshold_tokens=500,
        encoder=Item7ByteEncoder(),
        suffix="item7-generic-hint",
    )
    ref = refs[0]

    # When the transformed stream truncates.
    visible, next_command = _item7_visible_and_next(
        await _read(reader, f"cat {ref} | head -c 100000")
    )

    # Then the hint remains generic rather than claiming direct-source continuity.
    assert visible
    assert next_command == "wc|grep|head|tail|sed"
    assert next_command != f"tail -c +{len(visible.encode()) + 1} {ref}"


@pytest.mark.parametrize(
    "command_template",
    (
        "head -c 100000 {ref}",
        "tail -c 100000 {ref}",
        "cat {ref} | head -n 300",
        "tail -c +1 {ref} | sed -n '1,$p'",
        "cat {ref} | tail -c +1",
    ),
)
@pytest.mark.asyncio
async def test_ref_exec_bounded_or_transformed_truncation_keeps_generic_next(
    monkeypatch: pytest.MonkeyPatch,
    command_template: str,
) -> None:
    class ByteEncoder:
        def encode(self, text: str, **_kwargs: object) -> list[int]:
            return list(text.encode())

    # Given bounded and transformed commands whose output exceeds one page.
    source = "".join(f"line-{index:03d}-界界界\n" for index in range(300))
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (source,),
        threshold_tokens=180,
        encoder=ByteEncoder(),
    )

    # When the registered reader truncates each command.
    result = await _read(reader, command_template.format(ref=refs[0]))
    marker_tag = (
        "auto_compact_ref_range"
        if "<auto_compact_ref_range>" in result
        else "auto_compact_ref_truncated"
    )
    encoded_marker = result.split(f"<{marker_tag}>", 1)[1].split(
        f"</{marker_tag}>", 1
    )[0]

    # Then only the non-paste-ready generic continuation hint is exposed.
    assert json.loads(encoded_marker)["next"] == "wc|grep|head|tail|sed"


@pytest.mark.asyncio
async def test_ref_exec_byte_stages_preserve_multi_stage_pipeline_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        ("zero\nbeta\ngamma",),
    )
    ref = refs[0]

    assert await _read(reader, f"grep -n beta {ref} | head -c 3") == "2:b"
    assert await _read(reader, f"tail -c +6 {ref} | head -n 1") == "beta"
    assert await _read(reader, f"head -c 10 {ref} | tail -c 4 | wc -c") == "4"


@pytest.mark.asyncio
async def test_ref_exec_fixed_tail_preserves_transformed_full_window_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given two no-LF tool entries presented by one shared transformed upstream.
    reader, _, _, _ = await _reader_fixture(monkeypatch, ("alpha", "beta"))
    upstream = "ls tool | grep -n tool"
    presented = await _read(reader, upstream)
    total = len(presented.encode())

    # When head and fixed-tail consume the exact presented byte window.
    headed = await _read(reader, f"{upstream} | head -c {total}")
    tailed = await _read(reader, f"{upstream} | tail -c {total}")
    headed_lines = await _read(reader, f"{upstream} | head -c {total} | wc -l")
    tailed_lines = await _read(reader, f"{upstream} | tail -c {total} | wc -l")

    # Then both byte windows agree and preserve the two record boundaries.
    assert headed.encode() == tailed.encode()
    assert headed_lines == tailed_lines == "2"


@pytest.mark.asyncio
async def test_ref_exec_fixed_tail_matches_ordinary_mixed_lf_full_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given an ordinary source with one LF-terminated and one non-LF record.
    source = "alpha\nbeta"
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (source,))
    ref = refs[0]
    presented = await _read(reader, f"cat {ref}")
    total = len(presented.encode())

    # When direct byte head and fixed-tail consume the full presented window.
    headed = await _read(reader, f"head -c {total} {ref}")
    tailed = await _read(reader, f"tail -c {total} {ref}")

    # Then both reconstruct exactly the bytes presented by cat.
    assert presented.encode() == headed.encode() == tailed.encode()


@pytest.mark.parametrize(
    "source",
    ("", "\n", "\n\n", "\nalpha", "alpha\n\nbeta", "alpha\n", "alpha\nbeta"),
    ids=(
        "empty-source",
        "one-empty-line",
        "two-empty-lines",
        "leading-newline",
        "consecutive-newlines",
        "trailing-newline",
        "no-final-newline",
    ),
)
@pytest.mark.asyncio
async def test_ref_exec_cat_preserves_source_newline_bytes(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (source,))

    assert await _read(reader, f"cat {refs[0]}") == source


@pytest.mark.parametrize(
    ("source", "count", "expected"),
    (
        ("", 1, ""),
        ("\n", 1, "\n"),
        ("\n\n", 2, "\n\n"),
        ("\nalpha", 1, "\n"),
        ("alpha\n\nbeta\n", 3, "alpha\n\nbeta\n"),
        ("alpha\nbeta", 1, "alpha\n"),
        ("alpha\nbeta", 2, "alpha\nbeta"),
    ),
    ids=(
        "empty-source",
        "one-empty-line",
        "two-empty-lines",
        "leading-newline",
        "consecutive-and-trailing-newlines",
        "selected-line-has-newline",
        "selected-line-has-no-newline",
    ),
)
@pytest.mark.asyncio
async def test_ref_exec_head_preserves_selected_line_newline_bytes(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    count: int,
    expected: str,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (source,))

    assert await _read(reader, f"head -n {count} {refs[0]}") == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        ("head -n 1 {ref}", "alpha\n"),
        ("tail -n 1 {ref}", "alpha"),
        ("sed -n '1p' {ref}", "alpha"),
    ),
    ids=("head-preserves-lf", "tail-drops-lf", "sed-drops-lf"),
)
@pytest.mark.asyncio
async def test_ref_exec_reader_trailing_newline_contract_for_lf_terminated_source(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    expected: str,
) -> None:
    # Given one LF-terminated source record selected by each line stage.
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("alpha\n",))

    # When the command runs through the registered public reader.
    result = await _read(reader, command.format(ref=refs[0]))

    # Then its distinct terminal-newline contract remains byte-exact.
    assert result == expected


@pytest.mark.asyncio
async def test_ref_exec_reports_descriptive_usage_and_unknown_command_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, _, _, _ = await _reader_fixture(monkeypatch)

    expected_ref = "tool:<64 hex> or history:accp_<64 hex>"
    usage = await _read(reader, "")
    assert "usage" in usage.lower()
    assert expected_ref in usage
    unknown = await _read(reader, "find")
    assert "cat, grep, head, ls, sed, stat, tail, wc" in unknown
    assert expected_ref in unknown
    assert "tree" not in unknown
    assert "find," not in unknown
    assert "1,024" in await _read(reader, "x" * 1_025)


@pytest.mark.asyncio
async def test_ref_exec_rejects_shell_path_and_forbidden_syntax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
    ref = refs[0]

    commands = (
        f"cat {ref}; wc -l",
        "cat /etc/passwd",
        f"cat {ref} > out",
        f"cat $(printf {ref})",
        f"cat {ref} &",
        f"cat {ref}/*",
    )
    for command in commands:
        response = await _read(reader, command)
        assert response.lower().startswith("error:")
        assert "private-assistant" not in response


@pytest.mark.asyncio
async def test_ref_exec_rejects_malformed_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)

    assert "quote" in (await _read(reader, f"grep 'target {refs[0]}")).lower()
    assert "empty" in (await _read(reader, f"cat {refs[0]} | | wc -l")).lower()


@pytest.mark.asyncio
async def test_ref_exec_grep_supports_combined_flags_double_dash_numbering_and_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("One\none\n-two\nnone",))
    ref = refs[0]

    assert (
        await _read(reader, f"grep -Ein -- '^(one|-two)$' {ref}")
        == "1:One\n2:one\n3:-two"
    )
    assert await _read(reader, f"grep -Eic -- '^(one|-two)$' {ref}") == "3"
    assert await _read(reader, f"grep -Ei -- '^(one|-two)$' {ref} | wc -l") == "3"


def test_ref_exec_grep_only_is_advertised_and_parsed() -> None:
    ref = f"tool:{'a' * 64}"

    stage = mod._parse_ref_exec_command(f"grep -o needle {ref}")[0]
    description = mod.REF_EXEC_TOOL_SPEC["function"]["description"]

    assert stage.flags == frozenset({"o"})
    assert "[-o]" in description


@pytest.mark.asyncio
async def test_ref_exec_grep_only_emits_every_literal_and_regex_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        ("aba aba\nids=12,34",),
    )
    ref = refs[0]

    assert await _read(reader, f"grep -o aba {ref}") == "aba\naba"
    assert await _read(reader, f"grep -Eo '\\d+' {ref}") == "12\n34"


@pytest.mark.asyncio
async def test_ref_exec_grep_only_preserves_record_delimiters_for_byte_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given one source line containing multiple only-matching records.
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("aba aba",))

    # When a downstream byte stage selects through the first record delimiter.
    result = await _read(reader, f"grep -o aba {refs[0]} | head -c 4")

    # Then the transformed stream exposes grep's record delimiter as a byte.
    assert result == "aba\n"


def _item6_component_view(line: mod.RefExecLine) -> object:
    view_type = getattr(mod, "RefExecComponentView", None)
    assert view_type is not None, "item 6 component view is not implemented"
    view = getattr(line, "component_view", None)
    assert isinstance(view, view_type)
    return view


def test_ref_exec_byte_stage_retains_ordered_component_provenance() -> None:
    prefix = "12:"
    text = "A界B"
    source = mod.RefExecLine(
        text=text,
        number=7,
        byte_start=40,
        char_start=20,
        has_newline=True,
        display_prefix=prefix,
    )

    first = list(
        mod._ref_exec_head_bytes((source,), len("12:A界".encode()), threading.Event())
    )[0]
    nested = list(mod._ref_exec_tail_from_bytes((first,), 4, threading.Event()))[0]
    first_view = _item6_component_view(first)
    nested_view = _item6_component_view(nested)

    assert first.text == ""
    assert first.display_prefix == ""
    assert nested.text == ""
    assert tuple(component.text for component in first_view.components) == (prefix, text)
    assert tuple(component.kind for component in first_view.components) == (
        "display_prefix",
        "text",
    )
    assert all(
        component.text is original
        for component, original in zip(first_view.components, (prefix, text))
    )
    assert nested_view.components[0].text is text


@pytest.mark.asyncio
async def test_ref_exec_nested_multibyte_views_preserve_range_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("A界B\nlast",))

    result = await _read(
        reader,
        f"head -c 5 {refs[0]} | tail -c +3 | head -c 2",
    )

    assert result.startswith("B")
    payload = result.split("<auto_compact_ref_range>", 1)[1].split(
        "</auto_compact_ref_range>", 1
    )[0]
    assert json.loads(payload) == {"actual": "5-5", "requested": "1-2"}


@pytest.mark.asyncio
async def test_ref_exec_component_grep_preserves_regex_and_literal_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        ("xfoo\nfoo bar\nfoo\nbar",),
    )
    ref = refs[0]
    viewed = list(
        mod._ref_exec_head_bytes(
            (mod.RefExecLine("xfoo", 1, 0, 0, True),),
            5,
            threading.Event(),
        )
    )[0]
    _item6_component_view(viewed)

    xfoo = await _read(reader, f"head -c 4 {ref} | grep foo")
    numbered = await _read(reader, f"grep -En '^foo$' {ref} | grep '3:foo'")
    anchored = await _read(
        reader,
        f"grep -En '^foo bar$' {ref} | grep -E '^2:foo\\b'",
    )
    word = await _read(reader, f"head -c 8 {ref} | grep -E '\\bfoo\\b'")
    cross_line = await _read(reader, f"grep -E 'foo\\nbar' {ref}")

    assert xfoo == "xfoo"
    assert numbered == "3:foo"
    assert anchored == "2:foo bar"
    assert word == "foo"
    assert cross_line == ""


def test_ref_exec_grep_excludes_synthetic_lf_from_searchable_components() -> None:
    source = mod.RefExecLine(
        text="foo",
        number=1,
        byte_start=0,
        char_start=0,
        has_newline=True,
    )
    viewed = list(mod._ref_exec_head_bytes((source,), 4, threading.Event()))[0]
    _item6_component_view(viewed)
    stage = mod.RefExecStage(command="grep", pattern="foo\n")

    assert list(mod._ref_exec_grep((viewed,), stage, threading.Event())) == []


def test_ref_exec_component_wc_uses_cached_bytes_and_cross_component_word_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = mod.RefExecLine(
        text="foo bar",
        number=1,
        byte_start=0,
        char_start=0,
        has_newline=True,
        display_prefix="1:",
    )
    viewed = list(mod._ref_exec_head_bytes((source,), 10, threading.Event()))[0]
    view = _item6_component_view(viewed)

    def unexpected_measure(*_args: object, **_kwargs: object) -> tuple[int, str]:
        raise AssertionError("wc -c re-encoded a cached view component")

    monkeypatch.setattr(mod, "_measure_ref_text_parts_checked", unexpected_measure)
    byte_count = list(mod._ref_exec_wc((viewed,), "c", threading.Event()))[0].text
    word_count = list(mod._ref_exec_wc((viewed,), "w", threading.Event()))[0].text
    line_count = list(mod._ref_exec_wc((viewed,), "l", threading.Event()))[0].text

    assert sum(component.utf8_bytes for component in view.components) == 10
    assert (byte_count, word_count, line_count) == ("10", "2", "1")


def test_ref_exec_view_excerpt_excludes_synthetic_lf_sentinel() -> None:
    sentinel = "SYNTHETIC-LF-SENTINEL"
    source = "界" * 30_000 + "NEEDLE" + "後" * 30_000
    cancelled = threading.Event()
    viewed = mod.RefExecLine(
        "",
        1,
        0,
        0,
        False,
        component_view=mod.RefExecComponentView(
            (
                mod._ref_exec_component(
                    "text",
                    source,
                    cancelled,
                    source_start=(0, 0),
                ),
                mod._ref_exec_component("synthetic_lf", sentinel, cancelled),
            )
        ),
    )
    _item6_component_view(viewed)

    matched = list(
        mod._ref_exec_grep(
            (viewed,),
            mod.RefExecStage(command="grep", pattern="NEEDLE"),
            cancelled,
        )
    )[0]
    result = mod._ref_exec_grep_excerpt(
        matched,
        threshold_tokens=10_000,
        encoder=None,
        cancelled=cancelled,
    )

    assert "NEEDLE" in result
    assert sentinel not in result
    assert "<auto_compact_ref_excerpt>" in result


@pytest.mark.asyncio
async def test_ref_exec_component_pipeline_is_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, request, key = await _reader_fixture(monkeypatch, ("foo\nbar",))
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    entry = store.bindings[key].catalog[0]
    iterations = 0
    observed_view = False
    original_grep = mod._ref_exec_grep

    class OneShotSource:
        line_count = 2

        def iter_records(self):
            nonlocal iterations
            iterations += 1
            assert iterations == 1
            yield "foo"
            yield "bar"

    store.bindings[key] = dataclasses.replace(
        store.bindings[key],
        catalog=(dataclasses.replace(entry, source=OneShotSource()),),
    )

    def observe_grep(lines, stage, cancelled):
        nonlocal observed_view
        iterator = iter(lines)
        first = next(iterator)
        _item6_component_view(first)
        observed_view = True
        return original_grep(iter((first, *iterator)), stage, cancelled)

    monkeypatch.setattr(mod, "_ref_exec_grep", observe_grep)

    result = await _read(reader, f"head -c 7 {refs[0]} | grep foo | wc -c")

    assert result == "4"
    assert iterations == 1
    assert observed_view is True


def test_ref_exec_literal_only_matching_preserves_global_non_overlap() -> None:
    cancelled = threading.Event()
    line = mod.RefExecLine(
        "",
        1,
        0,
        0,
        False,
        component_view=mod.RefExecComponentView(
            (
                mod._ref_exec_component("display_prefix", "aaaa", cancelled),
                mod._ref_exec_component(
                    "text", "a", cancelled, source_start=(0, 0)
                ),
            )
        ),
    )
    expected = [match.group(0) for match in re.finditer("aa", "aaaaa")]

    matches = list(
        mod._ref_exec_grep(
            (line,),
            mod.RefExecStage(command="grep", flags=frozenset({"o"}), pattern="aa"),
            cancelled,
        )
    )

    assert [match.text for match in matches] == expected == ["aa", "aa"]


def test_ref_exec_ignorecase_crossing_literal_preserves_source_spelling() -> None:
    cancelled = threading.Event()
    line = mod.RefExecLine(
        "",
        1,
        0,
        0,
        False,
        component_view=mod.RefExecComponentView(
            (
                mod._ref_exec_component("display_prefix", "Ne", cancelled),
                mod._ref_exec_component(
                    "text", "EdLe", cancelled, source_start=(0, 0)
                ),
            )
        ),
    )
    stage = mod.RefExecStage(
        command="grep", flags=frozenset({"i", "o"}), pattern="needle"
    )

    matches = list(mod._ref_exec_grep((line,), stage, cancelled))

    assert [match.text for match in matches] == ["NeEdLe"]


def test_ref_exec_literal_crosses_three_ordered_components() -> None:
    cancelled = threading.Event()
    line = mod.RefExecLine(
        "",
        1,
        0,
        0,
        False,
        component_view=mod.RefExecComponentView(
            (
                mod._ref_exec_component("display_prefix", "a", cancelled),
                mod._ref_exec_component("display_prefix", "b", cancelled),
                mod._ref_exec_component(
                    "text", "c", cancelled, source_start=(0, 0)
                ),
            )
        ),
    )

    selected = list(
        mod._ref_exec_grep(
            (line,), mod.RefExecStage(command="grep", pattern="abc"), cancelled
        )
    )
    only_matching = list(
        mod._ref_exec_grep(
            (line,),
            mod.RefExecStage(command="grep", flags=frozenset({"o"}), pattern="abc"),
            cancelled,
        )
    )

    assert len(selected) == 1
    assert mod._ref_exec_materialize_line(
        selected[0], cancelled, include_synthetic_lf=False
    ) == "abc"
    assert [match.text for match in only_matching] == ["abc"]


def test_ref_exec_viewed_regex_retains_source_relative_coordinates() -> None:
    source = "prefixNEEDLEsuffix"
    cancelled = threading.Event()
    viewed = list(
        mod._ref_exec_head_bytes(
            (mod.RefExecLine(source, 1, 100, 50, False),),
            len(source.encode()),
            cancelled,
        )
    )[0]

    matched = list(
        mod._ref_exec_grep(
            (viewed,),
            mod.RefExecStage(command="grep", flags=frozenset({"E"}), pattern="NEEDLE"),
            cancelled,
        )
    )[0]

    assert (matched.match_start, matched.match_end) == (6, 12)
    assert (matched.byte_start, matched.char_start) == (100, 50)


def test_ref_exec_viewed_regex_excerpt_retains_source_coordinates() -> None:
    source = "prefixNEEDLEsuffix"
    cancelled = threading.Event()
    viewed = list(
        mod._ref_exec_head_bytes(
            (mod.RefExecLine(source, 1, 100, 50, False),),
            len(source.encode()),
            cancelled,
        )
    )[0]
    matched = list(
        mod._ref_exec_grep(
            (viewed,),
            mod.RefExecStage(command="grep", flags=frozenset({"E"}), pattern="NEEDLE"),
            cancelled,
        )
    )[0]

    excerpt = mod._ref_exec_grep_excerpt(
        matched,
        threshold_tokens=10_000,
        encoder=None,
        cancelled=cancelled,
    )
    visible, encoded_marker = excerpt.rsplit("\n<auto_compact_ref_excerpt>", 1)
    marker = json.loads(encoded_marker.removesuffix("</auto_compact_ref_excerpt>"))

    assert visible == source
    assert marker["match_byte_start"] == 106
    assert marker["match_byte_end"] == 112
    assert marker["match_char_start"] == 56
    assert marker["match_char_end"] == 62


def test_ref_exec_regex_crossing_presentation_boundary_is_not_source_backed() -> None:
    cancelled = threading.Event()
    line = mod.RefExecLine(
        "",
        1,
        100,
        50,
        False,
        component_view=mod.RefExecComponentView(
            (
                mod._ref_exec_component("display_prefix", "Ne", cancelled),
                mod._ref_exec_component(
                    "text", "EdLe", cancelled, source_start=(100, 50)
                ),
            )
        ),
    )
    stage = mod.RefExecStage(
        command="grep", flags=frozenset({"E", "i"}), pattern="needle"
    )

    matched = list(mod._ref_exec_grep((line,), stage, cancelled))[0]

    assert matched.match_start is None
    assert matched.match_end is None


def _item6_empty_literal_lines(
    cancelled: threading.Event,
) -> tuple[mod.RefExecLine, ...]:
    text_view = list(
        mod._ref_exec_head_bytes(
            (mod.RefExecLine("view", 3, 20, 10, False),),
            4,
            cancelled,
        )
    )[0]
    display_view = list(
        mod._ref_exec_head_bytes(
            (mod.RefExecLine("text", 4, 30, 20, False, display_prefix="1:"),),
            6,
            cancelled,
        )
    )[0]
    return (
        mod.RefExecLine("plain", 1, 0, 0, True),
        mod.RefExecLine("", 2, 6, 6, True),
        text_view,
        display_view,
    )


def test_ref_exec_empty_literal_selects_plain_and_component_records() -> None:
    cancelled = threading.Event()

    selected = list(
        mod._ref_exec_grep(
            _item6_empty_literal_lines(cancelled),
            mod.RefExecStage(command="grep", pattern=""),
            cancelled,
        )
    )

    assert [
        mod._ref_exec_materialize_line(
            line, cancelled, include_synthetic_lf=False
        )
        for line in selected
    ] == ["plain", "", "view", "1:text"]
    assert [(line.match_start, line.match_end) for line in selected] == [
        (0, 0),
        (0, 0),
        (0, 0),
        (None, None),
    ]


def test_ref_exec_empty_literal_count_includes_every_record() -> None:
    cancelled = threading.Event()
    stage = mod.RefExecStage(command="grep", flags=frozenset({"c"}), pattern="")

    counted = list(
        mod._ref_exec_grep(_item6_empty_literal_lines(cancelled), stage, cancelled)
    )

    assert [line.text for line in counted] == ["4"]


def test_ref_exec_empty_literal_only_matching_suppresses_zero_width() -> None:
    cancelled = threading.Event()
    stage = mod.RefExecStage(command="grep", flags=frozenset({"o"}), pattern="")

    matches = list(
        mod._ref_exec_grep(_item6_empty_literal_lines(cancelled), stage, cancelled)
    )

    assert matches == []


def test_ref_exec_collector_excludes_unmaterialized_synthetic_lf_at_exact_limit() -> None:
    cancelled = threading.Event()
    text = "x" * mod.REF_EXEC_RESPONSE_MAX_BYTES
    line = mod.RefExecLine(
        "",
        1,
        0,
        0,
        False,
        component_view=mod.RefExecComponentView(
            (
                mod._ref_exec_component(
                    "text", text, cancelled, source_start=(0, 0)
                ),
                mod._ref_exec_component(
                    "synthetic_lf",
                    "\n",
                    cancelled,
                    source_start=(len(text), len(text)),
                ),
            )
        ),
    )

    result = mod._collect_ref_exec_response(
        (line,),
        continuation_ref=None,
        final_grep=False,
        preserve_source_newlines=False,
        threshold_tokens=mod.REF_EXEC_RESPONSE_MAX_BYTES + 1,
        encoder=None,
        cancelled=cancelled,
    )

    assert result == text
    assert "<auto_compact_ref_" not in result


def test_ref_exec_collector_counts_preserved_synthetic_lf_at_hard_limit() -> None:
    cancelled = threading.Event()
    text = "x" * mod.REF_EXEC_RESPONSE_MAX_BYTES
    line = mod.RefExecLine(
        "",
        1,
        0,
        0,
        False,
        component_view=mod.RefExecComponentView(
            (
                mod._ref_exec_component(
                    "text", text, cancelled, source_start=(0, 0)
                ),
                mod._ref_exec_component(
                    "synthetic_lf",
                    "\n",
                    cancelled,
                    source_start=(len(text), len(text)),
                ),
            )
        ),
    )

    result = mod._collect_ref_exec_response(
        (line,),
        continuation_ref=None,
        final_grep=False,
        preserve_source_newlines=True,
        threshold_tokens=mod.REF_EXEC_RESPONSE_MAX_BYTES + 1,
        encoder=None,
        cancelled=cancelled,
    )

    assert result != text + "\n"
    assert len(result.encode()) <= mod.REF_EXEC_RESPONSE_MAX_BYTES
    assert "<auto_compact_ref_truncated>" in result


def test_ref_exec_collector_counts_transformed_separator_once_at_exact_limit() -> None:
    cancelled = threading.Event()
    first = mod.RefExecLine(
        "",
        1,
        0,
        0,
        False,
        component_view=mod.RefExecComponentView(
            (mod._ref_exec_component("text", "a", cancelled, source_start=(0, 0)),)
        ),
    )
    second_text = "b" * (mod.REF_EXEC_RESPONSE_MAX_BYTES - 2)
    second = mod.RefExecLine(
        "",
        2,
        1,
        1,
        False,
        component_view=mod.RefExecComponentView(
            (
                mod._ref_exec_component(
                    "text", second_text, cancelled, source_start=(1, 1)
                ),
            )
        ),
    )

    result = mod._collect_ref_exec_response(
        (first, second),
        continuation_ref=None,
        final_grep=False,
        preserve_source_newlines=False,
        threshold_tokens=mod.REF_EXEC_RESPONSE_MAX_BYTES + 1,
        encoder=None,
        cancelled=cancelled,
    )

    assert result == "a\n" + second_text
    assert len(result.encode()) == mod.REF_EXEC_RESPONSE_MAX_BYTES


@pytest.mark.parametrize(
    "case",
    (
        pytest.param(
            ("tail -c +2 {ref}", "task21_item4_f_peak_bytes", None),
            id="f-tail-byte-stage",
        ),
        pytest.param(
            (
                "tail -c +2 {ref} | wc -c",
                "task21_item4_g_peak_bytes",
                20 * 1024 * 1024,
            ),
            id="g-tail-byte-stage-wc",
        ),
    ),
)
@pytest.mark.asyncio
async def test_ref_exec_20mib_tail_byte_stage_stays_below_8mib(
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, str, int | None],
) -> None:
    command_template, peak_label, expected_count = case
    selected_bytes = 20 * 1024 * 1024
    text = "x" + "a" * selected_bytes
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch, (text,), encoder=CountingEncoder()
    )
    command = command_template.format(ref=refs[0])
    gc.collect()

    tracemalloc.start()
    try:
        result = await _read(reader, command)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"{peak_label}={peak_bytes}")
    if expected_count is None:
        assert result and not result.startswith("Error:")
    else:
        assert result == str(expected_count)
    assert peak_bytes < 8 * 1024 * 1024, peak_bytes


@pytest.mark.parametrize(
    ("command_template", "branch"),
    (
        pytest.param(
            "head -c {cut} {ref} | grep NEEDLE",
            "case-sensitive-find-bounds",
            id="case-sensitive-find-bounds",
        ),
        pytest.param(
            "head -c {cut} {ref} | grep -i needle",
            "escaped-ignorecase-pos-endpos",
            id="escaped-ignorecase-pos-endpos",
        ),
        pytest.param(
            "grep -n xfoo {ref} | head -c {cut} | grep '1:xfoo'",
            "component-boundary-window",
            id="component-boundary-window",
        ),
        pytest.param(
            "head -c 10000000 {ref} | grep zz",
            "nonmatching-literal",
            id="h-nonmatching-literal",
        ),
    ),
)
@pytest.mark.asyncio
async def test_ref_exec_20mib_literal_component_paths_stay_below_8mib(
    monkeypatch: pytest.MonkeyPatch,
    command_template: str,
    branch: str,
) -> None:
    cut = 20 * 1024 * 1024
    text = "xfooNEEDLE" + "a" * cut + "tail"
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch, (text,), encoder=CountingEncoder()
    )
    command = command_template.format(cut=cut, ref=refs[0])
    gc.collect()

    tracemalloc.start()
    try:
        result = await _read(reader, command)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"item6_peak_bytes[{branch}]={peak_bytes}")
    if branch == "nonmatching-literal":
        print(f"task21_item4_h_peak_bytes={peak_bytes}")
        assert result == ""
    else:
        assert "NEEDLE" in result or "xfoo" in result, branch
    assert peak_bytes < 8 * 1024 * 1024, (branch, peak_bytes)


@pytest.mark.parametrize(
    ("source", "command_template", "expected"),
    (
        pytest.param(
            "xfoo",
            "tail -c +2 {ref} | grep -E '^foo$'",
            "foo",
            id="tail-anchor",
        ),
        pytest.param(
            "xfoo",
            r"tail -c +2 {ref} | grep -E '\bfoo'",
            "foo",
            id="tail-word-boundary",
        ),
        pytest.param(
            "foo\nbar",
            "grep -n foo {ref} | head -c 5 | grep -E '^1:foo$'",
            "1:foo",
            id="number-head-anchor",
        ),
        pytest.param(
            "foo\nbar",
            r"head -c 4 {ref} | grep -E 'foo\s'",
            "",
            id="head-whitespace-nonmatch",
        ),
    ),
)
@pytest.mark.asyncio
async def test_ref_exec_public_component_view_semantics(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    command_template: str,
    expected: str,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (source,))

    assert await _read(reader, command_template.format(ref=refs[0])) == expected


@pytest.mark.asyncio
async def test_ref_exec_public_grep_number_tail_long_single_line_stays_below_8mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    N = 1_000_000
    text = "x" * (N - len("1:"))
    assert len(f"1:{text}".encode()) == N
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch, (text,), encoder=CountingEncoder()
    )
    ref = refs[0]
    gc.collect()

    tracemalloc.start()
    try:
        result = await _read(reader, f"grep -n x {ref} | tail -c 1000000")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"task21_item2_long_line_peak_bytes={peak_bytes}")
    assert result and not result.startswith("Error:")
    assert peak_bytes < 8 * 1024 * 1024, peak_bytes


@pytest.mark.asyncio
async def test_ref_exec_public_grep_number_tail_many_short_lines_stays_below_8mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    N = 1_000_000
    line_count = 50_000
    payload = "x" + "a" * 15
    text = f"{payload}\n" * line_count
    presented_bytes = sum(
        len(f"{line_number}:{payload}\n".encode())
        for line_number in range(1, line_count + 1)
    )
    assert presented_bytes > N
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch, (text,), encoder=CountingEncoder()
    )
    ref = refs[0]
    gc.collect()

    tracemalloc.start()
    try:
        result = await _read(reader, f"grep -n x {ref} | tail -c 1000000")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"task21_item2_many_short_lines_peak_bytes={peak_bytes}")
    assert result and not result.startswith("Error:")
    assert peak_bytes < 8 * 1024 * 1024, peak_bytes


@pytest.mark.asyncio
async def test_ref_exec_grep_only_skips_empty_matches_and_count_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("bbb\nxx\nxyx",))
    ref = refs[0]

    assert await _read(reader, f"grep -Eo 'x*' {ref}") == "xx\nx\nx"
    assert await _read(reader, f"grep -Eco 'x*' {ref}") == "3"
    assert await _read(reader, f"grep -Eco 'x+' {ref}") == "2"


def test_ref_exec_grep_only_filters_zero_width_spans_from_engine_order() -> None:
    source = mod.RefExecLine("a", 1, 0, 0, False)
    cancelled = threading.Event()

    for pattern in ("a*?", "(?=a)|a"):
        compiled = mod._REGEX.compile(pattern)
        expected = [
            match.group(0)
            for match in compiled.finditer("a")
            if match.end() > match.start()
        ]
        stage = mod.RefExecStage(
            command="grep",
            flags=frozenset({"E", "o"}),
            pattern=pattern,
        )

        actual = [
            line.text
            for line in mod._ref_exec_grep((source,), stage, cancelled)
        ]

        assert expected == ["a"]
        assert actual == expected


def test_ref_exec_grep_only_advances_source_coordinates_after_multibyte_prefix() -> None:
    source = mod.RefExecLine(
        text="界NEEDLE",
        number=9,
        byte_start=10,
        char_start=4,
        has_newline=False,
        display_prefix="7:",
    )
    stage = mod.RefExecStage(
        command="grep",
        flags=frozenset({"E", "n", "o"}),
        pattern="NEEDLE",
    )

    matches = list(mod._ref_exec_grep((source,), stage, threading.Event()))

    assert matches == [
        dataclasses.replace(
            source,
            text="NEEDLE",
            byte_start=13,
            char_start=5,
            display_prefix="1:",
            has_newline=True,
            atomic_match=True,
        )
    ]


@pytest.mark.parametrize("flags", [frozenset({"o"}), frozenset({"E", "o"})])
def test_ref_exec_grep_only_measures_repeated_prefixes_incrementally(
    monkeypatch: pytest.MonkeyPatch,
    flags: frozenset[str],
) -> None:
    repeats = 64
    unit = "界·MATCH"
    source_text = unit * repeats
    source = mod.RefExecLine(source_text, 9, 100, 20, False)
    stage = mod.RefExecStage(command="grep", flags=flags, pattern="MATCH")
    measured_intervals: list[tuple[int, int, int]] = []
    original_range_bytes = mod._ref_exec_utf8_range_bytes

    def record_range_bytes(
        text: str,
        start: int,
        end: int,
        cancelled: threading.Event,
    ) -> int:
        measured_bytes = original_range_bytes(text, start, end, cancelled)
        measured_intervals.append((start, end, measured_bytes))
        return measured_bytes

    monkeypatch.setattr(mod, "_ref_exec_utf8_range_bytes", record_range_bytes)

    matches = list(mod._ref_exec_grep((source,), stage, threading.Event()))

    unit_bytes = len(unit.encode())
    prefix_bytes = len("界·".encode())
    assert [match.text for match in matches] == ["MATCH"] * repeats
    assert [match.byte_start for match in matches] == [
        source.byte_start + index * unit_bytes + prefix_bytes
        for index in range(repeats)
    ]
    fixed_overhead_bytes = 0
    assert sum(interval[2] for interval in measured_intervals) == (
        len(source_text.encode()) + fixed_overhead_bytes
    )


def test_ref_exec_grep_only_rebases_incremental_measurement_by_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = threading.Event()
    source_text = "界MATCHéMATCH"
    full_component = mod._ref_exec_component(
        "text", source_text, cancelled, source_start=(100, 20)
    )
    first_end = len("界MATCH")
    line = mod.RefExecLine(
        "",
        9,
        100,
        20,
        False,
        component_view=mod.RefExecComponentView(
            (
                dataclasses.replace(
                    full_component,
                    end=first_end,
                    utf8_bytes=len("界MATCH".encode()),
                ),
                dataclasses.replace(
                    full_component,
                    start=first_end,
                    utf8_bytes=len("éMATCH".encode()),
                    source_byte_start=108,
                    source_char_start=26,
                ),
            )
        ),
    )
    measured_intervals: list[tuple[int, int]] = []
    original_range_bytes = mod._ref_exec_utf8_range_bytes

    def record_range_bytes(
        text: str,
        start: int,
        end: int,
        event: threading.Event,
    ) -> int:
        measured_intervals.append((start, end))
        return original_range_bytes(text, start, end, event)

    monkeypatch.setattr(mod, "_ref_exec_utf8_range_bytes", record_range_bytes)

    matches = list(
        mod._ref_exec_grep(
            (line,),
            mod.RefExecStage(command="grep", flags=frozenset({"o"}), pattern="MATCH"),
            cancelled,
        )
    )

    assert [match.byte_start for match in matches] == [103, 110]
    assert measured_intervals == [(0, 1), (1, 6), (6, 7), (7, 12)]


@pytest.mark.asyncio
async def test_ref_exec_numbered_grep_only_repeats_each_input_ordinal_per_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("aba aba\nskip\naba",))

    assert await _read(reader, f"grep -no aba {refs[0]}") == "1:aba\n1:aba\n3:aba"


@pytest.mark.asyncio
async def test_ref_exec_grep_only_rejects_an_oversized_first_match_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = "x" * (mod.REF_EXEC_RESPONSE_MAX_BYTES + 1)
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (oversized,))

    result = await _read(reader, f"grep -Eo 'x+' {refs[0]}")

    assert result.startswith("Error:")
    assert "tail -c +N" in result
    assert "byte" in result.lower()
    assert "x" * 100 not in result


@pytest.mark.asyncio
async def test_ref_exec_grep_only_drains_tampered_suffix_after_oversized_first_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = "x" * (mod.REF_EXEC_RESPONSE_MAX_BYTES + 1)
    original = f"{oversized}\ntrusted"
    reader, refs, request, key = await _reader_fixture(monkeypatch, (original,))
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    entry = store.bindings[key].catalog[0]
    store.bindings[key] = dataclasses.replace(
        store.bindings[key],
        catalog=(
            dataclasses.replace(
                entry,
                source=mod.ZeroCopySourceHandle(text=f"{oversized}\nchanged"),
            ),
        ),
    )

    result = await _read(reader, f"grep -Eo 'x+' {refs[0]}")

    assert result == "Error: externalized ref integrity verification failed"


@pytest.mark.asyncio
async def test_ref_exec_grep_only_truncates_between_complete_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = "a" * 40_000
    second = "b" * 40_000
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (f"{first}\n{second}",))

    result = await _read(reader, f"grep -Eo '[ab]+' {refs[0]}")

    visible, marker = result.split("\n<auto_compact_ref_truncated>", 1)
    assert visible == first
    assert second[:100] not in result
    assert json.loads(marker.removesuffix("</auto_compact_ref_truncated>")) == {
        "next": "wc|grep|head|tail|sed"
    }


def test_ref_exec_grep_only_is_lazy_cancelable_and_globally_timeout_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_yields = 0
    search_positions: list[int] = []
    cancelled = threading.Event()

    def source():
        nonlocal source_yields
        source_yields += 1
        yield mod.RefExecLine("aa", 1, 0, 0, False)
        source_yields += 1
        yield mod.RefExecLine("aa", 2, 3, 3, False)

    stage = mod.RefExecStage(
        command="grep",
        flags=frozenset({"E", "o"}),
        pattern="a",
    )
    matches = iter(mod._ref_exec_grep(source(), stage, cancelled))

    assert next(matches).text == "a"
    assert source_yields == 1
    cancelled.set()
    with pytest.raises(mod.RefExecError, match="cancelled"):
        next(matches)

    class Clock:
        elapsed = 0.0

        def monotonic(self) -> float:
            return self.elapsed

    clock = Clock()

    class TimeoutPattern:
        def finditer(self, _text: str, *, timeout: float):
            for position in (0, 1):
                search_positions.append(position)
                clock.elapsed += min(timeout, 1.1)
                if clock.elapsed >= mod.REF_EXEC_REGEX_BUDGET_SECONDS:
                    raise TimeoutError
                yield mod.re.compile("a").search("aa", position)

    monkeypatch.setattr(mod.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(
        mod,
        "_REGEX",
        SimpleNamespace(
            IGNORECASE=2,
            compile=lambda *_args, **_kwargs: TimeoutPattern(),
            error=RuntimeError,
        ),
    )
    with pytest.raises(mod.RefExecError, match="regex timeout"):
        list(mod._ref_exec_grep(source(), stage, threading.Event()))
    assert search_positions == [0, 1]


@pytest.mark.asyncio
async def test_ref_exec_grep_only_preserves_stage_ordinals_and_integrity_draining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, request, key = await _reader_fixture(monkeypatch, ("aa\nskip\na",))
    ref = refs[0]

    assert await _read(reader, f"grep -o a {ref} | grep -n a") == "1:a\n2:a\n3:a"
    assert await _read(reader, f"grep -o a {ref} | sed -n 2p") == "a"
    assert await _read(reader, f"grep -o a {ref} | head -2 | wc -l") == "2"

    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    entry = store.bindings[key].catalog[0]
    store.bindings[key] = dataclasses.replace(
        store.bindings[key],
        catalog=(
            dataclasses.replace(
                entry,
                source=mod.ZeroCopySourceHandle(text="aa\nskip\ntampered"),
            ),
        ),
    )
    assert await _read(reader, f"grep -o a {ref} | head -1") == (
        "Error: externalized ref integrity verification failed"
    )


@pytest.mark.asyncio
async def test_ref_exec_grep_far_offset_excerpt_and_matching_line_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = "界" * 30_000 + "NEEDLE" + "後" * 30_000
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (f"zero\n{line}\nlast",))

    result = await _read(reader, f"grep NEEDLE {refs[0]}")
    visible, encoded_marker = result.rsplit("\n<auto_compact_ref_excerpt>", 1)
    marker = json.loads(encoded_marker.removesuffix("</auto_compact_ref_excerpt>"))
    assert "NEEDLE" in result
    assert marker["line"] == 2
    assert marker["match_byte_start"] == len("zero\n".encode()) + len(
        ("界" * 30_000).encode()
    )
    assert marker["match_byte_end"] == marker["match_byte_start"] + len(
        "NEEDLE".encode()
    )
    assert marker["match_char_start"] == 30_005
    assert marker["match_char_end"] == 30_011
    visible_prefix, visible_suffix = visible.split("NEEDLE", 1)
    assert marker["omitted_prefix_bytes"] + len(visible_prefix.encode()) == len(
        ("界" * 30_000).encode()
    )
    assert marker["omitted_suffix_bytes"] + len(visible_suffix.encode()) == len(
        ("後" * 30_000).encode()
    )
    assert await _read(reader, f"grep NEEDLE {refs[0]} | wc -l") == "1"


@pytest.mark.parametrize(
    "command_template",
    ("head -c 27006 {ref} | grep NEEDLE", "tail -c 27006 {ref} | grep NEEDLE"),
    ids=("head", "tail"),
)
@pytest.mark.asyncio
async def test_ref_exec_byte_range_final_grep_uses_excerpt_when_over_budget(
    monkeypatch: pytest.MonkeyPatch,
    command_template: str,
) -> None:
    line = "x" * 27_000 + "NEEDLE"
    encoder = Item7ByteEncoder()
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (line,),
        threshold_tokens=1_000,
        encoder=encoder,
    )
    assert len(encoder.encode(line)) > 1_000

    result = await _read(reader, command_template.format(ref=refs[0]))

    assert "NEEDLE" in result
    assert "<auto_compact_ref_excerpt>" in result
    assert len(encoder.encode(result)) < 1_000
    assert result != line


@pytest.mark.asyncio
async def test_ref_exec_plain_grep_prior_blank_output_keeps_generic_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "\n" + "x" * 70_000 + "NEEDLE" + "y" * 70_000
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (source,))

    result = await _read(reader, f"grep -E '^$|NEEDLE' {refs[0]}")

    assert result.startswith("\n")
    assert "<auto_compact_ref_truncated>" in result
    assert "<auto_compact_ref_excerpt>" not in result
    _, encoded_marker = result.split("\n<auto_compact_ref_truncated>", 1)
    assert json.loads(
        encoded_marker.removesuffix("</auto_compact_ref_truncated>")
    ) == {"next": "wc|grep|head|tail|sed"}


@pytest.mark.asyncio
async def test_ref_exec_numbered_grep_excerpt_reports_multibyte_match_at_source_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = "界" + "後" * 60_000
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (line,))

    result = await _read(reader, f"grep -n 界 {refs[0]}")
    visible, encoded_marker = result.rsplit("\n<auto_compact_ref_excerpt>", 1)
    marker = json.loads(encoded_marker.removesuffix("</auto_compact_ref_excerpt>"))
    assert visible.startswith("1:")
    visible_source = visible.removeprefix("1:")
    assert marker["match_byte_start"] == 0
    assert marker["match_byte_end"] == len("界".encode())
    assert marker["match_char_start"] == 0
    assert marker["match_char_end"] == 1
    assert marker["omitted_prefix_bytes"] == 0
    assert marker["omitted_suffix_bytes"] == len(line.encode()) - len(
        visible_source.encode()
    )


@pytest.mark.asyncio
async def test_ref_exec_numbered_grep_excerpt_reports_far_multibyte_source_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_prefix = "界" * 30_000
    source_suffix = "後" * 30_000
    line = source_prefix + "針" + source_suffix
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (f"lead\n{line}",))

    result = await _read(reader, f"grep -n 針 {refs[0]}")
    visible, encoded_marker = result.rsplit("\n<auto_compact_ref_excerpt>", 1)
    marker = json.loads(encoded_marker.removesuffix("</auto_compact_ref_excerpt>"))
    assert visible.startswith("2:")
    visible_source = visible.removeprefix("2:")
    visible_prefix, visible_suffix = visible_source.split("針", 1)
    source_line_start_bytes = len("lead\n".encode())
    source_line_start_chars = len("lead\n")
    assert marker["match_byte_start"] == source_line_start_bytes + len(
        source_prefix.encode()
    )
    assert marker["match_byte_end"] == marker["match_byte_start"] + len("針".encode())
    assert marker["match_char_start"] == source_line_start_chars + len(source_prefix)
    assert marker["match_char_end"] == marker["match_char_start"] + 1
    assert marker["omitted_prefix_bytes"] + len(visible_prefix.encode()) == len(
        source_prefix.encode()
    )
    assert marker["omitted_suffix_bytes"] + len(visible_suffix.encode()) == len(
        source_suffix.encode()
    )


@pytest.mark.asyncio
async def test_ref_exec_numbered_grep_pipeline_preserves_original_source_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_prefix = "界" * 30_000
    source_suffix = "後" * 30_000
    line = source_prefix + "針" + source_suffix
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (line,))

    result = await _read(reader, f"grep -n 針 {refs[0]} | grep 針")
    visible, encoded_marker = result.rsplit("\n<auto_compact_ref_excerpt>", 1)
    marker = json.loads(encoded_marker.removesuffix("</auto_compact_ref_excerpt>"))
    assert visible.startswith("1:")
    visible_source = visible.removeprefix("1:")
    visible_prefix, visible_suffix = visible_source.split("針", 1)
    assert marker["match_byte_start"] == len(source_prefix.encode())
    assert marker["match_byte_end"] == marker["match_byte_start"] + len("針".encode())
    assert marker["match_char_start"] == len(source_prefix)
    assert marker["match_char_end"] == marker["match_char_start"] + 1
    assert marker["omitted_prefix_bytes"] + len(visible_prefix.encode()) == len(
        source_prefix.encode()
    )
    assert marker["omitted_suffix_bytes"] + len(visible_suffix.encode()) == len(
        source_suffix.encode()
    )


@pytest.mark.asyncio
async def test_ref_exec_repeated_numbered_grep_uses_current_stream_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("skip\nneedle",))

    assert await _read(reader, f"grep -n needle {refs[0]} | grep -n needle") == (
        "1:2:needle"
    )


@pytest.mark.asyncio
async def test_ref_exec_piped_sed_selects_current_stream_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("skip\nneedle",))

    assert await _read(reader, f"grep needle {refs[0]} | sed -n 1p") == "needle"


@pytest.mark.asyncio
async def test_ref_exec_filtered_numbering_preserves_source_excerpt_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_prefix = "界" * 30_000
    source_suffix = "後" * 30_000
    source_line = source_prefix + "針" + source_suffix
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        (f"skip\n{source_line}",),
    )

    result = await _read(reader, f"grep 針 {refs[0]} | grep -n 針")
    visible, encoded_marker = result.rsplit("\n<auto_compact_ref_excerpt>", 1)
    marker = json.loads(encoded_marker.removesuffix("</auto_compact_ref_excerpt>"))
    assert visible.startswith("1:")
    assert marker["line"] == 2
    assert marker["match_byte_start"] == len("skip\n".encode()) + len(
        source_prefix.encode()
    )
    assert marker["match_byte_end"] == marker["match_byte_start"] + len("針".encode())
    assert marker["match_char_start"] == len("skip\n") + len(source_prefix)
    assert marker["match_char_end"] == marker["match_char_start"] + 1


@pytest.mark.asyncio
async def test_ref_exec_downstream_display_prefix_match_has_no_source_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = "界" * 60_000
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (line,))

    result = await _read(reader, f"grep -n 界 {refs[0]} | grep '1:'")
    assert result.startswith("1:")
    assert "<auto_compact_ref_excerpt>" not in result
    assert "<auto_compact_ref_truncated>" in result


@pytest.mark.asyncio
async def test_ref_exec_grep_quoted_literal_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch, ("alpha beta\nalpha\nbeta",)
    )
    assert await _read(reader, f"grep 'alpha beta' {refs[0]}") == "alpha beta"


@pytest.mark.asyncio
async def test_ref_exec_grep_accepts_explicit_extended_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
    assert await _read(reader, f"grep -E '^(alpha|gamma)$' {refs[0]}") == "alpha\ngamma"


@pytest.mark.asyncio
async def test_ref_exec_grep_auto_detects_kb_exec_regex_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch, ("id=42\nword\nspace here\nfoo\nbar\n|",)
    )
    ref = refs[0]
    assert await _read(reader, f"grep '\\d+' {ref}") == "id=42"
    assert await _read(reader, f"grep 'space.*here' {ref}") == "space here"
    assert await _read(reader, f"grep 'foo|bar' {ref}") == "foo\nbar"
    assert await _read(reader, f"grep '[|]' {ref}") == "|"
    assert await _read(reader, f"grep '|' {ref}") == "id=42\nword\nspace here\nfoo\nbar\n|"


@pytest.mark.asyncio
async def test_ref_exec_grep_treats_unclosed_bracket_as_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("arr[0",))

    assert await _read(reader, f"grep 'arr[0' {refs[0]}") == "arr[0"


@pytest.mark.asyncio
async def test_ref_exec_grep_auto_detects_closed_bracket_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("price U",))

    assert await _read(reader, f"grep 'price [USD]' {refs[0]}") == "price U"


@pytest.mark.asyncio
async def test_ref_exec_grep_normalizes_bre_alternation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
    assert await _read(reader, f"grep 'alpha\\|gamma' {refs[0]}") == "alpha\ngamma"


@pytest.mark.asyncio
async def test_ref_exec_grep_rejects_invalid_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
    result = await _read(reader, f"grep -E '[' {refs[0]}")
    assert result.startswith("Error: invalid regex")
    assert "alpha" not in result


@pytest.mark.asyncio
async def test_ref_exec_grep_reports_regex_timeout_with_reached_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch, ("one\ntwo\nsecret-three\nfour",)
    )
    assert await _read(reader, f"grep -E '^one$' {refs[0]}") == "one"

    class AggregateClock:
        elapsed = 0.0
        search_calls = 0

        def monotonic(self) -> float:
            if self.search_calls >= 3 and self.elapsed < 2.0:
                self.elapsed = 2.001
            return self.elapsed

    clock = AggregateClock()
    timeout_arguments: list[float] = []
    compile_calls = 0

    class SubBudgetPattern:
        def search(self, line: str, *, timeout: float) -> None:
            del line
            timeout_arguments.append(timeout)
            delta = (0.72, 0.72, 0.55)[clock.search_calls]
            assert delta < timeout
            clock.elapsed += delta
            clock.search_calls += 1
            return None

    def compile_pattern(*_args: object, **_kwargs: object) -> SubBudgetPattern:
        nonlocal compile_calls
        compile_calls += 1
        return SubBudgetPattern()

    fake_regex = SimpleNamespace(
        IGNORECASE=2,
        compile=compile_pattern,
        error=RuntimeError,
    )
    monkeypatch.setattr(mod, "_REGEX", fake_regex)
    monkeypatch.setattr(mod.time, "monotonic", clock.monotonic)

    result = await _read(reader, f"grep -E x {refs[0]}")
    assert "timeout" in result.lower()
    assert compile_calls == 1
    assert timeout_arguments == pytest.approx([2.0, 1.28, 0.56])
    assert all(
        earlier > later
        for earlier, later in zip(timeout_arguments, timeout_arguments[1:])
    )
    assert clock.elapsed == pytest.approx(2.001)
    assert "last_completed_line=3" in result
    expected_last_byte = len("one\ntwo\nsecret-three\n".encode())
    assert f"last_completed_byte={expected_last_byte}" in result
    assert "secret-three" not in result
    assert await _read(reader, f"head -1 {refs[0]}") == "one\n"


@pytest.mark.asyncio
async def test_ref_exec_grep_reports_regex_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_REGEX", None)
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
    result = await _read(reader, f"grep -E target {refs[0]}")
    assert result == "Error: regex support is unavailable"


@pytest.mark.asyncio
async def test_ref_exec_supports_bounded_pipelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
    ref = refs[0]
    assert (
        await _read(reader, f"cat {ref} | grep a | sed -n '2,$p' | head -1")
        == "beta target"
    )
    assert await _read(reader, "ls tool | head -1") == ref


@pytest.mark.asyncio
async def test_ref_exec_rejects_invalid_pipeline_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
    ref = refs[0]
    commands = (
        f"head {ref} | ls",
        f"head {ref} | stat {ref}",
        f"head {ref} | grep target {ref}",
        f"head {ref} | cat",
        f"head {ref} | sed -n",
    )
    for command in commands:
        assert (await _read(reader, command)).startswith("Error:")


@pytest.mark.asyncio
async def test_ref_exec_enforces_tail_buffer_and_final_response_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge_lines = ("x" * (4 * 1024 * 1024) + "\n") * 3
    barrier = threading.Barrier(5)

    class ByteEncoder:
        def encode(self, text: str, **_kwargs: object) -> list[int]:
            return list(text.encode())

    class OverlappingTimeoutPattern:
        def search(self, _line: str, *, timeout: float) -> None:
            assert 0 < timeout <= 2.0
            barrier.wait(timeout=5)
            raise TimeoutError

    fake_regex = SimpleNamespace(
        IGNORECASE=2,
        compile=lambda *_args, **_kwargs: OverlappingTimeoutPattern(),
        error=RuntimeError,
    )
    monkeypatch.setattr(mod, "_REGEX", fake_regex)
    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        ("timeout", huge_lines, "a\n" * 70_000),
        threshold_tokens=1_000,
        encoder=ByteEncoder(),
    )

    async def concurrent_contracts() -> tuple[str, str, str]:
        timeout_result = await _read(reader, f"grep -E z {refs[0]}")
        tail_result = await _read(reader, f"tail -c {9 * 1024 * 1024} {refs[1]}")
        capped_result = await _read(reader, f"cat {refs[2]}")
        return timeout_result, tail_result, capped_result

    results = await asyncio.gather(*(concurrent_contracts() for _ in range(5)))
    for timeout_result, tail_result, capped_result in results:
        assert "timeout" in timeout_result.lower()
        assert "busy" not in timeout_result.lower()
        assert "queue" not in timeout_result.lower()
        assert "8 MiB" in tail_result
        assert len(capped_result.encode()) <= 65_536
        assert len(capped_result.encode()) < 1_000
        assert "<auto_compact_ref_truncated>" in capped_result
    assert "8 MiB" in await _read(
        reader, f"cat {refs[1]} | tail -c {9 * 1024 * 1024}"
    )
    assert "8 MiB" not in await _read(reader, f"tail -3 {refs[1]}")
    assert "8 MiB" not in await _read(reader, f"head -c {9 * 1024 * 1024} {refs[1]}")
    assert "8 MiB" not in await _read(reader, f"tail -c +1 {refs[1]}")


@pytest.mark.asyncio
async def test_ref_exec_scans_30mb_source_for_stat_grep_sed_and_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = ("a" * 1023 + "\n") * (29 * 1024) + "needle\n" + "z" * (1024 * 1024)
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (text,))
    ref = refs[0]

    assert f"utf8_bytes={len(text.encode())}" in await _read(reader, f"stat {ref}")
    assert await _read(reader, f"grep needle {ref}") == "needle"
    assert await _read(reader, f"sed -n {29 * 1024 + 1}p {ref}") == "needle"
    tail = await _read(reader, f"tail -1 {ref}")
    assert tail.startswith("z")
    assert len(tail.encode()) <= 65_536


@pytest.mark.asyncio
async def test_ref_exec_giant_wc_streams_without_prescan_or_word_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = ("word " * (6 * 1024 * 1024)) + "last"
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (text,))
    prescans = 0
    original_measure = mod._measure_ref_source_checked

    def observed_measure(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal prescans
        prescans += 1
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(mod, "_measure_ref_source_checked", observed_measure)
    tracemalloc.start()
    try:
        result = await _read(reader, f"wc -w {refs[0]}")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result == str(6 * 1024 * 1024 + 1)
    assert prescans == 0
    assert peak_bytes < 8 * 1024 * 1024


@pytest.mark.asyncio
async def test_ref_exec_giant_regex_streams_match_no_match_timeout_and_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    giant_line = "a" * (30 * 1024 * 1024) + "NEEDLE"
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (giant_line,))
    prescans = 0
    original_measure = mod._measure_ref_source_checked

    def observed_measure(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal prescans
        prescans += 1
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(mod, "_measure_ref_source_checked", observed_measure)
    matched = await _read(reader, f"grep -E 'NEEDLE$' {refs[0]}")
    unmatched = await _read(reader, f"grep -E '^absent$' {refs[0]}")

    entered = threading.Event()
    release = threading.Event()

    class BlockingPattern:
        def search(self, _line: str, *, timeout: float) -> None:
            assert 0 < timeout <= 2.0
            entered.set()
            release.wait(timeout=5)
            raise TimeoutError

    monkeypatch.setattr(
        mod,
        "_REGEX",
        SimpleNamespace(
            IGNORECASE=2,
            compile=lambda *_args, **_kwargs: BlockingPattern(),
            error=RuntimeError,
        ),
    )
    task = asyncio.create_task(_read(reader, f"grep -E x {refs[0]}"))
    while not entered.is_set():
        await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "NEEDLE" in matched
    assert unmatched == ""
    assert prescans == 0


@pytest.mark.asyncio
async def test_ref_exec_streams_expanding_pipeline_without_intermediate_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "\n".join(f"line-{index}" for index in range(50_000))
    giant_line = "界" * (2 * 1024 * 1024)
    reader, refs, _, _ = await _reader_fixture(monkeypatch, (text, giant_line))
    submissions = 0
    source_yields = 0
    downstream_yields = 0
    max_fit_bytes = 0
    cancellation_mode = False
    worker_running = threading.Event()
    cancellation_observed = threading.Event()
    worker_stopped = threading.Event()
    cancellation_progress = 0
    original_to_thread = mod.asyncio.to_thread
    original_source = mod._iter_ref_text_lines
    original_response_fits = mod._ref_exec_response_fits
    original_execute = mod._execute_ref_reader_sync
    original_check_cancelled = mod._check_ref_exec_cancelled
    original_grep = mod._ref_exec_grep

    async def observed_to_thread(
        function: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        nonlocal submissions
        submissions += 1
        return await original_to_thread(function, *args, **kwargs)

    def observed_source(source_text: str, cancelled: threading.Event):
        nonlocal cancellation_progress, source_yields
        if not cancellation_mode:
            for line in original_source(source_text, cancelled):
                source_yields += 1
                yield line
            return
        while True:
            mod._check_ref_exec_cancelled(cancelled)
            cancellation_progress += 1
            yield mod.RefExecLine("line", cancellation_progress, 0, 0, False)

    def observed_response_fits(value: str, **kwargs: object) -> bool:
        nonlocal max_fit_bytes
        max_fit_bytes = max(max_fit_bytes, len(value.encode()))
        return original_response_fits(value, **kwargs)

    def observed_grep(lines, stage, cancelled):
        nonlocal downstream_yields
        for line in original_grep(lines, stage, cancelled):
            if not cancellation_mode:
                downstream_yields += 1
            yield line

    def observed_execute(*args: object, **kwargs: object) -> str:
        if cancellation_mode:
            worker_running.set()
        try:
            return original_execute(*args, **kwargs)
        finally:
            if cancellation_mode:
                worker_stopped.set()

    def observed_check_cancelled(cancelled: threading.Event) -> None:
        if cancelled.is_set():
            cancellation_observed.set()
        original_check_cancelled(cancelled)

    monkeypatch.setattr(mod.asyncio, "to_thread", observed_to_thread)
    monkeypatch.setattr(mod, "_iter_ref_text_lines", observed_source)
    monkeypatch.setattr(mod, "_ref_exec_response_fits", observed_response_fits)
    monkeypatch.setattr(mod, "_execute_ref_reader_sync", observed_execute)
    monkeypatch.setattr(mod, "_check_ref_exec_cancelled", observed_check_cancelled)
    monkeypatch.setattr(mod, "_ref_exec_grep", observed_grep)
    violations: list[str] = []
    early_result = await _read(reader, f"grep -E line {refs[0]} | head -1")
    early_source_yields = source_yields
    first_command_submissions = submissions
    capped = await _read(reader, f"cat {refs[1]}")
    if early_result != "line-0":
        violations.append(f"early result was {early_result!r}")
    if early_source_yields != 50_000:
        violations.append(
            f"source integrity drain consumed {early_source_yields} lines"
        )
    if downstream_yields != 1:
        violations.append(
            f"source integrity drain executed {downstream_yields} downstream matches"
        )
    if first_command_submissions != 1:
        violations.append(
            f"reader command submitted {first_command_submissions} workers"
        )
    if len(capped.encode()) > 65_536:
        violations.append("final response exceeded byte cap")
    if max_fit_bytes > 65_536:
        violations.append(f"response fitting materialized {max_fit_bytes} bytes")

    execute_tree = ast.parse(
        textwrap.dedent(inspect.getsource(mod._execute_ref_reader_sync))
    )
    excerpt_tree = ast.parse(
        textwrap.dedent(inspect.getsource(mod._ref_exec_grep_excerpt))
    )
    grep_tree = ast.parse(textwrap.dedent(inspect.getsource(original_grep)))
    reader_tree = ast.parse(textwrap.dedent(inspect.getsource(mod._new_ref_reader)))
    if any(isinstance(node, ast.ListComp) for node in ast.walk(execute_tree)):
        violations.append("executor contains a cardinality-sized list comprehension")
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"join", "splitlines"}
        for tree in (execute_tree, excerpt_tree)
        for node in ast.walk(tree)
    ):
        violations.append("reader path contains join/splitlines materialization")
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "encode"
        for node in ast.walk(excerpt_tree)
    ):
        violations.append("grep excerpt encodes an unbounded prefix or suffix directly")
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "replace"
        and any(
            keyword.arg == "text"
            and any(
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "line"
                and value.attr == "text"
                for value in ast.walk(keyword.value)
            )
            for keyword in node.keywords
        )
        for node in ast.walk(grep_tree)
    ):
        violations.append(
            "grep concatenates presentation with an unbounded source line"
        )
    if any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "line"
        and node.value.attr == "text"
        and isinstance(node.slice, ast.Slice)
        and node.slice.upper is None
        for node in ast.walk(excerpt_tree)
    ):
        violations.append("grep excerpt copies an unbounded source suffix")
    to_thread_calls = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_thread"
        for node in ast.walk(reader_tree)
    )
    if to_thread_calls != 1:
        violations.append(f"reader contains {to_thread_calls} off-thread submissions")

    cancellation_mode = True
    task = asyncio.create_task(_read(reader, f"cat {refs[0]} | grep -E absent"))
    while not worker_running.is_set():
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    if not cancellation_observed.is_set():
        violations.append("cancellation returned before worker observed the flag")
    if not worker_stopped.is_set():
        violations.append("cancellation returned before worker terminated")
    progress_at_return = cancellation_progress
    await asyncio.sleep(0.01)
    if cancellation_progress != progress_at_return:
        violations.append("worker continued progressing after cancellation returned")
    cancellation_mode = False
    if await _read(reader, f"head -1 {refs[0]}") != "line-0\n":
        violations.append("benign reader recovery failed")
    assert violations == []


@pytest.mark.asyncio
async def test_ref_exec_rejects_tampered_suffix_after_pipeline_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "match\ntrusted-middle\ntrusted-suffix"
    reader, refs, request, key = await _reader_fixture(monkeypatch, (original,))
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    entry = store.bindings[key].catalog[0]
    store.bindings[key] = dataclasses.replace(
        store.bindings[key],
        catalog=(
            dataclasses.replace(
                entry,
                source=mod.ZeroCopySourceHandle(
                    text="match\ntrusted-middle\ntampered-suffix"
                ),
            ),
        ),
    )

    result = await _read(reader, f"grep match {refs[0]} | head -1")

    assert result == "Error: externalized ref integrity verification failed"


@pytest.mark.asyncio
async def test_ref_exec_rejects_tampered_suffix_after_response_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "matching-line\n" * 10_000
    original = prefix + "trusted-suffix"
    reader, refs, request, key = await _reader_fixture(monkeypatch, (original,))
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    entry = store.bindings[key].catalog[0]
    store.bindings[key] = dataclasses.replace(
        store.bindings[key],
        catalog=(
            dataclasses.replace(
                entry,
                source=mod.ZeroCopySourceHandle(text=prefix + "tampered-suffix"),
            ),
        ),
    )

    result = await _read(reader, f"grep matching {refs[0]}")

    assert result == "Error: externalized ref integrity verification failed"


def test_tool_ref_parser_accepts_fixed_length_text_hash_ref() -> None:
    digest = "a" * 64
    parsed = mod.parse_ref(f"tool:{digest}")
    assert parsed == mod.ParsedRef(kind="tool", value=digest)
    rejected = (
        f"tool:{'a' * 63}",
        f"tool:{'a' * 65}",
        f"tool:{'A' * 64}",
        f"tool:{'g' * 64}",
        f"tool:{digest}/extra",
        f"tool:{digest}:extra",
    )
    assert all(mod.parse_ref(value) is None for value in rejected)
    history = "accp_" + "b" * 64
    assert mod.parse_ref(f"history:{history}") == mod.ParsedRef(
        kind="history", value=history
    )


@pytest.mark.asyncio
async def test_ref_exec_truncates_utf8_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("界" * 40_000,))
    result = await _read(reader, f"cat {refs[0]}")
    encoded = result.encode("utf-8")
    assert len(encoded) <= 65_536
    assert encoded.decode("utf-8") == result
    assert "<auto_compact_ref_truncated>" in result


@pytest.mark.asyncio
async def test_ref_exec_reports_descriptive_lookup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, _, _, _ = await _reader_fixture(monkeypatch)
    missing = "tool:" + "0" * 64
    result = await _read(reader, f"cat {missing}")
    assert result == (
        "Error: externalized ref is not available in this binding. "
        "Expected REF: tool:<64 hex> or history:accp_<64 hex>"
    )
    assert "private-assistant" not in result


@pytest.mark.asyncio
async def test_ref_exec_lists_only_binding_local_refs_across_repeated_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, request, key = await _reader_fixture(monkeypatch, ("one", "two"))
    assert (await _read(reader, "ls tool")).splitlines() == list(refs)
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    sibling_key = dataclasses.replace(key, incoming_model_id="sibling")
    sibling_ref = "tool:" + "f" * 64
    store.bindings[sibling_key] = mod.RefBindingState(
        generation=1,
        catalog=(
            mod.RefCatalogEntry(
                manifest=mod.RefManifest(
                    ref=sibling_ref, utf8_bytes=7, sha256="f" * 64
                ),
                source=mod.ZeroCopySourceHandle(text="sibling"),
            ),
        ),
        registry={},
        reader=reader,
    )
    listed = await _read(reader, "ls")
    assert sibling_ref not in listed
    assert "private-assistant" not in listed


@pytest.mark.asyncio
async def test_ref_exec_lists_all_projection_reachable_refs_without_resolving_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch, ("one", "two", "three"))

    def reject_resolution(*_args: object, **_kwargs: object):
        raise AssertionError("ls must not resolve ref content")

    monkeypatch.setattr(mod, "_iter_ref_source_lines", reject_resolution)

    assert (await _read(reader, "ls tool")).splitlines() == list(refs)


@pytest.mark.asyncio
async def test_ref_exec_final_response_cap_tracks_threshold_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ByteEncoder:
        def encode(self, text: str, **_kwargs: object) -> list[int]:
            return list(text.encode())

    reader, refs, _, _ = await _reader_fixture(
        monkeypatch,
        ("x" * 20_000,),
        threshold_tokens=1_000,
        encoder=ByteEncoder(),
    )
    result = await _read(reader, f"cat {refs[0]}")
    assert len(result.encode()) < 1_000
    assert len(ByteEncoder().encode(result, disallowed_special=())) < 1_000
    assert "<auto_compact_ref_truncated>" in result

    failing_encoder = CountingEncoder(fail=True)
    failing_reader, failing_refs, _, _ = await _reader_fixture(
        monkeypatch,
        ("界" * 20_000,),
        threshold_tokens=1_000,
        encoder=failing_encoder,
        suffix="encoder-failure",
    )
    fallback = await _read(failing_reader, f"cat {failing_refs[0]}")
    assert failing_encoder.calls
    assert len(fallback.encode()) <= 65_536
    assert len(fallback.encode()) < 1_000
    assert "<auto_compact_ref_truncated>" in fallback


def _task3_surface() -> tuple[object, object, object, object]:
    loader = getattr(mod, "load_raw_chat_branch", None)
    builder = getattr(mod, "build_canonical_history_source", None)
    source_type = getattr(mod, "CanonicalHistorySourceHandle", None)
    error_type = getattr(mod, "CanonicalHistoryError", None)
    assert callable(loader), "Task 3 raw branch loader is not implemented"
    assert callable(builder), "Task 3 canonical history builder is not implemented"
    assert callable(source_type), "Task 3 canonical history source is not implemented"
    assert callable(error_type), "Task 3 canonical history error is not implemented"
    return loader, builder, source_type, error_type


def _canonical_json(value: dict[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _assert_canonical_sources_equal(
    actual: mod.CanonicalHistorySourceHandle,
    expected: mod.CanonicalHistorySourceHandle,
) -> None:
    assert tuple(actual.iter_records()) == tuple(expected.iter_records())
    assert actual.raw_record_limit == expected.raw_record_limit
    assert actual.line_count == expected.line_count
    assert actual.utf8_bytes == expected.utf8_bytes
    assert actual.raw_source_hash == expected.raw_source_hash


async def _build_known_core_canonical_source(
    raw_messages: list[dict[str, object]],
    *,
    source_message_count: int,
) -> mod.CanonicalHistorySourceHandle:
    try:
        return await mod.build_canonical_history_source(
            raw_messages,
            source_message_count=source_message_count,
        )
    except mod.CanonicalHistoryError as exc:
        pytest.fail(
            "canonical conversion unexpectedly rejected a known Core shape: "
            f"{exc.reason}"
        )


def _grouped_tool_image_output() -> list[dict[str, object]]:
    return [
        {
            "type": "function_call",
            "call_id": "image-call-1",
            "name": "lookup_one",
            "arguments": '{"index":1}',
        },
        {
            "type": "function_call",
            "call_id": "image-call-2",
            "name": "lookup_two",
            "arguments": '{"index":2}',
        },
        {
            "type": "function_call_output",
            "call_id": "image-call-1",
            "output": [
                {"type": "input_text", "text": "first image"},
                {"type": "input_image", "image_url": "private-image-1"},
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "image-call-2",
            "output": [
                {"type": "input_image", "image_url": "private-image-2"},
            ],
        },
    ]


def test_history_jsonl_raw_output_canonical_bytes_remain_stable() -> None:
    raw = (
        {
            "id": "assistant-images",
            "role": "assistant",
            "content": "stale UI content",
            "output": _grouped_tool_image_output(),
        },
    )

    records = tuple(mod._iter_canonical_history_records(raw, 1, None))
    payload = "\n".join(records).encode()

    assert records == (
        _canonical_json(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "image-call-1",
                        "name": "lookup_one",
                        "arguments": '{"index":1}',
                    },
                    {
                        "id": "image-call-2",
                        "name": "lookup_two",
                        "arguments": '{"index":2}',
                    },
                ],
            }
        ),
        _canonical_json(
            {
                "role": "tool",
                "tool_call_id": "image-call-1",
                "content": [
                    {"type": "text", "text": "first image"},
                    {"type": "omitted_media", "media": "image"},
                ],
            }
        ),
        _canonical_json(
            {
                "role": "tool",
                "tool_call_id": "image-call-2",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "omitted_media", "media": "image"},
                ],
            }
        ),
    )
    assert len(payload) == 449
    assert hashlib.sha256(payload).hexdigest() == (
        "9c8ce857b0bbf1b251238236acecc353b97887f1306f5722ca4094b325f31f7b"
    )
    assert len(records) == 3


@pytest.mark.parametrize(
    ("output", "expected_roles"),
    (
        pytest.param(
            [
                {
                    "type": "function_call",
                    "call_id": "ordinary-call",
                    "name": "lookup",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "ordinary-call",
                    "output": [{"type": "input_text", "text": "result"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
            ["assistant", "tool", "assistant"],
            id="ordinary-output-expansion",
        ),
        pytest.param(
            [
                *_grouped_tool_image_output(),
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "after images"}],
                },
            ],
            ["assistant", "tool", "tool", "user", "assistant"],
            id="image-pool-flush-before-non-image",
        ),
        pytest.param(
            _grouped_tool_image_output(),
            ["assistant", "tool", "tool", "user"],
            id="image-pool-flush-at-end",
        ),
        pytest.param(
            [
                {
                    "type": "reasoning",
                    "summary": [{"type": "output_text", "text": "private"}],
                }
            ],
            [],
            id="empty-conversion-clean-message-fallback",
        ),
    ),
)
def test_source_identity_count_matches_real_core_output_conversion(
    output: list[dict[str, object]],
    expected_roles: list[str],
) -> None:
    from open_webui.utils.misc import convert_output_to_messages

    raw_message = {
        "id": "raw-assistant",
        "role": "assistant",
        "content": "clean fallback",
        "output": output,
    }

    converted = convert_output_to_messages(
        output,
        raw=True,
        reasoning_format=None,
        flatten_tool_images=True,
    )

    assert [message["role"] for message in converted] == expected_roles
    assert mod._source_identity_message_count([raw_message]) == (len(converted) or 1)


def test_source_identity_span_calls_pinned_core_converter_without_modern_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_webui.utils import misc as core_misc

    calls: list[tuple[list[dict[str, object]], bool, str | None]] = []

    def pinned_converter(
        output: list[dict[str, object]],
        raw: bool = False,
        reasoning_format: str | None = None,
    ) -> list[dict[str, object]]:
        calls.append((output, raw, reasoning_format))
        return [
            {"role": "assistant", "content": "expanded"},
            {"role": "assistant", "content": "again"},
        ]

    monkeypatch.setattr(core_misc, "convert_output_to_messages", pinned_converter)
    message = {
        "role": "assistant",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "legacy"}],
            }
        ],
    }

    assert (
        mod._source_identity_message_span(
            message,
            transient_message_patterns=None,
        )
        == 2
    )
    assert calls == [(message["output"], True, None)]


def test_source_identity_span_requires_transient_patterns_keyword() -> None:
    parameter = inspect.signature(mod._source_identity_message_span).parameters[
        "transient_message_patterns"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_source_identity_count_excludes_real_core_image_flush_transient_user() -> None:
    from open_webui.utils.misc import convert_output_to_messages

    output = _grouped_tool_image_output()
    converted = convert_output_to_messages(
        output,
        raw=True,
        reasoning_format=None,
        flatten_tool_images=True,
    )
    transient_patterns = (
        re.compile(
            r"Here are the images from the tool results above\. Please analyze them\."
        ),
    )

    assert converted[-1] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Here are the images from the tool results above. Please analyze them.",
            },
            {
                "type": "image_url",
                "image_url": {"url": "private-image-1"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "private-image-2"},
            },
        ],
    }
    assert (
        mod._source_identity_message_count(
            [{"role": "assistant", "output": output}],
            transient_message_patterns=transient_patterns,
        )
        == 3
    )


def test_source_identity_count_returns_zero_when_all_converted_units_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_webui.utils import misc as core_misc

    def excluded_converter(
        _output: list[dict[str, object]],
        *,
        raw: bool,
        reasoning_format: str | None,
        flatten_tool_images: bool,
    ) -> list[dict[str, object]]:
        assert (raw, reasoning_format, flatten_tool_images) == (True, None, True)
        return [
            {"role": "system", "content": "generated system"},
            {"role": "user", "content": "temporary generated user"},
        ]

    monkeypatch.setattr(core_misc, "convert_output_to_messages", excluded_converter)

    assert (
        mod._source_identity_message_count(
            [{"role": "assistant", "output": [{"type": "message"}]}],
            transient_message_patterns=(re.compile(r"temporary generated user"),),
        )
        == 0
    )


def test_source_identity_count_falls_back_for_empty_converter_result_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_webui.utils import misc as core_misc

    def empty_converter(
        _output: list[dict[str, object]],
        *,
        raw: bool,
        reasoning_format: str | None,
        flatten_tool_images: bool,
    ) -> list[dict[str, object]]:
        assert (raw, reasoning_format, flatten_tool_images) == (True, None, True)
        return []

    monkeypatch.setattr(core_misc, "convert_output_to_messages", empty_converter)

    assert (
        mod._source_identity_message_count(
            [{"role": "assistant", "output": [{"type": "reasoning"}]}],
            transient_message_patterns=(),
        )
        == 1
    )


@pytest.mark.asyncio
async def test_canonical_history_source_filters_transient_converted_units() -> None:
    output = _grouped_tool_image_output()
    raw = [
        {"id": "prior-user", "role": "user", "content": "find images"},
        {"id": "assistant-images", "role": "assistant", "output": output},
        {"id": "current-user", "role": "user", "content": "current turn"},
    ]
    transient_patterns = (
        re.compile(
            r"Here are the images from the tool results above\. Please analyze them\."
        ),
    )

    source = await mod.build_canonical_history_source(
        raw,
        source_message_count=4,
        transient_message_patterns=transient_patterns,
    )

    assert source.raw_record_limit == 2


def test_raw_prefix_len_filters_transient_converted_units() -> None:
    output = _grouped_tool_image_output()
    messages = [
        {"role": "assistant", "output": output},
        {"role": "user", "content": "current turn"},
    ]
    transient_patterns = (
        re.compile(
            r"Here are the images from the tool results above\. Please analyze them\."
        ),
    )

    assert (
        mod._raw_prefix_len_for_source_count(
            messages,
            3,
            transient_message_patterns=transient_patterns,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_canonical_history_source_stops_before_current_user_at_core_boundary() -> (
    None
):
    from open_webui.utils.misc import convert_output_to_messages

    output = _grouped_tool_image_output()
    raw = [
        {"id": "prior-user", "role": "user", "content": "find images"},
        {
            "id": "assistant-images",
            "role": "assistant",
            "content": "stale UI content",
            "output": output,
        },
        {"id": "current-user", "role": "user", "content": "current turn"},
    ]
    converted = convert_output_to_messages(
        output,
        raw=True,
        reasoning_format=None,
        flatten_tool_images=True,
    )
    source_message_count = 1 + len(converted)
    expected_records = tuple(
        mod._iter_canonical_history_records(tuple(raw), 2, None)
    )
    expected_bytes = "\n".join(expected_records).encode()

    source = await mod.build_canonical_history_source(
        raw,
        source_message_count=source_message_count,
    )

    assert [message["role"] for message in converted] == [
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert source.raw_record_limit == 2
    assert tuple(source.iter_records()) == expected_records
    assert source.utf8_bytes == len(expected_bytes)
    assert source.raw_source_hash == hashlib.sha256(expected_bytes).hexdigest()
    assert source.line_count == len(expected_records) == 4
    assert all("current turn" not in record for record in source.iter_records())


@pytest.mark.asyncio
async def test_history_jsonl_is_cumulative_allowlisted_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, builder, _, _ = _task3_surface()
    calls: list[str] = []
    raw_messages = {
        "system": {
            "id": "system",
            "role": "system",
            "content": "hidden",
            "parentId": None,
        },
        "user": {
            "id": "user",
            "role": "user",
            "content": "hello",
            "parentId": "system",
            "info": {"ignored": True},
            "statusHistory": [{"ignored": True}],
            "contextSummary": "ignored",
        },
        "assistant": {
            "id": "assistant",
            "role": "assistant",
            "content": "answer",
            "output": [],
            "parentId": "user",
            "usage": {"total_tokens": 1},
            "reasoning_content": "private",
            "sources": [{"private": True}],
        },
        "current-user": {
            "id": "current-user",
            "role": "user",
            "content": "next",
            "parentId": "assistant",
            "files": [{"id": "private-file"}],
        },
    }

    class FakeChats:
        @staticmethod
        async def is_chat_owner(chat_id: str, user_id: str) -> bool:
            raise AssertionError(
                "history loader must not reauthorize admitted requests"
            )

        @staticmethod
        async def get_messages_map_by_chat_id(
            chat_id: str,
        ) -> dict[str, dict[str, object]]:
            calls.append("load")
            assert chat_id == "chat-1"
            return raw_messages

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = FakeChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    branch = await loader(
        chat_id="chat-1",
        metadata={"user_message_id": "current-user"},
    )
    source = await builder(branch, source_message_count=3)
    expected = (
        _canonical_json({"role": "user", "content": "hello"}),
        _canonical_json({"role": "assistant", "content": "answer"}),
        _canonical_json({"role": "user", "content": "next"}),
    )

    assert calls == ["load"]
    assert branch[2]["output"] is raw_messages["assistant"]["output"]
    assert tuple(source.iter_records()) == expected
    assert tuple(source.iter_records()) == expected
    assert source.line_count == 3
    assert "\n".join(source.iter_records()) == "\n".join(expected)
    assert not "\n".join(source.iter_records()).endswith("\n")


@pytest.mark.asyncio
async def test_history_jsonl_maps_expanded_count_only_at_raw_record_end() -> None:
    _, builder, _, _ = _task3_surface()
    raw = [
        {"id": "u", "role": "user", "content": "question"},
        {
            "id": "a",
            "role": "assistant",
            "content": "stale UI text",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": '{"b":2,"a":1}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": [{"type": "input_text", "text": "result"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
        },
        {"id": "u2", "role": "user", "content": "continue"},
    ]

    first = await builder(raw, source_message_count=1)
    expanded = await builder(raw, source_message_count=4)
    complete = await builder(raw, source_message_count=5)

    assert tuple(first.iter_records()) == (
        _canonical_json({"role": "user", "content": "question"}),
    )
    assert [json.loads(line)["role"] for line in expanded.iter_records()] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [json.loads(line)["role"] for line in complete.iter_records()] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]


@pytest.mark.asyncio
async def test_history_jsonl_rejects_inside_record_and_unsaved_sources() -> None:
    _, builder, _, error_type = _task3_surface()
    raw = [
        {"id": "u", "role": "user", "content": "question"},
        {
            "id": "a",
            "role": "assistant",
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
                    "output": [{"type": "input_text", "text": "result"}],
                },
            ],
        },
    ]

    with pytest.raises(error_type, match="inside a raw record"):
        await builder(raw, source_message_count=2)
    with pytest.raises(error_type, match="unsaved source"):
        await builder(raw, source_message_count=4)


@pytest.mark.asyncio
async def test_history_jsonl_accepts_v011_code_interpreter_bookkeeping() -> None:
    from open_webui.utils.misc import convert_output_to_messages

    _, builder, _, _ = _task3_surface()
    output = [
        {
            "type": "open_webui:code_interpreter",
            "id": "code-interpreter-1",
            "status": "completed",
            "code": "print('café')",
            "output": {"stdout": "café\n", "result": "42"},
            "start_tag": "<code_interpreter>",
            "end_tag": "</code_interpreter>",
            "attributes": {"data-language": "python"},
            "lang": "python",
            "started_at": 1_723_000_000,
            "ended_at": 1_723_000_002,
            "duration": 2,
        }
    ]
    stripped_output = [
        {
            "type": "open_webui:code_interpreter",
            "id": "code-interpreter-1",
            "status": "completed",
            "code": "print('café')",
            "output": {"stdout": "café\n", "result": "42"},
        }
    ]
    core_messages = convert_output_to_messages(
        output,
        raw=True,
        reasoning_format=None,
        flatten_tool_images=True,
    )
    expected_content = (
        "<code_interpreter>\nprint('café')\n</code_interpreter>\n"
        "<code_interpreter_output>\ncafé\n\n</code_interpreter_output>"
    )
    raw_message = {
        "id": "assistant-code-interpreter",
        "role": "assistant",
        "content": "stale direct content",
        "output": output,
    }
    stripped_message = {**raw_message, "output": stripped_output}

    assert core_messages == [{"role": "assistant", "content": expected_content}]
    expected = await builder([stripped_message], source_message_count=1)
    actual = await _build_known_core_canonical_source(
        [raw_message], source_message_count=1
    )

    assert tuple(actual.iter_records()) == (
        _canonical_json({"role": "assistant", "content": expected_content}),
    )
    _assert_canonical_sources_equal(actual, expected)


@pytest.mark.asyncio
async def test_history_jsonl_accepts_v011_function_output_bookkeeping() -> None:
    from open_webui.utils.misc import convert_output_to_messages

    _, builder, _, _ = _task3_surface()
    output = [
        {
            "type": "function_call",
            "id": "function-call-1",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query":"café"}',
            "status": "completed",
            "started_at": 1_723_000_000,
        },
        {
            "type": "function_call_output",
            "id": "function-output-1",
            "call_id": "call-1",
            "output": [{"type": "input_text", "text": "résultat"}],
            "status": "completed",
            "files": [{"id": "private-file"}],
            "embeds": [{"id": "private-embed"}],
        },
    ]
    stripped_output = [
        {
            "type": "function_call",
            "id": "function-call-1",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query":"café"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "id": "function-output-1",
            "call_id": "call-1",
            "output": [{"type": "input_text", "text": "résultat"}],
            "status": "completed",
        },
    ]
    core_messages = convert_output_to_messages(
        output,
        raw=True,
        reasoning_format=None,
        flatten_tool_images=True,
    )
    raw_message = {
        "id": "assistant-function-output",
        "role": "assistant",
        "content": "stale direct content",
        "output": output,
    }
    stripped_message = {**raw_message, "output": stripped_output}

    assert core_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "résultat",
    }
    expected = await builder([stripped_message], source_message_count=2)
    actual = await _build_known_core_canonical_source(
        [raw_message], source_message_count=2
    )

    assert tuple(actual.iter_records())[-1] == _canonical_json(
        {"role": "tool", "tool_call_id": "call-1", "content": "résultat"}
    )
    _assert_canonical_sources_equal(actual, expected)


@pytest.mark.asyncio
async def test_history_jsonl_ignores_code_interpreter_stderr() -> None:
    from open_webui.utils.misc import convert_output_to_messages

    _, builder, _, _ = _task3_surface()
    output = [
        {
            "type": "open_webui:code_interpreter",
            "id": "code-interpreter-stderr",
            "status": "completed",
            "code": "raise RuntimeError('boom')",
            "output": {"stderr": "boom"},
        }
    ]
    stripped_output = [
        {
            "type": "open_webui:code_interpreter",
            "id": "code-interpreter-stderr",
            "status": "completed",
            "code": "raise RuntimeError('boom')",
            "output": {},
        }
    ]
    core_messages = convert_output_to_messages(
        output,
        raw=True,
        reasoning_format=None,
        flatten_tool_images=True,
    )
    expected_content = (
        "<code_interpreter>\nraise RuntimeError('boom')\n</code_interpreter>"
    )
    raw_message = {"role": "assistant", "output": output}
    stripped_message = {"role": "assistant", "output": stripped_output}

    assert core_messages == [{"role": "assistant", "content": expected_content}]
    expected = await builder([stripped_message], source_message_count=1)
    actual = await _build_known_core_canonical_source(
        [raw_message], source_message_count=1
    )

    assert tuple(actual.iter_records()) == (
        _canonical_json({"role": "assistant", "content": expected_content}),
    )
    assert "code_interpreter_output" not in next(iter(actual.iter_records()))
    _assert_canonical_sources_equal(actual, expected)


@pytest.mark.asyncio
async def test_history_jsonl_accepts_v011_tagged_message_bookkeeping() -> None:
    from open_webui.utils.misc import convert_output_to_messages

    _, builder, _, _ = _task3_surface()
    output = [
        {
            "type": "message",
            "id": "tagged-message-1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "tagged café"}],
            "_tag_type": "open_webui:message",
            "start_tag": "<message>",
            "end_tag": "</message>",
            "attributes": {"data-origin": "responses"},
            "started_at": 1_723_000_000,
        }
    ]
    stripped_output = [
        {
            "type": "message",
            "id": "tagged-message-1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "tagged café"}],
        }
    ]
    core_messages = convert_output_to_messages(
        output,
        raw=True,
        reasoning_format=None,
        flatten_tool_images=True,
    )
    raw_message = {"role": "assistant", "output": output}
    stripped_message = {"role": "assistant", "output": stripped_output}

    assert core_messages == [{"role": "assistant", "content": "tagged café"}]
    expected = await builder([stripped_message], source_message_count=1)
    actual = await _build_known_core_canonical_source(
        [raw_message], source_message_count=1
    )

    assert tuple(actual.iter_records()) == (
        _canonical_json({"role": "assistant", "content": "tagged café"}),
    )
    _assert_canonical_sources_equal(actual, expected)


@pytest.mark.asyncio
async def test_history_jsonl_matches_core_ordered_tool_text() -> None:
    _, builder, _, _ = _task3_surface()
    arguments = '{ "z": 1, "a": [3, 2] }'
    output = [
        {
            "type": "message",
            "content": [
                {"type": "output_text", "text": "first"},
                {"type": "output_text", "text": "+second"},
            ],
        },
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": arguments,
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": [
                {"type": "input_text", "text": "A"},
                {"type": "input_text", "text": "B"},
            ],
        },
    ]
    raw = [
        {
            "id": "a",
            "role": "assistant",
            "output": output,
        }
    ]

    source = await builder(raw, source_message_count=2)
    records = [json.loads(line) for line in source.iter_records()]

    assert records == [
        {
            "role": "assistant",
            "content": "first+second",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "lookup",
                    "arguments": arguments,
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "AB"},
    ]
    from open_webui.utils.misc import convert_output_to_messages

    core_messages = convert_output_to_messages(output, raw=True, reasoning_format=None)
    assert core_messages[0]["content"] == records[0]["content"]
    assert core_messages[0]["tool_calls"][0]["id"] == records[0]["tool_calls"][0]["id"]
    assert core_messages[0]["tool_calls"][0]["function"] == {
        "name": records[0]["tool_calls"][0]["name"],
        "arguments": records[0]["tool_calls"][0]["arguments"],
    }
    assert core_messages[1] == records[1]

    zero_message_outputs = (
        [
            {
                "type": "function_call",
                "call_id": "orphan-call",
                "name": "lookup",
                "arguments": "{}",
            }
        ],
        [
            {
                "type": "function_call_output",
                "call_id": "orphan-result",
                "output": [{"type": "input_text", "text": "unpaired"}],
            }
        ],
        [
            {
                "type": "reasoning",
                "summary": [{"type": "output_text", "text": "private"}],
            }
        ],
        [{"type": "open_webui:unknown_extension", "content": "private"}],
        [
            {
                "type": "function_call",
                "call_id": "requested",
                "name": "lookup",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "completed",
                "output": [{"type": "input_text", "text": "misaligned"}],
            },
        ],
    )
    for index, zero_message_output in enumerate(zero_message_outputs):
        stale_content = f"stale-direct-content-{index}"
        assert (
            convert_output_to_messages(
                zero_message_output,
                raw=True,
                reasoning_format=None,
                flatten_tool_images=True,
            )
            == []
        )
        zero_source = await builder(
            [
                {
                    "id": f"assistant-{index}",
                    "role": "assistant",
                    "content": stale_content,
                    "output": zero_message_output,
                },
                {
                    "id": f"user-{index}",
                    "role": "user",
                    "content": f"next-{index}",
                },
            ],
            source_message_count=1,
        )
        expected_record = _canonical_json(
            {"role": "assistant", "content": stale_content}
        )
        expected_bytes = expected_record.encode()
        assert tuple(zero_source.iter_records()) == (expected_record,)
        assert zero_source.raw_record_limit == 1
        assert zero_source.line_count == 1
        assert zero_source.utf8_bytes == len(expected_bytes)
        assert zero_source.raw_source_hash == hashlib.sha256(expected_bytes).hexdigest()


@pytest.mark.asyncio
async def test_history_jsonl_fallback_supports_legacy_converter_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_webui.utils import misc as core_misc

    _, builder, _, _ = _task3_surface()
    calls: list[tuple[list[dict[str, object]], bool, str | None]] = []

    def legacy_converter(
        output: list[dict[str, object]],
        raw: bool = False,
        reasoning_format: str | None = None,
    ) -> list[dict[str, object]]:
        calls.append((output, raw, reasoning_format))
        return []

    monkeypatch.setattr(core_misc, "convert_output_to_messages", legacy_converter)
    output = [
        {
            "type": "reasoning",
            "summary": [{"type": "output_text", "text": "private"}],
        }
    ]

    source = await builder(
        [
            {
                "role": "assistant",
                "content": "legacy fallback content",
                "output": output,
            }
        ],
        source_message_count=1,
    )

    assert tuple(source.iter_records()) == (
        _canonical_json(
            {"role": "assistant", "content": "legacy fallback content"}
        ),
    )
    assert calls
    assert all(call == (output, True, None) for call in calls)


@pytest.mark.asyncio
async def test_history_jsonl_omits_reasoning_even_when_core_converts_details() -> None:
    from open_webui.utils.misc import convert_output_to_messages

    _, builder, _, _ = _task3_surface()
    output = [
        {
            "type": "reasoning",
            "id": "reasoning-1",
            "status": "completed",
            "start_tag": "<think>",
            "end_tag": "</think>",
            "attributes": {"type": "reasoning_content"},
            "content": [{"type": "output_text", "text": "private reasoning"}],
            "summary": None,
            "started_at": 1_723_000_000,
            "ended_at": 1_723_000_002,
            "duration": 2,
            "reasoning_details": [
                {"type": "reasoning.text", "text": "private details"}
            ],
        }
    ]
    converted = convert_output_to_messages(
        output,
        raw=True,
        reasoning_format=None,
        flatten_tool_images=True,
    )

    assert converted
    source = await builder(
        [{"role": "assistant", "content": "stale reasoning", "output": output}],
        source_message_count=len(converted),
    )

    assert tuple(source.iter_records()) == ()
    assert source.raw_record_limit == 1
    assert source.line_count == 0
    assert source.utf8_bytes == 0
    assert source.raw_source_hash == hashlib.sha256(b"").hexdigest()


def test_history_jsonl_strict_emitter_success_does_not_call_core_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_webui.utils import misc as core_misc

    calls: list[list[dict[str, object]]] = []

    def spy_converter(
        output: list[dict[str, object]],
        raw: bool = False,
        reasoning_format: str | None = None,
        flatten_tool_images: bool = False,
    ) -> list[dict[str, object]]:
        calls.append(output)
        return [{"role": "assistant", "content": "unexpected fallback"}]

    monkeypatch.setattr(core_misc, "convert_output_to_messages", spy_converter)
    raw_message = {
        "role": "assistant",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "strict success"}],
            }
        ],
    }

    emitted = tuple(mod._iter_canonical_messages_for_raw(raw_message, None))

    assert emitted == ({"role": "assistant", "content": "strict success"},)
    assert calls == []


def test_history_ref_format_remains_canonical_jsonl_v1() -> None:
    assert mod.HISTORY_REF_FORMAT == "canonical-history-jsonl-v1"


@pytest.mark.asyncio
async def test_history_jsonl_omits_known_media_and_rejects_unknown_shapes() -> None:
    _, builder, _, error_type = _task3_surface()
    accepted = [
        {
            "id": "u",
            "role": "user",
            "meta": {"ignored": "message metadata"},
            "content": [
                {"type": "text", "text": "see"},
                {"type": "input_image", "image_url": "data:image/png;base64,private"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://private.test/image"},
                },
            ],
        }
    ]
    source = await builder(accepted, source_message_count=1)

    assert json.loads(next(iter(source.iter_records()))) == {
        "role": "user",
        "content": [
            {"type": "text", "text": "see"},
            {"type": "omitted_media", "media": "image"},
            {"type": "omitted_media", "media": "image"},
        ],
    }
    tool_media = await builder(
        [
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-image",
                        "name": "image_lookup",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call-image",
                        "output": [
                            {"type": "input_text", "text": "caption"},
                            {"type": "input_image", "image_url": "private-url"},
                        ],
                    },
                ],
            }
        ],
        source_message_count=3,
    )
    assert json.loads(tuple(tool_media.iter_records())[1]) == {
        "role": "tool",
        "tool_call_id": "call-image",
        "content": [
            {"type": "text", "text": "caption"},
            {"type": "omitted_media", "media": "image"},
        ],
    }
    rejected_contents = (
        [{"type": "audio", "audio_url": "private"}],
        [{"type": "text", "text": "x", "extra": True}],
        [{"type": "image_url", "image_url": {"url": "x", "detail": "high"}}],
    )
    for content in rejected_contents:
        with pytest.raises(error_type, match="unknown content shape"):
            await builder(
                [{"role": "user", "content": content}], source_message_count=1
            )
    with pytest.raises(error_type, match="unknown message shape"):
        await builder(
            [{"role": "user", "content": "x", "invented": "private"}],
            source_message_count=1,
        )
    for incomplete in (
        {"role": "user"},
        {"role": "tool", "content": "x"},
        {"role": "tool", "tool_call_id": "call"},
    ):
        with pytest.raises(error_type, match="unknown message shape"):
            await builder([incomplete], source_message_count=1)


@pytest.mark.asyncio
async def test_history_jsonl_rejects_unknown_bare_output_item_type() -> None:
    _, builder, _, error_type = _task3_surface()

    with pytest.raises(error_type, match="unknown output shape"):
        await builder(
            [{"role": "assistant", "output": [{"type": "mystery_item"}]}],
            source_message_count=1,
        )


@pytest.mark.asyncio
async def test_history_jsonl_streams_30mb_item_with_stable_incremental_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, builder, _, _ = _task3_surface()
    raw_text = "start-" + "x" * (30 * 1024 * 1024) + "-NEEDLE-end"
    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []
    submissions = 0
    original_to_thread = mod.asyncio.to_thread

    async def observed_to_thread(
        function: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        nonlocal submissions
        submissions += 1

        def observed() -> object:
            worker_threads.append(threading.get_ident())
            return function(*args, **kwargs)

        return await original_to_thread(observed)

    monkeypatch.setattr(mod.asyncio, "to_thread", observed_to_thread)
    heartbeat = 0
    building = asyncio.create_task(
        builder(
            [{"id": "u", "role": "user", "content": raw_text}],
            source_message_count=1,
        )
    )
    while not building.done():
        heartbeat += 1
        await asyncio.sleep(0)
    source = await building
    expected_record = _canonical_json({"role": "user", "content": raw_text})
    expected_hash = hashlib.sha256(expected_record.encode()).hexdigest()
    ref = f"history:accp_{'b' * 64}"
    entry = mod.RefCatalogEntry(
        manifest=mod.RefManifest(
            ref=ref,
            utf8_bytes=len(expected_record.encode()),
            sha256=expected_hash,
        ),
        source=source,
    )
    reader, _, request, key = await _reader_fixture(monkeypatch, ())
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    store.bindings[key] = dataclasses.replace(store.bindings[key], catalog=(entry,))
    submissions_after_build = submissions

    assert heartbeat > 0
    assert submissions_after_build == 1
    assert worker_threads == [worker_threads[0]]
    assert worker_threads[0] != event_loop_thread
    assert source.utf8_bytes == len(expected_record.encode())
    assert source.sha256 == expected_hash
    assert source.raw_source_hash == expected_hash
    assert source.line_count == 1
    assert tuple(source.iter_records()) == (expected_record,)
    assert tuple(source.iter_records()) == (expected_record,)
    assert f"utf8_bytes={source.utf8_bytes}" in await _read(reader, f"stat {ref}")
    literal = await _read(reader, f"grep NEEDLE {ref}")
    regex = await _read(reader, f"grep -E 'NEEDLE' {ref}")
    piped = await _read(reader, f"cat {ref} | grep NEEDLE | grep -E 'NEEDLE'")
    assert "NEEDLE" in literal
    assert "NEEDLE" in regex
    assert "NEEDLE" in piped
    assert submissions == submissions_after_build + 4
    assert len(worker_threads) == submissions
    assert all(thread_id != event_loop_thread for thread_id in worker_threads)

    cancelled = threading.Event()
    worker_done = threading.Event()
    digest_updates = 0
    original_sha256 = hashlib.sha256

    class ObservedDigest:
        def __init__(self) -> None:
            self._digest = original_sha256()

        def update(self, value: bytes) -> None:
            nonlocal digest_updates
            digest_updates += 1
            self._digest.update(value)
            if digest_updates == 1:
                cancelled.set()

        def hexdigest(self) -> str:
            return self._digest.hexdigest()

    async def cancellation_observed_to_thread(
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        def observed() -> object:
            try:
                return function(*args, **kwargs)
            finally:
                worker_done.set()

        return await original_to_thread(observed)

    monkeypatch.setattr(mod.hashlib, "sha256", ObservedDigest)
    monkeypatch.setattr(mod, "threading", SimpleNamespace(Event=lambda: cancelled))
    monkeypatch.setattr(mod.asyncio, "to_thread", cancellation_observed_to_thread)
    cancelled_result = await _read(reader, f"stat {ref}")
    assert cancelled_result == "Error: reader command was cancelled"
    assert worker_done.wait(timeout=5)
    assert digest_updates == 1

    for function in (
        mod._build_canonical_history_source_sync,
        mod._iter_canonical_history_records,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            for node in ast.walk(tree)
        )


@pytest.mark.asyncio
async def test_history_jsonl_hashes_exact_returned_utf8_bytes() -> None:
    _, builder, _, _ = _task3_surface()
    fixtures = (
        'quote: "',
        "slash: \\",
        "controls:\n\t\b",
        "unicode: 界🙂é",
    )
    raw = [
        {"id": f"u-{index}", "role": "user", "content": value}
        for index, value in enumerate(fixtures)
    ]
    source = await builder(raw, source_message_count=len(raw))
    expected_records = tuple(
        _canonical_json({"role": "user", "content": value}) for value in fixtures
    )
    expected_bytes = "\n".join(expected_records).encode("utf-8")

    assert tuple(source.iter_records()) == expected_records
    assert source.utf8_bytes == len(expected_bytes)
    assert source.sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert source.line_count == len(fixtures)


def test_canonical_digest_maps_unpaired_surrogate_and_prioritizes_cancellation() -> (
    None
):
    digest = hashlib.sha256()

    with pytest.raises(mod.CanonicalHistoryError) as failure:
        mod._update_canonical_history_digest(
            digest,
            "bad-\ud800",
            separator=False,
        )

    assert failure.value.reason == "history source is not valid UTF-8"
    assert isinstance(failure.value.__cause__, UnicodeEncodeError)

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(mod.RefExecError, match="cancelled") as cancelled_failure:
        mod._update_canonical_history_digest(
            digest,
            "bad-\ud800",
            separator=False,
            cancelled=cancelled,
        )
    assert cancelled_failure.value.__cause__ is None


@pytest.mark.asyncio
async def test_pipe_propagates_canonical_failure_when_legacy_checkpoint_digest_is_unencodable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_message = {"id": "current-user", "role": "user", "content": "bad-\ud800"}
    checkpoint = {
        "id": f"accp_{'c' * 64}",
        "namespace": mod.CHECKPOINT_NAMESPACE,
        "user_id": "user-1",
        "chat_id": "chat-1",
        "pipe_function_id": "auto_compact",
        "profile_hash": mod.compute_profile_hash(),
        "source_hash": "sha256:" + "d" * 64,
        "source_message_count": 1,
        "state": "ready",
        "summary_text": "legacy checkpoint summary",
        "summary_meta": None,
        "summary_token_count": 3,
        "parent_checkpoint_id": None,
    }
    match = mod.ReusableCheckpointMatch(
        kind="exact",
        source_message_count=1,
        checkpoint=checkpoint,
    )

    forwarded: list[dict[str, object]] = []
    with pytest.raises(
        mod.CanonicalHistoryError,
        match="history source is not valid UTF-8",
    ):
        await _run_pipe_boundary(
            monkeypatch,
            messages=[raw_message],
            function_calling_capability=True,
            metadata_overrides={"user_message_id": "current-user"},
            raw_message_map=_linked_raw_message_map([raw_message]),
            reusable_checkpoint_matches=(match,),
            forwarded_capture=forwarded,
        )
    assert forwarded == []


@pytest.mark.asyncio
async def test_over_128_tool_refs_remain_readable_with_hash_dedup_and_no_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = tuple(f"item-{index}:" + "x" * 65_537 for index in range(129))
    records: list[dict[str, object]] = [
        {"id": "prior-user", "role": "user", "content": "start"}
    ]
    for index, text in enumerate(texts):
        records.extend(
            [
                _raw_native_round(
                    assistant_id=f"assistant-{index}",
                    call_id=f"call-{index}",
                    name="existing",
                    output_parts=[{"type": "input_text", "text": text}],
                ),
                {
                    "id": f"user-{index}",
                    "role": "user",
                    "content": f"continue-{index}",
                },
            ]
        )
    records[-1]["id"] = "current-user"
    duplicate_round = _raw_native_round(
        assistant_id="別-assistant-!@#",
        call_id="別-call-!@#",
        name="existing",
        output_parts=[{"type": "input_text", "text": texts[0]}],
    )
    messages = [
        *_expanded_core_messages(records),
        *_expanded_core_messages([duplicate_round]),
    ]

    result, forwarded, _, registry, request = await _run_pipe_boundary(
        monkeypatch,
        messages=messages,
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(records),
    )

    binding = _committed_ref_binding(request)
    reader = registry[mod.REF_EXEC_TOOL_NAME]["callable"]
    first_ref = f"tool:{hashlib.sha256(texts[0].encode()).hexdigest()}"
    last_ref = f"tool:{hashlib.sha256(texts[-1].encode()).hexdigest()}"
    listed = (await _read(reader, "ls tool")).splitlines()
    forwarded_tool_contents = [
        message["content"]
        for message in forwarded[0]["messages"]
        if message.get("role") == "tool"
    ]
    assert result == {"ok": True}
    assert len(binding.catalog) == 129
    assert len(listed) == 129
    assert len(set(listed)) == 129
    assert forwarded_tool_contents.count(first_ref) == 2
    assert all(
        isinstance(entry.source, mod.ZeroCopySourceHandle)
        for entry in binding.catalog
    )
    assert await _read(reader, f"wc -c {first_ref}") == str(len(texts[0].encode()))
    assert await _read(reader, f"wc -c {last_ref}") == str(len(texts[-1].encode()))


@pytest.mark.asyncio
async def test_imported_tool_text_with_non_ascii_and_symbol_ids_externalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assistant_id = "助手/🙂/$#[]"
    call_id = "呼出/界/!@#$%^&*()"
    raw = "imported:" + "z" * 70_000
    records = [
        {"id": "prior-user", "role": "user", "content": "lookup"},
        _raw_native_round(
            assistant_id=assistant_id,
            call_id=call_id,
            name="existing",
            output_parts=[{"type": "input_text", "text": raw}],
        ),
        {"id": "current-user", "role": "user", "content": "continue"},
    ]

    result, forwarded, _, registry, request = await _run_pipe_boundary(
        monkeypatch,
        messages=_expanded_core_messages(records),
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(records),
    )

    expected_ref = f"tool:{hashlib.sha256(raw.encode()).hexdigest()}"
    source = _committed_ref_binding(request).catalog[0].source
    reader = registry[mod.REF_EXEC_TOOL_NAME]["callable"]
    surfaces = "\n".join(
        (
            expected_ref,
            repr(forwarded[0]["metadata"]["auto_compact_ref_manifests"]),
            await _read(reader, "ls tool"),
            await _read(reader, f"stat {expected_ref}"),
            await _read(reader, "cat tool:" + "0" * 64),
        )
    )
    assert result == {"ok": True}
    assert isinstance(source, mod.ZeroCopySourceHandle)
    assert assistant_id not in surfaces
    assert call_id not in surfaces


@pytest.mark.asyncio
async def test_tool_text_uses_ordered_input_parts_and_exact_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts = (
        "first:",
        "界🙂\nsecond-half",
        "-continued\n",
        ("a" * 1_023 + "\n") * 4_096,
        "needle:",
        "跨part\n",
        ("z" * 1_023 + "\n") * 4_096,
        "last",
    )
    records = [
        {"id": "prior-user", "role": "user", "content": "lookup"},
        _raw_native_round(
            assistant_id="ordered-assistant",
            call_id="ordered-call",
            name="existing",
            output_parts=[{"type": "input_text", "text": part} for part in parts],
        ),
        {"id": "current-user", "role": "user", "content": "continue"},
    ]
    messages = _provider_visible_core_messages(records)
    provider_tool = next(
        message for message in messages if message.get("role") == "tool"
    )
    provider_text = provider_tool["content"]
    assert isinstance(provider_text, str)
    expected_digest = hashlib.sha256()
    expected_utf8_bytes = 0
    for part in parts:
        encoded = part.encode()
        expected_digest.update(encoded)
        expected_utf8_bytes += len(encoded)
    del encoded
    expected_lines = sum(part.count("\n") for part in parts) + 1

    tracemalloc.start()
    try:
        result, forwarded, _, registry, request = await _run_pipe_boundary(
            monkeypatch,
            messages=messages,
            metadata_overrides={"user_message_id": "current-user"},
            raw_message_map=_linked_raw_message_map(records),
        )
        digest = expected_digest.hexdigest()
        expected_ref = f"tool:{digest}"
        binding = _committed_ref_binding(request)
        reader = registry[mod.REF_EXEC_TOOL_NAME]["callable"]
        assert result == {"ok": True}
        assert forwarded[0]["messages"][2]["content"] == expected_ref
        assert binding.catalog[0].manifest == mod.RefManifest(
            ref=expected_ref,
            utf8_bytes=expected_utf8_bytes,
            sha256=digest,
        )
        assert isinstance(binding.catalog[0].source, mod.ZeroCopySourceHandle)
        assert binding.catalog[0].source.text is provider_text
        assert await _read(reader, f"wc -c {expected_ref}") == str(expected_utf8_bytes)
        assert await _read(reader, f"wc -l {expected_ref}") == str(expected_lines)
        assert await _read(reader, f"sed -n '1,2p' {expected_ref}") == (
            "first:界🙂\nsecond-half-continued"
        )
        assert await _read(reader, f"grep 'needle:跨part' {expected_ref}") == (
            "needle:跨part"
        )
        stat = await _read(reader, f"stat {expected_ref}")
        assert f"utf8_bytes={expected_utf8_bytes}" in stat
        assert f"lines={expected_lines}" in stat
        assert f"sha256={digest}" in stat
        assert "auto_compact_ref_truncated" in await _read(
            reader, f"cat {expected_ref}"
        )
        assert not hasattr(mod, "_ordered_core_tool_output_text")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    expected_ref = f"tool:{digest}"
    assert len(provider_text) == sum(len(part) for part in parts)
    assert peak_bytes < expected_utf8_bytes // 2


@pytest.mark.asyncio
async def test_same_request_30mb_text_uses_source_reference_without_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = "r" * (30 * 1024 * 1024)
    current_user = {"id": "current-user", "role": "user", "content": "run"}
    unsaved_round = _raw_native_round(
        assistant_id="unsaved-30mb",
        call_id="unsaved-30mb-call",
        name="existing",
        output_parts=[{"type": "input_text", "text": raw}],
    )

    result, forwarded, _, _, request = await _run_pipe_boundary(
        monkeypatch,
        messages=_expanded_core_messages([current_user, unsaved_round]),
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map([current_user]),
    )

    expected_ref = f"tool:{hashlib.sha256(raw.encode()).hexdigest()}"
    binding = _committed_ref_binding(request)
    source = binding.catalog[0].source
    assert result == {"ok": True}
    assert forwarded[0]["messages"][2]["content"] == expected_ref
    assert raw not in repr(forwarded[0])
    assert isinstance(source, mod.ZeroCopySourceHandle)
    assert source.text is raw


@pytest.mark.asyncio
async def test_ref_projection_preserves_complete_parallel_tool_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = "parallel-first:" + "a" * 70_000
    second = "parallel-second:" + "b" * 70_000
    raw_assistant = {
        "id": "parallel-assistant",
        "role": "assistant",
        "output": [
            {
                "type": "function_call",
                "call_id": "call-a",
                "name": "existing",
                "arguments": '{"value":1}',
            },
            {
                "type": "function_call",
                "call_id": "call-b",
                "name": "existing",
                "arguments": '{"value":2}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-a",
                "output": [{"type": "input_text", "text": first}],
            },
            {
                "type": "function_call_output",
                "call_id": "call-b",
                "output": [{"type": "input_text", "text": second}],
            },
        ],
    }
    records = [
        {"id": "prior-user", "role": "user", "content": "parallel"},
        raw_assistant,
        {"id": "current-user", "role": "user", "content": "continue"},
    ]
    expanded = _expanded_core_messages(records)
    expected_assistant = copy.deepcopy(expanded[1])

    result, forwarded, _, registry, request = await _run_pipe_boundary(
        monkeypatch,
        messages=expanded,
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(records),
    )

    first_ref = f"tool:{hashlib.sha256(first.encode()).hexdigest()}"
    second_ref = f"tool:{hashlib.sha256(second.encode()).hexdigest()}"
    projected = forwarded[0]["messages"]
    binding = _committed_ref_binding(request)
    reader = registry[mod.REF_EXEC_TOOL_NAME]["callable"]
    assert result == {"ok": True}
    assert projected[1] == expected_assistant
    assert [projected[2]["content"], projected[3]["content"]] == [first_ref, second_ref]
    assert projected[4]["content"] == "continue"
    assert all(
        isinstance(entry.source, mod.ZeroCopySourceHandle)
        for entry in binding.catalog
    )
    assert await _read(reader, f"wc -c {first_ref}") == str(len(first.encode()))
    assert await _read(reader, f"wc -c {second_ref}") == str(len(second.encode()))


@pytest.mark.asyncio
async def test_real_core_unrelated_message_rewrites_do_not_break_tool_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_webui.utils import middleware as core_middleware
    from open_webui.utils.misc import add_or_update_user_message

    raw = "core-rewrites:" + "x" * 70_000
    records = [
        {
            "id": "prior-user",
            "role": "user",
            "content": "look up",
            "files": [
                {
                    "type": "file",
                    "url": "https://files.invalid/context.txt",
                    "name": "context.txt",
                },
                {
                    "type": "image",
                    "url": "https://images.invalid/input.png",
                    "name": "input.png",
                },
            ],
        },
        _raw_native_round(
            assistant_id="persisted-assistant",
            call_id="persisted-call",
            name="existing",
            output_parts=[{"type": "input_text", "text": raw}],
        ),
        {
            "id": "current-user",
            "role": "user",
            "content": "<$skill-1|Research> continue",
        },
    ]
    provider_messages = core_middleware.process_messages_with_output(
        [
            {
                key: copy.deepcopy(value)
                for key, value in record.items()
                if key in _PROVIDER_VISIBLE_CORE_FIELDS
            }
            for record in records
        ]
    )

    class FakeChats:
        @staticmethod
        async def get_chat_by_id_and_user_id(
            chat_id: str, user_id: str
        ) -> SimpleNamespace:
            assert (chat_id, user_id) == ("chat-1", "user-1")
            return SimpleNamespace(
                chat={
                    "history": {
                        "messages": _linked_raw_message_map(records),
                        "currentId": "current-user",
                    }
                }
            )

    monkeypatch.setattr(core_middleware, "Chats", FakeChats)
    provider_messages = await core_middleware.add_file_context(
        provider_messages,
        "chat-1",
        SimpleNamespace(id="user-1"),
    )
    provider_messages[-1]["content"] = [
        {"type": "text", "text": provider_messages[-1]["content"]},
        {
            "type": "image_url",
            "image_url": {"url": "https://images.invalid/input.png"},
        },
    ]
    add_or_update_user_message("citation-rag", provider_messages, append=False)
    core_middleware.strip_skill_mentions(provider_messages)

    async def image_base64(image_url: str, user: object = None) -> str:
        assert image_url == "https://images.invalid/input.png"
        return "data:image/png;base64,Y29yZS1yZXdyaXRl"

    monkeypatch.setattr(core_middleware, "get_image_base64_from_url", image_base64)
    provider_messages = (
        await core_middleware.convert_url_images_to_base64(
            {"messages": provider_messages},
            user=SimpleNamespace(id="user-1"),
        )
    )["messages"]

    result, forwarded, _, _, request = await _run_pipe_boundary(
        monkeypatch,
        messages=provider_messages,
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(records),
    )

    expected_ref = f"tool:{hashlib.sha256(raw.encode()).hexdigest()}"
    projected_tool = next(
        message for message in forwarded[0]["messages"] if message.get("role") == "tool"
    )
    rewritten_surface = repr(provider_messages)
    assert "<attached_files>" in rewritten_surface
    assert "citation-rag" in rewritten_surface
    assert "data:image/png;base64,Y29yZS1yZXdyaXRl" in rewritten_surface
    assert "<$skill-1|Research>" not in rewritten_surface
    assert result == {"ok": True}
    assert projected_tool["content"] == expected_ref
    assert raw not in repr(forwarded[0])
    assert (
        isinstance(
            _committed_ref_binding(request).catalog[0].source,
            mod.ZeroCopySourceHandle,
        )
    )


@pytest.mark.asyncio
async def test_death_chat_ninth_search_result_survives_real_core_rag_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_webui.utils.misc import (
        add_or_update_user_message,
        convert_output_to_messages,
    )

    sources = [
        {
            "title": f"Regional climate resilience report {index + 1}",
            "url": f"https://research.example.org/reports/2026/resilience-{index + 1}",
            "published_at": f"2026-07-{index + 1:02d}",
            "snippet": (
                f"Report {index + 1} compares municipal adaptation budgets, flood-risk maps, "
                "public works schedules, and independently audited outcomes across coastal "
                "communities. The authors document methods, limitations, and links to the "
                "underlying public datasets so each finding can be checked against the source. "
            )
            * 8,
        }
        for index in range(34)
    ]
    ninth_result = json.dumps({"sources": sources}, ensure_ascii=False)
    assert len(ninth_result.encode()) > 65_536
    assert len(json.loads(ninth_result)["sources"]) == 34

    output: list[dict[str, object]] = []
    tool_names = (
        "search_news",
        "open_url",
        "search_docs",
        "query_archive",
        "lookup_dataset",
        "inspect_report",
        "search_papers",
        "open_dataset",
        "search_web",
    )
    for index, tool_name in enumerate(tool_names, start=1):
        call_id = f"death-chat-call-{index}"
        result_text = ninth_result if index == 9 else f"completed round {index}"
        output.extend(
            (
                {
                    "type": "function_call",
                    "id": f"death-chat-fc-{index}",
                    "call_id": call_id,
                    "name": tool_name,
                    "arguments": json.dumps({"round": index}),
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "id": f"death-chat-fco-{index}",
                    "call_id": call_id,
                    "output": [{"type": "input_text", "text": result_text}],
                    "status": "completed",
                },
            )
        )

    converted = convert_output_to_messages(
        output,
        raw=True,
        flatten_tool_images=True,
    )
    tenth_turn = "Compare the evidence and cite the strongest sources."
    raw_records = [
        {
            "id": "death-chat-assistant",
            "role": "assistant",
            "content": "stale UI content",
            "output": output,
        },
        {
            "id": "death-chat-user",
            "role": "user",
            "content": tenth_turn,
        },
    ]
    canonical_visible_branch = [
        *converted,
        {"role": "user", "content": tenth_turn},
    ]
    provider_messages = [
        *converted,
        {"role": "user", "content": tenth_turn},
    ]
    rag_content = '<source id="rag-1">Verified citation context</source>'
    add_or_update_user_message(rag_content, provider_messages, append=False)

    assert len(raw_records) == 2
    assert len(converted) == 18
    assert [
        message["tool_call_id"] for message in converted if message["role"] == "tool"
    ] == [f"death-chat-call-{index}" for index in range(1, 10)]
    assert converted[-2]["tool_calls"][0]["function"]["name"] == "search_web"
    assert provider_messages[-1]["content"] == f"{rag_content}\n{tenth_turn}"
    assert not any(
        provider_messages[start : start + len(canonical_visible_branch)]
        == canonical_visible_branch
        for start in range(len(provider_messages) - len(canonical_visible_branch) + 1)
    )

    result, forwarded, _, registry, request = await _run_pipe_boundary(
        monkeypatch,
        messages=provider_messages,
        metadata_overrides={"user_message_id": "death-chat-user"},
        raw_message_map=_linked_raw_message_map(raw_records),
    )

    expected_ref = f"tool:{hashlib.sha256(ninth_result.encode()).hexdigest()}"
    ninth_tool = next(
        message
        for message in forwarded[0]["messages"]
        if message.get("tool_call_id") == "death-chat-call-9"
    )
    binding = _committed_ref_binding(request)
    reader = registry[mod.REF_EXEC_TOOL_NAME]["callable"]
    assert result == {"ok": True}, (result, forwarded)
    assert len(forwarded) == 1
    assert ninth_tool["content"] == expected_ref
    assert ninth_result not in repr(forwarded[0])
    assert len(binding.catalog) == 1
    assert isinstance(binding.catalog[0].source, mod.ZeroCopySourceHandle)
    assert binding.catalog[0].source.text is ninth_result
    assert await _read(reader, f"wc -c {expected_ref}") == str(
        len(ninth_result.encode())
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    (
        pytest.param((False, False), id="missing-below-threshold"),
        pytest.param((True, False), id="empty-below-threshold"),
        pytest.param((False, True), id="missing-eligible"),
        pytest.param((True, True), id="empty-eligible"),
    ),
)
async def test_real_core_missing_or_empty_function_name_externalizes_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    scenario: tuple[bool, bool],
) -> None:
    from open_webui.utils.misc import convert_output_to_messages

    include_empty_name, eligible = scenario
    text = "empty-name:" + ("x" * 70_000 if eligible else "raw")
    function_call: dict[str, CoreFixtureValue] = {
        "type": "function_call",
        "call_id": "empty-name-call",
        "arguments": "{}",
    }
    if include_empty_name:
        function_call["name"] = ""
    core_messages = convert_output_to_messages(
        [
            function_call,
            {
                "type": "function_call_output",
                "call_id": "empty-name-call",
                "output": [{"type": "input_text", "text": text}],
            },
        ],
        raw=True,
        flatten_tool_images=True,
    )
    assert core_messages[0]["tool_calls"][0]["function"]["name"] == ""
    messages = [
        {
            "role": "user",
            "content": "<auto_compaction_context>summary</auto_compaction_context>",
        },
        *core_messages,
    ]
    caplog.set_level("WARNING", logger=mod.__name__)
    monkeypatch.setattr(
        mod,
        "_get_tiktoken_encoder",
        lambda _request=None: (CountingEncoder(count=999), "test"),
    )

    result, forwarded, _, registry, _ = await _run_pipe_boundary(
        monkeypatch,
        messages=messages,
    )

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and record.name == mod.__name__
    ]
    if eligible:
        expected_ref = f"tool:{hashlib.sha256(text.encode()).hexdigest()}"
        forwarded_tool = next(
            message
            for message in forwarded[0]["messages"]
            if message.get("role") == "tool"
        )
        rendered_context = forwarded[0]["messages"][0]
        assert result == {"ok": True}
        assert forwarded_tool["content"] == expected_ref
        assert set(rendered_context) == {"role", "content"}
        assert "<auto_compact_ref_manifests" not in rendered_context["content"]
        assert text not in repr(forwarded[0])
        assert warnings == []
        assert mod.REF_EXEC_TOOL_NAME in registry
    else:
        assert result == {"ok": True}
        assert forwarded[0]["messages"] == messages
        assert warnings == []
        assert mod.REF_EXEC_TOOL_NAME not in registry


@pytest.mark.asyncio
async def test_incomplete_tool_round_externalizes_eligible_result_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted = "control:" + "x" * 70_000
    control_records = [
        {"id": "prior-user", "role": "user", "content": "lookup"},
        _raw_native_round(
            assistant_id="control-assistant",
            call_id="control-call",
            name="existing",
            output_parts=[{"type": "input_text", "text": persisted}],
        ),
        {"id": "current-user", "role": "user", "content": "continue"},
    ]
    control = await _run_pipe_boundary(
        monkeypatch,
        messages=_expanded_core_messages(control_records),
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(control_records),
    )
    assert isinstance(
        _committed_ref_binding(control[4]).catalog[0].source,
        mod.ZeroCopySourceHandle,
    )

    malformed = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "未完/🙂/!@#",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                },
                {
                    "id": "missing-result",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "未完/🙂/!@#", "content": "界" * 17_000},
    ]
    monkeypatch.setattr(
        mod,
        "_get_tiktoken_encoder",
        lambda _request=None: (CountingEncoder(count=12_000), "test"),
    )
    eligible, eligible_forwards, _, eligible_registry, _ = await _run_pipe_boundary(
        monkeypatch,
        messages=malformed,
    )
    expected_ref = (
        f"tool:{hashlib.sha256(malformed[1]['content'].encode()).hexdigest()}"
    )
    assert eligible == {"ok": True}
    assert eligible_forwards[0]["messages"][1]["content"] == expected_ref
    assert malformed[1]["content"] not in repr(eligible_forwards[0])
    assert mod.REF_EXEC_TOOL_NAME in eligible_registry

    monkeypatch.setattr(
        mod,
        "_get_tiktoken_encoder",
        lambda _request=None: (CountingEncoder(count=999), "test"),
    )
    below, below_forwards, _, below_registry, _ = await _run_pipe_boundary(
        monkeypatch,
        messages=malformed,
    )
    assert below == {"ok": True}
    assert below_forwards[0]["messages"] == malformed
    assert mod.REF_EXEC_TOOL_NAME not in below_registry

    monkeypatch.setattr(
        mod,
        "_get_tiktoken_encoder",
        lambda _request=None: (CountingEncoder(fail=True), "test"),
    )
    (
        failed_encoder,
        failed_encoder_forwards,
        _,
        failed_encoder_registry,
        _,
    ) = await _run_pipe_boundary(
        monkeypatch,
        messages=malformed,
    )
    assert failed_encoder == {"ok": True}
    assert failed_encoder_forwards[0]["messages"] == malformed
    assert mod.REF_EXEC_TOOL_NAME not in failed_encoder_registry


@pytest.mark.asyncio
async def test_projection_warning_uses_only_fixed_privacy_safe_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert [field.name for field in dataclasses.fields(mod.RefProjectionError)] == [
        "stage"
    ]
    sensitive = (
        "private-payload",
        "private-call-id",
        "private-message-id",
        "chat-1",
        "user-1",
        "tool:" + "a" * 64,
        hashlib.sha256(b"private-payload").hexdigest(),
        "RefProjectionError",
    )
    valid = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": sensitive[1],
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": sensitive[1],
            "content": sensitive[0] + "x" * 70_000,
        },
    ]
    caplog.set_level("WARNING", logger=mod.__name__)
    monkeypatch.setattr(
        mod,
        "_get_tiktoken_encoder",
        lambda _request=None: (CountingEncoder(count=12_000), "test"),
    )

    def fail_registration(_attempt: mod.RefAttempt) -> mod.RefStateDelta:
        raise mod.RefProjectionError(stage="registration")

    monkeypatch.setattr(mod, "register_ref_attempt", fail_registration)

    result, forwarded, _, _, _ = await _run_pipe_boundary(
        monkeypatch,
        messages=valid,
        metadata_overrides={
            "message_id": sensitive[2],
        },
    )

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and record.name == mod.__name__
    ]
    assert result["error"]["code"] == "ref_projection_failed"
    assert forwarded == []
    assert warnings == [
        "Auto Compact ref projection failed: stage=registration reason=operation_failed"
    ]
    assert all(value not in warnings[0] for value in sensitive)


@pytest.mark.parametrize(
    ("candidate_names", "expected"),
    (
        ((None,), "unknown"),
        (("",), "unknown"),
        (("valid-name",), "valid-name"),
        (("valid-name", None), "valid-name"),
        (("valid-name", "valid-name"), "valid-name"),
        (("first", "second"), "unknown"),
        ((mod.REF_EXEC_TOOL_NAME, None), mod.REF_EXEC_TOOL_NAME),
    ),
)
def test_native_tool_name_classification_tolerates_invalid_candidates(
    candidate_names: tuple[str | None, ...],
    expected: str,
) -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "candidate-call",
                    "type": "function",
                    "function": {"name": name},
                }
                for name in candidate_names
            ],
        }
    ]

    assert mod._native_tool_names_by_call_id(messages) == {"candidate-call": expected}


@pytest.mark.asyncio
async def test_malformed_orphan_and_duplicate_call_ids_externalize_per_call_metadata() -> (
    None
):
    texts = {
        "empty": "empty:" + "a" * 70_000,
        "missing": "missing:" + "b" * 70_000,
        "valid": "valid:" + "c" * 70_000,
        "same": "same:" + "s" * 70_000,
        "ambiguous": "ambiguous:" + "d" * 70_000,
        "orphan": "orphan:" + "e" * 70_000,
        "reader": "reader:" + "f" * 70_000,
    }
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "empty", "type": "function", "function": {"name": ""}},
                {"id": "missing", "type": "function", "function": {}},
                {
                    "id": "valid",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                },
                {"id": "same", "type": "function", "function": {"name": "search_web"}},
                {"id": "same", "type": "function", "function": {"name": "search_web"}},
                {
                    "id": "missing-result",
                    "type": "function",
                    "function": {"name": "other", "arguments": "{}"},
                },
                {"id": "ambiguous", "type": "function", "function": {"name": "one"}},
                {"id": "ambiguous", "type": "function", "function": {"name": "two"}},
                {"id": "reader", "type": "function", "function": {"name": "existing"}},
                {
                    "id": "reader",
                    "type": "function",
                    "function": {"name": mod.REF_EXEC_TOOL_NAME},
                },
            ],
        },
        *(
            {"role": "tool", "tool_call_id": call_id, "content": texts[call_id]}
            for call_id in ("empty", "missing", "valid", "same", "ambiguous", "reader")
        ),
        {"role": "tool", "tool_call_id": "orphan", "content": texts["orphan"]},
        {"role": "user", "content": "continue"},
    ]
    plan = await mod.project_native_tool_texts(
        messages,
        threshold_tokens=1_000,
    )
    projected = await mod.apply_ref_projection_plan(messages, plan)

    refs = {
        call_id: f"tool:{hashlib.sha256(text.encode()).hexdigest()}"
        for call_id, text in texts.items()
    }
    forwarded_tools = {
        message["tool_call_id"]: message["content"]
        for message in projected
        if message.get("role") == "tool"
    }
    labels = {manifest.ref: manifest.tool for manifest in plan.render_manifests}
    assert forwarded_tools == {
        "empty": refs["empty"],
        "missing": refs["missing"],
        "valid": refs["valid"],
        "same": refs["same"],
        "ambiguous": refs["ambiguous"],
        "reader": refs["reader"],
        "orphan": refs["orphan"],
    }
    assert labels == {
        refs["empty"]: "unknown",
        refs["missing"]: "unknown",
        refs["valid"]: "existing",
        refs["same"]: "search_web",
        refs["ambiguous"]: "unknown",
        refs["reader"]: mod.REF_EXEC_TOOL_NAME,
        refs["orphan"]: "unknown",
    }
    assert all(
        texts[key] not in repr(projected)
        for key in (
            "empty",
            "missing",
            "valid",
            "same",
            "ambiguous",
            "reader",
            "orphan",
        )
    )


@pytest.mark.asyncio
async def test_hard_late_checkpoint_reprojects_selected_history_ref_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    current_messages = [
        *history_messages,
        {"role": "user", "content": "continue"},
    ]
    history_source = await mod.build_canonical_history_source(
        history_messages,
        source_message_count=len(history_messages),
    )
    checkpoint = {
        "id": f"accp_{'a' * 64}",
        "namespace": mod.CHECKPOINT_NAMESPACE,
        "user_id": "user-1",
        "chat_id": "chat-1",
        "pipe_function_id": "auto_compact",
        "profile_hash": mod.compute_profile_hash(),
        "source_hash": mod.compute_summary_source_hash(history_messages),
        "source_message_count": len(history_messages),
        "state": "ready",
        "summary_text": "late hard summary",
        "summary_meta": {
            "fixture": "hard-late",
            mod.SUMMARY_META_HISTORY_REF_KEY: {
                "format": mod.HISTORY_REF_FORMAT,
                "raw_source_hash": history_source.raw_source_hash,
            },
        },
        "summary_token_count": 7,
        "parent_checkpoint_id": None,
    }
    checkpoint_before = copy.deepcopy(checkpoint)
    selected_ref = f"history:{checkpoint['id']}"
    late_match = mod.ReusableCheckpointMatch(
        kind="parent",
        source_message_count=len(history_messages),
        checkpoint=checkpoint,
    )
    checkpoint_match_index = 0
    estimate_index = 0
    observed_checkpoint_ids: list[str | None] = []
    estimated_surfaces: list[tuple[set[str], bool]] = []
    marker = '<auto_compact_ref_manifests version="1"><![CDATA['

    def visible_manifest_refs(candidate: dict[str, object]) -> set[str]:
        messages = candidate.get("messages")
        if not isinstance(messages, list):
            return set()
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str) or marker not in content:
                continue
            payload = content.split(marker, maxsplit=1)[1].split(
                "]]></auto_compact_ref_manifests>", maxsplit=1
            )[0]
            return {item["ref"] for item in json.loads(payload)}
        return set()

    async def reusable_checkpoint(**_kwargs):
        nonlocal checkpoint_match_index
        matches = (None, late_match, None)
        match = matches[min(checkpoint_match_index, len(matches) - 1)]
        checkpoint_match_index += 1
        observed_checkpoint_ids.append(
            None if match is None else str(match.checkpoint["id"])
        )
        return match

    async def estimate_tokens(candidate: dict[str, object], *_args, **_kwargs):
        nonlocal estimate_index
        tools = candidate.get("tools")
        reader_schema_visible = isinstance(tools, list) and any(
            isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name") == mod.REF_EXEC_TOOL_NAME
            for tool in tools
        )
        estimated_surfaces.append(
            (visible_manifest_refs(candidate), reader_schema_visible)
        )
        values = (200, 10)
        value = values[min(estimate_index, len(values) - 1)]
        estimate_index += 1
        return value

    async def compact_with_checkpoint(**kwargs):
        body = copy.deepcopy(kwargs["body"])
        body["messages"] = [
            mod.render_summary_message_from_checkpoint(checkpoint),
            current_messages[-1],
        ]
        return body, True, len(history_messages)

    def configure(pipe: mod.Pipe) -> None:
        pipe.valves.ref_exec_enabled = True
        pipe.valves.trigger_input_tokens = 100
        pipe.valves.soft_trigger_ratio = 0.5
        monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint)
        monkeypatch.setattr(
            mod, "_estimate_provider_input_tokens_async", estimate_tokens
        )

    monkeypatch.setattr(
        mod,
        "_compact_body_with_reusable_checkpoint",
        compact_with_checkpoint,
    )
    result, forwarded, _, registry, request = await _run_pipe_boundary(
        monkeypatch,
        messages=current_messages,
        function_calling_capability=True,
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(
            [
                {"id": "prior-user", **history_messages[0]},
                {"id": "prior-assistant", **history_messages[1]},
                {"id": "current-user", **current_messages[-1]},
            ]
        ),
        configure_pipe=configure,
    )

    forwarded_body = forwarded[0]
    rendered = next(
        message
        for message in forwarded_body["messages"]
        if mod._is_rendered_summary_context_message(message)
    )
    manifest_json = (
        rendered["content"]
        .split(marker, maxsplit=1)[1]
        .split("]]></auto_compact_ref_manifests>", maxsplit=1)[0]
    )
    manifest_refs = {item["ref"] for item in json.loads(manifest_json)}
    reader_specs = [
        tool
        for tool in forwarded_body.get("tools", [])
        if tool["function"]["name"] == mod.REF_EXEC_TOOL_NAME
    ]
    binding = _committed_ref_binding(request)
    reader = registry[mod.REF_EXEC_TOOL_NAME]["callable"]
    expected_history = "\n".join(history_source.iter_records())
    resolve_calls = 0
    resolve_history_ref_catalog_entry = mod.resolve_history_ref_catalog_entry

    async def observe_history_resolution(*args: object, **kwargs: object):
        nonlocal resolve_calls
        resolve_calls += 1
        return await resolve_history_ref_catalog_entry(*args, **kwargs)

    monkeypatch.setattr(
        mod,
        "resolve_history_ref_catalog_entry",
        observe_history_resolution,
    )

    assert result == {"ok": True}
    assert observed_checkpoint_ids == [None, checkpoint["id"], None]
    assert estimated_surfaces[0] == (set(), False)
    assert estimated_surfaces[1] == ({selected_ref}, True)
    assert rendered["content"].count(marker) == 1
    assert rendered["content"].endswith(
        "]]></auto_compact_ref_manifests>\n</auto_compaction_context>"
    )
    assert selected_ref in manifest_refs
    assert len(reader_specs) == 1
    assert selected_ref in {entry.manifest.ref for entry in binding.catalog}
    assert await _read(reader, f"cat {selected_ref}") == expected_history
    assert await _read(reader, f"wc -l {selected_ref}") == str(
        history_source.line_count
    )
    assert resolve_calls == 1
    assert checkpoint == checkpoint_before


@pytest.mark.asyncio
async def test_final_checkpoint_lookup_reprojects_selected_candidate_when_plan_expands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_a_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    checkpoint_b_messages = [
        *checkpoint_a_messages,
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    current_messages = [
        *checkpoint_b_messages,
        {"role": "user", "content": "continue"},
    ]
    history_a = await mod.build_canonical_history_source(
        checkpoint_a_messages,
        source_message_count=len(checkpoint_a_messages),
    )
    history_b = await mod.build_canonical_history_source(
        checkpoint_b_messages,
        source_message_count=len(checkpoint_b_messages),
    )
    checkpoint_a = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_summary_source_hash(checkpoint_a_messages),
        source_message_count=len(checkpoint_a_messages),
        summary_text="checkpoint A summary",
        summary_meta={
            "fixture": "final-lookup-a",
            mod.SUMMARY_META_HISTORY_REF_KEY: {
                "format": mod.HISTORY_REF_FORMAT,
                "raw_source_hash": history_a.raw_source_hash,
            },
        },
        summary_token_count=7,
        parent_checkpoint_id=None,
        now=1,
    )
    checkpoint_b = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_summary_source_hash(checkpoint_b_messages),
        source_message_count=len(checkpoint_b_messages),
        summary_text="checkpoint B summary",
        summary_meta={
            "fixture": "final-lookup-b",
            mod.SUMMARY_META_HISTORY_REF_KEY: {
                "format": mod.HISTORY_REF_FORMAT,
                "raw_source_hash": history_b.raw_source_hash,
            },
        },
        summary_token_count=11,
        parent_checkpoint_id=checkpoint_a["id"],
        now=2,
    )
    checkpoint_a_before = copy.deepcopy(checkpoint_a)
    checkpoint_b_before = copy.deepcopy(checkpoint_b)
    ref_a = f"history:{checkpoint_a['id']}"
    ref_b = f"history:{checkpoint_b['id']}"
    expected_refs = {ref_a, ref_b}
    match_a = mod.ReusableCheckpointMatch(
        kind="parent",
        source_message_count=len(checkpoint_a_messages),
        checkpoint=checkpoint_a,
    )
    match_b = mod.ReusableCheckpointMatch(
        kind="exact",
        source_message_count=len(checkpoint_b_messages),
        checkpoint=checkpoint_b,
    )
    checkpoint_match_index = 0
    observed_checkpoint_ids: list[str | None] = []
    estimated_manifest_refs: list[set[str]] = []
    routed_manifest_refs: list[set[str]] = []
    marker = '<auto_compact_ref_manifests version="1"><![CDATA['

    def visible_manifest_refs(candidate: dict[str, object]) -> set[str]:
        messages = candidate.get("messages")
        if not isinstance(messages, list):
            return set()
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str) or marker not in content:
                continue
            payload = content.split(marker, maxsplit=1)[1].split(
                "]]></auto_compact_ref_manifests>", maxsplit=1
            )[0]
            return {item["ref"] for item in json.loads(payload)}
        return set()

    async def reusable_checkpoint(**_kwargs: object) -> mod.ReusableCheckpointMatch | None:
        nonlocal checkpoint_match_index
        matches = (None, match_a, match_b)
        match = matches[min(checkpoint_match_index, len(matches) - 1)]
        checkpoint_match_index += 1
        observed_checkpoint_ids.append(
            None if match is None else str(match.checkpoint["id"])
        )
        return match

    async def estimate_tokens(
        candidate: dict[str, object], *_args: object, **_kwargs: object
    ) -> int:
        refs = visible_manifest_refs(candidate)
        estimated_manifest_refs.append(refs)
        return 200 if not refs else 10

    async def compact_with_checkpoint(**kwargs: object):
        candidate = copy.deepcopy(kwargs["body"])
        candidate["messages"] = [
            mod.render_summary_message_from_checkpoint(checkpoint_a),
            current_messages[-1],
        ]
        return candidate, True, len(checkpoint_a_messages)

    class LineageStore:
        async def lookup_ready_descriptor_by_id(
            self, checkpoint_id: str, **_kwargs: object
        ) -> dict[str, object] | None:
            if checkpoint_id == checkpoint_a["id"]:
                return copy.deepcopy(checkpoint_a)
            return None

    original_route_params = mod._apply_resolved_model_route_params

    def apply_route_params(
        candidate: dict[str, object], **kwargs: object
    ) -> dict[str, object]:
        routed_manifest_refs.append(visible_manifest_refs(candidate))
        return original_route_params(candidate, **kwargs)

    def configure(pipe: mod.Pipe) -> None:
        pipe.valves.ref_exec_enabled = True
        pipe.valves.trigger_input_tokens = 100
        pipe.valves.soft_trigger_ratio = 0.5
        monkeypatch.setattr(mod, "_body_reusable_checkpoint_match", reusable_checkpoint)
        monkeypatch.setattr(
            mod, "_estimate_provider_input_tokens_async", estimate_tokens
        )

    monkeypatch.setattr(mod, "CheckpointStore", LineageStore)
    monkeypatch.setattr(
        mod,
        "_compact_body_with_reusable_checkpoint",
        compact_with_checkpoint,
    )
    monkeypatch.setattr(mod, "_apply_resolved_model_route_params", apply_route_params)
    result, forwarded, _, registry, request = await _run_pipe_boundary(
        monkeypatch,
        messages=current_messages,
        function_calling_capability=True,
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(
            [
                {"id": "prior-user", **checkpoint_a_messages[0]},
                {"id": "prior-assistant", **checkpoint_a_messages[1]},
                {"id": "middle-user", **checkpoint_b_messages[2]},
                {"id": "middle-assistant", **checkpoint_b_messages[3]},
                {"id": "current-user", **current_messages[-1]},
            ]
        ),
        configure_pipe=configure,
    )

    forwarded_body = forwarded[0]
    rendered = next(
        message
        for message in forwarded_body["messages"]
        if mod._is_rendered_summary_context_message(message)
    )
    manifest_refs = visible_manifest_refs(forwarded_body)
    binding = _committed_ref_binding(request)
    catalog_refs = {entry.manifest.ref for entry in binding.catalog}
    reader = registry[mod.REF_EXEC_TOOL_NAME]["callable"]
    preserved_fields = (
        "id",
        "profile_hash",
        "parent_checkpoint_id",
        "source_hash",
        "summary_text",
        "summary_meta",
        "summary_token_count",
    )

    assert result == {"ok": True}
    assert tuple(observed_checkpoint_ids) == (
        None,
        checkpoint_a["id"],
        checkpoint_b["id"],
    )
    assert ref_b not in rendered["content"]
    assert manifest_refs == {ref_a}
    assert routed_manifest_refs[-1] == {ref_a}
    assert estimated_manifest_refs[-1] == {ref_a}
    assert catalog_refs == expected_refs
    assert set((await _read(reader, "ls")).splitlines()) == catalog_refs
    assert await _read(reader, f"cat {ref_a}") == "\n".join(history_a.iter_records())
    assert await _read(reader, f"cat {ref_b}") == "\n".join(history_b.iter_records())
    assert checkpoint_a == checkpoint_a_before
    assert checkpoint_b == checkpoint_b_before
    assert {key: checkpoint_a[key] for key in preserved_fields} == {
        key: checkpoint_a_before[key] for key in preserved_fields
    }
    assert {key: checkpoint_b[key] for key in preserved_fields} == {
        key: checkpoint_b_before[key] for key in preserved_fields
    }


@pytest.mark.asyncio
async def test_apply_ref_manifests_removes_seeded_block_when_plan_omits_summary_history_ref() -> (
    None
):
    own_messages = [{"role": "user", "content": "owned history"}]
    own_source = await mod.build_canonical_history_source(
        own_messages,
        source_message_count=len(own_messages),
    )
    own_checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_summary_source_hash(own_messages),
        source_message_count=len(own_messages),
        summary_text="owned summary",
        summary_meta=None,
        parent_checkpoint_id=None,
        now=1,
    )
    own_ref = f"history:{own_checkpoint['id']}"
    own_plan = mod.build_history_ref_projection_plan(
        (
            mod.RefCatalogEntry(
                manifest=mod.RefManifest(
                    ref=own_ref,
                    utf8_bytes=own_source.utf8_bytes,
                    sha256=own_source.raw_source_hash,
                ),
                source=own_source,
            ),
        )
    )
    body = {
        "messages": [mod.render_summary_message_from_checkpoint(own_checkpoint)],
        "metadata": {},
    }
    marker = '<auto_compact_ref_manifests version="1">'
    mod._apply_ref_manifests(body, own_plan)
    assert body["messages"][0]["content"].count(marker) == 1

    other_messages = [{"role": "user", "content": "other history"}]
    other_source = await mod.build_canonical_history_source(
        other_messages,
        source_message_count=len(other_messages),
    )
    other_checkpoint = mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_summary_source_hash(other_messages),
        source_message_count=len(other_messages),
        summary_text="other summary",
        summary_meta=None,
        parent_checkpoint_id=None,
        now=2,
    )
    other_ref = f"history:{other_checkpoint['id']}"
    other_plan = mod.build_history_ref_projection_plan(
        (
            mod.RefCatalogEntry(
                manifest=mod.RefManifest(
                    ref=other_ref,
                    utf8_bytes=other_source.utf8_bytes,
                    sha256=other_source.raw_source_hash,
                ),
                source=other_source,
            ),
        )
    )
    tool_output = "other tool output:" + "t" * 70_000
    tool_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "other-tool-call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "other-tool-call",
            "content": tool_output,
        },
    ]
    tool_plan = await mod.project_native_tool_texts(
        tool_messages,
        threshold_tokens=1_000,
        encoder=CountingEncoder(count=1_001),
    )
    reapply_plan = mod.merge_ref_projection_plans(other_plan, tool_plan)
    assert reapply_plan is not None
    assert len(reapply_plan.manifests) == 2
    assert own_ref not in {manifest.ref for manifest in reapply_plan.manifests}

    mod._apply_ref_manifests(body, reapply_plan)

    tool_hash = hashlib.sha256(tool_output.encode()).hexdigest()
    content = body["messages"][0]["content"]
    assert content.count(marker) == 0
    assert "</auto_compact_ref_manifests>" not in content
    assert content.endswith("</auto_compaction_context>")
    assert body["metadata"]["auto_compact_ref_manifests"] == [
        {
            "ref": other_ref,
            "sha256": other_source.raw_source_hash,
            "utf8_bytes": other_source.utf8_bytes,
        },
        {
            "ref": f"tool:{tool_hash}",
            "sha256": tool_hash,
            "utf8_bytes": len(tool_output.encode()),
        },
    ]


@pytest.mark.asyncio
async def test_valid_eligible_round_externalizes_when_separate_malformed_round_is_below_threshold() -> (
    None
):
    eligible = "eligible:" + "x" * 70_000
    below_threshold = "raw"
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "eligible-call",
                    "type": "function",
                    "function": {"name": "検索/🙂/!@#", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "eligible-call", "content": eligible},
        {
            "role": "assistant",
            "tool_calls": [{"id": "malformed-call", "type": "function"}],
        },
        {
            "role": "tool",
            "tool_call_id": "malformed-call",
            "content": below_threshold,
        },
    ]
    plan = await mod.project_native_tool_texts(
        messages,
        threshold_tokens=1_000,
        encoder=CountingEncoder(count=999),
    )

    projected = await mod.apply_ref_projection_plan(messages, plan)

    expected_ref = f"tool:{hashlib.sha256(eligible.encode()).hexdigest()}"
    assert projected[1]["content"] == expected_ref
    assert projected[3]["content"] == below_threshold


@pytest.mark.asyncio
async def test_history_catalog_digest_match_cannot_substitute_native_tool_result() -> (
    None
):
    canonical = mod._canonical_history_record({"role": "user", "content": "same"})
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    history_ref = f"history:accp_{'a' * 64}"
    source = mod.CanonicalHistorySourceHandle(
        raw_messages=({"role": "user", "content": "same"},),
        raw_record_limit=1,
        transient_message_patterns=None,
        utf8_bytes=len(canonical.encode()),
        raw_source_hash=digest,
        line_count=1,
    )
    plan = mod.build_history_ref_projection_plan(
        (
            mod.RefCatalogEntry(
                manifest=mod.RefManifest(
                    ref=history_ref,
                    utf8_bytes=len(canonical.encode()),
                    sha256=digest,
                ),
                source=source,
            ),
        )
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tool-call",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tool-call", "content": canonical},
    ]

    projected = await mod.apply_ref_projection_plan(messages, plan)

    assert projected[1]["content"] == canonical


@pytest.mark.asyncio
async def test_multimodal_tool_result_stays_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "text-control:" + "x" * 70_000
    caption = "image-caption:" + "y" * 70_000
    raw_assistant = {
        "id": "multimodal-assistant",
        "role": "assistant",
        "output": [
            {
                "type": "function_call",
                "call_id": "text-call",
                "name": "existing",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "image-call",
                "name": "existing",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "text-call",
                "output": [{"type": "input_text", "text": text}],
            },
            {
                "type": "function_call_output",
                "call_id": "image-call",
                "output": [
                    {"type": "input_text", "text": caption},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,private",
                    },
                ],
            },
        ],
    }
    records = [
        {"id": "prior-user", "role": "user", "content": "lookup"},
        raw_assistant,
        {"id": "current-user", "role": "user", "content": "continue"},
    ]
    expanded = _expanded_core_messages(records)
    synthetic_user = copy.deepcopy(
        next(
            message
            for message in expanded
            if message.get("role") == "user"
            and isinstance(message.get("content"), list)
            and [part.get("type") for part in message["content"]]
            == ["text", "image_url"]
        )
    )

    result, forwarded, _, _, request = await _run_pipe_boundary(
        monkeypatch,
        messages=expanded,
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(records),
    )

    projected = forwarded[0]["messages"]
    projected_tools = {
        message["tool_call_id"]: message["content"]
        for message in projected
        if message.get("role") == "tool"
    }
    projected_synthetic_user = next(
        message
        for message in projected
        if message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and [part.get("type") for part in message["content"]] == ["text", "image_url"]
    )
    binding = _committed_ref_binding(request)
    assert result == {"ok": True}
    assert (
        projected_tools["text-call"]
        == f"tool:{hashlib.sha256(text.encode()).hexdigest()}"
    )
    assert (
        projected_tools["image-call"]
        == f"tool:{hashlib.sha256(caption.encode()).hexdigest()}"
    )
    assert projected_synthetic_user == synthetic_user
    assert len(binding.catalog) == 2
    sources_by_ref = {entry.manifest.ref: entry.source for entry in binding.catalog}
    text_source = sources_by_ref[f"tool:{hashlib.sha256(text.encode()).hexdigest()}"]
    image_source = sources_by_ref[
        f"tool:{hashlib.sha256(caption.encode()).hexdigest()}"
    ]
    assert isinstance(text_source, mod.ZeroCopySourceHandle)
    assert text_source.text is text
    assert isinstance(image_source, mod.ZeroCopySourceHandle)
    assert image_source.text is caption
    assert caption not in repr(forwarded[0])


@pytest.fixture
def task22_ready_checkpoint() -> dict[str, object]:
    history_messages = [
        {"role": "user", "content": "stored question"},
        {"role": "assistant", "content": "stored answer"},
    ]
    return mod.build_checkpoint_row(
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id="auto_compact",
        profile_hash=mod.compute_profile_hash(),
        source_hash=mod.compute_summary_source_hash(history_messages),
        source_message_count=len(history_messages),
        summary_text="stored checkpoint summary",
        summary_meta={
            mod.SUMMARY_META_HISTORY_REF_KEY: {
                "format": mod.HISTORY_REF_FORMAT,
                "raw_source_hash": "a" * 64,
            }
        },
        summary_token_count=4,
        parent_checkpoint_id=f"accp_{'b' * 64}",
        now=1,
    )


@pytest.mark.asyncio
async def test_checkpoint_ref_enrichment_maps_operational_error_to_storage_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    task22_ready_checkpoint: dict[str, object],
) -> None:
    private_detail = "private enrichment driver detail"

    class UnavailableChats:
        @staticmethod
        async def get_messages_map_by_chat_id(
            chat_id: str,
        ) -> dict[str, dict[str, object]]:
            assert chat_id == "chat-1"
            raise OperationalError(
                "SELECT private_enrichment_checkpoint",
                {"secret": "enrichment"},
                RuntimeError(private_detail),
            )

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = UnavailableChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    with pytest.raises(Exception) as failure:
        await mod.extend_ref_projection_plan_with_checkpoint(
            None,
            copy.deepcopy(task22_ready_checkpoint),
            request=SimpleNamespace(state=SimpleNamespace()),
            metadata={"chat_id": "chat-1", "user_message_id": "current-user"},
        )

    displayed = str(failure.value)
    assert type(failure.value).__name__ == "HistoryRefStorageUnavailableError"
    assert displayed == "Auto-compaction could not access its checkpoint store. Please retry."
    assert private_detail not in displayed
    assert "private_enrichment_checkpoint" not in displayed


@pytest.mark.asyncio
async def test_checkpoint_ref_catalog_maps_descriptor_operational_error_to_storage_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    task22_ready_checkpoint: dict[str, object],
) -> None:
    private_detail = "private descriptor driver detail"

    class DescriptorFailureStore:
        async def lookup_ready_descriptor_by_id(
            self,
            checkpoint_id: str,
            **_identity: object,
        ) -> dict[str, object] | None:
            assert checkpoint_id == task22_ready_checkpoint["parent_checkpoint_id"]
            raise OperationalError(
                "SELECT private_parent_descriptor",
                {"secret": "descriptor"},
                RuntimeError(private_detail),
            )

    async def successful_enrichment(**_kwargs: object) -> dict[str, object]:
        return copy.deepcopy(task22_ready_checkpoint)

    monkeypatch.setattr(mod, "CheckpointStore", DescriptorFailureStore)
    monkeypatch.setattr(
        mod,
        "enrich_checkpoint_history_ref",
        successful_enrichment,
    )

    with pytest.raises(Exception) as failure:
        await mod.extend_ref_projection_plan_with_checkpoint(
            None,
            copy.deepcopy(task22_ready_checkpoint),
            request=SimpleNamespace(state=SimpleNamespace()),
            metadata={"chat_id": "chat-1", "user_message_id": "current-user"},
        )

    displayed = str(failure.value)
    assert type(failure.value).__name__ == "HistoryRefStorageUnavailableError"
    assert displayed == "Auto-compaction could not access its checkpoint store. Please retry."
    assert private_detail not in displayed
    assert "private_parent_descriptor" not in displayed


@pytest.mark.asyncio
async def test_pipe_surfaces_history_ref_storage_unavailable_without_target_forward(
    monkeypatch: pytest.MonkeyPatch,
    task22_ready_checkpoint: dict[str, object],
) -> None:
    storage_error_type = getattr(mod, "HistoryRefStorageUnavailableError")
    forwarded: list[dict[str, object]] = []
    match = mod.ReusableCheckpointMatch(
        kind="parent",
        source_message_count=int(task22_ready_checkpoint["source_message_count"]),
        checkpoint=task22_ready_checkpoint,
    )

    async def unavailable_projection(*_args: object, **_kwargs: object) -> None:
        raise storage_error_type(
            "Auto-compaction could not access its checkpoint store. Please retry."
        )

    monkeypatch.setattr(
        mod,
        "extend_ref_projection_plan_with_checkpoint",
        unavailable_projection,
    )

    with pytest.raises(storage_error_type) as failure:
        await _run_pipe_boundary(
            monkeypatch,
            messages=[{"role": "user", "content": "continue"}],
            function_calling_capability=True,
            metadata_overrides={"user_message_id": "current-user"},
            reusable_checkpoint_matches=(match,),
            forwarded_capture=forwarded,
        )

    assert type(failure.value) is storage_error_type
    assert str(failure.value) == (
        "Auto-compaction could not access its checkpoint store. Please retry."
    )
    assert len(forwarded) == 0


@pytest.mark.asyncio
async def test_pipe_propagates_canonical_history_failure_without_forwarding_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_text = "original projection:" + "x" * 70_000
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "task22-call",
                    "type": "function",
                    "function": {"name": "existing", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "task22-call", "content": tool_text},
        {"role": "user", "content": "continue"},
    ]

    async def canonical_failure(*_args: object, **_kwargs: object) -> None:
        raise mod.CanonicalHistoryError(reason="task22 canonical fixture")

    monkeypatch.setattr(
        mod,
        "extend_ref_projection_plan_with_checkpoint",
        canonical_failure,
    )

    forwarded: list[dict[str, object]] = []
    with pytest.raises(mod.CanonicalHistoryError, match="task22 canonical fixture"):
        await _run_pipe_boundary(
            monkeypatch,
            messages=messages,
            function_calling_capability=True,
            metadata_overrides={"user_message_id": "current-user"},
            forwarded_capture=forwarded,
        )

    assert forwarded == []


@pytest.mark.asyncio
async def test_persisted_core_tool_reader_sanitizes_operational_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, _, request, key = await _reader_fixture(monkeypatch)
    checkpoint_id = f"accp_{'a' * 64}"
    raw_source_hash = hashlib.sha256(b"persisted history source").hexdigest()
    source = mod.HistoryRefSourceHandle(
        checkpoint_id=checkpoint_id,
        namespace=mod.CHECKPOINT_NAMESPACE,
        user_id="user-1",
        chat_id="chat-1",
        pipe_function_id=key.pipe_function_id,
        profile_hash=key.profile_hash,
        source_hash="b" * 64,
        source_message_count=1,
        raw_source_hash=raw_source_hash,
        user_message_id="current-user",
        transient_message_patterns=None,
    )
    entry = mod.RefCatalogEntry(
        manifest=mod.RefManifest(
            ref=f"history:{checkpoint_id}",
            utf8_bytes=None,
            sha256=raw_source_hash,
        ),
        source=source,
    )
    store = getattr(request.state, mod.REQUEST_STATE_REF_STORE_KEY)
    store.bindings[key] = dataclasses.replace(
        store.bindings[key],
        catalog=(entry,),
    )

    class UnavailableChats:
        @staticmethod
        async def get_messages_map_by_chat_id(
            chat_id: str,
        ) -> dict[str, dict[str, object]]:
            assert chat_id == "chat-1"
            raise OperationalError(
                "SELECT private_persisted_tool_source",
                {"secret": "reader"},
                RuntimeError("private persisted reader driver detail"),
            )

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = UnavailableChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    assert await _read(reader, f"cat {entry.manifest.ref}") == (
        "Error: externalized ref reader is unavailable"
    )


@pytest.mark.asyncio
async def test_reader_cancellation_preserves_cancelled_error_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, refs, _, _ = await _reader_fixture(monkeypatch)
    worker_entered = asyncio.Event()
    release_worker = asyncio.Event()
    cleanup_failed = asyncio.Event()

    async def failing_cleanup_to_thread(
        _function: Callable[..., object],
        *_args: object,
        **_kwargs: object,
    ) -> str:
        worker_entered.set()
        await release_worker.wait()
        cleanup_failed.set()
        raise RuntimeError("private cancellation cleanup detail")

    monkeypatch.setattr(mod.asyncio, "to_thread", failing_cleanup_to_thread)
    reading = asyncio.create_task(_read(reader, f"cat {refs[0]}"))
    await worker_entered.wait()
    reading.cancel()
    release_worker.set()

    with pytest.raises(asyncio.CancelledError):
        await reading

    assert cleanup_failed.is_set()


def _task25_persisted_tool_records(
    label: str,
) -> tuple[str, list[dict[str, object]]]:
    tool_text = f"{label}:" + "x" * 70_000
    return tool_text, [
        {"id": "prior-user", "role": "user", "content": "lookup"},
        _raw_native_round(
            assistant_id=f"{label}-assistant",
            call_id=f"{label}-call",
            name="existing",
            output_parts=[{"type": "input_text", "text": tool_text}],
        ),
        {"id": "current-user", "role": "user", "content": "continue"},
    ]


@pytest.mark.asyncio
async def test_pipe_projects_persisted_tool_refs_as_zero_copy_without_branch_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_text, records = _task25_persisted_tool_records("task25-projection")
    encoder = CountingEncoder(count=1_001)
    original_branch_loader = mod.load_raw_chat_branch
    branch_load_calls = 0

    async def observed_branch_load(
        *,
        chat_id: str,
        metadata: dict[str, CoreFixtureValue],
    ) -> list[dict[str, CoreFixtureValue]]:
        nonlocal branch_load_calls
        branch_load_calls += 1
        return await original_branch_loader(
            chat_id=chat_id,
            metadata=metadata,
        )

    def enable_ref_exec(pipe: mod.Pipe) -> None:
        pipe.valves.ref_exec_enabled = True
        pipe.valves.ref_substitution_threshold_tokens = 1_000

    monkeypatch.setattr(mod, "load_raw_chat_branch", observed_branch_load)
    monkeypatch.setattr(
        mod,
        "_get_tiktoken_encoder",
        lambda _request=None: (encoder, "task25"),
    )

    result, forwarded, _, _, request = await _run_pipe_boundary(
        monkeypatch,
        messages=_expanded_core_messages(records),
        function_calling_capability=True,
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(records),
        configure_pipe=enable_ref_exec,
        reusable_checkpoint_matches=(None,),
    )

    binding = _committed_ref_binding(request)
    assert result == {"ok": True}, (result, forwarded)
    assert binding.catalog
    source_types = tuple(type(entry.source).__name__ for entry in binding.catalog)
    assert all(
        isinstance(entry.source, mod.ZeroCopySourceHandle)
        for entry in binding.catalog
    ), source_types
    assert branch_load_calls == 0


@pytest.mark.asyncio
async def test_pipe_registered_tool_reader_avoids_branch_loading_for_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_text, records = _task25_persisted_tool_records("task25-reader")
    encoder = CountingEncoder(count=1_001)
    original_branch_loader = mod.load_raw_chat_branch
    branch_load_calls = 0

    async def observed_branch_load(
        *,
        chat_id: str,
        metadata: dict[str, CoreFixtureValue],
    ) -> list[dict[str, CoreFixtureValue]]:
        nonlocal branch_load_calls
        branch_load_calls += 1
        return await original_branch_loader(
            chat_id=chat_id,
            metadata=metadata,
        )

    def enable_ref_exec(pipe: mod.Pipe) -> None:
        pipe.valves.ref_exec_enabled = True
        pipe.valves.ref_substitution_threshold_tokens = 1_000

    monkeypatch.setattr(mod, "load_raw_chat_branch", observed_branch_load)
    monkeypatch.setattr(
        mod,
        "_get_tiktoken_encoder",
        lambda _request=None: (encoder, "task25"),
    )

    result, forwarded, _, registry, request = await _run_pipe_boundary(
        monkeypatch,
        messages=_expanded_core_messages(records),
        function_calling_capability=True,
        metadata_overrides={"user_message_id": "current-user"},
        raw_message_map=_linked_raw_message_map(records),
        configure_pipe=enable_ref_exec,
        reusable_checkpoint_matches=(None,),
    )

    binding = _committed_ref_binding(request)
    reader = registry[mod.REF_EXEC_TOOL_NAME]["callable"]
    assert result == {"ok": True}, (result, forwarded)
    assert len(binding.catalog) == 1
    branch_load_calls = 0

    read_result = await _read(reader, f"wc -c {binding.catalog[0].manifest.ref}")

    assert read_result == str(len(tool_text.encode()))
    assert branch_load_calls == 0
