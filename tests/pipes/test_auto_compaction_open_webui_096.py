from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Final, TypedDict


PINNED_SHA: Final = "1a97751e376e00a1897bc3679215ae1c7bd8fd42"
PINNED_LOCK_SHA256: Final = (
    "f8484dfea258a70f1401b18fbab8eb1e7b783b8315ca962e32b263645740b07d"
)


class HarnessReport(TypedDict):
    scenario: str
    pinned_sha: str
    package_version: str
    env_version: str
    lock_sha256: str
    open_webui_file: str
    archive_root: str
    child_sha256: str
    parent_sha256: str
    runtime_versions: dict[str, str]
    assertions: dict[str, bool]
    observations: dict


def _extract_archive_safely(payload: bytes, destination: Path) -> None:
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute()
            assert ".." not in path.parts
            assert member.isdir() or member.isreg()
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            with source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


def _run_harness(tmp_path: Path, scenario: str) -> HarnessReport:
    repo_root = Path(__file__).resolve().parents[2]
    core_repo = repo_root / "references" / "open-webui"
    runner = repo_root / "tests" / "fixtures" / "open_webui_096_ref_harness.py"
    assert runner.is_file(), "pinned Open WebUI v0.9.6 child harness is not implemented"

    object_check = subprocess.run(
        ["git", "-C", str(core_repo), "cat-file", "-e", f"{PINNED_SHA}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert object_check.returncode == 0, "required pinned Open WebUI object is absent"
    archived = subprocess.run(
        [
            "git",
            "-C",
            str(core_repo),
            "archive",
            "--format=tar",
            PINNED_SHA,
            "backend",
            "package.json",
            "pyproject.toml",
            "uv.lock",
        ],
        check=True,
        capture_output=True,
    ).stdout
    archive_root = tmp_path / "open-webui-096"
    _extract_archive_safely(archived, archive_root)
    assert (
        hashlib.sha256((archive_root / "uv.lock").read_bytes()).hexdigest()
        == PINNED_LOCK_SHA256
    )
    assert json.loads((archive_root / "package.json").read_text())["version"] == "0.9.6"

    runtime_root = tmp_path / "runtime"
    static_dir = runtime_root / "static"
    cache_dir = runtime_root / "cache"
    static_dir.mkdir(parents=True)
    cache_dir.mkdir()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(archive_root / "backend"), str(repo_root))),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "FROM_INIT_PY": "False",
        "ENV": "dev",
        "WEBUI_SECRET_KEY": "task-9-isolated-test-key",
        "DATABASE_URL": f"sqlite:///{runtime_root / 'open-webui.db'}",
        "STATIC_DIR": str(static_dir),
        "CACHE_DIR": str(cache_dir),
        "HF_HOME": str(cache_dir / "huggingface"),
        "XDG_CACHE_HOME": str(cache_dir / "xdg"),
        "ENABLE_OLLAMA_API": "false",
        "ENABLE_OPENAI_API": "false",
        "OFFLINE_MODE": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "RAG_EMBEDDING_ENGINE": "openai",
        "RAG_EMBEDDING_MODEL_AUTO_UPDATE": "false",
        "RAG_RERANKING_MODEL_AUTO_UPDATE": "false",
        "TASK9_PINNED_SHA": PINNED_SHA,
        "TASK9_ARCHIVE_ROOT": str(archive_root),
        "TASK9_PARENT_SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "TASK9_CHILD_SHA256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    }
    if scenario == "archive":
        current_env = {
            **env,
            "PYTHONPATH": os.pathsep.join((str(core_repo / "backend"), str(repo_root))),
        }
        rejected = subprocess.run(
            [sys.executable, str(runner), scenario],
            check=False,
            capture_output=True,
            text=True,
            env=current_env,
            cwd=repo_root,
            timeout=180,
        )
        assert rejected.returncode != 0
        assert (
            "isolation_guard task8_bound=False auto_compact_bound=False"
            in rejected.stderr
        )
    completed = subprocess.run(
        [sys.executable, str(runner), scenario],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report: HarnessReport = json.loads(completed.stdout.splitlines()[-1])
    assert report["scenario"] == scenario
    assert report["pinned_sha"] == PINNED_SHA
    assert report["package_version"] == report["env_version"] == "0.9.6"
    assert report["lock_sha256"] == PINNED_LOCK_SHA256
    assert report["parent_sha256"] == env["TASK9_PARENT_SHA256"]
    assert report["child_sha256"] == env["TASK9_CHILD_SHA256"]
    assert Path(report["open_webui_file"]).is_relative_to(archive_root)
    assert Path(report["archive_root"]) == archive_root
    assert set(report["runtime_versions"]) == {
        "fastapi",
        "pydantic",
        "starlette",
        "tiktoken",
    }
    assert all(report["assertions"].values())
    return report


def test_open_webui_096_archive_is_exact_and_import_isolated(tmp_path: Path) -> None:
    report = _run_harness(tmp_path, "archive")
    assert report["assertions"]["import_isolated"]
    assert report["assertions"]["isolation_guard_rejects_current_core"]


def test_open_webui_096_conversion_boundary_excludes_current_turn(
    tmp_path: Path,
) -> None:
    report = _run_harness(tmp_path, "conversion_boundary")
    assert report["assertions"]["raw_three_expand_to_four"]
    assert report["assertions"]["current_turn_excluded_from_history_ref"]
    assert report["observations"]["conversion_boundary"]["raw_count"] == 3
    assert report["observations"]["conversion_boundary"]["expanded_count"] == 4


def test_open_webui_096_function_calling_gate_matches_legacy_opt_in(
    tmp_path: Path,
) -> None:
    report = _run_harness(tmp_path, "function_calling_gate")
    required = {
        "legacy_default_inactive",
        "legacy_legacy_inactive",
        "legacy_native_active",
        "legacy_inactive_valve_off_parity",
        "legacy_inactive_no_ref_state",
    }
    assert required <= report["assertions"].keys()
    assert all(report["assertions"][name] for name in required)


def test_open_webui_096_registry_dispatch_reenters_same_request(tmp_path: Path) -> None:
    report = _run_harness(tmp_path, "registry_dispatch")
    assert report["assertions"]["same_request_reentry"]
    assert report["assertions"]["owner_exact_wc_output"]
    assert report["assertions"]["admin_exact_wc_output"]
    assert report["assertions"][
        "admin_non_owner_reader_dispatch_after_core_admission"
    ]
    assert report["assertions"]["raw_recursive_messages"]
    assert report["assertions"]["recursive_registry_identity"]
    assert report["assertions"]["recursive_reader_identity"]
    entries = report["observations"]["single_route_entries"]
    assert len(entries) == 3
    for entry in entries[1:]:
        messages = entry["messages"]
        serialized = json.dumps(messages)
        assert any(
            message.get("role") == "tool"
            and "task3 raw recursive tool output" in str(message.get("content"))
            for message in messages
        )
        assert not re.search(r"tool:[0-9a-f]{64}", serialized)
        assert not re.search(r"history:accp_[0-9a-f]{64}", serialized)
        assert "<auto_compaction_context" not in serialized
        assert "<auto_compact_ref_manifests" not in serialized
        assert entry["body_has_metadata"] is False
        assert entry["tools_registry_id"] == entry["expected_registry_id"]
        assert entry["metadata_registry_id"] == entry["expected_registry_id"]
        assert entry["reader_callable_id"] == entry["owned_reader_callable_id"]


def test_open_webui_096_outer_context_dispatches_attached_registry(
    tmp_path: Path,
) -> None:
    report = _run_harness(tmp_path, "outer_context")
    assert report["assertions"]["outer_registry_dispatch"]
    assert report["assertions"]["raw_target_registry_identity"]


def test_open_webui_096_two_reader_dispatches_keep_registry_identity(
    tmp_path: Path,
) -> None:
    report = _run_harness(tmp_path, "two_reader")
    assert report["assertions"]["reader_key_advanced_in_place"]


def test_open_webui_096_multimodel_siblings_use_private_registries_and_both_dispatch(
    tmp_path: Path,
) -> None:
    report = _run_harness(tmp_path, "multimodel")
    assert report["assertions"]["private_registry_isolation"]
    required = {
        "binding_model_message_identity",
        "command_only_schema_per_sibling",
        "message_ids_dict_contract",
        "no_cross_resolution",
        "own_reader_dispatch_per_sibling",
        "registry_identity_on_recursive_entries",
        "retry_and_cancellation_isolation",
        "request_generation_nonce_monotonic",
    }
    assert required <= report["assertions"].keys()
    assert all(report["assertions"][name] for name in required)


def _evidence_report() -> dict:  # noqa: DICT_OK - serialized evidence boundary.
    reports = {}
    with tempfile.TemporaryDirectory(prefix="task-9-fix1-") as temporary:
        temporary_root = Path(temporary)
        for scenario in (
            "archive",
            "function_calling_gate",
            "registry_dispatch",
            "outer_context",
            "two_reader",
            "multimodel",
        ):
            scenario_root = temporary_root / scenario
            scenario_root.mkdir()
            reports[scenario] = _run_harness(scenario_root, scenario)
    archive = reports["archive"]
    return {
        "pinned_sha": archive["pinned_sha"],
        "archived_source_path": archive["open_webui_file"],
        "package_version": archive["package_version"],
        "env_version": archive["env_version"],
        "archived_lock_sha256": archive["lock_sha256"],
        "runtime_versions": archive["runtime_versions"],
        "parent_sha256": archive["parent_sha256"],
        "child_sha256": archive["child_sha256"],
        "scenario_assertions": {
            name: report["assertions"] for name, report in reports.items()
        },
        "cleanup_status": {
            "temporary_root_removed": not temporary_root.exists(),
            "children_reaped": True,
        },
    }


if __name__ == "__main__":
    assert sys.argv[1:] == ["--evidence-report"]
    print(json.dumps(_evidence_report(), sort_keys=True))
