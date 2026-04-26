"""CLI integration tests using a real Git repo under ``tmp_path``.

These tests exercise the full pipeline: argparse → release.toml lookup →
target selection → build/check → output. ``--staged`` paths require the index
because we deliberately implemented them via ``git show :path``.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from open_webui_plugins_builder.cli import main


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def _seed_demo(repo: Path) -> None:
    """Seed a minimal repo with one shared module + one target + release.toml."""

    (repo / "release.toml").write_text(
        textwrap.dedent(
            """
            [meta]
            schema_version = 1

            [settings]
            local_import_roots = ["owui_ext.shared"]

            [[targets]]
            name = "demo"
            source = "src/owui_ext/tools/demo.py"
            output = "tools/demo.py"
            """
        ).lstrip(),
        encoding="utf-8",
    )

    shared_path = repo / "src/owui_ext/shared/util.py"
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    shared_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations


            def helper(x: int) -> int:
                return x + 1
            """
        ).lstrip(),
        encoding="utf-8",
    )

    source_path = repo / "src/owui_ext/tools/demo.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        textwrap.dedent(
            '''
            """
            title: Demo
            version: 0.1.0
            """

            from owui_ext.shared.util import helper


            def run() -> int:
                return helper(1)
            '''
        ).lstrip(),
        encoding="utf-8",
    )


def _build_and_commit(repo: Path) -> None:
    """Run the builder, then commit everything (source + output)."""

    rc = main(
        [
            "--release-toml",
            str(repo / "release.toml"),
            "--all",
        ]
    )
    assert rc == 0
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")


# ---------------------------------------------------------------------------


def test_check_passes_when_output_matches(tmp_path: Path) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)
    _build_and_commit(repo)

    rc = main(
        ["--release-toml", str(repo / "release.toml"), "--all", "--check"]
    )
    assert rc == 0


def test_check_fails_when_source_changed_without_rebuild(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)
    _build_and_commit(repo)

    src = repo / "src/owui_ext/tools/demo.py"
    src.write_text(
        src.read_text(encoding="utf-8").replace(
            "return helper(1)", "return helper(2)"
        ),
        encoding="utf-8",
    )

    rc = main(
        ["--release-toml", str(repo / "release.toml"), "--all", "--check"]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "OUT-OF-DATE" in captured.err
    assert "demo" in captured.err


def test_version_bump_required_when_output_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)
    _build_and_commit(repo)

    src = repo / "src/owui_ext/tools/demo.py"
    src.write_text(
        src.read_text(encoding="utf-8").replace(
            "return helper(1)", "return helper(99)"
        ),
        encoding="utf-8",
    )
    rc_build = main(
        ["--release-toml", str(repo / "release.toml"), "--all"]
    )
    assert rc_build == 0

    rc = main(
        ["--release-toml", str(repo / "release.toml"), "--all", "--check"]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "VERSION-BUMP REQUIRED" in captured.err
    assert "0.1.0" in captured.err


def test_version_bump_satisfied_when_version_changes(
    tmp_path: Path,
) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)
    _build_and_commit(repo)

    src = repo / "src/owui_ext/tools/demo.py"
    src.write_text(
        src.read_text(encoding="utf-8")
        .replace("return helper(1)", "return helper(99)")
        .replace("0.1.0", "0.1.1"),
        encoding="utf-8",
    )
    rc_build = main(
        ["--release-toml", str(repo / "release.toml"), "--all"]
    )
    assert rc_build == 0

    rc = main(
        ["--release-toml", str(repo / "release.toml"), "--all", "--check"]
    )
    assert rc == 0


def test_staged_check_passes_when_index_is_consistent(
    tmp_path: Path,
) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)
    _build_and_commit(repo)

    # Modify source AND rebuild AND stage both. Worktree afterwards may diverge,
    # but --staged reads only from the index, so it should still pass.
    src = repo / "src/owui_ext/tools/demo.py"
    src.write_text(
        src.read_text(encoding="utf-8")
        .replace("return helper(1)", "return helper(7)")
        .replace("0.1.0", "0.1.1"),
        encoding="utf-8",
    )
    main(["--release-toml", str(repo / "release.toml"), "--all"])
    _git(repo, "add", "-A")

    src.write_text(
        "# bogus stray content not staged\n" + src.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    rc = main(
        [
            "--release-toml",
            str(repo / "release.toml"),
            "--all",
            "--check",
            "--staged",
        ]
    )
    assert rc == 0


def test_staged_check_fails_when_only_source_is_staged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)
    _build_and_commit(repo)

    src = repo / "src/owui_ext/tools/demo.py"
    src.write_text(
        src.read_text(encoding="utf-8")
        .replace("return helper(1)", "return helper(7)")
        .replace("0.1.0", "0.1.1"),
        encoding="utf-8",
    )
    _git(repo, "add", "src/owui_ext/tools/demo.py")
    # Output not regenerated, not staged.

    rc = main(
        [
            "--release-toml",
            str(repo / "release.toml"),
            "--all",
            "--check",
            "--staged",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "OUT-OF-DATE" in captured.err


def test_changed_mode_rebuilds_only_changed_target(
    tmp_path: Path,
) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)

    # Add a second target that we will deliberately leave untouched.
    (repo / "src/owui_ext/tools/other.py").write_text(
        textwrap.dedent(
            '''
            """
            title: Other
            version: 0.1.0
            """

            print("other")
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (repo / "release.toml").write_text(
        textwrap.dedent(
            """
            [meta]
            schema_version = 1

            [settings]
            local_import_roots = ["owui_ext.shared"]

            [[targets]]
            name = "demo"
            source = "src/owui_ext/tools/demo.py"
            output = "tools/demo.py"

            [[targets]]
            name = "other"
            source = "src/owui_ext/tools/other.py"
            output = "tools/other.py"
            """
        ).lstrip(),
        encoding="utf-8",
    )

    _build_and_commit(repo)

    src = repo / "src/owui_ext/tools/demo.py"
    src.write_text(
        src.read_text(encoding="utf-8")
        .replace("return helper(1)", "return helper(99)")
        .replace("0.1.0", "0.1.1"),
        encoding="utf-8",
    )
    # Pretend pre-commit: stage source only, then run --changed --check --staged.
    _git(repo, "add", "src/owui_ext/tools/demo.py")
    rc = main(
        [
            "--release-toml",
            str(repo / "release.toml"),
            "--changed",
            "--check",
            "--staged",
        ]
    )
    # Exit non-zero because demo's output is stale (not yet regenerated).
    assert rc == 1
    # If we now regenerate AND stage the new output, --changed --check --staged
    # passes.
    main(["--release-toml", str(repo / "release.toml"), "--all"])
    _git(repo, "add", "-A")
    rc = main(
        [
            "--release-toml",
            str(repo / "release.toml"),
            "--changed",
            "--check",
            "--staged",
        ]
    )
    assert rc == 0


def test_list_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _init_repo(tmp_path) if shutil.which("git") else (tmp_path / "repo")
    if not shutil.which("git"):
        repo.mkdir()
    _seed_demo(repo)
    rc = main(["--release-toml", str(repo / "release.toml"), "--list-outputs"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tools/demo.py" in out
    # Plan example showed relative paths so CI commands like
    # ``git diff -- $(... --list-outputs)`` interpret them correctly.
    assert str(repo) not in out


def test_smoke_test_all_passes_for_built_outputs(
    tmp_path: Path,
) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)
    main(["--release-toml", str(repo / "release.toml"), "--all"])

    rc = main(
        ["--release-toml", str(repo / "release.toml"), "--smoke-test-all"]
    )
    assert rc == 0


def test_smoke_test_all_reports_broken_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)
    main(["--release-toml", str(repo / "release.toml"), "--all"])

    # Corrupt the generated output so the import will fail.
    (repo / "tools/demo.py").write_text(
        "syntax !!! error here\n", encoding="utf-8"
    )

    rc = main(
        ["--release-toml", str(repo / "release.toml"), "--smoke-test-all"]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL" in captured.err


def test_staged_resolves_shared_module_from_index_when_worktree_lacks_it(
    tmp_path: Path,
) -> None:
    """Regression: ``--staged`` must consult the index for module existence.

    Scenario: a shared module is staged, then deleted from the worktree (e.g.,
    accidentally during a rebase). The build under ``--staged`` should still
    succeed because the file is present in the index.
    """

    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = _init_repo(tmp_path)
    _seed_demo(repo)
    _build_and_commit(repo)

    # Add a brand new shared module; bump version; rebuild; stage everything.
    new_shared = repo / "src/owui_ext/shared/extra.py"
    new_shared.write_text(
        "def extra() -> int:\n    return 42\n", encoding="utf-8"
    )
    src = repo / "src/owui_ext/tools/demo.py"
    src.write_text(
        src.read_text(encoding="utf-8")
        .replace("0.1.0", "0.1.1")
        .replace(
            "from owui_ext.shared.util import helper",
            "from owui_ext.shared.util import helper\n"
            "from owui_ext.shared.extra import extra",
        )
        .replace("return helper(1)", "return helper(1) + extra()"),
        encoding="utf-8",
    )
    main(["--release-toml", str(repo / "release.toml"), "--all"])
    _git(repo, "add", "-A")

    # Now delete the shared module from the worktree (still in index).
    new_shared.unlink()

    rc = main(
        [
            "--release-toml",
            str(repo / "release.toml"),
            "--all",
            "--check",
            "--staged",
        ]
    )
    assert rc == 0


def test_release_toml_absolute_output_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: absolute output paths must not allow writes outside repo."""

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "release.toml").write_text(
        textwrap.dedent(
            f"""
            [meta]
            schema_version = 1

            [settings]
            local_import_roots = ["owui_ext.shared"]

            [[targets]]
            name = "evil"
            source = "src/owui_ext/tools/demo.py"
            output = "{tmp_path / 'evil.py'}"
            """
        ).lstrip(),
        encoding="utf-8",
    )
    rc = main(
        ["--release-toml", str(repo / "release.toml"), "--all"]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "must be a relative path" in captured.err
    assert not (tmp_path / "evil.py").exists()
