"""Target-selection logic for ``--changed``."""

from __future__ import annotations

from pathlib import Path

from open_webui_plugins_builder.config import ReleaseConfig, Target
from open_webui_plugins_builder.selection import select_changed_targets


def _config(tmp_path: Path) -> ReleaseConfig:
    return ReleaseConfig(
        repo_root=tmp_path.resolve(),
        local_import_roots=("owui_ext.shared",),
        source_root="src",
        targets=(
            Target(
                name="alpha",
                source=(tmp_path / "src/owui_ext/tools/alpha.py").resolve(),
                output=(tmp_path / "tools/alpha.py").resolve(),
            ),
            Target(
                name="beta",
                source=(tmp_path / "src/owui_ext/tools/beta.py").resolve(),
                output=(tmp_path / "tools/beta.py").resolve(),
            ),
        ),
    )


def test_release_toml_change_selects_all(tmp_path: Path) -> None:
    config = _config(tmp_path)
    selected = select_changed_targets(config, changed_files={"release.toml"})
    assert {t.name for t in selected} == {"alpha", "beta"}


def test_builder_change_selects_all(tmp_path: Path) -> None:
    config = _config(tmp_path)
    selected = select_changed_targets(
        config,
        changed_files={"open_webui_plugins_builder/open_webui_plugins_builder/cli.py"},
    )
    assert {t.name for t in selected} == {"alpha", "beta"}


def test_wrapper_script_change_selects_all(tmp_path: Path) -> None:
    config = _config(tmp_path)
    selected = select_changed_targets(
        config, changed_files={"scripts/build_release.py"}
    )
    assert {t.name for t in selected} == {"alpha", "beta"}


def test_target_source_change_selects_only_that_target(tmp_path: Path) -> None:
    config = _config(tmp_path)
    selected = select_changed_targets(
        config, changed_files={"src/owui_ext/tools/alpha.py"}
    )
    assert {t.name for t in selected} == {"alpha"}


def test_target_output_change_selects_only_that_target(tmp_path: Path) -> None:
    config = _config(tmp_path)
    selected = select_changed_targets(
        config, changed_files={"tools/alpha.py"}
    )
    assert {t.name for t in selected} == {"alpha"}


def test_shared_change_selects_all_targets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    selected = select_changed_targets(
        config, changed_files={"src/owui_ext/shared/util.py"}
    )
    # Without per-target dep tracking, any shared change invalidates everything.
    assert {t.name for t in selected} == {"alpha", "beta"}


def test_unrelated_change_selects_nothing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    selected = select_changed_targets(
        config, changed_files={"README.md", "tests/conftest.py"}
    )
    assert selected == []
