"""Schema validation for ``release.toml``."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from open_webui_plugins_builder.config import load_config, parse_config
from open_webui_plugins_builder.errors import BuildError


def _write_toml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "release.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_load_minimal_config(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
        [meta]
        schema_version = 1

        [settings]
        local_import_roots = ["owui_ext.shared"]

        [[targets]]
        name = "demo"
        source = "src/demo.py"
        output = "tools/demo.py"
        """,
    )

    config = load_config(path)

    assert config.local_import_roots == ("owui_ext.shared",)
    assert len(config.targets) == 1
    target = config.targets[0]
    assert target.name == "demo"
    assert target.source == (tmp_path / "src/demo.py").resolve()
    assert target.output == (tmp_path / "tools/demo.py").resolve()


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    raw = {"meta": {"schema_version": 2}}
    with pytest.raises(BuildError, match="Unsupported"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_meta_as_non_table(tmp_path: Path) -> None:
    """``meta = "x"`` is rejected with BuildError, not AttributeError.

    Without an isinstance guard, ``raw.get("meta")`` returns the string
    and the subsequent ``meta.get("schema_version")`` raised
    ``AttributeError`` -- main() does not catch that, so the CLI / pre-
    commit gate would exit with a stack trace instead of a clean
    "release.toml is invalid" message.
    """
    raw = {"meta": "x"}
    with pytest.raises(BuildError, match="meta"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_settings_as_non_table(tmp_path: Path) -> None:
    """Same protection for ``[settings]`` -- a stringified value would
    AttributeError on the next ``.get`` call without an isinstance
    guard.
    """
    raw = {"meta": {"schema_version": 1}, "settings": "x"}
    with pytest.raises(BuildError, match="settings"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_missing_local_roots(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {"local_import_roots": []},
        "targets": [{"name": "x", "source": "a.py", "output": "b.py"}],
    }
    with pytest.raises(BuildError, match="local_import_roots"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_duplicate_target_name(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {"local_import_roots": ["owui_ext.shared"]},
        "targets": [
            {"name": "a", "source": "a.py", "output": "out_a.py"},
            {"name": "a", "source": "b.py", "output": "out_b.py"},
        ],
    }
    with pytest.raises(BuildError, match="Duplicate target name"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_duplicate_output_path(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {"local_import_roots": ["owui_ext.shared"]},
        "targets": [
            {"name": "a", "source": "a.py", "output": "shared.py"},
            {"name": "b", "source": "b.py", "output": "shared.py"},
        ],
    }
    with pytest.raises(BuildError, match="Duplicate output path"):
        parse_config(raw, repo_root=tmp_path)


def test_default_source_root_is_src(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {"local_import_roots": ["owui_ext.shared"]},
        "targets": [{"name": "a", "source": "a.py", "output": "out.py"}],
    }
    config = parse_config(raw, repo_root=tmp_path)
    assert config.source_root == "src"


def test_custom_source_root(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {
            "local_import_roots": ["owui_ext.shared"],
            "source_root": "lib",
        },
        "targets": [{"name": "a", "source": "a.py", "output": "out.py"}],
    }
    config = parse_config(raw, repo_root=tmp_path)
    assert config.source_root == "lib"


def test_rejects_invalid_source_root(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {
            "local_import_roots": ["owui_ext.shared"],
            "source_root": "",
        },
        "targets": [{"name": "a", "source": "a.py", "output": "out.py"}],
    }
    with pytest.raises(BuildError, match="source_root"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_absolute_source_root(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {
            "local_import_roots": ["owui_ext.shared"],
            "source_root": "/etc",
        },
        "targets": [
            {"name": "a", "source": "src/a.py", "output": "out.py"}
        ],
    }
    with pytest.raises(BuildError, match="source_root"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_parent_traversal_source_root(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {
            "local_import_roots": ["owui_ext.shared"],
            "source_root": "../foo",
        },
        "targets": [
            {"name": "a", "source": "src/a.py", "output": "out.py"}
        ],
    }
    with pytest.raises(BuildError, match="source_root"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_symlink_escape_source_root(tmp_path: Path) -> None:
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "escape").symlink_to(sibling)

    raw = {
        "meta": {"schema_version": 1},
        "settings": {
            "local_import_roots": ["owui_ext.shared"],
            "source_root": "escape",
        },
        "targets": [
            {"name": "a", "source": "src/a.py", "output": "out.py"}
        ],
    }
    with pytest.raises(BuildError, match="resolves outside the repository"):
        parse_config(raw, repo_root=repo)


def test_wraps_toml_decode_error(tmp_path: Path) -> None:
    """Malformed TOML must surface as ``BuildError`` for the CLI handler."""

    path = tmp_path / "release.toml"
    path.write_text("this is = not = valid = toml\n", encoding="utf-8")
    with pytest.raises(BuildError, match="Failed to parse"):
        load_config(path)


def test_rejects_absolute_source_path(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {"local_import_roots": ["owui_ext.shared"]},
        "targets": [
            {"name": "evil", "source": "/etc/passwd", "output": "out.py"}
        ],
    }
    with pytest.raises(BuildError, match="must be a relative path"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_absolute_output_path(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {"local_import_roots": ["owui_ext.shared"]},
        "targets": [
            {"name": "evil", "source": "src/a.py", "output": "/tmp/evil.py"}
        ],
    }
    with pytest.raises(BuildError, match="must be a relative path"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_parent_traversal_in_output(tmp_path: Path) -> None:
    raw = {
        "meta": {"schema_version": 1},
        "settings": {"local_import_roots": ["owui_ext.shared"]},
        "targets": [
            {
                "name": "evil",
                "source": "src/a.py",
                "output": "../../tmp/evil.py",
            }
        ],
    }
    with pytest.raises(BuildError, match="must not contain '..'"):
        parse_config(raw, repo_root=tmp_path)


def test_rejects_symlink_escape_via_resolve(tmp_path: Path) -> None:
    """A path that resolves outside the repo must be refused even without ``..``.

    This guards against a future regression where the parts-based check is
    relaxed but the realpath escapes the repo via a symlink.
    """

    sibling = tmp_path / "sibling"
    sibling.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "escape").symlink_to(sibling)

    raw = {
        "meta": {"schema_version": 1},
        "settings": {"local_import_roots": ["owui_ext.shared"]},
        "targets": [
            {
                "name": "demo",
                "source": "escape/a.py",
                "output": "tools/demo.py",
            }
        ],
    }
    with pytest.raises(BuildError, match="resolves outside the repository"):
        parse_config(raw, repo_root=repo)
