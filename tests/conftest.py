"""Pytest configuration and shared fixtures for Open WebUI extensions testing."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Create a temporary database file that will be automatically cleaned up
_temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_temp_db_path = _temp_db.name
_temp_db.close()

# open_webui.config wipes STATIC_DIR at import time and repopulates it from
# the frontend build; point it away from the submodule's tracked static assets.
# Unconditional override: an inherited STATIC_DIR would let Core delete files
# under that real path. Restored in pytest_unconfigure.
_temp_static_dir = tempfile.mkdtemp(prefix="owui-static-")
_orig_static_dir = os.environ.get("STATIC_DIR")
os.environ["STATIC_DIR"] = _temp_static_dir

# Set environment variables before importing Open WebUI modules
os.environ.setdefault("ENV", "dev")
os.environ.setdefault("WEBUI_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_temp_db_path}")
os.environ.setdefault("ENABLE_OLLAMA_API", "false")
os.environ.setdefault("ENABLE_OPENAI_API", "false")


def pytest_configure(config):
    """Add Open WebUI backend and tools directories to Python path."""
    repo_root = Path(__file__).parent.parent
    backend_path = repo_root / "references" / "open-webui" / "backend"
    if backend_path.exists() and str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    tools_dir = repo_root / "tools"
    if tools_dir.exists() and str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))


def pytest_unconfigure(config):
    """Clean up temporary database file and static dir after tests."""
    try:
        Path(_temp_db_path).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        shutil.rmtree(_temp_static_dir, ignore_errors=True)
    except Exception:
        pass
    if _orig_static_dir is None:
        os.environ.pop("STATIC_DIR", None)
    else:
        os.environ["STATIC_DIR"] = _orig_static_dir


@pytest.fixture
def mock_user():
    """Provide a mock user dictionary for testing filters and tools."""
    return {
        "id": "test-user-id",
        "email": "test@example.com",
        "name": "Test User",
        "role": "user",
        "valves": {},
    }


@pytest.fixture
def mock_body():
    """Provide a mock request body for testing filters."""
    return {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello, world!"},
        ],
        "stream": False,
    }
