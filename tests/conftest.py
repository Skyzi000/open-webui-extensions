"""Pytest configuration and shared fixtures for Open WebUI extensions testing."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Create a temporary database file that will be automatically cleaned up
_temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_temp_db_path = _temp_db.name
_temp_db.close()

# Set environment variables before importing Open WebUI modules
os.environ.setdefault("ENV", "dev")
os.environ.setdefault("WEBUI_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_temp_db_path}")
os.environ.setdefault("ENABLE_OLLAMA_API", "false")
os.environ.setdefault("ENABLE_OPENAI_API", "false")


def pytest_configure(config):
    """Add Open WebUI backend to Python path."""
    repo_root = Path(__file__).parent.parent
    backend_path = repo_root / "references" / "open-webui" / "backend"
    if backend_path.exists() and str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))


def pytest_unconfigure(config):
    """Clean up temporary database file after tests."""
    try:
        Path(_temp_db_path).unlink(missing_ok=True)
    except Exception:
        pass


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
