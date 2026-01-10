"""Tests for tools/memory.py extension.

Tests that the Memory tool can be instantiated with Open WebUI dependencies.
"""

import sys
from pathlib import Path

import pytest

# Add tools directory to path
tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))


class TestMemoryToolCompatibility:
    """Test that Memory tool is compatible with current Open WebUI."""

    def test_import_and_instantiate(self):
        """Verify Tools class can be imported and instantiated.

        This test will fail if:
        - open_webui.models.memories cannot be imported
        - Memories class structure has changed incompatibly
        """
        from memory import Tools

        tool = Tools()
        assert tool is not None
        assert hasattr(tool, "valves")

    def test_has_expected_methods(self):
        """Verify Tools class has expected public methods."""
        from memory import Tools

        tool = Tools()
        expected_methods = [
            "add_memory",
            "delete_memory",
            "update_memory",
        ]
        for method in expected_methods:
            assert hasattr(tool, method), f"Tools missing method: {method}"
            assert callable(getattr(tool, method)), f"Tools.{method} is not callable"

    def test_valves_has_expected_fields(self):
        """Verify Valves has expected configuration fields."""
        from memory import Tools

        tool = Tools()
        expected_fields = ["USE_MEMORY", "CONFIRMATION_TIMEOUT", "DEBUG"]
        for field in expected_fields:
            assert hasattr(tool.valves, field), f"Valves missing field: {field}"
