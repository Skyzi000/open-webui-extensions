"""Tests for graphiti/tools/graphiti_memory_manage.py extension."""

import sys
from pathlib import Path

import pytest

# Add graphiti tools directory to path
graphiti_tools_dir = Path(__file__).parent.parent.parent / "graphiti" / "tools"
sys.path.insert(0, str(graphiti_tools_dir))


class TestGraphitiMemoryManageInstantiation:
    """Test that the Graphiti Memory Manage tool can be instantiated."""

    def test_import_tools_class(self):
        """Verify Tools class can be imported."""
        from graphiti_memory_manage import Tools

        assert Tools is not None

    def test_instantiate_tools(self):
        """Verify Tools class can be instantiated."""
        from graphiti_memory_manage import Tools

        tool = Tools()
        assert tool is not None

    def test_valves_initialization(self):
        """Verify Valves are properly initialized."""
        from graphiti_memory_manage import Tools

        tool = Tools()
        assert hasattr(tool, "valves")
        # Check for key valve attributes
        assert hasattr(tool.valves, "api_key")

    def test_has_expected_methods(self):
        """Verify Tools class has expected public methods."""
        from graphiti_memory_manage import Tools

        tool = Tools()
        expected_methods = [
            "search_entities",
            "search_facts",
            "search_episodes",
            "get_recent_episodes",
            "delete_by_uuids",
        ]
        for method in expected_methods:
            assert hasattr(tool, method), f"Tools missing method: {method}"
            assert callable(getattr(tool, method)), f"Tools.{method} is not callable"
