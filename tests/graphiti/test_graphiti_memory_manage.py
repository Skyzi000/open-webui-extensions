"""Tests for graphiti/tools/graphiti_memory_manage.py extension."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


class TestDeleteConfirmation:
    """Deletion runs only when the user explicitly approves the dialog."""

    @pytest.fixture
    def tool(self, monkeypatch):
        import graphiti_memory_manage
        from graphiti_memory_manage import Tools

        tool = Tools()
        tool.helper.ensure_graphiti_initialized = AsyncMock(return_value=True)
        tool.helper.graphiti = MagicMock()
        tool.helper.delete_episodes_by_uuids = AsyncMock(return_value=1)
        monkeypatch.setattr(
            graphiti_memory_manage.EpisodicNode,
            "get_by_uuids",
            AsyncMock(return_value=[SimpleNamespace(name="episode", content="content", created_at="2025-01-01")]),
        )
        return tool

    @pytest.fixture
    def user(self):
        return {"id": "test-user-id", "email": "test@example.com", "name": "Test User", "role": "user", "valves": {}}

    @pytest.mark.parametrize("reply, deleted, message", [
        (True, True, "Deleted 1 episode"),
        (False, False, "User cancelled"),
        ("gone", False, "Confirmation failed: Client session disconnected."),
    ])
    async def test_delete_by_uuids_deletes_only_after_approval(self, tool, user, monkeypatch, reply, deleted, message):
        from open_webui.socket import main as socket

        if reply == "gone":
            # Open WebUI's real event caller for a browser session that is gone
            monkeypatch.setattr(socket, "SESSION_POOL", {})
            event_call = await socket.get_event_call(
                {"session_id": "gone", "user_id": user["id"], "chat_id": "chat", "message_id": "message"}
            )
        else:
            event_call = AsyncMock(return_value=reply)

        result = await tool.delete_by_uuids(episode_uuids="ep-1", __user__=user, __event_call__=event_call)

        assert message in result
        assert tool.helper.delete_episodes_by_uuids.await_count == int(deleted)
