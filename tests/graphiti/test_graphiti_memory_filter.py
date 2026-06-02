"""Tests for graphiti/functions/filter/graphiti_memory.py extension."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# Add graphiti filter directory to path
graphiti_filter_dir = Path(__file__).parent.parent.parent / "graphiti" / "functions" / "filter"
sys.path.insert(0, str(graphiti_filter_dir))


class TestGraphitiMemoryFilterInstantiation:
    """Test that the Graphiti Memory filter can be instantiated."""

    def test_import_filter_class(self):
        """Verify Filter class can be imported."""
        from graphiti_memory import Filter

        assert Filter is not None

    def test_instantiate_filter(self):
        """Verify Filter class can be instantiated."""
        from graphiti_memory import Filter

        f = Filter()
        assert f is not None

    def test_valves_initialization(self):
        """Verify Valves are properly initialized."""
        from graphiti_memory import Filter

        f = Filter()
        assert hasattr(f, "valves")
        # Check for key valve attributes
        assert hasattr(f.valves, "priority")
        assert hasattr(f.valves, "api_key")

    def test_has_inlet_method(self):
        """Verify Filter class has inlet method."""
        from graphiti_memory import Filter

        f = Filter()
        assert hasattr(f, "inlet")
        assert callable(f.inlet)

    def test_has_outlet_method(self):
        """Verify Filter class has outlet method."""
        from graphiti_memory import Filter

        f = Filter()
        assert hasattr(f, "outlet")
        assert callable(f.outlet)

    @pytest.mark.asyncio
    async def test_channel_chat_runs_inlet_search(self):
        """Open WebUI 0.9.5 channel requests run through the full filter pipeline."""
        from graphiti_memory import Filter

        f = Filter()
        f._ensure_graphiti_initialized = AsyncMock(return_value=True)
        f.graphiti = SimpleNamespace(
            search_=AsyncMock(
                return_value=SimpleNamespace(edges=[], nodes=[])
            )
        )

        body = {"messages": [{"role": "user", "content": "hello"}]}
        user = {"id": "u1", "email": "u1@example.com", "role": "user", "valves": {}}

        result = await f.inlet(
            body,
            __event_emitter__=AsyncMock(),
            __user__=user,
            __metadata__={"chat_id": "channel:abc123", "message_id": "m1"},
        )

        assert result is body
        f._ensure_graphiti_initialized.assert_awaited_once()
        f.graphiti.search_.assert_awaited_once()
