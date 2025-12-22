"""Tests for functions/filter/user_info_injector.py extension."""

import sys
from pathlib import Path

import pytest

# Add filter directory to path
filter_dir = Path(__file__).parent.parent.parent / "functions" / "filter"
sys.path.insert(0, str(filter_dir))


class TestUserInfoInjectorInstantiation:
    """Test that the User Info Injector filter can be instantiated."""

    def test_import_filter_class(self):
        """Verify Filter class can be imported."""
        from user_info_injector import Filter

        assert Filter is not None

    def test_instantiate_filter(self):
        """Verify Filter class can be instantiated."""
        from user_info_injector import Filter

        f = Filter()
        assert f is not None

    def test_valves_initialization(self):
        """Verify Valves are properly initialized."""
        from user_info_injector import Filter

        f = Filter()
        assert hasattr(f, "valves")
        assert hasattr(f.valves, "priority")
        assert hasattr(f.valves, "include_role")
        assert hasattr(f.valves, "include_groups")

    def test_has_inlet_method(self):
        """Verify Filter class has inlet method."""
        from user_info_injector import Filter

        f = Filter()
        assert hasattr(f, "inlet")
        assert callable(f.inlet)


class TestUserInfoInjectorBehavior:
    """Test User Info Injector filter behavior."""

    def test_inlet_empty_messages(self):
        """Verify inlet handles empty messages."""
        from user_info_injector import Filter

        f = Filter()
        body = {"messages": []}
        result = f.inlet(body)
        assert result == body

    def test_inlet_no_user(self):
        """Verify inlet handles missing user."""
        from user_info_injector import Filter

        f = Filter()
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        result = f.inlet(body, __user__=None)
        # Should return unchanged
        assert len(result["messages"]) == 1

    def test_inlet_injects_user_info(self):
        """Verify inlet injects user info system message."""
        from user_info_injector import Filter

        f = Filter()
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        user = {
            "id": "test-user-id",
            "name": "Test User",
            "role": "user",
            "valves": {},
        }
        result = f.inlet(body, __user__=user)
        messages = result.get("messages", [])

        # Should have 2 messages now (system + user)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Test User" in messages[0]["content"]
        assert "Current User" in messages[0]["content"]
