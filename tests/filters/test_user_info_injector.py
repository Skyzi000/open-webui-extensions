"""Tests for functions/filter/user_info_injector.py extension."""

import sys
from pathlib import Path

import pytest

# Add filter directory to path
filter_dir = Path(__file__).parent.parent.parent / "functions" / "filter"
sys.path.insert(0, str(filter_dir))


class MockGroup:
    """Mock group object for testing Groups integration."""

    def __init__(self, name: str):
        self.name = name


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


class TestGroupsInjection:
    """Test Groups integration with monkeypatching.

    These tests verify that Groups.get_groups_by_member_id is properly called
    and group names are injected into the message, without requiring a database.
    """

    @pytest.fixture
    def base_user(self):
        """Base user data for testing."""
        return {
            "id": "test-user-id",
            "name": "Test User",
            "role": "user",
            "valves": {},
        }

    @pytest.fixture
    def base_body(self):
        """Base request body for testing."""
        return {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }

    def test_groups_injection_with_monkeypatch(self, monkeypatch, base_user, base_body):
        """Verify groups are injected when include_groups=True."""
        from user_info_injector import Filter

        mock_groups = [MockGroup("Engineering"), MockGroup("Admins")]

        from open_webui.models import groups as groups_module
        monkeypatch.setattr(
            groups_module.Groups,
            "get_groups_by_member_id",
            lambda member_id: mock_groups
        )

        f = Filter()
        f.valves.include_groups = True
        result = f.inlet(base_body, __user__=base_user)
        messages = result.get("messages", [])

        # Should have 2 messages (system + user)
        assert len(messages) == 2
        system_content = messages[0]["content"]

        # Verify group names are in the message
        assert "Engineering" in system_content, \
            "Groups should be injected into the message"
        assert "Admins" in system_content, \
            "All groups should be injected into the message"
        assert "Groups:" in system_content, \
            "Groups label should be present"

    def test_groups_exclusion_single(self, monkeypatch, base_user, base_body):
        """Verify excluded_groups UserValve filters out a single group."""
        from user_info_injector import Filter

        mock_groups = [MockGroup("Engineering"), MockGroup("Internal"), MockGroup("Admins")]

        from open_webui.models import groups as groups_module
        monkeypatch.setattr(
            groups_module.Groups,
            "get_groups_by_member_id",
            lambda member_id: mock_groups
        )

        f = Filter()
        f.valves.include_groups = True
        user = {**base_user, "valves": {"excluded_groups": "Internal"}}
        result = f.inlet(base_body, __user__=user)
        messages = result.get("messages", [])

        system_content = messages[0]["content"]

        # Verify Internal is excluded
        assert "Internal" not in system_content, \
            "Excluded groups should not appear in the message"
        # Verify other groups are still present
        assert "Engineering" in system_content
        assert "Admins" in system_content

    def test_groups_exclusion_multiple_comma_separated(self, monkeypatch, base_user, base_body):
        """Verify excluded_groups handles comma-separated list of groups.

        AGENTS.md specifies: 'accept comma-separated strings (e.g., "choiceA,choiceB")'
        """
        from user_info_injector import Filter

        mock_groups = [
            MockGroup("Engineering"),
            MockGroup("Internal"),
            MockGroup("QA"),
            MockGroup("Admins"),
        ]

        from open_webui.models import groups as groups_module
        monkeypatch.setattr(
            groups_module.Groups,
            "get_groups_by_member_id",
            lambda member_id: mock_groups
        )

        f = Filter()
        f.valves.include_groups = True
        # Test comma-separated exclusion with spaces
        user = {**base_user, "valves": {"excluded_groups": "Internal, QA"}}
        result = f.inlet(base_body, __user__=user)
        messages = result.get("messages", [])

        system_content = messages[0]["content"]

        # Verify both Internal and QA are excluded
        assert "Internal" not in system_content, \
            "First excluded group should not appear"
        assert "QA" not in system_content, \
            "Second excluded group should not appear"
        # Verify other groups are still present
        assert "Engineering" in system_content
        assert "Admins" in system_content

    def test_groups_disabled_does_not_call_api(self, monkeypatch, base_user, base_body):
        """Verify Groups API is not called when include_groups=False."""
        from user_info_injector import Filter

        api_called = {"count": 0}

        def mock_get_groups(member_id):
            api_called["count"] += 1
            return []

        from open_webui.models import groups as groups_module
        monkeypatch.setattr(
            groups_module.Groups,
            "get_groups_by_member_id",
            mock_get_groups
        )

        f = Filter()
        f.valves.include_groups = False
        f.inlet(base_body, __user__=base_user)

        assert api_called["count"] == 0, \
            "Groups API should not be called when include_groups=False"

    def test_groups_empty_shows_none(self, monkeypatch, base_user, base_body):
        """Verify 'Groups: None' is shown when user has no groups."""
        from user_info_injector import Filter

        from open_webui.models import groups as groups_module
        monkeypatch.setattr(
            groups_module.Groups,
            "get_groups_by_member_id",
            lambda member_id: []
        )

        f = Filter()
        f.valves.include_groups = True
        result = f.inlet(base_body, __user__=base_user)
        messages = result.get("messages", [])

        system_content = messages[0]["content"]
        assert "Groups: None" in system_content, \
            "Should show 'Groups: None' when user has no groups"

    def test_groups_all_excluded_shows_none(self, monkeypatch, base_user, base_body):
        """Verify 'Groups: None' is shown when all groups are excluded."""
        from user_info_injector import Filter

        mock_groups = [MockGroup("Internal"), MockGroup("QA")]

        from open_webui.models import groups as groups_module
        monkeypatch.setattr(
            groups_module.Groups,
            "get_groups_by_member_id",
            lambda member_id: mock_groups
        )

        f = Filter()
        f.valves.include_groups = True
        user = {**base_user, "valves": {"excluded_groups": "Internal,QA"}}
        result = f.inlet(base_body, __user__=user)
        messages = result.get("messages", [])

        system_content = messages[0]["content"]
        assert "Groups: None" in system_content, \
            "Should show 'Groups: None' when all groups are excluded"
