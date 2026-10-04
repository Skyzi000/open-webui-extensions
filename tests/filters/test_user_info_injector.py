"""Tests for functions/filter/user_info_injector.py extension."""

import copy
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

    async def test_inlet_empty_messages(self):
        """Verify inlet handles empty messages."""
        from user_info_injector import Filter

        f = Filter()
        body = {"messages": []}
        result = await f.inlet(body)
        assert result == body

    async def test_inlet_no_user(self):
        """Verify inlet handles missing user."""
        from user_info_injector import Filter

        f = Filter()
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        result = await f.inlet(body, __user__=None)
        # Should return unchanged
        assert len(result["messages"]) == 1

    async def test_inlet_appends_tagged_context_to_latest_user_message(self):
        """Verify the default adds no message and tags the text appended to the latest user message."""
        from user_info_injector import Filter

        f = Filter()
        f.valves.include_groups = False
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
        result = await f.inlet(body, __user__=user)

        assert result["messages"] == [
            {
                "role": "user",
                "content": "Hello\n\n<SYSTEM_CONTEXT>\n# Current User\nName: Test User\nRole: user\n</SYSTEM_CONTEXT>",
            },
        ]

    async def test_inlet_appends_to_first_text_part_without_mutating_the_original(self):
        """Verify list content gets the text in its first text part, which is the one Open WebUI reads as the prompt."""
        from user_info_injector import Filter

        f = Filter()
        f.valves.include_groups = False
        f.valves.include_role = False
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        latest = {"role": "user", "content": [{"type": "text", "text": "look"}, image]}
        original = copy.deepcopy(latest)

        result = await f.inlet({"messages": [latest]}, __user__={"id": "u1", "name": "Test User"})

        assert result["messages"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "look\n\n<SYSTEM_CONTEXT>\n# Current User\nName: Test User\n</SYSTEM_CONTEXT>",
                    },
                    image,
                ],
            }
        ]
        assert latest == original

    @pytest.mark.parametrize(
        "text",
        ["/skills:create", "  /Skills:Create summarize\nthis chat  ", "/skills:createx", "please /skills:create"],
    )
    async def test_inlet_skips_exactly_what_core_handles_as_skills_create(self, text):
        """Verify injection is skipped for, and only for, text that Open WebUI rewrites as a /skills:create command."""
        from open_webui.utils.skills import apply_skills_create_prompt
        from user_info_injector import Filter

        f = Filter()
        f.valves.include_groups = False
        result = await f.inlet(
            {"messages": [{"role": "user", "content": text}]},
            __user__={"id": "u1", "name": "Test User"},
        )

        skipped = result["messages"] == [{"role": "user", "content": text}]
        is_command = apply_skills_create_prompt([{"role": "user", "content": text}], allowed=True)
        assert skipped == is_command

    async def test_inlet_injects_into_message_without_text(self):
        """Verify an image-only message still gets the context, as its only text."""
        from user_info_injector import Filter

        f = Filter()
        f.valves.include_groups = False
        f.valves.include_role = False
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        result = await f.inlet(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": ""}, image]}]},
            __user__={"id": "u1", "name": "Test User"},
        )

        assert result["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<SYSTEM_CONTEXT>\n# Current User\nName: Test User\n</SYSTEM_CONTEXT>"},
                    image,
                ],
            }
        ]

    async def test_inlet_system_role_injects_untagged_message_before_latest_user(self):
        """Verify system role keeps the latest user message last."""
        from user_info_injector import Filter

        f = Filter()
        f.valves.include_groups = False
        f.valves.message_role = "system"
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
        result = await f.inlet(body, __user__=user)

        assert result["messages"] == [
            {"role": "system", "content": "# Current User\nName: Test User\nRole: user"},
            {"role": "user", "content": "Hello"},
        ]

    async def test_inlet_system_after_injects_untagged_message_after_latest_user(self):
        """Verify system_after lands right after the latest user message, before a trailing assistant turn."""
        from user_info_injector import Filter

        f = Filter()
        f.valves.include_groups = False
        f.valves.message_role = "system_after"
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "partial answer"},
            ]
        }
        user = {
            "id": "test-user-id",
            "name": "Test User",
            "role": "user",
            "valves": {},
        }
        result = await f.inlet(body, __user__=user)

        assert result["messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "# Current User\nName: Test User\nRole: user"},
            {"role": "assistant", "content": "partial answer"},
        ]


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

    async def test_groups_injection_with_monkeypatch(self, monkeypatch, base_user, base_body):
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
        result = await f.inlet(base_body, __user__=base_user)
        messages = result.get("messages", [])

        assert len(messages) == 1
        system_content = messages[0]["content"]

        # Verify group names are in the message
        assert "Engineering" in system_content, \
            "Groups should be injected into the message"
        assert "Admins" in system_content, \
            "All groups should be injected into the message"
        assert "Groups:" in system_content, \
            "Groups label should be present"

    async def test_groups_exclusion_single(self, monkeypatch, base_user, base_body):
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
        result = await f.inlet(base_body, __user__=user)
        messages = result.get("messages", [])

        system_content = messages[0]["content"]

        # Verify Internal is excluded
        assert "Internal" not in system_content, \
            "Excluded groups should not appear in the message"
        # Verify other groups are still present
        assert "Engineering" in system_content
        assert "Admins" in system_content

    async def test_groups_exclusion_multiple_comma_separated(self, monkeypatch, base_user, base_body):
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
        result = await f.inlet(base_body, __user__=user)
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

    async def test_groups_disabled_does_not_call_api(self, monkeypatch, base_user, base_body):
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
        await f.inlet(base_body, __user__=base_user)

        assert api_called["count"] == 0, \
            "Groups API should not be called when include_groups=False"

    async def test_groups_empty_shows_none(self, monkeypatch, base_user, base_body):
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
        result = await f.inlet(base_body, __user__=base_user)
        messages = result.get("messages", [])

        system_content = messages[0]["content"]
        assert "Groups: None" in system_content, \
            "Should show 'Groups: None' when user has no groups"

    async def test_groups_all_excluded_shows_none(self, monkeypatch, base_user, base_body):
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
        result = await f.inlet(base_body, __user__=user)
        messages = result.get("messages", [])

        system_content = messages[0]["content"]
        assert "Groups: None" in system_content, \
            "Should show 'Groups: None' when all groups are excluded"
