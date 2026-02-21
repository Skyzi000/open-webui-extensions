"""Tests for graphiti/functions/action/delete_graphiti_memory_action.py extension."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add graphiti action directory to path
graphiti_action_dir = Path(__file__).parent.parent.parent / "graphiti" / "functions" / "action"
sys.path.insert(0, str(graphiti_action_dir))

from delete_graphiti_memory_action import Action, find_episodes_by_message_id


class TestFindEpisodesByMessageId:
    """Tests for the find_episodes_by_message_id function."""

    @pytest.fixture
    def mock_driver(self):
        driver = MagicMock()
        driver.execute_query = AsyncMock()
        return driver

    async def test_should_find_episode_by_source_description_when_filter_format(self, mock_driver):
        """Filter format: source_description contains '_message_{id}'."""
        # Given
        message_id = "abc-123"
        record = {
            "uuid": "ep-1",
            "name": "some_episode",
            "group_id": "grp",
            "source_description": f"20250101T000000Z_Chat_chat1_message_{message_id}",
            "content": "Hello world",
            "source": "message",
            "created_at": "2025-01-01T00:00:00Z",
            "valid_at": "2025-01-01T00:00:00Z",
            "entity_edges": [],
        }
        mock_driver.execute_query.return_value = ([record], None, None)

        # When
        result = await find_episodes_by_message_id(mock_driver, message_id, "grp")

        # Then
        assert len(result) == 1
        assert result[0].uuid == "ep-1"

        call_args = mock_driver.execute_query.call_args
        assert call_args.kwargs["sd_pattern"] == f"_message_{message_id}"
        assert call_args.kwargs["name_pattern"] == f"_{message_id}"
        assert call_args.kwargs["group_id"] == "grp"
        assert call_args.kwargs["routing_"] == "r"

    async def test_should_find_episode_by_name_when_pipeline_format(self, mock_driver):
        """Pipeline format: name contains '_{id}'."""
        # Given
        message_id = "def-456"
        record = {
            "uuid": "ep-2",
            "name": f"Chat_Interaction_chat1_{message_id}",
            "group_id": "grp",
            "source_description": "Chat conversation",
            "content": "Some chat content",
            "source": "message",
            "created_at": "2025-01-01T00:00:00Z",
            "valid_at": "2025-01-01T00:00:00Z",
            "entity_edges": [],
        }
        mock_driver.execute_query.return_value = ([record], None, None)

        # When
        result = await find_episodes_by_message_id(mock_driver, message_id, "grp")

        # Then
        assert len(result) == 1
        assert result[0].uuid == "ep-2"

    async def test_should_return_empty_list_when_no_matches(self, mock_driver):
        # Given
        mock_driver.execute_query.return_value = ([], None, None)

        # When
        result = await find_episodes_by_message_id(mock_driver, "nonexistent-id", "grp")

        # Then
        assert result == []

    async def test_should_omit_group_id_filter_when_group_id_is_none(self, mock_driver):
        # Given
        mock_driver.execute_query.return_value = ([], None, None)

        # When
        await find_episodes_by_message_id(mock_driver, "msg-1", None)

        # Then: WHERE clause should not filter by group_id (RETURN clause always has it)
        call_args = mock_driver.execute_query.call_args
        query = call_args.args[0]
        assert "e.group_id = $group_id" not in query
        assert "group_id" not in call_args.kwargs

    async def test_should_include_group_id_filter_when_group_id_provided(self, mock_driver):
        # Given
        mock_driver.execute_query.return_value = ([], None, None)

        # When
        await find_episodes_by_message_id(mock_driver, "msg-1", "my-group")

        # Then
        call_args = mock_driver.execute_query.call_args
        query = call_args.args[0]
        assert "e.group_id = $group_id" in query
        assert call_args.kwargs["group_id"] == "my-group"

    async def test_should_return_multiple_episodes_when_multiple_matches(self, mock_driver):
        # Given
        message_id = "multi-123"
        records = [
            {
                "uuid": f"ep-{i}",
                "name": f"episode_{i}",
                "group_id": "grp",
                "source_description": f"20250101T000000Z_Chat_chat{i}_message_{message_id}",
                "content": f"Content {i}",
                "source": "message",
                "created_at": "2025-01-01T00:00:00Z",
                "valid_at": "2025-01-01T00:00:00Z",
                "entity_edges": [],
            }
            for i in range(3)
        ]
        mock_driver.execute_query.return_value = (records, None, None)

        # When
        result = await find_episodes_by_message_id(mock_driver, message_id, "grp")

        # Then
        assert len(result) == 3
        assert [ep.uuid for ep in result] == ["ep-0", "ep-1", "ep-2"]


class TestActionInstantiation:
    """Test that the Action class can be properly instantiated."""

    def test_should_instantiate_action(self):
        action = Action()
        assert action is not None

    def test_should_have_valves(self):
        action = Action()
        assert hasattr(action, "valves")

    def test_should_not_have_search_related_valves(self):
        action = Action()
        assert not hasattr(action.valves, "max_search_message_length")
        assert not hasattr(action.valves, "sanitize_search_query")
        assert not hasattr(action.valves, "search_strategy")

    def test_should_not_have_search_related_user_valves(self):
        user_valves = Action.UserValves.model_validate({})
        assert not hasattr(user_valves, "search_candidates")

    def test_should_not_have_removed_methods(self):
        action = Action()
        assert not hasattr(action, "_get_content_from_message")
        assert not hasattr(action, "_calculate_content_similarity")
        assert not hasattr(action, "_sanitize_search_query")
        assert not hasattr(action, "_get_search_config")


class TestActionMethod:
    """Tests for the action method's new ID-based matching flow."""

    @pytest.fixture
    def action(self):
        a = Action()
        a.valves = Action.Valves(api_key="test-key")
        a.graphiti = MagicMock()
        a.graphiti.driver = MagicMock()
        a.graphiti.driver.execute_query = AsyncMock()
        a.graphiti.build_indices_and_constraints = AsyncMock()
        a._indices_built = True
        a._last_config = a._get_config_hash()
        return a

    @pytest.fixture
    def user(self):
        return {
            "id": "test-user-id",
            "email": "test@example.com",
            "name": "Test User",
            "role": "user",
            "valves": {},
        }

    @pytest.fixture
    def event_emitter(self):
        return AsyncMock()

    @pytest.fixture
    def event_call(self):
        return AsyncMock(return_value=True)

    async def test_should_return_none_when_user_message_has_no_id(self, action, user, event_emitter, event_call):
        # Given: user message without an ID
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }

        # When
        result = await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then
        assert result is None
        # Verify "Message ID not found" status was emitted
        emitted_statuses = [
            call.args[0]["data"]["description"]
            for call in event_emitter.call_args_list
            if call.args[0].get("type") == "status"
        ]
        assert any("Message ID not found" in s for s in emitted_statuses)

    async def test_should_return_none_when_no_episodes_found(self, action, user, event_emitter, event_call):
        # Given: user message with ID but no matching episodes in DB
        body = {
            "messages": [
                {"role": "user", "content": "Hello", "id": "msg-001"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        action.graphiti.driver.execute_query.return_value = ([], None, None)

        # When
        result = await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then
        assert result is None
        emitted_statuses = [
            call.args[0]["data"]["description"]
            for call in event_emitter.call_args_list
            if call.args[0].get("type") == "status"
        ]
        assert any("No matching episodes found" in s for s in emitted_statuses)

    async def test_should_call_find_episodes_with_correct_message_id(self, action, user, event_emitter, event_call):
        # Given
        message_id = "msg-999"
        body = {
            "messages": [
                {"role": "user", "content": "Hello", "id": message_id},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        action.graphiti.driver.execute_query.return_value = ([], None, None)

        # When
        await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then
        call_args = action.graphiti.driver.execute_query.call_args
        assert call_args.kwargs["sd_pattern"] == f"_message_{message_id}"
        assert call_args.kwargs["name_pattern"] == f"_{message_id}"

    async def test_should_show_confirmation_and_delete_when_episode_found(self, action, user, event_emitter, event_call):
        # Given
        message_id = "msg-del-1"
        body = {
            "messages": [
                {"role": "user", "content": "Hello", "id": message_id},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        record = {
            "uuid": "ep-del-1",
            "name": "test_episode",
            "group_id": "test-user-id",
            "source_description": f"20250101T000000Z_Chat_chat1_message_{message_id}",
            "content": "Hello world content",
            "source": "message",
            "created_at": "2025-01-01T00:00:00Z",
            "valid_at": "2025-01-01T00:00:00Z",
            "entity_edges": [],
        }
        action.graphiti.driver.execute_query.return_value = ([record], None, None)
        action.graphiti.remove_episode = AsyncMock()

        # When
        await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then: confirmation dialog was shown
        event_call.assert_called_once()
        call_data = event_call.call_args.args[0]
        assert call_data["type"] == "confirmation"
        assert "test_episode" in call_data["data"]["message"]

        # Then: episode was deleted
        action.graphiti.remove_episode.assert_called_once_with("ep-del-1")

    async def test_should_not_delete_when_user_cancels(self, action, user, event_emitter):
        # Given
        message_id = "msg-cancel-1"
        body = {
            "messages": [
                {"role": "user", "content": "Hello", "id": message_id},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        record = {
            "uuid": "ep-cancel-1",
            "name": "test_episode",
            "group_id": "test-user-id",
            "source_description": f"20250101T000000Z_Chat_chat1_message_{message_id}",
            "content": "Hello world content",
            "source": "message",
            "created_at": "2025-01-01T00:00:00Z",
            "valid_at": "2025-01-01T00:00:00Z",
            "entity_edges": [],
        }
        action.graphiti.driver.execute_query.return_value = ([record], None, None)
        action.graphiti.remove_episode = AsyncMock()
        event_call = AsyncMock(return_value=False)

        # When
        await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then: episode was NOT deleted
        action.graphiti.remove_episode.assert_not_called()

    async def test_should_extract_id_from_clicked_user_message(self, action, user, event_emitter, event_call):
        # Given: clicked message is a user message
        body = {
            "messages": [
                {"role": "user", "content": "First message", "id": "msg-first"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message", "id": "msg-second"},
            ],
        }
        action.graphiti.driver.execute_query.return_value = ([], None, None)

        # When
        await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then: should use the clicked (last) user message's ID
        call_args = action.graphiti.driver.execute_query.call_args
        assert call_args.kwargs["sd_pattern"] == "_message_msg-second"

    async def test_should_extract_id_from_user_message_when_assistant_clicked(self, action, user, event_emitter, event_call):
        # Given: clicked message is an assistant message, should find preceding user message
        body = {
            "messages": [
                {"role": "user", "content": "My question", "id": "msg-question"},
                {"role": "assistant", "content": "My answer"},
            ],
        }
        action.graphiti.driver.execute_query.return_value = ([], None, None)

        # When
        await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then: should use the user message's ID (preceding the clicked assistant message)
        call_args = action.graphiti.driver.execute_query.call_args
        assert call_args.kwargs["sd_pattern"] == "_message_msg-question"

    async def test_should_delete_multiple_episodes_when_multiple_found(self, action, user, event_emitter, event_call):
        # Given: DB returns multiple matching episodes
        message_id = "msg-multi-1"
        body = {
            "messages": [
                {"role": "user", "content": "Hello", "id": message_id},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        records = [
            {
                "uuid": f"ep-multi-{i}",
                "name": f"episode_{i}",
                "group_id": "test-user-id",
                "source_description": f"20250101T000000Z_Chat_chat{i}_message_{message_id}",
                "content": f"Content for episode {i}",
                "source": "message",
                "created_at": "2025-01-01T00:00:00Z",
                "valid_at": "2025-01-01T00:00:00Z",
                "entity_edges": [],
            }
            for i in range(3)
        ]
        action.graphiti.driver.execute_query.return_value = (records, None, None)
        action.graphiti.remove_episode = AsyncMock()

        # When
        await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then: confirmation dialog shows numbered list
        call_data = event_call.call_args.args[0]
        assert "1. episode_0" in call_data["data"]["message"]
        assert "2. episode_1" in call_data["data"]["message"]
        assert "3. episode_2" in call_data["data"]["message"]

        # Then: all episodes were deleted
        assert action.graphiti.remove_episode.call_count == 3
        deleted_uuids = [call.args[0] for call in action.graphiti.remove_episode.call_args_list]
        assert deleted_uuids == ["ep-multi-0", "ep-multi-1", "ep-multi-2"]

    async def test_should_report_partial_failure_when_some_deletions_fail(self, action, user, event_emitter, event_call):
        # Given: DB returns 2 episodes, one deletion fails
        message_id = "msg-partial-1"
        body = {
            "messages": [
                {"role": "user", "content": "Hello", "id": message_id},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        records = [
            {
                "uuid": f"ep-partial-{i}",
                "name": f"episode_{i}",
                "group_id": "test-user-id",
                "source_description": f"20250101T000000Z_Chat_chat{i}_message_{message_id}",
                "content": f"Content {i}",
                "source": "message",
                "created_at": "2025-01-01T00:00:00Z",
                "valid_at": "2025-01-01T00:00:00Z",
                "entity_edges": [],
            }
            for i in range(2)
        ]
        action.graphiti.driver.execute_query.return_value = (records, None, None)
        action.graphiti.remove_episode = AsyncMock(
            side_effect=[None, Exception("DB connection lost")]
        )

        # When
        await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then: partial success (deleted_count > 0) still shows success status
        emitted_statuses = [
            call.args[0]["data"]["description"]
            for call in event_emitter.call_args_list
            if call.args[0].get("type") == "status"
        ]
        assert any("Episode deleted" in s for s in emitted_statuses)

    async def test_should_report_failure_when_all_deletions_fail(self, action, user, event_emitter, event_call):
        # Given: DB returns 1 episode but deletion fails
        message_id = "msg-allfail-1"
        body = {
            "messages": [
                {"role": "user", "content": "Hello", "id": message_id},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        record = {
            "uuid": "ep-allfail-1",
            "name": "episode_fail",
            "group_id": "test-user-id",
            "source_description": f"20250101T000000Z_Chat_chat1_message_{message_id}",
            "content": "Content to delete",
            "source": "message",
            "created_at": "2025-01-01T00:00:00Z",
            "valid_at": "2025-01-01T00:00:00Z",
            "entity_edges": [],
        }
        action.graphiti.driver.execute_query.return_value = ([record], None, None)
        action.graphiti.remove_episode = AsyncMock(side_effect=Exception("DB error"))

        # When
        await action.action(body, __user__=user, __event_emitter__=event_emitter, __event_call__=event_call)

        # Then: failure status shown (deleted_count == 0)
        emitted_statuses = [
            call.args[0]["data"]["description"]
            for call in event_emitter.call_args_list
            if call.args[0].get("type") == "status"
        ]
        assert any("Failed to delete episode" in s for s in emitted_statuses)
