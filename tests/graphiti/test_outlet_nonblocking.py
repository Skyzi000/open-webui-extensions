"""Tests for outlet() non-blocking behavior and background task management."""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add graphiti filter directory to path
graphiti_filter_dir = Path(__file__).parent.parent.parent / "graphiti" / "functions" / "filter"
sys.path.insert(0, str(graphiti_filter_dir))

from graphiti_memory import Filter


@pytest.fixture
def filter_instance():
    f = Filter()
    f.valves.read_only = False
    f.valves.debug_print = False
    return f


@pytest.fixture
def mock_event_emitter():
    return AsyncMock()


@pytest.fixture
def mock_user():
    return {
        "id": "test-user-id",
        "email": "test@example.com",
        "name": "Test User",
        "role": "user",
        "valves": {},
    }


@pytest.fixture
def mock_body():
    return {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
    }


@pytest.fixture
def mock_metadata():
    return {"chat_id": "test-chat-id", "message_id": "test-message-id"}


class TestLaunchBackgroundTask:
    """Tests for _launch_background_task() helper."""

    @pytest.mark.asyncio
    async def test_task_added_to_set(self, filter_instance):
        """Task should be tracked in _background_tasks while running."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_coro():
            started.set()
            await release.wait()

        task = filter_instance._launch_background_task(slow_coro())
        await started.wait()

        assert task in filter_instance._background_tasks

        release.set()
        await task

    @pytest.mark.asyncio
    async def test_task_removed_on_completion(self, filter_instance):
        """Task should be removed from _background_tasks after completion."""

        async def quick_coro():
            pass

        task = filter_instance._launch_background_task(quick_coro())
        await task

        # done_callback runs after await returns
        await asyncio.sleep(0)
        assert task not in filter_instance._background_tasks

    @pytest.mark.asyncio
    async def test_task_removed_on_exception(self, filter_instance):
        """Task should be removed from _background_tasks even if it raises."""

        async def failing_coro():
            raise ValueError("test error")

        task = filter_instance._launch_background_task(failing_coro())

        with pytest.raises(ValueError, match="test error"):
            await task

        await asyncio.sleep(0)
        assert task not in filter_instance._background_tasks


class TestOutletNonBlocking:
    """Tests for outlet() returning immediately without blocking."""

    @pytest.mark.asyncio
    async def test_outlet_returns_body_immediately(
        self, filter_instance, mock_event_emitter, mock_user, mock_body, mock_metadata
    ):
        """outlet() should return the body dict without awaiting background work."""
        background_started = asyncio.Event()
        background_release = asyncio.Event()

        async def blocking_save(**kwargs):
            background_started.set()
            await background_release.wait()

        with patch.object(filter_instance, "_save_memory_background", side_effect=blocking_save):
            result = await filter_instance.outlet(
                body=mock_body,
                __event_emitter__=mock_event_emitter,
                __user__=mock_user,
                __metadata__=mock_metadata,
                __id__="test-filter-id",
            )

        # outlet() returned immediately
        assert result is mock_body

        # Clean up background task
        background_release.set()
        for task in list(filter_instance._background_tasks):
            try:
                await task
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_outlet_launches_background_task(
        self, filter_instance, mock_event_emitter, mock_user, mock_body, mock_metadata
    ):
        """outlet() should create a background task for memory saving."""
        with patch.object(filter_instance, "_save_memory_background", new_callable=AsyncMock) as mock_save:
            await filter_instance.outlet(
                body=mock_body,
                __event_emitter__=mock_event_emitter,
                __user__=mock_user,
                __metadata__=mock_metadata,
                __id__="test-filter-id",
            )

        # Wait for background task to complete
        for task in list(filter_instance._background_tasks):
            await task
        await asyncio.sleep(0)

        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_background_receives_body_snapshot(
        self, filter_instance, mock_event_emitter, mock_user, mock_metadata
    ):
        """Background task should receive a deep copy of body, immune to later mutations."""
        body = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        captured_body = None

        async def capture_save(**kwargs):
            nonlocal captured_body
            captured_body = kwargs["body"]

        with patch.object(filter_instance, "_save_memory_background", side_effect=capture_save):
            await filter_instance.outlet(
                body=body,
                __event_emitter__=mock_event_emitter,
                __user__=mock_user,
                __metadata__=mock_metadata,
                __id__="test-filter-id",
            )

        # Mutate the original body after outlet() has returned
        body["messages"].append({"role": "user", "content": "MUTATED"})
        body["model"] = "MUTATED-model"

        # Wait for background task
        for task in list(filter_instance._background_tasks):
            try:
                await task
            except Exception:
                pass

        # Background received a snapshot unaffected by post-outlet mutations
        assert captured_body is not None
        assert captured_body["model"] == "test-model"
        assert len(captured_body["messages"]) == 2
        assert captured_body["messages"][-1]["content"] == "Hi there"


class TestOutletEarlyReturns:
    """Tests for outlet() early return conditions (these should NOT launch background tasks)."""

    @pytest.mark.asyncio
    async def test_read_only_mode_skips_background(
        self, filter_instance, mock_event_emitter, mock_user, mock_body, mock_metadata
    ):
        """When read_only is enabled, outlet should return immediately without background task."""
        filter_instance.valves.read_only = True

        result = await filter_instance.outlet(
            body=mock_body,
            __event_emitter__=mock_event_emitter,
            __user__=mock_user,
            __metadata__=mock_metadata,
        )

        assert result is mock_body
        assert len(filter_instance._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_user_disabled_skips_background(
        self, filter_instance, mock_event_emitter, mock_body, mock_metadata
    ):
        """When user has disabled the feature, outlet should return without background task."""
        disabled_user = {
            "id": "test-user-id",
            "email": "test@example.com",
            "name": "Test User",
            "role": "user",
            "valves": {"enabled": False},
        }

        result = await filter_instance.outlet(
            body=mock_body,
            __event_emitter__=mock_event_emitter,
            __user__=disabled_user,
            __metadata__=mock_metadata,
        )

        assert result is mock_body
        assert len(filter_instance._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_empty_messages_skips_background(
        self, filter_instance, mock_event_emitter, mock_user, mock_metadata
    ):
        """When body has no messages, outlet should return without background task."""
        empty_body = {"model": "test-model", "messages": []}

        result = await filter_instance.outlet(
            body=empty_body,
            __event_emitter__=mock_event_emitter,
            __user__=mock_user,
            __metadata__=mock_metadata,
        )

        assert result is empty_body
        assert len(filter_instance._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_no_user_skips_background(
        self, filter_instance, mock_event_emitter, mock_body, mock_metadata
    ):
        """When __user__ is None, outlet should return without background task."""
        result = await filter_instance.outlet(
            body=mock_body,
            __event_emitter__=mock_event_emitter,
            __user__=None,
            __metadata__=mock_metadata,
        )

        assert result is mock_body
        assert len(filter_instance._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_temporary_chat_skips_background(
        self, filter_instance, mock_event_emitter, mock_user, mock_body
    ):
        """When chat is temporary (local:*), outlet should return without background task."""
        temp_metadata = {"chat_id": "local:temp-123", "message_id": "msg-1"}

        result = await filter_instance.outlet(
            body=mock_body,
            __event_emitter__=mock_event_emitter,
            __user__=mock_user,
            __metadata__=temp_metadata,
        )

        assert result is mock_body
        assert len(filter_instance._background_tasks) == 0


class TestSaveMemoryBackgroundErrorHandling:
    """Tests for _save_memory_background() error isolation."""

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(
        self, filter_instance, mock_event_emitter, mock_user, mock_metadata
    ):
        """Exceptions in _save_memory_background should be caught, not propagated."""
        body = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        user_valves = Filter.UserValves()
        user_valves.show_status = False

        with patch.object(
            filter_instance, "_save_memory_background_inner", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            # Should NOT raise
            await filter_instance._save_memory_background(
                body=body,
                __event_emitter__=mock_event_emitter,
                __user__=mock_user,
                __metadata__=mock_metadata,
                user_valves=user_valves,
                filter_id_for_prefix="graphiti_memory",
                reference_time=datetime.now(timezone.utc),
            )

    @pytest.mark.asyncio
    async def test_exception_emits_error_status(
        self, filter_instance, mock_event_emitter, mock_user, mock_metadata
    ):
        """When an exception occurs and show_status is enabled, an error status should be emitted."""
        body = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        user_valves = Filter.UserValves()
        user_valves.show_status = True

        with patch.object(
            filter_instance, "_save_memory_background_inner", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            await filter_instance._save_memory_background(
                body=body,
                __event_emitter__=mock_event_emitter,
                __user__=mock_user,
                __metadata__=mock_metadata,
                user_valves=user_valves,
                filter_id_for_prefix="graphiti_memory",
                reference_time=datetime.now(timezone.utc),
            )

        # Verify error status was emitted
        mock_event_emitter.assert_called_once()
        call_args = mock_event_emitter.call_args[0][0]
        assert call_args["type"] == "status"
        assert call_args["data"]["done"] is True


class TestBackgroundTasksInit:
    """Tests for _background_tasks initialization."""

    def test_background_tasks_initialized(self):
        """Filter should initialize _background_tasks as an empty set."""
        f = Filter()
        assert hasattr(f, "_background_tasks")
        assert isinstance(f._background_tasks, set)
        assert len(f._background_tasks) == 0
