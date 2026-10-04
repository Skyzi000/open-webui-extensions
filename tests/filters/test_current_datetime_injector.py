"""Tests for functions/filter/current_datetime_injector.py extension."""

import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add filter directory to path
filter_dir = Path(__file__).parent.parent.parent / "functions" / "filter"
sys.path.insert(0, str(filter_dir))

USER = {"valves": {"timezone": "UTC"}}
DATETIME_TEXT = "# Current DateTime\n2026-10-04 (Sunday) 09:05:00 (UTC)"
TAGGED_DATETIME = f"<SYSTEM_CONTEXT>\n{DATETIME_TEXT}\n</SYSTEM_CONTEXT>"
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 4, 9, 5, 0, tzinfo=tz or timezone.utc)


@pytest.fixture
def injector(monkeypatch):
    import current_datetime_injector

    monkeypatch.setattr(current_datetime_injector, "datetime", FixedDateTime)
    return current_datetime_injector.Filter()


class TestCurrentDateTimeInjectorPlacement:
    """Test where and with which role the date/time message is injected."""

    def test_default_appends_tagged_context_to_latest_user_message(self, injector):
        """Verify the default adds no message and tags the text appended to the latest user message."""
        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "latest"},
            ]
        }

        messages = injector.inlet(body, __user__=USER)["messages"]

        assert messages == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": f"latest\n\n{TAGGED_DATETIME}"},
        ]

    def test_default_appends_to_first_text_part_without_mutating_the_original(self, injector):
        """Verify list content gets the text in its first text part, which is the one Open WebUI reads as the prompt."""
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        latest = {
            "role": "user",
            "content": [image, {"type": "text", "text": "look"}, {"type": "text", "text": "second"}],
        }
        original = copy.deepcopy(latest)

        messages = injector.inlet({"messages": [latest]}, __user__=USER)["messages"]

        assert messages == [
            {
                "role": "user",
                "content": [
                    image,
                    {"type": "text", "text": f"look\n\n{TAGGED_DATETIME}"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        assert latest == original

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("", TAGGED_DATETIME),
            (
                [{"type": "text", "text": ""}, IMAGE],
                [{"type": "text", "text": TAGGED_DATETIME}, IMAGE],
            ),
            ([IMAGE], [IMAGE, {"type": "text", "text": TAGGED_DATETIME}]),
        ],
    )
    def test_default_injects_into_message_without_text(self, injector, content, expected):
        """Verify an image-only message still gets the context, as its only text."""
        body = {"messages": [{"role": "user", "content": content}]}

        assert injector.inlet(body, __user__=USER)["messages"] == [{"role": "user", "content": expected}]

    @pytest.mark.parametrize(
        "text",
        [
            "/skills:create",
            "  /Skills:Create summarize\nthis chat  ",
            "/skills:create\tfrom notes",
            "/skills:createx",
            "please /skills:create",
        ],
    )
    def test_default_skips_exactly_what_core_handles_as_skills_create(self, injector, text):
        """Verify injection is skipped for, and only for, text that Open WebUI rewrites as a /skills:create command."""
        from open_webui.utils.skills import apply_skills_create_prompt

        messages = injector.inlet({"messages": [{"role": "user", "content": text}]}, __user__=USER)["messages"]

        skipped = messages == [{"role": "user", "content": text}]
        is_command = apply_skills_create_prompt([{"role": "user", "content": text}], allowed=True)
        assert skipped == is_command

    def test_system_role_injects_untagged_message_before_latest_user(self, injector):
        """Verify system role keeps the latest user message last."""
        injector.valves.message_role = "system"
        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "latest"},
            ]
        }

        messages = injector.inlet(body, __user__=USER)["messages"]

        assert messages == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "system", "content": DATETIME_TEXT},
            {"role": "user", "content": "latest"},
        ]

    def test_system_after_injects_untagged_message_after_latest_user(self, injector):
        """Verify system_after lands right after the latest user message."""
        injector.valves.message_role = "system_after"
        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "latest"},
            ]
        }

        messages = injector.inlet(body, __user__=USER)["messages"]

        assert messages == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "latest"},
            {"role": "system", "content": DATETIME_TEXT},
        ]

    def test_system_after_stays_adjacent_to_user_when_assistant_follows(self, injector):
        """Verify system_after goes between the user message and a trailing assistant turn."""
        injector.valves.message_role = "system_after"
        body = {
            "messages": [
                {"role": "user", "content": "latest"},
                {"role": "assistant", "content": "partial answer"},
            ]
        }

        messages = injector.inlet(body, __user__=USER)["messages"]

        assert messages == [
            {"role": "user", "content": "latest"},
            {"role": "system", "content": DATETIME_TEXT},
            {"role": "assistant", "content": "partial answer"},
        ]
