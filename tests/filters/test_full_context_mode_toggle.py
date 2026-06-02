"""Tests for functions/filter/full_context_mode_toggle.py."""

import sys
import types

from functions.filter.full_context_mode_toggle import Filter, folder_knowledge_file_ids


async def test_inlet_marks_both_payload_shapes_when_they_are_distinct():
    f = Filter()
    body = {
        "files": [
            {"id": "body-file", "type": "file", "name": "body.txt"},
        ],
        "metadata": {
            "files": [
                {"id": "meta-file", "type": "file", "name": "meta.txt"},
            ]
        },
    }

    result = await f.inlet(body)

    assert result["files"][0]["context"] == "full"
    assert result["metadata"]["files"][0]["context"] == "full"
    assert result["metadata"]["files"][0]["id"] == "meta-file"


async def test_inlet_skips_auto_injected_knowledge_entries():
    f = Filter()
    body = {
        "files": [
            {"id": "attached", "type": "file", "name": "user.txt"},
            {"id": "model-file", "type": "file", "name": "model.txt"},
            {"id": "collection-1", "name": "kb", "legacy": True},
            {
                "id": "collection-2",
                "type": "collection",
                "collection_names": ["c1", "c2"],
                "legacy": True,
            },
        ],
        "metadata": {
            "user_message": {
                "files": [
                    {"id": "attached", "type": "file", "name": "user.txt"},
                ],
            },
            "model": {
                "info": {
                    "meta": {
                        "knowledge": [
                            {"id": "model-file", "type": "file", "name": "model.txt"},
                            {"id": "collection-1", "name": "kb", "legacy": True},
                            {
                                "id": "collection-2",
                                "type": "collection",
                                "collection_names": ["c1", "c2"],
                                "legacy": True,
                            },
                        ]
                    }
                }
            }
        },
    }

    result = await f.inlet(body)

    assert result["files"][0]["context"] == "full"
    assert "context" not in result["files"][1]
    assert "context" not in result["files"][2]
    assert "context" not in result["files"][3]


async def test_inlet_allows_explicit_attachment_even_when_same_file_is_model_knowledge():
    f = Filter()
    body = {
        "files": [
            {"id": "shared-file", "type": "file", "name": "shared.txt"},
        ],
        "metadata": {
            "user_message": {
                "files": [
                    {"id": "shared-file", "type": "file", "name": "shared.txt"},
                ],
            },
            "model": {
                "info": {
                    "meta": {
                        "knowledge": [
                            {"id": "shared-file", "type": "file", "name": "shared.txt"}
                        ]
                    }
                }
            },
        },
    }

    result = await f.inlet(body)

    assert result["files"][0]["context"] == "full"


async def test_inlet_marks_explicit_legacy_and_collection_knowledge_items():
    f = Filter()
    body = {
        "files": [
            {"id": "legacy-kb", "name": "Legacy KB", "legacy": True},
            {
                "id": "collection-kb",
                "type": "collection",
                "collection_names": ["c1", "c2"],
                "legacy": True,
            },
        ],
        "metadata": {
            "user_message": {
                "files": [
                    {"id": "legacy-kb", "name": "Legacy KB", "legacy": True},
                    {
                        "id": "collection-kb",
                        "type": "collection",
                        "collection_names": ["c1", "c2"],
                        "legacy": True,
                    },
                ],
            },
            "model": {
                "info": {
                    "meta": {
                        "knowledge": [
                            {"id": "legacy-kb", "name": "Legacy KB", "legacy": True},
                            {
                                "id": "collection-kb",
                                "type": "collection",
                                "collection_names": ["c1", "c2"],
                                "legacy": True,
                            },
                        ]
                    }
                }
            },
        },
    }

    result = await f.inlet(body)

    assert result["files"][0]["context"] == "full"
    assert result["files"][1]["context"] == "full"


async def test_inlet_treats_files_as_explicit_when_user_message_is_missing():
    f = Filter()
    body = {
        "files": [
            {"id": "shared-file", "type": "file", "name": "shared.txt"},
        ],
        "metadata": {
            "model": {
                "info": {
                    "meta": {
                        "knowledge": [
                            {"id": "shared-file", "type": "file", "name": "shared.txt"}
                        ]
                    }
                }
            },
        },
    }

    result = await f.inlet(body)

    assert result["files"][0]["context"] == "full"


async def test_inlet_treats_missing_user_message_files_as_empty_attachments():
    f = Filter()
    body = {
        "files": [
            {"id": "model-file", "type": "file", "name": "model.txt"},
            {"id": "collection-1", "name": "kb", "legacy": True},
        ],
        "metadata": {
            "user_message": {
                "id": "user-message-id",
                "role": "user",
                "content": "Hello",
            },
            "model": {
                "info": {
                    "meta": {
                        "knowledge": [
                            {"id": "model-file", "type": "file", "name": "model.txt"},
                            {"id": "collection-1", "name": "kb", "legacy": True},
                        ]
                    }
                }
            },
        },
    }

    result = await f.inlet(body)

    assert "context" not in result["files"][0]
    assert "context" not in result["files"][1]


async def test_inlet_supports_legacy_parent_message_key():
    f = Filter()
    body = {
        "files": [
            {"id": "shared-file", "type": "file", "name": "shared.txt"},
        ],
        "metadata": {
            "parent_message": {
                "files": [
                    {"id": "shared-file", "type": "file", "name": "shared.txt"},
                ],
            },
            "model": {
                "info": {
                    "meta": {
                        "knowledge": [
                            {"id": "shared-file", "type": "file", "name": "shared.txt"}
                        ]
                    }
                }
            },
        },
    }

    result = await f.inlet(body)

    assert result["files"][0]["context"] == "full"


async def test_folder_knowledge_skips_channel_chat_lookup(monkeypatch):
    calls = []

    class FakeChats:
        @staticmethod
        async def get_chat_folder_id(chat_id, user_id):
            calls.append((chat_id, user_id))
            return "folder-id"

    chats_module = types.ModuleType("open_webui.models.chats")
    chats_module.Chats = FakeChats
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_module)

    result = await folder_knowledge_file_ids(
        {"chat_id": "channel:abc123"},
        {"id": "user-1"},
    )

    assert result == set()
    assert calls == []
