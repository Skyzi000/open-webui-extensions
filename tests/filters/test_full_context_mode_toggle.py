"""Tests for functions/filter/full_context_mode_toggle.py."""

from functions.filter.full_context_mode_toggle import Filter


def test_inlet_marks_both_payload_shapes_when_they_are_distinct():
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

    result = f.inlet(body)

    assert result["files"][0]["context"] == "full"
    assert result["metadata"]["files"][0]["context"] == "full"
    assert result["metadata"]["files"][0]["id"] == "meta-file"


def test_inlet_skips_auto_injected_knowledge_entries():
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
            "parent_message": {
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

    result = f.inlet(body)

    assert result["files"][0]["context"] == "full"
    assert "context" not in result["files"][1]
    assert "context" not in result["files"][2]
    assert "context" not in result["files"][3]


def test_inlet_allows_explicit_attachment_even_when_same_file_is_model_knowledge():
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

    result = f.inlet(body)

    assert result["files"][0]["context"] == "full"


def test_inlet_marks_explicit_legacy_and_collection_knowledge_items():
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
            "parent_message": {
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

    result = f.inlet(body)

    assert result["files"][0]["context"] == "full"
    assert result["files"][1]["context"] == "full"


def test_inlet_treats_files_as_explicit_when_parent_message_is_missing():
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

    result = f.inlet(body)

    assert result["files"][0]["context"] == "full"
