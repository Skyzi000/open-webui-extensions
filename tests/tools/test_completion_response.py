"""Unit tests for the shared ``completion_response`` helper.

These exercise ``format_chat_completion_error`` directly against the
source module, since every plugin inlines the same code at build time.
"""

from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse

from owui_ext.shared.completion_response import (
    _RESPONSE_BODY_PREVIEW_CHARS,
    format_chat_completion_error,
)


def test_dict_response_returns_none():
    assert format_chat_completion_error({"choices": []}) is None


def test_json_response_with_nested_error_message():
    response = JSONResponse(
        status_code=400,
        content={"error": {"message": "Bad request payload"}},
    )
    assert format_chat_completion_error(response) == "API error: Bad request payload"


def test_json_response_with_string_error_field():
    response = JSONResponse(
        status_code=502,
        content={"error": "upstream timeout"},
    )
    assert format_chat_completion_error(response) == "API error: upstream timeout"


def test_json_response_with_top_level_message():
    response = JSONResponse(
        status_code=429,
        content={"message": "rate limited"},
    )
    assert format_chat_completion_error(response) == "API error: rate limited"


def test_json_response_with_unparseable_body_falls_back_to_status():
    response = JSONResponse(status_code=500, content={})
    response.body = b"not-json"
    result = format_chat_completion_error(response)
    assert result == "API error (status 500): Failed to parse response"


def test_json_response_truncates_long_message_field():
    long_msg = "x" * (_RESPONSE_BODY_PREVIEW_CHARS + 200)
    response = JSONResponse(
        status_code=400,
        content={"error": {"message": long_msg}},
    )
    result = format_chat_completion_error(response)
    assert result.startswith("API error: ")
    assert result.endswith("...[truncated]")
    body_part = result.split(": ", 1)[1]
    assert len(body_part) == _RESPONSE_BODY_PREVIEW_CHARS + len("...[truncated]")


def test_json_response_truncates_long_string_error_field():
    long_msg = "y" * (_RESPONSE_BODY_PREVIEW_CHARS + 50)
    response = JSONResponse(status_code=502, content={"error": long_msg})
    result = format_chat_completion_error(response)
    assert result.endswith("...[truncated]")


def test_json_response_truncates_long_top_level_message():
    long_msg = "z" * (_RESPONSE_BODY_PREVIEW_CHARS + 50)
    response = JSONResponse(status_code=429, content={"message": long_msg})
    result = format_chat_completion_error(response)
    assert result.endswith("...[truncated]")


def test_json_response_truncates_unrecognized_dict_fallback():
    # No known message/error fields — falls back to str(error_data).
    long_value = "q" * (_RESPONSE_BODY_PREVIEW_CHARS + 50)
    response = JSONResponse(status_code=400, content={"unknown_field": long_value})
    result = format_chat_completion_error(response)
    assert result.startswith("API error: ")
    assert result.endswith("...[truncated]")


def test_plain_text_response_includes_status_type_and_body():
    response = PlainTextResponse(
        status_code=504,
        content="Gateway Timeout",
    )
    result = format_chat_completion_error(response)
    assert result == "API error (status 504, PlainTextResponse): Gateway Timeout"


def test_plain_text_response_with_empty_body_uses_empty_marker():
    response = PlainTextResponse(status_code=502, content="")
    result = format_chat_completion_error(response)
    assert result == "API error (status 502, PlainTextResponse): empty body"


def test_plain_text_response_truncates_long_body():
    long_body = "x" * (_RESPONSE_BODY_PREVIEW_CHARS + 200)
    response = PlainTextResponse(status_code=502, content=long_body)
    result = format_chat_completion_error(response)
    assert result.startswith("API error (status 502, PlainTextResponse): ")
    assert result.endswith("...[truncated]")
    body_part = result.split(": ", 1)[1]
    assert len(body_part) == _RESPONSE_BODY_PREVIEW_CHARS + len("...[truncated]")


def test_html_response_is_handled_via_response_branch():
    response = HTMLResponse(status_code=524, content="<html>cf timeout</html>")
    result = format_chat_completion_error(response)
    assert result == "API error (status 524, HTMLResponse): <html>cf timeout</html>"


def test_unknown_type_yields_unexpected_message():
    assert format_chat_completion_error("a string") == (
        "Unexpected response type: str"
    )
    assert format_chat_completion_error(None) == (
        "Unexpected response type: NoneType"
    )
    assert format_chat_completion_error([1, 2, 3]) == (
        "Unexpected response type: list"
    )
