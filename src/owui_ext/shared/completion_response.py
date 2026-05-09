"""Response coercion helper shared by owui_ext tool plugins.

Open WebUI's ``generate_chat_completion`` may return:

- ``dict`` on success (the OpenAI-shaped chat completion).
- ``starlette.responses.JSONResponse`` when the upstream provider
  returns an HTTP error with a JSON body.
- ``starlette.responses.PlainTextResponse`` when the upstream
  provider returns an HTTP error with a non-JSON body (HTML error
  pages from gateways/CDNs, plain-text rate-limit responses, etc.).
- Other ``starlette.responses.Response`` subclasses for less-common
  cases.

Plugins that drive their own agent loop need to surface the actual
upstream error to the parent LLM. Returning a generic
``Unexpected response type: <PlainTextResponse>`` hides the cause and
encourages the parent to retry blindly, which is exactly the failure
mode reported as Sub-Agent issue #11.

``format_chat_completion_error`` returns ``None`` for the success
path (caller proceeds to read ``response['choices']``) and a string
for every error path (caller returns the string up its own stack).
"""

import json
from typing import Any, Optional

from starlette.responses import JSONResponse, Response

_RESPONSE_BODY_PREVIEW_CHARS = 1024


def _truncate_preview(text: str) -> str:
    if len(text) > _RESPONSE_BODY_PREVIEW_CHARS:
        return text[:_RESPONSE_BODY_PREVIEW_CHARS] + "...[truncated]"
    return text


def _decode_response_body(response: Response) -> str:
    body = getattr(response, "body", None)
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        try:
            text = bytes(body).decode("utf-8", errors="replace")
        except Exception:
            text = repr(body)
    else:
        text = str(body)
    return _truncate_preview(text.strip())


def _extract_json_response_error(response: JSONResponse) -> str:
    status = getattr(response, "status_code", "unknown")
    try:
        error_data = json.loads(bytes(response.body).decode("utf-8"))
    except Exception:
        return f"API error (status {status}): Failed to parse response"
    if isinstance(error_data, dict):
        error_field = error_data.get("error")
        if isinstance(error_field, dict):
            msg = error_field.get("message")
            if isinstance(msg, str) and msg:
                return f"API error: {_truncate_preview(msg)}"
            return f"API error: {_truncate_preview(str(error_data))}"
        if isinstance(error_field, str) and error_field:
            return f"API error: {_truncate_preview(error_field)}"
        msg = error_data.get("message")
        if isinstance(msg, str) and msg:
            return f"API error: {_truncate_preview(msg)}"
        return f"API error: {_truncate_preview(str(error_data))}"
    return f"API error: {_truncate_preview(str(error_data))}"


def format_chat_completion_error(response: Any) -> Optional[str]:
    """Classify a ``generate_chat_completion`` response.

    Returns:
        ``None`` when ``response`` is a ``dict`` (success path); the
        caller should proceed to read ``response['choices']``.

        ``str`` describing the upstream failure for any other shape.
        ``JSONResponse`` bodies are unwrapped to surface the provider's
        error message; other ``Response`` subclasses (notably
        ``PlainTextResponse`` returned by Open WebUI core when the
        provider replies with non-JSON 400+ content) are reported with
        their status code and a truncated body preview so the caller
        can show the real cause to the parent loop.
    """
    if isinstance(response, dict):
        return None
    if isinstance(response, JSONResponse):
        return _extract_json_response_error(response)
    if isinstance(response, Response):
        status = getattr(response, "status_code", "unknown")
        body_text = _decode_response_body(response)
        type_name = type(response).__name__
        if body_text:
            return f"API error (status {status}, {type_name}): {body_text}"
        return f"API error (status {status}, {type_name}): empty body"
    return f"Unexpected response type: {type(response).__name__}"
