"""Exercise generated agent loops with real Core result/provider conversion."""

import copy
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


IMAGE = "data:image/png;base64,aGVsbG8="
STORED_IMAGE = "/api/v1/files/image/content"
FETCHED_IMAGE = "data:image/png;base64,c3RvcmVk"


def make_user():
    from open_webui.models.users import UserModel

    return UserModel(
        id="image-user", email="image@example.com", name="Image User", role="user",
        last_active_at=0, updated_at=0, created_at=0,
    )


def image_tools():
    async def screenshot():
        return {"result": {"screenshot": IMAGE}}

    async def text():
        return "text result"

    return {
        name: {
            "callable": fn, "type": "tool", "tool_id": "images",
            "spec": {"name": name, "parameters": {"type": "object", "properties": {}}},
        }
        for name, fn in (("screenshot", screenshot), ("text", text))
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name", [
    "sub_agent", "llm_review", "magi_decision_support", "multi_model_council",
])
@pytest.mark.parametrize("max_iterations", [1, 2])
@pytest.mark.parametrize("recursive_images", [True, False])
async def test_images_reach_next_and_final_provider_requests(monkeypatch, module_name, max_iterations, recursive_images):
    from open_webui.routers.openai import convert_responses_result, convert_to_responses_payload
    from open_webui.utils import chat, middleware

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_core_process_tool_result", middleware.process_tool_result)
    if not recursive_images:
        # Older Core (e.g. v0.8.12) already returns direct data images in files,
        # without the v0.11.4 recursive extractor. Keep that boundary explicit.
        monkeypatch.delattr(middleware, "extract_base64_images")
        monkeypatch.setattr(module, "_core_process_tool_result", AsyncMock(side_effect=[
            ("", [{"type": "image", "url": IMAGE}], []),
            ("text result", [], []),
        ]))
    sent = []

    async def complete(*, request, form_data, user, bypass_filter):
        sent.append(copy.deepcopy(form_data))
        if len(sent) == 1:
            return convert_responses_result({"output": [
                {"type": "function_call", "call_id": name, "name": name, "arguments": "{}"}
                for name in ("screenshot", "text")
            ]})
        return {"choices": [{"message": {"content": "done"}}]}

    monkeypatch.setattr(chat, "generate_chat_completion", complete)
    request = SimpleNamespace(state=SimpleNamespace(), app=SimpleNamespace(state=SimpleNamespace(MODELS={})))
    user = make_user()
    events = []

    async def emit(event):
        events.append(event)

    kwargs = dict(
        request=request, user=user, model_id="model", tools_dict=image_tools(),
        messages=[{"role": "user", "content": "Inspect the screenshot"}],
        max_iterations=max_iterations, apply_inlet_filters=False, event_emitter=emit,
        extra_params={"__request__": request, "__user__": user.model_dump(), "__metadata__": {}},
    )
    if module_name == "sub_agent":
        result = await module.run_sub_agent_loop(**kwargs)
    else:
        result = await module.run_agent_loop(**kwargs, agent_name="image agent")
    assert result == "done"
    assert len(sent) == 2
    messages = sent[1]["messages"]
    tool_index = next(i for i, message in enumerate(messages) if message["role"] == "tool")
    assert [message["tool_call_id"] for message in messages[tool_index:tool_index + 2]] == ["screenshot", "text"]
    if recursive_images:
        assert "[image]" in messages[tool_index]["content"]
    image_message = messages[tool_index + 2]
    assert image_message["role"] == "user"
    assert any(part.get("image_url", {}).get("url") == IMAGE for part in image_message["content"])
    assert not any(event["type"] == "files" for event in events)

    # This is the actual Core conversion used by Responses-configured providers.
    payload = convert_to_responses_payload(copy.deepcopy(sent[1]))
    assert any(
        part.get("type") == "input_image" and part.get("image_url") == IMAGE
        for item in payload["input"] for part in item.get("content", [])
        if isinstance(part, dict)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_emitter", [True, False])
async def test_parallel_images_match_direct_core_files(monkeypatch, with_emitter):
    import parallel_tools
    from open_webui.utils import middleware

    monkeypatch.setattr(parallel_tools, "_core_process_tool_result", middleware.process_tool_result)

    tools = image_tools()
    async def load_tools(**kwargs):
        return tools, {}

    monkeypatch.setattr(parallel_tools, "build_tools_dict", load_tools)
    user = make_user()
    metadata = {"chat_id": "image-chat", "message_id": "image-message"}
    wrapper_events = []
    stores, reads = [], []

    async def store(request, image, metadata, user):
        stores.append(image)
        return STORED_IMAGE

    async def read(url, *, user):
        reads.append(url)
        return IMAGE

    monkeypatch.setattr(middleware, "get_file_url_from_base64", store)
    monkeypatch.setattr(middleware, "get_image_base64_from_url", read)

    async def emit(event):
        wrapper_events.append(event)

    result = await parallel_tools.Tools().run_tools_parallel(
        tool_calls=[parallel_tools.ToolCallItem(name="screenshot")],
        __request__=SimpleNamespace(), __user__=user.model_dump(), __metadata__=metadata,
        __event_emitter__=emit if with_emitter else None,
    )
    assert isinstance(result, dict)
    assert not wrapper_events and not stores and not reads

    _, direct_files, _ = await middleware.process_tool_result(
        None, "screenshot", await tools["screenshot"]["callable"](), "tool", metadata=metadata, user=user,
    )
    _, parallel_files, _ = await middleware.process_tool_result(
        None, "run_tools_parallel", result, "tool", metadata=metadata, user=user,
    )
    assert direct_files == parallel_files == [{"type": "image", "url": IMAGE}]
    assert not stores and not reads


@pytest.mark.asyncio
@pytest.mark.parametrize("with_image", [True, False])
async def test_parallel_keeps_json_image_text_out_of_outer_core_extraction(monkeypatch, with_image):
    import parallel_tools
    from open_webui.utils import middleware

    other_image = "data:image/png;base64,Yg=="
    image_text = json.dumps({"image": other_image})
    ordinary_result = {"status": "ok", "rows": [{"value": 1}]}

    async def json_image_text():
        return image_text

    async def ordinary():
        return ordinary_result

    tools = image_tools()
    tools.update({
        name: {
            "callable": fn, "type": "tool",
            "spec": {"name": name, "parameters": {"type": "object", "properties": {}}},
        }
        for name, fn in (("json_image_text", json_image_text), ("ordinary", ordinary))
    })
    monkeypatch.setattr(parallel_tools, "_core_process_tool_result", middleware.process_tool_result)
    monkeypatch.setattr(parallel_tools, "build_tools_dict", AsyncMock(return_value=(tools, {})))
    user = make_user()

    # Core treats a JSON string as text, even if an image URI is inside it.
    direct_content, direct_files, _ = await middleware.process_tool_result(
        None, "json_image_text", image_text, "tool", metadata={}, user=user,
    )
    assert direct_content == image_text and direct_files == []

    names = (["screenshot"] if with_image else []) + ["json_image_text", "ordinary"]
    result = await parallel_tools.Tools().run_tools_parallel(
        [parallel_tools.ToolCallItem(name=name) for name in names],
        __request__=SimpleNamespace(), __user__=user.model_dump(), __metadata__={},
    )
    assert isinstance(result, dict if with_image else str)
    payload = result if with_image else json.loads(result)
    results = {item["tool_name"]: item["result"] for item in payload["results"]}
    assert results["ordinary"] == ordinary_result
    if with_image:
        assert results["json_image_text"] == image_text
    else:
        # Preserve the established JSON formatting when no images need forwarding.
        assert results["json_image_text"] == json.loads(image_text)

    _, outer_files, _ = await middleware.process_tool_result(
        None, "run_tools_parallel", copy.deepcopy(result), "tool", metadata={}, user=user,
    )
    assert outer_files == ([{"type": "image", "url": IMAGE}] if with_image else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["shared", "parallel_single", "parallel_batch"])
@pytest.mark.parametrize("with_emitter", [True, False])
@pytest.mark.parametrize("core_version, model_sees_saved_image", [("0.11.4", False), ("0.12.0", True)])
async def test_mcp_saved_images_reach_the_model_from_core_0_12(
    monkeypatch, executor, with_emitter, core_version, model_sees_saved_image,
):
    import open_webui.env
    import sub_agent
    import parallel_tools
    from open_webui.utils import files as core_files, middleware

    monkeypatch.setattr(open_webui.env, "VERSION", core_version)

    user = make_user()
    events = []
    uploads = []
    display_files = [{"type": "image", "url": STORED_IMAGE}, {"type": "audio", "url": "/api/v1/files/audio/content"}]

    async def store(request, data, metadata, user):
        uploads.append(data)
        return STORED_IMAGE if data.startswith("data:image/") else display_files[1]["url"]

    async def mixed_result():
        return [
            {"type": "image", "mimeType": "image/png", "data": "aGVsbG8="},
            {"type": "resource", "resource": {"mimeType": "image/png", "blob": "aGVsbG8="}},
            {"type": "audio", "mimeType": "audio/wav", "data": "YQ=="},
            {"type": "text", "text": "Screenshot and recording"},
        ]

    async def emit(event):
        events.append(event)

    tools = image_tools()
    tools["screenshot"].update(callable=mixed_result, type="mcp")
    monkeypatch.setattr(middleware, "get_file_url_from_base64", store)
    read_image = AsyncMock(return_value=FETCHED_IMAGE)
    monkeypatch.setattr(core_files, "get_image_base64_from_url", read_image)
    monkeypatch.setattr(sub_agent, "_core_process_tool_result", middleware.process_tool_result)
    monkeypatch.setattr(parallel_tools, "_core_process_tool_result", middleware.process_tool_result)
    extra = {"__request__": SimpleNamespace(), "__user__": user.model_dump(), "__metadata__": {}}
    emitter = emit if with_emitter else None
    if executor == "shared":
        result = await sub_agent.execute_tool_call(
            {"id": "call", "function": {"name": "screenshot", "arguments": "{}"}}, tools, extra, event_emitter=emitter,
        )
    elif executor == "parallel_single":
        result = await parallel_tools.execute_single_tool("screenshot", {}, tools, extra, event_emitter=emitter)
    else:
        monkeypatch.setattr(parallel_tools, "build_tools_dict", AsyncMock(return_value=(tools, {})))
        payload = await parallel_tools.Tools().run_tools_parallel(
            [parallel_tools.ToolCallItem(name="screenshot")], __request__=extra["__request__"], __user__=extra["__user__"], __metadata__={}, __event_emitter__=emitter,
        )
        result = payload["results"][0]
    assert result["images"] == ([FETCHED_IMAGE, IMAGE] if model_sees_saved_image else [IMAGE])
    assert len(uploads) == 2
    assert [call.args for call in read_image.await_args_list] == [(STORED_IMAGE,)] * model_sees_saved_image
    file_events = [event for event in events if event["type"] == "files"]
    assert [event["data"]["files"] for event in file_events] == ([display_files] if with_emitter else [])
    if executor != "shared" and not with_emitter:
        assert result["files"] == display_files


@pytest.mark.parametrize("extractor_present", [False, True])
@pytest.mark.parametrize("with_emitter", [True, False])
async def test_older_core_keeps_images_on_existing_files_path(monkeypatch, extractor_present, with_emitter):
    import parallel_tools
    from open_webui.utils import middleware

    if extractor_present:
        monkeypatch.setattr(middleware, "extract_base64_images", None)
    else:
        monkeypatch.delattr(middleware, "extract_base64_images")

    files = [{"type": "image", "url": IMAGE}]
    monkeypatch.setattr(parallel_tools, "_core_process_tool_result", AsyncMock(return_value=("", files, [])))
    monkeypatch.setattr(parallel_tools, "build_tools_dict", AsyncMock(return_value=(image_tools(), {})))
    emitter = AsyncMock() if with_emitter else None
    result = await parallel_tools.Tools().run_tools_parallel(
        [parallel_tools.ToolCallItem(name="screenshot")],
        __request__=SimpleNamespace(), __user__=make_user().model_dump(),
        __metadata__={}, __event_emitter__=emitter,
    )
    assert isinstance(result, str)
    item = json.loads(result)["results"][0]
    assert "images" not in item
    if with_emitter:
        assert "files" not in item
        emitter.assert_awaited_once_with({"type": "files", "data": {"files": files}})
    else:
        assert item["files"] == files
    assert files == [{"type": "image", "url": IMAGE}]


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name", ["sub_agent", "parallel_tools"])
@pytest.mark.parametrize("name", ["search_web", "view_file", "fetch_url"])
async def test_citations_follow_core_source_tools(monkeypatch, module_name, name):
    from open_webui.utils import middleware

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_core_process_tool_result", middleware.process_tool_result)
    responses = {
        "search_web": [{"title": "Page", "link": "https://example.com/page", "snippet": "Result"}],
        "view_file": {"id": "file", "filename": "File.txt", "content": "File text"},
        "fetch_url": "Page content",
    }

    async def tool(**kwargs):
        return json.dumps(responses[name])

    events = []

    async def emit(event):
        events.append(event)

    tools = {name: {"callable": tool, "spec": {"name": name, "parameters": {"properties": {"url": {"type": "string"}}}}}}
    params = {"url": "https://example.com/page"}
    if module_name == "sub_agent":
        await module.execute_tool_call({"id": "call", "function": {"name": name, "arguments": json.dumps(params)}}, tools, {}, event_emitter=emit)
    else:
        await module.execute_single_tool(name, params, tools, {}, event_emitter=emit)
    sources = [event["data"] for event in events if event["type"] == "source"]
    assert len(sources) == (0 if name == "search_web" else 1)
    if name == "view_file":
        assert sources[0]["metadata"][0]["file_id"] == "file"
    elif name == "fetch_url":
        assert sources[0]["metadata"][0]["url"] == params["url"]
