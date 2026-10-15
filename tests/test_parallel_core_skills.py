"""Parallel Tools preserves the skill callable advertised by real Core."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from open_webui.tools import builtin

import parallel_tools

# Core 0.12.0+ registers skill tools for every model that keeps its skills switch on.
CORE_REGISTERS_SKILL_TOOLS = hasattr(builtin, "read_skill_file")


@pytest.fixture
def skill_context(monkeypatch):
    from open_webui.models.config import Config
    from open_webui.models.users import UserModel

    monkeypatch.setattr(Config, "get_many", AsyncMock(return_value={}))
    monkeypatch.setattr(Config, "get", AsyncMock(
        side_effect=lambda key, default=None: {} if key == "user.permissions" else default
    ))
    user = UserModel(
        id="skill-user", email="skill@example.com", name="Skill User", role="user",
        last_active_at=0, updated_at=0, created_at=0,
    ).model_dump()
    model = {"id": "model", "info": {"meta": {"builtinTools": {
        category: False for category in ("time", "user_input", "knowledge", "chats", "tasks")
    }}}}
    request = SimpleNamespace(
        state=SimpleNamespace(internal=False), body=AsyncMock(return_value=b"{}"),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={"model": model})),
    )
    metadata = {"chat_id": "local:skill-test", "session_id": "browser", "tool_servers": []}
    return SimpleNamespace(request=request, user=user, model=model, metadata=metadata)


async def parent_skill_tools(context, skill_id):
    from open_webui.utils.tools import get_builtin_tools

    # Core passes its lazy manifest IDs separately from metadata.skill_ids;
    # automatically discovered workspace/terminal skills need not be selected.
    tools = await get_builtin_tools(
        context.request,
        {"__user__": context.user, "__metadata__": context.metadata, "__skill_ids__": [skill_id]},
        model=context.model,
    )
    # Core 0.12.0+ also returns read_skill_file and update_skill_files.
    assert "view_skill" in tools
    bound = tools["view_skill"]["callable"].__extra_params__
    assert bound["__user__"] is context.user
    assert bound["__metadata__"] is context.metadata
    assert bound["__request__"] is context.request
    return tools


async def run_skill(context, skill_id):
    result = await parallel_tools.Tools().run_tools_parallel(
        tool_calls=[{"name": "view_skill", "arguments": {"id": skill_id}}],
        __request__=context.request, __user__=context.user,
        __model__=context.model, __metadata__=context.metadata,
    )
    return json.loads(result)["results"][0]["result"]


@pytest.mark.parametrize("kind", ["workspace", "terminal"])
async def test_parallel_reaches_real_core_workspace_and_terminal_skill(monkeypatch, skill_context, kind):
    from open_webui.models.skills import Skills
    from open_webui.utils import terminals

    context = skill_context
    skill_id = "workspace-skill" if kind == "workspace" else "terminal:terminal%20skill"
    workspace_lookup = AsyncMock(return_value=SimpleNamespace(
        id=skill_id, user_id=context.user["id"], is_active=True,
        name="Workspace skill", content="Workspace instructions",
        description=None, meta={}, version_id="version-1",
        data={"files": [{"path": "SKILL.md", "content": "Workspace instructions"}]},
    ))
    terminal_lookup = AsyncMock(return_value={"name": "Terminal skill", "content": "Terminal instructions"})
    monkeypatch.setattr(Skills, "get_skill_by_id", workspace_lookup)
    monkeypatch.setattr(terminals, "get_terminal_skill", terminal_lookup)
    if kind == "terminal":
        context.metadata["terminal_id"] = "terminal-1"
    context.metadata["tools"] = await parent_skill_tools(context, skill_id)

    result = await run_skill(context, skill_id)

    assert result["content"] == ("Workspace instructions" if kind == "workspace" else "Terminal instructions")
    if kind == "workspace":
        workspace_lookup.assert_awaited_once()
        assert workspace_lookup.await_args.args == (skill_id,)
        terminal_lookup.assert_not_awaited()
    else:
        terminal_lookup.assert_awaited_once()
        assert terminal_lookup.await_args.args[:4] == (
            context.request, context.user, context.metadata, "terminal skill",
        )
        workspace_lookup.assert_not_awaited()


@pytest.mark.skipif(CORE_REGISTERS_SKILL_TOOLS, reason="Core 0.12.0+ needs no parent entry")
@pytest.mark.parametrize("parent_entry", ["missing", "wrong_type", "wrong_id"])
async def test_parallel_does_not_add_skill_without_core_builtin_entry(skill_context, parent_entry):
    context = skill_context
    tools = await parent_skill_tools(context, "workspace-skill")
    if parent_entry == "missing":
        tools = {}
    elif parent_entry == "wrong_type":
        tools["view_skill"]["type"] = "tool"
    else:
        tools["view_skill"]["tool_id"] = "builtin:other"
    context.metadata["tools"] = tools

    result = await run_skill(context, "workspace-skill")

    assert "Tool 'view_skill' not found" in result


@pytest.mark.skipif(not CORE_REGISTERS_SKILL_TOOLS, reason="Core before 0.12.0 has no skills switch")
async def test_parallel_follows_the_model_skills_switch(skill_context):
    context = skill_context
    context.model["info"]["meta"]["builtinTools"]["skills"] = False

    assert "Tool 'view_skill' not found" in await run_skill(context, "workspace-skill")


async def test_parallel_keeps_regular_tool_with_same_name(monkeypatch, skill_context):
    from open_webui.utils import tools as core_tools

    context = skill_context
    context.metadata["tools"] = await parent_skill_tools(context, "workspace-skill")
    context.metadata["tool_ids"] = ["custom"]
    regular = AsyncMock(return_value="Custom skill result")
    monkeypatch.setattr(core_tools, "get_tools", AsyncMock(return_value={"view_skill": {
        "type": "tool", "tool_id": "custom", "callable": regular,
        "spec": {"name": "view_skill", "parameters": {"properties": {"id": {"type": "string"}}}},
    }}))

    result = await run_skill(context, "workspace-skill")

    assert result == "Custom skill result"
    regular.assert_awaited_once_with(id="workspace-skill")


@pytest.mark.parametrize("name", ["read_skill_file", "grep_skill_files", "update_skill_files"])
async def test_parallel_reuses_the_parent_skill_tools_core_bound_to_this_chat(skill_context, name):
    context = skill_context
    tools = await parent_skill_tools(context, "workspace-skill")
    if name not in tools:
        pytest.skip(f"This Core has no {name}")
    context.metadata["tools"] = {name: tools[name]}
    context.model["info"]["meta"]["builtinTools"]["skills"] = False

    result = await parallel_tools.Tools().run_tools_parallel(
        tool_calls=[{"name": name, "arguments": {}}],
        __request__=context.request, __user__=context.user,
        __model__=context.model, __metadata__=context.metadata,
    )

    assert "not found" not in json.dumps(json.loads(result)["results"][0])
