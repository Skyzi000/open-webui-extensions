"""AST checks for Open WebUI tool schema compatibility."""

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import get_args, get_origin

import pytest

# Add tools directory to path
tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

import sub_agent  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_CLASS_LINT_PATH = REPO_ROOT / "scripts" / "lint_tools_class.py"
GRAPHITI_TOOLS_CLASS_LINT_PATH = REPO_ROOT / "graphiti" / "scripts" / "lint_tools_class.py"


def load_lint_module(path: Path, module_name: str):
    """Load the lint script as a Python module."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(
    params=[
        (TOOLS_CLASS_LINT_PATH, "lint_tools_class_parametrized"),
        (GRAPHITI_TOOLS_CLASS_LINT_PATH, "graphiti_lint_tools_class_parametrized"),
    ]
)
def lint_module_under_test(request):
    path, module_name = request.param
    return load_lint_module(path, module_name)


def test_tool_schema_lint_reports_no_repo_issues():
    lint_module = load_lint_module(TOOLS_CLASS_LINT_PATH, "lint_tools_class")
    errors = lint_module.check_paths([REPO_ROOT / "tools", REPO_ROOT / "graphiti" / "tools"])
    assert errors == []


def test_tool_schema_lint_detects_known_incompatible_patterns(tmp_path):
    lint_module = load_lint_module(TOOLS_CLASS_LINT_PATH, "lint_tools_class_tmp")
    sample_file = tmp_path / "bad_tool.py"
    sample_file.write_text(
        """class Tools:
    async def bad_args(self, tasks: list[dict]):
        \"\"\"
        Bad schema.

        Args:
            tasks: Should not use Google-style args.
        \"\"\"
        return \"\"

    async def bad_param(self, query: str):
        \"\"\"
        Bad param.

        :param query: First line.
            Second line that Open WebUI will drop.
        \"\"\"
        return \"\"
""",
        encoding="utf-8",
    )

    errors = lint_module.check_paths([sample_file])
    joined = "\n".join(errors)
    assert "uses Google-style Args block" in joined
    assert "uses raw dict/list[dict] annotation" in joined
    assert "has multiline :param" in joined


def test_graphiti_tool_schema_lint_reports_no_repo_issues():
    lint_module = load_lint_module(
        GRAPHITI_TOOLS_CLASS_LINT_PATH,
        "graphiti_lint_tools_class",
    )
    errors = lint_module.check_paths([REPO_ROOT / "graphiti" / "tools"])
    assert errors == []


def test_tool_schema_lint_detects_standard_multiline_sphinx_param(lint_module_under_test, tmp_path):
    sample_file = tmp_path / "bad_multiline_param.py"
    sample_file.write_text(
        """class Tools:
    async def search(self, query: str):
        \"\"\"
        Bad param.

        :param query:
            Second-line-only descriptions are dropped by Open WebUI.
        \"\"\"
        return \"\"
""",
        encoding="utf-8",
    )

    errors = lint_module_under_test.check_paths([sample_file])
    assert any("has multiline :param for 'query'" in error for error in errors)


def test_tool_schema_lint_detects_nested_model_raw_dict(lint_module_under_test, tmp_path):
    sample_file = tmp_path / "bad_nested_model.py"
    sample_file.write_text(
        """from pydantic import BaseModel

class ToolCallItem(BaseModel):
    name: str
    args: dict

class Tools:
    async def run(self, tool_calls: list[ToolCallItem]):
        \"\"\"
        Bad nested schema.

        :param tool_calls: List of tool calls.
        \"\"\"
        return \"\"
""",
        encoding="utf-8",
    )

    errors = lint_module_under_test.check_paths([sample_file])
    joined = "\n".join(errors)
    assert "uses raw dict/list[dict] annotation inside public model field 'ToolCallItem.args'" in joined


def test_tool_schema_lint_allows_explicit_nested_model_passthrough(lint_module_under_test, tmp_path):
    sample_file = tmp_path / "dynamic_passthrough.py"
    sample_file.write_text(
        """from pydantic import BaseModel

class ToolCallItem(BaseModel):
    name: str
    args: dict

class Tools:
    async def run(self, tool_calls: list[ToolCallItem]):
        \"\"\"
        Allowed nested schema.

        :param tool_calls: List of tool calls.
        \"\"\"
        return \"\"
""",
        encoding="utf-8",
    )

    original_allowlist = dict(lint_module_under_test.RAW_OBJECT_PASSTHROUGH_ALLOWLIST)
    lint_module_under_test.RAW_OBJECT_PASSTHROUGH_ALLOWLIST = {
        **original_allowlist,
        "dynamic_passthrough.py": {"ToolCallItem": {"args"}},
    }
    try:
        errors = lint_module_under_test.check_paths([sample_file])
    finally:
        lint_module_under_test.RAW_OBJECT_PASSTHROUGH_ALLOWLIST = original_allowlist

    assert errors == []


def test_tool_schema_lint_does_not_flag_other_sphinx_directives_as_multiline(
    lint_module_under_test, tmp_path
):
    sample_file = tmp_path / "good_sphinx_param.py"
    sample_file.write_text(
        """class Tools:
    async def search(self, query: str):
        \"\"\"
        Valid param.

        :param query: Single-line description.
        :type query: str
        :raises ValueError: Only raised internally.
        :return: Result text.
        \"\"\"
        return \"\"
""",
        encoding="utf-8",
    )

    errors = lint_module_under_test.check_paths([sample_file])
    assert errors == []


def test_tool_schema_lint_checks_keyword_only_public_params(lint_module_under_test, tmp_path):
    sample_file = tmp_path / "bad_kwonly_tool.py"
    sample_file.write_text(
        """class Tools:
    async def search(self, *, filters: dict[str, str]):
        \"\"\"
        Bad keyword-only param.

        :param filters:
            Open WebUI drops this continuation line.
        \"\"\"
        return \"\"
""",
        encoding="utf-8",
    )

    errors = lint_module_under_test.check_paths([sample_file])
    joined = "\n".join(errors)
    assert "has multiline :param for 'filters'" in joined
    assert "uses raw dict/list[dict] annotation for public parameter 'filters'" in joined


def test_pre_commit_configs_register_tools_class_hook_only():
    root_config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    graphiti_config = (REPO_ROOT / "graphiti" / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert root_config.count("id: lint-tools-class") == 1
    assert "entry: python scripts/lint_tools_class.py" in root_config
    assert "lint-tool-schema-compat" not in root_config

    assert graphiti_config.count("id: lint-tools-class") == 1
    assert "entry: python scripts/lint_tools_class.py" in graphiti_config
    assert "lint-tool-schema-compat" not in graphiti_config


def test_run_parallel_sub_agents_uses_named_task_model():
    annotation = inspect.signature(sub_agent.Tools.run_parallel_sub_agents).parameters["tasks"].annotation
    assert get_origin(annotation) is list
    assert get_args(annotation) == (sub_agent.SubAgentTaskItem,)
    assert sub_agent.SubAgentTaskItem.model_fields["description"].description
    assert sub_agent.SubAgentTaskItem.model_fields["prompt"].description


@pytest.mark.asyncio
async def test_run_parallel_sub_agents_rejects_over_limit_before_normalization():
    tool = sub_agent.Tools()
    tool.valves.MAX_PARALLEL_AGENTS = 1

    result = await tool.run_parallel_sub_agents(
        tasks=[
            {"description": "Task A", "prompt": "Do A"},
            {"description": "", "prompt": "This would fail normalization if reached"},
        ],
        __user__={},
        __request__=object(),
    )

    payload = json.loads(result)
    assert payload == {
        "error": "tasks count (2) exceeds MAX_PARALLEL_AGENTS (1)",
        "max_parallel_agents": 1,
    }


def test_normalize_parallel_sub_agent_tasks_preserves_backward_compatibility():
    tasks, error = sub_agent.normalize_parallel_sub_agent_tasks(
        [
            sub_agent.SubAgentTaskItem(description="Task A", prompt="Do A"),
            json.dumps({"description": "Task B", "prompt": "Do B"}),
        ]
    )

    assert error is None
    assert tasks == [
        {"description": "Task A", "prompt": "Do A"},
        {"description": "Task B", "prompt": "Do B"},
    ]


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ([{"description": "", "prompt": "Do A"}], "tasks[0].description cannot be empty"),
        ([{"description": "Task A"}], "tasks[0].prompt Field required"),
    ],
)
def test_normalize_parallel_sub_agent_tasks_returns_actionable_errors(payload, expected_error):
    _, error = sub_agent.normalize_parallel_sub_agent_tasks(payload)
    assert error is not None
    assert expected_error in error
