from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph
from agent.graph.nodes.prompt_node import make_prompt_node
from agent.skills import SkillConfig, SkillManager
from agent.skills.discovery import SkillDiscovery
from agent.skills.models import Skill
from agent.skills.parser import parse_skill_file
from agent.skills.tools import UseSkillTool
from agent.tools import PermissionRules, ToolContext, ToolExecutor, ToolRegistry


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "Do a focused task.",
    body: str = "Follow these steps.",
    extra_frontmatter: str = "",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        (
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"{extra_frontmatter}"
            "---\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )
    return skill_file


def test_parse_skill_frontmatter_fields(tmp_path: Path):
    skill_file = _write_skill(
        tmp_path,
        "review",
        extra_frontmatter=(
            "when_to_use: when reviewing code\n"
            "allowed_tools:\n"
            "  - read_file\n"
            "paths:\n"
            "  - src/**/*.py\n"
            "context: inline\n"
            "shell: powershell\n"
            "user_invocable: false\n"
        ),
    )

    skill = parse_skill_file(skill_file, source="project", max_bytes=262_144)

    assert skill.name == "review"
    assert skill.when_to_use == "when reviewing code"
    assert skill.allowed_tools == ["read_file"]
    assert skill.paths == ["src/**/*.py"]
    assert skill.context == "inline"
    assert skill.shell == "powershell"
    assert skill.user_invocable is False


def test_discovery_loads_and_deduplicates_by_priority(tmp_path: Path):
    workspace = tmp_path / "workspace"
    user_home = tmp_path / "home"
    bundled = tmp_path / "bundled"
    workspace.mkdir()
    user_home.mkdir()
    bundled.mkdir()

    _write_skill(bundled, "shared", description="bundled")
    _write_skill(user_home / ".agent" / "skills", "shared", description="user")
    _write_skill(workspace / ".agent" / "skills", "shared", description="project")

    discovery = SkillDiscovery(
        SkillConfig(),
        bundled_dir=bundled,
        user_home=user_home,
    )
    result = discovery.discover(workspace)

    assert [skill.name for skill in result.skills] == ["shared"]
    assert result.skills[0].description == "project"
    assert result.warnings


def test_prompt_injects_skill_summary_only(tmp_path: Path):
    class Registry:
        def get_model_tool_specs(self, ctx):
            return []

    prompt_node = make_prompt_node(Registry())
    result = prompt_node(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
            "skill_summaries": [
                {
                    "name": "review",
                    "description": "Review code.",
                    "when_to_use": "when asked to review",
                }
            ],
        }
    )

    assert "Available skills:" in result["system_prompt"]
    assert "- review: Review code." in result["system_prompt"]
    assert "Follow these steps" not in result["system_prompt"]


def test_use_skill_inline_loads_full_skill_on_call(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_skill(
        workspace / ".agent" / "skills",
        "inline-skill",
        body="Use ${AGENT_SKILL_DIR} with {{args}}.",
        extra_frontmatter="context: inline\n",
    )
    manager = SkillManager(
        config=SkillConfig(),
        discovery=SkillDiscovery(
            SkillConfig(),
            bundled_dir=tmp_path / "missing",
            user_home=tmp_path / "home",
        ),
    )
    executor = ToolExecutor(
        ToolRegistry([UseSkillTool(manager)]),
        permission_rules=PermissionRules.empty(),
    )

    result = executor.execute(
        {
            "id": "call_skill",
            "name": "use_skill",
            "args": {"name": "inline-skill", "args": "ARG"},
        },
        ToolContext(workspace_root=workspace),
    )

    assert result.ok is True
    assert "ARG" in result.content
    assert "inline-skill" in result.content


def test_use_skill_fork_uses_isolated_child_messages(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_skill(
        workspace / ".agent" / "skills",
        "fork-skill",
        body="Answer with the skill result for ${ARGUMENTS}.",
    )

    class Model:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            assert len([message for message in messages if message.type == "human"]) == 1
            assert "outer transcript" not in messages[-1].content
            return AIMessage(content="skill result")

    manager = SkillManager(
        Model(),
        config=SkillConfig(),
        discovery=SkillDiscovery(
            SkillConfig(),
            bundled_dir=tmp_path / "missing",
            user_home=tmp_path / "home",
        ),
    )

    result = manager.execute_skill(
        name="fork-skill",
        args="input",
        ctx=ToolContext(workspace_root=workspace, session_id="s"),
    )

    assert result.ok is True
    assert result.content == "skill result"


def test_allowed_tools_limit_child_registry(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_skill(
        workspace / ".agent" / "skills",
        "limited",
        extra_frontmatter="allowed_tools:\n  - read_file\n",
    )
    manager = SkillManager(
        config=SkillConfig(),
        discovery=SkillDiscovery(
            SkillConfig(),
            bundled_dir=tmp_path / "missing",
            user_home=tmp_path / "home",
        ),
    )
    skill = manager.find_skill("limited", workspace)

    registry = manager._child_registry(skill)
    names = [tool.name for tool in registry.get_tools(ToolContext(workspace_root=workspace))]

    assert names == ["read_file"]


def test_conditional_skill_matches_touched_paths(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_skill(
        workspace / ".agent" / "skills",
        "python-skill",
        extra_frontmatter="paths:\n  - src/*.py\n",
    )
    manager = SkillManager(
        config=SkillConfig(),
        discovery=SkillDiscovery(
            SkillConfig(),
            bundled_dir=tmp_path / "missing",
            user_home=tmp_path / "home",
        ),
    )

    patch = manager.prepare_context(
        {
            "workspace_root": str(workspace),
            "skill_touched_paths": [str(workspace / "src" / "app.py")],
        }
    )

    assert patch["conditional_skill_summaries"][0]["name"] == "python-skill"


def test_mcp_skill_does_not_execute_prompt_shell(tmp_path: Path):
    manager = SkillManager(config=SkillConfig())
    skill = Skill(
        name="remote",
        description="remote",
        body="Value: !`this-command-must-not-run`",
        path=tmp_path / "SKILL.md",
        base_dir=tmp_path,
        source="mcp",
    )

    rendered = manager.render_skill_prompt(
        skill,
        args="",
        ctx=ToolContext(workspace_root=tmp_path),
    )

    assert "!`this-command-must-not-run`" in rendered


def test_shell_prompt_denied_in_plan_mode(tmp_path: Path):
    manager = SkillManager(config=SkillConfig())
    skill = Skill(
        name="local",
        description="local",
        body="Value: !`echo hi`",
        path=tmp_path / "SKILL.md",
        base_dir=tmp_path,
        source="project",
    )

    result = manager.execute_skill(
        name="missing",
        args="",
        ctx=ToolContext(workspace_root=tmp_path, permission_mode="plan"),
    )

    assert result.error == "skill_not_found"
    try:
        manager.render_skill_prompt(
            skill,
            args="",
            ctx=ToolContext(workspace_root=tmp_path, permission_mode="plan"),
        )
    except RuntimeError as error:
        assert "not allowed in plan mode" in str(error)
    else:
        raise AssertionError("expected prompt shell permission failure")


def test_skill_recursion_limit(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_skill(workspace / ".agent" / "skills", "repeat")
    manager = SkillManager(
        config=SkillConfig(max_recursion_depth=1),
        discovery=SkillDiscovery(
            SkillConfig(max_recursion_depth=1),
            bundled_dir=tmp_path / "missing",
            user_home=tmp_path / "home",
        ),
    )

    result = manager.execute_skill(
        name="repeat",
        args="",
        ctx=ToolContext(workspace_root=workspace, skill_call_depth=1),
    )

    assert result.ok is False
    assert result.error == "skill_recursion_limit"
