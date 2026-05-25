from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.graph.nodes.prompt_node import DEFAULT_SYSTEM_PROMPT, make_prompt_node
from agent.observe.run_recorder import create_observer
from agent.prompt import PromptRuntime
from agent.prompt.defaults import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
from agent.skills import SkillManager, UseSkillTool
from agent.tools import ToolRegistry
from agent.tools.file_tools.read_file import ReadFileTool


class Registry:
    def get_model_tool_specs(self, ctx):
        return []


def test_default_prompt_is_sectioned_and_contains_existing_rules(tmp_path: Path):
    output = make_prompt_node(ToolRegistry([]))(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
        }
    )

    assert "You are a coding agent." in output["system_prompt"]
    assert "When you need file content, use read_file." in output["system_prompt"]
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in output["system_prompt"]
    assert output["prompt_section_stats"]
    assert output["prompt_runtime_managed"] is True


def test_custom_prompt_replaces_default_and_append_is_added(tmp_path: Path):
    output = make_prompt_node(ToolRegistry([]))(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
            "custom_system_prompt": "Custom base.",
            "append_system_prompt": "Append policy.",
        }
    )

    assert output["system_prompt"].startswith("Custom base.")
    assert output["system_prompt"].rstrip().endswith("Append policy.")
    assert "You are a coding agent." not in output["system_prompt"]


def test_override_prompt_has_highest_priority(tmp_path: Path):
    output = make_prompt_node(ToolRegistry([]))(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
            "override_system_prompt": "Override only.",
            "agent_system_prompt": "Agent prompt.",
            "custom_system_prompt": "Custom prompt.",
        }
    )

    assert output["system_prompt"].strip() == "Override only."


def test_legacy_system_prompt_field_still_replaces_default(tmp_path: Path):
    output = make_prompt_node(ToolRegistry([]))(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
            "system_prompt": "Legacy custom.",
        }
    )

    assert output["system_prompt"].strip() == "Legacy custom."


def test_managed_system_prompt_is_not_reused_as_custom(tmp_path: Path):
    prompt = make_prompt_node(ToolRegistry([]))
    first = prompt(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
        }
    )
    second = prompt(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
            "system_prompt": first["system_prompt"],
            "prompt_runtime_managed": True,
        }
    )

    assert second["system_prompt"].count("You are a coding agent.") == 1


def test_context_snapshot_llm_messages_are_used(tmp_path: Path):
    original = [HumanMessage(content="original")]
    compacted = [HumanMessage(content="compacted")]

    output = make_prompt_node(ToolRegistry([]))(
        {
            "workspace_root": str(tmp_path),
            "messages": original,
            "context_snapshot": {"llm_messages": compacted},
        }
    )

    assert output["llm_request"]["messages"] == compacted


def test_auto_memory_and_skills_still_inject_into_prompt(tmp_path: Path):
    output = make_prompt_node(ToolRegistry([]))(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
            "context_snapshot": {
                "relevant_memories": [
                    {
                        "id": "m1",
                        "type": "user",
                        "name": "Chinese",
                        "updated_at": "2026-05-23",
                        "description": "Use Chinese.",
                        "content": "Answer in Chinese.",
                        "how_to_apply": "Use Chinese.",
                    }
                ],
                "skill_summaries": [
                    {
                        "name": "review",
                        "description": "Review code.",
                        "when_to_use": "when asked",
                    }
                ],
            },
        }
    )

    assert "Relevant long-term memory" in output["system_prompt"]
    assert "Available skills:" in output["system_prompt"]
    assert "review" in output["system_prompt"]


def test_skill_tool_visibility_filter(tmp_path: Path):
    manager = SkillManager()
    registry = ToolRegistry([ReadFileTool(), UseSkillTool(manager)])
    prompt = make_prompt_node(registry)

    without_skills = prompt(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
        }
    )
    with_skills = prompt(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
            "context_snapshot": {
                "skill_summaries": [
                    {"name": "review", "description": "Review code."}
                ]
            },
        }
    )

    assert {tool["name"] for tool in without_skills["tool_specs"]} == {"read_file"}
    assert {tool["name"] for tool in with_skills["tool_specs"]} == {
        "read_file",
        "use_skill",
    }


def test_claude_md_is_injected_as_user_context(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text("Project instruction.", encoding="utf-8")

    output = make_prompt_node(ToolRegistry([]))(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
        }
    )

    assert "Project instruction." in output["system_prompt"]
    assert "user_context.claude_md" in {
        item["name"] for item in output["prompt_section_stats"]
    }


def test_git_status_is_skipped_outside_git_repo(tmp_path: Path):
    output = make_prompt_node(ToolRegistry([]))(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
        }
    )

    assert "Git status:" not in output["system_prompt"]


def test_prompt_dump_writes_snapshot_json(tmp_path: Path):
    observer = create_observer(base_dir=tmp_path / "runs", enabled=True)
    output = make_prompt_node(ToolRegistry([]), observer=observer)(
        {
            "workspace_root": str(tmp_path),
            "messages": [HumanMessage(content="hi")],
        }
    )

    snapshot_ref = output["prompt_snapshot_ref"]
    assert snapshot_ref is not None
    assert Path(snapshot_ref).exists()
    assert "system_prompt" in Path(snapshot_ref).read_text(encoding="utf-8")
