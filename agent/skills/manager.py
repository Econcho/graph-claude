from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent.observe.observer import RuntimeObserver
from agent.skills.discovery import SkillDiscovery, SkillDiscoveryResult
from agent.skills.models import Skill, SkillConfig, SkillSummary
from agent.skills.settings import load_skill_config
from agent.tools.base import ToolContext, ToolResult
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry


INLINE_SHELL_PATTERN = re.compile(r"!`([^`]+)`")
BLOCK_SHELL_PATTERN = re.compile(r"```!\s*\n(.*?)\n```", re.DOTALL)
MAX_SHELL_OUTPUT_CHARS = 8_000


class SkillManager:
    def __init__(
        self,
        llm_client=None,
        *,
        config: SkillConfig | None = None,
        settings_path: str | Path | None = None,
        observer: RuntimeObserver | None = None,
        discovery: SkillDiscovery | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or load_skill_config(settings_path)
        self.observer = observer
        self.discovery = discovery or SkillDiscovery(self.config)

    def prepare_context(self, state: dict[str, Any]) -> dict[str, Any]:
        workspace_root = state.get("workspace_root") or state.get("workspace")
        if not workspace_root or not self.config.enabled or state.get("skills_enabled") is False:
            return {
                "skills_enabled": False,
                "skill_summaries": [],
                "conditional_skill_summaries": [],
            }

        result = self.discovery.discover(workspace_root)
        summaries = self._limit_summaries(result.summaries)
        conditional = self._matching_conditional_summaries(
            result.summaries,
            workspace_root=Path(workspace_root),
            touched_paths=state.get("skill_touched_paths") or [],
        )

        if self.observer:
            self.observer.on_skill_discovery_loaded(
                workspace_root=str(workspace_root),
                skill_count=len(result.skills),
                warnings=result.warnings,
            )

        return {
            "skills_enabled": True,
            "skill_summaries": [summary.to_dict() for summary in summaries],
            "conditional_skill_summaries": [
                summary.to_dict() for summary in self._limit_summaries(conditional)
            ],
        }

    def find_skill(self, name: str, workspace_root: str | Path) -> Skill | None:
        for skill in self.discovery.discover(workspace_root).skills:
            if skill.name == name:
                return skill
        return None

    def execute_skill(
        self,
        *,
        name: str,
        args: str,
        ctx: ToolContext,
    ) -> ToolResult:
        if not self.config.enabled:
            return ToolResult(
                ok=False,
                content="Skill system is disabled.",
                error="skill_disabled",
            )

        depth = ctx.skill_call_depth
        if depth >= self.config.max_recursion_depth:
            return ToolResult(
                ok=False,
                content=f"Skill recursion limit reached: {self.config.max_recursion_depth}",
                error="skill_recursion_limit",
            )

        skill = self.find_skill(name, ctx.workspace_root)
        if skill is None:
            return ToolResult(
                ok=False,
                content=f"Skill not found: {name}",
                error="skill_not_found",
            )

        if self.observer:
            self.observer.on_skill_call_start(skill.name, skill.source, skill.context)

        try:
            prompt = self.render_skill_prompt(skill, args=args, ctx=ctx)
            if skill.context == "inline":
                result = ToolResult(
                    ok=True,
                    content=prompt,
                    data={
                        "skill": skill.name,
                        "context": skill.context,
                        "path": str(skill.path),
                    },
                )
            else:
                result = self._execute_fork(skill, prompt, args=args, ctx=ctx)
        except Exception as error:
            if self.observer:
                self.observer.on_skill_call_error(skill.name, error)
            return ToolResult(
                ok=False,
                content=f"Skill {skill.name} failed: {error}",
                error="skill_execution_error",
            )

        if self.observer:
            self.observer.on_skill_call_end(skill.name, result)
        return result

    def render_skill_prompt(
        self,
        skill: Skill,
        *,
        args: str,
        ctx: ToolContext,
    ) -> str:
        content = skill.body
        content = self._replace_arguments(content, args)
        content = self._replace_variables(content, skill, ctx)

        if (
            self.config.prompt_shell_enabled
            and skill.source != "mcp"
            and ("!`" in content or "```!" in content)
        ):
            content = self._execute_prompt_shell(content, skill, ctx)

        return content.strip() + "\n"

    def _replace_arguments(self, content: str, args: str) -> str:
        replacements = {
            "${ARGUMENTS}": args,
            "${AGENT_SKILL_ARGS}": args,
            "{{args}}": args,
            "{{ARGUMENTS}}": args,
        }
        for key, value in replacements.items():
            content = content.replace(key, value)
        return content

    def _replace_variables(
        self,
        content: str,
        skill: Skill,
        ctx: ToolContext,
    ) -> str:
        skill_dir = str(skill.base_dir).replace("\\", "/")
        replacements = {
            "${AGENT_SKILL_DIR}": skill_dir,
            "${CLAUDE_SKILL_DIR}": skill_dir,
            "${AGENT_SESSION_ID}": ctx.session_id or "",
            "${CLAUDE_SESSION_ID}": ctx.session_id or "",
        }
        for key, value in replacements.items():
            content = content.replace(key, value)
        return content

    def _execute_prompt_shell(
        self,
        content: str,
        skill: Skill,
        ctx: ToolContext,
    ) -> str:
        from agent.skills.tools import ShellCommandTool
        from agent.tools.permissions import PermissionRules

        executor = ToolExecutor(
            ToolRegistry([ShellCommandTool()]),
            permission_rules=PermissionRules.empty(),
        )

        def replace_match(match: re.Match[str]) -> str:
            command = match.group(1).strip()
            if not command:
                return ""

            if self.observer:
                self.observer.on_skill_prompt_shell_start(skill.name, command)

            result = executor.execute(
                {
                    "id": f"skill_shell_{skill.name}",
                    "name": "shell_command",
                    "args": {
                        "command": command,
                        "shell": skill.shell,
                    },
                },
                ctx,
            )

            if self.observer:
                self.observer.on_skill_prompt_shell_end(skill.name, command, result)

            if not result.ok:
                raise RuntimeError(result.content)
            return _truncate(result.content, MAX_SHELL_OUTPUT_CHARS)

        content = BLOCK_SHELL_PATTERN.sub(replace_match, content)
        return INLINE_SHELL_PATTERN.sub(replace_match, content)

    def _execute_fork(
        self,
        skill: Skill,
        prompt: str,
        *,
        args: str,
        ctx: ToolContext,
    ) -> ToolResult:
        if self.llm_client is None:
            return ToolResult(
                ok=False,
                content="Skill fork execution requires an llm_client.",
                error="skill_llm_missing",
            )

        from agent.graph import build_graph

        registry = self._child_registry(skill)
        graph = build_graph(
            llm_client=self.llm_client,
            registry=registry,
            observer=self.observer,
        )
        child_input = prompt
        if args:
            child_input = f"{prompt.rstrip()}\n\nUser arguments:\n{args}\n"

        result = graph.invoke(
            {
                "messages": [HumanMessage(content=child_input)],
                "workspace_root": str(ctx.workspace_root),
                "permission_mode": ctx.permission_mode,
                "session_id": ctx.session_id,
                "agent_id": f"skill:{skill.name}",
                "skill_call_depth": ctx.skill_call_depth + 1,
                "active_skill_name": skill.name,
                "auto_memory_enabled": False,
                "session_memory_enabled": False,
            },
            config={
                "configurable": {
                    "thread_id": f"{ctx.session_id or 'skill'}:{skill.name}:{ctx.skill_call_depth + 1}"
                }
            },
        )

        final_answer = result.get("final_answer") or ""
        return ToolResult(
            ok=True,
            content=str(final_answer),
            data={
                "skill": skill.name,
                "context": skill.context,
                "path": str(skill.path),
            },
        )

    def _child_registry(self, skill: Skill) -> ToolRegistry:
        from agent.skills.tools import ShellCommandTool, UseSkillTool
        from agent.tools.builtin import BUILTIN_TOOLS

        allowed = set(skill.allowed_tools or ["read_file", "write_file", "use_skill"])
        candidates = [*BUILTIN_TOOLS, ShellCommandTool(), UseSkillTool(self)]
        selected = [
            tool
            for tool in candidates
            if tool.name in allowed or any(alias in allowed for alias in tool.aliases)
        ]
        return ToolRegistry(selected)

    def _limit_summaries(self, summaries: list[SkillSummary]) -> list[SkillSummary]:
        limited: list[SkillSummary] = []
        total = 0
        for summary in summaries[: self.config.max_prompt_skills]:
            line = self.format_summary_line(summary)
            next_total = total + len(line) + 1
            if next_total > self.config.prompt_budget_chars:
                break
            limited.append(summary)
            total = next_total
        return limited

    def _matching_conditional_summaries(
        self,
        summaries: list[SkillSummary],
        *,
        workspace_root: Path,
        touched_paths: list[str],
    ) -> list[SkillSummary]:
        if not touched_paths:
            return []

        rel_paths = [_normalize_touched_path(workspace_root, path) for path in touched_paths]
        matched: list[SkillSummary] = []
        for summary in summaries:
            if not summary.paths:
                continue
            if any(
                fnmatch.fnmatch(rel_path, pattern)
                for rel_path in rel_paths
                for pattern in summary.paths
            ):
                matched.append(summary)
        return matched

    @staticmethod
    def format_summary_line(summary: SkillSummary) -> str:
        suffix = f" When to use: {summary.when_to_use}" if summary.when_to_use else ""
        return f"- {summary.name}: {summary.description}{suffix}"


def _normalize_touched_path(workspace_root: Path, path: str) -> str:
    raw = Path(path)
    try:
        if raw.is_absolute():
            return raw.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return raw.as_posix()
    return raw.as_posix()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n...[truncated {omitted} chars]"
