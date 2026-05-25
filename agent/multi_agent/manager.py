from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agent.multi_agent.models import MultiAgentConfig
from agent.multi_agent.runner import SubagentRunner
from agent.multi_agent.settings import load_multi_agent_config
from agent.multi_agent.stores import (
    AgentTaskStore,
    MailboxStore,
    TaskListStore,
    TeamStore,
    task_to_dict,
)
from agent.observe.observer import RuntimeObserver
from agent.tools.base import ToolContext, ToolResult


class MultiAgentManager:
    def __init__(
        self,
        llm_client=None,
        *,
        config: MultiAgentConfig | None = None,
        settings_path: str | Path | None = None,
        observer: RuntimeObserver | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or load_multi_agent_config(settings_path)
        self.observer = observer
        self.task_store = AgentTaskStore()
        self.team_store = TeamStore()
        self.mailbox_store = MailboxStore()
        self.task_list_store = TaskListStore()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_background_agents)
        self._futures: dict[str, Future] = {}
        self._registry_factory = None

    def set_registry_factory(self, factory) -> None:
        self._registry_factory = factory

    def spawn_agent(
        self,
        *,
        description: str,
        prompt: str,
        ctx: ToolContext,
        name: str | None = None,
        team_name: str | None = None,
        run_in_background: bool = False,
        mode: str | None = None,
    ) -> ToolResult:
        if not self.config.enabled:
            return ToolResult(False, "Multi-agent system is disabled.", error="multi_agent_disabled")
        if self.llm_client is None:
            return ToolResult(False, "Multi-agent requires an llm_client.", error="multi_agent_llm_missing")
        if ctx.subagent_depth >= self.config.max_subagent_depth:
            return ToolResult(
                False,
                f"Subagent depth limit reached: {self.config.max_subagent_depth}",
                error="subagent_depth_limit",
            )
        teammate = bool(team_name and name)
        if teammate and ctx.team_name:
            return ToolResult(False, "Teammates cannot spawn teammates.", error="teammate_spawn_denied")
        if teammate and run_in_background:
            return ToolResult(
                False,
                "In-process teammates cannot be started as background agents.",
                error="teammate_background_denied",
            )
        if ctx.team_name and run_in_background:
            return ToolResult(
                False,
                "In-process teammates cannot create background agents.",
                error="teammate_background_denied",
            )

        agent_id = _agent_id(name=name, team_name=team_name, depth=ctx.subagent_depth + 1)
        task = self.task_store.create(
            agent_id=agent_id,
            name=name,
            description=description,
            prompt=prompt,
            team_name=team_name,
        )
        if teammate and team_name and name:
            self.team_store.register_agent(
                workspace_root=ctx.workspace_root,
                team_name=team_name,
                agent_name=name,
            )

        self._emit(
            "agent_spawn_start",
            output_node="tool_executor",
            input_node="multi_agent",
            content={
                "task_id": task.task_id,
                "agent_id": agent_id,
                "team_name": team_name,
                "background": run_in_background,
            },
        )

        if run_in_background:
            future = self._executor.submit(
                self._run_task,
                task.task_id,
                prompt,
                ctx,
                name,
                team_name,
                teammate,
            )
            self._futures[task.task_id] = future
            if self.observer and hasattr(self.observer, "on_agent_delegated"):
                self.observer.on_agent_delegated(
                    task_id=task.task_id,
                    agent_id=agent_id,
                    mode="teammate" if teammate else "background",
                    status="running",
                )
            return ToolResult(
                True,
                f"Agent task started: {task.task_id}",
                data=task_to_dict(task),
            )

        result = self._run_task(task.task_id, prompt, ctx, name, team_name, teammate)
        task = self.task_store.get(task.task_id)
        if result.ok:
            return ToolResult(
                True,
                result.content,
                data=task_to_dict(task) if task else None,
            )
        return result

    def poll(self, task_id: str | None = None) -> ToolResult:
        if task_id:
            task = self.task_store.get(task_id)
            if task is None:
                return ToolResult(False, f"Agent task not found: {task_id}", error="agent_task_not_found")
            return ToolResult(True, _format_task(task_to_dict(task)), data=task_to_dict(task))

        tasks = [task_to_dict(task) for task in self.task_store.list()]
        return ToolResult(True, _format_tasks(tasks), data={"tasks": tasks})

    def consume_notifications(self) -> list[str]:
        if not self.config.notifications_enabled:
            return []
        return [notification.to_xml() for notification in self.task_store.consume_notifications()]

    def create_team(self, *, team_name: str, leader: str, ctx: ToolContext) -> ToolResult:
        payload = self.team_store.create_team(
            workspace_root=ctx.workspace_root,
            team_name=team_name,
            leader=leader,
        )
        self._emit("team_created", output_node="multi_agent", input_node="tool_result", content=payload)
        return ToolResult(True, f"Team created: {team_name}", data=payload)

    def send_message(self, *, to: str, message: str, ctx: ToolContext) -> ToolResult:
        if not ctx.team_name:
            return ToolResult(False, "send_message requires team_name in context.", error="missing_team")
        sender = ctx.agent_name or ctx.agent_id or "agent"
        recipients = self.mailbox_store.send(
            workspace_root=ctx.workspace_root,
            team_name=ctx.team_name,
            sender=sender,
            to=to,
            message=message,
        )
        payload = {"team_name": ctx.team_name, "from": sender, "to": recipients, "message": message}
        self._emit("agent_message_sent", output_node="multi_agent", input_node="mailbox", content=payload)
        self._emit("mailbox_message_written", output_node="multi_agent", input_node="mailbox", content=payload)
        return ToolResult(True, f"Message sent to: {', '.join(recipients)}", data=payload)

    def read_mailbox(self, ctx: ToolContext) -> list[dict[str, Any]]:
        if not self.config.mailbox_poll_enabled or not ctx.team_name or not ctx.agent_name:
            return []
        messages = self.mailbox_store.read(
            workspace_root=ctx.workspace_root,
            team_name=ctx.team_name,
            agent_name=ctx.agent_name,
        )
        if messages:
            self._emit(
                "mailbox_message_read",
                output_node="mailbox",
                input_node="context",
                content={"team_name": ctx.team_name, "agent_name": ctx.agent_name, "messages": messages},
            )
        return messages

    def create_shared_task(self, *, description: str, ctx: ToolContext) -> ToolResult:
        if not ctx.team_name:
            return ToolResult(False, "task_create requires team_name in context.", error="missing_team")
        task = self.task_list_store.create(
            workspace_root=ctx.workspace_root,
            team_name=ctx.team_name,
            description=description,
            created_by=ctx.agent_name or ctx.agent_id or "agent",
        )
        self._emit("team_task_created", output_node="multi_agent", input_node="task_list", content=task)
        return ToolResult(True, f"Team task created: {task['id']}", data=task)

    def list_shared_tasks(self, ctx: ToolContext) -> ToolResult:
        if not ctx.team_name:
            return ToolResult(False, "task_list requires team_name in context.", error="missing_team")
        tasks = self.task_list_store.list(workspace_root=ctx.workspace_root, team_name=ctx.team_name)
        return ToolResult(True, _format_tasks(tasks), data={"tasks": tasks})

    def update_shared_task(
        self,
        *,
        task_id: str,
        status: str | None,
        assignee: str | None,
        ctx: ToolContext,
    ) -> ToolResult:
        if not ctx.team_name:
            return ToolResult(False, "task_update requires team_name in context.", error="missing_team")
        task = self.task_list_store.update(
            workspace_root=ctx.workspace_root,
            team_name=ctx.team_name,
            task_id=task_id,
            status=status,
            assignee=assignee,
        )
        if task is None:
            return ToolResult(False, f"Team task not found: {task_id}", error="team_task_not_found")
        self._emit("team_task_updated", output_node="multi_agent", input_node="task_list", content=task)
        return ToolResult(True, f"Team task updated: {task_id}", data=task)

    def stop_shared_task(self, *, task_id: str, ctx: ToolContext) -> ToolResult:
        return self.update_shared_task(task_id=task_id, status="stopped", assignee=None, ctx=ctx)

    def _run_task(
        self,
        task_id: str,
        prompt: str,
        ctx: ToolContext,
        name: str | None,
        team_name: str | None,
        teammate: bool,
    ) -> ToolResult:
        task = self.task_store.get(task_id)
        if task is None:
            return ToolResult(False, f"Agent task not found: {task_id}", error="agent_task_not_found")
        try:
            runner = SubagentRunner(
                llm_client=self.llm_client,
                observer=self.observer,
                config=self.config,
                registry_factory=self._child_registry,
                multi_agent_manager=self,
            )
            result_state = runner.run(
                prompt=prompt,
                workspace_root=ctx.workspace_root,
                session_id=ctx.session_id,
                permission_mode=ctx.permission_mode,
                agent_id=task.agent_id,
                agent_name=name,
                team_name=team_name,
                subagent_depth=ctx.subagent_depth + 1,
                teammate=teammate,
            )
            final_answer = str(result_state.get("final_answer") or "")
            self.task_store.update(
                task_id,
                status="completed",
                final_answer=final_answer,
                result_state=result_state,
            )
            self._emit_status(task_id, "completed")
            self._emit(
                "agent_spawn_end",
                output_node="multi_agent",
                input_node="tool_result",
                content={"task_id": task_id, "agent_id": task.agent_id, "final_answer": final_answer},
            )
            if self.observer and hasattr(self.observer, "on_agent_delegated"):
                self.observer.on_agent_delegated(
                    task_id=task_id,
                    agent_id=task.agent_id,
                    mode="teammate" if teammate else "sync",
                    status="completed",
                    final_answer=final_answer,
                )
            return ToolResult(True, final_answer, data=task_to_dict(self.task_store.get(task_id)))
        except Exception as error:
            self.task_store.update(task_id, status="failed", error=str(error))
            self._emit_status(task_id, "failed")
            self._emit(
                "agent_spawn_error",
                output_node="multi_agent",
                input_node="tool_result",
                content={"task_id": task_id, "agent_id": task.agent_id, "error": str(error)},
            )
            if self.observer and hasattr(self.observer, "on_agent_delegated"):
                self.observer.on_agent_delegated(
                    task_id=task_id,
                    agent_id=task.agent_id,
                    mode="teammate" if teammate else "sync",
                    status="failed",
                    final_answer=str(error),
                )
            return ToolResult(False, f"Agent task failed: {error}", error="agent_task_failed")

    def _child_registry(self, *, teammate: bool):
        if self._registry_factory is None:
            raise RuntimeError("MultiAgentManager registry factory is not configured.")
        return self._registry_factory(teammate=teammate)

    def _emit_status(self, task_id: str, status: str) -> None:
        self._emit(
            "agent_task_status_changed",
            output_node="multi_agent",
            input_node="runtime",
            content={"task_id": task_id, "status": status},
        )

    def _emit(self, result_type: str, *, output_node: str, input_node: str, content: dict[str, Any]) -> None:
        if not self.observer:
            return
        self.observer.emit(result_type, output_node=output_node, input_node=input_node, content=content)


def _agent_id(*, name: str | None, team_name: str | None, depth: int) -> str:
    if team_name and name:
        return f"team:{team_name}:{name}"
    if name:
        return f"subagent:{name}:{depth}"
    return f"subagent:{depth}"


def _format_task(task: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in task.items())


def _format_tasks(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "No tasks."
    return "\n\n".join(_format_task(task) for task in tasks)
