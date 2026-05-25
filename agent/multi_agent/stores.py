from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.multi_agent.models import AgentNotification, AgentTask, AgentTaskStatus


class AgentTaskStore:
    def __init__(self):
        self._tasks: dict[str, AgentTask] = {}
        self._notifications: list[AgentNotification] = []
        self._lock = threading.Lock()

    def create(
        self,
        *,
        agent_id: str,
        name: str | None,
        description: str,
        prompt: str,
        team_name: str | None = None,
    ) -> AgentTask:
        task = AgentTask(
            task_id=f"task_{uuid4().hex[:12]}",
            agent_id=agent_id,
            name=name,
            description=description,
            prompt=prompt,
            team_name=team_name,
        )
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[AgentTask]:
        with self._lock:
            return list(self._tasks.values())

    def update(
        self,
        task_id: str,
        *,
        status: AgentTaskStatus | None = None,
        final_answer: str | None = None,
        error: str | None = None,
        result_state: dict[str, Any] | None = None,
    ) -> AgentTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if status is not None:
                task.status = status
            if final_answer is not None:
                task.final_answer = final_answer
            if error is not None:
                task.error = error
            if result_state is not None:
                task.result_state = result_state
            if status in {"completed", "failed", "killed"}:
                content = final_answer or error or ""
                self._notifications.append(
                    AgentNotification(
                        task_id=task.task_id,
                        agent_id=task.agent_id,
                        status=task.status,
                        content=content,
                    )
                )
            return task

    def consume_notifications(self) -> list[AgentNotification]:
        with self._lock:
            notifications = list(self._notifications)
            self._notifications = []
            return notifications


class TeamStore:
    def create_team(
        self,
        *,
        workspace_root: Path,
        team_name: str,
        leader: str,
    ) -> dict[str, Any]:
        team_dir = _team_dir(workspace_root, team_name)
        (team_dir / "inboxes").mkdir(parents=True, exist_ok=True)
        payload = {
            "team_name": team_name,
            "leader": leader,
            "agents": [leader],
        }
        (team_dir / "team.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tasks_path = team_dir / "tasks.json"
        if not tasks_path.exists():
            tasks_path.write_text("[]", encoding="utf-8")
        return payload

    def register_agent(
        self,
        *,
        workspace_root: Path,
        team_name: str,
        agent_name: str,
    ) -> None:
        team_dir = _team_dir(workspace_root, team_name)
        team_dir.mkdir(parents=True, exist_ok=True)
        path = team_dir / "team.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"team_name": team_name, "leader": "", "agents": []}
        else:
            payload = {"team_name": team_name, "leader": "", "agents": []}
        agents = payload.get("agents", [])
        if not isinstance(agents, list):
            agents = []
        if agent_name not in agents:
            agents.append(agent_name)
        payload["agents"] = agents
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_agents(self, *, workspace_root: Path, team_name: str) -> list[str]:
        path = _team_dir(workspace_root, team_name) / "team.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        agents = payload.get("agents", [])
        return [str(item) for item in agents] if isinstance(agents, list) else []


class MailboxStore:
    def send(
        self,
        *,
        workspace_root: Path,
        team_name: str,
        sender: str,
        to: str,
        message: str,
    ) -> list[str]:
        recipients = (
            TeamStore().list_agents(workspace_root=workspace_root, team_name=team_name)
            if to == "*"
            else [to]
        )
        written: list[str] = []
        for recipient in recipients:
            if recipient == sender:
                continue
            inbox = _team_dir(workspace_root, team_name) / "inboxes" / f"{recipient}.jsonl"
            inbox.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "id": f"msg_{uuid4().hex[:12]}",
                "from": sender,
                "to": recipient,
                "message": message,
            }
            with inbox.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written.append(recipient)
        return written

    def read(
        self,
        *,
        workspace_root: Path,
        team_name: str,
        agent_name: str,
    ) -> list[dict[str, Any]]:
        inbox = _team_dir(workspace_root, team_name) / "inboxes" / f"{agent_name}.jsonl"
        if not inbox.exists():
            return []
        records = []
        for line in inbox.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                records.append(item)
        inbox.write_text("", encoding="utf-8")
        return records


class TaskListStore:
    def create(
        self,
        *,
        workspace_root: Path,
        team_name: str,
        description: str,
        created_by: str,
    ) -> dict[str, Any]:
        tasks = self.list(workspace_root=workspace_root, team_name=team_name)
        task = {
            "id": f"team_task_{uuid4().hex[:12]}",
            "description": description,
            "status": "pending",
            "assignee": None,
            "created_by": created_by,
        }
        tasks.append(task)
        self._write(workspace_root, team_name, tasks)
        return task

    def list(self, *, workspace_root: Path, team_name: str) -> list[dict[str, Any]]:
        path = _team_dir(workspace_root, team_name) / "tasks.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def update(
        self,
        *,
        workspace_root: Path,
        team_name: str,
        task_id: str,
        status: str | None = None,
        assignee: str | None = None,
    ) -> dict[str, Any] | None:
        tasks = self.list(workspace_root=workspace_root, team_name=team_name)
        target = None
        for task in tasks:
            if task.get("id") == task_id:
                target = task
                break
        if target is None:
            return None
        if status is not None:
            target["status"] = status
        if assignee is not None:
            target["assignee"] = assignee
        self._write(workspace_root, team_name, tasks)
        return target

    def _write(
        self,
        workspace_root: Path,
        team_name: str,
        tasks: list[dict[str, Any]],
    ) -> None:
        path = _team_dir(workspace_root, team_name) / "tasks.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def _team_dir(workspace_root: Path, team_name: str) -> Path:
    return workspace_root / ".agent" / "teams" / team_name


def task_to_dict(task: AgentTask) -> dict[str, Any]:
    return asdict(task)
