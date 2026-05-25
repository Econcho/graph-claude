from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AgentTaskStatus = Literal["running", "completed", "failed", "killed"]


@dataclass(frozen=True)
class MultiAgentConfig:
    enabled: bool = True
    coordinator_enabled: bool = False
    max_background_agents: int = 4
    max_subagent_depth: int = 2
    default_subagent_tools: list[str] = field(
        default_factory=lambda: ["read_file", "write_file", "agent_poll"]
    )
    default_teammate_tools: list[str] = field(
        default_factory=lambda: [
            "read_file",
            "write_file",
            "send_message",
            "task_create",
            "task_list",
            "task_update",
            "task_stop",
        ]
    )
    notifications_enabled: bool = True
    mailbox_poll_enabled: bool = True


@dataclass
class AgentTask:
    task_id: str
    agent_id: str
    name: str | None
    description: str
    prompt: str
    status: AgentTaskStatus = "running"
    team_name: str | None = None
    final_answer: str | None = None
    error: str | None = None
    result_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentNotification:
    task_id: str
    agent_id: str
    status: AgentTaskStatus
    content: str

    def to_xml(self) -> str:
        return (
            f'<task-notification agent_id="{self.agent_id}" status="{self.status}">\n'
            f"task_id: {self.task_id}\n"
            f"{self.content}\n"
            "</task-notification>"
        )
