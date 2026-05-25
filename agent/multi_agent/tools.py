from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from agent.multi_agent.manager import MultiAgentManager


class AgentTool:
    name = "agent"
    description = "Run an isolated subagent or teammate for a delegated task."
    input_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "subagent_type": {"type": "string"},
            "model": {"type": "string"},
            "run_in_background": {"type": "boolean", "default": False},
            "name": {"type": "string"},
            "team_name": {"type": "string"},
            "mode": {"type": "string"},
        },
        "required": ["description", "prompt"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = False

    def __init__(self, manager: "MultiAgentManager"):
        self.manager = manager

    def is_enabled(self, ctx: ToolContext) -> bool:
        return self.manager.config.enabled

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _require_string(tool_input, "description")
        _require_string(tool_input, "prompt")
        if "run_in_background" in tool_input and not isinstance(
            tool_input["run_in_background"], bool
        ):
            raise ValueError("run_in_background must be a boolean")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.spawn_agent(
            description=tool_input["description"],
            prompt=tool_input["prompt"],
            ctx=ctx,
            name=tool_input.get("name"),
            team_name=tool_input.get("team_name"),
            run_in_background=bool(tool_input.get("run_in_background", False)),
            mode=tool_input.get("mode"),
        )


class AgentPollTool:
    name = "agent_poll"
    description = "Poll background agent task status and result."
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
        },
    }
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def __init__(self, manager: "MultiAgentManager"):
        self.manager = manager

    def is_enabled(self, ctx: ToolContext) -> bool:
        return self.manager.config.enabled

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        if "task_id" in tool_input and not isinstance(tool_input["task_id"], str):
            raise ValueError("task_id must be a string")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.poll(tool_input.get("task_id"))


class SendMessageTool:
    name = "send_message"
    description = "Send a message to a teammate inbox. Use to='*' to broadcast."
    input_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["to", "message"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = False

    def __init__(self, manager: "MultiAgentManager"):
        self.manager = manager

    def is_enabled(self, ctx: ToolContext) -> bool:
        return self.manager.config.enabled

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _require_string(tool_input, "to")
        _require_string(tool_input, "message")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.send_message(to=tool_input["to"], message=tool_input["message"], ctx=ctx)


class TeamCreateTool:
    name = "team_create"
    description = "Create a workspace-local in-process agent team."
    input_schema = {
        "type": "object",
        "properties": {
            "team_name": {"type": "string"},
            "leader": {"type": "string", "default": "leader"},
        },
        "required": ["team_name"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = False

    def __init__(self, manager: "MultiAgentManager"):
        self.manager = manager

    def is_enabled(self, ctx: ToolContext) -> bool:
        return self.manager.config.enabled

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _require_string(tool_input, "team_name")
        if "leader" in tool_input and not isinstance(tool_input["leader"], str):
            raise ValueError("leader must be a string")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.create_team(
            team_name=tool_input["team_name"],
            leader=tool_input.get("leader") or "leader",
            ctx=ctx,
        )


class TaskCreateTool:
    name = "task_create"
    description = "Create a shared team task."
    input_schema = {
        "type": "object",
        "properties": {"description": {"type": "string"}},
        "required": ["description"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = False

    def __init__(self, manager: "MultiAgentManager"):
        self.manager = manager

    def is_enabled(self, ctx: ToolContext) -> bool:
        return self.manager.config.enabled

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _require_string(tool_input, "description")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.create_shared_task(description=tool_input["description"], ctx=ctx)


class TaskListTool:
    name = "task_list"
    description = "List shared team tasks."
    input_schema = {"type": "object", "properties": {}}
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def __init__(self, manager: "MultiAgentManager"):
        self.manager = manager

    def is_enabled(self, ctx: ToolContext) -> bool:
        return self.manager.config.enabled

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        return

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.list_shared_tasks(ctx)


class TaskUpdateTool:
    name = "task_update"
    description = "Update a shared team task status or assignee."
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "status": {"type": "string"},
            "assignee": {"type": "string"},
        },
        "required": ["task_id"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = False

    def __init__(self, manager: "MultiAgentManager"):
        self.manager = manager

    def is_enabled(self, ctx: ToolContext) -> bool:
        return self.manager.config.enabled

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _require_string(tool_input, "task_id")
        if "status" in tool_input and not isinstance(tool_input["status"], str):
            raise ValueError("status must be a string")
        if "assignee" in tool_input and not isinstance(tool_input["assignee"], str):
            raise ValueError("assignee must be a string")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.update_shared_task(
            task_id=tool_input["task_id"],
            status=tool_input.get("status"),
            assignee=tool_input.get("assignee"),
            ctx=ctx,
        )


class TaskStopTool:
    name = "task_stop"
    description = "Mark a shared team task as stopped."
    input_schema = {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = False

    def __init__(self, manager: "MultiAgentManager"):
        self.manager = manager

    def is_enabled(self, ctx: ToolContext) -> bool:
        return self.manager.config.enabled

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _require_string(tool_input, "task_id")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.stop_shared_task(task_id=tool_input["task_id"], ctx=ctx)


def collaboration_tools(manager: "MultiAgentManager") -> list:
    return [
        AgentTool(manager),
        AgentPollTool(manager),
        SendMessageTool(manager),
        TeamCreateTool(manager),
        TaskCreateTool(manager),
        TaskListTool(manager),
        TaskUpdateTool(manager),
        TaskStopTool(manager),
    ]


def _require_string(tool_input: dict[str, Any], key: str) -> None:
    value = tool_input.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
