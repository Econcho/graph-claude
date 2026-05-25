from __future__ import annotations

import time
from pathlib import Path

from agent.graph.nodes.context_node import make_context_node
from agent.graph.nodes.bootstrap_node import bootstrap_node
from agent.multi_agent import (
    AgentPollTool,
    AgentTool,
    MultiAgentConfig,
    MultiAgentManager,
    SendMessageTool,
    TaskCreateTool,
    TaskListTool,
    TaskStopTool,
    TaskUpdateTool,
    TeamCreateTool,
)
from agent.prompt import PromptRuntime
from agent.tools import PermissionRules, ToolContext, ToolExecutor, ToolRegistry


class FakeLLMNoTool:
    def __init__(self, content: str = "child done"):
        self.content = content
        self.seen_messages = None

    def invoke(self, messages, tools, system_prompt=None):
        self.seen_messages = messages
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [],
        }


class FakeObserver:
    def __init__(self):
        self.events = []

    def emit(self, result_type, *, output_node, input_node, content=None):
        self.events.append(
            {
                "result_type": result_type,
                "output_node": output_node,
                "input_node": input_node,
                "content": content or {},
            }
        )


def _manager(llm=None, observer=None, **kwargs) -> MultiAgentManager:
    manager = MultiAgentManager(
        llm or FakeLLMNoTool(),
        config=MultiAgentConfig(**kwargs),
        observer=observer,
    )
    manager.set_registry_factory(lambda teammate=False: ToolRegistry([]))
    return manager


def test_agent_tool_runs_sync_subagent(tmp_path: Path):
    llm = FakeLLMNoTool("subagent result")
    manager = _manager(llm)
    executor = ToolExecutor(
        ToolRegistry([AgentTool(manager)]),
        permission_rules=PermissionRules.empty(),
    )

    result = executor.execute(
        {
            "id": "call_1",
            "name": "agent",
            "args": {
                "description": "Summarize",
                "prompt": "Do isolated work",
            },
        },
        ToolContext(workspace_root=tmp_path, session_id="s1"),
    )

    assert result.ok is True
    assert result.content == "subagent result"
    assert llm.seen_messages is not None
    assert len(llm.seen_messages) == 1


def test_agent_background_task_can_be_polled(tmp_path: Path):
    manager = _manager(FakeLLMNoTool("background done"))
    registry = ToolRegistry([AgentTool(manager), AgentPollTool(manager)])
    executor = ToolExecutor(registry, permission_rules=PermissionRules.empty())

    start = executor.execute(
        {
            "id": "call_1",
            "name": "agent",
            "args": {
                "description": "Background",
                "prompt": "Work later",
                "run_in_background": True,
            },
        },
        ToolContext(workspace_root=tmp_path, session_id="s1"),
    )

    assert start.ok is True
    task_id = start.data["task_id"]

    polled = None
    for _ in range(20):
        polled = executor.execute(
            {"id": "call_2", "name": "agent_poll", "args": {"task_id": task_id}},
            ToolContext(workspace_root=tmp_path, session_id="s1"),
        )
        if polled.data["status"] == "completed":
            break
        time.sleep(0.05)

    assert polled is not None
    assert polled.ok is True
    assert polled.data["status"] == "completed"
    assert polled.data["final_answer"] == "background done"


def test_subagent_depth_limit(tmp_path: Path):
    manager = _manager(max_subagent_depth=2)
    result = AgentTool(manager).run(
        {
            "description": "Too deep",
            "prompt": "No-op",
        },
        ToolContext(workspace_root=tmp_path, subagent_depth=2),
    )

    assert result.ok is False
    assert result.error == "subagent_depth_limit"


def test_coordinator_prompt_is_injected(tmp_path: Path):
    result = PromptRuntime(ToolRegistry([])).build_main_request(
        {
            "workspace_root": str(tmp_path),
            "messages": [],
            "coordinator_mode": True,
        }
    )

    assert "Coordinator mode is active" in result.system_prompt


def test_bootstrap_uses_multi_agent_settings_defaults(tmp_path: Path):
    output = bootstrap_node(
        {
            "user_input": "hello",
            "workspace_root": str(tmp_path),
        }
    )

    assert output["multi_agent_enabled"] is True
    assert output["coordinator_mode"] is False


def test_context_node_consumes_multi_agent_notifications(tmp_path: Path):
    manager = _manager()
    task = manager.task_store.create(
        agent_id="subagent:worker",
        name="worker",
        description="work",
        prompt="do work",
    )
    manager.task_store.update(task.task_id, status="completed", final_answer="done")

    output = make_context_node(multi_agent_manager=manager)(
        {
            "workspace_root": str(tmp_path),
            "messages": [],
            "session_id": "s1",
        }
    )

    notifications = output["context_snapshot"]["multi_agent_notifications"]
    assert len(notifications) == 1
    assert "<task-notification" in notifications[0]
    assert "done" in notifications[0]


def test_team_create_writes_team_files(tmp_path: Path):
    manager = _manager()
    result = TeamCreateTool(manager).run(
        {"team_name": "alpha", "leader": "lead"},
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is True
    assert (tmp_path / ".agent" / "teams" / "alpha" / "team.json").exists()
    assert (tmp_path / ".agent" / "teams" / "alpha" / "tasks.json").exists()


def test_send_message_and_broadcast_write_mailboxes(tmp_path: Path):
    manager = _manager()
    manager.team_store.create_team(
        workspace_root=tmp_path,
        team_name="alpha",
        leader="lead",
    )
    manager.team_store.register_agent(
        workspace_root=tmp_path,
        team_name="alpha",
        agent_name="a",
    )
    manager.team_store.register_agent(
        workspace_root=tmp_path,
        team_name="alpha",
        agent_name="b",
    )

    result = SendMessageTool(manager).run(
        {"to": "*", "message": "hello"},
        ToolContext(workspace_root=tmp_path, team_name="alpha", agent_name="lead"),
    )

    assert result.ok is True
    inbox_a = tmp_path / ".agent" / "teams" / "alpha" / "inboxes" / "a.jsonl"
    inbox_b = tmp_path / ".agent" / "teams" / "alpha" / "inboxes" / "b.jsonl"
    assert inbox_a.exists()
    assert inbox_b.exists()


def test_shared_task_tools(tmp_path: Path):
    manager = _manager()
    manager.team_store.create_team(
        workspace_root=tmp_path,
        team_name="alpha",
        leader="lead",
    )
    ctx = ToolContext(workspace_root=tmp_path, team_name="alpha", agent_name="lead")

    created = TaskCreateTool(manager).run({"description": "Implement x"}, ctx)
    assert created.ok is True
    task_id = created.data["id"]

    listed = TaskListTool(manager).run({}, ctx)
    assert listed.ok is True
    assert listed.data["tasks"][0]["id"] == task_id

    updated = TaskUpdateTool(manager).run(
        {"task_id": task_id, "status": "in_progress", "assignee": "lead"},
        ctx,
    )
    assert updated.ok is True
    assert updated.data["status"] == "in_progress"

    stopped = TaskStopTool(manager).run({"task_id": task_id}, ctx)
    assert stopped.ok is True
    assert stopped.data["status"] == "stopped"


def test_teammate_cannot_spawn_teammate(tmp_path: Path):
    manager = _manager()
    result = AgentTool(manager).run(
        {
            "description": "Spawn teammate",
            "prompt": "work",
            "team_name": "alpha",
            "name": "worker",
        },
        ToolContext(workspace_root=tmp_path, team_name="alpha", agent_name="lead"),
    )

    assert result.ok is False
    assert result.error == "teammate_spawn_denied"


def test_teammate_cannot_spawn_background_agent(tmp_path: Path):
    manager = _manager()
    result = AgentTool(manager).run(
        {
            "description": "Spawn background",
            "prompt": "work",
            "run_in_background": True,
        },
        ToolContext(workspace_root=tmp_path, team_name="alpha", agent_name="lead"),
    )

    assert result.ok is False
    assert result.error == "teammate_background_denied"


def test_multi_agent_tools_still_use_permission_system(tmp_path: Path):
    manager = _manager()
    executor = ToolExecutor(
        ToolRegistry([AgentTool(manager)]),
        permission_rules=PermissionRules.empty(),
        permission_asker=lambda tool, tool_input, decision: False,
    )

    result = executor.execute(
        {
            "id": "call_1",
            "name": "agent",
            "args": {
                "description": "Ask",
                "prompt": "work",
            },
        },
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is True


def test_multi_agent_observe_events(tmp_path: Path):
    observer = FakeObserver()
    manager = _manager(observer=observer)

    TeamCreateTool(manager).run(
        {"team_name": "alpha", "leader": "lead"},
        ToolContext(workspace_root=tmp_path),
    )

    assert any(event["result_type"] == "team_created" for event in observer.events)
