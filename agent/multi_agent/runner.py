from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent.multi_agent.models import MultiAgentConfig
from agent.tools.registry import ToolRegistry


class SubagentRunner:
    def __init__(
        self,
        *,
        llm_client,
        observer=None,
        config: MultiAgentConfig,
        registry_factory,
        multi_agent_manager=None,
    ):
        self.llm_client = llm_client
        self.observer = observer
        self.config = config
        self.registry_factory = registry_factory
        self.multi_agent_manager = multi_agent_manager

    def run(
        self,
        *,
        prompt: str,
        workspace_root: Path,
        session_id: str | None,
        permission_mode: str,
        agent_id: str,
        agent_name: str | None,
        team_name: str | None,
        subagent_depth: int,
        teammate: bool = False,
    ) -> dict[str, Any]:
        from agent.graph import build_graph

        registry: ToolRegistry = self.registry_factory(teammate=teammate)
        graph = build_graph(
            llm_client=self.llm_client,
            registry=registry,
            observer=self.observer,
            multi_agent_manager=self.multi_agent_manager,
        )
        return graph.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "workspace_root": str(workspace_root),
                "permission_mode": permission_mode,
                "session_id": session_id,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "team_name": team_name,
                "subagent_depth": subagent_depth,
                "auto_memory_enabled": False,
                "session_memory_enabled": False,
                "multi_agent_enabled": self.config.enabled,
                "coordinator_mode": False,
            },
            config={
                "configurable": {
                    "thread_id": f"{session_id or 'subagent'}:{agent_id}"
                }
            },
        )
