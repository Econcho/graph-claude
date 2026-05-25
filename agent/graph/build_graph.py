from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.config import build_default_model
from agent.graph.nodes import (
    bootstrap_node,
    make_context_node,
    make_finalize_node,
    make_llm_node,
    make_prompt_node,
    make_tool_executor_node,
    make_tool_result_node,
    tool_orchestrator_node,
)
from agent.graph.observed import observed_node, observed_route
from agent.graph.routes import route_after_llm, route_after_tool_result
from agent.graph.state import AgentState
from agent.memory.auto_memory import AutoMemoryManager
from agent.memory.context_compaction import ContextCompactionManager
from agent.memory.session_memory import SessionMemoryManager
from agent.mcp import McpManager
from agent.multi_agent import MultiAgentManager, collaboration_tools
from agent.observe.observer import RuntimeObserver
from agent.skills import ShellCommandTool, SkillManager, UseSkillTool
from agent.tools import BUILTIN_TOOLS, ToolExecutor, ToolRegistry


def build_graph(
    llm_client=None,
    registry: ToolRegistry | None = None,
    checkpointer=None,
    model=None,
    observer: RuntimeObserver | None = None,
    multi_agent_manager: MultiAgentManager | None = None,
):
    llm_client = llm_client or model or build_default_model()
    skill_manager = SkillManager(
        llm_client,
        observer=observer,
    )
    multi_agent_manager = multi_agent_manager or MultiAgentManager(
        llm_client,
        observer=observer,
    )
    multi_agent_tools = collaboration_tools(multi_agent_manager)

    def registry_factory(*, teammate: bool = False) -> ToolRegistry:
        allowed = (
            set(multi_agent_manager.config.default_teammate_tools)
            if teammate
            else set(multi_agent_manager.config.default_subagent_tools)
        )
        candidates = [
            *BUILTIN_TOOLS,
            ShellCommandTool(),
            UseSkillTool(skill_manager),
            *multi_agent_tools,
        ]
        return ToolRegistry(
            [
                tool
                for tool in candidates
                if tool.name in allowed or any(alias in allowed for alias in tool.aliases)
            ]
        )

    multi_agent_manager.set_registry_factory(registry_factory)

    if registry is None:
        base_tools = [
            *BUILTIN_TOOLS,
            ShellCommandTool(),
            UseSkillTool(skill_manager),
            *multi_agent_tools,
        ]
        mcp_tools = McpManager(observer=observer).discover_tools(
            skip_names=_tool_names_and_aliases(base_tools),
        )
        registry = ToolRegistry([*base_tools, *mcp_tools])
    tool_executor = ToolExecutor(registry, observer=observer)
    session_memory_manager = SessionMemoryManager(
        llm_client,
        observer=observer,
    )
    auto_memory_manager = AutoMemoryManager(
        llm_client,
        observer=observer,
    )
    context_compaction_manager = ContextCompactionManager(
        llm_client,
        observer=observer,
    )

    context = make_context_node(
        auto_memory_manager=auto_memory_manager,
        context_compaction_manager=context_compaction_manager,
        skill_manager=skill_manager,
        multi_agent_manager=multi_agent_manager,
    )
    prompt = make_prompt_node(registry, observer=observer)
    llm = make_llm_node(
        llm_client,
        observer=observer,
        session_memory_manager=session_memory_manager,
    )
    tool_executor_node = make_tool_executor_node(tool_executor, observer=observer)
    tool_result = make_tool_result_node(observer=observer)
    finalize = make_finalize_node(auto_memory_manager=auto_memory_manager)

    graph = StateGraph(AgentState)
    graph.add_node(
        "bootstrap",
        observed_node(name="bootstrap", fn=bootstrap_node, observer=observer),
    )
    graph.add_node(
        "context",
        observed_node(name="context", fn=context, observer=observer),
    )
    graph.add_node(
        "prompt",
        observed_node(name="prompt", fn=prompt, observer=observer),
    )
    graph.add_node(
        "llm",
        observed_node(name="llm", fn=llm, observer=observer),
    )
    graph.add_node(
        "tool_executor",
        observed_node(
            name="tool_executor",
            fn=tool_executor_node,
            observer=observer,
        ),
    )
    graph.add_node(
        "tool_orchestrator",
        observed_node(
            name="tool_orchestrator",
            fn=tool_orchestrator_node,
            observer=observer,
        ),
    )
    graph.add_node(
        "tool_result",
        observed_node(name="tool_result", fn=tool_result, observer=observer),
    )
    graph.add_node(
        "finalize",
        observed_node(name="finalize", fn=finalize, observer=observer),
    )

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "context")
    graph.add_edge("context", "prompt")
    graph.add_edge("prompt", "llm")
    graph.add_conditional_edges(
        "llm",
        observed_route(
            name="route_after_llm",
            route_fn=route_after_llm,
            observer=observer,
        ),
        {
            "tool_orchestrator": "tool_orchestrator",
            "finalize": "finalize",
        },
    )
    graph.add_edge("tool_orchestrator", "tool_executor")
    graph.add_edge("tool_executor", "tool_result")
    graph.add_conditional_edges(
        "tool_result",
        observed_route(
            name="route_after_tool_result",
            route_fn=route_after_tool_result,
            observer=observer,
        ),
        {
            "tool_orchestrator": "tool_orchestrator",
            "context": "context",
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def _tool_names_and_aliases(tools) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        names.add(tool.name)
        names.update(tool.aliases)
    return names
