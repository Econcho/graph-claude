from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import convert_to_messages

from agent.graph.state import AgentState
from agent.memory.session_memory import (
    SessionMemoryManager,
    post_llm_session_memory_hook,
)
from agent.observe.observer import RuntimeObserver
from agent.tools.adapters.langgraph_adapter import internal_model_specs_to_langchain_tools


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    if response is None:
        return []

    if isinstance(response, dict):
        return response.get("tool_calls", []) or []

    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        return tool_calls

    additional_kwargs = getattr(response, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        return additional_kwargs.get("tool_calls", []) or []

    return []


def _invoke_llm(llm_client, llm_request: dict[str, Any]):
    system_prompt = llm_request.get("system_prompt")
    messages = llm_request.get("messages", [])
    tool_specs = llm_request.get("tools", [])

    if hasattr(llm_client, "bind_tools"):
        bindable_tools = internal_model_specs_to_langchain_tools(tool_specs)
        bound_model = llm_client.bind_tools(bindable_tools)
        langchain_messages = convert_to_messages(messages)
        if system_prompt:
            langchain_messages = [SystemMessage(content=system_prompt), *langchain_messages]
        return bound_model.invoke(langchain_messages)

    return llm_client.invoke(
        messages=messages,
        tools=tool_specs,
        system_prompt=system_prompt,
    )


def make_llm_node(
    llm_client,
    observer: RuntimeObserver | None = None,
    session_memory_manager: SessionMemoryManager | None = None,
):
    def llm_node(state: AgentState) -> dict[str, Any]:
        llm_request = state.get("llm_request")

        if not llm_request:
            return {
                "error": {
                    "type": "missing_llm_request",
                    "message": "llm_request is required before llm_node.",
                }
            }

        if observer:
            observer.on_llm_start(llm_request)

        try:
            response = _invoke_llm(llm_client, llm_request)
        except Exception as error:
            if observer:
                observer.on_llm_error(error, llm_request)
            raise

        tool_calls = _extract_tool_calls(response)

        if observer:
            observer.on_llm_end(response, tool_calls)

        updated_state = {
            **state,
            "assistant_message": response,
            "messages": [*state.get("messages", []), response],
        }
        session_memory_update = post_llm_session_memory_hook(
            updated_state,
            session_memory_manager,
        )

        return {
            "assistant_message": response,
            "messages": [response],
            **session_memory_update,
        }

    return llm_node
