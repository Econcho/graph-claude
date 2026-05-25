from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages

from agent.memory.session_memory.store import REQUIRED_SECTIONS, SessionMemoryStore
from agent.memory.session_memory.tools import EditSessionMemoryTool
from agent.tools.adapters.langgraph_adapter import internal_model_specs_to_langchain_tools
from agent.tools.base import ToolContext


FORK_AGENT_SYSTEM_PROMPT = """You are a Session Memory update agent.

You are not the main coding agent.
You must not continue the user's task or answer the user.
Your only goal is to update the current session memory.

You must call edit_session_memory exactly once with the complete updated markdown.
Do not use any other tool.
Do not wrap the markdown in a code block.
"""


@dataclass(frozen=True)
class SessionMemoryForkInput:
    old_session_memory: str
    new_messages_since_last_summary: str
    recent_messages_window: str
    runtime_metadata: dict[str, Any]


@dataclass(frozen=True)
class SessionMemoryForkResult:
    status: str
    turns: int
    tool_called: bool
    content_chars: int = 0
    error: str | None = None


class SessionMemoryForkAgent:
    def __init__(
        self,
        llm_client,
        store: SessionMemoryStore,
        *,
        max_turns: int = 3,
    ):
        self.llm_client = llm_client
        self.store = store
        self.max_turns = max_turns
        self.tool = EditSessionMemoryTool(store)

    def run(self, fork_input: SessionMemoryForkInput) -> SessionMemoryForkResult:
        messages: list[Any] = [
            HumanMessage(content=self._build_user_prompt(fork_input)),
        ]
        tool_specs = [self.tool.to_model_spec()]
        ctx = ToolContext(
            workspace_root=self.store.workspace_root,
            permission_mode="accept_edits",
            session_id=self.store.session_id,
            agent_id="session_memory_fork_agent",
        )

        for turn in range(1, self.max_turns + 1):
            response = self._invoke(messages, tool_specs)
            messages.append(response)
            tool_calls = _extract_tool_calls(response)

            if not tool_calls:
                continue

            call = tool_calls[0]
            tool_name = call.get("name")
            if tool_name != self.tool.name:
                return SessionMemoryForkResult(
                    status="failed",
                    turns=turn,
                    tool_called=False,
                    error=f"Unexpected tool call: {tool_name}",
                )

            tool_input = call.get("args") or call.get("input") or {}
            try:
                self.tool.validate_input(tool_input, ctx)
                result = self.tool.run(tool_input, ctx)
            except Exception as error:
                return SessionMemoryForkResult(
                    status="failed",
                    turns=turn,
                    tool_called=True,
                    error=str(error),
                )

            messages.append(
                ToolMessage(
                    content=result.content,
                    tool_call_id=call.get("id") or "",
                    name=self.tool.name,
                    status="success" if result.ok else "error",
                )
            )

            if result.ok:
                data = result.data or {}
                return SessionMemoryForkResult(
                    status="success",
                    turns=turn,
                    tool_called=True,
                    content_chars=int(data.get("content_chars") or 0),
                )

            return SessionMemoryForkResult(
                status="failed",
                turns=turn,
                tool_called=True,
                error=result.content,
            )

        return SessionMemoryForkResult(
            status="failed",
            turns=self.max_turns,
            tool_called=False,
            error="Fork agent did not call edit_session_memory.",
        )

    def _invoke(self, messages: list[Any], tool_specs: list[dict[str, Any]]) -> Any:
        if hasattr(self.llm_client, "bind_tools"):
            bindable_tools = internal_model_specs_to_langchain_tools(tool_specs)
            bound_model = self.llm_client.bind_tools(bindable_tools)
            langchain_messages = [
                SystemMessage(content=FORK_AGENT_SYSTEM_PROMPT),
                *convert_to_messages(messages),
            ]
            return bound_model.invoke(langchain_messages)

        return self.llm_client.invoke(
            messages=messages,
            tools=tool_specs,
            system_prompt=FORK_AGENT_SYSTEM_PROMPT,
        )

    def _build_user_prompt(self, fork_input: SessionMemoryForkInput) -> str:
        required_sections = "\n".join(f"- {section}" for section in REQUIRED_SECTIONS)
        metadata = "\n".join(
            f"- {key}: {value}" for key, value in fork_input.runtime_metadata.items()
        )

        return f"""Update the Session Memory markdown.

Required sections:
{required_sections}

Runtime metadata:
{metadata}

Old Session Memory:
{fork_input.old_session_memory}

New messages since last summary:
{fork_input.new_messages_since_last_summary}

Recent messages window:
{fork_input.recent_messages_window}

Rules:
- Preserve still-valid information.
- Remove or rewrite stale information.
- Do not invent facts not present in messages.
- Keep Pending Work only for unfinished work.
- Call edit_session_memory with the complete updated markdown.
"""


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    if message is None:
        return []

    if isinstance(message, dict):
        return message.get("tool_calls", []) or []

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return tool_calls

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        return additional_kwargs.get("tool_calls", []) or []

    return []
