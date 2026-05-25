from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.memory.context_compaction.full_compact_prompt import (
    FULL_COMPACT_PROMPT,
    NO_TOOLS_PREAMBLE,
)
from agent.memory.context_compaction.session_compact import (
    _adjust_start_to_preserve_tool_pairs,
)
from agent.memory.context_compaction.token_counter import estimate_message_tokens
from agent.memory.session_memory.serializer import serialize_messages


FULL_COMPACT_SUMMARY_PREFIX = """This conversation was compacted because it was approaching the context limit.

Summary:
{summary}

Recent messages are preserved verbatim below.
Continue from where the conversation left off. Do not ask the user to repeat context.
"""


@dataclass(frozen=True)
class FullCompactResult:
    compacted: bool
    messages: list[Any]
    status: str
    reason: str
    summary_chars: int = 0
    error: str | None = None


class FullCompactAgent:
    def __init__(
        self,
        llm_client,
        *,
        recent_keep_max_tokens: int = 20_000,
        max_output_chars: int = 80_000,
    ):
        self.llm_client = llm_client
        self.recent_keep_max_tokens = recent_keep_max_tokens
        self.max_output_chars = max_output_chars

    def compact(self, messages: list[Any]) -> FullCompactResult:
        try:
            response = self._invoke(messages)
            if _extract_tool_calls(response):
                return FullCompactResult(
                    compacted=False,
                    messages=list(messages),
                    status="failed",
                    reason="tool_call_returned",
                    error="Full compact agent returned tool calls.",
                )

            content = _message_content(response)
            summary = format_compact_summary(
                content,
                max_chars=self.max_output_chars,
            )
            status = "full_compacted" if "<summary>" in content else "full_compacted_unstructured"
            recent_messages = _recent_messages_for_full_compact(
                messages,
                recent_keep_max_tokens=self.recent_keep_max_tokens,
            )
            summary_message = HumanMessage(
                content=FULL_COMPACT_SUMMARY_PREFIX.format(summary=summary)
            )
            return FullCompactResult(
                compacted=True,
                messages=[summary_message, *recent_messages],
                status=status,
                reason="threshold_exceeded",
                summary_chars=len(summary),
            )

        except Exception as error:
            return FullCompactResult(
                compacted=False,
                messages=list(messages),
                status="failed",
                reason="exception",
                error=str(error),
            )

    def _invoke(self, messages: list[Any]) -> Any:
        user_prompt = _build_full_compact_user_prompt(messages)
        if hasattr(self.llm_client, "bind_tools"):
            return self.llm_client.invoke(
                [
                    SystemMessage(content=NO_TOOLS_PREAMBLE),
                    HumanMessage(content=user_prompt),
                ]
            )

        return self.llm_client.invoke(
            messages=[{"role": "user", "content": user_prompt}],
            tools=[],
            system_prompt=NO_TOOLS_PREAMBLE,
        )


def format_compact_summary(text: str, *, max_chars: int = 80_000) -> str:
    without_analysis = re.sub(
        r"<analysis>[\s\S]*?</analysis>",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    match = re.search(
        r"<summary>([\s\S]*?)</summary>",
        without_analysis,
        flags=re.IGNORECASE,
    )
    if match:
        summary = match.group(1).strip()
    else:
        summary = without_analysis.strip()

    summary = re.sub(r"\n{3,}", "\n\n", summary).strip()
    if len(summary) > max_chars:
        omitted = len(summary) - max_chars
        summary = f"{summary[:max_chars]}\n...[truncated {omitted} chars]"
    return summary


def _build_full_compact_user_prompt(messages: list[Any]) -> str:
    return f"""{FULL_COMPACT_PROMPT}

Transcript:
{serialize_messages(messages)}
"""


def _recent_messages_for_full_compact(
    messages: list[Any],
    *,
    recent_keep_max_tokens: int,
) -> list[Any]:
    if not messages:
        return []

    start = len(messages)
    while start > 0:
        candidate = list(messages[start - 1 :])
        if estimate_message_tokens(candidate) > recent_keep_max_tokens:
            break
        start -= 1

    if start == len(messages):
        start = max(0, len(messages) - 1)

    start = _adjust_start_to_preserve_tool_pairs(messages, start)
    return list(messages[start:])


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, dict):
        return message.get("tool_calls", []) or []

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return tool_calls

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        return additional_kwargs.get("tool_calls", []) or []

    return []
