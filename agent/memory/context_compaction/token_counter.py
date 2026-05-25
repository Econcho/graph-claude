from __future__ import annotations

from typing import Any

from agent.memory.session_memory.serializer import serialize_messages


def estimate_message_tokens(messages: list[Any]) -> int:
    serialized = serialize_messages(messages)
    return max(1, len(serialized) // 4) if serialized else 0
