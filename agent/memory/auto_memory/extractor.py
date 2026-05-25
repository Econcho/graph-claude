from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.memory.session_memory.serializer import serialize_messages


AUTO_MEMORY_EXTRACTOR_SYSTEM_PROMPT = """You are an Auto Memory extraction agent.

You are not the main coding agent.
Your job is to identify durable cross-session memories from a completed run.
Return JSON only. Do not continue the user's task.

Store only durable information:
- user preferences
- feedback about how the agent should behave
- non-obvious project background that cannot be read from files or git
- useful external references

Never store as Auto Memory:
- code patterns, architecture, file paths, project structure, module summaries,
  function summaries, or API details that can be inferred from source code
- git history, recent changes, PR lists, or activity summaries; git log,
  git blame, and repository history are authoritative
- debugging plans, fix steps, root-cause notes, error traces, or workaround notes
- content already present in CLAUDE.md
- temporary task details, ongoing work, or current conversation context

If the user asks to save a PR list or activity summary, the main agent should ask:
"What is surprising or non-obvious?" Only that answer is suitable memory.
"""


@dataclass(frozen=True)
class AutoMemoryExtractorInput:
    messages: list[Any]
    final_answer: str
    memory_index: str
    memory_manifest: list[dict[str, Any]]
    runtime_metadata: dict[str, Any]
    recent_messages_window: int = 12


@dataclass(frozen=True)
class AutoMemoryExtractorResult:
    status: str
    proposals: list[dict[str, Any]]
    error: str | None = None


class AutoMemoryExtractor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract(
        self,
        extractor_input: AutoMemoryExtractorInput,
    ) -> AutoMemoryExtractorResult:
        try:
            response = self._invoke(self._build_user_prompt(extractor_input))
            content = _message_content(response)
            payload = _parse_json_object(content)
            proposals = payload.get("memory_proposals", [])
            if not isinstance(proposals, list):
                raise ValueError("memory_proposals must be a list.")
            return AutoMemoryExtractorResult(
                status="success",
                proposals=[_normalize_proposal(item) for item in proposals],
            )
        except Exception as error:
            return AutoMemoryExtractorResult(
                status="failed",
                proposals=[],
                error=str(error),
            )

    def _invoke(self, user_prompt: str) -> Any:
        if hasattr(self.llm_client, "bind_tools"):
            return self.llm_client.invoke(
                [
                    SystemMessage(content=AUTO_MEMORY_EXTRACTOR_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )

        return self.llm_client.invoke(
            messages=[{"role": "user", "content": user_prompt}],
            tools=[],
            system_prompt=AUTO_MEMORY_EXTRACTOR_SYSTEM_PROMPT,
        )

    def _build_user_prompt(self, extractor_input: AutoMemoryExtractorInput) -> str:
        messages = list(extractor_input.messages)
        recent_start = max(0, len(messages) - extractor_input.recent_messages_window)
        recent_messages = messages[recent_start:]
        metadata = json.dumps(
            extractor_input.runtime_metadata,
            ensure_ascii=False,
            indent=2,
        )
        manifest = json.dumps(
            extractor_input.memory_manifest,
            ensure_ascii=False,
            indent=2,
        )

        return f"""Extract Auto Memory proposals from this completed main-agent run.

Output schema:
{{
  "memory_proposals": [
    {{
      "action": "create | update | none",
      "type": "user | feedback | project | reference",
      "name": "short stable name",
      "description": "one sentence",
      "content": "durable memory content",
      "why_it_matters": "why future runs should know this",
      "how_to_apply": "how future runs should use this memory",
      "target_memory_id": "required only for update",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Use action "none" when nothing durable should be stored.
- Prefer update when an existing memory should be refined.
- Do not propose delete.
- Do not store secrets or full file contents.
- Do not store facts that the agent can reliably read from files.
- Do not store code patterns, architecture, file paths, project structure,
  function summaries, module summaries, or API details.
- Do not store git history, recent changes, PR lists, or activity summaries.
- Do not store debugging plans, fix steps, root-cause notes, traces, or
  workaround notes.
- Do not duplicate CLAUDE.md content.
- Do not store temporary task details, pending work, or current conversation
  context.
- If the user asked to save a PR list or activity summary, return action "none"
  unless the conversation identifies what is surprising or non-obvious.

Runtime metadata:
{metadata}

Existing MEMORY.md index:
{extractor_input.memory_index}

Existing memory manifest:
{manifest}

Recent messages:
{serialize_messages(recent_messages, start_index=recent_start)}

Final answer:
{extractor_input.final_answer}
"""


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("Extractor output must be a JSON object.")
    return payload


def _normalize_proposal(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"action": "none", "confidence": 0.0}

    action = str(value.get("action") or "none").strip().lower()
    if action not in {"create", "update", "none"}:
        action = "none"

    memory_type = str(value.get("type") or "project").strip().lower()
    if memory_type not in {"user", "feedback", "project", "reference"}:
        memory_type = "project"

    try:
        confidence = float(value.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "action": action,
        "type": memory_type,
        "name": str(value.get("name") or "").strip(),
        "description": str(value.get("description") or "").strip(),
        "content": str(value.get("content") or "").strip(),
        "why_it_matters": str(value.get("why_it_matters") or "").strip(),
        "how_to_apply": str(value.get("how_to_apply") or "").strip(),
        "target_memory_id": str(value.get("target_memory_id") or "").strip(),
        "confidence": max(0.0, min(1.0, confidence)),
    }
