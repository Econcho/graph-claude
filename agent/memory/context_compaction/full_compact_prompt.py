from __future__ import annotations


NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use read_file, write_file, bash, grep, or any other tool.
- You already have all context you need in the transcript below.
- Tool calls will be treated as failure.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.
"""


FULL_COMPACT_PROMPT = """Create a compact continuation summary for a coding-agent conversation.

Output exactly:

<analysis>
Briefly inspect the transcript and decide what must be preserved.
</analysis>

<summary>
1. Primary Request and Intent
2. Current State
3. Important Files and Facts
4. Decisions Made
5. Tool Results and Observations
6. Errors and Fixes
7. Pending Work
8. Immediate Next Step
</summary>

Rules:
- Preserve exact user preferences and corrections.
- Preserve the current unfinished task and immediate next step.
- Preserve important file paths, code facts, tool outcomes, errors, and fixes.
- Do not invent facts not present in the transcript.
- Keep the summary concise but sufficient for another agent to continue.
"""
