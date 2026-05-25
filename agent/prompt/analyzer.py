from __future__ import annotations

from agent.prompt.models import PromptSection


def estimate_prompt_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def with_token_estimate(section: PromptSection) -> PromptSection:
    return PromptSection(
        name=section.name,
        content=section.content,
        source=section.source,
        cacheable=section.cacheable,
        cache_break=section.cache_break,
        token_estimate=estimate_prompt_tokens(section.content),
    )


def section_stats(sections: list[PromptSection]) -> list[dict]:
    return [
        {
            "name": section.name,
            "source": section.source,
            "cacheable": section.cacheable,
            "cache_break": section.cache_break,
            "char_count": len(section.content),
            "token_estimate": section.token_estimate,
        }
        for section in sections
    ]
