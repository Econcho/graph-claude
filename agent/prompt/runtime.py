from __future__ import annotations

from typing import Any

from agent.observe.observer import RuntimeObserver
from agent.prompt.analyzer import section_stats
from agent.prompt.config import PromptConfig, load_prompt_config
from agent.prompt.contexts import SystemContextProvider, UserContextProvider
from agent.prompt.defaults import DefaultSystemPromptBuilder
from agent.prompt.dump import PromptDumpService
from agent.prompt.dynamic import DynamicPromptSectionBuilder
from agent.prompt.models import PromptBuildResult, PromptSection
from agent.prompt.resolver import EffectiveSystemPromptResolver
from agent.prompt.specialized import SpecializedPromptRegistry
from agent.prompt.tools import ToolSpecProvider
from agent.tools.registry import ToolRegistry


class PromptRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        config: PromptConfig | None = None,
        observer: RuntimeObserver | None = None,
    ):
        self.registry = registry
        self.config = config or load_prompt_config()
        self.observer = observer
        self.default_builder = DefaultSystemPromptBuilder()
        self.user_context = UserContextProvider(self.config)
        self.system_context = SystemContextProvider(self.config)
        self.dynamic_builder = DynamicPromptSectionBuilder()
        self.tool_specs = ToolSpecProvider(registry)
        self.resolver = EffectiveSystemPromptResolver()
        self.specialized_prompts = SpecializedPromptRegistry()

    def build_main_request(self, state: dict[str, Any]) -> PromptBuildResult:
        self._emit(
            "prompt_runtime_start",
            {"enabled": self.config.enabled},
        )
        try:
            result = self._build_main_request(state)
        except Exception as error:
            self._emit(
                "prompt_runtime_error",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            raise

        self._emit(
            "prompt_effective_resolved",
            {
                "section_count": len(result.sections),
                "message_count": len(result.messages),
                "tool_count": len(result.tools),
                "snapshot_ref": result.snapshot_ref,
            },
        )
        return result

    def _build_main_request(self, state: dict[str, Any]) -> PromptBuildResult:
        context_snapshot = state.get("context_snapshot", {})
        messages = context_snapshot.get("llm_messages") or state.get("messages", [])
        tools = self.tool_specs.build(state)

        default_sections = self.default_builder.build()
        dynamic_sections = [
            *self.user_context.build(state),
            *self.system_context.build(state),
            *self.dynamic_builder.build(state),
        ]

        custom_system_prompt = state.get("custom_system_prompt")
        if (
            custom_system_prompt is None
            and not state.get("prompt_runtime_managed", False)
            and state.get("system_prompt")
        ):
            custom_system_prompt = state.get("system_prompt")

        sections = self.resolver.resolve(
            default_sections=default_sections,
            dynamic_sections=dynamic_sections,
            override_system_prompt=state.get("override_system_prompt"),
            agent_system_prompt=state.get("agent_system_prompt"),
            custom_system_prompt=custom_system_prompt,
            append_system_prompt=state.get("append_system_prompt"),
        )

        stats = section_stats(sections)
        for section in sections:
            self._emit(
                "prompt_section_built",
                {
                    "name": section.name,
                    "source": section.source,
                    "cacheable": section.cacheable,
                    "cache_break": section.cache_break,
                    "token_estimate": section.token_estimate,
                    "char_count": len(section.content),
                },
            )

        system_prompt = "\n\n".join(
            section.content.strip() for section in sections if section.content.strip()
        ).rstrip() + "\n"
        dump = PromptDumpService(
            enabled=self.config.dump_enabled and bool(self.observer and self.observer.enabled),
            trace_path=self.observer.trace_path if self.observer else None,
        )
        snapshot, snapshot_ref = dump.write_snapshot(
            system_prompt=system_prompt,
            messages=list(messages),
            tools=tools,
            sections=sections,
            section_stats=stats,
        )
        if snapshot_ref:
            self._emit(
                "prompt_dump_written",
                {"prompt_snapshot_ref": snapshot_ref},
            )

        return PromptBuildResult(
            system_prompt=system_prompt,
            messages=list(messages),
            tools=tools,
            sections=sections,
            snapshot=snapshot,
            snapshot_ref=snapshot_ref,
        )

    def _emit(self, event_type: str, content: dict[str, Any]) -> None:
        if not self.observer:
            return
        self.observer.emit(
            event_type,
            output_node="prompt_runtime",
            input_node="prompt",
            content=content,
        )
