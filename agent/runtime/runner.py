from __future__ import annotations

from typing import Any

from agent.graph import build_graph
from agent.observe.mermaid import write_mermaid_from_trace
from agent.observe.run_recorder import create_observer
from agent.observe.summary import write_summary_from_trace


class AgentRunner:
    def __init__(
        self,
        llm_client,
        *,
        observe: bool = True,
        observe_dir: str = ".runs",
        observe_config: str | None = None,
        auto_memory: bool = True,
    ):
        self.auto_memory = auto_memory
        self.observer = create_observer(
            base_dir=observe_dir,
            enabled=observe,
            config_path=observe_config,
        )
        self.graph = build_graph(
            llm_client=llm_client,
            observer=self.observer,
        )

    def run(
        self,
        initial_state: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        initial_state = {
            "auto_memory_enabled": self.auto_memory,
            **initial_state,
        }
        self.observer.on_run_start(initial_state)

        try:
            result = self.graph.invoke(initial_state, config=config)
        except Exception as error:
            self.observer.on_run_error(error, initial_state)
            self._write_mermaid_trace()
            raise

        self.observer.on_run_end(result)
        self._write_mermaid_trace()
        return result

    def _write_mermaid_trace(self) -> None:
        if not self.observer.enabled or not self.observer.trace_path:
            return
        try:
            write_mermaid_from_trace(self.observer.trace_path)
            write_summary_from_trace(self.observer.trace_path)
        except Exception:
            return
