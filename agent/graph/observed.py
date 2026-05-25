from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.observe.observer import RuntimeObserver


def observed_node(
    *,
    name: str,
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    observer: RuntimeObserver | None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        if observer:
            observer.on_node_input(name, state)

        try:
            output = fn(state)

            if observer:
                observer.on_node_output(name, output)

            return output

        except Exception as error:
            if observer:
                observer.on_node_error(name, error, state)
            raise

    return wrapped


def observed_route(
    *,
    name: str,
    route_fn: Callable[[dict[str, Any]], str],
    observer: RuntimeObserver | None,
) -> Callable[[dict[str, Any]], str]:
    def wrapped(state: dict[str, Any]) -> str:
        decision = route_fn(state)

        if observer:
            observer.on_route_decision(
                route_name=name,
                decision=decision,
                state=state,
            )

        return decision

    return wrapped
