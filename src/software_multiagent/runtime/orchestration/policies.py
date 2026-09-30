"""Replaceable coordination policies over the topology-neutral agent runtime."""

from __future__ import annotations

from typing import Any, Protocol

from software_multiagent.core.contracts import RunSpec, RuntimeState, WorkflowNode


class CoordinationPolicy(Protocol):
    """Select graph entry and resolve a completed node's typed route."""

    def entry_node(self, spec: RunSpec, state: RuntimeState) -> str: ...

    def next_node(
        self,
        spec: RunSpec,
        state: RuntimeState,
        node: WorkflowNode,
        route: str,
    ) -> str | None: ...


class StaticGraphPolicy:
    """Deterministic baseline policy; adaptive policies can replace this object."""

    def entry_node(self, spec: RunSpec, state: RuntimeState) -> str:
        del state
        return spec.entry_node

    def next_node(
        self,
        spec: RunSpec,
        state: RuntimeState,
        node: WorkflowNode,
        route: str,
    ) -> str | None:
        del spec, state
        if route not in node.transitions:
            raise ValueError(f"unknown_route:{node.id}:{route}")
        return node.transitions[route]


class MemoryGatedResearchPolicy(StaticGraphPolicy):
    """Route to research only after the shared memory has assessed every gap."""

    def __init__(self, knowledge: Any | None) -> None:
        self.knowledge = knowledge

    def next_node(
        self,
        spec: RunSpec,
        state: RuntimeState,
        node: WorkflowNode,
        route: str,
    ) -> str | None:
        if node.id != "initial_plan" or route not in {"complete", "action_limit"}:
            return super().next_node(spec, state, node, route)
        gate = getattr(self.knowledge, "assess_research_need", None)
        if not callable(gate):
            # No usable persistent memory: preserve the safe legacy path.
            return "research"
        decision = gate(state.task)
        if decision.research_required:
            return "research"
        if decision.memory_resolved_count:
            return "final_plan"
        return "execute"
