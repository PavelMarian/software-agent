"""Auditable, Alembic-inspired plans for mutating professional software state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from software_multiagent.core.execution import SoftwareInterface


@dataclass(frozen=True)
class Action:
    id: str
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    mutates_workspace: bool = False
    verification: tuple[str, ...] = ()
    rollback: str | None = None
    artifact_inputs: tuple[str, ...] = ()
    artifact_outputs: tuple[str, ...] = ()
    interface: SoftwareInterface | None = None
    expected_effect: str = ""
    reversible: bool = False


@dataclass
class ActionGraph:
    actions: dict[str, Action] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    cancelled: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    revision: int = 0

    def add(self, action: Action) -> None:
        if not action.id:
            raise ValueError("action id must be non-empty")
        if action.id in self.actions:
            raise ValueError(f"duplicate action: {action.id}")
        if action.id in action.depends_on:
            raise ValueError("action cannot depend on itself")
        self.actions[action.id] = action
        self.revision += 1

    def replace(self, action: Action) -> None:
        """Rewrite a pending node while retaining a monotonic graph revision."""
        if action.id not in self.actions:
            raise ValueError(f"unknown action: {action.id}")
        if action.id in self.completed | self.cancelled:
            raise ValueError(f"cannot replace terminal action: {action.id}")
        previous = self.actions[action.id]
        self.actions[action.id] = action
        try:
            self.validate()
        except ValueError:
            self.actions[action.id] = previous
            raise
        self.failed.discard(action.id)
        self.revision += 1

    def cancel(self, action_id: str) -> None:
        self._require_pending(action_id)
        self.cancelled.add(action_id)
        self.revision += 1

    def mark_failed(self, action_id: str) -> None:
        self._require_pending(action_id)
        self.failed.add(action_id)
        self.revision += 1

    def validate(self) -> None:
        ids = set(self.actions)
        missing = {
            dependency
            for action in self.actions.values()
            for dependency in action.depends_on
            if dependency not in ids
        }
        if missing:
            raise ValueError(f"unknown dependencies: {', '.join(sorted(missing))}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(action_id: str) -> None:
            if action_id in visiting:
                raise ValueError("action graph contains a cycle")
            if action_id in visited:
                return
            visiting.add(action_id)
            for dependency in self.actions[action_id].depends_on:
                visit(dependency)
            visiting.remove(action_id)
            visited.add(action_id)

        for action_id in self.actions:
            visit(action_id)

    def ready(self) -> tuple[Action, ...]:
        return tuple(
            action
            for action in self.actions.values()
            if action.id not in self.completed | self.cancelled | self.failed
            and set(action.depends_on).issubset(self.completed)
        )

    def blocked(self) -> tuple[Action, ...]:
        terminal_without_success = self.cancelled | self.failed
        return tuple(
            action
            for action in self.actions.values()
            if action.id not in self.completed | terminal_without_success
            and bool(set(action.depends_on) & terminal_without_success)
        )

    def mark_completed(self, action_id: str) -> None:
        if action_id not in self.actions:
            raise ValueError(f"unknown action: {action_id}")
        if not set(self.actions[action_id].depends_on).issubset(self.completed):
            raise ValueError(f"dependencies are incomplete for action: {action_id}")
        self.completed.add(action_id)
        self.failed.discard(action_id)
        self.revision += 1

    def _require_pending(self, action_id: str) -> None:
        if action_id not in self.actions:
            raise ValueError(f"unknown action: {action_id}")
        if action_id in self.completed | self.cancelled:
            raise ValueError(f"action is already terminal: {action_id}")
