"""Injection points; production models, isolation and native checks are host-owned."""

from __future__ import annotations

from typing import Callable, ContextManager, Protocol

from software_multiagent.core.action_graph import Action
from software_multiagent.core.contracts import TaskContext, ToolResult
from software_multiagent.core.execution import ArtifactRef, KnowledgeContext, WorkItem
from software_multiagent.core.team import RepairDirective, Selection, TeamPlan


class ExperienceHeuristics(Protocol):
    """ERL extension point; implementations may retrieve learned experience."""

    def hints(self, task: TaskContext, context: KnowledgeContext) -> tuple[str, ...]: ...


class PlanBuilder(Protocol):
    def decompose(
        self, task: TaskContext, context: KnowledgeContext, hints: tuple[str, ...]
    ) -> tuple[WorkItem, ...]: ...


class CandidateGenerator(Protocol):
    """One private executor session; all software calls must use invoke."""

    def generate(
        self, task: TaskContext, plan: TeamPlan, invoke: Callable[[Action], ToolResult]
    ) -> tuple[ArtifactRef, ...]: ...


class CandidateSession(Protocol):
    """Isolated files AND external software state, not merely a distinct path.

    execute must enforce tool permissions and validate arguments even after
    rectification. checkpoint/rollback cover the target software's mutable state.
    export copies selected outputs to durable storage before session cleanup.
    """

    task: TaskContext
    isolation_id: str

    def execute(self, action: Action) -> ToolResult: ...

    def checkpoint(self) -> object: ...

    def rollback(self, checkpoint: object) -> None: ...

    def export(self, artifacts: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]: ...


class CandidateEnvironment(Protocol):
    def open(
        self, task: TaskContext, executor_id: str
    ) -> ContextManager[CandidateSession]: ...


class RepairStrategy(Protocol):
    def revise(self, plan: TeamPlan, selection: Selection, attempt: int) -> RepairDirective: ...
