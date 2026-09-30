"""Contracts for the candidate-based MAS skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from software_multiagent.core.action_graph import Action
from software_multiagent.core.execution import (
    ArtifactRef,
    KnowledgeContext,
    VerificationResult,
    WorkItem,
)


class TeamRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    RECTIFIER = "rectifier"
    VERIFIER = "verifier"
    REPAIR = "repair"
    MONITOR = "monitor"


@dataclass(frozen=True)
class TeamConfig:
    executor_count: int = 3
    max_repairs: int = 2

    def __post_init__(self) -> None:
        if self.executor_count < 1:
            raise ValueError("executor_count must be positive")
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")


@dataclass(frozen=True)
class TeamPlan:
    work_items: tuple[WorkItem, ...]
    context: KnowledgeContext = KnowledgeContext()
    experience_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rectification:
    # None rejects the call. Otherwise execute only this approved/revised action.
    action: Action | None
    reason: str = ""


@dataclass(frozen=True)
class Candidate:
    id: str
    executor_id: str
    artifacts: tuple[ArtifactRef, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class CandidateVerdict:
    candidate: Candidate
    verification: VerificationResult


@dataclass(frozen=True)
class Selection:
    verdicts: tuple[CandidateVerdict, ...]
    selected_id: str | None


@dataclass(frozen=True)
class RepairDirective:
    plan: TeamPlan
    retry: bool = True
    rollback: bool = True


@dataclass(frozen=True)
class TeamResult:
    selected: Candidate | None
    rounds: tuple[Selection, ...]
    repairs: int
    stop_reason: str


@dataclass(frozen=True)
class MonitorSpec:
    """Reserved configuration only; no worker, drift detection, or restart yet."""

    canonical_path: tuple[str, ...] = ()
    observed_roles: tuple[TeamRole, ...] = (
        TeamRole.PLANNER, TeamRole.EXECUTOR, TeamRole.RECTIFIER,
        TeamRole.VERIFIER, TeamRole.REPAIR,
    )
