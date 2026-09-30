"""Opt-in six-role skeleton, independent of the legacy sequential MAS bridge."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable

from software_multiagent.core.action_graph import Action
from software_multiagent.core.contracts import AgentSpec, RuntimeState, TaskContext, ToolResult
from software_multiagent.core.execution import VerificationResult, VerificationStatus
from software_multiagent.core.team import (
    Candidate, CandidateVerdict, MonitorSpec, Rectification, RepairDirective,
    Selection, TeamConfig, TeamPlan, TeamResult,
)
from software_multiagent.ports.protocols import KnowledgeProvider, NativeVerifier
from software_multiagent.ports.team import (
    CandidateEnvironment, CandidateGenerator, CandidateSession,
    ExperienceHeuristics, PlanBuilder, RepairStrategy,
)


@dataclass
class Planner:
    knowledge: KnowledgeProvider
    experience: ExperienceHeuristics
    builder: PlanBuilder

    def plan(self, task: TaskContext) -> TeamPlan:
        agent = AgentSpec("planner", "Decompose the task using knowledge and experience.")
        context = self.knowledge.context_for(task, agent, RuntimeState(task, "plan"))
        hints = self.experience.hints(task, context)
        return TeamPlan(self.builder.decompose(task, context, hints), context, hints)


@dataclass
class Rectifier:
    """Required pre-execution policy; no implicit allow-all default."""

    review: Callable[[Action, TaskContext], Rectification]

    def invoke(self, action: Action, session: CandidateSession) -> ToolResult:
        decision = self.review(deepcopy(action), session.task)
        if decision.action is None:
            return ToolResult(action.id, action.tool, error=f"rectified: {decision.reason}")
        if decision.action.id != action.id:
            raise ValueError("rectification must preserve the action id")
        result = session.execute(decision.action)
        # Preserve the original call identity when the tool itself was rewritten.
        return ToolResult(action.id, action.tool, result.output, result.error)


@dataclass
class Executor:
    id: str
    generator: CandidateGenerator
    rectifier: Rectifier

    def generate(self, session: CandidateSession, plan: TeamPlan, round_id: int) -> Candidate:
        artifacts = self.generator.generate(
            session.task, deepcopy(plan), lambda action: self.rectifier.invoke(action, session)
        )
        return Candidate(f"{self.id}:round-{round_id}", self.id, tuple(artifacts))


@dataclass
class Verifier:
    native: NativeVerifier

    def select(
        self, candidates: tuple[Candidate, ...], sessions: tuple[CandidateSession, ...]
    ) -> Selection:
        if len(candidates) != len(sessions):
            raise ValueError("each candidate requires its own evaluation session")
        verdicts = []
        for candidate, session in zip(candidates, sessions):
            result = (
                VerificationResult(candidate.id, VerificationStatus.FAILED, summary=candidate.error)
                if candidate.error is not None
                else self.native.verify(session.task, candidate.artifacts)
            )
            verdicts.append(CandidateVerdict(candidate, result))
        # Stable first-pass selection. A later ranking policy can compare native scores.
        winner = next((v.candidate.id for v in verdicts if v.verification.passed), None)
        return Selection(tuple(verdicts), winner)


@dataclass
class RepairAgent:
    strategy: RepairStrategy

    def repair(
        self, plan: TeamPlan, selection: Selection, attempt: int,
        sessions: tuple[CandidateSession, ...], checkpoints: tuple[object, ...],
    ) -> RepairDirective:
        directive = self.strategy.revise(plan, selection, attempt)
        if directive.rollback:
            for session, checkpoint in zip(sessions, checkpoints):
                session.rollback(checkpoint)
        return directive


class CandidateTeam:
    """Planner -> parallel rectified Executors -> Verifier -> bounded Repair.

    Callers supply real isolation, evaluator, and model adapters explicitly.
    Infrastructure/permission exceptions propagate and trigger session cleanup.
    Monitor is metadata only and never scheduled.
    """

    def __init__(
        self, *, planner: Planner, executor_factory: Callable[[str], Executor],
        verifier: Verifier, repair: RepairAgent, environments: CandidateEnvironment,
        config: TeamConfig = TeamConfig(), monitor: MonitorSpec = MonitorSpec(),
    ) -> None:
        self.planner = planner
        self.executor_factory = executor_factory
        self.verifier = verifier
        self.repair = repair
        self.environments = environments
        self.config = config
        self.monitor = monitor

    def run(self, task: TaskContext) -> TeamResult:
        plan = self.planner.plan(task)
        history: list[Selection] = []
        repairs = 0
        with ExitStack() as stack:
            executors = tuple(
                self.executor_factory(f"executor-{i + 1}")
                for i in range(self.config.executor_count)
            )
            if len({e.id for e in executors}) != len(executors):
                raise ValueError("executors must have unique identities")
            if len({id(e.generator) for e in executors}) != len(executors):
                raise ValueError("executors must have independent generator sessions")
            sessions = tuple(
                stack.enter_context(self.environments.open(task, executor.id))
                for executor in executors
            )
            if any(not s.isolation_id for s in sessions) or (
                len({s.isolation_id for s in sessions}) != len(sessions)
            ):
                raise ValueError("candidate environments must be independently isolated")
            for round_id in range(self.config.max_repairs + 1):
                checkpoints = tuple(session.checkpoint() for session in sessions)
                with ThreadPoolExecutor(max_workers=self.config.executor_count) as pool:
                    futures = [
                        pool.submit(executor.generate, session, plan, round_id)
                        for executor, session in zip(executors, sessions)
                    ]
                    candidates = tuple(future.result() for future in futures)
                selection = self.verifier.select(candidates, sessions)
                history.append(selection)
                if selection.selected_id is not None:
                    index = next(
                        i for i, candidate in enumerate(candidates)
                        if candidate.id == selection.selected_id
                    )
                    winner = candidates[index]
                    winner = replace(winner, artifacts=sessions[index].export(winner.artifacts))
                    return TeamResult(winner, tuple(history), repairs, "verified_candidate")
                if round_id == self.config.max_repairs:
                    break
                repairs += 1
                directive = self.repair.repair(
                    plan, selection, repairs, sessions, checkpoints
                )
                if not directive.retry:
                    return TeamResult(None, tuple(history), repairs, "repair_stopped")
                plan = directive.plan
        return TeamResult(None, tuple(history), repairs, "repair_limit")
