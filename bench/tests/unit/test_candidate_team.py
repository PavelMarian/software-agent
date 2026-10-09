from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from threading import Barrier

import pytest

from software_multiagent.core.action_graph import Action
from software_multiagent.core.contracts import TaskContext, ToolResult
from software_multiagent.core.execution import (
    KnowledgeContext, VerificationResult, VerificationStatus, WorkItem,
)
from software_multiagent.core.team import Rectification, RepairDirective, TeamConfig, TeamRole
from software_multiagent.runtime.orchestration.team import (
    CandidateTeam, Executor, Planner, Rectifier, RepairAgent, Verifier,
)


class FakeKnowledge:
    def context_for(self, task, agent, state):
        assert agent.id == "planner"
        return KnowledgeContext()


class FakeExperience:
    def hints(self, task, context):
        return ("fake experience",)


class FakePlanBuilder:
    def decompose(self, task, context, hints):
        assert hints == ("fake experience",)
        return (WorkItem("solve", task.objective),)


class FakeSession:
    def __init__(self, task, identifier):
        self.task = replace(task, workspace=task.workspace / identifier)
        self.isolation_id = identifier
        self.calls = []
        self.rollbacks = 0
        self.exports = 0
        self.closed = False

    def execute(self, action):
        self.calls.append(action)
        return ToolResult(action.id, action.tool, output="executed")

    def checkpoint(self):
        return len(self.calls)

    def rollback(self, checkpoint):
        self.rollbacks += 1
        del self.calls[checkpoint:]

    def export(self, artifacts):
        self.exports += 1
        return artifacts


class FakeEnvironments:
    def __init__(self):
        self.sessions = []

    @contextmanager
    def open(self, task, executor_id):
        session = FakeSession(task, executor_id)
        self.sessions.append(session)
        try:
            yield session
        finally:
            session.closed = True


class FakeGenerator:
    def __init__(self, barrier):
        self.barrier = barrier

    def generate(self, task, plan, invoke):
        # Sequential dispatch would time out here. No wall-time speed assertion.
        self.barrier.wait(timeout=5)
        assert invoke(Action("call-1", "proposed")).ok
        return ()


class FakeNativeEvaluator:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.calls = []

    def verify(self, task, artifacts):
        self.calls.append(task.workspace)
        return VerificationResult("native", next(self.statuses))


class FakeRepair:
    def __init__(self, retry=True, rollback=True):
        self.attempts = []
        self.retry = retry
        self.rollback = rollback

    def revise(self, plan, selection, attempt):
        assert selection.selected_id is None
        self.attempts.append(attempt)
        return RepairDirective(plan, retry=self.retry, rollback=self.rollback)


def make_team(statuses, *, config=TeamConfig(), strategy=None):
    environments = FakeEnvironments()
    native = FakeNativeEvaluator(statuses)
    strategy = strategy or FakeRepair()
    barrier = Barrier(config.executor_count)
    team = CandidateTeam(
        planner=Planner(FakeKnowledge(), FakeExperience(), FakePlanBuilder()),
        executor_factory=lambda identifier: Executor(
            identifier, FakeGenerator(barrier),
            Rectifier(lambda action, task: Rectification(replace(action, tool="approved"))),
        ),
        verifier=Verifier(native), repair=RepairAgent(strategy),
        environments=environments, config=config,
    )
    return team, environments, native, strategy


def test_three_parallel_candidates_are_rectified_and_independently_verified(tmp_path):
    team, environments, native, strategy = make_team([
        VerificationStatus.FAILED, VerificationStatus.PASSED, VerificationStatus.PASSED,
    ])
    result = team.run(TaskContext("task", "solve", tmp_path))
    assert result.selected.executor_id == "executor-2"
    assert len(set(native.calls)) == 3
    assert [s.exports for s in environments.sessions] == [0, 1, 0]
    assert all(s.calls[0].tool == "approved" for s in environments.sessions)
    assert all(s.closed for s in environments.sessions)
    assert strategy.attempts == []
    assert TeamRole.MONITOR.value == "monitor"
    assert team.monitor.canonical_path == ()


def test_repair_limit_is_two_retries_and_three_verified_rounds(tmp_path):
    team, environments, native, strategy = make_team([VerificationStatus.FAILED] * 9)
    result = team.run(TaskContext("task", "solve", tmp_path))
    assert result.selected is None
    assert result.stop_reason == "repair_limit"
    assert result.repairs == 2
    assert len(result.rounds) == 3
    assert len(native.calls) == 9
    assert strategy.attempts == [1, 2]
    assert all(s.rollbacks == 2 for s in environments.sessions)


def test_repair_success_requires_new_native_verification(tmp_path):
    team, environments, native, _ = make_team(
        [VerificationStatus.INCONCLUSIVE] * 3 + [VerificationStatus.PASSED] * 3
    )
    result = team.run(TaskContext("task", "solve", tmp_path))
    assert result.repairs == 1
    assert result.selected.id == "executor-1:round-1"
    assert len(native.calls) == 6
    assert all(s.rollbacks == 1 for s in environments.sessions)


def test_repair_can_stop_without_retry(tmp_path):
    team, environments, native, _ = make_team(
        [VerificationStatus.FAILED] * 3, strategy=FakeRepair(retry=False, rollback=False)
    )
    result = team.run(TaskContext("task", "solve", tmp_path))
    assert result.stop_reason == "repair_stopped"
    assert len(native.calls) == 3
    assert all(s.rollbacks == 0 for s in environments.sessions)


def test_rejected_call_never_reaches_software(tmp_path):
    session = FakeSession(TaskContext("task", "solve", tmp_path), "one")
    rectifier = Rectifier(lambda action, task: Rectification(None, "invalid call"))
    result = rectifier.invoke(Action("call", "dangerous"), session)
    assert not result.ok
    assert session.calls == []


def test_evaluator_exception_fails_closed_and_releases_sessions(tmp_path):
    team, environments, _, _ = make_team([])
    with pytest.raises(StopIteration):
        team.run(TaskContext("task", "solve", tmp_path))
    assert all(s.closed and s.exports == 0 for s in environments.sessions)


@pytest.mark.parametrize("kwargs", [{"executor_count": 0}, {"max_repairs": -1}])
def test_invalid_team_limits(kwargs):
    with pytest.raises(ValueError):
        TeamConfig(**kwargs)
