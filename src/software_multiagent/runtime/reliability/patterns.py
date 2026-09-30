"""Composable lifecycle mechanisms for one software agent.

Extensions operate on public task context only. Never inject a hidden benchmark
evaluator as an outcome check. Hooks run in registration order and fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

from software_multiagent.core.action_graph import Action
from software_multiagent.core.contracts import AgentSpec, RuntimeState, ToolResult
from software_multiagent.core.execution import VerificationResult, VerificationStatus
from software_multiagent.ports.protocols import KnowledgeProvider
from software_multiagent.runtime.reliability.mechanisms import (
    ExitCodeValidator,
    MetadataKnowledgeProvider,
)


class Phase(str, Enum):
    CONTEXT = "context"
    BEFORE_ACTION = "before_action"
    AFTER_ACTION = "after_action"
    BEFORE_FINISH = "before_finish"


@dataclass(frozen=True)
class HookContext:
    state: RuntimeState
    agent: AgentSpec
    action: Action | None = None
    result: ToolResult | None = None


@dataclass(frozen=True)
class HookResult:
    checks: tuple[VerificationResult, ...] = ()
    context: str = ""


class Mechanism:
    """Subclass and override only the hooks needed by the new pattern."""

    name = "mechanism"

    def apply(self, phase: Phase, context: HookContext) -> HookResult:
        return HookResult()


class MechanismRegistry:
    def __init__(self, mechanisms: Sequence[Mechanism] = ()) -> None:
        self._items: dict[str, Mechanism] = {}
        for item in mechanisms:
            self.register(item)

    def register(self, mechanism: Mechanism) -> None:
        if not mechanism.name or mechanism.name in self._items:
            raise ValueError(f"invalid or duplicate mechanism: {mechanism.name}")
        self._items[mechanism.name] = mechanism

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._items)

    @property
    def has_outcome_checks(self) -> bool:
        return any(isinstance(item, OutcomeChecks) for item in self._items.values())

    def apply(self, phase: Phase, context: HookContext) -> HookResult:
        checks: list[VerificationResult] = []
        parts: list[str] = []
        for item in self._items.values():
            try:
                result = item.apply(phase, context)
                checks.extend(result.checks)
                if result.context:
                    parts.append(f"[{item.name}]\n{result.context}")
            except PermissionError:
                raise
            except Exception as error:
                if getattr(error, "abort_agent_runtime", False):
                    raise
                checks.append(VerificationResult(
                    item.name, VerificationStatus.INCONCLUSIVE,
                    summary=f"mechanism error: {type(error).__name__}: {error}",
                ))
        return HookResult(tuple(checks), "\n\n".join(parts))


class TaskKnowledge(Mechanism):
    name = "task_knowledge"

    def __init__(self, provider: KnowledgeProvider | None = None) -> None:
        self.provider = provider or MetadataKnowledgeProvider()

    def apply(self, phase: Phase, context: HookContext) -> HookResult:
        if phase != Phase.CONTEXT:
            return HookResult()
        pack = self.provider.context_for(context.state.task, context.agent, context.state)
        return HookResult(context=pack.render())


class ExecutionFeedback(Mechanism):
    name = "execution_feedback"

    def apply(self, phase: Phase, context: HookContext) -> HookResult:
        if phase != Phase.AFTER_ACTION or context.action is None or context.result is None:
            return HookResult()
        output = context.result.output
        if context.result.ok and isinstance(output, Mapping) and output.get("status") in (
                "failed", "error", "cancelled"):
            return HookResult(checks=(VerificationResult(context.action.id + ":status",
                VerificationStatus.FAILED, summary=f"Operation {output['status']}: {output.get('error', '')}"),))
        if context.result.ok and isinstance(output, Mapping) and (
            output.get("status") in ("pending", "running", "queued")
            or (output.get("session_id") is not None and output.get("exit_code") is None)
        ):
            return HookResult(checks=(VerificationResult(
                context.action.id + ":pending", VerificationStatus.INCONCLUSIVE,
                summary="Operation is still running; observe its completion before dependent work.",
            ),))
        check = ExitCodeValidator().verify(context.action, context.result, context.state.task)
        return HookResult(checks=(check,) if check is not None else ())


class OutcomeChecks(Mechanism):
    """Required public checks; process success alone never grants task completion."""

    name = "outcome_checks"

    def __init__(self, checks: Sequence[Callable[[HookContext], VerificationResult]]) -> None:
        self.checks = tuple(checks)

    def apply(self, phase: Phase, context: HookContext) -> HookResult:
        if phase != Phase.BEFORE_FINISH:
            return HookResult()
        if not self.checks:
            return HookResult(checks=(VerificationResult(
                self.name, VerificationStatus.INCONCLUSIVE,
                summary="No public outcome validator configured; completion is unverified.",
            ),))
        results = []
        for index, check in enumerate(self.checks):
            try:
                result = check(context)
                if not isinstance(result, VerificationResult):
                    raise TypeError("validator must return VerificationResult")
                results.append(result)
            except PermissionError:
                raise
            except Exception as error:
                if getattr(error, "abort_agent_runtime", False):
                    raise
                results.append(VerificationResult(f"outcome-{index}", VerificationStatus.INCONCLUSIVE,
                                                  summary=f"validator unavailable: {error}"))
        return HookResult(checks=tuple(results))


class VerificationGate(OutcomeChecks):
    """Named acceptance gate; preserves OutcomeChecks extension compatibility."""


class PreExecutionGate:
    """Evaluate trusted adapter prerequisites without invoking its action handler."""

    def __init__(self, gateway):
        self.gateway = gateway

    def check(self, action, task):
        results = []
        for index, predicate in enumerate(self.gateway.contract_for(action.tool).preconditions):
            try:
                check = predicate(action, task)
                if not isinstance(check, VerificationResult):
                    raise TypeError("precondition must return VerificationResult")
                results.append(check)
            except PermissionError:
                raise
            except Exception as error:
                if getattr(error, "abort_agent_runtime", False):
                    raise
                results.append(VerificationResult(f"precondition-{index}", VerificationStatus.INCONCLUSIVE,
                                                  summary=f"precondition unavailable: {error}"))
        return tuple(results)


def default_mechanisms(
    checks: Sequence[Callable[[HookContext], VerificationResult]],
    *, knowledge: KnowledgeProvider | None = None,
) -> MechanismRegistry:
    return MechanismRegistry((TaskKnowledge(knowledge), ExecutionFeedback(), VerificationGate(checks)))
