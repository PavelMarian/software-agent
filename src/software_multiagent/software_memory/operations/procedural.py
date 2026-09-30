"""Executable workflow state and drift detection for domain-neutral software memory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from software_multiagent.software_memory.schema.models import (
    DriftFinding,
    DriftKind,
    ExecutionObservation,
    OperationContract,
    StatePredicate,
    VerificationState,
    Workflow,
    WorkflowCandidate,
    WorkflowExecutionState,
    WorkflowStep,
    WorkflowTransition,
    stable_id,
)


def predicate_slot(predicate: StatePredicate) -> str:
    """Identify one state variable independently from its value or polarity."""

    return json.dumps(
        {
            "predicate": predicate.predicate,
            "subject": predicate.subject,
            "arguments": predicate.arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def predicates_conflict(left: StatePredicate, right: StatePredicate) -> bool:
    if predicate_slot(left) != predicate_slot(right):
        return False
    return left.value != right.value or left.negated != right.negated


def predicate_satisfied(
    required: StatePredicate,
    observed: tuple[StatePredicate, ...],
) -> bool:
    return any(item.key == required.key for item in observed)


def merge_observed_state(
    current: tuple[StatePredicate, ...],
    observed: tuple[StatePredicate, ...],
) -> tuple[StatePredicate, ...]:
    """Apply an observed state delta, replacing prior values of the same state slot."""

    values = {predicate_slot(item): item for item in current}
    for item in observed:
        values[predicate_slot(item)] = item
    return tuple(values[key] for key in sorted(values))


class WorkflowRuntime:
    """Advance a workflow only from independently accepted observations."""

    def __init__(
        self,
        workflow: Workflow,
        contracts: Mapping[str, OperationContract],
    ) -> None:
        self.workflow = workflow
        self.contracts = dict(contracts)
        for step in workflow.steps:
            contract = self.contracts.get(step.operation_id)
            if contract is None or contract.software_id != workflow.software_id:
                raise ValueError(
                    f"workflow step references unavailable contract: {step.operation_id}"
                )
        self.steps = {step.step_id: step for step in workflow.steps}

    def start(
        self,
        run_id: str,
        *,
        observed_state: tuple[StatePredicate, ...] = (),
        produced_artifacts: tuple[str, ...] = (),
    ) -> WorkflowExecutionState:
        return WorkflowExecutionState(
            run_id=run_id,
            software_id=self.workflow.software_id,
            workflow_id=self.workflow.workflow_id,
            active_steps=self.workflow.entry_steps,
            satisfied_predicates=observed_state,
            produced_artifacts=tuple(dict.fromkeys(produced_artifacts)),
        )

    def observe(
        self,
        state: WorkflowExecutionState,
        observation: ExecutionObservation,
        *,
        produced_artifacts: tuple[str, ...] = (),
    ) -> WorkflowTransition:
        self._validate_observation(state, observation)
        contract = (
            self.contracts.get(observation.contract_id)
            if observation.contract_id is not None
            else None
        )
        observed_state = merge_observed_state(
            state.satisfied_predicates,
            observation.state_after,
        )
        accepted = (
            observation.verification == VerificationState.ACCEPTED
            and bool(observation.verifier_id)
        )
        succeeded = accepted and observation.succeeded is True
        failed = accepted and observation.succeeded is False

        active = list(state.active_steps)
        completed = list(state.completed_steps)
        failed_steps = list(state.failed_steps)
        pending = dict(state.pending_verifications)
        advanced: list[str] = []
        recovery: list[str] = []
        candidates = list(state.workflow_candidates)
        findings = self._detect_drift(state, observation, contract, observed_state)

        verified_pending = []
        if succeeded and observation.contract_id:
            for step_id, required in tuple(pending.items()):
                if observation.contract_id not in required:
                    continue
                remaining = tuple(item for item in required if item != observation.contract_id)
                if remaining:
                    pending[step_id] = remaining
                else:
                    pending.pop(step_id, None)
                    verified_pending.append(step_id)
        for step_id in verified_pending:
            self._complete_step(step_id, active, completed, advanced)

        canonical_steps = [
            self.steps[step_id]
            for step_id in tuple(active)
            if contract is not None and self.steps[step_id].operation_id == contract.contract_id
        ]
        if failed:
            for step in canonical_steps:
                if step.step_id in active:
                    active.remove(step.step_id)
                failed_steps.append(step.step_id)
                recovery.extend(step.on_failure)
                active.extend(step.on_failure)
        elif succeeded:
            for step in canonical_steps:
                if not self._step_complete(step, observed_state):
                    continue
                verification_operations = tuple(contract.verification_operations)
                if verification_operations:
                    pending[step.step_id] = verification_operations
                else:
                    self._complete_step(step.step_id, active, completed, advanced)

            if not canonical_steps and contract is not None:
                alternative_steps = [
                    self.steps[step_id]
                    for step_id in tuple(active)
                    if self.steps[step_id].completion_predicates
                    and self._step_newly_completed(
                        self.steps[step_id],
                        observation.state_before,
                        observed_state,
                    )
                ]
                for step in alternative_steps:
                    self._complete_step(step.step_id, active, completed, advanced)
                if alternative_steps:
                    candidates.append(
                        WorkflowCandidate(
                            candidate_id=stable_id(
                                "workflow_candidate",
                                state.run_id,
                                observation.observation_id,
                            ),
                            software_id=state.software_id,
                            source_workflow_id=state.workflow_id,
                            operation_ids=tuple(
                                dict.fromkeys((*state.operation_history, contract.contract_id))
                            ),
                            completion_predicates=tuple(
                                predicate
                                for step in alternative_steps
                                for predicate in step.completion_predicates
                            ),
                            observation_ids=tuple(
                                (*state.observation_history, observation.observation_id)
                            ),
                        )
                    )

        active = list(dict.fromkeys(active))
        completed = list(dict.fromkeys(completed))
        failed_steps = list(dict.fromkeys(failed_steps))
        recovery = list(dict.fromkeys(recovery))
        if not active and completed:
            status = "completed"
        elif not active and failed_steps:
            status = "failed"
        else:
            status = "running"
        next_state = state.model_copy(
            update={
                "active_steps": tuple(active),
                "completed_steps": tuple(completed),
                "failed_steps": tuple(failed_steps),
                "satisfied_predicates": observed_state,
                "produced_artifacts": tuple(
                    dict.fromkeys((*state.produced_artifacts, *produced_artifacts))
                ),
                "operation_history": (
                    state.operation_history
                    if observation.contract_id is None
                    else (*state.operation_history, observation.contract_id)
                ),
                "observation_history": (
                    *state.observation_history,
                    observation.observation_id,
                ),
                "pending_verifications": pending,
                "workflow_candidates": tuple(candidates),
                "status": status,
            }
        )
        return WorkflowTransition(
            previous_state=state,
            state=next_state,
            findings=findings,
            advanced_steps=tuple(advanced),
            recovery_steps=tuple(recovery),
        )

    def _detect_drift(
        self,
        state: WorkflowExecutionState,
        observation: ExecutionObservation,
        contract: OperationContract | None,
        observed_state: tuple[StatePredicate, ...],
    ) -> tuple[DriftFinding, ...]:
        findings: list[DriftFinding] = []
        contract_id = observation.contract_id
        if (
            contract_id
            and state.operation_history
            and state.operation_history[-1] == contract_id
            and {item.key for item in observation.state_before}
            == {item.key for item in observation.state_after}
        ):
            findings.append(
                DriftFinding(
                    kind=DriftKind.REPEATED_WITHOUT_STATE_CHANGE,
                    operation_id=contract_id,
                    message="operation repeated without an observed state change",
                )
            )

        expected_verifications = {
            operation
            for values in state.pending_verifications.values()
            for operation in values
        }
        if expected_verifications and contract_id not in expected_verifications:
            findings.append(
                DriftFinding(
                    kind=DriftKind.VERIFICATION_SKIPPED,
                    operation_id=contract_id,
                    step_ids=tuple(state.pending_verifications),
                    message="required verification operation was skipped",
                )
            )

        active_operations = {self.steps[step_id].operation_id for step_id in state.active_steps}
        completes_active = any(
            self.steps[step_id].completion_predicates
            and self._step_newly_completed(
                self.steps[step_id],
                observation.state_before,
                observed_state,
            )
            for step_id in state.active_steps
        )
        if (
            contract_id
            and contract_id not in active_operations
            and contract_id not in expected_verifications
        ):
            all_workflow_operations = {step.operation_id for step in self.workflow.steps}
            if contract_id in all_workflow_operations and not completes_active:
                findings.append(
                    DriftFinding(
                        kind=DriftKind.INVALID_OPERATION_ORDER,
                        operation_id=contract_id,
                        step_ids=state.active_steps,
                        message=(
                            "operation belongs to the workflow but is not currently enabled"
                        ),
                    )
                )
            elif not completes_active:
                findings.append(
                    DriftFinding(
                        kind=DriftKind.WORKFLOW_BOUNDARY_EXIT,
                        operation_id=contract_id,
                        step_ids=state.active_steps,
                        message="operation left the workflow without equivalent observed completion",
                    )
                )

        if contract is not None and observation.succeeded is True:
            missing_effects = tuple(
                effect.key
                for effect in contract.effects
                if not predicate_satisfied(effect, observed_state)
            )
            if missing_effects:
                findings.append(
                    DriftFinding(
                        kind=DriftKind.EFFECT_MISMATCH,
                        operation_id=contract.contract_id,
                        predicate_ids=missing_effects,
                        message="claimed success did not produce all declared observable effects",
                    )
                )
        return tuple(findings)

    def _complete_step(
        self,
        step_id: str,
        active: list[str],
        completed: list[str],
        advanced: list[str],
    ) -> None:
        if step_id in active:
            active.remove(step_id)
        completed.append(step_id)
        advanced.append(step_id)
        active.extend(self.steps[step_id].on_success)

    @staticmethod
    def _step_complete(
        step: WorkflowStep,
        observed_state: tuple[StatePredicate, ...],
    ) -> bool:
        return not step.completion_predicates or all(
            predicate_satisfied(predicate, observed_state)
            for predicate in step.completion_predicates
        )

    @classmethod
    def _step_newly_completed(
        cls,
        step: WorkflowStep,
        before: tuple[StatePredicate, ...],
        after: tuple[StatePredicate, ...],
    ) -> bool:
        return cls._step_complete(step, after) and not cls._step_complete(step, before)

    def _validate_observation(
        self,
        state: WorkflowExecutionState,
        observation: ExecutionObservation,
    ) -> None:
        if state.workflow_id != self.workflow.workflow_id:
            raise ValueError("workflow execution state belongs to another workflow")
        if state.software_id != self.workflow.software_id:
            raise ValueError("workflow execution state belongs to another software")
        if observation.software_id != state.software_id:
            raise ValueError("observation belongs to another software")
        if observation.task_id != state.run_id:
            raise ValueError("observation belongs to another workflow run")
