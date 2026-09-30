"""Traceable verification evidence, deterministic verdicts, and targeted repair."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Sequence

from pydantic import Field, model_validator

from software_multiagent.software_memory.schema.acceptance import AcceptanceContract, AcceptanceCriterion
from software_multiagent.software_memory.schema.models import ActorKind, FrozenModel, StatePredicate, stable_id, utc_now
from software_multiagent.software_memory.schema.verification import (
    CandidateStatus,
    EvidenceStrength,
    VerificationMethodType,
    VerificationPlan,
)


def _digest(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(text.encode("utf-8")).hexdigest()


class SignalOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class AssuranceLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RepairFailureType(str, Enum):
    VERIFICATION_FAILURE = "verification_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class VerificationSignal(FrozenModel):
    """Immutable verifier interpretation linked to, but separate from, raw output."""

    signal_id: str
    software_id: str
    task_id: str
    task_requirement_id: str
    acceptance_contract_id: str
    acceptance_contract_hash: str
    verification_plan_id: str
    criterion_id: str
    candidate_id: str
    operation_contract_id: str | None = None
    observation_id: str
    evidence_ids: tuple[str, ...] = ()
    evidence_source_ids: tuple[str, ...] = ()
    method_type: VerificationMethodType
    strength: EvidenceStrength
    outcome: SignalOutcome
    native_check: bool = False
    verifier_id: str
    verifier_kind: ActorKind
    independent_mechanism_id: str
    cross_run_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_verifier(self) -> "VerificationSignal":
        if self.verifier_kind == ActorKind.AGENT:
            raise ValueError("executor/agent cannot issue a verification signal")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("verification evidence IDs must be unique")
        return self


class CriterionVerdict(FrozenModel):
    criterion_id: str
    requirement_id: str
    mandatory: bool
    outcome: SignalOutcome
    assurance: AssuranceLevel
    signal_ids: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class AcceptanceVerdict(FrozenModel):
    verdict_id: str
    software_id: str
    task_id: str
    intent_lock_id: str
    intent_lock_hash: str
    acceptance_contract_id: str
    acceptance_contract_hash: str
    verification_plan_id: str
    outcome: SignalOutcome
    assurance: AssuranceLevel
    criteria: tuple[CriterionVerdict, ...]
    mandatory_coverage: float = Field(ge=0.0, le=1.0)
    missing_evidence: tuple[str, ...] = ()
    verdict_hash: str
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_hash(self) -> "AcceptanceVerdict":
        if _digest(self.hash_payload()) != self.verdict_hash:
            raise ValueError("acceptance verdict hash is invalid")
        return self

    def hash_payload(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "software_id": self.software_id,
            "intent_lock_id": self.intent_lock_id,
            "intent_lock_hash": self.intent_lock_hash,
            "acceptance_contract_id": self.acceptance_contract_id,
            "acceptance_contract_hash": self.acceptance_contract_hash,
            "verification_plan_id": self.verification_plan_id,
            "outcome": self.outcome.value,
            "assurance": self.assurance.value,
            "criteria": [item.model_dump(mode="json", exclude_none=True) for item in self.criteria],
            "mandatory_coverage": self.mandatory_coverage,
            "missing_evidence": self.missing_evidence,
        }


class TargetedRepairPackage(FrozenModel):
    repair_package_id: str
    software_id: str
    task_id: str
    intent_lock_id: str
    intent_lock_hash: str
    acceptance_contract_id: str
    acceptance_contract_hash: str
    criterion_id: str
    failure_type: RepairFailureType
    observation_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    related_contract_ids: tuple[str, ...] = ()
    repair_targets: tuple[StatePredicate, ...] = ()
    required_next_state: tuple[StatePredicate, ...] = ()
    prohibited_changes: tuple[str, ...] = (
        "locked_intent",
        "mandatory_acceptance_conditions",
    )
    rationale: str

    def assert_preserves(self, contract: AcceptanceContract) -> None:
        if (
            contract.intent_lock_id != self.intent_lock_id
            or contract.intent_lock_hash != self.intent_lock_hash
            or contract.acceptance_contract_id != self.acceptance_contract_id
            or contract.contract_hash != self.acceptance_contract_hash
        ):
            raise ValueError("repair attempts to change locked intent or acceptance conditions")


class VerdictAggregator:
    """Pure aggregation: identical contract, plan, and signals yield the same verdict."""

    def aggregate(
        self,
        contract: AcceptanceContract,
        plan: VerificationPlan,
        signals: Sequence[VerificationSignal],
    ) -> AcceptanceVerdict:
        if (
            plan.acceptance_contract_id != contract.acceptance_contract_id
            or plan.acceptance_contract_hash != contract.contract_hash
        ):
            raise ValueError("verification plan does not match the acceptance contract")
        candidates = {item.candidate_id: item for item in plan.candidates}
        criterion_ids = {item.criterion_id for item in contract.criteria}
        for signal in signals:
            if (
                signal.task_id != contract.task_id
                or signal.software_id != contract.software_id
                or signal.acceptance_contract_id != contract.acceptance_contract_id
                or signal.acceptance_contract_hash != contract.contract_hash
                or signal.verification_plan_id != plan.plan_id
                or signal.criterion_id not in criterion_ids
                or signal.candidate_id not in candidates
            ):
                raise ValueError("verification signal does not belong to this evidence graph")
        criterion_results = tuple(
            self._criterion_verdict(
                criterion,
                [signal for signal in signals if signal.criterion_id == criterion.criterion_id],
                candidates,
            )
            for criterion in contract.criteria
        )
        mandatory = [item for item in criterion_results if item.mandatory]
        passed = sum(item.outcome == SignalOutcome.PASSED for item in mandatory)
        coverage = passed / len(mandatory) if mandatory else 1.0
        if any(item.outcome == SignalOutcome.FAILED for item in mandatory):
            outcome = SignalOutcome.FAILED
        elif all(item.outcome == SignalOutcome.PASSED for item in mandatory):
            outcome = SignalOutcome.PASSED
        else:
            outcome = SignalOutcome.INCONCLUSIVE
        if outcome == SignalOutcome.FAILED:
            assurance = max(
                (item.assurance for item in mandatory if item.outcome == SignalOutcome.FAILED),
                default=AssuranceLevel.NONE,
                key=lambda value: list(AssuranceLevel).index(value),
            )
        else:
            assurance = min(
                (item.assurance for item in mandatory if item.outcome == SignalOutcome.PASSED),
                default=AssuranceLevel.NONE,
                key=lambda value: list(AssuranceLevel).index(value),
            )
        missing = tuple(
            message for item in criterion_results for message in item.missing_evidence
        )
        payload = {
            "task_id": contract.task_id,
            "software_id": contract.software_id,
            "intent_lock_id": contract.intent_lock_id,
            "intent_lock_hash": contract.intent_lock_hash,
            "acceptance_contract_id": contract.acceptance_contract_id,
            "acceptance_contract_hash": contract.contract_hash,
            "verification_plan_id": plan.plan_id,
            "outcome": outcome.value,
            "assurance": assurance.value,
            "criteria": [item.model_dump(mode="json", exclude_none=True) for item in criterion_results],
            "mandatory_coverage": coverage,
            "missing_evidence": missing,
        }
        digest = _digest(payload)
        return AcceptanceVerdict(
            verdict_id=stable_id("acceptance-verdict", contract.task_id, digest),
            verdict_hash=digest,
            **payload,
        )

    def repair_packages(
        self,
        contract: AcceptanceContract,
        verdict: AcceptanceVerdict,
        signals: Sequence[VerificationSignal],
    ) -> tuple[TargetedRepairPackage, ...]:
        criteria = {item.criterion_id: item for item in contract.criteria}
        packages = []
        for result in verdict.criteria:
            if result.outcome == SignalOutcome.PASSED:
                continue
            related = [item for item in signals if item.criterion_id == result.criterion_id]
            has_pass = any(item.outcome == SignalOutcome.PASSED for item in related)
            has_fail = any(item.outcome == SignalOutcome.FAILED for item in related)
            failure_type = (
                RepairFailureType.CONFLICTING_EVIDENCE
                if has_pass and has_fail
                else RepairFailureType.VERIFICATION_FAILURE
                if has_fail
                else RepairFailureType.INSUFFICIENT_EVIDENCE
            )
            criterion = criteria[result.criterion_id]
            package_id = stable_id(
                "targeted-repair", verdict.verdict_id, result.criterion_id, failure_type.value
            )
            packages.append(
                TargetedRepairPackage(
                    repair_package_id=package_id,
                    software_id=contract.software_id,
                    task_id=contract.task_id,
                    intent_lock_id=contract.intent_lock_id,
                    intent_lock_hash=contract.intent_lock_hash,
                    acceptance_contract_id=contract.acceptance_contract_id,
                    acceptance_contract_hash=contract.contract_hash,
                    criterion_id=result.criterion_id,
                    failure_type=failure_type,
                    observation_ids=tuple(dict.fromkeys(item.observation_id for item in related)),
                    evidence_ids=tuple(dict.fromkeys(e for item in related for e in item.evidence_ids)),
                    related_contract_ids=tuple(
                        dict.fromkeys(
                            item.operation_contract_id
                            for item in related
                            if item.operation_contract_id
                        )
                    ),
                    repair_targets=criterion.expected_predicates,
                    required_next_state=criterion.expected_predicates,
                    rationale=result.rationale,
                )
            )
        return tuple(packages)

    @staticmethod
    def _criterion_verdict(
        criterion: AcceptanceCriterion,
        signals: Sequence[VerificationSignal],
        candidates: dict[str, object],
    ) -> CriterionVerdict:
        strong_failures = [
            item
            for item in signals
            if item.outcome == SignalOutcome.FAILED
            and (item.strength == EvidenceStrength.STRONG or item.native_check)
        ]
        strong_passes = [
            item
            for item in signals
            if item.outcome == SignalOutcome.PASSED
            and item.strength == EvidenceStrength.STRONG
            and item.candidate_id in candidates
            and candidates[item.candidate_id].status == CandidateStatus.VALIDATED
        ]
        medium_passes = [
            item
            for item in signals
            if item.outcome == SignalOutcome.PASSED
            and item.strength == EvidenceStrength.MEDIUM
        ]
        independent_pair = None
        for index, left in enumerate(medium_passes):
            for right in medium_passes[index + 1 :]:
                if (
                    left.independent_mechanism_id != right.independent_mechanism_id
                    and left.method_type != right.method_type
                    and set(left.evidence_source_ids).isdisjoint(right.evidence_source_ids)
                ):
                    independent_pair = (left, right)
                    break
            if independent_pair:
                break
        if strong_failures:
            outcome, assurance = SignalOutcome.FAILED, AssuranceLevel.HIGH
            rationale = "conflicting strong signal or failed mandatory native check"
            selected = strong_failures
        elif strong_passes:
            outcome, assurance = SignalOutcome.PASSED, AssuranceLevel.HIGH
            rationale = "one validated strong signal passed"
            selected = strong_passes
        elif independent_pair:
            outcome, assurance = SignalOutcome.PASSED, AssuranceLevel.MEDIUM
            rationale = "two independent medium signals passed"
            selected = list(independent_pair)
        else:
            outcome, assurance = SignalOutcome.INCONCLUSIVE, AssuranceLevel.LOW
            rationale = "no validated strong signal or independent medium pair"
            selected = list(signals)
        base_confidence = {
            AssuranceLevel.NONE: 0.0,
            AssuranceLevel.LOW: 0.25,
            AssuranceLevel.MEDIUM: 0.7,
            AssuranceLevel.HIGH: 0.95,
        }[assurance]
        consistency = [item.cross_run_consistency for item in selected if item.cross_run_consistency is not None]
        confidence = base_confidence if not consistency else base_confidence * (0.75 + 0.25 * sum(consistency) / len(consistency))
        missing = () if outcome != SignalOutcome.INCONCLUSIVE else (
            f"criterion {criterion.criterion_id} needs one validated strong signal or two independent medium signals",
        )
        return CriterionVerdict(
            criterion_id=criterion.criterion_id,
            requirement_id=criterion.requirement_id,
            mandatory=criterion.mandatory,
            outcome=outcome,
            assurance=assurance,
            signal_ids=tuple(item.signal_id for item in selected),
            missing_evidence=missing,
            rationale=rationale,
            confidence=confidence,
        )
