"""Autonomous, domain-neutral construction of evidence-grounded verification plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from hashlib import sha256
from typing import Protocol, Sequence

from pydantic import Field, model_validator

from software_multiagent.software_memory.schema.acceptance import (
    AcceptanceContract,
    AcceptanceCriterion,
    CompilationStatus,
    CriterionCategory,
)
from software_multiagent.software_memory.schema.models import (
    FrozenModel,
    KnowledgeStatus,
    OperationContract,
    StatePredicate,
    VerificationState,
    stable_id,
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: object) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


class VerificationMethodType(str, Enum):
    NATIVE = "native"
    DETERMINISTIC_ASSERTION = "deterministic_assertion"
    METAMORPHIC = "metamorphic"
    DIFFERENTIAL = "differential"
    PROVENANCE = "provenance"
    GROUNDED_LLM_JUDGMENT = "grounded_llm_judgment"


class EvidenceStrength(IntEnum):
    WEAK = 1
    SUPPORTING = 2
    MEDIUM = 3
    STRONG = 4


class VerificationSafety(str, Enum):
    READ_ONLY = "read_only"
    MINIMALLY_INVASIVE = "minimally_invasive"
    MUTATING = "mutating"
    DESTRUCTIVE = "destructive"


class ControlStatus(str, Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNSAFE = "unsafe"


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"


class DiscoverySource(str, Enum):
    MEMORY = "memory"
    DOCUMENTATION = "documentation"
    CLI_INTROSPECTION = "cli_introspection"
    API_INTROSPECTION = "api_introspection"
    SCHEMA_INTROSPECTION = "schema_introspection"
    SAFE_PROBE = "safe_probe"


class ControlEvidence(FrozenModel):
    status: ControlStatus = ControlStatus.NOT_RUN
    positive_fixture_id: str | None = None
    negative_fixture_id: str | None = None
    positive_observation_id: str | None = None
    negative_observation_id: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def validate_controls(self) -> "ControlEvidence":
        if self.status == ControlStatus.PASSED and not (
            self.positive_observation_id and self.negative_observation_id
        ):
            raise ValueError("passed controls require positive and negative observations")
        if self.status in {ControlStatus.UNAVAILABLE, ControlStatus.UNSAFE} and not self.reason:
            raise ValueError("unavailable or unsafe controls require a reason")
        return self


class VerificationMethodCandidate(FrozenModel):
    candidate_id: str
    software_id: str
    criterion_id: str
    method_type: VerificationMethodType
    operation_contract_id: str | None = None
    observable_predicates: tuple[StatePredicate, ...]
    software_version: str | None = None
    expected_signals: tuple[StatePredicate, ...] = ()
    estimated_cost: float = Field(default=0.0, ge=0.0)
    safety: VerificationSafety
    bounded_side_effects: tuple[StatePredicate, ...] = ()
    isolated_target: bool = False
    timeout_seconds: float | None = Field(default=None, gt=0)
    cleanup_operation_id: str | None = None
    rollback_operation_id: str | None = None
    naturally_reversible: bool = False
    discovery_source: DiscoverySource = DiscoverySource.MEMORY
    evidence_ids: tuple[str, ...] = ()
    controls: ControlEvidence = Field(default_factory=ControlEvidence)
    direct_measurement: bool = False
    independent_reference: bool = False
    invariant_justification: str | None = None
    status: CandidateStatus = CandidateStatus.PROPOSED
    effective_strength: EvidenceStrength = EvidenceStrength.WEAK
    validation_observation_ids: tuple[str, ...] = ()
    validation_task_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    revision: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> "VerificationMethodCandidate":
        if not self.observable_predicates:
            raise ValueError("verification candidate needs an observable property")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("candidate evidence IDs must be unique")
        if self.safety == VerificationSafety.MINIMALLY_INVASIVE:
            if not self.bounded_side_effects or not self.isolated_target or not self.timeout_seconds:
                raise ValueError(
                    "minimally invasive method needs bounded effects, isolation, and timeout"
                )
            if not (
                self.cleanup_operation_id
                or self.rollback_operation_id
                or self.naturally_reversible
            ):
                raise ValueError("minimally invasive method needs cleanup, rollback, or reversibility")
        return self


class CandidateAssessment(FrozenModel):
    accepted: bool
    strength: EvidenceStrength
    reasons: tuple[str, ...] = ()
    control_required: bool = True


class VerificationPlan(FrozenModel):
    plan_id: str
    acceptance_contract_id: str
    acceptance_contract_hash: str
    software_id: str
    software_version: str | None = None
    candidates: tuple[VerificationMethodCandidate, ...]
    criterion_methods: dict[str, tuple[str, ...]]
    unresolved_criterion_ids: tuple[str, ...] = ()
    plan_hash: str

    @model_validator(mode="after")
    def validate_plan(self) -> "VerificationPlan":
        ids = {item.candidate_id for item in self.candidates}
        if len(ids) != len(self.candidates):
            raise ValueError("verification candidate IDs must be unique")
        if any(candidate_id not in ids for values in self.criterion_methods.values() for candidate_id in values):
            raise ValueError("verification plan maps an unknown candidate")
        if _hash(self.hash_payload()) != self.plan_hash:
            raise ValueError("verification plan hash is invalid")
        return self

    def hash_payload(self) -> dict[str, object]:
        return {
            "acceptance_contract_id": self.acceptance_contract_id,
            "acceptance_contract_hash": self.acceptance_contract_hash,
            "software_id": self.software_id,
            "software_version": self.software_version,
            "candidates": [item.model_dump(mode="json", exclude_none=True) for item in self.candidates],
            "criterion_methods": self.criterion_methods,
            "unresolved_criterion_ids": self.unresolved_criterion_ids,
        }

    @classmethod
    def create(
        cls,
        *,
        contract: AcceptanceContract,
        candidates: Sequence[VerificationMethodCandidate],
        criterion_methods: dict[str, tuple[str, ...]],
        unresolved_criterion_ids: Sequence[str] = (),
    ) -> "VerificationPlan":
        prototype = {
            "acceptance_contract_id": contract.acceptance_contract_id,
            "acceptance_contract_hash": contract.contract_hash,
            "software_id": contract.software_id,
            "software_version": contract.software_version,
            "candidates": [item.model_dump(mode="json", exclude_none=True) for item in candidates],
            "criterion_methods": criterion_methods,
            "unresolved_criterion_ids": tuple(unresolved_criterion_ids),
        }
        digest = _hash(prototype)
        return cls(
            plan_id=stable_id("verification-plan", contract.acceptance_contract_id, digest),
            plan_hash=digest,
            **prototype,
        )


class VerificationDiscoveryProvider(Protocol):
    """Plugin boundary for documentation, introspection, and safe probes."""

    def discover(
        self,
        criterion: AcceptanceCriterion,
        *,
        software_id: str,
        software_version: str | None,
    ) -> Sequence[VerificationMethodCandidate]: ...


class VerificationControlRunner(Protocol):
    """Sandbox-owned executor for safe positive/negative fixtures."""

    def run_controls(
        self,
        candidate: VerificationMethodCandidate,
        criterion: AcceptanceCriterion,
    ) -> ControlEvidence: ...


class VerificationPolicy:
    """Validate relevance and assurance without software-specific rules."""

    _BASE = {
        VerificationMethodType.NATIVE: EvidenceStrength.MEDIUM,
        VerificationMethodType.DETERMINISTIC_ASSERTION: EvidenceStrength.MEDIUM,
        VerificationMethodType.METAMORPHIC: EvidenceStrength.MEDIUM,
        VerificationMethodType.DIFFERENTIAL: EvidenceStrength.MEDIUM,
        VerificationMethodType.PROVENANCE: EvidenceStrength.SUPPORTING,
        VerificationMethodType.GROUNDED_LLM_JUDGMENT: EvidenceStrength.WEAK,
    }

    def assess(
        self,
        candidate: VerificationMethodCandidate,
        criterion: AcceptanceCriterion,
    ) -> CandidateAssessment:
        reasons: list[str] = []
        strength = self._BASE[candidate.method_type]
        if candidate.criterion_id != criterion.criterion_id:
            reasons.append("candidate targets a different acceptance criterion")
        if not self._relevant(candidate, criterion):
            reasons.append("observable property does not test the acceptance criterion")
        if candidate.safety in {VerificationSafety.MUTATING, VerificationSafety.DESTRUCTIVE}:
            reasons.append("autonomous verification forbids mutating or destructive methods")
        if candidate.controls.status == ControlStatus.FAILED:
            reasons.append("method failed to distinguish positive and negative controls")

        controlled = candidate.controls.status == ControlStatus.PASSED
        previously_validated = (
            candidate.status == CandidateStatus.VALIDATED
            and len(set(candidate.validation_task_ids)) >= 2
            and len(set(candidate.validation_observation_ids)) >= 2
        )
        promotable = controlled or previously_validated
        if promotable and candidate.direct_measurement:
            if candidate.method_type in {
                VerificationMethodType.NATIVE,
                VerificationMethodType.DETERMINISTIC_ASSERTION,
            }:
                strength = EvidenceStrength.STRONG
            elif candidate.method_type == VerificationMethodType.METAMORPHIC and candidate.invariant_justification:
                strength = EvidenceStrength.STRONG
            elif candidate.method_type == VerificationMethodType.DIFFERENTIAL and candidate.independent_reference:
                strength = EvidenceStrength.STRONG
            elif (
                candidate.method_type == VerificationMethodType.PROVENANCE
                and criterion.category == CriterionCategory.PROVENANCE
            ):
                strength = EvidenceStrength.STRONG

        if candidate.controls.status in {
            ControlStatus.NOT_RUN,
            ControlStatus.PARTIAL,
            ControlStatus.UNAVAILABLE,
            ControlStatus.UNSAFE,
        } and not previously_validated:
            strength = EvidenceStrength(max(EvidenceStrength.WEAK, strength - 1))
        if candidate.method_type == VerificationMethodType.GROUNDED_LLM_JUDGMENT:
            strength = EvidenceStrength.WEAK
        return CandidateAssessment(
            accepted=not reasons,
            strength=strength,
            reasons=tuple(reasons),
            control_required=not previously_validated,
        )

    @staticmethod
    def _relevant(
        candidate: VerificationMethodCandidate,
        criterion: AcceptanceCriterion,
    ) -> bool:
        expected = {item.key for item in criterion.expected_predicates}
        observed = {item.key for item in candidate.observable_predicates}
        return bool(expected & observed) or (
            candidate.operation_contract_id in criterion.verification_operation_ids
        )


@dataclass
class VerificationPlanBuilder:
    store: object
    policy: VerificationPolicy = VerificationPolicy()
    discovery_providers: tuple[VerificationDiscoveryProvider, ...] = ()
    control_runner: VerificationControlRunner | None = None

    def build(self, contract: AcceptanceContract) -> VerificationPlan:
        candidates: list[VerificationMethodCandidate] = []
        mapping: dict[str, tuple[str, ...]] = {}
        unresolved: list[str] = []
        for criterion in contract.criteria:
            if criterion.compilation_status != CompilationStatus.READY:
                unresolved.append(criterion.criterion_id)
                mapping[criterion.criterion_id] = ()
                continue
            discovered = [
                item
                for item in self.store.list_verification_candidates(contract.software_id)
                if item.criterion_id == criterion.criterion_id
                and item.software_version == contract.software_version
            ]
            discovered.extend(self._memory_candidates(contract, criterion))
            for provider in self.discovery_providers:
                discovered.extend(
                    provider.discover(
                        criterion,
                        software_id=contract.software_id,
                        software_version=contract.software_version,
                    )
                )
            accepted: list[str] = []
            seen: set[str] = set()
            for candidate in discovered:
                if candidate.candidate_id in seen:
                    continue
                seen.add(candidate.candidate_id)
                if (
                    candidate.controls.status == ControlStatus.NOT_RUN
                    and self.control_runner is not None
                    and candidate.safety
                    in {VerificationSafety.READ_ONLY, VerificationSafety.MINIMALLY_INVASIVE}
                ):
                    candidate = candidate.model_copy(
                        update={"controls": self.control_runner.run_controls(candidate, criterion)}
                    )
                assessment = self.policy.assess(candidate, criterion)
                evaluated = candidate.model_copy(
                    update={
                        "effective_strength": assessment.strength,
                        "status": candidate.status if assessment.accepted else CandidateStatus.REJECTED,
                        "rejection_reasons": assessment.reasons,
                    }
                )
                candidates.append(evaluated)
                if assessment.accepted:
                    accepted.append(evaluated.candidate_id)
            mapping[criterion.criterion_id] = tuple(accepted)
            adequate = [
                item for item in candidates
                if item.candidate_id in accepted
                and item.effective_strength >= EvidenceStrength.MEDIUM
                and item.method_type != VerificationMethodType.GROUNDED_LLM_JUDGMENT
            ]
            if criterion.mandatory and not adequate:
                unresolved.append(criterion.criterion_id)
        return VerificationPlan.create(
            contract=contract,
            candidates=candidates,
            criterion_methods=mapping,
            unresolved_criterion_ids=tuple(dict.fromkeys(unresolved)),
        )

    def _memory_candidates(
        self,
        contract: AcceptanceContract,
        criterion: AcceptanceCriterion,
    ) -> list[VerificationMethodCandidate]:
        operation_ids = list(criterion.verification_operation_ids)
        for operation in self.store.list_contracts(contract.software_id):
            if self._contract_relevant(operation, criterion):
                operation_ids.append(operation.contract_id)
        values: list[VerificationMethodCandidate] = []
        for operation_id in dict.fromkeys(operation_ids):
            operation = self.store.get_contract(operation_id)
            if operation is None or not operation.version_scope.matches(contract.software_version):
                continue
            observed = tuple(operation.success_signals or operation.effects)
            if not observed:
                continue
            method_type = self._method_type(operation)
            safety = self._safety(operation)
            metadata = operation.metadata
            controls = ControlEvidence.model_validate(metadata.get("verification_controls", {}))
            values.append(
                VerificationMethodCandidate(
                    candidate_id=stable_id("verification-method", criterion.criterion_id, operation.contract_id),
                    software_id=contract.software_id,
                    criterion_id=criterion.criterion_id,
                    method_type=method_type,
                    operation_contract_id=operation.contract_id,
                    observable_predicates=observed,
                    software_version=contract.software_version,
                    expected_signals=operation.success_signals,
                    estimated_cost=float(metadata.get("estimated_cost", 0.0)),
                    safety=safety,
                    bounded_side_effects=operation.side_effects,
                    isolated_target=bool(metadata.get("isolated_target", False)),
                    timeout_seconds=metadata.get("timeout_seconds"),
                    cleanup_operation_id=metadata.get("cleanup_operation_id"),
                    rollback_operation_id=metadata.get("rollback_operation_id"),
                    naturally_reversible=bool(metadata.get("naturally_reversible", False)),
                    evidence_ids=operation.evidence_ids,
                    controls=controls,
                    direct_measurement=bool(metadata.get("direct_measurement", True)),
                    independent_reference=bool(metadata.get("independent_reference", False)),
                    invariant_justification=metadata.get("invariant_justification"),
                )
            )
        return values

    @staticmethod
    def _contract_relevant(
        operation: OperationContract,
        criterion: AcceptanceCriterion,
    ) -> bool:
        expected = {item.key for item in criterion.expected_predicates}
        observed = {item.key for item in (*operation.success_signals, *operation.effects)}
        return bool(expected & observed)

    @staticmethod
    def _method_type(operation: OperationContract) -> VerificationMethodType:
        declared = operation.metadata.get("verification_method_type")
        if declared:
            return VerificationMethodType(declared)
        if operation.interface in {"cli", "api", "sql", "library"}:
            return VerificationMethodType.NATIVE
        return VerificationMethodType.DETERMINISTIC_ASSERTION

    @staticmethod
    def _safety(operation: OperationContract) -> VerificationSafety:
        if operation.risk_level == "low" and not operation.side_effects:
            return VerificationSafety.READ_ONLY
        return {
            "read_only": VerificationSafety.READ_ONLY,
            "low": VerificationSafety.MINIMALLY_INVASIVE,
            "medium": VerificationSafety.MUTATING,
            "high": VerificationSafety.MUTATING,
            "destructive": VerificationSafety.DESTRUCTIVE,
        }[operation.risk_level]
