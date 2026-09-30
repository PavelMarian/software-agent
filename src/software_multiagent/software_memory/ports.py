from __future__ import annotations

from typing import Protocol, runtime_checkable

from software_multiagent.software_memory.schema.acceptance import AcceptanceContract, ContractAmendment, IntentLock
from software_multiagent.software_memory.schema.verification import VerificationMethodCandidate, VerificationPlan
from software_multiagent.software_memory.schema.outcomes import AcceptanceVerdict, TargetedRepairPackage, VerificationSignal
from software_multiagent.software_memory.schema.models import (
    ActorKind,
    CausalFrontierResult,
    CompactEpisode,
    Entity,
    EvidenceRecord,
    ExecutionObservation,
    KnowledgeConflict,
    KnowledgePacket,
    KnowledgeRequest,
    KnowledgeStatus,
    OperationContract,
    ObservationInterpretation,
    OperationFingerprint,
    RepairKnowledgePacket,
    RepairKnowledgeRequest,
    RelatedEntity,
    SoftwareIdentity,
    SoftwareProfile,
    SkillCandidate,
    Workflow,
    StatePredicate,
)


@runtime_checkable
class MemoryReader(Protocol):
    def get_verification_signal(self, signal_id: str) -> VerificationSignal: ...
    def get_verification_candidate(self, candidate_id: str) -> VerificationMethodCandidate: ...
    def get_verification_plan(self, plan_id: str) -> VerificationPlan: ...
    def get_intent_lock(self, lock_id: str) -> IntentLock: ...
    def get_acceptance_contract(self, contract_id: str) -> AcceptanceContract: ...
    def list_acceptance_contracts(
        self, task_id: str
    ) -> tuple[AcceptanceContract, ...]: ...
    def list_capabilities(self, software_id: str) -> tuple[OperationFingerprint, ...]: ...
    def retrieve(self, request: KnowledgeRequest) -> KnowledgePacket: ...
    def retrieve_delta(
        self, request: KnowledgeRequest, previous: KnowledgePacket
    ) -> KnowledgePacket: ...
    def get_contract(self, contract_id: str, *, include_evidence: bool = False) -> KnowledgePacket: ...
    def get_entity(self, entity_id: str) -> Entity: ...
    def get_related_entities(self, software_id: str, entity_id: str) -> tuple[RelatedEntity, ...]: ...
    def get_workflow(self, workflow_id: str) -> Workflow: ...
    def get_evidence(self, evidence_id: str) -> EvidenceRecord: ...
    def explain_causal_frontier(
        self,
        *,
        software_id: str,
        desired_state: tuple[StatePredicate, ...],
        current_state: tuple[StatePredicate, ...] = (),
        version: str | None = None,
        max_depth: int = 3,
        max_candidates: int = 20,
    ) -> CausalFrontierResult: ...
    def retrieve_repair(self, request: RepairKnowledgeRequest) -> RepairKnowledgePacket: ...


@runtime_checkable
class MemoryWriter(Protocol):
    def record_verification_signal(self, signal: VerificationSignal) -> None: ...
    def evaluate_acceptance(
        self, acceptance_contract_id: str, verification_plan_id: str
    ) -> tuple[AcceptanceVerdict, tuple[TargetedRepairPackage, ...]]: ...
    def record_verification_candidate(self, candidate: VerificationMethodCandidate) -> None: ...
    def record_verification_plan(self, plan: VerificationPlan) -> None: ...
    def register_software(self, software: SoftwareIdentity) -> None: ...
    def record_intent_lock(self, intent: IntentLock) -> None: ...
    def record_acceptance_contract(self, contract: AcceptanceContract) -> None: ...
    def amend_acceptance_contract(
        self,
        previous_contract_id: str,
        new_contract: AcceptanceContract,
        *,
        actor_id: str,
        reason: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> ContractAmendment: ...
    def record_profile(self, profile: SoftwareProfile) -> None: ...
    def record_evidence(self, evidence: EvidenceRecord) -> None: ...
    def record_entity(self, entity: Entity) -> None: ...
    def record_relation(self, relation: Relation) -> None: ...
    def record_contract(self, contract: OperationContract) -> None: ...
    def record_workflow(self, workflow: Workflow) -> None: ...
    def record_observation(self, observation: ExecutionObservation) -> None: ...
    def record_interpretation(self, interpretation: ObservationInterpretation) -> None: ...
    def record_episode(self, episode: CompactEpisode) -> None: ...
    def record_skill_candidate(self, candidate: SkillCandidate) -> None: ...
    def revise_from_source(
        self,
        item,
        *,
        actor_id: str,
        actor_kind: ActorKind = ActorKind.INGESTOR,
        reason: str,
    ): ...


@runtime_checkable
class MemoryValidator(Protocol):
    def review_verification_candidate(
        self,
        candidate_id: str,
        *,
        accepted: bool,
        actor_id: str,
        actor_kind: ActorKind,
        reason: str,
        observation_ids: tuple[str, ...],
    ) -> VerificationMethodCandidate: ...
    def transition_status(
        self,
        item_kind: str,
        item_id: str,
        to_status: KnowledgeStatus,
        *,
        actor_id: str,
        actor_kind: ActorKind,
        reason: str,
        evidence_ids: tuple[str, ...] = (),
        observation_ids: tuple[str, ...] = (),
    ): ...

    def open_conflict(self, conflict: KnowledgeConflict, *, actor_kind: ActorKind) -> None: ...
    def review_skill_candidate(
        self,
        candidate_id: str,
        *,
        accepted: bool,
        actor_id: str,
        actor_kind: ActorKind,
        reason: str,
    ) -> SkillCandidate: ...
