from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from software_multiagent.software_memory.errors import IntegrityViolationError, TrustPolicyError
from software_multiagent.software_memory.schema.models import (
    ActorKind,
    CompactEpisode,
    ConflictState,
    Entity,
    EvidenceRecord,
    ExecutionObservation,
    KnowledgeConflict,
    KnowledgeStatus,
    OperationContract,
    ObservationInterpretation,
    Relation,
    SoftwareIdentity,
    SoftwareProfile,
    StatusTransition,
    SkillCandidate,
    VerificationState,
    Workflow,
    stable_id,
)
from software_multiagent.software_memory.operations.procedural import predicates_conflict
from software_multiagent.software_memory.persistence.storage import SQLiteMemoryStore
from software_multiagent.software_memory.schema.acceptance import (
    AcceptanceContract,
    CompilationStatus,
    ContractAmendment,
    IntentLock,
)
from software_multiagent.software_memory.schema.verification import (
    CandidateStatus,
    VerificationMethodCandidate,
    VerificationPlan,
    VerificationPolicy,
)
from software_multiagent.software_memory.schema.outcomes import (
    AcceptanceVerdict,
    TargetedRepairPackage,
    VerificationSignal,
    VerdictAggregator,
)


ItemKind = Literal["entity", "contract", "workflow", "relation"]
KnowledgeObject = Entity | OperationContract | Relation | Workflow


class PromotionError(TrustPolicyError):
    """Backward-compatible trust-policy error raised by promotion operations."""


_SUPPORTED = {
    KnowledgeStatus.EVIDENCE_SUPPORTED,
    KnowledgeStatus.CROSS_SOURCE_CONFIRMED,
    KnowledgeStatus.SOURCE_CODE_CONFIRMED,
    KnowledgeStatus.EXECUTION_VERIFIED,
    KnowledgeStatus.TASK_VERIFIED,
}

_REVIEW_ONLY = _SUPPORTED | {
    KnowledgeStatus.CONFLICTING,
    KnowledgeStatus.DEPRECATED,
    KnowledgeStatus.RETRACTED,
}


@dataclass
class MemoryService:
    """Mutation boundary enforcing referential integrity and trust policy."""

    store: SQLiteMemoryStore

    @contextmanager
    def batch(self):
        """Commit a set of writes atomically or roll the complete set back."""
        with self.store.batch():
            yield self

    def register_software(self, software: SoftwareIdentity) -> None:
        self.store.put_software(software)

    def record_intent_lock(self, intent: IntentLock) -> None:
        """Persist the immutable pre-execution interpretation of a task."""

        evidence_ids = tuple(
            evidence_id
            for requirement in intent.requirements
            for evidence_id in requirement.evidence_ids
        )
        if not self.store.evidence_exist(evidence_ids):
            raise IntegrityViolationError(
                "intent lock references unknown evidence",
                task_id=intent.task_id,
            )
        if intent.supersedes:
            previous = self.store.get_intent_lock(intent.supersedes)
            if previous is None or previous.task_id != intent.task_id:
                raise IntegrityViolationError("intent supersedes an unknown task lock")
            if intent.revision != previous.revision + 1:
                raise IntegrityViolationError("intent revision must advance by one")
        self.store.put_intent_lock(intent)

    def record_acceptance_contract(self, contract: AcceptanceContract) -> None:
        """Store a locked contract after referential and coverage validation."""

        self._require_software(contract.software_id)
        intent = self.store.get_intent_lock(contract.intent_lock_id)
        if intent is None or intent.task_id != contract.task_id:
            raise IntegrityViolationError("acceptance contract references unknown intent")
        if intent.lock_hash != contract.intent_lock_hash:
            raise IntegrityViolationError("acceptance contract references stale intent hash")
        requirements = {item.requirement_id: item for item in intent.requirements}
        if {item.requirement_id for item in contract.criteria} != set(requirements):
            raise IntegrityViolationError("acceptance contract does not cover locked intent")
        evidence_ids = tuple(
            dict.fromkeys(
                (
                    *(evidence_id for requirement in intent.requirements
                      for evidence_id in requirement.evidence_ids),
                    *(evidence_id for criterion in contract.criteria
                      for evidence_id in criterion.knowledge_evidence_ids),
                )
            )
        )
        self._validate_evidence_scope(contract.software_id, evidence_ids)
        for evidence_id in evidence_ids:
            evidence = self.store.get_evidence(evidence_id)
            if (
                evidence is not None
                and contract.software_version
                and evidence.source_version
                and evidence.source_version != contract.software_version
            ):
                raise IntegrityViolationError(
                    "acceptance evidence version does not match the contract",
                    evidence_id=evidence_id,
                )
        predicates = [
            predicate
            for criterion in contract.criteria
            for predicate in criterion.expected_predicates
        ]
        for index, left in enumerate(predicates):
            for right in predicates[index + 1:]:
                if predicates_conflict(left, right):
                    raise IntegrityViolationError(
                        "acceptance criteria contain conflicting predicates"
                    )
        for criterion in contract.criteria:
            requirement = requirements[criterion.requirement_id]
            if criterion.category != requirement.category:
                raise IntegrityViolationError("criterion changes requirement category")
            if requirement.mandatory and not criterion.mandatory:
                raise IntegrityViolationError("criterion weakens mandatory requirement")
            for operation_id in criterion.verification_operation_ids:
                operation = self._require_contract(operation_id, contract.software_id)
                if not operation.version_scope.matches(contract.software_version):
                    raise IntegrityViolationError(
                        "verification operation version does not match acceptance contract",
                        operation_id=operation_id,
                    )
        self.store.put_acceptance_contract(contract)

    def record_verification_candidate(
        self, candidate: VerificationMethodCandidate
    ) -> None:
        """Store a discovered method without allowing unreviewed trust promotion."""

        self._require_software(candidate.software_id)
        existing = self.store.get_verification_candidate(candidate.candidate_id)
        if existing is None and candidate.status == CandidateStatus.VALIDATED:
            raise PromotionError("a new verification method cannot start as validated")
        if candidate.operation_contract_id:
            operation = self._require_contract(
                candidate.operation_contract_id, candidate.software_id
            )
            if not operation.version_scope.matches(candidate.software_version):
                raise IntegrityViolationError(
                    "verification operation does not match candidate version"
                )
        for operation_id in (
            candidate.cleanup_operation_id,
            candidate.rollback_operation_id,
        ):
            if operation_id:
                self._require_contract(operation_id, candidate.software_id)
        self._validate_evidence_scope(candidate.software_id, candidate.evidence_ids)
        self.store.put_verification_candidate(candidate)

    def review_verification_candidate(
        self,
        candidate_id: str,
        *,
        accepted: bool,
        actor_id: str,
        actor_kind: ActorKind,
        reason: str,
        observation_ids: tuple[str, ...],
    ) -> VerificationMethodCandidate:
        """Promote only after independent accepted checks on several tasks."""

        if actor_kind not in {ActorKind.VALIDATOR, ActorKind.HUMAN}:
            raise PromotionError("verification method review requires validator authority")
        if not reason.strip():
            raise PromotionError("verification method review requires a reason")
        candidate = self.store.get_verification_candidate(candidate_id)
        if candidate is None:
            raise IntegrityViolationError("unknown verification method candidate")
        observations = []
        for observation_id in observation_ids:
            observation = self.store.get_observation(observation_id)
            if observation is None or observation.software_id != candidate.software_id:
                raise IntegrityViolationError("verification review references unknown observation")
            observations.append(observation)
        task_ids = {item.task_id for item in observations}
        independently_accepted = all(
            item.verification == VerificationState.ACCEPTED
            and item.verifier_id
            and item.actor_id != item.verifier_id
            for item in observations
        )
        if accepted and (
            len(observations) < 3 or len(task_ids) < 2 or not independently_accepted
        ):
            raise PromotionError(
                "promotion requires three independently accepted checks across two tasks"
            )
        revised = candidate.model_copy(
            update={
                "status": CandidateStatus.VALIDATED if accepted else CandidateStatus.REJECTED,
                "validation_observation_ids": tuple(dict.fromkeys(observation_ids)),
                "validation_task_ids": tuple(sorted(task_ids)),
                "rejection_reasons": () if accepted else (reason,),
                "revision": candidate.revision + 1,
            }
        )
        self.store.put_verification_candidate(revised)
        return revised

    def record_verification_plan(self, plan: VerificationPlan) -> None:
        contract = self.store.get_acceptance_contract(plan.acceptance_contract_id)
        if contract is None or contract.software_id != plan.software_id:
            raise IntegrityViolationError("verification plan references unknown acceptance contract")
        if contract.contract_hash != plan.acceptance_contract_hash:
            raise IntegrityViolationError("verification plan uses a stale acceptance contract")
        criteria = {item.criterion_id: item for item in contract.criteria}
        policy = VerificationPolicy()
        for candidate in plan.candidates:
            criterion = criteria.get(candidate.criterion_id)
            if criterion is None:
                raise IntegrityViolationError("verification candidate references unknown criterion")
            assessment = policy.assess(candidate, criterion)
            selected = candidate.candidate_id in plan.criterion_methods.get(
                candidate.criterion_id, ()
            )
            if selected and not assessment.accepted:
                raise IntegrityViolationError("verification plan selects a rejected method")
        with self.batch():
            for candidate in plan.candidates:
                if self.store.get_verification_candidate(candidate.candidate_id) is None:
                    self.record_verification_candidate(candidate)
            self.store.put_verification_plan(plan)

    def record_verification_signal(self, signal: VerificationSignal) -> None:
        """Attach a verifier-owned interpretation to an immutable raw observation."""

        contract = self.store.get_acceptance_contract(signal.acceptance_contract_id)
        plan = self.store.get_verification_plan(signal.verification_plan_id)
        observation = self.store.get_observation(signal.observation_id)
        if contract is None or plan is None or observation is None:
            raise IntegrityViolationError("verification signal has an unknown graph reference")
        if (
            signal.software_id != contract.software_id
            or signal.task_id != contract.task_id
            or signal.acceptance_contract_hash != contract.contract_hash
            or plan.acceptance_contract_id != contract.acceptance_contract_id
            or observation.software_id != signal.software_id
            or observation.task_id != signal.task_id
        ):
            raise IntegrityViolationError("verification signal crosses task or software scope")
        criteria = {item.criterion_id: item for item in contract.criteria}
        criterion = criteria.get(signal.criterion_id)
        requirement_ids = {
            item.requirement_id for item in contract.criteria
        }
        if criterion is None or signal.task_requirement_id not in requirement_ids:
            raise IntegrityViolationError("verification signal has an unknown criterion")
        if criterion.requirement_id != signal.task_requirement_id:
            raise IntegrityViolationError("verification signal breaks requirement traceability")
        candidate = next(
            (item for item in plan.candidates if item.candidate_id == signal.candidate_id),
            None,
        )
        if candidate is None or candidate.criterion_id != signal.criterion_id:
            raise IntegrityViolationError("verification signal has an unknown method")
        if signal.method_type != candidate.method_type or signal.strength > candidate.effective_strength:
            raise IntegrityViolationError("verification signal overstates method assurance")
        if signal.operation_contract_id != candidate.operation_contract_id:
            raise IntegrityViolationError("verification signal changes its operation contract")
        if observation.contract_id != signal.operation_contract_id:
            raise IntegrityViolationError("observation was produced by a different operation")
        self._validate_evidence_scope(signal.software_id, signal.evidence_ids)
        self.store.put_verification_signal(signal)

    def evaluate_acceptance(
        self, acceptance_contract_id: str, verification_plan_id: str
    ) -> tuple[AcceptanceVerdict, tuple[TargetedRepairPackage, ...]]:
        """Recompute and persist a deterministic verdict and targeted repair frontier."""

        contract = self.store.get_acceptance_contract(acceptance_contract_id)
        plan = self.store.get_verification_plan(verification_plan_id)
        if contract is None or plan is None:
            raise IntegrityViolationError("cannot evaluate an unknown contract or plan")
        signals = self.store.list_verification_signals(acceptance_contract_id)
        aggregator = VerdictAggregator()
        verdict = aggregator.aggregate(contract, plan, signals)
        packages = aggregator.repair_packages(contract, verdict, signals)
        with self.batch():
            self.store.put_acceptance_verdict(verdict)
            for package in packages:
                package.assert_preserves(contract)
                self.store.put_targeted_repair_package(package)
        return verdict, packages

    def amend_acceptance_contract(
        self,
        previous_contract_id: str,
        new_contract: AcceptanceContract,
        *,
        actor_id: str,
        reason: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> ContractAmendment:
        """Append a stricter/equivalent contract revision; never rewrite the old one."""

        previous = self.store.get_acceptance_contract(previous_contract_id)
        if previous is None:
            raise IntegrityViolationError("cannot amend unknown acceptance contract")
        if new_contract.supersedes != previous_contract_id:
            raise IntegrityViolationError("new contract must identify the revision it supersedes")
        if new_contract.revision != previous.revision + 1:
            raise IntegrityViolationError("acceptance contract revision must advance by one")
        if (
            new_contract.task_id != previous.task_id
            or new_contract.software_id != previous.software_id
            or new_contract.intent_lock_id != previous.intent_lock_id
        ):
            raise IntegrityViolationError(
                "repair amendment cannot change task, software, or intent"
            )
        old_by_requirement = {item.requirement_id: item for item in previous.criteria}
        new_by_requirement = {item.requirement_id: item for item in new_contract.criteria}
        for requirement_id, old in old_by_requirement.items():
            replacement = new_by_requirement.get(requirement_id)
            if replacement is None:
                raise IntegrityViolationError("amendment removes a locked criterion")
            if old.mandatory and not replacement.mandatory:
                raise IntegrityViolationError("amendment weakens a mandatory criterion")
            if (
                old.compilation_status == CompilationStatus.READY
                and replacement.compilation_status != CompilationStatus.READY
            ):
                raise IntegrityViolationError("amendment makes a ready criterion unverifiable")
        self._validate_evidence_scope(previous.software_id, evidence_ids)
        amendment = ContractAmendment(
            amendment_id=stable_id(
                "amendment", previous_contract_id, new_contract.acceptance_contract_id
            ),
            task_id=previous.task_id,
            previous_contract_id=previous_contract_id,
            new_contract_id=new_contract.acceptance_contract_id,
            actor_id=actor_id,
            reason=reason,
            evidence_ids=evidence_ids,
        )
        with self.batch():
            self.record_acceptance_contract(new_contract)
            self.store.put_contract_amendment(amendment)
        return amendment

    def record_profile(self, profile: SoftwareProfile) -> None:
        self._require_software(profile.software_id)
        self.store.put_profile(profile)

    def record_evidence(self, evidence: EvidenceRecord) -> None:
        self._require_software(evidence.software_id)
        for allowed_id in evidence.allowed_software_ids:
            self._require_software(allowed_id)
        self.store.put_evidence(evidence)

    def record_entity(self, entity: Entity) -> None:
        self._require_software(entity.software_id)
        self._require_record_status("entity", entity.entity_id, entity.status)
        self._validate_evidence_scope(entity.software_id, entity.evidence_ids)
        self.store.put_entity(entity)

    def record_relation(self, relation: Relation) -> None:
        self._require_software(relation.software_id)
        self._require_record_status("relation", relation.relation_id, relation.status)
        self._validate_evidence_scope(relation.software_id, relation.evidence_ids)
        source = self._require_entity(relation.source_entity_id, relation.software_id)
        target = self._require_entity(relation.target_entity_id, relation.software_id)
        del source, target
        self.store.put_relation(relation)

    def record_contract(self, contract: OperationContract) -> None:
        self._require_software(contract.software_id)
        self._require_record_status("contract", contract.contract_id, contract.status)
        self._validate_evidence_scope(contract.software_id, contract.evidence_ids)
        for entity_id in contract.related_entity_ids:
            self._require_entity(entity_id, contract.software_id)
        referenced_operations = {
            *contract.verification_operations,
            *(operation for repair in contract.repair_strategies for operation in repair.required_operations),
        }
        for operation_id in referenced_operations:
            self._require_contract(operation_id, contract.software_id)
        self.store.put_contract(contract)

    def record_workflow(self, workflow: Workflow) -> None:
        self._require_software(workflow.software_id)
        self._require_record_status("workflow", workflow.workflow_id, workflow.status)
        self._validate_evidence_scope(workflow.software_id, workflow.evidence_ids)
        for step in workflow.steps:
            self._require_contract(step.operation_id, workflow.software_id)
        self.store.put_workflow(workflow)

    def record_observation(self, observation: ExecutionObservation) -> None:
        """Store an observation without treating it as verified knowledge."""
        self._require_software(observation.software_id)
        self._validate_evidence_scope(observation.software_id, observation.evidence_ids)
        if observation.contract_id:
            self._require_contract(observation.contract_id, observation.software_id)
        if observation.actor_kind == ActorKind.AGENT and (
            observation.verification != VerificationState.UNREVIEWED
            or observation.verifier_id is not None
        ):
            raise PromotionError("an agent cannot verify its own observation")
        with self.batch():
            self.store.put_observation(observation)
            self._open_observation_conflict(observation)

    def record_interpretation(self, interpretation: ObservationInterpretation) -> None:
        """Store an interpretation separately from immutable raw execution output."""

        self._require_software(interpretation.software_id)
        observation = self.store.get_observation(interpretation.observation_id)
        if observation is None or observation.software_id != interpretation.software_id:
            raise IntegrityViolationError(
                "interpretation references unknown observation",
                observation_id=interpretation.observation_id,
            )
        if interpretation.actor_kind == ActorKind.AGENT and (
            interpretation.verification != VerificationState.UNREVIEWED
            or interpretation.verifier_id is not None
        ):
            raise PromotionError("an agent cannot verify its own interpretation")
        self.store.put_interpretation(interpretation)

    def record_episode(self, episode: CompactEpisode) -> None:
        """Persist a compact outcome record, never a conversational trajectory."""

        self._require_software(episode.software_id)
        observation = self.store.get_observation(episode.observation_id)
        if observation is None or observation.software_id != episode.software_id:
            raise IntegrityViolationError(
                "episode references unknown observation",
                observation_id=episode.observation_id,
            )
        if episode.task_id != observation.task_id:
            raise IntegrityViolationError("episode and observation task IDs disagree")
        if episode.action_contract_id != observation.contract_id:
            raise IntegrityViolationError("episode and observation contracts disagree")
        if (
            episode.succeeded != observation.succeeded
            or episode.verification != observation.verification
            or episode.verifier_id != observation.verifier_id
        ):
            raise IntegrityViolationError(
                "episode cannot change the raw observation verdict"
            )
        self.store.put_episode(episode)

    def record_skill_candidate(self, candidate: SkillCandidate) -> None:
        """Record an untrusted reusable procedure candidate."""

        self._require_software(candidate.software_id)
        if candidate.status != "proposed":
            raise PromotionError("new skill candidates must begin as proposed")
        for operation_id in candidate.operation_ids:
            self._require_contract(operation_id, candidate.software_id)
        self._skill_episodes(candidate)
        self.store.put_skill_candidate(candidate)

    def review_skill_candidate(
        self,
        candidate_id: str,
        *,
        accepted: bool,
        actor_id: str,
        actor_kind: ActorKind,
        reason: str,
    ) -> SkillCandidate:
        if accepted and actor_kind != ActorKind.VALIDATOR:
            raise PromotionError("skill acceptance requires validator authority")
        if not accepted and actor_kind not in {ActorKind.VALIDATOR, ActorKind.HUMAN}:
            raise PromotionError("skill rejection requires validator or human authority")
        if not reason.strip():
            raise PromotionError("skill review requires a reason")
        candidate = self.store.get_skill_candidate(candidate_id)
        if candidate is None:
            raise IntegrityViolationError(
                "unknown skill candidate", candidate_id=candidate_id
            )
        episodes = self._skill_episodes(candidate)
        if accepted:
            successful = tuple(
                episode
                for episode in episodes
                if episode.succeeded is True
                and episode.verification == VerificationState.ACCEPTED
                and episode.verifier_id
            )
            if len(successful) < 3:
                raise PromotionError(
                    "skill acceptance requires three independently verified successes"
                )
            if len({episode.task_id for episode in successful}) < 2:
                raise PromotionError("skill acceptance requires transfer across tasks")
            observed_versions = {
                episode.software_version
                for episode in successful
                if episode.software_version
            }
            if candidate.compatible_versions and not set(
                candidate.compatible_versions
            ).issubset(observed_versions):
                raise PromotionError(
                    "skill acceptance requires evidence for every compatible version"
                )
        updated = candidate.model_copy(
            update={
                "status": "accepted" if accepted else "rejected",
                "reviewed_by": actor_id,
                "review_reason": reason,
                "revision": candidate.revision + 1,
            }
        )
        self.store.put_skill_candidate(updated)
        return updated

    def revise_from_source(
        self,
        item: KnowledgeObject,
        *,
        actor_id: str,
        actor_kind: ActorKind = ActorKind.INGESTOR,
        reason: str,
    ) -> KnowledgeObject:
        """Replace changed extracted semantics and reset trust through an audit event."""
        if actor_kind not in {ActorKind.INGESTOR, ActorKind.SYSTEM}:
            raise PromotionError("source revision requires an ingestor or system actor")
        if item.status not in {KnowledgeStatus.EXTRACTED, KnowledgeStatus.VERSION_UNCONFIRMED}:
            raise PromotionError("source revision must reset trust to an initial status")
        if not reason.strip():
            raise PromotionError("source revision requires a reason")
        item_kind, item_id = self._item_identity(item)
        current = self._get_item(item_kind, item_id)
        if current.software_id != item.software_id:
            raise IntegrityViolationError("source revision changes software identity")
        self._validate_item_links(item)
        revised = item.model_copy(update={"revision": current.revision + 1})
        transition = StatusTransition(
            transition_id=stable_id(
                "transition",
                item_kind,
                item_id,
                current.status.value,
                revised.status.value,
                actor_id,
                str(len(self.store.status_history(item_id, item_kind=item_kind))),
            ),
            software_id=item.software_id,
            item_kind=item_kind,
            item_id=item_id,
            from_status=current.status,
            to_status=revised.status,
            actor_id=actor_id,
            actor_kind=actor_kind,
            reason=reason,
            evidence_ids=item.evidence_ids,
        )
        self.store.apply_status_transition(revised, transition)
        return revised

    def transition_status(
        self,
        item_kind: ItemKind,
        item_id: str,
        to_status: KnowledgeStatus,
        *,
        actor_id: str,
        actor_kind: ActorKind,
        reason: str,
        evidence_ids: tuple[str, ...] = (),
        observation_ids: tuple[str, ...] = (),
    ) -> KnowledgeObject:
        item = self._get_item(item_kind, item_id)
        if not reason.strip():
            raise PromotionError("status transition requires a reason")
        if actor_kind == ActorKind.AGENT:
            raise PromotionError("an acting agent cannot change knowledge trust status")
        if to_status in _REVIEW_ONLY and actor_kind not in {ActorKind.VALIDATOR, ActorKind.HUMAN}:
            raise PromotionError(f"{to_status.value} requires validator or human review")

        combined_evidence = tuple(dict.fromkeys((*item.evidence_ids, *evidence_ids)))
        if to_status in _SUPPORTED and not combined_evidence:
            raise PromotionError(f"{to_status.value} requires evidence")
        self._validate_evidence_scope(item.software_id, combined_evidence)
        self._validate_source_requirements(to_status, combined_evidence)
        if to_status in {KnowledgeStatus.EXECUTION_VERIFIED, KnowledgeStatus.TASK_VERIFIED}:
            self._require_verified_observations(item, observation_ids)

        transition = StatusTransition(
            transition_id=stable_id(
                "transition",
                item_kind,
                item_id,
                item.status.value,
                to_status.value,
                actor_id,
                str(len(self.store.status_history(item_id, item_kind=item_kind))),
            ),
            software_id=item.software_id,
            item_kind=item_kind,
            item_id=item_id,
            from_status=item.status,
            to_status=to_status,
            actor_id=actor_id,
            actor_kind=actor_kind,
            reason=reason,
            evidence_ids=combined_evidence,
            observation_ids=observation_ids,
        )
        updated = item.model_copy(
            update={
                "status": to_status,
                "evidence_ids": combined_evidence,
                "revision": item.revision + 1,
            }
        )
        self.store.apply_status_transition(updated, transition)
        return updated

    def transition_contract(
        self,
        contract_id: str,
        to_status: KnowledgeStatus,
        **arguments,
    ) -> OperationContract:
        updated = self.transition_status("contract", contract_id, to_status, **arguments)
        assert isinstance(updated, OperationContract)
        return updated

    def demote_contract(
        self,
        contract_id: str,
        *,
        actor_id: str,
        actor_kind: ActorKind,
        reason: str,
        conflicting: bool = True,
        evidence_ids: tuple[str, ...] = (),
        observation_ids: tuple[str, ...] = (),
    ) -> OperationContract:
        return self.transition_contract(
            contract_id,
            KnowledgeStatus.CONFLICTING if conflicting else KnowledgeStatus.RETRACTED,
            actor_id=actor_id,
            actor_kind=actor_kind,
            reason=reason,
            evidence_ids=evidence_ids,
            observation_ids=observation_ids,
        )

    def open_conflict(self, conflict: KnowledgeConflict, *, actor_kind: ActorKind) -> None:
        if actor_kind not in {
            ActorKind.INGESTOR,
            ActorKind.SYSTEM,
            ActorKind.VALIDATOR,
            ActorKind.HUMAN,
        }:
            raise PromotionError("actor cannot open a knowledge conflict")
        self._require_software(conflict.software_id)
        if self.store.get_conflict(conflict.conflict_id) is not None:
            raise IntegrityViolationError(
                "knowledge conflict already exists",
                conflict_id=conflict.conflict_id,
            )
        self._get_item(conflict.item_kind, conflict.item_id)
        if conflict.conflicting_item_id:
            self._get_item(conflict.item_kind, conflict.conflicting_item_id)
        self._validate_evidence_scope(conflict.software_id, conflict.evidence_ids)
        for observation_id in conflict.observation_ids:
            observation = self.store.get_observation(observation_id)
            if observation is None or observation.software_id != conflict.software_id:
                raise IntegrityViolationError(
                    "conflict references unknown observation",
                    observation_id=observation_id,
                )
        self.store.put_conflict(conflict)

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        actor_id: str,
        actor_kind: ActorKind,
        resolution: str,
        dismissed: bool = False,
    ) -> KnowledgeConflict:
        if actor_kind not in {ActorKind.VALIDATOR, ActorKind.HUMAN}:
            raise PromotionError("only validator or human can resolve a knowledge conflict")
        conflict = self.store.get_conflict(conflict_id)
        if conflict is None:
            raise IntegrityViolationError("unknown knowledge conflict", conflict_id=conflict_id)
        if conflict.state != ConflictState.OPEN:
            raise IntegrityViolationError("knowledge conflict is already closed", conflict_id=conflict_id)
        if not resolution.strip():
            raise IntegrityViolationError("conflict resolution must be non-empty")
        updated = conflict.model_copy(
            update={
                "state": ConflictState.DISMISSED if dismissed else ConflictState.RESOLVED,
                "resolution": resolution,
                "resolved_at": datetime.now(timezone.utc),
                "resolved_by": actor_id,
            }
        )
        self.store.put_conflict(updated)
        return updated

    def _open_observation_conflict(self, observation: ExecutionObservation) -> None:
        if (
            observation.verification != VerificationState.ACCEPTED
            or not observation.verifier_id
            or not observation.contract_id
            or observation.succeeded is not True
        ):
            return
        contract = self.store.get_contract(observation.contract_id)
        if contract is None:
            return
        contradictions = tuple(
            observed
            for expected in contract.effects
            for observed in observation.state_after
            if predicates_conflict(expected, observed)
        )
        if not contradictions:
            return
        conflict = KnowledgeConflict(
            conflict_id=stable_id(
                "conflict",
                contract.contract_id,
                observation.observation_id,
                "verified-effect-contradiction",
            ),
            software_id=observation.software_id,
            item_kind="contract",
            item_id=contract.contract_id,
            reason="independently verified observation contradicts a declared effect",
            evidence_ids=observation.evidence_ids,
            observation_ids=(observation.observation_id,),
        )
        self.open_conflict(conflict, actor_kind=ActorKind.VALIDATOR)
        for candidate in self.store.list_skill_candidates(observation.software_id):
            if (
                candidate.status == "accepted"
                and contract.contract_id in candidate.operation_ids
            ):
                self.store.put_skill_candidate(
                    candidate.model_copy(
                        update={
                            "status": "deprecated",
                            "reviewed_by": observation.verifier_id,
                            "review_reason": conflict.reason,
                            "revision": candidate.revision + 1,
                        }
                    )
                )

    def _skill_episodes(self, candidate: SkillCandidate) -> tuple[CompactEpisode, ...]:
        if len(set(candidate.source_episode_ids)) != len(candidate.source_episode_ids):
            raise IntegrityViolationError("skill source episodes must be unique")
        episodes: list[CompactEpisode] = []
        for episode_id in candidate.source_episode_ids:
            episode = self.store.get_episode(episode_id)
            if episode is None or episode.software_id != candidate.software_id:
                raise IntegrityViolationError(
                    "skill references unknown or cross-software episode",
                    episode_id=episode_id,
                )
            if episode.action_contract_id not in candidate.operation_ids:
                raise IntegrityViolationError(
                    "skill episode does not exercise a candidate operation",
                    episode_id=episode_id,
                )
            episodes.append(episode)
        return tuple(episodes)

    def _validate_source_requirements(
        self, status: KnowledgeStatus, evidence_ids: tuple[str, ...]
    ) -> None:
        evidence = tuple(self.store.get_evidence(item) for item in evidence_ids)
        records = tuple(item for item in evidence if item is not None)
        if status == KnowledgeStatus.CROSS_SOURCE_CONFIRMED:
            if len({item.source_uri for item in records}) < 2:
                raise PromotionError("cross-source confirmation requires two independent source URIs")
        if status == KnowledgeStatus.SOURCE_CODE_CONFIRMED:
            if not any(item.source_kind.value == "source_code" for item in records):
                raise PromotionError("source-code confirmation requires source-code evidence")

    def _require_verified_observations(
        self, item: KnowledgeObject, observation_ids: tuple[str, ...]
    ) -> None:
        if not observation_ids:
            raise PromotionError("execution-based promotion requires a verified observation")
        for observation_id in observation_ids:
            observation = self.store.get_observation(observation_id)
            if observation is None:
                raise PromotionError(f"unknown observation: {observation_id}")
            if observation.software_id != item.software_id:
                raise PromotionError("observation belongs to different software")
            if isinstance(item, OperationContract) and observation.contract_id != item.contract_id:
                raise PromotionError("observation belongs to a different contract")
            if observation.verification != VerificationState.ACCEPTED or not observation.verifier_id:
                raise PromotionError("observation was not independently accepted")

    def _validate_evidence_scope(self, software_id: str, evidence_ids: tuple[str, ...]) -> None:
        for evidence_id in evidence_ids:
            evidence = self.store.get_evidence(evidence_id)
            if evidence is None:
                raise IntegrityViolationError("unknown evidence reference", evidence_id=evidence_id)
            if evidence.software_id != software_id and software_id not in evidence.allowed_software_ids:
                raise IntegrityViolationError(
                    "cross-software evidence was not explicitly allowed",
                    evidence_id=evidence_id,
                    evidence_software_id=evidence.software_id,
                    target_software_id=software_id,
                )

    def _validate_item_links(self, item: KnowledgeObject) -> None:
        self._require_software(item.software_id)
        self._validate_evidence_scope(item.software_id, item.evidence_ids)
        if isinstance(item, Relation):
            self._require_entity(item.source_entity_id, item.software_id)
            self._require_entity(item.target_entity_id, item.software_id)
        elif isinstance(item, OperationContract):
            for entity_id in item.related_entity_ids:
                self._require_entity(entity_id, item.software_id)
            operation_ids = {
                *item.verification_operations,
                *(operation for repair in item.repair_strategies for operation in repair.required_operations),
            }
            for operation_id in operation_ids:
                self._require_contract(operation_id, item.software_id)
        elif isinstance(item, Workflow):
            for step in item.steps:
                self._require_contract(step.operation_id, item.software_id)

    @staticmethod
    def _item_identity(item: KnowledgeObject) -> tuple[ItemKind, str]:
        if isinstance(item, Entity):
            return "entity", item.entity_id
        if isinstance(item, OperationContract):
            return "contract", item.contract_id
        if isinstance(item, Relation):
            return "relation", item.relation_id
        return "workflow", item.workflow_id

    def _require_record_status(
        self, item_kind: ItemKind, item_id: str, status: KnowledgeStatus
    ) -> None:
        existing = self._find_item(item_kind, item_id)
        if existing is None:
            self._require_initial_status(status)
        elif status != existing.status:
            raise PromotionError("knowledge status changes must use the audited transition API")

    def _find_item(self, item_kind: ItemKind, item_id: str) -> KnowledgeObject | None:
        if item_kind == "entity":
            return self.store.get_entity(item_id)
        if item_kind == "contract":
            return self.store.get_contract(item_id)
        if item_kind == "relation":
            return self.store.get_relation(item_id)
        return self.store.get_workflow(item_id)

    def _get_item(self, item_kind: ItemKind, item_id: str) -> KnowledgeObject:
        item = self._find_item(item_kind, item_id)
        if item is None:
            raise IntegrityViolationError("unknown knowledge item", item_kind=item_kind, item_id=item_id)
        return item

    def _require_software(self, software_id: str) -> SoftwareIdentity:
        software = self.store.get_software(software_id)
        if software is None:
            raise IntegrityViolationError("unknown software identity", software_id=software_id)
        return software

    def _require_entity(self, entity_id: str, software_id: str) -> Entity:
        entity = self.store.get_entity(entity_id)
        if entity is None or entity.software_id != software_id:
            raise IntegrityViolationError(
                "unknown or cross-software entity reference",
                entity_id=entity_id,
                software_id=software_id,
            )
        return entity

    def _require_contract(self, contract_id: str, software_id: str) -> OperationContract:
        contract = self.store.get_contract(contract_id)
        if contract is None or contract.software_id != software_id:
            raise IntegrityViolationError(
                "unknown or cross-software operation reference",
                contract_id=contract_id,
                software_id=software_id,
            )
        return contract

    @staticmethod
    def _require_initial_status(status: KnowledgeStatus) -> None:
        if status not in {KnowledgeStatus.EXTRACTED, KnowledgeStatus.VERSION_UNCONFIRMED}:
            raise PromotionError(
                "new knowledge must begin as extracted or version_unconfirmed and be reviewed"
            )
