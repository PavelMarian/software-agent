"""Evidence-grounded intent locking and acceptance-contract compilation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Any, Sequence

from pydantic import Field, model_validator

from software_multiagent.software_memory.schema.models import (
    FrozenModel,
    RetrievalProfile,
    StatePredicate,
    stable_id,
    utc_now,
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _hash(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _unique_predicates(values: Sequence[StatePredicate]) -> tuple[StatePredicate, ...]:
    return tuple({item.key: item for item in values}.values())


class RequirementOrigin(str, Enum):
    TASK_QUOTE = "task_quote"
    SOFTWARE_PROFILE = "software_profile"
    METHOD_NORM = "method_norm"
    SAFETY_POLICY = "safety_policy"
    BENCHMARK_RULE = "benchmark_rule"


class CriterionCategory(str, Enum):
    OUTCOME = "outcome"
    METHOD = "method"
    ARTIFACT = "artifact"
    PROVENANCE = "provenance"
    SAFETY = "safety"


class CompilationStatus(str, Enum):
    READY = "ready"
    KNOWLEDGE_GAP = "knowledge_gap"
    NEEDS_HUMAN_SPECIFICATION = "needs_human_specification"
    UNSUPPORTED = "unsupported"


class AcceptanceContractStatus(str, Enum):
    LOCKED = "locked"
    SUPERSEDED = "superseded"


class TaskRequirement(FrozenModel):
    requirement_id: str
    text: str = Field(min_length=1, max_length=4_000)
    category: CriterionCategory = CriterionCategory.OUTCOME
    mandatory: bool = True
    origin: RequirementOrigin
    source_quote: str | None = None
    quote_start: int | None = Field(default=None, ge=0)
    quote_end: int | None = Field(default=None, ge=0)
    origin_reference: str | None = None
    evidence_ids: tuple[str, ...] = ()
    applicability: dict[str, Any] = Field(default_factory=dict)
    compilation_hint: CompilationStatus | None = None
    resolution_question: str | None = None

    @model_validator(mode="after")
    def validate_origin(self) -> "TaskRequirement":
        if self.origin == RequirementOrigin.TASK_QUOTE:
            if not self.source_quote or self.quote_start is None or self.quote_end is None:
                raise ValueError("task-quote requirement needs quote text and offsets")
            if self.quote_end <= self.quote_start:
                raise ValueError("source quote offsets must define a non-empty span")
        elif (
            self.source_quote is not None
            or self.quote_start is not None
            or self.quote_end is not None
        ):
            raise ValueError("implicit requirement cannot impersonate a task quote")
        elif not self.evidence_ids and not self.origin_reference:
            raise ValueError("implicit requirement needs evidence or an origin reference")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("requirement evidence IDs must be unique")
        if (
            self.compilation_hint == CompilationStatus.NEEDS_HUMAN_SPECIFICATION
            and not self.resolution_question
        ):
            raise ValueError("human-specification hint needs a resolution question")
        return self


class RequirementSeed(FrozenModel):
    """Compiler input; task quotes are resolved to exact offsets before locking."""

    text: str = Field(min_length=1, max_length=4_000)
    category: CriterionCategory = CriterionCategory.OUTCOME
    mandatory: bool = True
    origin: RequirementOrigin = RequirementOrigin.TASK_QUOTE
    source_quote: str | None = None
    origin_reference: str | None = None
    evidence_ids: tuple[str, ...] = ()
    applicability: dict[str, Any] = Field(default_factory=dict)
    compilation_hint: CompilationStatus | None = None
    resolution_question: str | None = None


class IntentLock(FrozenModel):
    lock_id: str
    task_id: str
    source_text: str = Field(min_length=1)
    source_hash: str
    requirements: tuple[TaskRequirement, ...]
    lock_hash: str
    revision: int = Field(default=1, ge=1)
    supersedes: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_lock(self) -> "IntentLock":
        if not self.requirements:
            raise ValueError("intent lock needs at least one requirement")
        ids = [item.requirement_id for item in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("intent requirement IDs must be unique")
        if sha256(self.source_text.encode("utf-8")).hexdigest() != self.source_hash:
            raise ValueError("intent source hash does not match source text")
        for item in self.requirements:
            if item.origin != RequirementOrigin.TASK_QUOTE:
                continue
            assert item.quote_start is not None and item.quote_end is not None
            if self.source_text[item.quote_start:item.quote_end] != item.source_quote:
                raise ValueError("source quote does not match the locked task text")
        payload = self.hash_payload(
            self.task_id,
            self.source_text,
            self.requirements,
            self.revision,
            self.supersedes,
        )
        if _hash(payload) != self.lock_hash:
            raise ValueError("intent lock hash is invalid")
        return self

    @staticmethod
    def hash_payload(
        task_id: str,
        source_text: str,
        requirements: Sequence[TaskRequirement],
        revision: int,
        supersedes: str | None,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "source_text": source_text,
            "requirements": [
                item.model_dump(mode="json", exclude_none=True) for item in requirements
            ],
            "revision": revision,
            "supersedes": supersedes,
        }

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        source_text: str,
        seeds: Sequence[RequirementSeed],
        revision: int = 1,
        supersedes: str | None = None,
    ) -> "IntentLock":
        requirements: list[TaskRequirement] = []
        search_offsets: dict[str, int] = {}
        for index, seed in enumerate(seeds):
            quote_start = quote_end = None
            quote = seed.source_quote
            if seed.origin == RequirementOrigin.TASK_QUOTE:
                quote = quote or seed.text
                start_at = search_offsets.get(quote, 0)
                quote_start = source_text.find(quote, start_at)
                if quote_start < 0:
                    raise ValueError(f"task quote is absent from source text: {quote!r}")
                quote_end = quote_start + len(quote)
                search_offsets[quote] = quote_end
            requirement_id = stable_id(
                "requirement",
                task_id,
                str(revision),
                str(index),
                seed.origin.value,
                seed.text,
            )
            requirements.append(
                TaskRequirement(
                    requirement_id=requirement_id,
                    text=seed.text,
                    category=seed.category,
                    mandatory=seed.mandatory,
                    origin=seed.origin,
                    source_quote=quote,
                    quote_start=quote_start,
                    quote_end=quote_end,
                    origin_reference=seed.origin_reference,
                    evidence_ids=seed.evidence_ids,
                    applicability=seed.applicability,
                    compilation_hint=seed.compilation_hint,
                    resolution_question=seed.resolution_question,
                )
            )
        payload = cls.hash_payload(task_id, source_text, requirements, revision, supersedes)
        lock_hash = _hash(payload)
        return cls(
            lock_id=stable_id("intent", task_id, str(revision), lock_hash),
            task_id=task_id,
            source_text=source_text,
            source_hash=sha256(source_text.encode("utf-8")).hexdigest(),
            requirements=tuple(requirements),
            lock_hash=lock_hash,
            revision=revision,
            supersedes=supersedes,
        )


class AcceptanceCriterion(FrozenModel):
    criterion_id: str
    requirement_id: str
    category: CriterionCategory
    mandatory: bool = True
    compilation_status: CompilationStatus
    expected_predicates: tuple[StatePredicate, ...] = ()
    verification_operation_ids: tuple[str, ...] = ()
    knowledge_evidence_ids: tuple[str, ...] = ()
    knowledge_gap: str | None = None
    human_question: str | None = None
    unsupported_reason: str | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def validate_compilation(self) -> "AcceptanceCriterion":
        if self.compilation_status == CompilationStatus.READY:
            if not self.expected_predicates and not self.verification_operation_ids:
                raise ValueError("ready criterion needs a predicate or verification operation")
        elif self.compilation_status == CompilationStatus.KNOWLEDGE_GAP:
            if not self.knowledge_gap:
                raise ValueError("knowledge-gap criterion needs a focused question")
        elif self.compilation_status == CompilationStatus.NEEDS_HUMAN_SPECIFICATION:
            if not self.human_question:
                raise ValueError("human-specification criterion needs a question")
        elif self.compilation_status == CompilationStatus.UNSUPPORTED:
            if not self.unsupported_reason:
                raise ValueError("unsupported criterion needs a reason")
        return self


class AcceptanceContract(FrozenModel):
    acceptance_contract_id: str
    task_id: str
    software_id: str
    software_version: str | None = None
    intent_lock_id: str
    intent_lock_hash: str
    criteria: tuple[AcceptanceCriterion, ...]
    status: AcceptanceContractStatus = AcceptanceContractStatus.LOCKED
    contract_hash: str
    revision: int = Field(default=1, ge=1)
    supersedes: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_contract(self) -> "AcceptanceContract":
        ids = [item.criterion_id for item in self.criteria]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("acceptance criterion IDs must be non-empty and unique")
        payload = self.hash_payload(
            self.task_id,
            self.software_id,
            self.software_version,
            self.intent_lock_id,
            self.intent_lock_hash,
            self.criteria,
            self.revision,
            self.supersedes,
        )
        if _hash(payload) != self.contract_hash:
            raise ValueError("acceptance contract hash is invalid")
        return self

    @staticmethod
    def hash_payload(
        task_id: str,
        software_id: str,
        software_version: str | None,
        intent_lock_id: str,
        intent_lock_hash: str,
        criteria: Sequence[AcceptanceCriterion],
        revision: int,
        supersedes: str | None,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "software_id": software_id,
            "software_version": software_version,
            "intent_lock_id": intent_lock_id,
            "intent_lock_hash": intent_lock_hash,
            "criteria": [item.model_dump(mode="json", exclude_none=True) for item in criteria],
            "revision": revision,
            "supersedes": supersedes,
        }

    @classmethod
    def create_locked(
        cls,
        *,
        intent: IntentLock,
        software_id: str,
        software_version: str | None,
        criteria: Sequence[AcceptanceCriterion],
        revision: int = 1,
        supersedes: str | None = None,
    ) -> "AcceptanceContract":
        requirement_ids = {item.requirement_id for item in intent.requirements}
        criterion_requirements = {item.requirement_id for item in criteria}
        if criterion_requirements != requirement_ids:
            raise ValueError("acceptance criteria must cover every locked requirement exactly")
        payload = cls.hash_payload(
            intent.task_id,
            software_id,
            software_version,
            intent.lock_id,
            intent.lock_hash,
            criteria,
            revision,
            supersedes,
        )
        contract_hash = _hash(payload)
        return cls(
            acceptance_contract_id=stable_id(
                "acceptance", intent.task_id, str(revision), contract_hash
            ),
            task_id=intent.task_id,
            software_id=software_id,
            software_version=software_version,
            intent_lock_id=intent.lock_id,
            intent_lock_hash=intent.lock_hash,
            criteria=tuple(criteria),
            contract_hash=contract_hash,
            revision=revision,
            supersedes=supersedes,
        )


class ContractAmendment(FrozenModel):
    amendment_id: str
    task_id: str
    previous_contract_id: str
    new_contract_id: str
    actor_id: str
    reason: str = Field(min_length=1, max_length=2_000)
    evidence_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)


@dataclass
class AcceptanceCompiler:
    """Compile locked requirements from existing evidence-grounded memory."""

    retriever: Any

    def compile(
        self,
        intent: IntentLock,
        *,
        software_id: str,
        software_version: str | None = None,
        include_profile_constraints: bool = True,
    ) -> tuple[IntentLock, AcceptanceContract]:
        from software_multiagent.software_memory.schema.models import KnowledgeRequest

        effective_intent = (
            self._with_profile_constraints(intent, software_id, software_version)
            if include_profile_constraints
            else intent
        )
        criteria: list[AcceptanceCriterion] = []
        for requirement in effective_intent.requirements:
            if requirement.compilation_hint == CompilationStatus.NEEDS_HUMAN_SPECIFICATION:
                criteria.append(
                    AcceptanceCriterion(
                        criterion_id=stable_id("criterion", requirement.requirement_id),
                        requirement_id=requirement.requirement_id,
                        category=requirement.category,
                        mandatory=requirement.mandatory,
                        compilation_status=CompilationStatus.NEEDS_HUMAN_SPECIFICATION,
                        knowledge_evidence_ids=requirement.evidence_ids,
                        human_question=requirement.resolution_question,
                        rationale="requirement is subjective or underspecified",
                    )
                )
                continue
            if requirement.compilation_hint == CompilationStatus.UNSUPPORTED:
                criteria.append(
                    AcceptanceCriterion(
                        criterion_id=stable_id("criterion", requirement.requirement_id),
                        requirement_id=requirement.requirement_id,
                        category=requirement.category,
                        mandatory=requirement.mandatory,
                        compilation_status=CompilationStatus.UNSUPPORTED,
                        knowledge_evidence_ids=requirement.evidence_ids,
                        unsupported_reason=(
                            requirement.resolution_question
                            or "No supported verification representation is available"
                        ),
                        rationale="requirement was classified as unsupported",
                    )
                )
                continue
            if requirement.origin == RequirementOrigin.SOFTWARE_PROFILE:
                predicate = StatePredicate(
                    predicate=requirement.text,
                    subject=software_id,
                )
                criteria.append(
                    self._criterion(
                        requirement,
                        CompilationStatus.READY,
                        expected_predicates=(predicate,),
                        evidence_ids=requirement.evidence_ids,
                        rationale="constraint supplied by the versioned software profile",
                    )
                )
                continue
            packet = self.retriever.retrieve(
                KnowledgeRequest(
                    software_id=software_id,
                    query=requirement.text,
                    version=software_version,
                    desired_capability=requirement.text,
                    max_items=5,
                    token_budget=1_500,
                    profile=RetrievalProfile.GENERIC,
                )
            )
            fingerprints = [item for item in packet.items if item.item_type == "fingerprint"]
            predicates: list[StatePredicate] = []
            verification_ids: list[str] = []
            evidence_ids = list(requirement.evidence_ids)
            for item in fingerprints[:2]:
                contract = self.retriever.store.get_contract(item.item_id)
                if contract is None:
                    continue
                predicates.extend(contract.success_signals or contract.effects)
                verification_ids.extend(contract.verification_operations)
                evidence_ids.extend(contract.evidence_ids)
            if predicates or verification_ids:
                criteria.append(
                    self._criterion(
                        requirement,
                        CompilationStatus.READY,
                        expected_predicates=_unique_predicates(predicates),
                        verification_ids=tuple(dict.fromkeys(verification_ids)),
                        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
                        rationale="compiled from retrieved operation contracts",
                    )
                )
            else:
                criteria.append(
                    self._criterion(
                        requirement,
                        CompilationStatus.KNOWLEDGE_GAP,
                        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
                        knowledge_gap=(
                            "Find a version-aligned observable predicate or native verification "
                            f"operation for: {requirement.text}"
                        ),
                        rationale="retrieval found no executable acceptance basis",
                    )
                )
        return effective_intent, AcceptanceContract.create_locked(
            intent=effective_intent,
            software_id=software_id,
            software_version=software_version,
            criteria=criteria,
            revision=effective_intent.revision,
        )

    @staticmethod
    def _criterion(
        requirement: TaskRequirement,
        status: CompilationStatus,
        *,
        expected_predicates: tuple[StatePredicate, ...] = (),
        verification_ids: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
        knowledge_gap: str | None = None,
        rationale: str,
    ) -> AcceptanceCriterion:
        return AcceptanceCriterion(
            criterion_id=stable_id("criterion", requirement.requirement_id),
            requirement_id=requirement.requirement_id,
            category=requirement.category,
            mandatory=requirement.mandatory,
            compilation_status=status,
            expected_predicates=expected_predicates,
            verification_operation_ids=verification_ids,
            knowledge_evidence_ids=evidence_ids,
            knowledge_gap=knowledge_gap,
            rationale=rationale,
        )

    def _with_profile_constraints(
        self,
        intent: IntentLock,
        software_id: str,
        software_version: str | None,
    ) -> IntentLock:
        profiles = self.retriever.store.list_profiles(software_id)
        existing = {item.text for item in intent.requirements}
        extra: list[RequirementSeed] = []
        for profile in profiles:
            mandatory = set(profile.metadata.get("mandatory_domain_predicates", ()))
            for predicate in profile.domain_predicates:
                if predicate in existing:
                    continue
                extra.append(
                    RequirementSeed(
                        text=predicate,
                        category=CriterionCategory.SAFETY,
                        mandatory=predicate in mandatory,
                        origin=RequirementOrigin.SOFTWARE_PROFILE,
                        origin_reference=profile.profile_id,
                        applicability={"software_id": software_id, "version": software_version},
                    )
                )
        if not extra:
            return intent
        original = [
            RequirementSeed(
                text=item.text,
                category=item.category,
                mandatory=item.mandatory,
                origin=item.origin,
                source_quote=item.source_quote,
                origin_reference=item.origin_reference,
                evidence_ids=item.evidence_ids,
                applicability=item.applicability,
                compilation_hint=item.compilation_hint,
                resolution_question=item.resolution_question,
            )
            for item in intent.requirements
        ]
        return IntentLock.create(
            task_id=intent.task_id,
            source_text=intent.source_text,
            seeds=(*original, *extra),
            revision=intent.revision,
            supersedes=intent.supersedes,
        )


__all__ = [
    "AcceptanceCompiler",
    "AcceptanceContract",
    "AcceptanceContractStatus",
    "AcceptanceCriterion",
    "CompilationStatus",
    "ContractAmendment",
    "CriterionCategory",
    "IntentLock",
    "RequirementOrigin",
    "RequirementSeed",
    "TaskRequirement",
]
