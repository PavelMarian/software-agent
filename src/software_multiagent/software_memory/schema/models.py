from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(prefix: str, *parts: str) -> str:
    normalized = "\x1f".join(str(part).strip() for part in parts)
    return f"{prefix}_{sha256(normalized.encode('utf-8')).hexdigest()[:24]}"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KnowledgeStatus(str, Enum):
    EXTRACTED = "extracted"
    EVIDENCE_SUPPORTED = "evidence_supported"
    CROSS_SOURCE_CONFIRMED = "cross_source_confirmed"
    SOURCE_CODE_CONFIRMED = "source_code_confirmed"
    EXECUTION_VERIFIED = "execution_verified"
    TASK_VERIFIED = "task_verified"
    VERSION_UNCONFIRMED = "version_unconfirmed"
    CONFLICTING = "conflicting"
    DEPRECATED = "deprecated"
    RETRACTED = "retracted"


class VerificationState(str, Enum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ActorKind(str, Enum):
    INGESTOR = "ingestor"
    AGENT = "agent"
    VALIDATOR = "validator"
    HUMAN = "human"
    SYSTEM = "system"


class EvidenceSourceKind(str, Enum):
    DOCUMENTATION = "documentation"
    OPENAPI = "openapi"
    CLI_HELP = "cli_help"
    SOURCE_CODE = "source_code"
    CONFIGURATION_SCHEMA = "configuration_schema"
    EXECUTION = "execution"
    BENCHMARK = "benchmark"
    HUMAN = "human"
    OTHER = "other"


class ContextMode(str, Enum):
    NONE = "none"
    FILTERED = "filtered"
    FULL_DUMP = "full_dump"


class RetrievalProfile(str, Enum):
    GENERIC = "generic"
    PLANNING = "planning"
    EXECUTION = "execution"
    REPAIR = "repair"


class PacketBudget(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    @property
    def tokens(self) -> int:
        return {
            PacketBudget.SMALL: 512,
            PacketBudget.MEDIUM: 1_500,
            PacketBudget.LARGE: 4_000,
        }[self]


class ConflictState(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class SoftwareInterface(FrozenModel):
    name: str
    modality: Literal["cli", "api", "gui", "file", "sql", "notebook", "library", "other"]
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSourceDescriptor(FrozenModel):
    source_kind: EvidenceSourceKind
    uri: str
    version_hint: str | None = None
    authoritative: bool = False
    priority: int = Field(default=0, ge=-100, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SoftwareIdentity(FrozenModel):
    software_id: str
    product: str = Field(min_length=1)
    version: str = Field(default="unknown", min_length=1)
    vendor: str | None = None
    edition: str | None = None
    build: str | None = None
    distribution: str | None = None
    platform: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        product: str,
        version: str = "unknown",
        *,
        vendor: str | None = None,
        edition: str | None = None,
        build: str | None = None,
        distribution: str | None = None,
        platform: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SoftwareIdentity":
        identity = (vendor or "", product, edition or "", version, build or "", distribution or "", platform or "")
        return cls(
            software_id=stable_id("software", *identity),
            product=product,
            version=version,
            vendor=vendor,
            edition=edition,
            build=build,
            distribution=distribution,
            platform=platform,
            metadata=metadata or {},
        )


class SoftwareProfile(FrozenModel):
    profile_id: str
    software_id: str
    interfaces: tuple[SoftwareInterface, ...] = ()
    knowledge_sources: tuple[KnowledgeSourceDescriptor, ...] = ()
    artifact_patterns: tuple[str, ...] = ()
    configuration_formats: tuple[str, ...] = ()
    discovery_operations: tuple[str, ...] = ()
    validation_operations: tuple[str, ...] = ()
    dangerous_operations: tuple[str, ...] = ()
    domain_predicates: tuple[str, ...] = ()
    extractor_preferences: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    revision: int = Field(default=1, ge=1)



class VersionScope(FrozenModel):
    exact: tuple[str, ...] = ()
    compatible: tuple[str, ...] = ()
    minimum: str | None = None
    maximum: str | None = None
    excluded: tuple[str, ...] = ()
    compatibility_verified: bool = False
    note: str | None = None

    def matches(self, version: str | None) -> bool:
        if not version or version == "unknown":
            return not self.exact and self.minimum is None and self.maximum is None
        if version in self.excluded:
            return False
        if self.exact or self.compatible:
            return version in {*self.exact, *self.compatible}
        # Version strings are intentionally not ordered here: product-specific version
        # semantics belong in an adapter. Generic core must not compare values such as
        # v2606, 10, 1.34, and rolling-release names lexicographically.
        return self.minimum is None and self.maximum is None


class EvidenceRecord(FrozenModel):
    evidence_id: str
    software_id: str
    source_kind: EvidenceSourceKind
    source_uri: str
    locator: str | None = None
    content: str = Field(min_length=1)
    content_hash: str
    retrieved_at: datetime = Field(default_factory=utc_now)
    source_version: str | None = None
    authoritative: bool = False
    allowed_software_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        software_id: str,
        source_kind: EvidenceSourceKind,
        source_uri: str,
        content: str,
        locator: str | None = None,
        source_version: str | None = None,
        authoritative: bool = False,
        allowed_software_ids: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> "EvidenceRecord":
        digest = sha256(content.encode("utf-8")).hexdigest()
        return cls(
            evidence_id=stable_id("evidence", software_id, source_uri, locator or "", digest),
            software_id=software_id,
            source_kind=source_kind,
            source_uri=source_uri,
            locator=locator,
            content=content,
            content_hash=digest,
            source_version=source_version,
            authoritative=authoritative,
            allowed_software_ids=allowed_software_ids,
            metadata=metadata or {},
        )


class Entity(FrozenModel):
    entity_id: str
    software_id: str
    entity_type: str = Field(min_length=1)
    name: str = Field(min_length=1)
    summary: str = ""
    aliases: tuple[str, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    version_scope: VersionScope = Field(default_factory=VersionScope)
    status: KnowledgeStatus = KnowledgeStatus.EXTRACTED
    verification: VerificationState = VerificationState.UNREVIEWED
    revision: int = Field(default=1, ge=1)


class Relation(FrozenModel):
    relation_id: str
    software_id: str
    source_entity_id: str
    relation_type: str = Field(min_length=1)
    target_entity_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    status: KnowledgeStatus = KnowledgeStatus.EXTRACTED
    revision: int = Field(default=1, ge=1)


class RelatedEntity(FrozenModel):
    relation: Relation
    entity: Entity
    direction: Literal["outgoing", "incoming"]


class StatePredicate(FrozenModel):
    predicate: str = Field(min_length=1)
    subject: str | None = None
    value: Any = True
    arguments: dict[str, Any] = Field(default_factory=dict)
    negated: bool = False

    @property
    def key(self) -> str:
        return stable_id(
            "predicate",
            self.predicate,
            self.subject or "",
            str(self.value),
            str(sorted(self.arguments.items())),
            str(self.negated),
        )


class OperationParameter(FrozenModel):
    name: str
    required: bool = False
    parameter_type: str | None = None
    description: str | None = None
    default: Any = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()


class FailureSignature(FrozenModel):
    signature_id: str
    pattern: str
    category: Literal["environment", "artifact", "state", "protocol", "approach", "unknown"] = "unknown"
    description: str = ""
    related_entity_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class RepairStrategy(FrozenModel):
    repair_id: str
    name: str
    description: str
    failure_signature_ids: tuple[str, ...] = ()
    required_operations: tuple[str, ...] = ()
    postconditions: tuple[StatePredicate, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class OperationContract(FrozenModel):
    contract_id: str
    software_id: str
    name: str = Field(min_length=1)
    interface: str = Field(min_length=1)
    purpose: str = ""
    parameters: tuple[OperationParameter, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    preconditions: tuple[StatePredicate, ...] = ()
    effects: tuple[StatePredicate, ...] = ()
    side_effects: tuple[StatePredicate, ...] = ()
    success_signals: tuple[StatePredicate, ...] = ()
    failure_signatures: tuple[FailureSignature, ...] = ()
    repair_strategies: tuple[RepairStrategy, ...] = ()
    verification_operations: tuple[str, ...] = ()
    related_entity_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    version_scope: VersionScope = Field(default_factory=VersionScope)
    risk_level: Literal["read_only", "low", "medium", "high", "destructive"] = "low"
    status: KnowledgeStatus = KnowledgeStatus.EXTRACTED
    verification: VerificationState = VerificationState.UNREVIEWED
    metadata: dict[str, Any] = Field(default_factory=dict)
    revision: int = Field(default=1, ge=1)
    supersedes: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def require_evidence_for_supported_status(self) -> "OperationContract":
        supported = {
            KnowledgeStatus.EVIDENCE_SUPPORTED,
            KnowledgeStatus.CROSS_SOURCE_CONFIRMED,
            KnowledgeStatus.SOURCE_CODE_CONFIRMED,
            KnowledgeStatus.EXECUTION_VERIFIED,
            KnowledgeStatus.TASK_VERIFIED,
        }
        if self.status in supported and not self.evidence_ids:
            raise ValueError(f"{self.status.value} contract requires evidence")
        return self


class OperationFingerprint(FrozenModel):
    contract_id: str
    software_id: str
    name: str
    interface: str
    purpose: str
    required_parameters: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    status: KnowledgeStatus
    version_scope: VersionScope

    @classmethod
    def from_contract(cls, contract: OperationContract) -> "OperationFingerprint":
        return cls(
            contract_id=contract.contract_id,
            software_id=contract.software_id,
            name=contract.name,
            interface=contract.interface,
            purpose=contract.purpose,
            required_parameters=tuple(p.name for p in contract.parameters if p.required),
            preconditions=tuple(p.predicate for p in contract.preconditions),
            effects=tuple(p.predicate for p in contract.effects),
            status=contract.status,
            version_scope=contract.version_scope,
        )


class WorkflowStep(FrozenModel):
    step_id: str
    operation_id: str
    requires: tuple[str, ...] = ()
    on_success: tuple[str, ...] = ()
    on_failure: tuple[str, ...] = ()
    completion_predicates: tuple[StatePredicate, ...] = ()


class Workflow(FrozenModel):
    workflow_id: str
    software_id: str
    name: str
    objective: str
    steps: tuple[WorkflowStep, ...]
    entry_steps: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()
    version_scope: VersionScope = Field(default_factory=VersionScope)
    status: KnowledgeStatus = KnowledgeStatus.EXTRACTED
    revision: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_graph(self) -> "Workflow":
        ids = {step.step_id for step in self.steps}
        if not ids or not set(self.entry_steps).issubset(ids):
            raise ValueError("workflow entry_steps must reference declared steps")
        targets = {target for step in self.steps for target in (*step.on_success, *step.on_failure)}
        if not targets.issubset(ids):
            raise ValueError("workflow transition references an unknown step")
        return self


class ExecutionObservation(FrozenModel):
    observation_id: str
    software_id: str
    contract_id: str | None
    actor_id: str
    actor_kind: ActorKind = ActorKind.AGENT
    task_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    state_before: tuple[StatePredicate, ...] = ()
    state_after: tuple[StatePredicate, ...] = ()
    output_excerpt: str = ""
    exit_code: int | None = None
    succeeded: bool | None = None
    verifier_id: str | None = None
    verification: VerificationState = VerificationState.UNREVIEWED
    evidence_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)


class StatusTransition(FrozenModel):
    transition_id: str
    software_id: str
    item_kind: Literal["entity", "contract", "workflow", "relation"]
    item_id: str
    from_status: KnowledgeStatus
    to_status: KnowledgeStatus
    actor_id: str
    actor_kind: ActorKind
    reason: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeConflict(FrozenModel):
    conflict_id: str
    software_id: str
    item_kind: Literal["entity", "contract", "workflow", "relation"]
    item_id: str
    conflicting_item_id: str | None = None
    reason: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    state: ConflictState = ConflictState.OPEN
    resolution: str | None = None
    opened_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None
    resolved_by: str | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "KnowledgeConflict":
        if self.state == ConflictState.OPEN:
            if self.resolution or self.resolved_at or self.resolved_by:
                raise ValueError("open conflict cannot contain resolution metadata")
        elif not self.resolution or not self.resolved_at or not self.resolved_by:
            raise ValueError("closed conflict requires resolution, time, and actor")
        return self


class KnowledgeRequest(FrozenModel):
    software_id: str
    query: str = ""
    version: str | None = None
    phase: str | None = None
    desired_capability: str | None = None
    known_entity_ids: tuple[str, ...] = ()
    known_files: tuple[str, ...] = ()
    known_commands: tuple[str, ...] = ()
    workflow_id: str | None = None
    workflow_position: tuple[str, ...] = ()
    current_state: tuple[StatePredicate, ...] = ()
    error_signature: str | None = None
    entity_types: tuple[str, ...] = ()
    max_items: int = Field(default=8, ge=1, le=100)
    token_budget: int = Field(default=1_500, ge=64, le=100_000)
    include_evidence: bool = False
    mode: ContextMode = ContextMode.FILTERED
    profile: RetrievalProfile = RetrievalProfile.GENERIC
    budget_profile: PacketBudget | None = None

    @model_validator(mode="after")
    def validate_retrieval_context(self) -> "KnowledgeRequest":
        if self.workflow_position and not self.workflow_id:
            raise ValueError("workflow_position requires workflow_id")
        if self.budget_profile and self.token_budget != self.budget_profile.tokens:
            raise ValueError("token_budget must match the selected fixed budget_profile")
        if len(set(self.known_entity_ids)) != len(self.known_entity_ids):
            raise ValueError("known_entity_ids must not contain duplicates")
        return self

    @classmethod
    def with_budget(
        cls,
        budget: PacketBudget,
        **values: Any,
    ) -> "KnowledgeRequest":
        return cls(budget_profile=budget, token_budget=budget.tokens, **values)


class KnowledgeItem(FrozenModel):
    item_id: str
    item_type: Literal["fingerprint", "contract", "entity", "evidence", "workflow"]
    score: float = 0.0
    content: dict[str, Any]
    estimated_tokens: int = 0


class KnowledgePacket(FrozenModel):
    request: KnowledgeRequest
    items: tuple[KnowledgeItem, ...] = ()
    estimated_tokens: int = 0
    truncated: bool = False
    retrieval_trace: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    candidate_count: int = Field(default=0, ge=0)
    selected_count: int = Field(default=0, ge=0)
    content_characters: int = Field(default=0, ge=0)
    budget_utilization: float = Field(default=0.0, ge=0.0)
    packet_digest: str = ""
    item_hashes: dict[str, str] = Field(default_factory=dict)
    is_delta: bool = False
    base_packet_digest: str | None = None
    removed_item_ids: tuple[str, ...] = ()


class FrontierCandidate(FrozenModel):
    contract_id: str
    name: str
    executable_now: bool
    relevant_effects: tuple[str, ...] = ()
    missing_preconditions: tuple[StatePredicate, ...] = ()
    depth: int = Field(default=0, ge=0)
    causal_path: tuple[str, ...] = ()


class CausalFrontierResult(FrozenModel):
    """Bounded backward-chaining result with an auditable explanation."""

    candidates: tuple[FrontierCandidate, ...] = ()
    unresolved_predicates: tuple[StatePredicate, ...] = ()
    conflicting_predicates: tuple[StatePredicate, ...] = ()
    trace: tuple[str, ...] = ()
    max_depth: int = Field(ge=0)
    max_candidates: int = Field(ge=1)
    truncated: bool = False


class WorkflowCandidate(FrozenModel):
    """Unreviewed alternative path observed during workflow execution."""

    candidate_id: str
    software_id: str
    source_workflow_id: str
    operation_ids: tuple[str, ...]
    completion_predicates: tuple[StatePredicate, ...] = ()
    observation_ids: tuple[str, ...] = ()
    status: Literal["unreviewed", "accepted", "rejected"] = "unreviewed"


class WorkflowExecutionState(FrozenModel):
    """Serializable procedural state kept outside conversational history."""

    run_id: str
    software_id: str
    workflow_id: str
    active_steps: tuple[str, ...]
    completed_steps: tuple[str, ...] = ()
    failed_steps: tuple[str, ...] = ()
    satisfied_predicates: tuple[StatePredicate, ...] = ()
    produced_artifacts: tuple[str, ...] = ()
    operation_history: tuple[str, ...] = ()
    observation_history: tuple[str, ...] = ()
    pending_verifications: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    workflow_candidates: tuple[WorkflowCandidate, ...] = ()
    status: Literal["running", "completed", "failed"] = "running"


class DriftKind(str, Enum):
    REPEATED_WITHOUT_STATE_CHANGE = "repeated_without_state_change"
    INVALID_OPERATION_ORDER = "invalid_operation_order"
    VERIFICATION_SKIPPED = "verification_skipped"
    EFFECT_MISMATCH = "effect_mismatch"
    WORKFLOW_BOUNDARY_EXIT = "workflow_boundary_exit"


class DriftFinding(FrozenModel):
    kind: DriftKind
    message: str
    operation_id: str | None = None
    step_ids: tuple[str, ...] = ()
    predicate_ids: tuple[str, ...] = ()


class WorkflowTransition(FrozenModel):
    previous_state: WorkflowExecutionState
    state: WorkflowExecutionState
    findings: tuple[DriftFinding, ...] = ()
    advanced_steps: tuple[str, ...] = ()
    recovery_steps: tuple[str, ...] = ()


class RepairCategory(str, Enum):
    ENVIRONMENT = "environment"
    ARTIFACT = "artifact"
    STATE = "state"
    PROTOCOL = "protocol"
    APPROACH = "approach"
    UNKNOWN = "unknown"


class NormalizedFailure(FrozenModel):
    signature: str
    category: RepairCategory = RepairCategory.UNKNOWN
    tokens: tuple[str, ...] = ()


class RepairKnowledgeRequest(FrozenModel):
    software_id: str
    error_text: str = Field(min_length=1)
    version: str | None = None
    failed_contract_id: str | None = None
    workflow_id: str | None = None
    workflow_step_id: str | None = None
    current_state: tuple[StatePredicate, ...] = ()
    known_artifacts: tuple[str, ...] = ()
    max_items: int = Field(default=8, ge=1, le=50)
    token_budget: int = Field(default=1_500, ge=128, le=20_000)

    @model_validator(mode="after")
    def step_requires_workflow(self) -> "RepairKnowledgeRequest":
        if self.workflow_step_id and not self.workflow_id:
            raise ValueError("workflow_step_id requires workflow_id")
        return self


class RepairKnowledgeItem(FrozenModel):
    item_id: str
    item_type: Literal["contract", "strategy", "entity", "evidence", "episode"]
    score: float
    content: dict[str, Any]
    estimated_tokens: int = Field(ge=0)


class RepairKnowledgePacket(FrozenModel):
    request: RepairKnowledgeRequest
    failure: NormalizedFailure
    items: tuple[RepairKnowledgeItem, ...] = ()
    trace: tuple[str, ...] = ()
    estimated_tokens: int = Field(default=0, ge=0)
    truncated: bool = False


class RepairPhase(str, Enum):
    INSPECT = "inspect"
    MUTATE = "mutate"
    RERUN = "rerun"
    VERIFY = "verify"
    COMPLETE = "complete"
    FAILED = "failed"


class RepairActionKind(str, Enum):
    INSPECT = "inspect"
    MUTATE = "mutate"
    RERUN = "rerun"
    VERIFY = "verify"
    CORRECT_PROTOCOL = "correct_protocol"
    CHANGE_APPROACH = "change_approach"


class RepairEvent(FrozenModel):
    event_id: str
    action: RepairActionKind
    operation_id: str | None = None
    target: str | None = None
    state_before: tuple[StatePredicate, ...] = ()
    state_after: tuple[StatePredicate, ...] = ()
    succeeded: bool | None = None
    independently_verified: bool = False


class RepairSessionState(FrozenModel):
    session_id: str
    category: RepairCategory
    phase: RepairPhase
    inspection_count: int = Field(default=0, ge=0)
    mutation_count: int = Field(default=0, ge=0)
    rerun_count: int = Field(default=0, ge=0)
    verification_count: int = Field(default=0, ge=0)
    events: tuple[RepairEvent, ...] = ()
    last_observed_state: tuple[StatePredicate, ...] = ()
    stop_reason: str | None = None


class RepairDecision(FrozenModel):
    allowed: bool
    state: RepairSessionState
    reason: str
    repeated_without_state_change: bool = False


class ObservationInterpretation(FrozenModel):
    interpretation_id: str
    software_id: str
    observation_id: str
    actor_id: str
    actor_kind: ActorKind
    summary: str = Field(min_length=1, max_length=2_000)
    failure_category: RepairCategory | None = None
    inferred_predicates: tuple[StatePredicate, ...] = ()
    verification: VerificationState = VerificationState.UNREVIEWED
    verifier_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class CompactEpisode(FrozenModel):
    episode_id: str
    software_id: str
    observation_id: str
    task_id: str
    software_version: str | None = None
    failure_signature: str | None = Field(default=None, max_length=500)
    action_contract_id: str | None = None
    state_changes: tuple[StatePredicate, ...] = ()
    outcome: str = Field(min_length=1, max_length=500)
    succeeded: bool | None = None
    verification: VerificationState = VerificationState.UNREVIEWED
    verifier_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class SkillCandidate(FrozenModel):
    candidate_id: str
    software_id: str
    name: str
    operation_ids: tuple[str, ...]
    source_episode_ids: tuple[str, ...]
    compatible_versions: tuple[str, ...] = ()
    status: Literal["proposed", "accepted", "rejected", "deprecated"] = "proposed"
    reviewed_by: str | None = None
    review_reason: str | None = None
    revision: int = Field(default=1, ge=1)
