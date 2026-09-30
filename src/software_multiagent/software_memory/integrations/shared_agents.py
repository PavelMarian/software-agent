"""Shared software-memory infrastructure for a multi-agent run.

The integration deliberately is not a memory agent.  It gives every role a bounded
view over one persistent store, keeps run-local coordination state outside chat, and
records agent output separately from independently verified verdicts.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from threading import RLock
from typing import Any, Mapping, Sequence

from software_multiagent.core.action_graph import Action
from software_multiagent.core.contracts import AgentSpec, RuntimeState, TaskContext, ToolResult
from software_multiagent.core.execution import (
    KnowledgeContext,
    KnowledgeFragment,
    VerificationResult,
    VerificationStatus,
)
from software_multiagent.ports.protocols import EventSink, NullEventSink
from software_multiagent.tools.registry import AgentCallableTool
from software_multiagent.software_memory.schema.models import (
    ActorKind,
    ConflictState,
    EvidenceRecord,
    EvidenceSourceKind,
    ExecutionObservation,
    KnowledgePacket,
    KnowledgeRequest,
    RepairKnowledgeRequest,
    RetrievalProfile,
    StatePredicate,
    VerificationState,
    stable_id,
)
from software_multiagent.software_memory.operations.retrieval import MemoryRetriever
from software_multiagent.software_memory.operations.service import MemoryService
from software_multiagent.software_memory.operations.repair import compact_episode


class MemoryRole(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"


@dataclass(frozen=True)
class RoleMemoryPolicy:
    """Role-specific limits; values bound memory, not the model's whole context."""

    token_budget: int
    max_items: int
    max_characters: int
    include_evidence: bool = False


class GapResolution(str, Enum):
    MEMORY_RESOLVED = "memory_resolved"
    RESEARCH_REQUIRED = "research_required"


@dataclass(frozen=True)
class KnowledgeGapAssessment:
    """Auditable result of exhausting memory for one planner-declared gap."""

    gap_id: str
    question: str
    resolution: GapResolution
    reason: str
    matched_item_ids: tuple[str, ...] = ()
    matched_evidence_ids: tuple[str, ...] = ()
    corpus_records_checked: int = 0
    retrieval_truncated: bool = False


@dataclass(frozen=True)
class ResearchGateDecision:
    assessments: tuple[KnowledgeGapAssessment, ...] = ()

    @property
    def research_required(self) -> bool:
        return any(
            item.resolution == GapResolution.RESEARCH_REQUIRED
            for item in self.assessments
        )

    @property
    def memory_resolved_count(self) -> int:
        return sum(
            item.resolution == GapResolution.MEMORY_RESOLVED
            for item in self.assessments
        )


DEFAULT_ROLE_POLICIES: Mapping[MemoryRole, RoleMemoryPolicy] = {
    MemoryRole.PLANNER: RoleMemoryPolicy(1_500, 10, 6_000),
    MemoryRole.RESEARCHER: RoleMemoryPolicy(1_500, 8, 6_000, True),
    MemoryRole.EXECUTOR: RoleMemoryPolicy(1_500, 8, 6_000),
    MemoryRole.EVALUATOR: RoleMemoryPolicy(1_500, 8, 6_000, True),
}


@dataclass
class MASMemoryRunState:
    """Structured working memory for one run; never stored in the corpus."""

    task_id: str
    software_id: str
    software_version: str | None = None
    selected_contract_ids: tuple[str, ...] = ()
    current_predicates: tuple[StatePredicate, ...] = ()
    desired_predicates: tuple[StatePredicate, ...] = ()
    workflow_id: str | None = None
    workflow_position: tuple[str, ...] = ()
    produced_artifacts: tuple[str, ...] = ()
    knowledge_gaps: dict[str, str] = field(default_factory=dict)
    resolved_gap_ids: set[str] = field(default_factory=set)
    memory_resolved_gap_ids: set[str] = field(default_factory=set)
    gap_assessments: dict[str, KnowledgeGapAssessment] = field(default_factory=dict)
    gap_resolution_fragments: dict[str, tuple[KnowledgeFragment, ...]] = field(
        default_factory=dict
    )
    evidence_candidate_ids: tuple[str, ...] = ()
    open_conflict_ids: tuple[str, ...] = ()
    last_error: str | None = None
    last_failed_contract_id: str | None = None
    previous_packets: dict[MemoryRole, KnowledgePacket] = field(default_factory=dict)
    delivered_direct_items: dict[MemoryRole, set[str]] = field(default_factory=dict)
    active: bool = True


def _predicates(value: Any) -> tuple[StatePredicate, ...]:
    if value is None:
        return ()
    if isinstance(value, StatePredicate):
        return (value,)
    if isinstance(value, Mapping):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    parsed: list[StatePredicate] = []
    for item in value:
        try:
            parsed.append(
                item if isinstance(item, StatePredicate) else StatePredicate.model_validate(item)
            )
        except (TypeError, ValueError):
            continue
    return tuple(parsed)


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence):
        return tuple(dict.fromkeys(str(item) for item in value if str(item)))
    return ()


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return sha256(_compact_json(value).encode("utf-8")).hexdigest()


_GAP_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "can", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "should", "the",
    "this", "to", "use", "used", "what", "when", "which", "with",
    "как", "какой", "какая", "какие", "для", "или", "при", "что", "это",
}


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w.+/-]+", value.casefold(), flags=re.UNICODE)
        if len(token) >= 3 and token not in _GAP_STOPWORDS
    }


class SharedSoftwareMemory:
    """Role-conditioned read service, run-state coordinator, and safe writer.

    Persistent records remain owned by ``software_memory``.  This class only stores
    cursors and identifiers for a live MAS run, so checkpoints and agent messages do
    not need to duplicate research reports or the knowledge corpus.
    """

    def __init__(
        self,
        retriever: MemoryRetriever,
        *,
        service: MemoryService | None = None,
        events: EventSink | None = None,
        policies: Mapping[MemoryRole, RoleMemoryPolicy] | None = None,
        role_aliases: Mapping[str, MemoryRole | str] | None = None,
    ) -> None:
        self.retriever = retriever
        self.service = service
        self.events = events or NullEventSink()
        self.policies = dict(DEFAULT_ROLE_POLICIES)
        if policies is not None:
            self.policies.update(policies)
        self.role_aliases = {
            key.casefold(): MemoryRole(value) for key, value in (role_aliases or {}).items()
        }
        self._runs: dict[str, MASMemoryRunState] = {}
        self._lock = RLock()

    def begin_run(self, task: TaskContext) -> None:
        with self._lock:
            metadata = task.metadata
            software_id = str(
                metadata.get("software_memory_id")
                or metadata.get("software_id")
                or task.target_software
                or ""
            )
            if not software_id:
                raise ValueError("software memory requires target_software or software_memory_id")
            gaps_value = metadata.get("knowledge_gaps", ())
            gaps = {
                stable_id("gap", task.task_id, text): text
                for text in _strings(gaps_value)
            }
            state = MASMemoryRunState(
                task_id=task.task_id,
                software_id=software_id,
                software_version=(
                    str(metadata["software_version"])
                    if metadata.get("software_version") is not None
                    else None
                ),
                selected_contract_ids=_strings(metadata.get("selected_contract_ids")),
                current_predicates=_predicates(metadata.get("current_state")),
                desired_predicates=_predicates(metadata.get("desired_state")),
                workflow_id=(str(metadata["workflow_id"]) if metadata.get("workflow_id") else None),
                workflow_position=_strings(metadata.get("workflow_position")),
                produced_artifacts=_strings(metadata.get("produced_artifacts")),
                knowledge_gaps=gaps,
            )
            self._validate_selection(state)
            state.open_conflict_ids = self._open_conflict_ids(software_id)
            self._runs[task.task_id] = state
        self._emit("memory_run_started", "memory", state)

    def finish_run(self, task: TaskContext) -> None:
        with self._lock:
            state = self._require_run(task.task_id)
            state.active = False
            state.open_conflict_ids = self._open_conflict_ids(state.software_id)
        self._emit("memory_run_finished", "memory", state)

    def clear_run(self, task_id: str) -> None:
        """Explicitly discard only transient state; persistent memory is untouched."""

        with self._lock:
            self._runs.pop(task_id, None)

    def snapshot(self, task_id: str) -> MASMemoryRunState:
        with self._lock:
            return deepcopy(self._require_run(task_id))

    def update_plan(
        self,
        task_id: str,
        *,
        selected_contract_ids: Sequence[str] | None = None,
        workflow_id: str | None = None,
        workflow_position: Sequence[str] | None = None,
        knowledge_gaps: Sequence[str] | None = None,
    ) -> None:
        """Persist planner handoff as IDs and short questions, never as a full report."""

        with self._lock:
            state = self._require_run(task_id)
            candidate = deepcopy(state)
            if selected_contract_ids is not None:
                candidate.selected_contract_ids = tuple(
                    dict.fromkeys(selected_contract_ids)
                )
            if workflow_id is not None:
                candidate.workflow_id = workflow_id or None
            if workflow_position is not None:
                candidate.workflow_position = tuple(dict.fromkeys(workflow_position))
            if knowledge_gaps is not None:
                candidate.knowledge_gaps = {
                    stable_id("gap", task_id, gap): gap for gap in knowledge_gaps if gap.strip()
                }
                candidate.resolved_gap_ids.difference_update(candidate.knowledge_gaps)
                candidate.memory_resolved_gap_ids.difference_update(
                    candidate.knowledge_gaps
                )
                candidate.gap_assessments = {
                    gap_id: assessment
                    for gap_id, assessment in candidate.gap_assessments.items()
                    if gap_id in candidate.knowledge_gaps
                }
                candidate.gap_resolution_fragments = {
                    gap_id: fragments
                    for gap_id, fragments in candidate.gap_resolution_fragments.items()
                    if gap_id in candidate.knowledge_gaps
                }
            self._validate_selection(candidate)
            candidate.previous_packets.clear()
            self._runs[task_id] = candidate
        self._emit("memory_plan_updated", "planner", candidate)

    def update_progress(
        self,
        task_id: str,
        *,
        current_predicates: Sequence[StatePredicate | Mapping[str, Any]] | None = None,
        workflow_position: Sequence[str] | None = None,
        produced_artifacts: Sequence[str] | None = None,
        last_error: str | None = None,
        failed_contract_id: str | None = None,
    ) -> None:
        with self._lock:
            state = self._require_run(task_id)
            candidate = deepcopy(state)
            if current_predicates is not None:
                candidate.current_predicates = _predicates(current_predicates)
            if workflow_position is not None:
                candidate.workflow_position = tuple(dict.fromkeys(workflow_position))
            if produced_artifacts is not None:
                candidate.produced_artifacts = tuple(dict.fromkeys(produced_artifacts))
            candidate.last_error = last_error
            candidate.last_failed_contract_id = failed_contract_id
            self._validate_selection(candidate)
            self._runs[task_id] = candidate

    def assess_research_need(self, task: TaskContext) -> ResearchGateDecision:
        """Exhaust relevant stored knowledge before permitting external research.

        This gate does not trust a planner's claim that knowledge is missing.  Each
        open question receives a fresh, expanded structured retrieval and a scan of
        the version-compatible evidence corpus.  Covered questions are removed from
        the researcher's work and their bounded evidence is handed to the final
        planner pass.  Only genuinely uncovered or conflicted questions remain.
        """

        with self._lock:
            state = self._require_run(task.task_id)
            assessments = tuple(
                self._assess_gap(task, state, gap_id, question)
                for gap_id, question in tuple(state.knowledge_gaps.items())
            )
            for assessment in assessments:
                state.gap_assessments[assessment.gap_id] = assessment
                if assessment.resolution != GapResolution.MEMORY_RESOLVED:
                    continue
                state.knowledge_gaps.pop(assessment.gap_id, None)
                state.memory_resolved_gap_ids.add(assessment.gap_id)
            state.previous_packets.pop(MemoryRole.PLANNER, None)
            decision = ResearchGateDecision(assessments)
        self.events.emit(
            "memory_research_gate",
            "memory",
            {
                "task_id": task.task_id,
                "research_required": decision.research_required,
                "memory_resolved_count": decision.memory_resolved_count,
                "remaining_gap_ids": list(state.knowledge_gaps),
                "assessments": [
                    {
                        "gap_id": item.gap_id,
                        "resolution": item.resolution.value,
                        "reason": item.reason,
                        "matched_item_ids": list(item.matched_item_ids),
                        "matched_evidence_ids": list(item.matched_evidence_ids),
                        "corpus_records_checked": item.corpus_records_checked,
                        "retrieval_truncated": item.retrieval_truncated,
                    }
                    for item in assessments
                ],
            },
        )
        return decision

    def _assess_gap(
        self,
        task: TaskContext,
        state: MASMemoryRunState,
        gap_id: str,
        question: str,
    ) -> KnowledgeGapAssessment:
        query_tokens = _meaningful_tokens(question)
        request = KnowledgeRequest(
            software_id=state.software_id,
            query=question,
            version=state.software_version,
            phase="research-gate",
            known_files=state.produced_artifacts,
            workflow_id=state.workflow_id,
            workflow_position=state.workflow_position,
            current_state=state.current_predicates,
            max_items=50,
            token_budget=12_000,
            include_evidence=True,
            profile=RetrievalProfile.PLANNING,
        )
        packet = self.retriever.retrieve(request)
        structured_minimum_hits = (
            1 if len(query_tokens) <= 2 else max(2, (len(query_tokens) + 2) // 3)
        )
        evidence_minimum_hits = (
            1 if len(query_tokens) <= 2 else max(2, (2 * len(query_tokens) + 4) // 5)
        )
        matched_items: list[Any] = []
        matched_evidence: dict[str, EvidenceRecord] = {}

        for item in packet.items:
            hits = query_tokens.intersection(
                _meaningful_tokens(_compact_json(item.content))
            )
            if len(hits) < structured_minimum_hits:
                continue
            if item.item_type == "evidence":
                evidence = self.retriever.store.get_evidence(item.item_id)
                if evidence is not None:
                    matched_evidence[evidence.evidence_id] = evidence
                continue
            if self._item_has_evidence(item.item_type, item.item_id):
                matched_items.append(item)

        corpus = self.retriever.store.list_evidence(state.software_id)
        ranked_evidence: list[tuple[float, EvidenceRecord]] = []
        for evidence in corpus:
            if not self._evidence_matches_version(evidence, state.software_version):
                continue
            # The corpus can contain tens of thousands of documentation sections.
            # Probe only the small query vocabulary instead of tokenizing every
            # section for every gate decision.
            normalized_content = evidence.content.casefold()
            hits = {token for token in query_tokens if token in normalized_content}
            if len(hits) < evidence_minimum_hits:
                continue
            score = len(hits) / max(1, len(query_tokens))
            if evidence.authoritative:
                score += 0.25
            ranked_evidence.append((score, evidence))
        ranked_evidence.sort(key=lambda value: (-value[0], value[1].evidence_id))
        for _score, evidence in ranked_evidence[:5]:
            matched_evidence[evidence.evidence_id] = evidence

        relevant_ids = {item.item_id for item in matched_items}
        relevant_ids.update(matched_evidence)
        conflicting_ids = {
            candidate
            for conflict in self.retriever.store.list_conflicts(
                state.software_id, state=ConflictState.OPEN
            )
            for candidate in (conflict.item_id, conflict.conflicting_item_id)
            if candidate in relevant_ids
        }
        trusted_evidence = tuple(
            evidence
            for evidence in matched_evidence.values()
            if evidence.authoritative or evidence.source_kind in {
                EvidenceSourceKind.DOCUMENTATION,
                EvidenceSourceKind.SOURCE_CODE,
                EvidenceSourceKind.CONFIGURATION_SCHEMA,
            }
        )
        covered = bool(matched_items or trusted_evidence) and not conflicting_ids
        if covered:
            fragments = [
                self._item_fragment(item, MemoryRole.PLANNER, direct=True)
                for item in matched_items[:4]
            ]
            fragments.extend(
                KnowledgeFragment(
                    f"gap-memory-evidence:{evidence.evidence_id}",
                    _compact_json(
                        {
                            "gap_id": gap_id,
                            "source_uri": evidence.source_uri,
                            "locator": evidence.locator,
                            "source_version": evidence.source_version,
                            "authoritative": evidence.authoritative,
                            "excerpt": evidence.content[:1_200],
                        }
                    ),
                    "software-memory:gap-resolution",
                    1.0,
                )
                for evidence in trusted_evidence[:3]
            )
            state.gap_resolution_fragments[gap_id] = tuple(fragments)
            reason = "version-compatible stored contracts/evidence cover the question"
            resolution = GapResolution.MEMORY_RESOLVED
        else:
            reason = (
                "relevant stored knowledge is conflicted"
                if conflicting_ids
                else "no sufficiently relevant version-compatible stored knowledge"
            )
            resolution = GapResolution.RESEARCH_REQUIRED
        return KnowledgeGapAssessment(
            gap_id=gap_id,
            question=question,
            resolution=resolution,
            reason=reason,
            matched_item_ids=tuple(item.item_id for item in matched_items[:8]),
            matched_evidence_ids=tuple(
                evidence.evidence_id for evidence in trusted_evidence[:8]
            ),
            corpus_records_checked=len(corpus),
            retrieval_truncated=packet.truncated,
        )

    def _item_has_evidence(self, item_type: str, item_id: str) -> bool:
        if item_type in {"contract", "fingerprint"}:
            item = self.retriever.store.get_contract(item_id)
        elif item_type == "workflow":
            item = self.retriever.store.get_workflow(item_id)
        elif item_type == "entity":
            item = self.retriever.store.get_entity(item_id)
        else:
            return False
        return bool(item is not None and item.evidence_ids)

    @staticmethod
    def _evidence_matches_version(
        evidence: EvidenceRecord, version: str | None
    ) -> bool:
        return not version or version == "unknown" or evidence.source_version in {None, version}

    def record_research_evidence(
        self,
        task_id: str,
        *,
        actor_id: str,
        gap_id: str,
        source_uri: str,
        locator: str,
        content: str,
        source_version: str | None,
        source_kind: EvidenceSourceKind = EvidenceSourceKind.DOCUMENTATION,
        authoritative: bool = False,
    ) -> str:
        """Let a researcher add raw evidence without promoting derived knowledge."""

        if self.service is None:
            raise RuntimeError("controlled write-back requires a MemoryService")
        if not locator.strip():
            raise ValueError("research evidence requires a source locator")
        with self._lock:
            state = self._require_run(task_id)
            if gap_id not in state.knowledge_gaps:
                raise ValueError("research evidence must answer an open knowledge gap")
            if (
                state.software_version
                and state.software_version != "unknown"
                and source_version != state.software_version
            ):
                raise ValueError("research evidence version does not match the run")
            evidence = EvidenceRecord.create(
                software_id=state.software_id,
                source_kind=source_kind,
                source_uri=source_uri,
                locator=locator,
                content=content,
                source_version=source_version,
                authoritative=authoritative,
                metadata={"candidate": True, "gap_id": gap_id, "researcher": actor_id},
            )
            existing = self.retriever.store.get_evidence(evidence.evidence_id)
            reused = existing is not None
            if existing is not None:
                if (
                    existing.software_id != evidence.software_id
                    or existing.content_hash != evidence.content_hash
                    or existing.source_uri != evidence.source_uri
                    or existing.locator != evidence.locator
                    or (
                        state.software_version
                        and existing.source_version
                        and existing.source_version != state.software_version
                    )
                ):
                    raise ValueError("existing evidence ID is incompatible with the research result")
                evidence = existing
            else:
                self.service.record_evidence(evidence)
            state.evidence_candidate_ids = tuple(
                dict.fromkeys((*state.evidence_candidate_ids, evidence.evidence_id))
            )
            del state.knowledge_gaps[gap_id]
            state.resolved_gap_ids.add(gap_id)
            state.previous_packets.pop(MemoryRole.RESEARCHER, None)
            state.previous_packets.pop(MemoryRole.PLANNER, None)
        self.events.emit(
            "memory_evidence_candidate",
            actor_id,
            {
                "task_id": task_id,
                "gap_id": gap_id,
                "evidence_id": evidence.evidence_id,
                "source_uri": source_uri,
                "locator": locator,
                "source_version": source_version,
                "reused": reused,
            },
        )
        return evidence.evidence_id

    def record_verdict(
        self,
        task_id: str,
        *,
        observation_id: str,
        verification: VerificationResult,
        actor_id: str | None = None,
    ) -> str:
        """Attach an immutable external/native verdict without rewriting raw output."""

        if self.service is None:
            raise RuntimeError("controlled write-back requires a MemoryService")
        with self._lock:
            state = self._require_run(task_id)
            raw = self.retriever.store.get_observation(observation_id)
            if raw is None or raw.software_id != state.software_id or raw.task_id != task_id:
                raise ValueError("verdict references an observation outside this run")
            reviewed = self._validator_observation(
                raw, verification, actor_id or verification.check_id
            )
            self._record_observation_once(
                reviewed,
                state.software_version,
                raw.output_excerpt if not verification.passed else None,
            )
            state.open_conflict_ids = self._open_conflict_ids(state.software_id)
            state.previous_packets.pop(MemoryRole.EVALUATOR, None)
        self.events.emit(
            "memory_verdict_recorded",
            actor_id or verification.check_id,
            {
                "task_id": task_id,
                "raw_observation_id": observation_id,
                "verdict_observation_id": reviewed.observation_id,
                "check_id": verification.check_id,
                "status": verification.status.value,
            },
        )
        return reviewed.observation_id

    def context_for(
        self, task: TaskContext, agent: AgentSpec, runtime: RuntimeState
    ) -> KnowledgeContext:
        with self._lock:
            if task.task_id not in self._runs:
                self.begin_run(task)
            state = self._require_run(task.task_id)
            self._sync_runtime_artifacts(state, runtime)
            role = self._role_for(agent, runtime)
            policy = self.policies[role]
            fragments = self._context_fragments(task, role, state, policy)
            context = KnowledgeContext(tuple(fragments), policy.max_characters)
            rendered_characters = len(context.render())
            self.events.emit(
                "memory_response",
                agent.id,
                {
                    "task_id": task.task_id,
                    "role": role.value,
                    "fragment_ids": [fragment.id for fragment in fragments],
                    "fragment_count": len(fragments),
                    "characters": rendered_characters,
                    "candidate_characters": sum(len(fragment.content) for fragment in fragments),
                    "selected_contract_ids": list(state.selected_contract_ids),
                    "workflow_id": state.workflow_id,
                    "open_gap_ids": list(state.knowledge_gaps),
                },
            )
            return context

    def observe_action(
        self,
        task: TaskContext,
        agent: AgentSpec,
        runtime: RuntimeState,
        action: Action,
        result: ToolResult,
        verification: VerificationResult | None,
    ) -> None:
        """Write raw agent output and an independent verdict as distinct records."""

        if self.service is None or action.tool.startswith("memory_"):
            return
        with self._lock:
            state = self._require_run(task.task_id)
            contract_id = self._contract_for_action(task, state, action)
            if contract_id is None:
                self.events.emit(
                    "memory_observation_skipped",
                    agent.id,
                    {
                        "task_id": task.task_id,
                        "action": action.id,
                        "tool": action.tool,
                        "reason": "no_unique_software_contract",
                    },
                )
                return
            before = state.current_predicates
            after = self._state_after(result.output) or before
            artifacts = tuple(
                dict.fromkeys((*state.produced_artifacts, *action.artifact_outputs))
            )
            error = (
                result.error
                or self._output_error(result.output)
                or (
                    verification.summary
                    if verification is not None
                    and verification.status == VerificationStatus.FAILED
                    else None
                )
            )
            state.current_predicates = after
            state.produced_artifacts = artifacts
            state.last_error = error
            state.last_failed_contract_id = contract_id if not result.ok else None
            raw = ExecutionObservation(
                observation_id=stable_id("observation", task.task_id, action.id, "agent"),
                software_id=state.software_id,
                contract_id=contract_id,
                actor_id=agent.id,
                actor_kind=ActorKind.AGENT,
                task_id=task.task_id,
                inputs=dict(action.arguments),
                state_before=self._state_before(result.output, before),
                state_after=after,
                output_excerpt=self._output_excerpt(result),
                exit_code=self._exit_code(result),
                succeeded=result.ok and (
                    verification is None or verification.status != VerificationStatus.FAILED
                ),
            )
            self._record_observation_once(raw, state.software_version, error)
            observation_ids = [raw.observation_id]
            if verification is not None:
                reviewed = self._validator_observation(raw, verification, verification.check_id)
                self._record_observation_once(reviewed, state.software_version, error)
                observation_ids.append(reviewed.observation_id)
            state.open_conflict_ids = self._open_conflict_ids(state.software_id)
            state.previous_packets.pop(MemoryRole.EVALUATOR, None)
            if error:
                state.previous_packets.pop(MemoryRole.EXECUTOR, None)
        self.events.emit(
            "memory_observation_recorded",
            agent.id,
            {
                "task_id": task.task_id,
                "action": action.id,
                "contract_id": contract_id,
                "observation_ids": observation_ids,
                "verified": verification is not None,
            },
        )

    def _context_fragments(
        self,
        task: TaskContext,
        role: MemoryRole,
        state: MASMemoryRunState,
        policy: RoleMemoryPolicy,
    ) -> list[KnowledgeFragment]:
        query = task.objective
        profile = RetrievalProfile.GENERIC
        include_evidence = policy.include_evidence
        if role == MemoryRole.PLANNER:
            profile = RetrievalProfile.PLANNING
        elif role == MemoryRole.RESEARCHER:
            conflict_query = " ".join(
                conflict.reason
                for conflict in self.retriever.store.list_conflicts(
                    state.software_id, state=ConflictState.OPEN
                )
            )
            query = " ".join(state.knowledge_gaps.values()) or conflict_query
            if not query:
                return []
            profile = RetrievalProfile.PLANNING
        elif role == MemoryRole.EXECUTOR:
            profile = RetrievalProfile.REPAIR if state.last_error else RetrievalProfile.EXECUTION

        request = KnowledgeRequest(
            software_id=state.software_id,
            query=query,
            version=state.software_version,
            phase=role.value,
            known_files=state.produced_artifacts,
            workflow_id=state.workflow_id,
            workflow_position=state.workflow_position,
            current_state=state.current_predicates,
            error_signature=state.last_error,
            max_items=policy.max_items,
            token_budget=policy.token_budget,
            include_evidence=include_evidence,
            profile=profile,
        )
        self.events.emit(
            "memory_request",
            role.value,
            {
                "task_id": task.task_id,
                "software_id": state.software_id,
                "version": state.software_version,
                "profile": profile.value,
                "token_budget": policy.token_budget,
                "query_digest": _digest(query),
                "known_file_count": len(state.produced_artifacts),
                "current_predicate_count": len(state.current_predicates),
            },
        )
        previous = state.previous_packets.get(role)
        packet = (
            self.retriever.retrieve_delta(request, previous)
            if previous is not None
            else self.retriever.retrieve(request)
        )
        state.previous_packets[role] = packet
        fragments = [self._item_fragment(item, role) for item in packet.items]

        frontier = None
        if role == MemoryRole.PLANNER and state.desired_predicates:
            frontier = self.retriever.explain_causal_frontier(
                software_id=state.software_id,
                desired_state=state.desired_predicates,
                current_state=state.current_predicates,
                version=state.software_version,
                max_candidates=policy.max_items,
            )
        if role == MemoryRole.EXECUTOR and state.desired_predicates:
            frontier = self.retriever.explain_causal_frontier(
                software_id=state.software_id,
                desired_state=state.desired_predicates,
                current_state=state.current_predicates,
                version=state.software_version,
                max_candidates=policy.max_items,
            )
            fragments.append(
                KnowledgeFragment(
                    "current-causal-frontier",
                    _compact_json(
                        {
                            "candidates": [
                                {
                                    "contract_id": item.contract_id,
                                    "executable_now": item.executable_now,
                                    "missing_preconditions": [
                                        value.model_dump(mode="json")
                                        for value in item.missing_preconditions
                                    ],
                                    "causal_path": item.causal_path,
                                }
                                for item in frontier.candidates
                            ],
                            "unresolved": [
                                item.model_dump(mode="json")
                                for item in frontier.unresolved_predicates
                            ],
                        }
                    ),
                    "software-memory:current-frontier",
                    1.0,
                )
            )
        if role == MemoryRole.PLANNER:
            fragments.extend(self._planner_handoff_fragments(state))
        if role == MemoryRole.PLANNER and frontier is not None:
            fragments.append(
                KnowledgeFragment(
                    "causal-frontier",
                    _compact_json(
                        {
                            "candidates": [
                                item.model_dump(mode="json")
                                for item in frontier.candidates
                            ],
                            "unresolved": [
                                item.model_dump(mode="json")
                                for item in frontier.unresolved_predicates
                            ],
                            "conflicting": [
                                item.model_dump(mode="json")
                                for item in frontier.conflicting_predicates
                            ],
                        }
                    ),
                    "software-memory:causal-frontier",
                    1.0,
                )
            )
        if role == MemoryRole.RESEARCHER:
            fragments.extend(self._research_fragments(state))
        if role in {MemoryRole.EXECUTOR, MemoryRole.EVALUATOR}:
            fragments.extend(self._selected_contract_fragments(role, state))
        if role == MemoryRole.EXECUTOR and state.last_error:
            repair = self.retriever.retrieve_repair(
                RepairKnowledgeRequest(
                    software_id=state.software_id,
                    error_text=state.last_error,
                    version=state.software_version,
                    failed_contract_id=state.last_failed_contract_id,
                    workflow_id=state.workflow_id,
                    current_state=state.current_predicates,
                    known_artifacts=state.produced_artifacts,
                    max_items=policy.max_items,
                    token_budget=policy.token_budget,
                )
            )
            fragments.extend(
                KnowledgeFragment(
                    f"repair:{item.item_type}:{item.item_id}",
                    _compact_json(item.content),
                    "software-memory:repair",
                    self._relevance(item.score),
                )
                for item in repair.items
            )
        if role == MemoryRole.EVALUATOR:
            fragments.extend(self._evaluation_fragments(state, policy.max_items))
        return self._deduplicate(fragments)

    def _selected_contract_fragments(
        self, role: MemoryRole, state: MASMemoryRunState
    ) -> list[KnowledgeFragment]:
        delivered = state.delivered_direct_items.setdefault(role, set())
        fragments: list[KnowledgeFragment] = []
        for contract_id in state.selected_contract_ids:
            packet = self.retriever.get_contract(
                contract_id, include_evidence=role == MemoryRole.EVALUATOR
            )
            for item in packet.items:
                key = f"{item.item_type}:{item.item_id}:{_digest(item.content)}"
                if key in delivered:
                    continue
                delivered.add(key)
                fragments.append(self._item_fragment(item, role, direct=True))
        return fragments

    def _research_fragments(self, state: MASMemoryRunState) -> list[KnowledgeFragment]:
        fragments = [
            KnowledgeFragment(
                f"knowledge-gap:{gap_id}",
                question,
                "mas-working-state",
                1.0,
            )
            for gap_id, question in state.knowledge_gaps.items()
        ]
        for conflict in self.retriever.store.list_conflicts(
            state.software_id, state=ConflictState.OPEN
        ):
            fragments.append(
                KnowledgeFragment(
                    f"conflict:{conflict.conflict_id}",
                    _compact_json(
                        {
                            "item_id": conflict.item_id,
                            "conflicting_item_id": conflict.conflicting_item_id,
                            "reason": conflict.reason,
                            "evidence_ids": conflict.evidence_ids,
                        }
                    ),
                    "software-memory:conflict",
                    1.0,
                )
            )
        return fragments

    def _planner_handoff_fragments(
        self, state: MASMemoryRunState
    ) -> list[KnowledgeFragment]:
        """Return bounded research deltas to the second planner pass."""

        delivered = state.delivered_direct_items.setdefault(MemoryRole.PLANNER, set())
        fragments: list[KnowledgeFragment] = []
        for gap_id, candidates in state.gap_resolution_fragments.items():
            for fragment in candidates:
                key = f"gap-resolution:{gap_id}:{fragment.id}:{_digest(fragment.content)}"
                if key in delivered:
                    continue
                delivered.add(key)
                fragments.append(fragment)
        for evidence_id in state.evidence_candidate_ids:
            evidence = self.retriever.store.get_evidence(evidence_id)
            if evidence is None:
                continue
            key = f"research-evidence:{evidence_id}:{evidence.content_hash}"
            if key in delivered:
                continue
            delivered.add(key)
            fragments.append(
                KnowledgeFragment(
                    f"research-evidence:{evidence_id}",
                    _compact_json(
                        {
                            "source_uri": evidence.source_uri,
                            "locator": evidence.locator,
                            "source_version": evidence.source_version,
                            "authoritative": evidence.authoritative,
                            "excerpt": evidence.content[:1_200],
                        }
                    ),
                    "software-memory:research-delta",
                    1.0,
                )
            )
        return fragments

    def _evaluation_fragments(
        self, state: MASMemoryRunState, limit: int
    ) -> list[KnowledgeFragment]:
        selected = set(state.selected_contract_ids)
        episodes = [
            episode
            for episode in self.retriever.store.list_episodes(state.software_id)
            if not selected or episode.action_contract_id in selected
        ][-limit:]
        return [
            KnowledgeFragment(
                f"episode:{episode.episode_id}",
                _compact_json(episode.model_dump(mode="json", exclude_none=True)),
                "software-memory:episode",
                0.72,
                (episode.action_contract_id,) if episode.action_contract_id else (),
            )
            for episode in episodes
        ]

    @staticmethod
    def _item_fragment(item: Any, role: MemoryRole, *, direct: bool = False) -> KnowledgeFragment:
        if direct:
            return SharedSoftwareMemory._direct_item_fragment(item, role)
        dependencies = ()
        if item.item_type in {"contract", "fingerprint"}:
            dependencies = tuple(str(value) for value in item.content.get("preconditions", ()))
        relevance = SharedSoftwareMemory._relevance(item.score)
        if role == MemoryRole.PLANNER:
            relevance = 0.96 if item.item_type == "workflow" else 0.88 if item.item_type == "fingerprint" else min(relevance, 0.70)
        elif role == MemoryRole.RESEARCHER:
            relevance = 0.86 if item.item_type in {"fingerprint", "workflow"} else min(relevance, 0.68)
        elif role == MemoryRole.EXECUTOR:
            relevance = 0.82 if item.item_type in {"fingerprint", "workflow"} else min(relevance, 0.64)
        elif role == MemoryRole.EVALUATOR:
            relevance = 0.65 if item.item_type in {"fingerprint", "workflow"} else min(relevance, 0.55)
        return KnowledgeFragment(
            f"{item.item_type}:{item.item_id}",
            _compact_json(item.content),
            f"software-memory:{role.value}{':direct' if direct else ''}",
            relevance,
            dependencies,
        )

    @staticmethod
    def _direct_item_fragment(item: Any, role: MemoryRole) -> KnowledgeFragment:
        content = dict(item.content)
        if item.item_type == "contract":
            fields = (
                ("contract_id", "name", "interface", "purpose", "parameters", "inputs",
                 "outputs", "preconditions", "effects", "side_effects", "success_signals",
                 "failure_signatures", "repair_strategies", "verification_operations",
                 "risk_level", "version_scope")
                if role == MemoryRole.EXECUTOR
                else
                ("contract_id", "name", "interface", "effects", "success_signals",
                 "verification_operations", "evidence_ids", "risk_level", "version_scope",
                 "status", "verification")
            )
            content = {key: content[key] for key in fields if key in content}
            relevance = 1.0
        elif item.item_type == "evidence":
            raw = str(content.get("content", ""))
            content = {
                key: content[key]
                for key in ("evidence_id", "source_kind", "source_uri", "locator",
                            "source_version", "authoritative", "content_hash")
                if key in content
            }
            content["excerpt"] = raw[:900]
            relevance = 0.97
        else:
            relevance = 0.95
        return KnowledgeFragment(
            f"{item.item_type}:{item.item_id}",
            _compact_json(content),
            f"software-memory:{role.value}:direct",
            relevance,
        )

    @staticmethod
    def _relevance(score: float) -> float:
        return max(0.0, min(1.0, 0.5 + score / 24.0))

    @staticmethod
    def _deduplicate(fragments: Sequence[KnowledgeFragment]) -> list[KnowledgeFragment]:
        result: list[KnowledgeFragment] = []
        seen: set[str] = set()
        for fragment in fragments:
            key = f"{fragment.id}:{_digest(fragment.content)}"
            if key not in seen:
                seen.add(key)
                result.append(fragment)
        return result

    def _role_for(self, agent: AgentSpec, runtime: RuntimeState) -> MemoryRole:
        for candidate in (runtime.current_node, agent.id):
            resolved = self.role_from_label(candidate)
            if resolved is not None:
                return resolved
        return MemoryRole.EXECUTOR

    def role_from_label(self, label: str) -> MemoryRole | None:
        normalized = label.casefold()
        if normalized in self.role_aliases:
            return self.role_aliases[normalized]
        if "research" in normalized:
            return MemoryRole.RESEARCHER
        if "plan" in normalized:
            return MemoryRole.PLANNER
        if any(value in normalized for value in ("verify", "evaluat", "critic")):
            return MemoryRole.EVALUATOR
        if any(value in normalized for value in ("execut", "repair", "implement")):
            return MemoryRole.EXECUTOR
        return None

    def _sync_runtime_artifacts(
        self, state: MASMemoryRunState, runtime: RuntimeState
    ) -> None:
        values = runtime.artifacts
        if "selected_contract_ids" in values:
            selected = _strings(values["selected_contract_ids"])
            if selected != state.selected_contract_ids:
                candidate = deepcopy(state)
                candidate.selected_contract_ids = selected
                self._validate_selection(candidate)
                state.selected_contract_ids = selected
        if "workflow_id" in values and values["workflow_id"]:
            state.workflow_id = str(values["workflow_id"])
        if "workflow_position" in values:
            state.workflow_position = _strings(values["workflow_position"])
        if "current_state" in values:
            state.current_predicates = _predicates(values["current_state"])
        if "knowledge_gaps" in values:
            incoming = {
                stable_id("gap", state.task_id, gap): gap
                for gap in _strings(values["knowledge_gaps"])
            }
            state.knowledge_gaps = {
                gap_id: gap
                for gap_id, gap in incoming.items()
                if gap_id not in state.resolved_gap_ids
            }
        reserved = {
            "selected_contract_ids", "workflow_id", "workflow_position",
            "current_state", "knowledge_gaps",
        }
        artifact_ids = tuple(key for key in values if key not in reserved)
        state.produced_artifacts = tuple(
            dict.fromkeys((*state.produced_artifacts, *artifact_ids))
        )

    def _validate_selection(self, state: MASMemoryRunState) -> None:
        if state.workflow_position and not state.workflow_id:
            raise ValueError("workflow_position requires workflow_id")
        for contract_id in state.selected_contract_ids:
            contract = self.retriever.store.get_contract(contract_id)
            if contract is None or contract.software_id != state.software_id:
                raise ValueError(
                    f"selected contract is unavailable for this software: {contract_id}"
                )
        if state.workflow_id:
            workflow = self.retriever.store.get_workflow(state.workflow_id)
            if workflow is None or workflow.software_id != state.software_id:
                raise ValueError("selected workflow is unavailable for this software")

    def _contract_for_action(
        self, task: TaskContext, state: MASMemoryRunState, action: Action
    ) -> str | None:
        explicit = action.arguments.get("software_memory_contract_id")
        if isinstance(explicit, str):
            return explicit
        mapping = task.metadata.get("tool_contract_map", {})
        if isinstance(mapping, Mapping) and isinstance(mapping.get(action.tool), str):
            return str(mapping[action.tool])
        matches = []
        for contract_id in state.selected_contract_ids:
            contract = self.retriever.store.get_contract(contract_id)
            aliases = contract.metadata.get("aliases", ()) if contract else ()
            if contract and (contract.name == action.tool or action.tool in aliases):
                matches.append(contract_id)
        return matches[0] if len(matches) == 1 else None

    def _record_observation_once(
        self,
        observation: ExecutionObservation,
        software_version: str | None,
        failure_signature: str | None,
    ) -> None:
        assert self.service is not None
        if self.retriever.store.get_observation(observation.observation_id) is not None:
            return
        self.service.record_observation(observation)
        self.service.record_episode(
            compact_episode(
                observation,
                software_version=software_version,
                failure_signature=failure_signature,
            )
        )

    @staticmethod
    def _validator_observation(
        raw: ExecutionObservation,
        verification: VerificationResult,
        actor_id: str,
    ) -> ExecutionObservation:
        verdict = {
            VerificationStatus.PASSED: VerificationState.ACCEPTED,
            VerificationStatus.FAILED: VerificationState.REJECTED,
            VerificationStatus.INCONCLUSIVE: VerificationState.INCONCLUSIVE,
        }[verification.status]
        return raw.model_copy(
            update={
                "observation_id": stable_id(
                    "observation",
                    raw.observation_id,
                    verification.check_id,
                    actor_id,
                ),
                "actor_id": actor_id,
                "actor_kind": ActorKind.VALIDATOR,
                "succeeded": verification.passed,
                "verifier_id": verification.check_id,
                "verification": verdict,
            }
        )

    @staticmethod
    def _state_after(output: Any) -> tuple[StatePredicate, ...]:
        return _predicates(output.get("state_after")) if isinstance(output, Mapping) else ()

    @staticmethod
    def _state_before(
        output: Any, fallback: tuple[StatePredicate, ...]
    ) -> tuple[StatePredicate, ...]:
        parsed = _predicates(output.get("state_before")) if isinstance(output, Mapping) else ()
        return parsed or fallback

    @staticmethod
    def _output_error(output: Any) -> str | None:
        if not isinstance(output, Mapping):
            return None
        for key in ("error", "stderr"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value[:4_000]
        exit_code = output.get("exit_code")
        return f"exit_code={exit_code}" if isinstance(exit_code, int) and exit_code != 0 else None

    @staticmethod
    def _output_excerpt(result: ToolResult) -> str:
        value = result.error if result.error is not None else result.output
        return " ".join(_compact_json(value).split())[:2_000]

    @staticmethod
    def _exit_code(result: ToolResult) -> int | None:
        if isinstance(result.output, Mapping) and isinstance(result.output.get("exit_code"), int):
            return int(result.output["exit_code"])
        return None

    def _open_conflict_ids(self, software_id: str) -> tuple[str, ...]:
        return tuple(
            item.conflict_id
            for item in self.retriever.store.list_conflicts(
                software_id, state=ConflictState.OPEN
            )
        )

    def _require_run(self, task_id: str) -> MASMemoryRunState:
        try:
            return self._runs[task_id]
        except KeyError as error:
            raise ValueError(f"unknown memory run: {task_id}") from error

    def _emit(self, event_type: str, actor: str, state: MASMemoryRunState) -> None:
        self.events.emit(
            event_type,
            actor,
            {
                "task_id": state.task_id,
                "software_id": state.software_id,
                "software_version": state.software_version,
                "selected_contract_ids": list(state.selected_contract_ids),
                "workflow_id": state.workflow_id,
                "workflow_position": list(state.workflow_position),
                "produced_artifacts": list(state.produced_artifacts),
                "open_gap_ids": list(state.knowledge_gaps),
                "open_conflict_ids": list(state.open_conflict_ids),
            },
        )


def memory_coordination_tools(memory: SharedSoftwareMemory) -> tuple[AgentCallableTool, ...]:
    """Build least-privilege structured handoff tools for MAS roles."""

    def require_role(agent: AgentSpec, expected: MemoryRole) -> None:
        if memory.role_from_label(agent.id) != expected:
            raise PermissionError(f"{agent.id} cannot perform {expected.value} memory writes")

    def update_plan(agent: AgentSpec, arguments: Mapping[str, Any], task: TaskContext) -> Any:
        require_role(agent, MemoryRole.PLANNER)
        memory.update_plan(
            task.task_id,
            selected_contract_ids=(
                _strings(arguments["selected_contract_ids"])
                if "selected_contract_ids" in arguments
                else None
            ),
            workflow_id=(
                str(arguments["workflow_id"]) if "workflow_id" in arguments else None
            ),
            workflow_position=(
                _strings(arguments["workflow_position"])
                if "workflow_position" in arguments
                else None
            ),
            knowledge_gaps=(
                _strings(arguments["knowledge_gaps"])
                if "knowledge_gaps" in arguments
                else None
            ),
        )
        state = memory.snapshot(task.task_id)
        return {
            "selected_contract_ids": state.selected_contract_ids,
            "workflow_id": state.workflow_id,
            "workflow_position": state.workflow_position,
            "open_gap_ids": tuple(state.knowledge_gaps),
        }

    def record_evidence(
        agent: AgentSpec, arguments: Mapping[str, Any], task: TaskContext
    ) -> Any:
        require_role(agent, MemoryRole.RESEARCHER)
        evidence_id = memory.record_research_evidence(
            task.task_id,
            actor_id=agent.id,
            gap_id=str(arguments["gap_id"]),
            source_uri=str(arguments["source_uri"]),
            locator=str(arguments["locator"]),
            content=str(arguments["content"]),
            source_version=str(arguments["source_version"]),
            source_kind=EvidenceSourceKind(str(arguments.get("source_kind", "documentation"))),
            authoritative=bool(arguments.get("authoritative", False)),
        )
        return {"evidence_id": evidence_id, "status": "candidate"}

    def update_progress(
        agent: AgentSpec, arguments: Mapping[str, Any], task: TaskContext
    ) -> Any:
        require_role(agent, MemoryRole.EXECUTOR)
        memory.update_progress(
            task.task_id,
            current_predicates=(
                arguments["current_predicates"]
                if "current_predicates" in arguments
                else None
            ),
            workflow_position=(
                _strings(arguments["workflow_position"])
                if "workflow_position" in arguments
                else None
            ),
            produced_artifacts=(
                _strings(arguments["produced_artifacts"])
                if "produced_artifacts" in arguments
                else None
            ),
            last_error=(
                str(arguments["last_error"]) if arguments.get("last_error") else None
            ),
            failed_contract_id=(
                str(arguments["failed_contract_id"])
                if arguments.get("failed_contract_id")
                else None
            ),
        )
        state = memory.snapshot(task.task_id)
        return {
            "current_predicate_ids": tuple(item.key for item in state.current_predicates),
            "workflow_position": state.workflow_position,
            "produced_artifacts": state.produced_artifacts,
        }

    string_array = {"type": "array", "items": {"type": "string"}}
    predicate_array = {"type": "array", "items": {"type": "object"}}
    return (
        AgentCallableTool(
            "memory_update_plan",
            "Store a compact plan handoff as contract/workflow IDs and open questions.",
            update_plan,
            input_schema={
                "type": "object",
                "properties": {
                    "selected_contract_ids": string_array,
                    "workflow_id": {"type": "string"},
                    "workflow_position": string_array,
                    "knowledge_gaps": string_array,
                },
                "additionalProperties": False,
            },
        ),
        AgentCallableTool(
            "memory_record_evidence",
            "Record sourced evidence for one open gap as an unpromoted candidate.",
            record_evidence,
            input_schema={
                "type": "object",
                "properties": {
                    "gap_id": {"type": "string", "minLength": 1},
                    "source_uri": {"type": "string", "minLength": 1},
                    "locator": {"type": "string", "minLength": 1},
                    "content": {"type": "string", "minLength": 1},
                    "source_version": {"type": "string", "minLength": 1},
                    "source_kind": {
                        "type": "string",
                        "enum": [item.value for item in EvidenceSourceKind],
                    },
                    "authoritative": {"type": "boolean"},
                },
                "required": [
                    "gap_id", "source_uri", "locator", "content", "source_version"
                ],
                "additionalProperties": False,
            },
        ),
        AgentCallableTool(
            "memory_update_progress",
            "Store achieved state, workflow position, artifacts, or the latest failure.",
            update_progress,
            input_schema={
                "type": "object",
                "properties": {
                    "current_predicates": predicate_array,
                    "workflow_position": string_array,
                    "produced_artifacts": string_array,
                    "last_error": {"type": "string"},
                    "failed_contract_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        ),
    )


__all__ = [
    "DEFAULT_ROLE_POLICIES",
    "GapResolution",
    "KnowledgeGapAssessment",
    "MASMemoryRunState",
    "MemoryRole",
    "ResearchGateDecision",
    "RoleMemoryPolicy",
    "SharedSoftwareMemory",
    "memory_coordination_tools",
]
