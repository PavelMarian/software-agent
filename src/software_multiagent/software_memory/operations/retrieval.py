from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Protocol, runtime_checkable

from software_multiagent.software_memory.errors import KnowledgeNotFoundError
from software_multiagent.software_memory.schema.models import (
    ConflictState,
    CausalFrontierResult,
    ContextMode,
    Entity,
    EvidenceRecord,
    FrontierCandidate,
    KnowledgeItem,
    KnowledgePacket,
    KnowledgeRequest,
    KnowledgeStatus,
    OperationContract,
    OperationFingerprint,
    RelatedEntity,
    RepairKnowledgePacket,
    RepairKnowledgeRequest,
    RetrievalProfile,
    StatePredicate,
    Workflow,
)
from software_multiagent.software_memory.operations.procedural import predicate_slot, predicates_conflict
from software_multiagent.software_memory.persistence.storage import SQLiteMemoryStore
from software_multiagent.software_memory.schema.acceptance import AcceptanceContract, IntentLock
from software_multiagent.software_memory.schema.verification import VerificationMethodCandidate, VerificationPlan
from software_multiagent.software_memory.schema.outcomes import VerificationSignal


_STATUS_WEIGHT = {
    KnowledgeStatus.TASK_VERIFIED: 7.0,
    KnowledgeStatus.EXECUTION_VERIFIED: 6.0,
    KnowledgeStatus.SOURCE_CODE_CONFIRMED: 5.0,
    KnowledgeStatus.CROSS_SOURCE_CONFIRMED: 4.0,
    KnowledgeStatus.EVIDENCE_SUPPORTED: 3.0,
    KnowledgeStatus.EXTRACTED: 1.0,
    KnowledgeStatus.VERSION_UNCONFIRMED: 0.25,
    KnowledgeStatus.CONFLICTING: -3.0,
    KnowledgeStatus.DEPRECATED: -5.0,
    KnowledgeStatus.RETRACTED: -100.0,
}


@runtime_checkable
class EmbeddingRetriever(Protocol):
    """Optional adapter implemented by LlamaIndex, Haystack, or another backend."""

    def search_memory(
        self,
        *,
        software_id: str,
        query: str,
        limit: int,
    ) -> tuple["EmbeddingHit", ...]: ...


@dataclass(frozen=True)
class EmbeddingHit:
    """Provider-neutral semantic hit; canonical content still comes from SQLite."""

    score: float
    item_type: str
    item_id: str


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _estimated_tokens(value: Any) -> int:
    return max(1, (len(_canonical(value)) + 3) // 4)


def _content_characters(items: Iterable[KnowledgeItem]) -> int:
    return sum(len(_canonical(item.content)) for item in items)


def _item_key(item: KnowledgeItem) -> str:
    return f"{item.item_type}:{item.item_id}"


def _item_hash(item: KnowledgeItem) -> str:
    payload = {"item_id": item.item_id, "item_type": item.item_type, "content": item.content}
    return sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _packet_digest(item_hashes: dict[str, str]) -> str:
    return sha256(_canonical(item_hashes).encode("utf-8")).hexdigest()


def serialize_packet(packet: KnowledgePacket) -> str:
    """Return a byte-stable JSON representation suitable for traces and caching."""
    return _canonical(packet.model_dump(mode="json", exclude_none=True))


def _predicate_keys(values: Iterable[StatePredicate]) -> set[str]:
    return {value.key for value in values}


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _contract_aliases(contract: OperationContract) -> tuple[str, ...]:
    aliases = contract.metadata.get("aliases", ())
    if isinstance(aliases, str):
        return (aliases,)
    if isinstance(aliases, (list, tuple)):
        return tuple(str(value) for value in aliases)
    return ()


def _contract_invocation(contract: OperationContract) -> str | None:
    value = contract.metadata.get("invocation")
    return str(value) if value else None


@dataclass
class MemoryRetriever:
    store: SQLiteMemoryStore
    embedding_retriever: EmbeddingRetriever | None = None

    def get_intent_lock(self, lock_id: str) -> IntentLock:
        intent = self.store.get_intent_lock(lock_id)
        if intent is None:
            raise KnowledgeNotFoundError("unknown intent lock", lock_id=lock_id)
        return intent

    def get_acceptance_contract(self, contract_id: str) -> AcceptanceContract:
        contract = self.store.get_acceptance_contract(contract_id)
        if contract is None:
            raise KnowledgeNotFoundError(
                "unknown acceptance contract", acceptance_contract_id=contract_id
            )
        return contract

    def list_acceptance_contracts(self, task_id: str) -> tuple[AcceptanceContract, ...]:
        return self.store.list_acceptance_contracts(task_id)

    def get_verification_candidate(self, candidate_id: str) -> VerificationMethodCandidate:
        candidate = self.store.get_verification_candidate(candidate_id)
        if candidate is None:
            raise KnowledgeNotFoundError(
                "unknown verification method", candidate_id=candidate_id
            )
        return candidate

    def get_verification_plan(self, plan_id: str) -> VerificationPlan:
        plan = self.store.get_verification_plan(plan_id)
        if plan is None:
            raise KnowledgeNotFoundError("unknown verification plan", plan_id=plan_id)
        return plan

    def get_verification_signal(self, signal_id: str) -> VerificationSignal:
        signal = self.store.get_verification_signal(signal_id)
        if signal is None:
            raise KnowledgeNotFoundError("unknown verification signal", signal_id=signal_id)
        return signal

    def list_capabilities(self, software_id: str) -> tuple[OperationFingerprint, ...]:
        self._require_software(software_id)
        return tuple(
            OperationFingerprint.from_contract(contract)
            for contract in self.store.list_contracts(software_id)
            if contract.status not in {KnowledgeStatus.RETRACTED, KnowledgeStatus.DEPRECATED}
        )

    def get_entity(self, entity_id: str) -> Entity:
        entity = self.store.get_entity(entity_id)
        if entity is None:
            raise KnowledgeNotFoundError("unknown entity", entity_id=entity_id)
        return entity

    def get_related_entities(self, software_id: str, entity_id: str) -> tuple[RelatedEntity, ...]:
        entity = self.get_entity(entity_id)
        if entity.software_id != software_id:
            raise KnowledgeNotFoundError(
                "entity does not belong to requested software",
                entity_id=entity_id,
                software_id=software_id,
            )
        results: list[RelatedEntity] = []
        for relation in self.store.related(software_id, (entity_id,)):
            outgoing = relation.source_entity_id == entity_id
            related_id = relation.target_entity_id if outgoing else relation.source_entity_id
            related = self.store.get_entity(related_id)
            if related is not None:
                results.append(
                    RelatedEntity(
                        relation=relation,
                        entity=related,
                        direction="outgoing" if outgoing else "incoming",
                    )
                )
        return tuple(results)

    def get_workflow(self, workflow_id: str) -> Workflow:
        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise KnowledgeNotFoundError("unknown workflow", workflow_id=workflow_id)
        return workflow

    def get_evidence(self, evidence_id: str) -> EvidenceRecord:
        evidence = self.store.get_evidence(evidence_id)
        if evidence is None:
            raise KnowledgeNotFoundError("unknown evidence", evidence_id=evidence_id)
        return evidence

    def retrieve_repair(self, request: RepairKnowledgeRequest) -> RepairKnowledgePacket:
        from software_multiagent.software_memory.operations.repair import RepairRetriever

        return RepairRetriever(self.store).retrieve(request)

    def retrieve(self, request: KnowledgeRequest) -> KnowledgePacket:
        self._require_software(request.software_id)
        factors = self._request_factors(request)
        trace = [f"request profile={request.profile.value} factors={','.join(factors) or 'none'}"]
        if request.mode == ContextMode.NONE:
            trace.append("memory disabled")
            return self._packet(request, (), trace=trace)

        contracts = self.store.list_contracts(request.software_id)
        contract_candidates: dict[str, tuple[float, OperationContract]] = {}
        entity_candidates: dict[str, tuple[float, Entity]] = {}
        semantic_contract_ids: set[str] = set()
        semantic_entity_ids: set[str] = set()
        semantic_workflows: dict[str, tuple[float, Workflow]] = {}
        semantic_evidence: dict[str, tuple[float, EvidenceRecord]] = {}
        entities_expanded = False
        query = " ".join(
            value
            for value in (
                request.query,
                request.desired_capability or "",
                request.error_signature or "",
                *request.known_files,
                *request.known_commands,
            )
            if value
        )

        if request.mode == ContextMode.FULL_DUMP:
            for contract in contracts:
                contract_candidates[contract.contract_id] = (0.0, contract)
            trace.append("full-dump ablation selected")
        else:
            self._add_exact_candidates(
                request,
                contracts,
                contract_candidates,
                entity_candidates,
            )
            lexical = self.store.search_contracts(
                request.software_id,
                query,
                max(request.max_items * 4, request.max_items),
            )
            for score, contract in lexical:
                self._add_contract(contract_candidates, contract, score)
            if query:
                for score, entity in self.store.search_entities(
                    request.software_id,
                    query,
                    max(request.max_items * 2, request.max_items),
                    request.entity_types,
                ):
                    self._add_entity(entity_candidates, entity, score)
                    for contract in contracts:
                        if entity.entity_id in contract.related_entity_ids:
                            self._add_contract(contract_candidates, contract, score + 1.0)
            self._expand_entities_one_hop(
                request,
                contract_candidates,
                entity_candidates,
                trace,
            )
            entities_expanded = True
            base_entity_ids = set(entity_candidates)
            if self.embedding_retriever is not None and query:
                semantic_hits = self.embedding_retriever.search_memory(
                    software_id=request.software_id,
                    query=query,
                    limit=max(request.max_items * 4, request.max_items),
                )
                for hit in semantic_hits:
                    if hit.item_type == "contract":
                        contract = self.store.get_contract(hit.item_id)
                        if (
                            contract is not None
                            and contract.software_id == request.software_id
                            and contract.contract_id not in contract_candidates
                        ):
                            self._add_contract(contract_candidates, contract, hit.score)
                            semantic_contract_ids.add(contract.contract_id)
                    elif hit.item_type == "entity":
                        entity = self.store.get_entity(hit.item_id)
                        if (
                            entity is not None
                            and entity.software_id == request.software_id
                            and entity.entity_id not in entity_candidates
                        ):
                            self._add_entity(entity_candidates, entity, hit.score)
                            semantic_entity_ids.add(entity.entity_id)
                    elif hit.item_type == "workflow":
                        workflow = self.store.get_workflow(hit.item_id)
                        if workflow is not None and workflow.software_id == request.software_id:
                            semantic_workflows[workflow.workflow_id] = (hit.score, workflow)
                    elif hit.item_type == "evidence" and request.include_evidence:
                        evidence = self.store.get_evidence(hit.item_id)
                        if evidence is not None and evidence.software_id == request.software_id:
                            semantic_evidence[evidence.evidence_id] = (hit.score, evidence)
                self._expand_entities_one_hop(
                    request,
                    contract_candidates,
                    entity_candidates,
                    trace,
                )
                semantic_entity_ids.update(set(entity_candidates) - base_entity_ids)
                trace.append(
                    "embedding adapter consulted "
                    f"hits={len(semantic_hits)} supplemental_contracts={len(semantic_contract_ids)} "
                    f"supplemental_entities={len(semantic_entity_ids)} "
                    f"supplemental_workflows={len(semantic_workflows)} "
                    f"supplemental_evidence={len(semantic_evidence)}"
                )
            if not contract_candidates and not entity_candidates and not query:
                for contract in contracts:
                    self._add_contract(contract_candidates, contract, 0.0)
            trace.append(
                f"retrieval candidates contracts={len(contract_candidates)} "
                f"entities={len(entity_candidates)}"
            )

        if not entities_expanded:
            self._expand_entities_one_hop(
                request,
                contract_candidates,
                entity_candidates,
                trace,
            )
        ranked, ranked_warnings = self._rank_contracts(request, contract_candidates)
        warnings = list(ranked_warnings)
        if request.mode != ContextMode.FULL_DUMP:
            ranked = [
                item for item in ranked if item[1].contract_id not in semantic_contract_ids
            ] + [
                item for item in ranked if item[1].contract_id in semantic_contract_ids
            ]
            ranked, variant_warnings = self._collapse_contract_variants(ranked)
            warnings.extend(variant_warnings)
        proposals: list[tuple[KnowledgeItem, OperationContract | None]] = []
        semantic_proposals: list[tuple[KnowledgeItem, OperationContract | None]] = []
        selected_contract_ids: set[str] = set()
        for score, contract in ranked:
            selected_contract_ids.add(contract.contract_id)
            if request.mode == ContextMode.FULL_DUMP:
                item_type = "contract"
                content = contract.model_dump(mode="json", exclude_none=True)
            else:
                item_type = "fingerprint"
                content = OperationFingerprint.from_contract(contract).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            target_proposals = (
                semantic_proposals
                if contract.contract_id in semantic_contract_ids
                else proposals
            )
            target_proposals.append(
                (
                    KnowledgeItem(
                        item_id=contract.contract_id,
                        item_type=item_type,
                        score=score,
                        content=content,
                        estimated_tokens=_estimated_tokens(content),
                    ),
                    contract,
                )
            )

        if request.profile == RetrievalProfile.PLANNING or request.workflow_id:
            for workflow in self.store.list_workflows(request.software_id):
                operation_ids = {step.operation_id for step in workflow.steps}
                if request.workflow_id and workflow.workflow_id != request.workflow_id:
                    continue
                if not request.workflow_id and not operation_ids.intersection(selected_contract_ids):
                    continue
                if not self._scope_matches(workflow.version_scope, request.version):
                    warnings.append(
                        f"version mismatch excluded workflow {workflow.workflow_id} ({workflow.name})"
                    )
                    continue
                content = workflow.model_dump(mode="json", exclude_none=True)
                proposals.append(
                    (
                        KnowledgeItem(
                            item_id=workflow.workflow_id,
                            item_type="workflow",
                            score=2.0,
                            content=content,
                            estimated_tokens=_estimated_tokens(content),
                        ),
                        None,
                    )
                )

        for score, workflow in sorted(
            semantic_workflows.values(),
            key=lambda value: (-value[0], value[1].workflow_id),
        ):
            if not self._scope_matches(workflow.version_scope, request.version):
                continue
            content = workflow.model_dump(mode="json", exclude_none=True)
            semantic_proposals.append(
                (
                    KnowledgeItem(
                        item_id=workflow.workflow_id,
                        item_type="workflow",
                        score=score,
                        content=content,
                        estimated_tokens=_estimated_tokens(content),
                    ),
                    None,
                )
            )

        for score, entity in sorted(
            entity_candidates.values(),
            key=lambda value: (-value[0], value[1].entity_id),
        ):
            if entity.status in {KnowledgeStatus.RETRACTED, KnowledgeStatus.DEPRECATED}:
                continue
            if not self._scope_matches(entity.version_scope, request.version):
                warnings.append(
                    f"version mismatch excluded entity {entity.entity_id} ({entity.name})"
                )
                continue
            content = entity.model_dump(mode="json", exclude_none=True)
            target_proposals = (
                semantic_proposals if entity.entity_id in semantic_entity_ids else proposals
            )
            target_proposals.append(
                (
                    KnowledgeItem(
                        item_id=entity.entity_id,
                        item_type="entity",
                        score=score,
                        content=content,
                        estimated_tokens=_estimated_tokens(content),
                    ),
                    None,
                )
            )

        if request.include_evidence:
            for score, contract in ranked:
                for evidence_id in contract.evidence_ids:
                    evidence = self.store.get_evidence(evidence_id)
                    if evidence is None:
                        continue
                    content = evidence.model_dump(mode="json", exclude_none=True)
                    proposals.append(
                        (
                            KnowledgeItem(
                                item_id=evidence.evidence_id,
                                item_type="evidence",
                                score=score,
                                content=content,
                                estimated_tokens=_estimated_tokens(content),
                            ),
                            None,
                        )
                    )

            for score, evidence in sorted(
                semantic_evidence.values(),
                key=lambda value: (-value[0], value[1].evidence_id),
            ):
                content = evidence.model_dump(mode="json", exclude_none=True)
                semantic_proposals.append(
                    (
                        KnowledgeItem(
                            item_id=evidence.evidence_id,
                            item_type="evidence",
                            score=score,
                            content=content,
                            estimated_tokens=_estimated_tokens(content),
                        ),
                        None,
                    )
                )

        proposals.extend(semantic_proposals)
        packed, truncated = self._pack(request, proposals)
        trace.append(
            f"packed items={len(packed)}/{len(proposals)} "
            f"estimated_tokens={sum(item.estimated_tokens for item in packed)}"
        )
        return self._packet(
            request,
            packed,
            trace=trace,
            warnings=tuple(dict.fromkeys(warnings)),
            candidate_count=len({_item_key(item) for item, _ in proposals}),
            truncated=truncated,
        )

    def retrieve_delta(
        self,
        request: KnowledgeRequest,
        previous: KnowledgePacket,
    ) -> KnowledgePacket:
        """Return only new or changed items while retaining a digest of the full packet."""
        if previous.request.software_id != request.software_id:
            raise ValueError("delta base belongs to different software")
        current = self.retrieve(request)
        changed = tuple(
            item
            for item in current.items
            if previous.item_hashes.get(_item_key(item)) != current.item_hashes[_item_key(item)]
        )
        removed = tuple(sorted(set(previous.item_hashes) - set(current.item_hashes)))
        tokens = sum(item.estimated_tokens for item in changed)
        return current.model_copy(
            update={
                "items": changed,
                "estimated_tokens": tokens,
                "selected_count": len(changed),
                "content_characters": _content_characters(changed),
                "budget_utilization": tokens / request.token_budget,
                "is_delta": True,
                "base_packet_digest": previous.packet_digest,
                "removed_item_ids": removed,
                "retrieval_trace": (
                    *current.retrieval_trace,
                    f"delta changed={len(changed)} removed={len(removed)}",
                ),
            }
        )

    def get_contract(self, contract_id: str, *, include_evidence: bool = False) -> KnowledgePacket:
        contract = self.store.get_contract(contract_id)
        if contract is None:
            raise KnowledgeNotFoundError("unknown operation contract", contract_id=contract_id)
        request = KnowledgeRequest(
            software_id=contract.software_id,
            query=contract.name,
            version=contract.version_scope.exact[0]
            if len(contract.version_scope.exact) == 1
            else None,
            max_items=max(1, 1 + len(contract.evidence_ids)),
            token_budget=100_000,
            include_evidence=include_evidence,
            mode=ContextMode.FILTERED,
        )
        content = contract.model_dump(mode="json", exclude_none=True)
        items = [
            KnowledgeItem(
                item_id=contract.contract_id,
                item_type="contract",
                score=_STATUS_WEIGHT[contract.status],
                content=content,
                estimated_tokens=_estimated_tokens(content),
            )
        ]
        if include_evidence:
            for evidence_id in contract.evidence_ids:
                evidence = self.store.get_evidence(evidence_id)
                if evidence is not None:
                    evidence_content = evidence.model_dump(mode="json", exclude_none=True)
                    items.append(
                        KnowledgeItem(
                            item_id=evidence_id,
                            item_type="evidence",
                            content=evidence_content,
                            estimated_tokens=_estimated_tokens(evidence_content),
                        )
                    )
        return self._packet(request, tuple(items), trace=["direct contract lookup"])

    def causal_frontier(
        self,
        *,
        software_id: str,
        desired_state: tuple[StatePredicate, ...],
        current_state: tuple[StatePredicate, ...] = (),
        version: str | None = None,
        max_depth: int = 3,
        max_candidates: int = 20,
    ) -> tuple[FrontierCandidate, ...]:
        return self.explain_causal_frontier(
            software_id=software_id,
            desired_state=desired_state,
            current_state=current_state,
            version=version,
            max_depth=max_depth,
            max_candidates=max_candidates,
        ).candidates

    def explain_causal_frontier(
        self,
        *,
        software_id: str,
        desired_state: tuple[StatePredicate, ...],
        current_state: tuple[StatePredicate, ...] = (),
        version: str | None = None,
        max_depth: int = 3,
        max_candidates: int = 20,
    ) -> CausalFrontierResult:
        """Find a bounded causally relevant operation closure for desired state."""

        self._require_software(software_id)
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        current = _predicate_keys(current_state)
        contracts = []
        for contract in self.store.list_contracts(software_id):
            if contract.status in {
                KnowledgeStatus.RETRACTED,
                KnowledgeStatus.DEPRECATED,
                KnowledgeStatus.CONFLICTING,
            }:
                continue
            if self._version_score(contract, version) == float("-inf"):
                continue
            contracts.append(contract)

        by_effect: dict[str, list[OperationContract]] = {}
        for contract in contracts:
            for effect in contract.effects:
                by_effect.setdefault(effect.key, []).append(contract)
        for values in by_effect.values():
            values.sort(key=lambda item: item.contract_id)

        candidates: dict[str, FrontierCandidate] = {}
        unresolved: dict[str, StatePredicate] = {}
        conflicting: dict[str, StatePredicate] = {}
        trace = [
            f"causal frontier goals={len(desired_state)} max_depth={max_depth} "
            f"max_candidates={max_candidates}"
        ]
        truncated = False

        state_by_slot: dict[str, list[StatePredicate]] = {}
        for predicate in current_state:
            state_by_slot.setdefault(predicate_slot(predicate), []).append(predicate)
        for values in state_by_slot.values():
            if any(
                predicates_conflict(left, right)
                for index, left in enumerate(values)
                for right in values[index + 1 :]
            ):
                for value in values:
                    conflicting[value.key] = value

        def walk(
            goal: StatePredicate,
            *,
            depth: int,
            downstream: tuple[str, ...],
            visited: frozenset[str],
        ) -> None:
            nonlocal truncated
            if goal.key in current:
                trace.append(f"satisfied depth={depth} predicate={goal.key}")
                return
            for observed in state_by_slot.get(predicate_slot(goal), ()):
                if predicates_conflict(goal, observed):
                    conflicting[observed.key] = observed
                    trace.append(
                        f"conflict depth={depth} required={goal.key} observed={observed.key}"
                    )
            producers = by_effect.get(goal.key, ())
            if not producers:
                unresolved[goal.key] = goal
                trace.append(f"unresolved depth={depth} predicate={goal.key}")
                return
            for contract in producers:
                if contract.contract_id in visited:
                    truncated = True
                    unresolved[goal.key] = goal
                    trace.append(f"cycle excluded contract={contract.contract_id}")
                    continue
                if contract.contract_id not in candidates and len(candidates) >= max_candidates:
                    truncated = True
                    trace.append("candidate limit reached")
                    return
                missing = tuple(
                    predicate
                    for predicate in contract.preconditions
                    if predicate.key not in current
                )
                path = (contract.contract_id, *downstream)
                candidate = FrontierCandidate(
                    contract_id=contract.contract_id,
                    name=contract.name,
                    executable_now=not missing,
                    relevant_effects=(goal.key,),
                    missing_preconditions=missing,
                    depth=depth,
                    causal_path=path,
                )
                previous = candidates.get(contract.contract_id)
                if previous is None:
                    candidates[contract.contract_id] = candidate
                else:
                    candidates[contract.contract_id] = candidate.model_copy(
                        update={
                            "depth": max(previous.depth, candidate.depth),
                            "causal_path": (
                                candidate.causal_path
                                if candidate.depth > previous.depth
                                else previous.causal_path
                            ),
                            "relevant_effects": tuple(
                                sorted(
                                    set(previous.relevant_effects)
                                    | set(candidate.relevant_effects)
                                )
                            ),
                        }
                    )
                trace.append(
                    f"producer depth={depth} predicate={goal.key} "
                    f"contract={contract.contract_id} missing={len(missing)}"
                )
                if not missing:
                    continue
                if depth >= max_depth:
                    truncated = True
                    for predicate in missing:
                        unresolved[predicate.key] = predicate
                    trace.append(f"depth limit reached contract={contract.contract_id}")
                    continue
                next_visited = visited | {contract.contract_id}
                for predicate in missing:
                    walk(
                        predicate,
                        depth=depth + 1,
                        downstream=path,
                        visited=next_visited,
                    )

        for goal in desired_state:
            walk(goal, depth=0, downstream=(), visited=frozenset())

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                not item.executable_now,
                -item.depth,
                item.name,
                item.contract_id,
            )
        )
        trace.append(
            f"causal frontier selected={len(ordered)} unresolved={len(unresolved)} "
            f"conflicts={len(conflicting)} truncated={str(truncated).lower()}"
        )
        return CausalFrontierResult(
            candidates=tuple(ordered),
            unresolved_predicates=tuple(unresolved[key] for key in sorted(unresolved)),
            conflicting_predicates=tuple(
                conflicting[key] for key in sorted(conflicting)
            ),
            trace=tuple(trace),
            max_depth=max_depth,
            max_candidates=max_candidates,
            truncated=truncated,
        )

    def _add_exact_candidates(
        self,
        request: KnowledgeRequest,
        contracts: tuple[OperationContract, ...],
        contract_candidates: dict[str, tuple[float, OperationContract]],
        entity_candidates: dict[str, tuple[float, Entity]],
    ) -> None:
        query = _normalized(request.query)
        known_commands = {_normalized(value) for value in request.known_commands}
        known_files = {_normalized(value) for value in request.known_files}
        workflow_contract_scores: dict[str, float] = {}
        if request.workflow_id:
            workflow = self.store.get_workflow(request.workflow_id)
            if workflow is not None and workflow.software_id == request.software_id:
                steps = {step.step_id: step for step in workflow.steps}
                for position in request.workflow_position:
                    step = steps.get(position)
                    if step is None:
                        continue
                    workflow_contract_scores[step.operation_id] = 11.0
                    for successor in step.on_success:
                        if successor in steps:
                            workflow_contract_scores[steps[successor].operation_id] = 9.0
        for contract in contracts:
            names = {
                _normalized(contract.contract_id),
                _normalized(contract.name),
                *(_normalized(value) for value in _contract_aliases(contract)),
            }
            invocation = _contract_invocation(contract)
            if invocation:
                names.add(_normalized(invocation))
            if query and query in names:
                self._add_contract(contract_candidates, contract, 12.0)
            if known_commands.intersection(names):
                self._add_contract(contract_candidates, contract, 10.0)
            if known_files.intersection(
                _normalized(value) for value in (*contract.inputs, *contract.outputs)
            ):
                self._add_contract(contract_candidates, contract, 8.0)
            if set(request.known_entity_ids).intersection(contract.related_entity_ids):
                self._add_contract(contract_candidates, contract, 5.0)
            if contract.contract_id in workflow_contract_scores:
                self._add_contract(
                    contract_candidates,
                    contract,
                    workflow_contract_scores[contract.contract_id],
                )
        for entity in self.store.list_entities(request.software_id):
            names = {
                _normalized(entity.entity_id),
                _normalized(entity.name),
                *(_normalized(value) for value in entity.aliases),
            }
            path = entity.attributes.get("path")
            if path:
                names.add(_normalized(str(path)))
            if (
                entity.entity_id in request.known_entity_ids
                or (query and query in names)
                or bool(known_files.intersection(names))
            ):
                self._add_entity(entity_candidates, entity, 10.0)
                for contract in contracts:
                    if entity.entity_id in contract.related_entity_ids:
                        self._add_contract(contract_candidates, contract, 5.0)

    def _expand_entities_one_hop(
        self,
        request: KnowledgeRequest,
        contracts: dict[str, tuple[float, OperationContract]],
        entities: dict[str, tuple[float, Entity]],
        trace: list[str],
    ) -> None:
        seed_ids = set(request.known_entity_ids)
        seed_ids.update(entities)
        for _, contract in contracts.values():
            seed_ids.update(contract.related_entity_ids)
        for entity_id in tuple(seed_ids):
            entity = self.store.get_entity(entity_id)
            if entity is not None and entity.software_id == request.software_id:
                self._add_entity(entities, entity, 2.0)
        relations = self.store.related(request.software_id, seed_ids)
        for relation in relations:
            for entity_id in (relation.source_entity_id, relation.target_entity_id):
                entity = self.store.get_entity(entity_id)
                if entity is not None:
                    self._add_entity(entities, entity, 1.0)
        if seed_ids:
            trace.append(f"one-hop expansion seeds={len(seed_ids)} relations={len(relations)}")

    def _rank_contracts(
        self,
        request: KnowledgeRequest,
        candidates: dict[str, tuple[float, OperationContract]],
    ) -> tuple[list[tuple[float, OperationContract]], tuple[str, ...]]:
        warnings: list[str] = []
        open_conflicts = {
            item_id
            for conflict in self.store.list_conflicts(
                request.software_id,
                state=ConflictState.OPEN,
            )
            for item_id in (conflict.item_id, conflict.conflicting_item_id)
            if item_id
        }
        latest_by_contract: dict[str, float] = {}
        for _, contract in candidates.values():
            timestamps = [
                evidence.retrieved_at.timestamp()
                for evidence_id in contract.evidence_ids
                if (evidence := self.store.get_evidence(evidence_id)) is not None
            ]
            latest_by_contract[contract.contract_id] = max(timestamps, default=0.0)
        newest = max(latest_by_contract.values(), default=0.0)
        ranked: list[tuple[float, OperationContract]] = []
        current = _predicate_keys(request.current_state)
        for lexical_score, contract in candidates.values():
            if contract.status in {KnowledgeStatus.RETRACTED, KnowledgeStatus.DEPRECATED}:
                continue
            version_score = self._version_score(contract, request.version)
            if version_score == float("-inf"):
                warnings.append(
                    f"version mismatch excluded contract {contract.contract_id} ({contract.name})"
                )
                continue
            evidence = tuple(
                item
                for evidence_id in contract.evidence_ids
                if (item := self.store.get_evidence(evidence_id)) is not None
            )
            authority_score = (
                0.75 * (sum(item.authoritative for item in evidence) / len(evidence))
                if evidence
                else 0.0
            )
            freshness_score = (
                0.25
                if newest and latest_by_contract[contract.contract_id] == newest
                else 0.0
            )
            source_diversity = 0.25 if len({item.source_uri for item in evidence}) > 1 else 0.0
            conflict_score = -4.0 if contract.contract_id in open_conflicts else 0.0
            profile_score = 0.0
            if request.profile == RetrievalProfile.REPAIR and request.error_signature:
                needle = request.error_signature.casefold()
                if any(
                    failure.pattern.casefold() in needle
                    or needle in failure.pattern.casefold()
                    for failure in contract.failure_signatures
                ):
                    profile_score += 6.0
            elif request.profile == RetrievalProfile.EXECUTION:
                missing = [item for item in contract.preconditions if item.key not in current]
                profile_score += 2.0 if not missing else -min(2.0, len(missing) * 0.5)
            elif request.profile == RetrievalProfile.PLANNING:
                profile_score += 1.0 if contract.effects else 0.0
            ranked.append(
                (
                    lexical_score
                    + _STATUS_WEIGHT[contract.status]
                    + version_score
                    + authority_score
                    + freshness_score
                    + source_diversity
                    + conflict_score
                    + profile_score,
                    contract,
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1].contract_id))
        return ranked, tuple(dict.fromkeys(warnings))

    def _pack(
        self,
        request: KnowledgeRequest,
        proposals: list[tuple[KnowledgeItem, OperationContract | None]],
    ) -> tuple[tuple[KnowledgeItem, ...], bool]:
        packed: list[KnowledgeItem] = []
        seen: set[str] = set()
        consumed = 0
        truncated = False
        for item, contract in proposals:
            key = _item_key(item)
            if key in seen:
                continue
            seen.add(key)
            if len(packed) >= request.max_items:
                truncated = True
                continue
            candidate = item
            if consumed + candidate.estimated_tokens > request.token_budget and contract is not None:
                compact = self._compact_contract_item(contract, item.score)
                if consumed + compact.estimated_tokens <= request.token_budget:
                    candidate = compact
            if consumed + candidate.estimated_tokens > request.token_budget:
                truncated = True
                continue
            packed.append(candidate)
            consumed += candidate.estimated_tokens
        return tuple(packed), truncated

    @staticmethod
    def _collapse_contract_variants(
        ranked: list[tuple[float, OperationContract]],
    ) -> tuple[list[tuple[float, OperationContract]], tuple[str, ...]]:
        """Expose one fingerprint per capability while retaining a variant warning."""
        selected: list[tuple[float, OperationContract]] = []
        counts: dict[str, int] = {}
        seen: set[str] = set()
        for score, contract in ranked:
            raw_key = contract.metadata.get("equivalence_key")
            key = str(raw_key) if raw_key else (
                f"{contract.interface}:{_normalized(_contract_invocation(contract) or contract.name)}"
            )
            counts[key] = counts.get(key, 0) + 1
            if key in seen:
                continue
            seen.add(key)
            selected.append((score, contract))
        warnings = tuple(
            f"{count} documented variants exist for {key}; highest-ranked contract returned"
            for key, count in sorted(counts.items())
            if count > 1
        )
        return selected, warnings

    @staticmethod
    def _compact_contract_item(contract: OperationContract, score: float) -> KnowledgeItem:
        # Preconditions and effects are never summarized away: they determine whether
        # the operation is safe and useful at the current causal frontier.
        content = {
            "contract_id": contract.contract_id,
            "name": contract.name,
            "interface": contract.interface,
            "preconditions": [
                item.model_dump(mode="json", exclude_none=True)
                for item in contract.preconditions
            ],
            "effects": [
                item.model_dump(mode="json", exclude_none=True) for item in contract.effects
            ],
            "status": contract.status.value,
            "compacted": True,
        }
        return KnowledgeItem(
            item_id=contract.contract_id,
            item_type="fingerprint",
            score=score,
            content=content,
            estimated_tokens=_estimated_tokens(content),
        )

    @staticmethod
    def _add_contract(
        values: dict[str, tuple[float, OperationContract]],
        contract: OperationContract,
        score: float,
    ) -> None:
        previous = values.get(contract.contract_id)
        if previous is None or score > previous[0]:
            values[contract.contract_id] = (score, contract)

    @staticmethod
    def _add_entity(
        values: dict[str, tuple[float, Entity]],
        entity: Entity,
        score: float,
    ) -> None:
        previous = values.get(entity.entity_id)
        if previous is None or score > previous[0]:
            values[entity.entity_id] = (score, entity)

    @staticmethod
    def _request_factors(request: KnowledgeRequest) -> tuple[str, ...]:
        fields = []
        for name, value in (
            ("query", request.query),
            ("version", request.version),
            ("phase", request.phase),
            ("capability", request.desired_capability),
            ("known_entities", request.known_entity_ids),
            ("known_files", request.known_files),
            ("known_commands", request.known_commands),
            ("workflow", request.workflow_id),
            ("workflow_position", request.workflow_position),
            ("current_state", request.current_state),
            ("error_signature", request.error_signature),
            ("entity_types", request.entity_types),
        ):
            if value:
                fields.append(name)
        return tuple(fields)

    def _packet(
        self,
        request: KnowledgeRequest,
        items: tuple[KnowledgeItem, ...],
        *,
        trace: Iterable[str],
        warnings: tuple[str, ...] = (),
        candidate_count: int | None = None,
        truncated: bool = False,
    ) -> KnowledgePacket:
        hashes = {_item_key(item): _item_hash(item) for item in items}
        tokens = sum(item.estimated_tokens for item in items)
        return KnowledgePacket(
            request=request,
            items=items,
            estimated_tokens=tokens,
            truncated=truncated,
            retrieval_trace=tuple(trace),
            warnings=warnings,
            candidate_count=len(items) if candidate_count is None else candidate_count,
            selected_count=len(items),
            content_characters=_content_characters(items),
            budget_utilization=tokens / request.token_budget,
            packet_digest=_packet_digest(hashes),
            item_hashes=hashes,
        )

    def _require_software(self, software_id: str) -> None:
        if self.store.get_software(software_id) is None:
            raise KnowledgeNotFoundError("unknown software identity", software_id=software_id)

    @staticmethod
    def _version_score(contract: OperationContract, version: str | None) -> float:
        scope = contract.version_scope
        if not version:
            constrained = bool(scope.exact or scope.compatible or scope.minimum or scope.maximum)
            return -0.5 if constrained else 0.0
        if scope.exact or scope.compatible:
            return 3.0 if scope.matches(version) else float("-inf")
        if scope.matches(version):
            return 1.0
        return float("-inf")

    @staticmethod
    def _scope_matches(scope: Any, version: str | None) -> bool:
        if not version:
            return True
        if scope.exact or scope.compatible or scope.minimum or scope.maximum:
            return scope.matches(version)
        return True
