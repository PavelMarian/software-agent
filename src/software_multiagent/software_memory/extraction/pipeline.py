"""Incremental evidence-to-memory compilation and conservative reconciliation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from software_multiagent.software_memory.extraction.deterministic import extract_deterministically
from software_multiagent.software_memory.extraction.models import (
    CompilationReport,
    EntityDraft,
    ExtractionState,
    OperationDraft,
    ReviewQueueItem,
    UnitCacheEntry,
    UnitExtraction,
)
from software_multiagent.software_memory.extraction.normalization import classify_unit, normalize_evidence
from software_multiagent.software_memory.extraction.structured import (
    StructuredExtractionClient,
    compact_details,
    extract_with_model,
)
from software_multiagent.software_memory.schema.models import (
    ActorKind,
    Entity,
    EvidenceRecord,
    KnowledgeConflict,
    KnowledgeStatus,
    OperationContract,
    VersionScope,
    stable_id,
)
from software_multiagent.software_memory.operations.service import MemoryService


def _without_runtime_fields(value: Any) -> dict[str, Any]:
    return value.model_dump(exclude={"revision", "status", "verification", "created_at"})


def _merge_unit_extractions(*values: UnitExtraction) -> UnitExtraction:
    first = values[0]
    return UnitExtraction(
        unit_id=first.unit_id,
        classification=first.classification,
        entities=tuple(item for value in values for item in value.entities),
        operations=tuple(item for value in values for item in value.operations),
        failures=tuple(item for value in values for item in value.failures),
        source_confirmations=tuple(
            item for value in values for item in value.source_confirmations
        ),
        warnings=tuple(item for value in values for item in value.warnings),
    )


def _load_state(path: Path, software_id: str) -> ExtractionState:
    if not path.exists():
        return ExtractionState(software_id=software_id)
    state = ExtractionState.model_validate_json(path.read_text(encoding="utf-8"))
    if state.software_id != software_id:
        raise ValueError("extraction state belongs to different software")
    return state


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if hasattr(value, "model_dump_json"):
        content = value.model_dump_json(indent=2)
    else:
        content = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    temporary.write_text(content + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_review_queue(path: Path) -> tuple[ReviewQueueItem, ...]:
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("review queue must be a JSON array")
    return tuple(ReviewQueueItem.model_validate(item) for item in raw)


@dataclass
class KnowledgeCompiler:
    service: MemoryService
    state_path: Path
    review_queue_path: Path
    structured_client: StructuredExtractionClient | None = None
    report_detail_limit: int = 12_000

    def compile(
        self,
        *,
        software_id: str,
        evidence: Iterable[EvidenceRecord] | None = None,
    ) -> CompilationReport:
        records = tuple(evidence or self.service.store.list_evidence(software_id))
        if any(item.software_id != software_id for item in records):
            raise ValueError("all evidence records must belong to the compiled software")
        state = _load_state(self.state_path, software_id)
        previous_review = _load_review_queue(self.review_queue_path)
        units = tuple(unit for record in records for unit in normalize_evidence(record))
        current_ids = {unit.unit_id for unit in units}
        removed = set(state.units) - current_ids
        cache = dict(state.units)
        processed = 0
        reused = 0
        details: list[dict[str, Any]] = []
        for unit in units:
            cached = cache.get(unit.unit_id)
            if cached is not None and cached.content_hash == unit.content_hash:
                reused += 1
                continue
            classification = classify_unit(unit)
            deterministic = extract_deterministically(unit, classification)
            extracted = deterministic
            if (
                self.structured_client is not None
                and classification.primary.value != "irrelevant"
                and not deterministic.entities
                and not deterministic.operations
            ):
                extracted = _merge_unit_extractions(
                    deterministic,
                    extract_with_model(unit, classification, self.structured_client),
                )
            cache[unit.unit_id] = UnitCacheEntry(
                content_hash=unit.content_hash,
                extraction=extracted,
            )
            processed += 1
            details.append(
                {
                    "unit_id": unit.unit_id,
                    "classification": classification.primary.value,
                    "entities": len(extracted.entities),
                    "operations": len(extracted.operations),
                    "warnings": list(extracted.warnings),
                }
            )
        for unit_id in removed:
            cache.pop(unit_id, None)

        active = tuple(cache[unit.unit_id].extraction for unit in units)
        counters: Counter[str] = Counter()
        review_items: list[ReviewQueueItem] = []
        for unit_id in sorted(removed):
            old = state.units[unit_id].extraction
            review_items.append(
                ReviewQueueItem(
                    review_id=stable_id("review", software_id, "removed-unit", unit_id),
                    software_id=software_id,
                    knowledge_kind="reconciliation",
                    knowledge_id=unit_id,
                    field_path="source_unit",
                    reason="previously extracted evidence unit disappeared from the active corpus",
                    evidence_ids=tuple(
                        dict.fromkeys(
                            item
                            for draft in (*old.entities, *old.operations)
                            for item in draft.evidence_ids
                        )
                    ),
                )
            )
        with self.service.batch():
            self._persist_entities(software_id, active, counters, review_items)
            contract_ids = self._persist_operations(
                software_id, active, counters, review_items
            )
            self._reconcile_previous_contracts(
                software_id,
                set(state.generated_contract_ids),
                contract_ids,
                counters,
                review_items,
            )
        state = ExtractionState(
            software_id=software_id,
            units=cache,
            generated_entity_ids=tuple(
                item.entity_id for item in self.service.store.list_entities(software_id)
                if item.attributes.get("generated_by") == "software_memory.extraction"
            ),
            generated_contract_ids=tuple(contract_ids),
        )
        _write_json(self.state_path, state)
        merged_review = {item.review_id: item for item in previous_review}
        for item in review_items:
            previous = merged_review.get(item.review_id)
            if previous is None or previous.state == "open":
                merged_review[item.review_id] = item
        _write_json(
            self.review_queue_path,
            [item.model_dump(mode="json") for item in sorted(merged_review.values(), key=lambda value: value.review_id)],
        )
        compacted_details, compacted = compact_details(tuple(details), self.report_detail_limit)
        classifications = Counter(
            extraction.classification.primary.value for extraction in active
        )
        return CompilationReport(
            software_id=software_id,
            evidence_records=len(records),
            normalized_units=len(units),
            units_processed=processed,
            units_reused=reused,
            units_removed=len(removed),
            classifications=dict(classifications),
            entity_drafts=sum(len(value.entities) for value in active),
            operation_drafts=sum(len(value.operations) for value in active),
            entities_created=counters["entities_created"],
            entities_updated=counters["entities_updated"],
            entities_skipped=counters["entities_skipped"],
            contracts_created=counters["contracts_created"],
            contracts_updated=counters["contracts_updated"],
            contracts_skipped=counters["contracts_skipped"],
            exact_duplicates_merged=counters["duplicates"],
            semantic_conflicts=counters["semantic_conflicts"],
            reconciliation_items=counters["reconciliation_items"] + len(removed),
            source_confirmations=sum(len(value.source_confirmations) for value in active),
            review_items=len(review_items),
            warnings=tuple(warning for value in active for warning in value.warnings),
            details=compacted_details,
            compacted=compacted,
        )

    def _persist_entities(
        self,
        software_id: str,
        extractions: tuple[UnitExtraction, ...],
        counters: Counter[str],
        review: list[ReviewQueueItem],
    ) -> None:
        groups: dict[tuple[str, str], list[EntityDraft]] = defaultdict(list)
        for extraction in extractions:
            for draft in extraction.entities:
                groups[(draft.entity_type, draft.name.casefold())].append(draft)
        for (entity_type, _), drafts in groups.items():
            first = drafts[0]
            evidence_ids = tuple(dict.fromkeys(item for draft in drafts for item in draft.evidence_ids))
            aliases = tuple(dict.fromkeys(item for draft in drafts for item in draft.aliases))
            attribute_candidates = {
                json.dumps(draft.attributes, sort_keys=True, default=str) for draft in drafts
            }
            if len(attribute_candidates) > 1:
                review.append(
                    ReviewQueueItem(
                        review_id=stable_id("review", software_id, entity_type, first.name),
                        software_id=software_id,
                        knowledge_kind="entity",
                        knowledge_id=stable_id("entity", software_id, entity_type, first.name.casefold()),
                        field_path="attributes",
                        reason="sources produced different entity attributes",
                        evidence_ids=evidence_ids,
                        candidate_values=tuple(json.loads(value) for value in sorted(attribute_candidates)),
                    )
                )
            entity = Entity(
                entity_id=stable_id("entity", software_id, entity_type, first.name.casefold()),
                software_id=software_id,
                entity_type=entity_type,
                name=first.name,
                summary=first.summary,
                aliases=aliases,
                attributes={
                    **first.attributes,
                    "generated_by": "software_memory.extraction",
                    "extraction_confidence": max(item.extraction_confidence for item in drafts),
                    "anchors": [anchor.model_dump(mode="json") for item in drafts for anchor in item.anchors],
                },
                evidence_ids=evidence_ids,
                version_scope=self._version_scope(evidence_ids),
                status=KnowledgeStatus.EXTRACTED,
            )
            current = self.service.store.get_entity(entity.entity_id)
            if current is None:
                self.service.record_entity(entity)
                counters["entities_created"] += 1
            elif _without_runtime_fields(current) == _without_runtime_fields(entity):
                counters["entities_skipped"] += 1
            else:
                self.service.revise_from_source(
                    entity,
                    actor_id="knowledge_compiler",
                    reason="source-derived entity semantics or provenance changed",
                )
                counters["entities_updated"] += 1

    def _persist_operations(
        self,
        software_id: str,
        extractions: tuple[UnitExtraction, ...],
        counters: Counter[str],
        review: list[ReviewQueueItem],
    ) -> set[str]:
        equivalences: dict[str, dict[str, list[OperationDraft]]] = defaultdict(lambda: defaultdict(list))
        for extraction in extractions:
            for draft in extraction.operations:
                equivalences[draft.equivalence_key][draft.semantic_hash].append(draft)
        generated: set[str] = set()
        for equivalence_key, variants in equivalences.items():
            variant_ids: list[str] = []
            for semantic_hash, drafts in variants.items():
                first = drafts[0]
                counters["duplicates"] += max(0, len(drafts) - 1)
                evidence_ids = tuple(dict.fromkeys(item for draft in drafts for item in draft.evidence_ids))
                contract_id = stable_id("contract", software_id, equivalence_key, semantic_hash)
                variant_ids.append(contract_id)
                generated.add(contract_id)
                ambiguous = tuple(dict.fromkeys(item for draft in drafts for item in draft.ambiguous_fields))
                contract = OperationContract(
                    contract_id=contract_id,
                    software_id=software_id,
                    name=first.name,
                    interface=first.interface,
                    purpose=first.purpose,
                    parameters=first.parameters,
                    inputs=first.inputs,
                    outputs=first.outputs,
                    preconditions=first.preconditions,
                    effects=first.effects,
                    side_effects=first.side_effects,
                    success_signals=first.success_signals,
                    failure_signatures=first.failure_signatures,
                    evidence_ids=evidence_ids,
                    version_scope=self._version_scope(evidence_ids),
                    status=KnowledgeStatus.EXTRACTED,
                    metadata={
                        **first.metadata,
                        "generated_by": "software_memory.extraction",
                        "equivalence_key": equivalence_key,
                        "semantic_hash": semantic_hash,
                        "aliases": list(dict.fromkeys(item for draft in drafts for item in draft.aliases)),
                        "extraction_methods": list(dict.fromkeys(draft.extraction_method for draft in drafts)),
                        "extraction_confidence": max(draft.extraction_confidence for draft in drafts),
                        "anchors": [anchor.model_dump(mode="json") for draft in drafts for anchor in draft.anchors],
                    },
                )
                current = self.service.store.get_contract(contract_id)
                if current is None:
                    self.service.record_contract(contract)
                    counters["contracts_created"] += 1
                elif _without_runtime_fields(current) == _without_runtime_fields(contract):
                    counters["contracts_skipped"] += 1
                else:
                    self.service.revise_from_source(
                        contract,
                        actor_id="knowledge_compiler",
                        reason="source-derived operation provenance changed",
                    )
                    counters["contracts_updated"] += 1
                for field in ambiguous:
                    review.append(
                        ReviewQueueItem(
                            review_id=stable_id("review", contract_id, field),
                            software_id=software_id,
                            knowledge_kind="claim",
                            knowledge_id=contract_id,
                            field_path=field,
                            reason="extractor marked this operation field as ambiguous",
                            evidence_ids=evidence_ids,
                        )
                    )
            if len(variant_ids) > 1:
                counters["semantic_conflicts"] += 1
                counters["reconciliation_items"] += 1
                primary = variant_ids[0]
                for other in variant_ids[1:]:
                    evidence_ids = tuple(
                        dict.fromkeys(
                            item
                            for contract_id in (primary, other)
                            for item in self.service.store.get_contract(contract_id).evidence_ids  # type: ignore[union-attr]
                        )
                    )
                    conflict_id = stable_id("conflict", primary, other)
                    if self.service.store.get_conflict(conflict_id) is None:
                        self.service.open_conflict(
                            KnowledgeConflict(
                                conflict_id=conflict_id,
                                software_id=software_id,
                                item_kind="contract",
                                item_id=primary,
                                conflicting_item_id=other,
                                reason="equivalent invocation has incompatible extracted semantics",
                                evidence_ids=evidence_ids,
                            ),
                            actor_kind=ActorKind.INGESTOR,
                        )
                    review.append(
                        ReviewQueueItem(
                            review_id=stable_id("review", conflict_id),
                            software_id=software_id,
                            knowledge_kind="reconciliation",
                            knowledge_id=conflict_id,
                            field_path="operation_contract",
                            reason="choose, merge, or version conflicting operation candidates",
                            evidence_ids=evidence_ids,
                            candidate_values=(primary, other),
                        )
                    )
        return generated

    def _reconcile_previous_contracts(
        self,
        software_id: str,
        previous: set[str],
        current: set[str],
        counters: Counter[str],
        review: list[ReviewQueueItem],
    ) -> None:
        """Retain replaced contracts and make source drift explicitly reviewable."""
        current_by_key: dict[str, list[OperationContract]] = defaultdict(list)
        for contract_id in current:
            contract = self.service.store.get_contract(contract_id)
            if contract is not None:
                current_by_key[str(contract.metadata.get("equivalence_key", ""))].append(contract)
        for old_id in sorted(previous - current):
            old = self.service.store.get_contract(old_id)
            if old is None:
                continue
            successors = current_by_key.get(str(old.metadata.get("equivalence_key", "")), [])
            candidate_values: tuple[Any, ...] = (old_id, *(item.contract_id for item in successors))
            evidence_ids = tuple(
                dict.fromkeys(
                    (*old.evidence_ids, *(evidence for item in successors for evidence in item.evidence_ids))
                )
            )
            reason = (
                "source changed the semantics of an equivalent operation"
                if successors
                else "operation disappeared from the active evidence corpus"
            )
            review_id = stable_id("review", software_id, "source-drift", old_id, *candidate_values[1:])
            review.append(
                ReviewQueueItem(
                    review_id=review_id,
                    software_id=software_id,
                    knowledge_kind="reconciliation",
                    knowledge_id=old_id,
                    field_path="operation_contract",
                    reason=reason,
                    evidence_ids=evidence_ids,
                    candidate_values=candidate_values,
                )
            )
            counters["reconciliation_items"] += 1
            for successor in successors:
                conflict_id = stable_id("conflict", old_id, successor.contract_id)
                if self.service.store.get_conflict(conflict_id) is None:
                    self.service.open_conflict(
                        KnowledgeConflict(
                            conflict_id=conflict_id,
                            software_id=software_id,
                            item_kind="contract",
                            item_id=old_id,
                            conflicting_item_id=successor.contract_id,
                            reason=reason,
                            evidence_ids=evidence_ids,
                        ),
                        actor_kind=ActorKind.INGESTOR,
                    )
                counters["semantic_conflicts"] += 1

    def _version_scope(self, evidence_ids: tuple[str, ...]) -> VersionScope:
        versions = tuple(
            dict.fromkeys(
                evidence.source_version
                for evidence_id in evidence_ids
                if (evidence := self.service.store.get_evidence(evidence_id)) is not None
                and evidence.source_version
            )
        )
        if len(versions) == 1:
            return VersionScope(exact=versions, compatibility_verified=False)
        return VersionScope(note="multiple or unknown source versions; reconciliation required")
