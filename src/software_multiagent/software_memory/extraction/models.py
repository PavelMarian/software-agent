"""Serializable contracts shared by extraction pipeline components."""

from __future__ import annotations

from enum import Enum
from hashlib import sha256
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from software_multiagent.software_memory.schema.models import FailureSignature, OperationParameter, StatePredicate, stable_id


class ExtractionModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MaterialKind(str, Enum):
    API = "api"
    CLI = "cli"
    CONFIGURATION = "configuration"
    TUTORIAL = "tutorial_workflow"
    ERROR_REFERENCE = "error_reference"
    SOURCE_CODE = "source_code"
    IRRELEVANT = "irrelevant"


class EvidenceBlock(ExtractionModel):
    block_id: str
    block_type: Literal["prose", "code", "table", "example"]
    content: str = Field(min_length=1)
    language: str | None = None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class EvidenceUnit(ExtractionModel):
    unit_id: str
    evidence_id: str
    software_id: str
    source_uri: str
    source_locator: str | None = None
    unit_locator: str
    heading_path: tuple[str, ...] = ()
    normalized_text: str = Field(min_length=1)
    content_hash: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    blocks: tuple[EvidenceBlock, ...] = ()
    source_kind: str
    source_version: str | None = None


class Classification(ExtractionModel):
    unit_id: str
    primary: MaterialKind
    labels: tuple[MaterialKind, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = ()


class ClaimAnchor(ExtractionModel):
    evidence_id: str
    unit_id: str
    locator: str
    quote: str = Field(min_length=1, max_length=2_000)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def offsets_are_ordered(self) -> "ClaimAnchor":
        if self.end_offset <= self.start_offset:
            raise ValueError("anchor end_offset must be greater than start_offset")
        return self


class EntityDraft(ExtractionModel):
    source_unit_id: str
    entity_key: str
    entity_type: str
    name: str
    summary: str = ""
    aliases: tuple[str, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: tuple[str, ...]
    anchors: tuple[ClaimAnchor, ...]
    extraction_method: str
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class OperationDraft(ExtractionModel):
    source_unit_id: str
    operation_key: str
    name: str
    interface: str
    purpose: str = ""
    aliases: tuple[str, ...] = ()
    invocation: str | None = None
    parameters: tuple[OperationParameter, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    preconditions: tuple[StatePredicate, ...] = ()
    effects: tuple[StatePredicate, ...] = ()
    side_effects: tuple[StatePredicate, ...] = ()
    success_signals: tuple[StatePredicate, ...] = ()
    failure_signatures: tuple[FailureSignature, ...] = ()
    evidence_ids: tuple[str, ...]
    anchors: tuple[ClaimAnchor, ...]
    ambiguous_fields: tuple[str, ...] = ()
    extraction_method: str
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def equivalence_key(self) -> str:
        value = self.invocation or self.name
        normalized = " ".join(value.lower().split())
        return f"{self.interface}:{normalized}"

    @property
    def semantic_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={
                "source_unit_id",
                "evidence_ids",
                "anchors",
                "aliases",
                "ambiguous_fields",
                "extraction_method",
                "extraction_confidence",
            },
        )
        def strip_provenance(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: strip_provenance(item)
                    for key, item in value.items()
                    if key not in {"evidence_ids", "source_unit_id"}
                }
            if isinstance(value, list):
                return [strip_provenance(item) for item in value]
            return value

        payload = strip_provenance(payload)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()


class SourceConfirmation(ExtractionModel):
    unit_id: str
    evidence_id: str
    symbol: str
    operation_key: str
    anchor: ClaimAnchor
    confirmed: bool = True


class UnitExtraction(ExtractionModel):
    unit_id: str
    classification: Classification
    entities: tuple[EntityDraft, ...] = ()
    operations: tuple[OperationDraft, ...] = ()
    failures: tuple[FailureSignature, ...] = ()
    source_confirmations: tuple[SourceConfirmation, ...] = ()
    warnings: tuple[str, ...] = ()


class ReviewQueueItem(ExtractionModel):
    review_id: str
    software_id: str
    knowledge_kind: Literal["entity", "contract", "claim", "reconciliation"]
    knowledge_id: str
    field_path: str
    reason: str
    evidence_ids: tuple[str, ...] = ()
    candidate_values: tuple[Any, ...] = ()
    state: Literal["open", "resolved", "dismissed"] = "open"


class CompilationReport(ExtractionModel):
    schema_version: int = 1
    software_id: str
    evidence_records: int
    normalized_units: int
    units_processed: int
    units_reused: int
    units_removed: int
    classifications: dict[str, int]
    entity_drafts: int
    operation_drafts: int
    entities_created: int
    entities_updated: int
    entities_skipped: int
    contracts_created: int
    contracts_updated: int
    contracts_skipped: int
    exact_duplicates_merged: int
    semantic_conflicts: int
    reconciliation_items: int
    source_confirmations: int
    review_items: int
    warnings: tuple[str, ...] = ()
    details: tuple[dict[str, Any], ...] = ()
    compacted: bool = False


class UnitCacheEntry(ExtractionModel):
    content_hash: str
    extraction: UnitExtraction


class ExtractionState(ExtractionModel):
    schema_version: int = 1
    software_id: str
    units: dict[str, UnitCacheEntry] = Field(default_factory=dict)
    generated_entity_ids: tuple[str, ...] = ()
    generated_contract_ids: tuple[str, ...] = ()


def draft_id(prefix: str, unit_id: str, key: str) -> str:
    return stable_id(prefix, unit_id, key)
