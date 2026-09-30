"""Provider-neutral, evidence-bound fallback for prose that rules cannot parse."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from software_multiagent.software_memory.extraction.deterministic import anchor
from software_multiagent.software_memory.extraction.models import (
    Classification,
    EntityDraft,
    EvidenceUnit,
    OperationDraft,
    UnitExtraction,
)
from software_multiagent.software_memory.schema.models import OperationParameter, StatePredicate


class StructuredExtractionClient(Protocol):
    """A model adapter receives exactly one bounded evidence unit per call."""

    def extract(
        self,
        *,
        unit: EvidenceUnit,
        schema: type[BaseModel],
        instructions: str,
    ) -> BaseModel | dict[str, Any]: ...


STRUCTURED_EXTRACTION_INSTRUCTIONS = """Extract only claims explicitly stated in this one evidence unit.
Omit fields and candidates whose values are absent; never infer, complete, or synthesize them.
For every candidate, parameter, precondition, effect, and success signal, copy an exact supporting
substring into anchor_quote. Return only data conforming to the supplied schema."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StructuredEntity(_Strict):
    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    summary: str = ""
    aliases: tuple[str, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)
    anchor_quote: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class StructuredParameter(_Strict):
    name: str = Field(min_length=1)
    required: bool = False
    parameter_type: str | None = None
    description: str | None = None
    anchor_quote: str = Field(min_length=1)


class StructuredPredicate(_Strict):
    predicate: str = Field(min_length=1)
    subject: str | None = None
    value: Any = True
    anchor_quote: str = Field(min_length=1)


class StructuredOperation(_Strict):
    name: str = Field(min_length=1)
    interface: str = Field(min_length=1)
    purpose: str = ""
    aliases: tuple[str, ...] = ()
    invocation: str | None = None
    parameters: tuple[StructuredParameter, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    preconditions: tuple[StructuredPredicate, ...] = ()
    effects: tuple[StructuredPredicate, ...] = ()
    success_signals: tuple[StructuredPredicate, ...] = ()
    anchor_quote: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguous_fields: tuple[str, ...] = ()


class StructuredPayload(_Strict):
    entities: tuple[StructuredEntity, ...] = ()
    operations: tuple[StructuredOperation, ...] = ()
    warnings: tuple[str, ...] = ()


def _predicate(value: StructuredPredicate) -> StatePredicate:
    return StatePredicate(predicate=value.predicate, subject=value.subject, value=value.value)


def extract_with_model(
    unit: EvidenceUnit,
    classification: Classification,
    client: StructuredExtractionClient,
) -> UnitExtraction:
    """Validate every generated claim against an exact quote in the supplied unit."""
    raw = client.extract(
        unit=unit,
        schema=StructuredPayload,
        instructions=STRUCTURED_EXTRACTION_INSTRUCTIONS,
    )
    payload = raw if isinstance(raw, StructuredPayload) else StructuredPayload.model_validate(raw)
    entities = []
    operations = []
    warnings = list(payload.warnings)
    for candidate in payload.entities:
        try:
            claim_anchor = anchor(unit, candidate.anchor_quote)
        except ValueError:
            warnings.append(f"rejected entity {candidate.name!r}: anchor is absent")
            continue
        entities.append(
            EntityDraft(
                source_unit_id=unit.unit_id,
                entity_key=candidate.name,
                entity_type=candidate.entity_type,
                name=candidate.name,
                summary=candidate.summary,
                aliases=candidate.aliases,
                attributes=candidate.attributes,
                evidence_ids=(unit.evidence_id,),
                anchors=(claim_anchor,),
                extraction_method="structured_fallback",
                extraction_confidence=candidate.confidence,
            )
        )
    for candidate in payload.operations:
        quotes = [candidate.anchor_quote]
        quotes.extend(item.anchor_quote for item in candidate.parameters)
        quotes.extend(item.anchor_quote for item in candidate.preconditions)
        quotes.extend(item.anchor_quote for item in candidate.effects)
        quotes.extend(item.anchor_quote for item in candidate.success_signals)
        try:
            claim_anchors = tuple(anchor(unit, quote) for quote in dict.fromkeys(quotes))
        except ValueError:
            warnings.append(f"rejected operation {candidate.name!r}: a claim anchor is absent")
            continue
        operations.append(
            OperationDraft(
                source_unit_id=unit.unit_id,
                operation_key=candidate.invocation or candidate.name,
                name=candidate.name,
                interface=candidate.interface,
                purpose=candidate.purpose,
                aliases=candidate.aliases,
                invocation=candidate.invocation,
                parameters=tuple(
                    OperationParameter(
                        name=value.name,
                        required=value.required,
                        parameter_type=value.parameter_type,
                        description=value.description,
                        evidence_ids=(unit.evidence_id,),
                    )
                    for value in candidate.parameters
                ),
                inputs=candidate.inputs,
                outputs=candidate.outputs,
                preconditions=tuple(_predicate(value) for value in candidate.preconditions),
                effects=tuple(_predicate(value) for value in candidate.effects),
                success_signals=tuple(_predicate(value) for value in candidate.success_signals),
                evidence_ids=(unit.evidence_id,),
                anchors=claim_anchors,
                ambiguous_fields=candidate.ambiguous_fields,
                extraction_method="structured_fallback",
                extraction_confidence=candidate.confidence,
            )
        )
    return UnitExtraction(
        unit_id=unit.unit_id,
        classification=classification,
        entities=tuple(entities),
        operations=tuple(operations),
        warnings=tuple(warnings),
    )


def compact_details(details: tuple[dict[str, Any], ...], max_chars: int) -> tuple[tuple[dict[str, Any], ...], bool]:
    """Bound reports without changing extracted knowledge or evidence."""
    if max_chars < 1:
        return (), bool(details)
    kept: list[dict[str, Any]] = []
    used = 0
    for detail in details:
        rendered = str(detail)
        if used + len(rendered) > max_chars:
            break
        kept.append(detail)
        used += len(rendered)
    return tuple(kept), len(kept) != len(details)
