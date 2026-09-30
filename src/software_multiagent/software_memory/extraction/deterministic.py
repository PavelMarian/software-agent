"""Conservative extractors for machine-readable and recurring documentation forms."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from software_multiagent.software_memory.extraction.models import (
    ClaimAnchor,
    Classification,
    EntityDraft,
    EvidenceUnit,
    MaterialKind,
    OperationDraft,
    SourceConfirmation,
    UnitExtraction,
)
from software_multiagent.software_memory.schema.models import FailureSignature, OperationParameter, StatePredicate, stable_id


def anchor(unit: EvidenceUnit, quote: str) -> ClaimAnchor:
    start = unit.normalized_text.find(quote)
    if start < 0:
        raise ValueError("claim anchor must be an exact substring of its evidence unit")
    return ClaimAnchor(
        evidence_id=unit.evidence_id,
        unit_id=unit.unit_id,
        locator=unit.unit_locator,
        quote=quote,
        start_offset=start,
        end_offset=start + len(quote),
    )


class DeterministicExtractor(Protocol):
    name: str

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool: ...

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction: ...


def _empty(unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
    return UnitExtraction(unit_id=unit.unit_id, classification=classification)


@dataclass(frozen=True)
class OpenApiExtractor:
    name: str = "openapi"

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool:
        return MaterialKind.API in classification.labels

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
        try:
            document = json.loads(unit.normalized_text)
        except json.JSONDecodeError:
            return _empty(unit, classification)
        if not isinstance(document, dict) or not isinstance(document.get("paths"), dict):
            return _empty(unit, classification)
        operations: list[OperationDraft] = []
        methods = {"get", "post", "put", "patch", "delete", "head", "options"}
        for path, path_item in document["paths"].items():
            if not isinstance(path_item, dict):
                continue
            for method, operation in path_item.items():
                if method.lower() not in methods or not isinstance(operation, dict):
                    continue
                quote = json.dumps({method: operation}, ensure_ascii=False, sort_keys=True)
                # Pretty/full JSON does not contain the canonical fragment. Anchor to an exact key.
                quoted_key = f'"{path}"'
                if quoted_key not in unit.normalized_text:
                    quoted_key = method
                parameters = []
                for value in (*path_item.get("parameters", ()), *operation.get("parameters", ())):
                    if not isinstance(value, dict) or "$ref" in value:
                        continue
                    schema = value.get("schema") if isinstance(value.get("schema"), dict) else {}
                    parameters.append(
                        OperationParameter(
                            name=str(value.get("name", "unknown")),
                            required=bool(value.get("required", False)),
                            parameter_type=schema.get("type"),
                            description=value.get("description"),
                            default=schema.get("default"),
                            constraints={
                                key: schema[key]
                                for key in ("enum", "minimum", "maximum", "pattern")
                                if key in schema
                            },
                            evidence_ids=(unit.evidence_id,),
                        )
                    )
                operation_id = str(operation.get("operationId") or f"{method.lower()} {path}")
                operations.append(
                    OperationDraft(
                        source_unit_id=unit.unit_id,
                        operation_key=operation_id,
                        name=operation_id,
                        interface="api",
                        purpose=str(operation.get("summary") or operation.get("description") or ""),
                        invocation=f"{method.upper()} {path}",
                        parameters=tuple(parameters),
                        inputs=("request",),
                        outputs=tuple(str(code) for code in operation.get("responses", {}).keys()),
                        evidence_ids=(unit.evidence_id,),
                        anchors=(anchor(unit, quoted_key),),
                        extraction_method=self.name,
                        extraction_confidence=0.98,
                        metadata={"http_method": method.upper(), "path": path},
                    )
                )
                del quote
        return UnitExtraction(
            unit_id=unit.unit_id,
            classification=classification,
            operations=tuple(operations),
        )


@dataclass(frozen=True)
class CliExtractor:
    name: str = "cli"
    _usage = re.compile(r"(?im)^\s*(?:usage|synopsis)\s*:\s*(?P<command>[^\n]+)")
    _documented = re.compile(
        r"(?im)(?:^|[.!?][ \t]+)[ \t]*(?P<command>[A-Za-z_]\w*(?:[ \t]+(?!(?:reads|writes|creates|generates|checks|validates|requires|starts|applies|displays|watches|submits)\b)[A-Za-z_-]+){0,3})[ \t]+"
        r"(?:reads|writes|creates|generates|checks|validates|requires|starts|applies|displays|watches|submits)\b"
    )
    _option = re.compile(
        r"(?m)^\s*(?P<flags>(?:-[\w?],?\s*)?--?[\w][\w-]*)(?:[ =](?P<value><[^>]+>|[A-Z][A-Z_-]+))?\s*(?P<description>.*)$"
    )

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool:
        return MaterialKind.CLI in classification.labels

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
        match = self._usage.search(unit.normalized_text)
        documented_matches = tuple(self._documented.finditer(unit.normalized_text))
        documented = next(
            (
                candidate
                for candidate in documented_matches
                if len(candidate.group("command").split()) == 1
                and candidate.group("command").casefold() not in {"a", "an", "the", "this"}
            ),
            documented_matches[0] if documented_matches else None,
        )
        kubectl = re.search(r"\bkubectl(?:\s+[a-z][\w-]*){1,3}(?:\s+--[\w=-]+)?", unit.normalized_text)
        selected = match or documented or kubectl
        if not selected:
            return _empty(unit, classification)
        if match or documented:
            invocation = " ".join(selected.group("command").split())
        else:
            invocation = " ".join(selected.group(0).split())
        command = invocation.split()[0]
        parameters = []
        for option in self._option.finditer(unit.normalized_text):
            flags = option.group("flags")
            name = flags.split("--")[-1] if "--" in flags else flags.split("-")[-1]
            parameters.append(
                OperationParameter(
                    name=name,
                    required=False,
                    parameter_type="string" if option.group("value") else "boolean",
                    description=option.group("description").strip() or None,
                    evidence_ids=(unit.evidence_id,),
                )
            )
        operation = OperationDraft(
            source_unit_id=unit.unit_id,
            operation_key=invocation,
            name=command,
            interface="cli",
            purpose="",
            invocation=invocation,
            parameters=tuple(parameters),
            evidence_ids=(unit.evidence_id,),
            anchors=(anchor(unit, selected.group(0).strip()),),
            extraction_method=self.name,
            extraction_confidence=0.94,
            metadata={
                "examples": [block.content for block in unit.blocks if block.block_type == "example"]
            },
        )
        return UnitExtraction(
            unit_id=unit.unit_id,
            classification=classification,
            operations=(operation,),
        )


@dataclass(frozen=True)
class ApplicationHeaderExtractor:
    """Extract executable applications declared in source/documentation headers."""

    name: str = "application_header"
    _header = re.compile(
        r"(?ims)^[ \t]*(?:Application|Utility|Command)[ \t]*\n"
        r"[ \t]*(?P<command>[A-Za-z_]\w*)[ \t]*\n+"
        r"[ \t]*Description[ \t]*\n[ \t]*(?P<description>[^\n]+)"
    )

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool:
        return bool(self._header.search(unit.normalized_text))

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
        match = self._header.search(unit.normalized_text)
        if match is None:
            return _empty(unit, classification)
        command = match.group("command")
        operation = OperationDraft(
            source_unit_id=unit.unit_id,
            operation_key=command,
            name=command,
            interface="cli",
            purpose=match.group("description").strip(),
            invocation=command,
            evidence_ids=(unit.evidence_id,),
            anchors=(anchor(unit, match.group(0).strip()),),
            extraction_method=self.name,
            extraction_confidence=0.98,
        )
        return UnitExtraction(
            unit_id=unit.unit_id,
            classification=classification,
            operations=(operation,),
        )


@dataclass(frozen=True)
class ToolListingExtractor:
    """Extract command names from reference lists of utilities, solvers, and scripts."""

    name: str = "tool_listing"
    _entry = re.compile(
        r"(?im)^[ \t]*(?P<command>[A-Za-z_]\w*)[ \t]+"
        r"(?P<kind>utility|solver|script|application)\b"
        r"(?:[ \t]*[:,-][ \t]*(?P<description>[^\n]+))?"
    )

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool:
        return bool(self._entry.search(unit.normalized_text))

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
        operations = []
        for match in self._entry.finditer(unit.normalized_text):
            command = match.group("command")
            operations.append(
                OperationDraft(
                    source_unit_id=unit.unit_id,
                    operation_key=command,
                    name=command,
                    interface="cli",
                    purpose=(match.group("description") or "").strip(),
                    invocation=command,
                    evidence_ids=(unit.evidence_id,),
                    anchors=(anchor(unit, match.group(0).strip()),),
                    extraction_method=self.name,
                    extraction_confidence=0.92,
                    metadata={"documented_kind": match.group("kind").casefold()},
                )
            )
        return UnitExtraction(
            unit_id=unit.unit_id,
            classification=classification,
            operations=tuple(operations),
        )


@dataclass(frozen=True)
class SchemaExtractor:
    name: str = "json_schema"

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool:
        return MaterialKind.CONFIGURATION in classification.labels

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
        try:
            document = json.loads(unit.normalized_text)
        except json.JSONDecodeError:
            return _empty(unit, classification)
        if not isinstance(document, dict):
            return _empty(unit, classification)
        is_schema = "$schema" in document or "properties" in document
        is_manifest = "apiVersion" in document and "kind" in document
        is_structured_configuration = unit.source_kind == "configuration_schema"
        if not (is_schema or is_manifest or is_structured_configuration):
            return _empty(unit, classification)
        name = str(document.get("title") or document.get("kind") or "configuration document")
        required = tuple(str(value) for value in document.get("required", ()))
        quote = next(
            (candidate for candidate in (f'"title"', f'"kind"', f'"$schema"') if candidate in unit.normalized_text),
            unit.normalized_text[: min(80, len(unit.normalized_text))],
        )
        entity = EntityDraft(
            source_unit_id=unit.unit_id,
            entity_key=name,
            entity_type="configuration",
            name=name,
            attributes={
                "format": "JSON Schema" if is_schema else "JSON/YAML manifest",
                "required_fields": required,
                "properties": tuple(document.get("properties", {}).keys()),
                "top_level_keys": tuple(document.keys()),
            },
            evidence_ids=(unit.evidence_id,),
            anchors=(anchor(unit, quote),),
            extraction_method=self.name,
            extraction_confidence=0.98,
        )
        return UnitExtraction(
            unit_id=unit.unit_id,
            classification=classification,
            entities=(entity,),
        )


@dataclass(frozen=True)
class ArtifactExtractor:
    name: str = "artifact_paths"
    _path = re.compile(
        r"(?<![\w:/.-])(?P<path>(?:\.?\.?/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_*?.-]+)(?![\w/.-])"
    )

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool:
        return classification.primary != MaterialKind.IRRELEVANT

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
        entities: list[EntityDraft] = []
        for path in dict.fromkeys(match.group("path") for match in self._path.finditer(unit.normalized_text)):
            if path.startswith(("http/", "https/")):
                continue
            quote = path
            suffix = PurePosixPath(path).suffix
            entities.append(
                EntityDraft(
                    source_unit_id=unit.unit_id,
                    entity_key=path,
                    entity_type="configuration" if suffix in {".json", ".yaml", ".yml", ".toml", ".ini"} or "Dict" in path else "artifact",
                    name=path,
                    attributes={"path": path},
                    evidence_ids=(unit.evidence_id,),
                    anchors=(anchor(unit, quote),),
                    extraction_method=self.name,
                    extraction_confidence=0.9,
                )
            )
        return UnitExtraction(unit_id=unit.unit_id, classification=classification, entities=tuple(entities))


@dataclass(frozen=True)
class FailureExtractor:
    name: str = "failure_signatures"
    _line = re.compile(r"(?im)^.*\b(?:error|fatal|exception|failed|failure)\b.*$")

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool:
        return MaterialKind.ERROR_REFERENCE in classification.labels

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
        failures = []
        for match in self._line.finditer(unit.normalized_text):
            value = match.group(0).strip()
            failures.append(
                FailureSignature(
                    signature_id=stable_id("failure", unit.software_id, value),
                    pattern=value,
                    category="unknown",
                    description="Extracted failure signature; category requires validation.",
                    evidence_ids=(unit.evidence_id,),
                )
            )
        return UnitExtraction(unit_id=unit.unit_id, classification=classification, failures=tuple(failures))


@dataclass(frozen=True)
class SourceConfirmationExtractor:
    """Emit reviewable confirmations; this hook never promotes trust by itself."""

    name: str = "source_confirmation"
    _symbol = re.compile(r"(?m)\b(?:class|def|function)\s+([A-Za-z_]\w*)")

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool:
        return MaterialKind.SOURCE_CODE in classification.labels

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
        confirmations = []
        for match in self._symbol.finditer(unit.normalized_text):
            symbol = match.group(1)
            confirmations.append(
                SourceConfirmation(
                    unit_id=unit.unit_id,
                    evidence_id=unit.evidence_id,
                    symbol=symbol,
                    operation_key=symbol,
                    anchor=anchor(unit, match.group(0)),
                )
            )
        return UnitExtraction(
            unit_id=unit.unit_id,
            classification=classification,
            source_confirmations=tuple(confirmations),
        )


@dataclass(frozen=True)
class SqlExtractor:
    name: str = "sql"
    _statement = re.compile(
        r"(?m)\b(?P<command>CREATE\s+TABLE|SET\s+LOCAL(?:\s+[A-Za-z_]\w*)?|BEGIN|COMMIT|ROLLBACK|EXPLAIN(?:\s+ANALYZE)?|SELECT)\b"
    )

    def supports(self, unit: EvidenceUnit, classification: Classification) -> bool:
        return any(self._statement.finditer(unit.normalized_text))

    def extract(self, unit: EvidenceUnit, classification: Classification) -> UnitExtraction:
        operations = []
        for value in dict.fromkeys(match.group("command") for match in self._statement.finditer(unit.normalized_text)):
            normalized = " ".join(value.upper().split())
            operations.append(
                OperationDraft(
                    source_unit_id=unit.unit_id,
                    operation_key=normalized,
                    name=normalized,
                    interface="sql",
                    purpose="documented SQL operation",
                    invocation=normalized,
                    evidence_ids=(unit.evidence_id,),
                    anchors=(anchor(unit, value),),
                    extraction_method=self.name,
                    extraction_confidence=0.9,
                )
            )
        return UnitExtraction(
            unit_id=unit.unit_id,
            classification=classification,
            operations=tuple(operations),
        )


DEFAULT_EXTRACTORS: tuple[DeterministicExtractor, ...] = (
    OpenApiExtractor(),
    SqlExtractor(),
    ApplicationHeaderExtractor(),
    ToolListingExtractor(),
    CliExtractor(),
    SchemaExtractor(),
    ArtifactExtractor(),
    FailureExtractor(),
    SourceConfirmationExtractor(),
)


def extract_deterministically(
    unit: EvidenceUnit,
    classification: Classification,
    extractors: tuple[DeterministicExtractor, ...] = DEFAULT_EXTRACTORS,
) -> UnitExtraction:
    entities: list[EntityDraft] = []
    operations: list[OperationDraft] = []
    failures: list[FailureSignature] = []
    confirmations: list[SourceConfirmation] = []
    warnings: list[str] = []
    for extractor in extractors:
        if not extractor.supports(unit, classification):
            continue
        try:
            extracted = extractor.extract(unit, classification)
        except (TypeError, ValueError) as exc:
            warnings.append(f"{extractor.name}: {exc}")
            continue
        entities.extend(extracted.entities)
        operations.extend(extracted.operations)
        failures.extend(extracted.failures)
        confirmations.extend(extracted.source_confirmations)
        warnings.extend(extracted.warnings)
    if failures:
        operations = [
            operation.model_copy(
                update={
                    "failure_signatures": tuple(
                        dict.fromkeys((*operation.failure_signatures, *failures))
                    )
                }
            )
            for operation in operations
        ]
    return UnitExtraction(
        unit_id=unit.unit_id,
        classification=classification,
        entities=tuple(entities),
        operations=tuple(operations),
        failures=tuple(failures),
        source_confirmations=tuple(confirmations),
        warnings=tuple(warnings),
    )
