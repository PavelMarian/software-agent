"""Deterministic normalization of immutable evidence into bounded units."""

from __future__ import annotations

import re
from hashlib import sha256

from software_multiagent.software_memory.schema.models import EvidenceRecord, EvidenceSourceKind, stable_id
from software_multiagent.software_memory.extraction.models import (
    Classification,
    EvidenceBlock,
    EvidenceUnit,
    MaterialKind,
)


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE = re.compile(r"^\s*```\s*([\w.+-]*)\s*$")
_CLI = re.compile(r"(?im)(?:^|\n)\s*(?:usage|synopsis)\s*:|(?:^|\s)--?[a-z][\w-]*")
_ERROR = re.compile(r"(?i)\b(error|fatal|exception|failed|failure|exit code|troubleshoot)\b")
_CONFIG = re.compile(r"(?i)\b(configuration|config|schema|manifest|yaml|toml|ini|json schema)\b")
_TUTORIAL = re.compile(r"(?i)\b(tutorial|workflow|step\s+\d+|first .+ then|getting started)\b")
_API = re.compile(r"(?i)\b(openapi|swagger|endpoint|request body|response schema|http (?:get|post|put|patch|delete))\b")
_SOURCE = re.compile(r"(?i)\b(class|function|method|namespace|source code|implementation)\b")
_OPERATION_SENTENCE = re.compile(
    r"(?im)(?:^|[.!?][ \t]+)[ \t]*(?:[A-Za-z_]\w*(?:[ \t]+[A-Za-z_-]+){0,3})[ \t]+"
    r"(?:reads|writes|creates|generates|checks|validates|requires|starts|applies|displays|watches|submits)\b"
)
_SQL = re.compile(r"\b(?:CREATE\s+TABLE|BEGIN|COMMIT|ROLLBACK|SET\s+LOCAL|EXPLAIN|SELECT)\b")


def _clean(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def _blocks(lines: list[str], first_line: int, unit_id: str) -> tuple[EvidenceBlock, ...]:
    result: list[EvidenceBlock] = []
    start = 0
    current_kind = "prose"
    current_language: str | None = None
    in_fence = False

    def append(kind: str, begin: int, end: int, language: str | None = None) -> None:
        content = _clean("\n".join(lines[begin:end]))
        if not content:
            return
        result.append(
            EvidenceBlock(
                block_id=stable_id("block", unit_id, str(first_line + begin), kind),
                block_type=kind,
                content=content,
                language=language,
                start_line=first_line + begin,
                end_line=first_line + end - 1,
            )
        )

    for index, line in enumerate(lines):
        fence = _FENCE.match(line)
        if fence:
            append(current_kind, start, index, current_language)
            if in_fence:
                in_fence = False
                current_kind = "prose"
                current_language = None
            else:
                in_fence = True
                current_kind = "code"
                current_language = fence.group(1) or "text"
            start = index + 1
            continue
        if in_fence or not line.strip():
            continue
        stripped = line.strip()
        kind = (
            "table"
            if stripped.startswith("|") and stripped.endswith("|")
            else "example"
            if re.search(r"(?i)^\s*(?:example|for example)\s*:", line)
            else "prose"
        )
        if kind != current_kind:
            append(current_kind, start, index, current_language)
            start = index
            current_kind = kind
            current_language = None
    append(current_kind, start, len(lines), current_language)
    return tuple(result)


def normalize_evidence(evidence: EvidenceRecord) -> tuple[EvidenceUnit, ...]:
    """Split one evidence record without crossing its source boundary."""
    text = _clean(evidence.content)
    lines = text.split("\n")
    sections: list[tuple[int, int, tuple[str, ...]]] = []
    headings: list[str] = []
    section_start = 0
    section_path: tuple[str, ...] = ()
    in_fence = False
    for index, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING.match(line)
        if not match:
            continue
        if index > section_start:
            sections.append((section_start, index, section_path))
        level = len(match.group(1))
        headings = headings[: level - 1] + [match.group(2).strip()]
        section_path = tuple(headings)
        section_start = index
    sections.append((section_start, len(lines), section_path))

    units: list[EvidenceUnit] = []
    for start, end, path in sections:
        content = _clean("\n".join(lines[start:end]))
        if not content:
            continue
        locator_suffix = "/".join(path) if path else f"lines-{start + 1}-{end}"
        unit_locator = "#".join(value for value in (evidence.locator or "document", locator_suffix))
        unit_id = stable_id("unit", evidence.evidence_id, unit_locator)
        units.append(
            EvidenceUnit(
                unit_id=unit_id,
                evidence_id=evidence.evidence_id,
                software_id=evidence.software_id,
                source_uri=evidence.source_uri,
                source_locator=evidence.locator,
                unit_locator=unit_locator,
                heading_path=path,
                normalized_text=content,
                content_hash=sha256(content.encode("utf-8")).hexdigest(),
                start_line=start + 1,
                end_line=end,
                blocks=_blocks(lines[start:end], start + 1, unit_id),
                source_kind=evidence.source_kind.value,
                source_version=evidence.source_version,
            )
        )
    return tuple(units)


def classify_unit(unit: EvidenceUnit) -> Classification:
    """Assign useful material labels without product-specific rules."""
    text = unit.normalized_text
    source = EvidenceSourceKind(unit.source_kind)
    scored: dict[MaterialKind, tuple[int, str]] = {}

    def add(kind: MaterialKind, score: int, reason: str) -> None:
        previous = scored.get(kind)
        if previous is None or score > previous[0]:
            scored[kind] = (score, reason)

    source_labels = {
        EvidenceSourceKind.OPENAPI: MaterialKind.API,
        EvidenceSourceKind.CLI_HELP: MaterialKind.CLI,
        EvidenceSourceKind.CONFIGURATION_SCHEMA: MaterialKind.CONFIGURATION,
        EvidenceSourceKind.SOURCE_CODE: MaterialKind.SOURCE_CODE,
    }
    if source in source_labels:
        add(source_labels[source], 100, f"source_kind={source.value}")
    for pattern, kind, reason in (
        (_API, MaterialKind.API, "API vocabulary"),
        (_CLI, MaterialKind.CLI, "CLI synopsis or options"),
        (_CONFIG, MaterialKind.CONFIGURATION, "configuration vocabulary"),
        (_TUTORIAL, MaterialKind.TUTORIAL, "ordered instructional language"),
        (_ERROR, MaterialKind.ERROR_REFERENCE, "failure vocabulary"),
        (_SOURCE, MaterialKind.SOURCE_CODE, "source-code vocabulary"),
        (_OPERATION_SENTENCE, MaterialKind.CLI, "documented operation sentence"),
        (_SQL, MaterialKind.CLI, "executable SQL statement"),
    ):
        if pattern.search(text):
            add(kind, 70 if kind == MaterialKind.ERROR_REFERENCE else 60, reason)
    if not scored:
        add(MaterialKind.IRRELEVANT, 20, "no operational material detected")
    ordered = sorted(scored, key=lambda item: (-scored[item][0], item.value))
    top_score = scored[ordered[0]][0]
    return Classification(
        unit_id=unit.unit_id,
        primary=ordered[0],
        labels=tuple(ordered),
        confidence=min(0.99, top_score / 100),
        reasons=tuple(scored[item][1] for item in ordered),
    )
