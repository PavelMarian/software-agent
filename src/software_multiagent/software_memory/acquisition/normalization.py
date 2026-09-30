from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from software_multiagent.software_memory.acquisition.models import (
    ApiOperation,
    Conflict,
    KnowledgeStatus,
    OperationFingerprint,
    stable_id,
)


def operation_key(operation: ApiOperation) -> str:
    return f"{operation.method.upper()} {operation.path.rstrip('/') or '/'}"


def _value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def normalize_operations(
    operations: tuple[ApiOperation, ...],
) -> tuple[tuple[ApiOperation, ...], tuple[Conflict, ...]]:
    grouped: dict[str, list[ApiOperation]] = defaultdict(list)
    for operation in operations:
        grouped[operation_key(operation)].append(operation)

    normalized: list[ApiOperation] = []
    conflicts: list[Conflict] = []
    compared_fields = ("description", "base_urls", "parameters", "request_body_schema", "responses", "authentication")

    for key, candidates in sorted(grouped.items()):
        chosen = max(
            candidates,
            key=lambda item: (
                item.status == KnowledgeStatus.EVIDENCE_SUPPORTED,
                len(item.parameters),
                len(item.responses),
                len(item.description or ""),
            ),
        )
        operation_conflicts: list[Conflict] = []
        for field in compared_fields:
            values_by_key: dict[str, Any] = {}
            evidence = []
            for candidate in candidates:
                value = getattr(candidate, field)
                if value not in (None, (), [], {}):
                    values_by_key[_value_key(value)] = value
                    evidence.extend(candidate.evidence)
            if len(values_by_key) > 1:
                operation_conflicts.append(
                    Conflict(
                        conflict_id=stable_id("conf", key, field, *values_by_key),
                        operation_key=key,
                        field=field,
                        values=tuple(values_by_key.values()),
                        evidence=tuple(
                            {
                                (item.source_id, item.section_id, item.pointer, item.quote): item
                                for item in evidence
                            }.values()
                        ),
                    )
                )
        if operation_conflicts:
            chosen = chosen.model_copy(update={"status": KnowledgeStatus.CONFLICTING})
        elif len(candidates) > 1 and chosen.status == KnowledgeStatus.EVIDENCE_SUPPORTED:
            chosen = chosen.model_copy(update={"status": KnowledgeStatus.CROSS_SOURCE_CONFIRMED})
        normalized.append(chosen)
        conflicts.extend(operation_conflicts)
    return tuple(normalized), tuple(conflicts)


def make_fingerprint(operation: ApiOperation) -> OperationFingerprint:
    required = tuple(
        f"{parameter.location}:{parameter.name}" for parameter in operation.parameters if parameter.required
    )
    optional = tuple(
        f"{parameter.location}:{parameter.name}" for parameter in operation.parameters if not parameter.required
    )
    purpose = (operation.description or operation.name).strip()
    if len(purpose) > 280:
        purpose = purpose[:277].rstrip() + "..."
    return OperationFingerprint(
        operation_id=operation.operation_id,
        method=operation.method,
        path=operation.path,
        purpose=purpose,
        required_parameters=required,
        optional_parameters=optional,
        request_content_types=operation.request_content_types,
        response_statuses=tuple(response.status_code for response in operation.responses),
        authentication=operation.authentication,
    )
