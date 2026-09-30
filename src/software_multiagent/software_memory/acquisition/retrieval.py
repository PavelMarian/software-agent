from __future__ import annotations

import re

from software_multiagent.software_memory.acquisition.models import ApiCatalog, ApiOperation, OperationFingerprint, SourceSection


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_./{}-]+", value.lower()) if len(token) > 1}


class CatalogReader:
    """Compact retrieval API that avoids placing the complete corpus in an agent context."""

    def __init__(self, catalog: ApiCatalog) -> None:
        self.catalog = catalog
        self._operations = {operation.operation_id: operation for operation in catalog.operations}
        self._sections = {
            section.section_id: section for source in catalog.sources for section in source.sections
        }

    def list_operations(
        self, *, tag: str | None = None, method: str | None = None
    ) -> tuple[OperationFingerprint, ...]:
        values = self.catalog.fingerprints
        if method:
            values = tuple(item for item in values if item.method == method.upper())
        if tag:
            operation_ids = {
                operation.operation_id
                for operation in self.catalog.operations
                if tag.lower() in {value.lower() for value in operation.tags}
            }
            values = tuple(item for item in values if item.operation_id in operation_ids)
        return values

    def get_operation(self, operation_id: str) -> ApiOperation | None:
        return self._operations.get(operation_id)

    def read_source_section(self, section_id: str) -> SourceSection | None:
        return self._sections.get(section_id)

    def search(self, query: str, *, limit: int = 8) -> tuple[OperationFingerprint, ...]:
        query_tokens = _tokens(query)
        ranked: list[tuple[float, OperationFingerprint]] = []
        for fingerprint in self.catalog.fingerprints:
            haystack = " ".join(
                (
                    fingerprint.method,
                    fingerprint.path,
                    fingerprint.purpose,
                    *fingerprint.required_parameters,
                    *fingerprint.optional_parameters,
                )
            )
            value_tokens = _tokens(haystack)
            overlap = len(query_tokens & value_tokens)
            exact_bonus = 2 if query.lower() in haystack.lower() else 0
            if overlap or exact_bonus:
                ranked.append((overlap + exact_bonus + overlap / max(1, len(value_tokens)), fingerprint))
        ranked.sort(key=lambda item: (-item[0], item[1].operation_id))
        return tuple(item for _score, item in ranked[: max(0, limit)])
