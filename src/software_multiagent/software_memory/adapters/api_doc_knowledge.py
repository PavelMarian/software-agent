"""Import the stable JSON/JSONL output of ``api_doc_knowledge``.

The adapter deliberately does not import that package.  The on-disk catalog is the
boundary, so either package can evolve and be tested independently.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping

from pydantic import BaseModel, ConfigDict, Field

from software_multiagent.software_memory.models import (
    ActorKind,
    Entity,
    EvidenceRecord,
    EvidenceSourceKind,
    KnowledgeConflict,
    KnowledgeStatus,
    OperationContract,
    OperationParameter,
    SoftwareIdentity,
    StatePredicate,
    VersionScope,
    stable_id,
)
from software_multiagent.software_memory.service import MemoryService
from software_multiagent.software_memory.storage import SQLiteMemoryStore


ADAPTER_NAME = "api_doc_knowledge"


class ImportCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_sources: int = 0
    input_sections: int = 0
    input_operations: int = 0
    evidence_created: int = 0
    evidence_skipped: int = 0
    entities_created: int = 0
    entities_updated: int = 0
    entities_skipped: int = 0
    contracts_created: int = 0
    contracts_updated: int = 0
    contracts_skipped: int = 0
    reconciliation_opened: int = 0
    reconciliation_skipped: int = 0


class ApiDocImportReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    adapter: str = ADAPTER_NAME
    import_mode: str = "operations_and_evidence"
    catalog_directory: str
    software_id: str
    product: str
    requested_version: str
    selected_ref: str | None = None
    commit_sha: str | None = None
    version_alignment: str
    knowledge_status: KnowledgeStatus
    import_namespace: str
    manifest_counts: dict[str, int]
    actual_counts: dict[str, int]
    counts_reconciled: bool
    counts: ImportCounts
    sample_evidence_ids: tuple[str, ...] = ()


class ApiDocKnowledgeAdapter:
    """Convert evidence catalogs and extracted HTTP API operations into memory."""

    def __init__(self, service: MemoryService) -> None:
        self.service = service
        self.store = service.store

    def import_catalog(
        self,
        catalog_directory: str | Path,
        *,
        product: str,
        vendor: str | None = None,
        edition: str | None = None,
        distribution: str | None = None,
        platform: str | None = None,
        import_operations: bool = True,
        report_path: str | Path | None = None,
    ) -> ApiDocImportReport:
        directory = Path(catalog_directory).resolve()
        manifest = _read_json(directory / "manifest.json")
        snapshot_path = directory / "repository_snapshot.json"
        snapshot = _read_json(snapshot_path) if snapshot_path.exists() else {}
        sources = tuple(_read_jsonl(directory / "sources.jsonl"))
        sections = tuple(_read_jsonl(directory / "sections.jsonl"))
        operations = tuple(_read_jsonl(directory / "operations.jsonl"))
        actual = {
            "sources": len(sources),
            "sections": len(sections),
            "operations": len(operations),
        }
        declared = {
            key: int(manifest.get("counts", {}).get(key, 0))
            for key in ("sources", "sections", "operations")
        }
        if declared != actual:
            raise ValueError(f"catalog counts do not reconcile: manifest={declared}, actual={actual}")

        requested_version = str(snapshot.get("requested_version") or manifest.get("version") or "unknown")
        alignment = str(
            snapshot.get("version_alignment")
            or manifest.get("corpus", {}).get("version_alignment")
            or "unconfirmed"
        )
        status = (
            KnowledgeStatus.VERSION_UNCONFIRMED
            if alignment == "unconfirmed"
            else KnowledgeStatus.EXTRACTED
        )
        software = SoftwareIdentity.create(
            product,
            requested_version,
            vendor=vendor,
            edition=edition,
            distribution=distribution,
            platform=platform,
            metadata={
                "requested_version": requested_version,
                "selected_ref": snapshot.get("selected_ref"),
                "commit_sha": snapshot.get("commit_sha"),
                "version_alignment": alignment,
            },
        )
        root_uri = str(
            manifest.get("corpus", {}).get("root_url")
            or snapshot.get("project_url")
            or directory.as_uri()
        )
        namespace = stable_id("import", ADAPTER_NAME, software.software_id, root_uri)
        version_scope = _version_scope(requested_version, snapshot, alignment)
        source_by_id = {str(item["source_id"]): item for item in sources}
        section_by_id = {str(item["section_id"]): item for item in sections}
        for section in sections:
            if str(section["source_id"]) not in source_by_id:
                raise ValueError(f"section references unknown source: {section['section_id']}")

        mutable_counts = {name: 0 for name in ImportCounts.model_fields}
        mutable_counts.update(
            input_sources=len(sources),
            input_sections=len(sections),
            input_operations=len(operations),
        )
        sample_ids: list[str] = []
        current_entity_ids: set[str] = set()
        current_contract_ids: set[str] = set()

        with self.service.batch():
            self.service.register_software(software)
            for evidence in self._catalog_evidence(
                software,
                manifest,
                snapshot,
                snapshot_path,
                sources,
                sections,
                source_by_id,
                requested_version,
                alignment,
                namespace,
            ):
                self._record_evidence(evidence, mutable_counts)
                if len(sample_ids) < 5:
                    sample_ids.append(evidence.evidence_id)

            if import_operations:
                for operation in operations:
                    reference_ids = self._operation_evidence(
                        software,
                        operation,
                        source_by_id,
                        section_by_id,
                        requested_version,
                        namespace,
                        mutable_counts,
                    )
                    entity, contract = _operation_knowledge(
                        software,
                        operation,
                        reference_ids,
                        status,
                        version_scope,
                        namespace,
                    )
                    current_entity_ids.add(entity.entity_id)
                    current_contract_ids.add(contract.contract_id)
                    self._upsert_entity(entity, mutable_counts)
                    self._upsert_contract(contract, mutable_counts)

                self._reconcile_missing(
                    software.software_id,
                    namespace,
                    current_entity_ids,
                    current_contract_ids,
                    mutable_counts,
                )

        report = ApiDocImportReport(
            import_mode=(
                "operations_and_evidence" if import_operations else "evidence_only"
            ),
            catalog_directory=str(directory),
            software_id=software.software_id,
            product=product,
            requested_version=requested_version,
            selected_ref=_optional_string(snapshot.get("selected_ref")),
            commit_sha=_optional_string(snapshot.get("commit_sha")),
            version_alignment=alignment,
            knowledge_status=status,
            import_namespace=namespace,
            manifest_counts=declared,
            actual_counts=actual,
            counts_reconciled=True,
            counts=ImportCounts(**mutable_counts),
            sample_evidence_ids=tuple(sample_ids),
        )
        if report_path is not None:
            _atomic_json(Path(report_path), report.model_dump(mode="json"))
        return report

    def _catalog_evidence(
        self,
        software: SoftwareIdentity,
        manifest: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        snapshot_path: Path,
        sources: tuple[Mapping[str, Any], ...],
        sections: tuple[Mapping[str, Any], ...],
        source_by_id: Mapping[str, Mapping[str, Any]],
        source_version: str,
        alignment: str,
        namespace: str,
    ) -> Iterator[EvidenceRecord]:
        common = {
            "import_adapter": ADAPTER_NAME,
            "import_namespace": namespace,
            "version_alignment": alignment,
        }
        for source in sources:
            content = str(source["raw_text"])
            calculated = sha256(content.encode("utf-8")).hexdigest()
            declared = str(source.get("content_hash") or calculated)
            # api_doc_knowledge hashes the downloaded source bytes before parsing;
            # EvidenceRecord hashes the normalized text. Preserve both rather than
            # incorrectly requiring the two representations to have the same digest.
            record = EvidenceRecord.create(
                software_id=software.software_id,
                source_kind=EvidenceSourceKind.DOCUMENTATION,
                source_uri=str(source.get("canonical_url") or source["url"]),
                locator=f"source:{source['source_id']}",
                content=content,
                source_version=source_version,
                authoritative=True,
                metadata={
                    **common,
                    "catalog_record_type": "source_document",
                    "source_id": source["source_id"],
                    "original_content_hash": declared,
                    "url": source.get("url"),
                    "canonical_url": source.get("canonical_url"),
                    "title": source.get("title"),
                    "page_type": source.get("page_type"),
                    "content_type": source.get("content_type"),
                    "http_status": source.get("http_status"),
                },
            )
            record = record.model_copy(
                update={
                    "evidence_id": stable_id(
                        "evidence_source",
                        software.software_id,
                        str(source.get("canonical_url") or source["url"]),
                        str(source["source_id"]),
                        declared,
                        calculated,
                    )
                }
            )
            retrieved = source.get("retrieved_at")
            if retrieved:
                record = record.model_copy(update={"retrieved_at": _parse_datetime(retrieved)})
            yield record
        for section in sections:
            source = source_by_id[str(section["source_id"])]
            record = EvidenceRecord.create(
                software_id=software.software_id,
                source_kind=EvidenceSourceKind.DOCUMENTATION,
                source_uri=str(source.get("canonical_url") or source["url"]),
                locator=str(section["section_id"]),
                content=str(section["content"]),
                source_version=source_version,
                authoritative=True,
                metadata={
                    **common,
                    "catalog_record_type": "source_section",
                    "section_id": section["section_id"],
                    "source_id": section["source_id"],
                    "heading_path": section.get("heading_path", []),
                    "ordinal": section.get("ordinal", 0),
                },
            )
            retrieved = source.get("retrieved_at")
            if retrieved:
                record = record.model_copy(update={"retrieved_at": _parse_datetime(retrieved)})
            yield record
        if snapshot:
            raw_snapshot = snapshot_path.read_text(encoding="utf-8")
            project_url = str(snapshot.get("project_url") or manifest.get("corpus", {}).get("root_url") or snapshot_path.as_uri())
            record = EvidenceRecord.create(
                software_id=software.software_id,
                source_kind=EvidenceSourceKind.DOCUMENTATION,
                source_uri=project_url,
                locator="repository_snapshot.json",
                content=raw_snapshot,
                source_version=source_version,
                authoritative=True,
                metadata={
                    **common,
                    "catalog_record_type": "repository_snapshot",
                    "provider": snapshot.get("provider"),
                    "project_id": snapshot.get("project_id"),
                    "project_path": snapshot.get("project_path"),
                    "selected_ref": snapshot.get("selected_ref"),
                    "commit_sha": snapshot.get("commit_sha"),
                    "requested_version": snapshot.get("requested_version"),
                    "repository_snapshot": dict(snapshot),
                },
            )
            if manifest.get("created_at"):
                record = record.model_copy(update={"retrieved_at": _parse_datetime(manifest["created_at"])})
            yield record

    def _operation_evidence(
        self,
        software: SoftwareIdentity,
        operation: Mapping[str, Any],
        source_by_id: Mapping[str, Mapping[str, Any]],
        section_by_id: Mapping[str, Mapping[str, Any]],
        source_version: str,
        namespace: str,
        counts: dict[str, int],
    ) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        groups: list[tuple[str, list[Mapping[str, Any]]]] = [
            ("operation", list(operation.get("evidence") or [])),
        ]
        for index, parameter in enumerate(operation.get("parameters") or []):
            groups.append((f"parameter:{index}", list(parameter.get("evidence") or [])))
        for index, response in enumerate(operation.get("responses") or []):
            groups.append((f"response:{index}", list(response.get("evidence") or [])))
        for key, references in groups:
            ids: list[str] = []
            for index, reference in enumerate(references):
                content = str(reference.get("quote") or "")
                section_id = _optional_string(reference.get("section_id"))
                if not content and section_id and section_id in section_by_id:
                    content = str(section_by_id[section_id]["content"])
                if not content:
                    content = json.dumps(reference, ensure_ascii=False, sort_keys=True)
                source_id = str(reference.get("source_id") or "unknown")
                source = source_by_id.get(source_id, {})
                source_url = str(reference.get("source_url") or source.get("canonical_url") or source.get("url") or "unknown:")
                locator = str(reference.get("pointer") or section_id or f"{operation['operation_id']}:{key}:{index}")
                digest = sha256(content.encode("utf-8")).hexdigest()
                evidence = EvidenceRecord(
                    evidence_id=stable_id(
                        "evidence_ref",
                        software.software_id,
                        source_url,
                        locator,
                        digest,
                        str(bool(reference.get("verified", False))),
                    ),
                    software_id=software.software_id,
                    source_kind=EvidenceSourceKind.OPENAPI,
                    source_uri=source_url,
                    locator=locator,
                    content=content,
                    content_hash=digest,
                    retrieved_at=_reference_time(reference, source),
                    source_version=source_version,
                    authoritative=True,
                    metadata={
                        "import_adapter": ADAPTER_NAME,
                        "import_namespace": namespace,
                        "catalog_record_type": "evidence_ref",
                        "source_id": source_id,
                        "section_id": section_id,
                        "source_url": source_url,
                        "heading_path": reference.get("heading_path", []),
                        "quote": reference.get("quote", ""),
                        "pointer": reference.get("pointer"),
                        "verified": bool(reference.get("verified", False)),
                    },
                )
                self._record_evidence(evidence, counts)
                ids.append(evidence.evidence_id)
            result[key] = tuple(ids)
        return result

    def _record_evidence(self, evidence: EvidenceRecord, counts: dict[str, int]) -> None:
        existing = self.store.get_evidence(evidence.evidence_id)
        if existing is not None:
            if existing.model_dump(exclude={"retrieved_at"}) != evidence.model_dump(
                exclude={"retrieved_at"}
            ):
                raise ValueError(f"immutable evidence collision: {evidence.evidence_id}")
            counts["evidence_skipped"] += 1
            return
        self.service.record_evidence(evidence)
        counts["evidence_created"] += 1

    def _upsert_entity(self, item: Entity, counts: dict[str, int]) -> None:
        existing = self.store.get_entity(item.entity_id)
        if existing is None:
            self.service.record_entity(item)
            counts["entities_created"] += 1
        elif existing.attributes.get("semantic_hash") == item.attributes.get("semantic_hash"):
            counts["entities_skipped"] += 1
        else:
            self.service.revise_from_source(
                item,
                actor_id=ADAPTER_NAME,
                reason="source API operation changed semantically",
            )
            counts["entities_updated"] += 1

    def _upsert_contract(self, item: OperationContract, counts: dict[str, int]) -> None:
        existing = self.store.get_contract(item.contract_id)
        if existing is None:
            self.service.record_contract(item)
            counts["contracts_created"] += 1
        elif existing.metadata.get("semantic_hash") == item.metadata.get("semantic_hash"):
            counts["contracts_skipped"] += 1
        else:
            self.service.revise_from_source(
                item,
                actor_id=ADAPTER_NAME,
                reason="source API operation changed semantically",
            )
            counts["contracts_updated"] += 1

    def _reconcile_missing(
        self,
        software_id: str,
        namespace: str,
        current_entity_ids: set[str],
        current_contract_ids: set[str],
        counts: dict[str, int],
    ) -> None:
        candidates: list[tuple[str, str, tuple[str, ...]]] = []
        candidates.extend(
            ("entity", item.entity_id, item.evidence_ids)
            for item in self.store.list_entities(software_id)
            if item.attributes.get("import_namespace") == namespace
            and item.entity_id not in current_entity_ids
        )
        candidates.extend(
            ("contract", item.contract_id, item.evidence_ids)
            for item in self.store.list_contracts(software_id)
            if item.metadata.get("import_namespace") == namespace
            and item.contract_id not in current_contract_ids
        )
        for kind, item_id, evidence_ids in candidates:
            conflict_id = stable_id("conflict", namespace, "missing_from_source", kind, item_id)
            if self.store.get_conflict(conflict_id) is not None:
                counts["reconciliation_skipped"] += 1
                continue
            self.service.open_conflict(
                KnowledgeConflict(
                    conflict_id=conflict_id,
                    software_id=software_id,
                    item_kind=kind,
                    item_id=item_id,
                    reason="knowledge item disappeared from the latest source catalog; review before deprecation or retraction",
                    evidence_ids=evidence_ids,
                ),
                actor_kind=ActorKind.INGESTOR,
            )
            counts["reconciliation_opened"] += 1


def _operation_knowledge(
    software: SoftwareIdentity,
    operation: Mapping[str, Any],
    evidence: Mapping[str, tuple[str, ...]],
    status: KnowledgeStatus,
    version_scope: VersionScope,
    namespace: str,
) -> tuple[Entity, OperationContract]:
    key = str(operation.get("operation_id") or f"{operation.get('method')} {operation.get('path')}")
    entity_id = stable_id("entity", software.software_id, namespace, "api_operation", key)
    all_evidence = tuple(
        dict.fromkeys(value for group in evidence.values() for value in group)
    )
    entity_payload = {
        "operation_id": key,
        "method": str(operation.get("method") or "").upper(),
        "path": str(operation.get("path") or ""),
        "base_urls": tuple(operation.get("base_urls") or ()),
        "tags": tuple(operation.get("tags") or ()),
        "deprecated": bool(operation.get("deprecated", False)),
        "description": str(operation.get("description") or ""),
    }
    semantic_hash = _hash_payload(entity_payload)
    entity = Entity(
        entity_id=entity_id,
        software_id=software.software_id,
        entity_type="api_operation",
        name=str(operation.get("name") or key),
        summary=str(operation.get("description") or ""),
        aliases=(key, f"{entity_payload['method']} {entity_payload['path']}"),
        attributes={
            **entity_payload,
            "import_adapter": ADAPTER_NAME,
            "import_namespace": namespace,
            "source_operation_key": key,
            "semantic_hash": semantic_hash,
        },
        evidence_ids=all_evidence,
        version_scope=version_scope,
        status=status,
    )
    parameters = tuple(
        OperationParameter(
            name=str(item.get("name") or "unnamed"),
            required=bool(item.get("required", False)),
            parameter_type=_optional_string(item.get("schema_type")),
            description=_optional_string(item.get("description")),
            default=item.get("default"),
            constraints={
                **dict(item.get("constraints") or {}),
                "location": item.get("location", "unknown"),
                "example": item.get("example"),
                "enum": item.get("enum", []),
            },
            evidence_ids=evidence.get(f"parameter:{index}", ()),
        )
        for index, item in enumerate(operation.get("parameters") or ())
    )
    responses = tuple(operation.get("responses") or ())
    success_signals = tuple(
        StatePredicate(
            predicate="http_status",
            subject=entity_payload["path"],
            value=str(response.get("status_code")),
        )
        for response in responses
        if str(response.get("status_code", "")).startswith("2")
    )
    contract_payload = {
        "name": str(operation.get("name") or key),
        "purpose": str(operation.get("description") or ""),
        "method": entity_payload["method"],
        "path": entity_payload["path"],
        "parameters": [item.model_dump(mode="json") for item in parameters],
        "request_body_schema": operation.get("request_body_schema"),
        "request_content_types": operation.get("request_content_types", []),
        "responses": responses,
        "authentication": operation.get("authentication", []),
        "deprecated": entity_payload["deprecated"],
    }
    contract = OperationContract(
        contract_id=stable_id("contract", software.software_id, namespace, key),
        software_id=software.software_id,
        name=str(operation.get("name") or key),
        interface="http_api",
        purpose=str(operation.get("description") or ""),
        parameters=parameters,
        inputs=tuple(item.name for item in parameters),
        outputs=tuple(str(item.get("status_code")) for item in responses),
        success_signals=success_signals,
        related_entity_ids=(entity_id,),
        evidence_ids=all_evidence,
        version_scope=version_scope,
        risk_level="read_only" if entity_payload["method"] in {"GET", "HEAD", "OPTIONS"} else "medium",
        status=status,
        metadata={
            "import_adapter": ADAPTER_NAME,
            "import_namespace": namespace,
            "source_operation_key": key,
            "method": entity_payload["method"],
            "path": entity_payload["path"],
            "deprecated": entity_payload["deprecated"],
            "semantic_hash": _hash_payload(contract_payload),
        },
    )
    return entity, contract


def _version_scope(version: str, snapshot: Mapping[str, Any], alignment: str) -> VersionScope:
    if alignment == "unconfirmed":
        return VersionScope(
            note=(
                f"requested version {version!r} was collected from ref "
                f"{snapshot.get('selected_ref')!r}; alignment is unconfirmed"
            )
        )
    return VersionScope(
        exact=(version,),
        compatibility_verified=alignment == "exact",
        note=f"documentation alignment: {alignment}",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{number}")
        yield value


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _reference_time(reference: Mapping[str, Any], source: Mapping[str, Any]) -> datetime:
    raw = reference.get("retrieved_at") or source.get("retrieved_at")
    return _parse_datetime(raw) if raw else datetime.fromtimestamp(0, timezone.utc)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog")
    parser.add_argument("database")
    parser.add_argument("--product", required=True)
    parser.add_argument("--vendor")
    parser.add_argument("--edition")
    parser.add_argument("--distribution")
    parser.add_argument("--platform")
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="import source documents and sections without treating operations as HTTP APIs",
    )
    parser.add_argument("--report")
    arguments = parser.parse_args(argv)
    store = SQLiteMemoryStore(arguments.database)
    try:
        report = ApiDocKnowledgeAdapter(MemoryService(store)).import_catalog(
            arguments.catalog,
            product=arguments.product,
            vendor=arguments.vendor,
            edition=arguments.edition,
            distribution=arguments.distribution,
            platform=arguments.platform,
            import_operations=not arguments.evidence_only,
            report_path=arguments.report,
        )
        print(report.model_dump_json(indent=2))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
