from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from software_multiagent.software_memory.acquisition.extraction import Doc2AgentExtractor
from software_multiagent.software_memory.acquisition.fetcher import DocumentationFetcher
from software_multiagent.software_memory.acquisition.models import ApiCatalog, DocumentationSource
from software_multiagent.software_memory.acquisition.normalization import make_fingerprint, normalize_operations
from software_multiagent.software_memory.acquisition.storage import FileCatalogStore


@dataclass(frozen=True)
class IngestionResult:
    catalog: ApiCatalog
    output_directory: Path


class ApiDocumentationPipeline:
    def __init__(
        self,
        *,
        fetcher: DocumentationFetcher,
        extractor: Doc2AgentExtractor | None,
        store: FileCatalogStore,
    ) -> None:
        self.fetcher = fetcher
        self.extractor = extractor
        self.store = store

    def ingest(self, source: DocumentationSource) -> IngestionResult:
        fetch_result = self.fetcher.fetch_with_manifest(source)
        documents = fetch_result.documents
        extracted = (
            tuple(
                operation
                for document in documents
                for operation in self.extractor.extract(document)
            )
            if self.extractor is not None
            else ()
        )
        operations, conflicts = normalize_operations(extracted)
        catalog = ApiCatalog(
            software=source.software,
            version=source.version,
            sources=documents,
            operations=operations,
            fingerprints=tuple(make_fingerprint(operation) for operation in operations),
            conflicts=conflicts,
            corpus_manifest=fetch_result.manifest,
        )
        output_directory = self.store.save(catalog)
        return IngestionResult(catalog=catalog, output_directory=output_directory)
