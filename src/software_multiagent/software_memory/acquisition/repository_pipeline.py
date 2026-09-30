from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from software_multiagent.software_memory.acquisition.extraction import Doc2AgentExtractor
from software_multiagent.software_memory.acquisition.github import GitHubCorpusConnector, GitHubFetchResult
from software_multiagent.software_memory.acquisition.local import LocalCorpusConnector, LocalFetchResult
from software_multiagent.software_memory.acquisition.models import (
    ApiCatalog,
    GitHubCorpusSource,
    LocalCorpusSource,
    RepositorySnapshot,
)
from software_multiagent.software_memory.acquisition.normalization import make_fingerprint, normalize_operations
from software_multiagent.software_memory.acquisition.storage import FileCatalogStore


@dataclass(frozen=True)
class RepositoryIngestionResult:
    catalog: ApiCatalog
    snapshot: RepositorySnapshot
    output_directory: Path


SourceT = TypeVar("SourceT", GitHubCorpusSource, LocalCorpusSource)
FetchT = TypeVar("FetchT", GitHubFetchResult, LocalFetchResult)


class RepositoryConnector(Protocol[SourceT, FetchT]):
    def fetch(self, source: SourceT) -> FetchT: ...


class RepositoryDocumentationPipeline:
    """Store a repository corpus, optionally extracting HTTP API operations."""

    def __init__(
        self,
        *,
        connector: RepositoryConnector,
        store: FileCatalogStore,
        extractor: Doc2AgentExtractor | None = None,
    ) -> None:
        self.connector = connector
        self.extractor = extractor
        self.store = store

    def ingest(self, source: SourceT) -> RepositoryIngestionResult:
        fetched = self.connector.fetch(source)
        extracted = (
            tuple(
                operation
                for document in fetched.documents
                for operation in self.extractor.extract(document)
            )
            if self.extractor is not None
            else ()
        )
        operations, conflicts = normalize_operations(extracted)
        catalog = ApiCatalog(
            software=source.software,
            version=source.requested_version,
            sources=fetched.documents,
            operations=operations,
            fingerprints=tuple(make_fingerprint(operation) for operation in operations),
            conflicts=conflicts,
            corpus_manifest=fetched.manifest,
        )
        output_directory = self.store.save(catalog)
        self.store.save_repository_snapshot(
            source.software, source.requested_version, fetched.snapshot
        )
        return RepositoryIngestionResult(catalog, fetched.snapshot, output_directory)


def github_documentation_pipeline(
    store: FileCatalogStore, *, extract_http_operations: bool = False
) -> RepositoryDocumentationPipeline:
    return RepositoryDocumentationPipeline(
        connector=GitHubCorpusConnector(),
        store=store,
        extractor=Doc2AgentExtractor() if extract_http_operations else None,
    )


def local_documentation_pipeline(
    store: FileCatalogStore, *, extract_http_operations: bool = False
) -> RepositoryDocumentationPipeline:
    return RepositoryDocumentationPipeline(
        connector=LocalCorpusConnector(),
        store=store,
        extractor=Doc2AgentExtractor() if extract_http_operations else None,
    )
