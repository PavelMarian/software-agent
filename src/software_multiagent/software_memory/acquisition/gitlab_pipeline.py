from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from software_multiagent.software_memory.acquisition.extraction import Doc2AgentExtractor
from software_multiagent.software_memory.acquisition.gitlab import GitLabCorpusConnector
from software_multiagent.software_memory.acquisition.models import ApiCatalog, GitLabCorpusSource, RepositorySnapshot
from software_multiagent.software_memory.acquisition.normalization import make_fingerprint, normalize_operations
from software_multiagent.software_memory.acquisition.storage import FileCatalogStore


@dataclass(frozen=True)
class GitLabIngestionResult:
    catalog: ApiCatalog
    snapshot: RepositorySnapshot
    output_directory: Path


class GitLabDocumentationPipeline:
    def __init__(
        self,
        *,
        connector: GitLabCorpusConnector,
        extractor: Doc2AgentExtractor,
        store: FileCatalogStore,
    ) -> None:
        self.connector = connector
        self.extractor = extractor
        self.store = store

    def ingest(self, source: GitLabCorpusSource) -> GitLabIngestionResult:
        fetched = self.connector.fetch(source)
        extracted = tuple(
            operation
            for document in fetched.documents
            for operation in self.extractor.extract(document)
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
        return GitLabIngestionResult(
            catalog=catalog,
            snapshot=fetched.snapshot,
            output_directory=output_directory,
        )
