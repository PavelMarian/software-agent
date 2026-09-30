"""Structured, evidence-backed knowledge extracted from API documentation."""

from software_multiagent.software_memory.acquisition.extraction import (
    Doc2AgentExtractor,
    JsonExtractionClient,
    OpenApiExtractor,
)
from software_multiagent.software_memory.acquisition.discovery import AutoCorpusDiscovery, DiscoveryResult
from software_multiagent.software_memory.acquisition.fetcher import DocumentationFetcher, FetchResponse, FetchResult, UrlPolicy
from software_multiagent.software_memory.acquisition.gitlab import GitLabCorpusConnector, GitLabFetchResult
from software_multiagent.software_memory.acquisition.gitlab_pipeline import GitLabDocumentationPipeline, GitLabIngestionResult
from software_multiagent.software_memory.acquisition.github import GitHubCorpusConnector, GitHubFetchResult
from software_multiagent.software_memory.acquisition.local import LocalCorpusConnector, LocalFetchResult
from software_multiagent.software_memory.acquisition.models import (
    ApiCatalog,
    ApiOperation,
    ApiParameter,
    Conflict,
    CorpusManifest,
    DocumentationSource,
    EvidenceRef,
    GitHubCorpusSource,
    GitLabCorpusSource,
    LocalCorpusSource,
    OperationFingerprint,
    PageType,
    RepositorySnapshot,
    SourceDocument,
    SourceSection,
)
from software_multiagent.software_memory.acquisition.pipeline import ApiDocumentationPipeline, IngestionResult
from software_multiagent.software_memory.acquisition.repository_pipeline import (
    RepositoryDocumentationPipeline,
    RepositoryIngestionResult,
    github_documentation_pipeline,
    local_documentation_pipeline,
)
from software_multiagent.software_memory.acquisition.retrieval import CatalogReader
from software_multiagent.software_memory.acquisition.storage import FileCatalogStore

__all__ = [
    "ApiCatalog",
    "ApiDocumentationPipeline",
    "ApiOperation",
    "ApiParameter",
    "CatalogReader",
    "Conflict",
    "CorpusManifest",
    "AutoCorpusDiscovery",
    "DiscoveryResult",
    "Doc2AgentExtractor",
    "DocumentationFetcher",
    "DocumentationSource",
    "EvidenceRef",
    "FetchResponse",
    "FetchResult",
    "FileCatalogStore",
    "GitLabCorpusConnector",
    "GitLabCorpusSource",
    "GitLabDocumentationPipeline",
    "GitLabFetchResult",
    "GitLabIngestionResult",
    "GitHubCorpusConnector",
    "GitHubCorpusSource",
    "GitHubFetchResult",
    "IngestionResult",
    "JsonExtractionClient",
    "LocalCorpusConnector",
    "LocalCorpusSource",
    "LocalFetchResult",
    "OpenApiExtractor",
    "OperationFingerprint",
    "PageType",
    "RepositorySnapshot",
    "RepositoryDocumentationPipeline",
    "RepositoryIngestionResult",
    "SourceDocument",
    "SourceSection",
    "UrlPolicy",
    "github_documentation_pipeline",
    "local_documentation_pipeline",
]

__version__ = "0.1.0"
