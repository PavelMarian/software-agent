from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def stable_id(prefix: str, *parts: str) -> str:
    value = "\x1f".join(part.strip() for part in parts)
    return f"{prefix}_{sha256(value.encode('utf-8')).hexdigest()[:20]}"


class PageType(str, Enum):
    API_REFERENCE = "api_reference"
    OVERVIEW = "overview"
    AUTHENTICATION = "authentication"
    TUTORIAL = "tutorial"
    ERROR_REFERENCE = "error_reference"
    VERSION_NOTE = "version_note"
    IRRELEVANT = "irrelevant"


class KnowledgeStatus(str, Enum):
    EXTRACTED = "extracted"
    EVIDENCE_SUPPORTED = "evidence_supported"
    CROSS_SOURCE_CONFIRMED = "cross_source_confirmed"
    CONFLICTING = "conflicting"
    DEPRECATED = "deprecated"


class DocumentationSource(BaseModel):
    """Bounded acquisition request for one API documentation site."""

    model_config = ConfigDict(frozen=True)

    url: str
    software: str = Field(min_length=1)
    version: str = Field(default="unknown", min_length=1)
    version_alignment: Literal["exact", "unconfirmed"] = "unconfirmed"
    allowed_domains: tuple[str, ...] = ()
    max_pages: int = Field(default=20, ge=1, le=10_000)
    max_depth: int = Field(default=2, ge=0, le=8)
    max_response_bytes: int = Field(default=2_000_000, ge=1_024, le=20_000_000)
    max_total_bytes: int = Field(default=10_000_000, ge=1_024, le=100_000_000)
    timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    crawl: bool = True
    restrict_to_initial_path: bool = True
    discovery_mode: Literal["links", "auto"] = "links"
    max_workers: int = Field(default=2, ge=1, le=32)


class GitLabCorpusSource(BaseModel):
    """Read-only request for documentation stored in a GitLab repository."""

    model_config = ConfigDict(frozen=True)

    project_url: str
    software: str = Field(min_length=1)
    requested_version: str = Field(default="unknown", min_length=1)
    ref: str | None = None
    documentation_roots: tuple[str, ...] = ("docs", "content", "documentation")
    allowed_extensions: tuple[str, ...] = (".md", ".mdx", ".rst", ".txt")
    max_archive_bytes: int = Field(default=100_000_000, ge=1_024, le=500_000_000)
    max_uncompressed_bytes: int = Field(default=200_000_000, ge=1_024, le=1_000_000_000)
    max_file_bytes: int = Field(default=4_000_000, ge=1_024, le=20_000_000)
    max_files: int = Field(default=20_000, ge=1, le=100_000)
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)


class GitHubCorpusSource(BaseModel):
    """Read-only request for a bounded corpus stored in a GitHub repository."""

    model_config = ConfigDict(frozen=True)

    project_url: str
    software: str = Field(min_length=1)
    requested_version: str = Field(default="unknown", min_length=1)
    ref: str | None = None
    documentation_roots: tuple[str, ...] = ("docs", "doc", "documentation")
    allowed_extensions: tuple[str, ...] = (".md", ".mdx", ".rst", ".txt", ".org")
    allowed_filenames: tuple[str, ...] = ()
    max_archive_bytes: int = Field(default=100_000_000, ge=1_024, le=500_000_000)
    max_uncompressed_bytes: int = Field(default=200_000_000, ge=1_024, le=1_000_000_000)
    max_file_bytes: int = Field(default=4_000_000, ge=1_024, le=20_000_000)
    max_files: int = Field(default=20_000, ge=1, le=100_000)
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)


class LocalCorpusSource(BaseModel):
    """Bounded request for documentation already present on the local filesystem."""

    model_config = ConfigDict(frozen=True)

    root_path: str
    software: str = Field(min_length=1)
    requested_version: str = Field(default="unknown", min_length=1)
    selected_ref: str | None = None
    commit_sha: str | None = None
    documentation_roots: tuple[str, ...] = ("docs", "doc", "documentation")
    allowed_extensions: tuple[str, ...] = (".md", ".mdx", ".rst", ".txt", ".org")
    allowed_filenames: tuple[str, ...] = ()
    max_uncompressed_bytes: int = Field(default=200_000_000, ge=1_024, le=1_000_000_000)
    max_file_bytes: int = Field(default=4_000_000, ge=1_024, le=20_000_000)
    max_files: int = Field(default=20_000, ge=1, le=100_000)


class RepositorySnapshot(BaseModel):
    provider: Literal["gitlab", "github", "local"] = "gitlab"
    project_id: int | str | None = None
    project_path: str
    project_url: str
    default_branch: str
    selected_ref: str
    commit_sha: str
    requested_version: str
    version_alignment: Literal["exact", "explicit_ref", "unconfirmed"]
    documentation_roots: tuple[str, ...]
    archive_bytes: int = 0
    uncompressed_documentation_bytes: int = 0
    file_count: int = 0


class EvidenceRef(BaseModel):
    source_id: str
    section_id: str | None = None
    source_url: str
    heading_path: tuple[str, ...] = ()
    quote: str = Field(default="", max_length=2_000)
    pointer: str | None = None
    verified: bool = False


class SourceSection(BaseModel):
    section_id: str
    source_id: str
    heading_path: tuple[str, ...] = ()
    content: str
    ordinal: int = Field(ge=0)


class SourceDocument(BaseModel):
    source_id: str
    url: str
    canonical_url: str
    content_type: str
    title: str | None = None
    page_type: PageType = PageType.OVERVIEW
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str
    raw_text: str
    links: tuple[str, ...] = ()
    sections: tuple[SourceSection, ...] = ()
    http_status: int = 200
    etag: str | None = None
    last_modified: str | None = None


class ApiParameter(BaseModel):
    name: str
    location: Literal["path", "query", "header", "cookie", "body", "unknown"] = "unknown"
    required: bool = False
    schema_type: str | None = None
    description: str | None = None
    default: Any = None
    example: Any = None
    enum: tuple[Any, ...] = ()
    constraints: dict[str, Any] = Field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = ()


class ApiResponse(BaseModel):
    status_code: str
    description: str | None = None
    content_types: tuple[str, ...] = ()
    response_schema: dict[str, Any] | None = None
    example: Any = None
    evidence: tuple[EvidenceRef, ...] = ()


class ApiOperation(BaseModel):
    operation_id: str
    name: str
    method: str
    path: str
    base_urls: tuple[str, ...] = ()
    description: str | None = None
    parameters: tuple[ApiParameter, ...] = ()
    request_body_schema: dict[str, Any] | None = None
    request_content_types: tuple[str, ...] = ()
    responses: tuple[ApiResponse, ...] = ()
    authentication: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    deprecated: bool = False
    status: KnowledgeStatus = KnowledgeStatus.EXTRACTED
    evidence: tuple[EvidenceRef, ...] = ()

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE"}:
            raise ValueError(f"unsupported HTTP method: {value}")
        return normalized

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        stripped = value.strip()
        return stripped if stripped.startswith(("/", "http://", "https://")) else f"/{stripped}"


class OperationFingerprint(BaseModel):
    operation_id: str
    method: str
    path: str
    purpose: str
    required_parameters: tuple[str, ...] = ()
    optional_parameters: tuple[str, ...] = ()
    request_content_types: tuple[str, ...] = ()
    response_statuses: tuple[str, ...] = ()
    authentication: tuple[str, ...] = ()


class Conflict(BaseModel):
    conflict_id: str
    operation_key: str
    field: str
    values: tuple[Any, ...]
    evidence: tuple[EvidenceRef, ...] = ()


class CorpusManifest(BaseModel):
    root_url: str
    platform: str = "generic"
    version_alignment: Literal["exact", "explicit_ref", "unconfirmed"] = "unconfirmed"
    strategies: tuple[str, ...] = ()
    discovered_urls: tuple[str, ...] = ()
    fetched_urls: tuple[str, ...] = ()
    failed_urls: dict[str, str] = Field(default_factory=dict)
    truncated: bool = False

    @property
    def discovered_count(self) -> int:
        return len(self.discovered_urls)

    @property
    def fetched_count(self) -> int:
        return len(self.fetched_urls)


class ApiCatalog(BaseModel):
    software: str
    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: tuple[SourceDocument, ...] = ()
    operations: tuple[ApiOperation, ...] = ()
    fingerprints: tuple[OperationFingerprint, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    corpus_manifest: CorpusManifest | None = None
