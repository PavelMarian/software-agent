from __future__ import annotations

import io
import json
import posixpath
import tarfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote, urlparse

from software_multiagent.software_memory.acquisition.fetcher import FetchResponse, UrllibTransport
from software_multiagent.software_memory.acquisition.models import (
    CorpusManifest,
    GitLabCorpusSource,
    RepositorySnapshot,
    SourceDocument,
)
from software_multiagent.software_memory.acquisition.parser import make_source_document


class GitLabTransport(Protocol):
    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> FetchResponse: ...


@dataclass(frozen=True)
class GitLabFetchResult:
    documents: tuple[SourceDocument, ...]
    manifest: CorpusManifest
    snapshot: RepositorySnapshot


class GitLabCorpusConnector:
    """Acquire a documentation snapshot with a small, bounded number of GitLab requests."""

    def __init__(self, transport: GitLabTransport | None = None) -> None:
        self.transport = transport or UrllibTransport(user_agent="api-doc-knowledge-gitlab/0.1")

    def fetch(self, source: GitLabCorpusSource) -> GitLabFetchResult:
        host, project_path = self._parse_project_url(source.project_url)
        api_root = f"https://{host}/api/v4"
        encoded_project = quote(project_path, safe="")
        project = self._get_json(
            f"{api_root}/projects/{encoded_project}", source, max_bytes=2_000_000
        )
        project_id = int(project["id"])
        project_api = f"{api_root}/projects/{project_id}"
        default_branch = str(project.get("default_branch") or "main")
        selected_ref, alignment = self._select_ref(
            project_api, default_branch, source
        )
        commit = self._get_json(
            f"{project_api}/repository/commits/{quote(selected_ref, safe='')}",
            source,
            max_bytes=2_000_000,
        )
        commit_sha = str(commit["id"])
        archive_url = (
            f"{project_api}/repository/archive.tar.gz?sha={quote(commit_sha, safe='')}"
        )
        archive_response = self.transport.fetch(
            archive_url,
            timeout=source.timeout_seconds,
            max_bytes=source.max_archive_bytes,
        )
        self._require_success(archive_response, archive_url)
        documents, uncompressed_bytes, roots = self._documents_from_archive(
            archive_response.content,
            source=source,
            project_url=str(project.get("web_url") or source.project_url).rstrip("/"),
            commit_sha=commit_sha,
        )
        document_urls = tuple(document.canonical_url for document in documents)
        manifest = CorpusManifest(
            root_url=source.project_url,
            platform="gitlab",
            version_alignment=alignment,
            strategies=("gitlab_project_api", "gitlab_repository_archive"),
            discovered_urls=document_urls,
            fetched_urls=document_urls,
            failed_urls={},
            truncated=False,
        )
        snapshot = RepositorySnapshot(
            project_id=project_id,
            project_path=str(project.get("path_with_namespace") or project_path),
            project_url=str(project.get("web_url") or source.project_url),
            default_branch=default_branch,
            selected_ref=selected_ref,
            commit_sha=commit_sha,
            requested_version=source.requested_version,
            version_alignment=alignment,
            documentation_roots=roots,
            archive_bytes=len(archive_response.content),
            uncompressed_documentation_bytes=uncompressed_bytes,
            file_count=len(documents),
        )
        return GitLabFetchResult(documents=documents, manifest=manifest, snapshot=snapshot)

    @staticmethod
    def _parse_project_url(project_url: str) -> tuple[str, str]:
        parsed = urlparse(project_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("GitLab project URL must be an absolute HTTPS URL")
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        if not path or any(part in {".", ".."} for part in path.split("/")):
            raise ValueError("invalid GitLab project path")
        return parsed.hostname.lower(), path

    def _select_ref(
        self,
        project_api: str,
        default_branch: str,
        source: GitLabCorpusSource,
    ) -> tuple[str, str]:
        if source.ref:
            return source.ref, "explicit_ref"
        branches = self._get_json(
            f"{project_api}/repository/branches?per_page=100",
            source,
            max_bytes=5_000_000,
        )
        tags = self._get_json(
            f"{project_api}/repository/tags?per_page=100",
            source,
            max_bytes=5_000_000,
        )
        refs = {
            str(item.get("name")): str(item.get("name"))
            for item in [*(branches if isinstance(branches, list) else []), *(tags if isinstance(tags, list) else [])]
            if isinstance(item, dict) and item.get("name")
        }
        version = source.requested_version.strip()
        candidates = (f"v{version}", version)
        lowered = {name.lower(): name for name in refs}
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()], "exact"
        return default_branch, "unconfirmed"

    def _get_json(
        self, url: str, source: GitLabCorpusSource, *, max_bytes: int
    ) -> Any:
        response = self.transport.fetch(
            url, timeout=source.timeout_seconds, max_bytes=max_bytes
        )
        self._require_success(response, url)
        try:
            return json.loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"GitLab returned invalid JSON for {url}") from error

    @staticmethod
    def _require_success(response: FetchResponse, url: str) -> None:
        if not 200 <= response.status < 300:
            raise ValueError(f"GitLab request failed with HTTP {response.status}: {url}")

    def _documents_from_archive(
        self,
        archive: bytes,
        *,
        source: GitLabCorpusSource,
        project_url: str,
        commit_sha: str,
    ) -> tuple[tuple[SourceDocument, ...], int, tuple[str, ...]]:
        documents: list[SourceDocument] = []
        total_bytes = 0
        detected_roots: set[str] = set()
        seen_paths: set[str] = set()
        try:
            tar = tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz")
        except tarfile.TarError as error:
            raise ValueError("GitLab archive is not a valid tar.gz file") from error
        with tar:
            for member in tar:
                if not member.isfile() or member.issym() or member.islnk():
                    continue
                relative = self._archive_relative_path(member.name)
                if relative is None or relative in seen_paths:
                    continue
                root = relative.parts[0] if relative.parts else ""
                suffix = relative.suffix.lower()
                if root not in source.documentation_roots or suffix not in source.allowed_extensions:
                    continue
                if member.size > source.max_file_bytes:
                    raise ValueError(f"documentation file exceeds max_file_bytes: {relative}")
                total_bytes += member.size
                if total_bytes > source.max_uncompressed_bytes:
                    raise ValueError("GitLab documentation exceeds max_uncompressed_bytes")
                if len(documents) >= source.max_files:
                    raise ValueError("GitLab documentation exceeds max_files")
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                content = handle.read(source.max_file_bytes + 1)
                if len(content) > source.max_file_bytes:
                    raise ValueError(f"documentation file exceeds max_file_bytes: {relative}")
                path = relative.as_posix()
                blob_url = f"{project_url}/-/blob/{commit_sha}/{quote(path, safe='/')}"
                content_type = "text/markdown" if suffix in {".md", ".mdx"} else "text/plain"
                documents.append(
                    make_source_document(
                        url=blob_url,
                        canonical_url=blob_url,
                        content=content,
                        content_type=content_type,
                        status=200,
                        headers={},
                    )
                )
                detected_roots.add(root)
                seen_paths.add(relative.as_posix())
        documents.sort(key=lambda item: item.canonical_url)
        return tuple(documents), total_bytes, tuple(sorted(detected_roots))

    @staticmethod
    def _archive_relative_path(name: str) -> PurePosixPath | None:
        normalized = posixpath.normpath(name.replace("\\", "/"))
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
            return None
        relative = PurePosixPath(*path.parts[1:])
        if not relative.parts or ".." in relative.parts:
            return None
        return relative
