from __future__ import annotations

import io
import json
import posixpath
import tarfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote, urljoin, urlparse

from software_multiagent.software_memory.acquisition.fetcher import FetchResponse, UrllibTransport
from software_multiagent.software_memory.acquisition.models import (
    CorpusManifest,
    GitHubCorpusSource,
    RepositorySnapshot,
    SourceDocument,
)
from software_multiagent.software_memory.acquisition.parser import make_source_document


class GitHubTransport(Protocol):
    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> FetchResponse: ...


@dataclass(frozen=True)
class GitHubFetchResult:
    documents: tuple[SourceDocument, ...]
    manifest: CorpusManifest
    snapshot: RepositorySnapshot


class GitHubCorpusConnector:
    """Download one bounded GitHub archive and expose selected files as evidence."""

    _redirect_hosts = {"api.github.com", "github.com", "codeload.github.com"}

    def __init__(self, transport: GitHubTransport | None = None) -> None:
        self.transport = transport or UrllibTransport(user_agent="api-doc-knowledge-github/0.1")

    def fetch(self, source: GitHubCorpusSource) -> GitHubFetchResult:
        owner, repository = self._parse_project_url(source.project_url)
        api_root = f"https://api.github.com/repos/{quote(owner)}/{quote(repository)}"
        project = self._get_json(api_root, source, max_bytes=2_000_000)
        default_branch = str(project.get("default_branch") or "main")
        selected_ref, alignment = self._select_ref(
            api_root, repository, default_branch, source
        )
        commit = self._get_json(
            f"{api_root}/commits/{quote(selected_ref, safe='')}",
            source,
            max_bytes=2_000_000,
        )
        commit_sha = str(commit["sha"])
        archive_url = f"{api_root}/tarball/{quote(commit_sha, safe='')}"
        archive_response = self._fetch_following_redirects(
            archive_url,
            timeout=source.timeout_seconds,
            max_bytes=source.max_archive_bytes,
        )
        self._require_success(archive_response, archive_url)
        project_url = str(project.get("html_url") or source.project_url).rstrip("/")
        documents, uncompressed_bytes, roots, truncated = self._documents_from_archive(
            archive_response.content,
            source=source,
            project_url=project_url,
            commit_sha=commit_sha,
        )
        urls = tuple(document.canonical_url for document in documents)
        manifest = CorpusManifest(
            root_url=source.project_url,
            platform="github",
            version_alignment=alignment,
            strategies=("github_repository_api", "github_repository_archive"),
            discovered_urls=urls,
            fetched_urls=urls,
            failed_urls={},
            truncated=truncated,
        )
        snapshot = RepositorySnapshot(
            provider="github",
            project_id=str(project.get("id")) if project.get("id") is not None else None,
            project_path=str(project.get("full_name") or f"{owner}/{repository}"),
            project_url=project_url,
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
        return GitHubFetchResult(documents, manifest, snapshot)

    @staticmethod
    def _parse_project_url(project_url: str) -> tuple[str, str]:
        parsed = urlparse(project_url)
        if parsed.scheme != "https" or parsed.hostname not in {"github.com", "www.github.com"}:
            raise ValueError("GitHub project URL must be an absolute github.com HTTPS URL")
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) != 2 or any(part in {".", ".."} for part in parts):
            raise ValueError("invalid GitHub project path")
        repository = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
        return parts[0], repository

    def _select_ref(
        self,
        api_root: str,
        repository: str,
        default_branch: str,
        source: GitHubCorpusSource,
    ) -> tuple[str, str]:
        if source.ref:
            return source.ref, "explicit_ref"
        branches = self._get_json(
            f"{api_root}/branches?per_page=100", source, max_bytes=5_000_000
        )
        tags = self._get_json(
            f"{api_root}/tags?per_page=100", source, max_bytes=5_000_000
        )
        refs = {
            str(item.get("name")): str(item.get("name"))
            for item in [
                *(branches if isinstance(branches, list) else []),
                *(tags if isinstance(tags, list) else []),
            ]
            if isinstance(item, dict) and item.get("name")
        }
        version = source.requested_version.strip()
        candidates = (f"v{version}", version)
        lowered = {name.lower(): name for name in refs}
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()], "exact"
        # A repository whose name ends in the requested version is itself a
        # versioned source, even when its maintained branch is named main/master.
        normalized_repository = repository.lower().replace("_", "-")
        if normalized_repository.endswith(f"-{version.lower()}"):
            return default_branch, "exact"
        return default_branch, "unconfirmed"

    def _get_json(
        self, url: str, source: GitHubCorpusSource, *, max_bytes: int
    ) -> Any:
        response = self._fetch_following_redirects(
            url, timeout=source.timeout_seconds, max_bytes=max_bytes
        )
        self._require_success(response, url)
        try:
            return json.loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"GitHub returned invalid JSON for {url}") from error

    def _fetch_following_redirects(
        self, url: str, *, timeout: float, max_bytes: int
    ) -> FetchResponse:
        current = url
        for _ in range(6):
            parsed = urlparse(current)
            if parsed.scheme != "https" or parsed.hostname not in self._redirect_hosts:
                raise ValueError(f"GitHub redirected outside the allowed hosts: {current}")
            response = self.transport.fetch(current, timeout=timeout, max_bytes=max_bytes)
            if response.status not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("location")
            if not location:
                raise ValueError(f"GitHub redirect did not include Location: {current}")
            current = urljoin(current, location)
        raise ValueError("too many GitHub redirects")

    @staticmethod
    def _require_success(response: FetchResponse, url: str) -> None:
        if not 200 <= response.status < 300:
            raise ValueError(f"GitHub request failed with HTTP {response.status}: {url}")

    def _documents_from_archive(
        self,
        archive: bytes,
        *,
        source: GitHubCorpusSource,
        project_url: str,
        commit_sha: str,
    ) -> tuple[tuple[SourceDocument, ...], int, tuple[str, ...], bool]:
        documents: list[SourceDocument] = []
        total_bytes = 0
        detected_roots: set[str] = set()
        seen_paths: set[str] = set()
        truncated = False
        try:
            tar = tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz")
        except tarfile.TarError as error:
            raise ValueError("GitHub archive is not a valid tar.gz file") from error
        with tar:
            for member in tar:
                if not member.isfile() or member.issym() or member.islnk():
                    continue
                relative = self._archive_relative_path(member.name)
                if relative is None or relative.as_posix() in seen_paths:
                    continue
                root = relative.parts[0] if relative.parts else ""
                suffix = relative.suffix.lower()
                if ("*" not in source.documentation_roots and root not in source.documentation_roots) or (
                    "*" not in source.allowed_extensions
                    and suffix not in source.allowed_extensions
                    and relative.name not in source.allowed_filenames
                ):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                if member.size > source.max_file_bytes:
                    sample = handle.read(min(member.size, 8192))
                    if _looks_textual(sample):
                        truncated = True
                        break
                    continue
                if len(documents) >= source.max_files:
                    truncated = True
                    break
                if total_bytes + member.size > source.max_uncompressed_bytes:
                    truncated = True
                    break
                content = handle.read(source.max_file_bytes + 1)
                if len(content) > source.max_file_bytes:
                    truncated = True
                    break
                if "*" in source.allowed_extensions and not _looks_textual(content):
                    continue
                total_bytes += member.size
                path = relative.as_posix()
                blob_url = f"{project_url}/blob/{commit_sha}/{quote(path, safe='/')}"
                content_type = (
                    "text/markdown" if suffix in {".md", ".mdx"} else "text/plain"
                )
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
                seen_paths.add(path)
        documents.sort(key=lambda item: item.canonical_url)
        return tuple(documents), total_bytes, tuple(sorted(detected_roots)), truncated

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


def _looks_textual(content: bytes) -> bool:
    """Reject binary archive members while retaining extensionless text files."""

    sample = content[:8192]
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    controls = sum(byte < 32 and byte not in {9, 10, 12, 13} for byte in sample)
    return controls / len(sample) < 0.02
