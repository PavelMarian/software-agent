from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from software_multiagent.software_memory.acquisition.models import (
    CorpusManifest,
    LocalCorpusSource,
    RepositorySnapshot,
    SourceDocument,
)
from software_multiagent.software_memory.acquisition.parser import make_source_document


@dataclass(frozen=True)
class LocalFetchResult:
    documents: tuple[SourceDocument, ...]
    manifest: CorpusManifest
    snapshot: RepositorySnapshot


class LocalCorpusConnector:
    """Read a bounded, symlink-safe corpus from an existing local checkout."""

    def fetch(self, source: LocalCorpusSource) -> LocalFetchResult:
        root = Path(source.root_path).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"local corpus root is not a directory: {root}")
        documents: list[SourceDocument] = []
        detected_roots: set[str] = set()
        total_bytes = 0
        digest = sha256()
        for configured_root in source.documentation_roots:
            candidate = (root / configured_root).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_dir():
                continue
            for path in sorted(candidate.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    continue
                if (
                    resolved.suffix.lower() not in source.allowed_extensions
                    and resolved.name not in source.allowed_filenames
                ):
                    continue
                size = resolved.stat().st_size
                if size > source.max_file_bytes:
                    continue
                total_bytes += size
                if total_bytes > source.max_uncompressed_bytes:
                    raise ValueError("local documentation exceeds max_uncompressed_bytes")
                if len(documents) >= source.max_files:
                    raise ValueError("local documentation exceeds max_files")
                content = resolved.read_bytes()
                relative = resolved.relative_to(root).as_posix()
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(content)
                uri = resolved.as_uri()
                suffix = resolved.suffix.lower()
                documents.append(
                    make_source_document(
                        url=uri,
                        canonical_url=uri,
                        content=content,
                        content_type=(
                            "text/markdown" if suffix in {".md", ".mdx"} else "text/plain"
                        ),
                        status=200,
                        headers={},
                    )
                )
                detected_roots.add(configured_root)
        documents.sort(key=lambda item: item.canonical_url)
        urls = tuple(document.canonical_url for document in documents)
        corpus_sha = source.commit_sha or digest.hexdigest()
        selected_ref = source.selected_ref or "local-snapshot"
        manifest = CorpusManifest(
            root_url=root.as_uri(),
            platform="local",
            version_alignment="exact" if source.commit_sha else "explicit_ref",
            strategies=("bounded_local_walk",),
            discovered_urls=urls,
            fetched_urls=urls,
            failed_urls={},
            truncated=False,
        )
        snapshot = RepositorySnapshot(
            provider="local",
            project_id=None,
            project_path=str(root),
            project_url=root.as_uri(),
            default_branch=selected_ref,
            selected_ref=selected_ref,
            commit_sha=corpus_sha,
            requested_version=source.requested_version,
            version_alignment="exact" if source.commit_sha else "explicit_ref",
            documentation_roots=tuple(sorted(detected_roots)),
            archive_bytes=0,
            uncompressed_documentation_bytes=total_bytes,
            file_count=len(documents),
        )
        return LocalFetchResult(tuple(documents), manifest, snapshot)
