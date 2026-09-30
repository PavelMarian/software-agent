"""Collect a complete, versioned documentation or source-repository corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from software_multiagent.software_memory.acquisition.fetcher import DocumentationFetcher
from software_multiagent.software_memory.acquisition.models import (
    DocumentationSource,
    GitHubCorpusSource,
)
from software_multiagent.software_memory.acquisition.pipeline import ApiDocumentationPipeline
from software_multiagent.software_memory.acquisition.repository_pipeline import (
    github_documentation_pipeline,
)
from software_multiagent.software_memory.acquisition.storage import FileCatalogStore


DEFAULT_SOURCE_EXTENSIONS = ("*",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    web = commands.add_parser("collect-web", help="crawl a complete documentation site")
    _common_arguments(web)
    web.add_argument("--url", required=True)
    web.add_argument("--allowed-domain", action="append", default=[])
    web.add_argument("--max-depth", type=int, default=8)
    web.add_argument("--max-pages", type=int, default=10_000)
    web.add_argument("--max-response-bytes", type=int, default=20_000_000)
    web.add_argument("--max-total-bytes", type=int, default=100_000_000)
    web.add_argument("--workers", type=int, default=8)
    web.add_argument("--allow-site-wide", action="store_true")

    github = commands.add_parser(
        "collect-github", help="collect all eligible text/source files from a GitHub repository"
    )
    _common_arguments(github)
    github.add_argument("--url", required=True)
    github.add_argument("--ref")
    github.add_argument("--root", action="append", default=[])
    github.add_argument("--extension", action="append", default=[])
    github.add_argument("--filename", action="append", default=[])
    github.add_argument("--max-archive-bytes", type=int, default=500_000_000)
    github.add_argument("--max-uncompressed-bytes", type=int, default=1_000_000_000)
    github.add_argument("--max-file-bytes", type=int, default=20_000_000)
    github.add_argument("--max-files", type=int, default=100_000)
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("output", type=Path)
    parser.add_argument("--software", required=True)
    parser.add_argument("--version", required=True)


def _require_complete(manifest) -> None:  # type: ignore[no-untyped-def]
    if manifest.truncated:
        raise SystemExit(
            "collection was truncated; increase the applicable safety limit and rebuild"
        )
    if manifest.failed_urls:
        sample = ", ".join(sorted(manifest.failed_urls)[:5])
        raise SystemExit(
            f"collection has {len(manifest.failed_urls)} failed URL(s); first failures: {sample}"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    store = FileCatalogStore(arguments.output)
    if arguments.command == "collect-web":
        result = ApiDocumentationPipeline(
            fetcher=DocumentationFetcher(), extractor=None, store=store
        ).ingest(
            DocumentationSource(
                url=arguments.url,
                software=arguments.software,
                version=arguments.version,
                version_alignment="exact",
                allowed_domains=tuple(arguments.allowed_domain),
                max_pages=arguments.max_pages,
                max_depth=arguments.max_depth,
                max_response_bytes=arguments.max_response_bytes,
                max_total_bytes=arguments.max_total_bytes,
                crawl=True,
                restrict_to_initial_path=not arguments.allow_site_wide,
                discovery_mode="auto",
                max_workers=arguments.workers,
            )
        )
    else:
        result = github_documentation_pipeline(store).ingest(
            GitHubCorpusSource(
                project_url=arguments.url,
                software=arguments.software,
                requested_version=arguments.version,
                ref=arguments.ref,
                documentation_roots=tuple(arguments.root or ("*",)),
                allowed_extensions=tuple(arguments.extension or DEFAULT_SOURCE_EXTENSIONS),
                allowed_filenames=tuple(arguments.filename),
                max_archive_bytes=arguments.max_archive_bytes,
                max_uncompressed_bytes=arguments.max_uncompressed_bytes,
                max_file_bytes=arguments.max_file_bytes,
                max_files=arguments.max_files,
                timeout_seconds=120,
            )
        )
    _require_complete(result.catalog.corpus_manifest)
    print(result.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
