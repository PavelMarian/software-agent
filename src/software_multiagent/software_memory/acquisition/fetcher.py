from __future__ import annotations

import ipaddress
import socket
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.error import HTTPError
from urllib.parse import urldefrag, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from software_multiagent.software_memory.acquisition.discovery import AutoCorpusDiscovery
from software_multiagent.software_memory.acquisition.models import CorpusManifest, DocumentationSource, SourceDocument
from software_multiagent.software_memory.acquisition.parser import make_source_document


@dataclass(frozen=True)
class FetchResponse:
    url: str
    status: int
    headers: dict[str, str]
    content: bytes


@dataclass(frozen=True)
class FetchResult:
    documents: tuple[SourceDocument, ...]
    manifest: CorpusManifest


class HttpTransport(Protocol):
    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> FetchResponse: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class UrllibTransport:
    def __init__(self, user_agent: str = "api-doc-knowledge/0.1") -> None:
        self._opener = build_opener(_NoRedirect())
        self._user_agent = user_agent

    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> FetchResponse:
        request = Request(url, headers={"User-Agent": self._user_agent, "Accept": "text/html, application/json, text/plain"})
        try:
            response = self._opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        content = response.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError(f"response exceeds {max_bytes} bytes: {url}")
        headers = {key.lower(): value for key, value in response.headers.items()}
        return FetchResponse(response.geturl(), int(response.status), headers, content)


class UrlPolicy:
    """Rejects SSRF-prone targets and keeps crawling inside the requested docs site."""

    def __init__(self, resolver: Callable[..., list[tuple]] | None = None) -> None:
        self._resolver = resolver or socket.getaddrinfo

    def validate(self, url: str, source: DocumentationSource) -> str:
        clean_url, _fragment = urldefrag(url)
        parsed = urlparse(clean_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"only absolute HTTP(S) URLs are allowed: {url}")
        if parsed.username or parsed.password:
            raise ValueError("credentials in documentation URLs are not allowed")

        initial = urlparse(source.url)
        allowed = {domain.lower().rstrip(".") for domain in source.allowed_domains}
        allowed.add((initial.hostname or "").lower().rstrip("."))
        hostname = parsed.hostname.lower().rstrip(".")
        if not any(hostname == domain or hostname.endswith(f".{domain}") for domain in allowed):
            raise ValueError(f"domain is outside the documentation scope: {hostname}")

        for result in self._resolver(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM):
            address = ipaddress.ip_address(result[4][0])
            if not address.is_global:
                raise ValueError(f"non-public documentation address is not allowed: {address}")
        return clean_url

    def is_crawl_candidate(self, url: str, source: DocumentationSource) -> bool:
        try:
            clean = self.validate(url, source)
        except ValueError:
            return False
        initial = urlparse(source.url)
        candidate = urlparse(clean)
        if source.restrict_to_initial_path:
            base_path = initial.path.rsplit("/", 1)[0].rstrip("/")
            if base_path and not candidate.path.startswith(f"{base_path}/") and candidate.path != base_path:
                return False
        ignored_suffixes = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".zip", ".tar", ".gz")
        return not candidate.path.lower().endswith(ignored_suffixes)


class DocumentationFetcher:
    _accepted_types = {
        "text/html", "application/xhtml+xml", "application/json",
        "application/vnd.oai.openapi+json", "text/plain", "text/markdown",
    }

    def __init__(self, transport: HttpTransport | None = None, policy: UrlPolicy | None = None) -> None:
        self.transport = transport or UrllibTransport()
        self.policy = policy or UrlPolicy()

    def fetch(self, source: DocumentationSource) -> tuple[SourceDocument, ...]:
        return self.fetch_with_manifest(source).documents

    def fetch_with_manifest(self, source: DocumentationSource) -> FetchResult:
        queue: deque[tuple[str, int]] = deque([(source.url, 0)])
        seen: set[str] = set()
        documents: list[SourceDocument] = []
        failed: dict[str, str] = {}
        total_bytes = 0
        platform = "generic"
        strategies: tuple[str, ...] = ("html_links",) if source.crawl else ()
        discovered_urls: set[str] = {source.url}
        truncated = False

        while queue and len(documents) < source.max_pages:
            requested_url, depth = queue.popleft()
            clean_url = self.policy.validate(requested_url, source)
            if clean_url in seen:
                continue
            seen.add(clean_url)

            try:
                response = self._fetch_following_redirects(clean_url, source)
            except (OSError, ValueError) as error:
                if not documents:
                    raise
                failed[clean_url] = str(error)
                continue
            total_bytes += len(response.content)
            if total_bytes > source.max_total_bytes:
                raise ValueError("documentation crawl exceeded max_total_bytes")
            if not 200 <= response.status < 300:
                raise ValueError(f"documentation request failed with HTTP {response.status}: {clean_url}")
            media_type = response.headers.get("content-type", "text/plain").split(";", 1)[0].lower()
            if media_type not in self._accepted_types:
                raise ValueError(f"unsupported documentation content type {media_type}: {clean_url}")

            canonical = self.policy.validate(response.url, source)
            document = make_source_document(
                url=clean_url,
                canonical_url=canonical,
                content=response.content,
                content_type=response.headers.get("content-type", media_type),
                status=response.status,
                headers=response.headers,
            )
            documents.append(document)
            discovered_urls.add(canonical)
            if len(documents) == 1 and source.discovery_mode == "auto":
                discovery = AutoCorpusDiscovery(self.transport, self.policy).discover(
                    source, response, document
                )
                platform = discovery.platform
                strategies = discovery.strategies
                discovered_urls.update(discovery.urls)
                if discovery.platform == "docusaurus" or "sitemap" in discovery.strategies:
                    remaining = max(0, source.max_pages - len(documents))
                    available_targets = [url for url in discovery.urls if url not in seen]
                    if len(available_targets) > remaining:
                        truncated = True
                    targets = available_targets[:remaining]
                    corpus_documents, corpus_failed, corpus_bytes = self._fetch_parallel(
                        targets, source
                    )
                    total_bytes += corpus_bytes
                    if total_bytes > source.max_total_bytes:
                        raise ValueError("documentation crawl exceeded max_total_bytes")
                    existing_canonical = {item.canonical_url for item in documents}
                    for item in corpus_documents:
                        if item.canonical_url not in existing_canonical:
                            documents.append(item)
                            existing_canonical.add(item.canonical_url)
                    failed.update(corpus_failed)
                    seen.update(targets)
                    queue.clear()
                else:
                    for discovered_url in discovery.urls:
                        if discovered_url not in seen:
                            queue.append((discovered_url, 1))
            if source.crawl and depth < source.max_depth:
                for link in document.links:
                    if link not in seen and self.policy.is_crawl_candidate(link, source):
                        discovered_urls.add(link)
                        queue.append((link, depth + 1))
        manifest = CorpusManifest(
            root_url=source.url,
            platform=platform,
            version_alignment=source.version_alignment,
            strategies=strategies,
            discovered_urls=tuple(sorted(discovered_urls)),
            fetched_urls=tuple(document.canonical_url for document in documents),
            failed_urls=failed,
            truncated=truncated or bool(queue),
        )
        return FetchResult(documents=tuple(documents), manifest=manifest)

    def _fetch_parallel(
        self, urls: list[str], source: DocumentationSource
    ) -> tuple[list[SourceDocument], dict[str, str], int]:
        if not urls:
            return [], {}, 0
        indexed: dict[str, int] = {url: index for index, url in enumerate(urls)}
        documents: list[SourceDocument] = []
        failed: dict[str, str] = {}
        total_bytes = 0

        def load(url: str) -> tuple[SourceDocument, int]:
            clean_url = self.policy.validate(url, source)
            response = self._fetch_following_redirects(clean_url, source)
            if not 200 <= response.status < 300:
                raise ValueError(
                    f"documentation request failed with HTTP {response.status}: {clean_url}"
                )
            media_type = response.headers.get("content-type", "text/plain").split(";", 1)[0].lower()
            if media_type not in self._accepted_types:
                raise ValueError(
                    f"unsupported documentation content type {media_type}: {clean_url}"
                )
            canonical = self.policy.validate(response.url, source)
            document = make_source_document(
                url=clean_url,
                canonical_url=canonical,
                content=response.content,
                content_type=response.headers.get("content-type", media_type),
                status=response.status,
                headers=response.headers,
            )
            return document, len(response.content)

        with ThreadPoolExecutor(max_workers=source.max_workers) as executor:
            futures = {executor.submit(load, url): url for url in urls}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    document, byte_count = future.result()
                    documents.append(document)
                    total_bytes += byte_count
                except (OSError, ValueError) as error:
                    failed[url] = str(error)
        documents.sort(key=lambda item: indexed.get(item.url, len(indexed)))
        return documents, failed, total_bytes

    def _fetch_following_redirects(self, url: str, source: DocumentationSource) -> FetchResponse:
        current = url
        for _ in range(6):
            current = self.policy.validate(current, source)
            response = self.transport.fetch(
                current, timeout=source.timeout_seconds, max_bytes=source.max_response_bytes
            )
            if response.status not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("location")
            if not location:
                raise ValueError(f"redirect without Location header: {current}")
            from urllib.parse import urljoin

            current = urljoin(current, location)
        raise ValueError(f"too many redirects while fetching documentation: {url}")
