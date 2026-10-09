from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class _TextExtractor(HTMLParser):
    """Small dependency-free HTML-to-text extractor for fetched pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data.strip())


def html_to_text(document: str, *, max_characters: int) -> str:
    parser = _TextExtractor()
    parser.feed(document)
    return "\n".join(parser.parts)[:max_characters]


class DirectWebSearch:
    """Bounded metasearch backed by DDGS search engines."""

    def __init__(
        self,
        *,
        timeout_seconds: int = 15,
        backend: str = "auto",
        fetch_pages: bool = True,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.backend = backend
        self.fetch_pages = fetch_pages

    @staticmethod
    def _allowed(url: str, domains: tuple[str, ...]) -> bool:
        if not domains:
            return True
        host = (urlparse(url).hostname or "").lower()
        return any(host == domain.lower() or host.endswith("." + domain.lower()) for domain in domains)

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        allowed_domains: tuple[str, ...] = (),
        max_content_chars: int = 64_000,
    ) -> list[Mapping[str, Any]]:
        if not query.strip():
            raise ValueError("web search query must be non-empty")
        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")
        if len(allowed_domains) > 10 or any(not item.strip() for item in allowed_domains):
            raise ValueError("allowed_domains must contain at most 10 non-empty domains")
        if not 4_000 <= max_content_chars <= 100_000:
            raise ValueError("max_content_chars must be between 4000 and 100000")
        try:
            from ddgs import DDGS
        except ImportError as error:
            raise RuntimeError("direct web search requires the 'ddgs' package") from error

        search_query = query.strip()
        if allowed_domains:
            sites = " OR ".join(f"site:{domain}" for domain in allowed_domains)
            search_query = f"{search_query} ({sites})"
        raw = DDGS(timeout=self.timeout_seconds).text(
            search_query,
            max_results=min(10, max_results * 2 if allowed_domains else max_results),
            backend=self.backend,
            safesearch="moderate",
        )
        results: list[Mapping[str, Any]] = []
        for item in raw:
            url = str(item.get("href") or item.get("url") or "")
            if not url.startswith(("http://", "https://")) or not self._allowed(url, allowed_domains):
                continue
            result: dict[str, Any] = {
                "title": str(item.get("title") or "")[:500],
                "url": url,
                "snippet": str(item.get("body") or item.get("description") or ""),
            }
            if self.fetch_pages:
                try:
                    content, truncated = self._fetch_page(url, max_content_chars)
                    result.update({
                        "content": content,
                        "content_chars": len(content),
                        "content_truncated": truncated,
                    })
                except Exception as error:
                    result.update({
                        "content": "",
                        "content_chars": 0,
                        "content_truncated": False,
                        "content_error": f"{type(error).__name__}: {error}",
                    })
            results.append(result)
            if len(results) >= max_results:
                break
        if not results:
            raise RuntimeError("web search returned no matching results")
        return results

    def _fetch_page(self, url: str, max_content_chars: int) -> tuple[str, bool]:
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; Software Benchmark research fetcher; +local benchmark)"
                ),
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            },
        )
        # Bound transfer size while allowing markup overhead beyond extracted text.
        byte_limit = min(2_000_000, max_content_chars * 8)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            content_type = str(response.headers.get("Content-Type", ""))
            if content_type and not any(
                marker in content_type.lower() for marker in ("text/", "html", "xml", "json")
            ):
                raise ValueError(f"unsupported page content type: {content_type}")
            raw = response.read(byte_limit + 1)
            transfer_truncated = len(raw) > byte_limit
            raw = raw[:byte_limit]
            charset = response.headers.get_content_charset() or "utf-8"
        parsed = html_to_text(raw.decode(charset, errors="replace"), max_characters=max_content_chars)
        return parsed, transfer_truncated or len(parsed) >= max_content_chars
