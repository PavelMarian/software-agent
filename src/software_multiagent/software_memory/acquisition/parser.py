from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

from software_multiagent.software_memory.acquisition.models import PageType, SourceDocument, SourceSection, stable_id


@dataclass(frozen=True)
class ParsedContent:
    title: str | None
    text: str
    sections: tuple[tuple[tuple[str, ...], str], ...]
    links: tuple[str, ...]


class _DocumentationHtmlParser(HTMLParser):
    _ignored = {"script", "style", "noscript", "svg", "canvas", "nav", "footer"}
    _block = {
        "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
        "figure", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main",
        "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
    }

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self.links: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False
        self._in_pre = False
        self._heading_level: int | None = None
        self._heading_text: list[str] = []
        self.headings: list[tuple[int, str, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._ignored:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag == "pre":
            self._in_pre = True
            self.parts.append("\n```\n")
        if tag in self._block and tag != "pre":
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self._heading_text = []
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                absolute = urljoin(self.base_url, href.strip())
                if absolute.startswith(("http://", "https://")):
                    self.links.append(absolute)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._ignored:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag == "pre":
            self.parts.append("\n```\n")
            self._in_pre = False
        if self._heading_level is not None and tag == f"h{self._heading_level}":
            heading = _clean_inline(" ".join(self._heading_text))
            if heading:
                self.headings.append((self._heading_level, heading, len(self.parts)))
                self.parts.append(f"\n{'#' * self._heading_level} {heading}\n")
            self._heading_level = None
            self._heading_text = []
        elif tag in self._block and tag != "pre":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        if self._heading_level is not None:
            self._heading_text.append(data)
            return
        if self._in_pre:
            self.parts.append(data)
        else:
            cleaned = _clean_inline(data)
            if cleaned:
                self.parts.append(cleaned + " ")


def _clean_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_document(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _markdown_sections(text: str) -> tuple[tuple[tuple[str, ...], str], ...]:
    headings: list[str] = []
    current: list[str] = []
    current_path: tuple[str, ...] = ()
    sections: list[tuple[tuple[str, ...], str]] = []

    def flush() -> None:
        content = "\n".join(current).strip()
        if content:
            sections.append((current_path, content))

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            current.append(line)
            continue
        flush()
        current.clear()
        level = len(match.group(1))
        heading = match.group(2).strip()
        headings[level - 1 :] = [heading]
        current_path = tuple(headings)
    flush()
    return tuple(sections) or (((), text),)


def parse_content(content: bytes, content_type: str, url: str) -> ParsedContent:
    charset_match = re.search(r"charset=([\w-]+)", content_type, re.I)
    encoding = charset_match.group(1) if charset_match else "utf-8"
    text = content.decode(encoding, errors="replace")
    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type in {"application/json", "application/vnd.oai.openapi+json"} or url.lower().endswith(".json"):
        try:
            parsed: Any = json.loads(text)
            pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
            title = parsed.get("info", {}).get("title") if isinstance(parsed, dict) else None
            return ParsedContent(title, pretty, (((), pretty),), ())
        except json.JSONDecodeError:
            pass

    if media_type in {"text/html", "application/xhtml+xml"} or "<html" in text[:500].lower():
        parser = _DocumentationHtmlParser(url)
        parser.feed(text)
        cleaned = _clean_document("".join(parser.parts))
        title = _clean_inline(" ".join(parser.title_parts)) or None
        return ParsedContent(title, cleaned, _markdown_sections(cleaned), tuple(dict.fromkeys(parser.links)))

    cleaned = _clean_document(text)
    return ParsedContent(None, cleaned, _markdown_sections(cleaned), ())


def classify_page(parsed: ParsedContent, content_type: str, url: str = "") -> PageType:
    sample = f"{parsed.title or ''}\n{parsed.text[:20_000]}".lower()
    slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1].lower()
    if slug in {"overview", "introduction", "index", "home"}:
        return PageType.OVERVIEW
    if "generated by doxygen" in sample:
        return PageType.API_REFERENCE
    if re.fullmatch(r"v?\d+(?:\.\d+)*", slug):
        return PageType.OVERVIEW
    endpoint_hits = len(
        re.findall(
            r"\b(get|post|put|patch|delete|head|options)\s+(?:https?://|/)[^\s`|]+",
            sample,
        )
    )
    if "openapi" in sample or "swagger" in sample or endpoint_hits >= 1:
        return PageType.API_REFERENCE
    if any(term in sample for term in ("authentication", "authorization", "api key", "oauth")):
        return PageType.AUTHENTICATION
    if any(term in sample for term in ("error codes", "errors", "status codes")):
        return PageType.ERROR_REFERENCE
    if any(term in sample for term in ("changelog", "release notes", "deprecated")):
        return PageType.VERSION_NOTE
    if any(term in sample for term in ("quickstart", "getting started", "tutorial")):
        return PageType.TUTORIAL
    return PageType.OVERVIEW


def make_source_document(
    *,
    url: str,
    canonical_url: str,
    content: bytes,
    content_type: str,
    status: int,
    headers: dict[str, str],
) -> SourceDocument:
    parsed = parse_content(content, content_type, canonical_url)
    source_id = stable_id("src", canonical_url, sha256(content).hexdigest())
    sections = tuple(
        SourceSection(
            section_id=stable_id("sec", source_id, str(index), "/".join(path)),
            source_id=source_id,
            heading_path=path,
            content=section_text,
            ordinal=index,
        )
        for index, (path, section_text) in enumerate(parsed.sections)
    )
    return SourceDocument(
        source_id=source_id,
        url=url,
        canonical_url=canonical_url,
        content_type=content_type,
        title=parsed.title,
        page_type=classify_page(parsed, content_type, canonical_url),
        content_hash=sha256(content).hexdigest(),
        raw_text=parsed.text,
        links=parsed.links,
        sections=sections,
        http_status=status,
        etag=headers.get("etag"),
        last_modified=headers.get("last-modified"),
    )
