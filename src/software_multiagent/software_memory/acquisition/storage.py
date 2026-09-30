from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from software_multiagent.software_memory.acquisition.models import (
    ApiCatalog,
    ApiOperation,
    Conflict,
    OperationFingerprint,
    RepositorySnapshot,
    SourceDocument,
    SourceSection,
)


def _safe_segment(value: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
    return segment or "unknown"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _jsonl(values: Iterable[BaseModel]) -> str:
    lines = [value.model_dump_json(exclude_none=True) for value in values]
    return "\n".join(lines) + ("\n" if lines else "")


class FileCatalogStore:
    """Human-inspectable, versioned JSON/JSONL store with atomic file replacement."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def catalog_path(self, software: str, version: str) -> Path:
        return self.root / _safe_segment(software) / _safe_segment(version)

    def save(self, catalog: ApiCatalog) -> Path:
        directory = self.catalog_path(catalog.software, catalog.version)
        sections = [section for source in catalog.sources for section in source.sections]
        sources_without_sections = [source.model_copy(update={"sections": ()}) for source in catalog.sources]
        manifest = {
            "schema_version": 1,
            "software": catalog.software,
            "version": catalog.version,
            "created_at": catalog.created_at.isoformat(),
            "counts": {
                "sources": len(catalog.sources),
                "sections": len(sections),
                "operations": len(catalog.operations),
                "fingerprints": len(catalog.fingerprints),
                "conflicts": len(catalog.conflicts),
            },
            "corpus": (
                catalog.corpus_manifest.model_dump(mode="json")
                if catalog.corpus_manifest is not None
                else None
            ),
        }
        _atomic_text(directory / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        _atomic_text(directory / "sources.jsonl", _jsonl(sources_without_sections))
        _atomic_text(directory / "sections.jsonl", _jsonl(sections))
        _atomic_text(directory / "operations.jsonl", _jsonl(catalog.operations))
        _atomic_text(directory / "fingerprints.jsonl", _jsonl(catalog.fingerprints))
        _atomic_text(directory / "conflicts.jsonl", _jsonl(catalog.conflicts))
        return directory

    def save_repository_snapshot(
        self, software: str, version: str, snapshot: RepositorySnapshot
    ) -> Path:
        path = self.catalog_path(software, version) / "repository_snapshot.json"
        _atomic_text(path, snapshot.model_dump_json(indent=2) + "\n")
        return path

    def load(self, software: str, version: str) -> ApiCatalog:
        directory = self.catalog_path(software, version)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        sections = self._read_jsonl(directory / "sections.jsonl", SourceSection)
        by_source: dict[str, list[SourceSection]] = {}
        for section in sections:
            by_source.setdefault(section.source_id, []).append(section)
        raw_sources = self._read_jsonl(directory / "sources.jsonl", SourceDocument)
        sources = tuple(
            source.model_copy(update={"sections": tuple(by_source.get(source.source_id, ()))})
            for source in raw_sources
        )
        return ApiCatalog(
            software=manifest["software"],
            version=manifest["version"],
            created_at=manifest["created_at"],
            sources=sources,
            operations=tuple(self._read_jsonl(directory / "operations.jsonl", ApiOperation)),
            fingerprints=tuple(
                self._read_jsonl(directory / "fingerprints.jsonl", OperationFingerprint)
            ),
            conflicts=tuple(self._read_jsonl(directory / "conflicts.jsonl", Conflict)),
            corpus_manifest=manifest.get("corpus"),
        )

    @staticmethod
    def _read_jsonl(path: Path, model_type):  # type: ignore[no-untyped-def]
        if not path.exists():
            return []
        return [
            model_type.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
