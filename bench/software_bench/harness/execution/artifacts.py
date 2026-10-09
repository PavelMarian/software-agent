from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from software_bench.core.artifacts import EncodedArtifact, decode_artifact

def _strip_submission_root(
    artifacts: Mapping[str, EncodedArtifact], root: str
) -> Mapping[str, EncodedArtifact]:
    prefix = root.rstrip("/") + "/"
    result: dict[str, EncodedArtifact] = {}
    for name, content in artifacts.items():
        normalized = name.replace("\\", "/")
        if not normalized.startswith(prefix):
            raise ValueError(f"captured artifact is outside submission_root: {name}")
        relative = normalized[len(prefix):]
        if relative:
            result[relative] = content
    return result


def _materialize_artifacts(
    artifacts: Mapping[str, EncodedArtifact], root: Path, *, field: str
) -> None:
    for name, content in artifacts.items():
        target = (root / name).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as error:
            raise PermissionError(f"artifact path escapes {field}: {name}") from error
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(decode_artifact(content, path=f"{field}.{name}"))


def _artifact_manifest(
    artifacts: Mapping[str, EncodedArtifact],
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for name, content in sorted(artifacts.items()):
        raw = decode_artifact(content, path=f"artifacts.{name}")
        result.append(
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
        )
    return result
