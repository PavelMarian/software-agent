from __future__ import annotations

import base64
import hashlib
from typing import Any, Mapping, TypeAlias

from software_bench.core.models import ValidationError


EncodedArtifact: TypeAlias = str | Mapping[str, str]


def encode_artifact(content: bytes) -> EncodedArtifact:
    """Keep ordinary text readable and encode arbitrary bytes losslessly."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "encoding": "base64",
            "data": base64.b64encode(content).decode("ascii"),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


def decode_artifact(value: Any, *, path: str = "artifact") -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be UTF-8 text or a base64 artifact object")
    if set(value) != {"encoding", "data", "sha256"} or value.get("encoding") != "base64":
        raise ValidationError(f"{path} has an invalid artifact object")
    data = value.get("data")
    digest = value.get("sha256")
    if not isinstance(data, str) or not isinstance(digest, str):
        raise ValidationError(f"{path} base64 data and sha256 must be strings")
    try:
        content = base64.b64decode(data, validate=True)
    except ValueError as error:
        raise ValidationError(f"{path} contains invalid base64") from error
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValidationError(f"{path} sha256 does not match decoded content")
    return content


def artifact_size(value: EncodedArtifact) -> int:
    return len(decode_artifact(value))
