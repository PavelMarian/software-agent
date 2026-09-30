"""Forward-compatible contracts for software operation, evidence, and isolation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class SoftwareInterface(str, Enum):
    CLI = "cli"
    API = "api"
    GUI = "gui"
    FILE = "file"
    DOCUMENTATION = "documentation"


@dataclass(frozen=True)
class KnowledgeFragment:
    """A compact, attributable piece of task or software knowledge."""

    id: str
    content: str
    source: str = "task"
    relevance: float = 1.0
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.content:
            raise ValueError("knowledge fragment id and content must be non-empty")
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError("knowledge relevance must be in [0, 1]")


@dataclass(frozen=True)
class KnowledgeContext:
    """Bounded context assembled independently from conversational history."""

    fragments: tuple[KnowledgeFragment, ...] = ()
    max_characters: int = 6_000

    def render(self) -> str:
        if self.max_characters <= 0:
            return ""
        ordered = sorted(
            self.fragments,
            key=lambda item: (-item.relevance, item.id),
        )
        lines: list[str] = []
        used = 0
        for fragment in ordered:
            dependencies = (
                f"; depends_on={','.join(fragment.depends_on)}"
                if fragment.depends_on
                else ""
            )
            line = f"[{fragment.id}; source={fragment.source}{dependencies}] {fragment.content}"
            remaining = self.max_characters - used
            if remaining <= 0:
                break
            clipped = line[:remaining]
            lines.append(clipped)
            used += len(clipped) + 1
        return "\n".join(lines)


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    media_type: str = "application/octet-stream"
    sha256: str | None = None
    producer: str | None = None
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = PurePosixPath(self.path.replace("\\", "/"))
        if not self.path or normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError("artifact path must be workspace-relative")


@dataclass(frozen=True)
class Evidence:
    id: str
    source: str
    summary: str = ""
    artifacts: tuple[ArtifactRef, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)
    software_native: bool = False


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class VerificationResult:
    check_id: str
    status: VerificationStatus
    evidence: tuple[Evidence, ...] = ()
    repair_targets: tuple[str, ...] = ()
    summary: str = ""

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED


@dataclass(frozen=True)
class RecoveryDecision:
    """Runtime-owned response to a failed independent verification."""

    retry: bool
    rollback: bool = False
    reason: str = ""


@dataclass(frozen=True)
class WorkItem:
    id: str
    objective: str
    depends_on: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    artifact_inputs: tuple[ArtifactRef, ...] = ()
    mutates_workspace: bool = False


@dataclass(frozen=True)
class WorkspaceLease:
    id: str
    path: Path
    isolated: bool
    base_revision: str | None = None


@dataclass(frozen=True)
class IntegrationResult:
    accepted: bool
    revision: str | None = None
    conflicts: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
