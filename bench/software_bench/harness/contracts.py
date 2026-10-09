from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from software_bench.core.artifacts import EncodedArtifact
from software_bench.core.config import ModeSpec
from software_bench.core.models import AgentTaskView, SubmissionKind
from software_bench.harness.environments import EnvironmentSession


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    done: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    measurement_complete: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    """Preferred integration: benchmark owns roles, tools, and orchestration."""

    adapter_id: str

    def generate(
        self,
        messages: list[Mapping[str, Any]],
        system_prompt: str,
        tools: list[Mapping[str, Any]],
        *,
        role_id: str,
        seed: int,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = True,
    ) -> ModelResponse: ...


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    task: AgentTaskView
    mode: ModeSpec
    environment: EnvironmentSession
    seed: int
    submission_kind: SubmissionKind = SubmissionKind.PATCH
    submission_paths: tuple[str, ...] = ()
    submission_root: str | None = None
    state_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameworkOutcome:
    patch: str = ""
    files: Mapping[str, EncodedArtifact] = field(default_factory=dict)
    evidence: Mapping[str, EncodedArtifact] = field(default_factory=dict)
    submission_kind: SubmissionKind = SubmissionKind.PATCH
    status: str = "completed"
    stop_reason: str = "framework_completed"
    measurement_complete: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class FrameworkObserver(Protocol):
    def record(
        self,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
    ) -> None: ...


class FrameworkAdapter(Protocol):
    """Secondary integration for black-box MAS frameworks; requires compliance checks."""

    adapter_id: str

    def run(self, request: RunRequest, observer: FrameworkObserver) -> FrameworkOutcome: ...


class AgentRuntimeAdapter(Protocol):
    """Pluggable orchestration that keeps benchmark-owned execution boundaries."""

    runtime_id: str
    version: str

    def run(
        self,
        request: RunRequest,
        adapter: ModelAdapter,
        observer: FrameworkObserver,
    ) -> Any: ...
