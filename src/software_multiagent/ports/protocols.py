"""Replaceable model, tool, observability, and persistence boundaries."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from software_multiagent.core.contracts import (
    AgentSpec,
    Message,
    ModelTurn,
    RuntimeState,
    TaskContext,
    ToolResult,
)
from software_multiagent.core.action_graph import Action
from software_multiagent.core.execution import (
    ArtifactRef,
    IntegrationResult,
    KnowledgeContext,
    RecoveryDecision,
    SoftwareInterface,
    VerificationResult,
    WorkItem,
    WorkspaceLease,
)


class ModelClient(Protocol):
    def generate(
        self,
        agent: AgentSpec,
        task: TaskContext,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]],
    ) -> ModelTurn: ...


class Tool(Protocol):
    name: str
    description: str
    input_schema: Mapping[str, Any]
    mutates_workspace: bool

    def execute(self, arguments: Mapping[str, Any], task: TaskContext) -> Any: ...


class EventSink(Protocol):
    def emit(self, event_type: str, actor: str, payload: Mapping[str, Any]) -> None: ...


class CheckpointStore(Protocol):
    def save(self, checkpoint_id: str, state: RuntimeState) -> None: ...

    def load(self, checkpoint_id: str) -> RuntimeState: ...


class KnowledgeProvider(Protocol):
    """Build a bounded context pack from documentation and dependency metadata."""

    def context_for(
        self,
        task: TaskContext,
        agent: AgentSpec,
        state: RuntimeState,
    ) -> KnowledgeContext: ...


class KnowledgeObserver(Protocol):
    """Optional write-through hook implemented by a shared memory service."""

    def observe_action(
        self,
        task: TaskContext,
        agent: AgentSpec,
        state: RuntimeState,
        action: Action,
        result: ToolResult,
        verification: VerificationResult | None,
    ) -> None: ...


class ActionValidator(Protocol):
    """Validate an executed action using software-native observable results."""

    def verify(
        self,
        action: Action,
        result: ToolResult,
        task: TaskContext,
    ) -> VerificationResult | None: ...


class RecoveryPolicy(Protocol):
    def decide(
        self,
        action: Action,
        verification: VerificationResult,
        attempt: int,
    ) -> RecoveryDecision: ...


class RollbackManager(Protocol):
    """Capture and restore external state for actions it knows how to reverse."""

    def checkpoint(self, action: Action, task: TaskContext) -> Any: ...

    def rollback(
        self,
        token: Any,
        action: Action,
        task: TaskContext,
    ) -> Evidence | None: ...


class SoftwareAdapter(Protocol):
    """One target-software boundary; CLI, API, and GUI stay distinct modalities."""

    id: str
    interfaces: tuple[SoftwareInterface, ...]

    def capabilities(self) -> Sequence[Mapping[str, Any]]: ...

    def invoke(
        self,
        capability: str,
        arguments: Mapping[str, Any],
        task: TaskContext,
    ) -> Any: ...


class NativeVerifier(Protocol):
    """Verification owned by target software or benchmark, never self-report alone."""

    def verify(
        self,
        task: TaskContext,
        artifacts: Sequence[ArtifactRef],
    ) -> VerificationResult: ...


class WorkspaceProvider(Protocol):
    """Isolation and explicit integration boundary for concurrent mutations."""

    def acquire(self, work_item: WorkItem) -> WorkspaceLease: ...

    def integrate(self, lease: WorkspaceLease) -> IntegrationResult: ...


class NullEventSink:
    def emit(self, event_type: str, actor: str, payload: Mapping[str, Any]) -> None:
        del event_type, actor, payload


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self.checkpoints: dict[str, RuntimeState] = {}

    def save(self, checkpoint_id: str, state: RuntimeState) -> None:
        self.checkpoints[checkpoint_id] = state

    def load(self, checkpoint_id: str) -> RuntimeState:
        try:
            return self.checkpoints[checkpoint_id]
        except KeyError as error:
            raise ValueError(f"unknown checkpoint: {checkpoint_id}") from error
