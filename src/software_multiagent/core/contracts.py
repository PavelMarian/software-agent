"""Stable, provider-independent contracts shared by every runtime component."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from software_multiagent.core.execution import Evidence, WorkItem
from software_multiagent.core.action_graph import ActionGraph
from software_multiagent.core.execution import VerificationResult


class Topology(str, Enum):
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"


class RoutingMode(str, Enum):
    """Runtime routing semantics selected by a declarative workflow preset."""

    STATIC = "static"
    MEMORY_GATED_RESEARCH = "memory_gated_research"


class StepStatus(str, Enum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    NEEDS_REVISION = "needs_revision"
    FAILED = "failed"


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    objective: str
    workspace: Path
    target_software: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSpec:
    id: str
    instructions: str
    tools: tuple[str, ...] = ()
    max_turns: int = 12
    terminal_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.instructions:
            raise ValueError("agent id and instructions must be non-empty")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if len(self.tools) != len(set(self.tools)):
            raise ValueError(f"agent {self.id} has duplicate tools")
        if not set(self.terminal_tools).issubset(self.tools):
            raise ValueError(f"agent {self.id} terminal tools must be allowed tools")


@dataclass(frozen=True)
class Message:
    sender: str
    content: str
    recipient: str | None = None
    kind: str = "text"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    output: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class ModelTurn:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    status: StepStatus = StepStatus.CONTINUE
    route: str | None = None
    usage: Usage = Usage()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowNode:
    id: str
    agent_id: str
    transitions: Mapping[str, str | None]
    max_visits: int = 1
    max_tool_calls: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_visits <= 0:
            raise ValueError("max_visits must be positive")
        if self.max_tool_calls is not None and self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive when specified")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when specified")


@dataclass(frozen=True)
class RunSpec:
    topology: Topology
    agents: tuple[AgentSpec, ...]
    nodes: tuple[WorkflowNode, ...]
    entry_node: str
    max_node_executions: int = 20
    routing_mode: RoutingMode = RoutingMode.STATIC

    def __post_init__(self) -> None:
        agent_ids = [agent.id for agent in self.agents]
        node_ids = [node.id for node in self.nodes]
        if not self.agents or len(agent_ids) != len(set(agent_ids)):
            raise ValueError("run requires uniquely named agents")
        if not self.nodes or len(node_ids) != len(set(node_ids)):
            raise ValueError("run requires uniquely named workflow nodes")
        if self.entry_node not in node_ids:
            raise ValueError("entry_node must name a workflow node")
        if any(node.agent_id not in agent_ids for node in self.nodes):
            raise ValueError("every workflow node must reference a declared agent")
        targets = {
            target
            for node in self.nodes
            for target in node.transitions.values()
            if target is not None
        }
        if not targets.issubset(node_ids):
            raise ValueError("workflow transition references an unknown node")
        if self.max_node_executions <= 0:
            raise ValueError("max_node_executions must be positive")
        if self.topology == Topology.SINGLE_AGENT and len(self.agents) != 1:
            raise ValueError("single-agent topology requires exactly one agent")


@dataclass
class RuntimeState:
    task: TaskContext
    current_node: str
    messages: list[Message] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    work_items: dict[str, WorkItem] = field(default_factory=dict)
    action_graph: ActionGraph = field(default_factory=ActionGraph)
    verifications: list[VerificationResult] = field(default_factory=list)
    repair_attempts: dict[str, int] = field(default_factory=dict)
    node_visits: dict[str, int] = field(default_factory=dict)
    node_tool_calls: dict[str, int] = field(default_factory=dict)
    node_tokens: dict[str, int] = field(default_factory=dict)
    node_executions: int = 0
    model_turns: int = 0
    usage: Usage = Usage()
    measurement_complete: bool = True


@dataclass(frozen=True)
class RunResult:
    status: StepStatus
    stop_reason: str
    messages: tuple[Message, ...]
    artifacts: Mapping[str, Any]
    evidence: tuple[Evidence, ...]
    work_items: Mapping[str, WorkItem]
    node_visits: Mapping[str, int]
    usage: Usage
    measurement_complete: bool
    model_turns: int
    verifications: tuple[VerificationResult, ...] = ()
    journal: tuple[Mapping[str, Any], ...] = ()
    journal_path: str | None = None
