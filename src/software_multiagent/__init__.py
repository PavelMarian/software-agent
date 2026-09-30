"""Domain-agnostic runtime for single agents and agent teams."""

__version__ = "0.1.0"

from software_multiagent.core.action_graph import Action, ActionGraph
from software_multiagent.core.contracts import (
    AgentSpec,
    Message,
    ModelTurn,
    RunResult,
    RunSpec,
    RoutingMode,
    StepStatus,
    TaskContext,
    ToolCall,
    ToolResult,
    Topology,
    WorkflowNode,
)
from software_multiagent.core.execution import (
    ArtifactRef,
    Evidence,
    IntegrationResult,
    SoftwareInterface,
    VerificationResult,
    VerificationStatus,
    WorkItem,
    WorkspaceLease,
)
from software_multiagent.ports.protocols import NativeVerifier, SoftwareAdapter, WorkspaceProvider
from software_multiagent.runtime.orchestration.engine import AgentRuntime
from software_multiagent.runtime.orchestration.langgraph import LangGraphMultiAgent
from software_multiagent.application import SoftwareMultiAgent, SoftwareRunRequest
from software_multiagent.runtime.orchestration.policies import (
    CoordinationPolicy,
    MemoryGatedResearchPolicy,
    StaticGraphPolicy,
)
from software_multiagent.runtime.orchestration.presets import (
    plan_execute_verify,
    plan_research_execute_verify,
    single_agent,
    solo_plan_execute_verify,
)
from software_multiagent.tools.registry import (
    ActionExecutionError, AgentCallableTool, CallableTool, ToolRegistry,
)

__all__ = [
    "Action",
    "ActionGraph",
    "ArtifactRef",
    "AgentRuntime",
    "LangGraphMultiAgent",
    "SoftwareMultiAgent",
    "SoftwareRunRequest",
    "AgentSpec",
    "AgentCallableTool",
    "ActionExecutionError",
    "CallableTool",
    "CoordinationPolicy",
    "Evidence",
    "IntegrationResult",
    "Message",
    "ModelTurn",
    "NativeVerifier",
    "RunResult",
    "RunSpec",
    "RoutingMode",
    "SoftwareInterface",
    "SoftwareAdapter",
    "StaticGraphPolicy",
    "MemoryGatedResearchPolicy",
    "StepStatus",
    "TaskContext",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "Topology",
    "VerificationResult",
    "VerificationStatus",
    "WorkItem",
    "WorkflowNode",
    "WorkspaceLease",
    "WorkspaceProvider",
    "plan_execute_verify",
    "plan_research_execute_verify",
    "single_agent",
    "solo_plan_execute_verify",
    "__version__",
]

from software_multiagent.runtime.reliability.patterns import (
    ExecutionFeedback, HookContext, HookResult, Mechanism, MechanismRegistry,
    OutcomeChecks, Phase, TaskKnowledge, default_mechanisms,
)
from software_multiagent.runtime.reliability.reliable_solo import (
    ReliableSingleAgent, SoloLimits, VerifiedFileRollback,
)
from software_multiagent.core.contracts import Usage

__all__ += [
    "ExecutionFeedback", "HookContext", "HookResult", "Mechanism", "MechanismRegistry",
    "OutcomeChecks", "Phase", "TaskKnowledge", "default_mechanisms",
    "ReliableSingleAgent", "SoloLimits", "VerifiedFileRollback", "Usage",
]

from software_multiagent.runtime.reasoning.react import (
    ReActDecision,
    ReActProtocol,
    ReActProtocolError,
)

__all__ += ["ReActDecision", "ReActProtocol", "ReActProtocolError"]

from software_multiagent.tools.gateway import ToolContract, ToolGateway
from software_multiagent.runtime.reasoning.context import BoundedContext
from software_multiagent.runtime.reliability.journal import ExecutionJournal
from software_multiagent.runtime.reliability.repair import Diagnosis, FailureKind, RepairPolicy
from software_multiagent.runtime.reliability.validators import FileContentCheck, NativeCommandCheck
from software_multiagent.runtime.reliability.patterns import PreExecutionGate, VerificationGate

__all__ += ["ToolContract", "ToolGateway", "BoundedContext", "ExecutionJournal",
            "Diagnosis", "FailureKind", "RepairPolicy", "FileContentCheck", "NativeCommandCheck",
            "PreExecutionGate", "VerificationGate"]
