"""Runtime-owned action contracts and pre-execution grounding."""
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from software_multiagent.core.contracts import AgentSpec, TaskContext, ToolCall, ToolResult
from software_multiagent.core.execution import VerificationResult
from software_multiagent.runtime.reliability.action_validation import _validate_arguments
from software_multiagent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolContract:
    # Resolvers are trusted adapter code, never model-supplied declarations.
    inputs: Callable[[Mapping[str, Any]], tuple[str, ...]] = lambda args: ()
    outputs: Callable[[Mapping[str, Any]], tuple[str, ...]] = lambda args: ()
    file_mutations_only: bool = False
    preconditions: tuple[Callable[[Any, TaskContext], VerificationResult], ...] = ()
    result_schema: Mapping[str, Any] | None = None
    poll_tool: str | None = None
    operation_key: str = "session_id"
    poll_argument: str = "session_id"
    description: str = ""


class ToolGateway:
    """All reliable-runtime calls pass through this gateway; handlers run once.

    Permission violations retain host abort semantics. Argument, grounding and
    precondition failures are observations that the agent can correct.
    """

    def __init__(self, registry: ToolRegistry, contracts: Mapping[str, ToolContract] | None = None):
        self.registry = registry
        self.contracts = dict(contracts or {})

    def contract_for(self, name):
        return self.contracts.get(name) or self.registry.contract_for(name) or ToolContract()

    def schemas_for(self, agent):
        schemas = self.registry.schemas_for(agent)
        names = {s["name"] for s in schemas}
        for name in names:
            poll = self.contract_for(name).poll_tool
            if poll and poll not in names:
                raise ValueError(f"poll tool {poll} for {name} must be registered and allowed")
        return tuple({**schema, "description": schema["description"] +
                      ("\nAction contract: " + self.contract_for(schema["name"]).description
                       if self.contract_for(schema["name"]).description else "")}
                     for schema in schemas)

    def prepare(self, agent: AgentSpec, call: ToolCall, task: TaskContext):
        if call.name not in agent.tools:
            raise PermissionError(f"tool not allowed for {agent.id}: {call.name}")
        schemas = {s["name"]: s for s in self.schemas_for(agent)}
        _validate_arguments(call.arguments, schemas[call.name]["input_schema"])
        contract = self.contract_for(call.name)
        action = replace(self.registry.action_for(call),
                         artifact_inputs=tuple(contract.inputs(call.arguments)),
                         artifact_outputs=tuple(contract.outputs(call.arguments)))
        # File-only declarations are checked before snapshots or tool dispatch.
        if contract.file_mutations_only:
            from software_multiagent.runtime.reliability.reliable_solo import VerifiedFileRollback
            for relative in action.artifact_inputs + action.artifact_outputs:
                VerifiedFileRollback._path(task, relative)
            if action.mutates_workspace and not action.artifact_outputs:
                raise ValueError("file mutation contract declares no outputs")
            for relative in action.artifact_inputs:
                if not VerifiedFileRollback._path(task, relative).is_file():
                    raise ValueError(f"required input file does not exist: {relative}")
        return action

    def execute(self, agent, call, task) -> ToolResult:
        result = self.registry.execute(agent, call, task)
        schema = self.contract_for(call.name).result_schema
        if result.ok and schema is not None:
            try:
                _validate_arguments(result.output, schema)
            except (ValueError, TypeError) as error:
                return replace(result, error=f"Invalid tool result: {error}",
                               metadata={"failure_kind": "artifact"})
        return result
