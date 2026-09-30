"""Typed tools and permission-aware dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from software_multiagent.core.action_graph import Action
from software_multiagent.core.contracts import AgentSpec, TaskContext, ToolCall, ToolResult
from software_multiagent.core.execution import SoftwareInterface
from software_multiagent.ports.protocols import Tool


class ActionExecutionError(RuntimeError):
    """A tool action ran but reported an unsuccessful execution outcome."""

    failure_kind = "action"


@dataclass(frozen=True)
class CallableTool:
    name: str
    description: str
    handler: Callable[[Mapping[str, Any], TaskContext], Any]
    input_schema: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "additionalProperties": True}
    )
    mutates_workspace: bool = False
    interface: SoftwareInterface = SoftwareInterface.CLI
    verification: tuple[str, ...] = ()
    rollback_tool: str | None = None
    expected_effect: str = ""
    action_contract: Any = None

    def execute(self, arguments: Mapping[str, Any], task: TaskContext) -> Any:
        return self.handler(arguments, task)


@dataclass(frozen=True)
class AgentCallableTool:
    """A shared tool whose concrete backend call depends on the active agent."""

    name: str
    description: str
    handler: Callable[[AgentSpec, Mapping[str, Any], TaskContext], Any]
    input_schema: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "additionalProperties": True}
    )
    mutates_workspace: bool = False
    interface: SoftwareInterface = SoftwareInterface.CLI
    verification: tuple[str, ...] = ()
    rollback_tool: str | None = None
    expected_effect: str = ""
    action_contract: Any = None

    def execute_for(
        self,
        agent: AgentSpec,
        arguments: Mapping[str, Any],
        task: TaskContext,
    ) -> Any:
        return self.handler(agent, arguments, task)


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def contract_for(self, name):
        return getattr(self._require(name), "action_contract", None)

    def schemas_for(self, agent: AgentSpec) -> tuple[Mapping[str, Any], ...]:
        schemas = []
        for name in agent.tools:
            tool = self._require(name)
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": dict(tool.input_schema),
                    "mutates_workspace": tool.mutates_workspace,
                    "interface": getattr(tool, "interface", SoftwareInterface.CLI).value,
                    "verification": list(getattr(tool, "verification", ())),
                }
            )
        return tuple(schemas)

    def action_for(self, call: ToolCall) -> Action:
        """Convert a model tool call into the runtime's auditable action contract."""
        tool = self._require(call.name)
        output_path = call.arguments.get("path")
        outputs = (output_path,) if isinstance(output_path, str) and output_path else ()
        rollback_tool = getattr(tool, "rollback_tool", None)
        return Action(
            id=call.id,
            tool=call.name,
            arguments=dict(call.arguments),
            mutates_workspace=bool(tool.mutates_workspace),
            verification=tuple(getattr(tool, "verification", ())),
            rollback=rollback_tool,
            artifact_outputs=outputs,
            interface=getattr(tool, "interface", SoftwareInterface.CLI),
            expected_effect=str(getattr(tool, "expected_effect", "") or tool.description),
            reversible=bool(rollback_tool or outputs),
        )

    def execute(self, agent: AgentSpec, call: ToolCall, task: TaskContext) -> ToolResult:
        if call.name not in agent.tools:
            raise PermissionError(f"tool not allowed for {agent.id}: {call.name}")
        try:
            tool = self._require(call.name)
            _validate_arguments(call.arguments, tool.input_schema)
            execute_for = getattr(tool, "execute_for", None)
            output = (
                execute_for(agent, call.arguments, task)
                if callable(execute_for)
                else tool.execute(call.arguments, task)
            )
            if isinstance(output, ToolResult):
                if output.call_id != call.id or output.tool_name != call.name:
                    raise ValueError("tool result identity does not match the call")
                return output
            return ToolResult(call.id, call.name, output=output)
        except PermissionError:
            # A host-enforced capability violation invalidates the run; exposing it
            # as ordinary tool feedback would let the model probe forbidden access.
            raise
        except Exception as error:  # tools are an integration boundary
            if getattr(error, "abort_agent_runtime", False):
                raise
            kind = getattr(error, "failure_kind", None)
            if kind not in {"action", "call", "state", "environment", "artifact",
                            "approach", "verification"}:
                kind = ("state" if isinstance(error, FileNotFoundError) else
                        "environment" if isinstance(error, (OSError, TimeoutError)) else "call")
            return ToolResult(call.id, call.name, error=f"{type(error).__name__}: {error}",
                              metadata={"failure_kind": kind})

    def _require(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ValueError(f"unknown tool: {name}") from error


def _validate_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate the small JSON-Schema subset used by benchmark tool declarations."""
    required = schema.get("required", ())
    if isinstance(required, (list, tuple)):
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(f"unknown arguments: {', '.join(unknown)}")
    for name, value in arguments.items():
        item = properties.get(name)
        if not isinstance(item, Mapping):
            continue
        expected = item.get("type")
        valid = {
            "string": isinstance(value, str),
            "array": isinstance(value, (list, tuple)),
            "object": isinstance(value, Mapping),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }.get(expected, True)
        if not valid:
            raise ValueError(f"argument {name} must have type {expected}")
        if isinstance(value, str) and len(value) < int(item.get("minLength", 0)):
            raise ValueError(f"argument {name} is too short")
        if isinstance(value, (list, tuple)) and len(value) < int(item.get("minItems", 0)):
            raise ValueError(f"argument {name} has too few items")
        choices = item.get("enum")
        if isinstance(choices, (list, tuple)) and value not in choices:
            raise ValueError(f"argument {name} is not one of the allowed values")
