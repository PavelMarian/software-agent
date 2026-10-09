from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from software_multiagent.core.contracts import (
    AgentSpec,
    Message,
    ModelTurn,
    StepStatus,
    TaskContext,
    ToolCall,
    Usage,
)
from software_multiagent.ports.protocols import EventSink
from software_multiagent.runtime.orchestration.engine import AgentRuntime
from software_multiagent.runtime.orchestration.presets import plan_execute_verify, single_agent
from software_multiagent.tools.registry import AgentCallableTool, CallableTool, ToolRegistry


class BenchmarkModelClient:
    """Adapt benchmark model responses to the reusable agent runtime contract."""

    def __init__(
        self, adapter: Any, *, seed: int = 0, inbox: Any = None,
        observer: Any = None, tool_choice: str = "auto",
        parallel_tool_calls: bool = True,
    ) -> None:
        """Initialize the model bridge and per-role inbox offsets.

        Args:
            adapter: Benchmark-owned model adapter.
            seed: Reproducibility seed forwarded to the adapter.
            inbox: Optional callable returning messages for a role identifier.
            observer: Optional benchmark trace observer.
            tool_choice: Model tool-choice policy.
            parallel_tool_calls: Whether the provider may emit parallel calls.

        Raises:
            ValueError: If ``tool_choice`` is unsupported.
        """
        if tool_choice not in {"auto", "required", "none"}:
            raise ValueError("tool_choice must be auto, required, or none")
        self.adapter = adapter
        self.seed = seed
        self.inbox = inbox
        self.observer = observer
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self._inbox_offsets: dict[str, int] = {}

    def generate(
        self,
        agent: AgentSpec,
        task: TaskContext,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        """Generate and normalize one runtime model turn.

        Args:
            agent: Runtime agent requesting the turn.
            task: Shared task context.
            messages: Runtime conversation messages.
            tool_schemas: Tools visible to the requesting agent.

        Returns:
            A normalized model turn with tool calls, status, and usage.
        """
        delivered: list[Mapping[str, Any]] = []
        if self.inbox is not None:
            inbox = list(self.inbox(agent.id))
            offset = self._inbox_offsets.get(agent.id, 0)
            delivered = [
                {
                    "role": "user",
                    "content": (
                        f"Message from teammate {item['from']}:\n{item['content']}"
                    ),
                }
                for item in inbox[offset:]
            ]
            if delivered and self.observer is not None:
                self.observer.record(
                    "messages_received",
                    agent.id,
                    {
                        "count": len(delivered),
                        "message_ids": [
                            str(item["id"])
                            for item in inbox[offset:]
                            if item.get("id")
                        ],
                        "senders": sorted(
                            {
                                str(item.get("from", ""))
                                for item in inbox[offset:]
                                if item.get("from")
                            }
                        ),
                    },
                )
            self._inbox_offsets[agent.id] = len(inbox)
        policy = {}
        if self.tool_choice != "auto" or not self.parallel_tool_calls:
            policy = {
                "tool_choice": self.tool_choice,
                "parallel_tool_calls": self.parallel_tool_calls,
            }
        response = self.adapter.generate(
            [
                {"role": "user", "content": task.objective},
                *[
                    {
                        "role": "assistant" if message.sender == agent.id else "user",
                        "content": message.content,
                    }
                    for message in messages
                    if message.kind != "action"
                ],
                *delivered,
            ],
            agent.instructions,
            [
                {
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema.get(
                        "input_schema",
                        {"type": "object", "additionalProperties": False},
                    ),
                }
                for schema in tool_schemas
            ],
            role_id=agent.id,
            seed=self.seed,
            **policy,
        )
        text = response.text or ""
        normalized = text.upper()
        calls = tuple(
            ToolCall(
                id=str(call.get("id", f"call-{index}")),
                name=str(call["name"]),
                arguments=dict(call.get("args", {})),
            )
            for index, call in enumerate(response.tool_calls)
        )
        if calls:
            status = StepStatus.CONTINUE
        elif "NEEDS_REVISION" in normalized:
            status = StepStatus.NEEDS_REVISION
        elif "TASK_FAILED" in normalized:
            status = StepStatus.FAILED
        elif response.done or "TASK_COMPLETE" in normalized:
            status = StepStatus.COMPLETE
        else:
            status = StepStatus.CONTINUE
        return ModelTurn(
            content=text,
            tool_calls=calls,
            status=status,
            usage=Usage(response.input_tokens, response.output_tokens),
            metadata={
                **dict(response.metadata),
                "measurement_complete": response.measurement_complete,
            },
        )


class BenchmarkEventSink(EventSink):
    """Charge model usage and mirror runtime transitions into the host trace."""

    def __init__(self, observer: Any) -> None:
        """Initialize the sink.

        Args:
            observer: Benchmark trace and budget observer.
        """
        self.observer = observer

    def emit(self, event_type: str, actor: str, payload: Mapping[str, Any]) -> None:
        """Normalize a runtime event and forward it to the benchmark observer.

        Args:
            event_type: Runtime event type.
            actor: Role responsible for the event.
            payload: Runtime event fields and optional usage counters.
        """
        copied = dict(payload)
        input_tokens = int(copied.pop("input_tokens", 0))
        output_tokens = int(copied.pop("output_tokens", 0))
        if event_type == "model_turn":
            normalized_type = "model_response"
        elif event_type == "permission_violation":
            normalized_type = "tool_called"
        else:
            normalized_type = f"software_multiagent.{event_type}"
        self.observer.record(
            normalized_type,
            actor,
            copied,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=1 if event_type == "permission_violation" else 0,
        )


@dataclass(frozen=True)
class BenchmarkAgentRun:
    """Represent runtime result fields required by the host orchestrator.

    Attributes:
        completed: Whether the workflow reached a complete status.
        stop_reason: Stable runtime termination reason.
        node_executions: Total workflow-node visits.
        measurement_complete: Whether usage measurement is complete.
        model_turns: Total model turns across visited nodes.
    """

    completed: bool
    stop_reason: str
    node_executions: int
    measurement_complete: bool
    model_turns: int


def _agent_tools(tool_executor: Any, role_ids: Sequence[str]) -> ToolRegistry:
    """Build a role-aware runtime registry from benchmark tool declarations.

    Args:
        tool_executor: Harness tool executor enforcing role policies.
        role_ids: Roles whose declarations are merged into the registry.

    Returns:
        A runtime registry that dispatches each call using the calling role.

    Raises:
        ValueError: If roles expose incompatible schemas for the same tool.
    """
    declarations: dict[str, Mapping[str, Any]] = {}
    for role_id in role_ids:
        for declaration in tool_executor.declarations_for(role_id):
            name = str(declaration["name"])
            previous = declarations.get(name)
            if previous is not None and (
                previous.get("parameters") != declaration.get("parameters")
                or previous.get("description") != declaration.get("description")
            ):
                raise ValueError(f"tool schema differs between roles: {name}")
            declarations[name] = declaration
    return ToolRegistry(
        [
            AgentCallableTool(
                name=name,
                description=str(declaration.get("description", "")),
                input_schema=dict(declaration.get("parameters", {})),
                mutates_workspace=bool(
                    declaration.get("mutates_workspace", name in {"write", "run"})
                ),
                handler=_role_tool_handler(tool_executor, name),
            )
            for name, declaration in declarations.items()
        ]
    )


def run_single_agent(
    *,
    task_id: str,
    objective: str,
    target_software: str | None,
    role_id: str,
    instructions: str,
    max_turns: int,
    adapter: Any,
    tool_executor: Any,
    observer: Any,
    seed: int,
    workspace: Path | None = None,
) -> BenchmarkAgentRun:
    """Run the production solo preset under benchmark controls.

    Args:
        task_id: Stable task instance identifier.
        objective: Public task objective.
        target_software: Target application name, if available.
        role_id: Solo role identifier.
        instructions: Solo role system instructions.
        max_turns: Maximum model turns for the role.
        adapter: Benchmark-owned model adapter.
        tool_executor: Harness tool executor.
        observer: Trace and budget observer.
        seed: Reproducibility seed.
        workspace: Optional task workspace path.

    Returns:
        Normalized workflow completion and usage fields.
    """
    declarations = tool_executor.declarations_for(role_id)
    tools = ToolRegistry(
        [
            CallableTool(
                name=declaration["name"],
                description=str(declaration.get("description", "")),
                input_schema=dict(declaration.get("parameters", {})),
                mutates_workspace=bool(
                    declaration.get(
                        "mutates_workspace",
                        declaration["name"] in {"write", "run"},
                    )
                ),
                handler=_tool_handler(tool_executor, role_id, declaration["name"]),
            )
            for declaration in declarations
        ]
    )
    agent = AgentSpec(
        id=role_id,
        instructions=instructions,
        tools=tuple(declaration["name"] for declaration in declarations),
        max_turns=max_turns,
        terminal_tools=("finish_phase",)
        if any(item["name"] == "finish_phase" for item in declarations)
        else (),
    )
    task = TaskContext(
        task_id=task_id,
        objective=objective,
        workspace=workspace or Path("."),
        target_software=target_software,
    )
    result = AgentRuntime(
        BenchmarkModelClient(adapter, seed=seed),
        tools,
        events=BenchmarkEventSink(observer),
    ).run(task, single_agent(agent))
    return BenchmarkAgentRun(
        completed=result.status == StepStatus.COMPLETE,
        stop_reason=result.stop_reason,
        node_executions=sum(result.node_visits.values()),
        measurement_complete=result.measurement_complete,
        model_turns=result.model_turns,
    )


def run_multi_agent(
    *,
    task_id: str,
    objective: str,
    target_software: str | None,
    agents: Sequence[AgentSpec],
    max_repairs: int,
    adapter: Any,
    tool_executor: Any,
    observer: Any,
    seed: int,
    workspace: Path | None = None,
) -> BenchmarkAgentRun:
    """Run the plan-execute-verify graph under benchmark controls.

    Args:
        task_id: Stable task instance identifier.
        objective: Public task objective.
        target_software: Target application name, if available.
        agents: Planner, executor, and verifier runtime specifications.
        max_repairs: Maximum verifier-triggered executor revisits.
        adapter: Benchmark-owned model adapter.
        tool_executor: Harness tool executor and message bus owner.
        observer: Trace and budget observer.
        seed: Reproducibility seed.
        workspace: Optional task workspace path.

    Returns:
        Normalized workflow completion and usage fields.

    Raises:
        ValueError: If the workflow does not receive exactly three agents.
    """
    if len(agents) != 3:
        raise ValueError("base multi-agent workflow requires planner, executor, verifier")
    planner, executor, verifier = agents
    if "message" in planner.tools and not planner.terminal_tools:
        planner = replace(planner, terminal_tools=("message",))
    result = AgentRuntime(
        BenchmarkModelClient(
            adapter,
            seed=seed,
            inbox=tool_executor.messages.for_role,
            observer=observer,
        ),
        _agent_tools(tool_executor, [agent.id for agent in agents]),
        events=BenchmarkEventSink(observer),
    ).run(
        TaskContext(
            task_id=task_id,
            objective=objective,
            workspace=workspace or Path("."),
            target_software=target_software,
        ),
        plan_execute_verify(
            planner,
            executor,
            verifier,
            max_repairs=max_repairs,
        ),
    )
    return BenchmarkAgentRun(
        completed=result.status == StepStatus.COMPLETE,
        stop_reason=result.stop_reason,
        node_executions=sum(result.node_visits.values()),
        measurement_complete=result.measurement_complete,
        model_turns=result.model_turns,
    )


def _tool_handler(tool_executor: Any, role_id: str, name: str):
    """Create a single-role runtime handler for one benchmark tool.

    Args:
        tool_executor: Harness tool executor.
        role_id: Role to charge and authorize.
        name: Tool name to invoke.

    Returns:
        A runtime-compatible tool handler.
    """

    def execute(arguments: Mapping[str, Any], _task: TaskContext) -> Mapping[str, Any]:
        """Invoke the bound benchmark tool and serialize its result."""
        return tool_executor.execute(role_id, name, arguments).to_dict()

    return execute


def _role_tool_handler(tool_executor: Any, name: str):
    """Create a role-aware runtime handler for one benchmark tool.

    Args:
        tool_executor: Harness tool executor.
        name: Tool name to invoke.

    Returns:
        A runtime-compatible handler that authorizes the calling agent.
    """

    def execute(
        agent: AgentSpec,
        arguments: Mapping[str, Any],
        _task: TaskContext,
    ) -> Mapping[str, Any]:
        """Invoke the bound tool as the runtime-provided agent."""
        return tool_executor.execute(agent.id, name, arguments).to_dict()

    return execute
