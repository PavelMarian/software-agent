from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from software_bench.core.models import AgentTopology
from software_bench.harness.agents.loop import AgentLoop, system_prompt, task_prompt
from software_bench.harness.contracts import ModelAdapter, RunRequest
from software_bench.harness.agents.tools import EventRecorder, MessageBus, ToolExecutor


@dataclass(frozen=True)
class OrchestratorOutcome:
    status: str
    stop_reason: str
    measurement_complete: bool
    phase_count: int
    agent_run_count: int
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    partial_submission: bool = False


class NativeOrchestrationStrategy(Protocol):
    def run(
        self, request: RunRequest, adapter: ModelAdapter, observer: EventRecorder
    ) -> OrchestratorOutcome: ...


class Orchestrator:
    """Dispatches a mode to its declared native agent topology."""

    def run(
        self, request: RunRequest, adapter: ModelAdapter, observer: EventRecorder
    ) -> OrchestratorOutcome:
        if not request.mode.supported:
            raise ValueError(f"mode {request.mode.id} is not executable yet")
        if request.mode.agent_runtime != "legacy":
            from software_bench.harness.models.registry import load_agent_runtime

            runtime = load_agent_runtime(request.mode.agent_runtime)
            outcome = runtime.run(request, adapter, observer)
            if not isinstance(outcome, OrchestratorOutcome):
                raise TypeError(
                    "agent runtime must return an OrchestratorOutcome"
                )
            return outcome
        strategies: dict[AgentTopology, NativeOrchestrationStrategy] = {
            AgentTopology.SINGLE_AGENT: SingleAgentOrchestrator(),
            AgentTopology.MULTI_AGENT: MultiAgentOrchestrator(),
        }
        strategy = strategies.get(request.mode.agent_topology)
        if strategy is None:
            raise ValueError(f"topology {request.mode.agent_topology} is not executable yet")
        return strategy.run(request, adapter, observer)


class SingleAgentOrchestrator:
    """Runs the topology-neutral agent-system runtime as the solo baseline."""

    def run(
        self, request: RunRequest, adapter: ModelAdapter, observer: EventRecorder
    ) -> OrchestratorOutcome:
        role = request.mode.roles[0]
        _messages, tools = _runtime(request, observer)
        observer.record("phase_started", role.id, {"phase": 0, "profile": role.profile})
        observer.record("agent_started", role.id, {"phase": 0, "profile": role.profile})
        try:
            from software_bench.harness.agents.software_multiagent import run_single_agent
        except ImportError as error:
            raise RuntimeError(
                "mas-professional-agents is required; install the benchmark package "
                "with the parent software_multiagent runtime"
            ) from error
        result = run_single_agent(
            task_id=request.task.instance_id,
            objective=task_prompt(request, role.id),
            target_software=request.task.target_software,
            role_id=role.id,
            instructions=system_prompt(request.mode.agent_topology, role),
            max_turns=request.mode.max_turns_per_phase,
            adapter=adapter,
            tool_executor=tools,
            observer=observer,
            seed=request.seed,
        )
        observer.record(
            "agent_finished",
            role.id,
            {
                "phase": 0,
                "turn_count": result.model_turns,
                "stop_reason": result.stop_reason,
            },
        )
        observer.record("phase_finished", role.id, {"phase": 0})
        return OrchestratorOutcome(
            status="completed" if result.completed else "agent_failed",
            stop_reason=result.stop_reason,
            measurement_complete=result.measurement_complete,
            phase_count=1,
            agent_run_count=1,
        )


class MultiAgentOrchestrator:
    """Runs the base AutoDS-like graph, retaining fixed phases for ablations."""

    def run(
        self, request: RunRequest, adapter: ModelAdapter, observer: EventRecorder
    ) -> OrchestratorOutcome:
        roles = {role.id: role for role in request.mode.roles}
        messages, tools = _runtime(request, observer)
        if set(roles) == {"planner", "executor", "verifier"}:
            try:
                from software_multiagent import AgentSpec
                from software_bench.harness.agents.software_multiagent import run_multi_agent
            except ImportError as error:
                raise RuntimeError(
                    "mas-professional-agents is required; install the benchmark package "
                    "with the parent software_multiagent runtime"
                ) from error
            agents = tuple(
                AgentSpec(
                    id=role.id,
                    instructions=system_prompt(request.mode.agent_topology, role),
                    tools=tuple(
                        declaration["name"]
                        for declaration in tools.declarations_for(role.id)
                    ),
                    max_turns=request.mode.max_turns_per_phase,
                    terminal_tools=("finish_phase",)
                    if "finish_phase" in role.tools
                    else (),
                )
                for role in request.mode.roles
            )
            max_repairs = max(0, request.mode.phases.count("executor") - 1)
            result = run_multi_agent(
                task_id=request.task.instance_id,
                objective=task_prompt(request, "team"),
                target_software=request.task.target_software,
                agents=agents,
                max_repairs=max_repairs,
                adapter=adapter,
                tool_executor=tools,
                observer=observer,
                seed=request.seed,
            )
            return OrchestratorOutcome(
                status="completed" if result.completed else "agent_failed",
                stop_reason=result.stop_reason,
                measurement_complete=result.measurement_complete,
                phase_count=result.node_executions,
                agent_run_count=result.node_executions,
            )

        # Planner/verifier ablations keep their declarative fixed-phase protocol.
        measurement_complete = True
        for phase_index, role_id in enumerate(request.mode.phases):
            role = roles[role_id]
            observer.record(
                "phase_started", role_id, {"phase": phase_index, "profile": role.profile}
            )
            result = AgentLoop().run(
                request,
                role,
                phase_index,
                adapter,
                tools,
                observer,
                messages.for_role(role_id),
            )
            measurement_complete = measurement_complete and result.measurement_complete
            observer.record("phase_finished", role_id, {"phase": phase_index})
        return OrchestratorOutcome(
            status="completed",
            stop_reason="multi_agent_protocol_completed",
            measurement_complete=measurement_complete,
            phase_count=len(request.mode.phases),
            agent_run_count=len(request.mode.phases),
        )


def _runtime(request: RunRequest, observer: EventRecorder) -> tuple[MessageBus, ToolExecutor]:
    roles = {role.id: role for role in request.mode.roles}
    messages = MessageBus(set(roles))
    return messages, ToolExecutor(
        request.environment,
        roles,
        request.mode.communication_enabled,
        observer,
        messages,
    )
