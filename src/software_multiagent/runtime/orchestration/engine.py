"""Graph runtime shared by solo and multi-agent configurations."""

from __future__ import annotations

from copy import deepcopy

from software_multiagent.core.contracts import (
    RunResult,
    RunSpec,
    RoutingMode,
    RuntimeState,
    StepStatus,
    TaskContext,
)
from software_multiagent.ports.protocols import (
    CheckpointStore,
    EventSink,
    InMemoryCheckpointStore,
    KnowledgeProvider,
    ModelClient,
    NullEventSink,
)
from software_multiagent.runtime.orchestration.agent_loop import AgentLoop
from software_multiagent.runtime.orchestration.policies import (
    CoordinationPolicy,
    MemoryGatedResearchPolicy,
    StaticGraphPolicy,
)
from software_multiagent.tools.registry import ToolRegistry


class AgentRuntime:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        *,
        events: EventSink | None = None,
        checkpoints: CheckpointStore | None = None,
        coordination_policy: CoordinationPolicy | None = None,
        knowledge: KnowledgeProvider | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.events = events or NullEventSink()
        self.checkpoints = checkpoints or InMemoryCheckpointStore()
        self.coordination_policy = coordination_policy
        self.knowledge = knowledge

    def run(self, task: TaskContext, spec: RunSpec) -> RunResult:
        begin_run = getattr(self.knowledge, "begin_run", None)
        if callable(begin_run):
            begin_run(task)
        try:
            return self._run(task, spec)
        finally:
            finish_run = getattr(self.knowledge, "finish_run", None)
            if callable(finish_run):
                finish_run(task)

    def _run(self, task: TaskContext, spec: RunSpec) -> RunResult:
        agents = {agent.id: agent for agent in spec.agents}
        nodes = {node.id: node for node in spec.nodes}
        state = RuntimeState(task=task, current_node=spec.entry_node)
        policy = self.coordination_policy or self._default_policy(spec)
        state.current_node = policy.entry_node(spec, state)
        loop = AgentLoop(self.model, self.tools, self.events, knowledge=self.knowledge)

        while state.node_executions < spec.max_node_executions:
            node = nodes[state.current_node]
            visits = state.node_visits.get(node.id, 0) + 1
            state.node_visits[node.id] = visits
            if visits > node.max_visits:
                return self._result(
                    state,
                    StepStatus.FAILED,
                    f"node_visit_limit:{node.id}",
                )

            checkpoint_id = f"{task.task_id}:{state.node_executions}:{node.id}"
            self.checkpoints.save(checkpoint_id, deepcopy(state))
            self.events.emit(
                "node_started",
                node.agent_id,
                {
                    "node": node.id,
                    "visit": visits,
                    "checkpoint": checkpoint_id,
                    "max_tool_calls": node.max_tool_calls,
                    "max_tokens": node.max_tokens,
                },
            )
            remaining_calls = (
                None
                if node.max_tool_calls is None
                else node.max_tool_calls - state.node_tool_calls.get(node.id, 0)
            )
            remaining_tokens = (
                None
                if node.max_tokens is None
                else node.max_tokens - state.node_tokens.get(node.id, 0)
            )
            outcome = loop.run(
                agents[node.agent_id],
                state,
                max_tool_calls=remaining_calls,
                max_tokens=remaining_tokens,
            )
            state.node_tool_calls[node.id] = (
                state.node_tool_calls.get(node.id, 0) + outcome.tool_calls_used
            )
            state.node_tokens[node.id] = (
                state.node_tokens.get(node.id, 0) + outcome.tokens_used
            )
            state.node_executions += 1
            route = outcome.turn.route or outcome.turn.status.value
            try:
                next_node = policy.next_node(spec, state, node, route)
            except ValueError as error:
                return self._result(state, StepStatus.FAILED, str(error))
            self.events.emit(
                "node_finished",
                node.agent_id,
                {"node": node.id, "route": route, "next_node": next_node},
            )
            if next_node is None:
                return self._result(state, outcome.turn.status, f"terminal_route:{route}")
            state.current_node = next_node

        return self._result(state, StepStatus.FAILED, "workflow_execution_limit")

    def _default_policy(self, spec: RunSpec) -> CoordinationPolicy:
        if spec.routing_mode == RoutingMode.MEMORY_GATED_RESEARCH:
            return MemoryGatedResearchPolicy(self.knowledge)
        return StaticGraphPolicy()

    @staticmethod
    def _result(state: RuntimeState, status: StepStatus, reason: str) -> RunResult:
        return RunResult(
            status=status,
            stop_reason=reason,
            messages=tuple(state.messages),
            artifacts=dict(state.artifacts),
            evidence=tuple(state.evidence),
            work_items=dict(state.work_items),
            node_visits=dict(state.node_visits),
            usage=state.usage,
            measurement_complete=state.measurement_complete,
            model_turns=state.model_turns,
        )
