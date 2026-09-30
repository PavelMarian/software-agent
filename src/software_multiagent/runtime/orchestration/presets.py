"""Small declarative presets; applications can define arbitrary workflow graphs."""

from __future__ import annotations

from software_multiagent.core.contracts import (
    AgentSpec,
    RoutingMode,
    RunSpec,
    Topology,
    WorkflowNode,
)


def single_agent(agent: AgentSpec, *, max_executions: int = 1) -> RunSpec:
    return RunSpec(
        topology=Topology.SINGLE_AGENT,
        agents=(agent,),
        nodes=(
            WorkflowNode(
                id="solve",
                agent_id=agent.id,
                transitions={"complete": None, "failed": None},
                max_visits=max_executions,
            ),
        ),
        entry_node="solve",
        max_node_executions=max_executions,
    )


def plan_execute_verify(
    planner: AgentSpec,
    executor: AgentSpec,
    verifier: AgentSpec,
    *,
    max_repairs: int = 2,
    planner_max_tool_calls: int = 12,
    planner_max_tokens: int = 15_000,
    executor_max_tool_calls: int = 60,
    executor_max_tokens: int = 65_000,
    verifier_max_tool_calls: int = 20,
    verifier_max_tokens: int = 20_000,
) -> RunSpec:
    """AutoDS-like roles with an explicit verifier-to-executor repair edge."""
    return RunSpec(
        topology=Topology.MULTI_AGENT,
        agents=(planner, executor, verifier),
        nodes=(
            WorkflowNode(
                "plan",
                planner.id,
                {"complete": "execute", "action_limit": None, "failed": None},
                max_tool_calls=planner_max_tool_calls,
                max_tokens=planner_max_tokens,
            ),
            WorkflowNode(
                "execute",
                executor.id,
                {"complete": "verify", "action_limit": "verify", "failed": None},
                max_visits=max_repairs + 1,
                max_tool_calls=executor_max_tool_calls,
                max_tokens=executor_max_tokens,
            ),
            WorkflowNode(
                "verify",
                verifier.id,
                {
                    "complete": None,
                    "needs_revision": "execute",
                    "action_limit": "execute",
                    "failed": None,
                },
                max_visits=max_repairs + 1,
                max_tool_calls=verifier_max_tool_calls,
                max_tokens=verifier_max_tokens,
            ),
        ),
        entry_node="plan",
        max_node_executions=3 + 2 * max_repairs,
    )


def plan_research_execute_verify(
    planner: AgentSpec,
    researcher: AgentSpec,
    executor: AgentSpec,
    evaluator: AgentSpec,
    *,
    max_repairs: int = 2,
    planner_max_tool_calls: int = 12,
    planner_max_tokens: int = 15_000,
    researcher_max_tool_calls: int = 12,
    researcher_max_tokens: int = 15_000,
    executor_max_tool_calls: int = 60,
    executor_max_tokens: int = 65_000,
    evaluator_max_tool_calls: int = 20,
    evaluator_max_tokens: int = 20_000,
) -> RunSpec:
    """Plan, resolve gaps from memory, research remaining gaps, execute, and verify.

    With shared software memory, the runtime independently exhausts relevant stored
    knowledge before routing to the researcher.  The researcher receives only gaps
    that the memory gate could not resolve.  Without a compatible memory provider,
    the conservative fallback retains the original single research visit.
    """

    return RunSpec(
        topology=Topology.MULTI_AGENT,
        agents=(planner, researcher, executor, evaluator),
        nodes=(
            WorkflowNode(
                "initial_plan",
                planner.id,
                {"complete": "research", "action_limit": "research", "failed": None},
                max_tool_calls=planner_max_tool_calls,
                max_tokens=planner_max_tokens,
            ),
            WorkflowNode(
                "research",
                researcher.id,
                {"complete": "final_plan", "action_limit": "final_plan", "failed": None},
                max_visits=1,
                max_tool_calls=researcher_max_tool_calls,
                max_tokens=researcher_max_tokens,
            ),
            WorkflowNode(
                "final_plan",
                planner.id,
                {"complete": "execute", "action_limit": "execute", "failed": None},
                max_tool_calls=planner_max_tool_calls,
                max_tokens=planner_max_tokens,
            ),
            WorkflowNode(
                "execute",
                executor.id,
                {"complete": "evaluate", "action_limit": "evaluate", "failed": None},
                max_visits=max_repairs + 1,
                max_tool_calls=executor_max_tool_calls,
                max_tokens=executor_max_tokens,
            ),
            WorkflowNode(
                "evaluate",
                evaluator.id,
                {
                    "complete": None,
                    "needs_revision": "execute",
                    "action_limit": "execute",
                    "failed": None,
                },
                max_visits=max_repairs + 1,
                max_tool_calls=evaluator_max_tool_calls,
                max_tokens=evaluator_max_tokens,
            ),
        ),
        entry_node="initial_plan",
        max_node_executions=5 + 2 * max_repairs,
        routing_mode=RoutingMode.MEMORY_GATED_RESEARCH,
    )


def solo_plan_execute_verify(
    agent: AgentSpec,
    *,
    max_repairs: int = 2,
) -> RunSpec:
    """Strong solo control: one stateful session executes the same staged graph."""
    return RunSpec(
        topology=Topology.SINGLE_AGENT,
        agents=(agent,),
        nodes=(
            WorkflowNode("plan", agent.id, {"complete": "execute", "failed": None}),
            WorkflowNode(
                "execute",
                agent.id,
                {"complete": "verify", "failed": None},
                max_visits=max_repairs + 1,
            ),
            WorkflowNode(
                "verify",
                agent.id,
                {
                    "complete": None,
                    "needs_revision": "execute",
                    "failed": None,
                },
                max_visits=max_repairs + 1,
            ),
        ),
        entry_node="plan",
        max_node_executions=3 + 2 * max_repairs,
    )
