"""One reusable tool-calling loop; orchestration is deliberately kept outside it."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from software_multiagent.core.contracts import (
    AgentSpec,
    Message,
    ModelTurn,
    RuntimeState,
    StepStatus,
)
from software_multiagent.ports.protocols import EventSink, ModelClient
from software_multiagent.ports.protocols import (
    ActionValidator,
    KnowledgeProvider,
    RecoveryPolicy,
    RollbackManager,
)
from software_multiagent.runtime.reliability.mechanisms import (
    BoundedRecoveryPolicy,
    ExitCodeValidator,
    NullKnowledgeProvider,
    NullRollbackManager,
)
from software_multiagent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class AgentOutcome:
    turn: ModelTurn
    turns_used: int
    tool_calls_used: int = 0
    tokens_used: int = 0


class AgentLoop:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        events: EventSink,
        *,
        knowledge: KnowledgeProvider | None = None,
        action_validator: ActionValidator | None = None,
        recovery_policy: RecoveryPolicy | None = None,
        rollback_manager: RollbackManager | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.events = events
        self.knowledge = knowledge or NullKnowledgeProvider()
        self.action_validator = action_validator or ExitCodeValidator()
        self.recovery_policy = recovery_policy or BoundedRecoveryPolicy()
        self.rollback_manager = rollback_manager or NullRollbackManager()

    def run(
        self,
        agent: AgentSpec,
        state: RuntimeState,
        *,
        max_tool_calls: int | None = None,
        max_tokens: int | None = None,
    ) -> AgentOutcome:
        previous_calls: tuple[str, ...] = ()
        repeated_call_sets = 0
        tool_calls_used = 0
        tokens_used = 0
        if max_tool_calls is not None and max_tool_calls <= 0:
            return self._limit_outcome(agent, "tool_calls", 0, 0, 0)
        if max_tokens is not None and max_tokens <= 0:
            return self._limit_outcome(agent, "tokens", 0, 0, 0)
        context = self.knowledge.context_for(state.task, agent, state).render()
        if context:
            state.messages.append(
                Message(
                    sender="knowledge",
                    recipient=agent.id,
                    kind="context",
                    content="Relevant task knowledge and dependencies:\n" + context,
                )
            )
            self.events.emit(
                "knowledge_context",
                agent.id,
                {"characters": len(context), "node": state.current_node},
            )
        for turn_index in range(agent.max_turns):
            visible = tuple(
                message
                for message in state.messages
                if message.recipient in {None, agent.id}
            )
            turn = self.model.generate(
                agent,
                state.task,
                visible,
                self.tools.schemas_for(agent),
            )
            state.model_turns += 1
            state.usage = state.usage + turn.usage
            tokens_used += turn.usage.input_tokens + turn.usage.output_tokens
            state.measurement_complete = state.measurement_complete and bool(
                turn.metadata.get("measurement_complete", True)
            )
            if turn.content:
                state.messages.append(Message(agent.id, turn.content))
            self.events.emit(
                "model_turn",
                agent.id,
                {
                    "turn": turn_index,
                    "status": turn.status.value,
                    "route": turn.route,
                    "tool_calls": len(turn.tool_calls),
                    "input_tokens": turn.usage.input_tokens,
                    "output_tokens": turn.usage.output_tokens,
                    "content": turn.content,
                    "metadata": dict(turn.metadata),
                },
            )
            recoverable_errors = turn.metadata.get("recoverable_tool_errors", ())
            if isinstance(recoverable_errors, (list, tuple)):
                for error in recoverable_errors:
                    if not isinstance(error, str):
                        continue
                    state.messages.append(
                        Message(
                            sender="runtime",
                            recipient=agent.id,
                            kind="control",
                            content=(
                                f"Recoverable model-tool protocol error: {error}. "
                                "Retry the action with a smaller valid JSON payload."
                            ),
                        )
                    )
                    self.events.emit(
                        "model_tool_error",
                        agent.id,
                        {"turn": turn_index, "error": error},
                    )
            call_fingerprints = tuple(
                f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
                for call in turn.tool_calls
            )
            if call_fingerprints and call_fingerprints == previous_calls:
                repeated_call_sets += 1
            else:
                repeated_call_sets = 0
            previous_calls = call_fingerprints
            terminal_tool_succeeded = False
            validation_failed = False
            for call in turn.tool_calls:
                if max_tool_calls is not None and tool_calls_used >= max_tool_calls:
                    return self._limit_outcome(
                        agent,
                        "tool_calls",
                        tool_calls_used,
                        tokens_used,
                        turn_index + 1,
                    )
                if call.name not in agent.tools:
                    self.events.emit(
                        "permission_violation",
                        agent.id,
                        {"tool": call.name, "call_id": call.id},
                    )
                    raise PermissionError(f"tool not allowed for {agent.id}: {call.name}")
                action = replace(
                    self.tools.action_for(call),
                    id=f"{agent.id}:{state.model_turns}:{call.id}",
                )
                state.action_graph.add(action)
                self.events.emit(
                    "action_prepared",
                    agent.id,
                    {
                        "action": action.id,
                        "tool": action.tool,
                        "interface": action.interface.value if action.interface else None,
                        "mutates_workspace": action.mutates_workspace,
                        "reversible": action.reversible,
                        "expected_effect": action.expected_effect,
                    },
                )
                rollback_token = (
                    self.rollback_manager.checkpoint(action, state.task)
                    if action.mutates_workspace
                    else None
                )
                result = self.tools.execute(agent, call, state.task)
                tool_calls_used += 1
                verification = None
                recovery = None
                if call.name not in agent.terminal_tools:
                    verification = self.action_validator.verify(action, result, state.task)
                if verification is not None:
                    state.verifications.append(verification)
                    state.evidence.extend(verification.evidence)
                    self.events.emit(
                        "action_verified",
                        agent.id,
                        {
                            "action": action.id,
                            "check": verification.check_id,
                            "status": verification.status.value,
                            "repair_targets": list(verification.repair_targets),
                            "summary": verification.summary,
                        },
                    )
                    if verification.status.value == "failed":
                        validation_failed = True
                        fingerprint = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
                        attempt = state.repair_attempts.get(fingerprint, 0) + 1
                        state.repair_attempts[fingerprint] = attempt
                        recovery = self.recovery_policy.decide(action, verification, attempt)
                        if recovery.rollback:
                            rollback_evidence = self.rollback_manager.rollback(
                                rollback_token, action, state.task
                            )
                            if rollback_evidence is not None:
                                state.evidence.append(rollback_evidence)
                                self.events.emit(
                                    "action_rolled_back",
                                    agent.id,
                                    {"action": action.id, "attempt": attempt},
                                )
                        state.action_graph.mark_failed(action.id)
                    elif result.ok:
                        state.action_graph.mark_completed(action.id)
                elif result.ok:
                    state.action_graph.mark_completed(action.id)
                else:
                    state.action_graph.mark_failed(action.id)
                terminal_tool_succeeded = terminal_tool_succeeded or (
                    call.name in agent.terminal_tools and result.ok and not validation_failed
                )
                state.messages.append(
                    Message(
                        sender=f"tool:{call.name}",
                        recipient=agent.id,
                        kind="tool_result",
                        content=json.dumps(
                            {
                                "call_id": result.call_id,
                                "ok": result.ok,
                                "output": result.output,
                                "error": result.error,
                                "verification": (
                                    {
                                        "check_id": verification.check_id,
                                        "status": verification.status.value,
                                        "summary": verification.summary,
                                        "repair_targets": list(verification.repair_targets),
                                    }
                                    if verification is not None
                                    else None
                                ),
                                "recovery": (
                                    {
                                        "retry": recovery.retry,
                                        "rollback": recovery.rollback,
                                        "reason": recovery.reason,
                                    }
                                    if recovery is not None
                                    else None
                                ),
                            },
                            default=str,
                        ),
                    )
                )
                self.events.emit(
                    "tool_result",
                    agent.id,
                    {"tool": call.name, "call_id": call.id, "ok": result.ok},
                )
                observe_action = getattr(self.knowledge, "observe_action", None)
                if callable(observe_action):
                    try:
                        observe_action(
                            state.task,
                            agent,
                            state,
                            action,
                            result,
                            verification,
                        )
                    except Exception as error:
                        # Memory is an evidence subsystem, not an authority to hide or
                        # alter the result of the software action that already happened.
                        self.events.emit(
                            "knowledge_write_error",
                            agent.id,
                            {
                                "action": action.id,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                        )
                if recovery is not None and not recovery.retry:
                    failed = ModelTurn(
                        content=(
                            f"Independent validation failed and the repair limit was reached: "
                            f"{verification.summary if verification else action.id}"
                        ),
                        status=StepStatus.FAILED,
                        route="failed",
                    )
                    state.messages.append(Message("runtime", failed.content, agent.id, "control"))
                    return AgentOutcome(
                        failed,
                        turn_index + 1,
                        tool_calls_used,
                        tokens_used,
                    )
            if terminal_tool_succeeded:
                terminal_route = next(
                    (
                        str(call.arguments.get("status", "complete"))
                        for call in turn.tool_calls
                        if call.name in agent.terminal_tools
                    ),
                    "complete",
                )
                try:
                    terminal_status = StepStatus(terminal_route)
                except ValueError:
                    terminal_status = StepStatus.FAILED
                    terminal_route = "failed"
                return AgentOutcome(
                    replace(turn, status=terminal_status, route=terminal_route),
                    turn_index + 1,
                    tool_calls_used,
                    tokens_used,
                )
            if max_tool_calls is not None and tool_calls_used >= max_tool_calls:
                return self._limit_outcome(
                    agent, "tool_calls", tool_calls_used, tokens_used, turn_index + 1
                )
            if max_tokens is not None and tokens_used >= max_tokens:
                return self._limit_outcome(
                    agent, "tokens", tool_calls_used, tokens_used, turn_index + 1
                )
            if repeated_call_sets == 2:
                state.messages.append(
                    Message(
                        sender="runtime",
                        recipient=agent.id,
                        kind="control",
                        content=(
                            "The same tool calls have been repeated without new evidence. "
                            "Choose a different action or finish the assigned phase with its "
                            "required terminal marker."
                        ),
                    )
                )
                self.events.emit(
                    "stall_warning",
                    agent.id,
                    {"turn": turn_index, "repeated_calls": call_fingerprints},
                )
            if turn.status != StepStatus.CONTINUE and not turn.tool_calls:
                return AgentOutcome(turn, turn_index + 1, tool_calls_used, tokens_used)
        exhausted = ModelTurn(
            content=f"Agent {agent.id} exhausted its turn limit.",
            status=StepStatus.FAILED,
            route="failed",
        )
        state.messages.append(Message(agent.id, exhausted.content))
        return AgentOutcome(exhausted, agent.max_turns, tool_calls_used, tokens_used)

    def _limit_outcome(
        self,
        agent: AgentSpec,
        limit: str,
        tool_calls_used: int,
        tokens_used: int,
        turns_used: int,
    ) -> AgentOutcome:
        self.events.emit(
            "node_limit_reached",
            agent.id,
            {
                "limit": limit,
                "tool_calls_used": tool_calls_used,
                "tokens_used": tokens_used,
            },
        )
        turn = ModelTurn(
            content=f"Agent node reached its {limit} limit.",
            status=StepStatus.CONTINUE,
            route="action_limit",
        )
        return AgentOutcome(turn, turns_used, tool_calls_used, tokens_used)
