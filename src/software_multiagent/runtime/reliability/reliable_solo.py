"""A bounded single-session agent with extensible checks and verified recovery."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from software_multiagent.core.contracts import (
    AgentSpec, Message, RunResult, RuntimeState, StepStatus, TaskContext,
)
from software_multiagent.core.execution import Evidence, VerificationResult, VerificationStatus
from software_multiagent.ports.protocols import EventSink, ModelClient, NullEventSink, RollbackManager
from software_multiagent.runtime.reliability.patterns import (
    ExecutionFeedback, HookContext, HookResult, MechanismRegistry, Phase, PreExecutionGate,
)
from software_multiagent.tools.registry import ToolRegistry
from software_multiagent.runtime.reliability.action_validation import _validate_arguments
from software_multiagent.runtime.reasoning.react import ReActProtocol, ReActProtocolError
from software_multiagent.tools.gateway import ToolGateway
from software_multiagent.runtime.reasoning.context import BoundedContext
from software_multiagent.runtime.reliability.journal import ExecutionJournal
from software_multiagent.runtime.reliability.repair import FailureKind, RepairPolicy
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class SoloLimits:
    max_tool_calls: int = 60
    max_tokens: int = 100_000
    max_seconds: float = 600
    max_repairs: int = 3
    max_recovery_attempts: int = 3
    max_repeated_failures: int = 2
    max_stalled_actions: int = 3
    max_context_characters: int = 24_000

    def __post_init__(self) -> None:
        for name in ("max_tool_calls", "max_tokens", "max_seconds",
                     "max_recovery_attempts", "max_repeated_failures",
                     "max_stalled_actions", "max_context_characters"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")


class ReliableSingleAgent:
    """Explicit ReAct with one AgentSpec and one model session.

    Terminal tools are control proposals: their handlers are never executed.
    Every completion path runs BEFORE_FINISH checks. Budget enforcement is
    cooperative at call boundaries; backends must enforce blocking-call timeouts.
    """

    def __init__(
        self, model: ModelClient, tools: ToolRegistry | ToolGateway, mechanisms: MechanismRegistry,
        *, limits: SoloLimits | None = None, rollback: RollbackManager | None = None,
        events: EventSink | None = None, react: ReActProtocol | None = None,
        repair_policy: RepairPolicy | None = None, context_manager: BoundedContext | None = None,
        journal_directory: Path | None = None,
    ) -> None:
        self.model, self.tools, self.mechanisms = model, tools, mechanisms
        self.limits = limits or SoloLimits()
        self.rollback = rollback if rollback is not None else VerifiedFileRollback()
        self.events = events or NullEventSink()
        self.react = react or ReActProtocol()
        self.gateway = tools if isinstance(tools, ToolGateway) else ToolGateway(tools)
        self.pre_execution_gate = PreExecutionGate(self.gateway)
        self.repair_policy = repair_policy or RepairPolicy()
        self.context_manager = context_manager or BoundedContext()
        self.journal_directory = journal_directory

    def run(self, task: TaskContext, agent: AgentSpec) -> RunResult:
        agent = self.react.prepare(agent)
        state = RuntimeState(task, "observe")
        started = time.monotonic()
        calls = repairs = recovery_attempts = 0
        awaiting_structured_action = False
        failures: dict[str, int] = {}
        last_observation = None
        repeated_observations = 0
        snapshots: list[tuple[Any, Any]] = []
        observation_index = 0
        thought_only_turns = 0
        pending = None
        journal = ExecutionJournal(self.journal_directory / (uuid4().hex + ".jsonl")
                                   if self.journal_directory else None)
        schemas = {item["name"]: item for item in self.gateway.schemas_for(agent)}

        def emit(kind: str, payload: dict[str, Any]) -> None:
            journal.emit(kind, agent.id, payload)
            self.events.emit(kind, agent.id, payload)

        def feedback(content: Any, *, source: str = "runtime", action_id: str | None = None) -> None:
            nonlocal observation_index, awaiting_structured_action
            observation_index += 1
            awaiting_structured_action = True
            text = json.dumps(content, default=str, ensure_ascii=False)
            state.messages.append(Message(
                source, f"Observation {observation_index}: {text}", agent.id, "observation",
                metadata={"observation_id": observation_index, "action_id": action_id},
            ))
            state.current_node = "observe"
            emit("react_observation", {"observation_id": observation_index,
                                      "action_id": action_id, "source": source, "data": content})

        def checks(phase: Phase, action=None, result=None):
            state.current_node = phase.value
            output = self.mechanisms.apply(phase, HookContext(state, agent, action, result))
            for check in output.checks:
                state.verifications.append(check)
                state.evidence.extend(check.evidence)
                emit("verification", {"phase": phase.value, "check": check.check_id,
                                      "status": check.status.value, "summary": check.summary,
                                      "evidence": [asdict(e) for e in check.evidence],
                                      "repair_targets": check.repair_targets})
            return output

        def finish(reason: str, success: bool = False) -> RunResult:
            # Never restore underneath a live process. Its operation identifier
            # remains in the journal for the host to cancel/reconcile safely.
            if not success and snapshots and not pending and reason != "rollback_failed":
                if not restore(snapshots):
                    reason = "rollback_failed"
                snapshots.clear()
            emit("run_finished", {"reason": reason, "success": success,
                                  "tool_calls": calls, "repairs": repairs,
                                  "recovery_attempts": recovery_attempts,
                                  "pending_operation": pending[0].id if pending else None})
            return RunResult(
                status=StepStatus.COMPLETE if success else StepStatus.FAILED,
                stop_reason=reason, messages=tuple(state.messages),
                artifacts=dict(state.artifacts), evidence=tuple(state.evidence),
                work_items=dict(state.work_items), node_visits=dict(state.node_visits),
                usage=state.usage, measurement_complete=state.measurement_complete,
                model_turns=state.model_turns,
                verifications=tuple(state.verifications), journal=journal.entries,
                journal_path=str(journal.path) if journal.path else None,
            )

        def restore(items: list[tuple[Any, Any]]) -> bool:
            if not items:
                return True
            assert self.rollback is not None
            for action, token in reversed(items):
                try:
                    evidence = self.rollback.rollback(token, action, task)
                    # Restoration must explicitly attest its postcondition.
                    if evidence is None or evidence.data.get("restored") is not True:
                        feedback({"recovery": "restoration unverified", "action": action.id})
                        return False
                    state.evidence.append(evidence)
                    state.action_graph.completed.discard(action.id)
                    state.action_graph.failed.add(action.id)
                    emit("action_rolled_back", {"action": action.id, "evidence": evidence.summary})
                    feedback({"restored": True, "evidence": evidence.summary,
                              "details": dict(evidence.data)}, action_id=action.id)
                except PermissionError:
                    raise
                except Exception as error:
                    if getattr(error, "abort_agent_runtime", False):
                        raise
                    feedback({"recovery": "rollback failed", "error": str(error)})
                    return False
            return True

        def rejected(reason: str, fingerprint: str, kind=FailureKind.CALL) -> bool:
            nonlocal repairs
            repairs += 1
            failures[fingerprint] = failures.get(fingerprint, 0) + 1
            diagnosis = self.repair_policy.diagnose(kind, reason, attempt=repairs,
                repeated=failures[fingerprint], max_repairs=self.limits.max_repairs,
                max_repeated=self.limits.max_repeated_failures)
            state.repair_attempts[fingerprint] = failures[fingerprint]
            feedback({"diagnosis": reason, "failure_kind": diagnosis.kind.value,
                      "repair_attempt": repairs, "retry": diagnosis.retry,
                      "next_step": diagnosis.strategy})
            emit("repair_requested", {"reason": reason, "attempt": repairs,
                                      "kind": diagnosis.kind.value, "strategy": diagnosis.strategy})
            return diagnosis.retry

        def recovery_rejected(reason: str, fingerprint: str) -> bool:
            """Retry a malformed recovery proposal without consuming repair budget."""
            nonlocal recovery_attempts
            recovery_attempts += 1
            state.repair_attempts[fingerprint] = state.repair_attempts.get(fingerprint, 0) + 1
            retry = recovery_attempts < self.limits.max_recovery_attempts
            strategy = "Issue exactly one structured native tool call using the latest Observation."
            feedback({"diagnosis": reason, "failure_kind": "protocol",
                      "repair_attempt": repairs, "recovery_attempt": recovery_attempts,
                      "retry": retry, "next_step": strategy})
            emit("repair_requested", {"reason": reason, "attempt": repairs,
                                      "recovery_attempt": recovery_attempts,
                                      "kind": "protocol", "strategy": strategy})
            return retry

        def budget() -> str | None:
            if time.monotonic() - started >= self.limits.max_seconds:
                return "time_limit"
            if state.usage.input_tokens + state.usage.output_tokens >= self.limits.max_tokens:
                return "token_limit"
            if calls >= self.limits.max_tool_calls:
                return "tool_call_limit"
            return None

        emit("run_started", {"mechanisms": self.mechanisms.names, "topology": "single_agent",
                             "strategy": "react",
                             "task_id": task.task_id, "objective": task.objective,
                             "constraints": task.metadata.get("constraints", ()),
                             "limits": asdict(self.limits),
                             "max_thought_characters": self.react.max_thought_characters,
                             "max_thought_only_turns": self.react.max_thought_only_turns})
        feedback({"objective": task.objective,
                  "state": "No tool observations yet. Inspect relevant state before modifying it."})
        for _ in range(agent.max_turns):
            if reason := budget():
                return finish(reason)
            context = checks(Phase.CONTEXT)
            if any(not item.passed for item in context.checks):
                return finish("context_mechanism_failed")
            visible = self.context_manager.build(state.messages, agent.id, context.context,
                self.limits.max_context_characters, working_state=json.dumps({
                    "pending": {"tool": pending[1].poll_tool, "operation": pending[2]}
                               if pending else None,
                    "repairs_used": repairs, "calls_remaining": self.limits.max_tool_calls - calls,
                    "constraints": task.metadata.get("constraints", ())}, default=str))
            emit("context_built", {"characters": sum(len(m.content) for m in visible),
                                   "messages": len(visible), "journal_messages": len(state.messages)})
            state.current_node = "think"
            turn = self.model.generate(agent, task, tuple(visible), tuple(schemas.values()))
            state.model_turns += 1
            state.usage = state.usage + turn.usage
            state.measurement_complete &= bool(turn.metadata.get("measurement_complete", True))
            emit("model_turn", {"content": turn.content,
                                "input_tokens": turn.usage.input_tokens,
                                "output_tokens": turn.usage.output_tokens})
            if reason := budget():
                return finish(reason)
            # Validate the whole proposal before any tool or terminal handling.
            # A malformed reply is journaled, but not treated as an observation.
            try:
                decision = self.react.parse(turn)
            except ReActProtocolError as error:
                fingerprint = f"react_protocol:{error.code}"
                emit("react_protocol_error", {"error": str(error), "code": error.code,
                                               "content": turn.content})
                if awaiting_structured_action:
                    if not recovery_rejected(str(error), "recovery:" + fingerprint):
                        return finish("recovery_action_limit")
                elif not rejected(str(error), fingerprint):
                    return finish("react_protocol_limit")
                continue
            state.messages.append(Message(agent.id, f"Thought: {decision.summary}", kind="thought",
                                          metadata={"based_on_observation": observation_index}))
            emit("react_thought", {"summary": decision.summary,
                                   "based_on_observation": observation_index})
            if decision.kind == "thought":
                if awaiting_structured_action:
                    if not recovery_rejected(
                        "The response after an Observation requires a structured action.",
                        "recovery:structured_action",
                    ):
                        return finish("recovery_action_limit")
                    continue
                thought_only_turns += 1
                if thought_only_turns >= self.react.max_thought_only_turns:
                    return finish("thought_only_limit")
                continue
            thought_only_turns = 0
            # A textual finish remains available to agents that have no native
            # terminal tool. Otherwise every Observation must be followed by a
            # structured call, including during repair.
            unstructured_finish = decision.kind == "finish" and not agent.terminal_tools
            if awaiting_structured_action and decision.kind != "action" and not unstructured_finish:
                if not recovery_rejected(
                    "The response after an Observation requires a structured action.",
                    "recovery:structured_action",
                ):
                    return finish("recovery_action_limit")
                continue
            if decision.kind == "action":
                awaiting_structured_action = False
                recovery_attempts = 0
                # Protocol repetition means consecutive violations. A valid
                # structured call starts a new series without erasing action-
                # execution failure fingerprints.
                for key in tuple(failures):
                    if key.startswith("react_protocol:"):
                        del failures[key]
            if turn.status == StepStatus.FAILED:
                state.messages.append(Message(agent.id, "Action: Fail", kind="action"))
                emit("react_action", {"tool": "Fail", "arguments": {}})
                feedback({"stopped": True, "reason": "model_failed"})
                return finish("model_failed")
            completion = turn.status == StepStatus.COMPLETE and not turn.tool_calls
            if pending and completion:
                if not rejected("An operation is still pending; poll its status before completion.",
                                "pending-finish", FailureKind.STATE):
                    return finish("pending_operation")
                continue
            batch_failed = False
            for call in turn.tool_calls:
                if reason := budget():
                    return finish(reason)
                calls += 1  # Rejected proposals also consume the action budget.
                state.current_node = "act"
                state.messages.append(Message(agent.id, "Action: " + json.dumps(
                    {"call_id": call.id, "tool": call.name, "arguments": call.arguments},
                    default=str), kind="action"))
                emit("react_action", {"call_id": call.id, "tool": call.name,
                                      "arguments": dict(call.arguments)})
                if call.name not in agent.tools:
                    emit("permission_violation", {"tool": call.name})
                    raise PermissionError(f"tool not allowed for {agent.id}: {call.name}")
                fingerprint = call.name + json.dumps(call.arguments, sort_keys=True, default=str)
                if pending and (call.name != pending[1].poll_tool or
                                call.arguments.get(pending[1].poll_argument) != pending[2]):
                    if not rejected("Poll the pending operation with its exact identifier before other actions.",
                                    "pending-action", FailureKind.STATE):
                        return finish("pending_operation")
                    continue
                try:
                    _validate_arguments(call.arguments, schemas[call.name]["input_schema"])
                except (ValueError, TypeError) as error:
                    if not rejected(f"invalid action: {error}", fingerprint):
                        return finish("repair_limit")
                    batch_failed = True
                    break
                if call.name in agent.terminal_tools:
                    status = call.arguments.get("status", "complete")
                    if status == "failed":
                        feedback({"stopped": True, "reason": "model_failed"}, action_id=call.id)
                        return finish("model_failed")
                    completion = status == "complete"
                    if not completion:
                        feedback({"completion_requested": False, "status": status,
                                  "next_step": "Inspect evidence and revise the next action."},
                                 action_id=call.id)
                    # Do not execute later operations after a finish proposal.
                    break
                try:
                    action = replace(self.gateway.prepare(agent, call, task),
                                     id=f"{agent.id}:{state.model_turns}:{calls}:{call.id}")
                except (ValueError, TypeError) as error:
                    if not rejected(str(error), fingerprint, FailureKind.STATE):
                        return finish("repair_limit")
                    continue
                state.action_graph.add(action)
                emit("action_prepared", {"action": action.id, "tool": action.tool,
                                         "arguments": dict(action.arguments)})
                gate = checks(Phase.BEFORE_ACTION, action)
                preconditions = self.pre_execution_gate.check(action, task)
                for check in preconditions:
                    state.verifications.append(check)
                    state.evidence.extend(check.evidence)
                    emit("verification", {"phase": "before_action", "check": check.check_id,
                                          "status": check.status.value, "summary": check.summary,
                                          "evidence": [asdict(e) for e in check.evidence]})
                if any(not item.passed for item in gate.checks + preconditions):
                    state.action_graph.mark_failed(action.id)
                    if not rejected("; ".join(c.summary for c in gate.checks + preconditions),
                                    fingerprint, FailureKind.STATE):
                        return finish("repair_limit")
                    batch_failed = True
                    break
                token = None
                if self.rollback is not None and action.mutates_workspace:
                    if isinstance(self.rollback, VerifiedFileRollback) and not self.gateway.contract_for(
                            action.tool).file_mutations_only:
                        if not rejected("Declare complete file outputs or supply a native recovery backend.",
                                        fingerprint, FailureKind.STATE):
                            return finish("checkpoint_unavailable")
                        continue
                    try:
                        token = self.rollback.checkpoint(action, task)
                    except PermissionError:
                        raise
                    except Exception as error:
                        if getattr(error, "abort_agent_runtime", False):
                            raise
                        feedback({"checkpoint": "failed", "error": str(error)})
                        return finish("checkpoint_failed")
                    if token is None:
                        if not rejected("checkpoint unsupported for this action", fingerprint):
                            return finish("checkpoint_unavailable")
                        batch_failed = True
                        break
                    snapshots.append((action, token))
                    emit("checkpoint_created", {"action": action.id,
                                                 "outputs": action.artifact_outputs})
                emit("action_started", {"action": action.id, "tool": action.tool})
                result = self.gateway.execute(agent, call, task)
                output = checks(Phase.AFTER_ACTION, action, result)
                if "execution_feedback" not in self.mechanisms.names:
                    mandatory = ExecutionFeedback().apply(Phase.AFTER_ACTION,
                        HookContext(state, agent, action, result))
                    state.verifications.extend(mandatory.checks)
                    for check in mandatory.checks:
                        state.evidence.extend(check.evidence)
                        emit("verification", {"phase": "after_action", "check": check.check_id,
                                              "status": check.status.value, "summary": check.summary,
                                              "evidence": [asdict(e) for e in check.evidence]})
                    output = HookResult(output.checks + mandatory.checks)
                if pending and result.ok and not output.checks and not (
                        isinstance(result.output, dict) and result.output.get("status") in ("completed", "succeeded")):
                    check = VerificationResult(action.id + ":poll", VerificationStatus.INCONCLUSIVE,
                                               summary="Poll did not establish operation completion.")
                    state.verifications.append(check)
                    output = HookResult((check,))
                raw = result.output if isinstance(result.output, Mapping) else {}
                running = (raw.get("status") in ("pending", "running", "queued") or
                           (raw.get("session_id") is not None and raw.get("exit_code") is None
                            and raw.get("status") not in ("completed", "succeeded", "failed", "cancelled", "error")))
                terminal_observed = result.ok and (raw.get("status") in (
                    "completed", "succeeded", "failed", "cancelled", "error") or
                    (isinstance(raw.get("exit_code"), int) and not isinstance(raw.get("exit_code"), bool)))
                unresolved_process = running or (pending is not None and not terminal_observed)
                failed = not unresolved_process and (not result.ok or any(
                    c.status == VerificationStatus.FAILED for c in output.checks
                ))
                feedback({"call_id": call.id, "tool": call.name, "output": result.output,
                        "error": result.error, "checks": [c.summary for c in output.checks]},
                         source=f"tool:{call.name}", action_id=action.id)
                emit("tool_result", {"action": action.id, "ok": result.ok,
                                     "output": result.output, "error": result.error})
                # A poll observes the original action; its success resolves that
                # action only when execution checks establish completion.
                was_pending = pending
                if unresolved_process:
                    if pending is None:
                        contract = self.gateway.contract_for(action.tool)
                        operation = result.output.get(contract.operation_key) if isinstance(result.output, dict) else None
                        pending = (action, contract, operation)
                    emit("action_pending", {"action": pending[0].id, "operation": pending[2]})
                elif was_pending:
                    if not failed:
                        state.action_graph.mark_completed(was_pending[0].id)
                        emit("action_completed", {"action": was_pending[0].id})
                    pending = None
                observation = fingerprint + json.dumps(
                    {"output": result.output, "error": result.error}, sort_keys=True, default=str,
                )
                repeated_observations = repeated_observations + 1 if observation == last_observation else 1
                last_observation = observation
                if failed:
                    state.action_graph.mark_failed(action.id)
                    if was_pending:
                        state.action_graph.mark_failed(was_pending[0].id)
                        if not restore(snapshots):
                            return finish("rollback_failed")
                        snapshots.clear()
                    elif token is not None:
                        if not restore([snapshots.pop()]):
                            return finish("rollback_failed")
                    kind = result.metadata.get("failure_kind", FailureKind.ENVIRONMENT.value)
                    if kind not in {k.value for k in FailureKind}:
                        kind = FailureKind.APPROACH
                    if not rejected(result.error or "; ".join(c.summary for c in output.checks),
                                    fingerprint, kind):
                        return finish("repair_limit")
                    batch_failed = True
                    break
                if unresolved_process:
                    feedback({"observation": "Execution is inconclusive; poll the operation before further work.",
                              "poll_tool": pending[1].poll_tool, "operation": pending[2]})
                    if repeated_observations >= self.limits.max_stalled_actions:
                        return finish("no_progress")
                    break
                if all(c.passed for c in output.checks):
                    state.action_graph.mark_completed(action.id)
                    emit("action_completed", {"action": action.id})
                else:
                    if not rejected("; ".join(c.summary for c in output.checks), fingerprint,
                                    FailureKind.VERIFICATION):
                        return finish("repair_limit")
                    break
                if repeated_observations >= self.limits.max_stalled_actions:
                    return finish("no_progress")
            if completion and not batch_failed:
                # A textual Finish has no native call but is still a ReAct action.
                if not turn.tool_calls:
                    state.messages.append(Message(agent.id, "Action: Finish", kind="action"))
                    emit("react_action", {"tool": "Finish", "arguments": {}})
                outcome = checks(Phase.BEFORE_FINISH)
                feedback({"completion_checks": [
                    {"check": c.check_id, "status": c.status.value, "summary": c.summary}
                    for c in outcome.checks
                ], "accepted": self.mechanisms.has_outcome_checks and bool(outcome.checks)
                    and all(c.passed for c in outcome.checks)})
                if time.monotonic() - started >= self.limits.max_seconds:
                    return finish("time_limit")
                if self.mechanisms.has_outcome_checks and outcome.checks and all(item.passed for item in outcome.checks):
                    return finish("verified_completion", True)
                summaries = "; ".join(item.summary for item in outcome.checks) or "No outcome checks"
                # Unknown is not evidence that the changes are incorrect.
                if any(c.status == VerificationStatus.FAILED for c in outcome.checks):
                    if not restore(snapshots):
                        return finish("rollback_failed")
                    snapshots.clear()
                kind = (FailureKind.ARTIFACT if any(c.status == VerificationStatus.FAILED
                                                   for c in outcome.checks)
                        else FailureKind.VERIFICATION)
                if not rejected(summaries, "completion:" + summaries, kind):
                    return finish("completion_unverified")
        return finish("turn_limit")


class VerifiedFileRollback:
    """Bounded snapshots of declared regular file outputs, not arbitrary CLI state.

    Use only for tools whose complete mutation set is artifact_outputs. Refuse
    symlinks and directories. Cloud/container state needs a native backend.
    """

    def __init__(self, max_bytes: int = 8_000_000) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes

    @staticmethod
    def _path(task: TaskContext, relative: str):
        from pathlib import Path, PureWindowsPath
        root = task.workspace.resolve()
        path = root / relative
        if (not relative or Path(relative).is_absolute() or PureWindowsPath(relative).drive
                or ".." in Path(relative).parts):
            raise PermissionError("rollback path must be workspace-relative")
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise PermissionError("rollback path escapes workspace") from error
        cursor = path
        while cursor != root:
            if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
                raise PermissionError("rollback does not follow symlinks")
            cursor = cursor.parent
        return path

    def checkpoint(self, action, task):
        if not action.artifact_outputs:
            return None
        values = {}
        used = 0
        missing_parents = set()
        for relative in action.artifact_outputs:
            path = self._path(task, relative)
            if path.exists() and not path.is_file():
                return None
            used += path.stat().st_size if path.exists() else 0
            if used > self.max_bytes:
                raise ValueError("checkpoint byte limit exceeded")
            content = path.read_bytes() if path.exists() else None
            if content is not None and len(content) + sum(len(v[0] or b'') for v in values.values()) > self.max_bytes:
                raise ValueError("checkpoint byte limit exceeded")
            values[relative] = (content, path.stat().st_mode if content is not None else None)
            parent = path.parent
            while not parent.exists():
                missing_parents.add(str(parent.relative_to(task.workspace.resolve())))
                parent = parent.parent
        return (task.workspace.resolve(), values, tuple(missing_parents))

    def rollback(self, token, action, task):
        root, values, missing_parents = token
        if root != task.workspace.resolve():
            raise PermissionError("checkpoint belongs to another workspace")
        # Validate every target before beginning restoration.
        for relative in values:
            path = self._path(task, relative)
            if path.exists() and not path.is_file():
                raise ValueError("rollback target is no longer a regular file")
        for relative, (content, mode) in values.items():
            path = self._path(task, relative)
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                path.chmod(mode)
        for relative in sorted(missing_parents, key=lambda p: len(Path(p).parts), reverse=True):
            parent = self._path(task, relative)
            if parent.is_dir():
                parent.rmdir()  # Refuse to erase undeclared new contents.
        restored = all(
            (not self._path(task, p).exists()) if data is None
            else (self._path(task, p).read_bytes() == data and self._path(task, p).stat().st_mode == mode)
            for p, (data, mode) in values.items()
        ) and all(not self._path(task, p).exists() for p in missing_parents)
        return Evidence(action.id + ":rollback", "file-snapshot",
                        "Verified declared file contents after restoration",
                        data={"restored": restored, "paths": tuple(values)}, software_native=True)
