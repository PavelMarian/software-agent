from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from software_bench.core.models import AgentTopology, RoleSpec
from software_bench.harness.contracts import ModelAdapter, RunRequest
from software_bench.harness.agents.tools import EventRecorder, ToolExecutor


@dataclass(frozen=True)
class AgentLoopOutcome:
    """Summarize one role phase executed by the legacy agent loop.

    Attributes:
        turn_count: Number of model turns consumed by the phase.
        measurement_complete: Whether every model response reported full usage.
        stop_reason: Stable reason the phase stopped.
    """

    turn_count: int
    measurement_complete: bool
    stop_reason: str


class AgentLoop:
    """Run one configured role while retaining harness tool and budget control."""

    def run(
        self,
        request: RunRequest,
        role: RoleSpec,
        phase_index: int,
        adapter: ModelAdapter,
        tools: ToolExecutor,
        observer: EventRecorder,
        incoming_messages: Sequence[Mapping[str, str]] = (),
    ) -> AgentLoopOutcome:
        """Execute model and tool turns for one role phase.

        Args:
            request: Normalized benchmark run request.
            role: Role policy and permissions for this phase.
            phase_index: Zero-based phase position in the workflow.
            adapter: Model adapter used to generate turns.
            tools: Harness-controlled tool executor.
            observer: Trace and aggregate-budget observer.
            incoming_messages: Teammate messages delivered before the first
                model turn.

        Returns:
            Phase usage and termination metadata.

        Raises:
            ValueError: If a model tool call has an invalid shape.
        """
        history: list[Mapping[str, Any]] = [
            {"role": "user", "content": task_prompt(request, role.id)}
        ]
        if incoming_messages:
            observer.record(
                "messages_received",
                role.id,
                {
                    "count": len(incoming_messages),
                    "message_ids": [
                        str(message["id"])
                        for message in incoming_messages
                        if message.get("id")
                    ],
                    "senders": sorted(
                        {
                            str(message.get("from", ""))
                            for message in incoming_messages
                            if message.get("from")
                        }
                    ),
                },
            )
            history.append(
                {
                    "role": "user",
                    "content": "Messages from teammates:\n"
                    + json.dumps(list(incoming_messages), ensure_ascii=False),
                }
            )

        observer.record(
            "agent_started",
            role.id,
            {"phase": phase_index, "profile": role.profile},
        )
        measurement_complete = True
        stop_reason = "turn_limit_reached"
        turn_count = 0
        for turn in range(request.mode.max_turns_per_phase):
            response = adapter.generate(
                history,
                system_prompt(request.mode.agent_topology, role),
                tools.declarations(role),
                role_id=role.id,
                seed=request.seed,
            )
            turn_count += 1
            measurement_complete = measurement_complete and response.measurement_complete
            observer.record(
                "model_response",
                role.id,
                {
                    "phase": phase_index,
                    "turn": turn,
                    "text": response.text,
                    "adapter_metadata": dict(response.metadata),
                },
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            history.append({"role": "assistant", "content": response.text})
            for call in response.tool_calls:
                name = call.get("name")
                args = call.get("args", {})
                if not isinstance(name, str) or not isinstance(args, Mapping):
                    raise ValueError("tool calls require string name and object args")
                result = tools.execute(role.id, name, args)
                history.append({"role": "tool", "name": name, "content": result.to_dict()})
            if response.done:
                stop_reason = "agent_done"
                break

        observer.record(
            "agent_finished",
            role.id,
            {
                "phase": phase_index,
                "turn_count": turn_count,
                "stop_reason": stop_reason,
            },
        )
        return AgentLoopOutcome(turn_count, measurement_complete, stop_reason)


def task_prompt(request: RunRequest, role_id: str) -> str:
    """Render the public task and submission contract for one role.

    Args:
        request: Normalized benchmark run request.
        role_id: Identifier of the role receiving the prompt.

    Returns:
        A role-specific user prompt containing the objective and workstreams.
    """
    streams = "\n".join(
        f"- {item.title}: {item.description}" for item in request.task.workstreams
    )
    if str(request.task.task_kind) == "software_evolution":
        objective = (
            f"Evolve {request.task.repo} from {request.task.start_version} "
            f"to {request.task.end_version}."
        )
    else:
        objective = (
            f"Use {request.task.target_software} in the provided environment to produce "
            "the requested executable artifact. Do not merely describe a solution."
        )
    submission_kind = str(request.submission_kind)
    if submission_kind == "artifact_bundle":
        submission = (
            "Submission contract: place every final deliverable under the workspace-root "
            f"directory {request.submission_root}/. The harness recursively collects that "
            "directory; do not put final deliverables elsewhere."
        )
    elif submission_kind == "environment_state":
        submission = (
            "Submission contract: complete the task through the declared environment submission "
            "tools. The harness captures the resulting live-state evidence automatically."
        )
    elif submission_kind == "workspace_files":
        paths = ", ".join(request.submission_paths) or "<none declared>"
        submission = (
            "Legacy submission contract: create the final files or directories at these exact "
            f"workspace-root paths: {paths}. Only artifacts under these paths are collected."
        )
    else:
        submission = "Submission contract: modify the workspace; the final Git patch is collected."
    return (
        f"Task {request.task.instance_id} for role {role_id}.\n{objective}\n\n"
        f"{request.task.problem_statement}\n\n{submission}\n\n"
        f"Workstreams:\n{streams or '- not curated'}"
    )


def system_prompt(topology: AgentTopology, role: RoleSpec) -> str:
    """Render role instructions for a selected agent topology.

    Args:
        topology: Single-agent or team execution topology.
        role: Role identity, instructions, and available tools.

    Returns:
        A system prompt with identity, tool, and phase-completion rules.
    """
    identity = (
        "You are the benchmark's single task-solving software agent."
        if topology == AgentTopology.SINGLE_AGENT
        else f"You are the {role.profile} member of a software-using agent team."
    )
    terminal = (
        " End the assigned phase by calling finish_phase exactly once; use needs_revision "
        "only when concrete verification defects require another executor visit."
        if "finish_phase" in role.tools
        else " When your assigned work is complete, include TASK_COMPLETE in the response."
    )
    return (
        f"{identity} {role.instructions} Use only the declared tools. "
        "Keep visible responses concise and evidence-based; do not reveal private reasoning."
        f"{terminal}"
    )
