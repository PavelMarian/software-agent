from __future__ import annotations

from typing import Mapping

from software_bench.harness.contracts import RunRequest
from software_bench.harness.execution.trace import TraceEvent

def _completed_phases(events: list[TraceEvent]) -> int:
    runtime_count = sum(
        event.event_type == "software_multiagent.node_finished" for event in events
    )
    return runtime_count or sum(event.event_type == "phase_finished" for event in events)


def _available_tools(request: RunRequest) -> tuple[str, ...]:
    try:
        dynamic = dict(
            getattr(request.environment, "tool_declarations", lambda: {})()
        )
    except Exception:
        dynamic = {}
    names: set[str] = set()
    for role in request.mode.roles:
        for name in role.tools:
            if name == "environment":
                names.update(dynamic)
            else:
                names.add(name)
    return tuple(sorted(names))


def _completed_agents(events: list[TraceEvent]) -> int:
    runtime_count = sum(
        event.event_type == "software_multiagent.node_finished" for event in events
    )
    return runtime_count or sum(event.event_type == "agent_finished" for event in events)


def _agent_runtime(request: RunRequest) -> Mapping[str, str]:
    if request.mode.agent_runtime != "legacy":
        from software_bench.harness.models.registry import load_agent_runtime

        runtime = load_agent_runtime(request.mode.agent_runtime)
        return {
            "id": runtime.runtime_id,
            "version": runtime.version,
            "strategy": request.mode.agent_runtime,
        }
    if str(request.mode.agent_topology) in {"single_agent", "multi_agent"}:
        from software_multiagent import __version__

        metadata = {"id": "software_multiagent", "version": __version__}
        return metadata
    return {"id": "benchmark_native", "version": "0.5.0"}


