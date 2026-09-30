"""FoamBench-to-neutral-task adapter; it does not construct an agent."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from software_multiagent.application import SoftwareMultiAgent, SoftwareRunRequest

def run(task, tools, context: Mapping[str, Any]) -> dict[str, Any]:
    """Translate public benchmark data and delegate to the primary multi-agent."""
    request = SoftwareRunRequest(
        task_id=task.instance_id,
        objective=task.problem_statement,
        workspace=Path(str(context["workspace"])),
        target_software=task.target_software,
        metadata=dict(task.metadata),
    )
    return SoftwareMultiAgent.run_configured(
        request,
        tools,
        provider=context.get("provider"),
        model_name=str(context["model"]) if context.get("model") else None,
    )
