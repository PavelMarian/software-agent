from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from software_multiagent.tool_generation.contracts.errors import ToolGenerationError
from software_multiagent.tool_generation.contracts.models import ApplicationRecipe
from software_multiagent.tool_generation.contracts.specs import write_spec
from software_multiagent.tool_generation.discovery.exploration import (
    ExplorerAgent,
    RepairAgent,
    explore_with_agent,
)
from software_multiagent.tool_generation.execution.environment import EnvironmentSession
from software_multiagent.tool_generation.execution.validation import validate_recipe
from software_multiagent.tool_generation.execution.wrappers import compile_wrappers
from software_multiagent.tool_generation.workflow.telemetry import GenerationTelemetry


ValidationError = ToolGenerationError


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    spec_path: Path
    report_path: Path
    passed: bool
    validation_level: str
    telemetry_path: Path


def load_recipe(path: str | Path) -> ApplicationRecipe:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read tool-generation recipe {source}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValidationError("tool-generation recipe must contain an object")
    return ApplicationRecipe.from_dict(value)


def generate_application_tools(
    recipe: ApplicationRecipe,
    output_dir: str | Path,
    *,
    environment: EnvironmentSession | None = None,
    replace: bool = False,
    repair_agent: RepairAgent | None = None,
    max_repairs: int = 0,
    telemetry: GenerationTelemetry | None = None,
    exploration: Mapping[str, Any] | None = None,
) -> BuildResult:
    """Compile, validate, optionally repair, and publish application tools.

    Args:
        recipe: Evidence-grounded source recipe.
        output_dir: Publication directory.
        environment: Optional live execution environment.
        replace: Replace an existing non-empty publication.
        repair_agent: Optional agent receiving failed live diagnostics.
        max_repairs: Maximum number of agent repair attempts.
        telemetry: Optional run telemetry collector.
        exploration: Optional exploration manifest saved with the build.

    Returns:
        Paths and validation state for the published build.
    """

    if max_repairs < 0:
        raise ValidationError("max_repairs must be non-negative")
    target = Path(output_dir).resolve()
    if target.exists() and any(target.iterdir()) and not replace:
        raise ValidationError(f"output directory is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    recorder = telemetry or GenerationTelemetry()
    try:
        if exploration is not None:
            _write_json(staging / "exploration.json", exploration)
        current = recipe
        validation: dict[str, Any] = {}
        for attempt in range(max_repairs + 1):
            attempt_root = staging / "attempts" / f"attempt-{attempt + 1}"
            with recorder.stage("wrapper_generation", attempt=attempt + 1):
                compiled = compile_wrappers(current, attempt_root / "wrappers")
                write_spec(compiled.application, attempt_root / "application.mcp.json")
            with recorder.stage("validation", attempt=attempt + 1):
                validation = dict(validate_recipe(compiled, environment, recorder))
            _write_json(attempt_root / "recipe.json", current.to_dict())
            _write_json(attempt_root / "validation.json", validation)
            if validation["passed"]:
                break
            if repair_agent is None or attempt >= max_repairs:
                break
            with recorder.stage("repair", attempt=attempt + 1):
                revised = repair_agent.repair(current.to_dict(), validation, attempt + 1)
                if revised is None:
                    recorder.emit("repair_declined", attempt=attempt + 1)
                    break
                candidate = ApplicationRecipe.from_dict(revised)
                if candidate.to_dict() == current.to_dict():
                    recorder.emit("repair_unchanged", attempt=attempt + 1)
                    break
                current = candidate
                recorder.emit("recipe_repaired", attempt=attempt + 1)
        _write_json(staging / "plan.json", current.to_dict())
        final_compiled = compile_wrappers(current, staging / "wrappers")
        final_value = final_compiled.to_dict()
        staging_text = str(staging / "wrappers")
        target_text = str(target / "wrappers")
        for tool in final_value["tools"]:
            tool["command"] = [
                token.replace(staging_text, target_text) for token in tool["command"]
            ]
        published_recipe = ApplicationRecipe.from_dict(final_value)
        write_spec(published_recipe.application, staging / "application.mcp.json")
        report = {
            "application": current.application.name,
            "tool_count": len(current.tools),
            "generated": [tool.spec.name for tool in current.tools],
            "validation": validation,
        }
        _write_json(staging / "validation.json", report)
        recorder.emit(
            "build_finished",
            status="passed" if validation["passed"] else "failed",
            attempts=sum(
                event.get("event") == "stage_started" and event.get("stage") == "validation"
                for event in recorder.events
            ),
            tool_count=len(current.tools),
        )
        _write_json(staging / "telemetry.json", recorder.to_dict())
        if not validation["passed"]:
            failed = _available_failed_path(target)
            os.replace(staging, failed)
            raise ValidationError(
                "generated application tools failed live validation; "
                f"diagnostics were preserved in {failed}"
            )
        if target.exists():
            if any(target.iterdir()):
                shutil.rmtree(target)
            else:
                target.rmdir()
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            if not any(event.get("event") == "build_finished" for event in recorder.events):
                recorder.emit("build_finished", status="failed")
            _write_json(staging / "telemetry.json", recorder.to_dict())
            failed = _available_failed_path(target)
            os.replace(staging, failed)
        raise
    validation_level = "live" if environment is not None else "static"
    return BuildResult(
        target,
        target / "application.mcp.json",
        target / "validation.json",
        True,
        validation_level,
        target / "telemetry.json",
    )


def generate_from_application(
    application_root: str | Path,
    output_dir: str | Path,
    explorer_agent: ExplorerAgent,
    *,
    environment: EnvironmentSession | None = None,
    repair_agent: RepairAgent | None = None,
    max_repairs: int = 2,
    replace: bool = False,
    telemetry: GenerationTelemetry | None = None,
) -> BuildResult:
    """Research an application with an agent and generate validated tools.

    Args:
        application_root: Repository or application source directory.
        output_dir: Publication directory.
        explorer_agent: Agent that proposes tools from inspected evidence.
        environment: Optional live execution environment.
        repair_agent: Optional debugger agent for failed validation.
        max_repairs: Maximum debugger revisions.
        replace: Replace an existing output directory.

    Returns:
        The completed generation result.
    """

    telemetry = telemetry or GenerationTelemetry()
    manifest, recipe = explore_with_agent(application_root, explorer_agent, telemetry=telemetry)
    return generate_application_tools(
        recipe,
        output_dir,
        environment=environment,
        replace=replace,
        repair_agent=repair_agent,
        max_repairs=max_repairs,
        telemetry=telemetry,
        exploration=manifest.to_dict(),
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _available_failed_path(target: Path) -> Path:
    candidate = target.parent / f"{target.name}.failed"
    index = 2
    while candidate.exists():
        candidate = target.parent / f"{target.name}.failed-{index}"
        index += 1
    return candidate
