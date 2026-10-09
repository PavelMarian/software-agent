from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from software_multiagent.tool_generation.contracts.models import ApplicationRecipe, ToolRecipe
from software_multiagent.tool_generation.contracts.specs import render_command, validate_arguments
from software_multiagent.tool_generation.execution.environment import EnvironmentSession
from software_multiagent.tool_generation.workflow.telemetry import GenerationTelemetry


_WHICH = (
    "import shutil,sys; path=shutil.which(sys.argv[1]); print(path or ''); "
    "raise SystemExit(0 if path else 1)"
)


@dataclass(frozen=True)
class ToolValidation:
    name: str
    status: str
    quality: str
    diagnostics: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["diagnostics"] = list(self.diagnostics)
        value["command"] = list(self.command)
        return value


def _preflight(
    recipe: ApplicationRecipe,
    environment: EnvironmentSession,
    telemetry: GenerationTelemetry | None = None,
) -> dict[str, Any]:
    executables = sorted(
        {
            tool.spec.command[0]
            for tool in recipe.tools
            if tool.spec.command and "{" not in tool.spec.command[0]
        }
    )
    checks = [_check_executable(executable, environment, telemetry) for executable in executables]
    return {
        "profile": recipe.application.name,
        "ready": all(bool(item["available"]) for item in checks),
        "executable_count": len(checks),
        "checks": checks,
    }


def validate_recipe(
    recipe: ApplicationRecipe,
    environment: EnvironmentSession | None,
    telemetry: GenerationTelemetry | None = None,
) -> Mapping[str, Any]:
    if environment is None:
        tools = [
            ToolValidation(
                tool.spec.name,
                "skipped",
                "generated",
                ("no validation environment was supplied",),
            )
            for tool in recipe.tools
        ]
        return {
            "passed": True,
            "level": "static",
            "preflight": None,
            "tools": [tool.to_dict() for tool in tools],
        }

    preflight = _preflight(recipe, environment, telemetry)
    available = {
        item["executable"]: bool(item["available"])
        for item in preflight["checks"]
    }
    for executable in recipe.required_executables:
        if executable not in available:
            check = _check_executable(executable, environment, telemetry)
            preflight["checks"].append(check)
            available[executable] = bool(check["available"])
    missing_required = [
        executable for executable in recipe.required_executables
        if not available.get(executable, False)
    ]
    preflight["ready"] = all(available.values())
    preflight["executable_count"] = len(available)
    results = [_validate_tool(environment, tool, telemetry) for tool in recipe.tools]
    passed = (
        bool(preflight["ready"])
        and not missing_required
        and all(item.status in {"passed", "skipped"} for item in results)
    )
    preflight["required_executables"] = list(recipe.required_executables)
    preflight["missing_required_executables"] = missing_required
    return {
        "passed": passed,
        "level": "live",
        "preflight": preflight,
        "tools": [item.to_dict() for item in results],
    }


def _check_executable(
    executable: str,
    environment: EnvironmentSession,
    telemetry: GenerationTelemetry | None = None,
) -> dict[str, Any]:
    diagnostics: list[str] = []
    for interpreter in ("python3", "python"):
        result = environment.run(
            [interpreter, "-c", _WHICH, executable],
            timeout_seconds=30,
        )
        if result.exit_code == 0:
            if telemetry is not None:
                telemetry.emit(
                    "executable_checked",
                    executable=executable,
                    available=True,
                    interpreter=interpreter,
                )
            return {
                "executable": executable,
                "available": True,
                "resolved_path": result.stdout.strip(),
                "diagnostic": "",
            }
        diagnostics.append(
            f"{interpreter}: exit={result.exit_code} {result.stderr.strip()}".strip()
        )
    if telemetry is not None:
        telemetry.emit("executable_checked", executable=executable, available=False)
    return {
        "executable": executable,
        "available": False,
        "resolved_path": "",
        "diagnostic": "; ".join(diagnostics),
    }


def _validate_tool(
    environment: EnvironmentSession,
    recipe: ToolRecipe,
    telemetry: GenerationTelemetry | None = None,
) -> ToolValidation:
    if recipe.sample_arguments is None:
        return ToolValidation(
            recipe.spec.name,
            "skipped",
            "generated",
            ("sample_arguments is null; only schema and command structure were checked",),
        )
    validate_arguments(recipe.spec.input_schema, recipe.sample_arguments)
    command = render_command(recipe.spec, recipe.sample_arguments)
    started = time.perf_counter()
    result = environment.run(command, timeout_seconds=recipe.spec.timeout_seconds)
    duration = time.perf_counter() - started
    stdout = result.stdout[-8000:]
    stderr = result.stderr[-4000:]
    exit_code = result.exit_code
    diagnostics: list[str] = []
    expected = recipe.expected
    if exit_code != expected.exit_code:
        diagnostics.append(f"exit code {exit_code}, expected {expected.exit_code}")
    diagnostics.extend(
        f"stdout does not contain {fragment!r}"
        for fragment in expected.stdout_contains
        if fragment not in stdout
    )
    diagnostics.extend(
        f"stderr does not contain {fragment!r}"
        for fragment in expected.stderr_contains
        if fragment not in stderr
    )
    if expected.stdout_json:
        try:
            json.loads(stdout)
        except json.JSONDecodeError as error:
            diagnostics.append(f"stdout is not valid JSON: {error}")
    files = set(environment.list_files())
    diagnostics.extend(
        f"expected path was not created: {path}"
        for path in expected.created_paths
        if path not in files
    )
    evidence_backed = bool(recipe.evidence and _has_correctness_expectation(recipe))
    validation = ToolValidation(
        recipe.spec.name,
        "failed" if diagnostics else "passed",
        "validated" if not diagnostics and evidence_backed else "smoke-tested",
        tuple(diagnostics),
        tuple(command),
        exit_code,
    )
    if telemetry is not None:
        telemetry.emit(
            "tool_invoked",
            tool=recipe.spec.name,
            status=validation.status,
            quality=validation.quality,
            duration_seconds=duration,
            exit_code=exit_code,
            stdout_characters=len(result.stdout),
            stderr_characters=len(result.stderr),
            diagnostic_count=len(diagnostics),
        )
    return validation


def _has_correctness_expectation(recipe: ToolRecipe) -> bool:
    expected = recipe.expected
    return bool(
        expected.stdout_contains
        or expected.stderr_contains
        or expected.created_paths
        or expected.stdout_json
    )
