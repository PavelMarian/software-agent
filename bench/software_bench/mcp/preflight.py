from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from software_bench.harness.environments import EnvironmentSession
from software_bench.mcp.models import ApplicationSpec


_WHICH = (
    "import shutil,sys; "
    "path=shutil.which(sys.argv[1]); "
    "print(path or ''); "
    "raise SystemExit(0 if path else 1)"
)


@dataclass(frozen=True)
class ExecutableCheck:
    """Describe availability of one executable required by an MCP profile.

    Attributes:
        executable: Executable name declared by a tool command.
        available: Whether the executable resolves in the environment.
        resolved_path: Resolved executable path when available.
        diagnostic: Failed interpreter probes when resolution fails.
    """

    executable: str
    available: bool
    resolved_path: str = ""
    diagnostic: str = ""


def preflight_profile(
    spec: ApplicationSpec, environment: EnvironmentSession
) -> Mapping[str, Any]:
    """Check every distinct profile executable without invoking target software.

    Args:
        spec: Application profile whose command heads are inspected.
        environment: Environment used to resolve executable names.

    Returns:
        A serializable report containing readiness, executable counts, and
        individual resolution checks.
    """
    command_heads = {tool.command[0] for tool in spec.tools}
    executables = sorted(name for name in command_heads if "{" not in name and "}" not in name)
    dynamic = sorted(command_heads - set(executables))
    checks = tuple(_check_executable(name, environment) for name in executables)
    return {
        "profile": spec.name,
        "environment_backend": environment.backend_id,
        "ready": all(check.available for check in checks),
        "tool_count": len(spec.tools),
        "executable_count": len(checks),
        "dynamic_executable_count": len(dynamic),
        "dynamic_executables": dynamic,
        "checks": [asdict(check) for check in checks],
    }


def _check_executable(
    executable: str, environment: EnvironmentSession
) -> ExecutableCheck:
    """Resolve an executable through an environment-local Python interpreter.

    Args:
        executable: Executable name to locate.
        environment: Environment in which the lookup command runs.

    Returns:
        The successful resolution or accumulated lookup diagnostics.
    """
    diagnostics: list[str] = []
    for interpreter in ("python3", "python"):
        result = environment.run(
            [interpreter, "-c", _WHICH, executable], timeout_seconds=30
        )
        if result.exit_code == 0:
            return ExecutableCheck(executable, True, result.stdout.strip())
        diagnostics.append(
            f"{interpreter}: exit={result.exit_code} {result.stderr.strip()}".strip()
        )
    return ExecutableCheck(executable, False, diagnostic="; ".join(diagnostics))
