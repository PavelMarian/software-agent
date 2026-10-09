from __future__ import annotations

from software_bench.harness.environments.common import resolve_host_command
from software_bench.harness.environments.contracts import EnvironmentSession, ExecutionResult
from software_bench.harness.environments.docker import DockerEnvironmentSession
from software_bench.harness.environments.factory import create_environment_session
from software_bench.harness.environments.local import LocalEnvironmentSession
from software_bench.harness.environments.managed import ManagedProcessEnvironmentSession

__all__ = [
    "DockerEnvironmentSession",
    "EnvironmentSession",
    "ExecutionResult",
    "LocalEnvironmentSession",
    "ManagedProcessEnvironmentSession",
    "create_environment_session",
    "resolve_host_command",
]
