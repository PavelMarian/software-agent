from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from software_bench.core.artifacts import EncodedArtifact
from software_bench.core.models import EnvironmentSpec, ValidationError
from software_bench.harness.environments.common import (
    _collect_local_files,
    _diff,
    _snapshot,
    _tail,
    resolve_host_command,
)
from software_bench.harness.environments.contracts import ExecutionResult

class LocalEnvironmentSession:
    """Local development backend preserving the original workspace behavior."""

    backend_id = "local"

    def __init__(
        self,
        workspace: Path,
        environment: EnvironmentSpec | None = None,
        *,
        remove_on_close: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValidationError(f"workspace does not exist: {self.workspace}")
        self._before = _snapshot(self.workspace)
        self._remove_on_close = remove_on_close
        raw_run_env = (
            environment.backend_config.get("run_environment", {})
            if environment is not None
            else {}
        )
        if not isinstance(raw_run_env, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_run_env.items()
        ):
            raise ValidationError("backend_config.run_environment must map strings to strings")
        self._run_environment = dict(raw_run_env)
        from software_bench.harness.environments.http import ConfiguredHttpTools

        self._http_tools = ConfiguredHttpTools(
            environment.backend_config if environment is not None else {},
            write_text=self.write_text,
        )

    def read_text(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def write_bytes(self, path: str, content: bytes) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def run(self, command: Sequence[str], *, timeout_seconds: float) -> ExecutionResult:
        try:
            completed = subprocess.run(
                resolve_host_command(command),
                cwd=self.workspace,
                env={**os.environ, **self._run_environment},
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return ExecutionResult(completed.stdout, completed.stderr, completed.returncode)
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(_tail(error.stdout), _tail(error.stderr), 124)
        except OSError as error:
            executable = command[0] if command else "<empty command>"
            return ExecutionResult("", f"cannot run {executable!r} on the host: {error}", 127)

    def list_files(self, path: str = ".") -> tuple[str, ...]:
        target = self._resolve(path)
        if target.is_file():
            return (target.relative_to(self.workspace).as_posix(),)
        if not target.is_dir():
            return ()
        return tuple(
            item.relative_to(self.workspace).as_posix()
            for item in sorted(target.rglob("*"))
            if item.is_file() and ".git" not in item.parts
        )

    def extract_patch(self) -> str:
        return _diff(self._before, _snapshot(self.workspace))

    def extract_files(self, paths: Sequence[str]) -> Mapping[str, EncodedArtifact]:
        return _collect_local_files(self.workspace, paths)

    def extract_text_files(self, paths: Sequence[str]) -> Mapping[str, EncodedArtifact]:
        """Compatibility alias for the pre-binary environment contract."""
        return self.extract_files(paths)

    def copy_from_host(self, source: Path, destination: str) -> None:
        target = self._resolve(destination)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def tool_declarations(self) -> Mapping[str, Mapping[str, Any]]:
        return self._http_tools.declarations()

    def invoke_tool(self, name: str, arguments: Mapping[str, Any]) -> ExecutionResult:
        return self._http_tools.invoke(name, arguments)

    def close(self) -> None:
        if self._remove_on_close and self.workspace.exists():
            shutil.rmtree(self.workspace)

    def _resolve(self, relative: str) -> Path:
        candidate = (self.workspace / relative).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as error:
            raise PermissionError(f"path escapes workspace: {relative}") from error
        return candidate


