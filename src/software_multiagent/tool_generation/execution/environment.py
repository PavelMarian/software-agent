from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from software_multiagent.tool_generation.contracts.errors import ToolGenerationError


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class EnvironmentSession(Protocol):
    def run(
        self, command: Sequence[str], *, timeout_seconds: float
    ) -> ExecutionResult: ...

    def list_files(self, path: str = ".") -> tuple[str, ...]: ...

    def close(self) -> None: ...


class LocalEnvironmentSession:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ToolGenerationError(f"workspace does not exist: {self.workspace}")

    def run(
        self, command: Sequence[str], *, timeout_seconds: float
    ) -> ExecutionResult:
        if not command:
            raise ToolGenerationError("application command cannot be empty")
        try:
            completed = subprocess.run(
                list(command),
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return ExecutionResult(
                completed.stdout,
                completed.stderr,
                completed.returncode,
            )
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(
                _text(error.stdout),
                _text(error.stderr),
                124,
            )
        except OSError as error:
            return ExecutionResult("", f"cannot run {command[0]!r}: {error}", 127)

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

    def close(self) -> None:
        return None

    def _resolve(self, relative: str) -> Path:
        candidate = (self.workspace / relative).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as error:
            raise PermissionError(f"path escapes workspace: {relative}") from error
        return candidate


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value
