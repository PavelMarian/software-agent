from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from software_bench.core.artifacts import EncodedArtifact

@dataclass(frozen=True)
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class EnvironmentSession(Protocol):
    """A prepared mutable task environment shared by benchmark roles."""

    backend_id: str

    def read_text(self, path: str) -> str: ...

    def write_text(self, path: str, content: str) -> None: ...

    def read_bytes(self, path: str) -> bytes: ...

    def write_bytes(self, path: str, content: bytes) -> None: ...

    def run(self, command: Sequence[str], *, timeout_seconds: float) -> ExecutionResult: ...

    def list_files(self, path: str = ".") -> tuple[str, ...]: ...

    def extract_patch(self) -> str: ...

    def extract_files(self, paths: Sequence[str]) -> Mapping[str, EncodedArtifact]: ...

    def copy_from_host(self, source: Path, destination: str) -> None: ...

    def tool_declarations(self) -> Mapping[str, Mapping[str, Any]]: ...

    def invoke_tool(self, name: str, arguments: Mapping[str, Any]) -> ExecutionResult: ...

    def close(self) -> None: ...


