"""Neutral file and program tool contracts with an injected domain backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping


class WorkspaceToolError(ValueError):
    """A tool request violates the workspace boundary or its declared schema."""


class WorkspaceFiles:
    """Constrained text-file access below a caller-selected workspace subdirectory."""

    def __init__(self, root: str | Path, *, writable_root: str = ".") -> None:
        self.root = Path(root).resolve()
        self.writable_root = self._resolve_root(writable_root)

    def _resolve_root(self, value: str) -> Path:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise WorkspaceToolError("workspace root must be relative")
        target = (self.root / Path(*path.parts)).resolve()
        if target != self.root and self.root not in target.parents:
            raise WorkspaceToolError("workspace root escapes base directory")
        return target

    def resolve(self, value: str) -> Path:
        if not isinstance(value, str) or not value or "\\" in value:
            raise WorkspaceToolError("path must be a non-empty POSIX relative path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or ":" in value:
            raise WorkspaceToolError("path escapes workspace")
        target = (self.writable_root / Path(*path.parts)).resolve()
        if target != self.writable_root and self.writable_root not in target.parents:
            raise WorkspaceToolError("path escapes writable workspace")
        return target

    def read_file(self, path: str, *, limit: int = 16_000) -> dict[str, str]:
        return {"content": self.resolve(path).read_text(encoding="utf-8", errors="replace")[-limit:]}

    def list_files(self, path: str = ".") -> dict[str, list[str]]:
        root = self.resolve(path) if path != "." else self.writable_root
        if not root.is_dir():
            raise WorkspaceToolError(f"not a directory: {path}")
        return {"files": [item.relative_to(self.writable_root).as_posix() for item in sorted(root.rglob("*")) if item.is_file()]}

    def write_file(self, path: str, content: str) -> dict[str, str]:
        if not isinstance(content, str):
            raise WorkspaceToolError("content must be text")
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": target.relative_to(self.writable_root).as_posix()}


ProgramRunner = Callable[[str, float, tuple[str, ...]], Mapping[str, Any]]
Verifier = Callable[[], Mapping[str, Any]]


@dataclass
class WorkspaceToolset:
    """Model-callable neutral tools; program semantics belong to an injected adapter."""

    files: WorkspaceFiles
    run_program: ProgramRunner
    verify_workspace: Verifier

    names = ("read_file", "list_files", "write_file", "run_program", "verify_workspace")

    def execute(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if name == "read_file":
            return self.files.read_file(str(arguments["path"]))
        if name == "list_files":
            return self.files.list_files(str(arguments.get("path", ".")))
        if name == "write_file":
            return self.files.write_file(str(arguments["path"]), arguments["content"])
        if name == "run_program":
            timeout = float(arguments.get("timeout_seconds", 300))
            arguments_list = arguments.get("arguments", ())
            if not isinstance(arguments_list, (list, tuple)) or not all(isinstance(item, str) for item in arguments_list):
                raise WorkspaceToolError("arguments must be a list of strings")
            return self.run_program(str(arguments["program"]), timeout, tuple(arguments_list))
        if name == "verify_workspace":
            return self.verify_workspace()
        raise WorkspaceToolError(f"unknown workspace tool: {name}")
