from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from software_multiagent.tool_generation.contracts.errors import ToolGenerationError
from software_multiagent.tool_generation.contracts.models import ApplicationRecipe
from software_multiagent.tool_generation.workflow.telemetry import GenerationTelemetry


@dataclass(frozen=True)
class ExplorationManifest:
    """Bounded evidence collected from an application repository."""

    root: str
    files: tuple[str, ...]
    documents: tuple[Mapping[str, Any], ...]
    python_symbols: tuple[Mapping[str, Any], ...]
    entry_points: tuple[Mapping[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe manifest for an explorer agent."""

        return {
            "root": self.root,
            "files": list(self.files),
            "documents": [dict(item) for item in self.documents],
            "python_symbols": [dict(item) for item in self.python_symbols],
            "entry_points": [dict(item) for item in self.entry_points],
        }


class ExplorerAgent(Protocol):
    """Agent boundary that converts grounded repository evidence into a recipe."""

    def propose(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        """Propose a recipe from a deterministic exploration manifest."""


class RepairAgent(Protocol):
    """Agent boundary for bounded recipe repair after live validation."""

    def repair(
        self,
        recipe: Mapping[str, Any],
        validation: Mapping[str, Any],
        attempt: int,
    ) -> Mapping[str, Any] | None:
        """Return a revised recipe or ``None`` when no safe repair is available."""


def inspect_application(
    root: str | Path,
    *,
    max_files: int = 1000,
    max_document_chars: int = 12000,
) -> ExplorationManifest:
    """Collect documentation, Python symbols, and declared entry points.

    Args:
        root: Application repository or source directory.
        max_files: Maximum number of repository files to enumerate.
        max_document_chars: Maximum text retained from each documentation file.

    Returns:
        A bounded, deterministic exploration manifest.

    Raises:
        ToolGenerationError: If the root is missing or exceeds the file budget.
    """

    source = Path(root).resolve()
    if not source.is_dir():
        raise ToolGenerationError(f"application source does not exist: {source}")
    paths = [item for item in sorted(source.rglob("*")) if item.is_file() and ".git" not in item.parts]
    if len(paths) > max_files:
        raise ToolGenerationError(
            f"application contains {len(paths)} files, exceeding exploration limit {max_files}"
        )
    relative = tuple(item.relative_to(source).as_posix() for item in paths)
    documents: list[Mapping[str, Any]] = []
    symbols: list[Mapping[str, Any]] = []
    entry_points: list[Mapping[str, str]] = []
    for path, rel in zip(paths, relative):
        lower = path.name.lower()
        if lower.startswith("readme") or path.suffix.lower() in {".md", ".rst"}:
            text = _read_text(path, max_document_chars)
            if text:
                documents.append({"path": rel, "text": text})
        if path.suffix == ".py":
            symbols.extend(_python_symbols(path, rel))
        if lower == "pyproject.toml":
            entry_points.extend(_pyproject_scripts(path))
    return ExplorationManifest(str(source), relative, tuple(documents), tuple(symbols), tuple(entry_points))


def explore_with_agent(
    root: str | Path,
    agent: ExplorerAgent,
    *,
    telemetry: GenerationTelemetry | None = None,
) -> tuple[ExplorationManifest, ApplicationRecipe]:
    """Inspect an application and ask an agent for a contract-checked recipe.

    Args:
        root: Application source directory.
        agent: Structured explorer agent.
        telemetry: Optional generation telemetry collector.

    Returns:
        The evidence manifest and validated recipe.
    """

    recorder = telemetry or GenerationTelemetry()
    with recorder.stage("exploration"):
        manifest = inspect_application(root)
        recorder.emit(
            "application_inspected",
            file_count=len(manifest.files),
            document_count=len(manifest.documents),
            symbol_count=len(manifest.python_symbols),
            entry_point_count=len(manifest.entry_points),
        )
        proposal = agent.propose(manifest.to_dict())
        recipe = ApplicationRecipe.from_dict(proposal)
        _verify_sources(recipe, manifest)
        recorder.emit("recipe_proposed", tool_count=len(recipe.tools))
        return manifest, recipe


def _verify_sources(recipe: ApplicationRecipe, manifest: ExplorationManifest) -> None:
    known = set(manifest.files)
    for tool in recipe.tools:
        source_path = tool.source.split(":", 1)[0].replace("\\", "/")
        if source_path not in known:
            raise ToolGenerationError(
                f"tool {tool.spec.name!r} cites source outside exploration manifest: {tool.source!r}"
            )


def _read_text(path: Path, limit: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _python_symbols(path: Path, relative: str) -> list[Mapping[str, Any]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []
    result: list[Mapping[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            item: dict[str, Any] = {
                "path": relative,
                "name": node.name,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "line": node.lineno,
                "doc": ast.get_docstring(node) or "",
            }
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                item["parameters"] = [argument.arg for argument in node.args.args]
            result.append(item)
    return result


def _pyproject_scripts(path: Path) -> list[Mapping[str, str]]:
    try:
        import tomllib

        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    scripts = value.get("project", {}).get("scripts", {})
    if not isinstance(scripts, Mapping):
        return []
    return [{"name": str(name), "target": str(target), "source": "pyproject.toml"} for name, target in scripts.items()]
