"""Run public tasks while keeping corpus reference assets outside agent inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
from time import monotonic
from typing import Any, Callable, Mapping

from foambench.corpus import FoamBenchCorpus
from foambench.models import FoamBenchError, FoamBenchTask
from foambench.workspace import FoamBenchWorkspace
from software_multiagent.tools.workspace import WorkspaceToolset


Driver = Callable[[FoamBenchTask, WorkspaceToolset, Mapping[str, Any]], Mapping[str, Any] | None]


@dataclass(frozen=True)
class RunOutcome:
    task_id: str
    status: str
    output_dir: str
    elapsed_seconds: float
    stop_reason: str | None = None
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_driver(reference: str) -> Driver:
    """Load a user-owned agent factory without coupling the runner to a provider SDK."""
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise FoamBenchError("driver must use the form package.module:function")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise FoamBenchError(f"cannot load driver {reference}: {error}") from error
    if not callable(candidate):
        raise FoamBenchError(f"driver {reference} is not callable")
    return candidate


def run_task(
    corpus: FoamBenchCorpus,
    task_id: str,
    output_dir: str | Path,
    *,
    driver: Driver,
    model: str | None = None,
    provider: str | None = None,
    image: str | None = None,
    seed: int = 0,
    resume: bool = False,
) -> RunOutcome:
    """Run one task; driver inputs are restricted to task metadata and public tools."""
    target = Path(output_dir).resolve()
    manifest_path = target / "run.json"
    if resume and manifest_path.is_file():
        previous = _read_json(manifest_path)
        if previous.get("status") == "completed":
            return RunOutcome(task_id, "completed", str(target), 0.0, skipped=True)
    if target.exists():
        raise FoamBenchError(f"run output already exists: {target}; pass --resume or choose another path")
    target.mkdir(parents=True)
    started = monotonic()
    task = corpus.load(task_id)
    _write_json(target / "task.public.json", task.to_dict())
    try:
        corpus.materialize(task_id, target / "workspace")
        workspace = FoamBenchWorkspace(target / "workspace", image=image)
        context = {
            "model": model,
            "provider": provider,
            "seed": seed,
            "run_dir": str(target),
            "workspace": str(target / "workspace"),
            "tool_names": workspace.toolset().names,
        }
        # The driver never receives corpus.reference_root(), evaluator paths, or reference file contents.
        result = dict(driver(task, workspace.toolset(), context) or {})
        submission = workspace.collect_submission()
        _write_json(target / "prediction.json", {"task_id": task.instance_id, "files": submission})
        elapsed = round(monotonic() - started, 6)
        outcome = RunOutcome(task.instance_id, "completed", str(target), elapsed, result.get("stop_reason"))
        _write_json(target / "run.json", {
            **outcome.to_dict(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model, "provider": provider, "image": image,
            "seed": seed,
            "driver_result": result,
            "public_verification": workspace.public_verification(),
        })
        return outcome
    except Exception as error:
        elapsed = round(monotonic() - started, 6)
        outcome = RunOutcome(task.instance_id, "failed", str(target), elapsed, type(error).__name__)
        _write_json(target / "run.json", {
            **outcome.to_dict(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error": str(error),
        })
        return outcome


def run_split(
    corpus: FoamBenchCorpus,
    output_dir: str | Path,
    *,
    driver: Driver,
    split: str = "all",
    model: str | None = None,
    provider: str | None = None,
    image: str | None = None,
    seed: int = 0,
    resume: bool = False,
) -> tuple[RunOutcome, ...]:
    """Run each task in a split sequentially and persist an aggregate report."""
    if split not in {"basic", "advanced", "all"}:
        raise FoamBenchError("split must be basic, advanced, or all")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in corpus.list_tasks() if split == "all" or task.split == split]
    if not tasks:
        raise FoamBenchError(f"no tasks found for split {split}")
    outcomes = tuple(
        run_task(corpus, task.instance_id, root / task.instance_id, driver=driver, model=model, provider=provider, image=image, seed=seed, resume=resume)
        for task in tasks
    )
    _write_json(root / "summary.json", {
        "split": split,
        "total": len(outcomes),
        "completed": sum(item.status == "completed" for item in outcomes),
        "failed": sum(item.status == "failed" for item in outcomes),
        "outcomes": [item.to_dict() for item in outcomes],
    })
    return outcomes


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
