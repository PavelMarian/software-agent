"""Import and read upstream FoamBench records without a benchmark-framework dependency."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from foambench.models import FoamBenchError, FoamBenchTask


MANIFEST = "task.json"
PUBLIC_SEED = "seed"
PRIVATE_REFERENCE = "private/reference"


class FoamBenchCorpus:
    """A local imported corpus whose task metadata is safe to show to an agent."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def task_root(self, instance_id: str) -> Path:
        target = self.root / instance_id
        if target.parent != self.root:
            raise FoamBenchError("unsafe task id")
        return target

    def load(self, instance_id: str) -> FoamBenchTask:
        root = self.task_root(instance_id)
        try:
            raw = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FoamBenchError(f"cannot load task {instance_id}: {error}") from error
        return FoamBenchTask.from_dict(raw)

    def list_tasks(self) -> tuple[FoamBenchTask, ...]:
        if not self.root.is_dir():
            return ()
        return tuple(self.load(path.name) for path in sorted(self.root.iterdir()) if (path / MANIFEST).is_file())

    def materialize(self, instance_id: str, workspace: str | Path) -> FoamBenchTask:
        task = self.load(instance_id)
        source = self.task_root(instance_id) / PUBLIC_SEED
        target = Path(workspace).resolve()
        if target.exists():
            raise FoamBenchError(f"workspace already exists: {target}")
        if not source.is_dir():
            raise FoamBenchError(f"task {instance_id} lacks a public seed workspace")
        shutil.copytree(source, target)
        return task

    def reference_root(self, instance_id: str) -> Path:
        root = self.task_root(instance_id) / PRIVATE_REFERENCE
        if not root.is_dir():
            raise FoamBenchError(f"task {instance_id} lacks private reference assets")
        return root


def import_dataset(
    dataset_file: str | Path,
    corpus_root: str | Path,
    *,
    split: str,
    image: str = "openfoam/openfoam10-paraview510",
    case_ids: Sequence[str] = (),
) -> tuple[FoamBenchTask, ...]:
    """Import selected upstream JSON records into an independent local corpus."""
    if split not in {"basic", "advanced"}:
        raise FoamBenchError("FoamBench split must be basic or advanced")
    source = Path(dataset_file).resolve()
    records = _read_dataset(source)
    selected = set(case_ids)
    unknown = selected - set(records)
    if unknown:
        raise FoamBenchError(f"unknown FoamBench case ids: {sorted(unknown)}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    corpus = FoamBenchCorpus(corpus_root)
    corpus.root.mkdir(parents=True, exist_ok=True)
    imported: list[FoamBenchTask] = []
    for case_id, record in records.items():
        if selected and case_id not in selected:
            continue
        files = _validate_record(case_id, record)
        task = FoamBenchTask(
            instance_id=_instance_id(split, case_id),
            source_case_id=case_id,
            split=split,
            problem_statement=files["usr_requirement"],
            image=image,
            metadata={
                "dataset_sha256": digest,
                "upstream": "https://github.com/NREL-Theseus/cfdllmbench",
            },
        )
        root = corpus.task_root(task.instance_id)
        if root.exists():
            raise FoamBenchError(f"corpus task already exists: {root}")
        (root / PUBLIC_SEED).mkdir(parents=True)
        _write_files(root / PRIVATE_REFERENCE, files)
        (root / MANIFEST).write_text(json.dumps(task.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        imported.append(task)
    return tuple(imported)


def _read_dataset(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FoamBenchError(f"cannot read FoamBench dataset {path}: {error}") from error
    if not isinstance(value, Mapping) or not value:
        raise FoamBenchError("FoamBench dataset must be a non-empty JSON object")
    return value


def _validate_record(case_id: str, value: Any) -> dict[str, str]:
    if not isinstance(case_id, str) or not case_id:
        raise FoamBenchError("FoamBench case id must be a non-empty string")
    if not isinstance(value, Mapping):
        raise FoamBenchError(f"FoamBench case {case_id} must map paths to strings")
    result: dict[str, str] = {}
    for name, content in value.items():
        if not isinstance(name, str) or not isinstance(content, str):
            raise FoamBenchError(f"FoamBench case {case_id} contains a non-string file")
        if name != "usr_requirement":
            _safe_relative(name)
        result[name] = content
    if not result.get("usr_requirement", "").strip():
        raise FoamBenchError(f"FoamBench case {case_id} has no usr_requirement")
    if len(result) == 1:
        raise FoamBenchError(f"FoamBench case {case_id} has no reference files")
    return result


def _write_files(root: Path, files: Mapping[str, str]) -> None:
    for name, content in files.items():
        if name == "usr_requirement":
            continue
        target = root.joinpath(*_safe_relative(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise FoamBenchError(f"unsafe FoamBench file path: {value}")
    return path


def _instance_id(split: str, case_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", case_id).strip("-.")
    if not safe:
        raise FoamBenchError(f"cannot derive instance id from FoamBench case: {case_id}")
    return f"foambench__{split}__{safe}"
