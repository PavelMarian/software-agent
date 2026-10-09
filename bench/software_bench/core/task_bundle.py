from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from software_bench.core.models import (
    EnvironmentSpec,
    EvaluationSpec,
    ProvenanceSpec,
    TaskBundle,
    TaskSpec,
    ValidationError,
)


BUNDLE_FILES = {
    "task": "task.json",
    "evaluation": "evaluation.json",
    "environment": "environment.json",
    "provenance": "provenance.json",
}


def load_task_bundle(path: str | Path, *, require_mas_ready: bool = False) -> TaskBundle:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValidationError(f"TaskBundle must be a directory: {root}")
    task = TaskSpec.from_dict(_read_object(root / BUNDLE_FILES["task"]))
    evaluation = EvaluationSpec.from_dict(_read_object(root / BUNDLE_FILES["evaluation"]))
    environment = EnvironmentSpec.from_dict(_read_object(root / BUNDLE_FILES["environment"]))
    provenance = ProvenanceSpec.from_dict(_read_object(root / BUNDLE_FILES["provenance"]))
    if require_mas_ready:
        task.validate_mas_ready()
    return TaskBundle(
        root=root,
        task=task,
        evaluation=evaluation,
        environment=environment,
        provenance=provenance,
    )


def write_task_bundle(bundle: TaskBundle, path: str | Path) -> None:
    """Write a generated/imported bundle; intended for dataset tooling."""
    from dataclasses import asdict

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    evaluation_value = {
        "strategy": str(bundle.evaluation.strategy),
        "submission_kind": str(bundle.evaluation.submission_kind),
        "test_patch": bundle.evaluation.test_patch,
        "FAIL_TO_PASS": list(bundle.evaluation.fail_to_pass),
        "PASS_TO_PASS": list(bundle.evaluation.pass_to_pass),
        "checks": [asdict(item) for item in bundle.evaluation.checks],
        "test_commands": [asdict(item) for item in bundle.evaluation.test_commands],
        "additional_oracles": [
            {**asdict(item), "layer": str(item.layer)}
            for item in bundle.evaluation.additional_oracles
        ],
        "evaluator_assets": bundle.evaluation.evaluator_assets,
        "backend_hint": bundle.evaluation.backend_hint,
    }
    if bundle.evaluation.submission_root is not None:
        evaluation_value["submission_root"] = bundle.evaluation.submission_root
    if bundle.evaluation.state_paths:
        evaluation_value["state_paths"] = list(bundle.evaluation.state_paths)
    if bundle.evaluation.submission_paths:
        evaluation_value["submission_paths"] = list(bundle.evaluation.submission_paths)
    values = {
        "task.json": asdict(bundle.task),
        "evaluation.json": evaluation_value,
        "environment.json": asdict(bundle.environment),
        "provenance.json": asdict(bundle.provenance),
    }
    for filename, value in values.items():
        (root / filename).write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path.name} root must be an object")
    return value
