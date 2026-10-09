from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from software_bench.core.models import CheckSpec, TestCommand, ValidationError
from software_bench.validation.evaluators.application import APPLICATION_RULES


def install_application_evaluator(
    bundle_root: Path,
    profile: str,
    validation: Mapping[str, Any] | None = None,
) -> tuple[tuple[CheckSpec, ...], tuple[TestCommand, ...]]:
    """Copy a standalone hidden evaluator and return its evaluation contract."""
    if profile not in APPLICATION_RULES:
        raise ValidationError(f"unknown application validation profile: {profile}")
    if validation is not None and not isinstance(validation, Mapping):
        raise ValidationError("application validation must be an object")

    evaluator = bundle_root / "evaluator"
    evaluator.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "evaluators" / "application.py"
    shutil.copy2(source, evaluator / "application_evaluator.py")
    specification = dict(validation or {})
    (evaluator / "validation.json").write_text(
        json.dumps(specification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checks = [
        CheckSpec("artifact_structure", weight=0.4),
        CheckSpec("native_validation", weight=0.6),
    ]
    if specification.get("metrics"):
        checks = [
            CheckSpec("artifact_structure", weight=0.25),
            CheckSpec("native_validation", weight=0.35),
            CheckSpec("numerical_accuracy", weight=0.4),
        ]
    command = TestCommand(
        id="application-validator",
        command=(
            "python",
            ".benchmark/application_evaluator.py",
            "--profile",
            profile,
            "--submission",
            ".",
            "--validation",
            ".benchmark/validation.json",
        ),
        parser="json-status",
    )
    return tuple(checks), (command,)
