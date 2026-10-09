from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from software_bench.core.models import (
    CheckSpec,
    EnvironmentSpec,
    EvaluationSpec,
    EvaluationStrategy,
    ProvenanceSpec,
    SubmissionKind,
    TaskBundle,
    TaskKind,
    TaskSpec,
    TestCommand,
    ValidationError,
    Workstream,
)
from software_bench.core.task_bundle import write_task_bundle
from software_bench.application_profiles import APPLICATION_IMAGES
from software_bench.validation import install_application_evaluator


@dataclass(frozen=True)
class ScientificSource:
    id: str
    target_software: str
    mcp_profile: str
    repository: str
    marker: str
    default_image: str


SCIENTIFIC_SOURCES: Mapping[str, ScientificSource] = {
    "quantum-espresso-test-suite": ScientificSource(
        "quantum-espresso-test-suite",
        "Quantum ESPRESSO",
        "quantum_espresso",
        "https://gitlab.com/QEF/q-e",
        "test-suite",
        APPLICATION_IMAGES["quantum_espresso"],
    ),
    "openmc-tests": ScientificSource(
        "openmc-tests",
        "OpenMC",
        "openmc",
        "https://github.com/openmc-dev/openmc",
        "tests/regression_tests",
        APPLICATION_IMAGES["openmc"],
    ),
    "su2-testcases": ScientificSource(
        "su2-testcases",
        "SU2",
        "su2",
        "https://github.com/su2code/TestCases",
        "serial_regression.py",
        APPLICATION_IMAGES["su2"],
    ),
    "energyplus-tests": ScientificSource(
        "energyplus-tests",
        "EnergyPlus",
        "energyplus",
        "https://github.com/NREL/EnergyPlus",
        "testfiles",
        APPLICATION_IMAGES["energyplus"],
    ),
    "modflow6-tests": ScientificSource(
        "modflow6-tests",
        "MODFLOW 6 with FloPy",
        "modflow",
        "https://github.com/MODFLOW-ORG/modflow6",
        "autotest",
        APPLICATION_IMAGES["modflow"],
    ),
}


def import_scientific_dataset(
    source_id: str,
    upstream_root: str | Path,
    recipe_file: str | Path,
    output_root: str | Path,
    *,
    image: str | None = None,
    case_ids: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Import selected cases from an official scientific regression checkout.

    The recipe identifies public case assets and hidden evaluator assets. Paths are
    resolved only below ``upstream_root`` so a recipe cannot copy arbitrary host files.
    """
    source = SCIENTIFIC_SOURCES.get(source_id)
    if source is None:
        raise ValidationError(f"unknown scientific source: {source_id}")
    upstream = Path(upstream_root).resolve()
    marker = _source_path(upstream, source.marker)
    if not marker.exists():
        raise ValidationError(
            f"{source_id} checkout marker is missing: {source.marker}"
        )
    recipe_path = Path(recipe_file).resolve()
    recipe = _read_object(recipe_path)
    if recipe.get("source") != source_id:
        raise ValidationError(
            f"scientific recipe source must be {source_id!r}"
        )
    raw_cases = recipe.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValidationError("scientific recipe cases must be a non-empty array")
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise ValidationError("each scientific case must be an object")
        case_id = _required_string(raw, "id")
        if case_id in by_id:
            raise ValidationError(f"duplicate scientific case id: {case_id}")
        by_id[case_id] = raw
    selected = set(case_ids)
    unknown = selected - set(by_id)
    if unknown:
        raise ValidationError(f"unknown scientific case ids: {sorted(unknown)}")

    recipe_digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    upstream_revision = _optional_string(recipe, "upstream_revision")
    output = Path(output_root).resolve()
    written: list[Path] = []
    for case_id, raw in by_id.items():
        if selected and case_id not in selected:
            continue
        instance_id = f"{source.id}__{_safe_id(case_id)}"
        root = output / instance_id
        if root.exists():
            raise ValidationError(f"output TaskBundle already exists: {root}")
        case_path = _required_string(raw, "case_path")
        workspace_value = raw.get("workspace_assets")
        evaluator_value = raw.get("evaluator_assets", [])
        recipe_evaluator_value = raw.get("recipe_evaluator_assets", [])
        public_sources = _mapping_sources(
            upstream,
            workspace_value,
            default_source=case_path,
            default_target="case",
            field="workspace_assets",
        )
        hidden_sources = _mapping_sources(
            upstream,
            evaluator_value,
            field="evaluator_assets",
        )
        overlap = [
            str(hidden.relative_to(upstream))
            for hidden in hidden_sources
            if any(_contains(public, hidden) for public in public_sources)
        ]
        if overlap:
            raise ValidationError(
                "hidden evaluator assets overlap public workspace sources: "
                f"{sorted(overlap)}"
            )
        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        _copy_mappings(
            upstream,
            workspace,
            workspace_value,
            default_source=case_path,
            default_target="case",
            field="workspace_assets",
        )
        (workspace / "output").mkdir()
        (workspace / "output" / ".gitkeep").write_text("", encoding="utf-8")

        has_evaluator = True
        if evaluator_value:
            _copy_mappings(
                upstream,
                root / "evaluator",
                evaluator_value,
                field="evaluator_assets",
            )
        if recipe_evaluator_value:
            _copy_mappings(
                recipe_path.parent,
                root / "evaluator",
                recipe_evaluator_value,
                field="recipe_evaluator_assets",
            )
        validation = raw.get("validation", {})
        if not isinstance(validation, Mapping):
            raise ValidationError("scientific case validation must be an object")
        default_checks, default_commands = install_application_evaluator(
            root, source.mcp_profile, validation
        )
        commands = _commands(raw, default_commands)
        checks = _checks(raw, default_checks)
        environment_value = raw.get("environment", {})
        if not isinstance(environment_value, Mapping):
            raise ValidationError("scientific case environment must be an object")
        timeout = environment_value.get("timeout_seconds", 1800)
        environment = EnvironmentSpec.from_dict(
            {
                "backend_hint": environment_value.get("backend_hint", "docker"),
                "image": image or environment_value.get("image") or source.default_image,
                "workdir": environment_value.get("workdir", "/workspace"),
                "platform": environment_value.get("platform", "linux/x86_64"),
                "timeout_seconds": timeout,
                "network_enabled": environment_value.get("network_enabled", False),
                "workspace_source": "bundle",
                "seed_path": "workspace",
                "shell_init": environment_value.get("shell_init"),
                "pass_env": environment_value.get("pass_env", []),
                "backend_config": environment_value.get("backend_config", {}),
            }
        )
        task = TaskSpec(
            instance_id=instance_id,
            problem_statement=_required_string(raw, "problem_statement"),
            task_kind=TaskKind.SOFTWARE_USE,
            target_software=source.target_software,
            workstreams=_workstreams(raw),
            metadata={
                "scientific_source": source.id,
                "source_case_id": case_id,
                "mcp_profile": source.mcp_profile,
                **_metadata(raw),
            },
        )
        evaluation = EvaluationSpec(
            strategy=EvaluationStrategy.COMMAND_CHECKS,
            submission_kind=SubmissionKind.ARTIFACT_BUNDLE,
            checks=checks,
            test_commands=commands,
            submission_root="output",
            evaluator_assets="evaluator" if has_evaluator else None,
        )
        provenance = ProvenanceSpec(
            source=source.id,
            source_id=case_id,
            metadata={
                "upstream": source.repository,
                "upstream_revision": upstream_revision,
                "case_path": case_path,
                "recipe_sha256": recipe_digest,
            },
        )
        write_task_bundle(
            TaskBundle(root, task, evaluation, environment, provenance), root
        )
        written.append(root)
    return tuple(written)


def _copy_mappings(
    upstream: Path,
    destination: Path,
    value: Any,
    *,
    field: str,
    default_source: str | None = None,
    default_target: str | None = None,
) -> tuple[Path, ...]:
    entries = _asset_entries(
        value,
        default_source=default_source,
        default_target=default_target,
        field=field,
    )
    copied_sources: list[Path] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValidationError(f"each {field} entry must be an object")
        source_name = _required_string(item, "source")
        target_name = _required_string(item, "target")
        source = _source_path(upstream, source_name)
        if not source.exists():
            raise ValidationError(f"scientific asset does not exist: {source_name}")
        target = _destination_path(destination, target_name)
        if target.exists():
            raise ValidationError(f"duplicate scientific asset target: {target_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        copied_sources.append(source)
    return tuple(copied_sources)


def _mapping_sources(
    upstream: Path,
    value: Any,
    *,
    field: str,
    default_source: str | None = None,
    default_target: str | None = None,
) -> tuple[Path, ...]:
    sources: list[Path] = []
    for item in _asset_entries(
        value,
        default_source=default_source,
        default_target=default_target,
        field=field,
    ):
        source_name = _required_string(item, "source")
        _required_string(item, "target")
        source = _source_path(upstream, source_name)
        if not source.exists():
            raise ValidationError(f"scientific asset does not exist: {source_name}")
        sources.append(source)
    return tuple(sources)


def _asset_entries(
    value: Any,
    *,
    field: str,
    default_source: str | None,
    default_target: str | None,
) -> list[Mapping[str, Any]]:
    if value is None:
        if default_source is None or default_target is None:
            entries: list[Any] = []
        else:
            entries = [{"source": default_source, "target": default_target}]
    elif isinstance(value, list):
        entries = value
    else:
        raise ValidationError(f"{field} must be an array")
    if not all(isinstance(item, Mapping) for item in entries):
        raise ValidationError(f"each {field} entry must be an object")
    return entries


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _commands(
    value: Mapping[str, Any], default: tuple[TestCommand, ...]
) -> tuple[TestCommand, ...]:
    raw = value.get("test_commands")
    if raw is None:
        return default
    if not isinstance(raw, list) or not raw:
        raise ValidationError("scientific case test_commands must be a non-empty array")
    return tuple(TestCommand.from_dict(item) for item in raw)


def _checks(
    value: Mapping[str, Any], default: tuple[CheckSpec, ...]
) -> tuple[CheckSpec, ...]:
    raw = value.get("checks")
    if raw is None:
        return default
    if not isinstance(raw, list) or not raw:
        raise ValidationError("scientific case checks must be a non-empty array")
    return tuple(CheckSpec.from_dict(item) for item in raw)


def _workstreams(value: Mapping[str, Any]) -> tuple[Workstream, ...]:
    raw = value.get("workstreams")
    if raw is not None:
        if not isinstance(raw, list):
            raise ValidationError("scientific case workstreams must be an array")
        return tuple(Workstream.from_dict(item) for item in raw)
    return (
        Workstream(
            "model_preparation",
            "Model preparation",
            "Inspect and prepare the upstream scientific model and its inputs.",
        ),
        Workstream(
            "simulation",
            "Simulation",
            "Select and execute the appropriate scientific application tools.",
        ),
        Workstream(
            "validation",
            "Validation",
            "Check convergence and scientific outputs before submission.",
        ),
    )


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = value.get("metadata", {})
    if not isinstance(raw, Mapping):
        raise ValidationError("scientific case metadata must be an object")
    return dict(raw)


def _source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValidationError(f"scientific source path escapes checkout: {relative}") from error
    return path


def _destination_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValidationError(f"unsafe scientific destination path: {relative}")
    return root.joinpath(*pure.parts)


def _safe_id(value: str) -> str:
    result = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    if not result.strip(".-"):
        raise ValidationError(f"cannot derive scientific instance id from {value!r}")
    return result


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValidationError(f"{key} must be a non-empty string")
    return item


def _optional_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key, "")
    if not isinstance(item, str):
        raise ValidationError(f"{key} must be a string")
    return item


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read scientific recipe {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValidationError("scientific recipe must contain an object")
    return value
