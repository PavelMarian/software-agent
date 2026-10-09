from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
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
from software_bench.application_profiles import environment_image
from software_bench.validation import install_application_evaluator


def import_executable_manifest(
    manifest_file: str | Path,
    output_root: str | Path,
    *,
    task_ids: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Import domain-neutral tasks whose behavior is described by commands and assets.

    Source-specific adapters should only translate their records into this manifest shape.
    The harness and agent loop never branch on the source benchmark name.
    """
    source = Path(manifest_file).resolve()
    value = _read_object(source)
    source_name = _required_string(value, "source")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValidationError("executable manifest tasks must be a non-empty array")
    selected = set(task_ids)
    available = {
        item.get("instance_id") for item in raw_tasks if isinstance(item, Mapping)
    }
    unknown = selected - available
    if unknown:
        raise ValidationError(f"unknown executable task ids: {sorted(unknown)}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    written: list[Path] = []
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise ValidationError("each executable task must be an object")
        instance_id = _required_string(raw, "instance_id")
        if selected and instance_id not in selected:
            continue
        root = Path(output_root).resolve() / _safe_id(instance_id)
        bundle = build_executable_task_bundle(
            raw,
            bundle_root=root,
            source_root=source.parent,
            source_name=source_name,
            source_digest=digest,
        )
        write_task_bundle(bundle, root)
        written.append(root)
    return tuple(written)


def build_executable_task_bundle(
    value: Mapping[str, Any],
    *,
    bundle_root: Path,
    source_root: Path,
    source_name: str,
    source_digest: str = "",
) -> TaskBundle:
    """Materialize one normalized software-use task and copy arbitrary assets."""
    instance_id = _required_string(value, "instance_id")
    if bundle_root.exists():
        raise ValidationError(f"output TaskBundle already exists: {bundle_root}")
    workspace_source = _required_string(value, "workspace")
    workspace_path = _asset_path(source_root, workspace_source)
    evaluator_name = value.get("evaluator")
    evaluator_path = (
        _asset_path(source_root, evaluator_name)
        if isinstance(evaluator_name, str) and evaluator_name
        else None
    )
    bundle_root.mkdir(parents=True)
    _copy_asset(workspace_path, bundle_root / "workspace")
    if evaluator_path is not None:
        _copy_asset(evaluator_path, bundle_root / "evaluator")

    raw_profiles = value.get("profiles", {})
    if not isinstance(raw_profiles, Mapping):
        raise ValidationError("profiles must be an object")
    environment_profile = _profile_id(raw_profiles, "environment")
    mcp_profile = _profile_id(raw_profiles, "mcp")
    validator_profile = _profile_id(raw_profiles, "validator")
    default_checks: tuple[CheckSpec, ...] | None = None
    default_commands: tuple[TestCommand, ...] | None = None
    if validator_profile:
        validation = value.get("validation", {})
        if not isinstance(validation, Mapping):
            raise ValidationError("application validation must be an object")
        default_checks, default_commands = install_application_evaluator(
            bundle_root, validator_profile, validation
        )

    raw_workstreams = value.get("workstreams", [])
    if not isinstance(raw_workstreams, list):
        raise ValidationError("workstreams must be an array")
    workstreams = tuple(Workstream.from_dict(item) for item in raw_workstreams)
    raw_checks = value.get("checks")
    if raw_checks is None and default_checks is not None:
        checks = default_checks
    else:
        if raw_checks is None:
            raw_checks = [{"id": "task_success"}]
        if not isinstance(raw_checks, list):
            raise ValidationError("checks must be an array")
        checks = tuple(CheckSpec.from_dict(item) for item in raw_checks)
    raw_commands = value.get("test_commands")
    if raw_commands is None and default_commands is not None:
        commands = default_commands
    else:
        if raw_commands is None:
            raw_commands = [
                {
                    "id": "evaluator",
                    "command": ["python3", ".benchmark/evaluate.py"],
                    "parser": "json-status",
                }
            ]
        if not isinstance(raw_commands, list):
            raise ValidationError("test_commands must be an array")
        commands = tuple(TestCommand.from_dict(item) for item in raw_commands)

    environment_value = value.get("environment", {})
    if not isinstance(environment_value, Mapping):
        raise ValidationError("environment must be an object")
    environment = EnvironmentSpec.from_dict(
        {
            "backend_hint": environment_value.get("backend_hint", "docker"),
            "image": environment_value.get("image") or (
                environment_image(environment_profile) if environment_profile else None
            ),
            "workdir": environment_value.get("workdir", "/workspace"),
            "platform": environment_value.get("platform", "linux/x86_64"),
            "timeout_seconds": environment_value.get("timeout_seconds", 1800),
            "network_enabled": environment_value.get("network_enabled", False),
            "workspace_source": "bundle",
            "seed_path": "workspace",
            "shell_init": environment_value.get("shell_init"),
            "pass_env": environment_value.get("pass_env", []),
            "backend_config": environment_value.get("backend_config", {}),
        }
    )
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValidationError("metadata must be an object")
    task_metadata = dict(metadata)
    if mcp_profile:
        task_metadata.setdefault("mcp_profile", mcp_profile)
    if environment_profile:
        task_metadata.setdefault("environment_profile", environment_profile)
    if validator_profile:
        task_metadata.setdefault("validator_profile", validator_profile)
    task = TaskSpec(
        instance_id=instance_id,
        problem_statement=_required_string(value, "problem_statement"),
        task_kind=TaskKind.SOFTWARE_USE,
        target_software=_required_string(value, "target_software"),
        workstreams=workstreams,
        metadata=task_metadata,
    )
    raw_kind = value.get("submission_kind")
    if raw_kind is None and "submission_paths" in value:
        submission_kind = SubmissionKind.WORKSPACE_FILES
    else:
        try:
            submission_kind = SubmissionKind(raw_kind or SubmissionKind.ARTIFACT_BUNDLE)
        except ValueError as error:
            raise ValidationError(f"unsupported submission_kind: {raw_kind!r}") from error
    submission_root = value.get("submission_root", "output")
    if submission_kind == SubmissionKind.ARTIFACT_BUNDLE and not isinstance(
        submission_root, str
    ):
        raise ValidationError("submission_root must be a string")
    state_paths = value.get("state_paths", [])
    submission_paths = value.get("submission_paths", [])
    for name, items in (("state_paths", state_paths), ("submission_paths", submission_paths)):
        if not isinstance(items, list) or not all(isinstance(item, str) and item for item in items):
            raise ValidationError(f"{name} must be a string array")
    evaluation = EvaluationSpec(
        strategy=EvaluationStrategy.COMMAND_CHECKS,
        submission_kind=submission_kind,
        checks=checks,
        test_commands=commands,
        submission_root=(
            submission_root if submission_kind == SubmissionKind.ARTIFACT_BUNDLE else None
        ),
        state_paths=tuple(state_paths),
        submission_paths=tuple(submission_paths),
        evaluator_assets=(
            "evaluator" if evaluator_path is not None or validator_profile else None
        ),
        backend_hint=value.get("evaluation_backend"),
    )
    provenance = ProvenanceSpec(
        source=source_name,
        source_id=instance_id,
        metadata={"manifest_sha256": source_digest, **dict(value.get("provenance", {}))},
    )
    return TaskBundle(bundle_root, task, evaluation, environment, provenance)


def _copy_asset(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    elif source.is_file():
        destination.mkdir(parents=True)
        shutil.copy2(source, destination / source.name)
    else:
        raise ValidationError(f"asset does not exist: {source}")


def _asset_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValidationError(f"asset path escapes manifest directory: {relative}") from error
    return path


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    if not safe.strip(".-"):
        raise ValidationError(f"cannot derive output directory from instance id: {value}")
    return safe


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValidationError(f"{key} must be a non-empty string")
    return item


def _profile_id(profiles: Mapping[str, Any], key: str) -> str | None:
    value = profiles.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"profiles.{key} must be a non-empty string")
    return value.strip()


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read executable manifest {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValidationError("executable manifest must contain an object")
    return value
