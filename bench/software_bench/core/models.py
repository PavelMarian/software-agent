from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class ValidationError(ValueError):
    """Raised when benchmark data violates a core contract."""


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class BenchmarkMode(StringEnum):
    UNRESTRICTED_SOLO = "unrestricted_solo"
    COMPUTE_MATCHED_SOLO = "compute_matched_solo"
    INDEPENDENT_ENSEMBLE = "independent_ensemble"
    FULL_MAS = "full_mas"
    MAS_NO_PLANNER = "mas_no_planner"
    MAS_NO_VERIFIER = "mas_no_verifier"


class AgentTopology(StringEnum):
    """Native harness topology selected by a benchmark mode."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    INDEPENDENT_ENSEMBLE = "independent_ensemble"


class TaskKind(StringEnum):
    SOFTWARE_EVOLUTION = "software_evolution"
    SOFTWARE_USE = "software_use"


class SubmissionKind(StringEnum):
    PATCH = "patch"
    ARTIFACT_BUNDLE = "artifact_bundle"
    ENVIRONMENT_STATE = "environment_state"
    # Compatibility with TaskBundles produced before schema 0.5.
    WORKSPACE_FILES = "workspace_files"


class EvaluationStrategy(StringEnum):
    SWE_PATCH = "swe_patch"
    COMMAND_CHECKS = "command_checks"


class OracleLayer(StringEnum):
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"


class WorkspaceAccess(StringEnum):
    NONE = "none"
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class Workstream:
    id: str
    title: str
    description: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Workstream:
        _reject_unknown(value, {"id", "title", "description"}, "workstream")
        return cls(
            id=_required_string(value, "id"),
            title=_required_string(value, "title"),
            description=_required_string(value, "description"),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Public task information that may be shown to every participant."""

    instance_id: str
    problem_statement: str
    task_kind: TaskKind = TaskKind.SOFTWARE_EVOLUTION
    target_software: str | None = None
    repo: str | None = None
    base_commit: str | None = None
    start_version: str | None = None
    end_version: str | None = None
    workstreams: tuple[Workstream, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskSpec:
        _reject_unknown(
            value,
            {
                "instance_id", "repo", "base_commit", "start_version", "end_version",
                "problem_statement", "task_kind", "target_software", "workstreams", "metadata",
            },
            "task",
        )
        streams = _object_array(value, "workstreams", minimum=0, default=[])
        try:
            task_kind = TaskKind(value.get("task_kind", TaskKind.SOFTWARE_EVOLUTION))
        except (TypeError, ValueError) as error:
            raise ValidationError(f"unsupported task_kind: {value.get('task_kind')!r}") from error
        task = cls(
            instance_id=_required_string(value, "instance_id"),
            problem_statement=_required_string(value, "problem_statement"),
            task_kind=task_kind,
            target_software=_optional_string(value, "target_software"),
            repo=_optional_string(value, "repo"),
            base_commit=_optional_string(value, "base_commit"),
            start_version=_optional_string(value, "start_version"),
            end_version=_optional_string(value, "end_version"),
            workstreams=tuple(Workstream.from_dict(item) for item in streams),
            metadata=_mapping(value.get("metadata", {}), "metadata"),
        )
        if task.task_kind == TaskKind.SOFTWARE_EVOLUTION:
            required = {
                "repo": task.repo,
                "base_commit": task.base_commit,
                "start_version": task.start_version,
                "end_version": task.end_version,
            }
            missing = [name for name, item in required.items() if item is None]
            if missing:
                raise ValidationError(
                    "software_evolution tasks require " + ", ".join(missing)
                )
            if task.start_version == task.end_version:
                raise ValidationError("start_version and end_version must differ")
        elif not task.target_software:
            raise ValidationError("software_use tasks require target_software")
        ids = [stream.id for stream in task.workstreams]
        if len(ids) != len(set(ids)):
            raise ValidationError("workstream ids must be unique")
        return task

    def validate_mas_ready(self) -> None:
        if len(self.workstreams) < 2:
            raise ValidationError("MAS-ready tasks require at least two workstreams")


@dataclass(frozen=True)
class AgentTaskView:
    """Explicit allow-list of fields supplied to an agent or framework."""

    instance_id: str
    problem_statement: str
    task_kind: TaskKind
    target_software: str | None
    repo: str | None
    base_commit: str | None
    start_version: str | None
    end_version: str | None
    workstreams: tuple[Workstream, ...]

    @classmethod
    def from_task(cls, task: TaskSpec) -> AgentTaskView:
        return cls(
            instance_id=task.instance_id,
            problem_statement=task.problem_statement,
            task_kind=task.task_kind,
            target_software=task.target_software,
            repo=task.repo,
            base_commit=task.base_commit,
            start_version=task.start_version,
            end_version=task.end_version,
            workstreams=task.workstreams,
        )


@dataclass(frozen=True)
class CheckSpec:
    id: str
    required: bool = True
    weight: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckSpec:
        _reject_unknown(value, {"id", "required", "weight"}, "check")
        return cls(
            id=_required_string(value, "id"),
            required=_boolean(value.get("required", True), "required"),
            weight=_positive_number(value.get("weight", 1.0), "weight"),
        )


@dataclass(frozen=True)
class TestCommand:
    id: str
    command: tuple[str, ...]
    parser: str
    timeout_seconds: float = 1800.0
    shell: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TestCommand:
        _reject_unknown(
            value, {"id", "command", "parser", "timeout_seconds", "shell"}, "test command"
        )
        return cls(
            id=_required_string(value, "id"),
            command=_command(value, "command"),
            parser=_required_string(value, "parser"),
            timeout_seconds=_positive_number(value.get("timeout_seconds", 1800), "timeout_seconds"),
            shell=_boolean(value.get("shell", False), "shell"),
        )


@dataclass(frozen=True)
class AdditionalOracle:
    id: str
    layer: OracleLayer
    command: tuple[str, ...]
    timeout_seconds: float = 300.0
    required: bool = True
    weight: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdditionalOracle:
        _reject_unknown(
            value,
            {"id", "layer", "command", "timeout_seconds", "required", "weight"},
            "additional oracle",
        )
        try:
            layer = OracleLayer(_required_string(value, "layer"))
        except (TypeError, ValueError) as error:
            raise ValidationError(
                f"unsupported additional oracle layer: {value.get('layer')!r}"
            ) from error
        return cls(
            id=_required_string(value, "id"),
            layer=layer,
            command=_command(value, "command"),
            timeout_seconds=_positive_number(value.get("timeout_seconds", 300), "timeout_seconds"),
            required=_boolean(value.get("required", True), "required"),
            weight=_positive_number(value.get("weight", 1.0), "weight"),
        )


@dataclass(frozen=True)
class EvaluationSpec:
    """Hidden test contract; never included in AgentTaskView."""

    strategy: EvaluationStrategy = EvaluationStrategy.SWE_PATCH
    submission_kind: SubmissionKind = SubmissionKind.PATCH
    test_patch: str = ""
    fail_to_pass: tuple[str, ...] = ()
    pass_to_pass: tuple[str, ...] = ()
    checks: tuple[CheckSpec, ...] = ()
    test_commands: tuple[TestCommand, ...] = ()
    additional_oracles: tuple[AdditionalOracle, ...] = ()
    submission_root: str | None = None
    state_paths: tuple[str, ...] = ()
    submission_paths: tuple[str, ...] = ()
    evaluator_assets: str | None = None
    backend_hint: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationSpec:
        _reject_unknown(
            value,
            {
                "strategy", "submission_kind", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
                "checks", "test_commands", "additional_oracles", "submission_paths",
                "submission_root", "state_paths",
                "evaluator_assets",
                "backend_hint",
            },
            "evaluation",
        )
        try:
            strategy = EvaluationStrategy(value.get("strategy", EvaluationStrategy.SWE_PATCH))
            submission_kind = SubmissionKind(
                value.get("submission_kind", SubmissionKind.PATCH)
            )
        except ValueError as error:
            raise ValidationError("unsupported evaluation strategy or submission kind") from error
        commands = _object_array(value, "test_commands", minimum=1)
        extra = _object_array(value, "additional_oracles", minimum=0, default=[])
        checks = _object_array(value, "checks", minimum=0, default=[])
        spec = cls(
            strategy=strategy,
            submission_kind=submission_kind,
            test_patch=_string_or_default(value, "test_patch"),
            fail_to_pass=_string_array(value, "FAIL_TO_PASS", minimum=0),
            pass_to_pass=_string_array(value, "PASS_TO_PASS", minimum=0),
            checks=tuple(CheckSpec.from_dict(item) for item in checks),
            test_commands=tuple(TestCommand.from_dict(item) for item in commands),
            additional_oracles=tuple(AdditionalOracle.from_dict(item) for item in extra),
            submission_root=_optional_string(value, "submission_root"),
            state_paths=_string_array(value, "state_paths", minimum=0, default=[]),
            submission_paths=_string_array(
                value, "submission_paths", minimum=0, default=[]
            ),
            evaluator_assets=_optional_string(value, "evaluator_assets"),
            backend_hint=_optional_string(value, "backend_hint"),
        )
        if spec.strategy == EvaluationStrategy.SWE_PATCH:
            if spec.submission_kind != SubmissionKind.PATCH:
                raise ValidationError("swe_patch evaluation requires patch submission")
            if not spec.fail_to_pass:
                raise ValidationError("swe_patch evaluation requires FAIL_TO_PASS")
        else:
            if spec.submission_kind not in {
                SubmissionKind.ARTIFACT_BUNDLE,
                SubmissionKind.ENVIRONMENT_STATE,
                SubmissionKind.WORKSPACE_FILES,
            }:
                raise ValidationError(
                    "command_checks evaluation requires artifact_bundle or environment_state"
                )
            if not spec.checks:
                raise ValidationError("command_checks evaluation requires checks")
            if (
                spec.submission_kind == SubmissionKind.ARTIFACT_BUNDLE
                and not spec.submission_root
            ):
                raise ValidationError("artifact_bundle submission requires submission_root")
            if (
                spec.submission_kind == SubmissionKind.ENVIRONMENT_STATE
                and not spec.state_paths
            ):
                raise ValidationError("environment_state submission requires state_paths")
            if spec.submission_kind == SubmissionKind.WORKSPACE_FILES and not spec.submission_paths:
                raise ValidationError("workspace_files submission requires submission_paths")
        if spec.submission_root:
            _relative_path(spec.submission_root, "submission_root")
        for path in spec.state_paths:
            _relative_path(path, "state path")
        for path in spec.submission_paths:
            _relative_path(path, "submission path")
        if spec.evaluator_assets:
            _relative_path(spec.evaluator_assets, "evaluator_assets")
        ids = [command.id for command in spec.test_commands]
        if len(ids) != len(set(ids)):
            raise ValidationError("test command ids must be unique")
        oracle_ids = [oracle.id for oracle in spec.additional_oracles]
        if len(oracle_ids) != len(set(oracle_ids)):
            raise ValidationError("additional oracle ids must be unique")
        overlap = set(spec.fail_to_pass) & set(spec.pass_to_pass)
        if overlap:
            raise ValidationError("FAIL_TO_PASS and PASS_TO_PASS must be disjoint")
        check_ids = [check.id for check in spec.checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValidationError("check ids must be unique")
        return spec


@dataclass(frozen=True)
class EnvironmentSpec:
    backend_hint: str = "docker"
    image: str | None = None
    workdir: str = "/testbed"
    platform: str = "linux/x86_64"
    timeout_seconds: float = 1800.0
    network_enabled: bool = False
    workspace_source: str = "git"
    seed_path: str | None = None
    shell_init: str | None = None
    pass_env: tuple[str, ...] = ()
    backend_config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EnvironmentSpec:
        _reject_unknown(
            value,
            {
                "backend_hint", "image", "workdir", "platform", "timeout_seconds",
                "network_enabled", "workspace_source", "seed_path", "shell_init",
                "pass_env", "backend_config",
            },
            "environment",
        )
        workspace_source = value.get("workspace_source", "git")
        if workspace_source not in {"git", "bundle"}:
            raise ValidationError("workspace_source must be git or bundle")
        spec = cls(
            backend_hint=_required_string(value, "backend_hint"),
            image=_optional_string(value, "image"),
            workdir=_required_string(value, "workdir"),
            platform=_required_string(value, "platform"),
            timeout_seconds=_positive_number(value.get("timeout_seconds", 1800), "timeout_seconds"),
            network_enabled=_boolean(value.get("network_enabled", False), "network_enabled"),
            workspace_source=workspace_source,
            seed_path=_optional_string(value, "seed_path"),
            shell_init=_optional_string(value, "shell_init"),
            pass_env=_string_array(value, "pass_env", minimum=0, default=[]),
            backend_config=_mapping(value.get("backend_config", {}), "backend_config"),
        )
        if spec.workspace_source == "bundle" and spec.seed_path is None:
            raise ValidationError("bundle workspace_source requires seed_path")
        if spec.seed_path:
            _relative_path(spec.seed_path, "seed_path")
        return spec


@dataclass(frozen=True)
class ProvenanceSpec:
    """Private reference solution and source metadata."""

    source: str
    source_id: str | None = None
    end_version_commit: str = ""
    gold_patch: str = ""
    prs: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProvenanceSpec:
        _reject_unknown(
            value,
            {"source", "source_id", "end_version_commit", "gold_patch", "prs", "metadata"},
            "provenance",
        )
        prs = _object_array(value, "prs", minimum=0, default=[])
        return cls(
            source=_required_string(value, "source"),
            source_id=_optional_string(value, "source_id"),
            end_version_commit=_string_or_default(value, "end_version_commit"),
            gold_patch=_string_or_default(value, "gold_patch"),
            prs=tuple(dict(item) for item in prs),
            metadata=_mapping(value.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class TaskBundle:
    root: Path
    task: TaskSpec
    evaluation: EvaluationSpec
    environment: EnvironmentSpec
    provenance: ProvenanceSpec

    @property
    def agent_view(self) -> AgentTaskView:
        return AgentTaskView.from_task(self.task)


@dataclass(frozen=True)
class Prediction:
    instance_id: str
    model_name_or_path: str
    model_patch: str
    submission_kind: SubmissionKind = SubmissionKind.PATCH
    files: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Budget:
    token_limit: int
    wall_time_seconds: float
    tool_call_limit: int

    def __post_init__(self) -> None:
        if self.token_limit <= 0 or self.wall_time_seconds <= 0 or self.tool_call_limit <= 0:
            raise ValidationError("all budget limits must be positive")


@dataclass(frozen=True)
class RoleSpec:
    id: str
    profile: str
    instructions: str
    workspace_access: WorkspaceAccess
    tools: tuple[str, ...]
    can_communicate: bool


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValidationError(f"{key} must be a non-empty string")
    return item


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValidationError(f"{key} must be a string")
    return item


def _string_or_default(value: Mapping[str, Any], key: str, default: str = "") -> str:
    item = value.get(key, default)
    if not isinstance(item, str):
        raise ValidationError(f"{key} must be a string")
    return item


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        raise ValidationError(f"{key} must be null or a non-empty string")
    return item


def _string_array(
    value: Mapping[str, Any],
    key: str,
    minimum: int,
    default: list[str] | None = None,
) -> tuple[str, ...]:
    items = value.get(key, default)
    if not isinstance(items, list) or len(items) < minimum:
        raise ValidationError(f"{key} must contain at least {minimum} items")
    if not all(isinstance(item, str) and item for item in items):
        raise ValidationError(f"{key} must contain only non-empty strings")
    if len(items) != len(set(items)):
        raise ValidationError(f"{key} items must be unique")
    return tuple(items)


def _object_array(
    value: Mapping[str, Any], key: str, minimum: int, default: list[Any] | None = None
) -> list[Mapping[str, Any]]:
    items = value.get(key, default)
    if not isinstance(items, list) or len(items) < minimum:
        raise ValidationError(f"{key} must contain at least {minimum} items")
    if not all(isinstance(item, Mapping) for item in items):
        raise ValidationError(f"{key} must contain only objects")
    return items


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{key} must be an object")
    return value


def _command(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    parts = value.get(key)
    if (
        not isinstance(parts, list)
        or not parts
        or not all(isinstance(part, str) and part for part in parts)
    ):
        raise ValidationError(f"{key} must be a non-empty string array")
    return tuple(parts)


def _positive_number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValidationError(f"{key} must be positive")
    return float(value)


def _boolean(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{key} must be boolean")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValidationError(f"unknown {label} fields: {', '.join(sorted(unknown))}")


def _relative_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValidationError(f"{label} must be a safe relative path: {value}")
    return path
