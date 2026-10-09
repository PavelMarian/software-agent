from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from software_bench.core.artifacts import decode_artifact
from software_bench.core.config import load_mode
from software_bench.core.models import Prediction, SubmissionKind, ValidationError
from software_bench.core.task_bundle import load_task_bundle
from software_bench.evaluation import load_evaluation_backend, score
from software_bench.harness.contracts import RunRequest
from software_bench.harness.environments import create_environment_session
from software_bench.harness.execution.runner import BenchmarkRunner
from software_bench.harness.models.registry import (
    ModelAdapterSettings,
    load_framework_adapter,
    load_model_adapter,
)
from software_bench.importers import DatasetImportRequest, import_dataset
from software_bench.mcp import (
    McpEnvironmentSession,
    automatic_spec,
    builtin_profiles,
    load_spec,
    preflight_profile,
)


@dataclass(frozen=True)
class BackendResult:
    """Represent a completed backend operation.

    Attributes:
        exit_code: Process-style status code.
        output: Human-readable output for the CLI.
    """

    exit_code: int
    output: str


@dataclass(frozen=True)
class RunOptions:
    """Collect parameters required to execute one benchmark run."""

    bundle: Path
    mode: Path
    roles_dir: Path
    framework_adapter: str | None
    model_adapter: str | None
    model: str | None
    base_url: str | None
    api_key_env: str | None
    temperature: float | None
    max_output_tokens: int
    adapter_max_retries: int
    adapter_timeout_seconds: float
    environment_backend: str
    workspace: Path | None
    output: Path
    run_id: str
    seed: int
    mcp_profile: str | None
    mcp_spec: Path | None
    mcp_allow_mutations: bool


@dataclass(frozen=True)
class PreflightOptions:
    """Collect parameters required for application executable preflight."""

    bundle: Path
    mcp_profile: str
    environment_backend: str
    workspace: Path | None
    run_id: str
    output: Path | None


@dataclass(frozen=True)
class EvaluationOptions:
    """Collect parameters required to evaluate one prediction."""

    bundle: Path
    prediction: Path
    workspace: Path | None
    output: Path
    backend: str


def validate_contracts(
    bundle_path: Path | None,
    mode_path: Path | None,
    roles_dir: Path,
    require_mas_ready: bool,
) -> BackendResult:
    """Validate selected TaskBundle and mode contracts."""
    if bundle_path is None and mode_path is None:
        raise ValidationError("provide --bundle, --mode, or both")
    messages: list[str] = []
    if bundle_path is not None:
        bundle = load_task_bundle(bundle_path, require_mas_ready=require_mas_ready)
        messages.append(f"TaskBundle OK: {bundle.task.instance_id}")
    if mode_path is not None:
        mode = load_mode(mode_path, roles_dir)
        state = "supported" if mode.supported else "declared-only"
        messages.append(
            f"mode OK: {mode.id} "
            f"({mode.agent_topology}, {len(mode.roles)} role(s), {state})"
        )
    return BackendResult(0, "\n".join(messages))


def run_benchmark(options: RunOptions) -> BackendResult:
    """Execute the benchmark-owned protocol from normalized options."""
    bundle = load_task_bundle(options.bundle, require_mas_ready=True)
    mode = load_mode(options.mode, options.roles_dir)
    framework_adapter = (
        load_framework_adapter(options.framework_adapter)
        if options.framework_adapter is not None
        else None
    )
    model_adapter = None
    if framework_adapter is None:
        model_adapter = load_model_adapter(
            options.model_adapter or "mock",
            ModelAdapterSettings(
                model=options.model,
                base_url=options.base_url,
                api_key_env=options.api_key_env,
                temperature=options.temperature,
                max_output_tokens=options.max_output_tokens,
                max_retries=options.adapter_max_retries,
                timeout_seconds=options.adapter_timeout_seconds,
            ),
        )
    workspace = options.workspace.resolve() if options.workspace is not None else None
    environment = create_environment_session(
        options.environment_backend,
        bundle,
        workspace=workspace,
        run_id=options.run_id,
    )
    if options.mcp_profile or options.mcp_spec:
        try:
            spec = _select_mcp_spec(bundle, options.mcp_profile, options.mcp_spec)
            if spec is not None:
                environment = McpEnvironmentSession(
                    environment,
                    spec,
                    allow_mutations=options.mcp_allow_mutations,
                )
        except Exception:
            environment.close()
            raise
    request = RunRequest(
        run_id=options.run_id,
        task=bundle.agent_view,
        mode=mode,
        environment=environment,
        seed=options.seed,
        submission_kind=bundle.evaluation.submission_kind,
        submission_paths=bundle.evaluation.submission_paths,
        submission_root=bundle.evaluation.submission_root,
        state_paths=bundle.evaluation.state_paths,
    )
    runner = BenchmarkRunner()
    record = (
        runner.run_framework(request, framework_adapter, options.output)
        if framework_adapter is not None
        else runner.run(request, model_adapter, options.output)
    )
    status = str(record.manifest["status"])
    return BackendResult(
        0 if status == "completed" else 1,
        f"run {status}: {options.run_id} -> {options.output}",
    )


def preflight(options: PreflightOptions) -> BackendResult:
    """Check application executables in the configured task environment."""
    bundle = load_task_bundle(options.bundle)
    spec = (
        automatic_spec(bundle)
        if options.mcp_profile == "auto"
        else builtin_profiles()[options.mcp_profile]
    )
    workspace = options.workspace.resolve() if options.workspace is not None else None
    environment = create_environment_session(
        options.environment_backend,
        bundle,
        workspace=workspace,
        run_id=options.run_id,
    )
    try:
        result = dict(preflight_profile(spec, environment))
    finally:
        environment.close()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered, encoding="utf-8")
        rendered = ""
    return BackendResult(0 if result["ready"] else 1, rendered)


def evaluate(options: EvaluationOptions) -> BackendResult:
    """Evaluate a normalized prediction and write its score report."""
    bundle = load_task_bundle(options.bundle)
    prediction = load_prediction(options.prediction)
    workspace = options.workspace.resolve() if options.workspace is not None else None
    evaluator = load_evaluation_backend(options.backend, bundle)
    evidence = evaluator.evaluate(bundle, prediction, workspace)
    result = score(bundle, evidence)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metric_name = "fix_rate" if "fix_rate" in result else "score"
    message = f"resolved={result['resolved']} {metric_name}={result[metric_name]:.3f}"
    return BackendResult(0 if result["resolved"] else 1, message)


def import_tasks(request: DatasetImportRequest) -> BackendResult:
    """Import TaskBundles through the unified dataset registry."""
    paths = import_dataset(request)
    return BackendResult(0, f"imported {len(paths)} task(s) -> {request.output}")


def load_prediction(path: Path) -> Prediction:
    """Load and validate a normalized prediction document."""
    value = _read_object(path)
    required = ("instance_id", "model_name_or_path")
    if not all(isinstance(value.get(key), str) for key in required):
        raise ValidationError(
            "prediction requires string instance_id and model_name_or_path"
        )
    model_patch = value.get("model_patch", "")
    if not isinstance(model_patch, str):
        raise ValidationError("prediction.model_patch must be a string")
    try:
        submission_kind = SubmissionKind(value.get("submission_kind", "patch"))
    except ValueError as error:
        raise ValidationError("prediction has unsupported submission_kind") from error
    files = _artifact_mapping(value, "files")
    evidence = _artifact_mapping(value, "evidence")
    overlap = set(files) & set(evidence)
    if overlap:
        raise ValidationError(f"prediction files and evidence overlap: {sorted(overlap)}")
    manifest = value.get("artifact_manifest")
    if manifest is not None:
        _validate_artifact_manifest(manifest, {**files, **evidence})
    return Prediction(
        value["instance_id"],
        value["model_name_or_path"],
        model_patch,
        submission_kind,
        dict(files),
        dict(evidence),
    )


def _select_mcp_spec(bundle: Any, profile: str | None, spec_path: Path | None) -> Any:
    """Resolve an optional manual or built-in application profile."""
    if spec_path is not None:
        return load_spec(spec_path)
    if profile == "auto":
        try:
            return automatic_spec(bundle)
        except ValidationError:
            return None
    return builtin_profiles()[profile] if profile else None


def _artifact_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Read and decode one artifact mapping from a prediction."""
    artifacts = value.get(key, {})
    if not isinstance(artifacts, Mapping) or not all(
        isinstance(name, str) for name in artifacts
    ):
        raise ValidationError(f"prediction.{key} must map string paths to artifacts")
    for name, content in artifacts.items():
        decode_artifact(content, path=f"{key}.{name}")
    return artifacts


def _validate_artifact_manifest(value: Any, artifacts: Mapping[str, Any]) -> None:
    """Validate artifact sizes and hashes against a prediction manifest."""
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValidationError("prediction.artifact_manifest must be an array of objects")
    entries: dict[str, Mapping[str, Any]] = {}
    for item in value:
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("size")
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValidationError("artifact manifest entries require path, sha256, and size")
        if path in entries:
            raise ValidationError(f"duplicate artifact manifest path: {path}")
        entries[path] = item
    if set(entries) != set(artifacts):
        raise ValidationError("artifact manifest paths do not match captured artifacts")
    for path, content in artifacts.items():
        raw = decode_artifact(content, path=f"artifacts.{path}")
        if entries[path]["size"] != len(raw):
            raise ValidationError(f"artifact manifest size mismatch: {path}")
        if entries[path]["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValidationError(f"artifact manifest sha256 mismatch: {path}")


def _read_object(path: Path) -> Mapping[str, Any]:
    """Read a JSON object from a path."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must contain a JSON object")
    return value
