from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
from pathlib import Path
from time import monotonic
from typing import Mapping, Protocol

from software_bench.core.artifacts import decode_artifact
from software_bench.core.models import (
    AdditionalOracle,
    EvaluationStrategy,
    Prediction,
    TaskBundle,
    TestCommand,
)
from software_bench.evaluation.parsers import load_log_parser
from software_bench.harness.environments import (
    DockerEnvironmentSession,
    EnvironmentSession,
    resolve_host_command,
)


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    status: str
    exit_code: int | None
    duration_seconds: float
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OracleResult:
    oracle_id: str
    layer: str
    required: bool
    status: str
    command: CommandResult

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_id": self.oracle_id,
            "layer": self.layer,
            "required": self.required,
            "status": self.status,
            "command": self.command.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationEvidence:
    patch_exists: bool
    patch_applied: bool
    test_statuses: Mapping[str, str]
    test_commands: tuple[CommandResult, ...]
    additional_oracles: tuple[OracleResult, ...]
    infrastructure_error: str | None = None
    backend_id: str = "unknown"


class EvaluationBackend(Protocol):
    backend_id: str

    def evaluate(
        self, bundle: TaskBundle, prediction: Prediction, workspace: Path | None = None
    ) -> EvaluationEvidence: ...


def load_evaluation_backend(name: str, bundle: TaskBundle) -> EvaluationBackend:
    if name == "auto":
        name = bundle.evaluation.backend_hint or bundle.environment.backend_hint
    if name in {"local", "local-fixture"}:
        return LocalCommandBackend()
    if name == "docker":
        return DockerEvaluationBackend()
    matches = entry_points(group="software_bench.evaluation_backends", name=name)
    if not matches:
        raise LookupError(
            f"evaluation backend {name!r} is not installed; expected an entry point in "
            "the 'software_bench.evaluation_backends' group"
        )
    return matches[0].load()()


class LocalCommandBackend:
    """Trusted fixture backend; it does not prepare or isolate repositories."""

    backend_id = "local-fixture"

    def evaluate(
        self, bundle: TaskBundle, prediction: Prediction, workspace: Path | None = None
    ) -> EvaluationEvidence:
        if prediction.instance_id != bundle.task.instance_id:
            raise ValueError("prediction instance_id does not match TaskBundle")
        if prediction.submission_kind != bundle.evaluation.submission_kind:
            raise ValueError("prediction submission_kind does not match TaskBundle")
        if bundle.evaluation.strategy == EvaluationStrategy.COMMAND_CHECKS:
            return _evaluate_workspace_files_local(bundle, prediction)
        if workspace is None:
            raise ValueError("local evaluation backend requires a workspace")
        statuses: dict[str, str] = {}
        command_results: list[CommandResult] = []
        infrastructure_error: str | None = None
        for spec in bundle.evaluation.test_commands:
            result = _run_test_command(spec, workspace)
            command_results.append(result)
            try:
                statuses.update(load_log_parser(spec.parser).parse(result.stdout))
            except (LookupError, ValueError, json.JSONDecodeError) as error:
                infrastructure_error = f"{type(error).__name__}: {error}"
        oracle_results = tuple(
            _run_additional_oracle(spec, workspace)
            for spec in bundle.evaluation.additional_oracles
        )
        return EvaluationEvidence(
            patch_exists=bool(prediction.model_patch),
            patch_applied=True,
            test_statuses=statuses,
            test_commands=tuple(command_results),
            additional_oracles=oracle_results,
            infrastructure_error=infrastructure_error,
            backend_id=self.backend_id,
        )


class DockerEvaluationBackend:
    """Evaluate a patch in a fresh container built from TaskBundle environment metadata."""

    backend_id = "docker"

    def evaluate(
        self, bundle: TaskBundle, prediction: Prediction, workspace: Path | None = None
    ) -> EvaluationEvidence:
        del workspace
        if prediction.instance_id != bundle.task.instance_id:
            raise ValueError("prediction instance_id does not match TaskBundle")
        if prediction.submission_kind != bundle.evaluation.submission_kind:
            raise ValueError("prediction submission_kind does not match TaskBundle")
        if bundle.evaluation.strategy == EvaluationStrategy.COMMAND_CHECKS:
            return self._evaluate_workspace_files(bundle, prediction)
        if not prediction.model_patch:
            return EvaluationEvidence(False, False, {}, (), (), backend_id=self.backend_id)

        session: DockerEnvironmentSession | None = None
        command_results: list[CommandResult] = []
        oracle_results: list[OracleResult] = []
        statuses: dict[str, str] = {}
        patch_applied = False
        infrastructure_error: str | None = None
        try:
            session = DockerEnvironmentSession(
                bundle.environment,
                base_commit=bundle.task.base_commit or "",
                bundle_root=bundle.root,
                run_id=f"eval-{bundle.task.instance_id}",
                label="evaluation",
            )
            patch_applied, detail = _apply_patch(
                session, prediction.model_patch, ".software-bench-candidate.patch"
            )
            if not patch_applied:
                infrastructure_error = f"candidate patch did not apply: {detail}"
                return EvaluationEvidence(
                    True, False, statuses, tuple(command_results), tuple(oracle_results),
                    infrastructure_error, self.backend_id,
                )
            if bundle.evaluation.test_patch:
                tests_applied, detail = _apply_patch(
                    session, bundle.evaluation.test_patch, ".software-bench-tests.patch"
                )
                if not tests_applied:
                    infrastructure_error = f"hidden test patch did not apply: {detail}"
                    return EvaluationEvidence(
                        True, True, statuses, tuple(command_results), tuple(oracle_results),
                        infrastructure_error, self.backend_id,
                    )

            for spec in bundle.evaluation.test_commands:
                result = _run_session_test_command(spec, session)
                command_results.append(result)
                try:
                    output = f"{result.stdout}\n{result.stderr}"
                    statuses.update(load_log_parser(spec.parser).parse(output))
                except (LookupError, ValueError, json.JSONDecodeError) as error:
                    infrastructure_error = f"{type(error).__name__}: {error}"
            oracle_results.extend(
                _run_session_oracle(spec, session)
                for spec in bundle.evaluation.additional_oracles
            )
        except Exception as error:
            infrastructure_error = f"{type(error).__name__}: {error}"
        finally:
            if session is not None:
                session.close()
        return EvaluationEvidence(
            True,
            patch_applied,
            statuses,
            tuple(command_results),
            tuple(oracle_results),
            infrastructure_error,
            self.backend_id,
        )

    def _evaluate_workspace_files(
        self, bundle: TaskBundle, prediction: Prediction
    ) -> EvaluationEvidence:
        artifacts = _prediction_artifacts(prediction)
        if not artifacts:
            return EvaluationEvidence(False, False, {}, (), (), backend_id=self.backend_id)
        session: DockerEnvironmentSession | None = None
        command_results: list[CommandResult] = []
        oracle_results: list[OracleResult] = []
        statuses: dict[str, str] = {}
        infrastructure_error: str | None = None
        applied = False
        try:
            session = DockerEnvironmentSession(
                bundle.environment,
                base_commit=bundle.task.base_commit or "",
                bundle_root=bundle.root,
                run_id=f"eval-{bundle.task.instance_id}",
                label="evaluation",
            )
            for name, content in artifacts.items():
                _validate_relative_path(name)
                _write_session_artifact(session, name, content)
            if bundle.evaluation.evaluator_assets:
                _copy_asset_tree_to_session(
                    bundle.root / bundle.evaluation.evaluator_assets,
                    session,
                    ".benchmark",
                )
            applied = True
            for spec in bundle.evaluation.test_commands:
                result = _run_session_test_command(spec, session)
                command_results.append(result)
                try:
                    statuses.update(load_log_parser(spec.parser).parse(result.stdout))
                except (LookupError, ValueError, json.JSONDecodeError) as error:
                    infrastructure_error = f"{type(error).__name__}: {error}"
            oracle_results.extend(
                _run_session_oracle(spec, session)
                for spec in bundle.evaluation.additional_oracles
            )
        except Exception as error:
            infrastructure_error = f"{type(error).__name__}: {error}"
        finally:
            if session is not None:
                session.close()
        return EvaluationEvidence(
            True,
            applied,
            statuses,
            tuple(command_results),
            tuple(oracle_results),
            infrastructure_error,
            self.backend_id,
        )


def _run_test_command(spec: TestCommand, workspace: Path) -> CommandResult:
    return _run_command(spec.id, spec.command, spec.timeout_seconds, workspace, shell=spec.shell)


def _evaluate_workspace_files_local(
    bundle: TaskBundle, prediction: Prediction
) -> EvaluationEvidence:
    artifacts = _prediction_artifacts(prediction)
    if not artifacts:
        return EvaluationEvidence(False, False, {}, (), (), backend_id="local-fixture")
    statuses: dict[str, str] = {}
    commands: list[CommandResult] = []
    oracles: list[OracleResult] = []
    infrastructure_error: str | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="software-bench-eval-") as temp:
            root = Path(temp)
            if bundle.environment.seed_path:
                shutil.copytree(
                    bundle.root / bundle.environment.seed_path,
                    root,
                    dirs_exist_ok=True,
                )
            for name, content in artifacts.items():
                target = root / _validate_relative_path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(decode_artifact(content, path=f"files.{name}"))
            if bundle.evaluation.evaluator_assets:
                shutil.copytree(
                    bundle.root / bundle.evaluation.evaluator_assets,
                    root / ".benchmark",
                    dirs_exist_ok=True,
                )
            for spec in bundle.evaluation.test_commands:
                result = _run_test_command(spec, root)
                commands.append(result)
                try:
                    statuses.update(load_log_parser(spec.parser).parse(result.stdout))
                except (LookupError, ValueError, json.JSONDecodeError) as error:
                    infrastructure_error = f"{type(error).__name__}: {error}"
            oracles.extend(
                _run_additional_oracle(spec, root)
                for spec in bundle.evaluation.additional_oracles
            )
    except Exception as error:
        infrastructure_error = f"{type(error).__name__}: {error}"
    return EvaluationEvidence(
        True,
        infrastructure_error is None,
        statuses,
        tuple(commands),
        tuple(oracles),
        infrastructure_error,
        "local-fixture",
    )


def _run_additional_oracle(spec: AdditionalOracle, workspace: Path) -> OracleResult:
    command = _run_command(spec.id, spec.command, spec.timeout_seconds, workspace, shell=False)
    return OracleResult(
        oracle_id=spec.id,
        layer=str(spec.layer),
        required=spec.required,
        status="passed" if command.exit_code == 0 else command.status,
        command=command,
    )


def _apply_patch(
    session: EnvironmentSession, patch: str, filename: str
) -> tuple[bool, str]:
    session.write_text(filename, patch)
    details: list[str] = []
    try:
        for options in (("--verbose",), ("--verbose", "--3way")):
            result = session.run(
                ["git", "apply", *options, filename],
                timeout_seconds=300,
            )
            details.append(result.stderr or result.stdout)
            if result.exit_code == 0:
                return True, details[-1][-4000:]
        return False, "\n".join(details)[-4000:]
    finally:
        session.run(["rm", "-f", filename], timeout_seconds=30)


def _run_session_test_command(
    spec: TestCommand, session: EnvironmentSession
) -> CommandResult:
    command = ("/bin/bash", "-lc", spec.command[0]) if spec.shell else spec.command
    return _run_session_command(spec.id, command, spec.timeout_seconds, session)


def _run_session_oracle(
    spec: AdditionalOracle, session: EnvironmentSession
) -> OracleResult:
    command = _run_session_command(spec.id, spec.command, spec.timeout_seconds, session)
    return OracleResult(
        oracle_id=spec.id,
        layer=str(spec.layer),
        required=spec.required,
        status="passed" if command.exit_code == 0 else command.status,
        command=command,
    )


def _run_session_command(
    command_id: str,
    command: tuple[str, ...],
    timeout_seconds: float,
    session: EnvironmentSession,
) -> CommandResult:
    started_at = monotonic()
    result = session.run(command, timeout_seconds=timeout_seconds)
    if result.exit_code == 0:
        status = "passed"
    elif result.exit_code == 124:
        status = "timeout"
    else:
        status = "failed"
    return CommandResult(
        command_id=command_id,
        status=status,
        exit_code=result.exit_code,
        duration_seconds=round(monotonic() - started_at, 6),
        stdout=result.stdout[-8000:],
        stderr=result.stderr[-4000:],
    )


def _run_command(
    command_id: str,
    command: tuple[str, ...],
    timeout_seconds: float,
    workspace: Path,
    *,
    shell: bool,
) -> CommandResult:
    started_at = monotonic()
    invocation: str | list[str] = command[0] if shell else resolve_host_command(command)
    try:
        completed = subprocess.run(
            invocation,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            shell=shell,
        )
        return CommandResult(
            command_id=command_id,
            status="passed" if completed.returncode == 0 else "failed",
            exit_code=completed.returncode,
            duration_seconds=round(monotonic() - started_at, 6),
            stdout=completed.stdout[-8000:],
            stderr=completed.stderr[-4000:],
        )
    except subprocess.TimeoutExpired as error:
        return CommandResult(
            command_id, "timeout", None, round(monotonic() - started_at, 6),
            _tail(error.stdout), _tail(error.stderr),
        )
    except OSError as error:
        return CommandResult(
            command_id, "error", None, round(monotonic() - started_at, 6), "", str(error)
        )


def _tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-4000:]


def _validate_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PermissionError(f"path escapes submission workspace: {value}")
    return path


def _prediction_artifacts(prediction: Prediction) -> Mapping[str, object]:
    if str(prediction.submission_kind) == "environment_state":
        return prediction.evidence
    return prediction.files


def _copy_asset_tree_to_session(
    source: Path,
    session: EnvironmentSession,
    destination: str,
) -> None:
    if not source.is_dir():
        raise ValueError(f"evaluator assets directory does not exist: {source}")
    if hasattr(session, "copy_from_host"):
        session.copy_from_host(source, destination)
        return
    # Compatibility for third-party sessions implementing the original text-only protocol.
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source).as_posix()
        session.write_text(f"{destination}/{relative}", path.read_text(encoding="utf-8"))


def _write_session_artifact(
    session: EnvironmentSession, name: str, value: object
) -> None:
    if isinstance(value, str):
        session.write_text(name, value)
        return
    content = decode_artifact(value, path=f"files.{name}")
    if not hasattr(session, "write_bytes"):
        raise ValueError("environment session does not support binary artifacts")
    session.write_bytes(name, content)
