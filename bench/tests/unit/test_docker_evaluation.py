from pathlib import Path

from software_bench.core.models import Prediction
from software_bench.core.task_bundle import load_task_bundle
from software_bench.evaluation import backend as backend_module
from software_bench.evaluation.backend import DockerEvaluationBackend
from software_bench.harness.environments import ExecutionResult


BUNDLE = load_task_bundle(Path(__file__).parents[1] / "fixtures" / "task_bundle")


class FakeDockerSession:
    instances = []

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.writes: dict[str, str] = {}
        self.closed = False
        self.instances.append(self)

    def write_text(self, path: str, content: str) -> None:
        self.writes[path] = content

    def run(self, command, *, timeout_seconds: float) -> ExecutionResult:
        del timeout_seconds
        if command[:2] == ("python", "-c") or command[:2] == ["python", "-c"]:
            if "json.dumps" in command[2]:
                return ExecutionResult(
                    '{"feature_contract":"PASSED","regression_contract":"PASSED"}\n'
                )
        if command[:2] == ("python", ".benchmark/evaluate.py") or command[:2] == [
            "python", ".benchmark/evaluate.py"
        ]:
            return ExecutionResult(
                '{"case_structure":"PASSED","solver_execution":"PASSED",'
                '"numerical_accuracy":"PASSED"}\n'
            )
        return ExecutionResult()

    def close(self) -> None:
        self.closed = True


def test_docker_backend_uses_fresh_session_and_hidden_test_patch(monkeypatch) -> None:
    FakeDockerSession.instances.clear()
    monkeypatch.setattr(backend_module, "DockerEnvironmentSession", FakeDockerSession)
    prediction = Prediction(BUNDLE.task.instance_id, "model", "candidate patch")

    evidence = DockerEvaluationBackend().evaluate(BUNDLE, prediction)

    session = FakeDockerSession.instances[0]
    assert evidence.patch_applied is True
    assert evidence.backend_id == "docker"
    assert evidence.infrastructure_error is None
    assert evidence.test_statuses["feature_contract"] == "PASSED"
    assert session.writes[".software-bench-candidate.patch"] == "candidate patch"
    assert session.writes[".software-bench-tests.patch"] == BUNDLE.evaluation.test_patch
    assert session.closed is True


def test_docker_backend_reports_missing_image_as_infrastructure_error() -> None:
    prediction = Prediction(BUNDLE.task.instance_id, "model", "candidate patch")

    evidence = DockerEvaluationBackend().evaluate(BUNDLE, prediction)

    assert evidence.patch_applied is False
    assert evidence.backend_id == "docker"
    assert "environment.image" in (evidence.infrastructure_error or "")


def test_docker_backend_materializes_workspace_submission_and_hidden_assets(
    monkeypatch,
) -> None:
    bundle = load_task_bundle(
        Path(__file__).parents[1] / "fixtures" / "software_use_bundle"
    )
    FakeDockerSession.instances.clear()
    monkeypatch.setattr(backend_module, "DockerEnvironmentSession", FakeDockerSession)
    prediction = Prediction(
        bundle.task.instance_id,
        "model",
        "",
        bundle.evaluation.submission_kind,
        {
            "system/controlDict": "application pisoFoam;",
            "constant/physicalProperties": "nu 1e-05;",
            "Allrun": "pisoFoam",
        },
    )

    evidence = DockerEvaluationBackend().evaluate(bundle, prediction)

    session = FakeDockerSession.instances[0]
    assert evidence.patch_applied is True
    assert evidence.test_statuses["numerical_accuracy"] == "PASSED"
    assert session.writes["system/controlDict"] == "application pisoFoam;"
    assert ".benchmark/evaluate.py" in session.writes
