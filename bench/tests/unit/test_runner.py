import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from software_bench.core.config import load_mode
from software_bench.core.models import SubmissionKind
from software_bench.core.task_bundle import load_task_bundle
from software_bench.harness.contracts import ModelResponse, RunRequest
from software_bench.harness.environments import LocalEnvironmentSession
from software_bench.harness.execution.runner import BenchmarkRunner, RunRecord


ROOT = Path(__file__).parents[2]
BUNDLE = load_task_bundle(ROOT / "tests" / "fixtures" / "task_bundle")
ROLES = ROOT / "configs" / "roles"


class OverBudgetAdapter:
    adapter_id = "over-budget"

    def generate(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(done=True, input_tokens=120_001)


class ForbiddenToolAdapter:
    adapter_id = "forbidden-tool"

    def generate(
        self,
        messages: list[Mapping[str, Any]],
        system_prompt: str,
        tools: list[Mapping[str, Any]],
        *,
        role_id: str,
        seed: int,
    ) -> ModelResponse:
        return ModelResponse(
            done=True,
            tool_calls=({"name": "write", "args": {"path": "x", "content": "x"}},),
        )


class CompletedAdapter:
    adapter_id = "completed"

    def generate(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(done=True, input_tokens=1, output_tokens=1)


def request(mode_name: str, workspace: Path) -> RunRequest:
    return RunRequest(
        "contract-test",
        BUNDLE.agent_view,
        load_mode(ROOT / "configs" / "modes" / f"{mode_name}.toml", ROLES),
        LocalEnvironmentSession(workspace),
        0,
    )


def workspace(tmp_path: Path) -> Path:
    target = tmp_path / "workspace"
    shutil.copytree(BUNDLE.root / "workspace", target)
    return target


def test_runner_normalizes_aggregate_budget_interruption(tmp_path: Path) -> None:
    record = BenchmarkRunner().run(
        request("compute_matched_solo", workspace(tmp_path)), OverBudgetAdapter(), tmp_path / "run"
    )

    assert record.manifest["status"] == "budget_exhausted"
    assert record.manifest["measurement_complete"] is True
    assert record.manifest["usage"]["total_tokens"] == 120_001
    assert record.manifest["partial_submission"] is True
    assert record.manifest["partial_submission_reason"]
    assert any(event.event_type == "model_response" for event in record.trace)
    assert record.trace[-1].event_type == "run_finished"


def test_benchmark_owned_tools_reject_role_permission_violation(tmp_path: Path) -> None:
    record = BenchmarkRunner().run(
        request("full_mas", workspace(tmp_path)), ForbiddenToolAdapter(), tmp_path / "run"
    )

    assert record.manifest["status"] == "adapter_error"
    assert record.manifest["measurement_complete"] is False
    assert record.manifest["usage"]["tool_calls"] == 1
    assert any(event.event_type == "tool_called" for event in record.trace)
    assert "PermissionError" in record.manifest["stop_reason"]


def test_runner_captures_environment_state_as_evidence(tmp_path: Path) -> None:
    root = workspace(tmp_path)
    (root / "submissions").mkdir()
    (root / "submissions/diagnosis.json").write_text("{}\n", encoding="utf-8")
    (root / "results.json").write_text('{"resolved": true}\n', encoding="utf-8")
    run_request = RunRequest(
        "state-contract",
        BUNDLE.agent_view,
        load_mode(ROOT / "configs/modes/compute_matched_solo.toml", ROLES),
        LocalEnvironmentSession(root),
        0,
        submission_kind=SubmissionKind.ENVIRONMENT_STATE,
        state_paths=("submissions", "results.json"),
    )

    record = BenchmarkRunner().run(run_request, CompletedAdapter(), tmp_path / "state-run")
    prediction = json.loads(
        (tmp_path / "state-run/prediction.json").read_text(encoding="utf-8")
    )

    assert record.files == {}
    assert set(record.evidence) == {"submissions/diagnosis.json", "results.json"}
    assert prediction["files"] == {}
    assert set(prediction["evidence"]) == {
        "submissions/diagnosis.json",
        "results.json",
    }


def test_runner_materializes_named_stage_artifacts(tmp_path: Path) -> None:
    record = RunRecord(
        manifest={
            "instance_id": "case", "adapter_id": "model",
            "submission_kind": "artifact_bundle",
        },
        trace=(),
        patch="",
        files={},
        evidence={},
        stage_artifacts={
            "initial_plan": {"payload": {"steps": ["inspect"]}},
            "research_brief": {"payload": {"unknowns": ["format"]}},
            "research_evidence": {"payload": {"searches": [{"query": "format"}]}},
            "research_report": {"payload": {"methods": ["supported format"]}},
            "execution_plan": {"payload": {"steps": ["implement"]}},
            "updated_plan": {"payload": {"steps": ["implement"]}},
        },
    )

    BenchmarkRunner._write(record, tmp_path)

    assert json.loads((tmp_path / "stage_artifacts/research_brief.json").read_text())[
        "payload"
    ]["unknowns"] == ["format"]
    assert json.loads((tmp_path / "stage_artifacts/plan.json").read_text())[
        "payload"
    ]["steps"] == ["implement"]
    assert (tmp_path / "stage_artifacts/initial_plan.json").is_file()
    assert (tmp_path / "stage_artifacts/research_evidence.json").is_file()
    assert (tmp_path / "stage_artifacts/research_report.json").is_file()
    assert (tmp_path / "stage_artifacts/updated_plan.json").is_file()
