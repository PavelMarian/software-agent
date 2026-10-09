import json
import shutil
from pathlib import Path

from software_bench.core.config import load_mode
from software_bench.core.models import SubmissionKind
from software_bench.core.task_bundle import load_task_bundle
from software_bench.harness.contracts import FrameworkOutcome, RunRequest
from software_bench.harness.environments import LocalEnvironmentSession
from software_bench.harness.execution.runner import BenchmarkRunner


ROOT = Path(__file__).parents[2]


class ArtifactFramework:
    adapter_id = "artifact-framework"

    def run(self, request, observer):
        observer.record("framework_step", "solo", {"step": "solve"}, tool_calls=1)
        return FrameworkOutcome(
            files={"result.txt": "framework result\n"},
            submission_kind=SubmissionKind.ARTIFACT_BUNDLE,
        )


def test_framework_runner_is_labeled_and_uses_normalized_prediction(
    tmp_path: Path,
) -> None:
    fixture = ROOT / "tests/fixtures/software_use_bundle"
    bundle = load_task_bundle(fixture)
    workspace = tmp_path / "workspace"
    shutil.copytree(fixture / "workspace", workspace)
    request = RunRequest(
        "framework-run",
        bundle.agent_view,
        load_mode(
            ROOT / "configs/modes/unrestricted_solo.toml",
            ROOT / "configs/roles",
        ),
        LocalEnvironmentSession(workspace),
        0,
        submission_kind=SubmissionKind.ARTIFACT_BUNDLE,
        submission_root="output",
    )

    record = BenchmarkRunner().run_framework(
        request, ArtifactFramework(), tmp_path / "run"
    )

    prediction = json.loads((tmp_path / "run/prediction.json").read_text())
    assert record.manifest["adapter_contract"] == "framework"
    assert record.manifest["usage"]["tool_calls"] == 1
    assert prediction["files"] == {"result.txt": "framework result\n"}
