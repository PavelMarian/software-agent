import json
import shutil
from pathlib import Path

from software_bench.cli import main


ROOT = Path(__file__).parents[2]
BUNDLE = ROOT / "tests" / "fixtures" / "task_bundle"
MODE = ROOT / "configs" / "modes" / "full_mas.toml"
ROLES = ROOT / "configs" / "roles"


def test_taskbundle_run_and_evaluation_produce_normalized_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    shutil.copytree(BUNDLE / "workspace", workspace)
    run_dir = tmp_path / "run"
    assert main(
        [
            "run",
            "--bundle", str(BUNDLE),
            "--mode", str(MODE),
            "--roles-dir", str(ROLES),
            "--model-adapter", "mock",
            "--workspace", str(workspace),
            "--output", str(run_dir),
            "--run-id", "integration-test",
            "--seed", "7",
        ]
    ) == 0

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["adapter_contract"] == "model"
    assert manifest["environment_backend"] == "local"
    assert manifest["mode"] == "full_mas"
    assert manifest["agent_topology"] == "multi_agent"
    assert manifest["agent_run_count"] == 3
    assert manifest["usage"]["total_tokens"] == 45
    assert manifest["usage"]["tool_calls"] == 7
    assert manifest["trace_metrics"]["active_role_count"] == 3
    assert manifest["trace_metrics"]["coordination"]["messages_sent"] == 1
    assert manifest["trace_metrics"]["coordination"]["message_delivery_rate"] == 1.0
    assert (run_dir / "prediction.json").is_file()

    result_path = run_dir / "result.json"
    assert main(
        [
            "evaluate",
            "--bundle", str(BUNDLE),
            "--prediction", str(run_dir / "prediction.json"),
            "--workspace", str(workspace),
            "--output", str(result_path),
        ]
    ) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["evaluation_backend"] == "local-fixture"
    assert result["resolved"] is True
    assert result["fix_rate"] == 1.0


def test_validate_accepts_mas_ready_bundle_and_mode() -> None:
    assert main(
        [
            "validate",
            "--bundle", str(BUNDLE),
            "--mas-ready",
            "--mode", str(MODE),
            "--roles-dir", str(ROLES),
        ]
    ) == 0
