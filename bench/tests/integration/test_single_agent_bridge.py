from __future__ import annotations

import json
from pathlib import Path

from software_bench.cli import main
from software_bench.core.config import load_mode
from software_bench.core.task_bundle import load_task_bundle
from software_bench.harness.contracts import RunRequest
from software_bench.harness.agents.loop import task_prompt


ROOT = Path(__file__).parents[2]
MODE = ROOT / "configs/modes/unrestricted_solo.toml"
ROLES = ROOT / "configs/roles"


def test_single_agent_runtime_reaches_normalized_prediction_boundary(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    workspace = assets / "workspace"
    evaluator = assets / "evaluator"
    workspace.mkdir(parents=True)
    evaluator.mkdir()
    (workspace / "README.md").write_text("Create the requested artifact.\n", encoding="utf-8")
    (evaluator / "evaluate.py").write_text(
        "import json\nprint(json.dumps({'artifact': 'PASSED'}))\n",
        encoding="utf-8",
    )
    manifest = assets / "tasks.json"
    manifest.write_text(
        json.dumps(
            {
                "source": "contract-fixture",
                "tasks": [
                    {
                        "instance_id": "baseline-smoke",
                        "problem_statement": "Create a small verified text artifact.",
                        "target_software": "the task workspace",
                        "workspace": "workspace",
                        "evaluator": "evaluator",
                        "workstreams": [
                            {
                                "id": "implementation",
                                "title": "Implementation",
                                "description": "Create the artifact.",
                            },
                            {
                                "id": "verification",
                                "title": "Verification",
                                "description": "Check the result.",
                            },
                        ],
                        "checks": [{"id": "artifact"}],
                        "test_commands": [
                            {
                                "id": "fixture-evaluator",
                                "command": ["python3", ".benchmark/evaluate.py"],
                                "parser": "json-status",
                            }
                        ],
                        "submission_kind": "artifact_bundle",
                        "submission_root": "output",
                        "environment": {
                            "backend_hint": "docker",
                            "image": "must-not-be-used:test",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tasks = tmp_path / "tasks"
    assert main(
        [
            "import-dataset",
            "--source", "executable-manifest",
            "--recipe", str(manifest),
            "--output", str(tasks),
        ]
    ) == 0
    bundle = tasks / "baseline-smoke"
    run = tmp_path / "run"

    assert main(
        [
            "run",
            "--bundle",
            str(bundle),
            "--mode",
            str(MODE),
            "--roles-dir",
            str(ROLES),
            "--model-adapter",
            "mock",
            "--no-docker",
            "--output",
            str(run),
            "--run-id",
            "single-agent-contract",
        ]
    ) == 0

    run_manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    prediction = json.loads((run / "prediction.json").read_text(encoding="utf-8"))
    assert run_manifest["agent_topology"] == "single_agent"
    assert run_manifest["agent_runtime"] == {
        "id": "software_multiagent",
        "version": "0.1.0",
    }
    assert run_manifest["agent_run_count"] == 1
    assert run_manifest["environment_backend"] == "local"
    assert run_manifest["usage"]["total_tokens"] == 15
    assert run_manifest["usage"]["tool_calls"] == 2
    assert prediction["submission_kind"] == "artifact_bundle"
    assert prediction["files"]["mock_solution.txt"] == "fixture solution\n"
    assert prediction["evidence"] == {}
    assert prediction["artifact_manifest"] == [
        {
            "path": "mock_solution.txt",
            "sha256": "23c16ba025e0344740c3bd5b3a0b9b6efce762d37dadf63982c21d38ca7d4667",
            "size": 17,
        }
    ]
    assert any(
        json.loads(line)["event_type"] == "software_multiagent.node_started"
        for line in (run / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    )
    loaded = load_task_bundle(bundle)
    request = RunRequest(
        run_id="prompt-contract",
        task=loaded.agent_view,
        mode=load_mode(MODE, ROLES),
        environment=object(),  # type: ignore[arg-type]
        seed=0,
        submission_kind=loaded.evaluation.submission_kind,
        submission_paths=loaded.evaluation.submission_paths,
        submission_root=loaded.evaluation.submission_root,
        state_paths=loaded.evaluation.state_paths,
    )
    assert "workspace-root directory output/" in task_prompt(request, "solo")

    score = run / "score.json"
    assert main(
        [
            "evaluate",
            "--bundle",
            str(bundle),
            "--prediction",
            str(run / "prediction.json"),
            "--no-docker",
            "--output",
            str(score),
        ]
    ) == 0
    assert json.loads(score.read_text(encoding="utf-8"))["resolved"] is True
