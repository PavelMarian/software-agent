from __future__ import annotations

import json
from pathlib import Path

from software_bench.core.task_bundle import load_task_bundle
from software_bench.importers.datasets.database import import_spider2_dataset
from software_bench.importers.datasets.live_service import import_sregym
from software_bench.importers.datasets.workbook import import_spreadsheetbench_dataset


def test_spreadsheetbench_importer_keeps_answers_hidden(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "42",
                    "instruction": "Fill the requested total.",
                    "instruction_type": "formula",
                    "answer_position": "Sheet1!B2",
                }
            ]
        ),
        encoding="utf-8",
    )
    case = tmp_path / "books" / "42"
    case.mkdir(parents=True)
    (case / "book_input.xlsx").write_bytes(b"input-binary")
    (case / "book_answer.xlsx").write_bytes(b"answer-binary")

    paths = import_spreadsheetbench_dataset(dataset, tmp_path / "books", tmp_path / "out")

    bundle = load_task_bundle(paths[0], require_mas_ready=True)
    assert str(bundle.evaluation.submission_kind) == "artifact_bundle"
    assert bundle.evaluation.submission_root == "output"
    assert (paths[0] / "workspace/inputs/book_input.xlsx").read_bytes() == b"input-binary"
    assert not (paths[0] / "workspace/inputs/book_answer.xlsx").exists()
    assert (paths[0] / "evaluator/gold/book_answer.xlsx").read_bytes() == b"answer-binary"


def test_spider2_lite_importer_uses_common_workspace_submission(tmp_path: Path) -> None:
    upstream = tmp_path / "Spider2"
    variant = upstream / "spider2-lite"
    suite = variant / "evaluation_suite"
    gold = suite / "gold"
    gold.mkdir(parents=True)
    (suite / "evaluate.py").write_text("print('official')\n", encoding="utf-8")
    (suite / "evaluate_utils.py").write_text("", encoding="utf-8")
    (gold / "spider2lite_eval.jsonl").write_text(
        json.dumps({"instance_id": "local001", "evaluation": {}}) + "\n",
        encoding="utf-8",
    )
    (gold / "exec_result").mkdir()
    (gold / "exec_result/local001.csv").write_text("x\n1\n", encoding="utf-8")
    (variant / "resource/databases/db1").mkdir(parents=True)
    (variant / "resource/databases/db1/database.sqlite").write_bytes(b"sqlite")
    dataset = variant / "spider2-lite.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "instance_id": "local001",
                "db": "db1",
                "question": "Return the requested rows.",
                "external_knowledge": "none",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paths = import_spider2_dataset(
        dataset, upstream, tmp_path / "out", variant="lite"
    )

    bundle = load_task_bundle(paths[0], require_mas_ready=True)
    assert bundle.task.instance_id == "spider2__lite__local001"
    assert str(bundle.evaluation.submission_kind) == "artifact_bundle"
    assert bundle.evaluation.submission_root == "output"
    assert (paths[0] / "workspace/output/answer.sql").is_file()
    assert (paths[0] / "evaluator/evaluation_suite/gold/exec_result/local001.csv").is_file()


def test_sregym_importer_declares_live_tools_without_a_new_task_shape(tmp_path: Path) -> None:
    upstream = tmp_path / "SREGym"
    source = upstream / "sregym/conductor/problem_sets.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "SREGYM_LITE_PROBLEMS = ('network_policy_block',)\n",
        encoding="utf-8",
    )

    paths = import_sregym(upstream, tmp_path / "out")

    bundle = load_task_bundle(paths[0], require_mas_ready=True)
    assert bundle.environment.backend_hint == "managed_process"
    assert bundle.evaluation.backend_hint == "local"
    tools = bundle.environment.backend_config["http_tools"]["tools"]
    assert [item["name"] for item in tools] == [
        "get_sregym_context",
        "submit_sregym",
        "collect_sregym_result",
    ]
    assert str(bundle.evaluation.submission_kind) == "environment_state"
    assert bundle.evaluation.state_paths == ("submissions", "results.json")
