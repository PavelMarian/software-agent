from __future__ import annotations

import json

from foambench.corpus import FoamBenchCorpus, import_dataset
from foambench.runner.execution import run_split, run_task


def _corpus(tmp_path):
    source = tmp_path / "tasks.json"
    source.write_text(json.dumps({
        "case-one": {"usr_requirement": "Build a case.", "0/U": "reference"},
        "case-two": {"usr_requirement": "Build another case.", "0/p": "reference"},
    }), encoding="utf-8")
    import_dataset(source, tmp_path / "corpus", split="basic")
    return FoamBenchCorpus(tmp_path / "corpus")


def driver(task, tools, context):
    assert "reference" not in task.problem_statement
    assert set(tools.names) == {"read_file", "list_files", "write_file", "run_program", "verify_workspace"}
    assert "reference_root" not in context
    tools.execute("write_file", {"path": "system/controlDict", "content": "application icoFoam;\n"})
    return {"stop_reason": "fixture_complete"}


def test_run_task_persists_public_trace_and_submission(tmp_path):
    corpus = _corpus(tmp_path)
    task_id = corpus.list_tasks()[0].instance_id

    outcome = run_task(corpus, task_id, tmp_path / "runs" / task_id, driver=driver, model="fixture")

    assert outcome.status == "completed"
    result = json.loads((tmp_path / "runs" / task_id / "prediction.json").read_text())
    assert result["files"] == {"system/controlDict": "application icoFoam;\n"}
    assert not (tmp_path / "runs" / task_id / "workspace" / "private").exists()


def test_run_split_and_resume(tmp_path):
    corpus = _corpus(tmp_path)
    root = tmp_path / "runs"

    outcomes = run_split(corpus, root, driver=driver, split="basic")
    resumed = run_split(corpus, root, driver=driver, split="basic", resume=True)

    assert len(outcomes) == 2
    assert all(item.status == "completed" for item in outcomes)
    assert all(item.skipped for item in resumed)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["completed"] == 2
