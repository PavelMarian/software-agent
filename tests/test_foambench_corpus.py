from __future__ import annotations

import json

import pytest

from foambench.corpus import FoamBenchCorpus, import_dataset
from foambench.models import FoamBenchError


def dataset(path):
    path.write_text(json.dumps({
        "LidDrivenCavity/1": {
            "usr_requirement": "Run a lid-driven cavity with pisoFoam.",
            "0/U": "velocity",
            "constant/physicalProperties": "nu 1e-05;",
            "system/controlDict": "application pisoFoam;",
            "10/U": "reference result",
        }
    }), encoding="utf-8")


def test_import_keeps_reference_files_outside_agent_seed(tmp_path):
    source = tmp_path / "basic.json"
    dataset(source)

    (task,) = import_dataset(source, tmp_path / "corpus", split="basic")
    corpus = FoamBenchCorpus(tmp_path / "corpus")

    assert task.instance_id == "foambench__basic__LidDrivenCavity-1"
    assert (corpus.reference_root(task.instance_id) / "10/U").read_text() == "reference result"
    assert not any((corpus.task_root(task.instance_id) / "seed").rglob("*"))
    assert "reference result" not in corpus.load(task.instance_id).problem_statement


def test_materialize_exposes_only_public_seed(tmp_path):
    source = tmp_path / "basic.json"
    dataset(source)
    (task,) = import_dataset(source, tmp_path / "corpus", split="basic")

    workspace = tmp_path / "run" / "workspace"
    FoamBenchCorpus(tmp_path / "corpus").materialize(task.instance_id, workspace)

    assert workspace.is_dir()
    assert not (workspace / "private").exists()
    assert not (workspace / "GT_Files").exists()


def test_import_rejects_unsafe_reference_path(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"case": {"usr_requirement": "x", "../secret": "x"}}), encoding="utf-8")

    with pytest.raises(FoamBenchError, match="unsafe"):
        import_dataset(source, tmp_path / "corpus", split="basic")
