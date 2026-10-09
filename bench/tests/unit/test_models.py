import json
from pathlib import Path

import pytest

from software_bench.core.models import TaskSpec, ValidationError
from software_bench.core.task_bundle import load_task_bundle, write_task_bundle


BUNDLE = Path(__file__).parents[1] / "fixtures" / "task_bundle"


def test_task_bundle_round_trips_without_merging_security_boundaries(tmp_path: Path) -> None:
    bundle = load_task_bundle(BUNDLE, require_mas_ready=True)
    write_task_bundle(bundle, tmp_path)
    restored = load_task_bundle(tmp_path, require_mas_ready=True)

    assert restored.task == bundle.task
    assert restored.evaluation == bundle.evaluation
    assert restored.provenance == bundle.provenance


def test_agent_view_does_not_expose_hidden_evaluation_or_gold_data() -> None:
    bundle = load_task_bundle(BUNDLE)
    visible = vars(bundle.agent_view)

    assert "gold_patch" not in visible
    assert "test_patch" not in visible
    assert "fail_to_pass" not in visible
    assert "TOP SECRET" not in repr(bundle.agent_view)


def test_imported_task_can_be_valid_before_mas_curation() -> None:
    source = json.loads((BUNDLE / "task.json").read_text(encoding="utf-8"))
    source["workstreams"] = []
    task = TaskSpec.from_dict(source)

    with pytest.raises(ValidationError, match="at least two workstreams"):
        task.validate_mas_ready()


def test_unknown_public_task_fields_are_rejected() -> None:
    source = json.loads((BUNDLE / "task.json").read_text(encoding="utf-8"))
    source["gold_patch"] = "leak"

    with pytest.raises(ValidationError, match="unknown task fields: gold_patch"):
        TaskSpec.from_dict(source)
