from dataclasses import replace
from pathlib import Path

from software_bench.core.task_bundle import load_task_bundle
from software_bench.evaluation.backend import EvaluationEvidence
from software_bench.evaluation.scoring import score


BUNDLE = load_task_bundle(Path(__file__).parents[1] / "fixtures" / "task_bundle")


def evidence(statuses: dict[str, str]) -> EvaluationEvidence:
    return EvaluationEvidence(True, True, statuses, (), ())


def test_fix_rate_is_zero_when_pass_to_pass_regresses() -> None:
    result = score(
        BUNDLE,
        evidence({"feature_contract": "PASSED", "regression_contract": "FAILED"}),
    )

    assert result["raw_fix_rate"] == 1.0
    assert result["fix_rate"] == 0.0
    assert result["resolved"] is False


def test_empty_pass_to_pass_is_regression_free() -> None:
    bundle = replace(BUNDLE, evaluation=replace(BUNDLE.evaluation, pass_to_pass=()))
    result = score(bundle, evidence({"feature_contract": "PASSED"}))

    assert result["fix_rate"] == 1.0
    assert result["PASS_TO_PASS"]["regression_free"] is True
