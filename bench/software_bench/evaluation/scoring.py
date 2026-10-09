from __future__ import annotations

from typing import Any

from software_bench.core.models import EvaluationStrategy, TaskBundle
from software_bench.evaluation.backend import EvaluationEvidence
from software_bench.evaluation.parsers import PASSED_STATUSES


def score(bundle: TaskBundle, evidence: EvaluationEvidence) -> dict[str, Any]:
    if bundle.evaluation.strategy == EvaluationStrategy.COMMAND_CHECKS:
        return _score_command_checks(bundle, evidence)
    return _score_swe_patch(bundle, evidence)


def _score_swe_patch(bundle: TaskBundle, evidence: EvaluationEvidence) -> dict[str, Any]:
    evaluation = bundle.evaluation
    statuses = evidence.test_statuses
    f2p_success = [
        item for item in evaluation.fail_to_pass if statuses.get(item) in PASSED_STATUSES
    ]
    f2p_failure = [item for item in evaluation.fail_to_pass if item not in f2p_success]
    p2p_success = [
        item for item in evaluation.pass_to_pass if statuses.get(item) in PASSED_STATUSES
    ]
    p2p_failure = [item for item in evaluation.pass_to_pass if item not in p2p_success]
    raw_fix_rate = len(f2p_success) / len(evaluation.fail_to_pass)
    regression_free = not p2p_failure
    fix_rate = raw_fix_rate if regression_free else 0.0
    required_oracles_pass = all(
        item.status == "passed" for item in evidence.additional_oracles if item.required
    )
    resolved = (
        evidence.patch_exists
        and evidence.patch_applied
        and not f2p_failure
        and regression_free
        and required_oracles_pass
        and evidence.infrastructure_error is None
    )
    return {
        "schema_version": "0.3.0",
        "instance_id": bundle.task.instance_id,
        "evaluation_backend": evidence.backend_id,
        "resolved": resolved,
        "patch_exists": evidence.patch_exists,
        "patch_applied": evidence.patch_applied,
        "fix_rate": fix_rate,
        "raw_fix_rate": raw_fix_rate,
        "FAIL_TO_PASS": {"success": f2p_success, "failure": f2p_failure},
        "PASS_TO_PASS": {
            "success": p2p_success,
            "failure": p2p_failure,
            "regression_free": regression_free,
        },
        "additional_oracles": [item.to_dict() for item in evidence.additional_oracles],
        "test_commands": [item.to_dict() for item in evidence.test_commands],
        "infrastructure_error": evidence.infrastructure_error,
    }


def _score_command_checks(bundle: TaskBundle, evidence: EvaluationEvidence) -> dict[str, Any]:
    statuses = evidence.test_statuses
    checks: list[dict[str, Any]] = []
    earned = 0.0
    available = 0.0
    required_pass = True
    for spec in bundle.evaluation.checks:
        passed = statuses.get(spec.id) in PASSED_STATUSES
        available += spec.weight
        if passed:
            earned += spec.weight
        if spec.required and not passed:
            required_pass = False
        checks.append(
            {
                "id": spec.id,
                "status": statuses.get(spec.id, "MISSING"),
                "passed": passed,
                "required": spec.required,
                "weight": spec.weight,
            }
        )
    required_oracles_pass = all(
        item.status == "passed" for item in evidence.additional_oracles if item.required
    )
    resolved = (
        evidence.patch_exists
        and evidence.patch_applied
        and required_pass
        and required_oracles_pass
        and evidence.infrastructure_error is None
    )
    return {
        "schema_version": "0.4.0",
        "instance_id": bundle.task.instance_id,
        "evaluation_strategy": str(bundle.evaluation.strategy),
        "evaluation_backend": evidence.backend_id,
        "resolved": resolved,
        "submission_exists": evidence.patch_exists,
        "submission_applied": evidence.patch_applied,
        "score": earned / available if available else 0.0,
        "checks": checks,
        "additional_oracles": [item.to_dict() for item in evidence.additional_oracles],
        "test_commands": [item.to_dict() for item in evidence.test_commands],
        "infrastructure_error": evidence.infrastructure_error,
    }
