"""Evidence-driven repair decisions, separate from model generation."""
from dataclasses import dataclass
from enum import Enum


class FailureKind(str, Enum):
    ACTION = "action"
    CALL = "call"
    STATE = "state"
    ENVIRONMENT = "environment"
    ARTIFACT = "artifact"
    APPROACH = "approach"
    VERIFICATION = "verification"


@dataclass(frozen=True)
class Diagnosis:
    kind: FailureKind
    reason: str
    strategy: str
    retry: bool
    attempt: int


class RepairPolicy:
    """Select a concrete next step from typed runtime/adapter evidence.

    Budgets belong to each run. No hidden cross-task retry state is retained.
    A domain adapter can subclass this policy to add specialized diagnosis.
    """

    def diagnose(self, kind, reason, *, attempt, repeated, max_repairs, max_repeated):
        kind = FailureKind(kind)
        strategies = {
            FailureKind.ACTION: "Correct the failed action using its execution output before retrying.",
            FailureKind.CALL: "Inspect the tool schema and correct the call without changing the objective.",
            FailureKind.STATE: "Inspect current state and satisfy the failed prerequisite before retrying.",
            FailureKind.ENVIRONMENT: "Diagnose the reported environment failure before retrying the operation.",
            FailureKind.ARTIFACT: "Inspect the failed artifact checks and repair the affected outputs.",
            FailureKind.APPROACH: "Gather evidence for an alternative approach; do not repeat the same attempt.",
            FailureKind.VERIFICATION: "Restore access to the missing validator or gather its required evidence.",
        }
        return Diagnosis(kind, reason, strategies[kind],
                         attempt <= max_repairs and repeated < max_repeated, attempt)
