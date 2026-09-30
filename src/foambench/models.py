"""Public, standalone task contracts for the FoamBench adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class FoamBenchError(ValueError):
    """Raised when a FoamBench corpus or workspace violates its public contract."""


@dataclass(frozen=True)
class FoamBenchTask:
    """Metadata visible to the agent; never contains reference solution files."""

    instance_id: str
    source_case_id: str
    split: str
    problem_statement: str
    target_software: str = "OpenFOAM 10"
    submission_root: str = "output"
    image: str = "openfoam/openfoam10-paraview510"
    timeout_seconds: float = 1800.0
    mpi_required: bool = False
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["metadata"] = dict(self.metadata or {})
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FoamBenchTask":
        required = ("instance_id", "source_case_id", "split", "problem_statement")
        absent = [name for name in required if not isinstance(value.get(name), str) or not value[name].strip()]
        if absent:
            raise FoamBenchError(f"task metadata is missing non-empty fields: {', '.join(absent)}")
        if value["split"] not in {"basic", "advanced"}:
            raise FoamBenchError("FoamBench split must be basic or advanced")
        return cls(
            instance_id=value["instance_id"],
            source_case_id=value["source_case_id"],
            split=value["split"],
            problem_statement=value["problem_statement"],
            target_software=str(value.get("target_software", "OpenFOAM 10")),
            submission_root=str(value.get("submission_root", "output")),
            image=str(value.get("image", "openfoam/openfoam10-paraview510")),
            timeout_seconds=float(value.get("timeout_seconds", 1800.0)),
            mpi_required=bool(value.get("mpi_required", False)),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class EvaluationRequest:
    """Private evaluation request constructed only after agent execution."""

    task: FoamBenchTask
    submission_root: Path
    reference_root: Path

