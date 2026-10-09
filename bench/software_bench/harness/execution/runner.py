from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from software_bench.core.artifacts import EncodedArtifact, artifact_size
from software_bench.harness.agents.orchestrator import Orchestrator, OrchestratorOutcome
from software_bench.harness.contracts import FrameworkAdapter, ModelAdapter, RunRequest
from software_bench.harness.execution.artifacts import (
    _artifact_manifest,
    _materialize_artifacts,
    _strip_submission_root,
)
from software_bench.harness.execution.budget import BudgetExceeded, BudgetLedger
from software_bench.harness.execution.metadata import (
    _agent_runtime,
    _available_tools,
    _completed_agents,
    _completed_phases,
)
from software_bench.harness.execution.trace import Observer, TraceEvent
from software_bench.metrics import summarize_trace

@dataclass(frozen=True)
class RunRecord:
    manifest: Mapping[str, Any]
    trace: tuple[TraceEvent, ...]
    patch: str
    files: Mapping[str, EncodedArtifact]
    evidence: Mapping[str, EncodedArtifact]
    stage_artifacts: Mapping[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    def run(self, request: RunRequest, adapter: ModelAdapter, output_dir: Path) -> RunRecord:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            ledger = BudgetLedger(request.mode.budget)
            observer = Observer(request, ledger)
            observer.record("run_started", "harness", {"adapter_id": adapter.adapter_id})
            try:
                outcome = Orchestrator().run(request, adapter, observer)
                ledger.refresh_wall_time(enforce=observer.enforce_budget)
            except BudgetExceeded as error:
                outcome = OrchestratorOutcome(
                    "budget_exhausted",
                    str(error),
                    True,
                    _completed_phases(observer.events),
                    _completed_agents(observer.events),
                    partial_submission=True,
                )
            except Exception as error:  # adapters and tools are integration boundaries
                outcome = OrchestratorOutcome(
                    "adapter_error",
                    f"{type(error).__name__}: {error}",
                    False,
                    _completed_phases(observer.events),
                    _completed_agents(observer.events),
                )
            try:
                submission_kind = str(request.submission_kind)
                if submission_kind == "patch":
                    patch = request.environment.extract_patch()
                    files: Mapping[str, EncodedArtifact] = {}
                    evidence: Mapping[str, EncodedArtifact] = {}
                elif submission_kind == "artifact_bundle":
                    if request.submission_root is None:
                        raise ValueError("artifact_bundle requires submission_root")
                    patch = ""
                    captured = request.environment.extract_files((request.submission_root,))
                    files = _strip_submission_root(captured, request.submission_root)
                    evidence = {}
                elif submission_kind == "environment_state":
                    patch = ""
                    files = {}
                    evidence = request.environment.extract_files(request.state_paths)
                else:
                    # Compatibility with pre-0.5 workspace_files bundles.
                    patch = ""
                    files = request.environment.extract_files(request.submission_paths)
                    evidence = {}
            except Exception as error:
                patch = ""
                files = {}
                evidence = {}
                outcome = OrchestratorOutcome(
                    "environment_error",
                    f"{type(error).__name__}: {error}",
                    False,
                    outcome.phase_count,
                    outcome.agent_run_count,
                    outcome.artifacts,
                )
            observer.record_final(outcome.status, outcome.stop_reason)
            manifest = self._manifest(
                request, adapter, ledger, outcome, observer.events, patch, files, evidence
            )
            record = RunRecord(
                manifest, tuple(observer.events), patch, files, evidence, outcome.artifacts
            )
            self._write(record, output_dir)
            return record
        finally:
            request.environment.close()

    def run_framework(
        self,
        request: RunRequest,
        adapter: FrameworkAdapter,
        output_dir: Path,
    ) -> RunRecord:
        """Run a labeled black-box framework through the normalized result boundary."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            ledger = BudgetLedger(request.mode.budget)
            observer = Observer(request, ledger)
            observer.record("run_started", "harness", {"adapter_id": adapter.adapter_id})
            try:
                framework = adapter.run(request, observer)
                ledger.refresh_wall_time(enforce=observer.enforce_budget)
                if framework.submission_kind != request.submission_kind:
                    raise ValueError(
                        "framework submission_kind does not match the TaskBundle"
                    )
                patch = framework.patch
                files = framework.files
                evidence = framework.evidence
                outcome = OrchestratorOutcome(
                    framework.status,
                    framework.stop_reason,
                    framework.measurement_complete,
                    0,
                    1,
                )
            except BudgetExceeded as error:
                patch, files, evidence = "", {}, {}
                outcome = OrchestratorOutcome(
                    "budget_exhausted", str(error), True, 0, 1, partial_submission=True
                )
            except Exception as error:
                patch, files, evidence = "", {}, {}
                outcome = OrchestratorOutcome(
                    "adapter_error",
                    f"{type(error).__name__}: {error}",
                    False,
                    0,
                    1,
                )
            observer.record_final(outcome.status, outcome.stop_reason)
            manifest = self._manifest(
                request,
                adapter,
                ledger,
                outcome,
                observer.events,
                patch,
                files,
                evidence,
                adapter_contract="framework",
            )
            record = RunRecord(
                manifest, tuple(observer.events), patch, files, evidence
            )
            self._write(record, output_dir)
            return record
        finally:
            request.environment.close()

    @staticmethod
    def _manifest(
        request: RunRequest,
        adapter: ModelAdapter,
        ledger: BudgetLedger,
        outcome: OrchestratorOutcome,
        events: list[TraceEvent],
        patch: str,
        files: Mapping[str, EncodedArtifact],
        evidence: Mapping[str, EncodedArtifact],
        *,
        adapter_contract: str = "model",
    ) -> dict[str, Any]:
        captured = {**files, **evidence}
        return {
            "schema_version": "0.5.0",
            "run_id": request.run_id,
            "instance_id": request.task.instance_id,
            "mode": str(request.mode.id),
            "agent_topology": str(request.mode.agent_topology),
            "agent_runtime": _agent_runtime(request),
            "adapter_id": adapter.adapter_id,
            "adapter_contract": adapter_contract,
            "environment_backend": request.environment.backend_id,
            "seed": request.seed,
            "status": outcome.status,
            "stop_reason": outcome.stop_reason,
            "partial_submission": outcome.partial_submission,
            "partial_submission_reason": (
                outcome.stop_reason if outcome.partial_submission else ""
            ),
            "measurement_complete": outcome.measurement_complete,
            "phase_count": outcome.phase_count,
            "agent_run_count": outcome.agent_run_count,
            "budget_policy": request.mode.budget_policy,
            "fair_comparison_group": request.mode.fair_comparison_group,
            "budget": asdict(request.mode.budget),
            "usage": ledger.usage.to_dict(),
            "roles": [
                {**asdict(role), "workspace_access": str(role.workspace_access)}
                for role in request.mode.roles
            ],
            "patch_size_bytes": len(patch.encode("utf-8")),
            "submission_kind": str(request.submission_kind),
            "submission_file_count": len(files),
            "evidence_file_count": len(evidence),
            "stage_artifact_count": len(outcome.artifacts),
            "submission_size_bytes": sum(
                len(name.encode("utf-8")) + artifact_size(content)
                for name, content in captured.items()
            ),
            "trace_metrics": summarize_trace(events, _available_tools(request)),
            "metadata": {"adapter": dict(getattr(adapter, "metadata", {}))},
        }

    @staticmethod
    def _write(record: RunRecord, output_dir: Path) -> None:
        (output_dir / "run.json").write_text(
            json.dumps(record.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (output_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
            for event in record.trace:
                handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        (output_dir / "patch.diff").write_text(record.patch, encoding="utf-8")
        submission_dir = output_dir / "submission"
        _materialize_artifacts(record.files, submission_dir, field="files")
        _materialize_artifacts(record.evidence, output_dir / "evidence", field="evidence")
        stage_dir = output_dir / "stage_artifacts"
        stage_names = {"execution_plan": "plan"}
        for name, content in sorted(record.stage_artifacts.items()):
            safe_name = stage_names.get(name, name)
            if not safe_name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"unsafe stage artifact name: {name}")
            stage_dir.mkdir(parents=True, exist_ok=True)
            (stage_dir / f"{safe_name}.json").write_text(
                json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        prediction = {
            "instance_id": record.manifest["instance_id"],
            "model_name_or_path": record.manifest["adapter_id"],
            "model_patch": record.patch,
            "submission_kind": record.manifest["submission_kind"],
            "files": dict(record.files),
            "evidence": dict(record.evidence),
            "artifact_manifest": _artifact_manifest({**record.files, **record.evidence}),
        }
        (output_dir / "prediction.json").write_text(
            json.dumps(prediction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


