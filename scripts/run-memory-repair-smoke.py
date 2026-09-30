"""Run real-LLM smoke tests for shared memory, tool recovery, and repair routing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from software_multiagent.application import SoftwareMultiAgent, SoftwareRunRequest
from software_multiagent.config import load_environment, target_model
from software_multiagent.core.contracts import TaskContext
from software_multiagent.software_memory.integrations.langgraph import LangGraphSharedMemory
from software_multiagent.software_memory.integrations.shared_agents import SharedSoftwareMemory
from software_multiagent.providers import create_chat_model
from software_multiagent.runtime.orchestration.langgraph import LangGraphMultiAgent
from software_multiagent.software_memory import MemoryRetriever, MemoryService, SQLiteMemoryStore
from software_multiagent.software_memory.schema.models import Entity, EvidenceRecord, EvidenceSourceKind, SoftwareIdentity, stable_id
from software_multiagent.tools.workspace import WorkspaceFiles, WorkspaceToolset

MEMORY_VALUE = "MEMORY_ONLY_VALUE_9F3C2A7B"
RESEARCH_VALUE = "RESEARCH_WRITE_CONFIRMED_42A1"
REPAIRED_VALUE = "REPAIRED_AFTER_VERIFIER_6D8E1C"
RECOVERED_VALUE = "RECOVERED_AFTER_TOOL_ERROR_73B9"


def _react_complete(result: dict[str, Any]) -> bool:
    trace = result.get("reasoning_trace", [])
    for role in ("planner", "researcher", "executor", "evaluator"):
        steps = [item for item in trace if item.get("role") == role]
        if len(steps) < 2 or steps[-1].get("kind") != "finish":
            return False
        if not any(item.get("kind") in {"thought", "action"} for item in steps[:-1]):
            return False
    return True


def _toolset(workspace: Path, verifier):  # type: ignore[no-untyped-def]
    return WorkspaceToolset(
        WorkspaceFiles(workspace),
        lambda program, timeout, arguments: {"exit_code": 127, "stderr": "program execution is unnecessary"},
        verifier,
    )


class TrackingTools:
    def __init__(self, wrapped: WorkspaceToolset) -> None:
        self.wrapped, self.names, self.calls = wrapped, wrapped.names, []

    def execute(self, name: str, arguments: dict[str, Any]):
        try:
            result = self.wrapped.execute(name, arguments)
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        self.calls.append({"name": name, "arguments": arguments, "result": result})
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    load_environment()
    provider, model_name = target_model()
    if provider not in {"openai", "openrouter"} or not model_name:
        raise SystemExit("configure provider and model in .env")
    root = arguments.output.resolve()
    if root.exists():
        raise SystemExit(f"output already exists: {root}")
    memory_workspace = root / "memory-run" / "workspace"
    repair_workspace = root / "repair-run" / "workspace"
    recovery_workspace = root / "tool-recovery-run" / "workspace"
    for workspace in (memory_workspace, repair_workspace, recovery_workspace):
        workspace.mkdir(parents=True)
    model = create_chat_model(provider=provider, model=model_name)

    software = SoftwareIdentity.create("Synthetic Text Workspace", "1")
    seed_evidence = EvidenceRecord.create(
        software_id=software.software_id,
        source_kind=EvidenceSourceKind.DOCUMENTATION,
        source_uri="fixture://synthetic-memory/required-value",
        locator="required-output",
        content=f"The entire content of memory.txt must be exactly {MEMORY_VALUE} with no newline.",
        source_version="1",
        authoritative=True,
    )
    entity = Entity(
        entity_id=stable_id("entity", software.software_id, "required phrase"),
        software_id=software.software_id,
        entity_type="task_requirement",
        name="required phrase for memory.txt",
        summary="Memory-only output requirement.",
        attributes={"required_text": MEMORY_VALUE, "newline": False},
        evidence_ids=(seed_evidence.evidence_id,),
    )
    gap = "Record the supplied synthetic research confirmation in persistent memory."

    def verify_memory():
        target = memory_workspace / "memory.txt"
        actual = target.read_text(encoding="utf-8") if target.is_file() else None
        return {"passed": actual == MEMORY_VALUE, "summary": "memory file checked", "actual": actual, "expected": MEMORY_VALUE}

    database = root / "memory-run" / "synthetic-memory.sqlite"
    with SQLiteMemoryStore(database) as store:
        service = MemoryService(store)
        service.register_software(software)
        service.record_evidence(seed_evidence)
        service.record_entity(entity)
        task = TaskContext(
            "synthetic-bidirectional-memory",
            (
                "Consult shared memory to discover the exact value for memory.txt; it is absent from this task. "
                "Create memory.txt with only that value. Planner must call memory_update_plan and preserve this gap: "
                f"'{gap}' Researcher must resolve the gap with memory_record_evidence using source_uri "
                "fixture://smoke/research, locator confirmation, content " + RESEARCH_VALUE +
                ", source_version 1, source_kind documentation, authoritative true. "
                "Executor must call memory_update_progress with memory.txt as a produced artifact."
            ),
            memory_workspace,
            "Synthetic Text Workspace",
            {"software_memory_id": software.software_id, "software_version": "1", "knowledge_gaps": [gap]},
        )
        bridge = LangGraphSharedMemory(SharedSoftwareMemory(MemoryRetriever(store), service=service), task)
        memory_result = LangGraphMultiAgent(
            model, _toolset(memory_workspace, verify_memory), shared_memory=bridge,
            require_memory_writes=True, max_repairs=1,
            max_executor_tool_calls=10, max_evaluator_tool_calls=4,
            required_memory_write_arguments={
                "planner": {"knowledge_gaps": [gap]},
                "researcher": {
                    "gap_id": stable_id("gap", task.task_id, gap),
                    "source_uri": "fixture://smoke/research",
                    "locator": "confirmation",
                    "content": RESEARCH_VALUE,
                    "source_version": "1",
                    "source_kind": "documentation",
                    "authoritative": True,
                },
                "executor": {"produced_artifacts": ["memory.txt"]},
            },
        ).run(task_id=task.task_id, objective=task.objective, target_software=task.target_software or "")
        memory_verification = verify_memory()
        memory_state = bridge.snapshot()
        stored_evidence = store.list_evidence(software.software_id)
        research_writes = [item for item in stored_evidence if item.metadata.get("researcher") == "researcher"]

    verifier_calls: list[dict[str, Any]] = []

    def verify_repair():
        target = repair_workspace / "repair.txt"
        actual = target.read_text(encoding="utf-8") if target.is_file() else None
        result = (
            {"passed": False, "summary": f"Replace the file with exactly {REPAIRED_VALUE}.", "actual": actual}
            if not verifier_calls else
            {"passed": actual == REPAIRED_VALUE, "summary": "repair checked", "actual": actual, "expected": REPAIRED_VALUE}
        )
        verifier_calls.append(result)
        return result

    repair_result = SoftwareMultiAgent.run(
        model,
        SoftwareRunRequest(
            "synthetic-forced-repair",
            "Create repair.txt with exact text INITIAL_CANDIDATE and no newline. If verification rejects it, follow its instruction exactly.",
            repair_workspace,
            "generic text workspace",
        ),
        _toolset(repair_workspace, verify_repair),
        runtime_options={
            "max_repairs": 1,
            "max_executor_tool_calls": 8,
            "max_evaluator_tool_calls": 4,
            "executor_tool_names": ("read_file", "list_files", "write_file"),
        },
    )
    final_repair = (repair_workspace / "repair.txt").read_text(encoding="utf-8")

    def verify_recovery():
        target = recovery_workspace / "recovered.txt"
        actual = target.read_text(encoding="utf-8") if target.is_file() else None
        return {"passed": actual == RECOVERED_VALUE, "summary": "tool recovery checked", "actual": actual, "expected": RECOVERED_VALUE}

    recovery_tools = TrackingTools(_toolset(recovery_workspace, verify_recovery))
    recovery_result = LangGraphMultiAgent(
        model, recovery_tools, max_repairs=1, max_executor_tool_calls=10,
        max_evaluator_tool_calls=4, executor_tool_names=("read_file", "list_files", "write_file"),
    ).run(
        task_id="synthetic-tool-error-recovery",
        objective=(
            "First call read_file for definitely-missing.txt so you observe the tool error. Then recover by creating "
            f"recovered.txt containing exactly {RECOVERED_VALUE} with no newline."
        ),
        target_software="generic text workspace",
    )
    recovery_verification = verify_recovery()
    observed_errors = [call for call in recovery_tools.calls if isinstance(call["result"], dict) and call["result"].get("error")]

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "provider": provider, "model": model_name,
        "memory_run": {
            "result": memory_result, "independent_verification": memory_verification,
            "research_write_count": len(research_writes), "evidence_count": len(stored_evidence),
            "produced_artifacts": list(memory_state.produced_artifacts), "open_gap_count": len(memory_state.knowledge_gaps),
        },
        "repair_run": {"result": repair_result, "verifier_call_count": len(verifier_calls), "verifier_calls": verifier_calls, "final_content": final_repair},
        "tool_recovery_run": {"result": recovery_result, "observed_tool_error_count": len(observed_errors), "calls": recovery_tools.calls, "independent_verification": recovery_verification},
    }
    (root / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    passed = (
        memory_result["status"] == "completed" and memory_result["memory_reads"] >= 5
        and memory_result["memory_writes"] >= 3 and memory_verification["passed"]
        and len(research_writes) >= 1 and "memory.txt" in memory_state.produced_artifacts
        and repair_result["status"] == "completed" and repair_result["repairs"] >= 1
        and len(verifier_calls) >= 2 and final_repair == REPAIRED_VALUE
        and recovery_result["status"] == "completed" and len(observed_errors) >= 1 and recovery_verification["passed"]
        and _react_complete(memory_result) and _react_complete(repair_result)
        and _react_complete(recovery_result)
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
