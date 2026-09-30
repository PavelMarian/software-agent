from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from software_multiagent.runtime.orchestration.langgraph import (
    LangGraphMultiAgent,
    _split_react_content,
)
from software_multiagent.core.contracts import TaskContext
from software_multiagent.software_memory.integrations.langgraph import LangGraphSharedMemory
from software_multiagent.software_memory.integrations.shared_agents import SharedSoftwareMemory
from software_multiagent.software_memory import (
    MemoryRetriever,
    MemoryService,
    SQLiteMemoryStore,
    SoftwareIdentity,
    stable_id,
)
from software_multiagent.tools.workspace import WorkspaceFiles, WorkspaceToolset


class ScriptedChatModel:
    def __init__(self, *responses: AIMessage) -> None:
        self.responses = iter(responses)
        self.bound_tool_names: list[tuple[str, ...]] = []

    def bind_tools(self, tools):  # type: ignore[no-untyped-def]
        self.bound_tool_names.append(tuple(tool.name for tool in tools))
        return self

    def invoke(self, messages):  # type: ignore[no-untyped-def]
        return next(self.responses)


def thought(content: str) -> AIMessage:
    return AIMessage(content=f"Thought: {content}", usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3})


def final(content: str) -> AIMessage:
    return AIMessage(content=f"Thought: Evidence is sufficient.\nFinal: {content}", usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3})


def reasoned(content: str) -> tuple[AIMessage, AIMessage]:
    return thought(f"Assess the evidence needed for {content}."), final(content)


def tool_call(name: str, arguments: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}],
    )


def test_react_final_parser_accepts_blank_line_and_final_answer_heading() -> None:
    assert _split_react_content(
        "Thought: Evidence is sufficient.\n\nFinal Answer: deliverable"
    ) == ("Evidence is sufficient.", "deliverable")
    assert _split_react_content(
        "Thought: Evidence is sufficient. **Final:** deliverable"
    ) == ("Evidence is sufficient.", "deliverable")


def test_eager_final_is_reassessed_once_for_every_role(tmp_path) -> None:
    model = ScriptedChatModel(
        final("early initial plan"), final("initial plan"),
        final("early research"), final("research"),
        final("early final plan"), final("final plan"),
        final("early execution"), final("execution"),
        final("early evaluation"), final("evaluation"),
    )
    tools = WorkspaceToolset(
        WorkspaceFiles(tmp_path),
        lambda program, timeout, arguments: {"exit_code": 0},
        lambda: {"passed": True, "summary": "ok"},
    )

    result = LangGraphMultiAgent(model, tools).run(
        task_id="eager-final", objective="reassess before finishing"
    )

    assert result["status"] == "completed"
    assert result["model_turns"] == 10
    for role in ("planner", "researcher", "executor", "evaluator"):
        role_steps = [item for item in result["reasoning_trace"] if item["role"] == role]
        assert any(item.get("proposed_final_rejected") for item in role_steps)
        assert role_steps[-1]["kind"] == "finish"


def test_thought_only_role_is_bounded_and_auditable(tmp_path) -> None:
    responses = []
    for role in ("initial planner", "researcher", "final planner", "executor", "evaluator"):
        responses.extend((thought(f"{role} step 1"), thought(f"{role} step 2"), thought(f"{role} bounded deliverable")))
    tools = WorkspaceToolset(
        WorkspaceFiles(tmp_path),
        lambda program, timeout, arguments: {"exit_code": 0},
        lambda: {"passed": True, "summary": "ok"},
    )

    result = LangGraphMultiAgent(ScriptedChatModel(*responses), tools).run(
        task_id="thought-only", objective="exercise bounded reasoning"
    )

    forced = [item for item in result["reasoning_trace"] if item.get("runtime_forced_finish")]
    assert len(forced) == 5
    assert result["status"] == "completed"


def test_langgraph_workflow_uses_langchain_tools(tmp_path) -> None:
    model = ScriptedChatModel(
        *reasoned("initial plan"),
        *reasoned("research"),
        *reasoned("final plan"),
        tool_call("write_file", {"path": "result.txt", "content": "ready"}),
        final("execution complete"),
        *reasoned("evaluation complete"),
    )
    tools = WorkspaceToolset(
        WorkspaceFiles(tmp_path),
        lambda program, timeout, arguments: {"exit_code": 0},
        lambda: {"passed": True, "summary": "ok"},
    )

    result = LangGraphMultiAgent(model, tools).run(
        task_id="task-1", objective="produce result", target_software="example"
    )

    assert result["status"] == "completed"
    assert result["framework"] == {"orchestration": "langgraph", "components": "langchain"}
    assert result["model_turns"] == 10
    assert result["input_tokens"] == 18
    assert result["output_tokens"] == 9
    assert {item["role"] for item in result["reasoning_trace"]} == {
        "planner", "researcher", "executor", "evaluator"
    }
    for role in ("planner", "researcher", "executor", "evaluator"):
        role_steps = [item for item in result["reasoning_trace"] if item["role"] == role]
        assert len(role_steps) >= 2
        assert role_steps[-1]["kind"] == "finish"
        assert any(item["kind"] in {"thought", "action"} for item in role_steps[:-1])
    assert (tmp_path / "result.txt").read_text() == "ready"
    assert any("write_file" in names for names in model.bound_tool_names)


def test_failed_evaluation_routes_back_to_executor(tmp_path) -> None:
    verifications = iter(
        (
            {"passed": False, "summary": "repair required"},
            {"passed": True, "summary": "fixed"},
        )
    )
    model = ScriptedChatModel(
        *reasoned("initial plan"),
        *reasoned("research"),
        *reasoned("final plan"),
        *reasoned("first execution"),
        *reasoned("first evaluation"),
        *reasoned("repaired execution"),
        *reasoned("second evaluation"),
    )
    tools = WorkspaceToolset(
        WorkspaceFiles(tmp_path),
        lambda program, timeout, arguments: {"exit_code": 0},
        lambda: next(verifications),
    )

    result = LangGraphMultiAgent(model, tools, max_repairs=2).run(
        task_id="task-2", objective="repair result"
    )

    assert result["status"] == "completed"
    assert result["repairs"] == 1
    assert result["model_turns"] == 14


def test_research_node_reads_persistent_memory_once(tmp_path) -> None:
    queries: list[str] = []
    model = ScriptedChatModel(
        *reasoned("initial plan"),
        *reasoned("memory fact incorporated"),
        *reasoned("final plan"),
        *reasoned("execution complete"),
        *reasoned("evaluation complete"),
    )
    tools = WorkspaceToolset(
        WorkspaceFiles(tmp_path),
        lambda program, timeout, arguments: {"exit_code": 0},
        lambda: {"passed": True, "summary": "ok"},
    )

    result = LangGraphMultiAgent(
        model,
        tools,
        memory_search=lambda query: queries.append(query) or {"fact": "memory-only"},
    ).run(task_id="memory-task", objective="read the required fact")

    assert result["memory_accesses"] == 1
    assert queries == ["read the required fact"]


def test_shared_memory_is_read_and_written_by_authorized_roles(tmp_path) -> None:
    database = tmp_path / "memory.sqlite"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    software = SoftwareIdentity.create("Synthetic Workspace", "1")
    question = "Which exact synthetic convention applies?"
    gap_id = stable_id("gap", "shared-memory-task", question)
    model = ScriptedChatModel(
        tool_call("memory_update_plan", {"knowledge_gaps": [question]}, "plan-write"),
        final("initial plan stored"),
        tool_call(
            "memory_record_evidence",
            {
                "gap_id": gap_id,
                "source_uri": "fixture://shared-memory",
                "locator": "line-1",
                "content": "The synthetic convention is MEMORY_WRITES_WORK.",
                "source_version": "1",
                "source_kind": "documentation",
                "authoritative": True,
            },
            "evidence-write",
        ),
        final("research stored"),
        *reasoned("final plan"),
        tool_call("write_file", {"path": "result.txt", "content": "ok"}, "file-write"),
        tool_call("memory_update_progress", {"produced_artifacts": ["result.txt"]}, "progress-write"),
        final("execution complete"),
        *reasoned("evaluation complete"),
    )
    tools = WorkspaceToolset(
        WorkspaceFiles(workspace),
        lambda program, timeout, arguments: {"exit_code": 0},
        lambda: {"passed": True, "summary": "ok"},
    )
    task = TaskContext(
        "shared-memory-task",
        "Exercise bidirectional memory",
        workspace,
        "Synthetic Workspace",
        {"software_memory_id": software.software_id, "software_version": "1"},
    )
    with SQLiteMemoryStore(database) as store:
        service = MemoryService(store)
        service.register_software(software)
        bridge = LangGraphSharedMemory(
            SharedSoftwareMemory(MemoryRetriever(store), service=service), task
        )
        result = LangGraphMultiAgent(model, tools, shared_memory=bridge).run(
            task_id=task.task_id,
            objective=task.objective,
            target_software=task.target_software or "",
        )
        evidence = store.list_evidence(software.software_id)
        state = bridge.snapshot()

    assert result["status"] == "completed"
    assert result["memory_reads"] == 5
    assert result["memory_writes"] == 3
    assert len(evidence) == 1
    assert evidence[0].metadata["researcher"] == "researcher"
    assert state.produced_artifacts == ("result.txt",)
    assert state.active is False
    assert any("memory_update_plan" in names for names in model.bound_tool_names)
    assert any("memory_record_evidence" in names for names in model.bound_tool_names)
    assert any("memory_update_progress" in names for names in model.bound_tool_names)


def test_progress_write_rejects_inconsistent_workflow_transactionally(tmp_path) -> None:
    software = SoftwareIdentity.create("Transactional Workspace", "1")
    task = TaskContext(
        "transactional-memory",
        "Keep memory consistent",
        tmp_path,
        metadata={"software_memory_id": software.software_id},
    )
    with SQLiteMemoryStore(tmp_path / "transactional.sqlite") as store:
        service = MemoryService(store)
        service.register_software(software)
        memory = SharedSoftwareMemory(MemoryRetriever(store), service=service)
        memory.begin_run(task)
        with pytest.raises(ValueError, match="workflow_position requires workflow_id"):
            memory.update_progress(task.task_id, workflow_position=("step-1",))
        state = memory.snapshot(task.task_id)

    assert state.workflow_id is None
    assert state.workflow_position == ()
