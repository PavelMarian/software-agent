"""LangGraph adapter for the original bidirectional shared software memory."""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from software_multiagent.core.action_graph import Action
from software_multiagent.core.contracts import AgentSpec, RuntimeState, TaskContext, ToolResult
from software_multiagent.core.execution import VerificationResult, VerificationStatus
from software_multiagent.software_memory.integrations.shared_agents import (
    MemoryRole,
    SharedSoftwareMemory,
    memory_coordination_tools,
)


class PlanMemoryInput(BaseModel):
    selected_contract_ids: list[str] | None = None
    workflow_id: str | None = None
    workflow_position: list[str] | None = None
    knowledge_gaps: list[str] | None = None


class EvidenceMemoryInput(BaseModel):
    gap_id: str
    source_uri: str
    locator: str
    content: str
    source_version: str
    source_kind: str = "documentation"
    authoritative: bool = False


class ProgressMemoryInput(BaseModel):
    current_predicates: list[dict[str, Any]] | None = None
    workflow_position: list[str] | None = None
    produced_artifacts: list[str] | None = None
    last_error: str | None = None
    failed_contract_id: str | None = None


_ROLE_TOOL = {
    MemoryRole.PLANNER: "memory_update_plan",
    MemoryRole.RESEARCHER: "memory_record_evidence",
    MemoryRole.EXECUTOR: "memory_update_progress",
}
_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "memory_update_plan": PlanMemoryInput,
    "memory_record_evidence": EvidenceMemoryInput,
    "memory_update_progress": ProgressMemoryInput,
}


class LangGraphSharedMemory:
    """Bind one task/run to the existing role-conditioned shared-memory API."""

    def __init__(self, memory: SharedSoftwareMemory, task: TaskContext) -> None:
        self.memory = memory
        self.task = task
        self.runtime = RuntimeState(task=task, current_node="planner")
        self.read_count = 0
        self.write_count = 0
        self.verification_count = 0
        self._coordination = {tool.name: tool for tool in memory_coordination_tools(memory)}
        self._tools = self._make_tools()

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        return self._tools

    def begin(self) -> None:
        self.memory.begin_run(self.task)

    def finish(self) -> None:
        self.memory.finish_run(self.task)

    def context_for(self, role: str) -> str:
        memory_role = MemoryRole(role)
        self.runtime.current_node = memory_role.value
        agent = self._agent(memory_role)
        rendered = self.memory.context_for(self.task, agent, self.runtime).render()
        self.read_count += 1
        return rendered

    def tool_names_for(self, role: str) -> tuple[str, ...]:
        name = _ROLE_TOOL.get(MemoryRole(role))
        return (name,) if name else ()

    def observe_tool(
        self,
        role: str,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        result: Any,
    ) -> None:
        if name.startswith("memory_"):
            return
        error = result.get("error") if isinstance(result, Mapping) else None
        action = Action(
            id=call_id,
            tool=name,
            arguments=dict(arguments),
            mutates_workspace=name == "write_file",
            artifact_outputs=(str(arguments["path"]),) if name == "write_file" and arguments.get("path") else (),
        )
        tool_result = ToolResult(
            call_id=call_id,
            tool_name=name,
            output=result,
            error=str(error) if error else None,
        )
        self.memory.observe_action(
            self.task,
            self._agent(MemoryRole(role)),
            self.runtime,
            action,
            tool_result,
            None,
        )

    def observe_verification(self, raw: Mapping[str, Any]) -> None:
        self.verification_count += 1
        action_id = f"public-verification-{self.verification_count}"
        passed = raw.get("passed") is True
        verification = VerificationResult(
            check_id=str(raw.get("check_id") or "public-verifier"),
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            summary=str(raw.get("summary") or ""),
        )
        result = ToolResult(action_id, "verify_workspace", output=dict(raw))
        self.memory.observe_action(
            self.task,
            self._agent(MemoryRole.EVALUATOR),
            self.runtime,
            Action(action_id, "verify_workspace"),
            result,
            verification,
        )

    def snapshot(self):  # type: ignore[no-untyped-def]
        return self.memory.snapshot(self.task.task_id)

    def _make_tools(self) -> tuple[BaseTool, ...]:
        result: list[BaseTool] = []
        for role, name in _ROLE_TOOL.items():
            source = self._coordination[name]

            def invoke(_role: MemoryRole = role, _name: str = name, **arguments: Any) -> Any:
                supplied = {key: value for key, value in arguments.items() if value is not None}
                output = self._coordination[_name].execute_for(
                    self._agent(_role), supplied, self.task
                )
                self.write_count += 1
                return output

            result.append(
                StructuredTool.from_function(
                    func=invoke,
                    name=name,
                    description=source.description,
                    args_schema=_INPUT_MODELS[name],
                )
            )
        return tuple(result)

    @staticmethod
    def _agent(role: MemoryRole) -> AgentSpec:
        tool = _ROLE_TOOL.get(role)
        return AgentSpec(
            id=role.value,
            instructions=f"Act as the {role.value}.",
            tools=(tool,) if tool else (),
        )


__all__ = ["LangGraphSharedMemory"]
