"""LangGraph orchestration and LangChain tool execution for software agents."""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any, Callable, Literal, Mapping, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from software_multiagent.tools.workspace import WorkspaceToolset
from software_multiagent.software_memory.integrations.langgraph import LangGraphSharedMemory
from software_multiagent.core.contracts import ModelTurn, StepStatus, ToolCall
from software_multiagent.runtime.reasoning.react import ReActProtocol, ReActProtocolError


class ReadFileInput(BaseModel):
    path: str


class ListFilesInput(BaseModel):
    path: str = "."


class WriteFileInput(BaseModel):
    path: str
    content: str


class RunProgramInput(BaseModel):
    program: str
    timeout_seconds: float = Field(default=300, gt=0)
    arguments: list[str] = Field(default_factory=list)


class VerifyWorkspaceInput(BaseModel):
    pass


class SearchMemoryInput(BaseModel):
    query: str


_TOOL_INPUTS: dict[str, type[BaseModel]] = {
    "read_file": ReadFileInput,
    "list_files": ListFilesInput,
    "write_file": WriteFileInput,
    "run_program": RunProgramInput,
    "verify_workspace": VerifyWorkspaceInput,
}

_TOOL_DESCRIPTIONS = {
    "read_file": "Read a UTF-8 text file inside the permitted workspace.",
    "list_files": "List files inside the permitted workspace.",
    "write_file": "Create or replace a UTF-8 text file inside the permitted workspace.",
    "run_program": "Run one permitted program against the workspace.",
    "verify_workspace": "Run deterministic public verification of the current workspace.",
    "search_memory": "Search persistent software memory for task-relevant knowledge.",
}


class MultiAgentState(TypedDict):
    task_id: str
    objective: str
    target_software: str
    messages: Annotated[list[BaseMessage], add_messages]
    initial_plan: str
    research: str
    final_plan: str
    execution_summary: str
    evaluation: str
    verification: dict[str, Any]
    repairs: int
    status: str
    stop_reason: str
    model_turns: Annotated[int, operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    memory_accesses: Annotated[int, operator.add]
    reasoning_trace: Annotated[list[dict[str, Any]], operator.add]


class LangGraphMultiAgent:
    """Planner/researcher/executor/evaluator workflow backed by LangGraph."""

    def __init__(
        self,
        model: Any,
        workspace_tools: WorkspaceToolset,
        *,
        max_repairs: int = 2,
        max_executor_tool_calls: int = 60,
        max_evaluator_tool_calls: int = 20,
        memory_search: Callable[[str], Mapping[str, Any]] | None = None,
        shared_memory: LangGraphSharedMemory | None = None,
        require_memory_writes: bool = False,
        required_memory_write_arguments: Mapping[str, Mapping[str, Any]] | None = None,
        react: ReActProtocol | None = None,
        max_role_model_turns: int = 80,
        executor_tool_names: tuple[str, ...] = (
            "read_file",
            "list_files",
            "write_file",
            "run_program",
            "verify_workspace",
        ),
    ) -> None:
        self.model = model
        self.workspace_tools = workspace_tools
        self.max_repairs = max_repairs
        self.max_executor_tool_calls = max_executor_tool_calls
        self.max_evaluator_tool_calls = max_evaluator_tool_calls
        self.memory_search = memory_search
        self.shared_memory = shared_memory
        self.require_memory_writes = require_memory_writes
        self.required_memory_write_arguments = dict(required_memory_write_arguments or {})
        self.react = react or ReActProtocol()
        self.max_role_model_turns = max_role_model_turns
        self.executor_tool_names = executor_tool_names
        self.tools = self._make_tools(workspace_tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.graph = self._build_graph()

    def _make_tools(self, workspace_tools: WorkspaceToolset) -> tuple[BaseTool, ...]:
        result: list[BaseTool] = []
        for name in workspace_tools.names:
            if name not in _TOOL_INPUTS:
                raise ValueError(f"workspace tool has no LangChain schema: {name}")

            def invoke_tool(_name: str = name, **arguments: Any) -> Mapping[str, Any]:
                return workspace_tools.execute(_name, arguments)

            result.append(
                StructuredTool.from_function(
                    func=invoke_tool,
                    name=name,
                    description=_TOOL_DESCRIPTIONS[name],
                    args_schema=_TOOL_INPUTS[name],
                )
            )
        if self.memory_search is not None:
            result.append(
                StructuredTool.from_function(
                    func=lambda query: self.memory_search(query),
                    name="search_memory",
                    description=_TOOL_DESCRIPTIONS["search_memory"],
                    args_schema=SearchMemoryInput,
                )
            )
        if self.shared_memory is not None:
            result.extend(self.shared_memory.tools)
        return tuple(result)

    def _build_graph(self):  # type: ignore[no-untyped-def]
        builder = StateGraph(MultiAgentState)
        builder.add_node("initial_plan", self._initial_plan)
        builder.add_node("research", self._research)
        builder.add_node("final_plan", self._final_plan)
        builder.add_node("execute", self._execute)
        builder.add_node("evaluate", self._evaluate)
        builder.add_edge(START, "initial_plan")
        builder.add_edge("initial_plan", "research")
        builder.add_edge("research", "final_plan")
        builder.add_edge("final_plan", "execute")
        builder.add_edge("execute", "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            self._after_evaluation,
            {"repair": "execute", "end": END},
        )
        return builder.compile(checkpointer=InMemorySaver(), name="software-multiagent")

    def run(
        self,
        *,
        task_id: str,
        objective: str,
        target_software: str = "",
    ) -> dict[str, Any]:
        initial: MultiAgentState = {
            "task_id": task_id,
            "objective": objective,
            "target_software": target_software,
            "messages": [],
            "initial_plan": "",
            "research": "",
            "final_plan": "",
            "execution_summary": "",
            "evaluation": "",
            "verification": {},
            "repairs": 0,
            "status": "running",
            "stop_reason": "",
            "model_turns": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "memory_accesses": 0,
            "reasoning_trace": [],
        }
        if self.shared_memory is not None:
            self.shared_memory.begin()
        try:
            state = self.graph.invoke(
                initial,
                config={
                    "configurable": {"thread_id": task_id},
                    "recursion_limit": 10 + 2 * self.max_repairs,
                },
            )
        finally:
            if self.shared_memory is not None:
                self.shared_memory.finish()
        memory_reads = self.shared_memory.read_count if self.shared_memory else state["memory_accesses"]
        memory_writes = self.shared_memory.write_count if self.shared_memory else 0
        return {
            "stop_reason": state["stop_reason"],
            "status": state["status"],
            "model_turns": state["model_turns"],
            "input_tokens": state["input_tokens"],
            "output_tokens": state["output_tokens"],
            "repairs": state["repairs"],
            "verification": state["verification"],
            "memory_accesses": memory_reads,
            "memory_reads": memory_reads,
            "memory_writes": memory_writes,
            "reasoning_trace": state["reasoning_trace"],
            "framework": {"orchestration": "langgraph", "components": "langchain"},
        }

    def _initial_plan(self, state: MultiAgentState) -> dict[str, Any]:
        text, messages, usage, reasoning = self._role(
            "planner",
            "Create a concrete initial plan using only the public task. Identify required files, commands, and acceptance checks.",
            state,
        )
        return {"initial_plan": text, "messages": messages, "reasoning_trace": reasoning, **usage}

    def _research(self, state: MultiAgentState) -> dict[str, Any]:
        memory_context = ""
        memory_accesses = 0
        if "search_memory" in self.tools_by_name:
            memory_result = self.tools_by_name["search_memory"].invoke(
                {"query": state["objective"]}
            )
            memory_context = (
                "\n\nPersistent software-memory lookup (use relevant values exactly):\n"
                + json.dumps(memory_result, ensure_ascii=False, default=str)
            )
            memory_accesses = 1
        text, messages, usage, reasoning = self._role(
            "researcher",
            "Resolve gaps and risks in the initial plan. Use supplied persistent memory and do not invent private reference data.",
            state,
            context=f"Initial plan:\n{state['initial_plan']}{memory_context}",
        )
        return {
            "research": text,
            "messages": messages,
            "memory_accesses": memory_accesses,
            "reasoning_trace": reasoning,
            **usage,
        }

    def _final_plan(self, state: MultiAgentState) -> dict[str, Any]:
        text, messages, usage, reasoning = self._role(
            "planner",
            "Produce the final executable plan, incorporating the research findings.",
            state,
            context=f"Initial plan:\n{state['initial_plan']}\n\nResearch:\n{state['research']}",
        )
        return {"final_plan": text, "messages": messages, "reasoning_trace": reasoning, **usage}

    def _execute(self, state: MultiAgentState) -> dict[str, Any]:
        repair = ""
        if state["repairs"]:
            repair = f"\n\nPrevious verification:\n{json.dumps(state['verification'], default=str)}"
        text, messages, usage, reasoning = self._role(
            "executor",
            "Implement the plan with tools. Run the required program and inspect results. Never claim success without observable evidence.",
            state,
            context=f"Final plan:\n{state['final_plan']}{repair}",
            tool_names=self.executor_tool_names,
            max_tool_calls=self.max_executor_tool_calls,
        )
        return {"execution_summary": text, "messages": messages, "reasoning_trace": reasoning, **usage}

    def _evaluate(self, state: MultiAgentState) -> dict[str, Any]:
        text, messages, usage, reasoning = self._role(
            "evaluator",
            "Independently inspect the public artifacts. Explain concrete risks; deterministic verification runs immediately after your review.",
            state,
            context=f"Execution summary:\n{state['execution_summary']}",
            tool_names=("read_file", "list_files"),
            max_tool_calls=self.max_evaluator_tool_calls,
        )
        verification = dict(self.tools_by_name["verify_workspace"].invoke({}))
        if self.shared_memory is not None:
            self.shared_memory.observe_verification(verification)
        passed = verification.get("passed") is True
        repairs = state["repairs"] + (0 if passed else 1)
        terminal = passed or repairs > self.max_repairs
        return {
            "evaluation": text,
            "verification": verification,
            "repairs": repairs,
            "status": "completed" if passed else ("failed" if terminal else "running"),
            "stop_reason": (
                "public_verification_passed"
                if passed
                else ("repair_limit" if terminal else "public_verification_failed")
            ),
            "messages": messages,
            "reasoning_trace": reasoning,
            **usage,
        }

    @staticmethod
    def _after_evaluation(state: MultiAgentState) -> Literal["repair", "end"]:
        return "end" if state["status"] in {"completed", "failed"} else "repair"

    def _role(
        self,
        role: str,
        instruction: str,
        state: MultiAgentState,
        *,
        context: str = "",
        tool_names: tuple[str, ...] = (),
        max_tool_calls: int = 0,
    ) -> tuple[str, list[BaseMessage], dict[str, int], list[dict[str, Any]]]:
        memory_context = ""
        memory_tool_names: tuple[str, ...] = ()
        if self.shared_memory is not None:
            rendered = self.shared_memory.context_for(role)
            if rendered:
                memory_context = f"\n\nRole-conditioned shared memory:\n{rendered}"
            memory_tool_names = self.shared_memory.tool_names_for(role)
            instruction += (
                " Shared memory is bidirectional: use your memory tool when there is "
                "a concrete plan handoff, sourced evidence, or execution progress worth recording."
            )
            required_arguments = self.required_memory_write_arguments.get(role)
            if self.require_memory_writes and required_arguments is not None:
                instruction += (
                    f" For this smoke assertion, call {memory_tool_names[0]} with exactly "
                    f"these arguments and no others: {json.dumps(required_arguments, ensure_ascii=False)}"
                )
        effective_tool_names = tuple(dict.fromkeys((*tool_names, *memory_tool_names)))
        effective_max_tool_calls = max_tool_calls or (8 if memory_tool_names else 0)
        instruction += (
            " Follow this role-level ReAct protocol. Every response starts with `Thought: ` "
            "followed by a one- or two-sentence decision summary, never private chain-of-thought. "
            "Issue at most one native tool call and wait for its runtime Observation before the "
            "next decision. Do not write Action or Observation blocks. Your first response must "
            "be a Thought step or a Thought plus one tool call, not a final answer. Finish only "
            "after at least one completed reasoning step, using `Thought: <evidence-based summary>` "
            "then a new line `Final: <role deliverable>`, with no tool call."
        )
        messages: list[BaseMessage] = [
            SystemMessage(content=f"You are the {role} in a software-engineering team. {instruction}"),
            HumanMessage(
                content=(
                    f"Task ID: {state['task_id']}\n"
                    f"Target software: {state['target_software']}\n"
                    f"Objective:\n{state['objective']}"
                    + (f"\n\nContext:\n{context}" if context else "")
                    + memory_context
                )
            ),
        ]
        selected = [self.tools_by_name[name] for name in effective_tool_names]
        runnable = self.model.bind_tools(selected) if selected else self.model
        generated: list[BaseMessage] = []
        input_tokens = output_tokens = model_turns = tool_calls = 0
        final_text = ""
        memory_write_called = False
        memory_write_reminded = False
        reasoning: list[dict[str, Any]] = []
        thought_only_turns = 0
        protocol_failures = 0
        completed_steps = 0
        while model_turns < self.max_role_model_turns:
            response = runnable.invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError(f"model returned {type(response).__name__}, expected AIMessage")
            model_turns += 1
            current_input, current_output = _usage(response)
            input_tokens += current_input
            output_tokens += current_output
            messages.append(response)
            generated.append(response)
            content = _message_text(response)
            summary, final_candidate = _split_react_content(content)
            status = StepStatus.COMPLETE if final_candidate is not None else StepStatus.CONTINUE
            turn = ModelTurn(
                content=f"Thought: {summary}" if summary is not None else content,
                tool_calls=tuple(
                    ToolCall(str(call["id"]), str(call["name"]), call.get("args") or {})
                    for call in response.tool_calls
                ),
                status=status,
            )
            try:
                decision = self.react.parse(turn)
                if final_candidate is not None and response.tool_calls:
                    raise ReActProtocolError(
                        "action_with_final", "Final cannot accompany a native action."
                    )
            except ReActProtocolError as error:
                protocol_failures += 1
                reasoning.append({
                    "role": role, "kind": "protocol_error", "code": error.code,
                    "summary": str(error),
                })
                if protocol_failures > 2:
                    raise RuntimeError(f"{role} exceeded the ReAct protocol repair limit") from error
                feedback = HumanMessage(
                    content=f"Runtime Observation: ReAct protocol error ({error.code}): {error} Correct the format and retry."
                )
                messages.append(feedback)
                generated.append(feedback)
                continue
            protocol_failures = 0
            reasoning.append({"role": role, "kind": decision.kind, "summary": decision.summary})
            if decision.kind == "finish":
                if completed_steps < 1:
                    reasoning[-1]["kind"] = "thought"
                    reasoning[-1]["proposed_final_rejected"] = True
                    completed_steps += 1
                    feedback = HumanMessage(
                        content="Runtime Observation: a final answer was proposed before a complete reasoning step. Reassess once, then finish."
                    )
                    messages.append(feedback)
                    generated.append(feedback)
                    continue
                if self.require_memory_writes and memory_tool_names and not memory_write_called:
                    if memory_write_reminded:
                        raise RuntimeError(f"{role} did not perform its required memory write")
                    required = self.required_memory_write_arguments.get(role)
                    suffix = f" Use exactly: {json.dumps(required, ensure_ascii=False)}" if required is not None else ""
                    feedback = HumanMessage(
                        content=f"Runtime Observation: call {memory_tool_names[0]} successfully before finishing.{suffix}"
                    )
                    messages.append(feedback)
                    generated.append(feedback)
                    memory_write_reminded = True
                    continue
                final_text = final_candidate or ""
                break
            if decision.kind == "thought":
                thought_only_turns += 1
                completed_steps += 1
                if thought_only_turns > self.react.max_thought_only_turns:
                    reasoning[-1]["kind"] = "finish"
                    reasoning[-1]["runtime_forced_finish"] = True
                    final_text = decision.summary
                    break
                if thought_only_turns == self.react.max_thought_only_turns:
                    feedback = HumanMessage(
                        content=(
                            "Runtime Observation: the thought-only budget is exhausted. "
                            "Your next response must use `Thought: <brief evidence summary>` "
                            "followed by `Final: <role deliverable>`, or take one available native action."
                        )
                    )
                else:
                    feedback = HumanMessage(
                        content="Runtime Observation: decision summary recorded; no action was taken. Reassess using the available evidence and either take one native action or provide Final."
                    )
                messages.append(feedback)
                generated.append(feedback)
                continue
            thought_only_turns = 0
            call = response.tool_calls[0]
            if response.tool_calls:
                tool_calls += 1
                if tool_calls > effective_max_tool_calls:
                    raise RuntimeError(
                        f"{role} exceeded the tool-call limit ({effective_max_tool_calls})"
                    )
                name = str(call["name"])
                tool = self.tools_by_name.get(name)
                if tool is None or name not in effective_tool_names:
                    result: Any = {"error": f"tool is not available to {role}: {name}"}
                else:
                    try:
                        result = tool.invoke(call.get("args") or {})
                    except Exception as error:  # tool failures are observations for repair
                        result = {"error": f"{type(error).__name__}: {error}"}
                if name in memory_tool_names and not (
                    isinstance(result, Mapping) and result.get("error")
                ):
                    memory_write_called = True
                if self.shared_memory is not None:
                    self.shared_memory.observe_tool(
                        role, str(call["id"]), name, call.get("args") or {}, result
                    )
                observation = ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=str(call["id"]),
                    name=name,
                )
                messages.append(observation)
                generated.append(observation)
                reasoning.append({
                    "role": role,
                    "kind": "observation",
                    "tool": name,
                    "summary": _observation_summary(result),
                })
                completed_steps += 1
        else:
            raise RuntimeError(f"{role} exceeded the model-turn limit ({self.max_role_model_turns})")
        return final_text, generated, {
            "model_turns": model_turns,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }, reasoning


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return json.dumps(message.content, ensure_ascii=False, default=str)


def _usage(message: AIMessage) -> tuple[int, int]:
    usage = message.usage_metadata or {}
    token_usage = message.response_metadata.get("token_usage", {})
    return (
        int(usage.get("input_tokens") or token_usage.get("prompt_tokens") or 0),
        int(usage.get("output_tokens") or token_usage.get("completion_tokens") or 0),
    )


def _split_react_content(content: str) -> tuple[str | None, str | None]:
    stripped = content.strip()
    prefix = re.match(r"^Thought:\s*", stripped, flags=re.IGNORECASE)
    if prefix is None:
        return None, None
    body = stripped[prefix.end():]
    final_header = re.search(
        r"(?im)(?:^|\r?\n|\s+)\s*\*{0,2}Final(?:\s+Answer)?\*{0,2}\s*:\s*\*{0,2}\s*",
        body,
    )
    if final_header is None:
        return body.strip(), None
    return body[:final_header.start()].strip(), body[final_header.end():].strip()


def _observation_summary(result: Any) -> str:
    if isinstance(result, Mapping):
        if result.get("error"):
            return f"tool failed: {str(result['error'])[:500]}"
        if "passed" in result:
            return f"verification passed={result.get('passed')}: {str(result.get('summary', ''))[:400]}"
    return "tool returned an observation"
