from __future__ import annotations

from typing import Any, Mapping

from software_bench.harness.contracts import ModelResponse


class MockModelAdapter:
    adapter_id = "mock"

    def generate(
        self,
        messages: list[Mapping[str, Any]],
        system_prompt: str,
        tools: list[Mapping[str, Any]],
        *,
        role_id: str,
        seed: int,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = True,
    ) -> ModelResponse:
        del tool_choice, parallel_tool_calls
        available = {item["name"] for item in tools}
        calls: list[Mapping[str, Any]] = []
        has_tool_result = any(
            '"call_id"' in str(item.get("content", "")) for item in messages
        )
        if not has_tool_result and "message" in available and role_id == "planner":
            calls.append(
                {"name": "message", "args": {"to": "executor", "content": "mock handoff"}}
            )
        if not has_tool_result and "write" in available:
            task_text = str(messages[0].get("content", "")) if messages else ""
            path = (
                "output/mock_solution.txt"
                if "workspace-root directory output/" in task_text
                else "mock_solution.txt"
            )
            calls.append(
                {"name": "write", "args": {"path": path, "content": "fixture solution\n"}}
            )
        elif not has_tool_result and "read" in available:
            calls.append({"name": "read", "args": {"path": "README.md"}})
        if not has_tool_result and "finish_phase" in available:
            calls.append(
                {
                    "name": "finish_phase",
                    "args": {"status": "complete", "summary": "mock phase completed"},
                }
            )
        return ModelResponse(
            text=f"{role_id} completed",
            tool_calls=tuple(calls),
            done=True,
            input_tokens=10,
            output_tokens=5,
        )
