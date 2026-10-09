from __future__ import annotations

from typing import Any, Mapping

from software_bench.harness.environments import EnvironmentSession, ExecutionResult
from software_bench.mcp.models import ApplicationSpec
from software_bench.mcp.server import ApplicationServer


class McpEnvironmentSession:
    """Decorator that makes MCP application tools visible to benchmark roles."""

    def __init__(
        self,
        environment: EnvironmentSession,
        spec: ApplicationSpec,
        *,
        allow_mutations: bool = False,
    ) -> None:
        self._environment = environment
        self._application = ApplicationServer(
            spec, environment, allow_mutations=allow_mutations
        )
        self.backend_id = f"{environment.backend_id}+mcp:{spec.name}"

    def tool_declarations(self) -> Mapping[str, Mapping[str, Any]]:
        existing = dict(self._environment.tool_declarations())
        additions = {
            item["name"]: {
                "name": item["name"],
                "description": item["description"],
                "parameters": item["inputSchema"],
                "annotations": item["annotations"],
            }
            for item in self._application.declarations()
        }
        overlap = set(existing) & set(additions)
        if overlap:
            raise ValueError(f"MCP tools shadow environment tools: {sorted(overlap)}")
        return {**existing, **additions}

    def invoke_tool(self, name: str, arguments: Mapping[str, Any]) -> ExecutionResult:
        if name in {item["name"] for item in self._application.declarations()}:
            result = self._application.call(name, arguments)["structuredContent"]
            return ExecutionResult(result["stdout"], result["stderr"], result["exit_code"])
        return self._environment.invoke_tool(name, arguments)

    def close(self) -> None:
        self._environment.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)
