from __future__ import annotations

import json
import re
import sys
from typing import Any, BinaryIO, Mapping, TextIO

from software_bench.core.models import ValidationError
from software_bench.harness.environments import EnvironmentSession
from software_bench.mcp.models import ApplicationSpec, ToolSpec, validate_arguments


class ApplicationServer:
    def __init__(
        self,
        spec: ApplicationSpec,
        environment: EnvironmentSession,
        *,
        allow_mutations: bool = False,
    ) -> None:
        self.spec = spec
        self.environment = environment
        self.allow_mutations = allow_mutations
        self._tools = {
            tool.name: tool for tool in spec.tools if allow_mutations or not tool.mutating
        }

    def declarations(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": dict(tool.input_schema),
                "annotations": {
                    "readOnlyHint": not tool.mutating,
                    "destructiveHint": tool.mutating,
                    "idempotentHint": not tool.mutating,
                    "openWorldHint": False,
                },
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            if any(item.name == name and item.mutating for item in self.spec.tools):
                raise PermissionError(f"MCP tool {name!r} requires mutation permission")
            raise LookupError(f"unknown MCP tool: {name}")
        validate_arguments(tool.input_schema, arguments)
        command = _render(tool, arguments)
        result = self.environment.run(command, timeout_seconds=tool.timeout_seconds)
        structured = {
            "command": command,
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-4000:],
            "exit_code": result.exit_code,
        }
        text = json.dumps(structured, ensure_ascii=False)
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": structured,
            "isError": result.exit_code != 0,
        }


class JsonRpcServer:
    """Protocol dispatcher kept transport-neutral for tests and embedding."""

    def __init__(self, application: ApplicationServer) -> None:
        self.application = application

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str):
            return _error(request_id, -32600, "invalid JSON-RPC request")
        if method.startswith("notifications/"):
            return None
        try:
            if method == "initialize":
                params = request.get("params", {})
                requested = params.get("protocolVersion") if isinstance(params, Mapping) else None
                result = {
                    "protocolVersion": requested or "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": self.application.spec.name,
                        "version": self.application.spec.version,
                    },
                    "instructions": self.application.spec.description,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.application.declarations()}
            elif method == "tools/call":
                params = request.get("params", {})
                if not isinstance(params, Mapping) or not isinstance(params.get("name"), str):
                    raise ValidationError("tools/call requires a tool name")
                arguments = params.get("arguments", {})
                if not isinstance(arguments, Mapping):
                    raise ValidationError("tools/call arguments must be an object")
                result = self.application.call(params["name"], arguments)
            else:
                return _error(request_id, -32601, f"method not found: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (LookupError, PermissionError, ValidationError, ValueError) as error:
            return _error(request_id, -32602, str(error))
        except Exception as error:  # keep the stdio server alive after application failures
            return _error(request_id, -32603, f"application tool failed: {error}")


def serve_stdio(
    server: JsonRpcServer,
    *,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
) -> None:
    """Serve newline-delimited or Content-Length framed JSON-RPC over stdio."""
    source = input_stream or sys.stdin.buffer
    sink = output_stream or sys.stdout.buffer
    try:
        while True:
            payload, framed = _read_message(source)
            if payload is None:
                break
            try:
                request = json.loads(payload)
                if not isinstance(request, Mapping):
                    raise ValueError("request root must be an object")
                response = server.handle(request)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                response = _error(None, -32700, f"parse error: {error}")
            if response is not None:
                _write_message(sink, response, framed=framed)
    finally:
        server.application.environment.close()


def _render(tool: ToolSpec, arguments: Mapping[str, Any]) -> list[str]:
    def replace(token: str) -> str:
        def value(match: re.Match[str]) -> str:
            return str(arguments[match.group(1)])

        return re.sub(r"\{([A-Za-z][A-Za-z0-9_]*)\}", value, token)

    rendered: list[str] = []
    for token in tool.command:
        expanded = re.fullmatch(r"\{([A-Za-z][A-Za-z0-9_]*)\.\.\.\}", token)
        if expanded:
            rendered.extend(str(item) for item in arguments[expanded.group(1)])
        else:
            rendered.append(replace(token))
    return rendered


def _read_message(source: BinaryIO) -> tuple[str | None, bool]:
    first = source.readline()
    if not first:
        return None, False
    if first.lower().startswith(b"content-length:"):
        try:
            length = int(first.split(b":", 1)[1].strip())
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        while True:
            line = source.readline()
            if line in {b"\r\n", b"\n", b""}:
                break
        return source.read(length).decode("utf-8"), True
    return first.decode("utf-8").strip(), False


def _write_message(sink: BinaryIO, response: Mapping[str, Any], *, framed: bool) -> None:
    body = json.dumps(response, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if framed:
        sink.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sink.write(body + (b"" if framed else b"\n"))
    sink.flush()


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
