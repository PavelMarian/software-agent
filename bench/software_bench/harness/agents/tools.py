from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from software_bench.core.models import RoleSpec, WorkspaceAccess
from software_bench.harness.environments import EnvironmentSession


class EventRecorder(Protocol):
    """Define the trace and wall-time services required by tool execution."""

    def record(
        self,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
    ) -> None:
        """Record a normalized benchmark event and its resource usage.

        Args:
            event_type: Stable trace event type.
            actor: Role or harness component producing the event.
            payload: Optional JSON-compatible event fields.
            input_tokens: Input tokens charged to the event.
            output_tokens: Output tokens charged to the event.
            tool_calls: Tool calls charged to the event.
        """
        ...

    def remaining_wall_time(self) -> float:
        """Return the remaining aggregate wall-time budget in seconds."""
        ...


@dataclass(frozen=True)
class ToolResult:
    """Represent normalized output from a harness-controlled tool call.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Process-style result code, where zero indicates success.
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for model history and trace payloads.

        Returns:
            A JSON-compatible result mapping.
        """
        return {"stdout": self.stdout, "stderr": self.stderr, "exit_code": self.exit_code}


class MessageBus:
    """Store in-run messages addressed to configured team roles."""

    def __init__(self, role_ids: set[str]) -> None:
        """Initialize an empty message bus.

        Args:
            role_ids: Valid message recipient identifiers.
        """
        self.role_ids = role_ids
        self.messages: list[dict[str, str]] = []

    def send(self, sender: str, recipient: str, content: str) -> dict[str, str]:
        """Append a message with a stable trace identifier.

        Args:
            sender: Sending role identifier.
            recipient: Receiving role identifier.
            content: Message body.

        Returns:
            The stored message record.

        Raises:
            PermissionError: If the recipient is not a configured role.
        """
        if recipient not in self.role_ids:
            raise PermissionError(f"unknown message recipient: {recipient}")
        message = {
            "id": f"message-{len(self.messages)}",
            "from": sender,
            "to": recipient,
            "content": content,
        }
        self.messages.append(message)
        return message

    def for_role(self, role_id: str) -> list[dict[str, str]]:
        """Return all messages addressed to a role in send order.

        Args:
            role_id: Recipient role identifier.

        Returns:
            Stored messages whose recipient matches ``role_id``.
        """
        return [message for message in self.messages if message["to"] == role_id]


class ToolExecutor:
    """Enforce role permissions while executing and tracing benchmark tools."""

    def __init__(
        self,
        environment: EnvironmentSession,
        roles: Mapping[str, RoleSpec],
        communication_enabled: bool,
        observer: EventRecorder,
        messages: MessageBus,
    ) -> None:
        """Initialize the executor and its environment dependencies.

        Args:
            environment: Workspace and application execution backend.
            roles: Role specifications keyed by role identifier.
            communication_enabled: Whether message calls are permitted.
            observer: Event recorder and aggregate wall-time source.
            messages: Shared in-run message bus.
        """
        self.environment = environment
        self.roles = roles
        self.communication_enabled = communication_enabled
        self.observer = observer
        self.messages = messages

    def declarations(self, role: RoleSpec) -> list[dict[str, Any]]:
        """Build the tool declarations visible to a role.

        Args:
            role: Role whose built-in and environment tools are requested.

        Returns:
            Model-facing tool declaration mappings.

        Raises:
            ValueError: If an environment tool shadows a built-in tool.
        """
        declarations = [_DECLARATIONS[name] for name in role.tools if name != "environment"]
        if "environment" in role.tools:
            dynamic = getattr(self.environment, "tool_declarations", lambda: {})()
            overlap = set(dynamic) & set(_DECLARATIONS)
            if overlap:
                raise ValueError(f"environment tools shadow built-ins: {sorted(overlap)}")
            declarations.extend(dynamic.values())
        return declarations

    def declarations_for(self, role_id: str) -> list[dict[str, Any]]:
        """Build tool declarations for a role identifier.

        Args:
            role_id: Configured role identifier.

        Returns:
            Model-facing declarations allowed for that role.
        """
        return self.declarations(self.roles[role_id])

    def execute(self, role_id: str, name: str, args: Mapping[str, Any]) -> ToolResult:
        """Execute one authorized tool call and record its request and result.

        Args:
            role_id: Calling role identifier.
            name: Tool name.
            args: Validated or model-supplied tool arguments.

        Returns:
            Normalized tool output.

        Raises:
            PermissionError: If the role cannot call the requested tool.
            ValueError: If the tool is unknown or its arguments are invalid.
        """
        role = self.roles[role_id]
        request_payload = {"tool": name, "args": dict(args)}
        self.observer.record("tool_called", role_id, request_payload, tool_calls=1)
        dynamic = (
            getattr(self.environment, "tool_declarations", lambda: {})()
            if "environment" in role.tools
            else {}
        )
        if name not in role.tools and name not in dynamic:
            raise PermissionError(f"role {role_id} cannot use tool {name}")
        if name == "read":
            result = self._read(role, args)
        elif name == "list":
            result = self._list(role, args)
        elif name == "write":
            result = self._write(role, args)
        elif name == "write_files":
            result = self._write_files(role, args)
        elif name == "run":
            result = self._run(args)
        elif name == "message":
            result = self._message(role, args)
        elif name == "finish_phase":
            result = self._finish_phase(args)
        elif name in dynamic:
            completed = self.environment.invoke_tool(name, args)
            result = ToolResult(
                completed.stdout[-8000:], completed.stderr[-4000:], completed.exit_code
            )
        else:
            raise ValueError(f"unknown tool: {name}")
        payload = {"tool": name, "args": dict(args), "result": result.to_dict()}
        self.observer.record("tool_result", role_id, payload)
        return result

    def _read(self, role: RoleSpec, args: Mapping[str, Any]) -> ToolResult:
        """Read a workspace text file under a role's access policy."""
        if role.workspace_access == WorkspaceAccess.NONE:
            raise PermissionError(f"role {role.id} cannot read the workspace")
        content = self.environment.read_text(_argument_string(args, "path"))
        return ToolResult(stdout=content[-8000:])

    def _write(self, role: RoleSpec, args: Mapping[str, Any]) -> ToolResult:
        """Replace one workspace text file under a role's access policy."""
        if role.workspace_access != WorkspaceAccess.READ_WRITE:
            raise PermissionError(f"role {role.id} cannot write the workspace")
        path = _argument_string(args, "path")
        content = _argument_string(args, "content", allow_empty=True)
        self.environment.write_text(path, content)
        return ToolResult(stdout=f"wrote {len(content)} bytes")

    def _write_files(self, role: RoleSpec, args: Mapping[str, Any]) -> ToolResult:
        """Replace multiple workspace text files in one authorized call."""
        if role.workspace_access != WorkspaceAccess.READ_WRITE:
            raise PermissionError(f"role {role.id} cannot write the workspace")
        files = args.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ValueError("write_files.files must be a non-empty object")
        if not all(
            isinstance(path, str) and path and isinstance(content, str)
            for path, content in files.items()
        ):
            raise ValueError("write_files.files must map paths to string content")
        for path, content in files.items():
            self.environment.write_text(path, content)
        return ToolResult(stdout=f"wrote {len(files)} files")

    def _list(self, role: RoleSpec, args: Mapping[str, Any]) -> ToolResult:
        """List workspace files under a role's access policy."""
        if role.workspace_access == WorkspaceAccess.NONE:
            raise PermissionError(f"role {role.id} cannot list the workspace")
        path = args.get("path", ".")
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        return ToolResult(stdout="\n".join(self.environment.list_files(path)[:2000]))

    def _run(self, args: Mapping[str, Any]) -> ToolResult:
        """Run an argv command within the remaining aggregate time budget."""
        command = args.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError("run.command must be a non-empty string array")
        timeout = args.get("timeout_seconds", 60)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("run.timeout_seconds must be a number")
        if timeout <= 0 or timeout > 1800:
            raise ValueError("run.timeout_seconds must be in (0, 1800]")
        timeout = min(float(timeout), self.observer.remaining_wall_time())
        if timeout <= 0:
            raise TimeoutError("team wall-time budget exhausted")
        completed = self.environment.run(command, timeout_seconds=timeout)
        return ToolResult(completed.stdout[-8000:], completed.stderr[-4000:], completed.exit_code)

    def _message(self, role: RoleSpec, args: Mapping[str, Any]) -> ToolResult:
        """Send and trace one authorized teammate handoff message."""
        if not self.communication_enabled or not role.can_communicate:
            raise PermissionError(f"role {role.id} cannot communicate in this mode")
        recipient = _argument_string(args, "to")
        content = _argument_string(args, "content")
        message = self.messages.send(role.id, recipient, content)
        self.observer.record(
            "message",
            role.id,
            {
                "message_id": message["id"],
                "recipient": recipient,
                "content": content,
            },
        )
        return ToolResult(stdout=f"message sent to {recipient}")

    @staticmethod
    def _finish_phase(args: Mapping[str, Any]) -> ToolResult:
        """Validate and normalize an explicit phase-completion result."""
        status = _argument_string(args, "status")
        if status not in {"complete", "needs_revision", "failed"}:
            raise ValueError("finish_phase.status has an unsupported value")
        summary = args.get("summary", "")
        if not isinstance(summary, str):
            raise ValueError("finish_phase.summary must be a string")
        return ToolResult(stdout=f"phase finished with status={status}: {summary}"[-8000:])

def _argument_string(args: Mapping[str, Any], key: str, allow_empty: bool = False) -> str:
    """Read a required string argument from a tool call.

    Args:
        args: Tool argument mapping.
        key: Argument name to retrieve.
        allow_empty: Whether an empty string is accepted.

    Returns:
        The validated string value.

    Raises:
        ValueError: If the argument is missing or has an invalid value.
    """
    value = args.get(key)
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValueError(f"{key} must be a string")
    return value


_DECLARATIONS: dict[str, dict[str, Any]] = {
    "list": {
        "name": "list",
        "description": "List files recursively below a workspace-relative path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "default": "."}
            },
            "additionalProperties": False,
        },
    },
    "read": {
        "name": "read",
        "description": "Read a UTF-8 text file relative to the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "description": "Relative path."}
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "write": {
        "name": "write",
        "description": "Replace a UTF-8 text file in the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "description": "Relative path."},
                "content": {"type": "string", "description": "Complete new file content."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    "write_files": {
        "name": "write_files",
        "description": "Replace several UTF-8 workspace files in one tool call.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {"type": "string"},
                }
            },
            "required": ["files"],
            "additionalProperties": False,
        },
    },
    "run": {
        "name": "run",
        "description": "Run one argv command in the task workspace and return process output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "description": "Executable followed by its arguments; no shell string.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1800,
                    "description": (
                        "Per-command timeout; solver runs may need more than 60 seconds."
                    ),
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    "message": {
        "name": "message",
        "description": "Send a concise handoff, finding, or blocker to another configured role.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "minLength": 1, "description": "Recipient role ID."},
                "content": {"type": "string", "minLength": 1, "description": "Message body."},
            },
            "required": ["to", "content"],
            "additionalProperties": False,
        },
    },
    "finish_phase": {
        "name": "finish_phase",
        "description": (
            "Finish the current workflow phase explicitly. Use complete when the assigned "
            "work is done, needs_revision when verification found concrete defects, or "
            "failed for an unrecoverable blocker."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["complete", "needs_revision", "failed"],
                },
                "summary": {
                    "type": "string",
                    "description": "Concise result, evidence, or blocker for later roles.",
                },
            },
            "required": ["status", "summary"],
            "additionalProperties": False,
        },
    },
}
