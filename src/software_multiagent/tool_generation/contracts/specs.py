from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from software_multiagent.tool_generation.contracts.errors import ToolGenerationError


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    command: tuple[str, ...]
    timeout_seconds: float = 60.0
    mutating: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolSpec":
        allowed = {
            "name",
            "description",
            "input_schema",
            "command",
            "timeout_seconds",
            "mutating",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ToolGenerationError(f"unknown MCP tool fields: {sorted(unknown)}")
        name = _string(value, "name")
        if not _NAME.fullmatch(name):
            raise ToolGenerationError(f"invalid MCP tool name: {name!r}")
        schema = value.get("input_schema", {})
        if not isinstance(schema, Mapping):
            raise ToolGenerationError("MCP input_schema must be an object")
        normalized_schema = _validate_schema(schema)
        command = value.get("command")
        if (
            not isinstance(command, Sequence)
            or isinstance(command, (str, bytes))
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ToolGenerationError("MCP tool command must be a non-empty string array")
        timeout = value.get("timeout_seconds", 60.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ToolGenerationError("MCP timeout_seconds must be a number")
        if timeout <= 0 or timeout > 1800:
            raise ToolGenerationError("MCP timeout_seconds must be in (0, 1800]")
        mutating = value.get("mutating", False)
        if not isinstance(mutating, bool):
            raise ToolGenerationError("MCP mutating must be a boolean")
        _validate_templates(tuple(command), normalized_schema)
        return cls(
            name,
            _string(value, "description"),
            normalized_schema,
            tuple(command),
            float(timeout),
            mutating,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "command": list(self.command),
            "timeout_seconds": self.timeout_seconds,
            "mutating": self.mutating,
        }


@dataclass(frozen=True)
class ApplicationSpec:
    name: str
    version: str
    description: str
    tools: tuple[ToolSpec, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApplicationSpec":
        allowed = {"name", "version", "description", "tools"}
        unknown = set(value) - allowed
        if unknown:
            raise ToolGenerationError(
                f"unknown MCP application fields: {sorted(unknown)}"
            )
        raw_tools = value.get("tools")
        if not isinstance(raw_tools, list) or not raw_tools:
            raise ToolGenerationError("MCP application tools must be a non-empty array")
        if not all(isinstance(item, Mapping) for item in raw_tools):
            raise ToolGenerationError("each MCP tool must be an object")
        tools = tuple(ToolSpec.from_dict(item) for item in raw_tools)
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ToolGenerationError("MCP tool names must be unique")
        return cls(
            _string(value, "name"),
            _string(value, "version"),
            _string(value, "description"),
            tools,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tools": [tool.to_dict() for tool in self.tools],
        }


def load_spec(path: str | Path) -> ApplicationSpec:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolGenerationError(f"cannot read MCP application spec {source}: {error}") from error
    if not isinstance(value, Mapping):
        raise ToolGenerationError("MCP application spec must contain an object")
    return ApplicationSpec.from_dict(value)


def write_spec(spec: ApplicationSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_arguments(
    schema: Mapping[str, Any], arguments: Mapping[str, Any]
) -> None:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ToolGenerationError(f"missing MCP tool arguments: {sorted(missing)}")
    if schema.get("additionalProperties", True) is False:
        unknown = set(arguments) - set(properties)
        if unknown:
            raise ToolGenerationError(f"unknown MCP tool arguments: {sorted(unknown)}")
    for name, value in arguments.items():
        item = properties.get(name)
        if not isinstance(item, Mapping):
            continue
        _validate_argument(name, value, item)


def render_command(tool: ToolSpec, arguments: Mapping[str, Any]) -> list[str]:
    def replace(token: str) -> str:
        return re.sub(
            r"\{([A-Za-z][A-Za-z0-9_]*)\}",
            lambda match: str(arguments[match.group(1)]),
            token,
        )

    rendered: list[str] = []
    for token in tool.command:
        expanded = re.fullmatch(r"\{([A-Za-z][A-Za-z0-9_]*)\.\.\.\}", token)
        if expanded:
            rendered.extend(str(item) for item in arguments[expanded.group(1)])
        else:
            rendered.append(replace(token))
    return rendered


def _validate_argument(name: str, value: Any, item: Mapping[str, Any]) -> None:
    expected = item.get("type")
    valid = (
        (expected == "string" and isinstance(value, str))
        or (
            expected == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (expected == "boolean" and isinstance(value, bool))
        or (expected == "array" and isinstance(value, list))
    )
    if expected is not None and not valid:
        raise ToolGenerationError(f"MCP argument {name!r} must have type {expected}")
    if isinstance(value, str):
        if item.get("minLength", 0) > len(value):
            raise ToolGenerationError(f"MCP argument {name!r} is too short")
        choices = item.get("enum")
        if choices is not None and value not in choices:
            raise ToolGenerationError(f"MCP argument {name!r} must be one of {choices}")
        if item.get("format") == "workspace-path":
            normalized = value.replace("\\", "/")
            if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
                raise ToolGenerationError(
                    f"MCP argument {name!r} must be workspace-relative"
                )
            if ".." in normalized.split("/"):
                raise ToolGenerationError(f"MCP argument {name!r} escapes the workspace")
    if isinstance(value, list):
        if len(value) < item.get("minItems", 0):
            raise ToolGenerationError(f"MCP argument {name!r} has too few items")
        item_schema = item.get("items", {})
        if (
            isinstance(item_schema, Mapping)
            and item_schema.get("type") == "string"
            and not all(isinstance(entry, str) for entry in value)
        ):
            raise ToolGenerationError(
                f"MCP argument {name!r} must contain only strings"
            )


def _validate_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.setdefault("type", "object")
    result.setdefault("properties", {})
    result.setdefault("additionalProperties", False)
    if result["type"] != "object" or not isinstance(result["properties"], Mapping):
        raise ToolGenerationError("MCP input_schema must describe an object")
    required = result.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ToolGenerationError("MCP input_schema.required must be a string array")
    if set(required) - set(result["properties"]):
        raise ToolGenerationError("MCP required arguments must exist in properties")
    return result


def _validate_templates(command: tuple[str, ...], schema: Mapping[str, Any]) -> None:
    fields = set(schema.get("properties", {}))
    referenced: set[str] = set()
    expanded: set[str] = set()
    for token in command:
        for match in re.finditer(r"\{([A-Za-z][A-Za-z0-9_]*)(\.\.\.)?\}", token):
            referenced.add(match.group(1))
            if match.group(2):
                if match.group(0) != token:
                    raise ToolGenerationError(
                        "expanded command arguments must occupy one token"
                    )
                expanded.add(match.group(1))
    unknown = referenced - fields
    if unknown:
        raise ToolGenerationError(
            f"command references undeclared arguments: {sorted(unknown)}"
        )
    optional = referenced - set(schema.get("required", []))
    if optional:
        raise ToolGenerationError(
            f"command placeholders must be required arguments: {sorted(optional)}"
        )
    properties = schema.get("properties", {})
    invalid = {
        name
        for name in expanded
        if not isinstance(properties.get(name), Mapping)
        or properties[name].get("type") != "array"
    }
    if invalid:
        raise ToolGenerationError(
            f"expanded command arguments must have array schemas: {sorted(invalid)}"
        )


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ToolGenerationError(f"{key} must be a non-empty string")
    return item
