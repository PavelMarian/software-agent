from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from software_multiagent.tool_generation.contracts.errors import ToolGenerationError
from software_multiagent.tool_generation.contracts.specs import (
    ApplicationSpec,
    ToolSpec,
    validate_arguments,
)


ValidationError = ToolGenerationError


@dataclass(frozen=True)
class ExpectedResult:
    exit_code: int = 0
    stdout_contains: tuple[str, ...] = ()
    stderr_contains: tuple[str, ...] = ()
    created_paths: tuple[str, ...] = ()
    stdout_json: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpectedResult":
        allowed = {
            "exit_code", "stdout_contains", "stderr_contains", "created_paths", "stdout_json"
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValidationError(f"unknown expected-result fields: {sorted(unknown)}")
        exit_code = value.get("exit_code", 0)
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValidationError("expected.exit_code must be an integer")
        stdout_json = value.get("stdout_json", False)
        if not isinstance(stdout_json, bool):
            raise ValidationError("expected.stdout_json must be a boolean")
        return cls(
            exit_code=exit_code,
            stdout_contains=_strings(value.get("stdout_contains", []), "stdout_contains"),
            stderr_contains=_strings(value.get("stderr_contains", []), "stderr_contains"),
            created_paths=_workspace_paths(value.get("created_paths", [])),
            stdout_json=stdout_json,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout_contains": list(self.stdout_contains),
            "stderr_contains": list(self.stderr_contains),
            "created_paths": list(self.created_paths),
            "stdout_json": self.stdout_json,
        }


@dataclass(frozen=True)
class WrapperRecipe:
    """Definition for a generated command, Python-callable, or HTTP wrapper."""

    kind: str
    config: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WrapperRecipe":
        """Parse a wrapper definition from a recipe object.

        Args:
            value: Wrapper mapping with a ``kind`` discriminator.

        Returns:
            A validated wrapper recipe.
        """

        allowed = {"kind", "module", "callable", "url", "method", "headers", "steps"}
        unknown = set(value) - allowed
        if unknown:
            raise ValidationError(f"unknown wrapper fields: {sorted(unknown)}")
        kind = _nonempty(value.get("kind"), "wrapper kind")
        if kind not in {"python", "http", "pipeline"}:
            raise ValidationError("wrapper kind must be python, http, or pipeline")
        required = {
            "python": ("module", "callable"),
            "http": ("url",),
            "pipeline": ("steps",),
        }[kind]
        for key in required:
            if key not in value:
                raise ValidationError(f"{kind} wrapper requires {key}")
        if kind == "python":
            _nonempty(value.get("module"), "wrapper module")
            _nonempty(value.get("callable"), "wrapper callable")
        if kind == "http":
            url = _nonempty(value.get("url"), "wrapper url")
            if not url.startswith(("http://", "https://")):
                raise ValidationError("wrapper url must use http or https")
            method = value.get("method", "POST")
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                raise ValidationError("unsupported wrapper HTTP method")
            headers = value.get("headers", {})
            if not isinstance(headers, Mapping) or not all(
                isinstance(key, str) and isinstance(item, str) for key, item in headers.items()
            ):
                raise ValidationError("wrapper headers must be a string mapping")
        if kind == "pipeline":
            steps = value.get("steps")
            if not isinstance(steps, list) or not steps or not all(
                isinstance(step, list)
                and step
                and all(isinstance(token, str) and token for token in step)
                for step in steps
            ):
                raise ValidationError("pipeline wrapper steps must be non-empty argv arrays")
        return cls(kind, {key: item for key, item in value.items() if key != "kind"})

    def to_dict(self) -> dict[str, Any]:
        """Return the wrapper definition as a JSON object."""

        return {"kind": self.kind, **dict(self.config)}


@dataclass(frozen=True)
class ToolRecipe:
    spec: ToolSpec
    source: str
    evidence: str
    sample_arguments: Mapping[str, Any] | None = None
    expected: ExpectedResult = field(default_factory=ExpectedResult)
    wrapper: WrapperRecipe | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolRecipe":
        extra = {"source", "evidence", "sample_arguments", "expected", "wrapper"}
        spec_keys = {
            "name", "description", "input_schema", "command", "timeout_seconds", "mutating"
        }
        unknown = set(value) - spec_keys - extra
        if unknown:
            raise ValidationError(f"unknown tool recipe fields: {sorted(unknown)}")
        source = _nonempty(value.get("source"), "tool source")
        evidence = value.get("evidence", "")
        if not isinstance(evidence, str):
            raise ValidationError("tool evidence must be a string")
        sample = value.get("sample_arguments")
        if sample is not None and not isinstance(sample, Mapping):
            raise ValidationError("sample_arguments must be an object or null")
        expected_value = value.get("expected", {})
        if not isinstance(expected_value, Mapping):
            raise ValidationError("expected must be an object")
        wrapper_value = value.get("wrapper")
        if wrapper_value is not None and not isinstance(wrapper_value, Mapping):
            raise ValidationError("wrapper must be an object or null")
        if wrapper_value is not None and "command" in value:
            raise ValidationError("tool must define either command or wrapper, not both")
        if wrapper_value is None and "command" not in value:
            raise ValidationError("tool must define command or wrapper")
        wrapper = WrapperRecipe.from_dict(wrapper_value) if wrapper_value is not None else None
        spec_value = {key: value[key] for key in spec_keys if key in value}
        if wrapper is not None:
            spec_value["command"] = ["python", "__generated_wrapper__.py"]
        spec = ToolSpec.from_dict(spec_value)
        if sample is not None:
            validate_arguments(spec.input_schema, sample)
        return cls(
            spec=spec,
            source=source,
            evidence=evidence.strip(),
            sample_arguments=dict(sample) if sample is not None else None,
            expected=ExpectedResult.from_dict(expected_value),
            wrapper=wrapper,
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            **self.spec.to_dict(),
            "source": self.source,
            "evidence": self.evidence,
            "sample_arguments": (
                dict(self.sample_arguments) if self.sample_arguments is not None else None
            ),
            "expected": self.expected.to_dict(),
        }
        if self.wrapper is not None:
            value.pop("command", None)
            value["wrapper"] = self.wrapper.to_dict()
        return value


@dataclass(frozen=True)
class ApplicationRecipe:
    application: ApplicationSpec
    tools: tuple[ToolRecipe, ...]
    required_executables: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApplicationRecipe":
        allowed = {"application", "required_executables", "tools"}
        unknown = set(value) - allowed
        if unknown:
            raise ValidationError(f"unknown application recipe fields: {sorted(unknown)}")
        application = value.get("application")
        if not isinstance(application, Mapping):
            raise ValidationError("application must be an object")
        raw_tools = value.get("tools")
        if not isinstance(raw_tools, list) or not raw_tools:
            raise ValidationError("tools must be a non-empty array")
        if not all(isinstance(item, Mapping) for item in raw_tools):
            raise ValidationError("each tool recipe must be an object")
        tools = tuple(ToolRecipe.from_dict(item) for item in raw_tools)
        names = [tool.spec.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValidationError("tool recipe names must be unique")
        app_value = dict(application)
        app_value["tools"] = [tool.spec.to_dict() for tool in tools]
        spec = ApplicationSpec.from_dict(app_value)
        required = _strings(value.get("required_executables", []), "required_executables")
        if any("/" in item or "\\" in item for item in required):
            raise ValidationError("required_executables must contain executable names, not paths")
        return cls(spec, tools, required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "application": {
                "name": self.application.name,
                "version": self.application.version,
                "description": self.application.description,
            },
            "required_executables": list(self.required_executables),
            "tools": [tool.to_dict() for tool in self.tools],
        }


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValidationError(f"{label} must be a string array")
    return tuple(value)


def _workspace_paths(value: Any) -> tuple[str, ...]:
    paths = _strings(value, "created_paths")
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValidationError("expected.created_paths must be workspace-relative")
    return paths
