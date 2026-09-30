"""Pre-execution schema validation for the opt-in reliable solo runtime."""
from typing import Any, Mapping


def _validate_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Recursive JSON-schema subset; unsupported validation keywords fail closed."""
    import math

    supported = {"type", "properties", "required", "additionalProperties", "items",
                 "enum", "const", "minLength", "maxLength", "minItems", "maxItems",
                 "minimum", "maximum", "description", "title", "default", "$schema"}

    def validate(value: Any, rule: Mapping[str, Any], path: str) -> None:
        unsupported = set(rule) - supported
        if unsupported:
            raise ValueError(f"unsupported schema keywords at {path}: {sorted(unsupported)}")
        kinds = {
            "string": isinstance(value, str),
            "array": isinstance(value, (list, tuple)),
            "object": isinstance(value, Mapping),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool), "null": value is None,
        }
        expected = rule.get("type")
        if expected is not None:
            types = expected if isinstance(expected, list) else [expected]
            if not any(kinds.get(t, False) for t in types):
                raise ValueError(f"{path} must have type {expected}")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"{path} is not one of the allowed values")
        if "const" in rule and value != rule["const"]:
            raise ValueError(f"{path} does not match const")
        if isinstance(value, Mapping):
            missing = set(rule.get("required", ())) - set(value)
            if missing:
                raise ValueError(f"missing required arguments at {path}: {sorted(missing)}")
            properties = rule.get("properties", {})
            for key, item in value.items():
                if key in properties:
                    validate(item, properties[key], f"{path}.{key}")
                elif rule.get("additionalProperties") is False:
                    raise ValueError(f"unknown argument: {path}.{key}")
                elif isinstance(rule.get("additionalProperties"), Mapping):
                    validate(item, rule["additionalProperties"], f"{path}.{key}")
        if isinstance(value, (list, tuple)):
            if not rule.get("minItems", 0) <= len(value) <= rule.get("maxItems", float('inf')):
                raise ValueError(f"invalid array length: {path}")
            for index, item in enumerate(value):
                if "items" in rule:
                    validate(item, rule["items"], f"{path}[{index}]")
        if isinstance(value, str):
            if not rule.get("minLength", 0) <= len(value) <= rule.get("maxLength", float('inf')):
                raise ValueError(f"invalid string length: {path}")
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite number: {path}")
            if not rule.get("minimum", -float('inf')) <= value <= rule.get("maximum", float('inf')):
                raise ValueError(f"number outside bounds: {path}")

    if not isinstance(arguments, Mapping):
        raise ValueError("arguments must be an object")
    validate(arguments, schema, "arguments")


