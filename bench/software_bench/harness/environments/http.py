from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Callable, Mapping

from software_bench.core.models import ValidationError
if TYPE_CHECKING:
    from software_bench.harness.environments import ExecutionResult


class ConfiguredHttpTools:
    """Expose trusted HTTP control-plane actions through one generic contract."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        write_text: Callable[[str, str], None],
    ) -> None:
        raw = config.get("http_tools")
        self._tools: dict[str, Mapping[str, Any]] = {}
        self._base_url = ""
        self._timeout = 30.0
        self._write_text = write_text
        if raw is None:
            return
        if not isinstance(raw, Mapping):
            raise ValidationError("backend_config.http_tools must be an object")
        base_url = raw.get("base_url")
        tools = raw.get("tools")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValidationError("http_tools.base_url must be an HTTP URL")
        if not isinstance(tools, list):
            raise ValidationError("http_tools.tools must be an array")
        timeout = raw.get("timeout_seconds", 30)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValidationError("http_tools.timeout_seconds must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        for item in tools:
            if not isinstance(item, Mapping):
                raise ValidationError("every HTTP tool must be an object")
            name = item.get("name")
            parameters = item.get("parameters")
            if not isinstance(name, str) or not name or not isinstance(parameters, Mapping):
                raise ValidationError("HTTP tools require name and parameters")
            if name in self._tools:
                raise ValidationError(f"duplicate HTTP tool: {name}")
            self._tools[name] = item

    def declarations(self) -> Mapping[str, Mapping[str, Any]]:
        return {
            name: {
                "name": name,
                "description": str(item.get("description", "Environment action.")),
                "parameters": dict(item["parameters"]),
                "mutates_workspace": bool(item.get("mutates_workspace", False)),
            }
            for name, item in self._tools.items()
        }

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> ExecutionResult:
        from software_bench.harness.environments import ExecutionResult
        item = self._tools.get(name)
        if item is None:
            raise ValueError(f"unknown environment tool: {name}")
        self._wait(item.get("wait_for"))
        method = str(item.get("method", "GET")).upper()
        path = item.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValidationError(f"HTTP tool {name} requires an absolute URL path")
        body_names = item.get("body_arguments", [])
        query_names = item.get("query_arguments", [])
        if not isinstance(body_names, list) or not isinstance(query_names, list):
            raise ValidationError(f"HTTP tool {name} argument mappings must be arrays")
        allowed = set(body_names) | set(query_names)
        unknown = set(arguments) - allowed
        if unknown:
            raise ValueError(f"unknown arguments for {name}: {sorted(unknown)}")
        missing = [key for key in allowed if key not in arguments]
        if missing:
            raise ValueError(f"missing arguments for {name}: {sorted(missing)}")
        query = urllib.parse.urlencode({key: arguments[key] for key in query_names})
        url = self._base_url + path + (f"?{query}" if query else "")
        data = None
        headers: dict[str, str] = {}
        if body_names:
            data = json.dumps({key: arguments[key] for key in body_names}).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=data, headers=headers, method=method),
                timeout=self._timeout,
            ) as response:
                text = response.read().decode("utf-8", errors="replace")
                code = response.status
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", errors="replace")
            return ExecutionResult("", text, error.code)
        except urllib.error.URLError as error:
            return ExecutionResult("", str(error.reason), 1)
        record_path = item.get("record_path")
        if isinstance(record_path, str) and record_path:
            path_args = {key: str(value) for key, value in arguments.items()}
            recorded = text
            if item.get("record_request") is True:
                try:
                    response_value: Any = json.loads(text)
                except json.JSONDecodeError:
                    response_value = text
                recorded = json.dumps(
                    {"request": dict(arguments), "response": response_value},
                    indent=2,
                    sort_keys=True,
                )
            self._write_text(record_path.format_map(path_args), recorded + "\n")
        return ExecutionResult(text, "", 0 if 200 <= code < 300 else code)

    def _wait(self, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, Mapping):
            raise ValidationError("wait_for must be an object")
        path = value.get("path")
        field = value.get("json_field")
        expected = value.get("equals")
        if not isinstance(path, str) or not isinstance(field, str):
            raise ValidationError("wait_for requires path and json_field")
        deadline = time.monotonic() + float(value.get("timeout_seconds", self._timeout))
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(self._base_url + path, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, Mapping) and payload.get(field) == expected:
                    return
            except (OSError, json.JSONDecodeError):
                pass
            time.sleep(0.25)
        raise TimeoutError(f"environment state did not reach {field}={expected!r}")
