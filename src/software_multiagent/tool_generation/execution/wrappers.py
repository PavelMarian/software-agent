from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from software_multiagent.tool_generation.contracts.errors import ToolGenerationError
from software_multiagent.tool_generation.contracts.models import ApplicationRecipe, ToolRecipe
from software_multiagent.tool_generation.contracts.specs import ApplicationSpec, ToolSpec


_RUNTIME = r'''import argparse
import importlib
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request

CONFIG = __CONFIG__
SCHEMA = __SCHEMA__
sys.path.insert(0, os.getcwd())

def parse():
    parser = argparse.ArgumentParser()
    for name, item in SCHEMA.get("properties", {}).items():
        parser.add_argument("--" + name.replace("_", "-"), dest=name, required=True,
                            nargs="+" if item.get("type") == "array" else None)
    raw = vars(parser.parse_args())
    result = {}
    for name, value in raw.items():
        kind = SCHEMA["properties"][name].get("type", "string")
        if kind == "array": result[name] = value
        elif kind == "integer": result[name] = int(value)
        elif kind == "number": result[name] = float(value)
        elif kind == "boolean": result[name] = value.lower() in {"1", "true", "yes"}
        else: result[name] = value
    return result

def render(tokens, arguments):
    result = []
    for token in tokens:
        if token.startswith("{") and token.endswith("...}"):
            result.extend(str(item) for item in arguments[token[1:-4]])
        else:
            for name, value in arguments.items(): token = token.replace("{" + name + "}", str(value))
            result.append(token)
    return result

def main():
    arguments = parse()
    kind = CONFIG["kind"]
    if kind == "python":
        target = importlib.import_module(CONFIG["module"])
        for part in CONFIG["callable"].split("."): target = getattr(target, part)
        value = target(**arguments)
        if value is not None: print(json.dumps(value, default=str) if not isinstance(value, str) else value)
        return 0
    if kind == "http":
        method = CONFIG.get("method", "POST")
        url = CONFIG["url"]
        data = None
        if method in {"GET", "DELETE"}: url += ("&" if "?" in url else "?") + urllib.parse.urlencode(arguments, doseq=True)
        else: data = json.dumps(arguments).encode("utf-8")
        headers = {"Content-Type": "application/json", **CONFIG.get("headers", {})}
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method)) as response:
            sys.stdout.write(response.read().decode("utf-8", errors="replace"))
        return 0
    for step in CONFIG["steps"]:
        completed = subprocess.run(render(step, arguments), text=True, capture_output=True, check=False)
        sys.stdout.write(completed.stdout); sys.stderr.write(completed.stderr)
        if completed.returncode: return completed.returncode
    return 0

if __name__ == "__main__": raise SystemExit(main())
'''


def compile_wrappers(recipe: ApplicationRecipe, root: str | Path) -> ApplicationRecipe:
    """Render non-command adapters and return a command-only recipe.

    Args:
        recipe: Source recipe containing optional wrapper definitions.
        root: Directory in which generated wrapper programs are written.

    Returns:
        A semantically equivalent recipe accepted by the command MCP runtime.
    """

    wrapper_root = Path(root).resolve()
    wrapper_root.mkdir(parents=True, exist_ok=True)
    tools: list[ToolRecipe] = []
    for tool in recipe.tools:
        if tool.wrapper is None:
            tools.append(tool)
            continue
        properties = tool.spec.input_schema.get("properties", {})
        required = set(tool.spec.input_schema.get("required", []))
        if set(properties) != required:
            raise ToolGenerationError(
                f"generated wrapper {tool.spec.name!r} requires every input property to be required"
            )
        script = wrapper_root / f"{tool.spec.name}.py"
        config = {"kind": tool.wrapper.kind, **dict(tool.wrapper.config)}
        script.write_text(
            _RUNTIME.replace("__CONFIG__", repr(config)).replace(
                "__SCHEMA__", repr(dict(tool.spec.input_schema))
            ),
            encoding="utf-8",
        )
        command = ["python", str(script)]
        for name, item in properties.items():
            command.append("--" + name.replace("_", "-"))
            command.append("{" + name + ("...}" if item.get("type") == "array" else "}"))
        spec = ToolSpec(
            tool.spec.name,
            tool.spec.description,
            tool.spec.input_schema,
            tuple(command),
            tool.spec.timeout_seconds,
            tool.spec.mutating,
        )
        tools.append(
            ToolRecipe(spec, tool.source, tool.evidence, tool.sample_arguments, tool.expected, None)
        )
    application = ApplicationSpec(
        recipe.application.name,
        recipe.application.version,
        recipe.application.description,
        tuple(item.spec for item in tools),
    )
    return ApplicationRecipe(application, tuple(tools), recipe.required_executables)
