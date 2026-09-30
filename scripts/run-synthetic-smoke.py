"""Run a minimal real-LLM smoke test through the production multi-agent graph."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from software_multiagent.config import load_environment, target_model
from software_multiagent.application import SoftwareMultiAgent, SoftwareRunRequest
from software_multiagent.providers import create_chat_model
from software_multiagent.tools.workspace import WorkspaceFiles, WorkspaceToolset


EXPECTED = "Hello, World!"


def _react_complete(result: dict) -> bool:
    trace = result.get("reasoning_trace", [])
    for role in ("planner", "researcher", "executor", "evaluator"):
        steps = [item for item in trace if item.get("role") == role]
        if len(steps) < 2 or steps[-1].get("kind") != "finish":
            return False
        if not any(item.get("kind") in {"thought", "action"} for item in steps[:-1]):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    load_environment()
    provider, model_name = target_model()
    if provider not in {"openai", "openrouter"} or not model_name:
        raise SystemExit("configure SOFTWARE_MULTIAGENT_PROVIDER and SOFTWARE_MULTIAGENT_MODEL in .env")

    run_directory = arguments.output.resolve()
    if run_directory.exists():
        raise SystemExit(f"output already exists: {run_directory}")
    workspace = run_directory / "workspace"
    workspace.mkdir(parents=True)

    files = WorkspaceFiles(workspace)

    def unavailable_program(program: str, timeout: float, command_arguments: tuple[str, ...]):
        return {
            "exit_code": 127,
            "stderr": "program execution is intentionally unnecessary for this text-only smoke test",
        }

    def verify_workspace():
        target = workspace / "hello.txt"
        if not target.is_file():
            return {"passed": False, "summary": "hello.txt is missing"}
        actual = target.read_text(encoding="utf-8")
        return {
            "passed": actual == EXPECTED,
            "summary": "hello.txt has the exact expected content" if actual == EXPECTED else "hello.txt content differs",
            "actual": actual,
            "expected": EXPECTED,
        }

    tools = WorkspaceToolset(files, unavailable_program, verify_workspace)
    model = create_chat_model(provider=provider, model=model_name)
    result = SoftwareMultiAgent.run(
        model,
        SoftwareRunRequest(
            task_id="synthetic-hello-world",
            objective=(
            "Create a UTF-8 text file named hello.txt in the workspace. "
            "Its entire content must be exactly: Hello, World! "
            "Do not add a newline or any other characters. This task needs no program execution."
            ),
            workspace=workspace,
            target_software="generic text workspace",
        ),
        tools,
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model_name,
        "result": result,
        "independent_verification": verify_workspace(),
    }
    (run_directory / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if (
        result["status"] == "completed"
        and report["independent_verification"]["passed"]
        and _react_complete(result)
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
