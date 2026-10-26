from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _status(value: bool) -> str:
    return "PASSED" if value else "FAILED"


def main() -> int:
    benchmark = Path(".benchmark")
    spec = json.loads((benchmark / "spec/adapter.json").read_text(encoding="utf-8"))
    required = [Path(name) for name in spec["required"]]
    required_ok = all(path.is_file() and path.stat().st_size > 0 for path in required)

    app = Path("/app")
    outputs = app / "outputs"
    tests = Path("/tests/correctness")
    logs = Path("/logs/verifier")
    try:
        outputs.mkdir(parents=True, exist_ok=True)
        tests.parent.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        local_inputs = Path("inputs")
        if local_inputs.is_dir():
            shutil.copytree(local_inputs, app / "inputs", dirs_exist_ok=True)
        source_tests = benchmark / "upstream_tests" / "correctness"
        shutil.copytree(source_tests, tests, dirs_exist_ok=True)
        for path in required:
            target = app / path.name if path.name == "final_answer.txt" else outputs / path.name
            if path.is_file():
                shutil.copy2(path, target)
        executable = shutil.which("rewardkit")
        command = [executable, str(tests)] if executable else [sys.executable, "-m", "rewardkit", str(tests)]
        completed = subprocess.run(
            command, cwd=app, capture_output=True, text=True, timeout=1200, check=False
        )
        if completed.stdout or completed.stderr:
            print((completed.stdout + completed.stderr)[-12000:], file=sys.stderr)
        reward_path = logs / "reward.json"
        values = json.loads(reward_path.read_text(encoding="utf-8")) if reward_path.is_file() else {}
        correctness_ok = (
            completed.returncode == 0
            and bool(values)
            and all(float(value) == 1.0 for value in values.values())
        )
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"MDArena correctness evaluator error: {type(error).__name__}: {error}", file=sys.stderr)
        correctness_ok = False

    print(json.dumps({
        "required_outputs": _status(required_ok),
        "upstream_correctness": _status(correctness_ok),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
