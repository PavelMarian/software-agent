from __future__ import annotations

import ast
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _status(value: bool) -> str:
    return "PASSED" if value else "FAILED"


def _compatible(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_compatible(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_compatible(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
    return left == right


def main() -> int:
    model = Path("model.py")
    submitted = Path("results.json")
    model_ok = model.is_file()
    chain_ok = False
    if model_ok:
        try:
            tree = ast.parse(model.read_text(encoding="utf-8"))
            calls = {getattr(node.func, "attr", "") for node in ast.walk(tree) if isinstance(node, ast.Call)}
            chain_ok = {"write_simulation", "run_simulation"} <= calls
        except (OSError, SyntaxError):
            pass
    submitted_value = None
    try:
        submitted_value = json.loads(submitted.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    shape_ok = (
        isinstance(submitted_value, dict)
        and submitted_value.get("normal_termination") is True
        and isinstance(submitted_value.get("observations"), dict)
        and bool(submitted_value["observations"])
        and isinstance(submitted_value.get("budget_summary"), dict)
        and bool(submitted_value["budget_summary"])
    )
    replay_ok = False
    if model_ok and shape_ok:
        with tempfile.TemporaryDirectory(prefix="modflow-replay-") as temp:
            work = Path(temp)
            shutil.copy2(model, work / "model.py")
            try:
                completed = subprocess.run(
                    [sys.executable, "model.py"], cwd=work, capture_output=True,
                    text=True, timeout=3300, check=False,
                )
                candidate = work / "results.json"
                if completed.stdout or completed.stderr:
                    print((completed.stdout + completed.stderr)[-6000:], file=sys.stderr)
                if completed.returncode == 0 and candidate.is_file():
                    replayed = json.loads(candidate.read_text(encoding="utf-8"))
                    replay_ok = _compatible(submitted_value, replayed)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                pass
    print(json.dumps({
        "model_script": _status(model_ok),
        "build_run_chain": _status(chain_ok),
        "result_contract": _status(shape_ok),
        "deterministic_replay": _status(replay_ok),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
