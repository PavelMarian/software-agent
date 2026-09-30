"""Validate and invoke a locally checked-out official FoamBench evaluator."""
from __future__ import annotations

from pathlib import Path
import subprocess
import json


REQUIRED_SCRIPTS = ("execution_report.py", "similarity_report.py", "nmse_report.py", "score_calculation.py")


def validate_root(root: str | Path) -> Path:
    root = Path(root).resolve()
    missing = [name for name in REQUIRED_SCRIPTS if not (root / name).is_file()]
    if missing:
        raise ValueError(f"not an upstream FoamBench directory; missing: {', '.join(missing)}")
    if not (root / "Dataset").is_dir():
        raise ValueError("upstream FoamBench directory has no Dataset/")
    return root


def evaluate(root: str | Path, *, timeout_seconds: float = 3600) -> tuple[dict[str, int], ...]:
    """Run the official reports in their documented order in an isolated evaluator container/workdir."""
    root = validate_root(root)
    results = []
    for script in REQUIRED_SCRIPTS:
        completed = subprocess.run(["python3", script], cwd=root, text=True, capture_output=True, timeout=timeout_seconds, check=False)
        results.append({"script": script, "exit_code": completed.returncode, "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]})
        if completed.returncode:
            break
    return tuple(results)


def stage_run(run_dir: str | Path, root: str | Path, *, name: str = "software-multiagent") -> Path:
    """Materialize one standalone prediction in the folder shape expected upstream."""
    run_dir, root = Path(run_dir).resolve(), validate_root(root)
    task = json.loads((run_dir / "task.public.json").read_text(encoding="utf-8"))
    prediction = json.loads((run_dir / "prediction.json").read_text(encoding="utf-8"))
    parts = [part for part in task["source_case_id"].replace("\\", "/").split("/") if part]
    base = root / "Dataset" / ("Basic" if task["split"] == "basic" else "Advanced")
    target = base.joinpath(*parts, name) if task["split"] == "basic" else base.joinpath(*parts, name)
    if target.exists():
        raise ValueError(f"upstream submission exists: {target}")
    for relative, content in prediction["files"].items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return target
