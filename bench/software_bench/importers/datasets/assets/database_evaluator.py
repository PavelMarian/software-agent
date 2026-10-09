from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    metadata = json.loads(Path(".benchmark/metadata.json").read_text(encoding="utf-8"))
    variant = metadata["variant"]
    instance_id = metadata["instance_id"]
    with tempfile.TemporaryDirectory(prefix="spider2-") as temp:
        result_dir = Path(temp) / "results"
        result_dir.mkdir()
        if variant == "dbt":
            source = Path("submission")
            if source.is_dir():
                shutil.copytree(source, result_dir, dirs_exist_ok=True)
        elif Path("answer.sql").is_file():
            shutil.copy2("answer.sql", result_dir / f"{instance_id}.sql")
        suite = Path(".benchmark/evaluation_suite")
        command = [
            sys.executable,
            str(suite / "evaluate.py"),
            "--result_dir",
            str(result_dir),
            "--gold_dir",
            str(suite / "gold"),
        ]
        if variant != "dbt":
            command.extend(["--mode", "sql", "--max_workers", "1"])
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=1700, check=False
            )
            passed = completed.returncode == 0 and _passed(completed.stdout, instance_id, variant)
        except (OSError, subprocess.SubprocessError):
            passed = False
    print(json.dumps({"answer": "PASSED" if passed else "FAILED"}))
    return 0


def _passed(output: str, instance_id: str, variant: str) -> bool:
    if variant != "dbt":
        for line in reversed(output.splitlines()):
            try:
                value = ast.literal_eval(line.strip())
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, dict) and value.get(instance_id) in (1, 1.0, True):
                return True
        return False
    return any(
        float(match.group(1)) == 1.0
        for match in re.finditer(r"(?m)^\s*(\d+(?:\.\d+)?)\s+\d+\s+\d+\s*$", output)
    )


if __name__ == "__main__":
    raise SystemExit(main())
