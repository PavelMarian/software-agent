from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    required = (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DATABASE",
        "POSTGRES_USERNAME",
        "POSTGRES_PASSWORD",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print("Missing disposable PostgreSQL settings: " + ", ".join(missing), file=sys.stderr)
        print(json.dumps({"database_semantics": "FAILED"}))
        return 0
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(".benchmark/upstream/verify.py"))],
            capture_output=True,
            text=True,
            timeout=1700,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout[-8000:], file=sys.stderr)
        if completed.stderr:
            print(completed.stderr[-4000:], file=sys.stderr)
        passed = completed.returncode == 0
    except Exception as error:
        print(f"PostgreSQL verifier error: {type(error).__name__}: {error}", file=sys.stderr)
        passed = False
    print(json.dumps({"database_semantics": "PASSED" if passed else "FAILED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
