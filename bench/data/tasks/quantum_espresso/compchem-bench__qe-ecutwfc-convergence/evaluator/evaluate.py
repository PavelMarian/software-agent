#!/usr/bin/env python3
"""Bridge the upstream exit-code verifier to Software Benchmark JSON statuses."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


verifier = Path(__file__).with_name("verify.py")
try:
    completed = subprocess.run(
        [sys.executable, str(verifier)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=295,
    )
    passed = completed.returncode == 0
    diagnostics = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
except subprocess.TimeoutExpired as error:
    passed = False
    diagnostics = f"Verifier timed out after {error.timeout} seconds"
if diagnostics:
    print(diagnostics, file=sys.stderr)
print(json.dumps({"task_success": "PASSED" if passed else "FAILED"}))
