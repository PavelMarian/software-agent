from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    try:
        results = json.loads(Path("results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        results = {}
    statuses = {}
    for stage in ("Diagnosis", "Mitigation"):
        value = results.get(stage, {})
        statuses[stage.lower()] = (
            "PASSED" if isinstance(value, dict) and value.get("success") is True else "FAILED"
        )
    print(json.dumps(statuses, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
