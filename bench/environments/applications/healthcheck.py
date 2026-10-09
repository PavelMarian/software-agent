from __future__ import annotations

import json
import os
import shutil
import sys


def main() -> int:
    executables = [
        item for item in os.environ.get("SOFTWARE_BENCH_EXECUTABLES", "").split(";") if item
    ]
    resolved = {name: shutil.which(name) for name in executables}
    print(json.dumps(resolved, sort_keys=True))
    return 0 if executables and all(resolved.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
