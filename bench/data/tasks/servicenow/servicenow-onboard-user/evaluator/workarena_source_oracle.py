"""Gate a ServiceNow result on an out-of-workspace WorkArena validate() receipt.

The managed runner must preserve the generated task object and call its
upstream validate(page, chat_messages) method after the agent episode.  It
writes the returned values and task identity to a runner-owned JSON path and
passes that path as WORKARENA_VALIDATION_RECEIPT.  Candidate-authored receipts
inside the submission directory are deliberately rejected.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-class", required=True)
    parser.add_argument("--submission", type=Path, default=Path("."))
    args = parser.parse_args()
    raw = os.environ.get("WORKARENA_VALIDATION_RECEIPT", "")
    if not raw:
        print(json.dumps({"status": "failed", "error": "managed runner did not provide WORKARENA_VALIDATION_RECEIPT"}))
        return 1
    receipt_path = Path(raw).resolve()
    submission = args.submission.resolve()
    if receipt_path == submission or submission in receipt_path.parents:
        print(json.dumps({"status": "failed", "error": "candidate-controlled WorkArena receipt rejected"}))
        return 1
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": f"invalid WorkArena receipt: {error}"}))
        return 1
    ok = (
        receipt.get("validator") == "browsergym.workarena.validate"
        and receipt.get("source_class") == args.source_class
        and receipt.get("done") is True
        and float(receipt.get("reward", 0.0)) >= 1.0
        and bool(receipt.get("episode_id"))
        and bool(receipt.get("seed") is not None)
    )
    print(json.dumps({"oracle": "WorkArena task.validate", "status": "passed" if ok else "failed", "receipt": {k: receipt.get(k) for k in ("source_class", "episode_id", "seed", "reward", "done")}}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
