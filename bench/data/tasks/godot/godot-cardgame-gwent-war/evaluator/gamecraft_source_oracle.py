"""Thin adapter around the unmodified GameCraft-Bench verifier.

The TaskBundle layout places the submitted Godot project at the evaluation
working directory rather than at /workspace/game.  The adapter changes only
that path in a temporary copy of the upstream rubric; replay, frame sampling,
rubric scoring, aggregation, and thresholding remain upstream behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--submission", type=Path, default=Path("."))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    judge = os.environ.get("GAMECRAFT_BENCH_JUDGE", "openai").strip().lower()
    if judge == "stub":
        raise SystemExit("The GameCraft stub judge is forbidden for benchmark scoring")

    source_root = args.source_package.resolve()
    sys.path.insert(0, str(source_root))
    from gamecraft_bench.verifier.score import score_project
    from gamecraft_bench.verifier.judges import get_judge

    submission = args.submission.resolve()
    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    rubric["build_check"] = dict(rubric["build_check"])
    rubric["build_check"]["cmd"] = (
        f'godot --headless --path "{submission}" --quit-after 5'
    )

    result_dir = Path(".benchmark/source-results/gamecraft")
    result_dir.mkdir(parents=True, exist_ok=True)
    runtime_rubric = result_dir / "rubric.runtime.json"
    runtime_rubric.write_text(json.dumps(rubric, indent=2), encoding="utf-8")

    result = score_project(
        project_dir=submission,
        rubric_path=runtime_rubric,
        output_dir=result_dir,
        judge=get_judge(),
    )
    summary = {
        "oracle": "GameCraft-Bench verifier",
        "reward": result.reward,
        "threshold": args.threshold,
        "build_ok": result.build_ok,
        "demos": len(result.demos),
        "errors": result.errors,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if result.build_ok and result.demos and result.reward >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
