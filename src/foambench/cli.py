"""CLI for the standalone corpus boundary; agent orchestration is added in the next milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foambench.corpus import FoamBenchCorpus, import_dataset
from foambench.runner.execution import load_driver, run_split, run_task
from foambench.upstream.foambench import evaluate as evaluate_upstream, stage_run
from software_multiagent.config import load_environment, target_model


def main(argv: list[str] | None = None) -> int:
    load_environment()
    parser = argparse.ArgumentParser(prog="foambench")
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-foambench", help="Import an upstream FoamBench JSON file")
    importer.add_argument("--dataset", required=True, type=Path)
    importer.add_argument("--corpus", required=True, type=Path)
    importer.add_argument("--split", required=True, choices=("basic", "advanced"))
    importer.add_argument("--case", action="append", default=[])
    show = commands.add_parser("show-task", help="Show public task metadata")
    show.add_argument("--corpus", required=True, type=Path)
    show.add_argument("--task", required=True)
    materialize = commands.add_parser("materialize", help="Create a disposable public task workspace")
    materialize.add_argument("--corpus", required=True, type=Path)
    materialize.add_argument("--task", required=True)
    materialize.add_argument("--workspace", required=True, type=Path)
    task_runner = commands.add_parser("run-task", help="Run one task through a standalone agent driver")
    _run_arguments(task_runner, include_task=True)
    split_runner = commands.add_parser("run-split", help="Run a corpus split through a standalone agent driver")
    _run_arguments(split_runner, include_task=False)
    split_runner.add_argument("--split", default="all", choices=("basic", "advanced", "all"))
    evaluator = commands.add_parser("evaluate-upstream", help="Run the official FoamBench report scripts in order")
    evaluator.add_argument("--foambench-root", required=True, type=Path)
    evaluator.add_argument("--timeout-seconds", type=float, default=3600)
    staging = commands.add_parser("stage-upstream", help="Stage one standalone prediction in official FoamBench Dataset layout")
    staging.add_argument("--run", required=True, type=Path)
    staging.add_argument("--foambench-root", required=True, type=Path)
    staging.add_argument("--name", default="software-multiagent")
    args = parser.parse_args(argv)
    if args.command == "evaluate-upstream":
        reports = evaluate_upstream(args.foambench_root, timeout_seconds=args.timeout_seconds)
        print(json.dumps({"reports": reports}, indent=2))
        return 0 if reports and reports[-1]["exit_code"] == 0 and len(reports) == 4 else 1
    if args.command == "stage-upstream":
        print(json.dumps({"staged": str(stage_run(args.run, args.foambench_root, name=args.name))}, indent=2))
        return 0
    if args.command == "import-foambench":
        tasks = import_dataset(args.dataset, args.corpus, split=args.split, case_ids=args.case)
        print(json.dumps({"imported": [task.instance_id for task in tasks]}, indent=2))
    elif args.command == "show-task":
        print(json.dumps(FoamBenchCorpus(args.corpus).load(args.task).to_dict(), indent=2, sort_keys=True))
    elif args.command == "materialize":
        task = FoamBenchCorpus(args.corpus).materialize(args.task, args.workspace)
        print(json.dumps({"task": task.instance_id, "workspace": str(args.workspace.resolve())}, indent=2))
    elif args.command == "run-task":
        outcome = run_task(
            FoamBenchCorpus(args.corpus), args.task, args.output, driver=load_driver(args.driver),
            model=args.model, provider=args.provider, image=args.image, seed=args.seed, resume=args.resume,
        )
        print(json.dumps(outcome.to_dict(), indent=2))
        return 0 if outcome.status == "completed" else 1
    else:
        outcomes = run_split(
            FoamBenchCorpus(args.corpus), args.output, driver=load_driver(args.driver), split=args.split,
            model=args.model, provider=args.provider, image=args.image, seed=args.seed, resume=args.resume,
        )
        failed = sum(item.status == "failed" for item in outcomes)
        print(json.dumps({"total": len(outcomes), "failed": failed, "output": str(args.output.resolve())}, indent=2))
        return 1 if failed else 0
    return 0


def _run_arguments(parser: argparse.ArgumentParser, *, include_task: bool) -> None:
    default_provider, default_model = target_model()
    parser.add_argument("--corpus", required=True, type=Path)
    if include_task:
        parser.add_argument("--task", required=True)
    parser.add_argument("--driver", default="foambench.drivers:run", help="Agent driver as package.module:function")
    parser.add_argument(
        "--provider",
        choices=("openai", "openrouter"),
        default=default_provider,
        help="Provider for the built-in driver (default: SOFTWARE_MULTIAGENT_PROVIDER)",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Provider-specific model identifier (default: SOFTWARE_MULTIAGENT_MODEL)",
    )
    parser.add_argument("--image", help="Run OpenFOAM tools in this isolated Docker image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")


if __name__ == "__main__":
    raise SystemExit(main())
