from __future__ import annotations

import argparse
import sys
from pathlib import Path

from software_bench.harness.execution.operations import (
    EvaluationOptions,
    PreflightOptions,
    RunOptions,
    evaluate,
    import_tasks,
    preflight,
    run_benchmark,
    validate_contracts,
)
from software_bench.core.models import ValidationError
from software_bench.importers import (
    DatasetImportRequest,
    dataset_sources,
)
from software_bench.mcp import builtin_profiles


def build_parser() -> argparse.ArgumentParser:
    """Build the unified benchmark command-line parser.

    Returns:
        A parser containing benchmark operations and registered importers.
    """
    parser = argparse.ArgumentParser(prog="software-bench")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate TaskBundle and mode contracts")
    validate.add_argument("--bundle", "--task", dest="bundle", type=Path)
    validate.add_argument("--mode", type=Path)
    validate.add_argument("--roles-dir", type=Path, default=Path("configs/roles"))
    validate.add_argument("--mas-ready", action="store_true")

    run = commands.add_parser("run", help="run the benchmark-owned protocol")
    run.add_argument("--bundle", "--task", dest="bundle", type=Path, required=True)
    run.add_argument("--mode", type=Path, required=True)
    run.add_argument("--roles-dir", type=Path, default=Path("configs/roles"))
    connection = run.add_mutually_exclusive_group()
    connection.add_argument("--model-adapter", "--adapter", dest="model_adapter")
    connection.add_argument(
        "--framework-adapter",
        help="run a labeled black-box framework adapter instead of benchmark orchestration",
    )
    run.add_argument("--model", help="provider model ID")
    run.add_argument("--base-url", help="custom endpoint for the openai-compatible adapter")
    run.add_argument("--api-key-env", help="environment variable containing the provider API key")
    run.add_argument("--temperature", type=float)
    run.add_argument("--max-output-tokens", type=int, default=8192)
    run.add_argument("--adapter-max-retries", type=int, default=4)
    run.add_argument("--adapter-timeout-seconds", type=float, default=300.0)
    run_backend = run.add_mutually_exclusive_group()
    run_backend.add_argument(
        "--environment-backend",
        default="auto",
        help="local, docker, auto, or an installed environment backend",
    )
    run_backend.add_argument(
        "--no-docker",
        action="store_const",
        const="local",
        dest="environment_backend",
        help="run directly on the host, even when the TaskBundle prefers Docker",
    )
    run.add_argument("--workspace", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--seed", type=int, default=0)
    mcp = run.add_mutually_exclusive_group()
    mcp.add_argument(
        "--mcp-profile",
        choices=("auto", *builtin_profiles().keys()),
        default="auto",
        help="expose a built-in application MCP profile to roles with environment access",
    )
    mcp.add_argument(
        "--mcp-spec",
        type=Path,
        help="expose a hand-written application MCP spec to environment-enabled roles",
    )
    mcp.add_argument(
        "--no-mcp",
        action="store_const",
        const=None,
        dest="mcp_profile",
        help="disable automatic application MCP tools",
    )
    mutation_policy = run.add_mutually_exclusive_group()
    mutation_policy.add_argument(
        "--mcp-allow-mutations",
        dest="mcp_allow_mutations",
        action="store_true",
        help="include application tools that can change workspace or software state",
    )
    mutation_policy.add_argument(
        "--mcp-read-only",
        dest="mcp_allow_mutations",
        action="store_false",
        help="expose only read-only application MCP tools",
    )
    run.set_defaults(mcp_allow_mutations=True)

    preflight = commands.add_parser(
        "preflight", help="check that an application profile is installed in a task environment"
    )
    preflight.add_argument("--bundle", type=Path, required=True)
    preflight.add_argument(
        "--mcp-profile", choices=("auto", *builtin_profiles().keys()), default="auto"
    )
    preflight_backend = preflight.add_mutually_exclusive_group()
    preflight_backend.add_argument("--environment-backend", default="auto")
    preflight_backend.add_argument(
        "--no-docker", action="store_const", const="local", dest="environment_backend"
    )
    preflight.add_argument("--workspace", type=Path)
    preflight.add_argument("--run-id", default="preflight")
    preflight.add_argument("--output", type=Path)

    evaluate = commands.add_parser("evaluate", help="evaluate a normalized prediction")
    evaluate.add_argument("--bundle", "--task", dest="bundle", type=Path, required=True)
    evaluate.add_argument("--prediction", type=Path, required=True)
    evaluate.add_argument("--workspace", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluation_backend = evaluate.add_mutually_exclusive_group()
    evaluation_backend.add_argument(
        "--backend",
        default="auto",
        help="local, docker, auto, or an installed evaluation backend",
    )
    evaluation_backend.add_argument(
        "--no-docker",
        action="store_const",
        const="local",
        dest="backend",
        help="evaluate directly on the host, even when the TaskBundle prefers Docker",
    )

    dataset = commands.add_parser(
        "import-dataset", help="convert records from a registered dataset source"
    )
    dataset.add_argument("--source", choices=tuple(dataset_sources()), required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--upstream", type=Path)
    dataset.add_argument("--recipe", type=Path)
    dataset.add_argument("--dataset", type=Path)
    dataset.add_argument("--assets", type=Path)
    dataset.add_argument("--select", dest="selection", action="append", default=[])
    dataset.add_argument("--image")
    dataset.add_argument("--port", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse and execute one benchmark command.

    Args:
        argv: Command-line arguments without the program name. Uses
            ``sys.argv`` when omitted.

    Returns:
        A process-style status code.
    """
    args = build_parser().parse_args(argv)
    try:
        result = _execute(args)
        if result.output:
            print(result.output, end="" if result.output.endswith("\n") else "\n")
        return result.exit_code
    except (
        ValidationError,
        LookupError,
        TypeError,
        ValueError,
        OSError,
        PermissionError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


def _execute(args: argparse.Namespace):
    """Convert parsed arguments to one backend operation."""
    if args.command == "validate":
        return validate_contracts(args.bundle, args.mode, args.roles_dir, args.mas_ready)
    if args.command == "run":
        return run_benchmark(RunOptions(**_selected(args, RunOptions)))
    if args.command == "preflight":
        return preflight(PreflightOptions(**_selected(args, PreflightOptions)))
    if args.command == "evaluate":
        return evaluate(EvaluationOptions(**_selected(args, EvaluationOptions)))
    if args.command == "import-dataset":
        return import_tasks(
            DatasetImportRequest(
                source=args.source,
                output=args.output,
                upstream=args.upstream,
                recipe=args.recipe,
                dataset=args.dataset,
                assets=args.assets,
                selection=tuple(args.selection),
                image=args.image,
                port=args.port,
            )
        )
    raise AssertionError(f"unhandled command: {args.command}")


def _selected(args: argparse.Namespace, options_type: type) -> dict[str, object]:
    """Select dataclass fields from an argparse namespace."""
    return {
        name: getattr(args, name)
        for name in options_type.__dataclass_fields__
    }


if __name__ == "__main__":
    raise SystemExit(main())
