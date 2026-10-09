from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from software_bench.core.models import ValidationError
from software_bench.core.task_bundle import load_task_bundle
from software_bench.harness.environments import LocalEnvironmentSession, create_environment_session
from software_bench.mcp.registry import automatic_spec, builtin_profiles, load_spec, write_spec
from software_bench.mcp.server import ApplicationServer, JsonRpcServer, serve_stdio
from software_bench.mcp.preflight import preflight_profile


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for application MCP operations.

    Returns:
        A configured argument parser with profile, validation, and serving
        subcommands.
    """
    parser = argparse.ArgumentParser(prog="python -m software_bench.mcp")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-profiles", help="list built-in application profiles")

    generate = commands.add_parser("generate", help="write an MCP spec")
    source = generate.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="select a profile from a TaskBundle")
    source.add_argument("--profile", choices=tuple(builtin_profiles()))
    generate.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser(
        "validate-spec", help="validate a hand-written MCP application spec"
    )
    validate.add_argument("--spec", type=Path, required=True)
    validate.add_argument(
        "--workspace",
        type=Path,
        help="also check declared executables in this application workspace",
    )
    validate.add_argument("--output", type=Path)

    serve = commands.add_parser("serve", help="serve an application MCP over stdio")
    source = serve.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bundle", type=Path, help="select profile and environment from TaskBundle"
    )
    source.add_argument("--spec", type=Path, help="serve a hand-written MCP application spec")
    serve.add_argument(
        "--profile", choices=tuple(builtin_profiles()), help="override bundle detection"
    )
    serve.add_argument("--workspace", type=Path)
    serve.add_argument("--environment-backend", default="auto")
    serve.add_argument("--run-id", default="mcp-server")
    serve.add_argument("--allow-mutations", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run an application MCP command.

    Args:
        argv: Command-line arguments without the program name. Uses
            ``sys.argv`` when omitted.

    Returns:
        Zero on success, one when executable preflight fails, or two when the
        command is invalid or cannot be completed.

    Raises:
        AssertionError: If argument parsing produces an unknown command.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-profiles":
            descriptions = {
                name: spec.description for name, spec in builtin_profiles().items()
            }
            print(json.dumps(descriptions, indent=2))
            return 0
        if args.command == "generate":
            spec = (
                automatic_spec(load_task_bundle(args.bundle))
                if args.bundle is not None
                else builtin_profiles()[args.profile]
            )
            write_spec(spec, args.output)
            print(f"wrote {spec.name} -> {args.output}")
            return 0
        if args.command == "validate-spec":
            spec = load_spec(args.spec)
            report = {
                "profile": spec.name,
                "valid": True,
                "validation_level": "static",
                "tool_count": len(spec.tools),
            }
            if args.workspace is not None:
                environment = LocalEnvironmentSession(args.workspace.resolve())
                try:
                    report.update(preflight_profile(spec, environment))
                finally:
                    environment.close()
                report["validation_level"] = "preflight"
                report["valid"] = bool(report["ready"])
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.output is None:
                print(rendered, end="")
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            return 0 if report["valid"] else 1
        if args.command == "serve":
            if args.bundle is not None:
                bundle = load_task_bundle(args.bundle)
                spec = builtin_profiles()[args.profile] if args.profile else automatic_spec(bundle)
                environment = create_environment_session(
                    args.environment_backend,
                    bundle,
                    workspace=args.workspace.resolve() if args.workspace else None,
                    run_id=args.run_id,
                )
            else:
                spec = load_spec(args.spec)
                if args.workspace is None:
                    raise ValidationError("--workspace is required when serving a manual spec")
                environment = LocalEnvironmentSession(args.workspace.resolve())
            serve_stdio(
                JsonRpcServer(
                    ApplicationServer(spec, environment, allow_mutations=args.allow_mutations)
                )
            )
            return 0
    except (ValidationError, LookupError, OSError, PermissionError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
