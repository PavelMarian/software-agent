from __future__ import annotations

import ast
import hashlib
import shutil
from pathlib import Path
from typing import Sequence

from software_bench.core.models import ValidationError
from software_bench.core.task_bundle import write_task_bundle
from software_bench.importers.manifest import build_executable_task_bundle
from software_bench.importers.naming import safe_id


def import_sregym(
    upstream_root: str | Path,
    output_root: str | Path,
    *,
    problem_ids: Sequence[str] = (),
    port: int = 0,
) -> tuple[Path, ...]:
    """Create self-contained live tasks backed by the official SREGym conductor."""
    upstream = Path(upstream_root).resolve()
    available = _problem_ids(upstream / "sregym/conductor/problem_sets.py")
    selected = tuple(problem_ids) if problem_ids else available
    unknown = set(selected) - set(available)
    if unknown:
        raise ValidationError(f"unknown SREGym problem ids: {sorted(unknown)}")
    if port < 0 or port > 65535:
        raise ValidationError("SREGym port must be in [0, 65535]")
    source_hash = hashlib.sha256(
        (upstream / "sregym/conductor/problem_sets.py").read_bytes()
    ).hexdigest()
    written: list[Path] = []
    for problem_id in selected:
        root = Path(output_root).resolve() / f"sregym__{safe_id(problem_id)}"
        staging = root.parent / f".{root.name}.assets"
        if root.exists() or staging.exists():
            raise ValidationError(f"output TaskBundle already exists: {root}")
        try:
            workspace = staging / "workspace"
            evaluator = staging / "evaluator"
            workspace.mkdir(parents=True)
            evaluator.mkdir(parents=True)
            (workspace / "RUNBOOK.md").write_text(
                "Diagnose and mitigate the live Kubernetes incident. Use cluster commands "
                "to inspect "
                "the system. Call submit_sregym for diagnosis and then mitigation. Finally call "
                "collect_sregym_result; this writes results.json for the benchmark submission.\n",
                encoding="utf-8",
            )
            shutil.copy2(
                Path(__file__).parent / "assets" / "live_driver.py",
                evaluator / "live_driver.py",
            )
            shutil.copy2(
                Path(__file__).parent / "assets" / "live_evaluator.py",
                evaluator / "evaluate.py",
            )
            shutil.copytree(
                upstream,
                evaluator / "upstream",
                ignore=shutil.ignore_patterns(".git", "results", ".runtime", "__pycache__"),
            )
            port_value = str(port) if port else "{port}"
            base_url = f"http://127.0.0.1:{port_value}"
            value = {
                "instance_id": f"sregym__{problem_id}",
                "problem_statement": (
                    "Investigate the active Kubernetes reliability incident, submit a precise "
                    "root-cause "
                    "diagnosis, apply a safe mitigation, and submit evidence of the mitigation."
                ),
                "target_software": "Kubernetes, kubectl, Helm, and the live SREGym environment",
                "workspace": "workspace",
                "evaluator": "evaluator",
                "workstreams": [
                    {
                        "id": "diagnosis",
                        "title": "Diagnosis",
                        "description": "Inspect live signals and identify the root cause.",
                    },
                    {
                        "id": "mitigation",
                        "title": "Mitigation",
                        "description": "Apply and verify a safe recovery.",
                    },
                    {
                        "id": "evidence",
                        "title": "Evidence",
                        "description": "Submit both stages and collect official results.",
                    },
                ],
                "checks": [
                    {"id": "diagnosis", "required": True},
                    {"id": "mitigation", "required": True},
                ],
                "test_commands": [{
                    "id": "sregym-results",
                    "command": ["python3", ".benchmark/evaluate.py"],
                    "parser": "json-status",
                    "timeout_seconds": 60,
                }],
                "submission_kind": "environment_state",
                "state_paths": ["submissions", "results.json"],
                "evaluation_backend": "local",
                "environment": {
                    "backend_hint": "managed_process",
                    "timeout_seconds": 3600,
                    "backend_config": {
                        "managed_process": {
                            "command": [
                                "python3",
                                "{bundle_root}/evaluator/live_driver.py",
                                "--upstream",
                                "{bundle_root}/evaluator/upstream",
                                "--problem",
                                problem_id,
                                "--workspace",
                                "{workspace}",
                            ],
                            "cwd": "{bundle_root}/evaluator/upstream",
                            "readiness_url": f"{base_url}/status",
                            "shutdown_url": f"{base_url}/shutdown",
                            "startup_timeout_seconds": 900,
                            "allocate_port": port == 0,
                            "environment": {
                                "API_BIND_HOST": "127.0.0.1",
                                "API_PORT": port_value,
                            },
                        },
                        "run_environment": {
                            "KUBECONFIG": "{workspace}/.environment/kubeconfig"
                        },
                        "http_tools": {
                            "base_url": base_url,
                            "timeout_seconds": 600,
                            "tools": [
                                {
                                    "name": "get_sregym_context",
                                    "description": (
                                        "Get the live application name, namespaces, and public "
                                        "description."
                                    ),
                                    "parameters": {
                                        "type": "object",
                                        "properties": {},
                                        "additionalProperties": False,
                                    },
                                    "method": "GET",
                                    "path": "/get_app",
                                },
                                {
                                    "name": "submit_sregym",
                                    "description": (
                                        "Submit one official SREGym diagnosis or mitigation stage."
                                    ),
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "stage": {
                                                "type": "string",
                                                "enum": ["diagnosis", "mitigation"],
                                            },
                                            "solution": {"type": "string", "minLength": 1},
                                        },
                                        "required": ["stage", "solution"],
                                        "additionalProperties": False,
                                    },
                                    "method": "POST",
                                    "path": "/submit",
                                    "body_arguments": ["stage", "solution"],
                                    "record_path": "submissions/{stage}.json",
                                    "record_request": True,
                                    "mutates_workspace": True,
                                },
                                {
                                    "name": "collect_sregym_result",
                                    "description": (
                                        "Wait for teardown and collect the official "
                                        "live evaluation "
                                        "result."
                                    ),
                                    "parameters": {
                                        "type": "object",
                                        "properties": {},
                                        "additionalProperties": False,
                                    },
                                    "method": "GET",
                                    "path": "/results",
                                    "wait_for": {
                                        "path": "/status",
                                        "json_field": "stage",
                                        "equals": "done",
                                        "timeout_seconds": 600,
                                    },
                                    "record_path": "results.json",
                                    "mutates_workspace": True,
                                },
                            ],
                        },
                    },
                },
                "metadata": {
                    "source_id": problem_id,
                    "live": True,
                    "mcp_profile": "kubernetes",
                },
                "provenance": {"problem_sets_sha256": source_hash},
            }
            bundle = build_executable_task_bundle(
                value,
                bundle_root=root,
                source_root=staging,
                source_name="SREGym/SREGym",
                source_digest=source_hash,
            )
            write_task_bundle(bundle, root)
            written.append(root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return tuple(written)


def _problem_ids(path: Path) -> tuple[str, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as error:
        raise ValidationError(f"cannot read SREGym problem set: {error}") from error
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SREGYM_LITE_PROBLEMS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
                return value
    raise ValidationError("SREGYM_LITE_PROBLEMS not found")
