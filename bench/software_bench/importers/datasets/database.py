from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from software_bench.core.models import ValidationError
from software_bench.core.task_bundle import write_task_bundle
from software_bench.importers.naming import safe_id
from software_bench.importers.manifest import build_executable_task_bundle


DEFAULT_DATABASE_IMAGE = "software-bench/database-suite:v1"
_VARIANTS = {"lite": "spider2-lite", "snow": "spider2-snow", "dbt": "spider2-dbt"}


def import_spider2_dataset(
    dataset_file: str | Path,
    upstream_root: str | Path,
    output_root: str | Path,
    *,
    variant: str,
    image: str = DEFAULT_DATABASE_IMAGE,
    case_ids: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Import Spider2 Lite, Snow, or DBT without changing the harness protocol."""
    if variant not in _VARIANTS:
        raise ValidationError(f"unsupported Spider2 variant: {variant}")
    dataset = Path(dataset_file).resolve()
    upstream = Path(upstream_root).resolve()
    records = _read_jsonl(dataset)
    selected = set(case_ids)
    available = {_instance_id(item) for item in records}
    unknown = selected - available
    if unknown:
        raise ValidationError(f"unknown Spider2 ids: {sorted(unknown)}")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    written: list[Path] = []
    for record in records:
        source_id = _instance_id(record)
        if selected and source_id not in selected:
            continue
        root = Path(output_root).resolve() / f"spider2__{variant}__{safe_id(source_id)}"
        staging = root.parent / f".{root.name}.assets"
        if root.exists() or staging.exists():
            raise ValidationError(f"output TaskBundle already exists: {root}")
        try:
            _stage_assets(staging, upstream, dataset, record, source_id, variant)
            pass_env = (
                ["GOOGLE_APPLICATION_CREDENTIALS", "BIGQUERY_PROJECT"]
                if variant == "lite"
                else [
                    "SNOWFLAKE_ACCOUNT",
                    "SNOWFLAKE_USER",
                    "SNOWFLAKE_PASSWORD",
                    "SNOWFLAKE_WAREHOUSE",
                    "SNOWFLAKE_DATABASE",
                ]
                if variant == "snow"
                else ["GOOGLE_APPLICATION_CREDENTIALS", "BIGQUERY_PROJECT"]
            )
            value = {
                "instance_id": f"spider2__{variant}__{source_id}",
                "problem_statement": _instruction(record),
                "target_software": "SQL and the Spider 2.0 data environment",
                "profiles": {"mcp": "sqlite"},
                "workspace": "workspace",
                "evaluator": "evaluator",
                "workstreams": [
                    {
                        "id": "schema",
                        "title": "Schema and context",
                        "description": "Inspect the supplied database context and references.",
                    },
                    {
                        "id": "query",
                        "title": "Solution",
                        "description": "Create the required SQL or DBT result artifact.",
                    },
                    {
                        "id": "validation",
                        "title": "Validation",
                        "description": "Validate syntax and generated outputs before submission.",
                    },
                ],
                "checks": [{"id": "answer", "required": True}],
                "test_commands": [{
                    "id": "spider2-evaluator",
                    "command": ["python3", ".benchmark/evaluate.py"],
                    "parser": "json-status",
                    "timeout_seconds": 1800,
                }],
                "submission_kind": "artifact_bundle",
                "submission_root": "output",
                "environment": {
                    "backend_hint": "docker",
                    "image": image,
                    "timeout_seconds": 1800,
                    "pass_env": pass_env,
                    "network_enabled": variant in {"snow", "dbt"}
                    or not source_id.startswith("local"),
                },
                "metadata": {
                    "source_id": source_id,
                    "variant": variant,
                    "db": record.get("db") or record.get("db_id"),
                },
                "provenance": {"dataset_sha256": digest},
            }
            bundle = build_executable_task_bundle(
                value,
                bundle_root=root,
                source_root=staging,
                source_name="xlang-ai/Spider2",
                source_digest=digest,
            )
            write_task_bundle(bundle, root)
            written.append(root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return tuple(written)


def _stage_assets(
    staging: Path,
    upstream: Path,
    dataset: Path,
    record: Mapping[str, Any],
    source_id: str,
    variant: str,
) -> None:
    workspace = staging / "workspace"
    evaluator = staging / "evaluator"
    workspace.mkdir(parents=True)
    evaluator.mkdir(parents=True)
    variant_root = upstream / _VARIANTS[variant]
    suite = variant_root / "evaluation_suite"
    if not suite.is_dir():
        raise ValidationError(f"Spider2 evaluation suite not found: {suite}")
    shutil.copytree(suite, evaluator / "evaluation_suite")
    shutil.copy2(
        Path(__file__).parent / "assets" / "database_evaluator.py",
        evaluator / "evaluate.py",
    )

    if variant == "dbt":
        project = variant_root / "examples" / source_id
        if not project.is_dir():
            raise ValidationError(f"Spider2 DBT project not found: {project}")
        shutil.copytree(project, workspace, dirs_exist_ok=True)
        (workspace / "output" / "submission").mkdir(parents=True, exist_ok=True)
        instructions = (
            "Produce the requested DBT result. Put results_metadata.jsonl and every referenced "
            "answer file below submission/. Paths in results_metadata.jsonl must be relative "
            "to submission/.\n"
        )
    else:
        (workspace / "output").mkdir(exist_ok=True)
        (workspace / "output" / "answer.sql").write_text(
            "-- Write the final query here.\n", encoding="utf-8"
        )
        context = record.get("external_knowledge")
        (workspace / "CONTEXT.md").write_text(
            f"Database: {record.get('db') or record.get('db_id') or 'see task'}\n"
            f"External knowledge: {context or 'none'}\n",
            encoding="utf-8",
        )
        instructions = "Write the final executable SQL query to output/answer.sql.\n"
        shutil.copy2(dataset, evaluator / dataset.name)
        resource = variant_root / "resource"
        if resource.is_dir():
            shutil.copytree(resource, evaluator / "resource")
    (workspace / "SUBMISSION.md").write_text(instructions, encoding="utf-8")
    _filter_gold(evaluator, source_id, variant)
    (evaluator / "metadata.json").write_text(
        json.dumps({"variant": variant, "instance_id": source_id}, indent=2) + "\n",
        encoding="utf-8",
    )


def _filter_gold(evaluator: Path, source_id: str, variant: str) -> None:
    gold = evaluator / "evaluation_suite" / "gold"
    source_name = {
        "dbt": "spider2_eval.jsonl",
        "lite": "spider2lite_eval.jsonl",
        "snow": "spider2snow_eval.jsonl",
    }[variant]
    source = gold / source_name
    if not source.is_file():
        raise ValidationError(f"Spider2 gold metadata not found: {source}")
    rows = [row for row in _read_jsonl(source) if _instance_id(row) == source_id]
    if len(rows) != 1:
        raise ValidationError(f"Spider2 gold record not found for {source_id}")
    # The official suites resolve auxiliary gold files from shared directories.
    # Keep those directories intact, but restrict the metadata index to this task.
    source.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read Spider2 dataset {path}: {error}") from error
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValidationError("Spider2 dataset must be non-empty JSONL objects")
    return rows


def _instance_id(record: Mapping[str, Any]) -> str:
    value = record.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValidationError("Spider2 record requires instance_id")
    return value


def _instruction(record: Mapping[str, Any]) -> str:
    value = record.get("instruction") or record.get("question")
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Spider2 record requires instruction or question")
    return value
