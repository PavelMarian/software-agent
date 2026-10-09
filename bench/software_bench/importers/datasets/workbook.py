from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from software_bench.application_profiles import APPLICATION_IMAGES
from software_bench.core.models import (
    CheckSpec,
    EnvironmentSpec,
    EvaluationSpec,
    EvaluationStrategy,
    ProvenanceSpec,
    SubmissionKind,
    TaskBundle,
    TaskKind,
    TaskSpec,
    TestCommand,
    ValidationError,
    Workstream,
)
from software_bench.core.task_bundle import write_task_bundle
from software_bench.importers.naming import safe_id


DEFAULT_WORKBOOK_IMAGE = APPLICATION_IMAGES["spreadsheet"]


def import_spreadsheetbench_dataset(
    dataset_file: str | Path,
    spreadsheets_root: str | Path,
    output_root: str | Path,
    *,
    image: str = DEFAULT_WORKBOOK_IMAGE,
    case_ids: Sequence[str] = (),
) -> tuple[Path, ...]:
    dataset = Path(dataset_file).resolve()
    records = _read_records(dataset)
    assets = Path(spreadsheets_root).resolve()
    selected = set(case_ids)
    available = {_record_id(item) for item in records}
    unknown = selected - available
    if unknown:
        raise ValidationError(f"unknown SpreadsheetBench ids: {sorted(unknown)}")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    written: list[Path] = []
    for record in records:
        case_id = _record_id(record)
        if selected and case_id not in selected:
            continue
        source_dir = _case_directory(record, assets, case_id)
        inputs = sorted(source_dir.glob("*_input.xls*"))
        answers = sorted(source_dir.glob("*_answer.xls*"))
        if not inputs or len(inputs) != len(answers):
            raise ValidationError(
                f"SpreadsheetBench {case_id} requires matching input and answer workbooks"
            )
        root = Path(output_root).resolve() / f"spreadsheetbench__{safe_id(case_id)}"
        if root.exists():
            raise ValidationError(f"output TaskBundle already exists: {root}")
        (root / "workspace" / "inputs").mkdir(parents=True)
        (root / "evaluator" / "gold").mkdir(parents=True)
        cases: list[dict[str, str]] = []
        answer_position = record.get("answer_position")
        if not isinstance(answer_position, str) or not answer_position:
            raise ValidationError(f"SpreadsheetBench {case_id} has no answer_position")
        for index, (input_path, answer_path) in enumerate(zip(inputs, answers), 1):
            public_name = input_path.name
            shutil.copy2(input_path, root / "workspace" / "inputs" / public_name)
            shutil.copy2(answer_path, root / "evaluator" / "gold" / answer_path.name)
            cases.append(
                {
                    "id": f"case_{index}",
                    "input": public_name,
                    "answer": answer_path.name,
                    "answer_position": answer_position,
                }
            )
        (root / "evaluator" / "metadata.json").write_text(
            json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copy2(
            Path(__file__).parent / "assets" / "workbook_evaluator.py",
            root / "evaluator" / "evaluate.py",
        )
        (root / "workspace" / "SOLUTION.md").write_text(
            "Create output/solution.py. It must accept INPUT_XLSX and OUTPUT_XLSX arguments, "
            "preserve the workbook, apply the requested transformation, and save OUTPUT_XLSX.\n",
            encoding="utf-8",
        )
        instruction = record.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValidationError(f"SpreadsheetBench {case_id} has no instruction")
        bundle = TaskBundle(
            root,
            TaskSpec(
                instance_id=f"spreadsheetbench__{case_id}",
                problem_statement=instruction,
                task_kind=TaskKind.SOFTWARE_USE,
                target_software="Python, openpyxl, and LibreOffice Calc",
                workstreams=(
                    Workstream(
                        "workbook_analysis",
                        "Workbook analysis",
                        "Inspect sheets, ranges, formulas, and formats.",
                    ),
                    Workstream(
                        "transformation",
                        "Transformation",
                        "Implement a reusable solution.py transformation.",
                    ),
                    Workstream(
                        "validation",
                        "Validation",
                        "Run the solution against all public input workbooks.",
                    ),
                ),
                metadata={
                    "source_id": case_id,
                    "instruction_type": record.get("instruction_type"),
                    "test_case_count": len(cases),
                    "mcp_profile": "spreadsheet",
                    "environment_profile": "spreadsheet",
                },
            ),
            EvaluationSpec(
                strategy=EvaluationStrategy.COMMAND_CHECKS,
                submission_kind=SubmissionKind.ARTIFACT_BUNDLE,
                checks=tuple(CheckSpec(item["id"], required=True) for item in cases),
                test_commands=(
                    TestCommand(
                        "spreadsheetbench-evaluator",
                        ("python3", ".benchmark/evaluate.py"),
                        "json-status",
                        timeout_seconds=1800,
                    ),
                ),
                submission_root="output",
                evaluator_assets="evaluator",
            ),
            EnvironmentSpec(
                backend_hint="docker",
                image=image,
                workdir="/workspace",
                platform="linux/x86_64",
                timeout_seconds=1800,
                network_enabled=False,
                workspace_source="bundle",
                seed_path="workspace",
            ),
            ProvenanceSpec(
                source="RUCKBReasoning/SpreadsheetBench",
                source_id=case_id,
                metadata={"dataset_sha256": digest},
            ),
        )
        write_task_bundle(bundle, root)
        written.append(root)
    return tuple(written)


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        if path.suffix.lower() == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read SpreadsheetBench dataset {path}: {error}") from error
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(item, Mapping) for item in values)
    ):
        raise ValidationError("SpreadsheetBench dataset must be a non-empty JSON array or JSONL")
    return values


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("id")
    if not isinstance(value, (str, int)) or not str(value):
        raise ValidationError("SpreadsheetBench record requires id")
    return str(value)


def _case_directory(record: Mapping[str, Any], root: Path, case_id: str) -> Path:
    relative = record.get("spreadsheet_path", case_id)
    if not isinstance(relative, str) or not relative:
        relative = case_id
    candidates = [(root / relative).resolve(), (root / case_id).resolve()]
    for candidate in candidates:
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValidationError(f"unsafe SpreadsheetBench path: {relative}") from error
        if candidate.is_dir():
            return candidate
    raise ValidationError(f"SpreadsheetBench workbook directory not found for {case_id}")
