import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from software_bench.core.artifacts import decode_artifact, encode_artifact
from software_bench.core.models import Prediction, ValidationError
from software_bench.core.task_bundle import load_task_bundle
from software_bench.importers.manifest import import_executable_manifest
from software_bench.evaluation.backend import LocalCommandBackend
from software_bench.evaluation.scoring import score


def test_binary_artifact_round_trip_and_integrity() -> None:
    content = b"\x00\xffworkbook"
    encoded = encode_artifact(content)

    assert not isinstance(encoded, str)
    assert decode_artifact(encoded) == content

    damaged = dict(encoded)
    damaged["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="sha256"):
        decode_artifact(damaged)


def test_executable_manifest_materializes_binary_assets_and_evaluates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    workspace = source / "case"
    evaluator = source / "judge"
    workspace.mkdir(parents=True)
    evaluator.mkdir(parents=True)
    (workspace / "input.xlsx").write_bytes(b"PK\x03\x04input")
    (evaluator / "gold.xlsx").write_bytes(b"PK\x03\x04answer")
    (evaluator / "evaluate.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "ok = Path('answer.xlsx').read_bytes() == Path('.benchmark/gold.xlsx').read_bytes()\n"
        "print(json.dumps({'task_success': 'PASSED' if ok else 'FAILED'}))\n",
        encoding="utf-8",
    )
    manifest = {
        "source": "example/executable",
        "tasks": [
            {
                "instance_id": "binary-1",
                "problem_statement": "Create the requested workbook.",
                "target_software": "Spreadsheet tool",
                "workspace": "case",
                "evaluator": "judge",
                "submission_kind": "artifact_bundle",
                "submission_root": "output",
                "test_commands": [
                    {
                        "id": "evaluator",
                        "command": [sys.executable, ".benchmark/evaluate.py"],
                        "parser": "json-status",
                    }
                ],
                "workstreams": [
                    {"id": "edit", "title": "Edit", "description": "Create output."},
                    {"id": "verify", "title": "Verify", "description": "Check output."},
                ],
                "environment": {
                    "backend_hint": "local",
                    "image": None,
                    "workdir": ".",
                    "platform": "test",
                    "timeout_seconds": 30,
                    "network_enabled": False,
                },
            }
        ],
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    paths = import_executable_manifest(manifest_path, tmp_path / "bundles")
    bundle = load_task_bundle(paths[0], require_mas_ready=True)
    prediction = Prediction(
        "binary-1",
        "model",
        "",
        bundle.evaluation.submission_kind,
        {"answer.xlsx": encode_artifact(b"PK\x03\x04answer")},
    )

    result = score(bundle, LocalCommandBackend().evaluate(bundle, prediction))
    assert result["resolved"] is True
    assert (paths[0] / "workspace" / "input.xlsx").read_bytes() == b"PK\x03\x04input"
    assert (paths[0] / "evaluator" / "gold.xlsx").read_bytes() == b"PK\x03\x04answer"


def test_executable_manifest_can_install_application_validator(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = source / "case"
    workspace.mkdir(parents=True)
    manifest = {
        "source": "example/applications",
        "tasks": [
            {
                "instance_id": "spreadsheet-native-1",
                "problem_statement": "Create a valid workbook under output/.",
                "target_software": "Spreadsheet tool",
                "profiles": {
                    "environment": "spreadsheet",
                    "mcp": "spreadsheet",
                    "validator": "spreadsheet",
                },
                "workspace": "case",
                "workstreams": [
                    {"id": "edit", "title": "Edit", "description": "Create output."},
                    {"id": "verify", "title": "Verify", "description": "Validate output."},
                ],
                "environment": {
                    "backend_hint": "local",
                    "workdir": ".",
                    "platform": "test",
                    "timeout_seconds": 30,
                    "network_enabled": False,
                },
            }
        ],
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paths = import_executable_manifest(manifest_path, tmp_path / "bundles")
    bundle = load_task_bundle(paths[0], require_mas_ready=True)

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", "<workbook/>")
    prediction = Prediction(
        bundle.task.instance_id,
        "model",
        "",
        bundle.evaluation.submission_kind,
        {"answer.xlsx": encode_artifact(payload.getvalue())},
    )

    result = score(bundle, LocalCommandBackend().evaluate(bundle, prediction))

    assert result["resolved"] is True
    assert bundle.task.metadata["mcp_profile"] == "spreadsheet"
    assert bundle.environment.image == "software-bench/spreadsheet:v1"
    assert (paths[0] / "evaluator/application_evaluator.py").is_file()
