from __future__ import annotations

import json
from pathlib import Path

import pytest

from software_bench.cli import main
from software_bench.core.models import ValidationError
from software_bench.core.task_bundle import load_task_bundle
from software_bench.importers.datasets.scientific import import_scientific_dataset
from software_bench.mcp import automatic_spec


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    upstream = tmp_path / "openmc"
    case = upstream / "tests/regression_tests/pincell"
    case.mkdir(parents=True)
    (case / "model.py").write_text("print('model')\n", encoding="utf-8")
    (case / "results_true.dat").write_text("keff 1.0\n", encoding="utf-8")
    harness = upstream / "tests/testing_harness.py"
    harness.write_text("# upstream harness\n", encoding="utf-8")
    recipe = tmp_path / "openmc-cases.json"
    evaluator = tmp_path / "benchmark_evaluate.py"
    evaluator.write_text("print('{}')\n", encoding="utf-8")
    recipe.write_text(
        json.dumps(
            {
                "source": "openmc-tests",
                "upstream_revision": "abc123",
                "cases": [
                    {
                        "id": "pincell",
                        "case_path": "tests/regression_tests/pincell",
                        "problem_statement": "Reproduce the pincell calculation.",
                        "workspace_assets": [
                            {
                                "source": "tests/regression_tests/pincell/model.py",
                                "target": "case/model.py",
                            }
                        ],
                        "evaluator_assets": [
                            {
                                "source": "tests/regression_tests/pincell/results_true.dat",
                                "target": "gold/results_true.dat",
                            },
                        ],
                        "recipe_evaluator_assets": [
                            {
                                "source": "benchmark_evaluate.py",
                                "target": "evaluate.py",
                            }
                        ],
                        "checks": [{"id": "numerical_regression"}],
                        "test_commands": [
                            {
                                "id": "official-regression",
                                "command": ["python3", ".benchmark/evaluate.py"],
                                "parser": "json-status",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return upstream, recipe


def test_scientific_import_separates_public_case_and_hidden_oracle(
    tmp_path: Path,
) -> None:
    upstream, recipe = _fixture(tmp_path)

    paths = import_scientific_dataset(
        "openmc-tests", upstream, recipe, tmp_path / "tasks"
    )

    bundle = load_task_bundle(paths[0], require_mas_ready=True)
    assert bundle.task.target_software == "OpenMC"
    assert bundle.task.metadata["mcp_profile"] == "openmc"
    assert bundle.provenance.metadata["upstream_revision"] == "abc123"
    assert (paths[0] / "workspace/case/model.py").is_file()
    assert not (paths[0] / "workspace/case/results_true.dat").exists()
    assert not (paths[0] / "workspace/gold").exists()
    assert (paths[0] / "evaluator/gold/results_true.dat").is_file()
    assert bundle.evaluation.submission_root == "output"
    assert automatic_spec(bundle).name == "software-bench-openmc"


def test_scientific_import_is_available_through_cli(tmp_path: Path) -> None:
    upstream, recipe = _fixture(tmp_path)
    output = tmp_path / "tasks"

    result = main(
        [
            "import-dataset",
            "--source", "openmc-tests",
            "--upstream", str(upstream),
            "--recipe", str(recipe),
            "--select", "pincell",
            "--output", str(output),
        ]
    )

    assert result == 0
    assert (output / "openmc-tests__pincell/task.json").is_file()


def test_scientific_import_rejects_assets_outside_upstream(tmp_path: Path) -> None:
    upstream, recipe = _fixture(tmp_path)
    value = json.loads(recipe.read_text(encoding="utf-8"))
    value["cases"][0]["workspace_assets"] = [
        {"source": "../secret", "target": "case"}
    ]
    recipe.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValidationError, match="escapes checkout"):
        import_scientific_dataset(
            "openmc-tests", upstream, recipe, tmp_path / "tasks"
        )


def test_scientific_import_rejects_hidden_oracle_inside_public_directory(
    tmp_path: Path,
) -> None:
    upstream, recipe = _fixture(tmp_path)
    value = json.loads(recipe.read_text(encoding="utf-8"))
    value["cases"][0]["workspace_assets"] = [
        {"source": "tests/regression_tests/pincell", "target": "case"}
    ]
    recipe.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "tasks"

    with pytest.raises(ValidationError, match="overlap public"):
        import_scientific_dataset("openmc-tests", upstream, recipe, output)

    assert not output.exists()


def test_scientific_import_installs_default_application_validator(
    tmp_path: Path,
) -> None:
    upstream, recipe = _fixture(tmp_path)
    value = json.loads(recipe.read_text(encoding="utf-8"))
    case = value["cases"][0]
    case.pop("checks")
    case.pop("test_commands")
    case["validation"] = {
        "required_paths": ["statepoint.*.h5"],
        "metrics": [
            {
                "id": "keff",
                "path": "summary.txt",
                "regex": "keff=([0-9.]+)",
                "expected": 1.0,
                "abs_tol": 0.001,
            }
        ],
    }
    recipe.write_text(json.dumps(value), encoding="utf-8")

    paths = import_scientific_dataset(
        "openmc-tests", upstream, recipe, tmp_path / "tasks"
    )
    bundle = load_task_bundle(paths[0], require_mas_ready=True)

    assert {item.id for item in bundle.evaluation.checks} == {
        "artifact_structure",
        "native_validation",
        "numerical_accuracy",
    }
    assert bundle.evaluation.test_commands[0].id == "application-validator"
    assert bundle.environment.image == "software-bench/openmc:v1"
    assert (paths[0] / "evaluator/application_evaluator.py").is_file()
    assert json.loads(
        (paths[0] / "evaluator/validation.json").read_text(encoding="utf-8")
    )["metrics"][0]["id"] == "keff"
