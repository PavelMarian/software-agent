from __future__ import annotations

import json
from pathlib import Path

import pytest

from software_bench.application_profiles import APPLICATION_IMAGES
from software_bench.cli import main
from software_bench.core.models import ValidationError
from software_bench.core.task_bundle import load_task_bundle
from software_bench.importers.datasets.application import (
    APPLICATION_DATASET_SOURCES,
    import_application_dataset,
)
from software_bench.importers.datasets.scientific import SCIENTIFIC_SOURCES
from software_bench.mcp import automatic_spec


def test_source_aware_importers_cover_every_application_profile() -> None:
    covered = {"spreadsheet", "sqlite", "kubernetes"}
    covered.update(source.mcp_profile for source in SCIENTIFIC_SOURCES.values())
    covered.update(
        source.mcp_profile for source in APPLICATION_DATASET_SOURCES.values()
    )

    assert covered == set(APPLICATION_IMAGES)


@pytest.mark.parametrize("source_id", APPLICATION_DATASET_SOURCES)
def test_application_sources_import_into_their_mcp_profiles(
    tmp_path: Path, source_id: str
) -> None:
    source = APPLICATION_DATASET_SOURCES[source_id]
    upstream = tmp_path / source_id
    marker = upstream / source.marker
    if marker.suffix:
        marker.parent.mkdir(parents=True)
        marker.write_text("marker\n", encoding="utf-8")
    else:
        marker.mkdir(parents=True)
    case = upstream / "cases/example"
    case.mkdir(parents=True)
    (case / "input.txt").write_text("input\n", encoding="utf-8")
    recipe = tmp_path / f"{source_id}.json"
    recipe.write_text(
        json.dumps(
            {
                "source": source_id,
                "upstream_revision": "abc123",
                "tasks": [
                    {
                        "id": "example",
                        "workspace": "cases/example",
                        "problem_statement": "Complete the application task.",
                        "validation": {"required_paths": ["result.txt"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    paths = import_application_dataset(
        source_id, upstream, recipe, tmp_path / "tasks"
    )

    bundle = load_task_bundle(paths[0], require_mas_ready=True)
    assert bundle.task.target_software == source.target_software
    assert bundle.task.metadata["mcp_profile"] == source.mcp_profile
    assert bundle.task.metadata["dataset_source"] == source_id
    assert bundle.provenance.metadata["upstream"] == source.repository
    assert bundle.provenance.metadata["upstream_revision"] == "abc123"
    assert automatic_spec(bundle).name == f"software-bench-{source.mcp_profile.replace('_', '-')}"
    assert (paths[0] / "workspace/input.txt").is_file()
    assert (paths[0] / "evaluator/application_evaluator.py").is_file()


def test_application_dataset_import_is_available_through_cli(tmp_path: Path) -> None:
    source = APPLICATION_DATASET_SOURCES["freecad-examples"]
    upstream = tmp_path / "freecad"
    upstream.mkdir()
    (upstream / source.marker).write_text("examples\n", encoding="utf-8")
    (upstream / "PartDesign").mkdir()
    (upstream / "PartDesign/model.FCStd").write_text("model\n", encoding="utf-8")
    recipe = tmp_path / "freecad.json"
    recipe.write_text(
        json.dumps(
            {
                "source": source.id,
                "tasks": [
                    {
                        "id": "part-design",
                        "case_path": "PartDesign",
                        "problem_statement": "Modify the parametric model.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "tasks"

    result = main(
        [
            "import-dataset",
            "--source", source.id,
            "--upstream", str(upstream),
            "--recipe", str(recipe),
            "--select", "part-design",
            "--output", str(output),
        ]
    )

    assert result == 0
    assert (output / "freecad-examples__part-design/task.json").is_file()


def test_application_dataset_rejects_a_mismatched_recipe(tmp_path: Path) -> None:
    upstream = tmp_path / "lammps"
    (upstream / "examples").mkdir(parents=True)
    (upstream / "examples/README").write_text("examples\n", encoding="utf-8")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps({"source": "freecad-examples", "tasks": [{"id": "x"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="recipe source"):
        import_application_dataset(
            "lammps-examples", upstream, recipe, tmp_path / "tasks"
        )
