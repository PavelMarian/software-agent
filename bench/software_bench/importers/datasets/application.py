from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from software_bench.application_profiles import APPLICATION_IMAGES
from software_bench.core.models import ValidationError
from software_bench.core.task_bundle import write_task_bundle
from software_bench.importers.manifest import build_executable_task_bundle


@dataclass(frozen=True)
class ApplicationDatasetSource:
    id: str
    target_software: str
    mcp_profile: str
    repository: str
    marker: str


APPLICATION_DATASET_SOURCES: Mapping[str, ApplicationDatasetSource] = {
    "foambench": ApplicationDatasetSource(
        "foambench",
        "OpenFOAM",
        "openfoam",
        "https://github.com/NLR-Theseus/cfdllmbench",
        "FoamBench",
    ),
    "qgis-processing-tests": ApplicationDatasetSource(
        "qgis-processing-tests",
        "QGIS",
        "qgis",
        "https://github.com/qgis/QGIS",
        "python/plugins/processing/tests/testdata",
    ),
    "paraview-regression-tests": ApplicationDatasetSource(
        "paraview-regression-tests",
        "ParaView",
        "paraview",
        "https://gitlab.kitware.com/paraview/paraview",
        "Clients/ParaView/Testing/Python",
    ),
    "freecad-examples": ApplicationDatasetSource(
        "freecad-examples",
        "FreeCAD",
        "freecad",
        "https://github.com/FreeCAD/Examples",
        "README.md",
    ),
    "gromacs-regressiontests": ApplicationDatasetSource(
        "gromacs-regressiontests",
        "GROMACS",
        "gromacs",
        "https://gitlab.com/gromacs/gromacs-regressiontests",
        "gmxtest.pl",
    ),
    "lammps-examples": ApplicationDatasetSource(
        "lammps-examples",
        "LAMMPS",
        "lammps",
        "https://github.com/lammps/lammps",
        "examples/README",
    ),
    "blender-benchmark": ApplicationDatasetSource(
        "blender-benchmark",
        "Blender",
        "blender",
        "https://projects.blender.org/archive/blender-benchmark-bundle",
        "README.md",
    ),
}


def import_application_dataset(
    source_id: str,
    upstream_root: str | Path,
    recipe_file: str | Path,
    output_root: str | Path,
    *,
    image: str | None = None,
    task_ids: Sequence[str] = (),
) -> tuple[Path, ...]:
    source = APPLICATION_DATASET_SOURCES.get(source_id)
    if source is None:
        raise ValidationError(f"unknown application dataset source: {source_id}")
    upstream = Path(upstream_root).resolve()
    marker = _source_path(upstream, source.marker)
    if not marker.exists():
        raise ValidationError(f"{source_id} checkout marker is missing: {source.marker}")
    recipe_path = Path(recipe_file).resolve()
    recipe = _read_object(recipe_path)
    if recipe.get("source") != source_id:
        raise ValidationError(f"application recipe source must be {source_id!r}")
    raw_tasks = recipe.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValidationError("application recipe tasks must be a non-empty array")

    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise ValidationError("each application task must be an object")
        task_id = _required_string(raw, "id")
        if task_id in by_id:
            raise ValidationError(f"duplicate application task id: {task_id}")
        by_id[task_id] = raw
    selected = set(task_ids)
    unknown = selected - set(by_id)
    if unknown:
        raise ValidationError(f"unknown application task ids: {sorted(unknown)}")

    recipe_digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    revision = recipe.get("upstream_revision", "")
    if not isinstance(revision, str):
        raise ValidationError("upstream_revision must be a string")
    output = Path(output_root).resolve()
    written: list[Path] = []
    for task_id, raw in by_id.items():
        if selected and task_id not in selected:
            continue
        instance_id = f"{source.id}__{_safe_id(task_id)}"
        normalized = _normalized_task(
            source,
            raw,
            instance_id=instance_id,
            image=image,
            revision=revision,
        )
        root = output / instance_id
        bundle = build_executable_task_bundle(
            normalized,
            bundle_root=root,
            source_root=upstream,
            source_name=source.id,
            source_digest=recipe_digest,
        )
        write_task_bundle(bundle, root)
        written.append(root)
    return tuple(written)


def _normalized_task(
    source: ApplicationDatasetSource,
    raw: Mapping[str, Any],
    *,
    instance_id: str,
    image: str | None,
    revision: str,
) -> dict[str, Any]:
    workspace = raw.get("workspace", raw.get("case_path"))
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValidationError("application task workspace must be a non-empty string")
    value = dict(raw)
    value.pop("id", None)
    value.pop("case_path", None)
    value.update(
        {
            "instance_id": instance_id,
            "workspace": workspace,
            "target_software": source.target_software,
            "profiles": {
                "environment": source.mcp_profile,
                "mcp": source.mcp_profile,
                "validator": source.mcp_profile,
            },
            "provenance": {
                **_mapping(raw.get("provenance", {}), "provenance"),
                "upstream": source.repository,
                "upstream_revision": revision,
                "source_task_id": raw["id"],
            },
            "metadata": {
                **_mapping(raw.get("metadata", {}), "metadata"),
                "dataset_source": source.id,
                "source_task_id": raw["id"],
            },
        }
    )
    value.setdefault(
        "workstreams",
        [
            {
                "id": "task_analysis",
                "title": "Task analysis",
                "description": (
                    "Inspect the upstream case and determine the required "
                    "application operations."
                ),
            },
            {
                "id": "application_execution",
                "title": "Application execution",
                "description": "Modify and run the case with the target application's MCP tools.",
            },
            {
                "id": "result_validation",
                "title": "Result validation",
                "description": (
                    "Validate native outputs and prepare the requested submission "
                    "artifacts."
                ),
            },
        ],
    )
    environment = _mapping(raw.get("environment", {}), "environment")
    value["environment"] = {
        **environment,
        "image": image or environment.get("image") or APPLICATION_IMAGES[source.mcp_profile],
    }
    return value


def _make_importer(source_id: str) -> Callable[..., tuple[Path, ...]]:
    def importer(
        upstream_root: str | Path,
        recipe_file: str | Path,
        output_root: str | Path,
        *,
        image: str | None = None,
        task_ids: Sequence[str] = (),
    ) -> tuple[Path, ...]:
        return import_application_dataset(
            source_id,
            upstream_root,
            recipe_file,
            output_root,
            image=image,
            task_ids=task_ids,
        )

    return importer


import_foambench = _make_importer("foambench")
import_qgis_processing_tests = _make_importer("qgis-processing-tests")
import_paraview_regression_tests = _make_importer("paraview-regression-tests")
import_freecad_examples = _make_importer("freecad-examples")
import_gromacs_regressiontests = _make_importer("gromacs-regressiontests")
import_lammps_examples = _make_importer("lammps-examples")
import_blender_benchmark = _make_importer("blender-benchmark")


def _source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValidationError(f"application source path escapes checkout: {relative}") from error
    return path


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"application task {field} must be an object")
    return dict(value)


def _safe_id(value: str) -> str:
    result = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    if not result.strip(".-"):
        raise ValidationError(f"cannot derive application instance id from {value!r}")
    return result


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValidationError(f"{key} must be a non-empty string")
    return item


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read application recipe {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValidationError("application recipe must contain an object")
    return value
