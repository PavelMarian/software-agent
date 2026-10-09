from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

from software_bench.validation.evaluators.application import (
    APPLICATION_RULES,
    evaluate_application,
)
from software_bench.application_profiles import APPLICATION_IMAGES
from software_bench.mcp import builtin_profiles


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ROOT = ROOT / "environments/applications"


def test_environment_catalog_covers_every_application_profile() -> None:
    catalog = json.loads(
        (ENVIRONMENT_ROOT / "catalog.json").read_text(encoding="utf-8")
    )["profiles"]
    profiles = builtin_profiles()

    assert set(catalog) == set(profiles) == set(APPLICATION_RULES)
    assert set(catalog) == set(APPLICATION_IMAGES)
    assert len({entry["image"] for entry in catalog.values()}) == len(catalog)
    for name, spec in profiles.items():
        declared = set(catalog[name]["executables"])
        required = {
            tool.command[0]
            for tool in spec.tools
            if "{" not in tool.command[0]
        }
        assert required <= declared
        assert catalog[name]["validator"] == name
        assert catalog[name]["image"] == APPLICATION_IMAGES[name]
        assert all("=" in package for package in catalog[name]["conda_packages"])
        dockerfile = catalog[name].get("dockerfile", "Dockerfile")
        assert (ENVIRONMENT_ROOT / dockerfile).is_file()


def test_application_image_build_matrix_is_present() -> None:
    assert (ENVIRONMENT_ROOT / "Dockerfile").is_file()
    assert (ENVIRONMENT_ROOT / "healthcheck.py").is_file()
    assert (ENVIRONMENT_ROOT / "build_images.py").is_file()


def test_environment_variants_use_the_same_catalog_and_build_directory() -> None:
    value = json.loads(
        (ENVIRONMENT_ROOT / "catalog.json").read_text(encoding="utf-8")
    )
    variants = value["variants"]

    assert set(variants) == {"database_suite"}
    for variant in variants.values():
        assert variant["profile"] in value["profiles"]
        assert (ENVIRONMENT_ROOT / variant["dockerfile"]).is_file()
        assert variant["pip_packages"]
        assert all("==" in package for package in variant["pip_packages"])


def test_environment_layout_has_no_dataset_named_directories() -> None:
    directories = {
        path.name for path in ENVIRONMENT_ROOT.parent.iterdir() if path.is_dir()
    }

    assert directories == {"applications"}


def test_native_validators_accept_minimal_real_format_artifacts(tmp_path: Path) -> None:
    roots = {name: tmp_path / name for name in APPLICATION_RULES}
    for root in roots.values():
        root.mkdir()

    (roots["blender"] / "scene.blend").write_bytes(b"BLENDER-v300")
    (roots["energyplus"] / "eplusout.err").write_text(
        "EnergyPlus Completed Successfully-- 0 Severe Errors\n", encoding="utf-8"
    )
    with zipfile.ZipFile(roots["freecad"] / "part.FCStd", "w") as archive:
        archive.writestr("Document.xml", "<Document/>")
    (roots["gromacs"] / "md.log").write_text("Finished mdrun\n", encoding="utf-8")
    (roots["kubernetes"] / "results.json").write_text("{}\n", encoding="utf-8")
    (roots["lammps"] / "log.lammps").write_text(
        "Loop time of 1.0 on 1 procs\n", encoding="utf-8"
    )
    (roots["modflow"] / "model.lst").write_text(
        "Normal termination of simulation\n", encoding="utf-8"
    )
    (roots["openmc"] / "statepoint.10.h5").write_bytes(b"\x89HDF\r\n\x1a\n")
    foam = roots["openfoam"] / "case"
    (foam / "system").mkdir(parents=True)
    (foam / "system/controlDict").write_text("application simpleFoam;\n", encoding="utf-8")
    (foam / "log.simpleFoam").write_text("End\n", encoding="utf-8")
    (roots["paraview"] / "mesh.vtk").write_bytes(b"# vtk DataFile Version 3.0\n")
    (roots["qgis"] / "layer.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
    )
    (roots["quantum_espresso"] / "scf.out").write_text(
        "! total energy = -10.0 Ry\nJOB DONE.\n", encoding="utf-8"
    )
    with zipfile.ZipFile(roots["spreadsheet"] / "book.xlsx", "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", "<workbook/>")
    connection = sqlite3.connect(roots["sqlite"] / "result.sqlite")
    connection.execute("CREATE TABLE result (value REAL)")
    connection.close()
    (roots["su2"] / "history.csv").write_text(
        '"Inner_Iter","CL","CD"\n1,0.5,0.02\n', encoding="utf-8"
    )

    for profile, root in roots.items():
        statuses, diagnostics = evaluate_application(profile, root)
        assert statuses["artifact_structure"] == "PASSED", (profile, diagnostics)
        assert statuses["native_validation"] == "PASSED", (profile, diagnostics)


def test_numerical_validator_uses_hidden_tolerances(tmp_path: Path) -> None:
    output = tmp_path / "qe"
    output.mkdir()
    (output / "scf.out").write_text(
        "! total energy = -10.0004 Ry\nJOB DONE.\n", encoding="utf-8"
    )
    validation = {
        "metrics": [
            {
                "id": "total_energy",
                "path": "**/scf.out",
                "regex": r"total energy\s*=\s*([-+0-9.]+)",
                "expected": -10.0,
                "abs_tol": 0.001,
            }
        ]
    }

    statuses, diagnostics = evaluate_application(
        "quantum_espresso", output, validation
    )

    assert statuses == {
        "artifact_structure": "PASSED",
        "native_validation": "PASSED",
        "numerical_accuracy": "PASSED",
    }, diagnostics
