from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping


APPLICATION_RULES: Mapping[str, Mapping[str, Any]] = {
    "blender": {"artifacts": ["**/*.blend"], "kind": "blender"},
    "energyplus": {
        "artifacts": ["**/eplusout.err", "**/eplusout.sql"],
        "logs": ["**/eplusout.err"],
        "success": ["0 Severe Errors"],
        "fatal": ["EnergyPlus Terminated--Fatal Error Detected"],
    },
    "freecad": {"artifacts": ["**/*.FCStd", "**/*.fcstd"], "kind": "freecad"},
    "gromacs": {
        "artifacts": ["**/*.tpr", "**/*.gro", "**/*.xtc", "**/*.log"],
        "logs": ["**/*.log"],
        "success": ["Finished mdrun", "GROMACS reminds you"],
        "fatal": ["Fatal error:"],
    },
    "kubernetes": {
        "artifacts": ["**/results.json", "**/state.json", "**/*.yaml", "**/*.yml"],
        "kind": "json-or-yaml",
    },
    "lammps": {
        "artifacts": ["**/log.lammps", "**/*.lammpstrj", "**/*.restart"],
        "logs": ["**/log.lammps"],
        "success": ["Loop time of"],
        "fatal": ["ERROR:"],
    },
    "modflow": {
        "artifacts": ["**/*.lst", "**/*.hds", "**/*.bud", "**/*.cbc"],
        "logs": ["**/*.lst"],
        "success": ["Normal termination of simulation"],
        "fatal": ["FAILED TO CONVERGE", "ERROR REPORT"],
    },
    "openmc": {
        "artifacts": ["**/statepoint.*.h5", "**/depletion_results.h5"],
        "kind": "hdf5",
    },
    "openfoam": {
        "artifacts": ["**/system/controlDict", "**/constant/polyMesh", "**/log.*"],
        "logs": ["**/log.*"],
        "success": ["End"],
        "fatal": ["FOAM FATAL ERROR", "FOAM exiting"],
    },
    "paraview": {
        "artifacts": ["**/*.vtk", "**/*.vtu", "**/*.vtp", "**/*.pvd", "**/*.csv"],
        "kind": "vtk",
    },
    "qgis": {
        "artifacts": ["**/*.gpkg", "**/*.qgz", "**/*.tif", "**/*.geojson"],
        "kind": "geospatial",
    },
    "quantum_espresso": {
        "artifacts": ["**/*.out", "**/*.xml", "**/*.save"],
        "logs": ["**/*.out"],
        "success": ["JOB DONE."],
        "fatal": ["Error in routine", "%%%%%%%%%%%%"],
    },
    "spreadsheet": {
        "artifacts": ["**/*.xlsx", "**/*.ods"],
        "kind": "spreadsheet",
    },
    "sqlite": {
        "artifacts": ["**/*.sqlite", "**/*.sqlite3", "**/*.db"],
        "kind": "sqlite",
    },
    "su2": {
        "artifacts": ["**/history.csv", "**/history.dat", "**/restart*.dat", "**/*.vtk"],
        "logs": ["**/history.csv", "**/history.dat"],
        "fatal": ["DIVERGENCE", "NaN"],
        "kind": "numeric-table",
    },
}


def evaluate_application(
    profile: str,
    submission: Path,
    validation: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], list[str]]:
    rules = APPLICATION_RULES.get(profile)
    if rules is None:
        raise ValueError(f"unknown application validation profile: {profile}")
    spec = dict(validation or {})
    diagnostics: list[str] = []
    artifacts = _matches(submission, rules.get("artifacts", []))
    required = spec.get("required_paths", [])
    forbidden = spec.get("forbidden_paths", [])
    structure_ok = bool(artifacts)
    if not artifacts:
        diagnostics.append("no profile-recognized artifact was submitted")
    for pattern in required:
        if not _matches(submission, [pattern]):
            structure_ok = False
            diagnostics.append(f"required path is missing: {pattern}")
    for pattern in forbidden:
        if _matches(submission, [pattern]):
            structure_ok = False
            diagnostics.append(f"forbidden path is present: {pattern}")

    native_ok = structure_ok
    if native_ok:
        native_ok = _validate_native(profile, submission, artifacts, rules, diagnostics)
    for check in spec.get("content_assertions", []):
        native_ok = _content_assertion(submission, check, diagnostics) and native_ok

    metrics = spec.get("metrics", [])
    numerical_status = "SKIPPED"
    if metrics:
        numerical_status = "PASSED"
        for metric in metrics:
            if not _metric_matches(submission, metric, diagnostics):
                numerical_status = "FAILED"
    return (
        {
            "artifact_structure": "PASSED" if structure_ok else "FAILED",
            "native_validation": "PASSED" if native_ok else "FAILED",
            "numerical_accuracy": numerical_status,
        },
        diagnostics,
    )


def _validate_native(
    profile: str,
    root: Path,
    artifacts: list[Path],
    rules: Mapping[str, Any],
    diagnostics: list[str],
) -> bool:
    ok = True
    log_files = _matches(root, rules.get("logs", []))
    texts = [_read_text(path) for path in log_files]
    if rules.get("logs") and not log_files:
        diagnostics.append(f"{profile}: required application log is missing")
        ok = False
    fatal = rules.get("fatal", [])
    if any(token.lower() in text.lower() for token in fatal for text in texts):
        diagnostics.append(f"{profile}: fatal marker found in application log")
        ok = False
    success = rules.get("success", [])
    if log_files and success and not any(
        token.lower() in text.lower() for token in success for text in texts
    ):
        diagnostics.append(f"{profile}: completion marker missing from application log")
        ok = False
    kind = rules.get("kind")
    try:
        if kind == "blender":
            ok = _has_magic(artifacts, b"BLENDER") and ok
        elif kind == "freecad":
            ok = _zip_contains(artifacts, "Document.xml") and ok
        elif kind == "spreadsheet":
            ok = _spreadsheet_valid(artifacts) and ok
        elif kind == "sqlite":
            ok = _sqlite_valid(artifacts) and ok
        elif kind == "hdf5":
            ok = _has_magic(artifacts, b"\x89HDF\r\n\x1a\n") and ok
        elif kind == "vtk":
            ok = _vtk_valid(artifacts) and ok
        elif kind == "geospatial":
            ok = _geospatial_valid(artifacts) and ok
        elif kind == "json-or-yaml":
            ok = _state_file_valid(artifacts) and ok
        elif kind == "numeric-table":
            ok = _numeric_table_valid(artifacts) and ok
    except (OSError, ValueError, zipfile.BadZipFile, sqlite3.DatabaseError) as error:
        diagnostics.append(f"{profile}: native artifact validation failed: {error}")
        return False
    if not ok and kind:
        diagnostics.append(f"{profile}: artifact does not satisfy its native format")
    return ok


def _matches(root: Path, patterns: Any) -> list[Path]:
    result: set[Path] = set()
    for pattern in patterns:
        result.update(path for path in root.glob(pattern) if path.is_file())
        result.update(path for path in root.glob(pattern) if path.is_dir())
    return sorted(result)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _has_magic(paths: list[Path], magic: bytes) -> bool:
    return any(path.is_file() and path.read_bytes()[: len(magic)] == magic for path in paths)


def _zip_contains(paths: list[Path], member: str) -> bool:
    for path in paths:
        if path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                if member in archive.namelist():
                    return True
    return False


def _spreadsheet_valid(paths: list[Path]) -> bool:
    for path in paths:
        if not path.is_file() or not zipfile.is_zipfile(path):
            continue
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        if path.suffix.lower() == ".xlsx" and {"[Content_Types].xml", "xl/workbook.xml"} <= names:
            return True
        if path.suffix.lower() == ".ods" and "content.xml" in names:
            return True
    return False


def _sqlite_valid(paths: list[Path]) -> bool:
    for path in paths:
        if path.suffix.lower() not in {".sqlite", ".sqlite3", ".db", ".gpkg"}:
            continue
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            if connection.execute("PRAGMA integrity_check").fetchone() == ("ok",):
                return True
        finally:
            connection.close()
    return False


def _vtk_valid(paths: list[Path]) -> bool:
    for path in paths:
        if path.suffix.lower() == ".csv" and path.stat().st_size > 0:
            return True
        head = path.read_bytes()[:256].lstrip()
        if head.startswith(b"# vtk DataFile") or b"<VTKFile" in head:
            return True
    return False


def _geospatial_valid(paths: list[Path]) -> bool:
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".gpkg" and _sqlite_valid([path]):
            return True
        if suffix == ".qgz" and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                if any(name.lower().endswith(".qgs") for name in archive.namelist()):
                    return True
        if suffix == ".tif" and path.read_bytes()[:4] in {b"II*\x00", b"MM\x00*"}:
            return True
        if suffix == ".geojson":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping) and value.get("type") in {
                "FeatureCollection", "Feature", "GeometryCollection",
            }:
                return True
    return False


def _state_file_valid(paths: list[Path]) -> bool:
    for path in paths:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, (Mapping, list)):
                return True
        elif path.suffix.lower() in {".yaml", ".yml"}:
            text = _read_text(path)
            if "apiVersion:" in text and "kind:" in text:
                return True
    return False


def _numeric_table_valid(paths: list[Path]) -> bool:
    for path in paths:
        if path.suffix.lower() not in {".csv", ".dat"}:
            continue
        values = re.findall(
            r"(?<![A-Za-z])[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?",
            _read_text(path),
        )
        if values and all(math.isfinite(float(value)) for value in values):
            return True
    return False


def _content_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    if not isinstance(value, Mapping):
        diagnostics.append("content assertion must be an object")
        return False
    pattern = value.get("path")
    regex = value.get("regex")
    if not isinstance(pattern, str) or not isinstance(regex, str):
        diagnostics.append("content assertion requires path and regex")
        return False
    files = _matches(root, [pattern])
    matched = any(re.search(regex, _read_text(path), re.MULTILINE) for path in files)
    if not matched:
        diagnostics.append(f"content assertion failed: {pattern} / {regex}")
    return matched


def _metric_matches(root: Path, value: Any, diagnostics: list[str]) -> bool:
    if not isinstance(value, Mapping):
        diagnostics.append("metric must be an object")
        return False
    metric_id = str(value.get("id", "metric"))
    path = value.get("path")
    pattern = value.get("regex")
    expected = value.get("expected")
    if not isinstance(path, str) or not isinstance(pattern, str) or not isinstance(
        expected, (int, float)
    ):
        diagnostics.append(f"{metric_id}: invalid metric declaration")
        return False
    values: list[float] = []
    for candidate in _matches(root, [path]):
        values.extend(float(match) for match in re.findall(pattern, _read_text(candidate)))
    if not values:
        diagnostics.append(f"{metric_id}: no numeric value matched")
        return False
    observed = values[-1] if value.get("select", "last") == "last" else values[0]
    absolute = float(value.get("abs_tol", 0.0))
    relative = float(value.get("rel_tol", 0.0))
    passed = math.isclose(observed, float(expected), abs_tol=absolute, rel_tol=relative)
    if not passed:
        diagnostics.append(
            f"{metric_id}: observed={observed} expected={expected} "
            f"abs_tol={absolute} rel_tol={relative}"
        )
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(APPLICATION_RULES), required=True)
    parser.add_argument("--submission", type=Path, default=Path("."))
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args(argv)
    validation = None
    if args.validation is not None:
        validation = json.loads(args.validation.read_text(encoding="utf-8"))
    statuses, diagnostics = evaluate_application(
        args.profile, args.submission.resolve(), validation
    )
    for diagnostic in diagnostics:
        print(diagnostic, file=sys.stderr)
    print(json.dumps(statuses, sort_keys=True))
    return 0 if all(value != "FAILED" for value in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
