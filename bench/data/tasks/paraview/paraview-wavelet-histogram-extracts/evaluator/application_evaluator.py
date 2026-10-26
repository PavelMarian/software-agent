from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen


APPLICATION_RULES: Mapping[str, Mapping[str, Any]] = {
    "freecad": {"artifacts": ["**/*.FCStd", "**/*.fcstd"], "kind": "freecad"},
    "gromacs": {
        "artifacts": ["**/*.tpr", "**/*.gro", "**/*.xtc", "**/*.log"],
        "logs": ["**/*.log"],
        "success": ["Finished mdrun", "GROMACS reminds you"],
        "fatal": ["Fatal error:"],
    },
    "godot": {"artifacts": ["**/project.godot", "**/*.godot", "**/results.json"], "kind": "godot"},
    "impress": {"artifacts": ["**/*.odp", "**/*.pptx", "**/*.pdf"], "kind": "presentation"},
    "inkscape": {"artifacts": ["**/*.svg", "**/*.svgz", "**/*.pdf"], "kind": "vector"},
    "kdenlive": {"artifacts": ["**/*.kdenlive", "**/*.mlt", "**/*.mp4", "**/*.webm"], "kind": "timeline"},
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
        "artifacts": [
            "**/*.vtk",
            "**/*.vtu",
            "**/*.vtp",
            "**/*.vti",
            "**/*.vtm",
            "**/*.vtmb",
            "**/*.pvd",
            "**/*.pvtu",
            "**/*.pvtp",
            "**/*.csv",
            "**/*.png",
        ],
        "kind": "vtk",
    },
    "qgis": {
        "artifacts": ["**/*.gpkg", "**/*.qgz", "**/*.tif", "**/*.geojson", "**/results.json"],
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
    "servicenow": {"artifacts": ["**/state.json", "**/results.json", "**/submission.json"], "kind": "json-or-yaml"},
    "writer": {"artifacts": ["**/*.odt", "**/*.docx", "**/*.pdf"], "kind": "document"},
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
    required_any = spec.get("required_any_paths", [])
    forbidden = spec.get("forbidden_paths", [])
    structure_ok = bool(artifacts)
    if not artifacts:
        diagnostics.append("no profile-recognized artifact was submitted")
    for pattern in required:
        if not _matches(submission, [pattern]):
            structure_ok = False
            diagnostics.append(f"required path is missing: {pattern}")
    if required_any and not _matches(submission, required_any):
        structure_ok = False
        diagnostics.append(
            "none of the alternative required paths is present: " + ", ".join(required_any)
        )
    for pattern in forbidden:
        if _matches(submission, [pattern]):
            structure_ok = False
            diagnostics.append(f"forbidden path is present: {pattern}")

    native_ok = structure_ok
    if native_ok:
        native_ok = _validate_native(profile, submission, artifacts, rules, diagnostics)
    semantic_checks = (
        ("content_assertions", _content_assertion),
        ("json_assertions", _json_assertion),
        ("sqlite_assertions", _sqlite_assertion),
        ("file_hashes", _file_hash_assertion),
        ("reference_file_assertions", _reference_file_assertion),
        ("reference_metrics", _reference_metric_assertion),
        ("archive_xml_assertions", _archive_xml_assertion),
        ("json_reference_assertions", _json_reference_assertion),
        ("csv_reference_assertions", _csv_reference_assertion),
        ("servicenow_assertions", _servicenow_assertion),
        ("freecad_geometry_assertions", _freecad_geometry_assertion),
        ("forbidden_file_hashes", _forbidden_file_hash_assertion),
    )
    semantic_requested = any(spec.get(name) for name, _ in semantic_checks)
    semantic_ok = True
    for name, evaluator in semantic_checks:
        for check in spec.get(name, []):
            semantic_ok = evaluator(submission, check, diagnostics) and semantic_ok
    native_ok = semantic_ok and native_ok

    metrics = spec.get("metrics", [])
    numerical_status = "SKIPPED"
    if metrics:
        numerical_status = "PASSED"
        for metric in metrics:
            if not _metric_matches(submission, metric, diagnostics):
                numerical_status = "FAILED"
    statuses = {
        "artifact_structure": "PASSED" if structure_ok else "FAILED",
        "native_validation": "PASSED" if native_ok else "FAILED",
        "numerical_accuracy": numerical_status,
    }
    if semantic_requested:
        statuses["semantic_correctness"] = "PASSED" if semantic_ok else "FAILED"
    return statuses, diagnostics


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
        if kind == "freecad":
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
        elif kind == "godot":
            ok = _godot_valid(artifacts) and ok
        elif kind == "vector":
            ok = _vector_valid(artifacts) and ok
        elif kind == "timeline":
            ok = _timeline_valid(artifacts) and ok
        elif kind == "document":
            ok = _document_valid(artifacts) and ok
        elif kind == "presentation":
            ok = _presentation_valid(artifacts) and ok
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
    xml_types = {
        ".vtu": "UnstructuredGrid",
        ".vtp": "PolyData",
        ".vti": "ImageData",
        ".vtm": "vtkMultiBlockDataSet",
        ".vtmb": "vtkMultiBlockDataSet",
        ".pvd": "Collection",
        ".pvtu": "PUnstructuredGrid",
        ".pvtp": "PPolyData",
    }
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".png" and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n":
            return True
        if suffix == ".csv" and path.stat().st_size > 0:
            return True
        head = path.read_bytes()[:256].lstrip()
        if suffix == ".vtk" and head.startswith(b"# vtk DataFile"):
            return True
        if suffix not in xml_types:
            continue
        try:
            root = element_tree.parse(path).getroot()
        except element_tree.ParseError:
            continue
        if root.tag.rsplit("}", 1)[-1] != "VTKFile":
            continue
        if root.attrib.get("type") != xml_types[suffix]:
            continue
        references = [
            node.attrib["file"]
            for node in root.iter()
            if "file" in node.attrib
        ]
        if suffix in {".vtm", ".vtmb", ".pvd", ".pvtu", ".pvtp"}:
            if references and all((path.parent / reference).is_file() for reference in references):
                return True
            continue
        if any(node.tag.rsplit("}", 1)[-1] == "Piece" for node in root.iter()):
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
        if path.name == "results.json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping) and isinstance(value.get("passed"), bool):
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


def _godot_valid(paths: list[Path]) -> bool:
    """Accept a parseable Godot project or an explicit structured run result."""
    for path in paths:
        if path.name == "project.godot" and "[application]" in _read_text(path):
            return True
        if path.suffix.lower() == ".json" and _state_file_valid([path]):
            return True
    return False


def _vector_valid(paths: list[Path]) -> bool:
    """Check that an SVG artifact has an SVG root or that a PDF was produced."""
    for path in paths:
        if path.suffix.lower() in {".svg", ".svgz"} and "<svg" in _read_text(path)[:1024].lower():
            return True
        if path.suffix.lower() == ".pdf" and _has_magic([path], b"%PDF-"):
            return True
    return False


def _timeline_valid(paths: list[Path]) -> bool:
    """Check XML timeline projects or non-empty rendered media."""
    for path in paths:
        if path.suffix.lower() in {".kdenlive", ".mlt"} and "<mlt" in _read_text(path)[:1024].lower():
            return True
        if path.suffix.lower() in {".mp4", ".webm"} and path.stat().st_size > 0:
            return True
    return False


def _document_valid(paths: list[Path]) -> bool:
    """Check ODT/DOCX packages or a generated PDF document."""
    for path in paths:
        if path.suffix.lower() == ".pdf" and _has_magic([path], b"%PDF-"):
            return True
        if not zipfile.is_zipfile(path):
            continue
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        if (path.suffix.lower() == ".odt" and "content.xml" in names) or (
            path.suffix.lower() == ".docx" and "word/document.xml" in names
        ):
            return True
    return False


def _presentation_valid(paths: list[Path]) -> bool:
    """Check ODP/PPTX packages or a generated PDF presentation."""
    for path in paths:
        if path.suffix.lower() == ".pdf" and _has_magic([path], b"%PDF-"):
            return True
        if not zipfile.is_zipfile(path):
            continue
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        if (path.suffix.lower() == ".odp" and "content.xml" in names) or (
            path.suffix.lower() == ".pptx" and "ppt/presentation.xml" in names
        ):
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


def _json_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Compare a JSON-pointer value against a hidden expected value."""
    if not isinstance(value, Mapping):
        diagnostics.append("JSON assertion must be an object")
        return False
    pattern = value.get("path")
    pointer = value.get("pointer", "")
    if not isinstance(pattern, str) or not isinstance(pointer, str) or "expected" not in value:
        diagnostics.append("JSON assertion requires path, pointer, and expected")
        return False
    for path in _matches(root, [pattern]):
        try:
            observed: Any = json.loads(path.read_text(encoding="utf-8"))
            for token in _json_pointer_tokens(pointer):
                if isinstance(observed, list):
                    observed = observed[int(token)]
                elif isinstance(observed, Mapping):
                    observed = observed[token]
                else:
                    raise KeyError(token)
        except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError):
            continue
        if observed == value["expected"]:
            return True
    diagnostics.append(f"JSON assertion failed: {pattern} {pointer}")
    return False


def _json_pointer_tokens(pointer: str) -> tuple[str, ...]:
    """Parse a restricted RFC 6901 pointer without exposing evaluator internals."""
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with '/'")
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))


def _sqlite_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Execute a hidden read-only SQL assertion against a submitted database."""
    if not isinstance(value, Mapping):
        diagnostics.append("SQLite assertion must be an object")
        return False
    pattern = value.get("path")
    query = value.get("query")
    expected = value.get("expected_rows")
    if not isinstance(pattern, str) or not isinstance(query, str) or not isinstance(expected, list):
        diagnostics.append("SQLite assertion requires path, query, and expected_rows")
        return False
    normalized = [tuple(row) if isinstance(row, list) else row for row in expected]
    for path in _matches(root, [pattern]):
        if path.suffix.lower() not in {".sqlite", ".sqlite3", ".db", ".gpkg"}:
            continue
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                rows = connection.execute(query).fetchall()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            continue
        if rows == normalized:
            return True
    diagnostics.append(f"SQLite assertion failed: {pattern}")
    return False


def _file_hash_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Check a hidden SHA-256 digest for an exact required deliverable."""
    if not isinstance(value, Mapping):
        diagnostics.append("file hash assertion must be an object")
        return False
    pattern = value.get("path")
    expected = value.get("sha256")
    if not isinstance(pattern, str) or not isinstance(expected, str):
        diagnostics.append("file hash assertion requires path and sha256")
        return False
    import hashlib

    for path in _matches(root, [pattern]):
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return True
    diagnostics.append(f"file hash assertion failed: {pattern}")
    return False


def _forbidden_file_hash_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Require the submitted artifact to differ from a supplied seed digest."""
    if not isinstance(value, Mapping):
        diagnostics.append("forbidden file hash assertion must be an object")
        return False
    pattern = value.get("path")
    forbidden = value.get("sha256")
    if not isinstance(pattern, str) or not isinstance(forbidden, str):
        diagnostics.append("forbidden file hash assertion requires path and sha256")
        return False
    import hashlib

    files = [path for path in _matches(root, [pattern]) if path.is_file()]
    passed = bool(files) and all(
        hashlib.sha256(path.read_bytes()).hexdigest() != forbidden for path in files
    )
    if not passed:
        diagnostics.append(f"submitted artifact is unchanged from seed: {pattern}")
    return passed


def _reference_file_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Compare a submitted artifact byte-for-byte with a hidden reference file."""
    if not isinstance(value, Mapping):
        diagnostics.append("reference file assertion must be an object")
        return False
    pattern = value.get("path")
    reference = value.get("reference")
    references = value.get("references")
    if isinstance(reference, str):
        names = [reference]
    elif isinstance(references, list) and all(isinstance(item, str) for item in references):
        names = references
    else:
        names = []
    if not isinstance(pattern, str) or not names:
        diagnostics.append("reference file assertion requires path and reference(s)")
        return False
    submitted = [path.read_bytes() for path in _matches(root, [pattern]) if path.is_file()]
    for name in names:
        try:
            expected = (root / ".benchmark" / "reference" / name).read_bytes()
        except OSError:
            diagnostics.append(f"hidden reference is missing: {name}")
            continue
        if expected in submitted:
            return True
    diagnostics.append(f"reference file assertion failed: {pattern}")
    return False


def _reference_metric_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Compare a submitted numeric observable with its hidden reference observable."""
    if not isinstance(value, Mapping):
        diagnostics.append("reference metric assertion must be an object")
        return False
    pattern = value.get("path")
    reference = value.get("reference")
    regex = value.get("regex")
    if not all(isinstance(item, str) and item for item in (pattern, reference, regex)):
        diagnostics.append("reference metric requires path, reference, and regex")
        return False
    try:
        expected_values = [float(match) for match in re.findall(regex, _read_text(
            root / ".benchmark" / "reference" / reference
        ))]
    except (OSError, ValueError):
        expected_values = []
    observed_values = [
        float(match)
        for candidate in _matches(root, [pattern])
        for match in re.findall(regex, _read_text(candidate))
    ]
    if not expected_values or not observed_values:
        diagnostics.append(f"reference metric did not match: {pattern}")
        return False
    select = value.get("select", "last")
    if select not in {"first", "last"}:
        diagnostics.append("reference metric select must be first or last")
        return False
    expected = expected_values[0] if select == "first" else expected_values[-1]
    observed = observed_values[0] if select == "first" else observed_values[-1]
    if math.isclose(
        observed,
        expected,
        abs_tol=float(value.get("abs_tol", 0.0)),
        rel_tol=float(value.get("rel_tol", 0.0)),
    ):
        return True
    diagnostics.append(
        f"reference metric mismatch: observed={observed} expected={expected}"
    )
    return False


def _archive_xml_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Compare a named XML member of an archive with a hidden reference archive."""
    if not isinstance(value, Mapping):
        diagnostics.append("archive XML assertion must be an object")
        return False
    pattern, reference, member = (value.get(key) for key in ("path", "reference", "member"))
    if not all(isinstance(item, str) and item for item in (pattern, reference, member)):
        diagnostics.append("archive XML assertion requires path, reference, and member")
        return False
    reference_path = root / ".benchmark" / "reference" / reference
    try:
        expected = _canonical_archive_xml(reference_path, member)
    except (OSError, KeyError, zipfile.BadZipFile, element_tree.ParseError):
        diagnostics.append(f"hidden XML reference is invalid: {reference}")
        return False
    for candidate in _matches(root, [pattern]):
        try:
            if _canonical_archive_xml(candidate, member) == expected:
                return True
        except (OSError, KeyError, zipfile.BadZipFile, element_tree.ParseError):
            continue
    diagnostics.append(f"archive XML assertion failed: {pattern}::{member}")
    return False


def _canonical_archive_xml(path: Path, member: str) -> bytes:
    """Produce a stable representation of one XML archive member."""
    with zipfile.ZipFile(path) as archive:
        root = element_tree.fromstring(archive.read(member))
    for node in root.iter():
        node.attrib = dict(sorted(node.attrib.items()))
        if node.text is not None:
            node.text = " ".join(node.text.split())
        if node.tail is not None:
            node.tail = " ".join(node.tail.split())
    return element_tree.tostring(root, encoding="utf-8")


def _json_reference_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Compare a JSON deliverable with a hidden deterministic replay or state file."""
    if not isinstance(value, Mapping):
        diagnostics.append("JSON reference assertion must be an object")
        return False
    pattern = value.get("path")
    reference = value.get("reference")
    if not isinstance(pattern, str) or not isinstance(reference, str):
        diagnostics.append("JSON reference assertion requires path and reference")
        return False
    try:
        expected = json.loads((root / ".benchmark" / "reference" / reference).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        diagnostics.append(f"hidden JSON reference is invalid: {reference}")
        return False
    for candidate in _matches(root, [pattern]):
        try:
            if json.loads(candidate.read_text("utf-8")) == expected:
                return True
        except (OSError, json.JSONDecodeError):
            continue
    diagnostics.append(f"JSON reference assertion failed: {pattern}")
    return False


def _csv_reference_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Compare CSV headers and numeric cells with a hidden reference using tolerances."""
    if not isinstance(value, Mapping):
        diagnostics.append("CSV reference assertion must be an object")
        return False
    pattern = value.get("path")
    reference = value.get("reference")
    if not isinstance(pattern, str) or not isinstance(reference, str):
        diagnostics.append("CSV reference assertion requires path and reference")
        return False
    try:
        expected = _read_csv(root / ".benchmark" / "reference" / reference)
    except OSError:
        diagnostics.append(f"hidden CSV reference is invalid: {reference}")
        return False
    for candidate in _matches(root, [pattern]):
        try:
            observed = _read_csv(candidate)
        except OSError:
            continue
        if _csv_matches(observed, expected, value):
            return True
    diagnostics.append(f"CSV reference assertion failed: {pattern}")
    return False


def _read_csv(path: Path) -> list[list[str]]:
    """Read a UTF-8 CSV deterministically without application-specific dependencies."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def _csv_matches(observed: list[list[str]], expected: list[list[str]], spec: Mapping[str, Any]) -> bool:
    """Compare rows while accepting configured floating-point roundoff."""
    if len(observed) != len(expected) or any(len(a) != len(b) for a, b in zip(observed, expected)):
        return False
    absolute = float(spec.get("abs_tol", 0.0))
    relative = float(spec.get("rel_tol", 0.0))
    for actual_row, expected_row in zip(observed, expected):
        for actual, wanted in zip(actual_row, expected_row):
            try:
                if math.isclose(float(actual), float(wanted), abs_tol=absolute, rel_tol=relative):
                    continue
            except ValueError:
                pass
            if actual != wanted:
                return False
    return True


def _servicenow_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Query a pinned ServiceNow table and compare only declared record fields."""
    if not isinstance(value, Mapping):
        diagnostics.append("ServiceNow assertion must be an object")
        return False
    base_url = os.environ.get(str(value.get("url_env", "SERVICENOW_URL")), "").rstrip("/")
    token = os.environ.get(str(value.get("token_env", "SERVICENOW_TOKEN")), "")
    table = value.get("table")
    query = value.get("query")
    expected = value.get("expected")
    if not base_url or not token or not all(isinstance(item, str) and item for item in (table, query)) or not isinstance(expected, Mapping):
        diagnostics.append("ServiceNow assertion lacks pinned-instance credentials or query contract")
        return False
    request = Request(
        f"{base_url}/api/now/table/{table}?sysparm_query={query}&sysparm_limit=1",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=float(value.get("timeout_seconds", 20))) as response:
            rows = json.loads(response.read().decode("utf-8")).get("result", [])
    except (URLError, OSError, ValueError, json.JSONDecodeError) as error:
        diagnostics.append(f"ServiceNow state query failed: {error}")
        return False
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        diagnostics.append("ServiceNow state query did not return exactly one record")
        return False
    if all(rows[0].get(key) == wanted for key, wanted in expected.items()):
        return True
    diagnostics.append("ServiceNow record does not satisfy the declared state contract")
    return False


def _freecad_geometry_assertion(root: Path, value: Any, diagnostics: list[str]) -> bool:
    """Compare aggregate solid geometry against a hidden FCStd reference."""
    if not isinstance(value, Mapping):
        diagnostics.append("FreeCAD geometry assertion must be an object")
        return False
    pattern = value.get("path")
    reference = value.get("reference")
    if not isinstance(pattern, str) or not isinstance(reference, str):
        diagnostics.append("FreeCAD geometry assertion requires path and reference")
        return False
    expected = _freecad_signature(root / ".benchmark" / "reference" / reference)
    if expected is None:
        diagnostics.append(f"cannot inspect hidden FreeCAD reference: {reference}")
        return False
    absolute = float(value.get("abs_tol", 1e-6))
    relative = float(value.get("rel_tol", 1e-6))
    for candidate in _matches(root, [pattern]):
        observed = _freecad_signature(candidate)
        if observed is None or observed["solids"] != expected["solids"]:
            continue
        if all(
            math.isclose(observed[key], expected[key], abs_tol=absolute, rel_tol=relative)
            for key in ("volume", "area", "x", "y", "z")
        ):
            return True
    diagnostics.append(f"FreeCAD geometry assertion failed: {pattern}")
    return False


def _freecad_signature(path: Path) -> dict[str, float] | None:
    """Extract aggregate geometry using the application's own headless runtime."""
    executable = shutil.which("FreeCADCmd") or shutil.which("freecadcmd")
    if executable is None or not path.is_file():
        return None
    code = (
        "import FreeCAD,json;d=FreeCAD.open(" + repr(str(path)) + ");"
        "s=[o.Shape for o in d.Objects if hasattr(o,'Shape') and not o.Shape.isNull()];"
        "b=[x.BoundBox for x in s];"
        "print('BENCHSIG '+json.dumps({'solids':sum(x.Solids.__len__() for x in s),"
        "'volume':sum(x.Volume for x in s),'area':sum(x.Area for x in s),"
        "'x':max((x.XMax for x in b),default=0)-min((x.XMin for x in b),default=0),"
        "'y':max((x.YMax for x in b),default=0)-min((x.YMin for x in b),default=0),"
        "'z':max((x.ZMax for x in b),default=0)-min((x.ZMin for x in b),default=0)}))"
    )
    completed = subprocess.run(
        [executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    for line in completed.stdout.splitlines():
        if line.startswith("BENCHSIG "):
            return json.loads(line.removeprefix("BENCHSIG "))
    return None


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
