from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from software_bench.core.models import TaskBundle, ValidationError
from software_bench.mcp.models import ApplicationSpec
from software_bench.mcp.profiles import PROFILE_BUILDERS


def builtin_profiles() -> Mapping[str, ApplicationSpec]:
    """Build all benchmark-owned application profiles."""
    return {
        profile_id: ApplicationSpec.from_dict(builder())
        for profile_id, builder in PROFILE_BUILDERS.items()
    }


def automatic_spec(bundle: TaskBundle) -> ApplicationSpec:
    """Select an explicit MCP profile, with conservative legacy inference as fallback."""
    profiles = builtin_profiles()
    explicit = bundle.task.metadata.get("mcp_profile")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValidationError("task metadata mcp_profile must be a non-empty string")
        profile_id = explicit.strip().lower()
        try:
            return profiles[profile_id]
        except KeyError as error:
            raise ValidationError(f"unknown MCP profile: {explicit}") from error

    signals = " ".join(
        filter(
            None,
            (
                bundle.task.target_software or "",
                bundle.provenance.source,
                bundle.environment.image or "",
            ),
        )
    ).lower()
    variant = str(bundle.task.metadata.get("variant", "")).lower()
    source_id = str(bundle.task.metadata.get("source_id", "")).lower()
    matchers: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("openfoam", ("openfoam",)),
        ("paraview", ("paraview", "vtk")),
        ("qgis", ("qgis", "geospatial", "gdal", "grass gis")),
        ("freecad", ("freecad", "cadquery", "parametric cad")),
        ("gromacs", ("gromacs",)),
        ("lammps", ("lammps",)),
        ("quantum_espresso", ("quantum espresso", "quantum_espresso", "qe-suite")),
        ("openmc", ("openmc",)),
        ("su2", ("su2", "stanford university unstructured")),
        ("energyplus", ("energyplus", "building energy")),
        ("modflow", ("modflow", "flopy", "groundwater")),
        ("kubernetes", ("kubernetes", "k8s", "sregym", "helm")),
        ("blender", ("blender",)),
        ("spreadsheet", ("spreadsheet", "libreoffice", "excel", "xlsx")),
        ("sqlite", ("sqlite", "spider2-lite", "spider 2.0 lite")),
    )
    for profile_id, terms in matchers:
        if any(term in signals for term in terms):
            return profiles[profile_id]
    if variant == "lite" and source_id.startswith("local"):
        return profiles["sqlite"]
    raise ValidationError(
        "no safe automatic MCP profile matches this TaskBundle; "
        "set task.metadata.mcp_profile, select a built-in profile, or provide a JSON spec"
    )


def load_spec(path: str | Path) -> ApplicationSpec:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read MCP application spec {source}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValidationError("MCP application spec must contain an object")
    return ApplicationSpec.from_dict(value)


def write_spec(spec: ApplicationSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
