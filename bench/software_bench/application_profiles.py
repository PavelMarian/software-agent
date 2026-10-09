from __future__ import annotations

from typing import Mapping

from software_bench.core.models import ValidationError


APPLICATION_IMAGES: Mapping[str, str] = {
    "blender": "software-bench/blender:v1",
    "energyplus": "software-bench/energyplus:v1",
    "freecad": "software-bench/freecad:v1",
    "gromacs": "software-bench/gromacs:v1",
    "kubernetes": "software-bench/kubernetes:v1",
    "lammps": "software-bench/lammps:v1",
    "modflow": "software-bench/modflow6:v1",
    "openmc": "software-bench/openmc:v1",
    "openfoam": "software-bench/openfoam:v1",
    "paraview": "software-bench/paraview:v1",
    "qgis": "software-bench/qgis:v1",
    "quantum_espresso": "software-bench/quantum-espresso:v1",
    "spreadsheet": "software-bench/spreadsheet:v1",
    "sqlite": "software-bench/sqlite:v1",
    "su2": "software-bench/su2:v1",
}


def environment_image(profile_id: str) -> str:
    try:
        return APPLICATION_IMAGES[profile_id]
    except KeyError as error:
        raise ValidationError(f"unknown environment profile: {profile_id}") from error
