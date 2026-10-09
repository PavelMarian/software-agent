from __future__ import annotations

from typing import Callable, Mapping

from software_bench.mcp.profiles.blender import build as blender
from software_bench.mcp.profiles.energyplus import build as energyplus
from software_bench.mcp.profiles.freecad import build as freecad
from software_bench.mcp.profiles.gromacs import build as gromacs
from software_bench.mcp.profiles.kubernetes import build as kubernetes
from software_bench.mcp.profiles.lammps import build as lammps
from software_bench.mcp.profiles.modflow import build as modflow
from software_bench.mcp.profiles.openmc import build as openmc
from software_bench.mcp.profiles.openfoam import build as openfoam
from software_bench.mcp.profiles.paraview import build as paraview
from software_bench.mcp.profiles.qgis import build as qgis
from software_bench.mcp.profiles.quantum_espresso import build as quantum_espresso
from software_bench.mcp.profiles.spreadsheet import build as spreadsheet
from software_bench.mcp.profiles.sqlite import build as sqlite
from software_bench.mcp.profiles.su2 import build as su2

PROFILE_BUILDERS: Mapping[str, Callable[[], dict[str, object]]] = {
    "blender": blender,
    "energyplus": energyplus,
    "freecad": freecad,
    "gromacs": gromacs,
    "kubernetes": kubernetes,
    "lammps": lammps,
    "modflow": modflow,
    "openmc": openmc,
    "openfoam": openfoam,
    "paraview": paraview,
    "qgis": qgis,
    "quantum_espresso": quantum_espresso,
    "spreadsheet": spreadsheet,
    "sqlite": sqlite,
    "su2": su2,
}

__all__ = ["PROFILE_BUILDERS"]
