from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    case = _path("Workspace-relative OpenFOAM case directory.")
    return {
        "name": "software-bench-openfoam",
        "version": "1.0.0",
        "description": "OpenFOAM case setup, mesh checking, solver execution, and post-processing.",
        "tools": [
            {
                "name": "openfoam_check_mesh", "description": "Run checkMesh for a case.",
                "input_schema": _schema({"case": case}, ["case"]),
                "command": ["checkMesh", "-case", "{case}"], "timeout_seconds": 300,
            },
            {
                "name": "openfoam_build_mesh", "description": "Generate a blockMesh mesh.",
                "input_schema": _schema({"case": case}, ["case"]),
                "command": ["blockMesh", "-case", "{case}"], "timeout_seconds": 300,
                "mutating": True,
            },
            {
                "name": "openfoam_run_solver",
                "description": "Run an allow-listed OpenFOAM solver.",
                "input_schema": _schema(
                    {
                        "solver": {
                            "type": "string",
                            "enum": ["rhoCentralFoam", "simpleFoam", "pisoFoam", "icoFoam"],
                        },
                        "case": case,
                    },
                    ["solver", "case"],
                ),
                "command": ["{solver}", "-case", "{case}"], "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "openfoam_post_process",
                "description": "Run a named postProcess function object.",
                "input_schema": _schema(
                    {"case": case, "function": {"type": "string", "minLength": 1}},
                    ["case", "function"],
                ),
                "command": ["postProcess", "-case", "{case}", "-func", "{function}"],
                "timeout_seconds": 600, "mutating": True,
            },
            {
                "name": "openfoam_read_dictionary",
                "description": "Read one entry from an OpenFOAM dictionary.",
                "input_schema": _schema(
                    {
                        "case": case,
                        "dictionary": _path("Dictionary path within the workspace."),
                        "entry": {"type": "string", "minLength": 1},
                    },
                    ["case", "dictionary", "entry"],
                ),
                "command": [
                    "foamDictionary", "-case", "{case}", "{dictionary}",
                    "-entry", "{entry}", "-value",
                ],
            },
            {
                "name": "openfoam_list_times",
                "description": "List simulation time directories recognized by OpenFOAM.",
                "input_schema": _schema({"case": case}, ["case"]),
                "command": ["foamListTimes", "-case", "{case}"],
            },
            {
                "name": "openfoam_decompose",
                "description": "Decompose a case for parallel execution.",
                "input_schema": _schema({"case": case}, ["case"]),
                "command": ["decomposePar", "-case", "{case}", "-force"],
                "timeout_seconds": 600, "mutating": True,
            },
            {
                "name": "openfoam_reconstruct",
                "description": "Reconstruct fields produced by a decomposed parallel run.",
                "input_schema": _schema({"case": case}, ["case"]),
                "command": ["reconstructPar", "-case", "{case}"],
                "timeout_seconds": 900, "mutating": True,
            },
            {
                "name": "openfoam_initialize_potential_flow",
                "description": "Initialize velocity and flux with potentialFoam.",
                "input_schema": _schema({"case": case}, ["case"]),
                "command": ["potentialFoam", "-case", "{case}"],
                "timeout_seconds": 600, "mutating": True,
            },
            {
                "name": "openfoam_export_vtk",
                "description": "Export case fields and mesh to VTK for downstream analysis.",
                "input_schema": _schema(
                    {"case": case, "arguments": _arguments("Additional foamToVTK options.")},
                    ["case", "arguments"],
                ),
                "command": ["foamToVTK", "-case", "{case}", "{arguments...}"],
                "timeout_seconds": 900, "mutating": True,
            },
        ],
    }
