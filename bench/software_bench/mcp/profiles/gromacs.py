from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    path = _path("Workspace-relative GROMACS input path.")
    return {
        "name": "software-bench-gromacs",
        "version": "1.0.0",
        "description": (
            "Compact molecular simulation preparation, execution, and validation profile."
        ),
        "tools": [
            {
                "name": "gromacs_check",
                "description": "Validate a GROMACS structure or trajectory.",
                "input_schema": _schema({"input": path}, ["input"]),
                "command": ["gmx", "check", "-f", "{input}"], "timeout_seconds": 300,
            },
            {
                "name": "gromacs_prepare_run",
                "description": (
                    "Compile topology, coordinates, and MDP settings into a run input file."
                ),
                "input_schema": _schema(
                    {
                        "mdp": _path("MDP parameter file."),
                        "coordinates": path,
                        "topology": _path("Topology file."),
                        "output": _path("Output TPR file."),
                    },
                    ["mdp", "coordinates", "topology", "output"],
                ),
                "command": [
                    "gmx", "grompp", "-f", "{mdp}", "-c", "{coordinates}",
                    "-p", "{topology}", "-o", "{output}",
                ],
                "timeout_seconds": 600, "mutating": True,
            },
            {
                "name": "gromacs_run",
                "description": "Run or resume a molecular dynamics simulation by deffnm prefix.",
                "input_schema": _schema(
                    {"prefix": _path("Run prefix without extension."), "arguments": _arguments()},
                    ["prefix", "arguments"],
                ),
                "command": ["gmx", "mdrun", "-deffnm", "{prefix}", "{arguments...}"],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "gromacs_report_version", "description": "Report GROMACS build details.",
                "input_schema": _schema({}, []), "command": ["gmx", "--version"],
            },
            {
                "name": "gromacs_build_topology",
                "description": (
                    "Generate a topology and processed structure from an input structure."
                ),
                "input_schema": _schema(
                    {
                        "input": path, "output": _path("Processed structure path."),
                        "topology": _path("Output topology path."),
                        "forcefield": {"type": "string", "minLength": 1},
                        "water": {"type": "string", "minLength": 1},
                    },
                    ["input", "output", "topology", "forcefield", "water"],
                ),
                "command": [
                    "gmx", "pdb2gmx", "-f", "{input}", "-o", "{output}",
                    "-p", "{topology}", "-ff", "{forcefield}", "-water", "{water}",
                ],
                "timeout_seconds": 600, "mutating": True,
            },
            {
                "name": "gromacs_define_box",
                "description": "Place a structure in a simulation box with a chosen geometry.",
                "input_schema": _schema(
                    {
                        "input": path, "output": _path("Boxed structure path."),
                        "box_type": {"type": "string", "minLength": 1},
                        "distance": {"type": "number"},
                    },
                    ["input", "output", "box_type", "distance"],
                ),
                "command": [
                    "gmx", "editconf", "-f", "{input}", "-o", "{output}",
                    "-bt", "{box_type}", "-d", "{distance}",
                ],
                "timeout_seconds": 300, "mutating": True,
            },
            {
                "name": "gromacs_solvate",
                "description": "Solvate a boxed structure and update its topology.",
                "input_schema": _schema(
                    {
                        "coordinates": path, "solvent": path,
                        "output": _path("Solvated structure path."),
                        "topology": _path("Topology to update."),
                    },
                    ["coordinates", "solvent", "output", "topology"],
                ),
                "command": [
                    "gmx", "solvate", "-cp", "{coordinates}", "-cs", "{solvent}",
                    "-o", "{output}", "-p", "{topology}",
                ],
                "timeout_seconds": 900, "mutating": True,
            },
            {
                "name": "gromacs_extend_run",
                "description": "Create an extended run input from an existing TPR file.",
                "input_schema": _schema(
                    {
                        "input": path, "output": _path("Extended TPR output."),
                        "picoseconds": {"type": "number"},
                    },
                    ["input", "output", "picoseconds"],
                ),
                "command": [
                    "gmx", "convert-tpr", "-s", "{input}", "-o", "{output}",
                    "-extend", "{picoseconds}",
                ],
                "timeout_seconds": 300, "mutating": True,
            },
        ],
    }
