from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    return {
        "name": "software-bench-lammps",
        "version": "1.0.0",
        "description": "LAMMPS input validation and molecular dynamics execution.",
        "tools": [
            {
                "name": "lammps_run",
                "description": "Run a LAMMPS input script with optional command-line variables.",
                "input_schema": _schema(
                    {"input": _path("LAMMPS input script."), "arguments": _arguments()},
                    ["input", "arguments"],
                ),
                "command": ["lmp", "-in", "{input}", "{arguments...}"],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "lammps_report_version", "description": "Report LAMMPS build information.",
                "input_schema": _schema({}, []), "command": ["lmp", "-help"],
            },
            {
                "name": "lammps_run_partitioned",
                "description": "Run a LAMMPS script with an explicit processor partition layout.",
                "input_schema": _schema(
                    {
                        "partition": {"type": "string", "minLength": 1},
                        "input": _path("LAMMPS input script."),
                        "arguments": _arguments(),
                    },
                    ["partition", "input", "arguments"],
                ),
                "command": [
                    "lmp", "-partition", "{partition}", "-in", "{input}", "{arguments...}",
                ],
                "timeout_seconds": 1800, "mutating": True,
            },
        ],
    }
