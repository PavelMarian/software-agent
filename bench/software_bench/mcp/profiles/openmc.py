from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    model = _path("OpenMC model directory or XML input path.")
    script = _path("Workspace-relative OpenMC Python workflow script.")
    return {
        "name": "software-bench-openmc",
        "version": "1.0.0",
        "description": "Monte Carlo transport, geometry, tallies, and depletion workflows.",
        "tools": [
            {
                "name": "openmc_report_version",
                "description": "Report the installed OpenMC Python package version.",
                "input_schema": _schema({}, []),
                "command": [
                    "python", "-c", "import openmc; print(openmc.__version__)",
                ],
            },
            {
                "name": "openmc_validate_model",
                "description": "Run a model-validation script without transporting particles.",
                "input_schema": _schema(
                    {"script": script, "model": model}, ["script", "model"]
                ),
                "command": ["python", "{script}", "validate", "{model}"],
                "timeout_seconds": 300,
            },
            {
                "name": "openmc_build_model",
                "description": "Build and export materials, geometry, settings, and tallies.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "output": _path("Output model directory."),
                        "arguments": _arguments(),
                    },
                    ["script", "output", "arguments"],
                ),
                "command": ["python", "{script}", "{output}", "{arguments...}"],
                "timeout_seconds": 600,
                "mutating": True,
            },
            {
                "name": "openmc_run_transport",
                "description": "Run a fixed-source or eigenvalue Monte Carlo calculation.",
                "input_schema": _schema(
                    {"model": model, "arguments": _arguments()},
                    ["model", "arguments"],
                ),
                "command": ["openmc", "{arguments...}", "{model}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "openmc_plot_geometry",
                "description": "Generate geometry plots declared by plots.xml.",
                "input_schema": _schema({"model": model}, ["model"]),
                "command": ["openmc", "-p", "{model}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "openmc_calculate_volumes",
                "description": "Run stochastic material and cell volume calculations.",
                "input_schema": _schema({"model": model}, ["model"]),
                "command": ["openmc", "--volume", "{model}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "openmc_restart",
                "description": "Resume transport from a particle restart file.",
                "input_schema": _schema(
                    {"restart": _path("Particle restart file."), "model": model},
                    ["restart", "model"],
                ),
                "command": ["openmc", "-r", "{restart}", "{model}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "openmc_extract_statepoint",
                "description": "Extract run metadata and global results from a statepoint.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "statepoint": _path("OpenMC statepoint HDF5 file."),
                        "output": _path("Output JSON file."),
                    },
                    ["script", "statepoint", "output"],
                ),
                "command": ["python", "{script}", "{statepoint}", "{output}"],
                "timeout_seconds": 300,
                "mutating": True,
            },
            {
                "name": "openmc_extract_tallies",
                "description": "Extract filtered tally means and uncertainties to a table.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "statepoint": _path("OpenMC statepoint HDF5 file."),
                        "output": _path("Output CSV or JSON table."),
                        "arguments": _arguments("Tally names or extraction options."),
                    },
                    ["script", "statepoint", "output", "arguments"],
                ),
                "command": [
                    "python", "{script}", "{statepoint}", "{output}", "{arguments...}",
                ],
                "timeout_seconds": 600,
                "mutating": True,
            },
            {
                "name": "openmc_run_depletion",
                "description": "Execute a coupled transport-depletion workflow script.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "model": model,
                        "output": _path("Depletion output directory."),
                        "arguments": _arguments(),
                    },
                    ["script", "model", "output", "arguments"],
                ),
                "command": [
                    "python", "{script}", "{model}", "{output}", "{arguments...}",
                ],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "openmc_compare_results",
                "description": "Compare two statepoints or depletion result sets quantitatively.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "reference": _path("Reference result file."),
                        "candidate": _path("Candidate result file."),
                        "output": _path("Comparison report path."),
                    },
                    ["script", "reference", "candidate", "output"],
                ),
                "command": [
                    "python", "{script}", "{reference}", "{candidate}", "{output}",
                ],
                "timeout_seconds": 600,
                "mutating": True,
            },
        ],
    }
