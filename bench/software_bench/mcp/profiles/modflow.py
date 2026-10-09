from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    simulation = _path("MODFLOW 6 simulation name file.")
    script = _path("Workspace-relative FloPy analysis script.")
    return {
        "name": "software-bench-modflow",
        "version": "1.0.0",
        "description": "Groundwater-flow modeling, budgets, observations, and grid analysis.",
        "tools": [
            {
                "name": "modflow_report_version",
                "description": "Report the installed MODFLOW 6 version.",
                "input_schema": _schema({}, []),
                "command": ["mf6", "-v"],
            },
            {
                "name": "modflow_run_simulation",
                "description": "Run a MODFLOW 6 simulation from its name file.",
                "input_schema": _schema({"simulation": simulation}, ["simulation"]),
                "command": ["mf6", "-s", "{simulation}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "modflow_validate_simulation",
                "description": "Load and validate simulation packages with FloPy.",
                "input_schema": _schema(
                    {"script": script, "simulation": simulation},
                    ["script", "simulation"],
                ),
                "command": ["python", "{script}", "validate", "{simulation}"],
                "timeout_seconds": 300,
            },
            {
                "name": "modflow_build_simulation",
                "description": "Build and write a parameterized MODFLOW simulation with FloPy.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "output": _path("Output simulation directory."),
                        "arguments": _arguments(),
                    },
                    ["script", "output", "arguments"],
                ),
                "command": ["python", "{script}", "{output}", "{arguments...}"],
                "timeout_seconds": 600,
                "mutating": True,
            },
            {
                "name": "modflow_inspect_listing",
                "description": "Extract convergence, solver, and water-balance diagnostics.",
                "input_schema": _schema(
                    {"script": script, "listing": _path("MODFLOW listing file.")},
                    ["script", "listing"],
                ),
                "command": ["python", "{script}", "{listing}"],
                "timeout_seconds": 300,
            },
            {
                "name": "modflow_extract_heads",
                "description": "Extract simulated hydraulic heads to a table or raster.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "heads": _path("Binary head file."),
                        "output": _path("Output head dataset."),
                        "arguments": _arguments("Time, layer, or format selectors."),
                    },
                    ["script", "heads", "output", "arguments"],
                ),
                "command": [
                    "python", "{script}", "{heads}", "{output}", "{arguments...}",
                ],
                "timeout_seconds": 600,
                "mutating": True,
            },
            {
                "name": "modflow_extract_budget",
                "description": "Extract cell-by-cell flow terms and aggregate water budgets.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "budget": _path("Binary cell-budget file."),
                        "output": _path("Output budget table."),
                        "terms": _arguments("Budget record names to extract."),
                    },
                    ["script", "budget", "output", "terms"],
                ),
                "command": [
                    "python", "{script}", "{budget}", "{output}", "{terms...}",
                ],
                "timeout_seconds": 600,
                "mutating": True,
            },
            {
                "name": "modflow_extract_observations",
                "description": "Normalize MODFLOW observation output to a comparison table.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "observations": _path("Observation CSV file."),
                        "output": _path("Normalized output table."),
                    },
                    ["script", "observations", "output"],
                ),
                "command": ["python", "{script}", "{observations}", "{output}"],
                "timeout_seconds": 300,
                "mutating": True,
            },
            {
                "name": "modflow_export_grid",
                "description": "Export a structured or unstructured model grid for GIS analysis.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "simulation": simulation,
                        "output": _path("Output GIS or mesh dataset."),
                    },
                    ["script", "simulation", "output"],
                ),
                "command": ["python", "{script}", "{simulation}", "{output}"],
                "timeout_seconds": 600,
                "mutating": True,
            },
            {
                "name": "modflow_run_zonebudget",
                "description": "Calculate subregional water budgets with ZoneBudget 6.",
                "input_schema": _schema(
                    {"input": _path("ZoneBudget 6 name file.")}, ["input"]
                ),
                "command": ["zbud6", "-in", "{input}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "modflow_compare_heads",
                "description": "Compare head fields and report spatial error statistics.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "reference": _path("Reference head file."),
                        "candidate": _path("Candidate head file."),
                        "output": _path("Comparison report."),
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
