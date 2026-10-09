from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    model = _path("EnergyPlus IDF or epJSON building model.")
    script = _path("Workspace-relative EnergyPlus analysis script.")
    return {
        "name": "software-bench-energyplus",
        "version": "1.0.0",
        "description": "Building-energy simulation, preprocessing, and time-series analysis.",
        "tools": [
            {
                "name": "energyplus_report_version",
                "description": "Report the installed EnergyPlus version.",
                "input_schema": _schema({}, []),
                "command": ["energyplus", "--version"],
            },
            {
                "name": "energyplus_run_simulation",
                "description": "Simulate a building model against an EPW weather file.",
                "input_schema": _schema(
                    {
                        "model": model,
                        "weather": _path("EPW weather file."),
                        "output": _path("Simulation output directory."),
                        "arguments": _arguments(),
                    },
                    ["model", "weather", "output", "arguments"],
                ),
                "command": [
                    "energyplus", "-w", "{weather}", "-d", "{output}",
                    "{arguments...}", "{model}",
                ],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "energyplus_expand_objects",
                "description": "Expand templates and ground objects, then run design days.",
                "input_schema": _schema(
                    {"model": model, "output": _path("Expansion output directory.")},
                    ["model", "output"],
                ),
                "command": [
                    "energyplus", "--expandobjects", "--design-day-only",
                    "--output-directory", "{output}", "{model}",
                ],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "energyplus_run_design_days",
                "description": "Run sizing-period design days without an annual simulation.",
                "input_schema": _schema(
                    {
                        "model": model,
                        "weather": _path("EPW weather file."),
                        "output": _path("Design-day output directory."),
                    },
                    ["model", "weather", "output"],
                ),
                "command": [
                    "energyplus", "--design-day-only", "-w", "{weather}",
                    "-d", "{output}", "{model}",
                ],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "energyplus_run_annual",
                "description": "Run a forced annual weather simulation for annual metrics.",
                "input_schema": _schema(
                    {
                        "model": model,
                        "weather": _path("EPW weather file."),
                        "output": _path("Annual simulation output directory."),
                    },
                    ["model", "weather", "output"],
                ),
                "command": [
                    "energyplus", "--annual", "-w", "{weather}",
                    "-d", "{output}", "{model}",
                ],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "energyplus_expand_macros",
                "description": "Expand an IMF macro model and execute its design days.",
                "input_schema": _schema(
                    {
                        "input": _path("IMF macro input file."),
                        "output": _path("Macro expansion output directory."),
                    },
                    ["input", "output"],
                ),
                "command": [
                    "energyplus", "--epmacro", "--design-day-only",
                    "--output-directory", "{output}", "{input}",
                ],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "energyplus_extract_csv",
                "description": "Convert ESO simulation output to selected CSV time series.",
                "input_schema": _schema(
                    {"instructions": _path("ReadVarsESO RVI instruction file.")},
                    ["instructions"],
                ),
                "command": ["ReadVarsESO", "{instructions}"],
                "timeout_seconds": 600,
                "mutating": True,
            },
            {
                "name": "energyplus_convert_model",
                "description": "Convert between supported IDF and epJSON model formats.",
                "input_schema": _schema(
                    {"model": model, "output": _path("Conversion output directory.")},
                    ["model", "output"],
                ),
                "command": [
                    "energyplus", "--convert-only", "--output-directory", "{output}",
                    "{model}",
                ],
                "timeout_seconds": 300,
                "mutating": True,
            },
            {
                "name": "energyplus_inspect_model",
                "description": "Summarize zones, schedules, constructions, and HVAC topology.",
                "input_schema": _schema(
                    {"script": script, "model": model}, ["script", "model"]
                ),
                "command": ["python", "{script}", "{model}"],
                "timeout_seconds": 300,
            },
            {
                "name": "energyplus_extract_metrics",
                "description": "Extract energy, comfort, load, and peak-demand metrics.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "results": _path("EnergyPlus result directory or SQL file."),
                        "output": _path("Output metrics JSON or table."),
                        "metrics": _arguments("Metric identifiers to extract."),
                    },
                    ["script", "results", "output", "metrics"],
                ),
                "command": [
                    "python", "{script}", "{results}", "{output}", "{metrics...}",
                ],
                "timeout_seconds": 600,
                "mutating": True,
            },
            {
                "name": "energyplus_compare_runs",
                "description": "Compare performance metrics from two simulation runs.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "reference": _path("Reference results."),
                        "candidate": _path("Candidate results."),
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
