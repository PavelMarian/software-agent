from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    config = _path("SU2 configuration file.")
    script = _path("Workspace-relative SU2 analysis script.")
    return {
        "name": "software-bench-su2",
        "version": "1.0.0",
        "description": "Primal, adjoint, mesh-deformation, and design-analysis workflows.",
        "tools": [
            {
                "name": "su2_report_version",
                "description": "Report the installed SU2 solver version.",
                "input_schema": _schema({}, []),
                "command": ["SU2_CFD", "--help"],
            },
            {
                "name": "su2_run_cfd",
                "description": "Run a direct compressible or incompressible CFD solution.",
                "input_schema": _schema({"config": config}, ["config"]),
                "command": ["SU2_CFD", "{config}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "su2_deform_mesh",
                "description": "Deform a computational mesh from design-surface changes.",
                "input_schema": _schema({"config": config}, ["config"]),
                "command": ["SU2_DEF", "{config}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "su2_export_solution",
                "description": "Merge partitions and convert solution output formats.",
                "input_schema": _schema({"config": config}, ["config"]),
                "command": ["SU2_SOL", "{config}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "su2_run_adjoint",
                "description": "Run a discrete-adjoint solution for a configured objective.",
                "input_schema": _schema({"config": config}, ["config"]),
                "command": ["SU2_CFD_AD", "{config}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "su2_compute_gradient",
                "description": "Project adjoint sensitivities onto design variables.",
                "input_schema": _schema({"config": config}, ["config"]),
                "command": ["SU2_DOT_AD", "{config}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "su2_analyze_geometry",
                "description": "Evaluate geometric constraints and design-variable effects.",
                "input_schema": _schema({"config": config}, ["config"]),
                "command": ["SU2_GEO", "{config}"],
                "timeout_seconds": 600,
                "mutating": True,
            },
            {
                "name": "su2_validate_config",
                "description": "Validate a configuration and referenced files before a run.",
                "input_schema": _schema(
                    {"script": script, "config": config}, ["script", "config"]
                ),
                "command": ["python", "{script}", "validate", "{config}"],
                "timeout_seconds": 300,
            },
            {
                "name": "su2_parameter_sweep",
                "description": "Execute a scripted parameter sweep over SU2 configurations.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "config": config,
                        "output": _path("Sweep output directory."),
                        "parameters": _arguments("Parameter-name/value sweep declarations."),
                    },
                    ["script", "config", "output", "parameters"],
                ),
                "command": [
                    "python", "{script}", "{config}", "{output}", "{parameters...}",
                ],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "su2_extract_convergence",
                "description": "Extract residual and objective histories from an SU2 run.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "history": _path("SU2 history file."),
                        "output": _path("Output convergence summary."),
                    },
                    ["script", "history", "output"],
                ),
                "command": ["python", "{script}", "{history}", "{output}"],
                "timeout_seconds": 300,
                "mutating": True,
            },
            {
                "name": "su2_compare_solutions",
                "description": "Compare aerodynamic coefficients or field solutions.",
                "input_schema": _schema(
                    {
                        "script": script,
                        "reference": _path("Reference SU2 result."),
                        "candidate": _path("Candidate SU2 result."),
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
