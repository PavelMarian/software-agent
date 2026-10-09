from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    return {
        "name": "software-bench-paraview",
        "version": "1.0.0",
        "description": "Batch scientific visualization and quantitative extraction with ParaView.",
        "tools": [
            {
                "name": "paraview_run_script",
                "description": "Execute a pvpython analysis or visualization script.",
                "input_schema": _schema(
                    {"script": _path("pvpython script."), "arguments": _arguments()},
                    ["script", "arguments"],
                ),
                "command": ["pvpython", "{script}", "{arguments...}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "paraview_run_batch",
                "description": "Execute a ParaView batch script, including MPI-capable workflows.",
                "input_schema": _schema(
                    {"script": _path("pvbatch script."), "arguments": _arguments()},
                    ["script", "arguments"],
                ),
                "command": ["pvbatch", "{script}", "{arguments...}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "paraview_report_version",
                "description": "Report the pinned ParaView runtime version.",
                "input_schema": _schema({}, []),
                "command": ["pvpython", "--version"],
            },
            {
                "name": "paraview_render_script",
                "description": "Execute a script whose declared output is a rendered image.",
                "input_schema": _schema(
                    {
                        "script": _path("ParaView rendering script."),
                        "input": _path("Scientific input dataset."),
                        "output": _path("Rendered image path."),
                        "arguments": _arguments(),
                    },
                    ["script", "input", "output", "arguments"],
                ),
                "command": ["pvpython", "{script}", "{input}", "{output}", "{arguments...}"],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "paraview_extract_script",
                "description": (
                    "Execute a quantitative extraction pipeline and save tabular output."
                ),
                "input_schema": _schema(
                    {
                        "script": _path("ParaView extraction script."),
                        "input": _path("Scientific input dataset."),
                        "output": _path("Output CSV or dataset path."),
                        "arguments": _arguments(),
                    },
                    ["script", "input", "output", "arguments"],
                ),
                "command": ["pvpython", "{script}", "{input}", "{output}", "{arguments...}"],
                "timeout_seconds": 1800, "mutating": True,
            },
        ],
    }
