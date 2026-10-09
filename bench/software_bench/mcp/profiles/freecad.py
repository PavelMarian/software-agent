from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    return {
        "name": "software-bench-freecad",
        "version": "1.0.0",
        "description": (
            "Headless parametric CAD construction, recomputation, inspection, and export."
        ),
        "tools": [
            {
                "name": "freecad_run_script",
                "description": "Run a FreeCAD Python script in headless mode.",
                "input_schema": _schema(
                    {"script": _path("FreeCAD Python script."), "arguments": _arguments()},
                    ["script", "arguments"],
                ),
                "command": ["FreeCADCmd", "{script}", "{arguments...}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "freecad_check_document",
                "description": (
                    "Open and recompute a native FreeCAD document to detect model errors."
                ),
                "input_schema": _schema({"document": _path("FCStd document.")}, ["document"]),
                "command": [
                    "FreeCADCmd", "-c",
                    "import FreeCAD as A,sys; d=A.openDocument(sys.argv[-1]); d.recompute(); "
                    "print(len(d.Objects))",
                    "{document}",
                ],
                "timeout_seconds": 600,
            },
            {
                "name": "freecad_report_version",
                "description": "Report the pinned FreeCAD runtime version.",
                "input_schema": _schema({}, []),
                "command": ["FreeCADCmd", "--version"],
            },
            {
                "name": "freecad_export_script",
                "description": (
                    "Run a controlled export script for STEP, STL, DXF, or another format."
                ),
                "input_schema": _schema(
                    {
                        "script": _path("FreeCAD export script."),
                        "document": _path("Input FCStd document."),
                        "output": _path("Output CAD or mesh artifact."),
                    },
                    ["script", "document", "output"],
                ),
                "command": ["FreeCADCmd", "{script}", "{document}", "{output}"],
                "timeout_seconds": 900, "mutating": True,
            },
            {
                "name": "freecad_mesh_script",
                "description": "Run a meshing script and save the resulting mesh artifact.",
                "input_schema": _schema(
                    {
                        "script": _path("FreeCAD meshing script."),
                        "document": _path("Input FCStd document."),
                        "output": _path("Output mesh path."),
                        "arguments": _arguments(),
                    },
                    ["script", "document", "output", "arguments"],
                ),
                "command": [
                    "FreeCADCmd", "{script}", "{document}", "{output}", "{arguments...}",
                ],
                "timeout_seconds": 1800, "mutating": True,
            },
        ],
    }
