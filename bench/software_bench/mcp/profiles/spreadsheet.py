from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    path = _path("Workspace-relative workbook path.")
    return {
        "name": "software-bench-spreadsheet",
        "version": "1.0.0",
        "description": "Inspect, transform, and recalculate spreadsheet workbooks.",
        "tools": [
            {
                "name": "spreadsheet_inspect_package",
                "description": "List the internal files in an XLSX or ODS package.",
                "input_schema": _schema({"path": path}, ["path"]),
                "command": ["python", "-m", "zipfile", "-l", "{path}"],
            },
            {
                "name": "spreadsheet_run_solution",
                "description": "Run a benchmark solution script on one input workbook.",
                "input_schema": _schema(
                    {
                        "script": _path("Workspace-relative Python solution script."),
                        "input": path,
                        "output": _path("Workspace-relative output workbook path."),
                    },
                    ["script", "input", "output"],
                ),
                "command": ["python", "{script}", "{input}", "{output}"],
                "timeout_seconds": 600, "mutating": True,
            },
            {
                "name": "spreadsheet_recalculate",
                "description": "Open and save a workbook headlessly so formulas are recalculated.",
                "input_schema": _schema(
                    {"path": path, "output_dir": _path("Workspace-relative output directory.")},
                    ["path", "output_dir"],
                ),
                "command": [
                    "libreoffice", "--headless", "--convert-to", "xlsx", "--outdir",
                    "{output_dir}", "{path}",
                ],
                "timeout_seconds": 600, "mutating": True,
            },
            {
                "name": "spreadsheet_export_pdf",
                "description": "Export a workbook to PDF through headless LibreOffice.",
                "input_schema": _schema(
                    {
                        "path": path,
                        "output_dir": _path("Workspace-relative PDF output directory."),
                    },
                    ["path", "output_dir"],
                ),
                "command": [
                    "libreoffice", "--headless", "--convert-to", "pdf", "--outdir",
                    "{output_dir}", "{path}",
                ],
                "timeout_seconds": 600, "mutating": True,
            },
            {
                "name": "spreadsheet_export_csv",
                "description": "Export the active sheet to CSV through headless LibreOffice.",
                "input_schema": _schema(
                    {
                        "path": path,
                        "output_dir": _path("Workspace-relative CSV output directory."),
                    },
                    ["path", "output_dir"],
                ),
                "command": [
                    "libreoffice", "--headless", "--convert-to", "csv", "--outdir",
                    "{output_dir}", "{path}",
                ],
                "timeout_seconds": 600, "mutating": True,
            },
        ],
    }
