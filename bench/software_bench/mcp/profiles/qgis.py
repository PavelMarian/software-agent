from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    return {
        "name": "software-bench-qgis",
        "version": "1.0.0",
        "description": "Compact geospatial processing profile for QGIS, GDAL, and OGR.",
        "tools": [
            {
                "name": "qgis_list_algorithms",
                "description": "List installed QGIS Processing algorithms.",
                "input_schema": _schema({}, []),
                "command": ["qgis_process", "list"],
            },
            {
                "name": "qgis_describe_algorithm",
                "description": "Show parameters and outputs for one Processing algorithm.",
                "input_schema": _schema(
                    {"algorithm": {"type": "string", "minLength": 1}}, ["algorithm"]
                ),
                "command": ["qgis_process", "help", "{algorithm}"],
            },
            {
                "name": "qgis_run_algorithm",
                "description": "Run one QGIS Processing algorithm with explicit argv parameters.",
                "input_schema": _schema(
                    {
                        "algorithm": {"type": "string", "minLength": 1},
                        "parameters": _arguments("PARAMETER=value arguments for qgis_process."),
                    },
                    ["algorithm", "parameters"],
                ),
                "command": ["qgis_process", "run", "{algorithm}", "--", "{parameters...}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "gdal_inspect_raster",
                "description": "Inspect raster metadata and statistics as JSON.",
                "input_schema": _schema({"path": _path("Raster dataset path.")}, ["path"]),
                "command": ["gdalinfo", "-json", "-stats", "{path}"],
                "timeout_seconds": 300,
            },
            {
                "name": "ogr_inspect_vector",
                "description": "Inspect vector layers, schemas, extents, and feature summaries.",
                "input_schema": _schema({"path": _path("Vector dataset path.")}, ["path"]),
                "command": ["ogrinfo", "-ro", "-so", "-al", "{path}"],
                "timeout_seconds": 300,
            },
            {
                "name": "gdal_translate_raster",
                "description": "Convert, subset, rescale, or re-encode a raster dataset.",
                "input_schema": _schema(
                    {
                        "input": _path("Input raster."), "output": _path("Output raster."),
                        "arguments": _arguments("Additional gdal_translate options."),
                    },
                    ["input", "output", "arguments"],
                ),
                "command": ["gdal_translate", "{arguments...}", "{input}", "{output}"],
                "timeout_seconds": 900, "mutating": True,
            },
            {
                "name": "gdal_warp_raster",
                "description": "Reproject, resample, or mosaic raster datasets.",
                "input_schema": _schema(
                    {
                        "input": _path("Input raster."), "output": _path("Output raster."),
                        "arguments": _arguments("Additional gdalwarp options."),
                    },
                    ["input", "output", "arguments"],
                ),
                "command": ["gdalwarp", "{arguments...}", "{input}", "{output}"],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "ogr_convert_vector",
                "description": "Convert, filter, reproject, or append vector data.",
                "input_schema": _schema(
                    {
                        "input": _path("Input vector dataset."),
                        "output": _path("Output vector dataset."),
                        "arguments": _arguments("Additional ogr2ogr options."),
                    },
                    ["input", "output", "arguments"],
                ),
                "command": ["ogr2ogr", "{arguments...}", "{output}", "{input}"],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "gdal_rasterize_vector",
                "description": "Burn vector features or attributes into a raster dataset.",
                "input_schema": _schema(
                    {
                        "input": _path("Input vector dataset."),
                        "output": _path("Output raster dataset."),
                        "arguments": _arguments("Rasterization options."),
                    },
                    ["input", "output", "arguments"],
                ),
                "command": ["gdal_rasterize", "{arguments...}", "{input}", "{output}"],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "gdal_build_vrt",
                "description": "Build a virtual raster mosaic from selected raster inputs.",
                "input_schema": _schema(
                    {
                        "output": _path("Output VRT path."),
                        "inputs": _arguments("Workspace-relative input raster paths."),
                    },
                    ["output", "inputs"],
                ),
                "command": ["gdalbuildvrt", "{output}", "{inputs...}"],
                "timeout_seconds": 900, "mutating": True,
            },
        ],
    }
