from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    return {
        "name": "software-bench-blender",
        "version": "1.0.0",
        "description": "Headless geometry, simulation, and rendering through Blender scripts.",
        "tools": [
            {
                "name": "blender_run_script",
                "description": "Run a Blender Python workflow in background mode.",
                "input_schema": _schema(
                    {"script": _path("Blender Python script."), "arguments": _arguments()},
                    ["script", "arguments"],
                ),
                "command": [
                    "blender", "--background", "--python", "{script}", "--",
                    "{arguments...}",
                ],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "blender_render_frame",
                "description": "Render one frame from a native Blender scene.",
                "input_schema": _schema(
                    {
                        "scene": _path("BLEND scene."), "output": _path("Output image path."),
                        "frame": {"type": "integer"},
                    },
                    ["scene", "output", "frame"],
                ),
                "command": [
                    "blender", "{scene}", "--background", "--render-output", "{output}",
                    "--render-frame", "{frame}",
                ],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "blender_report_version",
                "description": "Report Blender build information.",
                "input_schema": _schema({}, []), "command": ["blender", "--version"],
            },
            {
                "name": "blender_run_scene_script",
                "description": (
                    "Open a BLEND scene and run a modifying or analytical Python script."
                ),
                "input_schema": _schema(
                    {
                        "scene": _path("BLEND scene."),
                        "script": _path("Blender Python script."),
                        "arguments": _arguments(),
                    },
                    ["scene", "script", "arguments"],
                ),
                "command": [
                    "blender", "{scene}", "--background", "--python", "{script}", "--",
                    "{arguments...}",
                ],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "blender_render_animation",
                "description": "Render the configured animation range from a BLEND scene.",
                "input_schema": _schema(
                    {"scene": _path("BLEND scene."), "output": _path("Output pattern.")},
                    ["scene", "output"],
                ),
                "command": [
                    "blender", "{scene}", "--background", "--render-output", "{output}",
                    "--render-anim",
                ],
                "timeout_seconds": 1800, "mutating": True,
            },
            {
                "name": "blender_inspect_scene",
                "description": "Report object names, types, frame range, and active render engine.",
                "input_schema": _schema({"scene": _path("BLEND scene.")}, ["scene"]),
                "command": [
                    "blender", "{scene}", "--background", "--python-expr",
                    "import bpy,json; s=bpy.context.scene; print(json.dumps({"
                    "'objects':[(o.name,o.type) for o in s.objects],"
                    "'frames':[s.frame_start,s.frame_end],'engine':s.render.engine}))",
                ],
                "timeout_seconds": 300,
            },
        ],
    }
