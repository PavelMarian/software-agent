from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    qe_input = _path("Quantum ESPRESSO namelist input file.")
    return {
        "name": "software-bench-quantum-espresso",
        "version": "1.0.0",
        "description": (
            "Electronic-structure, phonon, spectroscopy, and reaction-path workflows."
        ),
        "tools": [
            {
                "name": "qe_report_version",
                "description": "Report the installed Quantum ESPRESSO pw.x build.",
                "input_schema": _schema({}, []),
                "command": ["pw.x", "-help"],
            },
            {
                "name": "qe_run_pw",
                "description": "Run an SCF, NSCF, relaxation, or variable-cell pw.x job.",
                "input_schema": _schema(
                    {"input": qe_input, "arguments": _arguments()},
                    ["input", "arguments"],
                ),
                "command": ["pw.x", "-in", "{input}", "{arguments...}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "qe_run_phonon",
                "description": "Calculate phonons and dynamical matrices with ph.x.",
                "input_schema": _schema({"input": qe_input}, ["input"]),
                "command": ["ph.x", "-in", "{input}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "qe_postprocess_density",
                "description": "Post-process charge density or wavefunctions with pp.x.",
                "input_schema": _schema({"input": qe_input}, ["input"]),
                "command": ["pp.x", "-in", "{input}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "qe_compute_dos",
                "description": "Compute an electronic density of states with dos.x.",
                "input_schema": _schema({"input": qe_input}, ["input"]),
                "command": ["dos.x", "-in", "{input}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "qe_compute_bands",
                "description": "Extract and reorder electronic bands with bands.x.",
                "input_schema": _schema({"input": qe_input}, ["input"]),
                "command": ["bands.x", "-in", "{input}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "qe_project_wavefunctions",
                "description": "Calculate projected DOS and atomic projections with projwfc.x.",
                "input_schema": _schema({"input": qe_input}, ["input"]),
                "command": ["projwfc.x", "-in", "{input}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "qe_run_neb",
                "description": "Run a nudged elastic band reaction-path calculation.",
                "input_schema": _schema({"input": qe_input}, ["input"]),
                "command": ["neb.x", "-in", "{input}"],
                "timeout_seconds": 1800,
                "mutating": True,
            },
            {
                "name": "qe_convert_force_constants",
                "description": "Convert dynamical matrices to real-space force constants.",
                "input_schema": _schema({"input": qe_input}, ["input"]),
                "command": ["q2r.x", "-in", "{input}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "qe_calculate_dispersion",
                "description": "Calculate phonon dispersions or vibrational DOS with matdyn.x.",
                "input_schema": _schema({"input": qe_input}, ["input"]),
                "command": ["matdyn.x", "-in", "{input}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
            {
                "name": "qe_analyze_modes",
                "description": "Analyze zone-center modes and infrared activities with dynmat.x.",
                "input_schema": _schema({"input": qe_input}, ["input"]),
                "command": ["dynmat.x", "-in", "{input}"],
                "timeout_seconds": 900,
                "mutating": True,
            },
        ],
    }
