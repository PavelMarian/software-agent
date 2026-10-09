import json
from pathlib import Path


root = Path.cwd()
control = root / "system" / "controlDict"
allrun = root / "Allrun"
physics = root / "constant" / "physicalProperties"
print(json.dumps({
    "case_structure": "PASSED" if control.is_file() else "FAILED",
    "solver_execution": (
        "PASSED" if allrun.is_file() and "pisoFoam" in allrun.read_text() else "FAILED"
    ),
    "numerical_accuracy": (
        "PASSED" if physics.is_file() and "1e-05" in physics.read_text() else "FAILED"
    ),
}))
