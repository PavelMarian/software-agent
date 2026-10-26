from __future__ import annotations

import json
import sys
from pathlib import Path


def _status(value: bool) -> str:
    return "PASSED" if value else "FAILED"


def main() -> int:
    base = Path(".benchmark")
    sys.path.insert(0, str(base.resolve()))
    from freecad_cam import check_freecad_cam_boolean_detailed

    config = json.loads((base / "spec" / "case.json").read_text(encoding="utf-8"))
    options = config["evaluator"]["options"].copy()
    options["reference_path"] = str((base / "reference" / "reference.FCStd").resolve())
    options["script_path"] = str((base / "freecad_cam_boolean.py").resolve())
    try:
        result = check_freecad_cam_boolean_detailed(
            str(Path("output/Unnamed.FCStd").resolve()), **options
        )
    except Exception as error:  # evaluator failures become failed checks, not parser errors
        print(f"FreeCAD evaluator error: {type(error).__name__}: {error}", file=sys.stderr)
        result = {"passed": False, "checks": {}}

    checks = result.get("checks", {}) if isinstance(result, dict) else {}
    structure = all(
        bool(checks.get(name))
        for name in ("reference_stock_present", "result_body_present", "result_stock_present")
    )
    preserved = bool(checks.get("body_unchanged")) and bool(checks.get("stock_unchanged"))
    toolpaths = bool(checks.get("cam_operation_nc_present")) and bool(checks.get("actual_cut_exists"))
    geometry = all(
        bool(checks.get(name))
        for name in ("expected_cut_positive", "undercut_within_limit", "overcut_within_limit")
    )
    print(json.dumps({
        "artifact_structure": _status(structure),
        "geometry_preservation": _status(preserved),
        "cam_toolpaths": _status(toolpaths),
        "material_removal_accuracy": _status(geometry),
    }, sort_keys=True))
    if isinstance(result, dict):
        print(json.dumps(result, default=str, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
