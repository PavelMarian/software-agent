"""Native ParaView checks derived from the originating regression workflows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

from paraview import servermanager
from paraview.simple import GetSources, LoadState, OpenDataFile, UpdatePipeline


class Failure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def png_info(path: Path) -> tuple[int, int, str]:
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 64, f"invalid PNG: {path}")
    width, height = struct.unpack(">II", data[16:24])
    require(width > 0 and height > 0, f"empty PNG geometry: {path}")
    return width, height, hashlib.sha256(data).hexdigest()


def csv_numeric_rows(path: Path) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    require(len(rows) >= 2, f"CSV has no data rows: {path}")
    numeric: list[list[float]] = []
    for row in rows[1:]:
        values = []
        for cell in row:
            try:
                values.append(float(cell))
            except ValueError:
                continue
        if values:
            numeric.append(values)
    require(numeric, f"CSV has no numeric data: {path}")
    return numeric


def proxy_types(group: str) -> set[str]:
    manager = servermanager.ProxyManager()
    return {proxy.GetXMLName() for proxy in manager.GetProxiesInGroup(group).values()}


def load_state(root: Path) -> tuple[set[str], set[str]]:
    state = root / "pipeline.pvsm"
    require(state.is_file(), "pipeline.pvsm is missing")
    LoadState(str(state))
    for proxy in GetSources().values():
        UpdatePipeline(proxy=proxy)
    return proxy_types("sources"), proxy_types("views")


def validate_save_data(root: Path) -> dict:
    contracts = {
        "selected-arrays-selected-blocks.vtm": ({"Normals"}, ["Elevation1"]),
        "selected-arrays-all-blocks.vtm": ({"Normals"}, ["Elevation1", "Elevation2"]),
        "all-arrays-selected-blocks.vtm": ({"Elevation", "Normals"}, ["Elevation1"]),
        "all-arrays-all-blocks.vtm": ({"Elevation", "Normals"}, ["Elevation1", "Elevation2"]),
    }
    evidence = {}
    for name, (wanted_arrays, wanted_blocks) in contracts.items():
        path = root / name
        require(path.is_file(), f"missing upstream SaveData case: {name}")
        reader = OpenDataFile(str(path))
        require(reader is not None, f"ParaView could not open {name}")
        UpdatePipeline(proxy=reader)
        arrays = {reader.PointData.GetArray(i).GetName() for i in range(reader.PointData.GetNumberOfArrays())}
        info = reader.GetDataInformation().DataInformation
        blocks = [info.GetBlockName(i) for i in range(1, info.GetNumberOfDataSets() + 1)]
        require(arrays == wanted_arrays, f"{name}: expected arrays {sorted(wanted_arrays)}, got {sorted(arrays)}")
        require(blocks == wanted_blocks, f"{name}: expected blocks {wanted_blocks}, got {blocks}")
        evidence[name] = {"arrays": sorted(arrays), "blocks": blocks}
    return evidence


def validate_animation(root: Path) -> dict:
    frames = sorted(root.glob("*.png"))
    require(len(frames) >= 6, "upstream animation check requires at least frames 0000..0005")
    infos = [png_info(path) for path in frames]
    require(all((w, h) == (800, 800) for w, h, _ in infos), "animation frames must be 800x800")
    require(len({digest for _, _, digest in infos}) >= 3, "animation frames are effectively static")
    require(any("0005" in path.stem for path in frames), "the upstream checked frame 0005 is missing")
    return {"frames": len(frames), "distinct_frames": len({x[2] for x in infos})}


def validate_color_histogram(root: Path) -> dict:
    sources, views = load_state(root)
    require(bool(sources & {"RTAnalyticSource", "Wavelet"}), f"Wavelet source absent: {sources}")
    require("XYHistogramChartView" in views, f"histogram chart view absent: {views}")
    require("RenderView" in views, f"render view absent: {views}")
    histogram = csv_numeric_rows(root / "histogram.csv")
    selected = csv_numeric_rows(root / "selected.csv")
    require(len(selected) <= len(histogram), "selected table cannot exceed histogram table")
    hi = png_info(root / "histogram.png")
    si = png_info(root / "selection.png")
    require(hi[2] != si[2], "the two configured views produced identical images")
    return {"sources": sorted(sources), "views": sorted(views), "histogram_rows": len(histogram), "selected_rows": len(selected)}


def vti_summary(path: Path) -> tuple[int, tuple[float, float]]:
    reader = OpenDataFile(str(path))
    require(reader is not None, f"ParaView could not open {path}")
    UpdatePipeline(proxy=reader)
    info = reader.GetDataInformation()
    require(info.GetNumberOfPoints() > 0, f"{path} contains no points")
    array = reader.PointData.GetArray("RTData")
    require(array is not None, f"RTData is missing from {path}")
    return info.GetNumberOfPoints(), tuple(array.GetRange())


def validate_wavelet_extracts(root: Path) -> dict:
    sources, views = load_state(root)
    require("Slice" in sources, f"Slice filter absent: {sources}")
    require(bool(sources & {"ExtractHistogram", "Histogram"}), f"histogram branch absent: {sources}")
    require("RenderView" in views and "XYHistogramChartView" in views, f"two-view layout absent: {views}")
    s0 = vti_summary(root / "extracts/data_000.vti")
    s5 = vti_summary(root / "extracts/data_005.vti")
    require(s0 != s5 or (root / "extracts/data_000.vti").read_bytes() != (root / "extracts/data_005.vti").read_bytes(), "temporal data extracts are identical")
    hist = [png_info(root / f"extracts/histogram_{step:03d}.png") for step in (0, 5)]
    render = [png_info(root / f"extracts/render_{step:03d}.png") for step in (0, 2, 4, 5)]
    require(hist[0][:2] != render[0][:2], "histogram and render extractors must use distinct resolutions")
    require(len({x[2] for x in hist}) == 2, "histogram does not change across timesteps")
    require(len({x[2] for x in render}) >= 3, "render sequence does not reflect temporal change")
    return {"sources": sorted(sources), "views": sorted(views), "data": [s0, s5]}


VALIDATORS = {
    "paraview-save-data-selection": validate_save_data,
    "paraview-savedatawitharrayselection": validate_save_data,
    "paraview-saveanimationmultiview": validate_animation,
    "paraview-color-histogram-selection": validate_color_histogram,
    "paraview-wavelet-histogram-extracts": validate_wavelet_extracts,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(VALIDATORS), required=True)
    parser.add_argument("--submission", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        evidence = VALIDATORS[args.task_id](args.submission.resolve())
    except (Failure, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"oracle": "ParaView upstream-derived", "status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({"oracle": "ParaView upstream-derived", "status": "passed", "evidence": evidence}, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
