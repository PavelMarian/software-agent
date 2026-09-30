from __future__ import annotations

import pytest

from foambench.models import FoamBenchError
from foambench.workspace import FoamBenchWorkspace


def test_case_path_stays_inside_output():
    assert FoamBenchWorkspace.case_path("system/controlDict").as_posix() == "output/system/controlDict"
    assert FoamBenchWorkspace.case_path("output/0/U").as_posix() == "output/0/U"
    for unsafe in ("../secret", "/etc/passwd", "output/../secret", "output\\x", "C:secret"):
        with pytest.raises(FoamBenchError):
            FoamBenchWorkspace.case_path(unsafe)


def test_workspace_writes_and_collects_only_case_output(tmp_path):
    workspace = FoamBenchWorkspace(tmp_path)
    workspace.write("system/controlDict", "application icoFoam;\n")
    workspace.write("0/U", "internalField uniform (0 0 0);\n")

    assert workspace.read("output/system/controlDict") == "application icoFoam;\n"
    assert workspace.collect_submission() == {
        "0/U": "internalField uniform (0 0 0);\n",
        "system/controlDict": "application icoFoam;\n",
    }


def test_adapter_exposes_neutral_tool_names(tmp_path):
    names = FoamBenchWorkspace(tmp_path).toolset().names

    assert names == (
        "read_file", "list_files", "write_file", "run_program", "verify_workspace",
    )
    assert all("foam" not in name for name in names)
