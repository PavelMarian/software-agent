import io
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from software_bench.core.models import ValidationError
from software_bench.core.task_bundle import load_task_bundle
from software_bench.harness.environments import ExecutionResult, LocalEnvironmentSession
from software_bench.mcp import (
    ApplicationServer,
    JsonRpcServer,
    McpEnvironmentSession,
    automatic_spec,
    builtin_profiles,
)
from software_bench.mcp.server import serve_stdio
from software_bench.mcp.registry import load_spec, write_spec
from software_bench.mcp.__main__ import main as mcp_main


class FakeEnvironment:
    backend_id = "fake"

    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def run(self, command, *, timeout_seconds):
        self.calls.append((list(command), timeout_seconds))
        return ExecutionResult("ok", "", 0)

    def close(self):
        self.closed = True


def test_manual_mcp_spec_validates_without_agent_package(tmp_path: Path) -> None:
    spec_path = tmp_path / "application.mcp.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "manual-example",
                "version": "1.0.0",
                "description": "Hand-written application tools.",
                "tools": [
                    {
                        "name": "manual_probe",
                        "description": "Run the application probe.",
                        "input_schema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        "command": ["python", "--version"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "validation.json"

    result = mcp_main(
        [
            "validate-spec",
            "--spec", str(spec_path),
            "--workspace", str(tmp_path),
            "--output", str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["valid"] is True
    assert report["validation_level"] == "preflight"


def test_automatic_openfoam_profile_uses_public_software_signal() -> None:
    root = Path(__file__).parents[2]
    bundle = load_task_bundle(root / "tests/fixtures/software_use_bundle")
    openfoam = replace(
        bundle,
        task=replace(bundle.task, target_software="OpenFOAM 12"),
    )
    assert automatic_spec(openfoam).name == "software-bench-openfoam"


def test_explicit_mcp_profile_overrides_legacy_inference() -> None:
    root = Path(__file__).parents[2]
    bundle = load_task_bundle(root / "tests/fixtures/software_use_bundle")
    explicit = replace(
        bundle,
        task=replace(bundle.task, metadata={"mcp_profile": "sqlite"}),
    )

    assert automatic_spec(explicit).name == "software-bench-sqlite"


def test_unknown_explicit_mcp_profile_fails_closed() -> None:
    root = Path(__file__).parents[2]
    bundle = load_task_bundle(root / "tests/fixtures/software_use_bundle")
    explicit = replace(
        bundle,
        task=replace(bundle.task, metadata={"mcp_profile": "missing"}),
    )

    with pytest.raises(ValidationError, match="unknown MCP profile"):
        automatic_spec(explicit)


def test_automatic_profiles_cover_spreadsheets_and_local_spider2() -> None:
    root = Path(__file__).parents[2]
    bundle = load_task_bundle(root / "tests/fixtures/software_use_bundle")
    spreadsheet = replace(
        bundle,
        task=replace(bundle.task, target_software="LibreOffice Calc"),
    )
    assert automatic_spec(spreadsheet).name == "software-bench-spreadsheet"

    spider = replace(
        bundle,
        task=replace(
            bundle.task,
            target_software="SQL and the Spider 2.0 data environment",
            metadata={"variant": "lite", "source_id": "local001"},
        ),
    )
    assert automatic_spec(spider).name == "software-bench-sqlite"


def test_scientific_profiles_are_compact_and_available() -> None:
    profiles = builtin_profiles()
    expected = {
        "blender", "energyplus", "freecad", "gromacs", "kubernetes", "lammps",
        "modflow", "openmc", "openfoam", "paraview", "qgis",
        "quantum_espresso", "spreadsheet", "sqlite", "su2",
    }
    assert set(profiles) == expected
    counts = [len(spec.tools) for spec in profiles.values()]
    assert all(3 <= count <= 14 for count in counts)
    assert sum(counts) == 125


def test_automatic_selection_covers_scientific_and_operational_software() -> None:
    root = Path(__file__).parents[2]
    bundle = load_task_bundle(root / "tests/fixtures/software_use_bundle")
    expected = {
        "QGIS 3.40": "software-bench-qgis",
        "ParaView and VTK": "software-bench-paraview",
        "FreeCAD": "software-bench-freecad",
        "GROMACS": "software-bench-gromacs",
        "LAMMPS": "software-bench-lammps",
        "Quantum ESPRESSO": "software-bench-quantum-espresso",
        "OpenMC": "software-bench-openmc",
        "SU2": "software-bench-su2",
        "EnergyPlus": "software-bench-energyplus",
        "MODFLOW 6 with FloPy": "software-bench-modflow",
        "Kubernetes": "software-bench-kubernetes",
        "Blender": "software-bench-blender",
    }
    for target, profile_name in expected.items():
        candidate = replace(bundle, task=replace(bundle.task, target_software=target))
        assert automatic_spec(candidate).name == profile_name


def test_manual_spec_round_trip(tmp_path: Path) -> None:
    expected = builtin_profiles()["spreadsheet"]
    path = tmp_path / "spreadsheet.mcp.json"
    write_spec(expected, path)
    assert load_spec(path) == expected


def test_mutating_tools_are_hidden_without_explicit_permission() -> None:
    environment = FakeEnvironment()
    server = ApplicationServer(builtin_profiles()["sqlite"], environment)
    names = {item["name"] for item in server.declarations()}
    assert "sqlite_query" in names
    assert "sqlite_execute" not in names
    with pytest.raises(PermissionError, match="mutation permission"):
        server.call("sqlite_execute", {"database": "db.sqlite", "sql": "DELETE FROM t"})


def test_arguments_are_validated_and_rendered_as_argv() -> None:
    environment = FakeEnvironment()
    server = ApplicationServer(builtin_profiles()["sqlite"], environment)
    server.call("sqlite_query", {"database": "data/db.sqlite", "sql": "SELECT * FROM t"})
    command, timeout = environment.calls[0]
    assert command[-2:] == ["data/db.sqlite", "SELECT * FROM t"]
    assert timeout == 60
    with pytest.raises(ValidationError, match="read-only SQL"):
        server.call("sqlite_query", {"database": "data/db.sqlite", "sql": "DROP TABLE t"})
    with pytest.raises(ValidationError, match="escapes"):
        server.call("sqlite_query", {"database": "../db.sqlite", "sql": "SELECT 1"})


def test_argument_arrays_expand_without_a_shell() -> None:
    environment = FakeEnvironment()
    server = ApplicationServer(
        builtin_profiles()["qgis"], environment, allow_mutations=True
    )
    server.call(
        "qgis_run_algorithm",
        {
            "algorithm": "native:buffer",
            "parameters": ["INPUT=data.gpkg", "DISTANCE=10", "OUTPUT=output.gpkg"],
        },
    )
    assert environment.calls[0][0] == [
        "qgis_process", "run", "native:buffer", "--",
        "INPUT=data.gpkg", "DISTANCE=10", "OUTPUT=output.gpkg",
    ]
    with pytest.raises(ValidationError, match="only strings"):
        server.call(
            "qgis_run_algorithm",
            {"algorithm": "native:buffer", "parameters": ["DISTANCE=10", 11]},
        )


@pytest.mark.parametrize(
    ("profile", "tool", "arguments", "expected"),
    [
        (
            "quantum_espresso",
            "qe_compute_dos",
            {"input": "qe/dos.in"},
            ["dos.x", "-in", "qe/dos.in"],
        ),
        (
            "openmc",
            "openmc_extract_tallies",
            {
                "script": "tools/tallies.py",
                "statepoint": "run/statepoint.100.h5",
                "output": "results/tallies.csv",
                "arguments": ["heating", "flux"],
            },
            [
                "python", "tools/tallies.py", "run/statepoint.100.h5",
                "results/tallies.csv", "heating", "flux",
            ],
        ),
        (
            "su2",
            "su2_run_adjoint",
            {"config": "case/adjoint.cfg"},
            ["SU2_CFD_AD", "case/adjoint.cfg"],
        ),
        (
            "energyplus",
            "energyplus_run_simulation",
            {
                "model": "building/model.idf",
                "weather": "weather/site.epw",
                "output": "results/run-1",
                "arguments": ["--annual"],
            },
            [
                "energyplus", "-w", "weather/site.epw", "-d", "results/run-1",
                "--annual", "building/model.idf",
            ],
        ),
        (
            "modflow",
            "modflow_extract_heads",
            {
                "script": "tools/heads.py",
                "heads": "run/model.hds",
                "output": "results/heads.csv",
                "arguments": ["--layer", "2"],
            },
            [
                "python", "tools/heads.py", "run/model.hds", "results/heads.csv",
                "--layer", "2",
            ],
        ),
    ],
)
def test_new_scientific_profiles_render_distinct_workflows(
    profile: str,
    tool: str,
    arguments: dict,
    expected: list[str],
) -> None:
    environment = FakeEnvironment()
    server = ApplicationServer(
        builtin_profiles()[profile], environment, allow_mutations=True
    )
    server.call(tool, arguments)
    assert environment.calls[0][0] == expected


def test_json_rpc_lists_and_calls_tools() -> None:
    environment = FakeEnvironment()
    server = JsonRpcServer(ApplicationServer(builtin_profiles()["openfoam"], environment))
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert listed["result"]["tools"][0]["inputSchema"]["type"] == "object"
    called = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "openfoam_check_mesh", "arguments": {"case": "."}},
        }
    )
    assert called["result"]["isError"] is False
    assert environment.calls == [(["checkMesh", "-case", "."], 300.0)]


def test_environment_bridge_exposes_the_same_tools_to_benchmark_roles() -> None:
    environment = FakeEnvironment()
    environment.tool_declarations = lambda: {}
    environment.invoke_tool = lambda name, arguments: ExecutionResult("existing", "", 0)
    bridge = McpEnvironmentSession(environment, builtin_profiles()["openfoam"])
    declarations = bridge.tool_declarations()
    assert "openfoam_check_mesh" in declarations
    assert "openfoam_build_mesh" not in declarations
    result = bridge.invoke_tool("openfoam_check_mesh", {"case": "."})
    assert result == ExecutionResult("ok", "", 0)


def test_stdio_supports_newline_json_rpc() -> None:
    environment = FakeEnvironment()
    source = io.BytesIO(
        (json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n").encode()
    )
    sink = io.BytesIO()
    serve_stdio(
        JsonRpcServer(ApplicationServer(builtin_profiles()["spreadsheet"], environment)),
        input_stream=source,
        output_stream=sink,
    )
    response = json.loads(sink.getvalue())
    assert response["result"] == {}
    assert environment.closed is True


def test_sqlite_profile_executes_in_a_real_workspace(tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE products (name TEXT, price INTEGER)")
    connection.execute("INSERT INTO products VALUES ('pump', 12)")
    connection.commit()
    connection.close()

    server = ApplicationServer(
        builtin_profiles()["sqlite"], LocalEnvironmentSession(tmp_path)
    )
    result = server.call(
        "sqlite_query",
        {"database": "catalog.sqlite", "sql": "SELECT name, price FROM products"},
    )
    payload = json.loads(result["structuredContent"]["stdout"])
    assert payload == {"columns": ["name", "price"], "rows": [["pump", 12]]}

    attempted_write = server.call(
        "sqlite_query",
        {
            "database": "catalog.sqlite",
            "sql": "WITH selected AS (SELECT 1) DELETE FROM products",
        },
    )
    assert attempted_write["isError"] is True
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT count(*) FROM products").fetchone() == (1,)
    connection.close()

    missing = server.call("sqlite_list_tables", {"database": "missing.sqlite"})
    assert missing["isError"] is True
    assert not (tmp_path / "missing.sqlite").exists()
