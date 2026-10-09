from pathlib import Path

import pytest

from software_bench.core.config import load_mode
from software_bench.harness.models import registry


class FakeEntryPoint:
    def __init__(self, factory):
        self.factory = factory

    def load(self):
        return self.factory


class FakeRuntime:
    runtime_id = "external-graph"
    version = "1.2.3"

    def run(self, request, adapter, observer):
        raise NotImplementedError


def test_agent_runtime_is_loaded_from_a_typed_entry_point(monkeypatch) -> None:
    def fake_entry_points(*, group, name):
        assert group == "software_bench.agent_runtimes"
        assert name == "external-graph"
        return [FakeEntryPoint(FakeRuntime)]

    monkeypatch.setattr(registry, "entry_points", fake_entry_points)

    runtime = registry.load_agent_runtime("external-graph")

    assert runtime.runtime_id == "external-graph"
    assert runtime.version == "1.2.3"


def test_agent_runtime_entry_point_identity_must_match(monkeypatch) -> None:
    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda **kwargs: [FakeEntryPoint(FakeRuntime)],
    )

    with pytest.raises(TypeError, match="mismatched runtime_id"):
        registry.load_agent_runtime("different-runtime")


def test_mode_contract_accepts_a_plugin_runtime_identifier(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    source = (root / "configs/modes/unrestricted_solo.toml").read_text(encoding="utf-8")
    mode_path = tmp_path / "plugin.toml"
    mode_path.write_text(
        source.replace(
            'agent_topology = "single_agent"',
            'agent_topology = "single_agent"\nagent_runtime = "external-graph"',
        ),
        encoding="utf-8",
    )

    mode = load_mode(mode_path, root / "configs/roles")

    assert mode.agent_runtime == "external-graph"
