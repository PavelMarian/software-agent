from pathlib import Path

import pytest

from software_bench.core.config import load_mode
from software_bench.core.models import AgentTopology, ValidationError
from software_bench.harness.models.registry import ModelAdapterSettings, load_model_adapter


ROOT = Path(__file__).parents[2]
MODES = ROOT / "configs" / "modes"
ROLES = ROOT / "configs" / "roles"


def test_every_mode_configuration_loads() -> None:
    loaded = [load_mode(path, ROLES) for path in sorted(MODES.glob("*.toml"))]

    assert len(loaded) == 6
    assert len({mode.id for mode in loaded}) == 6
    assert sum(mode.supported for mode in loaded) == 5
    assert len({(mode.id, mode.agent_runtime) for mode in loaded}) == 6


def test_compute_matched_modes_share_one_aggregate_budget() -> None:
    loaded = [load_mode(path, ROLES) for path in sorted(MODES.glob("*.toml"))]
    matched = [mode for mode in loaded if mode.fair_comparison_group == "compute_matched_v1"]

    assert len(matched) == 5
    assert len({mode.budget for mode in matched}) == 1
    assert all(mode.budget_policy == "team_total" for mode in matched)


def test_modes_declare_single_and_multi_agent_topologies() -> None:
    solo = load_mode(MODES / "compute_matched_solo.toml", ROLES)
    mas = load_mode(MODES / "full_mas.toml", ROLES)

    assert solo.agent_topology == AgentTopology.SINGLE_AGENT
    assert [role.id for role in solo.roles] == ["solo"]
    assert mas.agent_topology == AgentTopology.MULTI_AGENT
    assert [role.id for role in mas.roles] == ["planner", "executor", "verifier"]
    assert mas.agent_runtime == "legacy"
    assert all(role.instructions for role in (*solo.roles, *mas.roles))


def test_single_agent_topology_rejects_multiple_roles(tmp_path: Path) -> None:
    source = (MODES / "full_mas.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(
        source.replace('agent_topology = "multi_agent"', 'agent_topology = "single_agent"'),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="exactly one role"):
        load_mode(path, ROLES)


def test_openrouter_adapter_requires_an_explicit_model() -> None:
    with pytest.raises(ValueError, match="--model is required"):
        load_model_adapter("openrouter", ModelAdapterSettings())

