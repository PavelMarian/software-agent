from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from software_bench.core.models import (
    AgentTopology,
    BenchmarkMode,
    Budget,
    RoleSpec,
    ValidationError,
    WorkspaceAccess,
)


@dataclass(frozen=True)
class ModeSpec:
    id: BenchmarkMode
    agent_topology: AgentTopology
    budget: Budget
    budget_policy: str
    fair_comparison_group: str | None
    roles: tuple[RoleSpec, ...]
    phases: tuple[str, ...]
    communication_enabled: bool
    independent_candidates: bool
    supported: bool
    max_turns_per_phase: int
    agent_runtime: str = "legacy"


def load_mode(path: str | Path, roles_dir: str | Path) -> ModeSpec:
    source = Path(path)
    with source.open("rb") as handle:
        data = tomllib.load(handle)
    try:
        mode_id = BenchmarkMode(data["id"])
    except (KeyError, ValueError, TypeError) as error:
        raise ValidationError(f"invalid benchmark mode in {source}") from error

    budget_data = _object(data, "budget")
    budget = Budget(
        token_limit=_positive_int(budget_data, "token_limit"),
        wall_time_seconds=_positive_number(budget_data, "wall_time_seconds"),
        tool_call_limit=_positive_int(budget_data, "tool_call_limit"),
    )
    role_entries = data.get("roles")
    if not isinstance(role_entries, list) or not role_entries:
        raise ValidationError("mode.roles must be a non-empty array of tables")
    roles = tuple(_load_assignment(entry, Path(roles_dir)) for entry in role_entries)
    role_ids = [role.id for role in roles]
    if len(role_ids) != len(set(role_ids)):
        raise ValidationError("role assignment ids must be unique")
    phases = data.get("phases", role_ids)
    if not isinstance(phases, list) or not phases or not all(
        isinstance(item, str) and item for item in phases
    ):
        raise ValidationError("phases must be a non-empty string array")
    unknown_phases = set(phases) - set(role_ids)
    if unknown_phases:
        raise ValidationError(f"phases reference unknown roles: {sorted(unknown_phases)}")

    policy = data.get("budget_policy", "team_total")
    if policy not in {"team_total", "run_ceiling"}:
        raise ValidationError("budget_policy must be team_total or run_ceiling")
    fair_group = data.get("fair_comparison_group")
    if fair_group is not None and not isinstance(fair_group, str):
        raise ValidationError("fair_comparison_group must be a string or omitted")
    try:
        topology = AgentTopology(data["agent_topology"])
    except (KeyError, ValueError, TypeError) as error:
        raise ValidationError(f"invalid agent_topology in {source}") from error
    communication_enabled = _optional_boolean(data, "communication_enabled", True)
    independent_candidates = _optional_boolean(data, "independent_candidates", False)
    _validate_topology(
        topology,
        role_ids,
        phases,
        communication_enabled,
        independent_candidates,
    )
    runtime = data.get("agent_runtime", "legacy")
    if not isinstance(runtime, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.-]{0,63}", runtime):
        raise ValidationError("agent_runtime must be a valid runtime identifier")
    return ModeSpec(
        id=mode_id,
        agent_topology=topology,
        budget=budget,
        budget_policy=policy,
        fair_comparison_group=fair_group,
        roles=roles,
        phases=tuple(phases),
        communication_enabled=communication_enabled,
        independent_candidates=independent_candidates,
        supported=_optional_boolean(data, "supported", True),
        max_turns_per_phase=_positive_int(data, "max_turns_per_phase")
        if "max_turns_per_phase" in data
        else 20,
        agent_runtime=runtime,
    )


def _load_assignment(value: Any, roles_dir: Path) -> RoleSpec:
    if not isinstance(value, Mapping):
        raise ValidationError("every role assignment must be an object")
    role_id = value.get("id")
    profile = value.get("profile")
    if not isinstance(role_id, str) or not role_id:
        raise ValidationError("role assignment id must be a non-empty string")
    if not isinstance(profile, str) or not profile:
        raise ValidationError("role profile must be a non-empty string")
    profile_path = roles_dir / f"{profile}.toml"
    try:
        with profile_path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as error:
        raise ValidationError(f"role profile not found: {profile_path}") from error
    if data.get("name") != profile:
        raise ValidationError(f"role profile name does not match filename: {profile_path}")
    instructions = data.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValidationError(f"role profile requires non-empty instructions: {profile_path}")
    permissions = _object(data, "permissions")
    tools = permissions.get("tools")
    if not isinstance(tools, list) or not all(isinstance(item, str) and item for item in tools):
        raise ValidationError("permissions.tools must be a string array")
    try:
        workspace_access = WorkspaceAccess(permissions.get("workspace"))
    except (TypeError, ValueError) as error:
        raise ValidationError("permissions.workspace has an unsupported value") from error
    return RoleSpec(
        id=role_id,
        profile=profile,
        instructions=instructions,
        workspace_access=workspace_access,
        tools=tuple(tools),
        can_communicate=_boolean(permissions, "communicate"),
    )


def _validate_topology(
    topology: AgentTopology,
    role_ids: list[str],
    phases: list[str],
    communication_enabled: bool,
    independent_candidates: bool,
) -> None:
    if topology == AgentTopology.SINGLE_AGENT:
        if len(role_ids) != 1 or phases != role_ids:
            raise ValidationError("single_agent topology requires exactly one role and one phase")
        if communication_enabled:
            raise ValidationError("single_agent topology cannot enable inter-agent communication")
        if independent_candidates:
            raise ValidationError("single_agent topology cannot use independent candidates")
    elif topology == AgentTopology.MULTI_AGENT:
        if len(role_ids) < 2:
            raise ValidationError("multi_agent topology requires at least two roles")
        if independent_candidates:
            raise ValidationError("multi_agent topology cannot use independent candidates")
    elif not independent_candidates:
        raise ValidationError("independent_ensemble topology requires independent candidates")


def _object(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValidationError(f"{key} must be an object")
    return item


def _positive_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise ValidationError(f"{key} must be a positive integer")
    return item


def _positive_number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool) or item <= 0:
        raise ValidationError(f"{key} must be positive")
    return float(item)


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ValidationError(f"{key} must be boolean")
    return item


def _optional_boolean(value: Mapping[str, Any], key: str, default: bool) -> bool:
    item = value.get(key, default)
    if not isinstance(item, bool):
        raise ValidationError(f"{key} must be boolean")
    return item
