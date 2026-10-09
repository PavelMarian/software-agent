import shutil
from pathlib import Path
from typing import Any, Mapping

from software_bench.core.config import load_mode
from software_bench.core.task_bundle import load_task_bundle
from software_bench.harness.contracts import ModelResponse, RunRequest
from software_bench.harness.environments import LocalEnvironmentSession
from software_bench.harness.execution.runner import BenchmarkRunner


ROOT = Path(__file__).parents[2]
BUNDLE = load_task_bundle(ROOT / "tests" / "fixtures" / "task_bundle")
ROLES = ROOT / "configs" / "roles"


class RecordingAdapter:
    adapter_id = "recording"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.planner_handoff_sent = False

    def generate(
        self,
        messages: list[Mapping[str, Any]],
        system_prompt: str,
        tools: list[Mapping[str, Any]],
        *,
        role_id: str,
        seed: int,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "system_prompt": system_prompt,
                "tools": tools,
                "role_id": role_id,
                "seed": seed,
            }
        )
        tool_calls: tuple[Mapping[str, Any], ...] = ()
        if (
            role_id == "planner"
            and not self.planner_handoff_sent
            and any(tool["name"] == "message" for tool in tools)
        ):
            self.planner_handoff_sent = True
            tool_calls = (
                {
                    "name": "message",
                    "args": {"to": "executor", "content": "implement the planned change"},
                },
            )
        if any(tool["name"] == "finish_phase" for tool in tools):
            tool_calls = (
                *tool_calls,
                {
                    "name": "finish_phase",
                    "args": {"status": "complete", "summary": "done"},
                },
            )
        return ModelResponse(
            text="done",
            tool_calls=tool_calls,
            done=True,
            input_tokens=2,
            output_tokens=1,
        )


class RepairAdapter:
    adapter_id = "repair"

    def __init__(self) -> None:
        self.roles: list[str] = []
        self.verifications = 0

    def generate(self, *args: Any, role_id: str, **kwargs: Any) -> ModelResponse:
        self.roles.append(role_id)
        if role_id == "verifier":
            self.verifications += 1
            status = "needs_revision" if self.verifications == 1 else "complete"
        else:
            status = "complete"
        return ModelResponse(
            text="",
            tool_calls=(
                {
                    "name": "finish_phase",
                    "args": {"status": status, "summary": status},
                },
            ),
            done=False,
            input_tokens=1,
            output_tokens=1,
        )


def _run(mode_name: str, tmp_path: Path) -> tuple[RecordingAdapter, Any]:
    workspace = tmp_path / "workspace"
    shutil.copytree(BUNDLE.root / "workspace", workspace)
    mode = load_mode(ROOT / "configs" / "modes" / f"{mode_name}.toml", ROLES)
    request = RunRequest(
        f"{mode_name}-test",
        BUNDLE.agent_view,
        mode,
        LocalEnvironmentSession(workspace),
        7,
    )
    adapter = RecordingAdapter()
    record = BenchmarkRunner().run(request, adapter, tmp_path / "run")
    return adapter, record


def test_single_agent_route_runs_one_full_task_agent(tmp_path: Path) -> None:
    adapter, record = _run("compute_matched_solo", tmp_path)

    assert record.manifest["agent_topology"] == "single_agent"
    assert record.manifest["agent_run_count"] == 1
    assert [call["role_id"] for call in adapter.calls] == ["solo"]
    assert "single task-solving software agent" in adapter.calls[0]["system_prompt"]
    assert all("parameters" in tool for tool in adapter.calls[0]["tools"])
    assert sum(event.event_type == "agent_started" for event in record.trace) == 1


def test_mas_route_runs_each_role_through_the_common_agent_loop(tmp_path: Path) -> None:
    adapter, record = _run("full_mas", tmp_path)

    assert record.manifest["agent_topology"] == "multi_agent"
    assert record.manifest["agent_run_count"] == 3
    assert [call["role_id"] for call in adapter.calls] == [
        "planner", "executor", "verifier"
    ]
    assert "Identify dependencies" in adapter.calls[0]["system_prompt"]
    assert "Message from teammate planner" in adapter.calls[1]["messages"][-1]["content"]
    assert sum(
        event.event_type == "software_multiagent.node_started" for event in record.trace
    ) == 3


def test_mas_verifier_routes_one_bounded_repair(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    shutil.copytree(BUNDLE.root / "workspace", workspace)
    mode = load_mode(ROOT / "configs" / "modes" / "full_mas.toml", ROLES)
    adapter = RepairAdapter()

    record = BenchmarkRunner().run(
        RunRequest(
            "repair-test",
            BUNDLE.agent_view,
            mode,
            LocalEnvironmentSession(workspace),
            7,
        ),
        adapter,
        tmp_path / "run",
    )

    assert record.manifest["status"] == "completed"
    assert record.manifest["agent_run_count"] == 5
    assert adapter.roles == [
        "planner",
        "executor",
        "verifier",
        "executor",
        "verifier",
    ]
