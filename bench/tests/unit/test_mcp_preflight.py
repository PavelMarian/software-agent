from software_bench.harness.environments import ExecutionResult
from software_bench.mcp import builtin_profiles, preflight_profile


class FakeEnvironment:
    backend_id = "fake-scientific-image"

    def __init__(self, available):
        self.available = set(available)
        self.calls = []

    def run(self, command, *, timeout_seconds):
        self.calls.append((list(command), timeout_seconds))
        executable = command[-1]
        if executable in self.available:
            return ExecutionResult(f"/usr/bin/{executable}\n", "", 0)
        return ExecutionResult("", "missing", 1)


def test_preflight_reports_all_distinct_profile_executables() -> None:
    spec = builtin_profiles()["modflow"]
    environment = FakeEnvironment({"mf6", "python", "zbud6"})

    result = preflight_profile(spec, environment)

    assert result["ready"] is True
    assert result["tool_count"] == 11
    assert result["executable_count"] == 3
    assert {item["executable"] for item in result["checks"]} == {
        "mf6", "python", "zbud6",
    }


def test_preflight_fails_closed_when_an_executable_is_missing() -> None:
    result = preflight_profile(
        builtin_profiles()["modflow"], FakeEnvironment({"python", "zbud6"})
    )

    assert result["ready"] is False
    missing = [item for item in result["checks"] if not item["available"]]
    assert [item["executable"] for item in missing] == ["mf6"]


def test_preflight_reports_dynamic_executable_without_resolving_template() -> None:
    spec = builtin_profiles()["openfoam"]
    available = {
        tool.command[0]
        for tool in spec.tools
        if "{" not in tool.command[0] and "}" not in tool.command[0]
    }
    environment = FakeEnvironment(available)

    result = preflight_profile(spec, environment)

    assert result["ready"] is True
    assert result["dynamic_executables"] == ["{solver}"]
    assert all(call[0][-1] != "{solver}" for call in environment.calls)
