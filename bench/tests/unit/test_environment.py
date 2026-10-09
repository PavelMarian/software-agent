import subprocess
from pathlib import Path

from software_bench.core.models import EnvironmentSpec
from software_bench.harness.environments import docker as environment_module
from software_bench.harness.environments import DockerEnvironmentSession, LocalEnvironmentSession


def test_local_environment_preserves_workspace_and_extracts_patch(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("before\n", encoding="utf-8")
    session = LocalEnvironmentSession(tmp_path)

    assert session.read_text("README.md") == "before\n"
    session.write_text("README.md", "after\n")
    result = session.run(["python", "-c", "print('ok')"], timeout_seconds=10)

    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"
    assert "-before" in session.extract_patch()
    assert "+after" in session.extract_patch()


def test_local_environment_reports_missing_host_command(tmp_path: Path) -> None:
    session = LocalEnvironmentSession(tmp_path)

    result = session.run(
        ["software-bench-command-that-does-not-exist"],
        timeout_seconds=10,
    )

    assert result.exit_code == 127
    assert "cannot run" in result.stderr


def test_docker_environment_uses_managed_isolated_container(monkeypatch) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def fake_invoke(command, *, timeout_seconds, input_text=None, check=True):
        del timeout_seconds, check
        command = list(command)
        calls.append((command, input_text))
        stdout = ""
        if "cat" in command:
            stdout = "container text"
        if "diff" in command and "--binary" in command:
            stdout = "diff --git a/x b/x\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(environment_module, "_invoke", fake_invoke)
    spec = EnvironmentSpec(
        backend_hint="docker",
        image="example/task:latest",
        workdir="/testbed",
        platform="linux/x86_64",
        timeout_seconds=60,
        network_enabled=False,
    )

    session = DockerEnvironmentSession(spec, base_commit="abc123", run_id="test/run")
    assert session.read_text("README.md") == "container text"
    session.write_text("nested/file.txt", "content")
    assert session.extract_patch().startswith("diff --git")
    session.close()

    create = calls[0][0]
    assert create[:2] == ["docker", "create"]
    assert ["--network", "none"] == create[create.index("--network"):create.index("--network") + 2]
    assert "linux/amd64" in create
    assert any(input_text == "content" for _, input_text in calls)
    assert calls[-1][0][1:3] == ["rm", "--force"]


def test_bundle_docker_environment_copies_only_public_seed(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_invoke(command, *, timeout_seconds, input_text=None, check=True):
        del timeout_seconds, input_text, check
        command = list(command)
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(environment_module, "_invoke", fake_invoke)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "public.txt").write_text("public", encoding="utf-8")
    (tmp_path / "evaluator").mkdir()
    (tmp_path / "evaluator" / "secret.txt").write_text("secret", encoding="utf-8")
    spec = EnvironmentSpec(
        image="example/foam:latest",
        workdir="/workspace",
        workspace_source="bundle",
        seed_path="workspace",
    )

    session = DockerEnvironmentSession(
        spec,
        base_commit="",
        bundle_root=tmp_path,
        run_id="bundle-test",
    )
    session.close()

    copy_calls = [call for call in calls if call[:2] == ["docker", "cp"]]
    assert len(copy_calls) == 1
    assert "workspace" in copy_calls[0][2]
    assert "evaluator" not in copy_calls[0][2]


def test_bundle_docker_environment_runs_as_configured_non_root_user(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_invoke(command, *, timeout_seconds, input_text=None, check=True):
        del timeout_seconds, input_text, check
        command = list(command)
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(environment_module, "_invoke", fake_invoke)
    (tmp_path / "workspace").mkdir()
    spec = EnvironmentSpec(
        image="example/foam:latest",
        workdir="/workspace",
        workspace_source="bundle",
        seed_path="workspace",
        backend_config={
            "docker_user": {
                "name": "foam", "uid": 1000, "gid": 1000, "home": "/home/foam"
            }
        },
    )

    session = DockerEnvironmentSession(
        spec, base_commit="", bundle_root=tmp_path, run_id="non-root-test"
    )
    session.run(["id", "-u"], timeout_seconds=10)
    session.write_text("owned.txt", "content")
    session.close()

    agent_execs = [
        call for call in calls
        if call[:2] == ["docker", "exec"] and "--user" in call
        and call[call.index("--user") + 1] == "foam"
    ]
    assert agent_execs
    assert all("HOME=/home/foam" in call for call in agent_execs)
    assert any(
        call[:4] == ["docker", "exec", "--user", "0:0"] and "useradd" in " ".join(call)
        for call in calls
    )
    assert any(
        call[:4] == ["docker", "exec", "--user", "0:0"] and "chown" in call
        for call in calls
    )
