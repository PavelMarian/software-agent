from __future__ import annotations

import math
import os
import shlex
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from software_bench.core.artifacts import EncodedArtifact, encode_artifact
from software_bench.core.models import EnvironmentSpec, ValidationError
from software_bench.harness.environments.common import (
    _bundle_asset_path,
    _collect_local_files,
    _container_name,
    _docker_platform,
    _docker_user_config,
    _invoke,
    _tail,
)
from software_bench.harness.environments.contracts import ExecutionResult

class DockerEnvironmentSession:
    """Mutable Linux task container backed by a pre-built task image."""

    backend_id = "docker"

    def __init__(
        self,
        environment: EnvironmentSpec,
        *,
        base_commit: str,
        bundle_root: Path | None = None,
        run_id: str,
        label: str = "agent",
    ) -> None:
        if not environment.image:
            raise ValidationError("Docker backend requires environment.image")
        self.environment = environment
        self._docker_user = _docker_user_config(environment.backend_config)
        self.container_name = _container_name(run_id, label)
        self._closed = False
        from software_bench.harness.environments.http import ConfiguredHttpTools

        self._http_tools = ConfiguredHttpTools(
            environment.backend_config,
            write_text=self.write_text,
        )
        command = [
            "docker", "create", "--name", self.container_name,
            "--label", "software-bench.managed=true",
            "--label", f"software-bench.run-id={run_id}",
            "--workdir", environment.workdir,
            "--platform", _docker_platform(environment.platform),
        ]
        if not environment.network_enabled:
            command.extend(["--network", "none"])
        for name in environment.pass_env:
            if name in os.environ:
                command.extend(["--env", name])
        command.extend(
            [
                "--entrypoint", "/bin/sh", environment.image, "-c",
                "while :; do sleep 3600; done",
            ]
        )
        try:
            _invoke(command, timeout_seconds=environment.timeout_seconds)
            _invoke(["docker", "start", self.container_name], timeout_seconds=60)
            self._prepare_docker_user()
            self._initial_untracked: tuple[str, ...] = ()
            if environment.workspace_source == "git":
                if not base_commit:
                    raise ValidationError("git workspace requires task.base_commit")
                reset = self.run(
                    ["git", "reset", "--hard", base_commit],
                    timeout_seconds=min(environment.timeout_seconds, 300),
                )
                if reset.exit_code != 0:
                    raise RuntimeError(
                        f"cannot reset task repository: {reset.stderr or reset.stdout}"
                    )
                status = self.run(
                    ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
                    timeout_seconds=120,
                )
                self._initial_untracked = tuple(
                    entry[3:]
                    for entry in status.stdout.split("\0")
                    if entry.startswith("?? ") and len(entry) > 3
                )
            else:
                if bundle_root is None or environment.seed_path is None:
                    raise ValidationError("bundle workspace requires bundle root and seed_path")
                source = _bundle_asset_path(bundle_root, environment.seed_path)
                self.run(["mkdir", "-p", environment.workdir], timeout_seconds=30)
                _invoke(
                    ["docker", "cp", f"{source}/.",
                     f"{self.container_name}:{environment.workdir}"],
                    timeout_seconds=min(environment.timeout_seconds, 300),
                )
                self._chown_workspace(environment.workdir)
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def read_text(self, path: str) -> str:
        target = self._workspace_path(path)
        result = _invoke(
            [*self._docker_exec_prefix(), "cat", target],
            timeout_seconds=60,
        )
        return result.stdout

    def write_text(self, path: str, content: str) -> None:
        target = self._workspace_path(path)
        parent = str(PurePosixPath(target).parent)
        script = f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(target)}"
        _invoke(
            [*self._docker_exec_prefix(interactive=True), "/bin/sh", "-c", script],
            input_text=content,
            timeout_seconds=60,
        )

    def read_bytes(self, path: str) -> bytes:
        target = self._workspace_path(path)
        with tempfile.TemporaryDirectory(prefix="software-bench-read-") as temp:
            local = Path(temp) / "artifact"
            _invoke(
                ["docker", "cp", f"{self.container_name}:{target}", str(local)],
                timeout_seconds=60,
            )
            return local.read_bytes()

    def write_bytes(self, path: str, content: bytes) -> None:
        target = self._workspace_path(path)
        parent = str(PurePosixPath(target).parent)
        created = self.run(["mkdir", "-p", parent], timeout_seconds=30)
        if created.exit_code != 0:
            raise RuntimeError(f"cannot create artifact parent: {created.stderr}")
        with tempfile.TemporaryDirectory(prefix="software-bench-write-") as temp:
            local = Path(temp) / "artifact"
            local.write_bytes(content)
            _invoke(
                ["docker", "cp", str(local), f"{self.container_name}:{target}"],
                timeout_seconds=60,
            )
            self._chown_workspace(target)

    def run(self, command: Sequence[str], *, timeout_seconds: float) -> ExecutionResult:
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("command must contain non-empty strings")
        try:
            limit = max(1, math.ceil(timeout_seconds))
            invocation = list(command)
            if self.environment.shell_init:
                invocation = [
                    "/bin/bash",
                    "-lc",
                    f'{self.environment.shell_init}; exec "$@"',
                    "--",
                    *command,
                ]
            completed = _invoke(
                [
                    *self._docker_exec_prefix(),
                    "timeout", "--signal=TERM", "--kill-after=5s", f"{limit}s",
                    *invocation,
                ],
                timeout_seconds=timeout_seconds + 10,
                check=False,
            )
            return ExecutionResult(completed.stdout, completed.stderr, completed.returncode)
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(_tail(error.stdout), _tail(error.stderr), 124)

    def list_files(self, path: str = ".") -> tuple[str, ...]:
        target = self._workspace_path(path)
        result = self.run(["find", target, "-type", "f", "-print"], timeout_seconds=60)
        if result.exit_code != 0:
            return ()
        prefix = self.environment.workdir.rstrip("/") + "/"
        return tuple(
            absolute[len(prefix):]
            for absolute in result.stdout.splitlines()
            if absolute.startswith(prefix)
        )

    def extract_patch(self) -> str:
        if self.environment.workspace_source != "git":
            raise ValidationError("patch extraction requires a git workspace")
        staged = self.run(["git", "add", "-A"], timeout_seconds=120)
        if staged.exit_code != 0:
            raise RuntimeError(f"cannot stage Docker workspace changes: {staged.stderr}")
        if self._initial_untracked:
            self.run(
                ["git", "reset", "--", *self._initial_untracked],
                timeout_seconds=120,
            )
        result = self.run(
            ["git", "-c", "core.fileMode=false", "diff", "--cached", "--binary"],
            timeout_seconds=120,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"cannot extract Docker workspace patch: {result.stderr}")
        return result.stdout

    def extract_files(self, paths: Sequence[str]) -> Mapping[str, EncodedArtifact]:
        files: dict[str, EncodedArtifact] = {}
        for relative in paths:
            target = self._workspace_path(relative)
            found = self.run(["find", target, "-type", "f", "-print"], timeout_seconds=60)
            if found.exit_code != 0:
                continue
            for absolute in found.stdout.splitlines():
                prefix = self.environment.workdir.rstrip("/") + "/"
                if not absolute.startswith(prefix):
                    continue
                name = absolute[len(prefix):]
                files[name] = encode_artifact(self.read_bytes(name))
        return files

    def extract_text_files(self, paths: Sequence[str]) -> Mapping[str, EncodedArtifact]:
        return self.extract_files(paths)

    def copy_from_host(self, source: Path, destination: str) -> None:
        target = self._workspace_path(destination)
        parent = str(PurePosixPath(target).parent)
        created = self.run(["mkdir", "-p", parent], timeout_seconds=30)
        if created.exit_code != 0:
            raise RuntimeError(f"cannot create asset parent: {created.stderr}")
        suffix = "/." if source.is_dir() else ""
        if source.is_dir():
            self.run(["mkdir", "-p", target], timeout_seconds=30)
        _invoke(
            ["docker", "cp", f"{source}{suffix}", f"{self.container_name}:{target}"],
            timeout_seconds=min(self.environment.timeout_seconds, 300),
        )
        self._chown_workspace(target)

    def tool_declarations(self) -> Mapping[str, Mapping[str, Any]]:
        return self._http_tools.declarations()

    def invoke_tool(self, name: str, arguments: Mapping[str, Any]) -> ExecutionResult:
        return self._http_tools.invoke(name, arguments)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            _invoke(
                ["docker", "rm", "--force", self.container_name],
                timeout_seconds=60,
                check=False,
            )
        except RuntimeError:
            pass

    def _workspace_path(self, relative: str) -> str:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise PermissionError(f"path escapes workspace: {relative}")
        return str(PurePosixPath(self.environment.workdir) / path)

    def _docker_exec_prefix(self, *, interactive: bool = False) -> list[str]:
        command = ["docker", "exec"]
        if interactive:
            command.append("-i")
        if self._docker_user is not None:
            name, _uid, _gid, home = self._docker_user
            command.extend([
                "--user", name,
                "--env", f"HOME={home}",
                "--env", f"USER={name}",
                "--env", f"LOGNAME={name}",
            ])
        command.extend(["--workdir", self.environment.workdir, self.container_name])
        return command

    def _prepare_docker_user(self) -> None:
        if self._docker_user is None:
            return
        name, uid, gid, home = self._docker_user
        script = (
            f"getent group {gid} >/dev/null || groupadd --gid {gid} {shlex.quote(name)}; "
            f"id -u {shlex.quote(name)} >/dev/null 2>&1 || "
            f"useradd --create-home --uid {uid} --gid {gid} --shell /bin/bash {shlex.quote(name)}; "
            f"mkdir -p {shlex.quote(home)} {shlex.quote(self.environment.workdir)}; "
            f"chown -R {uid}:{gid} {shlex.quote(home)} {shlex.quote(self.environment.workdir)}"
        )
        _invoke(
            ["docker", "exec", "--user", "0:0", self.container_name, "/bin/sh", "-c", script],
            timeout_seconds=60,
        )

    def _chown_workspace(self, target: str) -> None:
        if self._docker_user is None:
            return
        _name, uid, gid, _home = self._docker_user
        _invoke(
            [
                "docker", "exec", "--user", "0:0", self.container_name,
                "chown", "-R", f"{uid}:{gid}", target,
            ],
            timeout_seconds=60,
        )


