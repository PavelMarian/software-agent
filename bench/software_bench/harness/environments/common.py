from __future__ import annotations

import difflib
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from software_bench.core.artifacts import EncodedArtifact, encode_artifact
from software_bench.core.models import ValidationError

def resolve_host_command(command: Sequence[str]) -> list[str]:
    """Return a host-portable invocation without changing the bundle contract."""
    invocation = list(command)
    if not invocation:
        raise ValidationError("command must not be empty")
    if invocation[0] in {"python", "python3"} and (
        os.name == "nt" or shutil.which(invocation[0]) is None
    ):
        invocation[0] = sys.executable
    return invocation


def _docker_user_config(
    backend_config: Mapping[str, Any],
) -> tuple[str, int, int, str] | None:
    raw = backend_config.get("docker_user")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValidationError("backend_config.docker_user must be an object")
    unknown = set(raw) - {"name", "uid", "gid", "home"}
    if unknown:
        raise ValidationError(
            f"unknown backend_config.docker_user fields: {sorted(unknown)}"
        )
    name = raw.get("name")
    uid = raw.get("uid")
    gid = raw.get("gid")
    home = raw.get("home")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z_][a-z0-9_-]*", name):
        raise ValidationError("backend_config.docker_user.name is invalid")
    if (
        not isinstance(uid, int) or isinstance(uid, bool) or uid <= 0
        or not isinstance(gid, int) or isinstance(gid, bool) or gid <= 0
    ):
        raise ValidationError("backend_config.docker_user uid and gid must be positive integers")
    if not isinstance(home, str):
        raise ValidationError("backend_config.docker_user.home must be an absolute path")
    home_path = PurePosixPath(home)
    if not home_path.is_absolute() or ".." in home_path.parts:
        raise ValidationError("backend_config.docker_user.home must be an absolute path")
    return name, uid, gid, str(home_path)


def _invoke(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Docker CLI is not installed or not available on PATH") from error
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {detail}")
    return completed


def _container_name(run_id: str, label: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "_.-" else "-"
        for character in run_id
    )
    return f"software-bench-{safe[:40]}-{label}-{uuid.uuid4().hex[:8]}"


def _docker_platform(platform: str) -> str:
    return platform.replace("x86_64", "amd64").replace("aarch64", "arm64")


def _snapshot(workspace: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            result[path.relative_to(workspace).as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return result


def _diff(before: Mapping[str, str], after: Mapping[str, str]) -> str:
    chunks: list[str] = []
    for name in sorted(set(before) | set(after)):
        old = before.get(name, "").splitlines(keepends=True)
        new = after.get(name, "").splitlines(keepends=True)
        if old != new:
            chunks.extend(difflib.unified_diff(old, new, fromfile=f"a/{name}", tofile=f"b/{name}"))
    return "".join(chunks)


def _tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-4000:]


def _bundle_asset_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise PermissionError(f"bundle asset escapes bundle: {relative}") from error
    if not candidate.is_dir():
        raise ValidationError(f"bundle asset directory does not exist: {candidate}")
    return candidate


def _collect_local_files(root: Path, paths: Sequence[str]) -> Mapping[str, EncodedArtifact]:
    result: dict[str, EncodedArtifact] = {}
    for relative in paths:
        target = (root / relative).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as error:
            raise PermissionError(f"submission path escapes workspace: {relative}") from error
        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = sorted(target.rglob("*"))
        else:
            candidates = []
        for candidate in candidates:
            if not candidate.is_file():
                continue
            result[candidate.relative_to(root).as_posix()] = encode_artifact(
                candidate.read_bytes()
            )
    return result


def _wait_for_url(
    url: str,
    *,
    process: subprocess.Popen[str],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"managed process exited during startup ({process.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"managed process did not become ready: {url}")


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _format_config(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, Mapping):
        return {key: _format_config(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_config(item, variables) for item in value]
    return value
