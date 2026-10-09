from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from software_bench.core.models import TaskBundle, ValidationError
from software_bench.harness.environments.common import (
    _bundle_asset_path,
    _format_config,
    _free_tcp_port,
    _wait_for_url,
)
from software_bench.harness.environments.local import LocalEnvironmentSession

class ManagedProcessEnvironmentSession(LocalEnvironmentSession):
    """Bundle-seeded local workspace plus a declaratively managed control plane."""

    backend_id = "managed_process"

    def __init__(self, bundle: TaskBundle, *, workspace: Path | None) -> None:
        remove_on_close = workspace is None
        if workspace is None:
            if bundle.environment.seed_path is None:
                raise ValidationError("managed process environment requires seed_path")
            workspace = Path(tempfile.mkdtemp(prefix="software-bench-managed-"))
            source = _bundle_asset_path(bundle.root, bundle.environment.seed_path)
            shutil.copytree(source, workspace, dirs_exist_ok=True)
        raw_process = bundle.environment.backend_config.get("managed_process")
        if not isinstance(raw_process, Mapping):
            if remove_on_close:
                shutil.rmtree(workspace)
            raise ValidationError("backend_config.managed_process must be an object")
        port = _free_tcp_port() if raw_process.get("allocate_port") is True else None
        variables = {
            "bundle_root": str(bundle.root),
            "workspace": str(workspace),
            **({"port": str(port)} if port is not None else {}),
        }
        runtime_config = _format_config(bundle.environment.backend_config, variables)
        runtime_environment = replace(bundle.environment, backend_config=runtime_config)
        super().__init__(workspace, runtime_environment, remove_on_close=remove_on_close)
        self._process: subprocess.Popen[str] | None = None
        self._service_log = None
        self._shutdown_url: str | None = None
        config = runtime_config.get("managed_process")
        if not isinstance(config, Mapping):
            self.close()
            raise ValidationError("backend_config.managed_process must be an object")
        command = config.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            self.close()
            raise ValidationError("managed_process.command must be a non-empty string array")
        values = variables
        rendered = [item.format_map(values) for item in command]
        cwd_value = config.get("cwd", "{bundle_root}")
        if not isinstance(cwd_value, str):
            self.close()
            raise ValidationError("managed_process.cwd must be a string")
        cwd = Path(cwd_value.format_map(values)).resolve()
        raw_env = config.get("environment", {})
        if not isinstance(raw_env, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_env.items()
        ):
            self.close()
            raise ValidationError("managed_process.environment must map strings to strings")
        process_env = dict(os.environ)
        process_env.update(
            {key: value.format_map(values) for key, value in raw_env.items()}
        )
        readiness_url = config.get("readiness_url")
        if not isinstance(readiness_url, str):
            self.close()
            raise ValidationError("managed_process.readiness_url must be a string")
        shutdown_url = config.get("shutdown_url")
        if shutdown_url is not None and not isinstance(shutdown_url, str):
            self.close()
            raise ValidationError("managed_process.shutdown_url must be a string")
        self._shutdown_url = shutdown_url
        try:
            log_path = self.workspace / ".environment" / "service.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._service_log = log_path.open("w", encoding="utf-8")
            self._process = subprocess.Popen(
                rendered,
                cwd=cwd,
                stdout=self._service_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=process_env,
            )
            _wait_for_url(
                readiness_url,
                process=self._process,
                timeout_seconds=float(config.get("startup_timeout_seconds", 600)),
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                shutdown_url = getattr(self, "_shutdown_url", None)
                if shutdown_url:
                    request = urllib.request.Request(shutdown_url, data=b"{}", method="POST")
                    with urllib.request.urlopen(request, timeout=300):
                        pass
                else:
                    process.terminate()
                process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired, urllib.error.URLError):
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        log = getattr(self, "_service_log", None)
        if log is not None:
            log.close()
            self._service_log = None
        super().close()


