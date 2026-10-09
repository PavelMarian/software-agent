from __future__ import annotations

import shutil
import tempfile
from importlib.metadata import entry_points
from pathlib import Path

from software_bench.core.models import TaskBundle, ValidationError
from software_bench.harness.environments.common import _bundle_asset_path
from software_bench.harness.environments.contracts import EnvironmentSession
from software_bench.harness.environments.docker import DockerEnvironmentSession
from software_bench.harness.environments.local import LocalEnvironmentSession
from software_bench.harness.environments.managed import ManagedProcessEnvironmentSession

def create_environment_session(
    backend: str,
    bundle: TaskBundle,
    *,
    workspace: Path | None,
    run_id: str,
) -> EnvironmentSession:
    if backend == "auto":
        backend = bundle.environment.backend_hint
    if backend in {"local", "local-fixture"}:
        if workspace is None:
            if (
                bundle.environment.workspace_source != "bundle"
                or bundle.environment.seed_path is None
            ):
                raise ValidationError("local git environment backend requires --workspace")
            temporary = Path(tempfile.mkdtemp(prefix="software-bench-workspace-"))
            source = _bundle_asset_path(bundle.root, bundle.environment.seed_path)
            shutil.copytree(source, temporary, dirs_exist_ok=True)
            return LocalEnvironmentSession(
                temporary,
                bundle.environment,
                remove_on_close=True,
            )
        return LocalEnvironmentSession(workspace, bundle.environment)
    if backend == "docker":
        return DockerEnvironmentSession(
            bundle.environment,
            base_commit=bundle.task.base_commit or "",
            bundle_root=bundle.root,
            run_id=run_id,
        )
    if backend == "managed_process":
        return ManagedProcessEnvironmentSession(bundle, workspace=workspace)
    matches = entry_points(group="software_bench.environment_backends", name=backend)
    if not matches:
        raise ValidationError(
            f"unsupported environment backend: {backend}; install a "
            "'software_bench.environment_backends' entry point"
        )
    factory = matches[0].load()
    return factory(
        bundle=bundle,
        workspace=workspace,
        run_id=run_id,
    )


