"""Small dependency-free loader for local experiment configuration."""

from __future__ import annotations

import os
from pathlib import Path


def load_environment(path: str | Path = ".env", *, override: bool = False) -> Path | None:
    """Load KEY=VALUE entries from an ignored local file.

    Existing process variables win unless ``override`` is explicitly requested.
    The parser intentionally supports only the simple format used by this project.
    """

    environment_file = Path(path)
    if not environment_file.is_file():
        return None
    for raw_line in environment_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        if name and (override or name not in os.environ):
            os.environ[name] = value
    return environment_file.resolve()


def target_model() -> tuple[str | None, str | None]:
    """Return the provider and target model selected for agent runs."""

    return (
        os.getenv("SOFTWARE_MULTIAGENT_PROVIDER") or None,
        os.getenv("SOFTWARE_MULTIAGENT_MODEL") or None,
    )
