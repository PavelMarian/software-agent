from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent


def load_catalog() -> Mapping[str, Any]:
    value = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
    catalog = dict(value["profiles"])
    variants = value.get("variants", {})
    overlap = set(catalog) & set(variants)
    if overlap:
        raise ValueError(f"environment names overlap: {sorted(overlap)}")
    catalog.update(variants)
    return catalog


def build_command(profile: str, value: Mapping[str, Any]) -> list[str]:
    dockerfile = ROOT / str(value.get("dockerfile", "Dockerfile"))
    return [
        "docker", "build", "--pull",
        "--tag", str(value["image"]),
        "--file", str(dockerfile),
        "--build-arg", f"CONDA_PACKAGES={' '.join(value['conda_packages'])}",
        "--build-arg", f"APT_PACKAGES={' '.join(value['apt_packages'])}",
        "--build-arg", f"PIP_PACKAGES={' '.join(value.get('pip_packages', []))}",
        "--build-arg", f"EXECUTABLES={';'.join(value['executables'])}",
        "--label", f"org.software-bench.profile={profile}",
        str(ROOT),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    catalog = load_catalog()
    selected = args.profile or sorted(catalog)
    unknown = set(selected) - set(catalog)
    if unknown:
        parser.error(f"unknown profiles: {sorted(unknown)}")
    for profile in selected:
        command = build_command(profile, catalog[profile])
        print(json.dumps(command))
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
