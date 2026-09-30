"""Safe, local operations for an agent-owned OpenFOAM case workspace."""

from __future__ import annotations

import subprocess
import shlex
from pathlib import Path, PurePosixPath
from typing import Sequence

from foambench.models import FoamBenchError
from software_multiagent.tools.workspace import WorkspaceFiles, WorkspaceToolset


PROGRAMS = frozenset({
    "blockMesh", "checkMesh", "snappyHexMesh", "surfaceFeatureExtract", "setFields",
    "simpleFoam", "pimpleFoam", "pisoFoam", "icoFoam", "interFoam", "rhoSimpleFoam",
    "rhoPimpleFoam", "rhoCentralFoam", "buoyantSimpleFoam", "buoyantPimpleFoam",
    "laplacianFoam", "potentialFoam", "sonicFoam", "scalarTransportFoam",
})


class FoamBenchWorkspace:
    """Public workspace boundary; all agent file access remains below output/."""

    def __init__(self, root: str | Path, *, image: str | None = None) -> None:
        self.root = Path(root).resolve()
        self.image = image
        self.files = WorkspaceFiles(self.root, writable_root="output")
        self.solver_evidence: dict[str, str] | None = None

    @staticmethod
    def case_path(value: str) -> PurePosixPath:
        if not isinstance(value, str):
            raise FoamBenchError("case path must be a string")
        if value in {"", "."}:
            value = "output"
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in value or ":" in value:
            raise FoamBenchError("use a relative path inside output/")
        return path if path.parts[0] == "output" else PurePosixPath("output") / path

    def _resolve(self, value: str) -> Path:
        relative = self.case_path(value)
        target = (self.root / Path(*relative.parts)).resolve()
        output = (self.root / "output").resolve()
        if target != output and output not in target.parents:
            raise FoamBenchError("case path escapes output/")
        return target

    def read(self, path: str, *, limit: int = 16_000) -> str:
        return self.files.read_file(self.case_path(path).relative_to("output").as_posix(), limit=limit)["content"]

    def list(self, path: str = "output") -> tuple[str, ...]:
        relative = self.case_path(path).relative_to("output").as_posix()
        return tuple("output/" + item for item in self.files.list_files(relative)["files"])

    def write(self, path: str, content: str) -> Path:
        relative = self.case_path(path).relative_to("output").as_posix()
        if not relative or relative == ".":
            raise FoamBenchError("write a file below output/")
        target = self.root / "output" / self.files.write_file(relative, content)["path"]
        self.solver_evidence = None
        return target

    def run(self, program: str, *, timeout_seconds: float = 300.0, extra_args: Sequence[str] = ()) -> dict[str, object]:
        if program not in PROGRAMS:
            raise FoamBenchError(f"unsupported OpenFOAM program: {program}")
        if any(not isinstance(arg, str) for arg in extra_args):
            raise FoamBenchError("program arguments must be strings")
        try:
            command = [program, "-case", "output", *extra_args]
            invocation = command if self.image is None else ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "8g", "-v", f"{self.root}:/workspace", "-w", "/workspace", self.image, "bash", "-lc", "source /opt/openfoam10/etc/bashrc && " + shlex.join(command)]
            completed = subprocess.run(
                invocation, cwd=self.root, capture_output=True,
                text=True, timeout=timeout_seconds, check=False,
            )
        except FileNotFoundError:
            return {"exit_code": 127, "stdout": "", "stderr": f"program not available: {program}"}
        except subprocess.TimeoutExpired as error:
            return {"exit_code": 124, "stdout": error.stdout or "", "stderr": error.stderr or "timeout"}
        self.solver_evidence = None
        if completed.returncode == 0 and program != "checkMesh" and "End" in completed.stdout.split():
            self.solver_evidence = {"program": program, "case_sha256": self.fingerprint(), "log_tail": completed.stdout[-8000:]}
        return {"exit_code": completed.returncode, "stdout": completed.stdout[-16000:], "stderr": completed.stderr[-4000:]}

    def fingerprint(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        output = self._resolve("output")
        if not output.is_dir():
            return digest.hexdigest()
        for item in sorted(path for path in output.rglob("*") if path.is_file()):
            digest.update(item.relative_to(self.root).as_posix().encode())
            digest.update(hashlib.sha256(item.read_bytes()).digest())
        return digest.hexdigest()

    def collect_submission(self) -> dict[str, str]:
        output = self._resolve("output")
        if not output.is_dir():
            raise FoamBenchError("output directory is missing")
        return {item.relative_to(output).as_posix(): item.read_text(encoding="utf-8", errors="replace") for item in output.rglob("*") if item.is_file()}

    def toolset(self) -> WorkspaceToolset:
        """Expose generic agent tools while keeping FoamBench rules in this adapter."""
        return WorkspaceToolset(
            self.files,
            lambda program, timeout, args: self.run(program, timeout_seconds=timeout, extra_args=args),
            self.public_verification,
        )

    def public_verification(self) -> dict[str, object]:
        if self.solver_evidence is None or self.solver_evidence["case_sha256"] != self.fingerprint():
            return {"passed": False, "summary": "Run the configured solver on the current workspace first."}
        evidence = dict(self.solver_evidence)
        mesh = self.run("checkMesh", timeout_seconds=180)
        passed = mesh["exit_code"] == 0 and "Mesh OK." in str(mesh["stdout"])
        return {"passed": passed, "summary": "Public solver and mesh checks completed.", "evidence": {**evidence, "check_mesh": mesh}}
