"""Public validators: trusted host configuration, never model self-grading."""
import hashlib
import subprocess
from dataclasses import dataclass

from software_multiagent.core.execution import Evidence, VerificationResult, VerificationStatus


@dataclass(frozen=True)
class FileContentCheck:
    check_id: str
    path: str
    expected: bytes

    def __call__(self, context):
        from software_multiagent.runtime.reliability.reliable_solo import VerifiedFileRollback
        path = VerifiedFileRollback._path(context.state.task, self.path)
        actual = path.read_bytes() if path.is_file() else None
        passed = actual == self.expected
        return VerificationResult(self.check_id,
            VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            evidence=(Evidence(self.check_id + ":content", str(path),
                data={"exists": actual is not None,
                      "sha256": hashlib.sha256(actual).hexdigest() if actual is not None else None},
                software_native=True),), summary=f"Check required content of {self.path}")


@dataclass(frozen=True)
class NativeCommandCheck:
    """Run a public semantic/constraint checker whose exit status is its verdict.

    command is fixed by the host, not supplied by the model. This must be a real
    validator (e.g. a test suite), not the command that merely creates an artifact.
    Output is spooled to disk and only a bounded tail is returned as evidence.
    """
    check_id: str
    command: tuple[str, ...]
    timeout: float = 30
    max_output_bytes: int = 16000

    def __post_init__(self):
        if not self.command or self.timeout <= 0 or self.max_output_bytes <= 0:
            raise ValueError("validator requires a command and positive limits")

    def __call__(self, context):
        import tempfile
        try:
            with tempfile.TemporaryFile() as output:
                result = subprocess.run(self.command, cwd=context.state.task.workspace,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    timeout=self.timeout, shell=False)
                size = output.tell()
                output.seek(max(0, size - self.max_output_bytes))
                text = output.read().decode("utf-8", errors="replace")
            status = VerificationStatus.PASSED if result.returncode == 0 else VerificationStatus.FAILED
            return VerificationResult(self.check_id, status, evidence=(Evidence(
                self.check_id + ":command", "native-validator", data={
                    "command": self.command, "exit_code": result.returncode,
                    "output": text, "truncated": size > self.max_output_bytes},
                software_native=True),), summary=f"{self.check_id}: exit_code={result.returncode}; {text}")
        except (OSError, subprocess.TimeoutExpired) as error:
            return VerificationResult(self.check_id, VerificationStatus.INCONCLUSIVE,
                                      summary=f"validator unavailable: {error}")
