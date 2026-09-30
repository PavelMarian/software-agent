"""Reusable knowledge, validation, and recovery mechanisms for software agents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from software_multiagent.core.action_graph import Action
from software_multiagent.core.contracts import AgentSpec, RuntimeState, TaskContext, ToolResult
from software_multiagent.core.execution import (
    ArtifactRef,
    Evidence,
    KnowledgeContext,
    KnowledgeFragment,
    RecoveryDecision,
    VerificationResult,
    VerificationStatus,
)


class NullKnowledgeProvider:
    def context_for(
        self, task: TaskContext, agent: AgentSpec, state: RuntimeState
    ) -> KnowledgeContext:
        del task, agent, state
        return KnowledgeContext()


class MetadataKnowledgeProvider:
    """Turn public task metadata into a compact, dependency-aware context pack."""

    def __init__(self, *, max_characters: int = 6_000) -> None:
        self.max_characters = max_characters

    def context_for(
        self, task: TaskContext, agent: AgentSpec, state: RuntimeState
    ) -> KnowledgeContext:
        del agent
        fragments: list[KnowledgeFragment] = []
        documentation = task.metadata.get("documentation", ())
        if isinstance(documentation, str):
            documentation = (documentation,)
        if isinstance(documentation, Mapping):
            documentation = tuple(documentation.values())
        if isinstance(documentation, Sequence):
            for index, item in enumerate(documentation):
                if isinstance(item, str) and item.strip():
                    fragments.append(
                        KnowledgeFragment(
                            f"documentation-{index}", item.strip(), "documentation", 0.8
                        )
                    )
        workstreams = task.metadata.get("workstreams", ())
        if isinstance(workstreams, Sequence) and not isinstance(workstreams, str):
            for index, item in enumerate(workstreams):
                if not isinstance(item, Mapping):
                    continue
                identifier = str(item.get("id", f"work-{index}"))
                description = str(item.get("description", "")).strip()
                dependencies = item.get("depends_on", ())
                if description:
                    fragments.append(
                        KnowledgeFragment(
                            identifier,
                            description,
                            "task-dependency",
                            1.0,
                            tuple(str(value) for value in dependencies)
                            if isinstance(dependencies, (list, tuple))
                            else (),
                        )
                    )
        submission = task.metadata.get("submission")
        if isinstance(submission, str) and submission.strip():
            fragments.append(
                KnowledgeFragment("submission", submission.strip(), "task-contract", 1.0)
            )
        for verification in state.verifications[-3:]:
            fragments.append(
                KnowledgeFragment(
                    f"verification-{verification.check_id}",
                    verification.summary or verification.status.value,
                    "native-verifier",
                    1.0,
                    verification.repair_targets,
                )
            )
        return KnowledgeContext(tuple(fragments), self.max_characters)


class ExitCodeValidator:
    """Interpret process-like tool output without asking the model to self-grade it."""

    def verify(
        self, action: Action, result: ToolResult, task: TaskContext
    ) -> VerificationResult | None:
        del task
        if result.error is not None:
            return VerificationResult(
                f"{action.id}:tool",
                VerificationStatus.FAILED,
                repair_targets=(action.id,),
                summary=result.error,
            )
        if not isinstance(result.output, Mapping) or "exit_code" not in result.output:
            return None
        exit_code = result.output.get("exit_code")
        if not isinstance(exit_code, int):
            return VerificationResult(
                f"{action.id}:exit-code",
                VerificationStatus.INCONCLUSIVE,
                summary="tool returned a non-integer exit code",
            )
        stdout = str(result.output.get("stdout", ""))[-2_000:]
        stderr = str(result.output.get("stderr", ""))[-2_000:]
        evidence = Evidence(
            id=f"{action.id}:process-result",
            source=action.tool,
            summary=f"exit_code={exit_code}",
            data={"exit_code": exit_code, "stdout": stdout, "stderr": stderr},
            software_native=True,
        )
        status = VerificationStatus.PASSED if exit_code == 0 else VerificationStatus.FAILED
        return VerificationResult(
            f"{action.id}:exit-code",
            status,
            evidence=(evidence,),
            repair_targets=() if exit_code == 0 else (action.id,),
            summary=(
                f"{action.tool} completed successfully"
                if exit_code == 0
                else f"{action.tool} failed with exit code {exit_code}: {stderr or stdout}"
            ),
        )


class BoundedRecoveryPolicy:
    """Allow bounded repair and prefer rollback for reversible failed mutations."""

    def __init__(self, max_repairs_per_action: int = 2) -> None:
        if max_repairs_per_action < 0:
            raise ValueError("max_repairs_per_action must be non-negative")
        self.max_repairs_per_action = max_repairs_per_action

    def decide(
        self,
        action: Action,
        verification: VerificationResult,
        attempt: int,
    ) -> RecoveryDecision:
        if verification.status != VerificationStatus.FAILED:
            return RecoveryDecision(False, False, "verification did not fail")
        retry = attempt <= self.max_repairs_per_action
        return RecoveryDecision(
            retry=retry,
            rollback=retry and action.mutates_workspace and action.reversible,
            reason=("repair permitted" if retry else "action repair limit reached"),
        )


class NullRollbackManager:
    def checkpoint(self, action: Action, task: TaskContext) -> None:
        del action, task
        return None

    def rollback(
        self, token: Any, action: Action, task: TaskContext
    ) -> Evidence | None:
        del token, action, task
        return None


@dataclass(frozen=True)
class _PathSnapshot:
    values: Mapping[str, bytes | None]


class LocalArtifactRollbackManager:
    """Restore explicitly declared file outputs inside a local task workspace."""

    def checkpoint(self, action: Action, task: TaskContext) -> _PathSnapshot | None:
        if not action.mutates_workspace or not action.artifact_outputs:
            return None
        values: dict[str, bytes | None] = {}
        for relative in action.artifact_outputs:
            target = _resolve_inside(task.workspace, relative)
            if target.exists() and not target.is_file():
                return None
            values[relative] = target.read_bytes() if target.is_file() else None
        return _PathSnapshot(values)

    def rollback(
        self, token: Any, action: Action, task: TaskContext
    ) -> Evidence | None:
        if not isinstance(token, _PathSnapshot):
            return None
        for relative, content in token.values.items():
            target = _resolve_inside(task.workspace, relative)
            if content is None:
                if target.is_file():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        return Evidence(
            id=f"{action.id}:rollback",
            source="local-workspace",
            summary="restored declared artifact outputs",
            data={"paths": sorted(token.values)},
            software_native=True,
        )


class SubmissionArtifactVerifier:
    """Independently verify the public file-submission contract at completion."""

    def verify(
        self, task: TaskContext, artifacts: Sequence[ArtifactRef]
    ) -> VerificationResult:
        del artifacts
        kind = task.metadata.get("submission_kind")
        expected: list[Path] = []
        repair_targets: list[str] = []
        if kind == "artifact_bundle":
            root = str(task.metadata.get("submission_root", "output"))
            target = _resolve_inside(task.workspace, root)
            if target.is_dir():
                expected = [path for path in sorted(target.rglob("*")) if path.is_file()]
            if not expected:
                repair_targets.append(root)
        elif kind == "workspace_files":
            for relative in task.metadata.get("submission_paths", ()):
                target = _resolve_inside(task.workspace, str(relative))
                if target.is_file():
                    expected.append(target)
                elif target.is_dir():
                    expected.extend(path for path in sorted(target.rglob("*")) if path.is_file())
                else:
                    repair_targets.append(str(relative))
        else:
            return VerificationResult(
                "submission-contract",
                VerificationStatus.INCONCLUSIVE,
                summary=f"no generic file verifier for submission kind {kind!r}",
            )
        refs = tuple(_artifact_ref(task.workspace, path) for path in expected)
        evidence = Evidence(
            "submission-artifacts",
            "workspace",
            f"found {len(refs)} submitted files",
            refs,
            software_native=True,
        )
        return VerificationResult(
            "submission-contract",
            VerificationStatus.FAILED if repair_targets else VerificationStatus.PASSED,
            evidence=(evidence,),
            repair_targets=tuple(repair_targets),
            summary=(
                f"missing required submission outputs: {', '.join(repair_targets)}"
                if repair_targets
                else f"submission contains {len(refs)} files"
            ),
        )


def _resolve_inside(workspace: Path, relative: str) -> Path:
    root = workspace.resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"path escapes workspace: {relative}") from error
    return target


def _artifact_ref(workspace: Path, path: Path) -> ArtifactRef:
    content = path.read_bytes()
    return ArtifactRef(
        path.relative_to(workspace.resolve()).as_posix(),
        sha256=hashlib.sha256(content).hexdigest(),
        producer="agent",
    )
