from __future__ import annotations

import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from software_multiagent.software_memory.errors import SnapshotFormatError
from software_multiagent.software_memory.persistence.migrations import SCHEMA_VERSION
from software_multiagent.software_memory.schema.models import (
    CompactEpisode,
    Entity,
    EvidenceRecord,
    ExecutionObservation,
    KnowledgeConflict,
    OperationContract,
    ObservationInterpretation,
    Relation,
    SoftwareIdentity,
    SoftwareProfile,
    StatusTransition,
    SkillCandidate,
    Workflow,
)
from software_multiagent.software_memory.schema.acceptance import AcceptanceContract, ContractAmendment, IntentLock
from software_multiagent.software_memory.schema.verification import VerificationMethodCandidate, VerificationPlan
from software_multiagent.software_memory.schema.outcomes import AcceptanceVerdict, TargetedRepairPackage, VerificationSignal
from software_multiagent.software_memory.persistence.storage import SQLiteMemoryStore


SNAPSHOT_FORMAT_VERSION = 1
T = TypeVar("T", bound=BaseModel)

_COLLECTIONS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("profiles.jsonl", SoftwareProfile, "software_profiles"),
    ("evidence.jsonl", EvidenceRecord, "evidence"),
    ("entities.jsonl", Entity, "entities"),
    ("relations.jsonl", Relation, "relations"),
    ("contracts.jsonl", OperationContract, "contracts"),
    ("workflows.jsonl", Workflow, "workflows"),
    ("observations.jsonl", ExecutionObservation, "observations"),
    ("interpretations.jsonl", ObservationInterpretation, "observation_interpretations"),
    ("episodes.jsonl", CompactEpisode, "compact_episodes"),
    ("skills.jsonl", SkillCandidate, "skill_candidates"),
    ("conflicts.jsonl", KnowledgeConflict, "knowledge_conflicts"),
    ("status_history.jsonl", StatusTransition, "status_history"),
    ("intent_locks.jsonl", IntentLock, "intent_locks"),
    ("acceptance_contracts.jsonl", AcceptanceContract, "acceptance_contracts"),
    ("contract_amendments.jsonl", ContractAmendment, "contract_amendments"),
    ("verification_candidates.jsonl", VerificationMethodCandidate, "verification_method_candidates"),
    ("verification_plans.jsonl", VerificationPlan, "verification_plans"),
    ("verification_signals.jsonl", VerificationSignal, "verification_signals"),
    ("acceptance_verdicts.jsonl", AcceptanceVerdict, "acceptance_verdicts"),
    ("targeted_repairs.jsonl", TargetedRepairPackage, "targeted_repair_packages"),
)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_jsonl(payloads: Iterable[str]) -> str:
    lines = [_canonical_json(json.loads(payload)) for payload in payloads]
    return "\n".join(lines) + ("\n" if lines else "")


def export_snapshot(
    store: SQLiteMemoryStore, software_id: str, directory: str | Path
) -> Path:
    """Export one software identity as a deterministic, checksummed snapshot."""
    software = store.get_software(software_id)
    if software is None:
        raise SnapshotFormatError("cannot export unknown software", software_id=software_id)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}

    software_text = _canonical_json(software.model_dump(mode="json", exclude_none=True)) + "\n"
    _atomic_text(output / "software.json", software_text)
    files["software.json"] = {
        "count": 1,
        "sha256": sha256(software_text.encode("utf-8")).hexdigest(),
    }

    for filename, _model, table in _COLLECTIONS:
        text = _canonical_jsonl(store.iter_payloads(table, software_id))
        _atomic_text(output / filename, text)
        files[filename] = {
            "count": len(text.splitlines()),
            "sha256": sha256(text.encode("utf-8")).hexdigest(),
        }

    manifest = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "storage_schema_version": SCHEMA_VERSION,
        "software_id": software_id,
        "files": files,
    }
    _atomic_text(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return output


def import_snapshot(store: SQLiteMemoryStore, directory: str | Path) -> SoftwareIdentity:
    """Restore a verified snapshot into a store where this software ID is absent."""
    source = Path(directory)
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotFormatError("snapshot manifest is missing or invalid") from error
    if manifest.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotFormatError(
            "unsupported snapshot format",
            found=manifest.get("snapshot_format_version"),
            supported=SNAPSHOT_FORMAT_VERSION,
        )
    if int(manifest.get("storage_schema_version", 0)) > SCHEMA_VERSION:
        raise SnapshotFormatError("snapshot requires a newer storage schema")

    documents: dict[str, str] = {}
    for filename, metadata in manifest.get("files", {}).items():
        try:
            text = (source / filename).read_text(encoding="utf-8")
        except OSError as error:
            raise SnapshotFormatError("snapshot data file is missing", filename=filename) from error
        digest = sha256(text.encode("utf-8")).hexdigest()
        if digest != metadata.get("sha256"):
            raise SnapshotFormatError("snapshot checksum mismatch", filename=filename)
        documents[filename] = text

    try:
        software = SoftwareIdentity.model_validate_json(documents["software.json"])
    except (KeyError, ValueError) as error:
        raise SnapshotFormatError("snapshot software identity is invalid") from error
    if software.software_id != manifest.get("software_id"):
        raise SnapshotFormatError("manifest and software identity disagree")
    if store.get_software(software.software_id) is not None:
        raise SnapshotFormatError(
            "snapshot import requires an absent software identity",
            software_id=software.software_id,
        )

    parsed: dict[str, list[BaseModel]] = {}
    try:
        for filename, model, _table in _COLLECTIONS:
            lines = documents.get(filename, "").splitlines()
            parsed[filename] = [model.model_validate_json(line) for line in lines if line.strip()]
    except ValueError as error:
        raise SnapshotFormatError("snapshot contains an invalid record", filename=filename) from error

    with store.batch():
        store.put_software(software)
        for item in parsed["profiles.jsonl"]:
            store.put_profile(item)  # type: ignore[arg-type]
        for item in parsed["evidence.jsonl"]:
            store.put_evidence(item)  # type: ignore[arg-type]
        for item in parsed["entities.jsonl"]:
            store.put_entity(item)  # type: ignore[arg-type]
        for item in parsed["relations.jsonl"]:
            store.put_relation(item)  # type: ignore[arg-type]
        for item in parsed["contracts.jsonl"]:
            store.put_contract(item)  # type: ignore[arg-type]
        for item in parsed["workflows.jsonl"]:
            store.put_workflow(item)  # type: ignore[arg-type]
        for item in parsed["observations.jsonl"]:
            store.put_observation(item)  # type: ignore[arg-type]
        for item in parsed["interpretations.jsonl"]:
            store.put_interpretation(item)  # type: ignore[arg-type]
        for item in parsed["episodes.jsonl"]:
            store.put_episode(item)  # type: ignore[arg-type]
        for item in parsed["skills.jsonl"]:
            store.put_skill_candidate(item)  # type: ignore[arg-type]
        for item in parsed["conflicts.jsonl"]:
            store.put_conflict(item)  # type: ignore[arg-type]
        for item in parsed["status_history.jsonl"]:
            store.restore_transition(item)  # type: ignore[arg-type]
        for item in parsed["intent_locks.jsonl"]:
            store.put_intent_lock(item)  # type: ignore[arg-type]
        for item in parsed["acceptance_contracts.jsonl"]:
            store.put_acceptance_contract(item)  # type: ignore[arg-type]
        for item in parsed["contract_amendments.jsonl"]:
            store.put_contract_amendment(item)  # type: ignore[arg-type]
        for item in parsed["verification_candidates.jsonl"]:
            store.put_verification_candidate(item)  # type: ignore[arg-type]
        for item in parsed["verification_plans.jsonl"]:
            store.put_verification_plan(item)  # type: ignore[arg-type]
        for item in parsed["verification_signals.jsonl"]:
            store.put_verification_signal(item)  # type: ignore[arg-type]
        for item in parsed["acceptance_verdicts.jsonl"]:
            store.put_acceptance_verdict(item)  # type: ignore[arg-type]
        for item in parsed["targeted_repairs.jsonl"]:
            store.put_targeted_repair_package(item)  # type: ignore[arg-type]
    return software
