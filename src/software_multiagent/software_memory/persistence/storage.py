from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel

from software_multiagent.software_memory.errors import (
    FTS5UnavailableError,
    StorageConflictError,
    StorageInitializationError,
    TransactionAbortedError,
)
from software_multiagent.software_memory.persistence.migrations import SCHEMA_VERSION, apply_migrations
from software_multiagent.software_memory.schema.models import (
    CompactEpisode,
    ConflictState,
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
from software_multiagent.software_memory.schema.acceptance import (
    AcceptanceContract,
    ContractAmendment,
    IntentLock,
)
from software_multiagent.software_memory.schema.verification import VerificationMethodCandidate, VerificationPlan
from software_multiagent.software_memory.schema.outcomes import AcceptanceVerdict, TargetedRepairPackage, VerificationSignal


class MemoryConflictError(StorageConflictError):
    """Backward-compatible name for immutable/revision storage conflicts."""


def _payload(value: BaseModel) -> str:
    return value.model_dump_json(exclude_none=True)


class SQLiteMemoryStore:
    """SQLite/FTS5 store for domain-neutral software knowledge."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self._transaction_depth = 0
        self._rollback_only = False
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteMemoryStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _migrate(self) -> None:
        try:
            apply_migrations(self.connection)
        except sqlite3.OperationalError as error:
            if "fts5" in str(error).casefold():
                raise FTS5UnavailableError(
                    "SQLite was built without required FTS5 support"
                ) from error
            raise StorageInitializationError("software-memory storage migration failed") from error
        self._check_fts5()

    def _check_fts5(self) -> None:
        try:
            self.connection.execute("SELECT count(*) FROM contract_fts").fetchone()
            self.connection.execute("SELECT count(*) FROM entity_fts").fetchone()
        except sqlite3.OperationalError as error:
            raise FTS5UnavailableError(
                "SQLite FTS5 tables are unavailable"
            ) from error

    @contextmanager
    def _transaction(self):
        outermost = self._transaction_depth == 0
        caught_here = False
        if outermost:
            self.connection.execute("BEGIN")
            self._rollback_only = False
        self._transaction_depth += 1
        try:
            yield self
        except sqlite3.IntegrityError as error:
            caught_here = True
            self._rollback_only = True
            raise MemoryConflictError("write violates a storage integrity constraint") from error
        except BaseException:
            caught_here = True
            self._rollback_only = True
            raise
        finally:
            self._transaction_depth -= 1
            if outermost:
                rollback = self._rollback_only
                try:
                    self.connection.rollback() if rollback else self.connection.commit()
                finally:
                    self._rollback_only = False
                if rollback and not caught_here:
                    raise TransactionAbortedError(
                        "batch transaction was rolled back after a nested write failed"
                    )

    @contextmanager
    def batch(self):
        """Group multiple store/service writes in one atomic transaction."""
        with self._transaction():
            yield self

    def put_software(self, software: SoftwareIdentity) -> None:
        with self._transaction():
            self.connection.execute(
                """INSERT INTO software(software_id, product, version, vendor, payload)
                   VALUES(?, ?, ?, ?, ?)
                   ON CONFLICT(software_id) DO UPDATE SET
                     product=excluded.product, version=excluded.version,
                     vendor=excluded.vendor, payload=excluded.payload""",
                (
                    software.software_id,
                    software.product,
                    software.version,
                    software.vendor,
                    _payload(software),
                ),
            )

    def put_intent_lock(self, intent: IntentLock) -> None:
        existing = self.get_intent_lock(intent.lock_id)
        if existing is not None:
            if existing.model_dump(exclude={"created_at"}) != intent.model_dump(
                exclude={"created_at"}
            ):
                raise MemoryConflictError("intent lock is immutable")
            return
        with self._transaction():
            self.connection.execute(
                """INSERT INTO intent_locks(
                     lock_id, task_id, revision, lock_hash, created_at, payload
                   ) VALUES(?, ?, ?, ?, ?, ?)""",
                (
                    intent.lock_id,
                    intent.task_id,
                    intent.revision,
                    intent.lock_hash,
                    intent.created_at.isoformat(),
                    _payload(intent),
                ),
              )

    def put_verification_candidate(self, candidate: VerificationMethodCandidate) -> None:
        existing = self.get_verification_candidate(candidate.candidate_id)
        self._require_revision("verification candidate", existing, candidate)
        with self._transaction():
            self.connection.execute(
                """INSERT INTO verification_method_candidates(
                     candidate_id, software_id, criterion_id, status, revision, payload
                   ) VALUES(?, ?, ?, ?, ?, ?)
                   ON CONFLICT(candidate_id) DO UPDATE SET
                     status=excluded.status, revision=excluded.revision,
                     payload=excluded.payload""",
                (
                    candidate.candidate_id,
                    candidate.software_id,
                    candidate.criterion_id,
                    candidate.status.value,
                    candidate.revision,
                    _payload(candidate),
                ),
            )

    def get_verification_candidate(
        self, candidate_id: str
    ) -> VerificationMethodCandidate | None:
        row = self.connection.execute(
            "SELECT payload FROM verification_method_candidates WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        return VerificationMethodCandidate.model_validate_json(row["payload"]) if row else None

    def list_verification_candidates(
        self, software_id: str
    ) -> tuple[VerificationMethodCandidate, ...]:
        rows = self.connection.execute(
            """SELECT payload FROM verification_method_candidates
               WHERE software_id=? ORDER BY candidate_id""",
            (software_id,),
        ).fetchall()
        return tuple(VerificationMethodCandidate.model_validate_json(row["payload"]) for row in rows)

    def put_verification_plan(self, plan: VerificationPlan) -> None:
        existing = self.get_verification_plan(plan.plan_id)
        if existing is not None:
            if existing != plan:
                raise MemoryConflictError("verification plan is immutable")
            return
        with self._transaction():
            self.connection.execute(
                """INSERT INTO verification_plans(
                     plan_id, acceptance_contract_id, software_id, plan_hash, payload
                   ) VALUES(?, ?, ?, ?, ?)""",
                (
                    plan.plan_id,
                    plan.acceptance_contract_id,
                    plan.software_id,
                    plan.plan_hash,
                    _payload(plan),
                ),
            )

    def get_verification_plan(self, plan_id: str) -> VerificationPlan | None:
        row = self.connection.execute(
            "SELECT payload FROM verification_plans WHERE plan_id=?", (plan_id,)
        ).fetchone()
        return VerificationPlan.model_validate_json(row["payload"]) if row else None

    def list_verification_plans(self, contract_id: str) -> tuple[VerificationPlan, ...]:
        rows = self.connection.execute(
            """SELECT payload FROM verification_plans
               WHERE acceptance_contract_id=? ORDER BY plan_id""",
            (contract_id,),
        ).fetchall()
        return tuple(VerificationPlan.model_validate_json(row["payload"]) for row in rows)

    def put_verification_signal(self, signal: VerificationSignal) -> None:
        existing = self.get_verification_signal(signal.signal_id)
        if existing is not None:
            if existing != signal:
                raise MemoryConflictError("verification signal is immutable")
            return
        with self._transaction():
            self.connection.execute(
                """INSERT INTO verification_signals(
                     signal_id, software_id, acceptance_contract_id, criterion_id,
                     observation_id, outcome, payload
                   ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (signal.signal_id, signal.software_id, signal.acceptance_contract_id,
                 signal.criterion_id, signal.observation_id, signal.outcome.value,
                 _payload(signal)),
            )

    def get_verification_signal(self, signal_id: str) -> VerificationSignal | None:
        row = self.connection.execute(
            "SELECT payload FROM verification_signals WHERE signal_id=?", (signal_id,)
        ).fetchone()
        return VerificationSignal.model_validate_json(row["payload"]) if row else None

    def list_verification_signals(self, contract_id: str) -> tuple[VerificationSignal, ...]:
        rows = self.connection.execute(
            """SELECT payload FROM verification_signals
               WHERE acceptance_contract_id=? ORDER BY signal_id""", (contract_id,)
        ).fetchall()
        return tuple(VerificationSignal.model_validate_json(row["payload"]) for row in rows)

    def put_acceptance_verdict(self, verdict: AcceptanceVerdict) -> None:
        existing = self.get_acceptance_verdict(verdict.verdict_id)
        if existing is not None:
            if existing.model_dump(exclude={"created_at"}) != verdict.model_dump(exclude={"created_at"}):
                raise MemoryConflictError("acceptance verdict is immutable")
            return
        with self._transaction():
            self.connection.execute(
                """INSERT INTO acceptance_verdicts(
                     verdict_id, software_id, acceptance_contract_id, outcome, payload
                   ) VALUES(?, ?, ?, ?, ?)""",
                (verdict.verdict_id, verdict.software_id, verdict.acceptance_contract_id,
                 verdict.outcome.value, _payload(verdict)),
            )

    def get_acceptance_verdict(self, verdict_id: str) -> AcceptanceVerdict | None:
        row = self.connection.execute(
            "SELECT payload FROM acceptance_verdicts WHERE verdict_id=?", (verdict_id,)
        ).fetchone()
        return AcceptanceVerdict.model_validate_json(row["payload"]) if row else None

    def put_targeted_repair_package(self, package: TargetedRepairPackage) -> None:
        existing = self.get_targeted_repair_package(package.repair_package_id)
        if existing is not None:
            if existing != package:
                raise MemoryConflictError("targeted repair package is immutable")
            return
        with self._transaction():
            self.connection.execute(
                """INSERT INTO targeted_repair_packages(
                     repair_package_id, software_id, acceptance_contract_id,
                     criterion_id, payload
                   ) VALUES(?, ?, ?, ?, ?)""",
                (package.repair_package_id, package.software_id,
                 package.acceptance_contract_id, package.criterion_id, _payload(package)),
            )

    def get_targeted_repair_package(self, package_id: str) -> TargetedRepairPackage | None:
        row = self.connection.execute(
            "SELECT payload FROM targeted_repair_packages WHERE repair_package_id=?",
            (package_id,),
        ).fetchone()
        return TargetedRepairPackage.model_validate_json(row["payload"]) if row else None

    def get_intent_lock(self, lock_id: str) -> IntentLock | None:
        row = self.connection.execute(
            "SELECT payload FROM intent_locks WHERE lock_id=?", (lock_id,)
        ).fetchone()
        return IntentLock.model_validate_json(row["payload"]) if row else None

    def list_intent_locks(self, task_id: str) -> tuple[IntentLock, ...]:
        rows = self.connection.execute(
            "SELECT payload FROM intent_locks WHERE task_id=? ORDER BY revision, lock_id",
            (task_id,),
        ).fetchall()
        return tuple(IntentLock.model_validate_json(row["payload"]) for row in rows)

    def put_acceptance_contract(self, contract: AcceptanceContract) -> None:
        existing = self.get_acceptance_contract(contract.acceptance_contract_id)
        if existing is not None:
            if existing.model_dump(exclude={"created_at"}) != contract.model_dump(
                exclude={"created_at"}
            ):
                raise MemoryConflictError("acceptance contract is immutable")
            return
        with self._transaction():
            self.connection.execute(
                """INSERT INTO acceptance_contracts(
                     acceptance_contract_id, task_id, software_id, intent_lock_id,
                     revision, contract_hash, created_at, payload
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    contract.acceptance_contract_id,
                    contract.task_id,
                    contract.software_id,
                    contract.intent_lock_id,
                    contract.revision,
                    contract.contract_hash,
                    contract.created_at.isoformat(),
                    _payload(contract),
                ),
            )

    def get_acceptance_contract(self, contract_id: str) -> AcceptanceContract | None:
        row = self.connection.execute(
            "SELECT payload FROM acceptance_contracts WHERE acceptance_contract_id=?",
            (contract_id,),
        ).fetchone()
        return AcceptanceContract.model_validate_json(row["payload"]) if row else None

    def list_acceptance_contracts(self, task_id: str) -> tuple[AcceptanceContract, ...]:
        rows = self.connection.execute(
            """SELECT payload FROM acceptance_contracts
               WHERE task_id=? ORDER BY revision, acceptance_contract_id""",
            (task_id,),
        ).fetchall()
        return tuple(
            AcceptanceContract.model_validate_json(row["payload"]) for row in rows
        )

    def put_contract_amendment(self, amendment: ContractAmendment) -> None:
        existing = self.get_contract_amendment(amendment.amendment_id)
        if existing is not None:
            if existing.model_dump(exclude={"created_at"}) != amendment.model_dump(
                exclude={"created_at"}
            ):
                raise MemoryConflictError("contract amendment is immutable")
            return
        with self._transaction():
            self.connection.execute(
                """INSERT INTO contract_amendments(
                     amendment_id, task_id, previous_contract_id,
                     new_contract_id, created_at, payload
                   ) VALUES(?, ?, ?, ?, ?, ?)""",
                (
                    amendment.amendment_id,
                    amendment.task_id,
                    amendment.previous_contract_id,
                    amendment.new_contract_id,
                    amendment.created_at.isoformat(),
                    _payload(amendment),
                ),
            )

    def get_contract_amendment(self, amendment_id: str) -> ContractAmendment | None:
        row = self.connection.execute(
            "SELECT payload FROM contract_amendments WHERE amendment_id=?",
            (amendment_id,),
        ).fetchone()
        return ContractAmendment.model_validate_json(row["payload"]) if row else None

    def list_contract_amendments(self, task_id: str) -> tuple[ContractAmendment, ...]:
        rows = self.connection.execute(
            """SELECT payload FROM contract_amendments
               WHERE task_id=? ORDER BY created_at, amendment_id""",
            (task_id,),
        ).fetchall()
        return tuple(ContractAmendment.model_validate_json(row["payload"]) for row in rows)

    def get_software(self, software_id: str) -> SoftwareIdentity | None:
        row = self.connection.execute(
            "SELECT payload FROM software WHERE software_id = ?", (software_id,)
        ).fetchone()
        return SoftwareIdentity.model_validate_json(row["payload"]) if row else None

    def put_profile(self, profile: SoftwareProfile) -> None:
        row = self.connection.execute(
            "SELECT payload FROM software_profiles WHERE profile_id = ?", (profile.profile_id,)
        ).fetchone()
        if row is not None:
            existing = SoftwareProfile.model_validate_json(row["payload"])
            if profile.revision < existing.revision:
                raise MemoryConflictError("cannot overwrite a newer software profile revision")
            if profile.revision == existing.revision and profile != existing:
                raise MemoryConflictError("changed software profile requires a higher revision")
        with self._transaction():
            self.connection.execute(
                """INSERT INTO software_profiles(profile_id, software_id, revision, payload)
                   VALUES(?, ?, ?, ?)
                   ON CONFLICT(profile_id) DO UPDATE SET
                     software_id=excluded.software_id, revision=excluded.revision,
                     payload=excluded.payload""",
                (profile.profile_id, profile.software_id, profile.revision, _payload(profile)),
            )

    def get_profile(self, profile_id: str) -> SoftwareProfile | None:
        row = self.connection.execute(
            "SELECT payload FROM software_profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        return SoftwareProfile.model_validate_json(row["payload"]) if row else None

    def list_profiles(self, software_id: str) -> tuple[SoftwareProfile, ...]:
        rows = self.connection.execute(
            "SELECT payload FROM software_profiles WHERE software_id=? ORDER BY profile_id",
            (software_id,),
        ).fetchall()
        return tuple(SoftwareProfile.model_validate_json(row["payload"]) for row in rows)

    def put_evidence(self, evidence: EvidenceRecord) -> None:
        existing = self.connection.execute(
            "SELECT payload FROM evidence WHERE evidence_id = ?", (evidence.evidence_id,)
        ).fetchone()
        if existing is not None:
            stored = EvidenceRecord.model_validate_json(existing["payload"])
            stable_stored = stored.model_dump(exclude={"retrieved_at"})
            stable_incoming = evidence.model_dump(exclude={"retrieved_at"})
            if stable_stored != stable_incoming:
                raise MemoryConflictError(f"immutable evidence differs: {evidence.evidence_id}")
            return
        with self._transaction():
            self.connection.execute(
                """INSERT INTO evidence(
                     evidence_id, software_id, source_kind, source_uri, locator,
                     content_hash, authoritative, payload
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evidence.evidence_id,
                    evidence.software_id,
                    evidence.source_kind.value,
                    evidence.source_uri,
                    evidence.locator,
                    evidence.content_hash,
                    int(evidence.authoritative),
                    _payload(evidence),
                ),
            )

    def get_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        row = self.connection.execute(
            "SELECT payload FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return EvidenceRecord.model_validate_json(row["payload"]) if row else None

    def list_evidence(self, software_id: str) -> tuple[EvidenceRecord, ...]:
        rows = self.connection.execute(
            "SELECT payload FROM evidence WHERE software_id=? ORDER BY evidence_id",
            (software_id,),
        ).fetchall()
        return tuple(EvidenceRecord.model_validate_json(row["payload"]) for row in rows)

    def evidence_exist(self, evidence_ids: Iterable[str]) -> bool:
        ids = tuple(dict.fromkeys(evidence_ids))
        if not ids:
            return True
        placeholders = ",".join("?" for _ in ids)
        count = self.connection.execute(
            f"SELECT count(*) AS n FROM evidence WHERE evidence_id IN ({placeholders})", ids
        ).fetchone()["n"]
        return count == len(ids)

    def put_entity(self, entity: Entity) -> None:
        self._require_evidence(entity.evidence_ids)
        existing = self.get_entity(entity.entity_id)
        self._require_revision("entity", existing, entity)
        with self._transaction():
            self.connection.execute(
                """INSERT INTO entities(entity_id, software_id, entity_type, name, status, payload)
                   VALUES(?, ?, ?, ?, ?, ?)
                   ON CONFLICT(entity_id) DO UPDATE SET
                     entity_type=excluded.entity_type, name=excluded.name,
                     status=excluded.status, payload=excluded.payload""",
                (
                    entity.entity_id,
                    entity.software_id,
                    entity.entity_type,
                    entity.name,
                    entity.status.value,
                    _payload(entity),
                ),
            )
            self.connection.execute("DELETE FROM entity_fts WHERE entity_id = ?", (entity.entity_id,))
            self.connection.execute(
                "INSERT INTO entity_fts VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    entity.entity_id,
                    entity.software_id,
                    entity.entity_type,
                    entity.name,
                    entity.summary,
                    " ".join(entity.aliases),
                    json.dumps(entity.attributes, ensure_ascii=False, sort_keys=True),
                ),
            )

    def get_entity(self, entity_id: str) -> Entity | None:
        row = self.connection.execute(
            "SELECT payload FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        return Entity.model_validate_json(row["payload"]) if row else None

    def list_entities(self, software_id: str) -> tuple[Entity, ...]:
        rows = self.connection.execute(
            "SELECT payload FROM entities WHERE software_id=? ORDER BY entity_id",
            (software_id,),
        ).fetchall()
        return tuple(Entity.model_validate_json(row["payload"]) for row in rows)

    def put_relation(self, relation: Relation) -> None:
        self._require_evidence(relation.evidence_ids)
        existing = self.get_relation(relation.relation_id)
        self._require_revision("relation", existing, relation)
        with self._transaction():
            self.connection.execute(
                """INSERT INTO relations(
                     relation_id, software_id, source_entity_id, relation_type,
                     target_entity_id, status, payload
                   ) VALUES(?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(relation_id) DO UPDATE SET
                     relation_type=excluded.relation_type, status=excluded.status,
                     payload=excluded.payload""",
                (
                    relation.relation_id,
                    relation.software_id,
                    relation.source_entity_id,
                    relation.relation_type,
                    relation.target_entity_id,
                    relation.status.value,
                    _payload(relation),
                ),
            )

    def get_relation(self, relation_id: str) -> Relation | None:
        row = self.connection.execute(
            "SELECT payload FROM relations WHERE relation_id=?", (relation_id,)
        ).fetchone()
        return Relation.model_validate_json(row["payload"]) if row else None

    def list_relations(self, software_id: str) -> tuple[Relation, ...]:
        rows = self.connection.execute(
            "SELECT payload FROM relations WHERE software_id=? ORDER BY relation_id",
            (software_id,),
        ).fetchall()
        return tuple(Relation.model_validate_json(row["payload"]) for row in rows)

    def related(self, software_id: str, entity_ids: Iterable[str]) -> tuple[Relation, ...]:
        ids = tuple(dict.fromkeys(entity_ids))
        if not ids:
            return ()
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"""SELECT payload FROM relations WHERE software_id = ? AND
                (source_entity_id IN ({placeholders}) OR target_entity_id IN ({placeholders}))""",
            (software_id, *ids, *ids),
        ).fetchall()
        return tuple(Relation.model_validate_json(row["payload"]) for row in rows)

    def put_contract(self, contract: OperationContract) -> None:
        self._require_evidence(contract.evidence_ids)
        existing = self.get_contract(contract.contract_id)
        if existing is not None and contract.revision < existing.revision:
            raise MemoryConflictError("cannot overwrite a newer contract revision")
        if existing is not None and contract.revision == existing.revision and existing != contract:
            raise MemoryConflictError("changed contract requires a higher revision")
        with self._transaction():
            self.connection.execute(
                """INSERT INTO contracts(
                     contract_id, software_id, name, interface, status, revision, payload
                   ) VALUES(?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(contract_id) DO UPDATE SET
                     name=excluded.name, interface=excluded.interface,
                     status=excluded.status, revision=excluded.revision,
                     payload=excluded.payload""",
                (
                    contract.contract_id,
                    contract.software_id,
                    contract.name,
                    contract.interface,
                    contract.status.value,
                    contract.revision,
                    _payload(contract),
                ),
            )
            self.connection.execute(
                "DELETE FROM contract_fts WHERE contract_id = ?", (contract.contract_id,)
            )
            self._insert_contract_fts(contract)

    def apply_contract_transition(
        self, contract: OperationContract, transition: StatusTransition
    ) -> None:
        self.apply_status_transition(contract, transition)

    def apply_status_transition(
        self,
        item: Entity | OperationContract | Relation | Workflow,
        transition: StatusTransition,
    ) -> None:
        """Atomically replace a knowledge item and append its trust audit record."""
        kind, item_id, current = self._identify_item(item)
        if current is None:
            raise KeyError(item_id)
        if item.revision != current.revision + 1:
            raise MemoryConflictError("transition must advance the item by one revision")
        if (
            transition.item_kind != kind
            or transition.item_id != item_id
            or transition.software_id != item.software_id
            or transition.from_status != current.status
            or transition.to_status != item.status
        ):
            raise MemoryConflictError("transition audit does not match knowledge status change")
        self._require_evidence(item.evidence_ids)
        with self._transaction():
            self.connection.execute(
                """INSERT INTO status_history(
                     transition_id, item_kind, item_id, from_status, to_status,
                     actor_kind, created_at, payload, software_id
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    transition.transition_id,
                    transition.item_kind,
                    transition.item_id,
                    transition.from_status.value,
                    transition.to_status.value,
                    transition.actor_kind.value,
                    transition.created_at.isoformat(),
                    _payload(transition),
                    transition.software_id,
                ),
            )
            if isinstance(item, OperationContract):
                self.connection.execute(
                    """UPDATE contracts SET name=?, interface=?, status=?, revision=?, payload=?
                       WHERE contract_id=?""",
                    (
                        item.name,
                        item.interface,
                        item.status.value,
                        item.revision,
                        _payload(item),
                        item.contract_id,
                    ),
                )
                self.connection.execute(
                    "DELETE FROM contract_fts WHERE contract_id=?", (item.contract_id,)
                )
                self._insert_contract_fts(item)
            elif isinstance(item, Entity):
                self.connection.execute(
                    "UPDATE entities SET name=?, status=?, payload=? WHERE entity_id=?",
                    (item.name, item.status.value, _payload(item), item.entity_id),
                )
                self.connection.execute("DELETE FROM entity_fts WHERE entity_id=?", (item.entity_id,))
                self._insert_entity_fts(item)
            elif isinstance(item, Relation):
                self.connection.execute(
                    "UPDATE relations SET status=?, payload=? WHERE relation_id=?",
                    (item.status.value, _payload(item), item.relation_id),
                )
            else:
                self.connection.execute(
                    "UPDATE workflows SET name=?, status=?, payload=? WHERE workflow_id=?",
                    (item.name, item.status.value, _payload(item), item.workflow_id),
                )

    def _identify_item(self, item):  # type: ignore[no-untyped-def]
        if isinstance(item, Entity):
            return "entity", item.entity_id, self.get_entity(item.entity_id)
        if isinstance(item, OperationContract):
            return "contract", item.contract_id, self.get_contract(item.contract_id)
        if isinstance(item, Relation):
            return "relation", item.relation_id, self.get_relation(item.relation_id)
        if isinstance(item, Workflow):
            return "workflow", item.workflow_id, self.get_workflow(item.workflow_id)
        raise TypeError(f"unsupported knowledge item: {type(item)!r}")

    def _insert_contract_fts(self, contract: OperationContract) -> None:
        parameters = " ".join(
            f"{parameter.name} {parameter.description or ''}" for parameter in contract.parameters
        )
        predicates = " ".join(
            predicate.predicate
            for predicate in (*contract.preconditions, *contract.effects, *contract.success_signals)
        )
        failures = " ".join(
            f"{failure.pattern} {failure.description}" for failure in contract.failure_signatures
        )
        self.connection.execute(
            "INSERT INTO contract_fts VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                contract.contract_id,
                contract.software_id,
                contract.name,
                contract.interface,
                contract.purpose,
                parameters,
                predicates,
                failures,
            ),
        )

    def _insert_entity_fts(self, entity: Entity) -> None:
        self.connection.execute(
            "INSERT INTO entity_fts VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                entity.entity_id,
                entity.software_id,
                entity.entity_type,
                entity.name,
                entity.summary,
                " ".join(entity.aliases),
                json.dumps(entity.attributes, ensure_ascii=False, sort_keys=True),
            ),
        )

    def get_contract(self, contract_id: str) -> OperationContract | None:
        row = self.connection.execute(
            "SELECT payload FROM contracts WHERE contract_id = ?", (contract_id,)
        ).fetchone()
        return OperationContract.model_validate_json(row["payload"]) if row else None

    def list_contracts(self, software_id: str) -> tuple[OperationContract, ...]:
        rows = self.connection.execute(
            "SELECT payload FROM contracts WHERE software_id = ? ORDER BY name", (software_id,)
        ).fetchall()
        return tuple(OperationContract.model_validate_json(row["payload"]) for row in rows)

    def search_contracts(
        self, software_id: str, query: str, limit: int
    ) -> tuple[tuple[float, OperationContract], ...]:
        if not query.strip():
            return tuple((0.0, item) for item in self.list_contracts(software_id)[:limit])
        match = " OR ".join(f'"{token}"' for token in _search_tokens(query))
        if not match:
            return ()
        rows = self.connection.execute(
            """SELECT c.payload, bm25(contract_fts) AS rank
               FROM contract_fts JOIN contracts c USING(contract_id)
               WHERE contract_fts MATCH ? AND contract_fts.software_id = ?
               ORDER BY rank LIMIT ?""",
            (match, software_id, limit),
        ).fetchall()
        return tuple((-float(row["rank"]), OperationContract.model_validate_json(row["payload"])) for row in rows)

    def search_entities(
        self, software_id: str, query: str, limit: int, entity_types: tuple[str, ...] = ()
    ) -> tuple[tuple[float, Entity], ...]:
        match = " OR ".join(f'"{token}"' for token in _search_tokens(query))
        if not match:
            return ()
        type_clause = ""
        values: list[object] = [match, software_id]
        if entity_types:
            type_clause = f" AND e.entity_type IN ({','.join('?' for _ in entity_types)})"
            values.extend(entity_types)
        values.append(limit)
        rows = self.connection.execute(
            f"""SELECT e.payload, bm25(entity_fts) AS rank
                FROM entity_fts JOIN entities e USING(entity_id)
                WHERE entity_fts MATCH ? AND entity_fts.software_id = ?{type_clause}
                ORDER BY rank LIMIT ?""",
            values,
        ).fetchall()
        return tuple((-float(row["rank"]), Entity.model_validate_json(row["payload"])) for row in rows)

    def put_workflow(self, workflow: Workflow) -> None:
        self._require_evidence(workflow.evidence_ids)
        existing = self.get_workflow(workflow.workflow_id)
        self._require_revision("workflow", existing, workflow)
        with self._transaction():
            self.connection.execute(
                """INSERT INTO workflows(workflow_id, software_id, name, status, payload)
                   VALUES(?, ?, ?, ?, ?)
                   ON CONFLICT(workflow_id) DO UPDATE SET
                     name=excluded.name, status=excluded.status, payload=excluded.payload""",
                (
                    workflow.workflow_id,
                    workflow.software_id,
                    workflow.name,
                    workflow.status.value,
                    _payload(workflow),
                ),
            )

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        row = self.connection.execute(
            "SELECT payload FROM workflows WHERE workflow_id=?", (workflow_id,)
        ).fetchone()
        return Workflow.model_validate_json(row["payload"]) if row else None

    def list_workflows(self, software_id: str) -> tuple[Workflow, ...]:
        rows = self.connection.execute(
            "SELECT payload FROM workflows WHERE software_id=? ORDER BY workflow_id",
            (software_id,),
        ).fetchall()
        return tuple(Workflow.model_validate_json(row["payload"]) for row in rows)

    def put_observation(self, observation: ExecutionObservation) -> None:
        self._require_evidence(observation.evidence_ids)
        with self._transaction():
            self.connection.execute(
                """INSERT INTO observations(
                     observation_id, software_id, contract_id, actor_kind,
                     verification, created_at, payload
                   ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (
                    observation.observation_id,
                    observation.software_id,
                    observation.contract_id,
                    observation.actor_kind.value,
                    observation.verification.value,
                    observation.created_at.isoformat(),
                    _payload(observation),
                ),
            )

    def get_observation(self, observation_id: str) -> ExecutionObservation | None:
        row = self.connection.execute(
            "SELECT payload FROM observations WHERE observation_id = ?", (observation_id,)
        ).fetchone()
        return ExecutionObservation.model_validate_json(row["payload"]) if row else None

    def list_observations(self, software_id: str) -> tuple[ExecutionObservation, ...]:
        rows = self.connection.execute(
            "SELECT payload FROM observations WHERE software_id=? ORDER BY observation_id",
            (software_id,),
        ).fetchall()
        return tuple(ExecutionObservation.model_validate_json(row["payload"]) for row in rows)

    def put_interpretation(self, interpretation: ObservationInterpretation) -> None:
        with self._transaction():
            self.connection.execute(
                """INSERT INTO observation_interpretations(
                     interpretation_id, software_id, observation_id,
                     verification, created_at, payload
                   ) VALUES(?, ?, ?, ?, ?, ?)""",
                (
                    interpretation.interpretation_id,
                    interpretation.software_id,
                    interpretation.observation_id,
                    interpretation.verification.value,
                    interpretation.created_at.isoformat(),
                    _payload(interpretation),
                ),
            )

    def get_interpretation(self, interpretation_id: str) -> ObservationInterpretation | None:
        row = self.connection.execute(
            "SELECT payload FROM observation_interpretations WHERE interpretation_id=?",
            (interpretation_id,),
        ).fetchone()
        return ObservationInterpretation.model_validate_json(row["payload"]) if row else None

    def list_interpretations(self, software_id: str) -> tuple[ObservationInterpretation, ...]:
        rows = self.connection.execute(
            """SELECT payload FROM observation_interpretations
               WHERE software_id=? ORDER BY interpretation_id""",
            (software_id,),
        ).fetchall()
        return tuple(ObservationInterpretation.model_validate_json(row["payload"]) for row in rows)

    def put_episode(self, episode: CompactEpisode) -> None:
        with self._transaction():
            self.connection.execute(
                """INSERT INTO compact_episodes(
                     episode_id, software_id, observation_id, task_id,
                     verification, created_at, payload
                   ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode.episode_id,
                    episode.software_id,
                    episode.observation_id,
                    episode.task_id,
                    episode.verification.value,
                    episode.created_at.isoformat(),
                    _payload(episode),
                ),
            )

    def get_episode(self, episode_id: str) -> CompactEpisode | None:
        row = self.connection.execute(
            "SELECT payload FROM compact_episodes WHERE episode_id=?",
            (episode_id,),
        ).fetchone()
        return CompactEpisode.model_validate_json(row["payload"]) if row else None

    def list_episodes(self, software_id: str) -> tuple[CompactEpisode, ...]:
        rows = self.connection.execute(
            """SELECT payload FROM compact_episodes
               WHERE software_id=? ORDER BY episode_id""",
            (software_id,),
        ).fetchall()
        return tuple(CompactEpisode.model_validate_json(row["payload"]) for row in rows)

    def put_skill_candidate(self, candidate: SkillCandidate) -> None:
        existing = self.get_skill_candidate(candidate.candidate_id)
        self._require_revision("skill candidate", existing, candidate)
        with self._transaction():
            self.connection.execute(
                """INSERT INTO skill_candidates(
                     candidate_id, software_id, status, revision, payload
                   ) VALUES(?, ?, ?, ?, ?)
                   ON CONFLICT(candidate_id) DO UPDATE SET
                     status=excluded.status,
                     revision=excluded.revision,
                     payload=excluded.payload""",
                (
                    candidate.candidate_id,
                    candidate.software_id,
                    candidate.status,
                    candidate.revision,
                    _payload(candidate),
                ),
            )

    def get_skill_candidate(self, candidate_id: str) -> SkillCandidate | None:
        row = self.connection.execute(
            "SELECT payload FROM skill_candidates WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        return SkillCandidate.model_validate_json(row["payload"]) if row else None

    def list_skill_candidates(self, software_id: str) -> tuple[SkillCandidate, ...]:
        rows = self.connection.execute(
            """SELECT payload FROM skill_candidates
               WHERE software_id=? ORDER BY candidate_id""",
            (software_id,),
        ).fetchall()
        return tuple(SkillCandidate.model_validate_json(row["payload"]) for row in rows)

    def status_history(
        self, item_id: str, *, item_kind: str | None = None
    ) -> tuple[StatusTransition, ...]:
        kind_clause = " AND item_kind=?" if item_kind else ""
        parameters = (item_id, item_kind) if item_kind else (item_id,)
        rows = self.connection.execute(
            f"SELECT payload, software_id FROM status_history "
            f"WHERE item_id=?{kind_clause} ORDER BY sequence",
            parameters,
        ).fetchall()
        return tuple(self._transition_from_row(row) for row in rows)

    @staticmethod
    def _transition_from_row(row: sqlite3.Row) -> StatusTransition:
        data = json.loads(row["payload"])
        data.setdefault("software_id", row["software_id"] or "unknown")
        return StatusTransition.model_validate(data)

    def restore_transition(self, transition: StatusTransition) -> None:
        """Restore an already audited transition from a trusted snapshot."""
        with self._transaction():
            self.connection.execute(
                """INSERT INTO status_history(
                     transition_id, item_kind, item_id, from_status, to_status,
                     actor_kind, created_at, payload, software_id
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    transition.transition_id,
                    transition.item_kind,
                    transition.item_id,
                    transition.from_status.value,
                    transition.to_status.value,
                    transition.actor_kind.value,
                    transition.created_at.isoformat(),
                    _payload(transition),
                    transition.software_id,
                ),
            )

    def put_conflict(self, conflict: KnowledgeConflict) -> None:
        self._require_evidence(conflict.evidence_ids)
        with self._transaction():
            self.connection.execute(
                """INSERT INTO knowledge_conflicts(
                     conflict_id, software_id, item_kind, item_id,
                     conflicting_item_id, status, reason, payload
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(conflict_id) DO UPDATE SET
                     status=excluded.status, reason=excluded.reason, payload=excluded.payload""",
                (
                    conflict.conflict_id,
                    conflict.software_id,
                    conflict.item_kind,
                    conflict.item_id,
                    conflict.conflicting_item_id,
                    conflict.state.value,
                    conflict.reason,
                    _payload(conflict),
                ),
            )

    def get_conflict(self, conflict_id: str) -> KnowledgeConflict | None:
        row = self.connection.execute(
            "SELECT payload FROM knowledge_conflicts WHERE conflict_id=?", (conflict_id,)
        ).fetchone()
        return KnowledgeConflict.model_validate_json(row["payload"]) if row else None

    def list_conflicts(
        self, software_id: str, *, state: ConflictState | None = None
    ) -> tuple[KnowledgeConflict, ...]:
        clause = " AND status=?" if state else ""
        parameters = (software_id, state.value) if state else (software_id,)
        rows = self.connection.execute(
            f"SELECT payload FROM knowledge_conflicts WHERE software_id=?{clause} ORDER BY conflict_id",
            parameters,
        ).fetchall()
        return tuple(KnowledgeConflict.model_validate_json(row["payload"]) for row in rows)

    def _require_evidence(self, evidence_ids: Iterable[str]) -> None:
        if not self.evidence_exist(evidence_ids):
            raise ValueError("one or more evidence references do not exist")

    @staticmethod
    def _require_revision(kind: str, existing, incoming) -> None:  # type: ignore[no-untyped-def]
        if existing is None:
            return
        if incoming.revision < existing.revision:
            raise MemoryConflictError(f"cannot overwrite a newer {kind} revision")
        if incoming.revision == existing.revision and incoming != existing:
            raise MemoryConflictError(f"changed {kind} requires a higher revision")

    def iter_payloads(self, table: str, software_id: str) -> Iterator[str]:
        if table not in {
            "software_profiles",
            "entities",
            "relations",
            "contracts",
            "workflows",
            "evidence",
            "observations",
            "observation_interpretations",
            "compact_episodes",
            "skill_candidates",
            "intent_locks",
            "acceptance_contracts",
            "contract_amendments",
            "verification_method_candidates",
            "verification_plans",
            "verification_signals",
            "acceptance_verdicts",
            "targeted_repair_packages",
            "status_history",
            "knowledge_conflicts",
        }:
            raise ValueError(f"unsupported table: {table}")
        order_columns = {
            "software_profiles": "profile_id",
            "entities": "entity_id",
            "relations": "relation_id",
            "contracts": "contract_id",
            "workflows": "workflow_id",
            "evidence": "evidence_id",
            "observations": "observation_id",
            "observation_interpretations": "interpretation_id",
            "compact_episodes": "episode_id",
            "skill_candidates": "candidate_id",
            "acceptance_contracts": "acceptance_contract_id",
            "verification_method_candidates": "candidate_id",
            "verification_plans": "plan_id",
            "verification_signals": "signal_id",
            "acceptance_verdicts": "verdict_id",
            "targeted_repair_packages": "repair_package_id",
            "status_history": "sequence",
            "knowledge_conflicts": "conflict_id",
        }
        if table == "status_history":
            rows = self.connection.execute(
                "SELECT payload, software_id FROM status_history "
                "WHERE software_id=? ORDER BY sequence",
                (software_id,),
            )
            for row in rows:
                yield _payload(self._transition_from_row(row))
            return
        if table == "intent_locks":
            rows = self.connection.execute(
                """SELECT payload FROM intent_locks
                   WHERE lock_id IN (
                     SELECT DISTINCT intent_lock_id FROM acceptance_contracts
                     WHERE software_id=?
                   ) ORDER BY lock_id""",
                (software_id,),
            )
            for row in rows:
                yield str(row["payload"])
            return
        if table == "contract_amendments":
            rows = self.connection.execute(
                """SELECT payload FROM contract_amendments
                   WHERE previous_contract_id IN (
                     SELECT acceptance_contract_id FROM acceptance_contracts
                     WHERE software_id=?
                   ) ORDER BY amendment_id""",
                (software_id,),
            )
            for row in rows:
                yield str(row["payload"])
            return
        rows = self.connection.execute(
            f"SELECT payload FROM {table} WHERE software_id=? ORDER BY {order_columns[table]}",
            (software_id,),
        )
        for row in rows:
            yield str(row["payload"])


def _search_tokens(query: str) -> tuple[str, ...]:
    # Quoted FTS terms avoid exposing user text as an FTS expression.
    values = []
    for raw in query.replace("/", " ").replace("-", " ").split():
        token = "".join(char for char in raw if char.isalnum() or char == "_")
        if token:
            values.append(token.replace('"', '""'))
    return tuple(dict.fromkeys(values))
