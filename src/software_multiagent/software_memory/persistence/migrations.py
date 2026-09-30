from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    sql: str


MIGRATIONS = (
    Migration(
        1,
        "initial software-memory schema",
        """
        CREATE TABLE software (
            software_id TEXT PRIMARY KEY,
            product TEXT NOT NULL,
            version TEXT NOT NULL,
            vendor TEXT,
            payload TEXT NOT NULL
        );
        CREATE TABLE software_profiles (
            profile_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            revision INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE evidence (
            evidence_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            source_kind TEXT NOT NULL,
            source_uri TEXT NOT NULL,
            locator TEXT,
            content_hash TEXT NOT NULL,
            authoritative INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE entity_fts USING fts5(
            entity_id UNINDEXED, software_id UNINDEXED, entity_type, name,
            summary, aliases, attributes, tokenize='unicode61'
        );
        CREATE TABLE relations (
            relation_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            source_entity_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            status TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX relations_source_idx ON relations(software_id, source_entity_id);
        CREATE INDEX relations_target_idx ON relations(software_id, target_entity_id);
        CREATE TABLE contracts (
            contract_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            name TEXT NOT NULL,
            interface TEXT NOT NULL,
            status TEXT NOT NULL,
            revision INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE contract_fts USING fts5(
            contract_id UNINDEXED, software_id UNINDEXED, name, interface,
            purpose, parameters, predicates, failures, tokenize='unicode61'
        );
        CREATE TABLE workflows (
            workflow_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE observations (
            observation_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            contract_id TEXT,
            actor_kind TEXT NOT NULL,
            verification TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX observations_contract_idx ON observations(software_id, contract_id);
        CREATE TABLE status_history (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            transition_id TEXT UNIQUE NOT NULL,
            item_kind TEXT NOT NULL,
            item_id TEXT NOT NULL,
            from_status TEXT NOT NULL,
            to_status TEXT NOT NULL,
            actor_kind TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE knowledge_conflicts (
            conflict_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            item_kind TEXT NOT NULL,
            item_id TEXT NOT NULL,
            conflicting_item_id TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            reason TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """,
    ),
    Migration(
        2,
        "generalized audit and conflict indexes",
        """
        ALTER TABLE status_history ADD COLUMN software_id TEXT;
        UPDATE status_history SET software_id = CASE item_kind
            WHEN 'contract' THEN (SELECT software_id FROM contracts WHERE contract_id=item_id)
            WHEN 'entity' THEN (SELECT software_id FROM entities WHERE entity_id=item_id)
            WHEN 'relation' THEN (SELECT software_id FROM relations WHERE relation_id=item_id)
            WHEN 'workflow' THEN (SELECT software_id FROM workflows WHERE workflow_id=item_id)
            ELSE NULL
        END;
        CREATE INDEX status_history_item_idx ON status_history(item_kind, item_id, sequence);
        CREATE INDEX status_history_software_idx ON status_history(software_id, sequence);
        CREATE INDEX knowledge_conflicts_item_idx
            ON knowledge_conflicts(software_id, item_kind, item_id, status);
        """,
    ),
    Migration(
        3,
        "episodic interpretations, compact episodes, and skill candidates",
        """
        CREATE TABLE observation_interpretations (
            interpretation_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            observation_id TEXT NOT NULL REFERENCES observations(observation_id),
            verification TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX observation_interpretations_observation_idx
            ON observation_interpretations(software_id, observation_id);
        CREATE TABLE compact_episodes (
            episode_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            observation_id TEXT NOT NULL REFERENCES observations(observation_id),
            task_id TEXT NOT NULL,
            verification TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX compact_episodes_contract_idx
            ON compact_episodes(software_id, task_id);
        CREATE TABLE skill_candidates (
            candidate_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            status TEXT NOT NULL,
            revision INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX skill_candidates_status_idx
            ON skill_candidates(software_id, status);
        """,
    ),
    Migration(
        4,
        "locked task intent and acceptance contracts",
        """
        CREATE TABLE intent_locks (
            lock_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            lock_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX intent_locks_task_idx ON intent_locks(task_id, revision);
        CREATE TABLE acceptance_contracts (
            acceptance_contract_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            intent_lock_id TEXT NOT NULL REFERENCES intent_locks(lock_id),
            revision INTEGER NOT NULL,
            contract_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX acceptance_contracts_task_idx
            ON acceptance_contracts(task_id, revision);
        CREATE TABLE contract_amendments (
            amendment_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            previous_contract_id TEXT NOT NULL
                REFERENCES acceptance_contracts(acceptance_contract_id),
            new_contract_id TEXT NOT NULL
                REFERENCES acceptance_contracts(acceptance_contract_id),
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX contract_amendments_task_idx
            ON contract_amendments(task_id, created_at);
        """,
    ),
    Migration(
        5,
        "autonomous verification methods and plans",
        """
        CREATE TABLE verification_method_candidates (
            candidate_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            criterion_id TEXT NOT NULL,
            status TEXT NOT NULL,
            revision INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX verification_method_candidates_status_idx
            ON verification_method_candidates(software_id, status);
        CREATE TABLE verification_plans (
            plan_id TEXT PRIMARY KEY,
            acceptance_contract_id TEXT NOT NULL
                REFERENCES acceptance_contracts(acceptance_contract_id),
            software_id TEXT NOT NULL REFERENCES software(software_id),
            plan_hash TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX verification_plans_contract_idx
            ON verification_plans(acceptance_contract_id);
        """,
    ),
    Migration(
        6,
        "verification evidence graph, verdicts, and targeted repair",
        """
        CREATE TABLE verification_signals (
            signal_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            acceptance_contract_id TEXT NOT NULL
                REFERENCES acceptance_contracts(acceptance_contract_id),
            criterion_id TEXT NOT NULL,
            observation_id TEXT NOT NULL REFERENCES observations(observation_id),
            outcome TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX verification_signals_criterion_idx
            ON verification_signals(acceptance_contract_id, criterion_id);
        CREATE TABLE acceptance_verdicts (
            verdict_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            acceptance_contract_id TEXT NOT NULL
                REFERENCES acceptance_contracts(acceptance_contract_id),
            outcome TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE targeted_repair_packages (
            repair_package_id TEXT PRIMARY KEY,
            software_id TEXT NOT NULL REFERENCES software(software_id),
            acceptance_contract_id TEXT NOT NULL
                REFERENCES acceptance_contracts(acceptance_contract_id),
            criterion_id TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """,
    ),
)

SCHEMA_VERSION = MIGRATIONS[-1].version


def apply_migrations(connection: sqlite3.Connection, *, target_version: int | None = None) -> int:
    """Apply ordered migrations and return the resulting schema version."""
    connection.execute(
        """CREATE TABLE IF NOT EXISTS schema_metadata (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
           )"""
    )
    connection.commit()
    row = connection.execute(
        "SELECT value FROM schema_metadata WHERE key='schema_version'"
    ).fetchone()
    current = int(row[0]) if row is not None else 0
    requested = SCHEMA_VERSION if target_version is None else target_version
    if current > SCHEMA_VERSION:
        raise RuntimeError("database schema is newer than this software-memory build")
    if requested > SCHEMA_VERSION or requested < current:
        raise ValueError(f"invalid migration target: {requested}")

    for migration in MIGRATIONS:
        if current < migration.version <= requested:
            try:
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + migration.sql
                    + "\nINSERT INTO schema_metadata(key, value) "
                    + f"VALUES('schema_version', '{migration.version}') "
                    + "ON CONFLICT(key) DO UPDATE SET value=excluded.value;\nCOMMIT;"
                )
            except Exception:
                connection.rollback()
                raise
            current = migration.version
    return current
