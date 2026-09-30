from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoryErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SoftwareMemoryError(Exception):
    code = "software_memory_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def as_info(self) -> MemoryErrorInfo:
        return MemoryErrorInfo(code=self.code, message=self.message, details=self.details)


class KnowledgeNotFoundError(SoftwareMemoryError):
    code = "knowledge_not_found"


class IntegrityViolationError(SoftwareMemoryError):
    code = "integrity_violation"


class TrustPolicyError(SoftwareMemoryError):
    code = "trust_policy_violation"


class StorageConflictError(SoftwareMemoryError):
    code = "storage_conflict"


class StorageInitializationError(SoftwareMemoryError):
    code = "storage_initialization_error"


class FTS5UnavailableError(SoftwareMemoryError):
    code = "fts5_unavailable"


class SnapshotFormatError(SoftwareMemoryError):
    code = "snapshot_format_error"


class TransactionAbortedError(SoftwareMemoryError):
    code = "transaction_aborted"
