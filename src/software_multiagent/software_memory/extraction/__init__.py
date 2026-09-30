"""Evidence-to-knowledge compilation without agent-framework dependencies."""

from software_multiagent.software_memory.extraction.models import (
    ClaimAnchor,
    Classification,
    CompilationReport,
    EntityDraft,
    EvidenceBlock,
    EvidenceUnit,
    MaterialKind,
    OperationDraft,
    ReviewQueueItem,
    UnitExtraction,
)
from software_multiagent.software_memory.extraction.normalization import classify_unit, normalize_evidence
from software_multiagent.software_memory.extraction.pipeline import KnowledgeCompiler
from software_multiagent.software_memory.extraction.structured import StructuredExtractionClient

__all__ = [
    "ClaimAnchor",
    "Classification",
    "CompilationReport",
    "EntityDraft",
    "EvidenceBlock",
    "EvidenceUnit",
    "MaterialKind",
    "OperationDraft",
    "ReviewQueueItem",
    "UnitExtraction",
    "KnowledgeCompiler",
    "StructuredExtractionClient",
    "classify_unit",
    "normalize_evidence",
]
