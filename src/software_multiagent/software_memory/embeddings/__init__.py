"""Derived semantic index stored separately from canonical software memory."""

from software_multiagent.software_memory.embeddings.client import (
    EmbeddingClient,
    EmbeddingClientError,
    OpenAICompatibleEmbeddingClient,
)
from software_multiagent.software_memory.embeddings.index import (
    EmbeddingDocument,
    EmbeddingIndexReport,
    EmbeddingIndexer,
    EmbeddingSearchResult,
    LanceDBEmbeddingRetriever,
    LanceDBEmbeddingStore,
)

__all__ = [
    "EmbeddingClient",
    "EmbeddingClientError",
    "EmbeddingDocument",
    "EmbeddingIndexReport",
    "EmbeddingIndexer",
    "EmbeddingSearchResult",
    "LanceDBEmbeddingRetriever",
    "LanceDBEmbeddingStore",
    "OpenAICompatibleEmbeddingClient",
]
