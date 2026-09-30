from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

try:
    import lancedb
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "LanceDB support requires: pip install 'software-multiagent[embeddings]'"
    ) from error

from software_multiagent.software_memory.embeddings.client import EmbeddingClient
from software_multiagent.software_memory.schema.models import Entity, EvidenceRecord, OperationContract, Workflow
from software_multiagent.software_memory.persistence.storage import SQLiteMemoryStore
from software_multiagent.software_memory.operations.retrieval import EmbeddingHit


EmbeddingItemType = Literal["evidence", "entity", "contract", "workflow"]
SUPPORTED_ITEM_TYPES: tuple[EmbeddingItemType, ...] = (
    "evidence", "entity", "contract", "workflow",
)


class EmbeddingDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    software_id: str
    item_type: EmbeddingItemType
    item_id: str
    text: str = Field(min_length=1)

    @property
    def content_hash(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


class EmbeddingSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    score: float
    software_id: str
    item_type: EmbeddingItemType
    item_id: str
    content_hash: str


class EmbeddingIndexReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    software_id: str
    provider: str
    model: str
    dimensions: int
    collection: str
    discovered: int
    embedded: int
    reused: int
    removed: int
    by_type: dict[str, int]
    ann_index_created: bool = False


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class LanceDBEmbeddingStore:
    """Separate, disposable LanceDB index for model-derived representations."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.database = lancedb.connect(self.path)

    def close(self) -> None:
        close = getattr(self.database, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "LanceDBEmbeddingStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def collection_name(client: EmbeddingClient) -> str:
        signature = _canonical({
            "provider": client.provider,
            "model": client.model,
            "requested_dimensions": client.dimensions,
        })
        return f"emb_{sha256(signature.encode('utf-8')).hexdigest()[:24]}"

    def _table_names(self) -> set[str]:
        if hasattr(self.database, "list_tables"):
            listing = self.database.list_tables()
            return set(listing.tables if hasattr(listing, "tables") else listing)
        return set(self.database.table_names())

    def _table(self, client: EmbeddingClient):
        name = self.collection_name(client)
        return self.database.open_table(name) if name in self._table_names() else None

    def existing_hashes(
        self,
        client: EmbeddingClient,
        software_id: str,
        item_types: tuple[EmbeddingItemType, ...],
    ) -> dict[tuple[str, str], str]:
        table = self._table(client)
        if table is None:
            return {}
        wanted = set(item_types)
        rows = table.to_arrow().select(
            ["software_id", "item_type", "item_id", "content_hash"]
        ).to_pylist()
        return {
            (str(row["item_type"]), str(row["item_id"])): str(row["content_hash"])
            for row in rows
            if row["software_id"] == software_id and row["item_type"] in wanted
        }

    def put_many(
        self,
        client: EmbeddingClient,
        documents_and_vectors: Iterable[tuple[EmbeddingDocument, tuple[float, ...]]],
    ) -> int:
        pairs = list(documents_and_vectors)
        if not pairs:
            return 0
        dimensions = {len(vector) for _, vector in pairs}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("embedding vectors have inconsistent dimensions")
        rows: list[dict[str, Any]] = [
            {
                "software_id": document.software_id,
                "item_type": document.item_type,
                "item_id": document.item_id,
                "content_hash": document.content_hash,
                "text": document.text,
                "provider": client.provider,
                "model": client.model,
                "dimensions": len(vector),
                "vector": list(vector),
            }
            for document, vector in pairs
        ]
        table = self._table(client)
        if table is None:
            self.database.create_table(self.collection_name(client), data=rows)
            return len(rows)
        for document, _vector in pairs:
            table.delete(" AND ".join((
                f"software_id = {_sql_literal(document.software_id)}",
                f"item_type = {_sql_literal(document.item_type)}",
                f"item_id = {_sql_literal(document.item_id)}",
            )))
        table.add(rows)
        return len(rows)

    def remove_stale(
        self,
        client: EmbeddingClient,
        software_id: str,
        item_types: tuple[EmbeddingItemType, ...],
        live_keys: set[tuple[str, str]],
    ) -> int:
        table = self._table(client)
        if table is None:
            return 0
        stale = set(self.existing_hashes(client, software_id, item_types)) - live_keys
        for item_type, item_id in stale:
            table.delete(" AND ".join((
                f"software_id = {_sql_literal(software_id)}",
                f"item_type = {_sql_literal(item_type)}",
                f"item_id = {_sql_literal(item_id)}",
            )))
        return len(stale)

    def dimensions(self, client: EmbeddingClient) -> int:
        table = self._table(client)
        if table is None:
            return 0
        rows = table.to_arrow().select(["dimensions"]).slice(0, 1).to_pylist()
        return int(rows[0]["dimensions"]) if rows else 0

    def create_ann_index(self, client: EmbeddingClient, *, minimum_rows: int = 256) -> bool:
        table = self._table(client)
        if table is None or table.count_rows() < minimum_rows:
            return False
        table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_FLAT",
            replace=True,
        )
        return True

    def search(
        self,
        *,
        client: EmbeddingClient,
        software_id: str,
        query_vector: tuple[float, ...],
        limit: int,
        item_types: tuple[EmbeddingItemType, ...] = SUPPORTED_ITEM_TYPES,
    ) -> tuple[EmbeddingSearchResult, ...]:
        if limit < 1:
            return ()
        table = self._table(client)
        if table is None:
            return ()
        types = ", ".join(_sql_literal(value) for value in item_types)
        predicate = (
            f"software_id = {_sql_literal(software_id)} AND item_type IN ({types})"
        )
        rows = (
            table.search(list(query_vector), vector_column_name="vector")
            .metric("cosine")
            .where(predicate, prefilter=True)
            .limit(limit)
            .select(["software_id", "item_type", "item_id", "content_hash"])
            .to_list()
        )
        return tuple(
            EmbeddingSearchResult(
                score=1.0 - float(row["_distance"]),
                software_id=str(row["software_id"]),
                item_type=str(row["item_type"]),
                item_id=str(row["item_id"]),
                content_hash=str(row["content_hash"]),
            )
            for row in rows
        )


def _trimmed(text: str, maximum: int) -> str:
    return " ".join(text.split())[:maximum]


def _evidence_text(item: EvidenceRecord) -> str:
    return "\n".join(value for value in (
        f"Evidence from {item.source_kind.value}",
        f"Source: {item.source_uri}",
        f"Location: {item.locator}" if item.locator else "",
        item.content,
    ) if value)


def _entity_text(item: Entity) -> str:
    return "\n".join(value for value in (
        f"{item.entity_type}: {item.name}", item.summary,
        f"Aliases: {', '.join(item.aliases)}" if item.aliases else "",
        f"Attributes: {_canonical(item.attributes)}" if item.attributes else "",
    ) if value)


def _contract_text(item: OperationContract) -> str:
    return "\n".join(value for value in (
        f"Operation: {item.name}",
        f"Interface: {item.interface}",
        f"Purpose: {item.purpose}" if item.purpose else "",
        f"Inputs: {', '.join(item.inputs)}" if item.inputs else "",
        f"Outputs: {', '.join(item.outputs)}" if item.outputs else "",
        "Parameters: "
        + _canonical([p.model_dump(mode="json", exclude_none=True) for p in item.parameters])
        if item.parameters else "",
        "Preconditions: "
        + _canonical([p.model_dump(mode="json") for p in item.preconditions])
        if item.preconditions else "",
        "Effects: " + _canonical([p.model_dump(mode="json") for p in item.effects])
        if item.effects else "",
        "Failures: "
        + _canonical([f.model_dump(mode="json") for f in item.failure_signatures])
        if item.failure_signatures else "",
    ) if value)


def _workflow_text(item: Workflow) -> str:
    return "\n".join((
        f"Workflow: {item.name}",
        f"Objective: {item.objective}",
        f"Steps: {_canonical([step.model_dump(mode='json') for step in item.steps])}",
    ))


@dataclass
class EmbeddingIndexer:
    memory_store: SQLiteMemoryStore
    index_store: LanceDBEmbeddingStore
    client: EmbeddingClient
    batch_size: int = 64
    max_input_characters: int = 24_000
    build_ann_index: bool = True
    ann_minimum_rows: int = 256

    def documents(
        self,
        software_id: str,
        item_types: tuple[EmbeddingItemType, ...] = SUPPORTED_ITEM_TYPES,
    ) -> tuple[EmbeddingDocument, ...]:
        if self.memory_store.get_software(software_id) is None:
            raise ValueError(f"unknown software_id: {software_id}")
        if not item_types or not set(item_types).issubset(SUPPORTED_ITEM_TYPES):
            raise ValueError("item_types must contain supported embedding item types")
        documents: list[EmbeddingDocument] = []
        collections = {
            "evidence": (
                (item, _evidence_text(item))
                for item in self.memory_store.list_evidence(software_id)
            ),
            "entity": (
                (item, _entity_text(item))
                for item in self.memory_store.list_entities(software_id)
            ),
            "contract": (
                (item, _contract_text(item))
                for item in self.memory_store.list_contracts(software_id)
            ),
            "workflow": (
                (item, _workflow_text(item))
                for item in self.memory_store.list_workflows(software_id)
            ),
        }
        id_fields = {
            "evidence": "evidence_id", "entity": "entity_id",
            "contract": "contract_id", "workflow": "workflow_id",
        }
        for item_type in item_types:
            for item, text in collections[item_type]:
                text = _trimmed(text, self.max_input_characters)
                if text:
                    documents.append(EmbeddingDocument(
                        software_id=software_id,
                        item_type=item_type,
                        item_id=getattr(item, id_fields[item_type]),
                        text=text,
                    ))
        return tuple(documents)

    def index(
        self,
        software_id: str,
        item_types: tuple[EmbeddingItemType, ...] = SUPPORTED_ITEM_TYPES,
    ) -> EmbeddingIndexReport:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        documents = self.documents(software_id, item_types)
        existing = self.index_store.existing_hashes(self.client, software_id, item_types)
        pending = [document for document in documents if existing.get(
            (document.item_type, document.item_id)
        ) != document.content_hash]
        dimensions: int | None = None
        for offset in range(0, len(pending), self.batch_size):
            batch = pending[offset:offset + self.batch_size]
            vectors = self.client.embed(tuple(document.text for document in batch))
            if len(vectors) != len(batch):
                raise ValueError("embedding client returned a different number of vectors")
            widths = {len(vector) for vector in vectors}
            if len(widths) != 1 or 0 in widths:
                raise ValueError("embedding client returned inconsistent dimensions")
            current_dimensions = next(iter(widths))
            if dimensions is not None and dimensions != current_dimensions:
                raise ValueError("embedding dimensions changed between batches")
            dimensions = current_dimensions
            self.index_store.put_many(self.client, zip(batch, vectors))
        dimensions = dimensions or self.index_store.dimensions(self.client)
        live_keys = {(document.item_type, document.item_id) for document in documents}
        removed = self.index_store.remove_stale(
            self.client, software_id, item_types, live_keys
        )
        ann_created = self.index_store.create_ann_index(
            self.client, minimum_rows=self.ann_minimum_rows
        ) if self.build_ann_index and (pending or removed) else False
        by_type = {item_type: 0 for item_type in item_types}
        for document in documents:
            by_type[document.item_type] += 1
        return EmbeddingIndexReport(
            software_id=software_id,
            provider=self.client.provider,
            model=self.client.model,
            dimensions=dimensions,
            collection=self.index_store.collection_name(self.client),
            discovered=len(documents),
            embedded=len(pending),
            reused=len(documents) - len(pending),
            removed=removed,
            by_type=by_type,
            ann_index_created=ann_created,
        )


@dataclass
class LanceDBEmbeddingRetriever:
    index_store: LanceDBEmbeddingStore
    client: EmbeddingClient
    query_count: int = 0
    hit_count: int = 0

    def search_memory(
        self, *, software_id: str, query: str, limit: int
    ) -> tuple[EmbeddingHit, ...]:
        self.query_count += 1
        vectors = self.client.embed((query,))
        if len(vectors) != 1:
            raise ValueError("embedding client returned an invalid query vector")
        results = self.index_store.search(
            client=self.client,
            software_id=software_id,
            query_vector=vectors[0],
            limit=limit,
            item_types=SUPPORTED_ITEM_TYPES,
        )
        self.hit_count += len(results)
        return tuple(
            EmbeddingHit(result.score, result.item_type, result.item_id)
            for result in results
        )
