from __future__ import annotations

from software_multiagent.software_memory import (
    EmbeddingHit,
    KnowledgeRequest,
    MemoryRetriever,
    MemoryService,
    OperationContract,
    SoftwareIdentity,
    SQLiteMemoryStore,
)


class FixedEmbeddingRetriever:
    def __init__(self, hits: tuple[EmbeddingHit, ...]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search_memory(
        self, *, software_id: str, query: str, limit: int
    ) -> tuple[EmbeddingHit, ...]:
        del software_id
        self.queries.append(query)
        return self.hits[:limit]


def test_vector_results_supplement_without_replacing_sqlite_results(tmp_path) -> None:
    software = SoftwareIdentity.create("Test Software", "1")
    lexical = OperationContract(
        contract_id="contract_lexical",
        software_id=software.software_id,
        name="primary command",
        interface="cli",
        purpose="Found directly by the canonical SQLite lookup.",
    )
    semantic = OperationContract(
        contract_id="contract_semantic",
        software_id=software.software_id,
        name="unrelated wording",
        interface="api",
        purpose="Found only through semantic similarity.",
    )
    semantic_variant = OperationContract(
        contract_id="contract_semantic_variant",
        software_id=software.software_id,
        name=lexical.name,
        interface=lexical.interface,
        purpose="A semantic-only variant must not replace the SQLite variant.",
    )
    embedding = FixedEmbeddingRetriever(
        (
            EmbeddingHit(score=10_000.0, item_type="contract", item_id=lexical.contract_id),
            EmbeddingHit(
                score=9_999.5,
                item_type="contract",
                item_id=semantic_variant.contract_id,
            ),
            EmbeddingHit(score=9_999.0, item_type="contract", item_id=semantic.contract_id),
        )
    )

    with SQLiteMemoryStore(tmp_path / "memory.sqlite") as store:
        service = MemoryService(store)
        service.register_software(software)
        service.record_contract(lexical)
        service.record_contract(semantic)
        service.record_contract(semantic_variant)
        retriever = MemoryRetriever(store, embedding_retriever=embedding)

        one_item = retriever.retrieve(
            KnowledgeRequest(
                software_id=software.software_id,
                query=lexical.name,
                max_items=1,
                token_budget=2_000,
            )
        )
        two_items = retriever.retrieve(
            KnowledgeRequest(
                software_id=software.software_id,
                query=lexical.name,
                max_items=3,
                token_budget=2_000,
            )
        )

    assert [item.item_id for item in one_item.items] == [lexical.contract_id]
    assert [item.item_id for item in two_items.items] == [
        lexical.contract_id,
        semantic.contract_id,
    ]
    assert len({item.item_id for item in two_items.items}) == 2
    assert embedding.queries == [lexical.name, lexical.name]
    assert any(
        "supplemental_contracts=1" in entry
        for entry in two_items.retrieval_trace
    )
