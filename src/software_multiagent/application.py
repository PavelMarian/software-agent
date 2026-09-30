"""Primary application boundary for configuring and running the multi-agent team."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping

from software_multiagent.config import load_environment, target_model
from software_multiagent.core.contracts import TaskContext
from software_multiagent.software_memory.integrations.langgraph import LangGraphSharedMemory
from software_multiagent.software_memory.integrations.shared_agents import SharedSoftwareMemory
from software_multiagent.providers import create_chat_model
from software_multiagent.runtime.orchestration.langgraph import LangGraphMultiAgent
from software_multiagent.software_memory import MemoryRetriever, MemoryService, SQLiteMemoryStore
from software_multiagent.tools.workspace import WorkspaceToolset


@dataclass(frozen=True)
class SoftwareRunRequest:
    task_id: str
    objective: str
    workspace: Path
    target_software: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SoftwareMultiAgent:
    """The sole production composition root for model, team, and shared memory."""

    @classmethod
    def run_configured(
        cls,
        request: SoftwareRunRequest,
        tools: WorkspaceToolset,
        *,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        load_environment()
        default_provider, default_model = target_model()
        selected_provider = provider or default_provider
        selected_model = model_name or default_model
        if selected_provider not in {"openai", "openrouter"}:
            raise ValueError("set provider to openai or openrouter")
        if not selected_model:
            raise ValueError("set model or SOFTWARE_MULTIAGENT_MODEL")
        model = create_chat_model(provider=selected_provider, model=selected_model)
        memory_path = os.getenv("SOFTWARE_MEMORY_DATABASE", "").strip()
        vector_path = os.getenv("SOFTWARE_MEMORY_VECTOR_DATABASE", "").strip()
        if vector_path and not memory_path:
            raise ValueError(
                "SOFTWARE_MEMORY_DATABASE is required when "
                "SOFTWARE_MEMORY_VECTOR_DATABASE is configured"
            )
        if not memory_path:
            return cls.run(model, request, tools)
        database = Path(memory_path).expanduser().resolve()
        if not database.is_file():
            raise FileNotFoundError(f"configured software-memory database does not exist: {database}")
        software_id = os.getenv("SOFTWARE_MEMORY_SOFTWARE_ID", "").strip()
        if not software_id:
            raise ValueError("SOFTWARE_MEMORY_SOFTWARE_ID is required when memory is enabled")
        with ExitStack() as stack:
            store = stack.enter_context(SQLiteMemoryStore(database))
            if store.get_software(software_id) is None:
                raise ValueError(f"configured software-memory identity is absent: {software_id}")
            embedding_retriever = None
            if vector_path:
                vector_database = Path(vector_path).expanduser().resolve()
                if not vector_database.is_dir():
                    raise FileNotFoundError(
                        "configured software-memory vector database does not exist: "
                        f"{vector_database}"
                    )
                embedding_api_key = os.getenv(
                    "SOFTWARE_MEMORY_EMBEDDING_API_KEY", ""
                ).strip()
                embedding_model = os.getenv(
                    "SOFTWARE_MEMORY_EMBEDDING_MODEL", ""
                ).strip()
                embedding_base_url = os.getenv(
                    "SOFTWARE_MEMORY_EMBEDDING_BASE_URL", ""
                ).strip()
                embedding_provider = os.getenv(
                    "SOFTWARE_MEMORY_EMBEDDING_PROVIDER", "openai-compatible"
                ).strip()
                missing = [
                    name
                    for name, value in (
                        ("SOFTWARE_MEMORY_EMBEDDING_API_KEY", embedding_api_key),
                        ("SOFTWARE_MEMORY_EMBEDDING_MODEL", embedding_model),
                        ("SOFTWARE_MEMORY_EMBEDDING_BASE_URL", embedding_base_url),
                    )
                    if not value
                ]
                if missing:
                    raise ValueError(
                        "vector memory requires: " + ", ".join(missing)
                    )
                from software_multiagent.software_memory.embeddings import (
                    LanceDBEmbeddingRetriever,
                    LanceDBEmbeddingStore,
                    OpenAICompatibleEmbeddingClient,
                )

                embedding_client = OpenAICompatibleEmbeddingClient(
                    api_key=embedding_api_key,
                    model=embedding_model,
                    base_url=embedding_base_url,
                    provider=embedding_provider,
                )
                vector_store = stack.enter_context(
                    LanceDBEmbeddingStore(vector_database)
                )
                if vector_store.dimensions(embedding_client) < 1:
                    raise ValueError(
                        "the configured vector database has no index for the "
                        "selected embedding provider and model"
                    )
                embedding_retriever = LanceDBEmbeddingRetriever(
                    vector_store,
                    embedding_client,
                )
            metadata = dict(request.metadata)
            metadata["software_memory_id"] = software_id
            version = os.getenv("SOFTWARE_MEMORY_SOFTWARE_VERSION", "").strip()
            if version:
                metadata["software_version"] = version
            memory_request = SoftwareRunRequest(
                request.task_id,
                request.objective,
                request.workspace,
                request.target_software,
                metadata,
            )
            shared = SharedSoftwareMemory(
                MemoryRetriever(store, embedding_retriever=embedding_retriever),
                service=MemoryService(store),
            )
            result = cls.run(model, memory_request, tools, shared=shared)
            result["memory_retrieval"] = {
                "sqlite": True,
                "vector": embedding_retriever is not None,
                "vector_queries": (
                    embedding_retriever.query_count if embedding_retriever else 0
                ),
                "vector_hits": (
                    embedding_retriever.hit_count if embedding_retriever else 0
                ),
                "policy": "sqlite_first_vector_supplemental",
            }
            return result

    @staticmethod
    def run(
        model: Any,
        request: SoftwareRunRequest,
        tools: WorkspaceToolset,
        *,
        shared: SharedSoftwareMemory | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        bridge = None
        if shared is not None:
            task = TaskContext(
                task_id=request.task_id,
                objective=request.objective,
                workspace=request.workspace,
                target_software=request.target_software,
                metadata=request.metadata,
            )
            bridge = LangGraphSharedMemory(shared, task)
        return LangGraphMultiAgent(
            model,
            tools,
            shared_memory=bridge,
            **dict(runtime_options or {}),
        ).run(
            task_id=request.task_id,
            objective=request.objective,
            target_software=request.target_software,
        )


__all__ = ["SoftwareMultiAgent", "SoftwareRunRequest"]
