from __future__ import annotations

import argparse
import json
import os

from software_multiagent.software_memory.embeddings import (
    EmbeddingIndexer,
    LanceDBEmbeddingStore,
    OpenAICompatibleEmbeddingClient,
)
from software_multiagent.software_memory.embeddings.index import SUPPORTED_ITEM_TYPES
from software_multiagent.software_memory.persistence.storage import SQLiteMemoryStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a separate semantic index for software-memory"
    )
    parser.add_argument("memory_database", help="canonical software-memory SQLite database")
    parser.add_argument("embedding_database", help="separate LanceDB directory for derived vectors")
    parser.add_argument("software_id")
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "SOFTWARE_MEMORY_EMBEDDING_BASE_URL", "https://api.openai.com/v1"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SOFTWARE_MEMORY_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("SOFTWARE_MEMORY_EMBEDDING_PROVIDER", "openai-compatible"),
    )
    parser.add_argument("--api-key-env", default="SOFTWARE_MEMORY_EMBEDDING_API_KEY")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-input-characters", type=int, default=24_000)
    parser.add_argument(
        "--item-type",
        dest="item_types",
        action="append",
        choices=SUPPORTED_ITEM_TYPES,
        help="repeat to restrict indexing; defaults to all types",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key in environment variable {args.api_key_env}")
    client = OpenAICompatibleEmbeddingClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        provider=args.provider,
        dimensions=args.dimensions,
    )
    item_types = tuple(args.item_types or SUPPORTED_ITEM_TYPES)
    with SQLiteMemoryStore(args.memory_database) as memory_store:
        with LanceDBEmbeddingStore(args.embedding_database) as index_store:
            report = EmbeddingIndexer(
                memory_store=memory_store,
                index_store=index_store,
                client=client,
                batch_size=args.batch_size,
                max_input_characters=args.max_input_characters,
            ).index(args.software_id, item_types)  # type: ignore[arg-type]
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
