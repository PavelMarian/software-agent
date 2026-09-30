"""Compile evidence already stored in a Software Memory database."""

from __future__ import annotations

import argparse
from pathlib import Path

from software_multiagent.software_memory.extraction.pipeline import KnowledgeCompiler
from software_multiagent.software_memory.operations.service import MemoryService
from software_multiagent.software_memory.persistence.storage import SQLiteMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("software_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--report-detail-limit", type=int, default=12_000)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with SQLiteMemoryStore(arguments.database) as store:
        report = KnowledgeCompiler(
            MemoryService(store),
            arguments.output / "extraction-state.json",
            arguments.output / "human-review-queue.json",
            report_detail_limit=arguments.report_detail_limit,
        ).compile(software_id=arguments.software_id)
    (arguments.output / "compilation-report.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(report.model_dump_json())


if __name__ == "__main__":
    main()
