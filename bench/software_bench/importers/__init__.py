from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from software_bench.core.models import ValidationError
from software_bench.importers.datasets.application import (
    APPLICATION_DATASET_SOURCES,
    import_application_dataset,
)
from software_bench.importers.datasets.database import (
    DEFAULT_DATABASE_IMAGE,
    import_spider2_dataset,
)
from software_bench.importers.datasets.live_service import import_sregym
from software_bench.importers.datasets.scientific import (
    SCIENTIFIC_SOURCES,
    import_scientific_dataset,
)
from software_bench.importers.datasets.workbook import (
    DEFAULT_WORKBOOK_IMAGE,
    import_spreadsheetbench_dataset,
)
from software_bench.importers.manifest import import_executable_manifest


@dataclass(frozen=True)
class DatasetImportRequest:
    source: str
    output: Path
    upstream: Path | None = None
    recipe: Path | None = None
    dataset: Path | None = None
    assets: Path | None = None
    selection: tuple[str, ...] = ()
    image: str | None = None
    port: int = 0


@dataclass(frozen=True)
class DatasetSource:
    id: str
    import_tasks: Callable[[DatasetImportRequest], tuple[Path, ...]]


def dataset_sources() -> Mapping[str, DatasetSource]:
    sources = {
        "spreadsheetbench": DatasetSource("spreadsheetbench", _import_workbook),
        "spider2-lite": DatasetSource(
            "spider2-lite", lambda request: _import_database(request, "lite")
        ),
        "spider2-snow": DatasetSource(
            "spider2-snow", lambda request: _import_database(request, "snow")
        ),
        "spider2-dbt": DatasetSource(
            "spider2-dbt", lambda request: _import_database(request, "dbt")
        ),
        "sregym": DatasetSource("sregym", _import_live_service),
        "executable-manifest": DatasetSource(
            "executable-manifest", _import_executable
        ),
    }
    sources.update(
        {
            source_id: DatasetSource(
                source_id,
                lambda request, selected=source_id: _import_scientific(
                    request, selected
                ),
            )
            for source_id in SCIENTIFIC_SOURCES
        }
    )
    sources.update(
        {
            source_id: DatasetSource(
                source_id,
                lambda request, selected=source_id: _import_application(
                    request, selected
                ),
            )
            for source_id in APPLICATION_DATASET_SOURCES
        }
    )
    return sources


def import_dataset(request: DatasetImportRequest) -> tuple[Path, ...]:
    source = dataset_sources().get(request.source)
    if source is None:
        raise ValidationError(f"unknown dataset source: {request.source}")
    return source.import_tasks(request)


def _import_workbook(request: DatasetImportRequest) -> tuple[Path, ...]:
    return import_spreadsheetbench_dataset(
        _required(request.dataset, "--dataset"),
        _required(request.assets, "--assets"),
        request.output,
        image=request.image or DEFAULT_WORKBOOK_IMAGE,
        case_ids=request.selection,
    )


def _import_database(
    request: DatasetImportRequest, variant: str
) -> tuple[Path, ...]:
    return import_spider2_dataset(
        _required(request.dataset, "--dataset"),
        _required(request.upstream, "--upstream"),
        request.output,
        variant=variant,
        image=request.image or DEFAULT_DATABASE_IMAGE,
        case_ids=request.selection,
    )


def _import_live_service(request: DatasetImportRequest) -> tuple[Path, ...]:
    return import_sregym(
        _required(request.upstream, "--upstream"),
        request.output,
        problem_ids=request.selection,
        port=request.port,
    )


def _import_executable(request: DatasetImportRequest) -> tuple[Path, ...]:
    return import_executable_manifest(
        _required(request.recipe, "--recipe"),
        request.output,
        task_ids=request.selection,
    )


def _import_scientific(
    request: DatasetImportRequest, source_id: str
) -> tuple[Path, ...]:
    return import_scientific_dataset(
        source_id,
        _required(request.upstream, "--upstream"),
        _required(request.recipe, "--recipe"),
        request.output,
        image=request.image,
        case_ids=request.selection,
    )


def _import_application(
    request: DatasetImportRequest, source_id: str
) -> tuple[Path, ...]:
    return import_application_dataset(
        source_id,
        _required(request.upstream, "--upstream"),
        _required(request.recipe, "--recipe"),
        request.output,
        image=request.image,
        task_ids=request.selection,
    )


def _required(value: Path | None, option: str) -> Path:
    if value is None:
        raise ValidationError(f"dataset source requires {option}")
    return value


__all__ = [
    "DatasetImportRequest",
    "DatasetSource",
    "dataset_sources",
    "import_dataset",
]

