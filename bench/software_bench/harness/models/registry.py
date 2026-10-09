from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from software_bench.harness.contracts import AgentRuntimeAdapter, FrameworkAdapter, ModelAdapter
from software_bench.harness.models.mock import MockModelAdapter
from software_bench.harness.models.openai_compatible import OpenAICompatibleModelAdapter


@dataclass(frozen=True)
class ModelAdapterSettings:
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    max_output_tokens: int = 8192
    max_retries: int = 4
    timeout_seconds: float = 300.0


def load_model_adapter(
    adapter_id: str, settings: ModelAdapterSettings | None = None
) -> ModelAdapter:
    settings = settings or ModelAdapterSettings()
    if adapter_id == "mock":
        return MockModelAdapter()
    if adapter_id == "openrouter":
        return _openai_compatible(
            "openrouter",
            settings,
            default_base_url="https://openrouter.ai/api/v1",
            default_api_key_env="OPENROUTER_API_KEY",
        )
    if adapter_id == "openai-compatible":
        return _openai_compatible(
            "openai-compatible",
            settings,
            default_base_url=None,
            default_api_key_env="OPENAI_API_KEY",
        )
    return _load("software_bench.model_adapters", adapter_id, "adapter_id")


def load_framework_adapter(adapter_id: str) -> FrameworkAdapter:
    return _load("software_bench.framework_adapters", adapter_id, "adapter_id")


def load_agent_runtime(runtime_id: str) -> AgentRuntimeAdapter:
    return _load("software_bench.agent_runtimes", runtime_id, "runtime_id")


def _openai_compatible(
    adapter_name: str,
    settings: ModelAdapterSettings,
    *,
    default_base_url: str | None,
    default_api_key_env: str,
) -> ModelAdapter:
    if settings.model is None:
        raise ValueError(f"--model is required for the {adapter_name} adapter")
    return OpenAICompatibleModelAdapter(
        adapter_name=adapter_name,
        model=settings.model,
        base_url=settings.base_url or default_base_url,
        api_key_env=settings.api_key_env or default_api_key_env,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        max_retries=settings.max_retries,
        timeout_seconds=settings.timeout_seconds,
    )


def _load(group: str, component_id: str, identity_field: str) -> Any:
    matches = entry_points(group=group, name=component_id)
    if not matches:
        raise LookupError(
            f"component {component_id!r} is not installed in entry-point group {group!r}"
        )
    component = matches[0].load()()
    if getattr(component, identity_field, None) != component_id:
        raise TypeError(
            f"entry point returned an object with a mismatched {identity_field}"
        )
    return component
