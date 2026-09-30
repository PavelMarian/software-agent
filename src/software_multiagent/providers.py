"""LangChain chat-model factory for OpenAI and OpenRouter."""
from __future__ import annotations

import os
from typing import Any

from software_multiagent.config import load_environment


def create_chat_model(
    *, provider: str, model: str, temperature: float | None = None, **options: Any
):
    """Create the LangChain ``ChatOpenAI`` integration for the selected endpoint."""

    load_environment()
    if provider not in {"openai", "openrouter"}:
        raise ValueError("provider must be openai or openrouter")
    if not model:
        raise ValueError("model must be non-empty")
    key_name = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
    key = os.environ.get(key_name)
    if not key:
        raise ValueError(f"required API key environment variable is not set: {key_name}")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise ImportError('Install provider support with: pip install -e ".[providers]"') from error
    configuration: dict[str, Any] = {
        "model": model,
        "api_key": key,
        "max_retries": 0,
        **options,
    }
    if temperature is not None:
        configuration["temperature"] = temperature
    if provider == "openrouter":
        configuration["base_url"] = "https://openrouter.ai/api/v1"
    return ChatOpenAI(**configuration)
