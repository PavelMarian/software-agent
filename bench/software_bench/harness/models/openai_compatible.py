from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from typing import Any, Mapping

from software_bench.harness.contracts import ModelResponse


class ModelAdapterError(RuntimeError):
    """Raised when a provider call cannot produce a benchmark response."""


class ModelAdapterProtocolError(ModelAdapterError):
    """Raised when a provider returns a malformed response or tool call."""


class OpenAICompatibleModelAdapter:
    """One-turn adapter for OpenAI-compatible Chat Completions endpoints."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float | None = None,
        max_output_tokens: int = 8192,
        max_retries: int = 4,
        timeout_seconds: float = 300.0,
        adapter_name: str = "openai-compatible",
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be a non-empty string")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.adapter_id = f"{adapter_name}:{model}"
        self.metadata = {
            "adapter_name": adapter_name,
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "max_retries": max_retries,
            "timeout_seconds": timeout_seconds,
        }
        self._sleep = sleep
        self._client = client or _create_client(api_key_env, base_url, timeout_seconds)

    def generate(
        self,
        messages: list[Mapping[str, Any]],
        system_prompt: str,
        tools: list[Mapping[str, Any]],
        *,
        role_id: str,
        seed: int,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = True,
    ) -> ModelResponse:
        if tool_choice not in {"auto", "required", "none"}:
            raise ValueError("tool_choice must be auto, required, or none")
        if tool_choice == "required" and not tools:
            raise ValueError("tool_choice=required needs at least one declared tool")
        request: dict[str, Any] = {
            "model": self.model,
            "messages": _provider_messages(messages, system_prompt),
            "max_tokens": self.max_output_tokens,
            "seed": seed,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if tools:
            request["tools"] = [_provider_tool(tool) for tool in tools]
            request["tool_choice"] = tool_choice
            if not parallel_tool_calls:
                request["parallel_tool_calls"] = False

        response, retry_count = self._call_with_retry(request)
        choice = _first(_get(response, "choices"))
        if choice is None:
            raise ModelAdapterProtocolError("provider response has no choices")
        message = _get(choice, "message")
        if message is None:
            raise ModelAdapterProtocolError("provider choice has no message")

        text = _get(message, "content", "") or ""
        if not isinstance(text, str):
            raise ModelAdapterProtocolError("assistant content must be a string")
        tool_calls_list: list[Mapping[str, Any]] = []
        recoverable_tool_errors: list[str] = []
        for item in _get(message, "tool_calls") or ():
            try:
                tool_calls_list.append(_parse_tool_call(item))
            except ModelAdapterProtocolError as error:
                recoverable_tool_errors.append(str(error))
        tool_calls = tuple(tool_calls_list)
        input_tokens, output_tokens, has_usage = _usage(response)
        finish_reason = _get(choice, "finish_reason")
        metadata = {
            "provider_model": _get(response, "model", self.model),
            "provider_request_id": _get(response, "id"),
            "finish_reason": finish_reason,
            "retry_count": retry_count,
            "role_id": role_id,
            "recoverable_tool_errors": tuple(recoverable_tool_errors),
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            done=_signals_completion(text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            measurement_complete=has_usage and retry_count == 0,
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def _call_with_retry(self, request: Mapping[str, Any]) -> tuple[Any, int]:
        retry_count = 0
        while True:
            try:
                return self._client.chat.completions.create(**dict(request)), retry_count
            except Exception as error:  # provider SDK is an optional integration boundary
                if retry_count >= self.max_retries or not _is_transient(error):
                    raise ModelAdapterError(
                        f"provider request failed after {retry_count + 1} attempt(s): "
                        f"{type(error).__name__}: {error}"
                    ) from error
                self._sleep(min(2**retry_count, 8))
                retry_count += 1


def _create_client(api_key_env: str, base_url: str | None, timeout_seconds: float) -> Any:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"required API key environment variable is not set: {api_key_env}")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImportError(
            "OpenAI-compatible adapters require the optional dependency: "
            'pip install -e ".[openrouter]"'
        ) from error
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout_seconds,
        "max_retries": 0,
    }
    if base_url is not None:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _provider_messages(
    messages: list[Mapping[str, Any]], system_prompt: str
) -> list[dict[str, str]]:
    result = [{"role": "system", "content": system_prompt}]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        content_text = (
            content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        )
        if role == "tool":
            tool_name = message.get("name", "tool")
            result.append(
                {"role": "user", "content": f"Result from {tool_name}:\n{content_text}"}
            )
        elif role in {"user", "assistant"}:
            result.append({"role": role, "content": content_text})
        else:
            raise ValueError(f"unsupported conversation role: {role!r}")
    return result


def _provider_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    name = tool.get("name")
    description = tool.get("description")
    parameters = tool.get("parameters")
    if not isinstance(name, str) or not name:
        raise ValueError("tool declaration requires a non-empty name")
    if not isinstance(description, str) or not isinstance(parameters, Mapping):
        raise ValueError(f"tool {name!r} requires description and parameters")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": dict(parameters),
        },
    }


def _parse_tool_call(value: Any) -> Mapping[str, Any]:
    function = _get(value, "function")
    if function is None:
        raise ModelAdapterProtocolError("tool call has no function")
    name = _get(function, "name")
    raw_arguments = _get(function, "arguments", "{}")
    if not isinstance(name, str) or not name:
        raise ModelAdapterProtocolError("tool call function has no name")
    if isinstance(raw_arguments, Mapping):
        arguments = dict(raw_arguments)
    elif isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as error:
            raise ModelAdapterProtocolError(
                f"tool call {name!r} returned invalid JSON arguments"
            ) from error
        if not isinstance(parsed, Mapping):
            raise ModelAdapterProtocolError(f"tool call {name!r} arguments must be an object")
        arguments = dict(parsed)
    else:
        raise ModelAdapterProtocolError(f"tool call {name!r} arguments have unsupported type")
    return {"name": name, "args": arguments}


def _usage(response: Any) -> tuple[int, int, bool]:
    usage = _get(response, "usage")
    input_tokens = _get(usage, "prompt_tokens") if usage is not None else None
    output_tokens = _get(usage, "completion_tokens") if usage is not None else None
    complete = _non_negative_int(input_tokens) and _non_negative_int(output_tokens)
    return (
        input_tokens if _non_negative_int(input_tokens) else 0,
        output_tokens if _non_negative_int(output_tokens) else 0,
        complete,
    )


def _signals_completion(text: str) -> bool:
    return re.search(r"\b(?:TASK_COMPLETE|DONE)\b", text, flags=re.IGNORECASE) is not None


def _is_transient(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 425, 429} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return True
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _first(value: Any) -> Any | None:
    return value[0] if isinstance(value, (list, tuple)) and value else None


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
