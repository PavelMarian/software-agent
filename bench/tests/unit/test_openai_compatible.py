from types import SimpleNamespace
from typing import Any

import pytest

from software_bench.harness.models.openai_compatible import (
    ModelAdapterProtocolError,
    OpenAICompatibleModelAdapter,
)


class FakeCompletions:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes: Any) -> None:
        self.completions = FakeCompletions(*outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class RateLimited(RuntimeError):
    status_code = 429


def _response(
    *,
    text: str = "",
    arguments: str = '{"path": "README.md"}',
    with_tool: bool = True,
    with_usage: bool = True,
) -> Any:
    tool_calls = (
        [SimpleNamespace(function=SimpleNamespace(name="read", arguments=arguments))]
        if with_tool
        else []
    )
    usage = SimpleNamespace(prompt_tokens=13, completion_tokens=5) if with_usage else None
    return SimpleNamespace(
        id="request-1",
        model="provider/model-version",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=tool_calls),
                finish_reason="tool_calls" if with_tool else "stop",
            )
        ],
        usage=usage,
    )


def _adapter(client: FakeClient, **kwargs: Any) -> OpenAICompatibleModelAdapter:
    return OpenAICompatibleModelAdapter(
        adapter_name="openrouter",
        model="provider/model",
        base_url="https://openrouter.ai/api/v1",
        client=client,
        sleep=lambda _: None,
        **kwargs,
    )


def test_adapter_normalizes_tools_calls_and_per_request_usage() -> None:
    client = FakeClient(_response())
    adapter = _adapter(client)
    tools = [
        {
            "name": "read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]

    result = adapter.generate(
        [{"role": "user", "content": "inspect"}],
        "system",
        tools,
        role_id="solo",
        seed=7,
    )

    assert result.tool_calls == ({"name": "read", "args": {"path": "README.md"}},)
    assert result.input_tokens == 13
    assert result.output_tokens == 5
    assert result.measurement_complete is True
    assert result.done is False
    assert result.metadata["provider_request_id"] == "request-1"
    request = client.completions.requests[0]
    assert request["seed"] == 7
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["function"]["parameters"]["required"] == ["path"]
    assert request["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in request


def test_adapter_can_require_exactly_one_tool_call() -> None:
    client = FakeClient(_response())
    adapter = _adapter(client)
    tools = [{"name": "read", "description": "Read", "parameters": {"type": "object"}}]

    result = adapter.generate(
        [], "system", tools, role_id="solo", seed=0,
        tool_choice="required", parallel_tool_calls=False,
    )

    request = client.completions.requests[0]
    assert request["tool_choice"] == "required"
    assert request["parallel_tool_calls"] is False
    assert result.metadata["tool_choice"] == "required"
    assert result.metadata["parallel_tool_calls"] is False


def test_required_tool_choice_needs_tools() -> None:
    adapter = _adapter(FakeClient())
    with pytest.raises(ValueError, match="needs at least one"):
        adapter.generate([], "system", [], role_id="solo", seed=0,
                         tool_choice="required")


def test_retry_is_bounded_and_marks_usage_as_incomplete() -> None:
    client = FakeClient(RateLimited("slow down"), _response(text="TASK_COMPLETE", with_tool=False))
    delays: list[float] = []
    adapter = OpenAICompatibleModelAdapter(
        adapter_name="openrouter",
        model="provider/model",
        client=client,
        max_retries=1,
        sleep=delays.append,
    )

    result = adapter.generate([], "system", [], role_id="solo", seed=0)

    assert len(client.completions.requests) == 2
    assert delays == [1]
    assert result.done is True
    assert result.measurement_complete is False
    assert result.metadata["retry_count"] == 1


def test_invalid_tool_arguments_are_returned_as_recoverable_feedback() -> None:
    adapter = _adapter(FakeClient(_response(arguments="not-json")))

    result = adapter.generate([], "system", [], role_id="solo", seed=0)

    assert result.tool_calls == ()
    assert "invalid JSON" in result.metadata["recoverable_tool_errors"][0]


def test_missing_usage_is_explicitly_marked_incomplete() -> None:
    adapter = _adapter(FakeClient(_response(with_tool=False, with_usage=False)))

    result = adapter.generate([], "system", [], role_id="solo", seed=0)

    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.measurement_complete is False
