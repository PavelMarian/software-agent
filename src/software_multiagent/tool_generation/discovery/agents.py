from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from software_multiagent.tool_generation.contracts.errors import ToolGenerationError
from software_multiagent.tool_generation.contracts.models import ApplicationRecipe
from software_multiagent.tool_generation.workflow.telemetry import GenerationTelemetry


class ChatModel(Protocol):
    """Minimal chat-model surface required by tool-generation agents."""

    def invoke(self, messages: Sequence[Any]) -> AIMessage:
        """Generate one response for a sequence of chat messages."""


class RecipeAgent:
    """Use one chat model as both grounded explorer and bounded debugger."""

    def __init__(
        self,
        model: ChatModel,
        *,
        telemetry: GenerationTelemetry | None = None,
        max_format_repairs: int = 2,
    ) -> None:
        self.model = model
        self.telemetry = telemetry
        self.max_format_repairs = max_format_repairs

    def propose(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        """Propose a recipe using only evidence in an exploration manifest.

        Args:
            manifest: Deterministically collected repository evidence.

        Returns:
            A recipe-shaped mapping.
        """

        return self._invoke(
            "explorer",
            "You are an application-interface explorer. Return only one JSON recipe. "
            "Every tool source must begin with an exact path in the manifest. Prefer useful "
            "workflow-level operations. Use command for documented CLIs, or wrapper.kind "
            "python, http, or pipeline when an adapter is needed. Never invent an interface.",
            {"manifest": manifest},
        )

    def repair(
        self,
        recipe: Mapping[str, Any],
        validation: Mapping[str, Any],
        attempt: int,
    ) -> Mapping[str, Any] | None:
        """Revise a failed recipe from concrete validation diagnostics.

        Args:
            recipe: Previous recipe.
            validation: Deterministic validation report.
            attempt: One-based repair attempt number.

        Returns:
            A complete revised recipe, or ``None`` if the agent declines repair.
        """

        return self._invoke(
            "debugger",
            "You are a tool-integration debugger. Return only the complete corrected JSON "
            "recipe. Change the smallest possible surface justified by diagnostics. Do not "
            "weaken correctness expectations merely to make validation pass.",
            {"attempt": attempt, "recipe": recipe, "validation": validation},
        )

    def _invoke(
        self,
        role: str,
        instruction: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        messages: list[Any] = [
            SystemMessage(content=instruction),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ]
        last_error: Exception | None = None
        for attempt in range(self.max_format_repairs + 1):
            response = self.model.invoke(messages)
            if not isinstance(response, AIMessage):
                raise ToolGenerationError(
                    f"{role} returned {type(response).__name__}, expected AIMessage"
                )
            self._record_usage(role, response, attempt + 1)
            content = response.content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            try:
                value = _json_object(content)
                if not isinstance(value, Mapping):
                    raise ToolGenerationError(f"{role} response must contain a JSON object")
                ApplicationRecipe.from_dict(value)
                return value
            except (ToolGenerationError, TypeError, ValueError) as error:
                last_error = error
                if self.telemetry is not None:
                    self.telemetry.emit(
                        "agent_format_repair",
                        role=role,
                        attempt=attempt + 1,
                        error=str(error),
                    )
                if attempt >= self.max_format_repairs:
                    break
                messages.extend(
                    [
                        response,
                        HumanMessage(
                            content=(
                                "The response failed the deterministic recipe gate: "
                                f"{error}. Return the complete corrected JSON object only."
                            )
                        ),
                    ]
                )
        raise ToolGenerationError(
            f"{role} exceeded the recipe-format repair limit: {last_error}"
        )

    def _record_usage(self, role: str, response: AIMessage, attempt: int) -> None:
        """Record one model response without storing prompt or generated content.

        Args:
            role: Agent role.
            response: Model response carrying usage metadata.
            attempt: One-based format attempt.
        """

        if self.telemetry is None:
            return
        usage = dict(response.usage_metadata or {})
        usage["attempt"] = attempt
        usage.update(
            {
                key: value
                for key, value in response.response_metadata.items()
                if key in {"model", "provider", "transport", "usage_complete"}
            }
        )
        self.telemetry.add_model_usage(role, usage)


def _json_object(content: str) -> Any:
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as error:
        raise ToolGenerationError(f"agent returned invalid recipe JSON: {error}") from error
