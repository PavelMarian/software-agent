from __future__ import annotations

from dataclasses import asdict, dataclass
from time import monotonic

from software_bench.core.models import Budget


class BudgetExceeded(RuntimeError):
    """Raised before a normalized resource event would exceed a run budget."""

    abort_agent_runtime = True


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    wall_time_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["total_tokens"] = self.total_tokens
        return result


class BudgetLedger:
    """Accounts usage once for the whole team, never per role."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.usage = Usage()
        self._started_at = monotonic()

    def consume(
        self,
        input_tokens: int,
        output_tokens: int,
        tool_calls: int,
        *,
        account_overrun: bool = False,
        enforce: bool = True,
    ) -> None:
        if min(input_tokens, output_tokens, tool_calls) < 0:
            raise ValueError("usage deltas cannot be negative")
        self.usage.wall_time_seconds = monotonic() - self._started_at
        next_tokens = self.usage.total_tokens + input_tokens + output_tokens
        next_calls = self.usage.tool_calls + tool_calls
        error: str | None = None
        if self.usage.wall_time_seconds > self.budget.wall_time_seconds:
            error = "team wall-time budget exceeded"
        elif next_tokens > self.budget.token_limit:
            error = "team token budget exceeded"
        elif next_calls > self.budget.tool_call_limit:
            error = "team tool-call budget exceeded"
        if error is None or account_overrun or not enforce:
            self.usage.input_tokens += input_tokens
            self.usage.output_tokens += output_tokens
            self.usage.tool_calls = next_calls
        if error is not None and enforce:
            raise BudgetExceeded(error)

    def refresh_wall_time(self, *, enforce: bool = True) -> None:
        self.usage.wall_time_seconds = monotonic() - self._started_at
        if enforce and self.usage.wall_time_seconds > self.budget.wall_time_seconds:
            raise BudgetExceeded("team wall-time budget exceeded")

    def remaining_wall_time(self) -> float:
        self.usage.wall_time_seconds = monotonic() - self._started_at
        return max(0.0, self.budget.wall_time_seconds - self.usage.wall_time_seconds)
