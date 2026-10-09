from __future__ import annotations

from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any, Mapping

from software_bench.harness.contracts import RunRequest
from software_bench.harness.execution.budget import BudgetExceeded, BudgetLedger

@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    elapsed_seconds: float
    event_type: str
    actor: str
    payload: Mapping[str, Any]
    input_tokens: int
    output_tokens: int
    tool_calls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Observer:
    def __init__(self, request: RunRequest, ledger: BudgetLedger) -> None:
        self.request = request
        self.ledger = ledger
        self.events: list[TraceEvent] = []
        self._started_at = monotonic()
        self._actors = {role.id for role in request.mode.roles} | {"harness"}
        self.enforce_budget = True

    def record(
        self,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
    ) -> None:
        if actor not in self._actors:
            raise ValueError(f"unknown trace actor: {actor}")
        try:
            self.ledger.consume(
                input_tokens,
                output_tokens,
                tool_calls,
                account_overrun=bool(input_tokens or output_tokens),
                enforce=self.enforce_budget,
            )
        except BudgetExceeded:
            self._append(
                event_type,
                actor,
                payload,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
            )
            raise
        self._append(
            event_type,
            actor,
            payload,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
        )

    def record_final(self, status: str, stop_reason: str) -> None:
        self._append("run_finished", "harness", {"status": status, "stop_reason": stop_reason})

    def remaining_wall_time(self) -> float:
        if not self.enforce_budget:
            return float("inf")
        return self.ledger.remaining_wall_time()

    def _append(
        self,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
    ) -> None:
        self.events.append(
            TraceEvent(
                sequence=len(self.events),
                elapsed_seconds=round(monotonic() - self._started_at, 6),
                event_type=event_type,
                actor=actor,
                payload=dict(payload or {}),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
            )
        )

