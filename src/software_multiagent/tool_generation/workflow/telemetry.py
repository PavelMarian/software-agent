from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


@dataclass
class GenerationTelemetry:
    """Collect detailed, serializable telemetry for one generation run."""

    started_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def emit(self, event: str, **payload: Any) -> None:
        """Record a timestamped event and increment its counter.

        Args:
            event: Stable event type.
            **payload: JSON-serializable event details.
        """

        self.events.append({"event": event, "at_seconds": time.time() - self.started_at, **payload})
        self.counters[event] = self.counters.get(event, 0) + 1

    @contextmanager
    def stage(self, name: str, **payload: Any) -> Iterator[None]:
        """Measure a named pipeline stage.

        Args:
            name: Stage identifier.
            **payload: Context copied to start and finish events.

        Yields:
            Control to the measured stage.
        """

        started = time.perf_counter()
        self.emit("stage_started", stage=name, **payload)
        try:
            yield
        except Exception as error:
            self.emit(
                "stage_finished",
                stage=name,
                status="failed",
                duration_seconds=time.perf_counter() - started,
                error_type=type(error).__name__,
                error=str(error),
                **payload,
            )
            raise
        else:
            self.emit(
                "stage_finished",
                stage=name,
                status="passed",
                duration_seconds=time.perf_counter() - started,
                **payload,
            )

    def add_model_usage(self, role: str, metadata: Mapping[str, Any]) -> None:
        """Record provider/model/token information returned by an agent.

        Args:
            role: Agent role, such as ``explorer`` or ``debugger``.
            metadata: Provider-specific usage metadata.
        """

        self.emit("model_usage", role=role, **dict(metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON representation of the collected telemetry."""

        duration = time.time() - self.started_at
        stage_durations: dict[str, float] = {}
        for event in self.events:
            if event["event"] == "stage_finished":
                stage = str(event["stage"])
                stage_durations[stage] = stage_durations.get(stage, 0.0) + float(
                    event.get("duration_seconds", 0.0)
                )
        return {
            "duration_seconds": duration,
            "counters": dict(sorted(self.counters.items())),
            "stage_durations_seconds": stage_durations,
            "events": list(self.events),
        }
