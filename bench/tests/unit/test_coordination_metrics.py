from __future__ import annotations

from types import SimpleNamespace

from software_bench.metrics import summarize_trace


def _event(
    sequence: int,
    event_type: str,
    actor: str,
    payload: dict[str, object],
) -> SimpleNamespace:
    return SimpleNamespace(
        sequence=sequence,
        event_type=event_type,
        actor=actor,
        payload=payload,
        input_tokens=0,
        output_tokens=0,
        tool_calls=int(event_type == "tool_called"),
    )


def test_coordination_metrics_use_observed_handoffs_and_artifact_reuse() -> None:
    events = [
        _event(0, "tool_called", "planner", {"tool": "write", "args": {"path": "plan.md"}}),
        _event(
            1,
            "tool_result",
            "planner",
            {
                "tool": "write",
                "args": {"path": "plan.md"},
                "result": {"exit_code": 0},
            },
        ),
        _event(
            2,
            "message",
            "planner",
            {"message_id": "message-0", "recipient": "executor", "content": "Use plan.md"},
        ),
        _event(
            3,
            "messages_received",
            "executor",
            {"count": 1, "message_ids": ["message-0"], "senders": ["planner"]},
        ),
        _event(4, "tool_called", "executor", {"tool": "read", "args": {"path": "plan.md"}}),
        _event(
            5,
            "tool_result",
            "executor",
            {
                "tool": "read",
                "args": {"path": "plan.md"},
                "result": {"exit_code": 0},
            },
        ),
        _event(6, "tool_called", "verifier", {"tool": "read", "args": {"path": "plan.md"}}),
    ]

    coordination = summarize_trace(events)["coordination"]

    assert coordination["message_delivery_rate"] == 1.0
    assert coordination["unique_handoff_edge_count"] == 1
    assert coordination["handoff_followed_by_action_count"] == 1
    assert coordination["cross_role_artifact_reuse_count"] == 1
    assert coordination["cross_role_tool_transition_count"] == 2
    assert coordination["redundant_cross_role_call_count"] == 1
    assert coordination["role_coverage"] == 0.666667
