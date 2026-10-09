from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable, Sequence


def summarize_trace(
    events: Iterable[Any], available_tools: Sequence[str] = ()
) -> dict[str, Any]:
    """Aggregate resource, tool-use, communication, and coordination metrics.

    Args:
        events: Ordered trace events exposing actor, type, payload, and usage
            fields.
        available_tools: Tool names exposed to the run for catalog-coverage
            measurement.

    Returns:
        A JSON-serializable summary grouped by communication, coordination,
        role, and tool usage.
    """
    per_role: dict[str, dict[str, int]] = defaultdict(_empty_role_summary)
    message_count = 0
    received_message_count = 0
    sent_message_ids: set[str] = set()
    received_message_ids: set[str] = set()
    receipt_batches = 0
    handoff_action_count = 0
    pending_receipts: dict[str, int] = defaultdict(int)
    communication_edges: set[tuple[str, str]] = set()
    coordinating_roles: set[str] = set()
    tool_calls: list[str] = []
    previous_tool_actor: str | None = None
    cross_role_tool_transitions = 0
    call_actors: dict[tuple[str, str], set[str]] = defaultdict(set)
    redundant_cross_role_calls = 0
    last_writer: dict[str, str] = {}
    cross_role_artifact_reuse = 0
    category_calls: dict[str, int] = defaultdict(int)
    failed_calls = 0
    for event in events:
        if event.actor == "harness":
            continue
        role = per_role[event.actor]
        role["events"] += 1
        role["input_tokens"] += event.input_tokens
        role["output_tokens"] += event.output_tokens
        role["tool_calls"] += event.tool_calls
        tool_name = event.payload.get("tool") if event.event_type == "tool_result" else None
        if tool_name == "read":
            role["file_reads"] += 1
        elif tool_name == "write":
            role["file_writes"] += 1
        elif event.event_type == "message":
            role["messages_sent"] += 1
            message_count += 1
            message_id = event.payload.get("message_id")
            sent_message_ids.add(
                message_id
                if isinstance(message_id, str) and message_id
                else f"trace-message-{event.sequence}"
            )
            recipient = event.payload.get("recipient")
            if isinstance(recipient, str) and recipient:
                communication_edges.add((event.actor, recipient))
                coordinating_roles.update((event.actor, recipient))
        elif event.event_type == "messages_received":
            count = event.payload.get("count", 0)
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                role["messages_received"] += count
                received_message_count += count
                message_ids = event.payload.get("message_ids", [])
                if isinstance(message_ids, list):
                    received_message_ids.update(
                        item for item in message_ids if isinstance(item, str) and item
                    )
                receipt_batches += 1
                pending_receipts[event.actor] += 1
                coordinating_roles.add(event.actor)
        if event.event_type == "tool_called":
            called = event.payload.get("tool")
            if isinstance(called, str) and called:
                tool_calls.append(called)
                category_calls[_tool_category(called)] += 1
                if previous_tool_actor is not None and previous_tool_actor != event.actor:
                    cross_role_tool_transitions += 1
                previous_tool_actor = event.actor
                if (
                    called not in {"message", "finish_phase"}
                    and pending_receipts[event.actor]
                ):
                    handoff_action_count += pending_receipts.pop(event.actor)
                args = event.payload.get("args", {})
                if isinstance(args, dict):
                    key = (called, json.dumps(args, sort_keys=True, default=str))
                    if call_actors[key] and event.actor not in call_actors[key]:
                        redundant_cross_role_calls += 1
                    call_actors[key].add(event.actor)
        elif event.event_type == "tool_result":
            result = event.payload.get("result", {})
            if isinstance(result, dict) and result.get("exit_code", 0) != 0:
                failed_calls += 1
            elif isinstance(result, dict):
                cross_role_artifact_reuse += _artifact_reuse(
                    event.actor,
                    event.payload,
                    last_writer,
                )
    unique_tools = sorted(set(tool_calls))
    catalog = sorted(set(available_tools))
    repeated = len(tool_calls) - len(unique_tools)
    active_role_count = len(per_role)
    return {
        "communication": {"message_count": message_count},
        "coordination": {
            "messages_sent": message_count,
            "messages_received": received_message_count,
            "unique_messages_delivered": len(
                sent_message_ids & received_message_ids
            ),
            "message_delivery_rate": (
                round(
                    len(sent_message_ids & received_message_ids)
                    / len(sent_message_ids),
                    6,
                )
                if sent_message_ids
                else None
            ),
            "unique_handoff_edge_count": len(communication_edges),
            "handoff_edges": [
                {"from": sender, "to": recipient}
                for sender, recipient in sorted(communication_edges)
            ],
            "participating_role_count": len(coordinating_roles),
            "role_coverage": (
                round(len(coordinating_roles) / active_role_count, 6)
                if active_role_count > 1
                else None
            ),
            "receipt_batch_count": receipt_batches,
            "handoff_followed_by_action_count": handoff_action_count,
            "handoff_action_rate": (
                round(handoff_action_count / receipt_batches, 6)
                if receipt_batches
                else None
            ),
            "cross_role_tool_transition_count": cross_role_tool_transitions,
            "cross_role_artifact_reuse_count": cross_role_artifact_reuse,
            "redundant_cross_role_call_count": redundant_cross_role_calls,
        },
        "active_role_count": active_role_count,
        "per_role": {key: value for key, value in sorted(per_role.items())},
        "tools": {
            "call_count": len(tool_calls),
            "unique_tool_count": len(unique_tools),
            "unique_tools": unique_tools,
            "available_tool_count": len(catalog),
            "catalog_coverage": (
                round(len(unique_tools) / len(catalog), 6) if catalog else None
            ),
            "category_count": len(category_calls),
            "category_calls": dict(sorted(category_calls.items())),
            "repeated_call_count": repeated,
            "repeated_call_fraction": (
                round(repeated / len(tool_calls), 6) if tool_calls else 0.0
            ),
            "failed_call_count": failed_calls,
        },
    }


def _tool_category(name: str) -> str:
    """Map a tool name to a coarse behavioral category.

    Args:
        name: Tool name recorded in the trace.

    Returns:
        A stable category label used by aggregate tool metrics.
    """
    if name in {"read", "list"}:
        return "workspace_inspection"
    if name == "write":
        return "workspace_mutation"
    if name == "run":
        return "general_execution"
    if name == "message":
        return "coordination"
    if name == "finish_phase":
        return "control"
    lowered = name.lower()
    inspection_terms = (
        "check", "describe", "inspect", "list", "report", "validate", "query",
        "read", "extract", "compare",
    )
    if any(term in lowered for term in inspection_terms):
        return "application_inspection"
    return "application_compute_or_mutation"


def _empty_role_summary() -> dict[str, int]:
    """Create a zero-valued accumulator for one role.

    Returns:
        Mutable counters for events, usage, file access, and messaging.
    """
    return {
        "events": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": 0,
        "file_reads": 0,
        "file_writes": 0,
        "messages_sent": 0,
        "messages_received": 0,
    }


def _artifact_reuse(
    actor: str,
    payload: dict[str, Any],
    last_writer: dict[str, str],
) -> int:
    """Track writes and count reads of artifacts produced by another role.

    Args:
        actor: Role responsible for the current tool result.
        payload: Tool-result payload containing the tool name and arguments.
        last_writer: Mutable mapping from artifact path to its latest writer.

    Returns:
        One when the event reads another role's artifact, otherwise zero.
    """
    tool = payload.get("tool")
    args = payload.get("args", {})
    if not isinstance(args, dict):
        return 0
    if tool == "read":
        path = args.get("path")
        return int(
            isinstance(path, str)
            and path in last_writer
            and last_writer[path] != actor
        )
    if tool == "write":
        path = args.get("path")
        if isinstance(path, str):
            last_writer[path] = actor
    elif tool == "write_files":
        files = args.get("files", {})
        if isinstance(files, dict):
            for path in files:
                if isinstance(path, str):
                    last_writer[path] = actor
    return 0
