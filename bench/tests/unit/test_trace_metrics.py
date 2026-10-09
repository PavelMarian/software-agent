from types import SimpleNamespace

from software_bench.metrics import summarize_trace


def _event(event_type, tool=None, *, exit_code=0):
    payload = {}
    if tool is not None:
        payload = {"tool": tool}
        if event_type == "tool_result":
            payload["result"] = {"exit_code": exit_code}
    return SimpleNamespace(
        actor="solver",
        event_type=event_type,
        payload=payload,
        input_tokens=0,
        output_tokens=0,
        tool_calls=int(event_type == "tool_called"),
    )


def test_trace_summary_measures_tool_diversity_and_failures() -> None:
    events = [
        _event("tool_called", "openmc_validate_model"),
        _event("tool_result", "openmc_validate_model"),
        _event("tool_called", "openmc_run_transport"),
        _event("tool_result", "openmc_run_transport", exit_code=1),
        _event("tool_called", "openmc_run_transport"),
        _event("tool_result", "openmc_run_transport"),
    ]

    summary = summarize_trace(
        events,
        ["openmc_validate_model", "openmc_run_transport", "openmc_extract_tallies"],
    )

    tools = summary["tools"]
    assert tools["call_count"] == 3
    assert tools["unique_tool_count"] == 2
    assert tools["available_tool_count"] == 3
    assert tools["catalog_coverage"] == 0.666667
    assert tools["category_count"] == 2
    assert tools["repeated_call_count"] == 1
    assert tools["failed_call_count"] == 1
