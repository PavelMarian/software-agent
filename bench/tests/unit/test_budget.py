from types import SimpleNamespace

import pytest

from software_bench.core.models import Budget
from software_bench.harness.execution.budget import BudgetExceeded, BudgetLedger
from software_bench.harness.execution.runner import Observer


def test_budget_is_aggregated_across_roles() -> None:
    ledger = BudgetLedger(Budget(token_limit=10, wall_time_seconds=60, tool_call_limit=2))
    ledger.consume(input_tokens=3, output_tokens=2, tool_calls=1)
    ledger.consume(input_tokens=2, output_tokens=3, tool_calls=1)

    assert ledger.usage.total_tokens == 10
    assert ledger.usage.tool_calls == 2


def test_budget_rejects_next_team_event_before_overrun() -> None:
    ledger = BudgetLedger(Budget(token_limit=5, wall_time_seconds=60, tool_call_limit=2))
    ledger.consume(input_tokens=3, output_tokens=2, tool_calls=1)

    with pytest.raises(BudgetExceeded, match="token"):
        ledger.consume(input_tokens=1, output_tokens=0, tool_calls=0)

    assert ledger.usage.total_tokens == 5


def test_accounting_only_mode_records_overrun_without_enforcing_it() -> None:
    ledger = BudgetLedger(Budget(token_limit=5, wall_time_seconds=60, tool_call_limit=1))

    ledger.consume(input_tokens=6, output_tokens=1, tool_calls=2, enforce=False)

    assert ledger.usage.total_tokens == 7
    assert ledger.usage.tool_calls == 2


def test_observer_enforces_budget_for_every_runtime_plugin() -> None:
    ledger = BudgetLedger(Budget(token_limit=5, wall_time_seconds=60, tool_call_limit=1))
    request = SimpleNamespace(
        mode=SimpleNamespace(agent_runtime="third_party_runtime", roles=[])
    )
    observer = Observer(request, ledger)

    with pytest.raises(BudgetExceeded, match="token"):
        observer.record(
            "model_response", "harness", input_tokens=6, output_tokens=1, tool_calls=2
        )

    assert ledger.usage.total_tokens == 7
    assert ledger.usage.tool_calls == 2
