from pathlib import Path

from foambench.drivers import run
from foambench.models import FoamBenchTask
from software_multiagent.application import SoftwareMultiAgent


def test_foambench_driver_only_translates_and_delegates(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(request, tools, **options):  # type: ignore[no-untyped-def]
        captured.update(request=request, tools=tools, options=options)
        return {"status": "completed"}

    monkeypatch.setattr(SoftwareMultiAgent, "run_configured", fake_run)
    task = FoamBenchTask(
        instance_id="case-1",
        source_case_id="source-1",
        split="basic",
        problem_statement="Create the case",
        target_software="OpenFOAM 10",
        metadata={"public": True},
    )
    marker = object()
    result = run(
        task,
        marker,
        {
            "workspace": str(tmp_path),
            "provider": "openrouter",
            "model": "example/model",
        },
    )

    assert result == {"status": "completed"}
    request = captured["request"]
    assert request.task_id == "case-1"
    assert request.objective == "Create the case"
    assert request.workspace == Path(tmp_path)
    assert request.metadata == {"public": True}
    assert captured["tools"] is marker
    assert captured["options"] == {
        "provider": "openrouter",
        "model_name": "example/model",
    }
