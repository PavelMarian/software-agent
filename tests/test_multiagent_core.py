import json
import sys

import pytest

from software_multiagent import (
    Action, AgentSpec, CallableTool, ExecutionJournal, FileContentCheck, HookResult,
    Mechanism, MechanismRegistry, ModelTurn, NativeCommandCheck, Phase,
    ReliableSingleAgent, SoloLimits, StepStatus, TaskContext, ToolCall, ToolContract,
    ToolRegistry, VerificationResult, VerificationStatus, VerifiedFileRollback,
    default_mechanisms,
)


class Script:
    def __init__(self, *turns):
        self.turns = iter(turns)
        self.views = []

    def generate(self, agent, task, messages, schemas):
        self.views.append(messages)
        return next(self.turns)


def call(name, **args):
    return ModelTurn("Thought: Execute the next evidence-gathering or repair step.",
                     tool_calls=(ToolCall("c", name, args),))


def done():
    return ModelTurn("Thought: Request public outcome verification.", status=StepStatus.COMPLETE)


def good(ctx):
    return VerificationResult("public", VerificationStatus.PASSED)


def run(tmp_path, model, tools, checks=(good,), **kwargs):
    registry = ToolRegistry(tools)
    return ReliableSingleAgent(model, registry, default_mechanisms(checks), **kwargs).run(
        TaskContext("test", "Meet the public requirements", tmp_path),
        AgentSpec("solo", "Use observations to complete the task", tuple(t.name for t in tools), max_turns=10))


def file_tool(handler, **contract):
    return CallableTool("write", "Write declared files", handler, mutates_workspace=True,
                        action_contract=ToolContract(outputs=lambda a: ("answer.txt",),
                                                     file_mutations_only=True, **contract))


def test_undeclared_mutation_is_rejected_before_handler(tmp_path):
    touched = []
    tool = CallableTool("write", "Write", lambda a, t: touched.append(True), mutates_workspace=True)
    result = run(tmp_path, Script(call("write", path="answer.txt")), [tool],
                 limits=SoloLimits(max_repairs=0))
    assert not touched
    assert result.stop_reason == "checkpoint_unavailable"


def test_complete_multifile_mutation_set_is_restored(tmp_path):
    (tmp_path / "existing").write_text("original")

    def change(a, task):
        (tmp_path / "existing").write_text("bad")
        (tmp_path / "new-dir").mkdir()
        (tmp_path / "new-dir" / "new").write_text("bad")
        return {"exit_code": 1}

    tool = CallableTool("change", "Change both files", change, mutates_workspace=True,
        action_contract=ToolContract(outputs=lambda a: ("existing", "new-dir/new"), file_mutations_only=True))
    result = run(tmp_path, Script(call("change")), [tool], limits=SoloLimits(max_repairs=0))
    assert result.stop_reason == "repair_limit"
    assert (tmp_path / "existing").read_text() == "original"
    assert not (tmp_path / "new-dir").exists()
    assert any(e.data.get("restored") for e in result.evidence)


@pytest.mark.parametrize("status", [VerificationStatus.FAILED, VerificationStatus.INCONCLUSIVE])
def test_precondition_rejects_without_checkpoint_or_execution(tmp_path, status):
    touched = []
    tool = file_tool(lambda a, t: touched.append(True), preconditions=(
        lambda action, task: VerificationResult("ready", status, summary="target not ready"),))
    result = run(tmp_path, Script(call("write")), [tool], limits=SoloLimits(max_repairs=0))
    assert not touched
    assert not any(e["event"] == "checkpoint_created" for e in result.journal)
    assert any(e["payload"].get("kind") == "state" for e in result.journal)


def test_missing_input_is_observed_before_mutation(tmp_path):
    touched = []
    tool = file_tool(lambda a, t: touched.append(True), inputs=lambda a: ("missing.txt",))
    result = run(tmp_path, Script(call("write")), [tool], limits=SoloLimits(max_repairs=0))
    assert not touched
    assert "required input file" in str(result.messages)


def test_bad_result_schema_is_diagnosed_and_restored(tmp_path):
    def change(a, t):
        (tmp_path / "answer.txt").write_text("bad")
        return {"value": "wrong type"}
    tool = file_tool(change, result_schema={"type": "object", "properties": {"value": {"type": "integer"}}})
    result = run(tmp_path, Script(call("write")), [tool], limits=SoloLimits(max_repairs=0))
    assert not (tmp_path / "answer.txt").exists()
    assert any(e["payload"].get("kind") == "artifact" for e in result.journal)


def test_pending_operation_blocks_wrong_actions_and_accepts_exact_poll(tmp_path):
    touched = []
    tool = CallableTool("launch", "Launch", lambda a, t: {"session_id": 9, "status": "running"},
        action_contract=ToolContract(poll_tool="poll"))
    poll = CallableTool("poll", "Poll", lambda a, t: {"exit_code": 0})
    write = CallableTool("other", "Unrelated call", lambda a, t: touched.append(True))
    result = run(tmp_path, Script(call("launch"), call("other"), call("poll", session_id=9), done()),
                 [tool, poll, write])
    assert result.status == StepStatus.COMPLETE
    assert not touched
    started = [e["payload"]["tool"] for e in result.journal if e["event"] == "action_started"]
    assert started == ["launch", "poll"]


def test_pending_operation_cannot_finish_or_restore_under_live_process(tmp_path):
    def launch(a, t):
        (tmp_path / "answer.txt").write_text("in progress")
        return {"session_id": 9, "status": "running"}
    result = run(tmp_path, Script(call("write"), done()), [file_tool(launch)],
                 limits=SoloLimits(max_repairs=0))
    assert result.stop_reason == "pending_operation"
    assert (tmp_path / "answer.txt").read_text() == "in progress"
    assert result.journal[-1]["payload"]["pending_operation"]


def test_failed_poll_restores_original_action(tmp_path):
    def launch(a, t):
        (tmp_path / "answer.txt").write_text("bad")
        return {"session_id": 9, "status": "running"}
    result = run(tmp_path, Script(call("write"), call("poll", session_id=9)),
        [file_tool(launch, poll_tool="poll"),
         CallableTool("poll", "Poll", lambda a, t: {"status": "failed", "error": "solver failed"})],
        limits=SoloLimits(max_repairs=0))
    assert result.stop_reason == "repair_limit"
    assert not (tmp_path / "answer.txt").exists()


def test_poll_transport_failure_does_not_restore_under_running_job(tmp_path):
    def launch(a, t):
        (tmp_path / "answer.txt").write_text("in progress")
        return {"session_id": 9, "status": "running"}
    def offline(a, t):
        raise ConnectionError("poll unavailable")
    result = run(tmp_path, Script(call("write"), call("poll", session_id=9), done()),
        [file_tool(launch, poll_tool="poll"), CallableTool("poll", "Poll", offline)],
        limits=SoloLimits(max_repairs=0))
    assert result.stop_reason == "pending_operation"
    assert (tmp_path / "answer.txt").read_text() == "in progress"


def test_inconclusive_postcheck_allows_new_inspection(tmp_path):
    class UnavailableCheck(Mechanism):
        def apply(self, phase, ctx):
            if phase == Phase.AFTER_ACTION and ctx.action.tool == "read":
                return HookResult((VerificationResult("offline", VerificationStatus.INCONCLUSIVE,
                                                      summary="validator offline"),))
            return HookResult()
    mechanisms = default_mechanisms([good])
    mechanisms.register(UnavailableCheck())
    inspected = []
    result = ReliableSingleAgent(Script(call("read"), call("inspect"), done()), ToolRegistry([
        CallableTool("read", "Read", lambda a, t: "data"),
        CallableTool("inspect", "Inspect validator", lambda a, t: inspected.append(True)),
    ]), mechanisms).run(TaskContext("x", "Solve", tmp_path), AgentSpec("a", "Solve", ("read", "inspect")))
    assert result.status == StepStatus.COMPLETE
    assert inspected == [True]


def test_budget_stop_restores_unaccepted_changes(tmp_path):
    tool = file_tool(lambda a, t: (tmp_path / "answer.txt").write_text("candidate"))
    result = run(tmp_path, Script(call("write")), [tool], limits=SoloLimits(max_tool_calls=1))
    assert result.stop_reason == "tool_call_limit"
    assert not (tmp_path / "answer.txt").exists()


def test_all_validators_run_even_when_one_raises(tmp_path):
    ran = []
    def broken(ctx):
        raise RuntimeError("offline")
    def second(ctx):
        ran.append(True)
        return VerificationResult("second", VerificationStatus.FAILED, summary="incorrect result")
    result = run(tmp_path, Script(done()), [], [broken, second], limits=SoloLimits(max_repairs=0))
    assert ran == [True]
    assert {v.status for v in result.verifications} == {VerificationStatus.INCONCLUSIVE, VerificationStatus.FAILED}


def test_generic_hook_cannot_impersonate_outcome_gate(tmp_path):
    class Unrelated(Mechanism):
        def apply(self, phase, ctx):
            return HookResult((good(ctx),))
    result = ReliableSingleAgent(Script(done()), ToolRegistry(), MechanismRegistry([Unrelated()]),
        limits=SoloLimits(max_repairs=0)).run(TaskContext("x", "Solve", tmp_path), AgentSpec("a", "Solve"))
    assert result.status == StepStatus.FAILED


def test_durable_journal_survives_bounded_context_and_runtime_reuse(tmp_path):
    tool = CallableTool("read", "Read log", lambda a, t: "x" * 10000)
    model = Script(call("read"), done(), call("read"), done())
    runtime = ReliableSingleAgent(model, ToolRegistry([tool]), default_mechanisms([good]),
        limits=SoloLimits(max_context_characters=200), journal_directory=tmp_path / "journals")
    task, agent = TaskContext("x", "Inspect", tmp_path), AgentSpec("a", "Inspect", ("read",))
    first, second = runtime.run(task, agent), runtime.run(task, agent)
    assert first.journal_path != second.journal_path
    assert ExecutionJournal.read(first.journal_path) == first.journal
    assert max(sum(len(m.content) for m in view) for view in model.views) <= 200
    assert "x" * 10000 in json.dumps(first.journal)


def test_journal_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        ExecutionJournal(path)
    assert path.read_text() == "original"


@pytest.mark.parametrize("code,status", [(0, VerificationStatus.PASSED), (1, VerificationStatus.FAILED)])
def test_native_validator_runs_real_process(tmp_path, code, status):
    check = NativeCommandCheck("native", (sys.executable, "-c", f"print('evidence'); raise SystemExit({code})"))
    result = run(tmp_path, Script(done()), [], [check], limits=SoloLimits(max_repairs=0))
    assert result.verifications[-1].status == status
    assert "evidence" in result.evidence[-1].data["output"]


def test_native_validator_timeout_cannot_accept(tmp_path):
    check = NativeCommandCheck("native", (sys.executable, "-c", "import time; time.sleep(10)"), timeout=0.05)
    result = run(tmp_path, Script(done()), [], [check], limits=SoloLimits(max_repairs=0))
    assert result.status == StepStatus.FAILED
    assert result.verifications[-1].status == VerificationStatus.INCONCLUSIVE


def test_file_content_check_rejects_executable_but_wrong_output(tmp_path):
    (tmp_path / "answer.txt").write_bytes(b"wrong")
    result = run(tmp_path, Script(done()), [], [FileContentCheck("content", "answer.txt", b"right")],
                 limits=SoloLimits(max_repairs=0))
    assert result.status == StepStatus.FAILED
    assert result.evidence[-1].data["sha256"]


def test_rollback_refuses_undeclared_new_directory_contents(tmp_path):
    task = TaskContext("x", "Solve", tmp_path)
    action = Action("a", "write", artifact_outputs=("new/file",))
    rollback = VerifiedFileRollback()
    token = rollback.checkpoint(action, task)
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "unrelated").write_text("keep")
    with pytest.raises(OSError):
        rollback.rollback(token, action, task)
    assert (tmp_path / "new" / "unrelated").read_text() == "keep"
