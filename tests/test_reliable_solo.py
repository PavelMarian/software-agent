from dataclasses import replace

import pytest

from software_multiagent import (
    AgentSpec, CallableTool, Evidence, HookResult, Mechanism, MechanismRegistry,
    ModelTurn, Phase, ReliableSingleAgent, SoloLimits, StepStatus, TaskContext,
    ToolCall, ToolRegistry, Usage, VerificationResult, VerificationStatus,
    VerifiedFileRollback, default_mechanisms,
    ToolContract,
)


class Script:
    def __init__(self, *turns):
        self.turns = iter(turns)
        self.views = []

    def generate(self, agent, task, messages, schemas):
        self.views.append(messages)
        turn = next(self.turns)
        return replace(turn, content="Thought: " + (turn.content or "Execute the next test action."))


def done():
    return ModelTurn("Finished", status=StepStatus.COMPLETE)


def write(text="good", path="answer.txt"):
    return ModelTurn(tool_calls=(ToolCall("call", "write", {"path": path, "text": text}),))


def answer_check(ctx):
    path = ctx.state.task.workspace / "answer.txt"
    ok = path.exists() and path.read_text() == "good"
    return VerificationResult("answer", VerificationStatus.PASSED if ok else VerificationStatus.FAILED,
                              summary="expected answer.txt to contain good")


@pytest.fixture
def setup(tmp_path):
    touched = []

    def handler(args, task):
        touched.append(args["text"])
        (task.workspace / args["path"]).write_text(args["text"])
        return {"exit_code": 0}

    tool = CallableTool("write", "Write the answer", handler,
        {"type": "object", "properties": {"path": {"type": "string"},
          "text": {"type": "string"}}, "required": ["path", "text"],
         "additionalProperties": False}, mutates_workspace=True,
         action_contract=ToolContract(outputs=lambda a: (a["path"],), file_mutations_only=True))
    return TaskContext("test", "Write good", tmp_path), AgentSpec(
        "solo", "Inspect, execute, verify and repair", ("write",), max_turns=12,
    ), ToolRegistry([tool]), touched


def test_real_file_repaired_after_final_rejection(setup):
    task, agent, tools, touched = setup
    model = Script(write("bad"), done(), write(), done())
    result = ReliableSingleAgent(model, tools, default_mechanisms([answer_check]),
                                 rollback=VerifiedFileRollback()).run(task, agent)
    assert result.status == StepStatus.COMPLETE
    assert touched == ["bad", "good"]
    assert any(e.data.get("restored") for e in result.evidence)
    assert "expected answer" in str(model.views[2])
    assert not any("bad" in str(m) and m.sender == "knowledge" for m in model.views[2])


def test_no_validator_cannot_complete(setup):
    task, agent, tools, _ = setup
    result = ReliableSingleAgent(Script(done(), done()), tools, default_mechanisms([])).run(task, agent)
    assert result.stop_reason == "completion_unverified"


def test_empty_registry_cannot_complete(setup):
    task, agent, tools, _ = setup
    result = ReliableSingleAgent(Script(done(), done()), tools, MechanismRegistry()).run(task, agent)
    assert result.status == StepStatus.FAILED


def test_precondition_plugin_prevents_side_effect_then_allows_repair(setup):
    task, agent, tools, touched = setup

    class TargetGuard(Mechanism):
        name = "target"

        def apply(self, phase, ctx):
            if phase == Phase.BEFORE_ACTION and ctx.action.arguments["path"] != "answer.txt":
                return HookResult((VerificationResult("target", VerificationStatus.FAILED,
                                                     summary="wrong target"),))
            return HookResult()

    plugins = default_mechanisms([answer_check])
    plugins.register(TargetGuard())
    result = ReliableSingleAgent(Script(write(path="wrong.txt"), write(), done()), tools,
                                 plugins).run(task, agent)
    assert result.status == StepStatus.COMPLETE
    assert touched == ["good"]
    assert not (task.workspace / "wrong.txt").exists()


def test_tool_failure_stops_batch_before_later_mutation(setup):
    task, agent, tools, touched = setup
    bad = ModelTurn(tool_calls=(ToolCall("a", "write", {"path": "answer.txt"}),
                               ToolCall("b", "write", {"path": "later.txt", "text": "bad"})))
    result = ReliableSingleAgent(Script(bad, write(), done()), tools,
                                 default_mechanisms([answer_check])).run(task, agent)
    assert result.status == StepStatus.COMPLETE
    assert touched == ["good"]


def test_terminal_handler_is_not_a_completion_bypass(setup):
    task, _, _, _ = setup
    called = []
    tool = CallableTool("finish", "Finish", lambda a, t: called.append(True))
    agent = AgentSpec("solo", "Solve", ("finish",), terminal_tools=("finish",))
    turn = ModelTurn(tool_calls=(ToolCall("f", "finish", {}),))
    result = ReliableSingleAgent(Script(turn, turn), ToolRegistry([tool]),
                                 default_mechanisms([answer_check])).run(task, agent)
    assert result.status == StepStatus.FAILED
    assert called == []


def test_token_limit_prevents_action_after_expensive_generation(setup):
    task, agent, tools, touched = setup
    turn = ModelTurn(tool_calls=write().tool_calls, usage=Usage(100, 1))
    result = ReliableSingleAgent(Script(turn), tools, default_mechanisms([answer_check]),
                                 limits=SoloLimits(max_tokens=100)).run(task, agent)
    assert result.stop_reason == "token_limit"
    assert touched == []
    assert result.usage.input_tokens == 100


def test_failure_limit_cannot_be_evaded_by_changing_arguments(setup):
    task, agent, tools, _ = setup
    bad = [ModelTurn(tool_calls=(ToolCall(str(i), "write", {"path": str(i)}),)) for i in range(3)]
    result = ReliableSingleAgent(Script(*bad), tools, default_mechanisms([answer_check]),
                                 limits=SoloLimits(max_repairs=2)).run(task, agent)
    assert result.stop_reason == "repair_limit"


def test_validator_exception_fails_closed(setup):
    task, agent, tools, _ = setup

    def broken(ctx):
        raise RuntimeError("validator unavailable")

    result = ReliableSingleAgent(Script(done(), done()), tools,
                                 default_mechanisms([broken])).run(task, agent)
    assert result.stop_reason == "completion_unverified"


def test_unverified_rollback_stops_execution(setup):
    task, agent, tools, touched = setup

    class BrokenRollback:
        def checkpoint(self, action, task):
            return True

        def rollback(self, token, action, task):
            return Evidence("rollback", "fake", data={"restored": False})

    result = ReliableSingleAgent(Script(write("bad"), done()), tools,
        default_mechanisms([answer_check]), rollback=BrokenRollback()).run(task, agent)
    assert result.stop_reason == "rollback_failed"
    assert touched == ["bad"]


def test_rollback_restores_preexisting_content(setup):
    task, agent, tools, _ = setup
    (task.workspace / "answer.txt").write_text("original")
    result = ReliableSingleAgent(Script(write("bad"), done()), tools,
        default_mechanisms([answer_check]), rollback=VerifiedFileRollback(),
        limits=SoloLimits(max_repairs=0)).run(task, agent)
    assert result.status == StepStatus.FAILED
    assert (task.workspace / "answer.txt").read_text() == "original"


def test_permission_error_propagates(setup):
    task, agent, tools, _ = setup
    turn = ModelTurn(tool_calls=(ToolCall("x", "forbidden", {}),))
    with pytest.raises(PermissionError):
        ReliableSingleAgent(Script(turn), tools, default_mechanisms([])).run(task, agent)


def test_no_progress_on_identical_reads(tmp_path):
    tool = CallableTool("read", "Read", lambda a, t: "unchanged")
    call = ModelTurn(tool_calls=(ToolCall("r", "read", {}),))
    agent = AgentSpec("solo", "Solve", ("read",))
    result = ReliableSingleAgent(Script(call, call, call), ToolRegistry([tool]),
        default_mechanisms([])).run(TaskContext("x", "Solve", tmp_path), agent)
    assert result.stop_reason == "no_progress"


def test_runtime_reusable_without_cross_task_failures(setup):
    task, agent, tools, _ = setup
    runtime = ReliableSingleAgent(Script(write(), done(), write(), done()), tools,
                                  default_mechanisms([answer_check]))
    assert runtime.run(task, agent).status == StepStatus.COMPLETE
    assert runtime.run(task, agent).status == StepStatus.COMPLETE


def test_context_budget_and_latest_feedback(setup):
    task, agent, tools, _ = setup
    model = Script(ModelTurn("x" * 1000), write(), done())
    result = ReliableSingleAgent(model, tools, default_mechanisms([answer_check]),
                                 limits=SoloLimits(max_context_characters=50)).run(task, agent)
    assert result.status == StepStatus.COMPLETE
    assert all(sum(len(m.content) for m in view) <= 50 for view in model.views)


def test_duplicate_plugin_rejected():
    registry = MechanismRegistry([Mechanism()])
    with pytest.raises(ValueError):
        registry.register(Mechanism())


def test_nested_schema_rejects_bad_item_before_execution(tmp_path):
    from software_multiagent.runtime.reliability.action_validation import _validate_arguments
    schema = {"type": "object", "properties": {"values": {"type": "array",
              "items": {"type": "integer", "minimum": 0}}}}
    with pytest.raises(ValueError):
        _validate_arguments({"values": [1, "bad"]}, schema)
    with pytest.raises(ValueError):
        _validate_arguments({"values": [-1]}, schema)
    _validate_arguments({"values": [0, 1]}, schema)


@pytest.mark.parametrize("path", ["../outside", "C:outside", "C:/outside"])
def test_rollback_rejects_escaping_paths(setup, path):
    from software_multiagent import Action
    task, _, _, _ = setup
    action = Action("a", "write", artifact_outputs=(path,), mutates_workspace=True)
    with pytest.raises(PermissionError):
        VerifiedFileRollback().checkpoint(action, task)


def test_async_result_skips_dependent_call_until_observed(tmp_path):
    called = []
    tool = CallableTool("launch", "Start", lambda a, t: {"session_id": 7, "status": "running"})
    later = CallableTool("later", "Dependent work", lambda a, t: called.append(True))
    turn = ModelTurn(tool_calls=(ToolCall("a", "launch", {}),))
    agent = AgentSpec("solo", "Run", ("launch", "later"), max_turns=1)
    result = ReliableSingleAgent(Script(turn), ToolRegistry([tool, later]),
        default_mechanisms([])).run(TaskContext("x", "Run", tmp_path), agent)
    assert result.status == StepStatus.FAILED
    assert called == []
    assert any("inconclusive" in message.content for message in result.messages)


def test_checkpoint_failure_blocks_mutation(setup):
    task, agent, tools, touched = setup
    (task.workspace / "answer.txt").write_text("original")
    result = ReliableSingleAgent(Script(write()), tools, default_mechanisms([answer_check]),
        rollback=VerifiedFileRollback(max_bytes=1)).run(task, agent)
    assert result.stop_reason == "checkpoint_failed"
    assert touched == []


def test_failed_action_rolls_back_even_when_repairs_exhausted(tmp_path):
    def fail(args, task):
        (task.workspace / "answer.txt").write_text("bad")
        return {"exit_code": 1}

    tool = CallableTool("write", "Write", fail, mutates_workspace=True,
        action_contract=ToolContract(outputs=lambda a: (a["path"],), file_mutations_only=True))
    result = ReliableSingleAgent(Script(write()), ToolRegistry([tool]),
        default_mechanisms([answer_check]), rollback=VerifiedFileRollback(),
        limits=SoloLimits(max_repairs=0)).run(TaskContext("x", "Run", tmp_path),
                                            AgentSpec("solo", "Run", ("write",)))
    assert result.stop_reason == "repair_limit"
    assert not (tmp_path / "answer.txt").exists()


def test_multiple_validators_all_must_pass(setup):
    task, agent, tools, _ = setup

    def unknown(ctx):
        return VerificationResult("scientific-correctness", VerificationStatus.INCONCLUSIVE)

    result = ReliableSingleAgent(Script(write(), done(), done()), tools,
        default_mechanisms([answer_check, unknown])).run(task, agent)
    assert result.stop_reason == "completion_unverified"
