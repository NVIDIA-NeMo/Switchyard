"""Unit tests for the verified-retry orchestrator (no network, no docker)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verified_retry.checks import derive_check_script, run_check_script
from verified_retry.judge import judge_attempt
from verified_retry.orchestrator import run_verified_retry
from verified_retry.spec import ExecutorResult, TaskSpec

INSTRUCTION = "Write the exact string 42 into a file named answer.txt."
CHECK_SCRIPT = 'test -f answer.txt && [ "$(cat answer.txt)" = "42" ] && echo ok'


class FakeLlm:
    """Scripted LLM: derives the check script, judges by peeking at check
    output (deterministic), and always compares by picking attempt 2."""

    def __init__(self, judge_confidence: float = 0.95, derive: str | None = CHECK_SCRIPT):
        self.judge_confidence = judge_confidence
        self.derive = derive
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt.split("\n", 1)[0][:60])
        if "automated acceptance check" in prompt:
            return self.derive if self.derive is not None else "NO_CHECKS"
        if "verifying whether ONE attempt" in prompt:
            good = "ok" in prompt.split("[executed acceptance checks]")[-1] or "wrote 42" in prompt
            return json.dumps(
                {"pass": good, "confidence": self.judge_confidence, "reason": "scripted"}
            )
        if "MOST LIKELY" in prompt:
            return json.dumps({"pick": "2", "reason": "scripted compare"})
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


def make_executor(succeed_on: int | None, crash_on: set[int] = frozenset()):
    """Executor stub: writes the right answer starting at attempt `succeed_on`,
    a wrong answer otherwise; also asserts workspace freshness."""
    counter = {"n": 0}

    def executor(workspace: Path, instruction: str) -> ExecutorResult:
        counter["n"] += 1
        n = counter["n"]
        marker = workspace / "pollution.txt"
        assert not marker.exists(), "workspace not pristine: prior attempt leaked"
        marker.write_text("attempt ran here")
        if n in crash_on:
            return ExecutorResult(exit_code=1, transcript=f"attempt {n} crashed")
        value = "42" if (succeed_on is not None and n >= succeed_on) else "41"
        (workspace / "answer.txt").write_text(value)
        return ExecutorResult(exit_code=0, transcript=f"attempt {n} wrote {value}")

    return executor, counter


def spec(tmp_path: Path, **kw) -> TaskSpec:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "README").write_text(INSTRUCTION)
    return TaskSpec(
        task_id="t", instruction=INSTRUCTION, workspace_src=src,
        **{"max_attempts": 4, **kw},
    )


def test_accepts_first_verified_attempt_and_stops(tmp_path):
    executor, counter = make_executor(succeed_on=1)
    res = run_verified_retry(spec(tmp_path), executor, FakeLlm(), tmp_path / "a")
    assert res.accepted and res.attempt_index == 1 and counter["n"] == 1
    assert (res.workspace / "answer.txt").read_text() == "42"


def test_retries_until_checks_pass(tmp_path):
    executor, counter = make_executor(succeed_on=3)
    res = run_verified_retry(spec(tmp_path), executor, FakeLlm(), tmp_path / "a")
    assert res.accepted and res.attempt_index == 3 and counter["n"] == 3
    assert not res.attempts[0].accepted and not res.attempts[0].checks.passed


def test_never_solves_runs_all_attempts_and_flags_unverified(tmp_path):
    executor, counter = make_executor(succeed_on=None)
    res = run_verified_retry(spec(tmp_path), executor, FakeLlm(), tmp_path / "a")
    assert not res.accepted and counter["n"] == 4
    assert res.attempt_index == 2 and res.pick_reason == "scripted compare"


def test_crash_is_never_accepted(tmp_path):
    executor, counter = make_executor(succeed_on=2, crash_on={1})
    res = run_verified_retry(spec(tmp_path), executor, FakeLlm(), tmp_path / "a")
    assert res.accepted and res.attempt_index == 2
    assert res.attempts[0].executor.crashed and res.attempts[0].verdict is None


def test_low_confidence_judge_vetoes_despite_passing_checks(tmp_path):
    executor, counter = make_executor(succeed_on=1)
    res = run_verified_retry(
        spec(tmp_path), executor, FakeLlm(judge_confidence=0.6), tmp_path / "a"
    )
    assert not res.accepted and counter["n"] == 4  # every attempt gated out


def test_no_checks_available_falls_back_to_judge_only(tmp_path):
    executor, _ = make_executor(succeed_on=1)
    llm = FakeLlm(derive=None)  # NO_CHECKS
    res = run_verified_retry(spec(tmp_path), executor, llm, tmp_path / "a")
    assert res.accepted and res.attempts[0].checks.available is False


def test_check_script_runner_pass_and_fail(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "answer.txt").write_text("42")
    assert run_check_script(CHECK_SCRIPT, ws).passed
    (ws / "answer.txt").write_text("41")
    assert not run_check_script(CHECK_SCRIPT, ws).passed


def test_derive_handles_fences_and_no_checks():
    assert derive_check_script(lambda p: f"```bash\n{CHECK_SCRIPT}\n```", "x") == CHECK_SCRIPT
    assert derive_check_script(lambda p: "NO_CHECKS", "x") is None


def test_unparseable_judge_fails_closed():
    v = judge_attempt(lambda p: "gibberish with no json", "task", "evidence", "")
    assert not v.passed and v.confidence == 0.0


def test_single_attempt_unverified_skips_compare_call(tmp_path):
    executor, _ = make_executor(succeed_on=None)
    llm = FakeLlm()
    res = run_verified_retry(
        spec(tmp_path, max_attempts=1), executor, llm, tmp_path / "a"
    )
    assert not res.accepted and res.attempt_index == 1
    assert res.pick_reason == "single candidate; comparison skipped"
    assert not any(c.startswith("Multiple independent") for c in llm.calls)
