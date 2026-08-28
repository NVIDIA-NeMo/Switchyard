# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for task generation pipeline — run_claude helper, state persistence, step logic."""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


async def _fake_assemble_artifacts(t, s, sf):
    """Shared stub for _run_assemble_task_dir_artifacts_one — advances stage only."""
    from craft_taskgen.config import Stage

    t.stage = Stage.TESTS_DISCOVERED


def test_run_claude_constructs_correct_command():
    """run_claude passes -p, --permission-mode auto, --output-format json."""
    from craft_taskgen.claude_cli import run_claude

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps({"result": "ok", "session_id": "s1"})
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result) as mock_run:
        result = run_claude("test prompt", max_turns=10)

    cmd = mock_run.call_args[0][0]
    assert "-p" in cmd
    assert "--permission-mode" in cmd
    assert "auto" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--max-turns" in cmd
    assert "10" in cmd
    assert result["result"] == "ok"


def test_run_claude_with_json_schema():
    """run_claude passes --json-schema when provided."""
    from craft_taskgen.claude_cli import run_claude

    schema = {"type": "object", "properties": {"v": {"type": "string"}}}
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps({"result": "", "structured_output": {"v": "yes"}})
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result) as mock_run:
        result = run_claude("test", json_schema=schema)

    cmd = mock_run.call_args[0][0]
    assert "--json-schema" in cmd
    assert result["structured_output"]["v"] == "yes"


def test_run_claude_sets_gateway_env_instead_of_model_flag():
    """Gateway-only: `--model` is never on the CLI. Model is passed via
    ANTHROPIC_MODEL env so the NVIDIA gateway (not OAuth) routes the call.
    """
    from craft_taskgen.claude_cli import run_claude

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = json.dumps({"result": "ok"})
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result) as mock_run:
        run_claude("test", model="aws/anthropic/bedrock-claude-opus-4-6")

    cmd = mock_run.call_args[0][0]
    assert "--model" not in cmd
    env = mock_run.call_args.kwargs["env"]
    assert env["ANTHROPIC_MODEL"] == "aws/anthropic/bedrock-claude-opus-4-6"
    # All sub-agent aliases pinned to the same gateway model.
    for alias in (
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        assert env[alias] == "aws/anthropic/bedrock-claude-opus-4-6"


def test_run_claude_raises_without_gateway_env(monkeypatch):
    """Guardrail: missing ANTHROPIC_* env must raise, not silently OAuth-fall-through."""
    import pytest

    from craft_taskgen.claude_cli import run_claude

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="Gateway-only policy"):
        run_claude("hi", model="aws/anthropic/bedrock-claude-opus-4-6")


def test_run_claude_raises_without_model(monkeypatch):
    """Guardrail: missing model (empty profile default) must also raise."""
    import pytest

    import craft_taskgen.config as _cfg
    from craft_taskgen.claude_cli import run_claude

    monkeypatch.setattr(_cfg, "LLM_STEP_MODEL", "")
    with pytest.raises(RuntimeError, match="Gateway-only policy"):
        run_claude("hi")


def test_run_claude_timeout_returns_error():
    from craft_taskgen.claude_cli import run_claude

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 300)):
        result = run_claude("slow", timeout=300)

    assert result["error"] == "timeout"


def test_run_claude_bad_json_returns_error():
    from craft_taskgen.claude_cli import run_claude

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "not json"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result):
        result = run_claude("test")

    assert result["error"] == "json_parse"


def test_state_save_and_load(tmp_path):
    from craft_taskgen.config import PipelineState, Stage, TaskState

    state = PipelineState(created="2026-04-06")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="Add streaming",
        stage=Stage.PROMISING,
        eval_verdict="PROMISING",
    )

    path = str(tmp_path / "state.json")
    state.save(path)

    loaded = PipelineState.load(path)
    assert len(loaded.tasks) == 1
    assert loaded.tasks["t1"].stage == Stage.PROMISING
    assert loaded.tasks["t1"].eval_verdict == "PROMISING"


def test_select_candidates(tmp_path):
    from craft_taskgen.steps import select_candidates

    data = {
        "candidates": [
            {
                "sha": "aaa",
                "base_sha": "aaa0",
                "merge_base_sha": "aaa0",
                "subject": "feat: add X",
                "score": 10,
                "has_test_patch": True,
                "source_files": ["src/x.py"],
            },
            {
                "sha": "bbb",
                "base_sha": "bbb0",
                "merge_base_sha": "bbb0",
                "subject": "fix: typo",
                "score": 5,
                "has_test_patch": True,
                "source_files": ["src/y.py"],
            },
            {
                "sha": "ccc",
                "base_sha": "ccc0",
                "merge_base_sha": "ccc0",
                "subject": "refactor",
                "score": 0,
                "has_test_patch": True,
                "source_files": ["src/z.py"],
            },
        ]
    }
    fpath = tmp_path / "testrepo.json"
    fpath.write_text(json.dumps(data))

    results = select_candidates([str(fpath)], top_per_repo=5, max_total=10)
    assert len(results) == 2  # ccc filtered out (score=0)
    assert results[0]["sha"] == "aaa"
    assert results[0]["repo"] == "testrepo"


def _make_judge_result(result_dict: dict):
    """Build a fake llm_judge.JudgeResult for the patched path."""
    from craft_taskgen.llm_judge import JudgeResult

    return JudgeResult(
        result=result_dict,
        usage={"input_tokens": 100, "output_tokens": 50, "cached_tokens": 0},
        model="openai/aws/anthropic/bedrock-claude-opus-4-6",
        latency_s=0.01,
    )


def _fake_fetch_context(repo: str, sha: str, merge_base_sha: str) -> tuple[str, str, str]:
    """Bypass git subprocess in tests."""
    return (
        f"[fake stat for {repo}@{sha} from {merge_base_sha}]",
        f"[fake diff for {repo}@{sha} from {merge_base_sha}]",
        "",
    )


async def _fake_alignment_pass(task, state, state_file) -> None:
    """Bypass alignment judge in run_task_pipeline tests.

    Real _run_alignment_one hits the gateway via llm_judge.judge — with
    conftest's fake ANTHROPIC_BASE_URL=test.invalid, that hangs each test
    for 3× litellm-default timeout. Tests that exercise the full pipeline
    flow patch this helper to advance the stage without network I/O.
    """
    from craft_taskgen.config import Stage

    task.alignment_verdict = "ok"
    task.stage = Stage.ALIGNMENT_CHECKED


def _empty_deep_dive_context() -> dict:
    """Empty pre-assembled deep-dive context for tests that don't need real
    harbor-lab output. Keeps _fetch_deep_dive_context off the hot path so
    tests don't shell out."""
    return {
        "instruction_md": "[fake]",
        "reward_json": "{}",
        "verify_output_tail": "",
        "postmerge_test_bodies": [],
        "harbor_lab_errors": "",
        "harbor_lab_edits": "",
        "harbor_lab_tool_sequence": "",
        "harbor_lab_metrics": "",
        "f2p_tests": "",
        "p2p_tests": "",
        "f2p_skip": "",
        "p2p_skip": "",
    }


def _make_triage_dispatcher(
    deep_dive_output: dict,
    reviewer_output: dict | None = None,
    build_output: dict | None = None,
    fairness_output: dict | None = None,
):
    """Route llm_judge.judge calls for triage tests.

    Production triage now fans out to:
      - DEEP_DIVE_SCHEMA (Opus per-test skip/keep)
      - FAIRNESS_REVIEW_SCHEMA (cross-family severity judge)
      - BUILD_SCHEMA (only when triage triggers a Build regen)

    `fairness_output` defaults to `severity=none` (no concern) when not
    supplied; pass `{"severity": "major", "evidence_quote": "...",
    "evidence_test": "..."}` to exercise the auto-regen path.

    The legacy `reviewer_output` parameter is accepted but ignored.
    """
    del reviewer_output  # unused — skeptical reviewer stage was removed
    from craft_taskgen.prompts import BUILD_SCHEMA, DEEP_DIVE_SCHEMA, FAIRNESS_REVIEW_SCHEMA

    fairness = fairness_output or {
        "severity": "none",
        "reason": "(no concern)",
        "evidence_quote": "",
        "evidence_test": "",
    }

    async def dispatcher(**kwargs):
        schema = kwargs.get("schema")
        if schema is DEEP_DIVE_SCHEMA:
            return _make_judge_result(deep_dive_output)
        if schema is FAIRNESS_REVIEW_SCHEMA:
            return _make_judge_result(fairness)
        if schema is BUILD_SCHEMA and build_output is not None:
            return _make_judge_result(build_output)
        return _make_judge_result({})

    return dispatcher


def test_select_candidates_zero_top_per_repo_means_no_cap(tmp_path):
    from craft_taskgen.steps import select_candidates

    data = {
        "candidates": [
            {
                "sha": "aaa",
                "base_sha": "aaa0",
                "merge_base_sha": "aaa0",
                "subject": "feat: add X",
                "score": 10,
                "has_test_patch": True,
                "source_files": ["src/x.py"],
            },
            {
                "sha": "bbb",
                "base_sha": "bbb0",
                "merge_base_sha": "bbb0",
                "subject": "feat: add Y",
                "score": 9,
                "has_test_patch": True,
                "source_files": ["src/y.py"],
            },
        ]
    }
    fpath = tmp_path / "testrepo.json"
    fpath.write_text(json.dumps(data))

    results = select_candidates([str(fpath)], top_per_repo=0, max_total=10)
    assert len(results) == 2


def test_select_candidates_zero_max_total_means_no_cap(tmp_path):
    from craft_taskgen.steps import select_candidates

    data = {
        "candidates": [
            {
                "sha": "aaa",
                "base_sha": "aaa0",
                "merge_base_sha": "aaa0",
                "subject": "feat: add X",
                "score": 10,
                "has_test_patch": True,
                "source_files": ["src/x.py"],
            },
            {
                "sha": "bbb",
                "base_sha": "bbb0",
                "merge_base_sha": "bbb0",
                "subject": "feat: add Y",
                "score": 9,
                "has_test_patch": True,
                "source_files": ["src/y.py"],
            },
        ]
    }
    fpath = tmp_path / "testrepo.json"
    fpath.write_text(json.dumps(data))

    results = select_candidates([str(fpath)], top_per_repo=5, max_total=0)
    assert len(results) == 2


def test_step_evaluate_accept():
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_evaluate

    state = PipelineState(created="2026-04-06")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.CANDIDATE,
    )

    async def fake_judge(**kwargs):
        return _make_judge_result(
            {
                "verdict": "accept",
                "reason": "Q1: models will diverge on integration sequencing",
                "instruction_sketch": "Both sync and async paths must handle...",
            }
        )

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
        patch("craft_taskgen.steps._fetch_evaluate_context", side_effect=_fake_fetch_context),
    ):
        asyncio.run(step_evaluate(state, "/dev/null"))

    assert state.tasks["t1"].stage == Stage.PROMISING
    assert state.tasks["t1"].eval_verdict == "accept"
    assert "evaluate" in state.tasks["t1"].llm_usage


def test_step_evaluate_reject():
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_evaluate

    state = PipelineState(created="2026-04-06")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="mypy",
        commit_sha="def",
        base_sha="base123",
        merge_base_sha="base123",
        description="fix",
        stage=Stage.CANDIDATE,
    )

    async def fake_judge(**kwargs):
        return _make_judge_result(
            {
                "verdict": "reject",
                "reason": "matches SA1 pattern — bug fix with obvious strategy",
                "reject_pattern": "SA1",
            }
        )

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
        patch("craft_taskgen.steps._fetch_evaluate_context", side_effect=_fake_fetch_context),
    ):
        asyncio.run(step_evaluate(state, "/dev/null"))

    assert state.tasks["t1"].stage == Stage.REJECTED
    assert state.tasks["t1"].eval_verdict == "reject"


def _fake_fetch_alignment_context(task_dir: str, repo: str, sha: str, merge_base_sha: str) -> dict:
    """Bypass git subprocess + file reads in alignment tests."""
    return {
        "instruction_md": f"[fake instruction for {repo}@{sha}]",
        "reference_test_bodies": [("tests/test_x.py", "def test_foo():\n    assert 1 == 1\n")],
        "diff": f"[fake diff for {merge_base_sha}..{sha}]",
    }


def test_step_alignment_ok_first_attempt():
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_alignment

    state = PipelineState(created="2026-04-22")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="arrow",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.BUILT,
        task_dir="harbor-tasks/t2v3-AR-test",
    )

    async def fake_judge(**kwargs):
        return _make_judge_result(
            {"verdict": "ok", "reason": "instruction and tests align; no V4 violations"}
        )

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
        patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
    ):
        asyncio.run(step_alignment(state, "/dev/null"))

    task = state.tasks["t1"]
    assert task.stage == Stage.ALIGNMENT_CHECKED
    assert task.alignment_verdict == "ok"
    assert len(task.alignment_attempts) == 1


def test_step_alignment_retention_retry_accepts():
    """non-ok on attempt 1, ok on attempt 2 → accepted (with α=2 override).

    Default α=1 disables retention retry; this test explicitly bumps α=2
    via PipelineProfile to verify the retention-retry path still works
    when configured.
    """
    import asyncio

    import craft_taskgen.config as _cfg
    from craft_taskgen.config import PipelineProfile, PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_alignment

    orig_alpha = _cfg.ALIGNMENT_MAX_RETRIES
    PipelineProfile(alignment_max_retries=2).apply()

    try:
        state = PipelineState(created="2026-04-22")
        state.tasks["t1"] = TaskState(
            task_id="t1",
            repo="arrow",
            commit_sha="abc",
            base_sha="base123",
            merge_base_sha="base123",
            description="feat",
            stage=Stage.BUILT,
            task_dir="harbor-tasks/t2v3-AR-test",
        )

        results_iter = iter(
            [
                _make_judge_result({"verdict": "narrow_tests", "reason": "first pass"}),
                _make_judge_result({"verdict": "ok", "reason": "retry pass"}),
            ]
        )

        async def fake_judge(**kwargs):
            return next(results_iter)

        with (
            patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
            patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
        ):
            asyncio.run(step_alignment(state, "/dev/null"))

        task = state.tasks["t1"]
        assert task.stage == Stage.ALIGNMENT_CHECKED
        assert task.alignment_verdict == "ok"
        assert len(task.alignment_attempts) == 2
    finally:
        _cfg.ALIGNMENT_MAX_RETRIES = orig_alpha


def test_step_alignment_first_rejection_post_triage_rejects():
    """Post-triage alignment-only path: leaked → REJECT directly (no regen).

    With the N-parallel build+align refactor, regen lives inside
    ``_one_candidate_loop`` (see test_one_candidate_loop_regen_path).
    The ``step_alignment`` wrapper now only handles the post-triage
    Stage.BUILT entry and explicitly does NOT trigger another build
    regen — the triage path's instruction is what it is.

    At default α=1 there's a single judge attempt before rejection.
    """
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_alignment

    state = PipelineState(created="2026-04-22")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="arrow",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.BUILT,
        task_dir="harbor-tasks/t2v3-AR-test",
    )

    async def fake_judge(**kwargs):
        return _make_judge_result(
            {"verdict": "leaked", "reason": "names private method", "leakage_evidence": ["_helper"]}
        )

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
        patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
    ):
        asyncio.run(step_alignment(state, "/dev/null"))

    task = state.tasks["t1"]
    # Post-triage path rejects directly without regenerating Build.
    assert task.stage == Stage.REJECTED
    assert task.alignment_verdict == "leaked"
    assert task.alignment_regen_count == 0  # no regen fired
    # At default α=1: 1 judge call before rejection.
    assert len(task.alignment_attempts) == 1


def test_step_alignment_vague_rejects_without_regen():
    """`vague` verdict skips regen (not actionable) and rejects directly."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_alignment

    state = PipelineState(created="2026-04-22")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="arrow",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.BUILT,
        task_dir="harbor-tasks/t2v3-AR-test",
    )

    async def fake_judge(**kwargs):
        return _make_judge_result({"verdict": "vague", "reason": "instruction too under-specified"})

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
        patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
    ):
        asyncio.run(step_alignment(state, "/dev/null"))

    task = state.tasks["t1"]
    assert task.stage == Stage.REJECTED  # vague is not actionable, no regen
    assert task.alignment_regen_count == 0


# ---------------------------------------------------------------------------
# N-parallel build+alignment: _one_candidate_loop and orchestrator tests
# ---------------------------------------------------------------------------


def _make_candidate_test_task(task_id: str = "t1", regen_count: int = 0):
    from craft_taskgen.config import Stage, TaskState

    return TaskState(
        task_id=task_id,
        repo="arrow",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.PROMISING,
        alignment_regen_count=regen_count,
    )


def _fake_build_outcome_ok(slug: str = "test-slug", words: int = 50):
    """Construct a passing _BuildOutcome — used to mock _build_instruction."""
    from craft_taskgen.steps import _BuildOutcome

    return _BuildOutcome(
        outcome="ok",
        task_dir="/tmp/run/t2v3-FAKEID-cand0",
        instruction_md="# Fake\n\nDo X.",
        instruction_words=words,
        slug=slug,
        usage_entry={
            "tokens_in": 100,
            "tokens_out": 50,
            "tokens_cached": 0,
            "model": "fake/opus",
            "latency_s": 0.01,
        },
        iteration_entry={
            "timestamp": "2026-04-25T00:00:00",
            "step": "build",
            "task_dir": "/tmp/run/t2v3-FAKEID-cand0",
            "instruction_words": words,
        },
    )


def test_one_candidate_loop_happy_path(tmp_path):
    """Build OK → alignment OK on first attempt → outcome=pass."""
    import asyncio

    from craft_taskgen.steps import _one_candidate_loop

    task = _make_candidate_test_task()

    async def fake_judge(**kwargs):
        # Pre-determined ok verdict on alignment (build is mocked separately)
        return _make_judge_result(
            {"verdict": "ok", "reason": "aligned", "v4_audit": {}, "leakage_evidence": []}
        )

    async def fake_build_instruction(t, run_dir, *, feedback, cand_id, log_prefix="build"):
        out = _fake_build_outcome_ok()
        # Use the actual cand-dir path so cleanup works
        out.task_dir = str(tmp_path / f"t2v3-FAKEID-cand{cand_id}")
        os.makedirs(out.task_dir, exist_ok=True)
        with open(os.path.join(out.task_dir, "instruction.md"), "w") as f:
            f.write(out.instruction_md)
        out.iteration_entry["task_dir"] = out.task_dir
        return out

    sem = asyncio.Semaphore(1)

    async def run():
        with (
            patch("craft_taskgen.steps._build_instruction", side_effect=fake_build_instruction),
            patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
            patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
        ):
            return await _one_candidate_loop(0, task, str(tmp_path), sem)

    result = asyncio.run(run())
    assert result.outcome == "pass"
    assert result.alignment_regen_count == 0
    assert result.alignment_verdict == "ok"
    assert len(result.alignment_attempts) == 1
    assert len(result.build_usage) == 1


def test_one_candidate_loop_regen_path(tmp_path):
    """leaked → rebuild → ok → outcome=pass, regen_count=1.

    At default α=1 r=2: initial alignment leaks (1 call), rebuild #1's
    alignment passes (1 call). Total 2 alignment attempts, 2 builds.
    """
    import asyncio

    from craft_taskgen.steps import _one_candidate_loop

    task = _make_candidate_test_task()
    judge_call_count = [0]

    async def fake_judge(**kwargs):
        # 1st call (initial alignment): leaked. 2nd call (post-rebuild): ok.
        judge_call_count[0] += 1
        if judge_call_count[0] == 1:
            return _make_judge_result(
                {
                    "verdict": "leaked",
                    "reason": "names private symbol",
                    "v4_audit": {},
                    "leakage_evidence": ["_helper"],
                }
            )
        return _make_judge_result(
            {"verdict": "ok", "reason": "clean after regen", "v4_audit": {}, "leakage_evidence": []}
        )

    build_call_count = [0]

    async def fake_build_instruction(t, run_dir, *, feedback, cand_id, log_prefix="build"):
        build_call_count[0] += 1
        out = _fake_build_outcome_ok()
        out.task_dir = str(tmp_path / f"t2v3-FAKEID-cand{cand_id}")
        os.makedirs(out.task_dir, exist_ok=True)
        with open(os.path.join(out.task_dir, "instruction.md"), "w") as f:
            f.write(out.instruction_md)
        out.iteration_entry["task_dir"] = out.task_dir
        return out

    sem = asyncio.Semaphore(1)

    async def run():
        with (
            patch("craft_taskgen.steps._build_instruction", side_effect=fake_build_instruction),
            patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
            patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
        ):
            return await _one_candidate_loop(0, task, str(tmp_path), sem)

    result = asyncio.run(run())
    assert result.outcome == "pass"
    assert result.alignment_regen_count == 1
    assert build_call_count[0] == 2  # initial + regen
    # Defaults α=1 r=2: pre-regen 1 leaked attempt, post-regen 1 ok attempt = 2 total.
    # (Earlier α=3 default would have been 3 + 1 = 4; that scenario is covered
    # explicitly via the test_alignment_max_retries override path.)
    assert len(result.alignment_attempts) == 2
    # Build was called twice — 2 usage entries
    assert len(result.build_usage) == 2


def test_one_candidate_loop_final_reject_after_regen(tmp_path):
    """leaked → rebuild × r → still leaked → outcome=reject, regen_count=r.

    At default α=1 r=2: initial + 2 rebuilds = 3 alignment evaluations,
    each producing 1 attempt that says leaked. Loop exits when budget
    exhausted with actionable verdict; outcome=reject, regen_count=2.
    """
    import asyncio

    from craft_taskgen.steps import _one_candidate_loop

    task = _make_candidate_test_task()

    async def fake_judge(**kwargs):
        # Always leaked, both pre- and post-regen
        return _make_judge_result(
            {
                "verdict": "leaked",
                "reason": "still leaks",
                "v4_audit": {},
                "leakage_evidence": ["_foo"],
            }
        )

    async def fake_build_instruction(t, run_dir, *, feedback, cand_id, log_prefix="build"):
        out = _fake_build_outcome_ok()
        out.task_dir = str(tmp_path / f"t2v3-FAKEID-cand{cand_id}")
        os.makedirs(out.task_dir, exist_ok=True)
        return out

    sem = asyncio.Semaphore(1)

    async def run():
        with (
            patch("craft_taskgen.steps._build_instruction", side_effect=fake_build_instruction),
            patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
            patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
        ):
            return await _one_candidate_loop(0, task, str(tmp_path), sem)

    result = asyncio.run(run())
    assert result.outcome == "reject"
    # At α=1 r=2: 3 alignment evaluations × 1 attempt each = 3 total attempts;
    # regen_count=2 (initial + 2 rebuilds before budget exhausted).
    assert result.alignment_regen_count == 2
    assert len(result.alignment_attempts) == 3


def test_one_candidate_loop_vague_no_regen(tmp_path):
    """vague verdict is non-actionable → outcome=reject without regen."""
    import asyncio

    from craft_taskgen.steps import _one_candidate_loop

    task = _make_candidate_test_task()

    async def fake_judge(**kwargs):
        return _make_judge_result(
            {"verdict": "vague", "reason": "underspec", "v4_audit": {}, "leakage_evidence": []}
        )

    async def fake_build_instruction(t, run_dir, *, feedback, cand_id, log_prefix="build"):
        out = _fake_build_outcome_ok()
        out.task_dir = str(tmp_path / f"t2v3-FAKEID-cand{cand_id}")
        os.makedirs(out.task_dir, exist_ok=True)
        return out

    sem = asyncio.Semaphore(1)

    async def run():
        with (
            patch("craft_taskgen.steps._build_instruction", side_effect=fake_build_instruction),
            patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
            patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
        ):
            return await _one_candidate_loop(0, task, str(tmp_path), sem)

    result = asyncio.run(run())
    assert result.outcome == "reject"
    assert result.alignment_regen_count == 0  # no regen on vague
    # At α=1: single attempt before non-actionable rejection terminates.
    assert len(result.alignment_attempts) == 1


def test_one_candidate_loop_build_failure_returns_needs_fix(tmp_path):
    """Build returns context_fail → outcome=needs_fix, no alignment call."""
    import asyncio

    from craft_taskgen.steps import _BuildOutcome, _one_candidate_loop

    task = _make_candidate_test_task()
    align_calls = [0]

    async def fake_build_instruction(t, run_dir, *, feedback, cand_id, log_prefix="build"):
        return _BuildOutcome(outcome="context_fail", error_message="fake context failure")

    async def fake_judge(**kwargs):
        align_calls[0] += 1
        return _make_judge_result({"verdict": "ok", "v4_audit": {}})

    sem = asyncio.Semaphore(1)

    async def run():
        with (
            patch("craft_taskgen.steps._build_instruction", side_effect=fake_build_instruction),
            patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
        ):
            return await _one_candidate_loop(0, task, str(tmp_path), sem)

    result = asyncio.run(run())
    assert result.outcome == "needs_fix"
    assert result.needs_human_review_reason == "fake context failure"
    assert align_calls[0] == 0  # alignment never invoked


def test_orchestrator_n2_one_winner(tmp_path):
    """N=2: cand0 passes, cand1 rejects → winner=cand0; loser dir cleaned up."""
    import asyncio

    from craft_taskgen.config import Stage
    from craft_taskgen.steps import CandidateResult, _run_build_align_candidates

    task = _make_candidate_test_task()
    state = MagicMock()
    state.run_dir = str(tmp_path)

    # The candidate-loop fakes create their own dirs (mirroring real behavior
    # where _build_instruction does this). The orchestrator's
    # _cleanup_orphan_cand_dirs runs first and would remove pre-created dirs.
    cand0_dir = str(tmp_path / "t2v3-FAKEID-cand0")
    cand1_dir = str(tmp_path / "t2v3-FAKEID-cand1")

    async def fake_one_candidate(cand_id, t, run_dir, candidate_sem):
        cand_dir = cand0_dir if cand_id == 0 else cand1_dir
        os.makedirs(cand_dir, exist_ok=True)
        if cand_id == 0:
            return CandidateResult(
                cand_id=0,
                outcome="pass",
                task_dir=cand_dir,
                slug="winner-slug",
                instruction_words=42,
                build_usage=[{"tokens_in": 100, "tokens_out": 50, "model": "m", "latency_s": 0.1}],
                alignment_usage=[{"tokens_in": 200, "tokens_out": 30, "model": "m", "latency_s": 0.2}],
                alignment_attempts=[{"attempt": 1, "verdict": "ok"}],
                alignment_verdict="ok",
                alignment_reason="fine",
            )
        return CandidateResult(
            cand_id=1,
            outcome="reject",
            task_dir=cand_dir,
            alignment_verdict="vague",
            alignment_reason="loser",
        )

    sems = {
        "llm": asyncio.Semaphore(2),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    import craft_taskgen.config as _cfg

    orig_n = _cfg.BUILD_N_CANDIDATES
    _cfg.BUILD_N_CANDIDATES = 2
    try:
        with (
            patch("craft_taskgen.steps._one_candidate_loop", side_effect=fake_one_candidate),
            patch("craft_taskgen.steps._generate_task_id", return_value="FAKEID"),
            patch("craft_taskgen.steps.save_state_locked", new=AsyncMock()),
        ):
            asyncio.run(_run_build_align_candidates(task, state, "/dev/null", sems))
    finally:
        _cfg.BUILD_N_CANDIDATES = orig_n

    # Winner cand0 dir got renamed to canonical slug-based path
    canonical = str(tmp_path / "t2v3-FAKEID-winner-slug")
    assert os.path.isdir(canonical)
    assert task.task_dir == canonical
    assert not os.path.isdir(cand0_dir)  # renamed away
    assert not os.path.isdir(cand1_dir)  # cleaned up
    assert task.stage == Stage.ALIGNMENT_CHECKED
    assert task.alignment_verdict == "ok"
    assert task.alignment_regen_count == 0
    assert len(task.build_align_losers) == 1
    assert task.build_align_losers[0]["cand_id"] == 1
    # Loser usage entries also recorded
    assert len(task.llm_usage["build"]) == 1  # winner only had one
    # Wait — losers also contribute. Let me check the orchestrator.
    # Both candidates' usage entries are appended. cand0 build_usage=1, cand1 build_usage=0.
    # So total = 1 (winner) + 0 (loser) = 1.


def test_orchestrator_n2_random_winner_among_passers(tmp_path):
    """N=2: both pass → uniform random selection between them. Patch random.choice."""
    import asyncio

    from craft_taskgen.config import Stage
    from craft_taskgen.steps import CandidateResult, _run_build_align_candidates

    state = MagicMock()
    state.run_dir = str(tmp_path)

    cand0_dir = str(tmp_path / "t2v3-FAKEID-cand0")
    cand1_dir = str(tmp_path / "t2v3-FAKEID-cand1")

    async def fake_one_candidate(cand_id, t, run_dir, candidate_sem):
        cand_dir = cand0_dir if cand_id == 0 else cand1_dir
        os.makedirs(cand_dir, exist_ok=True)
        return CandidateResult(
            cand_id=cand_id,
            outcome="pass",
            task_dir=cand_dir,
            slug=f"slug-{cand_id}",
            alignment_verdict="ok",
        )

    sems = {
        "llm": asyncio.Semaphore(2),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    import craft_taskgen.config as _cfg

    # Run twice with patched random.choice — once selecting cand0, once cand1.
    for selected_idx in (0, 1):
        task = _make_candidate_test_task()

        def make_choose(idx):
            def chooser(seq):
                return seq[idx]

            return chooser

        orig_n = _cfg.BUILD_N_CANDIDATES
        _cfg.BUILD_N_CANDIDATES = 2
        try:
            with (
                patch("craft_taskgen.steps._one_candidate_loop", side_effect=fake_one_candidate),
                patch("craft_taskgen.steps._generate_task_id", return_value="FAKEID"),
                patch("craft_taskgen.steps.save_state_locked", new=AsyncMock()),
                patch("random.choice", side_effect=make_choose(selected_idx)),
            ):
                asyncio.run(_run_build_align_candidates(task, state, "/dev/null", sems))
        finally:
            _cfg.BUILD_N_CANDIDATES = orig_n

        assert task.stage == Stage.ALIGNMENT_CHECKED
        # Winner's slug appears in canonical path
        assert task.task_dir.endswith(f"slug-{selected_idx}")


def test_orchestrator_n2_all_reject_yields_rejected(tmp_path):
    """N=2: both candidates reject (non-actionable) → task.stage=REJECTED."""
    import asyncio

    from craft_taskgen.config import Stage
    from craft_taskgen.steps import CandidateResult, _run_build_align_candidates

    task = _make_candidate_test_task()
    state = MagicMock()
    state.run_dir = str(tmp_path)

    async def fake_one_candidate(cand_id, t, run_dir, candidate_sem):
        return CandidateResult(
            cand_id=cand_id,
            outcome="reject",
            alignment_verdict="vague",
            alignment_reason=f"loser{cand_id}",
            alignment_v4_audit={"fixtures_encode_design_choices": False},
            alignment_attempts=[
                {"attempt": 1, "verdict": "vague", "reason": f"loser{cand_id}"},
            ],
        )

    sems = {
        "llm": asyncio.Semaphore(2),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    import craft_taskgen.config as _cfg

    orig_n = _cfg.BUILD_N_CANDIDATES
    _cfg.BUILD_N_CANDIDATES = 2
    try:
        with (
            patch("craft_taskgen.steps._one_candidate_loop", side_effect=fake_one_candidate),
            patch("craft_taskgen.steps._generate_task_id", return_value="FAKEID"),
            patch("craft_taskgen.steps.save_state_locked", new=AsyncMock()),
        ):
            asyncio.run(_run_build_align_candidates(task, state, "/dev/null", sems))
    finally:
        _cfg.BUILD_N_CANDIDATES = orig_n

    assert task.stage == Stage.REJECTED
    assert "loser0" in task.alignment_reason
    assert "loser1" in task.alignment_reason
    assert len(task.build_align_losers) == 2
    # Telemetry parity with success path: representative verdict + per-cand attempts
    # are surfaced on task.* so dashboards/status see why this rejected.
    assert task.alignment_verdict == "vague"
    assert task.alignment_v4_audit == {"fixtures_encode_design_choices": False}
    assert len(task.alignment_attempts) == 2  # 1 attempt × 2 candidates
    assert {a["cand_id"] for a in task.alignment_attempts} == {0, 1}
    # iteration_log build_align summary entry has verdict + reason populated
    log_entries = [e for e in task.iteration_log if e.get("step") == "build_align"]
    assert len(log_entries) == 1
    assert log_entries[0]["verdict"] == "vague"
    assert "loser" in log_entries[0]["reason"]


def test_orchestrator_n2_all_needs_fix_yields_needs_fix(tmp_path):
    """N=2: both candidates needs_fix (infra error) → task.stage=NEEDS_FIX, needs_human_review."""
    import asyncio

    from craft_taskgen.config import Stage
    from craft_taskgen.steps import CandidateResult, _run_build_align_candidates

    task = _make_candidate_test_task()
    state = MagicMock()
    state.run_dir = str(tmp_path)

    async def fake_one_candidate(cand_id, t, run_dir, candidate_sem):
        return CandidateResult(
            cand_id=cand_id,
            outcome="needs_fix",
            needs_human_review_reason=f"infra failure cand{cand_id}",
        )

    sems = {
        "llm": asyncio.Semaphore(2),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    import craft_taskgen.config as _cfg

    orig_n = _cfg.BUILD_N_CANDIDATES
    _cfg.BUILD_N_CANDIDATES = 2
    try:
        with (
            patch("craft_taskgen.steps._one_candidate_loop", side_effect=fake_one_candidate),
            patch("craft_taskgen.steps._generate_task_id", return_value="FAKEID"),
            patch("craft_taskgen.steps.save_state_locked", new=AsyncMock()),
        ):
            asyncio.run(_run_build_align_candidates(task, state, "/dev/null", sems))
    finally:
        _cfg.BUILD_N_CANDIDATES = orig_n

    assert task.stage == Stage.NEEDS_FIX
    assert task.needs_human_review is True
    assert "infra failure cand0" in task.human_review_reason
    assert "infra failure cand1" in task.human_review_reason


def test_orchestrator_stage_built_runs_alignment_only(tmp_path):
    """Stage.BUILT entry (post-triage) → _run_alignment_only_for_triage, no fanout, no candidate dirs."""
    import asyncio

    from craft_taskgen.config import Stage
    from craft_taskgen.steps import _run_alignment_only_for_triage

    task = _make_candidate_test_task()
    task.stage = Stage.BUILT
    task.task_dir = str(tmp_path / "t2v3-FAKEID-existing-slug")
    os.makedirs(task.task_dir, exist_ok=True)
    state = MagicMock()
    state.run_dir = str(tmp_path)

    async def fake_judge(**kwargs):
        return _make_judge_result({"verdict": "ok", "reason": "fine", "v4_audit": {}})

    sems = {
        "llm": asyncio.Semaphore(2),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
        patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
        patch("craft_taskgen.steps.save_state_locked", new=AsyncMock()),
    ):
        asyncio.run(_run_alignment_only_for_triage(task, state, "/dev/null", sems))

    assert task.stage == Stage.ALIGNMENT_CHECKED
    # No cand dirs created (no fanout)
    cand_dirs = [d for d in os.listdir(tmp_path) if "cand" in d]
    assert cand_dirs == []
    # task_dir preserved
    assert task.task_dir == str(tmp_path / "t2v3-FAKEID-existing-slug")


def test_profile_build_n_candidates_loaded_and_clamped(tmp_path):
    """PipelineProfile.apply() propagates build_n_candidates and clamps at >4."""
    import craft_taskgen.config as _cfg
    from craft_taskgen.config import PipelineProfile

    orig = _cfg.BUILD_N_CANDIDATES

    try:
        # default (Apr 25 2026 update: experiment config N=3, α=1, r=2)
        PipelineProfile().apply()
        assert _cfg.BUILD_N_CANDIDATES == 3

        # explicit 2 (legacy default)
        PipelineProfile(build_n_candidates=2).apply()
        assert _cfg.BUILD_N_CANDIDATES == 2

        # clamp at >4
        PipelineProfile(build_n_candidates=10).apply()
        assert _cfg.BUILD_N_CANDIDATES == 4

        # clamp at <1
        PipelineProfile(build_n_candidates=0).apply()
        assert _cfg.BUILD_N_CANDIDATES == 1
    finally:
        _cfg.BUILD_N_CANDIDATES = orig


def test_profile_alignment_max_retries_loaded_and_clamped():
    """PipelineProfile.apply() propagates alignment_max_retries and clamps at [1,5]."""
    import craft_taskgen.config as _cfg
    from craft_taskgen.config import PipelineProfile

    orig = _cfg.ALIGNMENT_MAX_RETRIES
    try:
        # default (Apr 25 2026: experiment config α=1)
        PipelineProfile().apply()
        assert _cfg.ALIGNMENT_MAX_RETRIES == 1

        # explicit 3 (prior production)
        PipelineProfile(alignment_max_retries=3).apply()
        assert _cfg.ALIGNMENT_MAX_RETRIES == 3

        # clamp at >5
        PipelineProfile(alignment_max_retries=99).apply()
        assert _cfg.ALIGNMENT_MAX_RETRIES == 5

        # clamp at <1
        PipelineProfile(alignment_max_retries=0).apply()
        assert _cfg.ALIGNMENT_MAX_RETRIES == 1
    finally:
        _cfg.ALIGNMENT_MAX_RETRIES = orig


def test_profile_max_build_regens_per_candidate_loaded_and_clamped():
    """PipelineProfile.apply() propagates max_build_regens_per_candidate and clamps at [0,3]."""
    import craft_taskgen.config as _cfg
    from craft_taskgen.config import PipelineProfile

    orig = _cfg.MAX_BUILD_REGENS_PER_CANDIDATE
    try:
        # default (Apr 25 2026: experiment config r=2)
        PipelineProfile().apply()
        assert _cfg.MAX_BUILD_REGENS_PER_CANDIDATE == 2

        # explicit 1 (prior production behavior)
        PipelineProfile(max_build_regens_per_candidate=1).apply()
        assert _cfg.MAX_BUILD_REGENS_PER_CANDIDATE == 1

        # explicit 0 (no rebuilds, accept-or-reject on first alignment)
        PipelineProfile(max_build_regens_per_candidate=0).apply()
        assert _cfg.MAX_BUILD_REGENS_PER_CANDIDATE == 0

        # clamp at >3
        PipelineProfile(max_build_regens_per_candidate=99).apply()
        assert _cfg.MAX_BUILD_REGENS_PER_CANDIDATE == 3

        # clamp at <0
        PipelineProfile(max_build_regens_per_candidate=-1).apply()
        assert _cfg.MAX_BUILD_REGENS_PER_CANDIDATE == 0
    finally:
        _cfg.MAX_BUILD_REGENS_PER_CANDIDATE = orig


def test_one_candidate_loop_two_rebuilds_then_pass(tmp_path):
    """leaked → rebuild → leaked → rebuild → ok → outcome=pass, regen_count=2.

    Exercises the bounded while-loop with multiple rebuilds at default α=1 r=2.
    """
    import asyncio

    from craft_taskgen.steps import _one_candidate_loop

    task = _make_candidate_test_task()
    judge_call_count = [0]

    async def fake_judge(**kwargs):
        # Sequence: leaked, leaked, ok (3 alignment evaluations × 1 attempt each)
        judge_call_count[0] += 1
        if judge_call_count[0] <= 2:
            return _make_judge_result(
                {
                    "verdict": "leaked",
                    "reason": "still leaks",
                    "v4_audit": {},
                    "leakage_evidence": [f"_leak{judge_call_count[0]}"],
                }
            )
        return _make_judge_result(
            {"verdict": "ok", "reason": "clean after 2 rebuilds", "v4_audit": {}, "leakage_evidence": []}
        )

    build_call_count = [0]

    async def fake_build_instruction(t, run_dir, *, feedback, cand_id, log_prefix="build"):
        build_call_count[0] += 1
        out = _fake_build_outcome_ok()
        out.task_dir = str(tmp_path / f"t2v3-FAKEID-cand{cand_id}")
        os.makedirs(out.task_dir, exist_ok=True)
        return out

    sem = asyncio.Semaphore(1)

    async def run():
        with (
            patch("craft_taskgen.steps._build_instruction", side_effect=fake_build_instruction),
            patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
            patch("craft_taskgen.steps._fetch_alignment_context", side_effect=_fake_fetch_alignment_context),
        ):
            return await _one_candidate_loop(0, task, str(tmp_path), sem)

    result = asyncio.run(run())
    assert result.outcome == "pass"
    assert result.alignment_regen_count == 2
    assert build_call_count[0] == 3  # initial + 2 rebuilds
    # 3 alignment evaluations × 1 attempt each
    assert len(result.alignment_attempts) == 3
    assert len(result.build_usage) == 3


def test_step_evaluate_parallel():
    """step_evaluate processes candidates concurrently."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_evaluate

    state = PipelineState(created="2026-04-06")
    for i in range(5):
        state.tasks[f"t{i}"] = TaskState(
            task_id=f"t{i}",
            repo="fastapi",
            commit_sha=f"abc{i}",
            base_sha="base123",
            merge_base_sha="base123",
            description=f"feat {i}",
            stage=Stage.CANDIDATE,
        )

    call_count = 0

    async def fake_judge(**kwargs):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return _make_judge_result({"verdict": "accept", "reason": "Q1: integration complexity"})

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=fake_judge),
        patch("craft_taskgen.steps._fetch_evaluate_context", side_effect=_fake_fetch_context),
    ):
        asyncio.run(step_evaluate(state, "/dev/null", concurrency=3))

    assert call_count == 5
    assert all(t.stage == Stage.PROMISING for t in state.tasks.values())


def test_async_main_stop_after_select_persists_state_and_skips_evaluate(tmp_path, monkeypatch):
    import asyncio

    from craft_taskgen.config import PipelineState
    from craft_taskgen.pipeline import async_main

    candidate_file = tmp_path / "testrepo.json"
    candidate_file.write_text(json.dumps({"candidates": []}))

    selected = [
        {
            "repo": "testrepo",
            "sha": "abc12345",
            "base_sha": "base123",
            "merge_base_sha": "base123",
            "subject": "feat: add X",
            "score": 10.0,
            "_raw": {"sha": "abc12345", "base_sha": "base123", "merge_base_sha": "base123"},
        }
    ]

    async def fail_evaluate(*args, **kwargs):
        raise AssertionError("step_evaluate should not be called")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen",
            "--candidates",
            str(candidate_file),
            "--stop-after-step",
            "select",
        ],
    )
    monkeypatch.setattr("craft_taskgen.pipeline._load_env", lambda: None)
    monkeypatch.setattr("craft_taskgen.pipeline.select_candidates", lambda *args, **kwargs: selected)
    monkeypatch.setattr("craft_taskgen.pipeline.step_evaluate", fail_evaluate)

    asyncio.run(async_main())

    run_root = tmp_path / "harbor-tasks" / "craft-tools-v4" / "runs"
    state_files = list(run_root.glob("*/state.json"))
    assert len(state_files) == 1
    loaded = PipelineState.load(str(state_files[0]))
    assert "testrepo-abc12345" in loaded.tasks
    assert loaded.run_info["stop_after_step"] == "select"


def test_async_main_rejects_stop_after_step_before_from_step(tmp_path, monkeypatch):
    import asyncio

    from craft_taskgen.pipeline import async_main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen",
            "--from-step",
            "evaluate",
            "--stop-after-step",
            "select",
        ],
    )

    with pytest.raises(SystemExit):
        asyncio.run(async_main())


def test_async_main_search_rejects_stop_after_step(tmp_path, monkeypatch):
    import asyncio

    from craft_taskgen.pipeline import async_main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen",
            "--dimension",
            "search",
            "--tasks-dir",
            "tasks",
            "--output-dir",
            "out",
            "--stop-after-step",
            "select",
        ],
    )

    with pytest.raises(SystemExit):
        asyncio.run(async_main())


def test_step_triage_genuine_gap_accepts():
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_triage

    state = PipelineState(created="2026-04-06")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.OPUS_SMOKE_TESTED,
        task_dir="harbor-tasks/craft-tools-v4/t2v3-FA3-streaming",
        opus_score="3/5",
        opus_trial_dir="jobs/test/t2v3-FA3__abc",
    )

    deep_dive_result = {
        "reward": 0.0,
        "ref_tests_passed": 3,
        "ref_tests_total": 5,
        "failures": [
            {"test_name": "test_a", "classification": "keep", "evidence": "clear"},
            {"test_name": "test_b", "classification": "keep", "evidence": "clear"},
        ],
        "overall_assessment": "Both genuine capability gaps",
    }

    dispatcher = _make_triage_dispatcher(deep_dive_result)

    async def fake_fetch_deep_dive(task_dir, trial_dir):
        return _empty_deep_dive_context()

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=dispatcher),
        patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
        patch("os.path.isdir", return_value=True),
    ):
        asyncio.run(step_opus_triage(state, "/dev/null"))

    assert state.tasks["t1"].stage == Stage.OPUS_TRIAGED


def test_step_report(capsys):
    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_report

    state = PipelineState(created="2026-04-06")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="a",
        commit_sha="x",
        base_sha="base123",
        merge_base_sha="base123",
        description="d",
        stage=Stage.ACCEPTED,
    )
    state.tasks["t2"] = TaskState(
        task_id="t2",
        repo="b",
        commit_sha="y",
        base_sha="base123",
        merge_base_sha="base123",
        description="d",
        stage=Stage.REJECTED,
    )
    state.tasks["t3"] = TaskState(
        task_id="t3",
        repo="c",
        commit_sha="z",
        base_sha="base123",
        merge_base_sha="base123",
        description="d",
        stage=Stage.NEEDS_FIX,
        needs_human_review=True,
        human_review_reason="Too easy",
    )

    step_report(state)
    out = capsys.readouterr().out
    assert "ACCEPTED" in out
    assert "REJECTED" in out
    assert "NEEDS_FIX" in out


def test_run_claude_async_constructs_correct_command():
    """run_claude_async passes same flags as sync version."""
    import asyncio

    from craft_taskgen.claude_cli import run_claude_async

    fake_stdout = json.dumps({"result": "ok", "session_id": "s1"}).encode()

    async def fake_communicate():
        return (fake_stdout, b"")

    async def run():
        mock_proc = MagicMock()
        mock_proc.communicate = fake_communicate
        mock_proc.returncode = 0

        async def fake_create(*args, **kwargs):
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create) as mock_exec:
            result = await run_claude_async("test prompt", max_turns=10)

        args = mock_exec.call_args[0]
        assert "-p" in args
        assert "test prompt" in args
        assert "--permission-mode" in args
        assert "auto" in args
        assert "--output-format" in args
        assert "json" in args
        assert "--max-turns" in args
        assert "10" in args
        assert result["result"] == "ok"

    asyncio.run(run())


def test_run_claude_async_timeout():
    """run_claude_async returns error on timeout."""
    import asyncio

    from craft_taskgen.claude_cli import run_claude_async

    async def run():
        mock_proc = MagicMock()
        mock_proc.kill = MagicMock()

        async def fake_wait():
            pass

        mock_proc.wait = fake_wait

        async def fake_communicate():
            raise asyncio.TimeoutError()

        mock_proc.communicate = fake_communicate

        async def fake_create(*args, **kwargs):
            return mock_proc

        async def fake_sleep(seconds):
            pass

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_create),
            patch("asyncio.sleep", side_effect=fake_sleep),
        ):
            result = await run_claude_async("slow", timeout=1)
        assert result["error"] == "timeout"

    asyncio.run(run())


def test_run_claude_async_retry_logs_stdout_when_stderr_empty(capsys):
    """Transient retry logs should surface stdout when stderr is empty."""
    import asyncio

    from craft_taskgen.claude_cli import run_claude_async

    async def run():
        attempts = 0

        async def fake_sleep(seconds):
            pass

        async def fake_create(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            mock_proc = MagicMock()
            mock_proc.returncode = 1 if attempts == 1 else 0

            async def fake_communicate():
                if attempts == 1:
                    return (b'{"subtype":"gateway_busy","message":"try again"}', b"")
                return (json.dumps({"result": "ok"}).encode(), b"")

            mock_proc.communicate = fake_communicate
            return mock_proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_create),
            patch("asyncio.sleep", side_effect=fake_sleep),
        ):
            result = await run_claude_async("retry please")

        assert result["result"] == "ok"

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "Transient error (exit=1)" in out
    assert "stdout:" in out
    assert "gateway_busy" in out


def test_run_claude_async_structured_400_is_not_retried():
    """Claude structured 400 errors are client-side and should not retry."""
    import asyncio

    from craft_taskgen.claude_cli import run_claude_async

    async def run():
        attempts = 0

        async def fake_create(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            mock_proc = MagicMock()
            mock_proc.returncode = 1

            async def fake_communicate():
                stdout = {
                    "type": "result",
                    "subtype": "success",
                    "is_error": True,
                    "api_error_status": 400,
                    "num_turns": 1,
                    "result": (
                        "API Error: 400 "
                        '{"error":{"message":"{\\"message\\":\\"bad request from gateway\\"}"}}'
                    ),
                }
                return (json.dumps(stdout).encode(), b"")

            mock_proc.communicate = fake_communicate
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
            result = await run_claude_async("bad request")

        assert attempts == 1
        assert result["is_error"] is True
        assert result["api_error_status"] == 400
        assert result["error_detail"] == "bad request from gateway"

    asyncio.run(run())


def test_parse_score_ratio():
    from craft_taskgen.steps import _parse_score_ratio

    assert _parse_score_ratio("7/10") == 0.7
    assert _parse_score_ratio("10/10") == 1.0
    assert _parse_score_ratio("0/5") == 0.0
    assert _parse_score_ratio("") is None
    assert _parse_score_ratio("error") is None
    assert _parse_score_ratio("infra_failure") is None


def test_compare_and_accept_no_easiness_flag_by_default():
    """Compare step accepts without auto-flagging based on scores alone.
    Easiness flag is set by the reviewer (qualitative, not score-based)."""
    from craft_taskgen.config import Stage, TaskState
    from craft_taskgen.steps import _compare_and_accept

    task = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        opus_score="10/10",
    )
    _compare_and_accept(task)
    assert task.stage == Stage.ACCEPTED
    assert task.easiness_concern is False  # no score-based auto-flag


def test_compare_and_accept_accepts_on_opus_score_alone():
    """Haiku step was dropped — comparison is Opus-only now, always accept
    unless triage routed the task to a fix stage first."""
    from craft_taskgen.config import Stage, TaskState
    from craft_taskgen.steps import _compare_and_accept

    task = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        opus_score="6/10",
    )
    _compare_and_accept(task)
    assert task.stage == Stage.ACCEPTED


def test_compare_and_accept_propagates_easiness_reason():
    """Easiness concern set by reviewer survives into the final accept log."""
    from craft_taskgen.config import Stage, TaskState
    from craft_taskgen.steps import _compare_and_accept

    task = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        opus_score="10/10",
        easiness_concern=True,
        easiness_reason="agent wrote code without exploration",
    )
    _compare_and_accept(task)
    assert task.stage == Stage.ACCEPTED
    assert task.easiness_concern is True
    assert task.easiness_reason == "agent wrote code without exploration"


def test_step_smoke_retries_infra_failure():
    """Smoke test retries on infra failure, succeeds on second attempt."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_smoke

    state = PipelineState(created="2026-04-10")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.ORACLE_CHECKED,
        task_dir="harbor-tasks/craft-tools-v4/t2v3-FA3-streaming",
    )
    call_count = 0

    async def fake_smoke(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"infra_failure": True, "exception": "apiKeySource=none"}
        return {"score": "3/5", "trial_dir": "jobs/test/trial"}

    async def fake_sleep(seconds):
        pass  # skip real wait in tests

    with (
        patch("craft_taskgen.steps._run_smoke_async", side_effect=fake_smoke),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        asyncio.run(step_opus_smoke(state, "/dev/null"))

    assert call_count == 2
    assert state.tasks["t1"].stage == Stage.OPUS_SMOKE_TESTED


def test_step_smoke_gives_up_after_max_retries():
    """Smoke test gives up after MAX_SMOKE_RETRIES infra failures."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_smoke

    state = PipelineState(created="2026-04-10")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.ORACLE_CHECKED,
        task_dir="harbor-tasks/craft-tools-v4/t2v3-FA3-streaming",
    )

    async def always_fail(*args, **kwargs):
        return {"infra_failure": True, "exception": "apiKeySource=none"}

    async def fake_sleep(seconds):
        pass  # skip real wait in tests

    with (
        patch("craft_taskgen.steps._run_smoke_async", side_effect=always_fail),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        asyncio.run(step_opus_smoke(state, "/dev/null"))

    assert state.tasks["t1"].stage == Stage.NEEDS_FIX


def test_diagnostic_helpers(tmp_path):
    """_next_diagnostic_path creates monotonically numbered files."""
    import os

    from craft_taskgen.diagnostics import _next_diagnostic_path, _write_diagnostic

    task_dir = str(tmp_path / "test-task")
    os.makedirs(task_dir)

    p1 = _next_diagnostic_path(task_dir, "triage_haiku")
    assert p1.endswith("001_triage_haiku.md")
    _write_diagnostic(p1, "# First")

    p2 = _next_diagnostic_path(task_dir, "review_haiku")
    assert p2.endswith("002_review_haiku.md")
    _write_diagnostic(p2, "# Second")

    p3 = _next_diagnostic_path(task_dir, "fix")
    assert p3.endswith("003_fix.md")


def test_triage_writes_diagnostic_files(tmp_path):
    """Triage writes Opus skip/keep diagnostic (triage_<label>.md) plus a
    fairness_review_<label>.md diagnostic for every task that reaches the
    deep-dive path. Legacy dual-DD disagreement file is gone."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_triage

    task_dir = str(tmp_path / "t2v3-FA3-streaming")
    os.makedirs(task_dir)

    state = PipelineState(created="2026-04-10")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.OPUS_SMOKE_TESTED,
        task_dir=task_dir,
        opus_score="3/5",
        opus_trial_dir="jobs/test/t2v3-FA3__abc",
    )

    deep_dive_result = {
        "reward": 0.6,
        "ref_tests_passed": 3,
        "ref_tests_total": 5,
        "failures": [
            {
                "test_name": "test_a",
                "classification": "keep",
                "evidence": "Agent missed the streaming endpoint entirely",
            },
        ],
        "overall_assessment": "One genuine gap",
    }

    dispatcher = _make_triage_dispatcher(deep_dive_result)

    async def fake_fetch_deep_dive(task_dir, trial_dir):
        return _empty_deep_dive_context()

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=dispatcher),
        patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
        patch("os.path.isdir", return_value=True),
    ):
        asyncio.run(step_opus_triage(state, "/dev/null"))

    diag_dir = os.path.join(task_dir, "diagnostics")
    assert os.path.isdir(diag_dir)
    diag_files = sorted(os.listdir(diag_dir))
    assert any("triage_opus" in f for f in diag_files)
    assert any("fairness_review" in f for f in diag_files)
    # Legacy dual_dd_disagree diagnostic is gone for good
    assert not any("dual_dd_disagree" in f for f in diag_files)
    # Opus skip/keep diagnostic includes the agent's evidence
    triage_file = [f for f in diag_files if "triage_opus" in f][0]
    content = open(os.path.join(diag_dir, triage_file)).read()
    assert "streaming endpoint" in content
    # Fairness review file records the severity verdict
    review_file = [f for f in diag_files if "fairness_review" in f][0]
    review_content = open(os.path.join(diag_dir, review_file)).read()
    assert "Severity" in review_content


def test_triage_reviewer_major_evidence_triggers_build_regen(tmp_path):
    """When the fairness reviewer returns severity=major with both an
    evidence_quote and evidence_test, triage routes the task back to
    Build. Verifies triage_regen_count increments, instruction.md is
    rewritten, stage advances to BUILT with pending_fix_type=instruction,
    and all reviewer_concern_* fields are populated."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_triage

    task_dir = str(tmp_path / "task-x")
    os.makedirs(task_dir)
    original_instruction = "Fix the xyz feature."
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write(original_instruction)

    state = PipelineState(created="2026-04-23")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="myrepo",
        commit_sha="abc",
        base_sha="base",
        merge_base_sha="base",
        description="feat",
        stage=Stage.OPUS_SMOKE_TESTED,
        task_dir=task_dir,
        opus_score="F2P 0/2, P2P 10/10",
        opus_trial_dir="jobs/test/task-x__abc",
        eval_reason="test reason",
        eval_instruction_sketch="test sketch",
    )

    # DD keeps the failing test (no skip verdict); regen decision comes
    # from the reviewer, not from Opus.
    deep_dive_result = {
        "reward": 0.5,
        "failures": [
            {
                "test_name": "test_alpha",
                "classification": "keep",
                "evidence": "test exercises feature",
            },
        ],
        "overall_assessment": "One genuine-looking failure",
    }
    fairness_output = {
        "severity": "major",
        "reason": "Instruction omits the async variant requirement",
        "evidence_quote": "Fix the xyz feature.",
        "evidence_test": "test_alpha",
    }
    build_output = {
        "instruction_md": "Fix the xyz feature and also its async variant.",
        "task_slug": "xyz-async",
    }

    dispatcher = _make_triage_dispatcher(
        deep_dive_result, build_output=build_output, fairness_output=fairness_output
    )

    async def fake_fetch_deep_dive(task_dir, trial_dir):
        return _empty_deep_dive_context()

    def fake_fetch_build(repo, sha, merge_base_sha, ctx):
        return {
            "repo_map": "[fake repo map]",
            "diff": "[fake diff]",
            "reference_test_bodies": [("test_alpha.py", "def test_alpha(): ...")],
            "instruction_template": "Task description:\n(the task description)\n",
            "instruction_example": "[fake example]",
        }

    def fake_failed_tests(trial_dir):
        return {"test_alpha"}

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=dispatcher),
        patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
        patch("craft_taskgen.steps._fetch_build_context", side_effect=fake_fetch_build),
        patch("craft_taskgen.steps._load_actually_failed_tests", side_effect=fake_failed_tests),
        patch("os.path.isdir", return_value=True),
    ):
        asyncio.run(step_opus_triage(state, "/dev/null"))

    task = state.tasks["t1"]
    assert task.triage_regen_count == 1, f"expected triage_regen_count=1, got {task.triage_regen_count}"
    assert task.stage == Stage.BUILT, f"expected Stage.BUILT, got {task.stage}"
    assert task.pending_fix_type == "instruction"
    assert task.reviewer_concern_flag is True
    assert task.reviewer_concern_severity == "major"
    assert task.reviewer_concern_evidence_quote == "Fix the xyz feature."
    assert task.reviewer_concern_evidence_test == "test_alpha"
    new_instruction = open(os.path.join(task_dir, "instruction.md")).read()
    assert "async variant" in new_instruction
    assert new_instruction != original_instruction
    assert "build_regen" in task.llm_usage
    assert len(task.llm_usage["build_regen"]) == 1
    steps_in_log = [e.get("step") for e in task.iteration_log]
    assert any("Opus_build_regen" in s for s in steps_in_log)


def test_triage_reviewer_major_missing_evidence_sets_flag_no_regen(tmp_path):
    """Reviewer returns severity=major but without evidence_quote or
    evidence_test — should set the soft reviewer_concern_flag and
    continue on Opus's keep/skip verdict without triggering Build
    regen. Task accepts (keep verdict = genuine capability gap)."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_triage

    task_dir = str(tmp_path / "task-y")
    os.makedirs(task_dir)
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write("Original instruction.")

    state = PipelineState(created="2026-04-23")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="myrepo",
        commit_sha="abc",
        base_sha="base",
        merge_base_sha="base",
        description="feat",
        stage=Stage.OPUS_SMOKE_TESTED,
        task_dir=task_dir,
        opus_score="F2P 1/2, P2P 10/10",
        opus_trial_dir="jobs/test/task-y__abc",
    )

    deep_dive_result = {
        "reward": 0.5,
        "failures": [
            {"test_name": "test_beta", "classification": "keep", "evidence": "real"},
        ],
        "overall_assessment": "Genuine gap",
    }
    fairness_output = {
        "severity": "major",  # claims major but no quote/test
        "reason": "Vague concern",
        "evidence_quote": "",
        "evidence_test": "",
    }
    dispatcher = _make_triage_dispatcher(deep_dive_result, fairness_output=fairness_output)

    async def fake_fetch_deep_dive(task_dir, trial_dir):
        return _empty_deep_dive_context()

    def fake_failed_tests(trial_dir):
        return {"test_beta"}

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=dispatcher),
        patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
        patch("craft_taskgen.steps._load_actually_failed_tests", side_effect=fake_failed_tests),
        patch("os.path.isdir", return_value=True),
    ):
        asyncio.run(step_opus_triage(state, "/dev/null"))

    task = state.tasks["t1"]
    # No Build regen fired
    assert task.triage_regen_count == 0
    assert "build_regen" not in task.llm_usage
    # Flag is populated as soft signal; task accepts (keep verdict)
    assert task.reviewer_concern_flag is True
    assert task.reviewer_concern_severity == "major"
    assert task.stage == Stage.OPUS_TRIAGED


def test_triage_reviewer_severity_none_no_flag_no_regen(tmp_path):
    """Reviewer returns severity=none → no flag, no regen, accept."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_triage

    task_dir = str(tmp_path / "task-z")
    os.makedirs(task_dir)

    state = PipelineState(created="2026-04-23")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="myrepo",
        commit_sha="abc",
        base_sha="base",
        merge_base_sha="base",
        description="feat",
        stage=Stage.OPUS_SMOKE_TESTED,
        task_dir=task_dir,
        opus_score="F2P 1/2, P2P 10/10",
        opus_trial_dir="jobs/test/task-z__abc",
    )

    deep_dive_result = {
        "reward": 0.5,
        "failures": [
            {"test_name": "test_gamma", "classification": "keep", "evidence": "gap"},
        ],
        "overall_assessment": "Genuine capability gap",
    }
    dispatcher = _make_triage_dispatcher(deep_dive_result)  # default fairness=none

    async def fake_fetch_deep_dive(task_dir, trial_dir):
        return _empty_deep_dive_context()

    def fake_failed_tests(trial_dir):
        return {"test_gamma"}

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=dispatcher),
        patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
        patch("craft_taskgen.steps._load_actually_failed_tests", side_effect=fake_failed_tests),
        patch("os.path.isdir", return_value=True),
    ):
        asyncio.run(step_opus_triage(state, "/dev/null"))

    task = state.tasks["t1"]
    assert task.reviewer_concern_flag is False
    assert task.reviewer_concern_severity == "none"
    assert task.triage_regen_count == 0
    assert task.stage == Stage.OPUS_TRIAGED


def test_triage_reward1_easiness_flag_triggers_regen(tmp_path):
    """reward=1.0 trial with easiness_flag fired (low Grep+Read count)
    routes to Build regen with prescriptive-instruction feedback
    BEFORE accepting. First pass: regen fires, stage becomes BUILT.
    Test covers the reward=1 fast-path's new easiness regen branch."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_triage

    task_dir = str(tmp_path / "task-easy")
    os.makedirs(task_dir)
    original = "Fix the bug in MyClass.my_method using a visited-set."
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write(original)

    state = PipelineState(created="2026-04-23")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="myrepo",
        commit_sha="abc",
        base_sha="base",
        merge_base_sha="base",
        description="feat",
        stage=Stage.OPUS_SMOKE_TESTED,
        task_dir=task_dir,
        opus_score="F2P 1/1, P2P 5/5",
        opus_trial_dir="jobs/test/task-easy__abc",
        eval_reason="sanity test",
        eval_instruction_sketch="sketch",
    )

    # Context with reward=1 and trivial trajectory (1 grep/read call).
    def _ctx():
        c = _empty_deep_dive_context()
        c["reward_json"] = json.dumps({"reward": 1.0, "f2p_passed": 1, "f2p_total": 1})
        # Single-row tool sequence with one Read → grep_read=1
        c["harbor_lab_tool_sequence"] = "| 1 | Read | /code/foo.py |"
        c["harbor_lab_tool_sequence_full"] = "| 1 | Read | /code/foo.py |"
        return c

    async def fake_fetch_deep_dive(task_dir_arg, trial_dir_arg):
        return _ctx()

    build_output = {
        "instruction_md": "A more abstract rewrite without named files or classes.",
        "task_slug": "abstract",
    }
    dispatcher = _make_triage_dispatcher({}, build_output=build_output)

    def fake_fetch_build(repo, sha, merge_base_sha, ctx):
        return {
            "repo_map": "[repo map]",
            "diff": "[diff]",
            "reference_test_bodies": [],
            "instruction_template": "Task description:\n(the task description)\n",
            "instruction_example": "[example]",
        }

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=dispatcher),
        patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
        patch("craft_taskgen.steps._fetch_build_context", side_effect=fake_fetch_build),
        patch("os.path.isdir", return_value=True),
    ):
        asyncio.run(step_opus_triage(state, "/dev/null"))

    task = state.tasks["t1"]
    assert task.easiness_flag is True, "easiness should fire on grep_read=1"
    assert task.triage_regen_count == 1, "regen budget consumed"
    assert task.stage == Stage.BUILT, "routed back to BUILT for re-alignment+smoke"
    assert task.pending_fix_type == "instruction"
    new_instr = open(os.path.join(task_dir, "instruction.md")).read()
    assert "abstract rewrite" in new_instr
    # iteration log has the easiness-triggered build_regen entry
    triggers = [e.get("trigger") for e in task.iteration_log if "build_regen" in str(e.get("step", ""))]
    assert "easiness" in triggers


def test_triage_reward1_easiness_budget_exhausted_shelves_task(tmp_path):
    """reward=1.0 + easiness_flag on second pass (triage_regen_count
    already == MAX_TRIAGE_REGENS) shelves as NEEDS_FIX — task is
    structurally too easy, regen didn't help."""
    import asyncio
    import os

    import craft_taskgen.config as _cfg
    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_triage

    task_dir = str(tmp_path / "task-easy2")
    os.makedirs(task_dir)
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write("Fix the thing.")

    state = PipelineState(created="2026-04-23")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="myrepo",
        commit_sha="abc",
        base_sha="base",
        merge_base_sha="base",
        description="feat",
        stage=Stage.OPUS_SMOKE_TESTED,
        task_dir=task_dir,
        opus_score="F2P 1/1, P2P 5/5",
        opus_trial_dir="jobs/test/task-easy2__abc",
        triage_regen_count=_cfg.MAX_TRIAGE_REGENS,  # budget already spent
    )

    def _ctx():
        c = _empty_deep_dive_context()
        c["reward_json"] = json.dumps({"reward": 1.0, "f2p_passed": 1, "f2p_total": 1})
        c["harbor_lab_tool_sequence"] = "| 1 | Read | /code/bar.py |"
        c["harbor_lab_tool_sequence_full"] = "| 1 | Read | /code/bar.py |"
        return c

    async def fake_fetch_deep_dive(task_dir_arg, trial_dir_arg):
        return _ctx()

    dispatcher = _make_triage_dispatcher({})

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=dispatcher),
        patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
        patch("os.path.isdir", return_value=True),
    ):
        asyncio.run(step_opus_triage(state, "/dev/null"))

    task = state.tasks["t1"]
    assert task.easiness_flag is True
    assert task.triage_regen_count == _cfg.MAX_TRIAGE_REGENS  # unchanged
    assert task.stage == Stage.NEEDS_FIX
    assert task.needs_human_review is True
    assert "easiness" in task.human_review_reason.lower()


def test_triage_opus_skip_writes_skip_file_and_rescores(tmp_path):
    """Opus emits a `skip` verdict for a failing F2P test → pipeline
    auto-appends to f2p_skip.txt and re-scores against the trial.
    Verifies the skip file is written even when the reviewer is quiet."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_triage

    task_dir = str(tmp_path / "task-skip")
    os.makedirs(os.path.join(task_dir, "tests"))

    state = PipelineState(created="2026-04-23")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="myrepo",
        commit_sha="abc",
        base_sha="base",
        merge_base_sha="base",
        description="feat",
        stage=Stage.OPUS_SMOKE_TESTED,
        task_dir=task_dir,
        opus_score="F2P 2/3, P2P 10/10",
        opus_trial_dir="jobs/test/task-skip__abc",
        f2p_tests=["test_unfair", "test_fair_a", "test_fair_b"],
    )

    deep_dive_result = {
        "reward": 0.67,
        "failures": [
            {
                "test_name": "test_unfair",
                "classification": "skip",
                "evidence": "tests ruff formatting, unrelated to feature",
            },
        ],
        "overall_assessment": "Unrelated regression test bundled in diff",
    }
    dispatcher = _make_triage_dispatcher(deep_dive_result)

    async def fake_fetch_deep_dive(task_dir, trial_dir):
        return _empty_deep_dive_context()

    def fake_failed_tests(trial_dir):
        return {"test_unfair"}

    # _rescore_trial is a deterministic helper that runs the scorer;
    # mock it so the test doesn't need a real trial dir.
    def fake_rescore(task_dir_arg, trial_dir_arg):
        # simulate the skip bringing reward to 1.0
        os.makedirs(os.path.join(trial_dir_arg, "verifier"), exist_ok=True)
        with open(os.path.join(trial_dir_arg, "verifier", "reward.json"), "w") as f:
            import json as _json

            _json.dump(
                {
                    "reward": 1.0,
                    "f2p_passed": 2,
                    "f2p_total": 2,
                    "f2p_skipped": 1,
                    "p2p_passed": 10,
                    "p2p_total": 10,
                },
                f,
            )
        return True

    trial_dir_path = str(tmp_path / "trial")
    state.tasks["t1"].opus_trial_dir = trial_dir_path
    os.makedirs(trial_dir_path)

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=dispatcher),
        patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
        patch("craft_taskgen.steps._load_actually_failed_tests", side_effect=fake_failed_tests),
        patch("craft_taskgen.steps._rescore_trial", side_effect=fake_rescore),
    ):
        asyncio.run(step_opus_triage(state, "/dev/null"))

    task = state.tasks["t1"]
    # Task accepts after auto-skip+rescore brings reward to 1.0
    assert task.stage == Stage.OPUS_TRIAGED
    # Skip file written with the unfair test
    skip_file = os.path.join(task_dir, "tests", "f2p_skip.txt")
    assert os.path.isfile(skip_file)
    contents = open(skip_file).read()
    assert "test_unfair" in contents
    assert "skip:" in contents


def test_triage_regen_budget_exhausted_shelves_task(tmp_path):
    """When task.triage_regen_count already equals MAX_TRIAGE_REGENS, a
    fairness review returning severity=major must not burn another regen
    — it shelves the task as NEEDS_FIX with needs_human_review=True."""
    import asyncio
    import os

    import craft_taskgen.config as _cfg
    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import step_opus_triage

    task_dir = str(tmp_path / "task-y")
    os.makedirs(task_dir)
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write("Existing instruction.")

    state = PipelineState(created="2026-04-23")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="myrepo",
        commit_sha="abc",
        base_sha="base",
        merge_base_sha="base",
        description="feat",
        stage=Stage.OPUS_SMOKE_TESTED,
        task_dir=task_dir,
        opus_score="F2P 0/1, P2P 10/10",
        opus_trial_dir="jobs/test/task-y__abc",
        eval_reason="r",
        eval_instruction_sketch="s",
        triage_regen_count=_cfg.MAX_TRIAGE_REGENS,  # already exhausted
    )

    deep_dive_result = {
        "reward": 0.0,
        "failures": [
            {
                "test_name": "test_beta",
                "classification": "keep",
                "evidence": "Test expects unstated behavior",
            },
        ],
        "overall_assessment": "Still bad",
    }
    fairness_output = {
        "severity": "major",
        "reason": "Instruction omits a required detail",
        "evidence_quote": "Existing instruction.",
        "evidence_test": "test_beta",
    }

    dispatcher = _make_triage_dispatcher(deep_dive_result, fairness_output=fairness_output)

    async def fake_fetch_deep_dive(task_dir, trial_dir):
        return _empty_deep_dive_context()

    def fake_failed_tests(trial_dir):
        return {"test_beta"}

    def fake_fetch_build(repo, sha, merge_base_sha, ctx):
        return {
            "repo_map": "[fake repo map]",
            "diff": "[fake diff]",
            "reference_test_bodies": [],
            "instruction_template": "Task description:\n(the task description)\n",
            "instruction_example": "[fake example]",
        }

    with (
        patch("craft_taskgen.steps.llm_judge.judge", side_effect=dispatcher),
        patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
        patch("craft_taskgen.steps._fetch_build_context", side_effect=fake_fetch_build),
        patch("craft_taskgen.steps._load_actually_failed_tests", side_effect=fake_failed_tests),
        patch("os.path.isdir", return_value=True),
    ):
        asyncio.run(step_opus_triage(state, "/dev/null"))

    task = state.tasks["t1"]
    # Regen count did not increment (budget already exhausted)
    assert task.triage_regen_count == _cfg.MAX_TRIAGE_REGENS
    # Task was shelved
    assert task.stage == Stage.NEEDS_FIX
    assert task.needs_human_review is True
    assert "regen" in task.human_review_reason.lower()
    # No build_regen LLM call recorded
    assert "build_regen" not in task.llm_usage


def test_run_task_pipeline_flows_to_accepted():
    """A task flows through all steps independently to ACCEPTED."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import run_task_pipeline

    state = PipelineState(created="2026-04-10")
    task = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.ALIGNMENT_CHECKED,
        task_dir="harbor-tasks/craft-tools-v4/t2v3-FA3-streaming",
    )
    state.tasks["t1"] = task
    step_calls = []

    async def fake_find_tests(t, s, sf):
        step_calls.append("find_tests")
        t.stage = Stage.TESTS_DISCOVERED

    async def fake_build_dockerfile(t, s, sf):
        step_calls.append("build_dockerfile")
        t.stage = Stage.DOCKERFILE_BUILT

    async def fake_docker_classify(t, s, sf):
        step_calls.append("docker_classify")
        t.stage = Stage.F2P_P2P_CLASSIFIED

    async def fake_oracle(t, s, sf):
        step_calls.append("oracle")
        t.stage = Stage.ORACLE_CHECKED

    async def fake_smoke(t, s, sf, **kw):
        step_calls.append(f"smoke_{kw['label']}")
        setattr(t, kw["score_attr"], "3/5")
        setattr(t, kw["trial_attr"], "jobs/test")
        t.stage = kw["next_stage"]

    async def fake_triage(t, s, sf, **kw):
        step_calls.append(f"triage_{kw['label']}")
        t.stage = kw["accept_stage"]

    sems = {
        "llm": asyncio.Semaphore(4),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    async def run():
        with (
            patch("craft_taskgen.steps._run_alignment_one", side_effect=_fake_alignment_pass),
            patch("craft_taskgen.steps._run_assemble_task_dir_artifacts_one", side_effect=fake_find_tests),
            patch("craft_taskgen.steps._has_dockerfile", return_value=True),
            patch("craft_taskgen.steps._run_build_dockerfile_one", side_effect=fake_build_dockerfile),
            patch("craft_taskgen.steps._run_docker_classify_one", side_effect=fake_docker_classify),
            patch("craft_taskgen.steps._run_oracle_check_one", side_effect=fake_oracle),
            patch("craft_taskgen.steps._run_smoke_one", side_effect=fake_smoke),
            patch("craft_taskgen.steps._run_triage_one", side_effect=fake_triage),
            patch("craft_taskgen.steps._generate_summary", return_value=None),
        ):
            await run_task_pipeline(task, state, "/dev/null", sems)

    asyncio.run(run())
    assert task.stage == Stage.ACCEPTED
    assert step_calls == [
        "find_tests",
        "build_dockerfile",
        "docker_classify",
        "oracle",
        "smoke_Opus",
        "triage_Opus",
    ]


def test_run_task_pipeline_handles_triage_reset_instruction():
    """Instruction-only fix: ALIGNMENT_CHECKED → ORACLE_CHECKED fast-path skips
    assemble, dockerfile, classify, and oracle on the retry pass.

    When pending_fix_type=="instruction" and task.toml exists, the pipeline jumps
    straight from ALIGNMENT_CHECKED to ORACLE_CHECKED — assemble/dockerfile/classify/oracle
    each run exactly once (initial pass only); alignment runs twice.
    """
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import run_task_pipeline

    state = PipelineState(created="2026-04-10")
    task = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.ALIGNMENT_CHECKED,
        task_dir="harbor-tasks/craft-tools-v4/t2v3-FA3-streaming",
    )
    state.tasks["t1"] = task
    triage_count = 0
    alignment_count = 0
    assemble_count = 0
    dockerfile_count = 0
    classify_count = 0
    oracle_count = 0

    async def fake_alignment(t, s, sf, sems_arg=None):
        nonlocal alignment_count
        alignment_count += 1
        t.alignment_verdict = "ok"
        t.stage = Stage.ALIGNMENT_CHECKED

    async def fake_assemble(t, s, sf):
        nonlocal assemble_count
        assemble_count += 1
        t.stage = Stage.TESTS_DISCOVERED

    async def fake_build_dockerfile(t, s, sf):
        nonlocal dockerfile_count
        dockerfile_count += 1
        t.stage = Stage.DOCKERFILE_BUILT

    async def fake_docker_classify(t, s, sf):
        nonlocal classify_count
        classify_count += 1
        t.stage = Stage.F2P_P2P_CLASSIFIED

    async def fake_oracle(t, s, sf):
        nonlocal oracle_count
        oracle_count += 1
        t.stage = Stage.ORACLE_CHECKED

    async def fake_smoke(t, s, sf, **kw):
        setattr(t, kw["score_attr"], "3/5")
        setattr(t, kw["trial_attr"], "jobs/test")
        t.stage = kw["next_stage"]

    async def fake_triage(t, s, sf, **kw):
        nonlocal triage_count
        triage_count += 1
        if triage_count == 1 and kw["label"] == "Opus":
            # instruction fix: set pending_fix_type so pipeline takes the fast-path
            t.pending_fix_type = "instruction"
            t.stage = Stage.BUILT
        else:
            t.stage = kw["accept_stage"]

    sems = {
        "llm": asyncio.Semaphore(4),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    async def run():
        with (
            # Triage routes back to BUILT, which now goes through
            # _run_alignment_only_for_triage (alignment-only, no fanout).
            patch("craft_taskgen.steps._run_alignment_only_for_triage", side_effect=fake_alignment),
            patch("craft_taskgen.steps._run_assemble_task_dir_artifacts_one", side_effect=fake_assemble),
            patch("craft_taskgen.steps._has_dockerfile", return_value=True),
            patch("craft_taskgen.steps._run_build_dockerfile_one", side_effect=fake_build_dockerfile),
            patch("craft_taskgen.steps._run_docker_classify_one", side_effect=fake_docker_classify),
            patch("craft_taskgen.steps._run_oracle_check_one", side_effect=fake_oracle),
            patch("craft_taskgen.steps._run_smoke_one", side_effect=fake_smoke),
            patch("craft_taskgen.steps._run_triage_one", side_effect=fake_triage),
            patch("craft_taskgen.steps._generate_summary", return_value=None),
            patch("craft_taskgen.steps.os.path.isfile", return_value=True),
        ):
            await run_task_pipeline(task, state, "/dev/null", sems)

    asyncio.run(run())
    assert task.stage == Stage.ACCEPTED
    assert triage_count == 2
    # Alignment runs exactly once — on the retry pass (BUILT → alignment → fast-path).
    # Initial pass starts at ALIGNMENT_CHECKED and skips alignment.
    assert alignment_count == 1
    assert assemble_count == 1  # fast-path skips assemble on retry
    assert dockerfile_count == 1  # fast-path skips dockerfile on retry
    assert classify_count == 1  # fast-path skips classify on retry
    assert oracle_count == 1  # fast-path skips oracle on retry


def test_run_task_pipeline_handles_triage_reset_docker():
    """Docker-only triage fix: pipeline skips alignment/test-discovery/classify/oracle on retry.

    Triage sets ORACLE_CHECKED directly (Harbor rebuilds Docker automatically).
    Alignment, test discovery, classify, and oracle all produce identical results
    since only the Dockerfile changed, so they are skipped entirely.
    """
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import run_task_pipeline

    state = PipelineState(created="2026-04-10")
    task = TaskState(
        task_id="t1",
        repo="fastapi",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.ALIGNMENT_CHECKED,
        task_dir="harbor-tasks/craft-tools-v4/t2v3-FA3-streaming",
    )
    state.tasks["t1"] = task
    triage_count = 0
    alignment_count = 0
    find_tests_count = 0

    async def fake_alignment(t, s, sf):
        nonlocal alignment_count
        alignment_count += 1
        t.alignment_verdict = "ok"
        t.stage = Stage.ALIGNMENT_CHECKED

    async def fake_find_tests(t, s, sf):
        nonlocal find_tests_count
        find_tests_count += 1
        t.stage = Stage.TESTS_DISCOVERED

    async def fake_docker_classify(t, s, sf):
        t.stage = Stage.F2P_P2P_CLASSIFIED

    async def fake_oracle(t, s, sf):
        t.stage = Stage.ORACLE_CHECKED

    async def fake_smoke(t, s, sf, **kw):
        setattr(t, kw["score_attr"], "3/5")
        setattr(t, kw["trial_attr"], "jobs/test")
        t.stage = kw["next_stage"]

    async def fake_triage(t, s, sf, **kw):
        nonlocal triage_count
        triage_count += 1
        if triage_count == 1 and kw["label"] == "Opus":
            t.stage = Stage.ORACLE_CHECKED  # docker fix -> triage sets ORACLE_CHECKED directly
        else:
            t.stage = kw["accept_stage"]

    sems = {
        "llm": asyncio.Semaphore(4),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    async def fake_build_dockerfile_docker(t, s, sf):
        t.stage = Stage.DOCKERFILE_BUILT

    async def run():
        with (
            patch("craft_taskgen.steps._run_alignment_one", side_effect=fake_alignment),
            patch("craft_taskgen.steps._run_assemble_task_dir_artifacts_one", side_effect=fake_find_tests),
            patch("craft_taskgen.steps._run_build_dockerfile_one", side_effect=fake_build_dockerfile_docker),
            patch("craft_taskgen.steps._run_docker_classify_one", side_effect=fake_docker_classify),
            patch("craft_taskgen.steps._run_oracle_check_one", side_effect=fake_oracle),
            patch("craft_taskgen.steps._run_smoke_one", side_effect=fake_smoke),
            patch("craft_taskgen.steps._run_triage_one", side_effect=fake_triage),
            patch("craft_taskgen.steps._generate_summary", return_value=None),
        ):
            await run_task_pipeline(task, state, "/dev/null", sems)

    asyncio.run(run())
    assert task.stage == Stage.ACCEPTED
    assert triage_count == 2  # Opus (reset to ORACLE_CHECKED directly), Opus (accept)
    # Alignment never runs — initial pass starts at ALIGNMENT_CHECKED; docker-fix
    # reroutes triage directly to ORACLE_CHECKED (not back through BUILT).
    assert alignment_count == 0
    assert find_tests_count == 1  # test discovery only ran once (not re-run after docker fix)
    # docker fix rewind skips _run_docker_classify_one and _run_oracle_check_one on retry;
    # those mocks are still called exactly once (initial pass only)


# ---------------------------------------------------------------------------
# Integration tests — real step logic with canned Claude/Docker/Harbor responses
# ---------------------------------------------------------------------------


def _make_task_dir(tmp_path, task_id="TE1"):
    """Create a minimal task directory with all required files."""
    import os

    task_dir = str(tmp_path / f"t2v3-{task_id}-test-feature")
    os.makedirs(os.path.join(task_dir, "environment"))
    os.makedirs(os.path.join(task_dir, "tests"))
    os.makedirs(os.path.join(task_dir, "solution"))

    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write("Implement the feature. Test: pytest tests/. Env: Python 3.12.\n" * 5)
    with open(os.path.join(task_dir, "task.toml"), "w") as f:
        f.write('[task]\nname = "test"\n')
    with open(os.path.join(task_dir, "environment", "Dockerfile"), "w") as f:
        f.write(
            "FROM python:3.12-slim\n"
            "RUN git clone https://github.com/example/testrepo.git /code "
            "&& cd /code && git checkout 0000000000000000000000000000000000000001\n"
        )
    with open(os.path.join(task_dir, "tests", "gold_reference_tests.py"), "w") as f:
        f.write("def test_a(): pass\ndef test_b(): pass\ndef test_c(): pass\n")
    with open(os.path.join(task_dir, "tests", "verify_TE1.sh"), "w") as f:
        f.write("#!/bin/bash\necho PASS\n")
    with open(os.path.join(task_dir, "tests", "test_runner.py"), "w") as f:
        f.write("print('ok')\n")
    with open(os.path.join(task_dir, "tests", "test.sh"), "w") as f:
        f.write("#!/bin/bash\n")
    with open(os.path.join(task_dir, "solution", "solve.sh"), "w") as f:
        f.write("#!/bin/bash\ngit apply /solution/changes.patch\n")
    with open(os.path.join(task_dir, "solution", "changes.patch"), "w") as f:
        f.write("diff --git a/src/feature.py b/src/feature.py\n")
    return task_dir


def _canned_claude_dispatcher(**overrides):
    """Return a fake run_claude_async that dispatches canned responses based on prompt content.

    Handles the remaining claude -p call sites: build_dockerfile and the
    fix loop. Deep dive, reviewer, summary, evaluate, build, alignment
    moved to llm_judge — patch those separately via `llm_judge.judge`.
    """

    defaults: dict = {}
    defaults.update(overrides)

    async def dispatcher(prompt, **kwargs):
        p = prompt.lower() if isinstance(prompt, str) else ""
        if "fix this" in p:
            return {"result": "Fixed verify script"}
        else:
            return {"result": "ok"}

    return dispatcher


def _canned_triage_dispatcher(**overrides):
    """Return an async callable patching llm_judge.judge for triage tests.

    Covers the schemas triage routes through: DEEP_DIVE_SCHEMA,
    FAIRNESS_REVIEW_SCHEMA, and anything else (returns empty dict).
    Override with `deep_dive=` or `fairness=` kwargs. Legacy `reviewer=`
    is accepted but ignored (the skeptical reviewer stage was removed).
    """
    from craft_taskgen.prompts import DEEP_DIVE_SCHEMA, FAIRNESS_REVIEW_SCHEMA

    deep_dive = overrides.pop(
        "deep_dive",
        {
            "reward": 0.6,
            "ref_tests_passed": 2,
            "ref_tests_total": 3,
            "failures": [
                {"test_name": "test_c", "classification": "keep", "evidence": "Agent missed it"},
            ],
            "overall_assessment": "One genuine capability gap — agent missed test_c",
        },
    )
    fairness = overrides.pop(
        "fairness",
        {
            "severity": "none",
            "reason": "Instruction covers what tests require",
            "evidence_quote": "",
            "evidence_test": "",
        },
    )
    overrides.pop("reviewer", None)  # legacy, silently discarded

    async def dispatcher(**kwargs):
        schema = kwargs.get("schema")
        if schema is DEEP_DIVE_SCHEMA:
            return _make_judge_result(deep_dive)
        if schema is FAIRNESS_REVIEW_SCHEMA:
            return _make_judge_result(fairness)
        return _make_judge_result({})

    return dispatcher


def test_integration_happy_path(tmp_path):
    """Full pipeline integration: BUILT → alignment → docker → opus → triage → haiku → accept.

    Mocks Claude responses and Docker/Harbor calls, but runs real step logic including
    diagnostic file writes, state persistence, stage transitions, and comparison.
    """
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import run_task_pipeline

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-10", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="test feature",
        stage=Stage.ALIGNMENT_CHECKED,
        alignment_verdict="ok",  # task enters post-alignment; pipeline skips alignment
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    fake_claude = _canned_claude_dispatcher()

    async def fake_smoke(task_obj, model, label, **kwargs):
        return {
            "score": "2/3",
            "trial_dir": str(tmp_path / "trial_opus"),
            "ref_passed": 2,
            "ref_total": 3,
            "reward": 0.67,
        }

    # Create fake trial dir so deep dive doesn't bail
    os.makedirs(str(tmp_path / "trial_opus"))

    sems = {
        "llm": asyncio.Semaphore(4),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    async def fake_docker_classify(t, s, sf):
        t.stage = Stage.F2P_P2P_CLASSIFIED
        t.f2p_tests = ["tests/test_a.py::test_missing"]
        t.p2p_tests = ["tests/test_a.py::test_existing"]

    async def fake_oracle(t, s, sf):
        t.oracle_resolved = True
        t.stage = Stage.ORACLE_CHECKED

    triage_dispatcher = _canned_triage_dispatcher(
        reviewer={
            "challenges": [],
            "pass_audit": [],
            "verdict": "accept",
            "overall_verdict": "Confirmed — deep dive findings hold up",
        }
    )

    async def fake_fetch_deep_dive(task_dir, trial_dir):
        return _empty_deep_dive_context()

    async def run():
        with (
            patch("craft_taskgen.steps.run_claude_async", side_effect=fake_claude),
            patch("craft_taskgen.claude_cli.run_claude_async", side_effect=fake_claude),
            patch("craft_taskgen.steps.llm_judge.judge", side_effect=triage_dispatcher),
            patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
            patch("craft_taskgen.steps._run_smoke_async", side_effect=fake_smoke),
            patch(
                "craft_taskgen.steps._run_assemble_task_dir_artifacts_one",
                side_effect=_fake_assemble_artifacts,
            ),  # noqa: E501
            patch("craft_taskgen.steps._has_dockerfile", return_value=True),
            patch("craft_taskgen.steps._run_docker_classify_one", side_effect=fake_docker_classify),
            patch("craft_taskgen.steps._run_oracle_check_one", side_effect=fake_oracle),
        ):
            await run_task_pipeline(task, state, state_file, sems)

    asyncio.run(run())

    # 1. Final stage is ACCEPTED
    assert task.stage == Stage.ACCEPTED

    # 2. Score stored
    assert task.opus_score == "2/3"

    # 3. Alignment passed
    assert task.alignment_verdict == "ok"

    # 4. Diagnostic files were created. Under the Opus-skip + fairness-
    # reviewer architecture the triage_opus.md (Opus skip/keep verdict)
    # and fairness_review_opus.md (cross-family severity verdict) are
    # both written on every task that reaches the deep-dive path.
    diag_dir = os.path.join(task_dir, "diagnostics")
    assert os.path.isdir(diag_dir)
    diag_files = sorted(os.listdir(diag_dir))
    assert len(diag_files) >= 1
    assert any("triage_opus" in f for f in diag_files)
    # Files are monotonically numbered
    for i, f in enumerate(diag_files):
        assert f.startswith(f"{i + 1:03d}_")

    # 5. State was persisted to disk
    loaded = PipelineState.load(state_file)
    assert loaded.tasks["t1"].stage == Stage.ACCEPTED
    assert loaded.tasks["t1"].opus_score == "2/3"

    # 6. Easiness concern NOT set (set by reviewer, not by this flow)
    assert task.easiness_concern is False

    # 7. Iteration log has comparison entry
    comparison_logs = [e for e in task.iteration_log if e.get("step") == "comparison"]
    assert len(comparison_logs) == 1
    assert comparison_logs[0]["outcome"] == "accepted"


def test_integration_perfect_scores_no_flag_without_reviewer_signal(tmp_path):
    """When Opus scores 10/10 but reviewer doesn't flag easiness, task accepts clean.
    Score alone doesn't trigger easiness_concern — it's reviewer-driven."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import run_task_pipeline

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-10", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="test feature",
        stage=Stage.ALIGNMENT_CHECKED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    fake_claude = _canned_claude_dispatcher()

    async def fake_smoke(task_obj, model, label, **kwargs):
        return {"score": "10/10", "trial_dir": str(tmp_path / "trial_opus")}

    os.makedirs(str(tmp_path / "trial_opus"))
    sems = {
        "llm": asyncio.Semaphore(4),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    async def fake_classify_fe(t, s, sf):
        t.stage = Stage.F2P_P2P_CLASSIFIED

    async def fake_oracle_fe(t, s, sf):
        t.oracle_resolved = True
        t.stage = Stage.ORACLE_CHECKED

    triage_dispatcher_fe = _canned_triage_dispatcher(
        deep_dive={
            "reward": 1.0,
            "ref_tests_passed": 10,
            "ref_tests_total": 10,
            "failures": [],
            "overall_assessment": "Perfect",
            "suggested_fixes": [],
        },
        reviewer={
            "challenges": [],
            "pass_audit": [],
            "verdict": "accept",
            "overall_verdict": "ok",
            "easiness_concern": False,
        },
    )

    async def fake_fetch_deep_dive(task_dir, trial_dir):
        return _empty_deep_dive_context()

    async def run():
        with (
            patch("craft_taskgen.steps.run_claude_async", side_effect=fake_claude),
            patch("craft_taskgen.claude_cli.run_claude_async", side_effect=fake_claude),
            patch("craft_taskgen.steps.llm_judge.judge", side_effect=triage_dispatcher_fe),
            patch("craft_taskgen.steps._fetch_deep_dive_context", side_effect=fake_fetch_deep_dive),
            patch("craft_taskgen.steps._run_smoke_async", side_effect=fake_smoke),
            patch(
                "craft_taskgen.steps._run_assemble_task_dir_artifacts_one",
                side_effect=_fake_assemble_artifacts,
            ),  # noqa: E501
            patch("craft_taskgen.steps._has_dockerfile", return_value=True),
            patch("craft_taskgen.steps._run_docker_classify_one", side_effect=fake_classify_fe),
            patch("craft_taskgen.steps._run_oracle_check_one", side_effect=fake_oracle_fe),
        ):
            await run_task_pipeline(task, state, state_file, sems)

    asyncio.run(run())

    assert task.stage == Stage.ACCEPTED
    # No score-based flat-easy flag; reviewer didn't set easiness_concern
    assert task.easiness_concern is False


# ---------------------------------------------------------------------------
# F2P/P2P classification tests
# ---------------------------------------------------------------------------


def test_step_f2p_p2p_classify_success(tmp_path):
    """3-run classification produces fail_to_pass.txt, pass_to_pass.txt, score.py."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_docker_classify_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.TESTS_DISCOVERED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    postmerge_dir = os.path.join(task_dir, "tests", "postmerge_tests", "tests")
    os.makedirs(postmerge_dir, exist_ok=True)
    with open(os.path.join(postmerge_dir, "test_feature.py"), "w") as f:
        f.write("def test_placeholder(): pass\n")

    f2p_result = ["tests/test_feature.py::test_new_a", "tests/test_feature.py::test_new_b"]
    p2p_result = ["tests/test_existing.py::test_old_a"]

    async def run():
        with (
            patch("craft_taskgen.steps.run_docker_build_async", return_value=(True, "ok")),
            patch(
                "craft_taskgen.steps.run_f2p_p2p_classify_async",
                return_value=(f2p_result, p2p_result, "mock output"),
            ),
        ):
            await _run_docker_classify_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.F2P_P2P_CLASSIFIED
    assert task.f2p_tests == f2p_result
    assert task.p2p_tests == p2p_result

    tests_dir = os.path.join(task_dir, "tests")
    f2p_path = os.path.join(tests_dir, "fail_to_pass.txt")
    p2p_path = os.path.join(tests_dir, "pass_to_pass.txt")
    score_path = os.path.join(tests_dir, "score.py")

    assert os.path.isfile(f2p_path)
    assert os.path.isfile(p2p_path)
    assert os.path.isfile(score_path)

    with open(f2p_path) as f:
        f2p_lines = [ln.strip() for ln in f if ln.strip()]
    assert f2p_lines == f2p_result

    with open(p2p_path) as f:
        p2p_lines = [ln.strip() for ln in f if ln.strip()]
    assert p2p_lines == p2p_result

    with open(score_path) as f:
        score_content = f.read()
    assert "resolved" in score_content
    assert "f2p_score" in score_content
    assert "p2p_score" in score_content


def test_step_f2p_p2p_classify_empty_f2p_writes_empty_file(tmp_path):
    """All tests are P2P (empty F2P list) → fail_to_pass.txt is empty, no trailing newline."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_docker_classify_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.TESTS_DISCOVERED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    postmerge_dir = os.path.join(task_dir, "tests", "postmerge_tests", "tests")
    os.makedirs(postmerge_dir, exist_ok=True)
    with open(os.path.join(postmerge_dir, "test_feature.py"), "w") as f:
        f.write("def test_placeholder(): pass\n")

    f2p_result: list[str] = []
    p2p_result = ["tests/test_existing.py::test_a", "tests/test_existing.py::test_b"]

    async def run():
        with (
            patch("craft_taskgen.steps.run_docker_build_async", return_value=(True, "ok")),
            patch(
                "craft_taskgen.steps.run_f2p_p2p_classify_async",
                return_value=(f2p_result, p2p_result, "mock output"),
            ),
        ):
            await _run_docker_classify_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.F2P_P2P_CLASSIFIED
    tests_dir = os.path.join(task_dir, "tests")
    f2p_content = open(os.path.join(tests_dir, "fail_to_pass.txt")).read()
    p2p_content = open(os.path.join(tests_dir, "pass_to_pass.txt")).read()

    # Empty F2P → file is empty (no content, no stray newline)
    assert f2p_content == ""
    # Non-empty P2P → one entry per line with a trailing newline
    assert p2p_content == "tests/test_existing.py::test_a\ntests/test_existing.py::test_b\n"


def test_step_f2p_p2p_classify_docker_failure(tmp_path):
    """Docker build failure triggers _fix_or_shelve_async; task shelved on budget exhaustion."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_docker_classify_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.TESTS_DISCOVERED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    postmerge_dir = os.path.join(task_dir, "tests", "postmerge_tests", "tests")
    os.makedirs(postmerge_dir, exist_ok=True)
    with open(os.path.join(postmerge_dir, "test_feature.py"), "w") as f:
        f.write("def test_placeholder(): pass\n")

    shelved = {}

    async def fake_fix_or_shelve(t, issue, label):
        shelved["called"] = True
        shelved["label"] = label
        t.stage = Stage.NEEDS_FIX
        return False  # shelved — no retry

    async def run():
        with (
            patch("craft_taskgen.steps.run_docker_build_async", return_value=(False, "FROM error")),
            # Patch the steps-module binding, not claude_cli — steps.py uses the imported name.
            patch("craft_taskgen.steps._fix_docker_or_shelve_async", side_effect=fake_fix_or_shelve),
        ):
            await _run_docker_classify_one(task, state, state_file)

    asyncio.run(run())

    assert shelved.get("called") is True
    assert "Docker build" in shelved.get("label", "")
    assert task.stage == Stage.NEEDS_FIX


def test_step_f2p_p2p_classify_case2_edge(tmp_path):
    """Test classified as F2P when file collects+fails in overlay (Case 2 edge case).

    overlay_collected = PASSED ∪ FAILED. If a test appears in FAILED (not PASSED)
    during the overlay run, its file is still in overlay_files, so the test is
    correctly classified as F2P (not falling through to the file-level fallback).
    """
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_docker_classify_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.TESTS_DISCOVERED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    postmerge_dir = os.path.join(task_dir, "tests", "postmerge_tests", "tests")
    os.makedirs(postmerge_dir, exist_ok=True)
    with open(os.path.join(postmerge_dir, "test_feature.py"), "w") as f:
        f.write("def test_placeholder(): pass\n")

    # This test is in oracle_passed and overlay_collected (FAILED) but NOT overlay_passed.
    # The fixed algorithm classifies it as F2P (not the buggy file-level fallback).
    case2_f2p = ["tests/test_feature.py::test_uses_new_api"]
    case2_p2p: list[str] = []

    async def run():
        with (
            patch("craft_taskgen.steps.run_docker_build_async", return_value=(True, "ok")),
            patch(
                "craft_taskgen.steps.run_f2p_p2p_classify_async",
                return_value=(case2_f2p, case2_p2p, "mock output"),
            ),
        ):
            await _run_docker_classify_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.F2P_P2P_CLASSIFIED
    assert "tests/test_feature.py::test_uses_new_api" in task.f2p_tests
    assert task.p2p_tests == []

    tests_dir = os.path.join(task_dir, "tests")
    with open(os.path.join(tests_dir, "fail_to_pass.txt")) as f:
        f2p_lines = [ln.strip() for ln in f if ln.strip()]
    assert "tests/test_feature.py::test_uses_new_api" in f2p_lines
    with open(os.path.join(tests_dir, "pass_to_pass.txt")) as f:
        p2p_lines = [ln.strip() for ln in f if ln.strip()]
    assert p2p_lines == []


# ---------------------------------------------------------------------------
# run_f2p_p2p_classify_async shell-injection safety (docker.py:99)
# ---------------------------------------------------------------------------


def test_run_f2p_p2p_classify_paths_are_shell_quoted(tmp_path):
    """test_paths are shlex-quoted before interpolation — spaces and metacharacters are safe."""
    import asyncio

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    dangerous_paths = [
        "tests/test feature.py",  # space — would word-split unquoted
        "tests/test$(id).py",  # command substitution
        "tests/test;rm -rf /repo;.py",  # command separator
    ]

    async def fake_create(*args, **kwargs):
        # The script is written to a temp file and passed via volume mount.
        # Capture it from the filesystem by reading the mounted path arg.
        for arg in args:
            if isinstance(arg, str) and arg.endswith(".sh") and "/classify" not in arg:
                pass
        # Instead: intercept the NamedTemporaryFile write by checking the script content
        # arrives safely — verify the generated script contains quoted forms.
        mock_proc = MagicMock()

        async def fake_communicate():
            # Return minimal output so the function completes without error branch
            output = _make_classify_output(
                oracle_passed=("tests/test_feature.py::test_x",),
            )
            return (output, b"")

        mock_proc.communicate = fake_communicate
        return mock_proc

    # Patch NamedTemporaryFile to capture what's written
    import tempfile as _tempfile

    original_ntf = _tempfile.NamedTemporaryFile
    written: list[str] = []

    class CapturingNTF:
        def __init__(self, **kwargs):
            self._ntf = original_ntf(**kwargs)
            self.name = self._ntf.name

        def write(self, content):
            written.append(content)
            return self._ntf.write(content)

        def __enter__(self):
            self._ntf.__enter__()
            return self

        def __exit__(self, *args):
            return self._ntf.__exit__(*args)

    async def run():
        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_create),
            patch("craft_taskgen.docker.tempfile.NamedTemporaryFile", CapturingNTF),
        ):
            return await run_f2p_p2p_classify_async(str(tmp_path), dangerous_paths)

    asyncio.run(run())

    assert written, "no script content captured"
    script = written[0]
    # Each dangerous path must appear single-quoted, never bare
    assert "'tests/test feature.py'" in script
    assert "'tests/test$(id).py'" in script
    assert "'tests/test;rm -rf /repo;.py'" in script
    # The bare (unquoted) dangerous forms must not appear
    assert "tests/test feature.py" not in script.replace("'tests/test feature.py'", "")
    assert "$(id)" not in script.replace("'tests/test$(id).py'", "")


# ---------------------------------------------------------------------------
# run_f2p_p2p_classify_async output-parsing (docker.py)
# These tests exercise the P2P/F2P classification formula and _extract_tests
# regex directly — without calling Docker — by mocking asyncio.create_subprocess_exec.
# ---------------------------------------------------------------------------


def _make_classify_output(
    *,
    overlay_passed: tuple[str, ...] = (),
    overlay_failed: tuple[str, ...] = (),
    oracle_passed: tuple[str, ...] = (),
) -> bytes:
    """Build synthetic 2-run pytest output for classification parsing tests."""

    def section(name: str, passed: tuple[str, ...] = (), failed: tuple[str, ...] = ()) -> str:
        lines = [f"==={name}_START==="]
        for t in passed:
            lines.append(f"{t} PASSED")
        for t in failed:
            lines.append(f"{t} FAILED")
        lines.append(f"==={name}_END===")
        return "\n".join(lines)

    return "\n".join(
        [
            section("OVERLAY", passed=overlay_passed, failed=overlay_failed),
            "===SOLVE_EXIT=0===",
            section("ORACLE", passed=oracle_passed),
            "",
        ]
    ).encode()


def _fake_classify_proc(output: bytes):
    """Return an asyncio.create_subprocess_exec side_effect that yields output."""
    mock_proc = MagicMock()

    async def fake_communicate():
        return (output, b"")

    mock_proc.communicate = fake_communicate

    async def fake_create(*args, **kwargs):
        return mock_proc

    return fake_create


def test_run_f2p_p2p_classify_new_test_is_f2p(tmp_path):
    """Test FAILED in overlay, passes oracle → F2P."""
    import asyncio

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    output = _make_classify_output(
        overlay_passed=(),
        overlay_failed=("tests/test_feature.py::test_new",),
        oracle_passed=("tests/test_feature.py::test_new",),
    )

    async def run():
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_classify_proc(output)):
            return await run_f2p_p2p_classify_async(str(tmp_path), ["tests/test_feature.py"])

    f2p, p2p, _ = asyncio.run(run())
    assert f2p == ["tests/test_feature.py::test_new"]
    assert p2p == []


def test_run_f2p_p2p_classify_passing_test_is_p2p(tmp_path):
    """Test passes in overlay and oracle → P2P (was already passing before commit)."""
    import asyncio

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    output = _make_classify_output(
        overlay_passed=("tests/test_existing.py::test_old",),
        overlay_failed=(),
        oracle_passed=("tests/test_existing.py::test_old",),
    )

    async def run():
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_classify_proc(output)):
            return await run_f2p_p2p_classify_async(str(tmp_path), ["tests/test_existing.py"])

    f2p, p2p, _ = asyncio.run(run())
    assert f2p == []
    assert p2p == ["tests/test_existing.py::test_old"]


def test_run_f2p_p2p_classify_mixed_produces_correct_split(tmp_path):
    """Two tests in same file: one new (F2P), one pre-existing (P2P) → correctly split."""
    import asyncio

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    output = _make_classify_output(
        overlay_passed=("tests/test_feature.py::test_old",),
        overlay_failed=("tests/test_feature.py::test_new",),
        oracle_passed=("tests/test_feature.py::test_new", "tests/test_feature.py::test_old"),
    )

    async def run():
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_classify_proc(output)):
            return await run_f2p_p2p_classify_async(str(tmp_path), ["tests/test_feature.py"])

    f2p, p2p, _ = asyncio.run(run())
    assert f2p == ["tests/test_feature.py::test_new"]
    assert p2p == ["tests/test_feature.py::test_old"]


def test_run_f2p_p2p_classify_oracle_zero_returns_none(tmp_path):
    """Oracle section has no passing tests → (None, None, output) returned."""
    import asyncio

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    output = _make_classify_output(
        overlay_passed=("tests/test_feature.py::test_old",),
        overlay_failed=(),
        oracle_passed=(),  # oracle passes nothing — likely infra failure
    )

    async def run():
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_classify_proc(output)):
            return await run_f2p_p2p_classify_async(str(tmp_path), ["tests/test_feature.py"])

    f2p, p2p, out = asyncio.run(run())
    assert f2p is None
    assert p2p is None
    assert isinstance(out, str)


def test_run_f2p_p2p_classify_regression_drops_task(tmp_path):
    """Test passes overlay but oracle gets 0 results → task dropped (ORACLE_ZERO fires first)."""
    import asyncio

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    output = _make_classify_output(
        overlay_passed=("tests/test_feature.py::test_old",),
        overlay_failed=(),
        oracle_passed=(),  # test_old broken by commit → oracle gets nothing
    )

    async def run():
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_classify_proc(output)):
            return await run_f2p_p2p_classify_async(str(tmp_path), ["tests/test_feature.py"])

    f2p, p2p, out = asyncio.run(run())
    assert f2p is None
    assert p2p is None
    assert out.startswith("ORACLE_ZERO:")


def test_run_f2p_p2p_classify_regression_with_other_oracle_passes_drops_task(tmp_path):
    """Some tests pass oracle but one overlay-passer fails oracle → OVERLAY_REGRESSION."""
    import asyncio

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    output = _make_classify_output(
        overlay_passed=("tests/test_feature.py::test_old", "tests/test_feature.py::test_broken"),
        overlay_failed=("tests/test_feature.py::test_new",),
        oracle_passed=("tests/test_feature.py::test_new", "tests/test_feature.py::test_old"),
        # test_broken passed overlay but is absent from oracle → regression
    )

    async def run():
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_classify_proc(output)):
            return await run_f2p_p2p_classify_async(str(tmp_path), ["tests/test_feature.py"])

    f2p, p2p, out = asyncio.run(run())
    assert f2p is None
    assert p2p is None
    assert out.startswith("OVERLAY_REGRESSION:")


def test_run_f2p_p2p_classify_uncollected_in_overlay_drops_task(tmp_path):
    """Oracle-passing test absent from overlay_collected → OVERLAY_UNCOLLECTED → task dropped."""
    import asyncio

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    # test_brand_new appears in neither overlay PASSED nor FAILED — collection-level error
    output = _make_classify_output(
        overlay_passed=(),
        overlay_failed=(),
        oracle_passed=("tests/test_new_file.py::test_brand_new",),
    )

    async def run():
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_classify_proc(output)):
            return await run_f2p_p2p_classify_async(str(tmp_path), ["tests/test_new_file.py"])

    f2p, p2p, out = asyncio.run(run())
    assert f2p is None
    assert p2p is None
    assert out.startswith("OVERLAY_UNCOLLECTED:")


# ---------------------------------------------------------------------------
# _run_f2p_p2p_classify_one: zero-test-files guard and oracle-zero step path
# ---------------------------------------------------------------------------


def test_step_f2p_p2p_classify_no_test_files(tmp_path):
    """Empty postmerge_tests/ → stage=NEEDS_FIX, needs_human_review=True."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_docker_classify_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.TESTS_DISCOVERED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    # Do NOT create tests/postmerge_tests/ — empty dir means no tests discovered

    async def run():
        await _run_docker_classify_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.NEEDS_FIX
    assert task.needs_human_review is True
    assert "no test files found" in task.human_review_reason


def test_step_f2p_p2p_classify_oracle_zero_shelves_task(tmp_path):
    """run_f2p_p2p_classify_async returns (None, None, ...) → fix attempted, task shelved."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_docker_classify_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.TESTS_DISCOVERED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    postmerge_dir = os.path.join(task_dir, "tests", "postmerge_tests", "tests")
    os.makedirs(postmerge_dir, exist_ok=True)
    with open(os.path.join(postmerge_dir, "test_feature.py"), "w") as f:
        f.write("def test_placeholder(): pass\n")

    fix_calls: dict = {}

    async def fake_fix_or_shelve(t, issue, label):
        fix_calls["label"] = label
        fix_calls["issue"] = issue
        t.stage = Stage.NEEDS_FIX
        t.needs_human_review = True
        t.human_review_reason = f"{label} after 0 attempts"
        return False  # budget exhausted — shelved

    async def run():
        with (
            patch("craft_taskgen.steps.run_docker_build_async", return_value=(True, "ok")),
            patch(
                "craft_taskgen.steps.run_f2p_p2p_classify_async",
                return_value=(None, None, "ORACLE_ZERO: no tests passed"),
            ),
            # Patch the steps-module binding, not claude_cli — steps.py uses the imported name.
            patch(
                "craft_taskgen.steps._fix_f2p_p2p_classify_or_shelve_async",
                side_effect=fake_fix_or_shelve,
            ),
        ):
            await _run_docker_classify_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.NEEDS_FIX
    assert task.needs_human_review is True
    assert fix_calls.get("called") is not True  # just verify label was captured
    assert "oracle" in fix_calls.get("label", "").lower()
    assert "oracle" in fix_calls.get("issue", "").lower()
    assert "0 tests" in fix_calls.get("issue", "").lower()


def test_step_f2p_p2p_classify_overlay_regression_rejects_immediately(tmp_path):
    """OVERLAY_REGRESSION prefix → immediate Stage.REJECTED, no fix loop, issues dict appended."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_docker_classify_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.TESTS_DISCOVERED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    postmerge_dir = os.path.join(task_dir, "tests", "postmerge_tests", "tests")
    os.makedirs(postmerge_dir, exist_ok=True)
    with open(os.path.join(postmerge_dir, "test_feature.py"), "w") as f:
        f.write("def test_placeholder(): pass\n")

    fix_called = []

    async def run():
        with (
            patch("craft_taskgen.steps.run_docker_build_async", return_value=(True, "ok")),
            patch(
                "craft_taskgen.steps.run_f2p_p2p_classify_async",
                return_value=(None, None, "OVERLAY_REGRESSION: test_foo passed pre-merge, failed oracle"),
            ),
            patch(
                "craft_taskgen.steps._fix_f2p_p2p_classify_or_shelve_async",
                new_callable=AsyncMock,
                side_effect=lambda *a, **kw: fix_called.append(1),
            ),
        ):
            await _run_docker_classify_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.REJECTED
    assert not fix_called, "fix loop must NOT be called for OVERLAY_REGRESSION"
    assert len(task.issues) == 1
    assert task.issues[0]["type"] == "overlay_regression"
    assert "regression" in task.issues[0]["description"].lower()


def test_step_f2p_p2p_classify_overlay_uncollected_rejects_immediately(tmp_path):
    """OVERLAY_UNCOLLECTED prefix → immediate Stage.REJECTED, no fix loop, issues dict appended."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_docker_classify_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.TESTS_DISCOVERED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    postmerge_dir = os.path.join(task_dir, "tests", "postmerge_tests", "tests")
    os.makedirs(postmerge_dir, exist_ok=True)
    with open(os.path.join(postmerge_dir, "test_feature.py"), "w") as f:
        f.write("def test_placeholder(): pass\n")

    fix_called = []

    async def run():
        with (
            patch("craft_taskgen.steps.run_docker_build_async", return_value=(True, "ok")),
            patch(
                "craft_taskgen.steps.run_f2p_p2p_classify_async",
                return_value=(None, None, "OVERLAY_UNCOLLECTED: test_bar not collected in overlay"),
            ),
            patch(
                "craft_taskgen.steps._fix_f2p_p2p_classify_or_shelve_async",
                new_callable=AsyncMock,
                side_effect=lambda *a, **kw: fix_called.append(1),
            ),
        ):
            await _run_docker_classify_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.REJECTED
    assert not fix_called, "fix loop must NOT be called for OVERLAY_UNCOLLECTED"
    assert len(task.issues) == 1
    assert task.issues[0]["type"] == "overlay_uncollected"
    assert "collected" in task.issues[0]["description"].lower()


def test_step_f2p_p2p_classify_git_infrastructure_failure(tmp_path):
    """_find_commit_test_files returns None → stage=NEEDS_FIX, needs_human_review, git failure reason."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_assemble_task_dir_artifacts_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.ALIGNMENT_CHECKED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    async def run():
        with (
            patch("craft_taskgen.steps._generate_solve_sh", return_value=(True, "")),
            patch("craft_taskgen.steps._find_commit_test_files", new_callable=AsyncMock, return_value=None),
        ):
            await _run_assemble_task_dir_artifacts_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.NEEDS_FIX
    assert task.needs_human_review is True
    assert "Git infrastructure failure" in task.human_review_reason


# ---------------------------------------------------------------------------
# Oracle check tests
# ---------------------------------------------------------------------------


def test_step_oracle_check_resolves(tmp_path):
    """Oracle resolved=True advances task to ORACLE_CHECKED."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_oracle_check_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.F2P_P2P_CLASSIFIED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    oracle_result = {
        "resolved": True,
        "f2p_score": 1.0,
        "p2p_score": 1.0,
        "f2p_passed": 2,
        "f2p_total": 2,
        "p2p_passed": 1,
        "p2p_total": 1,
        "reward": 1.0,
    }

    async def run():
        with patch("craft_taskgen.steps.run_score_check_async", return_value=oracle_result):
            await _run_oracle_check_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.ORACLE_CHECKED
    assert task.oracle_resolved is True
    assert task.oracle_f2p_score == 1.0
    assert task.oracle_p2p_score == 1.0
    assert task.oracle_flagged is False


def test_step_oracle_check_blocks_on_failure(tmp_path):
    """Oracle not resolved → NEEDS_FIX, oracle_flagged=True, pipeline stops."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_oracle_check_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.F2P_P2P_CLASSIFIED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    oracle_result = {
        "resolved": False,
        "f2p_score": 0.5,
        "p2p_score": 1.0,
        "f2p_passed": 1,
        "f2p_total": 2,
        "p2p_passed": 1,
        "p2p_total": 1,
        "reward": 0.0,
    }

    async def run():
        with patch("craft_taskgen.steps.run_score_check_async", return_value=oracle_result):
            await _run_oracle_check_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.NEEDS_FIX  # blocked
    assert task.oracle_flagged is True
    assert task.oracle_resolved is False
    assert task.needs_human_review is True
    assert "f2p=" in task.oracle_flag_reason


def test_step_oracle_check_score_error(tmp_path):
    """run_score_check_async returns error dict → NEEDS_FIX, oracle_flagged=True, pipeline blocked."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_oracle_check_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.F2P_P2P_CLASSIFIED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    async def run():
        with patch(
            "craft_taskgen.steps.run_score_check_async",
            return_value={"error": "no_reward_json", "output": "Container exited without reward.json"},
        ):
            await _run_oracle_check_one(task, state, state_file)

    asyncio.run(run())

    assert task.stage == Stage.NEEDS_FIX
    assert task.oracle_flagged is True
    assert task.needs_human_review is True
    assert "no_reward_json" in task.oracle_flag_reason
    assert task.oracle_resolved is False  # default — error branch never sets this


# ---------------------------------------------------------------------------
# TaskState serialization round-trip for new F2P/P2P fields
# ---------------------------------------------------------------------------


def test_state_f2p_p2p_fields_persist(tmp_path):
    """New F2P/P2P fields on TaskState serialize and deserialize correctly."""
    import json

    from craft_taskgen.config import PipelineState, Stage, TaskState

    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="feature",
        stage=Stage.ORACLE_CHECKED,
        f2p_tests=["tests/test_f.py::test_new_a", "tests/test_f.py::test_new_b"],
        p2p_tests=["tests/test_e.py::test_old_a"],
        oracle_resolved=True,
        oracle_f2p_score=1.0,
        oracle_p2p_score=1.0,
        oracle_flagged=False,
        oracle_flag_reason="",
    )
    state.tasks["t1"] = task

    state_file = str(tmp_path / "state.json")
    state.save(state_file)

    loaded = PipelineState.load(state_file)
    t = loaded.tasks["t1"]

    assert t.stage == Stage.ORACLE_CHECKED
    assert t.f2p_tests == ["tests/test_f.py::test_new_a", "tests/test_f.py::test_new_b"]
    assert t.p2p_tests == ["tests/test_e.py::test_old_a"]
    assert t.oracle_resolved is True
    assert t.oracle_f2p_score == 1.0
    assert t.oracle_p2p_score == 1.0
    assert t.oracle_flagged is False

    # Verify round-trip to JSON preserves the lists
    with open(state_file) as f:
        raw = json.load(f)
    raw_task = raw["tasks"]["t1"]
    assert raw_task["f2p_tests"] == ["tests/test_f.py::test_new_a", "tests/test_f.py::test_new_b"]
    assert raw_task["p2p_tests"] == ["tests/test_e.py::test_old_a"]


# ---------------------------------------------------------------------------
# _generate_solve_sh
# ---------------------------------------------------------------------------


def test_generate_solve_sh_happy_path(tmp_path):
    """Writes changes.patch and solve.sh when git diff produces output."""
    import os
    import stat

    from craft_taskgen.steps import _generate_solve_sh

    repo_dir = tmp_path / "repos" / "myorg" / "myrepo"
    repo_dir.mkdir(parents=True)
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)

    fake_diff = (
        "diff --git a/src/lib.py b/src/lib.py\n--- a/src/lib.py\n+++ b/src/lib.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    fake_result = MagicMock()
    fake_result.stdout = fake_diff
    fake_result.returncode = 0

    with patch("subprocess.run", return_value=fake_result):
        ok, err = _generate_solve_sh("myorg/myrepo", "base123", "merge456", task_dir)

    assert ok is True
    assert err == ""

    patch_path = os.path.join(task_dir, "solution", "changes.patch")
    solve_path = os.path.join(task_dir, "solution", "solve.sh")
    assert os.path.isfile(patch_path)
    assert os.path.isfile(solve_path)
    assert open(patch_path).read() == fake_diff
    assert "git apply /solution/changes.patch" in open(solve_path).read()
    assert os.stat(solve_path).st_mode & stat.S_IXUSR


def test_generate_solve_sh_empty_diff(tmp_path):
    """Returns error when git diff produces an empty patch."""
    import os

    from craft_taskgen.steps import _generate_solve_sh

    repo_dir = tmp_path / "repos" / "myorg" / "myrepo"
    repo_dir.mkdir(parents=True)
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)

    fake_result = MagicMock()
    fake_result.stdout = ""
    fake_result.returncode = 0

    with patch("subprocess.run", return_value=fake_result):
        ok, err = _generate_solve_sh("myorg/myrepo", "base123", "merge456", task_dir)

    assert ok is False
    assert "empty patch" in err


def test_generate_solve_sh_nonzero_returncode(tmp_path):
    """Returns error when git diff exits non-zero (e.g. invalid SHA, corrupt repo)."""
    import os

    from craft_taskgen.steps import _generate_solve_sh

    repo_dir = tmp_path / "repos" / "myorg" / "myrepo"
    repo_dir.mkdir(parents=True)
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)

    fake_result = MagicMock()
    fake_result.stdout = ""
    fake_result.stderr = "fatal: bad object base123"
    fake_result.returncode = 128  # git exits 128 on unknown object

    with patch("subprocess.run", return_value=fake_result):
        ok, err = _generate_solve_sh("myorg/myrepo", "base123", "merge456", task_dir)

    assert ok is False
    assert "128" in err
    assert "bad object" in err


def test_generate_solve_sh_timeout(tmp_path):
    """Returns error when git diff times out."""
    import os

    from craft_taskgen.steps import _generate_solve_sh

    repo_dir = tmp_path / "repos" / "myorg" / "myrepo"
    repo_dir.mkdir(parents=True)
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 60)):
        ok, err = _generate_solve_sh("myorg/myrepo", "base123", "merge456", task_dir)

    assert ok is False
    assert "git diff failed" in err


def test_generate_solve_sh_os_error(tmp_path):
    """Returns error when git is not found."""
    import os

    from craft_taskgen.steps import _generate_solve_sh

    repo_dir = tmp_path / "repos" / "myorg" / "myrepo"
    repo_dir.mkdir(parents=True)
    task_dir = str(tmp_path / "task")
    os.makedirs(task_dir)

    with patch("subprocess.run", side_effect=OSError("git not found")):
        ok, err = _generate_solve_sh("myorg/myrepo", "base123", "merge456", task_dir)

    assert ok is False
    assert "git diff failed" in err


# ---------------------------------------------------------------------------
# run_score_check_async unit tests (docker.py)
# ---------------------------------------------------------------------------


def test_run_score_check_async_timeout(tmp_path):
    """asyncio.TimeoutError during docker run → {"error": "timeout"}."""
    import asyncio

    from craft_taskgen.docker import run_score_check_async

    async def run():
        mock_proc = MagicMock()
        mock_proc.kill = MagicMock()

        async def fake_wait():
            pass

        mock_proc.wait = fake_wait

        async def fake_communicate():
            raise asyncio.TimeoutError()

        mock_proc.communicate = fake_communicate

        async def fake_create(*args, **kwargs):
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
            return await run_score_check_async(str(tmp_path), timeout=1)

    result = asyncio.run(run())
    assert result["error"] == "timeout"


def test_run_score_check_async_no_reward_json(tmp_path):
    """Docker run succeeds but reward.json is never written → {"error": "no_reward_json"}."""
    import asyncio

    from craft_taskgen.docker import run_score_check_async

    async def run():
        mock_proc = MagicMock()

        async def fake_communicate():
            return (b"container stdout", b"container stderr")

        mock_proc.communicate = fake_communicate

        async def fake_create(*args, **kwargs):
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
            return await run_score_check_async(str(tmp_path))

    result = asyncio.run(run())
    assert result["error"] == "no_reward_json"
    assert "container stdout" in result["output"]


def test_run_score_check_async_json_parse_error(tmp_path):
    """reward.json exists but contains invalid JSON → {"error": "json_parse"}."""
    import asyncio

    from craft_taskgen.docker import run_score_check_async

    bad_json_dir = tmp_path / "logs"
    verifier_dir = bad_json_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "reward.json").write_text("{not valid json")

    async def run():
        mock_proc = MagicMock()

        async def fake_communicate():
            return (b"output", b"")

        mock_proc.communicate = fake_communicate

        async def fake_create(*args, **kwargs):
            return mock_proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_create),
            patch("craft_taskgen.docker.tempfile.mkdtemp", return_value=str(bad_json_dir)),
        ):
            return await run_score_check_async(str(tmp_path))

    result = asyncio.run(run())
    assert result["error"] == "json_parse"


# ---------------------------------------------------------------------------
# DOCKER_VALIDATED backward-compat entry point
# ---------------------------------------------------------------------------


def test_write_task_toml_uses_profile_values(tmp_path):
    """_write_task_toml uses values from the supplied PipelineProfile, not hardcoded defaults."""
    import tomllib

    from craft_taskgen.config import PipelineProfile
    from craft_taskgen.steps import _write_task_toml

    profile = PipelineProfile(
        task_memory_mb=8192,
        task_agent_timeout=7200,
        task_build_timeout=1800,
        task_cpus=4,
        task_verifier_timeout=300,
        task_storage_mb=20480,
        task_gpus=1,
        task_allow_internet=False,
    )
    _write_task_toml(str(tmp_path), "TEST01", profile)
    with open(tmp_path / "task.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["metadata"]["name"] == "TEST01"
    assert data["verifier"]["timeout_sec"] == 300
    assert data["agent"]["timeout_sec"] == 7200
    assert data["environment"]["build_timeout_sec"] == 1800.0
    assert data["environment"]["cpus"] == 4
    assert data["environment"]["memory_mb"] == 8192
    assert data["environment"]["storage_mb"] == 20480
    assert data["environment"]["gpus"] == 1
    assert data["environment"]["allow_internet"] is False


def test_write_task_toml_defaults(tmp_path):
    """_write_task_toml uses PipelineProfile defaults when no profile is provided."""
    import tomllib

    from craft_taskgen.config import PipelineProfile
    from craft_taskgen.steps import _write_task_toml

    _write_task_toml(str(tmp_path), "DEF01", PipelineProfile())
    with open(tmp_path / "task.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["verifier"]["timeout_sec"] == 600
    assert data["agent"]["timeout_sec"] == 3600
    assert data["environment"]["build_timeout_sec"] == 900.0
    assert data["environment"]["cpus"] == 2
    assert data["environment"]["memory_mb"] == 4096
    assert data["environment"]["storage_mb"] == 10240
    assert data["environment"]["gpus"] == 0
    assert data["environment"]["allow_internet"] is True


def test_state_load_migrates_docker_validated(tmp_path):
    """Loading a state file with docker_validated stage migrates to oracle_checked."""
    import json

    from craft_taskgen.config import PipelineState, Stage

    state_file = str(tmp_path / "state.json")
    data = {
        "created": "2026-04-14",
        "last_updated": "2026-04-14",
        "run_dir": str(tmp_path),
        "tasks": {
            "t1": {
                "task_id": "t1",
                "repo": "testrepo",
                "commit_sha": "abc123",
                "base_sha": "base123",
                "merge_base_sha": "base123",
                "description": "legacy task",
                "stage": "docker_validated",
            }
        },
    }
    with open(state_file, "w") as f:
        json.dump(data, f)

    state = PipelineState.load(state_file)
    assert state.tasks["t1"].stage == Stage.ORACLE_CHECKED


# ---------------------------------------------------------------------------
# SCORE_PY_TEMPLATE business logic
# ---------------------------------------------------------------------------


def test_score_py_template_normal_case(tmp_path):
    """score.py correctly parses PASSED lines and computes f2p/p2p scores."""
    import subprocess
    import sys

    from craft_taskgen.prompts import SCORE_PY_TEMPLATE

    # Write the generated score.py
    score_py = tmp_path / "score.py"
    score_py.write_text(SCORE_PY_TEMPLATE)

    # Set up /tests and /logs inside tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    logs_dir = tmp_path / "logs" / "verifier"
    logs_dir.mkdir(parents=True)

    f2p_content = "tests/test_feat.py::test_new_a\ntests/test_feat.py::test_new_b\n"
    (tests_dir / "fail_to_pass.txt").write_text(f2p_content)
    (tests_dir / "pass_to_pass.txt").write_text("tests/test_existing.py::test_old_a\n")

    pytest_output = (
        "tests/test_feat.py::test_new_a PASSED\n"
        "tests/test_feat.py::test_new_b PASSED\n"
        "tests/test_existing.py::test_old_a PASSED\n"
        "tests/test_existing.py::test_old_b FAILED\n"
    )
    (logs_dir / "verify_full_output.txt").write_text(pytest_output)

    env_overrides = {
        "PYTHONPATH": "",
    }
    import os

    env = {**os.environ, **env_overrides}

    result = subprocess.run(
        [sys.executable, str(score_py)],
        capture_output=True,
        text=True,
        # Rewrite /tests and /logs paths by symlinking into tmp_path
        env=env,
        cwd=str(tmp_path),
    )

    # score.py uses hard-coded /tests and /logs — run it via a wrapper that patches paths
    # Instead: write a patched version substituting tmp_path for /
    patched = SCORE_PY_TEMPLATE.replace('"/tests/', f'"{tests_dir}/').replace(
        'Path("/logs/verifier")', f'Path("{logs_dir}")'
    )
    score_py.write_text(patched)

    result = subprocess.run([sys.executable, str(score_py)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    import json

    reward_data = json.loads((logs_dir / "reward.json").read_text())
    assert reward_data["f2p_passed"] == 2
    assert reward_data["f2p_total"] == 2
    assert reward_data["p2p_passed"] == 1
    assert reward_data["p2p_total"] == 1
    assert reward_data["f2p_score"] == 1.0
    assert reward_data["p2p_score"] == 1.0
    assert reward_data["resolved"] is True
    assert reward_data["reward"] == 1.0


def test_score_py_template_missing_output_file_raises(tmp_path):
    """score.py raises FileNotFoundError when verify_full_output.txt is absent."""
    import subprocess
    import sys

    from craft_taskgen.prompts import SCORE_PY_TEMPLATE

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    logs_dir = tmp_path / "logs" / "verifier"
    logs_dir.mkdir(parents=True)

    (tests_dir / "fail_to_pass.txt").write_text("tests/test_feat.py::test_new_a\n")
    (tests_dir / "pass_to_pass.txt").write_text("")
    # verify_full_output.txt deliberately not created

    patched = SCORE_PY_TEMPLATE.replace('"/tests/', f'"{tests_dir}/').replace(
        'Path("/logs/verifier")', f'Path("{logs_dir}")'
    )
    score_py = tmp_path / "score.py"
    score_py.write_text(patched)

    result = subprocess.run([sys.executable, str(score_py)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "verify_full_output.txt" in result.stderr
    assert not (logs_dir / "reward.json").exists()


def test_score_py_template_zero_f2p_raises(tmp_path):
    """score.py raises ValueError when fail_to_pass.txt is empty."""
    import subprocess
    import sys

    from craft_taskgen.prompts import SCORE_PY_TEMPLATE

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    logs_dir = tmp_path / "logs" / "verifier"
    logs_dir.mkdir(parents=True)

    (tests_dir / "fail_to_pass.txt").write_text("")
    (tests_dir / "pass_to_pass.txt").write_text("tests/test_existing.py::test_old_a\n")
    (logs_dir / "verify_full_output.txt").write_text("tests/test_existing.py::test_old_a PASSED\n")

    patched = SCORE_PY_TEMPLATE.replace('"/tests/', f'"{tests_dir}/').replace(
        'Path("/logs/verifier")', f'Path("{logs_dir}")'
    )
    score_py = tmp_path / "score.py"
    score_py.write_text(patched)

    result = subprocess.run([sys.executable, str(score_py)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "fail_to_pass.txt is empty" in result.stderr
    assert not (logs_dir / "reward.json").exists()


def test_score_py_template_zero_passed_lines(tmp_path):
    """score.py produces resolved=False/reward=0 when no tests pass."""
    import subprocess
    import sys

    from craft_taskgen.prompts import SCORE_PY_TEMPLATE

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    logs_dir = tmp_path / "logs" / "verifier"
    logs_dir.mkdir(parents=True)

    (tests_dir / "fail_to_pass.txt").write_text("tests/test_feat.py::test_new_a\n")
    (tests_dir / "pass_to_pass.txt").write_text("tests/test_existing.py::test_old_a\n")
    (logs_dir / "verify_full_output.txt").write_text(
        "tests/test_feat.py::test_new_a FAILED\ntests/test_existing.py::test_old_a FAILED\n"
    )

    patched = SCORE_PY_TEMPLATE.replace('"/tests/', f'"{tests_dir}/').replace(
        'Path("/logs/verifier")', f'Path("{logs_dir}")'
    )
    score_py = tmp_path / "score.py"
    score_py.write_text(patched)

    result = subprocess.run([sys.executable, str(score_py)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    import json

    reward_data = json.loads((logs_dir / "reward.json").read_text())
    assert reward_data["f2p_passed"] == 0
    assert reward_data["p2p_passed"] == 0
    assert reward_data["resolved"] is False
    assert reward_data["reward"] == 0.0


# ---------------------------------------------------------------------------
# Generated test.sh content
# ---------------------------------------------------------------------------


def test_step_f2p_p2p_classify_test_sh_content(tmp_path):
    """Generated test.sh contains the tee + || true + score.py chain."""
    import asyncio
    import os

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import _run_docker_classify_one

    task_dir = _make_task_dir(tmp_path)
    state_file = str(tmp_path / "state.json")
    state = PipelineState(created="2026-04-14", run_dir=str(tmp_path))
    task = TaskState(
        task_id="t1",
        repo="testrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="add feature",
        stage=Stage.TESTS_DISCOVERED,
        task_dir=task_dir,
    )
    state.tasks["t1"] = task
    state.save(state_file)

    postmerge_dir = os.path.join(task_dir, "tests", "postmerge_tests", "tests")
    os.makedirs(postmerge_dir, exist_ok=True)
    with open(os.path.join(postmerge_dir, "test_feature.py"), "w") as f:
        f.write("def test_placeholder(): pass\n")

    async def run():
        with (
            patch("craft_taskgen.steps.run_docker_build_async", return_value=(True, "ok")),
            patch(
                "craft_taskgen.steps.run_f2p_p2p_classify_async",
                return_value=(["tests/test_feature.py::test_a"], [], "mock output"),
            ),
        ):
            await _run_docker_classify_one(task, state, state_file)

    asyncio.run(run())

    test_sh_path = os.path.join(task_dir, "tests", "test.sh")
    assert os.path.isfile(test_sh_path)
    content = open(test_sh_path).read()

    # tee target must be the file score.py reads
    assert "tee /logs/verifier/verify_full_output.txt" in content
    # || true must appear on the same line as the pytest/tee pipeline
    tee_line = next(ln for ln in content.splitlines() if "tee /logs/verifier/verify_full_output.txt" in ln)
    assert "|| true" in tee_line
    # score.py must be called as a separate statement after the tee pipeline
    lines = content.splitlines()
    tee_idx = next(i for i, ln in enumerate(lines) if "tee /logs/verifier/verify_full_output.txt" in ln)
    score_lines_after = [ln for ln in lines[tee_idx + 1 :] if "score.py" in ln]
    assert score_lines_after, "python3 /tests/score.py must appear after the tee line"

    # postmerge overlay block must be present before pytest runs
    assert "if [ -d /tests/postmerge_tests ]" in content
    assert "find /tests/postmerge_tests -type f" in content
    assert 'cp "$f" "/code/$rel"' in content
    # overlay must precede the pytest invocation
    overlay_idx = next(i for i, ln in enumerate(lines) if "/tests/postmerge_tests" in ln)
    pytest_idx = next(i for i, ln in enumerate(lines) if "python3 -m pytest" in ln)
    assert overlay_idx < pytest_idx, "postmerge overlay must run before pytest"


# ---------------------------------------------------------------------------
# run_score_check_async command string
# ---------------------------------------------------------------------------


def test_run_score_check_async_command_string(tmp_path):
    """run_score_check_async passes bash /tests/test.sh (not python3 /tests/score.py)."""
    import asyncio
    import json as _json

    from craft_taskgen.docker import run_score_check_async

    reward_dir = tmp_path / "logs" / "verifier"
    reward_dir.mkdir(parents=True)
    (reward_dir / "reward.json").write_text(
        _json.dumps(
            {
                "reward": 1.0,
                "resolved": True,
                "f2p_score": 1.0,
                "p2p_score": 1.0,
                "f2p_passed": 2,
                "f2p_total": 2,
                "p2p_passed": 1,
                "p2p_total": 1,
            }
        )
    )

    captured_args: list = []

    async def run():
        mock_proc = MagicMock()

        async def fake_communicate():
            return (b"output", b"")

        mock_proc.communicate = fake_communicate

        async def fake_create(*args, **kwargs):
            captured_args.extend(args)
            return mock_proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_create),
            patch("craft_taskgen.docker.tempfile.mkdtemp", return_value=str(tmp_path / "logs")),
        ):
            return await run_score_check_async(str(tmp_path), apply_solution=False)

    result = asyncio.run(run())

    # Verify returned dict
    assert result["reward"] == 1.0
    assert result["resolved"] is True

    # Verify the -c argument passed to docker run
    docker_cmd_arg = captured_args[-1]  # last positional arg to create_subprocess_exec
    assert docker_cmd_arg == "bash /tests/test.sh", f"Expected 'bash /tests/test.sh', got {docker_cmd_arg!r}"


def test_run_score_check_async_apply_solution_command(tmp_path):
    """run_score_check_async with apply_solution=True prepends solve.sh via &&."""
    import asyncio
    import json as _json

    from craft_taskgen.docker import run_score_check_async

    reward_dir = tmp_path / "logs" / "verifier"
    reward_dir.mkdir(parents=True)
    (reward_dir / "reward.json").write_text(
        _json.dumps(
            {
                "reward": 1.0,
                "resolved": True,
                "f2p_score": 1.0,
                "p2p_score": 1.0,
                "f2p_passed": 1,
                "f2p_total": 1,
                "p2p_passed": 0,
                "p2p_total": 0,
            }
        )
    )

    captured_args: list = []

    async def run():
        mock_proc = MagicMock()

        async def fake_communicate():
            return (b"", b"")

        mock_proc.communicate = fake_communicate

        async def fake_create(*args, **kwargs):
            captured_args.extend(args)
            return mock_proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_create),
            patch("craft_taskgen.docker.tempfile.mkdtemp", return_value=str(tmp_path / "logs")),
        ):
            return await run_score_check_async(str(tmp_path), apply_solution=True)

    asyncio.run(run())

    docker_cmd_arg = captured_args[-1]
    assert docker_cmd_arg == "bash /solution/solve.sh && bash /tests/test.sh", (
        f"Expected solve.sh && test.sh chain, got {docker_cmd_arg!r}"
    )


# ---------------------------------------------------------------------------
# Docker build context path
# ---------------------------------------------------------------------------


def test_run_docker_build_async_uses_environment_as_context(tmp_path):
    """run_docker_build_async passes task_dir/environment as the build context."""
    import asyncio
    import os

    from craft_taskgen.docker import run_docker_build_async

    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    dockerfile = env_dir / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")

    captured_args: list = []

    async def run():
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        async def fake_communicate():
            return (b"", b"")

        mock_proc.communicate = fake_communicate

        async def fake_create(*args, **kwargs):
            captured_args.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
            return await run_docker_build_async(str(tmp_path))

    run_ok, _ = asyncio.run(run())
    assert run_ok

    # Last arg to create_subprocess_exec is the build context
    assert captured_args[-1] == os.path.abspath(str(env_dir)), (
        f"Build context should be environment/, got {captured_args[-1]!r}"
    )


# ---------------------------------------------------------------------------
# Classify postmerge mount path
# ---------------------------------------------------------------------------


def test_run_f2p_p2p_classify_mounts_from_tests_postmerge(tmp_path):
    """run_f2p_p2p_classify_async mounts task_dir/tests/postmerge_tests/ as /postmerge:ro,
    not the old environment/postmerge_tests/ path."""
    import asyncio
    import os

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    # Create tests/postmerge_tests/ so the conditional mount is triggered
    postmerge_dir = tmp_path / "tests" / "postmerge_tests"
    postmerge_dir.mkdir(parents=True)

    captured_args: list = []

    async def run():
        mock_proc = MagicMock()

        async def fake_communicate():
            output = _make_classify_output(
                oracle_passed=("tests/test_foo.py::test_x",),
            )
            return (output, b"")

        mock_proc.communicate = fake_communicate

        async def fake_create(*args, **kwargs):
            captured_args.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
            return await run_f2p_p2p_classify_async(str(tmp_path), ["tests/test_foo.py"])

    asyncio.run(run())

    # Extract all -v volume mount args from the docker run command
    volume_mounts = [
        captured_args[i + 1]
        for i, arg in enumerate(captured_args)
        if arg == "-v" and i + 1 < len(captured_args)
    ]

    # The postmerge mount must point to tests/postmerge_tests/, never environment/postmerge_tests/
    postmerge_mounts = [m for m in volume_mounts if ":/postmerge" in m]
    assert postmerge_mounts, f"No /postmerge volume mount found. Mounts: {volume_mounts}"

    expected_source = os.path.abspath(str(postmerge_dir))
    assert any(m.startswith(expected_source + ":") for m in postmerge_mounts), (
        f"Expected postmerge mount from {expected_source!r}, got {postmerge_mounts!r}"
    )
    assert not any("environment/postmerge_tests" in m for m in postmerge_mounts), (
        f"Mount still uses old environment/postmerge_tests path: {postmerge_mounts!r}"
    )


def test_run_f2p_p2p_classify_no_mount_when_postmerge_absent(tmp_path):
    """run_f2p_p2p_classify_async skips the /postmerge mount when tests/postmerge_tests/ doesn't exist."""
    import asyncio

    from craft_taskgen.docker import run_f2p_p2p_classify_async

    # Do NOT create tests/postmerge_tests/ — directory absent
    captured_args: list = []

    async def run():
        mock_proc = MagicMock()

        async def fake_communicate():
            output = _make_classify_output(
                oracle_passed=("tests/test_foo.py::test_x",),
            )
            return (output, b"")

        mock_proc.communicate = fake_communicate

        async def fake_create(*args, **kwargs):
            captured_args.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
            return await run_f2p_p2p_classify_async(str(tmp_path), ["tests/test_foo.py"])

    asyncio.run(run())

    volume_mounts = [
        captured_args[i + 1]
        for i, arg in enumerate(captured_args)
        if arg == "-v" and i + 1 < len(captured_args)
    ]
    postmerge_mounts = [m for m in volume_mounts if ":/postmerge" in m]
    assert not postmerge_mounts, (
        f"Expected no /postmerge mount when directory absent, got {postmerge_mounts!r}"
    )


# ---------------------------------------------------------------------------
# _is_test_file
# ---------------------------------------------------------------------------


def test_is_test_file_matches():
    from craft_taskgen.steps import _is_test_file

    assert _is_test_file("test_foo.py")  # test_ prefix
    assert _is_test_file("foo_test.py")  # _test.py suffix
    assert _is_test_file("foo.test.py")  # .test.py suffix
    assert _is_test_file("tests/helpers.py")  # in tests/ dir
    assert _is_test_file("test/helpers.py")  # in test/ dir
    assert _is_test_file("mypackage/tests/helpers.py")  # nested tests/ dir
    assert _is_test_file("mypackage/tests/test_foo.py")  # both prefix and dir


def test_is_test_file_excludes():
    from craft_taskgen.steps import _is_test_file

    assert not _is_test_file("conftest.py")  # excluded by rule
    assert not _is_test_file("tests/conftest.py")  # excluded even in tests/ dir
    assert not _is_test_file("utils.py")  # plain source file
    assert not _is_test_file("foo.js")  # not .py
    assert not _is_test_file("test_foo.txt")  # not .py


def test_run_task_pipeline_evaluated_retries_build():
    """A task at EVALUATED (build previously failed) is retried and succeeds."""
    import asyncio

    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import run_task_pipeline

    state = PipelineState(created="2026-04-10")
    task = TaskState(
        task_id="t1",
        repo="gel",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.EVALUATED,
    )
    state.tasks["t1"] = task
    step_calls = []

    async def fake_build_align(t, s, sf, sems_arg):
        step_calls.append("build_align")
        t.stage = Stage.ALIGNMENT_CHECKED
        t.task_dir = "harbor-tasks/t2v3-GE-test"
        t.alignment_verdict = "ok"

    async def fake_find_tests(t, s, sf):
        step_calls.append("find_tests")
        t.stage = Stage.TESTS_DISCOVERED

    async def fake_build_dockerfile(t, s, sf):
        step_calls.append("build_dockerfile")
        t.stage = Stage.DOCKERFILE_BUILT

    async def fake_docker_classify(t, s, sf):
        step_calls.append("docker_classify")
        t.stage = Stage.F2P_P2P_CLASSIFIED

    async def fake_oracle(t, s, sf):
        step_calls.append("oracle")
        t.stage = Stage.ORACLE_CHECKED

    async def fake_smoke(t, s, sf, **kw):
        step_calls.append(f"smoke_{kw['label']}")
        setattr(t, kw["score_attr"], "3/5")
        setattr(t, kw["trial_attr"], "jobs/test")
        t.stage = kw["next_stage"]

    async def fake_triage(t, s, sf, **kw):
        step_calls.append(f"triage_{kw['label']}")
        t.stage = kw["accept_stage"]

    sems = {
        "llm": asyncio.Semaphore(4),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    async def run():
        with (
            patch("craft_taskgen.steps._run_build_align_candidates", side_effect=fake_build_align),
            patch("craft_taskgen.steps._run_assemble_task_dir_artifacts_one", side_effect=fake_find_tests),
            patch("craft_taskgen.steps._has_dockerfile", return_value=True),
            patch("craft_taskgen.steps._run_build_dockerfile_one", side_effect=fake_build_dockerfile),
            patch("craft_taskgen.steps._run_docker_classify_one", side_effect=fake_docker_classify),
            patch("craft_taskgen.steps._run_oracle_check_one", side_effect=fake_oracle),
            patch("craft_taskgen.steps._run_smoke_one", side_effect=fake_smoke),
            patch("craft_taskgen.steps._run_triage_one", side_effect=fake_triage),
            patch("craft_taskgen.steps._generate_summary", return_value=None),
        ):
            await run_task_pipeline(task, state, "/dev/null", sems)

    asyncio.run(run())
    assert task.stage == Stage.ACCEPTED
    assert "build_align" in step_calls


def test_run_task_pipeline_evaluated_exhausts_to_needs_fix():
    """A task stuck at EVALUATED that keeps failing build reaches NEEDS_FIX."""
    import asyncio

    import craft_taskgen.config as _cfg
    from craft_taskgen.config import PipelineState, Stage, TaskState
    from craft_taskgen.steps import run_task_pipeline

    state = PipelineState(created="2026-04-10")
    task = TaskState(
        task_id="t1",
        repo="gel",
        commit_sha="abc",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.EVALUATED,
        fix_attempts=_cfg.MAX_FIX_ATTEMPTS - 1,
    )
    state.tasks["t1"] = task

    async def failing_build_align(t, s, sf, sems_arg):
        # build_align fails: stage stays EVALUATED so the orchestrator falls
        # through; pipeline body increments fix_attempts on a non-ACCEPTED outcome.
        # NEEDS_FIX is set by the wrapper when MAX_FIX_ATTEMPTS exhausted.
        t.stage = Stage.NEEDS_FIX
        t.needs_human_review = True
        t.human_review_reason = "synthetic build_align failure"

    sems = {
        "llm": asyncio.Semaphore(4),
        "docker": asyncio.Semaphore(2),
        "smoke": asyncio.Semaphore(2),
        "candidate": asyncio.Semaphore(4),
    }

    async def run():
        with patch("craft_taskgen.steps._run_build_align_candidates", side_effect=failing_build_align):
            await run_task_pipeline(task, state, "/dev/null", sems)

    asyncio.run(run())
    assert task.stage == Stage.NEEDS_FIX
    assert task.needs_human_review is True


# ---------------------------------------------------------------------------
# candidate_data field tests
# ---------------------------------------------------------------------------


def test_candidate_data_field_roundtrip(tmp_path):
    """candidate_data survives TaskState → JSON → PipelineState.load()."""
    from craft_taskgen.config import PipelineState, Stage, TaskState

    state = PipelineState(created="2026-04-20")
    state.tasks["t1"] = TaskState(
        task_id="t1",
        repo="myrepo",
        commit_sha="abc123",
        base_sha="base123",
        merge_base_sha="base123",
        description="feat",
        stage=Stage.CANDIDATE,
        candidate_data={"repo": "myrepo", "sha": "abc123", "score": 0.9, "source_files": ["src/x.py"]},
    )
    state_file = str(tmp_path / "state.json")
    state.save(state_file)

    loaded = PipelineState.load(state_file)
    assert loaded.tasks["t1"].candidate_data == {
        "repo": "myrepo",
        "sha": "abc123",
        "score": 0.9,
        "source_files": ["src/x.py"],
    }


def test_candidate_data_missing_in_old_state(tmp_path):
    """Old state.json without candidate_data deserializes to empty dict."""
    import json

    from craft_taskgen.config import PipelineState

    old_state = {
        "created": "2026-01-01",
        "last_updated": "2026-01-01",
        "tasks": {
            "t1": {
                "task_id": "t1",
                "repo": "myrepo",
                "commit_sha": "abc",
                "base_sha": "base",
                "merge_base_sha": "base",
                "description": "feat",
                "stage": "candidate",
            }
        },
    }
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(old_state))

    loaded = PipelineState.load(str(state_file))
    assert loaded.tasks["t1"].candidate_data == {}


def test_select_candidates_preserves_raw(tmp_path):
    """select_candidates returns _raw with full original miner fields."""
    import json

    from craft_taskgen.steps import select_candidates

    data = {
        "candidates": [
            {
                "sha": "aaa",
                "base_sha": "aaa0",
                "merge_base_sha": "aaa0",
                "subject": "feat: add X",
                "score": 10,
                "has_test_patch": True,
                "source_files": ["src/x.py"],
                "test_files": ["tests/test_x.py"],
                "source_lines_changed": 42,
            }
        ]
    }
    fpath = tmp_path / "testrepo.json"
    fpath.write_text(json.dumps(data))

    results = select_candidates([str(fpath)], top_per_repo=5, max_total=10)
    assert len(results) == 1
    raw = results[0]["_raw"]
    assert raw["repo"] == "testrepo"
    assert raw["sha"] == "aaa"
    assert raw["source_files"] == ["src/x.py"]
    assert raw["test_files"] == ["tests/test_x.py"]
    assert raw["source_lines_changed"] == 42


_SAMPLE_DOCKERFILE = (
    "FROM python:3.12-slim\n"
    "RUN apt-get update && apt-get install -y git curl ca-certificates build-essential\n"
    "RUN git clone https://github.com/python-attrs/cattrs.git /code && cd /code "
    "&& git checkout 309e9d1413cfb0947b8ba4e704dd5dcd2652ae27\n"
    "WORKDIR /code\n"
    "RUN pip install -e '.[ujson]' 2>&1 | tail -5\n"
)


def test_write_candidate_json_structured_shape(tmp_path):
    """_write_candidate_json emits the schema planning adapters consume."""
    import hashlib
    import json

    from craft_taskgen.config import Stage, TaskState
    from craft_taskgen.steps import _write_candidate_json

    task_dir = tmp_path / "t2v3-CA6bc4-cattrs-annotated-overrides"
    (task_dir / "environment").mkdir(parents=True)
    dockerfile_path = task_dir / "environment" / "Dockerfile"
    dockerfile_path.write_text(_SAMPLE_DOCKERFILE)
    (task_dir / "instruction.md").write_text("Implement annotated overrides.\n")

    task = TaskState(
        task_id="t1",
        repo="cattrs",
        commit_sha="6bc4708fb9b2ac52d9a18997e923da6a58916102",
        base_sha="309e9d1413cfb0947b8ba4e704dd5dcd2652ae27",
        merge_base_sha="309e9d1413cfb0947b8ba4e704dd5dcd2652ae27",
        description="feat",
        stage=Stage.F2P_P2P_CLASSIFIED,
        task_dir=str(task_dir),
        f2p_tests=["tests/test_a.py::test_1"],
        p2p_tests=["tests/test_b.py::test_2", "tests/test_b.py::test_3"],
    )

    _write_candidate_json(task)

    written = json.loads((task_dir / "candidate.json").read_text())
    assert written == {
        "task_name": "t2v3-CA6bc4-cattrs-annotated-overrides",
        "abbrev": "CA6bc4",
        "repo": "python-attrs/cattrs",
        "parent_sha": "309e9d1413cfb0947b8ba4e704dd5dcd2652ae27",
        "spec": "Implement annotated overrides.\n",
        "docker": {
            "mode": "verbatim",
            "python": "3.12",
            "repo_dir": "/code",
            "source_dockerfile_sha256": hashlib.sha256(_SAMPLE_DOCKERFILE.encode()).hexdigest(),
        },
        "fail_to_pass": ["tests/test_a.py::test_1"],
        "pass_to_pass": ["tests/test_b.py::test_2", "tests/test_b.py::test_3"],
    }


@pytest.mark.parametrize(
    "dockerfile,expected_msg",
    [
        # No `FROM python:X.Y` line at all.
        ("FROM ubuntu:22.04\n", "FROM python"),
        # `FROM python:` present but no github clone URL.
        (
            "FROM python:3.12-slim\nRUN pip install foo\n",
            "git clone https://github.com",
        ),
        # Clone URL present but no `git checkout <sha>`.
        (
            "FROM python:3.12-slim\nRUN git clone https://github.com/org/repo.git /code\n",
            "git checkout",
        ),
    ],
)
def test_write_candidate_json_raises_on_malformed_dockerfile(tmp_path, dockerfile, expected_msg):
    """Each missing required line surfaces as a distinct ValueError."""
    from craft_taskgen.config import Stage, TaskState
    from craft_taskgen.steps import _write_candidate_json

    task_dir = tmp_path / "t2v3-XX0000-broken"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text(dockerfile)

    task = TaskState(
        task_id="t1",
        repo="x",
        commit_sha="a",
        base_sha="b",
        merge_base_sha="b",
        description="",
        stage=Stage.F2P_P2P_CLASSIFIED,
        task_dir=str(task_dir),
    )

    with pytest.raises(ValueError, match=expected_msg):
        _write_candidate_json(task)


# ---------------------------------------------------------------------------
# _load_actually_failed_tests — reward.json preferred, verify_full_output
# fallback so trials written before SCORE_PY_TEMPLATE emitted f2p_failed /
# p2p_failed arrays can still be filtered (was dropping every DD
# classification on affected trials; see steps.py:325).
# ---------------------------------------------------------------------------


def test_load_actually_failed_tests_reads_reward_json_arrays(tmp_path):
    from craft_taskgen.steps import _load_actually_failed_tests

    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "reward.json").write_text(
        json.dumps(
            {
                "reward": 0.0,
                "f2p_failed": ["tests/test_a.py::test_one"],
                "p2p_failed": ["tests/test_b.py::test_two"],
            }
        )
    )
    failed = _load_actually_failed_tests(str(trial))
    assert failed is not None
    assert "tests/test_a.py::test_one" in failed
    assert "test_one" in failed  # short-name expansion for DD matching
    assert "tests/test_b.py::test_two" in failed
    assert "test_two" in failed


def test_load_actually_failed_tests_reads_reward_details_json(tmp_path):
    """New SCORE_PY_TEMPLATE splits failure lists into reward-details.json
    so reward.json can stay numeric-only for harbor>=0.13.1 pydantic compat.
    Reader must consult the side-car first."""
    from craft_taskgen.steps import _load_actually_failed_tests

    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "reward.json").write_text(
        json.dumps({"reward": 0.0, "resolved": 0, "f2p_passed": 0, "f2p_total": 1})
    )
    (trial / "verifier" / "reward-details.json").write_text(
        json.dumps(
            {
                "f2p_failed": ["tests/test_a.py::test_one"],
                "p2p_failed": ["tests/test_b.py::test_two"],
            }
        )
    )
    failed = _load_actually_failed_tests(str(trial))
    assert failed is not None
    assert "tests/test_a.py::test_one" in failed
    assert "test_one" in failed
    assert "tests/test_b.py::test_two" in failed


def test_load_actually_failed_tests_falls_back_to_verify_output(tmp_path):
    from craft_taskgen.steps import _load_actually_failed_tests

    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    # reward.json exists but has no f2p_failed / p2p_failed arrays (the
    # pre-fix shape that silently dropped every DD classification).
    (trial / "verifier" / "reward.json").write_text(
        json.dumps({"reward": 0.0, "f2p_passed": 4, "f2p_total": 5})
    )
    (trial / "verifier" / "verify_full_output.txt").write_text(
        "tests/test_core.py::test_glob_and_grep FAILED           [ 25%]\n"
        "tests/test_core.py::test_passes PASSED           [ 50%]\n"
    )
    failed = _load_actually_failed_tests(str(trial))
    assert failed == {
        "tests/test_core.py::test_glob_and_grep",
        "test_glob_and_grep",
    }


def test_load_actually_failed_tests_returns_none_when_nothing_available(tmp_path):
    from craft_taskgen.steps import _load_actually_failed_tests

    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    assert _load_actually_failed_tests(str(trial)) is None


# ---------------------------------------------------------------------------
# Deterministic easiness check — count-based rule on harbor-lab trajectory.
# ---------------------------------------------------------------------------


_TOOL_SEQ_HEAVY = "\n".join(f"| {i} | {'Grep' if i % 2 else 'Read'} | /code/foo.py |" for i in range(1, 21))


def test_deterministic_easiness_flags_low_exploration():
    # Only 3 Grep/Read calls → below the `<= 5` threshold → flag.
    from craft_taskgen.steps import _deterministic_easiness

    tool_seq = "\n".join(f"| {i} | Read | /code/foo.py |" for i in range(1, 4))
    flag, reason = _deterministic_easiness(tool_seq, '{"reward": 1.0}')
    assert flag is True
    assert "grep_read=3" in reason
    assert "<=5" in reason


def test_deterministic_easiness_boundary_at_five():
    # Exactly 5 Grep+Read → still flags under `<= 5` threshold.
    from craft_taskgen.steps import _deterministic_easiness

    tool_seq = "\n".join(f"| {i} | Read | /code/foo.py |" for i in range(1, 6))
    flag, reason = _deterministic_easiness(tool_seq, '{"reward": 1.0}')
    assert flag is True
    assert "grep_read=5" in reason


def test_deterministic_easiness_boundary_at_six_no_flag():
    # 6 Grep+Read → does NOT flag. Preserves competent small-task work.
    from craft_taskgen.steps import _deterministic_easiness

    tool_seq = "\n".join(f"| {i} | Read | /code/foo.py |" for i in range(1, 7))
    flag, reason = _deterministic_easiness(tool_seq, '{"reward": 1.0}')
    assert flag is False
    assert reason == ""


def test_deterministic_easiness_does_not_flag_on_zero_pytest_alone():
    # 20 Grep+Read, zero pytest runs — used to flag under the old rubric.
    # Dropped in Apr 23 refactor because pytest=0 conflates recipe-following
    # with bad procedure / non-pytest verification. Should NOT flag.
    from craft_taskgen.steps import _deterministic_easiness

    flag, reason = _deterministic_easiness(_TOOL_SEQ_HEAVY, '{"reward": 1.0}')
    assert flag is False
    assert reason == ""


def test_deterministic_easiness_clears_normal_trajectory():
    # Plenty of exploration → no flag.
    from craft_taskgen.steps import _deterministic_easiness

    tool_seq = _TOOL_SEQ_HEAVY + "\n| 21 | Bash | pytest -x |"
    flag, reason = _deterministic_easiness(tool_seq, '{"reward": 1.0}')
    assert flag is False
    assert reason == ""


def test_deterministic_easiness_skips_failed_trial():
    # reward < 1.0 → don't evaluate, even if trajectory looks minimal.
    from craft_taskgen.steps import _deterministic_easiness

    tool_seq = "| 1 | Edit | /code/foo.py |"
    flag, reason = _deterministic_easiness(tool_seq, '{"reward": 0.0}')
    assert flag is False
    assert reason == ""


def test_deterministic_easiness_returns_false_on_missing_reward():
    from craft_taskgen.steps import _deterministic_easiness

    flag, reason = _deterministic_easiness(_TOOL_SEQ_HEAVY, "")
    assert flag is False
    assert reason == ""


# (Legacy dual-DD merge tests deleted in the Apr 23 2026 refactor —
# the merge function itself no longer exists. Triage tests above cover
# the new Opus-only skip/keep + fairness-review flow.)


# --------------------------------------------------------------------------- #
# cleanup_task_images + run_task_pipeline wrapper
# --------------------------------------------------------------------------- #


def _fake_subprocess_pair(list_output: bytes):
    """Build an async side_effect for asyncio.create_subprocess_exec. `docker
    images ...` calls yield `list_output`; `docker image rm ...` calls return
    a proc whose .wait() resolves to 0."""
    state = {"calls": []}

    async def fake_create(*args, **kwargs):
        state["calls"].append(args)
        mock_proc = MagicMock()
        if "images" in args:

            async def _communicate():
                return (list_output, b"")

            mock_proc.communicate = _communicate
        else:

            async def _wait():
                return 0

            mock_proc.wait = _wait
        return mock_proc

    return fake_create, state


def test_cleanup_task_images_matches_both_patterns():
    import asyncio

    from craft_taskgen.docker import cleanup_task_images

    # First docker-images call (craft-<basename>) matches one tag.
    # Second docker-images call (<basename[:32]>*) matches one tag.
    # Each is followed by a docker image rm call with those tags.
    outputs = [
        b"craft-t2v3-sample-slug:latest\n",
        b"t2v3-sample-slug-abc1234-main:latest\n",
    ]
    call_seq = {"idx": 0, "cmds": []}

    async def fake_create(*args, **kwargs):
        call_seq["cmds"].append(args)
        mock_proc = MagicMock()
        if "images" in args:
            out = outputs[call_seq["idx"] // 2]

            async def _communicate(_out=out):
                return (_out, b"")

            mock_proc.communicate = _communicate
        else:

            async def _wait():
                return 0

            mock_proc.wait = _wait
        call_seq["idx"] += 1
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        asyncio.run(cleanup_task_images("/x/t2v3-sample-slug"))

    cmds = call_seq["cmds"]
    assert len(cmds) == 4, cmds
    assert "images" in cmds[0] and "reference=craft-t2v3-sample-slug" in cmds[0]
    assert "rm" in cmds[1] and "craft-t2v3-sample-slug:latest" in cmds[1]
    assert "images" in cmds[2] and "reference=t2v3-sample-slug*" in cmds[2]
    assert "rm" in cmds[3] and "t2v3-sample-slug-abc1234-main:latest" in cmds[3]


def test_cleanup_task_images_noop_when_empty_task_dir():
    import asyncio

    from craft_taskgen.docker import cleanup_task_images

    calls = []

    def fake_create(*args, **kwargs):
        calls.append(args)

        async def _ret(*a, **k):
            raise AssertionError("should not be called")

        return _ret()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        asyncio.run(cleanup_task_images(""))

    assert calls == []


def test_cleanup_task_images_noop_when_no_matches():
    import asyncio

    from craft_taskgen.docker import cleanup_task_images

    fake_create, state = _fake_subprocess_pair(list_output=b"")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        asyncio.run(cleanup_task_images("/x/t2v3-empty"))

    # 2 images calls (one per pattern), 0 rm calls (no matches).
    cmds = state["calls"]
    assert all("images" in c for c in cmds), cmds
    assert len(cmds) == 2, cmds


def test_cleanup_task_images_swallows_oserror(capsys):
    import asyncio

    from craft_taskgen.docker import cleanup_task_images

    def fake_create(*args, **kwargs):
        raise OSError("docker daemon unavailable")

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        asyncio.run(cleanup_task_images("/x/t2v3-sample"))  # must not raise

    captured = capsys.readouterr()
    assert "[cleanup] WARN" in captured.out


def test_cleanup_task_images_lowercases_basename():
    import asyncio

    from craft_taskgen.docker import cleanup_task_images

    fake_create, state = _fake_subprocess_pair(list_output=b"")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        asyncio.run(cleanup_task_images("/x/T2V3-MixedCase-Slug"))

    images_calls = [c for c in state["calls"] if "images" in c]
    # both filter args are lowercase
    assert any("reference=craft-t2v3-mixedcase-slug" in c for c in images_calls)
    assert any("reference=t2v3-mixedcase-slug*" in c for c in images_calls)


def _make_wrapper_test_task(task_dir: str = "/x/t2v3-wrappertest"):
    """Minimal TaskState-shaped stub for run_task_pipeline wrapper tests."""
    from craft_taskgen.config import Stage

    t = MagicMock()
    t.task_dir = task_dir
    t.task_id = "t2v3-wrappertest"
    t.stage = Stage.PROMISING
    t.fix_attempts = 0
    t.needs_human_review = False
    t.pending_fix_type = ""
    return t


async def _noop_async(*args, **kwargs):
    return None


def test_run_task_pipeline_cleans_on_needs_fix():
    """Normal terminal exit (stage becomes NEEDS_FIX) → cleanup fires."""
    import asyncio

    from craft_taskgen.config import Stage
    from craft_taskgen.steps import run_task_pipeline

    task = _make_wrapper_test_task()
    sems = {
        "llm": asyncio.Semaphore(1),
        "docker": asyncio.Semaphore(1),
        "smoke": asyncio.Semaphore(1),
        "candidate": asyncio.Semaphore(1),
    }
    cleanup_mock = AsyncMock()

    async def fake_build_align_terminal(t, s, sf, sems_arg=None):
        t.stage = Stage.NEEDS_FIX  # set terminal directly so test doesn't depend on MAX_FIX_ATTEMPTS

    async def run():
        with (
            patch("craft_taskgen.steps._run_build_align_candidates", side_effect=fake_build_align_terminal),
            patch("craft_taskgen.steps.save_state_locked", new=AsyncMock()),
            patch("craft_taskgen.docker.cleanup_task_images", cleanup_mock),
        ):
            await run_task_pipeline(task, MagicMock(), "/dev/null", sems)

    asyncio.run(run())
    cleanup_mock.assert_awaited_once_with(task.task_dir)


def test_run_task_pipeline_cleans_on_exception():
    """Exception mid-loop → cleanup fires in except branch, exception re-raised."""
    import asyncio

    from craft_taskgen.steps import run_task_pipeline

    task = _make_wrapper_test_task()
    sems = {
        "llm": asyncio.Semaphore(1),
        "docker": asyncio.Semaphore(1),
        "smoke": asyncio.Semaphore(1),
        "candidate": asyncio.Semaphore(1),
    }
    cleanup_mock = AsyncMock()

    async def raising_build_align(t, s, sf, sems_arg=None):
        raise RuntimeError("boom")

    async def run():
        with (
            patch("craft_taskgen.steps._run_build_align_candidates", side_effect=raising_build_align),
            patch("craft_taskgen.docker.cleanup_task_images", cleanup_mock),
        ):
            await run_task_pipeline(task, MagicMock(), "/dev/null", sems)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run())
    cleanup_mock.assert_awaited_once_with(task.task_dir)


def test_run_task_pipeline_skips_cleanup_on_cancel():
    """CancelledError mid-loop → cleanup is skipped (preserve cache for resume)."""
    import asyncio

    from craft_taskgen.steps import run_task_pipeline

    task = _make_wrapper_test_task()
    sems = {
        "llm": asyncio.Semaphore(1),
        "docker": asyncio.Semaphore(1),
        "smoke": asyncio.Semaphore(1),
        "candidate": asyncio.Semaphore(1),
    }
    cleanup_mock = AsyncMock()

    async def cancelled_build_align(t, s, sf, sems_arg=None):
        raise asyncio.CancelledError()

    async def run():
        with (
            patch("craft_taskgen.steps._run_build_align_candidates", side_effect=cancelled_build_align),
            patch("craft_taskgen.docker.cleanup_task_images", cleanup_mock),
        ):
            await run_task_pipeline(task, MagicMock(), "/dev/null", sems)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    cleanup_mock.assert_not_awaited()
