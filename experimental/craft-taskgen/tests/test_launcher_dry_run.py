# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/run-baselines.sh --dry-run output.

Most of the launcher is integration-tested (real harbor invocations are
expensive and need docker + network). This narrow test covers the
non-negotiable behaviors that are trivial to assert from dry-run output
alone.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run-baselines.sh"


def _dry_run(
    tmp_path: Path,
    extra_args: list[str],
    env: dict[str, str] | None = None,
    expected_status: int = 0,
    env_file: Path | None = None,
) -> str:
    """The helper runs the launcher with --dry-run and returns stdout and stderr.

    `.env` loading matters — the launcher reads `.env` from the CWD, so
    we chdir into tmp_path and write a fake .env so the test doesn't
    depend on the dev's real keys. An optional env_file exercises the production
    .env parser with the shipped example; other callers use fake credentials.
    """
    (tmp_path / ".env").write_text(
        env_file.read_text()
        if env_file is not None
        else "ANTHROPIC_API_KEY=sk-fake-anthropic-redact-me\n"
        "ANTHROPIC_BASE_URL=https://gateway.example.com\n"
        "OPENAI_API_KEY=sk-fake-openai-redact-me\n"
        "OPENAI_BASE_URL=https://gateway.example.com/v1\n"
    )
    fake_tasks = tmp_path / "fake-tasks"
    fake_tasks.mkdir(exist_ok=True)

    test_env = dict(os.environ, **(env or {}))
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--tasks-dir",
            str(fake_tasks),
            "--skip-preflight",
            "--dry-run",
            *extra_args,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=test_env,
        timeout=30,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_example_env_passes_literal_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The example must pass a literal URL because the launcher does not expand .env variables."""
    for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "JUDGE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    output = _dry_run(
        tmp_path,
        ["--agent", "opencode", "--model", "claude-haiku-4-5"],
        env_file=REPO_ROOT / ".env.example",
    )
    assert "--agent-env OPENAI_BASE_URL=https://gateway.example.com/v1" in output


@pytest.mark.parametrize(
    ("agent", "model"),
    [
        ("claude-code", "claude-opus-4-6"),
        ("codex", "gpt-5.3-codex"),
        ("opencode", "claude-haiku-4-5"),
    ],
)
def test_dry_run_keeps_api_keys_off_argv(tmp_path: Path, agent: str, model: str):
    """API keys must never reach the harbor command line. The command is
    visible to `ps` at runtime and leaks when users paste dry-run output into
    MRs, Slack, or issue reports. Keys are delivered to the agent via
    os.environ (the launcher's literal `.env` parser), not --agent-env, so they should
    be entirely absent from the command — not merely redacted.
    """
    output = _dry_run(tmp_path, ["--agent", agent, "--model", model])
    assert "sk-fake-anthropic-redact-me" not in output, f"ANTHROPIC_API_KEY leaked in {agent} dry-run output"
    assert "sk-fake-openai-redact-me" not in output, f"OPENAI_API_KEY leaked in {agent} dry-run output"
    # The secret keys must not be forwarded on the command line at all.
    assert "ANTHROPIC_API_KEY=" not in output, f"ANTHROPIC_API_KEY on {agent} command line"
    assert "OPENAI_API_KEY=" not in output, f"OPENAI_API_KEY on {agent} command line"
    assert "LLM_API_KEY=" not in output, f"LLM_API_KEY on {agent} command line"


def test_opencode_gateway_dispatch_prepends_gateway(tmp_path: Path):
    """The gateway/ prefix selects @ai-sdk/openai-compatible in the Harbor patch.

    The launcher adds this prefix to the endpoint's model ID.
    """
    output = _dry_run(tmp_path, ["--agent", "opencode", "--model", "claude-haiku-4-5"])
    assert "--model gateway/claude-haiku-4-5" in output, (
        "expected harbor --model to be prefixed with gateway/ for opencode+gateway"
    )


def test_opencode_vllm_dispatch_prepends_vllm(tmp_path: Path):
    """The vllm/ prefix selects @ai-sdk/openai-compatible in the Harbor patch.

    OPENAI_BASE_URL must identify the configured vLLM server.
    """
    output = _dry_run(
        tmp_path,
        ["--agent", "opencode", "--backend", "vllm", "--model", "model"],
        env={"VLLM_BASE_URL": "http://localhost:9000/v1", "VLLM_API_KEY": "EMPTY"},
    )
    assert "--model vllm/model" in output, "expected harbor --model prefixed with vllm/ for opencode+vllm"
    # Never double-prefixed: passing `vllm/model` should still yield `vllm/model`.


def test_opencode_legacy_vllm_input_is_normalized(tmp_path: Path):
    """An explicit vllm/ prefix must appear exactly once in Harbor's model ID."""
    output = _dry_run(
        tmp_path,
        ["--agent", "opencode", "--backend", "vllm", "--model", "vllm/model"],
        env={"VLLM_BASE_URL": "http://localhost:9000/v1", "VLLM_API_KEY": "EMPTY"},
    )
    assert "--model vllm/model" in output
    assert "vllm/vllm/" not in output


def test_opencode_prefixed_gateway_input_is_normalized(tmp_path: Path):
    """An explicit gateway/ prefix appears exactly once in Harbor's model ID."""
    output = _dry_run(tmp_path, ["--agent", "opencode", "--model", "gateway/claude-haiku-4-5"])
    assert "--model gateway/claude-haiku-4-5" in output, (
        "legacy gateway/ input should normalize back to single-prefix harbor --model"
    )
    assert "gateway/gateway/" not in output


def test_openhands_sdk_gateway_prepends_openai_and_wires_llm_env(tmp_path: Path):
    """openhands-sdk: prepend openai/ so LiteLLM uses chat-completions wire
    format; wire LLM_BASE_URL via --agent-env. LLM_API_KEY is exported into
    os.environ (read by the SDK) rather than --agent-env, to keep the secret
    off the harbor command line.
    """
    output = _dry_run(tmp_path, ["--agent", "openhands-sdk", "--model", "vendor/some-model"])
    assert "--model openai/vendor/some-model" in output
    assert "version=1.17.0" in output
    assert "max_iterations=200" in output
    assert "LLM_API_KEY=" not in output, (
        "LLM_API_KEY must stay off the command line (exported via os.environ)"
    )
    assert "LLM_BASE_URL=" in output


def test_openhands_sdk_idempotent_openai_prefix(tmp_path: Path):
    """If a user already wrote openai/ on the slug, don't double it."""
    output = _dry_run(tmp_path, ["--agent", "openhands-sdk", "--model", "openai/vendor/some-model"])
    assert "--model openai/vendor/some-model" in output


def test_claude_code_includes_max_turns_250(tmp_path: Path):
    """claude-code runs pin --agent-kwarg max_turns=250 (harbor's default
    is lower; the pin is documented in docs/runbooks/baseline-reproducibility.md).
    """
    output = _dry_run(tmp_path, ["--agent", "claude-code", "--model", "claude-opus-4-6"])
    assert "--agent-kwarg max_turns=250" in output, "expected max_turns=250 pin for claude-code"


def test_claude_code_disables_plan_mode_by_default(tmp_path: Path):
    """Default claude-code runs disable EnterPlanMode/ExitPlanMode so
    Haiku doesn't trap itself in plan mode (0 edits failure mode).
    """
    output = _dry_run(tmp_path, ["--agent", "claude-code", "--model", "claude-haiku-4-5"])
    assert "disallowed_tools=EnterPlanMode" in output, "expected EnterPlanMode disable by default"
    assert "ExitPlanMode" in output


def test_disable_plan_mode_env_opt_out(tmp_path: Path):
    """DISABLE_PLAN_MODE=0 is the documented opt-out for ablation runs."""
    output = _dry_run(
        tmp_path, ["--agent", "claude-code", "--model", "claude-haiku-4-5"], env={"DISABLE_PLAN_MODE": "0"}
    )
    assert "disallowed_tools=EnterPlanMode" not in output, (
        "DISABLE_PLAN_MODE=0 should suppress the disallowed_tools kwarg"
    )


def test_other_agents_no_plan_mode_kwarg(tmp_path: Path):
    """opencode/codex have no EnterPlanMode tool; the launcher must not
    pass disallowed_tools to them (would 400 the harbor invocation).
    """
    for agent, model in [
        ("opencode", "claude-haiku-4-5"),
        ("codex", "gpt-5.3-codex"),
    ]:
        output = _dry_run(tmp_path, ["--agent", agent, "--model", model])
        assert "disallowed_tools" not in output, f"{agent} dry-run should not include disallowed_tools kwarg"


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_sanity_agent_no_model_required(tmp_path: Path, agent: str):
    """Oracle/nop don't talk to a model; --model must be optional and
    the launcher must NOT pass `--model` or `--agent-kwarg version=...`
    to harbor.
    """
    output = _dry_run(tmp_path, ["--agent", agent])
    assert "--model " not in output, f"{agent} dry-run should not include --model"
    assert "--agent-kwarg version=" not in output, (
        f"{agent} has no CLI version to pin; --agent-kwarg version=... should be absent"
    )
    # Positive: the harbor command IS rendered with the agent name.
    assert f"--agent {agent}" in output


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_sanity_agent_skips_llm_envs(tmp_path: Path, agent: str):
    """Sanity agents don't speak to LLMs — none of the gateway/vllm env
    plumbing nor the claude-code opt-outs should appear.
    """
    output = _dry_run(tmp_path, ["--agent", agent])
    for env in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "CLAUDE_CODE_ATTRIBUTION_HEADER",
        "CLAUDE_CODE_ENABLE_TELEMETRY",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
        "CLAUDE_CODE_EFFORT_LEVEL",
    ):
        assert env not in output, f"{agent} dry-run should not include {env} in --agent-env"


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_sanity_agent_skips_agent_specific_kwargs(tmp_path: Path, agent: str):
    """No reasoning_effort, max_turns, or disallowed_tools for sanity agents."""
    output = _dry_run(tmp_path, ["--agent", agent])
    for kwarg in ("reasoning_effort", "max_turns", "disallowed_tools"):
        assert kwarg not in output, f"{agent} dry-run should not include --agent-kwarg {kwarg}"


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_sanity_agent_keeps_determinism_envs(tmp_path: Path, agent: str):
    """Determinism envs apply to every harbor run regardless of agent —
    in-container pytest reproducibility doesn't depend on what agent
    drove the run.
    """
    output = _dry_run(tmp_path, ["--agent", agent])
    assert "PYTHONHASHSEED=0" in output, f"{agent} dry-run missing PYTHONHASHSEED determinism env"
    assert "LC_ALL=C.UTF-8" in output, f"{agent} dry-run missing LC_ALL determinism env"


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_sanity_agent_with_model_drops_it(tmp_path: Path, agent: str):
    """If user passes --model anyway (e.g. CI script that always sets it),
    the launcher should drop it silently and warn — not crash, not
    forward it to harbor.
    """
    output = _dry_run(tmp_path, ["--agent", agent, "--model", "claude-opus-4-6"])
    assert "--model claude-opus-4-6" not in output
    # Warning should appear in output.
    assert "ignored" in output.lower(), f"{agent} should print a warning when --model is provided"


def test_unknown_agent_rejected(tmp_path: Path):
    """The launcher must reject unknown agent names early, not let them
    flow through to harbor where the error message is less obvious.
    """
    output = _dry_run(tmp_path, ["--agent", "totally-fake-agent"], expected_status=1)
    assert "must be claude-code, codex, opencode, openhands-sdk, qwen-coder, pi, oracle, or nop" in output
