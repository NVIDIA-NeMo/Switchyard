# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmark/nemo_gym/run.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def fake_runner_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """Use local tool doubles and an event log to check subprocess lifecycle ordering."""
    gym_dir = tmp_path / "Gym"
    bin_dir = tmp_path / "bin"
    results_dir = tmp_path / "results"
    event_log = tmp_path / "events.log"
    gym_bin = gym_dir / ".venv/bin"
    gym_bin.mkdir(parents=True)
    bin_dir.mkdir()

    _write_executable(
        gym_bin / "python",
        """#!/bin/bash
if [[ "${1:-}" == "$FAKE_COMPARE_PATH" ]]; then
    printf 'compare:%s\\n' "$*" >> "$EVENT_LOG"
    printf 'fixture comparison\\n'
    exit 0
fi
exec "$REAL_PYTHON" "$@"
""",
    )
    _write_executable(
        gym_bin / "gym",
        """#!/bin/bash
set -u
if [[ "${1:-}" == "eval" && "${2:-}" == "prepare" ]]; then
    printf 'prepare:%s\\n' "$*" >> "$EVENT_LOG"
    mkdir -p "$(dirname "$FAKE_BENCHMARK_DATA")"
    printf '{"task": 1}\\n' > "$FAKE_BENCHMARK_DATA"
    exit 0
fi
route=""
output=""
previous=""
for argument in "$@"; do
    if [[ "$previous" == "--model" ]]; then route="$argument"; fi
    if [[ "$previous" == "--output" ]]; then output="$argument"; fi
    previous="$argument"
done
if [[ "${FAKE_GYM_BEHAVIOR:-}" == "wait" ]]; then
    exec "$REAL_PYTHON" -c '
import os
import signal
import sys

route = sys.argv[1]
arguments = " ".join(sys.argv[2:])

def stop(number, _frame):
    with open(os.environ["EVENT_LOG"], "a", encoding="utf-8") as stream:
        stream.write(f"gym_stop:{route}:{os.getpid()}\\n")
    raise SystemExit(128 + number)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
with open(os.environ["EVENT_LOG"], "a", encoding="utf-8") as stream:
    stream.write(f"gym_start:{route}:{os.getpid()}:{arguments}\\n")
signal.pause()
' "$route" "$@"
fi
printf 'gym_start:%s:%s:%s\\n' "$route" "$$" "$*" >> "$EVENT_LOG"
run_dir="$(dirname "$output")"
mkdir -p "$run_dir"
printf '{"route": "%s"}\\n' "$route" > "$output"
if [[ "${FAKE_GYM_BEHAVIOR:-}" != "missing_ledger" ]]; then
    printf '{"event":"start"}\\n{"event":"finish"}\\n' > "$run_dir/litellm-calls.jsonl"
fi
printf 'gym_done:%s\\n' "$route" >> "$EVENT_LOG"
if [[ "$route" == "fixed" && "${FAKE_GYM_BEHAVIOR:-}" == "exit7" ]]; then exit 7; fi
if [[ "$route" == "fixed" && "${FAKE_GYM_BEHAVIOR:-}" == "sidecar" ]]; then
    printf '{"failure": true}\\n' > "$run_dir/rollouts_failures.jsonl"
fi
""",
    )
    _write_executable(
        bin_dir / "uv",
        """#!/bin/bash
set -u
printf 'uv:%s\\n' "$*" >> "$EVENT_LOG"
"$REAL_PYTHON" - "$NEMO_GYM_LITELLM_RESULTS" "$NEMO_GYM_LITELLM_PROFILE" \
    "$SWITCHYARD_LITELLM_CONFIG" "$FAKE_CALLBACK_PATH" "$NVIDIA_BASE_URL" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

results, profile, routing, callback = map(pathlib.Path, sys.argv[1:5])
runtime = {
    "mode": "litellm_libsy",
    "instance_id": os.environ["NEMO_GYM_LITELLM_INSTANCE_ID"],
    "litellm_version": "1.97.0",
    "switchyard_version": "0.2.0",
    "fastapi_version": "0.136.3",
    "starlette_version": "1.3.1",
    "routing_plugin": "switchyard_litellm.RandomRoutingPlugin",
    "models": {
        "fixed": ["nvidia_nim/nvidia/nemotron-3-super-120b-a12b"],
        "routed": [
            "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
            "nvidia_nim/openai/gpt-oss-20b",
        ],
    },
    "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
    "routing_sha256": hashlib.sha256(routing.read_bytes()).hexdigest(),
    "callback_sha256": hashlib.sha256(callback.read_bytes()).hexdigest(),
    "provider_base_sha256": hashlib.sha256(sys.argv[5].encode()).hexdigest(),
}
results.mkdir(parents=True, exist_ok=True)
(results / "litellm-runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
PY
exec "$REAL_PYTHON" -c '
import os
import signal
import sys

arguments = " ".join(sys.argv[1:])

def stop(_number, _frame):
    with open(os.environ["EVENT_LOG"], "a", encoding="utf-8") as stream:
        stream.write(f"proxy_stop:{os.getpid()}\\n")
    raise SystemExit(0)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
with open(os.environ["EVENT_LOG"], "a", encoding="utf-8") as stream:
    stream.write(f"proxy_start:{os.getpid()}:{arguments}\\n")
signal.pause()
' "$@"
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/bin/bash
url=""
for argument in "$@"; do case "$argument" in http://*) url="$argument" ;; esac; done
printf 'curl:%s\\n' "$url" >> "$EVENT_LOG"
[[ -s "$NEMO_GYM_LITELLM_RESULTS/litellm-runtime.json" ]] || exit 22
printf '{}\\n'
""",
    )
    _write_executable(
        bin_dir / "git",
        """#!/bin/bash
printf 'git:%s\\n' "$*" >> "$EVENT_LOG"
case "$*" in
    *" status "*) exit 0 ;;
    *" rev-parse "*) printf '%039d1\\n' 0 ;;
    *" describe "*) printf '%040d-dirty\\n' 2 ;;
esac
""",
    )
    for tool in ("cargo", "switchyard-server"):
        _write_executable(
            bin_dir / tool,
            f'#!/bin/bash\nprintf \'forbidden:{tool}:%s\\n\' "$*" >> "$EVENT_LOG"\nexit 99\n',
        )

    env = os.environ.copy()
    for key in (
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "NVIDIA_BASE_URL",
        "LITELLM_CONFIG",
        "SWITCHYARD_CONFIG",
        "LITELLM_PORT",
    ):
        env.pop(key, None)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "GYM_DIR": str(gym_dir),
            "RESULTS_DIR": str(results_dir),
            "EVENT_LOG": str(event_log),
            "FAKE_BENCHMARK_DATA": str(
                gym_dir / "benchmarks/mmlu-redux/data/mmlu-redux_benchmark.jsonl"
            ),
            "FAKE_COMPARE_PATH": str(ROOT / "benchmark/nemo_gym/compare.py"),
            "FAKE_CALLBACK_PATH": str(ROOT / "benchmark/nemo_gym/gym_routing_plugin.py"),
            "REAL_PYTHON": sys.executable,
        }
    )
    return env, results_dir, event_log


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _events(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_help_and_shell_syntax_need_no_setup(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    result = subprocess.run(
        ["/bin/bash", str(RUNNER), "--help"],
        cwd=tmp_path,
        env={"PATH": os.defpath},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    for name in (
        "GYM_DIR",
        "LITELLM_CONFIG",
        "SWITCHYARD_CONFIG",
        "NVIDIA_BASE_URL",
        "LITELLM_PORT",
        "LIMIT",
        "REPEATS",
        "CONCURRENCY",
    ):
        assert name in result.stdout
    assert list(tmp_path.iterdir()) == before
    assert subprocess.run(["/bin/bash", "-n", str(RUNNER)], check=False).returncode == 0


@pytest.mark.parametrize("invalid", ["existing-results", "zero-limit", "bad-port"])
def test_preflight_rejects_before_operations(
    fake_runner_env: tuple[dict[str, str], Path, Path], invalid: str
) -> None:
    env, results_dir, event_log = fake_runner_env
    if invalid == "existing-results":
        results_dir.mkdir()
        sentinel = results_dir / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
    else:
        sentinel = None
        env["LIMIT" if invalid == "zero-limit" else "LITELLM_PORT"] = "0"
    result = _run(env)
    assert result.returncode != 0
    assert _events(event_log) == []
    if sentinel is not None:
        assert sentinel.read_text(encoding="utf-8") == "keep"


def test_success_uses_one_stock_proxy_for_fixed_and_routed(
    fake_runner_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, results_dir, event_log = fake_runner_env
    env.update(
        {
            "LIMIT": "2",
            "REPEATS": "3",
            "CONCURRENCY": "1",
            "NEMO_GYM_LITELLM_INSTANCE_ID": "must-not-be-reused",
        }
    )
    result = _run(env)
    assert result.returncode == 0, result.stderr
    events = _events(event_log)

    preparations = [line for line in events if line.startswith("prepare:")]
    assert len(preparations) == 1
    assert "eval prepare --benchmark mmlu-redux" in preparations[0]
    starts = [line for line in events if line.startswith("proxy_start:")]
    assert len(starts) == 1
    proxy_pid = int(starts[0].split(":", 2)[1])
    for pin in (
        "litellm[proxy]==1.97.0",
        "fastapi==0.136.3",
        "starlette==1.3.1",
        "--num_workers 1",
    ):
        assert pin in starts[0]
    gym_runs = [line for line in events if line.startswith("gym_start:")]
    assert [line.split(":", 2)[1] for line in gym_runs] == ["fixed", "routed"]
    for route, invocation in zip(("fixed", "routed"), gym_runs, strict=True):
        run_dir = results_dir / route
        for expected in (
            "--benchmark mmlu-redux",
            "--model-type litellm_model",
            f"--model {route}",
            "--split benchmark",
            "--limit 2",
            "--num-repeats 3",
            "--concurrency 1",
            "--temperature 0",
            "--max-output-tokens 4096",
            "++policy_base_url=http://127.0.0.1:4000/v1",
            "++policy_api_key=unused",
            f"++model_call_capture_dir={run_dir}/model-calls",
            f"++nemo_gym_log_dir={run_dir}/server-logs",
            "++mcqa_simple_agent.responses_api_agents.simple_agent.max_steps=1",
        ):
            assert expected in invocation
        assert "switchyard_model" not in invocation
        assert ".deployment=" not in invocation
        assert "condition_dir" not in invocation
        assert (
            "++mmlu-redux_mcqa_simple_agent.responses_api_agents.simple_agent.max_steps="
            not in invocation
        )
    assert not any(line.startswith("forbidden:") for line in events)

    provenances = [
        json.loads((results_dir / route / "run-provenance.json").read_text())
        for route in ("fixed", "routed")
    ]
    assert provenances[0] == provenances[1]
    runtime = provenances[0]["runtime"]
    assert len(runtime["instance_id"]) == 32
    assert all(character in "0123456789abcdef" for character in runtime["instance_id"])
    assert runtime["instance_id"] != "must-not-be-reused"
    assert runtime["models"] == {
        "fixed": ["nvidia_nim/nvidia/nemotron-3-super-120b-a12b"],
        "routed": [
            "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
            "nvidia_nim/openai/gpt-oss-20b",
        ],
    }
    comparison = next(i for i, line in enumerate(events) if line.startswith("compare:"))
    stop = next(i for i, line in enumerate(events) if line.startswith("proxy_stop:"))
    assert all(events.index(f"gym_done:{route}") < stop for route in ("fixed", "routed"))
    assert stop < comparison
    assert not _pid_is_alive(proxy_pid)
    assert (results_dir / "comparison.txt").read_text(encoding="utf-8") == "fixture comparison\n"


@pytest.mark.parametrize("behavior", ["exit7", "sidecar"])
def test_fixed_failure_stops_before_routed_and_keeps_logs(
    fake_runner_env: tuple[dict[str, str], Path, Path], behavior: str
) -> None:
    env, results_dir, event_log = fake_runner_env
    env["FAKE_GYM_BEHAVIOR"] = behavior
    result = _run(env)
    events = _events(event_log)
    assert result.returncode != 0
    assert any(line.startswith("gym_start:fixed:") for line in events)
    assert not any(line.startswith("gym_start:routed:") for line in events)
    assert not any(line.startswith("compare:") for line in events)
    assert sum(line.startswith("proxy_stop:") for line in events) == 1
    assert (results_dir / "fixed/gym.log").exists()
    assert (results_dir / "litellm.log").exists()
    if behavior == "sidecar":
        assert (results_dir / "fixed/rollouts_failures.jsonl").stat().st_size > 0


def test_missing_ledger_stops_before_comparison(
    fake_runner_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, _, event_log = fake_runner_env
    env["FAKE_GYM_BEHAVIOR"] = "missing_ledger"
    result = _run(env)
    events = _events(event_log)
    assert result.returncode != 0
    assert "missing LiteLLM request evidence for fixed" in result.stderr
    assert not any(line.startswith("gym_start:routed:") for line in events)
    assert not any(line.startswith("compare:") for line in events)
    assert sum(line.startswith("proxy_stop:") for line in events) == 1


@pytest.mark.parametrize(
    ("signal_number", "expected_status"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_interrupt_stops_only_owned_children(
    fake_runner_env: tuple[dict[str, str], Path, Path],
    signal_number: signal.Signals,
    expected_status: int,
) -> None:
    env, _, event_log = fake_runner_env
    env["FAKE_GYM_BEHAVIOR"] = "wait"
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    runner = subprocess.Popen(
        ["/bin/bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    owned_pids: list[int] = []
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            starts = [
                line
                for line in _events(event_log)
                if line.startswith("proxy_start:") or line.startswith("gym_start:fixed:")
            ]
            if len(starts) == 2:
                break
            time.sleep(0.05)
        else:
            pytest.fail("fake Gym did not enter its wait state")
        owned_pids = [
            int(line.split(":", 2)[1] if line.startswith("proxy_start:") else line.split(":", 3)[2])
            for line in starts
        ]
        runner.send_signal(signal_number)
        stdout, stderr = runner.communicate(timeout=20)
        assert runner.returncode == expected_status, (stdout, stderr)
        events = _events(event_log)
        assert any(line.startswith("gym_stop:fixed:") for line in events)
        assert sum(line.startswith("proxy_stop:") for line in events) == 1
        assert all(not _pid_is_alive(pid) for pid in owned_pids)
        assert sentinel.poll() is None
    finally:
        if runner.poll() is None:
            os.killpg(runner.pid, signal.SIGKILL)
            runner.wait(timeout=5)
        for pid in owned_pids:
            if _pid_is_alive(pid):
                os.kill(pid, signal.SIGKILL)
        sentinel.terminate()
        sentinel.wait(timeout=5)
