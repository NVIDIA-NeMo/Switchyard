#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run fixed-model and Random-routed conditions over the same NeMo Gym rollouts.

set +x
set -euo pipefail

# Step 1: Choose settings and check the setup before starting any services.
# Environment variables override these defaults; each run needs a fresh results directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWITCHYARD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILE="$SCRIPT_DIR/litellm.yaml"
ROUTING="$SCRIPT_DIR/routes.toml"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${LITELLM_PORT:-4000}"
LIMIT="${LIMIT:-5}"
REPEATS="${REPEATS:-1}"
CONCURRENCY="${CONCURRENCY:-1}"

if [[ $# -gt 0 ]]; then
    cat <<'EOF'
Usage: bash benchmark/nemo_gym/run.sh
Set GYM_DIR to the Gym checkout from the README setup.

Optional environment variables (defaults):
  LIMIT=5, REPEATS=1, CONCURRENCY=1, LITELLM_PORT=4000
  RESULTS_DIR         Fresh output directory (results/<UTC timestamp>)
  NVIDIA_BASE_URL     Provider endpoint (NVIDIA API Catalog /v1)
Provider credentials come from the environment variables named in litellm.yaml.
EOF
    [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]] && exit 0
    exit 2
fi

die() { echo "error: $*" >&2; exit 1; }
[[ -n "${GYM_DIR:-}" ]] || die "set GYM_DIR; see the README setup"
GYM_DIR="$(cd "$GYM_DIR" && pwd)"
[[ "$RESULTS_DIR" = /* ]] || RESULTS_DIR="$PWD/$RESULTS_DIR"
[[ -f "$PROFILE" && -f "$ROUTING" ]] || die "LiteLLM profile or routing TOML does not exist"
[[ ! -e "$RESULTS_DIR" && ! -L "$RESULTS_DIR" ]] || die "results path already exists: $RESULTS_DIR"
[[ "$GYM_DIR" =~ ^[a-zA-Z0-9_./-]+$ && "$RESULTS_DIR" =~ ^[a-zA-Z0-9_./-]+$ ]] || die "Gym workspace/output paths must not contain spaces or shell metacharacters"
for value in "$LIMIT" "$REPEATS" "$CONCURRENCY" "$PORT"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "limits, repeats, concurrency and port must be positive integers"
done
GYM="$GYM_DIR/.venv/bin/gym"
PYTHON="$GYM_DIR/.venv/bin/python"
[[ -x "$GYM" && -x "$PYTHON" ]] || die "complete the README's Gym setup"
export PATH="$GYM_DIR/.venv/bin:$PATH"
for tool in git uv curl; do command -v "$tool" >/dev/null || die "missing required tool: $tool"; done
GYM_REVISION="$(git -C "$GYM_DIR" describe --always --dirty --abbrev=40)"
SWITCHYARD_REVISION="$(git -C "$SWITCHYARD_ROOT" describe --always --dirty --abbrev=40)"
"$PYTHON" - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
if not 1 <= port <= 65535 or port == 11000:
    raise SystemExit('error: choose a LiteLLM port in 1..65535 other than Gym port 11000')
for value in (port, 11000):
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', value))
    except OSError:
        raise SystemExit(f'error: port {value} is occupied; do not share a running Gym/proxy') from None
PY
umask 077
mkdir -p "$(dirname "$RESULTS_DIR")"
mkdir "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
BENCHMARK_DATA="$GYM_DIR/benchmarks/mmlu-redux/data/mmlu-redux_benchmark.jsonl"
ROOT_URL="http://127.0.0.1:$PORT"
GYM_PID=""
PROXY_PID=""

# Track the processes started by this script so exits and interrupts can stop them.
stop_process() {
    local pid="$1"
    [[ -n "$pid" ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
        for _ in {1..100}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        if kill -0 "$pid" 2>/dev/null; then
            echo "warning: escalating shutdown of run-owned process $pid" >&2
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    wait "$pid" 2>/dev/null || true
}
cleanup() {
    local status=$?
    trap - EXIT
    trap '' INT TERM
    stop_process "$GYM_PID"
    stop_process "$PROXY_PID"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_gym() {
    local log_file="$1" status=0
    shift
    (cd "$GYM_DIR"; exec "$PYTHON" -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])' "$GYM" "$@") >"$log_file" 2>&1 &
    GYM_PID=$!
    wait "$GYM_PID" || status=$?
    GYM_PID=""
    return "$status"
}

# Step 2: Prepare MMLU-Redux, reusing the prepared dataset if it already exists.
echo "Results: $RESULTS_DIR"
echo "Preparing up to $LIMIT tasks; repeats: $REPEATS; concurrency: $CONCURRENCY"
if [[ ! -s "$BENCHMARK_DATA" ]]; then
    run_gym "$RESULTS_DIR/prepare.log" eval prepare --benchmark mmlu-redux "hydra.run.dir=$RESULTS_DIR/hydra-prepare" || die "preparation failed; see $RESULTS_DIR/prepare.log"
fi
[[ -s "$BENCHMARK_DATA" ]] || die "MMLU-Redux preparation produced no data"

# Step 3: Start a local LiteLLM server and wait until it is ready.
# litellm.yaml defines the model groups; routes.toml configures Switchyard Random routing.
# Switchyard runs as a library inside LiteLLM, not as a separate server.
export NVIDIA_BASE_URL="${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}"
export SWITCHYARD_LITELLM_CONFIG="$ROUTING"
export NEMO_GYM_LITELLM_RESULTS="$RESULTS_DIR"
echo "Starting LiteLLM at $ROOT_URL; see $RESULTS_DIR/litellm.log"
PYTHONPATH="$SCRIPT_DIR:$SWITCHYARD_ROOT/examples/litellm/src${PYTHONPATH:+:$PYTHONPATH}" \
    uv run --project "$SWITCHYARD_ROOT/examples/litellm" --locked \
    --with 'litellm[proxy]==1.102.0' --with 'fastapi==0.136.3' --with 'starlette==1.3.1' \
    litellm --config "$PROFILE" --host 127.0.0.1 --port "$PORT" --num_workers 1 \
    >"$RESULTS_DIR/litellm.log" 2>&1 &
PROXY_PID=$!
ready=false
for _ in {1..240}; do
    kill -0 "$PROXY_PID" 2>/dev/null || die "LiteLLM exited; see $RESULTS_DIR/litellm.log"
    if curl -fsS --max-time 2 "$ROOT_URL/health/readiness/details" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || die "LiteLLM readiness timed out; see $RESULTS_DIR/litellm.log"

# Step 4: Save the checkout versions once for reference and prepare the output folders.
# Pairing uses Gym's materialized inputs, not configuration fingerprints.
printf 'Gym: %s\nSwitchyard: %s\n' "$GYM_REVISION" "$SWITCHYARD_REVISION" >"$RESULTS_DIR/versions.txt"
mkdir -p "$RESULTS_DIR"/{fixed,routed}/model-calls

# Step 5: Evaluate the fixed baseline, then Random routing, on the same tasks.
# Gym's litellm_model adapter calls the proxy; fixed and routed name its model groups.
# Only the group and output paths change; the evaluation settings stay the same.
# Both conditions use one dedicated LiteLLM instance and separate Gym capture paths.
for route in fixed routed; do
    run_dir="$RESULTS_DIR/$route"
    echo "Gym evaluation: gym eval run --benchmark mmlu-redux --model-type litellm_model --model $route"
    echo "Running $route; see $run_dir/gym.log"
    run_gym "$run_dir/gym.log" eval run \
        --benchmark mmlu-redux --model-type litellm_model --model "$route" \
        --output "$run_dir/rollouts.jsonl" --split benchmark \
        --limit "$LIMIT" --num-repeats "$REPEATS" --concurrency "$CONCURRENCY" \
        --temperature 0 --max-output-tokens 8192 \
        "++policy_base_url=$ROOT_URL/v1" ++policy_api_key=unused \
        ++route_failures_to_sidecar=true ++observability_enabled=true \
        "++model_call_capture_dir=$run_dir/model-calls" \
        "++nemo_gym_log_dir=$run_dir/server-logs" \
        ++mcqa_simple_agent.responses_api_agents.simple_agent.max_steps=1 \
        "hydra.run.dir=$run_dir/hydra" || die "$route evaluation failed; see $run_dir/gym.log"
    # Stop before the next condition if Gym reports terminal rollout failures.
    [[ ! -s "$run_dir/rollouts_failures.jsonl" ]] || die "$route has terminal rollout failures; see $run_dir/rollouts_failures.jsonl"
done

# Step 6: Stop LiteLLM, then compare the saved fixed and routed runs.
# compare.py checks complete, paired evidence before reporting rewards, tokens, and latency.
stop_process "$PROXY_PID"
PROXY_PID=""
echo "Comparing fixed and routed results"
"$PYTHON" "$SCRIPT_DIR/compare.py" "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/comparison.txt"
echo "Comparison written to $RESULTS_DIR/comparison.txt"
