#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run fixed-model and classifier-routed conditions over the same NeMo Gym rollouts.

set -euo pipefail

GYM_REVISION="e044a8ca795ece2c69b053d30c0a8dea7fa3b9f3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWITCHYARD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_DEPLOYMENT="$SCRIPT_DIR/routes.toml"
DEPLOYMENT="${SWITCHYARD_CONFIG:-$DEFAULT_DEPLOYMENT}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${SWITCHYARD_PORT:-4000}"
LIMIT="${LIMIT:-5}"
REPEATS="${REPEATS:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
[[ "$DEPLOYMENT" = /* ]] || DEPLOYMENT="$PWD/$DEPLOYMENT"
[[ "$RESULTS_DIR" = /* ]] || RESULTS_DIR="$PWD/$RESULTS_DIR"

usage() {
    cat <<EOF
Run matching MMLU-Redux rollouts through fixed and routed Switchyard conditions.

Usage:
  bash benchmark/nemo_gym/run.sh

Required environment:
  GYM_DIR             NeMo Gym checkout at $GYM_REVISION

Optional environment:
  SWITCHYARD_CONFIG   Deployment defining strong-only and policy-model
                      (default: benchmark/nemo_gym/routes.toml)
  RESULTS_DIR         Output directory (default: benchmark/nemo_gym/results/<timestamp>)
  SWITCHYARD_PORT     Local proxy port (default: 4000)
  LIMIT               Number of prepared tasks (default: 5)
  REPEATS             Rollouts per task (default: 1)
  CONCURRENCY         Concurrent Gym samples (default: 1)
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
        usage
        exit 0
    fi
    echo "error: unexpected argument: $1" >&2
    usage >&2
    exit 2
fi

[[ -n "${GYM_DIR:-}" ]] || die "set GYM_DIR to a NeMo Gym checkout at $GYM_REVISION"
[[ -f "$DEPLOYMENT" ]] || die "Switchyard configuration does not exist: $DEPLOYMENT"

GYM="$GYM_DIR/.venv/bin/gym"
BENCHMARK_DATA="$GYM_DIR/benchmarks/mmlu-redux/data/mmlu-redux_benchmark.jsonl"
TARGET_DIR="${CARGO_TARGET_DIR:-$SWITCHYARD_ROOT/target}"
[[ "$TARGET_DIR" = /* ]] || TARGET_DIR="$PWD/$TARGET_DIR"
SERVER="$TARGET_DIR/release/switchyard-server"
[[ -x "$GYM" ]] || die "run 'uv sync --frozen --no-dev' in $GYM_DIR"
[[ "$(git -C "$GYM_DIR" rev-parse HEAD)" == "$GYM_REVISION" ]] || {
    die "Gym must be checked out at $GYM_REVISION"
}
[[ -z "$(git -C "$GYM_DIR" status --porcelain)" ]] || {
    die "Gym checkout must be clean so its recorded revision is exact"
}
[[ ! -e "$RESULTS_DIR" ]] || die "results path already exists: $RESULTS_DIR"
if curl -sS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    die "port $PORT is already in use; set SWITCHYARD_PORT to a free port"
fi

echo "Results: $RESULTS_DIR"
cargo build --manifest-path "$SWITCHYARD_ROOT/Cargo.toml" --release -p switchyard-server
if [[ ! -s "$BENCHMARK_DATA" ]]; then
    (cd "$GYM_DIR" && "$GYM" eval prepare --benchmark mmlu-redux)
fi

SWITCHYARD_REVISION="$(git -C "$SWITCHYARD_ROOT" describe --always --dirty)"
SERVER_PID=""

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID"
        wait "$SERVER_PID" || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
    local log_file="$1"
    for _ in {1..80}; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            cat "$log_file" >&2
            return 1
        fi
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return
        fi
        sleep 0.25
    done
    echo "Switchyard did not become healthy; see $log_file" >&2
    return 1
}

run_condition() {
    local route="$1"
    local run_dir="$RESULTS_DIR/$route"
    local root_url="http://127.0.0.1:$PORT"
    local run_status=0
    echo "Running condition: $route"
    mkdir -p "$run_dir/model-calls"
    cp "$DEPLOYMENT" "$run_dir/routes.toml"

    # Router state and /v1/stats are process-wide, so each condition gets a fresh proxy.
    "$SERVER" --config "$DEPLOYMENT" --host 127.0.0.1 --port "$PORT" \
        >"$run_dir/switchyard.log" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$run_dir/switchyard.log"

    (
        cd "$GYM_DIR" || exit 1
        "$GYM" eval run \
            --benchmark mmlu-redux \
            --model-type switchyard_model \
            --model "$route" \
            --output "$run_dir/rollouts.jsonl" \
            --split benchmark \
            --limit "$LIMIT" \
            --num-repeats "$REPEATS" \
            --concurrency "$CONCURRENCY" \
            --temperature 0 \
            --max-output-tokens 1024 \
            +route_failures_to_sidecar=true \
            ++observability_enabled=true \
            ++model_call_capture_dir="$run_dir/model-calls" \
            ++policy_model.responses_api_models.switchyard_model.switchyard_base_url="$root_url/v1" \
            ++policy_model.responses_api_models.switchyard_model.condition_dir="$run_dir" \
            ++policy_model.responses_api_models.switchyard_model.proxy_provenance.gym_revision="$GYM_REVISION" \
            ++policy_model.responses_api_models.switchyard_model.proxy_provenance.switchyard_revision="$SWITCHYARD_REVISION"
    ) || run_status=$?

    # Capture proxy diagnostics even when Gym fails.
    if ! curl -fsS "$root_url/v1/stats" -o "$run_dir/switchyard-stats-raw.json"; then
        echo "warning: could not capture Switchyard stats for $route" >&2
        [[ "$run_status" -ne 0 ]] || run_status=1
    fi
    if ! curl -fsS "$root_url/metrics" -o "$run_dir/switchyard-metrics.prom"; then
        echo "warning: could not capture Switchyard metrics for $route" >&2
        [[ "$run_status" -ne 0 ]] || run_status=1
    fi
    stop_server
    return "$run_status"
}

run_condition strong-only
run_condition policy-model
"$GYM_DIR/.venv/bin/python" "$SCRIPT_DIR/compare.py" \
    "$RESULTS_DIR/strong-only" "$RESULTS_DIR/policy-model" | tee "$RESULTS_DIR/comparison.json"
echo "Comparison written to $RESULTS_DIR/comparison.json"
