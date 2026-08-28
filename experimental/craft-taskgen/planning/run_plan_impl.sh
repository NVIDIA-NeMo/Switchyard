#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Iterative planning pipeline: planner -> implementer -> score.
#
# Two stock harbor invocations with plan injection between them,
# plus a scoring step. One command in, one reward.txt out.
#
# (Note: "e2e" in CRAFT paper terminology means a single-phase baseline
# without a separate plan. This script is plan+implementer, not e2e.)
#
# Required env vars:
#   PLANNER_MODEL   harbor -m value for planner (e.g. aws/anthropic/bedrock-claude-opus-4-6)
#
# Optional env vars:
#   IMPL_MODEL            harbor -m value for implementer. Default: aws/anthropic/claude-haiku-4-5-v1
#                         (Haiku 4.5 is the methodology's fixed implementer; override to ablate.)
#   PLANNER_AGENT         default: claude-code
#   IMPL_AGENT            default: claude-code
#   PLANNER_AGENT_KWARGS  space-separated "k=v" pairs passed as --agent-kwarg to phase 1
#   IMPL_AGENT_KWARGS     space-separated "k=v" pairs passed as --agent-kwarg to phase 2
#   SOURCE_DATASET        path to a harbor task dir (absolute or relative to repo root).
#                         Default: harbor-tasks/craft-planning
#   TASKS                 glob forwarded to wrap_pipeline.py --filter (e.g. 'hugapi__hug-*')
#   N_PARALLEL            harbor -n, default 8
#   OUT_DIR               output root, default jobs/plan-impl-<timestamp>
#
# Example:
#   PLANNER_MODEL=aws/anthropic/bedrock-claude-opus-4-6 \
#   PLANNER_AGENT_KWARGS="api_base=$OPENAI_BASE_URL" \
#   IMPL_AGENT_KWARGS="api_base=$OPENAI_BASE_URL" \
#   TASKS='hugapi__hug-651' \
#   ./planning/run_plan_impl.sh

set -euo pipefail

: "${PLANNER_MODEL:?PLANNER_MODEL required (the planner model is what varies in this benchmark)}"

IMPL_MODEL="${IMPL_MODEL:-aws/anthropic/claude-haiku-4-5-v1}"
PLANNER_AGENT="${PLANNER_AGENT:-claude-code}"
IMPL_AGENT="${IMPL_AGENT:-claude-code}"
SOURCE_DATASET="${SOURCE_DATASET:-harbor-tasks/craft-planning}"
N_PARALLEL="${N_PARALLEL:-8}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT_DIR="${OUT_DIR:-jobs/plan-impl-${TIMESTAMP}}"
TASKS="${TASKS:-}"
PLANNER_AGENT_KWARGS="${PLANNER_AGENT_KWARGS:-}"
IMPL_AGENT_KWARGS="${IMPL_AGENT_KWARGS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve SOURCE_DATASET: accept absolute or relative (to repo root).
if [[ "$SOURCE_DATASET" = /* ]]; then
    SOURCE_DATASET_ABS="$SOURCE_DATASET"
else
    SOURCE_DATASET_ABS="$REPO_ROOT/$SOURCE_DATASET"
fi

if [[ ! -d "$SOURCE_DATASET_ABS" ]]; then
    echo "ERROR: SOURCE_DATASET not found: $SOURCE_DATASET_ABS" >&2
    exit 2
fi

if ! command -v harbor >/dev/null 2>&1; then
    echo "ERROR: harbor CLI not on PATH. Activate your harbor venv first." >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
ABS_OUT="$(cd "$OUT_DIR" && pwd)"

PLANNER_DATASET="$ABS_OUT/planner-dataset"
PLANNER_JOB="$ABS_OUT/planner"
IMPL_DATASET="$ABS_OUT/impl-dataset"
IMPL_JOB="$ABS_OUT/implementer"
RESULTS_DIR="$ABS_OUT/results"

# Persist config for reproducibility.
cat > "$ABS_OUT/run_config.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "source_dataset": "$SOURCE_DATASET",
  "planner_agent": "$PLANNER_AGENT",
  "planner_model": "$PLANNER_MODEL",
  "planner_agent_kwargs": "$PLANNER_AGENT_KWARGS",
  "impl_agent": "$IMPL_AGENT",
  "impl_model": "$IMPL_MODEL",
  "impl_agent_kwargs": "$IMPL_AGENT_KWARGS",
  "tasks_filter": "$TASKS",
  "n_parallel": $N_PARALLEL
}
EOF

build_kwarg_args() {
    local kwargs="$1"
    local -a out=()
    if [[ -n "$kwargs" ]]; then
        local pair
        for pair in $kwargs; do
            out+=(--agent-kwarg "$pair")
        done
    fi
    printf '%s\n' "${out[@]}"
}

planner_kwarg_args=()
if [[ -n "$PLANNER_AGENT_KWARGS" ]]; then
    mapfile -t planner_kwarg_args < <(build_kwarg_args "$PLANNER_AGENT_KWARGS")
fi
impl_kwarg_args=()
if [[ -n "$IMPL_AGENT_KWARGS" ]]; then
    mapfile -t impl_kwarg_args < <(build_kwarg_args "$IMPL_AGENT_KWARGS")
fi

filter_args=()
if [[ -n "$TASKS" ]]; then
    filter_args+=(--filter "$TASKS")
fi

# claude-code gateway opt-outs. These were previously set unconditionally by
# the harbor claude_code.py patch; that patch now omits them (they moved to
# scripts/run-baselines.sh --agent-env), so this direct-harbor caller passes
# them explicitly. Disables the attribution header (KV-cache invalidation =>
# ~90% slower gateway inference), telemetry, and the context_management beta
# field. Inert for non-claude agents.
claude_env_args=(
    --agent-env CLAUDE_CODE_ATTRIBUTION_HEADER=0
    --agent-env CLAUDE_CODE_ENABLE_TELEMETRY=0
    --agent-env CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
)

echo "=== [1/5] wrapping dataset in planner mode ==="
python3 "$SCRIPT_DIR/wrap_pipeline.py" \
    --source "$SOURCE_DATASET_ABS" \
    --mode planner \
    --output "$PLANNER_DATASET" \
    "${filter_args[@]}"

echo "=== [2/5] harbor run: planner (${PLANNER_AGENT} / ${PLANNER_MODEL}) ==="
harbor run \
    -p "$PLANNER_DATASET" \
    -a "$PLANNER_AGENT" \
    -m "$PLANNER_MODEL" \
    -o "$PLANNER_JOB" \
    -n "$N_PARALLEL" \
    "${claude_env_args[@]}" \
    "${planner_kwarg_args[@]}"

# Harbor creates a timestamped subdir inside -o. Resolve it.
PLANNER_JOB_INNER="$(find "$PLANNER_JOB" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "$PLANNER_JOB_INNER" ]]; then
    echo "ERROR: no planner job dir produced under $PLANNER_JOB" >&2
    exit 3
fi

echo "=== [3/5] wrapping dataset in implementer mode (injecting plans) ==="
python3 "$SCRIPT_DIR/wrap_pipeline.py" \
    --source "$SOURCE_DATASET_ABS" \
    --mode implementer \
    --plans-dir "$PLANNER_JOB_INNER" \
    --output "$IMPL_DATASET" \
    "${filter_args[@]}"

echo "=== [4/5] harbor run: implementer (${IMPL_AGENT} / ${IMPL_MODEL}) ==="
harbor run \
    -p "$IMPL_DATASET" \
    -a "$IMPL_AGENT" \
    -m "$IMPL_MODEL" \
    -o "$IMPL_JOB" \
    -n "$N_PARALLEL" \
    "${claude_env_args[@]}" \
    "${impl_kwarg_args[@]}"

IMPL_JOB_INNER="$(find "$IMPL_JOB" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "$IMPL_JOB_INNER" ]]; then
    echo "ERROR: no implementer job dir produced under $IMPL_JOB" >&2
    exit 3
fi

echo "=== [5/5] scoring ==="
python3 "$SCRIPT_DIR/join_scores.py" \
    --planner-job "$PLANNER_JOB_INNER" \
    --impl-job "$IMPL_JOB_INNER" \
    --output "$RESULTS_DIR"

echo
echo "reward: $(cat "$RESULTS_DIR/reward.txt")"
echo "wrote:  $ABS_OUT"
