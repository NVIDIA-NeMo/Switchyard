#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run the full pipeline unattended with preflight, logging, and backgrounding.
#
# Usage:
#   # Single-machine run with glob:
#   scripts/run-pipeline.sh 'candidates/*.json'
#
#   # Multi-machine shard (explicit file list):
#   scripts/run-pipeline.sh 'candidates/fastapi.json candidates/scrapy.json ...'
#
#   # Resume a stopped run:
#   RESUME=harbor-tasks/craft-tools-v4/runs/2026-04-17/state.json scripts/run-pipeline.sh
#
# Env vars (optional):
#   RESUME=path/state.json  Resume from existing state file (skips candidate selection).
#   MAX_EVALUATE=2500     Global cap on candidates evaluated by Claude (default 2500).
#                         Sized for a 100+ repo bulk run at top 50/repo. The downstream
#                         cap that actually controls build volume is
#                         max_promising_per_repo in the profile.
#   TOP_PER_REPO=50       Per-file slice of mined candidates passed to evaluate.
#                         Default matches the miner's typical `--top 50`.
#   PROFILE=path.toml     Pipeline profile (default profiles/craft-tools-v4.toml)
#   SHARD_LABEL=east-1    Freeform label written into run_info for provenance
#
# Writes state to {task_suite_dir}/runs/<YYYY-MM-DD-HHMMSS>/state.json (see the
# chosen profile for task_suite_dir) and logs to runs/<ts>-<host>.log.
# Launches in the background via nohup.
#
# After launch, check progress with:
#   uv run craft-taskgen-status <state.json path>
#   tail -f runs/*.log

set -euo pipefail

RESUME="${RESUME:-}"
CANDIDATES="${1:-candidates/*.json}"
MAX_EVALUATE="${MAX_EVALUATE:-2500}"
TOP_PER_REPO="${TOP_PER_REPO:-50}"
PROFILE="${PROFILE:-profiles/craft-tools-v4.toml}"
SHARD_LABEL="${SHARD_LABEL:-}"

if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
    echo "ERROR: RESUME state file not found: $RESUME" >&2
    exit 1
fi

if [[ ! -f "$PROFILE" ]]; then
    echo "ERROR: profile not found: $PROFILE" >&2
    exit 1
fi

# Pre-flight checks — abort on any failure. `--check-endpoints` makes a
# few cents of live API calls to validate auth end-to-end before committing
# to hours of unattended work.
echo "Running pre-flight checks..."
PREFLIGHT_CANDIDATES_ARG=""
if [[ -z "$RESUME" ]]; then
    # shellcheck disable=SC2086
    PREFLIGHT_CANDIDATES_ARG="--candidates $CANDIDATES"
fi
# shellcheck disable=SC2086
if ! uv run craft-taskgen-preflight \
        $PREFLIGHT_CANDIDATES_ARG \
        --profile "$PROFILE" \
        --check-endpoints; then
    echo ""
    echo "ABORT: pre-flight failed. Fix issues above before launching." >&2
    exit 1
fi

TS=$(date +%Y%m%d-%H%M%S)
mkdir -p runs
HOST_SHORT=$(hostname -s 2>/dev/null || hostname)
LABEL_SUFFIX=""
if [[ -n "$SHARD_LABEL" ]]; then
    LABEL_SUFFIX="-$SHARD_LABEL"
fi
LOG="runs/$TS-$HOST_SHORT$LABEL_SUFFIX.log"

echo ""
echo "Launching pipeline run..."
echo "  host:           $HOST_SHORT"
[[ -n "$SHARD_LABEL" ]] && echo "  shard:          $SHARD_LABEL"
echo "  profile:        $PROFILE"
if [[ -n "$RESUME" ]]; then
    echo "  resume:         $RESUME"
else
    echo "  candidates:     $CANDIDATES"
    echo "  max_evaluate:   $MAX_EVALUATE"
    echo "  top_per_repo:   $TOP_PER_REPO"
fi
echo "  log:            $LOG"
echo ""

if [[ -n "$RESUME" ]]; then
    nohup uv run craft-taskgen \
        --profile "$PROFILE" \
        --resume "$RESUME" \
        --no-checkpoint \
        > "$LOG" 2>&1 &
else
    # shellcheck disable=SC2086
    nohup uv run craft-taskgen \
        --profile "$PROFILE" \
        --candidates $CANDIDATES \
        --max-evaluate "$MAX_EVALUATE" \
        --top-per-repo "$TOP_PER_REPO" \
        --no-checkpoint \
        > "$LOG" 2>&1 &
fi

PID=$!
echo "Started PID $PID. Log: $LOG"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG"
echo "  uv run craft-taskgen-status <state-file>  # state file path appears in log shortly"
echo ""
echo "To stop: kill $PID  (state.json is safe to resume from via --resume)"
