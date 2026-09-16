#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASKS_FILE="${SCRIPT_DIR}/tb21_hermes_demo_tasks.txt"
RUN_ROOT="${SWITCHYARD_HARBOR_RUN_ROOT:-${HOME}/switchyard-harbor-run}"

dry_run=0
passthrough=()
for argument in "$@"; do
    if [[ "${argument}" == "--dry-run" ]]; then
        dry_run=1
    else
        passthrough+=("${argument}")
    fi
done

export SWITCHYARD_DOCKER_NETWORK="${SWITCHYARD_DOCKER_NETWORK:-switchyard-native-tb21}"
export ALLOWED_HOSTS="${ALLOWED_HOSTS:-host.docker.internal}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://host.docker.internal:4000/v1}"
export SWITCHYARD_BASE_URL="${SWITCHYARD_BASE_URL:-http://host.docker.internal:4000}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-switchyard-local}"
export CLOSED_BOOK_MODE=1
export SWITCHYARD_VERIFIER_PROXY_TOKEN="${SWITCHYARD_VERIFIER_PROXY_TOKEN:-$(
    python3 -c 'import secrets; print(secrets.token_hex(24))'
)}"
export SWITCHYARD_VERIFIER_HTTP_PROXY="${SWITCHYARD_VERIFIER_HTTP_PROXY:-http://verifier:${SWITCHYARD_VERIFIER_PROXY_TOKEN}@proxy:3129}"

task_args=()
while IFS= read -r raw_task || [[ -n "${raw_task}" ]]; do
    task="${raw_task%%#*}"
    task="${task//[[:space:]]/}"
    [[ -n "${task}" ]] && task_args+=(--include-task-name "${task}")
done < "${TASKS_FILE}"

job_name="tb21-hermes-demo-$(date -u +%Y%m%dT%H%M%SZ)"
command=(uv run --project "${REPO_ROOT}" --no-sync harbor run \
    --agent hermes \
    --model openai/switchyard/vgr \
    --path benchmark/datasets/terminal-bench-2-1-closed-book \
    --jobs-dir benchmark/native-runs/tb21/jobs \
    --job-name "${job_name}" \
    --n-concurrent 1 \
    --max-retries 0 \
    --agent-timeout-multiplier 2.0 \
    --environment-build-timeout-multiplier 90.0 \
    --artifact /etc/proxy-ca/strip.jsonl \
    --ve "HTTP_PROXY=${SWITCHYARD_VERIFIER_HTTP_PROXY}" \
    --ve "HTTPS_PROXY=${SWITCHYARD_VERIFIER_HTTP_PROXY}" \
    --ve "http_proxy=${SWITCHYARD_VERIFIER_HTTP_PROXY}" \
    --ve "https_proxy=${SWITCHYARD_VERIFIER_HTTP_PROXY}" \
    --ve "NO_PROXY=localhost,127.0.0.1,proxy" \
    --ve "no_proxy=localhost,127.0.0.1,proxy" \
    "${task_args[@]}" \
    "${passthrough[@]}")

if [[ "${dry_run}" -eq 1 ]]; then
    printf 'WORKDIR: %s\nHARBOR_CMD: ' "${RUN_ROOT}"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

[[ -d "${REPO_ROOT}/benchmark/datasets/terminal-bench-2-1-closed-book" ]] || {
    echo "Missing prepared Terminal-Bench 2.1 dataset." >&2
    echo "Run benchmark/prepare_harbor_dataset.py first." >&2
    exit 1
}

mkdir -p "${RUN_ROOT}"
ln -sfn "${REPO_ROOT}/benchmark" "${RUN_ROOT}/benchmark"
cd "${RUN_ROOT}"
docker network inspect "${SWITCHYARD_DOCKER_NETWORK}" >/dev/null 2>&1 ||
    docker network create "${SWITCHYARD_DOCKER_NETWORK}" >/dev/null

exec "${command[@]}"
