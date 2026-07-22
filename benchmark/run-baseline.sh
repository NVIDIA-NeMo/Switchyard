#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified single-run Harbor baseline launcher for Switchyard evaluations.
#
# Writes run_manifest.json before Harbor starts, then runs the server + Harbor
# under a background nohup wrapper by default. The wrapper finalizes the
# manifest with Harbor exit status and artifact presence after the run ends.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_HELPER="${SCRIPT_DIR}/run_manifest.py"
PATCH_FILE="${SWITCHYARD_HARBOR_PATCH_FILE:-${SCRIPT_DIR}/patches/harbor-agent-patches.diff}"
AGENT_VERSIONS_FILE="${SCRIPT_DIR}/agent-versions.env"

if [[ -f "${AGENT_VERSIONS_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${AGENT_VERSIONS_FILE}"
fi

CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.211}"
CODEX_VERSION="${CODEX_VERSION:-0.144.5}"
OPENCODE_VERSION="${OPENCODE_VERSION:-1.18.3}"
NODE_VERSION="${NODE_VERSION:-20.11.1}"

DEFAULT_HARBOR_MODEL="openai/gpt-5.2"

SERVER_PRESET="serve"
MODE=""
OUTPUT_DIR="${REPO_ROOT}/benchmark/tb_runs"
PORT=4000
SERVER_URL=""
HARBOR_SERVER_URL=""
ROUTING_PROFILES=""
MODEL=""
ROUTE_MODEL=""
AGENT="terminus-2"
HARBOR_MODEL="${DEFAULT_HARBOR_MODEL}"
HARBOR_PATH=""
HARBOR_BIN="${HARBOR_BIN:-}"
HARBOR_PYTHON="${HARBOR_PYTHON:-}"
HARBOR_SOURCE=""
BOOK_MODE="closed"
PROXY_STRIP_ARTIFACT="/etc/proxy-public/strip.jsonl"
VERIFIER_PROXY_TEMPLATE='${SWITCHYARD_VERIFIER_HTTP_PROXY}'
SWITCHYARD_DOCKER_IMAGE="${SWITCHYARD_DOCKER_IMAGE:-switchyard-baseline:local}"
SWITCHYARD_DOCKERFILE="${SWITCHYARD_DOCKERFILE:-benchmark/switchyard-server.Dockerfile}"
SWITCHYARD_DOCKER_SERVICE_NAME="${SWITCHYARD_DOCKER_SERVICE_NAME:-switchyard}"
SWITCHYARD_DOCKER_BUILD="${SWITCHYARD_DOCKER_BUILD:-1}"
SWITCHYARD_DOCKER_NETWORK="${SWITCHYARD_DOCKER_NETWORK:-}"
SWITCHYARD_DOCKER_CONTAINER="${SWITCHYARD_DOCKER_CONTAINER:-}"
UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-}"
UPSTREAM_API_KEY_ENV="${UPSTREAM_API_KEY_ENV:-}"
REASONING_EFFORT=""
N_CONCURRENT=8
MAX_RETRIES=2
AGENT_TIMEOUT_MULTIPLIER="1.0"
N_TASKS=""
TASK_ID=""
TASK_LIST_FILE=""
DRY_RUN=0
FOREGROUND=0
SKIP_HEALTH_CHECK=0
HARBOR_MODEL_SET=0
REASONING_EFFORT_SET=0
SWITCHYARD_ENABLED=1

HARBOR_EXTRA=()
HARBOR_CMD_PREFIX=()
HARBOR_PYTHON_CMD=()
SERVER_EXTRA=()
SERVER_DOCKER_CMD=()
ORIGINAL_ARGV=("$@")
HARBOR_PATCH_JSON="{}"

usage() {
    cat <<'EOF'
Run one pinned Harbor baseline and write a per-run manifest.

Required:
  --harbor-path PATH           Local generated Harbor dataset path.
                               Use benchmark/prepare_harbor_dataset.py to build it.

Main options:
  --mode NAME                  Manifest server mode label (default: serve).
  --output-dir PATH            Root output dir (default: benchmark/tb_runs)
  --port N                     Server port (default: 4000)
  --server-url URL             Host URL used for health and stats checks.
  --harbor-server-url URL      Server URL as seen from Harbor task containers.
                               Defaults to the Dockerized Switchyard service.
  --routing-profiles PATH      Route bundle YAML for switchyard serve. When
                               omitted, Harbor connects directly upstream.
  --model MODEL                Model Harbor should request. With
                               --routing-profiles this is a Switchyard route
                               key; without it, this is the upstream model.
                               Defaults --harbor-model to nvidia/MODEL for
                               opencode, openai/MODEL for other OpenAI-style
                               agents, and MODEL for claude-code/codex.
  --route-model MODEL          Deprecated alias for --model in Switchyard mode.
  --agent NAME                 Harbor agent (default: terminus-2)
  --harbor-model MODEL         Explicit Harbor model label override.
  --upstream-base-url URL      Direct-upstream OpenAI-compatible base URL
                               (default: https://openrouter.ai/api/v1).
  --upstream-api-key-env NAME  Env var containing the direct upstream API key
                               (default: UPSTREAM_API_KEY for a custom
                               upstream URL, otherwise OPENROUTER_API_KEY).
  --harbor-path PATH           Local Harbor dataset path; runner passes --path.
  --book-mode MODE             closed or open (default: closed). Both modes use
                               the generated dataset proxy topology; open mode
                               allows broad egress through the proxy.
  --reasoning-effort VALUE     Forwarded as --ak reasoning_effort=VALUE.
                               Defaults by agent/model; pass an empty value to omit.
  --harbor-bin PATH            Optional Harbor executable override
                               (default: uv run --no-sync harbor)
  --harbor-python PATH         Optional Python that can import the active Harbor
                               (default: uv run --no-sync python)
  --n-concurrent N             Harbor concurrency (default: 8)
  --max-retries N              Harbor max retries (default: 2)
  --agent-timeout-multiplier N Harbor agent timeout multiplier (default: 1.0)
  --n-tasks N                  Optional Harbor task cap
  --task-id NAME               Optional single task filter
  --task-list-file PATH        File of task names, one per line, # comments allowed
  --harbor-extra ARG           Repeatable raw Harbor arg token
  --server-extra ARG           Repeatable raw server-launch arg token
  SWITCHYARD_CLOSED_BOOK_PREFLIGHT=0 disables the benchmark Docker
                               reachability preflight in the background wrapper.
  SWITCHYARD_DOCKER_IMAGE      Closed-book Switchyard server image
                               (default: switchyard-baseline:local).
  SWITCHYARD_DOCKER_BUILD=0    Reuse SWITCHYARD_DOCKER_IMAGE instead of building it.
  SWITCHYARD_DOCKER_NETWORK    Optional existing Docker network for closed-book runs.
  --dry-run                    Print resolved commands and exit

Execution:
  --foreground                 Run wrapper in the foreground instead of nohup
  --skip-health-check          Do not wait for /health before Harbor
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

q() {
    printf "%q" "$1"
}

shell_join() {
    printf "%q " "$@"
}

json_array() {
    python3 - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
}

json_object_from_pairs() {
    python3 - "$@" <<'PY'
import json
import sys

items = sys.argv[1:]
out = {}
for idx in range(0, len(items), 2):
    key = items[idx]
    value = items[idx + 1] if idx + 1 < len(items) else ""
    if value != "":
        out[key] = value
print(json.dumps(out, sort_keys=True))
PY
}

json_merge_object_pairs() {
    python3 - "$@" <<'PY'
import json
import sys

out = json.loads(sys.argv[1] or "{}")
items = sys.argv[2:]
for idx in range(0, len(items), 2):
    key = items[idx]
    value = items[idx + 1] if idx + 1 < len(items) else ""
    if value != "":
        out[key] = value
print(json.dumps(out, sort_keys=True))
PY
}

lower_ascii() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

default_reasoning_effort_for() {
    local agent model
    agent="$(lower_ascii "$1")"
    model="$(lower_ascii "$2")"
    case "${agent}:${model}" in
        claude-code:*opus*|claude-code:*sonnet*|claude-code:*haiku*) echo "high" ;;
        codex:*gpt-5*|codex:*o3*|codex:*o4*) echo "high" ;;
        *) echo "" ;;
    esac
}

csv_append_unique() {
    python3 - "$@" <<'PY'
import sys

items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
for item in sys.argv[2:]:
    if item not in items:
        items.append(item)
print(",".join(items))
PY
}

merge_claude_closed_book_tools() {
    local i=0 token next value found=0
    local merged=()
    while [[ "${i}" -lt "${#HARBOR_EXTRA[@]}" ]]; do
        token="${HARBOR_EXTRA[$i]}"
        if [[ "${token}" == "--ak" || "${token}" == "--agent-kwarg" ]]; then
            next="${HARBOR_EXTRA[$((i + 1))]:-}"
            if [[ "${next}" == disallowed_tools=* ]]; then
                value="${next#disallowed_tools=}"
                merged+=("${token}" "disallowed_tools=$(csv_append_unique "${value}" WebFetch WebSearch)")
                found=1
                i=$((i + 2))
                continue
            fi
        fi
        merged+=("${token}")
        i=$((i + 1))
    done
    HARBOR_EXTRA=()
    if [[ "${#merged[@]}" -gt 0 ]]; then
        HARBOR_EXTRA=("${merged[@]}")
    fi
    if [[ "${found}" -eq 0 ]]; then
        HARBOR_CMD+=(--ak "disallowed_tools=WebFetch,WebSearch")
    fi
}

harbor_extra_has_agent_version() {
    local i=0 token next
    while [[ "${i}" -lt "${#HARBOR_EXTRA[@]}" ]]; do
        token="${HARBOR_EXTRA[$i]}"
        if [[ "${token}" == "--ak" || "${token}" == "--agent-kwarg" ]]; then
            next="${HARBOR_EXTRA[$((i + 1))]:-}"
            [[ "${next}" == version=* ]] && return 0
            i=$((i + 2))
            continue
        fi
        i=$((i + 1))
    done
    return 1
}

harbor_extra_has_agent_env() {
    local key="$1" i=0 token next
    while [[ "${i}" -lt "${#HARBOR_EXTRA[@]}" ]]; do
        token="${HARBOR_EXTRA[$i]}"
        if [[ "${token}" == "--ae" || "${token}" == "--agent-env" ]]; then
            next="${HARBOR_EXTRA[$((i + 1))]:-}"
            [[ "${next}" == "${key}="* ]] && return 0
            i=$((i + 2))
            continue
        fi
        i=$((i + 1))
    done
    return 1
}

add_agent_version_kwarg() {
    harbor_extra_has_agent_version && return 0
    case "${AGENT}" in
        claude-code) HARBOR_CMD+=(--ak "version=${CLAUDE_CODE_VERSION}") ;;
        codex) HARBOR_CMD+=(--ak "version=${CODEX_VERSION}") ;;
        opencode) HARBOR_CMD+=(--ak "version=${OPENCODE_VERSION}") ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --server-url) SERVER_URL="$2"; shift 2 ;;
        --harbor-server-url) HARBOR_SERVER_URL="$2"; shift 2 ;;
        --routing-profiles) ROUTING_PROFILES="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --route-model) ROUTE_MODEL="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --harbor-model) HARBOR_MODEL="$2"; HARBOR_MODEL_SET=1; shift 2 ;;
        --upstream-base-url) UPSTREAM_BASE_URL="$2"; shift 2 ;;
        --upstream-api-key-env) UPSTREAM_API_KEY_ENV="$2"; shift 2 ;;
        --harbor-path) HARBOR_PATH="$2"; shift 2 ;;
        --book-mode) BOOK_MODE="$2"; shift 2 ;;
        --harbor-bin) HARBOR_BIN="$2"; shift 2 ;;
        --harbor-python) HARBOR_PYTHON="$2"; shift 2 ;;
        --reasoning-effort) REASONING_EFFORT="$2"; REASONING_EFFORT_SET=1; shift 2 ;;
        --n-concurrent) N_CONCURRENT="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --agent-timeout-multiplier) AGENT_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --n-tasks) N_TASKS="$2"; shift 2 ;;
        --task-id) TASK_ID="$2"; shift 2 ;;
        --task-list-file) TASK_LIST_FILE="$2"; shift 2 ;;
        --harbor-extra) HARBOR_EXTRA+=("$2"); shift 2 ;;
        --server-extra) SERVER_EXTRA+=("$2"); shift 2 ;;
        --strong-model|--weak-model|--classifier-model|--executor-model|--planner-model|--profile|--strong-probability|--planner-cadence-n)
            die "$1 belongs to the removed legacy benchmark launchers; encode server routing in --routing-profiles instead"
            ;;
        --dry-run) DRY_RUN=1; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        --skip-health-check) SKIP_HEALTH_CHECK=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

cd "${REPO_ROOT}"

resolve_harbor() {
    if [[ -n "${HARBOR_BIN}" ]]; then
        [[ -x "${HARBOR_BIN}" ]] || die "--harbor-bin is not executable: ${HARBOR_BIN}"
        HARBOR_BIN="$(cd "$(dirname "${HARBOR_BIN}")" && pwd)/$(basename "${HARBOR_BIN}")"
        HARBOR_SOURCE="override"
        HARBOR_CMD_PREFIX=("${HARBOR_BIN}")
    else
        HARBOR_SOURCE="uv-dev"
        HARBOR_CMD_PREFIX=(uv run --no-sync harbor)
    fi

    if [[ -n "${HARBOR_PYTHON}" ]]; then
        [[ -x "${HARBOR_PYTHON}" ]] || die "--harbor-python is not executable: ${HARBOR_PYTHON}"
        HARBOR_PYTHON="$(cd "$(dirname "${HARBOR_PYTHON}")" && pwd)/$(basename "${HARBOR_PYTHON}")"
        HARBOR_PYTHON_CMD=("${HARBOR_PYTHON}")
    elif [[ "${HARBOR_SOURCE}" == "uv-dev" ]]; then
        HARBOR_PYTHON_CMD=(uv run --no-sync python)
    else
        HARBOR_PYTHON_CMD=(python3)
    fi
}

resolve_harbor

if [[ -n "${ROUTING_PROFILES}" ]]; then
    [[ -f "${ROUTING_PROFILES}" ]] || die "--routing-profiles not found: ${ROUTING_PROFILES}"
    ROUTING_PROFILES="$(cd "$(dirname "${ROUTING_PROFILES}")" && pwd)/$(basename "${ROUTING_PROFILES}")"
fi
UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-https://openrouter.ai/api/v1}"
if [[ -z "${UPSTREAM_API_KEY_ENV}" ]]; then
    if [[ -n "${UPSTREAM_API_KEY:-}" && "${UPSTREAM_BASE_URL}" != "https://openrouter.ai/api/v1" ]]; then
        UPSTREAM_API_KEY_ENV="UPSTREAM_API_KEY" # pragma: allowlist secret
    else
        UPSTREAM_API_KEY_ENV="OPENROUTER_API_KEY" # pragma: allowlist secret
    fi
fi
[[ "${UPSTREAM_API_KEY_ENV}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "--upstream-api-key-env must be a valid environment variable name"

if [[ -n "${ROUTING_PROFILES}" ]]; then
    if [[ -n "${MODEL}" && -n "${ROUTE_MODEL}" && "${MODEL}" != "${ROUTE_MODEL}" ]]; then
        die "--model and --route-model must name the same model when both are provided"
    fi
    if [[ -z "${MODEL}" && -n "${ROUTE_MODEL}" ]]; then
        MODEL="${ROUTE_MODEL}"
    fi
    SWITCHYARD_ENABLED=1
    SERVER_PRESET="serve"
    [[ -z "${MODE}" ]] && MODE="serve"
    [[ -n "${MODEL}" ]] || die "--model is required when using --routing-profiles"
    ROUTE_MODEL="${MODEL}"
else
    SWITCHYARD_ENABLED=0
    SERVER_PRESET="direct"
    [[ -z "${MODE}" ]] && MODE="direct"
    [[ -z "${ROUTE_MODEL}" ]] || die "--route-model requires --routing-profiles; use --model for direct upstream"
    [[ -n "${MODEL}" ]] || die "--model is required when running direct upstream"
    ROUTE_MODEL=""
    [[ -z "${SERVER_URL}" ]] || die "--server-url requires --routing-profiles"
    [[ -z "${HARBOR_SERVER_URL}" ]] || die "--harbor-server-url requires --routing-profiles"
    [[ "${#SERVER_EXTRA[@]}" -eq 0 ]] || die "--server-extra requires --routing-profiles"
    [[ -n "${!UPSTREAM_API_KEY_ENV:-}" ]] || die "direct upstream requires \$${UPSTREAM_API_KEY_ENV} to be set"
fi
if [[ "${HARBOR_MODEL_SET}" -eq 0 ]]; then
    if [[ "${SWITCHYARD_ENABLED}" -eq 1 ]]; then
        case "${AGENT}" in
            claude-code|codex) HARBOR_MODEL="${MODEL}" ;;
            opencode) HARBOR_MODEL="nvidia/${MODEL}" ;;
            *) HARBOR_MODEL="openai/${MODEL}" ;;
        esac
    else
        HARBOR_MODEL="${MODEL}"
    fi
fi

check_harbor_available() {
    if ! "${HARBOR_CMD_PREFIX[@]}" --version >/dev/null 2>&1; then
        echo "ERROR: Harbor is not available via:" >&2
        printf '  ' >&2
        shell_join "${HARBOR_CMD_PREFIX[@]}" >&2
        echo >&2
        echo "Run 'uv sync' first, or pass --harbor-bin." >&2
        exit 1
    fi
}

if [[ "${SWITCHYARD_ENABLED}" -eq 1 ]]; then
    SERVER_CMD=(uv run --no-sync switchyard
        --routing-profiles "${ROUTING_PROFILES}"
        serve
        --host 0.0.0.0
        --port "${PORT}")
    SERVER_DOCKER_CMD=(--routing-profiles "${ROUTING_PROFILES}"
        serve
        --host 0.0.0.0
        --port "${PORT}")
    SERVER_CONFIG_JSON="$(json_object_from_pairs \
        mode "${MODE}" routing_profiles "${ROUTING_PROFILES}" route_model "${ROUTE_MODEL}")"
    if [[ "${#SERVER_EXTRA[@]}" -gt 0 ]]; then
        SERVER_CMD+=("${SERVER_EXTRA[@]}")
        SERVER_DOCKER_CMD+=("${SERVER_EXTRA[@]}")
    fi
else
    UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL%/}"
    SERVER_CONFIG_JSON="$(json_object_from_pairs \
        mode "${MODE}" upstream_base_url "${UPSTREAM_BASE_URL}" \
        upstream_api_key_env "${UPSTREAM_API_KEY_ENV}" upstream_model "${MODEL}")"
fi

if [[ -n "${HARBOR_PATH}" ]]; then
    [[ -d "${HARBOR_PATH}" ]] || die "--harbor-path must be a directory: ${HARBOR_PATH}"
    HARBOR_PATH="$(cd "${HARBOR_PATH}" && pwd)"
fi
if [[ -z "${HARBOR_PATH}" ]]; then
    die "--harbor-path is required. Prepare one with: uv run --no-sync python benchmark/prepare_harbor_dataset.py"
fi

case "${BOOK_MODE}" in
    closed|open) ;;
    *) die "--book-mode must be closed or open" ;;
esac
[[ -f "${HARBOR_PATH}/switchyard_dataset_manifest.json" ]] || die "--harbor-path must contain switchyard_dataset_manifest.json: ${HARBOR_PATH}"

DATASET="$(basename "${HARBOR_PATH}")"

if [[ "${REASONING_EFFORT_SET}" -eq 0 ]]; then
    REASONING_EFFORT="$(default_reasoning_effort_for "${AGENT}" "${HARBOR_MODEL}")"
fi

if [[ -n "${TASK_LIST_FILE}" ]]; then
    [[ -f "${TASK_LIST_FILE}" ]] || die "--task-list-file not found: ${TASK_LIST_FILE}"
    TASK_LIST_FILE="$(cd "$(dirname "${TASK_LIST_FILE}")" && pwd)/$(basename "${TASK_LIST_FILE}")"
fi

if [[ "${SWITCHYARD_ENABLED}" -eq 1 ]]; then
    SERVER_CONTROL_URL="${SERVER_URL:-http://127.0.0.1:${PORT}}"
    SERVER_CONTROL_URL="${SERVER_CONTROL_URL%/}"
    if [[ "${SERVER_CONTROL_URL}" == */v1 ]]; then
        SERVER_ROOT_URL="${SERVER_CONTROL_URL%/v1}"
    else
        SERVER_ROOT_URL="${SERVER_CONTROL_URL}"
    fi
    SERVER_HEALTH_URL="${SERVER_ROOT_URL}/health"
    SERVER_STATS_URL="${SERVER_ROOT_URL}/v1/routing/stats"

    if [[ -z "${HARBOR_SERVER_URL}" ]]; then
        HARBOR_SERVER_URL="http://${SWITCHYARD_DOCKER_SERVICE_NAME}:${PORT}"
    fi
    HARBOR_SERVER_URL="${HARBOR_SERVER_URL%/}"
    if [[ "${HARBOR_SERVER_URL}" == */v1 ]]; then
        HARBOR_SERVER_ROOT_URL="${HARBOR_SERVER_URL%/v1}"
        HARBOR_BASE_URL="${HARBOR_SERVER_URL}"
    else
        HARBOR_SERVER_ROOT_URL="${HARBOR_SERVER_URL}"
        HARBOR_BASE_URL="${HARBOR_SERVER_ROOT_URL}/v1"
    fi
else
    SERVER_ROOT_URL=""
    SERVER_HEALTH_URL=""
    SERVER_STATS_URL=""
    HARBOR_BASE_URL="${UPSTREAM_BASE_URL}"
    if [[ "${HARBOR_BASE_URL}" == */v1 ]]; then
        HARBOR_SERVER_ROOT_URL="${HARBOR_BASE_URL%/v1}"
    else
        HARBOR_SERVER_ROOT_URL="${HARBOR_BASE_URL}"
    fi
fi

TS="$(date -u +%Y-%m-%d_%H-%M-%S)"
JOB_NAME="baseline-${SERVER_PRESET}-${MODE}-${TS}"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
RUN_DIR="${OUTPUT_DIR}/${JOB_NAME}"
MANIFEST_PATH="${RUN_DIR}/run_manifest.json"
LOG_PATH="${RUN_DIR}/${JOB_NAME}.log"
SERVER_LOG="${RUN_DIR}/server.log"
HARBOR_LOG="${RUN_DIR}/harbor.log"
HARBOR_RESULT_JSON="${RUN_DIR}/harbor_result.json"
ROUTING_STATS_JSON="${RUN_DIR}/routing_stats_final.json"
ROUTING_LOG_JSONL="${RUN_DIR}/routing_requests.jsonl"
ROUTING_STATS_BY_TASK_JSON="${RUN_DIR}/routing_stats_by_task.json"
if [[ "${SWITCHYARD_ENABLED}" -eq 1 ]]; then
    SERVER_CMD+=(--routing-log-file "${ROUTING_LOG_JSONL}")
    SERVER_DOCKER_CMD+=(--routing-log-file "${ROUTING_LOG_JSONL}")
fi
DOCKER_RUN_ID="$(printf '%s-%s' "${TS##*_}" "$$" | tr -c '[:alnum:]_.-' '-')"
SWITCHYARD_DOCKER_NETWORK="${SWITCHYARD_DOCKER_NETWORK:-switchyard-${DOCKER_RUN_ID}}"
SWITCHYARD_DOCKER_CONTAINER="${SWITCHYARD_DOCKER_CONTAINER:-switchyard-${DOCKER_RUN_ID}}"
CODEX_MODEL_CATALOG_HOST=""
if [[ "${AGENT}" == "codex" ]]; then
    CODEX_MODEL_CATALOG_HOST="${RUN_DIR}/codex_model_catalog.json"
fi
if [[ "${BOOK_MODE}" == "closed" || "${BOOK_MODE}" == "open" ]]; then
    if [[ "${SWITCHYARD_ENABLED}" -eq 1 ]]; then
        SERVER_CONFIG_JSON="$(json_merge_object_pairs "${SERVER_CONFIG_JSON}" \
            docker_image "${SWITCHYARD_DOCKER_IMAGE}" \
            docker_network "${SWITCHYARD_DOCKER_NETWORK}" \
            docker_container "${SWITCHYARD_DOCKER_CONTAINER}" \
            docker_service "${SWITCHYARD_DOCKER_SERVICE_NAME}")"
    else
        SERVER_CONFIG_JSON="$(json_merge_object_pairs "${SERVER_CONFIG_JSON}" \
            docker_network "${SWITCHYARD_DOCKER_NETWORK}")"
    fi
fi
ROUTING_PROFILES_DIR=""
if [[ -n "${ROUTING_PROFILES}" ]]; then
    ROUTING_PROFILES_DIR="$(cd "$(dirname "${ROUTING_PROFILES}")" && pwd)"
fi
CLASSIFIER_PROMPTS_JSON="{}"
if [[ -n "${ROUTING_PROFILES}" ]]; then
    CLASSIFIER_PROMPTS_JSON="$(
        uv run --no-sync python "${SCRIPT_DIR}/resolve_classifier_prompts.py" "${ROUTING_PROFILES}" \
            2>/dev/null || echo '{}'
    )"
    [[ -n "${CLASSIFIER_PROMPTS_JSON}" ]] || CLASSIFIER_PROMPTS_JSON="{}"
fi
WRAPPER="${RUN_DIR}/run-background.sh"
HARBOR_JOB_DIR="${RUN_DIR}/jobs/${JOB_NAME}"

HARBOR_CMD=("${HARBOR_CMD_PREFIX[@]}" run
    --agent "${AGENT}"
    --model "${HARBOR_MODEL}"
    --jobs-dir "${RUN_DIR}/jobs"
    --job-name "${JOB_NAME}"
    -n "${N_CONCURRENT}"
    --max-retries "${MAX_RETRIES}"
    --agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER}")

HARBOR_CMD+=(--path "${HARBOR_PATH}")

if [[ -n "${CODEX_MODEL_CATALOG_HOST}" ]]; then
    if ! harbor_extra_has_agent_env CODEX_MODEL_CATALOG_JSON; then
        HARBOR_CMD+=(--ae "CODEX_MODEL_CATALOG_JSON=${CODEX_MODEL_CATALOG_HOST}")
    fi
fi

HARBOR_CMD+=(--artifact "${PROXY_STRIP_ARTIFACT}")
if [[ "${BOOK_MODE}" == "closed" ]]; then
    HARBOR_CMD+=(--ve "HTTP_PROXY=${VERIFIER_PROXY_TEMPLATE}")
    HARBOR_CMD+=(--ve "HTTPS_PROXY=${VERIFIER_PROXY_TEMPLATE}")
    HARBOR_CMD+=(--ve "http_proxy=${VERIFIER_PROXY_TEMPLATE}")
    HARBOR_CMD+=(--ve "https_proxy=${VERIFIER_PROXY_TEMPLATE}")
    HARBOR_CMD+=(--ve "NO_PROXY=localhost,127.0.0.1,proxy")
    HARBOR_CMD+=(--ve "no_proxy=localhost,127.0.0.1,proxy")
    case "${AGENT}" in
        codex) HARBOR_CMD+=(--ae "CODEX_DISABLE_WEB_SEARCH=1") ;;
        claude-code) merge_claude_closed_book_tools ;;
        opencode) HARBOR_CMD+=(--ae "OPENCODE_DISABLE_WEBFETCH=1") ;;
    esac
fi

add_agent_version_kwarg

if [[ -n "${REASONING_EFFORT}" ]]; then
    HARBOR_CMD+=(--ak "reasoning_effort=${REASONING_EFFORT}")
fi

if [[ -n "${N_TASKS}" ]]; then
    HARBOR_CMD+=(--n-tasks "${N_TASKS}")
fi
if [[ -n "${TASK_ID}" ]]; then
    HARBOR_CMD+=(--include-task-name "${TASK_ID}")
fi
if [[ -n "${TASK_LIST_FILE}" ]]; then
    while IFS= read -r raw_task || [[ -n "${raw_task}" ]]; do
        task="${raw_task%%#*}"
        task="${task//[[:space:]]/}"
        [[ -n "${task}" ]] && HARBOR_CMD+=(--include-task-name "${task}")
    done < "${TASK_LIST_FILE}"
fi
if [[ "${#HARBOR_EXTRA[@]}" -gt 0 ]]; then
    HARBOR_CMD+=("${HARBOR_EXTRA[@]}")
fi

check_harbor_patch() {
    local py_display site patch_root verify_method verify_status verify_error
    py_display="$(shell_join "${HARBOR_PYTHON_CMD[@]}")"
    site="$(
        "${HARBOR_PYTHON_CMD[@]}" - <<'PY' 2>/dev/null || true
from pathlib import Path
import harbor

print(Path(harbor.__file__).resolve().parent)
PY
    )"
    if [[ -z "${site}" ]]; then
        echo "ERROR: Harbor is not importable via ${py_display}; cannot verify benchmark patches." >&2
        echo "       Run 'uv sync' first, or pass --harbor-python for the Harbor install used by --harbor-bin." >&2
        exit 1
    fi

    [[ -f "${PATCH_FILE}" ]] || die "Harbor patch file not found: ${PATCH_FILE}"

    patch_root="$(dirname "${site}")"
    verify_status="missing"
    verify_error=""
    if command -v git >/dev/null 2>&1; then
        verify_method="git-apply-reverse-check"
        if verify_error="$(cd "${patch_root}" && git apply --reverse --check -p1 "${PATCH_FILE}" 2>&1)"; then
            verify_status="applied"
        fi
    else
        verify_method="patch-reverse-dry-run"
        if verify_error="$(cd "${patch_root}" && patch --dry-run --reverse --batch --fuzz=0 -p1 < "${PATCH_FILE}" 2>&1)"; then
            verify_status="applied"
        fi
    fi

    HARBOR_PATCH_JSON="$(
        python3 - "${PATCH_FILE}" "${site}" "${patch_root}" "${verify_method}" "${verify_status}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

patch_file = Path(sys.argv[1])
site = Path(sys.argv[2])
patch_root = Path(sys.argv[3])
verify_method = sys.argv[4]
verify_status = sys.argv[5]
try:
    with patch_file.open("rb") as fh:
        digest = "sha256:" + hashlib.file_digest(fh, "sha256").hexdigest()
except OSError:
    digest = "sha256:missing"
print(json.dumps({
    "required": True,
    "status": verify_status,
    "verification": verify_method,
    "harbor_site": str(site),
    "patch_root": str(patch_root),
    "patch_file": str(patch_file.resolve()),
    "patch_file_digest": digest,
}, sort_keys=True))
PY
    )"

    if [[ "${verify_status}" != "applied" ]]; then
        echo "ERROR: The current Harbor patch is not applied cleanly in ${site}." >&2
        echo "       run-baseline.sh requires patched Harbor for reproducible evals." >&2
        echo "       Verification used: ${verify_method}" >&2
        if [[ -n "${verify_error}" ]]; then
            echo "       Patch verification output:" >&2
            echo "${verify_error}" | sed 's/^/         /' >&2
        fi
        echo "       Reinstall/recreate Harbor, then apply the current patch before launching a baseline:" >&2
        echo "         cd $(q "$(dirname "${site}")") && patch -p1 < $(q "${PATCH_FILE}")" >&2
        exit 1
    fi
}

print_resolved() {
    echo "Resolved baseline"
    echo "  run_dir:       ${RUN_DIR}"
    echo "  manifest:      ${MANIFEST_PATH}"
    echo "  server_preset: ${SERVER_PRESET}"
    echo "  server_mode:   ${MODE}"
    echo "  book_mode:     ${BOOK_MODE}"
    echo "  harbor_path:   ${HARBOR_PATH}"
    if [[ "${SWITCHYARD_ENABLED}" -eq 1 ]]; then
        echo "  server_url:    ${SERVER_ROOT_URL}"
    else
        echo "  upstream_url:  ${HARBOR_BASE_URL}"
        echo "  api_key_env:   ${UPSTREAM_API_KEY_ENV}"
    fi
    echo "  harbor_server: ${HARBOR_SERVER_ROOT_URL}"
    echo "  harbor_url:    ${HARBOR_BASE_URL}"
    if [[ "${BOOK_MODE}" == "closed" || "${BOOK_MODE}" == "open" ]]; then
        if [[ "${SWITCHYARD_ENABLED}" -eq 1 ]]; then
            echo "  docker_image:  ${SWITCHYARD_DOCKER_IMAGE}"
        fi
        echo "  docker_network: ${SWITCHYARD_DOCKER_NETWORK}"
        echo "  proxy_mode:    ${BOOK_MODE}"
    fi
    if [[ "${BOOK_MODE}" == "closed" ]]; then
        echo "  verifier_net: authenticated proxy egress"
    fi
    if [[ -n "${ROUTING_PROFILES}" ]]; then
        echo "  route_profile: ${ROUTING_PROFILES}"
        echo "  route_model:   ${ROUTE_MODEL}"
    fi
    printf '  harbor_cmd:    '
    shell_join "${HARBOR_CMD_PREFIX[@]}"
    echo "(${HARBOR_SOURCE})"
    printf '  harbor_python: '
    shell_join "${HARBOR_PYTHON_CMD[@]}"
    echo
    echo "  harbor_patch:  verified"
    echo "  agent/model:   ${AGENT} / ${HARBOR_MODEL}"
    echo "  reasoning:     ${REASONING_EFFORT:-unset}"
    if [[ -n "${CODEX_MODEL_CATALOG_HOST}" ]]; then
        echo "  codex_catalog: ${CODEX_MODEL_CATALOG_HOST}"
    fi
    echo "  concurrency:   ${N_CONCURRENT}"
    echo
    if [[ "${SWITCHYARD_ENABLED}" -eq 1 ]]; then
        printf 'SERVER_CMD: '
        shell_join "${SERVER_CMD[@]}"
        echo
    else
        echo "SERVER_CMD: <direct-upstream>"
    fi
    printf 'HARBOR_CMD: '
    shell_join "${HARBOR_CMD[@]}"
    echo
}

check_harbor_available
check_harbor_patch
print_resolved

if [[ "${DRY_RUN}" -eq 1 ]]; then
    exit 0
fi

mkdir -p "${RUN_DIR}"

if [[ -n "${CODEX_MODEL_CATALOG_HOST}" ]]; then
    uv run --no-sync python benchmark/codex_model_catalog.py \
        --output "${CODEX_MODEL_CATALOG_HOST}" \
        --model "${HARBOR_MODEL}"
fi

AGENT_VERSIONS_JSON="$(json_object_from_pairs \
    claude_code "${CLAUDE_CODE_VERSION}" codex "${CODEX_VERSION}" \
    opencode "${OPENCODE_VERSION}" node "${NODE_VERSION}")"
HARBOR_EXTRA_JSON="$(json_array)"
if [[ "${#HARBOR_EXTRA[@]}" -gt 0 ]]; then
    HARBOR_EXTRA_JSON="$(json_array "${HARBOR_EXTRA[@]}")"
fi

MANIFEST_CMD=(python3 "${MANIFEST_HELPER}" write
    --output "${MANIFEST_PATH}"
    --launcher-argv-json "$(json_array "${BASH_SOURCE[0]}" "${ORIGINAL_ARGV[@]}")"
    --server-preset "${SERVER_PRESET}"
    --server-mode "${MODE}"
    --server-url "${SERVER_ROOT_URL}"
    --server-port "${PORT}"
    --server-argv-json "$(json_array "${SERVER_CMD[@]}")"
    --server-config-json "${SERVER_CONFIG_JSON}"
    --classifier-prompts-json "${CLASSIFIER_PROMPTS_JSON}"
    --harbor-server-url "${HARBOR_SERVER_ROOT_URL}"
    --harbor-base-url "${HARBOR_BASE_URL}"
    --upstream-base-url "$([[ "${SWITCHYARD_ENABLED}" -eq 0 ]] && echo "${UPSTREAM_BASE_URL}")"
    --upstream-api-key-env "$([[ "${SWITCHYARD_ENABLED}" -eq 0 ]] && echo "${UPSTREAM_API_KEY_ENV}")"
    --route-model "${ROUTE_MODEL}"
    --harbor-command-json "$(json_array "${HARBOR_CMD_PREFIX[@]}")"
    --dataset-label "${DATASET}"
    --agent "${AGENT}"
    --harbor-model "${HARBOR_MODEL}"
    --reasoning-effort "${REASONING_EFFORT}"
    --n-concurrent "${N_CONCURRENT}"
    --max-retries "${MAX_RETRIES}"
    --agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER}"
    --n-tasks "${N_TASKS:--1}"
    --task-id "${TASK_ID}"
    --harbor-extra-json "${HARBOR_EXTRA_JSON}"
    --closed-book-mode "${BOOK_MODE}"
    --closed-book-gateway-enforced "1"
    --closed-book-hosted-tools-disabled "$([[ "${BOOK_MODE}" == "closed" ]] && echo 1 || echo 0)"
    --closed-book-proxy-strip-artifact "${PROXY_STRIP_ARTIFACT}"
    --harbor-patch-json "${HARBOR_PATCH_JSON}"
    --agent-versions-json "${AGENT_VERSIONS_JSON}"
    --run-dir "${RUN_DIR}"
    --log-path "${LOG_PATH}"
    --harbor-result-json "${HARBOR_RESULT_JSON}"
    --routing-stats-json "${ROUTING_STATS_JSON}"
    --routing-stats-status "$([[ "${SWITCHYARD_ENABLED}" -eq 1 ]] && echo predicted || echo not-requested)")
if [[ -n "${HARBOR_PATH}" ]]; then
    MANIFEST_CMD+=(--harbor-path "${HARBOR_PATH}")
fi
if [[ -n "${TASK_LIST_FILE}" ]]; then
    MANIFEST_CMD+=(--task-list-file "${TASK_LIST_FILE}")
fi
if [[ -n "${ROUTING_PROFILES}" ]]; then
    MANIFEST_CMD+=(--routing-profiles "${ROUTING_PROFILES}")
fi
if [[ -n "${CODEX_MODEL_CATALOG_HOST}" ]]; then
    MANIFEST_CMD+=(--codex-model-catalog "${CODEX_MODEL_CATALOG_HOST}")
fi
"${MANIFEST_CMD[@]}"

cat > "${WRAPPER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd $(q "${REPO_ROOT}")

SERVER_PRESET=$(q "${SERVER_PRESET}")
SERVER_ENABLED=$(q "${SWITCHYARD_ENABLED}")
AGENT=$(q "${AGENT}")
HARBOR_MODEL=$(q "${HARBOR_MODEL}")
UPSTREAM_API_KEY_ENV=$(q "${UPSTREAM_API_KEY_ENV}")
SERVER_ROOT_URL=$(q "${SERVER_ROOT_URL}")
HARBOR_SERVER_ROOT_URL=$(q "${HARBOR_SERVER_ROOT_URL}")
SERVER_HEALTH_URL=$(q "${SERVER_HEALTH_URL}")
SERVER_STATS_URL=$(q "${SERVER_STATS_URL}")
HARBOR_BASE_URL=$(q "${HARBOR_BASE_URL}")
MANIFEST_PATH=$(q "${MANIFEST_PATH}")
RUN_DIR=$(q "${RUN_DIR}")
REPO_ROOT=$(q "${REPO_ROOT}")
SERVER_LOG=$(q "${SERVER_LOG}")
HARBOR_LOG=$(q "${HARBOR_LOG}")
ROUTING_STATS_JSON=$(q "${ROUTING_STATS_JSON}")
ROUTING_LOG_JSONL=$(q "${ROUTING_LOG_JSONL}")
ROUTING_STATS_BY_TASK_JSON=$(q "${ROUTING_STATS_BY_TASK_JSON}")
HARBOR_JOB_DIR=$(q "${HARBOR_JOB_DIR}")
SKIP_HEALTH_CHECK=$(q "${SKIP_HEALTH_CHECK}")
BOOK_MODE=$(q "${BOOK_MODE}")
SWITCHYARD_DOCKER_IMAGE=$(q "${SWITCHYARD_DOCKER_IMAGE}")
SWITCHYARD_DOCKERFILE=$(q "${SWITCHYARD_DOCKERFILE}")
SWITCHYARD_DOCKER_BUILD=$(q "${SWITCHYARD_DOCKER_BUILD}")
SWITCHYARD_DOCKER_NETWORK=$(q "${SWITCHYARD_DOCKER_NETWORK}")
SWITCHYARD_DOCKER_CONTAINER=$(q "${SWITCHYARD_DOCKER_CONTAINER}")
SWITCHYARD_DOCKER_SERVICE_NAME=$(q "${SWITCHYARD_DOCKER_SERVICE_NAME}")
ROUTING_PROFILES_DIR=$(q "${ROUTING_PROFILES_DIR}")

SERVER_CMD=($(shell_join "${SERVER_CMD[@]}"))
SERVER_DOCKER_CMD=($(shell_join "${SERVER_DOCKER_CMD[@]}"))
HARBOR_CMD=($(shell_join "${HARBOR_CMD[@]}"))

SERVER_LOG_PID=""
DOCKER_NETWORK_CREATED=0
cleanup() {
    if [[ -n "\${SERVER_LOG_PID}" ]] && kill -0 "\${SERVER_LOG_PID}" >/dev/null 2>&1; then
        kill "\${SERVER_LOG_PID}" >/dev/null 2>&1 || true
        wait "\${SERVER_LOG_PID}" >/dev/null 2>&1 || true
    fi
    if [[ "\${SERVER_ENABLED}" == "1" ]]; then
        docker rm -f "\${SWITCHYARD_DOCKER_CONTAINER}" >/dev/null 2>&1 || true
    fi
    if [[ "\${DOCKER_NETWORK_CREATED}" == "1" ]]; then
        docker network rm "\${SWITCHYARD_DOCKER_NETWORK}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

export PYTHONHASHSEED=0
export LC_ALL=C.UTF-8

if [[ "\${SERVER_ENABLED}" == "1" ]]; then
    {
    echo "Starting Dockerized Switchyard service"
    echo "  image:   \${SWITCHYARD_DOCKER_IMAGE}"
    echo "  network: \${SWITCHYARD_DOCKER_NETWORK}"
    echo "  command: switchyard \${SERVER_DOCKER_CMD[*]}"
    if [[ "\${SWITCHYARD_DOCKER_BUILD}" != "0" ]]; then
        docker build -f "\${SWITCHYARD_DOCKERFILE}" -t "\${SWITCHYARD_DOCKER_IMAGE}" .
    fi
    if ! docker network inspect "\${SWITCHYARD_DOCKER_NETWORK}" >/dev/null 2>&1; then
        docker network create "\${SWITCHYARD_DOCKER_NETWORK}"
        DOCKER_NETWORK_CREATED=1
    fi
    docker rm -f "\${SWITCHYARD_DOCKER_CONTAINER}" >/dev/null 2>&1 || true
    DOCKER_RUN_ARGS=(
        -d --rm
        --name "\${SWITCHYARD_DOCKER_CONTAINER}"
        --network "\${SWITCHYARD_DOCKER_NETWORK}"
        --network-alias "\${SWITCHYARD_DOCKER_SERVICE_NAME}"
        -p "127.0.0.1:$(q "${PORT}"):$(q "${PORT}")"
        -v "\${REPO_ROOT}:\${REPO_ROOT}:ro"
        -v "\${RUN_DIR}:\${RUN_DIR}"
    )
    if [[ -n "\${ROUTING_PROFILES_DIR}" && "\${ROUTING_PROFILES_DIR}" != "\${REPO_ROOT}"* ]]; then
        DOCKER_RUN_ARGS+=(-v "\${ROUTING_PROFILES_DIR}:\${ROUTING_PROFILES_DIR}:ro")
    fi
    if [[ -n "\${UPSTREAM_API_KEY_ENV}" ]]; then
        DOCKER_RUN_ARGS+=(--env "\${UPSTREAM_API_KEY_ENV}")
    fi
    docker run "\${DOCKER_RUN_ARGS[@]}" \\
        --workdir "\${REPO_ROOT}" \\
        --env PYTHONHASHSEED=0 \\
        --env LC_ALL=C.UTF-8 \\
        --env OPENAI_API_KEY \\
        --env OPENROUTER_API_KEY \\
        --env UPSTREAM_API_KEY \\
        --env ANTHROPIC_API_KEY \\
        --env ANTHROPIC_AUTH_TOKEN \\
        --env SWITCHYARD_STRONG_API_KEY \\
        --env SWITCHYARD_WEAK_API_KEY \\
        --env SWITCHYARD_CLASSIFIER_API_KEY \\
        --env AWS_ACCESS_KEY_ID \\
        --env AWS_SECRET_ACCESS_KEY \\
        --env AWS_SESSION_TOKEN \\
        --env AWS_REGION \\
        --env AWS_DEFAULT_REGION \\
        "\${SWITCHYARD_DOCKER_IMAGE}" "\${SERVER_DOCKER_CMD[@]}"
    } > "\${SERVER_LOG}" 2>&1
    echo "\${SWITCHYARD_DOCKER_CONTAINER}" > "\${RUN_DIR}/server.pid"
    docker logs -f "\${SWITCHYARD_DOCKER_CONTAINER}" >> "\${SERVER_LOG}" 2>&1 &
    SERVER_LOG_PID="\$!"
else
    {
        echo "Switchyard disabled; using direct upstream"
        echo "  network: \${SWITCHYARD_DOCKER_NETWORK}"
        echo "  upstream: \${HARBOR_BASE_URL}"
        if ! docker network inspect "\${SWITCHYARD_DOCKER_NETWORK}" >/dev/null 2>&1; then
            docker network create "\${SWITCHYARD_DOCKER_NETWORK}"
            DOCKER_NETWORK_CREATED=1
        fi
    } > "\${SERVER_LOG}" 2>&1
    echo "\${SWITCHYARD_DOCKER_NETWORK}" > "\${RUN_DIR}/docker_network"
fi

if [[ "\${SERVER_ENABLED}" == "1" && "\${SKIP_HEALTH_CHECK}" != "1" ]]; then
    echo "Waiting for health: \${SERVER_HEALTH_URL}"
    healthy=0
    for _i in {1..240}; do
        if curl -fsS "\${SERVER_HEALTH_URL}" >/dev/null 2>&1; then
            healthy=1
            break
        fi
        sleep 0.5
    done
    if [[ "\${healthy}" != "1" ]]; then
        echo "ERROR: server did not become healthy: \${SERVER_HEALTH_URL}" >&2
        python3 benchmark/run_manifest.py finalize \\
            --manifest "\${MANIFEST_PATH}" \\
            --harbor-rc 124 \\
            --harbor-job-dir "\${HARBOR_JOB_DIR}" \\
            --routing-stats "\${ROUTING_STATS_JSON}" || true
        exit 124
    fi
fi

if [[ "\${SERVER_ENABLED}" == "1" && "\${SWITCHYARD_CLOSED_BOOK_PREFLIGHT:-1}" != "0" ]]; then
    echo "Checking benchmark Docker network reachability: \${HARBOR_SERVER_ROOT_URL}/health"
    if ! docker run --rm --network "\${SWITCHYARD_DOCKER_NETWORK}" \\
        "\${SWITCHYARD_PREFLIGHT_IMAGE:-python:3.12-slim}" \\
        python -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5).read()' \\
        "\${HARBOR_SERVER_ROOT_URL}/health" \\
        >/dev/null 2>&1; then
        echo "ERROR: benchmark Docker containers cannot reach Switchyard on network \${SWITCHYARD_DOCKER_NETWORK}." >&2
        echo "       Harbor would hang before reaching Switchyard; see \${SERVER_LOG}." >&2
        python3 benchmark/run_manifest.py finalize \\
            --manifest "\${MANIFEST_PATH}" \\
            --harbor-rc 125 \\
            --harbor-job-dir "\${HARBOR_JOB_DIR}" \\
            --routing-stats "\${ROUTING_STATS_JSON}" || true
        exit 125
    fi
fi

if [[ "\${SERVER_ENABLED}" == "1" ]]; then
    export OPENAI_API_KEY="\${OPENAI_API_KEY:-switchyard-local}"
else
    if [[ -z "\${!UPSTREAM_API_KEY_ENV:-}" ]]; then
        echo "ERROR: direct upstream requires \${UPSTREAM_API_KEY_ENV} to be set." >&2
        python3 benchmark/run_manifest.py finalize \\
            --manifest "\${MANIFEST_PATH}" \\
            --harbor-rc 126 \\
            --harbor-job-dir "\${HARBOR_JOB_DIR}" || true
        exit 126
    fi
    export OPENAI_API_KEY="\${!UPSTREAM_API_KEY_ENV}"
fi
export OPENAI_BASE_URL="\${HARBOR_BASE_URL}"
if [[ "\${BOOK_MODE}" == "closed" ]]; then
    export CLOSED_BOOK_MODE=1
else
    export CLOSED_BOOK_MODE=0
fi
if [[ "\${SERVER_ENABLED}" == "1" ]]; then
    export ALLOWED_HOSTS="\${ALLOWED_HOSTS:-\${SWITCHYARD_DOCKER_SERVICE_NAME}}"
else
    export ALLOWED_HOSTS="\${ALLOWED_HOSTS:-}"
fi
export SWITCHYARD_BASE_URL="\${SWITCHYARD_BASE_URL:-\${HARBOR_SERVER_ROOT_URL}}"
export SWITCHYARD_DOCKER_NETWORK
export SWITCHYARD_VERIFIER_PROXY_TOKEN="\${SWITCHYARD_VERIFIER_PROXY_TOKEN:-\$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"
export SWITCHYARD_VERIFIER_HTTP_PROXY="http://verifier:\${SWITCHYARD_VERIFIER_PROXY_TOKEN}@proxy:3129"
if [[ "\${BOOK_MODE}" == "closed" ]]; then
    if [[ "\${AGENT}" == "codex" ]]; then
        export CODEX_DISABLE_WEB_SEARCH="\${CODEX_DISABLE_WEB_SEARCH:-1}"
    elif [[ "\${AGENT}" == "opencode" ]]; then
        export OPENCODE_DISABLE_WEBFETCH="\${OPENCODE_DISABLE_WEBFETCH:-1}"
    fi
fi

if [[ "\${AGENT}" == "claude-code" ]]; then
    export ANTHROPIC_BASE_URL="\${HARBOR_SERVER_ROOT_URL}"
    if [[ "\${SERVER_ENABLED}" == "1" ]]; then
        export ANTHROPIC_AUTH_TOKEN="\${ANTHROPIC_AUTH_TOKEN:-switchyard}"
        export ANTHROPIC_API_KEY="\${ANTHROPIC_API_KEY:-}"
    else
        export ANTHROPIC_AUTH_TOKEN="\${ANTHROPIC_AUTH_TOKEN:-\${!UPSTREAM_API_KEY_ENV}}"
        export ANTHROPIC_API_KEY="\${ANTHROPIC_API_KEY:-\${!UPSTREAM_API_KEY_ENV}}"
    fi
    export ANTHROPIC_MODEL="\${ANTHROPIC_MODEL:-\${HARBOR_MODEL}}"
    export ANTHROPIC_SMALL_FAST_MODEL="\${ANTHROPIC_SMALL_FAST_MODEL:-\${HARBOR_MODEL}}"
    export ANTHROPIC_CUSTOM_MODEL_OPTION="\${ANTHROPIC_CUSTOM_MODEL_OPTION:-\${HARBOR_MODEL}}"
    export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY="\${CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY:-1}"
    if [[ "\${SERVER_ENABLED}" == "1" ]]; then
        export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="\${ANTHROPIC_CUSTOM_MODEL_OPTION_NAME:-\${HARBOR_MODEL##*/} (Switchyard)}"
        export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="\${ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION:-Routed via switchyard -> \${HARBOR_MODEL}}"
    else
        export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="\${ANTHROPIC_CUSTOM_MODEL_OPTION_NAME:-\${HARBOR_MODEL##*/} (direct)}"
        export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="\${ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION:-Direct upstream -> \${HARBOR_MODEL}}"
    fi
fi

echo "Running Harbor: \${HARBOR_CMD[*]}"
set +e
"\${HARBOR_CMD[@]}" > "\${HARBOR_LOG}" 2>&1
HARBOR_RC="\$?"
set -e
echo "\${HARBOR_RC}" > "\${RUN_DIR}/harbor_exit"

FINALIZE_CMD=(python3 benchmark/run_manifest.py finalize \\
    --manifest "\${MANIFEST_PATH}" \\
    --harbor-rc "\${HARBOR_RC}" \\
    --harbor-job-dir "\${HARBOR_JOB_DIR}")
if [[ "\${SERVER_ENABLED}" == "1" ]]; then
    curl -fsS "\${SERVER_STATS_URL}" -o "\${ROUTING_STATS_JSON}" >/dev/null 2>&1 || true
    FINALIZE_CMD+=(--routing-stats "\${ROUTING_STATS_JSON}"
        --routing-log "\${ROUTING_LOG_JSONL}"
        --routing-stats-by-task "\${ROUTING_STATS_BY_TASK_JSON}")
fi
"\${FINALIZE_CMD[@]}" || true

exit "\${HARBOR_RC}"
EOF
chmod +x "${WRAPPER}"

if [[ "${FOREGROUND}" -eq 1 ]]; then
    "${WRAPPER}" 2>&1 | tee "${LOG_PATH}"
    exit "${PIPESTATUS[0]}"
fi

nohup "${WRAPPER}" > "${LOG_PATH}" 2>&1 &
PID=$!
disown "${PID}" 2>/dev/null || true

echo
echo "Started PID ${PID}"
echo
echo "Monitor with:"
echo "  tail -f ${LOG_PATH}"
echo
echo "Stop with:"
echo "  kill ${PID}"
