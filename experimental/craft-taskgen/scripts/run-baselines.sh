#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Launch a Harbor baseline run (one agent, one model) against a task dataset.
#
# Harbor itself walks the dataset and runs N trials in parallel
# (--n-concurrent). This launcher just sets up env vars, runs preflight,
# and nohups the single harbor invocation so it survives terminal exit.
#
# Usage:
#   scripts/run-baselines.sh \
#       --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v1a \
#       --agent opencode \
#       --model aws/anthropic/bedrock-claude-haiku-4-5 \
#       --backend gateway \
#       --n-tasks 3
#
# For a local vLLM endpoint:
#   export VLLM_BASE_URL=http://localhost:8000/v1
#   export VLLM_API_KEY=EMPTY
#   scripts/run-baselines.sh \
#       --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v1a \
#       --agent opencode \
#       --model qwen/Qwen3.5-397B-A17B-FP8 \
#       --backend vllm
#
# Exits immediately after spawning harbor. The PID, log path, and kill
# hint are printed. Artifacts land under <output-dir>/.

set -euo pipefail

TASKS_DIR=""
AGENT=""
MODEL=""
BACKEND="gateway"
OUTPUT_DIR=""
TASK_NAME=""           # glob forwarded to harbor's --include-task-name (v0.6.4 renamed)
EXCLUDE_TASK_NAME=""   # glob forwarded to --exclude-task-name
N_TASKS=0              # 0 = no limit
N_CONCURRENT=4
SKIP_PREFLIGHT=0
DRY_RUN=0
FORCE_BUILD=0          # 1 = pass --force-build to harbor (invalidate image cache)

usage() {
    cat <<'EOF'
Launch a Harbor baseline run (one agent, one model) against a task dataset.

Harbor walks the dataset and runs trials in parallel (--n-concurrent).
This launcher sets up env vars, runs preflight, and nohups the harbor
invocation so it survives terminal exit. Prints the PID, log path, and
kill hint, then exits.

Required:
  --tasks-dir PATH          Task dataset directory (e.g. harbor-tasks/craft-taskgen-v1a)
  --agent NAME              One of: claude-code, codex, opencode, openhands-sdk, qwen-coder, pi
  --model ID                Model ID passed to harbor --model (e.g. azure/anthropic/claude-opus-4-6)

Optional:
  --backend gateway|vllm    Inference backend (default: gateway)
  --output-dir PATH         Output dir for harbor job artifacts (default: baselines/<ts>)
  --task-name GLOB[,...]    Forwarded to harbor --include-task-name (include filter).
                            Accepts a comma-separated list; each entry becomes
                            its own --include-task-name flag.
  --exclude-task-name GLOB  Forwarded to harbor --exclude-task-name
  --n-tasks N               Forwarded to harbor --n-tasks (0 = no limit; default: 0)
  --n-concurrent N          Forwarded to harbor --n-concurrent (default: 4)
  --skip-preflight          Skip the preflight check
  --force-build             Pass --force-build to harbor so Docker layer cache
                            is invalidated and task images are rebuilt. Use
                            after Dockerfile edits (e.g. agent-version pins).
  --dry-run                 Print the harbor command without executing

Examples:
  Gateway:
    scripts/run-baselines.sh \
        --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v1a \
        --agent opencode --model aws/anthropic/bedrock-claude-haiku-4-5 \
        --n-tasks 3

  vLLM (requires VLLM_BASE_URL in env):
    VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY \
      scripts/run-baselines.sh \
          --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v1a \
          --agent opencode --model qwen/Qwen3.5-397B-A17B-FP8 \
          --backend vllm
EOF
    exit "${1:-1}"
}

# Capture original argv (post-shell-quoting) before argparse consumes
# it, for run_manifest.launcher_argv.
ORIGINAL_ARGV_QUOTED=$(printf '%q ' "$@")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks-dir) TASKS_DIR="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --task-name) TASK_NAME="$2"; shift 2 ;;
        --exclude-task-name) EXCLUDE_TASK_NAME="$2"; shift 2 ;;
        --n-tasks) N_TASKS="$2"; shift 2 ;;
        --n-concurrent) N_CONCURRENT="$2"; shift 2 ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --force-build) FORCE_BUILD=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown arg: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$TASKS_DIR" || -z "$AGENT" ]]; then
    echo "ERROR: --tasks-dir and --agent are required" >&2
    usage 1
fi
# Validate --agent and classify it. IS_SANITY_AGENT gates the
# model/effort/cap/compaction wiring below — oracle and nop don't
# talk to an LLM, they just orchestrate harbor's own non-LLM agents
# which apply (or skip) the task's reference solution and run the
# verifier.
case "$AGENT" in
    claude-code|codex|opencode|openhands-sdk|qwen-coder|pi) IS_SANITY_AGENT=0 ;;
    oracle|nop) IS_SANITY_AGENT=1 ;;
    *)
        echo "ERROR: --agent must be claude-code, codex, opencode, openhands-sdk, qwen-coder, pi, oracle, or nop (got: $AGENT)" >&2
        exit 1
        ;;
esac
# --model is required for LLM-driven agents but irrelevant for sanity
# agents. If --model is given for sanity agents we drop it (with a
# warning) so the same launcher command shape works across agents in
# CI scripts.
if [[ "$IS_SANITY_AGENT" -eq 1 ]]; then
    if [[ -n "$MODEL" ]]; then
        echo "INFO: --model is ignored for --agent $AGENT (sanity check)" >&2
        MODEL=""
    fi
elif [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required for --agent $AGENT" >&2
    usage 1
fi
if [[ ! -d "$TASKS_DIR" ]]; then
    echo "ERROR: tasks dir not found: $TASKS_DIR" >&2
    exit 1
fi
if [[ "$BACKEND" != "gateway" && "$BACKEND" != "vllm" ]]; then
    echo "ERROR: --backend must be gateway or vllm (got: $BACKEND)" >&2
    exit 1
fi

# The `nvidia/` / `vllm/` prefixes are opencode-internal dispatch tokens
# (they tell opencode to load @ai-sdk/openai-compatible); they are not
# part of the model identity the endpoint sees. If a user passes either
# form on --model (legacy or explicit), strip it so every downstream
# consumer — preflight, reasoning_defaults lookup, sampling-family
# match, manifest — sees the canonical slug. The launcher re-adds the
# appropriate prefix when constructing harbor's --model (below).
if [[ "$AGENT" == "opencode" ]]; then
    # Strip the dispatch prefix only when it isn't actually the
    # canonical gateway vendor prefix. Gateway-vendor-prefixed slugs are
    # `aws/...`, `azure/...`, `openai/...` (Anthropic, Azure, etc.); the
    # `nvidia/...` and `vllm/...` here are opencode-internal dispatch
    # tokens that need stripping. nvidia-hosted models also start with
    # `nvidia/` (e.g. `nvidia/qwen/qwen3.6-35b-a3b`), so detect them by
    # the SECOND segment: if it's a known gateway-vendor prefix, the
    # leading `nvidia/` is a legacy dispatch token and we strip; if it
    # isn't (e.g. `nvidia/qwen`, `nvidia/zai-org`, `nvidia/nvidia`), the
    # whole slug IS the canonical name and must be preserved.
    if [[ "$MODEL" == nvidia/aws/* || "$MODEL" == nvidia/azure/* || "$MODEL" == nvidia/openai/* ]]; then
        MODEL="${MODEL#nvidia/}"
    elif [[ "$MODEL" == vllm/* ]]; then
        MODEL="${MODEL#vllm/}"
    fi
fi

# openhands-sdk: the SDK adapter passes self.model_name straight to LiteLLM,
# which interprets the first slug segment as a provider name (aws/, azure/,
# nvidia/) and demands provider-specific creds. To route through an
# OpenAI-compatible endpoint (NVIDIA gateway or local vLLM) we prepend
# `openai/` so LiteLLM treats the full slug as opaque against LLM_BASE_URL.
# Strip if the user already wrote it; we re-add below.
if [[ "$AGENT" == "openhands-sdk" && "$MODEL" == openai/* ]]; then
    MODEL="${MODEL#openai/}"
fi

# Harbor's opencode agent splits --model at the first `/` and uses the left
# segment as the provider key; the remainder is the model ID that actually
# goes on the wire. The patched opencode.py treats both `nvidia` and `vllm`
# as openai-compatible providers (different base URLs; same wire protocol).
HARBOR_MODEL="$MODEL"
if [[ "$AGENT" == "opencode" ]]; then
    if [[ "$BACKEND" == "gateway" ]]; then
        HARBOR_MODEL="nvidia/$MODEL"
    elif [[ "$BACKEND" == "vllm" ]]; then
        HARBOR_MODEL="vllm/$MODEL"
    fi
fi
if [[ "$AGENT" == "openhands-sdk" ]]; then
    HARBOR_MODEL="openai/$MODEL"
fi
# pi: harbor's patched pi.py recognizes `nvidia` as a custom
# openai-completions provider (writes per-model models.json at run time
# pointing at OPENAI_BASE_URL). The launcher prepends `nvidia/` as the
# pi-side dispatch token; pi.py strips it before sending on the wire,
# so the gateway sees the canonical slug.
if [[ "$AGENT" == "pi" && "$MODEL" == nvidia/* ]]; then
    # User wrote --model nvidia/...; we still need to prepend another
    # nvidia/ for pi's dispatch. Don't strip — just prepend below.
    :
fi
if [[ "$AGENT" == "pi" ]]; then
    HARBOR_MODEL="nvidia/$MODEL"
fi

TS=$(date +%Y%m%d-%H%M%S)
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="baselines/$TS"
fi
mkdir -p "$OUTPUT_DIR"

# Load .env so we can forward ANTHROPIC_*/OPENAI_*/VLLM_* into agent env.
# Parse KEY=VALUE lines instead of `source`-ing the file: `source` executes
# .env as a shell script, so any command in it would run (CWE-94).
if [[ -f .env ]]; then
    while IFS= read -r _envline || [[ -n "$_envline" ]]; do
        [[ "$_envline" =~ ^[[:space:]]*(#|$) ]] && continue
        _envline="${_envline#"${_envline%%[![:space:]]*}"}"   # ltrim
        [[ "$_envline" == export\ * ]] && _envline="${_envline#export }"
        if [[ "$_envline" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            _envkey="${BASH_REMATCH[1]}"
            _envval="${BASH_REMATCH[2]}"
            _envval="${_envval%\"}"; _envval="${_envval#\"}"     # strip paired "
            _envval="${_envval%\'}"; _envval="${_envval#\'}"     # strip paired '
            export "$_envkey=$_envval"
        fi
    done < .env
    unset _envline _envkey _envval
fi

# Pin the in-container LLM judge to the NVIDIA gateway independent of the
# agent's endpoint. Captured BEFORE the vllm branch rewrites OPENAI_*, so the
# task.toml verifier.env block (which reads ${JUDGE_BASE_URL}/${JUDGE_API_KEY})
# keeps pointing at the gateway even when the agent is routed to local vLLM.
# Explicit JUDGE_* overrides win; otherwise fall back to OPENAI_API_BASE /
# OPENAI_BASE_URL / OPENAI_API_KEY from .env.
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"

# Codex only emits reasoning tokens when it recognizes the model slug as
# reasoning-capable. Gateway-routed slugs aren't in codex's built-in catalog,
# so point CODEX_MODEL_CATALOG_JSON at the per-repo catalog (harbor patch
# reads this env var and passes -c model_catalog_json=... to `codex exec`).
#
# Filter to a single-model catalog at launch time. Harbor's codex.py inlines
# the catalog file content into the `bash -c '<setup_command>'` argv on
# docker exec; the full multi-model catalog blows past Linux's 128 KB
# MAX_ARG_STRLEN per argv element (E2BIG: "Argument list too long"), even
# though total ARG_MAX is ~2 MB. A single-model catalog is ~30 KB.
#
# Explicit CODEX_MODEL_CATALOG_JSON override is honored verbatim.
if [[ "$AGENT" == "codex" ]]; then
    DEFAULT_CATALOG="$(cd "$(dirname "$0")/.." && pwd)/patches/codex-model-catalog.json"
    if [[ -z "${CODEX_MODEL_CATALOG_JSON:-}" ]]; then
        if [[ ! -f "$DEFAULT_CATALOG" ]]; then
            echo "ERROR: source catalog not found: $DEFAULT_CATALOG" >&2
            exit 1
        fi
        FILTERED_CATALOG="$OUTPUT_DIR/codex-model-catalog-${MODEL//\//-}.json"
        # Pass paths/slug as argv (read via sys.argv) using a quoted heredoc, so
        # a $MODEL or path containing quotes cannot break out of the Python
        # string literal and inject code (CWE-94/CWE-78).
        if ! python3 - "$DEFAULT_CATALOG" "$MODEL" "$FILTERED_CATALOG" <<'PY'
import json, sys
src_path, slug, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
src = json.load(open(src_path))
matches = [m for m in src.get('models', []) if m.get('slug') == slug]
if not matches:
    sys.stderr.write(f'ERROR: no entry for slug={slug!r} in {src_path}; '
                     'add a row to patches/codex-model-catalog.json before launching.\n')
    sys.exit(1)
json.dump({'models': matches}, open(out_path, 'w'), indent=2)
PY
        then
            exit 1
        fi
        export CODEX_MODEL_CATALOG_JSON="$FILTERED_CATALOG"
    elif [[ ! -f "$CODEX_MODEL_CATALOG_JSON" ]]; then
        echo "WARNING: CODEX_MODEL_CATALOG_JSON=$CODEX_MODEL_CATALOG_JSON not found;" >&2
        echo "         codex will run without reasoning (catalog lookup will fail)." >&2
    fi
fi

if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
    echo "Running baseline preflight..."
    PREFLIGHT_ARGS=(--check-baselines --backend "$BACKEND" --agents "$AGENT")
    if [[ "$IS_SANITY_AGENT" -eq 0 ]]; then
        PREFLIGHT_ARGS+=(--model "$MODEL")
    fi
    if ! uv run craft-taskgen-preflight "${PREFLIGHT_ARGS[@]}"; then
        echo "" >&2
        echo "ABORT: preflight failed. Fix issues above or pass --skip-preflight." >&2
        exit 1
    fi

    # Every task.toml in the dataset must declare memory_mb >= 8192. Catches
    # datasets generated against an older default before harbor silently OOMs
    # a trial mid-run. Tasks may declare more than the floor (e.g. 16384 for
    # heavy builds); only values below the floor block.
    MIN_MEMORY_MB=8192
    mismatched=()
    while IFS= read -r toml; do
        actual=$(grep -E '^\s*memory_mb\s*=' "$toml" | head -1 | sed -E 's/.*=\s*([0-9]+).*/\1/')
        if [[ -z "$actual" ]] || (( actual < MIN_MEMORY_MB )); then
            mismatched+=("$(basename "$(dirname "$toml")"): memory_mb=${actual:-<missing>}")
        fi
    done < <(find "$TASKS_DIR" -mindepth 2 -maxdepth 2 -name task.toml)
    if [[ ${#mismatched[@]} -gt 0 ]]; then
        echo "" >&2
        echo "ABORT: ${#mismatched[@]} task.toml file(s) declare memory_mb < $MIN_MEMORY_MB:" >&2
        printf '  %s\n' "${mismatched[@]}" >&2
        echo "Regenerate the dataset (or fix the candidates) so every task.toml declares memory_mb >= $MIN_MEMORY_MB." >&2
        exit 1
    fi
    echo "OK: all task.toml files declare memory_mb >= $MIN_MEMORY_MB"
    echo ""
fi

AGENT_ENV_ARGS=()
_add_env() {
    AGENT_ENV_ARGS+=(--agent-env "$1=$2")
}

# Determinism envs. Applied unconditionally so in-container pytest is
# reproducible: PYTHONHASHSEED pins Python dict iteration order; LC_ALL
# pins locale-dependent string comparisons and sort. Recorded in the
# run manifest under the `determinism` section.
_add_env PYTHONHASHSEED 0
_add_env LC_ALL C.UTF-8

# Sanity agents (oracle/nop) don't speak to an LLM, so they don't need
# any of the gateway/vllm endpoint plumbing or the claude-code opt-outs.
if [[ "$IS_SANITY_AGENT" -eq 0 ]]; then
    if [[ "$BACKEND" == "gateway" ]]; then
        # Secret API keys are NOT passed via --agent-env: that lands on the
        # harbor process argv (visible to `ps`). `source .env` above already
        # exported ANTHROPIC_API_KEY / OPENAI_API_KEY into this process env, so
        # harbor inherits them in os.environ and every installed agent reads
        # them from there (opencode / claude-code read os.environ directly;
        # codex / openhands via _get_env). Only non-secret base URLs are
        # forwarded explicitly.
        _add_env ANTHROPIC_BASE_URL "${ANTHROPIC_BASE_URL:-}"
        _add_env OPENAI_BASE_URL "${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
        _add_env CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC 1
        _add_env CLAUDE_CODE_ATTRIBUTION_HEADER 0
        _add_env CLAUDE_CODE_ENABLE_TELEMETRY 0
        # Disable experimental betas so claude-code does not inject the
        # context_management beta field. Moved here from the harbor
        # claude_code.py patch (env vars reach the agent via --agent-env).
        _add_env CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS 1
        if [[ "$AGENT" == "openhands-sdk" ]]; then
            # SDK adapter reads LLM_API_KEY / LLM_BASE_URL directly (no provider
            # sniffing); ANTHROPIC_* / OPENAI_* are inert for it. Export the key
            # into os.environ (read via _get_env) instead of --agent-env so it
            # stays off the process argv; LLM_BASE_URL is not secret.
            export LLM_API_KEY="${OPENAI_API_KEY:-}"
            _add_env LLM_BASE_URL "${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
        fi
        if [[ "$AGENT" == "qwen-coder" ]]; then
            # Harbor's QwenCode agent reads OPENAI_MODEL inside the container
            # and falls back to "qwen3-coder-plus" if unset. Set it explicitly
            # to the gateway slug so we never silently drift to the default.
            _add_env OPENAI_MODEL "$MODEL"
        fi
        if [[ "$AGENT" == "pi" ]]; then
            # pi's harbor wrapper writes ~/.pi/agent/models.json at run-time
            # with baseUrl baked in from the host's OPENAI_BASE_URL; the
            # container needs OPENAI_API_KEY to authenticate. Forwarded by
            # the gateway branch above already, no extra plumbing needed.
            :
        fi
    else
        if [[ -z "${VLLM_BASE_URL:-}" ]]; then
            echo "ERROR: --backend vllm requires VLLM_BASE_URL in env/.env" >&2
            exit 1
        fi
        # Rewrite localhost / 127.0.0.1 → host.docker.internal so the container
        # can reach a vLLM running on the host. The host-gateway compose
        # override (added to CMD below) maps host.docker.internal → host-gateway,
        # which works on macOS Docker Desktop and on Linux.
        CONTAINER_BASE_URL="${VLLM_BASE_URL//localhost/host.docker.internal}"
        CONTAINER_BASE_URL="${CONTAINER_BASE_URL//127.0.0.1/host.docker.internal}"
        _add_env OPENAI_BASE_URL "$CONTAINER_BASE_URL"
        _add_env OPENAI_API_KEY "${VLLM_API_KEY:-EMPTY}"
        # Map host.docker.internal → host-gateway via harbor's native
        # --extra-docker-compose (replaces the old docker-compose-base.yaml
        # patch, which no longer exists as of harbor a56546f).
        VLLM_NEEDS_HOST_GATEWAY=1
        if [[ "$AGENT" == "openhands-sdk" ]]; then
            _add_env LLM_API_KEY "${VLLM_API_KEY:-EMPTY}"
            _add_env LLM_BASE_URL "$CONTAINER_BASE_URL"
        fi
        if [[ "$AGENT" == "qwen-coder" ]]; then
            _add_env OPENAI_MODEL "$MODEL"
        fi
    fi
fi

if [[ -n "${OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX:-}" ]]; then
    _add_env OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX "$OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"
fi

SUITE_NAME="$(basename "$TASKS_DIR")"
JOB_NAME="baseline-${AGENT}-${SUITE_NAME}-${TS}"
LOG="$OUTPUT_DIR/${JOB_NAME}.log"

# Pin the agent CLI version so runs are reproducible. Source of truth is
# src/craft_taskgen/adapters/_docker.py — bump there, pick up here for free.
# Sanity agents (oracle/nop) are pure harbor built-ins with no CLI
# version to pin.
AGENT_VERSION=""
if [[ "$IS_SANITY_AGENT" -eq 0 ]]; then
    AGENT_VERSION=$(uv run python -c "
import sys
from craft_taskgen.adapters._docker import (
    CLAUDE_CODE_VERSION, CODEX_VERSION, OPENCODE_VERSION, OPENHANDS_SDK_VERSION,
    QWEN_CODE_VERSION, PI_VERSION,
)
versions = {
    'claude-code':   CLAUDE_CODE_VERSION,
    'codex':         CODEX_VERSION,
    'opencode':      OPENCODE_VERSION,
    'openhands-sdk': OPENHANDS_SDK_VERSION,
    'qwen-coder':    QWEN_CODE_VERSION,
    'pi':            PI_VERSION,
}
agent = '$AGENT'
if agent not in versions:
    print(f'no pinned version for agent={agent!r} (known: {sorted(versions)})', file=sys.stderr)
    sys.exit(1)
print(versions[agent])
") || exit 1
fi

# Resolve reasoning_effort for this (agent, model) combo. Source of truth:
# src/craft_taskgen/baselines/reasoning_defaults.py. Env-var REASONING_EFFORT_OVERRIDE
# bypasses the table — set to empty string to disable pass-through entirely.
# Sanity agents (oracle/nop) don't reason; leave EFFORT/REASONING_* empty.
EFFORT=""
REASONING_SOURCE=""
REASONING_NOTES=""
if [[ "$IS_SANITY_AGENT" -eq 0 ]]; then
    EFFORT=$(uv run python - "$AGENT" "$MODEL" <<'PY'
import sys
from craft_taskgen.baselines.reasoning_defaults import effort_for
e = effort_for(sys.argv[1], sys.argv[2])
if e:
    print(e)
PY
)
    REASONING_SOURCE="reasoning_defaults"
    if [[ -n "${REASONING_EFFORT_OVERRIDE+x}" ]]; then
        EFFORT="$REASONING_EFFORT_OVERRIDE"
        REASONING_SOURCE="REASONING_EFFORT_OVERRIDE"
    fi
fi

# Per-agent wiring that is ALWAYS applied — independent of effort so the
# manifest values always match what the container received. Effort-
# dependent wiring lives in the second block below.
case "$AGENT" in
    claude-code)
        # Pin auto-compaction threshold explicitly (default per docs is
        # ~95%). Values above default have no effect, so this is the
        # closest-to-off setting Anthropic exposes. Always on — unrelated
        # to whether we're also setting effort.
        # https://code.claude.com/docs/en/env-vars
        _add_env CLAUDE_AUTOCOMPACT_PCT_OVERRIDE 95
        ;;
esac

# Agent-specific effort wiring — without these, effort is silently
# dropped. Only runs when EFFORT is actually set for this (agent, model).
if [[ -n "$EFFORT" ]]; then
    case "$AGENT" in
        codex)
            # Catalog env is already exported earlier in this script (see the
            # CODEX_MODEL_CATALOG_JSON block near the top). Surface its value
            # in the summary so the operator can confirm what's loaded.
            REASONING_NOTES="catalog=${CODEX_MODEL_CATALOG_JSON:-<unset>}"
            ;;
        claude-code)
            # Gateway-routed opus/sonnet drops effort silently without this
            # capability declaration. See
            # https://code.claude.com/docs/en/model-config.
            _CAPS="effort,thinking,adaptive_thinking,interleaved_thinking"
            _add_env ANTHROPIC_DEFAULT_OPUS_MODEL_SUPPORTED_CAPABILITIES "$_CAPS"
            _add_env ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES "$_CAPS"
            # CLAUDE_CODE_EFFORT_LEVEL is the documented env var
            # (https://code.claude.com/docs/en/env-vars) and takes precedence
            # over --effort / the effortLevel setting. Set it alongside the
            # --agent-kwarg reasoning_effort (which harbor renders as
            # --effort) so the effort flows even on CLI versions where the
            # flag name drifts.
            _add_env CLAUDE_CODE_EFFORT_LEVEL "$EFFORT"
            REASONING_NOTES="capabilities=$_CAPS; CLAUDE_CODE_EFFORT_LEVEL=$EFFORT"
            ;;
        opencode)
            # Harbor's opencode patch reads OPENCODE_REASONING_EFFORT on the
            # HOST when building the install.sh (os.environ.get inside
            # .venv/.../harbor/agents/installed/opencode.py), so we export it
            # rather than passing --agent-env (which forwards to the container
            # where the config has already been generated).
            export OPENCODE_REASONING_EFFORT="$EFFORT"
            REASONING_NOTES="OPENCODE_REASONING_EFFORT=$EFFORT (host-exported)"
            ;;
        openhands-sdk)
            # SDK adapter doesn't take reasoning_effort as a kwarg. Reasoning
            # for models like Nemotron-Ultra is configured at the endpoint
            # (vLLM reasoning-parser plugin). Effort here is a no-op.
            REASONING_NOTES="openhands-sdk: no effort kwarg (reasoning is endpoint-side)"
            ;;
        pi)
            # Pi exposes effort as --thinking {off,minimal,low,medium,high,xhigh}
            # (CLI_FLAGS in harbor/agents/installed/pi.py). Harbor surfaces it as
            # the `thinking` agent-kwarg, not `reasoning_effort`. So we rename
            # the kwarg below by setting PI_THINKING and emptying EFFORT so the
            # generic `reasoning_effort=$EFFORT` kwarg isn't appended.
            PI_THINKING="$EFFORT"
            EFFORT=""
            REASONING_NOTES="pi --thinking=$PI_THINKING (mapped from reasoning_defaults)"
            ;;
    esac
fi

# Qwen3 family on opencode: reasoning_effort is a no-op; Qwen3 uses
# server-side enable_thinking (default true with --reasoning-parser qwen3)
# and the HF model card recommends specific sampling parameters for
# thinking mode that are NOT opencode's defaults. Set them here so the
# paper can cite paper-recommended settings (Qwen3 tech report §3.3 +
# Qwen/Qwen3-32B HF card: temperature=0.6, top_p=0.95, top_k=20).
# Harbor reads OPENCODE_BUILD_{TEMPERATURE,TOP_P,TOP_K} on the host and
# writes them into opencode.json agent.build.* (opencode.py:374-391).
# Apply opencode-family sampling defaults per model card. Each entry is
# (family_label, match_test, T, p, k) where match_test is a bash
# pattern tested against "$MODEL". Empty `k` means the model card is
# silent on top_k — don't force a value.
#
# Sources:
#  - Qwen3: T=0.6 p=0.95 k=20    (Qwen3-32B HF card)
#  - MiniMax: T=1.0 p=0.95 k=40  (MiniMax-M2.5 HF card)
#  - Nemotron-Super-V3: T=1.0 p=0.95 — (nemotron-super-v3 model card
#    says "across all tasks", no top_k). Narrow match avoids Ultra/Nano
#    and the older Llama-3.3-Nemotron-Super-49B-v1.
#  - GLM (4.5+): T=1.0 p=0.95 — z.ai docs + HF generation_config.json.
#    top_k unspecified by model card.
apply_opencode_family_sampling() {
    local family="$1" T="$2" p="$3" k="$4"
    export OPENCODE_BUILD_TEMPERATURE="${OPENCODE_BUILD_TEMPERATURE:-$T}"
    export OPENCODE_BUILD_TOP_P="${OPENCODE_BUILD_TOP_P:-$p}"
    if [[ -n "$k" ]]; then
        export OPENCODE_BUILD_TOP_K="${OPENCODE_BUILD_TOP_K:-$k}"
        REASONING_NOTES="${REASONING_NOTES:+$REASONING_NOTES; }${family} sampling params T=${T} p=${p} k=${k}"
    else
        REASONING_NOTES="${REASONING_NOTES:+$REASONING_NOTES; }${family} sampling params T=${T} p=${p}"
    fi
}

if [[ "$AGENT" == "opencode" ]]; then
    if [[ "$MODEL" == *[Qq]wen* ]]; then
        apply_opencode_family_sampling "Qwen3" 0.6 0.95 20
        if [[ -n "${OPENCODE_ENABLE_THINKING:-}" ]]; then
            _add_env OPENCODE_ENABLE_THINKING "$OPENCODE_ENABLE_THINKING"
            REASONING_NOTES="OPENCODE_ENABLE_THINKING=$OPENCODE_ENABLE_THINKING (Qwen3 path, T=0.6 p=0.95 k=20)"
        fi
    elif [[ "$MODEL" == *[Mm]ini[Mm]ax* ]]; then
        apply_opencode_family_sampling "MiniMax" 1.0 0.95 40
    elif [[ "$MODEL" == *[Nn]emotron*[Ss]uper*[Vv]3* || "$MODEL" == *[Nn]emotron*3*[Ss]uper* || "$MODEL" == *[Nn]emotron-v3* ]]; then
        apply_opencode_family_sampling "Nemotron-Super-V3" 1.0 0.95 ""
    elif [[ "$MODEL" == *[Gg][Ll][Mm]* ]]; then
        apply_opencode_family_sampling "GLM" 1.0 0.95 ""
    fi
fi

# Global per-call output-token cap. Source of truth:
# src/craft_taskgen/baselines/output_cap.py. A cap, not a tuning knob —
# no override env. Applied per agent via whatever env the agent honors.
# Sanity agents (oracle/nop) don't generate output and have no cap to
# apply.
OUTPUT_CAP=0
OUTPUT_CAP_NOTES="N/A (sanity agent has no model output)"
if [[ "$IS_SANITY_AGENT" -eq 0 ]]; then
    OUTPUT_CAP=$(uv run python -c "from craft_taskgen.baselines import OUTPUT_TOKEN_CAP; print(OUTPUT_TOKEN_CAP)")
    case "$AGENT" in
        claude-code)
            # CLAUDE_CODE_MAX_OUTPUT_TOKENS (documented at code.claude.com/docs/en/env-vars)
            # is forwarded into the container by harbor (claude_code.py:870-871).
            _add_env CLAUDE_CODE_MAX_OUTPUT_TOKENS "$OUTPUT_CAP"
            OUTPUT_CAP_NOTES="CLAUDE_CODE_MAX_OUTPUT_TOKENS=$OUTPUT_CAP"
            ;;
        opencode)
            # Harbor's opencode agent reads OPENCODE_{BUILD,PLAN}_MAX_TOKENS on the
            # host at config-generation time (opencode.py:374-391) and writes them
            # into opencode.json under agent.{build,plan}.max_tokens.
            export OPENCODE_BUILD_MAX_TOKENS="$OUTPUT_CAP"
            export OPENCODE_PLAN_MAX_TOKENS="$OUTPUT_CAP"
            OUTPUT_CAP_NOTES="OPENCODE_{BUILD,PLAN}_MAX_TOKENS=$OUTPUT_CAP (host-exported)"
            ;;
        codex)
            # Codex's model_max_output_tokens config key is parsed but never
            # applied — openai/codex#4138. No functional cap today.
            OUTPUT_CAP_NOTES="UNCAPPED (openai/codex#4138)"
            ;;
        openhands-sdk)
            # SDK adapter has no max_tokens knob. Per the Ultra evaluator
            # handoff, max_tokens / max_completion_tokens are intentionally
            # dropped server-side (proxy interceptor) so generations aren't
            # capped below the model's max_output_tokens.
            OUTPUT_CAP_NOTES="UNCAPPED at agent (drop max_tokens server-side; see Ultra handoff)"
            ;;
        qwen-coder)
            # Qwen Code CLI honors a `maxOutputTokens` setting in ~/.qwen/settings.json
            # but harbor's install template doesn't plumb it. Runs uncapped.
            OUTPUT_CAP_NOTES="UNCAPPED at agent (no harbor plumbing for ~/.qwen/settings.json)"
            ;;
        pi)
            # Pi has no env var for per-call max_tokens; models.json's
            # maxTokensField only names the request field. Runs uncapped.
            OUTPUT_CAP_NOTES="UNCAPPED at agent (no env var for max_tokens)"
            ;;
    esac
fi

CMD=(
    uv run harbor run
    --path "$TASKS_DIR"
    --agent "$AGENT"
    --jobs-dir "$OUTPUT_DIR"
    --job-name "$JOB_NAME"
    --n-concurrent "$N_CONCURRENT"
    --yes
)
# --model and --agent-kwarg version=... only apply to LLM-driven agents.
# Harbor's oracle/nop agents take no model and have no CLI version.
if [[ -n "$HARBOR_MODEL" ]]; then
    CMD+=(--model "$HARBOR_MODEL")
fi
if [[ -n "$AGENT_VERSION" ]]; then
    CMD+=(--agent-kwarg "version=$AGENT_VERSION")
fi
if [[ -n "$EFFORT" ]]; then
    CMD+=(--agent-kwarg "reasoning_effort=$EFFORT")
fi
if [[ -n "${PI_THINKING:-}" ]]; then
    # pi-specific: --agent-kwarg thinking=<level> (see CLI_FLAGS in pi.py).
    CMD+=(--agent-kwarg "thinking=$PI_THINKING")
fi
# OpenHands SDK iteration cap. Matches the Ultra evaluator-sidecar pin (200).
if [[ "$AGENT" == "openhands-sdk" ]]; then
    CMD+=(--agent-kwarg "max_iterations=$(uv run python -c 'from craft_taskgen.adapters._docker import OPENHANDS_SDK_MAX_ITERATIONS; print(OPENHANDS_SDK_MAX_ITERATIONS)')")
fi
# Per-agent turn cap. Claude Code exposes max_turns via a harbor
# CliFlag (claude_code.py CLI_FLAGS); codex and opencode do not expose
# an equivalent knob — wall-clock timeout is their only cap. Document
# the asymmetry rather than pretending we control it.
MAX_TURNS=""
# Per-agent tool-restriction policy. Surfaced in the manifest via
# agent.disallowed_tools.
DISALLOWED_TOOLS=""
case "$AGENT" in
    claude-code)
        MAX_TURNS=250
        CMD+=(--agent-kwarg "max_turns=$MAX_TURNS")
        # Disable EnterPlanMode + ExitPlanMode by default. Haiku 4.5
        # self-selects into plan mode on ~50% of tasks, explores
        # read-only for 100+ turns, then treats "present plan" as
        # terminal — the trial exits with 0 code edits because
        # Write/Edit are blocked during plan mode and there's no
        # interactive user to approve ExitPlanMode. Repro:
        # plan-on produced 0 edits over 4+ trials; plan-off produced
        # 3-9 edits per trial within the first 50 turns. Mirrors
        # runner.py's smoke-test policy. Override for ablation
        # experiments via DISABLE_PLAN_MODE=0. Harbor's claude_code.py
        # already exposes this as a CliFlag, so no harbor patch needed.
        if [[ "${DISABLE_PLAN_MODE:-1}" == "1" ]]; then
            DISALLOWED_TOOLS="EnterPlanMode,ExitPlanMode"
            CMD+=(--agent-kwarg "disallowed_tools=$DISALLOWED_TOOLS")
        fi
        ;;
esac
if [[ "$FORCE_BUILD" -eq 1 ]]; then
    CMD+=(--force-build)
fi
if [[ "$N_TASKS" -gt 0 ]]; then
    CMD+=(--n-tasks "$N_TASKS")
fi
if [[ -n "$TASK_NAME" ]]; then
    # Allow comma-separated list; expand into one --include-task-name per entry.
    IFS=',' read -ra _TASK_NAMES <<< "$TASK_NAME"
    for _t in "${_TASK_NAMES[@]}"; do
        CMD+=(--include-task-name "$_t")
    done
fi
if [[ -n "$EXCLUDE_TASK_NAME" ]]; then
    CMD+=(--exclude-task-name "$EXCLUDE_TASK_NAME")
fi
if [[ "${VLLM_NEEDS_HOST_GATEWAY:-0}" -eq 1 ]]; then
    # vLLM-on-host reachability: map host.docker.internal → host-gateway via
    # harbor's native --extra-docker-compose (was a docker-compose-base.yaml patch
    # before the a56546f bump). See patches/compose-overrides/host-gateway.yaml.
    CMD+=(--extra-docker-compose "$(cd "$(dirname "$0")/.." && pwd)/patches/compose-overrides/host-gateway.yaml")
fi
CMD+=("${AGENT_ENV_ARGS[@]}")

echo "Launching baseline"
echo "  tasks_dir:    $TASKS_DIR"
if [[ -n "$AGENT_VERSION" ]]; then
    echo "  agent:        $AGENT (pinned @$AGENT_VERSION)"
else
    echo "  agent:        $AGENT (sanity check; no CLI version)"
fi
if [[ -n "$MODEL" ]]; then
    echo "  model:        $MODEL"
fi
if [[ "$HARBOR_MODEL" != "$MODEL" && -n "$HARBOR_MODEL" ]]; then
    case "$AGENT" in
        opencode)      _label="opencode provider dispatch" ;;
        openhands-sdk) _label="openai/ prefix for LiteLLM chat-completions" ;;
        *)             _label="provider prefix" ;;
    esac
    echo "  harbor --model: $HARBOR_MODEL ($_label)"
fi
echo "  backend:      $BACKEND"
if [[ -n "$EFFORT" ]]; then
    echo "  reasoning:    $EFFORT (source: $REASONING_SOURCE; $REASONING_NOTES)"
elif [[ -n "$REASONING_NOTES" ]]; then
    # No effort set, but a sampling-family note fired (Qwen3/MiniMax/Nemotron).
    echo "  reasoning:    unset; $REASONING_NOTES"
else
    echo "  reasoning:    unset (no row in reasoning_defaults for this agent+model)"
fi
if [[ "$IS_SANITY_AGENT" -eq 0 ]]; then
    echo "  output_cap:   $OUTPUT_CAP ($OUTPUT_CAP_NOTES)"
fi
if [[ "$AGENT" == "claude-code" ]]; then
    echo "  disallowed_tools: ${DISALLOWED_TOOLS:-<none> (DISABLE_PLAN_MODE=0)}"
fi
echo "  output_dir:   $OUTPUT_DIR"
echo "  n_concurrent: $N_CONCURRENT"
[[ "$N_TASKS" -gt 0 ]] && echo "  n_tasks:      $N_TASKS"
[[ -n "$TASK_NAME" ]] && echo "  task_name:    $TASK_NAME"
[[ -n "$EXCLUDE_TASK_NAME" ]] && echo "  exclude:      $EXCLUDE_TASK_NAME"
echo "  log:          $LOG"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    # Redact API keys. Docs recommend dry-run for verification and
    # people paste output into MRs / Slack; leaking raw gateway keys
    # from that output is a real risk. Real launch path below
    # (nohup ${CMD[@]}) is unaffected — keys go into the container
    # via --agent-env with their original values.
    printf '%q ' "${CMD[@]}" | sed -E 's/(_API_KEY=)[^ ]+/\1<redacted>/g'
    echo
    exit 0
fi

# Write the per-job reproducibility manifest BEFORE spawning harbor so
# it's available even if the run crashes early. Harbor writes its own
# result.json next to this; the two files are the full reproduction
# package. Schema + CLI:
# src/craft_taskgen/baselines/run_manifest.py
MANIFEST_DIR="$OUTPUT_DIR/$JOB_NAME"
mkdir -p "$MANIFEST_DIR"
MANIFEST_PATH="$MANIFEST_DIR/run_manifest.json"

# When --backend vllm, also probe /v1/models so the manifest captures
# the served model name, root (HF path), and max_model_len.
MANIFEST_VLLM_ARGS=()
if [[ "$BACKEND" == "vllm" ]]; then
    MANIFEST_VLLM_ARGS+=(--vllm-probe --vllm-api-key "${VLLM_API_KEY:-EMPTY}")
fi

# Manifest base_url: the endpoint THIS agent actually talks to. vllm
# backend goes to VLLM_BASE_URL; claude-code talks to ANTHROPIC_BASE_URL;
# codex and opencode talk to OPENAI_BASE_URL (even through the NVIDIA
# gateway, since that's the route their SDKs hit). Using a single
# priority-ordered fallback would mis-record codex's endpoint as
# ANTHROPIC_BASE_URL whenever both are set.
if [[ "$BACKEND" == "vllm" ]]; then
    MANIFEST_BASE_URL="${VLLM_BASE_URL:-}"
elif [[ "$AGENT" == "claude-code" ]]; then
    MANIFEST_BASE_URL="${ANTHROPIC_BASE_URL:-}"
else
    MANIFEST_BASE_URL="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
fi

uv run python -m craft_taskgen.baselines.run_manifest \
    --output          "$MANIFEST_PATH" \
    --tasks-dir       "$TASKS_DIR" \
    --agent           "$AGENT" \
    --agent-version   "$AGENT_VERSION" \
    --model           "$MODEL" \
    --backend         "$BACKEND" \
    --base-url        "$MANIFEST_BASE_URL" \
    --effort          "$EFFORT" \
    --reasoning-source "$REASONING_SOURCE" \
    --reasoning-notes "$REASONING_NOTES" \
    --output-cap      "$OUTPUT_CAP" \
    --output-cap-applied "$OUTPUT_CAP_NOTES" \
    --n-tasks         "$N_TASKS" \
    --n-concurrent    "$N_CONCURRENT" \
    --task-name       "$TASK_NAME" \
    --exclude-task-name "$EXCLUDE_TASK_NAME" \
    --launcher-argv   "$0 $ORIGINAL_ARGV_QUOTED" \
    --extra           'determinism.PYTHONHASHSEED="0"' \
    --extra           "determinism.LC_ALL=C.UTF-8" \
    --extra           "harness.max_turns=${MAX_TURNS:-null}" \
    --extra           "compaction.claude_code_pct_override=$([[ "$AGENT" == "claude-code" ]] && echo 95 || echo null)" \
    --extra           "compaction.opencode_auto=$([[ "$AGENT" == "opencode" ]] && echo false || echo null)" \
    --extra           "compaction.codex_auto_compact_token_limit_note=$([[ "$AGENT" == "codex" ]] && echo '\"uncapped (openai/codex#4138)\"' || echo null)" \
    --extra           "agent.harbor_model_arg=$(
        if [[ -n "$HARBOR_MODEL" ]]; then printf '"%s"' "$HARBOR_MODEL"; else echo null; fi
    )" \
    --extra           "agent.disallowed_tools=$(
        if [[ -n "$DISALLOWED_TOOLS" ]]; then printf '"%s"' "$DISALLOWED_TOOLS"; else echo null; fi
    )" \
    "${MANIFEST_VLLM_ARGS[@]}"
echo "  manifest:     $MANIFEST_PATH"
echo ""

# Wrap harbor so the manifest's outcomes field gets flipped from
# "predicted" to "present"/"missing" after harbor exits. Without this,
# outcomes.harbor_result_json points at a file that may never have
# existed (harbor crashed before writing result.json) and the reader
# has no way to tell from the manifest alone.
nohup bash -c '
    "$@"
    rc=$?
    uv run python -m craft_taskgen.baselines.run_manifest \
        --finalize "'"$MANIFEST_PATH"'" \
        --harbor-rc "$rc" || true
    exit "$rc"
' _ "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
disown $PID 2>/dev/null || true

echo "Started PID $PID"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG"
echo ""
echo "Stop with:"
echo "  kill $PID"
