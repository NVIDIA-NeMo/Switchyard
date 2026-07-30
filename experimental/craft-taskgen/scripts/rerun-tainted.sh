#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# rerun-tainted.sh — re-run tainted Harbor trials under a network firewall.
#
# Inputs: one or more Harbor run-group dirs (the parent of trial dirs, e.g.
# `data/extracted/jobs/v2-codex-gpt55/baseline-codex-craft-taskgen-v2-...`).
#
# What it does, per run-group dir:
#   1. Runs scripts/check_integrity.py to find trials that fetched upstream
#      source. Tainted trials get archived OUT of the run-group dir to
#      `archive/tainted/<original-relative-path>/` so the scorer doesn't
#      see them. The original directory layout is preserved under the
#      archive root for forensic inspection.
#   2. Reads run_manifest.json to recover the original launcher_argv
#      (tasks-dir, agent, model, backend, output-dir, ...).
#   3. Enables the docker firewall (scripts/docker-firewall.sh), launches
#      run-baselines.sh restricted to the tainted task IDs, waits for
#      harbor to finish, disables the firewall (in an EXIT trap).
#   4. Re-runs check_integrity.py on the new run-group dir. Any rerun
#      trial that is ALSO tainted also gets archived. One pass only —
#      no retry loop.
#   5. Prints a summary.
#
# Usage:
#   sudo ./scripts/rerun-tainted.sh <run-group-dir>...
#   ./scripts/rerun-tainted.sh --dry-run <run-group-dir>...
#
# Flags:
#   --dry-run                Print planned steps; do not modify anything
#                            and do not toggle the firewall. Safe on macOS.
#   --firewall-script PATH   Override default scripts/docker-firewall.sh.
#   -h, --help               This message.
#
# Env vars:
#   ARCHIVE_ROOT             Override the archive destination. Default:
#                            `archive/tainted` (relative to cwd). Tainted
#                            trial dirs are mv'd into this tree, preserving
#                            the original relative path.
#   CODEX_BINARY_PATH        Required when reruning codex tainted trials.
#                            Path to a tarball produced by
#                            scripts/build-codex-prebake.sh.
#
# Sudo policy: the wrapper does NOT call sudo internally. Operator wraps
# the entire invocation with sudo so all child commands run as root. If
# invoked without sudo (and not --dry-run), the wrapper aborts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
FIREWALL_SCRIPT="$SCRIPT_DIR/docker-firewall.sh"
RUN_GROUP_DIRS=()

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=1; shift ;;
        --firewall-script) FIREWALL_SCRIPT="$2"; shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        --)                shift; break ;;
        -*)                echo "ERROR: unknown flag: $1" >&2; usage >&2; exit 2 ;;
        *)                 RUN_GROUP_DIRS+=("$1"); shift ;;
    esac
done

if [[ ${#RUN_GROUP_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no run-group dirs given" >&2
    usage >&2
    exit 2
fi

# Sudo gate: must be root unless dry-run
if [[ "$DRY_RUN" -eq 0 && "$EUID" -ne 0 ]]; then
    echo "ERROR: re-run as root, e.g.:" >&2
    echo "  sudo $0 ${RUN_GROUP_DIRS[*]}" >&2
    echo "(or pass --dry-run for a no-op preview)" >&2
    exit 2
fi

# uv is typically installed in $SUDO_USER's ~/.local/bin, which sudo's
# default secure_path strips from PATH. Find and prepend it so child
# commands (uv run python ...) work without `sudo -E PATH="$PATH" ...`.
if ! command -v uv >/dev/null 2>&1; then
    candidate_dirs=()
    [[ -n "${SUDO_USER:-}" ]] && candidate_dirs+=("$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6)/.local/bin")
    candidate_dirs+=("$HOME/.local/bin" "/usr/local/bin" "/opt/uv/bin")
    for d in "${candidate_dirs[@]}"; do
        if [[ -n "$d" && -x "$d/uv" ]]; then
            export PATH="$d:$PATH"
            break
        fi
    done
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found on PATH. Common fix:" >&2
    echo "  sudo env PATH=\"\$HOME/.local/bin:\$PATH\" $0 ${RUN_GROUP_DIRS[*]}" >&2
    exit 2
fi

# Firewall script must exist (or be valid in dry-run, but dry-run still
# wants to print the path).
if [[ ! -f "$FIREWALL_SCRIPT" ]]; then
    echo "ERROR: firewall script not found: $FIREWALL_SCRIPT" >&2
    echo "Use --firewall-script to point at an alternate path." >&2
    exit 2
fi

# Validate run-group dirs and ensure each has a run_manifest.json
for d in "${RUN_GROUP_DIRS[@]}"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: not a directory: $d" >&2
        exit 2
    fi
    if [[ ! -f "$d/run_manifest.json" ]]; then
        echo "ERROR: $d has no run_manifest.json (cannot recover launcher args)" >&2
        exit 2
    fi
done

echo "rerun-tainted.sh"
echo "  dry_run:           $DRY_RUN"
echo "  firewall_script:   $FIREWALL_SCRIPT"
echo "  run_group_dirs:"
for d in "${RUN_GROUP_DIRS[@]}"; do
    echo "    $d"
done
echo ""

# ---------------------------------------------------------------------------
# Stage 1: integrity scan and archive
# ---------------------------------------------------------------------------

CSV_TMP="$(mktemp -t rerun-tainted-csv.XXXXXX)"
trap 'rm -f "$CSV_TMP"' EXIT

echo "[1/4] Running integrity check..."
if ! uv run python "$SCRIPT_DIR/check_integrity.py" \
        "${RUN_GROUP_DIRS[@]}" \
        --csv "$CSV_TMP" >/dev/null; then
    echo "ERROR: check_integrity.py failed" >&2
    exit 1
fi

# Parse the tainted rows and group by run-group dir
declare -A TAINTED_TASKS=()    # run_group_dir -> "task1 task2 task3"
declare -A TAINTED_TRIALS=()   # run_group_dir -> "trial_dir1 trial_dir2 ..."

while IFS=, read -r root trial_dir task agent resolved ws wf upstream_url_count any_web fetched_upstream first_url; do
    [[ "$fetched_upstream" == "1" ]] || continue
    # Find which input run-group dir this trial belongs to.
    # trial_dir is absolute or relative path; match against our inputs.
    for rg in "${RUN_GROUP_DIRS[@]}"; do
        rg_abs="$(cd "$rg" && pwd)"
        trial_abs="$(cd "$(dirname "$trial_dir")" 2>/dev/null && pwd || true)/$(basename "$trial_dir")"
        if [[ "$trial_abs" == "$rg_abs"/* ]]; then
            TAINTED_TASKS[$rg]+="$task "
            TAINTED_TRIALS[$rg]+="$trial_dir "
            break
        fi
    done
done < <(tail -n +2 "$CSV_TMP")  # skip header

n_tainted_total=0
for rg in "${!TAINTED_TRIALS[@]}"; do
    # Count words
    n=$(echo "${TAINTED_TRIALS[$rg]}" | wc -w | tr -d ' ')
    n_tainted_total=$((n_tainted_total + n))
done

if [[ $n_tainted_total -eq 0 ]]; then
    echo "  No tainted trials found across input run-group dirs. Done."
    exit 0
fi

echo "  Found $n_tainted_total tainted trial(s) across ${#TAINTED_TRIALS[@]} run-group(s)."

# Codex reruns require CODEX_BINARY_PATH to be set on the host (so the
# patched codex install template can extract the pre-baked tarball
# instead of bootstrapping NVM, which is blocked by the firewall). Warn
# loudly if any of the tainted trials are codex and CODEX_BINARY_PATH is
# missing — the rerun would otherwise fail at install time.
declare -i n_codex_pending=0
for rg in "${!TAINTED_TRIALS[@]}"; do
    manifest="$rg/run_manifest.json"
    agent_name=$(uv run python - "$manifest" <<'PY' 2>/dev/null || true
import json, sys
m = json.load(open(sys.argv[1]))
print(m.get('agent', {}).get('name') or '')
PY
)
    if [[ "$agent_name" == "codex" ]]; then
        n_in_rg=$(echo "${TAINTED_TRIALS[$rg]}" | wc -w | tr -d ' ')
        n_codex_pending=$((n_codex_pending + n_in_rg))
    fi
done
if [[ $n_codex_pending -gt 0 && -z "${CODEX_BINARY_PATH:-}" ]]; then
    echo "ERROR: $n_codex_pending of the tainted trial(s) are from codex," >&2
    echo "       but CODEX_BINARY_PATH is not set. Codex install under firewall" >&2
    echo "       requires a pre-baked node + codex tarball. Build one with:" >&2
    echo "         bash scripts/build-codex-prebake.sh /tmp/codex-prebake.tar.gz" >&2
    echo "       Then re-invoke as:" >&2
    echo "         sudo -E CODEX_BINARY_PATH=/tmp/codex-prebake.tar.gz $0 $*" >&2
    exit 2
fi
if [[ $n_codex_pending -gt 0 ]]; then
    echo "  CODEX_BINARY_PATH=${CODEX_BINARY_PATH} (will be uploaded for codex reruns)"
fi
echo ""

# ---------------------------------------------------------------------------
# Stage 2: archive originals
# ---------------------------------------------------------------------------
#
# Archives go OUTSIDE the input run-group dir so the scorer
# (scripts/summarize_baseline.py) doesn't need any awareness of them. The
# archive mirrors the original path under a top-level "archive/tainted/"
# tree, e.g.:
#     baselines/integrity-test/baseline-haiku-XXX/t2v3-task-yyy/
#  -> archive/tainted/baselines/integrity-test/baseline-haiku-XXX/t2v3-task-yyy/
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/tainted}"
echo "[2/4] Archiving tainted trial dirs to $ARCHIVE_ROOT/..."
for rg in "${!TAINTED_TRIALS[@]}"; do
    for trial in ${TAINTED_TRIALS[$rg]}; do
        [[ -n "$trial" ]] || continue
        if [[ ! -d "$trial" ]]; then
            echo "  WARN: trial dir gone: $trial" >&2
            continue
        fi
        # Strip leading ./ if present, normalise to relative path
        rel="${trial#./}"
        archived="$ARCHIVE_ROOT/$rel"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "  (dry-run) would mv $trial -> $archived"
        else
            if [[ -e "$archived" ]]; then
                echo "  WARN: archive target exists, skipping: $archived" >&2
                continue
            fi
            mkdir -p "$(dirname "$archived")"
            mv "$trial" "$archived"
            echo "  archived: $trial -> $archived"
        fi
    done
done
echo ""

# ---------------------------------------------------------------------------
# Stage 3: enable firewall and rerun
# ---------------------------------------------------------------------------

# Track the new run-group dirs that get created so we can integrity-check
# them at the end.
NEW_RUN_GROUPS=()

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[3/4] (dry-run) skipping firewall enable + rerun"
    for rg in "${!TAINTED_TASKS[@]}"; do
        tasks="${TAINTED_TASKS[$rg]}"
        echo "  would rerun: $rg with tasks: $tasks"
    done
    echo ""
    echo "[4/4] (dry-run) skipping post-rerun integrity check"
    echo ""
    echo "Dry-run complete. No filesystem changes were made beyond the"
    echo "tempfile $CSV_TMP (cleaned up on exit)."
    exit 0
fi

# Real run: enable firewall, register cleanup, rerun.
#
# Disable codex's server-side web_search tool for the rerun. Codex
# executes web_search at the inference gateway, NOT in the trial
# container, so an iptables firewall in the container can't block it.
# The patched harbor codex.py reads this env var on the host and adds
# `-c web_search="disabled"` to the codex CLI invocation (a runtime
# override that bypasses TOML config nesting). No-op for opencode and
# claude-code.
export CODEX_DISABLE_WEB_SEARCH=1

# Suppress .pyc generation while harbor runs under sudo. Without this,
# every harbor import path under .venv writes root-owned __pycache__
# entries that block subsequent non-root harbor reinstalls / patch
# resets. Harbor itself doesn't need bytecode caching for one-shot runs.
export PYTHONDONTWRITEBYTECODE=1

echo "[3/4] Enabling firewall..."
bash "$FIREWALL_SCRIPT" enable

cleanup_firewall() {
    echo ""
    echo "Cleanup: disabling firewall..."
    bash "$FIREWALL_SCRIPT" disable || echo "  WARN: firewall disable failed; check iptables manually" >&2
}

cleanup_docker_ownership() {
    # When sudo -E preserves HOME, docker buildx writes config to the
    # real user's $HOME/.docker as root. That breaks subsequent
    # `docker compose build` invocations by the real user. Chown back.
    if [[ -n "${SUDO_USER:-}" && -d "$HOME/.docker" ]]; then
        if chown -R "$SUDO_USER" "$HOME/.docker" 2>/dev/null; then
            echo "Cleanup: restored $HOME/.docker ownership to $SUDO_USER"
        fi
    fi
}

cleanup_pycache_ownership() {
    # PYTHONDONTWRITEBYTECODE=1 (set above) should prevent root-owned .pyc
    # files in .venv. As a defensive measure (in case some sub-process
    # unsets it), reclaim any root-owned __pycache__ entries that landed
    # under the project venv during this run.
    if [[ -n "${SUDO_USER:-}" && -d "$REPO_ROOT/.venv" ]]; then
        if find "$REPO_ROOT/.venv" -name __pycache__ -user root -exec chown -R "$SUDO_USER" {} + 2>/dev/null; then
            :
        fi
    fi
}

trap 'cleanup_firewall; cleanup_docker_ownership; cleanup_pycache_ownership; rm -f "$CSV_TMP"' EXIT

# Per run-group, recover launcher_argv and invoke run-baselines.sh.
# run-baselines.sh nohups harbor in the background; we capture the PID it
# prints and wait for it to exit before moving on.
for rg in "${!TAINTED_TASKS[@]}"; do
    manifest="$rg/run_manifest.json"
    echo ""
    echo "  Recovering launcher_argv from $manifest..."

    # Convert tasks list into a single glob pattern via task_name OR-glob;
    # harbor's --task-name accepts a glob, not a list. We launch one
    # baseline run per task to keep this simple and traceable.
    tasks_arr=( ${TAINTED_TASKS[$rg]} )
    # Deduplicate
    mapfile -t tasks_uniq < <(printf '%s\n' "${tasks_arr[@]}" | awk 'NF' | sort -u)

    for task in "${tasks_uniq[@]}"; do
        echo ""
        echo "  Rerunning task: $task"

        # Reconstruct args from launcher_argv. We need: --tasks-dir,
        # --agent, --model, --backend, --output-dir, --n-concurrent, etc.
        # We override --task-name to scope to one task and drop --n-tasks
        # if present. Output-dir we keep so the new run lands under the
        # same parent as the original (a fresh job-name dir will be picked
        # by run-baselines.sh).
        mapfile -t argv_args < <(uv run python - "$manifest" "$task" <<'PY'
import json, sys
manifest_path, task_name = sys.argv[1], sys.argv[2]
m = json.load(open(manifest_path))
argv = m.get('run', {}).get('launcher_argv') or []
# argv[0] is the script path; we'll drop it.
out = []
i = 1
skip_next = False
while i < len(argv):
    a = argv[i]
    if skip_next:
        skip_next = False
        i += 1
        continue
    if a in ('--task-name','--n-tasks','--exclude-task-name'):
        skip_next = True   # skip this and its value
        i += 1
        continue
    out.append(a)
    i += 1
out += ['--task-name', task_name]
for a in out:
    print(a)
PY
)

        echo "    invoking: scripts/run-baselines.sh ${argv_args[*]}"

        # Capture the PID printed by run-baselines.sh so we can wait.
        rb_log="$(mktemp -t rb-launch.XXXXXX)"
        if ! "$SCRIPT_DIR/run-baselines.sh" "${argv_args[@]}" 2>&1 | tee "$rb_log"; then
            echo "    WARN: run-baselines.sh launch failed for task $task" >&2
            rm -f "$rb_log"
            continue
        fi

        # Extract "Started PID NNNN" from output
        pid="$(grep -E '^Started PID ' "$rb_log" | awk '{print $3}' | tail -n1)"
        rm -f "$rb_log"
        if [[ -z "$pid" ]]; then
            echo "    WARN: could not parse PID from run-baselines.sh output for $task" >&2
            continue
        fi
        echo "    waiting for harbor PID $pid to finish..."
        # Poll until process exits
        while kill -0 "$pid" 2>/dev/null; do sleep 5; done
        echo "    harbor PID $pid finished."

        # Discover the new run-group dir. run-baselines.sh creates a
        # subdir under --output-dir matching `baseline-<agent>-<source>-<ts>`.
        # We'll match anything newer than 1 minute under the output-dir.
        out_dir=""
        for ((i=0; i<${#argv_args[@]}; i++)); do
            if [[ "${argv_args[$i]}" == "--output-dir" ]]; then
                out_dir="${argv_args[$((i+1))]}"
                break
            fi
        done
        if [[ -n "$out_dir" && -d "$out_dir" ]]; then
            # newest dir in $out_dir
            new_rg="$(ls -1dt "$out_dir"/*/ 2>/dev/null | head -n1)"
            if [[ -n "$new_rg" ]]; then
                NEW_RUN_GROUPS+=("$new_rg")
                echo "    new run-group: $new_rg"
            fi
        fi
    done
done

# ---------------------------------------------------------------------------
# Stage 4: integrity check on rerun output, archive any still-tainted
# ---------------------------------------------------------------------------
if [[ ${#NEW_RUN_GROUPS[@]} -gt 0 ]]; then
    echo ""
    echo "[4/4] Re-running integrity check on new run-group(s)..."
    if ! uv run python "$SCRIPT_DIR/check_integrity.py" \
            "${NEW_RUN_GROUPS[@]}" \
            --csv "$CSV_TMP" >/dev/null; then
        echo "  WARN: integrity recheck failed" >&2
    fi

    # Archive a rerun only if it BOTH attempted upstream fetch AND passed.
    # Tainted-but-failed reruns are kept: that's the firewall doing its job —
    # the agent tried to cheat, got blocked at the network layer, and failed
    # honestly. Tainted-and-passed reruns mean the firewall didn't block (or
    # the agent succeeded via memorization), so we drop them.
    n_still=0
    n_blocked=0
    while IFS=, read -r root trial_dir task agent resolved ws wf upstream_url_count any_web fetched_upstream first_url; do
        [[ "$fetched_upstream" == "1" ]] || continue
        [[ -d "$trial_dir" ]] || continue
        if [[ "$resolved" == "1" ]]; then
            rel="${trial_dir#./}"
            archived="$ARCHIVE_ROOT/$rel"
            mkdir -p "$(dirname "$archived")"
            mv "$trial_dir" "$archived"
            echo "  rerun cheated AND passed; archived: $trial_dir -> $archived"
            n_still=$((n_still+1))
        else
            echo "  rerun attempted upstream fetch but failed (firewall worked): $trial_dir (kept)"
            n_blocked=$((n_blocked+1))
        fi
    done < <(tail -n +2 "$CSV_TMP")
    echo "  Reruns cheated and passed (archived): $n_still"
    echo "  Reruns attempted-but-blocked-by-firewall (kept):  $n_blocked"
else
    echo ""
    echo "[4/4] No new run-group dirs found; skipping recheck."
fi

echo ""
echo "Done. Summary:"
echo "  tainted originals archived:                      $n_tainted_total"
echo "  reruns launched (run-groups):                    ${#NEW_RUN_GROUPS[@]}"
echo "  reruns that cheated AND passed (archived):       ${n_still:-0}"
echo "  reruns that attempted but were blocked (kept):   ${n_blocked:-0}"
echo ""
echo "Inspect: $(printf '%s ' "${NEW_RUN_GROUPS[@]}")"
