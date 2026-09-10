#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --job-name=swe-tw-selector
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -uo pipefail

root=${BENCHMARK_ROOT:?Set BENCHMARK_ROOT to a writable benchmark directory}
manifest=${MANIFEST:-"$root/manifest.txt"}
run_kind=${RUN_KIND:-matrix}
gym_dir=${GYM_DIR:?Set GYM_DIR to a NeMo Gym checkout}
credential=${INFERENCE_KEY_FILE:?Set INFERENCE_KEY_FILE to a mode-600 key file}
config=${CONFIG:-"$root/config.toml"}
compose="$gym_dir/responses_api_agents/harbor_agent/configs/docker-compose-clear-entrypoint.yaml"

read -r -a arms <<<"${ARMS:-direct-glm always-opus patched-escalation capability-selector}"
arm_count=${#arms[@]}
task_index=$((SLURM_ARRAY_TASK_ID / arm_count + 1))
arm_index=$((SLURM_ARRAY_TASK_ID % arm_count))
arm=${arms[$arm_index]}
row=$(sed -n "${task_index}p" "$manifest")
IFS='|' read -r dataset task_slug <<<"$row"
if [ -z "${dataset:-}" ] || [ -z "${task_slug:-}" ]; then
    echo "No manifest row $task_index for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

switchyard_bin=${SWITCHYARD_BIN:?Set SWITCHYARD_BIN to the server binary under test}
route=$arm
[ "$arm" = patched-escalation ] && route=escalation
[ "$arm" = capability-selector ] && route=capability
[ "$arm" = calibrated-escalation ] && route=calibrated-escalation

task="scale-ai/$task_slug"
run_dir="$root/$run_kind-$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID-$dataset-$task_slug-$arm"
case_dir="$run_dir/case"
mkdir -p "$case_dir" "$run_dir/tmp" "$run_dir/xdg-cache"

if [ ! -s "$credential" ] || [ "$(stat -c %a "$credential")" != 600 ]; then
    echo "A non-empty mode-600 inference credential is required." >&2
    exit 2
fi

export INFERENCE_API_KEY
INFERENCE_API_KEY=$(<"$credential")
export OPENAI_API_KEY="$INFERENCE_API_KEY"
export OPENAI_API_BASE=https://inference-api.nvidia.com/v1
export OPENAI_BASE_URL=https://inference-api.nvidia.com/v1
export EVAL_MODEL=azure/anthropic/claude-opus-4-8
export XDG_CACHE_HOME="$run_dir/xdg-cache"
export TMPDIR="$run_dir/tmp"
export PYTHONPATH="$gym_dir"

port=$((20000 + (SLURM_ARRAY_JOB_ID * 97 + SLURM_ARRAY_TASK_ID) % 30000))
session_id="$run_kind-$dataset-$task_slug-$arm-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"

"$switchyard_bin" \
    --config "$config" \
    --host 127.0.0.1 \
    --port "$port" \
    --routing-log-file "$case_dir/routing.jsonl" \
    >"$case_dir/switchyard.log" 2>&1 &
switchyard_pid=$!
cleanup() {
    kill "$switchyard_pid" 2>/dev/null || true
    wait "$switchyard_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null && break
    kill -0 "$switchyard_pid" 2>/dev/null || break
    sleep 1
done
if ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
    echo "switchyard_start_failed" > "$case_dir/run-status.txt"
    exit 3
fi

cd "$gym_dir"
set +e
uv run --with \
    'harbor @ git+https://github.com/laude-institute/harbor.git@527d50deb63a5d279e8c20593c18a2cbc7f61f9e' \
    harbor run \
    --agent responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym:Terminus2NemoGym \
    --model "switchyard/$route" \
    --ak "api_base=http://127.0.0.1:$port/v1" \
    --ak "session_id=$session_id" \
    --ak encode_terminus_tool_history=true \
    --ak interleaved_thinking=true \
    --ak enable_summarize=true \
    --ak collect_rollout_details=false \
    --ak 'trajectory_config={"raw_content":true}' \
    --ak 'model_info={"max_input_tokens":200000,"max_output_tokens":128000,"input_cost_per_token":0.0,"output_cost_per_token":0.0}' \
    --ak 'responses_create_params={"max_output_tokens":128000}' \
    --ak nemo_model_server_timeout_sec=900 \
    --dataset "scale-ai/swe-atlas-$dataset@1" \
    --include-task-name "$task" \
    --extra-docker-compose "$compose" \
    --n-concurrent 1 \
    --max-retries 0 \
    --agent-setup-timeout-multiplier 3 \
    --agent-timeout-multiplier 3 \
    --yes \
    --jobs-dir "$case_dir/jobs" \
    --job-name "$dataset-$task_slug-$arm" \
    --verifier-env "OPENAI_API_KEY=$OPENAI_API_KEY" \
    --verifier-env "OPENAI_API_BASE=$OPENAI_API_BASE" \
    --verifier-env "OPENAI_BASE_URL=$OPENAI_BASE_URL" \
    --verifier-env "EVAL_MODEL=$EVAL_MODEL" \
    >"$case_dir/harbor.log" 2>&1
harbor_status=$?
set -e

curl -fsS "http://127.0.0.1:$port/v1/stats" > "$case_dir/stats.json" || true
echo "$harbor_status" > "$case_dir/harbor-exit-code"

result=$(find "$case_dir/jobs" -mindepth 3 -maxdepth 3 -name result.json -type f -print -quit 2>/dev/null)
if [ -z "$result" ]; then
    echo "missing_result" > "$case_dir/run-status.txt"
    exit 4
fi

jq '{reward: (.verifier_result.rewards.reward // null), exception_info, agent_result}' \
    "$result" > "$case_dir/result-summary.json"
exception=$(jq -r '.exception_info // empty' "$result")
reward=$(jq -r '.verifier_result.rewards.reward // empty' "$result")
requests=$(jq -r '.total_requests // 0' "$case_dir/stats.json")
if [ "$harbor_status" -ne 0 ] || [ -n "$exception" ] || [ -z "$reward" ] || [ "$requests" -le 0 ]; then
    printf 'invalid harbor=%s exception=%s reward=%s requests=%s\n' \
        "$harbor_status" "${exception:-none}" "${reward:-missing}" "$requests" \
        > "$case_dir/run-status.txt"
    exit 5
fi

printf 'valid reward=%s requests=%s\n' "$reward" "$requests" > "$case_dir/run-status.txt"
echo "$dataset/$task_slug/$arm: reward=$reward requests=$requests"
