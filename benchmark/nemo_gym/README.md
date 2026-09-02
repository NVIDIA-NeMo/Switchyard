<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compare Switchyard routes with NeMo Gym

This example evaluates the same MMLU-Redux tasks through two routes in [`routes.toml`](routes.toml):

- `strong-only` always uses the strong target.
- `policy-model` uses the efficient model as a classifier, then routes to the efficient or strong
  target.

Gym owns the tasks, verifier, rewards, and rollout capture. Switchyard owns routing, model calls,
and routing statistics. Gym's `switchyard_model` adapter joins them over HTTP, so neither project
imports the other's core library. The script starts a fresh server from this Switchyard checkout
for each condition to isolate its statistics.

## Prerequisites

Install Python 3.13.14 or newer, Rust 1.96.1, Cargo, `curl`, Git, and `uv`. You also
need credentials for the model endpoints configured in [`routes.toml`](routes.toml). The bundled
configuration uses two NVIDIA-hosted models and reads `NVIDIA_API_KEY`; create a key on the
[NVIDIA API key page](https://build.nvidia.com/settings/api-keys). Switchyard can use another
supported provider or compatible endpoint by changing the LLM client and targets in the TOML.

Use this clean, pinned Gym checkout:

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git /path/to/Gym
git -C /path/to/Gym checkout e044a8ca795ece2c69b053d30c0a8dea7fa3b9f3
cd /path/to/Gym
uv sync --frozen --no-dev
```

## Run

From the Switchyard repository:

```bash
export NVIDIA_API_KEY="nvapi-..."
export GYM_DIR=/path/to/Gym
bash benchmark/nemo_gym/run.sh
```

To use another provider, copy `routes.toml`, retain the `strong-only` and `policy-model` route IDs,
and update its LLM client, targets, and `api_key_env`. Export each credential named by the
deployment, then point the runner at that TOML:

```bash
export OPENAI_API_KEY="..."
export GYM_DIR=/path/to/Gym
SWITCHYARD_CONFIG=/path/to/routes.toml bash benchmark/nemo_gym/run.sh
```

Switchyard validates the `api_key_env` entries when it loads the deployment. An unauthenticated
endpoint does not need a credential variable.

Run `bash benchmark/nemo_gym/run.sh --help` to see the optional environment overrides.

The default run evaluates five tasks and writes a timestamped directory under
`benchmark/nemo_gym/results/`. It is a workflow smoke test, not a benchmark result. The first run
also downloads and prepares the dataset. Gym starts its serving environment for each condition.
For a larger workflow check:

```bash
LIMIT=100 REPEATS=3 CONCURRENCY=4 RESULTS_DIR=/tmp/routing-eval \
  bash benchmark/nemo_gym/run.sh
```

`LIMIT` takes the first tasks in Gym's prepared file, so a small limit is not a representative
sample. The example above can make roughly 600 answer calls and 300 classifier calls before
retries. Use a recorded stratified subset or the full benchmark for representative results. The
script refuses to overwrite a result directory.

## Read the result

`comparison.json` pairs completed rollouts by task and repeat index and verifies identical inputs.
It reports:

- mean reward and routed-versus-baseline wins, ties, and losses;
- missing and unpaired completions, so failed tasks are not silently scored or discarded;
- paired answer-model tokens and endpoint latency from Gym;
- classifier tokens, answer and classifier latency, routing overhead, model totals, and
  classifier fail-open counts from Switchyard.

Gym's endpoint latency covers the whole routed request. Switchyard routing overhead includes the
classifier call, so routing overhead and classifier latency overlap and must not be added together.
Answer and classifier tokens remain separate. Switchyard totals are condition-wide, while quality
and answer usage are paired only over tasks completed by both conditions.

The result directory contains the following artifacts. Gym may also write its best-effort
`switchyard-stats.json` wrapper when its shutdown hook completes:

| Artifact | Meaning |
|---|---|
| `comparison.json` | Paired quality and usage comparison for both conditions. |
| `<condition>/rollouts.jsonl` | Completed Gym rollouts and rewards. |
| `<condition>/rollouts_materialized_inputs.jsonl` | Exact task/repeat inputs used by Gym. |
| `<condition>/rollouts_failures.jsonl` | Rollouts that failed before producing a scored result. |
| `<condition>/rollouts_aggregate_metrics.json` | Gym's aggregate benchmark metrics. |
| `<condition>/switchyard-condition.json` | Route and attached-proxy provenance written by Gym. |
| `<condition>/switchyard-stats-raw.json` | Raw `/v1/stats` captured while the proxy is alive. |
| `<condition>/switchyard-stats.json` | Best-effort Gym wrapper around `/v1/stats`. |
| `<condition>/switchyard-metrics.prom` | Prometheus metrics, including classifier fail-open reasons. |
| `<condition>/model-calls/` | Per-rollout model-call captures, including the served model. |
| `<condition>/routes.toml` | Exact Switchyard deployment copied for the condition. |
| `<condition>/switchyard.log` | Switchyard server output for diagnosis. |

Gym excludes failure-sidecar rows from its reward calculation. `comparison.json` reports those
rollouts as missing or unpaired instead of treating them as zero-reward answers.

Keep the materialized inputs because Gym's MMLU-Redux loader does not pin a Hugging Face dataset
revision. The workflow and evaluated inputs are reproducible; hosted model outputs, token counts,
and latency can still change between runs. Run from a clean Switchyard checkout when you need to
reproduce the exact server build. Dirty builds are labeled `-dirty`, but the source diff is not
archived with the result.

See the
[NeMo Gym Switchyard model-server documentation](https://docs.nvidia.com/nemo/gym/main/model-server/switchyard/)
for other benchmarks and hosted mode, and the
[LLM classifier guide](../../docs/routing_algorithms/llm_classifier_routing.md) for the routing
policy used here.
