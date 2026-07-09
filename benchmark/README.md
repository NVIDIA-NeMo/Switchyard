<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor Benchmarks

Use this guide to run Harbor Terminal-Bench Lite from a fresh Switchyard clone.
It covers the two smoke paths most people need first:

- **Direct upstream:** Harbor calls the provider directly. Switchyard is disabled.
- **Switchyard routing:** Harbor calls Switchyard, and Switchyard routes across two model tiers.

Both paths use the same generated dataset, task proxy, pinned agent versions, and run artifact
layout. Passing `--routing-profiles` starts Switchyard; omitting it disables Switchyard and points
Harbor directly at the upstream provider.

## Prerequisites

From the repo root:

```bash
uv sync
```

You also need Docker with Compose support, because baseline runs launch task containers and use the
generated benchmark proxy topology. Runs with `--routing-profiles` also start Switchyard inside
Docker.

Harbor is installed as a dev dependency. Check that the CLI resolves from the uv environment:

```bash
uv run --no-sync harbor --help
```

## Configure Your Provider

The checked-in smoke commands use OpenRouter's OpenAI-compatible endpoint by default:

```bash
export OPENROUTER_API_KEY="..."
```

To use another OpenAI-compatible provider, either export a generic upstream key:

```bash
export UPSTREAM_API_KEY="..."
export UPSTREAM_BASE_URL="https://provider.example/v1"
```

or pass the provider-specific key variable explicitly:

```bash
bash benchmark/run-baseline.sh \
  --upstream-base-url https://provider.example/v1 \
  --upstream-api-key-env PROVIDER_API_KEY \
  ...
```

Route-profile YAML files expand `${OPENROUTER_API_KEY}` by default. For another provider, copy a
profile and update its `api_key`, `base_url`, and model ids to match that provider.

## One-Time Setup

`run-baseline.sh` has a blanket preflight check for the current patch file. It reverse-checks the
exact diff against the installed Harbor tree, so stale or partial patch applications fail before
launching Harbor. Apply the patch to the current uv environment:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
HARBOR_SITE="$(
  cd "$REPO_ROOT"
  uv run --no-sync python - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"

cd "$HARBOR_SITE"
patch -p1 < "$REPO_ROOT/benchmark/patches/harbor-agent-patches.diff"
cd "$REPO_ROOT"
```

Reapply this after recreating the virtualenv, reinstalling Harbor, or running a forced dependency
reinstall.

The generated dataset is local build output and is not committed. This command downloads and exports
`openthoughts-tblite@2.0`, prebakes pinned agent versions into each task image, injects the
benchmark proxy, and writes `switchyard_dataset_manifest.json`:

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py --overwrite
```

Default output:

```text
benchmark/datasets/openthoughts-tblite-closed-book
```

To reuse an already exported Harbor dataset instead of downloading again:

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dir /path/to/exported/openthoughts-tblite \
  --overwrite
```

The pinned versions live in `benchmark/agent-versions.env`. To prepare a different Harbor dataset,
see [Benchmark Datasets](DATASETS.md).

Terminal-Bench 2.0 is supported through the same generated local proxy dataset path. The
TB2 export keeps model/tool egress on the closed-book path while allowlisting the package and data
sources required by the official Oracle solutions.

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dataset terminal-bench/terminal-bench-2 \
  --output-dir benchmark/datasets/terminal-bench-2-closed-book \
  --overwrite
```

Terminal-Bench 2.1 (the verified iteration of 2.0) is supported the same way and shares the 2.0
Oracle allowlist:

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dataset terminal-bench/terminal-bench-2-1 \
  --output-dir benchmark/datasets/terminal-bench-2-1-closed-book \
  --overwrite
```

SWE-Bench Pro is supported with the Harbor dataset `cais/swebenchpro`. The generated dataset uses
the same pinned-agent and closed-book proxy path without opening dataset-specific agent egress.

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dataset cais/swebenchpro \
  --output-dir benchmark/datasets/swebenchpro-closed-book \
  --overwrite
```

## Run Without Switchyard

Omit `--routing-profiles` to fully disable Switchyard. The runner still creates the benchmark
Docker network for the generated proxy sidecar, but Harbor sends model calls straight to
`${UPSTREAM_BASE_URL:-https://openrouter.ai/api/v1}` using `OPENROUTER_API_KEY` by default:

```bash
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --model openai/gpt-5.5 \
  --agent codex \
  --reasoning-effort xhigh \
  --n-tasks 1 \
  --n-concurrent 1 \
  --max-retries 0
```

For another OpenAI-compatible upstream, pass `--upstream-base-url` and
`--upstream-api-key-env`. Claude Code direct runs require an Anthropic-compatible upstream because
Switchyard translation is disabled.

## Run With Switchyard Routing

Pass `--routing-profiles` to start Switchyard and route Harbor traffic through it. This smoke test
uses `benchmark/routing-profiles/tb-lite-llm-classifier-opus-kimi-gemini.yaml`, an
LLM-as-classifier profile for coding-agent tasks:

```bash
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --routing-profiles benchmark/routing-profiles/tb-lite-llm-classifier-opus-kimi-gemini.yaml \
  --model switchyard \
  --agent codex \
  --reasoning-effort xhigh \
  --n-tasks 1 \
  --n-concurrent 1 \
  --max-retries 0
```

Use the route key from the YAML as `--model`. In this profile, the Gemini classifier selects the
target tier for each request, then Switchyard routes to one of:

- strong: `anthropic/claude-opus-4.7`
- weak: `moonshotai/kimi-k2.7-code`

Classifier model: `google/gemini-3.5-flash`.

To smoke-test a single-model Switchyard path instead, use one of:

```text
benchmark/routing-profiles/tb-lite-single-gpt-5-5.yaml
benchmark/routing-profiles/tb-lite-single-opus-4-7.yaml
```

By default, the runner starts in the background and prints the PID, log path, and kill command.

## Book Modes

Both book modes use the same generated `--harbor-path` dataset, prebaked agent images, and proxy
sidecar topology. Switchyard is Dockerized only when `--routing-profiles` is provided.

Closed-book mode is the default:

```bash
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --routing-profiles benchmark/routing-profiles/tb-lite-llm-classifier-opus-kimi-gemini.yaml \
  --model switchyard \
  --agent codex \
  --n-tasks 1
```

In closed-book mode, the proxy allows Switchyard/model traffic, blocks public cheat sources such as
`raw.githubusercontent.com`, strips hosted web/search/code tools from model API payloads, and adds
agent-specific web-disable settings where supported.

Open-book mode keeps the same proxy path but broadens egress:

```bash
bash benchmark/run-baseline.sh \
  --book-mode open \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --routing-profiles benchmark/routing-profiles/tb-lite-llm-classifier-opus-kimi-gemini.yaml \
  --model switchyard \
  --agent codex \
  --n-tasks 1
```

Use open-book mode only when the evaluation intentionally allows internet access. The manifest
records the mode, the local dataset digest, any copied routing profile, proxy metadata, upstream
base URL for direct runs, and agent version pins in both modes.

## Run A Full TB Lite Pass

After the smoke test succeeds, remove `--n-tasks 1`, raise concurrency to match your host and
provider quota, and let the runner use the background wrapper.

Direct upstream:

```bash
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --model openai/gpt-5.5 \
  --agent codex \
  --reasoning-effort xhigh \
  --n-concurrent 8 \
  --max-retries 2
```

Switchyard LLM-classifier routing:

```bash
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --routing-profiles benchmark/routing-profiles/tb-lite-llm-classifier-opus-kimi-gemini.yaml \
  --model switchyard \
  --agent codex \
  --reasoning-effort xhigh \
  --n-concurrent 8 \
  --max-retries 2
```

Tune `--n-concurrent` for your machine and provider quota. Use `--task-id`, `--task-list-file`, or
`--n-tasks` for subsets.

## Inspect A Run

Run directories are created under `benchmark/tb_runs/`. The most useful artifacts are:

```text
run_manifest.json
server.log
harbor.log
routing_stats_final.json
jobs/<job-name>/result.json
jobs/<job-name>/<task-id>/agent/trajectory.json
```

The manifest records the command, git state, Harbor patch provenance, local dataset digest, copied
routing profile when present, direct-upstream metadata when Switchyard is disabled, book-mode
settings, agent version pins, log paths, and final Harbor status. When the routing profile contains
deterministic LLM-classifier routes, `server.classifier_prompts` records each route's effective
prompt, prompt SHA-256, `max_request_chars`, and `recent_turn_window` for reproducibility. Direct
runs mark routing stats as `not-requested`.

## Docker Image Notes

Baseline runs build `switchyard-baseline:local` from `benchmark/switchyard-server.Dockerfile`.
The default is to rebuild before each run so the container matches the current checkout.

To reuse an already built image:

```bash
SWITCHYARD_DOCKER_BUILD=0 bash benchmark/run-baseline.sh ...
```

Only reuse the image when you know it already contains the current Switchyard code and Rust extension.

## Troubleshooting

If the runner reports that the current Harbor patch is not applied cleanly, recreate or reinstall the
uv environment and rerun the patch command from this README.

If port `4000` is busy, pass a different port:

```bash
bash benchmark/run-baseline.sh ... --port 4001
```

If the Docker reachability preflight fails, check Docker/Compose first. The preflight proves the
task container can reach Switchyard through the benchmark Docker network.
For local debugging only, it can be bypassed with `SWITCHYARD_CLOSED_BOOK_PREFLIGHT=0`.
