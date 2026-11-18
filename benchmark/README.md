<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor Benchmarks

For recorded results and their routing configs, see [Benchmarks](../docs/benchmarks.md).

Use this guide to run Harbor Terminal-Bench Lite from a fresh Switchyard clone.
It covers the two smoke paths most people need first:

- **Direct upstream:** Harbor calls the provider directly. Switchyard is disabled.
- **Switchyard routing:** Harbor calls Switchyard, and Switchyard routes across two model tiers.

Both paths use the same generated dataset, task proxy, pinned agent versions, and run artifact
layout. Passing `--server-config` starts the Rust server; omitting it disables Switchyard and points
Harbor directly at the upstream provider.

For a small automated MMLU-Redux example using NeMo Gym instead of Harbor, see
[Evaluate Switchyard routing with NeMo Gym](nemo_gym/README.md).

For task-level decision-model routing, see the
[Jev routing profile](routing-profiles/tb21-jev-opus-glm.toml).

## Prerequisites

From the repo root:

```bash
uv sync
```

You also need Docker with Compose support, because baseline runs launch task containers and use the
generated benchmark proxy topology. Runs with `--server-config` also start Switchyard inside
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

Rust server TOML files refer to the credential through `api_key_env`. For another provider, copy a
config and update its `api_key_env`, `base_url`, and model ids to match that provider.

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

Omit `--server-config` to fully disable Switchyard. The runner still creates the benchmark
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

Pass `--server-config` to start `switchyard-server` and route Harbor traffic through it. This smoke
test uses `benchmark/server-configs/tb-lite-llm-classifier-opus-kimi-gemini.toml`, a Rust
task-classifier configuration for coding-agent tasks:

```bash
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --server-config benchmark/server-configs/tb-lite-llm-classifier-opus-kimi-gemini.toml \
  --model switchyard \
  --agent codex \
  --reasoning-effort xhigh \
  --n-tasks 1 \
  --n-concurrent 1 \
  --max-retries 0
```

Use the route `id` from the TOML as `--model`. In this config, the Gemini classifier selects the
target tier for the task, then Switchyard routes to one of:

- strong: `anthropic/claude-opus-4.7`
- weak: `moonshotai/kimi-k2.7-code`

Classifier model: `google/gemini-3.5-flash`.

To smoke-test a single-model Switchyard path instead, use one of:

```text
benchmark/server-configs/tb-lite-single-gpt-5-5.toml
benchmark/server-configs/tb-lite-single-opus-4-7.toml
```

By default, the runner starts in the background and prints the PID, log path, and kill command.

## Book Modes

Both book modes use the same generated `--harbor-path` dataset, prebaked agent images, and proxy
sidecar topology. Switchyard is Dockerized only when `--server-config` is provided.

Closed-book mode is the default:

```bash
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --server-config benchmark/server-configs/tb-lite-llm-classifier-opus-kimi-gemini.toml \
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
  --server-config benchmark/server-configs/tb-lite-llm-classifier-opus-kimi-gemini.toml \
  --model switchyard \
  --agent codex \
  --n-tasks 1
```

Use open-book mode only when the evaluation intentionally allows internet access. The manifest
records the mode, the local dataset digest, a snapshot of the server config, proxy metadata, upstream
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
  --server-config benchmark/server-configs/tb-lite-llm-classifier-opus-kimi-gemini.toml \
  --model switchyard \
  --agent codex \
  --reasoning-effort xhigh \
  --n-concurrent 8 \
  --max-retries 2
```

Tune `--n-concurrent` for your machine and provider quota. Use `--task-id`, `--task-list-file`, or
`--n-tasks` for subsets.

### Run with the pi coding agent

```bash
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --server-config benchmark/server-configs/tb-lite-llm-classifier-opus-kimi-gemini.toml \
  --agent pi \
  --model switchyard \
  --reasoning-effort high \
  --harbor-extra --ae --harbor-extra PI_CONTEXT_WINDOW=200000 \
  --harbor-extra --ae --harbor-extra PI_MAX_OUTPUT_TOKENS=32000 \
  --n-concurrent 8 \
  --max-retries 2
```

With `--server-config`, the script passes the model label `switchyard/<route>` to Harbor's pi
agent. The patched agent then writes `~/.pi/agent/models.json` inside the task container. That file
defines a `switchyard` provider that points at `OPENAI_BASE_URL` and uses the `openai-completions`
API. The agent environment variables `PI_CONTEXT_WINDOW` and `PI_MAX_OUTPUT_TOKENS` set
`contextWindow` and `maxTokens` on that model entry. When they are unset, pi uses its defaults of
128000 and 16384. `--reasoning-effort` sets pi's `--thinking` level, so pass one of `off`,
`minimal`, `low`, `medium`, `high`, or `xhigh`. Without `--server-config`, pass pi's own provider
label as `--model`, for example `openrouter/openai/gpt-5.5`.

## Inspect A Run

Run directories are created under `benchmark/tb_runs/`. The most useful artifacts are:

```text
run_manifest.json
server.log
harbor.log
server_metrics_final.prom
routing_stats_final.json
jobs/<job-name>/result.json
jobs/<job-name>/<task-id>/agent/trajectory.json
```

The manifest records the command, git state, Harbor patch provenance, local dataset digest, copied
server config, direct-upstream metadata when Switchyard is disabled, book-mode settings, agent
version pins, log paths, and final Harbor status.

`server_metrics_final.prom` is the final `/metrics` snapshot. `routing_stats_final.json` is the
final aggregate `/v1/stats` snapshot, including model and tier calls, errors, tokens, and latency.
Neither artifact provides task or trial attribution. The runner writes them only after Harbor exits
and while the Rust server is still reachable; otherwise the manifest records them as missing.
`routing_requests.jsonl` and `routing_stats_by_task.json` are not produced by the Rust server.

## Docker Image Notes

Baseline runs build `switchyard-baseline:local` from
the repository-root `Dockerfile`.
The default is to rebuild before each run so the container matches the current checkout.

To reuse an already built image:

```bash
SWITCHYARD_DOCKER_BUILD=0 bash benchmark/run-baseline.sh ...
```

Only reuse the image when you know it already contains the current Rust `switchyard-server` binary.

## DeepSWE v1.1

Run Codex on a DeepSWE task in Docker with Harbor. Codex sends its Responses requests
to the local Switchyard server. The [`deepswe-single` route](server-configs/deepswe-single.toml)
forwards them to OpenRouter.

From the Switchyard repo root, check out the dataset and select a task.

```bash
git clone https://github.com/datacurve-ai/deep-swe benchmark/datasets/deep-swe
git -C benchmark/datasets/deep-swe checkout 0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea
task=benchmark/datasets/deep-swe/tasks/anko-default-function-arguments
```

### Start Switchyard

In a second terminal on the Linux Docker host, set `OPENROUTER_API_KEY`. From the
repo root, install the server and bind it to the Docker bridge to limit access to your key.

```bash
cargo install --locked --path crates/switchyard-server
bridge=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
switchyard-server --config benchmark/server-configs/deepswe-single.toml --host "$bridge" --port 8080
```

### Run Codex through Switchyard with Harbor

Docker's Linux kernel needs `nft_fib_inet` support. On ARM, enable amd64 emulation
and pull the task's `environment.docker_image` with `docker pull --platform linux/amd64`.

In the first terminal, point Codex at the server's bridge URL and route ID.

```bash
uv venv --python 3.12 benchmark/.venv
uv pip install --python benchmark/.venv/bin/python 'harbor==0.26.0'
bridge=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
OPENAI_API_KEY=unused benchmark/.venv/bin/harbor run --env docker -p "$task" -a codex \
  -m openai/deepswe-single --base-url "http://$bridge:8080/v1" \
  --ak version=0.154.0 --ak web_search=disabled --effort max
```

For another Switchyard route, change the server config and use its route ID as
`-m openai/<route-id>`. The [qualification settings](DEEPSWE_V11_QUALIFICATION.md)
list the routing profiles used for the [published results](../docs/benchmarks.md#deepswe).
For a direct baseline, substitute the provider's Responses URL, model ID and key.

For closed-book runs, also block hosted web tools at the gateway. To run fully
offline, configure Switchyard to use a local model and stage the dependencies below.

<details>
<summary>Offline setup</summary>

While connected, run the oracle once to cache the Docker images. It uses the
reference solution and needs no API key. Expect reward 1 in `verifier/reward.json`.

```bash
benchmark/.venv/bin/harbor run --env docker -p "$task" -a oracle
```

Stage the Codex binaries on the Docker host.

```bash
bundle_dir=$(mktemp -d "$PWD/benchmark/datasets/codex.XXXXXX")
archive=$(npm pack --silent --pack-destination "$bundle_dir" '@openai/codex@0.154.0-linux-x64')
tar -xzf "$bundle_dir/$archive" -C "$bundle_dir"
dist="$bundle_dir/package/vendor/x86_64-unknown-linux-musl"
mounts=$(jq -cn --arg d "$dist" '[
  {type:"bind", source:($d+"/bin/codex"), target:"/usr/local/bin/codex", read_only:true},
  {type:"bind", source:($d+"/codex-path/rg"), target:"/usr/local/bin/rg", read_only:true}
]')
```

Add `--ak skip_install=true --mounts "$mounts"` to the Codex command.

</details>

### Use Pier with the same route

For [Pier](https://github.com/datacurve-ai/pier) 0.3.1, start Switchyard with
`--port 443` in the [server command above](#start-switchyard). Pier's proxy allows
ports 80 and 443. Binding port 443 on Linux may require `CAP_NET_BIND_SERVICE`.

```bash
uv tool install --python 3.12 --with 'harbor==0.23.0' 'datacurve-pier==0.3.1'
bridge=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
pier run -p "$task" --agent codex --model openai/deepswe-single \
  --ae OPENAI_BASE_URL="http://$bridge:443/v1" --ae OPENAI_API_KEY=unused
```

For the full corpus, replace `"$task"` with `benchmark/datasets/deep-swe/tasks`.

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
