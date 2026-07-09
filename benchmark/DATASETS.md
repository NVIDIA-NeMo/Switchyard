<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Benchmark Datasets

The main [Harbor benchmark guide](README.md) uses the generated TB Lite dataset:

```text
benchmark/datasets/openthoughts-tblite-closed-book
```

Use this page when you want to prepare a different Harbor dataset for the same
Switchyard benchmark runner.

## Supported Datasets

These Harbor datasets are supported out of the box — `prepare_harbor_dataset.py` recognizes each
one and generates the right closed-book proxy allowlist automatically:

| Dataset | Harbor id | Closed-book allowlist |
|---------|-----------|-----------------------|
| TB Lite (default) | `openthoughts-tblite@2.0` | none beyond the closed-book base |
| Terminal-Bench 2.0 | `terminal-bench/terminal-bench-2` | TB2 Oracle package/data sources |
| Terminal-Bench 2.1 | `terminal-bench/terminal-bench-2-1` | shares the TB2 Oracle allowlist |
| SWE-Bench Pro | `cais/swebenchpro` | none beyond the closed-book base |

Any other Harbor dataset that matches the [supported shape](#supported-shape) can still be prepared;
it just receives no dataset-specific allowlist. Generate commands for each are below.

## Supported Shape

`prepare_harbor_dataset.py` is a repo-local dataset rewriter, not a published Harbor adapter. It can
prepare another Harbor dataset if the exported dataset has the same basic shape as TB Lite:

- task directories with `task.toml`
- optional `environment/Dockerfile`
- optional `environment/docker-compose.yaml`

The generated dataset must contain `switchyard_dataset_manifest.json`; `run-baseline.sh` checks for
that file before launch.

## Generate Another Dataset

Download and rewrite Terminal-Bench 2.0 from Harbor. The generated proxy allowlist includes
the package and data sources needed by the official Oracle solutions:

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dataset terminal-bench/terminal-bench-2 \
  --output-dir benchmark/datasets/terminal-bench-2-closed-book \
  --overwrite
```

Terminal-Bench 2.1 is the verified iteration of 2.0. It shares the same task family and Oracle
egress, so the generated proxy allowlist matches 2.0:

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dataset terminal-bench/terminal-bench-2-1 \
  --output-dir benchmark/datasets/terminal-bench-2-1-closed-book \
  --overwrite
```

Download and rewrite SWE-Bench Pro from Harbor. The Harbor dataset id is `cais/swebenchpro`; the
generated dataset keeps the closed-book proxy topology without opening dataset-specific agent
egress:

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dataset cais/swebenchpro \
  --output-dir benchmark/datasets/swebenchpro-closed-book \
  --overwrite
```

Rewrite an already exported dataset:

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dataset terminal-bench/terminal-bench-2 \
  --source-dir /path/to/exported/terminal-bench-2 \
  --output-dir benchmark/datasets/terminal-bench-2-closed-book \
  --overwrite
```

If you already exported TB Lite and only want to avoid downloading it again:

```bash
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dir /path/to/exported/openthoughts-tblite \
  --overwrite
```

## Run One Task First

Pass the generated directory to the runner:

```bash
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/terminal-bench-2-closed-book \
  --routing-profiles benchmark/routing-profiles/tb-lite-llm-classifier-opus-kimi-gemini.yaml \
  --model switchyard \
  --agent codex \
  --n-tasks 1 \
  --n-concurrent 1 \
  --max-retries 0
```

Use a routing profile and `--model` that match the route key you want Harbor agents to call. Omit
`--routing-profiles` for a direct-upstream comparison, as described in the main README.

Before reporting numbers for a new dataset, inspect:

```text
server.log
harbor.log
run_manifest.json
jobs/<job-name>/<task-id>/agent/trajectory.json
```

For Switchyard runs, also inspect `routing_stats_final.json`.

## What The Rewriter Changes

For each task, the rewriter prepares an image with pinned agent tooling:

- If `task.toml` has `[environment].docker_image`, it creates `environment/Dockerfile` with
  `FROM <original image>`, adds `USER root`, injects the pinned agent install layer, and removes
  `docker_image` from `task.toml`.
- If the task already has `environment/Dockerfile`, it appends the pinned agent install layer.
- If neither exists, it creates a minimal `environment/Dockerfile` from `ubuntu:22.04`.

The injected layer installs the pinned Node, Claude Code, Codex, and OpenCode versions from
`benchmark/agent-versions.env`, then runs `claude --version`, `codex --version`, and
`opencode --version` during image build. The base image must support `apt-get`, `apk`, or `yum`, or
already provide `curl`, `tar`, and `gzip`. It must be `x86_64/amd64` or `aarch64/arm64`.

For non-TB datasets, inspect generated Dockerfiles before a full run:

```bash
rg -n "SWITCHYARD_PREBAKED_AGENT_VERSIONS|npm install -g|USER " \
  benchmark/datasets/terminal-bench-2-closed-book
```

If an original Dockerfile switches to a non-root user before the end, the appended install layer may
not be able to install global tools. Move that `USER` line after the injected layer, or add a root
install block and restore the intended task user afterward.

The rewriter also injects benchmark proxy wiring:

- All task services except `proxy` move onto the internal-only `agent-internal` network.
- `network_mode` is removed from task services so they cannot bypass the proxy.
- The `main` service receives `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, and proxy CA environment.
- A generated entrypoint installs the proxy CA into the system trust store before agent commands run.
- A `proxy` service is added with access to both `agent-internal` and the external Switchyard
  network.
- `switchyard_dataset_manifest.json` records task digests, agent version pins, and proxy metadata.
