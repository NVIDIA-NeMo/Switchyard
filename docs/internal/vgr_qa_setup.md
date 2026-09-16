<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Set Up Switchyard VGR for QA

This guide gives QA the minimum setup needed to run an agent through
Verification-Gated Routing (VGR) on the RTX Spark test configuration. It covers
the currently supported cycle-1 path:

- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` using
  `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, served by `llama.cpp` as the local tier
- Opus 5 through the NVIDIA inference gateway as the cloud tier
- native ARM64 Windows `switchyard-server.exe`
- an agent connected directly to the Switchyard proxy
- optionally, Hermes and Terminal-Bench 2.1 in WSL2 and Docker Desktop
- text-only requests through the Switchyard OpenAI-compatible proxy

OpenShell, OPA, native privacy/no-egress mode, OCSF, OpenClaw, Router Sidecar,
LiteLLM, and nemo-relay qualification are not part of this setup.

Requirements marked **General** apply whenever QA runs an agent through VGR.
Requirements marked **Benchmark only** are additions for Harbor,
Terminal-Bench, or another containerized benchmark.

## Required software and access

Install or obtain these dependencies before starting QA.

### Native Windows

| Dependency | Requirement |
|---|---|
| Operating system | Windows 11 ARM64 on the approved RTX Spark 48 GB system |
| PowerShell | PowerShell 7 or newer |
| Git | Access to the internal Switchyard repository and QA branch |
| Rust | Rust 1.96.1 through `rustup`, with host `aarch64-pc-windows-msvc` |
| Visual Studio Build Tools | ARM64 MSVC tools, Windows 11 SDK, and Clang; use the approved 14.44 toolset |
| NVIDIA software | Current approved Windows GPU driver |
| `llama.cpp` | Release-pinned native Windows ARM64 `llama-server` build |
| Local model | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, file `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| Agent client | A client that accepts a custom OpenAI- or Anthropic-compatible base URL and model ID |
| Docker Desktop | **Benchmark only:** WSL2 integration, Docker Compose, and x64 emulation enabled |

### WSL2 — benchmark only

| Dependency | Requirement |
|---|---|
| Linux distribution | Ubuntu 22.04 / Debian 13.5 or newer under WSL2 |
| Docker client | Connected to Docker Desktop; `docker compose version` must succeed |
| `uv` | Current project-supported `uv` for Python environment management |
| Python | Python 3.12 managed by `uv` for Harbor |
| Command-line tools | Git, `patch`, `jq`, `curl`, `awk`, and standard GNU shell tools |

Switchyard itself does not require host Node.js, `botocore`, or `ripgrep`.
Install any additional runtime required by the selected agent. For the
benchmark path, agent-specific Node.js and `uv` versions are pinned and
installed inside the generated task images.

### Credentials and network access

**General**

- Internal Git access for `internal/vgr-qa-policy-2.11`.
- A valid `NVIDIA_API_KEY` for the Opus 5 cloud tier.
- Access to Hugging Face for the initial model download, unless the model is
  already present in the local `llama.cpp` cache.
- Local ports 4000 and 9931 available.
- Windows private-network firewall permission if an agent outside the native
  Windows host must reach Switchyard.

**Benchmark only**

- Access to Docker Hub, GHCR, PyPI, the benchmark source, and the approved
  GitHub Hermes source.
- Windows private-network firewall permission for Docker Desktop to reach
  native Switchyard on port 4000.

Do not put credentials in the repository, TOML files, or command history.
Enter them through masked prompts and keep them in the server process
environment.

## 1. Understand the process layout

Run each component in the location shown below.

| Component | Location | Address | Applies to |
|---|---|---|---|
| Qwen local model | Native Windows, `llama-server` | `127.0.0.1:9931` | General |
| Switchyard VGR | Native Windows, `switchyard-server.exe` | `127.0.0.1:4000` or `0.0.0.0:4000` | General |
| Agent | Native Windows or an approved client environment | Switchyard proxy | General |
| Hermes and Harbor | WSL2 | Connect through Docker | Benchmark only |
| Terminal-Bench tasks | Docker Desktop | `host.docker.internal:4000` | Benchmark only |

`host.docker.internal` is a Docker hostname. It normally does not resolve from a
plain WSL shell. Use it inside containers; use `127.0.0.1` from native Windows.

## 2. Prepare the repository

Use the QA branch in the Windows checkout:

```powershell
Set-Location C:\path\to\Switchyard
git switch internal/vgr-qa-policy-2.11
git pull --ff-only
git rev-parse HEAD
```

**Benchmark/shared-WSL note.** If WSL reports hundreds of modified files in
this shared Windows checkout, they are likely CRLF conversion noise. Configure
this checkout consistently:

```bash
cd /mnt/c/path/to/Switchyard
git config --local core.autocrlf true
git status --short
```

Do not discard real changes merely to make the checkout clean. Generated
benchmark datasets and results are expected to be untracked.

## 3. Start the local Qwen model

From native Windows PowerShell, run the release-pinned `llama-server.exe`
build. The command downloads the approved GGUF from Hugging Face on first use
and reuses the local cache afterward:

```powershell
llama-server.exe -hf "unsloth/Qwen3.6-35B-A3B-MTP-GGUF" `
  -hff "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" `
  --no-mmproj `
  --host 127.0.0.1 `
  --port 9931 `
  --batch-size 4096 `
  --ubatch-size 4096 `
  --ctx-size 65536 `
  --parallel 1 `
  -ngl 999 `
  --jinja `
  --chat-template-kwargs '{"preserve_thinking":true}' `
  --spec-type draft-mtp `
  --spec-draft-n-max 2
```

Verify the endpoint from another native Windows PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:9931/v1/models
```

## 4. Create the native Windows VGR configuration

The checked-in holdout configuration is also the starting template for the
prescribed QA model pair; using it does not require running a benchmark. Copy
it so the native Windows endpoint can be changed without editing the tracked
template:

```powershell
Copy-Item `
  benchmark\server-configs\vgr-holdout-qwen36-opus5.toml `
  benchmark\server-configs\vgr-holdout-native-windows.toml
```

In the copied file, make only these local-tier changes:

```toml
[llm_clients.local_qwen]
format = "openai_chat"
base_url = "http://127.0.0.1:9931/v1"
max_retries = 2

[targets.local]
id = "Qwen/Qwen3.6-35B-A3B"
llm_client = "local_qwen"
```

Keep the checked-in cloud and VGR blocks unchanged:

- cloud client: `https://inference-api.nvidia.com`;
- cloud model: `aws/anthropic/bedrock-claude-opus-5`;
- route ID: `switchyard/vgr`;
- route type: `vgr`;
- mode: `active`; and
- active approval: `prospective-validation-and-canary-approved`.

The holdout configuration uses a longer decision deadline for agentic
benchmarks. It is not the fixture for the separate 30-second POLICY deadline
test; that contract is represented by
`benchmark/configs/vgr-qa-policy-2.11.toml`.

Never place `NVIDIA_API_KEY` directly in TOML. The local `llama-server`
endpoint in this configuration does not require an API key.

## 5. Build and start native Switchyard

Use an ARM64 Visual Studio developer environment. The Rust host must be
`aarch64-pc-windows-msvc`:

```powershell
rustc -vV | Select-String host
cargo build --locked --release -p switchyard-server
```

Set the cloud credential and validate the configuration:

```powershell
$env:NVIDIA_API_KEY = Read-Host -MaskInput "NVIDIA_API_KEY"
$config = (Resolve-Path `
  "benchmark\server-configs\vgr-holdout-native-windows.toml").Path

.\target\release\switchyard-server.exe --config $config --dry-run
```

Start the server and keep this PowerShell window open:

```powershell
# Use 127.0.0.1 for a native agent. Use 0.0.0.0 only when Docker must connect.
$listenHost = "127.0.0.1"
$routingLog = (Join-Path (Get-Location) "routing_requests.jsonl")

.\target\release\switchyard-server.exe `
  --config $config `
  --host $listenHost `
  --port 4000 `
  --routing-log-file $routingLog
```

For **general native agent use**, keep `$listenHost` set to `127.0.0.1`.

For a **Docker-hosted agent or benchmark**, set `$listenHost` to `0.0.0.0` so
Docker Desktop can connect, and permit only the Windows private-network
firewall rule. Switchyard does not provide an internet-facing inbound
authentication boundary; do not expose port 4000 to a public network.

For a benchmark, `$routingLog` may instead be placed under
`benchmark\native-runs\`.

## 6. Verify the route before starting an agent

### General agent check

From native Windows PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:4000/health
Invoke-RestMethod http://127.0.0.1:4000/v1/models
```

Confirm that the health response is `ok` and that `switchyard/vgr` appears in
the model list.

### Additional Docker or benchmark check

When the agent or benchmark runs in Docker, confirm that a container can reach
native Switchyard:

```bash
docker run --rm python:3.12-slim \
  python -c 'import urllib.request; print(urllib.request.urlopen(
    "http://host.docker.internal:4000/health").read().decode())'
```

The expected response is:

```json
{"status":"ok"}
```

Optionally send one request through VGR from the same Docker network path:

```bash
SMOKE_ID="vgr-qa-smoke-$(date -u +%Y%m%dT%H%M%SZ)"

docker run --rm -i \
  -e SMOKE_ID="$SMOKE_ID" \
  python:3.12-slim python - <<'PY'
import json
import os
import urllib.request

smoke_id = os.environ["SMOKE_ID"]
request = urllib.request.Request(
    "http://host.docker.internal:4000/v1/chat/completions",
    data=json.dumps(
        {
            "model": "switchyard/vgr",
            "stream": False,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a Python add function and one assertion. "
                        "Return code only."
                    ),
                }
            ],
        }
    ).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer switchyard-local",
        "x-switchyard-intake-task": smoke_id,
        "x-switchyard-trial-id": smoke_id,
        "x-switchyard-session-id": smoke_id,
    },
)
with urllib.request.urlopen(request, timeout=900) as response:
    print(json.dumps(json.load(response), indent=2))
PY
```

This test passes operationally when:

1. the request returns a valid model response;
2. Qwen appears in the routing log as a classifier or local-tier call;
3. the terminal routing record has matching `vgr_predicted`,
   `vgr_effective`, and `vgr_served` values for the selected path; and
4. an escalation, when selected, reaches Opus 5.

A direct request proves VGR and both endpoints are working. It does not prove a
benchmark task passes. For a native agent, send the same request to
`http://127.0.0.1:4000/v1/chat/completions` from the Windows host.

## 7. Point an agent at VGR

Agents use Switchyard as an OpenAI-compatible endpoint:

| Agent location | Base URL |
|---|---|
| Native Windows | `http://127.0.0.1:4000/v1` |
| Docker Desktop container | `http://host.docker.internal:4000/v1` |
| Plain WSL process | Windows host/gateway address; `host.docker.internal` normally does not resolve |

For every location:

```text
Model:   switchyard/vgr
API key: any non-empty client placeholder
```

The placeholder client key is not an upstream credential. Upstream credentials
remain in the native Switchyard process.

For a general OpenAI-compatible agent running on native Windows, the equivalent
environment is:

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:4000/v1"
$env:OPENAI_API_KEY = "switchyard-local"
$env:OPENAI_MODEL = "switchyard/vgr"
```

For an Anthropic-compatible agent, use:

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:4000"
$env:ANTHROPIC_API_KEY = "switchyard-local"
$env:ANTHROPIC_MODEL = "switchyard/vgr"
```

Start the agent normally after applying the appropriate environment. No
Switchyard-specific agent plugin is required.

Every request belonging to one conversation must use the same
`x-switchyard-session-id`. The agent must resend the normal conversation
history, including assistant tool calls and tool results. VGR uses that
normalized history for task typing, continuation affinity, and evidence.

## 8. Optional: run one Hermes Terminal-Bench task

Everything in this section is **benchmark only**. Skip it for direct agent
setup.

### 8.1 Prepare WSL

Keep Linux Rust build output outside the Windows `target` directory:

```bash
ROOT=/mnt/c/path/to/Switchyard
export CARGO_TARGET_DIR="$HOME/.cache/switchyard-target-linux-arm64"

cd "$ROOT"
uv sync --locked --group dev --no-install-project
uv run --no-sync harbor --version
```

Apply the checked-in Harbor patch after creating or replacing `.venv`:

```bash
REPO_ROOT="$ROOT"
HARBOR_SITE="$(
  uv run --project "$ROOT" --no-sync python -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)"

cd "$HARBOR_SITE"
patch -p1 < "$REPO_ROOT/benchmark/patches/harbor-agent-patches.diff"
```

Do not apply the patch twice. Reapply it only after Harbor or `.venv` is
reinstalled.

Prepare the closed-book Terminal-Bench 2.1 dataset:

```bash
cd "$ROOT"
uv run --no-sync python benchmark/prepare_harbor_dataset.py \
  --source-dataset terminal-bench/terminal-bench-2-1 \
  --output-dir benchmark/datasets/terminal-bench-2-1-closed-book \
  --overwrite
```

On ARM, preparation rebuilds available source Dockerfiles so `uv`, Python, and
Hermes remain native. It also normalizes generated proxy scripts to LF and pins
Hermes and `uv` versions.

### 8.2 Use a stable WSL working directory

Do not keep a long-running Harbor process in a `/mnt/c` current directory.
Windows can replace that directory handle during Git activity, causing Python
`Path.resolve()` failures hours into a run.

```bash
RUNROOT="$HOME/switchyard-harbor-run"
mkdir -p "$RUNROOT"
ln -sfn "$ROOT/benchmark" "$RUNROOT/benchmark"
cd "$RUNROOT"
```

### 8.3 Configure the agent network

```bash
docker network inspect switchyard-native-tb21 >/dev/null 2>&1 ||
  docker network create switchyard-native-tb21

export SWITCHYARD_DOCKER_NETWORK=switchyard-native-tb21
export ALLOWED_HOSTS=host.docker.internal
export OPENAI_BASE_URL=http://host.docker.internal:4000/v1
export SWITCHYARD_BASE_URL=http://host.docker.internal:4000
export OPENAI_API_KEY=switchyard-local
export CLOSED_BOOK_MODE=1
export SWITCHYARD_VERIFIER_PROXY_TOKEN="$(
  python3 -c 'import secrets; print(secrets.token_hex(24))'
)"
export SWITCHYARD_VERIFIER_HTTP_PROXY="http://verifier:${SWITCHYARD_VERIFIER_PROXY_TOKEN}@proxy:3129"
```

### 8.4 Run the Hermes demo sample

The demo runs ten Terminal-Bench 2.1 tasks with frozen Hermes outcomes and
policy-stable VGR routes:

- five local-required tasks where the local tier passed and VGR selected local;
- five cloud-required tasks where local failed, cloud passed, and VGR selected
  cloud.

```bash
bash "$ROOT/benchmark/run-hermes-demo.sh"
```

The script uses the prepared closed-book dataset, creates a unique Harbor job,
and points the containerized Hermes agent at `switchyard/vgr` on the native
Switchyard server. The task list is
`benchmark/tb21_hermes_demo_tasks.txt`. Add `--dry-run` to print the resolved
Harbor command without creating a network or starting the benchmark.

## 9. Common setup failures

Each item indicates whether it applies generally or only to the benchmark
path.

### `HTTP 401` from the cloud gateway

**General.**

The `NVIDIA_API_KEY` available to native Switchyard is invalid or expired.
Rotate it, validate it outside the benchmark, and restart Switchyard. Never
store or commit the replacement.

### GitHub `HTTP 429` while building a task

**Benchmark only.**

The current dataset generator vendors the pinned Hermes installer, but the
installer still clones the pinned Hermes source during each task image build.
GitHub can rate-limit that clone. Retry after the rate limit clears.

### `RuntimeError` with `agent_setup: null`

**Benchmark only.**

The task failed before Hermes started. Inspect
`.exception_info.exception_message` in the trial `result.json`; common causes
are Docker build failures, GitHub rate limits, or proxy startup failures.

### `env: 'bash\r': No such file or directory`

**Benchmark only.**

The generated Linux script contains CRLF line endings. Regenerate the dataset
with the current preparer, which normalizes proxy text assets to LF.

### `FileNotFoundError` from `Path.resolve()`

**Benchmark only.**

The Harbor process retained a stale `/mnt/c` working-directory handle. Launch
from the stable WSL `RUNROOT` and use `uv run --project "$ROOT"`.

### `No module named 'botocore'`

**Benchmark only.**

LiteLLM may print optional Bedrock preload warnings. They are nonfatal when this
configuration reaches Opus through the configured NVIDIA endpoint.
