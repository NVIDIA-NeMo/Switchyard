# CLI Reference

This page is the canonical reference for the documented `switchyard`
subcommands. It mirrors the relevant output of `switchyard --help` and
`switchyard <verb> --help`. Tutorials and recipes live in
[Getting Started](getting_started.md); this page is reference material only.

## Verbs at a glance

| Verb | Audience | What it does |
|---|---|---|
| [`serve`](#switchyard-serve) | Ops | Long-running proxy server. Serve a YAML route bundle selected with the global `--routing-profiles` flag. |
| [`configure`](#switchyard-configure) | Both | Persists provider credentials, a saved routing-profile path, and skill-distillation settings under `~/.config/switchyard/`. With `--show`, also prints a redacted resolution snapshot. With `--list-models`, prints a searchable list of backend models. |
| [`verify`](#switchyard-verify) | Ops | Sequenced pass/fail checklist for the proxy and backend. Suitable for a readiness probe or CI install gate. |

## Global flags

These apply to the top-level `switchyard` command, before any verb.

| Flag | Purpose |
|---|---|
| `--version` | Print the installed Switchyard version (`switchyard X.Y.Z`) and exit. Reads the version from the installed package metadata. |
| `--routing-profiles PATH` / `-c PATH` | [Routing](#routing) bundle applied to `serve` and `configure`. Pass before the verb; separate with `--` for clarity. |
| `--enable-rl-logging` | Write local [RL trace logs](#rl-trace-logging), one `message_history` JSON file per turn, for `serve` route-bundle sessions. |
| `--rl-log-dir DIR` | Output directory for `--enable-rl-logging` traces (default: `./rl_data`). No effect without `--enable-rl-logging`. |

## Cross-cutting flag families

Most flags appear on more than one verb. Definitions live here so the per-verb
sections can stay short.

### Credentials and endpoint

| Flag | Purpose |
|---|---|
| `--api-key VALUE` | API key for the backend. Resolves through the [API-key waterfall](#api-key-resolution). |
| `--base-url URL` | Backend base URL. Resolves through the [base-URL waterfall](#base-url-resolution). |
| `--provider ID` | Provider id for saved configuration (default: `openrouter`). Used by `configure` setup, `--show`, and `--list-models`. |

### Backend format selection

The `format` field in a target or route configuration controls the API used for
upstream requests. Configuration files use these lowercase values:

| Value | Upstream behavior | Use when |
|---|---|---|
| `openai` | Sends to `/v1/chat/completions` without probing. | The upstream is OpenAI-compatible, including NIM and OpenRouter. |
| `anthropic` | Sends to `/v1/messages` without probing. | The upstream supports the Anthropic Messages API natively. |
| `responses` | Sends to `/v1/responses` without probing. | The upstream supports the OpenAI Responses API natively. |
| `auto` | Probes the upstream and selects a supported format. | The upstream is unknown or the same configuration must work across providers. |

`auto` resolves formats in this order:

1. Probe `/v1/chat/completions`; use `openai` when supported.
2. Probe `/v1/messages`; use `anthropic` when supported.
3. Probe `/v1/responses`; use `responses` when supported.
4. Fall back to `openai` (`/v1/chat/completions`).

Prefer an explicit format when the upstream contract is known so startup does
not require capability probes.

### Routing

Routing policies live in routing-profile YAML files. `serve` uses the global
`--routing-profiles` flag:

| Flag | Purpose |
|---|---|
| `--routing-profiles PATH` | Path to a routing-profile YAML bundle. Each entry under `routes:` builds its own chain. Public route types are `model`, `random_routing`, `stage_router`, `escalation_router`, and `deterministic`. Falls back to the path persisted by `switchyard --routing-profiles PATH -- configure` when omitted. |

For `type: deterministic` routes, the `classifier:` block also accepts:

| Key | Purpose |
|---|---|
| `prompt` | Optional classifier system-prompt override. Leave unset or blank to use the selected profile's built-in prompt. `${ENV_VAR}` references are expanded when the bundle is loaded. |
| `max_request_chars` | Optional cap on the serialized request summary sent to the classifier before truncation. Defaults to `16000`; minimum `256`. |
| `recent_turn_window` | Number of trailing conversation turns included alongside the stable system/first-user anchors. Defaults to `4`. |

Benchmark runs started through `benchmark/run-baseline.sh --server-config` copy the server TOML into the run directory and record its path and SHA-256 in `run_manifest.json` under `server.server_config_snapshot` and `server.server_config_snapshot_digest`.

Each route name becomes a model ID on `GET /v1/models`. Clients select a route
by setting the request's `model` field to that name.

### Intake sink (serve)

`serve` wires intake processors into every route in the loaded bundle. Requests
still opt in with `store=true` or `x-switchyard-intake-enabled=true`.

> **Note:** Switchyard has two independent ways to capture training data. The
> **Intake sink** posts live captures to nemo-platform.
> **`--enable-rl-logging`** writes local `message_history` JSON traces for
> `serve` route-bundle sessions. Either, both, or neither may be enabled.

| Flag | Purpose |
|---|---|
| `--intake-enabled` / `--enable-intake` | Enable the Intake sink. `--enable-intake` is a deprecated alias. Defaults to NMP SDK credentials (`nmp auth login` once). |
| `--intake-base-url URL` | Override intake base URL. |
| `--intake-workspace NAME` | Override workspace for Intake records. |
| `--intake-api-key VALUE` | Override bearer token. Disables the SDK's transparent refresh. |
| `--intake-target-url URL` | Post flat per-request telemetry to this full URL instead of chat-completions ingest. Defaults to `$SWITCHYARD_INTAKE_TARGET_URL`. |

The Intake sink posts live model-call captures to nemo-platform
`/apis/intake/v2/workspaces/{workspace}/ingest/chat-completions`. That endpoint
derives queryable token fields from `response.usage` and queryable cost fields
from top-level `cost_usd`, `cost_input_usd`, `cost_output_usd`, and
`cost_details`. Switchyard emits cost fields only when the served model has a
known pricing entry. `routing_stats_final.json` carries the run-level aggregate
`/v1/stats` snapshot (per-model and per-tier calls, errors, tokens, and
latency); it does not include cost estimates. When a session ID is present,
Switchyard also maps the Intake app/task labels into top-level
`evaluation_context.dataset_*` and `evaluation_context.test_case_id` for span
queries while keeping the original labels under `request.switchyard`.

Set `--intake-target-url` to a full posting URL to send captures to a different
store instead. Switchyard then posts a flat, top-level, type-prefixed document
(`s_` string, `l_` long, `f_` float, `b_` bool, `text_` debug) to that URL,
unauthenticated, rather than the nested nemo-platform payload.

### RL trace logging

`--enable-rl-logging` attaches a response-side logger to the proxy chain of a
`serve --routing-profiles` bundle. Each completed turn, streaming or not, is
written to its own JSON file under `--rl-log-dir` (default `./rl_data`), named
`{timestamp}_trace_{id}_{id}.json`. The schema matches the pre-1.0 trace format:

```json
{
  "uuid": "…",
  "messages": [
    {
      "role": "assistant",
      "content": "…",
      "tool_calls": []
    }
  ],
  "tools": [],
  "tool_choice": "auto",
  "token_count": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "is_valid": true
}
```

Turns without an assistant choice, such as upstream errors, are skipped.

### Transport (server verbs)

| Flag | Purpose |
|---|---|
| `--host HOST` | Host to bind to (default: `0.0.0.0`). |
| `--port PORT` / `-p PORT` | Port to bind to (default: `4000`, or `server.port` from `secrets.json`). |
| `--reload` | Enable uvicorn auto-reload. |
| `--workers N` / `-w N` | Number of uvicorn worker processes (default: `1`, or `$SWITCHYARD_WORKERS`). |

## Resolution waterfalls

### API-key resolution

Verification resolves the API key in this order, stopping at the first
non-empty value:

1. `--api-key` on the CLI
2. `$OPENROUTER_API_KEY`
3. `$NVIDIA_API_KEY`
4. `$OPENAI_API_KEY`
5. `$ANTHROPIC_API_KEY`
6. `secrets/secrets.json` → first provider section with `api_key` set, with
   `openrouter` then `nvidia` checked first

For OpenRouter, set `OPENROUTER_API_KEY`; `OPENROUTER_BASE_URL` is optional
because the built-in default is `https://openrouter.ai/api/v1`.

### Base-URL resolution

1. `--base-url` on the CLI
2. The base URL matching the selected environment credential:
   `$OPENROUTER_BASE_URL`, `$NVIDIA_BASE_URL`, or `$OPENAI_BASE_URL`
3. `secrets/secrets.json` → same section traversal as the API key
4. Default: OpenRouter (`https://openrouter.ai/api/v1`)

### `secrets.json` format

```json
{
  "openrouter": {
    "api_key": "sk-or-...",
    "base_url": "https://openrouter.ai/api/v1"
  },
  "server": {
    "port": 4000
  }
}
```

`secrets/` is gitignored. Never commit this file.

## `switchyard serve`

Serve a long-running proxy from a YAML route bundle. Each entry under `routes:`
builds a runnable chain and is exposed as a model on `GET /v1/models`.

The server exposes the OpenAI Chat Completions (`/v1/chat/completions`),
Anthropic Messages (`/v1/messages`), and OpenAI Responses (`/v1/responses`)
APIs on the same host and port.

**Synopsis**

```text
switchyard [--routing-profiles PATH] serve
                 [--host HOST] [--port PORT] [--workers N]
                 [--reload] [--inbound FORMAT]
                 [--routing-log-file PATH]
                 [--intake-enabled|--enable-intake [INTAKE OVERRIDES]]
```

**Flags**

| Flag | Source |
|---|---|
| `--routing-profiles PATH` / `-c PATH` | [Routing](#routing) path. Global flag; pass before `serve`. Falls back to the saved path from `switchyard --routing-profiles PATH -- configure`. |
| `--host`, `--port`/`-p`, `--reload` | [Transport](#transport-server-verbs). |
| `--inbound FORMAT` | Backwards-compatible no-op; all request APIs are always registered. |
| `--workers` / `-w` | uvicorn worker count. |
| `--routing-log-file PATH` | Append one JSONL routing record per request to `PATH`. Also exposes `GET /v1/routing/session-stats?session_id=...`. |
| `--intake-enabled` / `--enable-intake`, `--intake-base-url`, `--intake-workspace`, `--intake-api-key`, `--intake-target-url` | [Intake sink](#intake-sink-serve). |

**Notes**

- `serve` always registers `POST /v1/chat/completions`, `POST /v1/messages`,
  `POST /v1/responses`, `GET /v1/models`, and `GET /health`.
- `GET /v1/stats` and `GET /v1/routing/stats` expose route statistics.
- `--inbound` is retained as a compatibility no-op.

**Examples**

```bash
switchyard --routing-profiles routes.yaml -- serve --port 4000

switchyard --routing-profiles routes.yaml -- configure --target provider \
  --provider openrouter --api-key "$OPENROUTER_API_KEY" \
  --base-url https://openrouter.ai/api/v1 --no-tui --no-model-discovery
switchyard serve --port 4000

SWITCHYARD_WORKERS=4 switchyard --routing-profiles routes.yaml -- serve
```

## `switchyard configure`

Persist user-level Switchyard defaults under `~/.config/switchyard/`.
Credentials are stored separately from non-secret config, with owner-only file
permissions. Skill-distillation config also lives in
`~/.config/switchyard/config.json`.

**Synopsis**

```text
switchyard [--routing-profiles PATH] configure
                     [--show [--check] [--json] | --reset | --list-models]
                     [--target provider]
                     [--query SUBSTRING] [--limit N]
                     [--provider ID]
                     [--base-url URL] [--api-key VALUE]
                     [--skill-distillation NAMESPACE]
                     [--disable-skill-distillation]
                     [--no-model-discovery] [--no-tui]
```

**Modes (mutually exclusive)**

| Flag | What it does |
|---|---|
| _(none)_ | Interactive setup. Prompts for missing provider defaults and runs model discovery unless `--no-model-discovery`. |
| `--show` | Print the redacted saved config plus a provider, base-URL, and API-key resolution snapshot. Pair with `--check` for a live `GET /models` probe, or `--json` for the raw redacted JSON snapshot. |
| `--reset` | Delete persisted user config and credentials. |
| `--list-models` | Fetch `GET /models` from the resolved provider. Pair with `--query` to filter by substring and `--limit` to cap results. |

**Configuration knobs**

| Flag | Purpose |
|---|---|
| `--target provider` | Save provider-level defaults without requiring model-specific settings. |
| `--provider`, `--base-url`, `--api-key` | Provider-level defaults and one-off overrides for `--show` and `--list-models`. |
| `--routing-profiles PATH` | Global flag; pass before `configure`. Parses the YAML at `PATH` and stores the parsed bundle inline in `~/.config/switchyard/config.json`. Subsequent `serve` runs use it when no path is supplied. Pass an empty string to clear. |
| `--skill-distillation NAMESPACE` | Save a namespace for one skill that improves over time. Many sessions or trajectories can contribute to it; the namespace is not a session ID. |
| `--disable-skill-distillation` | Remove the saved skill-distillation config. Cannot be combined with `--skill-distillation`. |
| `--query` / `-q SUBSTRING` | With `--list-models`, case-insensitive substring filter. |
| `--limit N` | With `--list-models`, cap on the number of models printed (default: 50; pass `0` for unlimited). |
| `--no-model-discovery` | Skip `GET /models` and rely on existing provider values during interactive setup. |
| `--no-tui` | Use plain text prompts instead of the TUI selector. |
| `--check` | With `--show`, call `GET /models` against the resolved provider and report pass/fail. |

> **Saved bundles keep `${VAR}` references literal.** A saved routing-profile
> bundle stores `${OPENROUTER_API_KEY}` and other `${VAR}` references verbatim.
> Export them before running `serve`.

**Skill-distillation config**

```json
{
  "skill_distillation": {
    "namespace": "tooluniverse-trialqa"
  }
}
```

Namespaces must be a single safe path component: letters, numbers, dot,
underscore, and hyphen only. One namespace identifies one skill that improves
over time, and many sessions or trajectories can contribute to it.

**Examples**

```bash
switchyard configure

switchyard --routing-profiles routes.yaml -- configure --target provider \
  --provider openrouter --api-key "$OPENROUTER_API_KEY" \
  --base-url https://openrouter.ai/api/v1 --no-tui --no-model-discovery

switchyard configure --show
switchyard configure --show --check
switchyard configure --show --json

switchyard configure --list-models --query gpt
switchyard configure --list-models --limit 0 --provider openrouter \
  --api-key "$OPENROUTER_API_KEY" \
  --base-url https://openrouter.ai/api/v1

switchyard configure --skill-distillation tooluniverse-trialqa
switchyard configure --disable-skill-distillation
switchyard configure --reset
```

!!! note "Non-interactive / CI usage"
    Pass `--target provider` to save provider credentials without requiring
    model-specific settings.

    **`configure` requires an explicit `--api-key` flag.** It does not read
    `OPENROUTER_API_KEY` or another `*_API_KEY` environment variable, and it
    does not read `api_key` from a routing-profile bundle.

## `switchyard verify`

Sequenced pass/fail checklist that confirms a Switchyard install works
end-to-end against a real backend. It is suitable for readiness probes and
pre-deployment smoke tests.

**Synopsis**

```text
switchyard verify [--model ID] [--base-url URL] [--api-key VALUE]
                  [--port PORT] [--timeout SECONDS]
```

**Example model**

`openai/gpt-4o-mini` is a portable OpenRouter example. Pass `--model` when your
provider uses a different model ID.

**Checklist**

1. Resolve credentials (CLI → env → `secrets.json`).
2. Reach the backend via `GET /models`.
3. Probe `/v1/chat/completions`, `/v1/messages`, and `/v1/responses` support.
4. Start a proxy on a free port.
5. Round-trip a chat completion through the chain.
6. Tear the proxy down.

**Exit codes**

- `0`: every step passed.
- Non-zero: first failing step; the error message names the source it tried and
  what to fix.

**Examples**

```bash
switchyard verify
switchyard verify --model openai/gpt-4o-mini
switchyard verify --api-key "$OPENROUTER_API_KEY" \
  --base-url https://openrouter.ai/api/v1
```

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Backend credentials resolved by `verify`. |
| `OPENROUTER_BASE_URL`, `NVIDIA_BASE_URL`, `OPENAI_BASE_URL` | Backend base URL overrides paired with the selected provider credential. |
| `OPENAI_API_BASE` | Legacy alias for `OPENAI_BASE_URL`. Prefer `OPENAI_BASE_URL` for new configurations. |
| `SWITCHYARD_INTAKE_ENABLED` | Boolean equivalent of `--intake-enabled` / `--enable-intake`. |
| `SWITCHYARD_INTAKE_TARGET_URL` | Full posting URL for the alternate flat-document intake sink. |
| `SWITCHYARD_WORKERS` | Default uvicorn worker count for `serve`. |
| `SWITCHYARD_TELEMETRY_OPT_OUT` | Disable the `X-Switchyard-Version` telemetry header on outbound calls. |
| `SWITCHYARD_INTAKE_BASE_URL`, `SWITCHYARD_INTAKE_WORKSPACE`, `SWITCHYARD_INTAKE_API_KEY`, `SWITCHYARD_INTAKE_APP`, `SWITCHYARD_INTAKE_TASK`, `SWITCHYARD_SESSION_ID`, `SWITCHYARD_USER_ID` | Intake-sink overrides for CI and headless runs. |
| `NMP_ACCESS_TOKEN` | Fallback bearer token for the Intake sink when the NMP SDK config is not present. |

## See also

- [Getting Started](getting_started.md): install Switchyard and run your first request.
- [Architecture](architecture.md): system context and end-to-end request flow.
