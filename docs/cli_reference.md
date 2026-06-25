# CLI Reference

This page is the canonical reference for every `switchyard` subcommand. It mirrors the output of `switchyard --help` and `switchyard <verb> --help`. If you spot drift, file a docs ticket. Tutorials and recipes live in [Getting Started](getting_started.md); this page is reference material only.

## Verbs at a glance

| Verb | Audience | What it does |
|---|---|---|
| [`serve`](#switchyard-serve) | Ops | Long-running proxy server. Serve a v2 profile config with `serve --config`; each profile id and target id appears on `GET /v1/models`, and clients select one by setting the request `model`. |
| [`launch claude`](#switchyard-launch-claude) | Dev | Spawns Claude Code against a local proxy. Auto-picks a free port, sets `ANTHROPIC_*` env vars, tears the proxy down when Claude exits. `--smoke` runs a one-shot harness round-trip and exits. |
| [`launch codex`](#switchyard-launch-codex) | Dev | Spawns OpenAI Codex CLI against a local proxy and injects a transient `switchyard` provider via repeated `-c` flags. `--smoke` runs a one-shot Codex round-trip. |
| [`launch openclaw`](#switchyard-launch-openclaw) | Dev | Spawns `openclaw chat` against a local proxy using a transient `openclaw.json`; the user's OpenClaw configuration is untouched. `--smoke` runs a one-shot agent turn. |
| [`configure`](#switchyard-configure) | Both | Persists user-level defaults under `~/.config/switchyard/` (provider credentials, per-launcher model defaults, saved routing-profile path). With `--show`, also prints resolved provider, API-key source, and harness binary paths (and optional `GET /models` probe via `--check`). With `--list-models`, also prints a ranked / searchable list of the backend's models. |
| [`verify`](#switchyard-verify) | Ops | Sequenced pass/fail checklist for proxy + backend only. K8s readiness probe / CI install gate. No harness binary required. For harness-driven smoke tests, see `launch {claude,codex,openclaw} --smoke`. |

## Global flags

These apply to the top-level `switchyard` command, before any verb.

| Flag | Purpose |
|---|---|
| `--version` | Print the installed Switchyard version (`switchyard X.Y.Z`) and exit. Reads the version from the installed package metadata. |
| `--routing-profiles PATH` / `-c PATH` | Deprecated legacy [Routing](#routing) bundle applied to `serve`, `launch`, and `configure`. Pass before the verb; separate with `--` for clarity. |
| `--enable-rl-logging` | Write local [RL trace logs](#rl-trace-logging) (one `message_history` JSON file per turn) for `launch` and `serve` route-bundle sessions. Pass before the verb: `switchyard --enable-rl-logging launch claude`. Rejected by `serve --config` (the Rust profile server has no Python processor chain). |
| `--rl-log-dir DIR` | Output directory for `--enable-rl-logging` traces (default: `./rl_data`). No effect without `--enable-rl-logging`. |

## Cross-cutting flag families

Most flags appear on more than one verb. Definitions live here so the per-verb sections can stay short.

### Credentials and endpoint

| Flag | Purpose |
|---|---|
| `--api-key VALUE` | API key for the backend. Resolves through the [API-key waterfall](#api-key-resolution). |
| `--base-url URL` | Backend base URL. Resolves through the [base-URL waterfall](#base-url-resolution). |
| `--provider ID` | Provider id for saved configuration (default: `openrouter`). Used by `configure` (setup, `--show`, and `--list-models`). |

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

1. Probe `/v1/messages`; use `anthropic` when supported.
2. Probe `/v1/responses`; use `responses` when supported.
3. Fall back to `openai` and `/v1/chat/completions`.

Single-model Claude Code and Codex launches use `auto`. OpenClaw is pinned to
`openai`. Prefer an explicit format when the upstream contract is known so
startup does not require capability probes.

### Routing

Legacy routing policies that used to be standalone CLI verbs live in
routing-profile YAML files. Route bundles are deprecated; use v2 profile
configs for new `serve` setups. Two flags drive routing on `serve` and the
launchers:

| Flag | Purpose |
|---|---|
| `--model ID` | Single-model passthrough. Every request is rewritten to `model=ID` and forwarded to `--base-url`. |
| `--routing-profiles PATH` | Deprecated path to a routing-profile YAML bundle. Each entry under `routes:` builds its own chain. Public route types are `model`, `passthrough`, `random_routing`, `cascade`, and `deterministic`. Falls back to the path persisted by `switchyard --routing-profiles PATH -- configure` when omitted. |

On the launchers, the two flags are mutually exclusive: pass one or the other, not both.

- `--model ID`: single-model passthrough. Any model id from `GET /v1/models` is accepted; every request is rewritten to `model=ID` and forwarded upstream.
- `--routing-profiles PATH`: loads a multi-chain YAML bundle; the first declared route becomes the initial model.

For legacy route-bundle `type: deterministic` routes, the `classifier:` block also accepts:

| Key | Purpose |
|---|---|
| `prompt` | Optional classifier system-prompt override. Leave unset or blank to use the selected profile's built-in prompt. `${ENV_VAR}` references are expanded when the bundle is loaded. |
| `max_request_chars` | Optional cap on the serialized request summary sent to the classifier before truncation. Defaults to `16000`; minimum `256`. |
| `recent_turn_window` | Number of trailing conversation turns included alongside the stable system/first-user anchors. Defaults to `4`. |

Benchmark runs started through `benchmark/run-baseline.sh --routing-profiles` record the effective classifier prompt, prompt SHA-256, `max_request_chars`, and `recent_turn_window` in `run_manifest.json` under `server.classifier_prompts`.

**Selecting a v2 profile by id.** A `serve --config` proxy exposes every profile id (and every target id) as a model on `GET /v1/models`. There is no separate "profile" flag. A client, or a launcher using `--model <profile-id>`, selects a profile simply by naming its id as the model. This is the v2 counterpart to picking a route from a bundle. This selection pattern is specific to v2 profile configs; legacy route-bundle configs still use route names as model IDs.

### Intake sink (serve and launchers)

`serve` and launchers share the same Intake sink connection flags. `serve` wires intake processors into every route in the loaded bundle; requests still opt in with `store=true` or `x-switchyard-intake-enabled=true`. Launchers also inject those opt-in headers into the spawned client.

> **Note:** Switchyard has two independent ways to capture training data. The **Intake sink** (this section) posts live captures to nemo-platform. **`--enable-rl-logging`** (a [global flag](#global-flags)) writes local `message_history` JSON traces for `launch` and `serve` route-bundle sessions. See [RL trace logging](#rl-trace-logging). Either, both, or neither may be enabled.

| Flag | Purpose |
|---|---|
| `--intake-enabled` / `--enable-intake` | Enable the Intake sink. `--enable-intake` is a deprecated alias. Defaults to NMP SDK credentials (`nmp auth login` once). |
| `--intake-base-url URL` | Override intake base URL. |
| `--intake-workspace NAME` | Override workspace for Intake records. |
| `--intake-api-key VALUE` | Override bearer token. Disables the SDK's transparent refresh. |
| `--intake-nvdataflow-project PROJECT` | Post flat per-request telemetry to this NVDataflow project instead of chat-completions ingest. Defaults to `$SWITCHYARD_NVDATAFLOW_PROJECT`. |

Launchers additionally accept context fields stamped into each ingested request:

| Flag | Purpose |
|---|---|
| `--intake-app NAME` | App name stamped into the chat-completions ingest metadata. |
| `--intake-task NAME` | Task name stamped into the chat-completions ingest metadata (default: `developer-session`). |
| `--intake-session-id ID` | Session id stamped on every ingested request. |
| `--intake-user-id ID` | Anonymous user id stamped on every ingested request. Defaults to the stable per-machine id at `~/.switchyard/user_id`. |

The Intake sink posts live model-call captures to nemo-platform
`/apis/intake/v2/workspaces/{workspace}/ingest/chat-completions`. That endpoint
derives queryable token fields from `response.usage` and queryable cost fields
from top-level `cost_usd`, `cost_input_usd`, `cost_output_usd`, and
`cost_details`. Switchyard emits cost fields only when the served model has a
known pricing entry; `routing_stats_final.json` remains the run-level source
for aggregate routing/model cost estimates. When a session ID is present,
Switchyard also maps the Intake app/task labels into top-level
`evaluation_context.dataset_*` and `evaluation_context.test_case_id` for span
queries while keeping the original labels under `request.switchyard`.

### RL trace logging

`--enable-rl-logging` attaches a response-side logger to the proxy chain of any
`launch` session or `serve --routing-profiles` bundle. Each completed turn
(streaming or not) is written to its own JSON file under `--rl-log-dir`
(default `./rl_data`), named `{timestamp}_trace_{id}_{id}.json`. The schema
matches the pre-1.0 trace format:

```json
{
  "uuid": "…",
  "messages": [ … full request history …, {"role": "assistant", "content": "…", "tool_calls": […]} ],
  "tools": [ {"id": "…", "description": "…", "inputSchema": {"jsonSchema": { … }}} ],
  "tool_choice": "auto",
  "token_count": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "is_valid": true
}
```

Turns without an assistant choice (e.g. upstream errors) are skipped. The flag
works on `launch` and on `serve` with a route bundle; `serve --config` (the Rust
profile server) rejects it, since it has no Python processor chain to attach to.

### Transport (server verbs)

| Flag | Purpose |
|---|---|
| `--host HOST` | Host to bind to (default: `0.0.0.0`). |
| `--port PORT` / `-p PORT` | Port to bind to (default: `4000`, or `server.port` from `secrets.json`). |
| `--reload` | Enable uvicorn auto-reload. |
| `--workers N` / `-w N` | Number of uvicorn worker processes (default: `1`, or `$SWITCHYARD_WORKERS`). |

## Resolution waterfalls

### API-key resolution

Launcher and verify flows resolve the API key in this order, stopping at the
first non-empty value:

1. `--api-key` on the CLI
2. `$OPENROUTER_API_KEY`
3. `$NVIDIA_API_KEY`
4. `$OPENAI_API_KEY`
5. `$ANTHROPIC_API_KEY` (Claude/OpenClaw launchers and verify only)
6. `~/.config/switchyard/credentials.json` → the provider entry written by `configure` (launchers only)
7. `secrets/secrets.json` → first provider section with `api_key` set, with `openrouter` then `nvidia` checked first

For OpenRouter, set `OPENROUTER_API_KEY`; `OPENROUTER_BASE_URL` is optional
because the built-in default is `https://openrouter.ai/api/v1`.

### Base-URL resolution

1. `--base-url` on the CLI
2. The base URL matching the selected environment credential:
   `$OPENROUTER_BASE_URL`, `$NVIDIA_BASE_URL`, or `$OPENAI_BASE_URL`
3. `~/.config/switchyard/config.json` → the selected provider written by
   `configure` (launchers only)
4. `secrets/secrets.json` → same section traversal as the API key
5. Default: OpenRouter (`https://openrouter.ai/api/v1`)

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

Serve a long-running proxy from a Switchyard v2 **profile config**: one YAML/JSON/TOML file declaring `endpoints`, `targets`, and `profiles`. Files whose profiles are all Rust-defined use the Rust profile server. Files that include Python-defined profiles use the Python FastAPI adapter, while keeping the same endpoint paths. Each profile id and each target id is exposed as a model on `GET /v1/models`, so a client selects a profile by setting the request `model` to that id.

The server exposes the OpenAI Chat Completions (`/v1/chat/completions`), Anthropic Messages (`/v1/messages`), and OpenAI Responses (`/v1/responses`) APIs on the same host and port.

**Synopsis**

```text
switchyard [--routing-profiles PATH] serve [--config PATH]
                 [--host HOST] [--port PORT] [--workers N]
                 [--reload] [--inbound FORMAT]
                 [--intake-enabled|--enable-intake [INTAKE OVERRIDES]]
```

**Flags**

| Flag | Source |
|---|---|
| `--routing-profiles PATH` / `-c PATH` | Deprecated legacy [Routing](#routing) path. Global flag; pass before `serve`. Falls back to the saved path from `switchyard --routing-profiles PATH -- configure`. |
| `--config PATH` | Switchyard v2 profile-config YAML/JSON/TOML entrypoint. This is the primary serve path. Mutually exclusive with `--routing-profiles`. |
| `--host`, `--port`/`-p`, `--reload` | [Transport](#transport-server-verbs). |
| `--inbound FORMAT` | Valid only for legacy route-bundle serve (`--routing-profiles`); `serve --config` actively rejects it with an error. For legacy serve, the flag is a no-op — all request APIs are always registered regardless of the value (accepted for backwards compat only). |
| `--workers` / `-w` | uvicorn worker count. |
| `--intake-enabled` / `--enable-intake`, `--intake-base-url`, `--intake-workspace`, `--intake-api-key`, `--intake-nvdataflow-project` | [Intake sink](#intake-sink-serve-and-launchers). |

**Notes**

- `serve` always registers `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/responses`, `GET /v1/models`, and `GET /health`. There is no flag to expose just one request API.
- `GET /v1/stats` and `GET /v1/routing/stats` are available on both serve paths.
- The deprecated route-bundle path accepts `--inbound` for compatibility but ignores it; all supported request APIs are always registered.
- `serve --config` does not support `--reload`, `--workers > 1`, Intake options, `--enable-rl-logging`, or any explicit `--inbound` value.
- Rust-defined and Python-defined profiles use the same profile-config schema. The profile `type` decides which implementation builds that profile.
- Python-defined profiles are registered via `@profile_config`; the shipped `header-routing` profile is an example. A config can mix a Rust-defined profile and a Python-defined profile, and both profile ids are routable on the same served host and port.

**Examples**

```bash
# v2 profile config (primary): each profile id + target id is a model on /v1/models
switchyard serve --config examples/profiles.yaml --port 4000
# select a profile by id (the `model` field):
curl localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "smart-cascade", "messages": [{"role": "user", "content": "hi"}]}'

# mixed Rust/Python profile config on the same request APIs
switchyard serve --config examples/python_profile.yaml --port 4000
curl localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-switchyard-tier: strong' \
  -d '{"model": "smart", "messages": [{"role": "user", "content": "hi"}]}'

# Legacy route bundle on port 4000
switchyard --routing-profiles routes.yaml -- serve --port 4000

# Use the bundle previously persisted by `switchyard --routing-profiles ... -- configure`
switchyard --routing-profiles routes.yaml -- configure
switchyard serve --port 4000

# Multi-worker uvicorn (route-bundle path)
SWITCHYARD_WORKERS=4 switchyard --routing-profiles routes.yaml -- serve
```

## `switchyard launch claude`

Start a proxy on a free local port, spawn `claude` against it, and tear the proxy down when Claude exits.

**Synopsis**

```text
switchyard [--routing-profiles PATH] launch claude [--model ID]
                         [--base-url URL] [--api-key VALUE]
                         [--port PORT] [--timeout SECONDS]
                         [--intake-enabled [INTAKE OVERRIDES]]
                         [--reconfigure] [--dry-run] [--smoke]
                         [--no-tui] [--no-model-discovery]
                         [-- CLAUDE_ARGS...]
```

When neither CLI routing flag is passed, `launch claude` uses a saved routing bundle
first, then a saved `claude.model` as single-model passthrough. The built-in
LLM-as-classifier route is the default only when neither a bundle nor a single-model
default is resolved.

**Flags**

| Flag | Source |
|---|---|
| `--model`, `--routing-profiles` | [Routing](#routing). |
| `--api-key`, `--base-url` | [Credentials](#credentials-and-endpoint). |
| `--weak-model`, `--classifier-model`, `--profile`, `--classifier-min-confidence` | Tier overrides for the built-in LLM-as-classifier route. They are ignored when route resolution selects an explicit or saved bundle or single model. |
| `--intake-enabled` and overrides | [Intake sink](#intake-sink-serve-and-launchers). |
| `--port PORT` | Proxy port (default: auto-pick free port). |
| `--timeout SECONDS` | Request timeout for the backend LLM client. |
| `--reconfigure` | Run Claude setup before launching, even if defaults already exist. |
| `--dry-run` | Print resolved launch settings without starting the proxy or Claude. |
| `--smoke` | Start the proxy, run one `claude -p "<smoke>" --max-turns 1` round-trip, assert exit `0`, and exit. Requires `--model`; cannot be combined with `--routing-profiles`. |
| `--no-tui` | First-run setup: use plain prompts instead of the TUI selector. |
| `--no-model-discovery` | First-run setup: skip `GET /models` and type the model manually. |
| `CLAUDE_ARGS` | Anything after `--` is forwarded verbatim to `claude`. |

**Env vars set on the spawned `claude` process**

- `ANTHROPIC_BASE_URL`: pointed at the local proxy
- `ANTHROPIC_AUTH_TOKEN`: opaque placeholder; skips Console OAuth
- `ANTHROPIC_API_KEY`: set to `""` to silence the auth-conflict warning
- `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL`: pre-selected so Claude's `/model` picker shows your model
- `ANTHROPIC_CUSTOM_MODEL_OPTION`: registers the model in Claude Code's custom model UI entry
- `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`: tells Claude Code to call `GET /v1/models` on the proxy so the full registered route list appears in the picker

**Examples**

```bash
# Single-model passthrough
switchyard launch claude --model openai/gpt-4o-mini

# Use a routing bundle
switchyard --routing-profiles ~/.config/switchyard/profiles.yaml -- launch claude

# One-shot smoke test
switchyard launch claude --smoke --model openai/gpt-4o-mini

# Forward args to the underlying claude binary
switchyard launch claude --model openai/gpt-4o-mini -- --version
```

## `switchyard launch codex`

Start a proxy and spawn OpenAI Codex CLI against it. Codex talks to the proxy via
the OpenAI Responses API (`/v1/responses`). For single-model launches,
Switchyard probes the upstream and uses native Anthropic Messages, OpenAI
Responses, or Chat Completions according to the provider's capabilities.

**Synopsis**

```text
switchyard [--routing-profiles PATH] launch codex [--model ID]
                        [--base-url URL] [--api-key VALUE]
                        [--port PORT] [--timeout SECONDS]
                        [--intake-enabled [INTAKE OVERRIDES]]
                        [--reconfigure] [--dry-run] [--smoke]
                        [--no-tui] [--no-model-discovery]
                        [-- CODEX_ARGS...]
```

When neither CLI routing flag is passed, `launch codex` uses a saved routing bundle
first, then a saved `codex.model` as single-model passthrough. The built-in
LLM-as-classifier route is the default only when neither a bundle nor a single-model
default is resolved. `--weak-model`, `--classifier-model`, `--profile`, and
`--classifier-min-confidence` tune only that built-in route. `CODEX_ARGS` are forwarded
verbatim to `codex`.

**Provider override on the spawned `codex` process**

`launch codex` injects a transient `switchyard` provider into Codex via repeated `-c` flags. No edits to `~/.codex/config.toml` are required.

**Examples**

```bash
switchyard launch codex --model openai/gpt-4o-mini
switchyard --routing-profiles routes.yaml -- launch codex
switchyard launch codex --smoke --model openai/gpt-4o-mini
switchyard launch codex --model openai/gpt-4o-mini -- exec "explain this file"
```

## `switchyard launch openclaw`

For [OpenClaw](https://github.com/openclaw/openclaw). OpenClaw talks to the proxy via OpenAI Chat Completions (`/v1/chat/completions`); the chain translates as needed for the upstream backend.

**Synopsis**

```text
switchyard [--routing-profiles PATH] launch openclaw [--model ID]
                           [--base-url URL] [--api-key VALUE]
                           [--port PORT] [--timeout SECONDS]
                           [--intake-enabled [INTAKE OVERRIDES]]
                           [--reconfigure] [--dry-run] [--smoke]
                           [--no-tui] [--no-model-discovery]
                           [-- OPENCLAW_ARGS...]
```

When neither CLI routing flag is passed, `launch openclaw` uses a saved routing bundle
first, then a saved `openclaw.model` as single-model passthrough. The built-in
LLM-as-classifier route is the default only when neither a bundle nor a single-model
default is resolved. `--weak-model`, `--classifier-model`, `--profile`, and
`--classifier-min-confidence` tune only that built-in route.

The launcher spawns `openclaw chat`, an alias for `openclaw tui --local`, which is OpenClaw's interactive local terminal UI bound to the embedded agent runtime. `OPENCLAW_ARGS` are forwarded after `chat`, so pass `chat`-compatible flags (`--message`, `--thinking`, `--session`, etc.). `openclaw agent` is a non-interactive one-shot turn and is used only by the `--smoke` path.

**Provider override on the spawned `openclaw` process**

`launch openclaw` writes a transient `openclaw.json` to a temporary directory and points OpenClaw at it via `OPENCLAW_STATE_DIR` / `OPENCLAW_HOME` / `OPENCLAW_CONFIG_PATH`. The transient config declares a `models.providers.switchyard` block with `api: "openai-completions"`, `baseUrl` pointing at the proxy, and `apiKey: "${SWITCHYARD_API_KEY}"` (the launcher sets `SWITCHYARD_API_KEY=switchyard`, an opaque placeholder; the proxy ignores inbound auth). The user's real `~/.openclaw/` (sessions, channels, plugins) is **untouched** for the duration of the launch.

The tempdir is removed when `openclaw` exits, including on Ctrl-C.

**Examples**

```bash
switchyard launch openclaw --model openai/gpt-4o-mini
switchyard --routing-profiles routes.yaml -- launch openclaw
switchyard launch openclaw --smoke --model openai/gpt-4o-mini
switchyard launch openclaw --model openai/gpt-4o-mini -- --message "Hello"
```

## `switchyard configure`

Persist user-level Switchyard defaults under `~/.config/switchyard/`. Credentials are stored separately from non-secret config, with owner-only file permissions.
Skill distillation config also lives in this file under `skill_distillation` and can be updated without configuring provider credentials.

**Synopsis**

```text
switchyard [--routing-profiles PATH] configure [--show [--check] [--json] | --reset | --list-models]
                     [--target {all,provider,claude,codex,openclaw}]
                     [--query SUBSTRING] [--limit N]
                     [--provider ID]
                     [--base-url URL] [--api-key VALUE]
                     [--claude-model ID]   [--claude-base-url URL]   [--claude-api-key VALUE]
                     [--codex-model ID]    [--codex-base-url URL]    [--codex-api-key VALUE]
                     [--openclaw-model ID] [--openclaw-base-url URL] [--openclaw-api-key VALUE]
                     [--skill-distillation NAMESPACE] [--disable-skill-distillation]
                     [--no-model-discovery] [--no-tui]
```

**Modes (mutually exclusive)**

| Flag | What it does |
|---|---|
| _(none)_ | Interactive setup. Prompts for any missing default; runs the model-discovery TUI unless `--no-model-discovery`. |
| `--show` | Print the redacted saved config plus a resolution snapshot: resolved provider, base URL, API-key source, saved Claude / Codex defaults, routing-profile summary, and paths to the `claude` and `codex` harness binaries. Pair with `--check` for a live `GET /models` probe, or `--json` to emit only the raw redacted JSON snapshot. |
| `--reset` | Delete persisted user config and credentials. |
| `--list-models` | Fetch `GET /models` from the resolved provider and print a ranked, searchable list. Pair with `--target {claude,codex}` for launcher-targeted ranking, `--query` to filter by substring, `--limit` to cap results. |

**Configuration knobs**

| Flag | Purpose |
|---|---|
| `--target` | For setup: which defaults to write (`all` (default), `provider`, `claude`, `codex`, or `openclaw`). For `--list-models`: ranking target (`all`, `claude`, `codex`, or `openclaw`; `provider` is accepted and treated as `all`). |
| `--provider`, `--base-url`, `--api-key` | Provider-level defaults applied to every launcher. Also act as one-off overrides for `--show` (changes the row that's used to resolve "base URL source" and "API key source") and for the `--list-models` discovery call. |
| `--claude-*` / `--codex-*` / `--openclaw-*` | Per-launcher overrides on top of the provider defaults. |
| `--routing-profiles PATH` | Global flag; pass before `configure`. Parses the YAML at `PATH` and stores the parsed bundle inline in `~/.config/switchyard/config.json`. Subsequent `serve` and `launch` runs use this when no `--routing-profiles` is on the CLI. Pass an empty string to clear. |
| `--skill-distillation NAMESPACE` | Configure a safe local path component for future skill-distillation trace collection. This stores the namespace only; capture, drafting, and activation are not implemented by this config path. |
| `--disable-skill-distillation` | Remove the saved skill distillation config. Cannot be combined with `--skill-distillation`. |
| `--query` / `-q SUBSTRING` | With `--list-models`, case-insensitive substring filter. |
| `--limit N` | With `--list-models`, cap on the number of models printed (default: 50; pass `0` for unlimited). |
| `--no-model-discovery` | Skip `GET /models` and rely on explicit or existing model values during interactive setup. |
| `--no-tui` | Use plain text prompts instead of the TUI selector. |
| `--check` | With `--show`, call `GET /models` against the resolved provider and report pass/fail in the output. |

**Skill distillation config**

```json
{
  "skill_distillation": {
    "namespace": "tooluniverse-trialqa"
  }
}
```

Namespaces must be a single safe path component: letters, numbers, dot, underscore, and hyphen only.
The top-level key is omitted when skill distillation is not configured. `namespace` is the only supported key today; any extra manually edited keys are rejected instead of being treated as dormant product behavior.

**Examples**

```bash
# First-run interactive setup
switchyard configure

# Save a routing bundle as the default for serve + launchers
switchyard --routing-profiles routes.yaml -- configure

# Inspect what's stored, plus resolved provider / key source / harness paths
switchyard configure --show

# Inspect plus live GET /models probe
switchyard configure --show --check

# Raw redacted JSON, e.g. for tooling
switchyard configure --show --json

# One-off override for a probe (without persisting anything)
switchyard configure --show --provider openrouter --api-key "$OPENROUTER_API_KEY" --base-url https://openrouter.ai/api/v1 --check

# Browse the backend's models
switchyard configure --list-models --target claude --query gpt
switchyard configure --list-models --limit 0 --provider openrouter --api-key "$OPENROUTER_API_KEY" --base-url https://openrouter.ai/api/v1

# Set just the Claude default model non-interactively
switchyard configure --target claude --claude-model openai/gpt-4o-mini --no-tui

# Non-interactive / CI: save provider credentials only (no launcher models required)
switchyard configure --target provider --provider openrouter \
  --api-key "$OPENROUTER_API_KEY" --base-url https://openrouter.ai/api/v1 \
  --no-tui --no-model-discovery

# Enable skill distillation config without provider credentials
switchyard configure --skill-distillation tooluniverse-trialqa

# Disable skill distillation config without touching provider credentials
switchyard configure --disable-skill-distillation

# Wipe everything
switchyard configure --reset
```

!!! note "Non-interactive / CI usage"
    **`--target all` (the default) requires launcher models in no-TTY mode.** Without a
    TTY or `--claude-model` / `--codex-model` / `--openclaw-model`, the command exits with
    `No Claude model configured or discovered`. Pass `--target provider` to save only
    provider credentials, or supply explicit `--claude-model` / `--codex-model` / `--openclaw-model` values.

    **`configure` requires an explicit `--api-key` flag.** It does not read `OPENROUTER_API_KEY`
    (or any other `*_API_KEY` environment variable) and does not read `api_key` from a
    routing-profile bundle (`--routing-profiles`). Always pass `--api-key` when running
    non-interactively.

## `switchyard verify`

Sequenced pass/fail checklist that confirms a Switchyard install works end-to-end against a real backend, without spawning a harness binary. Fast (~1–3s on a healthy stack); suitable for K8s readiness probes and pre-deployment smoke tests.

For harness-driven smoke tests (proxy + spawn `claude` / `codex` / `openclaw` once with a fixed prompt), use [`launch claude --smoke`](#switchyard-launch-claude) / [`launch codex --smoke`](#switchyard-launch-codex) / [`launch openclaw --smoke`](#switchyard-launch-openclaw) instead.

**Synopsis**

```text
switchyard verify [--model ID] [--base-url URL] [--api-key VALUE]
                  [--port PORT] [--timeout SECONDS]
```

**Example model**

`openai/gpt-4o-mini` is a portable OpenRouter example. Pass `--model` when
your provider uses a different model ID.

**Checklist**

1. Resolve credentials (CLI → env → `secrets.json`).
2. Reach the backend via `GET /models`.
3. Probe `/v1/chat/completions`, `/v1/messages`, and `/v1/responses` support (informational; informs `BackendFormat.AUTO`).
4. Start a proxy on a free port.
5. Round-trip a chat completion through the chain.
6. Tear the proxy down.

**Exit codes**

- `0`: every step passed.
- Non-zero: first failing step; the error message names the source it tried and what to fix.

**Examples**

```bash
switchyard verify
switchyard verify --model openai/gpt-4o-mini
switchyard verify --api-key "$OPENROUTER_API_KEY" --base-url https://openrouter.ai/api/v1
```

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Backend credentials. Resolved in this order by launchers and `verify`; `ANTHROPIC_API_KEY` is only consulted where Anthropic input is supported. |
| `OPENROUTER_BASE_URL`, `NVIDIA_BASE_URL`, `OPENAI_BASE_URL` | Backend base URL overrides paired with the selected provider credential. |
| `OPENAI_API_BASE` | Legacy alias for `OPENAI_BASE_URL`. Consulted as a last fallback when neither `--base-url` nor `OPENAI_BASE_URL` is set. Prefer `OPENAI_BASE_URL` for new configurations. |
| `SWITCHYARD_INTAKE_ENABLED` | Boolean equivalent of `--intake-enabled` / `--enable-intake`. Set to `1` or `true` to enable the intake sink without a CLI flag. Precedence: CLI flag first, then this env var. |
| `SWITCHYARD_NVDATAFLOW_PROJECT` | NVDataflow project name for the alternate intake sink (paired with `--intake-nvdataflow-project`). Precedence: CLI flag first, then this env var. |
| `SWITCHYARD_WORKERS` | Default uvicorn worker count for `serve`. |
| `SWITCHYARD_TELEMETRY_OPT_OUT` | Disable the `X-Switchyard-Version` telemetry header on outbound calls. `NEMO_SWITCHYARD_TELEMETRY_OPT_OUT` is honored for backwards compatibility. |
| `SWITCHYARD_INTAKE_BASE_URL`, `SWITCHYARD_INTAKE_WORKSPACE`, `SWITCHYARD_INTAKE_API_KEY`, `SWITCHYARD_INTAKE_APP`, `SWITCHYARD_INTAKE_TASK`, `SWITCHYARD_SESSION_ID`, `SWITCHYARD_USER_ID` | Intake-sink overrides for CI / headless runs. Precedence: the matching CLI flag (e.g. `--intake-user-id`) first, then the env var, then any persisted / SDK default (e.g. `~/.switchyard/user_id`). |
| `NMP_ACCESS_TOKEN` | Fallback bearer token for the intake sink when the NMP SDK config is not present. |

## See also

- [Getting Started](getting_started.md): install Switchyard and run your first request.
- [Architecture](architecture.md): system context and end-to-end request flow.
