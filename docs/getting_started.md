# Getting Started with Switchyard

Switchyard has three native Rust execution paths:

- **Launcher path:** install the Python-distributed CLI and launch Claude Code,
  Codex, or OpenClaw through the packaged Rust server binding.
- **Server path:** build and run the standalone Rust server for API clients and
  custom deployments.
- **Library path:** embed the routing algorithms directly in your own Rust
  application with `switchyard-libsy`.

## Launcher Path

Use this path when you want Switchyard to start and configure a supported coding
agent for you.

### Install the CLI

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if it is
not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Then install the published Switchyard tool:

```bash
uv tool install --python 3.10 "nemo-switchyard[cli]"
```

This creates an isolated Python tool environment containing the `switchyard`
CLI, its launcher dependency, and the packaged PyO3 Rust extension.
`switchyard launch` starts the native Rust server through that extension; this
path does not install or run the standalone `switchyard-server` binary.

Install the coding agent you want to launch, then verify both commands are on
your `PATH`.

### Launch with the packaged OpenRouter deployment

The packaged deployment exposes the route ID `switchyard`. Export an OpenRouter
key and choose an agent:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
switchyard launch claude --model switchyard
```

Codex and OpenClaw use the same deployment:

```bash
switchyard launch codex --model switchyard
switchyard launch openclaw --model switchyard
```

### Launch with a custom deployment

Pass a native TOML deployment and select one of its route IDs:

```bash
switchyard launch codex --model my-route --config routes.toml
```

The launcher manages the native server lifecycle and points the selected coding
agent at it. The server configuration format is the same TOML schema used by the
standalone server below.

## Server Path

Use this path when you want a standalone proxy for API clients or need to
operate the Rust server directly.

### Prerequisites

- Git, a native build toolchain, and Rust with Cargo
- An API key for OpenRouter, OpenAI, Anthropic, or another OpenAI-compatible endpoint.
  To use OpenRouter, create an account at [openrouter.ai](https://openrouter.ai/)
  and generate a key from the [OpenRouter keys page](https://openrouter.ai/keys).

On Ubuntu or WSL, install the build prerequisites and Rust with `rustup`:

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

On macOS or native Windows, follow the
[official Rust installation instructions](https://rust-lang.org/tools/install/).
The Rust installer includes `rustc`, Cargo, and `rustup`.

Install `uv` for the repository's Python-based tooling and CI checks. It is not
required to build or run the Rust server:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If either installer updates your shell configuration, restart the shell before
continuing. Verify the tools:

```bash
git --version
rustc --version
cargo --version
uv --version
```

### Install the server

Install the Rust server from crates.io:

```bash
cargo install --locked switchyard-server
switchyard-server --help
```

Cargo builds the release binary and installs it into `~/.cargo/bin` by default.

### Configure

The Rust server reads an explicit TOML file.

Create `routes.toml` with an LLM-classifier route:

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.weak]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[targets.strong]
id = "openai/gpt-4o"
llm_client = "openrouter"

[routes.smart]
id = "switchyard"
type = "llm_classifier"
mode = "capability"
classifier_target = "weak"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5
```

`format` selects the upstream protocol and must be `openai_chat`,
`openai_responses`, or `anthropic_messages`. `api_key_env` names the environment
variable the server reads; the secret does not belong in the TOML file.

### Run the server

Export the provider credential, validate the configuration without binding a
socket, then start the release binary:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml \
  --host 127.0.0.1 --port 4000
```

Any client that speaks OpenAI Chat Completions, Anthropic Messages, or OpenAI
Responses API can connect. The route `id` is the model name clients use.

In another terminal:

```bash
curl http://localhost:4000/health
curl http://localhost:4000/v1/models
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"switchyard","messages":[{"role":"user","content":"hello"}]}'
```

### Routing algorithms

#### Choose a route type

This guide uses `llm_classifier`, which asks a classifier target whether each
request should use the weak or strong target. The Rust server also supports:

| Algorithm | Use it when | Config |
|---|---|---|
| [Random](routing_algorithms/random_routing.md) | You need a weighted split for A/B tests or baselines. | `random` |
| [LLM classifier](routing_algorithms/llm_classifier_routing.md) | Request content should decide whether to use the weak or strong target. | `llm_classifier` |
| [Stage router](routing_algorithms/stage_router_routing.md) | Tool-result and progress signals should select an efficient or capable target. | `stage_router` |

A single TOML file can declare multiple routes. The table key, such as
`routes.smart`, is a local configuration name; each route's `id` is exposed as a
model on `GET /v1/models`.

See [Routing Overview](routing_algorithms/overview.md) to compare strategies,
and the [`switchyard-server` guide](../crates/switchyard-server/README.md) for
the complete TOML schema, route options, TLS, and metrics.

### Troubleshooting

**No API key / auth error**

```bash
test -n "$OPENROUTER_API_KEY" && echo "key is set" || echo "key is missing"
switchyard-server --config routes.toml --dry-run
```

Confirm that `api_key_env` in `routes.toml` names the environment variable you
exported. The dry run validates the schema, environment lookup, target
references, and route construction without starting the server.

**Connection refused**

Check health: `curl http://localhost:4000/health`

**Telemetry header opt-out**

Switchyard adds an `X-Switchyard-Version` header to outbound LLM calls for
release attribution. No request or response content is included. To disable:

```bash
export SWITCHYARD_TELEMETRY_OPT_OUT=1
```

---

## Library Path

Use this path when you want routing inside your own Rust application rather than
behind a proxy. `switchyard-libsy` never calls a model itself: an algorithm
picks a target and hands the model call back to you.

### Add the dependencies

```toml
[dependencies]
async-trait = "0.1"
futures = "0.3"
switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
tokio = { version = "1", features = ["macros", "rt"] }
```

### Choose an algorithm

| Type | Purpose |
|---|---|
| `LlmTaskClassifier` | Ask a judge model to choose an efficient or capable target. |
| `StageRouter` | Route from signals already in the conversation, such as tool results and errors, with an optional judge fallback. |
| `LlmTaskClassifier` with escalation | Every turn runs on the efficient target first, and a judge reads that answer to decide whether to send the same request to the capable target. |
| `Random` | Select among any number of targets, uniform or weighted. |

These are the same strategies the server exposes as route types, so a deployment
can move between the server and library paths without changing routing
behaviour.

### Drive the algorithm

An algorithm yields a stream of steps. Each `Step::CallModel` is a model call your
host performs over its own transport, and the run ends with
`Step::Done` carrying the final response. Serving those calls yourself
is what lets libsy embed in a host that already owns its HTTP stack, retries,
and credentials.

For the request, response, and streaming types the steps carry, see
[`switchyard-protocol`](../crates/protocol/README.md).

---

## Next steps

- [Core Concepts](core_concepts.md): LLM clients, targets, and routes
- [`switchyard-server`](../crates/switchyard-server/README.md): server configuration,
  routing algorithms, TLS, and metrics
- [Rust API reference](reference/rust_api.md): generated libsy and protocol
  documentation, crate setup, and API boundaries
- [`switchyard-translation`](../crates/switchyard-translation/README.md):
  request, response, and stream translation
