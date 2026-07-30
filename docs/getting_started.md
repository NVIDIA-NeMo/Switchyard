# Getting Started with Switchyard

## Prerequisites

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

## Install

Build the Rust server from source:

```bash
git clone https://github.com/NVIDIA-NeMo/Switchyard.git
cd Switchyard
cargo build --locked --release -p switchyard-server
./target/release/switchyard-server --help
```

The repository pins Rust `1.96.1` in `rust-toolchain.toml`; `rustup` selects and
installs it automatically when Cargo runs from the repository. Prebuilt Rust
binaries are not published yet.

## Configure

The Rust server reads an explicit TOML file. It does not use the legacy Python
CLI's saved configuration or YAML routing profiles.

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
classifier_target = "weak"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5
```

`format` selects the upstream protocol and must be `openai_chat`,
`openai_responses`, or `anthropic_messages`. `api_key_env` names the environment
variable the server reads; the secret does not belong in the TOML file.

## Server mode

Export the provider credential, validate the configuration without binding a
socket, then start the release binary:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
./target/release/switchyard-server --config routes.toml --dry-run
./target/release/switchyard-server --config routes.toml \
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

## Routing algorithms

### Choose a route type

This guide uses `llm_classifier`, which asks a classifier target whether each
request should use the weak or strong target. The Rust server also supports:

| Algorithm | Use it when | Config |
|---|---|---|
| Random | You need a weighted split for A/B tests or baselines. | `random` |
| LLM classifier | Request content should decide whether to use the weak or strong target. | `llm_classifier` |
| Stage router | Tool-result and progress signals should select an efficient or capable target. | `stage_router` |

A single TOML file can declare multiple routes. The table key, such as
`routes.smart`, is a local configuration name; each route's `id` is exposed as a
model on `GET /v1/models`.

See the [`switchyard-server` guide](../crates/switchyard-server/README.md) for
the complete TOML schema, route options, TLS, and metrics.

<!-- TODO: Restore routing-guide links after those pages use the Rust TOML schema. -->

<!-- TODO: Add Python library usage back when the supported Rust-backed API is finalized. -->

## Troubleshooting

**No API key / auth error**

```bash
test -n "$OPENROUTER_API_KEY" && echo "key is set" || echo "key is missing"
./target/release/switchyard-server --config routes.toml --dry-run
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

<!-- TODO: Add contributor setup back when the documented workflow is Rust-first. -->

---

## Next steps

- [`switchyard-server`](../crates/switchyard-server/README.md): server configuration,
  routing algorithms, TLS, and metrics
- [`switchyard-libsy`](../crates/libsy/README.md): embed routing algorithms in a
  Rust application
- [`switchyard-protocol`](../crates/protocol/README.md): provider-neutral
  request, response, and streaming types
- [`switchyard-translation`](../crates/switchyard-translation/README.md):
  request, response, and stream translation

<!-- TODO: Restore architecture, CLI, and routing links when those pages are Rust-first. -->
