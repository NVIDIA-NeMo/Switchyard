# Installation Guide

Switchyard has separate packages for Python integrations and standalone Rust
serving.

## Requirements

- Python 3.10 or newer for `nemo-switchyard`
- Rust 1.96.1 or newer for `switchyard-server` and the Rust libraries
- Linux x86_64 wheels require an x86-64-v3 / AVX2-class CPU
- Linux aarch64 wheels require a Neoverse N1-class CPU

## Python Bindings

Install the Python package to embed libsy algorithms or host the native server
through PyO3:

```bash
pip install nemo-switchyard
```

The base package has no Python runtime dependencies. Its native extension owns
the libsy and server implementations.

## Coding-Agent Launchers

Install the CLI extra to launch Claude Code, Codex CLI, or OpenClaw through the
packaged native server:

```bash
uv tool install --python 3.10 "nemo-switchyard[cli]"
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
switchyard launch claude --model switchyard
```

The selected coding agent must already be installed and available on `PATH`.
Use `--config routes.toml` to select a custom native TOML deployment.

## Standalone Server

Install the native Rust proxy from crates.io:

```bash
cargo install --locked switchyard-server
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml --port 4000
```

See [Getting Started](docs/getting_started.md#server-path) for a complete TOML
deployment and [`switchyard-server`](crates/switchyard-server/README.md) for the
configuration reference.

### Container image

Build a production image from this repository (matches the checked-out commit).
The Dockerfile lives under `deploy/docker/`; use the repo root as the build
context:

```bash
docker build -f deploy/docker/Dockerfile -t switchyard-server:local .
docker run --rm -p 4000:4000 \
  -e API_KEY \
  -v "$PWD/deploy/docker/routes.passthrough.toml:/etc/switchyard/routes.toml:ro" \
  switchyard-server:local
```

`deploy/docker/Dockerfile` compiles `switchyard-server` with `cargo build
--locked --release`. Replace the example routes file and `API_KEY` with your
upstream. `benchmark/switchyard-rust-server.Dockerfile` remains available for
benchmark pipelines that prefer a smaller context.

### Helm chart

A minimal chart lives under `deploy/helm/switchyard`:

```bash
kubectl create secret generic switchyard-api-key --from-literal=API_KEY=...
helm upgrade --install switchyard deploy/helm/switchyard \
  --set image.repository=switchyard-server \
  --set image.tag=local
kubectl port-forward svc/switchyard 4000:4000
curl -sS http://127.0.0.1:4000/health
```

Override `routesConfig` for a real deployment. Optional NetworkPolicy support is
documented in `deploy/helm/switchyard/README.md`.

## Rust Libraries

Add the crates needed by an embedded application:

```toml
[dependencies]
switchyard-libsy = "0.2.0"
switchyard-protocol = "0.2.0"
switchyard-llm-client = "0.2.0"
switchyard-translation = "0.2.0"
```

`switchyard-libsy` owns algorithms, `switchyard-protocol` owns provider-neutral
request and response types, `switchyard-translation` owns wire conversion, and
`switchyard-llm-client` performs translated HTTP calls.

## Development

From a checkout:

```bash
uv sync
uv run maturin develop
cargo test --workspace
uv run pytest tests/ -v
```

The `dev` dependency group contains testing and linting tools and is not exposed
in the published wheel metadata.
