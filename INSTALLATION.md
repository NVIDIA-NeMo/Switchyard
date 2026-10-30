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

## Standalone Server

Install the native Rust proxy from crates.io:

```bash
cargo install --locked switchyard-server
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml --port 4000
```

See [Local Evaluation](docs/operations/local_evaluation.md) for a complete TOML
example and [`switchyard-server`](crates/switchyard-server/README.md) for the
configuration reference.

## Experimental desktop app

The desktop app, terminal app, and their installer live in
`experimental/switchyard-desktop`. They are outside Switchyard release QA and
have no support or compatibility guarantee. The installer also sets up the
server as a user service, saved server keys, and the Linux Codex service. See
the [desktop installation guide](experimental/switchyard-desktop/INSTALLATION.md).

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

See the experimental [desktop and terminal quickstart](experimental/switchyard-desktop/README.md) for
first-run setup, reviewed settings changes, usage, recovery, and safe uninstall.
