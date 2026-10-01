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

See [Getting Started](docs/getting_started.md#server-path) for a complete TOML
deployment and [`switchyard-server`](crates/switchyard-server/README.md) for the
configuration reference.

## Linux Codex Service

This setup is for single-user Linux machines. It requires a systemd user
session, the Rust toolchain listed above, and Codex CLI 0.134.0 or newer
logged in with a ChatGPT account. Another user's process could take the local
port while the service is stopped and receive your login and prompts.

From a checkout, preview or install the service:

```bash
make install-linux-dry-run
make install-linux
systemctl --user status switchyard
codex -p sy
```

The installer builds and installs `~/.switchyard/bin/switchyard-server`.
It creates `~/.switchyard/composite.toml` if missing and keeps existing edits.
It replaces `~/.config/systemd/user/switchyard.service` and restarts the service.
It writes `~/.codex/sy.config.toml`, backing up a changed profile first.
It leaves shell rc files unchanged. Use `codex -p sy` to select the profile;
`codex login` and other management commands still work as usual.
The profile format requires [Codex 0.134.0 or newer](https://developers.openai.com/codex/config-advanced#profiles).
Remove any old `[profiles.sy]` table from `~/.codex/config.toml` before using it.

Set `SY_HOME` or `SY_PORT` to change the server directory or port (default 4123).
`SY_HOME` must not contain whitespace, control characters, or a trailing
backslash; `SY_PORT` must contain only digits. `XDG_CONFIG_HOME` and `CODEX_HOME`
set the systemd and Codex config directories.

Read server logs with `journalctl --user -u switchyard`. Routing records are
stored in `~/.switchyard/routing.jsonl`. Use `systemctl --user edit switchyard`
for service changes that survive reinstalling.

Remove the service and profile with `make uninstall-linux`. This also removes
marked Codex aliases left by older installs from existing `.bashrc` and
`.zshrc` files. If upgrading an older install, uninstall first and run
`unalias codex` in any open shell. The server directory, routing records,
profile backups, and systemd drop-in files stay in place. Use the same path
overrides when installing and uninstalling.

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
