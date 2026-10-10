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
cargo desktop install --dry-run
cargo desktop install
systemctl --user status switchyard
codex -p sy
```

The installer builds and installs `~/.switchyard/bin/switchyard-server` and
`~/.switchyard/bin/switchyard-desktop`. Run the same command to reinstall from
the current checkout; update the checkout first to install newer code.
Run `~/.switchyard/bin/switchyard-desktop --tui` to open the terminal app.
It creates `~/.switchyard/composite.toml` if missing and keeps existing edits.
It replaces `~/.config/systemd/user/switchyard.service` and restarts the service.
It writes `~/.codex/sy.config.toml`, backing up a changed profile first.
It leaves shell rc files unchanged. Use `codex -p sy` to select the profile;
`codex login` and other management commands still work as usual.
The profile format requires [Codex 0.134.0 or newer](https://developers.openai.com/codex/config-advanced#profiles).
Remove any old `[profiles.sy]` table from `~/.codex/config.toml` before using it.

Set `SY_HOME` or `SY_PORT` to change the server directory or port (default 4123).
`SY_HOME` must be an absolute UTF-8 path without ASCII control characters.
`SY_PORT` must be a number from 1 to 65535. `XDG_CONFIG_HOME` and `CODEX_HOME`
set the systemd and Codex config directories.

Read server logs with `journalctl --user -u switchyard`. Routing records are
stored in `~/.switchyard/routing.jsonl`. Use `systemctl --user edit switchyard`
for service changes that survive reinstalling.

Remove the service and profile with `cargo desktop uninstall`. This also removes
marked Codex aliases left by older installs from existing `.bashrc` and
`.zshrc` files. If upgrading an older install, uninstall first and run
`unalias codex` in any open shell. The server directory, routing records,
profile backups, and systemd drop-in files stay in place. Use the same path
overrides when installing and uninstalling.

## Windows Desktop App

From PowerShell in the repository, run:

```powershell
cargo desktop install --dry-run
cargo desktop install
```

Windows needs the Rust MSVC toolchain, Visual Studio C++ build tools, and
[WebView2](https://v2.tauri.app/start/prerequisites/#windows).
The installer uses Task Scheduler for the current logged-in user. It installs
executables in `%LOCALAPPDATA%\Programs\Switchyard`, falling back to
`%USERPROFILE%\AppData\Local\Programs\Switchyard` when `LOCALAPPDATA` is unset.
It saves that location and adds a Start menu shortcut. Two independent tasks
start the server and desktop app after login with ordinary user privileges.
Quitting the app leaves the server running.

Run `cargo desktop install` again to build and install the current checkout.
This Cargo command waits for installation to finish. The app's source update
uses the saved checkout and runs from a temporary Rust executable so Windows
can release and replace the installed files. Running the installed installer
also starts a temporary executable and returns before installation finishes;
read `SY_HOME\logs\update.log` for progress and errors.
The installer builds and validates before stopping the tasks. If a binary
replacement fails, it tries to restore the previous binaries. If restoration
also fails, the error names the directory containing recovery files.
Rerun installation if a later task or settings operation fails; the complete
installation is not one transaction.

Settings, server config, accounts, and routing history stay in
`%USERPROFILE%\.switchyard` unless `SY_HOME` changes the directory. Server and app
logs are in `SY_HOME\logs`. Model-list keys entered in Routes are saved in the
current user's Windows Credential Manager. Coding tools keep their own login files.

Use `cargo desktop uninstall` to stop and remove both tasks, the shortcut, and
the installed executables. Uninstall preserves settings, accounts, and history,
and makes a recovery copy of the Codex profile before removing it.

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

See the [desktop and terminal quickstart](crates/switchyard-desktop/README.md) for
first-run setup, reviewed settings changes, usage, recovery, and safe uninstall.
