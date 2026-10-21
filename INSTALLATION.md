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

## Server Keys

In a server config, each LLM client and decision client names its API key
variable with `api_key_env`. A server you start from a terminal reads those
variables from your shell, as it always has. The installed server does not: on
macOS launchd starts it, on Linux systemd, and on Windows Task Scheduler. None
of them runs your shell, so the installed server never sees the variables that
`~/.zshrc` or `~/.bashrc` exports.

The installed server reads each key from the system credential store instead:
macOS Keychain, Windows Credential Manager, or the Linux Secret Service (GNOME
Keyring on Ubuntu and GNOME, KDE Wallet on Plasma). A key is saved under its
variable name, such as `NVIDIA_API_KEY`. Before each start, the installed
helper reads the keys that the config names and passes them to the server. A
saved key replaces any value the service manager passes. If a variable has no
saved key, the server does not start, and the server log names the variable.

To save, replace, or remove a key in the desktop app, open **Settings → Server
keys**. Type or paste the key, or choose **Import from shell** to read the
variable once from your login shell. From a terminal on macOS or Linux, run the
installed helper:

```bash
~/.switchyard/bin/switchyard-desktop-install key status --config ~/.switchyard/composite.toml
~/.switchyard/bin/switchyard-desktop-install key set NVIDIA_API_KEY      # prompts without echo
~/.switchyard/bin/switchyard-desktop-install key import NVIDIA_API_KEY   # reads it from your login shell
~/.switchyard/bin/switchyard-desktop-install key remove NVIDIA_API_KEY
```

On Windows, the installer puts the helper in
`%LOCALAPPDATA%\Programs\Switchyard`. There `key set` reads the key from
standard input without a prompt, and `key import` reads the variable from your
Windows environment variables. From PowerShell:

```powershell
$helper = "$env:LOCALAPPDATA\Programs\Switchyard\switchyard-desktop-install.exe"
& $helper key status --config "$env:USERPROFILE\.switchyard\composite.toml"
[Net.NetworkCredential]::new('', (Read-Host 'Key' -AsSecureString)).Password | & $helper key set NVIDIA_API_KEY
& $helper key import NVIDIA_API_KEY
& $helper key remove NVIDIA_API_KEY
```

Restart the server after you change a key. If you rename a variable in
`api_key_env`, save a key under the new name. The old key stays saved until you
remove it in Keychain Access, Credential Manager, or Seahorse.

`cargo desktop install` never saves, reads, or changes a key, even when your
terminal sets the variable. It checks the config without keys, then lists each
variable that has no saved key. On a first install, the server starts after you
save those keys in **Settings → Server keys** and restart it. In the terminal
app, press `7` to open the Settings page. There `k` lists the variables that the
config names and whether each has a saved key, `s` saves a key, and `i` imports
it from your login shell. Press `b` and then `y` to restart the server.
Uninstalling keeps the saved keys, the config, and history. Switchyard never
writes keys to app settings, service definitions, or logs. Model-list keys that
you enter in Routes are stored separately.

On macOS, Keychain ties each key to the helper that saved it. Keychain treats a
rebuilt helper as a new program, so after `cargo desktop install` macOS may ask
once per key whether to allow `switchyard-desktop-install`. Choose **Always
Allow**.

On Linux, the login keyring must be unlocked. A desktop login normally unlocks
it. When the server service starts after login, it waits up to a minute for the
keyring to unlock. Over SSH without a desktop session, or with a window manager
that does not unlock the keyring, the server does not start and its log says
the keyring is locked.

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
profile backups, and systemd drop-in files stay in place. Uninstall uses the
recorded installation. Conflicting environment overrides require
`cargo desktop uninstall --settings FILE` to select it explicitly.

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
can release and replace the installed files; read `SY_HOME\logs\update.log` for
its progress and errors. The installed installer cannot install, update, or
uninstall its own installation, because Windows keeps a running program from
being replaced. Use `cargo desktop` from the source checkout instead.
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
