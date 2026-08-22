# [bug] `switchyard-server` lacks cross-platform managed background / supervised mode

> Baseline: `origin/main` @ `053a61e` (post-`#501`). The proxy is now the standalone Rust binary `switchyard-server` (`crates/switchyard-server`); agents reach it over the wire by pointing their base-URL / model env at it.

## Symptom

`switchyard-server` currently runs strictly as a foreground process. There is no supported, first-class mechanism to run the server in a detached, supervised state that:
1. Outlives the launching terminal session across OSs.
2. Is isolated from job-control signals of unrelated foreground tasks.
3. Provides lifecycle management (PID tracking, graceful shutdown via CLI, restart handling).

Operators must hand-roll background process management via OS-specific utilities (`nohup`, `setsid`, systemd, launchd, or NSSM).

## Reproduction

```bash
# Terminal A — foreground server
switchyard-server --config routes.toml --port 4000

# Terminal B — agent running via proxy
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000/v1
export ANTHROPIC_MODEL=switchyard/classified
claude

# Closing Terminal A kills switchyard-server, breaking active agent sessions

```

Currently, `crates/switchyard-server/src/cli.rs` only handles foreground execution with `SIGINT`/`SIGTERM` draining (`--shutdown-timeout`). There is no built-in `--detach` flag, service supervisor integration, or process status/stop subcommands.

## Proposed solution

Implement a cross-platform background execution mode and provide service template files for system process managers.

### 1. Cross-platform `--detach` (Self-Spawn Pattern)

To avoid Async Runtime (Tokio) state corruption caused by POSIX `fork()`/`setsid()` post-initialization, implement a self-executing process spawner prior to booting the async runtime:

* **Unix (Linux/macOS):** Spawns a detached process with `Stdio::null()` for standard descriptors and creates a new session.
* **Windows:** Spawns a background process using `creation_flags` (`CREATE_NO_WINDOW` / `DETACHED_PROCESS`).
* **PID Management:** Automatically writes a `--pidfile <path>` (defaults to OS temp directory or `~/.switchyard/server.pid`).

### 2. Process Control Subcommands

Add process management commands to inspect and stop detached servers cleanly:

* `switchyard-server stop [--pidfile <path>]`: Sends a graceful shutdown signal to the running instance and waits for active requests to drain within `--shutdown-timeout`.
* `switchyard-server status [--pidfile <path>]`: Queries `/health` or checks PID vitality.

### 3. Service Templates

Provide documented templates in `docs/deploy/`:

* `switchyard-server.service` (Linux `systemd` user service).
* `com.switchyard.server.plist` (macOS `launchd`).

## Verification

1. **Detached Lifespan:** Run `switchyard-server --config routes.toml --detach --pidfile /tmp/sy.pid`, terminate the launching terminal, and confirm `GET /health` returns `200`.
2. **Graceful Stop:** Run `switchyard-server stop --pidfile /tmp/sy.pid` while an active request is in-flight; verify active requests drain within `--shutdown-timeout` before process exit.
3. **Cross-Platform:** Test detach and stop behavior on Linux, macOS, and Windows environments.
