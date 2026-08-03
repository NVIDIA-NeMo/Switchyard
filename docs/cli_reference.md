# CLI Reference

Switchyard has two command-line paths:

- **Launcher Path:** `switchyard launch` starts a supported coding agent and
  hosts the native Rust server through the packaged PyO3 binding.
- **Server Path:** `switchyard-server` runs the standalone Rust proxy for API
  clients and custom deployments.

## Launcher Path: `switchyard launch`

Install the launcher with `uv tool install --python 3.12 "nemo-switchyard[cli,server]"`. The
selected coding agent must also be installed and available on `PATH`.

### Usage

```bash
switchyard launch <agent> --model <route-id> [--config <deployment.toml>] [-- <agent args>]
```

Supported agents are `claude`, `codex`, and `openclaw`.

| Option | Required | Purpose |
|---|:---:|---|
| `--model ID` | Yes | Route ID from the selected deployment. |
| `--config PATH` | No | Native TOML deployment. Defaults to the packaged OpenRouter deployment. |
| `-- ...` | No | Arguments forwarded unchanged to the coding agent. |

### Packaged Deployment

The packaged deployment exposes the route ID `switchyard` and uses OpenRouter:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
switchyard launch claude --model switchyard
switchyard launch codex --model switchyard
switchyard launch openclaw --model switchyard
```

### Custom Deployment

Select a route ID from an explicit native TOML deployment:

```bash
switchyard launch claude --model my-route --config routes.toml
```

The launcher owns the server lifecycle and stops it when the coding agent exits.
This command does not install or run the standalone `switchyard-server` binary.

## Server Path: `switchyard-server`

Build the standalone binary with
`cargo build --locked --release -p switchyard-server`. It reads the same native
TOML deployment schema accepted by the launcher.

### Usage

```bash
switchyard-server --config <deployment.toml> [options]
```

When built from the repository, invoke it as
`./target/release/switchyard-server`.

| Option | Default | Purpose |
|---|---|---|
| `--config PATH` | Required | TOML file defining LLM clients, targets, and algorithm routes. |
| `--host HOST` | `0.0.0.0` | Address on which the server listens. |
| `-p, --port PORT` | `4000` | Port on which the server listens. |
| `--backlog BACKLOG` | `65535` | TCP listen backlog configured before accepting traffic. |
| `--dry-run` | Off | Validate the deployment without binding a socket. |
| `--tls-cert PATH` | None | PEM certificate path; requires `--tls-key`. |
| `--tls-key PATH` | None | PEM private-key path; requires `--tls-cert`. |
| `-h, --help` | — | Print command help. |
| `-V, --version` | — | Print the server version. |

Validate a deployment, then start the proxy:

```bash
./target/release/switchyard-server --config routes.toml --dry-run
./target/release/switchyard-server --config routes.toml \
  --host 127.0.0.1 --port 4000
```

## Related Documentation

- [Getting Started](getting_started.md): installation, configuration, and verification for both paths
- [`switchyard-server`](../crates/switchyard-server/README.md): complete TOML schema, TLS, and metrics
