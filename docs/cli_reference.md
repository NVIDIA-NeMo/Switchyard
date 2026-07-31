# CLI reference

Switchyard exposes two commands:

| Command | Purpose |
|---|---|
| `switchyard serve` | Run the Python server from a YAML routing bundle. |
| `switchyard launch` | Launch a coding agent through the native Rust server. |

## `switchyard serve`

```bash
switchyard serve --routing-profiles routes.yaml --port 4000
```

| Option | Purpose |
|---|---|
| `--routing-profiles PATH`, `-c PATH` | Required YAML routing bundle. |
| `--host HOST` | Bind host. |
| `--port PORT`, `-p PORT` | Bind port. |
| `--inbound {openai,anthropic,both}` | Accepted inbound format. |
| `--reload` | Enable Uvicorn reload. |
| `--workers N`, `-w N` | Uvicorn worker count. |
| `--enable-rl-logging` | Write RL traces. |
| `--rl-log-dir DIR` | RL trace directory. |
| `--routing-log-file PATH` | Append routing records as JSONL. |
| `--intake-enabled` | Enable Intake for opted-in requests. |
| `--enable-intake` | Deprecated alias for `--intake-enabled`. |
| `--intake-base-url URL` | Intake base URL. |
| `--intake-workspace NAME` | Intake workspace. |
| `--intake-api-key VALUE` | Intake API key. |
| `--intake-target-url URL` | Intake posting URL. |

## `switchyard launch`

```bash
export OPENROUTER_API_KEY="sk-or-..."
switchyard launch claude --model switchyard
```

Supported agents are `claude`, `codex`, and `openclaw`.

```text
switchyard launch <agent> --model <route-id> [--config <deployment.toml>] [-- <agent args>]
```

| Option | Purpose |
|---|---|
| `--model ID` | Required route ID from the deployment. |
| `--config PATH` | TOML deployment. Defaults to the packaged OpenRouter deployment. |
| `-- ...` | Arguments forwarded to the coding agent. |

Custom deployment syntax is documented in `crates/switchyard-server/README.md`.
