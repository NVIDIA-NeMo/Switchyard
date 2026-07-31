# CLI reference

The documented CLI workflow launches a coding agent through the native Rust
server.

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
