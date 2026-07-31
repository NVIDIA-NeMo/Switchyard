# Getting Started

## Install

```bash
uv tool install "nemo-switchyard[cli,server]"
```

## Launch a coding agent

The packaged deployment uses OpenRouter:

```bash
export OPENROUTER_API_KEY="sk-or-..."
switchyard launch claude --model switchyard
```

Codex and OpenClaw use the same shape:

```bash
switchyard launch codex --model switchyard
switchyard launch openclaw --model switchyard
```

To use another deployment:

```bash
switchyard launch claude --model my-route --config routes.toml
```

See `crates/switchyard-server/README.md` for the TOML schema.
