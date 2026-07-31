# Switchyard

Switchyard routes and translates LLM traffic for coding agents and API clients.
It supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages.

## Start

```bash
uv tool install "nemo-switchyard[cli,server]"
export OPENROUTER_API_KEY="sk-or-..."
switchyard launch claude --model switchyard
```

Use a custom native deployment when needed:

```bash
switchyard launch codex --model my-route --config routes.toml
```

The TOML schema is maintained in `crates/switchyard-server/README.md`.

## Read More

- [Getting Started](getting_started.md)
- [CLI Reference](cli_reference.md)
- [Routing Overview](routing_algorithms/overview.md)
- [Architecture](architecture.md)
- [Context-Window Handling](operations/context_window.md)
