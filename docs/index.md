---
title: "Home"
description: "Route and translate LLM traffic for coding agents and API clients with NVIDIA NeMo Switchyard."
position: 1
---
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

The TOML schema is maintained in
[`crates/switchyard-server/README.md`](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/switchyard-server/README.md).

## Read More

- [Getting Started](/get-started/getting-started)
- [CLI Reference](/reference/cli-reference)
- [Routing Overview](/routing/overview)
- [Architecture](/concepts/architecture)
- [Context-Window Handling](/operations/context-window-handling)
