# Agent Launchers

Switchyard launchers start a local proxy, configure the target coding agent to
use it, and shut the proxy down when the agent exits. Use them when you want
Claude Code, Codex, or OpenClaw to run through the same routing and translation
stack used by `switchyard serve`.

## Supported launchers

| Harness | Status | Entrypoint | Tested version | Inbound format |
|---|---|---|---|---|
| Claude Code | Supported | `switchyard launch claude` | 2.1.162 | Anthropic Messages |
| Codex CLI | Supported | `switchyard launch codex` | 0.137.0 | OpenAI Responses |
| OpenClaw | Experimental | `switchyard launch openclaw` | 2026.6.1 | OpenAI Chat Completions |

Other versions may work, but these are the versions covered by release docs.

## Start a launcher

Run `switchyard configure` once, then launch the agent:

```bash
switchyard configure
switchyard launch claude
switchyard launch codex
switchyard launch openclaw
```

Each launcher reads saved provider credentials from `~/.config/switchyard/`.
Interactive Claude Code and Codex sessions show live request and token totals
in the status footer. A saved route bundle is used when configured.

## Default routing behavior

The built-in LLM-as-classifier route is the default only when there is no
`--model`, no CLI `--routing-profiles`, no saved route bundle, and no
saved model default. Each turn is classified and dispatched to a weak or
strong tier using the validated coding-agent trio:

| Role | Model |
|---|---|
| Strong | Claude Opus 4.7 (`anthropic/claude-opus-4.7`) |
| Weak | Kimi K2.6 (`moonshotai/kimi-k2.6`) |
| Classifier | Gemini 3.5 Flash (`google/gemini-3.5-flash`) |

`--weak-model`, `--classifier-model`, `--profile`, and
`--classifier-min-confidence` tune only this built-in route. They are ignored
when route resolution selects an explicit or saved route bundle or single
model.

## Override the default route

Use `--model` for a single-model passthrough session:

```bash
switchyard launch claude --model openai/gpt-4o-mini
```

For launcher route-bundle routing, whether the bundle contains one route or
several, use:

```bash
switchyard --routing-profiles routes.yaml -- launch claude
```

`--model` and `--routing-profiles` are mutually exclusive. The first route in
the bundle becomes the initial model, and `/v1/models` plus the agent's model
picker expose the registered routes.

For route-bundle sessions, the footer reports active-model and aggregate
request/token counts, including errors. `/v1/routing/stats` aggregates usage
across the registered chains.

The same `--routing-profiles` bundle can be served directly with
`switchyard --routing-profiles routes.yaml -- serve`.

## Model requirements

Coding agents need models that support streaming and tool calling together. If
you pass `--model` explicitly, verify that model with a direct streaming +
tool-calling request before treating an empty response as a Switchyard issue.

### Agent stalls or produces no output

**Symptom:** the agent prints no output, and the upstream returns HTTP 200 with
`finish_reason=stop`, empty content, and no tool calls.

**Cause:** the selected model does not support streaming and tool calling
simultaneously.

**Diagnose:**

1. Probe the model directly, bypassing Switchyard, with a streaming +
   tool-calling request.
2. Switch back to the validated default route. If that works, the issue is
   model-specific.

In passthrough mode, Switchyard forwards the empty response as-is. Multi-target
routing profiles treat empty completions as errors and surface them explicitly.

### Claude Code with MCP tools

Bedrock-backed routes enforce a 64-character `toolSpec.name` limit. Claude
Code's MCP bridge can auto-inject longer tool names, such as
`mcp__plugin_microsoft_docs_microsoft_learn__microsoft_code_sample_search`,
which is 72 characters and produces a `BedrockException` 400 on tool-bearing
requests. If you hit this, point that tier at a non-Bedrock endpoint, such as
OpenRouter with `anthropic/claude-opus-4.7`, or choose another route target.

## Claude Code model picker

Claude Code's `/model` picker only shows routes whose id starts with `claude`
or `anthropic`, even though Switchyard's `/v1/models` returns every route. This
is a [Claude Code rule](https://code.claude.com/docs/en/llm-gateway#model-selection),
not a Switchyard restriction.

`switchyard launch claude` exposes route ids that do not already start with
`claude` or `anthropic` under a `claude-` alias alongside the original id. This
is an alias only; it does not create a second Switchyard chain.

| YAML key | Also reachable as |
|---|---|
| `opus-ds` | `claude-opus-ds` |
| `openai/gpt-4o` | `claude-openai/gpt-4o` |
| `claude-opus-direct` | `opus-direct` |

Both spellings work:

```bash
switchyard launch claude --model opus-ds
switchyard launch claude --model claude-opus-ds
```

The first registered model, which becomes `ANTHROPIC_CUSTOM_MODEL_OPTION`,
always has either an existing `claude`/`anthropic` prefix or the generated
`claude-` alias, so Claude Code's initial selection passes the picker filter.

This aliasing applies only to the Claude launcher. `switchyard launch codex`,
`switchyard launch openclaw`, and `switchyard serve` expose route ids verbatim.

You can also write the `claude-` prefix directly in route YAML if you
want the route ids to match exactly what appears in `/model`:

```yaml
routes:
  claude-opus-ds:
    type: stage_router
    ...
  claude-opus-direct:
    type: model
    ...
```

Other agent harnesses, such as Cursor or Aider, can point at the standalone
`switchyard serve` proxy. They do not have dedicated launchers today.
