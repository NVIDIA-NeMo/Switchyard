# libsy-server

A minimal [axum](https://github.com/tokio-rs/axum) server that exposes the OpenAI
Chat Completions, OpenAI Responses, and Anthropic Messages wire APIs and routes
every request through a libsy [`RandomAlgo`](../libsy), which picks a **weak-** or
**strong-tier** upstream and serves it via
[`switchyard-llm-client`](../libsy-llm-client). It is designed to sit in front of a
coding agent (e.g. Claude Code) for A/B-style routing between two models.

This crate is only the HTTP surface: routing, header normalization, SSE framing,
and error mapping. Each request is decoded to Switchyard's neutral IR, run through
the routing algorithm — which selects a tier and calls the upstream — and the
response is encoded back into the inbound wire format (buffered JSON or a stream of
wire events). It mirrors `switchyard-server` but swaps the profile-chain executor
for a libsy algorithm.

## Quickstart

```bash
# 1. Provide upstream credentials (whichever providers your tiers use).
export OPENAI_API_KEY="sk-..."       # for openai-chat / openai-responses tiers
export ANTHROPIC_API_KEY="sk-ant-..." # for anthropic-messages tiers

# 2. Run the proxy: a weak tier and a strong tier over one upstream gateway.
cargo run -p libsy-server -- \
    --base-url https://api.openai.com/v1 \
    --weak   gpt-4o-mini --weak-format   openai-chat \
    --strong gpt-4o      --strong-format openai-chat \
    --port 4000

# 3. Sanity-check it.
curl -s http://localhost:4000/health
curl -s http://localhost:4000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"anything","messages":[{"role":"user","content":"say hi"}]}'
```

The server advertises a single served model id, `switchyard`. The `model` field a
client sends is **ignored for routing** (the algorithm picks the tier) and the
response `model` is always restamped to the id the client asked for — the real
upstream model id never leaks.

## Point Claude Code at it

Claude Code speaks the Anthropic Messages API. Point it at the proxy with two
environment variables and run as normal:

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=sentinel   # any non-empty value; see below

claude                              # interactive
claude -p "refactor this function"  # one-shot / scriptable
```

- **The API key is a sentinel.** The proxy drops the caller's `Authorization` /
  `x-api-key` header and injects the *backend's* real credential (from
  `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`), so the key Claude Code sends never
  reaches an upstream and can be any non-empty placeholder.
- **The model name doesn't matter.** Claude Code sends e.g. `claude-3-5-sonnet`;
  the proxy routes by tier and restamps the response to that same name.
- **Streaming, tool use, and multi-turn all work** — inbound Anthropic requests are
  translated to each tier's upstream format and the response is translated back.

## Tiers, formats, and translation

Two tiers — `weak` and `strong` — are the random-routing targets. Each has its own
upstream wire format (`--weak-format` / `--strong-format`); every inbound request
(any of the three served endpoints) is decoded to the neutral IR, re-encoded to the
chosen tier's format for the upstream call, and translated back to the inbound
format for the client. So a tier can speak a *different* wire format than the client
— e.g. Claude Code (Anthropic) routed to an OpenAI Responses model.

Each tier authenticates with its format's provider key — OpenAI formats from
`OPENAI_API_KEY`, Anthropic Messages from `ANTHROPIC_API_KEY` — at the shared
`--base-url`; startup errors if a tier's key is unset.

Wire-format values: `openai-chat`, `openai-responses`, `anthropic-messages`.

## Affinity

By default every request is routed independently, so a single agent session mixes
tiers request-by-request. Pass `--affinity` to instead **pin related requests to one
tier**:

```bash
# Bare flag = session affinity: a whole session sticks to one tier.
cargo run -p libsy-server -- ... --affinity

# Or pin only sub-agents (root-agent turns keep routing randomly):
cargo run -p libsy-server -- ... --affinity subagent
```

Affinity keys on identifying request headers, normalized into request metadata:

- **`session`** keys on the session id. Claude Code's `X-Claude-Code-Session-Id` is
  recognized, so a whole Claude Code session (and concurrent sessions independently)
  pins to one tier. This makes A/B outcomes per-session and all-or-nothing rather
  than a per-request blend.
- **`subagent`** keys on session + agent id for requests identified as sub-agents
  (via Switchyard / Codex / Relay / Dynamo headers); root-agent turns stay random.

If a request carries no identifying header, affinity **falls open** to normal random
routing.

## Observability

Add `--log-routing` to print the chosen tier (and session id) for each request to
stderr:

```
[route][session=fb46caae-…] inbound=anthropic_messages -> gpt-4o
[route][session=fb46caae-…] inbound=anthropic_messages -> gpt-4o
```

With `--affinity`, this is how you confirm a session stayed pinned to one tier.

## Flags

| Flag | Required | Description |
|------|----------|-------------|
| `--base-url` | yes | Shared upstream base URL (e.g. `https://api.openai.com/v1`). |
| `--weak` / `--strong` | yes | Upstream model id for each tier. |
| `--weak-format` / `--strong-format` | yes | Wire format each tier's upstream speaks: `openai-chat`, `openai-responses`, `anthropic-messages`. |
| `--affinity [<kind>]` | no | Pin related requests to a tier. Bare = `session`; or `subagent`. Off when omitted. |
| `--log-routing` | no | Log each request's routing decision to stderr. |
| `--host` | no | Bind address (default `127.0.0.1`). |
| `--port` | no | Bind port (default `4000`). |

## Endpoints

| Method | Path                   | Purpose                          |
|--------|------------------------|----------------------------------|
| POST   | `/v1/chat/completions` | OpenAI Chat Completions inbound  |
| POST   | `/v1/messages`         | Anthropic Messages inbound       |
| POST   | `/v1/responses`        | OpenAI Responses inbound         |
| GET    | `/v1/models`           | Lists the single served model    |
| GET    | `/health`              | Liveness check                   |

## Testing

```bash
cargo test -p libsy-server
```

The integration tests drive `build_router` with `tower::oneshot` against a
`wiremock` upstream — no sockets, no credentials.
