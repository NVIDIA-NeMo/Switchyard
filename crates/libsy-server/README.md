# libsy-server

A minimal [axum](https://github.com/tokio-rs/axum) server that exposes the OpenAI
Chat Completions, OpenAI Responses, and Anthropic Messages wire APIs and routes
every request through a libsy [`RandomAlgo`](../libsy), which picks a weak- or
strong-tier upstream uniformly at random and serves it via
[`switchyard-llm-client`](../libsy-llm-client).

This crate is only the HTTP surface: routing, header normalization, SSE framing,
and error mapping. Each request is decoded to Switchyard's neutral IR, run
through the routing algorithm — which selects a target and calls the upstream —
and the response is encoded back into the same wire format (buffered JSON or a
stream of wire events). It mirrors `switchyard-server` but swaps the
profile-chain executor for a libsy algorithm.

## Endpoints

| Method | Path                   | Purpose                          |
|--------|------------------------|----------------------------------|
| POST   | `/v1/chat/completions` | OpenAI Chat Completions inbound  |
| POST   | `/v1/messages`         | Anthropic Messages inbound       |
| POST   | `/v1/responses`        | OpenAI Responses inbound         |
| GET    | `/v1/models`           | Lists the single served model    |
| GET    | `/health`              | Liveness check                   |

Clients address the server's served model id, `switchyard`; the response is
always restamped with that id rather than the real upstream model.

## Tiers, formats, and translation

Two tiers — `weak` and `strong` — are the random-routing targets. Each has its
own upstream wire format (`--weak-format` / `--strong-format`); every inbound
request (any of the three served endpoints) is decoded to the neutral IR,
re-encoded to the chosen tier's format for the upstream call, and translated back
to the inbound format for the client. Each tier authenticates with its format's
provider key — OpenAI formats from `OPENAI_API_KEY`, Anthropic Messages from
`ANTHROPIC_API_KEY` — at the shared `--base-url`; startup errors if a tier's key
is unset.

Wire-format values: `openai-chat`, `openai-responses`, `anthropic-messages`.

## Running

```bash
export OPENAI_API_KEY="sk-..."      # for openai-chat / openai-responses tiers
export ANTHROPIC_API_KEY="sk-..."   # for anthropic-messages tiers

cargo run -p libsy-server -- \
    --base-url https://api.openai.com/v1 \
    --weak gpt-4o-mini    --weak-format openai-chat \
    --strong gpt-4o       --strong-format openai-chat \
    --port 4000
```

Flags: `--base-url` (required), `--weak` / `--strong` (required tier model ids),
`--weak-format` / `--strong-format` (required tier wire formats), `--host`
(default `127.0.0.1`), `--port` (default `4000`).

## Testing

```bash
cargo test -p libsy-server
```

The integration tests drive `build_router` with `tower::oneshot` against a
`wiremock` upstream — no sockets, no credentials.
