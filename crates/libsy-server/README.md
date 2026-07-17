# libsy-server

A minimal [axum](https://github.com/tokio-rs/axum) server that exposes the OpenAI
Chat Completions, OpenAI Responses, and Anthropic Messages wire APIs and forwards
every request to a single upstream via
[`switchyard-llm-client`](../libsy-llm-client).

This crate is only the HTTP surface: routing, header normalization, SSE framing,
and error mapping. The actual work — decoding an inbound request to Switchyard's
neutral IR, calling the upstream, and encoding the response back into the same
wire format (buffered JSON or a stream of wire events) — lives in
`switchyard-llm-client`'s `TranslatingLlmClient::call_rewrite_model_raw`. It
mirrors `switchyard-server` but swaps the profile-chain executor for one client.

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

## Upstream and served formats

Serving is same-format: each inbound endpoint is forwarded to an upstream of the
same wire format. A backend is configured for every format whose provider key is
set — OpenAI Chat + Responses from `OPENAI_API_KEY`, Anthropic Messages from
`ANTHROPIC_API_KEY` — and the server serves exactly those endpoints. An inbound
request for a format with no configured backend gets a `400` with an
`unsupported_format` error.

## Running

```bash
export OPENAI_API_KEY="sk-..."      # and/or ANTHROPIC_API_KEY

cargo run -p libsy-server -- \
    --base-url https://api.openai.com/v1 \
    --model-name gpt-4o-mini \
    --port 4000
```

Flags: `--base-url` (required), `--model-name` (required upstream id),
`--host` (default `127.0.0.1`), `--port` (default `4000`).

## Testing

```bash
cargo test -p libsy-server
```

The integration tests drive `build_router` with `tower::oneshot` against a
`wiremock` upstream — no sockets, no credentials.
