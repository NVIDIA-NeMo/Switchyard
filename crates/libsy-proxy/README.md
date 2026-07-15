<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# libsy-proxy

A minimal HTTP proxy that puts the OpenAI and Anthropic wire APIs in front of a
single upstream model. It decodes each inbound request to Switchyard's neutral
IR, forwards it to your upstream via [`libsy-llm-client`], and encodes the reply
back into the same wire format the caller used — buffered JSON or SSE. That means
you can point a coding agent (Claude Code, Codex) at one endpoint and have it talk
to any OpenAI- or Anthropic-compatible backend unchanged.

The proxy serves a single model named **`switchyard`** and rewrites every call to
your `--model-name` upstream.

## Quickstart

### 1. Build

```bash
cargo build -p libsy-proxy --release
# binary at target/release/libsy-proxy
```

### 2. Run

Point it at an upstream and give it a credential. At least one of
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` must be set.

```bash
# OpenAI-compatible upstream (OpenAI, OpenRouter, vLLM, NIM, …)
export OPENAI_API_KEY="sk-..."
libsy-proxy \
  --base-url https://openrouter.ai/api/v1 \
  --model-name openai/gpt-4o-mini \
  --port 4000
```

On start it prints the bound address, the upstream, the formats it can serve, and
the endpoints:

```
libsy-proxy
  listening:     http://127.0.0.1:4000
  upstream:      https://openrouter.ai/api/v1 (model openai/gpt-4o-mini)
  upstream fmts: openai_chat, openai_responses
  serving model: switchyard
  endpoints:     GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/messages, POST /v1/responses
  stop:          Ctrl-C
```

### 3. Call it

```bash
# discovery
curl -s localhost:4000/v1/models
curl -s localhost:4000/health

# buffered chat completion
curl -s localhost:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"switchyard","messages":[{"role":"user","content":"say ok"}]}'

# streaming Anthropic Messages (works even against an OpenAI upstream)
curl -sN localhost:4000/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"switchyard","max_tokens":64,"stream":true,
       "messages":[{"role":"user","content":"say ok"}]}'
```

The inbound `model` field is ignored for routing — send `switchyard` (or anything);
the request is always forwarded to `--model-name`.

## Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--base-url` | yes | — | Upstream base URL (e.g. `https://api.openai.com/v1`). |
| `--model-name` | yes | — | Model id sent upstream on every call. |
| `--upstream-format` | no | — | Fallback upstream format: `openai-chat`, `openai-responses`, or `anthropic`. |
| `--host` | no | `127.0.0.1` | Bind address. |
| `--port` | no | `4000` | Bind port. |

### Credentials (env)

| Variable | Enables upstream format(s) | Auth sent upstream |
|----------|----------------------------|--------------------|
| `OPENAI_API_KEY` | `openai-chat`, `openai-responses` | `Authorization: Bearer …` |
| `ANTHROPIC_API_KEY` | `anthropic` | `x-api-key` + `anthropic-version` |

Set at least one. A backend is configured for every format whose key is present.
The proxy injects the real key itself, so the placeholder credential a coding agent
sends (`switchyard`/empty) is dropped automatically.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions (inbound) |
| `POST` | `/v1/messages` | Anthropic Messages (inbound) |
| `POST` | `/v1/responses` | OpenAI Responses (inbound) |
| `GET`  | `/v1/models` | Lists the served `switchyard` model |
| `GET`  | `/health` | Liveness probe |

## Upstream format selection

Every inbound request arrives in one of the three wire formats. For each request
the proxy picks the upstream format like this:

1. **Match** — if a backend exists for the inbound format's provider (i.e. you have
   its key), forward in that same format (no cross-translation).
2. **Fallback** — otherwise use `--upstream-format`.
3. Otherwise return `400` asking you to set the key or `--upstream-format`.

Example: your upstream is OpenAI-compatible (`OPENAI_API_KEY` only), but Claude Code
speaks Anthropic. Set `--upstream-format openai-chat`; inbound `/v1/messages` is then
translated to OpenAI Chat upstream and the reply is translated back to Anthropic.

```bash
export OPENAI_API_KEY="sk-..."
libsy-proxy \
  --base-url https://openrouter.ai/api/v1 \
  --model-name openai/gpt-4o-mini \
  --upstream-format openai-chat
```

## Use with a coding agent

### Claude Code

Claude Code speaks the Anthropic Messages API. Point it at the proxy:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:4000 \
ANTHROPIC_AUTH_TOKEN=switchyard \
ANTHROPIC_API_KEY= \
ANTHROPIC_MODEL=switchyard \
claude
```

If your upstream is OpenAI-compatible, add `--upstream-format openai-chat` when
starting the proxy (see above).

### Codex CLI

Codex speaks the OpenAI Responses API; point its provider at `.../v1`:

```bash
export OPENAI_API_KEY="sk-..."
libsy-proxy --base-url https://api.openai.com/v1 --model-name gpt-4o-mini
codex \
  -c model_provider=switchyard \
  -c model_providers.switchyard.base_url=http://127.0.0.1:4000/v1 \
  -c model_providers.switchyard.wire_api=responses \
  -m switchyard
```

## Errors

Errors are returned as OpenAI-style envelopes:

```json
{ "error": { "message": "...", "type": "...", "code": "..." } }
```

Upstream HTTP errors pass their status through; a context-window overflow becomes a
`400`; a mid-stream failure is delivered as a final SSE error frame (the `200` is
already committed).

## Scope

Deliberately minimal: single upstream, plaintext HTTP, no TLS/stats/metrics and no
`/v1/messages/count_tokens`. For multi-endpoint routing, health-aware serving, and
profile configs, use the full `switchyard` server.

[`libsy-llm-client`]: ../libsy-llm-client
