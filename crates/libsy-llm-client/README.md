<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# libsy-llm-client

An HTTP client that speaks Switchyard's neutral IR directly. You hand it a
[`libsy_protocol::Request`] and a model name; it looks up the configured backend,
encodes the request to that backend's wire format, adds auth and forwards your
headers, makes the call with a shared `reqwest::Client`, and decodes the reply
back into a [`libsy_protocol::Response`] — buffered or streamed.

It depends only on `libsy-protocol` and `switchyard-translation`; no server, no
provider SDK. For an HTTP front door built on top of it, see [`libsy-proxy`].

## Concepts

- **Two-layer map.** A client is built from `model → wire format → Backend`, so
  one model can be served over several upstream formats. `call(request, model, format)`
  picks the backend by `(model, format)`.
- **Backends.** A [`Backend`] is one of `OpenAiChat`, `OpenAiResponses`, or
  `Anthropic`, each wrapping an [`HttpBackendConfig`] (`base_url`, `api_key`,
  static `extra_headers`). The variant fixes the URL path and auth scheme
  (Bearer vs `x-api-key` + `anthropic-version`).
- **Model rewrite.** The resolved model name is both the map key and the model id
  sent upstream — it overwrites whatever `model` the request arrived with.
- **Streaming is chosen by the request.** If the encoded body has `stream: true`
  (i.e. `request.llm_request.stream`), you get `LlmResponse::Stream`; otherwise
  `LlmResponse::Agg`.

## Add the dependency

Within this workspace:

```toml
[dependencies]
switchyard-llm-client = { path = "../libsy-llm-client" }
switchyard-protocol = { path = "../libsy-protocol" }
switchyard-translation = { path = "../switchyard-translation" }   # for WireFormat
```

## Quickstart

### Build a client

```rust
use std::collections::{BTreeMap, HashMap};
use switchyard_llm_client::{Backend, HttpBackendConfig, LlmModelClient};
use switchyard_translation::WireFormat;

fn build_client() -> libsy_llm_client::Result<LlmModelClient> {
    let openai = HttpBackendConfig {
        base_url: "https://api.openai.com/v1".to_string(),
        api_key: std::env::var("OPENAI_API_KEY").ok(),
        extra_headers: BTreeMap::new(),
    };

    // model → format → backend
    let map = HashMap::from([(
        "gpt-4o-mini".to_string(),
        HashMap::from([(WireFormat::OpenAiChat, Backend::OpenAiChat(openai))]),
    )]);

    LlmModelClient::new(map)
}
```

### Buffered call

```rust
use libsy_protocol::{completion_text, text_request, Request};
use switchyard_translation::WireFormat;

async fn ask(client: &LlmModelClient) -> libsy_llm_client::Result<String> {
    let request = Request {
        llm_request: text_request(None, "Say hello in five words."),
        raw_request: None,
        metadata: None,
    };

    // model_name wins over request.llm_request.model; it is also sent upstream.
    let response = client
        .call(request, Some("gpt-4o-mini"), WireFormat::OpenAiChat)
        .await?;

    // Buffered backends return `Agg`; a stream can be folded with `.aggregate()`.
    let agg = response.llm_response.aggregate().await.unwrap();
    Ok(completion_text(&agg))
}
```

### Streaming call

Set `stream` on the IR request and drive the returned chunk stream:

```rust
use futures_util::StreamExt;
use libsy_protocol::{text_request, LlmResponse, LlmResponseChunk, Request};
use switchyard_translation::WireFormat;

async fn stream(client: &LlmModelClient) -> libsy_llm_client::Result<()> {
    let mut llm_request = text_request(None, "Count to five.");
    llm_request.stream = true;
    let request = Request { llm_request, raw_request: None, metadata: None };

    let response = client
        .call(request, Some("gpt-4o-mini"), WireFormat::OpenAiChat)
        .await?;

    if let LlmResponse::Stream(mut chunks) = response.llm_response {
        while let Some(item) = chunks.next().await {
            match item {
                Ok(LlmResponseChunk::TextDelta { text, .. }) => print!("{text}"),
                Ok(_) => {}                       // usage, tool-call deltas, message start/stop
                Err(error) => return Err(libsy_llm_client::LlmClientError::Stream(error.to_string())),
            }
        }
    }
    Ok(())
}
```

## Cross-format translation

The request/response are translated through the neutral IR, so the inbound shape
you build and the backend's wire format are independent. Pointing a
`WireFormat::AnthropicMessages` backend at an Anthropic endpoint while building
requests with the same helpers works the same way — pass the matching format to
`call`. Register several formats under one model to serve it over more than one
upstream API:

```rust
let backends = HashMap::from([
    (WireFormat::OpenAiChat,      Backend::OpenAiChat(openai_chat_cfg)),
    (WireFormat::OpenAiResponses, Backend::OpenAiResponses(openai_resp_cfg)),
    (WireFormat::AnthropicMessages, Backend::Anthropic(anthropic_cfg)),
]);
let map = HashMap::from([("my-model".to_string(), backends)]);
// client.formats_for("my-model") -> the configured formats
```

## Headers & auth

- The `Backend` variant sets auth: OpenAI formats send `Authorization: Bearer <key>`;
  Anthropic sends `x-api-key: <key>` plus `anthropic-version`.
- `request.metadata.http_headers` are forwarded upstream, **except** reserved ones:
  `host`, `content-length`, `connection`, and the backend-owned
  `authorization` / `x-api-key` / `anthropic-version` / `content-type`. So a
  caller's placeholder credential never overrides the backend's real key.
- Per-backend static headers go in `HttpBackendConfig::extra_headers`.

## Errors

`call` returns [`LlmClientError`]:

| Variant | When |
|---------|------|
| `MissingModel` | no `model_name` arg and no `request.llm_request.model` |
| `UnknownModel(model)` | model has no backends configured |
| `UnknownModelFormat { model, format }` | model has no backend for that format |
| `Translation(msg)` | encode/decode failed in the translation engine |
| `Transport(msg)` | connect/timeout/transport failure |
| `ContextWindowExceeded { model, message }` | upstream 400 detected as a context overflow (checked before `UpstreamHttp`, so callers can evict-and-retry) |
| `UpstreamHttp { status, body }` | any other non-2xx upstream response |
| `Stream(msg)` | mid-stream read / malformed frame |

[`libsy_protocol::Request`]: ../libsy-protocol
[`libsy_protocol::Response`]: ../libsy-protocol
[`libsy-proxy`]: ../libsy-proxy
[`Backend`]: src/backend.rs
[`HttpBackendConfig`]: src/backend.rs
[`LlmModelClient`]: src/client.rs
[`LlmClientError`]: src/error.rs
