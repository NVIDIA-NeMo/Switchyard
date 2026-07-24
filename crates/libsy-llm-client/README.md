<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# libsy-llm-client

An HTTP client that speaks Switchyard's neutral IR directly. You hand it a
[`switchyard_protocol::Request`] and a model name; it looks up the configured backend,
encodes the request to that backend's wire format, adds auth and forwards your
headers, makes the call with a shared `reqwest::Client`, and decodes the reply
back into a [`switchyard_protocol::Response`] — buffered or streamed.

It depends only on `switchyard-protocol` and `switchyard-translation`; no server, no
provider SDK.

## Concepts

- **Model configs.** A client is built from [`ModelConfig`] values. Each model has a
  default [`Backend`] and can have additional backends for other wire formats.
  [`TranslatingLlmClient::call_rewrite_model`] uses the request's metadata wire format
  when set, otherwise the model's default backend.
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
use std::collections::BTreeMap;
use switchyard_llm_client::{
    Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient,
};

fn build_client() -> switchyard_llm_client::Result<TranslatingLlmClient> {
    let openai = HttpBackendConfig {
        base_url: "https://api.openai.com/v1".to_string(),
        api_key: std::env::var("OPENAI_API_KEY").ok(),
        extra_headers: BTreeMap::new(),
    };

    let models = [ModelConfig::new(
        "gpt-4o-mini",
        Backend::OpenAiChat(openai),
        None,
    )];

    TranslatingLlmClient::new(&models)
}
```

### Buffered call

```rust
use switchyard_llm_client::{LlmClientError, TranslatingLlmClient};
use switchyard_protocol::{completion_text, text_request, Context, LlmResponse, Request};

async fn ask(client: &TranslatingLlmClient) -> switchyard_llm_client::Result<String> {
    let request = Request {
        llm_request: text_request(None, "Say hello in five words."),
        raw_request: None,
        metadata: None,
    };

    // model_name wins over request.llm_request.model; it is also sent upstream.
    let response = client
        .call_rewrite_model(Context::default(), request, Some("gpt-4o-mini"))
        .await?;

    match response.llm_response {
        LlmResponse::Agg(agg) => Ok(completion_text(&agg)),
        LlmResponse::Stream(_) => Err(LlmClientError::InvalidResponse {
            source: "expected a buffered response".into(),
        }),
    }
}
```

### Streaming call

Set `stream` on the IR request and drive the returned chunk stream:

```rust
use futures_util::StreamExt;
use switchyard_llm_client::TranslatingLlmClient;
use switchyard_protocol::{text_request, Context, LlmResponse, LlmResponseChunk, Request};

async fn stream(
    client: &TranslatingLlmClient,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut llm_request = text_request(None, "Count to five.");
    llm_request.stream = true;
    let request = Request { llm_request, raw_request: None, metadata: None };

    let response = client
        .call_rewrite_model(Context::default(), request, Some("gpt-4o-mini"))
        .await?;

    if let LlmResponse::Stream(mut chunks) = response.llm_response {
        while let Some(item) = chunks.next().await {
            match item {
                Ok(LlmResponseChunk::TextDelta { text, .. }) => print!("{text}"),
                Ok(_) => {}                       // usage, tool-call deltas, message start/stop
                Err(error) => return Err(error),
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
requests with the same helpers works the same way. Set `request.metadata.wire_format`
to select a non-default backend before calling `call_rewrite_model`. Register several
formats under one model to serve it over more than one upstream API:

```rust
use switchyard_llm_client::{
    Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient,
};

fn build_multi_format_client(
    openai_chat: HttpBackendConfig,
    openai_responses: HttpBackendConfig,
    anthropic: HttpBackendConfig,
) -> switchyard_llm_client::Result<TranslatingLlmClient> {
    let models = [ModelConfig::new(
        "my-model",
        Backend::OpenAiChat(openai_chat),
        Some(vec![
            Backend::OpenAiResponses(openai_responses),
            Backend::Anthropic(anthropic),
        ]),
    )];

    TranslatingLlmClient::new(&models)
}
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

`call_rewrite_model` returns [`LlmClientError`]:

| Variant | When |
|---------|------|
| `InvalidRequest { message }` | the request does not identify a model |
| `Configuration { message }` | the model or requested wire format has no configured backend |
| `RequestTranslation(msg)` | decoding the inbound request failed in the translation engine |
| `RequestEncoding(msg)` | re-encoding an already-decoded request to the wire format failed (internal fault) |
| `ResponseTranslation(msg)` | response decoding or encoding failed in the translation engine |
| `Timeout { source }` | request or response body read exceeded its timeout |
| `Transport { source }` | non-timeout connection or transport failure |
| `ContextWindowExceeded { model, message }` | upstream 400 detected as a context overflow (checked before `UpstreamHttp`, so callers can evict-and-retry) |
| `UpstreamHttp { status, body }` | any other non-2xx upstream response |
| `InvalidResponse { source }` | the upstream response could not be decoded |
| `Other(source)` | a client-specific failure outside the shared categories |

[`switchyard_protocol::Request`]: ../libsy-protocol
[`switchyard_protocol::Response`]: ../libsy-protocol
[`libsy-proxy`]: ../libsy-proxy
[`Backend`]: src/backend.rs
[`HttpBackendConfig`]: src/backend.rs
[`ModelConfig`]: src/client.rs
[`TranslatingLlmClient::call_rewrite_model`]: src/client.rs
[`LlmClientError`]: ../protocol/src/client.rs
