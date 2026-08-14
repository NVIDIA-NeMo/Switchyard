<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# libsy-llm-client

An HTTP client that speaks Switchyard's neutral IR directly. You hand it a
[`switchyard_protocol::Request`] and a [`ModelId`]; it looks up the configured backend,
encodes the request to that backend's wire format, adds auth and forwards your
headers, makes the call with a shared `reqwest::Client`, and decodes the reply
back into a [`switchyard_protocol::Response`] — buffered or streamed.

It also pairs the client with a libsy algorithm: [`run`] drives
[`Algorithm::run_stream`] and serves every model call the algorithm offloads, so a
host that just wants the answer never has to drive the step stream itself.

It depends on `switchyard-libsy`, `switchyard-protocol`, and
`switchyard-translation`; no server, no provider SDK.

## Concepts

- **Model configs.** A client is built from [`ModelConfig`] values, each keyed by a
  [`ModelId`]. Each model has a default [`Backend`] and can have additional backends
  for other wire formats. [`TranslatingLlmClient::call_rewrite_model`] uses the
  request's metadata wire format when set, otherwise the model's default backend.
- **Model ids.** A [`ModelId`] is a newtype over `String` that behaves like one — it
  derefs to `str`, compares against string literals, and converts from `&str` or
  `String` with `.into()`. Anywhere below that takes `impl Into<ModelId>` accepts a
  bare literal; the borrowed positions need a `ModelId` value to reference.
- **Backends.** A [`Backend`] is one of `OpenAiChat`, `OpenAiResponses`, or
  `Anthropic`, each wrapping an [`HttpBackendConfig`] (`base_url`, `api_key`,
  static `extra_headers`, default `extra_body` fields, and `max_retries`). The
  variant fixes the URL path and auth scheme (Bearer vs `x-api-key` +
  `anthropic-version`).
- **Model rewrite.** The resolved [`ModelId`] is both the map key and the model id
  sent upstream — it overwrites whatever `model` the request arrived with.
- **Streaming is chosen by the request.** If the encoded body has `stream: true`
  (i.e. `request.llm_request.stream`), you get `LlmResponse::Stream`; otherwise
  `LlmResponse::Agg`. OpenAI Chat streaming requests default
  `stream_options.include_usage` to `true`; an explicit caller value is preserved.

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
        forward_auth: false,
        extra_headers: BTreeMap::new(),
        extra_body: BTreeMap::new(),
        max_retries: 2,
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
use switchyard_protocol::{completion_text, text_request, LlmResponse, ModelId, Request};

async fn ask(client: &TranslatingLlmClient) -> switchyard_llm_client::Result<String> {
    let request = Request {
        llm_request: text_request(None, "Say hello in five words."),
        raw_request: None,
        metadata: None,
    };

    // model_name wins over request.llm_request.model; it is also sent upstream.
    let model = ModelId::from("gpt-4o-mini");
    let response = client.call_rewrite_model(request, Some(&model)).await?;

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
use switchyard_protocol::{text_request, LlmResponse, LlmResponseChunk, ModelId, Request};

async fn stream(
    client: &TranslatingLlmClient,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut llm_request = text_request(None, "Count to five.");
    llm_request.stream = true;
    let request = Request { llm_request, raw_request: None, metadata: None };

    let model = ModelId::from("gpt-4o-mini");
    let response = client.call_rewrite_model(request, Some(&model)).await?;

    if let LlmResponse::Stream(mut events) = response.llm_response {
        while let Some(item) = events.next().await {
            // Each event carries zero or more provider-neutral chunks; the ones
            // ignored here are usage, tool-call deltas, and message start/stop.
            for chunk in item?.normalized() {
                if let LlmResponseChunk::TextDelta { text, .. } = chunk {
                    print!("{text}");
                }
            }
        }
    }
    Ok(())
}
```

### Routing an algorithm

[`run`] takes a libsy algorithm and a [`ClientRouter`], and returns the final response plus
the trace of decisions the algorithm published. The router resolves each offloaded call to
the client for the target the algorithm selected; `ClientRouter::single` is the
single-provider case:

```rust
use std::sync::Arc;
use switchyard_libsy::Algorithm;
use switchyard_llm_client::{ClientRouter, TranslatingLlmClient};
use switchyard_protocol::Request;

async fn route(
    algorithm: Arc<dyn Algorithm>,
    client: Arc<TranslatingLlmClient>,
    request: Request,
) -> switchyard_libsy::Result<String> {
    let clients = ClientRouter::single(client);
    let (trace, _response) =
        switchyard_llm_client::run(algorithm, clients, request, None).await?;
    Ok(trace
        .last()
        .map(|decision| decision.selected_model_id().to_string())
        .unwrap_or_default())
}
```

When targets are served by different clients — a judge on one provider and the serving
models on another, say — build the router from `ModelId -> client` with
[`ClientRouter::new`] instead. A model missing from that map fails with
`LlmClientError::Configuration` rather than silently going to another provider.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use switchyard_llm_client::ClientRouter;
use switchyard_protocol::{ModelId, RoutedLlmClient};

fn split_router(
    judge: Arc<dyn RoutedLlmClient>,
    serving: Arc<dyn RoutedLlmClient>,
) -> ClientRouter {
    ClientRouter::new(HashMap::from([
        (ModelId::from("gpt-4o-mini"), judge),
        (ModelId::from("claude-sonnet-4-5"), serving),
    ]))
}
```

A router is not itself a client: [`ClientRouter::route`] hands back a
[`switchyard_protocol::RoutedLlmClient`] and the caller makes the call.

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
  login, API-key, version, and content headers. So a caller's placeholder
  credential never overrides the backend's real key.
- `HttpBackendConfig::forward_auth` uses the caller's credential instead of the
  backend's configured key. OpenAI backends forward `authorization`,
  `chatgpt-account-id`, and `x-openai-fedramp`. Anthropic backends forward
  `authorization` or `x-api-key`; they also keep `oauth-*` values from
  `anthropic-beta` and remove other caller-supplied beta values.
- Per-backend custom headers go in `HttpBackendConfig::extra_headers`. Set credentials with
  `api_key`. OpenAI backends reject `Authorization`; Anthropic backends reject `x-api-key`
  and `anthropic-version`. Header names are case-insensitive.
- Per-target top-level request defaults go in `HttpBackendConfig::extra_body`.
  The merge is shallow and fields already present in the request take precedence.
- `HttpBackendConfig::max_retries` controls additional attempts after retryable
  transport failures, timeouts, HTTP 408/429, and 5xx responses. Buffered body
  transport failures are retried; streaming body failures are not replayed after
  the response has been returned.

Retries replay the same upstream request. A transport failure can therefore
duplicate a request that the provider processed but did not finish returning,
and the retry budget plus capped `Retry-After` delays determines total latency.

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
[`ModelId`]: ../protocol/src/model_id.rs
[`TranslatingLlmClient::call_rewrite_model`]: src/client.rs
[`LlmClientError`]: ../protocol/src/client.rs
