<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# switchyard-translation

Pure-Rust translation between LLM provider wire formats. Requests, responses, and
streaming events are converted **through a neutral conversation IR** rather than
point-to-point, so adding a provider is one codec, not one codec per pair.

The crate has no dependency on provider SDKs, HTTP servers, Python, or FFI — it takes
`serde_json::Value` in and gives `serde_json::Value` out.

Supported built-in formats ([`WireFormat`]):

| Variant | Identifier | API |
|---|---|---|
| `OpenAiChat` | `openai_chat` | OpenAI Chat Completions |
| `AnthropicMessages` | `anthropic_messages` | Anthropic Messages |
| `OpenAiResponses` | `openai_responses` | OpenAI Responses |

## Add the dependency

```toml
[dependencies]
switchyard-translation = { path = "../switchyard-translation" }
serde_json = "1"
```

## The model

Every operation routes through the IR:

```text
source body ──decode──▶ ConversationRequest / ConversationResponse ──encode──▶ target body
   (Value)                        (neutral IR)                                    (Value)
```

- **`TranslationEngine`** is the stateless entry point. `TranslationEngine::default()`
  is pre-loaded with all built-in codecs.
- **`TranslationPolicy`** controls loss handling, unknown-field preservation, and
  deterministic id generation. `TranslationPolicy::default()` preserves unknown fields
  and reports lossy conversions as diagnostics rather than failing.
- Every call returns diagnostics alongside its result — translation is *loss-aware*, not
  lossy-silent.

The IR types live in the [`libsy-protocol`] crate and are re-exported here; the top-level
request/response types are named `LlmRequest`/`LlmResponse` there and re-exported as
`ConversationRequest`/`ConversationResponse` for translation's callers.

## Translate a request

```rust
use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

let engine = TranslationEngine::default();

let anthropic_body = json!({
    "model": "claude-sonnet-4-20250514",
    "system": [{"type": "text", "text": "Be helpful."}],
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
});

let output = engine.translate_request(
    WireFormat::AnthropicMessages, // source
    WireFormat::OpenAiChat,        // target
    &anthropic_body,
    &TranslationPolicy::default(),
)?;

// `output.body` is an OpenAI Chat request; `output.diagnostics` notes anything
// dropped or remapped on the way (e.g. Anthropic-only fields).
println!("{}", serde_json::to_string_pretty(&output.body)?);
for diag in &output.diagnostics {
    eprintln!("[{:?}] {}: {}", diag.severity, diag.code, diag.message);
}
```

## Translate a response

Responses flow the opposite direction (provider → client), but the call is symmetric:

```rust
use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

let engine = TranslationEngine::default();
let openai_response = json!({
    "id": "chatcmpl-1",
    "model": "gpt-4o",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hi there."},
        "finish_reason": "stop"
    }]
});

let output = engine.translate_response(
    WireFormat::OpenAiChat,
    WireFormat::AnthropicMessages,
    &openai_response,
    &TranslationPolicy::default(),
)?;
```

## Work with the IR directly

To route, inspect, or rewrite requests, decode to the IR and encode back yourself instead
of using the one-shot `translate_*` helpers:

```rust
use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

let engine = TranslationEngine::default();
let policy = TranslationPolicy::default();

let mut decoded = engine
    .decode_request(WireFormat::OpenAiChat, &json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}]
    }), &policy)?;

// `decoded.request` is a `ConversationRequest` — a normal Rust struct.
decoded.request.model = Some("gpt-4o-mini".to_string());

let reencoded = engine.encode_request(WireFormat::OpenAiChat, &decoded.request, &policy)?;
```

## Translate a stream

Streaming is stateful: create one `StreamTranslationState` per stream and feed events
through as they arrive. One source event may expand to zero or more target events.

```rust
use serde_json::json;
use switchyard_translation::{StreamTranslationState, TranslationEngine, WireFormat};

let engine = TranslationEngine::default();
let mut state = StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::AnthropicMessages);

for chunk in incoming_openai_chunks {
    let out_events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiChat,
        WireFormat::AnthropicMessages,
        &chunk,
    )?;
    for event in out_events {
        send_to_client(event);
    }
}

// After the source stream closes, flush any trailing target events (stop events, usage).
for event in engine.finish_stream(&mut state, WireFormat::AnthropicMessages)? {
    send_to_client(event);
}
```

## Custom formats

Register your own codec on a `FormatRegistry` to add a format or override a built-in, then
build an engine from it:

```rust,ignore
use switchyard_translation::{FormatRegistry, TranslationEngine};

let mut registry = FormatRegistry::with_builtins();
registry.register(MyProviderCodec); // impl FormatCodec
let engine = TranslationEngine::new(registry);
```

## Where to look next

- `src/ir.rs` (in `libsy-protocol`) — the neutral IR type definitions.
- `src/policy.rs` — knobs for loss, preservation, and id generation.
- `src/codecs/` — per-provider buffered and streaming codecs; the reference for writing
  your own.
- `tests/` — `request_translation.rs`, `response_translation.rs`, `stream_translation.rs`,
  and `lossless_roundtrip.rs` are runnable, worked examples of every path above.

[`WireFormat`]: src/format.rs
[`libsy-protocol`]: ../libsy-protocol
