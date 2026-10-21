# switchyard-translation

Pure Rust translation between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
request, response, and streaming formats.

The crate translates through provider-neutral LLM types from `switchyard-protocol` and does not
depend on provider SDKs, HTTP servers, Python, or FFI bindings.

## Translate a request

```rust
use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

let request = json!({
    "model": "switchyard",
    "input": "What's the weather in Paris?",
    "tools": [{
        "type": "function",
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            }
        }
    }]
});

let engine = TranslationEngine::default();
let policy = TranslationPolicy::default();

for destination in [WireFormat::OpenAiChat, WireFormat::AnthropicMessages] {
    let result = engine.translate_request(
        WireFormat::OpenAiResponses,
        destination,
        &request,
        &policy,
    )?;
    println!(
        "Responses -> {}\n{}\n",
        destination.as_str(),
        serde_json::to_string_pretty(&result.body)?
    );
}
```

## Translate a response

```rust
use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

let response = json!({
    "output": [{
        "type": "function_call",
        "call_id": "call_weather",
        "name": "get_weather",
        "arguments": "{\"city\":\"Paris\"}"
    }]
});

let engine = TranslationEngine::default();
let policy = TranslationPolicy::default();

for destination in [WireFormat::OpenAiChat, WireFormat::AnthropicMessages] {
    let result = engine.translate_response(
        WireFormat::OpenAiResponses,
        destination,
        &response,
        &policy,
    )?;
    println!(
        "Responses -> {}\n{}\n",
        destination.as_str(),
        serde_json::to_string_pretty(&result.body)?
    );
}
```

## Translation Policy

In the previous examples we used the default translation policy (`TranslationPolicy::default()`),
which determines whether unsupported details survive, warn, or fail. The `default` policy preserves
unknown fields, allows known information loss with warnings, generates stable IDs when missing,
keeps source payloads in memory, and leaves target capabilities unspecified.

The `lossy_conversion_policy` field controls whether known conversions that lose
information are allowed. `LossyConversionPolicy` has two variants:

- `AllowWithDiagnostics` (default): continue translation and record a
  `lossy_conversion` warning in the result's `diagnostics` field.
- `Reject`: return `TranslationError::LossyConversion` when a known lossy conversion
  is encountered.

Calling `translate_request`/`translate_response` annotates every diagnostic with
source and target formats.

To reject known lossy conversions:

```rust
use switchyard_translation::{LossyConversionPolicy, TranslationPolicy};

let policy = TranslationPolicy {
    lossy_conversion_policy: LossyConversionPolicy::Reject,
    ..TranslationPolicy::default()
};
```

## License

Licensed under the Apache License, Version 2.0. See the
[Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard) for details.
