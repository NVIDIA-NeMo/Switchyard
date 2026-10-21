# switchyard-translation

Pure Rust translation between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
request, response, and streaming formats.

The crate translates through provider-neutral LLM types from `switchyard-protocol` and does not
depend on provider SDKs, HTTP servers, Python, or FFI bindings.

## Translate a request

```rust
use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

fn translate_request() -> Result<(), Box<dyn std::error::Error>> {
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
            "Responses -> {}\n{}\nDiagnostics: {}\n",
            destination.as_str(),
            serde_json::to_string_pretty(&result.body)?,
            serde_json::to_string_pretty(&result.diagnostics)?
        );
    }

    Ok(())
}
```

## Translate a response

```rust
use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

fn translate_response() -> Result<(), Box<dyn std::error::Error>> {
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
            "Responses -> {}\n{}\nDiagnostics: {}\n",
            destination.as_str(),
            serde_json::to_string_pretty(&result.body)?,
            serde_json::to_string_pretty(&result.diagnostics)?
        );
    }

    Ok(())
}
```

## Translation policy

The examples above use `TranslationPolicy::default()`. A translation policy decides what happens
when an item cannot be translated exactly: keep it, warn about it, or return an error.

The default policy:

- Preserves unknown fields.
- Allows known information loss with warnings.
- Generates stable IDs when the source has none.
- Keeps the source payload in memory.
- Leaves target capabilities unspecified.

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
