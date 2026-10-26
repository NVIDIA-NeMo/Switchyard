# LLM Translation

For a small integration example, read **[Use the translation crate](../../protocols/llm.md#use-the-translation-crate)**.
For field mappings, use **[LLM Requests](llm_requests.md)** and
**[Responses and Streams](llm_responses.md)**.

<a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html"><code>TranslationEngine</code></a> converts between provider JSON and Switchyard protocol
types. It makes no network calls. <a href="../../../rust/switchyard_protocol/format/enum.WireFormat.html"><code>WireFormat</code></a> identifies the built-in Chat
Completions, Responses, and Anthropic Messages formats. <a href="../../../rust/switchyard_protocol/format/struct.FormatId.html"><code>FormatId</code></a> can also
identify custom formats registered through the codec interfaces.

## Engine operations

All operations below are methods of <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html"><code>TranslationEngine</code></a>. The request and
buffered-response methods take a <a href="../../../rust/switchyard_translation/policy/struct.TranslationPolicy.html"><code>TranslationPolicy</code></a>.

| Method | Input and result | Use it for |
|---|---|---|
| <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.decode_request"><code>decode_request</code></a> | Provider JSON to <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html"><code>LlmRequest</code></a>, plus diagnostics. | Accepting a request from a provider-facing gateway or harness. |
| <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.encode_request"><code>encode_request</code></a> | <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html"><code>LlmRequest</code></a> to provider JSON, plus diagnostics. | Sending a call through an existing client. |
| <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.translate_request"><code>translate_request</code></a> | Source provider JSON to target provider JSON. | Direct request conversion without handling the protocol value yourself. |
| <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.decode_response"><code>decode_response</code></a> | Provider JSON to <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html"><code>AggLlmResponse</code></a>, plus diagnostics. | Returning a buffered response to libsy. |
| <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.encode_response"><code>encode_response</code></a> | <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html"><code>AggLlmResponse</code></a> to provider JSON, plus diagnostics. | Returning a buffered answer in the caller's format. |
| <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.translate_response"><code>translate_response</code></a> | Source provider JSON to target provider JSON. | Direct buffered-response conversion. |
| <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.decode_stream_event"><code>decode_stream_event</code></a> | One parsed provider event to <a href="../../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html"><code>LlmResponseStreamEvent</code></a>. | Retaining the source event alongside neutral chunks. |
| <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.encode_stream_event"><code>encode_stream_event</code></a> | One <a href="../../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html"><code>LlmResponseStreamEvent</code></a> to zero or more provider events. | Replaying or converting incremental output. |
| <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.finish_stream"><code>finish_stream</code></a> | Stream state to any remaining target events. | Closing the target format after the source stream ends. |

Buffered encoders return <a href="../../../rust/switchyard_translation/engine/struct.TranslationOutput.html"><code>TranslationOutput</code></a> with the encoded body and
<a href="../../../rust/switchyard_translation/diagnostic/struct.TranslationDiagnostic.html"><code>TranslationDiagnostic</code></a> values. Inspect diagnostics from decoding and encoding;
the direct translation helpers combine them and annotate source/target formats.

## Policies

These defaults come from <a href="../../../rust/switchyard_translation/policy/struct.TranslationPolicy.html#method.default"><code>TranslationPolicy::default()</code></a>.

| Field and type | Default | Meaning |
|---|---|---|
| <a href="../../../rust/switchyard_translation/policy/struct.TranslationPolicy.html#structfield.unknown_field_policy"><code>TranslationPolicy.unknown_field_policy: UnknownFieldPolicy</code></a> | <a href="../../../rust/switchyard_translation/policy/enum.UnknownFieldPolicy.html#variant.Preserve"><code>UnknownFieldPolicy::Preserve</code></a> | Keep unknown provider fields where the codec can retain them. Alternatives warn and drop, or reject. |
| <a href="../../../rust/switchyard_translation/policy/struct.TranslationPolicy.html#structfield.lossy_conversion_policy"><code>TranslationPolicy.lossy_conversion_policy: LossyConversionPolicy</code></a> | <a href="../../../rust/switchyard_translation/policy/enum.LossyConversionPolicy.html#variant.AllowWithDiagnostics"><code>LossyConversionPolicy::AllowWithDiagnostics</code></a> | Allow known losses and record warnings. Use <a href="../../../rust/switchyard_translation/policy/enum.LossyConversionPolicy.html#variant.Reject"><code>LossyConversionPolicy::Reject</code></a> to fail on known lossy conversions. |
| <a href="../../../rust/switchyard_translation/policy/struct.TranslationPolicy.html#structfield.deterministic_ids"><code>TranslationPolicy.deterministic_ids: DeterministicIdPolicy</code></a> | <a href="../../../rust/switchyard_translation/policy/enum.DeterministicIdPolicy.html#variant.GenerateStable"><code>DeterministicIdPolicy::GenerateStable</code></a>, prefix `sw` | Generate stable IDs when needed. The alternative preserves IDs without generation. |
| <a href="../../../rust/switchyard_translation/policy/struct.TranslationPolicy.html#structfield.preservation"><code>TranslationPolicy.preservation: PreservationPolicy</code></a> | <a href="../../../rust/switchyard_translation/policy/enum.PreservationPolicy.html#variant.InMemory"><code>PreservationPolicy::InMemory</code></a> | Retain original parsed bodies in protocol values for replay. |
| <a href="../../../rust/switchyard_translation/policy/struct.TranslationPolicy.html#structfield.target_capabilities"><code>TranslationPolicy.target_capabilities: TargetCapabilities</code></a> | All capability fields unspecified | Optionally constrain supported tools, media, reasoning, schema output, and other target features. |

Unspecified capabilities are not proof that a model supports a feature. A
provider can still reject the encoded request. Known-loss rejection covers the
codec's implemented checks, rather than proving semantic equivalence for every
possible provider extension.

## Preservation and edits

**An original body takes precedence over reconstructed fields for its own format.**
The codec looks up a request in <a href="../../../rust/switchyard_protocol/llm/struct.PreservationMetadata.html#structfield.requests"><code>PreservationMetadata.requests</code></a>, or a buffered
response in <a href="../../../rust/switchyard_protocol/llm/struct.PreservationMetadata.html#structfield.responses"><code>PreservationMetadata.responses</code></a>.

| Policy | Behavior | Boundary |
|---|---|---|
| <a href="../../../rust/switchyard_translation/policy/enum.PreservationPolicy.html#variant.InMemory"><code>PreservationPolicy::InMemory</code></a> | Keep original JSON in protocol values. | Same-format replay works while that preservation data remains available. |
| <a href="../../../rust/switchyard_translation/policy/enum.PreservationPolicy.html#variant.Embed"><code>PreservationPolicy::Embed</code></a> | Also embed preservation metadata in translated JSON. | Supports multi-hop round trips when intermediaries retain it; provider metadata limits can prevent use. |
| <a href="../../../rust/switchyard_translation/policy/enum.PreservationPolicy.html#variant.Disabled"><code>PreservationPolicy::Disabled</code></a> | Do not use preservation for replay. | Encoders reconstruct the body from current protocol fields. |

If you edit a decoded request, remove that format's entry from
<a href="../../../rust/switchyard_protocol/llm/struct.PreservationMetadata.html#structfield.requests"><code>PreservationMetadata.requests</code></a> or encode with <a href="../../../rust/switchyard_translation/policy/enum.PreservationPolicy.html#variant.Disabled"><code>PreservationPolicy::Disabled</code></a>.
This applies to model changes, messages, generation settings, and schema
enforcement. Apply the same rule to buffered response edits.

Replay preserves parsed JSON values, not original HTTP bytes, whitespace, or
headers. Input cleanup can happen before capture, including removal of Codex
internal compaction markers. <a href="../../../rust/switchyard_protocol/envelope/struct.Request.html#structfield.raw_request"><code>Request.raw_request</code></a> is host-owned and is not the
codec's replay source. Preserved data can contain the full conversation and
other sensitive provider fields.

## Stream translation

Use a separate <a href="../../../rust/switchyard_translation/codecs/stream/struct.StreamTranslationState.html"><code>StreamTranslationState</code></a> for each response stream. Feed parsed
provider events to <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.decode_stream_event"><code>decode_stream_event</code></a>, pass the resulting event to
<a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.encode_stream_event"><code>encode_stream_event</code></a>, and call <a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.finish_stream"><code>finish_stream</code></a> when the source closes.
The encoder may emit several target events for one input event.

When the source and target formats match, a preserved event replays its retained
JSON. Cross-format encoding uses normalized chunks. After changing a stream
event, use <a href="../../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html#method.replace_normalized"><code>replace_normalized()</code></a> to clear raw data that no longer describes it.
The [response reference](llm_responses.md#stream-lifetime-and-preservation)
explains what aggregation retains and drops.

## Conversion limits

Retention and destination support are separate. Anthropic has no distinct
developer authority, the OpenAI formats have no standard tool-result error flag,
and provider built-in tools or encrypted reasoning may lack equivalents.

Use <a href="../../../rust/switchyard_protocol/llm/struct.ProviderExtensions.html"><code>ProviderExtensions</code></a> and preservation to retain source data, then inspect
diagnostics and the destination model's capabilities. For a custom format,
<a href="../../../rust/switchyard_translation/codecs/trait.FormatCodec.html"><code>FormatCodec</code></a> and <a href="../../../rust/switchyard_translation/codecs/stream/trait.StreamCodec.html"><code>StreamCodec</code></a> define the codec contracts;
<a href="../../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.with_registries"><code>TranslationEngine::with_registries</code></a> accepts explicit registries.
