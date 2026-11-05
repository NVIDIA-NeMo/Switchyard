# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate provider requests, responses, and streams.

    from switchyard.translation import TranslationEngine, WireFormat

    engine = TranslationEngine()
    result = engine.translate_request(
        WireFormat.OPENAI_CHAT,
        WireFormat.ANTHROPIC_MESSAGES,
        {"model": "example", "messages": [{"role": "user", "content": "Hello"}]},
    )
    body = result["body"]
    diagnostics = result["diagnostics"]

All methods accept a WireFormat member or its string value for each format.
Bodies and IR values must be dictionaries containing JSON-compatible values.
Buffered translate and encode results have "body" and "diagnostics" keys. decode_request
returns "request" and "diagnostics" keys; decode_response returns "response" and
"diagnostics" keys. Diagnostics contain severity (Info,
Warning, or Error), code, message, source, target, and path; the last three fields
may be None. The default policy allows lossy conversions with diagnostics.

Incorrect top-level argument types raise TypeError. Malformed IR dictionaries raise
ValueError during deserialization. Unknown formats and translation
failures raise TranslationError, whose kind and details attributes
identify the failure. str(error) provides a human-readable message.

To restore Responses tool namespaces and custom tool calls, pass the original
decoded request's extensions when encoding the response::

    decoded_request = engine.decode_request(source_format, request_body)
    encoded = engine.encode_request(backend_format, decoded_request["request"])

    # Send encoded["body"] to the backend and receive backend_response.
    decoded_response = engine.decode_response(backend_format, backend_response)

    result = engine.encode_response(
        source_format,
        decoded_response["response"],
        request_extensions=decoded_request["request"]["extensions"],
    )

request_extensions uses the IR shape {"fields": {...}}. Omitting it, passing None,
or passing an empty dictionary or {"fields": {}} all encode without request context.

For streaming format conversion, translate SSE byte chunks into provider events::

    async with engine.decode_stream(backend_format, byte_stream) as decoded:
        async with engine.encode_stream(
            source_format,
            decoded,
            served_model="selected-model",
            request_extensions=decoded_request["request"]["extensions"],
        ) as encoded:
            async for event in encoded:
                handle_provider_event(event)

Alternately the translate_stream method can be used::

    async with engine.translate_stream(
        backend_format,
        source_format,
        byte_stream,
        served_model="selected-model",
        request_extensions=decoded_request["request"]["extensions"],
    ) as encoded:
        async for event in encoded:
            handle_provider_event(event)

translate_stream composes decode_stream and encode_stream in Python and closes
both internal iterators. Use the separate methods when working with IR events.

The stream methods return async iterators that also support async context management.
Prefer async with to release translation resources on early exit. Each iterator owns
its translation state and can outlive the engine. The caller owns the input stream
and its transport; closing a translation iterator never calls the input's close
methods. Cancelling a read ends that translation iterator and cancels its pending
input read, which may also terminate the input iterator. Cleanup errors propagate.

Decoded items are IR dictionaries with preservation and normalized keys. Encoded
items are provider event dictionaries; the caller supplies any SSE output framing.
Rust handles normal stream finalization. Closing early does not emit completion
events. There is no separate diagnostics list for streaming: DecodeError and
StreamError remain IR chunks, while Rust stream failures raise TranslationError.
Python input exceptions retain their original types. Streaming TranslationError
kinds also include client categories such as ResponseTranslation and Configuration;
UpstreamFailure preserves the encoded error payload in details["error"].
"""

from switchyard_rust.translation import TranslationEngine, TranslationError, WireFormat

__all__ = ["TranslationEngine", "TranslationError", "WireFormat"]
