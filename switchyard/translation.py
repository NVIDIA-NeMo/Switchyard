# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate buffered provider requests and responses.

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
Translate and encode results have "body" and "diagnostics" keys. decode_request
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
"""

from switchyard_rust.translation import TranslationEngine, TranslationError, WireFormat

__all__ = ["TranslationEngine", "TranslationError", "WireFormat"]
