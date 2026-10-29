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

Both methods accept a WireFormat member or its string value for each format.
Bodies must be dictionaries containing JSON-compatible values. Each result has
"body" and "diagnostics" keys. Diagnostics are dictionaries with severity (Info,
Warning, or Error), code, message, source, target, and path; the last three fields
may be None. The default policy allows lossy conversions with diagnostics.

Incorrect top-level argument types raise TypeError. Unknown formats and translation
failures raise TranslationError, whose kind and details attributes
identify the failure. str(error) provides a human-readable message.

Translation uses the default policy and handles complete requests and responses.
Calls do not retain request context for subsequent response translation.
"""

from switchyard_rust.translation import TranslationEngine, TranslationError, WireFormat

__all__ = ["TranslationEngine", "TranslationError", "WireFormat"]
