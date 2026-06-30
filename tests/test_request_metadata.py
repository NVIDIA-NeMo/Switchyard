# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for header helpers in :mod:`switchyard.lib.request_metadata`."""

from __future__ import annotations

import pytest

from switchyard.lib.proxy_context import CTX_CALLER_API_KEY, ProxyContext
from switchyard.lib.request_metadata import (
    CTX_PROFILE_REQUEST_HEADERS,
    attach_caller_api_key,
    attach_request_metadata,
    extract_caller_api_key,
    redact_sensitive_headers,
)
from switchyard_rust.components import RequestMetadata


class TestExtractCallerApiKey:
    """``extract_caller_api_key`` parses caller credentials from HTTP headers.

    Multi-tenant deploys forward each caller's key per request. The dedicated
    ``x-switchyard-api-key`` header is preferred (it survives proxies such as
    LiteLLM that strip ``Authorization``); ``Authorization: Bearer <key>`` and
    ``x-api-key`` remain supported for direct callers. The codex launcher sends
    ``"switchyard"`` as a sentinel placeholder, which must not be forwarded.
    """

    @pytest.mark.parametrize(
        "headers, expected",
        [
            ({"Authorization": "Bearer nvapi-real"}, "nvapi-real"),
            ({"authorization": "bearer nvapi-lowercase"}, "nvapi-lowercase"),
            ({"x-api-key": "nvapi-via-x-api-key"}, "nvapi-via-x-api-key"),
            ({"X-Api-Key": "nvapi-titlecase-header"}, "nvapi-titlecase-header"),
            # The dedicated forwarded header is honored...
            ({"x-switchyard-api-key": "nvapi-forwarded"}, "nvapi-forwarded"),
            ({"X-Switchyard-Api-Key": "nvapi-fwd-titlecase"}, "nvapi-fwd-titlecase"),
            # ...and wins over Authorization and x-api-key when several are set
            # (the case behind a proxy that strips Authorization upstream).
            (
                {
                    "x-switchyard-api-key": "forwarded-wins",
                    "Authorization": "Bearer lose",
                    "x-api-key": "lose-too",
                },
                "forwarded-wins",
            ),
            # Authorization wins over x-api-key when both are present.
            (
                {"Authorization": "Bearer first", "x-api-key": "second"},
                "first",
            ),
            # Surrounding whitespace is stripped.
            ({"Authorization": "Bearer   nvapi-spaces   "}, "nvapi-spaces"),
            ({"x-switchyard-api-key": "  nvapi-fwd-spaces  "}, "nvapi-fwd-spaces"),
        ],
    )
    def test_extraction(self, headers: dict[str, str], expected: str) -> None:
        assert extract_caller_api_key(headers) == expected

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": ""},
            {"Authorization": "Bearer "},
            # Non-bearer schemes are not forwarded.
            {"Authorization": "Basic dXNlcjpwYXNz"},
            # Codex launcher sentinel — in any supported header.
            {"Authorization": "Bearer switchyard"},
            {"x-api-key": "switchyard"},
            {"x-switchyard-api-key": "switchyard"},
            {"x-switchyard-api-key": ""},
            # Case-insensitive sentinel match — both halves of the case
            # space should be treated as the same placeholder.
            {"Authorization": "Bearer Switchyard"},
        ],
    )
    def test_no_key_returned(self, headers: dict[str, str]) -> None:
        assert extract_caller_api_key(headers) is None


class TestRedactSensitiveHeaders:
    """Credential headers are scrubbed before the header map is retained."""

    def test_redacts_credential_headers(self) -> None:
        headers = {
            "Authorization": "Bearer nvapi-real",
            "x-api-key": "nvapi-x",
            "x-switchyard-api-key": "nvapi-forwarded",
            "x-switchyard-intake-app": "demo",
            "content-type": "application/json",
        }
        redacted = redact_sensitive_headers(headers)
        assert redacted["Authorization"] == "[REDACTED]"
        assert redacted["x-api-key"] == "[REDACTED]"
        assert redacted["x-switchyard-api-key"] == "[REDACTED]"
        # Non-credential headers pass through untouched.
        assert redacted["x-switchyard-intake-app"] == "demo"
        assert redacted["content-type"] == "application/json"

    def test_matching_is_case_insensitive(self) -> None:
        redacted = redact_sensitive_headers({"X-Switchyard-Api-Key": "nvapi-real"})
        assert redacted["X-Switchyard-Api-Key"] == "[REDACTED]"


class TestCallerKeyForwardedButNotRetained:
    """The endpoint extracts the caller key for upstream use, then retains a
    redacted header map so the key cannot leak into profile metadata, intake,
    logs, or traces."""

    def test_key_extracted_but_redacted_in_stored_headers(self) -> None:
        headers = {
            "x-switchyard-api-key": "nvapi-secret",
            "x-switchyard-intake-app": "demo",
        }
        ctx = ProxyContext()
        # Mirror the endpoint: both helpers receive the raw headers.
        attach_request_metadata(ctx, RequestMetadata.from_headers(headers), headers)
        attach_caller_api_key(ctx, headers)

        # Extracted for upstream forwarding...
        assert ctx.metadata[CTX_CALLER_API_KEY] == "nvapi-secret"
        # ...but the retained header map carries no raw credential.
        stored = ctx.metadata[CTX_PROFILE_REQUEST_HEADERS]
        assert stored["x-switchyard-api-key"] == "[REDACTED]"
        assert stored["x-switchyard-intake-app"] == "demo"
