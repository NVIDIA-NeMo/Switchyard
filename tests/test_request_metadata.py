# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for header helpers in :mod:`switchyard.lib.request_metadata`."""

from __future__ import annotations

import pytest

from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.request_metadata import (
    RequestMetadata,
    attach_request_metadata,
    extract_caller_api_key,
)


class TestExtractCallerApiKey:
    """``extract_caller_api_key`` parses caller credentials from HTTP headers.

    Multi-tenant deploys forward each caller's key per request via
    ``Authorization: Bearer <key>``; ``x-api-key`` is a documented
    fallback. The codex launcher sends ``"switchyard"`` as a sentinel
    placeholder, which must not be forwarded upstream.
    """

    @pytest.mark.parametrize(
        "headers, expected",
        [
            ({"Authorization": "Bearer nvapi-real"}, "nvapi-real"),
            ({"authorization": "bearer nvapi-lowercase"}, "nvapi-lowercase"),
            ({"x-api-key": "nvapi-via-x-api-key"}, "nvapi-via-x-api-key"),
            ({"X-Api-Key": "nvapi-titlecase-header"}, "nvapi-titlecase-header"),
            # Authorization wins over x-api-key when both are present.
            (
                {"Authorization": "Bearer first", "x-api-key": "second"},
                "first",
            ),
            # Surrounding whitespace is stripped.
            ({"Authorization": "Bearer   nvapi-spaces   "}, "nvapi-spaces"),
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
            # Codex launcher sentinel.
            {"Authorization": "Bearer switchyard"},
            {"x-api-key": "switchyard"},
            # Case-insensitive sentinel match — both halves of the case
            # space should be treated as the same placeholder.
            {"Authorization": "Bearer Switchyard"},
        ],
    )
    def test_no_key_returned(self, headers: dict[str, str]) -> None:
        assert extract_caller_api_key(headers) is None


def test_attach_request_metadata_correlates_generic_request_id() -> None:
    """The opaque request ID crosses the HTTP/profile compatibility boundary."""
    ctx = ProxyContext()

    attach_request_metadata(
        ctx,
        RequestMetadata(),
        {"X-Request-Id": "benchmark-request-1"},
    )

    assert ctx.request_id == "benchmark-request-1"


def test_attach_request_metadata_preserves_existing_request_id() -> None:
    ctx = ProxyContext(request_id="existing-request")

    attach_request_metadata(
        ctx,
        RequestMetadata(),
        {"x-request-id": "header-request"},
    )

    assert ctx.request_id == "existing-request"


def test_attach_request_metadata_ignores_blank_request_id() -> None:
    ctx = ProxyContext()

    attach_request_metadata(ctx, RequestMetadata(), {"x-request-id": "  "})

    assert ctx.request_id is None
