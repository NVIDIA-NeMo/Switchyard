# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""request metadata helpers for HTTP endpoint context."""

from collections.abc import Mapping
from typing import Any

from switchyard.lib.proxy_context import CTX_CALLER_API_KEY
from switchyard_rust.components import IntakeRequestMetadata, RequestMetadata

CTX_REQUEST_METADATA = "_request_metadata"
CTX_PROFILE_REQUEST_HEADERS = "_profile_request_headers"

# Existing Switchyard session header. Do not add aliases here unless a
# concrete client requires one; keeping one spelling avoids ambiguity.
PROXY_SESSION_ID_HEADER = "proxy_x_session_id"
REQUEST_ID_HEADER = "x-request-id"
INTAKE_ENABLED_HEADER = "x-switchyard-intake-enabled"
INTAKE_APP_HEADER = "x-switchyard-intake-app"
INTAKE_TASK_HEADER = "x-switchyard-intake-task"

# Sentinel values our own launchers send as the ``Authorization`` /
# ``OPENAI_API_KEY`` value so coding agents satisfy their "no key set"
# preconditions. Treat as if no key was supplied. The codex launcher
# sets ``OPENAI_API_KEY="switchyard"`` (see codex_cli_launcher.py).
_CALLER_KEY_SENTINELS = frozenset({"switchyard", ""})


def attach_request_metadata(
    ctx: Any,
    metadata: RequestMetadata,
    headers: Mapping[str, str] | None = None,
) -> None:
    """Attach request metadata to both Python and Rust-owned context storage."""
    ctx.metadata[CTX_REQUEST_METADATA] = metadata
    if headers is not None:
        ctx.metadata[CTX_PROFILE_REQUEST_HEADERS] = dict(headers)
        if ctx.request_id is None:
            request_id = _case_insensitive_header(headers, REQUEST_ID_HEADER)
            if request_id:
                ctx.request_id = request_id
    metadata.apply_to_context(ctx)


def _case_insensitive_header(headers: Mapping[str, str], name: str) -> str | None:
    """Return one non-empty header value without assuming normalized names."""
    expected = name.casefold()
    for header_name, value in headers.items():
        normalized = value.strip()
        if header_name.casefold() == expected and normalized:
            return normalized
    return None


def attach_caller_api_key(ctx: Any, headers: Mapping[str, str]) -> None:
    """Attach the caller-supplied API key to *ctx* when the request carries one."""
    caller_key = extract_caller_api_key(headers)
    if caller_key is not None:
        ctx.metadata[CTX_CALLER_API_KEY] = caller_key


def extract_caller_api_key(headers: Mapping[str, str]) -> str | None:
    """Pull the caller-supplied API key out of an HTTP request's headers.

    Precedence: ``Authorization: Bearer <key>`` first, then ``x-api-key``.
    Returns ``None`` when neither header is present, the bearer scheme is
    missing, or the value is a known launcher sentinel (so coding-agent
    placeholder keys do not get forwarded upstream as real credentials).
    """
    auth = headers.get("authorization") or headers.get("Authorization")
    if auth:
        scheme, _, value = auth.partition(" ")
        if scheme.lower() == "bearer" and value:
            candidate = value.strip()
            if candidate.lower() not in _CALLER_KEY_SENTINELS:
                return candidate
    api_key = headers.get("x-api-key") or headers.get("X-Api-Key")
    if api_key:
        candidate = api_key.strip()
        if candidate.lower() not in _CALLER_KEY_SENTINELS:
            return candidate
    return None


__all__ = [
    "CTX_REQUEST_METADATA",
    "CTX_PROFILE_REQUEST_HEADERS",
    "INTAKE_APP_HEADER",
    "INTAKE_ENABLED_HEADER",
    "INTAKE_TASK_HEADER",
    "PROXY_SESSION_ID_HEADER",
    "REQUEST_ID_HEADER",
    "IntakeRequestMetadata",
    "RequestMetadata",
    "attach_caller_api_key",
    "attach_request_metadata",
    "extract_caller_api_key",
]
