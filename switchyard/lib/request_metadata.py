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
INTAKE_ENABLED_HEADER = "x-switchyard-intake-enabled"
INTAKE_APP_HEADER = "x-switchyard-intake-app"
INTAKE_TASK_HEADER = "x-switchyard-intake-task"

# Sentinel values our own launchers send as the ``Authorization`` /
# ``OPENAI_API_KEY`` value so coding agents satisfy their "no key set"
# preconditions. Treat as if no key was supplied. The codex launcher
# sets ``OPENAI_API_KEY="switchyard"`` (see codex_cli_launcher.py).
_CALLER_KEY_SENTINELS = frozenset({"switchyard", ""})

# Dedicated forwarded credential header. Preferred over ``Authorization``
# because a proxy in front of Switchyard (e.g. LiteLLM) consumes the
# ``Authorization`` header for its own auth and strips it before the upstream
# call, while a custom header passes through untouched — so a BYO-key caller
# behind such a proxy stays correctly attributed for upstream inference spend.
CALLER_API_KEY_HEADER = "x-switchyard-api-key"  # pragma: allowlist secret

# Request headers whose values carry a caller credential. Redacted before the
# header map is retained on the context (``CTX_PROFILE_REQUEST_HEADERS``), so the
# caller's key never reaches profile metadata, intake, logs, or traces. The key
# is still forwarded upstream via ``CTX_CALLER_API_KEY`` (extracted by
# ``attach_caller_api_key`` from the raw headers, before redaction).
_SENSITIVE_HEADERS = frozenset({"authorization", "x-api-key", CALLER_API_KEY_HEADER})
_REDACTED = "[REDACTED]"


def attach_request_metadata(
    ctx: Any,
    metadata: RequestMetadata,
    headers: Mapping[str, str] | None = None,
) -> None:
    """Attach request metadata to both Python and Rust-owned context storage."""
    ctx.metadata[CTX_REQUEST_METADATA] = metadata
    if headers is not None:
        # Redact credential headers before retaining the map: the caller key is
        # already extracted into ``CTX_CALLER_API_KEY`` for upstream forwarding,
        # so nothing downstream needs the raw value, and a retained/logged header
        # map must not expose it.
        ctx.metadata[CTX_PROFILE_REQUEST_HEADERS] = redact_sensitive_headers(headers)
    metadata.apply_to_context(ctx)


def redact_sensitive_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of *headers* with credential-bearing values redacted.

    Headers named in :data:`_SENSITIVE_HEADERS` (``Authorization``, ``x-api-key``,
    ``x-switchyard-api-key``) have their values replaced with ``"[REDACTED]"`` so a
    retained or logged header map can never expose the caller's API key.
    Header-name matching is case-insensitive.
    """
    return {
        name: (_REDACTED if name.lower() in _SENSITIVE_HEADERS else value)
        for name, value in headers.items()
    }


def attach_caller_api_key(ctx: Any, headers: Mapping[str, str]) -> None:
    """Attach the caller-supplied API key to *ctx* when the request carries one."""
    caller_key = extract_caller_api_key(headers)
    if caller_key is not None:
        ctx.metadata[CTX_CALLER_API_KEY] = caller_key


def extract_caller_api_key(headers: Mapping[str, str]) -> str | None:
    """Pull the caller-supplied API key out of an HTTP request's headers.

    Precedence: the dedicated ``x-switchyard-api-key`` header first, then
    ``Authorization: Bearer <key>``, then ``x-api-key``. The dedicated header is
    preferred because a proxy in front of Switchyard (e.g. LiteLLM) strips
    ``Authorization`` before the upstream call; the custom header survives, so the
    caller's key — not a service key — is the credential billed for upstream
    inference. Returns ``None`` when no usable header is present, the bearer
    scheme is missing, or the value is a known launcher sentinel (so coding-agent
    placeholder keys do not get forwarded upstream as real credentials).
    """
    forwarded = headers.get(CALLER_API_KEY_HEADER) or headers.get("X-Switchyard-Api-Key")
    if forwarded:
        candidate = forwarded.strip()
        if candidate.lower() not in _CALLER_KEY_SENTINELS:
            return candidate
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
    "CALLER_API_KEY_HEADER",
    "CTX_REQUEST_METADATA",
    "CTX_PROFILE_REQUEST_HEADERS",
    "INTAKE_APP_HEADER",
    "INTAKE_ENABLED_HEADER",
    "INTAKE_TASK_HEADER",
    "PROXY_SESSION_ID_HEADER",
    "IntakeRequestMetadata",
    "RequestMetadata",
    "attach_caller_api_key",
    "attach_request_metadata",
    "extract_caller_api_key",
    "redact_sensitive_headers",
]
