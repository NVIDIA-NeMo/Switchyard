# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ProxyContext constants and Rust-owned context binding."""

from switchyard_rust.core import ProxyContext

# ---------------------------------------------------------------------------
# Metadata key constants
# ---------------------------------------------------------------------------
# Use these instead of bare strings to avoid silent typos and to make
# cross-component key contracts discoverable at import time.

#: Stores a deep-copy of the incoming request dict.
#: Written by RequestBufferProcessor.
CTX_ORIGINAL_REQUEST = "original_request"

#: Model that was actually used for the LLM call, after any routing/override.
#: Written by routing backends and processors.
CTX_PROXY_ACTUAL_MODEL = "_proxy_actual_model"

#: Routing metadata dict produced by RouteLLMRequestProcessor.
CTX_ROUTING = "_routing"

#: Original inbound format stored by translation layers.
CTX_ORIGINAL_FORMAT = "_original_format"

#: Original model name stored by translation layers.
CTX_ORIGINAL_MODEL = "_original_model"

#: Target wire format chosen by a router (e.g. random routing, RouteLLM).
#: Written by router request-side components; read by
#: FormatTranslateRequestProcessor.
CTX_TARGET_FORMAT = "_target_format"

#: Caller-supplied API key extracted from the inbound request's
#: ``x-switchyard-api-key`` header (preferred — survives proxies that strip
#: ``Authorization``), or ``Authorization: Bearer <key>`` / ``x-api-key``. Set by
#: the HTTP endpoint after header parsing; consumed by backends that support
#: opt-in per-caller credential forwarding.
#: Absent when the caller did not supply a credential or supplied a known
#: launcher-sentinel placeholder.
CTX_CALLER_API_KEY = "_caller_api_key"

#: Upstream HTTP status code recorded by a Python backend when an LLM
#: provider returns a non-2xx response. Endpoints inspect this on the
#: error path to passthrough the upstream status (e.g. 401 from a bad
#: API key) instead of masking it as a 500. Rust backend errors don't
#: round-trip structurally — they are reported as opaque strings via
#: ``SwitchyardError::Upstream`` — so this is currently the Python-side
#: channel only.
CTX_UPSTREAM_HTTP_STATUS = "_upstream_http_status"

#: Upstream HTTP response body recorded alongside
#: :data:`CTX_UPSTREAM_HTTP_STATUS`. May be a string or a JSON-decodable
#: dict — endpoints pass it through to the caller verbatim.
CTX_UPSTREAM_HTTP_BODY = "_upstream_http_body"

#: Set truthy by a backend that records its own per-attempt
#: ``switchyard.lib.endpoints.outcome_metrics`` upstream-attempt counters
#: (e.g. :class:`LatencyServiceLLMBackend`, which retries across endpoints and
#: must count each attempt). When present, the endpoint layer skips its
#: single-attempt fallback recording for this request so retry fan-out is not
#: double-counted. Absent for the Rust native / passthrough / multi backends,
#: which issue exactly one upstream attempt per call and have no Python retry
#: loop — those rely on the endpoint fallback.
CTX_UPSTREAM_ATTEMPTS_RECORDED = "_upstream_attempts_recorded"


__all__ = [
    "CTX_CALLER_API_KEY",
    "CTX_ORIGINAL_FORMAT",
    "CTX_ORIGINAL_MODEL",
    "CTX_ORIGINAL_REQUEST",
    "CTX_PROXY_ACTUAL_MODEL",
    "CTX_ROUTING",
    "CTX_TARGET_FORMAT",
    "CTX_UPSTREAM_ATTEMPTS_RECORDED",
    "CTX_UPSTREAM_HTTP_BODY",
    "CTX_UPSTREAM_HTTP_STATUS",
    "ProxyContext",
]
