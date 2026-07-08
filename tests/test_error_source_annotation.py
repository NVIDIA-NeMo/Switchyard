# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""failure-source annotation on error responses and the event log.

Every client-facing error carries ``x-switchyard-error-source`` naming the
layer that originated it (``switchyard`` | ``provider``), plus
``x-switchyard-upstream-model`` when a routing selection had happened. The
backend-side ctx stamps are covered in ``test_latency_service_llm_backend.py``;
these tests cover the endpoint layer that renders them.
"""

from __future__ import annotations

import json
import logging

import pytest

from switchyard.lib.endpoints.dispatch import model_not_found_response
from switchyard.lib.endpoints.error_envelope import (
    ERROR_SOURCE_HEADER,
    UPSTREAM_MODEL_HEADER,
    error_response,
    upstream_error_response,
)
from switchyard.lib.endpoints.upstream_error import (
    context_exhausted_response,
    handle_chain_exception,
    upstream_response_from_ctx,
)
from switchyard.lib.endpoints.upstream_error_log import log_upstream_attempt_failure
from switchyard.lib.proxy_context import (
    CTX_ERROR_SOURCE,
    CTX_UPSTREAM_HTTP_BODY,
    CTX_UPSTREAM_HTTP_STATUS,
    CTX_UPSTREAM_MODEL,
    ProxyContext,
)

_UPSTREAM_401 = {
    "error": {"message": "bad key", "type": "auth_error", "code": "invalid_api_key"}
}


# --- envelope builders -------------------------------------------------------


def test_synthesized_envelope_defaults_to_switchyard_source() -> None:
    resp = error_response(400, "bad", error_type="invalid_request_error", code="invalid_body")
    assert resp.headers[ERROR_SOURCE_HEADER] == "switchyard"
    assert UPSTREAM_MODEL_HEADER not in resp.headers


def test_upstream_envelope_labels_provider_and_upstream_model() -> None:
    resp = upstream_error_response(429, _UPSTREAM_401, upstream_model="gpt-5")
    assert resp.headers[ERROR_SOURCE_HEADER] == "provider"
    assert resp.headers[UPSTREAM_MODEL_HEADER] == "gpt-5"


def test_model_not_found_labels_switchyard() -> None:
    resp = model_not_found_response("nope")
    assert resp.status_code == 404
    assert resp.headers[ERROR_SOURCE_HEADER] == "switchyard"


def test_context_exhausted_labels_switchyard() -> None:
    resp = context_exhausted_response(RuntimeError("pool exhausted"), "openai")
    assert resp.status_code == 400
    assert resp.headers[ERROR_SOURCE_HEADER] == "switchyard"


# --- ctx-driven recovery paths ----------------------------------------------


def _ctx_with_upstream_stash(**extra: object) -> ProxyContext:
    ctx = ProxyContext()
    ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 401
    ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = _UPSTREAM_401
    for key, value in extra.items():
        ctx.metadata[key] = value
    return ctx


def test_stashed_status_defaults_to_provider() -> None:
    """A backend that stashes an upstream status without a source label gets
    the passthrough default — the stash channel exists for provider errors."""
    resp = upstream_response_from_ctx(_ctx_with_upstream_stash())
    assert resp is not None
    assert resp.status_code == 401
    assert resp.headers[ERROR_SOURCE_HEADER] == "provider"
    assert UPSTREAM_MODEL_HEADER not in resp.headers


def test_stashed_switchyard_source_overrides_provider_default() -> None:
    """caller_required-style rejections ride the upstream channel but must
    surface as switchyard-originated."""
    ctx = _ctx_with_upstream_stash(**{CTX_ERROR_SOURCE: "switchyard"})
    resp = upstream_response_from_ctx(ctx)
    assert resp is not None
    assert resp.headers[ERROR_SOURCE_HEADER] == "switchyard"


def test_stashed_upstream_model_reaches_header() -> None:
    ctx = _ctx_with_upstream_stash(**{CTX_UPSTREAM_MODEL: "gpt-5"})
    resp = upstream_response_from_ctx(ctx)
    assert resp is not None
    assert resp.headers[UPSTREAM_MODEL_HEADER] == "gpt-5"


def test_unexpected_internal_500_labels_switchyard() -> None:
    resp = handle_chain_exception(
        RuntimeError("boom"), ProxyContext(), inbound="openai", log_msg="test failure"
    )
    assert resp.status_code == 500
    assert resp.headers[ERROR_SOURCE_HEADER] == "switchyard"
    assert UPSTREAM_MODEL_HEADER not in resp.headers


def test_network_failure_500_labels_provider() -> None:
    """A status-less upstream fault (network error after retries) renders as
    the internal 500 envelope but is labeled provider via the ctx stamp."""
    ctx = ProxyContext()
    ctx.metadata[CTX_ERROR_SOURCE] = "provider"
    ctx.metadata[CTX_UPSTREAM_MODEL] = "gpt-5"

    resp = handle_chain_exception(
        RuntimeError("connection reset"), ctx, inbound="openai", log_msg="test failure"
    )

    assert resp.status_code == 500
    assert json.loads(bytes(resp.body))["error"]["code"] == "internal_chain_error"
    assert resp.headers[ERROR_SOURCE_HEADER] == "provider"
    assert resp.headers[UPSTREAM_MODEL_HEADER] == "gpt-5"


# --- structured event log ----------------------------------------------------


def test_attempt_failure_log_carries_upstream_model_and_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="switchyard.upstream_errors"):
        log_upstream_attempt_failure(
            model="route-id",
            attempt=1,
            status_code=429,
            error=RuntimeError("rate limited"),
            upstream_model="gpt-5",
        )

    record = json.loads(caplog.records[-1].message)
    assert record["model"] == "route-id"
    assert record["upstream_model"] == "gpt-5"
    assert record["error_source"] == "provider"
    assert record["code"] == "429"
