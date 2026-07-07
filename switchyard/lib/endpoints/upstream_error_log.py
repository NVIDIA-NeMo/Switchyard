# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured per-attempt upstream-failure log for Loki ingestion.

The aggregate ``switchyard_upstream_attempts_total`` counter on ``/metrics``
answers "how many of each error code" but, by Prometheus' data model, holds
no per-event timestamps. This module is its per-event complement: it emits
one JSON line per *failed* upstream attempt on a dedicated logger, carrying
the exact event timestamp so a Loki/Grafana pipeline can audit, replay, or
plot individual failures.

The line *is* a JSON object (not a human sentence with structured ``extra``)
on purpose: Switchyard configures plain-text logging via ``logging.basicConfig``
with no JSON formatter, so embedding the document in the message is what makes
``| json`` work in a Loki query with zero deployment-side formatter config.
The dedicated ``switchyard.upstream_errors`` logger still propagates to the
root handler, so the line also shows on the console.

The ``code`` and ``outcome`` fields are computed with the same helpers as the
metric labels (:func:`~switchyard.lib.endpoints.outcome_metrics.code_label`,
:func:`~switchyard.lib.endpoints.outcome_metrics.classify`) so the event log
joins cleanly to ``switchyard_upstream_attempts_total``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from switchyard.lib.endpoints.outcome_metrics import classify, code_label

#: Dedicated logger so operators can route or level upstream-failure events
#: independently of the (noisier) backend logger. Propagates to root.
log = logging.getLogger("switchyard.upstream_errors")

#: Structured-log event name — the value a Loki query filters on
#: (``| json | event="upstream_attempt_failed"``).
EVENT_NAME = "upstream_attempt_failed"

#: Upstream error bodies can be large; cap the logged message so a single
#: pathological error cannot blow up a log line / Loki entry.
_MAX_ERROR_CHARS = 500


def log_upstream_attempt_failure(
    *,
    model: str,
    attempt: int,
    status_code: int | None,
    error: BaseException,
    upstream_model: str | None = None,
) -> None:
    """Emit one structured JSON record for a single failed upstream attempt.

    ``status_code`` is the raw upstream HTTP status, or ``None`` for a
    non-HTTP failure (network error, pre-status timeout) — recorded as
    ``status_code: null`` with ``code="none"``. ``attempt`` is 1-based.
    ``upstream_model`` is the model actually sent upstream (``body["model"]``)
    when the caller knows it; ``model`` remains the internal route/endpoint id.

    ``code`` and ``outcome`` mirror the labels on
    ``switchyard_upstream_attempts_total`` so the event log is joinable to
    the aggregate counter. ``error_source`` is always ``provider`` — this
    event exists only for failures of actual upstream attempts. The record
    is logged at WARNING.
    """
    record = {
        "event": EVENT_NAME,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model,
        "upstream_model": upstream_model,
        "attempt": attempt,
        "status_code": status_code,
        "code": code_label(status_code),
        # None (non-HTTP failure) is a retryable_error, matching how
        # record_upstream_attempt buckets it.
        "outcome": "retryable_error" if status_code is None else classify(status_code),
        # Attempt failures are upstream-side by definition; the constant field
        # keeps the event joinable to the response-header/span vocabulary.
        "error_source": "provider",
        "error_type": type(error).__name__,
        "error": str(error)[:_MAX_ERROR_CHARS],
    }
    # Compact separators keep the line small; the message is valid JSON.
    log.warning(json.dumps(record, separators=(",", ":")))


__all__ = ["EVENT_NAME", "log_upstream_attempt_failure"]
