# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outcome counters used to compute router-vs-direct error-rate ratios.

Four process-wide counters published on ``/metrics``:

* ``switchyard_client_responses_total{outcome}`` — every HTTP response
  returned to a client on an LLM-serving route. The denominator for the
  router-served error rate.
* ``switchyard_upstream_attempts_total{outcome, code}`` — every individual
  upstream call attempt (including ones absorbed by retry). The
  denominator for the direct-to-endpoint baseline error rate. The ``code``
  label carries the raw upstream HTTP status (``"429"``, ``"500"`` …) so a
  dashboard can plot the error-code distribution over time;
  ``code="none"`` marks a non-HTTP failure (network error, pre-status
  timeout) that has no status code. Unknown codes are clamped to their
  class (``"4xx"`` / ``"5xx"`` / …) to keep label cardinality bounded.
* ``switchyard_router_retry_recovered_total`` — global counter
  incremented whenever a request's first upstream attempt failed and a
  subsequent attempt succeeded — direct evidence the steering logic
  rescued the request.
* ``switchyard_classifier_fail_open_triggered_total{reason}`` — classifier
  fallback events where routing used the default tier because the classifier
  could not produce a usable verdict.

Bucket semantics:

* ``success``        — HTTP 2xx.
* ``retryable_error`` — HTTP 429 / 500 / 504 — the categories the
  success criterion measures (router should be absorbing these).
* ``other_error``    — everything else (400 / 401 / 403 / 422 / …),
  i.e. bad payload, bad credentials, high-reasoning timeout. Excluded
  from the success criterion per the spec.

Computing the ratios from these::

    router_error_rate = client_responses{outcome="retryable_error"}
                      / sum(client_responses)

    direct_error_rate = sum(upstream_attempts{outcome="retryable_error"})
                      / sum(upstream_attempts)

    error_rate_reduction = direct_error_rate - router_error_rate

Because ``upstream_attempts`` now carries the ``code`` label, a bare
selector returns one series per code — always aggregate it with ``sum()``
when you want the layer total. The error-code distribution itself is just
``sum by (code) (rate(upstream_attempts{code!="200"}[5m]))``.

The two layers have different denominators by design: one client request
can produce N upstream attempts (retry fan-out), and that asymmetry is
exactly the reason a health-aware router reduces the rate seen by the
client.
"""

from __future__ import annotations

from threading import Lock
from typing import Literal

OutcomeBucket = Literal["success", "retryable_error", "other_error"]

CLASSIFIER_FAIL_OPEN_REASONS = (
    "upstream_5xx",
    "upstream_4xx",
    "timeout",
    "ssl",
    "parse_error",
    "low_confidence",
    "other",
)

#: HTTP status codes the success criterion counts as router-rescuable
#: errors. 429 (rate limit), 500 (server error), 504 (gateway timeout).
RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 504})

#: Status codes emitted verbatim as the ``code`` label. Anything else seen
#: on the wire is clamped to its class (``"4xx"`` / ``"5xx"`` / …) so a
#: misbehaving upstream returning oddball codes cannot inflate label
#: cardinality. Covers the common success / client-error / server-error
#: codes an LLM endpoint actually returns.
KNOWN_STATUS_CODES: frozenset[int] = frozenset(
    {200, 400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504}
)

#: ``code`` label value for a non-HTTP failure (network error, pre-status
#: timeout) — the request never received a status line, so there is no code.
NO_STATUS_CODE: str = "none"

_lock = Lock()
_client_responses: dict[str, int] = {
    "success": 0,
    "retryable_error": 0,
    "other_error": 0,
}


def _seed_upstream() -> dict[tuple[str, str], int]:
    """Fresh upstream-attempt counter with canonical ``(outcome, code)`` series at 0.

    Seeding the codes a dashboard plots means their time series exist from
    process start, so a Grafana ``rate()`` renders a flat zero line rather
    than "no data" before the first matching attempt. Non-seeded codes (a
    one-off 403, say) are created lazily on first occurrence.
    """
    return {
        ("success", "200"): 0,
        ("retryable_error", "429"): 0,
        ("retryable_error", "500"): 0,
        ("retryable_error", "504"): 0,
        ("retryable_error", NO_STATUS_CODE): 0,
    }


#: Keyed by ``(outcome, code)``: ``code`` is the upstream HTTP status as a
#: string, ``"none"`` for a non-HTTP failure, or a clamped ``"Nxx"`` class.
_upstream_attempts: dict[tuple[str, str], int] = _seed_upstream()
_retry_recovered: int = 0
_classifier_fail_open: dict[str, int] = dict.fromkeys(CLASSIFIER_FAIL_OPEN_REASONS, 0)


def classify(status_code: int) -> OutcomeBucket:
    """Map an HTTP status code to its outcome bucket.

    2xx → ``success``. The codes listed in :data:`RETRYABLE_STATUSES`
    (429 / 500 / 504) → ``retryable_error``. Everything else (1xx, 3xx,
    most 4xx, other 5xx) → ``other_error``.
    """
    if 200 <= status_code < 300:
        return "success"
    if status_code in RETRYABLE_STATUSES:
        return "retryable_error"
    return "other_error"


def code_label(status_code: int | None) -> str:
    """Render the ``code`` label for one upstream attempt.

    ``None`` (non-HTTP failure) → :data:`NO_STATUS_CODE`. A code in
    :data:`KNOWN_STATUS_CODES` is emitted verbatim (``"429"``). Any other
    HTTP code is clamped to its class (``"4xx"``, ``"5xx"``, …), and an
    out-of-range value to ``"other"``, so label cardinality stays bounded
    no matter what an upstream returns.
    """
    if status_code is None:
        return NO_STATUS_CODE
    if status_code in KNOWN_STATUS_CODES:
        return str(status_code)
    if 100 <= status_code < 600:
        return f"{status_code // 100}xx"
    return "other"


def record_client_response(status_code: int) -> None:
    """Record one HTTP response sent to a client on an LLM-serving route."""
    bucket = classify(status_code)
    with _lock:
        _client_responses[bucket] += 1


def record_upstream_attempt(status_code: int | None) -> None:
    """Record one individual upstream attempt outcome.

    ``status_code=None`` is used for non-HTTP failures (network errors,
    pre-status timeouts) and is bucketed as ``retryable_error`` — those
    are exactly the kind of fault a health-aware router should be able
    to absorb by retrying on a different endpoint.
    """
    bucket: OutcomeBucket
    if status_code is None:
        bucket = "retryable_error"
    else:
        bucket = classify(status_code)
    key = (bucket, code_label(status_code))
    with _lock:
        _upstream_attempts[key] = _upstream_attempts.get(key, 0) + 1


def record_retry_recovered() -> None:
    """Record that a retry succeeded after at least one prior attempt failed.

    Direct evidence the router's steering logic kicked in: without
    retry, this request would have surfaced as a client-side error.
    """
    global _retry_recovered
    with _lock:
        _retry_recovered += 1


def classifier_fail_open_reason(exc: BaseException) -> str:
    """Map classifier failures onto bounded Prometheus reasons."""
    status = _status_code(exc)
    if status is not None:
        if 500 <= status < 600:
            return "upstream_5xx"
        if 400 <= status < 500:
            return "upstream_4xx"
    text = _exception_chain_text(exc)
    if "ssl" in text or "certificate" in text:
        return "ssl"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    return "other"


def record_classifier_fail_open(reason: str) -> None:
    """Record one classifier fallback-to-default event."""
    if reason not in _classifier_fail_open:
        reason = "other"
    with _lock:
        _classifier_fail_open[reason] += 1


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _exception_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.extend((type(current).__name__, str(current)))
        current = current.__cause__ or current.__context__
    return " ".join(parts).lower()


def render_lines() -> list[str]:
    """Render the current counter state as Prometheus exposition lines.

    Returns an ordered list of lines (no trailing newline). The
    ``/metrics`` endpoint concatenates this with the accumulator output.
    """
    with _lock:
        client = dict(_client_responses)
        upstream = dict(_upstream_attempts)
        recovered = _retry_recovered
        fail_open = dict(_classifier_fail_open)

    lines: list[str] = []
    lines.append(
        "# HELP switchyard_client_responses_total "
        "HTTP responses returned to clients on LLM-serving routes, "
        "bucketed by outcome (success / retryable_error / other_error)."
    )
    lines.append("# TYPE switchyard_client_responses_total counter")
    for outcome in ("success", "retryable_error", "other_error"):
        lines.append(
            f'switchyard_client_responses_total{{outcome="{outcome}"}} '
            f"{client[outcome]}"
        )

    lines.append(
        "# HELP switchyard_upstream_attempts_total "
        "Individual upstream call attempts, bucketed by outcome and labelled "
        "with the upstream HTTP status code (code=\"none\" for non-HTTP "
        "failures). One client request can produce multiple attempts via retry."
    )
    lines.append("# TYPE switchyard_upstream_attempts_total counter")
    # Sorted for deterministic exposition order across scrapes.
    for outcome, code in sorted(upstream):
        lines.append(
            f'switchyard_upstream_attempts_total{{outcome="{outcome}",code="{code}"}} '
            f"{upstream[(outcome, code)]}"
        )

    lines.append(
        "# HELP switchyard_router_retry_recovered_total "
        "Requests whose first upstream attempt failed but a subsequent "
        "attempt succeeded — direct evidence steering logic rescued the request."
    )
    lines.append("# TYPE switchyard_router_retry_recovered_total counter")
    lines.append(f"switchyard_router_retry_recovered_total {recovered}")

    lines.append(
        "# HELP switchyard_classifier_fail_open_triggered_total "
        "Classifier fallbacks to the default tier after unusable classifier verdicts."
    )
    lines.append("# TYPE switchyard_classifier_fail_open_triggered_total counter")
    for reason in CLASSIFIER_FAIL_OPEN_REASONS:
        lines.append(
            f'switchyard_classifier_fail_open_triggered_total{{reason="{reason}"}} '
            f"{fail_open[reason]}"
        )

    return lines


def _reset_for_tests() -> None:
    """Zero every counter — tests only."""
    global _retry_recovered, _upstream_attempts
    with _lock:
        for key in _client_responses:
            _client_responses[key] = 0
        # Rebuild rather than zero in place: drops any lazily-added codes so
        # each test starts from the same canonical seed.
        _upstream_attempts = _seed_upstream()
        _retry_recovered = 0
        for reason in _classifier_fail_open:
            _classifier_fail_open[reason] = 0


__all__ = [
    "CLASSIFIER_FAIL_OPEN_REASONS",
    "KNOWN_STATUS_CODES",
    "NO_STATUS_CODE",
    "RETRYABLE_STATUSES",
    "OutcomeBucket",
    "classifier_fail_open_reason",
    "classify",
    "code_label",
    "record_classifier_fail_open",
    "record_client_response",
    "record_retry_recovered",
    "record_upstream_attempt",
    "render_lines",
]
