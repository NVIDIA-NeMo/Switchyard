# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``LLMBackend`` that routes across many endpoints by Latency Service verdicts.

This is the usage case for Inference Hub deployments where a central
Latency Service owns heartbeat probing and statistical profiling.  The
backend holds a pool of ``OpenAILLMClient`` instances keyed by model ID,
reads health verdicts from a locally-cached map maintained by a
:class:`HealthPoller` daemon thread, and picks a healthy endpoint on
every request.

Chain integration::

    request-side work → LatencyServiceLLMBackend → response-side work → TranslationEngine

Request-format translation is delegated to
:class:`switchyard_rust.translation.TranslationEngine` after an endpoint is
selected, so Chat endpoints keep receiving Chat Completions while
Responses-mode endpoints receive the OpenAI Responses API natively.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass

from openai import APIStatusError, AsyncStream

from switchyard.lib import metrics, spans
from switchyard.lib.backends.health_poller import (
    EndpointHealth,
    EndpointHealthStatus,
    HealthPoller,
)
from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.chat_response.openai_responses import ResponsesApiStream
from switchyard.lib.config.latency_service_backend_config import (
    LatencyServiceBackendConfig,
)
from switchyard.lib.endpoints.upstream_error_log import log_upstream_attempt_failure
from switchyard.lib.llm_client import OpenAILLMClient
from switchyard.lib.proxy_context import (
    CTX_CALLER_API_KEY,
    CTX_UPSTREAM_ATTEMPTS_RECORDED,
    CTX_UPSTREAM_HTTP_BODY,
    CTX_UPSTREAM_HTTP_STATUS,
    ProxyContext,
)
from switchyard.lib.roles import LLMBackend
from switchyard.lib.session_affinity import SessionAffinity
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse, request_type_matches
from switchyard_rust.translation import TranslationEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RouteDecision:
    """One endpoint-selection outcome, carried to the ``route_decision`` span.

    ``candidates`` is the set the picker chose among (the winning health tier),
    and ``was_fastest`` is whether ``selected`` had the lowest known latency in
    that set.  ``affinity`` is ``True`` when ``selected`` came from a session
    pin (reused endpoint) rather than the latency-aware picker.
    """

    selected: str
    candidates: tuple[str, ...]
    was_fastest: bool
    affinity: bool = False


class LatencyServiceLLMBackend(LLMBackend):
    """Routes to healthy endpoints based on Latency Service health verdicts.

    On construction, builds a pool of ``OpenAILLMClient`` instances
    (one per configured endpoint, keyed by ``model``) and starts a
    :class:`HealthPoller` daemon thread that refreshes the in-memory
    health cache every ``poll_interval_s`` seconds.

    On each ``call()``:

    1. Pick an endpoint from the health cache — ``HEALTHY`` preferred,
       then ``UNKNOWN``, then ``DEGRADED``.  Within the chosen tier,
       selection is weighted by inverse ``last_latency_ms`` when every
       candidate has a known sample; otherwise it falls back to uniform
       random.
    2. Translate the request to that endpoint's configured OpenAI API
       surface and override the body's ``model`` with the selected endpoint ID.
    3. Call ``OpenAILLMClient`` on the configured API surface. On a transient error
       (429 / 408 / 5xx / network), retry with a different endpoint
       (dedup prevents re-selecting the same one within a single
       request).  On a 4xx client error the request is rejected
       identically by every replica, so the loop fails fast and passes
       the upstream status through.
    4. Wrap the response into a Rust-owned ``ChatResponse``.
    """

    def __init__(
        self,
        config: LatencyServiceBackendConfig,
    ) -> None:
        if not config.endpoints:
            raise ValueError("At least one endpoint must be configured")

        self._config = config
        self._translation = TranslationEngine()
        self._clients: dict[str, OpenAILLMClient] = {}
        self._upstream_models: dict[str, str] = {}
        self._request_types: dict[str, ChatRequestType] = {}
        self._health_cache: dict[str, EndpointHealth] = {}
        self._cache_lock = threading.Lock()

        # Session affinity: pins a conversation to the endpoint that last served
        # it so its upstream prompt/KV cache stays warm. Shared coordinator with
        # the classifier router; the latency-specific reuse policy (health-gated,
        # failover-aware) lives in ``_select_endpoint_decision``.
        self._affinity = SessionAffinity(
            enabled=config.session_affinity,
            max_sessions=config.affinity_max_sessions,
        )
        # Cumulative warm-reuse counters, published on /metrics when affinity is
        # enabled. A "hit" is a turn served by an existing pin; a "miss" is a
        # first/unpinnable turn routed by the latency-aware picker. Counted once
        # per request (first attempt only) so failover retries don't inflate
        # them. Incremented and read only on the event loop, so no lock needed.
        self._affinity_hits = 0
        self._affinity_misses = 0

        for ep_cfg in config.endpoints:
            model_id = ep_cfg.model
            if not model_id:
                raise ValueError(
                    "Every endpoint must have a 'model' field — "
                    "this is the key used by the Latency Service"
                )
            if model_id in self._clients:
                raise ValueError(f"Duplicate model ID: {model_id}")

            self._clients[model_id] = OpenAILLMClient(
                api_key=ep_cfg.api_key,
                base_url=ep_cfg.base_url,
                timeout=ep_cfg.timeout,
                # The ``call`` retry loop already retries on a *different*
                # endpoint, which is what a health-aware router wants. Letting
                # the SDK also retry (default 2) on the *same* endpoint stacks
                # multiplicatively (up to (1+max_retries) x 3 attempts) and adds
                # exponential-backoff sleeps that hold the request — and its
                # buffered body — alive longer, amplifying connection-pool
                # pressure during an upstream incident. Disable it here.
                max_retries=0,
            )
            self._upstream_models[model_id] = ep_cfg.upstream_model or model_id
            self._request_types[model_id] = _latency_endpoint_request_type(
                ep_cfg.request_type
            )
            self._health_cache[model_id] = EndpointHealth(
                status=EndpointHealthStatus.UNKNOWN,
            )
            log.info(
                "LatencyServiceLLMBackend endpoint: model=%s upstream_model=%s "
                "request_type=%s base_url=%s",
                model_id,
                self._upstream_models[model_id],
                ep_cfg.request_type,
                ep_cfg.base_url,
            )

        self._poller = HealthPoller(
            latency_service_url=config.latency_service_url,
            model_ids=list(self._clients.keys()),
            health_cache=self._health_cache,
            cache_lock=self._cache_lock,
            poll_interval_s=config.poll_interval_s,
            poll_timeout_s=config.poll_timeout_s,
        )
        # Publish LS verdicts + poll/affinity health on /metrics via OTel
        # observable gauges/counters. Registering here (and clearing in
        # ``shutdown``) ties the source's lifetime to the backend's, so a
        # re-built chain doesn't leak a callable over a torn-down cache.
        metrics.register_latency_service_source(self._metrics_snapshot)
        self._poller.start()

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        """OpenAI request types accepted by the configured endpoints."""
        ordered = [ChatRequestType.OPENAI_CHAT, ChatRequestType.OPENAI_RESPONSES]
        configured = set(self._request_types.values())
        return [request_type for request_type in ordered if request_type in configured]

    # -- Endpoint selection (reads from cache, never blocks on network) -----

    def _select_endpoint(self) -> str:
        """Pick a model ID — see :meth:`_select_endpoint_decision`."""
        return self._select_endpoint_decision().selected

    def _select_endpoint_decision(
        self,
        pinned_endpoint: str | None = None,
        tried: frozenset[str] = frozenset(),
    ) -> _RouteDecision:
        """Pick a model ID and report the candidate set it chose among.

        When ``pinned_endpoint`` is set (this conversation's session-affinity
        pin) and still usable — HEALTHY/UNKNOWN and not already failed this
        request — it is reused directly, bypassing the latency-aware pick so the
        conversation sticks to one endpoint and its upstream cache stays warm.

        Otherwise tier priority is HEALTHY > UNKNOWN > DEGRADED.  Within the
        chosen tier, selection is **inverse-latency weighted** when every
        candidate has a known ``last_latency_ms`` — endpoints that the Latency
        Service most recently saw as faster receive proportionally more
        traffic.  If any candidate's latency is unknown, the picker falls back
        to uniform random for that tier; this keeps behaviour predictable while
        the poller is warming up or when the upstream reports nulls.

        Returns the selected endpoint plus the candidates considered and
        whether the pick was the lowest-latency one — the signals the
        ``switchyard.route_decision`` span reports.
        """
        with self._cache_lock:
            snapshot = dict(self._health_cache)

        # -- Affinity fast-path: reuse the pinned endpoint while it stays
        # healthy and hasn't already failed this request. Skips latency
        # weighting entirely — the whole point is to not chase a marginally
        # faster endpoint at the cost of cache locality.
        if (
            pinned_endpoint is not None
            and pinned_endpoint not in tried
            and _affinity_usable(snapshot.get(pinned_endpoint))
        ):
            return _RouteDecision(
                selected=pinned_endpoint,
                candidates=(pinned_endpoint,),
                was_fastest=False,
                affinity=True,
            )

        by_health: dict[EndpointHealthStatus, list[str]] = {
            h: [] for h in EndpointHealthStatus
        }
        for mid, hp in snapshot.items():
            by_health[hp.status].append(mid)

        for tier in (
            EndpointHealthStatus.HEALTHY,
            EndpointHealthStatus.UNKNOWN,
            EndpointHealthStatus.DEGRADED,
        ):
            if by_health[tier]:
                candidates = by_health[tier]
                selected = self._pick_by_latency(candidates, snapshot)
                return _RouteDecision(
                    selected=selected,
                    candidates=tuple(candidates),
                    was_fastest=self._is_fastest(selected, candidates, snapshot),
                )

        candidates = list(self._clients.keys())
        selected = random.choice(candidates)
        return _RouteDecision(
            selected=selected,
            candidates=tuple(candidates),
            was_fastest=self._is_fastest(selected, candidates, snapshot),
        )

    @staticmethod
    def _is_fastest(
        selected: str,
        candidates: list[str],
        snapshot: dict[str, EndpointHealth],
    ) -> bool:
        """Whether *selected* had the lowest known ``last_latency_ms``.

        ``False`` when no candidate has a usable latency sample (selection was
        uniform, so "fastest" is undefined). Over many requests the fraction of
        ``True`` shows how often inverse-latency weighting picked the front-runner.
        """
        known: dict[str, float] = {}
        for mid in candidates:
            latency = snapshot[mid].last_latency_ms
            if latency is not None and latency > 0:
                known[mid] = latency
        if not known or selected not in known:
            return False
        return known[selected] == min(known.values())

    def _poll_age_ms(self) -> float | None:
        """Milliseconds since the last successful Latency Service poll, or ``None``."""
        age_s = self._poller.seconds_since_last_success
        return age_s * 1000.0 if age_s is not None else None

    @staticmethod
    def _pick_by_latency(
        candidates: list[str],
        snapshot: dict[str, EndpointHealth],
    ) -> str:
        """Inverse-latency weighted random pick across ``candidates``.

        Returns a uniform random pick when only one candidate exists, when
        any candidate's ``last_latency_ms`` is unknown, or when any sample
        is non-positive (treated as bogus — the Latency Service should
        never report 0 or negative, but be defensive).
        """
        if len(candidates) == 1:
            return candidates[0]
        latencies = [snapshot[c].last_latency_ms for c in candidates]
        if any(lat is None or lat <= 0 for lat in latencies):
            return random.choice(candidates)
        weights = [1.0 / lat for lat in latencies]  # type: ignore[operator]
        return random.choices(candidates, weights=weights, k=1)[0]

    # -- Request processing (hot path — no Latency Service call) ------------

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        # This backend records its own per-attempt upstream counters via
        # ``metrics.record_upstream_attempt`` below (one per failover attempt).
        # Claim attempt accounting for this request so the endpoint-layer
        # fallback in ``dispatch`` / ``handle_chain_exception`` does not
        # double-count the retry fan-out.
        ctx.metadata[CTX_UPSTREAM_ATTEMPTS_RECORDED] = True
        # Captured before the per-attempt ``body["model"]`` override so the span
        # records the model the client asked for, not the selected endpoint.
        incoming_model = request.model
        # Resolve the session-affinity pin once (keyed on the stable conversation
        # prefix). ``None`` when affinity is disabled or the conversation isn't
        # pinned yet, so every attempt routes purely by health + latency.
        pinned_endpoint = self._affinity.pinned(ctx, request)

        last_exc: Exception | None = None
        tried: set[str] = set()
        api_key_override = self._api_key_override_for_policy(
            ctx.metadata.get(CTX_CALLER_API_KEY)
        )

        for attempt in range(1 + self._config.max_retries):
            # -- Route decision span: which endpoint, out of which candidates --
            with spans.stage_span("switchyard.route_decision") as route_span:
                decision = self._select_endpoint_decision(pinned_endpoint, frozenset(tried))
                model_id = decision.selected
                candidates = decision.candidates
                was_fastest = decision.was_fastest
                if model_id in tried and len(tried) < len(self._clients):
                    remaining = [m for m in self._clients if m not in tried]
                    model_id = random.choice(remaining)
                    # A forced dedup pick, not the latency-weighted choice.
                    candidates = tuple(remaining)
                    was_fastest = False
                tried.add(model_id)
                # Count the warm-reuse outcome once per request (the first
                # decision); later attempts are failover, not an affinity signal.
                if self._affinity.enabled and attempt == 0:
                    if decision.affinity:
                        self._affinity_hits += 1
                    else:
                        self._affinity_misses += 1
                if route_span is not None:
                    spans._set_attrs(route_span, {
                        "switchyard.model": incoming_model,
                        "switchyard.candidate_endpoints": ",".join(candidates),
                        "switchyard.selected_endpoint": model_id,
                        "switchyard.was_fastest_selected": was_fastest,
                        "switchyard.affinity_hit": decision.affinity,
                        "switchyard.latency_service_poll_age_ms": self._poll_age_ms(),
                    })

            upstream_model = self._upstream_models[model_id]
            target_request_type = self._request_types[model_id]
            body = self._body_for_endpoint_request_type(ctx, request, target_request_type)
            body["model"] = upstream_model
            log.debug(
                "LatencyServiceLLMBackend: attempt=%d model=%s upstream=%s "
                "request_type=%s stream=%s",
                attempt + 1,
                model_id,
                upstream_model,
                target_request_type.value,
                body.get("stream"),
            )

            # -- Upstream attempt span: outcome of this one upstream call -----
            with spans.stage_span("switchyard.upstream_attempt") as attempt_span:
                if attempt_span is not None:
                    spans._set_attrs(attempt_span, {
                        "switchyard.model": incoming_model,
                        "switchyard.selected_endpoint": model_id,
                        "switchyard.retry_count": attempt,
                    })
                started_at = time.monotonic()
                try:
                    result = await self._call_endpoint(
                        model_id,
                        target_request_type,
                        api_key=api_key_override,
                        body=body,
                    )
                except APIStatusError as exc:
                    if attempt_span is not None:
                        spans._set_attrs(attempt_span, {
                            "switchyard.outcome": metrics.classify(exc.status_code),
                            "switchyard.upstream_status_code": exc.status_code,
                            "switchyard.error_code": metrics.code_label(exc.status_code),
                        })
                    _record_upstream_attempt(exc.status_code)
                    # Per-event structured log (Loki) — the timestamped complement
                    # to the aggregate outcome counter recorded above.
                    log_upstream_attempt_failure(
                        model=model_id,
                        attempt=attempt + 1,
                        status_code=exc.status_code,
                        error=exc,
                    )
                    last_exc = exc
                    # A client error (4xx other than 408/429) is deterministic —
                    # replicas reject the same payload identically — so fail fast
                    # instead of burning attempts; the post-loop passthrough
                    # surfaces the upstream status to the client.
                    if not _is_retryable_status(exc.status_code):
                        break
                    continue
                except Exception as exc:
                    if attempt_span is not None:
                        spans._set_attrs(attempt_span, {
                            "switchyard.outcome": "retryable_error",
                            "switchyard.error_code": metrics.NO_STATUS_CODE,
                        })
                    # Non-HTTP failure (network, pre-status timeout, SDK error) —
                    # treated as a retryable_error in the outcome ratio: those
                    # are exactly the faults a health-aware router should absorb
                    # by selecting a different endpoint on the next attempt.
                    _record_upstream_attempt(None)
                    # status_code=None → logged as code="none" (no HTTP status).
                    log_upstream_attempt_failure(
                        model=model_id,
                        attempt=attempt + 1,
                        status_code=None,
                        error=exc,
                    )
                    last_exc = exc
                    continue

                backend_latency_ms = (time.monotonic() - started_at) * 1000.0
                if attempt_span is not None:
                    spans._set_attrs(attempt_span, {
                        "switchyard.outcome": "success",
                        "switchyard.upstream_status_code": 200,
                    })
                _record_upstream_attempt(200)
                # A successful attempt after at least one failure is direct
                # evidence the steering logic rescued this request. Counted
                # once per client request, not per recovered retry.
                if attempt > 0:
                    metrics.record_retry_recovered()

                # ``ctx.selected_model`` is the cross-language hook downstream
                # token/cost attribution reads to bucket usage per endpoint.
                # Without it, every call buckets to ``model="<unknown>"``.
                ctx.selected_model = model_id
                # ``backend_call_latency_ms`` lets the chain executor record the
                # upstream-only latency for this Python backend (which can't be
                # measured by the executor's own backend-stage timer because the
                # retry fan-out happens inside this method).
                ctx.backend_call_latency_ms = backend_latency_ms

                # Pin this conversation to the endpoint that served it so later
                # turns reuse it (warm cache). Re-pinning on every success also
                # follows a recovery: if the previous pin degraded and we
                # re-routed, the endpoint that worked becomes the new pin.
                # (No-op when affinity is disabled.)
                self._affinity.pin(ctx, request, model_id)

                return _chat_response_for_request_type(target_request_type, result)

        # All attempts failed. If the last failure was an upstream HTTP
        # error (e.g. 401 from a bad API key), record the status code and
        # body on ctx so the endpoint can passthrough the upstream status
        # rather than masking it as a 500. The exception itself still
        # propagates — the chain is errored — but the endpoint reads ctx
        # to decide the response code.
        if isinstance(last_exc, APIStatusError):
            ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = last_exc.status_code
            upstream_body = _extract_upstream_body(last_exc)
            if upstream_body is not None:
                ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = upstream_body

        raise last_exc  # type: ignore[misc]

    def _body_for_endpoint_request_type(
        self,
        ctx: ProxyContext,
        request: ChatRequest,
        target_request_type: ChatRequestType,
    ) -> dict[str, object]:
        try:
            normalized = self._translation.request_to_any_of(
                request, [target_request_type],
            )
        except ValueError as exc:
            # Transparent-router contract: a payload the upstream provider would
            # reject (e.g. an unsupported message role like "api") must surface
            # as a provider-compatible 400, not a generic 500. The Rust
            # translation layer raises a ``ValueError`` whose message is
            # prefixed with the stable error kind; an invalid-value rejection is
            # recorded as an upstream 400 so the endpoint passes it through.
            _stash_invalid_request_error(ctx, exc)
            raise
        if not request_type_matches(normalized, target_request_type):
            raise TypeError(
                "LatencyServiceLLMBackend expected request type "
                f"{target_request_type.value} after translation"
            )
        return dict(normalized.body)

    async def _call_endpoint(
        self,
        model_id: str,
        target_request_type: ChatRequestType,
        *,
        api_key: str | None,
        body: dict[str, object],
    ) -> object:
        if target_request_type == ChatRequestType.OPENAI_RESPONSES:
            return await self._clients[model_id].aresponses(
                api_key=api_key,
                **body,
            )
        return await self._clients[model_id].acompletion(
            api_key=api_key,
            **body,
        )

    def _api_key_override_for_policy(self, caller_api_key: object) -> str | None:
        if self._config.credential_policy != "caller_override":
            return None
        if isinstance(caller_api_key, str) and caller_api_key.strip():
            return caller_api_key
        return None

    # -- Lifecycle ----------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the background :class:`HealthPoller` daemon.

        Picked up automatically by ``NemoSwitchyardServer``'s component
        teardown hook (see ``server.py``'s lifespan context manager).
        Safe to call multiple times.
        """
        metrics.register_latency_service_source(None)
        self._poller.stop()

    def is_ready(self) -> bool:
        """True once the background poller has completed at least one successful poll."""
        return self._poller.has_polled

    # -- Metrics source -----------------------------------------------------

    def _metrics_snapshot(self) -> dict[str, object]:
        """Return a snapshot of poll/health/affinity state for the OTel gauges.

        Read by the observable-gauge/counter callbacks in
        :mod:`switchyard.lib.metrics` on each scrape. Snapshotted under
        ``self._cache_lock`` so a poll concurrent with a scrape can't produce a
        mixed view across endpoints. Affinity counters are included only when
        session affinity is enabled, keeping the surface clean for the common
        per-turn-routing case.
        """
        with self._cache_lock:
            health_snapshot = dict(self._health_cache)

        snapshot: dict[str, object] = {
            "endpoint_status": {
                model_id: health.status.value
                for model_id, health in health_snapshot.items()
            },
            "endpoint_last_latency_ms": {
                model_id: health.last_latency_ms
                for model_id, health in health_snapshot.items()
            },
            # ``poll_ok`` reflects the latest poll attempt; combined with
            # ``poll_age_seconds`` scrapers can tell "never polled" (no age) from
            # "polled, latest attempt failed" (age present, ok=0).
            "poll_ok": self._poller.last_poll_ok,
            "polls": self._poller.poll_successes,
            "poll_failures": self._poller.poll_failures,
        }
        age = self._poller.seconds_since_last_success
        if age is not None:
            snapshot["poll_age_seconds"] = age
        if self._config.session_affinity:
            snapshot["affinity_hits"] = self._affinity_hits
            snapshot["affinity_misses"] = self._affinity_misses
        return snapshot


def _latency_endpoint_request_type(value: str) -> ChatRequestType:
    if value == "openai_responses":
        return ChatRequestType.OPENAI_RESPONSES
    return ChatRequestType.OPENAI_CHAT


def _chat_response_for_request_type(
    request_type: ChatRequestType,
    result: object,
) -> ChatResponse:
    if request_type == ChatRequestType.OPENAI_RESPONSES:
        if isinstance(result, AsyncStream):
            return ChatResponse.openai_responses_stream(ResponsesApiStream(result))
        return ChatResponse.openai_responses_completion(result)
    if isinstance(result, AsyncStream):
        return ChatResponse.openai_stream(ResponseStream(result))
    return ChatResponse.openai_completion(result)


def _record_upstream_attempt(status_code: int | None) -> None:
    """Record one upstream attempt outcome via OTel.

    ``status_code=None`` (non-HTTP failure) is bucketed as ``retryable_error``
    — exactly the fault a health-aware router should absorb by failing over.
    """
    outcome = "retryable_error" if status_code is None else metrics.classify(status_code)
    metrics.record_upstream_attempt(outcome=outcome, code=metrics.code_label(status_code))


def _is_retryable_status(status_code: int) -> bool:
    """Whether an upstream status warrants failing over to a different endpoint.

    Retries 5xx + 408 + 429 (transient/capacity/server-side); other 4xx are the
    client's payload, which every replica rejects identically, so fail fast.
    Deliberately broader than ``metrics.RETRYABLE_STATUSES`` (metrics bucketing)
    — e.g. 502/503 must fail over but aren't metrics-retryable.
    """
    return status_code >= 500 or status_code in (408, 429)


def _affinity_usable(health: EndpointHealth | None) -> bool:
    """Whether a pinned endpoint may still serve its session.

    The pin holds while the endpoint is HEALTHY or UNKNOWN — the same tiers the
    latency-aware picker trusts. A DEGRADED verdict (or an endpoint that
    vanished from the health cache) breaks the pin so the next turn re-routes:
    locality must yield to health, or sticky routing would funnel a session
    into a failing endpoint.
    """
    return health is not None and health.status in (
        EndpointHealthStatus.HEALTHY,
        EndpointHealthStatus.UNKNOWN,
    )


# Prefix of the stringified translation error for a client-invalid payload.
# ``TranslationError::kind()`` is the stable FFI signal for the error variant
# (``crates/switchyard-translation/src/error.rs``); ``py_translation_error``
# formats translation errors as ``"{kind}: {message}"``, so an invalid-value
# rejection (e.g. an unsupported message role) is recognizable by this prefix.
_TRANSLATION_INVALID_VALUE_PREFIX = "InvalidValue:"


def _stash_invalid_request_error(ctx: ProxyContext, error: Exception) -> None:
    """Record a translation invalid-value error as an upstream HTTP 400 on ``ctx``.

    Mirrors the upstream-status passthrough used for provider HTTP errors: the
    Compatibility execution flattens the raised exception to a generic error
    string, but ``ctx.metadata`` survives the error path, so the endpoint's
    ``upstream_response_from_ctx`` can return a provider-compatible 400. This is
    a no-op for any other error (it keeps propagating and FastAPI maps it to a
    500 as before).
    """
    message = str(error)
    if not message.startswith(_TRANSLATION_INVALID_VALUE_PREFIX):
        return
    detail = message[len(_TRANSLATION_INVALID_VALUE_PREFIX) :].strip()
    ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 400
    ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = {
        "error": {
            "message": detail or "invalid request payload",
            "type": "invalid_request_error",
            "code": "invalid_value",
        }
    }


def _extract_upstream_body(error: APIStatusError) -> object | None:
    """Best-effort extraction of the upstream response body for passthrough.

    Prefers the structured ``body`` attribute (set by the OpenAI SDK for
    JSON responses). Falls back to ``response.text`` for responses the
    SDK couldn't decode. Returns ``None`` when nothing usable is present
    — the endpoint then synthesizes a generic error envelope.
    """
    body: object | None = getattr(error, "body", None)
    if body is not None:
        return body
    response = getattr(error, "response", None)
    if response is None:
        return None
    text: object | None = getattr(response, "text", None)
    return text if text else None
