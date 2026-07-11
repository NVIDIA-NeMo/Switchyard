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

import json
import logging
import random
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from openai import APIStatusError, AsyncStream

from switchyard.lib.affinity_pin_store import AffinityPinStore
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
from switchyard.lib.endpoints import outcome_metrics, prometheus_emitter
from switchyard.lib.endpoints.upstream_error_log import log_upstream_attempt_failure
from switchyard.lib.llm_client import OpenAILLMClient
from switchyard.lib.prometheus_exposition import format_number, render_labels
from switchyard.lib.proxy_context import (
    CTX_CALLER_API_KEY,
    CTX_ERROR_SOURCE,
    CTX_ROUTE_SELECTION,
    CTX_UPSTREAM_ATTEMPTS_RECORDED,
    CTX_UPSTREAM_HTTP_BODY,
    CTX_UPSTREAM_HTTP_STATUS,
    CTX_UPSTREAM_MODEL,
    ERROR_SOURCE_PROVIDER,
    ERROR_SOURCE_SWITCHYARD,
    ProxyContext,
)
from switchyard.lib.roles import LLMBackend
from switchyard.lib.session_affinity import SessionAffinity
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard.lib.tracing import routing_span, set_tags
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse, request_type_matches
from switchyard_rust.translation import TranslationEngine

log = logging.getLogger(__name__)

#: Request header read by a LiteLLM front proxy; its JSON value is copied into
#: the provider spend-log DB row, tying that row back to the Switchyard route
#: that selected it (see :data:`CTX_ROUTE_SELECTION` for the payload contract).
SPEND_LOGS_METADATA_HEADER = "x-litellm-spend-logs-metadata"

#: ``router_strategy`` value identifying this backend's selection algorithm.
ROUTER_STRATEGY_LATENCY = "latency"


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
        *,
        stats_accumulator: StatsAccumulator | None = None,
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
        # When provided, the backend records success/error/latency directly
        # into the accumulator on each attempt. This mirrors what the Rust
        # ``StatsLlmBackend`` does for native backends; the Python-only
        # latency-service backend can't be wrapped by it, so we record
        # in-place to keep ``/metrics`` populated.
        self._stats = stats_accumulator

        # Session affinity: pins a conversation to the endpoint that last served
        # it so its upstream prompt/KV cache stays warm. Shared coordinator with
        # the classifier router; the latency-specific reuse policy (health-gated,
        # failover-aware) lives in ``_select_endpoint_decision``. An optional
        # shared L2 (Redis) lets pins outlive this worker and be read by peers,
        # so a conversation stays pinned across replicas and pod churn.
        self._affinity = SessionAffinity(
            enabled=config.session_affinity,
            max_sessions=config.affinity_max_sessions,
            l2=_build_affinity_l2(config),
        )
        # Cumulative warm-reuse counters, published on /metrics when affinity is
        # enabled. A "hit" is a turn served by an existing pin; a "miss" is a
        # first/unpinnable turn routed by the latency-aware picker. Counted once
        # per request (first attempt only) so failover retries don't inflate
        # them. Incremented and read only on the event loop, so no lock needed.
        self._affinity_hits = 0
        self._affinity_misses = 0

        # Per-model upstream-attempt counts for the latency-service-scoped
        # /metrics breakdown, keyed by (requested model, selected upstream model,
        # outcome, HTTP code). Kept separate from the layer-aggregate
        # ``switchyard_upstream_attempts_total`` so that counter stays model-free.
        # Cardinality is bounded: ``upstream_model`` comes from config, and
        # ``requested_model`` is normalized to a config-derived id (route id or
        # endpoint id) or the ``"other"`` sentinel in ``_record_model_attempt``
        # so a caller-supplied model string can't grow the series set. Mutated
        # and read only on the event loop (like the affinity counters), so no
        # lock.
        self._upstream_attempts_by_model: dict[tuple[str, str, str, str], int] = {}

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

        # Bounded ``requested_model`` label set: the configured endpoint ids
        # plus the client-facing route id when the deployment supplied one
        # (IH clients request the route key, e.g. ``nvidia/switchyard/gpt-5.4``,
        # never an endpoint id). Both sources are config-derived, so metric
        # cardinality stays bounded.
        self._requested_model_ids = frozenset(self._clients) | (
            frozenset((config.route_model,)) if config.route_model else frozenset()
        )

        self._poller = HealthPoller(
            latency_service_url=config.latency_service_url,
            model_ids=list(self._clients.keys()),
            health_cache=self._health_cache,
            cache_lock=self._cache_lock,
            poll_interval_s=config.poll_interval_s,
            poll_timeout_s=config.poll_timeout_s,
        )
        # Publish LS verdicts + poll health on /metrics. Registering here
        # (and unregistering in ``shutdown``) ties emitter lifetime to the
        # backend's lifetime, so a re-built chain doesn't leak a closure
        # over a torn-down cache.
        prometheus_emitter.register(self._render_prometheus_lines)
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
        # This backend records its own per-attempt ``outcome_metrics`` counters
        # below (one per failover attempt). Claim attempt accounting for this
        # request so the endpoint-layer fallback in ``dispatch`` /
        # ``handle_chain_exception`` does not double-count the retry fan-out.
        ctx.metadata[CTX_UPSTREAM_ATTEMPTS_RECORDED] = True
        api_key_override = self._api_key_override_for_policy(
            ctx.metadata.get(CTX_CALLER_API_KEY)
        )
        # ``caller_required`` never falls back to the configured endpoint key:
        # reject before any upstream call so the service credential can't
        # authenticate caller inference — and before the affinity lookup, so a
        # doomed request never pays a pin resolution (possibly a shared-store
        # round-trip). The rejection deliberately stays invisible to
        # ``switchyard_upstream_attempts_total`` (the accounting flag above is
        # already claimed): no upstream attempt happened, and counting a
        # synthetic 401 would skew the direct-to-endpoint baseline error rate.
        # ``api_key_override`` is ``None`` here only when no usable caller key
        # was supplied.
        if self._config.credential_policy == "caller_required" and api_key_override is None:
            _reject_missing_caller_api_key(ctx)

        # Captured before the per-attempt ``body["model"]`` override so the span
        # records the model the client asked for, not the selected endpoint.
        incoming_model = request.model
        # One correlation id per client request, shared by every failover
        # attempt's spend-logs header and by the response headers — the join
        # key between a front proxy's spend-log row and the provider rows.
        correlation_id = str(uuid.uuid4())
        # Resolve the session-affinity pin once (keyed on the stable conversation
        # prefix). ``None`` when affinity is disabled or the conversation isn't
        # pinned yet, so every attempt routes purely by health + latency.
        pinned_endpoint = await self._affinity.pinned(ctx, request)

        last_exc: Exception | None = None
        # Upstream model of the last failed attempt, surfaced on the error
        # envelope/span/log so operators can tell *which* endpoint the
        # provider-originated failure came from.
        last_upstream_model: str | None = None
        tried: set[str] = set()

        for attempt in range(1 + self._config.max_retries):
            # -- Route decision span: which endpoint, out of which candidates --
            with routing_span("switchyard.route_decision") as route_span:
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
                set_tags(route_span, {
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
            # Per-attempt route-selection record: each upstream call is stamped
            # with the endpoint it actually hits, so a failover retry's
            # spend-log row doesn't inherit the failed attempt's selection.
            route_selection = self._route_selection(
                incoming_model=incoming_model,
                model_id=model_id,
                upstream_model=upstream_model,
                correlation_id=correlation_id,
            )
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
            with routing_span("switchyard.upstream_attempt") as attempt_span:
                set_tags(attempt_span, {
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
                        extra_headers={
                            SPEND_LOGS_METADATA_HEADER: json.dumps(route_selection),
                        },
                    )
                except APIStatusError as exc:
                    set_tags(attempt_span, {
                        "switchyard.outcome": outcome_metrics.classify(exc.status_code),
                        "switchyard.upstream_status_code": exc.status_code,
                        "switchyard.error_code": outcome_metrics.code_label(exc.status_code),
                        "switchyard.error_source": ERROR_SOURCE_PROVIDER,
                        "switchyard.upstream_model": upstream_model,
                    })
                    if self._stats is not None:
                        await self._stats.record_error(model_id)
                    outcome_metrics.record_upstream_attempt(exc.status_code)
                    self._record_model_attempt(
                        incoming_model,
                        upstream_model,
                        outcome_metrics.classify(exc.status_code),
                        outcome_metrics.code_label(exc.status_code),
                    )
                    # Per-event structured log (Loki) — the timestamped complement
                    # to the aggregate outcome counter recorded above.
                    log_upstream_attempt_failure(
                        model=model_id,
                        attempt=attempt + 1,
                        status_code=exc.status_code,
                        error=exc,
                        upstream_model=upstream_model,
                    )
                    last_exc = exc
                    last_upstream_model = upstream_model
                    # A client error (4xx other than 408/429) is deterministic —
                    # replicas reject the same payload identically — so fail fast
                    # instead of burning attempts; the post-loop passthrough
                    # surfaces the upstream status to the client.
                    if not _is_retryable_status(exc.status_code):
                        break
                    continue
                except Exception as exc:
                    set_tags(attempt_span, {
                        "switchyard.outcome": "retryable_error",
                        "switchyard.error_code": outcome_metrics.NO_STATUS_CODE,
                        "switchyard.error_source": ERROR_SOURCE_PROVIDER,
                        "switchyard.upstream_model": upstream_model,
                    })
                    if self._stats is not None:
                        await self._stats.record_error(model_id)
                    # Non-HTTP failure (network, pre-status timeout, SDK error) —
                    # treated as a retryable_error in the outcome ratio: those
                    # are exactly the faults a health-aware router should absorb
                    # by selecting a different endpoint on the next attempt.
                    outcome_metrics.record_upstream_attempt(None)
                    self._record_model_attempt(
                        incoming_model,
                        upstream_model,
                        "retryable_error",
                        outcome_metrics.NO_STATUS_CODE,
                    )
                    # status_code=None → logged as code="none" (no HTTP status).
                    log_upstream_attempt_failure(
                        model=model_id,
                        attempt=attempt + 1,
                        status_code=None,
                        error=exc,
                        upstream_model=upstream_model,
                    )
                    last_exc = exc
                    last_upstream_model = upstream_model
                    continue

                backend_latency_ms = (time.monotonic() - started_at) * 1000.0
                set_tags(attempt_span, {
                    "switchyard.outcome": "success",
                    "switchyard.upstream_status_code": 200,
                })
                if self._stats is not None:
                    await self._stats.record_success(model_id, backend_latency_ms)
                outcome_metrics.record_upstream_attempt(200)
                self._record_model_attempt(
                    incoming_model, upstream_model, "success", "200",
                )
                # A successful attempt after at least one failure is direct
                # evidence the steering logic rescued this request. Counted
                # once per client request, not per recovered retry.
                if attempt > 0:
                    outcome_metrics.record_retry_recovered()

                # ``ctx.selected_model`` is the cross-language hook the Rust
                # ``StatsResponseProcessor`` reads to attribute token usage and
                # end-to-end latency per endpoint. Without it, the response
                # processor sees no ``BackendSelection`` and buckets every call
                # to ``model="<unknown>"`` on /metrics.
                ctx.selected_model = model_id
                # ``backend_call_latency_ms`` lets the Rust ``StatsResponseProcessor``
                # compute ``routing_overhead_ms = total_latency - backend_latency``
                # for this Python-only backend, which can't be wrapped by the
                # Rust ``StatsLlmBackend`` that normally publishes this signal.
                ctx.backend_call_latency_ms = backend_latency_ms

                # The selection that actually served the request, surfaced to
                # the endpoint layer as ``x-switchyard-*`` response headers so
                # a front proxy can enrich its own spend-log row with the same
                # correlation id the provider row received.
                ctx.metadata[CTX_ROUTE_SELECTION] = route_selection

                # Pin this conversation to the endpoint that served it so later
                # turns reuse it (warm cache). Re-pinning on every success also
                # follows a recovery: if the previous pin degraded and we
                # re-routed, the endpoint that worked becomes the new pin.
                # (No-op when affinity is disabled.)
                await self._affinity.pin(ctx, request, model_id)

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

        # Failure-origin annotation for the client-facing error envelope:
        # every exhausted attempt failed at a selected upstream, so what the
        # endpoint surfaces — the HTTP passthrough above, or the 500 for
        # status-less network failures — is provider-originated. (The stash
        # helpers below mark Switchyard-originated rejections instead.)
        ctx.metadata[CTX_ERROR_SOURCE] = ERROR_SOURCE_PROVIDER
        if last_upstream_model is not None:
            ctx.metadata[CTX_UPSTREAM_MODEL] = last_upstream_model

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

    def _route_selection(
        self,
        *,
        incoming_model: str | None,
        model_id: str,
        upstream_model: str,
        correlation_id: str,
    ) -> dict[str, str | None]:
        """Build the route-selection payload for one upstream attempt.

        This is the JSON value of the outbound ``x-litellm-spend-logs-metadata``
        header and, for the attempt that succeeds, the
        :data:`CTX_ROUTE_SELECTION` record behind the ``x-switchyard-*``
        response headers. ``router_model`` falls back to the configured route id
        when the client body carried no model; ``router_selected_provider`` is
        the upstream model's leading path segment (IH/LiteLLM naming, e.g.
        ``"openai/openai/gpt-5.4"`` → ``"openai"``).
        """
        return {
            "router_model": incoming_model or self._config.route_model,
            "router_strategy": ROUTER_STRATEGY_LATENCY,
            "router_selected_endpoint": model_id,
            "router_selected_model": upstream_model,
            "router_selected_provider": upstream_model.split("/", 1)[0],
            "router_correlation_id": correlation_id,
        }

    async def _call_endpoint(
        self,
        model_id: str,
        target_request_type: ChatRequestType,
        *,
        api_key: str | None,
        body: dict[str, object],
        extra_headers: dict[str, str],
    ) -> object:
        # ``extra_headers`` rides the client wrapper's ``**kwargs`` into the
        # OpenAI SDK's per-request header merge (over ``default_headers``).
        # A passthrough body may itself carry an SDK-style ``extra_headers``
        # field (shape-preserving translation keeps unknown keys); merge it
        # under ours rather than letting it collide with the keyword — it used
        # to ride ``**body`` into the same SDK parameter, and the spend-logs
        # header must win a name conflict so callers can't spoof it. A
        # non-mapping value is dropped: it could only ever have broken the
        # SDK call.
        client_extra = body.pop("extra_headers", None)
        headers: dict[str, object] = (
            {**client_extra, **extra_headers}
            if isinstance(client_extra, Mapping)
            else dict(extra_headers)
        )
        if target_request_type == ChatRequestType.OPENAI_RESPONSES:
            return await self._clients[model_id].aresponses(
                api_key=api_key,
                extra_headers=headers,
                **body,
            )
        return await self._clients[model_id].acompletion(
            api_key=api_key,
            extra_headers=headers,
            **body,
        )

    def _api_key_override_for_policy(self, caller_api_key: object) -> str | None:
        """Per-request key to forward upstream, or ``None`` to use the endpoint key.

        ``configured_endpoint`` never forwards a caller key. ``caller_override``
        and ``caller_required`` forward a usable caller key; they differ only in
        the no-key case, which ``call`` handles before any upstream attempt
        (``caller_override`` falls back to the endpoint key, ``caller_required``
        returns 401).
        """
        if self._config.credential_policy == "configured_endpoint":
            return None
        if isinstance(caller_api_key, str) and caller_api_key.strip():
            return caller_api_key
        return None

    def _record_model_attempt(
        self,
        requested_model: str | None,
        upstream_model: str,
        outcome: str,
        code: str,
    ) -> None:
        """Count one upstream attempt for the per-model /metrics breakdown.

        Event-loop only (no lock). ``requested_model`` is the client-supplied
        model; it is bounded to a config-derived id — the route id
        (``config.route_model``) or a configured endpoint id — with the
        ``"other"`` sentinel as fallback before becoming a Prometheus label, so
        caller-controlled input can't create unbounded metric-series
        cardinality.
        """
        requested = (
            requested_model if requested_model in self._requested_model_ids else "other"
        )
        key = (requested, upstream_model, outcome, code)
        self._upstream_attempts_by_model[key] = (
            self._upstream_attempts_by_model.get(key, 0) + 1
        )

    # -- Lifecycle ----------------------------------------------------------

    async def shutdown(self) -> None:
        """Stop the background :class:`HealthPoller` daemon and release the
        shared session-affinity store, when one is configured.

        Picked up automatically by the server's component teardown hook
        (``_run_lifecycle_method``), which awaits awaitable ``shutdown``
        results. Safe to call multiple times.
        """
        prometheus_emitter.unregister(self._render_prometheus_lines)
        self._poller.stop()
        await self._affinity.aclose()

    def is_ready(self) -> bool:
        """True once the background poller has completed at least one successful poll."""
        return self._poller.has_polled

    # -- Metrics emitter ----------------------------------------------------

    def _render_prometheus_lines(self) -> list[str]:
        """Emit per-endpoint health verdicts and poll-loop health gauges.

        Snapshotted under ``self._cache_lock`` so a poll concurrent with a
        scrape can't produce a mixed view across endpoints. Output is
        Prometheus exposition lines without trailing newline — composed
        by :mod:`switchyard.lib.endpoints.prometheus_emitter`.
        """
        with self._cache_lock:
            snapshot = dict(self._health_cache)

        lines: list[str] = []
        lines.append(
            "# HELP switchyard_endpoint_status "
            "Latency-Service verdict per endpoint (1 = current status; "
            "exactly one status row per model is non-zero)."
        )
        lines.append("# TYPE switchyard_endpoint_status gauge")
        for model_id, health in sorted(snapshot.items()):
            current = health.status.value
            for status in EndpointHealthStatus:
                value = 1 if status.value == current else 0
                labels = render_labels({"model": model_id, "status": status.value})
                lines.append(f"switchyard_endpoint_status{labels} {value}")

        lines.append(
            "# HELP switchyard_endpoint_last_latency_ms "
            "Last latency sample (ms) reported by the Latency Service per endpoint. "
            "Absent until the first poll publishes a non-null sample."
        )
        lines.append("# TYPE switchyard_endpoint_last_latency_ms gauge")
        for model_id, health in sorted(snapshot.items()):
            if health.last_latency_ms is None:
                continue
            labels = render_labels({"model": model_id})
            lines.append(
                f"switchyard_endpoint_last_latency_ms{labels} "
                f"{format_number(health.last_latency_ms)}"
            )

        lines.append(
            "# HELP switchyard_latency_service_poll_ok "
            "1 when the last poll succeeded, 0 when it failed or has not yet run."
        )
        lines.append("# TYPE switchyard_latency_service_poll_ok gauge")
        last_age = self._poller.seconds_since_last_success
        # ``poll_ok`` reflects the latest poll attempt. Combined with
        # ``poll_age_seconds``, scrapers can tell "never polled" (no age line)
        # from "polled, but the latest attempt failed" (age present, ok=0).
        poll_ok = 1 if self._poller.last_poll_ok else 0
        lines.append(f"switchyard_latency_service_poll_ok {poll_ok}")

        lines.append(
            "# HELP switchyard_latency_service_poll_age_seconds "
            "Monotonic seconds since the last successful poll. Absent before the "
            "first success."
        )
        lines.append("# TYPE switchyard_latency_service_poll_age_seconds gauge")
        if last_age is not None:
            lines.append(
                "switchyard_latency_service_poll_age_seconds "
                f"{format_number(last_age)}"
            )

        lines.append(
            "# HELP switchyard_latency_service_polls_total "
            "Total successful health polls since the poller started."
        )
        lines.append("# TYPE switchyard_latency_service_polls_total counter")
        lines.append(
            f"switchyard_latency_service_polls_total {self._poller.poll_successes}"
        )

        lines.append(
            "# HELP switchyard_latency_service_poll_failures_total "
            "Total failed health polls; each failure resets every endpoint to "
            "UNKNOWN."
        )
        lines.append("# TYPE switchyard_latency_service_poll_failures_total counter")
        lines.append(
            "switchyard_latency_service_poll_failures_total "
            f"{self._poller.poll_failures}"
        )

        # Warm-reuse counters — only meaningful (and only emitted) when session
        # affinity is enabled, so the metric surface stays clean for the common
        # per-turn-routing case. Reuse rate = hits / (hits + misses).
        if self._config.session_affinity:
            lines.append(
                "# HELP switchyard_affinity_hits_total "
                "Conversation turns served by an existing session-affinity pin "
                "(warm endpoint reuse)."
            )
            lines.append("# TYPE switchyard_affinity_hits_total counter")
            lines.append(f"switchyard_affinity_hits_total {self._affinity_hits}")

            lines.append(
                "# HELP switchyard_affinity_misses_total "
                "First or unpinnable turns routed by the latency-aware picker "
                "while session affinity was enabled."
            )
            lines.append("# TYPE switchyard_affinity_misses_total counter")
            lines.append(f"switchyard_affinity_misses_total {self._affinity_misses}")

            # Shared-store (L2) counters — emitted only when a shared pin store
            # is configured. Hits measure cross-worker/churn pin reuse; errors
            # count fail-open store operations (the alerting signal for a store
            # that is degraded while requests keep succeeding).
            if self._affinity.l2_enabled:
                lines.append(
                    "# HELP switchyard_affinity_l2_hits_total "
                    "Pins resolved from the shared (L2) affinity store after an "
                    "in-process (L1) miss — cross-worker warm reuse."
                )
                lines.append("# TYPE switchyard_affinity_l2_hits_total counter")
                lines.append(
                    f"switchyard_affinity_l2_hits_total {self._affinity.l2_hits}"
                )

                lines.append(
                    "# HELP switchyard_affinity_l2_errors_total "
                    "Shared (L2) affinity-store operations that failed open "
                    "(get or put); routing fell back to in-process pins only."
                )
                lines.append("# TYPE switchyard_affinity_l2_errors_total counter")
                lines.append(
                    f"switchyard_affinity_l2_errors_total {self._affinity.l2_errors}"
                )

                lines.append(
                    "# HELP switchyard_affinity_l2_breaker_open "
                    "1 while the shared-store circuit breaker is open "
                    "(operations skipped without a network attempt); 0 when closed."
                )
                lines.append("# TYPE switchyard_affinity_l2_breaker_open gauge")
                lines.append(
                    "switchyard_affinity_l2_breaker_open "
                    f"{int(self._affinity.l2_breaker_open)}"
                )

        # Per-model upstream-attempt breakdown — the latency-service-scoped
        # complement to the layer-aggregate ``switchyard_upstream_attempts_total``.
        # Series are created lazily per (requested_model, upstream_model, outcome,
        # code), so the surface stays empty until traffic flows. Snapshotted to a
        # plain dict for a stable view across the emission loop.
        attempts_by_model = dict(self._upstream_attempts_by_model)
        if attempts_by_model:
            lines.append(
                "# HELP switchyard_latency_upstream_attempts_total "
                "Upstream call attempts from the latency-service backend, labelled "
                "by requested (route) model, selected upstream model, outcome, and "
                "HTTP status code (code=\"none\" for non-HTTP failures). Per-model "
                "complement to the layer-aggregate switchyard_upstream_attempts_total; "
                "cardinality is bounded by the configured endpoint set."
            )
            lines.append("# TYPE switchyard_latency_upstream_attempts_total counter")
            # Sorted for deterministic exposition order across scrapes.
            for (requested_model, upstream_model, outcome, code), count in sorted(
                attempts_by_model.items()
            ):
                labels = render_labels({
                    "requested_model": requested_model,
                    "upstream_model": upstream_model,
                    "outcome": outcome,
                    "code": code,
                })
                lines.append(
                    f"switchyard_latency_upstream_attempts_total{labels} {count}"
                )
        return lines


def _latency_endpoint_request_type(value: str) -> ChatRequestType:
    if value == "openai_responses":
        return ChatRequestType.OPENAI_RESPONSES
    return ChatRequestType.OPENAI_CHAT


def _chat_response_for_request_type(
    request_type: ChatRequestType,
    result: object,
) -> ChatResponse:
    if request_type == ChatRequestType.OPENAI_RESPONSES:
        # Streaming yields raw SSE frame strings (``RawSSEFrameStream``) so the
        # upstream events pass through verbatim; anything async-iterable is a
        # stream, a plain mapping is the exact non-streaming JSON body.
        if isinstance(result, AsyncStream) or hasattr(result, "__aiter__"):
            return ChatResponse.openai_responses_stream(ResponsesApiStream(result))
        return ChatResponse.openai_responses_completion(result)
    if isinstance(result, AsyncStream):
        return ChatResponse.openai_stream(ResponseStream(result))
    return ChatResponse.openai_completion(result)


def _is_retryable_status(status_code: int) -> bool:
    """Whether an upstream status warrants failing over to a different endpoint.

    Retries 5xx + 408 + 429 (transient/capacity/server-side); other 4xx are the
    client's payload, which every replica rejects identically, so fail fast.
    Deliberately broader than ``outcome_metrics.RETRYABLE_STATUSES`` (metrics
    bucketing) — e.g. 502/503 must fail over but aren't metrics-retryable.
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


def _build_affinity_l2(config: LatencyServiceBackendConfig) -> AffinityPinStore | None:
    """Build the optional shared L2 pin store from config (``None`` = L1-only).

    ``affinity_store="memory"`` (the default) keeps pins per process. ``"redis"``
    imports :class:`RedisPinStore` lazily so the ``redis`` dependency stays
    optional. Config validation guarantees a URL when the store is Redis.
    """
    if config.affinity_store != "redis":
        return None
    from switchyard.lib.redis_pin_store import RedisPinStore

    assert config.affinity_store_url is not None  # enforced by config validator
    return RedisPinStore(
        config.affinity_store_url,
        ttl_seconds=config.affinity_store_ttl_seconds,
        key_prefix=config.affinity_key_prefix,
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
    # This 400 rides the upstream-status channel but is Switchyard's own
    # translation rejection — label it so the error-source header is truthful.
    ctx.metadata[CTX_ERROR_SOURCE] = ERROR_SOURCE_SWITCHYARD


def _reject_missing_caller_api_key(ctx: ProxyContext) -> None:
    """Reject a ``caller_required`` request that carries no caller API key.

    Stashes an HTTP 401 and a provider-compatible error envelope on ``ctx`` the
    same way upstream-status passthrough does, then raises so the chain errors
    out before any upstream call. No upstream is contacted or counted;
    ``upstream_response_from_ctx`` reads the stashed status to return a 401. This
    is what keeps the configured endpoint key from ever authenticating caller
    inference under the ``caller_required`` policy.
    """
    ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 401
    ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = {
        "error": {
            "message": "missing caller API key; supply it via the x-switchyard-api-key header",
            "type": "invalid_request_error",
            "code": "missing_caller_api_key",
        }
    }
    # Rides the upstream-status channel but no upstream was contacted — this
    # is Switchyard's own credential-policy rejection.
    ctx.metadata[CTX_ERROR_SOURCE] = ERROR_SOURCE_SWITCHYARD
    raise PermissionError("caller_required policy: missing caller API key")


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
