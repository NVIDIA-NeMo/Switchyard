# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration models for the Latency Service usage case.

``LatencyServiceEndpoint`` describes one LLM backend monitored by the
Latency Service.  ``LatencyServiceBackendConfig`` bundles the full
backend configuration — URL of the Latency Service, the endpoint list,
and polling/retry parameters.

The ``model`` field on each endpoint doubles as the endpoint ID used by
the Latency Service's health API — mirroring the routing-by-model-name
convention the rest of the library already follows.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LatencyServiceRequestType = Literal["openai_chat", "openai_responses"]
LatencyServiceCredentialPolicy = Literal[
    "configured_endpoint", "caller_override", "caller_required"
]


class LatencyServiceEndpoint(BaseModel):
    """One LLM backend registered with the Latency Service.

    The ``model`` field is the endpoint ID used by the Latency Service —
    it must be unique across the endpoint list and it is the value the
    Latency Service returns health verdicts under.  By default it is also
    the value written into ``body["model"]`` when calling the upstream;
    set ``upstream_model`` when the upstream expects a different name
    (e.g. routing the latency-service key ``"openai/gpt-5.5"`` through an
    IH gateway that expects ``"openai/openai/gpt-5.5"``).

    Attributes:
        model: Latency-service lookup key.  Must be unique across the
            endpoint list.  Also used as ``body["model"]`` unless
            ``upstream_model`` is set.
        upstream_model: Optional override for ``body["model"]`` sent to
            the upstream LLM.  Defaults to ``model`` when ``None``.
        api_key: API key for the backing LLM API.
        base_url: Base URL for the backing LLM API (include ``/v1``).
        timeout: Request timeout in seconds, forwarded to the underlying
            ``OpenAILLMClient``.  ``None`` uses the client default.
        request_type: Upstream OpenAI API surface used for this endpoint.
            ``"openai_chat"`` sends ``/v1/chat/completions``; ``"openai_responses"``
            sends ``/v1/responses`` for Responses-only upstream models.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    upstream_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    request_type: LatencyServiceRequestType = "openai_chat"

    @field_validator("request_type", mode="before")
    @classmethod
    def _normalize_request_type(cls, value: object) -> object:
        if value == "chat":
            return "openai_chat"
        if value == "responses":
            return "openai_responses"
        return value


class LatencyServiceBackendConfig(BaseModel):
    """Configuration for :class:`LatencyServiceLLMBackend`.

    Attributes:
        latency_service_url: Base URL of the Latency Service
            (e.g. ``"http://latency-service.inference-hub.svc:8080"``).
        endpoints: LLM backends to route across.  Each must have a
            unique ``model`` — this is the routing + health-lookup key.
        poll_interval_s: How often the background poller refreshes
            health from the Latency Service.  Health is cached between
            polls; the request hot path never blocks on a network call.
        poll_timeout_s: Timeout for the health API call.
        max_retries: On error, retry on a different endpoint up to this
            many times.  Dedup prevents re-selecting an endpoint that
            already failed for the same request.
        route_model: Client-facing route id this backend serves (the
            route-table / YAML route key, e.g.
            ``"nvidia/switchyard/gpt-5.4"``).  Metrics-only: it joins the
            bounded ``requested_model`` label set on
            ``switchyard_latency_upstream_attempts_total`` so route-key
            traffic is attributed instead of collapsing to ``"other"``.
            Has no effect on routing.
        credential_policy: Which credential authenticates the upstream call.
            The caller key is read from the ``x-switchyard-api-key`` header
            (preferred — it survives proxies such as LiteLLM that strip
            ``Authorization``), then ``Authorization: Bearer`` / ``x-api-key``, on
            every ingress path (``/chat/completions``, ``/responses``,
            ``/messages``).  ``"configured_endpoint"`` always uses each endpoint's
            configured ``api_key`` and ignores any caller key.  ``"caller_override"``
            opts into BYO-key forwarding: a caller key is used when present, else
            the call falls back to the configured ``api_key``.  ``"caller_required"``
            forwards the caller key but never falls back — a request with no caller
            key is rejected with HTTP 401 and the configured ``api_key`` is never
            used for upstream inference (use this for per-user spend attribution,
            e.g. multi-tenant gateway routes).
        session_affinity: When ``True``, pin each conversation to the endpoint
            that first served it (cache stays warm); a pin is broken only when
            its endpoint degrades or the call fails. Per process. Default off.
        affinity_max_sessions: Bounded-LRU cap on pinned conversations; ignored
            when ``session_affinity`` is off.
        affinity_store: Shared L2 pin store behind the in-process LRU. ``"memory"``
            (default) keeps pins per process; ``"redis"`` shares them across
            workers/pods and persists them across pod churn. The store is
            best-effort — an L2 error never fails a request.
        affinity_store_url: Connection URL for the shared store (e.g.
            ``"redis://host:6379/0"``). Required when ``affinity_store`` is
            ``"redis"``.
        affinity_store_ttl_seconds: Expiry for a shared pin. The backend re-pins
            on every successful turn, so an active conversation slides its TTL.
        affinity_key_prefix: Namespace prefix for shared-store keys.
        enable_stats: When ``True`` (default), the factory wires a
            :class:`StatsRequestProcessor` + :class:`StatsResponseProcessor`
            pair sharing one :class:`StatsAccumulator` and wraps the
            backend in :class:`StatsLlmBackend`, so the chain contributes
            ``GET /metrics``, ``GET /v1/stats``, and the legacy
            ``GET /v1/routing/stats`` aliases via the standard
            ``get_endpoint()`` mechanism in :func:`build_switchyard_app`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    latency_service_url: str = ""
    endpoints: list[LatencyServiceEndpoint] = Field(default_factory=list)
    route_model: str | None = None
    poll_interval_s: float = 10.0
    poll_timeout_s: float = 5.0
    max_retries: int = 2
    credential_policy: LatencyServiceCredentialPolicy = "configured_endpoint"
    session_affinity: bool = False
    affinity_max_sessions: int = Field(default=10_000, ge=0)
    affinity_store: Literal["memory", "redis"] = "memory"
    affinity_store_url: str | None = None
    affinity_store_ttl_seconds: int = Field(default=3_600, gt=0)
    affinity_key_prefix: str = "swyd:pin:"
    enable_stats: bool = True

    @model_validator(mode="after")
    def _affinity_capacity_nonzero_when_enabled(self) -> Self:
        # A zero-capacity affinity store retains nothing — silently non-sticky.
        if self.session_affinity and self.affinity_max_sessions == 0:
            raise ValueError(
                "affinity_max_sessions must be > 0 when session_affinity is enabled"
            )
        return self

    @model_validator(mode="after")
    def _redis_store_requires_url_and_affinity(self) -> Self:
        # A shared store is dead config unless affinity is on and reachable.
        if self.affinity_store == "redis":
            if not self.session_affinity:
                raise ValueError(
                    'affinity_store="redis" requires session_affinity to be enabled'
                )
            if not self.affinity_store_url:
                raise ValueError(
                    'affinity_store="redis" requires affinity_store_url to be set'
                )
        return self
