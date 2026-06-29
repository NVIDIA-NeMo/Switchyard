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
LatencyServiceCredentialPolicy = Literal["configured_endpoint", "caller_override"]


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
        credential_policy: Which credential wins when the inbound HTTP
            request carries a caller key.  ``"configured_endpoint"`` keeps
            using each endpoint's configured ``api_key``; ``"caller_override"``
            opts into BYO-key forwarding.
        session_affinity: When ``True``, pin each conversation to the endpoint
            that first served it (cache stays warm); a pin is broken only when
            its endpoint degrades or the call fails. Per process. Default off.
        affinity_max_sessions: Bounded-LRU cap on pinned conversations; ignored
            when ``session_affinity`` is off.
        enable_stats: Retained config key (accepted, no longer wires anything).
            Per-request metrics are emitted by the chain executor via
            OpenTelemetry and exposed at ``GET /metrics``; see
            :mod:`switchyard.lib.observability`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    latency_service_url: str = ""
    endpoints: list[LatencyServiceEndpoint] = Field(default_factory=list)
    poll_interval_s: float = 10.0
    poll_timeout_s: float = 5.0
    max_retries: int = 2
    credential_policy: LatencyServiceCredentialPolicy = "configured_endpoint"
    session_affinity: bool = False
    affinity_max_sessions: int = Field(default=10_000, ge=0)
    enable_stats: bool = True

    @model_validator(mode="after")
    def _affinity_capacity_nonzero_when_enabled(self) -> Self:
        # A zero-capacity affinity store retains nothing — silently non-sticky.
        if self.session_affinity and self.affinity_max_sessions == 0:
            raise ValueError(
                "affinity_max_sessions must be > 0 when session_affinity is enabled"
            )
        return self
