"""TokenForge policy processor -- Phase B enforcement inside Switchyard.

Switchyard's processor contract is duck-typed. From `switchyard/lib/roles.py`:

    "Request-side and response-side components are now plain objects with async
     `process(...)` methods."

There is no base class, no ABC, no registry, no entry point. The contract is
enforced by reflection in `switchyard_rust/core.py`:

  * `process` must exist, be callable, and be a coroutine function.
  * it must return a `ChatRequest` (nominal isinstance check).
  * any exception is wrapped: `raise _processor_error(error) from error`.

Optional lifecycle hooks discovered by getattr: `startup()`, `shutdown()`, and
`get_endpoint()` -- the last one lets a processor contribute its own FastAPI
routes, which we use to expose budget state for operators.

Registration is by instance, into the ordered `pre_routing_request_processors`
list passed to the table builder. Route YAML can never declare a processor.

    IMPORTANT -- read design spec section 4.2 before relying on this alone.
    On a `deterministic` route the classifier LLM is called BEFORE tier
    selection, so a denial here has already burned classifier tokens. Primary
    enforcement belongs at TokenForge Edge; this processor is defense in depth
    plus the hook for post-classification routing decisions.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx

log = logging.getLogger("tokenforge.policy")

# Switchyard context metadata keys (string keys on ctx.metadata).
CTX_PROFILE_REQUEST_HEADERS = "_profile_request_headers"
CTX_CALLER_API_KEY = "_caller_api_key"
CTX_UPSTREAM_HTTP_STATUS = "_upstream_http_status"
CTX_ERROR_SOURCE = "_error_source"

TIER_HEADER = "x-switchyard-tier"


class TokenForgePolicyError(RuntimeError):
    """Carries a status the envelope should use.

    Today Switchyard flattens every processor exception to
    `SwitchyardProcessorError(str(error))`, so `status` does not survive to the
    HTTP response on its own -- we stamp `_upstream_http_status` on the context
    as a workaround. Upstream PR #1 (typed processor rejection) fixes this
    properly; when it lands, this class becomes the payload.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class TokenForgePolicyProcessor:
    """Pre-flight budget, entitlement and cost-aware-tier enforcement."""

    def __init__(
        self,
        decision_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 0.015,
        fail_open: bool = True,
    ) -> None:
        self._decision_url = decision_url
        self._api_key = api_key
        self._timeout_s = timeout_s
        # Sovereign and FSI deployments should set fail_open=False: if the
        # control plane is unreachable, deny rather than serve ungoverned spend.
        self._fail_open = fail_open
        self._client: httpx.AsyncClient | None = None

    # --- lifecycle (optional hooks, discovered by getattr) ----------------

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(timeout=self._timeout_s)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- the contract -----------------------------------------------------

    async def process(self, ctx: Any, request: Any) -> Any:
        headers = _headers(ctx)
        tenant_id = _first(headers, "x-tokenforge-tenant-id")

        if not tenant_id:
            # Unattributed traffic. In a multi-tenant deployment this means the
            # caller reached Switchyard without passing TokenForge Edge.
            if not self._fail_open:
                _stamp_status(ctx, 401)
                raise TokenForgePolicyError(401, "no_tenant", "no tenant attribution")
            log.warning("tokenforge: unattributed request served (fail_open)")
            return request

        decision = await self._decide(
            tenant_id=tenant_id,
            contract_id=_first(headers, "x-tokenforge-contract-id"),
            route=getattr(request, "model", None),
            requested_tier=_first(headers, TIER_HEADER),
            session_id=_first(headers, "x-switchyard-session-id"),
            agent_id=_first(headers, "x-switchyard-agent-id"),
            is_subagent=_first(headers, "x-switchyard-is-subagent"),
        )

        ctx.metadata["x-tokenforge-decision-id"] = decision["decision_id"]

        action = decision.get("action", "allow")

        if action == "deny":
            # 402 Payment Required is the honest status for a budget cap.
            _stamp_status(ctx, decision.get("status", 402))
            raise TokenForgePolicyError(
                decision.get("status", 402),
                decision.get("code", "budget_exceeded"),
                decision.get("message", "token budget exceeded"),
            )

        if action == "throttle":
            # Downgrade rather than deny. The elegant part: Switchyard's shipped
            # `header_routing` profile already routes on `x-switchyard-tier`, so
            # this needs no Switchyard code at all.
            forced = decision.get("force_tier", "weak")
            _set_header(ctx, TIER_HEADER, forced)
            cap = decision.get("max_tokens_cap")
            if cap:
                request = _clamp_max_tokens(request, int(cap))
            log.info(
                "tokenforge: throttled tenant=%s tier=%s cap=%s",
                tenant_id,
                forced,
                cap,
            )

        return request

    # --- optional: contribute an operator endpoint ------------------------

    def get_endpoint(self) -> Any | None:
        """Switchyard calls this to let a processor register FastAPI routes.

        Returning None keeps the PoC surface minimal. A production build returns
        an `Endpoint` exposing per-tenant budget state -- but note that
        Switchyard has NO auth on any route, so anything registered here is
        public. Do not expose tenant data this way outside a trust boundary.
        """
        return None

    # --- internals --------------------------------------------------------

    async def _decide(self, **payload: Any) -> dict[str, Any]:
        decision_id = "tfd_" + uuid.uuid4().hex[:24]
        if self._client is None:  # startup() not called
            return {"action": "allow", "decision_id": decision_id}

        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._client.post(
                self._decision_url,
                json={"decision_id": decision_id, **payload},
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()
            body.setdefault("decision_id", decision_id)
            return body
        except Exception as error:  # timeout, 5xx, unreachable control plane
            if self._fail_open:
                log.warning("tokenforge: decision failed, failing open: %s", error)
                return {"action": "allow", "decision_id": decision_id, "degraded": True}
            _stamp_status(None, 503)
            raise TokenForgePolicyError(
                503, "policy_unavailable", "policy control plane unavailable"
            ) from error


# ---------------------------------------------------------------------------
# ctx helpers. `ctx.metadata` is dict-like with string keys; the retained header
# map is dict[str, list[str]] with lowercased names, and the three credential
# headers are already redacted to "[REDACTED]" by Switchyard.
# ---------------------------------------------------------------------------


def _headers(ctx: Any) -> dict[str, Any]:
    try:
        return ctx.metadata.get(CTX_PROFILE_REQUEST_HEADERS) or {}
    except AttributeError:
        return {}


def _first(headers: dict[str, Any], name: str) -> str | None:
    value = headers.get(name)
    if isinstance(value, list):
        return value[0].strip() if value else None
    return value.strip() if isinstance(value, str) else None


def _set_header(ctx: Any, name: str, value: str) -> None:
    headers = _headers(ctx)
    headers[name] = [value]
    ctx.metadata[CTX_PROFILE_REQUEST_HEADERS] = headers


def _stamp_status(ctx: Any, status: int) -> None:
    if ctx is None:
        return
    try:
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = status
        ctx.metadata[CTX_ERROR_SOURCE] = "switchyard"
    except Exception:  # ctx shape is not guaranteed across versions
        log.debug("tokenforge: could not stamp status %s", status)


def _clamp_max_tokens(request: Any, cap: int) -> Any:
    """Clamp `max_tokens` in the request body.

    `ChatRequest` has no typed `messages` or `max_tokens` field -- the body is an
    untyped dict, mutated via `replace_body()`. Both OpenAI-style `max_tokens`
    and the newer `max_completion_tokens` are handled.
    """
    body = dict(request.to_body())
    for key in ("max_tokens", "max_completion_tokens"):
        current = body.get(key)
        if isinstance(current, int):
            body[key] = min(current, cap)
        elif current is None and key == "max_tokens":
            body[key] = cap
    request.replace_body(body)
    return request
