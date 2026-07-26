"""Switchyard stand-in for the TokenForge demo.

    ####################################################################
    #  THIS IS NOT SWITCHYARD. It is a stand-in that reproduces         #
    #  Switchyard's *observable contract* so the TokenForge integration #
    #  can be demonstrated on a machine without Python 3.12+ or a Rust  #
    #  toolchain (`nemo-switchyard` is not published to PyPI).          #
    #                                                                  #
    #  Swap it for the real proxy with two commands -- no TokenForge    #
    #  code changes:                                                   #
    #    export SWITCHYARD_INTAKE_TARGET_URL=http://localhost:9900/...  #
    #    switchyard serve --routing-profiles config/route.tokenforge.yaml
    ####################################################################

Everything reproduced here was read from the repository at `main`:

  * Endpoints: POST /v1/chat/completions, GET /v1/models, GET /v1/stats
    (alias /v1/routing/stats), POST /v1/stats/reset, GET /metrics, GET /health.
  * Credential precedence: x-switchyard-api-key -> Authorization: Bearer ->
    x-api-key, with sentinels {"switchyard", ""} treated as absent, and all
    three redacted to "[REDACTED]" in the retained header map.
  * Header-driven tier selection via `x-switchyard-tier`.
  * Intake sink: async POST off the response path, opt-in per request via
    `x-switchyard-intake-enabled: true` or body `store=true`, carrying cost_usd,
    cost_details, the token breakdown and the routing decision.
  * MODEL_PRICING values verbatim from switchyard/lib/cost_estimator.py.
  * Prometheus metric names and content type
    "text/plain; version=0.0.4; charset=utf-8", with only `model`/`tier` labels
    and NO cost metric and NO tenant label.
  * `/metrics` and `/v1/stats/reset` unauthenticated -- the proxy ignores
    inbound auth entirely.

Deliberately faithful omissions: no authentication, no multi-tenancy, no quota
or budget enforcement, no cost-aware routing, no caching. Those absences are the
reason TokenForge exists.
"""

from __future__ import annotations

import random
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from _http import Ctx, Service, post_json

# --- verbatim from switchyard/lib/cost_estimator.py, USD per 1M tokens -------
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "nvidia/nvidia/Nemotron-3-Nano-30B-A3B": {
        "input": 0.05, "output": 0.20, "cached": 0.005, "cache_write": 0.05,
    },
    "nvidia/moonshotai/kimi-k2.6": {
        "input": 0.95, "output": 4.00, "cached": 0.16, "cache_write": 0.95,
    },
}

STRONG = "nvidia/moonshotai/kimi-k2.6"
WEAK = "nvidia/nvidia/Nemotron-3-Nano-30B-A3B"

# Mirrors tokenforge/config/route.tokenforge.yaml.
ROUTES: Dict[str, Dict[str, Any]] = {
    "tf-tiered": {"type": "random_routing", "strong": STRONG, "weak": WEAK,
                  "strong_probability": 0.0},
    "tf-escalating": {"type": "deterministic", "strong": STRONG, "weak": WEAK,
                      "classifier": WEAK},
    "tf-nemotron": {"type": "model", "target": WEAK},
}

_SENSITIVE = frozenset({"authorization", "x-api-key", "x-switchyard-api-key"})
_KEY_SENTINELS = frozenset({"switchyard", ""})

_lock = threading.Lock()
_stats = {
    "total_requests": 0, "total_errors": 0,
    "prompt_tokens": 0, "completion_tokens": 0,
    "cached_tokens": 0, "cache_creation_tokens": 0,
}
_by_model: Dict[Tuple[str, str], int] = {}

service = Service("switchyard-sim", 8080)
_intake_url: Optional[str] = None
_rng = random.Random(1337)


def configure(intake_target_url: Optional[str]) -> None:
    """Equivalent of --intake-target-url / $SWITCHYARD_INTAKE_TARGET_URL."""
    global _intake_url
    _intake_url = intake_target_url


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _select_tier(route: Dict[str, Any], ctx: Ctx) -> Tuple[str, str, bool]:
    """Return (tier, model, classifier_was_called).

    `x-switchyard-tier` is honoured for router types, exactly as Switchyard's
    shipped `header_routing` profile does. This is what lets TokenForge downgrade
    a near-cap tenant with no Switchyard code at all.
    """
    kind = route["type"]
    if kind == "model":
        return "direct", route["target"], False

    requested = (ctx.header("x-switchyard-tier") or "").lower()
    if requested in {"strong", "weak"}:
        return requested, route[requested], False

    if kind == "deterministic":
        # A real classifier LLM call happens here -- BEFORE tier selection. This
        # is precisely why budget denial belongs at TokenForge Edge and not in a
        # request processor: a denial here has already burned classifier tokens.
        escalate = _rng.random() < 0.35
        tier = "strong" if escalate else "weak"
        return tier, route[tier], True

    tier = "strong" if _rng.random() < route.get("strong_probability", 0.0) else "weak"
    return tier, route[tier], False


def _usage(model: str) -> Dict[str, int]:
    prompt = _rng.randint(1_200, 24_000)
    cached = int(prompt * _rng.choice([0.0, 0.0, 0.55, 0.8]))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": _rng.randint(120, 2_400),
        "cached_tokens": cached,
        "cache_creation_tokens": 0 if cached else _rng.choice([0, 0, prompt // 2]),
    }


def _cost(model: str, usage: Dict[str, int]) -> Dict[str, Any]:
    """Switchyard emits cost only for models present in MODEL_PRICING."""
    rate = MODEL_PRICING.get(model)
    if rate is None:
        return {}
    base_input = max(usage["prompt_tokens"] - usage["cached_tokens"], 0)
    base = base_input * rate["input"] / 1e6
    cached = usage["cached_tokens"] * rate["cached"] / 1e6
    write = usage["cache_creation_tokens"] * rate["cache_write"] / 1e6
    output = usage["completion_tokens"] * rate["output"] / 1e6
    return {
        "cost_usd": round(base + cached + write + output, 10),
        "cost_input_usd": round(base + cached + write, 10),
        "cost_output_usd": round(output, 10),
        "cost_details": {
            "base_input": round(base, 10),
            "cached_input": round(cached, 10),
            "cache_write": round(write, 10),
        },
    }


def _retained_headers(ctx: Ctx) -> Dict[str, list]:
    """Switchyard retains headers as dict[str, list[str]], lowercased, with the
    three credential headers redacted."""
    out: Dict[str, list] = {}
    for name, value in ctx.headers.items():
        out[name] = ["[REDACTED]"] if name in _SENSITIVE else [value]
    return out


def _caller_api_key(ctx: Ctx) -> Optional[str]:
    """Verified precedence. Note Switchyard NEVER validates this key -- it only
    harvests it and forwards it upstream as the caller's billing credential."""
    for name in ("x-switchyard-api-key",):
        value = ctx.header(name)
        if value and value.lower() not in _KEY_SENTINELS:
            return value
    auth = ctx.header("authorization")
    if auth:
        scheme, _, value = auth.partition(" ")
        if scheme.lower() == "bearer" and value:
            return value
    return ctx.header("x-api-key")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@service.route("POST", "/v1/chat/completions")
def chat_completions(ctx: Ctx):
    model = ctx.body.get("model") or ""
    route = ROUTES.get(model)
    if route is None:
        with _lock:
            _stats["total_errors"] += 1
        return 404, {"error": {"message": f"No route registered for model {model}",
                               "type": "model_not_found", "code": "model_not_found"}}

    tier, served_model, classifier_called = _select_tier(route, ctx)
    usage = _usage(served_model)

    # A classifier call is real spend that a downstream denial cannot recover.
    classifier_usage = {"prompt_tokens": _rng.randint(400, 1200),
                        "completion_tokens": 12} if classifier_called else None

    cost = _cost(served_model, usage)

    with _lock:
        _stats["total_requests"] += 1
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens",
                    "cache_creation_tokens"):
            _stats[key] += usage[key]
        if classifier_usage:
            _stats["prompt_tokens"] += classifier_usage["prompt_tokens"]
            _stats["completion_tokens"] += classifier_usage["completion_tokens"]
        _by_model[(model, tier)] = _by_model.get((model, tier), 0) + 1

    correlation_id = "swyd_%012x" % _rng.getrandbits(48)
    request_id = ctx.header("x-request-id") or ("req_%012x" % _rng.getrandbits(48))

    _emit_intake(ctx, model, served_model, tier, usage, cost,
                 correlation_id, request_id, classifier_usage)

    return 200, {
        "id": "chatcmpl-" + correlation_id,
        "object": "chat.completion",
        "model": served_model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": "[simulated completion]"}}],
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        },
        # Response headers in the real proxy; surfaced in the body here so the
        # demo driver can assert on them without header plumbing.
        "_switchyard": {
            "x-switchyard-upstream-model": served_model,
            "x-switchyard-router-correlation-id": correlation_id,
            "x-switchyard-tier": tier,
        },
    }


def _emit_intake(ctx: Ctx, route: str, served_model: str, tier: str,
                 usage: Dict[str, int], cost: Dict[str, Any],
                 correlation_id: str, request_id: str,
                 classifier_usage: Optional[Dict[str, int]]) -> None:
    """Fire-and-forget POST to the intake target, off the response path.

    Opt-in per request, matching Switchyard: `x-switchyard-intake-enabled: true`
    or body `store=true`.
    """
    enabled = (ctx.header("x-switchyard-intake-enabled") or "").lower() in {"1", "true", "yes"}
    if not (enabled or ctx.body.get("store") is True):
        return
    if not _intake_url:
        return

    record = {
        "provider": "switchyard",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model": route,
        "user_id": "anon-machine-id",   # Switchyard's ~/.switchyard/user_id
        "routing": {
            "router_type": ROUTES[route]["type"],
            "routed_to": tier,
            "router_selected_model": served_model,
            "router_selected_provider": "nim",
            "router_correlation_id": correlation_id,
        },
        "usage": dict(usage),
        "request_headers": _retained_headers(ctx),
        "capture_content": False,
    }
    record.update(cost)
    if classifier_usage:
        record["classifier_usage"] = classifier_usage

    # Real Switchyard uses a bounded queue with retries; a thread is close
    # enough and preserves the important property: off the response path.
    threading.Thread(
        target=lambda: _safe_post(_intake_url, record), daemon=True
    ).start()


def _safe_post(url: str, record: dict) -> None:
    try:
        post_json(url, record)
    except Exception:  # noqa: BLE001 - fire and forget, matching the real sink
        pass


@service.route("GET", "/v1/models")
def models(ctx: Ctx):
    return 200, {"object": "list", "data": [
        {"id": name, "object": "model", "owned_by": "switchyard"} for name in ROUTES
    ]}


@service.route("GET", "/v1/stats")
def stats(ctx: Ctx):
    with _lock:
        return 200, dict(_stats)


@service.route("GET", "/v1/routing/stats")
def routing_stats(ctx: Ctx):
    return stats(ctx)


@service.route("POST", "/v1/stats/reset")
def reset_stats(ctx: Ctx):
    # Unauthenticated in the real proxy, faithfully reproduced. This is one of
    # the reasons Switchyard must never face external customers directly.
    with _lock:
        for key in _stats:
            _stats[key] = 0
        _by_model.clear()
    return 200, {"status": "reset"}


@service.route("GET", "/metrics")
def metrics(ctx: Ctx):
    """Prometheus text 0.0.4. Note: no cost metric, no tenant label -- only
    `model` and optional `tier`. Unauthenticated by design."""
    with _lock:
        lines = [
            "# TYPE switchyard_total_requests counter",
            "switchyard_total_requests %d" % _stats["total_requests"],
            "# TYPE switchyard_total_errors counter",
            "switchyard_total_errors %d" % _stats["total_errors"],
            "# TYPE switchyard_prompt_tokens_total counter",
            "switchyard_prompt_tokens_total %d" % _stats["prompt_tokens"],
            "# TYPE switchyard_completion_tokens_total counter",
            "switchyard_completion_tokens_total %d" % _stats["completion_tokens"],
            "# TYPE switchyard_cached_tokens_total counter",
            "switchyard_cached_tokens_total %d" % _stats["cached_tokens"],
            "# TYPE switchyard_cache_creation_tokens_total counter",
            "switchyard_cache_creation_tokens_total %d" % _stats["cache_creation_tokens"],
            "# TYPE switchyard_requests_total counter",
        ]
        for (model, tier), count in sorted(_by_model.items()):
            lines.append(
                'switchyard_requests_total{model="%s",tier="%s"} %d' % (model, tier, count)
            )
    return 200, ("text/plain; version=0.0.4; charset=utf-8", "\n".join(lines) + "\n")


@service.route("GET", "/health")
def health(ctx: Ctx):
    return 200, {"status": "ok"}
