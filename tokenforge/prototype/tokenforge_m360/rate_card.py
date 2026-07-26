"""Versioned rate card and margin computation.

Switchyard's own price table (`switchyard/lib/cost_estimator.py`, `MODEL_PRICING`)
is a hardcoded Python dict of SUPPLIER list prices, USD per 1M tokens. It is not
configurable from YAML or env. TokenForge therefore keeps its own rate card for
two independent reasons:

  1. Customer PRICE is not supplier COST. Margin is the whole product.
  2. Negotiated NCP rates differ from list price, so Switchyard's `cost_usd` is
     itself wrong for most enterprise deployments -- we need a cost override too.

Externalizing `MODEL_PRICING` upstream is PR #2 in the design spec. Until it
lands, `cost_override_per_1m` is how a negotiated rate gets applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

USD_PER_1M = 1_000_000.0


@dataclass(frozen=True)
class TokenRate:
    """Prices in USD per 1M tokens, matching Switchyard's ModelPriceData shape."""

    input: float
    output: float
    cached: float = 0.0
    cache_write: float = 0.0


@dataclass(frozen=True)
class RateCard:
    """A pinned, versioned rate card.

    `rate_card_id` is stamped onto the request at preflight and echoed back on
    the intake record, so a late-arriving usage record rates at OCCURRENCE time
    rather than receipt time. That is what makes mid-cycle price changes safe.
    """

    rate_card_id: str
    currency: str = "USD"
    # Customer-facing price, keyed by model id.
    price_per_1m: dict[str, TokenRate] = field(default_factory=dict)
    # Negotiated supplier cost, when it differs from Switchyard's list price.
    cost_override_per_1m: dict[str, TokenRate] = field(default_factory=dict)
    # Per-escalation governance fee: charged when the router picks `strong`.
    escalation_fee_usd: float = 0.0
    # Flat per-request gateway fee.
    gateway_fee_usd: float = 0.0
    # Multiplier applied when the caller requested a latency/priority tier.
    tier_multiplier: dict[str, float] = field(default_factory=dict)

    def price_tokens(
        self,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        cache_creation_tokens: int = 0,
        tier: str | None = None,
        escalated: bool = False,
    ) -> float:
        """Customer price for one request. Returns 0.0 for an unpriced model.

        An unpriced model is a revenue leak, not a free request -- the caller is
        expected to quarantine rather than silently zero-rate. See
        `intake_receiver.py`.
        """
        rate = self.price_per_1m.get(model)
        if rate is None:
            return 0.0

        # Cached reads are billed at the cached rate, so subtract them from the
        # base input count. Switchyard reports `cached_tokens` as a SUBSET of
        # `prompt_tokens`, matching provider prompt-cache semantics.
        base_input = max(prompt_tokens - cached_tokens, 0)

        amount = (
            base_input * rate.input
            + completion_tokens * rate.output
            + cached_tokens * rate.cached
            + cache_creation_tokens * rate.cache_write
        ) / USD_PER_1M

        amount *= self.tier_multiplier.get(tier or "", 1.0)
        amount += self.gateway_fee_usd
        if escalated:
            amount += self.escalation_fee_usd
        return round(amount, 8)

    def is_priced(self, model: str) -> bool:
        return model in self.price_per_1m


def compute_margin(price_usd: float, cost_usd: float) -> tuple[float, float]:
    """Return (margin_usd, margin_pct). Margin pct is 0.0 at zero price."""
    margin = round(price_usd - cost_usd, 8)
    pct = round((margin / price_usd) * 100.0, 4) if price_usd > 0 else 0.0
    return margin, pct


# ---------------------------------------------------------------------------
# PoC rate card. Costs mirror Switchyard's MODEL_PRICING at `main`; prices are
# illustrative and set to demonstrate the Nutanix deck's margin thesis.
# ---------------------------------------------------------------------------
POC_RATE_CARD = RateCard(
    rate_card_id="rc_poc_v1",
    price_per_1m={
        # Switchyard cost: input 0.05 / output 0.20 / cached 0.005
        "nvidia/nvidia/Nemotron-3-Nano-30B-A3B": TokenRate(
            input=0.18, output=0.70, cached=0.02, cache_write=0.18
        ),
        # Switchyard cost: input 0.95 / output 4.00 / cached 0.16
        "nvidia/moonshotai/kimi-k2.6": TokenRate(
            input=2.20, output=9.00, cached=0.40, cache_write=2.20
        ),
    },
    escalation_fee_usd=0.0005,
    gateway_fee_usd=0.0001,
    tier_multiplier={"strong": 1.0, "weak": 1.0, "realtime": 1.35, "batch": 0.70},
)
