"""Analytical model for the marginal cost of LLM agent trajectories.

The model intentionally exposes assumptions instead of pretending that FLOPs,
energy, or allocated serving cost can be known exactly without telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class LLMArchitecture:
    """Transformer dimensions. `active_experts` controls MoE compute per token."""

    name: str
    layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    attention_heads: int
    kv_heads: int | None = None
    experts: int = 1
    active_experts: int = 1
    gated_mlp: bool = True
    bytes_per_weight: float = 2.0
    bytes_per_kv_element: float = 2.0

    def __post_init__(self) -> None:
        positive = (self.layers, self.hidden_size, self.intermediate_size,
                    self.vocab_size, self.attention_heads, self.experts,
                    self.active_experts)
        if any(x <= 0 for x in positive):
            raise ValueError("architecture dimensions must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.active_experts > self.experts:
            raise ValueError("active_experts cannot exceed experts")

    @property
    def effective_kv_heads(self) -> int:
        return self.kv_heads or self.attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.attention_heads

    @property
    def active_parameter_count(self) -> int:
        """Approximate parameters touched for one token (embeddings excluded)."""
        h, i = self.hidden_size, self.intermediate_size
        kv = self.effective_kv_heads * self.head_dim
        attention = h * h + 2 * h * kv + h * h  # Q, K/V, output
        mlp = (3 if self.gated_mlp else 2) * h * i * self.active_experts
        return self.layers * (attention + mlp)

    @property
    def total_parameter_count(self) -> int:
        h, i = self.hidden_size, self.intermediate_size
        kv = self.effective_kv_heads * self.head_dim
        attention = h * h + 2 * h * kv + h * h
        mlp = (3 if self.gated_mlp else 2) * h * i * self.experts
        return self.layers * (attention + mlp) + self.vocab_size * h

    def kv_bytes_per_token(self) -> float:
        return (self.layers * 2 * self.effective_kv_heads * self.head_dim
                * self.bytes_per_kv_element)


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    accelerators: int
    peak_flops_per_accelerator: float
    accelerator_power_watts: float
    host_power_watts: float = 0.0
    embodied_overhead_fraction: float = 0.0
    pue: float = 1.0
    dollars_per_accelerator_hour: float = 0.0

    def __post_init__(self) -> None:
        if self.accelerators <= 0 or self.peak_flops_per_accelerator <= 0:
            raise ValueError("hardware capacity must be positive")
        if self.pue < 1 or self.embodied_overhead_fraction < 0:
            raise ValueError("PUE must be >= 1 and overhead non-negative")


@dataclass(frozen=True)
class ServingProfile:
    """Serving effects. Utilization converts theoretical FLOPs to wall time."""

    name: str
    prefill_flop_utilization: float = 0.45
    decode_flop_utilization: float = 0.25
    average_batch_size: float = 1.0
    batch_efficiency: float = 1.0
    replica_share: float = 1.0
    fixed_call_latency_seconds: float = 0.0
    idle_power_fraction: float = 0.45

    def __post_init__(self) -> None:
        fractions = (self.prefill_flop_utilization, self.decode_flop_utilization,
                     self.batch_efficiency, self.replica_share,
                     self.idle_power_fraction)
        if any(not 0 < x <= 1 for x in fractions):
            raise ValueError("serving fractions must be in (0, 1]")
        if self.average_batch_size < 1 or self.fixed_call_latency_seconds < 0:
            raise ValueError("invalid batch size or latency")


@dataclass(frozen=True)
class PricingProfile:
    """Provider prices are per million tokens. Set to None for self-hosted."""

    input_per_million: float | None = None
    cached_input_per_million: float | None = None
    output_per_million: float | None = None
    fixed_per_call: float = 0.0


@dataclass(frozen=True)
class TrajectoryCall:
    name: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_mode: Literal["prefix_kv", "billing_only", "none"] = "prefix_kv"
    speculative_acceptance: float = 0.0
    draft_flops_per_proposed_token: float = 0.0
    tool_wait_seconds: float = 0.0
    keep_replica_during_tool_wait: bool = False

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.cached_input_tokens) < 0:
            raise ValueError("token counts cannot be negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input cannot exceed input")
        if not 0 <= self.speculative_acceptance < 1:
            raise ValueError("speculative_acceptance must be in [0, 1)")


@dataclass(frozen=True)
class AgentTrajectory:
    calls: tuple[TrajectoryCall, ...]


@dataclass(frozen=True)
class CallEstimate:
    name: str
    flops: float
    joules: float
    dollars: float
    compute_seconds: float
    allocated_seconds: float
    kv_cache_peak_bytes: float
    logical_tokens: int
    computed_tokens: int
    billed_tokens: int


@dataclass(frozen=True)
class TrajectoryEstimate:
    calls: tuple[CallEstimate, ...]
    flops: float = field(init=False)
    joules: float = field(init=False)
    dollars: float = field(init=False)
    allocated_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "flops", sum(c.flops for c in self.calls))
        object.__setattr__(self, "joules", sum(c.joules for c in self.calls))
        object.__setattr__(self, "dollars", sum(c.dollars for c in self.calls))
        object.__setattr__(self, "allocated_seconds", sum(c.allocated_seconds for c in self.calls))


class ModelEstimator:
    def __init__(self, architecture: LLMArchitecture, hardware: HardwareProfile,
                 serving: ServingProfile, pricing: PricingProfile | None = None):
        self.arch = architecture
        self.hw = hardware
        self.serving = serving
        self.pricing = pricing

    def _prefill_flops(self, new_tokens: int, cached_prefix: int) -> float:
        if not new_tokens:
            return 0.0
        # Dense projections/MLP: 2 FLOPs per active parameter. Attention score
        # and value aggregation: 4*h per query-key pair, including cached keys.
        linear = 2.0 * self.arch.active_parameter_count * new_tokens
        pairs = new_tokens * cached_prefix + new_tokens * (new_tokens + 1) / 2
        attention = 4.0 * self.arch.layers * self.arch.hidden_size * pairs
        return linear + attention

    def _decode_flops(self, output_tokens: int, initial_context: int) -> float:
        if not output_tokens:
            return 0.0
        linear = 2.0 * self.arch.active_parameter_count * output_tokens
        pairs = output_tokens * initial_context + output_tokens * (output_tokens - 1) / 2
        attention = 4.0 * self.arch.layers * self.arch.hidden_size * pairs
        return linear + attention

    def estimate_call(self, call: TrajectoryCall) -> CallEstimate:
        cache_computes_saved = call.cache_mode == "prefix_kv"
        cached = call.cached_input_tokens if cache_computes_saved else 0
        new_input = call.input_tokens - cached
        prefill = self._prefill_flops(new_input, cached)
        decode = self._decode_flops(call.output_tokens, call.input_tokens)

        # Speculation reduces serial decode steps, not arithmetic: the target
        # verifies proposed tokens and rejected proposals are wasted work.
        # Acceptance=0 disables speculation; otherwise it is accepted/proposed.
        speculation_factor = (1.0 if call.speculative_acceptance == 0 else
                              1.0 / call.speculative_acceptance)
        target_decode = decode * speculation_factor
        proposed = call.output_tokens * speculation_factor
        draft = proposed * call.draft_flops_per_proposed_token
        total_flops = prefill + target_decode + draft

        capacity = self.hw.accelerators * self.hw.peak_flops_per_accelerator
        prefill_s = prefill / (capacity * self.serving.prefill_flop_utilization)
        decode_s = (target_decode + draft) / (capacity * self.serving.decode_flop_utilization)
        compute_s = (prefill_s + decode_s) / (self.serving.average_batch_size
                                              * self.serving.batch_efficiency)
        allocated_s = compute_s + self.serving.fixed_call_latency_seconds
        if call.keep_replica_during_tool_wait:
            allocated_s += call.tool_wait_seconds

        active_power = (self.hw.accelerators * self.hw.accelerator_power_watts
                        + self.hw.host_power_watts)
        dynamic_s = compute_s
        idle_s = max(0.0, allocated_s - compute_s)
        facility_j = (dynamic_s * active_power + idle_s * active_power
                      * self.serving.idle_power_fraction) * self.hw.pue
        joules = facility_j * (1.0 + self.hw.embodied_overhead_fraction)

        allocated_accel_hours = (allocated_s * self.hw.accelerators
                                  * self.serving.replica_share / 3600)
        self_hosted = allocated_accel_hours * self.hw.dollars_per_accelerator_hour
        if self.pricing is None:
            dollars = self_hosted
        else:
            p = self.pricing
            uncached = call.input_tokens - call.cached_input_tokens
            cached_rate = (p.cached_input_per_million if p.cached_input_per_million
                           is not None else p.input_per_million)
            dollars = p.fixed_per_call
            dollars += uncached * (p.input_per_million or 0) / 1e6
            dollars += call.cached_input_tokens * (cached_rate or 0) / 1e6
            dollars += call.output_tokens * (p.output_per_million or 0) / 1e6

        return CallEstimate(
            name=call.name, flops=total_flops, joules=joules, dollars=dollars,
            compute_seconds=compute_s, allocated_seconds=allocated_s,
            kv_cache_peak_bytes=self.arch.kv_bytes_per_token()
            * (call.input_tokens + call.output_tokens),
            logical_tokens=call.input_tokens + call.output_tokens,
            computed_tokens=new_input + call.output_tokens,
            billed_tokens=call.input_tokens + call.output_tokens,
        )

    def estimate(self, trajectory: AgentTrajectory) -> TrajectoryEstimate:
        return TrajectoryEstimate(tuple(self.estimate_call(c) for c in trajectory.calls))
