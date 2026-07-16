# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named :class:`RandomRoutingConfig` presets keyed by model pair.

Each preset encodes an opinionated strong/weak model pair with explicit
backend formats.  A
preset name describes *which two models* — not *which workload* — so
the same pair can back Claude Code, PinchBench, SWE-bench, or any
other router-driven benchmark without name collision.

- :attr:`PRESETS` — the canonical list of preset ids, suitable for
  ``argparse(choices=...)`` on the CLI.
- :meth:`get` — classmethod returning the builder for a given id,
  raising :class:`ValueError` with the full allowed list on a miss.
- One :func:`staticmethod` per preset returning a fully-specified
  :class:`RandomRoutingConfig`.

**Strict API** — presets fix the model pair. Callers may only adjust
connectivity (``api_key``, ``base_url``) and the coin bias
(``strong_probability``). Need a different model? Drop back to
:class:`RandomRoutingConfig` and build it directly. Rationale: a
preset's whole point is "this specific pair is the one we ship" —
every model override reopens that decision, so we force the explicit
path instead.

``strong_probability`` controls routing — higher value = more
strong-tier traffic.

Example::

    from switchyard import ProfileSwitchyard, RandomRoutingProfileConfig, RandomRoutingPresets

    config = RandomRoutingPresets.opus_nemotron_super(
        api_key=nvidia_api_key,
        strong_probability=0.3,
    )
    switchyard = ProfileSwitchyard(RandomRoutingProfileConfig.from_config(config).build())
"""

from __future__ import annotations

from typing import Protocol, cast

from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
)
from switchyard.lib.profiles.random_routing import RandomRoutingConfig


class RandomRoutingPresetBuilder(Protocol):
    """Callable shape every :class:`RandomRoutingPresets` builder implements.

    Pinning the signature as a :class:`Protocol` does two jobs at once:
    gives :meth:`RandomRoutingPresets.get` a type-precise return
    annotation (no ``Callable[..., ...]`` Any-escape hatch), and
    documents the preset contract in one place — every new preset must
    accept exactly these keyword-only arguments.
    """

    def __call__(
        self,
        *,
        api_key: str | None = ...,
        base_url: str = ...,
        strong_probability: float = ...,
    ) -> RandomRoutingConfig: ...

# Shipping presets route through OpenRouter's OpenAI-compatible endpoint
# by default; callers override with ``base_url=`` for another gateway.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter model strings for the named random-routing presets.
#
# 4.6 vs 4.7 split — we ship both versions as parallel presets rather
# than replacing 4.6 with 4.7 because PinchBench / SWE-bench
# runs are built around 4.6 (see :meth:`opus_nemotron_super`'s
# docstring); ``*_47_*`` variants exist for callers who want the
# newer generation at the cost of re-validating the routing sweet
# spot.  Drop the 4.6 constants + presets once 4.7 benchmark history
# catches up.
_MODEL_OPUS_4_6 = "anthropic/claude-opus-4.6"
_MODEL_OPUS_4_7 = "anthropic/claude-opus-4.7"
_MODEL_GPT_5_2 = "openai/gpt-5.2"
_MODEL_GPT_5_5 = "openai/gpt-5.5"
_MODEL_NEMOTRON_SUPER = "nvidia/nemotron-3-super-120b-a12b"
_MODEL_KIMI_K2_6 = "moonshotai/kimi-k2.6"
_MODEL_MINIMAX_M2_7 = "minimax/minimax-m2.7"


class RandomRoutingPresets:
    """Builder of pre-built :class:`RandomRoutingConfig`s keyed by model pair.

    Each preset is a named strong+weak pair.
    Callers vary only the per-deployment knobs (``api_key``,
    ``base_url``, ``strong_probability``); the model identities and
    reasoning/output budgets are part of the preset's identity.

    Example::

        config = RandomRoutingPresets.opus_nemotron_super(
            api_key="...",
            strong_probability=0.5,
        )

        # Dispatch by id (CLI / config-driven workflows):
        builder = RandomRoutingPresets.get("opus_nemotron_super")
        config = builder(api_key="...", strong_probability=0.5)
    """

    #: Canonical preset ids.  Kept as a class attribute so CLI parsers
    #: can use ``argparse(choices=RandomRoutingPresets.PRESETS)`` and
    #: help text auto-lists the available presets.  Presets are named
    #: by their model pair (not by workload) so the same pair can
    #: back Claude Code, PinchBench, SWE-bench, etc. without
    #: workload-specific aliases.
    PRESETS: list[str] = [
        "opus_nemotron_super",
        "gpt5_nemotron_super",
        "opus_kimi",
        "opus_minimax",
        "opus_47_nemotron_super",
        "opus_47_kimi",
        "opus_47_minimax",
        "opus_47_gpt55",
    ]

    @classmethod
    def get(cls, preset_id: str) -> RandomRoutingPresetBuilder:
        """Return the builder method for *preset_id*.

        Raises:
            ValueError: If *preset_id* is not in :attr:`PRESETS`.  The
                error lists every allowed name so mistyped ids
                self-document their fix.
        """
        if preset_id not in cls.PRESETS:
            raise ValueError(
                f"Unknown random-routing preset {preset_id!r}. "
                f"Available: {cls.PRESETS}"
            )
        # Every PRESETS entry names a ``@staticmethod`` on this class,
        # so ``getattr`` resolves to the builder callable.  ``cast``
        # narrows the :class:`Any` that ``getattr`` produces back to
        # the shared :class:`RandomRoutingPresetBuilder` signature —
        # all presets implement it (verified by
        # :class:`TestStrictness`'s kwargs-only tests).
        return cast(RandomRoutingPresetBuilder, getattr(cls, preset_id))

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @staticmethod
    def opus_nemotron_super(
        *,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        strong_probability: float = 0.5,
    ) -> RandomRoutingConfig:
        """Claude Opus 4.6 (strong) + Nemotron-3 Super v3 (weak).

        The flagship strong/weak pair — strong-signal reasoning from
        Opus for multi-step work, cheaper reasoning from Nemotron for
        straightforward calls. This is the native Rust-backed upgrade
        of the legacy Claude Code random-routing pair at a 50/50 split.
        PinchBench on the original Opus 4.6 + Nemotron pair
        showed balanced routing matching all-strong accuracy, which is
        why this preset stays on 4.6 rather than chasing the latest
        point release.
        """
        return RandomRoutingConfig(
            strong=LlmTarget(
                id="strong",
                model=_MODEL_OPUS_4_6,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            weak=LlmTarget(
                id="weak",
                model=_MODEL_NEMOTRON_SUPER,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            strong_probability=strong_probability,
            fallback_target_on_evict="strong",
        )

    @staticmethod
    def gpt5_nemotron_super(
        *,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        strong_probability: float = 0.5,
    ) -> RandomRoutingConfig:
        """GPT-5.2 (strong) + Nemotron-3 Super v3 (weak).

        Same weak tier as :meth:`opus_nemotron_super` but swaps Opus
        for GPT-5.2 on the strong side. Matches the default
        strong+weak pair used by the random-routing CLI
        examples.
        """
        return RandomRoutingConfig(
            strong=LlmTarget(
                id="strong",
                model=_MODEL_GPT_5_2,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            weak=LlmTarget(
                id="weak",
                model=_MODEL_NEMOTRON_SUPER,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            strong_probability=strong_probability,
            fallback_target_on_evict="strong",
        )

    @staticmethod
    def opus_kimi(
        *,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        strong_probability: float = 0.5,
    ) -> RandomRoutingConfig:
        """Claude Opus 4.6 (strong) + Moonshot Kimi K2.6 (weak).

        Moonshot-weak-tier alternative to :meth:`opus_nemotron_super`
        — useful when Kimi's training mix better matches your
        workload's prompt style or language coverage than Nemotron's.
        """
        return RandomRoutingConfig(
            strong=LlmTarget(
                id="strong",
                model=_MODEL_OPUS_4_6,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            weak=LlmTarget(
                id="weak",
                model=_MODEL_KIMI_K2_6,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            strong_probability=strong_probability,
            fallback_target_on_evict="strong",
        )

    @staticmethod
    def opus_minimax(
        *,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        strong_probability: float = 0.5,
    ) -> RandomRoutingConfig:
        """Claude Opus 4.6 (strong) + MiniMax M2.7 (weak).

        Same Opus 4.6 strong tier as :meth:`opus_nemotron_super`;
        swaps Nemotron for MiniMax M2.7 on the weak side.  Use when
        MiniMax has better coverage for your workload's languages or
        prompt styles, or when you want to compare two reasoning-
        capable weak tiers side-by-side at the same split.
        """
        return RandomRoutingConfig(
            strong=LlmTarget(
                id="strong",
                model=_MODEL_OPUS_4_6,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            weak=LlmTarget(
                id="weak",
                model=_MODEL_MINIMAX_M2_7,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            strong_probability=strong_probability,
            fallback_target_on_evict="strong",
        )

    # ------------------------------------------------------------------
    # Opus 4.7 variants
    # ------------------------------------------------------------------
    #
    # Parallel to the 4.6 presets above — same weak tiers, only the
    # strong-tier version changes. Kept as separate factories (not a ``version=`` kwarg
    # on the 4.6 presets) because each preset's whole point is
    # "this specific pair is the one we ship"; parameterising the
    # version would reopen that decision on every call.

    @staticmethod
    def opus_47_nemotron_super(
        *,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        strong_probability: float = 0.5,
    ) -> RandomRoutingConfig:
        """Claude Opus 4.7 (strong) + Nemotron-3 Super v3 (weak).

        Pick this when you want the newer Opus generation and are
        willing to re-validate the routing sweet spot against the 4.6
        variant.
        """
        return RandomRoutingConfig(
            strong=LlmTarget(
                id="strong",
                model=_MODEL_OPUS_4_7,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            weak=LlmTarget(
                id="weak",
                model=_MODEL_NEMOTRON_SUPER,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            strong_probability=strong_probability,
            fallback_target_on_evict="strong",
        )

    @staticmethod
    def opus_47_kimi(
        *,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        strong_probability: float = 0.5,
    ) -> RandomRoutingConfig:
        """Claude Opus 4.7 (strong) + Moonshot Kimi K2.6 (weak).

        4.7 counterpart to :meth:`opus_kimi` — same weak tier, same
        split knob, newer strong tier.
        """
        return RandomRoutingConfig(
            strong=LlmTarget(
                id="strong",
                model=_MODEL_OPUS_4_7,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            weak=LlmTarget(
                id="weak",
                model=_MODEL_KIMI_K2_6,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            strong_probability=strong_probability,
            fallback_target_on_evict="strong",
        )

    @staticmethod
    def opus_47_minimax(
        *,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        strong_probability: float = 0.5,
    ) -> RandomRoutingConfig:
        """Claude Opus 4.7 (strong) + MiniMax M2.7 (weak).

        4.7 counterpart to :meth:`opus_minimax`. Use when you want to
        compare Opus 4.7's routing balance against the 4.6 baseline on
        MiniMax-weak traffic.
        """
        return RandomRoutingConfig(
            strong=LlmTarget(
                id="strong",
                model=_MODEL_OPUS_4_7,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            weak=LlmTarget(
                id="weak",
                model=_MODEL_MINIMAX_M2_7,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            strong_probability=strong_probability,
            fallback_target_on_evict="strong",
        )

    @staticmethod
    def opus_47_gpt55(
        *,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        strong_probability: float = 0.5,
    ) -> RandomRoutingConfig:
        """Claude Opus 4.7 (strong) + GPT-5.5 (weak).

        Cross-provider 4.7 pair — Anthropic Opus on the strong side,
        OpenAI GPT-5.5 on the weak side.  Useful when you want a
        weak tier that shares a training lineage with the strong-tier
        baselines GPT-5.x runs are compared against, instead of a
        Nemotron / Kimi / MiniMax weak tier.

        Both tiers use OpenRouter's OpenAI-compatible chat completions path.
        """
        return RandomRoutingConfig(
            strong=LlmTarget(
                id="strong",
                model=_MODEL_OPUS_4_7,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            weak=LlmTarget(
                id="weak",
                model=_MODEL_GPT_5_5,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
            ),
            strong_probability=strong_probability,
            fallback_target_on_evict="strong",
        )
