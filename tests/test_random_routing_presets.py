# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`RandomRoutingPresets`.

Covers three concerns:

- **Contract**: each shipping preset returns a valid
  :class:`RandomRoutingConfig` with the documented model pair and
  backend format; defaults cover connectivity / bias tunables; strict
  keyword-only API rejects unknown kwargs.
- **Dispatch**: :meth:`RandomRoutingPresets.get` resolves valid ids,
  rejects unknown ids with a helpful message, and stays in sync with
  :attr:`RandomRoutingPresets.PRESETS`.
- **Propagation**: ``api_key`` / ``base_url`` / ``strong_probability``
  reach both tiers on every preset.
"""

from __future__ import annotations

import pytest

from switchyard.lib.backends.llm_target import BackendFormat
from switchyard.lib.profiles.random_routing import (
    RandomRoutingConfig,
)
from switchyard.lib.profiles.random_routing_presets import RandomRoutingPresets

# Keep the canonical model strings here (instead of import-reusing
# module-private constants) so tests catch accidental model-string edits in
# ``random_routing_presets.py``. These are OpenRouter catalog ids.
_MODEL_OPUS_4_6 = "anthropic/claude-opus-4.6"
_MODEL_OPUS_4_7 = "anthropic/claude-opus-4.7"
_MODEL_GPT_5_2 = "openai/gpt-5.2"
_MODEL_GPT_5_5 = "openai/gpt-5.5"
_MODEL_NEMOTRON_SUPER = "nvidia/nemotron-3-super-120b-a12b"
_MODEL_KIMI_K2_6 = "moonshotai/kimi-k2.6"
_MODEL_MINIMAX_M2_7 = "minimax/minimax-m2.7"

_OPENROUTER = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# PRESETS table integrity
# ---------------------------------------------------------------------------


class TestPresetsTable:
    """``PRESETS`` is in sync with the ``@staticmethod`` builder set."""

    def test_presets_list_matches_staticmethods(self):
        # Every name in ``PRESETS`` resolves to a callable builder on the
        # class.  Inverse — every builder is in ``PRESETS`` — also holds
        # so operators can't call a preset that's missing from the CLI's
        # ``--preset`` choices list.
        for name in RandomRoutingPresets.PRESETS:
            builder = getattr(RandomRoutingPresets, name, None)
            assert callable(builder), f"{name} is not a callable builder"

    def test_presets_contents(self):
        assert set(RandomRoutingPresets.PRESETS) == {
            "opus_nemotron_super",
            "gpt5_nemotron_super",
            "opus_kimi",
            "opus_minimax",
            "opus_47_nemotron_super",
            "opus_47_kimi",
            "opus_47_minimax",
            "opus_47_gpt55",
        }


# ---------------------------------------------------------------------------
# get() dispatch
# ---------------------------------------------------------------------------


class TestGetDispatch:
    @pytest.mark.parametrize("preset_id", RandomRoutingPresets.PRESETS)
    def test_known_preset_returns_builder(self, preset_id):
        builder = RandomRoutingPresets.get(preset_id)
        # Builder takes only kwargs; a bare invocation should produce a
        # valid config with documented defaults.
        config = builder()
        assert isinstance(config, RandomRoutingConfig)

    def test_unknown_preset_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            RandomRoutingPresets.get("does_not_exist")
        msg = str(exc_info.value)
        # Error must name the offender and enumerate valid options so
        # the caller can fix a typo without opening source.
        assert "does_not_exist" in msg
        for name in RandomRoutingPresets.PRESETS:
            assert name in msg


# ---------------------------------------------------------------------------
# opus_nemotron_super
# ---------------------------------------------------------------------------


class TestOpusNemotronSuperContract:
    """Opus 4.6 strong, Nemotron-3 Super weak."""

    def test_default_config(self):
        config = RandomRoutingPresets.opus_nemotron_super()
        assert config.strong.model == _MODEL_OPUS_4_6
        assert config.weak.model == _MODEL_NEMOTRON_SUPER
        assert config.strong.format is BackendFormat.OPENAI
        assert config.weak.format is BackendFormat.OPENAI
        assert config.strong_probability == 0.5
        # Defaults: no creds injected, base_url points at OpenRouter.
        assert config.strong.api_key is None
        assert config.strong.base_url == _OPENROUTER
        assert config.weak.api_key is None
        assert config.weak.base_url == _OPENROUTER

    def test_credential_overrides_reach_both_tiers(self):
        config = RandomRoutingPresets.opus_nemotron_super(
            api_key="sk-test",
            base_url="https://custom.example/v1",
        )
        assert config.strong.api_key == "sk-test"
        assert config.strong.base_url == "https://custom.example/v1"
        assert config.weak.api_key == "sk-test"
        assert config.weak.base_url == "https://custom.example/v1"

    def test_strong_probability_override(self):
        config = RandomRoutingPresets.opus_nemotron_super(strong_probability=0.3)
        assert config.strong_probability == 0.3

    def test_invalid_probability_rejected(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            RandomRoutingPresets.opus_nemotron_super(strong_probability=1.5)


# ---------------------------------------------------------------------------
# gpt5_nemotron_super
# ---------------------------------------------------------------------------


class TestGpt5NemotronSuperContract:
    """GPT-5.2 strong + Nemotron-3 Super v3 weak."""

    def test_default_config(self):
        config = RandomRoutingPresets.gpt5_nemotron_super()
        assert config.strong.model == _MODEL_GPT_5_2
        assert config.weak.model == _MODEL_NEMOTRON_SUPER
        assert config.strong.format is BackendFormat.OPENAI
        assert config.weak.format is BackendFormat.OPENAI
        assert config.strong_probability == 0.5
        assert config.strong.base_url == _OPENROUTER
        assert config.weak.base_url == _OPENROUTER

    def test_credential_overrides_reach_both_tiers(self):
        config = RandomRoutingPresets.gpt5_nemotron_super(
            api_key="sk-test",
            base_url="https://custom.example/v1",
        )
        assert config.strong.api_key == "sk-test"
        assert config.weak.api_key == "sk-test"


# ---------------------------------------------------------------------------
# opus_kimi
# ---------------------------------------------------------------------------


class TestOpusKimiContract:
    """Opus 4.6 strong + Kimi K2.6 weak."""

    def test_default_config(self):
        config = RandomRoutingPresets.opus_kimi()
        assert config.strong.model == _MODEL_OPUS_4_6
        assert config.weak.model == _MODEL_KIMI_K2_6
        assert config.strong.format is BackendFormat.OPENAI
        assert config.weak.format is BackendFormat.OPENAI
        assert config.strong_probability == 0.5

    def test_credential_overrides_reach_both_tiers(self):
        config = RandomRoutingPresets.opus_kimi(
            api_key="sk-test",
            base_url="https://custom.example/v1",
        )
        assert config.strong.api_key == "sk-test"
        assert config.weak.api_key == "sk-test"


# ---------------------------------------------------------------------------
# opus_minimax
# ---------------------------------------------------------------------------


class TestOpusMinimaxContract:
    """Opus 4.6 strong + MiniMax M2.7 weak."""

    def test_default_config(self):
        config = RandomRoutingPresets.opus_minimax()
        assert config.strong.model == _MODEL_OPUS_4_6
        assert config.weak.model == _MODEL_MINIMAX_M2_7
        assert config.strong.format is BackendFormat.OPENAI
        assert config.weak.format is BackendFormat.OPENAI
        assert config.strong_probability == 0.5

    def test_credential_overrides_reach_both_tiers(self):
        config = RandomRoutingPresets.opus_minimax(
            api_key="sk-test",
            base_url="https://custom.example/v1",
        )
        assert config.strong.api_key == "sk-test"
        assert config.weak.api_key == "sk-test"


# ---------------------------------------------------------------------------
# opus_47_nemotron_super
# ---------------------------------------------------------------------------


class TestOpus47NemotronSuperContract:
    """4.7 counterpart to ``opus_nemotron_super``."""

    def test_default_config(self):
        config = RandomRoutingPresets.opus_47_nemotron_super()
        assert config.strong.model == _MODEL_OPUS_4_7
        assert config.weak.model == _MODEL_NEMOTRON_SUPER
        assert config.strong.format is BackendFormat.OPENAI
        assert config.weak.format is BackendFormat.OPENAI
        assert config.strong_probability == 0.5
        assert config.strong.api_key is None
        assert config.strong.base_url == _OPENROUTER
        assert config.weak.api_key is None
        assert config.weak.base_url == _OPENROUTER

    def test_credential_overrides_reach_both_tiers(self):
        config = RandomRoutingPresets.opus_47_nemotron_super(
            api_key="sk-test",
            base_url="https://custom.example/v1",
        )
        assert config.strong.api_key == "sk-test"
        assert config.strong.base_url == "https://custom.example/v1"
        assert config.weak.api_key == "sk-test"
        assert config.weak.base_url == "https://custom.example/v1"


# ---------------------------------------------------------------------------
# opus_47_kimi
# ---------------------------------------------------------------------------


class TestOpus47KimiContract:
    """4.7 counterpart to ``opus_kimi``."""

    def test_default_config(self):
        config = RandomRoutingPresets.opus_47_kimi()
        assert config.strong.model == _MODEL_OPUS_4_7
        assert config.weak.model == _MODEL_KIMI_K2_6
        assert config.strong.format is BackendFormat.OPENAI
        assert config.weak.format is BackendFormat.OPENAI
        assert config.strong_probability == 0.5

    def test_credential_overrides_reach_both_tiers(self):
        config = RandomRoutingPresets.opus_47_kimi(
            api_key="sk-test",
            base_url="https://custom.example/v1",
        )
        assert config.strong.api_key == "sk-test"
        assert config.weak.api_key == "sk-test"


# ---------------------------------------------------------------------------
# opus_47_minimax
# ---------------------------------------------------------------------------


class TestOpus47MinimaxContract:
    """4.7 counterpart to ``opus_minimax``."""

    def test_default_config(self):
        config = RandomRoutingPresets.opus_47_minimax()
        assert config.strong.model == _MODEL_OPUS_4_7
        assert config.weak.model == _MODEL_MINIMAX_M2_7
        assert config.strong.format is BackendFormat.OPENAI
        assert config.weak.format is BackendFormat.OPENAI
        assert config.strong_probability == 0.5

    def test_credential_overrides_reach_both_tiers(self):
        config = RandomRoutingPresets.opus_47_minimax(
            api_key="sk-test",
            base_url="https://custom.example/v1",
        )
        assert config.strong.api_key == "sk-test"
        assert config.weak.api_key == "sk-test"


# ---------------------------------------------------------------------------
# opus_47_gpt55
# ---------------------------------------------------------------------------


class TestOpus47Gpt55Contract:
    """4.7 strong + GPT-5.5 weak."""

    def test_default_config(self):
        config = RandomRoutingPresets.opus_47_gpt55()
        assert config.strong.model == _MODEL_OPUS_4_7
        assert config.weak.model == _MODEL_GPT_5_5
        assert config.strong.format is BackendFormat.OPENAI
        assert config.weak.format is BackendFormat.OPENAI
        assert config.strong_probability == 0.5
        assert config.strong.api_key is None
        assert config.strong.base_url == _OPENROUTER
        assert config.weak.api_key is None
        assert config.weak.base_url == _OPENROUTER

    def test_credential_overrides_reach_both_tiers(self):
        config = RandomRoutingPresets.opus_47_gpt55(
            api_key="sk-test",
            base_url="https://custom.example/v1",
        )
        assert config.strong.api_key == "sk-test"
        assert config.strong.base_url == "https://custom.example/v1"
        assert config.weak.api_key == "sk-test"
        assert config.weak.base_url == "https://custom.example/v1"


# ---------------------------------------------------------------------------
# Strictness — presets reject model/format overrides
# ---------------------------------------------------------------------------


class TestStrictness:
    """Presets expose only (api_key, base_url, strong_probability).

    Attempts to override the fixed model pair or backend format must fail at
    the Python level — Python's TypeError on unexpected kwargs is the
    enforcement mechanism, so these tests pin the surface area.
    """

    @pytest.mark.parametrize("preset_id", RandomRoutingPresets.PRESETS)
    @pytest.mark.parametrize(
        "forbidden_kwarg",
        [
            "strong_model",
            "weak_model",
            "strong_backend_format",
            "weak_backend_format",
        ],
    )
    def test_preset_rejects_unknown_kwarg(self, preset_id, forbidden_kwarg):
        builder = RandomRoutingPresets.get(preset_id)
        with pytest.raises(TypeError):
            builder(**{forbidden_kwarg: "anything"})

    @pytest.mark.parametrize("preset_id", RandomRoutingPresets.PRESETS)
    def test_preset_rejects_positional_args(self, preset_id):
        builder = RandomRoutingPresets.get(preset_id)
        with pytest.raises(TypeError):
            builder("sk-test")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_public_api_exposes_random_routing_presets():
    """``RandomRoutingPresets`` is importable from the top-level package."""
    from switchyard import RandomRoutingPresets as _Presets

    assert _Presets is RandomRoutingPresets
