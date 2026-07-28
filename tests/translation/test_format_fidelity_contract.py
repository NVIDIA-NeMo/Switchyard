# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format-fidelity contract: AUTO probes upstream in priority order.

BackendFormat.AUTO probes:
  1. /v1/chat/completions → OPENAI
  2. /v1/messages         → ANTHROPIC
  3. /v1/responses        → RESPONSES
  4. fallback             → OPENAI (Chat Completions assumed universal)

The TranslationEngine converts any inbound format to any backend format
through a neutral IR, so all combinations are valid regardless of client.
"""

import pytest

from switchyard.lib.backends import backend_format_resolver as resolver_mod
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.llm_client_builder import resolve_llm_target


class TestAutoFormatResolution:
    """BackendFormat.AUTO probes upstream capabilities and selects the best format."""

    def test_auto_resolves_to_responses_for_openai_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chat Completions fails, /v1/messages fails, /v1/responses succeeds → RESPONSES."""
        monkeypatch.setattr(
            resolver_mod, "probe_openai_chat_completions_support_sync",
            lambda *, base_url, api_key, **_kw: False,
        )
        monkeypatch.setattr(
            resolver_mod, "probe_anthropic_messages_support_sync",
            lambda *, base_url, api_key, **_kw: False,
        )
        monkeypatch.setattr(
            resolver_mod, "probe_openai_responses_support_sync",
            lambda *, base_url, api_key, **_kw: True,
        )

        target = LlmTarget(
            model="openai/gpt-5.2",
            format=BackendFormat.AUTO,
            base_url="https://api.openai.com/v1",
            api_key="sk-test",  # pragma: allowlist secret
        )
        resolved = resolve_llm_target(target)

        assert resolved.format is BackendFormat.RESPONSES

    def test_auto_falls_back_to_openai_for_nim_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NIM upstream: Chat Completions succeeds → OPENAI (first probe wins)."""
        monkeypatch.setattr(
            resolver_mod, "probe_openai_chat_completions_support_sync",
            lambda *, base_url, api_key, **_kw: True,
        )
        monkeypatch.setattr(
            resolver_mod, "probe_anthropic_messages_support_sync",
            lambda *, base_url, api_key, **_kw: pytest.fail("should not probe Anthropic"),
        )
        monkeypatch.setattr(
            resolver_mod, "probe_openai_responses_support_sync",
            lambda *, base_url, api_key, **_kw: pytest.fail("should not probe Responses"),
        )

        target = LlmTarget(
            model="nvidia/nvidia/nemotron-nano-9b-v2",
            format=BackendFormat.AUTO,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="sk-test",  # pragma: allowlist secret
        )
        resolved = resolve_llm_target(target)

        assert resolved.format is BackendFormat.OPENAI

    def test_auto_resolves_to_anthropic_for_messages_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chat Completions fails, /v1/messages succeeds → ANTHROPIC."""
        monkeypatch.setattr(
            resolver_mod, "probe_openai_chat_completions_support_sync",
            lambda *, base_url, api_key, **_kw: False,
        )
        monkeypatch.setattr(
            resolver_mod, "probe_anthropic_messages_support_sync",
            lambda *, base_url, api_key, **_kw: True,
        )

        target = LlmTarget(
            model="some-non-prefixed-model",
            format=BackendFormat.AUTO,
            base_url="https://api.anthropic.com/v1",
            api_key="sk-test",  # pragma: allowlist secret
        )
        resolved = resolve_llm_target(target)

        assert resolved.format is BackendFormat.ANTHROPIC

    def test_auto_prefix_fast_path_skips_all_probes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """anthropic/ and claude prefixes → ANTHROPIC immediately, no probes fired."""
        monkeypatch.setattr(
            resolver_mod, "probe_openai_chat_completions_support_sync",
            lambda **_kw: pytest.fail("prefix fast-path must not probe"),
        )
        monkeypatch.setattr(
            resolver_mod, "probe_anthropic_messages_support_sync",
            lambda **_kw: pytest.fail("prefix fast-path must not probe"),
        )
        monkeypatch.setattr(
            resolver_mod, "probe_openai_responses_support_sync",
            lambda **_kw: pytest.fail("prefix fast-path must not probe"),
        )

        target = LlmTarget(
            model="anthropic/claude-sonnet-4-5",
            format=BackendFormat.AUTO,
            base_url="https://api.anthropic.com/v1",
            api_key="sk-test",  # pragma: allowlist secret
        )
        resolved = resolve_llm_target(target)

        assert resolved.format is BackendFormat.ANTHROPIC

    def test_explicit_formats_bypass_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit RESPONSES/OPENAI/ANTHROPIC are honored as-is without probing."""
        def fail_probe(**_kw: object) -> bool:
            pytest.fail("explicit formats must not trigger a probe")

        monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync", fail_probe)
        monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync", fail_probe)
        monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync", fail_probe)

        for fmt in (BackendFormat.RESPONSES, BackendFormat.OPENAI, BackendFormat.ANTHROPIC):
            target = LlmTarget(
                model="some/model",
                format=fmt,
                base_url="https://api.openai.com/v1",
                api_key="sk-test",  # pragma: allowlist secret
            )
            assert resolve_llm_target(target).format is fmt
