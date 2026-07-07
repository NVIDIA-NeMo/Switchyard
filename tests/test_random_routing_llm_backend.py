# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Rust-owned random-routing backend path."""

from __future__ import annotations

import pytest

from switchyard.lib.backends.backend_format_resolver import (
    BackendFormatResolution,
    BackendFormatResolver,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.backends.multi_llm_backend import (
    build_multi_llm_backend,
    build_native_backend,
    build_target_backend,
    resolve_llm_target,
)
from switchyard.lib.processors.random_routing_request_processor import (
    RandomRoutingRequestProcessor,
)
from switchyard.lib.profiles.random_routing import (
    RandomRoutingConfig,
    RandomRoutingProfileConfig,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.components import (
    AnthropicNativeBackend,
    MultiLlmBackend,
    OpenAiNativeBackend,
)
from switchyard_rust.core import ChatRequest


def _target(
    target_id: str,
    model: str,
    *,
    format: object = BackendFormat.OPENAI,
    api_key: str | None = "sk-test",
    base_url: str | None = "https://example.invalid/v1",
) -> LlmTarget:
    return LlmTarget(
        id=target_id,
        model=model,
        format=format,
        api_key=api_key,
        base_url=base_url,
    )


def _request(model: str = "client-model") -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    })


class TestNativeBackendConstruction:
    def test_explicit_openai_target_builds_openai_native_backend(self) -> None:
        target = _target("strong", "gpt-test", format=BackendFormat.OPENAI)

        backend = build_native_backend(target)

        assert isinstance(backend, OpenAiNativeBackend)
        assert backend.target == target
        assert [item.value for item in backend.supported_request_types] == [
            "openai_chat"
        ]

    def test_nemotron_super_target_disables_thinking_by_default(self) -> None:
        target = _target(
            "weak",
            "nvidia/nvidia/nemotron-3-super-v3",
            format=BackendFormat.OPENAI,
        )

        backend = build_native_backend(target)

        assert backend.target.extra_body == {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_explicit_extra_body_wins_over_runtime_defaults(self) -> None:
        target = LlmTarget(
            id="weak",
            model="nvidia/nvidia/nemotron-3-super-v3",
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

        backend = build_native_backend(target)

        assert backend.target.extra_body == {
            "chat_template_kwargs": {"enable_thinking": True},
        }

    def test_explicit_anthropic_target_builds_anthropic_native_backend(self) -> None:
        target = _target("strong", "claude-test", format=BackendFormat.ANTHROPIC)

        backend = build_native_backend(target)

        assert isinstance(backend, AnthropicNativeBackend)
        assert backend.target == target
        assert [item.value for item in backend.supported_request_types] == [
            "anthropic"
        ]

    def test_auto_target_is_resolved_before_native_backend_build(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = _target("strong", "anthropic/claude-test", format=BackendFormat.AUTO)
        seen: list[LlmTarget] = []

        def fake_resolve(value: LlmTarget) -> BackendFormatResolution:
            seen.append(value)
            return BackendFormatResolution(
                format=BackendFormat.ANTHROPIC,
                reason="test resolution",
            )

        monkeypatch.setattr(BackendFormatResolver, "resolve", fake_resolve)

        resolved = resolve_llm_target(target)
        backend = build_native_backend(target)

        assert seen == [target, target]
        assert resolved.format == BackendFormat.ANTHROPIC
        assert resolved.model == target.model
        assert isinstance(backend, AnthropicNativeBackend)
        assert backend.target.format == BackendFormat.ANTHROPIC

    @pytest.mark.parametrize("missing", ["base_url", "api_key"])
    def test_auto_resolution_requires_probe_inputs(self, missing: str) -> None:
        target = _target(
            "strong",
            "anthropic/claude-test",
            format=BackendFormat.AUTO,
            base_url=None if missing == "base_url" else "https://example.invalid/v1",
            api_key=None if missing == "api_key" else "sk-test",
        )

        with pytest.raises(ValueError, match=f"requires {missing}"):
            build_native_backend(target)


class TestMultiLlmBackendConstruction:
    def test_build_target_backend_keeps_target_metadata(self) -> None:
        target = _target("weak", "weak-model")

        target_backend = build_target_backend(target)

        assert target_backend.target == target

    def test_multi_backend_advertises_all_inbound_request_types(self) -> None:
        backend = build_multi_llm_backend({
            "strong": _target("strong", "strong-model"),
            "weak": _target("weak", "weak-model"),
        })

        assert isinstance(backend, MultiLlmBackend)
        assert backend.target_ids() == ["strong", "weak"]
        assert [item.value for item in backend.supported_request_types] == [
            "openai_chat",
            "openai_responses",
            "anthropic",
            "gemini",
        ]

    def test_mapping_keys_fill_default_target_ids(self) -> None:
        backend = build_multi_llm_backend({
            "strong": LlmTarget(
                model="strong-model",
                api_key="sk-test",
                base_url="https://example.invalid/v1",
            ),
            "weak": LlmTarget(
                model="weak-model",
                api_key="sk-test",
                base_url="https://example.invalid/v1",
            ),
        })

        assert backend.target_ids() == ["strong", "weak"]

    def test_duplicate_target_ids_are_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="duplicate LLM target id"):
            build_multi_llm_backend([
                _target("duplicate", "strong-model"),
                _target("duplicate", "weak-model"),
            ])

    async def test_invalid_context_target_errors_before_network(self) -> None:
        backend = build_multi_llm_backend([
            _target("strong", "strong-model"),
            _target("weak", "weak-model"),
        ])
        ctx = ProxyContext()
        ctx.selected_target = "missing"

        with pytest.raises(RuntimeError, match="selected target missing is not configured"):
            await backend.call(ctx, _request())

    async def test_missing_selection_errors_before_network(self) -> None:
        backend = build_multi_llm_backend([
            _target("strong", "strong-model"),
            _target("weak", "weak-model"),
        ])

        with pytest.raises(RuntimeError, match="multiple targets but no selected target"):
            await backend.call(ProxyContext(), _request("client-only-model"))


class TestRandomRoutingProcessorContract:
    async def test_processor_rewrites_model_and_stamps_context(self) -> None:
        config = RandomRoutingConfig(
            strong=_target("strong", "strong-model"),
            weak=_target("weak", "weak-model"),
            strong_probability=1.0,
            rng_seed=7,
            fallback_target_on_evict="strong",
        )
        processor = RandomRoutingRequestProcessor(config.processor_config)
        ctx = ProxyContext()

        decision = processor.select("client-model")
        out = await processor.process(ctx, _request())

        assert decision["tier"] == "strong"
        assert decision["selected_target"] == "strong"
        assert decision["selected_model"] == "strong-model"
        assert decision["original_model"] == "client-model"
        assert out.model == "strong-model"
        assert ctx.selected_target == "strong"

    async def test_processor_preserves_non_model_request_fields(self) -> None:
        config = RandomRoutingConfig(
            strong=_target("strong", "strong-model"),
            weak=_target("weak", "weak-model"),
            strong_probability=0.0,
            rng_seed=7,
            fallback_target_on_evict="strong",
        )
        processor = RandomRoutingRequestProcessor(config.processor_config)
        ctx = ProxyContext()
        request = ChatRequest.openai_chat({
            "model": "client-model",
            "messages": [{"role": "user", "content": "keep me"}],
            "temperature": 0.2,
            "max_tokens": 123,
        })

        out = await processor.process(ctx, request)

        assert out.model == "weak-model"
        assert out.body["messages"] == [{"role": "user", "content": "keep me"}]
        assert out.body["temperature"] == 0.2
        assert out.body["max_tokens"] == 123
        assert ctx.selected_target == "weak"

    def test_factory_builds_processor_config_without_backend_fluff(self) -> None:
        config = RandomRoutingConfig(
            strong=_target("strong", "strong-model"),
            weak=_target("weak", "weak-model"),
            strong_probability=0.75,
            rng_seed=99,
            preset="opus_47_minimax",
            fallback_target_on_evict="strong",
        )

        processor_config = config.processor_config

        assert processor_config.strong == config.strong
        assert processor_config.weak == config.weak
        assert processor_config.strong_probability == 0.75
        assert processor_config.rng_seed == 99
        assert not hasattr(processor_config, "enable_stats")
        assert not hasattr(processor_config, "preset")

    async def test_profile_stats_backend_and_response_processor_share_accumulator(self) -> None:
        from switchyard.lib.processors.stats_response_processor_accumulator import (
            StatsResponseProcessor,
        )
        from switchyard.lib.stats_accumulator import StatsAccumulator
        from switchyard_rust.components import StatsLlmBackend

        stats = StatsAccumulator()
        config = RandomRoutingConfig(
            strong=_target("strong", "strong-model"),
            weak=_target("weak", "weak-model"),
            strong_probability=0.5,
            enable_stats=True,
            fallback_target_on_evict="strong",
        )
        profile = (
            RandomRoutingProfileConfig.from_config(config)
            .build()
            .with_runtime_components(
                stats_accumulator=stats,
                enable_stats=config.enable_stats,
            )
        )
        components = profile.iter_components()
        backend = next(component for component in components if isinstance(component, StatsLlmBackend))
        response_processor = next(
            component
            for component in components
            if isinstance(component, StatsResponseProcessor)
        )

        assert isinstance(backend, StatsLlmBackend)
        assert isinstance(response_processor, StatsResponseProcessor)

        await backend.accumulator.record_success("strong-model", tier="strong")
        snapshot = response_processor.accumulator.snapshot_sync()
        assert snapshot["total_requests"] == 1
        assert snapshot["tiers"]["strong"]["calls"] == 1
