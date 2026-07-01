# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Python-defined profile protocols and config decorators."""

from dataclasses import FrozenInstanceError, dataclass, is_dataclass
from typing import cast
from uuid import uuid4

import pytest

from switchyard import (
    Profile,
    ProfileConfigError,
    ProfileInput,
    build_profile,
    profile_config,
    profile_config_type,
)
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.table import (
    lookup_profile_config,
    register_profile_config,
    registered_profile_config_types,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    SwitchyardContextPoolExhaustedError,
    SwitchyardContextWindowExceededError,
)
from switchyard_rust.profiles import ProfileRequestMetadata


@dataclass(frozen=True)
class _StaticProcessedRequest:
    input: ProfileInput
    marker: str


class _StaticProfile:
    def __init__(self, content: str) -> None:
        self._content = content

    async def process(self, input: ProfileInput) -> _StaticProcessedRequest:
        return _StaticProcessedRequest(input=input, marker="processed")

    async def rprocess(
        self,
        processed: _StaticProcessedRequest,
        response: ChatResponse,
    ) -> ChatResponse:
        assert processed.marker == "processed"
        return response

    async def run(self, input: ProfileInput) -> ChatResponse:
        processed = await self.process(input)
        return await self.rprocess(
            processed,
            ChatResponse.openai_completion({
                "id": "python-profile-test",
                "object": "chat.completion",
                "model": processed.input.request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self._content},
                        "finish_reason": "stop",
                    }
                ],
            }),
        )


class _MetadataCapturingProcessor:
    def __init__(self) -> None:
        self.metadata: dict[str, object] = {}

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        self.metadata = dict(ctx.metadata)
        return request


class _StaticBackend(LLMBackend):
    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, _ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        return ChatResponse.openai_completion({
            "id": "python-profile-chain",
            "object": "chat.completion",
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        })


class _PickWeakProcessor:
    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        ctx.selected_target = "weak"
        return request


class _AlwaysOverflowBackend(LLMBackend):
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        _ = request
        self.calls.append(ctx.selected_target)
        error = SwitchyardContextWindowExceededError("target overflowed")
        error.target_id = ctx.selected_target
        raise error


class _OverflowWeakBackend(LLMBackend):
    """Overflows on 'weak', succeeds on any other target."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        self.calls.append(ctx.selected_target)
        if ctx.selected_target == "weak":
            error = SwitchyardContextWindowExceededError("weak overflowed")
            error.target_id = "weak"
            raise error
        return ChatResponse.openai_completion({
            "id": "chain-ok",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        })


def _request(*, msg: str = "hi") -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": "client/model",
        "messages": [{"role": "user", "content": msg}],
    })


async def test_profile_config_decorator_builds_dataclass_and_profile_runtime() -> None:
    @profile_config("unit-python-static", register=False)
    class StaticProfileConfig:
        content: str

        def build(self) -> _StaticProfile:
            return _StaticProfile(self.content)

    config = StaticProfileConfig(content="ok")

    assert is_dataclass(StaticProfileConfig)
    assert not hasattr(config, "__dict__")
    assert profile_config_type(config) == "unit-python-static"
    assert profile_config_type(StaticProfileConfig) == "unit-python-static"
    with pytest.raises(FrozenInstanceError):
        cast(object, config).content = "mutated"  # type: ignore[attr-defined]

    profile = build_profile(config)
    assert isinstance(profile, Profile)
    input = ProfileInput(_request())
    assert isinstance(input.metadata, ProfileRequestMetadata)
    response = await profile.run(input)

    assert response.body["id"] == "python-profile-test"
    assert response.body["model"] == "client/model"
    assert response.body["choices"][0]["message"]["content"] == "ok"


async def test_component_chain_profile_preserves_input_headers_in_context() -> None:
    processor = _MetadataCapturingProcessor()
    profile = ComponentChainProfile(
        request_processors=[processor],
        backend=_StaticBackend(),
    )
    metadata = ProfileRequestMetadata.from_headers({
        "X-Switchyard-Trace": ["trace-a", "trace-b"],
        "X-Request-Id": "request-1",
    })

    await profile.run(ProfileInput(_request(), metadata=metadata))

    assert processor.metadata["x-switchyard-trace"] == ["trace-a", "trace-b"]
    assert processor.metadata["x-request-id"] == ["request-1"]


async def test_component_chain_raises_pool_exhausted_when_fallback_overflows() -> None:
    """A second context-window overflow reports the final attempted target."""
    backend = _AlwaysOverflowBackend()
    profile = ComponentChainProfile(
        request_processors=[_PickWeakProcessor()],
        backend=backend,
        fallback_target_on_evict="strong",
    )

    with pytest.raises(SwitchyardContextPoolExhaustedError) as excinfo:
        await profile.run(ProfileInput(_request()))

    assert backend.calls == ["weak", "strong"]
    assert excinfo.value.last_target_id == "strong"
    assert excinfo.value.reason == "all attempted targets returned context-window overflow"


async def test_session_eviction_skips_evicted_target_on_subsequent_turn() -> None:
    """After weak overflows in turn 1, turn 2 of the same session skips weak entirely."""
    backend = _OverflowWeakBackend()
    profile = ComponentChainProfile(
        request_processors=[_PickWeakProcessor()],
        backend=backend,
        fallback_target_on_evict="strong",
    )
    request = _request(msg="hi")

    # Turn 1: weak overflows → fallback to strong.
    await profile.run(ProfileInput(request))
    assert backend.calls == ["weak", "strong"]

    backend.calls.clear()

    # Turn 2: same session → weak is pre-empted, strong called directly.
    await profile.run(ProfileInput(request))
    assert backend.calls == ["strong"]


async def test_session_eviction_records_fallback_target_after_pool_exhaustion() -> None:
    """Both targets are written to session cache when pool is exhausted."""
    from switchyard.lib.session_key import session_key_from_body

    backend = _AlwaysOverflowBackend()
    profile = ComponentChainProfile(
        request_processors=[_PickWeakProcessor()],
        backend=backend,
        fallback_target_on_evict="strong",
    )
    request = _request(msg="session-pool")

    with pytest.raises(SwitchyardContextPoolExhaustedError):
        await profile.run(ProfileInput(request))

    assert profile._session_evictions is not None
    session_key = session_key_from_body(request.body)
    recorded = profile._session_evictions.get(session_key) or frozenset()
    assert "weak" in recorded
    assert "strong" in recorded


async def test_session_eviction_does_not_bleed_between_sessions() -> None:
    """Eviction in one session does not affect a different session."""
    backend = _OverflowWeakBackend()
    profile = ComponentChainProfile(
        request_processors=[_PickWeakProcessor()],
        backend=backend,
        fallback_target_on_evict="strong",
    )

    # Session A: weak overflows → evicted for session A.
    await profile.run(ProfileInput(_request(msg="session-a")))

    backend.calls.clear()

    # Session B (different first message → different session key): weak is tried.
    await profile.run(ProfileInput(_request(msg="session-b")))
    assert backend.calls == ["weak", "strong"]


def test_profile_config_decorator_accepts_existing_dataclass() -> None:
    @profile_config("unit-python-prebuilt-dataclass", register=False)
    @dataclass(frozen=True)
    class ExistingDataclassConfig:
        content: str

        def build(self) -> _StaticProfile:
            return _StaticProfile(self.content)

    config = ExistingDataclassConfig(content="prebuilt")

    assert is_dataclass(ExistingDataclassConfig)
    assert profile_config_type(config) == "unit-python-prebuilt-dataclass"
    assert build_profile(config).run is not None


def test_profile_config_decorator_registers_and_rejects_duplicates() -> None:
    profile_type = f"unit-python-registered-{uuid4().hex}"

    @profile_config(profile_type, register=True)
    class RegisteredProfileConfig:
        content: str = "registered"

        def build(self) -> _StaticProfile:
            return _StaticProfile(self.content)

    assert lookup_profile_config(profile_type) is RegisteredProfileConfig
    assert profile_type in registered_profile_config_types()

    with pytest.raises(ProfileConfigError, match="already registered"):

        @profile_config(profile_type, register=True)
        class DuplicateProfileConfig:
            content: str = "duplicate"

            def build(self) -> _StaticProfile:
                return _StaticProfile(self.content)


def test_profile_config_decorator_rejects_invalid_declarations() -> None:
    with pytest.raises(ProfileConfigError, match="must not be empty"):
        profile_config(" ")

    with pytest.raises(ProfileConfigError, match="must define build"):

        @profile_config("unit-python-missing-build", register=False)
        class MissingBuildConfig:
            content: str = "missing"


def test_register_profile_config_validates_direct_callers() -> None:
    @dataclass(frozen=True)
    class MissingBuildConfig:
        value: str = "missing"

    with pytest.raises(ProfileConfigError, match="must define build"):
        register_profile_config(
            f"unit-python-direct-invalid-{uuid4().hex}",
            cast(object, MissingBuildConfig),  # type: ignore[arg-type]
        )


def test_table_helpers_stay_out_of_top_level_namespace() -> None:
    import switchyard
    import switchyard.lib.profiles as profiles

    assert "lookup_profile_config" not in switchyard.__all__
    assert "register_profile_config" not in switchyard.__all__
    assert "registered_profile_config_types" not in switchyard.__all__
    assert "lookup_profile_config" not in profiles.__all__
    assert "register_profile_config" not in profiles.__all__


def test_build_profile_rejects_configs_that_do_not_return_profiles() -> None:
    @profile_config("unit-python-bad-build", register=False)
    class BadBuildConfig:
        def build(self) -> object:
            return object()

    with pytest.raises(ProfileConfigError, match="did not return a Profile"):
        build_profile(BadBuildConfig())


def test_build_profile_rejects_run_only_profiles() -> None:
    class RunOnlyProfile:
        async def run(self, input: ProfileInput) -> ChatResponse:
            return ChatResponse.openai_completion({
                "id": "python-profile-run-only",
                "object": "chat.completion",
                "model": input.request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            })

    @profile_config("unit-python-run-only", register=False)
    class RunOnlyConfig:
        def build(self) -> RunOnlyProfile:
            return RunOnlyProfile()

    with pytest.raises(ProfileConfigError, match="run/process/rprocess"):
        build_profile(RunOnlyConfig())


def test_build_profile_rejects_sync_run_profiles() -> None:
    class SyncRunProfile:
        async def process(self, input: ProfileInput) -> ProfileInput:
            return input

        async def rprocess(
            self,
            processed: ProfileInput,
            response: ChatResponse,
        ) -> ChatResponse:
            return response

        def run(self, input: ProfileInput) -> ChatResponse:
            return ChatResponse.openai_completion({"model": input.request.model})

    @profile_config("unit-python-sync-run", register=False)
    class SyncRunConfig:
        def build(self) -> SyncRunProfile:
            return SyncRunProfile()

    with pytest.raises(ProfileConfigError, match="run\\(\\) must be async"):
        build_profile(SyncRunConfig())


def test_build_profile_rejects_sync_process_profiles() -> None:
    class SyncProcessProfile:
        def process(self, input: ProfileInput) -> ProfileInput:
            return input

        async def rprocess(
            self,
            processed: ProfileInput,
            response: ChatResponse,
        ) -> ChatResponse:
            return response

        async def run(self, input: ProfileInput) -> ChatResponse:
            return ChatResponse.openai_completion({"model": input.request.model})

    @profile_config("unit-python-sync-process", register=False)
    class SyncProcessConfig:
        def build(self) -> SyncProcessProfile:
            return SyncProcessProfile()

    with pytest.raises(ProfileConfigError, match="process\\(\\) must be async"):
        build_profile(SyncProcessConfig())


def test_build_profile_rejects_sync_rprocess_profiles() -> None:
    class SyncRprocessProfile:
        async def process(self, input: ProfileInput) -> ProfileInput:
            return input

        def rprocess(
            self,
            processed: ProfileInput,
            response: ChatResponse,
        ) -> ChatResponse:
            return response

        async def run(self, input: ProfileInput) -> ChatResponse:
            return ChatResponse.openai_completion({"model": input.request.model})

    @profile_config("unit-python-sync-rprocess", register=False)
    class SyncRprocessConfig:
        def build(self) -> SyncRprocessProfile:
            return SyncRprocessProfile()

    with pytest.raises(ProfileConfigError, match="rprocess\\(\\) must be async"):
        build_profile(SyncRprocessConfig())
