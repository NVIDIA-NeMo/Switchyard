# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the no-op profile."""

from switchyard.lib.profiles import NoopProfile, NoopProfileConfig, ProfileSwitchyard
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import ChatRequest, ChatResponseType, response_type_matches
from switchyard_rust.profiles import ProfileInput


def _openai_request(*, stream: bool = False) -> ChatRequest:
    return ChatRequest.openai_chat(
        {"model": "noop", "messages": [{"role": "user", "content": "ping"}], "stream": stream}
    )


def _anthropic_request() -> ChatRequest:
    return ChatRequest.anthropic(
        {"model": "claude-3", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 10}
    )


class TestNoOpProfileNonStreaming:
    async def test_returns_completion_response(self) -> None:
        profile = NoopProfileConfig().build()
        resp = await profile.run(ProfileInput(_openai_request(stream=False)))
        assert response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)

    async def test_completion_has_pong_content(self) -> None:
        profile = NoopProfileConfig().build()
        resp = await profile.run(ProfileInput(_openai_request(stream=False)))
        assert response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert resp.body["choices"][0]["message"]["content"] == "pong"

    async def test_completion_is_valid_chat_completion(self) -> None:
        profile = NoopProfileConfig().build()
        resp = await profile.run(ProfileInput(_openai_request(stream=False)))
        assert response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert resp.body["object"] == "chat.completion"
        assert resp.body["choices"][0]["finish_reason"] == "stop"

    async def test_accepts_anthropic_request(self) -> None:
        """Translates non-OpenAI request formats before generating the response."""
        profile = NoopProfileConfig().build()
        resp = await profile.run(ProfileInput(_anthropic_request()))
        assert response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert resp.body["choices"][0]["message"]["content"] == "pong"


class TestNoOpProfileStreaming:
    async def test_returns_streaming_response(self) -> None:
        profile = NoopProfileConfig().build()
        resp = await profile.run(ProfileInput(_openai_request(stream=True)))
        assert response_type_matches(resp, ChatResponseType.OPENAI_STREAM)

    async def test_stream_yields_chunks(self) -> None:
        profile = NoopProfileConfig().build()
        resp = await profile.run(ProfileInput(_openai_request(stream=True)))
        chunks = [chunk async for chunk in resp.stream]
        assert len(chunks) >= 1

    async def test_stream_contains_pong(self) -> None:
        profile = NoopProfileConfig().build()
        resp = await profile.run(ProfileInput(_openai_request(stream=True)))
        contents: list[str] = []
        async for chunk in resp.stream:
            for choice in chunk.choices:
                if choice.delta.content:
                    contents.append(choice.delta.content)
        assert "".join(contents) == "pong"


class TestNoOpRecipe:
    def test_noop_recipe_returns_profile_backed_switchyard_adapter(self) -> None:
        sy = ProfileSwitchyard(NoopProfileConfig().build())
        assert isinstance(sy, ProfileSwitchyard)

    async def test_noop_recipe_end_to_end(self) -> None:
        sy = ProfileSwitchyard(NoopProfileConfig().build())
        # ``sy.call()`` preserves the serving contract while no-op construction
        # now goes through the Profile abstraction instead of a backend pipeline.
        resp = await sy.call(_openai_request(stream=False))
        assert resp["choices"][0]["message"]["content"] == "pong"


class TestNoOpProfile:
    def test_config_builds_noop_profile(self) -> None:
        profile = NoopProfileConfig().build()
        assert isinstance(profile, NoopProfile)

    async def test_profile_run_returns_completion_response(self) -> None:
        profile = NoopProfileConfig().build()

        resp = await profile.run(ProfileInput(_openai_request(stream=False)))

        assert response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert resp.body["choices"][0]["message"]["content"] == "pong"

    async def test_profile_process_and_rprocess_are_explicit_hooks(self) -> None:
        profile = NoopProfileConfig().build()
        profile_input = ProfileInput(_openai_request(stream=False))
        response = await profile.run(profile_input)

        processed = await profile.process(profile_input)
        processed_response = await profile.rprocess(processed, response)

        assert processed.request.model == "noop"
        assert processed_response is response

    async def test_noop_call_reports_routing_stats(self) -> None:
        """The no-op profile self-reports its call and model to the stats accumulator."""
        stats = StatsAccumulator()
        profile = NoopProfileConfig().build().with_runtime_components(
            stats_accumulator=stats
        )

        await ProfileSwitchyard(profile).call(_openai_request(stream=False))

        snap = stats.snapshot_sync()
        assert snap["total_requests"] == 1
        assert "noop" in snap["models"]
        assert snap["models"]["noop"]["calls"] == 1
        assert "<unknown>" not in snap["models"]
