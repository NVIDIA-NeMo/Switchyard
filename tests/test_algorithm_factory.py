# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the Python algorithm factory."""

from switchyard.libsy import AlgorithmFactory, ConfiguredAlgorithm, Step


async def test_factory_builds_a_usable_python_handle() -> None:
    """Configured handles run without a separate model map."""
    routes = AlgorithmFactory.from_toml("""
schema_version = 1
[llm_clients.host]
format = "openai_chat"
base_url = "https://host.invalid/v1"
[targets]
fast = { id = "gateway/fast", llm_client = "host" }
[routes.main]
id = "switchyard/main"
type = "passthrough"
target = "fast"
""").build()
    route = routes["switchyard/main"]
    assert isinstance(route, ConfiguredAlgorithm)
    steps = [
        step
        async for step in route.run_stream(
            {
                "model": "switchyard/main",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            }
        )
    ]
    assert len(steps) == 1
    assert isinstance(steps[0], Step.Done)
    assert steps[0].outcome.selected_model_ids == ["gateway/fast"]
