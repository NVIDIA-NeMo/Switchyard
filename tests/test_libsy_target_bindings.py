# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from switchyard import libsy


class _Client:
    async def call(
        self,
        context: libsy.protocol.Context,
        request: libsy.protocol.Request,
        decision: object,
    ) -> libsy.protocol.Response:
        return libsy.protocol.Response(libsy.protocol.AggLlmResponse())


def test_libsy_targets_hold_assignable_clients() -> None:
    first_client = _Client()
    second_client = _Client()
    target = libsy.LlmTarget("strong", llm_client=first_client)
    targets = libsy.LlmTargetSet([target])

    assert target.semantic_name == "strong"
    assert target.llm_client is first_client
    assert targets.targets == [target]
    assert targets.get_target("strong") is target
    assert len(targets) == 1

    target.llm_client = second_client
    assert target.llm_client is second_client
    target.llm_client = None
    assert target.llm_client is None

    with pytest.raises(KeyError, match="missing"):
        targets.get_target("missing")
    with pytest.raises(TypeError, match="must define call"):
        libsy.LlmTarget("invalid", llm_client=object())
