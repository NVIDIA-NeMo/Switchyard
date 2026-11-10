# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public binding conversion preserves application exceptions."""

from typing import Any

import pytest

from switchyard_rust import libsy


def request_body() -> dict[str, Any]:
    return {
        "model": "fixture",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    }


@pytest.mark.parametrize("method", ["model_dump", "to_dict"])
@pytest.mark.parametrize("base", [object, dict])
def test_conversion_getter_error_propagates(method: str, base: type) -> None:
    error = RuntimeError("conversion getter failed")

    def fail(self: Any) -> Any:
        raise error

    request_type = type("Request", (base,), {method: property(fail)})
    request = request_type(request_body()) if base is dict else request_type()
    with pytest.raises(RuntimeError) as caught:
        libsy.noop().run_stream(request, {"any": ["fixture"]})
    assert caught.value is error


@pytest.mark.parametrize("method", ["model_dump", "to_dict"])
def test_conversion_method_error_propagates(method: str) -> None:
    error = RuntimeError("conversion method failed")

    def fail(self: Any, **kwargs: Any) -> Any:
        raise error

    request = type("Request", (), {method: fail})()
    with pytest.raises(RuntimeError) as caught:
        libsy.noop().run_stream(request, {"any": ["fixture"]})
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("method", [None, "model_dump", "to_dict"])
async def test_supported_request_conversion(method: str | None) -> None:
    def convert(self: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs == ({"mode": "json", "exclude_none": True} if method == "model_dump" else {})
        return request_body()

    request = request_body() if method is None else type("Request", (), {method: convert})()
    steps = [step async for step in libsy.noop().run_stream(request, {"any": ["fixture"]})]
    assert len(steps) == 1
    assert isinstance(steps[0], libsy.Step.Done)
    normalized = steps[0].outcome.request
    assert normalized["model"] == "fixture"
    assert normalized["messages"] == request_body()["messages"]
