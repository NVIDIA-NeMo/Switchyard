# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
ADAPTER = REPO / "benchmark" / "prefill_probe_terminus_2.py"


def _load_adapter_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("switchyard_prefill_probe_terminus_2", ADAPTER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_adapter_preserves_prompt_and_injects_exact_input_for_the_full_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_adapter_module()
    observed: list[tuple[str, object, object, dict[str, object]]] = []

    async def fake_run(self, instruction: str, environment: object, context: object) -> None:
        observed.append(
            (instruction, environment, context, copy.deepcopy(self._llm_call_kwargs))
        )
        observed.append(
            (instruction, environment, context, copy.deepcopy(self._llm_call_kwargs))
        )

    monkeypatch.setattr(module.Terminus2, "run", fake_run)
    agent = object.__new__(module.PrefillProbeTerminus2)
    original_kwargs = {
        "extra_body": {"existing": {"nested": True}},
        "timeout": 30,
    }
    agent._llm_call_kwargs = original_kwargs
    environment = object()
    context = object()
    instruction = "  exact raw instruction\nwith original whitespace  "

    await agent.run(instruction, environment, context)

    assert len(observed) == 2
    for forwarded_instruction, forwarded_environment, forwarded_context, call_kwargs in observed:
        assert forwarded_instruction == instruction
        assert forwarded_environment is environment
        assert forwarded_context is context
        assert call_kwargs == {
            "extra_body": {
                "existing": {"nested": True},
                module.PREFILL_PROBE_INPUT_FIELD: instruction,
            },
            "timeout": 30,
        }
    assert original_kwargs == {
        "extra_body": {"existing": {"nested": True}},
        "timeout": 30,
    }


async def test_adapter_initializes_extra_body_and_refreshes_input_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_adapter_module()
    observed: list[dict[str, object]] = []

    async def fake_run(self, instruction: str, environment: object, context: object) -> None:
        observed.append(copy.deepcopy(self._llm_call_kwargs))

    monkeypatch.setattr(module.Terminus2, "run", fake_run)
    agent = object.__new__(module.PrefillProbeTerminus2)
    agent._llm_call_kwargs = {"timeout": 30}

    await agent.run("first instruction", object(), object())
    await agent.run("second instruction", object(), object())

    assert observed == [
        {
            "timeout": 30,
            "extra_body": {module.PREFILL_PROBE_INPUT_FIELD: "first instruction"},
        },
        {
            "timeout": 30,
            "extra_body": {module.PREFILL_PROBE_INPUT_FIELD: "second instruction"},
        },
    ]


async def test_adapter_rejects_non_mapping_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter_module()

    async def fake_run(self, instruction: str, environment: object, context: object) -> None:
        raise AssertionError("stock Terminus should not run with invalid extra_body")

    monkeypatch.setattr(module.Terminus2, "run", fake_run)
    agent = object.__new__(module.PrefillProbeTerminus2)
    agent._llm_call_kwargs = {"extra_body": "invalid"}

    with pytest.raises(TypeError, match="extra_body must be a dictionary"):
        await agent.run("instruction", object(), object())
