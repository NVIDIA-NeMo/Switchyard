# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
from litellm.types.router import RoutingContext
from switchyard_litellm.configuration import load_routing_plugin


def routing_context(
    messages: list[dict[str, Any]],
    candidates: list[str],
) -> RoutingContext:
    """Build the LiteLLM boundary object passed to configured plugins."""
    return RoutingContext(
        raw_messages=messages,
        structured_messages=messages,
        candidate_models=candidates,
        metadata={},
        signals={},
    )


async def test_loads_stage_picker_and_threshold(tmp_path: Path) -> None:
    config_path = tmp_path / "switchyard.toml"
    config_path.write_text(
        'algorithm = "stage"\n'
        'picker = "capable_first"\n'
        "confidence_threshold = 0.5\n"
        "recent_window = 3\n"
        "only_on_wrong_signal_escalation = true\n"
    )
    plugin = load_routing_plugin(config_path)
    context = routing_context(
        [{"role": "user", "content": "Say hello."}],
        ["provider/capable", "provider/efficient"],
    )

    await plugin.run(context)

    assert context.candidate_models == ["provider/capable"]


async def test_loads_stage_prompt_and_handoff_rewrites(tmp_path: Path) -> None:
    config_path = tmp_path / "switchyard.toml"
    config_path.write_text(
        'algorithm = "stage"\n'
        'picker = "capable_first"\n'
        "confidence_threshold = 0.5\n"
        'escalation_note = "The efficient tier failed."\n'
        'deescalation_note = "The capable tier recovered."\n'
        'capable_system_prompt = "Use the capable tier."\n'
        'efficient_system_prompt = "Use the efficient tier."\n'
    )
    plugin = load_routing_plugin(config_path)
    context = routing_context(
        [{"role": "user", "content": "Say hello."}],
        ["provider/capable", "provider/efficient"],
    )

    await plugin.run(context)
    rewritten = await plugin.async_pre_call_deployment_hook(
        {
            "messages": [{"role": "user", "content": "Say hello."}],
            "metadata": {"routing_plugin_signals": context.signals},
        },
        None,
    )

    assert rewritten is not None
    assert rewritten["messages"][0] == {
        "role": "system",
        "content": "Use the capable tier.",
    }

    settled_messages = [
        {"role": "user", "content": "Implement the change."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "write_1",
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": '{"file_path":"x.py","content":"pass"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "write_1", "content": "Wrote x.py"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "test_1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": '{"command":"pytest"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "test_1", "content": "5 passed in 0.12s"},
    ]
    settled_context = routing_context(
        settled_messages,
        ["provider/capable", "provider/efficient"],
    )

    await plugin.run(settled_context)
    settled_rewrite = await plugin.async_pre_call_deployment_hook(
        {
            "messages": settled_messages,
            "metadata": {"routing_plugin_signals": settled_context.signals},
        },
        None,
    )

    assert settled_context.candidate_models == ["provider/efficient"]
    assert settled_rewrite is not None
    assert settled_rewrite["messages"][0] == {
        "role": "system",
        "content": "Use the efficient tier.",
    }
    assert settled_rewrite["messages"][-1] == {
        "role": "user",
        "content": "The capable tier recovered.",
    }


async def test_loads_random_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "switchyard.toml"
    config_path.write_text('algorithm = "random"\nseed = 6\nweights = [0.0, 1.0]\n')
    plugin = load_routing_plugin(config_path)
    context = routing_context(
        [{"role": "user", "content": "Say hello."}],
        ["provider/alpha", "provider/beta"],
    )

    await plugin.run(context)

    assert context.candidate_models == ["provider/beta"]


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("", "algorithm"),
        ("algorithm = 7\n", "algorithm"),
        ('algorithm = "unknown"\n', "unsupported algorithm"),
        ('algorithm = "stage"\nconfidence_threshold = 0.5\n', "picker"),
        (
            'algorithm = "stage"\npicker = "fast"\nconfidence_threshold = 0.5\n',
            "picker",
        ),
        ('algorithm = "stage"\npicker = "efficient_first"\n', "confidence_threshold"),
        (
            'algorithm = "stage"\npicker = "efficient_first"\nconfidence_threshold = true\n',
            "confidence_threshold",
        ),
        (
            'algorithm = "stage"\npicker = "efficient_first"\nconfidence_threshold = nan\n',
            "finite",
        ),
        (
            'algorithm = "stage"\npicker = "efficient_first"\nconfidence_threshold = -0.1\n',
            "between 0 and 1",
        ),
        (
            'algorithm = "stage"\npicker = "efficient_first"\nconfidence_threshold = 1.1\n',
            "between 0 and 1",
        ),
        (
            'algorithm = "stage"\npicker = "efficient_first"\n'
            "confidence_threshold = 0.5\nrecent_window = true\n",
            "recent_window",
        ),
        (
            'algorithm = "stage"\npicker = "efficient_first"\n'
            "confidence_threshold = 0.5\nrecent_window = -1\n",
            "recent_window",
        ),
        (
            'algorithm = "stage"\npicker = "efficient_first"\n'
            "confidence_threshold = 0.5\nonly_on_wrong_signal_escalation = 1\n",
            "only_on_wrong_signal_escalation",
        ),
        (
            'algorithm = "stage"\npicker = "efficient_first"\n'
            'confidence_threshold = 0.5\ncapable_system_prompt = ""\n',
            "capable_system_prompt",
        ),
        (
            'algorithm = "stage"\npicker = "efficient_first"\n'
            'confidence_threshold = 0.5\ndeescalation_note = "Recovered."\n',
            "requires escalation_note",
        ),
        (
            'algorithm = "stage"\npicker = "efficient_first"\n'
            "confidence_threshold = 0.5\nextra = true\n",
            "unknown keys",
        ),
        ('algorithm = "random"\nextra = true\n', "unknown keys"),
        ('algorithm = "random"\nseed = true\n', "seed"),
        ('algorithm = "random"\nseed = -1\n', "seed"),
        ('algorithm = "random"\nseed = 18446744073709551616\n', "seed"),
        ('algorithm = "random"\nweights = 1.0\n', "weights"),
        ('algorithm = "random"\nweights = []\n', "nonempty"),
        ('algorithm = "random"\nweights = [true]\n', "weights"),
        ('algorithm = "random"\nweights = [-1.0, 2.0]\n', "nonnegative"),
        ('algorithm = "random"\nweights = [nan, 1.0]\n', "finite"),
        ('algorithm = "random"\nweights = [0.0, 0.0]\n', "positive"),
    ],
)
def test_rejects_invalid_static_configuration(
    tmp_path: Path,
    contents: str,
    match: str,
) -> None:
    config_path = tmp_path / "switchyard.toml"
    config_path.write_text(contents)

    with pytest.raises(ValueError, match=match) as error:
        load_routing_plugin(config_path)

    assert str(config_path) in str(error.value)


def test_wraps_missing_file_with_its_path(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.toml"

    with pytest.raises(ValueError, match="Could not load") as error:
        load_routing_plugin(config_path)

    assert str(config_path) in str(error.value)


def test_wraps_malformed_toml_with_its_path(tmp_path: Path) -> None:
    config_path = tmp_path / "switchyard.toml"
    config_path.write_text('algorithm = "random"\nweights = [\n')

    with pytest.raises(ValueError, match="Could not load") as error:
        load_routing_plugin(config_path)

    assert str(config_path) in str(error.value)
