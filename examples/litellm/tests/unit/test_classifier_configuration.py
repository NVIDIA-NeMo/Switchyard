# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from switchyard_litellm import ClassifierRoutingPlugin
from switchyard_litellm.configuration import load_routing_plugin
from switchyard_litellm.plugins.judge import LiteLLMJudge

VALID = (
    'algorithm = "llm_classifier"\n'
    'judge_model = "switchyard-judge"\n'
    "base_threshold = 0.5\n"
    "threshold_step = 0.1\n"
    "max_output_tokens = 512\n"
    'response_format_type = "json_schema"\n'
    'judge_max_tokens_field = "max_tokens"\n'
    "judge_timeout_seconds = 30\n"
    'prompt = "Forecast whether the efficient model solves the task."\n'
)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "switchyard.toml"
    path.write_text(text)
    return path


def test_loads_classifier_plugin_with_a_litellm_judge(tmp_path: Path) -> None:
    plugin = load_routing_plugin(write(tmp_path, VALID))

    assert isinstance(plugin, ClassifierRoutingPlugin)
    assert isinstance(plugin._judge, LiteLLMJudge)
    assert plugin._judge.model == "switchyard-judge"
    assert plugin._judge._max_tokens_field == "max_tokens"
    assert plugin._judge._completion_params == {"timeout": 30.0}
    assert plugin._base_threshold == 0.5
    assert plugin._threshold_step == 0.1
    assert plugin._max_output_tokens == 512


def test_minimal_classifier_config_uses_defaults(tmp_path: Path) -> None:
    plugin = load_routing_plugin(
        write(tmp_path, 'algorithm = "llm_classifier"\njudge_model = "j"\nbase_threshold = 0.6\n')
    )

    assert isinstance(plugin, ClassifierRoutingPlugin)
    assert plugin._threshold_step == 0.0
    assert plugin._max_output_tokens == 4096
    assert plugin._response_format_type == "json_schema"
    assert plugin._judge._max_tokens_field == "max_completion_tokens"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            'algorithm = "llm_classifier"\nbase_threshold = 0.5\n',
            "judge_model must be a nonempty string",
        ),
        ('algorithm = "llm_classifier"\njudge_model = "j"\n', "base_threshold is required"),
        (
            'algorithm = "llm_classifier"\njudge_model = "j"\nbase_threshold = 1.5\n',
            "between 0 and 1",
        ),
        (
            'algorithm = "llm_classifier"\njudge_model = "j"\nbase_threshold = 0.9\nthreshold_step = 0.1\n',
            "must not exceed 1",
        ),
        (
            'algorithm = "llm_classifier"\njudge_model = "j"\nbase_threshold = 0.5\nmax_output_tokens = 0\n',
            "at least 1",
        ),
        (
            'algorithm = "llm_classifier"\njudge_model = "j"\nbase_threshold = 0.5\nresponse_format_type = "yaml"\n',
            "response_format_type",
        ),
        (
            'algorithm = "llm_classifier"\njudge_model = "j"\nbase_threshold = 0.5\njudge_max_tokens_field = "max_len"\n',
            "judge_max_tokens_field",
        ),
        (
            'algorithm = "llm_classifier"\njudge_model = "j"\nbase_threshold = 0.5\njudge_timeout_seconds = 0\n',
            "greater than 0",
        ),
        (
            'algorithm = "llm_classifier"\njudge_model = "j"\nbase_threshold = 0.5\npicker = "efficient_first"\n',
            "unknown keys: picker",
        ),
    ],
)
def test_rejects_invalid_classifier_configs(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_routing_plugin(write(tmp_path, text))
