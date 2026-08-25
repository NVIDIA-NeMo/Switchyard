# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from litellm.proxy.types_utils.utils import get_instance_fn
from switchyard_litellm import RandomRoutingPlugin, StageRoutingPlugin
from switchyard_litellm.configuration import (
    CONFIG_ENV,
    load_routing_plugin_from_environment,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = PACKAGE_ROOT / "deployment" / "profiles"
PLUGIN_PATH = "switchyard_litellm.configuration.configured_plugin.ROUTING_PLUGIN"


def _expected_models() -> list[dict[str, object]]:
    return [
        {
            "model_name": "switchyard",
            "litellm_params": {
                "model": "openrouter/openai/gpt-5.6-sol",
                "api_key": "os.environ/OPENROUTER_API_KEY",
            },
        },
        {
            "model_name": "switchyard",
            "litellm_params": {
                "model": "openrouter/openai/gpt-5.6-terra",
                "api_key": "os.environ/OPENROUTER_API_KEY",
            },
        },
    ]


def test_environment_loader_requires_a_path() -> None:
    with pytest.raises(ValueError, match=CONFIG_ENV):
        load_routing_plugin_from_environment({})


def test_environment_loader_uses_the_configured_path(tmp_path: Path) -> None:
    config_path = tmp_path / "switchyard.toml"
    config_path.write_text('algorithm = "random"\nseed = 9\n')

    plugin = load_routing_plugin_from_environment({CONFIG_ENV: str(config_path)})

    assert isinstance(plugin, RandomRoutingPlugin)


def test_checked_in_profiles_separate_models_from_routing_policy() -> None:
    expected_policies = {
        "stage": {
            "algorithm": "stage",
            "picker": "efficient_first",
            "confidence_threshold": 0.5,
            "recent_window": 3,
            "only_on_wrong_signal_escalation": True,
            "escalation_note": "The efficient tier failed; continue from its work.",
            "deescalation_note": "The capable tier completed the recovery.",
            "capable_system_prompt": "Handle this request as the capable tier.",
            "efficient_system_prompt": "Handle this request as the efficient tier.",
        },
        "random": {"algorithm": "random", "seed": 6},
    }

    for profile, policy in expected_policies.items():
        profile_root = PROFILE_ROOT / profile
        assert tomllib.loads((profile_root / "switchyard.toml").read_text()) == policy
        assert yaml.safe_load((profile_root / "litellm.yaml").read_text()) == {
            "model_list": _expected_models(),
            "router_settings": {"plugins": [PLUGIN_PATH]},
            "litellm_settings": {"callbacks": [PLUGIN_PATH]},
        }


@pytest.mark.parametrize(
    ("profile", "plugin_type"),
    [("stage", StageRoutingPlugin), ("random", RandomRoutingPlugin)],
)
def test_litellm_resolves_the_profile_configured_plugin(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    plugin_type: type[StageRoutingPlugin] | type[RandomRoutingPlugin],
) -> None:
    profile_root = PROFILE_ROOT / profile
    monkeypatch.setenv(CONFIG_ENV, str(profile_root / "switchyard.toml"))
    sys.modules.pop("switchyard_litellm.configuration.configured_plugin", None)

    plugin = get_instance_fn(
        value=PLUGIN_PATH,
        config_file_path=str(profile_root / "litellm.yaml"),
    )

    assert isinstance(plugin, plugin_type)
