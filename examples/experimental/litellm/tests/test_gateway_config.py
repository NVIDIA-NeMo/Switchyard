# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_gateway_uses_openrouter_models_and_credential() -> None:
    config = yaml.safe_load((PACKAGE_ROOT / "litellm-config.yaml").read_text())
    models = {
        item["model_name"]: item["litellm_params"] for item in config["model_list"]
    }

    assert models == {
        "strong": {
            "model": "openrouter/openai/gpt-5.6-sol",
            "api_key": "os.environ/OPENROUTER_API_KEY",
        },
        "fast": {
            "model": "openrouter/moonshotai/kimi-k3",
            "api_key": "os.environ/OPENROUTER_API_KEY",
        },
    }
    assert config["litellm_settings"] == {"drop_params": True}

    compose = yaml.safe_load((PACKAGE_ROOT / "compose.yaml").read_text())
    assert compose["services"]["litellm"]["environment"] == {
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
    }
    assert (PACKAGE_ROOT / ".env.example").read_text() == "OPENROUTER_API_KEY=\n"
