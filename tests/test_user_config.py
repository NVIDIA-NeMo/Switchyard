# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import stat
import sys

import pytest

from switchyard.cli.config.user_config import (
    DEFAULT_OPENROUTER_BASE_URL,
    PRIMARY_TIER,
    LaunchConfig,
    LaunchCredentials,
    LaunchRouteConfig,
    LaunchTierEndpointConfig,
    ProviderConfig,
    SkillDistillationConfig,
    UserConfig,
    UserConfigError,
    UserCredentials,
    build_redacted_snapshot,
    load_user_config,
    load_user_credentials,
    resolve_provider_connectivity,
    save_user_config,
    save_user_credentials,
)
from switchyard.cli.output import format_config_snapshot
from switchyard.cli.status import StatusRequest, render_status


@pytest.fixture(autouse=True)
def _clear_provider_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the developer's shell credentials from influencing any test in this file."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_user_config_round_trips_and_credentials_are_private(tmp_path):
    bundle = {
        "routes": {
            "example/model": {
                "type": "model",
                "target": {
                    "model": "example/model",
                    "api_key": "sk-test",
                    "base_url": "https://example.invalid/v1",
                },
            },
        },
    }
    save_user_config(
        UserConfig(
            providers={
                "nvidia": ProviderConfig(base_url="https://example.test/v1"),
            },
            launch={
                "claude": LaunchConfig(
                    model="aws/anthropic/bedrock-claude-opus-4-7",
                    route=LaunchRouteConfig(
                        type="single",
                        model="aws/anthropic/bedrock-claude-opus-4-7",
                        endpoints={
                            PRIMARY_TIER: LaunchTierEndpointConfig(
                                base_url="https://strong.example/v1",
                            ),
                        },
                    ),
                ),
                "codex": LaunchConfig(model="openai/openai/openai/gpt-5.5"),
            },
            routing_profiles=bundle,
        ),
        config_dir=tmp_path,
    )
    save_user_credentials(
        UserCredentials(
            api_keys={"nvidia": "nvapi-secret"},
            launch={
                "claude": LaunchCredentials(
                    api_keys={PRIMARY_TIER: "strong-secret"},
                ),
            },
        ),
        config_dir=tmp_path,
    )

    config = load_user_config(tmp_path)
    credentials = load_user_credentials(tmp_path)

    assert config.provider("nvidia").base_url == "https://example.test/v1"
    assert config.launch_target("claude").model == "aws/anthropic/bedrock-claude-opus-4-7"
    route = config.launch_target("claude").effective_route()
    assert route.type == "single"
    assert route.endpoint(PRIMARY_TIER).base_url == "https://strong.example/v1"
    assert config.launch_target("codex").model == "openai/openai/openai/gpt-5.5"
    assert config.routing_profiles == bundle
    assert credentials.api_key("nvidia") == "nvapi-secret"
    assert credentials.launch_target("claude").api_key(PRIMARY_TIER) == "strong-secret"

    mode = stat.S_IMODE((tmp_path / "credentials.json").stat().st_mode)
    assert mode == 0o600


def test_top_level_routing_profiles_round_trips(tmp_path):
    bundle = {"routes": {"example/model": {"type": "model"}}}
    save_user_config(UserConfig(routing_profiles=bundle), config_dir=tmp_path)
    loaded = load_user_config(tmp_path)
    assert loaded.routing_profiles == bundle


def test_top_level_routing_profiles_clears_when_none(tmp_path):
    save_user_config(
        UserConfig(routing_profiles={"routes": {"a": {"type": "model"}}}),
        config_dir=tmp_path,
    )
    save_user_config(UserConfig(routing_profiles=None), config_dir=tmp_path)
    assert load_user_config(tmp_path).routing_profiles is None


def test_default_skill_distillation_config_is_not_persisted(tmp_path):
    save_user_config(UserConfig(), config_dir=tmp_path)

    raw = json.loads((tmp_path / "config.json").read_text())
    assert "skill_distillation" not in raw
    assert load_user_config(tmp_path).skill_distillation == SkillDistillationConfig()


def test_skill_distillation_config_round_trips_and_surfaces(
    monkeypatch,
    tmp_path,
):
    skill_config = SkillDistillationConfig(
        namespace="tooluniverse-trialqa",
    )
    save_user_config(
        UserConfig(skill_distillation=skill_config),
        config_dir=tmp_path,
    )

    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["skill_distillation"] == {
        "namespace": "tooluniverse-trialqa",
    }
    assert load_user_config(tmp_path).skill_distillation == skill_config

    snapshot = build_redacted_snapshot(tmp_path)
    assert snapshot["skill_distillation"] == raw["skill_distillation"]
    rendered = format_config_snapshot(snapshot)
    assert "Skill distillation" in rendered
    assert "namespace: tooluniverse-trialqa" in rendered

    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
    status = render_status(StatusRequest())
    assert "skill distillation: configured; namespace: tooluniverse-trialqa" in status
    assert "trace capture: project-local when implemented" in status


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "namespace is required"),
        ({"namespace": "ok", "enabled": False}, "supports only namespace"),
        ({"namespace": "ok", "save_sessions": False}, "supports only namespace"),
        ({"namespace": 7}, "must be a non-empty string"),
        ({"namespace": "bad/name"}, "letters, numbers"),
        ({"namespace": " bad"}, "leading or trailing whitespace"),
        ({"namespace": "ok", "session_store": "remote"}, "supports only namespace"),
        ({"namespace": "ok", "trigger": "instant"}, "supports only namespace"),
        ({"namespace": "ok", "mount": "global"}, "supports only namespace"),
        ({"namespace": "ok", "lookback_sessions": 0}, "supports only namespace"),
    ],
)
def test_skill_distillation_config_validates(
    tmp_path,
    body,
    message,
):
    (tmp_path / "config.json").write_text(json.dumps({
        "skill_distillation": body,
    }))

    with pytest.raises(UserConfigError, match=message):
        load_user_config(tmp_path)


def test_configure_skill_only_update_does_not_require_api_key(
    monkeypatch,
    tmp_path,
    capsys,
):
    from switchyard.cli.switchyard_cli import _build_parser, _cmd_configure

    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "switchyard.cli.command_utils.is_interactive_terminal",
        lambda: False,
    )
    monkeypatch.setattr(
        "switchyard.cli.configure_command.is_interactive_terminal",
        lambda: False,
    )

    parser = _build_parser()
    args = parser.parse_args([
        "configure",
        "--skill-distillation", "tooluniverse-trialqa",
    ])

    _cmd_configure(args)

    config = load_user_config(tmp_path)
    assert config.skill_distillation.configured is True
    assert config.skill_distillation.namespace == "tooluniverse-trialqa"
    assert not (tmp_path / "credentials.json").exists()
    assert "Saved Switchyard config" in capsys.readouterr().out


def test_configure_disable_skill_distillation_clears_config(
    monkeypatch,
    tmp_path,
    capsys,
):
    from switchyard.cli.switchyard_cli import _build_parser, _cmd_configure

    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
    save_user_config(
        UserConfig(
            skill_distillation=SkillDistillationConfig(
                namespace="tooluniverse-trialqa",
            ),
        ),
        config_dir=tmp_path,
    )

    parser = _build_parser()
    args = parser.parse_args([
        "configure",
        "--disable-skill-distillation",
    ])

    _cmd_configure(args)

    assert load_user_config(tmp_path).skill_distillation == SkillDistillationConfig()
    raw = json.loads((tmp_path / "config.json").read_text())
    assert "skill_distillation" not in raw
    assert not (tmp_path / "credentials.json").exists()
    assert "Saved Switchyard config" in capsys.readouterr().out


def test_configure_disable_skill_distillation_rejects_conflicting_flags():
    from switchyard.cli.switchyard_cli import _build_parser, _cmd_configure

    parser = _build_parser()
    args = parser.parse_args([
        "configure",
        "--disable-skill-distillation",
        "--skill-distillation",
        "tooluniverse-trialqa",
    ])

    with pytest.raises(UserConfigError, match="cannot be combined"):
        _cmd_configure(args)


def test_configure_invalid_skill_namespace_reports_user_error(
    monkeypatch,
    tmp_path,
):
    from switchyard.cli import switchyard_cli

    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [
        "switchyard",
        "configure",
        "--skill-distillation", "bad/name",
    ])

    with pytest.raises(SystemExit) as excinfo:
        switchyard_cli.main()

    message = str(excinfo.value)
    assert "error: invalid user config" in message
    assert "letters, numbers, dot, underscore, and hyphen" in message


def test_redacted_snapshot_surfaces_only_route_ids(tmp_path):
    """The snapshot exposes route ids but never the full bundle (env-var
    references inside the bundle may resolve to secrets at run time)."""
    save_user_config(
        UserConfig(routing_profiles={"routes": {
            "alpha/model": {"type": "model"},
            "beta/model": {"type": "passthrough"},
        }}),
        config_dir=tmp_path,
    )
    snapshot = build_redacted_snapshot(tmp_path)
    saved = snapshot["routing_profiles"]
    assert isinstance(saved, dict)
    assert saved == {"route_ids": ["alpha/model", "beta/model"]}


def test_resolve_provider_connectivity_precedence(monkeypatch, tmp_path):
    save_user_config(
        UserConfig(
            default_provider="nvidia",
            providers={"nvidia": ProviderConfig(base_url="https://user.test/v1")},
        ),
        config_dir=tmp_path,
    )
    save_user_credentials(
        UserCredentials(api_keys={"nvidia": "user-key"}),
        config_dir=tmp_path,
    )
    secrets = {
        "nvidia": {
            "api_key": "secrets-key",
            "base_url": "https://secrets.test/v1",
        },
    }

    resolved = resolve_provider_connectivity(
        cli_api_key=None,
        cli_base_url=None,
        api_key_env_vars=("NVIDIA_API_KEY",),
        base_url_env_vars=("NVIDIA_BASE_URL",),
        secrets=secrets,
        config_dir=tmp_path,
    )
    assert resolved.api_key == "user-key"
    assert resolved.base_url == "https://user.test/v1"

    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://env.test/v1")
    resolved = resolve_provider_connectivity(
        cli_api_key="cli-key",
        cli_base_url="https://cli.test/v1",
        api_key_env_vars=("NVIDIA_API_KEY",),
        base_url_env_vars=("NVIDIA_BASE_URL",),
        secrets=secrets,
        config_dir=tmp_path,
    )
    assert resolved.api_key == "cli-key"
    assert resolved.base_url == "https://cli.test/v1"


def test_resolve_provider_connectivity_ignores_unmatched_base_url_env(
    monkeypatch,
    tmp_path,
):
    save_user_config(
        UserConfig(
            default_provider="nvidia",
            providers={"nvidia": ProviderConfig(base_url="https://user.test/v1")},
        ),
        config_dir=tmp_path,
    )
    save_user_credentials(
        UserCredentials(api_keys={"nvidia": "user-key"}),
        config_dir=tmp_path,
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    resolved = resolve_provider_connectivity(
        cli_api_key=None,
        cli_base_url=None,
        api_key_env_vars=("NVIDIA_API_KEY", "OPENAI_API_KEY"),
        base_url_env_vars=("NVIDIA_BASE_URL", "OPENAI_BASE_URL"),
        config_dir=tmp_path,
    )

    assert resolved.api_key == "user-key"
    assert resolved.base_url == "https://user.test/v1"


def test_resolve_provider_connectivity_pairs_api_key_and_base_url_env(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/v1")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nvidia.test/v1")

    resolved = resolve_provider_connectivity(
        cli_api_key=None,
        cli_base_url=None,
        api_key_env_vars=("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"),
        base_url_env_vars=("OPENROUTER_BASE_URL", "NVIDIA_BASE_URL", "OPENAI_BASE_URL"),
        config_dir=tmp_path,
    )

    assert resolved.provider == "openrouter"
    assert resolved.api_key == "openrouter-key"  # pragma: allowlist secret  # pragma: allowlist secret
    assert resolved.base_url == "https://openrouter.test/v1"


def test_resolve_provider_connectivity_keeps_nvidia_env_fallback(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nvidia.test/v1")

    resolved = resolve_provider_connectivity(
        cli_api_key=None,
        cli_base_url=None,
        api_key_env_vars=("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"),
        base_url_env_vars=("OPENROUTER_BASE_URL", "NVIDIA_BASE_URL", "OPENAI_BASE_URL"),
        config_dir=tmp_path,
    )

    assert resolved.provider == "nvidia"
    assert resolved.api_key == "nvidia-key"  # pragma: allowlist secret  # pragma: allowlist secret
    assert resolved.base_url == "https://nvidia.test/v1"


def test_resolve_provider_connectivity_uses_selected_env_provider_default_base_url(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/v1")

    resolved = resolve_provider_connectivity(
        cli_api_key=None,
        cli_base_url=None,
        api_key_env_vars=("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"),
        base_url_env_vars=("OPENROUTER_BASE_URL", "NVIDIA_BASE_URL", "OPENAI_BASE_URL"),
        config_dir=tmp_path,
    )

    assert resolved.provider == "nvidia"
    assert resolved.api_key == "nvidia-key"  # pragma: allowlist secret  # pragma: allowlist secret
    assert resolved.base_url == "https://inference-api.nvidia.com/v1"


def test_resolve_provider_connectivity_ignores_unsupported_env_provider(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    resolved = resolve_provider_connectivity(
        cli_api_key=None,
        cli_base_url=None,
        api_key_env_vars=("ANTHROPIC_API_KEY",),
        base_url_env_vars=("OPENROUTER_BASE_URL",),
        config_dir=tmp_path,
    )

    assert resolved.provider == "openrouter"
    assert resolved.api_key == "anthropic-key"  # pragma: allowlist secret  # pragma: allowlist secret
    assert resolved.base_url == DEFAULT_OPENROUTER_BASE_URL


def test_resolve_provider_connectivity_uses_default_base_url(tmp_path):
    resolved = resolve_provider_connectivity(
        cli_api_key=None,
        cli_base_url=None,
        api_key_env_vars=("OPENROUTER_API_KEY",),
        base_url_env_vars=("OPENROUTER_BASE_URL",),
        config_dir=tmp_path,
    )
    assert resolved.api_key is None
    assert resolved.base_url == DEFAULT_OPENROUTER_BASE_URL


def test_redacted_snapshot_hides_api_key(tmp_path):
    save_user_config(
        UserConfig(
            providers={"nvidia": ProviderConfig(base_url="https://example.test/v1")},
            launch={
                "codex": LaunchConfig(
                    model="openai/openai/openai/gpt-5.5",
                    route=LaunchRouteConfig(
                        type="single",
                        model="openai/openai/openai/gpt-5.5",
                        endpoints={
                            PRIMARY_TIER: LaunchTierEndpointConfig(
                                base_url="https://codex.example/v1",
                            ),
                        },
                    ),
                ),
            },
        ),
        config_dir=tmp_path,
    )
    save_user_credentials(
        UserCredentials(
            api_keys={"nvidia": "nvapi-abcdef123456"},
            launch={
                "codex": LaunchCredentials(
                    api_keys={PRIMARY_TIER: "codex-abcdef123456"},
                ),
            },
        ),
        config_dir=tmp_path,
    )

    snapshot = build_redacted_snapshot(tmp_path)
    providers = snapshot["providers"]
    assert isinstance(providers, dict)
    provider = providers["nvidia"]
    assert isinstance(provider, dict)
    assert provider["api_key"] == "nvap...3456"
    launch = snapshot["launch"]
    assert isinstance(launch, dict)
    codex = launch["codex"]
    assert isinstance(codex, dict)
    route = codex["route"]
    assert isinstance(route, dict)
    endpoints = route["endpoints"]
    assert isinstance(endpoints, dict)
    primary = endpoints[PRIMARY_TIER]
    assert isinstance(primary, dict)
    assert primary["base_url"] == "https://codex.example/v1"
    assert primary["api_key"] == "code...3456"
