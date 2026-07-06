# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation of ``switchyard configure``."""

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from switchyard.cli.command_utils import (
    build_launch_config_wizard,
    discover_models,
    is_interactive_terminal,
)
from switchyard.cli.config.user_config import (
    DEFAULT_SECRETS_SECTION_PRIORITY,
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
    _default_base_url_for_provider,
    build_redacted_snapshot,
    get_config_path,
    get_credentials_path,
    load_user_config,
    load_user_credentials,
    reset_user_config,
    resolve_provider_connectivity,
    save_user_config,
    save_user_credentials,
)
from switchyard.cli.launchers.claude_alias import claude_alias_for
from switchyard.cli.model_catalog.model_discovery import (
    choose_default_claude_model,
    choose_default_codex_model,
    claude_model_candidates,
    codex_model_candidates,
    fetch_model_ids,
)
from switchyard.cli.models import ModelListRequest, ModelListTarget, render_models
from switchyard.cli.output import format_config_snapshot
from switchyard.cli.route_bundle import routing_profile_model_ids
from switchyard.cli.status import StatusRequest, render_status
from switchyard.cli.tui.launch_config_wizard import LaunchConfigWizard
from switchyard.server.server_util import load_secrets

logger = logging.getLogger(__name__)


def _skill_distillation_args_present(args: argparse.Namespace) -> bool:
    return bool(args.disable_skill_distillation) or args.skill_distillation is not None


def _provider_or_launcher_args_present(args: argparse.Namespace) -> bool:
    return any(
        value
        for value in (
            args.api_key,
            args.base_url,
            args.claude_model,
            args.claude_base_url,
            args.claude_api_key,
            args.codex_model,
            args.codex_base_url,
            args.codex_api_key,
            args.openclaw_model,
            args.openclaw_base_url,
            args.openclaw_api_key,
        )
    ) or args.routing_profiles is not None


def _skill_only_config_update(args: argparse.Namespace) -> bool:
    return (
        _skill_distillation_args_present(args)
        and not _provider_or_launcher_args_present(args)
    )


def _apply_skill_distillation_args(
    existing: SkillDistillationConfig,
    args: argparse.Namespace,
) -> SkillDistillationConfig:
    if args.disable_skill_distillation:
        if args.skill_distillation is not None:
            raise UserConfigError(
                "--disable-skill-distillation cannot be combined with "
                "--skill-distillation"
            )
        return SkillDistillationConfig()

    return SkillDistillationConfig(
        namespace=(
            args.skill_distillation
            if args.skill_distillation is not None
            else existing.namespace
        ),
    )


@dataclass(frozen=True)
class _LaunchEndpointSelection:
    """Configure-time endpoint choice for one launch tier."""

    endpoint: LaunchTierEndpointConfig | None
    api_key: str | None
    effective_base_url: str
    effective_api_key: str


def _endpoint_override(
    base_url: str | None,
    *,
    default_base_url: str,
) -> LaunchTierEndpointConfig | None:
    if not base_url or base_url == default_base_url:
        return None
    return LaunchTierEndpointConfig(base_url=base_url)


def _api_key_override(api_key: str | None, *, default_api_key: str) -> str | None:
    if not api_key or api_key == default_api_key:
        return None
    return api_key


def _select_launch_endpoint(
    *,
    wizard: LaunchConfigWizard | None,
    interactive: bool,
    label: str,
    cli_base_url: str | None,
    cli_api_key: str | None,
    existing_endpoint: LaunchTierEndpointConfig,
    existing_api_key: str | None,
    default_base_url: str,
    default_api_key: str,
) -> _LaunchEndpointSelection:
    """Resolve configure-time endpoint overrides for one launch tier."""

    existing_base_url = existing_endpoint.base_url
    existing_effective_base_url = existing_base_url or default_base_url
    existing_effective_api_key = existing_api_key or default_api_key
    if cli_base_url or cli_api_key:
        effective_base_url = cli_base_url or existing_effective_base_url
        effective_api_key = cli_api_key or existing_effective_api_key
        return _LaunchEndpointSelection(
            endpoint=_endpoint_override(
                effective_base_url,
                default_base_url=default_base_url,
            ),
            api_key=_api_key_override(
                effective_api_key,
                default_api_key=default_api_key,
            ),
            effective_base_url=effective_base_url,
            effective_api_key=effective_api_key,
        )

    has_existing_override = bool(existing_base_url or existing_api_key)
    if interactive and wizard:
        mode = wizard.select_endpoint_mode(
            label,
            default="custom" if has_existing_override else "default",
        )
        if mode == "default":
            return _LaunchEndpointSelection(
                endpoint=None,
                api_key=None,
                effective_base_url=default_base_url,
                effective_api_key=default_api_key,
            )
        effective_base_url = wizard.prompt_endpoint_base_url(
            label,
            existing_effective_base_url,
        )
        effective_api_key = wizard.prompt_endpoint_api_key(
            label,
            existing_effective_api_key,
        )
        return _LaunchEndpointSelection(
            endpoint=_endpoint_override(
                effective_base_url,
                default_base_url=default_base_url,
            ),
            api_key=_api_key_override(
                effective_api_key,
                default_api_key=default_api_key,
            ),
            effective_base_url=effective_base_url,
            effective_api_key=effective_api_key,
        )

    return _LaunchEndpointSelection(
        endpoint=_endpoint_override(
            existing_effective_base_url,
            default_base_url=default_base_url,
        ),
        api_key=_api_key_override(
            existing_effective_api_key,
            default_api_key=default_api_key,
        ),
        effective_base_url=existing_effective_base_url,
        effective_api_key=existing_effective_api_key,
    )


def _route_endpoints(
    primary: _LaunchEndpointSelection,
) -> dict[str, LaunchTierEndpointConfig] | None:
    endpoints: dict[str, LaunchTierEndpointConfig] = {}
    if primary.endpoint:
        endpoints[PRIMARY_TIER] = primary.endpoint
    return endpoints or None


def _launch_credentials(
    primary: _LaunchEndpointSelection,
) -> LaunchCredentials:
    api_keys: dict[str, str] = {}
    if primary.api_key:
        api_keys[PRIMARY_TIER] = primary.api_key
    return LaunchCredentials(api_keys=api_keys or None)


def _read_routing_profiles_file(path: str) -> dict[str, object]:
    """Read and parse a routing-profiles YAML file into a JSON-storable dict.

    Env-var references (``${VAR}``) inside the YAML are preserved as
    literal strings; they expand at *load* time on each
    ``serve`` / ``launch`` run, not now.  This means saving a bundle on
    one machine and using it on another picks up the *local* environment.
    """
    resolved = Path(path).expanduser()
    try:
        raw = resolved.read_text()
    except FileNotFoundError as exc:
        raise SystemExit(
            f"configure --routing-profiles: file not found: {resolved}"
        ) from exc
    except OSError as exc:
        raise SystemExit(
            f"configure --routing-profiles: cannot read {resolved}: {exc}"
        ) from exc

    import yaml  # type: ignore[import-untyped,unused-ignore]
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(
            f"configure --routing-profiles: invalid YAML in {resolved}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise SystemExit(
            f"configure --routing-profiles: {resolved} must contain a YAML "
            "mapping (got a list or scalar at the top level)"
        )
    return parsed


def _discover_models_for_endpoint(
    *,
    cache: dict[tuple[str, str], list[str]],
    base_url: str,
    api_key: str,
    disabled: bool,
) -> list[str]:
    key = (base_url, api_key)
    if key not in cache:
        cache[key] = discover_models(base_url, api_key, disabled=disabled)
    return cache[key]


def _merge_candidate_ids(*sources: Iterable[str]) -> list[str]:
    """Return the union of model-id sources in first-seen order."""
    seen: set[str] = set()
    merged: list[str] = []
    for source in sources:
        for value in source:
            if value and value not in seen:
                seen.add(value)
                merged.append(value)
    return merged


def _list_models(args: argparse.Namespace) -> None:
    """Print a ranked list of backend models — fold of ``switchyard models``."""
    from switchyard.server.server_util import load_secrets

    connectivity = resolve_provider_connectivity(
        cli_api_key=args.api_key,
        cli_base_url=args.base_url,
        api_key_env_vars=("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"),
        base_url_env_vars=("OPENROUTER_BASE_URL", "NVIDIA_BASE_URL", "OPENAI_BASE_URL"),
        secrets=load_secrets(),
        secrets_section_priority=DEFAULT_SECRETS_SECTION_PRIORITY,
        default_provider=getattr(args, "provider", None),
    )
    if not connectivity.api_key:
        raise SystemExit(
            "configure --list-models: no API key resolved. Run "
            "`switchyard configure`, set OPENROUTER_API_KEY, or pass --api-key."
        )
    model_ids = fetch_model_ids(connectivity.base_url, connectivity.api_key)
    # `--target provider` is meaningful for `configure` setup but not for
    # model ranking; collapse it to `all` so the renderer treats the
    # ranking target as the launcher-neutral default.
    raw_target = getattr(args, "target", "all") or "all"
    target: ModelListTarget
    if raw_target == "claude":
        target = "claude"
    elif raw_target == "codex":
        target = "codex"
    else:
        target = "all"
    print(render_models(
        model_ids,
        ModelListRequest(
            target=target,
            query=getattr(args, "query", None),
            limit=getattr(args, "limit", 50),
        ),
    ))


def cmd_configure(args: argparse.Namespace) -> None:
    """Set, show, or reset user-level launcher defaults."""

    if getattr(args, "list_models", False):
        _list_models(args)
        return

    if args.show:
        snapshot = build_redacted_snapshot()
        if bool(getattr(args, "json", False)):
            print(json.dumps(snapshot, indent=2, sort_keys=True))
            return
        print(format_config_snapshot(snapshot))
        print()
        print(render_status(
            StatusRequest(
                cli_api_key=getattr(args, "api_key", None),
                cli_base_url=getattr(args, "base_url", None),
                provider=getattr(args, "provider", None),
                check=bool(getattr(args, "check", False)),
                secrets=load_secrets(),
            ),
        ))
        return

    if args.reset:
        removed = reset_user_config()
        if removed:
            print("Removed Switchyard user config:")
            for path in removed:
                print(f"  {path}")
        else:
            print("No Switchyard user config found.")
        return

    provider = args.provider
    target_scope = getattr(args, "target", "all")
    configure_claude = target_scope in ("all", "claude")
    configure_codex = target_scope in ("all", "codex")
    configure_openclaw = target_scope in ("all", "openclaw")
    existing_config = load_user_config()
    existing_credentials = load_user_credentials()
    skill_distillation = _apply_skill_distillation_args(
        existing_config.skill_distillation,
        args,
    )
    if _skill_only_config_update(args):
        save_user_config(
            UserConfig(
                default_provider=existing_config.default_provider,
                providers=existing_config.providers,
                launch=existing_config.launch,
                routing_profiles=existing_config.routing_profiles,
                skill_distillation=skill_distillation,
            ),
        )
        print(f"Saved Switchyard config to {get_config_path()}")
        return

    existing_provider = existing_config.provider(provider)
    existing_claude = existing_config.launch_target("claude")
    existing_codex = existing_config.launch_target("codex")
    existing_openclaw = existing_config.launch_target("openclaw")
    existing_claude_credentials = existing_credentials.launch_target("claude")
    existing_codex_credentials = existing_credentials.launch_target("codex")
    existing_openclaw_credentials = existing_credentials.launch_target("openclaw")
    reuse_existing_provider = bool(getattr(args, "reuse_existing_provider", False))
    interactive = is_interactive_terminal()
    wizard = build_launch_config_wizard(args) if interactive else None
    if wizard:
        wizard.start(target=target_scope)

    base_url_default = existing_provider.base_url or _default_base_url_for_provider(provider)
    if args.base_url:
        base_url = args.base_url
    elif reuse_existing_provider and existing_provider.base_url:
        base_url = existing_provider.base_url
    elif interactive and wizard:
        base_url = wizard.prompt_default_base_url(base_url_default)
    else:
        base_url = base_url_default

    api_key = args.api_key
    if not api_key:
        existing_api_key = existing_credentials.api_key(provider)
        prompt_default_api_key = (
            existing_api_key
            or getattr(args, "prompt_default_api_key", None)
        )
        prompt_default_api_key_source = getattr(
            args, "prompt_default_api_key_source", None,
        )
        if reuse_existing_provider and existing_api_key:
            api_key = existing_api_key
        elif interactive and wizard:
            api_key = wizard.prompt_default_api_key(
                prompt_default_api_key,
                default_source=prompt_default_api_key_source,
            )
        else:
            api_key = prompt_default_api_key
    if not api_key:
        raise SystemExit(
            "configure requires an API key. Pass --api-key or run in an "
            "interactive terminal."
        )

    model_catalog_cache: dict[tuple[str, str], list[str]] = {}

    def _models_for(selection: _LaunchEndpointSelection) -> list[str]:
        return _discover_models_for_endpoint(
            cache=model_catalog_cache,
            base_url=selection.effective_base_url,
            api_key=selection.effective_api_key,
            disabled=args.no_model_discovery,
        )

    # Resolve routing-profiles up front (CLI > existing config > interactive
    # prompt) so the model picker can surface route ids + their tier models
    # alongside the upstream catalog. Without this the wizard offers only the
    # raw `/v1/models` response and the user can't pick e.g. `opus-ds-stage_router`
    # even when their YAML declares it.
    routing_profiles = existing_config.routing_profiles
    if args.routing_profiles is not None:
        # Empty string clears, any other value is a path whose YAML we parse.
        routing_profiles = (
            _read_routing_profiles_file(args.routing_profiles)
            if args.routing_profiles else None
        )
    elif interactive and wizard:
        prompted = wizard.prompt_routing_profiles(default=None)
        if prompted:
            routing_profiles = _read_routing_profiles_file(prompted)
    routing_profile_models = routing_profile_model_ids(routing_profiles)
    # When a routing-profiles bundle is in play, its first route is the model a
    # launcher sends by default (the launcher seeds the agent with
    # `table.registered_models()[0]`). Make that the configured model default
    # too so `configure` and `launch` agree; an explicit `--*-model` still wins.
    default_route_model = routing_profile_models[0] if routing_profile_models else None
    # Claude Code's gateway-discovery filter only accepts `claude-`/`anthropic-`
    # prefixed ids. The claude launcher registers each route under both spellings
    # via `_with_claude_aliases`; save the prefixed form here so configure and
    # the launcher pick the same default.
    default_claude_route_model = default_route_model
    if default_claude_route_model is not None:
        default_claude_route_model = (
            claude_alias_for(default_claude_route_model) or default_claude_route_model
        )

    existing_claude_route = existing_claude.effective_route()
    claude_model = existing_claude_route.model
    claude_primary_endpoint: _LaunchEndpointSelection | None = None
    if configure_claude:
        claude_primary_endpoint = _select_launch_endpoint(
            wizard=wizard,
            interactive=interactive,
            label="Claude Code",
            cli_base_url=args.claude_base_url,
            cli_api_key=args.claude_api_key,
            existing_endpoint=existing_claude_route.endpoint(PRIMARY_TIER),
            existing_api_key=existing_claude_credentials.api_key(PRIMARY_TIER),
            default_base_url=base_url,
            default_api_key=api_key,
        )
        claude_model_ids = _merge_candidate_ids(
            routing_profile_models, _models_for(claude_primary_endpoint)
        )
        discovered_claude = choose_default_claude_model(claude_model_ids)
        claude_default = (
            args.claude_model
            or default_claude_route_model
            or existing_claude_route.model
            or discovered_claude
        )
        if not claude_default and not interactive:
            raise SystemExit(
                "No Claude model configured or discovered. Pass --claude-model "
                "or run interactively."
            )

        if args.claude_model:
            claude_model = args.claude_model
        elif interactive and wizard:
            claude_model = wizard.select_model(
                "Claude Code",
                preferred_model_ids=claude_model_candidates(claude_model_ids),
                all_model_ids=claude_model_ids,
                default=claude_default,
            )
        else:
            claude_model = claude_default

    existing_codex_route = existing_codex.effective_route()
    codex_model = existing_codex_route.model
    codex_primary_endpoint: _LaunchEndpointSelection | None = None
    if configure_codex:
        codex_primary_endpoint = _select_launch_endpoint(
            wizard=wizard,
            interactive=interactive,
            label="Codex",
            cli_base_url=args.codex_base_url,
            cli_api_key=args.codex_api_key,
            existing_endpoint=existing_codex_route.endpoint(PRIMARY_TIER),
            existing_api_key=existing_codex_credentials.api_key(PRIMARY_TIER),
            default_base_url=base_url,
            default_api_key=api_key,
        )
        codex_model_ids = _merge_candidate_ids(
            routing_profile_models, _models_for(codex_primary_endpoint)
        )
        discovered_codex = choose_default_codex_model(codex_model_ids)
        codex_default = (
            args.codex_model
            or default_route_model
            or existing_codex_route.model
            or discovered_codex
        )
        if not codex_default and not interactive:
            raise SystemExit(
                "No Codex model configured or discovered. Pass --codex-model "
                "or run interactively."
            )

        if args.codex_model:
            codex_model = args.codex_model
        elif interactive and wizard:
            codex_model = wizard.select_model(
                "Codex",
                preferred_model_ids=codex_model_candidates(codex_model_ids),
                all_model_ids=codex_model_ids,
                default=codex_default,
            )
        else:
            codex_model = codex_default

    existing_openclaw_route = existing_openclaw.effective_route()
    openclaw_model = existing_openclaw_route.model
    openclaw_primary_endpoint: _LaunchEndpointSelection | None = None
    if configure_openclaw:
        openclaw_primary_endpoint = _select_launch_endpoint(
            wizard=wizard,
            interactive=interactive,
            label="OpenClaw",
            cli_base_url=args.openclaw_base_url,
            cli_api_key=args.openclaw_api_key,
            existing_endpoint=existing_openclaw_route.endpoint(PRIMARY_TIER),
            existing_api_key=existing_openclaw_credentials.api_key(PRIMARY_TIER),
            default_base_url=base_url,
            default_api_key=api_key,
        )
        openclaw_model_ids = _merge_candidate_ids(
            routing_profile_models, _models_for(openclaw_primary_endpoint)
        )
        # OpenClaw and Codex both speak OpenAI Chat Completions to the
        # backend, so the Codex preference ranking is a reasonable default
        # until OpenClaw-specific signals emerge.
        discovered_openclaw = choose_default_codex_model(openclaw_model_ids)
        openclaw_default = (
            args.openclaw_model
            or default_route_model
            or existing_openclaw_route.model
            or discovered_openclaw
        )
        if not openclaw_default and not interactive:
            raise SystemExit(
                "No OpenClaw model configured or discovered. Pass "
                "--openclaw-model or run interactively."
            )

        if args.openclaw_model:
            openclaw_model = args.openclaw_model
        elif interactive and wizard:
            openclaw_model = wizard.select_model(
                "OpenClaw",
                preferred_model_ids=codex_model_candidates(openclaw_model_ids),
                all_model_ids=openclaw_model_ids,
                default=openclaw_default,
            )
        else:
            openclaw_model = openclaw_default

    providers = dict(existing_config.providers or {})
    providers[provider] = ProviderConfig(base_url=base_url)
    launch = dict(existing_config.launch or {})
    if configure_claude:
        if claude_primary_endpoint is None:
            raise AssertionError("Claude endpoint selection was not resolved")
        launch["claude"] = LaunchConfig(
            model=claude_model,
            route=LaunchRouteConfig(
                type="single",
                model=claude_model,
                endpoints=_route_endpoints(primary=claude_primary_endpoint),
            ),
        )
    if configure_codex:
        if codex_primary_endpoint is None:
            raise AssertionError("Codex endpoint selection was not resolved")
        launch["codex"] = LaunchConfig(
            model=codex_model,
            route=LaunchRouteConfig(
                type="single",
                model=codex_model,
                endpoints=_route_endpoints(primary=codex_primary_endpoint),
            ),
        )
    if configure_openclaw:
        if openclaw_primary_endpoint is None:
            raise AssertionError("OpenClaw endpoint selection was not resolved")
        launch["openclaw"] = LaunchConfig(
            model=openclaw_model,
            route=LaunchRouteConfig(
                type="single",
                model=openclaw_model,
                endpoints=_route_endpoints(primary=openclaw_primary_endpoint),
            ),
        )
    save_user_config(
        UserConfig(
            default_provider=provider,
            providers=providers,
            launch=launch,
            routing_profiles=routing_profiles,
            skill_distillation=skill_distillation,
        ),
    )

    api_keys = dict(existing_credentials.api_keys or {})
    api_keys[provider] = api_key
    launch_credentials = dict(existing_credentials.launch or {})
    if configure_claude:
        if claude_primary_endpoint is None:
            raise AssertionError("Claude endpoint selection was not resolved")
        launch_credentials["claude"] = _launch_credentials(claude_primary_endpoint)
    if configure_codex:
        if codex_primary_endpoint is None:
            raise AssertionError("Codex endpoint selection was not resolved")
        launch_credentials["codex"] = _launch_credentials(codex_primary_endpoint)
    if configure_openclaw:
        if openclaw_primary_endpoint is None:
            raise AssertionError("OpenClaw endpoint selection was not resolved")
        launch_credentials["openclaw"] = _launch_credentials(openclaw_primary_endpoint)
    save_user_credentials(
        UserCredentials(api_keys=api_keys, launch=launch_credentials),
    )

    print(f"Saved Switchyard config to {get_config_path()}")
    print(f"Saved Switchyard credentials to {get_credentials_path()}")


__all__ = ["cmd_configure"]
