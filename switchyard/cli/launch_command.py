# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implement ``switchyard launch`` subcommands."""

import argparse
import logging
import os
from dataclasses import replace
from importlib import import_module
from typing import Protocol, cast

from switchyard.cli.command_utils import (
    is_interactive_terminal,
    strip_forwarded_args,
)
from switchyard.cli.config.user_config import (
    DEFAULT_SECRETS_SECTION_PRIORITY,
    PRIMARY_TIER,
    WEAK_TIER,
    LaunchRouteConfig,
    LaunchTarget,
    LaunchTierEndpointConfig,
    ProviderConnectivity,
    load_user_config,
    load_user_credentials,
    resolve_provider_connectivity,
)
from switchyard.cli.configure_command import cmd_configure
from switchyard.cli.configure_request import ConfigureRequest, ConfigureTarget
from switchyard.cli.intake_cli_config import IntakeCliConfig
from switchyard.cli.launchers.launch_intake_config import LaunchIntakeConfig
from switchyard.cli.output import format_dry_run
from switchyard.cli.routing import (
    LaunchTierConnectivity,
    build_deterministic_routing_config,
    require_route_model,
)
from switchyard.lib import startup_timing
from switchyard.lib.backends.llm_target import BackendFormat
from switchyard.lib.profiles.random_routing import (
    RandomRoutingConfig,
)
from switchyard.server.server_util import load_secrets, resolve_rl_log_dir

logger = logging.getLogger(__name__)


class _YamlModule(Protocol):
    def safe_dump(
        self,
        data: object,
        stream: object,
        sort_keys: bool = ...,
    ) -> object: ...


def _api_key_prompt_default_source(
    args: argparse.Namespace,
    api_key_env_vars: tuple[str, ...],
    resolved_api_key: str | None,
) -> str | None:
    if args.api_key or not resolved_api_key:
        return None
    for env_var in api_key_env_vars:
        if os.environ.get(env_var) == resolved_api_key:
            return f"${env_var}"
    return None


def _resolve_launch_connectivity(
    args: argparse.Namespace,
    api_key_env_vars: tuple[str, ...],
) -> ProviderConnectivity:
    """Resolve launch credentials without inheriting repo secrets port."""

    return resolve_provider_connectivity(
        cli_api_key=args.api_key,
        cli_base_url=args.base_url,
        api_key_env_vars=api_key_env_vars,
        base_url_env_vars=("OPENROUTER_BASE_URL", "NVIDIA_BASE_URL", "OPENAI_BASE_URL"),
        secrets=load_secrets(),
        secrets_section_priority=DEFAULT_SECRETS_SECTION_PRIORITY,
    )


def resolve_launch_connectivity(
    args: argparse.Namespace,
    api_key_env_vars: tuple[str, ...],
) -> tuple[str | None, str]:
    """Resolve launch credentials without inheriting repo secrets port."""

    connectivity = _resolve_launch_connectivity(
        args,
        api_key_env_vars=api_key_env_vars,
    )
    return connectivity.api_key, connectivity.base_url


# Marker for the OpenRouter gateway. The zero-flag deterministic default trio
# uses OpenRouter-only model ids, so a resolved endpoint that isn't OpenRouter
# cannot serve them.
_OPENROUTER_URL_MARKER = "openrouter.ai"


def _reject_openrouter_only_default_trio(target: str, base_url: str) -> None:
    """Fail fast when the zero-config default trio can't reach the provider.

    The deterministic zero-flag default uses OpenRouter-only model ids
    (``anthropic/claude-opus-4.7`` / ``moonshotai/kimi-k2.6`` /
    ``google/gemini-3.5-flash``). This runs only on the deterministic path, so
    ``--model`` has already been ruled out, and ``--weak-model`` /
    ``--classifier-model`` do not help because the strong tier stays
    ``anthropic/claude-opus-4.7``. So when the resolved endpoint is not
    OpenRouter, raise :class:`SystemExit` pointing the user at ``--model``,
    ``--routing-profiles``, or ``OPENROUTER_API_KEY`` instead of launching a
    trio that can't work.
    """
    if _OPENROUTER_URL_MARKER in base_url:
        return
    raise SystemExit(
        f"launch {target}: the zero-config default trio "
        f"(anthropic/claude-opus-4.7, moonshotai/kimi-k2.6, "
        f"google/gemini-3.5-flash) is OpenRouter-only, but the resolved "
        f"endpoint is {base_url!r}. Pass --model, use --routing-profiles, or "
        f"set OPENROUTER_API_KEY."
    )


def _is_deterministic_launch(
    target: str,
    args: argparse.Namespace,
    route: LaunchRouteConfig | None,
    routing_profiles: str | None,
) -> bool:
    """Return ``True`` when the resolved launch will use deterministic routing.

    For ``claude`` / ``codex``: deterministic is the implicit default when
    no single-model passthrough (CLI ``--model`` or saved
    ``configured_route.model``) and no routing-profiles bundle (CLI
    ``--routing-profiles`` or saved bundle) is in play. Pass
    ``route=None`` to check before the route is resolved (e.g. inside
    :func:`launch_requirements_satisfied`); the resolver inspects
    ``args.model`` and the saved single-model config directly.

    For ``openclaw``: same implicit default as claude/codex (the legacy
    ``--deterministic`` opt-in flag was removed).
    """
    if target not in ("claude", "codex", "openclaw"):
        return False
    if routing_profiles:
        return False
    if route is not None:
        return not route.model
    # Pre-resolution check: replicate route_from_launch_args' "args.model
    # wins, else configured_route.model" lookup without re-running the merge.
    if getattr(args, "model", None):
        return False
    target_key = cast(LaunchTarget, target)
    configured_launch = load_user_config().launch_target(target_key)
    configured_route = configured_launch.effective_route()
    return not configured_route.model


_SAVED_BUNDLE_TEMPFILES: list[str] = []


def _materialize_saved_bundle(bundle: dict[str, object]) -> str:
    """Re-serialize a saved bundle dict to a YAML tempfile and return its path.

    The launcher's downstream API takes a YAML path; saved bundles are
    stored as parsed dicts in ``config.json``. The tempfile is retained
    for the process lifetime so the launcher's per-request reads keep
    working without re-materializing each time.
    """
    import tempfile

    yaml = cast(_YamlModule, import_module("yaml"))
    fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="switchyard-routes-",
    )
    try:
        yaml.safe_dump(bundle, fd, sort_keys=False)
    finally:
        fd.close()
    _SAVED_BUNDLE_TEMPFILES.append(fd.name)
    return fd.name


def _resolve_routing_profiles(args: argparse.Namespace) -> str | None:
    """Resolve the routing-profile YAML path for a launch.

    Precedence:
        CLI ``--routing-profiles PATH``  >  saved parsed bundle in
        ``UserConfig.routing_profiles`` (only when no ``--model`` is on
        the CLI).

    Passing ``--model X`` alone is an explicit opt-in to a single-model
    launch; the saved bundle is *not* injected on top. When the fallback
    fires we re-serialize the saved bundle to a tempfile (the launcher
    API takes a path).
    """
    if args.routing_profiles:
        return cast(str, args.routing_profiles)
    if args.model:
        return None
    saved = load_user_config().routing_profiles
    if not saved:
        return None
    return _materialize_saved_bundle(saved)


def _resolve_initial_from_profiles(*, target: str, routing_profiles: str) -> str:
    """Return the first route key from *routing_profiles*."""
    from switchyard.cli.route_bundle import (
        parse_routing_profiles_file,
        routing_profile_model_ids,
    )

    ids = routing_profile_model_ids(parse_routing_profiles_file(routing_profiles))
    if not ids:
        raise SystemExit(
            f"launch {target}: --routing-profiles file has no routes; "
            f"add at least one route."
        )
    return ids[0]


def launch_requirements_satisfied(
    args: argparse.Namespace,
    target: str,
    api_key_env_vars: tuple[str, ...],
) -> bool:
    if bool(getattr(args, "reconfigure", False)):
        return False
    api_key, _base_url = resolve_launch_connectivity(
        args,
        api_key_env_vars=api_key_env_vars,
    )
    routing_profiles = getattr(args, "routing_profiles", None)
    target_key = cast(LaunchTarget, target)
    user_config = load_user_config()
    configured_launch = user_config.launch_target(target_key)
    configured_route = configured_launch.effective_route()
    launch_credentials = load_user_credentials().launch_target(target_key)
    # LLM-classifier routing is self-sufficient: the preset bundle supplies
    # models, profile, classifier — only the primary API key has to come from
    # the user. We land here for claude/codex/openclaw when no --model and no
    # routing-profiles are configured — the implicit default.
    if _is_deterministic_launch(
        target=target, args=args, route=None, routing_profiles=routing_profiles,
    ):
        return bool(api_key or launch_credentials.api_key(PRIMARY_TIER))
    # --routing-profiles (CLI flag or saved top-level bundle) is self-sufficient:
    # the YAML carries every chain's credentials and the first route becomes
    # the initial agent model. The saved bundle is only consulted when no
    # --model is on the CLI (passing --model X is an explicit opt-in to a
    # single-model launch).
    if routing_profiles:
        return True
    if not args.model and user_config.routing_profiles:
        return True
    has_model = bool(args.model or configured_route.model)
    has_primary_key = bool(api_key or launch_credentials.api_key(PRIMARY_TIER))
    return has_model and has_primary_key


def maybe_bootstrap_launch_config(
    args: argparse.Namespace,
    target: str,
    api_key_env_vars: tuple[str, ...],
) -> None:
    """Run first-time interactive config when launch needs persisted defaults."""

    if launch_requirements_satisfied(
        args,
        target=target,
        api_key_env_vars=api_key_env_vars,
    ):
        return
    if not is_interactive_terminal():
        raise SystemExit(
            f"Switchyard is missing {target} launch defaults. Run "
            f"`switchyard configure --target {target}` or pass "
            f"--model and --api-key for a one-off launch."
        )

    connectivity = _resolve_launch_connectivity(
        args,
        api_key_env_vars=api_key_env_vars,
    )
    resolved_api_key = connectivity.api_key
    print(f"Switchyard is missing {target} launch defaults. Starting setup.")
    # Build the same request `switchyard configure` would. Any field we don't
    # set here — skill distillation, routing profiles, the --show/--list modes —
    # takes its default from ConfigureRequest, so setup can't crash on one we
    # forgot to pass.
    cmd_configure(ConfigureRequest(
        target=cast(ConfigureTarget, target),
        provider=connectivity.provider,
        base_url=connectivity.base_url,
        api_key=args.api_key,
        prompt_default_api_key=resolved_api_key,
        prompt_default_api_key_source=_api_key_prompt_default_source(
            args=args,
            api_key_env_vars=api_key_env_vars,
            resolved_api_key=resolved_api_key,
        ),
        reuse_existing_provider=True,
        claude_model=args.model if target == "claude" else None,
        codex_model=args.model if target == "codex" else None,
        openclaw_model=args.model if target == "openclaw" else None,
        no_model_discovery=getattr(args, "no_model_discovery", False),
        no_tui=getattr(args, "no_tui", False),
    ))
    if not args.api_key:
        configured_api_key = load_user_credentials().api_key(connectivity.provider)
        if configured_api_key:
            args.api_key = configured_api_key


def route_from_launch_args(
    args: argparse.Namespace,
    configured_route: LaunchRouteConfig,
) -> LaunchRouteConfig:
    """Merge one-off launch flags onto the configured route.

    The launcher CLI no longer exposes routing-policy flags; routing
    policies live in routing-profile YAML. We always emit a single-tier
    route here. Legacy saved random routes are rejected upstream by
    :func:`_warn_if_legacy_random_config`.
    """

    model = getattr(args, "model", None) or configured_route.model
    endpoints = None
    primary_endpoint = configured_route.endpoint(PRIMARY_TIER)
    if primary_endpoint.base_url:
        endpoints = {PRIMARY_TIER: primary_endpoint}
    return LaunchRouteConfig(
        type="single",
        model=model,
        endpoints=endpoints,
    )


def route_from_random_config(config: RandomRoutingConfig) -> LaunchRouteConfig:
    """Render a random-routing config as a launcher route for summaries."""

    endpoints: dict[str, LaunchTierEndpointConfig] = {}
    if config.strong.endpoint.base_url:
        endpoints[PRIMARY_TIER] = LaunchTierEndpointConfig(
            base_url=config.strong.endpoint.base_url,
        )
    if config.weak.endpoint.base_url:
        endpoints[WEAK_TIER] = LaunchTierEndpointConfig(
            base_url=config.weak.endpoint.base_url,
        )
    return LaunchRouteConfig(
        type="random",
        model=config.strong.model,
        weak_model=config.weak.model,
        strong_probability=config.strong_probability,
        endpoints=endpoints or None,
    )


def resolve_route_tier_connectivity(
    target: str,
    route: LaunchRouteConfig,
    tier: str,
    default_api_key: str | None,
    default_base_url: str,
) -> LaunchTierConnectivity:
    credentials = load_user_credentials()
    launch_credentials = credentials.launch_target(cast(LaunchTarget, target))
    endpoint = route.endpoint(tier)
    return LaunchTierConnectivity(
        api_key=launch_credentials.api_key(tier) or default_api_key,
        base_url=endpoint.base_url or default_base_url,
    )


def require_launch_tier_key(
    target: str,
    tier: str,
    connectivity: LaunchTierConnectivity,
) -> LaunchTierConnectivity:
    if connectivity.api_key:
        return connectivity
    logger.warning(
        "No API key resolved for %s %s tier. Set --api-key, an API key env "
        "var, run `switchyard configure`, or add secrets/secrets.json. "
        "Using 'dummy' as placeholder.",
        target,
        tier,
    )
    return replace(connectivity, api_key="dummy")


def placeholder_launch_tier_key(
    connectivity: LaunchTierConnectivity,
) -> LaunchTierConnectivity:
    if connectivity.api_key:
        return connectivity
    return replace(connectivity, api_key="dummy")


def _run_launch_smoke(
    target: str,
    model: str | None,
    connectivity: LaunchTierConnectivity,
    port: int | None,
    timeout: float | None,
) -> None:
    """Dispatch the harness-driven smoke checklist and exit.

    ``--smoke`` on ``launch claude`` / ``launch codex`` / ``launch openclaw``
    runs :func:`verify_claude` / :func:`verify_codex` /
    :func:`verify_openclaw` and exits with their return code. Requires an
    explicit ``--model``; ``--routing-profiles`` is rejected before this
    function is reached. ``raise SystemExit(returncode)`` returns control to
    the user without spawning the interactive harness.
    """
    from switchyard.server.verify import (
        verify_claude,
        verify_codex,
        verify_openclaw,
    )

    if not model:
        raise SystemExit(
            f"launch {target} --smoke: no model resolved. "
            "Pass --model (smoke ignores --routing-profiles; use --model directly)."
        )
    if not connectivity.api_key:
        raise SystemExit(
            f"launch {target} --smoke: no API key resolved. Set --api-key, "
            "an API key env var, or run `switchyard configure`."
        )
    verifiers = {
        "claude": verify_claude,
        "codex": verify_codex,
        "openclaw": verify_openclaw,
    }
    verifier = verifiers[target]
    raise SystemExit(verifier(
        model=model,
        base_url=connectivity.base_url,
        api_key=connectivity.api_key,
        port=port,
        timeout=timeout,
    ))


def resolve_launch_intake_config(
    args: argparse.Namespace,
    target: str,
    default_app: str,
) -> LaunchIntakeConfig | None:
    """Build launch intake config from CLI args / env, or ``None``."""
    intake = IntakeCliConfig.from_launch_args(args)
    if not intake.enabled:
        return None

    return LaunchIntakeConfig.from_resolved(
        base_url=intake.base_url,
        workspace=intake.workspace,
        api_key=intake.api_key,
        app=intake.app or default_app,
        task=intake.task or "developer-session",
        session_id=intake.session_id,
        user_id=intake.user_id,
        target_url=intake.target_url,
        target=target,
    )




def cmd_launch_claude(args: argparse.Namespace) -> None:
    """Start a proxy and spawn ``claude`` against it.

    Three shapes (deterministic is the default when no flags are given):

      * ``(no flags)`` — LLM-classifier strong/weak routing using the
        validated coding-agent trio (Claude Opus 4.7 + Nemotron-3 Super
        + DeepSeek V4 Flash classifier). Override individual tiers with
        ``--weak-model`` / ``--classifier-model`` / ``--profile``.
      * ``--model X`` — single-tier passthrough to X.
      * ``--routing-profiles FILE`` (global flag) — YAML-driven multi-chain table.
        ``--model`` is optional; falls back to the first YAML route.

    Random / latency-aware routing live in the YAML schema.
    """
    if getattr(args, "startup_timing", False):
        startup_timing.enable()
    startup_timing.mark("launch invoked")
    if args.routing_profiles and getattr(args, "smoke", False):
        raise SystemExit(
            "launch claude: --smoke and --routing-profiles cannot be combined. "
            "Pass --model directly to pick the model to smoke-test."
        )
    if args.dry_run and getattr(args, "smoke", False):
        raise SystemExit(
            "launch claude: --smoke and --dry-run cannot be combined. "
            "--smoke runs a live round-trip; --dry-run would be ignored."
        )
    if getattr(args, "smoke", False) and not args.model:
        raise SystemExit(
            "launch claude: --smoke requires --model. "
            "Pass --model directly to pick the model to smoke-test."
        )
    if args.routing_profiles and args.model:
        raise SystemExit(
            "launch claude: --model and --routing-profiles are mutually exclusive.\n"
            "Pass --routing-profiles as a global flag before the subcommand:\n"
            "  switchyard --routing-profiles FILE -- launch claude"
        )
    from switchyard.cli.launchers.claude_code_launcher import (
        launch_claude,
        launch_claude_deterministic_routing,
    )
    from switchyard.cli.launchers.launcher_runtime import ensure_system_ssl_trust

    # Fixes the classifier's "Connection error" failure on Linux dev hosts
    # behind a corporate SSL intercept (Python httpx uses certifi by default;
    # the intercept CA only lives in the system keystore). No-op elsewhere.
    ensure_system_ssl_trust()

    routing_profiles = _resolve_routing_profiles(args)
    if not args.dry_run and not getattr(args, "smoke", False):
        maybe_bootstrap_launch_config(
            args,
            target="claude",
            api_key_env_vars=(
                "OPENROUTER_API_KEY",
                "NVIDIA_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
            ),
        )
    configured_launch = load_user_config().launch_target("claude")
    configured_route = configured_launch.effective_route()
    _warn_if_legacy_random_config(configured_route, target="claude")
    # CLI --routing-profiles is a clean-slate override: route bundle and
    # default model both come from the CLI YAML. The saved config.json launch
    # route is ignored so a saved model id that doesn't exist in the new
    # bundle can't leak through. Gated on the CLI flag specifically — the
    # saved-bundle fallback path still inherits config.json's launch.<t>.model.
    if args.routing_profiles:
        configured_route = LaunchRouteConfig()
    route = route_from_launch_args(args, configured_route)
    # Deterministic LLM-classifier routing is the implicit default for
    # ``launch claude`` when neither single-model passthrough nor a
    # routing-profiles bundle is in play.
    deterministic = _is_deterministic_launch(
        target="claude", args=args, route=route, routing_profiles=routing_profiles,
    )
    if not deterministic and not (routing_profiles and not route.model):
        require_route_model(route, target="claude")

    api_key, base_url = resolve_launch_connectivity(
        args,
        api_key_env_vars=(
            "OPENROUTER_API_KEY",
            "NVIDIA_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ),
    )
    primary_connectivity = resolve_route_tier_connectivity(
        target="claude",
        route=route,
        tier=PRIMARY_TIER,
        default_api_key=api_key,
        default_base_url=base_url,
    )
    weak_connectivity = resolve_route_tier_connectivity(
        target="claude",
        route=route,
        tier=WEAK_TIER,
        default_api_key=api_key,
        default_base_url=base_url,
    )
    claude_args = strip_forwarded_args(args.claude_args)

    if getattr(args, "smoke", False):
        _run_launch_smoke(
            target="claude",
            model=route.model,
            connectivity=primary_connectivity,
            port=args.port,
            timeout=args.timeout,
        )

    if args.dry_run:
        dry_run_classifier_model: str | None = None
        dry_run_profile: str | None = None
        dry_run_min_confidence: float | None = None
        dry_run_route = route
        if deterministic:
            deterministic_config = build_deterministic_routing_config(
                LaunchRouteConfig(
                    type="deterministic",
                    model=getattr(args, "model", None),
                    weak_model=getattr(args, "weak_model", None),
                ),
                primary=placeholder_launch_tier_key(primary_connectivity),
                weak=placeholder_launch_tier_key(weak_connectivity),
                classifier_model=getattr(args, "classifier_model", None),
                profile_name=getattr(args, "profile", None),
                classifier_min_confidence=getattr(
                    args, "classifier_min_confidence", None,
                ),
                backend_format=BackendFormat.OPENAI,
                strong_backend_format=BackendFormat.AUTO,
                timeout=args.timeout,
            )
            dry_run_route = LaunchRouteConfig(
                type="deterministic",
                model=deterministic_config.strong.model,
                weak_model=deterministic_config.weak.model,
            )
            dry_run_classifier_model = deterministic_config.classifier.model
            dry_run_profile = deterministic_config.profile_name
            dry_run_min_confidence = deterministic_config.classifier_min_confidence
        print(format_dry_run(
            target="claude",
            route=dry_run_route,
            base_url=primary_connectivity.base_url,
            api_key_set=bool(primary_connectivity.api_key),
            port=args.port,
            timeout=args.timeout,
            forwarded_args=claude_args,
            classifier_model=dry_run_classifier_model,
            profile=dry_run_profile,
            classifier_min_confidence=dry_run_min_confidence,
            routing_profiles=routing_profiles,
        ))
        return

    intake = resolve_launch_intake_config(
        args, target="claude", default_app="claude-code-switchyard",
    )
    startup_timing.mark("config + credentials resolved")

    if deterministic:
        _reject_openrouter_only_default_trio(
            target="claude",
            base_url=primary_connectivity.base_url,
        )
        primary_connectivity = require_launch_tier_key(
            target="claude",
            tier=PRIMARY_TIER,
            connectivity=primary_connectivity,
        )
        weak_connectivity = placeholder_launch_tier_key(weak_connectivity)
        deterministic_config = build_deterministic_routing_config(
            LaunchRouteConfig(
                type="deterministic",
                model=getattr(args, "model", None),
                weak_model=getattr(args, "weak_model", None),
            ),
            primary=primary_connectivity,
            weak=weak_connectivity,
            classifier_model=getattr(args, "classifier_model", None),
            profile_name=getattr(args, "profile", None),
            classifier_min_confidence=getattr(
                args, "classifier_min_confidence", None,
            ),
            backend_format=BackendFormat.OPENAI,
            strong_backend_format=BackendFormat.AUTO,
            timeout=args.timeout,
        )
        raise SystemExit(launch_claude_deterministic_routing(
            config=deterministic_config,
            port=args.port,
            claude_args=claude_args,
            intake=intake,
            discovery_disabled=bool(getattr(args, "no_model_discovery", False)),
            rl_log_dir=resolve_rl_log_dir(args),
        ))

    if routing_profiles:
        initial = _resolve_initial_from_profiles(
            target="claude",
            routing_profiles=routing_profiles,
        )
    else:
        primary_connectivity = require_launch_tier_key(
            target="claude",
            tier=PRIMARY_TIER,
            connectivity=primary_connectivity,
        )
        initial = require_route_model(route, target="claude")
    raise SystemExit(launch_claude(
        model=initial,
        base_url=primary_connectivity.base_url or "",
        api_key=primary_connectivity.api_key or "dummy",
        port=args.port,
        timeout=args.timeout,
        claude_args=claude_args,
        intake=intake,
        routing_profiles=routing_profiles,
        rl_log_dir=resolve_rl_log_dir(args),
    ))


def cmd_launch_codex(args: argparse.Namespace) -> None:
    """Start a proxy and spawn ``codex`` against it.

    Same shape as :func:`cmd_launch_claude`: deterministic is the
    default when no flags are given; ``--model X`` or
    ``--routing-profiles FILE`` (global flag) opts out.
    """
    if args.routing_profiles and getattr(args, "smoke", False):
        raise SystemExit(
            "launch codex: --smoke and --routing-profiles cannot be combined. "
            "Pass --model directly to pick the model to smoke-test."
        )
    if args.dry_run and getattr(args, "smoke", False):
        raise SystemExit(
            "launch codex: --smoke and --dry-run cannot be combined. "
            "--smoke runs a live round-trip; --dry-run would be ignored."
        )
    if getattr(args, "smoke", False) and not args.model:
        raise SystemExit(
            "launch codex: --smoke requires --model. "
            "Pass --model directly to pick the model to smoke-test."
        )
    if args.routing_profiles and args.model:
        raise SystemExit(
            "launch codex: --model and --routing-profiles are mutually exclusive.\n"
            "Pass --routing-profiles as a global flag before the subcommand:\n"
            "  switchyard --routing-profiles FILE -- launch codex"
        )
    from switchyard.cli.launchers.codex_cli_launcher import (
        launch_codex,
        launch_codex_deterministic_routing,
    )
    from switchyard.cli.launchers.launcher_runtime import ensure_system_ssl_trust

    ensure_system_ssl_trust()

    routing_profiles = _resolve_routing_profiles(args)
    if not args.dry_run and not getattr(args, "smoke", False):
        maybe_bootstrap_launch_config(
            args,
            target="codex",
            api_key_env_vars=("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"),
        )
    configured_launch = load_user_config().launch_target("codex")
    configured_route = configured_launch.effective_route()
    _warn_if_legacy_random_config(configured_route, target="codex")
    if args.routing_profiles:
        configured_route = LaunchRouteConfig()
    route = route_from_launch_args(args, configured_route)
    # Deterministic LLM-classifier routing is the implicit default for
    # ``launch codex`` when neither single-model passthrough nor a
    # routing-profiles bundle is in play.
    deterministic = _is_deterministic_launch(
        target="codex", args=args, route=route, routing_profiles=routing_profiles,
    )
    if not deterministic and not (routing_profiles and not route.model):
        require_route_model(route, target="codex")

    api_key, base_url = resolve_launch_connectivity(
        args, api_key_env_vars=("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"),
    )
    primary_connectivity = resolve_route_tier_connectivity(
        target="codex",
        route=route,
        tier=PRIMARY_TIER,
        default_api_key=api_key,
        default_base_url=base_url,
    )
    weak_connectivity = resolve_route_tier_connectivity(
        target="codex",
        route=route,
        tier=WEAK_TIER,
        default_api_key=api_key,
        default_base_url=base_url,
    )
    codex_args = strip_forwarded_args(args.codex_args)

    if getattr(args, "smoke", False):
        _run_launch_smoke(
            target="codex",
            model=route.model,
            connectivity=primary_connectivity,
            port=args.port,
            timeout=args.timeout,
        )

    if args.dry_run:
        dry_run_classifier_model: str | None = None
        dry_run_profile: str | None = None
        dry_run_min_confidence: float | None = None
        dry_run_route = route
        if deterministic:
            deterministic_config = build_deterministic_routing_config(
                LaunchRouteConfig(
                    type="deterministic",
                    model=getattr(args, "model", None),
                    weak_model=getattr(args, "weak_model", None),
                ),
                primary=placeholder_launch_tier_key(primary_connectivity),
                weak=placeholder_launch_tier_key(weak_connectivity),
                classifier_model=getattr(args, "classifier_model", None),
                profile_name=getattr(args, "profile", None),
                classifier_min_confidence=getattr(
                    args, "classifier_min_confidence", None,
                ),
                backend_format=BackendFormat.OPENAI,
                timeout=args.timeout,
            )
            dry_run_route = LaunchRouteConfig(
                type="deterministic",
                model=deterministic_config.strong.model,
                weak_model=deterministic_config.weak.model,
            )
            dry_run_classifier_model = deterministic_config.classifier.model
            dry_run_profile = deterministic_config.profile_name
            dry_run_min_confidence = deterministic_config.classifier_min_confidence
        print(format_dry_run(
            target="codex",
            route=dry_run_route,
            base_url=primary_connectivity.base_url,
            api_key_set=bool(primary_connectivity.api_key),
            port=args.port,
            timeout=args.timeout,
            forwarded_args=codex_args,
            classifier_model=dry_run_classifier_model,
            profile=dry_run_profile,
            classifier_min_confidence=dry_run_min_confidence,
            routing_profiles=routing_profiles,
        ))
        return

    intake = resolve_launch_intake_config(
        args, target="codex", default_app="codex-switchyard",
    )

    if deterministic:
        _reject_openrouter_only_default_trio(
            target="codex",
            base_url=primary_connectivity.base_url,
        )
        primary_connectivity = require_launch_tier_key(
            target="codex",
            tier=PRIMARY_TIER,
            connectivity=primary_connectivity,
        )
        weak_connectivity = placeholder_launch_tier_key(weak_connectivity)
        deterministic_config = build_deterministic_routing_config(
            LaunchRouteConfig(
                type="deterministic",
                model=getattr(args, "model", None),
                weak_model=getattr(args, "weak_model", None),
            ),
            primary=primary_connectivity,
            weak=weak_connectivity,
            classifier_model=getattr(args, "classifier_model", None),
            profile_name=getattr(args, "profile", None),
            classifier_min_confidence=getattr(
                args, "classifier_min_confidence", None,
            ),
            backend_format=BackendFormat.OPENAI,
            timeout=args.timeout,
        )
        raise SystemExit(launch_codex_deterministic_routing(
            config=deterministic_config,
            port=args.port,
            codex_args=codex_args,
            intake=intake,
            discovery_disabled=bool(getattr(args, "no_model_discovery", False)),
            rl_log_dir=resolve_rl_log_dir(args),
        ))

    if routing_profiles:
        initial = _resolve_initial_from_profiles(
            target="codex",
            routing_profiles=routing_profiles,
        )
    else:
        primary_connectivity = require_launch_tier_key(
            target="codex",
            tier=PRIMARY_TIER,
            connectivity=primary_connectivity,
        )
        initial = require_route_model(route, target="codex")
    raise SystemExit(launch_codex(
        model=initial,
        base_url=primary_connectivity.base_url or "",
        api_key=primary_connectivity.api_key or "dummy",
        port=args.port,
        timeout=args.timeout,
        codex_args=codex_args,
        intake=intake,
        routing_profiles=routing_profiles,
        rl_log_dir=resolve_rl_log_dir(args),
    ))


def cmd_launch_openclaw(args: argparse.Namespace) -> None:
    """Start a proxy and spawn ``openclaw chat`` against it.

    Same shape as :func:`cmd_launch_claude` / :func:`cmd_launch_codex`:
    LLM-classifier routing by default, or single ``--model`` /
    ``--routing-profiles FILE`` (global flag) to opt out. The launcher writes a transient
    ``openclaw.json`` in a tempdir and points OpenClaw at it via
    ``OPENCLAW_STATE_DIR`` / ``OPENCLAW_HOME`` / ``OPENCLAW_CONFIG_PATH``
    — the user's real ``~/.openclaw/`` (sessions, channels, plugins)
    stays untouched.

    ``--smoke`` dispatches to :func:`verify_openclaw` and exits with its
    return code — same shape as ``launch claude/codex --smoke``.
    """
    if args.routing_profiles and getattr(args, "smoke", False):
        raise SystemExit(
            "launch openclaw: --smoke and --routing-profiles cannot be combined. "
            "Pass --model directly to pick the model to smoke-test."
        )
    if args.dry_run and getattr(args, "smoke", False):
        raise SystemExit(
            "launch openclaw: --smoke and --dry-run cannot be combined. "
            "--smoke runs a live round-trip; --dry-run would be ignored."
        )
    if getattr(args, "smoke", False) and not args.model:
        raise SystemExit(
            "launch openclaw: --smoke requires --model. "
            "Pass --model directly to pick the model to smoke-test."
        )
    if args.routing_profiles and args.model:
        raise SystemExit(
            "launch openclaw: --model and --routing-profiles are mutually exclusive.\n"
            "Pass --routing-profiles as a global flag before the subcommand:\n"
            "  switchyard --routing-profiles FILE -- launch openclaw"
        )
    from switchyard.cli.launchers.launcher_runtime import ensure_system_ssl_trust
    from switchyard.cli.launchers.openclaw_launcher import (
        launch_openclaw,
        launch_openclaw_deterministic_routing,
    )

    ensure_system_ssl_trust()

    routing_profiles = _resolve_routing_profiles(args)
    if not args.dry_run and not getattr(args, "smoke", False):
        maybe_bootstrap_launch_config(
            args,
            target="openclaw",
            api_key_env_vars=(
                "OPENROUTER_API_KEY",
                "NVIDIA_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
            ),
        )
    configured_launch = load_user_config().launch_target("openclaw")
    configured_route = configured_launch.effective_route()
    _warn_if_legacy_random_config(configured_route, target="openclaw")
    if args.routing_profiles:
        configured_route = LaunchRouteConfig()
    route = route_from_launch_args(args, configured_route)
    # LLM-classifier routing is the implicit default for ``launch openclaw``
    # when neither single-model passthrough nor a routing-profiles bundle is
    # in play.
    deterministic = _is_deterministic_launch(
        target="openclaw", args=args, route=route, routing_profiles=routing_profiles,
    )
    if not deterministic and not (routing_profiles and not route.model):
        require_route_model(route, target="openclaw")

    api_key, base_url = resolve_launch_connectivity(
        args, api_key_env_vars=(
            "OPENROUTER_API_KEY",
            "NVIDIA_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ),
    )
    primary_connectivity = resolve_route_tier_connectivity(
        target="openclaw",
        route=route,
        tier=PRIMARY_TIER,
        default_api_key=api_key,
        default_base_url=base_url,
    )
    weak_connectivity = resolve_route_tier_connectivity(
        target="openclaw",
        route=route,
        tier=WEAK_TIER,
        default_api_key=api_key,
        default_base_url=base_url,
    )
    openclaw_args = strip_forwarded_args(args.openclaw_args)

    if getattr(args, "smoke", False):
        _run_launch_smoke(
            target="openclaw",
            model=route.model,
            connectivity=primary_connectivity,
            port=args.port,
            timeout=args.timeout,
        )

    if args.dry_run:
        dry_run_classifier_model: str | None = None
        dry_run_profile: str | None = None
        dry_run_min_confidence: float | None = None
        dry_run_route = route
        if deterministic:
            deterministic_config = build_deterministic_routing_config(
                LaunchRouteConfig(
                    type="deterministic",
                    model=getattr(args, "model", None),
                    weak_model=getattr(args, "weak_model", None),
                ),
                primary=placeholder_launch_tier_key(primary_connectivity),
                weak=placeholder_launch_tier_key(weak_connectivity),
                classifier_model=getattr(args, "classifier_model", None),
                profile_name=getattr(args, "profile", None),
                classifier_min_confidence=getattr(
                    args, "classifier_min_confidence", None,
                ),
                backend_format=BackendFormat.OPENAI,
                timeout=args.timeout,
            )
            dry_run_route = LaunchRouteConfig(
                type="deterministic",
                model=deterministic_config.strong.model,
                weak_model=deterministic_config.weak.model,
            )
            dry_run_classifier_model = deterministic_config.classifier.model
            dry_run_profile = deterministic_config.profile_name
            dry_run_min_confidence = deterministic_config.classifier_min_confidence
        print(format_dry_run(
            target="openclaw",
            route=dry_run_route,
            base_url=primary_connectivity.base_url,
            api_key_set=bool(primary_connectivity.api_key),
            port=args.port,
            timeout=args.timeout,
            forwarded_args=openclaw_args,
            classifier_model=dry_run_classifier_model,
            profile=dry_run_profile,
            classifier_min_confidence=dry_run_min_confidence,
            routing_profiles=routing_profiles,
        ))
        return

    intake = resolve_launch_intake_config(
        args, target="openclaw", default_app="openclaw-switchyard",
    )

    if deterministic:
        _reject_openrouter_only_default_trio(
            target="openclaw",
            base_url=primary_connectivity.base_url,
        )
        primary_connectivity = require_launch_tier_key(
            target="openclaw",
            tier=PRIMARY_TIER,
            connectivity=primary_connectivity,
        )
        weak_connectivity = placeholder_launch_tier_key(weak_connectivity)
        deterministic_config = build_deterministic_routing_config(
            LaunchRouteConfig(
                type="deterministic",
                model=getattr(args, "model", None),
                weak_model=getattr(args, "weak_model", None),
            ),
            primary=primary_connectivity,
            weak=weak_connectivity,
            classifier_model=getattr(args, "classifier_model", None),
            profile_name=getattr(args, "profile", None),
            classifier_min_confidence=getattr(
                args, "classifier_min_confidence", None,
            ),
            backend_format=BackendFormat.OPENAI,
            timeout=args.timeout,
        )
        raise SystemExit(launch_openclaw_deterministic_routing(
            config=deterministic_config,
            port=args.port,
            openclaw_args=openclaw_args,
            intake=intake,
            discovery_disabled=bool(getattr(args, "no_model_discovery", False)),
            rl_log_dir=resolve_rl_log_dir(args),
        ))

    if routing_profiles:
        initial = _resolve_initial_from_profiles(
            target="openclaw",
            routing_profiles=routing_profiles,
        )
    else:
        primary_connectivity = require_launch_tier_key(
            target="openclaw",
            tier=PRIMARY_TIER,
            connectivity=primary_connectivity,
        )
        initial = require_route_model(route, target="openclaw")
    raise SystemExit(launch_openclaw(
        model=initial,
        base_url=primary_connectivity.base_url or "",
        api_key=primary_connectivity.api_key or "dummy",
        port=args.port,
        timeout=args.timeout,
        openclaw_args=openclaw_args,
        intake=intake,
        routing_profiles=routing_profiles,
        rl_log_dir=resolve_rl_log_dir(args),
    ))


def _warn_if_legacy_random_config(
    route: LaunchRouteConfig, target: str,
) -> None:
    """Surface a clear error when saved config is still on random-routing.

    The ``random-routing`` CLI subcommand and ``--routing random`` /
    ``--preset`` / ``--weak-model`` launcher flags were removed; routing
    policies live in routing-profile YAML files. Saved configs from
    earlier versions may still declare ``route.type = "random"``; bail
    with a recovery hint instead of silently launching the wrong shape.
    """
    if route.type != "random":
        return
    raise SystemExit(
        f"launch {target}: saved config has route.type=\"random\" but\n"
        "random-routing CLI flags have been removed. Express your routing\n"
        "policy as a routing-profile YAML and pass it via\n"
        "  --routing-profiles PATH\n"
        "or run `switchyard configure` to reset the saved route to single.\n"
        "See `switchyard serve --help` and the docs for the YAML schema."
    )


__all__ = [
    "cmd_launch_claude",
    "cmd_launch_codex",
    "cmd_launch_openclaw",
    "launch_requirements_satisfied",
    "maybe_bootstrap_launch_config",
    "resolve_launch_connectivity",
    "resolve_launch_intake_config",
    "resolve_route_tier_connectivity",
    "route_from_launch_args",
    "route_from_random_config",
]
