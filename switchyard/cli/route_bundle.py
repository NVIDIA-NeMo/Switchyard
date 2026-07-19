# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a model-dispatch table from a YAML route bundle.

Each YAML route becomes one entry in a :class:`RouteTable`. Chains are
built through the shared :mod:`switchyard.lib.route_table_builders` helpers — the
same path the Claude/Codex launchers take — so the table has uniform shape
no matter which front-end produced it.

The flow is::

    raw dict (from YAML / programmatic)
        │
        ▼  _parse_route_bundle_dict
    RouteBundle (defaults + routes + optional pre/post processors)
        │
        ▼  build_table_from_bundle
    RouteTable

Launchers skip the parser and construct a :class:`RouteBundle` directly so
argparse-driven and YAML-driven assembly share one builder.
"""

import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast, overload

from switchyard.cli.model_catalog.model_discovery import fetch_model_ids
from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target
from switchyard.lib.config import LatencyServiceBackendConfig, LatencyServiceEndpoint
from switchyard.lib.processors.llm_classifier import DEFAULT_MAX_REQUEST_CHARS
from switchyard.lib.processors.llm_classifier.presets import PROFILE_FACTORIES
from switchyard.lib.profiles import (
    DeterministicRoutingProfileConfig,
    LatencyServiceProfileConfig,
    PlanExecuteProfileConfig,
    ProfileSwitchyard,
    RouteLLMProfileConfig,
    StageRouterProfileConfig,
)
from switchyard.lib.profiles.deterministic_routing_config import DeterministicRoutingConfig
from switchyard.lib.profiles.plan_execute_config import PlanExecuteConfig
from switchyard.lib.profiles.plan_execute_presets import PlanExecutePresets
from switchyard.lib.profiles.random_routing import RandomRoutingConfig
from switchyard.lib.profiles.routellm import RouteLLMConfig
from switchyard.lib.profiles.stage_router_config import StageRouterConfig
from switchyard.lib.route_table import ChainRuntime, RouteTable
from switchyard.lib.route_table_builders import (
    build_passthrough_table,
    build_random_routing_switchyard,
    build_random_routing_table,
    build_tier_passthrough_switchyard,
)
from switchyard.lib.stats_accumulator import StatsAccumulator

logger = logging.getLogger(__name__)


def _default_discovery_fn(base_url: str, api_key: str) -> list[str]:
    """Wrap provider catalog fetching for the lib-level callable shape.

    Same wrapping pattern the Claude/Codex launchers use, so a YAML route's
    tier catalogs hydrate the same way the launchers do. Failures intentionally
    raise so table builders can preserve non-fatal warning metadata for
    ``/v1/models``.
    """
    return fetch_model_ids(base_url, api_key)


def _merge_table(
    table: RouteTable,
    sub_table: RouteTable,
) -> None:
    """Merge entries, warnings, and listing defaults from *sub_table*."""
    was_empty = not table.registered_models()
    for sub_model, sub_chain, sub_metadata in sub_table.items():
        table.register(sub_model, sub_chain, metadata=sub_metadata)
    for warning in sub_table.model_listing_warnings():
        table.add_model_listing_warning(warning)
    if was_empty:
        default_model = sub_table.default_model()
        if default_model is not None and default_model in table.registered_models():
            table.set_default_model(default_model)


@dataclass(frozen=True)
class RouteBundle:
    """Parsed route-bundle config — input to :func:`build_table_from_bundle`.

    The YAML loader produces one of these. The ``routes`` map is keyed by
    inbound model id (the YAML route key) and each value is a ``dict``-shaped
    route description matching the YAML schema.

    ``stats_accumulator`` lets callers thread their own accumulator into the
    builder so a launcher that's merging a YAML table on top of its own
    can share the same instance with downstream readers (live stats footer,
    ``/v1/routing/stats``). ``None`` makes the builder create a fresh one.

    Per-chain processor injection (Intake telemetry, custom hooks) is
    call-site runtime state, not bundle data. ``serve`` and launchers may pass
    processors through the table builder, but route YAML never declares
    those processors itself.
    """

    defaults: Mapping[str, Any] = field(default_factory=dict)
    routes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    stats_accumulator: StatsAccumulator | None = None


def llm_target_to_route_dict(target: LlmTarget) -> dict[str, Any]:
    """Convert an :class:`LlmTarget` into the dict shape a route bundle accepts.

    Used by launchers to construct :class:`RouteBundle` route entries from
    already-built tier targets without serializing through YAML. Only emits
    fields with non-default values so the loader's defaults cascade still
    applies.
    """
    data: dict[str, Any] = {
        "model": target.model,
        "format": str(target.format),
    }
    if target.endpoint.api_key:
        data["api_key"] = target.endpoint.api_key
    if target.endpoint.base_url:
        data["base_url"] = target.endpoint.base_url
    if target.endpoint.timeout_secs is not None:
        data["timeout_secs"] = target.endpoint.timeout_secs
    return data

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_TARGET_DEFAULT_KEYS = frozenset({
    "api_key",
    "base_url",
    "format",
    "backend_format",
    "timeout",
    "timeout_secs",
    "extra_body",
    "extra_headers",
    "endpoint",
})
_COMMON_ROUTE_KEYS = frozenset({
    "type",
    "kind",
    "defaults",
    "display_name",
    "description",
})
_ROUTE_METADATA_KEYS = frozenset({
    "type",
    "kind",
    "display_name",
    "description",
})
_TARGET_KEYS = _TARGET_DEFAULT_KEYS | frozenset({
    "id",
    "model",
    "tuning",
})
_TARGET_DEFAULT_ROUTE_KEYS = _TARGET_DEFAULT_KEYS | frozenset({"defaults"})
_MODEL_ROUTE_KEYS = _ROUTE_METADATA_KEYS | _TARGET_DEFAULT_ROUTE_KEYS | frozenset({
    "target",
    "model",
    # Opts this route's single target into token capture. Applied to the target
    # before coercion.
    "token_capture_engine",
    # Opts this route into token-continuity injection (requires
    # token_capture_engine). Parser names mirror the vLLM server flags.
    "token_injection",
    "tool_parser",
    "reasoning_parser",
})
_RANDOM_ROUTING_ROUTE_KEYS = _ROUTE_METADATA_KEYS | _TARGET_DEFAULT_ROUTE_KEYS | frozenset({
    "strong",
    "weak",
    "strong_probability",
    "enable_stats",
    "rng_seed",
    "preset",
    "fallback_target_on_evict",
})
_ROUTELLM_ROUTE_KEYS = _ROUTE_METADATA_KEYS | _TARGET_DEFAULT_ROUTE_KEYS | frozenset({
    "strong",
    "weak",
    "threshold",
    "router_type",
    "classifier_model",
    "enable_stats",
    "fallback_target_on_evict",
})
_PASSTHROUGH_SETTING_KEYS = frozenset({
    "api_key",
    "base_url",
    "timeout",
    "timeout_secs",
})
_PASSTHROUGH_ROUTE_KEYS = (
    _ROUTE_METADATA_KEYS
    | frozenset({
        "defaults",
        "enable_stats",
        "token_capture_engine",
        "token_injection",
        "tool_parser",
        "reasoning_parser",
    })
    | _PASSTHROUGH_SETTING_KEYS
)
_LATENCY_ENDPOINT_DEFAULT_KEYS = frozenset({
    "api_key",
    "base_url",
    "timeout",
    "timeout_secs",
})
_LATENCY_ENDPOINT_KEYS = _LATENCY_ENDPOINT_DEFAULT_KEYS | frozenset(
    {"model", "upstream_model", "request_type"}
)
_LATENCY_SERVICE_ROUTE_KEYS = _ROUTE_METADATA_KEYS | frozenset({
    "defaults",
    "endpoints",
    "latency_service_url",
    "latency_url",
    "poll_interval_s",
    "poll_timeout_s",
    "max_retries",
    "credential_policy",
    "enable_stats",
    "session_affinity",
    "affinity_max_sessions",
    "affinity_store",
    "affinity_store_url",
    "affinity_store_ttl_seconds",
    "affinity_key_prefix",
}) | _LATENCY_ENDPOINT_DEFAULT_KEYS
_NOOP_ROUTE_KEYS = _ROUTE_METADATA_KEYS
_DETERMINISTIC_ROUTE_KEYS = (
    _ROUTE_METADATA_KEYS
    | _TARGET_DEFAULT_ROUTE_KEYS
    | frozenset({
        "classifier",
        "strong",
        "weak",
        "profile",
        "enable_stats",
        "fallback_target_on_evict",
        "tier_timeout_s",
        "session_affinity",
        "affinity_max_sessions",
        "affinity_warmup_turns",
    })
)
_STAGE_ROUTER_ROUTE_KEYS = (
    _ROUTE_METADATA_KEYS
    | _TARGET_DEFAULT_ROUTE_KEYS
    | frozenset({
        "strong",
        "weak",
        "picker",
        "confidence_threshold",
        "signal_recent_window",
        "classifier",
        "enable_stats",
        "fallback_target_on_evict",
    })
)
_PLAN_EXECUTE_ROUTE_KEYS = (
    _ROUTE_METADATA_KEYS
    | _TARGET_DEFAULT_ROUTE_KEYS
    | frozenset({
        "planner",
        "executor",
        "cadence_n",
        "disable_reasoning",
        "fail_open",
        "enable_stats",
        "fallback_target_on_evict",
    })
)
_DETERMINISTIC_CLASSIFIER_KEYS = frozenset({
    "model",
    "api_key",
    "base_url",
    "timeout",
    "timeout_secs",
    "min_confidence",
    "fail_open",
    "recent_turn_window",
    "prompt",
    "max_request_chars",
})
_STAGE_ROUTER_CLASSIFIER_KEYS = frozenset({
    "model",
    "api_key",
    "base_url",
    "timeout",
    "timeout_secs",
    "recent_turn_window",
})
_CLASSIFIER_DEFAULT_KEYS = frozenset({
    "api_key",
    "base_url",
    "timeout",
    "timeout_secs",
})
_ROUTE_KEYS_BY_TYPE: Mapping[str, frozenset[str]] = {
    "model": _MODEL_ROUTE_KEYS,
    "random_routing": _RANDOM_ROUTING_ROUTE_KEYS,
    "routellm": _ROUTELLM_ROUTE_KEYS,
    "latency_service": _LATENCY_SERVICE_ROUTE_KEYS,
    "noop": _NOOP_ROUTE_KEYS,
    "passthrough": _PASSTHROUGH_ROUTE_KEYS,
    "deterministic": _DETERMINISTIC_ROUTE_KEYS,
    "stage_router": _STAGE_ROUTER_ROUTE_KEYS,
    "plan_execute": _PLAN_EXECUTE_ROUTE_KEYS,
}
_DEFAULT_KEYS_BY_TYPE: Mapping[str, frozenset[str]] = {
    "model": _TARGET_DEFAULT_KEYS,
    "random_routing": _TARGET_DEFAULT_KEYS,
    "routellm": _TARGET_DEFAULT_KEYS,
    "latency_service": _LATENCY_ENDPOINT_DEFAULT_KEYS,
    "passthrough": _PASSTHROUGH_SETTING_KEYS,
    "noop": frozenset(),
    "deterministic": _TARGET_DEFAULT_KEYS,
    "stage_router": _TARGET_DEFAULT_KEYS,
    "plan_execute": _TARGET_DEFAULT_KEYS,
}

# Shipping planner/executor model ids for `type: plan_execute` routes, so an
# omitted tier reproduces the retired `--plan-execute` flag. ``api_key=""`` is a
# placeholder — only the tier `.model` strings are read; credentials cascade
# from the route defaults.
_PLAN_EXECUTE_DEFAULTS = PlanExecutePresets.coding_agent_default(api_key="")


def _deterministic_profile_factories() -> dict[
    str, Any,
]:
    """LLM-classifier presets recognized by ``type: deterministic`` routes."""
    return dict(PROFILE_FACTORIES)


_DETERMINISTIC_PROFILE_FACTORIES = _deterministic_profile_factories()


class RouteBundleConfigError(ValueError):
    """Raised when a YAML route bundle is malformed."""


class _YamlModule(Protocol):
    YAMLError: type[Exception]

    def safe_load(self, stream: str) -> object: ...


#: Tier fields drilled into by :func:`routing_profile_model_ids`. The
#: ``classifier`` tier is intentionally NOT surfaced — it is an internal-only
#: LLM call, not a user-facing target.
_USER_FACING_TIER_FIELDS: tuple[str, ...] = (
    "strong", "weak", "planner", "executor", "target",
)


def parse_routing_profiles_file(path: str | Path) -> dict[str, object]:
    """Read *path* and return the parsed YAML dict (no env-var expansion)."""
    yaml = cast(_YamlModule, import_module("yaml"))
    resolved = Path(path)
    try:
        raw = resolved.read_text(encoding="utf-8")
        loaded = yaml.safe_load(raw)
    except FileNotFoundError as exc:
        raise RouteBundleConfigError(f"{resolved}: file not found") from exc
    except (OSError, UnicodeError) as exc:
        raise RouteBundleConfigError(
            f"{resolved}: cannot read: {_format_exception_one_line(exc)}"
        ) from exc
    except yaml.YAMLError as exc:
        raise RouteBundleConfigError(
            f"{resolved}: invalid YAML: {_format_exception_one_line(exc)}"
        ) from exc
    return loaded if isinstance(loaded, dict) else {}


def _format_exception_one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())


def routing_profile_model_ids(
    routing_profiles: Mapping[str, object] | None,
) -> list[str]:
    """User-callable model ids from a parsed routing-profiles bundle.

    Returns each route's YAML key followed by its tier ``model`` fields
    (``strong`` / ``weak`` for stage_router/deterministic/routellm/random_routing,
    ``planner`` / ``executor`` for plan_execute, ``target`` for
    ``model`` / ``passthrough``). Declaration order, later duplicates dropped.
    Returns ``[]`` for a ``None`` or empty bundle.

    Used by both the configure wizard (preview the picker) and the launchers
    (validate ``--model`` against the YAML without paying the cost of a full
    table build + catalog discovery).
    """
    if not routing_profiles:
        return []
    routes = routing_profiles.get("routes")
    if not isinstance(routes, Mapping):
        return []
    seen: set[str] = set()
    ordered: list[str] = []

    def _add(value: object) -> None:
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            ordered.append(value)

    for route_id, route in routes.items():
        _add(route_id)
        if not isinstance(route, Mapping):
            continue
        for tier_field in _USER_FACING_TIER_FIELDS:
            tier = route.get(tier_field)
            if isinstance(tier, Mapping):
                _add(tier.get("model"))
            elif isinstance(tier, str):
                _add(tier)
    return ordered


def load_route_bundle_table(
    path: str | Path,
    stats_accumulator: StatsAccumulator | None = None,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
    token_capture_enabled: bool = False,
) -> RouteTable:
    """Load *path* and return a table keyed by route model id.

    Pass ``stats_accumulator`` when a caller (typically a launcher) is merging
    this YAML table on top of one it already built, so YAML-declared chains
    record into the same accumulator surfaced at ``/v1/routing/stats`` and the
    live stats footer. ``None`` (default) creates a fresh accumulator — what
    standalone ``switchyard serve`` callers want.

    ``pre_routing_request_processors`` / ``extra_response_processors`` let
    callers attach process-level components such as Intake telemetry to every
    YAML-declared route.

    ``token_capture_enabled`` honors routes' ``token_capture_engine`` declarations
    (deriving token-capture request params); when ``False`` the key is ignored
    with a warning so clients never receive token fields nobody will capture.
    """
    return build_route_bundle_table(
        parse_routing_profiles_file(path),
        stats_accumulator=stats_accumulator,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
        token_capture_enabled=token_capture_enabled,
    )


def build_route_bundle_table(
    raw: object,
    stats_accumulator: StatsAccumulator | None = None,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
    token_capture_enabled: bool = False,
) -> RouteTable:
    """Parse *raw* dict and build a :class:`RouteTable`.

    Thin entrypoint that the YAML loader and any other dict-driven caller
    uses. Parses + env-expands + validates the dict into a :class:`RouteBundle`,
    then delegates to :func:`build_table_from_bundle`. Optional
    ``stats_accumulator`` overrides the parsed bundle's default ``None``.
    """
    if not token_capture_enabled:
        raw = _strip_token_capture_engine(raw)
    bundle = _parse_route_bundle_dict(raw)
    if stats_accumulator is not None:
        bundle = replace(bundle, stats_accumulator=stats_accumulator)
    return build_table_from_bundle(
        bundle,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
    )


def route_bundle_declares_token_capture(raw_or_path: object) -> bool:
    """True when the bundle declares ``token_capture_engine`` on any route.

    Accepts either a path to a routing-profiles YAML file or an
    already-parsed bundle dict. Token capture activates under
    ``--enable-rl-logging`` when at least one route opts in via
    ``token_capture_engine``. A bundle that cannot be parsed reports ``False`` —
    the real table load surfaces the parse error.
    """
    if raw_or_path is None:
        return False
    raw: object
    if isinstance(raw_or_path, str | Path):
        try:
            raw = parse_routing_profiles_file(raw_or_path)
        except RouteBundleConfigError:
            return False
    else:
        raw = raw_or_path

    def _walk(node: object) -> bool:
        if isinstance(node, Mapping):
            return "token_capture_engine" in node or any(_walk(value) for value in node.values())
        if isinstance(node, list):
            return any(_walk(item) for item in node)
        return False

    return _walk(raw)


def _strip_token_capture_engine(raw: object) -> object:
    """Drop ``token_capture_engine`` keys when token capture is disabled.

    The field derives token-capture request params into the route target's
    ``extra_body`` (see ``llm_target_with_token_capture``); honoring it while
    RL logging is off would make clients receive token fields that nothing
    captures. Warns once so the config mismatch is visible.
    """

    stripped = 0

    def _walk(node: object) -> object:
        nonlocal stripped
        if isinstance(node, Mapping):
            out: dict[str, object] = {}
            for key, value in node.items():
                if key == "token_capture_engine":
                    stripped += 1
                    continue
                # Injection rides on capture; without capture it must not
                # activate (its records would go unstored).
                if key == "token_injection":
                    continue
                out[key] = _walk(value)
            return out
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    result = _walk(raw)
    if stripped:
        logger.warning(
            "route bundle declares token_capture_engine on %d route(s) but token "
            "capture is disabled; ignoring (enable with --enable-rl-logging)",
            stripped,
        )
    return result


def _parse_route_bundle_dict(raw: object) -> RouteBundle:
    """Validate *raw* and return a :class:`RouteBundle`.

    Performs env-var expansion, top-level-key validation, and the
    "routes must contain at least one route" check. Per-route schema
    validation runs inside :func:`build_table_from_bundle` so callers
    that construct a bundle programmatically still benefit.
    """
    bundle = _require_mapping(_expand_env(raw), "route bundle")
    _validate_allowed_keys(bundle, frozenset({"defaults", "routes"}), "route bundle")
    defaults = _optional_mapping(bundle.get("defaults", {}), "defaults")
    _validate_allowed_keys(defaults, _TARGET_DEFAULT_KEYS, "defaults")
    routes_raw = _require_mapping(bundle.get("routes"), "routes")
    if not routes_raw:
        raise RouteBundleConfigError("routes must contain at least one route")
    routes: dict[str, Mapping[str, Any]] = {
        name: _require_mapping(spec, f"routes.{name}")
        for name, spec in routes_raw.items()
    }
    return RouteBundle(defaults=defaults, routes=routes)


def build_table_from_bundle(
    bundle: RouteBundle,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> RouteTable:
    """Build a :class:`RouteTable` from a :class:`RouteBundle`.

    Single assembly path for both the YAML loader and the launchers.

    If ``bundle.stats_accumulator`` is set, the same accumulator is shared
    across every produced chain so the live stats footer and
    ``/v1/routing/stats`` see consistent numbers.

    ``pre_routing_request_processors`` / ``extra_response_processors`` are
    call-time kwargs (not bundle data). ``serve`` and launchers pass
    process-level processors such as Intake here when CLI/env config enables
    them; YAML routes never declare those processors themselves.
    """
    table = RouteTable()
    stats = bundle.stats_accumulator or StatsAccumulator()
    for model_id, route_raw in bundle.routes.items():
        if not isinstance(model_id, str) or not model_id:
            raise RouteBundleConfigError("routes keys must be non-empty strings")
        route = _normalize_route(model_id, route_raw)
        route_type = _route_type(model_id, route)
        _validate_route_keys(model_id, route, route_type)
        route_defaults = _target_defaults(bundle.defaults, route)

        # `random_routing` always expands into the launcher-shaped N+1 entries:
        # each tier's configured model registered as a direct passthrough +
        # a virtual routing-policy id under the route's YAML key, plus each
        # tier's GET /v1/models catalog hydrated into the same table. This
        # matches the Claude/Codex launcher behavior so client model pickers
        # always see strong/weak (and the rest of the catalog) as direct
        # overrides.
        if route_type == "random_routing":
            _merge_random_routing_route(
                table, model_id, route,
                target_defaults=route_defaults, stats=stats,
                pre_routing_request_processors=pre_routing_request_processors,
                extra_response_processors=extra_response_processors,
            )
            continue

        # `passthrough` routes hydrate their single tier's catalog into the
        # table alongside the configured target model. (`model` routes are
        # pure aliases — they register under the route key only, no catalog.)
        if route_type == "passthrough":
            _merge_discovered_single_tier(
                table,
                model_id,
                route,
                route_type=route_type,
                target_defaults=route_defaults,
                stats=stats,
                pre_routing_request_processors=pre_routing_request_processors,
                extra_response_processors=extra_response_processors,
            )
            continue

        # `stage_router` and `deterministic` routes register the routing-policy chain
        # at the route key AND hydrate each tier's catalog (`strong` + `weak`)
        # into the table as direct passthroughs — same client-facing
        # model-picker experience as random_routing's discovery path.
        if route_type in ("stage_router", "deterministic"):
            _merge_multi_target_discovery(
                table,
                model_id,
                route,
                route_type=route_type,
                target_defaults=route_defaults,
                stats=stats,
                pre_routing_request_processors=pre_routing_request_processors,
                extra_response_processors=extra_response_processors,
            )
            continue

        switchyard = _build_switchyard_for_route(
            model_id,
            route,
            route_type=route_type,
            target_defaults=route_defaults,
            stats=stats,
            pre_routing_request_processors=pre_routing_request_processors,
            extra_response_processors=extra_response_processors,
        )
        table.register(
            model_id,
            switchyard,
            metadata=_route_metadata(model_id, route, route_type),
        )
    return table


def _merge_random_routing_route(
    table: RouteTable,
    model_id: str,
    route: Mapping[str, object],
    target_defaults: Mapping[str, object],
    stats: StatsAccumulator,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> None:
    """Expand a ``random_routing`` route into virtual id + tier passthroughs.

    Always goes through :func:`build_random_routing_table` from
    :mod:`switchyard.lib.route_table_builders` — the same path the Claude/Codex
    launchers take. Registration order matches the unified rule (YAML route
    key first, discovered/tier entries after):

      - The route's YAML key registered as the virtual routing-policy id.
      - Each tier's configured model registered as a direct passthrough.
      - Each tier's ``GET /v1/models`` catalog hydrated alongside.

    Tier registration and catalog hydration both happen unconditionally so
    client model pickers see strong/weak (and the rest of the catalog) as
    direct overrides for the random-routing default.
    """
    was_empty = not table.registered_models()
    random_config = RandomRoutingConfig.model_validate(
        _route_config(route, target_defaults, ("strong", "weak"))
    )
    sub_table = build_random_routing_table(
        config=random_config,
        stats=stats,
        random_routing_switchyard=build_random_routing_switchyard(
            random_config,
            stats,
            pre_routing_request_processors=pre_routing_request_processors,
            extra_response_processors=extra_response_processors,
        ),
        routing_model=model_id,
        discovery_fn=_default_discovery_fn,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
    )
    # Register the YAML key (the routing-policy virtual id) first so
    # `registered_models()[0]` matches the user's declared route key,
    # then the tier passthroughs + catalog entries the builder produced.
    for sub_model, sub_chain, sub_metadata in sorted(
        sub_table.items(), key=lambda item: item[0] != model_id,
    ):
        table.register(sub_model, sub_chain, metadata=sub_metadata)
    for warning in sub_table.model_listing_warnings():
        table.add_model_listing_warning(warning)
    if was_empty:
        default_model = sub_table.default_model()
        if default_model is not None and default_model in table.registered_models():
            table.set_default_model(default_model)


def _merge_discovered_single_tier(
    table: RouteTable,
    model_id: str,
    route: Mapping[str, object],
    route_type: str,
    target_defaults: Mapping[str, object],
    stats: StatsAccumulator,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> None:
    """Expand a ``passthrough`` route with catalog discovery.

    Builds a single ``LlmTarget`` from the route's fields, then goes through
    :func:`build_passthrough_table` with the shared
    :func:`_default_discovery_fn` — the same path the launcher's per-tier
    passthrough registration takes. Follows the unified ordering rule:

      - The YAML route key registered first (aliasing the tier's chain when
        the key differs from ``tier.model``).
      - The tier's configured model registered next.
      - Every catalog entry from the tier's ``GET /v1/models`` after that.
    """
    tier = _passthrough_target(model_id, route, target_defaults, route_type)
    sub_table = build_passthrough_table(
        (tier,),
        stats,
        enable_stats=_optional_bool(route.get("enable_stats"), default=True),
        discovery_fn=_default_discovery_fn,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
    )
    was_empty = not table.registered_models()
    items = list(sub_table.items())
    alias_registered = False
    if model_id != tier.model:
        tier_chain, tier_metadata = next(
            ((ch, md) for mid, ch, md in items if mid == tier.model),
            (None, {}),
        )
        if tier_chain is not None:
            # YAML key first as an alias to the tier's chain so
            # `registered_models()[0]` is always the user-declared route key.
            table.register(
                model_id,
                tier_chain,
                metadata={**dict(tier_metadata), "display_name": model_id},
            )
            alias_registered = True
    for sub_model, sub_chain, sub_metadata in items:
        if sub_model == model_id:
            if alias_registered:
                continue
            # When YAML key == tier.model this entry is the route key itself;
            # let it register here so it lands at position 0 of the table.
            table.register(sub_model, sub_chain, metadata=sub_metadata)
            continue
        table.register(sub_model, sub_chain, metadata=sub_metadata)
    for warning in sub_table.model_listing_warnings():
        table.add_model_listing_warning(warning)
    if was_empty and model_id in table.registered_models():
        table.set_default_model(model_id)


def _merge_multi_target_discovery(
    table: RouteTable,
    model_id: str,
    route: Mapping[str, object],
    *,
    route_type: str,
    target_defaults: Mapping[str, object],
    stats: StatsAccumulator,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> None:
    """Expand a ``stage_router``/``deterministic`` route with catalog discovery.

    Registers two layers, route key first so ``registered_models()[0]`` is the
    user-declared YAML key:

    1. The route's primary routing-policy chain at the route key (the regular
       stage_router/deterministic switchyard).
    2. Each tier (``strong`` + ``weak``) registered as a direct passthrough with
       its catalog hydrated via :func:`_default_discovery_fn` — same shape the
       launcher's per-tier registration produces, so client model pickers see
       strong/weak as standalone choices alongside the routing policy.

    The ``classifier`` tier (when present on the route) is intentionally not
    discovered — it is an internal-only LLM call, not a user-facing target.
    """
    was_empty = not table.registered_models()
    switchyard = _build_switchyard_for_route(
        model_id,
        route,
        route_type=route_type,
        target_defaults=target_defaults,
        stats=stats,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
    )
    table.register(
        model_id,
        switchyard,
        metadata=_route_metadata(model_id, route, route_type),
    )

    strong = _target_value(
        route.get("strong"), target_defaults, default_id="strong", where="strong",
    )
    weak = _target_value(
        route.get("weak"), target_defaults, default_id="weak", where="weak",
    )
    sub_table = build_passthrough_table(
        (strong, weak),
        stats,
        enable_stats=_optional_bool(route.get("enable_stats"), default=True),
        discovery_fn=_default_discovery_fn,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
    )
    for sub_model, sub_chain, sub_metadata in sub_table.items():
        # The route key wins over any discovered/configured model id collision.
        if sub_model == model_id:
            continue
        table.register(sub_model, sub_chain, metadata=sub_metadata)
    for warning in sub_table.model_listing_warnings():
        table.add_model_listing_warning(warning)
    if was_empty and model_id in table.registered_models():
        table.set_default_model(model_id)


def _build_switchyard_for_route(
    model_id: str,
    route: Mapping[str, object],
    route_type: str,
    target_defaults: Mapping[str, object],
    stats: StatsAccumulator,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    if route_type in ("model", "passthrough"):
        # Both kinds resolve to a single-tier passthrough chain — same shape the
        # launcher produces via build_passthrough_table's per-tier registration.
        # The "passthrough" kind synthesizes a target whose model defaults to the
        # route's table key when no explicit target/model is given.
        target = _passthrough_target(model_id, route, target_defaults, route_type)
        return build_tier_passthrough_switchyard(
            target,
            stats,
            enable_stats=_optional_bool(route.get("enable_stats"), default=True),
            extra_request_processors=pre_routing_request_processors,
            extra_response_processors=extra_response_processors,
            backend_wrapper=_token_injection_wrapper(model_id, route, target),
        )

    if route_type == "routellm":
        routellm_config = RouteLLMConfig.model_validate(
            _route_config(route, target_defaults, ("strong", "weak"))
        )
        return ProfileSwitchyard(
            RouteLLMProfileConfig.from_config(routellm_config)
            .build()
            .with_runtime_components(
                stats_accumulator=stats,
                enable_stats=routellm_config.enable_stats,
                pre_request_processors=pre_routing_request_processors,
                response_processors=extra_response_processors,
            )
        )

    if route_type == "latency_service":
        return _latency_service_switchyard(
            model_id,
            route,
            target_defaults,
            stats=stats,
            extra_request_processors=pre_routing_request_processors,
            extra_response_processors=extra_response_processors,
        )

    if route_type == "noop":
        from switchyard.lib.profiles.noop import NoopProfileConfig

        return ProfileSwitchyard(
            NoopProfileConfig()
            .build()
            .with_runtime_components(
                stats_accumulator=stats,
                pre_request_processors=pre_routing_request_processors,
                response_processors=extra_response_processors,
            )
        )

    if route_type == "deterministic":
        return _deterministic_switchyard(
            model_id,
            route,
            target_defaults=target_defaults,
            stats=stats,
            pre_routing_request_processors=pre_routing_request_processors,
            extra_response_processors=extra_response_processors,
        )

    if route_type == "stage_router":
        return _stage_router_switchyard(
            model_id,
            route,
            target_defaults=target_defaults,
            stats=stats,
            pre_routing_request_processors=pre_routing_request_processors,
            extra_response_processors=extra_response_processors,
        )

    if route_type == "plan_execute":
        return _plan_execute_switchyard(
            model_id,
            route,
            target_defaults=target_defaults,
            stats=stats,
            pre_routing_request_processors=pre_routing_request_processors,
            extra_response_processors=extra_response_processors,
        )

    raise RouteBundleConfigError(f"unsupported route type {route_type!r}")


def _deterministic_switchyard(
    model_id: str,
    route: Mapping[str, object],
    target_defaults: Mapping[str, object],
    stats: StatsAccumulator,
    pre_routing_request_processors: Sequence[Any],
    extra_response_processors: Sequence[Any],
) -> ChainRuntime:
    """Build the LLM-classifier deterministic-routing chain for a route.

    Wires the chain assembled from the LLM-classifier router primitives:

        StatsRequestProcessor
          → LLMClassifierRequestProcessor    (real LLM call → RouteSignals)
          → SignalTierSelectorRequestProcessor   (collapse to strong/weak)
          → DeterministicRoutingLLMBackend       (per-tier OpenAI backend)
          → DefaultResponseTranslator

    YAML schema::

        type: deterministic
        profile: general            # general | coding_agent | openclaw
        classifier:
          model: google/gemini-3.5-flash
          api_key: ${OPENROUTER_API_KEY}
          base_url: https://openrouter.ai/api/v1
          timeout_secs: 30.0
          min_confidence: 0.6       # tier-selector confidence floor
          fail_open: true           # on classifier error, route to strong
          recent_turn_window: 4
        strong:
          model: anthropic/claude-opus-4.7
          api_key: ${OPENROUTER_API_KEY}
          base_url: https://openrouter.ai/api/v1
        weak:
          model: moonshotai/kimi-k2.6
          api_key: ${OPENROUTER_API_KEY}
          base_url: https://openrouter.ai/api/v1
    """
    classifier_raw = route.get("classifier")
    if not isinstance(classifier_raw, Mapping):
        raise RouteBundleConfigError(
            f"route {model_id!r}: type=deterministic requires a `classifier:` "
            "mapping with model/api_key/base_url",
        )
    classifier = _classifier_mapping(
        classifier_raw,
        target_defaults,
        allowed_keys=_DETERMINISTIC_CLASSIFIER_KEYS,
        where=f"{model_id}.classifier",
    )

    strong = _target_value(
        route.get("strong"), target_defaults, default_id="strong", where="strong",
    )
    weak = _target_value(
        route.get("weak"), target_defaults, default_id="weak", where="weak",
    )

    fallback_target_on_evict = _required_str(
        route.get("fallback_target_on_evict"),
        f"{model_id}.fallback_target_on_evict",
    )
    valid_ids = {strong.id, weak.id}
    if fallback_target_on_evict not in valid_ids:
        raise RouteBundleConfigError(
            f"route {model_id!r}: fallback_target_on_evict="
            f"{fallback_target_on_evict!r} must match one of {sorted(valid_ids)} "
            f"(the configured strong/weak target ids)",
        )

    profile_name = _optional_str(route.get("profile")) or "general"
    if profile_name not in _DETERMINISTIC_PROFILE_FACTORIES:
        raise RouteBundleConfigError(
            f"route {model_id!r}: unknown profile {profile_name!r}; "
            f"expected one of {sorted(_DETERMINISTIC_PROFILE_FACTORIES)}",
        )

    config_data: dict[str, object] = {
        "strong": strong,
        "weak": weak,
        "classifier": {
            "id": "classifier",
            "model": _required_str(
                classifier.get("model"), f"{model_id}.classifier.model"
            ),
            "api_key": _required_str(
                classifier.get("api_key"), f"{model_id}.classifier.api_key"
            ),
            "base_url": _required_str(
                classifier.get("base_url"), f"{model_id}.classifier.base_url"
            ),
            "timeout_secs": _optional_float(
                classifier.get("timeout_secs"),
                default=30.0,
            ),
        },
        "fallback_target_on_evict": fallback_target_on_evict,
        "profile_name": profile_name,
        "classifier_min_confidence": _optional_float(
            classifier.get("min_confidence"), default=0.6,
        ),
        "classifier_fail_open": _optional_bool(
            classifier.get("fail_open"), default=True
        ),
        "classifier_recent_turn_window": _optional_int(
            classifier.get("recent_turn_window"), default=4,
        ),
        "classifier_system_prompt": _optional_str(classifier.get("prompt")),
        "classifier_max_request_chars": _optional_int(
            classifier.get("max_request_chars"),
            default=DEFAULT_MAX_REQUEST_CHARS,
        ),
        "classifier_timeout_s": _optional_float(
            classifier.get("timeout_secs"), default=30.0
        ),
        "enable_stats": _optional_bool(route.get("enable_stats"), default=True),
    }
    if "tier_timeout_s" in route:
        config_data["tier_timeout_s"] = _optional_float(
            route.get("tier_timeout_s"),
            default=None,
        )
    if "session_affinity" in route:
        config_data["session_affinity"] = _optional_bool(
            route.get("session_affinity"), default=False
        )
    if "affinity_max_sessions" in route:
        config_data["affinity_max_sessions"] = _optional_int(
            route.get("affinity_max_sessions"), default=10_000
        )
    if "affinity_warmup_turns" in route:
        config_data["affinity_warmup_turns"] = _optional_int(
            route.get("affinity_warmup_turns"), default=0
        )

    config = DeterministicRoutingConfig.model_validate(config_data)
    return ProfileSwitchyard(
        DeterministicRoutingProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            stats_accumulator=stats,
            enable_stats=config.enable_stats,
            pre_request_processors=pre_routing_request_processors,
            response_processors=extra_response_processors,
        )
    )


def _stage_router_switchyard(
    model_id: str,
    route: Mapping[str, object],
    target_defaults: Mapping[str, object],
    stats: StatsAccumulator,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    """Build a stage-router-routing Switchyard from a YAML ``type: stage_router`` route.

    Schema (mapped onto :class:`StageRouterConfig`)::

        route:
          type: stage_router
          picker: capable_first       # or efficient_first
          confidence_threshold: 0.7            # default; range [0.0, 1.0]
          signal_recent_window: 3              # Rust sliding-window size
          strong: <target spec>                # e.g. { id: strong, model: ..., api_key: ..., format: anthropic }
          weak:   <target spec>                # e.g. { id: weak,   model: ..., api_key: ..., format: openai }
          classifier:                          # optional; omit to skip the LLM fallback
            model: google/gemini-3.5-flash
            api_key: ${SWITCHYARD_CLASSIFIER_API_KEY}
            base_url: https://openrouter.ai/api/v1
            timeout_secs: 30.0
            recent_turn_window: 3
          enable_stats: true

    Each tier spec accepts the same shapes as other route types
    (``{ id, model, api_key, base_url, ... }`` or a model-id string);
    per-target tuning fields are honoured via :func:`_target_value`.
    """
    if route.get("strong") is None or route.get("weak") is None:
        raise RouteBundleConfigError(
            f"route {model_id!r}: stage_router route requires both 'strong' and "
            f"'weak' target specs",
        )
    if route.get("classifier") is not None:
        route = dict(route)
        route["classifier"] = _classifier_mapping(
            route["classifier"],
            target_defaults,
            allowed_keys=_STAGE_ROUTER_CLASSIFIER_KEYS,
            where=f"{model_id}.classifier",
        )
    # The deprecated bundle keeps the shared strong/weak tier keys (also used by
    # deterministic/routellm); map them onto StageRouterConfig's capable/efficient fields.
    resolved = _route_config(route, target_defaults, ("strong", "weak"))
    resolved["capable"] = resolved.pop("strong")
    resolved["efficient"] = resolved.pop("weak")
    stage_router_config = StageRouterConfig.model_validate(resolved)
    return ProfileSwitchyard(
        StageRouterProfileConfig.from_config(stage_router_config)
        .build()
        .with_runtime_components(
            stats_accumulator=stats,
            enable_stats=stage_router_config.enable_stats,
            pre_request_processors=pre_routing_request_processors,
            response_processors=extra_response_processors,
        )
    )


def _plan_execute_switchyard(
    model_id: str,
    route: Mapping[str, object],
    *,
    target_defaults: Mapping[str, object],
    stats: StatsAccumulator,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    """Build a strong-planner / weak-executor chain from a ``type: plan_execute`` route.

    Mirrors :func:`_stage_router_switchyard`: tiers and scalar fields go through the
    shared ``_route_config`` → :meth:`PlanExecuteConfig.model_validate` path. An
    omitted ``planner`` / ``executor`` defaults to the shipping preset model so a
    minimal route reproduces the retired ``--plan-execute`` flag; every other
    field falls back to ``PlanExecuteConfig``'s own defaults.
    """
    # Seed omitted tiers with the preset model before the shared coercion path
    # (`_route_config` would otherwise reject a missing tier).
    route = dict(route)
    if route.get("planner") is None:
        route["planner"] = _PLAN_EXECUTE_DEFAULTS.planner.model
    if route.get("executor") is None:
        route["executor"] = _PLAN_EXECUTE_DEFAULTS.executor.model
    config_data = _route_config(route, target_defaults, ("planner", "executor"))
    config_data.setdefault(
        "fallback_target_on_evict", cast(LlmTarget, config_data["planner"]).id,
    )
    config = PlanExecuteConfig.model_validate(config_data)
    return ProfileSwitchyard(
        PlanExecuteProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            stats_accumulator=stats,
            enable_stats=config.enable_stats,
            pre_request_processors=pre_routing_request_processors,
            response_processors=extra_response_processors,
        )
    )


def _passthrough_target(
    model_id: str,
    route: Mapping[str, object],
    target_defaults: Mapping[str, object],
    route_type: str,
) -> LlmTarget:
    """Resolve the LlmTarget for a ``model`` or ``passthrough`` route.

    ``model`` routes require an explicit ``target`` or ``model`` field. The
    ``passthrough`` kind is permissive: if no target is given, the route's
    table key becomes the model name and the rest of the target is
    populated from defaults plus inline route fields.
    """
    target_raw = route.get("target", route.get("model"))
    if target_raw is None:
        if route_type == "model":
            raise RouteBundleConfigError(f"route {model_id!r} requires target or model")
        target_raw = model_id
    target = _target_mapping(target_raw, target_defaults, default_id=model_id, where="target")
    # Apply the route's token_capture_engine to its single target.
    route_engine = route.get("token_capture_engine")
    if route_engine is not None:
        target["token_capture_engine"] = route_engine
    return coerce_llm_target(target, default_id=model_id)


def _token_injection_wrapper(
    model_id: str,
    route: Mapping[str, object],
    target: LlmTarget,
) -> Callable[[Any], Any] | None:
    """Backend wrapper enabling token-continuity injection, or ``None``.

    ``token_injection: true`` opts the route in; it requires
    ``token_capture_engine`` (injection's records must be captured) and an
    explicit target ``base_url`` for the direct ``/tokenize`` and
    ``/v1/completions`` calls. Parser names are optional and mirror the vLLM
    server's ``--tool-call-parser`` / ``--reasoning-parser`` flags.
    """
    if not route.get("token_injection"):
        return None
    if route.get("token_capture_engine") is None:
        raise RouteBundleConfigError(
            f"route {model_id!r}: token_injection requires token_capture_engine"
        )
    endpoint = target.endpoint
    base_url = endpoint.base_url if endpoint is not None else None
    if not base_url:
        raise RouteBundleConfigError(
            f"route {model_id!r}: token_injection requires an explicit base_url"
        )
    tool_parser = _optional_str(route["tool_parser"]) if "tool_parser" in route else None
    reasoning_parser = (
        _optional_str(route["reasoning_parser"]) if "reasoning_parser" in route else None
    )
    api_key = endpoint.api_key if endpoint is not None else None
    timeout_secs = (endpoint.timeout_secs if endpoint is not None else None) or 600.0
    model = target.model

    def _wrap(backend: Any) -> Any:
        from switchyard.lib.backends.token_injection_backend import TokenInjectionBackend

        return TokenInjectionBackend(
            backend,
            base_url=base_url,
            model=model,
            api_key=api_key,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            timeout_secs=timeout_secs,
        )

    return _wrap


def _latency_service_switchyard(
    model_id: str,
    route: Mapping[str, object],
    target_defaults: Mapping[str, object],
    stats: StatsAccumulator,
    extra_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    endpoints_raw = _require_sequence(route.get("endpoints"), "latency_service.endpoints")
    endpoints = [
        _latency_endpoint(value, target_defaults, index=index)
        for index, value in enumerate(endpoints_raw)
    ]
    enable_stats = _optional_bool(route.get("enable_stats"), default=True)
    credential_policy = (
        _optional_str(route["credential_policy"])
        if "credential_policy" in route
        else "configured_endpoint"
    )
    config = LatencyServiceBackendConfig.model_validate({
        "latency_service_url": _required_str(
            route.get("latency_service_url", route.get("latency_url")),
            "latency_service.latency_service_url",
        ),
        "endpoints": endpoints,
        # The YAML route key is what clients send as ``model``; hand it to the
        # backend so the per-model metric can attribute route-key traffic.
        "route_model": model_id,
        "poll_interval_s": _optional_float(route.get("poll_interval_s"), default=10.0),
        "poll_timeout_s": _optional_float(route.get("poll_timeout_s"), default=5.0),
        "max_retries": _optional_int(route.get("max_retries"), default=2),
        "credential_policy": credential_policy,
        "enable_stats": enable_stats,
        "session_affinity": _optional_bool(route.get("session_affinity"), default=False),
        "affinity_max_sessions": _optional_int(
            route.get("affinity_max_sessions"), default=10_000
        ),
        "affinity_store": _optional_str(route.get("affinity_store")) or "memory",
        "affinity_store_url": _optional_str(route.get("affinity_store_url")),
        "affinity_store_ttl_seconds": _optional_int(
            route.get("affinity_store_ttl_seconds"), default=3_600
        ),
        "affinity_key_prefix": _optional_str(route.get("affinity_key_prefix")) or "swyd:pin:",
    })
    return ProfileSwitchyard(
        LatencyServiceProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            stats_accumulator=stats,
            enable_stats=config.enable_stats,
            pre_request_processors=extra_request_processors,
            response_processors=extra_response_processors,
        )
    )


def _route_config(
    route: Mapping[str, object],
    target_defaults: Mapping[str, object],
    target_fields: tuple[str, ...],
) -> dict[str, object]:
    data = {
        key: value
        for key, value in route.items()
        if key not in _COMMON_ROUTE_KEYS
        and key not in _TARGET_DEFAULT_KEYS
    }
    for tier_field in target_fields:
        data[tier_field] = _target_value(
            route.get(tier_field),
            target_defaults,
            default_id=tier_field,
            where=tier_field,
        )
    return data


def _target_value(
    raw: object,
    defaults: Mapping[str, object],
    default_id: str,
    where: str,
) -> LlmTarget:
    if raw is None:
        raise RouteBundleConfigError(f"{where} target is required")
    return coerce_llm_target(
        _target_mapping(raw, defaults, default_id=default_id, where=where),
        default_id=default_id,
    )


def _classifier_mapping(
    raw: object,
    defaults: Mapping[str, object],
    *,
    allowed_keys: frozenset[str],
    where: str,
) -> dict[str, object]:
    classifier = {
        key: value
        for key, value in defaults.items()
        if key in _CLASSIFIER_DEFAULT_KEYS
    }
    classifier_mapping = _require_mapping(raw, where)
    _validate_allowed_keys(classifier_mapping, allowed_keys, where)
    classifier.update(classifier_mapping)
    if "timeout" in classifier_mapping and "timeout_secs" not in classifier_mapping:
        classifier["timeout_secs"] = classifier_mapping["timeout"]
    elif "timeout_secs" not in classifier and "timeout" in classifier:
        classifier["timeout_secs"] = classifier["timeout"]
    classifier.pop("timeout", None)
    return classifier


def _target_mapping(
    raw: object,
    defaults: Mapping[str, object],
    default_id: str,
    where: str,
) -> dict[str, object]:
    target = dict(defaults)
    if isinstance(raw, str):
        target["model"] = raw
    elif isinstance(raw, Mapping):
        target_mapping = _require_mapping(raw, where)
        _validate_allowed_keys(target_mapping, _TARGET_KEYS, where)
        target.update(target_mapping)
    else:
        raise RouteBundleConfigError(f"{where} must be a string or mapping")
    target.setdefault("id", default_id)
    return target


def _latency_endpoint(
    raw: object,
    defaults: Mapping[str, object],
    index: int,
) -> LatencyServiceEndpoint:
    data = dict(defaults)
    where = f"latency_service.endpoints[{index}]"
    if isinstance(raw, str):
        data["model"] = raw
    elif isinstance(raw, Mapping):
        endpoint_mapping = _require_mapping(raw, where)
        _validate_allowed_keys(endpoint_mapping, _LATENCY_ENDPOINT_KEYS, where)
        data.update(endpoint_mapping)
    else:
        raise RouteBundleConfigError(f"{where} must be a string or mapping")

    if "timeout" not in data and "timeout_secs" in data:
        data["timeout"] = data["timeout_secs"]

    endpoint_data = {
        key: data[key]
        for key in ("model", "upstream_model", "api_key", "base_url", "timeout", "request_type")
        if key in data
    }
    return LatencyServiceEndpoint.model_validate(endpoint_data)


def _target_defaults(
    bundle_defaults: Mapping[str, object],
    route: Mapping[str, object],
) -> dict[str, object]:
    defaults = dict(bundle_defaults)
    nested = route.get("defaults")
    if nested is not None:
        defaults.update(_require_mapping(nested, "route.defaults"))
    for key in _TARGET_DEFAULT_KEYS:
        if key in route:
            defaults[key] = route[key]
    return defaults


def _normalize_route(model_id: str, raw: object) -> Mapping[str, object]:
    if isinstance(raw, str):
        return {"type": "model", "target": raw}
    if isinstance(raw, Mapping):
        return _require_mapping(raw, f"route {model_id!r}")
    raise RouteBundleConfigError(f"route {model_id!r} must be a string or mapping")


def _route_type(model_id: str, route: Mapping[str, object]) -> str:
    raw_type = route.get("type", route.get("kind"))
    if raw_type is None:
        if not route:
            return "noop"
        if "latency_service_url" in route or "latency_url" in route:
            return "latency_service"
        if "strong" in route and "weak" in route:
            if any(key in route for key in ("threshold", "router_type", "classifier_model")):
                return "routellm"
            return "random_routing"
        if "target" in route or "model" in route:
            return "model"
        raise RouteBundleConfigError(
            f"route {model_id!r} requires an explicit type or a recognizable route shape"
        )
    if not isinstance(raw_type, str):
        raise RouteBundleConfigError("route type must be a string")

    normalized = raw_type.lower().replace("-", "_")
    aliases = {
        "direct": "model",
        "llm_target": "model",
        "model": "model",
        "target": "model",
        "random": "random_routing",
        "random_routing": "random_routing",
        "route_llm": "routellm",
        "routellm": "routellm",
        "latency": "latency_service",
        "latency_service": "latency_service",
        "noop": "noop",
        "no_op": "noop",
        "passthrough": "passthrough",
        "deterministic": "deterministic",
        "llm_classifier": "deterministic",
        "llm_classifier_routing": "deterministic",
        "stage_router": "stage_router",
        "stage_router_routing": "stage_router",
        "plan": "plan_execute",
        "plan_execute": "plan_execute",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise RouteBundleConfigError(f"unsupported route type {raw_type!r}") from exc


def _validate_route_keys(
    model_id: str,
    route: Mapping[str, object],
    route_type: str,
) -> None:
    where = f"route {model_id!r}"
    _validate_allowed_keys(route, _ROUTE_KEYS_BY_TYPE[route_type], where)
    if "defaults" in route:
        defaults = _require_mapping(route["defaults"], f"{where}.defaults")
        _validate_allowed_keys(
            defaults,
            _DEFAULT_KEYS_BY_TYPE[route_type],
            f"{where}.defaults",
        )


def _validate_allowed_keys(
    mapping: Mapping[str, object],
    allowed_keys: frozenset[str],
    where: str,
) -> None:
    unknown = sorted(set(mapping) - allowed_keys)
    if unknown:
        raise RouteBundleConfigError(
            f"unknown key(s) for {where}: {', '.join(unknown)}"
        )


def _route_metadata(
    model_id: str,
    route: Mapping[str, object],
    route_type: str,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "display_name": _optional_str(route.get("display_name")) or model_id,
        "switchyard": {"profile": route_type},
    }
    description = _optional_str(route.get("description"))
    if description is not None:
        metadata["description"] = description
    return metadata


def _expand_env(value: object) -> object:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RouteBundleConfigError("route bundle keys must be strings")
            result[key] = _expand_env(item)
        return result
    return value


def _expand_env_string(value: str) -> str:
    missing = [name for name in _ENV_REF_RE.findall(value) if name not in os.environ]
    if missing:
        raise RouteBundleConfigError(
            f"missing environment variable(s): {', '.join(sorted(set(missing)))}"
        )
    return os.path.expandvars(value)


def _require_mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RouteBundleConfigError(f"{where} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RouteBundleConfigError(f"{where} keys must be strings")
        result[key] = item
    return result


def _optional_mapping(value: object, where: str) -> dict[str, object]:
    if value is None:
        return {}
    return _require_mapping(value, where)


def _require_sequence(value: object, where: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise RouteBundleConfigError(f"{where} must be a list")
    return value


def _required_str(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RouteBundleConfigError(f"{where} must be a non-empty string")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RouteBundleConfigError(f"expected string, got {type(value).__name__}")
    return value


@overload
def _optional_float(value: object, default: float) -> float: ...


@overload
def _optional_float(value: object, default: None = None) -> float | None: ...


def _optional_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RouteBundleConfigError(f"expected number, got {type(value).__name__}")
    return float(value)


def _optional_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise RouteBundleConfigError(f"expected integer, got {type(value).__name__}")
    return value


def _optional_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RouteBundleConfigError(f"expected boolean, got {type(value).__name__}")
    return value


__all__ = [
    "RouteBundleConfigError",
    "build_route_bundle_table",
    "load_route_bundle_table",
    "parse_routing_profiles_file",
    "route_bundle_declares_token_capture",
    "routing_profile_model_ids",
]
