# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build :class:`RouteTable` instances for client-selected model routing.

Two layers compose model-name dispatch:

* :func:`build_passthrough_table` — the **discovery layer**. Given a set of
  tiers, registers each tier's configured model as a direct passthrough and
  optionally hydrates additional entries from each tier's ``GET /v1/models``
  catalog via a caller-supplied ``discovery_fn``.
* :func:`register_random_routing_policy` — the **routing-policy layer**. Adds a
  virtual model id to an existing table whose chain dispatches via the
  random-routing weighted coin.

:func:`build_random_routing_table` composes the two for the launcher and
YAML route-bundle paths. Both callers share this assembly to keep behavior
consistent.

Discovery is injected as a callable so this module stays free of CLI-only
dependencies. Pass ``discovery_fn=None`` (the default) to skip discovery
entirely.
"""

import hashlib
import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.model_listing import (
    combined_model_capabilities,
    model_capabilities,
)
from switchyard.lib.processors.llm_classifier.presets import (
    classifier_prompt_sha256,
    resolve_classifier_prompt,
)
from switchyard.lib.profiles.deterministic_routing_config import (
    DeterministicRoutingConfig,
)
from switchyard.lib.profiles.plan_execute_config import PlanExecuteConfig
from switchyard.lib.profiles.random_routing import RandomRoutingConfig
from switchyard.lib.route_table import ChainRuntime, RouteTable

logger = logging.getLogger(__name__)

#: Callable that returns the model ids advertised by an OpenAI-compatible
#: ``GET /models`` endpoint. Builders record a non-fatal warning if it raises.
DiscoveryFn: TypeAlias = Callable[[str, str], list[str]]


def random_routing_virtual_model_id(config: RandomRoutingConfig) -> str:
    """Stable model id representing this random-routing pair.

    Launcher clients route by the request ``model`` field. Using the strong
    model name for the random chain makes the client picker ambiguous because
    "strong" actually means "random". This virtual id gives the routed pair its
    own model identity while leaving the real strong/weak ids available for
    direct model overrides.
    """
    fingerprint = hashlib.sha1(
        "\0".join((
            config.strong.model,
            config.weak.model,
            f"{config.strong_probability:.12g}",
        )).encode("utf-8"),
    ).hexdigest()[:8]
    return f"switchyard-default-random-{fingerprint}"


def build_random_routing_switchyard(
    config: RandomRoutingConfig,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    """Build the primary random-routing chain used by launchers and bundles."""
    from switchyard.lib.profiles import ProfileSwitchyard, RandomRoutingProfileConfig

    return ProfileSwitchyard(
        RandomRoutingProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            pre_request_processors=pre_routing_request_processors,
            response_processors=extra_response_processors,
        )
    )


def build_tier_passthrough_switchyard(
    tier: LlmTarget,
    extra_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    """Build a single-tier chain for an explicitly selected model."""
    from switchyard.lib.profiles import PassthroughProfileConfig, ProfileSwitchyard

    return ProfileSwitchyard(
        PassthroughProfileConfig(
            target=tier,
        )
        .build()
        .with_runtime_components(
            pre_request_processors=extra_request_processors,
            response_processors=extra_response_processors,
        )
    )


def build_single_model_table(
    model: str,
    switchyard: ChainRuntime,
    metadata_source: str = "configured",
) -> RouteTable:
    """Wrap one pre-built chain in a table so ``/v1/models`` can list it."""
    table = RouteTable()
    table.register(
        model,
        switchyard,
        metadata={
            "description": f"Direct Switchyard route to {model}.",
            "capabilities": model_capabilities(model),
            "switchyard": {"profile": "passthrough", "source": metadata_source},
        },
        default=True,
    )
    return table


def build_passthrough_table(
    tiers: Sequence[LlmTarget],
    discovery_fn: DiscoveryFn | None = None,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> RouteTable:
    """Build a table of passthrough chains for *tiers* and their catalogs.

    For each tier this:

    1. Registers the tier's configured model as a direct passthrough.
    2. If ``discovery_fn`` is supplied and the tier has both a ``base_url`` and
       ``api_key``, calls ``discovery_fn(base_url, api_key)`` and registers each
       remaining model as a passthrough that reuses the tier's connectivity.
       Discovered models use :class:`BackendFormat.AUTO` so each direct route
       resolves its backend format independently.

    Catalog fetches are deduped by ``(base_url, api_key)`` so two tiers that
    share an endpoint hit it once. Entries already in the table (including
    a tier's configured model) are never overwritten by a later discovered
    duplicate.
    """
    table = RouteTable()
    catalog_cache: dict[tuple[str, str], list[str]] = {}

    def _register_passthrough(
        tier: LlmTarget,
        source: str,
        description: str,
    ) -> None:
        table.register(
            tier.model,
            build_tier_passthrough_switchyard(
                tier,
                extra_request_processors=pre_routing_request_processors,
                extra_response_processors=extra_response_processors,
            ),
            metadata={
                "description": description,
                "capabilities": model_capabilities(tier.model),
                "switchyard": {"profile": "passthrough", "source": source},
            },
        )

    # Pass 1: register every tier's configured model so the picker always
    # surfaces them first, regardless of where they fall in catalog order.
    for tier in tiers:
        if tier.model and tier.model not in table.registered_models():
            _register_passthrough(
                tier,
                source="configured",
                description=f"Direct Switchyard route to {tier.model}.",
            )

    # Pass 2: hydrate from each tier's GET /v1/models catalog when a
    # discovery_fn is supplied.
    if discovery_fn is None:
        return table

    for tier in tiers:
        if not tier.endpoint.base_url or not tier.endpoint.api_key:
            continue

        cache_key = (tier.endpoint.base_url, tier.endpoint.api_key)
        if cache_key not in catalog_cache:
            try:
                catalog_cache[cache_key] = discovery_fn(
                    tier.endpoint.base_url, tier.endpoint.api_key,
                )
            except Exception as exc:
                logger.warning(
                    "Could not hydrate model catalog from %s: %s",
                    tier.endpoint.base_url,
                    exc,
                )
                table.add_model_listing_warning(
                    "Model discovery failed for "
                    f"{tier.endpoint.base_url}: {exc}"
                )
                catalog_cache[cache_key] = []

        for model_id in catalog_cache[cache_key]:
            if not model_id or model_id in table.registered_models():
                continue
            _register_passthrough(
                LlmTarget(
                    id=f"discovered-{model_id}",
                    model=model_id,
                    # Discovered via GET /v1/models on an OpenAI-compatible endpoint —
                    # Chat Completions is the safe default. Users who need a different
                    # format for a discovered model should declare it explicitly.
                    format=BackendFormat.OPENAI,
                    api_key=tier.endpoint.api_key,
                    base_url=tier.endpoint.base_url,
                    timeout_secs=tier.endpoint.timeout_secs,
                ),
                source="discovered",
                description=(
                    f"Direct Switchyard passthrough to discovered model "
                    f"{model_id}."
                ),
            )

    return table


def register_random_routing_policy(
    table: RouteTable,
    config: RandomRoutingConfig,
    random_routing_switchyard: ChainRuntime,
    routing_model: str | None = None,
) -> str:
    """Layer a random-routing policy on top of a passthrough table.

    Returns the virtual model id under which the policy was registered. Strong
    and weak models registered separately by :func:`build_passthrough_table`
    remain available as direct overrides via the client's model picker.
    """
    virtual_model = routing_model or random_routing_virtual_model_id(config)
    table.register(
        virtual_model,
        random_routing_switchyard,
        metadata={
            "display_name": "Switchyard random routing",
            "description": (
                "Random routes "
                f"{config.strong.model} (strong) and {config.weak.model} (weak), "
                f"p_strong={config.strong_probability:.2f}."
            ),
            "capabilities": combined_model_capabilities([
                config.strong.model,
                config.weak.model,
            ]),
            "switchyard": {
                "profile": "random_routing",
                "strong_model": config.strong.model,
                "weak_model": config.weak.model,
                "strong_probability": config.strong_probability,
            },
        },
        default=True,
    )
    return virtual_model


def build_random_routing_table(
    config: RandomRoutingConfig,
    random_routing_switchyard: ChainRuntime,
    routing_model: str | None = None,
    discovery_fn: DiscoveryFn | None = None,
    extra_response_processors: Sequence[Any] = (),
    pre_routing_request_processors: Sequence[Any] = (),
) -> RouteTable:
    """Compose the discovery and random-routing-policy layers.

    First builds a passthrough table for the strong and weak tiers (plus
    everything advertised by their ``GET /v1/models`` catalogs when
    ``discovery_fn`` is supplied), then layers the random-routing virtual model
    on top.
    """
    table = build_passthrough_table(
        (config.strong, config.weak),
        discovery_fn=discovery_fn,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
    )
    register_random_routing_policy(
        table,
        config,
        random_routing_switchyard=random_routing_switchyard,
        routing_model=routing_model,
    )
    return table


def deterministic_routing_virtual_model_id(
    config: DeterministicRoutingConfig,
) -> str:
    """Stable model id representing this deterministic-routing bundle."""
    prompt = resolve_classifier_prompt(
        config.profile_name,
        config.classifier_system_prompt,
    )
    fingerprint = hashlib.sha1(
        "\0".join((
            config.strong.model,
            config.weak.model,
            config.classifier.model,
            config.profile_name,
            classifier_prompt_sha256(prompt),
            str(config.classifier_max_request_chars),
            str(config.classifier_recent_turn_window),
        )).encode("utf-8"),
    ).hexdigest()[:8]
    return f"switchyard-deterministic-{fingerprint}"


def build_deterministic_routing_switchyard(
    config: DeterministicRoutingConfig,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    """Build the deterministic-routing chain used by the launcher."""
    from switchyard.lib.profiles import (
        DeterministicRoutingProfileConfig,
        ProfileSwitchyard,
    )

    return ProfileSwitchyard(
        DeterministicRoutingProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            pre_request_processors=pre_routing_request_processors,
            response_processors=extra_response_processors,
        )
    )


def register_deterministic_routing_policy(
    table: RouteTable,
    config: DeterministicRoutingConfig,
    deterministic_routing_switchyard: ChainRuntime,
    routing_model: str | None = None,
) -> str:
    """Layer a deterministic-routing policy on top of a passthrough table.

    Strong + weak passthrough entries are registered separately by
    :func:`build_passthrough_table`; the classifier is not user-selectable.
    """
    prompt = resolve_classifier_prompt(
        config.profile_name,
        config.classifier_system_prompt,
    )
    virtual_model = routing_model or deterministic_routing_virtual_model_id(config)
    table.register(
        virtual_model,
        deterministic_routing_switchyard,
        metadata={
            "display_name": "Switchyard deterministic routing",
            "description": (
                "LLM-classifier routes between "
                f"{config.strong.model} (strong) and {config.weak.model} (weak) "
                f"using {config.classifier.model} (classifier, "
                f"profile={config.profile_name})."
            ),
            "capabilities": combined_model_capabilities([
                config.strong.model,
                config.weak.model,
            ]),
            "switchyard": {
                "profile": "deterministic_routing",
                "strong_model": config.strong.model,
                "weak_model": config.weak.model,
                "classifier_model": config.classifier.model,
                "classifier_profile": config.profile_name,
                "classifier_prompt_sha256": classifier_prompt_sha256(prompt),
                "classifier_max_request_chars": config.classifier_max_request_chars,
                "classifier_recent_turn_window": config.classifier_recent_turn_window,
                "classifier_min_confidence": config.classifier_min_confidence,
            },
        },
        default=True,
    )
    return virtual_model


def build_deterministic_routing_table(
    config: DeterministicRoutingConfig,
    deterministic_routing_switchyard: ChainRuntime,
    routing_model: str | None = None,
    discovery_fn: DiscoveryFn | None = None,
    extra_response_processors: Sequence[Any] = (),
    pre_routing_request_processors: Sequence[Any] = (),
) -> RouteTable:
    """Compose discovery + deterministic-routing-policy layers."""
    table = build_passthrough_table(
        (config.strong, config.weak),
        discovery_fn=discovery_fn,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
    )
    register_deterministic_routing_policy(
        table,
        config,
        deterministic_routing_switchyard=deterministic_routing_switchyard,
        routing_model=routing_model,
    )
    return table


def plan_execute_virtual_model_id(config: PlanExecuteConfig) -> str:
    """Stable model id representing this plan-execute bundle.

    Mirrors :func:`deterministic_routing_virtual_model_id` — fingerprints
    the planner + executor models so a launcher with multiple plan-execute
    bundles per process still produces a unique virtual id per bundle.
    """
    fingerprint = hashlib.sha1(
        "\0".join((
            config.planner.model,
            config.executor.model,
            str(config.cadence_n),
        )).encode("utf-8"),
    ).hexdigest()[:8]
    return f"switchyard-plan-execute-{fingerprint}"


def build_plan_execute_switchyard(
    config: PlanExecuteConfig,
    pre_routing_request_processors: Sequence[Any] = (),
    extra_response_processors: Sequence[Any] = (),
) -> ChainRuntime:
    """Build the plan-execute chain used by the launcher."""
    from switchyard.lib.profiles import PlanExecuteProfileConfig, ProfileSwitchyard

    return ProfileSwitchyard(
        PlanExecuteProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            pre_request_processors=pre_routing_request_processors,
            response_processors=extra_response_processors,
        )
    )


def register_plan_execute_policy(
    table: RouteTable,
    config: PlanExecuteConfig,
    plan_execute_switchyard: ChainRuntime,
    routing_model: str | None = None,
) -> str:
    """Layer a plan-execute policy on top of a passthrough table.

    The executor model is registered separately by
    :func:`build_passthrough_table`; the planner is not user-selectable
    (it's the routing logic, not a destination).
    """
    virtual_model = routing_model or plan_execute_virtual_model_id(config)
    table.register(
        virtual_model,
        plan_execute_switchyard,
        metadata={
            "display_name": "Switchyard plan-execute",
            "description": (
                f"Strong planner ({config.planner.model}) drafts a plan "
                f"every {config.cadence_n} turn(s); cheap executor "
                f"({config.executor.model}) continues from the plan."
            ),
            "capabilities": model_capabilities(config.executor.model),
            "switchyard": {
                "profile": "plan_execute",
                "planner_model": config.planner.model,
                "executor_model": config.executor.model,
                "cadence_n": config.cadence_n,
            },
        },
        default=True,
    )
    return virtual_model


def build_plan_execute_table(
    config: PlanExecuteConfig,
    plan_execute_switchyard: ChainRuntime,
    routing_model: str | None = None,
    discovery_fn: DiscoveryFn | None = None,
    extra_response_processors: Sequence[Any] = (),
    pre_routing_request_processors: Sequence[Any] = (),
) -> RouteTable:
    """Compose discovery + plan-execute-policy layers.

    Only the executor tier is registered as a direct passthrough — the
    planner is internal to the chain and never user-selectable from the
    client's ``/model`` picker.
    """
    table = build_passthrough_table(
        (config.executor,),
        discovery_fn=discovery_fn,
        pre_routing_request_processors=pre_routing_request_processors,
        extra_response_processors=extra_response_processors,
    )
    register_plan_execute_policy(
        table,
        config,
        plan_execute_switchyard=plan_execute_switchyard,
        routing_model=routing_model,
    )
    return table


__all__ = [
    "DiscoveryFn",
    "build_deterministic_routing_table",
    "build_deterministic_routing_switchyard",
    "build_passthrough_table",
    "build_plan_execute_table",
    "build_plan_execute_switchyard",
    "build_random_routing_table",
    "build_random_routing_switchyard",
    "build_single_model_table",
    "build_tier_passthrough_switchyard",
    "deterministic_routing_virtual_model_id",
    "plan_execute_virtual_model_id",
    "random_routing_virtual_model_id",
    "register_deterministic_routing_policy",
    "register_plan_execute_policy",
    "register_random_routing_policy",
]
