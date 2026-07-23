# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one v2 profile config into runnable Rust- and Python-defined profiles.

The YAML/JSON/TOML schema is shared. Rust remains the source of truth for file
format detection, environment interpolation, endpoint inheritance, target
resolution, and Rust-defined profile validation. Python only classifies profile
types registered through :func:`profile_config` and builds those runtimes from
Rust-parsed profile bodies.
"""

import importlib
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, cast

from switchyard.lib.profiles.protocols import ProfileInput, ProfileRunner
from switchyard.lib.profiles.subagent_override import SubagentOverrideProfile
from switchyard.lib.profiles.table import (
    ProfileConfigError,
    build_profile,
    lookup_profile_config,
    registered_profile_config_types,
)
from switchyard_rust.components import LlmTarget
from switchyard_rust.core import ChatResponse
from switchyard_rust.profiles import (
    ProfileConfigPlan,
    parse_profile_config_path,
)

__all__ = ["load_profiles", "load_profiles_and_targets"]

_PROFILE_TARGET = "profile_target"


def load_profiles(path: str | Path) -> dict[str, ProfileRunner]:
    """Build every profile in ``path`` into a runnable profile keyed by profile ID.

    Rust-defined profiles (``passthrough``, ``random-routing``, ...) and
    Python-defined profiles registered through :func:`profile_config` are built
    through the same call, sharing one resolved set of targets. Each returned
    value exposes ``async run(input)`` (see :class:`ProfileRunner`).

    Raises :class:`ProfileConfigError` or ``switchyard_rust`` config errors with
    profile-level context when the config is invalid.
    """
    profiles, _targets = load_profiles_and_targets(path)
    return profiles


def load_profiles_and_targets(
    path: str | Path,
) -> tuple[dict[str, ProfileRunner], dict[str, LlmTarget]]:
    """Build profiles and return the resolved target map used by those profiles."""
    _register_shipped_python_profiles()
    document: Any = parse_profile_config_path(path)
    python_profiles = _python_profiles(document)
    plan = document.without_profiles(list(python_profiles)).resolve()

    built: dict[str, ProfileRunner] = {}
    for profile_id in plan.profile_ids():
        built[profile_id] = _RustProfileRunner(plan.build_profile(profile_id))
    for profile_id, (profile_type, body) in python_profiles.items():
        built[profile_id] = _build_python_profile(profile_id, profile_type, body, plan)
    for profile_id in list(built):
        built[profile_id] = _wrap_subagent_override(profile_id, built[profile_id], document, plan)
    return built, _targets(plan)


def _wrap_subagent_override(
    profile_id: str,
    profile: ProfileRunner,
    document: Any,
    plan: ProfileConfigPlan,
) -> ProfileRunner:
    """Wrap a built profile when its envelope names a ``subagent_target``.

    Rust profiles already apply :class:`SubagentOverrideProfile` internally at
    build time, so Python-level wrapping is skipped for them. Python-defined
    profiles receive the Python wrapper here, which adds sub-agent routing
    without Rust-level stats (stats for Python profiles remain in the
    Python stats pipeline).

    An unknown target reference is a startup configuration error.
    """
    target_id = document.profile_subagent_target(profile_id)
    if target_id is None:
        return profile
    # Rust profiles handle subagent routing internally; skip double-wrapping.
    if isinstance(profile, _RustProfileRunner):
        return profile
    target = plan.target(target_id)
    if target is None:
        raise ProfileConfigError(
            f"profile {profile_id}: subagent_target references unknown target {target_id!r}"
        )
    from switchyard.lib.profiles.passthrough import PassthroughProfileConfig

    return SubagentOverrideProfile(profile, PassthroughProfileConfig(target=target).build())


class _RustProfileRunner:
    """Adapts an erased Rust ``Profile`` to the ``run(input)`` runner contract."""

    def __init__(self, profile: Any) -> None:
        self._profile = profile

    @property
    def profile_id(self) -> str:
        return cast(str, self._profile.profile_id)

    async def run(self, input: ProfileInput) -> ChatResponse:
        return cast(ChatResponse, await self._profile.run(input.request, input.metadata))


def _build_python_profile(
    profile_id: str,
    profile_type: str,
    body: dict[str, Any],
    plan: ProfileConfigPlan,
) -> ProfileRunner:
    """Construct one Python-defined profile, resolving its target references."""
    config_cls: Any = lookup_profile_config(profile_type)
    if not is_dataclass(config_cls):
        raise ProfileConfigError(
            f"profile {profile_id}: {config_cls.__qualname__} must be a dataclass"
        )

    field_names = {field.name for field in fields(config_cls)}
    unknown = sorted(set(body) - field_names - {"type"})
    if unknown:
        raise ProfileConfigError(
            f"profile {profile_id}: unknown field(s) {unknown} for profile type {profile_type!r}"
        )

    kwargs: dict[str, Any] = {}
    for field in fields(config_cls):
        if field.name not in body:
            continue  # let the dataclass default apply
        value = body[field.name]
        if field.metadata.get(_PROFILE_TARGET):
            value = _resolve_targets(profile_id, value, plan)
        kwargs[field.name] = value

    # config_cls is a registered dataclass; bind it through Any so mypy does not
    # treat it as the (non-instantiable) DataclassInstance protocol.
    ctor: Any = config_cls
    try:
        config = ctor(**kwargs)
    except TypeError as exc:  # missing required field, wrong arity, ...
        raise ProfileConfigError(f"profile {profile_id}: {exc}") from exc
    return cast(ProfileRunner, build_profile(config))


def _register_shipped_python_profiles() -> None:
    """Import shipped Python profile modules so their config types register."""
    importlib.import_module("switchyard.lib.profiles.header_routing")


def _python_profiles(
    document: Any,
) -> dict[str, tuple[str, dict[str, Any]]]:
    python_types = set(registered_profile_config_types())
    profiles: dict[str, tuple[str, dict[str, Any]]] = {}
    for profile_id in document.profile_ids():
        profile_type = document.profile_type(profile_id)
        if profile_type is None or profile_type not in python_types:
            continue
        body = document.profile_body(profile_id)
        if not isinstance(body, dict):
            raise ProfileConfigError(f"profile {profile_id}: profile body must be a mapping")
        profiles[profile_id] = (profile_type, body)
    return profiles


def _targets(plan: ProfileConfigPlan) -> dict[str, LlmTarget]:
    targets: dict[str, LlmTarget] = {}
    for target_id in plan.target_ids():
        target = plan.target(target_id)
        if target is not None:
            targets[target_id] = target
    return targets


def _resolve_targets(
    profile_id: str, value: Any, plan: ProfileConfigPlan
) -> LlmTarget | list[LlmTarget] | None:
    """Resolve a ``profile_target`` field's id reference(s) into ``LlmTarget``s.

    The value shape selects the arity, mirroring Rust ``#[profile_target]``
    support for single, optional, and list-valued target fields.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return _require_target(profile_id, value, plan)
    if isinstance(value, list):
        return [_require_target(profile_id, item, plan) for item in value]
    raise ProfileConfigError(
        f"profile {profile_id}: target reference must be a target id string or "
        f"list of ids, got {type(value).__name__}"
    )


def _require_target(profile_id: str, target_id: Any, plan: ProfileConfigPlan) -> LlmTarget:
    if not isinstance(target_id, str):
        raise ProfileConfigError(
            f"profile {profile_id}: target reference must be a string, got "
            f"{type(target_id).__name__}"
        )
    target = plan.target(target_id)
    if target is None:
        raise ProfileConfigError(f"profile {profile_id}: references unknown target {target_id!r}")
    return target
