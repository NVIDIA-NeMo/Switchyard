# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create candidate-bound LiteLLM plugins from Switchyard TOML configuration."""

from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Collection, Mapping
from pathlib import Path

from switchyard_litellm.plugins import RandomRoutingPlugin, StageRoutingPlugin

RoutingPlugin = RandomRoutingPlugin | StageRoutingPlugin

CONFIG_ENV = "SWITCHYARD_LITELLM_CONFIG"
_STAGE_KEYS = frozenset(
    {
        "algorithm",
        "picker",
        "confidence_threshold",
        "recent_window",
        "escalation_note",
        "deescalation_note",
        "only_on_wrong_signal_escalation",
        "capable_system_prompt",
        "efficient_system_prompt",
    }
)
_RANDOM_KEYS = frozenset({"algorithm", "seed", "weights"})
_MAX_RANDOM_SEED = (1 << 64) - 1


def load_routing_plugin(path: str | Path) -> RoutingPlugin:
    """Load a Stage or Random routing plugin from TOML."""
    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            values: Mapping[str, object] = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(
            f"Could not load Switchyard routing config {config_path}: {error}"
        ) from error

    algorithm = _required_string(values, "algorithm", config_path)
    if algorithm == "stage":
        return _stage_plugin(values, config_path)
    if algorithm == "random":
        return _random_plugin(values, config_path)
    raise ValueError(
        f"Switchyard routing config {config_path} has unsupported algorithm {algorithm!r}"
    )


def load_routing_plugin_from_environment(
    environment: Mapping[str, str] = os.environ,
) -> RoutingPlugin:
    """Load the routing plugin named by ``SWITCHYARD_LITELLM_CONFIG``."""
    path = environment.get(CONFIG_ENV)
    if not path:
        raise ValueError(f"{CONFIG_ENV} must name a Switchyard routing TOML file")
    return load_routing_plugin(path)


def _stage_plugin(values: Mapping[str, object], config_path: Path) -> StageRoutingPlugin:
    _reject_unknown_keys(values, _STAGE_KEYS, config_path)
    picker = _required_string(values, "picker", config_path)
    if picker not in {"capable_first", "efficient_first"}:
        raise ValueError(
            f"Switchyard routing config {config_path} picker must be "
            "'capable_first' or 'efficient_first'"
        )
    escalation_note = _optional_string(values, "escalation_note", config_path)
    deescalation_note = _optional_string(values, "deescalation_note", config_path)
    if deescalation_note is not None and escalation_note is None:
        raise ValueError(
            f"Switchyard routing config {config_path} deescalation_note requires escalation_note"
        )
    return StageRoutingPlugin(
        picker=picker,
        confidence_threshold=_required_confidence_threshold(values, config_path),
        recent_window=_optional_integer(
            values,
            "recent_window",
            config_path,
        ),
        escalation_note=escalation_note,
        deescalation_note=deescalation_note,
        only_on_wrong_signal_escalation=_optional_boolean(
            values,
            "only_on_wrong_signal_escalation",
            config_path,
            default=True,
        ),
        capable_system_prompt=_optional_string(
            values,
            "capable_system_prompt",
            config_path,
        ),
        efficient_system_prompt=_optional_string(
            values,
            "efficient_system_prompt",
            config_path,
        ),
    )


def _random_plugin(values: Mapping[str, object], config_path: Path) -> RandomRoutingPlugin:
    _reject_unknown_keys(values, _RANDOM_KEYS, config_path)
    return RandomRoutingPlugin(
        seed=_optional_integer(
            values,
            "seed",
            config_path,
            maximum=_MAX_RANDOM_SEED,
        ),
        weights=_optional_weights(values, config_path),
    )


def _reject_unknown_keys(
    values: Mapping[str, object],
    allowed: Collection[str],
    config_path: Path,
) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(
            f"Switchyard routing config {config_path} has unknown keys: {', '.join(unknown)}"
        )


def _required_string(
    values: Mapping[str, object],
    key: str,
    config_path: Path,
) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Switchyard routing config {config_path} {key} must be a nonempty string")
    return value


def _optional_string(
    values: Mapping[str, object],
    key: str,
    config_path: Path,
) -> str | None:
    if key not in values:
        return None
    value = values[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Switchyard routing config {config_path} {key} must be a nonempty string")
    return value


def _required_confidence_threshold(
    values: Mapping[str, object],
    config_path: Path,
) -> float:
    value = values.get("confidence_threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Switchyard routing config {config_path} confidence_threshold must be a finite number"
        )
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError(
            f"Switchyard routing config {config_path} confidence_threshold must be a finite number"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Switchyard routing config {config_path} confidence_threshold "
            "must be between 0 and 1 inclusive"
        )
    return threshold


def _optional_integer(
    values: Mapping[str, object],
    key: str,
    config_path: Path,
    *,
    maximum: int | None = None,
) -> int | None:
    if key not in values:
        return None
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Switchyard routing config {config_path} {key} must be a nonnegative integer"
        )
    if value < 0 or maximum is not None and value > maximum:
        upper_bound = f" no greater than {maximum}" if maximum is not None else ""
        raise ValueError(
            f"Switchyard routing config {config_path} {key} must be a nonnegative integer"
            f"{upper_bound}"
        )
    return value


def _optional_boolean(
    values: Mapping[str, object],
    key: str,
    config_path: Path,
    *,
    default: bool,
) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Switchyard routing config {config_path} {key} must be a boolean")
    return value


def _optional_weights(
    values: Mapping[str, object],
    config_path: Path,
) -> list[float] | None:
    if "weights" not in values:
        return None
    value = values["weights"]
    if not isinstance(value, list):
        raise ValueError(
            f"Switchyard routing config {config_path} weights must be a nonempty array"
        )
    if not value:
        raise ValueError(
            f"Switchyard routing config {config_path} weights must be a nonempty array"
        )

    weights: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"Switchyard routing config {config_path} weights must contain only numbers"
            )
        weight = float(item)
        if not math.isfinite(weight):
            raise ValueError(f"Switchyard routing config {config_path} weights must be finite")
        if weight < 0.0:
            raise ValueError(f"Switchyard routing config {config_path} weights must be nonnegative")
        weights.append(weight)

    if not any(weight > 0.0 for weight in weights):
        raise ValueError(
            f"Switchyard routing config {config_path} weights must contain a positive value"
        )
    return weights


__all__ = [
    "CONFIG_ENV",
    "RoutingPlugin",
    "load_routing_plugin",
    "load_routing_plugin_from_environment",
]
