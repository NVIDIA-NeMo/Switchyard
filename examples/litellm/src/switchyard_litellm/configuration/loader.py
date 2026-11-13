# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create candidate-bound LiteLLM plugins from Switchyard TOML configuration.

``tomllib`` decodes TOML values into Python types. This module's helpers enforce
the plugin schema and value constraints after that decoding step.
"""

from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Collection, Mapping
from pathlib import Path

from switchyard_litellm.plugins import (
    ClassifierRoutingPlugin,
    LiteLLMJudge,
    RandomRoutingPlugin,
    StageRoutingPlugin,
)

RoutingPlugin = ClassifierRoutingPlugin | RandomRoutingPlugin | StageRoutingPlugin

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
_CLASSIFIER_KEYS = frozenset(
    {
        "algorithm",
        "judge_model",
        "base_threshold",
        "threshold_step",
        "prompt",
        "max_output_tokens",
        "response_format_type",
        "judge_max_tokens_field",
        "judge_timeout_seconds",
    }
)
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
    if algorithm == "llm_classifier":
        return _classifier_plugin(values, config_path)
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
    """Validate parsed Stage fields and construct its candidate-bound plugin."""
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


def _classifier_plugin(values: Mapping[str, object], config_path: Path) -> ClassifierRoutingPlugin:
    """Validate parsed capability-classifier fields and construct its candidate-bound plugin.

    The judge is served through LiteLLM by :class:`LiteLLMJudge` against the
    deployment named by ``judge_model``; in the proxy that is the proxy's own
    router, so the judge call is spend-tracked like any other.
    """
    _reject_unknown_keys(values, _CLASSIFIER_KEYS, config_path)
    judge_model = _required_string(values, "judge_model", config_path)
    response_format_type = _optional_string(values, "response_format_type", config_path)
    if response_format_type is not None and response_format_type not in {
        "json_schema",
        "json_object",
    }:
        raise ValueError(
            f"Switchyard routing config {config_path} response_format_type must be "
            "'json_schema' or 'json_object'"
        )
    max_tokens_field = _optional_string(values, "judge_max_tokens_field", config_path)
    if max_tokens_field is not None and max_tokens_field not in {
        "max_completion_tokens",
        "max_tokens",
    }:
        raise ValueError(
            f"Switchyard routing config {config_path} judge_max_tokens_field must be "
            "'max_completion_tokens' or 'max_tokens'"
        )
    max_output_tokens = _optional_integer(values, "max_output_tokens", config_path)
    if max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError(
            f"Switchyard routing config {config_path} max_output_tokens must be at least 1"
        )
    base_threshold = _required_unit_number(values, "base_threshold", config_path)
    threshold_step = _optional_unit_number(values, "threshold_step", config_path)
    if threshold_step is not None and base_threshold + 2 * threshold_step > 1.0:
        raise ValueError(
            f"Switchyard routing config {config_path} base_threshold + 2 * threshold_step "
            "must not exceed 1"
        )
    timeout = _optional_positive_number(values, "judge_timeout_seconds", config_path)
    judge_params: dict[str, object] = {}
    if timeout is not None:
        judge_params["timeout"] = timeout
    judge = LiteLLMJudge(
        judge_model,
        max_tokens_field=max_tokens_field or "max_completion_tokens",
        **judge_params,
    )
    return ClassifierRoutingPlugin(
        judge=judge,
        judge_model=judge_model,
        base_threshold=base_threshold,
        threshold_step=threshold_step if threshold_step is not None else 0.0,
        prompt=_optional_string(values, "prompt", config_path),
        max_output_tokens=max_output_tokens if max_output_tokens is not None else 4096,
        response_format_type=response_format_type or "json_schema",
    )


def _required_unit_number(values: Mapping[str, object], key: str, config_path: Path) -> float:
    """Return a required finite number from 0 through 1."""
    if key not in values:
        raise ValueError(f"Switchyard routing config {config_path} {key} is required")
    number = _optional_unit_number(values, key, config_path)
    assert number is not None
    return number


def _optional_unit_number(
    values: Mapping[str, object], key: str, config_path: Path
) -> float | None:
    """Return an optional finite number from 0 through 1."""
    if key not in values:
        return None
    value = values[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Switchyard routing config {config_path} {key} must be a finite number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"Switchyard routing config {config_path} {key} must be between 0 and 1 inclusive"
        )
    return number


def _optional_positive_number(
    values: Mapping[str, object], key: str, config_path: Path
) -> float | None:
    """Return an optional finite number greater than 0."""
    if key not in values:
        return None
    value = values[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Switchyard routing config {config_path} {key} must be a finite number")
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"Switchyard routing config {config_path} {key} must be greater than 0")
    return number


def _random_plugin(values: Mapping[str, object], config_path: Path) -> RandomRoutingPlugin:
    """Validate parsed Random fields and construct its candidate-bound plugin."""
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
    """Reject parsed TOML keys outside an algorithm's supported schema."""
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
    """Return a required nonempty string already decoded by ``tomllib``."""
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Switchyard routing config {config_path} {key} must be a nonempty string")
    return value


def _optional_string(
    values: Mapping[str, object],
    key: str,
    config_path: Path,
) -> str | None:
    """Return an optional nonempty string already decoded by ``tomllib``."""
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
    """Validate and return the parsed Stage confidence threshold."""
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
    """Validate and return an optional nonnegative TOML integer."""
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
    """Validate and return an optional TOML Boolean with its default."""
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Switchyard routing config {config_path} {key} must be a boolean")
    return value


def _optional_weights(
    values: Mapping[str, object],
    config_path: Path,
) -> list[float] | None:
    """Validate and normalize optional Random weights decoded from TOML."""
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
