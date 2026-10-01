# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded TypeSafe/Jev routing evidence without provider calls."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = 1
PROBABILITY_SUM_TOLERANCE = 0.02


class FixtureError(ValueError):
    """Report invalid or unsupported replay evidence."""


def require(condition: bool, message: str) -> None:
    """Reject incomplete evidence before calculating metrics."""
    if not condition:
        raise FixtureError(message)


def object_value(value: Any, name: str) -> dict[str, Any]:
    """Read one JSON object with a useful field diagnostic."""
    require(isinstance(value, dict), f"{name} must be an object")
    return cast(dict[str, Any], value)


def string_value(value: Any, name: str) -> str:
    """Read one non-empty string field."""
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be a non-empty string")
    return value


def number_value(value: Any, name: str, *, maximum: float | None = None) -> float:
    """Read one finite, nonnegative numeric field."""
    require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        f"{name} must be a finite, nonnegative number",
    )
    number = float(value)
    if maximum is not None:
        require(number <= maximum, f"{name} must be at most {maximum}")
    return number


def string_list(value: Any, name: str) -> list[str]:
    """Read one non-empty list of unique strings."""
    require(isinstance(value, list) and bool(value), f"{name} must be a non-empty list")
    values = [string_value(item, f"{name}[]") for item in value]
    require(len(values) == len(set(values)), f"{name} must not contain duplicates")
    return values


def probabilities(value: Any, labels: list[str], name: str) -> dict[str, float]:
    """Validate one complete probability distribution."""
    document = object_value(value, name)
    require(set(document) == set(labels), f"{name} must contain every candidate exactly once")
    parsed = {
        label: number_value(document[label], f"{name}.{label}", maximum=1.0) for label in labels
    }
    total = sum(parsed.values())
    require(
        abs(total - 1.0) <= PROBABILITY_SUM_TOLERANCE,
        f"{name} must sum to one within {PROBABILITY_SUM_TOLERANCE}",
    )
    return parsed


def selected_label(labels: list[str], values: dict[str, float]) -> str:
    """Select the highest probability, breaking ties by configured candidate order."""
    return max(labels, key=values.__getitem__)


def rounded(value: float) -> float:
    """Keep reports readable and stable across harmless floating-point noise."""
    return round(value, 12)


def replay(document: dict[str, Any]) -> dict[str, Any]:
    """Validate one fixture document and return its deterministic replay report."""
    schema_version = document.get("schema_version")
    require(
        type(schema_version) in (int, float) and schema_version == SCHEMA_VERSION,
        f"schema_version must be {SCHEMA_VERSION}",
    )
    suite = string_value(document.get("suite"), "suite")
    model = string_value(document.get("model"), "model")
    candidate_rows = document.get("candidates")
    require(
        isinstance(candidate_rows, list) and len(candidate_rows) >= 2,
        "candidates needs at least two entries",
    )

    candidates = [object_value(row, "candidates[]") for row in candidate_rows]
    labels = [string_value(row.get("label"), "candidates[].label") for row in candidates]
    require(len(labels) == len(set(labels)), "candidate labels must be unique")
    for row in candidates:
        string_value(row.get("description"), "candidates[].description")

    default_target = string_value(document.get("default_target"), "default_target")
    require(default_target in labels, "default_target must name a candidate")
    threshold = number_value(document.get("base_threshold"), "base_threshold", maximum=1.0)
    case_rows = document.get("cases")
    require(isinstance(case_rows, list) and bool(case_rows), "cases must be a non-empty list")

    fixed_quality = dict.fromkeys(labels, 0.0)
    fixed_cost: dict[str, float | None] = dict.fromkeys(labels, 0.0)
    selected_counts = dict.fromkeys(labels, 0)
    routed_quality = 0.0
    routed_cost: float | None = 0.0
    fallback_count = 0
    order_sensitive_count = 0
    maximum_movement = 0.0
    case_ids: set[str] = set()
    case_reports: list[dict[str, Any]] = []

    for case_index, case_value in enumerate(case_rows):
        case = object_value(case_value, f"cases[{case_index}]")
        case_id = string_value(case.get("id"), f"cases[{case_index}].id")
        require(case_id not in case_ids, f"duplicate case id {case_id!r}")
        case_ids.add(case_id)

        order_rows = case.get("orders")
        require(
            isinstance(order_rows, list) and bool(order_rows),
            f"case {case_id!r} needs at least one candidate order",
        )
        seen_orders: set[tuple[str, ...]] = set()
        distributions: list[dict[str, float]] = []
        order_winners: list[str] = []
        resolved_model: str | None = None
        for order_index, order_value in enumerate(order_rows):
            order = object_value(order_value, f"case {case_id!r} order {order_index}")
            order_model = string_value(
                order.get("resolved_model"), f"case {case_id!r} order {order_index}.resolved_model"
            )
            if resolved_model is None:
                resolved_model = order_model
            else:
                require(
                    order_model == resolved_model, f"case {case_id!r} has mixed resolved models"
                )
            order_labels = string_list(
                order.get("candidate_order"),
                f"case {case_id!r} order {order_index}.candidate_order",
            )
            require(
                set(order_labels) == set(labels) and len(order_labels) == len(labels),
                f"case {case_id!r} order {order_index} must contain every candidate",
            )
            order_key = tuple(order_labels)
            require(order_key not in seen_orders, f"case {case_id!r} repeats a candidate order")
            seen_orders.add(order_key)
            distribution = probabilities(
                order.get("probabilities"),
                labels,
                f"case {case_id!r} order {order_index}.probabilities",
            )
            distributions.append(distribution)
            order_winners.append(selected_label(order_labels, distribution))

        averaged = {
            label: sum(distribution[label] for distribution in distributions) / len(distributions)
            for label in labels
        }
        average_total = sum(averaged.values())
        averaged = {label: value / average_total for label, value in averaged.items()}
        classifier_target = selected_label(labels, averaged)
        uniform = 1.0 / len(labels)
        unbounded_confidence = (averaged[classifier_target] - uniform) / (1.0 - uniform)
        confidence = max(0.0, min(1.0, unbounded_confidence))

        fallback = confidence < threshold
        final_target = default_target if fallback else classifier_target
        fallback_count += int(fallback)
        selected_counts[final_target] += 1

        order_sensitive = len(set(order_winners)) > 1
        order_sensitive_count += int(order_sensitive)
        movement = max(
            max(distribution[label] for distribution in distributions)
            - min(distribution[label] for distribution in distributions)
            for label in labels
        )
        maximum_movement = max(maximum_movement, movement)

        outcomes = object_value(case.get("outcomes"), f"case {case_id!r}.outcomes")
        require(
            set(outcomes) == set(labels),
            f"case {case_id!r}.outcomes must contain every candidate exactly once",
        )
        parsed_quality: dict[str, float] = {}
        parsed_cost: dict[str, float | None] = {}
        for label in labels:
            outcome = object_value(outcomes[label], f"case {case_id!r}.outcomes.{label}")
            quality = number_value(
                outcome.get("quality"), f"case {case_id!r}.outcomes.{label}.quality", maximum=1.0
            )
            cost_value = outcome.get("cost")
            cost = (
                None
                if cost_value is None
                else number_value(cost_value, f"case {case_id!r}.outcomes.{label}.cost")
            )
            parsed_quality[label] = quality
            parsed_cost[label] = cost
            fixed_quality[label] += quality
            if fixed_cost[label] is not None:
                fixed_cost[label] = None if cost is None else fixed_cost[label] + cost

        routed_quality += parsed_quality[final_target]
        selected_cost = parsed_cost[final_target]
        if routed_cost is not None:
            routed_cost = None if selected_cost is None else routed_cost + selected_cost
        case_reports.append(
            {
                "id": case_id,
                "resolved_model": resolved_model,
                "average_probabilities": {label: rounded(averaged[label]) for label in labels},
                "classifier_target": classifier_target,
                "confidence": rounded(confidence),
                "fallback": fallback,
                "selected_target": final_target,
                "order_sensitive": order_sensitive,
                "maximum_probability_movement": rounded(movement),
            }
        )

    best_quality = max(fixed_quality.values())
    quality_ties = [label for label in labels if fixed_quality[label] == best_quality]
    if all(fixed_cost[label] is not None for label in quality_ties):
        best_fixed = min(quality_ties, key=lambda label: fixed_cost[label])
    else:
        best_fixed = quality_ties[0]
    fixed_report = {
        label: {
            "quality": rounded(fixed_quality[label]),
            "cost": None if fixed_cost[label] is None else rounded(fixed_cost[label]),
        }
        for label in labels
    }
    best_fixed_cost = fixed_cost[best_fixed]
    cost_delta = (
        None
        if routed_cost is None or best_fixed_cost is None
        else rounded(routed_cost - best_fixed_cost)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "model": model,
        "base_threshold": threshold,
        "default_target": default_target,
        "case_count": len(case_reports),
        "selected_counts": selected_counts,
        "fallback_count": fallback_count,
        "order_sensitive_case_count": order_sensitive_count,
        "maximum_probability_movement": rounded(maximum_movement),
        "routed": {
            "quality": rounded(routed_quality),
            "cost": None if routed_cost is None else rounded(routed_cost),
        },
        "fixed_targets": fixed_report,
        "best_fixed_target": best_fixed,
        "quality_regret_vs_best_fixed": rounded(fixed_quality[best_fixed] - routed_quality),
        "cost_delta_vs_best_fixed": cost_delta,
        "cases": case_reports,
    }


def read_fixture(path: Path) -> dict[str, Any]:
    """Read one replay fixture from disk."""
    return object_value(json.loads(path.read_text(encoding="utf-8")), str(path))


def main(argv: list[str] | None = None) -> int:
    """Replay one fixture and write stable JSON to stdout or a file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="Versioned TypeSafe replay fixture")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report = replay(read_fixture(args.fixture))
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(encoded)
        else:
            args.output.write_text(encoded, encoding="utf-8")
    except (FixtureError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"Cannot replay fixture: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
