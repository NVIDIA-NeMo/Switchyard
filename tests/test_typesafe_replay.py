# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark/typesafe_replay.py"
FIXTURE = REPO / "benchmark/typesafe/fixtures/synthetic-routing.json"


@pytest.fixture
def replay_module() -> ModuleType:
    """Load the replay tool without installing Switchyard or provider dependencies."""
    spec = importlib.util.spec_from_file_location("switchyard_typesafe_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture_document() -> dict[str, object]:
    """Return an isolated copy of the checked-in synthetic fixture."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_synthetic_fixture_reports_routing_quality_cost_and_order_sensitivity(
    replay_module: ModuleType, fixture_document: dict[str, object]
) -> None:
    """Replay aggregation, fallback, and fixed-baseline comparisons together."""
    report = replay_module.replay(fixture_document)

    assert report["case_count"] == 3
    assert report["selected_counts"] == {"capable": 1, "efficient": 2}
    assert report["fallback_count"] == 1
    assert report["order_sensitive_case_count"] == 1
    assert report["maximum_probability_movement"] == 0.4
    assert report["routed"] == {"quality": 2.0, "cost": 14.0}
    assert report["fixed_targets"] == {
        "capable": {"quality": 3.0, "cost": 30.0},
        "efficient": {"quality": 1.0, "cost": 6.0},
    }
    assert report["best_fixed_target"] == "capable"
    assert report["quality_regret_vs_best_fixed"] == 1.0
    assert report["cost_delta_vs_best_fixed"] == -16.0
    assert report["cases"][2] == {
        "id": "order-sensitive-request",
        "average_probabilities": {"capable": 0.5, "efficient": 0.5},
        "classifier_target": "capable",
        "confidence": 0.0,
        "fallback": True,
        "selected_target": "efficient",
        "order_sensitive": True,
        "maximum_probability_movement": 0.4,
    }


def test_cli_output_is_byte_stable_and_offline() -> None:
    """Run twice with an empty environment and require byte-identical JSON."""
    command = [sys.executable, str(SCRIPT), str(FIXTURE)]
    first = subprocess.run(command, check=False, capture_output=True, env={})
    second = subprocess.run(command, check=False, capture_output=True, env={})

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["suite"] == "synthetic-two-target-routing"


def test_unsupported_schema_version_is_rejected(
    replay_module: ModuleType, fixture_document: dict[str, object]
) -> None:
    """Reject fixture semantics the evaluator does not understand."""
    fixture_document["schema_version"] = 2

    with pytest.raises(replay_module.FixtureError, match="schema_version must be 1"):
        replay_module.replay(fixture_document)


def test_invalid_probability_distribution_is_rejected(
    replay_module: ModuleType, fixture_document: dict[str, object]
) -> None:
    """Do not calculate metrics from incomplete provider evidence."""
    document = deepcopy(fixture_document)
    document["cases"][0]["orders"][0]["probabilities"] = {
        "capable": 0.8,
        "efficient": 0.8,
    }

    with pytest.raises(replay_module.FixtureError, match="must sum to one"):
        replay_module.replay(document)


def test_duplicate_candidate_order_is_rejected(
    replay_module: ModuleType, fixture_document: dict[str, object]
) -> None:
    """Prevent one repeated ordering from receiving accidental extra weight."""
    document = deepcopy(fixture_document)
    first_order = document["cases"][0]["orders"][0]
    document["cases"][0]["orders"][1] = deepcopy(first_order)

    with pytest.raises(replay_module.FixtureError, match="repeats a candidate order"):
        replay_module.replay(document)
