# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write explicit policy circuits without fitting or an optimizer."""

import json
from pathlib import Path

FEATURES = [
    "new_user_turn", "task_bytes", "change_task", "analysis_task",
    "history_depth", "recent_turns", "severity", "repeated_failure",
    "recovery", "mutation_share", "observation_share", "planning_share",
    "tests_passed", "unknown_streak", "has_mutated", "truncated",
    "previous_efficient", "previous_capable",
]
CIRCUITS = [
    ("initial_planning", {"new_user_turn": 1, "task_bytes": 1, "change_task": .5}, -1.5, 0, 4),
    ("continue_planning", {"previous_capable": 1, "has_mutated": -1}, 0, 0, 1.5),
    ("unresolved_error", {"severity": 1, "recovery": -1}, -.3, 0, 2.5),
    ("repeated_failure", {"repeated_failure": 1, "recovery": -1}, 0, 0, 1.8),
    ("failing_edits", {"severity": 1, "mutation_share": 1, "recovery": -1}, -1.3, 0, 2),
    ("clean_execution", {"mutation_share": 1, "recovery": 1, "severity": -1}, -.5, 1.1, 0),
    ("recovered_execution", {"recovery": 1, "has_mutated": 1}, -1.25, 1, 0),
    ("verified_execution", {"tests_passed": 1, "has_mutated": 1}, -1, 1.3, 0),
    ("unproductive_commands", {"history_depth": 1, "unknown_streak": 1, "recovery": -1}, -.5, 0, 1),
    ("replanning", {"planning_share": 1, "has_mutated": 1, "severity": 1}, -1.5, 0, .7),
    ("simple_start", {"new_user_turn": 1, "task_bytes": -1, "change_task": -.5}, 0, .2, 0),
    ("efficient_continuity", {"previous_efficient": 1}, 0, .15, 0),
    ("recover_before_handback", {"previous_capable": 1, "severity": 1, "recovery": -1}, -1.3, 0, 1.5),
]


def main() -> None:
    """Save the literal matrices and their neuron names for review."""
    root = Path(__file__).resolve().parent
    weights = {
        "schema_version": 1,
        "input_weights": [[terms.get(f, 0) for f in FEATURES] for _, terms, _, _, _ in CIRCUITS],
        "hidden_bias": [bias for _, _, bias, _, _ in CIRCUITS],
        "output_weights": [[c[3] for c in CIRCUITS], [c[4] for c in CIRCUITS]],
        "output_bias": [.05, .15],
        "cost_bias": [.01, 1],
    }
    (root / "reference_weights.json").write_text(json.dumps(weights, indent=2) + "\n")
    (root / "circuits.json").write_text(json.dumps({
        "features": FEATURES,
        "neurons": [name for name, *_ in CIRCUITS],
        "targets": ["efficient", "capable"],
        "status": "Development revision 2, live task performance unvalidated",
        "revision": "Ignore isolated soft errors in three escalation circuits and recalibrate initial planning after task excerpt scope correction",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
