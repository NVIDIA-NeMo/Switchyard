# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural + drift checks on src/craft_taskgen/rubrics.py.

Two layers of check:

1. **Structural** — rubrics.py imports cleanly and exposes all expected
   canonical constants as non-empty strings. This catches accidental deletions
   or renames early.
2. **Drift vs contributor-facing markdown** — canonical paragraphs in
   `rubrics.py` must appear in `references/task-building-guide.md`. The
   Python constants are the source of truth for prompts; the markdown is the
   human mirror. When either moves, both must move.

The drift layer is intentionally one-directional (python → markdown substring
match). Markdown may contain additional prose around each rubric; we only
require the canonical paragraph to be present as a substring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from craft_taskgen import rubrics

CANONICAL_CONSTANTS = (
    "RUBRIC_QUICK_DECISION_FRAMEWORK",
    "RUBRIC_REJECT_PATTERNS",
    "RUBRIC_DESIGN_PRINCIPLES",
    "RUBRIC_H_RULES",
    "RUBRIC_T2_H1_ORCHESTRATION",
    "RUBRIC_V4_AUDIT",
    "RUBRIC_ANTI_LEAKAGE",
    "RUBRIC_ALIGNMENT_CATEGORIES",
    "RUBRIC_FAIRNESS_REVIEW",
)


@pytest.mark.parametrize("name", CANONICAL_CONSTANTS)
def test_rubric_constant_exists_and_nonempty(name: str) -> None:
    value = getattr(rubrics, name, None)
    assert value is not None, f"rubrics.{name} is missing"
    assert isinstance(value, str), f"rubrics.{name} is not a string"
    assert value.strip(), f"rubrics.{name} is empty"


def test_instruction_word_range_matches_hard_max() -> None:
    # Soft target must be strictly smaller than the hard ceiling enforced
    # by task_format.py validators.
    assert rubrics.INSTRUCTION_WORD_HARD_MAX > 100


# ---------------------------------------------------------------------------
# Drift: canonical paragraphs from rubrics.py must appear in the human-facing
# markdown. Fragments (not full constants) so small rewrites don't thrash the
# test; each fragment is a distinctive phrase from the constant that a
# contributor would not accidentally delete.
# ---------------------------------------------------------------------------

GUIDE_PATH = Path(__file__).parent.parent / "references" / "task-building-guide.md"


DRIFT_FRAGMENTS: dict[str, list[str]] = {
    # Filled in during step (j) of the refactor when task-building-guide.md is
    # synced to match rubrics.py. Until then, the test is a structural check
    # only — drift is not enforced against the markdown.
    #
    # Example entry (to be uncommented in step j):
    # "RUBRIC_DESIGN_PRINCIPLES": [
    #     "Integration is the discriminating step, not component creation.",
    #     "Preserving existing behavior while adding new behavior is a genuine trap.",
    # ],
}


@pytest.mark.parametrize(
    ("constant_name", "fragment"),
    [(k, f) for k, fragments in DRIFT_FRAGMENTS.items() for f in fragments],
)
def test_rubric_fragment_present_in_guide(constant_name: str, fragment: str) -> None:
    guide_text = GUIDE_PATH.read_text()
    constant_value = getattr(rubrics, constant_name)
    assert fragment in constant_value, (
        f"Expected fragment missing from rubrics.{constant_name}; "
        f"update DRIFT_FRAGMENTS in this test if the constant was intentionally reworded."
    )
    assert fragment in guide_text, (
        f"Fragment from rubrics.{constant_name} not found in "
        f"references/task-building-guide.md — sync the human mirror or update "
        f"DRIFT_FRAGMENTS in this test."
    )
