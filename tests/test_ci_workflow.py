# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the primary CI workflow."""

from pathlib import Path
from typing import Any

import yaml


def test_full_ci_depends_on_changed_paths_not_pr_title() -> None:
    """Ensure a docs-prefixed title cannot bypass substantive CI jobs."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    workflow: dict[str, Any] = yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)
    changes = workflow["jobs"]["changes"]

    full_ci = changes["outputs"]["full_ci"]
    assert "pull_request.title" not in full_ci

    filter_step = next(step for step in changes["steps"] if step.get("id") == "filter")
    assert "pull_request.title" not in filter_step["if"]
    assert filter_step["if"] == "github.event_name == 'pull_request'"
