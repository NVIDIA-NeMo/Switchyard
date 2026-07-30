# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the planning-task scorer.

Usage::

    craft-taskgen-planning-score \\
        --candidates-dir path/to/candidates \\
        --dataset-dir path/to/harbor-planning-dataset \\
        --work-dir path/to/scratch \\
        [--planner-a-model opus] [--planner-b-model haiku] \\
        [--implementer-model sonnet] \\
        [--filter task_name] [--skip-harbor]

Records a ``planning_scores`` block on each candidate JSON with per-planner
F2P/P2P and the planner-A-minus-B delta. Does not tag tasks; tagging is a
separate downstream step that inspects the empirical distribution first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidates-dir", required=True)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Harbor planning dataset (output of `craft-taskgen-convert --adapter planning`)",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Scratch dir for planner/implementer datasets and harbor trials",
    )
    parser.add_argument("--filter", default=None)
    parser.add_argument("--planner-a-model", default=None, help="Strong planner (default: Opus 4.6)")
    parser.add_argument("--planner-b-model", default=None, help="Baseline planner (default: Haiku 4.5)")
    parser.add_argument(
        "--implementer-model",
        default=None,
        help="Fixed implementer model (default: Sonnet 4.6)",
    )
    parser.add_argument("--agent", default=None, help="Harbor agent (default: claude-code)")
    parser.add_argument("--api-base", default=None, help="Override inference API base URL")
    parser.add_argument(
        "--skip-harbor",
        action="store_true",
        help="Skip harbor runs and only score existing trials (debug)",
    )
    parser.add_argument(
        "--skip-synth",
        action="store_true",
        help="Skip gold-plan synth (useful when candidates already carry gold_plan)",
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from craft_taskgen.planning import scorer

    kwargs: dict[str, Any] = {
        "candidates_dir": args.candidates_dir,
        "dataset_dir": args.dataset_dir,
        "work_dir": args.work_dir,
        "task_filter": args.filter,
        "skip_harbor": args.skip_harbor,
        "skip_synth": args.skip_synth,
    }
    for name in (
        "planner_a_model",
        "planner_b_model",
        "implementer_model",
        "agent",
        "api_base",
    ):
        val = getattr(args, name, None)
        if val:
            kwargs[name] = val
    return scorer.run_score(**kwargs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="craft-taskgen-planning-score",
        description=__doc__,
    )
    _add_args(parser)
    args = parser.parse_args()
    result = _run(args)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
