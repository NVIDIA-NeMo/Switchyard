# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search pipeline orchestrator: runs search steps sequentially with state management.

Called by the main pipeline.py when --dimension search is used.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from craft_taskgen.config import _load_env
from craft_taskgen.search.config import SEARCH_STEPS, SearchPipelineState
from craft_taskgen.search.steps import SEARCH_STEP_FUNCS


def run_search_pipeline(
    *,
    tasks_dir: str,
    repos_dir: str,
    output_dir: str,
    concurrency: int = 4,
    limit: int = 0,
    from_step: str = "extract",
    resume_path: str | None = None,
    profile_data: dict | None = None,
) -> None:
    """Run the search-from-T2 pipeline."""
    _load_env()

    # Load or create state
    if resume_path and os.path.exists(resume_path):
        state = SearchPipelineState.load(resume_path)
        state_file = resume_path
        print(f"Resumed from {resume_path}")
        print(f"  Completed stages: {state.stages_completed}")
    else:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        run_dir = os.path.join(output_dir, "runs", ts)
        os.makedirs(run_dir, exist_ok=True)
        state = SearchPipelineState(
            created=datetime.now().isoformat(),
            run_dir=run_dir,
            tasks_dir=tasks_dir,
            repos_dir=repos_dir,
            output_dir=output_dir,
            concurrency=concurrency,
            limit=limit,
            profile_data=profile_data or {},
        )
        state_file = os.path.join(run_dir, "state.json")

    # Determine which steps to run
    start_idx = SEARCH_STEPS.index(from_step)
    steps_to_run = SEARCH_STEPS[start_idx:]
    print(f"Pipeline stages: {' -> '.join(steps_to_run)}")
    print(f"Output: {state.output_dir}")
    print()

    for step_name in steps_to_run:
        if step_name in state.stages_completed and step_name != from_step:
            continue

        state.current_stage = step_name
        state.save(state_file)

        try:
            func = SEARCH_STEP_FUNCS[step_name]
            func(state)
        except NotImplementedError as e:
            print(f"\n  SKIP: {e}", file=sys.stderr)
            print(f"  Resume: --resume {state_file} --from-step {step_name}")
            break
        except Exception as e:
            print(f"\n  ERROR in step '{step_name}': {e}", file=sys.stderr)
            print(f"  Resume: --resume {state_file} --from-step {step_name}")
            state.save(state_file)
            break

        if step_name not in state.stages_completed:
            state.stages_completed.append(step_name)
        state.current_stage = ""
        state.save(state_file)

    print(f"\nPipeline complete. State: {state_file}")
