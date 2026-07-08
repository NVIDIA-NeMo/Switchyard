# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill-distillation session capture helpers for launchers."""

import json
import logging
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from switchyard.cli.config.user_config import load_user_config
from switchyard.lib.skill_distillation_store import SkillDistillationSessionCapture
from switchyard.lib.stats_accumulator import StatsAccumulator

logger = logging.getLogger(__name__)

RUN_CONTEXT_PATH_ENV = "SWITCHYARD_SKILL_DISTILLATION_RUN_CONTEXT_PATH"
ACTIVE_SKILL_EVIDENCE_PATH_ENV = "SWITCHYARD_SKILL_DISTILLATION_ACTIVE_EVIDENCE_PATH"
_MAX_LAUNCH_METADATA_BYTES = 1_000_000


def _load_launch_metadata(name: str, project_dir: Path) -> dict[str, Any] | None:
    raw_path = os.environ.get(name)
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_dir / path
    try:
        metadata_path = path.resolve(strict=True)
        metadata_path.relative_to(project_dir)
        metadata_stat = path.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{name} must name a real file inside the launch project") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata_stat.st_mode)
        or metadata_stat.st_nlink != 1
        or metadata_stat.st_size > _MAX_LAUNCH_METADATA_BYTES
    ):
        raise ValueError(f"{name} must name a single-link regular JSON file under 1 MB")
    try:
        payload: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must contain a UTF-8 JSON object") from exc
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must contain a JSON object with string keys")
    return payload


@dataclass
class LaunchSkillDistillationSession:
    """Session capture state owned by one launcher run."""

    capture: SkillDistillationSessionCapture | None
    exit_code: int | None = None


def build_launch_skill_distillation_session(
    *,
    target: str,
    display_model: str,
    strategy_summary: str | None = None,
) -> SkillDistillationSessionCapture | None:
    """Return a session capture when the saved namespace is configured."""

    config = load_user_config().skill_distillation
    if not config.configured or config.namespace is None:
        return None
    project_dir = Path.cwd().resolve()
    run_context = _load_launch_metadata(RUN_CONTEXT_PATH_ENV, project_dir)
    active_skill_evidence = _load_launch_metadata(
        ACTIVE_SKILL_EVIDENCE_PATH_ENV,
        project_dir,
    )
    try:
        return SkillDistillationSessionCapture(
            namespace=config.namespace,
            launch_target=target,
            display_model=display_model,
            strategy_summary=strategy_summary,
            project_dir=project_dir,
            run_context=run_context,
            active_skill_evidence=active_skill_evidence,
        )
    except OSError as exc:
        logger.warning(
            "Skill distillation: failed to initialize local session store: %s",
            exc,
        )
        return None


@contextmanager
def launch_skill_distillation_session(
    *,
    target: str,
    display_model: str,
    strategy_summary: str | None = None,
    stats: StatsAccumulator,
) -> Iterator[LaunchSkillDistillationSession]:
    """Create and finalize skill-distillation capture for one launcher run."""

    state = LaunchSkillDistillationSession(
        capture=build_launch_skill_distillation_session(
            target=target,
            display_model=display_model,
            strategy_summary=strategy_summary,
        ),
    )
    error: BaseException | None = None
    try:
        yield state
    except BaseException as exc:
        error = exc
        raise
    finally:
        if state.capture is not None:
            state.capture.finish(
                exit_code=state.exit_code,
                stats=stats,
                error=error,
            )


__all__ = [
    "ACTIVE_SKILL_EVIDENCE_PATH_ENV",
    "LaunchSkillDistillationSession",
    "RUN_CONTEXT_PATH_ENV",
    "build_launch_skill_distillation_session",
    "launch_skill_distillation_session",
]
