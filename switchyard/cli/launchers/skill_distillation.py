# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill-distillation session capture helpers for launchers."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from switchyard.cli.config.user_config import load_user_config
from switchyard.lib.skill_distillation_store import SkillDistillationSessionCapture
from switchyard.lib.stats_accumulator import StatsAccumulator

logger = logging.getLogger(__name__)


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
    try:
        return SkillDistillationSessionCapture(
            namespace=config.namespace,
            launch_target=target,
            display_model=display_model,
            strategy_summary=strategy_summary,
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
    "LaunchSkillDistillationSession",
    "build_launch_skill_distillation_session",
    "launch_skill_distillation_session",
]
