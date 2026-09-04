"""Core datatypes for the verified-retry orchestrator.

Design provenance: parameters and layer choices come from the TB-2.1
Kimi-K3-Max study (2026-08). Measured there:
  - LLM judge from evidence alone: 82.5% verdict accuracy, 35.6% false-pass.
  - Confidence gate at >=0.85 removes 14/16 false-passes (keeps 43/70 true).
  - Comparative pick over candidates: 87% accuracy vs 62.5% random.
  - Executable checks are the only layer that matches a benchmark verifier;
    every observed false-pass came from trusting narrated (not executed)
    evidence.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# An LLM is any prompt -> completion-text callable; tests inject fakes and
# production wraps an API client. Keeping this a bare callable keeps the
# orchestrator import-clean of any provider SDK.
LlmFn = Callable[[str], str]


@dataclass
class ExecutorResult:
    """Outcome of one executor (agent) run inside a workspace."""

    exit_code: int
    transcript: str
    wall_seconds: float = 0.0

    @property
    def crashed(self) -> bool:
        return self.exit_code != 0


# An executor runs the task in the given workspace and returns its result.
ExecutorFn = Callable[[Path, str], ExecutorResult]


@dataclass
class CheckResult:
    """Result of executing the derived check script in a workspace."""

    available: bool  # False when no executable checks could be derived
    passed: bool
    output: str = ""


@dataclass
class Verdict:
    """LLM judge verdict on a single attempt."""

    passed: bool
    confidence: float
    reason: str = ""


@dataclass
class Attempt:
    index: int
    workspace: Path
    executor: ExecutorResult
    checks: CheckResult
    verdict: Verdict | None
    accepted: bool


@dataclass
class FinalResult:
    """What the orchestrator hands back to the caller."""

    accepted: bool  # True iff an attempt passed verification
    attempt_index: int  # 1-based index of the returned attempt
    workspace: Path  # the returned attempt's (archived) workspace
    attempts: list[Attempt] = field(default_factory=list)
    pick_reason: str = ""  # set when the comparative judge chose (unverified)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)


@dataclass
class TaskSpec:
    task_id: str
    instruction: str
    workspace_src: Path  # pristine task workspace; never mutated
    max_attempts: int = 4
    # Acceptance requires the judge to say pass with at least this
    # confidence (0.85 per the study's threshold sweep). Retries are cheap;
    # false-passes are not.
    accept_confidence: float = 0.85
    check_timeout_seconds: float = 300.0

    def stamp(self) -> str:
        return f"{self.task_id}-{int(time.time())}"
