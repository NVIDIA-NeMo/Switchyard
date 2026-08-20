"""The verified-retry loop: fresh attempt -> execute checks -> gated judge ->
accept or retry; comparative pick if nothing verifies.

Acceptance policy (in order):
  1. The executor must not have crashed.
  2. If executable checks exist, they MUST pass (execution outranks opinion).
  3. The judge must say pass with confidence >= spec.accept_confidence.
     When checks passed, the judge acts as a veto layer; when no checks
     could be derived, the judge is the only verifier and the gate matters
     most.
Every attempt starts from a pristine copy of the task workspace - attempt
independence is what makes retries worth anything.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from .checks import derive_check_script, run_check_script
from .judge import judge_attempt, judge_compare
from .spec import Attempt, ExecutorFn, FinalResult, LlmFn, TaskSpec

EVIDENCE_TAIL_CHARS = 40_000


def _evidence(transcript: str, check_output: str) -> str:
    tail = transcript[-EVIDENCE_TAIL_CHARS:]
    return f"{tail}\n\n[executed acceptance checks]\n{check_output}"


def run_verified_retry(
    spec: TaskSpec,
    executor: ExecutorFn,
    llm: LlmFn,
    archive_dir: Path | None = None,
) -> FinalResult:
    archive_root = Path(archive_dir or tempfile.mkdtemp(prefix="vr-")) / spec.stamp()
    archive_root.mkdir(parents=True, exist_ok=True)

    # Derive checks from the instruction ONLY, before any attempt exists.
    check_script = derive_check_script(llm, spec.instruction)

    attempts: list[Attempt] = []
    for i in range(1, spec.max_attempts + 1):
        workspace = archive_root / f"attempt-{i}"
        shutil.copytree(spec.workspace_src, workspace)

        started = time.monotonic()
        result = executor(workspace, spec.instruction)
        result.wall_seconds = time.monotonic() - started

        checks = run_check_script(check_script, workspace, spec.check_timeout_seconds)

        verdict = None
        accepted = False
        if not result.crashed and (not checks.available or checks.passed):
            verdict = judge_attempt(
                llm,
                spec.instruction,
                _evidence(result.transcript, checks.output),
                checks.output if checks.available else "",
            )
            accepted = verdict.passed and verdict.confidence >= spec.accept_confidence

        attempts.append(
            Attempt(
                index=i,
                workspace=workspace,
                executor=result,
                checks=checks,
                verdict=verdict,
                accepted=accepted,
            )
        )
        if accepted:
            return FinalResult(
                accepted=True, attempt_index=i, workspace=workspace, attempts=attempts
            )

    # Nothing verified: comparative judge picks the least-bad candidate.
    # The result is explicitly flagged unverified (accepted=False).
    non_crashed = [a for a in attempts if not a.executor.crashed] or attempts
    candidates = {
        str(a.index): _evidence(a.executor.transcript, a.checks.output)
        for a in non_crashed
    }
    if len(candidates) == 1:
        pick, reason = next(iter(candidates)), "single candidate; comparison skipped"
    else:
        pick, reason = judge_compare(llm, spec.instruction, candidates)
    picked = next(a for a in attempts if str(a.index) == pick)
    return FinalResult(
        accepted=False,
        attempt_index=picked.index,
        workspace=picked.workspace,
        attempts=attempts,
        pick_reason=reason,
    )
