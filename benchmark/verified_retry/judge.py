"""Layers 2-3: confidence-gated LLM judge and comparative candidate pick.

Measured behaviour this module is designed around (TB-2.1 study, 2026-08):
judges are weak at absolute verdicts (35.6% false-pass from narrated
evidence) but strong comparatively (87% correct picks), and their stated
confidence separates true from false passes (0.83 vs 0.69 mean) - so the
absolute verdict is gated on confidence and used as a veto layer on top of
executed checks, never as the sole verifier.
"""
from __future__ import annotations

import json
import re

from .spec import LlmFn, Verdict

JUDGE_PROMPT = """You are verifying whether ONE attempt at a task actually \
satisfied the task's stated requirements. Judge only from demonstrated \
evidence: commands the attempt ran and their observed output, executed check \
results, produced artifacts. Confident prose is not evidence. Re-read the \
exact stated deliverables (paths, formats, tolerances) and confirm they were \
literally met.

Task statement:
---
{instruction}
---
Executed acceptance checks (ground evidence, weigh heavily):
---
{check_output}
---
Attempt evidence (final portion of the executor transcript):
---
{evidence}
---

Reply with a single JSON object: {{"pass": true|false, "confidence": 0.0-1.0, \
"reason": "<one sentence>"}}. confidence 0.5 means coin-flip, 1.0 certain. \
If the evidence does not DEMONSTRATE a stated requirement, do not assume it."""

COMPARE_PROMPT = """Multiple independent attempts at the same task are below. \
None passed automated verification, so pick the attempt MOST LIKELY to \
satisfy the task's stated requirements, comparing demonstrated evidence only.

Task statement:
---
{instruction}
---
{candidates}

Reply with a single JSON object: {{"pick": "<attempt id>", "reason": "<one \
sentence>"}}."""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"judge reply had no JSON object: {text[:200]!r}")
    return json.loads(m.group(0))


def judge_attempt(
    llm: LlmFn, instruction: str, evidence: str, check_output: str
) -> Verdict:
    reply = llm(
        JUDGE_PROMPT.format(
            instruction=instruction,
            check_output=check_output or "(no executable checks were available)",
            evidence=evidence,
        )
    )
    try:
        obj = _extract_json(reply)
        return Verdict(
            passed=bool(obj["pass"]),
            confidence=float(obj.get("confidence", 0.0)),
            reason=str(obj.get("reason", "")),
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        # Unparseable verdict fails CLOSED: an unverified attempt is not
        # accepted (retries are cheap; shipping unverified work is not).
        return Verdict(passed=False, confidence=0.0, reason="unparseable judge reply")


def judge_compare(
    llm: LlmFn, instruction: str, candidates: dict[str, str]
) -> tuple[str, str]:
    """Pick the best of several failed-verification candidates.

    candidates maps attempt-id -> evidence text. Returns (pick, reason);
    falls back to the last candidate if the reply is unusable.
    """
    blocks = "\n".join(
        f"Attempt {cid}:\n---\n{ev}\n---" for cid, ev in candidates.items()
    )
    reply = llm(COMPARE_PROMPT.format(instruction=instruction, candidates=blocks))
    fallback = list(candidates)[-1]
    try:
        obj = _extract_json(reply)
        pick = str(obj.get("pick", ""))
        if pick not in candidates:
            return fallback, "judge picked unknown id; returned last attempt"
        return pick, str(obj.get("reason", ""))
    except (ValueError, json.JSONDecodeError):
        return fallback, "unparseable compare reply; returned last attempt"
