"""Layer 1: executable checks derived from the task statement.

The check script is derived ONCE, from the instruction only, BEFORE any
attempt runs. That ordering is load-bearing: a checker written after seeing
a solution inherits the solution's misreading of the task (the observed
failure mode on ambiguous tasks), while one written blind encodes only the
stated requirements.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .spec import CheckResult, LlmFn

DERIVE_PROMPT = """You are writing an automated acceptance check for a task, \
BEFORE any attempt at the task exists. You see only the task statement.

Write a single bash script that exits 0 if and only if a workspace satisfies \
every requirement the task statement makes machine-checkable: required files \
existing at their exact stated paths, outputs matching stated values or \
tolerances, commands the statement says must succeed, services it says must \
respond. Execute real commands against the workspace (the script runs with \
the workspace as its working directory); print a line per check so failures \
are diagnosable. Do NOT attempt to solve the task inside the script, and do \
NOT invent requirements the statement does not make.

If the statement contains NOTHING machine-checkable, output exactly the \
single line NO_CHECKS instead of a script.

Task statement:
---
{instruction}
---

Reply with only the bash script (or NO_CHECKS), no commentary."""


def derive_check_script(llm: LlmFn, instruction: str) -> str | None:
    """Ask the LLM for a check script; None when nothing is checkable."""
    reply = llm(DERIVE_PROMPT.format(instruction=instruction)).strip()
    if not reply or "NO_CHECKS" in reply.splitlines()[0]:
        return None
    # Tolerate a fenced code block.
    m = re.search(r"```(?:bash|sh)?\n(.*?)```", reply, re.DOTALL)
    script = m.group(1) if m else reply
    return script.strip() or None


def run_check_script(
    script: str | None, workspace: Path, timeout_seconds: float = 300.0
) -> CheckResult:
    """Execute the derived checks inside the workspace."""
    if script is None:
        return CheckResult(available=False, passed=False, output="no executable checks")
    try:
        proc = subprocess.run(
            ["bash", "-c", script],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(available=True, passed=False, output="check script timed out")
    output = (proc.stdout + proc.stderr)[-8000:]
    return CheckResult(available=True, passed=proc.returncode == 0, output=output)
