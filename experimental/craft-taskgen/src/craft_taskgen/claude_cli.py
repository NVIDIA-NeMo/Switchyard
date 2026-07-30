# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude CLI helper and auto-fix logic for the task generation pipeline.

Gateway-only policy: every `claude -p` invocation must route through the
NVIDIA LiteLLM gateway (`ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` from .env).
OAuth fallback is forbidden because this machine shares a personal Claude
account that has a budget.

The gateway-env builder lives in `gateway.py` now (shared with `llm_judge.py`
for direct-API calls). `_build_gateway_env` is preserved as a re-export so
existing callers (preflight, etc.) don't need to change.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import subprocess

import craft_taskgen.config as _cfg
from craft_taskgen.config import (
    Stage,
    TaskState,
)
from craft_taskgen.diagnostics import _next_diagnostic_path, _write_diagnostic
from craft_taskgen.gateway import build_gateway_env, get_openai_gateway_creds
from craft_taskgen.prompts import (
    fix_docker_prompt,
    fix_f2p_p2p_classify_prompt,
)

_SUMMARY_MODEL = "aws/anthropic/bedrock-claude-sonnet-4-6"

# Preserved re-export for callers that still reach for the underscore name
# (e.g. preflight.py before the step-j docs sweep). Prefer `build_gateway_env`
# from craft_taskgen.gateway in new code.
_build_gateway_env = build_gateway_env


def _format_subprocess_hint(stderr: str, stdout: str, *, max_chars: int = 200) -> str:
    """Return a concise retry hint from subprocess output.

    Prefer stderr, but fall back to stdout because `claude -p --output-format json`
    sometimes emits useful structured errors there while keeping stderr empty.
    """
    for label, text in (("stderr", stderr), ("stdout", stdout)):
        hint = text[:max_chars].strip().replace("\n", " ")
        if hint:
            return f"{label}: {hint}"
    return "no stdout/stderr"


def _extract_result_error_message(result_text: str) -> str:
    """Extract a concise API error message from Claude's result field."""
    if not result_text:
        return ""

    # Common shape:
    # API Error: 400 {"error":{"message":"{\"message\":\"...\"}"}}
    m = re.match(r"API Error:\s*\d+\s+(.*)", result_text, re.DOTALL)
    payload = m.group(1) if m else result_text
    try:
        outer = json.loads(payload)
        message = outer.get("error", {}).get("message", "")
        if not message:
            return result_text[:500]
        try:
            inner = json.loads(message)
            if isinstance(inner, dict) and inner.get("message"):
                return str(inner["message"])[:500]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return str(message)[:500]
    except (json.JSONDecodeError, TypeError, ValueError):
        return result_text[:500]


def _extract_cli_error(result: dict) -> dict | None:
    """Parse Claude CLI JSON error payloads from stdout."""
    stdout = result.get("stdout", "")
    if not stdout:
        return None
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if not payload.get("is_error"):
        return None

    parsed = {
        "subtype": payload.get("subtype", ""),
        "api_error_status": payload.get("api_error_status"),
        "num_turns": payload.get("num_turns"),
        "error_detail": _extract_result_error_message(payload.get("result", "")),
    }
    return parsed


def run_claude(
    prompt: str,
    *,
    max_turns: int | None = None,
    timeout: int | None = None,
    cwd: str | None = None,
    json_schema: dict | None = None,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
) -> dict:
    """Shell out to `claude -p` and return parsed JSON output.

    Returns dict with keys: result, session_id, structured_output (if schema),
    or {error: ...} on failure.

    Gateway-only: raises RuntimeError if ANTHROPIC_API_KEY/BASE_URL aren't
    set or `model` is missing.
    """
    if max_turns is None:
        max_turns = _cfg.DEFAULT_MAX_TURNS
    if timeout is None:
        timeout = _cfg.DEFAULT_TIMEOUT
    model = model or _cfg.LLM_STEP_MODEL
    env = build_gateway_env(model)  # raises if gateway env/model missing

    cmd = [
        _cfg.CLAUDE_CMD,
        "-p",
        prompt,
        "--permission-mode",
        _cfg.DEFAULT_PERMISSION_MODE,
        "--output-format",
        "json",
        "--max-turns",
        str(max_turns),
    ]
    if json_schema:
        cmd.extend(["--json-schema", json.dumps(json_schema)])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if allowed_tools:
        for tool in allowed_tools:
            cmd.extend(["--allowedTools", tool])
    # Model is set via ANTHROPIC_MODEL env (Harbor pattern) — no --model flag.
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or os.getcwd(),
            env=env,
        )
        if result.returncode != 0:
            return {
                "error": "nonzero_exit",
                "returncode": result.returncode,
                "stderr": result.stderr[:2000],
                "stdout": result.stdout[:2000],
            }
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except json.JSONDecodeError:
        return {"error": "json_parse", "raw": result.stdout[:2000] if result else ""}


_TRANSIENT_ERRORS = {"nonzero_exit", "timeout"}
_MAX_RETRIES = 5
_BACKOFF_BASE = 5  # seconds — CLI errors resolve fast, no need for long waits
_BACKOFF_CAP = 30  # max backoff per retry


async def run_claude_async(
    prompt: str,
    *,
    max_turns: int | None = None,
    timeout: int | None = None,
    cwd: str | None = None,
    json_schema: dict | None = None,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
) -> dict:
    """Async version of run_claude. Retries transient errors with exponential backoff.

    Gateway-only: raises RuntimeError if ANTHROPIC_API_KEY/BASE_URL aren't
    set or `model` is missing.
    """
    if max_turns is None:
        max_turns = _cfg.DEFAULT_MAX_TURNS
    if timeout is None:
        timeout = _cfg.DEFAULT_TIMEOUT
    model = model or _cfg.LLM_STEP_MODEL
    env = build_gateway_env(model)  # raises if gateway env/model missing

    cmd = [
        _cfg.CLAUDE_CMD,
        "-p",
        prompt,
        "--permission-mode",
        _cfg.DEFAULT_PERMISSION_MODE,
        "--output-format",
        "json",
        "--max-turns",
        str(max_turns),
    ]
    if json_schema:
        cmd.extend(["--json-schema", json.dumps(json_schema)])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if allowed_tools:
        for tool in allowed_tools:
            cmd.extend(["--allowedTools", tool])
    # Model is set via ANTHROPIC_MODEL env (Harbor pattern) — no --model flag.

    for attempt in range(_MAX_RETRIES + 1):
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or os.getcwd(),
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode != 0:
                stdout_str = stdout.decode()[:2000]
                result = {
                    "error": "nonzero_exit",
                    "returncode": proc.returncode,
                    "stderr": stderr.decode()[:2000],
                    "stdout": stdout_str,
                }
                cli_error = _extract_cli_error(result)
                if cli_error:
                    result.update(cli_error)
                    status = result.get("api_error_status")
                    subtype = result.get("subtype", "")
                    if subtype in ("error_max_turns",) or (
                        isinstance(status, int) and 400 <= status < 500 and status != 429
                    ):
                        result["error"] = subtype or f"api_error_{status}"
                        result["is_error"] = True
                        return result  # Don't retry — structural/client error
                if attempt < _MAX_RETRIES:
                    backoff = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP) * (0.5 + random.random())
                    output_hint = _format_subprocess_hint(result["stderr"], result["stdout"])
                    print(
                        f"    -> Transient error (exit={result['returncode']}), "
                        f"retrying in {backoff:.0f}s... [{output_hint}]"
                    )
                    await asyncio.sleep(backoff)
                    continue
                return result
            return json.loads(stdout.decode())
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            if attempt < _MAX_RETRIES:
                backoff = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP) * (0.5 + random.random())
                print(f"    -> Timeout, retrying in {backoff:.0f}s...")
                await asyncio.sleep(backoff)
                continue
            return {"error": "timeout"}
        except json.JSONDecodeError:
            # json_parse is not transient — don't retry
            return {"error": "json_parse", "raw": stdout.decode()[:2000] if stdout else ""}


# Module-level lock — created lazily to avoid issues with event loop not running at import time
_state_lock: asyncio.Lock | None = None


def _get_state_lock() -> asyncio.Lock:
    global _state_lock
    if _state_lock is None:
        _state_lock = asyncio.Lock()
    return _state_lock


async def save_state_locked(state, state_file: str) -> None:
    """Save state with lock to prevent concurrent JSON writes."""
    async with _get_state_lock():
        state.save(state_file)


def summarize(text: str, *, model: str = _SUMMARY_MODEL, max_chars: int = 1500) -> str:
    """Generate a 1-2 sentence summary via the NVIDIA inference gateway."""
    try:
        import openai

        api_key, base_url = get_openai_gateway_creds()
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize what was changed in 1-2 sentences (max 30 words). "
                        "Focus on WHICH FILES changed and WHAT was fixed. "
                        "No markdown, no headers, no preamble — just the summary sentence.\n\n"
                        f"{text[:max_chars]}"
                    ),
                }
            ],
            max_tokens=80,
            temperature=0,
        )
        summary = resp.choices[0].message.content.strip()
        if summary:
            return summary[:150]
    except Exception as e:
        print(f"    -> Summary failed ({e})")
    return "(summary unavailable)"


async def _run_fix_attempt(task: TaskState, issue: str, prompt: str) -> bool:
    """Execute one fix attempt with the given pre-built prompt. Shared by all fix entry points."""
    result = await run_claude_async(
        prompt,
        allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
        model=_cfg.LLM_STEP_MODEL or None,
    )

    if "error" in result:
        stderr_hint = result.get("stderr", "")[:200].strip().replace("\n", " ")
        task.fix_history.append(f"ERROR: {result['error']} — {stderr_hint}")
        print(f"    -> Fix attempt {task.fix_attempts} ERROR: {result['error']} [{stderr_hint}]")
        return False

    full_response = result.get("result", "")
    summary = await asyncio.get_running_loop().run_in_executor(None, summarize, full_response)
    task.fix_history.append({"summary": summary, "full": full_response})
    print(f"    -> Fix attempt {task.fix_attempts} applied: {summary[:80]}")

    if task.task_dir:
        fix_content = (
            f"# Fix Attempt {task.fix_attempts}\n\n"
            f"**Issue:** {issue}\n\n"
            f"**Summary:** {summary}\n\n"
            f"## Full Response\n\n{full_response}"
        )
        diag_path = _next_diagnostic_path(task.task_dir, "fix")
        _write_diagnostic(diag_path, fix_content)

    return True


def _build_fix_history(task: TaskState) -> str:
    return "\n".join(
        f"  Attempt {i + 1}: {h.get('summary', h) if isinstance(h, dict) else str(h)[:200]}"
        for i, h in enumerate(task.fix_history)
    )


async def _attempt_fix_docker_async(task: TaskState, issue: str) -> bool:
    """Attempt a fix scoped to environment/Dockerfile only — used for Docker build failures."""
    if task.fix_attempts >= _cfg.MAX_FIX_ATTEMPTS:
        print(f"    -> MAX ATTEMPTS ({_cfg.MAX_FIX_ATTEMPTS}) reached, giving up")
        return False
    prompt = fix_docker_prompt(task.task_dir, issue, task.fix_attempts + 1, _build_fix_history(task))
    task.fix_attempts += 1
    return await _run_fix_attempt(task, issue, prompt)


async def _attempt_fix_f2p_p2p_classify_async(task: TaskState, issue: str) -> bool:
    """Attempt a fix for F2P/P2P classification failures (Dockerfile + test discovery)."""
    if task.fix_attempts >= _cfg.MAX_FIX_ATTEMPTS:
        print(f"    -> MAX ATTEMPTS ({_cfg.MAX_FIX_ATTEMPTS}) reached, giving up")
        return False
    history = _build_fix_history(task)
    prompt = fix_f2p_p2p_classify_prompt(task.task_dir, issue, task.fix_attempts + 1, history)
    task.fix_attempts += 1
    return await _run_fix_attempt(task, issue, prompt)


def _shelve(task: TaskState, reason: str) -> None:
    task.stage = Stage.NEEDS_FIX
    task.needs_human_review = True
    task.human_review_reason = f"{reason} after {task.fix_attempts} attempts"


async def _fix_docker_or_shelve_async(task: TaskState, issue: str, reason: str) -> bool:
    """Attempt a Docker fix; shelve the task if all attempts are exhausted."""
    if await _attempt_fix_docker_async(task, issue):
        return True
    _shelve(task, reason)
    return False


async def _fix_f2p_p2p_classify_or_shelve_async(task: TaskState, issue: str, reason: str) -> bool:
    """Attempt a classification fix; shelve the task if all attempts are exhausted."""
    if await _attempt_fix_f2p_p2p_classify_async(task, issue):
        return True
    _shelve(task, reason)
    return False
