#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone verifier for CRAFT Search tasks.

100% stdlib Python 3.12 — no pip dependencies required.
Reads the agent's /app/answer.json and compares against /tests/gold_answer.json.
Writes reward float to /logs/verifier/reward.txt and detailed metrics to
/logs/verifier/reward.json.
"""

from __future__ import annotations  # noqa: I001

import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime


# ---------------------------------------------------------------------------
# Scoring primitives (copied from src/scoring/primitives.py)
# ---------------------------------------------------------------------------


def precision(predicted: set[str], gold: set[str]) -> float:
    if not predicted:
        return 0.0
    return len(predicted & gold) / len(predicted)


def recall(predicted: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(predicted & gold) / len(gold)


def f1_score(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    p = precision(predicted, gold)
    r = recall(predicted, gold)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = ("/repo/", "repo/", "/code/", "code/", "./")


def normalize_file_path(path: str) -> str:
    """Normalize a file path: strip repo-root prefix, lowercase for case-insensitive match."""
    path = path.strip().lower()
    for prefix in _STRIP_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    return path.rstrip("/")


def normalize_function_name(name: str) -> str:
    """Normalize a fully-qualified function name: strip whitespace, preserve case."""
    return name.strip()


def normalize_function_tail(name: str) -> str:
    """Extract ClassName.method (last 2 dot-segments) for fuzzy matching.

    Handles mismatches like ``pendulum.DateTime.diff_for_humans`` vs
    ``pendulum.datetime.DateTime.diff_for_humans`` — same class and method,
    different module path granularity.
    """
    parts = name.strip().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else name.strip()


def normalize_files(items: list[str]) -> set[str]:
    """Normalize a list of file paths for comparison."""
    return {normalize_file_path(item) for item in items if item.strip()}


def normalize_functions(items: list[str]) -> set[str]:
    """Normalize a list of function names for comparison."""
    return {normalize_function_name(item) for item in items if item.strip()}


def _exact_or_tail_match_count(predicted: set[str], gold: set[str]) -> int:
    """Count matches with exact equality preferred, falling back to tail-equality.

    Uses one-to-one matching via multiset (Counter) intersection so that a
    single tail in `gold` cannot be over-credited by multiple `predicted`
    items sharing that tail. Symmetric in `predicted` and `gold`: swap the
    arguments to compute precision-hits vs recall-hits.

    Steps:
      1. Take exact set intersection — these consume both sides.
      2. For the remainder, take Counter-min of tail-normalized counts.
    """
    matched_exact = predicted & gold
    remaining_pred = predicted - matched_exact
    remaining_gold = gold - matched_exact
    pred_tails = Counter(normalize_function_tail(p) for p in remaining_pred)
    gold_tails = Counter(normalize_function_tail(g) for g in remaining_gold)
    fuzzy = sum((pred_tails & gold_tails).values())
    return len(matched_exact) + fuzzy


# ---------------------------------------------------------------------------
# LLM-as-Judge assertion coverage
# ---------------------------------------------------------------------------


def build_judge_prompt(assertions: list[str], explanation: str) -> str:
    """Build the prompt for the LLM judge to evaluate assertion coverage."""
    numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(assertions))
    return (
        "You are a lenient code review judge. Given an agent's explanation of code behavior "
        "and a list of factual assertions, determine which assertions are supported "
        "by the explanation.\n\n"
        "An assertion is SUPPORTED if the explanation conveys the same factual claim, "
        "even if using different words, different level of detail, or different phrasing. "
        "Focus on semantic meaning, not exact names or vocabulary.\n\n"
        "## Leniency Rules (apply all of these)\n"
        "- **Name equivalence**: If the assertion mentions `Foo.bar()` and the explanation "
        "describes the same behavior using a general term like 'the parser' or 'that method', "
        "mark SUPPORTED. Exact class/function names are NOT required.\n"
        "- **Abstraction level**: If the assertion says 'uses the visitor pattern' and the "
        "explanation describes the same behavior at implementation level (e.g., 'visit_select() "
        "is called'), or vice versa, mark SUPPORTED.\n"
        "- **Attribute aliases**: If the assertion references one attribute name and the "
        "explanation references a different name for the same data (e.g., a property vs its "
        "backing field), mark SUPPORTED.\n"
        "- **Mechanism equivalence**: If both describe the same end result through different "
        "intermediate steps, mark SUPPORTED. Implementation details may vary.\n\n"
        "Only mark NOT_SUPPORTED if the explanation genuinely does not address the claim, "
        "describes a contradictory mechanism, or is about a completely different topic.\n\n"
        f"## Agent's Explanation\n{explanation}\n\n"
        f"## Assertions to Evaluate\n{numbered}\n\n"
        "For each assertion, respond with SUPPORTED or NOT_SUPPORTED and a brief reason.\n"
        "Then on the final line, write: SCORE: X/Y (where X = supported count, Y = total)"
    )


def parse_judge_response(response: str, total: int) -> float:
    """Extract SCORE: X/Y from judge response. Returns X/Y as float, 0.0 on failure."""
    import re

    match = re.search(r"SCORE:\s*(\d+)\s*/\s*(\d+)", response)
    if not match:
        return 0.0
    x, y = int(match.group(1)), int(match.group(2))
    if y == 0:
        return 1.0
    # Use the total we expect, not what the LLM says, to avoid manipulation
    return min(x, total) / total


def llm_call(base_url: str, api_key: str, model: str, prompt: str) -> str:
    """Call an OpenAI-compatible chat completions endpoint using stdlib urllib."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def llm_judge_assertions(assertions: list[str], explanation: str) -> float:
    """Use LLM to judge what fraction of gold assertions are covered by the explanation.

    Retries up to 3 times on transient failures (timeouts, connection errors).

    Returns:
        Float 0.0-1.0 for coverage, or -1.0 sentinel if no judge is available.
    """
    if not assertions:
        return 1.0

    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("JUDGE_MODEL", "aws/anthropic/bedrock-claude-sonnet-4-6")

    if not api_key or not base_url:
        print(
            "[judge] Set OPENAI_API_KEY and OPENAI_BASE_URL (or the JUDGE_* overrides) in the environment",
            file=sys.stderr,
        )
        return -1.0  # sentinel: caller skips assertion weight

    prompt = build_judge_prompt(assertions, explanation)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(
            f"[judge] Calling {model} at {base_url} with {len(assertions)} assertions"
            f" (attempt {attempt}/{max_retries})",
            file=sys.stderr,
        )
        try:
            response = llm_call(base_url, api_key, model, prompt)
            print(f"[judge] Response: {response[:2000]}", file=sys.stderr)
            return parse_judge_response(response, len(assertions))
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as exc:
            print(f"[judge] Attempt {attempt} failed: {exc}", file=sys.stderr)
            if attempt < max_retries:
                import time

                time.sleep(2 * attempt)

    print("[judge] All retries exhausted", file=sys.stderr)
    return -1.0  # judge unavailable — fall back to nav-only scoring


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------


def load_json(path: str) -> dict | None:
    """Load JSON file, returning None on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def score(answer_path: str = "/app/answer.json", gold_path: str = "/tests/gold_answer.json") -> dict:
    """Score agent answer against gold answer. Returns metrics dict."""
    gold = load_json(gold_path)
    if gold is None:
        return {"error": "gold_answer.json missing or malformed", "reward": 0.0}

    answer = load_json(answer_path)
    if answer is None:
        return {
            "error": "answer.json missing or malformed",
            "reward": 0.0,
            "navigation_f1": 0.0,
            "assertion_coverage": 0.0,
        }

    # Normalize files and functions separately (files lowercase, functions preserve case)
    gold_files = normalize_files(gold.get("files", []))
    gold_functions = normalize_functions(gold.get("functions", []))
    alt_files = normalize_files(gold.get("alt_files", []))
    alt_functions = normalize_functions(gold.get("alt_functions", []))

    agent_files = normalize_files(answer.get("files", []))
    agent_functions = normalize_functions(answer.get("functions", []))

    # Files — strict recall (primary only) + lenient precision (primary | alt).
    # Naming an alt is not a false positive; alts do not substitute for primaries.
    expanded_files = gold_files | alt_files
    file_hits_strict = len(agent_files & gold_files)
    file_hits_lenient = len(agent_files & expanded_files)
    file_recall = file_hits_strict / len(gold_files) if gold_files else 1.0
    file_precision = file_hits_lenient / len(agent_files) if agent_files else 0.0
    file_f1 = (
        2 * file_precision * file_recall / (file_precision + file_recall)
        if file_precision + file_recall > 0
        else 0.0
    )
    # Lenient diagnostic (matches the prior behavior — alts credit recall, capped at 1).
    file_recall_lenient = min(file_hits_lenient / len(gold_files), 1.0) if gold_files else 1.0
    file_f1_lenient = (
        2 * file_precision * file_recall_lenient / (file_precision + file_recall_lenient)
        if file_precision + file_recall_lenient > 0
        else 0.0
    )

    # Functions — strict recall + lenient precision, both with tail-match leniency
    # for module-path granularity drift (`pkg.Mod.Cls.m` vs `pkg.Cls.m`).
    # Function matching uses one-to-one tail-fuzzy matching to prevent a single
    # tail in gold (e.g., '*.run', '*.__init__') from over-crediting multiple
    # agent functions that happen to share that tail across different modules.
    expanded_funcs = gold_functions | alt_functions
    agent_tails = {normalize_function_tail(f) for f in agent_functions}
    expanded_tails = {normalize_function_tail(f) for f in expanded_funcs}
    # Strict recall: gold (canonical only) ∩ agent, exact-or-tail one-to-one.
    func_recall_hits = _exact_or_tail_match_count(gold_functions, agent_functions)
    # Lenient recall: gold (canonical | alt) ∩ agent. Numerator uses one-to-one
    # against agent; denominator stays len(gold_functions). Capped at 1.0 below.
    func_recall_hits_lenient = _exact_or_tail_match_count(expanded_funcs, agent_functions)
    func_recall = func_recall_hits / len(gold_functions) if gold_functions else 1.0
    # Precision: agent ∩ (gold ∪ alt), one-to-one (no tail re-use).
    func_precision_hits = _exact_or_tail_match_count(agent_functions, expanded_funcs)
    func_precision = func_precision_hits / len(agent_functions) if agent_functions else 0.0
    func_f1 = (
        2 * func_precision * func_recall / (func_precision + func_recall)
        if func_precision + func_recall > 0
        else 0.0
    )
    func_recall_lenient = min(func_recall_hits_lenient / len(gold_functions), 1.0) if gold_functions else 1.0
    func_f1_lenient = (
        2 * func_precision * func_recall_lenient / (func_precision + func_recall_lenient)
        if func_precision + func_recall_lenient > 0
        else 0.0
    )

    # Per-axis F1 averaged is the headline navigation score (strict recall + lenient precision).
    has_files = bool(gold_files)
    has_functions = bool(gold_functions)
    if has_files and has_functions:
        nav_score = 0.5 * file_f1 + 0.5 * func_f1
        nav_score_lenient = 0.5 * file_f1_lenient + 0.5 * func_f1_lenient
        nav_recall = 0.5 * file_recall + 0.5 * func_recall
        nav_recall_lenient = 0.5 * file_recall_lenient + 0.5 * func_recall_lenient
    elif has_files:
        nav_score = file_f1
        nav_score_lenient = file_f1_lenient
        nav_recall = file_recall
        nav_recall_lenient = file_recall_lenient
    elif has_functions:
        nav_score = func_f1
        nav_score_lenient = func_f1_lenient
        nav_recall = func_recall
        nav_recall_lenient = func_recall_lenient
    else:
        nav_score = nav_score_lenient = 1.0
        nav_recall = nav_recall_lenient = 1.0

    # Combined F1 as diagnostic only (primary gold only)
    gold_set = gold_files | gold_functions
    agent_set = agent_files | agent_functions
    nav_f1 = f1_score(agent_set, gold_set)

    # Option B (secondary table): per-axis Jaccard / IoU.
    # IoU = |agent ∩ (primary∪alt)| / |agent ∪ primary| — alts in numerator
    # only, primary alone in denominator. Symmetric and robust to over- and
    # under-fetch equally. For functions we operate on tail-normalized sets so
    # tail-fuzzy matching is consistent on both sides of the ratio.
    file_iou_denom = len(agent_files | gold_files)
    file_iou = file_hits_lenient / file_iou_denom if file_iou_denom > 0 else 1.0

    gold_tails_set = {normalize_function_tail(f) for f in gold_functions}
    func_iou_num = len(agent_tails & expanded_tails)
    func_iou_denom = len(agent_tails | gold_tails_set)
    func_iou = func_iou_num / func_iou_denom if func_iou_denom > 0 else 1.0

    if has_files and has_functions:
        nav_iou = 0.5 * file_iou + 0.5 * func_iou
    elif has_files:
        nav_iou = file_iou
    elif has_functions:
        nav_iou = func_iou
    else:
        nav_iou = 1.0

    # Assertion coverage via LLM judge
    gold_assertions = gold.get("assertions", [])
    agent_explanation = answer.get("explanation", "")
    assert_cov = llm_judge_assertions(gold_assertions, agent_explanation)

    if assert_cov < 0:
        # No judge available — pure navigation score
        reward = nav_score
        has_judge = False
    elif gold_assertions:
        reward = 0.5 * nav_score + 0.5 * assert_cov
        has_judge = True
    else:
        reward = nav_score
        has_judge = False

    return {
        "reward": round(reward, 4),
        "file_precision": round(file_precision, 4),
        "file_recall": round(file_recall, 4),
        "file_recall_lenient": round(file_recall_lenient, 4),
        "file_f1": round(file_f1, 4),
        "file_f1_lenient": round(file_f1_lenient, 4),
        "function_precision": round(func_precision, 4),
        "function_recall": round(func_recall, 4),
        "function_recall_lenient": round(func_recall_lenient, 4),
        "function_f1": round(func_f1, 4),
        "function_f1_lenient": round(func_f1_lenient, 4),
        "navigation_score": round(nav_score, 4),
        "navigation_score_lenient": round(nav_score_lenient, 4),
        "navigation_recall": round(nav_recall, 4),
        "navigation_recall_lenient": round(nav_recall_lenient, 4),
        "navigation_f1": round(nav_f1, 4),
        "file_iou": round(file_iou, 4),
        "function_iou": round(func_iou, 4),
        "navigation_iou": round(nav_iou, 4),
        "assertion_coverage": round(max(assert_cov, 0.0), 4),
        "gold_files": sorted(gold_files),
        "gold_functions": sorted(gold_functions),
        "alt_files": sorted(alt_files),
        "alt_functions": sorted(alt_functions),
        "agent_files": sorted(agent_files),
        "agent_functions": sorted(agent_functions),
        "gold_items": sorted(gold_set),
        "agent_items": sorted(agent_set),
        "has_assertions": bool(gold_assertions),
        "judge_available": has_judge,
        "agent_explanation": agent_explanation,
    }


def extract_process_metrics(agent_dir: str = "/logs/agent") -> dict:
    """Extract process metrics from agent transcript artifacts.

    Tries the ATIF trajectory first (codex, opencode, and anything else that
    produces `trajectory.json`). Falls back to the claude-code JSONL stream
    when trajectory.json is absent — this is the common case for claude-code,
    whose post-run trajectory converter races with session-file ownership and
    frequently fails to emit trajectory.json.

    Best-effort: returns {} when nothing can be parsed.
    """
    traj_path = os.path.join(agent_dir, "trajectory.json")
    if os.path.exists(traj_path):
        try:
            with open(traj_path) as f:
                traj = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            traj = None
        if traj:
            return _metrics_from_atif(traj)

    cc_path = os.path.join(agent_dir, "claude-code.txt")
    if os.path.exists(cc_path):
        return _metrics_from_claude_code(cc_path)

    return {}


def _metrics_from_atif(traj: dict) -> dict:
    """Parse the ATIF schema that codex + opencode (and harbor's converter) emit."""
    steps = traj.get("steps", [])
    fm = traj.get("final_metrics", {})

    tool_breakdown: dict[str, int] = {}
    empty_obs = 0
    total_tool_calls = 0
    for step in steps:
        tool_calls = step.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function_name", "unknown")
            tool_breakdown[fn] = tool_breakdown.get(fn, 0) + 1
            total_tool_calls += 1
        if tool_calls and not step.get("observation"):
            empty_obs += 1

    execution_time = None
    if len(steps) >= 2:
        try:
            first_ts = steps[0].get("timestamp", "")
            last_ts = steps[-1].get("timestamp", "")
            if first_ts and last_ts:
                t0 = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                execution_time = round((t1 - t0).total_seconds(), 1)
        except (ValueError, TypeError):
            pass

    result = {
        "tool_call_count": total_tool_calls,
        "tool_call_breakdown": tool_breakdown,
        "agent_steps": len(steps),
        "empty_observation_count": empty_obs,
        "empty_observation_rate": round(empty_obs / total_tool_calls, 4) if total_tool_calls else 0.0,
        "input_tokens": fm.get("total_prompt_tokens"),
        "output_tokens": fm.get("total_completion_tokens"),
        "cached_tokens": fm.get("total_cached_tokens"),
        "transcript_source": "trajectory.json",
    }
    if execution_time is not None:
        result["execution_time_sec"] = execution_time
    return result


def _metrics_from_claude_code(path: str) -> dict:
    """Parse the claude-code stream-JSON transcript (one JSON object per line).

    Each `type: "assistant"` line is a turn; its content array may contain
    `tool_use` blocks. Assistant-level `usage` has input/output/cache counts.
    """
    tool_breakdown: dict[str, int] = {}
    agent_steps = 0
    total_tool_calls = 0
    input_uncached = 0
    output_tokens = 0
    cache_read = 0
    cache_create = 0

    try:
        f = open(path, errors="replace")
    except OSError:
        return {}
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant":
                continue
            agent_steps += 1
            msg = rec.get("message") or {}
            for c in msg.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    name = c.get("name", "unknown")
                    tool_breakdown[name] = tool_breakdown.get(name, 0) + 1
                    total_tool_calls += 1
            usage = msg.get("usage") or {}
            input_uncached += usage.get("input_tokens", 0) or 0
            output_tokens += usage.get("output_tokens", 0) or 0
            cache_read += usage.get("cache_read_input_tokens", 0) or 0
            cache_create += usage.get("cache_creation_input_tokens", 0) or 0

    if agent_steps == 0:
        return {}

    total_input = input_uncached + cache_read + cache_create

    return {
        "tool_call_count": total_tool_calls,
        "tool_call_breakdown": tool_breakdown,
        "agent_steps": agent_steps,
        "empty_observation_count": 0,
        "empty_observation_rate": 0.0,
        "input_tokens": total_input,
        "output_tokens": output_tokens,
        "cached_tokens": cache_read,
        "transcript_source": "claude-code.txt",
    }


def main() -> None:
    metrics = score()

    # Add process metrics from agent trajectory
    process = extract_process_metrics()
    if process:
        metrics["process_metrics"] = process

    os.makedirs("/logs/verifier", exist_ok=True)

    # Write reward float
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(str(metrics["reward"]))

    # Write detailed metrics
    with open("/logs/verifier/reward.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Reward: {metrics['reward']}")
    if process:
        print(
            f"Process: {process.get('tool_call_count', 0)} tool calls, "
            f"{process.get('agent_steps', 0)} steps, "
            f"{process.get('input_tokens', '?')} input tokens"
        )
    if "error" in metrics:
        print(f"Error: {metrics['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
