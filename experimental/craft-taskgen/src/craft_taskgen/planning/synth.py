# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gold-plan synthesis.

Per candidate: one Opus 4.6 call over (spec + PR description + PR diff) →
``gold_plan`` recorded on the candidate JSON + a human-readable ``.md``
sidecar. Always runs as the first step of the scorer pipeline so every
candidate gets a plan artifact regardless of downstream outcome.

Calls the NVIDIA inference gateway directly via litellm (same endpoint
craft-taskgen/search and craft-iterative-planning/v1 hit). No new SDK
dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm

from craft_taskgen.planning.prompts import build_gold_plan_prompt

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "openai/aws/anthropic/bedrock-claude-opus-4-6"
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_MAX_TOKENS = 32768
DEFAULT_MAX_RETRIES = 5


def _default_base_url() -> str:
    """Inference gateway base URL, from `OPENAI_BASE_URL`. No endpoint is hardcoded."""
    return os.environ.get("OPENAI_BASE_URL", "")


@dataclass
class SynthConfig:
    model: str = DEFAULT_MODEL
    base_url: str = field(default_factory=_default_base_url)
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_retries: int = DEFAULT_MAX_RETRIES
    overwrite: bool = False


def fetch_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch a PR's unified diff via the gh CLI."""
    owner, name = repo.split("/", 1)
    proc = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{owner}/{name}/pulls/{pr_number}",
            "-H",
            "Accept: application/vnd.github.v3.diff",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning("gh api diff failed for %s#%s: %s", repo, pr_number, proc.stderr[:300])
        return ""
    return proc.stdout


def fetch_pr_metadata(repo: str, pr_number: int) -> dict[str, Any]:
    """Fetch PR title + body via gh CLI."""
    owner, name = repo.split("/", 1)
    proc = subprocess.run(
        ["gh", "api", f"repos/{owner}/{name}/pulls/{pr_number}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning("gh api pull for %s#%s failed: %s", repo, pr_number, proc.stderr[:300])
        return {"title": "", "body": "", "linked_issues": []}
    data = json.loads(proc.stdout)
    return {
        "title": data.get("title", "") or "",
        "body": data.get("body", "") or "",
        "linked_issues": [],
    }


def _build_context(candidate: dict[str, Any], diff: str, pr_meta: dict[str, Any]) -> dict[str, Any]:
    issue_context = ""
    for iss in pr_meta.get("linked_issues", []) or []:
        issue_context += (
            f"\n### Issue #{iss.get('number', '?')}: {iss.get('title', '')}\n{iss.get('body', '')}\n"
        )
    return {
        "repo": candidate["repo"],
        "title": pr_meta.get("title") or candidate.get("title") or candidate.get("task_name", ""),
        "body": pr_meta.get("body") or candidate.get("body") or candidate.get("spec") or "",
        "issue_context": issue_context.strip(),
        "category": candidate.get("category", "other"),
        "source_files_modified": list(candidate.get("src_files", [])),
        "source_files_added": list(candidate.get("source_files_added", [])),
        "test_files": list(candidate.get("test_files", [])),
        "diff": diff,
    }


def _extract_json_plan(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


async def _call_opus(prompt: str, sem: asyncio.Semaphore, cfg: SynthConfig) -> dict[str, Any]:
    """Single completion against the NVIDIA gateway via litellm. Retries 429s."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    for attempt in range(cfg.max_retries):
        async with sem:
            try:
                response = await litellm.acompletion(
                    model=cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=cfg.max_tokens,
                    api_key=api_key,
                    api_base=cfg.base_url,
                    timeout=600,
                )
                text = response.choices[0].message.content or ""
                return _extract_json_plan(text)
            except Exception as exc:
                msg = str(exc)
                if "429" in msg and attempt < cfg.max_retries - 1:
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        "Rate limited, waiting %ds (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        cfg.max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
    raise RuntimeError("Exhausted retries without returning a response")


async def synthesize_one(
    candidate: dict[str, Any],
    diff: str,
    pr_meta: dict[str, Any],
    sem: asyncio.Semaphore,
    cfg: SynthConfig,
) -> str:
    ctx = _build_context(candidate, diff, pr_meta)
    result = await _call_opus(build_gold_plan_prompt(ctx), sem, cfg)
    plan = result.get("gold_plan", "")
    if not plan.strip():
        raise ValueError(f"Opus returned empty gold_plan for {candidate.get('task_name')}")
    return plan


async def _run_async(
    candidate_paths: list[Path],
    cfg: SynthConfig,
    plans_dir: Path,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    synthesized: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    async def _one(path: Path) -> None:
        candidate = json.loads(path.read_text())
        task_name = candidate.get("task_name") or path.stem

        if candidate.get("gold_plan") and not cfg.overwrite:
            logger.info("skip %s (gold_plan present)", task_name)
            skipped.append(task_name)
            return

        repo = candidate.get("repo")
        pr_number = candidate.get("pr")
        if not (repo and pr_number):
            failed.append((task_name, "candidate missing 'repo' or 'pr'"))
            return

        diff = fetch_pr_diff(repo, pr_number)
        if not diff:
            failed.append((task_name, "empty diff from gh api"))
            return
        pr_meta = fetch_pr_metadata(repo, pr_number)

        try:
            plan = await synthesize_one(candidate, diff, pr_meta, sem, cfg)
        except Exception as exc:
            failed.append((task_name, f"{type(exc).__name__}: {exc}"))
            logger.exception("synth failed for %s", task_name)
            return

        candidate["gold_plan"] = plan
        path.write_text(json.dumps(candidate, indent=2) + "\n")
        (plans_dir / f"{task_name}.md").write_text(plan + "\n" if not plan.endswith("\n") else plan)
        synthesized.append(task_name)
        logger.info("wrote gold plan for %s (%d chars)", task_name, len(plan))

    await asyncio.gather(*(_one(p) for p in candidate_paths))

    return {
        "synthesized": synthesized,
        "skipped": skipped,
        "failed": failed,
    }


def run_synth(
    candidates_dir: str,
    plans_dir: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    overwrite: bool = False,
    filter_task: str | None = None,
) -> dict[str, Any]:
    """Synthesize gold plans for every candidate.

    - Writes ``gold_plan`` onto each candidate JSON (idempotent; set
      ``overwrite=True`` to regenerate).
    - Emits sibling ``{plans_dir or candidates/gold-plans}/<task>.md`` for
      human reading.
    """
    cand_path = Path(candidates_dir)
    if not cand_path.is_dir():
        raise FileNotFoundError(f"candidates_dir not found: {candidates_dir}")

    out_plans = Path(plans_dir) if plans_dir else cand_path / "gold-plans"
    out_plans.mkdir(parents=True, exist_ok=True)

    paths = sorted(cand_path.glob("*.json"))
    if filter_task:
        paths = [p for p in paths if p.stem == filter_task or filter_task in p.stem]
    if not paths:
        raise FileNotFoundError(f"no candidate JSONs matched in {candidates_dir} (filter={filter_task!r})")

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        raise RuntimeError("OPENAI_API_KEY / ANTHROPIC_API_KEY not set (source .env for NVIDIA gateway)")

    cfg = SynthConfig(
        model=model,
        base_url=base_url or _default_base_url(),
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        overwrite=overwrite,
    )
    return asyncio.run(_run_async(paths, cfg, out_plans))
