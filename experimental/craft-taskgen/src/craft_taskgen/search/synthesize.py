# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthesize Search tasks from Tools implementation problems.

Three seed strategies feed into a shared LLM synthesis + cross-judging pipeline:

  A) Feature-recon seed -- developer planning an implementation
  B) Developer-question seed -- natural questions while working on a feature
  C) Test-grounded seed -- questions about behavior verified by gold tests

All seeds -> 3-model generation -> cross-judging (4 dimensions) -> select best.

Ported from craft-bench scripts/search/synthesize_from_t2.py.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import litellm

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Models (LiteLLM names for the NVIDIA inference gateway)
# ---------------------------------------------------------------------------

SYNTH_MODELS = {
    "sonnet-4.6": "openai/aws/anthropic/bedrock-claude-sonnet-4-6",
    "gemini-3.1-pro": "openai/gcp/google/gemini-3.1-pro-preview",
    "gpt-5.4": "openai/us/azure/openai/gpt-5.4",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYNTH_PROMPT = """\
You are an expert benchmark designer creating a codebase navigation task.

## Repository: {repo}

## Repo Map

Below is a structural map showing file paths, class hierarchies, and \
function/method signatures ranked by architectural importance:

```
{repo_map}
```

## Context

{scenario}

## Your Task

Generate 1 codebase navigation question with a complete gold answer.

### Question requirements

The question must sound like a message a developer would post in Slack -- \
typed quickly, frustrated or confused. NOT AI-composed.

What makes it sound HUMAN:
- Terse. 1-3 sentences, 15-40 words.
- Imprecise language. "this thing", "the auth stuff"
- ONE clear confusion, not a survey
- Mention ONE specific thing observed, not a symmetric comparison

BANNED (LLM tells):
- "I suspect", "I'd expect", "interestingly"
- Balanced A-vs-B phrasing
- Meta-questions: "What determines...", "What code is responsible..."
- More than 2 named API concepts per question

Adversarial -- DO NOT LEAK THE ANSWER:
- Do NOT name specific source files, internal modules, or private functions
- Do NOT reference test names or test behavior
- Public API methods from docs are fine (limit 1-2)
- KEY TEST: If someone could write a partial answer from the question alone, \
it's too detailed.

### Gold answer requirements

ONLY reference files and functions you can see in the repo map above. \
Do NOT invent function names.

- **files**: 2-5 specific files the agent should find (relative paths from repo root)
- **functions**: 2-6 fully qualified function/method names \
(e.g., "module.submodule.ClassName.method_name"). Must be METHODS, not just class names.
- **explanation**: 3-5 sentences -- what the code does, how pieces connect, WHY
- **assertions**: 3-6 specific factual claims extracted FROM your explanation

## Output Format

Return ONLY valid JSON, no markdown fences or commentary:

{{
  "instruction": "natural developer question...",
  "gold_answer": {{
    "files": ["path/to/file.py", "path/to/other.py"],
    "functions": ["module.Class.method", "module.other_function"],
    "explanation": "3-5 sentence explanation...",
    "assertions": ["claim 1", "claim 2", "claim 3"]
  }},
  "tier": "{difficulty}",
  "oracle_min_calls": 8
}}"""

JUDGE_PROMPT = """\
You are judging a synthesized benchmark question + gold answer.

## Question
{question}

## Gold Answer
Files: {gold_files}
Functions: {gold_functions}
Explanation: {gold_explanation}

## Seed context
{scenario}

Rate (1-5 each):
1. naturalness: Does the question sound like a real person typed it quickly? \
Not AI, not textbook. (5=completely human)
2. scope_preservation: Do the gold files/functions actually answer this question? \
Are the functions real methods (not just class names)? (5=perfectly scoped)
3. solvability: Could an expert agent following this question arrive at the gold \
answer? (5=definitely solvable)
4. not_formulaic: Does the question avoid LLM patterns -- no balanced comparisons, \
no hedge phrases, no keyword stuffing? (5=no LLM tells)

JSON only:
{{"naturalness": N, "scope_preservation": N, "solvability": N, \
"not_formulaic": N, "reasoning": "..."}}"""

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _llm_call(model: str, prompt: str, api_key: str, base_url: str, max_tokens: int = 200) -> str:
    """Make a single LLM call with retries."""
    temp = 1.0 if "gpt-5" in model else 0.9
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temp,
        api_key=api_key,
        api_base=base_url,
        num_retries=3,
        timeout=60,
    )
    return response.choices[0].message.content.strip()


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from model output."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_judge_response(text: str) -> dict:
    """Parse JSON judge response, handling markdown fencing."""
    content = text.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[: content.rfind("```")]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": text[:200]}


# ---------------------------------------------------------------------------
# Seed builders
# ---------------------------------------------------------------------------


def build_seeds_a(ctx: dict[str, Any], repo: str) -> list[dict[str, Any]]:
    """Seed A: Feature-recon. Developer planning an implementation."""
    repo_map = ctx.get("repo_map", "")
    difficulty = ctx.get("difficulty", "hard")
    return [
        {
            "angle": "feature_recon",
            "scenario": (f"A developer is about to implement this feature in {repo}: {ctx['instruction']}"),
            "repo_map": repo_map,
            "difficulty": difficulty,
        },
        {
            "angle": "bug_investigation",
            "scenario": (
                f"A developer is debugging an issue with this behavior in {repo}: "
                f"{ctx['instruction']} — they need to trace how the affected code "
                f"path is currently wired so they can reproduce and fix the bug."
            ),
            "repo_map": repo_map,
            "difficulty": difficulty,
        },
        {
            "angle": "refactor_prep",
            "scenario": (
                f"A developer is preparing to refactor the code in {repo} that "
                f"implements: {ctx['instruction']} — they need to locate every "
                f"call site and ensure they understand the existing contract "
                f"before moving or renaming things."
            ),
            "repo_map": repo_map,
            "difficulty": difficulty,
        },
    ]


def build_seeds_b(ctx: dict[str, Any], repo: str) -> list[dict[str, Any]]:
    """Seed B: Developer questions. Multiple angles per problem."""
    repo_map = ctx.get("repo_map", "")
    difficulty = ctx.get("difficulty", "hard")
    return [
        {
            "angle": "architecture",
            "scenario": (
                f"A developer is modifying {repo} to add: {ctx['instruction']} -- "
                f"they need to understand how the existing architecture works in the "
                f"area they'll be changing."
            ),
            "repo_map": repo_map,
            "difficulty": difficulty,
        },
        {
            "angle": "extension_point",
            "scenario": (
                f"A developer wants to extend {repo} with new functionality: "
                f"{ctx['instruction']} -- they're looking for the extension point "
                f"and how similar features hook in."
            ),
            "repo_map": repo_map,
            "difficulty": difficulty,
        },
        {
            "angle": "data_flow",
            "scenario": (
                f"A developer is tracing the data/control flow for this in {repo}: "
                f"{ctx['instruction']} -- they want to see where inputs are "
                f"validated, transformed, dispatched, and where downstream code "
                f"finally consumes them."
            ),
            "repo_map": repo_map,
            "difficulty": difficulty,
        },
        {
            "angle": "error_paths",
            "scenario": (
                f"A developer is investigating how errors and edge cases are "
                f"handled for: {ctx['instruction']} in {repo} -- they need to "
                f"know which exceptions get raised, where they are caught, and "
                f"how malformed or missing inputs degrade."
            ),
            "repo_map": repo_map,
            "difficulty": difficulty,
        },
    ]


def build_seeds_c(ctx: dict[str, Any], repo: str) -> list[dict[str, Any]]:
    """Seed C: Test-grounded. Questions about behavior verified by gold tests."""
    gold_meta = ctx["gold_test_metadata"]
    if gold_meta.get("is_documentation_only"):
        return []

    test_functions = gold_meta.get("test_functions", [])
    if not test_functions:
        return []

    groups = _group_test_functions(test_functions, gold_meta.get("repo_imports", {}))
    seeds = []

    repo_map = ctx.get("repo_map", "")
    difficulty = ctx.get("difficulty", "hard")

    for _group_name, group_funcs in groups.items():
        if not group_funcs:
            continue

        behaviors = []
        for tf in group_funcs:
            behavior = tf["name"].replace("test_", "", 1).replace("__", " with ").replace("_", " ")
            behaviors.append(behavior)
        summary = "; ".join(behaviors[:4])
        if len(behaviors) > 4:
            summary += f" (and {len(behaviors) - 4} more)"

        seeds.append(
            {
                "angle": "test_behavior",
                "scenario": (
                    f"In {repo}, there's code that handles: {summary}. "
                    f"A developer wants to understand how this behavior is implemented "
                    f"because they need to modify or extend it."
                ),
                "repo_map": repo_map,
                "difficulty": difficulty,
            },
        )

    return seeds


SEED_BUILDERS = {
    "A": build_seeds_a,
    "B": build_seeds_b,
    "C": build_seeds_c,
}

# ---------------------------------------------------------------------------
# Approach C helpers
# ---------------------------------------------------------------------------


def _group_test_functions(
    test_functions: list[dict], repo_imports: dict[str, list[str]]
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for tf in test_functions:
        primary = None
        for imp in tf.get("repo_imports", []):
            if imp.startswith("tests."):
                continue
            parts = imp.split(".")
            primary = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            break
        if not primary:
            parts = tf["name"].split("_")
            primary = "_".join(parts[1:3]) if len(parts) >= 3 else tf["name"]
        groups.setdefault(primary, []).append(tf)
    return groups


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def generate_candidates(seed: dict, repo: str, api_key: str, base_url: str) -> dict[str, dict]:
    """Generate candidate question+gold from all 3 models. Returns parsed JSON per model."""
    repo_map = seed.get("repo_map", "(repo map not available)")
    difficulty = seed.get("difficulty", "hard")

    prompt = SYNTH_PROMPT.format(
        scenario=seed["scenario"],
        repo=repo,
        repo_map=repo_map,
        difficulty=difficulty,
    )

    candidates: dict[str, dict] = {}
    for label, model in SYNTH_MODELS.items():
        raw = ""
        try:
            raw = _llm_call(model, prompt, api_key, base_url, max_tokens=1024)
            parsed = json.loads(_strip_markdown_fences(raw))
            if not isinstance(parsed, dict):
                parsed = parsed[0] if isinstance(parsed, list) and parsed else {}
            candidates[label] = parsed
        except (json.JSONDecodeError, AttributeError) as e:
            candidates[label] = {"error": f"JSON parse: {e}", "raw": raw[:200]}
        except Exception as e:
            candidates[label] = {"error": str(e)}
    return candidates


def cross_judge(seed: dict, candidates: dict[str, dict], api_key: str, base_url: str) -> dict[str, dict]:
    """Cross-judge: each model's question+gold is judged by the other two."""
    scored: dict[str, dict] = {}
    model_labels = list(candidates.keys())
    score_keys = ["naturalness", "scope_preservation", "solvability", "not_formulaic"]

    for target_label in model_labels:
        cand = candidates[target_label]
        if "error" in cand:
            scored[target_label] = {**cand, "judge_scores": {}, "mean": 0.0}
            continue

        question = cand.get("instruction", "")
        gold = cand.get("gold_answer", {})
        gold_files_str = ", ".join(gold.get("files", []))
        gold_funcs_str = ", ".join(gold.get("functions", []))
        gold_expl = gold.get("explanation", "")

        judge_labels = [ml for ml in model_labels if ml != target_label]
        judge_scores: dict[str, dict] = {}

        for judge_label in judge_labels:
            judge_model = SYNTH_MODELS[judge_label]
            prompt = JUDGE_PROMPT.format(
                question=question,
                scenario=seed["scenario"],
                gold_files=gold_files_str,
                gold_functions=gold_funcs_str,
                gold_explanation=gold_expl,
            )
            try:
                response = _llm_call(judge_model, prompt, api_key, base_url)
                parsed = _parse_judge_response(response)
                if "parse_error" not in parsed:
                    mean_score = sum(parsed.get(k, 0) for k in score_keys) / len(score_keys)
                    judge_scores[judge_label] = {
                        "scores": {k: parsed.get(k, 0) for k in score_keys},
                        "mean": round(mean_score, 2),
                        "reasoning": parsed.get("reasoning", ""),
                    }
                else:
                    judge_scores[judge_label] = {"error": parsed.get("raw", "")}
            except Exception as e:
                judge_scores[judge_label] = {"error": str(e)}

        valid_scores = [js["mean"] for js in judge_scores.values() if "mean" in js]
        mean = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0

        scored[target_label] = {
            **cand,
            "question": question,
            "judge_scores": judge_scores,
            "mean": mean,
        }

    return scored


def leaks_gold(question: str, gold_files: list[str], gold_functions: list[str]) -> bool:
    """Check if the question leaks gold answer contents."""
    q_lower = question.lower()
    for f in gold_files:
        if f.lower() in q_lower:
            return True
        module_path = f.lower().replace("/", ".").replace(".py", "")
        if module_path in q_lower and len(module_path) > 10:
            return True
    for fn in gold_functions:
        if fn.lower() in q_lower:
            return True
        leaf = fn.split(".")[-1]
        if leaf.startswith("_") and leaf.lower() in q_lower:
            return True
    return False


def process_seed(
    seed: dict,
    ctx: dict,
    repo: str,
    approach: str,
    api_key: str,
    base_url: str,
) -> dict:
    """Full pipeline for one seed: generate question+gold -> cross-judge -> select."""
    raw_candidates = generate_candidates(seed, repo, api_key, base_url)
    scored = cross_judge(seed, raw_candidates, api_key, base_url)

    for label, cand in scored.items():
        q = cand.get("question", "")
        gold = cand.get("gold_answer", {})
        gf = gold.get("files", [])
        gfn = gold.get("functions", [])
        if q and "error" not in cand and leaks_gold(q, gf, gfn):
            cand["rejected"] = "leaks_gold"
            cand["mean"] = 0.0

    best_label = max(scored, key=lambda k: scored[k].get("mean", 0))
    best = scored[best_label]
    best_gold = best.get("gold_answer", {})

    uid = uuid.uuid4().hex[:8]
    return {
        "id": f"craft-{repo}-{approach.lower()}-{uid}",
        "parent_t2_task": ctx["task_id"],
        "repo": repo,
        "approach": approach,
        "seed_angle": seed["angle"],
        "scenario": seed["scenario"],
        "gold_files": best_gold.get("files", []),
        "gold_functions": best_gold.get("functions", []),
        "gold_assertions": best_gold.get("assertions", []),
        "gold_explanation": best_gold.get("explanation", ""),
        "candidates": scored,
        "selected": best_label,
        "question": best.get("question", ""),
        "mean_judge_score": best.get("mean", 0.0),
        "status": "pending",
    }


# ---------------------------------------------------------------------------
# Output conversion (for Harbor adapter compatibility)
# ---------------------------------------------------------------------------


def result_to_search_task(result: dict) -> dict[str, Any]:
    """Convert a pipeline result to the search_tasks.json format."""
    return {
        "id": result["id"],
        "parent_t2_task": result["parent_t2_task"],
        "repo": result["repo"],
        "approach": result["approach"],
        "dimension": "search",
        "tier": "hard",
        "instruction": result["question"],
        "gold_answer": {
            "dimension": "search",
            "files": result["gold_files"],
            "functions": result["gold_functions"],
            "explanation": result.get("gold_explanation", ""),
            "assertions": result.get("gold_assertions", []),
            "alt_files": [],
            "alt_functions": [],
        },
        "oracle_min_calls": None,
        "ground_truth_source": {
            "fact_type": f"t2_{result.get('seed_angle', 'unknown')}",
            "details": {
                "parent_t2_task": result["parent_t2_task"],
                "approach": result["approach"],
                "seed_angle": result.get("seed_angle", ""),
                "mean_judge_score": result.get("mean_judge_score", 0),
                "selected_model": result.get("selected", ""),
            },
        },
    }


def repo_name_from_url(url: str) -> str:
    cleaned = url.rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    return cleaned.split("/")[-1] or "unknown"


# ---------------------------------------------------------------------------
# Top-level run function (called by step_synthesize)
# ---------------------------------------------------------------------------


def run_synthesis(
    *,
    contexts_dir: str,
    output_dir: str | None = None,
    approaches: list[str] | None = None,
    task_filter: str | None = None,
    concurrency: int = 3,
    dry_run: bool = False,
) -> None:
    """Run the full synthesis + cross-judging pipeline.

    Args:
        contexts_dir: Directory containing _all_contexts.json (from extract step).
        output_dir: Base output directory. Each approach writes to approach-{a,b,c}/ under this.
                    Defaults to contexts_dir.
        approaches: List of approach letters to run (default: ["A", "B", "C"]).
        task_filter: Single task ID to process (optional).
        concurrency: Number of parallel LLM calls.
        dry_run: If True, show seeds without calling LLMs.
    """
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    if not dry_run and (not api_key or not base_url):
        print(
            "ERROR: Set JUDGE_API_KEY/OPENAI_API_KEY and JUDGE_BASE_URL/OPENAI_BASE_URL.",
            file=sys.stderr,
        )
        raise RuntimeError("Missing API key or base URL for synthesis")

    combined_path = os.path.join(contexts_dir, "_all_contexts.json")
    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"{combined_path} not found. Run the extract step first.")
    with open(combined_path) as f:
        all_contexts = json.load(f)
    if task_filter:
        all_contexts = [c for c in all_contexts if c["task_id"] == task_filter]

    if approaches is None:
        approaches = ["A", "B", "C"]

    for approach in approaches:
        approach_output = output_dir or contexts_dir
        approach_dir = os.path.join(approach_output, f"approach-{approach.lower()}")
        os.makedirs(approach_dir, exist_ok=True)

        build_seeds = SEED_BUILDERS[approach]
        print(f"\n=== Approach {approach} ===")

        work_items: list[tuple[dict, dict, str]] = []
        for ctx in all_contexts:
            repo = repo_name_from_url(ctx["solve_info"]["upstream_url"])
            seeds = build_seeds(ctx, repo)
            for seed in seeds:
                work_items.append((seed, ctx, repo))

        if dry_run:
            for seed, ctx, repo in work_items:
                print(f"  {ctx['task_id']} [{seed['angle']}]:")
                print(f"    scenario: {seed['scenario'][:100]}...")
                print(f"    gold_files: {seed.get('gold_files', [])}")
                print(f"    gold_functions: {seed.get('gold_functions', [])[:3]}")
            print(f"\n  {len(work_items)} seeds (dry-run, no LLM calls)")
            continue

        print(f"  {len(work_items)} seeds x 3 models + cross-judge")
        all_results: list[dict] = []
        errors = 0
        start = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(process_seed, seed, ctx, repo, approach, api_key, base_url): (seed, ctx)
                for seed, ctx, repo in work_items
            }
            for i, future in enumerate(as_completed(futures), 1):
                seed, ctx = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    q = result["question"][:80] if result.get("question") else "(empty)"
                    score = result.get("mean_judge_score", 0)
                    rejected = any(c.get("rejected") for c in result.get("candidates", {}).values())
                    flag = " [LEAKED]" if rejected else ""
                    print(f"  [{i}/{len(work_items)}] {result['id']} (score={score:.1f}{flag}): {q}")
                except Exception as e:
                    errors += 1
                    print(f"  [{i}/{len(work_items)}] ERROR ({ctx['task_id']}/{seed['angle']}): {e}")

                if i % 5 == 0 or i == len(work_items):
                    elapsed = time.time() - start
                    print(f"    ({elapsed:.0f}s elapsed, {errors} errors)", file=sys.stderr)

        # Write full results (for human review)
        review_path = os.path.join(approach_dir, "review.json")
        review = {
            "summary": {
                "total_seeds": len(work_items),
                "total_results": len(all_results),
                "errors": errors,
                "models": list(SYNTH_MODELS.keys()),
                "mean_judge_score": round(
                    sum(r.get("mean_judge_score", 0) for r in all_results) / max(len(all_results), 1), 2
                ),
            },
            "results": {r["id"]: r for r in all_results},
        }
        with open(review_path, "w") as f:
            json.dump(review, f, indent=2)
        print(f"\n  Review file: {review_path}")

        # Write Harbor-compatible search_tasks.json
        search_tasks = []
        for r in all_results:
            if not r.get("question") or r["question"].startswith("ERROR:"):
                continue
            if any(
                c.get("rejected")
                for c in r.get("candidates", {}).values()
                if c.get("question") == r["question"]
            ):
                continue
            search_tasks.append(result_to_search_task(r))

        tasks_path = os.path.join(approach_dir, "search_tasks.json")
        with open(tasks_path, "w") as f:
            json.dump(search_tasks, f, indent=2)
        print(f"  Search tasks: {tasks_path} ({len(search_tasks)} tasks)")

        scores = [r.get("mean_judge_score", 0) for r in all_results if r.get("question")]
        print(f"\n  Summary for Approach {approach}:")
        print(f"    Seeds: {len(work_items)}, Results: {len(all_results)}, Errors: {errors}")
        print(f"    Usable tasks: {len(search_tasks)}")
        if scores:
            print(
                f"    Judge scores: mean={sum(scores) / len(scores):.2f}, "
                f"min={min(scores):.2f}, max={max(scores):.2f}"
            )
