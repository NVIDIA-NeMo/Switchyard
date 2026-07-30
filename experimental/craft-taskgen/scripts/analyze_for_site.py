#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate site/data/*.json for the DeepSWE-style site analyses.

Produces three output files:

- site/data/efficiency_scatter.json — three scatters (output tokens
  vs Resolved%, wall-clock vs Resolved%, est. cost vs Resolved%) over
  the K=5 top-line leaderboard agents. Mean per-trial tokens / wall-clock
  come from site/data/efficiency.json; pass-rates come from
  site/data/leaderboard.json; cost is computed via harbor-lab's pricing
  module.
- site/data/task_corpus.json — per-task instruction-length, gold-patch
  lines-added, files-touched distributions + language pie. Read from the
  CRAFT-bench task directories under ~/projects/craft-bench/harbor-tasks/.
- site/data/repo_diversity.json — per-repo GitHub stars + tree-file
  count for the diversity scatter; one dot per distinct source repo.

Run as: uv run python scripts/analyze_for_site.py
Or per-section: --only efficiency | corpus | repos
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_DATA = REPO_ROOT / "site" / "data"
# Default location of the craft-bench harbor-tasks dir; overridable via
# --tasks-dir so the script runs on any machine, not just the author's.
DEFAULT_TASKS_DIR = Path.home() / "projects" / "craft-bench" / "harbor-tasks" / "craft-taskgen-v2b"

PRICING_NOTE = (
    "Per-trial cost computed via harbor-lab's pricing module — the same rates "
    "`harbor-lab routed-cost` reports. Nemotron-3 Ultra (EA) has no public rate "
    "and is omitted from the cost scatter. Gateway-billed cost is not exposed "
    "by the inference-api endpoint."
)


def _load_pricing_table():
    """Import harbor-lab's pricing module on demand. Returns (estimate_cost, table)
    or (None, None) if harbor-lab isn't installed. Done lazily so that
    `--only corpus` / `--only repos` runs don't require harbor-lab."""
    candidates = [
        Path.home() / "projects" / "harbor-lab" / "src",
        Path("/data/projects/harbor-lab/src"),
    ]
    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from harbor_lab.pricing import _load_pricing, estimate_cost  # type: ignore[import]
    except ImportError as exc:
        print(
            f"warn: harbor_lab.pricing unavailable: {exc}; cost will be None",
            file=sys.stderr,
        )
        return None, None
    return estimate_cost, _load_pricing()


# ----------------------------------------------------------------------------
# A1+A2+A3 — efficiency scatters
# ----------------------------------------------------------------------------


def _eff_metric_mean(eff_row: dict, key: str) -> int | float | None:
    """Read a `{mean, sd}` block from an efficiency.json row. Returns None
    when the block is missing OR present-but-null (e.g. `{mean: null}`). Both
    cases used to crash via `int(None * 1000)`."""
    block = eff_row.get(key)
    if not isinstance(block, dict):
        return None
    m = block.get("mean")
    return m if m is not None else None


# Map leaderboard model labels to harbor-lab pricing keys we can resolve.
# Anything not in this map gets cost=None (which the scatter renders as "—").
_LEADERBOARD_MODEL_WIRE: dict[str, str] = {
    "GPT-5.5": "openai/openai/gpt-5.5",  # litellm: gpt-5.5
    "Opus 4.7": "azure/anthropic/claude-opus-4-7",  # litellm: claude-opus-4-7
    "GLM-5.1": "zai-org/glm-5.1",  # harbor-lab override
    "Haiku 4.5": "azure/anthropic/claude-haiku-4-5",
    "Qwen-3.6-35B-A3B": "qwen/qwen3.6-35b-a3b",  # harbor-lab override
    "Nemotron-3 Ultra (EA)": "nvidia/nemotron-3-ultra-preview",  # no pricing
}


def build_efficiency() -> dict:
    """Per-row efficiency vs Resolved% from the K=5 top-line tables.

    Joins site/data/leaderboard.json (Resolved%) with site/data/efficiency.json
    (mean tokens, mean wall-clock). Per-trial cost is the K=5 mean tokens
    × pricing rates from harbor-lab. One dot per leaderboard row that also
    appears in efficiency.json (rows missing efficiency data are skipped).
    """
    lb = json.loads((SITE_DATA / "leaderboard.json").read_text())
    eff = json.loads((SITE_DATA / "efficiency.json").read_text())

    # Map (agent, model_starts_with) to efficiency rows. efficiency.json
    # sometimes carries a trailing "xhigh" / reasoning hint in the model
    # field that leaderboard.json doesn't, so match on prefix.
    eff_by_pair: dict[tuple[str, str], dict] = {}
    for r in eff.get("rows", []):
        eff_by_pair[(r["agent"], r["model"].split()[0])] = r

    estimate_cost, pricing_table = _load_pricing_table()

    def _est_cost(wire: str | None, n_in: int, n_cache: int, n_out: int) -> float | None:
        if not wire or estimate_cost is None or pricing_table is None:
            return None
        try:
            est = estimate_cost(wire, n_in, n_cache, n_out, pricing_table)
        except Exception as exc:  # noqa: BLE001 — pricing module may raise on malformed entries
            print(f"warn: estimate_cost({wire!r}) raised: {exc}", file=sys.stderr)
            return None
        return est["total_cost"] if est else None

    points = []
    for row in lb.get("rows", []):
        agent_label = row.get("agent")
        model_label = row.get("model")
        if agent_label is None or model_label is None:
            continue
        eff_row = eff_by_pair.get((agent_label, model_label.split()[0]))
        if not eff_row:
            continue  # leaderboard row without efficiency table data; skip

        # efficiency.json stores input/cached/output token means in THOUSANDS;
        # multiply by 1000 to get raw token counts for the pricing call. The
        # _eff_metric_mean helper returns None for missing-or-null blocks so
        # we don't crash on `int(None * 1000)`.
        in_k = _eff_metric_mean(eff_row, "input_k")
        cache_k = _eff_metric_mean(eff_row, "cached_k")
        out_k = _eff_metric_mean(eff_row, "output_k")
        n_in = int(in_k * 1000) if in_k is not None else 0
        n_cache = int(cache_k * 1000) if cache_k is not None else 0
        n_out = int(out_k * 1000) if out_k is not None else 0
        wire = _LEADERBOARD_MODEL_WIRE.get(model_label)
        cost = _est_cost(wire, n_in, n_cache, n_out)

        points.append(
            {
                "agent": agent_label,
                "model": model_label,
                "label": f"{agent_label} · {model_label}",
                "resolved": row.get("resolved"),
                "mean_output_tokens": n_out if out_k is not None else None,
                "mean_duration_s": _eff_metric_mean(eff_row, "wall_s"),
                "mean_cost_usd": cost,
                "n": row.get("n"),
            }
        )

    return {
        "eyebrow": "K=5 leaderboard · 3 dimensions",
        "thesis": "Accuracy vs cost",
        "deck": (
            "Mean per-trial output tokens, wall-clock, and estimated cost vs "
            "Resolved% for each top-line agent on CRAFT-bench. One dot per "
            "agent configuration in the leaderboard."
        ),
        "pricing_note": PRICING_NOTE,
        "points": points,
    }


# ----------------------------------------------------------------------------
# A4 + A6 — task corpus stats
# ----------------------------------------------------------------------------


def _gold_patch_stats(patch_path: Path) -> tuple[int, int]:
    """Return (lines_added, distinct_files_touched) from a unified-diff file."""
    try:
        text = patch_path.read_text(errors="replace")
    except OSError:
        return (0, 0)
    files: set[str] = set()
    added = 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            # `diff --git a/<path> b/<path>` -> take a/<path>
            m = re.match(r"diff --git a/(\S+) b/", line)
            if m:
                files.add(m.group(1))
        elif line.startswith("+++") or line.startswith("---"):
            continue  # diff header noise
        elif line.startswith("+") and not line.startswith("++"):
            added += 1
    return (added, len(files))


def _dockerfile_repo(env_dockerfile: Path) -> str | None:
    """Pull the github.com/{owner}/{repo} from the Dockerfile's clone command."""
    try:
        text = env_dockerfile.read_text(errors="replace")
    except OSError:
        return None
    m = re.search(r"github\.com/([\w.-]+/[\w.-]+?)\.git", text)
    if m:
        return m.group(1)
    m = re.search(r"github\.com/([\w.-]+/[\w.-]+?)(?=\s|\")", text)
    return m.group(1) if m else None


def build_task_corpus(tasks_dir: Path) -> dict:
    """Per-task instruction-length + gold-patch stats; aggregated to a
    distribution plus a language pie. Language is derived in
    build_repo_diversity (from gh api), so the per-task language slot here
    is filled with a placeholder and the pie is built from those repos."""
    per_task: list[dict] = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if not (task_dir / "task.toml").is_file():
            continue
        instruction = task_dir / "instruction.md"
        gold_patch = task_dir / "solution" / "changes.patch"
        if not instruction.is_file() or not gold_patch.is_file():
            continue
        prompt_chars = len(instruction.read_text(errors="replace"))
        lines_added, files_touched = _gold_patch_stats(gold_patch)
        per_task.append(
            {
                "task": task_dir.name,
                "prompt_chars": prompt_chars,
                "gold_lines_added": lines_added,
                "gold_files_touched": files_touched,
            }
        )

    def quartiles(values: list[int]) -> dict[str, float]:
        if not values:
            return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0, "mean": 0}
        s = sorted(values)
        n = len(s)
        return {
            "min": s[0],
            "q1": s[n // 4],
            "median": s[n // 2],
            "q3": s[(3 * n) // 4],
            "max": s[-1],
            "mean": round(sum(s) / n, 1),
        }

    chars = [t["prompt_chars"] for t in per_task]
    lines = [t["gold_lines_added"] for t in per_task]
    files = [t["gold_files_touched"] for t in per_task]
    n = len(per_task)

    # Cross-benchmark comparison. CRAFT row reports the median (p50) since
    # the corpus has a long right tail (one task's gold patch contains 360
    # auto-generated Sphinx HTML files); median is outlier-robust without
    # carving any task out of the corpus.
    def craft_median(xs: list[int]) -> int:
        return sorted(xs)[len(xs) // 2] if xs else 0

    benchmark_comparison = [
        {
            "benchmark": "SWE-bench Verified",
            "prompt_chars": 1700,
            "ref_lines_added": 10,
            "files_modified": 1,
        },
        {
            "benchmark": "SWE-bench Pro",
            "prompt_chars": 4614,
            "ref_lines_added": 120,
            "files_modified": 5,
        },
        {
            "benchmark": "DeepSWE",
            "prompt_chars": 2158,
            "ref_lines_added": 668,
            "files_modified": 7,
        },
        {
            "benchmark": "CRAFT-bench (p50)",
            "prompt_chars": craft_median(chars),
            "ref_lines_added": craft_median(lines),
            "files_modified": craft_median(files),
            "self": True,
        },
    ]

    return {
        "eyebrow": f"Corpus · n={n} tasks",
        "thesis": "Inside CRAFT-bench",
        "deck": (
            "Distribution of per-task instruction length (chars), gold-patch "
            "lines added, and gold-patch files touched across the CRAFT-bench "
            "task set. Smaller is shorter / less code; larger is more."
        ),
        "n_tasks": n,
        "prompt_chars": quartiles(chars),
        "gold_lines_added": quartiles(lines),
        "gold_files_touched": quartiles(files),
        # The CRAFT pipeline currently builds Python tasks only (see
        # src/craft_taskgen/adapters/_docker.py — every Dockerfile starts
        # from a python:{ver}-slim base). Stated explicitly here, rather
        # than via a "language-guess" helper that pretended to consider
        # the repo but always returned "Python".
        "language_counts": [{"language": "Python", "n": n}],
        "benchmark_comparison": benchmark_comparison,
        # The renderer treats this as HTML so it can carry a <a href> citation.
        "benchmark_comparison_note_html": (
            "External numbers from the "
            '<a href="https://deepswe.datacurve.ai/blog" target="_blank" '
            'rel="noopener">DeepSWE blog</a> '
            "(DeepSWE, SWE-bench Verified, SWE-bench Pro means). "
            "CRAFT-bench row is the p50 (median) across all 92 tasks — "
            "outlier-robust, so no task is carved out of the corpus."
        ),
    }


# ----------------------------------------------------------------------------
# A5 — repo diversity
# ----------------------------------------------------------------------------


def _gh_repo_meta(repo: str) -> dict | None:
    """Run `gh api repos/{repo}` and pull stars + size_kb + language. Cached
    in memory only — caller dedups."""
    try:
        out = subprocess.run(
            ["gh", "api", f"repos/{repo}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        d = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    return {
        "repo": repo,
        "stars": d.get("stargazers_count"),
        "size_kb": d.get("size"),  # repo size on disk, kilobytes
        "language": d.get("language"),
    }


def build_repo_diversity(tasks_dir: Path) -> dict:
    """One dot per distinct source repo: GitHub stars vs repo size; dot size
    = task count; color = language."""
    repo_to_tasks: dict[str, list[str]] = defaultdict(list)
    for task_dir in sorted(tasks_dir.iterdir()):
        if not (task_dir / "task.toml").is_file():
            continue
        repo = _dockerfile_repo(task_dir / "environment" / "Dockerfile")
        if not repo:
            continue
        repo_to_tasks[repo].append(task_dir.name)

    print(f"  fetching GitHub metadata for {len(repo_to_tasks)} repos…", file=sys.stderr)
    points: list[dict] = []
    for repo, tasks in sorted(repo_to_tasks.items()):
        meta = _gh_repo_meta(repo)
        if not meta:
            print(f"  WARN: could not fetch metadata for {repo!r}", file=sys.stderr)
            continue
        points.append(
            {
                "repo": repo,
                "stars": meta["stars"],
                "size_kb": meta["size_kb"],
                "language": meta["language"] or "Unknown",
                "task_count": len(tasks),
            }
        )

    return {
        "eyebrow": f"Repo diversity · n={len(points)} source repos",
        "thesis": "Repo diversity",
        "deck": (
            "One dot per source repo. Stars (log) on x; repo size on disk "
            "(log, kB) on y; dot size scales with how many tasks came from "
            "that repo."
        ),
        "points": points,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


_BUILDER_NAMES = ("efficiency", "corpus", "repos")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        choices=_BUILDER_NAMES,
        action="append",
        help="Only build the named section(s). Default: all.",
    )
    p.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help=(
            "Path to the craft-bench harbor-tasks directory (used by "
            "corpus + repos sections). Default: %(default)s."
        ),
    )
    args = p.parse_args()
    SITE_DATA.mkdir(parents=True, exist_ok=True)

    # Bind builders here so --tasks-dir propagates without globals.
    builders = {
        "efficiency": ("efficiency_scatter.json", build_efficiency),
        "corpus": ("task_corpus.json", lambda: build_task_corpus(args.tasks_dir)),
        "repos": ("repo_diversity.json", lambda: build_repo_diversity(args.tasks_dir)),
    }

    todo = args.only or list(builders)
    for name in todo:
        outfile, builder = builders[name]
        if name in {"corpus", "repos"} and not args.tasks_dir.is_dir():
            print(
                f"warn: [{name}] tasks-dir does not exist: {args.tasks_dir} — skipping",
                file=sys.stderr,
            )
            continue
        print(f"[{name}] building…", file=sys.stderr)
        data = builder()
        target = SITE_DATA / outfile
        target.write_text(json.dumps(data, indent=2) + "\n")
        print(f"[{name}] -> {target.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
