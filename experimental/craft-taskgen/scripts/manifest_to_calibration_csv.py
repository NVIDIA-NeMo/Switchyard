# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert rerun-accepts-v2 cohort (per-repo JSON + MANIFEST.csv) into the
flat CSV shape that ``scripts/calibrate-alignment.py`` consumes.

Cohort layout on craftbench02 (and assumed by default flags here):

    /data/projects/craft-taskgen/candidates/rerun-accepts-v2/
        MANIFEST.csv               — one row per PR, includes provenance flags
                                     (in_v1a, in_mother_run_41, accepted_in_runs, ...)
        <repo>.json                — miner output schema
                                     {repo, github_repo, after, n_candidates,
                                      candidates: [{sha, base_sha, subject, ...}]}
        ...

Output: a single flat CSV with the columns ``calibrate-alignment.py`` needs
plus the manifest's provenance flags so downstream filtering (e.g. "tasks
that alignment-rejected in run-1") is one ``--filter accepted_in_runs=...``
expression rather than a re-mining step.

Output columns:
    task_id, repo, commit_sha, base_sha, subject, pr_url, instruction_md,
    in_v1a, in_mother_run_41, accepted_in_runs

(``instruction_md`` is intentionally empty — the cohort is a candidate-PR
list, not a historical instruction set. ``calibrate-alignment.py --mode=full``
generates fresh instructions; ``--mode=alignment-only`` would skip these
rows since instruction_md is empty, which is the right behavior.)

Usage (from craftbench02):

    cd /data/projects/craft-taskgen
    uv run python scripts/manifest_to_calibration_csv.py \\
        --candidates-dir /data/projects/craft-taskgen/candidates/rerun-accepts-v2 \\
        --output /data/projects/craft-taskgen/candidates/rerun-accepts-v2/calibration_input.csv

The ``task_id`` is generated as ``{REPO_PREFIX}{sha[:6]}`` matching
``craft_taskgen.steps._generate_task_id`` so per-row results join cleanly
against any state.json artifacts that already use those IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def _repo_prefix(repo: str) -> str:
    """Match `_generate_task_id`'s repo prefix: first letter of each
    word in the repo basename, uppercased, max 4 chars.

    Example: "tunnelvision/tunix" → "T2"; "TheoEpsteinNV/SteptronOss" → "SO".
    """
    # Take the basename (after the last "/") if path-like
    name = repo.split("/")[-1] if "/" in repo else repo
    # Split on hyphens, underscores, or camelCase-uppercase boundaries
    parts = re.split(r"[-_]+|(?=[A-Z])", name)
    parts = [p for p in parts if p]
    if not parts:
        return name[:4].upper()
    initials = "".join(p[0].upper() for p in parts if p[0].isalpha())[:4]
    return initials or name[:4].upper()


def _generate_task_id_local(repo: str, commit_sha: str) -> str:
    """Mirror of ``craft_taskgen.steps._generate_task_id`` for offline use.

    Avoids importing from craft_taskgen so this script runs even when the
    repo's venv isn't set up (e.g., on a fresh craftbench02 checkout).
    """
    prefix = _repo_prefix(repo)
    return f"{prefix}{commit_sha[:6]}"


def _resolve_pr_url(github_repo: str, sha: str) -> str:
    """Best-effort PR URL: github.com/{github_repo}/commit/{sha}.

    Miner JSON doesn't always carry the PR number; the commit URL is
    a stable proxy and is what the bigtest.csv format historically used.
    """
    return f"https://github.com/{github_repo}/commit/{sha}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidates-dir",
        default="/data/projects/craft-taskgen/candidates/rerun-accepts-v2",
        help="Directory containing per-repo JSON files + MANIFEST.csv.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Path to MANIFEST.csv. Defaults to <candidates-dir>/MANIFEST.csv.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Defaults to <candidates-dir>/calibration_input.csv.",
    )
    args = parser.parse_args()

    cohort_dir = Path(args.candidates_dir)
    if not cohort_dir.is_dir():
        print(f"ERROR: candidates dir not found: {cohort_dir}", file=sys.stderr)
        return 1

    manifest_path = Path(args.manifest) if args.manifest else cohort_dir / "MANIFEST.csv"
    output_path = Path(args.output) if args.output else cohort_dir / "calibration_input.csv"

    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    # Read manifest into a {(repo, commit_sha): row} index. The manifest's
    # column is "commit_sha"; per-repo JSONs use "sha" — handle both here.
    manifest_rows: dict[tuple[str, str], dict[str, str]] = {}
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha = row.get("commit_sha") or row.get("sha", "")
            key = (row.get("repo", ""), sha)
            manifest_rows[key] = row
    print(f"Loaded {len(manifest_rows)} rows from manifest", file=sys.stderr)

    # Walk per-repo JSONs and emit one row per candidate, joined with manifest
    out_rows: list[dict[str, str]] = []
    json_files = sorted(p for p in cohort_dir.iterdir() if p.suffix == ".json" and p.name != "MANIFEST.json")
    print(f"Reading {len(json_files)} per-repo JSON files", file=sys.stderr)

    n_unmatched = 0
    for jf in json_files:
        with jf.open() as f:
            doc = json.load(f)
        repo = doc.get("repo", "")
        github_repo = doc.get("github_repo", "")
        for cand in doc.get("candidates", []):
            sha = cand.get("sha", "")
            base_sha = cand.get("base_sha", "")
            subject = cand.get("subject", "")
            if not (repo and sha):
                continue
            manifest = manifest_rows.get((repo, sha), {})
            if not manifest:
                n_unmatched += 1
            tid = _generate_task_id_local(repo, sha)
            out_rows.append(
                {
                    "task_id": tid,
                    "repo": repo,
                    "commit_sha": sha,
                    "base_sha": base_sha,
                    "subject": subject,
                    "pr_url": _resolve_pr_url(github_repo or repo, sha) if github_repo else "",
                    "instruction_md": "",
                    # Provenance flags from manifest — keep raw strings.
                    # The actual column name in the rerun-accepts-v2 manifest
                    # is "runs_accepted" (semicolon-joined run timestamps),
                    # paired with "n_runs_accepted" (count, 0..3).
                    "in_v1a": manifest.get("in_v1a", ""),
                    "in_mother_run_41": manifest.get("in_mother_run_41", ""),
                    "accepted_in_runs": manifest.get("runs_accepted") or manifest.get("accepted_in_runs", ""),
                    "n_runs_accepted": manifest.get("n_runs_accepted", ""),
                }
            )

    if n_unmatched:
        print(f"WARNING: {n_unmatched} candidates had no matching manifest row", file=sys.stderr)

    fieldnames = [
        "task_id",
        "repo",
        "commit_sha",
        "base_sha",
        "subject",
        "pr_url",
        "instruction_md",
        "in_v1a",
        "in_mother_run_41",
        "accepted_in_runs",
        "n_runs_accepted",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
