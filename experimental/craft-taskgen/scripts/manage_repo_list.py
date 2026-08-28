#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add or remove repos from a craft-repos-style CSV, then dedup and swebench-check.

Reads from the latest references/repo_list_v{N}.csv and writes to v{N+1}.csv.

Usage — add:
    # Single GitHub URL
    python3 scripts/manage_repo_list.py add https://github.com/pallets/jinja

    # Text file of URLs (one per line)
    python3 scripts/manage_repo_list.py add /tmp/new_repos.txt

    # Pre-fetched CSV (skips fetch step)
    python3 scripts/manage_repo_list.py add /tmp/new_repos.csv

    # Override source list
    python3 scripts/manage_repo_list.py --list references/repo_list_v2.csv add https://github.com/pallets/jinja

Usage — remove:
    # By short_name or github_repo slug
    python3 scripts/manage_repo_list.py remove jinja
    python3 scripts/manage_repo_list.py remove pallets/jinja

    # Text file of short_names / slugs (one per line)
    python3 scripts/manage_repo_list.py remove /tmp/remove.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REFS_DIR = Path("references")
DEFAULT_EXCLUDED = Path("references/excluded-repos.csv")
FIELDS = ["short_name", "github_repo", "github_url", "stars", "license", "domain", "description"]

DOMAIN_RULES = [
    (
        "AI/ML",
        [
            "llm",
            "gpt",
            "neural",
            "nlp",
            "embedding",
            "rag",
            "agent",
            "voice",
            "speech",
            "asr",
            "tts",
            "machine learning",
            "machine-learning",
            "scikit",
            "sklearn",
            "deep learning",
            "deep-learning",
            "pytorch",
            "tensorflow",
            "keras",
            "xgboost",
            "classifier",
            "regression",
            "clustering",
        ],
    ),
    (
        "Web/API",
        [
            "web",
            "http",
            "api",
            "server",
            "fastapi",
            "django",
            "flask",
            "websocket",
            "rest",
            "graphql",
            "socket",
            "proxy",
            "tunnel",
            "vpn",
            "wsgi",
            "asgi",
            "template",
        ],
    ),
    ("CLI/TUI", ["cli", "terminal", "tui", "command-line", "shell", "prompt", "console", "repl", "curses"]),
    (
        "Data/DB",
        [
            "database",
            "sql",
            "data",
            "pandas",
            "csv",
            "json",
            "parser",
            "parse",
            "format",
            "markdown",
            "text",
            "pdf",
            "extract",
            "index",
            "search",
            "protobuf",
        ],
    ),
    (
        "Infra/System",
        [
            "docker",
            "deploy",
            "cloud",
            "infra",
            "system",
            "process",
            "thread",
            "async",
            "queue",
            "task",
            "worker",
            "archive",
            "compress",
            "audio",
            "video",
            "stream",
            "codec",
        ],
    ),
]


def _find_latest_list() -> Path:
    versioned = []
    for p in REFS_DIR.glob("repo_list_v*.csv"):
        m = re.match(r"repo_list_v(\d+)\.csv", p.name)
        if m:
            versioned.append((int(m.group(1)), p))
    if not versioned:
        sys.exit(f"No repo_list_v*.csv found in {REFS_DIR}/")
    return max(versioned)[1]


def _next_version(current: Path) -> Path:
    m = re.match(r"(repo_list_v)(\d+)(\.csv)", current.name)
    if not m:
        sys.exit(f"Cannot determine next version from {current.name}")
    return REFS_DIR / f"{m.group(1)}{int(m.group(2)) + 1}{m.group(3)}"


def classify_domain(short_name: str, description: str, topics: list[str]) -> str:
    text = (short_name + " " + (description or "") + " " + " ".join(topics or [])).lower()
    for domain, keywords in DOMAIN_RULES:
        if any(kw in text for kw in keywords):
            return domain
    return "Other"


def _parse_github_url(url: str) -> str:
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not m:
        sys.exit(f"ERROR: not a valid bare GitHub repo URL: {url}\n  Expected: https://github.com/owner/repo")
    return f"{m.group(1)}/{m.group(2)}"


def fetch_repo(github_repo: str) -> dict | None:
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{github_repo}",
            "--jq",
            '[.name, (.description // ""), .stargazers_count,'
            ' (.license.spdx_id // "None"), .html_url, (.topics // [])]',
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR fetching {github_repo}: {result.stderr.strip()}", file=sys.stderr)
        return None
    try:
        name, description, stars, license_id, url, topics = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ERROR fetching {github_repo}: unexpected API response: {e}", file=sys.stderr)
        return None
    domain = classify_domain(name, description, topics)
    return {
        "short_name": name,
        "github_repo": github_repo,
        "github_url": url,
        "stars": str(stars),
        "license": license_id,
        "domain": domain,
        "description": description,
    }


def _dedup_and_exclude(rows: list[dict], excluded_path: Path) -> list[dict]:
    excluded: set[str] = set()
    if excluded_path.exists():
        with excluded_path.open() as f:
            excluded = {r["github_repo"] for r in csv.DictReader(f)}

    seen: set[str] = set()
    kept, n_dup, n_excl = [], 0, 0
    for r in rows:
        key = r["github_repo"]
        if key in seen:
            print(f"  [dedup] removed duplicate: {r['short_name']}")
            n_dup += 1
        elif key in excluded:
            print(f"  [excluded] removed: {r['short_name']}")
            n_excl += 1
        else:
            seen.add(key)
            kept.append(r)
    if n_dup or n_excl:
        print(f"  Removed {n_dup} duplicate(s), {n_excl} excluded repo(s)")
    return kept


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def cmd_add(args: argparse.Namespace) -> None:
    source: Path = args.list or _find_latest_list()
    out: Path = args.out or _next_version(source)
    with source.open() as f:
        existing = list(csv.DictReader(f)) if source.exists() else []

    inp = args.input
    new_rows: list[dict] = []

    failed: list[str] = []
    if inp.startswith("http"):
        github_repo = _parse_github_url(inp)
        print(f"Fetching {github_repo}...")
        info = fetch_repo(github_repo)
        if info is None:
            sys.exit(f"ERROR: fetch failed for {github_repo}. Check the URL and your GitHub CLI auth.")
        new_rows = [info]
    else:
        p = Path(inp)
        if not p.exists():
            sys.exit(f"Error: {p} not found")
        content = p.read_text()
        first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
        if first_line.startswith("http"):
            urls = [line.strip() for line in content.splitlines() if line.strip().startswith("http")]
            for url in urls:
                github_repo = _parse_github_url(url)
                print(f"Fetching {github_repo}...")
                info = fetch_repo(github_repo)
                if info:
                    new_rows.append(info)
                else:
                    failed.append(url)
        else:
            with p.open() as f:
                new_rows = list(csv.DictReader(f))
            print(f"Loaded {len(new_rows)} repos from {p}")

    if failed:
        print(f"\nERROR: {len(failed)} fetch(es) failed:", file=sys.stderr)
        for u in failed:
            print(f"  {u}", file=sys.stderr)
        sys.exit(1)

    if not new_rows:
        sys.exit("Nothing to add.")

    cleaned = _dedup_and_exclude(existing + new_rows, args.exclude)
    existing_repos = {e["github_repo"] for e in existing}
    if len(cleaned) == len(existing) and not any(r["github_repo"] not in existing_repos for r in new_rows):
        print("Nothing new to add — all repos were duplicates or excluded. No file written.")
        return
    _write(out, cleaned)
    net_added = len(cleaned) - len(existing)
    print(f"\nSource:  {source}  ({len(existing)} repos)")
    print(f"Output:  {out}  ({len(cleaned)} repos)")
    print(f"Fetched: {len(new_rows)}  |  Net added: {net_added}")


def cmd_remove(args: argparse.Namespace) -> None:
    source: Path = args.list or _find_latest_list()
    out: Path = args.out or _next_version(source)
    if not source.exists():
        sys.exit(f"Error: {source} not found")

    with source.open() as f:
        rows = list(csv.DictReader(f))

    inp = args.input
    p = Path(inp)
    if p.exists():
        remove_keys = {line.strip() for line in p.read_text().splitlines() if line.strip()}
    else:
        remove_keys = {inp.strip()}

    kept, removed = [], []
    for r in rows:
        if r["short_name"] in remove_keys or r["github_repo"] in remove_keys:
            removed.append(r)
        else:
            kept.append(r)

    if not removed:
        print(f"No matching repos found for: {remove_keys}")
        return

    _write(out, kept)
    print(f"\nSource:  {source}  ({len(rows)} repos)")
    print(f"Output:  {out}  ({len(kept)} repos)")
    print(f"Removed: {[r['short_name'] for r in removed]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add or remove repos from a craft-repos-style CSV.")
    parser.add_argument(
        "--list", type=Path, default=None, help="Source CSV (default: latest references/repo_list_v*.csv)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output CSV (default: references/repo_list_v{N+1}.csv)"
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=DEFAULT_EXCLUDED,
        help="Excluded repos CSV (default: references/excluded-repos.csv)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add repos (URL, URL list file, or pre-fetched CSV)")
    p_add.add_argument("input", help="GitHub URL, text file of URLs, or pre-fetched CSV")

    p_rm = sub.add_parser("remove", help="Remove repos by short_name, github_repo slug, or file of those")
    p_rm.add_argument("input", help="short_name, github_repo slug, or text file of those")

    args = parser.parse_args()
    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "remove":
        cmd_remove(args)


if __name__ == "__main__":
    main()
