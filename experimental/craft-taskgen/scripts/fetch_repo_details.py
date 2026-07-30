#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch GitHub metadata for a list of repo URLs and output a craft-repos-style CSV.

Usage:
    # 1. Create a text file with one GitHub URL per line, e.g.:
    #      https://github.com/pallets/jinja
    #      https://github.com/psf/requests
    #
    # 2. Run:
    #      python3 scripts/fetch_repo_details.py /path/to/urls.txt --out /path/to/output.csv
    #      python3 scripts/fetch_repo_details.py https://github.com/pallets/jinja --out /path/to/output.csv
    #
    # Requires: gh CLI authenticated (gh auth login)
    # Output columns: short_name, github_repo, github_url, stars, license, domain, description
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# First-match wins — order matters
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
            "socks",
            "wsgi",
            "asgi",
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
        print(f"  ERROR {github_repo}: {result.stderr.strip()}", file=sys.stderr)
        return None
    try:
        name, description, stars, license_id, url, topics = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ERROR {github_repo}: unexpected API response: {e}", file=sys.stderr)
        return None
    return {
        "name": name,
        "description": description,
        "stars": stars,
        "license": license_id,
        "url": url,
        "topics": topics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GitHub metadata and output craft-repos-style CSV.")
    parser.add_argument("input", help="GitHub URL or file of URLs (one per line)")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output CSV (default: stdout for URL, <input>.csv for file)"
    )
    args = parser.parse_args()

    if args.input.startswith("http"):
        urls = [args.input.strip()]
        out_file: Path | None = args.out
    else:
        urls_file = Path(args.input)
        out_file = args.out or urls_file.with_suffix(".csv")
        with urls_file.open() as f:
            urls = [line.strip() for line in f.read().splitlines() if line.strip().startswith("http")]

    rows = []
    failed = []
    for url in urls:
        github_repo = _parse_github_url(url)
        print(f"Fetching {github_repo}...", file=sys.stderr)
        info = fetch_repo(github_repo)
        if not info:
            failed.append(url)
            continue

        domain = classify_domain(info["name"], info["description"], info["topics"])
        rows.append(
            {
                "short_name": info["name"],
                "github_repo": github_repo,
                "github_url": info["url"],
                "stars": info["stars"],
                "license": info["license"],
                "domain": domain,
                "description": info["description"],
            }
        )

    fieldnames = ["short_name", "github_repo", "github_url", "stars", "license", "domain", "description"]
    if out_file:
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)}/{len(urls)} repos to {out_file}", file=sys.stderr)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if failed:
        print(f"\nERROR: {len(failed)} fetch(es) failed:", file=sys.stderr)
        for u in failed:
            print(f"  {u}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
