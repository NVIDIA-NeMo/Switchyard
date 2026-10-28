#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank GitHub PR reviewers using your authenticated GitHub CLI (gh).

Usage: python scripts/github_reviewers.py [OWNER/REPO]
Defaults to NVIDIA-NeMo/Switchyard. Scans all PR history.
Each comment earns one point; each approval earns three. Counts PR discussion
comments, inline comments, and nonempty submitted review bodies. An approval
with a body earns four points. Excludes the PR author's own activity.
Only users with at least one recorded APPROVED review appear, up to 20.
GitHub's current review records determine approvals; dismissed approvals
are excluded. Repeated approvals count separately.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain


@dataclass
class Activity:
    comments: int = 0
    approvals: int = 0

    @property
    def score(self) -> int:
        return self.comments + 3 * self.approvals


def gh_json(*args: str):
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    # gh --paginate emits consecutive JSON arrays on older CLI versions.
    decoder = json.JSONDecoder()
    remaining = result.stdout.lstrip()
    pages = []
    while remaining:
        page, end = decoder.raw_decode(remaining)
        pages.append(page)
        remaining = remaining[end:].lstrip()
    return pages


def records(endpoint: str):
    pages = gh_json("api", "--paginate", f"{endpoint}?per_page=100")
    for page in pages:
        yield from page


def count_activity(activity, author, reviews, comments):
    for review in reviews:
        login = (review.get("user") or {}).get("login")
        if not login or login == author or review["state"] == "PENDING":
            continue
        if review["state"] == "APPROVED":
            activity[login].approvals += 1
        if (review.get("body") or "").strip():
            activity[login].comments += 1
    for comment in comments:
        login = (comment.get("user") or {}).get("login")
        if login and login != author and (comment.get("body") or "").strip():
            activity[login].comments += 1


def ranked(activity):
    return sorted(
        ((login, counts) for login, counts in activity.items() if counts.approvals),
        key=lambda item: (-item[1].score, -item[1].approvals, item[0].lower()),
    )[:20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo",
        nargs="?",
        default="NVIDIA-NeMo/Switchyard",
        help="GitHub repository: OWNER/REPO (default: NVIDIA-NeMo/Switchyard)",
    )
    args = parser.parse_args()
    repo = args.repo
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        parser.error("repository must be OWNER/REPO")

    activity = defaultdict(Activity)
    # The all-state list includes open, closed, and merged PRs.
    pages = gh_json(
        "api",
        "--paginate",
        f"repos/{repo}/pulls?state=all&per_page=100",
    )
    for page in pages:
        for pr in page:
            number = pr["number"]
            print(f"Scanning {repo} PR #{number}...", file=sys.stderr)
            base = f"repos/{repo}"
            reviews = records(f"{base}/pulls/{number}/reviews")
            inline = records(f"{base}/pulls/{number}/comments")
            discussion = records(f"{base}/issues/{number}/comments")
            count_activity(
                activity,
                (pr.get("user") or {}).get("login"),
                reviews,
                chain(inline, discussion),
            )

    print(f"{repo} — all PR history; comment = 1 point, approval = 3 points")
    print(f"{'Reviewer':<40} {'Comments':>8} {'Approvals':>9} {'Score':>8}")
    for login, counts in ranked(activity):
        print(f"{login:<40} {counts.comments:>8} {counts.approvals:>9} {counts.score:>8}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        sys.exit("Install the GitHub CLI (gh), then run gh auth login.")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"GitHub CLI failed: {exc.stderr.strip()}")
