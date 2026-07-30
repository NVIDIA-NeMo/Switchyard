# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for PR-reference importers."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

PR_URL_RE = re.compile(
    r"https?://github\.com/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/(?P<number>\d+)",
    re.IGNORECASE,
)
GITHUB_REPO_RE = re.compile(
    r"github\.com[:/](?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?(?:[/?#].*)?$",
    re.IGNORECASE,
)
OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
INSTANCE_ID_RE = re.compile(r"-(\d+)$")

PR_URL_KEYS = (
    "pull_request",
    "pull_request_url",
    "pr_url",
    "pr",
    "url",
    "html_url",
)
REPO_KEYS = ("repo", "github_repo", "repository", "project")
PR_NUMBER_KEYS = ("pr_number", "pull_number", "pull_request_number", "number")


def normalize_repo_slug(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if OWNER_REPO_RE.fullmatch(text):
        return text

    # Some datasets use owner__repo naming.
    if "/" not in text and "__" in text:
        maybe = text.replace("__", "/", 1)
        if OWNER_REPO_RE.fullmatch(maybe):
            return maybe

    match = GITHUB_REPO_RE.search(text)
    if match:
        return match.group("repo")
    return None


def parse_pr_url(raw: Any) -> tuple[str, int] | None:
    if raw is None:
        return None
    match = PR_URL_RE.search(str(raw))
    if not match:
        return None
    return match.group("repo"), int(match.group("number"))


def parse_pr_number(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        number = int(text)
        return number if number > 0 else None
    return None


def extract_pr_ref(record: dict[str, Any]) -> tuple[str, int] | None:
    """Extract (owner/repo, pr_number) from a dataset record."""
    for key in PR_URL_KEYS:
        parsed = parse_pr_url(record.get(key))
        if parsed:
            return parsed

    repo = None
    for key in REPO_KEYS:
        repo = normalize_repo_slug(record.get(key))
        if repo:
            break
    if not repo:
        return None

    for key in PR_NUMBER_KEYS:
        number = parse_pr_number(record.get(key))
        if number:
            return repo, number

    # Fallback for rows with instance_id like "owner__repo-12345".
    instance_id = record.get("instance_id")
    if instance_id is not None:
        match = INSTANCE_ID_RE.search(str(instance_id))
        if match:
            return repo, int(match.group(1))
    return None


def load_records(
    path: Path,
    *,
    allow_json_object_map: bool = False,
    record_id_key: str = "source_record_id",
) -> list[dict[str, Any]]:
    """Load records from jsonl/json/csv/tsv.

    When `allow_json_object_map=True`, a JSON object of `{id: {...record...}}`
    is expanded into rows and each row gets `{record_id_key: id}` if missing.
    """
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".jl"}:
        rows: list[dict[str, Any]] = []
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(item).__name__}")
                rows.append(item)
        return rows

    if suffix in {".csv", ".tsv"}:
        with open(path, newline="") as f:
            dialect = "excel-tab" if suffix == ".tsv" else "excel"
            reader = csv.DictReader(f, dialect=dialect)
            return [dict(row) for row in reader]

    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if all(isinstance(x, dict) for x in data):
                return list(data)
            raise ValueError(f"{path}: expected list[object], got mixed list")
        if isinstance(data, dict):
            for key in ("instances", "data", "rows", "records"):
                value = data.get(key)
                if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                    return list(value)
            if allow_json_object_map and all(isinstance(v, dict) for v in data.values()):
                rows = []
                for record_id, record in data.items():
                    row = dict(record)
                    row.setdefault(record_id_key, str(record_id))
                    rows.append(row)
                return rows
            return [data]
        raise ValueError(f"{path}: expected JSON object or array")

    raise ValueError(f"Unsupported input format for {path} (expected .jsonl/.json/.csv/.tsv)")


def fetch_merged_pr(github_repo: str, pr_number: int) -> dict[str, Any] | None:
    """Fetch PR metadata via gh API, returning analyzer input dict or None if unusable."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{github_repo}/pulls/{pr_number}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"gh api timed out for {github_repo}#{pr_number}")
    except OSError as e:
        raise RuntimeError(f"gh api failed to start for {github_repo}#{pr_number}: {e}")

    if result.returncode != 0:
        raise RuntimeError(f"gh api failed for {github_repo}#{pr_number}: {result.stderr[:200]}")

    try:
        pr = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"gh api returned non-JSON for {github_repo}#{pr_number}: {e}") from e

    merged_at = pr.get("merged_at")
    if not merged_at:
        return None

    sha = pr.get("merge_commit_sha")
    base_sha = (pr.get("base") or {}).get("sha")
    if not sha or not base_sha:
        return None

    return {
        "sha": sha,
        "base_sha": base_sha,
        "subject": pr.get("title", ""),
        "author": (pr.get("user") or {}).get("login", "unknown"),
        "date": merged_at,
        "pr_number": int(pr_number),
    }
