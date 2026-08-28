# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""check_integrity.py — scan agent trajectories for benchmark-integrity violations.

Focused on craft-bench / Harbor trial directories. Two transcript formats are
supported: ATIF `agent/trajectory.json` (codex, opencode, and any other ATIF
producer) and claude-code's JSONL `agent/claude-code.txt`.

Two flavors of signal:
  - "any web access" — WebSearch / WebFetch / curl-style command issued at all
  - "upstream fetch" — a URL or curl/wget/git-clone targeted at code-hosting
    domains likely to leak the reference solution
    (raw.githubusercontent.com, api.github.com, huggingface.co/<model>/...
    excluding /docs, gitlab*.com)

Usage:
    uv run python scripts/check_integrity.py <trial_root>... [--csv out.csv]

Trial roots are walked recursively. Each result row corresponds to one
trial; rows with no signal are omitted from the printed report (but always
included in the CSV when --csv is given).

For each task that has at least one upstream-fetch trial, the report
breaks the task's pass/fail counts into clean vs tainted subsets so
reviewers can decide whether the task itself remains scorable.

Inspired by harbor-datasets' `tools/check-integrity.py`, which uses a
JSON config of forbidden patterns. This script hard-codes craft-bench-style
heuristics (open-internet containers + upstream-source leak via curl); if the
catalog of forbidden domains expands, lift the regex constants into a config.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

URL_RE = re.compile(r"https?://[^\s'\")]+", re.IGNORECASE)
UPSTREAM_HOSTS = re.compile(
    r"(raw\.githubusercontent\.com|api\.github\.com|huggingface\.co/(?!docs)"
    r"|gitlab(?:-master)?\.com)",
    re.IGNORECASE,
)
# Identifiers that perform an outbound HTTP request. If a shell command body
# contains one of these AND an upstream URL, treat it as a fetch. We use this
# to filter out false positives where URLs appear as test fixtures, mock
# response data, or assertion-expected strings inside python heredocs.
FETCHER_RE = re.compile(
    r"\b(curl|wget|git\s+clone|urlopen|urllib\.request|"
    r"requests\.(?:get|post|put|head|delete|request)|"
    r"httpx\.(?:get|post|put|head|delete|request)|"
    r"aiohttp\.|fetch\()",
    re.IGNORECASE,
)


def _coerce_args(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"_raw": args}
    return args or {}


def _scan_atif(trajectory_path: Path) -> dict:
    """Walk an ATIF trajectory.json and surface web/upstream signals.

    Handles codex (`exec_command`/`cmd`) and opencode (`bash`/`command`) variants
    of the same schema.

    URLs only count when they appear inside an actual fetch site:
      - a WebFetch/url_fetch tool arg (Anthropic-style web tool), or
      - the body of a shell command (codex `cmd`, opencode `command`),
        regardless of whether the agent uses curl/wget/git-clone or a
        Python-level fetcher (`urllib.request.urlopen`, `requests.get`, etc.).

    URLs that show up in edit/read/write arg strings (e.g. an `oldString`
    for a docstring containing an example URL) are NOT flagged, since the
    agent is editing in-container code, not fetching anything.
    """
    out = {
        "ws_count": 0,
        "wf_count": 0,
        "upstream_urls": set(),
        "shell_fetch_cmds": [],
        "any_web": False,
    }
    try:
        d = json.loads(trajectory_path.read_text())
    except Exception:
        return out
    for step in d.get("steps", []):
        for tc in step.get("tool_calls", []) or []:
            fn = (tc.get("function_name") or "").lower()
            args = _coerce_args(tc.get("arguments"))
            if "web_search" in fn or fn == "websearch":
                out["ws_count"] += 1
                out["any_web"] = True
            if "web_fetch" in fn or fn in ("webfetch", "url_fetch"):
                out["wf_count"] += 1
                out["any_web"] = True
                if isinstance(args, dict):
                    url = args.get("url") or ""
                    if isinstance(url, str) and UPSTREAM_HOSTS.search(url):
                        out["upstream_urls"].add(url)
            # Shell fetch detection — codex args.cmd or opencode args.command.
            # The command must reference a fetcher (curl/wget/git-clone or a
            # Python HTTP lib like urllib.request/requests/httpx/aiohttp) AND
            # contain an upstream URL.
            cmd = None
            if isinstance(args, dict):
                cmd = args.get("cmd") or args.get("command")
            if cmd and FETCHER_RE.search(cmd):
                ext = [u for u in URL_RE.findall(cmd) if UPSTREAM_HOSTS.search(u)]
                if ext or "git clone" in cmd:
                    out["shell_fetch_cmds"].append(cmd[:200])
                    out["upstream_urls"].update(ext)
                    out["any_web"] = True
    return out


def _scan_claude_code(cc_path: Path) -> dict:
    """Walk a claude-code JSONL transcript and surface web/upstream signals."""
    out = {
        "ws_count": 0,
        "wf_count": 0,
        "upstream_urls": set(),
        "shell_fetch_cmds": [],
        "any_web": False,
    }
    try:
        f = cc_path.open()
    except Exception:
        return out
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        msg = obj.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            inp = block.get("input") or {}
            if name == "WebSearch":
                out["ws_count"] += 1
                out["any_web"] = True
            elif name == "WebFetch":
                out["wf_count"] += 1
                out["any_web"] = True
                url = inp.get("url") if isinstance(inp, dict) else None
                if url and UPSTREAM_HOSTS.search(url):
                    out["upstream_urls"].add(url)
            elif name == "Bash":
                cmd = inp.get("command", "") if isinstance(inp, dict) else ""
                if cmd and FETCHER_RE.search(cmd):
                    ext = [u for u in URL_RE.findall(cmd) if UPSTREAM_HOSTS.search(u)]
                    if ext or "git clone" in cmd:
                        out["shell_fetch_cmds"].append(cmd[:200])
                        out["upstream_urls"].update(ext)
                        out["any_web"] = True
    return out


def _resolved(trial_dir: Path):
    rj = trial_dir / "verifier" / "reward.json"
    if not rj.is_file():
        return None
    try:
        return bool(json.loads(rj.read_text()).get("resolved"))
    except Exception:
        return None


def _task_name(trial_dir: Path) -> str:
    res_p = trial_dir / "result.json"
    if res_p.is_file():
        try:
            return json.loads(res_p.read_text()).get("task_name") or trial_dir.name.rsplit("-", 1)[0]
        except Exception:
            pass
    return trial_dir.name.rsplit("-", 1)[0]


def _agent_kind(trial_dir: Path) -> str | None:
    """Return 'atif', 'claude-code', or None depending on which transcript exists.

    'atif' covers any agent that emits ATIF trajectory.json (codex, opencode, ...).
    'claude-code' is claude-code's JSONL transcript.

    Prefer claude-code.txt over rebuilt trajectory.json for claude-code trials —
    the harbor-lab rebuild drops WebFetch/WebSearch tool_calls, which would
    mask cheating. trajectory.json remains the only option for codex/opencode
    (they don't write claude-code.txt).
    """
    if (trial_dir / "agent" / "claude-code.txt").is_file():
        return "claude-code"
    if (trial_dir / "agent" / "trajectory.json").is_file():
        return "atif"
    return None


def scan_root(root: Path) -> list[dict]:
    """Walk a trial root and return one record per trial.

    Uses ``os.walk(followlinks=True)`` so that aggregator dirs assembled
    from ``ln -s`` of individual iter dirs are walked correctly.
    ``Path.rglob`` does not follow directory symlinks on Python 3.12.
    """
    import os as _os

    records = []
    seen_real: set[str] = set()
    for dirpath, _dirnames, filenames in _os.walk(root, followlinks=True):
        real = _os.path.realpath(dirpath)
        if real in seen_real:
            continue
        seen_real.add(real)
        if "result.json" not in filenames:
            continue
        trial_dir = Path(dirpath)
        kind = _agent_kind(trial_dir)
        if kind is None:
            continue
        if kind == "atif":
            sig = _scan_atif(trial_dir / "agent" / "trajectory.json")
        else:
            sig = _scan_claude_code(trial_dir / "agent" / "claude-code.txt")
        records.append(
            {
                "root": str(root),
                "trial_dir": str(trial_dir),
                "task": _task_name(trial_dir),
                "agent": kind,
                "resolved": _resolved(trial_dir),
                "ws": sig["ws_count"],
                "wf": sig["wf_count"],
                "upstream_url_count": len(sig["upstream_urls"]),
                "upstream_urls": sorted(sig["upstream_urls"]),
                "shell_fetch_cmds": sig["shell_fetch_cmds"],
                "any_web": sig["any_web"],
                "fetched_upstream": bool(sig["upstream_urls"] or sig["shell_fetch_cmds"]),
            }
        )
    return records


def emit_report(records: list[dict]) -> None:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)

    upstream_records = [r for r in records if r["fetched_upstream"]]
    print("# Integrity scan\n")
    print(f"Trials scanned: {len(records)}")
    print(f"Trials with any web access: {sum(1 for r in records if r['any_web'])}")
    print(f"Trials with upstream-source fetch: {len(upstream_records)}")
    pass_tainted = sum(1 for r in upstream_records if r["resolved"])
    fail_tainted = sum(1 for r in upstream_records if r["resolved"] is False)
    print(f"  of which PASS: {pass_tainted}")
    print(f"  of which fail: {fail_tainted}")
    print()

    upstream_tasks = sorted({r["task"] for r in upstream_records})
    if not upstream_tasks:
        return

    print("## Per-task breakdown (tasks where at least one trial fetched upstream code)\n")
    for task in upstream_tasks:
        rows = by_task[task]
        n_trials = len(rows)
        clean_pass = sum(1 for r in rows if r["resolved"] and not r["fetched_upstream"])
        tainted_pass = sum(1 for r in rows if r["resolved"] and r["fetched_upstream"])
        clean_fail = sum(1 for r in rows if r["resolved"] is False and not r["fetched_upstream"])
        tainted_fail = sum(1 for r in rows if r["resolved"] is False and r["fetched_upstream"])
        print(f"### {task}  ({n_trials} trial(s))")
        print(
            f"  passes: clean={clean_pass}  tainted={tainted_pass}    "
            f"fails: clean={clean_fail}  tainted={tainted_fail}"
        )
        for r in sorted(rows, key=lambda x: (not x["fetched_upstream"], x["resolved"] is not True)):
            if not r["fetched_upstream"]:
                continue
            res = "PASS" if r["resolved"] else ("fail" if r["resolved"] is False else "?")
            sigs = []
            if r["ws"]:
                sigs.append(f"WS={r['ws']}")
            if r["wf"]:
                sigs.append(f"WF={r['wf']}")
            if r["shell_fetch_cmds"]:
                sigs.append(f"shell={len(r['shell_fetch_cmds'])}")
            if r["upstream_urls"]:
                sigs.append(f"upstream={r['upstream_url_count']}")
            print(f"  [{res}] {r['trial_dir']}  ({', '.join(sigs)})")
            for u in r["upstream_urls"][:3]:
                print(f"      ↳ {u}")
            if len(r["upstream_urls"]) > 3:
                print(f"      ↳ ...+{len(r['upstream_urls']) - 3} more")
        print()


def write_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "root",
                "trial_dir",
                "task",
                "agent",
                "resolved",
                "ws",
                "wf",
                "upstream_url_count",
                "any_web",
                "fetched_upstream",
                "first_upstream_url",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r["root"],
                    r["trial_dir"],
                    r["task"],
                    r["agent"],
                    "" if r["resolved"] is None else int(r["resolved"]),
                    r["ws"],
                    r["wf"],
                    r["upstream_url_count"],
                    int(r["any_web"]),
                    int(r["fetched_upstream"]),
                    (r["upstream_urls"][0] if r["upstream_urls"] else ""),
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "roots", nargs="+", type=Path, help="Trial roots (recursively walked for result.json+agent/*)."
    )
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output of every scanned trial.")
    args = ap.parse_args()

    all_records: list[dict] = []
    for root in args.roots:
        if not root.is_dir():
            print(f"skip (not a dir): {root}", file=sys.stderr)
            continue
        rs = scan_root(root)
        print(f"  scanned {len(rs)} trials under {root}", file=sys.stderr)
        all_records.extend(rs)

    emit_report(all_records)

    if args.csv:
        write_csv(all_records, args.csv)
        print(f"\nWrote per-trial CSV to {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
