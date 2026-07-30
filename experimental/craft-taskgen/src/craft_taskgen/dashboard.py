#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate an interactive HTML dashboard from pipeline state JSON.

Usage:
    craft-taskgen-dashboard state.json
    craft-taskgen-dashboard --watch state.json
    open dashboard.html
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from craft_taskgen.config import PipelineContext

STAGE_ORDER = [
    "candidate",
    "evaluated",
    "promising",
    "built",
    "alignment_checked",
    "tests_discovered",
    "dockerfile_built",
    "f2p_p2p_classified",
    "oracle_checked",
    "opus_smoke_tested",
    "opus_triaged",
    "accepted",
]
STAGE_LABELS = {
    "candidate": "Candidate",
    "evaluated": "Evaluated",
    "promising": "Promising",
    "built": "Task Created",
    "alignment_checked": "Alignment",
    "tests_discovered": "Tests Found",
    "dockerfile_built": "Dockerfile",
    "f2p_p2p_classified": "F2P/P2P",
    "oracle_checked": "Oracle",
    "opus_smoke_tested": "Opus",
    "opus_triaged": "Opus OK",
    "accepted": "Accepted",
    "needs_fix": "Needs Human",
    "rejected": "Rejected",
}
NEXT_STEP_LABEL = {
    "promising": "task creation",
    "built": "alignment check",
    "alignment_checked": "assemble task artifacts",
    "tests_discovered": "write Dockerfile",
    "dockerfile_built": "F2P/P2P classify",
    "f2p_p2p_classified": "oracle check",
    "oracle_checked": "Opus smoke test",
    "opus_smoke_tested": "Opus triage",
    "opus_triaged": "accept/reject",
}
IN_PROGRESS_LABELS = {
    "evaluate": "Evaluating",
    "build": "Writing instruction",
    "alignment": "Alignment judge",
    "assemble_task_dir_artifacts": "Assembling task artifacts",
    "build_dockerfile": "Writing Dockerfile",
    "f2p_p2p_classify": "F2P/P2P classify",  # old name (pre-reorder MR)
    "docker_classify": "F2P/P2P classify",
    "oracle_check": "Oracle check",
    "Opus_smoke": "Opus smoke test",
    "Opus_deep_dive": "Opus deep dive",
    "Opus_review": "Opus skeptical review",
    "Opus_fix": "Opus auto-fix",
    "comparison": "Finalize accept/reject",
}


def _esc(s: str | None) -> str:
    if not s:
        return ""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _trunc(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s


VERDICT_DISPLAY = {
    "accept": "Accept",
    "reject": "Reject",
}


def _verdict(v: str) -> str:
    return VERDICT_DISPLAY.get(v, v)


def _make_expandable(text: str, threshold: int = 200) -> str:
    if len(text) <= threshold:
        return text
    short = text[:threshold]
    return (
        f'<span class="truncated">{short}...</span>'
        f'<span class="full" style="display:none">{text}</span> '
        f'<a class="expand-link" onclick="event.stopPropagation();expandText(this)">show more</a>'
    )


def _stage_idx(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return -1


def _render_iteration_timeline(t: dict) -> str:
    log = t.get("iteration_log", [])
    fix_history = t.get("fix_history", [])
    if not log:
        return ""

    blocks = ""
    for entry in log:
        step = entry.get("step", "?")
        ts = entry.get("timestamp", "")

        time_str = ""
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%H:%M")
            except Exception:
                pass

        # --- Determine title, body, color, and whether this is a compact row or rich card ---
        title = step
        body = ""
        color = "var(--text2)"
        rich = False  # compact row by default; rich = card with body

        if step == "evaluate":
            verdict = entry.get("verdict", "?")
            color = "var(--green)" if verdict == "accept" else "var(--red)"
            title = f"Task viability: {_verdict(verdict)}"

        elif step == "build":
            color = "var(--green)"
            words = entry.get("instruction_words", "?")
            task_dir = entry.get("task_dir", "")
            dir_name = task_dir.split("/")[-1] if task_dir else "?"
            title = f"Task creation: {dir_name} ({words} words)"

        elif step == "build_failed":
            color = "var(--red)"
            title = f"Build failed (ID: {entry.get('task_id_assigned', '?')})"

        elif step == "assemble_task_dir_artifacts":
            n_tests = entry.get("n_test_paths", entry.get("test_paths_count", 0))
            color = "var(--green)"
            title = (
                f"Assembled task artifacts ({n_tests} test file(s), solve.sh ready)"
                if n_tests
                else "Assembled task artifacts"
            )

        elif step == "build_dockerfile":
            color = "var(--green)"
            title = "Dockerfile created"

        elif step in ("f2p_p2p_classify", "docker_classify"):
            f2p_count = entry.get("f2p_count", 0)
            p2p_count = entry.get("p2p_count", 0)
            color = "var(--green)"
            title = f"F2P/P2P classified: {f2p_count} F2P, {p2p_count} P2P"

        elif step == "oracle_check":
            resolved = entry.get("oracle_resolved", False)
            color = "var(--green)" if resolved else "var(--red)"
            title = "Oracle: resolved" if resolved else "Oracle: NOT resolved"

        elif step == "docker_build_fail":
            color = "var(--amber)"
            title = "Docker build failed"

        elif step == "docker_unmod_pass":
            color = "var(--amber)"
            passing = entry.get("passing_tests", [])
            failing = entry.get("failing_tests", [])
            title = f"Tests pass on unmodified repo ({len(passing)} to remove, {len(failing)} to keep)"

        elif step == "docker_ref_fail":
            color = "var(--amber)"
            title = "Reference commit fails"

        elif step == "docker_fix":
            rich = True
            color = "var(--amber)"
            fix_idx = entry.get("fix_attempt", 0)
            trigger = entry.get("trigger", "")
            trigger_label = {
                "build_fail": "Docker build",
                "unmod_pass": "verifier too lenient",
                "ref_fail": "reference commit",
            }.get(trigger, trigger)
            title = f"Task fix #{fix_idx} — {trigger_label}"
            body = ""
            if fix_idx and fix_idx <= len(fix_history):
                fh = fix_history[fix_idx - 1]
                summary = fh.get("summary", "") if isinstance(fh, dict) else str(fh)[:150]
                if summary:
                    body = _esc(summary)

        elif "_smoke" in step:
            score = entry.get("score", "")
            model = step.replace("_smoke", "").capitalize()
            color = "var(--blue)"
            title = f"{model} smoke test: {score}"

        elif "_deep_dive" in step:
            rich = True
            color = "var(--blue)"
            assessment = _esc(entry.get("assessment", ""))
            dive_failures = entry.get("failures", [])
            title = "Deep dive"
            body = _trunc(assessment, 300) if assessment else ""
            if dive_failures:
                max_show = 3
                lines = []
                for df in dive_failures[:max_show]:
                    cls = df.get("classification", "?")
                    tname = df.get("test", "?").split("::")[-1]
                    dot = "●" if cls == "keep" else "○"
                    lines.append(f"{dot} {tname}: {cls}")
                if len(dive_failures) > max_show:
                    remaining = len(dive_failures) - max_show
                    # Count by classification
                    cls_counts: dict[str, int] = {}
                    for df in dive_failures[max_show:]:
                        c = df.get("classification", "?")
                        cls_counts[c] = cls_counts.get(c, 0) + 1
                    summary = ", ".join(f"{v} {k}" for k, v in cls_counts.items())
                    lines.append(f"<i>... +{remaining} more ({summary})</i>")
                body += "<br>" + "<br>".join(lines) if body else "<br>".join(lines)

        elif "_review" in step:
            rich = True
            color = "var(--accent)"
            verdict = _esc(entry.get("reviewer_verdict", ""))
            reclass = entry.get("reclassifications", [])
            coincidental = entry.get("coincidental_passes", [])
            title = "Skeptical reviewer"
            body = _trunc(verdict, 300) if verdict else ""
            if reclass:
                lines = [f"{r['test'].split('::')[-1]}: {r['from']} → {r['to']}" for r in reclass]
                body += "<br>Reclassified: " + ", ".join(lines)
            if coincidental:
                body += f"<br>Coincidental passes: {', '.join(c.split('::')[-1] for c in coincidental)}"

        elif "_fix" in step:
            rich = True
            color = "var(--amber)"
            issues = entry.get("issues_found", [])
            fix_idx = entry.get("fix_attempt", 0)
            issue_list = ", ".join(f"{i['test'].split('::')[-1]}: {i['classification']}" for i in issues)
            title = f"Task fix #{fix_idx} — Triage"
            body = issue_list
            if fix_idx and fix_idx <= len(fix_history):
                fh = fix_history[fix_idx - 1]
                summary = fh.get("summary", "") if isinstance(fh, dict) else str(fh)[:150]
                if summary:
                    body += f"<br>{_esc(summary)}" if body else _esc(summary)
        elif "_accepted" in step:
            color = "var(--green)"
            issues = entry.get("issues", [])
            keep_count = [i for i in issues if i.get("classification") == "keep"]
            model = step.replace("_accepted", "").capitalize()
            title = f"Passed {model} triage — {len(keep_count)} legit failing tests"

        elif step == "comparison":
            outcome = entry.get("outcome", "?")
            opus = entry.get("opus_score", "?")
            color = "var(--green)" if outcome == "accepted" else "var(--red)"
            title = f"Finalize: {outcome.upper()} (Opus={opus})"

        if rich:
            blocks += f"""<div class="tl-card" style="border-left-color:{color}">
                <div class="tl-header">
                    <span class="tl-time">{time_str}</span>
                    <span class="tl-title">{_esc(title)}</span>
                </div>
                {"<div class='tl-body'>" + body + "</div>" if body else ""}
            </div>"""
        else:
            blocks += f"""<div class="tl-row">
                <span class="tl-time">{time_str}</span>
                <span class="tl-dot" style="background:{color}"></span>
                <span class="tl-label">{_esc(title)}</span>
            </div>"""

    # Show in-progress indicator if a step is currently running
    in_progress = t.get("in_progress_step", "")
    if in_progress:
        step_label = IN_PROGRESS_LABELS.get(in_progress, in_progress.replace("_", " ").capitalize())
        blocks += f"""<div class="tl-row tl-in-progress">
            <span class="tl-time">now</span>
            <span class="tl-dot tl-dot-pulse" style="background:var(--blue)"></span>
            <span class="tl-label">{step_label}...</span>
        </div>"""

    return f'<div class="detail-block"><h4>Timeline</h4><div class="tl-timeline">{blocks}</div></div>'


def generate_html(state_path: str) -> str:
    with open(state_path) as f:
        data = json.load(f)

    tasks = data.get("tasks", {})
    last_updated = data.get("last_updated", "unknown")
    try:
        updated_dt = datetime.fromisoformat(last_updated)
        age = datetime.now() - updated_dt
        age_str = (
            f"{int(age.total_seconds())}s ago"
            if age.total_seconds() < 120
            else f"{int(age.total_seconds() / 60)}m ago"
        )
    except Exception:
        age_str = ""

    # Counts
    stage_counts: dict[str, int] = {}
    for t in tasks.values():
        s = t.get("stage", "unknown")
        stage_counts[s] = stage_counts.get(s, 0) + 1

    total = len(tasks)
    instruction_preamble = PipelineContext().instruction_preamble
    accepted = stage_counts.get("accepted", 0)
    rejected = stage_counts.get("rejected", 0)
    needs_fix = stage_counts.get("needs_fix", 0)
    in_progress = total - accepted - rejected - needs_fix
    total_fixes = sum(t.get("fix_attempts", 0) for t in tasks.values())

    # Sort: running first, then queued, needs_fix, accepted, rejected
    def sort_key(item):
        _, t = item
        s = t.get("stage", "")
        if t.get("in_progress_step"):
            return (0, t.get("repo", ""))
        elif s == "accepted":
            return (2, t.get("repo", ""))
        elif s == "needs_fix":
            return (3, t.get("repo", ""))
        elif s == "rejected":
            return (4, t.get("repo", ""))
        else:
            return (1, t.get("repo", ""))

    sorted_tasks = sorted(tasks.items(), key=sort_key)

    # Build task rows + details
    task_rows = ""
    for tid, t in sorted_tasks:
        stage = t.get("stage", "?")
        opus = t.get("opus_score", "")
        fix_n = t.get("fix_attempts", 0)
        repo = t.get("repo", "?")
        desc = _trunc(t.get("description", ""), 60)
        in_prog = t.get("in_progress_step", "")
        if in_prog:
            label = IN_PROGRESS_LABELS.get(in_prog, in_prog.replace("_", " ").capitalize()) + "..."
        elif stage in NEXT_STEP_LABEL:
            label = f"Queued: {NEXT_STEP_LABEL[stage]}"
        else:
            label = STAGE_LABELS.get(stage, stage)

        if stage == "accepted":
            badge_cls = "badge-accepted"
        elif stage == "rejected":
            badge_cls = "badge-rejected"
        elif stage == "needs_fix":
            badge_cls = "badge-warn"
        elif in_prog:
            badge_cls = "badge-running"
        else:
            badge_cls = "badge-active"

        # Timeline dots
        current_idx = _stage_idx(stage)
        in_prog = t.get("in_progress_step", "")
        dots = ""
        for i, st in enumerate(STAGE_ORDER):
            if st == stage and in_prog:
                # Completed this stage, next is in progress
                dots += '<span class="dot dot-done"></span>'
            elif st == stage:
                dots += '<span class="dot dot-current"></span>'
            elif i == current_idx + 1 and in_prog:
                dots += '<span class="dot dot-pulse"></span>'
            elif i < current_idx:
                dots += '<span class="dot dot-done"></span>'
            else:
                dots += '<span class="dot dot-future"></span>'
        if stage == "needs_fix":
            dots += '<span class="dot dot-warn"></span>'
        elif stage == "rejected":
            dots += '<span class="dot dot-fail"></span>'

        # Detail content
        task_dir = t.get("task_dir", "")
        opus_trial = t.get("opus_trial_dir", "")
        review_reason = _esc(t.get("human_review_reason", ""))

        # Read live files from disk if task_dir exists
        instruction_html = ""
        tests_html = ""
        if task_dir and os.path.isdir(task_dir):
            instr_path = os.path.join(task_dir, "instruction.md")
            if os.path.isfile(instr_path):
                with open(instr_path) as f:
                    raw = f.read()
                # Strip boilerplate environment section
                if "## Environment" in raw:
                    raw = raw[: raw.index("## Environment")].strip()
                # Strip leading "#..." header
                if raw.startswith("# "):
                    raw = raw.split("\n", 1)[1].strip()
                # Strip preamble line
                if raw.startswith(instruction_preamble):
                    raw = raw.split("\n", 1)[1].strip()
                instruction_html = _esc(raw)

            f2p_path = os.path.join(task_dir, "tests", "fail_to_pass.txt")
            p2p_path = os.path.join(task_dir, "tests", "pass_to_pass.txt")
            if os.path.isfile(f2p_path):
                with open(f2p_path) as f:
                    f2p_count = sum(1 for line in f if line.strip())
                p2p_count = 0
                if os.path.isfile(p2p_path):
                    with open(p2p_path) as f:
                        p2p_count = sum(1 for line in f if line.strip())
                tests_html = f"{f2p_count} F2P, {p2p_count} P2P"
            else:
                gold_path = os.path.join(task_dir, "tests", "gold_reference_tests.py")
                if os.path.isfile(gold_path):
                    with open(gold_path) as f:
                        test_count = sum(1 for line in f if line.strip().startswith("def test_"))
                    tests_html = f"{test_count} gold tests"

        # fix_history and issues are now rendered inside the timeline blocks

        task_rows += f"""
        <div class="task-card" onclick="toggle('{tid}')">
            <div class="task-header">
                <div class="task-id">{tid}</div>
                <div class="task-dots">{dots}</div>
                <span class="badge {badge_cls}">{label}</span>
                <div class="task-scores">
                    {"<span class='score'>Opus " + opus + "</span>" if opus else ""}
                </div>
                {f'<span class="fix-count">{fix_n}×fix</span>' if fix_n else ""}
            </div>
            <div class="task-sub">{repo} — {desc}</div>
            {f'<div class="task-summary">{_esc(t.get("summary", ""))}</div>' if t.get("summary") else ""}
            <div id="d-{tid}" class="task-detail" style="display:none">
                {_render_iteration_timeline(t)}
                {f'<div class="detail-block"><h4 class="collapsible" onclick="toggleSection(this)">Instruction (latest) ▾</h4><div class="instruction-pre" style="display:none">{instruction_html}</div></div>' if instruction_html else ""}
                <hr style="border:none;border-top:1px solid var(--border);margin:12px 0">
                <div class="detail-section">
                    <div class="detail-col">
                        <h4>Tests</h4>
                        <p>{tests_html if tests_html else "—"}</p>
                    </div>
                    <div class="detail-col">
                        <h4>Paths (latest run)</h4>
                        <p class="mono">{task_dir}</p>
                        {f'<p class="mono">Opus: {opus_trial}</p>' if opus_trial else ""}
                    </div>
                </div>
                {f'<div class="detail-block review-block"><h4>For Human Review</h4><p>This task needs human intervention — the auto-fix loop could not resolve the issue.<br><b>Reason:</b> {review_reason}</p></div>' if review_reason else ""}
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CRAFT Pipeline</title>
<meta http-equiv="refresh" content="30">
<script>
// Minimal markdown renderer (no external deps)
function minimd(s) {{
    if (!s) return '';
    var lines = s.split('\\n');
    var out = [];
    for (var i = 0; i < lines.length; i++) {{
        var L = lines[i];
        if (L.match(/^### /)) {{ out.push('<h3>' + L.slice(4) + '</h3>'); }}
        else if (L.match(/^## /)) {{ out.push('<h2>' + L.slice(3) + '</h2>'); }}
        else if (L.match(/^# /)) {{ out.push('<h1>' + L.slice(2) + '</h1>'); }}
        else if (L.match(/^- /)) {{ out.push('<li>' + L.slice(2) + '</li>'); }}
        else if (L.match(/^\\d+\\. /)) {{ out.push('<li>' + L.replace(/^\\d+\\.\\s*/, '') + '</li>'); }}
        else if (L.trim() === '') {{ if (out.length && out[out.length-1] !== '<br>') out.push('<br>'); }}
        else {{ out.push(L); }}
    }}
    var html = out.join('\\n');
    html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    return html;
}}
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {{
    --bg: #0c0e14;
    --surface: #141620;
    --surface2: #1a1d2e;
    --border: #252840;
    --text: #c8cad0;
    --text2: #6b7084;
    --green: #34d399;
    --green-bg: #0d2818;
    --red: #f87171;
    --red-bg: #2a1215;
    --amber: #fbbf24;
    --amber-bg: #2a2008;
    --blue: #60a5fa;
    --blue-bg: #0c1a30;
    --accent: #818cf8;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text);
        padding: 24px 32px; line-height: 1.5; }}
h1 {{ font-size: 1.1em; font-weight: 700; color: #fff; letter-spacing: 0.08em;
      text-transform: uppercase; }}
.header {{ display: flex; justify-content: space-between; align-items: center;
           margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
.header-right {{ font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: var(--text2); }}
.cards {{ display: flex; gap: 12px; margin-bottom: 24px; }}
.stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
              padding: 14px 20px; flex: 1; }}
.stat-num {{ font-family: 'JetBrains Mono', monospace; font-size: 1.8em; font-weight: 600; }}
.stat-label {{ font-size: 0.75em; color: var(--text2); text-transform: uppercase;
               letter-spacing: 0.06em; margin-top: 2px; }}
.task-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
              margin-bottom: 8px; cursor: pointer; transition: border-color 0.15s; }}
.task-card:hover {{ border-color: var(--accent); }}
.task-header {{ display: flex; align-items: center; gap: 12px; padding: 12px 16px; }}
.task-id {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85em; font-weight: 600;
            color: #fff; min-width: 160px; }}
.task-dots {{ display: flex; gap: 3px; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; }}
.dot-done {{ background: var(--green); opacity: 0.6; }}
.dot-current {{ background: var(--blue); box-shadow: 0 0 6px var(--blue); }}
.dot-pulse {{ background: var(--blue); animation: pulse 1.5s ease-in-out infinite; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 0.3; box-shadow: none; }} 50% {{ opacity: 1; box-shadow: 0 0 8px var(--blue); }} }}
.dot-future {{ background: var(--border); }}
.dot-warn {{ background: var(--amber); }}
.dot-fail {{ background: var(--red); }}
.badge {{ font-family: 'JetBrains Mono', monospace; font-size: 0.7em; padding: 3px 8px;
          border-radius: 4px; font-weight: 600; letter-spacing: 0.04em; }}
.badge-accepted {{ background: var(--green-bg); color: var(--green); }}
.badge-rejected {{ background: var(--red-bg); color: var(--red); }}
.badge-warn {{ background: var(--amber-bg); color: var(--amber); }}
.badge-active {{ background: var(--blue-bg); color: var(--blue); }}
.badge-running {{ background: var(--blue); color: #fff; animation: pulse-badge 1.2s ease-in-out infinite; }}
@keyframes pulse-badge {{ 0%, 100% {{ opacity: 0.7; }} 50% {{ opacity: 1; }} }}
.task-scores {{ font-family: 'JetBrains Mono', monospace; font-size: 0.8em; margin-left: auto; }}
.score {{ margin-left: 10px; color: var(--text2); }}
.fix-count {{ font-family: 'JetBrains Mono', monospace; font-size: 0.7em; color: var(--amber);
              margin-left: 8px; }}
.task-sub {{ padding: 0 16px 10px; font-size: 0.8em; color: var(--text2); }}
.task-summary {{ padding: 4px 16px 10px; font-size: 0.85em; color: var(--text); line-height: 1.5;
    background: rgba(74,170,74,0.08); border-left: 3px solid var(--green); margin: 0 12px 8px; padding: 8px 12px;
    border-radius: 4px; }}
.task-detail {{ padding: 0 16px 16px; }}
.detail-section {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px; }}
.detail-col h4 {{ font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.08em;
                  color: var(--accent); margin-bottom: 4px; }}
.detail-col p {{ font-size: 0.82em; color: var(--text2); }}
.mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72em; word-break: break-all; }}
.detail-block {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }}
.detail-block h4 {{ font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.08em;
                    color: var(--accent); margin-bottom: 8px; }}
.review-block p {{ font-size: 0.82em; color: var(--amber); }}
.collapsible {{ cursor: pointer; user-select: none; }}
.collapsible:hover {{ color: #fff; }}
.collapsible.collapsed {{ opacity: 0.7; }}
.tl-timeline {{ display: flex; flex-direction: column; gap: 2px; }}
.tl-row {{ display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 0.82em; }}
.tl-dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}
.tl-dot-pulse {{ animation: pulse 1.5s ease-in-out infinite; }}
.tl-label {{ color: var(--text); }}
.tl-card {{ border-left: 3px solid var(--border); padding: 8px 12px; border-radius: 0 6px 6px 0;
            background: var(--bg); font-size: 0.82em; margin: 4px 0; }}
.tl-header {{ display: flex; align-items: center; gap: 10px; }}
.tl-time {{ font-family: 'JetBrains Mono', monospace; font-size: 0.8em; color: var(--text2);
            min-width: 36px; }}
.tl-title {{ color: #fff; font-weight: 500; }}
.tl-body {{ color: var(--text2); margin-top: 4px; padding-left: 46px; line-height: 1.5; }}
.tl-in-progress {{ animation: pulse-border 1.5s ease-in-out infinite; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 0.3; }} 50% {{ opacity: 1; }} }}
@keyframes pulse-border {{ 0%, 100% {{ opacity: 0.5; }} 50% {{ opacity: 1; }} }}
.instruction-pre {{ background: var(--bg); padding: 12px; border-radius: 6px; font-size: 0.82em;
                    line-height: 1.7; color: var(--text); white-space: pre-line; }}
.md-content h1, .md-content h2, .md-content h3 {{ color: #fff; font-size: 0.95em; margin: 8px 0 4px; }}
.md-content p {{ margin: 4px 0; }}
.md-content code {{ font-family: 'JetBrains Mono', monospace; background: var(--surface2);
                    padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
.md-content pre {{ background: var(--surface2); padding: 8px; border-radius: 4px;
                   overflow-x: auto; font-size: 0.85em; }}
.md-content ul, .md-content ol {{ padding-left: 20px; margin: 4px 0; }}
.md-content strong {{ color: #fff; }}
@media (max-width: 768px) {{
    .cards {{ flex-wrap: wrap; }}
    .detail-section {{ grid-template-columns: 1fr; }}
    .task-header {{ flex-wrap: wrap; }}
}}
</style>
<script>
function toggleSection(h) {{
    event.stopPropagation();
    var el = h.nextElementSibling;
    var open = el.style.display === 'none';
    el.style.display = open ? 'block' : 'none';
    h.classList.toggle('collapsed');
}}
function toggle(id) {{
    var el = document.getElementById('d-' + id);
    var open = el.style.display === 'none';
    el.style.display = open ? 'block' : 'none';
    // Persist in sessionStorage
    var expanded = JSON.parse(sessionStorage.getItem('expanded') || '{{}}');
    if (open) expanded[id] = true; else delete expanded[id];
    sessionStorage.setItem('expanded', JSON.stringify(expanded));
}}
function expandText(link) {{
    var parent = link.parentElement;
    var trunc = parent.querySelector('.truncated');
    var full = parent.querySelector('.full');
    var id = link.dataset.eid || '';
    if (trunc && trunc.style.display !== 'none') {{
        if (trunc) trunc.style.display = 'none';
        full.style.display = full.tagName === 'DIV' ? 'block' : 'inline';
        link.textContent = 'show less';
        if (full.classList.contains('md-content') && !full.dataset.rendered) {{
            full.innerHTML = minimd(full.textContent);
            full.dataset.rendered = '1';
        }}
        if (id) {{ var exp = JSON.parse(sessionStorage.getItem('exp2') || '{{}}'); exp[id] = true; sessionStorage.setItem('exp2', JSON.stringify(exp)); }}
    }} else {{
        if (trunc) trunc.style.display = 'inline';
        full.style.display = 'none';
        link.textContent = 'show more';
        if (id) {{ var exp = JSON.parse(sessionStorage.getItem('exp2') || '{{}}'); delete exp[id]; sessionStorage.setItem('exp2', JSON.stringify(exp)); }}
    }}
}}
// Restore expanded state after refresh + render markdown
window.addEventListener('DOMContentLoaded', function() {{
    var expanded = JSON.parse(sessionStorage.getItem('expanded') || '{{}}');
    for (var id in expanded) {{
        var el = document.getElementById('d-' + id);
        if (el) el.style.display = 'block';
    }}
    // Render markdown in all .md-content elements
    document.querySelectorAll('.md-content:not([data-rendered])').forEach(function(el) {{
        el.innerHTML = minimd(el.textContent);
        el.dataset.rendered = '1';
    }});
    // Restore inner expand state
    var exp2 = JSON.parse(sessionStorage.getItem('exp2') || '{{}}');
    for (var eid in exp2) {{
        var link = document.querySelector('[data-eid="' + eid + '"]');
        if (link) expandText(link);
    }}
}});
</script>
</head>
<body>
<div class="header">
    <h1>CRAFT &middot; Task Pipeline</h1>
    <div class="header-right">{os.path.basename(state_path)} &middot; {age_str}</div>
</div>

<div class="cards">
    <div class="stat-card">
        <div class="stat-num" style="color:var(--green)">{accepted}</div>
        <div class="stat-label">Accepted</div>
    </div>
    <div class="stat-card">
        <div class="stat-num" style="color:var(--blue)">{in_progress}</div>
        <div class="stat-label">In Progress</div>
    </div>
    <div class="stat-card">
        <div class="stat-num" style="color:var(--amber)">{needs_fix}</div>
        <div class="stat-label">Needs Human</div>
    </div>
    <div class="stat-card">
        <div class="stat-num" style="color:var(--red)">{rejected}</div>
        <div class="stat-label">Rejected</div>
    </div>
    <div class="stat-card">
        <div class="stat-num" style="color:var(--accent)">{total_fixes}</div>
        <div class="stat-label">Fix Iterations</div>
    </div>
</div>

{task_rows}

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_file", help="Pipeline state JSON file")
    parser.add_argument("--output", default="dashboard.html", help="Output HTML file")
    parser.add_argument("--watch", action="store_true", help="Regenerate every 10s")
    args = parser.parse_args()

    if args.watch:
        print(f"Watching {args.state_file} → {args.output} (Ctrl+C to stop)", flush=True)
        while True:
            try:
                html = generate_html(args.state_file)
                Path(args.output).write_text(html)
            except Exception as e:
                print(f"  Error: {e}", flush=True)
            time.sleep(10)
    else:
        html = generate_html(args.state_file)
        Path(args.output).write_text(html)
        print(f"Dashboard written to {args.output}")


if __name__ == "__main__":
    main()
