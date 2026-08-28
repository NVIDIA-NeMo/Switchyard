# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a self-contained side-by-side HTML comparing instructions.

Joins calibrate-alignment.py's output CSV (column ``new_instruction_md``) with
the original SWE-Bench-Pro ``problem_statement`` per ``task_id`` and emits a
single HTML file with each PR rendered in two columns: our pipeline's
instruction on the left, SWE-Pro's curated problem_statement on the right.

Markdown is rendered server-side via the ``markdown`` package so the page has
no external CDN dependencies. CSS is inlined. The page works offline and can
be hosted via ``python -m http.server`` for sharing inside NVIDIA.

Usage:
    uv run python scripts/swepro/render.py \\
        --calib swepro_calib.csv \\
        --output-html swepro_comparison.html \\
        --output-csv swepro_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import markdown


def _maybe_json_decode(text: str) -> str:
    """SWE-Pro ships some text fields as JSON-quoted strings (leading `"`, escaped `\\n`).

    Detect that shape and decode so markdown receives real newlines.
    """
    if not text:
        return text
    s = text.strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"') and "\\n" in s:
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                return decoded
        except json.JSONDecodeError:
            pass
    return text


def _load_swepro_fields(dataset_name: str, split: str) -> dict[str, dict[str, str]]:
    """Return {instance_id: {problem_statement, requirements, interface}}.

    SWE-Pro tasks present all three to the solving agent — the comparison view
    needs all three to be apples-to-apples with our pipeline's instruction.
    """
    from datasets import load_dataset  # type: ignore

    sys.stderr.write(f"Loading {dataset_name} ({split} split) for SWE-Pro field lookup...\n")
    ds = load_dataset(dataset_name, split=split)
    return {
        rec["instance_id"]: {
            "problem_statement": _maybe_json_decode(rec.get("problem_statement", "") or ""),
            "requirements": _maybe_json_decode(rec.get("requirements", "") or ""),
            "interface": _maybe_json_decode(rec.get("interface", "") or ""),
        }
        for rec in ds
    }


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _render_md(text: str) -> str:
    if not text:
        return '<p class="empty">(empty)</p>'
    return markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "sane_lists", "nl2br"],
    )


def _verdict_badge(verdict: str) -> str:
    cls = {
        "ok": "badge-ok",
        "accept": "badge-ok",
        "reject": "badge-bad",
        "leaked": "badge-bad",
        "narrow_tests": "badge-warn",
        "narrow_tests_only": "badge-warn",
    }.get((verdict or "").strip(), "badge-neutral")
    label = html.escape(verdict or "—")
    return f'<span class="badge {cls}">{label}</span>'


def _classify_drop_stage(row: dict[str, str]) -> str:
    eval_v = (row.get("new_eval_verdict") or "").strip()
    instr = (row.get("new_instruction_md") or "").strip()
    align = (row.get("alignment_verdict") or "").strip()
    if eval_v and eval_v != "accept":
        return "eval"
    if not instr:
        return "build"
    if align and align != "ok":
        return "align"
    return "ok"


CSS = """
:root {
  --bg: #0e1116;
  --bg-card: #161b22;
  --bg-table: #11161d;
  --fg: #e6edf3;
  --fg-muted: #8b949e;
  --accent: #58a6ff;
  --border: #30363d;
  --code-bg: #1e242c;
  --good: #2ea043;
  --warn: #d29922;
  --bad: #f85149;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
header.page-header {
  padding: 28px 36px 18px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #11161d 0%, #0e1116 100%);
}
header.page-header h1 { margin: 0 0 6px; font-size: 22px; font-weight: 600; }
header.page-header p { margin: 0; color: var(--fg-muted); font-size: 14px; }
main { padding: 24px 36px 60px; max-width: 1600px; margin: 0 auto; }
section.summary {
  margin-bottom: 28px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}
section.summary h2 {
  margin: 0;
  padding: 14px 18px;
  font-size: 15px;
  font-weight: 600;
  border-bottom: 1px solid var(--border);
  background: var(--bg-table);
}
table.summary-table { width: 100%; border-collapse: collapse; font-size: 13px; }
table.summary-table th, table.summary-table td {
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.summary-table th {
  font-weight: 600;
  color: var(--fg-muted);
  background: var(--bg-table);
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.04em;
  position: sticky;
  top: 0;
}
table.summary-table tr:last-child td { border-bottom: none; }
table.summary-table td.numeric { text-align: right; font-variant-numeric: tabular-nums; }
table.summary-table a { color: var(--accent); text-decoration: none; }
table.summary-table a:hover { text-decoration: underline; }
.task-id { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace; font-size: 12px; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.02em;
}
.badge-ok { background: rgba(46, 160, 67, 0.16); color: var(--good); }
.badge-warn { background: rgba(210, 153, 34, 0.16); color: var(--warn); }
.badge-bad { background: rgba(248, 81, 73, 0.16); color: var(--bad); }
.badge-neutral { background: rgba(139, 148, 158, 0.16); color: var(--fg-muted); }
article.pr-card {
  margin-bottom: 28px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  scroll-margin-top: 16px;
}
article.pr-card > header {
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-table);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  justify-content: space-between;
}
article.pr-card > header .title-line {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
article.pr-card > header h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  color: var(--accent);
  word-break: break-all;
}
article.pr-card > header .repo {
  color: var(--fg-muted);
  font-size: 13px;
}
article.pr-card > header a.pr-link {
  color: var(--accent);
  font-size: 12px;
  text-decoration: none;
}
article.pr-card > header a.pr-link:hover { text-decoration: underline; }
.subject {
  padding: 10px 18px;
  border-bottom: 1px solid var(--border);
  color: var(--fg-muted);
  font-style: italic;
  font-size: 13px;
}
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
}
.two-col > div { padding: 18px 20px; min-width: 0; }
.two-col > div:first-child { border-right: 1px solid var(--border); }
.two-col h4 {
  margin: 0 0 12px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--fg-muted);
  font-weight: 600;
}
.two-col h4 .wc { font-weight: 400; opacity: 0.8; margin-left: 8px; }
.md-body { font-size: 14px; line-height: 1.6; }
.md-body p:first-child { margin-top: 0; }
.md-body p:last-child { margin-bottom: 0; }
.md-body p { margin: 8px 0; }
.md-body h1, .md-body h2, .md-body h3, .md-body h4 {
  margin: 14px 0 8px;
  font-weight: 600;
}
.md-body h1 { font-size: 18px; }
.md-body h2 { font-size: 16px; }
.md-body h3 { font-size: 15px; }
.md-body h4 { font-size: 14px; color: var(--fg-muted); }
.md-body code {
  background: var(--code-bg);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  font-size: 12.5px;
}
.md-body pre {
  background: var(--code-bg);
  padding: 10px 14px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 12.5px;
  border: 1px solid var(--border);
}
.md-body pre code { background: transparent; padding: 0; }
.md-body ul, .md-body ol { padding-left: 22px; margin: 8px 0; }
.md-body li { margin: 4px 0; }
.md-body blockquote {
  margin: 8px 0;
  padding: 4px 12px;
  border-left: 3px solid var(--border);
  color: var(--fg-muted);
}
.md-body table {
  border-collapse: collapse;
  margin: 10px 0;
  font-size: 13px;
}
.md-body table th, .md-body table td {
  border: 1px solid var(--border);
  padding: 5px 10px;
}
.md-body table th { background: var(--bg-table); }
.empty { color: var(--fg-muted); font-style: italic; }
section.swepro-field {
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px dashed var(--border);
}
section.swepro-field:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
section.swepro-field h5 {
  margin: 0 0 8px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--accent);
  font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}
section.swepro-field h5 .wc { color: var(--fg-muted); font-weight: 400; margin-left: 8px; }
@media (max-width: 1100px) {
  .two-col { grid-template-columns: 1fr; }
  .two-col > div:first-child { border-right: none; border-bottom: 1px solid var(--border); }
}
"""


def _build_markdown(rows: list[dict[str, str]], dataset_name: str) -> str:
    """Render side-by-side comparison as a markdown document.

    No summary table — caller said one isn't needed. Each PR gets a section
    with title, metadata line, and stacked subsections (ours then theirs).
    """
    parts: list[str] = []
    parts.append("# SWE-Pro ↔ craft-taskgen instruction comparison\n")
    parts.append(
        f"{len(rows)} PRs from `{dataset_name}` (test split) where our pipeline produced an instruction "
        "(eval-accept + alignment-ok). Each section shows our pipeline's build output followed by "
        "SWE-Pro's three-part task definition (`problem_statement`, `requirements`, `interface`).\n"
    )

    for i, row in enumerate(rows, start=1):
        task_id = row.get("task_id", "")
        repo = row.get("repo", "")
        subject = (row.get("subject", "") or "").strip()
        pr_url = row.get("pr_url", "")
        our_md = (row.get("new_instruction_md", "") or "").strip()
        their_problem = (row.get("swepro_problem_statement", "") or "").strip()
        their_reqs = (row.get("swepro_requirements", "") or "").strip()
        their_iface = (row.get("swepro_interface", "") or "").strip()
        regen = row.get("new_regen_count", "0")
        align_v = row.get("alignment_verdict", "")

        title = subject or task_id
        parts.append(f"\n---\n\n## {i}. {title}\n")
        meta_bits = [f"`{repo}`", f"alignment={align_v or '—'}", f"regens={regen}"]
        if pr_url:
            meta_bits.append(f"[PR]({pr_url})")
        meta_bits.append(f"`{task_id}`")
        parts.append("*" + " · ".join(meta_bits) + "*\n")

        parts.append(f"\n### Our pipeline’s instruction ({_word_count(our_md)} words)\n")
        parts.append(f"\n{our_md or '_(empty)_'}\n")

        their_total = _word_count(their_problem) + _word_count(their_reqs) + _word_count(their_iface)
        parts.append(f"\n### SWE-Pro task definition ({their_total} words total)\n")
        parts.append(f"\n#### problem_statement ({_word_count(their_problem)} words)\n")
        parts.append(f"\n{their_problem or '_(empty)_'}\n")
        parts.append(f"\n#### requirements ({_word_count(their_reqs)} words)\n")
        parts.append(f"\n{their_reqs or '_(empty)_'}\n")
        parts.append(f"\n#### interface ({_word_count(their_iface)} words)\n")
        parts.append(f"\n{their_iface or '_(empty)_'}\n")

    return "".join(parts)


def _build_html(rows: list[dict[str, str]], dataset_name: str) -> str:
    n_total = len(rows)
    n_ok = sum(1 for r in rows if _classify_drop_stage(r) == "ok")

    summary_rows = []
    cards = []
    for i, row in enumerate(rows):
        task_id = row.get("task_id", "")
        repo = row.get("repo", "")
        subject = row.get("subject", "")
        pr_url = row.get("pr_url", "")
        eval_v = row.get("new_eval_verdict", "")
        align_v = row.get("alignment_verdict", "")
        regen = row.get("new_regen_count", "0")
        our_md = row.get("new_instruction_md", "") or ""
        their_problem = row.get("swepro_problem_statement", "") or ""
        their_reqs = row.get("swepro_requirements", "") or ""
        their_iface = row.get("swepro_interface", "") or ""
        their_combined_words = _word_count(their_problem) + _word_count(their_reqs) + _word_count(their_iface)
        drop = _classify_drop_stage(row)
        anchor = f"pr-{i}"

        summary_rows.append(
            "<tr>"
            f'<td class="task-id"><a href="#{anchor}">{html.escape(task_id)}</a></td>'
            f"<td>{html.escape(repo)}</td>"
            f"<td>{_verdict_badge(eval_v)}</td>"
            f"<td>{_verdict_badge(align_v)}</td>"
            f'<td class="numeric">{_word_count(our_md)}</td>'
            f'<td class="numeric">{their_combined_words}</td>'
            f'<td class="numeric">{html.escape(regen)}</td>'
            f"<td>{_verdict_badge(drop)}</td>"
            "</tr>"
        )

        pr_link = (
            f'<a class="pr-link" href="{html.escape(pr_url)}" target="_blank" rel="noopener">PR ↗</a>'
            if pr_url
            else ""
        )
        their_inner = (
            f'<section class="swepro-field"><h5>problem_statement '
            f'<span class="wc">{_word_count(their_problem)} words</span></h5>'
            f'<div class="md-body">{_render_md(their_problem)}</div></section>'
            f'<section class="swepro-field"><h5>requirements '
            f'<span class="wc">{_word_count(their_reqs)} words</span></h5>'
            f'<div class="md-body">{_render_md(their_reqs)}</div></section>'
            f'<section class="swepro-field"><h5>interface '
            f'<span class="wc">{_word_count(their_iface)} words</span></h5>'
            f'<div class="md-body">{_render_md(their_iface)}</div></section>'
        )
        cards.append(
            f'<article class="pr-card" id="{anchor}">'
            "<header>"
            '<div class="title-line">'
            f"<h3>{html.escape(task_id)}</h3>"
            f'<span class="repo">{html.escape(repo)}</span>'
            f"</div>"
            f"{pr_link}"
            "</header>"
            f'<div class="subject">{html.escape(subject)}</div>'
            '<div class="two-col">'
            "<div>"
            f'<h4>Our pipeline’s instruction <span class="wc">{_word_count(our_md)} words</span></h4>'
            f'<div class="md-body">{_render_md(our_md)}</div>'
            "</div>"
            "<div>"
            f'<h4>SWE-Pro task definition <span class="wc">{their_combined_words} words total</span></h4>'
            f"{their_inner}"
            "</div>"
            "</div>"
            "</article>"
        )

    intro_text = (
        f"{n_total} PRs sampled from <code>{html.escape(dataset_name)}</code> (test split). "
        f"{n_ok} reached alignment-ok. Each card shows our pipeline’s build output (left) "
        "next to SWE-Pro’s curated <code>problem_statement</code> (right)."
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SWE-Pro vs craft-taskgen instruction comparison</title>
<style>{CSS}</style>
</head>
<body>
<header class="page-header">
<h1>SWE-Pro ↔ craft-taskgen instruction comparison</h1>
<p>{intro_text}</p>
</header>
<main>
<section class="summary">
<h2>Summary</h2>
<table class="summary-table">
<thead><tr>
<th>task_id</th><th>repo</th><th>eval</th><th>alignment</th>
<th class="numeric">our&nbsp;words</th><th class="numeric">swe-pro&nbsp;words</th>
<th class="numeric">regens</th><th>dropped&nbsp;at</th>
</tr></thead>
<tbody>
{"".join(summary_rows)}
</tbody>
</table>
</section>
{"".join(cards)}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True, help="calibrate-alignment.py output CSV")
    parser.add_argument("--output-html", default="swepro_comparison.html", help="Output HTML path")
    parser.add_argument("--output-csv", default="swepro_comparison.csv", help="Output joined CSV path")
    parser.add_argument(
        "--output-md",
        default="swepro_comparison.md",
        help="Output markdown path (for paste into Google Docs / Slack)",
    )
    parser.add_argument(
        "--dataset",
        default="ScaleAI/SWE-bench_Pro",
        help="HuggingFace dataset (default: ScaleAI/SWE-bench_Pro)",
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include rows our pipeline didn't produce an instruction for (eval/build/align dropped). "
        "Default: drop them so the page only shows successful comparisons.",
    )
    args = parser.parse_args()

    if not Path(args.calib).is_file():
        parser.error(f"--calib not found: {args.calib}")

    with open(args.calib) as f:
        rows = list(csv.DictReader(f))
    sys.stderr.write(f"Loaded {len(rows)} rows from {args.calib}\n")

    if not args.include_rejected:
        before = len(rows)
        rows = [r for r in rows if (r.get("new_instruction_md") or "").strip()]
        sys.stderr.write(f"Filtered out {before - len(rows)} rows with empty new_instruction_md\n")

    swepro_lookup = _load_swepro_fields(args.dataset, args.split)

    enriched: list[dict[str, str]] = []
    missing = 0
    for row in rows:
        task_id = row.get("task_id", "")
        fields = swepro_lookup.get(task_id, {})
        if not fields:
            missing += 1
        new_row = dict(row)
        new_row["swepro_problem_statement"] = fields.get("problem_statement", "")
        new_row["swepro_requirements"] = fields.get("requirements", "")
        new_row["swepro_interface"] = fields.get("interface", "")
        enriched.append(new_row)
    if missing:
        sys.stderr.write(
            f"WARNING: {missing} rows had no matching task_id in {args.dataset}; "
            "SWE-Pro columns blank for those.\n"
        )

    out_csv = Path(args.output_csv)
    keep_cols = [
        "task_id",
        "repo",
        "subject",
        "pr_url",
        "new_eval_verdict",
        "new_eval_reason",
        "new_instruction_md",
        "new_regen_count",
        "alignment_verdict",
        "alignment_reason",
        "swepro_problem_statement",
        "swepro_requirements",
        "swepro_interface",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keep_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)
    sys.stderr.write(f"Wrote {len(enriched)} rows to {out_csv}\n")

    html_text = _build_html(enriched, dataset_name=args.dataset)
    Path(args.output_html).write_text(html_text)
    n_ok = sum(1 for r in enriched if _classify_drop_stage(r) == "ok")
    sys.stderr.write(
        f"Wrote {len(html_text):,} bytes to {args.output_html} "
        f"({n_ok} of {len(enriched)} reached alignment-ok)\n"
    )

    md_text = _build_markdown(enriched, dataset_name=args.dataset)
    Path(args.output_md).write_text(md_text)
    sys.stderr.write(f"Wrote {len(md_text):,} bytes to {args.output_md}\n")


if __name__ == "__main__":
    main()
