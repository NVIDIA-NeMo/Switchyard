# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for gold-plan synthesis.

Lifted from craft-iterative-planning/scraper/synthesize_parallel.py, which
produced the v1 gold-plan corpus (309/309 tasks synthesized successfully).
"""

from __future__ import annotations

DIFF_CHARS = 15000


_PROMPT_TEMPLATE = """You are generating a gold reference implementation plan for a coding agent benchmark.

Given the full PR data below (description, issues, files, and diff), write the \
implementation plan that would produce these changes. Write it as the plan a senior \
engineer would produce after thoroughly investigating the codebase. Be specific about \
which files to modify or create, what to change in each, and why.

## PR Data"""


def build_gold_plan_prompt(ctx: dict) -> str:
    """Return the full prompt. ``ctx`` provides repo, title, category, body,
    issue_context, source_files_modified, source_files_added, test_files, diff.
    """
    return f"""{_PROMPT_TEMPLATE}

**Repo:** {ctx["repo"]}
**Title:** {ctx["title"]}
**Category:** {ctx["category"]}

**PR Description:**
{ctx["body"]}

**Linked Issues:**
{ctx["issue_context"] or "None"}

**Source files modified:** {", ".join(ctx["source_files_modified"])}
**Source files added:** {", ".join(ctx["source_files_added"]) or "None"}
**Test files:** {", ".join(ctx["test_files"])}

**Diff:**
```
{ctx["diff"][:DIFF_CHARS]}
```

## Output Format

Return valid JSON with exactly one field:
{{
  "gold_plan": "The detailed implementation plan. Be specific about files, functions, and changes."
}}"""
