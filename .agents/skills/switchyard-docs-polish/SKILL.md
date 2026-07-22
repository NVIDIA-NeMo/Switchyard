---
name: switchyard-docs-polish
description: Improve an existing Switchyard documentation page without changing its technical scope. Use when asked to polish, tighten, clarify, reorganize, or improve the readability of established Markdown content under docs/.
---

# Polish Switchyard Documentation

Adapted from NVIDIA Tech Docs Skill Library's `doc-polish` workflow (Apache-2.0).

Read `AGENTS.md` and `docs/AGENTS.md` completely before editing. Use this workflow only when the
page's foundation is sound; route missing or questionable technical content through
`switchyard-docs-example-audit` or a source-backed drafting pass first.

## Workflow

1. Identify the page's audience, purpose, and Diataxis type.
2. Preserve its technical scope and collect a small before-state assessment:
   - Is the outcome visible early?
   - Is the page scannable?
   - Are prerequisites and next steps easy to find?
   - Are examples complete enough for the intended audience?
3. Improve signal-to-noise. Remove repetition, filler, throat-clearing, and headings that do not
   help navigation. Replace vague claims with concrete language without inventing facts.
4. Align the structure with the page type. Do not turn a reference into a tutorial or mix a long
   conceptual explanation into a task procedure.
5. Apply progressive disclosure: essential path first, optional detail later, advanced material
   behind a clear heading or supported disclosure element.
6. Improve headings, paragraphs, lists, terminology, and links. Use MkDocs features already enabled
   in `mkdocs.yml`; do not introduce syntax from another documentation framework.
7. Review the diff and remove changes that are merely stylistic churn.
8. Run the strict docs build.

## Boundaries

- Do not alter documented behavior unless source evidence proves the old text wrong.
- Do not add features, examples, recommendations, or promises to make a page feel complete.
- Do not replace precise technical language with marketing language.
- Do not restructure neighboring pages unless the user asked for broader information architecture.

## Handoff

Summarize the material improvements, call out any accuracy questions left untouched, and report the
exact validation performed.
