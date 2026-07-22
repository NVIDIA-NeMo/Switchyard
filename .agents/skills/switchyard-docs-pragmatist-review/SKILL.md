---
name: switchyard-docs-pragmatist-review
description: Review Switchyard documentation for copy-paste usability and scannability from a busy developer's perspective. Use to find incomplete commands, non-runnable examples, buried prerequisites, missing expected output, and avoidable task friction.
---

# Review Switchyard Documentation for Task Usability

Adapted from NVIDIA Tech Docs Skill Library's `doc-persona-pragmatist` workflow (Apache-2.0).

Read `AGENTS.md` and `docs/AGENTS.md` completely. Review without editing unless the user also asks
for fixes.

Assume an experienced engineer with a deadline who scans headings and examples before reading long
explanations.

## Review Lenses

1. **Copy-paste path:** Check whether commands and code blocks are complete in their stated context.
2. **Setup:** Find missing imports, variables, files, credentials, working-directory assumptions,
   and prerequisite commands.
3. **Scannability:** Confirm the primary command or example appears before optional detail and under
   a heading a task-oriented reader would recognize.
4. **Expected result:** Check whether the reader can tell that each key step succeeded.
5. **Failure path:** Check whether common errors have an immediate fix or a precise link.
6. **End-to-end completeness:** Prefer one complete runnable path over fragments that require the
   reader to assemble hidden context.

Do not demand that every conceptual page become a quickstart. Judge the page against its declared
purpose and audience.

## Findings

For each material issue, identify the location, explain the concrete failure or friction, and propose
the smallest fix. Separate confirmed runnability failures from untested concerns. End with an
estimated time-to-first-success assessment and say clearly when the page is ready for a busy reader.
