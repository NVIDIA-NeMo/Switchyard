---
name: switchyard-docs-draft
description: Draft net-new Switchyard documentation from repository evidence. Use when creating a new guide, tutorial, explanation, or reference page under docs/; not for polishing an existing page or reviewing examples only.
---

# Draft Switchyard Documentation

Adapted from NVIDIA Tech Docs Skill Library's `doc-draft` workflow (Apache-2.0).

Read `AGENTS.md` and `docs/AGENTS.md` completely before drafting.

## Workflow

1. Restate the requested document and identify its likely audience and Diataxis type: tutorial,
   how-to, explanation, or reference.
2. Inspect the current documentation tree and `mkdocs.yml`. Decide whether to update an existing
   page or create a public or internal page.
3. Enumerate evidence sources before writing:
   - public source and configuration;
   - CLI definitions or generated help;
   - request and response schemas;
   - tests and examples;
   - related documentation.
4. Build a claim list. Include only claims supported by the inspected artifacts. Keep uncertain
   claims out of published text and report them as maintainer questions.
5. Draft the shortest document that lets the target reader complete or understand the task.
6. Derive code examples from verified APIs and behavior. An example may compose public APIs into a
   new runnable snippet, but every symbol, argument, flag, field, and default must be verified.
7. Integrate the page into `mkdocs.yml` as required by `docs/AGENTS.md`.
8. Run the strict docs build and report the evidence used, unresolved questions, and validation.

## Drafting Rules

- Front-load the outcome and prerequisites.
- Keep tutorials sequential, how-to guides task-oriented, explanations conceptual, and references
  exhaustive within their declared scope.
- Prefer one complete example over several fragments.
- Label intentionally abbreviated examples or pseudocode.
- Do not publish speculative recommendations, performance claims, or unverified compatibility.
- Do not put source `file:line` annotations into reader-facing prose.

## Handoff

Report the created or updated paths, document type, principal evidence sources, unresolved questions,
and exact validation commands. Do not claim a live provider example was tested unless it was.
