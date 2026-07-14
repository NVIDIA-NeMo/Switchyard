# Documentation instructions

These instructions apply to everything under `docs/`. More specific rules in
`docs/fern/AGENTS.md` also apply inside `docs/fern/`.

## Before editing

- Read `.agents/skills/switchyard-docs/SKILL.md` completely.
- Read `.agents/skills/switchyard-codebase-exploration/SKILL.md` when documentation cites code,
  commands, configuration, or public APIs.
- Treat `docs/fern/versions/nightly.yml` as the source of truth for the published page set, order,
  labels, and routes.

## Content boundaries

- Published pages are `docs/**/*.mdx`, excluding `docs/fern/`.
- `docs/internal/**/*.md` files are unpublished. Do not add them to Fern navigation or link to them
  from published pages unless the task explicitly promotes that content.
- Keep contributor guidance in `README.md` and `AGENTS.md`; do not add either file to Fern
  navigation.
- Use public `switchyard` imports in examples, not `switchyard.lib.*` internals.
- Link repository files outside `docs/` with absolute GitHub URLs.

## Fern authoring

- Give every published page non-empty `title` and `description` frontmatter.
- Keep navigation sections label-only and place authored pages under `contents:`. Do not assign the
  same MDX page to both a section `path:` and a child page.
- Use canonical, version-agnostic Fern routes derived from navigation labels for internal links.
- Do not link unpublished internal notes from published MDX.
- Use Fern callout components instead of GitHub admonition syntax.

## Validation

Run the smallest relevant checks after every docs change:

```bash
cd docs && make check
uv run pytest tests/test_fern_docs.py -v
```

Before pushing, also follow the repository-wide validation requirements in the root `AGENTS.md`.
If a docs workflow, route invariant, or ownership boundary changes, update
`.agents/skills/switchyard-docs/SKILL.md` in the same change.
