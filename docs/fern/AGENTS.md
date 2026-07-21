# Fern infrastructure instructions

These instructions apply to `docs/fern/` in addition to `docs/AGENTS.md` and the repository root
`AGENTS.md`.

## Safety

- Read `.agents/skills/switchyard-docs/SKILL.md` before changing Fern configuration, navigation,
  redirects, or workflows.
- `cd docs && make check` and `cd docs && make preview` are local-only.
- A hosted Fern preview and production publication both change external state. Run either only when
  the user explicitly asks.
- Never expose `DOCS_FERN_TOKEN` to a `pull_request` job or check out untrusted PR code in the
  trusted `workflow_run` job.
- Derive the preview PR number from the trusted `workflow_run` payload and the Fern CLI version from
  the default branch. PR artifacts may provide docs content and changed-file hints only.
- Reject fork-originated runs before collecting or processing preview artifacts. Fork PRs receive
  required Fern validation but no secret-bearing hosted preview.
- Keep previews PR-numbered and cancel superseded runs. Serialize production publishing, and
  isolate repository write permission from Fern CLI execution.

## Site structure

- Keep published nightly MDX at `docs/**/*.mdx`, outside `docs/fern/`.
- Do not create `docs/fern/versions/nightly/pages/`.
- Keep `versions/nightly.yml` paths relative to that file; current authored pages use `../../`.
- Keep Switchyard navigation sections label-only so pages resolve as `/section/page`.
- In `docs.yml`, use full `/nemo/switchyard/...` paths for redirect sources and destinations.

## Redirect ownership

- Fern redirects in `docs.yml` handle URLs that already reach the Fern domain.
- `generate_legacy_redirect_site.py` handles the former GitHub Pages domain.
- When a legacy route changes, update both redirect owners and the regression expectations in
  `tests/test_fern_docs.py` when needed.
- Do not replace the old Pages site until the publishing workflow has verified the Fern custom
  domain home page.

## Validation

After changing anything in this directory, run:

```bash
cd docs && make check
uv run pytest tests/test_fern_docs.py -v
```

Before pushing, follow the root `AGENTS.md` requirements. Update
`.agents/skills/switchyard-docs/SKILL.md` whenever these workflows or invariants change.
