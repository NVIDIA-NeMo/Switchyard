---
name: switchyard-docs
description: Use when adding or editing Switchyard docs or contributor guidance under `docs/`, changing Fern configuration or navigation, debugging `fern check`, previewing or publishing docs, or reviewing the Fern GitHub Actions workflows.
---

# Switchyard Fern Docs

## Overview

Published documentation is authored as MDX at the top level of `docs/`. Fern infrastructure lives
under `docs/fern/`; do not put nightly content under `docs/fern/versions/nightly/pages/`.

`docs/README.md` is the contributor entry point. Scoped instructions live in `docs/AGENTS.md` and
`docs/fern/AGENTS.md`; neither guidance file is published. Keep those files synchronized with this
skill when the authoring, validation, redirect, or publishing workflow changes.

The public page set, order, and labels come from `docs/fern/versions/nightly.yml`. Markdown files
under `docs/internal/` are intentionally unpublished design and operations notes. Do not add them to
the Fern navigation unless the content is deliberately being promoted to public documentation.

`fern check` is the contract. Run it locally through `docs/Makefile`; CI runs the same CLI version
pinned in `docs/fern/fern.config.json`.

## Quick Reference

| Situation | Command |
|---|---|
| Validate the site | `cd docs && make check` |
| Test routes and redirects | `uv run pytest tests/test_fern_docs.py -v` |
| Run a local preview | `cd docs && make preview` |
| Remove local Fern artifacts | `cd docs && make clean` |
| List published pages | `sed -n '1,240p' docs/fern/versions/nightly.yml` |
| Find unpublished notes | `find docs/internal -type f -name '*.md' -print` |
| Check MDX safety | `cd docs && make check` |
| Check internal Fern links | `uv run pytest tests/test_fern_docs.py::test_internal_mdx_links_resolve_to_navigation_routes -v` |

## Where Things Live

- **Published content** → `docs/**/*.mdx`, excluding `docs/fern/`.
- **Unpublished notes** → `docs/internal/**/*.md`.
- **Docs contributor guide and scoped instructions** → `docs/README.md` and `docs/AGENTS.md`.
- **Fern infrastructure guide and scoped instructions** → `docs/fern/README.md` and
  `docs/fern/AGENTS.md`.
- **Site and product configuration** → `docs/fern/docs.yml`.
- **Nightly navigation** → `docs/fern/versions/nightly.yml`.
- **Fern CLI pin** → `docs/fern/fern.config.json`.
- **Local commands** → `docs/Makefile`.
- **CI validation** → `.github/workflows/fern-docs-ci.yml`.
- **PR preview collection** → `.github/workflows/fern-docs-preview-build.yml`.
- **PR preview publishing and comment** → `.github/workflows/fern-docs-preview-comment.yml`.
- **Release publishing** → `.github/workflows/publish-fern-docs.yml`.
- **GitHub Pages redirect generator** → `docs/fern/generate_legacy_redirect_site.py`.
- **Route/redirect regression tests** → `tests/test_fern_docs.py`.
- **Shared NVIDIA presentation** → `global-theme: nvidia`; theme assets are not vendored here.

## Adding a Published Page

1. Create `docs/<section>/<page>.mdx` with concise frontmatter, including `title` and `position`.
2. Add it to `docs/fern/versions/nightly.yml` in the intended sidebar location. Preserve existing
   labels and order unless the change explicitly redesigns navigation.
3. Point `path:` from the version file back to the authored page with `../../`, for example:

   ```yaml
   - page: Health-aware Routing
     path: ../../routing_algorithms/health_aware_routing.mdx
   ```

4. Use version-agnostic Fern URLs for published-page links. Fern URLs derive from the slugified
   `page:` and `section:` labels, not filenames. For example, `page: Core Concepts` is
   `/concepts/core-concepts` inside every version.
5. Link repository files outside `docs/` with an absolute GitHub URL. They are not part of Fern's
   content tree.
6. Import examples from `switchyard`, not `switchyard.lib.*`; public examples must use exports from
   `switchyard/__init__.py`.
7. Run `cd docs && make check` before pushing.

## Editing Navigation

- Treat `docs/fern/versions/nightly.yml` as the source of truth for public scope and order.
- Use `section:` for labeled groups with nested `contents:`. Do not combine `folder:` with
  `contents:`; Fern rejects that schema.
- Keep Switchyard sections label-only. Do not add a `path:` to a section and then repeat that same
  file as a child page; Fern publishes the section landing route instead of the intended child
  route. Every authored page in this site belongs under `contents:` and resolves as
  `/section/page`.
- Keep the first navigation entry as the home page.
- Paths in nightly navigation point to `../../*.mdx`. Frozen releases, when added, own independent
  copies under `docs/fern/versions/<version>/pages/`.
- Do not add `latest.yml` as a copy of nightly. A future `latest` alias must point at a frozen GA
  tree.

## Links and MDX

- MDX treats bare angle-bracket text as JSX. Escape comparisons such as `<100ms` as `&lt;100ms`
  and wrap placeholders such as `<MODEL>` in backticks.
- HTML void tags must self-close, for example `<img src="..." />`.
- Use MDX comments (`{/* note */}`), not HTML comments.
- Built-in Fern components do not require imports. Add product-specific components only when a page
  needs them, then register their directory in `docs/fern/docs.yml`.
- Do not link unpublished `docs/internal/` notes from published MDX.
- Preserve titled MkDocs admonitions with the Fern callout `title` prop, for example
  `<Note title="Deployment boundary">`.
- Give every published page a non-empty `description` in frontmatter.

## Redirects

- Use full `/nemo/switchyard/...` paths in both `source` and `destination`.
- Add exact mappings when old filenames and new navigation slugs differ. A generic `index.html`
  catch-all cannot translate underscores or changed section hierarchy.
- Cover both the directory route and its generated `index.html` form for every former MkDocs page.
- Keep `docs/fern/generate_legacy_redirect_site.py` synchronized with `docs/fern/docs.yml`. The
  generator owns redirects on `nvidia-nemo.github.io/Switchyard`; Fern redirects only own requests
  that already reach the Fern domain.
- Run `uv run pytest tests/test_fern_docs.py -v` after changing navigation, links, or redirects.

## CI Hygiene

The Fern workflows follow the same split as NeMo Curator:

- `fern-docs-ci.yml` is a reusable `fern check` workflow called by the repository-wide CI. Its job
  is included in the required `CI Success` aggregate, so Fern failures cannot merge as an optional
  check.
- `fern-docs-preview-build.yml` runs only for same-repository PR branches, uses no secrets, and
  uploads the complete `docs/` tree plus a changed-MDX hint. Fork PRs stop after required Fern
  validation. The complete tree is required because nightly navigation under `docs/fern/`
  references MDX pages through `../../` paths. It uses PR-numbered concurrency and cancels
  superseded runs.
- `fern-docs-preview-comment.yml` runs through `workflow_run`, downloads the collector artifact,
  rejects fork-originated runs, derives the target PR from the trusted event, reads the Fern version
  from a separate default-branch checkout, and generates a PR-numbered preview with
  `DOCS_FERN_TOKEN`. Commenting runs in a separate job with only pull-request write permission. It
  must never check out the PR branch, run PR-provided scripts, or trust artifact metadata for the
  target PR/tool version.
- `publish-fern-docs.yml` publishes only for `docs/v*` tags or manual dispatch, then replaces the
  old GitHub Pages content with static redirects to the custom Fern domain. The workflow verifies
  the custom-domain home page first; if it remains unavailable, GitHub Pages stays untouched.
  Publishing is serialized, and the repository-write redirect job never receives the Fern token.
  A merge to `main` validates the site but does not publish either destination.

Every workflow installs the exact Fern CLI version from `docs/fern/fern.config.json` and runs Fern
from `docs/fern/`. The privileged preview reads that pin from the trusted default branch rather
than the artifact. Preview generation and publishing require the `DOCS_FERN_TOKEN` organization
secret. All Fern workflow actions use immutable commit pins and supported Node LTS; secret-bearing
jobs disable package-manager caching and validate the Fern version before installing it. Checkout
steps do not persist credentials. Do not collapse the collector and trusted preview phases or use
`pull_request_target`.

## Failure Map

| Symptom | Fix |
|---|---|
| `Path does not exist` | Correct the `../../` path in `versions/nightly.yml` or restore the referenced MDX file. |
| Navigation object does not match schema | Use `section` with `contents`, or a standalone `folder`; do not combine `folder` and `contents`. |
| Unsupported JSX tag | Replace it with a built-in Fern component or register the product component directory. |
| MDX parse error near `<` | Escape prose comparisons/placeholders or self-close the HTML element. |
| Link works on GitHub but not Fern | Rewrite it to the canonical Fern URL or an absolute GitHub URL for repository-only files. |
| Child page returns 404 but section works | Remove the duplicate section `path`; keep the authored page only under `contents`. |
| Legacy redirect lands on another 404 | Add an exact old-path → current-slug mapping before broad redirect patterns. |
| `global-theme` authentication failure | Verify NVIDIA Fern authentication; do not vendor a local copy of the shared theme. |

## Anti-Patterns

- Adding unpublished design notes to navigation because they happen to live under `docs/`.
- Adding `README.md` or `AGENTS.md` contributor guidance to published navigation.
- Recreating `docs/fern/versions/nightly/pages/`; nightly content is authored at `docs/*.mdx`.
- Deriving links from filenames instead of the navigation title/slug.
- Assigning the same MDX file to a section `path` and one of its child pages.
- Hardcoding `/nightly/` in shared page links.
- Vendoring NVIDIA theme CSS, logos, favicon, or footer components.
- Reintroducing MkDocs configuration, hooks, or a parallel docs build.
- Running a secret-bearing Fern preview directly from a `pull_request` job.
- Publishing production docs on every merge to `main` instead of a docs tag or manual dispatch.

## References

- `docs/fern/docs.yml` — Fern instance, theme, versions, and redirects.
- `docs/README.md` — contributor entry point for docs content.
- `docs/AGENTS.md` — scoped instructions for the full docs tree.
- `docs/fern/README.md` — Fern infrastructure and CI guide.
- `docs/fern/AGENTS.md` — scoped instructions for Fern configuration and redirects.
- `docs/fern/versions/nightly.yml` — public navigation and authored-page paths.
- `docs/fern/fern.config.json` — Fern CLI pin.
- `docs/Makefile` — local check and preview commands.
- `docs/fern/generate_legacy_redirect_site.py` — redirect-only GitHub Pages output.
- `tests/test_fern_docs.py` — navigation, link, metadata, and redirect invariants.
- `.github/workflows/fern-docs-ci.yml` — CI-equivalent validation.
- `.github/workflows/fern-docs-preview-build.yml` — same-repository preview source collection.
- `.github/workflows/fern-docs-preview-comment.yml` — trusted preview generation and PR comment.
- `.github/workflows/publish-fern-docs.yml` — tag/manual production publishing.
- [`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md) — broader repository validation.
- [`switchyard-codebase-exploration`](../switchyard-codebase-exploration/SKILL.md) — impact mapping for docs that cite code or APIs.
