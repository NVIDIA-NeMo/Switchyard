# NeMo Switchyard Fern docs

This directory contains the Fern configuration, navigation, redirect tooling, and CI-facing inputs
for the [NeMo Switchyard documentation](https://nemo-switchyard.docs.buildwithfern.com/nemo/switchyard/home).
Published MDX pages live one level up under `docs/`; they are not duplicated under
`docs/fern/versions/nightly/pages/`.

## Layout

```text
docs/
├── Makefile                         # pinned local check and preview commands
├── **/*.mdx                         # published documentation pages
├── internal/**/*.md                 # unpublished project notes
└── fern/
    ├── README.md                    # this guide
    ├── AGENTS.md                    # scoped agent instructions
    ├── fern.config.json             # Fern organization and CLI version pin
    ├── docs.yml                     # instance, theme, versions, and redirects
    ├── generate_legacy_redirect_site.py
    └── versions/nightly.yml         # public navigation and page paths
```

## Local development

Run the supported commands from the repository root:

```bash
cd docs && make check       # validate Fern configuration, navigation, links, and MDX
cd docs && make preview     # serve a local preview at http://localhost:3000
cd docs && make clean       # remove local Fern artifacts
```

The Makefile reads the CLI version from `fern.config.json` and runs it with `npx`, so a global Fern
installation is not required.

Run the route and redirect regression tests separately:

```bash
uv run pytest tests/test_fern_docs.py -v
```

## Navigation and routes

`versions/nightly.yml` defines the published page set and sidebar. Its page paths point back to MDX
files under `docs/` with `../../` paths. Navigation labels determine the public slugs; filenames do
not.

Keep sections label-only and put pages under `contents:`. Assigning an MDX file to a section and a
child page changes the route Fern publishes and can make the expected child URL return 404.

## Redirects

`docs.yml` owns redirects for requests that reach the Fern domain. Use complete
`/nemo/switchyard/...` source and destination paths and add exact mappings when legacy filenames or
section names do not match current slugs.

`generate_legacy_redirect_site.py` produces the redirect-only site served from the former GitHub
Pages domain. Keep its redirect map aligned with `docs.yml`; tests enforce the shared mappings.

## CI and publishing

| Workflow | Purpose |
|---|---|
| `.github/workflows/fern-docs-ci.yml` | Run `fern check` for docs changes |
| `.github/workflows/fern-docs-preview-build.yml` | Collect untrusted PR docs without secrets |
| `.github/workflows/fern-docs-preview-comment.yml` | Build the trusted hosted preview and upsert the PR comment |
| `.github/workflows/publish-fern-docs.yml` | Publish tagged/manual releases and update the redirect-only Pages site |

Hosted previews and production publishing require the `DOCS_FERN_TOKEN` organization secret. Do
not run a secret-bearing preview from a `pull_request` job, and do not publish production docs from
an ordinary merge to `main`.

For content authoring, start with [`../README.md`](../README.md). For scoped agent rules, see
[`AGENTS.md`](AGENTS.md) and the
[`switchyard-docs`](../../.agents/skills/switchyard-docs/SKILL.md) skill.
