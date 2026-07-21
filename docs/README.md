# NeMo Switchyard docs

Switchyard documentation is authored as [Fern](https://buildwithfern.com/) MDX in this directory.
The current Fern site is available at
**[nemo-switchyard.docs.buildwithfern.com/nemo/switchyard](https://nemo-switchyard.docs.buildwithfern.com/nemo/switchyard/home)**.

## Where to make changes

| Change | Location |
|---|---|
| Published page | `docs/**/*.mdx` |
| Public navigation | [`fern/versions/nightly.yml`](fern/versions/nightly.yml) |
| Site settings and redirects | [`fern/docs.yml`](fern/docs.yml) |
| Internal, unpublished notes | `docs/internal/**/*.md` |
| Fern build and publishing details | [`fern/README.md`](fern/README.md) |

Markdown files under `internal/` are intentionally excluded from the public navigation. Promote one
to MDX only when it is ready to become public documentation.

## Common commands

Run these from the repository root:

```bash
cd docs && make check
uv run pytest tests/test_fern_docs.py -v
cd docs && make preview
```

`make check` uses the Fern CLI version pinned in
[`fern/fern.config.json`](fern/fern.config.json). `make preview` starts a local server and does not
publish the site.

## Add or edit a page

1. Add or update the MDX page under `docs/`.
2. Add new pages to [`fern/versions/nightly.yml`](fern/versions/nightly.yml) in the intended order.
3. Use canonical Fern routes derived from the navigation labels for internal links.
4. Run the Fern check and route regression tests above.

Every pull request runs Fern validation inside the required `CI Success` aggregate. Docs changes
from repository branches also start a PR-numbered hosted preview: the collector uploads content
without secrets, while the trusted workflow derives PR identity and the Fern CLI version from
GitHub/default-branch state before publishing. Preview generation and PR commenting use separate
permission scopes. Fork PRs receive Fern validation but do not publish hosted previews.

Legacy GitHub Pages paths are handled in two places: Fern redirects live in
[`fern/docs.yml`](fern/docs.yml), while
[`fern/generate_legacy_redirect_site.py`](fern/generate_legacy_redirect_site.py) builds the static
redirect-only GitHub Pages site. Keep both mappings synchronized.

For agent-specific editing rules, see [`AGENTS.md`](AGENTS.md) and the
[`switchyard-docs`](../.agents/skills/switchyard-docs/SKILL.md) skill.
