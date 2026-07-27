---
name: switchyard-docs
description: Edit or debug the published Switchyard MkDocs site. Use for docs pages, mkdocs.yml navigation, mkdocs_hooks.py source links, strict build failures, local previews, or .github/workflows/docs.yml.
---

# Switchyard Documentation

The published site is the subset of `docs/` selected by `mkdocs.yml`. Strict MkDocs warnings fail
CI, so fix warnings rather than weakening validation.

## Source Of Truth

| Concern | File |
|---|---|
| Navigation, exclusions, theme | `mkdocs.yml` |
| Source-link rewriting | `mkdocs_hooks.py` |
| Local commands | `docs/Makefile` |
| Build, preview, deployment | `.github/workflows/docs.yml` |
| Documentation dependencies | `pyproject.toml` `docs` group |

## Workflow

1. Decide whether a new page is public. Public pages go in `nav`; internal notes go in
   `exclude_docs`.
2. Match the surrounding file naming and documentation style.
3. Use relative links. For repository files outside `docs/`, use paths relative to the Markdown
   file and let `mkdocs_hooks.py` produce the source URL.
4. Verify public examples use supported public imports and current CLI syntax.
5. Run the strict build:

```bash
cd docs
make publish
```

## CI Constraints

- Keep the docs workflow path-filtered.
- Keep default permissions read-only and grant write access only to deployment jobs.
- Keep PR previews limited to same-repository pull requests.
- Do not use `pull_request_target` to run untrusted documentation code with write permissions.
- Build once and pass the `site/` artifact to preview and deployment jobs.
- Preserve `keep_files: true` so a main deployment does not remove PR previews.

Common strict-build fixes are direct: add or remove a missing nav entry, repair the relative link,
or explicitly exclude an internal page. Do not disable strict mode.
