# Switchyard Documentation

These instructions apply to all files under `docs/`. Read the repository-level `AGENTS.md` first,
then load the documentation workflow skill that matches the task.

## Published Site Contract

The published site is a deliberately small subset of `docs/`. Pages listed under `nav:` in
`mkdocs.yml` are public. Internal design notes belong under an `exclude_docs` pattern. Do not leave
a page in neither set: strict MkDocs treats that as a warning and fails the build.

`mkdocs build --strict` is the contract. Fix warnings instead of weakening strict mode.

Repository files outside `docs_dir` must use paths relative to the Markdown source. The
`mkdocs_hooks.py` hook rewrites valid in-repository targets to source URLs using `repo_url` and
`extra.source_ref`. `MKDOCS_SOURCE_REF` selects the source revision; local builds default to `main`.

The docs dependencies live in the `docs` dependency group in `pyproject.toml`. Do not add a
separate requirements file.

## Commands

| Situation | Command |
|---|---|
| Sync docs dependencies | `cd docs && make env` |
| Strict build matching CI | `cd docs && make publish` |
| Incremental build | `cd docs && make html` |
| Live preview | `cd docs && make live` |
| Remove generated site | `cd docs && make clean` |

## Adding or Publishing a Page

1. Decide whether the page is public or internal.
2. Put the file under `docs/` using the naming pattern of neighboring pages.
3. Add a public page to `nav:` in `mkdocs.yml`, or cover an internal page with `exclude_docs`.
4. Use relative links between documentation pages.
5. Link to repository files outside `docs/` relative to the Markdown file; let
   `mkdocs_hooks.py` create the source URL.
6. Run `cd docs && make publish` before handing off the change.

## Accuracy Rules

- Ground behavioral claims in source, configuration, schemas, tests, or command output from the
  current checkout. Omit or flag claims that cannot be verified.
- Published Python examples import from `switchyard`, not `switchyard.lib.*`. Confirm every public
  symbol appears in `switchyard/__init__.py`'s `__all__`.
- Verify CLI flags, defaults, nesting, and deprecations against the parser before publishing them.
- Verify configuration and payload examples against their owning model, loader, translator, or
  endpoint tests.
- NVIDIA examples may use `NVIDIA_API_KEY` where the CLI supports that fallback. OpenRouter
  examples use `https://openrouter.ai/api/v1` and pass `"$OPENROUTER_API_KEY"` through `--api-key`,
  or save it with `switchyard configure --provider openrouter`, unless source proves a different
  resolution path.
- Keep evidence in the handoff or review report, not as `file:line` annotations in published prose.

## CI and Security Boundaries

- Keep docs CI path-filtered to documentation inputs.
- Preserve read-only workflow permissions except where a job explicitly needs write access.
- Keep PR previews limited to same-repository PRs. Do not use `pull_request_target` to run
  untrusted documentation changes with write credentials.
- Preserve `MKDOCS_SOURCE_REF` so previews link to the reviewed commit.
- Build the site once and hand the same artifact to preview and deployment jobs.

## Anti-Patterns

- Publishing internal design notes as user documentation.
- Hard-coding same-repository GitHub blob or tree URLs.
- Disabling strict mode to hide a warning.
- Publishing examples that import internal modules or contain unverified flags and defaults.
- Editing generated site output or a generated docs environment.
- Adding a second docs dependency file outside `pyproject.toml` and `uv.lock`.
