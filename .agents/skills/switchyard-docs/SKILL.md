---
name: switchyard-docs
description: Use when adding or editing pages on the published Switchyard MkDocs site (`docs/`, `mkdocs.yml`, `.github/workflows/docs.yml`), wiring a new page into the nav, debugging a `mkdocs build --strict` failure, previewing the site locally, or reviewing changes to the docs CI workflow. Triggers on phrases like "add a docs page", "the docs site is broken", "mkdocs strict is failing", "preview docs locally", or any edit under `docs/**` that should reach the published Pages site.
---

# Switchyard Docs Site

## Overview

**The published site is a deliberately small subset of `docs/`.** Most files under `docs/` are internal design notes; only the pages listed in `mkdocs.yml`'s `nav:` (and not in `exclude_docs`) ship to the public site. When in doubt, prefer not to publish — add a file as internal first, then promote it once the content is stable.

**Strict mode is the contract.** `mkdocs build --strict` runs in CI (`.github/workflows/docs.yml`) and locally via `make publish`. Any warning — missing link target, unrecognized anchor, ambiguous file reference — fails the build. If you broke strict, fix the warning rather than relaxing the gate.

Repository files outside `docs_dir` are linked with paths relative to the Markdown source. The
`mkdocs_hooks.py` `on_page_markdown` hook rewrites valid in-repository targets to source URLs using
`repo_url` and `extra.source_ref` from `mkdocs.yml` before MkDocs validates links. The source ref
comes from `MKDOCS_SOURCE_REF`, with `main` as the local fallback. CI sets it to the PR head SHA or
push SHA so preview and published links resolve against the exact source commit. Missing targets and
paths that escape the repository remain relative so strict validation still exposes them.

The docs site uses the same `uv` toolchain as the rest of the repo. Dependencies live in the `docs` group in `pyproject.toml`; there is no separate `requirements*.txt`.

## Quick Reference

| Situation | Command |
|---|---|
| Sync the docs dependency group | `cd docs && make env`  (= `uv sync --only-group docs`) |
| Live-reload preview at http://127.0.0.1:8000 | `cd docs && make live` |
| One-shot strict build (mirrors CI) | `cd docs && make publish` |
| Plain incremental build | `cd docs && make html` |
| Remove `site/` | `cd docs && make clean` |
| List published pages | `grep -A20 "^nav:" mkdocs.yml` |
| List excluded `docs/*.md` | `grep -A10 "^exclude_docs:" mkdocs.yml` |

The MkDocs config and the workflow live at the repo root; `docs/Makefile` cd's there for you.

## Where Things Live

Before editing, discover the current state — don't memorize it:

- **Published page set** → `nav:` block in `mkdocs.yml`. The order there is the order on the site.
- **Hidden-from-publish files** → `exclude_docs:` block in `mkdocs.yml`. Anything under `docs/` that isn't in `nav:` and isn't excluded will trigger a strict-build warning.
- **Internal design notes** → also under `docs/`, but excluded. They live next to published pages so cross-linking from internal notes to public pages stays trivial.
- **Build/preview entry points** → `docs/Makefile` (thin wrapper over `uv run --only-group docs mkdocs ...`).
- **Docs dependencies** → `docs` group in `pyproject.toml`; resolved into `uv.lock` like the rest of the project.
- **Site styling** → `docs/stylesheets/` (extra CSS) and the `theme:` block in `mkdocs.yml`.
- **Repository source-link hook** → `mkdocs_hooks.py`, configured by `repo_url` and
  `extra.source_ref` in `mkdocs.yml`. `MKDOCS_SOURCE_REF` overrides the local `main` fallback.
- **CI** → `.github/workflows/docs.yml`. Triggered on `docs/**`, `mkdocs_hooks.py`, `mkdocs.yml`,
  `pyproject.toml`, `uv.lock`, and the workflow file itself.

Run `ls docs/` and `grep -A20 "^nav:" mkdocs.yml` to see the current set in one shot.

## Adding a Page

1. **Decide whether it's public or internal.** Internal design notes stay out of `nav` and go into `exclude_docs`. Don't ship anything that isn't stable enough to read cold.
2. **Place the file under `docs/`.** Use existing naming: lower_snake or UPPER_SNAKE — match the surrounding pages on the same topic.
3. **Wire it into `mkdocs.yml`** in two places:
   - Add the entry under `nav:` with a human-readable title.
   - Remove it from `exclude_docs` if a stale entry exists.
4. **Use relative links between published pages** (e.g. `[Architecture](architecture.md)`). MkDocs strict mode resolves these against `docs_dir`.
5. **Link to repository files outside `docs_dir` relative to the Markdown file.** For example,
   `docs/operations/example.md` links to `examples/config.yaml` as
   `../../examples/config.yaml`. The source-link hook turns an existing in-repository file into a
   `/blob/<source-ref>/...` URL and a directory into `/tree/<source-ref>/...` during the build.
6. **Run `make publish` locally** before pushing. CI runs `mkdocs build --strict`; reproduce that locally.
7. **Code samples in docs are tested by reading.** They are not executed by CI, so the burden is on the author. Verify imports resolve from `switchyard/__init__.py`'s `__all__` (anything else is an internal path that may move).
8. **Provider examples must match CLI resolution.** NVIDIA examples may use `NVIDIA_API_KEY` where the CLI supports that fallback. OpenRouter examples should use `https://openrouter.ai/api/v1` and pass `"$OPENROUTER_API_KEY"` via `--api-key`, or save it with `switchyard configure --provider openrouter`, unless the code being documented actually adds an `OPENROUTER_API_KEY` fallback.

## CI Hygiene

`.github/workflows/docs.yml` mirrors the conventions in `ci.yml` — match them when editing:

- **Path-filtered triggers** on `docs/**`, `mkdocs_hooks.py`, `mkdocs.yml`, `pyproject.toml`,
  `uv.lock`, and the workflow file. Keeps unrelated PRs out of the docs job graph.
- **Docs-only PRs skip full CI.** The lightweight change detector and required `CI Success` job
  still run, while published-site validation remains owned by this workflow and the focused README
  or getting-started workflows run when their source pages change.
- **Read-only default permissions** at the workflow level. Each job re-declares write scopes only when it needs them (`contents: write` for the Pages deploy, `pull-requests: write` for the preview comment).
- **`concurrency` group `${{ github.workflow }}-${{ github.ref }}`** with `cancel-in-progress: ${{ github.event_name == 'pull_request' }}`. Superseded PR runs are cancelled; main runs queue so the `gh-pages` writes don't race.
- **Workflow-level `MKDOCS_SOURCE_REF`** resolves from
  `${{ github.event.pull_request.head.sha || github.sha }}`. Pull requests use the immutable head
  commit; pushes use the pushed commit. Keep this environment override on both build event types.
- **`astral-sh/setup-uv@v6`** with `enable-cache: true` and `cache-dependency-glob: "uv.lock"`. Cache is keyed off the same lockfile the rest of the project uses.
- **`uv sync --only-group docs --locked`** installs just the docs group, deterministically.
- **`uv run --only-group docs mkdocs build --strict`** is the build step. The local equivalent is `make publish`.
- **Preview job gated to same-repo PRs.** `GITHUB_TOKEN` from a fork PR has no write scope on `gh-pages`, so the `if:` clause checks `head.repo.full_name == github.repository`. Don't switch to `pull_request_target` to "fix" this — that runs untrusted PR code with write secrets.
- **Single artifact handoff between jobs.** The `build` job uploads `site/`; `deploy` and `preview` download it. Don't duplicate the build inside either downstream job — it diverges from what was validated.

If the workflow needs a new job (e.g. link check, spell check), keep it inside this file and gate it on the same path filter so it doesn't fan out to unrelated PRs.

## Failure → Fix Map

| Symptom | Fix |
|---|---|
| `WARNING - A reference to 'X.md' is included in the 'nav' configuration ... is not found in the documentation files` | The nav points at a file that doesn't exist under `docs/`. Add the file or drop the nav entry. |
| `WARNING - Doc file 'X.md' contains a link 'Y.md', but the doc does not exist` | Bad relative link. Use a path relative to the Markdown file. Existing targets outside `docs_dir` are rewritten by `mkdocs_hooks.py`; missing and repository-escaping paths intentionally remain visible to strict validation. |
| `WARNING - Doc file 'X.md' is excluded from the build but its 'nav' entry references it` | The file is both in `nav:` and in `exclude_docs:`. Pick one. |
| `WARNING - The following pages exist in the docs directory, but are not included in the "nav" configuration` | Either add the page to `nav:` (publish it) or add it to `exclude_docs:` (hide it). The "do nothing" option doesn't exist under strict. |
| Repository links in a PR preview point at `main` | Confirm `MKDOCS_SOURCE_REF` is present in the build job and resolves to `github.event.pull_request.head.sha`; local builds intentionally fall back to `main`. |
| Preview job fails with `Permission to <repo>.git denied to github-actions[bot]` on a fork PR | Expected — the preview job is gated to same-repo PRs. If the gate is firing on a same-repo PR, check that `github.event.pull_request.head.repo.full_name` resolves correctly. |
| GitHub Pages site loses PR previews after a `main` deploy | Confirm `keep_files: true` on the `peaceiris/actions-gh-pages` step. The main deploy must not wipe `pr-preview/*`. |
| `make env` fails with `uv: command not found` | Install `uv` per the project Setup docs (see `AGENTS.md`). The Makefile shells out to `uv`; no separate Python toolchain is bootstrapped. |

## Anti-Patterns

- **Publishing internal design notes.** Anything under `docs/` that isn't a user-facing walkthrough or reference belongs in `exclude_docs`, not in `nav:`. Internal notes drift faster than the public surface and confuse users who land on them from search.
- **Hard-coded same-repository GitHub blob/tree URLs.** They pin the repository and source ref,
  which breaks forks and branch-specific builds. Use a path relative to the Markdown file and let
  `mkdocs_hooks.py` derive the source URL from `mkdocs.yml`.
- **Disabling `strict: true`** to make a CI failure go away. The warning is the bug. The two legitimate fixes are: fix the broken reference, or add the file to `exclude_docs`.
- **Importing internal symbols in published examples.** Code in published pages must import from `switchyard` (the public API), not from `switchyard.lib.*` or any other internal path. Anything outside `switchyard/__init__.__all__` can move without a deprecation.
- **Reintroducing a separate `requirements-mkdocs.txt`.** Docs deps belong in the `docs` group in `pyproject.toml` so `uv.lock` is the single source of pinned versions. Two pinning surfaces drift.
- **Switching the preview job to `pull_request_target`** to make fork previews work. That trades a missing preview for a real supply-chain risk. The current `if:` gate is the right answer.

## References

- `mkdocs.yml` — published-site config
- `mkdocs_hooks.py` — repository-relative source-link rewriting
- `docs/README.md` — local build instructions
- `docs/Makefile` — `env / html / publish / live / clean`
- `.github/workflows/docs.yml` — build + deploy + preview pipeline
- `pyproject.toml` — `[dependency-groups]` `docs = [...]`
- [`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md) — broader CI/validation guidance (ruff, mypy, pytest gates)
- [`switchyard-codebase-exploration`](../switchyard-codebase-exploration/SKILL.md) — load before editing docs that cite specific symbols, file paths, or APIs
