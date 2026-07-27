---
name: publish-python-release
description: Build or publish nemo-switchyard Python distributions through the repository's GitHub Actions workflow. Use when asked to build a development wheel, cut a Python release, publish to PyPI, create a release tag, or debug .github/workflows/publish.yml.
---

# Publish The Python Package

Use `.github/workflows/publish.yml` as the source of truth. The supporting runbook is
`docs/internal/release_workflow.md`.

## Release Paths

| Intent | Trigger | Result |
|---|---|---|
| One temporary wheel | Manual dispatch with `build_dev_artifact=true` | Linux x86_64 GitHub artifact, retained for one day |
| Full matrix test | Manual dispatch with `build_dev_matrix=true` | sdist and wheel artifacts; no publish |
| Official release | Push `vMAJOR.MINOR.PATCH` matching `pyproject.toml` | Full matrix, PyPI publish, GitHub Release |

Manual builds are artifact-only. Official releases use PyPI Trusted Publishing through the
GitHub environment named `pypi`; do not add a long-lived PyPI token.

## Workflow

1. Read the current workflow and `pyproject.toml`; do not rely on remembered inputs or versions.
2. Confirm the requested version and whether the user explicitly authorized creating a tag.
3. For workflow changes, run the focused release-helper tests and inspect the workflow diff.
4. Dispatch a development build or push the authorized release tag.
5. Monitor every matrix job and verify the resulting package from a clean temporary environment.

Example development dispatch:

```bash
gh workflow run publish.yml \
  -f build_dev_artifact=true \
  -f build_dev_matrix=false \
  -f dev_version=0.0.1.dev0
```

## Guardrails

- Never create or push a tag unless the user explicitly requested it.
- Never publish a manual development build to PyPI.
- Never reuse a deleted PyPI filename or version; PyPI permanently reserves filenames.
- Keep release versions consistent across the tag, `pyproject.toml`, and package metadata.
- Verify installation outside the checkout so local imports cannot mask packaging defects.
- Cargo crate publishing is a separate manual workflow. Inspect each crate's `Cargo.toml`,
  dependency publication order, and crates.io ownership before running `cargo publish`.

## Release-Infrastructure Validation

```bash
uv run ruff check scripts/release/set_dev_wheel_version.py tests/test_dev_wheel_versioning.py
uv run pytest tests/test_dev_wheel_versioning.py -v
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --print-version
git diff --check
```
