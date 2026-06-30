---
name: publish-package
description: Build and publish Switchyard packages through the correct public or Devzone release channel. Use when asked to publish, release, ship, tag, cut a version, or prepare a Kitmaker/Artifactory wheel release.
---

# Publish Switchyard

Switchyard has separate package release channels. Never mix their credentials or CI systems.

| Channel | Trigger | CI owner | Destination | Runbook |
|---|---|---|---|---|
| Public PyPI | Root `v*` tag or manual workflow | GitHub Actions `.github/workflows/publish.yml` | Public PyPI + GitHub Release after unblock | `.github/workflows/publish.yml` |
| Devzone prerelease | Manual workflow with `.dev` version | GitHub Actions + Kitmaker Portal | `pypi.nvidia.com` | `docs/internal/kitmaker_prerelease_wheels.md` |

For public-safe prerelease wheels on `pypi.nvidia.com`, read
`docs/internal/kitmaker_prerelease_wheels.md`. Kitmaker owns Devzone uploads; do not direct-publish
with `uv publish`.

## Guardrails

- Do not create tags unless the user explicitly asks for a tag-based release.
- Do not create GitHub Releases for Devzone prereleases.
- Do not use `uv publish` for `pypi.nvidia.com`; submit staged wheel URLs to Kitmaker.
- Keep `.dev` prereleases public-safe because `pypi.nvidia.com` is externally visible.
- Treat `upload: false` Kitmaker dry-runs as the default preflight.
- Require an explicit protected environment before any Kitmaker request uses `upload: true`.

## Devzone Prerelease Shape

Use the workflow-dispatch path in `.github/workflows/devzone-prerelease.yml`.

It builds the normal abi3 wheel matrix, stamps the wheel metadata as project
`nemo-switchyard`, uploads wheels to an Artifactory direct-download path, and optionally submits
those URLs to Kitmaker.

| Input | Default | Meaning |
|---|---|---|
| `version` | `0.0.1.dev0` | PEP 440 prerelease version for wheel metadata |
| `target_sha` | current workflow SHA | Commit to build |
| `kitmaker_dry_run` | `true` | Submit `upload: false` Kitmaker checks |
| `kitmaker_upload` | `false` | Submit `upload: true` after environment approval |

## Required Secrets

The Devzone workflow expects these GitHub secrets:

| Secret | Purpose |
|---|---|
| `ARTIFACTORY_URL` | Direct Artifactory base URL for staged wheel files |
| `ARTIFACTORY_USER` | Artifactory upload user |
| `ARTIFACTORY_TOKEN` | Artifactory upload token |
| `KITMAKER_API_TOKEN` | Kitmaker Portal API token |
| `KITMAKER_PROJECT_ID` | Portal project id for `nemo-switchyard` |
| `KITMAKER_PIC_EMAIL` | PIC email to include in Kitmaker release payload |

## Local Preflight For Release-Infrastructure Changes

When editing release scripts, package metadata, package release docs, or this skill, run:

```bash
uv run ruff check .
uv run pytest tests/test_internal_release_versioning.py -v
uv run pytest tests/test_devzone_prerelease_versioning.py -v
python scripts/release/set_internal_version.py internal/v0.1.1-rc.1 --print-python-version
python scripts/release/set_devzone_prerelease_version.py 0.0.1.dev0 --print-version
git diff --check
```

For docs changes under `docs/`, also run the strict MkDocs build when practical:

```bash
cd docs
make publish
```

## Failure Map

| Symptom | Fix |
|---|---|
| Artifactory upload gets 401/403 | Check `ARTIFACTORY_USER` and `ARTIFACTORY_TOKEN` |
| Kitmaker cannot fetch the wheel | Use a direct `/artifactory/...` URL, not a PyPI simple/API URL |
| Kitmaker rejects project name | Portal project name and wheel metadata must both be `nemo-switchyard` |
| Install imports checkout version | Verify from a temporary directory outside the repo |

## References

- Devzone prerelease runbook: `docs/internal/kitmaker_prerelease_wheels.md`
- Internal version helper: `scripts/release/set_internal_version.py`
- Devzone version helper: `scripts/release/set_devzone_prerelease_version.py`
- Public GitHub build workflow: `.github/workflows/publish.yml`
