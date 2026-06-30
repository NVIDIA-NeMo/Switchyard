---
name: publish-package
description: Build and publish Switchyard packages through the current OSS-style GitHub release path. Use when asked to publish, release, ship, tag, cut a version, build wheel artifacts, or prepare a package release.
---

# Publish Switchyard

Switchyard currently uses the OSS-style NeMo release shape in GitHub Actions. Keep temporary dev
wheels separate from official tag-gated release builds.

| Channel | Trigger | CI owner | Destination | Runbook |
|---|---|---|---|---|
| Dev wheel artifact | Manual `publish.yml` dispatch with `build_dev_artifact=true` | GitHub Actions | One-day GitHub artifact | `docs/internal/dev_wheel_artifacts.md` |
| Official release build | Root `v*` tag | GitHub Actions `.github/workflows/publish.yml` | Full release artifact matrix; publish job still disabled | `.github/workflows/publish.yml` |

GitHub-hosted runners cannot currently reach NVIDIA-internal Artifactory or Kitmaker Portal from
this repo, and GitHub Packages is not a PyPI-compatible package index. Do not add Artifactory,
Kitmaker, or Devzone upload calls back to the GitHub workflow unless the runner/network story
changes and the release process is explicitly approved.

## Guardrails

- Do not create tags unless the user explicitly asks for a tag-based release.
- Do not create GitHub Releases for dev wheel artifacts.
- Do not publish dev wheels to any package index from GitHub Actions.
- Keep `.dev` artifacts public-safe because GitHub Actions artifacts may be shared for review.
- Full wheel matrices belong only on root `v*` tag releases.
- Manual dev builds should build exactly one Linux x86_64 wheel artifact with one-day retention.
- Leave the public publish job disabled until the OSS release gate is approved.

## Dev Wheel Artifact Shape

Use the workflow-dispatch path in `.github/workflows/publish.yml`.

It stamps build-local package metadata as `nemo-switchyard`, builds one manylinux x86_64 abi3 wheel,
uploads it as `dev-wheel-linux-x86_64` with one-day retention, then downloads it again to verify the
wheel `Name` and `Version`.

| Input | Default | Meaning |
|---|---|---|
| `build_dev_artifact` | `false` | Set to `true` to build one temporary wheel artifact |
| `dev_version` | `0.0.1.dev0` | PEP 440 `.dev` version for wheel metadata |

## Required Secrets

The dev artifact build does not require release secrets.

The official publish job is currently disabled. When public publishing is approved, configure the
credentials required by the chosen PyPI publishing method before removing the hard-disabled guard.

## Local Preflight For Release-Infrastructure Changes

When editing release scripts, package metadata, package release docs, or this skill, run:

```bash
uv run ruff check scripts/release/set_dev_wheel_version.py tests/test_dev_wheel_versioning.py
uv run pytest tests/test_internal_release_versioning.py -v
uv run pytest tests/test_dev_wheel_versioning.py -v
python scripts/release/set_internal_version.py internal/v0.1.1-rc.1 --print-python-version
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --print-version
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
| GitHub artifact verifier cannot find wheels | Check the `dev-wheel-*` artifact names and retention |
| Artifact build creates a `switchyard` wheel | Confirm `scripts/release/set_dev_wheel_version.py` ran before `maturin build` |
| Full matrix runs on manual dispatch | Ensure release jobs are tag-gated with `github.event_name == 'push'` and `refs/tags/v` |
| Install imports checkout version | Verify from a temporary directory outside the repo |

## References

- Dev wheel artifact runbook: `docs/internal/dev_wheel_artifacts.md`
- Internal version helper: `scripts/release/set_internal_version.py`
- Dev wheel version helper: `scripts/release/set_dev_wheel_version.py`
- Public GitHub build workflow: `.github/workflows/publish.yml`
