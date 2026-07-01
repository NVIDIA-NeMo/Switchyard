---
name: publish-package
description: Build and publish Switchyard packages through the current OSS-style GitHub release path. Use when asked to publish, release, ship, tag, cut a version, build wheel artifacts, or prepare a package release.
---

# Publish Switchyard

Switchyard currently uses the OSS-style NeMo release shape in GitHub Actions. Keep temporary dev
wheels separate from official tag-gated release builds.

| Channel | Trigger | CI owner | Destination | Runbook |
|---|---|---|---|---|
| Dev wheel artifact | Manual `publish.yml` dispatch with `build_dev_artifact=true` | GitHub Actions | One-day GitHub artifact | `docs/internal/release_workflow.md` |
| Dev matrix artifact | Manual `publish.yml` dispatch with `build_dev_matrix=true` | GitHub Actions | Full matrix GitHub artifacts | `docs/internal/release_workflow.md` |
| Rust crate package dry-run | Manual `publish.yml` dispatch with `package_rust_crates=true` | GitHub Actions | `.crate` artifacts only | `docs/internal/release_workflow.md` |
| First Rust crate publish | Manual `publish.yml` dispatch from `main` with `publish_rust_crates=true` | GitHub Actions | crates.io via `cargo publish` | `docs/internal/release_workflow.md` |
| Official release build | Root `vMAJOR.MINOR.PATCH` tag | GitHub Actions `.github/workflows/publish.yml` | Full release artifact matrix + crates.io + PyPI Trusted Publishing via `uv publish` | `.github/workflows/publish.yml` |

Release publishing stays on the public GitHub/PyPI/crates.io path. Manual branch builds only
produce artifacts; root release tags publish through PyPI Trusted Publishing and crates.io.

## Guardrails

- Do not create tags unless the user explicitly asks for a tag-based release.
- Do not create GitHub Releases for dev wheel artifacts.
- Do not publish dev wheels to PyPI from manual workflow dispatch.
- Do not publish Rust crates from branches. Manual Rust crate publishing is allowed only from
  `main` and must use the `crates-io` GitHub environment.
- Keep `.dev` artifacts public-safe because GitHub Actions artifacts may be shared for review.
- Full wheel matrices may run manually for validation, but PyPI publishing belongs only on root
  `vMAJOR.MINOR.PATCH` tag releases.
- Manual dev builds should build exactly one Linux x86_64 wheel artifact with one-day retention.
- Use PyPI Trusted Publishing with the GitHub environment named `pypi`; do not add long-lived PyPI tokens.
- Use a crates.io API token only in the GitHub environment named `crates-io`, stored as
  `CARGO_REGISTRY_TOKEN`.

## Dev Wheel Artifact Shape

Use the workflow-dispatch path in `.github/workflows/publish.yml`.

It stamps a build-local `.dev` version, builds one manylinux x86_64 abi3 wheel, uploads it as
`dev-wheel-linux-x86_64` with one-day retention, then downloads it again to verify the wheel `Name`
and `Version`.

| Input | Default | Meaning |
|---|---|---|
| `build_dev_artifact` | `false` | Set to `true` to build one temporary wheel artifact |
| `build_dev_matrix` | `false` | Set to `true` to build the complete sdist and wheel matrix as artifacts |
| `package_rust_crates` | `false` | Set to `true` to run `cargo publish --dry-run` and upload `.crate` artifacts |
| `publish_rust_crates` | `false` | Set to `true` from `main` to publish the first Rust crates after approval |
| `dev_version` | `0.0.1.dev0` | PEP 440 `.dev` version for wheel metadata |

## Required Secrets

The artifact-only dev build does not require release secrets.

The official publish job uses `uv publish --trusted-publishing always`; no PyPI token is required.
Manual dev builds never publish to PyPI.

Rust crate publishing uses `cargo publish`, so the `crates-io` GitHub environment must define
`CARGO_REGISTRY_TOKEN`. `switchyard-py` is intentionally not published to crates.io because it is
distributed inside the Python wheel.

Before cutting the first tag, create the pending PyPI trusted publisher for:

| Field | Value |
|---|---|
| Project | `nemo-switchyard` |
| Owner | `NVIDIA-NeMo` |
| Repository | `Switchyard` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

## Local Preflight For Release-Infrastructure Changes

When editing release scripts, package metadata, package release docs, or this skill, run:

```bash
uv run ruff check scripts/release/set_dev_wheel_version.py tests/test_dev_wheel_versioning.py
uv run pytest tests/test_dev_wheel_versioning.py -v
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --print-version
python scripts/release/publish_crates.py
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
| Release publish cannot mint trusted token | Confirm PyPI pending publisher and GitHub `pypi` environment both match the workflow |
| Rust crate publish cannot authenticate | Confirm `CARGO_REGISTRY_TOKEN` is set in the `crates-io` environment |
| Rust crate dry-run rejects path dependencies | Confirm internal published dependencies include both `path = ...` and `version = "..."` |
| Full matrix runs on every commit | Ensure matrix jobs are only `workflow_dispatch` or root release-tag gated |
| Install imports checkout version | Verify from a temporary directory outside the repo |

## References

- Release workflow runbook: `docs/internal/release_workflow.md`
- Dev wheel version helper: `scripts/release/set_dev_wheel_version.py`
- Rust crate publish helper: `scripts/release/publish_crates.py`
- Public GitHub build workflow: `.github/workflows/publish.yml`
