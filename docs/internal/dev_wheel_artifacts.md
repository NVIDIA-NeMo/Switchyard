# Dev Wheel Artifacts

Audience: Switchyard maintainers who need a temporary wheel to test before an official release.

Switchyard currently follows the OSS-style NeMo path for GitHub builds:

- regular CI runs tests, linting, type checks, Rust checks, and slim-install smoke checks;
- manual dev builds create one Linux x86_64 wheel as a one-day GitHub Actions artifact;
- manual dev builds can optionally publish that exact verified wheel to PyPI;
- manual dev matrix builds can publish the full sdist and wheel set to PyPI;
- root `v*` tags run the complete release validation and wheel matrix;
- public PyPI/GitHub publishing happens only from approved `v*` tag releases.

Wheel metadata uses the public distribution name `nemo-switchyard`, while the Python import and CLI
stay `switchyard`.

## Why Dev Builds Are Explicit Opt-In

We tried the usual NVIDIA-internal handoff paths first. GitHub-hosted runners could not resolve
`artifactory.nvidia.com` or `kitmaker-portal.nvidia.com`, and this repository does not currently
have the Dynamo-style NVIDIA self-hosted release runner setup. GitHub Packages also does not provide
a PyPI-compatible package index.

Because of those constraints, GitHub dev builds default to short-lived artifacts. Publishing a dev
wheel to PyPI is possible, but it is a separate explicit workflow input because PyPI versions are
effectively immutable once uploaded.

## Manual Dev Artifact Build

Use the workflow-dispatch path in `.github/workflows/publish.yml`:

```text
Actions -> Build and publish release distributions -> Run workflow
```

Set:

| Input | Value |
|---|---|
| `build_dev_artifact` | `true` |
| `publish_dev_to_pypi` | `false` |
| `dev_version` | `0.0.1.dev0` |

The workflow:

1. Stamps build-local metadata to `version = "0.0.1.dev0"`.
2. Builds one manylinux x86_64 wheel.
3. Uploads `dev-wheel-linux-x86_64` with one-day retention.
4. Downloads the artifact again and verifies the wheel `Name` and `Version` metadata.

## Manual Dev PyPI Upload

Use this only when a public prerelease upload is intended:

| Input | Value |
|---|---|
| `build_dev_artifact` | `true` |
| `publish_dev_to_pypi` | `true` |
| `dev_version` | `0.0.1.dev0` |

The publish job runs only after the wheel metadata verifier succeeds. It uses
`uv publish --trusted-publishing always dist/*.whl`, so PyPI project creation and uploads require a
matching pending trusted publisher:

| Field | Value |
|---|---|
| Project | `nemo-switchyard` |
| Owner | `NVIDIA-NeMo` |
| Repository | `Switchyard` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

This publishes only the single Linux x86_64 dev wheel. Use a root `v*` tag for the complete release
matrix.

## Manual Dev Matrix PyPI Upload

Use this to prove the complete release matrix can publish before cutting an official tag:

| Input | Value |
|---|---|
| `build_dev_artifact` | `false` |
| `publish_dev_to_pypi` | `false` |
| `build_dev_matrix` | `true` |
| `publish_dev_matrix_to_pypi` | `true` |
| `dev_version` | `0.0.1.dev0` |

This path stamps the requested `.dev` version, runs the release checks, builds the sdist, builds the
full abi3 wheel matrix, and publishes the combined distributions with:

```bash
uv publish --trusted-publishing always --check-url https://pypi.org/simple/nemo-switchyard/ dist/*
```

`--check-url` lets the job skip files that were already uploaded for the same version. That matters
when a single-wheel dev upload already published the Linux x86_64 wheel.

## Official Release Build

Create a root `v*` tag only when a real release has been approved. Tag pushes run:

- Python release checks on Python 3.12, 3.13, and 3.14;
- Rust fmt, clippy, and workspace tests;
- source distribution build;
- full abi3 wheel matrix for Linux x86_64, Linux aarch64, macOS x86_64, macOS arm64, and Windows x86_64;
- native wheel smoke installs where the runner can execute the artifact.

The official `publish` job uses `uv publish --trusted-publishing always`, so PyPI project creation
and uploads require the same pending trusted publisher:

| Field | Value |
|---|---|
| Project | `nemo-switchyard` |
| Owner | `NVIDIA-NeMo` |
| Repository | `Switchyard` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Do not create a root `v*` tag until the PyPI pending publisher and GitHub `pypi` environment are
ready.

## Local Metadata Helper

To preview the metadata stamp locally:

```bash
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --print-version
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --package-name nemo-switchyard
```

Do not commit the stamped package metadata unless the release process explicitly requires it.
