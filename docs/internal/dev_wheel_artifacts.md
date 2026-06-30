# Dev Wheel Artifacts

Audience: Switchyard maintainers who need a temporary wheel to test before an official release.

Switchyard currently follows the OSS-style NeMo path for GitHub builds:

- regular CI runs tests, linting, type checks, Rust checks, and slim-install smoke checks;
- manual dev builds create one Linux x86_64 wheel as a one-day GitHub Actions artifact;
- manual dev matrix builds create the full sdist and wheel set as GitHub Actions artifacts;
- root `vMAJOR.MINOR.PATCH` tags run the complete release validation and wheel matrix;
- public PyPI/GitHub publishing happens only from approved `vMAJOR.MINOR.PATCH` tag releases.

Wheel metadata uses the public distribution name `nemo-switchyard`, while the Python import and CLI
stay `switchyard`.

## Why Dev Builds Are Explicit Opt-In

We tried the usual NVIDIA-internal handoff paths first. GitHub-hosted runners could not resolve
`artifactory.nvidia.com` or `kitmaker-portal.nvidia.com`, and this repository does not currently
have the Dynamo-style NVIDIA self-hosted release runner setup. GitHub Packages also does not provide
a PyPI-compatible package index.

Because of those constraints, GitHub dev builds stay as short-lived artifacts. PyPI publishing is
reserved for tag-driven releases because PyPI versions are effectively immutable once uploaded.

## Manual Dev Artifact Build

Use the workflow-dispatch path in `.github/workflows/publish.yml`:

```text
Actions -> Build and publish release distributions -> Run workflow
```

Set:

| Input | Value |
|---|---|
| `build_dev_artifact` | `true` |
| `dev_version` | `0.0.1.dev0` |

The workflow:

1. Stamps build-local metadata to `version = "0.0.1.dev0"`.
2. Builds one manylinux x86_64 wheel.
3. Uploads `dev-wheel-linux-x86_64` with one-day retention.
4. Downloads the artifact again and verifies the wheel `Name` and `Version` metadata.

## Manual Dev Matrix Artifact Build

Use this to prove the complete release matrix before cutting an official tag:

| Input | Value |
|---|---|
| `build_dev_artifact` | `false` |
| `build_dev_matrix` | `true` |
| `dev_version` | `0.0.1.dev0` |

This path stamps the requested `.dev` version, runs the release checks, builds the sdist, builds the
full abi3 wheel matrix, and uploads the distributions as GitHub Actions artifacts. It does not
publish anything to PyPI.

## Official Release Build

Create a root `vMAJOR.MINOR.PATCH` tag only when a real release has been approved. Tag pushes run:

- Python release checks on Python 3.12, 3.13, and 3.14;
- Rust fmt, clippy, and workspace tests;
- source distribution build;
- full abi3 wheel matrix for Linux x86_64, Linux aarch64, macOS x86_64, macOS arm64,
  Windows x86_64, and Windows arm64;
- native wheel smoke installs where the runner can execute the artifact.

The workflow rejects release tags that do not exactly match `pyproject.toml`'s package version. For
example, package version `0.0.1` must be released with the `v0.0.1` tag.

The official `publish` job uses `uv publish --trusted-publishing always`, so PyPI project creation
and uploads require a matching pending trusted publisher:

| Field | Value |
|---|---|
| Project | `nemo-switchyard` |
| Owner | `NVIDIA-NeMo` |
| Repository | `Switchyard` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Do not create a root release tag until the PyPI pending publisher and GitHub `pypi` environment are
ready.

## Local Metadata Helper

To preview the metadata stamp locally:

```bash
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --print-version
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --package-name nemo-switchyard
```

Do not commit the stamped package metadata unless the release process explicitly requires it.
