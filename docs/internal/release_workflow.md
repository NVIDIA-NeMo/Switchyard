# Release Workflow

Audience: Switchyard maintainers who need temporary wheel artifacts or an official PyPI release.

Switchyard currently follows the OSS-style NeMo path for GitHub builds:

- regular CI runs tests, linting, type checks, Rust checks, and slim-install smoke checks;
- manual dev builds create one Linux x86_64 wheel as a one-day GitHub Actions artifact;
- manual dev matrix builds create the full sdist and wheel set as GitHub Actions artifacts;
- manual Rust crate package runs create `.crate` artifacts without publishing;
- root `vMAJOR.MINOR.PATCH` tags run the complete release validation and wheel matrix;
- public PyPI/GitHub/crates.io publishing happens only from approved `vMAJOR.MINOR.PATCH` tag
  releases, except the first Rust crate publish can be dispatched from `main` after approval.

Wheel metadata uses the public distribution name `nemo-switchyard`, while the Python import and CLI
stay `switchyard`.

## Why Dev Builds Are Explicit Opt-In

GitHub dev builds are for validation and review. They should not publish from branch state, and
GitHub Packages is not a PyPI-compatible package index.

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

## Manual Rust Crate Package Dry-Run

Use this before the first crates.io publish or after changing Rust release metadata:

| Input | Value |
|---|---|
| `package_rust_crates` | `true` |
| `publish_rust_crates` | `false` |

The workflow runs `cargo publish --dry-run` for each public crate in dependency order and uploads
the generated `.crate` files as the `switchyard-rust-crates` artifact. This is safe on branches and
does not require a crates.io token. During dry-run only, the helper patches crates.io dependencies
back to local workspace paths so dependent crates can be verified before the first dependency
versions exist in the public registry. Real publishes do not use those patches.

The public crate publish order is:

| Order | Crate |
|---:|---|
| 1 | `switchyard-core` |
| 2 | `switchyard-translation` |
| 3 | `switchyard-components` |
| 4 | `switchyard-components-v2-macros` |
| 5 | `switchyard-components-v2` |
| 6 | `switchyard-server` |

`switchyard-py` is intentionally not published to crates.io. It is the PyO3 extension crate shipped
inside the `nemo-switchyard` Python wheel.

## Manual First Rust Crate Publish

The first crates.io publish can be dispatched from `main` after the crate metadata MR has merged.
Set:

| Input | Value |
|---|---|
| `publish_rust_crates` | `true` |

This path is blocked unless the workflow is running from `refs/heads/main`. It also requires:

| GitHub setting | Value |
|---|---|
| Environment | `crates-io` |
| Secret | `CARGO_REGISTRY_TOKEN` |

Create a crates.io API token with publish rights and store it as `CARGO_REGISTRY_TOKEN` in the
`crates-io` environment. The script publishes in dependency order and pauses between uploads so the
registry index can settle before publishing dependent crates.

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
ready. Official tag releases also publish the public Rust crates before the Python distribution
publish job starts, so the `crates-io` environment and `CARGO_REGISTRY_TOKEN` secret must be ready
before cutting a tag.

After Rust crate publishing completes, verify:

| Check | URL |
|---|---|
| Crate listing | `https://crates.io/crates/<crate-name>` |
| Rust docs | `https://docs.rs/<crate-name>/<version>/<crate-name-with-underscores>/` |

Capture any docs.rs failures as follow-up work instead of republishing the same version; crates.io
versions are immutable.

## Local Metadata Helper

To preview the metadata stamp locally:

```bash
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --print-version
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --package-name nemo-switchyard
```

Do not commit the stamped package metadata unless the release process explicitly requires it.

To validate Rust crate release metadata locally:

```bash
python scripts/release/publish_crates.py
```

This runs the same `cargo publish --dry-run` sequence used by GitHub Actions.
Before committing release-infra changes, use `--allow-dirty` to test the staged manifests locally
without weakening the CI path:

```bash
python scripts/release/publish_crates.py --allow-dirty
```
