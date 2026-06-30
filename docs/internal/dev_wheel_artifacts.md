# Dev Wheel Artifacts

Audience: Switchyard maintainers who need a temporary wheel to test before an official release.

Switchyard currently follows the OSS-style NeMo path for GitHub builds:

- regular CI runs tests, linting, type checks, Rust checks, and slim-install smoke checks;
- manual dev builds create one Linux x86_64 wheel as a one-day GitHub Actions artifact;
- root `v*` tags run the complete release validation and wheel matrix;
- public PyPI/GitHub publishing remains disabled until the OSS release gate is approved.

The wheel metadata for temporary artifacts uses the public distribution name `nemo-switchyard`,
while the Python import stays `switchyard`.

## Why This Is Artifact-Only

We tried the usual NVIDIA-internal handoff paths first. GitHub-hosted runners could not resolve
`artifactory.nvidia.com` or `kitmaker-portal.nvidia.com`, and this repository does not currently
have the Dynamo-style NVIDIA self-hosted release runner setup. GitHub Packages also does not provide
a PyPI-compatible package index.

Because of those constraints, GitHub dev builds are intentionally short-lived artifacts, not package
index uploads. Download and inspect them manually, then use the official tag-gated release path when
the project is ready for public publishing.

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

1. Stamps build-local metadata to `name = "nemo-switchyard"` and `version = "0.0.1.dev0"`.
2. Builds one manylinux x86_64 wheel.
3. Uploads `dev-wheel-linux-x86_64` with one-day retention.
4. Downloads the artifact again and verifies the wheel `Name` and `Version` metadata.

## Official Release Build

Create a root `v*` tag only when a real release has been approved. Tag pushes run:

- Python release checks on Python 3.12, 3.13, and 3.14;
- Rust fmt, clippy, and workspace tests;
- source distribution build;
- full abi3 wheel matrix for Linux x86_64, Linux aarch64, macOS x86_64, macOS arm64, and Windows x86_64;
- native wheel smoke installs where the runner can execute the artifact.

The `publish` job is still hard-disabled in the workflow. Remove that guard only when the public
PyPI/GitHub release process is approved.

## Local Metadata Helper

To preview the metadata stamp locally:

```bash
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --print-version
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --package-name nemo-switchyard
```

Do not commit the stamped package metadata unless the release process explicitly requires it.
