# Release Workflow

Audience: Switchyard maintainers who need temporary wheel artifacts or an official PyPI release.

Switchyard currently follows the OSS-style NeMo path for GitHub builds:

- regular CI runs tests, linting, type checks, Rust checks, and slim-install smoke checks;
- manual dev builds create one Linux x86_64 wheel as a one-day GitHub Actions artifact;
- manual dev matrix builds create the full sdist and wheel set as GitHub Actions artifacts;
- root `vMAJOR.MINOR.PATCH` tags run the complete release validation and wheel matrix;
- public PyPI, crates.io, and GitHub publishing happens only from approved
  `vMAJOR.MINOR.PATCH` tag releases.

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

## Official Release Build

Create a root `vMAJOR.MINOR.PATCH` tag only when a real release has been approved. Tag pushes run:

- Python release checks on Python 3.12 through 3.14;
- Rust fmt, clippy, and workspace tests;
- source distribution build;
- full abi3 wheel matrix for Linux x86_64, Linux aarch64, macOS x86_64, macOS arm64,
  Windows x86_64, and Windows arm64;
- native wheel smoke installs where the runner can execute the artifact.

CI and distribution builds use Python 3.12 or newer. Package metadata and the wheel's stable ABI
still target Python 3.10 or newer, but CI no longer tests Python 3.10 or 3.11.

The workflow rejects release tags that do not exactly match `pyproject.toml`'s package version. For
example, package version `0.2.0` must be released with the `v0.2.0` tag. The Rust workspace and
Python package versions must also match.

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

The same tag publishes these crates to crates.io in dependency order:

1. `switchyard-protocol`
2. `switchyard-translation`
3. `switchyard-libsy`
4. `switchyard-llm-client`
5. `prefill-router`
6. `switchyard-runner`
7. `switchyard-server`

Add a repository Actions secret named `CARGO_REGISTRY_TOKEN` containing a crates.io API token that
can publish all seven crates and create new crates. The job waits for each version to reach the
crates.io index before publishing its dependents. If publication stops partway through, use
GitHub's **Re-run failed jobs** action so successful crate jobs are not repeated.

## Relay Plugin Bundles

Official Relay plugin bundles are built and distributed by
[NeMo Relay Plugins](https://github.com/NVIDIA/NeMo-Relay-Plugins/releases).
They contain the native library, completed manifest, schema, and license
notices. This delivery path does not require publishing
`switchyard-nemo-relay-plugin` to crates.io or adding it to the crate list above.

For Switchyard `0.3.0`, coordinate a plugin release pinned to the intended
Switchyard release commit. The plugin's `release.toml` records its own version
and `source.sha`. Versions can match, but the source pin establishes which
Switchyard code is bundled. Follow the plugin repository's
[release process](https://github.com/NVIDIA/NeMo-Relay-Plugins/blob/main/RELEASE.md)
to review and publish its draft. A Switchyard tag alone does not publish it.

Keep source-build validation and published-bundle validation as separate QA
results. Before publication, build and package the exact source revision under
test. Passing that path does not validate the eventual downloaded archive,
sidecars, or repository access. Keep published-bundle validation pending until
the official release is available.

After publication, validate the delivery path from
[the user installation guide](../../crates/switchyard-nemo-relay-plugin/README.md#install-a-released-bundle):

1. Confirm the intended users can access the published release. Verify the
   documented repository access and SSO/SAML requirements.
2. Download each supported platform's archive and matching `.sha256` and
   `.json` sidecars from that release. Verify the archive checksum and check
   the metadata's source commit, platform, and tested Relay version.
3. Extract the downloaded archive into a fresh installation directory. Follow
   the documented registration, deployment configuration, trust policy, and
   enablement steps using a supported official Relay runtime.
4. Restart Relay and run a routed-request smoke test with the installed bundle.
   Record the release tag, source commit, archive checksum, platform, Relay
   version, and result. A local rebuild is not a substitute for this check.

## Local Metadata Helper

To preview the metadata stamp locally:

```bash
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0 --print-version
python scripts/release/set_dev_wheel_version.py 0.0.1.dev0
```

Do not commit the stamped package metadata unless the release process explicitly requires it.
