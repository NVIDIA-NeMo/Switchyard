# Kitmaker Devzone Prerelease Wheels

Audience: Switchyard maintainers publishing public prerelease wheels to NVIDIA's Devzone Python
index at `pypi.nvidia.com`.

This path is for wheels that are safe to expose publicly but should not go to public PyPI yet.
The workflow intentionally stages wheels as short-lived GitHub Actions artifacts first. Kitmaker
Portal submission is a separate follow-up after those artifacts have been reviewed. Do not create
GitHub release tags for this path.

## What Kitmaker Portal Is

Kitmaker Portal is NVIDIA's release system for public Python wheels. The Portal UI manages
projects, owners, API tokens, and service accounts. The Portal API submits wheel release requests,
runs Kitmaker checks, and publishes accepted artifacts to PyPI and Devzone according to Kitmaker
policy.

For `.dev` prerelease versions, Kitmaker publishes to `pypi.nvidia.com` only. It does not publish
those packages to public PyPI. The project name still must be registered on PyPI before any Devzone
release to avoid dependency-confusion risk.

Do not use direct `uv publish` to `pypi.nvidia.com` for this path. Kitmaker owns Devzone uploads.

## Switchyard Values

Use these values for the first prerelease line:

| Field | Value |
|---|---|
| Distribution/project name | `nemo-switchyard` |
| Import package | `switchyard` |
| Repository | `https://github.com/NVIDIA-NeMo/Switchyard` |
| Prerelease version | `0.0.1.dev0` |
| Wheel metadata name | `nemo-switchyard` |
| Wheel filename prefix | `nemo_switchyard-` |
| CI artifact targets | Linux x86_64, Linux aarch64, macOS x86_64, macOS arm64, Windows x86_64 |

`0.0.1.dev` is accepted by Python packaging tools but normalizes to `0.0.1.dev0`. Prefer writing
the normalized form explicitly in release notes and installer commands.

## Preconditions

Before submitting a release:

1. Confirm the artifact is public-safe. `pypi.nvidia.com` is externally visible.
2. Create a Kitmaker Portal project named exactly `nemo-switchyard`.
3. Make sure the Portal project owner/PIC is the person or service account submitting the release.
4. Register or reserve the `nemo-switchyard` project name on PyPI through Kitmaker.
5. Create a Kitmaker API token.
6. Build wheels with public-index-safe version metadata and platform tags.
7. Download the one-day GitHub artifacts before they expire.
8. Submit the reviewed wheel artifacts through the approved Kitmaker handoff.

For CI automation, use a Kitmaker service-account token instead of a personal user token.

## GitHub Secrets

The workflow `.github/workflows/devzone-prerelease.yml` does not need release secrets to build and
verify GitHub wheel artifacts.

Kitmaker credentials are needed only for the later Portal/API submission:

| Secret | Purpose |
|---|---|
| `KITMAKER_API_TOKEN` | Kitmaker Portal API token |
| `KITMAKER_PROJECT_ID` | Portal project id for `nemo-switchyard` |
| `KITMAKER_PIC_EMAIL` | PIC email in Kitmaker release payloads |

## Build Metadata

For prerelease wheel builds, the CI job temporarily stamps:

```toml
[project]
name = "nemo-switchyard"
version = "0.0.1.dev0"
```

This is build-local metadata only. The source tree can keep its normal development version. Keep
the Python import and Rust extension module unchanged:

```toml
[tool.maturin]
module-name = "switchyard_rust._switchyard_rust"
```

## Workflow

Start the manual workflow:

```text
Actions -> Build Devzone prerelease wheels -> Run workflow
```

Use these inputs for the first GitHub-artifact-only staging run:

| Input | Value |
|---|---|
| `version` | `0.0.1.dev0` |
| `target_sha` | empty, or a full source SHA |

The workflow:

1. Validates the `.dev` version and source SHA.
2. Builds the abi3 wheel matrix.
3. Verifies every wheel has `Name: nemo-switchyard` and the requested version.
4. Uploads each wheel as a `devzone-wheel-*` GitHub Actions artifact with one-day retention.
5. Downloads the artifacts in a verifier job and re-checks the wheel metadata.

The first GitHub-artifact-only run intentionally skips the full Python/Rust release checks and wheel
smoke installs. It proves the `.dev` metadata, wheel build, GitHub artifact upload, and artifact
download path before adding Kitmaker or release validation back into the loop.

GitHub Actions artifacts are authenticated, short-lived handoff artifacts. If the Kitmaker API path
requires direct fetch URLs, do not submit raw GitHub artifact API URLs unless the Portal flow has
explicitly been validated with them. Download the artifacts before expiry and use the approved
Kitmaker handoff.

## Artifact Requirements

Kitmaker checks package metadata, trove classifiers, rendered description, wheel tags, ABI tags,
platform tags, and source-distribution structure when applicable.

For Linux wheels, use the manylinux shape built by CI. A local `uv build` on a workstation usually
produces a `linux_x86_64` wheel, which is acceptable for an endpoint/auth probe but not the
artifact to publish as the real prerelease.

## Portal Setup

Create and test an API token:

```bash
export KITMAKER_API_TOKEN="kmp_..."

curl -H "Authorization: Bearer ${KITMAKER_API_TOKEN}" \
  "https://kitmaker-portal.nvidia.com/api/v0/projects" | jq .
```

If the host does not trust NVIDIA IT certificates, Kitmaker docs allow adding `--insecure` to the
`curl` commands. Prefer fixing local CA trust for repeatable automation.

Find the `project_id` for `nemo-switchyard` from the projects response. The API request's
`project_name` must exactly match the Portal project name.

## Manual Dry Run

If you need to debug the Kitmaker API manually, use `upload: false` with a URL or handoff shape that
Portal can fetch:

```bash
curl -X POST "https://kitmaker-portal.nvidia.com/api/v0/projects/<project_id>/releases" \
  -H "Authorization: Bearer ${KITMAKER_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "nemo-switchyard",
    "payload": [
      {
        "pic": "your.email@nvidia.com",
        "job_type": "wheel-release-job",
        "url": "https://<kitmaker-fetchable-wheel-url>/nemo_switchyard-0.0.1.dev0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "upload": false
      }
    ]
  }'
```

Save the returned `release_uuid`.

## Monitor

Poll release status:

```bash
curl -H "Authorization: Bearer ${KITMAKER_API_TOKEN}" \
  "https://kitmaker-portal.nvidia.com/api/v0/status/<release_uuid>" | jq .
```

Expected status progression is usually `pending`, `building`, then either `completed` or `failed`.

## Manual Publish

After the dry run passes and the artifact handoff is final, repeat the release request with
`upload: true`:

```json
"upload": true
```

Because `0.0.1.dev0` is a prerelease, Kitmaker should publish it only to Devzone
(`pypi.nvidia.com`), not public PyPI.

## Verify Install

Verify from a temporary directory so the checkout cannot shadow the installed wheel:

```bash
cd "$(mktemp -d)"

uv run --isolated --no-project \
  --extra-index-url "https://pypi.nvidia.com" \
  --with "nemo-switchyard==0.0.1.dev0" \
  python -c "import switchyard; print(switchyard.__version__); print(switchyard.__file__)"
```

## Common Failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `Project name mismatch` | Request `project_name` does not match Portal project | Use `nemo-switchyard` exactly |
| `Not authorized to create releases` | Token owner is not project owner/PIC | Transfer Portal project ownership or use the correct service token |
| GitHub artifact expired | Artifacts are retained for one day | Rerun the workflow and download the new artifact |
| Release fails during URL fetch | Wheel URL is not directly downloadable by Portal | Use a Kitmaker-approved fetchable artifact handoff |
| Wheel tag check fails | Local wheel uses `linux_x86_64` instead of manylinux | Build with the manylinux release workflow |
| Install cannot find package | Package name or version mismatch | Install `nemo-switchyard==0.0.1.dev0` with `--pre` or explicit version |

## References

- Kitmaker wheel docs: `https://kitmaker.gitlab-master-pages.nvidia.com/kitmaker-docs/users/wheels/index.html`
- Kitmaker Portal API docs: `https://kitmaker.gitlab-master-pages.nvidia.com/kitmaker-docs/users/portal-api/index.html`
- Wheel release API: `https://kitmaker.gitlab-master-pages.nvidia.com/kitmaker-docs/users/portal-api/wheel-release-api.html`
- Devzone prerelease workflow: `../../.github/workflows/devzone-prerelease.yml`
