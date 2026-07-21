# CI Options and Workflow Design

This directory is the maintainer-facing entry point for choosing local validation and designing
GitHub Actions workflows. The defaults below incorporate patterns used in NVIDIA NeMo Data
Designer and NeMo Curator, adapted to Switchyard's required `CI Success` aggregate and public-fork
threat model.

## Pattern Provenance

- **NeMo Curator** informed the two-stage preview model: build from the pull request without
  credentials, then perform privileged work from trusted workflow metadata.
- **NeMo Data Designer** informed same-repository preview policy, PR-scoped cancellation, bot
  comment filtering, serialized publishing, and keeping action/runtime pins current.
- **Switchyard** adds the stable `CI Success` aggregate, job-level permission separation, trusted
  default-branch tool configuration, and regression tests for workflow security invariants.

## Choose a Validation Scope

Use the selector to turn changed paths into the smallest relevant local command set:

```bash
# Uncommitted work, including staged and untracked files
python scripts/select_validation.py --changed

# The whole branch, including committed work
git fetch origin main
python scripts/select_validation.py --base origin/main

# A proposed path before editing
python scripts/select_validation.py --path .github/workflows/example.yml
```

`--json` provides the same plan to editor or agent integrations. The selector never schedules live
provider tests; those require explicit intent and credentials. Before pushing, use the full hard
gate in `.github/workflows/ci.yml` when the change crosses multiple ownership areas or affects
packaging, shared runtime behavior, or the required aggregate.

## Choose a Workflow Shape

| Need | Preferred option | Why |
| --- | --- | --- |
| Required PR validation | Call a reusable workflow from `ci.yml` and add its job to `CI Success.needs` | Branch protection keeps one stable required check while new gates cannot become accidentally optional. |
| Path-filtered or advisory validation | Use a separate `pull_request` workflow | It can skip irrelevant changes without making the required aggregate wait for a check that never appears. |
| Secretless preview build | Run on `pull_request` with read-only permissions | Fork code can be checked without exposing secrets or write tokens. |
| Secret-bearing preview or PR comment | Use a trusted `workflow_run` follow-up for same-repository PRs only | Untrusted PR code and trusted credentials never execute in the same job. |
| Production publishing | Trigger from the default branch or a release tag and serialize runs | Prevents an older deployment from racing and overwriting a newer one. |
| Expensive release matrix | Use tag or manual dispatch | Keeps routine PR feedback fast while preserving an explicit full-matrix option. |

## Required Design Checks

### Stable required aggregation

- Every hard gate must be a direct or transitive dependency of `CI Success`.
- Keep branch protection pointed at `CI Success`, not a changing list of matrix jobs.
- Do not put path filters on a required workflow unless a stable always-running aggregate accounts
  for the skipped result.

### Trust boundaries

- Treat pull-request source, artifacts, cache contents, and artifact filenames as untrusted.
- Derive PR identity, repository identity, and tool versions from trusted event metadata or the
  default-branch checkout—not from a downloaded artifact.
- Restrict secret-bearing previews to same-repository PRs. Fork PRs receive validation only.
- A `workflow_run` workflow must exist on the default branch before GitHub will trigger it.

### Concurrency

- For PR work, group by workflow plus PR number and cancel superseded runs.
- For production publishing, use one stable deployment group and do not cancel an active publish.
- Avoid grouping only by branch when multiple PRs or deployment targets can share it.

### Permissions and secrets

- Start privileged workflows with `permissions: {}` and grant only the permissions each job needs.
- Separate generation, commenting, and repository deployment jobs when they use different secrets
  or write scopes.
- Keep `pull-requests: write`, `contents: write`, and publishing tokens out of jobs that execute PR
  code.
- When upserting bot comments, match both a unique marker and the expected bot author.

### Supply chain and toolchains

- Pin third-party actions to immutable commit SHAs and retain a version comment for update tools and
  reviewers.
- Pin command-line tools exactly when their output or hosted behavior is release-sensitive.
- Use runtime versions still supported by the action ecosystem; upgrade intentionally and test the
  full workflow path.
- Run `actionlint` for workflow syntax and `zizmor --pedantic .github/workflows` for security. Keep
  suppressions narrow, inline, and justified.

## Review Checklist

Before merging a workflow change, answer all of these:

1. Is the check required, advisory, privileged follow-up, or release-only?
2. If required, does failure reach `CI Success`?
3. Can fork-controlled code, metadata, artifacts, or caches reach a secret or write permission?
4. Is superseded PR work cancelled and production work serialized?
5. Are permissions job-scoped and external actions immutable?
6. Are tool versions selected from trusted configuration?
7. Do regression tests encode the security and aggregation invariants?
8. Do `actionlint`, `zizmor`, and the path-selected local validations pass?

Workflow-specific operational guidance remains with its owner, such as the
[release workflow](../../docs/internal/release_workflow.md) for package publication and the
[Switchyard docs skill](../../.agents/skills/switchyard-docs/SKILL.md) for Fern publishing.
