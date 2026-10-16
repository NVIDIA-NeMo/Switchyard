---
name: switchyard-pr-preparation
description: Prepare or update a Switchyard pull request, check its scope, and identify CI changes that require separate user confirmation before editing.
---

# Switchyard PR Preparation

Read the repository's `AGENTS.md` and inspect the complete diff against the intended PR base,
including uncommitted and untracked files. Check that each change belongs to the requested scope.

## CI changes require separate confirmation

Before changing CI, show the proposed files, behavior, and reason, then obtain explicit user
confirmation for those CI changes. General approval to fix code, prepare a PR, or address review
comments does not authorize CI changes. A confirmation already given for the specific CI change
remains valid.

This covers GitHub workflows and actions, commit or PR lint rules, and scripts or configuration
changes that alter CI checks, triggers, permissions, or required behavior. Do not weaken a check
or change a repository policy to make an unrelated PR pass.

Read-only inspection and local reproduction can proceed while confirmation is pending. If the
existing diff already contains CI changes, identify their origin and show them separately for
review. Do not silently include or discard them.

## Before presenting the PR

Review the full diff again for CI changes, including files outside `.github/` that CI executes.
State which CI behavior changes, the specific confirmation received, and how it was validated.
Keep unrelated CI policy changes in a separate PR and confirm that scope with the user.

Use the repository's commit and PR title conventions. Follow the existing approval requirements
for commits, pushes, and posting the PR; this skill grants no publication permission.
