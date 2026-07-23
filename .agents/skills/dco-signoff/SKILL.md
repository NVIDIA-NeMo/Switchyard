---
name: dco-signoff
description: Fix missing DCO (Developer Certificate of Origin) sign-off on branch commits and prevent the issue going forward. Use when CI reports a DCO check failure, a PR is blocked by "Signed-off-by missing", or a collaborator asks you to add sign-off to commits.
---

# DCO Sign-Off

Every commit merged into Switchyard must carry a `Signed-off-by` trailer:

```
Signed-off-by: Your Name <your@email.com>
```

This is enforced by the DCO bot on every pull request. A branch with any commit missing this trailer will be blocked from merge.

## Detect missing sign-offs

Check which branch commits are missing the trailer:

```bash
git log origin/main..HEAD --format="%H %s" | while read sha msg; do
  if ! git show -s --format="%B" "$sha" | grep -q "^Signed-off-by:"; then
    echo "MISSING: $sha  $msg"
  fi
done
```

If the output is empty, all commits are already signed and no action is needed.

## Fix: add sign-off to all branch commits

Rebase the branch onto `origin/main`, adding `--signoff` to retrofit the trailer on every commit:

```bash
git rebase origin/main --signoff
```

Then push. Because a rebase rewrites SHAs, you must force-push:

```bash
git push --force-with-lease origin HEAD
```

`--force-with-lease` is safer than `--force`: it aborts if the remote has received new commits since your last fetch, protecting against overwriting a collaborator's work.

### When the rebase hits a conflict

Resolve each conflict normally, then continue:

```bash
git add <resolved-files>
git rebase --continue
```

The `--signoff` flag was set at rebase-start; `--continue` applies it to each commit as it lands. You do not need to pass `--signoff` again.

## Prevent it going forward

Pass `-s` (shorthand for `--signoff`) on every `git commit`:

```bash
git commit -s -m "type(scope): your message"
```

Or add it to the repo's local git config so it is applied automatically:

```bash
git config commit.gpgSign false   # unrelated — don't confuse with signoff
```

There is no `commit.signoff = true` git config option; the `-s` flag must be used explicitly each time, or the workflow must always pass it. The safest habit is to include `-s` in every `git commit` invocation.

## What a correct trailer looks like

```
fix(protocol): correct sub-agent detection for Claude Code

Signed-off-by: Lin Jia <linj@nvidia.com>
```

The name and email must match the contributor's Git identity (`git config user.name` and `git config user.email`). A mismatch causes the DCO bot to reject the commit even when the trailer is present.

## Verify the fix

After rebasing and pushing, confirm locally before relying on CI:

```bash
git log origin/main..HEAD --format="%H %s%n%b" | grep -E "(^[0-9a-f]{40}|Signed-off-by)"
```

Every commit SHA should be followed by a `Signed-off-by:` line.

## Boundaries

### Always do

- Use `--force-with-lease` instead of `--force` when pushing a rebased branch.
- Verify that `git config user.name` and `git config user.email` match the expected identity before adding sign-off — a mismatch is caught by the DCO bot.
- Run the detection check first; if all commits already have sign-off, do nothing.

### Ask first

- Rebasing a branch that has open review comments attached to specific commit SHAs — the rewrite makes those comments orphaned on GitHub.
- Force-pushing a branch that is also used by another collaborator's local checkout.

### Never do

- Force-push `main` or any protected branch to add sign-off. The correct fix for a merged commit is to ensure future commits are signed; retroactive rewriting of main history is not possible.
- Use `--no-verify` to skip the DCO pre-push hook if one is configured — that bypasses the gate without fixing the root cause.
