---
name: switchyard-pr-reviewer
description: >
  Multi-mode, adversarially-verified PR review for switchyard that drafts inline comments
  the maintainers' way. Use when asked to "review this PR", "do a live/code review", "walk
  me through this PR and flag anything", "check correctness / tests / design vs the ticket",
  "is this serious / blocking?", "review the Rust changes", or "post comments on a PR".
  Sequences correctness, test-quality, ticket-coherence, simplify, docs, stale-comment, and
  Rust-craft modes; dispatches Rust to rust-code-reviewer. Every phase has a self-contained
  manual path; when the host CLI provides them, built-in slash skills are optional
  accelerators. Drafts first, posts only the approved subset, never auto-posts.
---

# Switchyard PR Reviewer

The switchyard-specific **orchestrator** that sequences review MODES and enforces two
contracts: **adversarial verification** (every finding refuted-or-confirmed against real
code) and **anti-slop** (no obvious/restating comments, no auto-posting). It routes Rust to
the `rust-code-reviewer` skill. **The workflow is self-contained** — the built-in slash
skills named below are *optional accelerators* for hosts that have them (e.g. Claude Code);
where they're absent, run the phase's manual steps.

Three pointers (the phases below carry the full contract):
1. Draft first; post only the approved subset (Phases 8–9).
2. Verify every finding against real code before reporting (Phase 3).
3. Comments-only intent never edits code (dispatch table + Phase 2).

**Prerequisites:** an authenticated `gh` CLI (PR data + posting) and, for ticket coherence, the `linear-server` MCP (optional — degrade to the PR description if absent).

---

## When to use / dispatch

Pick modes by what the diff touches and what the user asked. Run them as PHASES (below);
multiple usually apply. The **Optional accelerator** column names a Claude Code slash-skill
that can speed a phase up **if the host CLI has it** — it is never required; absent it, run
that phase's manual steps. **Refer to, don't copy, any linked tool.**

| Diff signal / user intent | Mode → Phase | Optional accelerator (else run the phase manually) |
|---|---|---|
| "review this PR", general bug hunt | Correctness (Phase 2/3) | `/code-review` if available (effort high→max = broader, may be uncertain) |
| "walk me through it and flag" / "live review" | Guided walk-through (intent→plug-in→per-fn→tests, one chunk/turn) | — (interactive, no fan-out) |
| Cleanup / "I dislike big docstrings" / quality, NOT bugs | Simplify: reuse / simplification / efficiency / altitude | Run the 4 angles **manually** for a comments-only review. `/simplify` (if available) is a shortcut ONLY when the user asked to APPLY cleanups — it mutates the tree by default, no comment-only flag. |
| `crates/*` Rust, PyO3/FFI, translation, async, locks | Rust-craft (P0/FFI-first) | **rust-code-reviewer** skill — do NOT restate its rules |
| New/changed tests, suspicious coverage | Test-quality (Phase 5) | — (fan-out one agent per test file) |
| Linear ticket / "is this coherent with the ticket" / "done-when" | Design-coherence (Phase 4) | `mcp__linear-server__get_issue` (or read the ticket however the host exposes it) |
| `*.md` / README / docs changed | Docs-accuracy + production-doc structural review (Phase 4b) | — (docs lens) |
| In-code comments/docstrings that may contradict shipped code | Stale-comment coherence (Phase 2 finder) | — (grep) |
| Security-sensitive (auth, secrets, headers, deserialization) | Security (Phase 2) | `/security-review` if available |
| "approve it / LGTM" | Summary verdict | `gh pr review <N> --approve` |
| "review thoroughly, give me your views" (structured artifact) | Verdict→Overview→Verification→Findings (Phase 8 format) | `/review` if available |

Mixed Python + Rust: run Python modes here, dispatch `crates/*` hunks to
`rust-code-reviewer` (P0-first if the user said "P0s only").

---

## Severity taxonomy

Use the word labels consistently. Correctness outranks cleanup/altitude when a cap forces a
cut. Tag the finding TYPE in parens, e.g. `Blocking (vacuous — the "clip" test never clips)`,
`Blocking (project rule)`, `Important (reuse — duplicates StageRouterFactory.validate)`.

| Label | Meaning | Examples from past PRs |
|---|---|---|
| **Blocking** | Real bug, hard repo-rule violation, dead public/FFI surface, vacuous/misleading test, must-fix-before-merge | wrong-slot `extra_request_processors=`; `tests_passed` substring veto; "clip" test never clips |
| **Important** | Correctness degradation, design/altitude leak, hot-path waste, contract/parity inconsistency — real but non-blocking | wire-format parsing in policy layer; 3-copy OpenAI `/v1/models` shape; type-safety hole |
| **Minor** | Stale comment/doc, small footgun, missing intent comment, edge case acceptable-as-is | "15 scorers" where it's 14; dict-ctor skips `${ENV}` interpolation |
| **Style** | Idiom, naming, redundant test, AI over-enumeration — fold into summary, rarely post individually | `plan` alias + self-justifying test; obvious `///` docs restating the fn name |

Rust mode reports **P0 / P1 / P2**; map P0→Blocking, P1→Important, P2→Minor.

---

## PHASES

### Phase 0 — Scope & gather
**Goal:** know exactly what the change CLAIMS to do before reading a line of logic.
**How:**
- `gh pr view <N> --json title,body,headRefOid,headRefName,commits`, `gh pr diff <N> --stat`, then full `gh pr diff <N>`.
- Fetch the branch locally — **the code is often not on `main`:** `git fetch origin <branch>`, read source from it.
- Pull the linked Linear ticket (SWITCH-*) named in the title/body via `mcp__linear-server__get_issue`; capture its scope and "Done when" criteria. No ticket → use the PR description as stated intent.
- Record `HEAD_SHA=$(gh pr view <N> --json headRefOid -q .headRefOid)`.

**Exit gate:** state in 1–2 lines "what this PR does" and "what the ticket asked for", and whether they're additive/bridge vs. behavior-changing.

### Phase 1 — Dispatch to modes
**Goal:** choose the minimal set of modes that fit this diff.
**How:** apply the dispatch table. Honor explicit user scoping ("P0s only", "just the Rust", "design + ticket coherence", "tests for wrong reasons"). For "walk me through", switch to interactive one-chunk-per-turn cadence: system context (the 4-role chain) → the function's plug-in point → per-field behavior → tests, pausing each turn. Don't drop the user into a function cold.

**Exit gate:** an ordered mode list, and whether posting/fan-out is in scope.

### Phase 2 — Per-mode finders (fan-out)
**Goal:** surface candidates with `file:line` + one-line summary + concrete failure/cost.
**How** (read the actual current source, not just the diff; grep callers of changed signatures and references to deleted symbols):
- **Correctness:** manual line-by-line scan + removed-behavior auditor + cross-file tracer (optionally accelerate with `/code-review` if the host has it). Trace each runtime field to where it's set, including into Rust. Carry a provisional **BUG / DEAD-CODE / DESIGN-CHOICE** label into Phase 3.
- **Simplify:** run the 4 angles (reuse / simplification / efficiency / altitude) MANUALLY for a comments-only review, one angle per agent, in a single message. Use `/simplify` only if it's available AND the user asked to apply cleanups.
- **Stale code-comment / coherence-with-shipped-code:** grep for comments that disagree with the code beside them — count comments that no longer match; docstrings/knob lists describing options the loader now rejects; mentions of a branch deleted in this PR; AGENTS.md-banned bookkeeping comments (`SWITCH-*`, `Phase-A`, commit hashes, `TODO(step-N)`). Minor normally, Important if it would mislead a maintainer.
- **Test-quality:** one agent per changed test file (see Phase 5).
- **Rust-craft:** the `rust-code-reviewer` skill on `crates/*` hunks.
- **Big PRs:** the user may ask for an explicit N-agent sweep ("10 sub agents", "don't post anything") partitioned by area × lens. Honor "analysis-only".

**Dedup:** collapse findings at the same line/mechanism; note convergence ("found by 2 agents") as a confidence signal — NOT extra severity.

**Exit gate:** a deduped candidate list, each with evidence anchors and a provisional bug/dead/design label.

### Phase 3 — Adversarial verification gate (mandatory)
**Goal:** kill false alarms. **A candidate survives only if you re-read the code and it still holds.**
**How — for each candidate, label CONFIRMED / PLAUSIBLE / REFUTED, and classify BUG / DEAD-CODE / DESIGN-CHOICE:**
- Read the real code path end-to-end. Trace fields across the FFI boundary. Confirm wiring is live (endpoint mounted? export consumed?) before calling anything dead.
- **Refute by execution or grep, not by reasoning.** If you reason instead of running, SAY SO ("this is reasoning, not a test").
- **Prove a rule inert by cross-checking the PR's own calibration data and commits.** (Mined: a `=`-instead-of-`+=` scorer that fired 0/30,851 times AND was absent from `DEFAULT_WEIGHTS` → DEAD-CODE, not a live bug — which changes the fix.)
- If a fan-out agent grepped, confirm it grepped the **right working tree** (classic false negative: grepping a sibling branch lacking this PR's Python).
- Drop REFUTED. Keep CONFIRMED + PLAUSIBLE. De-escalate judgment calls to Minor/Style or a question.

**Worked false-alarms to kill** (each really survived a finder and died on a re-read):

| Candidate (sounds real) | Why it's REFUTED on reading the code |
|---|---|
| "streaming bypasses the detection path" | the status check precedes the stream branch; detection runs |
| "error escapes the handler → 500" | always retried then translated to 400; never escapes |
| "new error helper duplicates the existing one" | one passes the body verbatim, the other builds per-inbound envelopes — different jobs |
| "this config / field has zero consumers" | agent grepped the wrong tree; it's wired on the PR branch |
| "lazy singleton is a lifecycle leak" | the type is stateless — no startup/shutdown, not a leak |
| "catch order shadows the broader error" | the errors are siblings, not parent/child — harmless |
| local test run fails | diagnose env (e.g. toolchain version mismatch) — NOT a PR defect |

**Exit gate:** every surviving finding has a "verified against `<file:line>`" note and a bug/dead/design label; every dropped one has a one-line reason.

### Phase 4 — Design-coherence vs the ticket / stated intent
**Goal:** does the implementation match what the PR/ticket CLAIMS, at the right altitude?
**How:**
- Compare impl to the ticket's scope and "Done when". Divergence is often *improvement* — say so — but isolate the one real gap (e.g. ticket promised "all multi-target routes" but impl is stage-router-only).
- Altitude: special cases on shared infra, wire-format parsing leaking into the policy layer, tuning-data-masquerading-as-config → recommend generalizing.
- Defend **intentional** asymmetries instead of manufacturing a finding.
- Scope creep / mixed concerns / gratuitous churn → call out per commit-discipline.

**Exit gate:** a coherence verdict (matches / improves-on / diverges-from) with the one real gap isolated.

### Phase 4b — Production-doc structural review (when a `*.md` design/production doc changed)
**Goal:** a credible, roughly-halved doc — not a render-only pass.
1. **Lead with the credibility killer:** find the contradiction first (conflicting numbers / one story told several ways). A self-contradicting doc is untrustworthy regardless of prose.
2. **Produce a CUT / FIX / KEEP plan with an explicit line budget**, targeting ~half: CUT fragile ASCII diagrams, derivable knob tables, drift-prone calibration dumps; FIX stale references deleted in this PR; KEEP the production core.
3. Render bugs and conflicting numbers are Important/Minor; the CUT/FIX/KEEP plan is the deliverable.

**Exit gate:** a CUT/FIX/KEEP plan with a target line count and the lead credibility-killer named.

### Phase 5 — Test-quality pass
**Goal:** catch tests that look rigorous but exercise nothing, and the missing core-path test.
**Headline gate — the deletion/mutation test:** a test survives only if **deleting or mutating the branch it claims to cover makes it fail.** Apply this first to every changed test.
- **Vacuous / passes-for-wrong-reason:** the "clip" test whose weight sum is already in range; a test pinned so the gated branch is never reached.
- **Claims a path it doesn't run:** a "concurrent" test looping sequentially.
- **Framework-only / tautological:** asserting `return request` is unmutated; `confidence == abs(score)`; a "rejects unknown picker" actually caught by a pydantic `Literal` a layer earlier.
- **Over-enumeration (AI bloat):** collapse 4 near-identical "rejects bad key" to ~3; name it, don't post each.
- **Misleading name:** `..._dispatches_and_translates` on an identity case.
- **The biggest gap:** is the *core value path* exercised end-to-end?
- Don't demand Rust unit tests for one-line PyO3 delegations — testing through Python matches repo style.

**Exit gate:** each flagged test names the wrong-reason it passes (or the gap) and survives/fails the deletion test, with a concrete fix.

### Phase 6 — Anti-slop filter
**Goal:** drop low-value noise; enforce brevity. Run on every surviving finding AND your own comment bodies. See [`references/anti-slop-checklist.md`](references/anti-slop-checklist.md).
**Drop / fold-into-summary if the comment:** restates a type-system guarantee (pydantic `Literal`/type hint); asserts an identity/tautology; duplicates a guard a layer earlier; restates the obvious; is an over-enumeration item; is pre-existing/out-of-diff; was retired by real-world context.
**Enforce on bodies:** SHORT and direct; no thinking-out-loud or "wait… actually…"; no project-management refs; uncertain intent → frame as a question ("Intentional?"); never claim reasoning as tested. **Do flag** the author's own thinking-out-loud comments left in the diff (post as a Minor "drop this stray comment").

**Exit gate:** a tight, post-able set; everything dropped has a one-line reason.

### Phase 7 — Severity triage & dedup
Assign Blocking/Important/Minor/Style (+ TYPE tag), merge duplicates, order most-severe-first (correctness before cleanup). Map any P0/P1/P2 from Rust mode onto the word scale.

**Exit gate:** a single ranked list, no duplicates.

### Phase 8 — Calibrate & present (draft first)
Structure as **Verdict → What this PR does → Verified (with `file:line`) → Findings (ranked, severity-tagged, one-line fix each) → Style (folded) → Bottom line.** Separate **blocking/correctness** from **latent/nice-to-have**. When asked "these are nothing serious right?", answer directly — don't hedge. Offer: post all / a curated subset / hold and hand a summary. **Do not post yet.**

**Exit gate:** user has the draft and has chosen what to post.

### Phase 9 — Post (explicit request only)
Post exactly the approved subset, anchored to the current head SHA, never auto. Verify each post landed **by content** (re-fetch threads), not by trusting the API response. Resolve only your OWN threads, after verifying the fix in code ("commit messages aren't proof"). Never touch or submit the user's pending review.

**Exit gate:** approved comments posted, each on the right line; nothing else changed.

---

## Posting mechanics

```bash
N=<PR>
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
SHA=$(gh pr view "$N" --json headRefOid -q .headRefOid)   # always fresh — head moves
BODY_FILE=$(mktemp)
printf '%s' 'short, direct, present-tense body' > "$BODY_FILE"
post() { gh api "repos/$REPO/pulls/$N/comments" -f commit_id="$SHA" -f path="$1" -F line="$2" -f side=RIGHT -F body=@"$3" --jq '.html_url'; }
post "<file>" <line> "$BODY_FILE"
```
`-f` = raw string (`commit_id`/`path`/`side`); `-F` = typed field (`line`, `body=@file`). Use temp-file bodies by default; inline `-f body="..."` only for a short single-line ASCII body with no shell metacharacters.

**Anchoring learned the hard way:** modified files are commentable only on lines inside the diff hunk; very large single-file additions can return `thread:null` → re-anchor on a smaller consuming file/line, never probe-spam. **One pending review per user** — if the user has an open pending review, append via GraphQL `addPullRequestReviewThread` (don't submit) or hand paste-ready text; standalone REST only after they submit. Verify by content: `gh api repos/$REPO/pulls/$N/comments --jq '.[] | "\(.path):\(.line) | \(.user.login)"'`.

**Other modes:** approve → `gh pr review <N> --approve --body-file file` (only when asked); long doc review → one summary comment at line 1; PR body edits → `gh api -X PATCH /repos/$REPO/pulls/<N> -F body=@file`.

**Never:** auto-post, post the full agent dump, retry a failing mutation into the user's live review, or fix code when only comments were requested.
