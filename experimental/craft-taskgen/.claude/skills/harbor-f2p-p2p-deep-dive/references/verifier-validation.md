# Verifier Validation Questions

Before accepting an **F2P** test failure as a genuine capability gap, apply these questions. Each was learned from a real false negative.

These questions are tuned for F2P (FAIL_TO_PASS) tests — the ones constructed to validate the requested feature, where false negatives are most common. The default verdict for P2P (PASS_TO_PASS) failures is regression (capability gap); apply these questions to a P2P only in the rare-exception cases listed in the parent SKILL.md "Test classes: F2P vs P2P" section.

## 1. Does the instruction specify what the test checks?

Read the assertion. Search the instruction for the corresponding requirement. If the test checks behavior the instruction never mentions (error-handling conventions, exit patterns, output formatting), it's an unstated assumption.

*Example: a test expected `SystemExit` on bad paths but the instruction said nothing about error handling. Fix: extend instruction with "invalid loop paths should log an error and exit the process."*

## 2. Does the instruction's wording match the test scope?

Instructions can accidentally narrow scope below what tests exercise. If the instruction says "X" but tests also require "Y," agents will reasonably implement only X.

*Example: instruction said "generators" but tests used `Iterable`/`AsyncIterable`. Both models implemented only generators.*

## 3. Did multiple models fail the same way?

If a strong model and a weak model both make the same "mistake," it's almost certainly an instruction or verifier issue. Genuine capability gaps produce *different* failure modes across tiers.

**Multi-trial generalization**: in cohorts with N trials × M models, applying this question to the universally-failing set (tests that fail in every complete-listing trial) provides evidence that scales with trial count. The "all models, all trials" pattern is suggestive — not conclusive — of an instruction or verifier issue, and still requires per-test triangulation in SKILL.md step 7 to confirm. Conversely, a test that some model passes on some trial is demonstrably solvable; capability-gap is a live hypothesis for that test regardless of how many trials failed it.

## 4. Would a correct alternative implementation fail?

Imagine a senior engineer who reads only the instruction (not the reference commit). Would their reasonable implementation pass this test? If not, the test is over-constrained for the instruction given.

## 5. Did passing tests pass for the right reason?

Don't only deep-dive failures — audit passes too. For each passing test, check whether the instruction actually guided the agent to the right design choice, or whether the agent made an arbitrary choice that happened to match. The way to check: read the instruction, then read the test's import/fixture assumptions. Is there a direct link (instruction says "put X in module Y," test imports X from Y)? Or is the match coincidental (instruction says "put X somewhere," agent guessed Y, test imports from Y)?

This matters because coincidental passes are fragile — they'll break if you change the instruction wording. If you update an instruction and a previously-passing test starts failing, the most likely cause is that the original pass was lucky, and the updated instruction nudged the agent to a different (equally valid) design choice that doesn't match the test.

### What to do with a coincidental pass

Once you've identified one, pick one of three actions:

1. **Tighten the instruction** — nail down the design choice the test assumes (preferred when the choice is genuinely required for the feature). The pass becomes principled, not lucky.
2. **Loosen the test** — rewrite the assertion against the public contract instead of the implementation detail (preferred when the test is over-specific and the design choice is arbitrary).
3. **Flag and accept** — if neither (1) nor (2) is in scope, mark the task as instruction-fragile in the report. Future re-runs may flip the verdict on this test, and reviewers should know the result is sensitive to instruction wording.

Pick (1) or (2) when you control the task; pick (3) when reporting on a frozen benchmark.
