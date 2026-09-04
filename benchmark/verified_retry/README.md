# Verified-retry orchestrator

A production-legal best-of-N harness: run an executor agent up to N times in
fresh workspaces, verify each attempt WITHOUT any benchmark answer key, stop
at the first verified success, and fall back to a comparative LLM pick when
nothing verifies.

## Why this shape (measured, TB-2.1 Kimi-K3-Max study, 2026-08)

- Retries are the dominant accuracy lever: single-attempt 70.5% vs oracle
  pass@4 83.1% (K3 baseline, 4 replicate runs). No advisor/prompt config
  moved the single-attempt mean.
- An LLM judge alone is NOT verifier-strength: 82.5% verdict accuracy with a
  35.6% false-pass rate on truly-failing attempts. Every false-pass came
  from trusting narrated (not executed) evidence.
- Judge confidence separates truth: true-pass mean 0.83 vs false-pass 0.69;
  gating acceptance at >=0.85 removed 14/16 false-passes.
- Comparative picking is much stronger than absolute judging: 87% correct
  picks (vs 62.5% random), implying ~78.7%/89 for judge-selected best-of-4 —
  ≈ a frontier model's single-attempt score, with no oracle.

## The three verification layers

1. `checks.py` — an LLM derives an executable check script from the task
   statement ONLY, BEFORE any attempt exists (prevents inheriting a
   solution's misreading), and the script is executed in the workspace.
   Execution outranks opinion; where the statement is machine-checkable this
   layer is benchmark-verifier-strength by construction.
2. `judge.py::judge_attempt` — confidence-gated LLM verdict on demonstrated
   evidence; veto layer over checks, sole verifier when no checks exist.
   Unparseable verdicts fail closed.
3. `judge.py::judge_compare` — when no attempt verifies within budget, a
   comparative pick over archived candidates, explicitly flagged unverified.

## Boundaries

- This is NOT a Switchyard route/strategy: the gateway never touches the
  workspace, and executing checks requires the workspace. It wraps the
  executor where the sandbox lives.
- Executor and LLM are injected callables (`spec.ExecutorFn`, `spec.LlmFn`) —
  no provider SDK, no docker required for the core. Container/TB-image
  integration is a follow-up on a docker-capable host.
- Benchmark scores produced with this harness are pass@N-with-own-verifier;
  report them as such, never as single-attempt accuracy.

Run tests: `uv run pytest benchmark/verified_retry/tests -q`
