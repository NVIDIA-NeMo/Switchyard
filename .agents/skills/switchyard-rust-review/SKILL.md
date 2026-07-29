---
name: switchyard-rust-review
description: Review Switchyard Rust changes for correctness and maintainability. Use for pull requests or diffs touching crates, PyO3 bindings, async runtime behavior, streaming, protocol types, translation, algorithms, or LLM clients.
---

# Switchyard Rust Review

Review the changed code and its live callers, not the entire repository. Findings lead, ordered by
severity and anchored to a file and line. Verify each candidate against the current code path before
reporting it; do not turn general Rust preferences into findings.

## Review Checks

- **Errors:** Production and test code must not use `.expect()`. Avoid `unwrap()` or panics on
  fallible paths; propagate typed errors. Convert every PyO3 boundary failure into `PyErr`.
- **Ownership:** Question `Arc`, `Mutex`, `Box`, and clones only when the code lacks a concrete
  ownership, sharing, trait-object, or concurrency requirement. Do not prescribe a wrapper by
  reflex.
- **Async:** Check for locks held across `.await`, blocking work on executor threads, unbounded
  queues, untracked spawned tasks, missing cancellation, and ignored backpressure.
- **FFI:** Preserve Python-visible types, defaults, exceptions, and lifetimes. Do not hold the GIL
  during blocking or awaited work.
- **Streaming:** Preserve event order, termination, partial chunks, usage accounting, and error
  frames across every supported wire format.
- **Boundaries:** Keep protocol types neutral, translation in translation code, LLM calls in
  clients, and routing behavior in algorithms or server wiring.
- **Observability:** Use structured `tracing` fields, appropriate levels, bounded metric labels,
  and no prompt content or credentials in logs.
- **Tests:** Require behavioral coverage for changed contracts, especially protocol/translation
  parity, streaming termination, errors, and Python/Rust boundaries. Avoid repetitive input
  enumeration.

## Validation

Select the narrowest test that exercises the changed contract, then run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p <affected-crate>
```

Use `cargo test --workspace` for shared protocol, public API, or cross-crate changes. Run affected
Python tests when PyO3 behavior changes. Report anything not run and why.
