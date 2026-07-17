# Switchyard changelist — v0.1.0 → main

**Provenance**
- v0.1.0 = commit `cb2940480d92f0296e16f4e50c1bea9daebdb789` (annotated tag `v0.1.0`, tagged 2026-06-30 by nachiketb; confirmed against origin, local, PyPI upload date, and NVBugs tickets 6401773/6401772).
- Compared against `origin/main` @ `694aa27d8df4d7b73f699d43073a9ece1654aea5` (fetched 2026-07-16).
- Range: `git log cb2940480d92f0296e16f4e50c1bea9daebdb789..694aa27d8df4d7b73f699d43073a9ece1654aea5` — 30 squash-merged PRs.
- Method: three independent derivations reconciled — Path A (code diff), Path B (commit messages), Path C (PR bodies + Linear tickets). Where paths disagreed, the code (Path A) was the tie-breaker. See the reconciliation notes at the bottom.

Coverage: 30/30 — all of #6,#17,#20,#22,#24,#25,#28,#29,#34,#37,#40,#43,#45,#52,#53,#54,#58,#62,#64,#65,#66,#67,#72,#73,#76,#78,#80,#81,#83,#84 are represented (breaking items are additionally cross-referenced under their category, so a few appear more than once). None missing.

---

## [0.2.0] - 2026-07-16

_30 PRs, +17.5k/-4.6k across 238 files since v0.1.0 / cb294048 (2026-06-30). Highlights: the cascade→stage-router rename, RouteLLM removal, the new escalation_router + shared Redis session-affinity, per-user spend attribution, and the new libsy / switchyard-protocol Rust crates._

### Breaking changes

- Rename `cascade` router to `stage_router` and its `strong`/`weak` tiers to `capable`/`efficient` — no compatibility shims; existing `type: cascade` configs, picker names, imports, recipes, and dashboard keys must be updated (#20)
- Remove RouteLLM routing strategy, the `routellm` route type/inference, and the `[gpu]` optional extra (~800MB); strong+weak+classifier configs now fail key validation instead of silently inferring `routellm` (#84)
- Advertise only target names (e.g. `classifier`/`weak`/`strong`), not the underlying upstream model ids; requests using the raw upstream model name now return ModelNotFound (#67)
- Remove `serve_addr`/`serve_addr_with_backlog` and narrow switchyard-server visibility so only `run_server` (and `build_switchyard_router`) stay public (#52)
- libsy: make `LlmRequest`/`LlmResponse` aliases over the shared conversation IR, dropping the bespoke prompt/completion fields from the crate's public API (#58 — SWITCH-620)
- libsy/switchyard-protocol: add streaming responses; rename `LlmResponse`→`AggLlmResponse` and `LlmStreamEvent`→`LlmResponseChunk` and thread a `ctx` param through the producer API (also renames the translation IR) (#73 — SWITCH-949)
- libsy: move `Decision`/`RoutedLlmClient` into switchyard-protocol, rename `LlmClient`→`RoutedLlmClient`, and thread request `ctx` through `call()` (#76)

_Note: #58/#73/#76 affect the pre-1.0, unpublished libsy/switchyard-protocol crates (no external consumers yet). Path A additionally flags the Rust min-toolchain bump to 1.96.1 (#25) as a build-time break for contributors on older rustc._

### Added

- Configure a skill-distillation namespace (`switchyard configure --skill-distillation NAME` / `--disable-skill-distillation`) and land the source-neutral `switchyard-skill-distillation` Rust contract crate; config-only this release (no capture/distill/load yet) (#6 — SWITCH-626)
- New `libsy` crate: embeddable, provider-agnostic multi-LLM routing/orchestration library ("ask, don't call") plus reference algorithms and runnable example agents (#17 — SWITCH-923)
- Report per-model `max_observed_context_tokens` (prompt+completion high-water mark) in `/v1/stats` (#24)
- Latency-router parity batch: `caller_required` credential policy (401 when caller key absent), per-endpoint `request_type`, per-model upstream-attempt metrics with `route_model` attribution, error-source/upstream-model response headers, verbatim OpenAI Responses passthrough, and an optional shared Redis L2 session-affinity pin store (`[affinity-redis]`) with fail-open circuit breaker (#28)
- Terminal-Bench 2.1 dataset support in the Harbor preparer (#45)
- Optional TLS/HTTPS termination for switchyard-server via `--tls-cert`/`--tls-key` (#53)
- Route-selection spend attribution: outbound `x-litellm-spend-logs-metadata` header + `x-switchyard-*` response headers to join provider spend rows to the logical route (#54 — SWITCH-914)
- New `escalation_router` profile: judge-latched one-way weak→strong escalation with session-affinity latch (#62)
- Serve concurrent model calls in libsy's `Algorithm::run` (bounded fan-out) (#66)
- Streaming response support in libsy (#73 — SWITCH-949) _(also a breaking rename; see above)_
- No-op routing algorithm for libsy (isolation/perf/integration testing) (#78)

### Changed

- Profile `type` discriminator now folds `-` and `_` as equivalent — see also Fixed (#22)
- Startup banner prints ready-to-run curl for `/v1/messages` and `/v1/responses` (and HTTPS-aware URLs when TLS is on), max_tokens bumped 8→256 (#65)
- libsy IR alias refactor (#58 — SWITCH-620), streaming (#73), and protocol-boundary cleanup (#76) _(all breaking; see above)_

### Fixed

- LLM classifier / v2 profile configs spelled with snake_case (e.g. `type: random_routing`) no longer fail with "unknown profile type" (#22)
- Fix the docs LLM classifier-routing example that registered a model twice and failed at startup (#37)
- Calculate token-usage stats for streaming responses (`/v1/routing/stats` streaming zero-usage bug), wired into all four profiles; latency reported at first token (#64 — closes #55, #18)
- Advertise only target names, not the underlying model directory — also fixes the `SwitchyardDuplicateRegistrationError` raised when one upstream model backs two targets (e.g. classifier + weak) (#67 — closes #51) _(breaking; see above)_
- Anthropic streaming: preserve `stop_reason` at `message_stop` instead of dropping it (#73)

### Removed

- Drop the detect-secrets pre-commit hook and CI job in favor of GitHub secret scanning (#43)
- Remove two unused server functions and tighten switchyard-server visibility (#52) _(breaking; see above)_
- Remove RouteLLM routing support and its `[gpu]` dependency footprint (#84) _(breaking; see above)_

### Internal

- Upgrade the pinned Rust toolchain 1.94.0 → 1.96.1 across workspace and benchmark build (#25)
- Add root CODEOWNERS requiring @NVIDIA-NeMo/switchyard_team review; repoint the issue-template discussions link to the public repo (#29)
- Enable the LLM and codex_cli conflict-resolution tiers in the sir-merge-a-lot config (#34)
- Add an external-contribution quick guide to CONTRIBUTING.md (plus DCO-required and 404-link fixes) (#40)
- Prepare `switchyard-protocol` and `switchyard-translation` crates for their first crates.io publish (packaging metadata, README/LICENSE/NOTICE; publishes nothing) (#72 — SWITCH-456)
- Give `Algorithm::process_signals` a default no-op impl (#80)
- Promote `RandomAlgo` uniform-random routing from libsy-examples into core libsy (#81)
- Bump benchmark harness agent pins (Claude Code 2.1.211, Codex 0.144.5, OpenCode 1.18.3) (#83)

---

## Reconciliation notes (resolved via GitHub / Linear / Slack / git research — 2026-07-17)

All 16 cross-path discrepancies were researched against primary sources (PR bodies, Linear tickets, Slack, and the range diff) and adversarially verified. **12 confirmed the original reconciliation; 4 corrected it** (#5, #9, #13, #16); the 5 unattributed changes are now attributed to specific PRs. ✎ marks the items that changed the changelog above.

1. **#22 (Changed vs Fixed) — VERIFIED Fixed.** PR title `fix:`; `parse_profile_config` (profiles/macros.rs) + a regression test repair a real serve-startup abort ("unknown profile type") for snake_case v2 `type` names. (85996980)
2. **#25 breaking flag — VERIFIED Internal.** Only three pinned-version lines (`Cargo.toml` rust-version 1.94→1.96.1, rust-toolchain.toml, benchmark Dockerfile ARG); no code/API change. A real build-from-source bump but not a changelog "breaking." Kept Internal with the build-time note. (e6ce3ef8)
3. **#34 "missing from Path A" — VERIFIED present in diff.** Commit 08490b4a (`.sir-merge-a-lot.yml`, +17/-5) *is* in the range; Path A dropped it during synthesis — it was never actually absent.
4. **#37 (Fixed vs Internal/docs) — VERIFIED Fixed (docs).** PR "Fixes #36"; the example registered `gpt-4o-mini` twice and failed at startup with `SwitchyardDuplicateRegistrationError`. A real defect fix in a docs example, not tidying. (717391ac)
5. **#40 attribution — CORRECTED ✎.** #40 is Internal (docs-only). The issue-template discussions-URL fix (`config.yml`: `NVIDIA/switchyard` → `NVIDIA-NeMo/Switchyard`) belongs to **#40** (c7bd0a3e), **not #29** — #29 (266fd33f) only edits CODEOWNERS. (Prior note wrongly folded it into #29.)
6. **#52 (Removed+breaking vs Internal) — VERIFIED Removed+breaking.** At v0.1.0 `serve_addr`/`serve_addr_with_backlog` were `pub async fn` (lib.rs:147/152); #52 (1a98edb5) removes both and privatizes `serve`/`state_from_config_path`, leaving only `run_server` public. Pre-1.0 crate, so no external consumer breaks, but the public-API removal is real.
7. **#64 (Added vs Fixed) — VERIFIED Fixed.** PR `fix:`, "Fixed the /v1/routing/stats streaming zero-usage bug" + repro test, closes #55/#18; author's Slack daily log calls it a bug fix. (efbf0517)
8. **#65 (Changed vs Added) — VERIFIED Changed.** Appends two curl lines to the existing startup banner; no new feature. (c9950776)
9. **#67 (Changed vs Fixed) — CORRECTED → Fixed ✎.** PR `fix:` (487fa4c8), "Closes #51", "removes the bug that prevents using the same model for classifier and weak"; CodeRabbit + author's Slack log both file it under Bug Fixes. A breaking fix is still a fix — moved from Changed to Fixed (stays breaking). GH #51 (`SwitchyardDuplicateRegistrationError`) confirmed closed.
10. **#72 (Changed vs Internal) — VERIFIED Internal.** Body: "only prepares the packages… does not publish either crate"; diff touches no CI/publish. SWITCH-456 ("Publish first Rust crate versions", Backlog) is the actual publish. (8675f8c8)
11. **#73 breaking flag — VERIFIED breaking.** Diff of c7a6475f confirms `LlmResponse` struct → `AggLlmResponse`, `LlmStreamEvent` → `LlmResponseChunk` (relocated to stream.rs), and `ctx: Context` threaded through the producer API. (The git-archaeology pass's "renames are false" claim was itself wrong.) Pre-1.0 crate.
12. **#76 breaking flag — VERIFIED breaking** (by direct git inspection; the workflow resolver errored out). Diff of 694aa27d confirms trait `LlmClient` → `RoutedLlmClient`, `call()` gains a `ctx: Context` param, and `Decision`/`RoutedLlmClient` move to the new `libsy-protocol/src/client.rs` (+53). Pre-1.0 crate.
13. **#78 method-rename claim — CORRECTED: claim is FALSE ✎.** Commit 8a14f966 is purely additive (135 insertions, 0 deletions): it adds new `Request::requested_model()` / `Response::selected_model()` accessors; `git log -S "fn model("` over the range is empty — nothing was renamed. #78 is correctly non-breaking.
14. **#80/#81/#83 (Changed vs Internal) — VERIFIED Internal.** #80 trait default no-op impl, #81 moves `RandomAlgo` into core libsy, #83 bumps benchmark harness pins — no user-facing runtime change.
15. **Unattributed diff changes — ATTRIBUTED ✎.** Per git `-S`/`-G` over the range: **(a)** `stop_reason`-at-`message_stop` → **#73** (c7a6475f) — a real user-facing streaming fix, now surfaced as its own Fixed line; **(b)** `session_key_from_body` introduced in **#28** (6a9942ce), the depth param + Python `str`→`Optional[str]` return added later in **#62** (b03b6a6f); **(c)** the `strong.id != weak.id` validator → **#62**; **(d)** `content_text` tool_use flattening → **#62**; **(e)** `x-switchyard-api-key` header + credential redaction → **#28**. None originate from #54. (b)–(e) are internal plumbing legitimately folded into #28/#62.
16. **Linear linkage — CORRECTED ✎.** #6→SWITCH-626 is topical inference (no literal id in body/branch); SWITCH-626 is the skill-distillation *epic*, and PR #6 is actually attached to child tickets SWITCH-800/801. #58→SWITCH-620 is **explicitly cited** in the PR body ("Linear: SWITCH-620") and #58 is attached on that ticket — but SWITCH-620 is a 24-hr stress-test task (label Test), not a dedicated IR-refactor ticket; the PR was piggybacked onto a topically-mismatched ticket, not loosely inferred. #72→SWITCH-456 confirmed ("Related: SWITCH-456", the publish task). PRs with a clear Linear ticket: #6 (SWITCH-800/801, epic 626), #17 (SWITCH-923), #54 (SWITCH-914), #58 (SWITCH-620), #62 (SWITCH-944), #72 (SWITCH-456), #73 (SWITCH-949).
