# NeMo Relay: Known Issues

Gaps between the [NeMo Relay Recipe](nemo_relay.md) and a fully routed
interactive Claude Code session. Verified 2026-07-15 against libsy `main`
from a fresh public checkout.

| Step | Status |
|---|---|
| Fresh clone builds hermetically; Path A e2e passes | works |
| Claude Code TUI launches; all traffic flows through the gateway and plugin | works |
| Requests carrying `cache_control` reach the router | works |
| Buffered requests routed by the libsy classifier | works |
| Streamed requests routed by libsy | **breaks** |
| Classifier scoring call accepted by Anthropic | **breaks** |

## 1. libsy has no streaming response path

`CallLlmRequest::respond` and `Step::ReturnToAgent` carry only buffered
responses, so a host cannot pass a live token stream through an algorithm.
Claude Code streams its interactive calls, so they dispatch the trusted
fallback (`libsy_streaming_unsupported`) instead of a libsy decision. Same
prompt, sent both ways through Path A:

```text
buffered -> classifier scored 0.9 -> served by strong-model
streamed -> no classifier call    -> served by weak-model (fallback)
```

**Owner:** libsy. Streaming exists on the unmerged `grclark/simple-proxy`
branch; when it lands, this fixes with no recipe changes.

## 2. Classifier scoring call rejected upstream (HTTP 400)

Root cause proven by capturing the request the plugin sends:
`LlmClassifierOrchAlgo` sets no output params (`..LlmRequest::default()`,
`llm_class.rs`), and the Anthropic encoder injects a hardcoded default of
`max_tokens: 128000` when unset (`codecs/anthropic/buffered.rs:207`). Haiku
4.5's output ceiling is 64,000, so Anthropic rejects the scoring call with
HTTP 400 and the whole request falls back (via issue 3).

```json
{"model": "claude-haiku-4-5-20251001", "max_tokens": 128000, "messages": [...]}
```

**Owner:** switchyard-translation (remove or clamp the 128k default) and libsy
reference algorithms (set explicit small output params; a score needs a few
tokens). Not a Relay issue.

## 3. No fail-open on classifier call errors

`LlmClassifierOrchAlgo` fails open only on an unparseable score. A classifier
call *error* propagates and kills the run, so the whole request falls back
instead of routing strong.

**Owner:** libsy reference algorithms.

## 4. Provider errors are opaque in routing events

The Relay plugin's `provider_error_summary` formats upstream failures as
`class:http_status` and discards the response body. Issue 2 had to be
root-caused by capturing the request instead of reading the logs.

**Owner:** Relay plugin. Not a libsy issue.

## 5. Classifier prompt includes system boilerplate (minor)

`LlmClassifierOrchAlgo` builds its scoring prompt by flattening all text in
the request, so agent system content (billing headers, system reminders)
precedes the user's question. Does not fail, but skews scores.

**Owner:** libsy reference algorithms.

## Toolchain note

Relay pins rustc 1.93.0; Switchyard's MSRV is 1.96.1. Prefix Relay builds with
`RUSTUP_TOOLCHAIN=1.96.1` until the toolchains converge.
