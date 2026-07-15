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

## Switchyard / libsy improvements

### S1. No streaming response path (libsy core)

`CallLlmRequest::respond` and `Step::ReturnToAgent` carry only buffered
responses, so a host cannot pass a live token stream through an algorithm.
Claude Code streams its interactive calls, so they dispatch the trusted
fallback (`libsy_streaming_unsupported`) instead of a libsy decision. Same
prompt, sent both ways through Path A:

```text
buffered -> classifier scored 0.9 -> served by strong-model
streamed -> no classifier call    -> served by weak-model (fallback)
```

Streaming exists on the unmerged `grclark/simple-proxy` branch; when it
lands, this fixes with no recipe changes.

### S2. Anthropic encoder defaults `max_tokens` to 128,000 (switchyard-translation)

When a request sets no output params, the encoder injects
`max_tokens: 128000` (`codecs/anthropic/buffered.rs:207`). Haiku 4.5's
ceiling is 64,000, so Anthropic rejects the classifier's scoring call with
HTTP 400 and the whole request falls back (via S3). Proven by capturing the
sent request:

```json
{"model": "claude-haiku-4-5-20251001", "max_tokens": 128000, "messages": [...]}
```

Fix pair: clamp or remove the encoder default, and have
`LlmClassifierOrchAlgo` set explicit small output params (a score needs a
few tokens).

### S3. No fail-open on classifier call errors (libsy-examples)

`LlmClassifierOrchAlgo` fails open only on an unparseable score. A classifier
call *error* propagates and kills the run, so the whole request falls back
instead of routing strong.

### S4. Classifier prompt includes system boilerplate (libsy-examples, minor)

The scoring prompt flattens all text in the request, so agent system content
(billing headers, system reminders) precedes the user's question. Does not
fail, but skews scores.

## NeMo Relay improvements

### R1. Provider errors are opaque in routing events

The plugin's `provider_error_summary` formats upstream failures as
`class:http_status` and discards the response body. S2 had to be root-caused
by capturing the request instead of reading the logs. Fix: include a bounded
body snippet in error marks.

### R2. Toolchain pin below Switchyard's MSRV

Relay pins rustc 1.93.0; Switchyard's MSRV is 1.96.1. Prefix Relay builds
with `RUSTUP_TOOLCHAIN=1.96.1` until Relay bumps its pin (or Switchyard
lowers its MSRV).
