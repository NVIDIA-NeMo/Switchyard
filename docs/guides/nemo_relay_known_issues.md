# NeMo Relay: Known Issues

What stands between the [NeMo Relay Recipe](nemo_relay.md) and a fully routed
interactive Claude Code session. Status verified 2026-07-15 against libsy
`main` from a fresh public checkout.

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
responses, so no host can pass a live token stream through an algorithm.
Claude Code streams its interactive calls, so those requests emit
`switchyard.routing.error` (`libsy_stream`) and dispatch the trusted
per-protocol fallback (`libsy_streaming_unsupported`) instead of a libsy
decision. Demonstrated with the same prompt sent both ways through Path A:

```text
buffered "This is a hard problem: ..."  -> classifier scored 0.9 -> served by strong-model
streamed "This is a hard problem: ..."  -> no classifier call    -> served by weak-model (trusted fallback)
```

**Owner:** libsy. Streaming support exists on the unmerged
`grclark/simple-proxy` branch (`LlmResponse` becomes buffered-or-stream); when
it lands on `main`, the Relay plugin can fulfill promises with live streams
and no recipe config changes are needed.

## 2. Classifier scoring call rejected upstream (HTTP 400)

In a live TUI session, Claude Code's buffered utility calls reached libsy and
produced classifier decisions, but the scoring call to Anthropic failed with
`invalid_request:http_400`. Leading suspect: `LlmClassifierOrchAlgo` builds
its request with `..LlmRequest::default()` (no output params) and the
Anthropic encoder defaults `max_tokens` to 128,000 when unset, above Haiku's
output ceiling.

**Owner:** switchyard-translation (encoder default) and/or libsy reference
algorithms (set explicit output params on synthesized calls).

## 3. No fail-open on classifier call errors

`LlmClassifierOrchAlgo` fails open only on an unparseable score. A classifier
call *error* propagates and kills the run, so the whole request falls back
(`decision_error`) instead of routing strong.

**Owner:** libsy reference algorithms.

## 4. Provider errors are opaque in routing events

The Relay plugin condenses upstream failures to summaries like
`invalid_request:http_400`, discarding the response body. Issue 2 had to be
root-caused from source instead of logs.

**Owner:** Relay plugin (include a bounded body snippet in error marks).

## Toolchain note

Relay pins rustc 1.93.0; Switchyard's MSRV is 1.96.1. Until the toolchains
converge, prefix Relay builds with `RUSTUP_TOOLCHAIN=1.96.1`.
