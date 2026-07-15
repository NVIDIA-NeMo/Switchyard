# NeMo Relay Recipe

Run Switchyard **as a library** inside [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay):
routing decisions are made in-process by [libsy](../../crates/libsy/README.md)
while Relay owns the proxy, dispatch, credentials, and observability. No
`switchyard serve` process is involved.

| Mode | Decision path | Switchyard process |
|---|---|---|
| Server (`topic/nemo-relay-integration`) | Relay POSTs to the HTTP Decision API | required |
| Library (this recipe) | Relay calls libsy in-process | none |

libsy never performs a network call: each offloaded `CallLlm` promise is
fulfilled by Relay's dispatch chain and answered with
`CallLlmRequest::respond(Ok/Err)`. libsy's semantic target names map onto the
Relay plugin's `TargetBinding` table, which binds each name to a backend URL,
model, and protocol.

## Prerequisites

- Rust 1.96.1 (Switchyard's MSRV; Relay pins 1.93.0, so prefix builds with
  `RUSTUP_TOOLCHAIN=1.96.1` until the toolchains converge)
- Claude Code, for the agent launcher path

## Install

Build the Relay CLI with the Switchyard plugin (reference branch:
[feat/libsy-decision-backend](https://github.com/ryan-lempka/NeMo-Relay/tree/feat/libsy-decision-backend)):

```bash
git clone -b feat/libsy-decision-backend https://github.com/ryan-lempka/NeMo-Relay.git
cd NeMo-Relay
RUSTUP_TOOLCHAIN=1.96.1 cargo build -p nemo-relay-cli --features switchyard
```

## Configure

Copy the two example configs into the project you will launch from:

- [examples/nemo_relay/config.toml](../../examples/nemo_relay/config.toml) → `.nemo-relay/config.toml`
- [examples/nemo_relay/plugins.toml](../../examples/nemo_relay/plugins.toml) → `.nemo-relay/plugins.toml`

The plugin config selects `decision_backend = "libsy"` with the LLM-classifier
algorithm: a Haiku target scores each request, and the score routes to an Opus
(strong) or Sonnet (weak) target. Claude Code's own auth headers pass through
to Anthropic. Verify with:

```bash
./target/debug/nemo-relay doctor
```

## Path A: Gateway mode

Runs Relay's gateway with libsy routing and a deterministic fake provider; no
credentials needed. This is the self-contained e2e shipped on the Relay branch:

```bash
RUSTUP_TOOLCHAIN=1.96.1 ./examples/switchyard/run-libsy-e2e.sh
```

It verifies an easy prompt routes weak, a hard prompt routes strong, and the
classifier and routed calls both flow through Relay dispatch. Test by hand
with curl against `http://127.0.0.1:4042/v1/chat/completions`.

## Path B: Agent launcher

From the directory containing `.nemo-relay/`:

```bash
./target/debug/nemo-relay claude
```

This opens the normal interactive Claude Code TUI through the Relay gateway.
Inspect routing decisions afterwards:

```bash
grep switchyard.routing nemo-relay-events-*.jsonl
```

## Status against libsy `main` (verified 2026-07-15)

| Step | Status |
|---|---|
| Fresh clone builds hermetically; Path A e2e passes | works |
| TUI launches; all traffic flows through the gateway and plugin | works |
| Requests carrying `cache_control` reach the router | works |
| Buffered requests routed by the libsy classifier | works |
| Streamed requests routed by libsy | **breaks** |

The break, demonstrated with the same prompt sent both ways through Path A:

```text
buffered "This is a hard problem: ..."  -> classifier scored 0.9 -> served by strong-model
streamed "This is a hard problem: ..."  -> no classifier call    -> served by weak-model (trusted fallback)
```

The cause is libsy's response contract on `main`: `CallLlmRequest::respond`
and `Step::ReturnToAgent` carry only buffered responses, so no host can pass
a live token stream through an algorithm. Claude Code streams its interactive
calls, so those requests emit `switchyard.routing.error` (`libsy_stream`) and
dispatch the trusted per-protocol fallback (`libsy_streaming_unsupported`)
instead of a libsy decision. Streaming support exists on the unmerged
`grclark/simple-proxy` branch; when it lands, the last table row flips with no
config changes.

## The embedding contract

A host needs three things from libsy:

- `Algorithm`: one shared `Arc<dyn Algorithm>` serves concurrent requests.
- `run_stream(ctx, request)`: a stream of `Step`s; the host serves each
  `Step::CallLlm` and fulfills it with `respond(...)`.
- The conversation IR (`LlmRequest` / `LlmResponse` from `libsy-protocol`),
  which Relay already speaks through `switchyard-translation`.

See the [libsy README](../../crates/libsy/README.md) for the full API.
