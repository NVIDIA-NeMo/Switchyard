# NeMo Relay Recipe

Run Switchyard **as a library** inside [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay):
routing decisions are made in-process by [libsy](../../crates/libsy/README.md)
while Relay owns the proxy, dispatch, credentials, and observability. No
`switchyard serve` process is involved.

libsy never performs a network call: each offloaded `CallLlm` promise is
fulfilled by Relay's dispatch chain and answered with
`CallLlmRequest::respond(Ok/Err)`. libsy's semantic target names map onto the
Relay plugin's `TargetBinding` table, which binds each name to a backend URL,
model, and protocol.

Current gaps are tracked separately in
[NeMo Relay: Known Issues](nemo_relay_known_issues.md).

## Prerequisites

- Rust 1.96.1
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
credentials needed:

```bash
RUSTUP_TOOLCHAIN=1.96.1 ./examples/switchyard/run-libsy-e2e.sh
```

Expected receipt:

```text
ok: easy prompt routes weak
ok: hard prompt routes strong
ok: classifier and routed calls both flowed through Relay dispatch (4 upstream calls)
libsy in-process routing e2e passed
```

## Path B: Agent launcher

From the directory containing `.nemo-relay/`:

```bash
./target/debug/nemo-relay claude
```

This opens the normal interactive Claude Code TUI through the Relay gateway.

## Validate

Routing decisions land in `nemo-relay-events-*.jsonl` in the working
directory. One line per decision:

```bash
grep -h switchyard.routing.decision nemo-relay-events-*.jsonl
```

Expected receipt: `switchyard.routing.decision` marks naming the classifier
step and the routed target, e.g.

```json
{"backend_id": "classifier", "reason_summary": "classifying request via classifier", ...}
{"backend_id": "weak", "target_model": "claude-sonnet-5", ...}
```

A decision naming `weak` or `strong` is the proof: libsy scored the request
in-process and Relay dispatched the routed call.

## The embedding contract

A host needs three things from libsy:

- `Algorithm`: one shared `Arc<dyn Algorithm>` serves concurrent requests.
- `run_stream(ctx, request)`: a stream of `Step`s; the host serves each
  `Step::CallLlm` and fulfills it with `respond(...)`.
- The conversation IR (`LlmRequest` / `LlmResponse` from `libsy-protocol`),
  which Relay already speaks through `switchyard-translation`.

See the [libsy README](../../crates/libsy/README.md) for the full API.
