# NeMo Relay Recipe

Run Switchyard **as a library** inside [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay):
routing decisions are made in-process by [libsy](../../crates/libsy/README.md)
while Relay owns the proxy, dispatch, credentials, and observability. No
`switchyard serve` process is involved.

libsy never performs a network call: each offloaded `CallLlm` promise is
fulfilled by Relay's dispatch chain and answered with
`CallLlmRequest::respond(Ok/Err)`. libsy's semantic target names map onto the
Relay plugin's `TargetBinding` table, which binds each name to a backend URL,
model, protocol, and credentials. Buffered and streamed requests are both
routed; keep every target on the agent's protocol (as below) so streams pass
through without translation.

## Prerequisites

- Rust 1.96.1 or later
- Claude Code, for the agent launcher path
- An OpenRouter API key, from the
  [OpenRouter keys page](https://openrouter.ai/keys)

## Install

Build the Relay CLI with the Switchyard plugin (reference branch:
[feat/libsy-decision-backend](https://github.com/ryan-lempka/NeMo-Relay/tree/feat/libsy-decision-backend)):

```bash
git clone -b feat/libsy-decision-backend https://github.com/ryan-lempka/NeMo-Relay.git
cd NeMo-Relay
cargo build -p nemo-relay-cli --features switchyard
```

## Configure

Copy the two example configs into the project you will launch from, and
provide your OpenRouter key:

- [examples/nemo_relay/config.toml](../../examples/nemo_relay/config.toml) → `.nemo-relay/config.toml`
- [examples/nemo_relay/plugins.toml](../../examples/nemo_relay/plugins.toml) → `.nemo-relay/plugins.toml`

```bash
export OPENROUTER_AUTHORIZATION="Bearer $OPENROUTER_API_KEY"
./target/debug/nemo-relay doctor
```

The plugin config selects `decision_backend = "libsy"` with random routing
between a strong target (Opus) and a weak target (Sonnet), both served by
OpenRouter's Anthropic-compatible endpoint. Since Relay v0.7, target bindings
own their credentials: the agent's own login is never forwarded to a routed
target.

## Path A: Gateway mode

Runs Relay's gateway with libsy routing and a deterministic fake provider; no
credentials needed:

```bash
./examples/switchyard/run-libsy-e2e.sh
```

Expected receipt:

```text
ok: easy prompt routes weak
ok: hard prompt routes strong
ok: streamed hard prompt routes strong
ok: classifier and routed calls all flowed through Relay dispatch (6 upstream calls)
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
directory, one mark per decision:

```bash
grep -h -o '"backend_id":"[a-z]*"' nemo-relay-events-*.jsonl | sort | uniq -c
```

Expected receipt: `switchyard.routing.decision` marks naming `strong` and
`weak`, with responses served by both models across the session. A decision
naming a target is the proof: libsy selected it in-process and Relay
dispatched the routed call.

## The embedding contract

A host needs three things from libsy:

- `Algorithm`: one shared `Arc<dyn Algorithm>` serves concurrent requests.
- `run_stream(ctx, request)`: a stream of `Step`s; the host serves each
  `Step::CallLlm` and fulfills it with `respond(...)` — a buffered response
  or a live chunk stream.
- The conversation IR (`LlmRequest`, `AggLlmResponse`, `LlmResponseChunk`
  from `switchyard-protocol`), which Relay already speaks through
  `switchyard-translation`.

See the [libsy README](../../crates/libsy/README.md) for the full API.
