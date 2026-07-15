# NeMo Relay Recipe

This recipe shows how to run Switchyard **as a library** inside
[NeMo Relay](https://github.com/NVIDIA/NeMo-Relay), so routing decisions are
made in-process by [libsy](../../crates/libsy/README.md)
while Relay keeps ownership of the actual LLM calls. No `switchyard serve`
process is involved.

> **Status:** the buffered request path is validated end to end against
> NeMo Relay's `feat/libsy-decision-backend` branch (classifier call and
> routed call both fulfilled by Relay dispatch). Streaming requests are not
> yet routed by libsy; they dispatch Relay's trusted per-protocol fallback.

## Two ways to pair Switchyard with Relay

| Mode | Decision path | Switchyard process | Where it lives |
|---|---|---|---|
| Server | Relay POSTs to Switchyard's HTTP Decision API | Required (`switchyard serve`) | Relay `crates/switchyard` plugin, Switchyard `topic/nemo-relay-integration` |
| Library (this recipe) | Relay calls libsy in-process | None | `decision_backend = "libsy"` in Relay's Switchyard plugin |

Relay's Switchyard plugin README anticipated this: a future in-process
decision provider replacing the HTTP Decision API call without changing
Relay-owned dispatch and observability. `decision_backend = "libsy"` is that
provider.

## How the pieces fit

```text
agent (e.g. Claude Code)
   │  Anthropic / OpenAI wire protocol
   ▼
Relay gateway (local proxy)
   │  managed LLM call
   ▼
LLM execution intercept (Relay's Switchyard plugin)
   │
   ├─ libsy Algorithm decides which target to call
   │     Step::CallLlm ──▶ Relay fulfills via next(request) ──▶ provider
   │     Step::Decision ──▶ switchyard.routing.decision mark events
   │     Step::ReturnToAgent ──▶ final response
   ▼
response translated back to the inbound protocol
```

libsy never performs a network call. Each offloaded `CallLlm` promise is
fulfilled by Relay's own dispatch chain (`next(request)`), and the result is
handed back with `CallLlmRequest::respond(Ok/Err)`. The classifier's scoring
call is a Relay-managed provider call like any other, so it shows up in
Relay's observability. libsy's semantic target names (`"strong"`, `"weak"`,
`"classifier"`) map onto the plugin's existing `TargetBinding` table, which
binds each name to a Relay-owned backend URL, model, and protocol.

## Prerequisites

- Rust 1.96.1 (Switchyard's MSRV; note Relay pins 1.93.0, so build Relay with
  `RUSTUP_TOOLCHAIN=1.96.1` until the toolchains converge)
- NeMo Relay checkout on the `feat/libsy-decision-backend` branch
- Provider credentials for your targets (the demo below needs none)

## Quick start (validated)

Run the self-contained demo from a NeMo Relay checkout:

```bash
RUSTUP_TOOLCHAIN=1.96.1 ./examples/switchyard/run-libsy-e2e.sh
```

This starts a deterministic fake provider and Relay with the libsy backend,
then verifies an easy prompt routes to the weak model and a hard prompt
routes to the strong model, with the classifier and routed calls both flowing
through Relay dispatch.

The plugin configuration (`examples/switchyard/libsy-plugins.toml`):

```toml
[[components]]
kind = "switchyard"
enabled = true

[components.config]
mode = "enforce"
decision_backend = "libsy"
request_materialization = "summary_only"
context_mode = "payload_only"
enabled_inbound_profiles = ["openai_chat"]

[components.config.libsy]
algorithm = "llm_classifier"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
threshold = 0.5

[components.config.default_targets]
openai_chat = "weak"

[components.config.targets.classifier]
model = "classifier-model"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "http://127.0.0.1:4102"

# targets.strong and targets.weak follow the same shape
```

To point at real providers, replace the target `base_url` / `model` values and
supply credentials with per-target `headers` / `header_env`.

## Launching Claude Code: the end-to-end we want

The target experience: from a project directory with the two files below,
`nemo-relay claude` opens the normal interactive Claude Code TUI, and every
LLM call the agent makes is scored by the classifier and routed strong/weak
in-process, with the full decision trail in ATOF events. Everything here is
copy-pasteable; the only unmet dependency is libsy streaming (see the status
table).

Build Relay with the plugin (reference branch:
[feat/libsy-decision-backend](https://github.com/ryan-lempka/NeMo-Relay/tree/feat/libsy-decision-backend)):

```bash
git clone -b feat/libsy-decision-backend https://github.com/ryan-lempka/NeMo-Relay.git
cd NeMo-Relay
RUSTUP_TOOLCHAIN=1.96.1 cargo build -p nemo-relay-cli --features switchyard
```

`.nemo-relay/config.toml`:

```toml
[agents.claude]
command = "claude"
```

`.nemo-relay/plugins.toml` (complete; Haiku scores each request, at or above
0.5 routes to Opus, below routes to Sonnet; Claude Code's own auth headers
pass through to Anthropic):

```toml
version = 1

[[components]]
kind = "switchyard"
enabled = true

[components.config]
mode = "enforce"
decision_backend = "libsy"
request_materialization = "summary_only"
context_mode = "payload_only"
enabled_inbound_profiles = ["anthropic_messages"]

[components.config.libsy]
algorithm = "llm_classifier"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
threshold = 0.5

[components.config.default_targets]
anthropic_messages = "strong"

[components.config.targets.classifier]
model = "claude-haiku-4-5-20251001"
protocol = "anthropic_messages"
endpoint = "/v1/messages"
base_url = "https://api.anthropic.com"

[components.config.targets.strong]
model = "claude-opus-4-8"
protocol = "anthropic_messages"
endpoint = "/v1/messages"
base_url = "https://api.anthropic.com"

[components.config.targets.weak]
model = "claude-sonnet-5"
protocol = "anthropic_messages"
endpoint = "/v1/messages"
base_url = "https://api.anthropic.com"

[[components]]
kind = "observability"
enabled = true

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
mode = "append"
```

Launch and inspect:

```bash
./target/debug/nemo-relay claude          # interactive TUI
grep switchyard.routing nemo-relay-events-*.jsonl
```

## Status against libsy `main` (verified 2026-07-15, fresh public checkout)

| Step | Status |
|---|---|
| Fresh clone builds hermetically; `run-libsy-e2e.sh` passes | works |
| TUI launches; all traffic flows through the gateway and plugin | works |
| Requests carrying `cache_control` reach the router (same-protocol passthrough) | works |
| Buffered requests routed by the libsy classifier | works |
| Streamed requests routed by libsy | **breaks** |

The break, demonstrated with the same prompt sent both ways through the demo
stack:

```text
buffered "This is a hard problem: ..."  -> classifier scored 0.9 -> served by strong-model
streamed "This is a hard problem: ..."  -> no classifier call    -> served by weak-model (trusted fallback)
```

The wrong model answers the streamed request purely because of transport
framing. The cause is libsy's response contract on `main`:
`CallLlmRequest::respond` and `Step::ReturnToAgent` carry only buffered
responses, so no host can pass a live token stream through an algorithm.
Claude Code streams its interactive calls, so those requests emit
`switchyard.routing.error` (`libsy_stream`) and dispatch the trusted
per-protocol fallback (`libsy_streaming_unsupported`) instead of a libsy
decision. Streaming support exists on the unmerged `grclark/simple-proxy`
branch (`LlmResponse` becomes buffered-or-stream); when it lands on `main`,
the Relay plugin can fulfill promises with live streams and the last table
row flips without config changes.

## The embedding contract

The host (Relay) needs exactly three things from libsy:

- `Algorithm` trait object: one shared `Arc<dyn Algorithm>` serves concurrent
  requests with no lock.
- `run_stream(ctx, request)`: returns a stream of `Step`s. The host serves
  `Step::CallLlm` items itself and fulfills each with `respond(...)`.
- The conversation IR (`LlmRequest` / `LlmResponse` from `libsy-protocol`),
  which Relay's plugin already speaks through `switchyard-translation`.

See the [libsy README](../../crates/libsy/README.md)
for the full API walkthrough.
