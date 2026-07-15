# NeMo Relay Recipe

This recipe shows how to run Switchyard **as a library** inside
[NeMo Relay](https://github.com/NVIDIA/NeMo-Relay), so routing decisions are
made in-process by [libsy](https://github.com/NVIDIA-NeMo/Switchyard/tree/main/crates/libsy)
while Relay keeps ownership of the actual LLM calls. No `switchyard serve`
process is involved.

> **Status:** work in progress. Steps marked *validated* have been run end to
> end; everything else is the intended shape and may change.

## Two ways to pair Switchyard with Relay

| Mode | Decision path | Switchyard process | Where it lives |
|---|---|---|---|
| Server | Relay POSTs to Switchyard's HTTP Decision API | Required (`switchyard serve`) | Relay `crates/switchyard` plugin, Switchyard `topic/nemo-relay-integration` |
| Library (this recipe) | Relay calls libsy in-process | None | libsy embedded in Relay's Switchyard plugin |

Relay's Switchyard plugin already anticipates this: its README notes that a
future in-process decision provider may replace the HTTP Decision API call
without changing Relay-owned dispatch and observability.

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
   ├─ libsy Algorithm decides which model to call
   │     Step::CallLlm ──▶ Relay fulfills via next(request) ──▶ provider
   │     Step::Decision ──▶ routing trace / observability
   │     Step::ReturnToAgent ──▶ final response
   ▼
response translated back to the inbound protocol
```

libsy never performs a network call. Each offloaded `CallLlm` promise is
fulfilled by Relay's own dispatch chain (`next(request)`), and the result is
handed back with `CallLlmRequest::respond(Ok/Err)`. This keeps provider
credentials, retries, and telemetry entirely on the Relay side.

## Prerequisites

- Rust 1.96.1 (both repos pin their toolchain)
- A Switchyard checkout on `main` (libsy landed in
  [PR #17](https://github.com/NVIDIA-NeMo/Switchyard/pull/17))
- A NeMo Relay checkout
- Provider credentials for at least two models (a strong and a weak target)

## Quick start

*(to be validated; commands will firm up as the integration lands)*

1. Build Relay with the Switchyard feature:

   ```bash
   cargo build -p nemo-relay-cli --features switchyard
   ```

2. Configure the Switchyard component for library-mode decisions in
   `plugins.toml`, selecting a libsy algorithm (the LLM classifier routes
   strong/weak based on a classifier model's difficulty score):

   ```toml
   [[components]]
   kind = "switchyard"
   enabled = true

   [components.config]
   # decision_backend = "libsy"   # in-process, no decision_api_url needed
   # algorithm = "llm_classifier"
   ```

3. Launch an agent through Relay:

   ```bash
   nemo-relay claude -- "Summarize this repository."
   ```

4. Inspect routing decisions in Relay's ATOF output
   (`.nemo-relay/atof/events.jsonl`) and mark events
   (`switchyard.routing.decision`).

## The embedding contract

The host (Relay) needs exactly three things from libsy:

- `Algorithm` trait object: one shared `Arc<dyn Algorithm>` serves concurrent
  requests with no lock.
- `run_stream(ctx, request)`: returns a stream of `Step`s. The host serves
  `Step::CallLlm` items itself and fulfills each with `respond(...)`.
- The conversation IR (`LlmRequest` / `LlmResponse` from `libsy-protocol`),
  which Relay's plugin already speaks through `switchyard-translation`.

See the [libsy README](https://github.com/NVIDIA-NeMo/Switchyard/tree/main/crates/libsy)
for the full API walkthrough.
