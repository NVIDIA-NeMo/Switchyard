# Switchyard libsy for Node.js

TypeScript bindings for composite routing. Rust builds a Decision request and uses
the caller's answer with local tool signals to select a model. The caller handles
classifier calls, completion requests, and branch state.

Requires Node.js 22.19 or newer and the repository's Rust toolchain. The package
is private. Its native addon is built for the local host. Linux is covered in CI.

## Build

From the repository root:

```bash
npm --prefix bindings/typescript ci
npm --prefix bindings/typescript run build
npm --prefix bindings/typescript run typecheck
npm --prefix bindings/typescript test
```

The build uses Cargo's development profile and copies the addon and license files
into the package. Rebuild after changing Rust code. Tests use local decision
callbacks.

Install the built package in a local Node project:

```bash
npm install /absolute/path/to/Switchyard/bindings/typescript
```

## API

```ts
import { CompositeRouter, type DecisionRequest, type DecisionResponse } from "@switchyard/libsy";

const router = CompositeRouter.create({
  targets: {
    capable: "provider/large-model",
    efficient: "provider/small-model",
    classifier: "provider/decision-model",
  },
  classifier: {
    classify_trigger: "user_turn",
    fail_open: false,
    decision: {
      cutoff: 0.4,
      candidates: { a: "capable", b: "efficient" },
      evidence: { description: "a handles complex work; b handles routine work" },
    },
  },
  stage: { efficient_first_threshold: 0.5, capable_first_threshold: 0.2 },
});

// Supply an adapter for your classifier provider.
declare function decide(request: DecisionRequest): Promise<DecisionResponse>;
const controller = new AbortController();
const selection = await router.route([
  { role: "user", content: [{ type: "text", text: "Fix the parser" }] },
], { decide, signal: controller.signal });
// selection.model is a configured target model ID.
```

`create` synchronously validates the options. Reuse the router for concurrent
requests. `route` takes `RoutingMessage` values with text, tool calls, and tool
results. Pass the current branch's conversation on each call.

`decide` receives protocol JSON with `context` and named `questions`. Questions
and answer values use `{ type, data }` tags. The router supplies packaged
instructions unless `classifier.decision.instructions` is set. A response looks
like this:

```ts
const response: DecisionResponse = {
  answers: { route: {
    value: { type: "choice", data: {
      selected: "advantage",
      probabilities: { advantage: 0.6, no_advantage: 0.4 },
    } },
  } },
};
```

The judge selects capable when `probabilities.advantage` exceeds `cutoff`.
Stage tool signals then select the final model. `user_turn` judges new user
messages; `new_session` judges when carried state has no tier.

Store `selection.state` when present and pass it as `options.state` on the next
request on the same branch. Keep the previous state when the property is absent.
Treat `CompositeState` as opaque. It carries the judge's tier and stage holds.

## Errors and cancellation

| Code | Meaning |
| --- | --- |
| `ERR_CONFIG` | Invalid composite options |
| `ERR_INVALID_REQUEST` | Invalid messages, state, or decision response JSON |
| `ERR_ROUTING` | The algorithm or classifier call failed |
| `ERR_METRICS` | The metrics collector or export failed |
| `ABORT_ERR` | Routing was cancelled; the error name is `AbortError` |

A rejected `decide` call follows `classifier.fail_open`. With `false`, routing
fails with the original error as `cause`. With `true`, routing keeps the previous
tier or uses the stage default. An answer without a usable advantage score also
keeps that fallback tier.

An aborted signal closes the Rust run and rejects promptly, including while
`decide` is pending. Late callback results are ignored. Pass the same signal to
your classifier client to cancel its work too.

## In-process metrics

`Metrics` exposes the Rust collector without a server. Create it before routing
so the collector receives the full run. All instances share one collector. The
first instance installs the Rust meter provider, with a limit of 20,000 label
combinations per instrument. Further combinations enter an overflow series.

```ts
import { Metrics, type MetricContext } from "@switchyard/libsy";

const metrics = new Metrics();
const metricContext: MetricContext = {
  session_id: "session-1", agent_id: "root",
  agent_turn_idx: 0, user_turn_idx: 0,
};
const selection = await router.route(messages, { decide, metricContext });

// Record completion measurements from the host's model client.
metrics.record({
  id: "completion-1", context: metricContext, model: selection.model,
  call_kind: "answer", is_success: true, duration_ms: 1200, ttft_ms: 150,
  usage: { input_tokens: 100, output_tokens: 30, total_tokens: 130 },
});

const session = metrics.stats({ session_id: "session-1", agent_id: "root" });
const turn = metrics.stats(metricContext);
const all = metrics.stats();
const prometheus = metrics.prometheus();
```

`route` labels its routing metrics with the supplied context. Hosts supply their
own nonnegative turn indices. Use `record` for host-owned classifier and completion
usage and timing. Call kinds are `decision`, `answer`, and `request` (for a failed
request before completion). Recording the same ID within a session twice returns
`false` and adds no measurements, so saved observations can be replayed on reload.

`stats` returns metric series with labels and counter, gauge, or histogram values.
`prometheus` returns Prometheus text. Both accept any subset of the four context
fields; omit the filter to get all collected metrics.

## Task-completion forecasts

`taskPredictionRequest` and `taskPrediction` wrap the Rust
`switchyard-token-prediction` crate:

```ts
import { taskPredictionRequest, taskPrediction } from "@switchyard/libsy";

const request = taskPredictionRequest("provider/judge", [
  { source: "user", message: "Fix the parser" },
], [{ uncached_input_tokens: 800, output_tokens: 100, reasoning_tokens: 40 }]);
const forecast = taskPrediction(await decide(request));
// forecast.predicted_tokens is the estimated total task budget, including prior usage.
// forecast.impossible_probability is reported separately.
```

Supply the observed task history and completed-call usage. The forecast counts
uncached input plus output, including reasoning once, across the whole task.
History preparation and transport belong to the caller. The request contains
80 finite budget ranges through 1.5 million tokens and an `impossible` option.
The response must include probabilities for all 81 options. Token means are
conditional on a finite budget and are `null` when all mass is on `impossible`.
