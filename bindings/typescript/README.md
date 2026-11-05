# Switchyard libsy for Node.js

TypeScript bindings for routing algorithms. The host handles model and decision
calls. `AlgorithmFactory` builds configured algorithms from deployment TOML;
`CompositeRouter` provides the smaller composite routing API.

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
into the package. Rebuild after changing Rust code. Tests use local callbacks and
HTTP fixtures.

Install the built package in a local Node project:

```bash
npm install /absolute/path/to/Switchyard/bindings/typescript
```

## Configured algorithms

```ts
import { AlgorithmFactory, LlmResponse, Step } from "@switchyard/libsy";

const routes = AlgorithmFactory.fromToml(source).build();
const steps = routes["public/main"].runStream(request, { headers });
```

`source` is deployment TOML and `request` is a normalized Switchyard request.
The factory validates configuration without resolving credentials or creating
provider clients. Reuse configured algorithms to retain algorithm-owned state.
Configurations that require target eligibility checks must use Runner admission;
the factory rejects them.

The iterator yields Python-compatible routing steps:

- `Step.CallModel`: serve `call.request` using `call.models`, then pass a Promise
  of `LlmResponse.Agg` or `LlmResponse.Stream` to `call.respond`.
- `Step.CallDecision`: serve `call.request` using `call.model`, then pass a
  Promise of `DecisionResponse` to `call.respond`.
- `Step.Done`: read the selected models, rewritten requests, optional response,
  and metadata from `step.outcome`.

Model calls and selected models use logical target names from the TOML.
Decision calls use configured provider IDs. The host owns provider settings,
target prompts, transport, and final completion calls when the outcome has no
response. Responses and streams use normalized protocol JSON, not provider wire
formats or SSE bytes. `ContextWindowExceededError` reports a typed context-window
failure to the algorithm.

Calls can overlap. Keep reading steps while serving concurrent calls, and handle
rejections from every `respond` or `fail` Promise. A non-recovering call rejects
its responder; ignoring that error loses the original failure.

Returning the routing iterator stops Rust from waiting on outstanding calls.
JavaScript cannot cancel a supplied Promise. For buffered calls, use a host-owned
controller; here `callProvider` returns the response expected by `call.respond`:

```ts
const controller = new AbortController();
const work = callProvider(call.request, { signal: controller.signal });
try {
  await call.respond(work);
} finally {
  controller.abort();
  await work.catch(() => {});
}
```

The provider client must honor the signal and have a request timeout. This also
cleans up work when the algorithm stops waiting before the response arrives.
For `LlmResponse.Stream`, `respond` finishes when the stream is delivered. Keep
its HTTP request alive until stream completion or cancellation.

## CompositeRouter

Capable-first routing defaults to pinning the efficient tier after switching to
it. Set `stage.efficient_hold_turns: 0` to disable this hold. Efficient-first
defaults to no efficient hold. Persist the returned state to retain holds.

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
| `ERR_CONFIG` | Invalid options or deployment configuration |
| `ERR_INVALID_REQUEST` | Invalid messages, state, or decision response JSON |
| `ERR_ROUTING` | The algorithm or classifier call failed |
| `ABORT_ERR` | Routing was cancelled; the error name is `AbortError` |

A rejected `decide` call follows `classifier.fail_open`. With `false`, routing
fails with the original error as `cause`. With `true`, routing keeps the previous
tier or uses the stage default. An answer without a usable advantage score also
keeps that fallback tier.

An aborted signal closes the Rust run and rejects promptly, including while
`decide` is pending. Late callback results are ignored. Pass the same signal to
your classifier client to cancel its work too.
