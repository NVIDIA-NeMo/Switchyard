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
| `ABORT_ERR` | Routing was cancelled; the error name is `AbortError` |

A rejected `decide` call follows `classifier.fail_open`. With `false`, routing
fails with the original error as `cause`. With `true`, routing keeps the previous
tier or uses the stage default. An answer without a usable advantage score also
keeps that fallback tier.

An aborted signal closes the Rust run and rejects promptly, including while
`decide` is pending. Late callback results are ignored. Pass the same signal to
your classifier client to cancel its work too.
