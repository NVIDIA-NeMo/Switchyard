# Switchyard libsy for Node.js

TypeScript bindings for Switchyard routing. Load configured algorithms from TOML,
or construct a Composite router directly. Your gateway handles model and decision
calls, authentication, and branch state.

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
into the package. Rebuild after changing Rust code. Tests use local callbacks.

Install the built package in a local Node project:

```bash
npm install /absolute/path/to/Switchyard/bindings/typescript
```

## Configured algorithms from TOML

`AlgorithmFactory` uses Runner's deployment schema and algorithm builder without
creating provider clients or reading credentials. Save this as `routes.toml`:

```toml
schema_version = 1
[llm_clients.gateway]
format = "openai_chat"
base_url = "https://models.example/v1"
[targets]
efficient = { id = "host/small-model", llm_client = "gateway" }
capable = { id = "host/large-model", llm_client = "gateway" }
[routes.coding]
id = "switchyard/coding"
type = "stage_router"
efficient_target = "efficient"
capable_target = "capable"
picker = "efficient_first"
confidence_threshold = 0.5
```

```ts
import { readFile } from "node:fs/promises";
import { AlgorithmFactory } from "@switchyard/libsy";

const routes = AlgorithmFactory.fromToml(await readFile("routes.toml", "utf8")).build();
const algorithm = routes.get("switchyard/coding")!;
const outcome = await algorithm.route({
  model: "switchyard/coding",
  messages: [{ role: "user", content: [{ type: "text", text: "Fix the parser" }] }],
}, {
  headers: { "x-switchyard-session-id": "task-123" },
});
console.log(outcome.selectedModelIds);
```

Keys use the route's public `id`, not its TOML table name. Reuse handles and pass
`outcome.state` as `options.state` on the next request on the same branch. `headers`
supplies routing metadata, such as session and subagent identity.

Requests and model replies use Switchyard's normalized protocol, not raw OpenAI or
Anthropic bodies. This Stage route needs no callbacks. Algorithms that request model
calls use `options.call({ request, models })`. Try candidates in order, sending
`{ ...request, model: target }`, and return `{ response, servedModel }`. `response` is
normalized buffered response JSON; `servedModel` is the successful configured
candidate ID, which may differ from the provider's `response.model`. Throw if all
candidates fail. `options.decide` handles typed decision requests. A missing required
callback fails with `ERR_CONFIG`.

Rejected callbacks are reported as generic client errors. This binding does not
preserve provider-specific error types.

The result includes the rewritten request, ordered selected model IDs, metadata,
state, and an optional buffered answer. When `response` is null, the host makes the
answer call. Translation, target prompts, body settings, and response streaming
remain host-owned; the factory does not apply those TOML settings.

## Composite router

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
| `ERR_CONFIG` | Invalid configuration or a required callback is missing |
| `ERR_INVALID_REQUEST` | Invalid request, headers, state, or callback response JSON |
| `ERR_ROUTING` | The algorithm or a callback failed |
| `ABORT_ERR` | Routing was cancelled; the error name is `AbortError` |

A rejected `decide` call follows `classifier.fail_open`. With `false`, routing
fails with the original error as `cause`. With `true`, routing keeps the previous
tier or uses the stage default. An answer without a usable advantage score also
keeps that fallback tier.

Both APIs accept an `AbortSignal`. An aborted signal closes the Rust run and
rejects promptly, including while a callback is pending. Late callback results
are ignored. Pass the same signal to your clients to cancel their work too.
