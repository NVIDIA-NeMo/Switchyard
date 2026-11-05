# Switchyard libsy for Node.js

TypeScript bindings for composite routing and buffered JSON translation.
Composite routing uses the caller's decision response and local tool signals to
select a model. The caller handles classifier calls, completion requests, and
branch state.

Capable-first routing defaults to pinning the efficient tier after switching to
it. Set `stage.efficient_hold_turns: 0` to disable this hold. Efficient-first
defaults to no efficient hold. Persist the returned state to retain holds.

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
    decision: { cutoff: 0.6 },
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

`decide` receives protocol JSON with message-array `input` and named `questions`.
Questions and answer values use `{ type, data }` tags. The router supplies packaged
instructions unless `classifier.decision.instructions` is set. A response looks
like this:

```ts
const response: DecisionResponse = {
  answers: { route: {
    value: { type: "choice", data: {
      selected: "capable",
      probabilities: { capable: 0.7, efficient: 0.3 },
    } },
  } },
};
```

The judge selects capable when `probabilities.capable` is above `cutoff`
(default `0.6`). `choices` and `apply_cutoff_to` replace the default choices
and the choice the cutoff applies to.
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
tier or uses the stage default. An answer without a usable probability for the
`apply_cutoff_to` choice (`capable` by default) follows the same policy.

An aborted signal closes the Rust run and rejects promptly, including while
`decide` is pending. Late callback results are ignored. Pass the same signal to
your classifier client to cancel its work too.

## Buffered JSON translation

Translate complete OpenAI Chat Completions, OpenAI Responses, and Anthropic
Messages bodies with the built-in codecs and default policy:

```ts
import { TranslationEngine, WireFormat } from "@switchyard/libsy";

const engine = new TranslationEngine();
const { request, diagnostics } = engine.decodeRequest(WireFormat.OPENAI_CHAT, {
  model: "example",
  messages: [{ role: "user", content: "Hello" }],
});
const { body } = engine.encodeRequest(WireFormat.ANTHROPIC_MESSAGES, request);
```

`decodeRequest` and `decodeResponse` return normalized JSON under `request` or
`response`, plus `diagnostics`. `encodeRequest`, `encodeResponse`,
`translateRequest`, and `translateResponse` return `body` and `diagnostics`.
Inputs must be JSON objects. Translation failures raise `TranslationError` with
`kind` and `details`; invalid normalized input has code `ERR_INVALID_REQUEST`.

Calls retain no request context. To restore request-specific tool names, pass
decoded `request.extensions` as the third argument to `encodeResponse` or the
fourth argument to `translateResponse`. Decoded values also retain provider data
for same-format replay, so editing normalized fields does not automatically
replace that saved body. This API handles buffered JSON; HTTP transport and SSE
streams remain with the caller.
