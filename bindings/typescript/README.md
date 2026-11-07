# Switchyard libsy for Node.js

TypeScript bindings for routing algorithms and buffered JSON translation.
The host handles model and decision calls. `AlgorithmFactory` builds configured
algorithms from deployment TOML; `CompositeRouter` provides the smaller composite
routing API.

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
`build()` rejects configurations that require target eligibility checks. Use
`buildConfigured()` to return `ConfiguredRoute` handles with those checks while
keeping model calls in the host:

```ts
const routes = AlgorithmFactory.fromToml(source).buildConfigured();
const coding = routes["switchyard/coding"];
const targetModels = coding.targetModels();
const steps = coding.runStream(request, {
  requiredLabels: ["private", "customer_data"],
  taskId: trustedTaskId,
});
```

The application owns the labels and task identity. Omitted `requiredLabels` means
no assessment was supplied; `[]` explicitly supplies no new requirements.
`runStream()` throws immediately for malformed options or invalid label syntax.
Target eligibility is checked on the first iterator read, before any model call is
yielded; failures reject that read. Handle errors from both creating and reading
the iterator. Routes configured with `scope = "task"`
retain requirements on the reused handle; later requests can add labels but
cannot clear them. `taskId` takes precedence over a task ID derived from headers.
Retained requirements are local to that handle and process.
Handles from `build()` accept only headers; supplied labels (including `[]`) or
`taskId` are rejected.

`coding.targetModels()` maps every possible call target to `{ client, id }`,
including judges, subagents and ordered alternatives. Keys match emitted calls:
logical target names for LLM calls, provider model IDs for Decision calls.
This is a lookup, not an eligible pool; serve only emitted calls and outcome
candidates. `targetModels()` is available only on `ConfiguredRoute` handles from
`buildConfigured()`.

For an emitted model `call`, use the host's existing client registry:

```ts
const { client, id } = targetModels[call.models[0]];
const work = clients[client].call({ ...call.request, model: id });
await call.respond(work);
```

The iterator yields Python-compatible routing steps:

- `Step.CallModel`: serve `call.request` using `call.models[0]`, then pass a Promise
  of `LlmResponse.Agg` or `LlmResponse.Stream` to `call.respond`.
- `Step.CallDecision`: serve `call.request` using `call.model`, then pass a
  Promise of `DecisionResponse` to `call.respond`.
- `Step.Done`: read the selected models, rewritten requests, optional response,
  and metadata from `step.outcome`.

For routing-time model calls, use only the first candidate and pass failures to
`call.respond` or `call.fail`. The reply has no field for the actual served logical
target. When `Step.Done` has no response, the host may try `outcome.selected_model_ids`
in order to serve the final answer, applying the outcome's requests.

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

The classifier picks one of two stage profiles for each turn: `efficient_first`
or `capable_first`. Each profile keeps its own holds. The capable-first profile
defaults to pinning the efficient tier after switching to it. Set
`stage.capable_first.efficient_hold_turns: 0` to disable this hold. The
efficient-first profile defaults to no efficient hold. Persist the returned state
to retain holds.

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
  },
  stage: {
    efficient_first: { threshold: 0.5 },
    capable_first: { threshold: 0.2 },
  },
});

// Supply an adapter for your classifier provider.
declare function decide(request: DecisionRequest): Promise<DecisionResponse>;
const controller = new AbortController();
const selection = await router.route([
  { role: "user", content: [{ type: "text", text: "Fix the parser" }] },
], { decide, signal: controller.signal });
// selection.model is a configured target model ID.
```

Each profile accepts `threshold`, `recent_window`, `capable_hold_turns`,
`efficient_hold_turns`, and `capable_system_prompt`. When a profile routes to
the capable model with a `capable_system_prompt`, `selection.systemPrompt` holds
that prompt. Prepend it to the system prompt of the request you send.

`create` synchronously validates the options. Reuse the router for concurrent
requests. `route` takes `RoutingMessage` values with text, tool calls, and tool
results. Pass the current branch's conversation on each call.

`decide` receives protocol JSON with message-array `input` and named `questions`.
Questions and answer values use `{ type, data }` tags. The router uses the
packaged instructions and the `capable` and `efficient` choices. A response looks
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

The judge selects capable-first when `probabilities.capable` is above
`classifier.cutoff`, which defaults to `0.6`. Stage tool signals in the selected
profile then select the final model. `user_turn` judges new user
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
tier or uses the efficient-first profile. An answer without a usable probability for the
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
