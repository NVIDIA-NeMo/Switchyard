# Switchyard runner for Node.js

Local TypeScript bindings for choosing a model with Switchyard. The caller sends
the selected model's completion request and owns session state.

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

The build uses Cargo's development profile. It copies the addon and license files
into the package. Rebuild after changing Rust code. Three loopback tests cover
model selection, safe errors, and cancellation. The Pi tests also exercise auto
routing with tool history.

Install the built package in a local Node project:

```bash
npm install /absolute/path/to/Switchyard/bindings/typescript
```

## API

```ts
import { readFileSync } from "node:fs";
import { Runner } from "@switchyard/runner";

const runner = Runner.fromToml(readFileSync("deployment.toml", "utf8"));
const controller = new AbortController();
const decision = await runner.decide("switchyard/auto", "Explain this bug", {
  signal: controller.signal,
});
// decision: { target: "luna", model: "gpt-5.6-luna" }
```

`fromToml` synchronously loads and validates a version-1 runner deployment. Reuse
that runner for later decisions. `decide` takes a configured route **ID** and either
one user text prompt or an array of `RoutingMessage` values. It returns the selected
target name and configured model ID. Concurrent decisions can share a runner.

Use conversation messages for algorithms such as `auto` that inspect tool history.
The exported `RoutingMessage` and `RoutingContent` types cover text, tool calls,
and tool results in Switchyard's format:

```ts
const decision = await runner.decide("switchyard/auto", [
  { role: "user", content: [{ type: "text", text: "Fix the tests" }] },
  { role: "assistant", content: [{
    type: "tool_call", id: "call-1", name: "bash", arguments: { command: "pytest" },
  }] },
  { role: "tool", content: [{
    type: "tool_result", tool_call_id: "call-1", is_error: true,
    content: [{ type: "text", text: "out of memory" }],
  }] },
]);
```

Routing state is fresh for each call. Supply the current branch's conversation on
each decision. The API leaves session identity unset, so recovery signals come
from the transcript rather than cross-request capable holds.

`AbortSignal` cancels the Rust routing future, including pending classifier HTTP
work and retry waits. Cancellation rejects with `name: "AbortError"` and
`code: "ABORT_ERR"`. Other errors have one of these codes:

| Code | Meaning |
| --- | --- |
| `ERR_CONFIG` | Invalid or unsupported configuration, including missing classifier credentials |
| `ERR_UNKNOWN_ROUTE` | The requested route ID is absent |
| `ERR_INVALID_REQUEST` | Routing messages have an invalid shape |
| `ERR_ROUTING` | The algorithm or classifier call failed |
| `ERR_UNSUPPORTED_OUTCOME` | The outcome cannot be represented by a model choice |

Configuration errors preserve the runner's diagnostic message. TOML parse errors
include a line and column when available, with document excerpts kept in Rust.
Provider failures use safe summaries.

## Supported routes

Use `auto`, `passthrough` with its subagent policy unset, or `llm_classifier` in
capability mode. Auto uses local tool signals with an efficient-first default.
Capability mode supports both LLM judges and System One decision judges.
Use `classify_trigger = "every_request"` and `message_hash_fallback = false`
(their defaults). The host persists any chosen model or phase.

The host owns completion prompts, reasoning, and request options. Completion
targets must leave `system_prompt`, `reasoning_effort`, `extra_body`, and
`omit_body_fields` unset. Classifier targets can use their own settings. Clients
must leave `forward_auth` disabled.

Classifier API keys come from the existing `api_key_env` settings. The binding
requires no completion-provider credentials when those clients omit
`api_key_env`. The runner still constructs their clients during loading, so they
need valid HTTP(S) base URLs. For host-owned dispatch, use a credential-free
placeholder such as `http://127.0.0.1:1/v1`.

Use the returned **target name** to look up the host's provider and model. Model
IDs alone can overlap across providers. The binding returns the selected target;
the host owns completion retries and fallback policy.

See [`examples/pi`](../../examples/pi/README.md) for a virtual-model extension.
