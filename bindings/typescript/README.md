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
model selection, safe errors, and cancellation using the Pi example configuration.

Install the built package in a local Node project:

```bash
npm install /absolute/path/to/Switchyard/bindings/typescript
```

## API

```ts
import { readFileSync } from "node:fs";
import { Runner } from "@switchyard/runner";

const runner = Runner.fromToml(readFileSync("switchyard.toml", "utf8"));
const controller = new AbortController();
const decision = await runner.decide("switchyard/planner", "Explain this bug", {
  signal: controller.signal,
});
// decision: { target: "complex", model: "gpt-5.6-sol" }
```

`fromToml` synchronously loads and validates a version-1 runner deployment. Reuse
that runner for later decisions. `decide` takes a configured route **ID** and one
user text prompt. It returns the selected target name and configured model ID.
Concurrent decisions can share a runner.

`AbortSignal` cancels the Rust routing future, including pending classifier HTTP
work and retry waits. Cancellation rejects with `name: "AbortError"` and
`code: "ABORT_ERR"`. Other errors have one of these codes:

| Code | Meaning |
| --- | --- |
| `ERR_CONFIG` | Invalid or unsupported configuration, including missing classifier credentials |
| `ERR_UNKNOWN_ROUTE` | The requested route ID is absent |
| `ERR_ROUTING` | The algorithm or classifier call failed |
| `ERR_UNSUPPORTED_OUTCOME` | The outcome cannot be represented by a model choice |

Configuration errors preserve the runner's diagnostic message. TOML parse errors
include a line and column when available, with document excerpts kept in Rust.
Provider failures use safe summaries.

## Supported routes

Use `passthrough` with its subagent policy unset, or `llm_classifier` in capability
mode. Capability mode supports both LLM judges and System One decision judges.
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
