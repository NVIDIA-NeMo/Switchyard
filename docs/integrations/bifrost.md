# Use Switchyard with Bifrost

The experimental [Bifrost plugin](../../examples/bifrost/README.md) runs
Switchyard inside Bifrost through its [Go plugin
API](https://github.com/maximhq/bifrost/blob/dev/docs/plugins/getting-started.mdx).
Switchyard selects a target and applies request changes before Bifrost chooses
the provider. Bifrost owns credentials, retries, inference, and response streaming.

## Build

Follow the [source build and test steps](../../examples/bifrost/README.md).
This integration requires a Switchyard checkout and pins Bifrost core 1.13.0.
Build on the host's Linux or macOS platform with matching architecture, Go
version, dependency versions, and build flags. Align `go.mod` with your host
before building for another Bifrost version.

## Configure

Add this entry to Bifrost's `plugins` list. Replace the plugin path and target
models with your deployment values. Configure provider credentials in Bifrost.

```json
{
  "name": "nvidia-switchyard",
  "enabled": true,
  "path": "/deployment/Switchyard/target/switchyard-bifrost.so",
  "config": {
    "bifrost_url": "http://127.0.0.1:8080",
    "timeout_seconds": 300,
    "routes": {
      "switchyard": {
        "algorithm": {
          "type": "stage_router",
          "capable_target": "capable",
          "efficient_target": "efficient",
          "picker": "efficient_first",
          "confidence_threshold": 0.5
        },
        "targets": {
          "capable": "openai/capable-model",
          "efficient": "openai/efficient-model"
        },
        "models": {
          "any": ["openai/capable-model", "openai/efficient-model"],
          "capable": ["openai/capable-model"],
          "efficient": ["openai/efficient-model"]
        }
      }
    }
  }
}
```

Each public route uses the Go binding's existing `algorithm` and `targets`
configuration, plus `models` for the binding's runtime model groups.
`models.any` lists eligible targets in order; Stage also needs `capable` and
`efficient`. These groups contain target model IDs. Random weights follow the
order in `models.any`. Algorithm target names refer to entries in `targets`. Target
values must be Bifrost `provider/model` IDs. Add more entries under `routes`
for other public model names. Register one plugin instance per process.

Send a Chat Completions request to Bifrost with `"model": "switchyard"`:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"switchyard","messages":[{"role":"user","content":"Fix this bug"}]}'
```

The `x-switchyard-selected-model` response header reports the selected target,
including for streaming responses. Request fields retain their caller values
unless routing changes them. Stage handoff notes are applied before inference.

## Algorithms and execution

The plugin executes Noop, Passthrough, Random, Auto, Stage, LLM Classifier,
Composite, Advisor, and VGR algorithms through the shared Go bindings.
Prefill routing is excluded.

Use Switchyard's existing `AlgorithmSpec` settings and the Go binding's runtime
model groups. See the [algorithm guides](../routing_algorithms/overview.md) for
configuration. Passthrough needs only its target in `models.any`; Random weights
follow that group's order. Subagent policies use separate `subagent_models`
groups alongside `models`. VGR active mode retains its approval requirement and
needs a verifier that returns Chat logprobs; unavailable evidence follows VGR policy.

Decision-only outcomes continue through Bifrost's normal inference path. Their
ordered fallback targets are passed to Bifrost. If candidates need different
prepared requests, the plugin executes them in order using the matching request.

Algorithms can call answering models or judges before they finish routing.
The plugin sends these requests to the local Bifrost instance at `bifrost_url`.
The URL must use a loopback IP address, such as `127.0.0.1` or `[::1]`. Proxies
and redirects are disabled. Targets must be direct Bifrost `provider/model` IDs;
they cannot also be public Switchyard route names.

Internal calls use `/v1/chat/completions` and `/v1/decisions`. They retain the
caller's authentication and `x-bf-compat` headers so Bifrost applies its provider access rules,
credentials, limits, and accounting to every call. Allow both public route names
and their internal targets in the caller's Bifrost policy. The per-call HTTP
timeout defaults to 300 seconds; `timeout_seconds` overrides it. Algorithm
consultation deadlines and request cancellation also stop pending calls.
Authentication and admission failures (401, 403, 429) stop routing. Other model
failures follow the algorithm's error policy and ordered model candidates.

Direct and decision-backed classifiers need a decision-capable target. For a chat
model with a Responses API, Bifrost's opt-in
[decision-to-Responses compatibility](https://github.com/maximhq/bifrost/blob/dev/docs/features/compat-plugin.mdx)
can provide that API. The plugin does not replace a decision with an unstructured
chat guess.

## Sessions and subagents

The shared protocol parser reads Switchyard headers and supported harness-native
headers. Use `x-switchyard-session-id` for a conversation and
`x-switchyard-agent-id` for a child. Mark delegated work with
`x-switchyard-is-subagent: true`; `false` retains parent routing. Send
`x-switchyard-session-final: true` to release retained session state.
See [subagent routing](../routing_algorithms/subagent_routing.md) for identity rules.

Session IDs and message-hash retention are scoped by the caller's authentication
headers. Different credentials use separate retained decisions. Anonymous callers
share the anonymous scope. Retained state lives in the plugin process and is
released on cleanup; it does not survive a restart.

## Responses and limits

Completed answers from Advisor, escalation, VGR, and Noop return directly from
the HTTP hook. Buffered JSON and streaming responses use Switchyard's codecs.
When routing has already generated an answer, streaming starts after that answer
and any review have completed. The codec emits a buffered synthetic stream;
it supports text, reasoning, and tool calls. Use buffered JSON when an answer
contains other content types that the synthetic stream cannot preserve.
Decision-only routes retain Bifrost's normal provider streaming.

The `x-switchyard-selected-model` header identifies the routing target. A
Bifrost-managed retry or fallback can serve another model; use the response model
and Bifrost telemetry to identify the model that answered.

The plugin handles HTTP POST requests on `/v1/chat/completions`,
`/openai/v1/chat/completions`, and `/openai/chat/completions`. Other endpoints and
unmanaged model names pass through. The HTTP hooks do not run for Bifrost Go SDK
calls. Register one plugin instance per process. Configuration errors fail
initialization; invalid managed requests fail before inference.
