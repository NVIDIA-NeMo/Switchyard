# Routing Overview

The `switchyard-server` binary loads a native TOML deployment. Each route's `id`
becomes a model ID available through OpenAI Chat Completions, Anthropic Messages,
and OpenAI Responses requests.

Use this page to choose a routing strategy, then open its detailed page for
configuration and tuning. For the vocabulary these pages use, see
[Core Concepts](../core_concepts.md).

## Choose a strategy

| Strategy | Use it when | Route `type` |
|---|---|---|
| [Random Routing](random_routing.md) | You need a fixed traffic split for A/B tests, baselines, or cost experiments. | `random` |
| [LLM Classifier Routing](llm_classifier_routing.md) | Request content should decide whether a turn needs the weak or strong tier. | `llm_classifier` |
| [Stage-Router Routing](stage_router_routing.md) | Tool-result and agent-progress signals should route most turns without an extra classifier call. | `stage_router` |
| [Escalation-Router Routing](escalation_router_routing.md) | Start every task on the weak tier and escalate to strong when an LLM judge detects trouble. | `llm_classifier` with `escalation` |

## Common route shape

A deployment has three layers. LLM clients describe how to reach a provider,
targets name models on those clients, and routes decide which target serves a
request:

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.strong]
id = "openai/gpt-4o"
llm_client = "openrouter"

[targets.weak]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[routes.fast]
id = "fast"
type = "passthrough"
target = "weak"

[routes.smart]
id = "smart"
type = "random"
targets = ["strong", "weak"]
weights = [3, 7]
```

Clients send a route's `id` (`fast` or `smart`) as the request's model ID. The
table name in `[routes.smart]` is local to the file; only `id` is visible to
clients. A single deployment can serve multiple routes on the same host and port.

Credentials stay outside the file: `api_key_env` names an environment variable
that is read at startup. Omit it for an upstream that needs no credential, such
as a local model server.

The examples use model IDs from the
[OpenRouter model catalog](https://openrouter.ai/api/v1/models). Select IDs
available to your account before deploying; catalog availability can change.

## Direct model routes

A `passthrough` route registers one target under one model ID with no routing
decision:

```toml
[routes.fast]
id = "fast"
type = "passthrough"
target = "weak"
```

Use a routing strategy instead when requests must be split or classified across
targets. There is no catalog auto-discovery, so to expose several upstream
models, add one `passthrough` route per model.

## Self-hosted targets

Any target can point at an OpenAI-compatible model server you operate. For
example, start a local vLLM server:

```bash
vllm serve ./my-rl-qwen --served-model-name my-rl-qwen --port 8000
```

Then declare it as its own LLM client and target. The client needs no
`api_key_env` when the server does not require a credential:

```toml
[llm_clients.local_vllm]
format = "openai_chat"
base_url = "http://localhost:8000/v1"

[targets.local]
id = "my-rl-qwen"
llm_client = "local_vllm"

[routes.local]
id = "local"
type = "passthrough"
target = "local"
```

Switchyard does not start or manage the model server; it only sends requests to
the configured endpoint.

## Run a deployment

After building the Rust server, as described in
[Getting Started](../getting_started.md#build-the-server), export the provider
credential, validate the configuration, and start the binary:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
./target/release/switchyard-server --config routes.toml --dry-run
./target/release/switchyard-server --config routes.toml \
  --host 127.0.0.1 --port 4000
```

Send a request using a route ID:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smart","messages":[{"role":"user","content":"hello"}]}'
```

For the complete TOML schema and every route option, refer to the
[TOML Schema](../reference/toml_schema.md).
