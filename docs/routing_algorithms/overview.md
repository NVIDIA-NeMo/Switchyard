# Routing Overview

The `switchyard-server` binary loads a native TOML deployment. Each route's `id`
becomes a model ID available through OpenAI Chat Completions, Anthropic Messages,
and OpenAI Responses requests.

Use this page to choose a routing strategy, then open its detailed page for
configuration and tuning. For the vocabulary these pages use, see
[Core Concepts](../core_concepts.md).

## Choose a strategy

Start with **Auto**. Choose Task or Execution when you want more control over
how requests move between an efficient model and a capable one.

| Choice | Use it when | Route `type` |
|---|---|---|
| **[Auto](#auto)** | You want Switchyard's recommended preset. | `auto` |
| **[Task](llm_classifier_routing.md)** | You want an LLM to judge which model can handle the task. | `llm_classifier` |
| **[Execution](stage_router_routing.md)** | You want tool results and agent progress to guide each request. | `stage_router` |

Task uses the LLM classifier's `capability` mode. Its `classify_trigger` setting
controls whether the judge runs for each request, each user turn, or once per
session. Execution uses the stage router, which reads tool-result history and
can optionally call an LLM judge. The TOML configuration keys are unchanged.

### Auto

Auto is a preset, not a separate routing algorithm. It currently uses Execution
with `picker = "efficient_first"`, `confidence_threshold = 0.5`, and no LLM
judge. It does not compare strategies at runtime. Use `stage_router` directly
to tune these settings.

> Auto requires a [source build](../getting_started.md#build-from-source) until v0.3.0 is published.

See the [Auto configuration reference](../reference/toml_schema.md#auto) for
the required targets.

## More options

These options remain available when you need a different routing policy.

| Strategy | Use it when | Route `type` |
|---|---|---|
| [Plan/Execute](plan_execute_routing.md) | Use a capable model to inspect and plan, then switch to an efficient model after the first file mutation. | `plan_execute` |
| [Composite](composite_routing.md) | Combine Task and Execution. A classifier sets the stage router's default tier. | `composite` |
| [Escalation](escalation_router_routing.md) | Start on the efficient model and escalate when an LLM judge detects trouble. | `llm_classifier` with `mode = "escalation"` |
| [Custom](llm_classifier_routing.md#custom-multi-target-routing) | Route among two or more models using your own classification schema and rules. | `llm_classifier` with `mode = "custom"` |
| [Advisor Gate](advisor_gate_routing.md) | Keep one executor model and have a stronger advisor review its plans and completion claims. | `advisor` |
| [Sub-Agent-Aware Routing](subagent_routing.md) | Delegated sub-agents should use a separate routing policy from the parent agent. | `passthrough`, `stage_router`, or `composite` with `subagents` |
| [Random Routing](random_routing.md) | You need a fixed traffic split for A/B tests, baselines, or cost experiments. | `random` |
| [Fixed Model](#direct-model-routes) | Send every request to one target without a routing decision. | `passthrough` |

### Experimental

[Prefill Router](../reference/toml_schema.md#prefill_router) uses a
checkpoint-backed classifier. It is experimental in v0.3.0. Switchyard does not
provide or support a checkpoint, exporter, or compatible encoder assets.

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

After installing the Rust server, as described in
[Getting Started](../getting_started.md#install-the-server), export the provider
credential, validate the configuration, and start the binary:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml \
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
