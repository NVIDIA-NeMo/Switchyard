# Switchyard NeMo Relay Plugin

`switchyard-nemo-relay-plugin` is a native NeMo Relay dynamic plugin. It loads
a standard Switchyard TOML deployment from a file or Relay's nested plugin
configuration and executes its configured routes through `switchyard-runner`.

The plugin does not define a second routing or target configuration language.
`switchyard-server` and Relay therefore use the same targets, client pooling,
algorithm construction, retry policy, and route validation.

## Install

Build the platform bundle with the package script, then configure Relay to load
the generated `relay-plugin.toml` manifest. The plugin requires NeMo Relay
`>=0.8.0,<1.0`.

## Configure Relay

Use exactly one Switchyard deployment source. To share an existing deployment
file with `switchyard-server`, configure its path:

```toml
[[plugins.dynamic]]
manifest = "./plugins/switchyard/relay-plugin.toml"

[plugins.dynamic.config]
priority = 0
switchyard_config_path = "/etc/switchyard/routes.toml"
```

`switchyard_config_path` is a Switchyard version-1 TOML deployment, accepted by both
`switchyard-server` and `switchyard-runner`. See the
[server configuration guide](../switchyard-server/CONFIGURATION.md) for the
deployment schema and routing algorithms.

To keep the deployment in the Relay configuration, nest the same version-1
Switchyard configuration under `switchyard_config`:

```toml
[[plugins.dynamic]]
manifest = "./plugins/switchyard/relay-plugin.toml"

[plugins.dynamic.config]
priority = 0

[plugins.dynamic.config.switchyard_config]
schema_version = 1

[plugins.dynamic.config.switchyard_config.llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[plugins.dynamic.config.switchyard_config.targets.default]
id = "example/model"
llm_client = "primary"

[plugins.dynamic.config.switchyard_config.routes.default]
id = "switchyard/default"
type = "passthrough"
target = "default"
```

## Request handling

For OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages calls,
the plugin decodes the Relay request and checks the requested model against the
deployment's route IDs.

- A configured route is executed by `switchyard-runner`.
- An unknown model calls Relay's continuation unchanged.
- The returned provider response is encoded back into the caller's wire format.
- Streaming responses are returned as unpolled translated streams; Relay owns
  cancellation and the outer serving-call lifecycle.

The plugin emits a routing request mark, routing-model usage marks, measured
routing-overhead marks, and a selected-model decision mark. Answer-call usage
continues to belong to Relay's outer LLM lifecycle.

## Observability

When Relay is configured with OTLP logs and metrics exporters, the plugin emits
typed telemetry through Relay's native plugin runtime:

- Routing request, decision, and overhead marks are Info logs.
- Per-routing-model call marks are Debug logs, including their outcome and
  normalized usage when available.
- Terminal routing and response-finalization failures are Error logs. Their
  payload contains only the safe Switchyard failure summary; it excludes
  provider response bodies and free-form provider messages.
- Metrics use bounded attributes only: algorithm for
  `switchyard.routing.requests`; outcome for `switchyard.routing.llm_calls`
  and `switchyard.routing.llm_call.duration`;
  and safe failure kind, category, and phase for
  `switchyard.routing.failures`. The plugin also records
  `switchyard.routing.overhead` as a histogram. Durations use milliseconds.

The plugin deliberately does not attach model IDs, sessions, requests, or
provider messages as metric attributes, preventing unbounded cardinality or
sensitive data from reaching the metrics exporter.

## Failure policy

`switchyard-llm-client` owns provider retry and route-candidate fallback
behavior. The plugin does not maintain a separate trusted-default target or
rerun routing after an execution failure. Failures outside the shared runner,
including response translation failures, are returned to Relay.
