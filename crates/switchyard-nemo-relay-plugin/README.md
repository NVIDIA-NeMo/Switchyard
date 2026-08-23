# Switchyard NeMo Relay Plugin

`switchyard-nemo-relay-plugin` is a native NeMo Relay dynamic plugin. It loads
a standard Switchyard TOML deployment and executes its configured routes in
Relay through `switchyard-runner`.

The plugin does not define a second routing or target configuration language.
`switchyard-server` and Relay therefore use the same targets, client pooling,
algorithm construction, retry policy, and route validation.

## Install

Build the platform bundle with the package script, then configure Relay to load
the generated `relay-plugin.toml` manifest. The plugin requires NeMo Relay
`>=0.8.0,<1.0`.

## Configure Relay

Point the dynamic plugin configuration at an existing Switchyard deployment:

```toml
[[plugins.dynamic]]
plugin_id = "nvidia.switchyard"

[plugins.dynamic.config]
priority = 0
deployment_path = "/etc/switchyard/routes.toml"
```

`deployment_path` is a Switchyard version-1 TOML deployment, accepted by both
`switchyard-server` and `switchyard-runner`. See the
[server configuration guide](../switchyard-server/CONFIGURATION.md) for the
deployment schema and routing algorithms.

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

## Failure policy

`switchyard-llm-client` owns provider retry and route-candidate fallback
behavior. The plugin does not maintain a separate trusted-default target or
rerun routing after an execution failure. Failures outside the shared runner,
including response translation failures, are returned to Relay.
