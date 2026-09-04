# Switchyard NeMo Relay Plugin

`switchyard-nemo-relay-plugin` is a native NeMo Relay dynamic plugin. It loads
a standard Switchyard TOML deployment from a file or Relay's nested plugin
configuration and executes its configured routes through `switchyard-runner`.

The plugin does not define a second routing or target configuration language.
`switchyard-server` and Relay therefore use the same targets, client pooling,
algorithm construction, retry policy, and route validation.

## Install

The plugin requires NeMo Relay `>=0.8.1,<0.9.0`, a Rust toolchain, and Python 3
for the packaging script. Run every command from the repository root.

**1. Build the shared library.**

```bash
cargo build --release -p switchyard-nemo-relay-plugin
```

The artifact is `target/release/libswitchyard_nemo_relay_plugin.so` on Linux
and `target/release/libswitchyard_nemo_relay_plugin.dylib` on macOS.

**2. Package the bundle.** The script copies the library, the config schema,
and license files into an empty directory and writes a `relay-plugin.toml`
manifest with the library name and its SHA-256 digest filled in. Relay verifies
that digest before it loads the library, so rebuild the bundle after every
rebuild of the library.

Linux:

```bash
python crates/switchyard-nemo-relay-plugin/scripts/package_bundle.py \
  --library target/release/libswitchyard_nemo_relay_plugin.so \
  --output ./plugins/switchyard
```

macOS:

```bash
python crates/switchyard-nemo-relay-plugin/scripts/package_bundle.py \
  --library target/release/libswitchyard_nemo_relay_plugin.dylib \
  --output ./plugins/switchyard
```

Pass `--archive switchyard-plugin.tar.gz` (or `.zip`) to also produce an
archive for distribution.

**3. Register the plugin.**

```bash
nemo-relay plugins validate ./plugins/switchyard/relay-plugin.toml
nemo-relay plugins add --user ./plugins/switchyard/relay-plugin.toml
```

`add` writes a `[[plugins.dynamic]]` entry to your user `plugins.toml`
(`~/.config/nemo-relay/plugins.toml` or `$XDG_CONFIG_HOME/nemo-relay/plugins.toml`).
The plugin is not enabled yet; enabling before the deployment is configured
fails validation because the plugin requires a Switchyard configuration.

**4. Configure the deployment and trust policy** in that `plugins.toml`, as
described in [Configure Relay](#configure-relay).

**5. Enable and validate the plugin**, then restart Relay. The manifest ships
with `enabled = false`; Relay validates a disabled plugin but never loads it.

```bash
nemo-relay plugins enable nvidia.switchyard
nemo-relay plugins validate nvidia.switchyard
```

`validate` evaluates the manifest, the plugin configuration, the artifact
digest, and the host trust policy.

## Configure Relay

Add a `config` table to the `[[plugins.dynamic]]` entry that `plugins add`
wrote, and a policy override for the plugin. Use exactly one Switchyard
deployment source. To share an existing deployment file with
`switchyard-server`, configure its path:

```toml
[[plugins.dynamic]]
manifest = "./plugins/switchyard/relay-plugin.toml"

[plugins.dynamic.config]
priority = 0
switchyard_config_path = "/etc/switchyard/routes.toml"

[plugins.policy.overrides."nvidia.switchyard"]
attestation = "integrity_only"
```

The policy override is required. The generated manifest carries a SHA-256
digest but no signature, and Relay 0.8 refuses to activate an unsigned dynamic
plugin at gateway start unless its host policy says otherwise; `plugins
validate` still passes without the override, so the failure only shows up at
startup as `requires integrity.signature under host policy`. Native plugins
run inside the Relay process without a sandbox, so only install a bundle you
built or obtained from a source you trust. To require a signature instead, sign
the artifact with an Ed25519 key and list it in `trusted_public_keys`; see
Relay's
[discoverable plugins guide](https://github.com/NVIDIA/NeMo-Relay/blob/main/docs/configure-plugins/discoverable-plugins.mdx)
for the policy keys.

`switchyard_config_path` is a Switchyard version-1 TOML deployment, accepted by both
`switchyard-server` and `switchyard-runner`. See the
[TOML schema reference](../../docs/reference/toml_schema.md) for every key and
the [routing overview](../../docs/routing_algorithms/overview.md) for the
algorithms.

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

Each route's target client must use the caller's wire format: `openai_chat`,
`openai_responses`, or `anthropic_messages`. The runner selects the upstream
backend from that format rather than translating a route to a different
provider API. When one upstream model must serve multiple caller formats,
declare a target and route for each corresponding client format.

The plugin emits a routing request mark, routing-model call marks, measured
routing-overhead marks, and a selected-model decision mark. Token usage is
emitted as Switchyard metrics for both routing-model and answer-model calls;
Relay retains ownership of the outer LLM lifecycle.

## Observability

When Relay is configured with OTLP logs and metrics exporters, the plugin emits
typed telemetry through Relay's native plugin runtime:

- Routing request, decision, and overhead marks are Info logs.
- Per-routing-model call marks are Debug logs, including their outcome and
  latency, but not token usage.
- Terminal routing and response-finalization failures are Error logs. Their
  payload contains only the safe Switchyard failure summary; it excludes
  provider response bodies and free-form provider messages.
- Metrics use bounded attributes only: algorithm for
  `switchyard.routing.requests`; outcome for `switchyard.routing.llm_calls`
  and `switchyard.routing.llm_call.duration`;
  and safe failure kind, category, phase, and optional upstream HTTP status for
  `switchyard.routing.failures`. `switchyard.routing.overhead` records total
  routing latency, including routing-model calls; durations use milliseconds.
- `switchyard.routing.llm_tokens` records normalized token usage with
  `call_role` (`routing` or `answer`), configured `target_model`, and
  `token_type` attributes. A provider may omit usage for streaming responses;
  the plugin does not synthesize zero-value measurements.

The plugin does not attach sessions, requests, or provider messages as metric
attributes. `target_model` comes from the configured Switchyard target set,
rather than arbitrary caller input, keeping the metric cardinality bounded by
the deployment.

## Failure policy

`switchyard-llm-client` owns provider retry and route-candidate fallback
behavior. The plugin does not maintain a separate trusted-default target or
rerun routing after an execution failure. Failures outside the shared runner,
including response translation failures, are returned to Relay.
