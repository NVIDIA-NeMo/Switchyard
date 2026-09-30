# Switchyard NeMo Relay Plugin

`switchyard-nemo-relay-plugin` is a native NeMo Relay dynamic plugin. It loads
a standard Switchyard TOML deployment from a file or Relay's nested plugin
configuration and executes its configured routes through `switchyard-runner`.

The plugin does not define a second routing or target configuration language.
`switchyard-server` and Relay therefore use the same targets, client pooling,
algorithm construction, retry policy, and route validation.

## Install

The plugin requires NeMo Relay `>=0.8.0, <1.0.0`.

Relay 0.8.x and 0.9.0 can lose upstream error status and details when this native
plugin is enabled, including for models outside its configured routes. This is
a known issue tracked by [NeMo Relay PR #1109](https://github.com/NVIDIA/NeMo-Relay/pull/1109).
Until a fix is available, isolate unmanaged traffic in a plugin-disabled gateway.
See the [upstream error compatibility note](../../docs/integrations/nemo_relay.md#upstream-error-compatibility).

### Install a released bundle

Official plugin bundles are distributed through the
[NeMo Relay Plugins repository](https://github.com/NVIDIA/NeMo-Relay-Plugins/releases).
Each bundle includes the native library, a completed `relay-plugin.toml`, the
configuration schema, and license notices. Installing a bundle requires no
Rust build or separate `switchyard-nemo-relay-plugin` crate installation.

The repository currently requires NVIDIA GitHub repository access. Sign in
with an account that has access and complete organization SSO/SAML authorization
where required, including for credentials used by `gh`. An HTTP 404 can mean
that your account or credentials lack access; it does not prove that a bundle
is missing.

1. Select a published `switchyard-plugin-<version>` release for Switchyard
   `0.3.0`. Plugin versions are managed separately and can match the Switchyard
   version. Confirm the source commit in the release notes and `.json` metadata
   rather than relying on the version number alone. A Switchyard release does
   not itself publish a plugin bundle. If no matching published bundle is
   available, use [Build from source](#build-from-source).
2. Download the archive for your platform and its matching `.sha256` and `.json`
   sidecars. For Linux x86_64, the archive is named
   `switchyard-plugin-<version>-linux-x86_64.tar.gz`. Check the metadata's
   `source_commit`, `platform`, and `relay` fields for the source and tested host.
3. Verify the downloaded archive against its `.sha256` file before extracting
   it. On Linux, run `sha256sum -c <archive>.sha256` from the download directory,
   replacing `<archive>` with the archive's filename.
4. Extract the archive into a directory you plan to keep, then follow
   [Register and enable the plugin](#register-and-enable-the-plugin). Use the
   actual path to the extracted `relay-plugin.toml`; the examples below use
   `./plugins/switchyard/relay-plugin.toml`.

### Build from source

Build from source when you need a Switchyard commit that does not have a
released bundle or when you need to customize the plugin. This path requires a
Rust toolchain and Python 3 for the packaging script. Run every command from
the repository root.

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

### Register and enable the plugin

These steps apply to both a downloaded bundle and a bundle built from source.

**1. Register the plugin.**

```bash
nemo-relay plugins validate ./plugins/switchyard/relay-plugin.toml
nemo-relay plugins add --user ./plugins/switchyard/relay-plugin.toml
```

`add` writes a `[[plugins.dynamic]]` entry to your user `plugins.toml`
(`~/.config/nemo-relay/plugins.toml` or `$XDG_CONFIG_HOME/nemo-relay/plugins.toml`).
The plugin is not enabled yet; enabling before the deployment is configured
fails validation because the plugin requires a Switchyard configuration.

**2. Configure the deployment and trust policy** in that `plugins.toml`, as
described in [Configure Relay](#configure-relay).

**3. Enable and validate the plugin**, then restart Relay. The manifest ships
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

The caller and selected target may use different supported API formats:
`openai_chat`, `openai_responses`, or `anthropic_messages`. Switchyard translates
the request into the selected target's configured format and returns buffered
or streaming responses in the caller's original format. With server-owned
credentials, one route targeting an `openai_chat` client can serve all three
caller formats. Separate targets and routes are not required solely for format
translation.

The native Relay plugin rejects routes that use `forward_auth = true` during
configuration validation and activation. This includes routing-model calls and
alternate targets. Relay does not provide caller credentials to the plugin's
provider calls. Secure forwarding support is tracked in
[NeMo Relay #1108](https://github.com/NVIDIA/NeMo-Relay/issues/1108).

For deployment-owned credentials, remove `forward_auth` or set it to `false`
and configure `api_key_env` on each authenticated client. The two options cannot
be enabled together. If each caller must use its own provider credential, use
standalone `switchyard-server`. Standalone forwarding requires the caller and
target to use the same credential family: OpenAI-compatible (Chat Completions
and Responses) or Anthropic (Messages).

Support for provider-specific fields depends on the source and target formats.
Test any fields that your application relies on before deploying a translated
route. See the [integration guide](../../docs/integrations/nemo_relay.md#request-handling)
for request handling, header forwarding, and streaming details.

The plugin emits routing request, model-call, measured-overhead, and decision
marks. Call marks distinguish routing from answer calls; decisions distinguish
selected from served models. Token metrics cover both call roles, while Relay
retains ownership of the outer LLM lifecycle.

## Observability

When Relay is configured with OTLP logs and metrics exporters, the plugin emits
typed telemetry through Relay's native plugin runtime:

- Routing request, decision, and overhead marks are Info logs.
- Per-model call marks are Debug logs with `call_role`, outcome, and latency,
  but no token usage. Streaming marks cover stream creation; later failures are
  reported separately.
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

Every non-metric mark sets `data_schema.name` to the mark name and
`data_schema.version` to `1`. Consumers should tolerate unknown fields and
values. Removing or renaming fields, changing their type, or changing their
meaning requires a new schema version.

| Mark | Data fields |
| --- | --- |
| `switchyard.routing.requested` | `algorithm` |
| `switchyard.routing.llm_call` | `call_index`, `selected_model`, `call_role`, `outcome`, `latency_ms` |
| `switchyard.routing.overhead` | `latency_ms` |
| `switchyard.routing.decision` | `algorithm`, optional `outcome_id`, `selected_model`, nullable `served_model`, nullable `fallback_used`, and optional `evidence` |
| `switchyard.routing.error` | `failure_kind`; route-execution failures also include `category`, `phase`, nullable `upstream_status` and `target`, and may include `outcome_id` and `evidence` |

`served_model` and `fallback_used` are `null` when serving metadata is unavailable.
`outcome_id` is present when the algorithm runner supplies outcome metadata.
`evidence` is an object containing the supported string fields `source`, `verdict`,
`trigger`, and `reason_code`, and numeric fields `score`, `confidence`, and `threshold`.
String values longer than 64 bytes are omitted and should be stable, non-sensitive labels.

## Failure policy

### Provider credential redaction

The plugin replaces configured `api_key_env` credentials with `[REDACTED]` in
buffered responses and each translated stream event before returning them to
Relay. It also redacts returned error strings, routing mark data and metadata,
metric attributes and metadata, and plugin telemetry-emission diagnostics.
The intended upstream still receives the original credential. Upstream response
headers are not returned through the plugin's JSON execution intercepts.

The plugin and standalone server share the credential replacement helpers.
Redaction runs after translation, including on preserved provider fields and
JSON member names. It does not buffer the response stream or change cancellation.
Each event is handled independently: a credential split across events or separate
JSON strings is **not** reconstructed or redacted as a whole. String values and error
text are checked for both raw credentials and their JSON-escaped forms, including
one embedded JSON serialization layer such as serialized tool arguments. Further
repeated escaping and other encodings or transformations are outside this policy.
Preventing reconstruction across stream events requires a separate stateful policy
for each logical text or tool argument field; per-event redaction does not provide
that guarantee.

### Execution failures

`switchyard-llm-client` owns provider retry and route-candidate fallback
behavior. The plugin does not maintain a separate trusted-default target or
rerun routing after an execution failure. Failures outside the shared runner,
including response translation failures, are returned to Relay.
