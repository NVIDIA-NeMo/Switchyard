<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard NeMo Relay Dynamic Plugin

This crate builds the external `nvidia.switchyard` native plugin. It embeds
`switchyard-libsy` and drives `Algorithm::run_stream`; NeMo Relay remains
responsible for provider transport and the observability substrate. The plugin
resolves each target's provider headers and credentials and coordinates the
Switchyard retry and trusted-fallback policy.

The plugin requires NeMo Relay native API v2. It uses the safe Rust continuation
facade from `nemo-relay-plugin`; the C ABI remains the binary boundary, but no
raw callbacks, handles, host tables, or unsafe FFI glue appear in this crate. It
does not link the Relay runtime, use `switchyard-llm-client`, or start
`switchyard-server`. `switchyard-translation` is the only request, response, and
stream translation layer.

The crate is a source/build unit and is not published to crates.io. Operators
install a release bundle containing the compiled shared library, materialized
`relay-plugin.toml`, `config.schema.json`, licensing files, and checksum.

During development, `nemo-relay-plugin` is pinned to the Relay ABI v2 feature
commit. Replace that Git dependency with the first published compatible SDK
version before releasing a bundle.

## Runtime contract

For every managed LLM call, the plugin:

1. decodes the caller body with `switchyard-translation`;
2. drives the configured libsy algorithm through `Algorithm::run_stream`;
3. records each real `Decision`;
4. translates every `CallLlm` request to the selected target protocol;
5. asks Relay's safe native API v2 continuation to dispatch the translated
   request;
6. passes the actual response, stream, or typed provider failure back through
   `CallLlmRequest::respond`; and
7. translates `ReturnToAgent` back to the caller protocol.

Switchyard owns routing, translation, target URLs, and target credentials.
Relay validates and transports the selected HTTP target, runs it through the
captured LLM continuation, and owns stream transport and event export.
Switchyard retries or falls back only before the first caller event; after
commitment, a late provider failure is returned without retry. Target data never
enters `LlmRequest.headers`, marks, or spans. The plugin contains no Relay
provider codecs and does not use private dispatch headers.

Calls outside the enabled profiles return the SDK's explicit `Passthrough`
outcome. Relay then forwards the downstream provider stream through its bounded
host queue; unmanaged provider events do not cross the plugin ABI.

Provider failures use HTTP semantics: status, a bounded body, and safe response
headers when Relay received an HTTP response; otherwise a transport, timeout,
cancelled, invalid-request, guardrail, or internal kind. Relay reports those
neutral failure facts; the Switchyard plugin owns retry and fallback policy.
HTTP 408, 425, 429, 500, 502, 503, and 504 plus transport and timeout failures
retry. The plugin does not inspect provider bodies to reclassify HTTP 400
context-window or HTTP 404 model errors.

`switchyard-translation` is used for same-protocol routes as well as
cross-protocol routes. Same-protocol response events replay their preserved
provider JSON, including unknown fields. Cross-protocol routes encode the
normalized fields shared by the source and destination protocols and reject
lossy conversions.

## Configuration

The manifest requires `compat.native_api = "2"`. A Relay project config can
register the bundle and configure a seeded weighted-random router as follows:

```toml
version = 1

[[plugins.dynamic]]
manifest = "/opt/switchyard-relay-plugin/relay-plugin.toml"

[plugins.dynamic.config]
version = 2
priority = 0
max_retries = 3
enabled_inbound_profiles = [
  "openai_chat",
  "openai_responses",
  "anthropic_messages",
]

[plugins.dynamic.config.algorithm]
kind = "random"
seed = 42

[plugins.dynamic.config.default_targets]
openai_chat = "chat-default"
openai_responses = "responses-default"
anthropic_messages = "anthropic-default"

[plugins.dynamic.config.targets.fast]
model = "provider/model"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "https://provider.example.com"
weight = 1

[plugins.dynamic.config.targets.fast.header_env]
authorization = "PROVIDER_AUTHORIZATION"
```

Target map keys such as `fast` are the semantic model names exposed to libsy.
The target binding remains authoritative for the provider model, protocol, URL,
and headers. `header_env` resolves credentials in the plugin process without
putting them in configuration or libsy metadata.

Version-1 service configuration is rejected with a migration error. The plugin
does not provide decision-only or observe-only execution.

## Build and bundle

Build the source crate normally, then materialize an operator bundle from the
platform library:

```bash
cargo build --release -p switchyard-nemo-relay-plugin
python3 crates/switchyard-nemo-relay-plugin/scripts/package_bundle.py \
  --library target/release/libswitchyard_nemo_relay_plugin.so \
  --output dist/switchyard-nemo-relay-plugin-linux-x86_64
```

On macOS the library suffix is `.dylib`. The bundle builder copies the shared
library, manifest, JSON schema, `LICENSE`, and `NOTICE`, materializes the
artifact digest in `relay-plugin.toml`, and writes `SHA256SUMS`.

Operators install the binary bundle rather than this Rust crate:

```bash
nemo-relay plugins validate /opt/switchyard-relay-plugin/relay-plugin.toml
nemo-relay plugins add --project /opt/switchyard-relay-plugin/relay-plugin.toml
nemo-relay plugins enable nvidia.switchyard
nemo-relay plugins inspect nvidia.switchyard
```

## Validation

The process E2E requires a Relay binary that implements native API v2 and a
compiled plugin library:

```bash
python3 crates/switchyard-nemo-relay-plugin/tests/e2e/run_e2e.py \
  --relay-bin /path/to/nemo-relay \
  --plugin-library /path/to/libswitchyard_nemo_relay_plugin.so
```

It launches a local three-protocol fake provider and real Relay process. The
test covers:

- exact same-protocol unknown-field and raw-stream replay;
- isolated target credentials and headers without source-header inheritance;
- buffered and streaming OpenAI Chat, OpenAI Responses, and Anthropic
  Messages routes;
- cross-protocol request/response translation;
- 12 concurrent independent random-router calls;
- genuine requested and decision marks;
- an LLM-classifier call followed by its selected provider call;
- a retryable provider failure with a fresh run; and
- non-retryable failure with exactly-once trusted fallback.
