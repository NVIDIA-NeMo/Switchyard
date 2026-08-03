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

Relay polls the plugin's pending Rust futures cooperatively on its Tokio
runtime and restores the captured continuation and scope context on every poll.
An active `run_stream` policy therefore does not occupy a blocking worker for
the lifetime of the request. Provider results, stream capacity, and
cancellation wake the policy through the generic native task contract.

The plugin future itself remains executor-neutral. Relay and a native plugin
can link distinct copies of Tokio, so polling from Relay's runtime does not
enter plugin-local Tokio state across the dynamic-library boundary. libsy's
`run_stream` and driver response timeout are therefore poll-driven rather than
calling `tokio::spawn` or `tokio::time::timeout`.

The crate is a `cdylib`-only source/build unit and is not published to
crates.io. Operators install a release bundle containing the compiled shared
library, materialized `relay-plugin.toml`, `config.schema.json`, licensing
files, and checksum.

## Supported routers

This initial plugin release supports exactly two libsy algorithms:

- seeded, weighted `random` routing; and
- capability-based `llm_classifier` routing, where a judge selects the weak or
  strong target before the final provider call.

`stage_router` and the response-judging escalation mode are intentionally not
part of this release. Their configuration is rejected rather than silently
mapped onto one of the supported algorithms. They are being developed as
separate follow-ups so their request-mutation and streaming-response contracts
can be reviewed independently.

During development, `nemo-relay-plugin` is pinned to the Relay native API v2
feature commit. Native API v2 remains unreleased, so every bundle must be
rebuilt against the exact pinned revision; an older v2 bundle must not be used
with a newer draft host table. Replace the Git dependency with the first
published compatible SDK version before releasing a bundle.

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
Each continuation target is an absolute HTTP(S) URL plus headers, and Relay
dispatches it with HTTP `POST` through the captured LLM continuation. Relay
validates the target and owns stream transport and event export.
Switchyard retries or falls back only before the first caller event; after
commitment, a late provider failure is returned without retry. Target URLs,
transport headers, and credentials never enter `LlmRequest.headers`, marks, or
spans; semantic target names remain visible in genuine routing marks. The
plugin contains no Relay provider codecs and does not use private dispatch
headers.

Only supported LLM execution names whose mapped protocol appears in
`default_targets` are managed. Every other buffered or streaming call uses the
SDK's explicit `Passthrough` path. These ordinary untargeted calls remain
inside Relay's managed LLM lifecycle, but their provider events do not cross
the plugin ABI. Targeted provider streams permit at most one pending pull;
plugin output and direct pass-through use Relay's bounded host queue.

Provider failures use HTTP semantics. Relay supplies status, a bounded body,
and safe response headers when it received an HTTP response; Switchyard passes
the status and body to libsy but does not currently use the response headers.
Failures without an HTTP response use a transport, timeout, cancelled,
invalid-request, guardrail, or internal kind. The Switchyard plugin alone owns
retry and fallback policy.
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

[plugins.dynamic.config.algorithm]
kind = "random"
seed = 42

[plugins.dynamic.config.default_targets]
openai_chat = "fast"

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
and headers. Each `default_targets` key both enables that inbound protocol and
names its trusted fallback. `header_env` resolves credentials in the plugin
process without putting them in configuration or libsy metadata.

For `kind = "llm_classifier"`, the classifier thresholds, affinity options, and
`recent_turn_window` use libsy's `TaskClassifierConfig` directly. The plugin
adds only the semantic `classifier_target`, `weak_target`, and `strong_target`
bindings required to resolve Relay continuations. The classifier target must
use `openai_chat` or `openai_responses`: libsy's judge request requires a JSON
schema response format that cannot be encoded losslessly for Anthropic
Messages. An `escalation` table is not accepted by this release.

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

- same-protocol preservation of unknown buffered and raw stream-event fields
  across all three protocols;
- isolated target credentials and headers without source-header inheritance;
- buffered and streaming OpenAI Chat, OpenAI Responses, and Anthropic
  Messages routes;
- cross-protocol request/response translation;
- 12 concurrent independent random-router calls;
- genuine requested and decision marks;
- LLM-classifier weak and strong selections followed by their provider calls;
- buffered and streaming retry reselection plus exactly-once trusted fallback;
- empty-stream fallback and a committed late error with no retry;
- untargeted buffered and streaming pass-through with no Switchyard marks; and
- target credential replacement without recording credential values.

Translation unit tests separately require exact parsed-JSON replay for
same-protocol raw stream events.
