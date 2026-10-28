# OpenShell middleware with Switchyard Runner

Experimental reference integration requiring the [OpenShell buffered-response
prototype](https://github.com/afourniernv/OpenShell/commit/adc5219fb99a19ebbd56d109dca960d2fc632370),
based on 0.1.5. It does not work with released OpenShell. The Rust
service embeds Runner and builds and tests with Cargo.

The administrator marks source hosts in sandbox policy with
`config.action: mark_restricted`. Before OpenShell allows HTTP access to a
selected source, middleware records a restriction for the authenticated sandbox.
Model hosts use `config.action: execute`; those requests carry the saved decision
into Runner. This is worker-owned state, not an OpenShell privacy flag. The agent
cannot set or clear it through request headers.

The agent sends buffered OpenAI Chat requests. Middleware calls `Route::execute`,
which admits the restriction, runs the configured router, and executes approved
targets with their own endpoints, formats and credentials. Switchyard translates
the answer back to OpenAI Chat. The new terminal response tells OpenShell not to
forward the original request. Existing response middleware still checks the answer.

The example covers judge-free Passthrough and Stage. Streaming is rejected before
execution. Provider credentials belong to this trusted service, not the agent.

## Build and test

Requires the repository Rust toolchain and `protoc` on PATH.
From this directory:

```sh
export OPENSHELL_PROTO_DIR=/path/to/patched/OpenShell/proto
cargo build
cargo test
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
```

Use the protocol files from the same patched checkout as the gateway and
supervisor. The released protocol cannot build this prototype. This standalone
example is outside the product workspace and release pipeline.

`tests/execution.rs` starts the Rust worker and calls its gRPC interface against
local OpenAI Chat and Anthropic Messages fixtures. It covers policy actions,
authentication, retained restrictions, private fallback and response translation.
It does not run OpenShell itself or call real models.

The middleware requires `openshell.supervisor-middleware.buffered-response`
during registration. An older gateway rejects it instead of ignoring the new
response and forwarding the original request.

## Configure

Edit `config/routes.toml` with the model IDs and endpoints. Use existing
`api_key_env` client settings for service-owned credentials. `restricted = true`
marks an approved target; `routes.<name>.restricted` configures its restricted pool.

Copy `config/middleware.example.json` and supply the gateway's JWT public key,
issuer, audience and the middleware's TLS certificate/key. Only the public JWT
key belongs in this service. Use an address reachable from the gateway and
sandbox supervisors. Routes, credentials, TLS, JWT verification and the journal
belong to this service. Source selection belongs in sandbox policy.

Create an empty journal **only for a new deployment**, then start the service:

```sh
mkdir -p .cache
(umask 077; set -C; : > .cache/restrictions.jsonl)
cargo run -- config/middleware.example.json
```

Keep that journal on restart. Missing files, invalid JSON and incomplete final
records fail startup; write failures stop admission. Deleting complete records
or emptying the journal is not detected. Do not recreate it to recover a running task.

Register the service in OpenShell gateway TOML:

```toml
[[openshell.supervisor.middleware]]
name = "switchyard"
grpc_endpoint = "https://middleware.example:50051"
tls_ca_cert_path = "/etc/openshell/middleware-ca.pem"
audience = "urn:openshell:extension:middleware:switchyard"
max_payload_bytes = 262144
timeout = "30s"
```

Merge the excerpt in `config/policy.example.yaml` into the sandbox's existing
policy. Existing filesystem and network grants are omitted here:

```yaml
network_middlewares:
  restricted-source:
    middleware: switchyard
    order: 10
    on_error: fail_closed
    config:
      action: mark_restricted
    endpoints:
      include: [records.example]
  model-execution:
    middleware: switchyard
    order: 20
    on_error: fail_closed
    config:
      action: execute
    endpoints:
      include: [models.example]
```

OpenShell passes each entry's existing `config` object to the worker. Each
`config` must contain only `action`, set to `mark_restricted` or `execute`. Missing or
unknown actions and extra config fields are rejected. This handoff needs no new
OpenShell policy API; buffered execution still needs the prototype above.

Selectors match hosts, not paths, schemes or ports. The source action records a
restriction on any admitted HTTP access to its hosts. Keep source and model hosts
disjoint. The buffered-response capability is service-wide, so every attachment
must be the final applicable request stage and use `fail_closed`, including the
source attachment. Overlapping selectors would leave one attachment non-final
and fail request-chain validation. Order values must be unique across the policy, including
entries for disjoint hosts.

Network rules must permit the intended API paths and block direct access to
other model endpoints. Registration alone does not block bypasses.

## Scope and limits

- Source policy, JWT verification and retention belong to this integration.
  Runner owns admission; libsy enforces the selected lane's model IDs.
- The sandbox ID is Runner's task ID here. Restriction is sandbox-wide, not a
  separate policy for every conversation inside it.
- One service instance owns the journal. Replicas, expiry/reset and a production
  state store are outside this example.
- Source recording precedes ALLOW. Already-admitted calls cannot be recalled
  when another request later restricts the sandbox.
- Journal synchronization blocks the request path. This is not a production
  latency or availability claim.
- Middleware outbound model calls happen outside the sandbox. Operators must
  approve and protect its provider connections too.
- Buffered execution must finish within OpenShell's 30-second maximum deadline. Cancellation
  cannot undo an already-dispatched provider request; execution is not retried
  by this integration.
- This is a REST-only example. Streaming, WebSockets, auxiliaries and judge-based
  routes are not covered.

See [OpenShell middleware configuration](https://docs.nvidia.com/openshell/latest/extensibility/supervisor-middleware/configure)
for registration, authentication and policy lifecycle.
