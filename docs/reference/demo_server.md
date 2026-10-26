# How the Reference Server Uses Libsy

The local server is a small HTTP application built around Switchyard's configured
runner. Use it to try routing quickly or inspect how the components fit together.
Its role is local demos and personal agent sessions; it is not a supported
production deployment or a service to scale.

For a runnable setup, use **[local evaluation](../operations/local_evaluation.md)**.
For embedding the core in your own application, use the **[Rust
quickstart](../getting_started.md)** and **[host integration guide](host_integration.md)**.

## Follow one request through the components

| Stage | Component | Responsibility | Implementation |
|---|---|---|---|
| Startup | Demo server | Read command-line options and load configured routes. | [Server entry point](../../crates/switchyard-server/src/main.rs) |
| Configuration | Runner | Resolve algorithms, model IDs, client configuration, and credentials from TOML. | [Runner](../../crates/switchyard-runner/src/runner.rs) and [configuration](../../crates/switchyard-runner/src/config.rs) |
| Input boundary | HTTP adapter and translation | Decode a supported provider request into protocol data and attach host metadata. | [HTTP adapters](../../crates/switchyard-server/src) and [translation guide](../protocols/llm.md#use-the-translation-crate) |
| Routing | Libsy | Evaluate history, request model work when needed, and return a routing outcome. | [Mechanics of Libsy](../mechanics_of_libsy.md) |
| Model execution | Optional HTTP client | Serve the calls required by the configured route and make a final answer call when needed. | [Client execution](../../crates/libsy-llm-client/src/run.rs) |
| Output boundary | HTTP adapter and translation | Encode the response or stream in the caller's API format. | [Response and stream contracts](protocols/llm_responses.md) |

The **[runner TOML reference](toml_schema.md)** is shared configuration for
configured integrations, including Relay. TOML does not require using this HTTP
server. A host embedding libsy directly can construct its algorithms and model
pool in Rust instead.

## Keep the reusable boundaries

The server's HTTP routes are one host for the Step exchange. Your host can keep
its existing request admission, authentication, model dispatch, tool loop, and
response delivery.

Use the **[protocol guides](../protocols/index.md)** to connect your input and
output data. Reuse **[translation](protocols/translation.md)** if you want the
supplied provider conversions. Use your existing transport, or choose the
**[optional HTTP client](../../crates/libsy-llm-client/README.md)**.

Follow **[Mechanics](../mechanics_of_libsy.md)** for call fulfillment and outcome
handling. That is the shared execution explanation; this page describes only
the choices made by the reference server.

## Recognize host-specific behavior

| Demo-server surface | What an embedding host supplies instead |
|---|---|
| HTTP model and route aliases | Existing model inventory and routing entry points. |
| Session and agent headers | Protocol metadata derived from the host's identities. |
| Local HTTP statistics and routing logs | Host observations and [telemetry collection](opentelemetry.md). |
| CLI, listening address, and shutdown | The application's service lifecycle. |

These details are examples, not requirements of libsy. Choose lifecycle, access
controls, resource limits, and any multi-process state strategy for the gateway
or application you operate. An in-memory decision retained by one algorithm
instance is not automatically shared with other processes.

For the local server's actual options and endpoints, use the
**[CLI reference](../cli_reference.md)** and
**[server README](../../crates/switchyard-server/README.md)**.
