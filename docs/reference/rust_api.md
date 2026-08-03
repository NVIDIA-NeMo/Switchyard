# Rust API

Switchyard publishes generated API documentation for its embeddable routing
library and provider-neutral protocol types.

## `switchyard-libsy`

Use libsy to construct targets, run built-in routing algorithms, implement a
custom algorithm, or drive model calls through the host-facing step stream.

<a href="../../rust/switchyard_libsy/">Open the generated libsy API documentation</a>

## `switchyard-protocol`

Use protocol types for normalized conversations, response streams, routed LLM
clients, decisions, metadata, and wire-format identifiers.

<a href="../../rust/switchyard_protocol/">Open the generated protocol API documentation</a>

## Crate boundary

Applications embedding libsy normally depend on both crates:

```toml
[dependencies]
switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
```

Import algorithms and orchestration types from `switchyard_libsy`. Import
requests, responses, metadata, decisions, and LLM client traits from
`switchyard_protocol`. Keep both dependencies on compatible versions.

## CI previews

The Documentation workflow publishes a preview for same-repository pull
requests. Open the workflow's **Deploy PR preview** summary to find its preview
URL, then use the Rust API links above. The **docs-site** build artifact contains
the same MkDocs and rustdoc output for one day.
