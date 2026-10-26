# Rust API

Switchyard publishes generated API documentation for its embeddable routing
library, provider-neutral protocol types, and format translation.

## `switchyard-libsy`

Use libsy to construct targets, run built-in routing algorithms, implement a
custom algorithm, or drive model calls through the host-facing step stream.

<a href="../../rust/switchyard_libsy/">Open the generated libsy API documentation</a>

## `switchyard-protocol`

Use protocol types for normalized conversations, response streams, routed LLM
clients, decisions, metadata, and wire-format identifiers.

<a href="../../rust/switchyard_protocol/">Open the generated protocol API documentation</a>

## `switchyard-translation`

Use translation to convert requests, responses, and streams between OpenAI Chat
Completions, OpenAI Responses, and Anthropic Messages formats.

<a href="../../rust/switchyard_translation/">Open the generated translation API documentation</a>

## `switchyard-llm-client`

Use the optional HTTP clients for LLM and decision calls, or the supplied
execution helpers when your application does not drive the step stream itself.

<a href="../../rust/switchyard_llm_client/">Open the generated client API documentation</a>

For field meanings and provider API mappings, use the [LLM protocol reference](protocols/llm_requests.md)
or [decision protocol reference](protocols/decision_questions.md).

## Crate boundary

Use the [Rust quickstart](../getting_started.md#install) for compatible dependency
versions. [Mechanics of Libsy](../mechanics_of_libsy.md) explains how these crates
work together during a run.

Import algorithms and orchestration types from `switchyard_libsy`. Import
requests, responses, metadata, decisions, and LLM client traits from
`switchyard_protocol`. Keep both dependencies on compatible versions.

Add `switchyard-translation` when your application directly converts requests,
responses, or streams between provider formats. Use a version compatible with
your other Switchyard dependencies.

## CI previews

The Documentation workflow publishes a preview for same-repository pull
requests. Open the workflow's **Deploy PR preview** summary to find its preview
URL, then use the Rust API links above. The **docs-site** build artifact contains
the same MkDocs and rustdoc output for one day.
