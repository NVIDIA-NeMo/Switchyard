# Rust API

Choose the crates your integration needs. The links below open generated Rust
API documentation for the core library and its optional supporting components.

## `switchyard-libsy`

Use libsy to construct the model pool, run built-in routing algorithms, implement
a custom algorithm, or handle requested work through the Step stream.

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

## `switchyard-runner`

Use the optional runner to load TOML configuration, resolve named routes, and
execute them inside a serving integration. See
<a href="../../rust/switchyard_runner/struct.Runner.html"><code>Runner</code></a>
and the [configuration reference](toml_schema.md).

<a href="../../rust/switchyard_runner/">Open the generated runner API documentation</a>

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

The runner and HTTP client are optional for direct libsy integration. Your host
can construct the algorithm and execute the requested work itself. The
[demo server](../operations/local_evaluation.md) is a reference application around
the runner; [Python bindings](host_integration.md#python-bindings) expose the
same routing core to Python applications.
