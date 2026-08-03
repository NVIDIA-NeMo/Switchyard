# switchyard-libsy

Provider-neutral routing and multi-model orchestration for LLM applications.
An `Algorithm` chooses one or more semantic `LlmTarget`s; each target's
`RoutedLlmClient` performs model I/O.

## Setup

```toml
[dependencies]
switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
```

## Quick start

This complete `src/main.rs` constructs a uniform random router over two semantic
targets. Add a client to each target before calling `Algorithm::run`, or let the
host fulfill calls through `Algorithm::run_stream`.

```rust
use std::sync::Arc;

use switchyard_libsy::{Algorithm, LlmTarget, LlmTargetSet, Random};

fn main() -> switchyard_libsy::Result<()> {
    let target = |name: &str| LlmTarget {
        semantic_name: name.into(),
        llm_client: None,
    };
    let targets = LlmTargetSet::new(vec![target("fast"), target("strong")]);
    let router: Arc<dyn Algorithm> = Arc::new(Random::new(targets, None, None)?);
    println!("{}", router.name());
    Ok(())
}
```

## Built-in algorithms

| Algorithm | Purpose |
|---|---|
| `Passthrough` | Always call one configured target. |
| `Random` | Select among any number of targets using uniform or weighted routing. |
| `LlmTaskClassifier` | Ask a judge model to choose an efficient or capable target. |
| `StageRouter` | Route coding-agent turns from tool and progress signals, with an optional judge fallback. |

`Noop` is a test helper, not a production routing algorithm.

## How it fits together

`LlmTarget` pairs a semantic routing name with an optional `RoutedLlmClient`.
An `Algorithm` selects targets and records decisions. Use `Algorithm::run` with
target-owned clients, or `Algorithm::run_stream` when the host owns model
transport. Request, response, usage, and streaming types come from
`switchyard-protocol`.

## Examples

- [Client-backed execution](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/research_agent.rs)
- [Host-owned model calls](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/research_agent_core.rs)
- [Streaming responses](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/streaming_agent.rs)
- [Custom algorithm](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/ensemble.rs)

## License

Licensed under the Apache License, Version 2.0.
